"""Rollout collection for PPO training with the hybrid (gate + Beta)
policy head.

Each stored transition carries:
  - the 3-wide gate mask (legal gate actions)
  - the sampled gate index
  - the raise chip delta (0 when gate != Raise)
  - the `(min_raise, max_raise)` bounds used to map u ↔ chips
  - the gate+Beta log-prob under the sampling policy
  - the critic's value estimate

Only learner-seat trajectories contribute to the batch; pool-mix
opponent seats do not store anything.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from plo5bp._engine import compute_aggression_bonus_batch  # type: ignore[attr-defined]
from plo5bp.actions import ALL_IN, GATE_ACTIONS, GATE_CHECK_CALL, GATE_RAISE
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.env import BombPotEnv
from plo5bp.env_batched import BatchedBombPotEnv
from plo5bp.network import ActorCritic
from plo5bp.selfplay import OpponentPool


@dataclass
class Batch:
    obs: torch.Tensor           # (T, OBS_DIM) f32
    gate_masks: torch.Tensor    # (T, GATE_ACTIONS) bool
    gate_actions: torch.Tensor  # (T,) long — sampled gate index
    raise_chips: torch.Tensor   # (T,) long — chip delta (0 for non-Raise)
    raise_bounds: torch.Tensor  # (T, 2) long — (min_raise, max_raise)
    log_probs: torch.Tensor     # (T,) f32 — sampling log-prob
    values: torch.Tensor        # (T,) f32 — critic at sampling time
    returns: torch.Tensor       # (T,) f32
    advantages: torch.Tensor    # (T,) f32 — normalized
    # Aggression-bonus diagnostics (set by the rollout collector).
    # `aggr_bonus_total_bb` is the sum of pot-fraction bonus added to
    # learner-step rewards; `aggr_steps_total` is the count of learner
    # steps; `aggr_bonus_steps` is the count of learner steps that
    # actually received a non-zero bonus.
    # `aggr_steps_total_by_street` and `aggr_bonus_steps_by_street`
    # bucket those counts by street (0=flop, 1=turn, 2=river) so we
    # can read per-street bonus%. Bomb pots have no preflop action
    # (engine starts on the flop), so 3 buckets cover every learner step.
    # mean_per_step = total / max(1, steps).
    # applies_pct = 100 * bonus_steps / max(1, steps).
    aggr_bonus_total_bb: float = 0.0
    aggr_steps_total: int = 0
    aggr_bonus_steps: int = 0
    aggr_steps_total_by_street: tuple[int, int, int] = (0, 0, 0)
    aggr_bonus_steps_by_street: tuple[int, int, int] = (0, 0, 0)


def _build_frozen_model(
    state_dict: dict,
    hidden_dim: int,
    device: torch.device,
    num_layers: int = 2,
) -> ActorCritic:
    model = ActorCritic(hidden_dim=hidden_dim, num_layers=num_layers)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _aggression_bonus_bb(
    gate: int,
    commit_delta_chips: int,
    bet_to_call_chips: int,
    street_commit_actor_chips: int,
    pot_chips_pre: int,
    c: float,
) -> float:
    """Pot-fraction reward bonus for voluntarily aggressive actions.

    Returns `c * min(1.0, aggressive_chips / pot_chips_pre)` for
    GATE_RAISE (which now also covers stack-bound short shoves), else 0.
    The "aggressive chips" are chips committed BEYOND the actor's
    amount-to-call:

        call_chips = max(0, bet_to_call - street_commit_actor)
        aggressive  = max(0, commit_delta - call_chips)

    so a short shove that can't fully match the facing bet (commit_delta
    ≤ call_chips) registers zero bonus. Bonus is capped at the pot to
    prevent the policy from learning to overbet for unbounded reward.
    """
    if c <= 0.0 or gate != GATE_RAISE:
        return 0.0
    call_chips = max(0, int(bet_to_call_chips) - int(street_commit_actor_chips))
    aggressive = max(0, int(commit_delta_chips) - call_chips)
    if aggressive <= 0 or pot_chips_pre <= 0:
        return 0.0
    ratio = aggressive / float(pot_chips_pre)
    if ratio > 1.0:
        ratio = 1.0
    return float(c) * ratio


def _apply_retroactive_bonus(
    steps: list[tuple],
    costs: list[float],
    pots_bb: list[float],
    streets: list[int],
    payout_chips: int,
    total_pot_chips: int,
    c: float,
) -> tuple[float, list[int]]:
    """Add pot-relative retroactive aggression bonus to qualifying steps
    based on hero's share of the final pot.

    Per qualifying step `t`, bonus = `c * pots_bb[t]` — the pot in bb
    at the moment of the decision. The pot-relative shape makes the
    bonus louder where chip-delta variance is largest (river deep
    pots) and quieter on the flop (where the policy is already
    aggressive). `c` is in bb-of-bonus per bb-of-pot.

    - share > 50% → bonus on GATE_RAISE steps only.
    - share == 50% → bonus on GATE_RAISE + GATE_CHECK_CALL steps that
      committed chips (call, not check). CHECK_CALL trajectory tuples
      always store chips=0; the "did this step commit chips" signal
      comes from `costs[t] < 0` (the per-step cost is
      `-delta * reward_norm` plus any aggression bonus, which is itself
      zero for non-RAISE gates).
    - share < 50% → no bonus.

    Returns `(bonus_total_bb, bumped_by_street)` where
    `bumped_by_street` is a 3-list indexed by street
    (0=flop, 1=turn, 2=river). Bomb pots have no preflop action so
    every qualifying step lands in one of these three buckets;
    `streets[t]` is the engine street index (1=flop, 2=turn,
    3=river), which is mapped to bucket `streets[t] - 1`. Callers
    fold both into rollout-level diagnostics.
    """
    bumped_by_street: list[int] = [0, 0, 0]
    if total_pot_chips <= 0:
        return 0.0, bumped_by_street
    two_pay = 2 * int(payout_chips)
    if two_pay < total_pot_chips:
        return 0.0, bumped_by_street
    include_call = (two_pay == total_pot_chips)
    added = 0.0
    for t, step in enumerate(steps):
        gate_t = int(step[2])
        is_raise = gate_t == GATE_RAISE
        is_call_with_chips = (
            include_call and gate_t == GATE_CHECK_CALL and costs[t] < 0.0
        )
        if not (is_raise or is_call_with_chips):
            continue
        bonus = c * pots_bb[t]
        costs[t] += bonus
        added += bonus
        bucket = int(streets[t]) - 1
        if 0 <= bucket < 3:
            bumped_by_street[bucket] += 1
    return added, bumped_by_street


def _flush_trajectory(
    traj: list[tuple],
    per_step_costs_bb: list[float],
    terminal_won_bb: float,
    gamma: float,
    lam: float,
    *,
    all_obs: list,
    all_gate_masks: list,
    all_gate_actions: list,
    all_raise_chips: list,
    all_raise_bounds: list,
    all_log_probs: list,
    all_values: list,
    all_returns: list,
    all_advantages: list,
) -> None:
    """GAE-λ flush for a single seat's trajectory under the forward-EV
    reward signal.

    Reward at each step `t` in `traj` is `per_step_costs_bb[t]` — the
    chips this seat committed AT that step expressed in bb units, signed
    negative (since chips going in are a cost). At the LAST step the
    seat additionally collects `terminal_won_bb` — the gross winnings
    for the hand (`won = payouts + total_commit`). Sum across all
    steps in the seat's trajectory equals `won − total_commit` (the
    legacy chip-delta payout), but redistributed step-by-step so the
    value head learns chip change FROM THE DECISION POINT FORWARD —
    sunk costs (antes, prior-street commits) excluded.
    """
    n_steps = len(traj)
    if n_steps == 0:
        return
    assert len(per_step_costs_bb) == n_steps, (
        f"per_step_costs_bb length {len(per_step_costs_bb)} != n_steps {n_steps}"
    )
    vals = [t[6] for t in traj]
    advs = [0.0] * n_steps
    last_gae = 0.0
    for t in reversed(range(n_steps)):
        reward_t = per_step_costs_bb[t]
        if t == n_steps - 1:
            reward_t += terminal_won_bb
        next_v = 0.0 if t == n_steps - 1 else vals[t + 1]
        delta = reward_t + gamma * next_v - vals[t]
        last_gae = delta + gamma * lam * last_gae
        advs[t] = last_gae
    rets = [advs[t] + vals[t] for t in range(n_steps)]
    all_obs.extend(t[0] for t in traj)
    all_gate_masks.extend(t[1] for t in traj)
    all_gate_actions.extend(t[2] for t in traj)
    all_raise_chips.extend(t[3] for t in traj)
    all_raise_bounds.extend(t[4] for t in traj)
    all_log_probs.extend(t[5] for t in traj)
    all_values.extend(vals)
    all_returns.extend(rets)
    all_advantages.extend(advs)


def _finalize_batch(
    all_obs: list,
    all_gate_masks: list,
    all_gate_actions: list,
    all_raise_chips: list,
    all_raise_bounds: list,
    all_log_probs: list,
    all_values: list,
    all_returns: list,
    all_advantages: list,
    device: torch.device | str = "cpu",
    aggr_bonus_total_bb: float = 0.0,
    aggr_steps_total: int = 0,
    aggr_bonus_steps: int = 0,
    aggr_steps_total_by_street: tuple[int, int, int] = (0, 0, 0),
    aggr_bonus_steps_by_street: tuple[int, int, int] = (0, 0, 0),
) -> Batch:
    obs_t = torch.from_numpy(np.stack(all_obs, axis=0)).to(device)
    gm_t = torch.from_numpy(np.stack(all_gate_masks, axis=0)).to(device)
    ga_t = torch.tensor(all_gate_actions, dtype=torch.long, device=device)
    rc_t = torch.tensor(all_raise_chips, dtype=torch.long, device=device)
    rb_t = torch.tensor(
        np.stack(all_raise_bounds, axis=0), dtype=torch.long, device=device
    )
    lp_t = torch.tensor(all_log_probs, dtype=torch.float32, device=device)
    v_t = torch.tensor(all_values, dtype=torch.float32, device=device)
    ret_t = torch.tensor(all_returns, dtype=torch.float32, device=device)
    adv_t = torch.tensor(all_advantages, dtype=torch.float32, device=device)

    adv_mean = adv_t.mean()
    adv_std = adv_t.std().clamp(min=1e-8)
    adv_t = (adv_t - adv_mean) / adv_std

    return Batch(
        obs=obs_t,
        gate_masks=gm_t,
        gate_actions=ga_t,
        raise_chips=rc_t,
        raise_bounds=rb_t,
        log_probs=lp_t,
        values=v_t,
        returns=ret_t,
        advantages=adv_t,
        aggr_bonus_total_bb=float(aggr_bonus_total_bb),
        aggr_steps_total=int(aggr_steps_total),
        aggr_bonus_steps=int(aggr_bonus_steps),
        aggr_steps_total_by_street=tuple(int(x) for x in aggr_steps_total_by_street),
        aggr_bonus_steps_by_street=tuple(int(x) for x in aggr_bonus_steps_by_street),
    )


def collect_rollout(
    learner: ActorCritic,
    pool: OpponentPool,
    game_config: GameConfig,
    train_config: TrainingConfig,
    rng: np.random.Generator,
) -> Batch:
    """Serial rollout driver. Each env has a separate `BombPotEnv`; the
    learner batch-forwards over all learner-acting envs per step and
    frozen opponents forward one-by-one.
    """
    n_envs = train_config.num_envs
    n_seats = game_config.num_seats
    reward_norm = 1.0 / float(game_config.bb)
    gamma = train_config.gamma
    lam = train_config.lam
    pool_mix_prob = float(train_config.pool_mix_prob)
    pool_opp_seats = int(train_config.pool_opp_seats)
    pool_opp_seats = max(0, min(pool_opp_seats, n_seats - 1))

    device = next(learner.parameters()).device

    envs = [BombPotEnv(game_config, ev_runout_samples=train_config.ev_runout_samples)
            for _ in range(n_envs)]
    obs_vecs: list[np.ndarray] = []
    infos = []
    learner_seats: list[set[int]] = [set(range(n_seats)) for _ in range(n_envs)]
    opp_models: list[ActorCritic | None] = [None] * n_envs

    def _assign_pool_mix(env_idx: int) -> None:
        if len(pool) == 0 or pool_opp_seats == 0 or rng.random() >= pool_mix_prob:
            learner_seats[env_idx] = set(range(n_seats))
            opp_models[env_idx] = None
            return
        sd = pool.sample()
        assert sd is not None
        opp_models[env_idx] = _build_frozen_model(
            sd, train_config.hidden_dim, device, train_config.num_layers
        )
        opp_set = set(rng.choice(n_seats, size=pool_opp_seats, replace=False).tolist())
        learner_seats[env_idx] = set(range(n_seats)) - opp_set

    for i, env in enumerate(envs):
        _assign_pool_mix(i)
        seed = int(rng.integers(0, 2**63 - 1))
        button = int(rng.integers(0, n_seats))
        o, info = env.reset(seed, button)
        obs_vecs.append(o)
        infos.append(info)

    # Per env, per seat: list of (obs, gate_mask, gate, chips, bounds, log_p, value).
    trajectories: list[list[list[tuple]]] = [
        [[] for _ in range(n_seats)] for _ in range(n_envs)
    ]
    # Parallel to trajectories: per-step cost (in bb units, signed
    # negative for chips committed). Drives the forward-EV reward signal.
    cost_trajs: list[list[list[float]]] = [
        [[] for _ in range(n_seats)] for _ in range(n_envs)
    ]
    # Parallel to trajectories: pre-step pot (bb units). Consumed by
    # the pot-relative retroactive aggression bonus.
    pot_trajs: list[list[list[float]]] = [
        [[] for _ in range(n_seats)] for _ in range(n_envs)
    ]
    # Parallel to trajectories: pre-step street index (0..3). Used by
    # `_apply_retroactive_bonus` to bucket per-street bonus counts.
    street_trajs: list[list[list[int]]] = [
        [[] for _ in range(n_seats)] for _ in range(n_envs)
    ]

    all_obs: list[np.ndarray] = []
    all_gate_masks: list[np.ndarray] = []
    all_gate_actions: list[int] = []
    all_raise_chips: list[int] = []
    all_raise_bounds: list[np.ndarray] = []
    all_log_probs: list[float] = []
    all_values: list[float] = []
    all_returns: list[float] = []
    all_advantages: list[float] = []

    aggression_bonus_c = float(train_config.aggression_bonus_c)
    retroactive_bonus_c = float(train_config.retroactive_bonus_c)
    aggr_bonus_total_bb = 0.0
    aggr_steps_total = 0
    aggr_bonus_steps = 0
    # 3 buckets: 0=flop, 1=turn, 2=river. Bomb pots have no preflop
    # action so we never bucket street index 0.
    aggr_steps_total_by_street: list[int] = [0, 0, 0]
    aggr_bonus_steps_by_street: list[int] = [0, 0, 0]

    while len(all_obs) < train_config.rollout_length:
        learner_idx: list[int] = []
        opp_idx: list[int] = []
        for i in range(n_envs):
            actor = infos[i].actor
            if actor in learner_seats[i]:
                learner_idx.append(i)
            else:
                opp_idx.append(i)

        gates_per_env = np.zeros(n_envs, dtype=np.int64)
        chips_per_env = np.zeros(n_envs, dtype=np.uint64)
        log_probs_per_env = np.zeros(n_envs, dtype=np.float32)
        values_per_env = np.zeros(n_envs, dtype=np.float32)

        if learner_idx:
            batch_obs = np.stack([obs_vecs[i] for i in learner_idx], axis=0)
            batch_gm = np.stack([infos[i].gate_mask for i in learner_idx], axis=0)
            batch_bounds = np.stack(
                [
                    np.array(
                        [infos[i].min_raise_chips, infos[i].max_raise_chips],
                        dtype=np.int64,
                    )
                    for i in learner_idx
                ],
                axis=0,
            )
            o_t = torch.from_numpy(batch_obs).to(device)
            m_t = torch.from_numpy(batch_gm).to(device)
            b_t = torch.from_numpy(batch_bounds).to(device)
            with torch.no_grad():
                gates_t, chips_t, log_probs_t, values_t = learner.act(o_t, m_t, b_t)
            g_np = gates_t.cpu().numpy()
            c_np = chips_t.cpu().numpy()
            lp_np = log_probs_t.cpu().numpy()
            v_np = values_t.cpu().numpy()
            for k, i in enumerate(learner_idx):
                gates_per_env[i] = int(g_np[k])
                chips_per_env[i] = np.uint64(max(0, int(c_np[k])))
                log_probs_per_env[i] = float(lp_np[k])
                values_per_env[i] = float(v_np[k])

        if opp_idx:
            for i in opp_idx:
                opp = opp_models[i]
                assert opp is not None
                o_t = torch.from_numpy(obs_vecs[i]).unsqueeze(0).to(device)
                m_t = torch.from_numpy(infos[i].gate_mask).unsqueeze(0).to(device)
                b_t = torch.tensor(
                    [[infos[i].min_raise_chips, infos[i].max_raise_chips]],
                    dtype=torch.long,
                ).to(device)
                with torch.no_grad():
                    g_t, c_t, _, _ = opp.act(o_t, m_t, b_t)
                gates_per_env[i] = int(g_t.cpu().numpy()[0])
                chips_per_env[i] = np.uint64(max(0, int(c_t.cpu().numpy()[0])))

        for i in range(n_envs):
            env = envs[i]
            info = infos[i]
            actor = info.actor
            gate = int(gates_per_env[i])
            chips = int(chips_per_env[i])
            bounds_i = np.array(
                [info.min_raise_chips, info.max_raise_chips], dtype=np.int64
            )
            if actor in learner_seats[i]:
                trajectories[i][actor].append((
                    obs_vecs[i],
                    info.gate_mask,
                    gate,
                    chips if gate == GATE_RAISE else 0,
                    bounds_i,
                    float(log_probs_per_env[i]),
                    float(values_per_env[i]),
                ))

            # Pre-step pot / call signals for the aggression bonus —
            # captured before the step mutates env state.
            if actor is not None and actor in learner_seats[i]:
                _pre_total = np.asarray(info.total_commit, dtype=np.int64)
                _pot_pre_chips = int(_pre_total.sum()) if _pre_total.size else 0
                _bet_to_call_pre = int(info.raw_obs.get("bet_to_call", 0))
                _street_commits_pre = info.raw_obs.get("street_commit", [])
                if _street_commits_pre is not None and len(_street_commits_pre) > actor:
                    _sc_actor_pre = int(_street_commits_pre[actor])
                else:
                    _sc_actor_pre = 0

            next_obs, rewards, done, next_info = env.step_hybrid(gate, chips)

            # Forward-EV per-step cost: chips the actor put in THIS step,
            # signed negative, in bb units. Only learner-seat acts feed
            # the trajectory.
            #
            # Aggression bonus disabled post-3-gate-collapse: the
            # working theory is that the previous Raise-vs-AllIn gate
            # competition forced policy mass into Fold/CheckCall, and
            # the pot-fraction shaping was compensating for that
            # passivity. With AllIn folded into the Raise gate the
            # network shouldn't need the bonus to find aggression.
            # Re-enable by uncommenting and passing
            # `--aggression-bonus-c <c>` if passivity recurs.
            if actor is not None and actor in learner_seats[i]:
                delta = float(next_info.commit_delta[actor])
                bonus_bb = _aggression_bonus_bb(
                    gate,
                    int(delta),
                    _bet_to_call_pre,
                    _sc_actor_pre,
                    _pot_pre_chips,
                    aggression_bonus_c,
                )
                _street_pre = int(info.raw_obs.get("street", 0))
                cost_trajs[i][actor].append(-delta * reward_norm + bonus_bb)
                pot_trajs[i][actor].append(_pot_pre_chips * reward_norm)
                street_trajs[i][actor].append(_street_pre)
                aggr_bonus_total_bb += bonus_bb
                aggr_steps_total += 1
                _bucket = _street_pre - 1  # 1=flop → bucket 0; 3=river → 2.
                if 0 <= _bucket < 3:
                    aggr_steps_total_by_street[_bucket] += 1
                if bonus_bb > 0.0:
                    aggr_bonus_steps += 1

            if done:
                # Gross winnings = payouts + total_commit (recovers the
                # `won` component the engine subtracted out for chip-delta).
                won = rewards + next_info.total_commit.astype(np.float32)
                total_pot_chips = int(
                    np.asarray(next_info.total_commit, dtype=np.int64).sum()
                )
                for seat in learner_seats[i]:
                    added, bumped_by_street = _apply_retroactive_bonus(
                        trajectories[i][seat],
                        cost_trajs[i][seat],
                        pot_trajs[i][seat],
                        street_trajs[i][seat],
                        int(round(float(won[seat]))),
                        total_pot_chips,
                        retroactive_bonus_c,
                    )
                    aggr_bonus_total_bb += added
                    aggr_bonus_steps += sum(bumped_by_street)
                    for s in range(3):
                        aggr_bonus_steps_by_street[s] += bumped_by_street[s]
                for seat in learner_seats[i]:
                    traj = trajectories[i][seat]
                    cost_list = cost_trajs[i][seat]
                    if not traj:
                        continue
                    terminal_won_bb = float(won[seat]) * reward_norm
                    _flush_trajectory(
                        traj,
                        cost_list,
                        terminal_won_bb,
                        gamma,
                        lam,
                        all_obs=all_obs,
                        all_gate_masks=all_gate_masks,
                        all_gate_actions=all_gate_actions,
                        all_raise_chips=all_raise_chips,
                        all_raise_bounds=all_raise_bounds,
                        all_log_probs=all_log_probs,
                        all_values=all_values,
                        all_returns=all_returns,
                        all_advantages=all_advantages,
                    )
                trajectories[i] = [[] for _ in range(n_seats)]
                cost_trajs[i] = [[] for _ in range(n_seats)]
                pot_trajs[i] = [[] for _ in range(n_seats)]
                street_trajs[i] = [[] for _ in range(n_seats)]
                _assign_pool_mix(i)
                seed = int(rng.integers(0, 2**63 - 1))
                button = int(rng.integers(0, n_seats))
                next_obs, next_info = env.reset(seed, button)

            obs_vecs[i] = next_obs
            infos[i] = next_info

    return _finalize_batch(
        all_obs,
        all_gate_masks,
        all_gate_actions,
        all_raise_chips,
        all_raise_bounds,
        all_log_probs,
        all_values,
        all_returns,
        all_advantages,
        device=device,
        aggr_bonus_total_bb=aggr_bonus_total_bb,
        aggr_steps_total=aggr_steps_total,
        aggr_bonus_steps=aggr_bonus_steps,
        aggr_steps_total_by_street=tuple(aggr_steps_total_by_street),
        aggr_bonus_steps_by_street=tuple(aggr_bonus_steps_by_street),
    )


def collect_rollout_batched(
    learner: ActorCritic,
    pool: OpponentPool,
    game_config: GameConfig,
    train_config: TrainingConfig,
    rng: np.random.Generator,
) -> Batch:
    """Batched rollout using `BatchedBombPotEnv` + snapshot-bucket
    opponent forwards. Drives all envs through `apply_hybrid_batch`."""
    n_envs = train_config.num_envs
    n_seats = game_config.num_seats
    reward_norm = 1.0 / float(game_config.bb)
    gamma = train_config.gamma
    lam = train_config.lam
    pool_mix_prob = float(train_config.pool_mix_prob)
    pool_opp_seats = int(train_config.pool_opp_seats)
    pool_opp_seats = max(0, min(pool_opp_seats, n_seats - 1))
    device = next(learner.parameters()).device

    env = BatchedBombPotEnv(
        n_envs, game_config, ev_runout_samples=train_config.ev_runout_samples
    )

    snapshot_models: dict[int, ActorCritic] = {}

    def _get_snapshot_model(sd_idx: int) -> ActorCritic:
        m = snapshot_models.get(sd_idx)
        if m is not None:
            return m
        sd = copy.deepcopy(pool.snapshots[sd_idx])
        m = _build_frozen_model(
            sd, train_config.hidden_dim, device, train_config.num_layers
        )
        snapshot_models[sd_idx] = m
        return m

    learner_seats: list[set[int]] = [set(range(n_seats)) for _ in range(n_envs)]
    # (n_envs, n_seats) bool — `True` where seat is a learner seat in that
    # env. Mirrors `learner_seats` (the source of truth) and is updated in
    # lockstep by `_assign_pool_mix`. Enables vectorized "is this actor a
    # learner seat?" lookups in the per-step hot loop.
    learner_seats_mask = np.ones((n_envs, n_seats), dtype=bool)
    # (n_envs,) i64 — pool snapshot index per env, or -1 for self-play.
    # Replaces the per-env Python list `env_snapshot_idx` so opp grouping
    # can be vectorized via `np.unique` over the masked column.
    env_snapshot_idx_arr = np.full(n_envs, -1, dtype=np.int64)

    def _assign_pool_mix(env_idx: int) -> None:
        if len(pool) == 0 or pool_opp_seats == 0 or rng.random() >= pool_mix_prob:
            learner_seats[env_idx] = set(range(n_seats))
            learner_seats_mask[env_idx] = True
            env_snapshot_idx_arr[env_idx] = -1
            return
        sd_idx = int(rng.integers(0, len(pool.snapshots)))
        env_snapshot_idx_arr[env_idx] = sd_idx
        _get_snapshot_model(sd_idx)
        opp_seats_arr = rng.choice(n_seats, size=pool_opp_seats, replace=False)
        opp_set = set(opp_seats_arr.tolist())
        learner_seats[env_idx] = set(range(n_seats)) - opp_set
        row = np.ones(n_seats, dtype=bool)
        row[opp_seats_arr] = False
        learner_seats_mask[env_idx] = row

    for i in range(n_envs):
        _assign_pool_mix(i)

    init_seeds = rng.integers(0, 2**63 - 1, size=n_envs, dtype=np.int64).astype(
        np.uint64
    )
    init_buttons = rng.integers(0, n_seats, size=n_envs, dtype=np.int64).astype(
        np.uint8
    )
    env.reset_batch(init_seeds, init_buttons)

    # Array-backed per-(env, seat) trajectory storage. Replaces the old
    # `trajectories[i][seat]` list-of-tuples — every per-step append is
    # now a vectorized fancy-index write. `MAX_STEPS_PER_SEAT` caps the
    # per-(env, seat) action count for one hand. PLO5 hands cap out
    # well below this even with deep stacks; the assertion in the
    # snapshot block flags overflow rather than silently corrupting data.
    MAX_STEPS_PER_SEAT = 32
    traj_lengths = np.zeros((n_envs, n_seats), dtype=np.int32)
    traj_chunk_idx = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.int32)
    traj_slot = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.int32)
    traj_gate = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.int8)
    traj_chips = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.int64)
    traj_min_raise = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.int64)
    traj_max_raise = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.int64)
    traj_log_p = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
    traj_value = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
    # Aggression-bonus parallel arrays, mirrored shape.
    costs_arr = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
    pots_arr = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
    streets_arr = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.int8)

    # Per-step chunked obs / gate-mask storage. Each step appends ONE
    # (k_step, OBS_DIM) f32 chunk and ONE (k_step, GATE_ACTIONS) bool
    # chunk where `k_step` is the count of active learner envs that
    # step. Trajectories store `(chunk_idx, slot)` integer pointers
    # into these chunks; flush time gathers via `chunks[ci][sl]`.
    step_obs_chunks: list[np.ndarray] = []
    step_gm_chunks: list[np.ndarray] = []

    all_obs: list[np.ndarray] = []
    all_gate_masks: list[np.ndarray] = []
    all_gate_actions: list[int] = []
    all_raise_chips: list[int] = []
    all_raise_bounds: list[np.ndarray] = []
    all_log_probs: list[float] = []
    all_values: list[float] = []
    all_returns: list[float] = []
    all_advantages: list[float] = []

    aggression_bonus_c = float(train_config.aggression_bonus_c)
    retroactive_bonus_c = float(train_config.retroactive_bonus_c)
    aggr_bonus_total_bb = 0.0
    aggr_steps_total = 0
    aggr_bonus_steps = 0
    # 3 buckets: 0=flop, 1=turn, 2=river. Bomb pots have no preflop
    # action so we never bucket street index 0.
    aggr_steps_total_by_street: list[int] = [0, 0, 0]
    aggr_bonus_steps_by_street: list[int] = [0, 0, 0]

    def _forward(model: ActorCritic, group: np.ndarray, obs_arr: np.ndarray,
                 gate_mask_arr: np.ndarray, min_raise_arr: np.ndarray,
                 max_raise_arr: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        b_obs = obs_arr[group]
        b_gm = gate_mask_arr[group]
        b_bounds = np.stack(
            [min_raise_arr[group].astype(np.int64), max_raise_arr[group].astype(np.int64)],
            axis=-1,
        )
        o_t = torch.from_numpy(b_obs).to(device)
        m_t = torch.from_numpy(b_gm).to(device)
        b_t = torch.from_numpy(b_bounds).to(device)
        with torch.no_grad():
            g_t, c_t, lp_t, v_t = model.act(o_t, m_t, b_t)
        return (
            g_t.cpu().numpy().astype(np.uint8),
            c_t.cpu().numpy().astype(np.int64),
            lp_t.cpu().numpy(),
            v_t.cpu().numpy(),
        )

    env_idx_range = np.arange(n_envs)

    while len(all_obs) < train_config.rollout_length:
        obs = env._obs
        gate_masks = env._gate_mask
        min_raise = env._min_raise
        max_raise = env._max_raise
        actors = env._actors
        dones = env._dones
        # Pre-step total_commit / bet_to_call / street_commit / street
        # snapshots for the aggression-bonus and forward-EV reward.
        pre_total_commit = env._total_commit.copy()
        pre_bet_to_call = env._bet_to_call.copy()
        pre_street_commit = env._street_commit.copy()
        pre_street = env._street.copy()

        # Vectorized classification: active learner envs vs active
        # opponent envs. `safe_actors` clamps -1 to 0 so the indexing
        # operation can't blow up; the result is masked by `active`.
        active = ~dones
        safe_actors = np.where(actors >= 0, actors, 0).astype(np.intp)
        is_learner_active = active & learner_seats_mask[env_idx_range, safe_actors]
        is_opp_active = active & ~is_learner_active
        learner_idx_np = np.nonzero(is_learner_active)[0]

        gates_per_env = np.zeros(n_envs, dtype=np.uint8)
        chips_per_env = np.zeros(n_envs, dtype=np.uint64)
        log_probs_per_env = np.zeros(n_envs, dtype=np.float32)
        values_per_env = np.zeros(n_envs, dtype=np.float32)

        if learner_idx_np.size:
            g_np, c_np, lp_np, v_np = _forward(
                learner, learner_idx_np, obs, gate_masks, min_raise, max_raise
            )
            gates_per_env[learner_idx_np] = g_np
            chips_per_env[learner_idx_np] = np.maximum(c_np, 0).astype(np.uint64)
            log_probs_per_env[learner_idx_np] = lp_np
            values_per_env[learner_idx_np] = v_np

        # Group active opponent envs by snapshot index. `np.unique` over
        # the masked column replaces the per-env dict-build loop.
        if is_opp_active.any():
            opp_snap_col = np.where(is_opp_active, env_snapshot_idx_arr, -1)
            for sd_idx in np.unique(opp_snap_col[opp_snap_col >= 0]):
                sd_idx_int = int(sd_idx)
                group = np.nonzero(opp_snap_col == sd_idx)[0]
                m = _get_snapshot_model(sd_idx_int)
                g_np, c_np, _, _ = _forward(
                    m, group, obs, gate_masks, min_raise, max_raise
                )
                gates_per_env[group] = g_np
                chips_per_env[group] = np.maximum(c_np, 0).astype(np.uint64)

        # Vectorized trajectory snapshot: one bulk obs/gm copy per step,
        # plus fancy-index writes into the per-(env, seat) arrays. No
        # per-env Python loop.
        if learner_idx_np.size:
            chunk_idx = len(step_obs_chunks)
            step_obs_chunks.append(obs[learner_idx_np].copy())
            step_gm_chunks.append(gate_masks[learner_idx_np].copy())

            learner_actors = safe_actors[learner_idx_np]
            slots = traj_lengths[learner_idx_np, learner_actors]
            if (slots >= MAX_STEPS_PER_SEAT).any():
                raise RuntimeError(
                    f"per-seat trajectory length exceeded "
                    f"MAX_STEPS_PER_SEAT={MAX_STEPS_PER_SEAT}"
                )

            l_gates = gates_per_env[learner_idx_np]
            l_chips = chips_per_env[learner_idx_np]
            traj_chunk_idx[learner_idx_np, learner_actors, slots] = chunk_idx
            traj_slot[learner_idx_np, learner_actors, slots] = np.arange(
                learner_idx_np.size, dtype=np.int32
            )
            traj_gate[learner_idx_np, learner_actors, slots] = l_gates.astype(np.int8)
            traj_chips[learner_idx_np, learner_actors, slots] = np.where(
                l_gates == GATE_RAISE, l_chips.astype(np.int64), 0
            )
            traj_min_raise[learner_idx_np, learner_actors, slots] = (
                min_raise[learner_idx_np].astype(np.int64)
            )
            traj_max_raise[learner_idx_np, learner_actors, slots] = (
                max_raise[learner_idx_np].astype(np.int64)
            )
            traj_log_p[learner_idx_np, learner_actors, slots] = (
                log_probs_per_env[learner_idx_np]
            )
            traj_value[learner_idx_np, learner_actors, slots] = (
                values_per_env[learner_idx_np]
            )

        # Short-shove redirect: rows where the network emitted GATE_RAISE
        # but the engine zeroed `min_raise` (sub-min-raise stack with
        # `legal[ALL_IN]`) get remapped to the Rust dispatcher's AllIn
        # arm (gate=3). Trajectory snapshot above already recorded the
        # network's emitted gate (GATE_RAISE) and chips_i; the remap is
        # purely a Rust-dispatch concern and the chip amount is ignored
        # by the AllIn arm.
        gates_dispatch = gates_per_env
        short_shove = (
            (gates_per_env == GATE_RAISE)
            & (env._min_raise == np.uint64(0))
            & env._legal[:, ALL_IN]
        )
        if short_shove.any():
            gates_dispatch = gates_per_env.copy()
            gates_dispatch[short_shove] = 3
        newly_terminal = np.asarray(
            env._be.apply_hybrid_batch(gates_dispatch, chips_per_env), dtype=bool
        )
        # Refresh now to capture post-step total_commit (and everything
        # else) BEFORE reset_terminal_batch wipes terminal envs.
        env._refresh()
        post_total_commit = env._total_commit

        # Rust-parallel aggression bonus + per-step cost/pot/street
        # bookkeeping. Replaces the per-env Python arithmetic loop.
        agg = compute_aggression_bonus_batch(
            actors,
            dones,
            learner_seats_mask,
            gates_per_env,
            pre_total_commit,
            post_total_commit.astype(np.int64) if post_total_commit.dtype != np.int64 else post_total_commit,
            pre_bet_to_call,
            pre_street_commit,
            pre_street,
            float(aggression_bonus_c),
            float(reward_norm),
        )
        valid_arr = np.asarray(agg["valid"], dtype=bool)
        valid_idx = np.nonzero(valid_arr)[0]
        if valid_idx.size:
            cost_inc_arr = np.asarray(agg["cost_increment"], dtype=np.float32)
            pot_pre_bb_arr = np.asarray(agg["pot_pre_bb"], dtype=np.float32)
            street_pre_arr = np.asarray(agg["street_pre"], dtype=np.int8)
            valid_actors = safe_actors[valid_idx]
            valid_slots = traj_lengths[valid_idx, valid_actors]
            costs_arr[valid_idx, valid_actors, valid_slots] = cost_inc_arr[valid_idx]
            pots_arr[valid_idx, valid_actors, valid_slots] = pot_pre_bb_arr[valid_idx]
            streets_arr[valid_idx, valid_actors, valid_slots] = street_pre_arr[valid_idx]
            traj_lengths[valid_idx, valid_actors] += 1

        aggr_bonus_total_bb += float(agg["total_bonus_bb"])
        aggr_steps_total += int(agg["total_steps"])
        aggr_bonus_steps += int(agg["bonus_steps"])
        sbs = np.asarray(agg["steps_by_street"])
        bbs = np.asarray(agg["bonus_steps_by_street"])
        for s in range(3):
            aggr_steps_total_by_street[s] += int(sbs[s])
            aggr_bonus_steps_by_street[s] += int(bbs[s])

        if newly_terminal.any():
            if env._ev_runout_samples > 0:
                ev_seeds = env._reset_seeds ^ np.uint64(0x9E3779B97F4A7C15)
                payouts_f32 = np.asarray(
                    env._be.payouts_ev_batch(env._ev_runout_samples, ev_seeds),
                    dtype=np.float32,
                )
            else:
                payouts_f32 = np.asarray(env._be.payouts_batch(), dtype=np.float32)
            won_f32 = payouts_f32 + post_total_commit.astype(np.float32)

            reset_mask = newly_terminal
            new_seeds = rng.integers(
                0, 2**63 - 1, size=n_envs, dtype=np.int64
            ).astype(np.uint64)
            new_buttons = rng.integers(
                0, n_seats, size=n_envs, dtype=np.int64
            ).astype(np.uint8)

            for i in np.nonzero(newly_terminal)[0]:
                i_int = int(i)
                total_pot_chips = int(post_total_commit[i_int].sum())
                # Materialize per-(env, seat) trajectory back into the
                # tuple-list form that `_apply_retroactive_bonus` /
                # `_flush_trajectory` accept. Per-step Python iteration
                # here is bounded by hand length (small) × learner seats
                # (≤ n_seats) and only fires on terminal envs, so it's a
                # small fraction of the per-step work.
                for seat in learner_seats[i_int]:
                    L = int(traj_lengths[i_int, seat])
                    if L == 0:
                        continue
                    traj_list: list[tuple] = []
                    for t in range(L):
                        ci = int(traj_chunk_idx[i_int, seat, t])
                        sl = int(traj_slot[i_int, seat, t])
                        bounds_t = np.array(
                            [
                                int(traj_min_raise[i_int, seat, t]),
                                int(traj_max_raise[i_int, seat, t]),
                            ],
                            dtype=np.int64,
                        )
                        traj_list.append(
                            (
                                step_obs_chunks[ci][sl],
                                step_gm_chunks[ci][sl],
                                int(traj_gate[i_int, seat, t]),
                                int(traj_chips[i_int, seat, t]),
                                bounds_t,
                                float(traj_log_p[i_int, seat, t]),
                                float(traj_value[i_int, seat, t]),
                            )
                        )
                    cost_list = costs_arr[i_int, seat, :L].astype(float).tolist()
                    pot_list = pots_arr[i_int, seat, :L].astype(float).tolist()
                    street_list = streets_arr[i_int, seat, :L].astype(int).tolist()

                    added, bumped_by_street = _apply_retroactive_bonus(
                        traj_list,
                        cost_list,
                        pot_list,
                        street_list,
                        int(round(float(won_f32[i_int, seat]))),
                        total_pot_chips,
                        retroactive_bonus_c,
                    )
                    aggr_bonus_total_bb += added
                    aggr_bonus_steps += sum(bumped_by_street)
                    for s in range(3):
                        aggr_bonus_steps_by_street[s] += bumped_by_street[s]

                    terminal_won_bb = float(won_f32[i_int, seat]) * reward_norm
                    _flush_trajectory(
                        traj_list,
                        cost_list,
                        terminal_won_bb,
                        gamma,
                        lam,
                        all_obs=all_obs,
                        all_gate_masks=all_gate_masks,
                        all_gate_actions=all_gate_actions,
                        all_raise_chips=all_raise_chips,
                        all_raise_bounds=all_raise_bounds,
                        all_log_probs=all_log_probs,
                        all_values=all_values,
                        all_returns=all_returns,
                        all_advantages=all_advantages,
                    )

                # Reset trajectory state for this env. Underlying chunk
                # storage is kept; entries owned by this env's flushed
                # trajectories are now referenced through `all_obs`.
                traj_lengths[i_int, :] = 0
                _assign_pool_mix(i_int)

            env._be.reset_terminal_batch(new_seeds, new_buttons, reset_mask)
            env._reset_seeds = np.where(reset_mask, new_seeds, env._reset_seeds)
            # Second refresh to pick up the post-reset state for the next
            # iteration. Only paid on iterations that triggered a reset.
            env._refresh()

    return _finalize_batch(
        all_obs,
        all_gate_masks,
        all_gate_actions,
        all_raise_chips,
        all_raise_bounds,
        all_log_probs,
        all_values,
        all_returns,
        all_advantages,
        device=device,
        aggr_bonus_total_bb=aggr_bonus_total_bb,
        aggr_steps_total=aggr_steps_total,
        aggr_bonus_steps=aggr_bonus_steps,
        aggr_steps_total_by_street=tuple(aggr_steps_total_by_street),
        aggr_bonus_steps_by_street=tuple(aggr_bonus_steps_by_street),
    )


def iter_minibatches(
    batch: Batch, batch_size: int, rng: np.random.Generator
) -> Iterable[Batch]:
    n = batch.obs.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    device = batch.obs.device
    for start in range(0, n, batch_size):
        sel = torch.from_numpy(idx[start : start + batch_size]).to(device)
        yield Batch(
            obs=batch.obs[sel],
            gate_masks=batch.gate_masks[sel],
            gate_actions=batch.gate_actions[sel],
            raise_chips=batch.raise_chips[sel],
            raise_bounds=batch.raise_bounds[sel],
            log_probs=batch.log_probs[sel],
            values=batch.values[sel],
            returns=batch.returns[sel],
            advantages=batch.advantages[sel],
        )
