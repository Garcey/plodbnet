"""Rollout collection for PPO training with the hybrid (gate + anchor
sizing) policy head.

Each stored transition carries:
  - the 3-wide gate mask (legal gate actions)
  - the sampled gate index
  - the raise chip delta (0 when gate != Raise)
  - the (min_raise, max_raise, pot, to_call) sizing context — the
    anchor grid / brackets are a pure function of it, so evaluate()
    recomputes masks instead of storing them
  - the sampled anchor index and refinement u (v2 head; v1 stores -1/u)
  - the gate+sizing log-prob under the sampling policy
  - the value estimate (CentralCritic when provided, else the actor's)
  - the hero-rotated opponent hole cards, compact (5, hole_count) u8 — the
    centralized critic's extra input (255 = empty slot)

Only learner-seat trajectories contribute to the batch; pool-mix
opponent seats do not store anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import torch
from torch.profiler import record_function

from plo5bp._engine import compute_aggression_bonus_batch  # type: ignore[attr-defined]
from plo5bp.actions import ALL_IN, GATE_ACTIONS, GATE_CHECK_CALL, GATE_RAISE
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.env import BombPotEnv
from plo5bp.env_batched import BatchedBombPotEnv
from plo5bp.network import (
    ActorCritic,
    CentralCritic,
    build_actor_from_state_dict,
    opp_holes_multihot,
)
from plo5bp.selfplay import OpponentPool
from plo5bp.sizing import sizing_from_info

# Slack rows per env appended to the obs-pool / output-slab capacity beyond
# rollout_target: learner steps can still be written after `wcursor` last
# crosses the target (the in-flight hands flush to completion). 32 steps/env
# is ~50x observed need; the overflow guards below turn the (near-impossible)
# overshoot into a clean error. Module-level (P7+P8): collect_rollout_batched
# AND collect_rollout_multiconfig's shared-staging allocation must derive the
# per-sub capacity from the SAME formula.

POOL_SLACK_PER_ENV = 32


class _PinnedStepH2D:
    """Long-lived pinned host staging for per-step act() H2D (CUDA only).

    Capacity is ``num_envs`` (max group size). Each call copies the compact
    batch into pinned host, then non_blocking DMA to device.

    Non-overlap safety (single buffer, no races): every ``_forward`` ends
    with blocking device-to-host reads (``.cpu().numpy()`` on the default
    stream). That drains H2D + compute before the next fill of this staging
    buffer. We never overwrite host staging while the GPU still needs it.
    Callers use the returned ``o_t`` only until the next ``_forward``
    (learner then critic same step; critic runs before any opp ``_forward``).

    Distinct from ``PLO5BP_PIN_ROLLOUT`` (multi-GB finalize slabs — default
    OFF). This path is always on for CUDA; pin cost is O(num_envs * obs_dim)
    once per sub-rollout, not tens of GB.
    """

    __slots__ = ("enabled", "device", "cap", "obs_h", "gm_h", "sz_h")

    def __init__(self, capacity: int, obs_dim: int, device: torch.device) -> None:
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.cap = int(capacity)
        self.enabled = self.device.type == "cuda" and self.cap > 0
        if not self.enabled:
            self.obs_h = self.gm_h = self.sz_h = None  # type: ignore[assignment]
            return
        self.obs_h = torch.empty(
            (self.cap, obs_dim), dtype=torch.float32, pin_memory=True
        )
        self.gm_h = torch.empty(
            (self.cap, GATE_ACTIONS), dtype=torch.bool, pin_memory=True
        )
        self.sz_h = torch.empty((self.cap, 4), dtype=torch.int64, pin_memory=True)

    def upload(
        self,
        b_obs: np.ndarray,
        b_gm: np.ndarray,
        b_sizing: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Host arrays (k, ...) -> device. Values match torch.from_numpy.to."""
        k = int(b_obs.shape[0])
        if k == 0:
            return (
                torch.empty(
                    (0, b_obs.shape[1]), dtype=torch.float32, device=self.device
                ),
                torch.empty(
                    (0, GATE_ACTIONS), dtype=torch.bool, device=self.device
                ),
                torch.empty((0, 4), dtype=torch.int64, device=self.device),
            )
        if (not self.enabled) or k > self.cap:
            return (
                torch.from_numpy(np.ascontiguousarray(b_obs)).to(self.device),
                torch.from_numpy(np.ascontiguousarray(b_gm)).to(self.device),
                torch.from_numpy(np.ascontiguousarray(b_sizing)).to(self.device),
            )
        # Fill pinned host via numpy views (one memcpy each; no per-step pin).
        obs_np = np.ascontiguousarray(b_obs, dtype=np.float32)
        gm_np = np.ascontiguousarray(b_gm, dtype=bool)
        sz_np = np.ascontiguousarray(b_sizing, dtype=np.int64)
        self.obs_h.numpy()[:k] = obs_np
        self.gm_h.numpy()[:k] = gm_np
        self.sz_h.numpy()[:k] = sz_np
        # DMA to device (default stream). Safe to refill only after the
        # caller's blocking D2H at the end of _forward.
        o_t = self.obs_h[:k].to(self.device, non_blocking=True)
        m_t = self.gm_h[:k].to(self.device, non_blocking=True)
        b_t = self.sz_h[:k].to(self.device, non_blocking=True)
        return o_t, m_t, b_t




@dataclass
class Batch:
    obs: torch.Tensor           # (T, OBS_DIM) f32
    gate_masks: torch.Tensor    # (T, GATE_ACTIONS) bool
    gate_actions: torch.Tensor  # (T,) long — sampled gate index
    raise_chips: torch.Tensor   # (T,) long — chip delta (0 for non-Raise)
    sizing: torch.Tensor        # (T, 4) long — (min, max, pot, to_call)
    anchor_actions: torch.Tensor  # (T,) long — sampled anchor (-1 for v1)
    refine_u: torch.Tensor      # (T,) f32 — sampled refinement u
    opp_holes: torch.Tensor     # (T, 5, hole_count) u8 — rotated opp holes
    log_probs: torch.Tensor     # (T,) f32 — sampling JOINT log-prob
    values: torch.Tensor        # (T,) f32 — critic at sampling time
    returns: torch.Tensor       # (T,) f32
    advantages: torch.Tensor    # (T,) f32 — normalized
    # Per-head sampling log-probs for per-head KL diagnostics. gate is
    # always meaningful; anchor is the raise-row anchor log-prob (0 for
    # v1). beta_kl is derived as total_kl - gate_kl - anchor_kl.
    old_gate_logp: torch.Tensor    # (T,) f32
    old_anchor_logp: torch.Tensor  # (T,) f32
    # Terminal-row flag: True where this row is the seat's LAST decision of
    # the hand. At those rows `returns` equals the raw realized reward
    # exactly (the trace has no future term), so they carry free ground-truth
    # Q labels — the qT boundary-residual canary (V7_DESIGN.md WS1.3) reads
    # them. Populated by the BATCHED collector only; None on serial paths
    # (VRPO/Q-aux runs are batched-only, and the canary skips when None).
    is_terminal: torch.Tensor | None = None  # (T,) bool
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
    # Per-row ABSOLUTE entropy coefficient (V5_DESIGN.md B5): mix-configs
    # attaches each sub-rollout's tier coef so per-tier entropy control
    # exists under mixing (previously one coef covered the whole mixed
    # update and the per-tier anneal design silently didn't apply).
    # None = single-config batch; ppo uses its scalar coef.
    ent_coef_rows: torch.Tensor | None = None
    # Per-tier F/T/R counters (mix-configs only): tier -> (bonus_steps_by
    # _street, steps_by_street). Restores per-tier aggression telemetry
    # that the concat used to pool away.
    tier_ftr: dict | None = None


def _build_frozen_model(
    state_dict: dict,
    hidden_dim: int,
    device: torch.device,
    num_layers: int = 2,
    model_cls: type = ActorCritic,
) -> ActorCritic:
    # Class, obs width, AND anchor spec are sniffed from the state dict
    # itself so pool snapshots rebuild exactly what was frozen across
    # head versions and variants — an NLH v4 snapshot has a 995-wide
    # torso and a 12-anchor ladder that the old hardcoded defaults
    # shape-failed on. `model_cls` is retained for caller compatibility;
    # the sniffed class always matches it for well-formed snapshots.
    model = build_actor_from_state_dict(
        state_dict, hidden_dim=hidden_dim, num_layers=num_layers
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _hole_cache(env) -> np.ndarray:
    """(num_seats, 5) u8 per-hand hole cache from `env.all_hole_cards()`.
    Variants with fewer hole cards (NLH: 2) are right-padded with 255 —
    the multihot expansion's empty sentinel — so the critic's opponent
    input width is fixed across variants. PLO rows are unchanged."""
    holes = env.all_hole_cards()
    out = np.full((len(holes), 5), 255, dtype=np.uint8)
    for s, h in enumerate(holes):
        out[s, : len(h)] = h
    return out


def _rotate_opp_holes(holes: np.ndarray, actor: int) -> np.ndarray:
    """(num_seats, hole_w) per-hand hole cards → hero-rotated
    (5, hole_w) opponent block (hole_w = 5 for PLO5, 6 for PLO6): slot
    j = seat (actor + 1 + j) % num_seats; 255 padding for slots beyond
    num_seats - 1. Matches the encoder's rotation."""
    n_seats = holes.shape[0]
    out = np.full((5, holes.shape[1]), 255, dtype=np.uint8)
    for j in range(min(5, n_seats - 1)):
        out[j] = holes[(actor + 1 + j) % n_seats]
    return out


def _rotate_opp_holes_batch(
    holes_cache: np.ndarray, env_idx: np.ndarray, actors: np.ndarray
) -> np.ndarray:
    """Vectorized `_rotate_opp_holes`: holes_cache (N, S, hole_w) u8 →
    (B, 5, hole_w) u8 for the given env rows/actors."""
    n_seats = holes_cache.shape[1]
    j5 = np.arange(5)
    seats = (actors[:, None].astype(np.int64) + 1 + j5[None, :]) % n_seats
    out = holes_cache[env_idx[:, None], seats]
    invalid = (j5 + 1) >= n_seats
    if invalid.any():
        out = np.where(invalid[None, :, None], np.uint8(255), out)
    return np.ascontiguousarray(out)


def _critic_values(
    critic: CentralCritic,
    device: torch.device,
    obs: "np.ndarray | torch.Tensor",
    opp_np: np.ndarray,
) -> np.ndarray:
    """Centralized-critic forward for a learner step group.

    `obs` may be an already-uploaded device tensor (P9: the batched driver
    passes the actor forward's `o_t` — identical bytes, so the values are
    unchanged) or a numpy array (the serial driver's path, byte-identical
    to before)."""
    if isinstance(obs, torch.Tensor):
        o_t = obs
    else:
        o_t = torch.from_numpy(obs).to(device)
    h_t = torch.from_numpy(opp_np).to(device)
    with torch.inference_mode():
        v_t = critic(o_t, opp_holes_multihot(h_t))
    return v_t.float().cpu().numpy()


def _critic_q_values(
    critic: CentralCritic,
    device: torch.device,
    obs: "np.ndarray | torch.Tensor",
    opp_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(V, Q) from the centralized critic's dueling head — the VRPO /
    Expected-SARSA advantage path. Q is (B, q_actions) over
    [Fold, CheckCall, Raise@anchor_0..k]; at the zero-init head Q == V.

    `obs` may be an already-uploaded device tensor (P9, see _critic_values).
    V and Q return via ONE coalesced D2H (cat then split — pure copies of
    the same float32 values, one CUDA sync instead of two); the splits are
    numpy views, and every consumer is stride-agnostic."""
    if isinstance(obs, torch.Tensor):
        o_t = obs
    else:
        o_t = torch.from_numpy(obs).to(device)
    h_t = torch.from_numpy(opp_np).to(device)
    with torch.inference_mode():
        v_t, q_t = critic.q_values(o_t, opp_holes_multihot(h_t))
        vq = torch.cat((v_t.float().unsqueeze(1), q_t.float()), dim=1)
    vq_np = vq.cpu().numpy()
    return vq_np[:, 0], vq_np[:, 1:]


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
    all_sizing: list,
    all_anchors: list,
    all_refine_u: list,
    all_opp_holes: list,
    all_log_probs: list,
    all_values: list,
    all_returns: list,
    all_advantages: list,
    all_gate_logp: list,
    all_anchor_logp: list,
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
    all_sizing.extend(t[4] for t in traj)
    all_log_probs.extend(t[5] for t in traj)
    all_values.extend(vals)
    all_returns.extend(rets)
    all_advantages.extend(advs)
    all_anchors.extend(t[7] for t in traj)
    all_refine_u.extend(t[8] for t in traj)
    all_opp_holes.extend(t[9] for t in traj)
    all_gate_logp.extend(t[10] for t in traj)
    all_anchor_logp.extend(t[11] for t in traj)


def _vrpo_advantage_scan(
    costs_t: np.ndarray,
    won_bb: np.ndarray,
    q_taken: np.ndarray,
    vpi: np.ndarray,
    last_t_arr: np.ndarray,
    flush_mask: np.ndarray,
    gamma_f: np.float32,
    lam_f: np.float32,
) -> np.ndarray:
    """VRPO / Q-boosted advantage (Fan & Farina, arXiv:2605.19235, eq 3.2):

        A_t = (Q(s_t,a_t) - V^pi(s_t)) + sum_{k>=0} (lam*gamma)^k d+_{t+k},
        d+_t = r_t + gamma*V^pi(s_{t+1}) - Q(s_t,a_t)

    i.e. the action-preference LEADING TERM plus the lambda-trace of
    Expected-SARSA residuals. Telescoped view: plain GAE whose downstream
    bootstraps use Q at the sampled future actions — the residuals vanish
    pathwise as Q calibrates, leaving Q - V^pi (the true advantage).

    2026-07-12 FIX: the original implementation (V5_DESIGN W2.5, shipped
    2026-07-07) lambda-traced d+ ONLY — the leading term was dropped at the
    spec level and propagated into code and test. Under that form the
    advantage -> 0 as Q calibrates, and a terminal fold's advantage was
    -Q[FOLD] (0 once fold supervision pins the column; a SUBSIDY when the
    column drifts negative) instead of -V^pi. Root cause of the v6
    lock-fold pathology (vSix1 fold-subsidy era, vSix2 ratchet-on-pin).

    Zero-init parity is preserved: at Q == V the leading term is ~0 and
    d+ reduces to the GAE residual, so vrpo on a fresh checkpoint still
    matches GAE up to f32 reduction noise. Pinned — along with the fixed-Q
    pins that discriminate the full formula from the residual-only form —
    in tests/python/test_vrpo_advantage.py.

    Shapes: costs_t/q_taken/vpi are (T, S, L) f32; won_bb (f32),
    last_t_arr (int), flush_mask (bool) are (T, S). Slots outside a seat's
    trajectory, or with flush_mask False, return 0 — same masking as the
    GAE scan.
    """
    T, S, L = q_taken.shape
    last_es = np.zeros((T, S), dtype=np.float32)
    trace = np.zeros((T, S, L), dtype=np.float32)
    for t in range(L - 1, -1, -1):
        is_last = (t == last_t_arr) & flush_mask
        active_tm = (t <= last_t_arr) & flush_mask
        reward_t = costs_t[..., t] + np.where(is_last, won_bb, np.float32(0.0))
        if t + 1 < L:
            next_vpi = np.where(is_last, np.float32(0.0), vpi[..., t + 1])
        else:
            next_vpi = np.zeros((T, S), dtype=np.float32)
        delta = reward_t + gamma_f * next_vpi - q_taken[..., t]
        new_es = delta + gamma_f * lam_f * last_es
        last_es = np.where(active_tm, new_es, last_es)
        trace[..., t] = np.where(active_tm, last_es, np.float32(0.0))
    active = (
        np.arange(L, dtype=np.int64)[None, None, :]
        <= last_t_arr[..., None].astype(np.int64)
    ) & flush_mask[..., None]
    return np.where(active, (q_taken - vpi) + trace, np.float32(0.0))


def _finalize_batch(
    all_obs: list,
    all_gate_masks: list,
    all_gate_actions: list,
    all_raise_chips: list,
    all_sizing: list,
    all_anchors: list,
    all_refine_u: list,
    all_opp_holes: list,
    all_log_probs: list,
    all_values: list,
    all_returns: list,
    all_advantages: list,
    all_gate_logp: list,
    all_anchor_logp: list,
    device: torch.device | str = "cpu",
    aggr_bonus_total_bb: float = 0.0,
    aggr_steps_total: int = 0,
    aggr_bonus_steps: int = 0,
    aggr_steps_total_by_street: tuple[int, int, int] = (0, 0, 0),
    aggr_bonus_steps_by_street: tuple[int, int, int] = (0, 0, 0),
    adv_clip: float = 0.0,
) -> Batch:
    obs_t = torch.from_numpy(np.stack(all_obs, axis=0)).to(device)
    gm_t = torch.from_numpy(np.stack(all_gate_masks, axis=0)).to(device)
    ga_t = torch.tensor(all_gate_actions, dtype=torch.long, device=device)
    rc_t = torch.tensor(all_raise_chips, dtype=torch.long, device=device)
    sz_t = torch.tensor(
        np.stack(all_sizing, axis=0), dtype=torch.long, device=device
    )
    an_t = torch.tensor(all_anchors, dtype=torch.long, device=device)
    ru_t = torch.tensor(all_refine_u, dtype=torch.float32, device=device)
    oh_t = torch.from_numpy(np.stack(all_opp_holes, axis=0)).to(device)
    lp_t = torch.tensor(all_log_probs, dtype=torch.float32, device=device)
    glp_t = torch.tensor(all_gate_logp, dtype=torch.float32, device=device)
    alp_t = torch.tensor(all_anchor_logp, dtype=torch.float32, device=device)
    v_t = torch.tensor(all_values, dtype=torch.float32, device=device)
    ret_t = torch.tensor(all_returns, dtype=torch.float32, device=device)
    adv_t = torch.tensor(all_advantages, dtype=torch.float32, device=device)

    adv_mean = adv_t.mean()
    adv_std = adv_t.std().clamp(min=1e-8)
    adv_t = (adv_t - adv_mean) / adv_std
    if adv_clip > 0.0:
        # Tame fat tails: PPO's clip bounds the ratio, not the advantage
        # weight, so a 30σ outlier sample (deep-stack all-in pots) gets
        # 30x gradient weight. See TrainingConfig.adv_clip.
        adv_t = adv_t.clamp(-adv_clip, adv_clip)

    return Batch(
        obs=obs_t,
        gate_masks=gm_t,
        gate_actions=ga_t,
        raise_chips=rc_t,
        sizing=sz_t,
        anchor_actions=an_t,
        refine_u=ru_t,
        opp_holes=oh_t,
        log_probs=lp_t,
        values=v_t,
        returns=ret_t,
        advantages=adv_t,
        old_gate_logp=glp_t,
        old_anchor_logp=alp_t,
        aggr_bonus_total_bb=float(aggr_bonus_total_bb),
        aggr_steps_total=int(aggr_steps_total),
        aggr_bonus_steps=int(aggr_bonus_steps),
        aggr_steps_total_by_street=tuple(int(x) for x in aggr_steps_total_by_street),
        aggr_bonus_steps_by_street=tuple(int(x) for x in aggr_bonus_steps_by_street),
    )


def _finalize_batch_arr(
    all_obs_arr: np.ndarray,
    all_gm_arr: np.ndarray,
    all_ga_arr: np.ndarray,
    all_rc_arr: np.ndarray,
    all_sz_arr: np.ndarray,
    all_an_arr: np.ndarray,
    all_ru_arr: np.ndarray,
    all_oh_arr: np.ndarray,
    all_lp_arr: np.ndarray,
    all_v_arr: np.ndarray,
    all_ret_arr: np.ndarray,
    all_adv_arr: np.ndarray,
    all_glp_arr: np.ndarray,
    all_alp_arr: np.ndarray,
    wcursor: int,
    device: torch.device | str = "cpu",
    aggr_bonus_total_bb: float = 0.0,
    aggr_steps_total: int = 0,
    aggr_bonus_steps: int = 0,
    aggr_steps_total_by_street: tuple[int, int, int] = (0, 0, 0),
    aggr_bonus_steps_by_street: tuple[int, int, int] = (0, 0, 0),
    adv_clip: float = 0.0,
    all_last_arr: np.ndarray | None = None,
) -> Batch:
    """Slab-based finalize: each `all_*_arr` is preallocated and written
    contiguously. Slice to `[:wcursor]` and copy once to `device` per
    slab. No `np.stack` over millions of small arrays.
    """
    # When the source slabs are pinned (CUDA path) `non_blocking=True`
    # lets the H2D copies queue against the default stream and overlap
    # downstream compute; on CPU device the flag is a no-op.
    with record_function("step11/finalize_h2d"):
        obs_t = torch.from_numpy(all_obs_arr[:wcursor]).to(device, non_blocking=True)
        gm_t = torch.from_numpy(all_gm_arr[:wcursor]).to(device, non_blocking=True)
        ga_t = torch.from_numpy(all_ga_arr[:wcursor]).to(device, non_blocking=True)
        rc_t = torch.from_numpy(all_rc_arr[:wcursor]).to(device, non_blocking=True)
        sz_t = torch.from_numpy(all_sz_arr[:wcursor]).to(device, non_blocking=True)
        an_t = torch.from_numpy(all_an_arr[:wcursor]).to(device, non_blocking=True)
        ru_t = torch.from_numpy(all_ru_arr[:wcursor]).to(device, non_blocking=True)
        oh_t = torch.from_numpy(all_oh_arr[:wcursor]).to(device, non_blocking=True)
        lp_t = torch.from_numpy(all_lp_arr[:wcursor]).to(device, non_blocking=True)
        glp_t = torch.from_numpy(all_glp_arr[:wcursor]).to(device, non_blocking=True)
        alp_t = torch.from_numpy(all_alp_arr[:wcursor]).to(device, non_blocking=True)
        v_t = torch.from_numpy(all_v_arr[:wcursor]).to(device, non_blocking=True)
        ret_t = torch.from_numpy(all_ret_arr[:wcursor]).to(device, non_blocking=True)
        adv_t = torch.from_numpy(all_adv_arr[:wcursor]).to(device, non_blocking=True)
        last_t = (
            torch.from_numpy(all_last_arr[:wcursor]).to(device, non_blocking=True)
            if all_last_arr is not None
            else None
        )

        adv_mean = adv_t.mean()
        adv_std = adv_t.std().clamp(min=1e-8)
        adv_t = (adv_t - adv_mean) / adv_std
        if adv_clip > 0.0:
            # Same fat-tail clamp as _finalize_batch (serial parity).
            adv_t = adv_t.clamp(-adv_clip, adv_clip)

    return Batch(
        obs=obs_t,
        gate_masks=gm_t,
        gate_actions=ga_t,
        raise_chips=rc_t,
        sizing=sz_t,
        anchor_actions=an_t,
        refine_u=ru_t,
        opp_holes=oh_t,
        log_probs=lp_t,
        values=v_t,
        returns=ret_t,
        advantages=adv_t,
        old_gate_logp=glp_t,
        old_anchor_logp=alp_t,
        is_terminal=last_t,
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
    critic: CentralCritic | None = None,
) -> Batch:
    """Serial rollout driver. Each env has a separate `BombPotEnv`; the
    learner batch-forwards over all learner-acting envs per step and
    frozen opponents forward one-by-one.

    With `critic` provided, GAE values come from the centralized critic
    (which sees all hole cards); otherwise the actor's own value head is
    used (v1 behavior / profiling fallback).
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

    if getattr(train_config, "advantage_estimator", "gae") == "vrpo":
        # VRPO / Expected-SARSA advantages are implemented only in the batched
        # collector (the training path). The serial driver serves UI / eval /
        # exploit, which always use GAE.
        raise NotImplementedError(
            "advantage_estimator='vrpo' is supported only by "
            "collect_rollout_batched; the serial collect_rollout uses GAE."
        )

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
            sd, train_config.hidden_dim, device, train_config.num_layers,
            model_cls=type(learner),
        )
        opp_set = set(rng.choice(n_seats, size=pool_opp_seats, replace=False).tolist())
        learner_seats[env_idx] = set(range(n_seats)) - opp_set

    # Per-hand hole-card cache (static within a hand) — feeds the
    # centralized critic and the stored opp_holes blocks.
    hole_caches: list[np.ndarray] = [
        np.full((n_seats, 5), 255, dtype=np.uint8) for _ in range(n_envs)
    ]

    for i, env in enumerate(envs):
        _assign_pool_mix(i)
        seed = int(rng.integers(0, 2**63 - 1))
        button = int(rng.integers(0, n_seats))
        o, info = env.reset(seed, button)
        obs_vecs.append(o)
        infos.append(info)
        hole_caches[i] = _hole_cache(env)

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
    all_sizing: list[np.ndarray] = []
    all_anchors: list[int] = []
    all_refine_u: list[float] = []
    all_opp_holes: list[np.ndarray] = []
    all_log_probs: list[float] = []
    all_values: list[float] = []
    all_returns: list[float] = []
    all_advantages: list[float] = []
    all_gate_logp: list[float] = []
    all_anchor_logp: list[float] = []

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
        anchors_per_env = np.full(n_envs, -1, dtype=np.int64)
        refine_u_per_env = np.zeros(n_envs, dtype=np.float32)
        log_probs_per_env = np.zeros(n_envs, dtype=np.float32)
        gate_logp_per_env = np.zeros(n_envs, dtype=np.float32)
        anchor_logp_per_env = np.zeros(n_envs, dtype=np.float32)
        values_per_env = np.zeros(n_envs, dtype=np.float32)
        # The sizing context per env, built ONCE per step — the exact
        # tensor fed to act() is also what gets stored (no recompute drift).
        sizing_per_env = np.zeros((n_envs, 4), dtype=np.int64)
        for i in range(n_envs):
            sizing_per_env[i] = sizing_from_info(infos[i])

        if learner_idx:
            batch_obs = np.stack([obs_vecs[i] for i in learner_idx], axis=0)
            batch_gm = np.stack([infos[i].gate_mask for i in learner_idx], axis=0)
            batch_sizing = sizing_per_env[learner_idx]
            o_t = torch.from_numpy(batch_obs).to(device)
            m_t = torch.from_numpy(batch_gm).to(device)
            b_t = torch.from_numpy(batch_sizing).to(device)
            with torch.no_grad():
                _act_out = learner.act(o_t, m_t, b_t)
            g_np = _act_out.gate.cpu().numpy()
            c_np = _act_out.chips.cpu().numpy()
            an_np = _act_out.anchor.cpu().numpy()
            ru_np = _act_out.refine_u.cpu().numpy()
            lp_np = _act_out.log_prob.cpu().numpy()
            glp_np = _act_out.gate_log_prob.cpu().numpy()
            alp_np = _act_out.anchor_log_prob.cpu().numpy()
            if critic is not None:
                batch_opp = np.stack(
                    [
                        _rotate_opp_holes(hole_caches[i], int(infos[i].actor))
                        for i in learner_idx
                    ],
                    axis=0,
                )
                v_np = _critic_values(critic, device, batch_obs, batch_opp)
            else:
                v_np = _act_out.value.cpu().numpy()
            for k, i in enumerate(learner_idx):
                gates_per_env[i] = int(g_np[k])
                chips_per_env[i] = np.uint64(max(0, int(c_np[k])))
                anchors_per_env[i] = int(an_np[k])
                refine_u_per_env[i] = float(ru_np[k])
                log_probs_per_env[i] = float(lp_np[k])
                gate_logp_per_env[i] = float(glp_np[k])
                anchor_logp_per_env[i] = float(alp_np[k])
                values_per_env[i] = float(v_np[k])

        if opp_idx:
            for i in opp_idx:
                opp = opp_models[i]
                assert opp is not None
                o_t = torch.from_numpy(obs_vecs[i]).unsqueeze(0).to(device)
                m_t = torch.from_numpy(infos[i].gate_mask).unsqueeze(0).to(device)
                b_t = torch.from_numpy(sizing_per_env[i : i + 1]).to(device)
                with torch.no_grad():
                    _opp_out = opp.act(o_t, m_t, b_t)
                gates_per_env[i] = int(_opp_out.gate.cpu().numpy()[0])
                chips_per_env[i] = np.uint64(
                    max(0, int(_opp_out.chips.cpu().numpy()[0]))
                )

        for i in range(n_envs):
            env = envs[i]
            info = infos[i]
            actor = info.actor
            gate = int(gates_per_env[i])
            chips = int(chips_per_env[i])
            if actor in learner_seats[i]:
                trajectories[i][actor].append((
                    obs_vecs[i],
                    info.gate_mask,
                    gate,
                    chips if gate == GATE_RAISE else 0,
                    sizing_per_env[i].copy(),
                    float(log_probs_per_env[i]),
                    float(values_per_env[i]),
                    int(anchors_per_env[i]),
                    float(refine_u_per_env[i]),
                    _rotate_opp_holes(hole_caches[i], actor),
                    float(gate_logp_per_env[i]),
                    float(anchor_logp_per_env[i]),
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
                        all_sizing=all_sizing,
                        all_anchors=all_anchors,
                        all_refine_u=all_refine_u,
                        all_opp_holes=all_opp_holes,
                        all_log_probs=all_log_probs,
                        all_values=all_values,
                        all_returns=all_returns,
                        all_advantages=all_advantages,
                        all_gate_logp=all_gate_logp,
                        all_anchor_logp=all_anchor_logp,
                    )
                trajectories[i] = [[] for _ in range(n_seats)]
                cost_trajs[i] = [[] for _ in range(n_seats)]
                pot_trajs[i] = [[] for _ in range(n_seats)]
                street_trajs[i] = [[] for _ in range(n_seats)]
                _assign_pool_mix(i)
                seed = int(rng.integers(0, 2**63 - 1))
                button = int(rng.integers(0, n_seats))
                next_obs, next_info = env.reset(seed, button)
                hole_caches[i] = _hole_cache(env)

            obs_vecs[i] = next_obs
            infos[i] = next_info

    return _finalize_batch(
        all_obs,
        all_gate_masks,
        all_gate_actions,
        all_raise_chips,
        all_sizing,
        all_anchors,
        all_refine_u,
        all_opp_holes,
        all_log_probs,
        all_values,
        all_returns,
        all_advantages,
        all_gate_logp,
        all_anchor_logp,
        device=device,
        aggr_bonus_total_bb=aggr_bonus_total_bb,
        aggr_steps_total=aggr_steps_total,
        aggr_bonus_steps=aggr_bonus_steps,
        aggr_steps_total_by_street=tuple(aggr_steps_total_by_street),
        aggr_bonus_steps_by_street=tuple(aggr_bonus_steps_by_street),
        adv_clip=float(getattr(train_config, "adv_clip", 0.0)),
    )


# k=3/k=4 Monte-Carlo budget for the opp-outcome obs feature during
# BATCHED TRAINING rollout. Lower than the 1024-sample serial/UI/eval
# default to cut the dominant per-decision encode cost (the feature is
# ~94% of obs-build, so ~halving the MC budget ~doubles obs-build
# throughput). The ~1-3% extra MC noise on 12 of 991 dims is a benign,
# fine-tune-safe regularizer; the UI/eval/serial paths keep 1024 so the
# study tool still computes the more accurate estimate.
# 2026-06-20: MC=256 destabilized the gate in the shallow clubgg block
# (vTwo10 saturated-collapsed at u490, in clubgg, even after an LR cut to
# 1.5e-4). Raised to 384 as a less-noisy compromise — still cheaper than
# the 1024 UI/eval path, but with ~1/3 less MC variance on the 12 dims.
TRAIN_OPP_OUTCOME_MC = 384


def collect_rollout_batched(
    learner: ActorCritic,
    pool: OpponentPool,
    game_config: GameConfig,
    train_config: TrainingConfig,
    rng: np.random.Generator,
    critic: CentralCritic | None = None,
    out_slabs: "dict[str, np.ndarray] | None" = None,
    snapshot_cache: "dict[int, ActorCritic] | None" = None,
    env: "BatchedBombPotEnv | None" = None,
    env_cache: "dict[tuple, BatchedBombPotEnv] | None" = None,
) -> Batch:
    """Batched rollout using `BatchedBombPotEnv` + snapshot-bucket
    opponent forwards. Drives all envs through `apply_hybrid_batch`.

    `snapshot_cache` (P5): caller-owned cache of built frozen-opponent
    models, keyed by pool snapshot index. Multiconfig passes one dict for
    the whole update (pool membership is frozen across it — the caller
    snapshots AFTER trainer.update), so each pool member is constructed
    once per update instead of once per sub-rollout (~240x -> ~8x builds).
    Default None = a fresh local dict, today's exact behavior. CUDA-run
    bit-exact (module init consumes the CPU torch generator; batched
    sampling uses the CUDA generator); on CPU-only runs the shared cache
    shifts the sampling stream from the second sub-rollout on (no batched
    bit-exact contract — parity is at the env level).

    `out_slabs` (P7+P8, multiconfig shared staging): caller-provided numpy
    VIEWS — one per output slab, keyed obs/gm/ga/rc/sz/an/ru/oh/lp/glp/alp/
    v/ret/adv — into one big preallocated host buffer. When given, the
    collector writes into them instead of allocating its own slabs and
    returns a Batch of CPU view-tensors (no device copy except the tiny
    per-sub advantage-normalization hop, which stays on the learner device
    for bit-exactness with the legacy path). Default None = the original
    self-allocating, finalize-to-device path, byte-identical to before."""
    n_envs = train_config.num_envs
    n_seats = game_config.num_seats
    reward_norm = 1.0 / float(game_config.bb)
    gamma = train_config.gamma
    lam = train_config.lam
    pool_mix_prob = float(train_config.pool_mix_prob)
    pool_opp_seats = int(train_config.pool_opp_seats)
    pool_opp_seats = max(0, min(pool_opp_seats, n_seats - 1))
    device = next(learner.parameters()).device

    # VRPO / Expected-SARSA advantage flip (V5_DESIGN.md W2.5). Needs the
    # centralized critic's dueling Q head; train.py additionally requires
    # q_aux_coef>0 so the head is trained (else the flip is identical to GAE).
    use_vrpo = getattr(train_config, "advantage_estimator", "gae") == "vrpo"
    if use_vrpo and (critic is None or getattr(critic, "q_actions", 0) <= 0):
        raise ValueError(
            "advantage_estimator='vrpo' needs a CentralCritic with a dueling "
            "Q head (q_actions>0)."
        )

    # Env reuse (P3, 2026-07-12): multiconfig passes `env_cache` keyed by
    # (n_envs, num_seats, variant). When a sub-rollout shares that key with a
    # prior sub, reconfigure stacks/ante in place instead of reallocating
    # ~N Rust GameStates + Python cache arrays. `env=` is a direct override
    # for tests. Single-config callers leave both None → fresh env (unchanged).
    if env is not None:
        if env.n != n_envs:
            raise ValueError(
                f"env.n={env.n} != train_config.num_envs={n_envs}"
            )
        if not env.can_reconfigure(game_config):
            raise ValueError(
                "provided env cannot reconfigure to game_config "
                f"(seats/variant mismatch)"
            )
        env.reconfigure(game_config)
        # Keep EV / MC knobs aligned with this train_config.
        env._ev_runout_samples = int(train_config.ev_runout_samples)
    else:
        cache_key = (n_envs, game_config.num_seats, game_config.variant)
        if env_cache is not None and cache_key in env_cache:
            env = env_cache[cache_key]
            env.reconfigure(game_config)
            env._ev_runout_samples = int(train_config.ev_runout_samples)
        else:
            env = BatchedBombPotEnv(
                n_envs,
                game_config,
                ev_runout_samples=train_config.ev_runout_samples,
                opp_outcome_mc=TRAIN_OPP_OUTCOME_MC,
            )
            if env_cache is not None:
                env_cache[cache_key] = env

    # P5: reuse the caller's per-update cache when given (multiconfig), else a
    # local one (single-config callers — unchanged behavior).
    snapshot_models: dict[int, ActorCritic] = (
        snapshot_cache if snapshot_cache is not None else {}
    )

    def _get_snapshot_model(sd_idx: int) -> ActorCritic:
        m = snapshot_models.get(sd_idx)
        if m is not None:
            return m
        # No deepcopy (P5): pool.snapshot() stores detached clones, this path
        # bypasses OpponentPool.sample() (whose deepcopy protects the SERIAL
        # path), load_state_dict copies rather than aliases, and nothing here
        # mutates the dict.
        m = _build_frozen_model(
            pool.snapshots[sd_idx],
            train_config.hidden_dim, device, train_config.num_layers,
            model_cls=type(learner),
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
    # Per-hand hole cache (holes are static within a hand): one bulk
    # fetch per reset wave feeds the critic input + stored opp blocks.
    holes_cache = np.asarray(env._be.all_hole_cards_batch(), dtype=np.uint8)

    # Array-backed per-(env, seat) trajectory storage. Every per-step
    # append is a vectorized fancy-index write. `MAX_STEPS_PER_SEAT`
    # caps the per-(env, seat) action count for one hand. The worst
    # case is a 1bb-increment min-raise war (engine floors bets at 1bb
    # and raise increments at the last raise size): a seat commits
    # >=2bb per aggressive action, so a 300bb stack (the --stack-range
    # default cap) tops out near ~150 actions per seat per hand. 32 was
    # exceeded in practice on a deep-tier rollout (vTwo1 update 94,
    # 2026-06-10). 192 bounds the theoretical worst case with margin;
    # cost is only the per-(env, seat) trajectory arrays (~4GB at 49k
    # envs) — the obs pool / output slabs use POOL_SLACK_PER_ENV below,
    # NOT this capacity, and flush temporaries are bounded by the
    # actual max trajectory length per flush.
    MAX_STEPS_PER_SEAT = 192
    traj_lengths = np.zeros((n_envs, n_seats), dtype=np.int32)
    # Absolute index into `step_obs_pool` / `step_gm_pool` per (env, seat, slot).
    traj_obs_idx = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.int64)
    traj_gate = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.int8)
    traj_chips = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.int64)
    traj_sizing = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT, 4), dtype=np.int64)
    traj_anchor = np.full((n_envs, n_seats, MAX_STEPS_PER_SEAT), -1, dtype=np.int8)
    traj_u = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
    traj_log_p = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
    traj_gate_lp = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
    traj_anchor_lp = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
    traj_value = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
    # VRPO: Q(s_t, a_t) and V^π(s_t)=Σ_a π(a)Q(s_t,a) per learner step, laid
    # out like traj_value; the Expected-SARSA(λ) scan consumes them. None when
    # the estimator is GAE (never indexed in that path).
    traj_q_taken = (
        np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
        if use_vrpo else None
    )
    traj_vpi = (
        np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
        if use_vrpo else None
    )
    # Per-step cost / pot / street parallel arrays, mirrored shape.
    costs_arr = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
    pots_arr = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.float32)
    streets_arr = np.zeros((n_envs, n_seats, MAX_STEPS_PER_SEAT), dtype=np.int8)

    # Flat pre-allocated obs / gate-mask pool. Each step appends
    # `learner_idx_np.size` rows at `pool_cursor`; trajectory slots
    # store the absolute pool index. One contiguous buffer makes the
    # terminal-flush gather a single `np.take` rather than a Python
    # walk over chunked storage.
    rollout_target = int(train_config.rollout_length)
    # Slack for learner steps written after `wcursor` last crossed the
    # rollout target (in-flight, unflushed hands). This is a per-env
    # STATISTICAL bound (~avg hand length, a handful of steps), NOT the
    # per-seat capacity above — there is no reason for it to scale with
    # MAX_STEPS_PER_SEAT. Note the cost of oversizing is only VIRTUAL
    # address space (np.empty pages materialize on first write and the
    # slack tail is mostly never written), but keeping the bound honest
    # documents the actual requirement, and the explicit guards below
    # turn a (near-impossible) overflow into a clean error instead of a
    # silent numpy shape mismatch. 32 steps/env is ~50x observed need.
    # (POOL_SLACK_PER_ENV is module-level — shared with multiconfig's
    # shared-staging allocation.)
    pool_cap = rollout_target + n_envs * POOL_SLACK_PER_ENV
    # env.obs_dim, not the OBS_DIM constant: the batched env's layout is
    # per-variant (991 PLO / 995 NLH).
    step_obs_pool = np.empty((pool_cap, env.obs_dim), dtype=np.float32)
    step_gm_pool = np.empty((pool_cap, GATE_ACTIONS), dtype=bool)
    pool_cursor = 0

    # Pre-allocated output slabs. Eliminates the `np.stack` over millions
    # of small arrays at finalize time. `wcursor` tracks the count of
    # transitions written so far across all terminal flushes. On CUDA,
    # back the slabs with pinned host memory so the finalize transfer
    # can run with `non_blocking=True` and overlap the first PPO forward.
    out_cap = rollout_target + n_envs * POOL_SLACK_PER_ENV
    if out_slabs is not None:
        # P8 shared staging: write into the caller's views of one big host
        # buffer. Capacity is computed with the SAME formula the caller used,
        # so the slab-overflow guard below keeps identical semantics; the
        # width assert catches any variant/obs-dim drift between the caller's
        # allocation and this env.
        all_obs_arr = out_slabs["obs"]
        all_gm_arr = out_slabs["gm"]
        all_ga_arr = out_slabs["ga"]
        all_rc_arr = out_slabs["rc"]
        all_sz_arr = out_slabs["sz"]
        all_an_arr = out_slabs["an"]
        all_ru_arr = out_slabs["ru"]
        all_oh_arr = out_slabs["oh"]
        all_lp_arr = out_slabs["lp"]
        all_glp_arr = out_slabs["glp"]
        all_alp_arr = out_slabs["alp"]
        all_v_arr = out_slabs["v"]
        all_ret_arr = out_slabs["ret"]
        all_adv_arr = out_slabs["adv"]
        all_last_arr = out_slabs["last"]
        assert all_obs_arr.shape[0] == out_cap, (
            f"out_slabs capacity {all_obs_arr.shape[0]} != expected {out_cap}"
        )
        assert all_obs_arr.shape[1] == env.obs_dim, (
            f"out_slabs obs width {all_obs_arr.shape[1]} != env {env.obs_dim}"
        )
        assert all_oh_arr.shape[2] == game_config.hole_count, (
            f"out_slabs hole width {all_oh_arr.shape[2]} != "
            f"config {game_config.hole_count}"
        )
    else:
        # Pinned host memory makes the finalize H2D copy overlap downstream compute
        # (via non_blocking=True), but PINNING THE ~36GB obs slab can cost 60+ SECONDS
        # PER UPDATE on some hosts (measured: torch 2.11 / AMD EPYC pins 36GB in ~64s,
        # single-threaded — it was the dominant per-update cost and looked like a
        # hang). The pinning tax dwarfs the few seconds non_blocking saves at the
        # finalize, so pinning is DEFAULT OFF. Re-enable with PLO5BP_PIN_ROLLOUT=1 on
        # hosts where large-buffer pinning is cheap. Output is bit-identical either
        # way (non_blocking=True on non-pinned memory simply degrades to a blocking
        # copy — same data).
        _pin = (
            (device.type == "cuda" if isinstance(device, torch.device)
             else str(device).startswith("cuda"))
            and os.environ.get("PLO5BP_PIN_ROLLOUT", "0") == "1"
        )
        _pinned_keepalive: list[torch.Tensor] = []

        def _alloc_slab(shape, np_dtype, torch_dtype):
            if _pin:
                t = torch.empty(shape, dtype=torch_dtype, pin_memory=True)
                _pinned_keepalive.append(t)
                return t.numpy()
            return np.empty(shape, dtype=np_dtype)

        all_obs_arr = _alloc_slab((out_cap, env.obs_dim), np.float32, torch.float32)
        all_gm_arr = _alloc_slab((out_cap, GATE_ACTIONS), bool, torch.bool)
        all_ga_arr = _alloc_slab(out_cap, np.int64, torch.int64)
        all_rc_arr = _alloc_slab(out_cap, np.int64, torch.int64)
        all_sz_arr = _alloc_slab((out_cap, 4), np.int64, torch.int64)
        all_an_arr = _alloc_slab(out_cap, np.int64, torch.int64)
        all_ru_arr = _alloc_slab(out_cap, np.float32, torch.float32)
        all_oh_arr = _alloc_slab(
            (out_cap, 5, game_config.hole_count), np.uint8, torch.uint8
        )
        all_lp_arr = _alloc_slab(out_cap, np.float32, torch.float32)
        all_glp_arr = _alloc_slab(out_cap, np.float32, torch.float32)
        all_alp_arr = _alloc_slab(out_cap, np.float32, torch.float32)
        all_v_arr = _alloc_slab(out_cap, np.float32, torch.float32)
        all_ret_arr = _alloc_slab(out_cap, np.float32, torch.float32)
        all_adv_arr = _alloc_slab(out_cap, np.float32, torch.float32)
        all_last_arr = _alloc_slab(out_cap, bool, torch.bool)
    wcursor = 0

    aggression_bonus_c = float(train_config.aggression_bonus_c)
    retroactive_bonus_c = float(train_config.retroactive_bonus_c)
    aggr_bonus_total_bb = 0.0
    aggr_steps_total = 0
    aggr_bonus_steps = 0
    # 3 buckets: 0=flop, 1=turn, 2=river. Bomb pots have no preflop
    # action so we never bucket street index 0.
    aggr_steps_total_by_street: list[int] = [0, 0, 0]
    aggr_bonus_steps_by_street: list[int] = [0, 0, 0]

    # P5 step H2D: long-lived pinned staging (CUDA). See _PinnedStepH2D.
    _step_h2d = _PinnedStepH2D(n_envs, env.obs_dim, device)

    def _forward(model: ActorCritic, group: np.ndarray, obs_arr: np.ndarray,
                 gate_mask_arr: np.ndarray, sizing_arr: np.ndarray,
                 want_marginal: bool) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray | None, torch.Tensor,
    ]:
        b_obs = obs_arr[group]
        b_gm = gate_mask_arr[group]
        b_sizing = sizing_arr[group]
        with record_function("step2/learner_h2d"):
            # Pinned staging + non_blocking H2D on CUDA; CPU path is
            # from_numpy.to. Values bit-identical; see _PinnedStepH2D.
            o_t, m_t, b_t = _step_h2d.upload(b_obs, b_gm, b_sizing)
        with record_function("step3/learner_forward"):
            with torch.inference_mode():
                # P10: only the LEARNER's marginal is consumed (VRPO V^pi at the
                # call site); opponent-group forwards discard it. Gate the
                # compute + its extra D2H sync on want_marginal rather than the
                # collector-wide use_vrpo. Bit-exact: the marginal is a pure
                # post-sampling readout — no extra network pass, no RNG draw
                # (network.act docstring), so the sampled action is unchanged.
                if want_marginal:
                    _fw_out = model.act(o_t, m_t, b_t, return_marginal=True)
                else:
                    _fw_out = model.act(o_t, m_t, b_t)
        with record_function("step5/action_d2h"):
            # Coalesce device->host copies (each forces a full CUDA sync)
            # into two by stacking same-dtype outputs: one int64 transfer
            # for (gate, chips, anchor) and one float32 transfer for
            # (log_prob, refine_u, value, gate_log_prob, anchor_log_prob).
            # Bit-identical to per-tensor copies — only the sync count changes.
            ints = torch.stack(
                (_fw_out.gate, _fw_out.chips, _fw_out.anchor), dim=0
            ).cpu().numpy()
            floats = torch.stack(
                (_fw_out.log_prob, _fw_out.refine_u, _fw_out.value,
                 _fw_out.gate_log_prob, _fw_out.anchor_log_prob), dim=0
            ).cpu().numpy()
            marg_np = (
                _fw_out.action_marginal.float().cpu().numpy()
                if _fw_out.action_marginal is not None
                else None
            )
            return (
                ints[0].astype(np.uint8),
                ints[1].astype(np.int64),
                ints[2].astype(np.int64),
                floats[0],
                floats[1],
                floats[2],
                floats[3],
                floats[4],
                marg_np,
                # P9: the already-uploaded obs tensor — the critic forward for
                # the learner group reuses it instead of re-fancy-indexing and
                # re-uploading the identical rows (~45GB/update duplicated).
                o_t,
            )

    env_idx_range = np.arange(n_envs)

    while wcursor < rollout_target:
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
        anchors_per_env = np.full(n_envs, -1, dtype=np.int64)
        refine_u_per_env = np.zeros(n_envs, dtype=np.float32)
        log_probs_per_env = np.zeros(n_envs, dtype=np.float32)
        gate_logp_per_env = np.zeros(n_envs, dtype=np.float32)
        anchor_logp_per_env = np.zeros(n_envs, dtype=np.float32)
        values_per_env = np.zeros(n_envs, dtype=np.float32)
        if use_vrpo:
            q_taken_per_env = np.zeros(n_envs, dtype=np.float32)
            vpi_per_env = np.zeros(n_envs, dtype=np.float32)

        # Per-step sizing context (min, max, pot, to_call) — built once;
        # the same array feeds act() and the trajectory store.
        to_call_step = np.maximum(
            pre_bet_to_call.astype(np.int64)
            - pre_street_commit[env_idx_range, safe_actors].astype(np.int64),
            0,
        )
        sizing_step = np.stack(
            [
                min_raise.astype(np.int64),
                max_raise.astype(np.int64),
                env._pot.astype(np.int64),
                to_call_step,
            ],
            axis=-1,
        )

        if learner_idx_np.size:
            g_np, c_np, an_np, lp_np, ru_np, v_np, glp_np, alp_np, marg_np, o_t_l = (
                _forward(
                    learner, learner_idx_np, obs, gate_masks, sizing_step,
                    want_marginal=use_vrpo,
                )
            )
            gates_per_env[learner_idx_np] = g_np
            chips_per_env[learner_idx_np] = np.maximum(c_np, 0).astype(np.uint64)
            anchors_per_env[learner_idx_np] = an_np
            refine_u_per_env[learner_idx_np] = ru_np
            log_probs_per_env[learner_idx_np] = lp_np
            gate_logp_per_env[learner_idx_np] = glp_np
            anchor_logp_per_env[learner_idx_np] = alp_np
            if critic is not None:
                with record_function("step3b/critic_forward"):
                    opp_block = _rotate_opp_holes_batch(
                        holes_cache, learner_idx_np, safe_actors[learner_idx_np]
                    )
                    if use_vrpo:
                        # P9: o_t_l IS from_numpy(obs[learner_idx_np]).to(device)
                        # — identical bytes, uploaded once by _forward.
                        v_l, q_l = _critic_q_values(
                            critic, device, o_t_l, opp_block
                        )
                        values_per_env[learner_idx_np] = v_l
                        if q_l.shape[-1] == 3:
                            # Pooled raise column (q_pooled): the gate IS the
                            # q index (GATE_RAISE == 2), and the 13-way act()
                            # marginal collapses to gate probs exactly —
                            # Σ_k p_raise·π(k) = p_raise.
                            q_idx_l = g_np.astype(np.int64)
                            marg_l = np.stack(
                                [
                                    marg_np[:, 0],
                                    marg_np[:, 1],
                                    marg_np[:, 2:].sum(-1),
                                ],
                                axis=-1,
                            )
                        else:
                            # q index: Fold/CheckCall use the gate directly,
                            # Raise uses 2+anchor (matches ppo.py q_idx and
                            # the marginal layout).
                            q_idx_l = np.where(
                                g_np == GATE_RAISE, 2 + an_np, g_np.astype(np.int64)
                            )
                            marg_l = marg_np
                        rows_l = np.arange(learner_idx_np.size)
                        q_taken_per_env[learner_idx_np] = q_l[rows_l, q_idx_l]
                        vpi_per_env[learner_idx_np] = (marg_l * q_l).sum(-1)
                    else:
                        # P9: same obs-tensor reuse as the VRPO branch.
                        values_per_env[learner_idx_np] = _critic_values(
                            critic, device, o_t_l, opp_block
                        )
            else:
                values_per_env[learner_idx_np] = v_np

        # Group active opponent envs by snapshot index. `np.unique` over
        # the masked column replaces the per-env dict-build loop.
        if is_opp_active.any():
            with record_function("step4/opp_forwards"):
                opp_snap_col = np.where(is_opp_active, env_snapshot_idx_arr, -1)
                for sd_idx in np.unique(opp_snap_col[opp_snap_col >= 0]):
                    sd_idx_int = int(sd_idx)
                    group = np.nonzero(opp_snap_col == sd_idx)[0]
                    m = _get_snapshot_model(sd_idx_int)
                    g_np, c_np, _, _, _, _, _, _, _, _ = _forward(
                        m, group, obs, gate_masks, sizing_step, want_marginal=False
                    )
                    gates_per_env[group] = g_np
                    chips_per_env[group] = np.maximum(c_np, 0).astype(np.uint64)

        # Vectorized trajectory snapshot: one bulk obs/gm copy into the
        # flat pool, plus fancy-index writes into the per-(env, seat)
        # arrays. The (env, seat, slot) -> pool index map is one int.
        if learner_idx_np.size:
            k_step = learner_idx_np.size
            pool_end = pool_cursor + k_step
            if pool_end > pool_cap:
                raise RuntimeError(
                    f"obs pool overflow: {pool_end} > pool_cap={pool_cap} "
                    f"(rollout_target={rollout_target} + "
                    f"{n_envs}x{POOL_SLACK_PER_ENV} slack)"
                )
            step_obs_pool[pool_cursor:pool_end] = obs[learner_idx_np]
            step_gm_pool[pool_cursor:pool_end] = gate_masks[learner_idx_np]
            pool_indices = np.arange(pool_cursor, pool_end, dtype=np.int64)
            pool_cursor = pool_end

            learner_actors = safe_actors[learner_idx_np]
            slots = traj_lengths[learner_idx_np, learner_actors]
            if (slots >= MAX_STEPS_PER_SEAT).any():
                raise RuntimeError(
                    f"per-seat trajectory length exceeded "
                    f"MAX_STEPS_PER_SEAT={MAX_STEPS_PER_SEAT}"
                )

            l_gates = gates_per_env[learner_idx_np]
            l_chips = chips_per_env[learner_idx_np]
            traj_obs_idx[learner_idx_np, learner_actors, slots] = pool_indices
            traj_gate[learner_idx_np, learner_actors, slots] = l_gates.astype(np.int8)
            traj_chips[learner_idx_np, learner_actors, slots] = np.where(
                l_gates == GATE_RAISE, l_chips.astype(np.int64), 0
            )
            traj_sizing[learner_idx_np, learner_actors, slots] = (
                sizing_step[learner_idx_np]
            )
            traj_anchor[learner_idx_np, learner_actors, slots] = (
                anchors_per_env[learner_idx_np].astype(np.int8)
            )
            traj_u[learner_idx_np, learner_actors, slots] = (
                refine_u_per_env[learner_idx_np]
            )
            traj_log_p[learner_idx_np, learner_actors, slots] = (
                log_probs_per_env[learner_idx_np]
            )
            traj_gate_lp[learner_idx_np, learner_actors, slots] = (
                gate_logp_per_env[learner_idx_np]
            )
            traj_anchor_lp[learner_idx_np, learner_actors, slots] = (
                anchor_logp_per_env[learner_idx_np]
            )
            traj_value[learner_idx_np, learner_actors, slots] = (
                values_per_env[learner_idx_np]
            )
            if use_vrpo:
                traj_q_taken[learner_idx_np, learner_actors, slots] = (
                    q_taken_per_env[learner_idx_np]
                )
                traj_vpi[learner_idx_np, learner_actors, slots] = (
                    vpi_per_env[learner_idx_np]
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
        with record_function("step6+7/rust_apply"):
            newly_terminal = np.asarray(
                env._be.apply_hybrid_batch(gates_dispatch, chips_per_env), dtype=bool
            )
        # Refresh now to capture post-step total_commit (and everything
        # else) BEFORE reset_terminal_batch wipes terminal envs.
        # Skip the expensive obs encode for newly-terminal rows: their
        # post-apply obs is never read (reset + subset-refresh re-deal
        # them before the next act). Pack still runs full-batch so
        # total_commit / legal / actors stay correct for payouts and
        # aggression bookkeeping. Bit-exact for non-terminal rows;
        # terminal rows get zeros (same as a full encode of actor==-1).
        with record_function("step1a/refresh"):
            if newly_terminal.any() and not newly_terminal.all():
                env._refresh(encode_mask=~newly_terminal)
            elif newly_terminal.all():
                env._refresh(encode_mask=np.zeros(n_envs, dtype=bool))
            else:
                env._refresh()
        post_total_commit = env._total_commit

        # Rust-parallel aggression bonus + per-step cost/pot/street
        # bookkeeping. Replaces the per-env Python arithmetic loop.
        with record_function("step8/aggression_bonus"):
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
            with record_function("step9a/payouts"):
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

            # Vectorized terminal flush: process every newly-terminal env
            # in one numpy block. Replaces the per-(env, seat, t) Python
            # walk that built tuple lists for `_apply_retroactive_bonus`
            # and `_flush_trajectory`. Net work: one (T, S, L) bonus
            # mask + an L-step backward GAE scan vectorized over T*S +
            # one np.take per output slab.
            #
            # L bounds every flush temporary by the longest trajectory
            # actually present in this flush (typically 8-16 actions),
            # NOT the MAX_STEPS_PER_SEAT capacity — with MAX=192 and
            # thousands of terminals per step, (T, S, MAX) temporaries
            # would cost ~GBs of allocation/zeroing traffic per flush
            # for slots that are empty by construction. Slots in
            # [L, MAX) are inactive, so the output is bit-identical.
            term_envs = np.nonzero(newly_terminal)[0]
            T = int(term_envs.size)
            if T:
                S = n_seats
                lengths = traj_lengths[term_envs]                       # (T, S)
                flush_mask = learner_seats_mask[term_envs] & (lengths > 0)  # (T, S)
                L = int(lengths.max())
                t_idx = np.arange(L, dtype=np.int32)
                active_step = (
                    (t_idx[None, None, :] < lengths[..., None])
                    & flush_mask[..., None]
                )  # (T, S, L)

                # Retroactive bonus (vectorized over (T, S, L)).
                with record_function("step9b/retroactive_bonus"):
                    payout_chips = np.rint(won_f32[term_envs]).astype(np.int64)  # (T, S)
                    total_pot_chips = post_total_commit[term_envs].astype(np.int64).sum(axis=1)  # (T,)
                    two_pay = 2 * payout_chips
                    share_eq = (two_pay == total_pot_chips[:, None]) & (
                        total_pot_chips[:, None] > 0
                    )  # (T, S)
                    share_gt = two_pay > total_pot_chips[:, None]            # (T, S)

                    # Mixed fancy+basic indexing copies only the [:L]
                    # region — never materialize (T, S, MAX).
                    gates_t = traj_gate[term_envs, :, :L]                    # (T, S, L)
                    costs_t = costs_arr[term_envs, :, :L]                    # (T, S, L) copy
                    pots_t = pots_arr[term_envs, :, :L]                      # (T, S, L)
                    streets_t = streets_arr[term_envs, :, :L]                # (T, S, L)

                    is_raise = (gates_t == GATE_RAISE)
                    is_call_with_chips = (gates_t == GATE_CHECK_CALL) & (costs_t < 0.0)
                    qualified = active_step & (
                        (share_gt[..., None] & is_raise)
                        | (share_eq[..., None] & (is_raise | is_call_with_chips))
                    )
                    # Counter updates are unconditional — even with c=0 we
                    # report the count of steps that *would* receive a bonus
                    # so disabled-bonus runs can still measure aggression
                    # share by street. Cost mutation is gated on c.
                    aggr_bonus_steps += int(qualified.sum())
                    for s_idx in range(3):
                        aggr_bonus_steps_by_street[s_idx] += int(
                            ((streets_t == (s_idx + 1)) & qualified).sum()
                        )
                    if retroactive_bonus_c != 0.0:
                        bonus = qualified.astype(np.float32) * (
                            np.float32(retroactive_bonus_c) * pots_t
                        )
                        costs_t += bonus
                        aggr_bonus_total_bb += float(bonus.sum())

                # GAE backward scan vectorized over (T, S).
                with record_function("step9c/gae_scan"):
                    vals_t = traj_value[term_envs, :, :L]                    # (T, S, L)
                    won_bb = won_f32[term_envs].astype(np.float32) * np.float32(reward_norm)
                    last_gae = np.zeros((T, S), dtype=np.float32)
                    advs_t = np.zeros((T, S, L), dtype=np.float32)
                    last_t_arr = (lengths.astype(np.int32) - 1)              # (T, S)
                    gamma_f = np.float32(gamma)
                    lam_f = np.float32(lam)
                    # L is the longest trajectory in this flush; slots in
                    # [length, L) are inactive (`is_last`/`active_tm`
                    # all-False), so the scan is bit-identical to one
                    # over the full MAX capacity.
                    for t in range(L - 1, -1, -1):
                        is_last = (t == last_t_arr) & flush_mask
                        active_tm = (t <= last_t_arr) & flush_mask
                        reward_t = costs_t[..., t] + np.where(is_last, won_bb, np.float32(0.0))
                        if t + 1 < L:
                            next_v = np.where(is_last, np.float32(0.0), vals_t[..., t + 1])
                        else:
                            next_v = np.zeros((T, S), dtype=np.float32)
                        delta = reward_t + gamma_f * next_v - vals_t[..., t]
                        new_gae = delta + gamma_f * lam_f * last_gae
                        last_gae = np.where(active_tm, new_gae, last_gae)
                        advs_t[..., t] = np.where(active_tm, last_gae, np.float32(0.0))
                    rets_t = advs_t + vals_t

                    # VRPO / Q-boosted advantage (arXiv:2605.19235 eq 3.2):
                    # Â = (Q(s,a) − V^π(s)) + λ-trace of δ⁺, with
                    # δ⁺ = r + γ·V^π(s') − Q(s,a), V^π = Σ_a π(a)Q(s,a)
                    # (traj_vpi). `advantages` use this; `returns` stay GAE
                    # (the V-head target). At Q≡V (zero-init head) the
                    # leading term is ~0 and δ⁺ reduces to the GAE residual,
                    # so this matches the scan above (golden parity test).
                    # 2026-07-12: leading term RESTORED — the shipped form
                    # was residual-only; see _vrpo_advantage_scan.
                    if use_vrpo:
                        q_es = traj_q_taken[term_envs, :, :L]        # (T, S, L)
                        vpi_es = traj_vpi[term_envs, :, :L]          # (T, S, L)
                        adv_out_t = _vrpo_advantage_scan(
                            costs_t, won_bb, q_es, vpi_es,
                            last_t_arr, flush_mask, gamma_f, lam_f,
                        )
                    else:
                        adv_out_t = advs_t

                # Gather the (T, S, L) → (n_new,) flat slab and copy
                # into the preallocated output arrays at `wcursor`.
                with record_function("step9d/slab_copies"):
                    flat = active_step.ravel()
                    n_new = int(flat.sum())
                    if n_new:
                        if wcursor + n_new > out_cap:
                            raise RuntimeError(
                                f"output slab overflow: {wcursor + n_new} > "
                                f"out_cap={out_cap} (rollout_target="
                                f"{rollout_target} + {n_envs}x"
                                f"{POOL_SLACK_PER_ENV} slack)"
                            )
                        sel = np.nonzero(flat)[0]
                        obs_idx = traj_obs_idx[term_envs, :, :L].ravel()[sel]
                        end = wcursor + n_new
                        np.take(step_obs_pool, obs_idx, axis=0, out=all_obs_arr[wcursor:end])
                        np.take(step_gm_pool, obs_idx, axis=0, out=all_gm_arr[wcursor:end])
                        all_ga_arr[wcursor:end] = traj_gate[term_envs, :, :L].ravel()[sel].astype(np.int64)
                        all_rc_arr[wcursor:end] = traj_chips[term_envs, :, :L].ravel()[sel]
                        all_sz_arr[wcursor:end] = (
                            traj_sizing[term_envs, :, :L].reshape(-1, 4)[sel]
                        )
                        all_an_arr[wcursor:end] = (
                            traj_anchor[term_envs, :, :L].ravel()[sel].astype(np.int64)
                        )
                        all_ru_arr[wcursor:end] = traj_u[term_envs, :, :L].ravel()[sel]
                        # Rotated opp holes are constant per (env, seat) per
                        # hand: build one (T, S, 5, 5) block from the hand's
                        # hole cache and index it by sel // L.
                        seat_ids = np.arange(S)
                        j5 = np.arange(5)
                        rot_seats = (seat_ids[:, None] + 1 + j5[None, :]) % S
                        rot_block = holes_cache[term_envs][:, rot_seats]
                        invalid = (j5 + 1) >= S
                        if invalid.any():
                            rot_block = np.where(
                                invalid[None, None, :, None], np.uint8(255), rot_block
                            )
                        ts_idx = sel // L
                        all_oh_arr[wcursor:end] = (
                            rot_block.reshape(
                                T * S, 5, game_config.hole_count
                            )[ts_idx]
                        )
                        all_lp_arr[wcursor:end] = traj_log_p[term_envs, :, :L].ravel()[sel]
                        all_glp_arr[wcursor:end] = traj_gate_lp[term_envs, :, :L].ravel()[sel]
                        all_alp_arr[wcursor:end] = traj_anchor_lp[term_envs, :, :L].ravel()[sel]
                        all_v_arr[wcursor:end] = vals_t.ravel()[sel]
                        all_ret_arr[wcursor:end] = rets_t.ravel()[sel]
                        all_adv_arr[wcursor:end] = adv_out_t.ravel()[sel]
                        # Terminal flag: the seat's last decision of the hand
                        # (same (T, S, L) ravel space as every slab above; sel
                        # rows are active ⊂ flush, so no extra masking).
                        all_last_arr[wcursor:end] = (
                            t_idx[None, None, :] == last_t_arr[..., None]
                        ).ravel()[sel]
                        wcursor = end

                with record_function("step9e/pool_mix"):
                    traj_lengths[term_envs] = 0
                    for i_int in term_envs.tolist():
                        _assign_pool_mix(int(i_int))

            with record_function("step9f/reset_terminal"):
                env._be.reset_terminal_batch(new_seeds, new_buttons, reset_mask)
                env._reset_seeds = np.where(reset_mask, new_seeds, env._reset_seeds)
                # Refresh the per-hand hole cache for the re-dealt envs
                # only (holes are static within a hand; non-reset envs
                # keep their prior rows). Subset fetch mirrors
                # observation_and_features_subset_batch.
                term_idx = term_envs.astype(np.int64)
                holes_sub = np.asarray(
                    env._be.all_hole_cards_subset_batch(term_idx), dtype=np.uint8
                )
                holes_cache[term_envs] = holes_sub
                # Second refresh to pick up the post-reset state for the next
                # iteration. `reset_terminal_batch` mutates ONLY the masked
                # (terminal) envs, and nothing above mutated non-masked envs'
                # engine state since the post-apply refresh — so only the
                # reset rows need re-packing/re-encoding. The subset refresh
                # is bit-exact-equivalent to a full `_refresh()` here (see
                # `_refresh_subset` docstring + test_refresh_subset_parity).
                env._refresh_subset(reset_mask)

    if out_slabs is not None:
        # P7 shared-staging finalize: no full H2D — the rows already sit in the
        # caller's big host buffer. Only the per-sub advantage normalization
        # hops to the learner device: it ran on CUDA in the legacy path
        # (_finalize_batch_arr), and a CPU reimplementation would drift f32
        # reduction order, so ship the tiny (wcursor,) vector up, run the
        # IDENTICAL op sequence, and write the result back into the slab.
        with record_function("step11/shared_adv_norm"):
            adv_view = all_adv_arr[:wcursor]
            adv_t = torch.from_numpy(adv_view).to(device, non_blocking=True)
            adv_mean = adv_t.mean()
            adv_std = adv_t.std().clamp(min=1e-8)
            adv_t = (adv_t - adv_mean) / adv_std
            _adv_clip = float(getattr(train_config, "adv_clip", 0.0))
            if _adv_clip > 0.0:
                # Same fat-tail clamp as _finalize_batch (serial parity).
                adv_t = adv_t.clamp(-_adv_clip, _adv_clip)
            np.copyto(adv_view, adv_t.cpu().numpy())
        return Batch(
            obs=torch.from_numpy(all_obs_arr[:wcursor]),
            gate_masks=torch.from_numpy(all_gm_arr[:wcursor]),
            gate_actions=torch.from_numpy(all_ga_arr[:wcursor]),
            raise_chips=torch.from_numpy(all_rc_arr[:wcursor]),
            sizing=torch.from_numpy(all_sz_arr[:wcursor]),
            anchor_actions=torch.from_numpy(all_an_arr[:wcursor]),
            refine_u=torch.from_numpy(all_ru_arr[:wcursor]),
            opp_holes=torch.from_numpy(all_oh_arr[:wcursor]),
            log_probs=torch.from_numpy(all_lp_arr[:wcursor]),
            values=torch.from_numpy(all_v_arr[:wcursor]),
            returns=torch.from_numpy(all_ret_arr[:wcursor]),
            advantages=torch.from_numpy(adv_view),
            old_gate_logp=torch.from_numpy(all_glp_arr[:wcursor]),
            old_anchor_logp=torch.from_numpy(all_alp_arr[:wcursor]),
            is_terminal=torch.from_numpy(all_last_arr[:wcursor]),
            aggr_bonus_total_bb=float(aggr_bonus_total_bb),
            aggr_steps_total=int(aggr_steps_total),
            aggr_bonus_steps=int(aggr_bonus_steps),
            aggr_steps_total_by_street=tuple(
                int(x) for x in aggr_steps_total_by_street
            ),
            aggr_bonus_steps_by_street=tuple(
                int(x) for x in aggr_bonus_steps_by_street
            ),
        )

    return _finalize_batch_arr(
        all_obs_arr,
        all_gm_arr,
        all_ga_arr,
        all_rc_arr,
        all_sz_arr,
        all_an_arr,
        all_ru_arr,
        all_oh_arr,
        all_lp_arr,
        all_v_arr,
        all_ret_arr,
        all_adv_arr,
        all_glp_arr,
        all_alp_arr,
        wcursor,
        device=device,
        aggr_bonus_total_bb=aggr_bonus_total_bb,
        aggr_steps_total=aggr_steps_total,
        aggr_bonus_steps=aggr_bonus_steps,
        aggr_steps_total_by_street=tuple(aggr_steps_total_by_street),
        aggr_bonus_steps_by_street=tuple(aggr_bonus_steps_by_street),
        adv_clip=float(getattr(train_config, "adv_clip", 0.0)),
        all_last_arr=all_last_arr,
    )


# ---- vThree: mix many (seats,stacks) configs within ONE update ------------
# Each update's gradient averages over N configs spanning all stack tiers,
# instead of one config + a 50-update block. This removes the consecutive-
# shallow exposure that saturated the gate (vTwo10-13 all died ~38 clubgg
# updates in). Implemented as a thin wrapper over the bit-exact single-config
# collector: split → host-concat → global advantage re-normalization.

_BATCH_TENSOR_FIELDS = (
    "obs", "gate_masks", "gate_actions", "raise_chips", "sizing",
    "anchor_actions", "refine_u", "opp_holes", "log_probs", "values",
    "returns", "advantages", "old_gate_logp", "old_anchor_logp",
)
# is_terminal is OPTIONAL (None on serial paths and hand-built test
# batches), so it is handled like ent_coef_rows — explicitly, not via the
# always-present tuple above.


def _batch_to_device(batch: Batch, device: torch.device) -> Batch:
    """Move every tensor field of a Batch to `device`; scalar diagnostics ride
    along unchanged. Used to evacuate each sub-rollout to host RAM before the
    next starts, so GPU peak stays at a single sub-rollout (not all N)."""
    moved = {f: getattr(batch, f).to(device) for f in _BATCH_TENSOR_FIELDS}
    if batch.ent_coef_rows is not None:
        moved["ent_coef_rows"] = batch.ent_coef_rows.to(device)
    if batch.is_terminal is not None:
        moved["is_terminal"] = batch.is_terminal.to(device)
    return replace(batch, **moved)


def _concat_batches(batches: list[Batch], adv_clip: float) -> Batch:
    """Concatenate sub-rollout Batches along the transition axis, then RE-normalize
    the combined advantages GLOBALLY (mean 0 / std 1, then the fat-tail clamp) so
    no single config's value scale dominates. Scalar aggression diagnostics sum.

    NOTE (2026-06-23): a per-config variant (preserve each sub-rollout's own
    unit-std; no global pool) was tried as a suspected full-LR collapse fix. It
    did NOT fix full LR AND it destabilized the low-LR run (the anchor spikes
    stopped settling — Ha 1.4->0.4 by u15 where global norm had held). So global
    renorm is retained; it was the more stable of the two."""
    if len(batches) == 1:
        return batches[0]

    def _cat(field: str) -> torch.Tensor:
        return torch.cat([getattr(b, field) for b in batches], dim=0)

    adv = _cat("advantages")
    adv = (adv - adv.mean()) / adv.std().clamp(min=1e-8)
    if adv_clip > 0.0:
        adv = adv.clamp(-adv_clip, adv_clip)

    def _sum3(field: str) -> tuple[int, int, int]:
        vals = [int(sum(getattr(b, field)[i] for b in batches)) for i in range(3)]
        return (vals[0], vals[1], vals[2])

    merged = {f: _cat(f) for f in _BATCH_TENSOR_FIELDS}
    merged["advantages"] = adv
    # Optional field: concatenate only when every sub carries it (batched
    # collectors always do; hand-built test batches may not).
    if all(b.is_terminal is not None for b in batches):
        merged["is_terminal"] = _cat("is_terminal")
    return Batch(
        **merged,
        aggr_bonus_total_bb=float(sum(b.aggr_bonus_total_bb for b in batches)),
        aggr_steps_total=int(sum(b.aggr_steps_total for b in batches)),
        aggr_bonus_steps=int(sum(b.aggr_bonus_steps for b in batches)),
        aggr_steps_total_by_street=_sum3("aggr_steps_total_by_street"),
        aggr_bonus_steps_by_street=_sum3("aggr_bonus_steps_by_street"),
    )


def collect_rollout_multiconfig(
    learner: ActorCritic,
    pool: OpponentPool,
    configs: list[GameConfig],
    train_config: TrainingConfig,
    rng: np.random.Generator,
    critic: CentralCritic | None = None,
    config_tiers: "list[str] | None" = None,
    tier_ent: "dict[str, float] | None" = None,
    _legacy_staging: bool = False,
) -> Batch:
    """One update's rollout MIXED across `configs` distinct (seats,stacks) setups.

    Runs `collect_rollout_batched` once per config — each sub-rollout sized
    `num_envs // N` envs and `rollout_length // N` learner steps.

    Staging (P7+P8, 2026-07-09): sub-rollouts write directly into per-sub
    VIEWS of one big preallocated host buffer (bases chained by each sub's
    actual row count, so the buffer holds exactly what torch.cat used to
    produce, in the same row order, with zero staging copies). Only the tiny
    per-sub advantage-normalization vector visits the learner device (it ran
    on CUDA in the legacy path — CPU math would drift f32 reduction order);
    the combined batch then ships to the device ONCE. This replaces the
    legacy pipeline (finalize each sub to CUDA -> evacuate to host ->
    torch.cat a second full-size host copy -> upload), which moved ~90GB of
    redundant PCIe traffic and doubled the host transient per update. The
    legacy path is retained behind `_legacy_staging=True` SOLELY as the
    bit-exactness reference for tests/python/test_multiconfig_staging.py —
    both paths must produce bitwise-identical Batches. GPU peak during
    collection drops to just the model forwards (no sub-batch residency).
    Pool snapshots are taken by the caller per-update, so calling the
    collector N times here does not over-snapshot.

    Per-tier machinery (V5_DESIGN.md B5, both optional and additive):
    `config_tiers` labels each config with its stack tier; with `tier_ent`
    it attaches per-row ABSOLUTE entropy coefs (`ent_coef_rows`) so each
    tier keeps its own coefficient inside the mixed update, and per-tier
    F/T/R counters (`tier_ftr`) so the aggression stop-loss telemetry
    survives the concat."""
    n = len(configs)
    if n == 0:
        raise ValueError("collect_rollout_multiconfig requires >= 1 config")
    if config_tiers is not None and len(config_tiers) != n:
        raise ValueError(
            f"config_tiers length {len(config_tiers)} != configs {n}"
        )
    device = next(learner.parameters()).device
    host = torch.device("cpu")
    sub_config = replace(
        train_config,
        num_envs=max(1, train_config.num_envs // n),
        rollout_length=max(1, train_config.rollout_length // n),
    )
    adv_clip = float(getattr(train_config, "adv_clip", 0.0))
    # P5: one frozen-opponent model cache for the WHOLE update — pool
    # membership is frozen across it (the caller snapshots after
    # trainer.update), so each member builds once instead of once per
    # sub-rollout. Dies with this call; bounded at pool capacity (~8 models,
    # ~0.5GB — the same worst case a single sub-rollout already reaches).
    # NOT the reverted cross-update opponent cache (that one persisted
    # across updates and grew).
    snapshot_cache: dict[int, ActorCritic] = {}
    # P3: reuse BatchedBombPotEnv across sub-rollouts that share
    # (n_envs, num_seats, variant). Stack samples at the same seat count
    # reconfigure in place; different seat counts get distinct engines.
    # Dies with this call (same lifetime as snapshot_cache).
    env_cache: dict[tuple, BatchedBombPotEnv] = {}

    if _legacy_staging:
        # Reference path (test-only): finalize each sub to the learner device,
        # evacuate to host, torch.cat, global renorm inside _concat_batches.
        host_batches: list[Batch] = []
        for cfg in configs:
            sub = collect_rollout_batched(
                learner, pool, cfg, sub_config, rng, critic=critic,
                snapshot_cache=snapshot_cache,
                env_cache=env_cache,
            )
            host_batches.append(_batch_to_device(sub, host))
            del sub
        combined = _concat_batches(host_batches, adv_clip)
        subs: list[Batch] = host_batches
    else:
        # P7+P8 shared staging: one big host buffer, per-sub views, chained
        # bases. Capacity per sub uses the collector's own formula (asserted
        # inside it); total = worst case of every sub filling to cap.
        n_sub_envs = sub_config.num_envs
        sub_cap = sub_config.rollout_length + n_sub_envs * POOL_SLACK_PER_ENV
        total_cap = n * sub_cap
        hole_count = configs[0].hole_count
        assert all(c.hole_count == hole_count for c in configs), (
            "mixed hole widths across mix-configs are unsupported"
        )
        obs_dim = OBS_DIM_NLH if configs[0].variant == "nlh_single" else OBS_DIM
        # Same pin gate as the collector's own slabs (default OFF — see the
        # pinning-tax comment there); pinning one big buffer instead of N
        # small ones is otherwise equivalent.
        _pin = (
            device.type == "cuda"
            and os.environ.get("PLO5BP_PIN_ROLLOUT", "0") == "1"
        )
        _pinned_keepalive: list[torch.Tensor] = []

        def _alloc(shape, np_dtype, torch_dtype):
            if _pin:
                t = torch.empty(shape, dtype=torch_dtype, pin_memory=True)
                _pinned_keepalive.append(t)
                return t.numpy()
            return np.empty(shape, dtype=np_dtype)

        big = {
            "obs": _alloc((total_cap, obs_dim), np.float32, torch.float32),
            "gm": _alloc((total_cap, GATE_ACTIONS), bool, torch.bool),
            "ga": _alloc(total_cap, np.int64, torch.int64),
            "rc": _alloc(total_cap, np.int64, torch.int64),
            "sz": _alloc((total_cap, 4), np.int64, torch.int64),
            "an": _alloc(total_cap, np.int64, torch.int64),
            "ru": _alloc(total_cap, np.float32, torch.float32),
            "oh": _alloc((total_cap, 5, hole_count), np.uint8, torch.uint8),
            "lp": _alloc(total_cap, np.float32, torch.float32),
            "glp": _alloc(total_cap, np.float32, torch.float32),
            "alp": _alloc(total_cap, np.float32, torch.float32),
            "v": _alloc(total_cap, np.float32, torch.float32),
            "ret": _alloc(total_cap, np.float32, torch.float32),
            "adv": _alloc(total_cap, np.float32, torch.float32),
            "last": _alloc(total_cap, bool, torch.bool),
        }
        subs = []
        base = 0
        for cfg in configs:
            views = {key: arr[base : base + sub_cap] for key, arr in big.items()}
            sub = collect_rollout_batched(
                learner, pool, cfg, sub_config, rng,
                critic=critic, out_slabs=views,
                snapshot_cache=snapshot_cache,
                env_cache=env_cache,
            )
            rows = sub.obs.shape[0]
            # Chain the next sub's base to this sub's actual row count so the
            # buffer's first `total` rows reproduce torch.cat's layout exactly.
            assert rows <= sub_cap and base + rows <= total_cap, (
                f"shared-staging overflow: base={base} rows={rows} "
                f"sub_cap={sub_cap} total_cap={total_cap}"
            )
            subs.append(sub)
            base += rows
        total = base

        def _t(key: str) -> torch.Tensor:
            return torch.from_numpy(big[key][:total])

        adv = _t("adv")
        if n > 1:
            # Global advantage re-normalization — the identical ops
            # _concat_batches applies (and, like it, skipped when there is
            # only one sub-rollout).
            adv = (adv - adv.mean()) / adv.std().clamp(min=1e-8)
            if adv_clip > 0.0:
                adv = adv.clamp(-adv_clip, adv_clip)

        def _sum3(field: str) -> tuple[int, int, int]:
            vals = [int(sum(getattr(b, field)[i] for b in subs)) for i in range(3)]
            return (vals[0], vals[1], vals[2])

        combined = Batch(
            obs=_t("obs"),
            gate_masks=_t("gm"),
            gate_actions=_t("ga"),
            raise_chips=_t("rc"),
            sizing=_t("sz"),
            anchor_actions=_t("an"),
            refine_u=_t("ru"),
            opp_holes=_t("oh"),
            log_probs=_t("lp"),
            values=_t("v"),
            returns=_t("ret"),
            advantages=adv,
            old_gate_logp=_t("glp"),
            old_anchor_logp=_t("alp"),
            is_terminal=_t("last"),
            aggr_bonus_total_bb=float(sum(b.aggr_bonus_total_bb for b in subs)),
            aggr_steps_total=int(sum(b.aggr_steps_total for b in subs)),
            aggr_bonus_steps=int(sum(b.aggr_bonus_steps for b in subs)),
            aggr_steps_total_by_street=_sum3("aggr_steps_total_by_street"),
            aggr_bonus_steps_by_street=_sum3("aggr_bonus_steps_by_street"),
        )

    if config_tiers is not None:
        if tier_ent is not None:
            combined.ent_coef_rows = torch.cat([
                torch.full(
                    (b.obs.shape[0],),
                    float(tier_ent[t]),
                    dtype=torch.float32,
                )
                for b, t in zip(subs, config_tiers)
            ])
        ftr: dict[str, tuple[list[int], list[int]]] = {}
        for b, t in zip(subs, config_tiers):
            bonus, steps = ftr.setdefault(t, ([0, 0, 0], [0, 0, 0]))
            for s in range(3):
                bonus[s] += int(b.aggr_bonus_steps_by_street[s])
                steps[s] += int(b.aggr_steps_total_by_street[s])
        combined.tier_ftr = {
            t: (tuple(v[0]), tuple(v[1])) for t, v in ftr.items()
        }
    return _batch_to_device(combined, device)


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
            sizing=batch.sizing[sel],
            anchor_actions=batch.anchor_actions[sel],
            refine_u=batch.refine_u[sel],
            opp_holes=batch.opp_holes[sel],
            log_probs=batch.log_probs[sel],
            values=batch.values[sel],
            returns=batch.returns[sel],
            advantages=batch.advantages[sel],
            old_gate_logp=batch.old_gate_logp[sel],
            old_anchor_logp=batch.old_anchor_logp[sel],
            is_terminal=(
                batch.is_terminal[sel]
                if batch.is_terminal is not None
                else None
            ),
            ent_coef_rows=(
                batch.ent_coef_rows[sel]
                if batch.ent_coef_rows is not None
                else None
            ),
        )
