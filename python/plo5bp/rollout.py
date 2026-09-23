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

import copy
import os
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterable

import numpy as np
import torch
from torch.profiler import record_function

from plo5bp._engine import compute_aggression_bonus_batch  # type: ignore[attr-defined]
from plo5bp.actions import ALL_IN, GATE_ACTIONS, GATE_CHECK_CALL, GATE_RAISE
from plo5bp.compact_obs import (
    RUST_PACKER_AVAILABLE,
    CompactObsLayout,
    PackedObs,
    layout_for,
    pack_rows_into,
)
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM, OBS_DIM_MINIMAL
from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.env import BombPotEnv
from plo5bp.env_batched import BatchedBombPotEnv
from plo5bp.network import (
    ActorCritic,
    ActOut,
    CentralCritic,
    build_actor_from_state_dict,
    opp_holes_multihot,
)
from plo5bp.selfplay import OpponentPool
from plo5bp.sizing import sizing_from_info

# Slack rows per env appended to the obs-pool / output-slab capacity beyond
# rollout_target. What lands past the target (review 2026-09-20 A6 — the old
# note here claimed in-flight hands "flush to completion"; they did NOT, they
# were dropped):
#   - drain_inflight ON (default): every hand in flight when the target is
#     reached is played out and flushed, so the batch ends roughly
#     n_envs x (rows of one length-biased hand) past the target;
#   - drain_inflight OFF (legacy): only the last flush wave's overshoot lands
#     in the output slabs, but the obs POOL still holds the unflushed rows of
#     the abandoned in-flight hands.
# The slack is only the INITIAL capacity: pool and slabs grow on demand
# (`_slack_per_env` learns the real need so growth stays rare), so an
# undersized guess costs one copy, never a crash or a dropped row.
# Module-level (P7+P8): collect_rollout_batched AND
# collect_rollout_multiconfig's shared-staging allocation size from the same
# helper.

POOL_SLACK_PER_ENV = 32

# Largest rows-per-env actually needed beyond rollout_target by any collection
# in this process (monotone max). A SIZING HINT ONLY — it changes buffer
# capacities, never a collected value — so later collections start right-sized
# instead of re-paying a grow copy every update.
_observed_slack_per_env = 0

# A hand that is already over AT DEAL (every seat all-in on the ante/blinds —
# nobody can act) yields no decision; it is re-dealt with a fresh seed. After
# this many consecutive done-at-deal re-deals of one env the config itself
# cannot produce a live hand and the collectors raise instead of spinning
# (review 2026-09-20 A1: the batched loop used to spin forever there).
_MAX_REDEALS = 64


def _slack_per_env() -> int:
    seen = _observed_slack_per_env
    return max(POOL_SLACK_PER_ENV, seen + seen // 4 + 4)


def _note_slack_used(rows: int, rollout_target: int, n_envs: int) -> None:
    global _observed_slack_per_env
    extra = -(-max(0, int(rows) - int(rollout_target)) // max(1, int(n_envs)))
    if extra > _observed_slack_per_env:
        _observed_slack_per_env = extra


def _resolve_drain_inflight(
    train_config: TrainingConfig, override: "bool | None"
) -> bool:
    """Whether hands still in flight when the row target is reached are played
    out and flushed (True, the default) or dropped (False = the pre-2026-09-20
    behavior, byte-identical). An explicit collector kwarg wins; else the
    TrainingConfig field; else True. Read via getattr so configs that predate
    the field (old checkpoints' stamped dicts, hand-built test configs) work."""
    if override is not None:
        return bool(override)
    return bool(getattr(train_config, "drain_inflight", True))


def _dead_config_error(where: str, game_config: GameConfig, n_dead: int) -> RuntimeError:
    return RuntimeError(
        f"{where}: {n_dead} env(s) were still terminal AT DEAL after "
        f"{_MAX_REDEALS} re-deals — this GameConfig cannot produce a hand with "
        "a decision (fewer than two seats can act after posting the ante/"
        f"blinds): {game_config!r}. Resample the config (scripts/train.py "
        "_sample_game_config does) instead of collecting from it."
    )



try:  # Unix: kernel CPU time + page faults per step-timer region
    import resource as _resource
except ImportError:  # Windows
    _resource = None


def _rusage() -> tuple[float, int]:
    """(system-CPU seconds, minor page faults) of the whole process so far —
    all threads. (0.0, 0) where `resource` is unavailable (Windows)."""
    if _resource is None:
        return 0.0, 0
    r = _resource.getrusage(_resource.RUSAGE_SELF)
    return r.ru_stime, r.ru_minflt


class _TimedRF:
    """record_function + optional wall timer (when PLO5BP_STEP_TIMERS=1)."""

    __slots__ = ("_name", "_rf", "_t0", "_ru0", "_enabled")

    def __init__(self, name: str) -> None:
        self._name = name
        self._rf = record_function(name)
        self._enabled = os.environ.get("PLO5BP_STEP_TIMERS", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        self._t0 = 0.0

    def __enter__(self):
        self._rf.__enter__()
        if self._enabled:
            self._ru0 = _rusage()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        global _ACTIVE_STEP_TIMERS
        if self._enabled and _ACTIVE_STEP_TIMERS is not None:
            _ACTIVE_STEP_TIMERS._add(
                self._name, time.perf_counter() - self._t0, self._ru0
            )
        return self._rf.__exit__(*exc)


_ACTIVE_STEP_TIMERS: _StepTimers | None = None


class _StepTimers:
    """Cheap wall-clock accumulators for rollout sub-steps.

    Enabled when env ``PLO5BP_STEP_TIMERS=1`` (or truthy). Uses perf_counter
    around named regions; zero overhead when disabled. Safe under the
    single-threaded rollout loop (no locks). Each region also accumulates the
    process's kernel CPU time and minor page faults spent inside it (Unix;
    `_rusage`) — a region that stalls in the kernel (page faults, memory
    compaction) shows up in those columns, not just in wall time.
    """

    __slots__ = ("enabled", "totals", "counts", "sys", "faults", "_stack")

    def __init__(self) -> None:
        self.enabled = os.environ.get("PLO5BP_STEP_TIMERS", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.sys: dict[str, float] = {}
        self.faults: dict[str, int] = {}
        self._stack: list[tuple[str, float, tuple[float, int]]] = []

    def _add(self, name: str, dt: float, ru0: tuple[float, int]) -> None:
        sys1, flt1 = _rusage()
        self.totals[name] = self.totals.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1
        self.sys[name] = self.sys.get(name, 0.0) + (sys1 - ru0[0])
        self.faults[name] = self.faults.get(name, 0) + (flt1 - ru0[1])

    def begin(self, name: str) -> None:
        if self.enabled:
            self._stack.append((name, time.perf_counter(), _rusage()))

    def end(self) -> None:
        if not self.enabled or not self._stack:
            return
        name, t0, ru0 = self._stack.pop()
        self._add(name, time.perf_counter() - t0, ru0)

    def report(self, label: str = "rollout") -> None:
        if not self.enabled or not self.totals:
            return
        grand = sum(self.totals.values())
        print(f"\n===== step timers ({label}) total_accounted={grand:.1f}s =====")
        rows = sorted(self.totals.items(), key=lambda kv: -kv[1])
        print(
            f"  {'name':36s}  {'sec':>10s}  {'%':>6s}  {'n':>8s}  {'ms/call':>8s}"
            f"  {'sys_s':>7s}  {'faults_k':>8s}"
        )
        for name, sec in rows:
            n = self.counts.get(name, 0)
            pct = 100.0 * sec / grand if grand > 0 else 0.0
            mspc = 1000.0 * sec / n if n else 0.0
            print(
                f"  {name:36s}  {sec:10.1f}  {pct:5.1f}%  {n:8d}  {mspc:8.2f}"
                f"  {self.sys.get(name, 0.0):7.1f}  {self.faults.get(name, 0) / 1e3:8.1f}"
            )
        print(f"[step-timers] accounted={grand:.1f}s across {sum(self.counts.values())} calls")




class _PinnedStepH2D:
    """Long-lived pinned host staging for per-step act() H2D (CUDA only).

    Capacity is ``num_envs`` (max group size) per slot. Default ``n_slots=2``
    double-buffers so consecutive opp groups can H2D into alternate slots
    without waiting for the previous group's H2D (attack #1 double-pin).

    Safety rules:
    - **Learner path** uses slot 0 and ends with blocking D2H — that drains
      the default stream before the next learner upload, so slot 0 is free.
    - **Opp multi-group path** alternates slots 0/1. Before refilling a slot
      the host calls ``wait_slot(s)`` which ``event.synchronize()``s only that
      slot's prior H2D (not a full device sync). With 2 slots, group *i* only
      waits on group *i-2*'s H2D — usually already done while *i-1* ran.
    - Never overwrite a slot until its H2D event has completed on the host.

    Distinct from ``PLO5BP_PIN_ROLLOUT`` (multi-GB finalize slabs — default
    OFF). Pin cost is O(n_slots * num_envs * obs_dim) once per sub-rollout.
    """

    __slots__ = (
        "enabled", "device", "cap", "n_slots",
        "obs_h", "gm_h", "sz_h", "_h2d_events",
    )

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        device: torch.device,
        n_slots: int = 2,
    ) -> None:
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.cap = int(capacity)
        self.n_slots = max(1, int(n_slots))
        self.enabled = self.device.type == "cuda" and self.cap > 0
        self._h2d_events: list = [None] * self.n_slots
        if not self.enabled:
            self.obs_h = self.gm_h = self.sz_h = None  # type: ignore[assignment]
            return
        # Shape (n_slots, cap, ...) — one pin bank per slot.
        self.obs_h = torch.empty(
            (self.n_slots, self.cap, obs_dim),
            dtype=torch.float32,
            pin_memory=True,
        )
        self.gm_h = torch.empty(
            (self.n_slots, self.cap, GATE_ACTIONS),
            dtype=torch.bool,
            pin_memory=True,
        )
        self.sz_h = torch.empty(
            (self.n_slots, self.cap, 4),
            dtype=torch.int64,
            pin_memory=True,
        )

    def wait_slot(self, slot: int = 0) -> None:
        """Block host until this slot's last H2D has finished (safe to refill)."""
        if not self.enabled:
            return
        s = int(slot) % self.n_slots
        ev = self._h2d_events[s]
        if ev is not None:
            ev.synchronize()

    def upload(
        self,
        b_obs: np.ndarray,
        b_gm: np.ndarray,
        b_sizing: np.ndarray,
        slot: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Host arrays (k, ...) -> device via pinned slot. Values match from_numpy.to.

        Caller must ``wait_slot(slot)`` before refill if that slot may still
        have an in-flight H2D (opp multi-group path). Learner path relies on
        post-act blocking D2H instead.
        """
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
        s = int(slot) % self.n_slots
        obs_np = np.ascontiguousarray(b_obs, dtype=np.float32)
        gm_np = np.ascontiguousarray(b_gm, dtype=bool)
        sz_np = np.ascontiguousarray(b_sizing, dtype=np.int64)
        self.obs_h[s].numpy()[:k] = obs_np
        self.gm_h[s].numpy()[:k] = gm_np
        self.sz_h[s].numpy()[:k] = sz_np
        o_t = self.obs_h[s, :k].to(self.device, non_blocking=True)
        m_t = self.gm_h[s, :k].to(self.device, non_blocking=True)
        b_t = self.sz_h[s, :k].to(self.device, non_blocking=True)
        if self._h2d_events[s] is None:
            self._h2d_events[s] = torch.cuda.Event()
        self._h2d_events[s].record()
        return o_t, m_t, b_t




@dataclass
class Batch:
    # (T, OBS_DIM) f32 — or its compact storage (compact_obs.PackedObs), whose
    # row indexing (obs[rows], as iter_minibatches does) yields dense f32 rows.
    obs: "torch.Tensor | PackedObs"
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
    """(num_seats, hole_w) u8 per-hand hole cache from
    `env.all_hole_cards()`, at the VARIANT's hole width (NLH 2 / PLO4 4 /
    PLO5 5 / PLO6 6) — the same compact layout the batched collector stores
    (`all_hole_cards_batch`). The critic's input width is fixed by
    `opp_holes_multihot` (5 slots x 52), not by this array.

    (review 2026-09-20 A16) This used to hard-code width 5: PLO6 raised on
    the 6-card assignment, and NLH/PLO4 rows were right-padded with 255,
    which the old multihot then scattered over a genuine card 51. PLO5 rows
    are unchanged. A seat with no cards (not dealt in) stays all-255."""
    holes = env.all_hole_cards()
    out = np.full((len(holes), env.config.hole_count), 255, dtype=np.uint8)
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
    obs_host: "torch.Tensor | PackedObs",
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
    slab. No `np.stack` over millions of small arrays. `obs_host` is the
    first `wcursor` stored observations, already a host view (dense tensor
    or compact `PackedObs` — see `_obs_from_slabs`).
    """
    # When the source slabs are pinned (CUDA path) `non_blocking=True`
    # lets the H2D copies queue against the default stream and overlap
    # downstream compute; on CPU device the flag is a no-op.
    with _TimedRF("step11/finalize_h2d"):
        obs_t = obs_host.to(device, non_blocking=True)
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
    drain_inflight: "bool | None" = None,
) -> Batch:
    """Serial rollout driver. Each env has a separate `BombPotEnv`; the
    learner batch-forwards over all learner-acting envs per step and
    frozen opponents forward one-by-one.

    With `critic` provided, GAE values come from the centralized critic
    (which sees all hole cards); otherwise the actor's own value head is
    used (v1 behavior / profiling fallback).

    `drain_inflight` (None = `train_config.drain_inflight`, default True):
    see `_resolve_drain_inflight` / the batched collector's docstring.
    """
    n_envs = train_config.num_envs
    drain = _resolve_drain_inflight(train_config, drain_inflight)
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

    envs = [BombPotEnv(
                game_config,
                ev_runout_samples=train_config.ev_runout_samples,
                obs_mode=str(getattr(train_config, "obs_mode", "full")),
            )
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
    # centralized critic and the stored opp_holes blocks. Variant hole
    # width (review 2026-09-20 A16; was a hard-coded 5).
    hole_caches: list[np.ndarray] = [
        np.full((n_seats, game_config.hole_count), 255, dtype=np.uint8)
        for _ in range(n_envs)
    ]

    def _deal(env_idx: int):
        """Deal env `env_idx` a fresh hand. A hand that is already terminal
        AT DEAL (every seat all-in on the ante/blinds) has no decision to
        collect: re-deal with a fresh seed, bounded, then raise — this
        driver used to die on `assert opp is not None` there, and the
        batched one spun forever (review 2026-09-20 A1). The first draw is
        the pre-fix (seed, button) pair, so live configs are byte-identical."""
        env = envs[env_idx]
        for _ in range(_MAX_REDEALS + 1):
            seed = int(rng.integers(0, 2**63 - 1))
            button = int(rng.integers(0, n_seats))
            o, info = env.reset(seed, button)
            if not env.is_terminal():
                return o, info
        raise _dead_config_error("collect_rollout", game_config, 1)

    for i, env in enumerate(envs):
        _assign_pool_mix(i)
        o, info = _deal(i)
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

    # PRODUCTION BEHAVIOR CHANGE (review 2026-09-20 A6) — drain_inflight:
    # once the row target is reached, finished envs are no longer re-dealt
    # (`live[i]` goes False) and the loop keeps stepping until every hand
    # still in flight has finished and flushed. The old loop exited at the
    # target and DROPPED those hands, under-sampling long hands by ~len/W.
    # With drain off `live` stays all-True: byte-identical to before.
    live = [True] * n_envs
    while len(all_obs) < train_config.rollout_length or (drain and any(live)):
        learner_idx: list[int] = []
        opp_idx: list[int] = []
        for i in range(n_envs):
            if not live[i]:
                continue
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
            if live[i]:
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
            if not live[i]:
                continue
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
                if drain and len(all_obs) >= train_config.rollout_length:
                    # Target reached: this env deals no further hand (A6).
                    live[i] = False
                    continue
                _assign_pool_mix(i)
                next_obs, next_info = _deal(i)
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


# Output-slab layout — ONE definition for collect_rollout_batched's own slabs
# and collect_rollout_multiconfig's shared staging buffer (they used to carry
# two hand-synced 15-line allocation lists). Order is irrelevant; keys are the
# `out_slabs` contract. The observation is ONE dense float32 slab ("obs") or,
# with compact storage (compact_obs.py), a bit slab plus a verbatim real-column
# slab ("obs_bits", "obs_real"); `_slab_keys(layout)` is the full key set.
# Initial per-(env, seat) trajectory capacity of the batched collector; it
# doubles on demand up to MAX_STEPS_PER_SEAT (see `_grow_traj` there).
_TRAJ_CAP_INIT = 32

# Per-(env, seat, slot) trajectory arrays of the batched collector:
# (name, per-slot tail shape, dtype, initial fill). `q_taken`/`vpi` exist only
# for the VRPO estimator.
_TRAJ_SPEC = (
    ("obs_idx", (), np.int64, 0),   # absolute index into the step obs/gm pool
    ("gate", (), np.int8, 0),
    ("chips", (), np.int64, 0),
    ("sizing", (4,), np.int64, 0),
    ("anchor", (), np.int8, -1),
    ("u", (), np.float32, 0),
    ("log_p", (), np.float32, 0),
    ("gate_lp", (), np.float32, 0),
    ("anchor_lp", (), np.float32, 0),
    ("value", (), np.float32, 0),
    ("q_taken", (), np.float32, 0),  # VRPO: Q(s_t, a_t)
    ("vpi", (), np.float32, 0),      # VRPO: V^pi(s_t) = sum_a pi(a) Q(s_t, a)
    ("costs", (), np.float32, 0),    # per-step cost / pot / street
    ("pots", (), np.float32, 0),
    ("streets", (), np.int8, 0),
)
_TRAJ_VRPO_ONLY = frozenset({"q_taken", "vpi"})


def _alloc_traj(
    n_envs: int, n_seats: int, cap: int, use_vrpo: bool
) -> dict[str, "np.ndarray | None"]:
    out: dict[str, np.ndarray | None] = {}
    for name, tail, dtype, fill in _TRAJ_SPEC:
        if name in _TRAJ_VRPO_ONLY and not use_vrpo:
            out[name] = None
            continue
        shape = (n_envs, n_seats, cap) + tail
        out[name] = np.full(shape, fill, dtype=dtype) if fill else np.zeros(shape, dtype=dtype)
    return out


# Rollout buffers REUSED across sub-rollouts and updates (2026-09-23). The
# batched collector used to allocate its trajectory arrays and per-step
# observation pool afresh for every sub-rollout (30 per update) and
# multiconfig its whole shared staging buffer every update — at vMin1's 44M
# rows, ~50-65 GB of freshly faulted, kernel-zeroed memory per update, and on
# the first RunPod host (fragmented memory) single updates stalled at 4x the
# time of their neighbors in exactly those allocations. Reuse is EXACT: every
# read of a trajectory slot / pool row / staging row is confined to what the
# CURRENT collection wrote (flush masks, `_vrpo_advantage_scan`, `[:wcursor]`
# views), and every buffer is created zero-filled, so a stale value is always
# finite. Pinned by tests/python/test_buffer_reuse.py.
_TRAJ_BUFFERS: dict[tuple, dict] = {}      # (n_envs, n_seats, vrpo) -> {"cap", <name>: array}
_OBS_POOL_BUFFERS: dict[tuple, dict] = {}  # (obs_dim, layout) -> {"cap", "obs", "gm"}
_STAGING_BUFFERS: dict[tuple, tuple] = {}  # (obs_dim, hole, pin, layout) -> (allocator, big, cap)


def _clear_rollout_buffers() -> None:
    """Drop every reused rollout buffer (tests; frees the memory)."""
    _TRAJ_BUFFERS.clear()
    _OBS_POOL_BUFFERS.clear()
    _STAGING_BUFFERS.clear()


def _reuse_staging(device: torch.device) -> bool:
    """Multiconfig keeps its big host staging buffer across updates only where
    the finished batch is COPIED off it (CUDA learner): on a CPU learner the
    returned Batch tensors are views of the buffer itself, which the next
    collection would overwrite under a caller still holding the batch."""
    return device.type == "cuda"

_SLAB_KEYS = (
    "gm", "ga", "rc", "sz", "an", "ru", "oh",
    "lp", "glp", "alp", "v", "ret", "adv", "last",
)


def _obs_spec(
    obs_dim: int, layout: CompactObsLayout | None
) -> dict[str, tuple[tuple[int, ...], type, torch.dtype]]:
    """Per-row shape + numpy/torch dtypes of the stored observation slab(s)."""
    if layout is None:
        return {"obs": ((int(obs_dim),), np.float32, torch.float32)}
    assert layout.obs_dim == int(obs_dim), (layout.obs_dim, obs_dim)
    return {
        "obs_bits": ((layout.n_bytes,), np.uint8, torch.uint8),
        "obs_real": ((layout.n_real,), np.float32, torch.float32),
    }


def _slab_keys(layout: CompactObsLayout | None) -> tuple[str, ...]:
    return (("obs",) if layout is None else ("obs_bits", "obs_real")) + _SLAB_KEYS


def _alloc_obs_pool(
    cap: int, obs_dim: int, layout: CompactObsLayout | None
) -> dict[str, np.ndarray]:
    """The per-step observation pool (rows appended at decision time, gathered
    into the output slabs at hand end), in the slabs' observation layout."""
    return {
        key: np.empty((int(cap), *shape), dtype=np_dtype)
        for key, (shape, np_dtype, _td) in _obs_spec(obs_dim, layout).items()
    }


def _obs_from_slabs(
    slabs: dict[str, np.ndarray], rows: int, layout: CompactObsLayout | None
) -> "torch.Tensor | PackedObs":
    """Host view (no copy) of the first `rows` stored observations."""
    if layout is None:
        return torch.from_numpy(slabs["obs"][:rows])
    return PackedObs(
        torch.from_numpy(slabs["obs_bits"][:rows]),
        torch.from_numpy(slabs["obs_real"][:rows]),
        layout,
    )


_WARNED_DENSE_FALLBACK = False


def _resolve_obs_layout(
    train_config: TrainingConfig, variant: str
) -> CompactObsLayout | None:
    """Compact observation storage for a collection (compact_obs.py), or None
    = dense float32 rows. ONE resolver for the collector's own pool/slabs and
    multiconfig's shared staging, so the two always agree. Storage only: the
    rows unpack bit-exactly, so batches, RNG and training are unchanged."""
    global _WARNED_DENSE_FALLBACK
    if not bool(getattr(train_config, "compact_obs", True)):
        return None
    layout = layout_for(variant, str(getattr(train_config, "obs_mode", "full")))
    if layout is not None and not RUST_PACKER_AVAILABLE:
        if not _WARNED_DENSE_FALLBACK:
            print(
                "[obs-storage] WARNING: this engine build has no pack_obs_rows "
                "(built before compact storage) -- storing observations DENSE. "
                "Rebuild: .venv/Scripts/maturin develop --release",
                flush=True,
            )
            _WARNED_DENSE_FALLBACK = True
        return None
    return layout


class _SlabAllocator:
    """Allocates / grows a `_slab_keys(layout)` dict of host output slabs, optionally
    in pinned memory (PLO5BP_PIN_ROLLOUT — see the pinning-tax note in
    collect_rollout_batched). `grow` is the A6 replacement for the old hard
    overflow error: a bigger set + one copy of the rows already written."""

    def __init__(
        self,
        obs_dim: int,
        hole_count: int,
        pin: bool,
        layout: CompactObsLayout | None = None,
    ) -> None:
        self._pin = bool(pin)
        # Pinned tensors own the memory behind their numpy views.
        self._keepalive: list[torch.Tensor] = []
        f32, i64 = (np.float32, torch.float32), (np.int64, torch.int64)
        self._spec: dict[str, tuple[tuple[int, ...], type, torch.dtype]] = {
            **_obs_spec(obs_dim, layout),
            "gm": ((GATE_ACTIONS,), bool, torch.bool),
            "ga": ((), *i64),
            "rc": ((), *i64),
            "sz": ((4,), *i64),
            "an": ((), *i64),
            "ru": ((), *f32),
            "oh": ((5, int(hole_count)), np.uint8, torch.uint8),
            "lp": ((), *f32),
            "glp": ((), *f32),
            "alp": ((), *f32),
            "v": ((), *f32),
            "ret": ((), *f32),
            "adv": ((), *f32),
            "last": ((), bool, torch.bool),
        }
        self.keys = _slab_keys(layout)
        assert tuple(self._spec) == self.keys

    def alloc(self, cap: int) -> dict[str, np.ndarray]:
        self._keepalive = []
        out: dict[str, np.ndarray] = {}
        for key, (tail, np_dtype, torch_dtype) in self._spec.items():
            shape = (int(cap), *tail)
            if self._pin:
                t = torch.empty(shape, dtype=torch_dtype, pin_memory=True)
                self._keepalive.append(t)
                out[key] = t.numpy()
            else:
                out[key] = np.empty(shape, dtype=np_dtype)
        return out

    def grow(
        self, slabs: dict[str, np.ndarray], used_rows: int, new_cap: int
    ) -> dict[str, np.ndarray]:
        old_keepalive = self._keepalive
        new = self.alloc(new_cap)
        for key in self.keys:
            new[key][:used_rows] = slabs[key][:used_rows]
        del old_keepalive
        return new


def _draw_pool_mix(
    rng: np.random.Generator,
    k: int,
    n_seats: int,
    pool_size: int,
    pool_opp_seats: int,
    pool_mix_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Opponent assignment for `k` hands about to be dealt (batched collector):
    returns (pool snapshot index per hand, -1 = pure self-play; (k, n_seats)
    learner-seat mask).

    Per hand this is the distribution the old per-env loop drew: with
    probability `pool_mix_prob`, `pool_opp_seats` distinct seats chosen
    uniformly go to ONE uniformly drawn pool snapshot and the rest stay learner
    seats; otherwise every seat is a learner seat (pool empty / 0 opp seats /
    prob <= 0: always self-play, no draws). PRODUCTION BEHAVIOR CHANGE
    (2026-09-23), RNG stream only: the draws are batched (a few numpy calls per
    step instead of ~3 per finished hand — the loop cost 4-10 ms/step at the
    vMin1 table count), so the sequence of deals/assignments differs from the
    pre-change collector; the distribution is identical. The uniform subset is
    the first `pool_opp_seats` seats of a uniformly random permutation (argsort
    of iid uniform keys)."""
    snap = np.full(k, -1, dtype=np.int64)
    mask = np.ones((k, n_seats), dtype=bool)
    if k == 0 or pool_size == 0 or pool_opp_seats == 0 or pool_mix_prob <= 0.0:
        return snap, mask
    mixed = np.nonzero(rng.random(k) < pool_mix_prob)[0]
    if mixed.size:
        snap[mixed] = rng.integers(0, pool_size, size=mixed.size)
        opp = np.argsort(rng.random((mixed.size, n_seats)), axis=1)[:, :pool_opp_seats]
        mask[mixed[:, None], opp] = False
    return snap, mask


class _StackedOpponents:
    """Every pool snapshot's actor as ONE batched call per step (2026-09-23).

    The batched collector used to call each snapshot's `act()` separately every
    step — up to `opponent_pool_size` (8) calls. With a small network each call
    is almost pure GPU launch overhead (~200 small kernels, most of them in the
    sampling tail), so a step paid it once per snapshot. Pool membership is
    frozen for the whole update, so the snapshots' weights are stacked once per
    sub-rollout and each step runs ONE `torch.func.vmap`'d forward (a batched
    matmul per layer; each snapshot sees only its own rows, padded to the
    largest group) and ONE `_act_from_heads` sampling pass over all opponent
    rows.

    PRODUCTION BEHAVIOR CHANGE (RNG stream only): batched matmuls round
    differently from per-snapshot ones (~1e-6 on the logits) and the single
    sampling pass draws the RNG in a different order, so trajectories are not
    bit-identical to the per-snapshot path — but every opponent row still
    samples its OWN snapshot's policy, and opponent rows are never trained on.
    `--no-batched-opponents` restores the per-snapshot calls."""

    def __init__(self, models: list) -> None:
        self.template = models[0]
        self.n = len(models)
        self.params, self.buffers = torch.func.stack_module_state(models)
        base = copy.deepcopy(models[0]).to("meta")  # structure only

        def _forward(params, buffers, obs, gate_mask):
            return torch.func.functional_call(base, (params, buffers), (obs, gate_mask))

        self._vforward = torch.func.vmap(_forward)

    @staticmethod
    def supported(models: list) -> bool:
        """One stack needs the same class, identical parameter/buffer shapes
        and the same sampling constants in every snapshot — always true within
        a run; a pool seeded from differently-shaped checkpoints (or a v1
        pool, which has no `_act_from_heads`) keeps per-snapshot calls."""
        if not models or not hasattr(models[0], "_act_from_heads"):
            return False
        m0 = models[0]
        shapes = {k: tuple(v.shape) for k, v in m0.state_dict().items()}
        consts = ("anchor_spec", "_size_floor", "_size_span", "_mix_floor", "_mixture_k")
        for m in models[1:]:
            if type(m) is not type(m0):
                return False
            if {k: tuple(v.shape) for k, v in m.state_dict().items()} != shapes:
                return False
            if any(getattr(m, c, None) != getattr(m0, c, None) for c in consts):
                return False
        return True

    def act(
        self,
        obs: torch.Tensor,
        gate_mask: torch.Tensor,
        sizing: torch.Tensor,
        slot_g: torch.Tensor,
        slot_j: torch.Tensor,
        n_max: int,
        deterministic: bool = False,
    ) -> ActOut:
        """Sample (or argmax, `deterministic`) the opponent rows `obs` /
        `gate_mask` / `sizing` (device tensors); row i belongs to snapshot
        `slot_g[i]` at padded position `slot_j[i]` (< `n_max`). Padding rows
        see an all-False gate mask and are dropped before sampling."""
        obs_pad = obs.new_zeros((self.n, n_max, obs.shape[-1]))
        obs_pad[slot_g, slot_j] = obs
        gm_pad = gate_mask.new_zeros((self.n, n_max, gate_mask.shape[-1]))
        gm_pad[slot_g, slot_j] = gate_mask
        heads = self._vforward(self.params, self.buffers, obs_pad, gm_pad)
        return self.template._act_from_heads(
            *(h[slot_g, slot_j] for h in heads), sizing, deterministic=deterministic
        )


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
    drain_inflight: "bool | None" = None,
    out_slabs_grow: "Callable[[int, int], dict[str, np.ndarray]] | None" = None,
) -> Batch:
    """Batched rollout using `BatchedBombPotEnv` + snapshot-bucket
    opponent forwards. Drives all envs through `apply_hybrid_batch`.

    `drain_inflight` (review 2026-09-20 A6; None = `train_config
    .drain_inflight`, default True — PRODUCTION BEHAVIOR CHANGE): once
    `rollout_length` rows are flushed, finished envs are NOT re-dealt and the
    loop keeps stepping until every hand still in flight has finished and
    flushed, so the batch is "every hand STARTED before the target was
    reached" (a stopping-time sample — unbiased in hand length). False = the
    legacy exit-at-target, which dropped the in-flight hands (the one in
    progress at a cut is length-biased long, so long hands were under-sampled
    by ~len/W: ~7.5% of started hands at the production ratio, a 40-action
    pot sampled ~15-19% less than a 6-action one) — row count, order and RNG
    consumption are byte-identical to the pre-fix collector. With drain on
    the batch is LARGER than `rollout_length` by about n_envs x (rows of one
    in-flight hand); size `--rollout-length` / GPU memory accordingly.

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
    self-allocating, finalize-to-device path, byte-identical to before.
    The views' length IS the capacity; `out_slabs_grow(used_rows, min_rows)`
    (optional) must return replacement views of >= `min_rows` rows whose
    first `used_rows` rows are preserved — called when a flush would
    overflow. Without it an overflow raises, as before."""
    n_envs = train_config.num_envs
    drain = _resolve_drain_inflight(train_config, drain_inflight)
    global _ACTIVE_STEP_TIMERS
    if _ACTIVE_STEP_TIMERS is not None:
        step_timers = _ACTIVE_STEP_TIMERS
    else:
        step_timers = _StepTimers()
        _ACTIVE_STEP_TIMERS = step_timers
    # Wall time from here to the first loop iteration: env build/reuse, the
    # per-sub trajectory arrays + obs pool allocations, the first deal/refresh.
    step_timers.begin("step0/setup")
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
        cache_key = (
            n_envs, game_config.num_seats, game_config.variant,
            str(getattr(train_config, "obs_mode", "full")),
        )
        if env_cache is not None and cache_key in env_cache:
            env = env_cache[cache_key]
            env.reconfigure(game_config)
            env._ev_runout_samples = int(train_config.ev_runout_samples)
        else:
            _obs_mode = str(getattr(train_config, "obs_mode", "full"))
            # Minimal obs drops opp-outcome features; opp_mc=0 skips MC entirely.
            _opp_mc = 0 if _obs_mode == "minimal" else TRAIN_OPP_OUTCOME_MC
            env = BatchedBombPotEnv(
                n_envs,
                game_config,
                ev_runout_samples=train_config.ev_runout_samples,
                opp_outcome_mc=_opp_mc,
                obs_mode=_obs_mode,
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

    # (n_envs, n_seats) bool — `True` where seat is a learner seat in that
    # env (the only record of it: the per-env Python sets that used to mirror
    # it were never read). Set per deal by `_draw_pool_mix`; enables the
    # vectorized "is this actor a learner seat?" lookups in the hot loop.
    learner_seats_mask = np.ones((n_envs, n_seats), dtype=bool)
    # (n_envs,) i64 — pool snapshot index per env, or -1 for self-play.
    # Replaces the per-env Python list `env_snapshot_idx` so opp grouping
    # can be vectorized via `np.unique` over the masked column.
    env_snapshot_idx_arr = np.full(n_envs, -1, dtype=np.int64)

    # Eager-build every pool member once up front (cheap if snapshot_cache
    # already warm from multiconfig). Avoids first-use build mid-hand when a
    # late terminal re-mix draws a not-yet-seen index. Models stay on `device`
    # for the whole sub-rollout / update (P5); no cross-update cache.
    for _sd in range(len(pool.snapshots)):
        _get_snapshot_model(_sd)

    def _assign_pool_mix(env_ids: np.ndarray) -> None:
        """Opponent assignment for the hands about to be dealt in `env_ids`."""
        snap, mask = _draw_pool_mix(
            rng, int(env_ids.size), n_seats, len(pool.snapshots),
            pool_opp_seats, pool_mix_prob,
        )
        env_snapshot_idx_arr[env_ids] = snap
        learner_seats_mask[env_ids] = mask

    _assign_pool_mix(np.arange(n_envs, dtype=np.int64))

    # Batched opponents (_StackedOpponents): ONE stacked forward + ONE sampling
    # pass for every pool snapshot per step. None = per-snapshot calls
    # (--no-batched-opponents, an empty pool, or snapshots of mixed shape).
    _stacked_opp: _StackedOpponents | None = None
    if bool(getattr(train_config, "batched_opponents", True)) and len(pool.snapshots):
        _snap_models = [_get_snapshot_model(i) for i in range(len(pool.snapshots))]
        if _StackedOpponents.supported(_snap_models):
            _stacked_opp = _StackedOpponents(_snap_models)

    init_seeds = rng.integers(0, 2**63 - 1, size=n_envs, dtype=np.int64).astype(
        np.uint64
    )
    init_buttons = rng.integers(0, n_seats, size=n_envs, dtype=np.int64).astype(
        np.uint8
    )
    env.reset_batch(init_seeds, init_buttons)

    def _redeal_done_at_deal(env_ids: np.ndarray) -> np.ndarray:
        """Re-deal (fresh seed + button) every env in `env_ids` that is
        already terminal AT DEAL, until each has a live hand; returns the
        ids that were re-dealt at least once (their hole cards changed).

        (review 2026-09-20 A1) With every stack <= ante the hand is over at
        deal: `dones` is True straight after the reset, `apply_hybrid_batch`
        never reports a NEWLY terminal env, nothing flushes or resets — the
        `while wcursor < rollout_target` loop span forever, silently (the
        guardians only watch PID + entropy). Done-at-deal is otherwise a
        normal, cheap outcome (no decision = no row; the engine runs such a
        hand out at deal), so it is simply re-dealt; only a config that can
        NEVER deal a live hand raises. Draws RNG only when some env is dead,
        so ordinary configs consume exactly the pre-fix stream."""
        touched = np.zeros(n_envs, dtype=bool)
        dead = np.zeros(n_envs, dtype=bool)
        dead[env_ids] = env._dones[env_ids]
        tries = 0
        while dead.any():
            if tries >= _MAX_REDEALS:
                raise _dead_config_error(
                    "collect_rollout_batched", game_config, int(dead.sum())
                )
            tries += 1
            touched |= dead
            seeds = rng.integers(
                0, 2**63 - 1, size=n_envs, dtype=np.int64
            ).astype(np.uint64)
            buttons = rng.integers(
                0, n_seats, size=n_envs, dtype=np.int64
            ).astype(np.uint8)
            env._be.reset_terminal_batch(seeds, buttons, dead)
            env._reset_seeds = np.where(dead, seeds, env._reset_seeds)
            env._refresh_subset(dead)
            dead &= env._dones
        return np.nonzero(touched)[0]

    _redeal_done_at_deal(np.arange(n_envs, dtype=np.int64))
    # Per-hand hole cache (holes are static within a hand): one bulk
    # fetch per reset wave feeds the critic input + stored opp blocks.
    holes_cache = np.asarray(env._be.all_hole_cards_batch(), dtype=np.uint8)
    # Attack #4 S2: hero-rotated opp-hole blocks per (env, seat), filled on
    # deal/reset. Flush indexes this instead of rebuilding every terminal wave.
    _hole_w = int(game_config.hole_count)
    holes_rot_cache = np.full(
        (n_envs, n_seats, 5, _hole_w), 255, dtype=np.uint8
    )

    def _fill_holes_rot(env_ids: np.ndarray) -> None:
        if env_ids.size == 0:
            return
        hc = holes_cache[env_ids]
        seat_ids = np.arange(n_seats, dtype=np.int64)
        j5 = np.arange(5, dtype=np.int64)
        rot_seats = (seat_ids[:, None] + 1 + j5[None, :]) % n_seats
        block = hc[:, rot_seats]
        invalid = (j5 + 1) >= n_seats
        if invalid.any():
            block = np.where(
                invalid[None, None, :, None], np.uint8(255), block
            )
        holes_rot_cache[env_ids] = block

    _fill_holes_rot(np.arange(n_envs, dtype=np.int64))

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
    # envs) — the obs pool / output slabs are sized by `_slack_per_env`
    # below (growing on demand), NOT this capacity, and flush temporaries
    # are bounded by the actual max trajectory length per flush.
    MAX_STEPS_PER_SEAT = 192
    # The arrays start at `_TRAJ_CAP_INIT` slots and DOUBLE on demand up to
    # that cap (`_grow_traj`; exact — slots at or past a seat's length are
    # never read). A hand uses < ~16 decisions per seat, so the old fixed
    # 192-slot allocation zero-filled ~735 MB per vMin1 sub-rollout
    # (7,333 envs x 6 seats x 192 x 87 B; ~2.4 s per 30-config update) for
    # slots that stay empty (2026-09-23).
    # The arrays are REUSED across sub-rollouts / updates with the same
    # (n_envs, n_seats, estimator) — see _TRAJ_BUFFERS; only the per-seat
    # lengths restart at zero (stale slots past them are never read).
    traj_key = (int(n_envs), int(n_seats), bool(use_vrpo))
    traj_buf = _TRAJ_BUFFERS.get(traj_key)
    if traj_buf is None:
        _cap0 = min(_TRAJ_CAP_INIT, MAX_STEPS_PER_SEAT)
        traj_buf = {"cap": _cap0, **_alloc_traj(n_envs, n_seats, _cap0, use_vrpo)}
        _TRAJ_BUFFERS[traj_key] = traj_buf
    traj_cap = int(traj_buf["cap"])
    traj_lengths = np.zeros((n_envs, n_seats), dtype=np.int32)

    def _bind_traj() -> None:
        nonlocal traj_obs_idx, traj_gate, traj_chips, traj_sizing, traj_anchor
        nonlocal traj_u, traj_log_p, traj_gate_lp, traj_anchor_lp, traj_value
        nonlocal traj_q_taken, traj_vpi, costs_arr, pots_arr, streets_arr
        b = traj_buf
        # Absolute index into `step_obs_pool` / `step_gm_pool` per (env, seat, slot).
        traj_obs_idx, traj_gate, traj_chips = b["obs_idx"], b["gate"], b["chips"]
        traj_sizing, traj_anchor, traj_u = b["sizing"], b["anchor"], b["u"]
        traj_log_p, traj_gate_lp = b["log_p"], b["gate_lp"]
        traj_anchor_lp, traj_value = b["anchor_lp"], b["value"]
        # VRPO: Q(s_t, a_t) and V^π(s_t)=Σ_a π(a)Q(s_t,a) per learner step, laid
        # out like traj_value; the Expected-SARSA(λ) scan consumes them. None
        # when the estimator is GAE (never indexed in that path).
        traj_q_taken, traj_vpi = b["q_taken"], b["vpi"]
        # Per-step cost / pot / street parallel arrays, mirrored shape.
        costs_arr, pots_arr, streets_arr = b["costs"], b["pots"], b["streets"]

    traj_obs_idx = traj_gate = traj_chips = traj_sizing = traj_anchor = None
    traj_u = traj_log_p = traj_gate_lp = traj_anchor_lp = traj_value = None
    traj_q_taken = traj_vpi = costs_arr = pots_arr = streets_arr = None
    _bind_traj()

    def _grow_traj(min_cap: int) -> None:
        """Re-allocate every per-(env, seat) trajectory array with at least
        `min_cap` slots (doubling, capped at MAX_STEPS_PER_SEAT), copying the
        slots written so far; fresh slots get each array's initial fill. The
        grown arrays replace the cached ones (later sub-rollouts start big)."""
        nonlocal traj_cap
        new_cap = min(MAX_STEPS_PER_SEAT, max(int(min_cap), 2 * traj_cap))
        grown = _alloc_traj(n_envs, n_seats, new_cap, use_vrpo)
        for name, arr in grown.items():
            if arr is not None:
                arr[:, :, :traj_cap] = traj_buf[name]
            traj_buf[name] = arr
        traj_buf["cap"] = new_cap
        traj_cap = new_cap
        _bind_traj()

    # Flat pre-allocated obs / gate-mask pool. Each step appends
    # `learner_idx_np.size` rows at `pool_cursor`; trajectory slots
    # store the absolute pool index. One contiguous buffer makes the
    # terminal-flush gather a single `np.take` rather than a Python
    # walk over chunked storage.
    rollout_target = int(train_config.rollout_length)
    # Slack for learner steps written past the rollout target: with
    # drain_inflight every in-flight hand's rows (flushed as those hands
    # finish), without it the abandoned in-flight hands' unflushed rows
    # (pool only). This is a per-env STATISTICAL figure (~one hand's rows),
    # NOT the per-seat capacity above — there is no reason for it to scale
    # with MAX_STEPS_PER_SEAT. The cost of oversizing is only VIRTUAL
    # address space (np.empty pages materialize on first write and the
    # slack tail is mostly never written). It is the INITIAL capacity
    # only: an undersized guess GROWS (one copy, `_grow_pool` /
    # `_grow_slabs` below) instead of raising, and `_slack_per_env` learns
    # the real need so later collections start right-sized (review
    # 2026-09-20 A6 — a drained cold-start policy plays long hands and can
    # exceed any fixed per-env constant). Module-level helper — shared with
    # multiconfig's shared-staging allocation.
    pool_cap = rollout_target + n_envs * _slack_per_env()
    # env.obs_dim, not the OBS_DIM constant: the batched env's layout is
    # per-variant (991 PLO / 995 NLH).
    # Compact observation storage (compact_obs.py; None = dense float32). The
    # pool and the output slabs share the layout; multiconfig's shared staging
    # uses the same resolver, so the `out_slabs` it hands over match too.
    obs_layout = _resolve_obs_layout(train_config, game_config.variant)
    # The pool is REUSED across sub-rollouts / updates (_OBS_POOL_BUFFERS);
    # only rows below `pool_cursor` — this collection's — are ever read.
    pool_key = (int(env.obs_dim), obs_layout.name if obs_layout is not None else "dense")
    pool_buf = _OBS_POOL_BUFFERS.get(pool_key)
    if pool_buf is None or pool_buf["cap"] < pool_cap:
        pool_buf = {
            "cap": pool_cap,
            "obs": _alloc_obs_pool(pool_cap, env.obs_dim, obs_layout),
            "gm": np.empty((pool_cap, GATE_ACTIONS), dtype=bool),
        }
        _OBS_POOL_BUFFERS[pool_key] = pool_buf
    pool_cap = int(pool_buf["cap"])
    step_obs_pool, step_gm_pool = pool_buf["obs"], pool_buf["gm"]
    pool_cursor = 0

    def _grow_pool(min_rows: int) -> None:
        nonlocal step_obs_pool, step_gm_pool, pool_cap
        new_cap = max(int(min_rows), pool_cap + max(pool_cap // 4, n_envs))
        new_obs = _alloc_obs_pool(new_cap, env.obs_dim, obs_layout)
        new_gm = np.empty((new_cap, GATE_ACTIONS), dtype=bool)
        for key, arr in step_obs_pool.items():
            new_obs[key][:pool_cursor] = arr[:pool_cursor]
        new_gm[:pool_cursor] = step_gm_pool[:pool_cursor]
        step_obs_pool, step_gm_pool, pool_cap = new_obs, new_gm, new_cap
        pool_buf.update(cap=new_cap, obs=new_obs, gm=new_gm)

    # Pre-allocated output slabs (`_slab_keys` layout). Eliminates the
    # `np.stack` over millions of small arrays at finalize time. `wcursor`
    # tracks the count of transitions written so far across all terminal
    # flushes. On CUDA, back the slabs with pinned host memory so the finalize
    # transfer can run with `non_blocking=True` and overlap the first PPO
    # forward.
    if out_slabs is not None:
        # P8 shared staging: write into the caller's views of one big host
        # buffer. The views' length IS the capacity (the caller sizes them
        # with the same `_slack_per_env` helper and may hand over all of its
        # remaining buffer); the width asserts catch any variant/obs-dim
        # drift between the caller's allocation and this env.
        slabs = {key: out_slabs[key] for key in _slab_keys(obs_layout)}
        for key, (shape, np_dtype, _td) in _obs_spec(env.obs_dim, obs_layout).items():
            assert slabs[key].shape[1:] == shape and slabs[key].dtype == np_dtype, (
                f"out_slabs {key} {slabs[key].shape[1:]} {slabs[key].dtype} != "
                f"env layout {shape} {np.dtype(np_dtype)}"
            )
        assert slabs["oh"].shape[2] == game_config.hole_count, (
            f"out_slabs hole width {slabs['oh'].shape[2]} != "
            f"config {game_config.hole_count}"
        )
        _slab_alloc = None
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
        _slab_alloc = _SlabAllocator(
            env.obs_dim, game_config.hole_count, _pin, layout=obs_layout
        )
        slabs = _slab_alloc.alloc(rollout_target + n_envs * _slack_per_env())
    out_cap = int(slabs["gm"].shape[0])
    wcursor = 0

    def _grow_slabs(min_rows: int) -> None:
        """Make room for `min_rows` output rows, preserving the `wcursor`
        rows already flushed (A6: replaces the old hard overflow error)."""
        nonlocal slabs, out_cap
        if _slab_alloc is not None:
            new_cap = max(int(min_rows), out_cap + max(out_cap // 4, n_envs))
            slabs = _slab_alloc.grow(slabs, wcursor, new_cap)
        elif out_slabs_grow is not None:
            grown = out_slabs_grow(wcursor, int(min_rows))
            slabs = {key: grown[key] for key in _slab_keys(obs_layout)}
        else:
            raise RuntimeError(
                f"output slab overflow: {min_rows} > out_cap={out_cap} "
                f"(rollout_target={rollout_target}, {n_envs} envs) and the "
                "caller's out_slabs came without an out_slabs_grow callback"
            )
        out_cap = int(slabs["gm"].shape[0])
        assert out_cap >= min_rows, f"slab grow fell short: {out_cap} < {min_rows}"

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

    def _act_to_host(_fw_out, o_t: torch.Tensor | None = None) -> tuple:
        """Coalesce device ActOut -> host numpy (one CUDA sync for the stacks).

        Bit-identical layout to the pre-#1 per-forward D2H. When ``o_t`` is
        provided it is returned as the last element (learner/critic reuse).
        """
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
        base = (
            ints[0].astype(np.uint8),
            ints[1].astype(np.int64),
            ints[2].astype(np.int64),
            floats[0],
            floats[1],
            floats[2],
            floats[3],
            floats[4],
            marg_np,
        )
        if o_t is not None:
            return base + (o_t,)
        return base

    def _learner_act_device(
        group: np.ndarray,
        obs_arr: np.ndarray,
        gate_mask_arr: np.ndarray,
        sizing_arr: np.ndarray,
        want_marginal: bool,
    ):
        """Learner H2D + act; leave ActOut + o_t on device (no D2H).

        Attack #2 Phase 1: defer host pull until after opp acts are queued so
        the step has one coalesced learner D2H instead of sync-per-stage.
        act() order unchanged (still before any opp act).
        """
        b_obs = obs_arr[group]
        b_gm = gate_mask_arr[group]
        b_sizing = sizing_arr[group]
        with _TimedRF("step2/learner_h2d"):
            _step_h2d.wait_slot(0)
            o_t, m_t, b_t = _step_h2d.upload(b_obs, b_gm, b_sizing, slot=0)
        with _TimedRF("step3/learner_forward"):
            with torch.inference_mode():
                if want_marginal:
                    _fw_out = learner.act(o_t, m_t, b_t, return_marginal=True)
                else:
                    _fw_out = learner.act(o_t, m_t, b_t)
        return _fw_out, o_t

    # Double-pin opp path: alternate slots 0/1 so group i+1 H2D does not
    # wait for group i's H2D (only waits when reusing a slot = group i-2).
    _opp_pin_slot: list = [0]

    def _opp_upload_act(
        model: ActorCritic,
        group: np.ndarray,
        obs_arr: np.ndarray,
        gate_mask_arr: np.ndarray,
        sizing_arr: np.ndarray,
    ):
        """H2D + act for one opp group; leave results on device (no D2H).

        Uses alternating ``_step_h2d`` pin slots. act() order is still
        unique(sd) ascending (same CUDA RNG as pre-#1).
        """
        b_obs = obs_arr[group]
        b_gm = gate_mask_arr[group]
        b_sizing = sizing_arr[group]
        slot = _opp_pin_slot[0]
        _step_h2d.wait_slot(slot)
        o_t, m_t, b_t = _step_h2d.upload(b_obs, b_gm, b_sizing, slot=slot)
        _opp_pin_slot[0] = 1 - slot if _step_h2d.n_slots > 1 else 0
        with torch.inference_mode():
            return model.act(o_t, m_t, b_t)

    def _opp_acts(
        is_opp: np.ndarray,
        obs_arr: np.ndarray,
        gate_mask_arr: np.ndarray,
        sizing_arr: np.ndarray,
    ) -> list[tuple[np.ndarray, object]]:
        """Act every opponent row in `is_opp`, results left on device (no D2H):
        [(env rows, ActOut)] — ONE entry covering every snapshot on the stacked
        path (rows grouped by snapshot), one per snapshot group otherwise
        (ascending snapshot index, the pre-2026-09-23 behavior)."""
        snap_col = np.where(is_opp, env_snapshot_idx_arr, -1)
        if _stacked_opp is None:
            groups: list[tuple[np.ndarray, object]] = []
            for sd_idx in np.unique(snap_col[snap_col >= 0]):
                group = np.nonzero(snap_col == sd_idx)[0]
                m = _get_snapshot_model(int(sd_idx))
                groups.append(
                    (group, _opp_upload_act(m, group, obs_arr, gate_mask_arr, sizing_arr))
                )
            return groups
        rows = np.nonzero(snap_col >= 0)[0]
        if rows.size == 0:
            return []
        order = np.argsort(snap_col[rows], kind="stable")
        rows = rows[order]
        g = snap_col[rows]
        counts = np.bincount(g, minlength=_stacked_opp.n)
        j = np.arange(rows.size) - (np.cumsum(counts) - counts)[g]
        slot = _opp_pin_slot[0]
        _step_h2d.wait_slot(slot)
        o_t, m_t, b_t = _step_h2d.upload(
            obs_arr[rows], gate_mask_arr[rows], sizing_arr[rows], slot=slot
        )
        _opp_pin_slot[0] = 1 - slot if _step_h2d.n_slots > 1 else 0
        g_t = torch.from_numpy(g).to(device)
        j_t = torch.from_numpy(j).to(device)
        with torch.inference_mode():
            fw = _stacked_opp.act(o_t, m_t, b_t, g_t, j_t, int(counts.max()))
        return [(rows, fw)]

    # Attack #2 Phase 2: cross-iteration act prefetch under terminal host work.
    # Default OFF — enable with PLO5BP_ROLLOUT_OVERLAP=1 after parity confidence.
    _rollout_overlap = (
        device.type == "cuda"
        and os.environ.get("PLO5BP_ROLLOUT_OVERLAP", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )
    _act_prefetch: dict | None = None

    def _queue_acts_for_mask(active_mask: np.ndarray) -> dict:
        """Run learner+opp+critic acts for envs where active_mask is True.

        Returns a dict of device-side outputs + host index arrays. Does NOT
        D2H. Caller is responsible for coalesced host pull.
        Empty active_mask → empty dict with zero-size indices.
        """
        active_mask = np.asarray(active_mask, dtype=bool)
        if not active_mask.any():
            return {
                "learner_idx": np.zeros(0, dtype=np.int64),
                "learner_fw": None,
                "learner_o_t": None,
                "critic_v_t": None,
                "critic_q_t": None,
                "opp_groups": [],
                "safe_actors": np.where(
                    env._actors >= 0, env._actors, 0
                ).astype(np.intp),
                "sizing_step": np.zeros((n_envs, 4), dtype=np.int64),
                "obs": env._obs.copy(),
                "gate_masks": env._gate_mask.copy(),
                "live_mask": np.zeros(n_envs, dtype=bool),
            }
        actors_q = env._actors
        dones_q = env._dones
        safe_q = np.where(actors_q >= 0, actors_q, 0).astype(np.intp)
        # Restrict to mask ∩ ~dones ∩ actor>=0
        live = active_mask & (~dones_q) & (actors_q >= 0)
        is_learn = live & learner_seats_mask[env_idx_range, safe_q]
        is_opp = live & ~is_learn
        l_idx = np.nonzero(is_learn)[0]
        l_fw = None
        l_o = None
        c_v = None
        c_q = None
        # sizing for ALL envs (same formula as main loop) — only live rows used
        pre_btc = env._bet_to_call
        pre_sc = env._street_commit
        to_call_q = np.maximum(
            pre_btc.astype(np.int64) - pre_sc[env_idx_range, safe_q].astype(np.int64),
            0,
        )
        sizing_q = np.stack(
            [
                env._min_raise.astype(np.int64),
                env._max_raise.astype(np.int64),
                env._pot.astype(np.int64),
                to_call_q,
            ],
            axis=-1,
        )
        if l_idx.size:
            l_fw, l_o = _learner_act_device(
                l_idx, env._obs, env._gate_mask, sizing_q,
                want_marginal=use_vrpo,
            )
            if critic is not None:
                opp_blk = _rotate_opp_holes_batch(
                    holes_cache, l_idx, safe_q[l_idx]
                )
                h_t = torch.from_numpy(opp_blk).to(
                    device, non_blocking=(device.type == "cuda")
                )
                with torch.inference_mode():
                    if use_vrpo:
                        c_v, c_q = critic.q_values(
                            l_o, opp_holes_multihot(h_t)
                        )
                    else:
                        c_v = critic(l_o, opp_holes_multihot(h_t))
        o_groups: list[tuple[np.ndarray, object]] = []
        if is_opp.any():
            o_groups = _opp_acts(is_opp, env._obs, env._gate_mask, sizing_q)
        return {
            "learner_idx": l_idx,
            "learner_fw": l_fw,
            "learner_o_t": l_o,
            "critic_v_t": c_v,
            "critic_q_t": c_q,
            "opp_groups": o_groups,
            "safe_actors": safe_q,
            "sizing_step": sizing_q,
            "obs": env._obs.copy(),
            "gate_masks": env._gate_mask.copy(),
            "live_mask": live,
        }

    def _d2h_queued_acts(q: dict) -> dict:
        """Coalesce D2H for a _queue_acts_for_mask result → host arrays."""
        n = n_envs
        gates = np.zeros(n, dtype=np.uint8)
        chips = np.zeros(n, dtype=np.uint64)
        anchors = np.full(n, -1, dtype=np.int64)
        refine_u = np.zeros(n, dtype=np.float32)
        log_probs = np.zeros(n, dtype=np.float32)
        gate_logp = np.zeros(n, dtype=np.float32)
        anchor_logp = np.zeros(n, dtype=np.float32)
        values = np.zeros(n, dtype=np.float32)
        q_taken = np.zeros(n, dtype=np.float32) if use_vrpo else None
        vpi = np.zeros(n, dtype=np.float32) if use_vrpo else None
        l_idx = q["learner_idx"]
        with _TimedRF("step5/action_d2h"):
            if q["learner_fw"] is not None and l_idx.size:
                g_np, c_np, an_np, lp_np, ru_np, v_np, glp_np, alp_np, marg_np = (
                    _act_to_host(q["learner_fw"])  # type: ignore[misc]
                )
                gates[l_idx] = g_np
                chips[l_idx] = np.maximum(c_np, 0).astype(np.uint64)
                anchors[l_idx] = an_np
                refine_u[l_idx] = ru_np
                log_probs[l_idx] = lp_np
                gate_logp[l_idx] = glp_np
                anchor_logp[l_idx] = alp_np
                c_v = q["critic_v_t"]
                c_q = q["critic_q_t"]
                if c_v is not None:
                    if use_vrpo and c_q is not None:
                        vq = torch.cat(
                            [c_v.float().unsqueeze(-1), c_q.float()], dim=-1
                        ).cpu().numpy()
                        v_l = vq[..., 0]
                        q_l = vq[..., 1:]
                        values[l_idx] = v_l
                        if q_l.shape[-1] == 3:
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
                            q_idx_l = np.where(
                                g_np == GATE_RAISE,
                                2 + an_np,
                                g_np.astype(np.int64),
                            )
                            marg_l = marg_np
                        rows_l = np.arange(l_idx.size)
                        q_taken[l_idx] = q_l[rows_l, q_idx_l]
                        vpi[l_idx] = (marg_l * q_l).sum(-1)
                    else:
                        values[l_idx] = c_v.float().cpu().numpy()
                else:
                    values[l_idx] = v_np
            o_groups = q["opp_groups"]
            if o_groups:
                g_parts = [out.gate for _, out in o_groups]
                c_parts = [out.chips for _, out in o_groups]
                g_cat = torch.cat(g_parts, dim=0)
                c_cat = torch.cat(c_parts, dim=0)
                gc = torch.stack(
                    (g_cat.to(torch.int64), c_cat.to(torch.int64)), dim=0
                ).cpu().numpy()
                g_all = gc[0].astype(np.uint8)
                c_all = np.maximum(gc[1], 0).astype(np.uint64)
                cursor = 0
                for group, _ in o_groups:
                    n_g = int(group.size)
                    gates[group] = g_all[cursor : cursor + n_g]
                    chips[group] = c_all[cursor : cursor + n_g]
                    cursor += n_g
        return {
            "gates_per_env": gates,
            "chips_per_env": chips,
            "anchors_per_env": anchors,
            "refine_u_per_env": refine_u,
            "log_probs_per_env": log_probs,
            "gate_logp_per_env": gate_logp,
            "anchor_logp_per_env": anchor_logp,
            "values_per_env": values,
            "q_taken_per_env": q_taken,
            "vpi_per_env": vpi,
            "learner_idx_np": l_idx,
            "safe_actors": q["safe_actors"],
            "sizing_step": q["sizing_step"],
            "obs": q["obs"],
            "gate_masks": q["gate_masks"],
        }

    env_idx_range = np.arange(n_envs)

    # PRODUCTION BEHAVIOR CHANGE (review 2026-09-20 A6) — drain_inflight.
    # Phase 1 (`wcursor < rollout_target`) is the pre-fix loop verbatim:
    # every finished env is re-dealt. Phase 2 (drain only) starts once the
    # target is reached: finished envs are flushed but NOT re-dealt, and the
    # loop runs until every env is done, i.e. every hand STARTED in phase 1 is
    # in the batch. With drain off the loop exits at the target exactly as
    # before (in-flight hands dropped) — byte-identical rows, order and RNG.
    step_timers.end()  # step0/setup
    while wcursor < rollout_target or (drain and not env._dones.all()):
        step_timers.begin("step0/prestep")
        obs = env._obs
        gate_masks = env._gate_mask
        min_raise = env._min_raise
        max_raise = env._max_raise
        actors = env._actors
        dones = env._dones
        if dones.all():
            # Unreachable by construction (every reset re-deals done-at-deal
            # envs or raises) — but an all-done table can never make progress,
            # so fail loudly rather than spin (review 2026-09-20 A1).
            raise RuntimeError(
                "collect_rollout_batched: no live env below the row target "
                f"({wcursor}/{rollout_target} rows) — config {game_config!r}"
            )
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
        step_timers.end()  # step0/prestep

        # Attack #2 Phase 2: consume prefetched acts from prior iteration
        # (queued under terminal host work). Skip device act this iter.
        _used_prefetch = False
        if _act_prefetch is not None:
            with _TimedRF("step2x/consume_prefetch"):
                pf = _act_prefetch
                _act_prefetch = None
                gates_per_env = pf["gates_per_env"]
                chips_per_env = pf["chips_per_env"]
                anchors_per_env = pf["anchors_per_env"]
                refine_u_per_env = pf["refine_u_per_env"]
                log_probs_per_env = pf["log_probs_per_env"]
                gate_logp_per_env = pf["gate_logp_per_env"]
                anchor_logp_per_env = pf["anchor_logp_per_env"]
                values_per_env = pf["values_per_env"]
                if use_vrpo:
                    q_taken_per_env = pf["q_taken_per_env"]
                    vpi_per_env = pf["vpi_per_env"]
                learner_idx_np = pf["learner_idx_np"]
                safe_actors = pf["safe_actors"]
                sizing_step = pf["sizing_step"]
                # Obs/masks at act time (may differ from current env if we
                # only prefetched a subset — full-mask prefetch matches).
                obs = pf["obs"]
                gate_masks = pf["gate_masks"]
                _used_prefetch = True

        # Attack #2 Phase 1: learner act on device, then opp acts on device,
        # then ONE coalesced D2H for learner (+ critic) and opp gates/chips.
        # act() call order unchanged: full learner batch, then opp groups in
        # unique(sd) order — bit-exact vs pre-#2 trajectories.
        if not _used_prefetch:
            _learner_fw = None
            _learner_o_t = None
            _critic_v_t = None
            _critic_q_t = None
            _opp_block_np = None
            if learner_idx_np.size:
                _learner_fw, _learner_o_t = _learner_act_device(
                    learner_idx_np, obs, gate_masks, sizing_step,
                    want_marginal=use_vrpo,
                )
                # Host prep that does not need D2H: opp-hole rotation for critic
                # can run while learner kernels finish (and before/during opp acts).
                if critic is not None:
                    with _TimedRF("step3a/opp_holes_rot"):
                        _opp_block_np = _rotate_opp_holes_batch(
                            holes_cache, learner_idx_np, safe_actors[learner_idx_np]
                        )
                    with _TimedRF("step3b/critic_forward"):
                        h_t = torch.from_numpy(_opp_block_np).to(
                            device, non_blocking=(device.type == "cuda")
                        )
                        with torch.inference_mode():
                            if use_vrpo:
                                _critic_v_t, _critic_q_t = critic.q_values(
                                    _learner_o_t, opp_holes_multihot(h_t)
                                )
                            else:
                                _critic_v_t = critic(
                                    _learner_o_t, opp_holes_multihot(h_t)
                                )

            # Group active opponent envs by snapshot index. Attack #1: all opp
            # act()s before any opp D2H; double-pin between groups.
            opp_groups: list[tuple[np.ndarray, object]] = []
            if is_opp_active.any():
                with _TimedRF("step4a/opp_h2d_act"):
                    opp_groups = _opp_acts(
                        is_opp_active, obs, gate_masks, sizing_step
                    )

            # Coalesced host pull: learner ActOut (+ critic) and all opp gates/chips.
            with _TimedRF("step5/action_d2h"):
                if _learner_fw is not None:
                    g_np, c_np, an_np, lp_np, ru_np, v_np, glp_np, alp_np, marg_np = (
                        _act_to_host(_learner_fw)  # type: ignore[misc]
                    )
                    gates_per_env[learner_idx_np] = g_np
                    chips_per_env[learner_idx_np] = np.maximum(c_np, 0).astype(np.uint64)
                    anchors_per_env[learner_idx_np] = an_np
                    refine_u_per_env[learner_idx_np] = ru_np
                    log_probs_per_env[learner_idx_np] = lp_np
                    gate_logp_per_env[learner_idx_np] = glp_np
                    anchor_logp_per_env[learner_idx_np] = alp_np
                    if _critic_v_t is not None:
                        if use_vrpo and _critic_q_t is not None:
                            # One D2H for V||Q (same as _critic_q_values coalesce).
                            vq = torch.cat(
                                [_critic_v_t.float().unsqueeze(-1), _critic_q_t.float()],
                                dim=-1,
                            ).cpu().numpy()
                            v_l = vq[..., 0]
                            q_l = vq[..., 1:]
                            values_per_env[learner_idx_np] = v_l
                            if q_l.shape[-1] == 3:
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
                                q_idx_l = np.where(
                                    g_np == GATE_RAISE, 2 + an_np, g_np.astype(np.int64)
                                )
                                marg_l = marg_np
                            rows_l = np.arange(learner_idx_np.size)
                            q_taken_per_env[learner_idx_np] = q_l[rows_l, q_idx_l]
                            vpi_per_env[learner_idx_np] = (marg_l * q_l).sum(-1)
                        else:
                            values_per_env[learner_idx_np] = (
                                _critic_v_t.float().cpu().numpy()
                            )
                    else:
                        values_per_env[learner_idx_np] = v_np

                if opp_groups:
                    g_parts = [out.gate for _, out in opp_groups]
                    c_parts = [out.chips for _, out in opp_groups]
                    g_cat = torch.cat(g_parts, dim=0)
                    c_cat = torch.cat(c_parts, dim=0)
                    gc = torch.stack(
                        (g_cat.to(torch.int64), c_cat.to(torch.int64)), dim=0
                    ).cpu().numpy()
                    g_all = gc[0].astype(np.uint8)
                    c_all = np.maximum(gc[1], 0).astype(np.uint64)
                    cursor = 0
                    for group, _ in opp_groups:
                        n_g = int(group.size)
                        gates_per_env[group] = g_all[cursor : cursor + n_g]
                        chips_per_env[group] = c_all[cursor : cursor + n_g]
                        cursor += n_g
                del opp_groups

        # Vectorized trajectory snapshot: one bulk obs/gm copy into the
        # flat pool, plus fancy-index writes into the per-(env, seat)
        # arrays. The (env, seat, slot) -> pool index map is one int.
        if learner_idx_np.size:
            k_step = learner_idx_np.size
            pool_end = pool_cursor + k_step
            if pool_end > pool_cap:
                step_timers.begin("step3c/pool_grow")
                _grow_pool(pool_end)
                step_timers.end()  # step3c/pool_grow
            step_timers.begin("step3c/obs_pack")
            if obs_layout is None:
                step_obs_pool["obs"][pool_cursor:pool_end] = obs[learner_idx_np]
            else:
                pack_rows_into(
                    obs, learner_idx_np, obs_layout,
                    step_obs_pool["obs_bits"], step_obs_pool["obs_real"],
                    pool_cursor,
                )
            step_gm_pool[pool_cursor:pool_end] = gate_masks[learner_idx_np]
            step_timers.end()  # step3c/obs_pack
            pool_indices = np.arange(pool_cursor, pool_end, dtype=np.int64)
            pool_cursor = pool_end

            learner_actors = safe_actors[learner_idx_np]
            slots = traj_lengths[learner_idx_np, learner_actors]
            if (slots >= MAX_STEPS_PER_SEAT).any():
                raise RuntimeError(
                    f"per-seat trajectory length exceeded "
                    f"MAX_STEPS_PER_SEAT={MAX_STEPS_PER_SEAT}"
                )
            if int(slots.max()) >= traj_cap:
                step_timers.begin("step3c/traj_grow")
                _grow_traj(int(slots.max()) + 1)
                step_timers.end()  # step3c/traj_grow

            step_timers.begin("step3c/traj_writes")
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
            step_timers.end()  # step3c/traj_writes

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
        with _TimedRF("step6+7/rust_apply"):
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
        # `active &` only matters in the A6 drain phase, where envs that
        # finished on an EARLIER step stay done (never re-dealt) and need no
        # encode either; before the target `active` is all-True, so the mask
        # is `~newly_terminal` exactly as before.
        with _TimedRF("step1a/refresh"):
            _enc = active & ~newly_terminal
            if _enc.all():
                env._refresh()
            else:
                env._refresh(encode_mask=_enc)
        post_total_commit = env._total_commit

        # Rust-parallel aggression bonus + per-step cost/pot/street
        # bookkeeping. Replaces the per-env Python arithmetic loop.
        with _TimedRF("step8/aggression_bonus"):
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

        # Attack #2 Phase 2: while terminal host runs, queue next-step acts
        # for envs that are still live (not newly terminal). After reset,
        # queue acts for re-dealt actives. D2H into _act_prefetch for the
        # next loop iteration. Flag OFF → identical control flow to pre-#2.
        _prefetch_queued = None
        if (
            _rollout_overlap
            and newly_terminal.any()
            and not newly_terminal.all()
            and wcursor < rollout_target
        ):
            with _TimedRF("step2x/act_wave_active"):
                # Live non-terminal envs already refreshed; act for them now.
                _nt_mask = ~newly_terminal
                _prefetch_queued = _queue_acts_for_mask(_nt_mask)

        if newly_terminal.any():
            with _TimedRF("step9a/payouts"):
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
                with _TimedRF("step9b/retroactive_bonus"):
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
                with _TimedRF("step9c/gae_scan"):
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
                with _TimedRF("step9d/slab_copies"):
                    # Attack #4 S1: one flat plan (sel/obs_idx), slice traj once
                    # to (T,S,L), gather with that plan. S2: opp-holes from
                    # holes_rot_cache (filled on deal/reset). Same sel order
                    # as pre-#4 (bit-exact row order).
                    flat = active_step.ravel()
                    n_new = int(flat.sum())
                    if n_new:
                        if wcursor + n_new > out_cap:
                            _grow_slabs(wcursor + n_new)
                        sel = np.nonzero(flat)[0]
                        end = wcursor + n_new
                        # Single (T,S,L) window into traj storage.
                        obs_idx_tsl = traj_obs_idx[term_envs, :, :L]
                        gate_tsl = traj_gate[term_envs, :, :L]
                        chips_tsl = traj_chips[term_envs, :, :L]
                        sizing_tsl = traj_sizing[term_envs, :, :L]
                        anchor_tsl = traj_anchor[term_envs, :, :L]
                        u_tsl = traj_u[term_envs, :, :L]
                        lp_tsl = traj_log_p[term_envs, :, :L]
                        glp_tsl = traj_gate_lp[term_envs, :, :L]
                        alp_tsl = traj_anchor_lp[term_envs, :, :L]
                        obs_idx = obs_idx_tsl.ravel()[sel]
                        for key, pool_arr in step_obs_pool.items():
                            np.take(
                                pool_arr, obs_idx, axis=0,
                                out=slabs[key][wcursor:end],
                            )
                        np.take(
                            step_gm_pool, obs_idx, axis=0,
                            out=slabs["gm"][wcursor:end],
                        )
                        slabs["ga"][wcursor:end] = gate_tsl.ravel()[sel].astype(
                            np.int64, copy=False
                        )
                        slabs["rc"][wcursor:end] = chips_tsl.ravel()[sel]
                        slabs["sz"][wcursor:end] = sizing_tsl.reshape(-1, 4)[sel]
                        slabs["an"][wcursor:end] = anchor_tsl.ravel()[sel].astype(
                            np.int64, copy=False
                        )
                        slabs["ru"][wcursor:end] = u_tsl.ravel()[sel]
                        # (T, S, 5, hole_w) already rotated; ts_idx = env*S+seat
                        ts_idx = sel // L
                        slabs["oh"][wcursor:end] = holes_rot_cache[term_envs].reshape(
                            T * S, 5, _hole_w
                        )[ts_idx]
                        slabs["lp"][wcursor:end] = lp_tsl.ravel()[sel]
                        slabs["glp"][wcursor:end] = glp_tsl.ravel()[sel]
                        slabs["alp"][wcursor:end] = alp_tsl.ravel()[sel]
                        slabs["v"][wcursor:end] = vals_t.ravel()[sel]
                        slabs["ret"][wcursor:end] = rets_t.ravel()[sel]
                        slabs["adv"][wcursor:end] = adv_out_t.ravel()[sel]
                        slabs["last"][wcursor:end] = (
                            t_idx[None, None, :] == last_t_arr[..., None]
                        ).ravel()[sel]
                        wcursor = end

                traj_lengths[term_envs] = 0

            # A6 drain: once the row target is reached the finished envs are
            # NOT re-dealt — they stay done (`active` masks them out) while
            # the hands still in flight play to completion. The draws below
            # sit AFTER the flush (which consumes no RNG) so the numpy stream
            # keeps its pre-fix order — seeds, buttons, pool-mix — and a
            # drain-off run is byte-identical.
            redeal = not (drain and wcursor >= rollout_target)
            if redeal:
                new_seeds = rng.integers(
                    0, 2**63 - 1, size=n_envs, dtype=np.int64
                ).astype(np.uint64)
                new_buttons = rng.integers(
                    0, n_seats, size=n_envs, dtype=np.int64
                ).astype(np.uint8)
                with _TimedRF("step9e/pool_mix"):
                    # Batched draws for every re-dealt hand (see _draw_pool_mix).
                    _assign_pool_mix(term_envs)

                with _TimedRF("step9f/reset_terminal"):
                    env._be.reset_terminal_batch(new_seeds, new_buttons, reset_mask)
                    env._reset_seeds = np.where(
                        reset_mask, new_seeds, env._reset_seeds
                    )
                    # Second refresh to pick up the post-reset state for the next
                    # iteration. `reset_terminal_batch` mutates ONLY the masked
                    # (terminal) envs, and nothing above mutated non-masked envs'
                    # engine state since the post-apply refresh — so only the
                    # reset rows need re-packing/re-encoding. The subset refresh
                    # is bit-exact-equivalent to a full `_refresh()` here (see
                    # `_refresh_subset` docstring + test_refresh_subset_parity).
                    env._refresh_subset(reset_mask)
                    # A1: a re-dealt hand that is already over at deal is
                    # re-dealt again (bounded), never left to stall the loop.
                    _redeal_done_at_deal(term_envs)
                    # Refresh the per-hand hole cache for the re-dealt envs
                    # only (holes are static within a hand; non-reset envs
                    # keep their prior rows). Subset fetch mirrors
                    # observation_and_features_subset_batch. After the A1
                    # re-deal pass so the cache holds the FINAL deal's cards.
                    term_idx = term_envs.astype(np.int64)
                    holes_sub = np.asarray(
                        env._be.all_hole_cards_subset_batch(term_idx), dtype=np.uint8
                    )
                    holes_cache[term_envs] = holes_sub
                    _fill_holes_rot(term_envs.astype(np.int64))

            # Wave B: re-dealt envs may now need an action; queue after reset.
            if _prefetch_queued is not None:
                with _TimedRF("step2x/act_wave_redealt"):
                    _rd_mask = newly_terminal.copy()
                    _q_b = _queue_acts_for_mask(_rd_mask)
                # Merge device queues then single D2H into host prefetch.
                with _TimedRF("step2x/prefetch_d2h"):
                    # Merge: start from wave A host pull, overlay wave B.
                    host_a = _d2h_queued_acts(_prefetch_queued)
                    host_b = _d2h_queued_acts(_q_b)
                    # Combine learner indices and arrays.
                    for key in (
                        "gates_per_env",
                        "chips_per_env",
                        "anchors_per_env",
                        "refine_u_per_env",
                        "log_probs_per_env",
                        "gate_logp_per_env",
                        "anchor_logp_per_env",
                        "values_per_env",
                    ):
                        # wave B only wrote re-dealt rows; copy those over A
                        m = newly_terminal
                        host_a[key][m] = host_b[key][m]
                    if use_vrpo:
                        host_a["q_taken_per_env"][newly_terminal] = host_b[
                            "q_taken_per_env"
                        ][newly_terminal]
                        host_a["vpi_per_env"][newly_terminal] = host_b[
                            "vpi_per_env"
                        ][newly_terminal]
                    # Learner idx = A then B (disjoint masks). Do NOT np.unique:
                    # unique sorts and reorders rows vs act order.
                    host_a["learner_idx_np"] = np.concatenate(
                        [host_a["learner_idx_np"], host_b["learner_idx_np"]]
                    )
                    # Obs/masks at act time: Wave A used non-term rows; Wave B
                    # re-dealt rows. Subset refresh only mutates terminal rows,
                    # so env._obs now matches both waves' act-time observations.
                    host_a["obs"] = env._obs.copy()
                    host_a["gate_masks"] = env._gate_mask.copy()
                    host_a["safe_actors"] = np.where(
                        env._actors >= 0, env._actors, 0
                    ).astype(np.intp)
                    # sizing from current env for traj
                    _sa = host_a["safe_actors"]
                    _tc = np.maximum(
                        env._bet_to_call.astype(np.int64)
                        - env._street_commit[env_idx_range, _sa].astype(np.int64),
                        0,
                    )
                    host_a["sizing_step"] = np.stack(
                        [
                            env._min_raise.astype(np.int64),
                            env._max_raise.astype(np.int64),
                            env._pot.astype(np.int64),
                            _tc,
                        ],
                        axis=-1,
                    )
                    # Drop prefetch if terminal flush already filled the batch —
                    # otherwise the next iter would apply extra steps past target.
                    if wcursor < rollout_target:
                        _act_prefetch = host_a

    # Sizing hint for the next collection's pool/slab slack (never a value).
    _note_slack_used(max(wcursor, pool_cursor), rollout_target, n_envs)

    if out_slabs is not None:
        # P7 shared-staging finalize: no full H2D — the rows already sit in the
        # caller's big host buffer. Only the per-sub advantage normalization
        # hops to the learner device: it ran on CUDA in the legacy path
        # (_finalize_batch_arr), and a CPU reimplementation would drift f32
        # reduction order, so ship the tiny (wcursor,) vector up, run the
        # IDENTICAL op sequence, and write the result back into the slab.
        with _TimedRF("step11/shared_adv_norm"):
            adv_view = slabs["adv"][:wcursor]
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
            obs=_obs_from_slabs(slabs, wcursor, obs_layout),
            gate_masks=torch.from_numpy(slabs["gm"][:wcursor]),
            gate_actions=torch.from_numpy(slabs["ga"][:wcursor]),
            raise_chips=torch.from_numpy(slabs["rc"][:wcursor]),
            sizing=torch.from_numpy(slabs["sz"][:wcursor]),
            anchor_actions=torch.from_numpy(slabs["an"][:wcursor]),
            refine_u=torch.from_numpy(slabs["ru"][:wcursor]),
            opp_holes=torch.from_numpy(slabs["oh"][:wcursor]),
            log_probs=torch.from_numpy(slabs["lp"][:wcursor]),
            values=torch.from_numpy(slabs["v"][:wcursor]),
            returns=torch.from_numpy(slabs["ret"][:wcursor]),
            advantages=torch.from_numpy(adv_view),
            old_gate_logp=torch.from_numpy(slabs["glp"][:wcursor]),
            old_anchor_logp=torch.from_numpy(slabs["alp"][:wcursor]),
            is_terminal=torch.from_numpy(slabs["last"][:wcursor]),
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

    # Report only when this call owns the timers (not multiconfig parent).
    # Multiconfig sets a parent before the loop and reports once after.
    if os.environ.get("PLO5BP_STEP_TIMERS_OWNED", "1").strip() != "0":
        step_timers.report(label="collect_rollout_batched")
        _ACTIVE_STEP_TIMERS = None
    return _finalize_batch_arr(
        _obs_from_slabs(slabs, wcursor, obs_layout),
        slabs["gm"],
        slabs["ga"],
        slabs["rc"],
        slabs["sz"],
        slabs["an"],
        slabs["ru"],
        slabs["oh"],
        slabs["lp"],
        slabs["v"],
        slabs["ret"],
        slabs["adv"],
        slabs["glp"],
        slabs["alp"],
        wcursor,
        device=device,
        aggr_bonus_total_bb=aggr_bonus_total_bb,
        aggr_steps_total=aggr_steps_total,
        aggr_bonus_steps=aggr_bonus_steps,
        aggr_steps_total_by_street=tuple(aggr_steps_total_by_street),
        aggr_bonus_steps_by_street=tuple(aggr_bonus_steps_by_street),
        adv_clip=float(getattr(train_config, "adv_clip", 0.0)),
        all_last_arr=slabs["last"],
    )


# ---- vThree: mix many (seats,stacks) configs within ONE update ------------
# Each update's gradient averages over N configs spanning all stack tiers,
# instead of one config + a 50-update block. This removes the consecutive-
# shallow exposure that saturated the gate (vTwo10-13 all died ~38 clubgg
# updates in). Implemented as a thin wrapper over the bit-exact single-config
# collector: split → host-concat → a final pooled advantage re-normalization
# which is numerically a NO-OP — advantages are effectively normalized PER
# CONFIG (see the `_concat_batches` docstring; review 2026-09-20 A5).

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
    """Concatenate sub-rollout Batches along the transition axis, then
    re-normalize the combined advantages (mean 0 / std 1, then the fat-tail
    clamp). Scalar aggression diagnostics sum.

    WHAT THIS ACTUALLY DOES (review 2026-09-20 A5 — the earlier text here
    claimed a GLOBAL normalization "so no single config's value scale
    dominates"; that is not what happens): every sub-rollout arrives ALREADY
    normalized to mean 0 / std 1 (+ clamp) by its own collector
    (`_finalize_batch_arr`, or the shared-staging `step11/shared_adv_norm`
    hop). Pooling N unit-variance, zero-mean vectors gives mean ~0 / std ~1,
    so the renorm below rescales by ~1.000x — numerically a no-op. Measured:
    raw advantage sigma 6.96bb (10bb config) vs 34.9bb (250bb config) -> both
    0.9996 after (`.claude/reviews/repro-2026-09-20/agent_rollout/
    mix_norm.py`). Advantage normalization under --mix-configs is therefore
    PER CONFIG: a 10bb config's transitions carry the same advantage scale as
    a 250bb config's. A genuinely pooled normalization (normalize the RAW
    advantages once, across configs) would be a production behavior change
    and is NOT implemented.

    NOTE (2026-06-23): a "per-config variant (preserve each sub-rollout's own
    unit-std; no global pool)" was tried as a suspected full-LR collapse fix
    and judged less stable (Ha 1.4->0.4 by u15). Given the above, that A/B
    compared two numerically near-identical normalizations, so its stability
    conclusion was run-to-run noise, not evidence for this renorm. The op is
    kept only because removing it would perturb f32 bits for no benefit."""
    if len(batches) == 1:
        return batches[0]

    def _cat(field: str) -> "torch.Tensor | PackedObs":
        parts = [getattr(b, field) for b in batches]
        if isinstance(parts[0], PackedObs):
            return PackedObs.cat(parts)
        return torch.cat(parts, dim=0)

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
    drain_inflight: "bool | None" = None,
) -> Batch:
    """One update's rollout MIXED across `configs` distinct (seats,stacks) setups.

    Runs `collect_rollout_batched` once per config — each sub-rollout sized
    `num_envs // N` envs and `rollout_length // N` learner steps (a row
    TARGET: with `drain_inflight` — resolved ONCE here and passed to every
    sub, see `collect_rollout_batched` — each sub also flushes its in-flight
    hands, so the combined batch runs past `rollout_length`).

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
    # Aggregate step timers across all sub-rollouts (one report at end).
    global _ACTIVE_STEP_TIMERS
    _parent_timers = _StepTimers()
    _ACTIVE_STEP_TIMERS = _parent_timers
    _prev_owned = os.environ.get("PLO5BP_STEP_TIMERS_OWNED")
    os.environ["PLO5BP_STEP_TIMERS_OWNED"] = "0"
    n = len(configs)
    if n == 0:
        raise ValueError("collect_rollout_multiconfig requires >= 1 config")
    if config_tiers is not None and len(config_tiers) != n:
        raise ValueError(
            f"config_tiers length {len(config_tiers)} != configs {n}"
        )
    device = next(learner.parameters()).device
    host = torch.device("cpu")
    # Resolved BEFORE `replace`: a drain flag attached to a config that
    # predates the dataclass field would not survive it.
    drain = _resolve_drain_inflight(train_config, drain_inflight)
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
        # evacuate to host, torch.cat, pooled renorm inside _concat_batches
        # (numerically a no-op — see its docstring).
        host_batches: list[Batch] = []
        for cfg in configs:
            sub = collect_rollout_batched(
                learner, pool, cfg, sub_config, rng, critic=critic,
                snapshot_cache=snapshot_cache,
                env_cache=env_cache,
                drain_inflight=drain,
            )
            host_batches.append(_batch_to_device(sub, host))
            del sub
        combined = _concat_batches(host_batches, adv_clip)
        subs: list[Batch] = host_batches
        sub_rows = [int(b.obs.shape[0]) for b in host_batches]
    else:
        # P7+P8 shared staging: one big host buffer, per-sub views, chained
        # bases. Each sub gets a view of ALL the buffer's remaining rows
        # (`arr[base:]`), so the per-sub slack is POOLED: a long-handed config
        # borrows what a short-handed one left unused. Initial capacity =
        # every sub at target + `_slack_per_env` rows/env; if the pooled
        # buffer still runs out (A6: a drained sub can need more than any
        # fixed constant), `_grow` reallocates it bigger and copies the rows
        # written so far — rare once `_slack_per_env` has learned the need.
        n_sub_envs = sub_config.num_envs
        sub_cap = sub_config.rollout_length + n_sub_envs * _slack_per_env()
        total_cap = n * sub_cap
        hole_count = configs[0].hole_count
        assert all(c.hole_count == hole_count for c in configs), (
            "mixed hole widths across mix-configs are unsupported"
        )
        if configs[0].variant == "nlh_single":
            obs_dim = OBS_DIM_NLH
        elif str(getattr(train_config, "obs_mode", "full")) == "minimal":
            obs_dim = OBS_DIM_MINIMAL
        else:
            obs_dim = OBS_DIM
        # Same pin gate as the collector's own slabs (default OFF — see the
        # pinning-tax comment there); pinning one big buffer instead of N
        # small ones is otherwise equivalent.
        _pin = (
            device.type == "cuda"
            and os.environ.get("PLO5BP_PIN_ROLLOUT", "0") == "1"
        )
        obs_layout = _resolve_obs_layout(train_config, configs[0].variant)
        # Reused across updates on a CUDA learner (_STAGING_BUFFERS /
        # _reuse_staging): the finished batch is copied to the GPU at the end
        # of this call, so the next collection may overwrite the buffer.
        reuse = _reuse_staging(device)
        stage_key = (
            int(obs_dim), int(hole_count), bool(_pin),
            obs_layout.name if obs_layout is not None else "dense",
        )
        cached = _STAGING_BUFFERS.get(stage_key) if reuse else None
        if cached is not None and cached[2] >= total_cap:
            staging, big, total_cap = cached
        else:
            staging = _SlabAllocator(obs_dim, hole_count, _pin, layout=obs_layout)
            big = staging.alloc(total_cap)
            if reuse:
                _STAGING_BUFFERS[stage_key] = (staging, big, total_cap)
        base = 0

        def _grow(used_rows: int, min_rows: int) -> dict[str, np.ndarray]:
            """`out_slabs_grow` for the sub running at `base`: it has written
            `used_rows` and needs `min_rows`. Everything below
            `base + used_rows` is live and is copied across."""
            nonlocal big, total_cap
            total_cap = max(
                base + int(min_rows), total_cap + max(total_cap // 4, 1)
            )
            big = staging.grow(big, base + int(used_rows), total_cap)
            if reuse:
                _STAGING_BUFFERS[stage_key] = (staging, big, total_cap)
            return {key: arr[base:] for key, arr in big.items()}

        # Per-sub scalar diagnostics only: a sub's tensors are VIEWS of `big`,
        # and holding them would pin a superseded buffer alive after a grow.
        subs = []
        sub_rows: list[int] = []
        for cfg in configs:
            views = {key: arr[base:] for key, arr in big.items()}
            sub = collect_rollout_batched(
                learner, pool, cfg, sub_config, rng,
                critic=critic, out_slabs=views,
                snapshot_cache=snapshot_cache,
                env_cache=env_cache,
                drain_inflight=drain,
                out_slabs_grow=_grow,
            )
            rows = int(sub.obs.shape[0])
            # Chain the next sub's base to this sub's actual row count so the
            # buffer's first `total` rows reproduce torch.cat's layout exactly.
            assert base + rows <= total_cap, (
                f"shared-staging overflow: base={base} rows={rows} "
                f"total_cap={total_cap}"
            )
            _empty = torch.empty(0)
            subs.append(replace(
                sub,
                **{f: _empty for f in _BATCH_TENSOR_FIELDS},
                is_terminal=None,
            ))
            sub_rows.append(rows)
            del sub, views
            base += rows
        total = base

        def _t(key: str) -> torch.Tensor:
            return torch.from_numpy(big[key][:total])

        adv = _t("adv")
        if n > 1:
            # Pooled advantage re-normalization — the identical ops
            # _concat_batches applies (and, like it, skipped when there is
            # only one sub-rollout). Each sub is already unit-normalized, so
            # this is numerically a no-op: normalization is PER CONFIG (see
            # the _concat_batches docstring; review 2026-09-20 A5).
            adv = (adv - adv.mean()) / adv.std().clamp(min=1e-8)
            if adv_clip > 0.0:
                adv = adv.clamp(-adv_clip, adv_clip)

        def _sum3(field: str) -> tuple[int, int, int]:
            vals = [int(sum(getattr(b, field)[i] for b in subs)) for i in range(3)]
            return (vals[0], vals[1], vals[2])

        combined = Batch(
            obs=_obs_from_slabs(big, total, obs_layout),
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
                    (rows,),
                    float(tier_ent[t]),
                    dtype=torch.float32,
                )
                for rows, t in zip(sub_rows, config_tiers)
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
    os.environ.pop("PLO5BP_STEP_TIMERS_OWNED", None)
    if _prev_owned is not None:
        os.environ["PLO5BP_STEP_TIMERS_OWNED"] = _prev_owned
    _parent_timers.report(label="collect_rollout_multiconfig")
    _ACTIVE_STEP_TIMERS = None

    return _batch_to_device(combined, device)


def _minibatch_bounds(n: int, batch_size: int) -> list[tuple[int, int]]:
    """[start, stop) slices of the shuffled index for one epoch.

    PRODUCTION BEHAVIOR CHANGE, small (review 2026-09-20 A10): a tail
    SMALLER THAN HALF a batch is folded back into the full minibatches
    instead of becoming its own step. The collectors always overshoot the
    row target a little, and train.py derives batch_size =
    ceil(rollout_length / num_minibatches), so every epoch ended in a runt
    (num_minibatches+1)-th minibatch — a few hundred rows that still took a
    full-LR optimizer step and its own KL-guard check on a very noisy
    gradient (e.g. 96,297 rows at batch_size 6,000 -> 16 full minibatches +
    a 297-row 17th; now 16 minibatches of 6,018/6,019 rows).

    The folded rows are spread EVENLY over the full minibatches rather than
    stacked on the last one: the largest minibatch is then
    batch_size * (1 + <0.5/k), not 1.5x — matters on the pod, where
    activation memory scales with the minibatch and sits near the VRAM
    ceiling. Tails >= half a batch, and rollouts shorter than one batch,
    keep the old slicing exactly."""
    n_full, tail = divmod(n, batch_size)
    if n_full == 0 or tail == 0 or 2 * tail >= batch_size:
        return [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]
    base, extra = divmod(n, n_full)
    bounds: list[tuple[int, int]] = []
    start = 0
    for k in range(n_full):
        stop = start + base + (1 if k < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def iter_minibatches(
    batch: Batch, batch_size: int, rng: np.random.Generator
) -> Iterable[Batch]:
    n = batch.obs.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    device = batch.obs.device
    for start, stop in _minibatch_bounds(n, batch_size):
        sel = torch.from_numpy(idx[start:stop]).to(device)
        yield Batch(
            # Compact storage unpacks HERE, per minibatch, on the learner
            # device (PackedObs row indexing yields dense f32, bit-exact).
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
