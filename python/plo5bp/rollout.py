"""Rollout collection for PPO training with the hybrid (gate + anchor
sizing) policy head.

Each stored transition carries:
  - the 3-wide gate mask (legal gate actions)
  - the sampled gate index
  - the raise chip delta (0 when gate != Raise)
  - the (min_raise, max_raise, pot, to_call) sizing context — the
    anchor grid / brackets are a pure function of it, so evaluate()
    recomputes masks instead of storing them
  - the sampled anchor index and refinement u (sampled on every row; only a
    Raise uses them)
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

from plo5bp.engine_abi import functions as _engine_functions

# The engine's per-step rollout kernels (numpy references of the first three:
# rollout_reference.py). A stale engine without them is an import error
# (engine_abi), never a silent fallback.
(
    _rust_flush_trajectories,
    _rust_record_learner_steps,
    _rust_aggression_record,
    _rust_gather_rows_multi,
) = _engine_functions(
    "flush_trajectories", "record_learner_steps", "aggression_record_batch",
    "gather_rows_multi",
)

# Below this many rows the per-step copies stay in numpy (tests force the
# engine path with 1). The engine call itself runs small copies on the
# calling thread and only wakes its thread pool for big ones.
_GATHER_RUST_MIN_ROWS = 1


def _gather_rows_multi(
    pairs: "list[tuple[np.ndarray, np.ndarray]]",
    rows: "np.ndarray | None",
    start: int = 0,
) -> None:
    """``dst[start:start + k] = src[rows]`` for every (src, dst) pair
    (``src[:k]``, k = len(src), when `rows` is None) -- the rollout's
    per-step host copies: packed observation rows into the pinned upload
    slots and the trajectory pool, gate-mask and sizing rows. One engine
    call for all the pairs (2-D, C-contiguous, one dtype per pair); numpy
    otherwise. The bytes are the same either way."""
    k = int(pairs[0][0].shape[0]) if rows is None else int(rows.shape[0])
    if (
        k >= _GATHER_RUST_MIN_ROWS
        and all(
            s.dtype == d.dtype
            and s.ndim == 2
            and d.ndim == 2
            and s.shape[1] == d.shape[1]
            and s.flags.c_contiguous
            and d.flags.c_contiguous
            for s, d in pairs
        )
    ):
        _rust_gather_rows_multi(
            [s.view(np.uint8) for s, _ in pairs],
            [d.view(np.uint8) for _, d in pairs],
            int(start),
            None if rows is None else np.ascontiguousarray(rows, dtype=np.int64),
        )
        return
    for src, dst in pairs:
        if rows is None:
            dst[start : start + k] = src
        else:
            np.take(src, rows, axis=0, out=dst[start : start + k])


def _gather_rows(
    src: np.ndarray, rows: "np.ndarray | None", dst: np.ndarray, start: int = 0
) -> None:
    """`_gather_rows_multi` for one array pair."""
    _gather_rows_multi([(src, dst)], rows, start)

# Output slabs the Rust flush writes (compact observation storage; dense
# storage: `_BatchedCollector._flush_out`).
def _rust_flush_out(slabs: dict, lo: int, hi: int) -> dict:
    """The Rust flush's output views of rows [lo, hi): a float16 obs_real slab
    (obs_real_f16) goes as its uint16 view under "obs_real_f16"."""
    out = {key: slabs[key][lo:hi] for key in _RUST_FLUSH_OUT_KEYS}
    if out["obs_real"].dtype == np.float16:
        out["obs_real_f16"] = out.pop("obs_real").view(np.uint16)
    return out


_RUST_FLUSH_OUT_KEYS = (
    "obs_bits", "obs_real", "gm", "ga", "rc", "sz", "an", "ru", "oh",
    "lp", "glp", "alp", "v", "ret", "adv", "last",
)
from plo5bp.actions import ALL_IN, GATE_ACTIONS, GATE_CHECK_CALL, GATE_RAISE
from plo5bp.compact_obs import (
    CompactObsLayout,
    PackedObs,
    layout_for,
    pack_rows_into,
)
from plo5bp.compact_obs import unpack as _unpack_compact
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
# Set by collect_rollout_multiconfig (inside try/finally) while its
# sub-collections run: they add to these timers and the parent reports once.
# A module variable, not an environment flag (2026-09-28, ML-046: the old
# PLO5BP_STEP_TIMERS_OWNED env var stayed set for the rest of the process when
# a sub-rollout raised).
_STEP_TIMERS_PARENT: "_StepTimers | None" = None


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
        "enabled", "device", "cap", "n_slots", "layout",
        "obs_h", "bits_h", "real_h", "gm_h", "sz_h", "_h2d_events",
        "_packed_k",
    )

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        device: torch.device,
        n_slots: int = 2,
        layout: CompactObsLayout | None = None,
    ) -> None:
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.cap = int(capacity)
        self.n_slots = max(1, int(n_slots))
        self.enabled = self.device.type == "cuda" and self.cap > 0
        # Compact transport (see upload_rows): the slots hold PACKED rows.
        self.layout = layout
        self._h2d_events: list = [None] * self.n_slots
        # Row count of each slot's last PACKED upload (None = dense / none).
        self._packed_k: list = [None] * self.n_slots
        self.obs_h = self.bits_h = self.real_h = None  # type: ignore[assignment]
        if not self.enabled:
            self.gm_h = self.sz_h = None  # type: ignore[assignment]
            return
        # Shape (n_slots, cap, ...) — one pin bank per slot.
        if self.layout is not None:
            self.bits_h = torch.empty(
                (self.n_slots, self.cap, self.layout.n_bytes),
                dtype=torch.uint8,
                pin_memory=True,
            )
            self.real_h = torch.empty(
                (self.n_slots, self.cap, self.layout.n_real),
                dtype=torch.float32,
                pin_memory=True,
            )
        else:
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
        obs_np = np.ascontiguousarray(b_obs, dtype=np.float32)
        if self.layout is not None:
            return self._upload_packed(
                obs_np, np.arange(k, dtype=np.int64), b_gm, b_sizing, slot
            )
        s = int(slot) % self.n_slots
        self._packed_k[s] = None
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

    def upload_rows(
        self,
        obs_arr: np.ndarray,
        rows: np.ndarray,
        gm_arr: np.ndarray,
        sizing_arr: np.ndarray,
        slot: int = 0,
        packed: "tuple[np.ndarray, np.ndarray] | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``upload(obs_arr[rows], gm_arr[rows], sizing_arr[rows], slot)``
        without the dense host gather: under a compact layout the rows are
        packed straight out of `obs_arr` into the pinned slot (Rust,
        parallel) -- or, with `packed` = (bits, real), the packed copy of
        `obs_arr` the env's encoder keeps (BatchedBombPotEnv.packed_obs),
        simply gathered -- cross PCIe packed (~7x fewer bytes than float
        rows) and are unpacked on the device into the IDENTICAL float32 rows
        (compact_obs.unpack is the exact inverse of the packer)."""
        rows = np.asarray(rows, dtype=np.int64)
        k = int(rows.size)
        if (
            self.layout is None
            or not self.enabled
            or k == 0
            or k > self.cap
            or obs_arr.dtype != np.float32
            or not obs_arr.flags.c_contiguous
        ):
            return self.upload(
                obs_arr[rows], gm_arr[rows], sizing_arr[rows], slot=slot
            )
        if gm_arr.dtype == np.bool_ and sizing_arr.dtype == np.int64:
            return self._upload_packed(
                obs_arr, rows, gm_arr, sizing_arr, slot, packed, aux_rows=rows
            )
        return self._upload_packed(
            obs_arr, rows, gm_arr[rows], sizing_arr[rows], slot, packed
        )

    def _upload_packed(
        self,
        obs_arr: np.ndarray,
        rows: np.ndarray,
        b_gm: np.ndarray,
        b_sizing: np.ndarray,
        slot: int,
        packed: "tuple[np.ndarray, np.ndarray] | None" = None,
        aux_rows: "np.ndarray | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`aux_rows` None: `b_gm` / `b_sizing` are the k rows already; else
        they are the full per-env arrays and `aux_rows` picks the k rows."""
        k = int(rows.size)
        s = int(slot) % self.n_slots
        self._packed_k[s] = None
        if packed is not None and aux_rows is not None:
            # Observation, gate-mask and sizing rows in ONE copy call.
            _gather_rows_multi(
                [
                    (packed[0], self.bits_h[s].numpy()),
                    (packed[1], self.real_h[s].numpy()),
                    (b_gm, self.gm_h[s].numpy()),
                    (b_sizing, self.sz_h[s].numpy()),
                ],
                rows,
            )
            self._packed_k[s] = k
        elif packed is not None:
            _gather_rows_multi(
                [(packed[0], self.bits_h[s].numpy()), (packed[1], self.real_h[s].numpy())],
                rows,
            )
            self._packed_k[s] = k
        else:
            pack_rows_into(
                obs_arr, rows, self.layout,
                self.bits_h[s].numpy(), self.real_h[s].numpy(), 0,
            )
            self._packed_k[s] = k
        if aux_rows is not None:
            if packed is None:
                _gather_rows_multi(
                    [(b_gm, self.gm_h[s].numpy()), (b_sizing, self.sz_h[s].numpy())],
                    aux_rows,
                )
        else:
            self.gm_h[s].numpy()[:k] = np.asarray(b_gm, dtype=bool)
            self.sz_h[s].numpy()[:k] = np.asarray(b_sizing, dtype=np.int64)
        bits_t = self.bits_h[s, :k].to(self.device, non_blocking=True)
        real_t = self.real_h[s, :k].to(self.device, non_blocking=True)
        m_t = self.gm_h[s, :k].to(self.device, non_blocking=True)
        b_t = self.sz_h[s, :k].to(self.device, non_blocking=True)
        if self._h2d_events[s] is None:
            self._h2d_events[s] = torch.cuda.Event()
        self._h2d_events[s].record()
        return _unpack_compact(bits_t, real_t, self.layout), m_t, b_t

    def packed_rows(
        self, slot: int, k: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Host views of `slot`'s packed rows and their gate-mask rows when
        its last upload was a PACKED one of exactly `k` rows, else None --
        the learner's act-time rows, reused for trajectory storage (the
        packer's bytes, so storing them equals packing the same rows again;
        the gate masks are the uploaded rows' own)."""
        s = int(slot) % self.n_slots
        if not self.enabled or self._packed_k[s] != int(k):
            return None
        return (
            self.bits_h[s].numpy()[:k],
            self.real_h[s].numpy()[:k],
            self.gm_h[s].numpy()[:k],
        )




@dataclass
class Batch:
    # (T, OBS_DIM) f32 — or its compact storage (compact_obs.PackedObs), whose
    # row indexing (obs[rows], as iter_minibatches does) yields dense f32 rows.
    obs: "torch.Tensor | PackedObs"
    gate_masks: torch.Tensor    # (T, GATE_ACTIONS) bool
    gate_actions: torch.Tensor  # (T,) long — sampled gate index
    raise_chips: torch.Tensor   # (T,) long — chip delta (0 for non-Raise)
    sizing: torch.Tensor        # (T, 4) long — (min, max, pot, to_call)
    anchor_actions: torch.Tensor  # (T,) long — sampled anchor (used on raise rows)
    refine_u: torch.Tensor      # (T,) f32 — sampled refinement u
    opp_holes: torch.Tensor     # (T, 5, hole_count) u8 — rotated opp holes
    log_probs: torch.Tensor     # (T,) f32 — sampling JOINT log-prob
    values: torch.Tensor        # (T,) f32 — critic at sampling time
    returns: torch.Tensor       # (T,) f32
    advantages: torch.Tensor    # (T,) f32 — normalized
    # Per-head sampling log-probs for per-head KL diagnostics. gate is
    # always meaningful; anchor is the raise-row anchor log-prob (0 on
    # other rows). beta_kl is derived as total_kl - gate_kl - anchor_kl.
    old_gate_logp: torch.Tensor    # (T,) f32
    old_anchor_logp: torch.Tensor  # (T,) f32
    # Terminal-row flag: True where this row is the seat's LAST decision of
    # the hand. At those rows `returns` equals the raw realized reward
    # exactly (the trace has no future term), so they carry free ground-truth
    # Q labels — the qT boundary-residual canary (V7_DESIGN.md WS1.3) reads
    # them. Populated by the BATCHED collector only; None on serial paths
    # (VRPO/Q-aux runs are batched-only, and the canary skips when None).
    is_terminal: torch.Tensor | None = None  # (T,) bool
    # F/T/R aggression telemetry (set by the rollout collectors; the names
    # are from the retired aggression bonuses, ML-030): `aggr_steps_total`
    # is the count of learner steps, `aggr_bonus_steps` the count of WINNING
    # aggression steps (a raise with > 50% of the final pot; with exactly
    # 50% also a call -- `_winning_aggression_steps`), and the *_by_street
    # tuples bucket both by street (0=flop, 1=turn, 2=river; bomb pots have
    # no preflop action). F/T/R = 100 * bonus_steps / steps per street (the
    # log's `aggr%`). `aggr_bonus_total_bb` is always 0 now (the bonuses
    # are retired).
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
    # Row spans of each stack tier in the combined batch (mix-configs only):
    # tier -> [(start, stop), ...] in batch row order. Metadata for per-tier
    # diagnostics (ppo.value_health); None on single-config batches.
    tier_rows: dict | None = None


def _build_frozen_model(state_dict: dict, device: torch.device) -> ActorCritic:
    # Class, size, obs width AND anchor spec are sniffed from the state dict
    # itself so pool snapshots rebuild exactly what was frozen across head
    # versions and variants — an NLH v4 snapshot has a 995-wide torso and a
    # 12-anchor ladder that the old hardcoded defaults shape-failed on.
    model = build_actor_from_state_dict(state_dict)
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


def _winning_aggression_steps(
    steps: list[tuple],
    costs: list[float],
    streets: list[int],
    payout_chips: int,
    total_pot_chips: int,
) -> list[int]:
    """Per-street (flop, turn, river) counts of one learner seat's WINNING
    aggression steps in a finished hand -- the F/T/R telemetry numerator:

    - share of the pot > 50%: every GATE_RAISE step;
    - share == 50%: also every GATE_CHECK_CALL step that committed chips (a
      call, not a check -- trajectory tuples store chips=0 for CHECK_CALL, so
      "committed chips" is `costs[t] < 0`);
    - share < 50%: none.

    `streets[t]` is the engine street (1 flop .. 3 river), bucketed as
    street - 1. The batched collector's flush kernel counts the same steps.
    (2026-09-28, ML-030: the retroactive aggression BONUS these steps once
    earned -- and the per-step aggression bonus -- are retired; every stem
    since vTwo ran them at 0. Only the count remains.)"""
    by_street: list[int] = [0, 0, 0]
    if total_pot_chips <= 0:
        return by_street
    two_pay = 2 * int(payout_chips)
    if two_pay < total_pot_chips:
        return by_street
    include_call = two_pay == total_pot_chips
    for t, step in enumerate(steps):
        gate_t = int(step[2])
        is_raise = gate_t == GATE_RAISE
        is_call_with_chips = (
            include_call and gate_t == GATE_CHECK_CALL and costs[t] < 0.0
        )
        if not (is_raise or is_call_with_chips):
            continue
        bucket = int(streets[t]) - 1
        if 0 <= bucket < 3:
            by_street[bucket] += 1
    return by_street


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
    (which sees all hole cards); otherwise the actor's own (display) value
    head is used (UI / eval / profiling callers without a critic).

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
        opp_models[env_idx] = _build_frozen_model(sd, device)
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
    # Parallel to trajectories: pre-step street index (0..3). Used by
    # `_winning_aggression_steps` to bucket the per-street F/T/R counts.
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

    # F/T/R telemetry: learner steps and winning-aggression steps
    # (`_winning_aggression_steps`), per street. 3 buckets: 0=flop, 1=turn,
    # 2=river (bomb pots have no preflop action).
    aggr_steps_total = 0
    aggr_bonus_steps = 0
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

            next_obs, rewards, done, next_info = env.step_hybrid(gate, chips)

            # Forward-EV per-step cost: chips the actor put in THIS step,
            # signed negative, in bb units. Only learner-seat acts feed the
            # trajectory. `0.0 - x` (not `-x`): a check's cost is +0.0, as
            # it was when the retired aggression bonus (always 0.0) was added.
            if actor is not None and actor in learner_seats[i]:
                delta = float(next_info.commit_delta[actor])
                _street_pre = int(info.raw_obs.get("street", 0))
                cost_trajs[i][actor].append(0.0 - delta * reward_norm)
                street_trajs[i][actor].append(_street_pre)
                aggr_steps_total += 1
                _bucket = _street_pre - 1  # 1=flop → bucket 0; 3=river → 2.
                if 0 <= _bucket < 3:
                    aggr_steps_total_by_street[_bucket] += 1

            if done:
                # Gross winnings = payouts + total_commit (recovers the
                # `won` component the engine subtracted out for chip-delta).
                won = rewards + next_info.total_commit.astype(np.float32)
                total_pot_chips = int(
                    np.asarray(next_info.total_commit, dtype=np.int64).sum()
                )
                for seat in learner_seats[i]:
                    bumped_by_street = _winning_aggression_steps(
                        trajectories[i][seat],
                        cost_trajs[i][seat],
                        street_trajs[i][seat],
                        int(round(float(won[seat]))),
                        total_pot_chips,
                    )
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
# throughput). The ~1-3% extra MC noise on the MC-sampled opp-outcome dims
# (982-989 of the 1171-dim full layout; minimal obs has none) is a benign,
# fine-tune-safe regularizer; the UI/eval/serial paths keep 1024 so the
# study tool still computes the more accurate estimate.
# 2026-06-20: MC=256 destabilized the gate in the shallow clubgg block
# (vTwo10 saturated-collapsed at u490, in clubgg, even after an LR cut to
# 1.5e-4). Raised to 384 as a less-noisy compromise — still cheaper than
# the 1024 UI/eval path, but with ~1/3 less MC variance on those dims.
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
# CURRENT collection wrote (flush masks, the flush kernel's per-seat
# lengths, `[:wcursor]` views), and every buffer is created zero-filled, so a
# stale value is always finite. Pinned by tests/python/training/test_buffer_reuse.py.
_TRAJ_BUFFERS: dict[tuple, dict] = {}      # (n_envs, n_seats, vrpo) -> {"cap", <name>: array}
_OBS_POOL_BUFFERS: dict[tuple, dict] = {}  # (obs_dim, layout) -> {"cap", "obs", "gm"}
_STAGING_BUFFERS: dict[tuple, tuple] = {}  # (obs_dim, hole, pin, layout) -> (allocator, big, cap)


# Called (no arguments) right before a finished rollout batch is copied to
# the learner device. scripts/train.py --gpu-lock takes its cross-process lock
# here so runs sharing one GPU never hold two batches + PPO working sets at
# once (released after the update). None = no-op; never touches any value.
GPU_PHASE_HOOK: "Callable[[], None] | None" = None


def _enter_gpu_phase() -> None:
    hook = GPU_PHASE_HOOK
    if hook is not None:
        hook()


def _flat_traj_view(arr: np.ndarray) -> np.ndarray:
    """A (n_envs, n_seats, cap[, w]) trajectory array as a flat
    (n_envs * n_seats * cap[, w]) VIEW -- writes through it land in `arr`, so
    one precomputed slot index (env * n_seats + seat) * cap + slot replaces
    three-array fancy indexing on every read and write."""
    assert arr.flags.c_contiguous, "trajectory arrays are allocated C-contiguous"
    v = arr.reshape((-1,) + arr.shape[3:])
    assert v.ctypes.data == arr.ctypes.data  # a view, never a copy
    return v


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
        # float16 under TrainingConfig.obs_real_f16 (the output slabs only;
        # the per-step pool and the act-time uploads stay float32).
        "obs_real": ((layout.n_real,), layout.real_np_dtype, layout.real_torch_dtype),
    }


def _slab_keys(layout: CompactObsLayout | None) -> tuple[str, ...]:
    return (("obs",) if layout is None else ("obs_bits", "obs_real")) + _SLAB_KEYS


def _alloc_obs_pool(
    cap: int, obs_dim: int, layout: CompactObsLayout | None
) -> dict[str, np.ndarray]:
    """The per-step observation pool (rows appended at decision time, gathered
    into the output slabs at hand end), in the slabs' observation layout."""
    pool_layout = layout.pool_layout if layout is not None else None
    return {
        key: np.empty((int(cap), *shape), dtype=np_dtype)
        for key, (shape, np_dtype, _td) in _obs_spec(obs_dim, pool_layout).items()
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


def _resolve_obs_layout(
    train_config: TrainingConfig, variant: str
) -> CompactObsLayout | None:
    """Compact observation storage for a collection (compact_obs.py), or None
    = dense float32 rows (NLH, or --no-compact-obs). ONE resolver for the
    collector's own pool/slabs and multiconfig's shared staging, so the two
    always agree. Storage only: the rows unpack bit-exactly, so batches, RNG
    and training are unchanged."""
    if not bool(getattr(train_config, "compact_obs", True)):
        return None
    return layout_for(
        variant,
        str(getattr(train_config, "obs_mode", "full")),
        real_f16=bool(getattr(train_config, "obs_real_f16", False)),
    )


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


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """splitmix64's finalizer over uint64 arrays (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


class _CrnDeals:
    """Common random numbers for one sub-rollout (TrainingConfig.crn_streams,
    2026-09-28, ML-004): env j's k-th hand -- its deal seed, button and
    opponent assignment -- is a pure function of (key, j, k), key = (run seed,
    update, sub-rollout). Two runs that differ only in their policies (a recipe
    wave's candidates) then play the SAME hands in every env, whatever their
    hand lengths. The shared stream draws n_envs seeds per re-deal WAVE, so
    env j's k-th hand depends on how many waves came before it -- i.e. on the
    policy -- and runs decorrelate from their first hand end on.

    Counter-based: bits = splitmix64(base ^ splitmix64(env << 32 | hand << 8 |
    field)), `base` from SeedSequence(key); one call per wave, vectorized."""

    _SEED, _BUTTON, _MIX, _SNAP, _SEATS = 0, 1, 2, 3, 8  # fields; seats 8..8+S

    def __init__(self, key: "tuple[int, ...]", n_envs: int) -> None:
        self.base = np.random.SeedSequence([int(k) for k in key]).generate_state(
            1, dtype=np.uint64
        )[0]
        self.hand = np.zeros(int(n_envs), dtype=np.uint64)  # hands dealt, per env

    def _bits(self, env_ids: np.ndarray, field: int) -> np.ndarray:
        """64 random bits of each env's CURRENT hand (its next deal), `field`."""
        x = (
            (env_ids.astype(np.uint64) << np.uint64(32))
            | (self.hand[env_ids] << np.uint64(8))
            | np.uint64(field)
        )
        return _splitmix64(self.base ^ _splitmix64(x))

    def _uniform(self, env_ids: np.ndarray, field: int) -> np.ndarray:
        return (self._bits(env_ids, field) >> np.uint64(11)).astype(np.float64) * (
            1.0 / float(1 << 53)
        )

    def deal(self, env_ids: np.ndarray, n_seats: int) -> "tuple[np.ndarray, np.ndarray]":
        """(seeds u64 in [0, 2^63), buttons u8) of each env's next hand; the
        envs' hand counters advance."""
        env_ids = np.asarray(env_ids, dtype=np.int64)
        seeds = self._bits(env_ids, self._SEED) & np.uint64((1 << 63) - 1)
        buttons = (self._bits(env_ids, self._BUTTON) % np.uint64(n_seats)).astype(np.uint8)
        self.hand[env_ids] += np.uint64(1)
        return seeds, buttons

    def pool_mix(
        self,
        env_ids: np.ndarray,
        n_seats: int,
        pool_size: int,
        pool_opp_seats: int,
        pool_mix_prob: float,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """`_draw_pool_mix`'s distribution for each env's next hand, from that
        hand's own numbers (call it before `deal`)."""
        env_ids = np.asarray(env_ids, dtype=np.int64)
        k = int(env_ids.size)
        snap = np.full(k, -1, dtype=np.int64)
        mask = np.ones((k, n_seats), dtype=bool)
        if k == 0 or pool_size == 0 or pool_opp_seats == 0 or pool_mix_prob <= 0.0:
            return snap, mask
        mixed = np.nonzero(self._uniform(env_ids, self._MIX) < pool_mix_prob)[0]
        if mixed.size:
            ids = env_ids[mixed]
            snap[mixed] = (self._bits(ids, self._SNAP) % np.uint64(pool_size)).astype(np.int64)
            keys = np.stack(
                [self._uniform(ids, self._SEATS + s) for s in range(n_seats)], axis=1
            )
            opp = np.argsort(keys, axis=1)[:, :pool_opp_seats]
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
        need_log_probs: bool = False,
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
        # Opponent rows are never trained on: by default skip the log-prob
        # tail (same samples and RNG stream -- see _act_from_heads).
        return self.template._act_from_heads(
            *(h[slot_g, slot_j] for h in heads), sizing,
            deterministic=deterministic, need_log_probs=need_log_probs,
        )


class _OpponentCache(dict):
    """The frozen pool opponents of ONE update (P5): pool snapshot index ->
    built model (the dict itself), plus the `_StackedOpponents` built from
    them (`stacked`, for the models `stacked_key` names). Pool membership is
    frozen for the whole update (the caller snapshots AFTER trainer.update),
    so every sub-rollout of it shares both: each member is built once per
    update instead of once per sub-rollout, and the snapshots' weights are
    stacked once instead of 30 times (2026-09-28, ML-043; stacking copies the
    weights, so the stacked forward is bit-identical). A plain dict still
    works as `snapshot_cache` (the stack is then rebuilt per collection)."""

    def __init__(self) -> None:
        super().__init__()
        self.stacked_key: "tuple | None" = None
        self.stacked: "_StackedOpponents | None" = None


class _StepActs:
    """Every env's outputs of one collection step: gate + raise chips for
    learner AND opponent rows (what the engine applies), and for learner rows
    the anchor, refinement u, log-probs and value (+ VRPO's Q(s, a) and
    V^pi(s)). Allocated once per collection; `reset` restores, every step, the
    exact contents the per-step np.zeros / np.full used to allocate
    (2026-09-28, ML-043) -- nothing reads a previous step's values."""

    __slots__ = (
        "gate", "chips", "anchor", "u", "log_p", "gate_lp", "anchor_lp",
        "value", "q_taken", "vpi",
    )

    def __init__(self, n: int, vrpo: bool) -> None:
        self.gate = np.zeros(n, dtype=np.uint8)
        self.chips = np.zeros(n, dtype=np.uint64)
        self.anchor = np.full(n, -1, dtype=np.int64)
        self.u = np.zeros(n, dtype=np.float32)
        self.log_p = np.zeros(n, dtype=np.float32)
        self.gate_lp = np.zeros(n, dtype=np.float32)
        self.anchor_lp = np.zeros(n, dtype=np.float32)
        self.value = np.zeros(n, dtype=np.float32)
        self.q_taken = np.zeros(n, dtype=np.float32) if vrpo else None
        self.vpi = np.zeros(n, dtype=np.float32) if vrpo else None

    def reset(self) -> None:
        self.gate.fill(0)
        self.chips.fill(0)
        self.anchor.fill(-1)
        for a in (self.u, self.log_p, self.gate_lp, self.anchor_lp, self.value):
            a.fill(0.0)
        if self.q_taken is not None:
            self.q_taken.fill(0.0)
            self.vpi.fill(0.0)

    def per_env(self, sizing: np.ndarray) -> dict[str, np.ndarray]:
        """The `record_learner_steps` kernel's per-env input."""
        out = {
            "gate": self.gate, "chips": self.chips, "sizing": sizing,
            "anchor": self.anchor, "u": self.u, "log_p": self.log_p,
            "gate_lp": self.gate_lp, "anchor_lp": self.anchor_lp,
            "value": self.value,
        }
        if self.q_taken is not None:
            out["q_taken"] = self.q_taken
            out["vpi"] = self.vpi
        return out


@dataclass
class _Step:
    """What one step reads from the env before acting (`_BatchedCollector.
    _prestep`). The pre-step commit / street arrays are REFERENCES: the
    post-apply refresh replaces the env's cached arrays with fresh ones
    (asserted in `_apply`), so these keep the pre-step values."""

    obs: np.ndarray
    gate_masks: np.ndarray
    actors: np.ndarray
    dones: np.ndarray
    pre_total_commit: np.ndarray
    pre_bet_to_call: np.ndarray
    pre_street_commit: np.ndarray
    pre_street: np.ndarray
    active: np.ndarray        # ~dones
    safe_actors: np.ndarray   # actors with -1 clamped to 0 (masked by `active`)
    is_opp_active: np.ndarray
    learner_idx: np.ndarray   # envs whose actor is a learner seat
    sizing: np.ndarray        # (N, 4) (min_raise, max_raise, pot, to_call)


def _act_to_host(fw: ActOut) -> tuple:
    """Device ActOut -> host numpy with one sync per stack: (gate u8, chips
    i64, anchor i64, log_prob, refine_u, value, gate_log_prob,
    anchor_log_prob, action_marginal or None)."""
    ints = torch.stack((fw.gate, fw.chips, fw.anchor), dim=0).cpu().numpy()
    floats = torch.stack(
        (fw.log_prob, fw.refine_u, fw.value, fw.gate_log_prob, fw.anchor_log_prob),
        dim=0,
    ).cpu().numpy()
    marg_np = (
        fw.action_marginal.float().cpu().numpy()
        if fw.action_marginal is not None
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
    )


@dataclass(frozen=True)
class _Kernels:
    """The collector's per-step engine kernels (step3c record, step8 cost
    record, step9 flush) -- or their numpy references (rollout_reference.py),
    same signatures, under PLO5BP_NUMPY_FLUSH=1."""

    record: Callable
    aggression: Callable
    flush: Callable


_RUST_KERNELS = _Kernels(
    _rust_record_learner_steps, _rust_aggression_record, _rust_flush_trajectories
)


def _kernels() -> _Kernels:
    if os.environ.get("PLO5BP_NUMPY_FLUSH", "0") == "1":
        from plo5bp import rollout_reference as ref

        return _Kernels(
            ref.record_learner_steps, ref.aggression_record_batch, ref.flush_trajectories
        )
    return _RUST_KERNELS


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
    crn_key: "tuple[int, ...] | None" = None,
) -> Batch:
    """Batched rollout using `BatchedBombPotEnv` + snapshot-bucket
    opponent forwards. Drives all envs through `apply_hybrid_batch`; the work
    itself is `_BatchedCollector`.

    `crn_key` (TrainingConfig.crn_streams, default None = the shared `rng`
    stream): deals, buttons and opponent assignments come from `_CrnDeals`
    keyed by it -- env j's k-th hand is the same in every run with that key.

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
    models, keyed by pool snapshot index. Multiconfig passes one
    `_OpponentCache` for the whole update (pool membership is frozen across
    it — the caller snapshots AFTER trainer.update), so each pool member is
    constructed (and the members stacked) once per update instead of once per
    sub-rollout. Default None = a fresh local dict. CUDA-run bit-exact (module
    init consumes the CPU torch generator; batched sampling uses the CUDA
    generator); on CPU-only runs the shared cache shifts the sampling stream
    from the second sub-rollout on (no batched bit-exact contract — parity is
    at the env level).

    `out_slabs` (P7+P8, multiconfig shared staging): caller-provided numpy
    VIEWS — one per output slab (`_slab_keys`) — into one big preallocated
    host buffer. When given, the collector writes into them instead of
    allocating its own slabs and returns a Batch of CPU view-tensors (no
    device copy except the tiny per-sub advantage-normalization hop, which
    stays on the learner device for bit-exactness with the legacy path).
    Default None = the self-allocating, finalize-to-device path. The views'
    length IS the capacity; `out_slabs_grow(used_rows, min_rows)` (optional)
    must return replacement views of >= `min_rows` rows whose first
    `used_rows` rows are preserved — called when a flush would overflow.
    Without it an overflow raises."""
    global _ACTIVE_STEP_TIMERS
    # Under a multiconfig parent its timers aggregate every sub; a standalone
    # call owns fresh ones (a stale _ACTIVE_STEP_TIMERS left behind by a call
    # that raised can no longer be mistaken for a parent).
    owns_timers = _STEP_TIMERS_PARENT is None
    step_timers = _StepTimers() if owns_timers else _STEP_TIMERS_PARENT
    _ACTIVE_STEP_TIMERS = step_timers
    batch = _BatchedCollector(
        learner, pool, game_config, train_config, rng,
        critic=critic, out_slabs=out_slabs, snapshot_cache=snapshot_cache,
        env=env, env_cache=env_cache, drain_inflight=drain_inflight,
        out_slabs_grow=out_slabs_grow, step_timers=step_timers, crn_key=crn_key,
    ).run()
    # Report only when this call owns the timers (not under a multiconfig
    # parent, which reports once for all its subs).
    if owns_timers:
        step_timers.report(label="collect_rollout_batched")
        _ACTIVE_STEP_TIMERS = None
    return batch


class _BatchedCollector:
    """One `collect_rollout_batched` collection (2026-09-28, ML-007: split out
    of a 1,700-line function built from ~20 closures). `__init__` is the setup
    (env, opponents, the first deal, trajectory / pool / output buffers);
    `run` steps every env until the row target (+ the A6 drain) and
    finalizes. One method per step phase:

        _prestep -> _act -> _record -> _apply -> _record_costs
                 -> (hands over) _payouts -> _flush -> _redeal

    Bit-exact with the single function it replaced: the same operations in
    the same order and the same RNG draws -- numpy: config deals, buttons,
    pool-mix (only ever in `__init__` and `_redeal` / `_redeal_done_at_deal`);
    torch: opponent model builds (`_snapshot_model`), then per step the
    learner's then the opponents' sampling."""

    # Per-(env, seat) trajectory length cap for one hand. The worst case is a
    # 1bb-increment min-raise war (engine floors bets at 1bb and raise
    # increments at the last raise size): a seat commits >=2bb per aggressive
    # action, so a 300bb stack (the --stack-range default cap) tops out near
    # ~150 actions per seat per hand. 32 was exceeded in practice on a
    # deep-tier rollout (vTwo1 update 94, 2026-06-10); 192 bounds the
    # theoretical worst case with margin. The arrays start at
    # `_TRAJ_CAP_INIT` slots and DOUBLE on demand up to this cap
    # (`_grow_traj`; exact — slots at or past a seat's length are never
    # read): a hand uses < ~16 decisions per seat, so the old fixed 192-slot
    # allocation zero-filled ~735 MB per vMin1 sub-rollout for slots that
    # stay empty (2026-09-23). The obs pool / output slabs are sized by
    # `_slack_per_env` (growing on demand), NOT this capacity, and flush
    # temporaries are bounded by the longest trajectory per flush.
    MAX_STEPS_PER_SEAT = 192

    def __init__(
        self,
        learner: ActorCritic,
        pool: OpponentPool,
        game_config: GameConfig,
        train_config: TrainingConfig,
        rng: np.random.Generator,
        *,
        critic: CentralCritic | None,
        out_slabs: "dict[str, np.ndarray] | None",
        snapshot_cache: "dict[int, ActorCritic] | None",
        env: "BatchedBombPotEnv | None",
        env_cache: "dict[tuple, BatchedBombPotEnv] | None",
        drain_inflight: "bool | None",
        out_slabs_grow: "Callable[[int, int], dict[str, np.ndarray]] | None",
        step_timers: _StepTimers,
        crn_key: "tuple[int, ...] | None" = None,
    ) -> None:
        self.learner = learner
        self.pool = pool
        self.game_config = game_config
        self.train_config = train_config
        self.rng = rng
        self.critic = critic
        self.timers = step_timers
        n_envs = self.n_envs = train_config.num_envs
        self.drain = _resolve_drain_inflight(train_config, drain_inflight)
        # Wall time from here to the first loop iteration: env build/reuse, the
        # per-sub trajectory arrays + obs pool allocations, the first deal/refresh.
        step_timers.begin("step0/setup")
        n_seats = self.n_seats = game_config.num_seats
        self.reward_norm = 1.0 / float(game_config.bb)
        self.gamma = train_config.gamma
        self.lam = train_config.lam
        self.pool_mix_prob = float(train_config.pool_mix_prob)
        self.pool_opp_seats = max(0, min(int(train_config.pool_opp_seats), n_seats - 1))
        self.device = device = next(learner.parameters()).device

        # VRPO / Expected-SARSA advantage flip (V5_DESIGN.md W2.5). Needs the
        # centralized critic's dueling Q head; train.py additionally requires
        # q_aux_coef>0 so the head is trained (else the flip is identical to GAE).
        self.use_vrpo = getattr(train_config, "advantage_estimator", "gae") == "vrpo"
        if self.use_vrpo and (critic is None or getattr(critic, "q_actions", 0) <= 0):
            raise ValueError(
                "advantage_estimator='vrpo' needs a CentralCritic with a dueling "
                "Q head (q_actions>0)."
            )

        env = self.env = self._make_env(env, env_cache)

        # P5: reuse the caller's per-update cache when given (multiconfig), else a
        # local one (single-config callers).
        self.snapshot_models: dict[int, ActorCritic] = (
            snapshot_cache if snapshot_cache is not None else {}
        )
        # (n_envs, n_seats) bool — `True` where seat is a learner seat in that
        # env. Set per deal by `_draw_pool_mix`; drives the vectorized "is this
        # actor a learner seat?" lookups in the hot loop.
        self.learner_seats_mask = np.ones((n_envs, n_seats), dtype=bool)
        # (n_envs,) i64 — pool snapshot index per env, or -1 for self-play.
        self.env_snapshot_idx = np.full(n_envs, -1, dtype=np.int64)
        # Eager-build every pool member once up front (cheap if snapshot_cache
        # already warm from multiconfig). Avoids first-use build mid-hand when a
        # late terminal re-mix draws a not-yet-seen index. Models stay on `device`
        # for the whole sub-rollout / update (P5); no cross-update cache.
        for sd in range(len(pool.snapshots)):
            self._snapshot_model(sd)
        all_envs = np.arange(n_envs, dtype=np.int64)
        # Common random numbers (ML-004): per-(env, hand) deals and opponent
        # assignments instead of the shared `rng` stream.
        self.crn = _CrnDeals(crn_key, n_envs) if crn_key is not None else None
        self._assign_pool_mix(all_envs)
        self.stacked_opp = self._stacked_opponents(snapshot_cache)

        if self.crn is not None:
            init_seeds, init_buttons = self.crn.deal(all_envs, n_seats)
        else:
            init_seeds = rng.integers(
                0, 2**63 - 1, size=n_envs, dtype=np.int64
            ).astype(np.uint64)
            init_buttons = rng.integers(
                0, n_seats, size=n_envs, dtype=np.int64
            ).astype(np.uint8)
        # snapshot=False: the collector reads the env's arrays in place (the
        # snapshot copied the whole observation buffer -- 186 MB per sub-rollout
        # at 58k envs -- only to be dropped).
        env.reset_batch(init_seeds, init_buttons, snapshot=False)
        self._redeal_done_at_deal(all_envs)
        # Per-hand hole cache (holes are static within a hand): one bulk
        # fetch per reset wave feeds the critic input + stored opp blocks.
        self.holes_cache = np.asarray(env._be.all_hole_cards_batch(), dtype=np.uint8)
        # Hero-rotated opp-hole blocks per (env, seat), filled on deal/reset;
        # the flush and the critic index it. Reused across sub-rollouts /
        # updates of the same shape: the full-batch fill below rewrites every
        # element before anything reads it.
        self.hole_w = int(game_config.hole_count)
        hr_key = (int(n_envs), int(n_seats), self.hole_w)
        self.holes_rot_cache = _HOLES_ROT_BUFFERS.get(hr_key)
        if self.holes_rot_cache is None:
            self.holes_rot_cache = np.full(
                (n_envs, n_seats, 5, self.hole_w), 255, dtype=np.uint8
            )
            _HOLES_ROT_BUFFERS[hr_key] = self.holes_rot_cache
        self._fill_holes_rot(all_envs)

        # Per-(env, seat, slot) trajectory arrays (`_TRAJ_SPEC`), REUSED across
        # sub-rollouts / updates with the same (n_envs, n_seats, estimator) —
        # see _TRAJ_BUFFERS; only the per-seat lengths restart at zero (stale
        # slots past them are never read). Every per-step write goes through
        # the flat views (`traj_flat`, one slot index per row).
        traj_key = (int(n_envs), int(n_seats), bool(self.use_vrpo))
        self.traj_buf = _TRAJ_BUFFERS.get(traj_key)
        if self.traj_buf is None:
            cap0 = min(_TRAJ_CAP_INIT, self.MAX_STEPS_PER_SEAT)
            self.traj_buf = {"cap": cap0, **_alloc_traj(n_envs, n_seats, cap0, self.use_vrpo)}
            _TRAJ_BUFFERS[traj_key] = self.traj_buf
        self.traj_cap = int(self.traj_buf["cap"])
        self.traj_lengths = np.zeros((n_envs, n_seats), dtype=np.int32)
        self._bind_traj()

        # Flat pre-allocated obs / gate-mask pool. Each step appends the
        # learner rows at `pool_cursor`; trajectory slots store the absolute
        # pool index, so the terminal flush gathers with one index array.
        self.rollout_target = int(train_config.rollout_length)
        # Slack for learner steps written past the rollout target: with
        # drain_inflight every in-flight hand's rows (flushed as those hands
        # finish), without it the abandoned in-flight hands' unflushed rows
        # (pool only). A per-env STATISTICAL figure (~one hand's rows) and the
        # INITIAL capacity only: an undersized guess GROWS (one copy,
        # `_grow_pool` / `_grow_slabs`) instead of raising, and `_slack_per_env`
        # learns the real need so later collections start right-sized (review
        # 2026-09-20 A6). Oversizing costs only VIRTUAL address space.
        pool_cap = self.rollout_target + n_envs * _slack_per_env()
        # Compact observation storage (compact_obs.py; None = dense float32).
        # The pool and the output slabs share the layout; multiconfig's shared
        # staging uses the same resolver, so the `out_slabs` it hands over
        # match too. The pool is REUSED across sub-rollouts / updates
        # (_OBS_POOL_BUFFERS); only rows below `pool_cursor` are ever read.
        self.obs_layout = _resolve_obs_layout(train_config, game_config.variant)
        pool_key = (
            int(env.obs_dim),
            self.obs_layout.name if self.obs_layout is not None else "dense",
        )
        self.pool_buf = _OBS_POOL_BUFFERS.get(pool_key)
        if self.pool_buf is None or self.pool_buf["cap"] < pool_cap:
            self.pool_buf = {
                "cap": pool_cap,
                "obs": _alloc_obs_pool(pool_cap, env.obs_dim, self.obs_layout),
                "gm": np.empty((pool_cap, GATE_ACTIONS), dtype=bool),
            }
            _OBS_POOL_BUFFERS[pool_key] = self.pool_buf
        self.pool_cap = int(self.pool_buf["cap"])
        self.step_obs_pool = self.pool_buf["obs"]
        self.step_gm_pool = self.pool_buf["gm"]
        self.pool_cursor = 0

        # Output slabs (`_slab_keys` layout); `wcursor` counts the rows
        # flushed so far.
        self.out_slabs_grow = out_slabs_grow
        self.shared_staging = out_slabs is not None
        if out_slabs is not None:
            # P8 shared staging: write into the caller's views of one big host
            # buffer. The views' length IS the capacity (the caller sizes them
            # with the same `_slack_per_env` helper and may hand over all of its
            # remaining buffer); the width asserts catch any variant/obs-dim
            # drift between the caller's allocation and this env.
            self.slabs = {key: out_slabs[key] for key in _slab_keys(self.obs_layout)}
            for key, (shape, np_dtype, _td) in _obs_spec(env.obs_dim, self.obs_layout).items():
                assert self.slabs[key].shape[1:] == shape and self.slabs[key].dtype == np_dtype, (
                    f"out_slabs {key} {self.slabs[key].shape[1:]} {self.slabs[key].dtype} != "
                    f"env layout {shape} {np.dtype(np_dtype)}"
                )
            assert self.slabs["oh"].shape[2] == game_config.hole_count, (
                f"out_slabs hole width {self.slabs['oh'].shape[2]} != "
                f"config {game_config.hole_count}"
            )
            self.slab_alloc = None
        else:
            # Pinned host memory makes the finalize H2D copy overlap downstream
            # compute (non_blocking=True), but PINNING THE ~36GB obs slab can
            # cost 60+ SECONDS PER UPDATE on some hosts (measured: torch 2.11 /
            # AMD EPYC pins 36GB in ~64s, single-threaded — it was the dominant
            # per-update cost and looked like a hang). The pinning tax dwarfs
            # the few seconds non_blocking saves at the finalize, so pinning is
            # DEFAULT OFF. Re-enable with PLO5BP_PIN_ROLLOUT=1 on hosts where
            # large-buffer pinning is cheap. Output is bit-identical either way
            # (non_blocking=True on non-pinned memory degrades to a blocking
            # copy — same data).
            pin = device.type == "cuda" and os.environ.get("PLO5BP_PIN_ROLLOUT", "0") == "1"
            self.slab_alloc = _SlabAllocator(
                env.obs_dim, game_config.hole_count, pin, layout=self.obs_layout
            )
            self.slabs = self.slab_alloc.alloc(self.rollout_target + n_envs * _slack_per_env())
        self.out_cap = int(self.slabs["gm"].shape[0])
        self.wcursor = 0

        # F/T/R telemetry (per street: 0=flop, 1=turn, 2=river -- bomb pots
        # have no preflop action): learner steps, and winning-aggression steps
        # (the flush kernel's retroactive-bonus qualification). The retired
        # aggression / retroactive bonuses (ML-030) go to the kernels as 0, so
        # `aggr_bonus_total_bb` stays 0.
        self.aggr_bonus_total_bb = 0.0
        self.aggr_steps_total = 0
        self.aggr_bonus_steps = 0
        self.aggr_steps_total_by_street: list[int] = [0, 0, 0]
        self.aggr_bonus_steps_by_street: list[int] = [0, 0, 0]

        # P5 step H2D: long-lived pinned staging (CUDA). See _PinnedStepH2D.
        # Slot 0 = the learner's act rows (kept intact until the trajectory store
        # copies them, see step3c/obs_pack); opponents alternate slots 1 and 2.
        # Compact layout -> rows cross PCIe packed and unpack on the device.
        self.step_h2d = _PinnedStepH2D(
            n_envs, env.obs_dim, device, n_slots=3,
            layout=self.obs_layout.pool_layout if self.obs_layout is not None else None,
        )
        self.opp_slot = 1
        # The env packs every observation as it encodes it (same bytes as
        # pack_rows_into), so the uploads gather packed rows instead of
        # re-reading and packing the dense ones -- and since nothing on this
        # path reads the env's DENSE rows then, the env stops writing them
        # (packed-only encoding: no (N, obs_dim) float write per refresh).
        self.env_packed_on = bool(
            self.step_h2d.enabled
            and self.step_h2d.layout is not None
            and env.enable_packed_obs(self.step_h2d.layout, dense=False)
        )

        self.kernels = _kernels()
        self.env_idx_range = np.arange(n_envs)
        self.acts = _StepActs(n_envs, self.use_vrpo)
        self.sizing = np.zeros((n_envs, 4), dtype=np.int64)
        step_timers.end()  # step0/setup

    # ------------------------------------------------------------ setup ---

    def _make_env(
        self,
        env: "BatchedBombPotEnv | None",
        env_cache: "dict[tuple, BatchedBombPotEnv] | None",
    ) -> BatchedBombPotEnv:
        """Env reuse (P3, 2026-07-12): multiconfig passes `env_cache` keyed by
        (n_envs, num_seats, variant, obs mode). When a sub-rollout shares that
        key with a prior sub, reconfigure stacks/ante in place instead of
        reallocating ~N Rust GameStates + Python cache arrays. `env=` is a
        direct override for tests. Single-config callers leave both None ->
        a fresh env."""
        n_envs, game_config, tc = self.n_envs, self.game_config, self.train_config
        if env is not None:
            if env.n != n_envs:
                raise ValueError(f"env.n={env.n} != train_config.num_envs={n_envs}")
            if not env.can_reconfigure(game_config):
                raise ValueError(
                    "provided env cannot reconfigure to game_config "
                    "(seats/variant mismatch)"
                )
            env.reconfigure(game_config, clear_obs=False)  # reset_batch follows
            # Keep EV / MC knobs aligned with this train_config.
            env._ev_runout_samples = int(tc.ev_runout_samples)
            return env
        obs_mode = str(getattr(tc, "obs_mode", "full"))
        cache_key = (n_envs, game_config.num_seats, game_config.variant, obs_mode)
        if env_cache is not None and cache_key in env_cache:
            env = env_cache[cache_key]
            env.reconfigure(game_config, clear_obs=False)  # reset_batch follows
            env._ev_runout_samples = int(tc.ev_runout_samples)
            return env
        env = BatchedBombPotEnv(
            n_envs,
            game_config,
            ev_runout_samples=tc.ev_runout_samples,
            # Minimal obs drops opp-outcome features; opp_mc=0 skips MC entirely.
            opp_outcome_mc=0 if obs_mode == "minimal" else TRAIN_OPP_OUTCOME_MC,
            obs_mode=obs_mode,
        )
        if env_cache is not None:
            env_cache[cache_key] = env
        return env

    def _snapshot_model(self, sd_idx: int) -> ActorCritic:
        m = self.snapshot_models.get(sd_idx)
        if m is not None:
            return m
        # No deepcopy (P5): pool.snapshot() stores detached clones, this path
        # bypasses OpponentPool.sample() (whose deepcopy protects the SERIAL
        # path), load_state_dict copies rather than aliases, and nothing here
        # mutates the dict.
        m = _build_frozen_model(self.pool.snapshots[sd_idx], self.device)
        self.snapshot_models[sd_idx] = m
        return m

    def _stacked_opponents(self, cache) -> "_StackedOpponents | None":
        """Batched opponents (_StackedOpponents): ONE stacked forward + ONE
        sampling pass for every pool snapshot per step. None = per-snapshot
        calls (--no-batched-opponents, an empty pool, or snapshots of mixed
        shape). Built once per update when `cache` is an `_OpponentCache`."""
        n_snap = len(self.pool.snapshots)
        if not (bool(getattr(self.train_config, "batched_opponents", True)) and n_snap):
            return None
        models = [self._snapshot_model(i) for i in range(n_snap)]
        key = tuple(id(m) for m in models)
        if isinstance(cache, _OpponentCache) and cache.stacked_key == key:
            return cache.stacked
        stacked = _StackedOpponents(models) if _StackedOpponents.supported(models) else None
        if isinstance(cache, _OpponentCache):
            cache.stacked_key, cache.stacked = key, stacked
        return stacked

    def _assign_pool_mix(self, env_ids: np.ndarray) -> None:
        """Opponent assignment for the hands about to be dealt in `env_ids`."""
        if self.crn is not None:
            snap, mask = self.crn.pool_mix(
                env_ids, self.n_seats, len(self.pool.snapshots),
                self.pool_opp_seats, self.pool_mix_prob,
            )
        else:
            snap, mask = _draw_pool_mix(
                self.rng, int(env_ids.size), self.n_seats, len(self.pool.snapshots),
                self.pool_opp_seats, self.pool_mix_prob,
            )
        self.env_snapshot_idx[env_ids] = snap
        self.learner_seats_mask[env_ids] = mask

    def _redeal_done_at_deal(self, env_ids: np.ndarray) -> np.ndarray:
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
        env, n_envs = self.env, self.n_envs
        touched = np.zeros(n_envs, dtype=bool)
        dead = np.zeros(n_envs, dtype=bool)
        dead[env_ids] = env._dones[env_ids]
        tries = 0
        while dead.any():
            if tries >= _MAX_REDEALS:
                raise _dead_config_error(
                    "collect_rollout_batched", self.game_config, int(dead.sum())
                )
            tries += 1
            touched |= dead
            seeds, buttons = self._draw_deals(dead)
            env._be.reset_terminal_batch(seeds, buttons, dead)
            env._reset_seeds = np.where(dead, seeds, env._reset_seeds)
            env._refresh_subset(dead)
            dead &= env._dones
        return np.nonzero(touched)[0]

    def _draw_deals(self, mask: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """Full-length (seeds, buttons) arrays for a `reset_terminal_batch` of
        the envs in `mask` (other entries unused): n_envs draws from the
        shared stream, or -- with CRN -- each masked env's next hand."""
        n_envs = self.n_envs
        if self.crn is None:
            seeds = self.rng.integers(0, 2**63 - 1, size=n_envs, dtype=np.int64).astype(np.uint64)
            buttons = self.rng.integers(0, self.n_seats, size=n_envs, dtype=np.int64).astype(np.uint8)
            return seeds, buttons
        ids = np.nonzero(mask)[0]
        seeds = np.zeros(n_envs, dtype=np.uint64)
        buttons = np.zeros(n_envs, dtype=np.uint8)
        seeds[ids], buttons[ids] = self.crn.deal(ids, self.n_seats)
        return seeds, buttons

    def _fill_holes_rot(self, env_ids: np.ndarray) -> None:
        """Hero-rotated opponent holes of every seat of `env_ids`
        (= `_rotate_opp_holes_batch` for each seat as the actor)."""
        if env_ids.size == 0:
            return
        n_seats = self.n_seats
        hc = self.holes_cache[env_ids]
        seat_ids = np.arange(n_seats, dtype=np.int64)
        j5 = np.arange(5, dtype=np.int64)
        rot_seats = (seat_ids[:, None] + 1 + j5[None, :]) % n_seats
        block = hc[:, rot_seats]
        invalid = (j5 + 1) >= n_seats
        if invalid.any():
            block = np.where(invalid[None, None, :, None], np.uint8(255), block)
        self.holes_rot_cache[env_ids] = block

    # ------------------------------------------------------------ buffers ---

    def _bind_traj(self) -> None:
        # One flat VIEW per trajectory array (`_flat_traj_view`): a row's slot
        # is the single index (env * n_seats + seat) * cap + slot. The VRPO
        # arrays (q_taken / vpi) exist only for that estimator.
        self.traj_flat: dict[str, np.ndarray] = {
            name: _flat_traj_view(arr)
            for name, arr in self.traj_buf.items()
            if name != "cap" and arr is not None
        }

    def _grow_traj(self, min_cap: int) -> None:
        """Re-allocate every per-(env, seat) trajectory array with at least
        `min_cap` slots (doubling, capped at MAX_STEPS_PER_SEAT), copying the
        slots written so far; fresh slots get each array's initial fill. The
        grown arrays replace the cached ones (later sub-rollouts start big)."""
        new_cap = min(self.MAX_STEPS_PER_SEAT, max(int(min_cap), 2 * self.traj_cap))
        grown = _alloc_traj(self.n_envs, self.n_seats, new_cap, self.use_vrpo)
        for name, arr in grown.items():
            if arr is not None:
                arr[:, :, : self.traj_cap] = self.traj_buf[name]
            self.traj_buf[name] = arr
        self.traj_buf["cap"] = new_cap
        self.traj_cap = new_cap
        self._bind_traj()

    def _grow_pool(self, min_rows: int) -> None:
        new_cap = max(int(min_rows), self.pool_cap + max(self.pool_cap // 4, self.n_envs))
        new_obs = _alloc_obs_pool(new_cap, self.env.obs_dim, self.obs_layout)
        new_gm = np.empty((new_cap, GATE_ACTIONS), dtype=bool)
        used = self.pool_cursor
        for key, arr in self.step_obs_pool.items():
            new_obs[key][:used] = arr[:used]
        new_gm[:used] = self.step_gm_pool[:used]
        self.step_obs_pool, self.step_gm_pool, self.pool_cap = new_obs, new_gm, new_cap
        self.pool_buf.update(cap=new_cap, obs=new_obs, gm=new_gm)

    def _grow_slabs(self, min_rows: int) -> None:
        """Make room for `min_rows` output rows, preserving the `wcursor`
        rows already flushed (A6: replaces the old hard overflow error)."""
        if self.slab_alloc is not None:
            new_cap = max(int(min_rows), self.out_cap + max(self.out_cap // 4, self.n_envs))
            self.slabs = self.slab_alloc.grow(self.slabs, self.wcursor, new_cap)
        elif self.out_slabs_grow is not None:
            grown = self.out_slabs_grow(self.wcursor, int(min_rows))
            self.slabs = {key: grown[key] for key in _slab_keys(self.obs_layout)}
        else:
            raise RuntimeError(
                f"output slab overflow: {min_rows} > out_cap={self.out_cap} "
                f"(rollout_target={self.rollout_target}, {self.n_envs} envs) and the "
                "caller's out_slabs came without an out_slabs_grow callback"
            )
        self.out_cap = int(self.slabs["gm"].shape[0])
        assert self.out_cap >= min_rows, f"slab grow fell short: {self.out_cap} < {min_rows}"

    def _flush_pools(self) -> dict[str, np.ndarray]:
        """The per-step pool as the flush kernel reads it: packed bits + real
        columns + gate masks. DENSE storage passes each float32 row as its
        raw bytes (a layout with no real columns): the kernel's byte copy is
        the same float32 row, so dense runs (NLH, --no-compact-obs) flush
        through the same kernel as compact ones."""
        if self.obs_layout is not None:
            pools = dict(self.step_obs_pool)
        else:
            dense = self.step_obs_pool["obs"]
            pools = {
                "obs_bits": dense.view(np.uint8),
                "obs_real": np.empty((dense.shape[0], 0), dtype=np.float32),
            }
        pools["gm"] = self.step_gm_pool
        return pools

    def _flush_out(self, lo: int, hi: int) -> dict[str, np.ndarray]:
        """The flush kernel's output views of slab rows [lo, hi)."""
        if self.obs_layout is not None:
            return _rust_flush_out(self.slabs, lo, hi)
        out = {key: self.slabs[key][lo:hi] for key in _SLAB_KEYS}
        out["obs_bits"] = self.slabs["obs"][lo:hi].view(np.uint8)
        out["obs_real"] = np.empty((hi - lo, 0), dtype=np.float32)
        return out

    # --------------------------------------------------------------- loop ---

    def run(self) -> Batch:
        """Step until the row target (+ the drain), then finalize.

        PRODUCTION BEHAVIOR CHANGE (review 2026-09-20 A6) — drain_inflight.
        Phase 1 (`wcursor < rollout_target`): every finished env is re-dealt.
        Phase 2 (drain only) starts once the target is reached: finished envs
        are flushed but NOT re-dealt, and the loop runs until every env is
        done, i.e. every hand STARTED in phase 1 is in the batch. With drain
        off the loop exits at the target (in-flight hands dropped) —
        byte-identical rows, order and RNG to the pre-fix collector."""
        env = self.env
        while self.wcursor < self.rollout_target or (self.drain and not env._dones.all()):
            self.timers.begin("step0/prestep")
            st = self._prestep()
            self.timers.end()  # step0/prestep
            learner_packed = self._act(st)
            if st.learner_idx.size:
                self._record(st, learner_packed)
            newly_terminal, post_total_commit = self._apply(st)
            self._record_costs(st, post_total_commit)
            if newly_terminal.any():
                term_envs = np.nonzero(newly_terminal)[0]
                won_term = self._payouts(term_envs, post_total_commit)
                self._flush(term_envs, won_term, post_total_commit)
                # A6 drain: once the row target is reached the finished envs
                # are NOT re-dealt — they stay done (`active` masks them out)
                # while the hands still in flight play to completion. The
                # re-deal's draws sit AFTER the flush (which consumes no RNG)
                # so the numpy stream keeps its pre-fix order — seeds,
                # buttons, pool-mix — and a drain-off run is byte-identical.
                if not (self.drain and self.wcursor >= self.rollout_target):
                    self._redeal(term_envs, newly_terminal)
        # Sizing hint for the next collection's pool/slab slack (never a value).
        _note_slack_used(max(self.wcursor, self.pool_cursor), self.rollout_target, self.n_envs)
        return self._finalize()

    def _prestep(self) -> _Step:
        env = self.env
        actors = env._actors
        dones = env._dones
        if dones.all():
            # Unreachable by construction (every reset re-deals done-at-deal
            # envs or raises) — but an all-done table can never make progress,
            # so fail loudly rather than spin (review 2026-09-20 A1).
            raise RuntimeError(
                "collect_rollout_batched: no live env below the row target "
                f"({self.wcursor}/{self.rollout_target} rows) — config "
                f"{self.game_config!r}"
            )
        # Vectorized classification: active learner envs vs active opponent
        # envs. `safe_actors` clamps -1 to 0 so the indexing can't blow up; the
        # result is masked by `active`.
        active = ~dones
        safe_actors = np.where(actors >= 0, actors, 0).astype(np.intp)
        is_learner_active = active & self.learner_seats_mask[self.env_idx_range, safe_actors]
        # Per-step sizing context (min, max, pot, to_call) — built once; the
        # same array feeds act() and the trajectory store. Written into one
        # buffer per collection (ML-043): the same values np.stack gave.
        sizing = self.sizing
        sizing[:, 0] = env._min_raise
        sizing[:, 1] = env._max_raise
        sizing[:, 2] = env._pot
        sizing[:, 3] = np.maximum(
            env._bet_to_call.astype(np.int64)
            - env._street_commit[self.env_idx_range, safe_actors].astype(np.int64),
            0,
        )
        return _Step(
            obs=env._obs,
            gate_masks=env._gate_mask,
            actors=actors,
            dones=dones,
            pre_total_commit=env._total_commit,
            pre_bet_to_call=env._bet_to_call,
            pre_street_commit=env._street_commit,
            pre_street=env._street,
            active=active,
            safe_actors=safe_actors,
            is_opp_active=active & ~is_learner_active,
            learner_idx=np.nonzero(is_learner_active)[0],
            sizing=sizing,
        )

    # ---------------------------------------------------------------- act ---

    def _act(self, st: _Step) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
        """Sample every live env's action into `self.acts`: learner act (+
        critic) on the device, then the opponents' acts, then ONE coalesced
        D2H (Attack #2 Phase 1). act() call order: the full learner batch,
        then the opponents (one stacked call, or per snapshot in ascending
        index order). Returns the learner's uploaded packed rows (slot 0) for
        the trajectory store, or None."""
        acts = self.acts
        acts.reset()
        learner_fw = critic_v = critic_q = None
        learner_packed = None
        rows = st.learner_idx
        if rows.size:
            learner_fw, o_t = self._learner_act(rows, st.obs, st.gate_masks, st.sizing)
            # Slot 0 now holds exactly these rows, packed (opponents upload
            # through slots 1/2): step3c/obs_pack stores them.
            learner_packed = self.step_h2d.packed_rows(0, rows.size)
            if self.critic is not None:
                critic_v, critic_q = self._critic_forward(rows, st.safe_actors, o_t)
        opp_groups: list[tuple[np.ndarray, object]] = []
        if st.is_opp_active.any():
            with _TimedRF("step4a/opp_h2d_act"):
                opp_groups = self._opp_acts(
                    st.is_opp_active, st.obs, st.gate_masks, st.sizing
                )
        with _TimedRF("step5/action_d2h"):
            if learner_fw is not None:
                self._learner_to_host(rows, learner_fw, critic_v, critic_q)
            if opp_groups:
                self._opp_to_host(opp_groups)
            del opp_groups
        return learner_packed

    def _packed_for(self, obs_arr: np.ndarray):
        """The env's packed copy when `obs_arr` IS its live obs buffer (the
        copy describes exactly that buffer); None for anything else (those
        rows are packed on the fly)."""
        if self.env_packed_on and obs_arr is self.env._obs:
            return self.env.packed_obs()
        return None

    def _learner_act(
        self,
        rows: np.ndarray,
        obs_arr: np.ndarray,
        gate_mask_arr: np.ndarray,
        sizing_arr: np.ndarray,
    ):
        """Learner H2D + act; ActOut + the uploaded obs stay on the device
        (the host pull is one coalesced D2H in `_act`)."""
        with _TimedRF("step2/learner_h2d"):
            self.step_h2d.wait_slot(0)
            o_t, m_t, b_t = self.step_h2d.upload_rows(
                obs_arr, rows, gate_mask_arr, sizing_arr, slot=0,
                packed=self._packed_for(obs_arr),
            )
        with _TimedRF("step3/learner_forward"):
            with torch.inference_mode():
                if self.use_vrpo:
                    fw = self.learner.act(o_t, m_t, b_t, return_marginal=True)
                else:
                    fw = self.learner.act(o_t, m_t, b_t)
        return fw, o_t

    def _critic_forward(self, rows: np.ndarray, safe_actors: np.ndarray, o_t: torch.Tensor):
        """Centralized critic on the learner rows (reusing the actor's
        uploaded obs): V, and with VRPO also the dueling Q columns."""
        with _TimedRF("step3a/opp_holes_rot"):
            # = _rotate_opp_holes_batch(holes_cache, rows, actors): the
            # per-hand cache holds every seat's rotation.
            opp_block = self.holes_rot_cache[rows, safe_actors[rows]]
        with _TimedRF("step3b/critic_forward"):
            device = self.device
            h_t = torch.from_numpy(opp_block).to(device, non_blocking=(device.type == "cuda"))
            with torch.inference_mode():
                if self.use_vrpo:
                    return self.critic.q_values(o_t, opp_holes_multihot(h_t))
                return self.critic(o_t, opp_holes_multihot(h_t)), None

    def _next_opp_slot(self, slot: int) -> int:
        # Double-pin opp path: alternate slots 1/2 so group i+1's H2D does not
        # wait for group i's (only for a slot's reuse = group i-2).
        return 3 - slot if self.step_h2d.n_slots > 2 else slot

    def _opp_upload_act(
        self,
        model: ActorCritic,
        group: np.ndarray,
        obs_arr: np.ndarray,
        gate_mask_arr: np.ndarray,
        sizing_arr: np.ndarray,
    ):
        """H2D + act for one snapshot's rows; results stay on the device."""
        slot = self.opp_slot
        self.step_h2d.wait_slot(slot)
        o_t, m_t, b_t = self.step_h2d.upload_rows(
            obs_arr, group, gate_mask_arr, sizing_arr, slot=slot,
            packed=self._packed_for(obs_arr),
        )
        self.opp_slot = self._next_opp_slot(slot)
        with torch.inference_mode():
            return model.act(o_t, m_t, b_t)

    def _opp_acts(
        self,
        is_opp: np.ndarray,
        obs_arr: np.ndarray,
        gate_mask_arr: np.ndarray,
        sizing_arr: np.ndarray,
    ) -> list[tuple[np.ndarray, object]]:
        """Act every opponent row in `is_opp`, results left on device (no D2H):
        [(env rows, ActOut)] — ONE entry covering every snapshot on the stacked
        path (rows grouped by snapshot), one per snapshot group otherwise
        (ascending snapshot index, the pre-2026-09-23 behavior)."""
        snap_col = np.where(is_opp, self.env_snapshot_idx, -1)
        if self.stacked_opp is None:
            groups: list[tuple[np.ndarray, object]] = []
            for sd_idx in np.unique(snap_col[snap_col >= 0]):
                group = np.nonzero(snap_col == sd_idx)[0]
                m = self._snapshot_model(int(sd_idx))
                groups.append(
                    (group, self._opp_upload_act(m, group, obs_arr, gate_mask_arr, sizing_arr))
                )
            return groups
        rows = np.nonzero(snap_col >= 0)[0]
        if rows.size == 0:
            return []
        order = np.argsort(snap_col[rows], kind="stable")
        rows = rows[order]
        g = snap_col[rows]
        counts = np.bincount(g, minlength=self.stacked_opp.n)
        j = np.arange(rows.size) - (np.cumsum(counts) - counts)[g]
        slot = self.opp_slot
        self.step_h2d.wait_slot(slot)
        o_t, m_t, b_t = self.step_h2d.upload_rows(
            obs_arr, rows, gate_mask_arr, sizing_arr, slot=slot,
            packed=self._packed_for(obs_arr),
        )
        self.opp_slot = self._next_opp_slot(slot)
        g_t = torch.from_numpy(g).to(self.device)
        j_t = torch.from_numpy(j).to(self.device)
        with torch.inference_mode():
            fw = self.stacked_opp.act(o_t, m_t, b_t, g_t, j_t, int(counts.max()))
        return [(rows, fw)]

    def _learner_to_host(self, rows: np.ndarray, fw: ActOut, critic_v, critic_q) -> None:
        acts = self.acts
        g_np, c_np, an_np, lp_np, ru_np, v_np, glp_np, alp_np, marg_np = _act_to_host(fw)
        acts.gate[rows] = g_np
        acts.chips[rows] = np.maximum(c_np, 0).astype(np.uint64)
        acts.anchor[rows] = an_np
        acts.u[rows] = ru_np
        acts.log_p[rows] = lp_np
        acts.gate_lp[rows] = glp_np
        acts.anchor_lp[rows] = alp_np
        if critic_v is None:
            acts.value[rows] = v_np
        elif self.use_vrpo and critic_q is not None:
            # One D2H for V||Q: cat, copy, split (pure copies).
            vq = torch.cat(
                [critic_v.float().unsqueeze(-1), critic_q.float()], dim=-1
            ).cpu().numpy()
            v_l = vq[..., 0]
            q_l = vq[..., 1:]
            acts.value[rows] = v_l
            if q_l.shape[-1] == 3:
                q_idx_l = g_np.astype(np.int64)
                marg_l = np.stack(
                    [marg_np[:, 0], marg_np[:, 1], marg_np[:, 2:].sum(-1)], axis=-1
                )
            else:
                q_idx_l = np.where(g_np == GATE_RAISE, 2 + an_np, g_np.astype(np.int64))
                marg_l = marg_np
            rows_l = np.arange(rows.size)
            acts.q_taken[rows] = q_l[rows_l, q_idx_l]
            acts.vpi[rows] = (marg_l * q_l).sum(-1)
        else:
            acts.value[rows] = critic_v.float().cpu().numpy()

    def _opp_to_host(self, opp_groups: list[tuple[np.ndarray, object]]) -> None:
        g_cat = torch.cat([out.gate for _, out in opp_groups], dim=0)
        c_cat = torch.cat([out.chips for _, out in opp_groups], dim=0)
        gc = torch.stack((g_cat.to(torch.int64), c_cat.to(torch.int64)), dim=0).cpu().numpy()
        g_all = gc[0].astype(np.uint8)
        c_all = np.maximum(gc[1], 0).astype(np.uint64)
        cursor = 0
        for group, _ in opp_groups:
            n_g = int(group.size)
            self.acts.gate[group] = g_all[cursor : cursor + n_g]
            self.acts.chips[group] = c_all[cursor : cursor + n_g]
            cursor += n_g

    # ------------------------------------------------------------- record ---

    def _record(self, st: _Step, learner_packed) -> None:
        """step3c: append the learner rows to the obs pool and record their
        step in the trajectory arrays (one bulk copy + one kernel call)."""
        rows = st.learner_idx
        k_step = rows.size
        pool_start = self.pool_cursor
        pool_end = pool_start + k_step
        if pool_end > self.pool_cap:
            self.timers.begin("step3c/pool_grow")
            self._grow_pool(pool_end)
            self.timers.end()  # step3c/pool_grow
        self.timers.begin("step3c/obs_pack")
        pool = self.step_obs_pool
        if self.obs_layout is None:
            pool["obs"][pool_start:pool_end] = st.obs[rows]
        elif learner_packed is not None:
            # The uploaded rows (slot 0) straight into the pool, one call.
            _gather_rows_multi(
                [
                    (learner_packed[0], pool["obs_bits"]),
                    (learner_packed[1], pool["obs_real"]),
                    (learner_packed[2], self.step_gm_pool),
                ],
                None,
                pool_start,
            )
        else:
            pack_rows_into(
                st.obs, rows, self.obs_layout,
                pool["obs_bits"], pool["obs_real"], pool_start,
            )
        if learner_packed is None:
            _gather_rows(st.gate_masks, rows, self.step_gm_pool, pool_start)
        self.timers.end()  # step3c/obs_pack
        self.pool_cursor = pool_end

        learner_actors = st.safe_actors[rows]
        slots = self.traj_lengths[rows, learner_actors]
        if (slots >= self.MAX_STEPS_PER_SEAT).any():
            raise RuntimeError(
                f"per-seat trajectory length exceeded "
                f"MAX_STEPS_PER_SEAT={self.MAX_STEPS_PER_SEAT}"
            )
        if int(slots.max()) >= self.traj_cap:
            self.timers.begin("step3c/traj_grow")
            self._grow_traj(int(slots.max()) + 1)
            self.timers.end()  # step3c/traj_grow
        self.timers.begin("step3c/traj_writes")
        # One flat slot index for every per-(env, seat, slot) array (after any
        # _grow_traj above, so it uses the current capacity).
        tw = (rows * self.n_seats + learner_actors) * self.traj_cap + slots
        self.kernels.record(
            tw.astype(np.int64, copy=False),
            rows.astype(np.int64, copy=False),
            int(pool_start), self.traj_flat, self.acts.per_env(st.sizing),
        )
        self.timers.end()  # step3c/traj_writes

    # -------------------------------------------------------------- apply ---

    def _apply(self, st: _Step) -> tuple[np.ndarray, np.ndarray]:
        """Apply every env's action, then re-read the engine. Returns
        (newly-terminal mask, post-step total_commit)."""
        env = self.env
        gates = self.acts.gate
        # Short-shove redirect: rows where the network emitted GATE_RAISE but
        # the engine zeroed `min_raise` (sub-min-raise stack with
        # `legal[ALL_IN]`) go to the Rust dispatcher's AllIn arm (gate=3). The
        # trajectory already recorded the network's gate (GATE_RAISE) and
        # chips; the remap is purely a dispatch concern (the AllIn arm ignores
        # the chip amount).
        short_shove = (
            (gates == GATE_RAISE)
            & (env._min_raise == np.uint64(0))
            & env._legal[:, ALL_IN]
        )
        if short_shove.any():
            gates = gates.copy()
            gates[short_shove] = 3
        with _TimedRF("step6+7/rust_apply"):
            newly_terminal = np.asarray(
                env._be.apply_hybrid_batch(gates, self.acts.chips), dtype=bool
            )
        # Refresh now to capture post-step total_commit (and everything else)
        # BEFORE reset_terminal_batch wipes terminal envs. Newly-terminal rows
        # skip the obs encode: their post-apply obs is never read (reset +
        # subset-refresh re-deal them before the next act); the engine caches
        # (total_commit / legal / actors) still refresh for every env. Kept
        # rows bit-exact; skipped rows get zeros. `active &` only matters in
        # the A6 drain phase, where envs that finished on an EARLIER step stay
        # done and need no encode either; before the target `active` is
        # all-True.
        with _TimedRF("step1a/refresh"):
            enc = st.active & ~newly_terminal
            if enc.all():
                env._refresh()
            else:
                env._refresh(encode_mask=enc)
        post_total_commit = env._total_commit
        assert (
            post_total_commit is not st.pre_total_commit
            and env._bet_to_call is not st.pre_bet_to_call
            and env._street_commit is not st.pre_street_commit
            and env._street is not st.pre_street
        ), "the refresh updated the pre-step snapshot arrays in place"
        return newly_terminal, post_total_commit

    def _record_costs(self, st: _Step, post_total_commit: np.ndarray) -> None:
        """step8: every acting learner seat's step cost (chips put in, in bb,
        negative; the retired aggression bonus is passed as 0), pre-step pot
        and street at its trajectory slot, and the slot count +1 -- one engine
        call. Also the F/T/R step counters."""
        with _TimedRF("step8/aggression_bonus"):
            tb, ts, bs, sbs, bbs = self.kernels.aggression(
                st.actors,
                st.dones,
                self.learner_seats_mask,
                self.acts.gate,
                st.pre_total_commit,
                post_total_commit.astype(np.int64)
                if post_total_commit.dtype != np.int64
                else post_total_commit,
                st.pre_bet_to_call,
                st.pre_street_commit,
                st.pre_street,
                0.0,  # the retired per-step aggression bonus
                float(self.reward_norm),
                self.traj_flat["costs"],
                self.traj_flat["pots"],
                self.traj_flat["streets"],
                self.traj_lengths,
                int(self.traj_cap),
            )
            self.aggr_bonus_total_bb += float(tb)
            self.aggr_steps_total += int(ts)
            self.aggr_bonus_steps += int(bs)
            for s in range(3):
                self.aggr_steps_total_by_street[s] += int(sbs[s])
                self.aggr_bonus_steps_by_street[s] += int(bbs[s])

    # ------------------------------------------------------ finished hands ---

    def _payouts(self, term_envs: np.ndarray, post_total_commit: np.ndarray) -> np.ndarray:
        """Gross winnings (payout + total commit, chips) of the hands that
        just finished, (T, S) float32."""
        env = self.env
        with _TimedRF("step9a/payouts"):
            # Only the newly-terminal rows: in the drain phase every env that
            # finished on an EARLIER step is still terminal, and a whole-batch
            # call would re-run all of their runouts each step (same seeds ->
            # the same values).
            if env._ev_runout_samples > 0:
                ev_seeds = env._reset_seeds ^ np.uint64(0x9E3779B97F4A7C15)
                payouts_term = np.asarray(
                    env._be.payouts_ev_subset(
                        env._ev_runout_samples,
                        ev_seeds[term_envs],
                        term_envs.astype(np.int64),
                    ),
                    dtype=np.float32,
                )
            else:
                payouts_term = np.asarray(env._be.payouts_batch(), dtype=np.float32)[term_envs]
            return payouts_term + post_total_commit[term_envs].astype(np.float32)

    def _flush(
        self, term_envs: np.ndarray, won_term: np.ndarray, post_total_commit: np.ndarray
    ) -> None:
        """steps 9b-9d in ONE kernel call (engine `flush_trajectories`, numpy's
        exact f32 operation order -- rollout_reference.flush_trajectories is
        the oracle): per finished hand and learner seat, the winning-aggression
        qualification (the F/T/R counters; the retired bonus is passed as 0),
        the GAE / VRPO backward scans, and the rows gathered into the output
        slabs at `wcursor` (C order of the (hand, seat, slot) window)."""
        with _TimedRF("step9d/flush_rust"):
            n_seats = self.n_seats
            lengths = self.traj_lengths[term_envs]                  # (T, S) i32
            flush_mask = self.learner_seats_mask[term_envs] & (lengths > 0)
            n_new = int(lengths[flush_mask].sum())
            if n_new:
                payout_chips = np.rint(won_term).astype(np.int64)  # (T, S)
                total_pot_chips = post_total_commit[term_envs].astype(np.int64).sum(axis=1)
                two_pay = 2 * payout_chips
                share_eq = (two_pay == total_pot_chips[:, None]) & (
                    total_pot_chips[:, None] > 0
                )
                share_gt = two_pay > total_pot_chips[:, None]
                won_bb = won_term.astype(np.float32) * np.float32(self.reward_norm)
                if self.wcursor + n_new > self.out_cap:
                    self._grow_slabs(self.wcursor + n_new)
                end = self.wcursor + n_new
                written, b_steps, b_street, b_total = self.kernels.flush(
                    term_envs.astype(np.int64, copy=False),
                    np.ascontiguousarray(lengths, dtype=np.int32),
                    flush_mask, won_bb, share_gt, share_eq,
                    self.traj_flat, self.traj_cap,
                    self._flush_pools(),
                    self.holes_rot_cache.reshape(self.n_envs * n_seats, 5 * self.hole_w),
                    np.float32(self.gamma), np.float32(self.lam),
                    np.float32(0.0),  # the retired retroactive bonus
                    self._flush_out(self.wcursor, end),
                )
                assert written == n_new, (written, n_new)
                self.aggr_bonus_steps += int(b_steps)
                for s_idx in range(3):
                    self.aggr_bonus_steps_by_street[s_idx] += int(b_street[s_idx])
                self.wcursor = end
        # Always, whatever the flush wrote: a stale length would grow the
        # seat's trajectory across hands.
        self.traj_lengths[term_envs] = 0

    def _redeal(self, term_envs: np.ndarray, reset_mask: np.ndarray) -> None:
        """Deal fresh hands to the envs that just finished."""
        env = self.env
        if self.crn is None:
            # The shared stream: seeds + buttons first, then the pool mix
            # (the order every stem so far drew them in).
            new_seeds, new_buttons = self._draw_deals(reset_mask)
            with _TimedRF("step9e/pool_mix"):
                # Batched draws for every re-dealt hand (see _draw_pool_mix).
                self._assign_pool_mix(term_envs)
        else:
            # CRN: the hand's opponent assignment, then its deal (which
            # advances the envs' hand counters).
            with _TimedRF("step9e/pool_mix"):
                self._assign_pool_mix(term_envs)
            new_seeds, new_buttons = self._draw_deals(reset_mask)
        with _TimedRF("step9f/reset_terminal"):
            env._be.reset_terminal_batch(new_seeds, new_buttons, reset_mask)
            env._reset_seeds = np.where(reset_mask, new_seeds, env._reset_seeds)
            # `reset_terminal_batch` mutates ONLY the masked envs, and nothing
            # since the post-apply refresh mutated the others' engine state, so
            # only the reset rows need re-packing/re-encoding — bit-exact with a
            # full `_refresh()` (see `_refresh_subset` + test_refresh_subset_parity).
            env._refresh_subset(reset_mask)
            # A1: a re-dealt hand that is already over at deal is re-dealt
            # again (bounded), never left to stall the loop.
            self._redeal_done_at_deal(term_envs)
            # The per-hand hole cache for the re-dealt envs only (holes are
            # static within a hand), after the A1 pass so it holds the FINAL
            # deal's cards.
            term_idx = term_envs.astype(np.int64)
            self.holes_cache[term_envs] = np.asarray(
                env._be.all_hole_cards_subset_batch(term_idx), dtype=np.uint8
            )
            self._fill_holes_rot(term_idx)

    # ----------------------------------------------------------- finalize ---

    def _finalize(self) -> Batch:
        slabs, wcursor = self.slabs, self.wcursor
        aggr = dict(
            aggr_bonus_total_bb=float(self.aggr_bonus_total_bb),
            aggr_steps_total=int(self.aggr_steps_total),
            aggr_bonus_steps=int(self.aggr_bonus_steps),
            aggr_steps_total_by_street=tuple(int(x) for x in self.aggr_steps_total_by_street),
            aggr_bonus_steps_by_street=tuple(int(x) for x in self.aggr_bonus_steps_by_street),
        )
        adv_clip = float(getattr(self.train_config, "adv_clip", 0.0))
        if self.shared_staging:
            # P7 shared-staging finalize: no full H2D — the rows already sit in
            # the caller's big host buffer. Only the per-sub advantage
            # normalization hops to the learner device: it ran on CUDA in the
            # legacy path (_finalize_batch_arr), and a CPU reimplementation
            # would drift f32 reduction order, so ship the tiny (wcursor,)
            # vector up, run the IDENTICAL op sequence, and write the result
            # back into the slab.
            with _TimedRF("step11/shared_adv_norm"):
                adv_view = slabs["adv"][:wcursor]
                adv_t = torch.from_numpy(adv_view).to(self.device, non_blocking=True)
                adv_mean = adv_t.mean()
                adv_std = adv_t.std().clamp(min=1e-8)
                adv_t = (adv_t - adv_mean) / adv_std
                if adv_clip > 0.0:
                    # Same fat-tail clamp as _finalize_batch (serial parity).
                    adv_t = adv_t.clamp(-adv_clip, adv_clip)
                np.copyto(adv_view, adv_t.cpu().numpy())
            return Batch(
                obs=_obs_from_slabs(slabs, wcursor, self.obs_layout),
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
                **aggr,
            )
        _enter_gpu_phase()  # a standalone collection ships its batch now
        return _finalize_batch_arr(
            _obs_from_slabs(slabs, wcursor, self.obs_layout),
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
            device=self.device,
            adv_clip=adv_clip,
            all_last_arr=slabs["last"],
            **aggr,
        )


# ---- vThree: mix many (seats,stacks) configs within ONE update ------------
# Each update's gradient averages over N configs spanning all stack tiers,
# instead of one config + a 50-update block. This removes the consecutive-
# shallow exposure that saturated the gate (vTwo10-13 all died ~38 clubgg
# updates in). Implemented as a thin wrapper over the bit-exact single-config
# collector: split → host-concat → a final pooled advantage re-normalization
# which is numerically a NO-OP — advantages are effectively normalized PER
# CONFIG (see the `_concat_batches` docstring; review 2026-09-20 A5).

# Per-(env, seat) hero-rotated opponent-hole blocks, by (n_envs, n_seats,
# hole width) -- reused like the trajectory buffers (collect_rollout_batched).
_HOLES_ROT_BUFFERS: dict = {}

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
    0.9996 after (`docs/reviews/repro-2026-09-20/agent_rollout/
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
    crn_key: "tuple[int, ...] | None" = None,
) -> Batch:
    """One update's rollout MIXED across `configs` distinct (seats,stacks)
    setups -- see `_collect_multiconfig` for everything it does. This wrapper
    owns the step timers: every sub-collection adds to one set, reported once,
    and the parent marker is cleared even when a sub raises. `crn_key` (ML-004):
    sub-rollout i collects with CRN key (*crn_key, i)."""
    global _ACTIVE_STEP_TIMERS, _STEP_TIMERS_PARENT
    parent = _StepTimers()
    _STEP_TIMERS_PARENT = _ACTIVE_STEP_TIMERS = parent
    try:
        combined, device = _collect_multiconfig(
            learner, pool, configs, train_config, rng, critic=critic,
            config_tiers=config_tiers, tier_ent=tier_ent,
            _legacy_staging=_legacy_staging, drain_inflight=drain_inflight,
            crn_key=crn_key,
        )
    finally:
        _STEP_TIMERS_PARENT = None
        _ACTIVE_STEP_TIMERS = None
    parent.report(label="collect_rollout_multiconfig")

    _enter_gpu_phase()
    if getattr(train_config, "batch_on_host", False) and device.type != "cpu":
        # PPO gathers every minibatch from these host rows (HostBatchLoader);
        # nothing reads them after the update, before the next collection
        # reuses the staging buffer.
        return combined
    return _batch_to_device(combined, device)


def _collect_multiconfig(
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
    crn_key: "tuple[int, ...] | None" = None,
) -> "tuple[Batch, torch.device]":
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
    bit-exactness reference for tests/python/training/test_multiconfig_staging.py —
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
    # trainer.update), so each member builds (and the members stack) once
    # instead of once per sub-rollout. Dies with this call; bounded at pool
    # capacity (~8 models, ~0.5GB — the same worst case a single sub-rollout
    # already reaches). NOT the reverted cross-update opponent cache (that
    # one persisted across updates and grew).
    snapshot_cache = _OpponentCache()
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
        for i, cfg in enumerate(configs):
            sub = collect_rollout_batched(
                learner, pool, cfg, sub_config, rng, critic=critic,
                snapshot_cache=snapshot_cache,
                env_cache=env_cache,
                drain_inflight=drain,
                crn_key=None if crn_key is None else (*crn_key, i),
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
        for i, cfg in enumerate(configs):
            views = {key: arr[base:] for key, arr in big.items()}
            sub = collect_rollout_batched(
                learner, pool, cfg, sub_config, rng,
                critic=critic, out_slabs=views,
                snapshot_cache=snapshot_cache,
                env_cache=env_cache,
                drain_inflight=drain,
                out_slabs_grow=_grow,
                crn_key=None if crn_key is None else (*crn_key, i),
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
        spans: dict[str, list[tuple[int, int]]] = {}
        start = 0
        for rows, t in zip(sub_rows, config_tiers):
            spans.setdefault(t, []).append((start, start + int(rows)))
            start += int(rows)
        combined.tier_rows = spans
    return combined, device


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
    for sel in iter_minibatch_indices(batch, batch_size, rng):
        yield gather_minibatch(batch, sel)


# Wall seconds spent in iter_minibatch_indices' shuffles, process-wide (a
# one-element list so importers see updates). Reporting only.
SHUFFLE_SECONDS: list[float] = [0.0]


def iter_minibatch_indices(
    batch: Batch, batch_size: int, rng: np.random.Generator
) -> Iterable[torch.Tensor]:
    """The row indices (device tensor) of each minibatch of `iter_minibatches`
    -- the same shuffle, bounds and RNG use -- without gathering the rows, so
    a caller can gather a minibatch in chunks (PPO micro-batching). The
    shuffle's wall time accumulates in `SHUFFLE_SECONDS` (PPOStats.shuffle_s,
    2026-09-28 ML-045: a single-threaded pass over ~160M indices per epoch
    that no timer covered)."""
    n = batch.obs.shape[0]
    t0 = time.perf_counter()
    with record_function("step12a0/shuffle"):
        idx = np.arange(n)
        rng.shuffle(idx)
    SHUFFLE_SECONDS[0] += time.perf_counter() - t0
    device = batch.obs.device
    for start, stop in _minibatch_bounds(n, batch_size):
        yield torch.from_numpy(idx[start:stop]).to(device)


class HostBatchLoader:
    """Minibatches of a HOST-resident Batch on the learner device
    (TrainingConfig.batch_on_host). Every field's rows are gathered on the
    CPU (the engine's parallel byte-row copy) into pinned staging of
    `capacity` rows and copied over asynchronously; compact observations
    cross packed and are unpacked on the device. `gather(sel)` equals
    `gather_minibatch` on a device-resident copy of the batch: the same rows,
    the same bytes, unpacked by the same function."""

    def __init__(self, batch: Batch, device: torch.device, capacity: int) -> None:
        self.device = torch.device(device)
        self.cap = max(1, int(capacity))
        pin = self.device.type == "cuda"
        self.layout = batch.obs.layout if isinstance(batch.obs, PackedObs) else None
        # (field, host rows as a 2-D numpy view, pinned staging tensor)
        self._fields: list[tuple[str, np.ndarray, torch.Tensor]] = []

        def add(name: str, t: torch.Tensor) -> None:
            if t.device.type != "cpu":
                raise ValueError(f"HostBatchLoader: '{name}' is not on the host")
            host = t.detach().contiguous()
            stage = torch.empty(
                (self.cap,) + tuple(host.shape[1:]), dtype=host.dtype, pin_memory=pin
            )
            self._fields.append((name, host.numpy().reshape(host.shape[0], -1), stage))

        if self.layout is not None:
            add("obs_bits", batch.obs.bits)
            add("obs_real", batch.obs.real)
        else:
            add("obs", batch.obs)
        for f in _BATCH_TENSOR_FIELDS[1:]:
            add(f, getattr(batch, f))
        if batch.is_terminal is not None:
            add("is_terminal", batch.is_terminal)
        if batch.ent_coef_rows is not None:
            add("ent_coef_rows", batch.ent_coef_rows)
        self._host_fields = {
            "gate_masks": batch.gate_masks.detach().contiguous().numpy(),
            "returns": batch.returns.detach().contiguous().numpy(),
        }
        self._copied = None  # event: the last copies out of the staging are done

    def field_rows(self, name: str, sel: torch.Tensor) -> torch.Tensor:
        """`batch.<name>[sel]` on the learner device (any number of rows, a
        whole minibatch) for "gate_masks" / "returns" -- the micro-batching
        fold denominator's and q-norm variance's input, gathered in parallel
        instead of by a single-threaded CPU tensor index."""
        src = self._host_fields[name]
        rows = np.ascontiguousarray(sel.detach().cpu().numpy(), dtype=np.int64)
        out = np.empty((rows.shape[0],) + src.shape[1:], dtype=src.dtype)
        _gather_rows(src, rows, out)
        return torch.from_numpy(out).to(self.device)

    def gate_mask_rows(self, sel: torch.Tensor) -> torch.Tensor:
        """`batch.gate_masks[sel]` on the learner device (see field_rows)."""
        return self.field_rows("gate_masks", sel)

    def gather(self, sel: torch.Tensor) -> Batch:
        rows = np.ascontiguousarray(sel.detach().cpu().numpy(), dtype=np.int64)
        k = int(rows.shape[0])
        if k > self.cap:
            raise ValueError(f"HostBatchLoader: {k} rows > capacity {self.cap}")
        if self._copied is not None:
            self._copied.synchronize()
        # Every field's rows in ONE parallel pass (field by field, the small
        # fields stayed below the parallel threshold: a 1M-row chunk of a
        # ~360M-row batch is ~17M random reads, and single-threaded they made
        # host-batch PPO ~3x slower than device-resident PPO).
        _gather_rows_multi(
            [(src, stage.numpy().reshape(self.cap, -1)) for _, src, stage in self._fields],
            rows,
        )
        out: dict[str, torch.Tensor] = {}
        for name, _, stage in self._fields:
            out[name] = stage[:k].to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            ev = torch.cuda.Event()
            ev.record()
            self._copied = ev
        if self.layout is not None:
            obs = _unpack_compact(out.pop("obs_bits"), out.pop("obs_real"), self.layout)
        else:
            obs = out.pop("obs")
        return Batch(
            obs=obs,
            **{f: out[f] for f in _BATCH_TENSOR_FIELDS[1:]},
            is_terminal=out.get("is_terminal"),
            ent_coef_rows=out.get("ent_coef_rows"),
        )


def gather_minibatch(batch: Batch, sel: torch.Tensor) -> Batch:
    """The rows `sel` of `batch` as a Batch (compact observations unpacked)."""
    return Batch(
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
