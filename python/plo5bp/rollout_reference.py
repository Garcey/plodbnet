"""numpy reference implementations of the batched rollout's Rust kernels.

The batched collector (`rollout.collect_rollout_batched`) runs three per-step
kernels in the engine (plo5bp._engine):

- `record_learner_steps`   -- step3c: the learner rows' trajectory record;
- `aggression_record_batch` -- step8: the per-step cost / pot / street record
  (and the long-retired aggression bonus, 0 in every current run);
- `flush_trajectories`     -- step9b-9d: every finished hand's retroactive-bonus
  qualification (the F/T/R "winning aggression" counters), GAE / VRPO backward
  scans and the gathers into the output slabs.

Each Rust kernel reproduces numpy's exact float32 operation order; the numpy
code it replaced lives HERE, with the kernel's exact signature and return
value, as the oracle:

- tests swap it in (``PLO5BP_NUMPY_FLUSH=1`` makes the collector call these
  instead of the engine -- tests/python/training/test_rust_flush.py pins every Batch
  field bitwise equal both ways);
- a debugging A/B needs nothing else.

Nothing in a production run calls this module (2026-09-28, ML-029: these
blocks used to sit inside the collector's hot loop as fallbacks for older
engines and dense storage; the engine is always current now -- engine_abi --
and dense storage goes through the same Rust flush, see
`rollout._flush_pools`).
"""

from __future__ import annotations

import numpy as np

from plo5bp._engine import compute_aggression_bonus_batch  # type: ignore[attr-defined]
from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE


def vrpo_advantage_scan(
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
    in tests/python/training/test_vrpo_advantage.py.

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


def record_learner_steps(
    slot: np.ndarray,
    learner_idx: np.ndarray,
    pool_start: int,
    traj: dict,
    per_env: dict,
) -> None:
    """The engine's `record_learner_steps`: learner row i (env
    `learner_idx[i]`) goes to FLAT trajectory slot `slot[i]`; `traj` holds the
    flat trajectory arrays (`rollout._flat_traj_view`), `per_env` the step's
    per-env arrays (gate u8, chips u64, sizing (N, 4) i64, anchor i64, u /
    log_p / gate_lp / anchor_lp / value f32, and q_taken / vpi f32 when `traj`
    has them)."""
    rows = learner_idx
    l_gates = per_env["gate"][rows]
    traj["obs_idx"][slot] = np.arange(
        int(pool_start), int(pool_start) + int(rows.size), dtype=np.int64
    )
    traj["gate"][slot] = l_gates.astype(np.int8)
    traj["chips"][slot] = np.where(
        l_gates == GATE_RAISE, per_env["chips"][rows].astype(np.int64), 0
    )
    traj["sizing"][slot] = per_env["sizing"][rows]
    traj["anchor"][slot] = per_env["anchor"][rows].astype(np.int8)
    traj["u"][slot] = per_env["u"][rows]
    traj["log_p"][slot] = per_env["log_p"][rows]
    traj["gate_lp"][slot] = per_env["gate_lp"][rows]
    traj["anchor_lp"][slot] = per_env["anchor_lp"][rows]
    traj["value"][slot] = per_env["value"][rows]
    if "q_taken" in traj:
        traj["q_taken"][slot] = per_env["q_taken"][rows]
        traj["vpi"][slot] = per_env["vpi"][rows]


def aggression_record_batch(
    actors: np.ndarray,
    dones: np.ndarray,
    learner_mask: np.ndarray,
    gates: np.ndarray,
    pre_total_commit: np.ndarray,
    post_total_commit: np.ndarray,
    pre_bet_to_call: np.ndarray,
    pre_street_commit: np.ndarray,
    pre_street: np.ndarray,
    c: float,
    reward_norm: float,
    costs: np.ndarray,
    pots: np.ndarray,
    streets: np.ndarray,
    traj_lengths: np.ndarray,
    traj_cap: int,
) -> tuple:
    """The engine's `aggression_record_batch`: `compute_aggression_bonus_batch`
    (the per-env step values) + their record at each acting learner seat's
    current slot of the FLAT `costs` / `pots` / `streets` arrays, advancing
    `traj_lengths`. Returns (bonus total bb, steps, bonus steps, steps by
    street, bonus steps by street)."""
    agg = compute_aggression_bonus_batch(
        actors, dones, learner_mask, gates, pre_total_commit, post_total_commit,
        pre_bet_to_call, pre_street_commit, pre_street, float(c), float(reward_norm),
    )
    n_seats = int(traj_lengths.shape[1])
    valid_idx = np.nonzero(np.asarray(agg["valid"], dtype=bool))[0]
    if valid_idx.size:
        cost_inc_arr = np.asarray(agg["cost_increment"], dtype=np.float32)
        pot_pre_bb_arr = np.asarray(agg["pot_pre_bb"], dtype=np.float32)
        street_pre_arr = np.asarray(agg["street_pre"], dtype=np.int8)
        safe_actors = np.where(actors >= 0, actors, 0).astype(np.intp)
        valid_actors = safe_actors[valid_idx]
        valid_slots = traj_lengths[valid_idx, valid_actors]
        vw = (valid_idx * n_seats + valid_actors) * int(traj_cap) + valid_slots
        costs[vw] = cost_inc_arr[valid_idx]
        pots[vw] = pot_pre_bb_arr[valid_idx]
        streets[vw] = street_pre_arr[valid_idx]
        traj_lengths[valid_idx, valid_actors] += 1
    return (
        float(agg["total_bonus_bb"]),
        int(agg["total_steps"]),
        int(agg["bonus_steps"]),
        tuple(int(x) for x in np.asarray(agg["steps_by_street"])),
        tuple(int(x) for x in np.asarray(agg["bonus_steps_by_street"])),
    )


def flush_trajectories(
    term_envs: np.ndarray,
    lengths: np.ndarray,
    flush_mask: np.ndarray,
    won_bb: np.ndarray,
    share_gt: np.ndarray,
    share_eq: np.ndarray,
    traj: dict,
    traj_cap: int,
    pools: dict,
    holes_rot: np.ndarray,
    gamma: np.float32,
    lam: np.float32,
    retro_c: np.float32,
    out: dict,
) -> tuple:
    """The engine's `flush_trajectories`: flush the finished hands
    `term_envs` into `out` (views of exactly the rows being written).

    Inputs as the engine takes them: `lengths` / `flush_mask` / `won_bb` /
    `share_gt` / `share_eq` are (T, S); `traj` the FLAT trajectory arrays;
    `pools` the per-step observation pool as {"obs_bits" (P, nb) u8,
    "obs_real" (P, nr) f32, "gm" (P, 3) bool}; `holes_rot` (n_envs * S,
    5 * hole_w) u8; `out` carries "obs_bits", "obs_real" (f32) or
    "obs_real_f16" (the float16 slab as its u16 view), and gm / ga / rc / sz /
    an / ru / oh / lp / glp / alp / v / ret / adv / last. Returns (rows
    written, qualifying bonus steps, per-street (flop, turn, river) counts,
    bonus total).

    Every (T, S, L) temporary is bounded by L = the longest trajectory in THIS
    flush (typically 8-16 actions), not the trajectory capacity: slots in
    [length, L) are inactive by construction, so the output is bit-identical
    to a scan over the full capacity."""
    T, S = int(lengths.shape[0]), int(lengths.shape[1])
    cap = int(traj_cap)
    L = int(lengths.max()) if lengths.size else 0
    t_idx = np.arange(L, dtype=np.int32)
    active_step = (
        (t_idx[None, None, :] < lengths[..., None])
        & flush_mask[..., None]
    )  # (T, S, L)

    def window(name: str) -> np.ndarray:
        """(T, S, L) copy of the flushed hands' slots of a trajectory array."""
        a = traj[name]
        return a.reshape((-1, S, cap) + a.shape[1:])[term_envs, :, :L]

    # step9b: retroactive-bonus qualification (the F/T/R counters) and, with a
    # nonzero coefficient, the bonus itself.
    gates_t = window("gate")
    costs_t = window("costs")
    pots_t = window("pots")
    streets_t = window("streets")
    is_raise = gates_t == GATE_RAISE
    is_call_with_chips = (gates_t == GATE_CHECK_CALL) & (costs_t < 0.0)
    qualified = active_step & (
        (share_gt[..., None] & is_raise)
        | (share_eq[..., None] & (is_raise | is_call_with_chips))
    )
    bonus_steps = int(qualified.sum())
    by_street = tuple(
        int(((streets_t == (s_idx + 1)) & qualified).sum()) for s_idx in range(3)
    )
    bonus_total = 0.0
    if float(retro_c) != 0.0:
        bonus = qualified.astype(np.float32) * (np.float32(retro_c) * pots_t)
        costs_t += bonus
        bonus_total = float(bonus.sum())

    # step9c: GAE backward scan vectorized over (T, S).
    vals_t = window("value")
    last_gae = np.zeros((T, S), dtype=np.float32)
    advs_t = np.zeros((T, S, L), dtype=np.float32)
    last_t_arr = lengths.astype(np.int32) - 1
    gamma_f = np.float32(gamma)
    lam_f = np.float32(lam)
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
    # VRPO: `advantages` use the Expected-SARSA form, `returns` stay GAE (the
    # V-head target).
    if "q_taken" in traj:
        adv_out_t = vrpo_advantage_scan(
            costs_t, won_bb, window("q_taken"), window("vpi"),
            last_t_arr, flush_mask, gamma_f, lam_f,
        )
    else:
        adv_out_t = advs_t

    # step9d: gather the (T, S, L) window's active slots, in C order, into
    # the output rows.
    flat = active_step.ravel()
    n_new = int(flat.sum())
    if n_new:
        sel = np.nonzero(flat)[0]
        # Each selected (t, s, l) -> its flat trajectory slot
        # ((env * S + seat) * cap + l).
        ts_idx = sel // L
        l_sel = sel - ts_idx * L
        t_sel = ts_idx // S
        es_sel = term_envs[t_sel] * S + (ts_idx - t_sel * S)
        tsel = es_sel * cap + l_sel
        obs_idx = traj["obs_idx"][tsel]
        np.take(pools["obs_bits"], obs_idx, axis=0, out=out["obs_bits"])
        if "obs_real_f16" in out:
            # numpy's round-to-nearest-even float16 cast (the engine's too).
            out["obs_real_f16"][:] = (
                pools["obs_real"][obs_idx].astype(np.float16).view(np.uint16)
            )
        else:
            np.take(pools["obs_real"], obs_idx, axis=0, out=out["obs_real"])
        np.take(pools["gm"], obs_idx, axis=0, out=out["gm"])
        out["ga"][:] = traj["gate"][tsel]
        out["rc"][:] = traj["chips"][tsel]
        out["sz"][:] = traj["sizing"][tsel]
        out["an"][:] = traj["anchor"][tsel]
        out["ru"][:] = traj["u"][tsel]
        hole_w = int(out["oh"].shape[2])
        out["oh"][:] = holes_rot.reshape(-1, 5, hole_w)[es_sel]
        out["lp"][:] = traj["log_p"][tsel]
        out["glp"][:] = traj["gate_lp"][tsel]
        out["alp"][:] = traj["anchor_lp"][tsel]
        out["v"][:] = vals_t.ravel()[sel]
        out["ret"][:] = rets_t.ravel()[sel]
        out["adv"][:] = adv_out_t.ravel()[sel]
        out["last"][:] = l_sel == last_t_arr.ravel()[ts_idx]
    return n_new, bonus_steps, by_street, bonus_total
