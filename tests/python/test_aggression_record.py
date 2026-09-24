"""Fused aggression bonus + trajectory record (2026-09-24).

`aggression_record_batch` must leave the flat trajectory arrays, the per-seat
slot counts and every counter exactly as the numpy path does
(`compute_aggression_bonus_batch` + the fancy-index writes in the rollout's
step8), with the bonus on and off, and refuse a slot outside the capacity.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp._engine import compute_aggression_bonus_batch

engine = pytest.importorskip("plo5bp._engine")
aggression_record_batch = getattr(engine, "aggression_record_batch", None)
pytestmark = pytest.mark.skipif(
    aggression_record_batch is None, reason="engine without aggression_record_batch"
)


def _case(rng: np.random.Generator, n: int, s: int, cap: int):
    actors = rng.integers(-1, s, size=n).astype(np.int8)
    dones = rng.random(n) < 0.1
    lm = rng.random((n, s)) < 0.7
    gates = rng.integers(0, 3, size=n).astype(np.uint8)
    pre_tc = rng.integers(0, 400_000, size=(n, s)).astype(np.int64)
    post_tc = pre_tc + rng.integers(0, 200_000, size=(n, s)).astype(np.int64)
    pre_btc = rng.integers(0, 100_000, size=n).astype(np.uint64)
    pre_sc = rng.integers(0, 100_000, size=(n, s)).astype(np.uint64)
    pre_street = rng.integers(0, 4, size=n).astype(np.uint8)
    lengths = rng.integers(0, cap - 1, size=(n, s)).astype(np.int32)
    traj = {
        "costs": rng.standard_normal(n * s * cap).astype(np.float32),
        "pots": rng.standard_normal(n * s * cap).astype(np.float32),
        "streets": rng.integers(-3, 3, size=n * s * cap).astype(np.int8),
    }
    return (actors, dones, lm, gates, pre_tc, post_tc, pre_btc, pre_sc, pre_street), lengths, traj


def _numpy_path(args, c, norm, lengths, traj, s, cap):
    actors = args[0]
    agg = compute_aggression_bonus_batch(*args, c, norm)
    valid_idx = np.nonzero(np.asarray(agg["valid"], dtype=bool))[0]
    if valid_idx.size:
        cost = np.asarray(agg["cost_increment"], dtype=np.float32)
        pot = np.asarray(agg["pot_pre_bb"], dtype=np.float32)
        street = np.asarray(agg["street_pre"], dtype=np.int8)
        va = np.where(actors >= 0, actors, 0).astype(np.intp)[valid_idx]
        vw = (valid_idx * s + va) * cap + lengths[valid_idx, va]
        traj["costs"][vw] = cost[valid_idx]
        traj["pots"][vw] = pot[valid_idx]
        traj["streets"][vw] = street[valid_idx]
        lengths[valid_idx, va] += 1
    return (
        float(agg["total_bonus_bb"]), int(agg["total_steps"]), int(agg["bonus_steps"]),
        tuple(int(x) for x in np.asarray(agg["steps_by_street"])),
        tuple(int(x) for x in np.asarray(agg["bonus_steps_by_street"])),
    )


@pytest.mark.parametrize("c", [0.0, 0.35])
@pytest.mark.parametrize("n,s", [(9000, 6), (40, 2)])
def test_fused_matches_numpy_path(c, n, s) -> None:
    rng = np.random.default_rng(int(c * 100) + n + s)
    cap = 16
    args, lengths, traj = _case(rng, n, s, cap)
    norm = 1.0 / 10_000.0
    l0, t0 = lengths.copy(), {k: v.copy() for k, v in traj.items()}
    want = _numpy_path(args, c, norm, l0, t0, s, cap)
    l1, t1 = lengths.copy(), {k: v.copy() for k, v in traj.items()}
    got = aggression_record_batch(*args, c, norm, t1["costs"], t1["pots"], t1["streets"], l1, cap)
    assert got[1:] == want[1:]
    assert got[0] == want[0]  # the bonus total: same terms, same (env) order
    assert np.array_equal(l1, l0)
    for k in traj:
        assert np.array_equal(t1[k].view(np.uint8), t0[k].view(np.uint8)), k


def test_slot_outside_capacity_is_refused() -> None:
    rng = np.random.default_rng(5)
    args, lengths, traj = _case(rng, 50, 3, 8)
    actors, dones, lm = args[0], args[1], args[2]
    i = int(np.nonzero((actors >= 0) & ~dones & lm[np.arange(50), np.maximum(actors, 0)])[0][0])
    lengths[i, actors[i]] = 8
    with pytest.raises(ValueError, match="outside"):
        aggression_record_batch(*args, 0.0, 1e-4, traj["costs"], traj["pots"], traj["streets"], lengths, 8)
