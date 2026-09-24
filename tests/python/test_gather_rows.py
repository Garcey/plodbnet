"""Parallel per-step host copies (2026-09-24).

`gather_rows_into` (the engine's parallel byte-row copy) must write exactly
the bytes numpy's row gather / slice copy writes, for every dtype the rollout
feeds it, and refuse bad shapes instead of writing out of bounds; the
rollout's `_gather_rows` wrapper must give the same array on either path.
`record_learner_steps` must write the numpy assignments' values whether it
takes its parallel path (>= 2 * REC_MIN_LEN strictly increasing slots) or the
sequential one (any other order).
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp import rollout as R
from plo5bp._engine import gather_rows_into, record_learner_steps

GATE_RAISE = 2


def _arrays(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    return {
        "u8": rng.integers(0, 256, size=(n, 88), dtype=np.uint8),
        "f32": rng.standard_normal((n, 92)).astype(np.float32),
        "bool": rng.random((n, 3)) < 0.5,
        "i64": rng.integers(-(2**40), 2**40, size=(n, 4), dtype=np.int64),
    }


@pytest.mark.parametrize("kind", ["u8", "f32", "bool", "i64"])
def test_gather_matches_numpy(kind: str) -> None:
    rng = np.random.default_rng(1)
    src = _arrays(rng, 7000)[kind]
    rows = rng.integers(0, src.shape[0], size=5000).astype(np.int64)  # repeats too
    for start in (0, 17):
        want = np.zeros((6000,) + src.shape[1:], dtype=src.dtype)
        want[start : start + rows.size] = src[rows]
        got = np.zeros_like(want)
        gather_rows_into(src.view(np.uint8), got.view(np.uint8), start, rows)
        assert np.array_equal(got, want)
        # rows=None: the first len(src) rows, in order (a contiguous copy).
        part = src[:3000]
        want2 = np.zeros_like(want)
        want2[start : start + 3000] = part
        got2 = np.zeros_like(want)
        gather_rows_into(part.view(np.uint8), got2.view(np.uint8), start)
        assert np.array_equal(got2, want2)


def test_gather_refuses_bad_shapes() -> None:
    src = np.zeros((10, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="row widths"):
        gather_rows_into(src, np.zeros((10, 9), dtype=np.uint8), 0)
    with pytest.raises(ValueError, match="out of range"):
        gather_rows_into(src, np.zeros((10, 8), dtype=np.uint8), 0, np.array([3, 10]))
    with pytest.raises(ValueError, match="overflow"):
        gather_rows_into(src, np.zeros((12, 8), dtype=np.uint8), 5)
    with pytest.raises(ValueError, match="C-contiguous"):
        gather_rows_into(src[:, ::2], np.zeros((10, 4), dtype=np.uint8), 0)


@pytest.mark.parametrize("kind", ["u8", "f32", "bool", "i64"])
def test_rollout_wrapper_same_on_both_paths(kind: str, monkeypatch) -> None:
    rng = np.random.default_rng(2)
    src = _arrays(rng, 9000)[kind]
    rows = np.sort(rng.choice(src.shape[0], size=4000, replace=False)).astype(np.int64)
    outs = []
    for min_rows in (1, 10**9):  # Rust path, numpy path
        monkeypatch.setattr(R, "_GATHER_RUST_MIN_ROWS", min_rows)
        a = np.zeros((5000,) + src.shape[1:], dtype=src.dtype)
        R._gather_rows(src, rows, a, 9)
        b = np.zeros((5000,) + src.shape[1:], dtype=src.dtype)
        R._gather_rows(src[:4500], None, b, 3)
        outs.append((a, b))
    assert np.array_equal(outs[0][0], outs[1][0])
    assert np.array_equal(outs[0][1], outs[1][1])


def _record_case(rng: np.random.Generator, n_env: int, k: int, ascending: bool, vrpo: bool):
    m = n_env * 6 * 16
    per_env = {
        "gate": rng.integers(0, 3, size=n_env).astype(np.uint8),
        "chips": rng.integers(0, 2**40, size=n_env).astype(np.uint64),
        "sizing": rng.integers(0, 2**40, size=(n_env, 4)).astype(np.int64),
        "anchor": rng.integers(-1, 11, size=n_env).astype(np.int64),
    }
    keys = ["u", "log_p", "gate_lp", "anchor_lp", "value"] + (["q_taken", "vpi"] if vrpo else [])
    for key in keys:
        per_env[key] = rng.standard_normal(n_env).astype(np.float32)
    traj = {
        "obs_idx": np.full(m, -7, dtype=np.int64),
        "gate": np.full(m, -3, dtype=np.int8),
        "chips": np.full(m, -5, dtype=np.int64),
        "sizing": np.full((m, 4), -9, dtype=np.int64),
        "anchor": np.full(m, -2, dtype=np.int8),
    }
    for key in keys:
        traj[key] = np.full(m, np.nan, dtype=np.float32)
    lidx = np.sort(rng.choice(n_env, size=k, replace=False)).astype(np.int64)
    within = rng.integers(0, 6 * 16, size=k)
    slot = (lidx * 6 * 16 + within).astype(np.int64)  # strictly increasing
    if not ascending:
        perm = rng.permutation(k)
        slot, lidx = slot[perm], lidx[perm]
    return per_env, traj, keys, slot, lidx


def _record_reference(per_env, traj, keys, slot, lidx, pool_start):
    t = {key: v.copy() for key, v in traj.items()}
    g = per_env["gate"][lidx]
    t["obs_idx"][slot] = pool_start + np.arange(slot.size)
    t["gate"][slot] = g.astype(np.int8)
    t["chips"][slot] = np.where(g == GATE_RAISE, per_env["chips"][lidx].astype(np.int64), 0)
    t["sizing"][slot] = per_env["sizing"][lidx]
    t["anchor"][slot] = per_env["anchor"][lidx].astype(np.int8)
    for key in keys:
        t[key][slot] = per_env[key][lidx]
    return t


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("vrpo", [True, False])
def test_record_learner_steps_parallel_and_sequential(ascending: bool, vrpo: bool) -> None:
    rng = np.random.default_rng(3 + 2 * int(ascending) + int(vrpo))
    per_env, traj, keys, slot, lidx = _record_case(rng, 12000, 6000, ascending, vrpo)
    want = _record_reference(per_env, traj, keys, slot, lidx, 1234)
    got = {key: v.copy() for key, v in traj.items()}
    flat = {key: (v.reshape(-1, 4) if key == "sizing" else v) for key, v in got.items()}
    record_learner_steps(slot, lidx, 1234, flat, per_env)
    for key in want:
        assert np.array_equal(got[key], want[key], equal_nan=True), key
