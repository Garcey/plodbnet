"""In-place FULL encoder (2026-09-26) is bit-identical to the copying one.

`observation_encoded_into` writes the (N, OBS_DIM) rows of the full layout
straight into the env's cached obs buffer instead of returning a fresh array
the env then copies (`BatchedBombPotEnv._refresh` uses it when
PLO5_RUST_ENCODER=1 and obs_mode=full) — at 29k envs the fresh array was
~137 MB of page faults + a copy on every step of the full-obs runs.

Pinned here, as for the minimal encoders (test_minimal_encoder_into.py):
1. engine level -- the same bits as `observation_encoded_batch` whatever
   garbage the buffer held; rows with a False `encode_mask` entry zeroed; the
   aux dict identical;
2. env level -- an env on the in-place path and one forced onto the numpy
   reference encoder (2026-09-28: the env's only Rust path is the in-place
   `BatchedEngine.encode`; the copying one is gone) stay identical (every cached array, obs bitwise) through a
   rollout-shaped drive, and the obs buffer keeps its identity;
3. argument validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp._engine import BatchedEngine
from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.env_batched import BatchedBombPotEnv

pytestmark = pytest.mark.skipif(
    not hasattr(BatchedEngine, "observation_encoded_into"),
    reason="in-place full encoder not in this engine build",
)

_AUX = ("actor", "legal_mask", "min_raise", "max_raise", "total_commit",
        "bet_to_call", "street_commit", "street", "pot")
_CACHED = ("_legal", "_gate_mask", "_min_raise", "_max_raise", "_actors", "_dones",
           "_total_commit", "_bet_to_call", "_street_commit", "_street", "_pot")


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _full_env(n: int, cfg: GameConfig, monkeypatch, *, into: bool) -> BatchedBombPotEnv:
    monkeypatch.setenv("PLO5_RUST_ENCODER", "1")
    env = BatchedBombPotEnv(n, cfg, obs_mode="full", opp_outcome_mc=64)
    assert env._use_rust_encoder and env._encode_layout == "full"
    if not into:
        env._use_rust_encoder = False  # the numpy reference encoder
    return env


def _legal_actions(env: BatchedBombPotEnv, rng: np.random.Generator):
    n = env.n
    gates = np.full(n, GATE_CHECK_CALL, dtype=np.uint8)
    chips = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        legal = np.nonzero(env._gate_mask[i])[0]
        if legal.size == 0:
            continue
        g = int(rng.choice(legal))
        gates[i] = g
        if g == GATE_RAISE:
            lo, hi = int(env._min_raise[i]), int(env._max_raise[i])
            chips[i] = np.uint64(lo if hi <= lo else int(rng.integers(lo, hi + 1)))
    return gates, chips


@pytest.mark.parametrize("seats", [2, 4, 6])
def test_engine_into_matches_the_copying_encoder(seats, monkeypatch) -> None:
    cfg = GameConfig(num_seats=seats, starting_stack=400000, ante=30000, bb=10000)
    env = _full_env(48, cfg, monkeypatch, into=True)
    rng = np.random.default_rng(seats)
    env.reset_batch(rng.integers(0, 2**63 - 1, size=48, dtype=np.int64).astype(np.uint64),
                    rng.integers(0, seats, size=48).astype(np.uint8))
    for step in range(12):
        ref = env._be.observation_encoded_batch()
        want = np.asarray(ref["obs"], dtype=np.float32)
        mask = rng.random(48) < 0.7
        buf = rng.standard_normal((48, OBS_DIM)).astype(np.float32)  # garbage it must overwrite
        aux = env._be.observation_encoded_into(buf, mask)
        assert "obs" not in aux
        assert np.array_equal(_bits(buf[mask]), _bits(want[mask])), f"step {step}"
        assert not buf[~mask].any(), "a skipped row is zero-filled"
        buf2 = rng.standard_normal((48, OBS_DIM)).astype(np.float32)
        env._be.observation_encoded_into(buf2)  # no mask = every row
        assert np.array_equal(_bits(buf2), _bits(want))
        for k in _AUX:
            assert np.array_equal(np.asarray(aux[k]), np.asarray(ref[k])), k
        gates, chips = _legal_actions(env, rng)
        env.step_hybrid_batch(gates, chips)


@pytest.mark.parametrize("seats,n,seed", [(2, 40, 1), (3, 64, 2), (6, 64, 3)])
def test_env_in_place_path_matches_the_copying_path(seats, n, seed, monkeypatch) -> None:
    cfg = GameConfig(num_seats=seats, starting_stack=400000, ante=30000, bb=10000)
    fast = _full_env(n, cfg, monkeypatch, into=True)
    slow = _full_env(n, cfg, monkeypatch, into=False)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, seats, size=n).astype(np.uint8)
    fast.reset_batch(seeds, buttons)
    slow.reset_batch(seeds, buttons)
    buf = fast._obs

    def same(ctx: str) -> None:
        assert fast._obs is buf, f"{ctx}: obs buffer was replaced"
        assert np.array_equal(_bits(fast._obs), _bits(slow._obs)), f"{ctx}: obs"
        for name in _CACHED:
            a, b = getattr(fast, name), getattr(slow, name)
            assert a.dtype == b.dtype and np.array_equal(a, b), f"{ctx}: {name}"

    same("reset")
    partial = resets = 0
    for step in range(40):
        gates, chips = _legal_actions(fast, rng)
        nt_f = np.asarray(fast._be.apply_hybrid_batch(gates, chips), dtype=bool)
        nt_s = np.asarray(slow._be.apply_hybrid_batch(gates, chips), dtype=bool)
        assert np.array_equal(nt_f, nt_s)
        if nt_f.all():
            fast._refresh()
            slow._refresh()
        else:
            partial += int(nt_f.any())
            fast._refresh(encode_mask=~nt_f)
            slow._refresh(encode_mask=~nt_f)
        same(f"step {step} refresh")
        if nt_f.any():
            ns = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
            nb = rng.integers(0, seats, size=n).astype(np.uint8)
            fast._be.reset_terminal_batch(ns, nb, nt_f)
            slow._be.reset_terminal_batch(ns, nb, nt_f)
            fast._refresh_subset(nt_f)
            slow._refresh_subset(nt_f)
            resets += 1
            same(f"step {step} refresh_subset")
    assert partial > 0 and resets > 0


def test_into_argument_validation(monkeypatch) -> None:
    cfg = GameConfig(num_seats=3, starting_stack=400000, ante=30000, bb=10000)
    env = _full_env(8, cfg, monkeypatch, into=True)
    with pytest.raises(ValueError, match="C-contiguous"):
        env._be.observation_encoded_into(np.zeros((8, OBS_DIM - 1), np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        env._be.observation_encoded_into(np.zeros((OBS_DIM, 8), np.float32).T)
    with pytest.raises(ValueError, match="encode_mask"):
        env._be.observation_encoded_into(np.zeros((8, OBS_DIM), np.float32), np.ones(7, bool))


# --------------------------------------------------------------------------
# Packed outputs for the FULL layout (2026-09-26): the in-place full encoders
# also write the compact packed copy (or only it, out=None), like the minimal
# ones -- the rollout's uploads gather those packed rows instead of re-reading
# and packing the ~4.7 KB dense rows.
# --------------------------------------------------------------------------

from plo5bp.compact_obs import FULL_LAYOUT, pack_rows_np  # noqa: E402


def _assert_full_packed_copy(env: BatchedBombPotEnv, ctx: str) -> None:
    bits, real = pack_rows_np(env._obs, np.arange(env.n), FULL_LAYOUT)
    assert np.array_equal(env._obs_bits, bits), f"{ctx}: bits"
    assert np.array_equal(_bits(env._obs_real), _bits(real)), f"{ctx}: reals"


@pytest.mark.parametrize("seats,n,seed", [(2, 24, 41), (6, 40, 42)])
def test_full_env_packed_copy_tracks_every_refresh(seats, n, seed, monkeypatch) -> None:
    cfg = GameConfig(num_seats=seats, starting_stack=400000, ante=30000, bb=10000)
    env = _full_env(n, cfg, monkeypatch, into=True)
    rng = np.random.default_rng(seed)
    env.reset_batch(
        rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64),
        rng.integers(0, seats, size=n).astype(np.uint8),
    )
    assert env.enable_packed_obs(FULL_LAYOUT) and env.packed_obs() is not None
    _assert_full_packed_copy(env, "enable")
    for step in range(40):
        gates, chips = _legal_actions(env, rng)
        nt = np.asarray(env._be.apply_hybrid_batch(gates, chips), dtype=bool)
        if step % 7 == 3:
            env._refresh(encode_mask=np.zeros(n, dtype=bool))  # all-skipped branch
        else:
            env._refresh(encode_mask=~nt)
        _assert_full_packed_copy(env, f"step {step} refresh")
        if nt.any():
            env._be.reset_terminal_batch(
                rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64),
                rng.integers(0, seats, size=n).astype(np.uint8), nt,
            )
            env._refresh_subset(nt)
            _assert_full_packed_copy(env, f"step {step} refresh_subset")


@pytest.mark.parametrize("seats,n,seed", [(2, 24, 51), (6, 40, 52)])
def test_full_env_packed_only_tracks_dense_twin(seats, n, seed, monkeypatch) -> None:
    cfg = GameConfig(num_seats=seats, starting_stack=400000, ante=30000, bb=10000)
    lean = _full_env(n, cfg, monkeypatch, into=True)
    twin = _full_env(n, cfg, monkeypatch, into=True)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, seats, size=n).astype(np.uint8)
    for e in (lean, twin):
        e.reset_batch(seeds, buttons)
    assert lean.enable_packed_obs(FULL_LAYOUT, dense=False)
    assert twin.enable_packed_obs(FULL_LAYOUT)
    assert np.isnan(lean._obs).all()

    def same(ctx: str) -> None:
        assert np.array_equal(lean._obs_bits, twin._obs_bits), f"{ctx}: bits"
        assert np.array_equal(_bits(lean._obs_real), _bits(twin._obs_real)), f"{ctx}: reals"
        for name in _CACHED:
            assert np.array_equal(getattr(lean, name), getattr(twin, name)), f"{ctx}: {name}"
        assert np.isnan(lean._obs).all(), f"{ctx}: dense rows were written"

    same("enable")
    for step in range(40):
        gates, chips = _legal_actions(twin, rng)
        nt = np.asarray(lean._be.apply_hybrid_batch(gates, chips), dtype=bool)
        assert np.array_equal(nt, np.asarray(twin._be.apply_hybrid_batch(gates, chips), dtype=bool))
        mask = np.zeros(n, dtype=bool) if step % 7 == 3 else ~nt
        lean._refresh(encode_mask=mask)
        twin._refresh(encode_mask=mask)
        same(f"step {step} refresh")
        if nt.any():
            ns = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
            nb = rng.integers(0, seats, size=n).astype(np.uint8)
            for e in (lean, twin):
                e._be.reset_terminal_batch(ns, nb, nt)
                e._refresh_subset(nt)
            same(f"step {step} refresh_subset")
    s_lean = lean.reset_batch(seeds, buttons)
    s_twin = twin.reset_batch(seeds, buttons)
    assert np.array_equal(_bits(s_lean.obs), _bits(s_twin.obs))
    same("reset_batch")
    assert lean.enable_packed_obs(FULL_LAYOUT, dense=True)
    assert np.array_equal(_bits(lean._obs), _bits(twin._obs))
