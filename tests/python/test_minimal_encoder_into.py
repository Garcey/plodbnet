"""In-place minimal encoders (2026-09-23) are bit-identical to the copying ones.

`observation_encoded_minimal_into` / `observation_encoded_minimal_subset_into`
write the 796-dim rows straight into the env's cached obs buffer instead of
returning a fresh array the env then copies (`BatchedBombPotEnv._refresh` /
`_refresh_subset` use them when PLO5_RUST_ENCODER=1 and obs_mode=minimal).

Pinned here:
1. engine level -- the same bits as the copying encoders whatever garbage the
   buffer held before; rows with a False `encode_mask` entry zeroed; rows a
   subset call did not select left untouched; the aux dict identical;
2. env level -- an env on the in-place path and one forced onto the copying
   path stay identical (every cached array, obs compared bitwise) through a
   rollout-shaped drive: apply -> refresh(encode_mask=~newly_terminal) ->
   reset_terminal_batch -> refresh_subset; and the obs buffer keeps its
   identity (the rollout holds `obs = env._obs` across the refresh);
3. argument validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp._engine import BatchedEngine
from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.env_batched import BatchedBombPotEnv

pytestmark = pytest.mark.skipif(
    not hasattr(BatchedEngine, "observation_encoded_minimal_into"),
    reason="in-place minimal encoder not in this engine build",
)

_AUX = (
    "actor",
    "legal_mask",
    "min_raise",
    "max_raise",
    "total_commit",
    "bet_to_call",
    "street_commit",
    "street",
    "pot",
)

_CACHED = (
    "_legal",
    "_gate_mask",
    "_min_raise",
    "_max_raise",
    "_actors",
    "_dones",
    "_total_commit",
    "_bet_to_call",
    "_street_commit",
    "_street",
    "_pot",
)


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _minimal_env(n: int, cfg: GameConfig, monkeypatch, *, into: bool) -> BatchedBombPotEnv:
    monkeypatch.setenv("PLO5_RUST_ENCODER", "1")
    env = BatchedBombPotEnv(n, cfg, obs_mode="minimal", opp_outcome_mc=0)
    assert env._rust_minimal and env._rust_minimal_into
    if not into:
        env._rust_minimal_into = False  # the pre-2026-09-23 copying path
    return env


def _legal_actions(env: BatchedBombPotEnv, rng: np.random.Generator):
    """A uniformly random LEGAL gate per env (raises at a random legal size);
    terminal envs get a no-op CheckCall."""
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


def _assert_aux_equal(a, b, ctx: str) -> None:
    for k in _AUX:
        av, bv = np.asarray(a[k]), np.asarray(b[k])
        assert av.dtype == bv.dtype and av.shape == bv.shape, f"{ctx}: {k}"
        assert np.array_equal(av, bv), f"{ctx}: aux {k} differs"


@pytest.mark.parametrize("seats", [2, 3, 6])
def test_engine_into_matches_copying_encoders(seats, monkeypatch) -> None:
    n = 40
    cfg = GameConfig(num_seats=seats, starting_stack=400000, ante=30000, bb=10000)
    env = _minimal_env(n, cfg, monkeypatch, into=True)
    rng = np.random.default_rng(1000 + seats)
    env.reset_batch(
        rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64),
        rng.integers(0, seats, size=n).astype(np.uint8),
    )
    be = env._be
    checked = 0
    for step in range(12):
        ref = be.observation_encoded_minimal_batch()
        ref_obs = np.asarray(ref["obs"])
        assert ref_obs.shape == (n, OBS_DIM_MINIMAL)

        # Full, no mask: every row rewritten, whatever the buffer held.
        out = np.full((n, OBS_DIM_MINIMAL), np.nan, dtype=np.float32)
        aux = be.observation_encoded_minimal_into(out)
        assert "obs" not in aux
        assert np.array_equal(_bits(out), _bits(ref_obs)), step
        _assert_aux_equal(aux, ref, f"into step {step}")

        # Full with a mask: kept rows encoded, skipped rows zero.
        keep = rng.random(n) < 0.6
        out = rng.standard_normal((n, OBS_DIM_MINIMAL)).astype(np.float32)
        aux = be.observation_encoded_minimal_into(out, keep)
        want = ref_obs.copy()
        want[~keep] = 0.0
        assert np.array_equal(_bits(out), _bits(want)), step
        _assert_aux_equal(aux, ref, f"masked into step {step}")

        # Subset: selected rows rewritten in place, every other row untouched.
        k = int(rng.integers(1, n + 1))
        idx = np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)
        garbage = rng.standard_normal((n, OBS_DIM_MINIMAL)).astype(np.float32)
        out = garbage.copy()
        aux = be.observation_encoded_minimal_subset_into(idx, out)
        ref_sub = be.observation_encoded_minimal_subset_batch(idx)
        want = garbage.copy()
        want[idx] = np.asarray(ref_sub["obs"])
        assert np.array_equal(_bits(out), _bits(want)), step
        _assert_aux_equal(aux, ref_sub, f"subset into step {step}")
        checked += 1

        gates, chips = _legal_actions(env, rng)
        env.step_hybrid_batch(gates, chips)
    assert checked == 12


def test_engine_into_empty_subset_is_noop(monkeypatch) -> None:
    cfg = GameConfig(num_seats=4, starting_stack=300000, ante=30000, bb=10000)
    env = _minimal_env(6, cfg, monkeypatch, into=True)
    env.reset_batch(np.arange(6, dtype=np.uint64) + 7, np.zeros(6, dtype=np.uint8))
    out = np.full((6, OBS_DIM_MINIMAL), 3.5, dtype=np.float32)
    aux = env._be.observation_encoded_minimal_subset_into(np.zeros(0, dtype=np.int64), out)
    assert np.all(out == 3.5)
    assert np.asarray(aux["actor"]).shape == (0,)


@pytest.mark.parametrize("seats,n,seed", [(2, 24, 11), (3, 32, 12), (6, 48, 13)])
def test_env_in_place_path_matches_copying_path(seats, n, seed, monkeypatch) -> None:
    cfg = GameConfig(num_seats=seats, starting_stack=400000, ante=30000, bb=10000)
    fast = _minimal_env(n, cfg, monkeypatch, into=True)
    slow = _minimal_env(n, cfg, monkeypatch, into=False)
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
    for step in range(60):
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
    cfg = GameConfig(num_seats=3, starting_stack=300000, ante=30000, bb=10000)
    env = _minimal_env(5, cfg, monkeypatch, into=True)
    env.reset_batch(np.arange(5, dtype=np.uint64) + 1, np.zeros(5, dtype=np.uint8))
    be = env._be
    d = OBS_DIM_MINIMAL
    with pytest.raises(ValueError, match="C-contiguous"):
        be.observation_encoded_minimal_into(np.zeros((4, d), dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        be.observation_encoded_minimal_into(np.asfortranarray(np.zeros((5, d), dtype=np.float32)))
    with pytest.raises(ValueError, match="encode_mask"):
        be.observation_encoded_minimal_into(np.zeros((5, d), dtype=np.float32), np.ones(4, dtype=bool))
    out = np.zeros((5, d), dtype=np.float32)
    for bad in ([3, 1], [1, 1], [0, 5], [-1, 2]):
        with pytest.raises(ValueError):
            be.observation_encoded_minimal_subset_into(np.asarray(bad, dtype=np.int64), out)
    with pytest.raises(ValueError, match="C-contiguous"):
        be.observation_encoded_minimal_subset_into(
            np.asarray([0, 2], dtype=np.int64), np.zeros((5, d + 1), dtype=np.float32)
        )
