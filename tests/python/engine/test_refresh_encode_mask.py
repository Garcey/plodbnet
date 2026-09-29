"""Parity: `_refresh(encode_mask=...)` vs full `_refresh` for live rows.

After apply, the rollout skips encoding newly-terminal envs (their obs is
never consumed before reset). Pack always runs full-batch. This test asserts:

1. Live (non-terminal) rows match a full-encode refresh bit-exactly.
2. Terminal rows are all-zero obs (same as full encode of actor == -1).
3. Engine caches (commit / legal / actors / …) match full refresh for ALL rows.
"""

from __future__ import annotations

import numpy as np

from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.env_batched import BatchedBombPotEnv

_CACHED_NON_OBS = (
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


def _sample_legal(env: BatchedBombPotEnv, rng: np.random.Generator):
    n = env.n
    gm = env._gate_mask
    gates = np.full(n, GATE_CHECK_CALL, dtype=np.uint8)
    chips = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        legal = np.nonzero(gm[i])[0]
        if legal.size == 0:
            continue
        g = int(rng.choice(legal))
        gates[i] = g
        if g == GATE_RAISE:
            lo = int(env._min_raise[i])
            hi = int(env._max_raise[i])
            if hi > lo:
                chips[i] = int(rng.integers(lo, hi + 1))
            else:
                chips[i] = hi
    return gates, chips


def test_encode_mask_matches_full_refresh_on_live_rows():
    cfg = GameConfig(num_seats=6)
    n = 32
    full = BatchedBombPotEnv(n, cfg)
    skip = BatchedBombPotEnv(n, cfg)
    rng = np.random.default_rng(20260712)
    seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, cfg.num_seats, size=n, dtype=np.int64).astype(np.uint8)
    full.reset_batch(seeds, buttons)
    skip.reset_batch(seeds, buttons)

    saw_partial = False
    for step in range(80):
        gates, chips = _sample_legal(full, rng)
        nt_f = np.asarray(
            full._be.apply_hybrid_batch(gates, chips), dtype=bool
        )
        nt_s = np.asarray(
            skip._be.apply_hybrid_batch(gates, chips), dtype=bool
        )
        assert np.array_equal(nt_f, nt_s)

        full._refresh()
        if nt_s.any() and not nt_s.all():
            skip._refresh(encode_mask=~nt_s)
            saw_partial = True
        elif nt_s.all():
            skip._refresh(encode_mask=np.zeros(n, dtype=bool))
            saw_partial = True
        else:
            skip._refresh()

        # Engine caches: all rows, full match.
        for name in _CACHED_NON_OBS:
            assert np.array_equal(getattr(full, name), getattr(skip, name)), (
                f"step {step}: cache {name} mismatch"
            )

        # Obs: live rows bit-exact; terminal rows zero on both.
        live = ~full._dones
        if live.any():
            assert np.array_equal(full._obs[live], skip._obs[live]), (
                f"step {step}: live obs mismatch"
            )
        term = full._dones
        if term.any():
            assert np.all(full._obs[term] == 0.0)
            assert np.all(skip._obs[term] == 0.0)

        # Reset terminals so the next step has a mix of states.
        if nt_f.any():
            new_seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(
                np.uint64
            )
            new_buttons = rng.integers(
                0, cfg.num_seats, size=n, dtype=np.int64
            ).astype(np.uint8)
            full._be.reset_terminal_batch(new_seeds, new_buttons, nt_f)
            skip._be.reset_terminal_batch(new_seeds, new_buttons, nt_s)
            full._refresh_subset(nt_f)
            skip._refresh_subset(nt_s)

    assert saw_partial, "expected at least one partial-encode step in the drive"


def test_encode_mask_with_rust_encoder(monkeypatch):
    """#1 must skip encode for newly-terminal rows even when PLO5_RUST_ENCODER=1."""
    monkeypatch.setenv("PLO5_RUST_ENCODER", "1")
    cfg = GameConfig(num_seats=6)
    n = 16
    full = BatchedBombPotEnv(n, cfg)
    skip = BatchedBombPotEnv(n, cfg)
    assert full._use_rust_encoder is True
    rng = np.random.default_rng(7)
    seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, cfg.num_seats, size=n, dtype=np.int64).astype(np.uint8)
    full.reset_batch(seeds, buttons)
    skip.reset_batch(seeds, buttons)
    saw = False
    for step in range(40):
        gates, chips = _sample_legal(full, rng)
        nt = np.asarray(full._be.apply_hybrid_batch(gates, chips), dtype=bool)
        np.asarray(skip._be.apply_hybrid_batch(gates, chips), dtype=bool)
        full._refresh()
        if nt.any() and not nt.all():
            skip._refresh(encode_mask=~nt)
            saw = True
        elif nt.all():
            skip._refresh(encode_mask=np.zeros(n, dtype=bool))
            saw = True
        else:
            skip._refresh()
        live = ~full._dones
        if live.any():
            assert np.array_equal(full._obs[live], skip._obs[live]), step
        if full._dones.any():
            assert np.all(skip._obs[full._dones] == 0.0)
        for name in _CACHED_NON_OBS:
            assert np.array_equal(getattr(full, name), getattr(skip, name)), name
        if nt.any():
            ns = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
            nb = rng.integers(0, cfg.num_seats, size=n, dtype=np.int64).astype(np.uint8)
            full._be.reset_terminal_batch(ns, nb, nt)
            skip._be.reset_terminal_batch(ns, nb, nt)
            full._refresh_subset(nt)
            skip._refresh_subset(nt)
    assert saw


def test_all_hole_cards_subset_matches_full():
    cfg = GameConfig(num_seats=6)
    n = 24
    env = BatchedBombPotEnv(n, cfg)
    rng = np.random.default_rng(99)
    seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, cfg.num_seats, size=n, dtype=np.int64).astype(np.uint8)
    env.reset_batch(seeds, buttons)

    full = np.asarray(env._be.all_hole_cards_batch(), dtype=np.uint8)
    idx = np.array([0, 3, 7, 11, 20], dtype=np.int64)
    sub = np.asarray(env._be.all_hole_cards_subset_batch(idx), dtype=np.uint8)
    assert sub.shape == (idx.size, cfg.num_seats, cfg.hole_count)
    assert np.array_equal(sub, full[idx])

    # After partial reset, subset of re-dealt envs matches full refetch.
    mask = np.zeros(n, dtype=bool)
    mask[[1, 5, 9]] = True
    new_seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
    new_buttons = rng.integers(0, cfg.num_seats, size=n, dtype=np.int64).astype(
        np.uint8
    )
    env._be.reset_terminal_batch(new_seeds, new_buttons, mask)
    full2 = np.asarray(env._be.all_hole_cards_batch(), dtype=np.uint8)
    term = np.nonzero(mask)[0].astype(np.int64)
    sub2 = np.asarray(env._be.all_hole_cards_subset_batch(term), dtype=np.uint8)
    assert np.array_equal(sub2, full2[term])
    # Unchanged envs keep prior holes.
    keep = ~mask
    assert np.array_equal(full2[keep], full[keep])
