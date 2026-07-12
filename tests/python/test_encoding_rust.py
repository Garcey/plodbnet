"""Bit-exact parity for the Rust observation encoder.

`PyBatchedEngine.observation_encoded_batch()` / `..._subset_batch(idx)` port
the numpy `encode_observation_batch` assembly into Rust. They must be
byte-for-byte identical to BOTH the numpy batch encoder AND the scalar
`encode_observation` reference (the chain scalar ↔ numpy ↔ rust, all
`np.array_equal`). These tests drive matched serial `GameState`s and a
`BatchedEngine` in lockstep (mirroring the proven harness in
`test_encoding_batch.py`) and assert that three-way equality across the full
param grid plus edge cases random rollouts under-cover (heterogeneous
dead-chips, clip boundaries, all-in, empty/partial board, terminal, 2 vs 6
seats, subset).

Skipped automatically when the engine wasn't rebuilt with the new methods, so
the suite stays green on a stale binary; lifts as soon as `maturin develop`
exposes them.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp._engine import BatchedEngine, GameState  # type: ignore[attr-defined]
from plo5bp.config import (
    VARIANT_PLO4,
    VARIANT_PLO5,
    VARIANT_PLO6,
    GameConfig,
)
from plo5bp.encoding import encode_observation, encode_observation_batch

_HAS_RUST_ENCODER = hasattr(BatchedEngine, "observation_encoded_batch")
# The Rust encoder implements the pre-obs-v2 991-dim layout; the
# 2026-07-06 obs-v2 tail (OBS_DIM 1020, V5_DESIGN.md §3.2) is not ported
# to it and env_batched force-disables PLO5_RUST_ENCODER. The 3-way
# parity suite stays skipped until the Rust port catches up — do NOT
# "fix" it by comparing only the first 991 dims (a silently truncated
# obs is exactly the bug the gate exists to prevent).
# Flipped False 2026-07-12: the v7 batch-2 tail (OBS_DIM 1171) is not ported
# to the Rust encoder, which still emits 1020; env_batched width-gates it off.
# The 3-way parity suite stays skipped until the tail blocks are ported and
# _RUST_ENCODER_OBS_DIM is bumped. Do NOT compare only the first 1020 dims.
_RUST_ENCODER_CURRENT = False
pytestmark = pytest.mark.skipif(
    not (_HAS_RUST_ENCODER and _RUST_ENCODER_CURRENT),
    reason=(
        "Rust obs encoder predates the obs-v2 layout (991 vs 1020) and is "
        "force-disabled in env_batched; port the tail blocks, then flip "
        "_RUST_ENCODER_CURRENT"
    ),
)


# The force-disable itself is pinned in test_encoding_batch.py
# (test_env_batched_refuses_stale_rust_encoder) — this module's skip
# would otherwise swallow it.

# Aux fields the rollout reads off the bundle (besides obs).
_AUX = ("actor", "legal_mask", "min_raise", "max_raise", "total_commit",
        "bet_to_call", "street_commit", "street")


def _augmented_scalar_obs(gs: GameState) -> dict:
    raw = dict(gs.observation_dict())
    actor = raw["actor"]
    if actor is None:
        return raw
    raw["hero_category_a"] = int(gs.hero_category(actor, 0))
    raw["hero_category_b"] = int(gs.hero_category(actor, 1))
    return raw


def _make_serial(num_seats, starting_stack, ante, bb, starting_stacks, seed, button,
                 variant=VARIANT_PLO5):
    if starting_stacks is None:
        gs = GameState(num_seats, starting_stack, ante, bb, variant=variant)
    else:
        gs = GameState(
            num_seats, starting_stack, ante, bb,
            starting_stacks=np.asarray(starting_stacks, dtype=np.uint64),
            variant=variant,
        )
    gs.reset(int(seed), int(button))
    return gs


def _make_batched(batch_size, num_seats, starting_stack, ante, bb, starting_stacks,
                  variant=VARIANT_PLO5):
    if starting_stacks is None:
        return BatchedEngine(
            batch_size, num_seats=num_seats, starting_stack=starting_stack,
            ante=ante, bb=bb, variant=variant,
        )
    return BatchedEngine(
        batch_size, num_seats=num_seats, starting_stack=starting_stack,
        ante=ante, bb=bb,
        starting_stacks=np.asarray(starting_stacks, dtype=np.uint64),
        variant=variant,
    )


def _assert_rust_matches(be, serial, config, step: int) -> None:
    """Rust bundle == scalar encoder (per live env) == numpy batch encoder;
    aux fields == numpy bundle."""
    rust = be.observation_encoded_batch()
    np_bundle = be.observation_and_features_batch()
    cat_a = np.asarray(np_bundle["hero_cat_a"])
    cat_b = np.asarray(np_bundle["hero_cat_b"])
    numpy_obs = encode_observation_batch(np_bundle, cat_a, cat_b, config)
    rust_obs = np.asarray(rust["obs"])

    assert np.array_equal(rust_obs, numpy_obs), (
        f"step {step}: rust vs numpy-batch mismatch at "
        f"{np.argwhere(rust_obs != numpy_obs)[:3].tolist()}"
    )
    for i in range(len(serial)):
        if serial[i].is_terminal():
            assert not rust_obs[i].any(), f"step {step} env {i}: terminal row not zero"
            continue
        scalar_vec = encode_observation(_augmented_scalar_obs(serial[i]), config)
        assert np.array_equal(rust_obs[i], scalar_vec), (
            f"step {step} env {i}: rust vs scalar mismatch at "
            f"dim {int(np.argmax(rust_obs[i] != scalar_vec))}"
        )
    for key in _AUX:
        assert np.array_equal(
            np.asarray(rust[key]), np.asarray(np_bundle[key])
        ), f"step {step}: aux field {key} mismatch"


def _config(num_seats, starting_stack, ante, bb, starting_stacks, variant=VARIANT_PLO5):
    kw = {}
    if starting_stacks is not None:
        kw["starting_stacks"] = tuple(int(x) for x in starting_stacks)
    return GameConfig(
        num_seats=num_seats, starting_stack=starting_stack, ante=ante, bb=bb,
        variant=variant, **kw
    )


def _drive(num_seats, starting_stack, batch_size, base_seed,
           starting_stacks=None, max_steps=200, variant=VARIANT_PLO5) -> None:
    ante, bb = 30000, 10000
    config = _config(num_seats, starting_stack, ante, bb, starting_stacks, variant)
    rng = np.random.default_rng(base_seed)
    env_rngs = [np.random.default_rng(base_seed * 1000 + i) for i in range(batch_size)]
    seeds = rng.integers(0, 2**63 - 1, size=batch_size, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, num_seats, size=batch_size, dtype=np.int64).astype(np.uint8)

    serial = [
        _make_serial(num_seats, starting_stack, ante, bb, starting_stacks,
                     seeds[i], buttons[i], variant=variant)
        for i in range(batch_size)
    ]
    be = _make_batched(batch_size, num_seats, starting_stack, ante, bb, starting_stacks,
                       variant=variant)
    be.reset_batch(seeds, buttons)

    _assert_rust_matches(be, serial, config, step=-1)  # initial state

    step = 0
    while not be.is_terminal_batch().all() and step < max_steps:
        actions = np.zeros(batch_size, dtype=np.uint8)
        for i in range(batch_size):
            if not serial[i].is_terminal():
                mask = np.asarray(serial[i].legal_action_mask(), dtype=bool)
                actions[i] = int(env_rngs[i].choice(np.flatnonzero(mask)))
        be.apply_action_batch(actions)
        for i in range(batch_size):
            if not serial[i].is_terminal():
                serial[i].apply_action(int(actions[i]))
        _assert_rust_matches(be, serial, config, step=step)
        step += 1


@pytest.mark.parametrize("num_seats", [2, 3, 4, 6])
@pytest.mark.parametrize("starting_stack", [200000, 1000000])
@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("base_seed", [0, 1])
def test_rust_encoder_parity_grid(num_seats, starting_stack, batch_size, base_seed) -> None:
    _drive(num_seats, starting_stack, batch_size, base_seed)


def test_rust_encoder_heterogeneous_dead_chips() -> None:
    """Mixed shallow/deep stacks force `starting > eff_cap` on the deep seat, so
    the effective-stack dead-chips clamp fires — the riskiest f64 chain and the
    one uniform-stack rollouts never trigger."""
    for stacks in [(200000, 2000000), (200000, 1500000, 400000),
                   (300000, 3000000, 250000, 1200000)]:
        _drive(len(stacks), stacks[0], batch_size=8, base_seed=7, starting_stacks=stacks)


def test_rust_encoder_deep_clip_boundaries() -> None:
    """Very deep uniform stacks push SPR past its 4.0 clip and create large
    bets — exercises the SPR/pot-odds/bet-pct clip+floor paths."""
    _drive(3, 5000000, batch_size=16, base_seed=3)


def test_rust_encoder_two_and_six_seats() -> None:
    """2 and 6 seats exercise the rotation/padding edges; the early steps of
    each hand cover partial/empty-board rows."""
    for ns in (2, 6):
        _drive(ns, 400000, batch_size=8, base_seed=5)


def test_rust_encoder_subset_matches_full() -> None:
    """`observation_encoded_subset_batch(idx)` rows must equal the full
    `observation_encoded_batch()` rows at those indices — the rollout's
    post-reset path relies on this."""
    ante, bb, ns = 30000, 10000, 3
    n = 12
    rng = np.random.default_rng(99)
    seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, ns, size=n, dtype=np.int64).astype(np.uint8)
    be = _make_batched(n, ns, 400000, ante, bb, None)
    be.reset_batch(seeds, buttons)
    for _ in range(6):
        acts = np.zeros(n, dtype=np.uint8)
        mb = be.legal_mask_batch()
        for i in range(n):
            if not be.is_terminal_batch()[i]:
                acts[i] = int(np.flatnonzero(np.asarray(mb[i], dtype=bool))[0])
        be.apply_action_batch(acts)

    full = np.asarray(be.observation_encoded_batch()["obs"])
    for idx in ([0], [1, 4, 7], list(range(0, n, 2)), list(range(n))):
        idx_arr = np.asarray(idx, dtype=np.int64)
        sub = np.asarray(be.observation_encoded_subset_batch(idx_arr)["obs"])
        assert np.array_equal(sub, full[idx_arr]), f"subset {idx} mismatch vs full"
    empty = np.asarray(
        be.observation_encoded_subset_batch(np.zeros(0, dtype=np.int64))["obs"]
    )
    assert empty.shape[0] == 0


# --- PLO4 / PLO6 variant parity (the batch encoder must key the hole
# multi-hot off the variant's hole width, not a hardcoded 5). Regression
# guard for the merged hole/board loop that panicked on PLO4 (index OOB)
# and silently dropped the 6th hole card on PLO6. ---


@pytest.mark.parametrize(
    "variant, hole_count",
    [(VARIANT_PLO4, 4), (VARIANT_PLO6, 6)],
)
@pytest.mark.parametrize("num_seats", [2, 6])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_rust_encoder_parity_plo4_plo6(variant, hole_count, num_seats, batch_size) -> None:
    """Rust==numpy==scalar three-way parity over a full rollout for PLO4/PLO6.
    Exercises (via `_drive` -> `_assert_rust_matches`) that the Rust encoder no
    longer panics on PLO4 (hole width 4) and is byte-identical to the numpy
    encoder on PLO6 (hole width 6)."""
    _drive(num_seats, 200000, batch_size, base_seed=13, variant=variant)


@pytest.mark.parametrize(
    "variant, hole_count",
    [(VARIANT_PLO4, 4), (VARIANT_PLO5, 5), (VARIANT_PLO6, 6)],
)
def test_rust_encoder_hole_multihot_popcount(variant, hole_count) -> None:
    """The Rust-encoded hole multi-hot (obs[:52]) must have popcount == the
    variant's hole width at the initial (flop) state — PLO4 4, PLO5 5, PLO6 6.
    PLO4 also confirms the encoder returns instead of panicking with an
    index-out-of-bounds on the 5th (nonexistent) hole slot."""
    ante, bb, num_seats = 30000, 10000, 6
    n = 8
    be = _make_batched(n, num_seats, 200000, ante, bb, None, variant=variant)
    seeds = np.arange(100, 100 + n, dtype=np.uint64)
    buttons = np.zeros(n, dtype=np.uint8)
    be.reset_batch(seeds, buttons)
    obs = np.asarray(be.observation_encoded_batch()["obs"])  # must not panic
    for i in range(n):
        pc = int(obs[i, :52].sum())
        assert pc == hole_count, (
            f"{variant} env {i}: hole multi-hot popcount {pc} != {hole_count}"
        )
