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
from plo5bp.config import GameConfig
from plo5bp.encoding import encode_observation, encode_observation_batch

_HAS_RUST_ENCODER = hasattr(BatchedEngine, "observation_encoded_batch")
pytestmark = pytest.mark.skipif(
    not _HAS_RUST_ENCODER,
    reason="Rust encoder not built; run `maturin develop --release`",
)

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


def _make_serial(num_seats, starting_stack, ante, bb, starting_stacks, seed, button):
    if starting_stacks is None:
        gs = GameState(num_seats, starting_stack, ante, bb)
    else:
        gs = GameState(
            num_seats, starting_stack, ante, bb,
            starting_stacks=np.asarray(starting_stacks, dtype=np.uint64),
        )
    gs.reset(int(seed), int(button))
    return gs


def _make_batched(batch_size, num_seats, starting_stack, ante, bb, starting_stacks):
    if starting_stacks is None:
        return BatchedEngine(
            batch_size, num_seats=num_seats, starting_stack=starting_stack,
            ante=ante, bb=bb,
        )
    return BatchedEngine(
        batch_size, num_seats=num_seats, starting_stack=starting_stack,
        ante=ante, bb=bb,
        starting_stacks=np.asarray(starting_stacks, dtype=np.uint64),
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


def _config(num_seats, starting_stack, ante, bb, starting_stacks):
    kw = {}
    if starting_stacks is not None:
        kw["starting_stacks"] = tuple(int(x) for x in starting_stacks)
    return GameConfig(
        num_seats=num_seats, starting_stack=starting_stack, ante=ante, bb=bb, **kw
    )


def _drive(num_seats, starting_stack, batch_size, base_seed,
           starting_stacks=None, max_steps=200) -> None:
    ante, bb = 30000, 10000
    config = _config(num_seats, starting_stack, ante, bb, starting_stacks)
    rng = np.random.default_rng(base_seed)
    env_rngs = [np.random.default_rng(base_seed * 1000 + i) for i in range(batch_size)]
    seeds = rng.integers(0, 2**63 - 1, size=batch_size, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, num_seats, size=batch_size, dtype=np.int64).astype(np.uint8)

    serial = [
        _make_serial(num_seats, starting_stack, ante, bb, starting_stacks,
                     seeds[i], buttons[i])
        for i in range(batch_size)
    ]
    be = _make_batched(batch_size, num_seats, starting_stack, ante, bb, starting_stacks)
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
