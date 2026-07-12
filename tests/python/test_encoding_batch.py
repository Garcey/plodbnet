"""Bit-exact parity of encode_observation_batch vs encode_observation.

At every step of a scripted random-legal rollout, the vectorized encoder
fed with (batched_obs_arrays, hero_category_batch) must produce rows
byte-identical to the scalar `encode_observation` called on each env's
augmented `observation_dict()`.

The augmentation mirrors `BombPotEnv._pack_obs` exactly:
  hero_category_a/b = gs.hero_category(actor, 0|1)

Terminal envs (actor == -1) must produce zero rows in both paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp._engine import BatchedEngine, GameState  # type: ignore[attr-defined]
from plo5bp.config import GameConfig
from plo5bp.encoding import (
    OBS_DIM,
    encode_observation,
    encode_observation_batch,
)


def _random_legal_action(state: GameState, rng: np.random.Generator) -> int:
    mask = np.asarray(state.legal_action_mask(), dtype=bool)
    legal = np.flatnonzero(mask)
    assert legal.size > 0
    return int(rng.choice(legal))


def _augmented_scalar_obs(gs: GameState) -> dict:
    raw = dict(gs.observation_dict())
    actor = raw["actor"]
    if actor is None:
        return raw
    raw["hero_category_a"] = int(gs.hero_category(actor, 0))
    raw["hero_category_b"] = int(gs.hero_category(actor, 1))
    return raw


def _drive_encoder_parity(
    num_seats: int,
    starting_stack: int,
    batch_size: int,
    base_seed: int,
) -> None:
    ante = 30000
    bb = 10000
    config = GameConfig(
        num_seats=num_seats, starting_stack=starting_stack, ante=ante, bb=bb
    )
    rng = np.random.default_rng(base_seed)
    env_rngs = [
        np.random.default_rng(base_seed * 1_000 + i) for i in range(batch_size)
    ]
    seeds = rng.integers(0, 2**63 - 1, size=batch_size, dtype=np.int64).astype(
        np.uint64
    )
    buttons = rng.integers(0, num_seats, size=batch_size, dtype=np.int64).astype(
        np.uint8
    )

    # Parallel engines.
    serial: list[GameState] = []
    for i in range(batch_size):
        gs = GameState(num_seats, starting_stack, ante, bb)
        gs.reset(int(seeds[i]), int(buttons[i]))
        serial.append(gs)

    be = BatchedEngine(
        batch_size,
        num_seats=num_seats,
        starting_stack=starting_stack,
        ante=ante,
        bb=bb,
    )
    be.reset_batch(seeds, buttons)

    step = 0
    while not be.is_terminal_batch().all():
        actors = be.actor_batch()
        # Build per-env (seat, board) inputs for the batched path.
        # Use seat=0/board=0 for terminal rows — the encoder masks them out.
        live = actors != -1
        seats = np.where(live, actors, 0).astype(np.uint8)
        boards_a = np.zeros(batch_size, dtype=np.uint8)
        boards_b = np.ones(batch_size, dtype=np.uint8)

        cat_a = be.hero_category_batch(seats, boards_a)
        cat_b = be.hero_category_batch(seats, boards_b)

        obs_arrays = be.observation_arrays()
        vec_batched = encode_observation_batch(
            obs_arrays, cat_a, cat_b, config
        )

        # Scalar reference for each env.
        for i in range(batch_size):
            if serial[i].is_terminal():
                expected = np.zeros(OBS_DIM, dtype=np.float32)
            else:
                raw = _augmented_scalar_obs(serial[i])
                expected = encode_observation(raw, config)
            assert np.array_equal(vec_batched[i], expected), (
                f"env {i} step {step}: encoder mismatch\n"
                f"  first diff at dim {int(np.argmax(vec_batched[i] != expected))}"
            )

        # Advance both engines in lockstep.
        actions = np.zeros(batch_size, dtype=np.uint8)
        for i in range(batch_size):
            if not serial[i].is_terminal():
                actions[i] = _random_legal_action(serial[i], env_rngs[i])
        be.apply_action_batch(actions)
        for i in range(batch_size):
            if not serial[i].is_terminal():
                serial[i].apply_action(int(actions[i]))

        step += 1
        if step > 400:
            raise AssertionError("hand did not terminate within 400 steps")


@pytest.mark.parametrize("num_seats", [2, 3, 4, 6])
@pytest.mark.parametrize("starting_stack", [200000, 1000000])
@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("base_seed", [0, 1])
def test_encoder_batch_parity(
    num_seats: int, starting_stack: int, batch_size: int, base_seed: int
) -> None:
    _drive_encoder_parity(num_seats, starting_stack, batch_size, base_seed)


def test_encoder_batch_terminal_row_is_zero() -> None:
    """Rows where actor == -1 must be all zeros in the batched output."""
    config = GameConfig(num_seats=6, starting_stack=200000, ante=30000, bb=10000)
    n = 4
    be = BatchedEngine(
        n, num_seats=6, starting_stack=200000, ante=30000, bb=10000
    )
    be.reset_batch(
        np.arange(n, dtype=np.uint64) + 1, np.zeros(n, dtype=np.uint8)
    )
    # Drive env 0 to terminal by folding every actor until hand ends.
    serial0 = GameState(6, 2000, 300, 100)
    serial0.reset(1, 0)
    # Crash-terminate env 0 via fold repeatedly — no Rust-side-effect on other envs
    # because the test uses actor-specific masks.
    while not be.is_terminal_batch()[0]:
        acts = np.zeros(n, dtype=np.uint8)  # action 0 = Fold
        # For envs that can't fold (no bet to face), pick CheckCall.
        mb = be.legal_mask_batch()
        for i in range(n):
            if be.is_terminal_batch()[i]:
                continue
            # Prefer fold; fall back to check/call if fold illegal.
            if mb[i][0]:
                acts[i] = 0
            elif mb[i][1]:
                acts[i] = 1
            else:
                acts[i] = int(np.flatnonzero(mb[i])[0])
        be.apply_action_batch(acts)

    # All envs should eventually terminate because every actor folds when legal.
    assert be.is_terminal_batch().all()
    actors = be.actor_batch()
    assert (actors == -1).all()

    obs_arrays = be.observation_arrays()
    # Dummy inputs — must not affect output.
    cat = np.zeros(n, dtype=np.uint8)
    vec = encode_observation_batch(obs_arrays, cat, cat, config)
    assert vec.shape == (n, OBS_DIM)
    assert not vec.any()


def test_env_batched_rust_encoder_width_gated(monkeypatch):
    """Rust encoder opt-in via PLO5_RUST_ENCODER; width-gated to
    OBS_DIM == _RUST_ENCODER_OBS_DIM (1171 after v7 tail port)."""
    import plo5bp.env_batched as eb
    from plo5bp.config import GameConfig

    # Default (flag unset): numpy encoder.
    assert eb.BatchedBombPotEnv(2, GameConfig(num_seats=2))._use_rust_encoder is False
    # Opt in: enabled when widths match (1171).
    monkeypatch.setenv("PLO5_RUST_ENCODER", "1")
    assert eb.OBS_DIM == eb._RUST_ENCODER_OBS_DIM
    assert eb.BatchedBombPotEnv(2, GameConfig(num_seats=2))._use_rust_encoder is True
