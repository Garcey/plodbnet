"""Parity of BatchedBombPotEnv vs BombPotEnv.

For matched (seeds, buttons, action-sequence) the two envs must emit
identical (obs, legal_mask, reward, done) at every step.

Rewards are asserted bit-exact via np.array_equal on float32. Terminal
rows: obs all-zero, legal_mask all-false, rewards equal to scalar
`payouts()` (or `payouts_ev()` under EV mode).
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.env_batched import BatchedBombPotEnv


def _drive_env_parity(
    num_seats: int,
    starting_stack: int,
    batch_size: int,
    base_seed: int,
    ev_runout_samples: int = 0,
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

    serial: list[BombPotEnv] = []
    serial_obs: list[np.ndarray] = []
    serial_infos = []
    for i in range(batch_size):
        env = BombPotEnv(config, ev_runout_samples=ev_runout_samples)
        o, info = env.reset(int(seeds[i]), int(buttons[i]))
        serial.append(env)
        serial_obs.append(o)
        serial_infos.append(info)

    batched = BatchedBombPotEnv(
        batch_size, config, ev_runout_samples=ev_runout_samples
    )
    bstep = batched.reset_batch(seeds, buttons)

    # Initial observations must match.
    for i in range(batch_size):
        assert np.array_equal(bstep.obs[i], serial_obs[i]), (
            f"env {i} initial: obs mismatch"
        )
        assert np.array_equal(bstep.legal_mask[i], serial_infos[i].legal_mask)
        assert (int(bstep.actors[i]) == -1) == (serial_infos[i].actor is None)

    alive_serial = [not env.is_terminal() for env in serial]
    step = 0
    while any(alive_serial):
        # Choose actions for each live env.
        actions = np.zeros(batch_size, dtype=np.uint8)
        for i in range(batch_size):
            if alive_serial[i]:
                mask = np.asarray(
                    serial[i].legal_action_mask(), dtype=bool
                )
                legal = np.flatnonzero(mask)
                actions[i] = int(env_rngs[i].choice(legal))

        bstep = batched.step_batch(actions)

        # Advance serial envs that were live before step.
        for i in range(batch_size):
            if not alive_serial[i]:
                continue
            obs_s, rew_s, done_s, info_s = serial[i].step(int(actions[i]))
            # newly_terminal[i] should match done_s for this step.
            assert bool(bstep.newly_terminal[i]) == bool(done_s), (
                f"env {i} step {step}: newly_terminal mismatch "
                f"batched={bool(bstep.newly_terminal[i])} scalar={bool(done_s)}"
            )
            # Rewards vector parity (per-seat).
            assert np.array_equal(bstep.rewards[i], rew_s), (
                f"env {i} step {step}: rewards mismatch "
                f"batched={bstep.rewards[i]} scalar={rew_s}"
            )
            # Observation + legal mask parity (zeros for terminal, encoded
            # vector for live).
            assert np.array_equal(bstep.obs[i], obs_s), (
                f"env {i} step {step}: obs mismatch"
            )
            assert np.array_equal(bstep.legal_mask[i], info_s.legal_mask), (
                f"env {i} step {step}: legal_mask mismatch"
            )
            # Actor parity.
            expected_actor = -1 if info_s.actor is None else int(info_s.actor)
            assert int(bstep.actors[i]) == expected_actor, (
                f"env {i} step {step}: actor mismatch "
                f"batched={int(bstep.actors[i])} scalar={expected_actor}"
            )
            alive_serial[i] = not done_s

        # For envs that were already terminal, batched should still report
        # dones=True with zero rewards.
        for i in range(batch_size):
            if bstep.dones[i] and not bstep.newly_terminal[i]:
                assert not bstep.rewards[i].any(), (
                    f"env {i} step {step}: residual reward on stale-terminal env"
                )

        step += 1
        if step > 400:
            raise AssertionError("hand did not terminate within 400 steps")


@pytest.mark.parametrize("num_seats", [2, 4, 6])
@pytest.mark.parametrize("starting_stack", [200000, 1000000])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("base_seed", [0, 1])
def test_env_batched_parity(
    num_seats: int, starting_stack: int, batch_size: int, base_seed: int
) -> None:
    _drive_env_parity(num_seats, starting_stack, batch_size, base_seed)


def test_env_batched_parity_with_ev_runout() -> None:
    _drive_env_parity(
        num_seats=6,
        starting_stack=200000,
        batch_size=8,
        base_seed=3,
        ev_runout_samples=32,
    )


def test_reset_terminal_batch_refreshes_masked_envs_only() -> None:
    """After stepping env 0 only, reset_terminal_batch with mask=[1,0,0,0]
    should update env 0's button/obs but leave the others untouched."""
    config = GameConfig(num_seats=6, starting_stack=200000, ante=30000, bb=10000)
    n = 4
    be = BatchedBombPotEnv(n, config)
    seeds = np.arange(n, dtype=np.uint64) + 50
    buttons = np.zeros(n, dtype=np.uint8)
    be.reset_batch(seeds, buttons)

    actors_before = be.current_actors().copy()
    obs_before = be.observation().copy()

    new_seeds = np.array([999, 0, 0, 0], dtype=np.uint64)
    new_buttons = np.array([3, 0, 0, 0], dtype=np.uint8)
    mask = np.array([True, False, False, False], dtype=bool)
    bstep = be.reset_terminal_batch(new_seeds, new_buttons, mask)

    # Env 0: button=3 → actor=4 (utg on 6-max). Must differ from env 0 before.
    assert bstep.actors[0] == 4
    # Envs 1-3: unchanged.
    for i in range(1, n):
        assert bstep.actors[i] == actors_before[i]
        assert np.array_equal(bstep.obs[i], obs_before[i])
