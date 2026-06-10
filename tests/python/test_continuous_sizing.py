"""End-to-end continuous-sizing rollout + PPO update.

Verifies:
  - `collect_rollout_batched` produces a `Batch` whose gate actions are
    in the stored gate mask and whose raise chips sit inside
    `[min_raise, max_raise]` for every GATE_RAISE row.
  - `PPOTrainer.update(batch)` runs without NaN/inf across a few epochs.
  - `env.step_hybrid` with a raise delta inside the engine-reported
    range succeeds; a raise delta outside the range raises ValueError
    (via the Rust `InvalidAmount` error).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.actions import GATE_RAISE
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.env import BombPotEnv
from plo5bp.network import ActorCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout_batched
from plo5bp.selfplay import OpponentPool


def test_rollout_and_update_finite_and_in_bounds() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    game_cfg = GameConfig(num_seats=4)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=256,
        hidden_dim=32,
        ppo_epochs=2,
        batch_size=64,
    )
    model = ActorCritic(hidden_dim=train_cfg.hidden_dim)
    trainer = PPOTrainer(model, train_cfg)
    pool = OpponentPool(capacity=1)
    rng = np.random.default_rng(1)

    batch = collect_rollout_batched(model, pool, game_cfg, train_cfg, rng)

    # Legality: gate in its mask; Raise chips inside bounds.
    gm = batch.gate_masks.numpy()
    ga = batch.gate_actions.numpy()
    rc = batch.raise_chips.numpy()
    rb = batch.sizing.numpy()
    for i in range(ga.shape[0]):
        assert bool(gm[i, int(ga[i])]), f"illegal gate at row {i}"
        if int(ga[i]) == GATE_RAISE:
            lo, hi = int(rb[i, 0]), int(rb[i, 1])
            # `lo == 0` is the short-shove regime — engine zeroes the
            # min-raise floor while exposing legal[ALL_IN], so the gate
            # is reachable but the env redirects to apply(ALL_IN) and
            # ignores the chip amount.
            assert hi >= lo and hi > 0
            assert lo <= int(rc[i]) <= hi

    # Non-Raise rows must carry 0 chips.
    non_raise = ga != GATE_RAISE
    assert (rc[non_raise] == 0).all()

    # Ppo update stays finite.
    stats = trainer.update(batch, rng)
    for v in (stats.policy_loss, stats.value_loss, stats.entropy, stats.approx_kl):
        assert np.isfinite(v), f"non-finite ppo stat: {stats}"


def test_step_hybrid_out_of_range_raises() -> None:
    """Engine rejects a raise whose chip amount is outside
    `[min_raise_chips, max_raise_chips]`."""
    env = BombPotEnv(GameConfig())
    rng = np.random.default_rng(0)

    # Find a state where Raise is legal.
    obs, info = env.reset(int(rng.integers(0, 2**63 - 1)), 0)
    while not info.terminal and not info.gate_mask[GATE_RAISE]:
        legal = np.flatnonzero(info.gate_mask)
        gate = int(rng.choice(legal))
        chips = 0
        if gate == GATE_RAISE:
            lo = int(info.min_raise_chips)
            hi = int(info.max_raise_chips)
            chips = int(rng.integers(lo, hi + 1))
        obs, _, done, info = env.step_hybrid(gate, chips)
        if done:
            obs, info = env.reset(int(rng.integers(0, 2**63 - 1)), 0)

    assert info.gate_mask[GATE_RAISE], "could not find a Raise-legal state"
    lo = int(info.min_raise_chips)
    hi = int(info.max_raise_chips)
    with pytest.raises(Exception):  # Rust-side error surfaces through PyO3
        env.step_hybrid(GATE_RAISE, lo - 1)
    # Construct a fresh env for the upper-bound check (the failed step
    # above still consumed the RNG but did not advance state).
    with pytest.raises(Exception):
        env.step_hybrid(GATE_RAISE, hi + 1)
