"""Smoke + invariant tests for `collect_rollout_batched`.

Covers:
  - shape / dtype / legal-action invariant
  - advantage normalization
  - chip conservation (aggregate returns are finite, shape-consistent)
  - snapshot-group batching path: with a populated pool and pool_mix_prob=1,
    opponent forwards must still produce legal actions on every row.

Strict (bit-exact) parity with the serial `collect_rollout` isn't asserted
because snapshot-group batching changes the torch RNG-consumption order;
matching distributions in expectation is validated by the training smoke
test.
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.network import ActorCritic
from plo5bp.rollout import collect_rollout_batched
from plo5bp.selfplay import OpponentPool


def test_batched_rollout_smoke_self_play_only() -> None:
    torch.manual_seed(0)
    game_cfg = GameConfig()
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=512,
        hidden_dim=32,
    )
    model = ActorCritic(hidden_dim=train_cfg.hidden_dim)
    model.eval()
    pool = OpponentPool(capacity=4)  # empty -> pure self-play path
    rng = np.random.default_rng(0)

    batch = collect_rollout_batched(model, pool, game_cfg, train_cfg, rng)

    assert batch.obs.shape[0] >= train_cfg.rollout_length
    assert batch.obs.shape[1] == model.torso[0].in_features

    gm = batch.gate_masks.numpy()
    ga = batch.gate_actions.numpy()
    rc = batch.raise_chips.numpy()
    rb = batch.raise_bounds.numpy()
    for i in range(ga.shape[0]):
        assert bool(gm[i, int(ga[i])]), (
            f"illegal gate {int(ga[i])} at row {i}"
        )
        if int(ga[i]) == 2:  # GATE_RAISE
            lo, hi = int(rb[i, 0]), int(rb[i, 1])
            assert lo <= int(rc[i]) <= hi, (
                f"chips {int(rc[i])} out of [{lo}, {hi}] at row {i}"
            )

    adv = batch.advantages.numpy()
    assert abs(float(adv.mean())) < 1e-5
    assert abs(float(adv.std()) - 1.0) < 1e-3

    assert torch.isfinite(batch.returns).all()
    assert torch.isfinite(batch.values).all()
    assert batch.returns.shape == batch.values.shape


def test_batched_rollout_exercises_snapshot_group_path() -> None:
    """With pool_mix_prob=1.0 and a populated pool, every env is assigned
    a snapshot; snapshot-grouped opponent forwards must emit legal actions.
    """
    torch.manual_seed(1)
    game_cfg = GameConfig(num_seats=6)
    train_cfg = TrainingConfig(
        num_envs=16,
        rollout_length=256,
        hidden_dim=32,
        pool_mix_prob=1.0,
        pool_opp_seats=2,
    )
    learner = ActorCritic(hidden_dim=train_cfg.hidden_dim)
    learner.eval()

    pool = OpponentPool(capacity=4)
    # Populate pool with two snapshots so snapshot-group batching has
    # distinct buckets to exercise.
    for _ in range(2):
        snap_model = ActorCritic(hidden_dim=train_cfg.hidden_dim)
        pool.snapshot(snap_model)

    rng = np.random.default_rng(7)
    batch = collect_rollout_batched(learner, pool, game_cfg, train_cfg, rng)

    # Basic sanity + legality.
    assert batch.obs.shape[0] >= train_cfg.rollout_length
    gm = batch.gate_masks.numpy()
    ga = batch.gate_actions.numpy()
    rc = batch.raise_chips.numpy()
    rb = batch.raise_bounds.numpy()
    for i in range(ga.shape[0]):
        assert bool(gm[i, int(ga[i])])
        if int(ga[i]) == 2:
            lo, hi = int(rb[i, 0]), int(rb[i, 1])
            assert lo <= int(rc[i]) <= hi

    assert torch.isfinite(batch.returns).all()
    assert torch.isfinite(batch.values).all()
