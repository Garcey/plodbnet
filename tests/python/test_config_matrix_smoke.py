"""Config-matrix smoke: 3 batched PPO updates at every (seats, stack)
planned for week-long training, to catch shape/dtype regressions that
only surface at specific seat counts or stack depths.

Covers plan verification #6 at reduced scale. Full 1k-update smoke per
cell is a pre-launch checklist item, not CI — those runs live in
`scripts/train.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.network import ActorCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout_batched
from plo5bp.selfplay import OpponentPool


CONFIGS = [
    (seats, stack)
    for seats in (2, 3, 4, 5, 6)
    for stack in (200000, 500000, 1000000)  # 20bb, 50bb, 100bb
]


@pytest.mark.parametrize("num_seats,starting_stack", CONFIGS)
def test_batched_ppo_smoke_across_configs(
    num_seats: int, starting_stack: int
) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    game_cfg = GameConfig(num_seats=num_seats, starting_stack=starting_stack)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=128,
        hidden_dim=32,
        ppo_epochs=1,
        batch_size=64,
        pool_mix_prob=0.0,
        pool_opp_seats=0,
    )
    model = ActorCritic(hidden_dim=train_cfg.hidden_dim)
    trainer = PPOTrainer(model, train_cfg)
    pool = OpponentPool(capacity=1)
    rng = np.random.default_rng(0)

    for _ in range(3):
        batch = collect_rollout_batched(model, pool, game_cfg, train_cfg, rng)
        stats = trainer.update(batch, rng)
        assert np.isfinite(stats.policy_loss)
        assert np.isfinite(stats.value_loss)
        assert np.isfinite(stats.entropy)
        assert np.isfinite(stats.approx_kl)

        # Batch-level invariants.
        assert batch.obs.shape[0] >= train_cfg.rollout_length
        assert batch.obs.shape[1] == model.torso[0].in_features
        gm = batch.gate_masks.numpy()
        ga = batch.gate_actions.numpy()
        rc = batch.raise_chips.numpy()
        rb = batch.sizing.numpy()
        for i in range(ga.shape[0]):
            assert bool(gm[i, int(ga[i])])
            if int(ga[i]) == 2:  # GATE_RAISE
                lo, hi = int(rb[i, 0]), int(rb[i, 1])
                assert lo <= int(rc[i]) <= hi
