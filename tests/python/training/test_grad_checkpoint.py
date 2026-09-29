"""V6 gradient checkpointing: recompute torso activations in backward.

Identical math (forward values unchanged), only backward memory differs. Pins:
the checkpointed forward is bit-close to the plain forward; a full PPO update
with checkpointing AND the kl-anchor magnet on stays finite (the magnet reuses
evaluate()'s forward graph — the interaction the red-team flagged); and the
rollout's no-grad path is untouched (checkpoint only fires under grad).
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.actions import GATE_ACTIONS
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.network import ActorCriticV5, CentralCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout_batched
from plo5bp.selfplay import OpponentPool

Q_ACTIONS = 2 + 11


def test_checkpoint_forward_values_identical() -> None:
    torch.manual_seed(0)
    m = ActorCriticV5(hidden_dim=32, num_layers=3)  # residual torso
    obs = torch.randn(8, OBS_DIM)
    gm = torch.ones(8, GATE_ACTIONS, dtype=torch.bool)

    m._grad_checkpoint = False
    plain = m.forward(obs, gm)
    m._grad_checkpoint = True  # grad is enabled by default → checkpoint fires
    ckpt = m.forward(obs, gm)
    for a, b in zip(plain, ckpt):
        assert torch.allclose(a.detach(), b.detach(), atol=1e-5)


def test_grad_checkpoint_ppo_update_finite_with_magnet() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    game_cfg = GameConfig(num_seats=4)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=128,
        hidden_dim=32,
        num_layers=3,
        batch_size=64,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        grad_checkpoint=True,
        kl_anchor_coef=0.05,  # magnet ON — checkpoint must not break its reuse
    )
    model = ActorCriticV5(hidden_dim=32, num_layers=3)
    critic = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=Q_ACTIONS)
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    # the trainer sets the runtime flag on the trainable model + critic
    assert model._grad_checkpoint and critic._grad_checkpoint

    rng = np.random.default_rng(0)
    pool = OpponentPool(capacity=1)
    model.eval()
    # rollout runs under inference_mode → checkpoint is skipped (no crash)
    batch = collect_rollout_batched(model, pool, game_cfg, train_cfg, rng, critic=critic)
    model.train()
    stats = trainer.update(batch, rng)
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert np.isfinite(getattr(stats, name)), f"non-finite {name}: {stats}"
