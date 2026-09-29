"""V6 optimizer hygiene: AdamW β2 knob + stateless per-tensor AGC.

Both default to no-ops (β2=0.999 is AdamW's own default; agc_clip=0), so the
existing training path is unchanged. Pins: β2 reaches the optimizer; AGC clips a
tensor whose grad exceeds clip*||param|| and leaves a small grad untouched (and
is stateless, so it never touches the kl_hard rollback); a PPO update stays
finite with both on.
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.network import ActorCriticV2, CentralCritic
from plo5bp.ppo import PPOTrainer, _adaptive_grad_clip_
from plo5bp.rollout import collect_rollout_batched
from plo5bp.selfplay import OpponentPool


def test_adam_b2_wired_and_default() -> None:
    t = PPOTrainer(ActorCriticV2(hidden_dim=32), TrainingConfig(adam_b2=0.98))
    assert abs(t.optimizer.param_groups[0]["betas"][1] - 0.98) < 1e-9
    d = PPOTrainer(ActorCriticV2(hidden_dim=32), TrainingConfig())
    assert abs(d.optimizer.param_groups[0]["betas"][1] - 0.999) < 1e-9  # unchanged


def test_agc_clips_large_leaves_small_and_skips_none() -> None:
    big = torch.nn.Parameter(torch.ones(10))  # ||p|| = sqrt(10)
    big.grad = torch.ones(10) * 100.0         # far above clip*||p||
    _adaptive_grad_clip_([big], clip=0.1)
    assert float(big.grad.norm()) <= 0.1 * float(big.detach().norm()) + 1e-5

    small = torch.nn.Parameter(torch.ones(10))
    small.grad = torch.full((10,), 1e-3)      # under the threshold
    before = small.grad.clone()
    _adaptive_grad_clip_([small], clip=1.0)
    assert torch.allclose(small.grad, before)

    # grad=None param is skipped, not a crash
    _adaptive_grad_clip_([torch.nn.Parameter(torch.ones(3))], clip=0.1)


def test_value_loss_coef_default_and_flows() -> None:
    assert TrainingConfig().value_loss_coef == 0.5  # == the old hardcode
    torch.manual_seed(3)
    np.random.seed(3)
    game_cfg = GameConfig(num_seats=4)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=128,
        hidden_dim=32,
        batch_size=64,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        value_loss_coef=2.0,
    )
    model = ActorCriticV2(hidden_dim=32)
    critic = CentralCritic(hidden_dim=64, num_blocks=1)
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    rng = np.random.default_rng(3)
    pool = OpponentPool(capacity=1)
    model.eval()
    batch = collect_rollout_batched(model, pool, game_cfg, train_cfg, rng, critic=critic)
    model.train()
    stats = trainer.update(batch, rng)
    assert np.isfinite(stats.value_loss) and np.isfinite(stats.policy_loss)


def test_agc_ppo_update_finite() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    game_cfg = GameConfig(num_seats=4)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=128,
        hidden_dim=32,
        batch_size=64,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        agc_clip=0.05,
        adam_b2=0.98,
    )
    model = ActorCriticV2(hidden_dim=32)
    critic = CentralCritic(hidden_dim=64, num_blocks=1)
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    rng = np.random.default_rng(0)
    pool = OpponentPool(capacity=1)
    model.eval()
    batch = collect_rollout_batched(model, pool, game_cfg, train_cfg, rng, critic=critic)
    model.train()
    stats = trainer.update(batch, rng)
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert np.isfinite(getattr(stats, name)), f"non-finite {name}: {stats}"
