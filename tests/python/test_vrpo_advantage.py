"""VRPO / Expected-SARSA(λ) advantage flip (V5_DESIGN.md W2.5).

The load-bearing correctness pin is the GOLDEN PARITY: with the critic's
dueling Q head at its zero-init state (Q ≡ V for every action), the
Expected-SARSA(λ) advantage reduces *exactly* to V-based GAE(λ). That is
what makes flipping `advantage_estimator="vrpo"` a no-op on a fresh v5
checkpoint and only a divergence once the Q head is trained — i.e. the
flip is a code change, not a checkpoint break.

Also pinned:
  - the flip actually changes the advantage signal once Q ≠ V;
  - a PPO update stays finite under the vrpo estimator;
  - the serial collector rejects vrpo (batched-only);
  - `act(return_marginal=True)` returns a valid (2+anchor)-way distribution
    over the Q-head action layout, and None when not requested.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from plo5bp.actions import GATE_ACTIONS
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.network import ActorCriticV2, CentralCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout, collect_rollout_batched
from plo5bp.selfplay import OpponentPool

NUM_SEATS = 4
Q_ACTIONS = 2 + 11  # Fold, CheckCall, Raise@anchor_0..10 (PLO 11-anchor grid)


def _cfg(estimator: str) -> tuple[GameConfig, TrainingConfig]:
    game_cfg = GameConfig(num_seats=NUM_SEATS)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=256,
        hidden_dim=32,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        q_aux_coef=0.5,
        advantage_estimator=estimator,
    )
    return game_cfg, train_cfg


def _run(estimator: str, seed: int, critic: CentralCritic):
    """One batched rollout with a fresh model seeded by `seed`. The model
    weights AND the post-init torch-RNG state are a pure function of `seed`,
    and `return_marginal` adds no RNG draw, so two runs at the same seed +
    same critic have bit-identical trajectories — only the advantage math
    can differ."""
    game_cfg, train_cfg = _cfg(estimator)
    torch.manual_seed(seed)
    model = ActorCriticV2(hidden_dim=32)
    model.eval()
    rng = np.random.default_rng(seed)
    pool = OpponentPool(capacity=1)
    return collect_rollout_batched(
        model, pool, game_cfg, train_cfg, rng, critic=critic
    )


def test_vrpo_reduces_to_gae_at_zero_init_q() -> None:
    # Zero-init adv_head → Q ≡ V for every action, so Expected-SARSA(λ) is
    # bit-identical to GAE(λ) up to f32 reduction-order noise.
    torch.manual_seed(100)
    critic = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=Q_ACTIONS).eval()
    gae = _run("gae", 7, critic)
    vrpo = _run("vrpo", 7, critic)
    assert gae.advantages.shape == vrpo.advantages.shape, (
        gae.advantages.shape, vrpo.advantages.shape
    )
    assert torch.allclose(gae.advantages, vrpo.advantages, atol=1e-4), (
        f"max |Δadv| = {float((gae.advantages - vrpo.advantages).abs().max())}"
    )
    # `returns` (the value-head target) stay GAE-based under both.
    assert torch.allclose(gae.returns, vrpo.returns, atol=1e-4)


def test_vrpo_differs_from_gae_with_trained_q() -> None:
    # Perturb the dueling advantage head so Q ≠ V (a "trained" head): the
    # flip must now move the advantage signal.
    torch.manual_seed(101)
    critic = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=Q_ACTIONS)
    with torch.no_grad():
        critic.adv_head.weight.normal_(0.0, 0.5)
        critic.adv_head.bias.normal_(0.0, 0.5)
    critic.eval()
    gae = _run("gae", 9, critic)
    vrpo = _run("vrpo", 9, critic)
    assert torch.isfinite(vrpo.advantages).all()
    assert not torch.allclose(gae.advantages, vrpo.advantages, atol=1e-3), (
        "vrpo advantages coincide with GAE even though Q != V"
    )


def test_vrpo_ppo_update_finite() -> None:
    torch.manual_seed(102)
    np.random.seed(102)
    game_cfg, train_cfg = _cfg("vrpo")
    train_cfg = dataclasses.replace(train_cfg, batch_size=64)
    model = ActorCriticV2(hidden_dim=32)
    critic = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=Q_ACTIONS)
    with torch.no_grad():
        critic.adv_head.weight.normal_(0.0, 0.3)
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    rng = np.random.default_rng(102)
    pool = OpponentPool(capacity=1)
    model.eval()
    batch = collect_rollout_batched(
        model, pool, game_cfg, train_cfg, rng, critic=critic
    )
    model.train()
    stats = trainer.update(batch, rng)
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "q_loss"):
        assert np.isfinite(getattr(stats, name)), f"non-finite {name}: {stats}"


def test_serial_collector_rejects_vrpo() -> None:
    game_cfg, train_cfg = _cfg("vrpo")
    torch.manual_seed(1)
    model = ActorCriticV2(hidden_dim=32).eval()
    critic = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=Q_ACTIONS).eval()
    rng = np.random.default_rng(1)
    pool = OpponentPool(capacity=1)
    with pytest.raises(NotImplementedError):
        collect_rollout(model, pool, game_cfg, train_cfg, rng, critic=critic)


def test_action_marginal_is_valid_distribution() -> None:
    torch.manual_seed(3)
    model = ActorCriticV2(hidden_dim=32).eval()
    B = 16
    obs = torch.randn(B, OBS_DIM)
    gate_mask = torch.ones(B, GATE_ACTIONS, dtype=torch.bool)
    sizing = (
        torch.tensor([[200, 10000, 5000, 0]], dtype=torch.int64)
        .expand(B, 4)
        .contiguous()
    )
    out = model.act(obs, gate_mask, sizing, return_marginal=True)
    assert out.action_marginal is not None
    assert out.action_marginal.shape == (B, Q_ACTIONS)
    assert (out.action_marginal >= 0).all()
    assert torch.allclose(
        out.action_marginal.sum(-1), torch.ones(B), atol=1e-5
    )
    # Off by default → free when the estimator is GAE.
    assert model.act(obs, gate_mask, sizing).action_marginal is None
