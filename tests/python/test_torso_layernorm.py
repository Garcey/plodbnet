"""v6 plasticity change (V6_RESEARCH.md #4): pre-activation LayerNorm on the
residual torso of BOTH actor and critic, plus its required weight-decay-to-init
companion.

Pins:
  - the residual block gains a `norm.*` submodule only under use_norm (off →
    byte-identical param names, so pre-v6 checkpoints are untouched);
  - actor + critic accept `torso_layernorm`, and the state dict is
    auto-detectable (`_torso_has_norm`) so the pool-snapshot rebuild and the UI
    reconstruct the right architecture without a saved flag — round-trips load
    strict, forward-identical;
  - LayerNorm requires the residual torso (num_layers >= 3);
  - the L2-to-init companion snapshots the TRUNK weight matrices only (heads
    stay free), the penalty rises as a trunk weight leaves init, and a PPO
    update stays finite with it on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.actions import GATE_ACTIONS
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.network import (
    ActorCriticV5,
    CentralCritic,
    _ResidualBlock,
    _torso_has_norm,
    build_actor_from_state_dict,
    build_critic_from_state_dict,
)
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout_batched
from plo5bp.selfplay import OpponentPool

Q_ACTIONS = 2 + 11


def test_residual_block_norm_gated() -> None:
    on = _ResidualBlock(16, use_norm=True)
    assert on.norm is not None
    assert "norm.weight" in on.state_dict() and "linear.weight" in on.state_dict()
    x = torch.randn(4, 16)
    y = on(x)
    assert y.shape == (4, 16) and torch.isfinite(y).all()

    off = _ResidualBlock(16, use_norm=False)
    assert off.norm is None
    assert "norm.weight" not in off.state_dict()


def test_actor_critic_norm_state_dict_detectable() -> None:
    a_on = ActorCriticV5(hidden_dim=32, num_layers=3, torso_layernorm=True)
    a_off = ActorCriticV5(hidden_dim=32, num_layers=3, torso_layernorm=False)
    assert _torso_has_norm(a_on.state_dict())
    assert not _torso_has_norm(a_off.state_dict())

    c_on = CentralCritic(hidden_dim=32, num_blocks=1, q_actions=Q_ACTIONS, torso_layernorm=True)
    c_off = CentralCritic(hidden_dim=32, num_blocks=1, q_actions=Q_ACTIONS, torso_layernorm=False)
    assert _torso_has_norm(c_on.state_dict())
    assert not _torso_has_norm(c_off.state_dict())


def test_layernorm_requires_residual_depth() -> None:
    with pytest.raises(ValueError):
        ActorCriticV5(hidden_dim=32, num_layers=2, torso_layernorm=True)


def test_actor_roundtrip_load_and_forward_identical() -> None:
    torch.manual_seed(0)
    m = ActorCriticV5(hidden_dim=32, num_layers=3, torso_layernorm=True).eval()
    rebuilt = build_actor_from_state_dict(
        m.state_dict(), hidden_dim=32, num_layers=3
    ).eval()
    assert _torso_has_norm(rebuilt.state_dict())
    rebuilt.load_state_dict(m.state_dict())  # strict load must succeed
    B = 8
    obs = torch.randn(B, OBS_DIM)
    gm = torch.ones(B, GATE_ACTIONS, dtype=torch.bool)
    with torch.no_grad():
        for x, y in zip(m.forward(obs, gm), rebuilt.forward(obs, gm)):
            assert torch.allclose(x, y)


def test_critic_roundtrip_load_ln_and_plain() -> None:
    c = CentralCritic(hidden_dim=32, num_blocks=1, q_actions=Q_ACTIONS, torso_layernorm=True)
    rc = build_critic_from_state_dict(c.state_dict())
    assert _torso_has_norm(rc.state_dict())
    rc.load_state_dict(c.state_dict())

    c0 = CentralCritic(hidden_dim=32, num_blocks=1, q_actions=Q_ACTIONS, torso_layernorm=False)
    rc0 = build_critic_from_state_dict(c0.state_dict())
    assert not _torso_has_norm(rc0.state_dict())
    rc0.load_state_dict(c0.state_dict())


def test_l2_init_snapshots_trunk_only() -> None:
    torch.manual_seed(0)
    model = ActorCriticV5(hidden_dim=32, num_layers=3, torso_layernorm=True)
    critic = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=Q_ACTIONS, torso_layernorm=True)
    cfg = TrainingConfig(hidden_dim=32, num_layers=3, l2_init_coef=1e-3)
    trainer = PPOTrainer(model, cfg, critic=critic)

    named = list(model.named_parameters()) + list(critic.named_parameters())
    expected_trunk = [p for n, p in named if "torso" in n and p.dim() >= 2]
    assert trainer._l2_init_pairs, "no trunk params snapshotted"
    assert len(trainer._l2_init_pairs) == len(expected_trunk)

    # heads must NOT be regularized
    snap_ids = {id(p) for p, _ in trainer._l2_init_pairs}
    head_params = [p for n, p in named if "head" in n and p.dim() >= 2]
    assert head_params and all(id(hp) not in snap_ids for hp in head_params)

    # equal to init at construction; penalty rises as a trunk weight moves
    for p, p0 in trainer._l2_init_pairs:
        assert torch.equal(p, p0)
    with torch.no_grad():
        trainer._l2_init_pairs[0][0].add_(1.0)
    pen = sum(((p - p0) ** 2).sum() for p, p0 in trainer._l2_init_pairs)
    assert float(pen.detach()) > 0.0


def test_l2_init_ppo_update_finite() -> None:
    torch.manual_seed(1)
    np.random.seed(1)
    game_cfg = GameConfig(num_seats=4)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=128,
        hidden_dim=32,
        num_layers=3,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        batch_size=64,
        torso_layernorm=True,
        l2_init_coef=1e-3,
        q_aux_coef=0.5,
    )
    model = ActorCriticV5(hidden_dim=32, num_layers=3, torso_layernorm=True)
    critic = CentralCritic(
        hidden_dim=64, num_blocks=1, q_actions=Q_ACTIONS, torso_layernorm=True
    )
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    pool = OpponentPool(capacity=1)
    rng = np.random.default_rng(1)
    model.eval()
    batch = collect_rollout_batched(model, pool, game_cfg, train_cfg, rng, critic=critic)
    model.train()
    stats = trainer.update(batch, rng)
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert np.isfinite(getattr(stats, name)), f"non-finite {name}: {stats}"
