"""PPO v2 smoke: anchor head + centralized critic + KL-to-EMA flag.

  - update() with (ActorCriticV2, CentralCritic) produces finite stats
    including the new entropy decomposition (gate/anchor/beta).
  - First minibatch of the first epoch sees ratio == 1 (approx_kl ≈ 0)
    because evaluate replays the stored (gate, anchor, u).
  - kl_anchor_coef == 0 → no EMA reference model is built at all;
    > 0 → reference exists, KL term is finite, EMA advances.
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.network import ActorCriticV2, CentralCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout_batched
from plo5bp.selfplay import OpponentPool


def _setup(**cfg_overrides):
    torch.manual_seed(0)
    np.random.seed(0)
    game_cfg = GameConfig(num_seats=4)
    cfg = dict(
        num_envs=4,
        rollout_length=128,
        hidden_dim=32,
        ppo_epochs=2,
        batch_size=64,
    )
    cfg.update(cfg_overrides)
    train_cfg = TrainingConfig(**cfg)
    model = ActorCriticV2(hidden_dim=train_cfg.hidden_dim)
    critic = CentralCritic(
        hidden_dim=train_cfg.critic_hidden_dim,
        num_blocks=train_cfg.critic_num_blocks,
    )
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    pool = OpponentPool(capacity=1)
    rng = np.random.default_rng(0)
    model.eval()
    batch = collect_rollout_batched(
        model, pool, game_cfg, train_cfg, rng, critic=critic
    )
    model.train()
    return trainer, batch, rng


def test_v2_update_finite_stats() -> None:
    trainer, batch, rng = _setup(critic_hidden_dim=64, critic_num_blocks=1)
    assert trainer._ref is None  # flag off → no EMA model built
    stats = trainer.update(batch, rng)
    for name in (
        "policy_loss", "value_loss", "entropy", "approx_kl",
        "display_loss", "gate_entropy", "anchor_entropy", "beta_entropy",
    ):
        assert np.isfinite(getattr(stats, name)), f"non-finite {name}: {stats}"
    assert stats.kl_anchor == 0.0
    # Critic in the loop → the display head trains as plain regression.
    assert stats.display_loss > 0.0


def test_first_minibatch_ratio_is_one() -> None:
    # Single epoch + one giant minibatch → the only evaluate happens on
    # pre-update weights, so approx_kl must be ~0 (replay exactness).
    trainer, batch, rng = _setup(
        critic_hidden_dim=64, critic_num_blocks=1,
        ppo_epochs=1, batch_size=1_000_000,
    )
    stats = trainer.update(batch, rng)
    assert abs(stats.approx_kl) < 1e-3, stats


def test_kl_anchor_flag_on() -> None:
    trainer, batch, rng = _setup(
        critic_hidden_dim=64, critic_num_blocks=1, kl_anchor_coef=0.01
    )
    assert trainer._ref is not None
    ref_before = [p.clone() for p in trainer._ref.parameters()]
    stats = trainer.update(batch, rng)
    assert np.isfinite(stats.kl_anchor)
    # Reference started as a copy of the actor; after the actor moved,
    # the post-update EMA must have nudged the reference too.
    moved = any(
        not torch.equal(b, p)
        for b, p in zip(ref_before, trainer._ref.parameters())
    )
    assert moved

def test_kl_guard_trips_and_skips_step() -> None:
    # Corrupt the stored log-probs so the very first minibatch shows a
    # huge approx_kl: the guard must abort before any optimizer step,
    # leaving the model untouched.
    trainer, batch, rng = _setup(critic_hidden_dim=64, critic_num_blocks=1)
    batch.log_probs.add_(10.0)  # kl = mean(stored - current) ≈ +10
    params_before = [p.clone() for p in trainer.model.parameters()]
    stats = trainer.update(batch, rng)
    assert stats.kl_stopped_at == 0, stats
    assert stats.kl_stop > 0.5, stats
    unchanged = all(
        torch.equal(b, p)
        for b, p in zip(params_before, trainer.model.parameters())
    )
    assert unchanged, "guard tripped but an optimizer step was applied"


def test_kl_guard_disabled_lets_update_through() -> None:
    trainer, batch, rng = _setup(
        critic_hidden_dim=64, critic_num_blocks=1, target_kl=0.0
    )
    batch.log_probs.add_(10.0)
    params_before = [p.clone() for p in trainer.model.parameters()]
    stats = trainer.update(batch, rng)
    assert stats.kl_stopped_at == -1, stats
    moved = any(
        not torch.equal(b, p)
        for b, p in zip(params_before, trainer.model.parameters())
    )
    assert moved, "update should proceed when the guard is off"


def test_entropy_bonus_gate_gradient_is_pure_gate_entropy() -> None:
    # The entropy bonus must not pay the gate head for shifting mass
    # onto Raise: the anchor head's ~2.4 nats made the joint-entropy
    # gradient anti-fold and drove fold to ~5e-5 everywhere within 5
    # updates (vTwo1 2026-06-11). With p_raise detached, the entropy's
    # gradient w.r.t. the gate head must equal the gradient of pure
    # gate entropy — the conditional sizing term contributes nothing
    # through the gate.
    trainer, batch, rng = _setup(critic_hidden_dim=64, critic_num_blocks=1)
    model = trainer.model
    n = min(256, batch.obs.shape[0])
    args = (
        batch.obs[:n], batch.gate_masks[:n], batch.sizing[:n],
        batch.gate_actions[:n], batch.anchor_actions[:n],
        batch.refine_u[:n],
    )
    _, entropy, _, gate_h, _, _ = model.evaluate(*args)
    g_total = torch.autograd.grad(
        entropy.sum(), model.gate_head.weight, retain_graph=True
    )[0]
    g_gate = torch.autograd.grad(gate_h.sum(), model.gate_head.weight)[0]
    assert torch.allclose(g_total, g_gate, atol=1e-6), (
        "entropy bonus leaks non-gate-entropy gradient into the gate head "
        f"(max delta {(g_total - g_gate).abs().max().item():.3e})"
    )


def test_display_value_head_detached_from_torso() -> None:
    # v2's display value head regresses raw-bb returns (hundreds of bb);
    # trained through the shared torso its gradients dwarf the policy
    # gradient. It must read a detached torso: gradient flows to the
    # head itself, never to torso parameters.
    trainer, batch, rng = _setup(critic_hidden_dim=64, critic_num_blocks=1)
    model = trainer.model
    n = min(64, batch.obs.shape[0])
    _, _, value, _, _, _ = model.evaluate(
        batch.obs[:n], batch.gate_masks[:n], batch.sizing[:n],
        batch.gate_actions[:n], batch.anchor_actions[:n],
        batch.refine_u[:n],
    )
    loss = value.pow(2).sum()
    torso_params = list(model.torso.parameters())
    grads = torch.autograd.grad(
        loss, torso_params + [model.value_head.weight], allow_unused=True
    )
    assert grads[-1] is not None and grads[-1].abs().sum() > 0
    assert all(g is None for g in grads[:-1]), (
        "display value loss reaches torso parameters"
    )
