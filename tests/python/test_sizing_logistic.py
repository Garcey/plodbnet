"""v4 sizing head: ordinal discretized-logistic over the 11 anchors.

Covers the new pieces on top of the (unchanged) v2 act/evaluate/PPO machinery:
  - `_discretized_logistic_probs`: sums to 1 over legal anchors; the END anchors
    absorb the tails so exact-min / exact-pot stay hittable and concentratable
    (the property a plain continuous head would lose); legality masking +
    renormalization; the scale floor keeps interior modes off a one-hot spike.
  - `ActorCriticV4`: correct head wiring (size_head, no anchor_head, v3 sniff),
    finite act/evaluate, and act↔evaluate log-prob PARITY (the 1.0-ratio contract).
  - end-to-end PPO update with the v4 head produces finite stats.
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.actions import GATE_ACTIONS
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.network import (
    ANCHOR_COUNT,
    ActorCriticV2,
    ActorCriticV4,
    _discretized_logistic_probs,
    model_class_for_state_dict,
)
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout_batched
from plo5bp.selfplay import OpponentPool

_SIZING = torch.tensor([[200, 10000, 5000, 0]], dtype=torch.int64)  # all 11 anchors legal


# ---- the discretized-logistic helper ------------------------------------

def test_probs_sum_to_one_over_legal() -> None:
    mu = torch.tensor([5.0, 0.0, 9.0])
    s = torch.tensor([1.0, 0.5, 2.0])
    legal = torch.ones(3, ANCHOR_COUNT, dtype=torch.bool)
    p = _discretized_logistic_probs(mu, s, legal)
    assert p.shape == (3, ANCHOR_COUNT)
    assert torch.allclose(p.sum(-1), torch.ones(3), atol=1e-5)
    assert (p >= 0).all()


def test_endpoints_absorb_tails_and_concentrate() -> None:
    # The exact-pot / exact-min requirement: pushing mu past an end with a tight
    # spread puts (almost) all the mass on that exact anchor.
    legal = torch.ones(1, ANCHOR_COUNT, dtype=torch.bool)
    p_pot = _discretized_logistic_probs(torch.tensor([12.0]), torch.tensor([0.3]), legal)
    assert p_pot[0, ANCHOR_COUNT - 1] > 0.9          # exact pot reachable + concentratable
    p_min = _discretized_logistic_probs(torch.tensor([-2.0]), torch.tensor([0.3]), legal)
    assert p_min[0, 0] > 0.9                          # exact min reachable + concentratable


def test_interior_mode_is_not_one_hot_at_scale_floor() -> None:
    # With the scale floored, an interior preference still leaks to neighbours
    # (no one-hot collapse) — the anti-collapse property.
    legal = torch.ones(1, ANCHOR_COUNT, dtype=torch.bool)
    p = _discretized_logistic_probs(torch.tensor([5.0]), torch.tensor([0.3]), legal)
    assert p[0, 5] < 0.95 and p[0, 4] > 0.01 and p[0, 6] > 0.01
    ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
    assert ent.item() > 0.3                           # entropy bounded above 0


def test_legality_masking_renormalizes() -> None:
    legal = torch.ones(1, ANCHOR_COUNT, dtype=torch.bool)
    legal[0, 3] = False
    legal[0, 7] = False
    p = _discretized_logistic_probs(torch.tensor([5.0]), torch.tensor([1.5]), legal)
    assert p[0, 3] < 1e-6 and p[0, 7] < 1e-6
    assert abs(p.sum().item() - 1.0) < 1e-5


def test_capped_high_mu_lands_on_highest_legal() -> None:
    # Legal-edge tail absorption: mu pinned past the top of a CAPPED legal range
    # with a tight spread must pile mass on the highest LEGAL anchor (nearest mu),
    # NOT on min. Pre-fix this gave ~100% on anchor 0 — the only legal anchor
    # whose raw-CDF formula didn't cancel to ~0 under the [-12,12] clamp.
    legal = torch.zeros(1, ANCHOR_COUNT, dtype=torch.bool)
    legal[0, :7] = True                              # anchors 0..6 (min..60%) legal
    p = _discretized_logistic_probs(torch.tensor([12.0]), torch.tensor([0.35]), legal)
    assert int(p.argmax()) == 6, p
    assert p[0, 6] > 0.9 and p[0, 0] < 0.05
    assert (p[0, 7:] == 0).all()                     # illegal anchors stay zero
    assert abs(p.sum().item() - 1.0) < 1e-5


def test_capped_low_mu_lands_on_lowest_legal() -> None:
    # Symmetric: mu past the bottom of a range floored above min -> lowest legal.
    legal = torch.zeros(1, ANCHOR_COUNT, dtype=torch.bool)
    legal[0, 4:] = True                              # anchors 4..10 legal
    p = _discretized_logistic_probs(torch.tensor([-2.0]), torch.tensor([0.35]), legal)
    assert int(p.argmax()) == 4, p
    assert p[0, 4] > 0.9
    assert (p[0, :4] == 0).all()


def test_single_legal_anchor_gets_all_mass() -> None:
    # Degenerate 1-anchor legal set: that anchor takes ~all the mass for any mu.
    for k in (0, 5, ANCHOR_COUNT - 1):
        legal = torch.zeros(1, ANCHOR_COUNT, dtype=torch.bool)
        legal[0, k] = True
        p = _discretized_logistic_probs(torch.tensor([12.0]), torch.tensor([0.5]), legal)
        assert p[0, k] > 0.999, (k, p)


# ---- the ActorCriticV4 head ----------------------------------------------

def test_v4_head_wiring_and_sniff() -> None:
    m = ActorCriticV4(hidden_dim=32)
    assert m.head_version == 3
    sd = m.state_dict()
    assert "size_head.weight" in sd and "anchor_head.weight" not in sd
    assert "refine_head.weight" in sd        # Beta refine kept (Option B)
    assert model_class_for_state_dict(sd) is ActorCriticV4
    # v2 still sniffs to v2 (regression guard for the shared sniffer).
    assert model_class_for_state_dict(ActorCriticV2(hidden_dim=32).state_dict()) is ActorCriticV2


def test_v4_act_evaluate_finite_and_parity() -> None:
    torch.manual_seed(0)
    m = ActorCriticV4(hidden_dim=32).eval()
    B = 16
    obs = torch.randn(B, OBS_DIM)
    gate_mask = torch.ones(B, GATE_ACTIONS, dtype=torch.bool)
    sizing = _SIZING.expand(B, 4).contiguous()

    out = m.act(obs, gate_mask, sizing, deterministic=False)
    assert out.anchor.min() >= 0 and out.anchor.max() < ANCHOR_COUNT
    assert torch.isfinite(out.log_prob).all()
    assert (out.chips >= 0).all()

    lp, ent, val, gate_h, anchor_h, beta_h, glp, alp = m.evaluate(
        obs, gate_mask, sizing, out.gate, out.anchor, out.refine_u
    )
    for t in (lp, ent, val, gate_h, anchor_h, beta_h):
        assert torch.isfinite(t).all()
    # PARITY: replaying the stored action reproduces act()'s joint log-prob
    # (the "ratio == 1 at epoch start" contract PPO depends on).
    assert torch.allclose(out.log_prob, lp, atol=1e-5), (out.log_prob - lp).abs().max()


def test_v4_ppo_update_finite() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    game_cfg = GameConfig(num_seats=4)
    train_cfg = TrainingConfig(
        num_envs=4, rollout_length=128, hidden_dim=32, ppo_epochs=2, batch_size=64,
        critic_hidden_dim=64, critic_num_blocks=1,
    )
    model = ActorCriticV4(hidden_dim=train_cfg.hidden_dim)
    from plo5bp.network import CentralCritic
    critic = CentralCritic(
        hidden_dim=train_cfg.critic_hidden_dim, num_blocks=train_cfg.critic_num_blocks
    )
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    pool = OpponentPool(capacity=1)
    rng = np.random.default_rng(0)
    model.eval()
    batch = collect_rollout_batched(model, pool, game_cfg, train_cfg, rng, critic=critic)
    model.train()
    stats = trainer.update(batch, rng)
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl",
                 "gate_entropy", "anchor_entropy", "beta_entropy"):
        assert np.isfinite(getattr(stats, name)), f"non-finite {name}: {stats}"
