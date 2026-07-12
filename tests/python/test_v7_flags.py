"""v7 WS1 substrate (V7_DESIGN.md): Q-surface semantics flags + the
terminal-boundary telemetry. Pins:

- `q_base_raw`: the dueling base becomes the RAW-space bin mean
  (Σ p_i·symexp(c_i)) while the display V stays the symlog-mean readout;
  the adv rows are unchanged (Q − base identical to the legacy Q − V).
- `q_fold_zero`: Q[FOLD] ≡ 0 exactly; sibling columns byte-identical to
  the unflagged critic; requires no state-dict change either direction.
- Both flags flow through q_values AND train_outputs (one composer).
- `Batch.is_terminal`: batched collectors flag each seat's last decision
  row; every fold row is terminal with return exactly 0 (the ground-truth
  identity the qF canary rests on); survives the multiconfig staging path.
- PPOStats.q_term_err (qT canary): defaults 0.0; finite through a real
  batched update; reads EXACTLY 0-fold-error under q_fold_zero.
"""

import numpy as np
import pytest
import torch

from plo5bp.actions import GATE_FOLD
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.network import (
    ActorCriticV2,
    ActorCriticV5,
    CentralCritic,
    _symexp,
    build_critic_from_state_dict,
)
from plo5bp.ppo import PPOStats, PPOTrainer
from plo5bp.rollout import collect_rollout_batched, collect_rollout_multiconfig
from plo5bp.selfplay import OpponentPool

OBS = 64
HID = 32


def _critic(**kw) -> CentralCritic:
    torch.manual_seed(11)
    c = CentralCritic(
        obs_dim=OBS, hidden_dim=HID, num_blocks=1,
        q_actions=3, value_bins=11, **kw,
    )
    with torch.no_grad():
        c.adv_head.weight.normal_(0.0, 0.3)
        c.adv_head.bias.normal_(0.0, 0.3)
    return c


def _inputs(n=7):
    torch.manual_seed(5)
    return torch.randn(n, OBS), torch.zeros(n, 260)


def test_q_base_raw_semantics():
    c = _critic(q_base_raw=True)
    obs, opp = _inputs()
    with torch.inference_mode():
        v, q = c.q_values(obs, opp)
        z = c.torso(torch.cat([obs, opp], dim=-1))
        probs = torch.softmax(c.value_head(z), dim=-1)
        base = (probs * _symexp(c._value_centers)).sum(-1)
        # display V unchanged: still symexp of the symlog-space mean
        assert torch.allclose(v, _symexp((probs * c._value_centers).sum(-1)))
        assert torch.allclose(q, base[:, None] + c.adv_head(z))
    # adv rows are base-independent: Q − base here == Q − V legacy
    c_off = _critic()
    c_off.load_state_dict(c.state_dict())
    with torch.inference_mode():
        v_off, q_off = c_off.q_values(obs, opp)
        assert torch.allclose(q - base[:, None], q_off - v_off[:, None])


def test_q_base_raw_zero_init_q_equals_raw_base():
    torch.manual_seed(3)
    c = CentralCritic(
        obs_dim=OBS, hidden_dim=HID, num_blocks=1,
        q_actions=3, value_bins=11, q_base_raw=True,
    )
    obs, opp = _inputs()
    with torch.inference_mode():
        _v, q = c.q_values(obs, opp)
        z = c.torso(torch.cat([obs, opp], dim=-1))
        probs = torch.softmax(c.value_head(z), dim=-1)
        base = (probs * c._raw_value_centers).sum(-1)
    assert torch.allclose(q, base[:, None].expand_as(q))


def test_q_fold_zero_pins_col0():
    c_on = _critic(q_fold_zero=True)
    c_off = _critic()
    c_off.load_state_dict(c_on.state_dict())
    obs, opp = _inputs()
    with torch.inference_mode():
        _v_on, q_on = c_on.q_values(obs, opp)
        _v_off, q_off = c_off.q_values(obs, opp)
    assert torch.equal(q_on[:, 0], torch.zeros_like(q_on[:, 0]))
    assert torch.equal(q_on[:, 1:], q_off[:, 1:])


def test_q_base_raw_requires_bins():
    with pytest.raises(ValueError):
        CentralCritic(
            obs_dim=OBS, hidden_dim=HID, num_blocks=1,
            q_actions=3, value_bins=0, q_base_raw=True,
        )


def test_train_outputs_matches_q_values_under_flags():
    # Both call sites must route through the same composer.
    c = _critic(q_base_raw=True, q_fold_zero=True)
    obs, opp = _inputs()
    with torch.inference_mode():
        _v1, q1 = c.q_values(obs, opp)
        _v2, _logits, q2 = c.train_outputs(obs, opp)
    assert torch.allclose(q1, q2)


def test_builder_passes_flags():
    c = _critic(q_base_raw=True, q_fold_zero=True)
    rebuilt = build_critic_from_state_dict(
        c.state_dict(), q_fold_zero=True, q_base_raw=True
    )
    obs, opp = _inputs()
    with torch.inference_mode():
        _va, qa = c.q_values(obs, opp)
        _vb, qb = rebuilt.q_values(obs, opp)
    assert torch.allclose(qa, qb)
    # default (flags off) still round-trips the same state dict strictly
    build_critic_from_state_dict(c.state_dict())


def _small_batched(train_cfg=None, game_cfg=None, critic=None):
    torch.manual_seed(0)
    np.random.seed(0)
    game_cfg = game_cfg or GameConfig(num_seats=4)
    train_cfg = train_cfg or TrainingConfig(
        num_envs=8, rollout_length=128, hidden_dim=32
    )
    model = ActorCriticV2(hidden_dim=32)
    model.eval()
    rng = np.random.default_rng(0)
    pool = OpponentPool(capacity=1)
    return collect_rollout_batched(
        model, pool, game_cfg, train_cfg, rng, critic=critic
    )


def test_is_terminal_batched():
    batch = _small_batched()
    t = batch.is_terminal
    assert t is not None and t.dtype == torch.bool
    assert t.shape == batch.gate_actions.shape
    frac = t.float().mean().item()
    assert 0.0 < frac < 1.0
    fold_rows = batch.gate_actions == GATE_FOLD
    assert fold_rows.any(), "collection produced no folds — enlarge rollout"
    # A fold ends the seat's hand: every fold row is terminal, and at a
    # terminal row `returns` is the raw realized reward — exactly 0 for a
    # folder (per-step-cost rewards, sunk chips excluded).
    assert bool(t[fold_rows].all())
    assert torch.equal(
        batch.returns[fold_rows], torch.zeros_like(batch.returns[fold_rows])
    )


def test_is_terminal_multiconfig_staging():
    torch.manual_seed(123)
    np.random.seed(123)
    model = ActorCriticV2(hidden_dim=32)
    model.eval()
    train_cfg = TrainingConfig(num_envs=6, rollout_length=120, hidden_dim=32)
    rng = np.random.default_rng(7)
    pool = OpponentPool(capacity=1)
    batch = collect_rollout_multiconfig(
        model, pool,
        [GameConfig(num_seats=3), GameConfig(num_seats=5)],
        train_cfg, rng, critic=None,
    )
    t = batch.is_terminal
    assert t is not None and t.dtype == torch.bool
    fold_rows = batch.gate_actions == GATE_FOLD
    assert bool(t[fold_rows].all())


def test_q_term_err_field_default():
    s = PPOStats(policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0)
    assert s.q_term_err == 0.0


def test_v7_flags_full_update():
    # The v7 candidate surface end-to-end: pooled Q + raw base + pinned fold
    # column + VRPO + HL-Gauss, through a real batched collect + update.
    # Under q_fold_zero the qF canary must read EXACTLY 0; qT is finite.
    torch.manual_seed(0)
    np.random.seed(0)
    game_cfg = GameConfig(num_seats=4)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=128,
        hidden_dim=32,
        num_layers=3,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        batch_size=64,
        advantage_estimator="vrpo",
        q_aux_coef=0.5,
        q_pooled=True,
        q_fold_zero=True,
        q_base_raw=True,
        value_bins=51,
        value_support=1500.0,
    )
    model = ActorCriticV5(hidden_dim=32, num_layers=3)
    critic = CentralCritic(
        hidden_dim=64,
        num_blocks=1,
        q_actions=3,
        value_bins=51,
        q_fold_zero=True,
        q_base_raw=True,
    )
    with torch.no_grad():
        critic.adv_head.weight.normal_(0.0, 0.2)
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    rng = np.random.default_rng(0)
    pool = OpponentPool(capacity=1)
    model.eval()
    batch = collect_rollout_batched(
        model, pool, game_cfg, train_cfg, rng, critic=critic
    )
    assert batch.is_terminal is not None
    model.train()
    stats = trainer.update(batch, rng)
    for name in ("policy_loss", "value_loss", "q_loss", "q_term_err"):
        assert np.isfinite(getattr(stats, name)), f"non-finite {name}"
    assert stats.q_fold_err == 0.0
