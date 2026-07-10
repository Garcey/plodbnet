"""V6 keystone: distributional / HL-Gauss critic value head over a symlog
support (dynamic-range fix), with V = symexp(E[bins]) kept scalar so the dueling
Q and every caller are unchanged.

Pins: symlog round-trips; the categorical head's scalar reduction recovers the
target; the HL-Gauss soft target sums to 1; a full PPO update through the
distributional branch (q-aux + magnet on) stays finite; the scalar head is left
non-distributional (byte-identical path); and a distributional checkpoint
round-trips through the pool/UI loader (value_bins auto-sniffed).
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.network import (
    ActorCriticV5,
    CentralCritic,
    _symexp,
    _symlog,
    build_critic_from_state_dict,
)
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout_batched
from plo5bp.selfplay import OpponentPool

Q_ACTIONS = 2 + 11
OPP_DIM = 5 * 52


def test_symlog_roundtrip() -> None:
    x = torch.tensor([0.0, 5.0, -20.0, 100.0, 1500.0, -250.0])
    assert torch.allclose(_symexp(_symlog(x)), x, atol=1e-2)


def test_distributional_shapes_and_scalar_reduction() -> None:
    torch.manual_seed(0)
    c = CentralCritic(hidden_dim=32, num_blocks=1, q_actions=Q_ACTIONS, value_bins=51)
    obs = torch.randn(4, OBS_DIM)
    opp = torch.randn(4, OPP_DIM)
    v = c(obs, opp)
    assert v.shape == (4,)  # forward stays scalar
    vt, logits, q = c.train_outputs(obs, opp)
    assert vt.shape == (4,) and logits.shape == (4, 51) and q.shape == (4, Q_ACTIONS)

    # a head whose probs == the HL-Gauss target recovers the return via
    # symexp(E[bins]) — the reduction is essentially unbiased.
    ret = torch.tensor([0.0, 5.0, -20.0, 100.0])
    y = _symlog(ret)[:, None]
    inv = 1.0 / (c.hlgauss_sigma * 1.4142135623730951)
    cdf = 0.5 * (1.0 + torch.erf((c._value_edges[None, :] - y) * inv))
    target = cdf[:, 1:] - cdf[:, :-1]
    target = target / target.sum(-1, keepdim=True)
    recovered = _symexp((target * c._value_centers).sum(-1))
    assert torch.allclose(recovered, ret, atol=0.5), recovered


def test_hlgauss_target_sums_to_one_and_loss_finite() -> None:
    torch.manual_seed(1)
    c = CentralCritic(hidden_dim=32, num_blocks=1, q_actions=Q_ACTIONS, value_bins=41)
    obs = torch.randn(6, OBS_DIM)
    opp = torch.randn(6, OPP_DIM)
    _, logits, _ = c.train_outputs(obs, opp)
    ret = torch.tensor([0.0, 3.0, -3.0, 250.0, -250.0, 12.0])
    loss = c.hlgauss_value_loss(logits, ret)
    assert torch.isfinite(loss) and loss.item() > 0.0

    y = _symlog(ret)[:, None]
    inv = 1.0 / (c.hlgauss_sigma * 1.4142135623730951)
    cdf = 0.5 * (1.0 + torch.erf((c._value_edges[None, :] - y) * inv))
    target = cdf[:, 1:] - cdf[:, :-1]
    assert torch.allclose(target.sum(-1), torch.ones(6), atol=1e-4)


def test_scalar_critic_stays_non_distributional() -> None:
    c = CentralCritic(hidden_dim=32, num_blocks=1, q_actions=Q_ACTIONS)  # value_bins=0
    v, logits, q = c.train_outputs(torch.randn(4, OBS_DIM), torch.randn(4, OPP_DIM))
    assert v.shape == (4,) and logits is None and q.shape == (4, Q_ACTIONS)
    trainer = PPOTrainer(ActorCriticV5(hidden_dim=32, num_layers=3), TrainingConfig(), critic=c)
    assert trainer._distributional is False and trainer._critic_train is None


def test_distributional_ppo_update_finite() -> None:
    torch.manual_seed(2)
    np.random.seed(2)
    game_cfg = GameConfig(num_seats=4)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=128,
        hidden_dim=32,
        num_layers=3,
        batch_size=64,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        q_aux_coef=0.5,
        kl_anchor_coef=0.05,  # magnet on — full stack
        value_bins=51,
    )
    model = ActorCriticV5(hidden_dim=32, num_layers=3)
    critic = CentralCritic(
        hidden_dim=64, num_blocks=1, q_actions=Q_ACTIONS, value_bins=51
    )
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    assert trainer._distributional and trainer._critic_train is not None
    rng = np.random.default_rng(2)
    pool = OpponentPool(capacity=1)
    model.eval()
    batch = collect_rollout_batched(model, pool, game_cfg, train_cfg, rng, critic=critic)
    model.train()
    stats = trainer.update(batch, rng)
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert np.isfinite(getattr(stats, name)), f"non-finite {name}: {stats}"
    assert stats.value_loss > 0.0  # HL-Gauss cross-entropy


def test_distributional_warmstart_roundtrip() -> None:
    torch.manual_seed(3)
    c = CentralCritic(hidden_dim=32, num_blocks=1, q_actions=Q_ACTIONS, value_bins=51)
    rebuilt = build_critic_from_state_dict(c.state_dict())
    assert rebuilt.value_bins == 51
    rebuilt.load_state_dict(c.state_dict())  # strict
    obs = torch.randn(5, OBS_DIM)
    opp = torch.randn(5, OPP_DIM)
    with torch.no_grad():
        assert torch.allclose(c(obs, opp), rebuilt(obs, opp), atol=1e-5)
