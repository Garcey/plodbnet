"""--critic-lr (2026-09-24): the centralized critic in its own AdamW group.

critic_lr = 0 keeps the single param group every stem trained with; > 0 puts
the critic in a second group at that rate, `set_lr` keeps the ratio through
the warmup ramp / live lr edits, and an optimizer sidecar written by either
layout restores into the other (the parameter order is the same).
"""

from __future__ import annotations

import torch

from plo5bp.config import TrainingConfig
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.network import ActorCriticV5, CentralCritic
from plo5bp.ppo import PPOTrainer


def _trainer(critic_lr: float) -> PPOTrainer:
    torch.manual_seed(0)
    actor = ActorCriticV5(hidden_dim=16, obs_dim=OBS_DIM_MINIMAL, num_layers=3, torso_layernorm=True)
    critic = CentralCritic(obs_dim=OBS_DIM_MINIMAL, hidden_dim=16, num_blocks=1, q_actions=3, value_bins=51)
    cfg = TrainingConfig(lr=2e-4, critic_lr=critic_lr, obs_mode="minimal")
    return PPOTrainer(actor, cfg, critic=critic)


def test_default_is_one_group() -> None:
    t = _trainer(0.0)
    assert len(t.optimizer.param_groups) == 1
    t.set_lr(3e-4)
    assert t.optimizer.param_groups[0]["lr"] == 3e-4


def test_separate_critic_group_follows_the_ratio() -> None:
    t = _trainer(5e-5)
    groups = t.optimizer.param_groups
    assert len(groups) == 2
    assert groups[0]["lr"] == 2e-4 and groups[1]["lr"] == 5e-5
    assert {id(p) for p in groups[1]["params"]} == {id(p) for p in t.critic.parameters()}
    t.set_lr(1e-4)  # e.g. half-way through a warmup ramp
    assert groups[0]["lr"] == 1e-4
    assert abs(groups[1]["lr"] - 2.5e-5) < 1e-18


def test_sidecar_restores_across_layouts() -> None:
    one, two = _trainer(0.0), _trainer(5e-5)
    for p in one._all_params:  # give the single-group optimizer some state
        p.grad = torch.ones_like(p)
    one.optimizer.step()
    side = one.optimizer_sidecar_state()
    ok, why = two.load_optimizer_moments(side)
    assert ok, why
    s1, s2 = one.optimizer.state_dict()["state"], two.optimizer.state_dict()["state"]
    assert s1.keys() == s2.keys()
    for k in s1:
        assert torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"])
    assert len(two.optimizer.param_groups) == 2
