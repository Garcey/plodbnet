"""Critic redesign (2026-09-26 regression diagnosis; docs/training-log.md "Regression
diagnosis + redesign").

Pinned:
- CentralCritic's new choices (SiLU / GELU torso, LayerNorm'd input block,
  raw-space-mean V readout) default OFF and leave a legacy critic's state dict
  and outputs untouched; any non-default choice rides in the `_arch` buffer so
  `build_critic_from_state_dict` rebuilds the same function;
- v_raw reads V as sum_i p_i symexp(c_i) (the distribution's mean), and the Q
  base follows it;
- q_fold_zero pins Q[FOLD] to exactly 0;
- PPOTrainer: critic-only passes (critic_extra_epochs, critic_minibatches)
  never touch an actor parameter (value, grad or Adam moment); the
  actor-frozen warm-up leaves the actor bit-identical and reports the critic's
  losses; critic_q_norm divides the Q losses by (return variance + 1) and is
  a no-op when off;
- load_optimizer_moments restores ONLY the actor's moments into a trainer with
  a new critic when allow_actor_only_moments is set, and refuses otherwise.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.network import (
    ActorCriticV2,
    CentralCritic,
    _symexp,
    build_critic_from_state_dict,
)
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout_batched
from plo5bp.selfplay import OpponentPool

BB = 10_000
OBS = 64


def _x(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, OBS, generator=g), torch.rand(n, 260, generator=g).round()


def test_defaults_are_the_legacy_critic():
    torch.manual_seed(0)
    c = CentralCritic(obs_dim=OBS, hidden_dim=16, num_blocks=2, q_actions=3,
                      torso_layernorm=True, value_bins=51)
    sd = c.state_dict()
    assert "_arch" not in sd and "torso.0.1.weight" not in sd
    assert isinstance(c.torso[0][1], torch.nn.ReLU)
    re = build_critic_from_state_dict(sd)
    o, p = _x()
    assert torch.equal(re(o, p), c(o, p))


@pytest.mark.parametrize("act", ["silu", "gelu"])
@pytest.mark.parametrize("in_norm", [False, True])
@pytest.mark.parametrize("v_raw", [False, True])
def test_new_choices_rebuild_from_the_state_dict(act, in_norm, v_raw):
    torch.manual_seed(1)
    c = CentralCritic(obs_dim=OBS, hidden_dim=16, num_blocks=2, q_actions=3,
                      torso_layernorm=True, value_bins=51, act=act, in_norm=in_norm,
                      v_raw=v_raw)
    sd = c.state_dict()
    assert "_arch" in sd
    re = build_critic_from_state_dict(sd)
    assert (re.act_name, re.in_norm, re.v_raw) == (act, in_norm, v_raw)
    o, p = _x()
    assert torch.equal(re(o, p), c(o, p))
    v1, q1 = re.q_values(o, p)
    v2, q2 = c.q_values(o, p)
    assert torch.equal(v1, v2) and torch.equal(q1, q2)


def test_v_raw_is_the_mean_of_the_predicted_distribution():
    torch.manual_seed(2)
    c = CentralCritic(obs_dim=OBS, hidden_dim=16, num_blocks=1, q_actions=3,
                      torso_layernorm=True, value_bins=51, v_raw=True, act="silu")
    with torch.no_grad():
        c.value_head.weight.normal_(0, 0.5)
    o, p = _x(8)
    z = c.torso(torch.cat([o, p], -1))
    probs = F.softmax(c.value_head(z), -1)
    want = (probs * _symexp(c._value_centers)).sum(-1)
    assert torch.allclose(c(o, p), want, atol=1e-5)
    v, q = c.q_values(o, p)
    assert torch.allclose(q[:, 0], v, atol=1e-6)  # zero-init adv_head: Q == V
    legacy = _symexp((probs * c._value_centers).sum(-1))
    assert not torch.allclose(want, legacy)  # the two readouts differ


def test_q_fold_zero_pins_the_fold_column():
    torch.manual_seed(3)
    c = CentralCritic(obs_dim=OBS, hidden_dim=16, num_blocks=1, q_actions=3,
                      torso_layernorm=True, value_bins=51, v_raw=True, q_fold_zero=True)
    with torch.no_grad():
        c.adv_head.weight.normal_()
    o, p = _x(6)
    _v, q = c.q_values(o, p)
    assert torch.equal(q[:, 0], torch.zeros(6)) and q[:, 1:].abs().sum() > 0


def _setup(obs_mode="minimal", **tc_kw):
    from plo5bp.encoding import OBS_DIM_MINIMAL

    torch.manual_seed(11)
    model = ActorCriticV2(hidden_dim=32, obs_dim=OBS_DIM_MINIMAL)
    critic = CentralCritic(obs_dim=OBS_DIM_MINIMAL, hidden_dim=32, num_blocks=1, act="silu",
                           in_norm=True)
    tc = TrainingConfig(num_envs=8, rollout_length=200, hidden_dim=32, obs_mode=obs_mode,
                        ppo_epochs=2, batch_size=48, **tc_kw)
    rng = np.random.default_rng(5)
    cfg = GameConfig(num_seats=4, starting_stack=60 * BB, ante=3 * BB, bb=BB)
    batch = collect_rollout_batched(model, OpponentPool(capacity=1), cfg, tc, rng, critic=critic)
    return model, critic, tc, batch


def _clone(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def test_critic_only_passes_never_touch_the_actor():
    model, critic, tc, batch = _setup(critic_extra_epochs=2, critic_minibatches=7)
    t = PPOTrainer(model, tc, critic=critic)
    a0, c0 = _clone(model), _clone(critic)
    stats = t.update(batch, np.random.default_rng(1), actor_frozen=True)
    for k, v in model.state_dict().items():
        assert torch.equal(v, a0[k]), f"actor {k} changed while frozen"
    assert any(not torch.equal(v, c0[k]) for k, v in critic.state_dict().items())
    for p in t._actor_params:
        assert not t.optimizer.state.get(p), "an actor Adam state was created while frozen"
    assert stats.value_loss > 0 and stats.policy_loss == 0.0 and stats.approx_kl == 0.0


def test_extra_epochs_after_ppo_leave_the_actor_where_ppo_put_it():
    model, critic, tc, batch = _setup(critic_extra_epochs=1, critic_minibatches=5)
    model2, critic2 = copy.deepcopy(model), copy.deepcopy(critic)
    tc0 = dataclasses.replace(tc, critic_extra_epochs=0)
    t_extra = PPOTrainer(model, tc, critic=critic)
    t_plain = PPOTrainer(model2, tc0, critic=critic2)
    t_extra.update(batch, np.random.default_rng(3))
    t_plain.update(batch, np.random.default_rng(3))
    for k, v in model.state_dict().items():
        assert torch.equal(v, model2.state_dict()[k]), f"actor {k} differs"
    assert any(not torch.equal(v, critic2.state_dict()[k]) for k, v in critic.state_dict().items())


def test_q_norm_scales_by_the_return_variance_and_is_off_by_default():
    model, critic, tc, batch = _setup()
    t = PPOTrainer(model, tc, critic=critic)
    assert t._q_norm(batch) == 1.0
    tc2 = dataclasses.replace(tc, critic_q_norm=True)
    t2 = PPOTrainer(model, tc2, critic=critic)
    want = 1.0 / (batch.returns.float().var() + 1.0)
    assert torch.isclose(torch.as_tensor(t2._q_norm(batch)), want)


def test_actor_only_moment_restore():
    model, critic, tc, batch = _setup()
    t = PPOTrainer(model, tc, critic=critic)
    t.update(batch, np.random.default_rng(1))
    side = t.optimizer_sidecar_state()
    # A trainer with a DIFFERENT critic (new width): refused unless allowed.
    new_critic = CentralCritic(obs_dim=critic.obs_dim, hidden_dim=48, num_blocks=1)
    t2 = PPOTrainer(copy.deepcopy(model), tc, critic=new_critic)
    ok, why = t2.load_optimizer_moments(side)
    assert not ok and "shapes differ" in why
    ok, why = t2.load_optimizer_moments(side, allow_actor_only=True)
    assert ok and "ACTOR ONLY" in why
    for p, p_old in zip(t2._actor_params, t._actor_params):
        m_new = t2.optimizer.state[p]["exp_avg"]
        m_old = t.optimizer.state[p_old]["exp_avg"]
        assert torch.equal(m_new, m_old)
    for p in t2._critic_params:
        assert not t2.optimizer.state.get(p), "the new critic must start cold"
