"""PPO guards and per-update health numbers (2026-09-28).

- ML-001: a non-finite critic-only step is refused (weights stay finite).
- ML-010: a finite loss with a non-finite GRADIENT is a hard trip, and the
  pre-clip gradient norms are reported.
- ML-003 / ML-012 / ML-040: clip fraction, k3 KL, the pre-step KL (`kl0`,
  exactly the rollout-vs-PPO mismatch: ~0 on CPU/f32), true vs bonus entropy,
  explained variance / bias per street and tier (`value_health`).
- ML-011: with critic_q_norm, `critic_q_norm_minibatch` makes a micro-batched
  step the same gradient as the whole minibatch (PPO and critic-only passes).
- ML-041: the sync-free AGC equals the old per-tensor-sync version bit for bit.
- ML-047: the live-control surface of the trainer.
"""

from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np
import pytest
import torch

from plo5bp.compact_obs import PackedObs, as_dense
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.network import ActorCriticV5, CentralCritic
from plo5bp.ppo import PPOTrainer, _adaptive_grad_clip_, value_health
from plo5bp.rollout import collect_rollout_multiconfig
from plo5bp.selfplay import OpponentPool

BB = 10_000


@pytest.fixture(scope="module")
def rollout():
    torch.manual_seed(0)
    learner = ActorCriticV5(
        hidden_dim=16, obs_dim=OBS_DIM_MINIMAL, num_layers=3, torso_layernorm=True
    )
    critic = CentralCritic(
        obs_dim=OBS_DIM_MINIMAL, hidden_dim=16, num_blocks=1, q_actions=3,
        value_bins=51, act="silu", in_norm=True, v_raw=True, q_fold_zero=True,
    )
    cfg = TrainingConfig(
        num_envs=48, rollout_length=1500, obs_mode="minimal", advantage_estimator="vrpo"
    )
    configs = [
        GameConfig(num_seats=s, starting_stack=st * BB, ante=3 * BB, bb=BB)
        for s, st in ((2, 40), (6, 20), (4, 120))
    ]
    tiers = ["clubgg", "deep", "clubgg_deep"]
    batch = collect_rollout_multiconfig(
        learner, OpponentPool(capacity=4, seed=0), configs, cfg,
        np.random.default_rng(5), critic=critic, config_tiers=tiers,
        tier_ent={"clubgg": 0.2, "deep": 0.3, "clubgg_deep": 0.25},
    )
    return learner, critic, batch


def _tc(batch, **kw) -> TrainingConfig:
    n = int(batch.obs.shape[0])
    base = dict(
        num_envs=48, rollout_length=1500, obs_mode="minimal", ppo_epochs=2,
        batch_size=(n + 2) // 3, q_aux_coef=0.5, q_fold_sup_coef=15.0,
        target_kl=0.0, kl_hard=10.0,
    )
    base.update(kw)
    return TrainingConfig(**base)


def _run(rollout, batch=None, actor_frozen=False, **kw):
    learner, critic, b = rollout
    batch = b if batch is None else batch
    m, c = copy.deepcopy(learner), copy.deepcopy(critic)
    t = PPOTrainer(m, _tc(batch, **kw), critic=c)
    torch.manual_seed(11)
    stats = t.update(batch, np.random.default_rng(12), actor_frozen=actor_frozen)
    return stats, m, c, t


def _finite(module) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in module.parameters())


# ------------------------------------------------------------------ ML-001

def test_critic_only_nan_step_is_refused_and_weights_stay_finite(rollout):
    _, _, batch = rollout
    bad = dataclasses.replace(batch, returns=batch.returns.clone())
    bad.returns[7] = float("nan")  # one row: one minibatch per critic epoch
    stats, m, c, _ = _run(
        rollout, batch=bad, actor_frozen=True,
        critic_extra_epochs=1, critic_minibatches=6,
    )
    epochs = 2 + 1  # actor frozen: ppo_epochs + critic_extra_epochs
    assert stats.critic_skipped == epochs
    assert stats.critic_steps == epochs * 6 - epochs
    assert _finite(c), "a refused step still reached the critic"
    assert math.isfinite(stats.critic_value_loss)


def test_critic_only_passes_unchanged_when_finite(rollout):
    a, _, c_a, _ = _run(rollout, critic_extra_epochs=1, critic_minibatches=6)
    b, _, c_b, _ = _run(rollout, critic_extra_epochs=1, critic_minibatches=6)
    assert a.critic_skipped == 0 and a.critic_steps == 6
    for (k, x), y in zip(c_a.state_dict().items(), c_b.state_dict().values()):
        assert torch.equal(x, y), k


# ------------------------------------------------------------------ ML-010

def test_nonfinite_gradient_with_finite_loss_is_a_hard_trip(rollout):
    learner, critic, batch = rollout
    m, c = copy.deepcopy(learner), copy.deepcopy(critic)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    t = PPOTrainer(m, _tc(batch), critic=c)
    # A hook turns one gradient into inf; the loss itself stays finite.
    c.value_head.weight.register_hook(lambda g: g * float("inf"))
    stats = t.update(batch, np.random.default_rng(12))
    assert stats.nonfinite_grad and stats.kl_stopped_at == 0
    assert stats.rolled_back  # kl_hard > 0: the update is discarded
    assert math.isnan(stats.kl_stop)  # train.py's livelock alarm counts it
    assert _finite(m) and _finite(c)
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_gradient_norms_are_reported(rollout):
    stats, *_ = _run(rollout)
    assert stats.grad_norm_actor > 0 and stats.grad_norm_critic > 0
    assert stats.grad_norm_actor_max >= stats.grad_norm_actor
    assert 0.0 <= stats.grad_clip_actor <= 1.0 and 0.0 <= stats.grad_clip_critic <= 1.0
    assert stats.grad_norm_display > 0
    assert not stats.nonfinite_grad


# ---------------------------------------------------- ML-003 / -012 / -040

def test_health_numbers(rollout):
    stats, *_ = _run(rollout)
    assert 0.0 <= stats.clip_frac <= 1.0
    assert stats.kl_k3 >= 0.0
    # CPU, float32 obs: evaluate() reproduces the rollout's log-probs, so the
    # first minibatch (before any step) sees no mismatch.
    assert abs(stats.kl0) < 1e-5 and stats.ratio_dev0 < 1e-4
    # sizing scale 1.0: the bonus entropy IS the policy entropy
    assert stats.entropy == pytest.approx(stats.entropy_bonus, rel=1e-6)
    scaled, *_ = _run(rollout, sizing_entropy_scale=0.3)
    assert scaled.entropy > scaled.entropy_bonus  # the sizing part is down-weighted


def test_value_health_ev_and_bias(rollout):
    _, _, batch = rollout
    h = value_health(batch)
    assert h["all"]["n"] == int(batch.obs.shape[0])
    assert set(h["tier"]) == {"clubgg", "deep", "clubgg_deep"}
    assert sum(v["n"] for v in h["tier"].values()) == h["all"]["n"]
    assert sum(v["n"] for v in h["street"].values()) == h["all"]["n"]
    # A perfect critic reads EV 1 / bias 0; a constant offset moves only the bias.
    perfect = dataclasses.replace(batch, values=batch.returns.clone())
    hp = value_health(perfect)
    assert hp["all"]["ev"] == pytest.approx(1.0) and hp["all"]["bias"] == pytest.approx(0.0)
    shifted = dataclasses.replace(batch, values=batch.returns - 2.0)
    hs = value_health(shifted)
    assert hs["all"]["ev"] == pytest.approx(1.0, abs=1e-5)
    assert hs["all"]["bias"] == pytest.approx(-2.0, abs=1e-4)


def test_packed_column_equals_the_dense_column(rollout):
    _, _, batch = rollout
    assert isinstance(batch.obs, PackedObs)
    dense = as_dense(batch.obs)
    for col in (0, 156, 157, 158, 159, 176, 184, OBS_DIM_MINIMAL - 1):
        assert torch.equal(batch.obs.column(col), dense[:, col].float()), col


# ------------------------------------------------------------------ ML-011

def _crit(c):
    return torch.cat([p.detach().flatten() for p in c.parameters()])


@pytest.mark.parametrize("extra", [0, 1])
def test_q_norm_minibatch_makes_micro_batching_the_same_gradient(rollout, extra):
    kw = dict(critic_q_norm=True, critic_extra_epochs=extra, critic_minibatches=3)
    _, _, c_whole, _ = _run(rollout, **kw)
    n = int(rollout[2].obs.shape[0])
    micro = max(1, (n // 3) // 4)
    _, _, c_chunk, _ = _run(rollout, micro_batch_rows=micro, critic_q_norm_minibatch=True, **kw)
    _, _, c_legacy, _ = _run(rollout, micro_batch_rows=micro, **kw)
    whole = _crit(c_whole)
    err_scoped = (whole - _crit(c_chunk)).abs().max().item()
    err_legacy = (whole - _crit(c_legacy)).abs().max().item()
    assert err_scoped < 1e-5, err_scoped
    assert err_legacy > 10 * max(err_scoped, 1e-7), (err_legacy, err_scoped)


def test_q_norm_minibatch_is_a_no_op_without_micro_batching(rollout):
    kw = dict(critic_q_norm=True, critic_extra_epochs=1, critic_minibatches=3)
    _, _, a, _ = _run(rollout, **kw)
    _, _, b, _ = _run(rollout, critic_q_norm_minibatch=True, **kw)
    assert torch.equal(_crit(a), _crit(b))


# ------------------------------------------------------------------ ML-041

def _agc_old(params, clip: float, eps: float = 1e-3) -> None:
    with torch.no_grad():
        for p in params:
            g = p.grad
            if g is None:
                continue
            g_norm = g.detach().norm()
            max_norm = clip * p.detach().norm().clamp_min(eps)
            if float(g_norm) > float(max_norm):
                g.mul_(max_norm / g_norm.clamp_min(1e-12))


def test_sync_free_agc_is_bit_identical():
    torch.manual_seed(3)
    params = []
    for scale in (1e-3, 1.0, 50.0, 0.0):
        p = torch.nn.Parameter(torch.randn(7, 5))
        p.grad = torch.randn(7, 5) * scale
        params.append(p)
    p0 = torch.nn.Parameter(torch.zeros(3))  # a zero param: eps floor
    p0.grad = torch.randn(3)
    params.append(p0)
    twins = []
    for p in params:
        q = torch.nn.Parameter(p.detach().clone())
        q.grad = p.grad.clone()
        twins.append(q)
    _agc_old(params, 0.1)
    _adaptive_grad_clip_(twins, 0.1)
    for p, q in zip(params, twins):
        assert torch.equal(p.grad, q.grad)


# ------------------------------------------------------------------ ML-047

def test_live_control_surface(rollout):
    learner, critic, batch = rollout
    t = PPOTrainer(copy.deepcopy(learner), _tc(batch, clip_prob_dependent=True),
                   critic=copy.deepcopy(critic))
    changed = t.apply_live_control(clip_room_mid=0.07, target_kl=t.target_kl)
    assert changed == {"clip_room_mid": (0.05, 0.07)}
    assert t.live_value("clip_room_mid") == 0.07 and t._clip_room_mid == 0.07
    with pytest.raises(KeyError):
        t.apply_live_control(lr=1.0)
