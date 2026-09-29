"""Regression tests for the 2026-09-20 code review, training workstream —
ppo.py items:

- A8  a non-finite KL (or loss) is a hard trip: the step is never applied.
- A3  Adam moments round-trip through the optimizer sidecar state.
- A14 the l2-init reference tensors round-trip too (decay toward the
      ORIGINAL init across relaunches, not the relaunch point).
- A20 value_clip <= 0 disables value clipping (it used to freeze the critic).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from plo5bp.config import TrainingConfig
from plo5bp.network import ActorCriticV5, CentralCritic, opp_holes_multihot
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import Batch

OBS, H, N = 40, 32, 1024


def _make(seed=0, **cfg_kw):
    torch.manual_seed(seed)
    model = ActorCriticV5(hidden_dim=H, obs_dim=OBS, num_layers=3)
    critic = CentralCritic(obs_dim=OBS, hidden_dim=H, num_blocks=1, q_actions=13)
    cfg = TrainingConfig(
        hidden_dim=H, num_layers=3, batch_size=256, ppo_epochs=2, **cfg_kw
    )
    return model, critic, cfg


def _batch(model, critic, seed=0, nan_obs=False, nan_return=False):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(N, OBS, generator=g)
    sizing = torch.tensor([[10_000, 180_000, 180_000, 0]] * N, dtype=torch.int64)
    gm = torch.ones(N, 3, dtype=torch.bool)
    holes = torch.randint(0, 52, (N, 5, 5), generator=g).to(torch.uint8)
    with torch.no_grad():
        out = model.act(obs, gm, sizing)
        v = critic(obs, opp_holes_multihot(holes))
    ret = v + torch.randn(N, generator=g) * 5
    adv = torch.randn(N, generator=g)
    if nan_obs:
        obs = obs.clone()
        obs[7, 3] = float("nan")  # one bad row (e.g. an encoder NaN)
    if nan_return:
        ret = ret.clone()
        ret[11] = float("nan")
    return Batch(
        obs=obs, gate_masks=gm, gate_actions=out.gate, raise_chips=out.chips,
        sizing=sizing, anchor_actions=out.anchor, refine_u=out.refine_u,
        opp_holes=holes, log_probs=out.log_prob, values=v, returns=ret,
        advantages=adv, old_gate_logp=out.gate_log_prob,
        old_anchor_logp=out.anchor_log_prob,
        is_terminal=torch.zeros(N, dtype=torch.bool),
    )


def _flat(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


@pytest.fixture
def no_dist_validation():
    """scripts/train.py turns torch.distributions argument validation OFF
    process-wide (P13), so in a real run a NaN logit is NOT caught at the
    Categorical constructor — it reaches the KL guard. Mirror that here."""
    prev = torch.distributions.Distribution._validate_args
    torch.distributions.Distribution.set_default_validate_args(False)
    yield
    torch.distributions.Distribution.set_default_validate_args(prev)


# --------------------------------------------------------------------- A8
@pytest.mark.parametrize("poison", ["nan_obs", "nan_return"])
@pytest.mark.parametrize(
    "guards", [dict(target_kl=0.5, kl_hard=10.0), dict(target_kl=0.0, kl_hard=0.0)]
)
def test_a8_nonfinite_minibatch_never_steps(poison, guards, no_dist_validation):
    model, critic, cfg = _make(**guards)
    tr = PPOTrainer(model, cfg, critic=critic)
    # Warm Adam so a rollback has real moments to restore.
    tr.update(_batch(model, critic, seed=1), np.random.default_rng(1))
    before = _flat(tr._all_params).clone()
    stats = tr.update(
        _batch(model, critic, seed=2, **{poison: True}), np.random.default_rng(2)
    )
    after = _flat(tr._all_params)
    assert torch.isfinite(after).all(), "a non-finite step was applied"
    assert stats.kl_stopped_at >= 0 and not math.isfinite(stats.kl_stop)
    if guards["kl_hard"] > 0.0:
        # snapshot available -> the WHOLE update is rolled back
        assert stats.rolled_back
        assert torch.equal(before, after)
    else:
        # no snapshot: refuse the bad step, keep the finite ones applied
        assert not stats.rolled_back
    # the run survives: the next clean update trains normally
    nxt = tr.update(_batch(model, critic, seed=3), np.random.default_rng(3))
    assert math.isfinite(nxt.policy_loss)
    # (a clean update may still soft-stop on a FINITE kl — never a nan one)
    assert nxt.kl_stopped_at == -1 or math.isfinite(nxt.kl_stop)
    assert torch.isfinite(_flat(tr._all_params)).all()


def test_a8_finite_update_is_unaffected():
    # Same seeds, guards on vs off: with no trip the guard only READS kl.
    outs = []
    for guards in (dict(target_kl=1e9, kl_hard=1e9), dict(target_kl=0.0, kl_hard=0.0)):
        model, critic, cfg = _make(**guards)
        tr = PPOTrainer(model, cfg, critic=critic)
        st = tr.update(_batch(model, critic, seed=4), np.random.default_rng(4))
        assert st.kl_stopped_at == -1
        outs.append(_flat(tr._all_params))
    assert torch.equal(outs[0], outs[1])


# ---------------------------------------------------------------- A3 / A14
def test_a3_optimizer_sidecar_roundtrip_resumes_identically():
    """Train 2 updates; snapshot weights + sidecar; then (a) keep training and
    (b) rebuild a FRESH trainer from the weights + sidecar and train the same
    update. Warm-resumed == uninterrupted, and a cold-Adam resume is not."""
    model, critic, cfg = _make(target_kl=0.0, kl_hard=0.0, l2_init_coef=1e-3)
    tr = PPOTrainer(model, cfg, critic=critic)
    for s in (1, 2):
        tr.update(_batch(model, critic, seed=s), np.random.default_rng(s))
    ckpt_model = copy.deepcopy(model.state_dict())
    ckpt_critic = copy.deepcopy(critic.state_dict())
    sidecar = copy.deepcopy(tr.optimizer_sidecar_state())
    nxt = _batch(model, critic, seed=3)
    tr.update(nxt, np.random.default_rng(3))
    uninterrupted = _flat(tr._all_params)

    def resume(load_sidecar: bool):
        m2, c2, _ = _make(seed=99, target_kl=0.0, kl_hard=0.0, l2_init_coef=1e-3)
        m2.load_state_dict(ckpt_model)
        c2.load_state_dict(ckpt_critic)
        tr2 = PPOTrainer(m2, cfg, critic=c2)
        if load_sidecar:
            ok, why = tr2.load_optimizer_moments(sidecar)
            assert ok, why
            ok, why = tr2.load_l2_init_refs(sidecar)
            assert ok, why
        tr2.update(nxt, np.random.default_rng(3))
        return _flat(tr2._all_params)

    assert torch.equal(resume(True), uninterrupted)
    assert not torch.equal(resume(False), uninterrupted)


def test_a3_sidecar_keeps_this_runs_hyperparameters():
    model, critic, cfg = _make(target_kl=0.0, kl_hard=0.0)
    tr = PPOTrainer(model, cfg, critic=critic)
    tr.update(_batch(model, critic, seed=1), np.random.default_rng(1))
    sidecar = tr.optimizer_sidecar_state()
    assert all(not t.is_cuda for st in sidecar["optimizer_state"].values()
               for t in st.values() if torch.is_tensor(t))
    m2, c2, _ = _make(seed=5)
    cfg2 = TrainingConfig(
        hidden_dim=H, num_layers=3, batch_size=256, ppo_epochs=2,
        lr=7e-5, adam_b2=0.98,
    )
    tr2 = PPOTrainer(m2, cfg2, critic=c2)
    ok, _why = tr2.load_optimizer_moments(sidecar)
    assert ok
    pg = tr2.optimizer.param_groups[0]
    assert pg["lr"] == 7e-5 and pg["betas"] == (0.9, 0.98)  # NOT the sidecar's
    assert len(tr2.optimizer.state) == len(sidecar["optimizer_state"])


def test_a3_sidecar_with_other_shapes_is_refused():
    model, critic, cfg = _make()
    tr = PPOTrainer(model, cfg, critic=critic)
    tr.update(_batch(model, critic, seed=1), np.random.default_rng(1))
    sidecar = tr.optimizer_sidecar_state()
    torch.manual_seed(0)
    other_critic = CentralCritic(  # pooled dueling head: 3 columns, not 13
        obs_dim=OBS, hidden_dim=H, num_blocks=1, q_actions=3
    )
    other = ActorCriticV5(hidden_dim=H, obs_dim=OBS, num_layers=3)
    tr2 = PPOTrainer(other, cfg, critic=other_critic)
    ok, why = tr2.load_optimizer_moments(sidecar)
    assert not ok and "shape" in why
    assert len(tr2.optimizer.state) == 0  # untouched -> Adam starts cold
    ok, why = tr2.load_optimizer_moments({})
    assert not ok


def test_a14_l2_init_refs_survive_a_relaunch():
    model, critic, cfg = _make(target_kl=0.0, kl_hard=0.0, l2_init_coef=1e-3)
    tr = PPOTrainer(model, cfg, critic=critic)
    init_refs = [p0.clone() for _p, p0 in tr._l2_init_pairs]
    assert init_refs, "trunk references expected with l2_init_coef > 0"
    for s in (1, 2, 3):
        tr.update(_batch(model, critic, seed=s), np.random.default_rng(s))
    sidecar = tr.optimizer_sidecar_state()

    # Relaunch: a new trainer built on the TRAINED weights re-snapshots the
    # relaunch point (the bug) ...
    tr2 = PPOTrainer(model, cfg, critic=critic)
    assert any(
        not torch.equal(p0, r) for (_p, p0), r in zip(tr2._l2_init_pairs, init_refs)
    )
    # ... until the persisted references are re-attached.
    ok, why = tr2.load_l2_init_refs(sidecar)
    assert ok, why
    for (_p, p0), r in zip(tr2._l2_init_pairs, init_refs):
        assert torch.equal(p0, r)

    # l2_init off -> nothing to restore (and nothing persisted)
    m3, c3, cfg3 = _make()
    tr3 = PPOTrainer(m3, cfg3, critic=c3)
    assert tr3.optimizer_sidecar_state()["l2_init"] == {}
    ok, _why = tr3.load_l2_init_refs(sidecar)
    assert not ok


# -------------------------------------------------------------------- A20
def test_a20_value_clip_zero_disables_clipping():
    moves = {}
    for vc in (1e-6, 0.0, -1.0, 1e9):
        model, critic, cfg = _make(target_kl=0.0, kl_hard=0.0, value_clip=vc)
        tr = PPOTrainer(model, cfg, critic=critic)
        c0 = _flat(critic.parameters()).clone()
        tr.update(_batch(model, critic, seed=6), np.random.default_rng(6))
        moves[vc] = _flat(critic.parameters()) - c0
    # 0 / negative == "no clipping" == an effectively infinite radius
    assert torch.equal(moves[0.0], moves[1e9])
    assert torch.equal(moves[-1.0], moves[1e9])
    # ... and NOT the old reading of 0 as a zero-width clip radius (which
    # behaves like this 1e-6 one: gradient only where the critic got worse).
    assert not torch.equal(moves[1e-6], moves[1e9])
