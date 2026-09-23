"""Batched opponent acting (`rollout._StackedOpponents`, 2026-09-23).

The batched collector acts every pool snapshot's opponent rows with ONE
stacked (vmapped) forward and ONE `_act_from_heads` sampling pass instead of
one `act()` per snapshot. Pinned here:

- `act()` is exactly `forward()` + `_act_from_heads()` (the learner's act is
  bit-identical after the split),
- the stacked forward gives every snapshot its own head outputs (up to float
  reassociation in the batched matmul),
- greedy (deterministic) actions equal each snapshot's own `act()`,
- pools the stack cannot hold fall back to per-snapshot calls,
- a real rollout with a populated pool goes through the stacked path only
  when `batched_opponents` is on, and produces a sane batch either way.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.network import ActorCritic, ActorCriticV2, ActorCriticV4, ActorCriticV5
from plo5bp.rollout import _StackedOpponents, collect_rollout_multiconfig
from plo5bp.selfplay import OpponentPool

BB = 10_000
D = OBS_DIM_MINIMAL
_ARCHS = [
    (ActorCriticV2, dict(num_layers=2)),
    (ActorCriticV4, dict(num_layers=3, torso_layernorm=True)),
    (ActorCriticV5, dict(num_layers=3, torso_layernorm=True)),
    (ActorCriticV5, dict(num_layers=2)),
]


def _snapshots(cls, kw, n=4, seed=0):
    torch.manual_seed(seed)
    models = []
    for _ in range(n):
        m = cls(hidden_dim=48, obs_dim=D, **kw).eval()
        with torch.no_grad():  # distinct, non-trivial snapshots (v5 heads init at zero)
            for p in m.parameters():
                p.add_(0.2 * torch.randn_like(p))
        models.append(m)
    return models


def _inputs(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    obs = (torch.rand(n, D, generator=g) < 0.1).float()
    obs[:, 176:188] = torch.rand(n, 12, generator=g) * 40  # stacks / pot-ish scalars
    gm = torch.ones(n, 3, dtype=torch.bool)
    gm[: n // 3, 0] = False  # nothing to fold to
    gm[n // 2 :, 2] = torch.rand(n - n // 2, generator=g) > 0.2
    pot = torch.randint(20, 400, (n,), generator=g) * BB
    sizing = torch.stack(
        [torch.full((n,), 2 * BB), pot * 3, pot, torch.zeros(n, dtype=torch.int64)], -1
    )
    return obs, gm, sizing


def _slots(n_models, n):
    g = torch.randint(0, n_models, (n,), generator=torch.Generator().manual_seed(2))
    g, _ = torch.sort(g)
    counts = torch.bincount(g, minlength=n_models)
    j = torch.arange(n) - (torch.cumsum(counts, 0) - counts)[g]
    return g, j, int(counts.max())


@pytest.mark.parametrize("cls,kw", _ARCHS)
def test_act_is_forward_plus_act_from_heads(cls, kw) -> None:
    (m,) = _snapshots(cls, kw, n=1)
    obs, gm, sizing = _inputs(200)
    torch.manual_seed(9)
    a = m.act(obs, gm, sizing, return_marginal=True)
    torch.manual_seed(9)
    b = m._act_from_heads(*m.forward(obs, gm), sizing, return_marginal=True)
    for x, y in zip(a, b):
        assert (x is None and y is None) or torch.equal(x, y)


@pytest.mark.parametrize("cls,kw", _ARCHS)
def test_stacked_forward_gives_each_snapshot_its_own_heads(cls, kw) -> None:
    models = _snapshots(cls, kw)
    stack = _StackedOpponents(models)
    obs, gm, _sizing = _inputs(300)
    g, j, n_max = _slots(len(models), 300)
    obs_pad = obs.new_zeros((len(models), n_max, D))
    obs_pad[g, j] = obs
    gm_pad = gm.new_zeros((len(models), n_max, 3))
    gm_pad[g, j] = gm
    with torch.inference_mode():
        heads = stack._vforward(stack.params, stack.buffers, obs_pad, gm_pad)
        for k, m in enumerate(models):
            rows = (g == k).nonzero().squeeze(-1)
            own = m.forward(obs[rows], gm[rows])
            for h_stack, h_own in zip(heads, own):
                torch.testing.assert_close(h_stack[k, j[rows]], h_own, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("cls,kw", _ARCHS)
def test_greedy_actions_equal_each_snapshots_own_act(cls, kw) -> None:
    models = _snapshots(cls, kw)
    stack = _StackedOpponents(models)
    obs, gm, sizing = _inputs(300)
    g, j, n_max = _slots(len(models), 300)
    with torch.inference_mode():
        out = stack.act(obs, gm, sizing, g, j, n_max, deterministic=True)
        for k, m in enumerate(models):
            rows = (g == k).nonzero().squeeze(-1)
            own = m.act(obs[rows], gm[rows], sizing[rows], deterministic=True)
            assert torch.equal(out.gate[rows], own.gate)
            assert torch.equal(out.anchor[rows], own.anchor)
            assert torch.equal(out.chips[rows], own.chips)
            torch.testing.assert_close(out.log_prob[rows], own.log_prob, rtol=1e-4, atol=1e-4)


def test_supported_only_for_stackable_pools() -> None:
    v5 = _snapshots(ActorCriticV5, dict(num_layers=3, torso_layernorm=True), n=2)
    assert _StackedOpponents.supported(v5)
    wider = ActorCriticV5(hidden_dim=64, obs_dim=D, num_layers=3, torso_layernorm=True)
    assert not _StackedOpponents.supported(v5 + [wider])  # shape mismatch
    v4 = _snapshots(ActorCriticV4, dict(num_layers=3, torso_layernorm=True), n=1)
    assert not _StackedOpponents.supported(v5 + v4)  # mixed classes
    assert not _StackedOpponents.supported([ActorCritic(hidden_dim=32)])  # v1: no split
    assert not _StackedOpponents.supported([])


@pytest.mark.parametrize("batched", [True, False])
def test_rollout_with_a_pool_uses_the_stack_only_when_enabled(monkeypatch, batched) -> None:
    calls = {"n": 0}
    real_act = _StackedOpponents.act

    def counting_act(self, *a, **k):
        calls["n"] += 1
        return real_act(self, *a, **k)

    monkeypatch.setattr(_StackedOpponents, "act", counting_act)
    torch.manual_seed(0)
    learner = ActorCriticV5(hidden_dim=32, obs_dim=D, num_layers=3, torso_layernorm=True)
    pool = OpponentPool(capacity=4)
    for k in range(3):  # three distinct past snapshots
        snap = ActorCriticV5(hidden_dim=32, obs_dim=D, num_layers=3, torso_layernorm=True)
        with torch.no_grad():
            for p in snap.parameters():
                p.add_(0.1 * (k + 1) * torch.randn_like(p))
        pool.snapshot(snap)
    tc = TrainingConfig(
        num_envs=12, rollout_length=240, hidden_dim=32, num_layers=3,
        obs_mode="minimal", pool_mix_prob=1.0, pool_opp_seats=2,
        batched_opponents=batched,
    )
    cfgs = [
        GameConfig(num_seats=6, starting_stack=60 * BB, ante=3 * BB, bb=BB),
        GameConfig(num_seats=4, starting_stack=150 * BB, ante=3 * BB, bb=BB),
    ]
    batch = collect_rollout_multiconfig(learner, pool, cfgs, tc, np.random.default_rng(4))
    assert (calls["n"] > 0) == batched
    assert batch.obs.shape[0] >= tc.rollout_length
    assert torch.isfinite(batch.advantages).all() and torch.isfinite(batch.log_probs).all()
