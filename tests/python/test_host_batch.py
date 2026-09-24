"""Host-resident PPO batch (TrainingConfig.batch_on_host, 2026-09-24).

`HostBatchLoader.gather(sel)` must give exactly `gather_minibatch(batch, sel)`
-- every field, bit for bit, compact observations unpacked -- whatever rows
are asked for and however often the staging is reused; and with a CUDA
learner a whole PPO update from the host batch must equal the update from the
device-resident batch (skipped without CUDA; the pod's GPU digest covers the
training loop end to end).
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.network import ActorCriticV5, CentralCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import (
    HostBatchLoader,
    _batch_to_device,
    collect_rollout_multiconfig,
    gather_minibatch,
)
from plo5bp.selfplay import OpponentPool


@pytest.fixture(scope="module")
def rollout():
    torch.manual_seed(0)
    learner = ActorCriticV5(
        hidden_dim=16, obs_dim=OBS_DIM_MINIMAL, num_layers=3, torso_layernorm=True
    )
    critic = CentralCritic(
        obs_dim=OBS_DIM_MINIMAL, hidden_dim=16, num_blocks=1, q_actions=3, value_bins=51
    )
    cfg = TrainingConfig(
        num_envs=48, rollout_length=4000, obs_mode="minimal", advantage_estimator="vrpo"
    )
    configs = [
        GameConfig(num_seats=s, starting_stack=st * 10_000, ante=30_000, bb=10_000)
        for s, st in ((2, 40), (6, 20), (4, 120))
    ]
    tiers = ["clubgg", "deep", "clubgg_deep"]
    batch = collect_rollout_multiconfig(
        learner, OpponentPool(capacity=4, seed=0), configs, cfg,
        np.random.default_rng(5), critic=critic, config_tiers=tiers,
        tier_ent={"clubgg": 0.2, "deep": 0.3, "clubgg_deep": 0.25},
    )
    assert batch.ent_coef_rows is not None and batch.is_terminal is not None
    return learner, critic, batch


def _same(a, b) -> None:
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, torch.Tensor):
            assert x.dtype == y.dtype and x.shape == y.shape, f.name
            assert torch.equal(x, y), f.name
        elif x is None:
            assert y is None, f.name


@pytest.mark.parametrize("min_rows", [1, 10**9])  # engine gather / numpy gather
def test_loader_equals_gather_minibatch(rollout, monkeypatch, min_rows) -> None:
    from plo5bp import rollout as R

    monkeypatch.setattr(R, "_GATHER_RUST_MIN_ROWS", min_rows)
    _, _, batch = rollout
    n = int(batch.obs.shape[0])
    loader = HostBatchLoader(batch, torch.device("cpu"), capacity=n)
    rng = np.random.default_rng(7)
    for k in (n, n // 3, 1, 0):
        sel = torch.from_numpy(rng.permutation(n)[:k].astype(np.int64))
        _same(loader.gather(sel), gather_minibatch(batch, sel))
    # The fold-denominator input: gate-mask rows for a whole minibatch (any size).
    big = torch.from_numpy(rng.integers(0, n, size=3 * n).astype(np.int64))
    assert torch.equal(loader.gate_mask_rows(big), batch.gate_masks[big])
    with pytest.raises(ValueError, match="capacity"):
        HostBatchLoader(batch, torch.device("cpu"), capacity=3).gather(
            torch.arange(4)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA learner")
@pytest.mark.parametrize("micro", [0, 97])
def test_host_update_equals_device_update(rollout, micro) -> None:
    learner, critic, batch = rollout
    n = int(batch.obs.shape[0])
    results = []
    for host in (False, True):
        m = copy.deepcopy(learner).cuda()
        c = copy.deepcopy(critic).cuda()
        tc = TrainingConfig(
            num_envs=48, rollout_length=4000, obs_mode="minimal", ppo_epochs=2,
            batch_size=(n + 2) // 3, q_aux_coef=0.5, q_fold_sup_coef=15.0,
            l2_init_coef=1e-4, target_kl=0.0, kl_hard=0.0, micro_batch_rows=micro,
            batch_on_host=host,
        )
        b = batch if host else _batch_to_device(batch, torch.device("cuda"))
        trainer = PPOTrainer(m, tc, critic=c)
        torch.manual_seed(11)
        stats = trainer.update(b, np.random.default_rng(12))
        results.append((stats, m.state_dict(), c.state_dict()))
    (s0, m0, c0), (s1, m1, c1) = results
    for sd0, sd1 in ((m0, m1), (c0, c1)):
        for k in sd0:
            assert torch.equal(sd0[k], sd1[k]), k
    assert vars(s0) == vars(s1)
