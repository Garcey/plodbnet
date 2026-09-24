"""PPO micro-batching (TrainingConfig.micro_batch_rows, 2026-09-23).

Splitting each minibatch into chunks and accumulating gradients (per-row means
weighted by each chunk's share of rows; the fold-supervision term keeps the
minibatch-wide denominator; L2-to-init and the KL anchor weighted to sum
once) must give the same update as whole minibatches -- up to float
summation order -- and exactly the old path when it is off or when a
minibatch already fits. Exercised with the v6-style terms: distributional
critic with the dueling Q head, fold supervision, L2-to-init, per-tier
entropy rows.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.network import ActorCriticV5, CentralCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout_multiconfig
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
        num_envs=48, rollout_length=1500, obs_mode="minimal", advantage_estimator="vrpo"
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
    assert batch.ent_coef_rows is not None
    return learner, critic, batch


def _update(rollout, micro: int):
    learner, critic, batch = rollout
    m, c = copy.deepcopy(learner), copy.deepcopy(critic)
    n = int(batch.obs.shape[0])
    tc = TrainingConfig(
        num_envs=48, rollout_length=1500, obs_mode="minimal", ppo_epochs=2,
        batch_size=(n + 2) // 3, q_aux_coef=0.5, q_fold_sup_coef=15.0,
        l2_init_coef=1e-4, target_kl=0.0, kl_hard=0.0, micro_batch_rows=micro,
    )
    trainer = PPOTrainer(m, tc, critic=c)
    torch.manual_seed(11)
    stats = trainer.update(batch, np.random.default_rng(12))
    return stats, m.state_dict(), c.state_dict(), (n + 2) // 3


def test_micro_batched_update_matches_whole_minibatches(rollout) -> None:
    s0, m0, c0, mb_rows = _update(rollout, 0)
    s1, m1, c1, _ = _update(rollout, 97)  # several uneven chunks per minibatch
    assert mb_rows > 97 * 3
    for sd0, sd1 in ((m0, m1), (c0, c1)):
        for k in sd0:
            torch.testing.assert_close(sd1[k], sd0[k], rtol=1e-4, atol=1e-6, msg=k)
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "display_loss",
              "q_loss", "gate_entropy", "anchor_entropy", "beta_entropy",
              "q_fold_err", "q_term_err"):
        a, b = getattr(s0, k), getattr(s1, k)
        assert b == pytest.approx(a, rel=1e-3, abs=1e-6), k


def test_off_or_fitting_minibatch_is_the_exact_old_path(rollout) -> None:
    s0, m0, c0, mb_rows = _update(rollout, 0)
    s1, m1, c1, _ = _update(rollout, mb_rows + 5)  # every minibatch fits
    for sd0, sd1 in ((m0, m1), (c0, c1)):
        for k in sd0:
            assert torch.equal(sd0[k], sd1[k]), k
    assert vars(s0) == vars(s1)
