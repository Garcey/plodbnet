"""Per-tier machinery under mix-configs (V5_DESIGN.md B5) + Q-aux.

Before this change the mixed update consumed exactly one entropy coef
(`tier_ent[mix_tiers[0]]`), per-tier control-file edits were silently
ignored, and per-tier F/T/R was unmeasurable (counters summed in the
concat). Now: `collect_rollout_multiconfig(config_tiers=, tier_ent=)`
attaches per-row ABSOLUTE coefs + per-tier F/T/R counters, minibatches
carry the rows, and ppo pays each transition its own tier's rate.
Also covers the critic's auxiliary Q regression (`--q-aux-coef`).
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.network import ActorCriticV5, CentralCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout_multiconfig, iter_minibatches
from plo5bp.selfplay import OpponentPool


def _tiny_cfg(**kw) -> TrainingConfig:
    base = dict(
        num_envs=8, rollout_length=96, hidden_dim=32, ppo_epochs=1,
        batch_size=64, critic_hidden_dim=64, critic_num_blocks=1,
    )
    base.update(kw)
    return TrainingConfig(**base)


def _collect_mix(model, critic, train_cfg, tier_ent):
    pool = OpponentPool(capacity=1, seed=0)
    rng = np.random.default_rng(0)
    configs = [
        GameConfig(num_seats=3, starting_stack=200_000),
        GameConfig(num_seats=4, starting_stack=400_000),
        GameConfig(num_seats=2, starting_stack=2_000_000),
    ]
    tiers = ["clubgg", "clubgg", "deep"]
    model.eval()
    batch = collect_rollout_multiconfig(
        model, pool, configs, train_cfg, rng, critic=critic,
        config_tiers=tiers, tier_ent=tier_ent,
    )
    model.train()
    return batch, rng


def test_multiconfig_attaches_per_row_coefs_and_tier_ftr() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    train_cfg = _tiny_cfg()
    model = ActorCriticV5(hidden_dim=32)
    critic = CentralCritic(hidden_dim=64, num_blocks=1)
    tier_ent = {"clubgg": 0.10, "deep": 0.45}
    batch, rng = _collect_mix(model, critic, train_cfg, tier_ent)

    assert batch.ent_coef_rows is not None
    assert batch.ent_coef_rows.shape[0] == batch.obs.shape[0]
    vals = set(np.unique(batch.ent_coef_rows.cpu().numpy()).tolist())
    assert vals <= {np.float32(0.10), np.float32(0.45)}
    assert len(vals) == 2  # both tiers actually contributed rows
    # Per-tier F/T/R counters exist and sum to the pooled totals.
    assert set(batch.tier_ftr) == {"clubgg", "deep"}
    pooled = [0, 0, 0]
    for _bonus, steps in batch.tier_ftr.values():
        for s in range(3):
            pooled[s] += steps[s]
    assert tuple(pooled) == tuple(batch.aggr_steps_total_by_street)
    # Minibatches carry the rows (shuffled but value-preserving).
    mb = next(iter_minibatches(batch, 64, rng))
    assert mb.ent_coef_rows is not None
    assert set(np.unique(mb.ent_coef_rows.cpu().numpy()).tolist()) <= vals


def test_ppo_update_consumes_per_row_coefs() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    train_cfg = _tiny_cfg()
    model = ActorCriticV5(hidden_dim=32)
    critic = CentralCritic(hidden_dim=64, num_blocks=1)
    batch, rng = _collect_mix(model, critic, train_cfg, {"clubgg": 0.1, "deep": 0.4})
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    stats = trainer.update(batch, rng)
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert np.isfinite(getattr(stats, name))


def test_q_aux_trains_the_dueling_head() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    train_cfg = _tiny_cfg(q_aux_coef=0.5)
    model = ActorCriticV5(hidden_dim=32)
    critic = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=13)
    batch, rng = _collect_mix(model, critic, train_cfg, {"clubgg": 0.1, "deep": 0.4})
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    assert trainer._critic_qv is not None
    before = critic.adv_head.weight.detach().clone()
    stats = trainer.update(batch, rng)
    assert np.isfinite(stats.q_loss) and stats.q_loss > 0.0
    assert not torch.equal(before, critic.adv_head.weight)


def test_q_aux_off_leaves_head_untouched() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    train_cfg = _tiny_cfg(q_aux_coef=0.0)
    model = ActorCriticV5(hidden_dim=32)
    critic = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=13)
    batch, rng = _collect_mix(model, critic, train_cfg, {"clubgg": 0.1, "deep": 0.4})
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    stats = trainer.update(batch, rng)
    assert stats.q_loss == 0.0
    assert float(critic.adv_head.weight.abs().max()) == 0.0  # still zero-init
