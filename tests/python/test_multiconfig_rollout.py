"""vThree multi-config rollout: per-update mixing of N (seats,stacks) configs.

Covers the new logic added on top of the bit-exact single-config collector:
  - `_concat_batches` concatenates sub-rollouts field-by-field, re-normalizes
    the combined advantages GLOBALLY (mean 0 / std 1, then the fat-tail clamp),
    and sums the scalar aggression diagnostics;
  - `collect_rollout_multiconfig` runs end-to-end over several configs (varying
    seat counts) and returns one finite, normalized batch on the learner device.
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.actions import GATE_ACTIONS
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.network import ActorCriticV2
from plo5bp.rollout import Batch, _concat_batches, collect_rollout_multiconfig
from plo5bp.selfplay import OpponentPool


def _fake_batch(t: int, seed: int) -> Batch:
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.rand(*s, generator=g)
    return Batch(
        obs=rnd(t, OBS_DIM),
        gate_masks=torch.ones(t, GATE_ACTIONS, dtype=torch.bool),
        gate_actions=torch.randint(0, GATE_ACTIONS, (t,), generator=g),
        raise_chips=torch.randint(0, 100, (t,), generator=g),
        sizing=torch.randint(0, 100, (t, 4), generator=g),
        anchor_actions=torch.randint(0, 11, (t,), generator=g),
        refine_u=rnd(t),
        opp_holes=torch.randint(0, 52, (t, 5, 5), dtype=torch.uint8, generator=g),
        log_probs=rnd(t) - 1.0,
        values=rnd(t),
        returns=rnd(t),
        advantages=rnd(t) * 3.0,  # non-unit scale to exercise the re-norm
        old_gate_logp=rnd(t) - 1.0,
        old_anchor_logp=rnd(t) - 1.0,
        aggr_bonus_total_bb=float(seed),
        aggr_steps_total=t,
        aggr_bonus_steps=t // 2,
        aggr_steps_total_by_street=(t, t, t),
        aggr_bonus_steps_by_street=(1, 2, 3),
    )


def test_concat_batches_shapes_fields_and_per_config_norm() -> None:
    b1, b2 = _fake_batch(4, 0), _fake_batch(6, 1)
    out = _concat_batches([b1, b2], adv_clip=8.0)

    assert out.obs.shape == (10, OBS_DIM)
    assert torch.equal(out.obs, torch.cat([b1.obs, b2.obs], 0))
    assert torch.equal(out.gate_actions, torch.cat([b1.gate_actions, b2.gate_actions], 0))
    assert torch.equal(out.opp_holes, torch.cat([b1.opp_holes, b2.opp_holes], 0))

    # Advantages PRESERVE each sub-rollout's own (per-config) scale — NO global
    # re-normalization (which mixed incomparable stack-depth scales and drove the
    # full-LR collapse). The combined is exactly the concatenation, here all
    # within the fat-tail clamp.
    assert torch.equal(
        out.advantages, torch.cat([b1.advantages, b2.advantages], 0)
    )

    # Scalar diagnostics sum.
    assert out.aggr_steps_total == b1.aggr_steps_total + b2.aggr_steps_total
    assert out.aggr_bonus_steps == b1.aggr_bonus_steps + b2.aggr_bonus_steps
    assert out.aggr_steps_total_by_street == (10, 10, 10)
    assert out.aggr_bonus_steps_by_street == (2, 4, 6)
    assert abs(out.aggr_bonus_total_bb - 1.0) < 1e-9  # 0.0 + 1.0


def test_concat_single_batch_is_passthrough() -> None:
    b = _fake_batch(5, 7)
    assert _concat_batches([b], adv_clip=8.0) is b


def test_concat_advantages_fat_tail_clamped() -> None:
    b1, b2 = _fake_batch(50, 2), _fake_batch(50, 3)
    b1.advantages[0] = 1000.0  # outlier -> clamped by the fat-tail safety
    out = _concat_batches([b1, b2], adv_clip=8.0)
    assert out.advantages.max().item() <= 8.0 + 1e-5
    assert out.advantages.min().item() >= -8.0 - 1e-5


def test_collect_rollout_multiconfig_smoke() -> None:
    torch.manual_seed(0)
    train_cfg = TrainingConfig(num_envs=6, rollout_length=120, hidden_dim=32)
    model = ActorCriticV2(hidden_dim=train_cfg.hidden_dim)
    model.eval()
    pool = OpponentPool(capacity=1)  # empty -> pure self-play
    rng = np.random.default_rng(0)
    # Mixed seat counts within one update — the obs is padded/masked, so the
    # concatenation is shape-uniform across configs.
    configs = [GameConfig(num_seats=3), GameConfig(num_seats=4), GameConfig(num_seats=5)]

    batch = collect_rollout_multiconfig(model, pool, configs, train_cfg, rng, critic=None)

    assert batch.obs.shape[1] == OBS_DIM
    # Combined collected at least ~rollout_length transitions (3 sub-rollouts of
    # rollout_length//3 each, each overshooting its target slightly).
    assert batch.obs.shape[0] >= train_cfg.rollout_length - 5
    assert batch.obs.device.type == "cpu"
    for f in ("obs", "advantages", "returns", "values", "log_probs"):
        assert torch.isfinite(getattr(batch, f)).all(), f
    # Per-config normalized: each sub-rollout is mean-0 from its own finalize,
    # so the concatenation is ~mean-0 without any global re-normalization.
    assert abs(batch.advantages.mean().item()) < 1e-3
