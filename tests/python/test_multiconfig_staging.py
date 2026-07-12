"""P7+P8 shared-staging bit-exactness (2026-07-09).

`collect_rollout_multiconfig` now stages sub-rollouts by writing each one
directly into per-sub views of ONE preallocated host buffer (bases chained by
actual row counts), replacing the legacy pipeline (finalize each sub to the
learner device -> evacuate to host -> torch.cat -> upload). The legacy path is
retained behind `_legacy_staging=True` purely as the reference here.

These tests run the SAME collection twice — identical torch/np seeding, same
model instance — through both paths and assert every Batch tensor field,
scalar diagnostic, `ent_coef_rows`, and `tier_ftr` is bitwise identical. Any
base-chaining off-by-one, view aliasing mistake, dtype drift, or divergence in
the per-sub/global advantage normalization shows up as a hard mismatch.
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.network import ActorCriticV2
from plo5bp.rollout import _BATCH_TENSOR_FIELDS, Batch, collect_rollout_multiconfig
from plo5bp.selfplay import OpponentPool


def _collect(
    model: ActorCriticV2,
    configs: list[GameConfig],
    tiers: "list[str] | None",
    tier_ent: "dict[str, float] | None",
    legacy: bool,
) -> Batch:
    """One deterministic multiconfig collection. Both RNG sources are re-seeded
    per call (act() samples from the GLOBAL torch RNG; env seeds / pool-mix
    draws come from the passed numpy generator), so two calls with the same
    model produce identical trajectories — isolating the staging under test."""
    train_cfg = TrainingConfig(num_envs=6, rollout_length=120, hidden_dim=32)
    torch.manual_seed(123)
    rng = np.random.default_rng(7)
    pool = OpponentPool(capacity=1)  # empty -> pure self-play
    return collect_rollout_multiconfig(
        model,
        pool,
        configs,
        train_cfg,
        rng,
        critic=None,
        config_tiers=tiers,
        tier_ent=tier_ent,
        _legacy_staging=legacy,
    )


def _assert_bit_identical(a: Batch, b: Batch) -> None:
    for f in _BATCH_TENSOR_FIELDS:
        ta, tb = getattr(a, f), getattr(b, f)
        assert ta.dtype == tb.dtype, f"{f}: dtype {ta.dtype} vs {tb.dtype}"
        assert ta.shape == tb.shape, f"{f}: shape {ta.shape} vs {tb.shape}"
        assert torch.equal(ta, tb), f"{f}: values differ"
    assert a.aggr_bonus_total_bb == b.aggr_bonus_total_bb
    assert a.aggr_steps_total == b.aggr_steps_total
    assert a.aggr_bonus_steps == b.aggr_bonus_steps
    assert a.aggr_steps_total_by_street == b.aggr_steps_total_by_street
    assert a.aggr_bonus_steps_by_street == b.aggr_bonus_steps_by_street
    ea, eb = a.ent_coef_rows, b.ent_coef_rows
    assert (ea is None) == (eb is None), "ent_coef_rows presence differs"
    if ea is not None:
        assert ea.dtype == eb.dtype and ea.shape == eb.shape
        assert torch.equal(ea, eb), "ent_coef_rows values differ"
    ta, tb = a.is_terminal, b.is_terminal
    assert (ta is None) == (tb is None), "is_terminal presence differs"
    if ta is not None:
        assert ta.dtype == tb.dtype and ta.shape == tb.shape
        assert torch.equal(ta, tb), "is_terminal values differ"
    assert getattr(a, "tier_ftr", None) == getattr(b, "tier_ftr", None)


def test_shared_staging_bit_identical_to_legacy_multi_config() -> None:
    torch.manual_seed(0)
    model = ActorCriticV2(hidden_dim=32)
    model.eval()
    configs = [GameConfig(num_seats=3), GameConfig(num_seats=5)]
    tiers = ["clubgg", "deep"]
    tier_ent = {"clubgg": 0.10, "deep": 0.18}

    legacy = _collect(model, configs, tiers, tier_ent, legacy=True)
    shared = _collect(model, configs, tiers, tier_ent, legacy=False)
    _assert_bit_identical(legacy, shared)


def test_shared_staging_bit_identical_single_config() -> None:
    # n == 1 parity: _concat_batches has a len==1 passthrough that SKIPS the
    # global advantage re-normalization — the shared path must skip it too.
    torch.manual_seed(0)
    model = ActorCriticV2(hidden_dim=32)
    model.eval()
    configs = [GameConfig(num_seats=4)]

    legacy = _collect(model, configs, None, None, legacy=True)
    shared = _collect(model, configs, None, None, legacy=False)
    _assert_bit_identical(legacy, shared)


def test_shared_staging_bit_identical_with_pool_opponents() -> None:
    # P5 + P7/P8 composition: a non-empty opponent pool exercises the
    # per-update snapshot cache (one dict shared across sub-rollouts) and the
    # opponent-forward path inside BOTH staging modes. Building the frozen
    # model consumes the CPU torch RNG identically in the two runs (fresh
    # seeding each), so run-vs-run bit-equality still isolates the staging.
    torch.manual_seed(0)
    model = ActorCriticV2(hidden_dim=32)
    model.eval()
    torch.manual_seed(1)
    frozen = ActorCriticV2(hidden_dim=32)  # distinct frozen opponent weights
    frozen.eval()

    def collect(legacy: bool):
        train_cfg = TrainingConfig(num_envs=6, rollout_length=120, hidden_dim=32)
        torch.manual_seed(123)
        rng = np.random.default_rng(7)
        pool = OpponentPool(capacity=2)
        pool.snapshot(frozen, tag=0)
        return collect_rollout_multiconfig(
            model,
            pool,
            [GameConfig(num_seats=3), GameConfig(num_seats=5)],
            train_cfg,
            rng,
            critic=None,
            config_tiers=["clubgg", "deep"],
            tier_ent={"clubgg": 0.10, "deep": 0.18},
            _legacy_staging=legacy,
        )

    _assert_bit_identical(collect(True), collect(False))


def test_shared_staging_three_configs_row_alignment() -> None:
    # Three sub-rollouts of different seat counts: exercises two base-chain
    # joints (the alignment failure mode) plus tier bookkeeping with a
    # repeated tier label.
    torch.manual_seed(0)
    model = ActorCriticV2(hidden_dim=32)
    model.eval()
    configs = [
        GameConfig(num_seats=3),
        GameConfig(num_seats=4),
        GameConfig(num_seats=6),
    ]
    tiers = ["clubgg", "deep", "clubgg"]
    tier_ent = {"clubgg": 0.10, "deep": 0.18}

    legacy = _collect(model, configs, tiers, tier_ent, legacy=True)
    shared = _collect(model, configs, tiers, tier_ent, legacy=False)
    _assert_bit_identical(legacy, shared)
