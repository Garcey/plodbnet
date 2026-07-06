"""v2 rollout invariants (anchor sizing head + centralized critic).

The highest-value check of the v2 plumbing: re-running `evaluate` on a
collected batch must reproduce the STORED log-probs (PPO ratio == 1.0
at epoch start) for BOTH collectors. act() samples (gate, anchor, u)
and evaluate() replays them — any drift between the sizing context
built at act time and the one stored in the batch, or any chips-based
reverse-engineering, breaks this exactness.

Also pinned here:
  - stored anchors are legal under masks recomputed from the stored
    sizing context;
  - opp_holes blocks carry valid card indices with 255 padding in the
    slots beyond num_seats - 1;
  - with a CentralCritic, the stored `values` are reproducible from
    the stored (obs, opp_holes) pair.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.actions import GATE_RAISE
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.network import ActorCriticV2, CentralCritic, opp_holes_multihot
from plo5bp.rollout import collect_rollout, collect_rollout_batched
from plo5bp.selfplay import OpponentPool
from plo5bp.sizing import anchor_grid_torch

NUM_SEATS = 4


def _collect(collector, seed: int, critic: CentralCritic | None = None):
    torch.manual_seed(seed)
    game_cfg = GameConfig(num_seats=NUM_SEATS)
    train_cfg = TrainingConfig(
        num_envs=4,
        rollout_length=128,
        hidden_dim=32,
    )
    model = ActorCriticV2(hidden_dim=train_cfg.hidden_dim)
    model.eval()
    pool = OpponentPool(capacity=1)  # empty → pure self-play
    rng = np.random.default_rng(seed)
    batch = collector(model, pool, game_cfg, train_cfg, rng, critic=critic)
    return model, batch


@pytest.mark.parametrize(
    "collector", [collect_rollout, collect_rollout_batched]
)
def test_evaluate_reproduces_stored_logprobs(collector) -> None:
    model, batch = _collect(collector, seed=0)
    with torch.no_grad():
        lp, ent, _val, gh, ah, bh, _glp, _alp, *_raw = model.evaluate(
            batch.obs,
            batch.gate_masks,
            batch.sizing,
            batch.gate_actions,
            batch.anchor_actions,
            batch.refine_u,
        )
    # Collection forwards ran in per-step groups (different gemm batch
    # sizes) so allow f32 reduction-order noise; real replay bugs are
    # orders of magnitude larger.
    assert torch.allclose(lp, batch.log_probs, atol=1e-4), (
        f"max |Δlog_prob| = {float((lp - batch.log_probs).abs().max())}"
    )
    for t in (ent, gh, ah, bh):
        assert torch.isfinite(t).all()


@pytest.mark.parametrize(
    "collector", [collect_rollout, collect_rollout_batched]
)
def test_stored_anchors_legal_under_recomputed_masks(collector) -> None:
    _model, batch = _collect(collector, seed=1)
    raise_rows = batch.gate_actions == GATE_RAISE
    assert raise_rows.any(), "rollout produced no Raise rows"
    grid = anchor_grid_torch(batch.sizing)
    anchors = batch.anchor_actions
    assert (anchors >= 0).all() and (anchors <= 10).all()
    picked_legal = grid.legal.gather(-1, anchors[..., None]).squeeze(-1)
    assert picked_legal[raise_rows].all()

    # Raise chips inside [min_raise, max_raise]; non-raise rows carry 0.
    sz = batch.sizing.numpy()
    rc = batch.raise_chips.numpy()
    rr = raise_rows.numpy()
    assert (rc[rr] >= sz[rr, 0]).all() and (rc[rr] <= sz[rr, 1]).all()
    assert (rc[~rr] == 0).all()


@pytest.mark.parametrize(
    "collector", [collect_rollout, collect_rollout_batched]
)
def test_opp_holes_block_layout(collector) -> None:
    _model, batch = _collect(collector, seed=2)
    oh = batch.opp_holes.numpy()
    assert oh.shape[1:] == (5, 5)
    assert ((oh < 52) | (oh == 255)).all()
    # 4-seat game → 3 opponents; slots 3, 4 are padding.
    assert (oh[:, NUM_SEATS - 1 :, :] == 255).all()
    assert (oh[:, : NUM_SEATS - 1, :] < 52).all()


@pytest.mark.parametrize(
    "collector", [collect_rollout, collect_rollout_batched]
)
def test_critic_values_reproducible_from_stored_inputs(collector) -> None:
    torch.manual_seed(99)
    critic = CentralCritic(hidden_dim=64, num_blocks=1)
    critic.eval()
    _model, batch = _collect(collector, seed=3, critic=critic)
    with torch.inference_mode():
        v = critic(batch.obs, opp_holes_multihot(batch.opp_holes))
    assert torch.allclose(v, batch.values, atol=1e-4), (
        f"max |Δvalue| = {float((v - batch.values).abs().max())}"
    )


def test_v2_pool_snapshot_path() -> None:
    """_build_frozen_model must rebuild pool snapshots as the learner's
    class — a hardcoded v1 ActorCritic would fail at load_state_dict."""
    torch.manual_seed(5)
    game_cfg = GameConfig(num_seats=6)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=128,
        hidden_dim=32,
        pool_mix_prob=1.0,
        pool_opp_seats=2,
    )
    learner = ActorCriticV2(hidden_dim=train_cfg.hidden_dim)
    learner.eval()
    pool = OpponentPool(capacity=2)
    pool.snapshot(ActorCriticV2(hidden_dim=train_cfg.hidden_dim))
    rng = np.random.default_rng(5)
    batch = collect_rollout_batched(learner, pool, game_cfg, train_cfg, rng)
    assert batch.obs.shape[0] >= train_cfg.rollout_length
    assert torch.isfinite(batch.log_probs).all()
