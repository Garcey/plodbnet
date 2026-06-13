"""ActorCriticV2 / CentralCritic invariants.

The load-bearing test is act↔evaluate log-prob parity: PPO's importance
ratio must be exactly 1.0 at epoch start, which requires evaluate() to
reproduce act()'s log-prob for the stored (gate, anchor, u) across every
sizing regime.
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.actions import GATE_RAISE
from plo5bp.network import (
    ActorCritic,
    ActorCriticV2,
    CentralCritic,
    opp_holes_multihot,
)
from plo5bp.sizing import anchor_grid_np

OBS_DIM = 991


def _model(seed=0):
    torch.manual_seed(seed)
    m = ActorCriticV2(hidden_dim=64, num_layers=2).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _random_batch(rng, n):
    obs = torch.from_numpy(
        rng.standard_normal((n, OBS_DIM)).astype(np.float32)
    )
    gate_mask = torch.from_numpy(
        np.stack(
            [rng.random(n) < 0.9, np.ones(n, bool), rng.random(n) < 0.95],
            axis=-1,
        )
    )
    # Ensure at least one legal gate per row.
    gate_mask[:, 1] = True

    # Sizing covering every regime: normal, stack-capped, min==max,
    # short-shove (min==0, max>0), to_call==0.
    pot = rng.integers(600, 20_000_000, size=n)
    tc = np.where(rng.random(n) < 0.3, 0, rng.integers(1, 2_000_000, size=n))
    pl_max = tc + pot + tc
    mx = np.where(rng.random(n) < 0.4,
                  rng.integers(1, np.maximum(pl_max, 2)),  # stack-capped
                  pl_max)
    mn = rng.integers(1, np.maximum(mx, 2))
    regime = rng.random(n)
    mn = np.where(regime < 0.15, 0, mn)            # short shove
    mn = np.where((regime >= 0.15) & (regime < 0.25), mx, mn)  # min==max
    sizing = torch.from_numpy(
        np.stack([mn, mx, pot, tc], axis=-1).astype(np.int64)
    )
    return obs, gate_mask, sizing


def test_act_evaluate_log_prob_parity():
    m = _model()
    rng = np.random.default_rng(0)
    for trial in range(5):
        obs, gm, sizing = _random_batch(rng, 512)
        torch.manual_seed(100 + trial)
        out = m.act(obs, gm, sizing, deterministic=False)
        (log_prob_eval, entropy, value, gate_h, anchor_h, beta_h,
         _glp, _alp) = m.evaluate(
            obs, gm, sizing, out.gate, out.anchor, out.refine_u
        )
        diff = (out.log_prob - log_prob_eval).abs().max().item()
        assert diff <= 1e-5, f"act/evaluate log-prob drift {diff}"
        assert torch.isfinite(entropy).all()
        assert torch.isfinite(out.log_prob).all()


def test_sampled_chips_legal_and_anchor_consistent():
    m = _model(1)
    rng = np.random.default_rng(1)
    obs, gm, sizing = _random_batch(rng, 1024)
    torch.manual_seed(7)
    out = m.act(obs, gm, sizing, deterministic=False)
    s = sizing.numpy()
    grid = anchor_grid_np(s[:, 0], s[:, 1], s[:, 2], s[:, 3])
    raise_rows = (out.gate == GATE_RAISE).numpy()
    chips = out.chips.numpy()
    anchors = out.anchor.numpy()
    mn = np.minimum(s[:, 0], s[:, 1])
    assert np.all(chips[raise_rows] >= mn[raise_rows])
    assert np.all(chips[raise_rows] <= s[:, 1][raise_rows])
    # Sampled anchors are legal under the recomputed grid.
    legal_at = grid.legal[np.arange(len(anchors)), anchors]
    assert legal_at.all(), "sampled anchor must be legal"
    # Non-raise rows emit zero chips.
    assert np.all(chips[~raise_rows] == 0)


def test_deterministic_atoms_and_centers():
    m = _model(2)
    rng = np.random.default_rng(2)
    obs, gm, sizing = _random_batch(rng, 512)
    out = m.act(obs, gm, sizing, deterministic=True)
    s = sizing.numpy()
    grid = anchor_grid_np(s[:, 0], s[:, 1], s[:, 2], s[:, 3])
    anchors = out.anchor.numpy()
    chips = out.chips.numpy()
    raise_rows = (out.gate == GATE_RAISE).numpy()
    refine_at = grid.refine_ok[np.arange(len(anchors)), anchors]
    anchor_chips = grid.chips[np.arange(len(anchors)), anchors]
    # Atom anchors emit the anchor chips exactly.
    atom_rows = raise_rows & ~refine_at
    assert np.array_equal(chips[atom_rows], anchor_chips[atom_rows])
    # Fresh (untrained-ish) refine heads have alpha≈beta -> u≈0.5 -> the
    # deterministic refined size stays inside the anchor's bracket.
    ref_rows = raise_rows & refine_at
    lo = grid.lo[np.arange(len(anchors)), anchors]
    hi = grid.hi[np.arange(len(anchors)), anchors]
    assert np.all(chips[ref_rows] >= lo[ref_rows])
    assert np.all(chips[ref_rows] <= hi[ref_rows])


def test_masked_anchor_entropy_zero_contribution():
    # Single-point range: only anchor 0 legal -> anchor entropy 0, no
    # refinement entropy; total sizing entropy contribution must be 0.
    m = _model(3)
    obs = torch.randn(4, OBS_DIM)
    gm = torch.ones(4, 3, dtype=torch.bool)
    sizing = torch.tensor([[500, 500, 900, 0]] * 4, dtype=torch.int64)
    out = m.act(obs, gm, sizing, deterministic=False)
    lp, entropy, _v, gate_h, anchor_h, beta_h, _glp, _alp = m.evaluate(
        obs, gm, sizing, out.gate, out.anchor, out.refine_u
    )
    gate_logits, _, _, _ = m(obs, gm)
    gate_dist = torch.distributions.Categorical(logits=gate_logits)
    assert torch.allclose(entropy, gate_dist.entropy(), atol=1e-6), (
        "degenerate sizing must contribute zero entropy beyond the gate"
    )
    assert torch.allclose(anchor_h, torch.zeros_like(anchor_h), atol=1e-6)
    assert torch.allclose(beta_h, torch.zeros_like(beta_h), atol=1e-6)


def test_v1_act_returns_actout_with_sentinel_anchor():
    m = ActorCritic(hidden_dim=32, num_layers=1).eval()
    obs = torch.randn(8, OBS_DIM)
    gm = torch.ones(8, 3, dtype=torch.bool)
    bounds = torch.tensor([[100, 900]] * 8, dtype=torch.long)
    out = m.act(obs, gm, bounds)
    assert (out.anchor == -1).all()
    # Also accepts the v2-style (B,4) sizing.
    sizing4 = torch.tensor([[100, 900, 600, 0]] * 8, dtype=torch.long)
    out4 = m.act(obs, gm, sizing4)
    assert out4.chips.shape == (8,)


def test_central_critic_shapes_and_multihot():
    crit = CentralCritic(hidden_dim=128, num_blocks=1)
    obs = torch.randn(16, OBS_DIM)
    holes = torch.full((16, 5, 5), 255, dtype=torch.uint8)
    holes[:, 0] = torch.tensor([0, 5, 17, 33, 51], dtype=torch.uint8)
    mh = opp_holes_multihot(holes)
    assert mh.shape == (16, 260)
    assert mh.sum().item() == 16 * 5  # one filled opponent slot
    assert mh[0, 0] == 1.0 and mh[0, 51] == 1.0
    assert mh[:, 52:].sum() == 0  # empty slots stay zero
    v = crit(obs, mh)
    assert v.shape == (16,)
    assert torch.isfinite(v).all()
