"""Canonical anchor-grid math (plo5bp.sizing) — the single source of
truth for the v2 sizing head. numpy/torch parity here underwrites PPO's
act/evaluate log-prob symmetry."""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.sizing import (
    ANCHOR_COUNT,
    anchor_grid_np,
    anchor_grid_torch,
    refine_chips_np,
    refine_chips_torch,
)


def _grid_pair(mn, mx, pot, tc):
    g_np = anchor_grid_np(mn, mx, pot, tc)
    sizing = torch.tensor([[mn, mx, pot, tc]], dtype=torch.int64)
    g_t = anchor_grid_torch(sizing)
    return g_np, g_t


def test_hand_computed_normal_case():
    # pot 900, to_call 0 (opening): base 900. min bet 100, PL max 900.
    g, _ = _grid_pair(100, 900, 900, 0)
    chips = g.chips.tolist()
    # f=0 -> clip(0,100,900)=100 (the min-bet floor); f=0.5 -> 450; f=1 -> 900.
    assert chips[0] == 100
    assert chips[5] == 450
    assert chips[10] == 900
    # Half-up rounding: k=1 -> 90; k=3 -> 270.
    assert chips[1] == 90 or chips[1] == 100  # clipped at min 100
    assert chips[3] == 270
    assert g.legal.tolist()[0] is True
    # 90 < 100 floor -> k=1 clamps to 100 == anchor0 -> deduped illegal.
    assert g.legal.tolist()[1] is False
    assert sum(g.legal.tolist()) >= 1


def test_facing_bet_case_matches_ui_potsize():
    # pot 1800 (incl. opp bet), to_call 900: base 2700.
    # 100% pot raise delta = 900 + 2700 = 3600 (engine PL cap).
    g, _ = _grid_pair(1800, 3600, 1800, 900)
    assert g.chips.tolist()[10] == 3600
    # f=0.5 -> 900 + 1350 = 2250.
    assert g.chips.tolist()[5] == 2250
    # anchor 0 clamps to min raise (1800).
    assert g.chips.tolist()[0] == 1800


def test_monotone_and_dedupe():
    rng = np.random.default_rng(0)
    for _ in range(500):
        pot = int(rng.integers(0, 20_000_000))
        tc = int(rng.integers(0, pot + 1)) if pot else 0
        mx = int(rng.integers(1, 25_000_000))
        mn = int(rng.integers(1, mx + 1))
        g = anchor_grid_np(mn, mx, pot, tc)
        c = g.chips
        assert np.all(c[1:] >= c[:-1]), "anchor chips must be monotone"
        lg = g.legal
        # legal anchors have strictly distinct chips
        legal_chips = c[lg]
        assert len(set(legal_chips.tolist())) == len(legal_chips)
        assert lg[0], "anchor 0 always legal outside short-shove"
        assert np.all((g.lo <= c) & (c <= g.hi))
        # interior-only refinement
        assert not g.refine_ok[0] and not g.refine_ok[10]


def test_cover_short_single_point():
    g, _ = _grid_pair(500, 500, 900, 0)  # min == max
    assert g.legal.tolist() == [True] + [False] * 10
    assert g.chips.tolist()[0] == 500
    assert not g.refine_ok.any()


def test_short_shove_only_allin_atom():
    g, _ = _grid_pair(0, 700, 1200, 600)  # engine short-shove regime
    legal = g.legal.tolist()
    assert legal == [False] * 10 + [True]
    assert g.chips.tolist()[10] == 700
    assert not g.refine_ok.any()


def test_at_least_one_legal_when_raise_legal():
    rng = np.random.default_rng(1)
    for _ in range(500):
        mx = int(rng.integers(1, 10_000_000))
        mn = int(rng.integers(0, mx + 1))
        pot = int(rng.integers(0, 10_000_000))
        tc = int(rng.integers(0, 1_000_000))
        g = anchor_grid_np(mn, mx, pot, tc)
        assert g.legal.any(), "raise legal (max>0) must leave >=1 anchor"


def test_numpy_torch_bit_parity():
    rng = np.random.default_rng(2)
    n = 1000
    mx = rng.integers(1, 25_000_000, size=n)
    mn = np.where(rng.random(n) < 0.15, 0, rng.integers(1, 1_000_000, size=n))
    mn = np.minimum(mn, mx)
    pot = rng.integers(0, 20_000_000, size=n)
    tc = rng.integers(0, 2_000_000, size=n)
    g_np = anchor_grid_np(mn, mx, pot, tc)
    sizing = torch.from_numpy(
        np.stack([mn, mx, pot, tc], axis=-1).astype(np.int64)
    )
    g_t = anchor_grid_torch(sizing)
    for a, b in zip(g_np, g_t):
        assert np.array_equal(np.asarray(a), b.numpy()), "np/torch grid drift"

    # Refinement parity over interior anchors.
    anchor = rng.integers(1, 10, size=n)
    u = rng.random(n)
    c_np = refine_chips_np(anchor, u, mn, mx, pot, tc)
    c_t = refine_chips_torch(
        torch.from_numpy(anchor.astype(np.int64)),
        torch.from_numpy(u),
        sizing,
    )
    assert np.array_equal(c_np, c_t.numpy())


def test_refinement_centered_on_anchor():
    # u = 0.5 must reproduce the anchor chips exactly (lazy center).
    rng = np.random.default_rng(3)
    for _ in range(200):
        pot = int(rng.integers(100, 5_000_000))
        tc = int(rng.integers(0, pot))
        mx = tc + pot + tc  # PL cap
        mn = max(1, int(rng.integers(1, max(2, mx // 4))))
        g = anchor_grid_np(mn, mx, pot, tc)
        for k in range(1, 10):
            c = refine_chips_np(k, 0.5, mn, mx, pot, tc)
            assert int(c) == int(g.chips[k]), (
                f"u=0.5 at anchor {k} must equal anchor chips"
            )
        # u=0 / u=1 hit the bracket edges.
        c_lo = refine_chips_np(5, 0.0, mn, mx, pot, tc)
        c_hi = refine_chips_np(5, 1.0, mn, mx, pot, tc)
        assert int(g.lo[5]) == int(c_lo)
        assert int(g.hi[5]) == int(c_hi)


def test_chip_precision_above_f32_range():
    # Pots near 18M chips (6 seats x 300bb) — int64 math must stay exact.
    pot = 18_000_003
    tc = 1
    mx = tc + pot + tc
    g = anchor_grid_np(1, mx, pot, tc)
    # f=1: to_call + base exactly.
    assert int(g.chips[10]) == tc + pot + tc
    # Adjacent anchors must differ by ~base/10, never collapse via float loss.
    diffs = np.diff(g.chips)
    assert np.all(diffs[1:] > 0)
