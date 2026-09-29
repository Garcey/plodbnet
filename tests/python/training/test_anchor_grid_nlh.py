"""NLH anchor spec: ladder semantics + np/torch parity + PLO bit-exactness.

Pins:
- The generalized (per-mille spec) grid reproduces the pre-spec PLO
  closed forms bit-exactly (chips, brackets, legality) — existing v2/v4
  checkpoints see identical sizing math.
- NLH spec: the ALL-IN atom is always max_raise; overbet anchors extend
  past the pot; shallow stacks collapse the overbet region onto the
  jam via the strictly-greater dedupe; short-shove selects only the top
  atom; refinement is centered (u=0.5 lands exactly on the anchor).
- numpy/torch twins are bit-identical for the NLH spec (the PLO twin
  parity is pinned by test_anchor_grid.py).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.sizing import (
    NLH_ANCHOR_SPEC,
    PLO_ANCHOR_SPEC,
    anchor_grid_np,
    anchor_grid_torch,
    refine_chips_np,
    refine_chips_torch,
)

RNG = np.random.default_rng(20260702)


def _random_sizing_rows(n: int) -> np.ndarray:
    """Plausible (min_raise, max_raise, pot, to_call) rows spanning
    shallow to 250bb-deep NLH spots (chips at 1bb = 10000)."""
    rows = []
    for _ in range(n):
        pot = int(RNG.integers(10_000, 3_000_000))
        to_call = int(RNG.integers(0, 2) * RNG.integers(0, pot + 1))
        mn = int(RNG.integers(10_000, 200_000))
        # NL max: up to 250bb stack behind, independent of pot.
        mx = int(RNG.integers(mn, 2_500_000 + mn))
        rows.append((mn, mx, pot, to_call))
    # Degenerate regimes.
    rows.append((0, 500_000, 100_000, 20_000))   # short-shove
    rows.append((0, 0, 100_000, 0))              # raise unavailable
    rows.append((50_000, 50_000, 100_000, 0))    # single-point range
    return np.asarray(rows, dtype=np.int64)


def test_plo_spec_matches_legacy_closed_forms_bit_exactly():
    """The per-mille generalization must reproduce (k*base+5)//10 and
    ((2k±1)*base+10)//20 for every anchor including the atoms."""
    rows = _random_sizing_rows(500)
    mn, mx, pot, tc = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    grid = anchor_grid_np(mn, mx, pot, tc, spec=PLO_ANCHOR_SPEC)

    mr = np.minimum(mn, mx)
    base = pot + tc
    k = np.arange(11, dtype=np.int64)
    b = base[:, None]
    t = tc[:, None]
    lo_ref = np.clip(t + ((2 * k - 1) * b + 10) // 20, mr[:, None], mx[:, None])
    hi_ref = np.clip(t + ((2 * k + 1) * b + 10) // 20, mr[:, None], mx[:, None])
    chips_ref = np.clip(t + (k * b + 5) // 10, mr[:, None], mx[:, None])
    # Short-shove rows overwrite the top anchor with mx (same both ways).
    ss = (mn == 0) & (mx > 0)
    chips_ref[ss, 10] = mx[ss]

    np.testing.assert_array_equal(grid.chips, chips_ref)
    np.testing.assert_array_equal(grid.lo, lo_ref)
    np.testing.assert_array_equal(grid.hi, hi_ref)

    # Legacy refine expression is used verbatim for the PLO spec.
    anchor = RNG.integers(1, 10, size=len(rows))
    u = RNG.random(len(rows))
    got = refine_chips_np(anchor, u, mn, mx, pot, tc, spec=PLO_ANCHOR_SPEC)
    frac = (2.0 * anchor.astype(np.float64) - 1.0) / 20.0 + u / 10.0
    raw = np.floor(tc + frac * base + 0.5).astype(np.int64)
    ref = np.clip(raw, mr, mx)
    np.testing.assert_array_equal(got, ref)


def test_nlh_np_torch_parity():
    rows = _random_sizing_rows(500)
    mn, mx, pot, tc = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    g_np = anchor_grid_np(mn, mx, pot, tc, spec=NLH_ANCHOR_SPEC)
    g_t = anchor_grid_torch(torch.from_numpy(rows), spec=NLH_ANCHOR_SPEC)
    np.testing.assert_array_equal(g_np.chips, g_t.chips.numpy())
    np.testing.assert_array_equal(g_np.legal, g_t.legal.numpy())
    np.testing.assert_array_equal(g_np.lo, g_t.lo.numpy())
    np.testing.assert_array_equal(g_np.hi, g_t.hi.numpy())
    np.testing.assert_array_equal(g_np.refine_ok, g_t.refine_ok.numpy())

    anchor = RNG.integers(1, NLH_ANCHOR_SPEC.count - 1, size=len(rows))
    u = RNG.random(len(rows))
    r_np = refine_chips_np(anchor, u, mn, mx, pot, tc, spec=NLH_ANCHOR_SPEC)
    r_t = refine_chips_torch(
        torch.from_numpy(anchor),
        torch.from_numpy(u),
        torch.from_numpy(rows),
        spec=NLH_ANCHOR_SPEC,
    )
    np.testing.assert_array_equal(r_np, r_t.numpy())


def test_nlh_allin_atom_is_max_raise_and_deep_overbets_are_live():
    # 100bb-deep single-raised pot: pot 45000, facing nothing postflop.
    mn, mx, pot, tc = 10_000, 990_000, 45_000, 0
    grid = anchor_grid_np(mn, mx, pot, tc, spec=NLH_ANCHOR_SPEC)
    assert grid.chips[-1] == mx, "all-in atom always carries max_raise"
    assert bool(grid.legal[-1]), "deep stack: jam is a distinct legal anchor"
    # Overbet anchors express sizes strictly above the pot anchor.
    pot_anchor = 6  # frac 1000 pm
    assert grid.chips[9] > grid.chips[pot_anchor]
    # Every legal anchor's chips strictly increase.
    legal_chips = grid.chips[grid.legal]
    assert np.all(np.diff(legal_chips) > 0)
    # The old 0..100% ladder tops out at pot — the atom is what reaches mx.
    assert grid.chips[pot_anchor] == tc + (1000 * (pot + tc) + 500) // 1000
    assert grid.chips[-1] > grid.chips[pot_anchor] * 10


def test_nlh_shallow_collapses_overbets_onto_jam():
    # 20bb: stack behind < pot → every frac ≥ some point clamps to mx and
    # dedupes; the jam is represented by the FIRST anchor that hits mx.
    mn, mx, pot, tc = 10_000, 150_000, 200_000, 0
    grid = anchor_grid_np(mn, mx, pot, tc, spec=NLH_ANCHOR_SPEC)
    at_max = grid.chips == mx
    assert at_max.sum() >= 2, "several anchors clamp to the shove"
    first_at_max = int(np.argmax(at_max))
    assert bool(grid.legal[first_at_max])
    assert not grid.legal[at_max & (np.arange(grid.chips.shape[-1]) > first_at_max)].any()


def test_nlh_short_shove_selects_only_top_atom():
    grid = anchor_grid_np(0, 80_000, 100_000, 30_000, spec=NLH_ANCHOR_SPEC)
    assert bool(grid.legal[-1])
    assert grid.legal.sum() == 1
    assert grid.chips[-1] == 80_000
    assert not grid.refine_ok.any()


def test_nlh_refine_center_hits_anchor_chips():
    rows = _random_sizing_rows(300)
    mn, mx, pot, tc = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    grid = anchor_grid_np(mn, mx, pot, tc, spec=NLH_ANCHOR_SPEC)
    for k in range(1, NLH_ANCHOR_SPEC.count - 1):
        anchor = np.full(len(rows), k, dtype=np.int64)
        u = np.full(len(rows), 0.5)
        got = refine_chips_np(anchor, u, mn, mx, pot, tc, spec=NLH_ANCHOR_SPEC)
        np.testing.assert_array_equal(got, grid.chips[:, k])


def test_nlh_brackets_never_swallow_neighbours():
    s = NLH_ANCHOR_SPEC
    for k in range(1, len(s.fracs_pm)):
        assert s.bracket_lo_pm[k] >= s.fracs_pm[k - 1], k
        assert s.bracket_hi_pm[k - 1] <= s.fracs_pm[k], k
        # Exact centering (lazy u=0.5 on the anchor).
        assert s.bracket_lo_pm[k] + s.bracket_hi_pm[k] == 2 * s.fracs_pm[k]


def test_nlh_spec_shape_constants():
    assert NLH_ANCHOR_SPEC.count == 12
    assert NLH_ANCHOR_SPEC.refinable[0] is False
    assert NLH_ANCHOR_SPEC.refinable[-1] is False
    assert all(NLH_ANCHOR_SPEC.refinable[1:-1])
    assert PLO_ANCHOR_SPEC.count == 11
    # PLO brackets reduce to the historical ±50 pm.
    assert PLO_ANCHOR_SPEC.bracket_lo_pm == tuple(100 * k - 50 for k in range(11))
    assert PLO_ANCHOR_SPEC.bracket_hi_pm == tuple(100 * k + 50 for k in range(11))


def test_atoms_have_no_refine_even_when_bracket_nondegenerate():
    rows = _random_sizing_rows(100)
    mn, mx, pot, tc = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    grid = anchor_grid_np(mn, mx, pot, tc, spec=NLH_ANCHOR_SPEC)
    assert not grid.refine_ok[:, 0].any()
    assert not grid.refine_ok[:, -1].any()
