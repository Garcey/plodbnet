"""Canonical anchor-grid sizing math for the v2 anchor sizing head.

THE single source of truth for mapping the 11 pot-fraction anchors
(f_k = k/10, k = 0..10) to chip amounts, legality masks, and refinement
brackets. The same math runs in the network (torch), the rollout
collectors and UI (numpy), and the tests — any drift between
implementations breaks PPO's act/evaluate log-prob parity, so both
variants here are pinned bit-identical by tests/python/test_anchor_grid.py.

Conventions (all chip quantities are int64 chip DELTAS the actor adds,
matching the engine's `apply_raise_chips` interface):

- ``base = pot + to_call`` is the pot after a call; a "100% pot" raise
  is ``to_call + base`` — the pot-limit maximum (engine.rs pl_total) and
  the same convention as the UI's potSize() shortcuts.
- ``chips_k = clip(to_call + round_half_up(f_k * base), min_raise, max_raise)``.
  Rounding is canonical half-up via integer math: (k*base + 5) // 10.
- Anchors clamp monotonically, so legality dedupe is "strictly greater
  than the previous anchor's chips"; anchor 0 is always legal (it IS
  the min-raise after clamping — label it "min", not "0%").
- Interior anchors (k=1..9) carry a Beta refinement slider over the
  fraction bracket [f_k - 0.05, f_k + 0.05] (u in [0,1], lazy center on
  the anchor). Anchors 0 and 10 are ATOMS — no slider.
- SHORT-SHOVE regime (engine zeroes min_raise while Raise stays legal;
  the env redirects to apply(AllIn) and ignores chips): the only legal
  anchor is k=10, an atom whose chips display as max_raise (the all-in).

The grid is a pure function of (min_raise, max_raise, pot, to_call), so
masks and brackets are recomputable anywhere — PPO's evaluate() never
reverse-engineers anchors from chips; the sampled (anchor, u) are stored.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

ANCHOR_COUNT = 11
ANCHOR_FRACS = tuple(k / 10.0 for k in range(ANCHOR_COUNT))
# Interior refinement bracket half-width in fraction space.
BRACKET_HALF = 0.05


class AnchorGrid(NamedTuple):
    """Per-anchor sizing data; arrays have trailing dim ANCHOR_COUNT."""

    chips: np.ndarray | torch.Tensor      # int64 — clamped chip delta per anchor
    legal: np.ndarray | torch.Tensor      # bool  — deduped legality
    lo: np.ndarray | torch.Tensor         # int64 — bracket lower chip bound
    hi: np.ndarray | torch.Tensor         # int64 — bracket upper chip bound
    refine_ok: np.ndarray | torch.Tensor  # bool  — interior + non-collapsed bracket


def anchor_grid_np(
    min_raise: np.ndarray | int,
    max_raise: np.ndarray | int,
    pot: np.ndarray | int,
    to_call: np.ndarray | int,
) -> AnchorGrid:
    """Numpy anchor grid. Inputs broadcast; outputs gain a trailing
    (ANCHOR_COUNT,) axis."""
    mn = np.asarray(min_raise, dtype=np.int64)
    mx = np.asarray(max_raise, dtype=np.int64)
    pot_a = np.asarray(pot, dtype=np.int64)
    tc = np.asarray(to_call, dtype=np.int64)
    mn, mx, pot_a, tc = np.broadcast_arrays(mn, mx, pot_a, tc)

    mr = np.minimum(mn, mx)  # defensive clamp (UI collapses min for display)
    base = pot_a + tc
    k = np.arange(ANCHOR_COUNT, dtype=np.int64)
    shape = mn.shape + (ANCHOR_COUNT,)

    def _bcast(x: np.ndarray) -> np.ndarray:
        return np.broadcast_to(x[..., None], shape)

    base_b = _bcast(base)
    tc_b = _bcast(tc)
    mr_b = _bcast(mr)
    mx_b = _bcast(mx)

    chips = np.clip(tc_b + (k * base_b + 5) // 10, mr_b, mx_b)
    lo = np.clip(tc_b + ((2 * k - 1) * base_b + 10) // 20, mr_b, mx_b)
    hi = np.clip(tc_b + ((2 * k + 1) * base_b + 10) // 20, mr_b, mx_b)

    legal = np.ones(shape, dtype=bool)
    legal[..., 1:] = chips[..., 1:] > chips[..., :-1]
    interior = (k >= 1) & (k <= 9)
    refine_ok = legal & interior & (hi > lo)

    # Short-shove: only the all-in atom (k=10) is selectable.
    ss = (mn == 0) & (mx > 0)
    if np.any(ss):
        ss_b = _bcast(ss)
        is_top = np.broadcast_to(k == ANCHOR_COUNT - 1, shape)
        legal = np.where(ss_b, is_top, legal)
        chips = np.where(ss_b & is_top, mx_b, chips)
        refine_ok = np.where(ss_b, False, refine_ok)

    return AnchorGrid(chips=chips, legal=legal, lo=lo, hi=hi, refine_ok=refine_ok)


def anchor_grid_torch(sizing: torch.Tensor) -> AnchorGrid:
    """Torch anchor grid from a (..., 4) int64 sizing tensor
    [min_raise, max_raise, pot, to_call]. Branch-free (torch.compile
    dynamic-shape friendly); bit-identical to anchor_grid_np."""
    mn = sizing[..., 0]
    mx = sizing[..., 1]
    pot = sizing[..., 2]
    tc = sizing[..., 3]

    mr = torch.minimum(mn, mx)
    base = pot + tc
    k = torch.arange(ANCHOR_COUNT, dtype=torch.int64, device=sizing.device)

    base_b = base.unsqueeze(-1)
    tc_b = tc.unsqueeze(-1)
    mr_b = mr.unsqueeze(-1)
    mx_b = mx.unsqueeze(-1)

    chips = torch.clamp(
        tc_b + torch.div(k * base_b + 5, 10, rounding_mode="floor"),
        mr_b, mx_b,
    )
    lo = torch.clamp(
        tc_b + torch.div((2 * k - 1) * base_b + 10, 20, rounding_mode="floor"),
        mr_b, mx_b,
    )
    hi = torch.clamp(
        tc_b + torch.div((2 * k + 1) * base_b + 10, 20, rounding_mode="floor"),
        mr_b, mx_b,
    )

    first = torch.ones_like(chips[..., :1], dtype=torch.bool)
    legal = torch.cat([first, chips[..., 1:] > chips[..., :-1]], dim=-1)
    interior = (k >= 1) & (k <= 9)
    refine_ok = legal & interior & (hi > lo)

    ss = ((mn == 0) & (mx > 0)).unsqueeze(-1)
    is_top = (k == ANCHOR_COUNT - 1).expand_as(legal)
    legal = torch.where(ss, is_top, legal)
    chips = torch.where(ss & is_top, mx_b.expand_as(chips), chips)
    refine_ok = torch.where(ss, torch.zeros_like(refine_ok), refine_ok)

    return AnchorGrid(chips=chips, legal=legal, lo=lo, hi=hi, refine_ok=refine_ok)


def refine_chips_np(
    anchor: np.ndarray,
    u: np.ndarray,
    min_raise: np.ndarray | int,
    max_raise: np.ndarray | int,
    pot: np.ndarray | int,
    to_call: np.ndarray | int,
) -> np.ndarray:
    """Map (anchor k, refinement u in [0,1]) to a chip delta.

    frac = (2k-1)/20 + u/10 — the bracket [f_k-0.05, f_k+0.05] with the
    lazy center (u=0.5) exactly on the anchor. Float64 before rounding:
    chip counts exceed float32's 2^24 integer range.
    """
    mn = np.asarray(min_raise, dtype=np.int64)
    mx = np.asarray(max_raise, dtype=np.int64)
    pot_a = np.asarray(pot, dtype=np.float64)
    tc = np.asarray(to_call, dtype=np.float64)
    base = pot_a + tc
    frac = (2.0 * np.asarray(anchor, dtype=np.float64) - 1.0) / 20.0 \
        + np.asarray(u, dtype=np.float64) / 10.0
    raw = np.floor(tc + frac * base + 0.5).astype(np.int64)
    return np.clip(raw, np.minimum(mn, mx), mx)


def refine_chips_torch(
    anchor: torch.Tensor,
    u: torch.Tensor,
    sizing: torch.Tensor,
) -> torch.Tensor:
    """Torch twin of refine_chips_np; sizing is (..., 4) int64."""
    mn = torch.minimum(sizing[..., 0], sizing[..., 1])
    mx = sizing[..., 1]
    base = (sizing[..., 2] + sizing[..., 3]).to(torch.float64)
    tc = sizing[..., 3].to(torch.float64)
    frac = (2.0 * anchor.to(torch.float64) - 1.0) / 20.0 + u.to(torch.float64) / 10.0
    raw = torch.floor(tc + frac * base + 0.5).to(torch.int64)
    return torch.clamp(raw, mn, mx)


def sizing_from_info(info) -> np.ndarray:
    """Build the (4,) int64 sizing context [min_raise, max_raise, pot,
    to_call] from a StepInfo. `to_call` is the actor's outstanding call
    amount (uncapped by stack: whenever Raise is legal, stack > to_call)."""
    raw = info.raw_obs
    actor = info.actor
    bet_to_call = int(raw.get("bet_to_call", 0))
    commit = int(raw["street_commit"][actor]) if actor is not None else 0
    to_call = max(0, bet_to_call - commit)
    return np.array(
        [int(info.min_raise_chips), int(info.max_raise_chips),
         int(raw.get("pot", 0)), to_call],
        dtype=np.int64,
    )
