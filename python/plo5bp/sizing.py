"""Canonical anchor-grid sizing math for the v2+ anchor sizing heads.

THE single source of truth for mapping a variant's pot-fraction anchor
ladder (an :class:`AnchorSpec`) to chip amounts, legality masks, and
refinement brackets. The same math runs in the network (torch), the
rollout collectors and UI (numpy), and the tests — any drift between
implementations breaks PPO's act/evaluate log-prob parity, so the
numpy/torch twins are pinned bit-identical by
tests/python/test_anchor_grid.py (PLO spec, incl. bit-exactness vs the
pre-spec closed forms) and tests/python/test_anchor_grid_nlh.py.

Two specs exist:

- ``PLO_ANCHOR_SPEC`` — the original 11 pot-fraction anchors
  (f_k = k/10, k = 0..10) for the pot-limit double-board bomb pot.
  Anchor 0 is the min atom; anchor 10 is the pot atom (which doubles
  as the all-in in the short-shove regime, and equals the PL cap when
  stacks are deep). Module-level defaults keep every existing call
  site bit-exact — v2/v4 PLO checkpoints are unaffected.
- ``NLH_ANCHOR_SPEC`` — the no-limit hold'em ladder: min atom, fracs
  25/33/50/66/80/100/125/160/200/275% of pot, then an explicit ALL-IN
  atom whose chips are always ``max_raise``. The overbet region is
  near-geometric so the ordinal (v4) head's index axis approximates a
  log-size axis, and the all-in atom is what makes jams expressible at
  any SPR: the clamp only pulls anchors DOWN into
  [min_raise, max_raise], so a 0..100%-pot ladder can never reach a
  deep-stack all-in. Under the v4 head the ladder is only the SUPPORT
  of the (mu, s) discretized logistic — the legal-edge tail absorption
  lands the whole upper tail on the all-in atom (mu high = jam).

Conventions (all chip quantities are int64 chip DELTAS the actor adds,
matching the engine's `apply_raise_chips` interface):

- ``base = pot + to_call`` is the pot after a call; a "100% pot" raise
  is ``to_call + base`` — the pot-limit maximum (engine.rs pl_total)
  and the same convention as the UI's potSize() shortcuts. Overbet
  fractions extend the same formula past 100%.
- Fractions live in integer PER-MILLE so chip math stays exact:
  ``chips_k = clip(to_call + (frac_pm_k * base + 500) // 1000,
  min_raise, max_raise)`` — canonical half-up rounding via integer
  ops. For the legacy PLO fracs this reduces exactly to the historical
  ``(k * base + 5) // 10``.
- Anchors clamp monotonically, so legality dedupe is "strictly greater
  than the previous anchor's chips"; anchor 0 is always legal (it IS
  the min-raise after clamping — label it "min", not "0%").
- Refinement brackets are symmetric about each refinable anchor with
  half-width ``min(gap to prev frac, gap to next frac) // 2`` per-mille
  — the lazy center (u=0.5, the untrained Beta mean) sits exactly on
  the anchor and a bracket never swallows a neighbouring anchor. For
  the uniformly spaced PLO ladder this reduces exactly to the
  historical ±0.05 brackets.
- The FIRST and LAST anchors are ATOMS — no slider. (PLO: min + pot;
  NLH: min + all-in.) Interior anchors carry Beta(α_k, β_k) sliders,
  so a network's refine head always has ``spec.count - 2`` slots.
- SHORT-SHOVE regime (engine zeroes min_raise while Raise stays legal;
  the env redirects to apply(AllIn) and ignores chips): the only legal
  anchor is the TOP one, an atom whose chips display as max_raise.

The grid is a pure function of (min_raise, max_raise, pot, to_call,
spec), so masks and brackets are recomputable anywhere — PPO's
evaluate() never reverse-engineers anchors from chips; the sampled
(anchor, u) are stored.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class AnchorSpec(NamedTuple):
    """A variant's anchor ladder. Fractions in integer per-mille of
    ``base = pot + to_call``; ``fracs_pm[0]`` must be 0 (the min atom).
    With ``allin_atom`` the ladder gains a final anchor whose chips are
    always ``max_raise`` (no fraction, no slider)."""

    name: str
    fracs_pm: tuple[int, ...]
    allin_atom: bool
    bracket_lo_pm: tuple[int, ...]
    bracket_hi_pm: tuple[int, ...]
    #: True where the anchor carries a Beta refinement slider. Always
    #: False at index 0 and the last index (atoms).
    refinable: tuple[bool, ...]
    #: Use the pre-spec float expression in refine_chips_* so the legacy
    #: PLO path stays bit-exact with historical checkpoints and tests.
    #: The generalized expression is the same rational math but a
    #: different float op ORDER — a 1-ulp drift at a .5 rounding
    #: boundary would break act/evaluate log-prob parity.
    legacy_refine: bool

    @property
    def count(self) -> int:
        return len(self.fracs_pm) + (1 if self.allin_atom else 0)


def _make_spec(
    name: str,
    fracs_pm: tuple[int, ...],
    allin_atom: bool,
    legacy_refine: bool = False,
) -> AnchorSpec:
    assert len(fracs_pm) >= 2 and fracs_pm[0] == 0, "ladder starts at the min atom"
    assert all(b > a for a, b in zip(fracs_pm, fracs_pm[1:])), "fracs must increase"
    lo: list[int] = []
    hi: list[int] = []
    for k, f in enumerate(fracs_pm):
        gaps = []
        if k > 0:
            gaps.append(f - fracs_pm[k - 1])
        if k < len(fracs_pm) - 1:
            gaps.append(fracs_pm[k + 1] - f)
        half = min(gaps) // 2
        lo.append(f - half)
        hi.append(f + half)
    count = len(fracs_pm) + (1 if allin_atom else 0)
    refinable = tuple(0 < i < count - 1 for i in range(count))
    return AnchorSpec(
        name=name,
        fracs_pm=tuple(fracs_pm),
        allin_atom=allin_atom,
        bracket_lo_pm=tuple(lo),
        bracket_hi_pm=tuple(hi),
        refinable=refinable,
        legacy_refine=legacy_refine,
    )


#: Legacy PLO ladder: 0,10,...,100% pot. Brackets reduce to ±50 pm.
PLO_ANCHOR_SPEC: AnchorSpec = _make_spec(
    "plo5_pot",
    tuple(100 * k for k in range(11)),
    allin_atom=False,
    legacy_refine=True,
)

#: NLH ladder: min, 25..275% pot (near-geometric overbets), ALL-IN atom.
NLH_ANCHOR_SPEC: AnchorSpec = _make_spec(
    "nlh_overbet",
    (0, 250, 330, 500, 660, 800, 1000, 1250, 1600, 2000, 2750),
    allin_atom=True,
)

# Legacy module-level constants (PLO). Existing imports — network.py,
# ppo.py, UI — keep meaning exactly what they always did.
ANCHOR_COUNT = PLO_ANCHOR_SPEC.count
ANCHOR_FRACS = tuple(f / 1000.0 for f in PLO_ANCHOR_SPEC.fracs_pm)
# Interior refinement bracket half-width in fraction space (PLO ladder).
BRACKET_HALF = 0.05


class AnchorGrid(NamedTuple):
    """Per-anchor sizing data; arrays have trailing dim ``spec.count``."""

    chips: np.ndarray | torch.Tensor      # int64 — clamped chip delta per anchor
    legal: np.ndarray | torch.Tensor      # bool  — deduped legality
    lo: np.ndarray | torch.Tensor         # int64 — bracket lower chip bound
    hi: np.ndarray | torch.Tensor         # int64 — bracket upper chip bound
    refine_ok: np.ndarray | torch.Tensor  # bool  — refinable + non-collapsed bracket


def anchor_grid_np(
    min_raise: np.ndarray | int,
    max_raise: np.ndarray | int,
    pot: np.ndarray | int,
    to_call: np.ndarray | int,
    spec: AnchorSpec = PLO_ANCHOR_SPEC,
) -> AnchorGrid:
    """Numpy anchor grid. Inputs broadcast; outputs gain a trailing
    (spec.count,) axis."""
    mn = np.asarray(min_raise, dtype=np.int64)
    mx = np.asarray(max_raise, dtype=np.int64)
    pot_a = np.asarray(pot, dtype=np.int64)
    tc = np.asarray(to_call, dtype=np.int64)
    mn, mx, pot_a, tc = np.broadcast_arrays(mn, mx, pot_a, tc)

    mr = np.minimum(mn, mx)  # defensive clamp (UI collapses min for display)
    base = pot_a + tc
    count = spec.count
    nf = len(spec.fracs_pm)
    fr = np.asarray(spec.fracs_pm, dtype=np.int64)
    blo = np.asarray(spec.bracket_lo_pm, dtype=np.int64)
    bhi = np.asarray(spec.bracket_hi_pm, dtype=np.int64)
    fshape = mn.shape + (nf,)

    def _bcast_f(x: np.ndarray) -> np.ndarray:
        return np.broadcast_to(x[..., None], fshape)

    base_f = _bcast_f(base)
    tc_f = _bcast_f(tc)
    mr_f = _bcast_f(mr)
    mx_f = _bcast_f(mx)

    chips = np.clip(tc_f + (fr * base_f + 500) // 1000, mr_f, mx_f)
    lo = np.clip(tc_f + (blo * base_f + 500) // 1000, mr_f, mx_f)
    hi = np.clip(tc_f + (bhi * base_f + 500) // 1000, mr_f, mx_f)

    if spec.allin_atom:
        atom = mx[..., None]
        chips = np.concatenate([chips, atom], axis=-1)
        lo = np.concatenate([lo, atom], axis=-1)
        hi = np.concatenate([hi, atom], axis=-1)

    shape = mn.shape + (count,)
    legal = np.ones(shape, dtype=bool)
    legal[..., 1:] = chips[..., 1:] > chips[..., :-1]
    refinable = np.asarray(spec.refinable, dtype=bool)
    refine_ok = legal & refinable & (hi > lo)

    # Short-shove: only the top atom (all-in) is selectable.
    ss = (mn == 0) & (mx > 0)
    if np.any(ss):
        ss_b = np.broadcast_to(ss[..., None], shape)
        k = np.arange(count, dtype=np.int64)
        is_top = np.broadcast_to(k == count - 1, shape)
        mx_b = np.broadcast_to(mx[..., None], shape)
        legal = np.where(ss_b, is_top, legal)
        chips = np.where(ss_b & is_top, mx_b, chips)
        refine_ok = np.where(ss_b, False, refine_ok)

    return AnchorGrid(chips=chips, legal=legal, lo=lo, hi=hi, refine_ok=refine_ok)


def anchor_grid_torch(
    sizing: torch.Tensor,
    spec: AnchorSpec = PLO_ANCHOR_SPEC,
) -> AnchorGrid:
    """Torch anchor grid from a (..., 4) int64 sizing tensor
    [min_raise, max_raise, pot, to_call]. Branch-free per spec
    (torch.compile dynamic-shape friendly); bit-identical to
    anchor_grid_np."""
    mn = sizing[..., 0]
    mx = sizing[..., 1]
    pot = sizing[..., 2]
    tc = sizing[..., 3]

    mr = torch.minimum(mn, mx)
    base = pot + tc
    count = spec.count
    dev = sizing.device
    fr = torch.tensor(spec.fracs_pm, dtype=torch.int64, device=dev)
    blo = torch.tensor(spec.bracket_lo_pm, dtype=torch.int64, device=dev)
    bhi = torch.tensor(spec.bracket_hi_pm, dtype=torch.int64, device=dev)

    base_b = base.unsqueeze(-1)
    tc_b = tc.unsqueeze(-1)
    mr_b = mr.unsqueeze(-1)
    mx_b = mx.unsqueeze(-1)

    chips = torch.clamp(
        tc_b + torch.div(fr * base_b + 500, 1000, rounding_mode="floor"),
        mr_b, mx_b,
    )
    lo = torch.clamp(
        tc_b + torch.div(blo * base_b + 500, 1000, rounding_mode="floor"),
        mr_b, mx_b,
    )
    hi = torch.clamp(
        tc_b + torch.div(bhi * base_b + 500, 1000, rounding_mode="floor"),
        mr_b, mx_b,
    )

    if spec.allin_atom:
        chips = torch.cat([chips, mx_b], dim=-1)
        lo = torch.cat([lo, mx_b], dim=-1)
        hi = torch.cat([hi, mx_b], dim=-1)

    first = torch.ones_like(chips[..., :1], dtype=torch.bool)
    legal = torch.cat([first, chips[..., 1:] > chips[..., :-1]], dim=-1)
    refinable = torch.tensor(spec.refinable, dtype=torch.bool, device=dev)
    refine_ok = legal & refinable & (hi > lo)

    ss = ((mn == 0) & (mx > 0)).unsqueeze(-1)
    k = torch.arange(count, dtype=torch.int64, device=dev)
    is_top = (k == count - 1).expand_as(legal)
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
    spec: AnchorSpec = PLO_ANCHOR_SPEC,
) -> np.ndarray:
    """Map (anchor k, refinement u in [0,1]) to a chip delta.

    frac spans the anchor's bracket [lo_pm, hi_pm] with the lazy center
    (u=0.5) exactly on the anchor. Float64 before rounding: chip counts
    exceed float32's 2^24 integer range. The legacy PLO spec keeps the
    historical expression ``(2k-1)/20 + u/10`` verbatim (same rational
    value, but a different float op order could drift 1 ulp at a .5
    rounding boundary and break stored-action parity).
    """
    mn = np.asarray(min_raise, dtype=np.int64)
    mx = np.asarray(max_raise, dtype=np.int64)
    pot_a = np.asarray(pot, dtype=np.float64)
    tc = np.asarray(to_call, dtype=np.float64)
    base = pot_a + tc
    if spec.legacy_refine:
        frac = (2.0 * np.asarray(anchor, dtype=np.float64) - 1.0) / 20.0 \
            + np.asarray(u, dtype=np.float64) / 10.0
    else:
        blo = np.asarray(spec.bracket_lo_pm, dtype=np.float64)
        span = np.asarray(spec.bracket_hi_pm, dtype=np.float64) - blo
        idx = np.clip(
            np.asarray(anchor, dtype=np.int64), 0, len(spec.fracs_pm) - 1
        )
        frac = (blo[idx] + np.asarray(u, dtype=np.float64) * span[idx]) / 1000.0
    raw = np.floor(tc + frac * base + 0.5).astype(np.int64)
    return np.clip(raw, np.minimum(mn, mx), mx)


def refine_chips_torch(
    anchor: torch.Tensor,
    u: torch.Tensor,
    sizing: torch.Tensor,
    spec: AnchorSpec = PLO_ANCHOR_SPEC,
) -> torch.Tensor:
    """Torch twin of refine_chips_np; sizing is (..., 4) int64."""
    mn = torch.minimum(sizing[..., 0], sizing[..., 1])
    mx = sizing[..., 1]
    base = (sizing[..., 2] + sizing[..., 3]).to(torch.float64)
    tc = sizing[..., 3].to(torch.float64)
    if spec.legacy_refine:
        frac = (2.0 * anchor.to(torch.float64) - 1.0) / 20.0 \
            + u.to(torch.float64) / 10.0
    else:
        dev = sizing.device
        blo = torch.tensor(spec.bracket_lo_pm, dtype=torch.float64, device=dev)
        bhi = torch.tensor(spec.bracket_hi_pm, dtype=torch.float64, device=dev)
        span = bhi - blo
        idx = anchor.to(torch.int64).clamp(0, len(spec.fracs_pm) - 1)
        frac = (blo[idx] + u.to(torch.float64) * span[idx]) / 1000.0
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
