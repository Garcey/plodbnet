"""Canonical NLH train roots — ClubGG 5/10($5) locked (decision #4).

Phase 0–2 sampling strata randomize SPR / seats / asymmetry *around*
this stake structure. Chip unit matches the rest of the project:
1 bb = 10000 chips ($10 at $5/$10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from plo5bp.config import VARIANT_NLH, GameConfig
from plo5bp.sizing import NLH_ANCHOR_SPEC


# Locked ClubGG reference table (user decision 2026-07-16).
CLUBGG_BB_CHIPS = 10_000
CLUBGG_SB_CHIPS = 5_000
CLUBGG_ANTE_CHIPS = 5_000  # $5 ante at $5/$10 → 0.5 bb
CLUBGG_DEFAULT_STACK_BB = 100.0
CLUBGG_STACK_MIN_BB = 100.0
CLUBGG_STACK_MAX_BB = 250.0


@dataclass(frozen=True)
class ClubGGRoot:
    """Immutable stake structure for the ClubGG 5/10($5) table."""

    name: str = "clubgg_5_10_5"
    variant: str = VARIANT_NLH
    bb: int = CLUBGG_BB_CHIPS
    sb: int = CLUBGG_SB_CHIPS
    ante: int = CLUBGG_ANTE_CHIPS
    default_stack_bb: float = CLUBGG_DEFAULT_STACK_BB
    stack_min_bb: float = CLUBGG_STACK_MIN_BB
    stack_max_bb: float = CLUBGG_STACK_MAX_BB
    anchor_spec_name: str = NLH_ANCHOR_SPEC.name

    def game_config(
        self,
        *,
        num_seats: int = 2,
        stack_bb: float | None = None,
        starting_stacks_bb: Sequence[float] | None = None,
    ) -> GameConfig:
        if starting_stacks_bb is not None:
            stacks = tuple(int(round(float(s) * self.bb)) for s in starting_stacks_bb)
            if len(stacks) != num_seats:
                raise ValueError(
                    f"starting_stacks_bb length {len(stacks)} != num_seats {num_seats}"
                )
            return GameConfig(
                num_seats=num_seats,
                starting_stack=stacks[0],
                starting_stacks=stacks,
                ante=self.ante,
                bb=self.bb,
                sb=self.sb,
                variant=self.variant,
            )
        sbb = self.default_stack_bb if stack_bb is None else float(stack_bb)
        chips = int(round(sbb * self.bb))
        return GameConfig(
            num_seats=num_seats,
            starting_stack=chips,
            ante=self.ante,
            bb=self.bb,
            sb=self.sb,
            variant=self.variant,
        )


CLUBGG_NLH_ROOT = ClubGGRoot()


# SPR strata for Phase 0–2 label sampling (decision #4 still anchors realism).
# Bands chosen so shallow/deep both get mass; preflop 100bb ≈ SPR ~20–55
# depending on pot (blinds+antes).
SPR_STRATA: tuple[tuple[str, float, float], ...] = (
    ("ultra_short", 0.5, 2.0),
    ("short", 2.0, 6.0),
    ("mid", 6.0, 15.0),
    ("deep", 15.0, 40.0),
    ("very_deep", 40.0, 80.0),
)

STREET_NAMES = ("preflop", "flop", "turn", "river")


@dataclass(frozen=True)
class RootSample:
    """One offline-solve / train root."""

    root_name: str
    num_seats: int
    street: int  # 0=preflop .. 3=river
    spr_band: str
    stack_bb: tuple[float, ...]  # per seat effective stack in bb
    pot_bb: float
    button: int
    seed: int
    spr: float  # pot-relative effective SPR at hero (min stack / pot)

    def to_game_config(self, root: ClubGGRoot = CLUBGG_NLH_ROOT) -> GameConfig:
        return root.game_config(
            num_seats=self.num_seats,
            starting_stacks_bb=self.stack_bb,
        )

    def as_dict(self) -> dict:
        return {
            "root_name": self.root_name,
            "num_seats": self.num_seats,
            "street": self.street,
            "street_name": STREET_NAMES[self.street],
            "spr_band": self.spr_band,
            "stack_bb": list(self.stack_bb),
            "pot_bb": self.pot_bb,
            "button": self.button,
            "seed": self.seed,
            "spr": self.spr,
        }


def _draw_spr(
    rng: np.random.Generator,
    band: tuple[str, float, float] | None = None,
) -> tuple[str, float]:
    if band is None:
        band = SPR_STRATA[int(rng.integers(0, len(SPR_STRATA)))]
    name, lo, hi = band
    # Log-uniform within band — denser near the shallow edge of deep tiers.
    u = float(rng.uniform(0.0, 1.0))
    spr = lo * ((hi / lo) ** u) if lo > 0 else hi * u
    return name, float(spr)


def sample_train_roots(
    n: int,
    *,
    seed: int = 0,
    root: ClubGGRoot = CLUBGG_NLH_ROOT,
    seats: Sequence[int] = (2,),
    streets: Sequence[int] = (1, 2, 3),  # HU postflop default for Phase 0
    hu_heavy: bool = True,
    pot_bb_range: tuple[float, float] = (1.5, 40.0),
) -> list[RootSample]:
    """Sample Phase 0 train roots around the ClubGG stake structure.

    Day-1 labels are **HU-heavy postflop** (unique NE, measurable). Multiway
    seats may be requested later; T1 still *plays* 2–6 via net generalization.
    """
    rng = np.random.default_rng(int(seed))
    out: list[RootSample] = []
    seat_choices = list(seats)
    street_choices = list(streets)
    if not seat_choices or not street_choices:
        raise ValueError("seats and streets must be non-empty")

    for i in range(int(n)):
        if hu_heavy and 2 in seat_choices and float(rng.random()) < 0.85:
            n_seats = 2
        else:
            n_seats = int(seat_choices[int(rng.integers(0, len(seat_choices)))])
        street = int(street_choices[int(rng.integers(0, len(street_choices)))])
        band_name, spr = _draw_spr(rng)
        pot_bb = float(rng.uniform(pot_bb_range[0], pot_bb_range[1]))
        # Effective stacks from SPR ≈ min_stack / pot; allow mild asymmetry.
        base_stack = max(1.0, spr * pot_bb)
        stacks: list[float] = []
        for _ in range(n_seats):
            asym = float(rng.uniform(0.7, 1.3))
            stacks.append(round(base_stack * asym, 2))
        button = int(rng.integers(0, n_seats))
        out.append(
            RootSample(
                root_name=root.name,
                num_seats=n_seats,
                street=street,
                spr_band=band_name,
                stack_bb=tuple(stacks),
                pot_bb=round(pot_bb, 3),
                button=button,
                seed=int(rng.integers(0, 2**63 - 1)),
                spr=round(spr, 4),
            )
        )
    return out


def iter_spr_grid(
    *,
    root: ClubGGRoot = CLUBGG_NLH_ROOT,
    seats: int = 2,
    streets: Sequence[int] = (1, 2, 3),
    spr_points: Sequence[float] = (1.0, 3.0, 8.0, 15.0, 30.0),
    pot_bb: float = 10.0,
    seed0: int = 0,
) -> Iterator[RootSample]:
    """Deterministic SPR×street grid for probe / smoke solves."""
    for si, street in enumerate(streets):
        for sj, spr in enumerate(spr_points):
            stack = round(float(spr) * float(pot_bb), 2)
            yield RootSample(
                root_name=root.name,
                num_seats=seats,
                street=int(street),
                spr_band=_nearest_band(float(spr)),
                stack_bb=tuple([stack] * seats),
                pot_bb=float(pot_bb),
                button=0,
                seed=int(seed0) + si * 1000 + sj,
                spr=float(spr),
            )


def _nearest_band(spr: float) -> str:
    for name, lo, hi in SPR_STRATA:
        if lo <= spr < hi:
            return name
    return SPR_STRATA[-1][0]
