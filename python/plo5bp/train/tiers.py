"""Table-config sampling: one (seats, stacks) GameConfig per draw, per stack
tier ("clubgg", "clubgg_deep", "deep", ...). Training mixes these tiers every
update; the evaluation tools draw from the same functions so their tables are
the training tables."""

from __future__ import annotations

import numpy as np

from plo5bp.config import (
    VARIANT_NLH,
    VARIANT_PLO5,
    GameConfig,
)
from plo5bp.train.control import _warn_once


_VALID_STACK_DISTS = (
    "uniform", "clubgg", "clubgg_deep", "clubgg_mix", "deep", "full_mix",
    "clubgg_real",
)
# Every name `_sample_game_config` implements (the --stack-dist choices).
# --mix-tiers is validated against this, and the sampler itself raises on
# anything else (review 2026-09-20 A12): an unknown name used to fall through
# to the uniform --stack-range branch, so a `clubg_deep` typo silently trained
# on uniform 1-300bb (median 155bb instead of 52bb). (2026-09-28, ML-030: the
# retired NLH-PPO lineage's "nlh_topoff" tier and "agro_deep" -- = "deep" plus
# the retired aggression bonus -- are gone.)
_KNOWN_STACK_DISTS = _VALID_STACK_DISTS


# ClubGG-realistic per-seat stack bands (bb). Weights sum to 1.
# Reflects table conditions at $20/bb: most stacks hover 20-40bb after
# a few orbits; deep stacks (75bb+) present in ~50% of hands by
# independent per-seat sampling.
_CLUBGG_STACK_BANDS: tuple[tuple[float, float, float], ...] = (
    (1.0, 20.0, 0.05),    # Short: 1-20 bb
    (20.0, 40.0, 0.50),   # Hover: 20-40 bb (dominant)
    (40.0, 75.0, 0.18),   # Warm: 40-75 bb
    (75.0, 150.0, 0.17),  # Big: 75-150 bb
    (150.0, 300.0, 0.10), # Monster: 150-300 bb
)

# ClubGG "deep" per-seat stack bands (bb). Weights sum to 1.
# Models the $80-buy-in / $0.80-ante game, ~2x deeper than the $0.60
# game. Probability concentrated on 30-65bb (63%); 65-80bb seats
# expected ~1.3 per 6-handed table; minimal weight on <20bb; small
# 20-30bb tail for seats that lost a few hands without auto top-up.
_CLUBGG_DEEP_STACK_BANDS: tuple[tuple[float, float, float], ...] = (
    (1.0, 20.0, 0.02),    # Short: 1-20 bb
    (20.0, 30.0, 0.06),   # Lost-a-few: 20-30 bb
    (30.0, 40.0, 0.16),
    (40.0, 50.0, 0.22),
    (50.0, 65.0, 0.25),   # Mode
    (65.0, 80.0, 0.22),
    (80.0, 120.0, 0.07),
)

# ClubGG-realistic seat-count weights.
_CLUBGG_SEAT_WEIGHTS: dict[int, float] = {
    6: 0.30,
    5: 0.25,
    4: 0.25,
    3: 0.15,
    2: 0.10,
}

# "clubgg_real" (2026-10-03): the owner's OWN ClubGG tables, measured from their hand
# histories — 953 hands of PLO5 double-board bomb pots at $10/$20, 3bb ante, 6-max
# (aggregates only: nothing about any player is kept). Each seat's starting stack:
# 20bb is the buy-in / top-up point (14% of seats sit right on it, 10% are below
# it), with a long tail to ~350bb (clipped to --stack-range, default 300bb). The
# seat count comes with the tier (_CLUBGG_REAL_SEAT_WEIGHTS: the real tables are
# mostly 5-6 handed; --seats-dist does not apply to it), and every seat is drawn
# independently, which reproduces how the real tables MIX short and deep stacks: 58%
# of the real hands have a <=25bb seat and a >=100bb seat at once (the clubgg /
# clubgg_deep / deep mix: 10%). Median stack 43bb (that mix: 71bb); median
# stack-to-pot on the flop 2.5 (that mix: 5.3).
_CLUBGG_REAL_STACK_BANDS: tuple[tuple[float, float, float], ...] = (
    (1.0, 10.0, 0.012),
    (10.0, 19.5, 0.097),
    (19.5, 20.5, 0.136),   # the 20bb buy-in / top-up
    (20.5, 30.0, 0.127),
    (30.0, 40.0, 0.102),
    (40.0, 50.0, 0.080),
    (50.0, 65.0, 0.086),
    (65.0, 80.0, 0.073),
    (80.0, 100.0, 0.064),
    (100.0, 150.0, 0.114),
    (150.0, 250.0, 0.075),
    (250.0, 350.0, 0.034),
)
_CLUBGG_REAL_SEAT_WEIGHTS: dict[int, float] = {
    6: 0.415,
    5: 0.312,
    4: 0.209,
    3: 0.052,
    2: 0.012,
}

def _sample_clubgg_stack_bb(
    stack_lo_bb: float,
    stack_hi_bb: float,
    rng: np.random.Generator,
    bands: tuple[tuple[float, float, float], ...] = _CLUBGG_STACK_BANDS,
) -> float:
    # Pick a band by weight, then uniform within the band. Bands are
    # clipped to the [stack_lo_bb, stack_hi_bb] range; bands that fall
    # entirely outside the range contribute zero weight.
    weights = []
    ranges = []
    for lo, hi, w in bands:
        c_lo = max(lo, stack_lo_bb)
        c_hi = min(hi, stack_hi_bb)
        if c_hi > c_lo:
            weights.append(w)
            ranges.append((c_lo, c_hi))
    if not weights:
        return stack_lo_bb
    total = sum(weights)
    probs = [w / total for w in weights]
    idx = int(rng.choice(len(ranges), p=probs))
    lo, hi = ranges[idx]
    return float(rng.uniform(lo, hi))


def _sample_clubgg_seats(
    seats_choices: tuple[int, ...],
    rng: np.random.Generator,
    weights: dict[int, float] | None = None,
) -> int:
    # Restrict to the intersection of the weight table and user-supplied
    # seat range; renormalize. Seats not in the table fall back to
    # uniform probability across the remaining weighted seats so we
    # never silently drop them. Default table = ClubGG PLO weights.
    table = _CLUBGG_SEAT_WEIGHTS if weights is None else weights
    weights_l = [table.get(n, 0.0) for n in seats_choices]
    total = sum(weights_l)
    if total <= 0.0:
        return int(rng.choice(seats_choices))
    probs = [w / total for w in weights_l]
    return int(rng.choice(seats_choices, p=probs))


# Stack re-draws before `_sample_game_config` gives up and clamps (A1).
_STACK_RESAMPLE_TRIES = 32

def _sample_game_config(
    seats_choices: tuple[int, ...],
    stack_lo_bb: float,
    stack_hi_bb: float,
    bb: int,
    ante: int,
    rng: np.random.Generator,
    stack_dist: str = "uniform",
    seats_dist: str = "uniform",
    variant: str = VARIANT_PLO5,
    sb: int = 0,
) -> tuple[GameConfig, str]:
    if stack_dist == "clubgg_real":
        # (the real tables' seat counts come with their stacks)
        n_seats = _sample_clubgg_seats(seats_choices, rng, weights=_CLUBGG_REAL_SEAT_WEIGHTS)
    elif seats_dist == "clubgg":
        n_seats = _sample_clubgg_seats(seats_choices, rng)
    else:
        n_seats = int(rng.choice(seats_choices))

    if stack_dist not in _KNOWN_STACK_DISTS:
        # A12: never fall through to the uniform branch on a typo.
        raise ValueError(
            f"unknown stack_dist {stack_dist!r} — valid: {_KNOWN_STACK_DISTS}"
        )
    effective_stack_dist = stack_dist
    if stack_dist == "clubgg_mix":
        effective_stack_dist = "clubgg_deep" if rng.random() < 0.5 else "clubgg"
    elif stack_dist == "full_mix":
        effective_stack_dist = str(
            rng.choice(("clubgg", "clubgg_deep", "deep"))
        )

    def _draw_depths_bb() -> np.ndarray:
        if effective_stack_dist == "clubgg":
            return np.array(
                [_sample_clubgg_stack_bb(stack_lo_bb, stack_hi_bb, rng) for _ in range(n_seats)]
            )
        if effective_stack_dist == "clubgg_deep":
            return np.array(
                [
                    _sample_clubgg_stack_bb(
                        stack_lo_bb, stack_hi_bb, rng, bands=_CLUBGG_DEEP_STACK_BANDS
                    )
                    for _ in range(n_seats)
                ]
            )
        if effective_stack_dist == "deep":
            return rng.uniform(100.0, 250.0, size=n_seats)
        if effective_stack_dist == "clubgg_real":
            return np.array(
                [
                    _sample_clubgg_stack_bb(
                        stack_lo_bb, stack_hi_bb, rng, bands=_CLUBGG_REAL_STACK_BANDS
                    )
                    for _ in range(n_seats)
                ]
            )
        if stack_lo_bb == stack_hi_bb:
            return np.full(n_seats, stack_lo_bb)
        return rng.uniform(stack_lo_bb, stack_hi_bb, size=n_seats)

    # A hand needs at least TWO seats that can still act after posting, or
    # there is nothing to decide: a seat with stack <= ante is all-in on the
    # ante (NLH: <= ante + bb — it may also owe the big blind). With fewer
    # than two such seats the hand runs out at deal, and a config where that
    # is ALWAYS so gives the collector no row, ever — the batched loop span
    # forever on it (review 2026-09-20 A1; ~6e-5 per update with the clubgg
    # 1-20bb band, i.e. ~6% per 1,000 updates, silent under the guardians).
    # Exactly one live seat is the same problem in a milder form: a whole
    # sub-rollout of forced single-action rows. Resample the stacks (the
    # tier draw above is kept); the FIRST draw is the pre-fix one, so every
    # config that was already playable is byte-identical.
    live_floor = ante + (bb if variant == VARIANT_NLH else 0)
    for _ in range(_STACK_RESAMPLE_TRIES):
        depths_bb = _draw_depths_bb()
        stacks = tuple(int(round(float(d) * bb)) for d in depths_bb)
        if sum(s > live_floor for s in stacks) >= 2:
            break
    else:
        # The distribution itself sits at/below the ante (e.g. a fixed
        # --stack-range under 3bb): resampling cannot help. Lift the two
        # deepest seats to one bb behind after posting, loudly, rather than
        # hand the collector a config it must refuse.
        lift = sorted(range(n_seats), key=lambda i: stacks[i], reverse=True)[:2]
        stacks = tuple(
            max(s, live_floor + bb) if i in lift else s
            for i, s in enumerate(stacks)
        )
        _warn_once(
            f"[config] stack_dist={effective_stack_dist!r} range "
            f"{stack_lo_bb:g}:{stack_hi_bb:g}bb cannot seat two players with "
            f"more than {live_floor / bb:g}bb (ante"
            + ("+bb" if variant == VARIANT_NLH else "")
            + f") after {_STACK_RESAMPLE_TRIES} draws — CLAMPED the two "
            f"deepest seats to {(live_floor + bb) / bb:g}bb"
        )
    cfg = GameConfig(
        num_seats=n_seats,
        starting_stack=stacks[0],
        ante=ante,
        bb=bb,
        starting_stacks=stacks,
        sb=sb,
        variant=variant,
    )
    return cfg, effective_stack_dist
