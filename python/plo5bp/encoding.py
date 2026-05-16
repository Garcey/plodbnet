"""Observation encoding. Produces a fixed-length float32 vector from the
dict emitted by `PyGameState.observation_dict()` (augmented in `env._pack_obs`
with Rust-computed hand categories).

Layout (946 dims total):
  0..52     hero hole multi-hot (52)
  52..104   board A multi-hot (52)
  104..156  board B multi-hot (52)
  156..160  street one-hot (Preflop/Flop/Turn/River) — preflop always zero
  160..168  active mask, hero-rotated, padded to 8
  168..176  all-in mask, hero-rotated, padded to 8
  176..184  stacks / bb (stack-depth in bb), hero-rotated, padded to 8
  184..188  scalars: pot, bet_to_call, min_bet, max_bet — all / bb (bb units)
  188..196  relative-position one-hot of actor (actor - hero) mod num_seats
  196..740  history: last 32 actions oldest-first, each slot 17 dims
            (seat-one-hot hero-rel 8 + gate one-hot 4 + street one-hot 4 + chips/bb 1).
            Gate is {Fold=0, Check=1, Call=2, Raise=3} derived at encode time
            from (action, chips): CheckCall with chips==0 is Check, with chips>0
            is Call; any Bet*/AllIn is Raise. chips is the seat's street-total
            commit at the moment of the action, divided by cfg.bb.
  740..748  SPR per seat, hero-rotated, padded to 8 (stack/max(pot,1), clip[0,4])
  748..749  pot odds (to_call / (pot + to_call), 0 if no bet to face)
  749..758  hero hand category one-hot on board A (9 categories)
  758..767  hero hand category one-hot on board B (9 categories)
  767..769  hero draw flags on board A (flush, straight)
  769..771  hero draw flags on board B (flush, straight)
  771..776  pair-with-board count on A: count of hero hole cards matching
            the rank of the i-th board card (sorted by rank descending),
            5 slots, trailing zeros for streets < river
  776..781  pair-with-board count on B (same semantics)
  781..785  board A pair structure (paired, double_paired, tripled, quadded)
  785..789  board B pair structure (same semantics)
  789..802  hero rank histogram: slot r = count of hero hole cards at rank r
            (rank 0=2, 12=A). Board-agnostic; closes the pocket-pair-
            not-on-board blind spot left by the pair-with-board feature.
  802..840  straight/flush/SF block on board A (38 dims):
              802       flush_nut_distance       (0 if hero has no flush; uncapped)
              803       straight_nut_distance    (0 if hero has no straight; uncapped)
              804..814  straight_outs_per_window (10 dims, slot 0=wheel, 9=broadway)
              814..824  straight_possible_per_window (10 dims, board-only binary)
              824..828  flush_possible_per_suit  (4 dims, board-only binary)
              828..832  flush_draw_outs[s]       (4 dims, per suit; needs 2-2 split)
              832..836  nut_flush_draw_outs[s]   (4 dims; produces nut after hit)
              836..840  straight_flush_draw_outs[s] (4 dims; intersects flush + straight)
  840..878  straight/flush/SF block on board B (38 dims, same layout)
  878..886  hero-rotated seat-exists mask: slot k = 1 iff (hero + k) % num_seats
            is a real seat, else 0. Structural; doesn't depend on stack state,
            so a 0-chip seat is still distinguishable from a padded slot.
  886..894  per-seat hand-total commit, hero-rotated, /cfg.bb (raw, no clamp).
  894..902  per-seat street commit, hero-rotated, /cfg.bb (raw, no clamp).
  902..910  last-aggressor one-hot, hero-relative; all-zero when no raise yet.
  910..918  hero distance to button: one-hot of (button - hero) % num_seats.
  918..931  shared-rank mask: slot r = 1 iff rank r appears on BOTH boards.
  931..935  per-suit cross-board flush MADE on both boards: hero ≥2-of-s
            AND board_a ≥3-of-s AND board_b ≥3-of-s.
  935..939  per-suit cross-board flush DRAW on both boards: hero ≥2-of-s
            AND board_a 2-of-s AND board_b 2-of-s.
  939..943  per-suit cross-board flush MIXED: hero ≥2-of-s AND
            (one board ≥3-of-s, the other 2-of-s).
  943       cross-board straight MADE on both: ∃ pair {r1,r2} ⊆ hero ranks
            making a straight on A and on B (windows may differ).
  944       cross-board straight DRAW on both: ∃ pair drawing (4-rank
            coverage) on A and on B (and not made on either).
  945       cross-board straight MIXED: ∃ pair made on one, drawing on
            the other.
  946..958  opp-outcome fractions: 12 dims (3 hand sizes × 4 outcomes),
            row-major [k][outcome] for k ∈ {2, 3, 4}. Per k, the four
            outcomes are: opp scoops hero, opp quarters hero, hero
            scoops opp, hero quarters opp. Each entry is a fraction in
            [0, 1] of unseen-deck k-card opponent combos producing that
            outcome at the current board rank (PLO5 rule: exactly 2 from
            k + 3 from visible board, evaluated independently per board).
            Computed in Rust (`GameState::opp_outcome_fractions`); k=2,3
            exhaustive, k=4 MC-sampled (1024) with a deterministic seed
            from observation-visible state. All-zero pre-flop / terminal.
  958..959  bet-faced as fraction of pot-bet-into: to_call /
            max(pot - to_call, 1), clipped [0, 4]. 0 when no bet to
            face. The pot the bet was made into (i.e., pot before the
            facing bet); a "half-pot bet" reads as ~0.5, "pot bet" as
            ~1.0, "2x pot overbet" as ~2.0. Distinct from pot odds
            (to_call / (pot + to_call)) — raw bet-sizing the network
            can read directly without untangling a ratio.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping

import numpy as np

from plo5bp._engine import (  # type: ignore[attr-defined]
    cross_board_straight_batch as _rust_cross_board_straight,
    draw_flags_batch as _rust_draw_flags,
    pair_features_batch as _rust_pair_features,
    straight_flush_features_batch as _rust_sf_features,
)
from plo5bp.actions import CHECK_CALL, FOLD
from plo5bp.config import GameConfig

OBS_DIM: int = 959

_HOLE_OFF = 0
_BOARD_A_OFF = 52
_BOARD_B_OFF = 104
_STREET_OFF = 156
_ACTIVE_OFF = 160
_ALLIN_OFF = 168
_STACKS_OFF = 176
_SCALARS_OFF = 184
_REL_POS_OFF = 188
_HISTORY_OFF = 196
_HISTORY_DEPTH = 32
_HISTORY_SLOT_DIM = 17
_MAX_SEATS = 8
_NUM_STREET_ONEHOT = 4
_NUM_CATEGORIES = 9  # high-card..straight-flush

# Per-slot relative offsets within a history slot (sum = _HISTORY_SLOT_DIM).
_HISTORY_SEAT_OFF_REL = 0  # 8 dims (hero-relative seat one-hot)
_HISTORY_GATE_OFF_REL = 8  # 4 dims {Fold, Check, Call, Raise}
_HISTORY_STREET_OFF_REL = 12  # 4 dims (preflop, flop, turn, river)
_HISTORY_CHIPS_OFF_REL = 16  # 1 dim (chips / cfg.bb)

# Encoder-side gate enum (distinct from the policy gate which is 3-way:
# Fold/CheckCall/Raise). The encoder breaks CheckCall apart so the
# network sees check vs call explicitly.
_GATE_FOLD = 0
_GATE_CHECK = 1
_GATE_CALL = 2
_GATE_RAISE = 3

_SPR_OFF = 740
_POT_ODDS_OFF = 748
_CAT_A_OFF = 749
_CAT_B_OFF = 758
_DRAW_A_OFF = 767
_DRAW_B_OFF = 769
_PAIR_COUNT_A_OFF = 771
_PAIR_COUNT_B_OFF = 776
_BOARD_STRUCT_A_OFF = 781
_BOARD_STRUCT_B_OFF = 785
_HERO_RANK_HIST_OFF = 789

_FLUSH_NUT_DIST_A_OFF = 802
_STRAIGHT_NUT_DIST_A_OFF = 803
_STRAIGHT_OUTS_A_OFF = 804
_STRAIGHT_POSSIBLE_A_OFF = 814
_FLUSH_POSSIBLE_A_OFF = 824
_FLUSH_DRAW_OUTS_A_OFF = 828
_NUT_FLUSH_DRAW_OUTS_A_OFF = 832
_SF_DRAW_OUTS_A_OFF = 836

_FLUSH_NUT_DIST_B_OFF = 840
_STRAIGHT_NUT_DIST_B_OFF = 841
_STRAIGHT_OUTS_B_OFF = 842
_STRAIGHT_POSSIBLE_B_OFF = 852
_FLUSH_POSSIBLE_B_OFF = 862
_FLUSH_DRAW_OUTS_B_OFF = 866
_NUT_FLUSH_DRAW_OUTS_B_OFF = 870
_SF_DRAW_OUTS_B_OFF = 874

_SEAT_EXISTS_OFF = 878  # 8 dims; structural seat-presence, hero-rotated
_TOTAL_COMMIT_OFF = 886  # 8 dims; per-seat hand-total commit, hero-rotated, /bb
_STREET_COMMIT_OFF = 894  # 8 dims; per-seat street commit, hero-rotated, /bb
_LAST_AGGRESSOR_OFF = 902  # 8 dims; hero-rel one-hot of last aggressor (or all-zero)
_HERO_BTN_DIST_OFF = 910  # 8 dims; one-hot of (button - hero) % num_seats

_SHARED_RANKS_OFF = 918  # 13 dims; rank present on both A and B
_FLUSH_MADE_BOTH_OFF = 931  # 4 dims; per-suit hero-involved made on both
_FLUSH_DRAW_BOTH_OFF = 935  # 4 dims; per-suit hero-involved draw on both
_FLUSH_MIXED_OFF = 939  # 4 dims; per-suit hero-involved made on one + draw on other
_STRAIGHT_MADE_BOTH_OFF = 943  # 1 dim; same hero rank-pair makes straight on both
_STRAIGHT_DRAW_BOTH_OFF = 944  # 1 dim; same hero pair draws (4-rank cov) on both
_STRAIGHT_MIXED_OFF = 945  # 1 dim; same hero pair made on one, drawing on other

_OPP_OUTCOME_OFF = 946  # 12 dims; [k=2,3,4][outcome] fractions in [0,1]
_OPP_OUTCOME_DIM = 12
_BET_PCT_POT_OFF = 958  # 1 dim; to_call / max(pot, 1), clipped [0, 4]

# 10 straight windows: slot 0 = wheel (A,2,3,4,5); slots 1..9 = consecutive
# 5-rank windows starting at rank 0..8. Slot 9 = broadway (T,J,Q,K,A).
_STRAIGHT_WINDOWS: tuple[frozenset[int], ...] = (
    frozenset({12, 0, 1, 2, 3}),
    frozenset({0, 1, 2, 3, 4}),
    frozenset({1, 2, 3, 4, 5}),
    frozenset({2, 3, 4, 5, 6}),
    frozenset({3, 4, 5, 6, 7}),
    frozenset({4, 5, 6, 7, 8}),
    frozenset({5, 6, 7, 8, 9}),
    frozenset({6, 7, 8, 9, 10}),
    frozenset({7, 8, 9, 10, 11}),
    frozenset({8, 9, 10, 11, 12}),
)


def _cross_board_straight(
    hole_idx: list[int], board_a_idx: list[int], board_b_idx: list[int]
) -> tuple[float, float, float]:
    """Return (made_both, draw_both, mixed) cross-board straight indicators.

    For each window W, iterate every 2-rank subset of hero's ranks in W and
    classify the (board_a, board_b) coverage:
      - cov = |pair ∪ board_ranks_in_W| ; cov == 5 ⇒ made, cov == 4 ⇒ draw.
    A pair is collected per (board, state); a pair that fires "made" in
    some W on board A wins over a "draw" in another W (so a pair is a
    "made_a" pair if any window makes it). The same hero rank-pair can
    use different windows on each board — captures the JT freeroll across
    KQx / Q9x / 98x.
    """
    if not board_a_idx or not board_b_idx:
        return 0.0, 0.0, 0.0
    hero_ranks = {c // 4 for c in hole_idx}
    ba_ranks = {c // 4 for c in board_a_idx}
    bb_ranks = {c // 4 for c in board_b_idx}

    pair_made_a: set[frozenset[int]] = set()
    pair_draw_a: set[frozenset[int]] = set()
    pair_made_b: set[frozenset[int]] = set()
    pair_draw_b: set[frozenset[int]] = set()

    for W in _STRAIGHT_WINDOWS:
        H_W = hero_ranks & W
        if len(H_W) < 2:
            continue
        ba_W = ba_ranks & W
        bb_W = bb_ranks & W
        for r1, r2 in combinations(sorted(H_W), 2):
            pair = frozenset((r1, r2))
            cov_a = len(pair | ba_W)
            cov_b = len(pair | bb_W)
            if cov_a >= 5:
                pair_made_a.add(pair)
            elif cov_a == 4:
                pair_draw_a.add(pair)
            if cov_b >= 5:
                pair_made_b.add(pair)
            elif cov_b == 4:
                pair_draw_b.add(pair)

    # A pair that made on one board "wins" over a draw on the same board
    # only when collapsing per-pair. For cross-board indicators we keep
    # made and draw sets separate per board so a pair can be in
    # made_a ∩ draw_b for the mixed case.
    made_both = bool(pair_made_a & pair_made_b)
    draw_both = bool((pair_draw_a - pair_made_a) & (pair_draw_b - pair_made_b))
    mixed_pairs = (pair_made_a & (pair_draw_b - pair_made_b)) | (
        pair_made_b & (pair_draw_a - pair_made_a)
    )
    mixed = bool(mixed_pairs)
    return float(made_both), float(draw_both), float(mixed)


# Bitmasks reused by `_cross_board_straight_batch`. Built at import.
_PAIR_BITS_13 = np.array(
    [(1 << r1) | (1 << r2) for r1 in range(13) for r2 in range(r1 + 1, 13)],
    dtype=np.uint16,
)  # (78,) — every 2-rank subset of {0..12}.
_WINDOW_BITS_13 = np.array(
    [sum(1 << r for r in W) for W in _STRAIGHT_WINDOWS],
    dtype=np.uint16,
)  # (10,)
_RANK_WEIGHTS_13 = (1 << np.arange(13, dtype=np.uint16)).astype(np.uint16)


def _cross_board_straight_batch(
    hole_rank_mask: np.ndarray,
    ba_rank_mask: np.ndarray,
    bb_rank_mask: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized `_cross_board_straight` over all envs.

    Inputs are `(n, 13)` bool rank-presence masks plus `(n,)` bool gate
    that is True only when both boards have at least one visible card
    (matches the scalar's `if not board_a_idx or not board_b_idx`
    short-circuit). Returns three `(n,)` float32 arrays:
    `(made_both, draw_both, mixed)`.

    Pairs are encoded as 78 13-bit masks; windows as 10 13-bit masks.
    `np.bitwise_count` lets `cov` reduce to a popcount over `pair |
    (board & window)`, so the per-env / per-window / per-pair classifier
    is a few `(n, 78)` boolean reductions.
    """
    n = hole_rank_mask.shape[0]
    hero_bits = (hole_rank_mask.astype(np.uint16) * _RANK_WEIGHTS_13).sum(axis=1).astype(np.uint16)
    ba_bits = (ba_rank_mask.astype(np.uint16) * _RANK_WEIGHTS_13).sum(axis=1).astype(np.uint16)
    bb_bits = (bb_rank_mask.astype(np.uint16) * _RANK_WEIGHTS_13).sum(axis=1).astype(np.uint16)

    hero_has_pair = (hero_bits[:, None] & _PAIR_BITS_13[None, :]) == _PAIR_BITS_13[None, :]
    made_a = np.zeros((n, _PAIR_BITS_13.shape[0]), dtype=bool)
    draw_a = np.zeros((n, _PAIR_BITS_13.shape[0]), dtype=bool)
    made_b = np.zeros((n, _PAIR_BITS_13.shape[0]), dtype=bool)
    draw_b = np.zeros((n, _PAIR_BITS_13.shape[0]), dtype=bool)
    for w_idx in range(_WINDOW_BITS_13.shape[0]):
        W = _WINDOW_BITS_13[w_idx]
        pair_in_W = (_PAIR_BITS_13 & W) == _PAIR_BITS_13
        if not pair_in_W.any():
            continue
        ba_W = (ba_bits & W).astype(np.uint16)
        bb_W = (bb_bits & W).astype(np.uint16)
        cov_a = np.bitwise_count(ba_W[:, None] | _PAIR_BITS_13[None, :])
        cov_b = np.bitwise_count(bb_W[:, None] | _PAIR_BITS_13[None, :])
        gate = hero_has_pair & pair_in_W[None, :]
        made_a |= (cov_a >= 5) & gate
        draw_a |= (cov_a == 4) & gate
        made_b |= (cov_b >= 5) & gate
        draw_b |= (cov_b == 4) & gate

    draw_only_a = draw_a & ~made_a
    draw_only_b = draw_b & ~made_b
    made_both = (made_a & made_b).any(axis=1) & valid
    draw_both = (draw_only_a & draw_only_b).any(axis=1) & valid
    mixed = ((made_a & draw_only_b) | (made_b & draw_only_a)).any(axis=1) & valid
    return (
        made_both.astype(np.float32),
        draw_both.astype(np.float32),
        mixed.astype(np.float32),
    )


def _cross_board_features(
    hole_idx: list[int], board_a_idx: list[int], board_b_idx: list[int]
) -> np.ndarray:
    """Return a (28,) cross-board feature vector.

    Layout matches the offsets `_SHARED_RANKS_OFF`..`_STRAIGHT_MIXED_OFF`
    relative to slot 0:
      0..13   shared rank mask (rank present on both boards).
      13..17  per-suit flush MADE on both (hero ≥2-of-s, both boards ≥3-of-s).
      17..21  per-suit flush DRAW on both (hero ≥2-of-s, both boards 2-of-s).
      21..25  per-suit flush MIXED (hero ≥2-of-s, one ≥3-of-s, the other 2).
      25..28  cross-board straight indicators (made_both, draw_both, mixed).
    """
    out = np.zeros(28, dtype=np.float32)
    if not board_a_idx or not board_b_idx:
        return out

    ba_ranks = {c // 4 for c in board_a_idx}
    bb_ranks = {c // 4 for c in board_b_idx}
    shared = ba_ranks & bb_ranks
    for r in shared:
        out[r] = 1.0

    hole_suit = [0, 0, 0, 0]
    ba_suit = [0, 0, 0, 0]
    bb_suit = [0, 0, 0, 0]
    for c in hole_idx:
        hole_suit[c % 4] += 1
    for c in board_a_idx:
        ba_suit[c % 4] += 1
    for c in board_b_idx:
        bb_suit[c % 4] += 1
    for s in range(4):
        if hole_suit[s] < 2:
            continue
        a3 = ba_suit[s] >= 3
        b3 = bb_suit[s] >= 3
        a2 = ba_suit[s] == 2
        b2 = bb_suit[s] == 2
        if a3 and b3:
            out[13 + s] = 1.0
        elif a2 and b2:
            out[17 + s] = 1.0
        elif (a3 and b2) or (a2 and b3):
            out[21 + s] = 1.0

    made_both, draw_both, mixed = _cross_board_straight(
        hole_idx, board_a_idx, board_b_idx
    )
    out[25] = made_both
    out[26] = draw_both
    out[27] = mixed
    return out


def _gate_from_action(action: int, chips: int) -> int:
    """Map the engine's 8-action enum + chips to the encoder's 4-way gate.

    Fold → 0; CheckCall with chips==0 → 1 (Check); CheckCall with chips>0
    → 2 (Call); any Bet*/AllIn → 3 (Raise). Distinct from the 3-way
    policy gate; this is encoder-only so the network can tell a check
    apart from a 0-chip "call" of nothing.
    """
    if action == FOLD:
        return _GATE_FOLD
    if action == CHECK_CALL:
        return _GATE_CHECK if chips == 0 else _GATE_CALL
    return _GATE_RAISE


def _draw_flags(hole_idx: list[int], board_idx: list[int]) -> tuple[float, float]:
    """Return (flush_draw, straight_draw) flags for hero's best draw on a board.

    Flush draw: hero has ≥2 cards of a suit AND board has exactly 2 of that
    suit (total 4 → one more completes). On river the draw is useless but
    still reported; encoding callers may gate by street.

    Straight draw: the union of hero's ranks and board's ranks contains 4
    consecutive ranks (including wheel A-2-3-4). Coarse: doesn't verify
    that the 4 cards respect the 2-hole + 2-board split.
    """
    if not board_idx:
        return 0.0, 0.0
    # Suits
    hole_suits = [c % 4 for c in hole_idx]
    board_suits = [c % 4 for c in board_idx]
    flush = 0.0
    for s in range(4):
        hs = sum(1 for x in hole_suits if x == s)
        bs = sum(1 for x in board_suits if x == s)
        if hs >= 2 and bs == 2:
            flush = 1.0
            break
    # Ranks (0..=12; ace = 12, wheel handled via bit 13)
    rank_set = 0
    for c in hole_idx:
        rank_set |= 1 << (c // 4)
    for c in board_idx:
        rank_set |= 1 << (c // 4)
    # Add "ace-low" bit 13 if ace present.
    if rank_set & (1 << 12):
        rank_set |= 1 << 13  # shadow bit so A-2-3-4 window works
    straight = 0.0
    for start in range(11):  # windows 0..3, 1..4, ..., 10..13
        window = ((1 << 4) - 1) << start
        if bin(rank_set & window).count("1") >= 4:
            straight = 1.0
            break
    return flush, straight


def _pair_features(
    hole_idx: list[int], board_idx: list[int]
) -> tuple[tuple[float, float, float, float, float], tuple[float, float, float, float]]:
    """Per-board pair-with-board features.

    Returns:
      counts: 5-tuple. counts[i] = number of hero hole cards matching the
        rank of the i-th board card (board cards sorted by rank descending).
        Trailing slots are 0 when fewer than 5 board cards are visible.
        On paired boards, slots tied on rank carry the same value (both
        the K-slots on flop K-K-9 hold the count of hero K's).
      struct: 4-tuple (paired, double_paired, tripled, quadded). Monotonic.
    """
    if not board_idx:
        return (0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)

    board_rank_counts = [0] * 13
    hero_rank_counts = [0] * 13
    for c in board_idx:
        board_rank_counts[c // 4] += 1
    for c in hole_idx:
        hero_rank_counts[c // 4] += 1

    # Sort board card ranks descending (each board card contributes one
    # entry — paired boards repeat ranks).
    sorted_ranks = sorted((c // 4 for c in board_idx), reverse=True)
    counts = [0.0] * 5
    for i, rank in enumerate(sorted_ranks[:5]):
        counts[i] = float(hero_rank_counts[rank])

    paired = float(any(b >= 2 for b in board_rank_counts))
    double_paired = float(sum(1 for b in board_rank_counts if b >= 2) >= 2)
    tripled = float(any(b >= 3 for b in board_rank_counts))
    quadded = float(any(b >= 4 for b in board_rank_counts))

    return (
        (counts[0], counts[1], counts[2], counts[3], counts[4]),
        (paired, double_paired, tripled, quadded),
    )


def _straight_flush_features(
    hole_idx: list[int], board_idx: list[int], visible_count: np.ndarray
) -> np.ndarray:
    """Per-board straight / flush / straight-flush features. Returns (38,) f32.

    `visible_count` is a (13, 4) int array with `visible_count[r, s] == 1` iff
    card `(rank=r, suit=s)` is on hero_hole, board_a, or board_b — the global
    cross-board visibility table. Cards visible anywhere are excluded from
    out counts and from "unaccounted" blocker tallies.

    Returned 38-dim layout (offsets relative to the start of the slice):
       0       flush_nut_distance         (count of unaccounted higher suit-s ranks; 0 if no flush)
       1       straight_nut_distance      (count of higher possible windows; 0 if no straight)
       2..12   straight_outs_per_window   (10 dims; slot 0=wheel, 9=broadway)
       12..22  straight_possible_per_window (board-only: ≥3 distinct ranks of W on board)
       22..26  flush_possible_per_suit    (board-only: ≥3 of suit s on board)
       26..30  flush_draw_outs[s]         (per suit, single-card flush hits remaining)
       30..34  nut_flush_draw_outs[s]     (subset of flush_draw_outs that produce nut)
       34..38  straight_flush_draw_outs[s] (per suit; (rank,suit) hits completing both
                                            a flush and a straight in some window)

    PLO 2+3 enforcement: hero must use exactly 2 distinct hole-card ranks
    contributing to the straight/SF; H_W is the SET of distinct hero ranks
    in window W (so hero with KKxx can't use both kings toward broadway).
    """
    out = np.zeros(38, dtype=np.float32)
    if not board_idx:
        return out

    board_ranks_set = {c // 4 for c in board_idx}
    hole_ranks_set = {c // 4 for c in hole_idx}

    board_suit_counts = [0] * 4
    hole_suit_counts = [0] * 4
    for c in board_idx:
        board_suit_counts[c % 4] += 1
    for c in hole_idx:
        hole_suit_counts[c % 4] += 1

    vct = visible_count.sum(axis=1)  # (13,) — global visibility per rank

    # Per-suit max hole rank (-1 if hero has no card of that suit).
    hole_max_per_suit = [-1, -1, -1, -1]
    for c in hole_idx:
        s = c % 4
        r = c // 4
        if r > hole_max_per_suit[s]:
            hole_max_per_suit[s] = r

    # Per-suit hole ranks (sets) for SF window evaluation.
    hole_ranks_per_suit: list[set[int]] = [set(), set(), set(), set()]
    board_ranks_per_suit: list[set[int]] = [set(), set(), set(), set()]
    for c in hole_idx:
        hole_ranks_per_suit[c % 4].add(c // 4)
    for c in board_idx:
        board_ranks_per_suit[c % 4].add(c // 4)

    # ---- Per-window straight features ----
    makes_window = [False] * 10
    straight_outs = [0] * 10
    straight_possible = [0] * 10
    # Cache the candidate-rank sets per window for SF reuse.
    straight_out_cands: list[set[int]] = [set() for _ in range(10)]

    for w_i, W in enumerate(_STRAIGHT_WINDOWS):
        B_W = W & board_ranks_set
        H_W = W & hole_ranks_set
        L = W - B_W
        nL = len(L)
        nH = len(H_W)
        nB = len(B_W)
        straight_possible[w_i] = 1 if nB >= 3 else 0
        # Hero already makes the straight in W?
        makes = (L <= H_W) and (nH >= 2) and (nL <= 2)
        makes_window[w_i] = makes
        if makes or nH < 2:
            continue
        M = L - H_W  # ranks needed but not in hero hole
        nM = len(M)
        if nL == 3 and nM == 0:
            # All 3 missing-from-board ranks are in hero hole; any board hit
            # of a rank in L drops |L'| to 2 with L' ⊆ H_W → makes.
            cands = set(L)
            straight_out_cands[w_i] = cands
            straight_outs[w_i] = sum(4 - int(vct[r]) for r in cands)
        elif nM == 1 and 1 <= nL <= 3:
            # The one missing rank that, when on board, completes the straight.
            r_star = next(iter(M))
            cands = {r_star}
            straight_out_cands[w_i] = cands
            straight_outs[w_i] = 4 - int(vct[r_star])
        # else: 0 outs (≥2 ranks missing from hero hole, or |L| ≥ 4).

    # ---- Straight nut distance ----
    if any(makes_window):
        h_max = max(i for i, m in enumerate(makes_window) if m)
        straight_nut_distance = sum(
            straight_possible[w] for w in range(h_max + 1, 10)
        )
    else:
        straight_nut_distance = 0

    # ---- Flush features ----
    flush_possible = [1.0 if board_suit_counts[s] >= 3 else 0.0 for s in range(4)]

    # Made flush (at most one suit can satisfy board_suit_counts[s] >= 3).
    flush_nut_distance = 0
    for s in range(4):
        if board_suit_counts[s] >= 3 and hole_suit_counts[s] >= 2:
            h1 = hole_max_per_suit[s]
            for r in range(h1 + 1, 13):
                if visible_count[r, s] == 0:
                    flush_nut_distance += 1
            break

    # Per-suit flush draw outs, nut flush draw outs, SF draw outs.
    flush_draw_outs = [0] * 4
    nut_flush_draw_outs = [0] * 4
    sf_draw_outs = [0] * 4
    for s in range(4):
        if not (hole_suit_counts[s] >= 2 and board_suit_counts[s] == 2):
            continue
        visible_s = int(visible_count[:, s].sum())
        flush_draw_outs[s] = 13 - visible_s
        h1 = hole_max_per_suit[s]
        blockers = sum(
            1 for r in range(h1 + 1, 13) if visible_count[r, s] == 0
        )
        if blockers == 0:
            nut_flush_draw_outs[s] = flush_draw_outs[s]
        elif blockers == 1:
            nut_flush_draw_outs[s] = 1
        else:
            nut_flush_draw_outs[s] = 0

        # SF draw outs: (r, s) cards that complete both a flush AND a SF.
        # SF in window W requires (W \ B_s) ⊆ H_s, |H_s ∩ W| ≥ 2, |W \ B_s| ≤ 2
        # AFTER adding (r, s) to the suit-restricted board ranks B_s.
        H_s = hole_ranks_per_suit[s]
        B_s = board_ranks_per_suit[s]
        sf_cands: set[int] = set()
        for W in _STRAIGHT_WINDOWS:
            B_s_W = W & B_s
            H_s_W = W & H_s
            L_s = W - B_s_W
            nL_s = len(L_s)
            nH_s = len(H_s_W)
            already = (L_s <= H_s_W) and (nH_s >= 2) and (nL_s <= 2)
            if already or nH_s < 2:
                continue
            M_s = L_s - H_s_W
            nM_s = len(M_s)
            if nL_s == 3 and nM_s == 0:
                sf_cands.update(L_s)
            elif nM_s == 1 and 1 <= nL_s <= 3:
                sf_cands.add(next(iter(M_s)))
        sf_draw_outs[s] = sum(
            1 for r in sf_cands if visible_count[r, s] == 0
        )

    # ---- Pack output ----
    out[0] = float(flush_nut_distance)
    out[1] = float(straight_nut_distance)
    for i, x in enumerate(straight_outs):
        out[2 + i] = float(x)
    for i, x in enumerate(straight_possible):
        out[12 + i] = float(x)
    for i, x in enumerate(flush_possible):
        out[22 + i] = x
    for i, x in enumerate(flush_draw_outs):
        out[26 + i] = float(x)
    for i, x in enumerate(nut_flush_draw_outs):
        out[30 + i] = float(x)
    for i, x in enumerate(sf_draw_outs):
        out[34 + i] = float(x)
    return out


def encode_observation(obs: Mapping[str, Any], config: GameConfig) -> np.ndarray:
    """Encode a single observation dict into a (886,) float32 array."""
    out = np.zeros(OBS_DIM, dtype=np.float32)
    num_seats = config.num_seats
    hero = obs["actor"]
    if hero is None:
        return out

    for idx in obs["hero_hole"]:
        out[_HOLE_OFF + int(idx)] = 1.0
    for idx in obs["board_a"]:
        out[_BOARD_A_OFF + int(idx)] = 1.0
    for idx in obs["board_b"]:
        out[_BOARD_B_OFF + int(idx)] = 1.0

    street_idx = int(obs["street"])
    if 0 <= street_idx < _NUM_STREET_ONEHOT:
        out[_STREET_OFF + street_idx] = 1.0

    folded = obs["folded"]
    all_in = obs["all_in"]
    stacks = obs["stacks"]
    eff_cap = obs["eff_stack_cap"]
    starting = config.resolved_stacks
    inv_bb = 1.0 / float(config.bb)
    # Effective remaining: subtract the "dead" portion of starting stack
    # that's above max-other-reachable. dead = max(0, starting - eff_cap),
    # frozen at hand start. For non-deep seats dead == 0 and effective
    # equals own_remaining; for the deepest seat it removes both the
    # unreachable starting chips and (since remaining = starting - committed)
    # any chips already committed beyond the reachable cap. The network is
    # invariant to chips above max-other-reachable.
    dead_chips = [
        max(0, int(starting[s]) - int(eff_cap[s])) for s in range(num_seats)
    ]
    eff_per_seat = [
        max(0.0, float(stacks[s]) - float(dead_chips[s])) for s in range(num_seats)
    ]
    for k in range(num_seats):
        seat = (hero + k) % num_seats
        if not folded[seat]:
            out[_ACTIVE_OFF + k] = 1.0
        if all_in[seat]:
            out[_ALLIN_OFF + k] = 1.0
        out[_STACKS_OFF + k] = eff_per_seat[seat] * inv_bb

    pot = float(obs["pot"])
    btc = float(obs["bet_to_call"])
    out[_SCALARS_OFF + 0] = pot * inv_bb
    out[_SCALARS_OFF + 1] = btc * inv_bb
    out[_SCALARS_OFF + 2] = float(obs["min_bet"]) * inv_bb
    out[_SCALARS_OFF + 3] = float(obs["max_bet"]) * inv_bb

    actor_rel = (hero - hero) % num_seats
    out[_REL_POS_OFF + actor_rel] = 1.0

    history = obs["history"]
    if len(history) > _HISTORY_DEPTH:
        history = history[-_HISTORY_DEPTH:]
    for slot, (seat, action, chips, street_idx) in enumerate(history):
        base = _HISTORY_OFF + slot * _HISTORY_SLOT_DIM
        rel_seat = (seat - hero) % num_seats
        out[base + _HISTORY_SEAT_OFF_REL + rel_seat] = 1.0
        gate = _gate_from_action(int(action), int(chips))
        out[base + _HISTORY_GATE_OFF_REL + gate] = 1.0
        s_idx = int(street_idx)
        if 0 <= s_idx < _NUM_STREET_ONEHOT:
            out[base + _HISTORY_STREET_OFF_REL + s_idx] = 1.0
        out[base + _HISTORY_CHIPS_OFF_REL] = float(chips) * inv_bb

    # SPR uses the same effective remaining as _STACKS_OFF.
    pot_safe = max(pot, 1.0)
    for k in range(num_seats):
        seat = (hero + k) % num_seats
        spr = eff_per_seat[seat] / pot_safe
        out[_SPR_OFF + k] = min(max(spr, 0.0), 4.0)

    # Pot odds.
    street_commit = obs.get("street_commit", [0] * num_seats)
    hero_street_commit = float(street_commit[hero]) if hero < len(street_commit) else 0.0
    to_call = max(btc - hero_street_commit, 0.0)
    if to_call > 0.0:
        out[_POT_ODDS_OFF] = to_call / (pot + to_call)

    # Bet-faced as fraction of pot-bet-into. Half-pot bet → ~0.5; pot bet
    # → ~1.0; overbet > 1.0. Distinct framing from pot odds.
    if to_call > 0.0:
        pot_before_bet = max(pot - to_call, 1.0)
        out[_BET_PCT_POT_OFF] = min(to_call / pot_before_bet, 4.0)

    # Hand category one-hots (populated by env._pack_obs).
    cat_a = int(obs.get("hero_category_a", 0))
    cat_b = int(obs.get("hero_category_b", 0))
    if 0 <= cat_a < _NUM_CATEGORIES:
        out[_CAT_A_OFF + cat_a] = 1.0
    if 0 <= cat_b < _NUM_CATEGORIES:
        out[_CAT_B_OFF + cat_b] = 1.0

    # Draw flags.
    hole_list = [int(x) for x in obs["hero_hole"]]
    board_a_list = [int(x) for x in obs["board_a"]]
    board_b_list = [int(x) for x in obs["board_b"]]
    f_a, s_a = _draw_flags(hole_list, board_a_list)
    f_b, s_b = _draw_flags(hole_list, board_b_list)
    out[_DRAW_A_OFF + 0] = f_a
    out[_DRAW_A_OFF + 1] = s_a
    out[_DRAW_B_OFF + 0] = f_b
    out[_DRAW_B_OFF + 1] = s_b

    # Pair-with-board counts + board pair structure.
    counts_a, struct_a = _pair_features(hole_list, board_a_list)
    counts_b, struct_b = _pair_features(hole_list, board_b_list)
    for i in range(5):
        out[_PAIR_COUNT_A_OFF + i] = counts_a[i]
        out[_PAIR_COUNT_B_OFF + i] = counts_b[i]
    for i in range(4):
        out[_BOARD_STRUCT_A_OFF + i] = struct_a[i]
        out[_BOARD_STRUCT_B_OFF + i] = struct_b[i]

    # Hero rank histogram — board-agnostic, surfaces pocket pairs / trips /
    # quads of any rank (including ones not on either board).
    for c in hole_list:
        out[_HERO_RANK_HIST_OFF + (c // 4)] += 1.0

    # Straight / flush / SF block — needs cross-board visibility.
    visible_count = np.zeros((13, 4), dtype=np.int32)
    for c in hole_list:
        visible_count[c // 4, c % 4] = 1
    for c in board_a_list:
        visible_count[c // 4, c % 4] = 1
    for c in board_b_list:
        visible_count[c // 4, c % 4] = 1
    sf_a = _straight_flush_features(hole_list, board_a_list, visible_count)
    sf_b = _straight_flush_features(hole_list, board_b_list, visible_count)
    out[_FLUSH_NUT_DIST_A_OFF : _FLUSH_NUT_DIST_A_OFF + 38] = sf_a
    out[_FLUSH_NUT_DIST_B_OFF : _FLUSH_NUT_DIST_B_OFF + 38] = sf_b

    # Structural seat-exists mask. Decoupled from stack/active state so a
    # 0-chip or fully-folded seat still reads as "this slot is a real seat".
    for k in range(num_seats):
        out[_SEAT_EXISTS_OFF + k] = 1.0

    # Per-seat commits (hand-total + street), hero-rotated. street_commit was
    # already loaded above for the pot-odds block; reuse the local. Values are
    # raw chips/bb, no clamp (matches _STACKS_OFF convention).
    total_commit = obs["total_commit"]
    for k in range(num_seats):
        seat = (hero + k) % num_seats
        out[_TOTAL_COMMIT_OFF + k] = float(total_commit[seat]) * inv_bb
        out[_STREET_COMMIT_OFF + k] = float(street_commit[seat]) * inv_bb

    # Last aggressor (hero-relative one-hot); all-zero when no raise yet.
    last_agg = int(obs.get("last_aggressor", -1))
    if 0 <= last_agg < num_seats:
        out[_LAST_AGGRESSOR_OFF + (last_agg - hero) % num_seats] = 1.0

    # Hero distance to button.
    button = int(obs["button"])
    out[_HERO_BTN_DIST_OFF + (button - hero) % num_seats] = 1.0

    # Cross-board interactions (shared ranks + hero-involved flush/straight
    # coupling). Reuses the hole/board lists already built for the SF block.
    cross = _cross_board_features(hole_list, board_a_list, board_b_list)
    out[_SHARED_RANKS_OFF : _SHARED_RANKS_OFF + 28] = cross

    # Opp-outcome fractions (12 dims, [k=2,3,4][scoop_opp, quarter_opp,
    # scoop_hero, quarter_hero]). Computed in the Rust engine and surfaced
    # via observation_dict. Pre-flop / terminal states yield zeros.
    opp_fr = obs.get("opp_outcome_fractions")
    if opp_fr is not None:
        out[_OPP_OUTCOME_OFF : _OPP_OUTCOME_OFF + _OPP_OUTCOME_DIM] = np.asarray(
            opp_fr, dtype=np.float32
        )

    return out


# -----------------------------------------------------------------------------
# Vectorized encoder (Phase B).
#
# Consumes the stacked arrays emitted by `PyBatchedEngine.observation_arrays()`
# and the per-env category arrays (computed separately via
# `hero_category_batch` against the batched engine's current-actor seats).
# Produces an (N, 918) float32 identical to calling `encode_observation`
# once per env.
#
# Bit-exact parity with `encode_observation` is enforced by a golden test in
# `tests/python/test_encoding_batch.py`. Do NOT change a dim layout here
# without mirroring it in the scalar path — they must stay in lockstep.
# -----------------------------------------------------------------------------


def _draw_flags_batch(
    hole: np.ndarray, board: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized `_draw_flags`. `hole` is (N, 5) u8 with 255 sentinels for
    empty slots; `board` is (N, 5) u8 with 255 sentinels. Returns
    `(flush: (N,) f32, straight: (N,) f32)`.

    Semantics mirror the scalar path exactly:
    - flush: some suit has ≥2 in hole AND exactly 2 on board.
    - straight: the rank-union (including ace-low shadow) contains any 4
      consecutive ranks.
    - Returns (0, 0) when the board has no cards (all sentinels).
    """
    n = hole.shape[0]
    hole_valid = hole < 52
    board_valid = board < 52

    hole_suits = (hole & 3).astype(np.int64)
    board_suits = (board & 3).astype(np.int64)
    hole_suit_counts = np.zeros((n, 4), dtype=np.int32)
    board_suit_counts = np.zeros((n, 4), dtype=np.int32)
    for k in range(5):
        vh = hole_valid[:, k]
        if vh.any():
            np.add.at(
                hole_suit_counts,
                (np.nonzero(vh)[0], hole_suits[vh, k]),
                1,
            )
        vb = board_valid[:, k]
        if vb.any():
            np.add.at(
                board_suit_counts,
                (np.nonzero(vb)[0], board_suits[vb, k]),
                1,
            )
    flush_per_suit = (hole_suit_counts >= 2) & (board_suit_counts == 2)
    flush = flush_per_suit.any(axis=1)

    hole_ranks = (hole >> 2).astype(np.int64)
    board_ranks = (board >> 2).astype(np.int64)
    rank_mask = np.zeros((n, 14), dtype=bool)
    for k in range(5):
        vh = hole_valid[:, k]
        if vh.any():
            rank_mask[np.nonzero(vh)[0], hole_ranks[vh, k]] = True
        vb = board_valid[:, k]
        if vb.any():
            rank_mask[np.nonzero(vb)[0], board_ranks[vb, k]] = True
    # Ace-low shadow: bit 13 follows bit 12 (ace).
    rank_mask[:, 13] |= rank_mask[:, 12]
    straight = np.zeros(n, dtype=bool)
    for start in range(11):
        straight |= rank_mask[:, start : start + 4].all(axis=1)

    board_has_cards = board_valid.any(axis=1)
    flush_f = np.where(board_has_cards, flush, False).astype(np.float32)
    straight_f = np.where(board_has_cards, straight, False).astype(np.float32)
    return flush_f, straight_f


def _pair_features_batch(
    hole: np.ndarray, board: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized `_pair_features`. `hole` is (N, 5) u8 with 255 sentinels;
    `board` is (N, 5) u8 with 255 sentinels.

    Returns:
      counts: (N, 5) float32. counts[n, i] = number of hero hole cards
        in env n matching the rank of the i-th-highest valid board card.
        Trailing slots are 0 when fewer than 5 board cards are visible.
      struct: (N, 4) float32. (paired, double_paired, tripled, quadded)
        per env. Returns all zeros when the board has no cards.
    """
    n = hole.shape[0]
    hole_valid = hole < 52
    board_valid = board < 52

    # Per-rank counts on hole and board, shape (N, 13).
    hole_ranks = (hole >> 2).astype(np.int64)
    board_ranks = (board >> 2).astype(np.int64)
    hole_rank_counts = np.zeros((n, 13), dtype=np.int32)
    board_rank_counts = np.zeros((n, 13), dtype=np.int32)
    for k in range(5):
        vh = hole_valid[:, k]
        if vh.any():
            np.add.at(
                hole_rank_counts,
                (np.nonzero(vh)[0], hole_ranks[vh, k]),
                1,
            )
        vb = board_valid[:, k]
        if vb.any():
            np.add.at(
                board_rank_counts,
                (np.nonzero(vb)[0], board_ranks[vb, k]),
                1,
            )

    # Sort board ranks descending. Mask sentinels to a low rank (-1) so they
    # sort to the tail; then negate-sort gives us descending real ranks
    # followed by sentinels. We use np.where to produce the sort key.
    sort_key = np.where(board_valid, board_ranks, -1)
    sorted_desc = -np.sort(-sort_key, axis=1)  # (N, 5), descending; -1 in trailing slots
    valid_slot = sorted_desc >= 0  # (N, 5)

    # Gather hero counts at each slot's rank. For sentinel slots, gather
    # from index 0 then mask to 0.
    gather_idx = np.where(valid_slot, sorted_desc, 0).astype(np.int64)
    rows = np.broadcast_to(np.arange(n)[:, None], (n, 5))
    counts = hole_rank_counts[rows, gather_idx].astype(np.float32)
    counts = np.where(valid_slot, counts, 0.0).astype(np.float32)

    # Pair structure bits.
    paired = (board_rank_counts >= 2).any(axis=1).astype(np.float32)
    double_paired = ((board_rank_counts >= 2).sum(axis=1) >= 2).astype(np.float32)
    tripled = (board_rank_counts >= 3).any(axis=1).astype(np.float32)
    quadded = (board_rank_counts >= 4).any(axis=1).astype(np.float32)
    struct = np.stack([paired, double_paired, tripled, quadded], axis=1)

    # Empty-board fallback: zero everything out.
    board_has_cards = board_valid.any(axis=1)
    counts = np.where(board_has_cards[:, None], counts, 0.0).astype(np.float32)
    struct = np.where(board_has_cards[:, None], struct, 0.0).astype(np.float32)
    return counts, struct


def _straight_flush_features_batch(
    hole: np.ndarray,
    board: np.ndarray,
    visible_count_batch: np.ndarray,
) -> np.ndarray:
    """Vectorized `_straight_flush_features`. Returns `(N, 38)` float32 with
    the same per-board sub-layout as the scalar helper.

    `visible_count_batch` is `(N, 13, 4)` int with `visible_count_batch[n, r, s]`
    equal to 1 iff card `(r, s)` is in env n's hero hole, board A, or board B.
    """
    n = hole.shape[0]
    out = np.zeros((n, 38), dtype=np.float32)

    hole_valid = hole < 52
    board_valid = board < 52
    has_board = board_valid.any(axis=1)
    if not has_board.any():
        return out

    hole_ranks = (hole >> 2).astype(np.int64)
    hole_suits = (hole & 3).astype(np.int64)
    board_ranks = (board >> 2).astype(np.int64)
    board_suits = (board & 3).astype(np.int64)

    # (N, 13) presence masks — distinct rank presence on hole / board.
    hole_rank_mask = np.zeros((n, 13), dtype=bool)
    board_rank_mask = np.zeros((n, 13), dtype=bool)
    # (N, 13, 4) presence masks — per-(rank, suit) presence.
    hole_rank_suit = np.zeros((n, 13, 4), dtype=bool)
    board_rank_suit = np.zeros((n, 13, 4), dtype=bool)
    # (N, 4) suit counts on hole / board.
    hole_suit_counts = np.zeros((n, 4), dtype=np.int32)
    board_suit_counts = np.zeros((n, 4), dtype=np.int32)
    for k in range(5):
        vh = hole_valid[:, k]
        if vh.any():
            idx = np.nonzero(vh)[0]
            hole_rank_mask[idx, hole_ranks[vh, k]] = True
            hole_rank_suit[idx, hole_ranks[vh, k], hole_suits[vh, k]] = True
            np.add.at(hole_suit_counts, (idx, hole_suits[vh, k]), 1)
        vb = board_valid[:, k]
        if vb.any():
            idx = np.nonzero(vb)[0]
            board_rank_mask[idx, board_ranks[vb, k]] = True
            board_rank_suit[idx, board_ranks[vb, k], board_suits[vb, k]] = True
            np.add.at(board_suit_counts, (idx, board_suits[vb, k]), 1)

    vct = visible_count_batch.sum(axis=2)  # (N, 13)
    unseen = visible_count_batch == 0  # (N, 13, 4)

    # Per-suit max hole rank — -1 if hero has no card of that suit.
    hole_max_per_suit = -np.ones((n, 4), dtype=np.int32)
    for k in range(5):
        vh = hole_valid[:, k]
        if vh.any():
            idx = np.nonzero(vh)[0]
            np.maximum.at(
                hole_max_per_suit,
                (idx, hole_suits[vh, k]),
                hole_ranks[vh, k].astype(np.int32),
            )

    # Per-window straight features.
    makes_window = np.zeros((n, 10), dtype=bool)
    straight_outs = np.zeros((n, 10), dtype=np.int32)
    straight_possible = np.zeros((n, 10), dtype=np.int32)
    # SF candidate (rank, suit) accumulated across windows. (N, 13, 4)
    sf_cand = np.zeros((n, 13, 4), dtype=bool)

    arange_n = np.arange(n)

    for w_i, W_set in enumerate(_STRAIGHT_WINDOWS):
        W_mask = np.zeros(13, dtype=bool)
        for r in W_set:
            W_mask[r] = True

        # Suit-agnostic (regular straight) computation.
        B_W_mask = board_rank_mask & W_mask
        H_W_mask = hole_rank_mask & W_mask
        L_mask = W_mask[None, :] & ~board_rank_mask
        M_mask = L_mask & ~hole_rank_mask
        nB_W = B_W_mask.sum(axis=1)
        nH_W = H_W_mask.sum(axis=1)
        nL = L_mask.sum(axis=1)
        nM = M_mask.sum(axis=1)
        makes = (nM == 0) & (nH_W >= 2) & (nL <= 2)
        makes_window[:, w_i] = makes
        straight_possible[:, w_i] = (nB_W >= 3).astype(np.int32)

        L_outs_sum = (L_mask * (4 - vct)).sum(axis=1)
        # nM == 1: argmax returns the index of the single True in M_mask.
        m_argmax = np.argmax(M_mask.astype(np.int8), axis=1)
        single_M_outs = 4 - vct[arange_n, m_argmax]

        gate = (~makes) & (nH_W >= 2)
        cond_3_0 = (nL == 3) & (nM == 0) & gate
        cond_M1 = (nM == 1) & (nL >= 1) & (nL <= 3) & gate
        outs = np.zeros(n, dtype=np.int32)
        outs = np.where(cond_3_0, L_outs_sum, outs)
        outs = np.where(cond_M1, single_M_outs, outs)
        straight_outs[:, w_i] = outs

        # Suit-restricted (SF) computation, broadcast over (N, 13, 4).
        # Want masks shaped (N, 13, 4):
        #   B_s_W[n, r, s] = (r in W) AND (board has (r, s))
        #   H_s_W[n, r, s] = (r in W) AND (hole has (r, s))
        #   L_s_W[n, r, s] = (r in W) AND NOT (board has (r, s))
        #   M_s_W[n, r, s] = L_s_W AND NOT (hole has (r, s))
        # Note: the per-suit `n_s` counts use axis=1 (the rank axis).
        W_mask_3d = W_mask[None, :, None]  # (1, 13, 1)
        H_s_W = hole_rank_suit & W_mask_3d  # (N, 13, 4)
        L_s_W = W_mask_3d & ~board_rank_suit  # (N, 13, 4)
        M_s_W = L_s_W & ~hole_rank_suit
        nH_s = H_s_W.sum(axis=1)  # (N, 4)
        nL_s = L_s_W.sum(axis=1)
        nM_s = M_s_W.sum(axis=1)
        already_s = (nM_s == 0) & (nH_s >= 2) & (nL_s <= 2)
        gate_s = (~already_s) & (nH_s >= 2)
        cond_3_0_s = (nL_s == 3) & (nM_s == 0) & gate_s  # (N, 4)
        cond_M1_s = (nM_s == 1) & (nL_s >= 1) & (nL_s <= 3) & gate_s

        # Broadcast (N, 4) per-suit gates over the rank axis to (N, 13, 4).
        cand_w = (L_s_W & cond_3_0_s[:, None, :]) | (
            M_s_W & cond_M1_s[:, None, :]
        )
        sf_cand |= cand_w

    # Straight nut distance.
    weighted = makes_window.astype(np.int32) * (np.arange(10, dtype=np.int32) + 1)
    h_max = weighted.max(axis=1) - 1  # -1 if no makes
    higher = np.arange(10, dtype=np.int32)[None, :] > h_max[:, None]
    straight_nut_dist = (straight_possible * higher).sum(axis=1)
    any_made_straight = makes_window.any(axis=1)
    straight_nut_dist = np.where(any_made_straight, straight_nut_dist, 0)

    # Flush features.
    flush_possible = (board_suit_counts >= 3).astype(np.int32)  # (N, 4)
    visible_per_suit = visible_count_batch.sum(axis=1)  # (N, 4)

    flush_draw_mask = (hole_suit_counts >= 2) & (board_suit_counts == 2)  # (N, 4)
    flush_draw_outs = np.where(flush_draw_mask, 13 - visible_per_suit, 0).astype(
        np.int32
    )

    # Blockers per (N, 4): unseen ranks > h1 in suit s.
    ranks_arr = np.arange(13, dtype=np.int32)
    above_h1 = ranks_arr[None, :, None] > hole_max_per_suit[:, None, :]  # (N, 13, 4)
    blockers = (above_h1 & unseen).sum(axis=1)  # (N, 4)
    nut_flush_draw_outs = np.where(
        flush_draw_mask & (blockers == 0),
        flush_draw_outs,
        np.where(flush_draw_mask & (blockers == 1), 1, 0),
    ).astype(np.int32)

    # Made-flush nut distance (at most one suit per env can satisfy).
    made_flush_mask = (board_suit_counts >= 3) & (hole_suit_counts >= 2)  # (N, 4)
    any_made_flush = made_flush_mask.any(axis=1)
    made_suit = np.argmax(made_flush_mask.astype(np.int8), axis=1)  # (N,)
    h1_made = hole_max_per_suit[arange_n, made_suit]  # (N,)
    above_made = ranks_arr[None, :] > h1_made[:, None]  # (N, 13)
    unseen_for_made = unseen[arange_n, :, made_suit]  # (N, 13)
    flush_nut_dist = (above_made & unseen_for_made).sum(axis=1).astype(np.int32)
    flush_nut_dist = np.where(any_made_flush, flush_nut_dist, 0)

    # SF outs: filter sf_cand by visibility and flush-draw on this suit.
    sf_filtered = sf_cand & unseen & flush_draw_mask[:, None, :]
    sf_outs_per_suit = sf_filtered.sum(axis=1).astype(np.int32)  # (N, 4)

    out[:, 0] = flush_nut_dist.astype(np.float32)
    out[:, 1] = straight_nut_dist.astype(np.float32)
    out[:, 2:12] = straight_outs.astype(np.float32)
    out[:, 12:22] = straight_possible.astype(np.float32)
    out[:, 22:26] = flush_possible.astype(np.float32)
    out[:, 26:30] = flush_draw_outs.astype(np.float32)
    out[:, 30:34] = nut_flush_draw_outs.astype(np.float32)
    out[:, 34:38] = sf_outs_per_suit.astype(np.float32)

    # Empty-board envs (no board cards) already produce all-zero rows because
    # every per-rank/suit mask is False; no explicit zeroing required.
    return out


def encode_observation_batch(
    obs_arrays: "Mapping[str, np.ndarray]",
    hero_category_a: np.ndarray,
    hero_category_b: np.ndarray,
    config: GameConfig,
) -> np.ndarray:
    """Vectorized observation encoder. Returns `(N, OBS_DIM)` float32.

    `obs_arrays` is the dict from `PyBatchedEngine.observation_arrays()`.
    Categories are passed separately because they're computed via
    `hero_category_batch` at the current-actor seat for each env
    (mirroring how `env._pack_obs` augments the scalar dict).

    Terminal envs (actor == -1) produce all-zero rows, matching the scalar
    encoder's early-return behavior.
    """
    actor = obs_arrays["actor"]
    n = actor.shape[0]
    num_seats = config.num_seats
    # Keep inv_bb in f64 so all scalar arithmetic matches the Python scalar
    # encoder bit-exactly. The final f32 cast happens only on assignment.
    inv_bb = 1.0 / float(config.bb)

    out = np.zeros((n, OBS_DIM), dtype=np.float32)

    live_mask = actor != -1
    if not live_mask.any():
        return out

    hero_idx = np.where(live_mask, actor, 0).astype(np.int64)

    # Hole / board multi-hots.
    hole = obs_arrays["hero_hole"]
    ba = obs_arrays["board_a"]
    bb = obs_arrays["board_b"]
    for src, offset in ((hole, _HOLE_OFF), (ba, _BOARD_A_OFF), (bb, _BOARD_B_OFF)):
        valid = (src < 52) & live_mask[:, None]
        if valid.any():
            rows = np.broadcast_to(np.arange(n)[:, None], src.shape)[valid]
            cols = src[valid].astype(np.int64) + offset
            out[rows, cols] = 1.0

    # Street one-hot (only 0..=3 populate; Preflop/Flop/Turn/River).
    street = obs_arrays["street"].astype(np.int64)
    street_valid = (street < _NUM_STREET_ONEHOT) & live_mask
    if street_valid.any():
        rows = np.nonzero(street_valid)[0]
        out[rows, _STREET_OFF + street[rows]] = 1.0

    # Hero-rotated seat-indexed fields: folded/all_in/stacks over num_seats
    # slots, padded to 8. Scalar code loops for k in range(num_seats); we
    # build a gather index and np.take_along_axis.
    rot = (hero_idx[:, None] + np.arange(num_seats, dtype=np.int64)[None, :]) % num_seats
    folded = obs_arrays["folded"]
    all_in = obs_arrays["all_in"]
    stacks = obs_arrays["stacks"]
    folded_rot = np.take_along_axis(folded, rot, axis=1)
    all_in_rot = np.take_along_axis(all_in, rot, axis=1)

    active_rot = (~folded_rot).astype(np.float32)
    out[live_mask, _ACTIVE_OFF : _ACTIVE_OFF + num_seats] = active_rot[live_mask]
    out[live_mask, _ALLIN_OFF : _ALLIN_OFF + num_seats] = all_in_rot[live_mask].astype(np.float32)
    # Stacks / inv_bb done in f64 to match scalar path's f64 division.
    # See scalar encoder comment for the dead-chips semantics.
    stacks_f64 = stacks.astype(np.float64)
    eff_cap = obs_arrays["eff_stack_cap"].astype(np.float64)
    starting = np.asarray(config.resolved_stacks, dtype=np.float64)
    dead = np.maximum(0.0, starting[None, :] - eff_cap)
    effective = np.maximum(0.0, stacks_f64 - dead)
    effective_rot = np.take_along_axis(effective, rot, axis=1)
    out[live_mask, _STACKS_OFF : _STACKS_OFF + num_seats] = (
        effective_rot[live_mask] * inv_bb
    )

    # Scalars — all arithmetic in f64, assignment casts to f32.
    pot = obs_arrays["pot"].astype(np.float64)
    bet_to_call = obs_arrays["bet_to_call"].astype(np.float64)
    min_bet = obs_arrays["min_bet"].astype(np.float64)
    max_bet = obs_arrays["max_bet"].astype(np.float64)
    out[live_mask, _SCALARS_OFF + 0] = pot[live_mask] * inv_bb
    out[live_mask, _SCALARS_OFF + 1] = bet_to_call[live_mask] * inv_bb
    out[live_mask, _SCALARS_OFF + 2] = min_bet[live_mask] * inv_bb
    out[live_mask, _SCALARS_OFF + 3] = max_bet[live_mask] * inv_bb

    # Relative position: actor_rel = 0 for live envs (since obs is always
    # encoded from the actor's POV, `hero == actor`).
    out[live_mask, _REL_POS_OFF] = 1.0

    # History: last 32 entries oldest-first. history_len encodes kept count.
    # Per-slot layout: 8 hero-rel seat one-hot + 4 gate one-hot + 4 street
    # one-hot + 1 chips/bb scalar. Gate derived from (action, chips) per
    # `_gate_from_action`: Fold→0, CheckCall&chips==0→Check, CheckCall&chips>0
    # →Call, anything else→Raise.
    history_seat = obs_arrays["history_seat"].astype(np.int64)
    history_action = obs_arrays["history_action"].astype(np.int64)
    history_chips = obs_arrays["history_chips"].astype(np.int64)
    history_street = obs_arrays["history_street"].astype(np.int64)
    history_len = obs_arrays["history_len"].astype(np.int64)
    slot_idx = np.arange(_HISTORY_DEPTH, dtype=np.int64)[None, :]
    valid_slots = (slot_idx < history_len[:, None]) & live_mask[:, None]
    if valid_slots.any():
        rows = np.broadcast_to(np.arange(n)[:, None], (n, _HISTORY_DEPTH))[valid_slots]
        slots = np.broadcast_to(slot_idx, (n, _HISTORY_DEPTH))[valid_slots]
        rel_seats = (history_seat[valid_slots] - hero_idx[rows]) % num_seats
        actions_flat = history_action[valid_slots]
        chips_flat = history_chips[valid_slots]
        street_flat = history_street[valid_slots]
        base = _HISTORY_OFF + slots * _HISTORY_SLOT_DIM
        out[rows, base + _HISTORY_SEAT_OFF_REL + rel_seats] = 1.0

        # Gate derivation: matches _gate_from_action elementwise.
        is_fold = actions_flat == FOLD
        is_cc = actions_flat == CHECK_CALL
        is_check = is_cc & (chips_flat == 0)
        is_call = is_cc & (chips_flat > 0)
        gate = np.where(
            is_fold,
            _GATE_FOLD,
            np.where(is_check, _GATE_CHECK, np.where(is_call, _GATE_CALL, _GATE_RAISE)),
        )
        out[rows, base + _HISTORY_GATE_OFF_REL + gate] = 1.0

        # Street one-hot — only valid 0..3 entries fire; sentinel -1
        # masks itself out via the bounds check.
        street_valid = (street_flat >= 0) & (street_flat < _NUM_STREET_ONEHOT)
        if street_valid.any():
            out[
                rows[street_valid],
                base[street_valid] + _HISTORY_STREET_OFF_REL + street_flat[street_valid],
            ] = 1.0

        out[rows, base + _HISTORY_CHIPS_OFF_REL] = chips_flat.astype(np.float64) * inv_bb

    # SPR mirrors the scalar path: uses effective stack so chips above
    # max-other-reachable don't enter the network input.
    pot_safe = np.maximum(pot, 1.0)
    spr = effective_rot / pot_safe[:, None]
    np.clip(spr, 0.0, 4.0, out=spr)
    out[live_mask, _SPR_OFF : _SPR_OFF + num_seats] = spr[live_mask]

    # Pot odds. f64 throughout; assignment casts to f32.
    street_commit = obs_arrays["street_commit"]
    hero_street_commit = np.take_along_axis(
        street_commit, hero_idx[:, None], axis=1
    )[:, 0].astype(np.float64)
    to_call = np.maximum(bet_to_call - hero_street_commit, 0.0)
    denom = pot + to_call
    pot_odds = np.where(to_call > 0.0, to_call / np.where(denom > 0.0, denom, 1.0), 0.0)
    out[live_mask, _POT_ODDS_OFF] = pot_odds[live_mask]

    # Bet-faced as fraction of pot-bet-into. pot - to_call is the pot the
    # facing bet was made into; clamp to ≥1 to avoid div-by-zero.
    pot_before_bet = np.maximum(pot - to_call, 1.0)
    bet_pct_pot = np.where(to_call > 0.0, to_call / pot_before_bet, 0.0)
    np.clip(bet_pct_pot, 0.0, 4.0, out=bet_pct_pot)
    out[live_mask, _BET_PCT_POT_OFF] = bet_pct_pot[live_mask]

    # Hand-category one-hots (only 0..=8 populate).
    for cats, off in (
        (hero_category_a.astype(np.int64), _CAT_A_OFF),
        (hero_category_b.astype(np.int64), _CAT_B_OFF),
    ):
        cat_valid = (cats < _NUM_CATEGORIES) & live_mask
        if cat_valid.any():
            rows = np.nonzero(cat_valid)[0]
            out[rows, off + cats[rows]] = 1.0

    # Draw flags.
    f_a, s_a, f_b, s_b = _rust_draw_flags(hole, ba, bb)
    out[live_mask, _DRAW_A_OFF + 0] = f_a[live_mask]
    out[live_mask, _DRAW_A_OFF + 1] = s_a[live_mask]
    out[live_mask, _DRAW_B_OFF + 0] = f_b[live_mask]
    out[live_mask, _DRAW_B_OFF + 1] = s_b[live_mask]

    # Pair-with-board counts + board pair structure.
    counts_a, struct_a, counts_b, struct_b = _rust_pair_features(hole, ba, bb)
    out[live_mask, _PAIR_COUNT_A_OFF : _PAIR_COUNT_A_OFF + 5] = counts_a[live_mask]
    out[live_mask, _PAIR_COUNT_B_OFF : _PAIR_COUNT_B_OFF + 5] = counts_b[live_mask]
    out[live_mask, _BOARD_STRUCT_A_OFF : _BOARD_STRUCT_A_OFF + 4] = struct_a[live_mask]
    out[live_mask, _BOARD_STRUCT_B_OFF : _BOARD_STRUCT_B_OFF + 4] = struct_b[live_mask]

    # Hero rank histogram. Mirrors the np.add.at pattern used by
    # _pair_features_batch's hero count tensor.
    hole_valid = hole < 52
    hole_ranks = (hole >> 2).astype(np.int64)
    hist = np.zeros((n, 13), dtype=np.float32)
    for k in range(5):
        vh = hole_valid[:, k]
        if vh.any():
            np.add.at(hist, (np.nonzero(vh)[0], hole_ranks[vh, k]), 1.0)
    out[live_mask, _HERO_RANK_HIST_OFF : _HERO_RANK_HIST_OFF + 13] = hist[live_mask]

    # Straight / flush / SF block — needs (N, 13, 4) cross-board visibility.
    # Flatten the per-slot loop: one fancy-index assignment per source.
    visible_count_batch = np.zeros((n, 13, 4), dtype=np.int8)
    for src in (hole, ba, bb):
        src_valid = src < 52
        if not src_valid.any():
            continue
        env_idx, slot_idx = np.nonzero(src_valid)
        cards = src[env_idx, slot_idx]
        rk = (cards >> 2).astype(np.intp)
        sk = (cards & 3).astype(np.intp)
        visible_count_batch[env_idx, rk, sk] = 1
    sf_a, sf_b = _rust_sf_features(hole, ba, bb, visible_count_batch)
    out[live_mask, _FLUSH_NUT_DIST_A_OFF : _FLUSH_NUT_DIST_A_OFF + 38] = sf_a[
        live_mask
    ]
    out[live_mask, _FLUSH_NUT_DIST_B_OFF : _FLUSH_NUT_DIST_B_OFF + 38] = sf_b[
        live_mask
    ]

    # Structural seat-exists mask. Constant per config; written only on live
    # rows so terminal envs stay all-zero.
    out[live_mask, _SEAT_EXISTS_OFF : _SEAT_EXISTS_OFF + num_seats] = 1.0

    # Per-seat commits, hero-rotated. street_commit already loaded above for
    # pot-odds; reuse the same tensor.
    total_commit = obs_arrays["total_commit"].astype(np.float64)
    street_commit_f64 = street_commit.astype(np.float64)
    total_commit_rot = np.take_along_axis(total_commit, rot, axis=1)
    street_commit_rot = np.take_along_axis(street_commit_f64, rot, axis=1)
    out[live_mask, _TOTAL_COMMIT_OFF : _TOTAL_COMMIT_OFF + num_seats] = (
        total_commit_rot[live_mask] * inv_bb
    )
    out[live_mask, _STREET_COMMIT_OFF : _STREET_COMMIT_OFF + num_seats] = (
        street_commit_rot[live_mask] * inv_bb
    )

    # Last aggressor: hero-relative one-hot for live rows whose
    # last_aggressor != -1.
    last_aggressor = obs_arrays["last_aggressor"].astype(np.int64)
    agg_valid = live_mask & (last_aggressor >= 0)
    if agg_valid.any():
        rows = np.nonzero(agg_valid)[0]
        rel_agg = (last_aggressor[rows] - hero_idx[rows]) % num_seats
        out[rows, _LAST_AGGRESSOR_OFF + rel_agg] = 1.0

    # Hero distance to button.
    button = obs_arrays["button"].astype(np.int64)
    if live_mask.any():
        rows = np.nonzero(live_mask)[0]
        btn_rel = (button[rows] - hero_idx[rows]) % num_seats
        out[rows, _HERO_BTN_DIST_OFF + btn_rel] = 1.0

    # Cross-board interactions. Shared-rank mask, per-suit hero-involved
    # flush blocks, and the cross-board straight indicators all vectorize
    # via 13-bit rank-presence masks per env.
    hole_valid = hole < 52
    ba_valid = ba < 52
    bb_valid = bb < 52
    hole_ranks_idx = np.where(hole_valid, hole >> 2, 0)
    ba_ranks_idx = np.where(ba_valid, ba >> 2, 0)
    bb_ranks_idx = np.where(bb_valid, bb >> 2, 0)

    hole_rank_mask = np.zeros((n, 13), dtype=bool)
    ba_rank_mask = np.zeros((n, 13), dtype=bool)
    bb_rank_mask = np.zeros((n, 13), dtype=bool)
    for k in range(5):
        vh = hole_valid[:, k]
        if vh.any():
            hole_rank_mask[np.nonzero(vh)[0], hole_ranks_idx[vh, k]] = True
        va = ba_valid[:, k]
        if va.any():
            ba_rank_mask[np.nonzero(va)[0], ba_ranks_idx[va, k]] = True
        vb = bb_valid[:, k]
        if vb.any():
            bb_rank_mask[np.nonzero(vb)[0], bb_ranks_idx[vb, k]] = True
    shared_mask = ba_rank_mask & bb_rank_mask
    out[live_mask, _SHARED_RANKS_OFF : _SHARED_RANKS_OFF + 13] = shared_mask[
        live_mask
    ].astype(np.float32)

    hole_suit_counts = np.zeros((n, 4), dtype=np.int8)
    ba_suit_counts = np.zeros((n, 4), dtype=np.int8)
    bb_suit_counts = np.zeros((n, 4), dtype=np.int8)
    for src_valid, src_arr, dest in (
        (hole_valid, hole, hole_suit_counts),
        (ba_valid, ba, ba_suit_counts),
        (bb_valid, bb, bb_suit_counts),
    ):
        suits = (src_arr & 3).astype(np.int64)
        for k in range(5):
            vk = src_valid[:, k]
            if vk.any():
                np.add.at(dest, (np.nonzero(vk)[0], suits[vk, k]), 1)
    h_ge2 = hole_suit_counts >= 2
    a_ge3 = ba_suit_counts >= 3
    b_ge3 = bb_suit_counts >= 3
    a_eq2 = ba_suit_counts == 2
    b_eq2 = bb_suit_counts == 2
    made_both_suit = h_ge2 & a_ge3 & b_ge3
    draw_both_suit = h_ge2 & a_eq2 & b_eq2
    mixed_suit = h_ge2 & ((a_ge3 & b_eq2) | (a_eq2 & b_ge3))
    out[live_mask, _FLUSH_MADE_BOTH_OFF : _FLUSH_MADE_BOTH_OFF + 4] = (
        made_both_suit[live_mask].astype(np.float32)
    )
    out[live_mask, _FLUSH_DRAW_BOTH_OFF : _FLUSH_DRAW_BOTH_OFF + 4] = (
        draw_both_suit[live_mask].astype(np.float32)
    )
    out[live_mask, _FLUSH_MIXED_OFF : _FLUSH_MIXED_OFF + 4] = mixed_suit[
        live_mask
    ].astype(np.float32)

    if live_mask.any():
        boards_visible = ba_valid.any(axis=1) & bb_valid.any(axis=1)
        cb_md, cb_dr, cb_mx = _rust_cross_board_straight(
            hole_rank_mask, ba_rank_mask, bb_rank_mask, boards_visible
        )
        out[live_mask, _STRAIGHT_MADE_BOTH_OFF] = cb_md[live_mask]
        out[live_mask, _STRAIGHT_DRAW_BOTH_OFF] = cb_dr[live_mask]
        out[live_mask, _STRAIGHT_MIXED_OFF] = cb_mx[live_mask]

    opp_fr = obs_arrays.get("opp_outcome_fractions")
    if opp_fr is not None:
        out[:, _OPP_OUTCOME_OFF : _OPP_OUTCOME_OFF + _OPP_OUTCOME_DIM] = opp_fr.astype(
            np.float32, copy=False
        )

    return out
