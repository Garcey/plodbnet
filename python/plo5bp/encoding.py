"""Observation encoding. Produces a fixed-length float32 vector from the
dict emitted by `PyGameState.observation_dict()` (augmented in `env._pack_obs`
with Rust-computed hand categories).

Layout (991 dims total):
  0..52     hero hole multi-hot (52)
  52..104   board A multi-hot (52)
  104..156  board B multi-hot (52)
  156..160  street one-hot (Preflop/Flop/Turn/River) — preflop always zero
  160..168  active mask, hero-rotated, padded to 8
  168..176  all-in mask, hero-rotated, padded to 8
  176..184  stacks / bb (stack-depth in bb), hero-rotated, padded to 8
  184..188  scalars: pot, bet_to_call, min_bet, max_bet — all / bb (bb units)
  188..196  relative-position one-hot of actor (actor - hero) mod num_seats
  196..772  history: last 32 actions oldest-first, each slot 18 dims
            (seat-one-hot hero-rel 8 + gate one-hot 4 + street one-hot 4
            + chips/bb 1 + chips/pot-before 1).
            Gate is {Fold=0, Check=1, Call=2, Raise=3} derived at encode time
            from (action, chips): CheckCall with chips==0 is Check, with chips>0
            is Call; any Bet*/AllIn is Raise. chips is the per-action chip
            DELTA the action added to the pot (engine ActionRecord.chips;
            antes are never recorded). Slot dim 16 is chips / cfg.bb; slot
            dim 17 is chips / pot-before-the-action, clipped [0, 2], where
            pot_before(slot j) = current_pot − Σ chips of slots ≥ j (valid
            under 32-slot truncation: truncated actions all precede the
            visible window, so their chips stay inside the subtracted-from
            pot). Speaks the same pot-fraction language as the v2 anchor
            sizing head.
  772..780  SPR per seat, hero-rotated, padded to 8 (stack/max(pot,1), clip[0,4])
  780..781  pot odds (to_call / (pot + to_call), 0 if no bet to face)
  781..790  hero hand category one-hot on board A (9 categories)
  790..799  hero hand category one-hot on board B (9 categories)
  799..801  hero draw flags on board A (flush, straight)
  801..803  hero draw flags on board B (flush, straight)
  803..808  pair-with-board count on A: count of hero hole cards matching
            the rank of the i-th board card (sorted by rank descending),
            5 slots, trailing zeros for streets < river
  808..813  pair-with-board count on B (same semantics)
  813..817  board A pair structure (paired, double_paired, tripled, quadded)
  817..821  board B pair structure (same semantics)
  821..834  hero rank histogram: slot r = count of hero hole cards at rank r
            (rank 0=2, 12=A). Board-agnostic; closes the pocket-pair-
            not-on-board blind spot left by the pair-with-board feature.
  834..872  straight/flush/SF block on board A (38 dims):
              834       flush_nut_distance       (0 if hero has no flush; uncapped)
              835       straight_nut_distance    (0 if hero has no straight; uncapped)
              836..846  straight_outs_per_window (10 dims, slot 0=wheel, 9=broadway)
              846..856  straight_possible_per_window (10 dims, board-only binary)
              856..860  flush_possible_per_suit  (4 dims, board-only binary)
              860..864  flush_draw_outs[s]       (4 dims, per suit; needs 2-2 split)
              864..868  nut_flush_draw_outs[s]   (4 dims; produces nut after hit)
              868..872  straight_flush_draw_outs[s] (4 dims; intersects flush + straight)
  872..910  straight/flush/SF block on board B (38 dims, same layout)
  910..918  hero-rotated seat-exists mask: slot k = 1 iff (hero + k) % num_seats
            is a real seat, else 0. Structural; doesn't depend on stack state,
            so a 0-chip seat is still distinguishable from a padded slot.
  918..926  per-seat hand-total commit, hero-rotated, /cfg.bb (raw, no clamp).
  926..934  per-seat street commit, hero-rotated, /cfg.bb (raw, no clamp).
  934..942  last-aggressor one-hot, hero-relative; all-zero when no raise yet.
  942..950  hero distance to button: one-hot of (button - hero) % num_seats.
  950..963  shared-rank mask: slot r = 1 iff rank r appears on BOTH boards.
  963..967  per-suit cross-board flush MADE on both boards: hero ≥2-of-s
            AND board_a ≥3-of-s AND board_b ≥3-of-s.
  967..971  per-suit cross-board flush DRAW on both boards: hero ≥2-of-s
            AND board_a 2-of-s AND board_b 2-of-s.
  971..975  per-suit cross-board flush MIXED: hero ≥2-of-s AND
            (one board ≥3-of-s, the other 2-of-s).
  975       cross-board straight MADE on both: ∃ pair {r1,r2} ⊆ hero ranks
            making a straight on A and on B (windows may differ).
  976       cross-board straight DRAW on both: ∃ pair drawing (4-rank
            coverage) on A and on B (and not made on either).
  977       cross-board straight MIXED: ∃ pair made on one, drawing on
            the other.
  978..990  opp-outcome fractions: 12 dims (3 hand sizes × 4 outcomes),
            row-major [k][outcome] for k ∈ {2, 3, 4}. Per k, the four
            outcomes are: opp scoops hero, opp quarters hero, hero
            scoops opp, hero quarters opp. Each entry is a fraction in
            [0, 1] of unseen-deck k-card opponent combos producing that
            outcome at the current board rank (PLO5 rule: exactly 2 from
            k + 3 from visible board, evaluated independently per board).
            Computed in Rust (`GameState::opp_outcome_fractions`); k=2,3
            exhaustive, k=4 MC-sampled (1024) with a deterministic seed
            from observation-visible state. All-zero pre-flop / terminal.
  990..991  bet-faced as fraction of pot-bet-into: to_call /
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
from plo5bp.sizing import (  # v7 STK-2 raise-ladder envelope
    ANCHOR_COUNT,
    n_legal_anchors_np,
)

OBS_DIM: int = 1171  # v7 batch-2 tail (stack+board+dual) appended after 1019

# v1 (pre-anchor-head era) observation layout: 17-dim history slots, no
# pot-fraction dim, tail blocks 32 lower. v1 checkpoints can keep
# serving in the UI via `downgrade_obs_to_v1`, which is an EXACT
# projection — the v2 layout is purely additive.
OBS_DIM_V1: int = 959

# The 991-dim layout that v2/v4 stems (through vFour4) trained at —
# everything before the obs-v2 tail append of 2026-07-06 (V5_DESIGN.md
# §3.2). Those checkpoints keep serving via `downgrade_obs_to_v2` (a
# plain tail slice; the append is exactly function-preserving).
OBS_DIM_V2: int = 991

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
_HISTORY_SLOT_DIM = 18
_MAX_SEATS = 8
_NUM_STREET_ONEHOT = 4
_NUM_CATEGORIES = 9  # high-card..straight-flush

# Per-slot relative offsets within a history slot (sum = _HISTORY_SLOT_DIM).
_HISTORY_SEAT_OFF_REL = 0  # 8 dims (hero-relative seat one-hot)
_HISTORY_GATE_OFF_REL = 8  # 4 dims {Fold, Check, Call, Raise}
_HISTORY_STREET_OFF_REL = 12  # 4 dims (preflop, flop, turn, river)
_HISTORY_CHIPS_OFF_REL = 16  # 1 dim (chips / cfg.bb)
_HISTORY_FRAC_OFF_REL = 17  # 1 dim (chips / pot-before-action, clip [0, 2])

# Encoder-side gate enum (distinct from the policy gate which is 3-way:
# Fold/CheckCall/Raise). The encoder breaks CheckCall apart so the
# network sees check vs call explicitly.
_GATE_FOLD = 0
_GATE_CHECK = 1
_GATE_CALL = 2
_GATE_RAISE = 3

_SPR_OFF = 772
_POT_ODDS_OFF = 780
_CAT_A_OFF = 781
_CAT_B_OFF = 790
_DRAW_A_OFF = 799
_DRAW_B_OFF = 801
_PAIR_COUNT_A_OFF = 803
_PAIR_COUNT_B_OFF = 808
_BOARD_STRUCT_A_OFF = 813
_BOARD_STRUCT_B_OFF = 817
_HERO_RANK_HIST_OFF = 821

_FLUSH_NUT_DIST_A_OFF = 834
_STRAIGHT_NUT_DIST_A_OFF = 835
_STRAIGHT_OUTS_A_OFF = 836
_STRAIGHT_POSSIBLE_A_OFF = 846
_FLUSH_POSSIBLE_A_OFF = 856
_FLUSH_DRAW_OUTS_A_OFF = 860
_NUT_FLUSH_DRAW_OUTS_A_OFF = 864
_SF_DRAW_OUTS_A_OFF = 868

_FLUSH_NUT_DIST_B_OFF = 872
_STRAIGHT_NUT_DIST_B_OFF = 873
_STRAIGHT_OUTS_B_OFF = 874
_STRAIGHT_POSSIBLE_B_OFF = 884
_FLUSH_POSSIBLE_B_OFF = 894
_FLUSH_DRAW_OUTS_B_OFF = 898
_NUT_FLUSH_DRAW_OUTS_B_OFF = 902
_SF_DRAW_OUTS_B_OFF = 906

_SEAT_EXISTS_OFF = 910  # 8 dims; structural seat-presence, hero-rotated
_TOTAL_COMMIT_OFF = 918  # 8 dims; per-seat hand-total commit, hero-rotated, /bb
_STREET_COMMIT_OFF = 926  # 8 dims; per-seat street commit, hero-rotated, /bb
_LAST_AGGRESSOR_OFF = 934  # 8 dims; hero-rel one-hot of last aggressor (or all-zero)
_HERO_BTN_DIST_OFF = 942  # 8 dims; one-hot of (button - hero) % num_seats

_SHARED_RANKS_OFF = 950  # 13 dims; rank present on both A and B
_FLUSH_MADE_BOTH_OFF = 963  # 4 dims; per-suit hero-involved made on both
_FLUSH_DRAW_BOTH_OFF = 967  # 4 dims; per-suit hero-involved draw on both
_FLUSH_MIXED_OFF = 971  # 4 dims; per-suit hero-involved made on one + draw on other
_STRAIGHT_MADE_BOTH_OFF = 975  # 1 dim; same hero rank-pair makes straight on both
_STRAIGHT_DRAW_BOTH_OFF = 976  # 1 dim; same hero pair draws (4-rank cov) on both
_STRAIGHT_MIXED_OFF = 977  # 1 dim; same hero pair made on one, drawing on other

_OPP_OUTCOME_OFF = 978  # 12 dims; [k=2,3,4][outcome] fractions in [0,1]
_OPP_OUTCOME_DIM = 12
_BET_PCT_POT_OFF = 990  # 1 dim; to_call / max(pot - to_call, 1), clipped [0, 4]

# ---- obs v2 tail (appended 2026-07-06, V5_DESIGN.md §3.2) -------------------
# Everything below is a pure append: dims 0..991 are byte-identical to the
# pre-v5 layout, so 991-era checkpoints serve via `downgrade_obs_to_v2` and
# warm-starts zero-pad the first-layer columns (function-preserving).
_PER_BOARD_OUTCOME_OFF = 991  # 8 dims; hero ahead/tie/behind per board
_PER_BOARD_OUTCOME_DIM = 8    # (k=2 exhaustive) + win-exactly-one + tie-both
_BLOCKER_A_OFF = 999   # 4 dims; unconditional blockers-to-nuts, board A
_BLOCKER_B_OFF = 1003  # 4 dims; board B (see _blocker_features)
_EFF_PRICE_OFF = 1007  # 5 dims; stack-capped price + commitment + log1p money
_SPR_LOG_OFF = 1012    # 8 dims; log1p(effective SPR) per seat, UNCLIPPED —
#                        the [0,4]-clipped _SPR_OFF block saturates for the
#                        entire deep tier at the flop (true SPR 5.4-13.9)

# ---- obs v3 batch-2 tail (stack + board + dual; 2026-07-12) -----------------
# Pure tail append after 1019 (V7_OBS_IMPL_PLAN.md). Everything 0..1020 is
# byte-identical to the obs-v2 layout, so downgrade_obs_to_v2/v1 stay exact
# tail slices and old checkpoints keep serving. Blocks marked [ENGINE] need
# Rust plumbing (Chunk B) and stay 0.0 until then; both encoders write zeros
# there, so serial/batched parity holds through Chunk A.
_OBS_V2_5_TAIL_OFF = 1020  # first v7 batch-2 dim
# Stack geometry 1020..1061 (41)
_STK1_OFF = 1020   # 4  money / raise-exposure behind        [ENGINE: acted_this_street]
_STK2_OFF = 1024   # 6  raise-ladder envelope
_STK4_OFF = 1030   # 8  per-seat commitment ratio
_STK5_OFF = 1038   # 4  spr-after-action
_STK6_OFF = 1042   # 2  geometric jam plan
_STK7_OFF = 1044   # 2  pot-ceiling implied odds
_STK8_OFF = 1046   # 3  side-pot eligibility (winnable pot)
_STK9_OFF = 1049   # 2  call-risk fraction
_STK10_OFF = 1051  # 2  ante-pot bloat
_STK11_OFF = 1053  # 8  per-seat price-to-continue
# Board texture 1061..1139 (78)
_BRD1_OFF = 1061   # 10 board rank ladder
_BRD2_OFF = 1071   # 12 board suit census (==2 per suit, ==4, ==5) per board
_BRD4_OFF = 1083   # 6  arrival volatility census
_BRD5_OFF = 1089   # 6  hero vulnerability outs
_BRD6_OFF = 1095   # 4  straight out union
_BRD7_OFF = 1099   # 2  boat+ outs                            [ENGINE: Rust fn]
_BRD8_OFF = 1101   # 4  flush-draw rank quality
_BRD9_OFF = 1105   # 4  backdoor draw census (flop-gated)
_BRD10_OFF = 1109  # 2  future nut-flush blocker
_BRD11_OFF = 1111  # 20 turn/river card identity (deal order)
_BRD12_OFF = 1131  # 4  hero improve outs                     [ENGINE: Rust fn]
_BRD13_OFF = 1135  # 4  board nut ceiling class
# Double-board 1139..1171 (32)
_DUAL1_OFF = 1139  # 2  split-adjusted price ladder
_DUAL2_OFF = 1141  # 10 best-hand card usage / coverage       [ENGINE: winning pair]
_DUAL3_OFF = 1151  # 6  nut-lock / freeroll flags
_DUAL4_OFF = 1157  # 5  guaranteed pot share                  [ENGINE: k=2 g_min/max]
_DUAL5_OFF = 1162  # 9  villain cross-board coverage
assert _DUAL5_OFF + 9 == OBS_DIM, "v7 batch-2 tail must end exactly at OBS_DIM"

# Index map projecting the v2 (991) layout onto the exact v1 (959)
# layout: pre-history block verbatim, first 17 of each 18-dim history
# slot, then the tail (identical content, shifted by +32 in v2).
_V1_SLOT_DIM = 17
_V1_INDEX: np.ndarray = np.concatenate([
    np.arange(_HISTORY_OFF),
    np.concatenate([
        _HISTORY_OFF + s * _HISTORY_SLOT_DIM + np.arange(_V1_SLOT_DIM)
        for s in range(_HISTORY_DEPTH)
    ]),
    np.arange(_SPR_OFF, OBS_DIM_V2),
]).astype(np.int64)
assert _V1_INDEX.shape[0] == OBS_DIM_V1


def downgrade_obs_to_v1(vec: np.ndarray) -> np.ndarray:
    """Project a current-layout observation onto the v1 (..., 959) layout.

    Exact: drops each history slot's pot-fraction dim, un-shifts the
    post-history tail, and (all indices being < 991) implicitly drops the
    obs-v2 tail. Used by the UI to keep serving v1-era checkpoints
    (trained at OBS_DIM 959) after the encoder upgrades.
    """
    return np.ascontiguousarray(vec[..., _V1_INDEX])


def downgrade_obs_to_v2(vec: np.ndarray) -> np.ndarray:
    """Slice a current-layout observation onto the 991-dim layout that
    v2/v4 stems trained at. Exact — the obs-v2 additions are a pure tail
    append."""
    return np.ascontiguousarray(vec[..., :OBS_DIM_V2])



# ---- bare-visibility / "minimal" obs mode (experiment stem) ----------------
# Table-visible state only: cards, street, who's in / all-in, stacks, pot
# pricing scalars, commits, seat structure, button, action history.
# NO SPR/odds/categories/draws/blockers/opp-outcome MC/v2-v7 engineered tails.
# Used by `--obs-mode minimal` (cold-start only). Gather is exact.
_MINIMAL_RANGES: tuple[tuple[int, int], ...] = (
    (_HOLE_OFF, _BOARD_B_OFF + 52),          # hole + board A + board B (156)
    (_STREET_OFF, _STREET_OFF + 4),          # street one-hot (4)
    (_ACTIVE_OFF, _ACTIVE_OFF + 8),          # active mask (8)
    (_ALLIN_OFF, _ALLIN_OFF + 8),            # all-in mask (8)
    (_STACKS_OFF, _STACKS_OFF + 8),          # stacks/bb (8)
    (_SCALARS_OFF, _SCALARS_OFF + 4),        # pot, to_call, min_bet, max_bet (4)
    (_HISTORY_OFF, _HISTORY_OFF + _HISTORY_DEPTH * _HISTORY_SLOT_DIM),  # 576
    (_SEAT_EXISTS_OFF, _SEAT_EXISTS_OFF + 8),       # seat-exists (8)
    (_TOTAL_COMMIT_OFF, _TOTAL_COMMIT_OFF + 8),     # hand total commit (8)
    (_STREET_COMMIT_OFF, _STREET_COMMIT_OFF + 8),   # street commit (8)
    (_HERO_BTN_DIST_OFF, _HERO_BTN_DIST_OFF + 8),   # button vs hero (8)
)
_MINIMAL_INDEX: np.ndarray = np.concatenate(
    [np.arange(a, b, dtype=np.int64) for a, b in _MINIMAL_RANGES]
)
OBS_DIM_MINIMAL: int = int(_MINIMAL_INDEX.shape[0])
assert OBS_DIM_MINIMAL == 796, f"unexpected minimal width {_MINIMAL_INDEX.shape[0]}"


def project_obs_minimal(vec: np.ndarray) -> np.ndarray:
    """Project full-layout obs `(..., OBS_DIM)` -> bare-visibility `(..., 796)`.

    Exact gather of table-visible dims only. Safe on vectors and batches.
    Prefer `encode_observation_minimal` / `encode_observation_batch_minimal`
    when encoding fresh state — those never compute the dropped tails.
    """
    return np.ascontiguousarray(vec[..., _MINIMAL_INDEX])


# Compact offsets for the 796-d bare-visibility layout (same content as
# gather(_MINIMAL_INDEX) on a full vector, contiguous).
_M_HOLE = 0
_M_BOARD_A = 52
_M_BOARD_B = 104
_M_STREET = 156
_M_ACTIVE = 160
_M_ALLIN = 168
_M_STACKS = 176
_M_SCALARS = 184
_M_HISTORY = 188  # 32 * 18 = 576; full layout history starts at 196
_M_SEAT_EXISTS = 764
_M_TOTAL_COMMIT = 772
_M_STREET_COMMIT = 780
_M_HERO_BTN = 788
assert _M_HERO_BTN + 8 == OBS_DIM_MINIMAL


def encode_observation_minimal(
    obs: Mapping[str, Any], config: GameConfig
) -> np.ndarray:
    """Encode table-visible dims only into `(OBS_DIM_MINIMAL,)` float32.

    Bit-exact with `project_obs_minimal(encode_observation(...))` but never
    computes SPR/odds/categories/draws/blockers/opp-MC/v2-v7 tails.
    """
    out = np.zeros(OBS_DIM_MINIMAL, dtype=np.float32)
    num_seats = config.num_seats
    hero = obs["actor"]
    if hero is None:
        return out

    for idx in obs["hero_hole"]:
        out[_M_HOLE + int(idx)] = 1.0
    for idx in obs["board_a"]:
        out[_M_BOARD_A + int(idx)] = 1.0
    for idx in obs["board_b"]:
        out[_M_BOARD_B + int(idx)] = 1.0

    street_idx = int(obs["street"])
    if 0 <= street_idx < _NUM_STREET_ONEHOT:
        out[_M_STREET + street_idx] = 1.0

    folded = obs["folded"]
    all_in = obs["all_in"]
    stacks = obs["stacks"]
    eff_cap = obs["eff_stack_cap"]
    starting = config.resolved_stacks
    inv_bb = 1.0 / float(config.bb)
    dead_chips = [
        max(0, int(starting[s]) - int(eff_cap[s])) for s in range(num_seats)
    ]
    eff_per_seat = [
        max(0.0, float(stacks[s]) - float(dead_chips[s])) for s in range(num_seats)
    ]
    for k in range(num_seats):
        seat = (hero + k) % num_seats
        if not folded[seat]:
            out[_M_ACTIVE + k] = 1.0
        if all_in[seat]:
            out[_M_ALLIN + k] = 1.0
        out[_M_STACKS + k] = eff_per_seat[seat] * inv_bb

    pot = float(obs["pot"])
    btc = float(obs["bet_to_call"])
    out[_M_SCALARS + 0] = pot * inv_bb
    out[_M_SCALARS + 1] = btc * inv_bb
    out[_M_SCALARS + 2] = float(obs["min_bet"]) * inv_bb
    out[_M_SCALARS + 3] = float(obs["max_bet"]) * inv_bb

    history = obs["history"]
    if len(history) > _HISTORY_DEPTH:
        history = history[-_HISTORY_DEPTH:]
    pot_now_chips = int(obs["pot"])
    pot_before = [0] * len(history)
    suffix = 0
    for j in range(len(history) - 1, -1, -1):
        suffix += int(history[j][2])
        pot_before[j] = pot_now_chips - suffix
    for slot, (seat, action, chips, street_h) in enumerate(history):
        base = _M_HISTORY + slot * _HISTORY_SLOT_DIM
        rel_seat = (seat - hero) % num_seats
        out[base + _HISTORY_SEAT_OFF_REL + rel_seat] = 1.0
        gate = _gate_from_action(int(action), int(chips))
        out[base + _HISTORY_GATE_OFF_REL + gate] = 1.0
        s_idx = int(street_h)
        if 0 <= s_idx < _NUM_STREET_ONEHOT:
            out[base + _HISTORY_STREET_OFF_REL + s_idx] = 1.0
        out[base + _HISTORY_CHIPS_OFF_REL] = float(chips) * inv_bb
        frac = float(chips) / float(max(pot_before[slot], 1))
        out[base + _HISTORY_FRAC_OFF_REL] = min(max(frac, 0.0), 2.0)

    for k in range(num_seats):
        out[_M_SEAT_EXISTS + k] = 1.0

    street_commit = obs.get("street_commit", [0] * num_seats)
    total_commit = obs["total_commit"]
    for k in range(num_seats):
        seat = (hero + k) % num_seats
        out[_M_TOTAL_COMMIT + k] = float(total_commit[seat]) * inv_bb
        out[_M_STREET_COMMIT + k] = float(street_commit[seat]) * inv_bb

    button = int(obs["button"])
    out[_M_HERO_BTN + (button - hero) % num_seats] = 1.0
    return out


def encode_observation_batch_minimal(
    obs_arrays: "Mapping[str, np.ndarray]",
    config: GameConfig,
) -> np.ndarray:
    """Vectorized bare-visibility encoder. Returns `(N, OBS_DIM_MINIMAL)`.

    Bit-exact with `project_obs_minimal(encode_observation_batch(...))`.
    Does not read categories, draws, opp-outcome, or any engineered tail.
    """
    actor = obs_arrays["actor"]
    n = actor.shape[0]
    num_seats = config.num_seats
    inv_bb = 1.0 / float(config.bb)
    out = np.zeros((n, OBS_DIM_MINIMAL), dtype=np.float32)

    live_mask = actor != -1
    if not live_mask.any():
        return out

    hero_idx = np.where(live_mask, actor, 0).astype(np.int64)

    hole = obs_arrays["hero_hole"]
    ba = obs_arrays["board_a"]
    bb = obs_arrays["board_b"]
    for src, offset in ((hole, _M_HOLE), (ba, _M_BOARD_A), (bb, _M_BOARD_B)):
        valid = (src < 52) & live_mask[:, None]
        if valid.any():
            rows = np.broadcast_to(np.arange(n)[:, None], src.shape)[valid]
            cols = src[valid].astype(np.int64) + offset
            out[rows, cols] = 1.0

    street = obs_arrays["street"].astype(np.int64)
    street_valid = (street < _NUM_STREET_ONEHOT) & live_mask
    if street_valid.any():
        rows = np.nonzero(street_valid)[0]
        out[rows, _M_STREET + street[rows]] = 1.0

    rot = (hero_idx[:, None] + np.arange(num_seats, dtype=np.int64)[None, :]) % num_seats
    folded = obs_arrays["folded"]
    all_in = obs_arrays["all_in"]
    stacks = obs_arrays["stacks"]
    folded_rot = np.take_along_axis(folded, rot, axis=1)
    all_in_rot = np.take_along_axis(all_in, rot, axis=1)
    active_rot = (~folded_rot).astype(np.float32)
    out[live_mask, _M_ACTIVE : _M_ACTIVE + num_seats] = active_rot[live_mask]
    out[live_mask, _M_ALLIN : _M_ALLIN + num_seats] = all_in_rot[live_mask].astype(
        np.float32
    )

    stacks_f64 = stacks.astype(np.float64)
    eff_cap = obs_arrays["eff_stack_cap"].astype(np.float64)
    starting = np.asarray(config.resolved_stacks, dtype=np.float64)
    dead = np.maximum(0.0, starting[None, :] - eff_cap)
    effective = np.maximum(0.0, stacks_f64 - dead)
    effective_rot = np.take_along_axis(effective, rot, axis=1)
    out[live_mask, _M_STACKS : _M_STACKS + num_seats] = (
        effective_rot[live_mask] * inv_bb
    )

    pot = obs_arrays["pot"].astype(np.float64)
    bet_to_call = obs_arrays["bet_to_call"].astype(np.float64)
    min_bet = obs_arrays["min_bet"].astype(np.float64)
    max_bet = obs_arrays["max_bet"].astype(np.float64)
    out[live_mask, _M_SCALARS + 0] = pot[live_mask] * inv_bb
    out[live_mask, _M_SCALARS + 1] = bet_to_call[live_mask] * inv_bb
    out[live_mask, _M_SCALARS + 2] = min_bet[live_mask] * inv_bb
    out[live_mask, _M_SCALARS + 3] = max_bet[live_mask] * inv_bb

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
        base = _M_HISTORY + slots * _HISTORY_SLOT_DIM
        out[rows, base + _HISTORY_SEAT_OFF_REL + rel_seats] = 1.0

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

        street_ok = (street_flat >= 0) & (street_flat < _NUM_STREET_ONEHOT)
        if street_ok.any():
            out[
                rows[street_ok],
                base[street_ok] + _HISTORY_STREET_OFF_REL + street_flat[street_ok],
            ] = 1.0

        out[rows, base + _HISTORY_CHIPS_OFF_REL] = chips_flat.astype(np.float64) * inv_bb
        pot_chips_i64 = obs_arrays["pot"].astype(np.int64)
        suffix = np.cumsum(history_chips[:, ::-1], axis=1)[:, ::-1]
        pot_before = pot_chips_i64[:, None] - suffix
        pot_before_flat = pot_before[valid_slots]
        frac = chips_flat.astype(np.float64) / np.maximum(
            pot_before_flat, 1
        ).astype(np.float64)
        out[rows, base + _HISTORY_FRAC_OFF_REL] = np.clip(frac, 0.0, 2.0)

    out[live_mask, _M_SEAT_EXISTS : _M_SEAT_EXISTS + num_seats] = 1.0

    total_commit = obs_arrays["total_commit"].astype(np.float64)
    street_commit = obs_arrays["street_commit"].astype(np.float64)
    tc_rot = np.take_along_axis(total_commit, rot, axis=1)
    sc_rot = np.take_along_axis(street_commit, rot, axis=1)
    out[live_mask, _M_TOTAL_COMMIT : _M_TOTAL_COMMIT + num_seats] = (
        tc_rot[live_mask] * inv_bb
    )
    out[live_mask, _M_STREET_COMMIT : _M_STREET_COMMIT + num_seats] = (
        sc_rot[live_mask] * inv_bb
    )

    button = obs_arrays["button"].astype(np.int64)
    btn_rel = (button - hero_idx) % num_seats
    out[live_mask, _M_HERO_BTN + btn_rel[live_mask]] = 1.0
    return out


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


#: (10, 13) bool matrix form of _STRAIGHT_WINDOWS for the batched path;
#: row order matches the tuple (wheel first → broadway last, i.e. sorted
#: by the straight's top rank).
_WINDOW_MATRIX: np.ndarray = np.zeros((10, 13), dtype=bool)
for _wi, _w in enumerate(_STRAIGHT_WINDOWS):
    for _r in _w:
        _WINDOW_MATRIX[_wi, _r] = True
del _wi, _w, _r

# Python-int window bitmasks (same order as _STRAIGHT_WINDOWS). Used by the
# serial BRD-5/6 / DUAL-5 hot paths so they avoid frozenset allocs while
# staying bit-exact with the set formulation (popcount == set length).
_WINDOW_BITS_PY: tuple[int, ...] = tuple(
    sum(1 << r for r in W) for W in _STRAIGHT_WINDOWS
)
_PAIR_BITS_PY: tuple[int, ...] = tuple(
    (1 << r1) | (1 << r2) for r1 in range(13) for r2 in range(r1 + 1, 13)
)
_MASK13 = 0x1FFF  # low 13 rank bits


def _ranks_mask(cards) -> int:
    """13-bit presence mask of ranks in an iterable of card indices."""
    m = 0
    for c in cards:
        m |= 1 << (int(c) // 4)
    return m


def _danger_straight_outs(
    board_mask: int, hole_mask: int, unseen_per_rank
) -> int:
    """BRD-5 danger_straight: count of unseen rank-copies that open a
    field straight hero does not currently make. Bit-exact with the
    frozenset formulation (popcount ≡ set size)."""
    danger = 0
    for r in range(13):
        ur = int(unseen_per_rank[r])
        if ur <= 0:
            continue
        new_board = board_mask | (1 << r)
        field = False
        for W in _WINDOW_BITS_PY:
            if (W & new_board).bit_count() >= 3:
                field = True
                break
        if not field:
            continue
        hero_makes = False
        for W in _WINDOW_BITS_PY:
            L = W & ~new_board
            nL = L.bit_count()
            nH = (W & hole_mask).bit_count()
            if nH >= 2 and nL <= 2 and (L & ~hole_mask) == 0:
                hero_makes = True
                break
        if not hero_makes:
            danger += ur
    return danger


def _straight_out_union(
    board_mask: int, hole_mask: int, unseen_per_rank
) -> tuple[int, int]:
    """BRD-6 (union_outs, nut_outs). Bit-exact with the frozenset path."""
    union_mask = 0
    for W in _WINDOW_BITS_PY:
        L = W & ~board_mask
        nL = L.bit_count()
        nH = (W & hole_mask).bit_count()
        makes = (L & ~hole_mask) == 0 and nH >= 2 and nL <= 2
        if makes or nH < 2:
            continue
        M = L & ~hole_mask
        nM = M.bit_count()
        if nL == 3 and nM == 0:
            union_mask |= L
        elif nM == 1 and 1 <= nL <= 3:
            union_mask |= M
    union_outs = 0
    nut_outs = 0
    for r in range(13):
        if not (union_mask >> r) & 1:
            continue
        ur = int(unseen_per_rank[r])
        union_outs += ur
        new_board = board_mask | (1 << r)
        h_max = -1
        for wi, W in enumerate(_WINDOW_BITS_PY):
            L = W & ~new_board
            nL = L.bit_count()
            nH = (W & hole_mask).bit_count()
            if (L & ~hole_mask) == 0 and nH >= 2 and nL <= 2:
                if wi > h_max:
                    h_max = wi
        if h_max < 0:
            continue
        nut_dist = 0
        for wi in range(h_max + 1, 10):
            if (_WINDOW_BITS_PY[wi] & new_board).bit_count() >= 3:
                nut_dist += 1
        if nut_dist == 0:
            nut_outs += ur
    return union_outs, nut_outs


def _scoop_pair_count(ba_mask: int, bb_mask: int) -> int:
    """DUAL-5 scoop-pair count: # of 2-rank pairs that complete a straight
    on BOTH boards. Bit-exact with the frozenset double-loop."""
    scoop = 0
    for pair in _PAIR_BITS_PY:
        made_a = False
        made_b = False
        for W in _WINDOW_BITS_PY:
            if (pair & W) != pair:
                continue
            if (pair | (ba_mask & W)).bit_count() >= 5:
                made_a = True
            if (pair | (bb_mask & W)).bit_count() >= 5:
                made_b = True
            if made_a and made_b:
                break
        if made_a and made_b:
            scoop += 1
    return scoop


def _blocker_features(hole_idx: list[int], board_idx: list[int]) -> np.ndarray:
    """Unconditional blockers-to-nuts for ONE board (obs v2 P3, 4 dims).

    "Unconditional": hero need not hold the made hand or the draw — the
    pre-v5 flush/straight digests only fired when hero was drawing or
    made, leaving bare-blocker information (the bluff-selection signal)
    recoverable only from the raw 52-bit multi-hots. Dims:

    0. hero holds the TOP missing card of the board's flush suit (a suit
       with >= 3 board cards; two such suits cannot coexist on 5 cards).
       0 when no flush is possible.
    1. count of the top-3 missing flush-suit cards hero holds, / 3.
    2. hero cards whose rank completes the NUT straight (the highest
       5-rank window where the board supplies >= 3 distinct ranks),
       counted over the window's missing ranks, / 4, clipped to 1.
       0 when no straight is possible.
    3. hero cards matching the board's highest PAIRED rank, / 2 (the
       trips/boat blocker). 0 on unpaired boards.
    """
    out = np.zeros(4, dtype=np.float32)
    if len(board_idx) < 3:
        return out

    board_rank_counts = [0] * 13
    board_suit_counts = [0] * 4
    board_suit_ranks: list[set[int]] = [set(), set(), set(), set()]
    for c in board_idx:
        r, s = c // 4, c % 4
        board_rank_counts[r] += 1
        board_suit_counts[s] += 1
        board_suit_ranks[s].add(r)
    hero_rank_counts = [0] * 13
    hero_cards = set(hole_idx)
    for c in hole_idx:
        hero_rank_counts[c // 4] += 1

    # Flush blockers.
    for s in range(4):
        if board_suit_counts[s] >= 3:
            missing = [r for r in range(12, -1, -1) if r not in board_suit_ranks[s]]
            if missing and (missing[0] * 4 + s) in hero_cards:
                out[0] = 1.0
            held = sum(1 for r in missing[:3] if (r * 4 + s) in hero_cards)
            out[1] = held / 3.0
            break

    # Nut-straight blockers: highest qualifying window (tuple is ordered
    # by top rank, so scan from the end).
    board_rank_set = {r for r in range(13) if board_rank_counts[r] > 0}
    for w in reversed(_STRAIGHT_WINDOWS):
        if len(w & board_rank_set) >= 3:
            missing_ranks = w - board_rank_set
            blockers = sum(hero_rank_counts[r] for r in missing_ranks)
            out[2] = min(blockers, 4) / 4.0
            break

    # Board-pair blockers: highest paired rank.
    for r in range(12, -1, -1):
        if board_rank_counts[r] >= 2:
            out[3] = min(hero_rank_counts[r], 2) / 2.0
            break

    return out


def _blocker_features_batch(
    hole: np.ndarray,
    hole_valid: np.ndarray,
    board: np.ndarray,
    board_valid: np.ndarray,
) -> np.ndarray:
    """Vectorized `_blocker_features` for one board: (N, 4) float32.
    Bit-exact vs the scalar helper (integer counts, identical f64
    divisions, same first-match tie-breaks: argmax on a reversed mask ==
    the scalar's descending-rank / tuple-order scans)."""
    n = hole.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    has_board = board_valid.sum(axis=1) >= 3
    if not has_board.any():
        return out
    rows = np.arange(n)

    board_presence = np.zeros((n, 13, 4), dtype=bool)
    ei, si = np.nonzero(board_valid)
    cards = board[ei, si].astype(np.int64)
    board_presence[ei, cards >> 2, cards & 3] = True
    hero_presence = np.zeros((n, 13, 4), dtype=bool)
    ei, si = np.nonzero(hole_valid)
    hcards = hole[ei, si].astype(np.int64)
    hero_presence[ei, hcards >> 2, hcards & 3] = True

    # Flush blockers (unique suit with >= 3 board cards, when it exists).
    board_suit_counts = board_presence.sum(axis=1)
    flush_suit_exists = board_suit_counts >= 3
    has_flush = flush_suit_exists.any(axis=1)
    suit_idx = np.argmax(flush_suit_exists, axis=1)
    suit_board_desc = board_presence[rows, :, suit_idx][:, ::-1]  # idx 0 = rank 12
    suit_hero_desc = hero_presence[rows, :, suit_idx][:, ::-1]
    missing_desc = ~suit_board_desc
    cum = np.cumsum(missing_desc, axis=1)
    top1 = missing_desc & (cum <= 1)
    top3 = missing_desc & (cum <= 3)
    out[:, 0] = ((suit_hero_desc & top1).any(axis=1) & has_flush).astype(np.float32)
    out[:, 1] = np.where(has_flush, (suit_hero_desc & top3).sum(axis=1) / 3.0, 0.0)

    # Nut-straight blockers.
    board_rank_mask = board_presence.any(axis=2)
    win_counts = board_rank_mask.astype(np.int8) @ _WINDOW_MATRIX.T.astype(np.int8)
    qualifying = win_counts >= 3
    has_straight = qualifying.any(axis=1)
    nut_idx = (len(_STRAIGHT_WINDOWS) - 1) - np.argmax(qualifying[:, ::-1], axis=1)
    missing_ranks = _WINDOW_MATRIX[nut_idx] & ~board_rank_mask
    hero_rank_counts = hero_presence.sum(axis=2)
    blockers = (hero_rank_counts * missing_ranks).sum(axis=1)
    out[:, 2] = np.where(has_straight, np.minimum(blockers, 4) / 4.0, 0.0)

    # Board-pair blockers (highest paired rank).
    board_rank_counts = board_presence.sum(axis=2)
    paired = board_rank_counts >= 2
    has_pair = paired.any(axis=1)
    pair_rank = 12 - np.argmax(paired[:, ::-1], axis=1)
    pair_block = hero_rank_counts[rows, pair_rank]
    out[:, 3] = np.where(has_pair, np.minimum(pair_block, 2) / 2.0, 0.0)

    out[~has_board] = 0.0
    return out


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


# ---- obs v3 batch-2 tail helpers (serial) ----------------------------------
# Each writes ONLY its own tail columns into the pre-zeroed `out`. Kept as
# isolated functions so the batched twins can be verified block-by-block and
# so the three categories don't collide. [ENGINE] sub-blocks stay 0.0 until
# Chunk B. Every dim's spec is V7_OBS_CANDIDATES.md; parity twin is the
# `_..._batch` function below. Bodies filled 2026-07-12.


def _encode_stack_v3(
    out: np.ndarray,
    *,
    config: GameConfig,
    hero: int,
    num_seats: int,
    folded,
    all_in,
    eff_per_seat,
    total_commit,
    street_commit,
    pot: float,
    btc: float,
    min_bet: float,
    max_bet: float,
    to_call: float,
    eff_to_call: float,
    hero_stack: float,
    inv_bb: float,
    street_idx: int,
    acted=None,
) -> None:
    """Stack/pot/price geometry, dims 1020..1061. `acted` is the engine's
    per-seat acted_this_street bits (None on pre-v7 fixtures → STK-1 zero)."""
    # Shared hero-frame scalars.
    hero_sc = float(street_commit[hero])
    hero_commit = float(total_commit[hero])
    pot_denom = max(pot, 1.0)

    # ---- STK-1: money / raise-exposure still behind (4) @ _STK1_OFF ----
    # Pending set (engine pending rule) over OPPONENTS: alive, not all-in,
    # and (not yet acted this street OR street commit below the live bet
    # level). Aggregates the effective money that can still punish a bluff
    # / pay off a value bet. All-zero when the pending set is empty (hero
    # closes / betting dead).
    if acted is not None:
        max_cap = 0.0
        sum_cap = 0.0
        max_eff = 0.0
        any_pending = False
        for s in range(num_seats):
            if s == hero or folded[s] or all_in[s]:
                continue
            if acted[s] and float(street_commit[s]) >= btc:
                continue
            any_pending = True
            eff_s = eff_per_seat[s]
            owed = min(max(btc - float(street_commit[s]), 0.0), eff_s)
            cap = max(0.0, eff_s - owed)
            if cap > max_cap:
                max_cap = cap
            sum_cap = sum_cap + cap
            if eff_s > max_eff:
                max_eff = eff_s
        if any_pending:
            out[_STK1_OFF + 0] = np.log1p(max_cap / pot_denom)
            out[_STK1_OFF + 1] = np.log1p(sum_cap / pot_denom)
            out[_STK1_OFF + 2] = np.log1p(max_eff * inv_bb)
            out[_STK1_OFF + 3] = 1.0 if max_eff >= hero_stack else 0.0

    # ---- STK-2: raise-ladder envelope (6) @ _STK2_OFF ----
    # min_d/max_d are the raise DELTAS (additional chips) recovered from the
    # already-packed totals; identical to sizing_from_info's min/max_raise_chips.
    min_d = min_bet - hero_sc
    max_d = max_bet - hero_sc
    base = pot + to_call
    raise_legal = max_d > to_call
    if raise_legal:
        base_safe = max(base, 1.0)
        eff_denom = max(hero_stack, 1.0)
        out[_STK2_OFF + 0] = min(max((min_d - to_call) / base_safe, 0.0), 1.0)
        out[_STK2_OFF + 1] = min(max((max_d - to_call) / base_safe, 0.0), 1.0)
        out[_STK2_OFF + 2] = min(max(min_d / eff_denom, 0.0), 1.0)
        out[_STK2_OFF + 3] = min(max(max_d / eff_denom, 0.0), 1.0)
        out[_STK2_OFF + 4] = 1.0 if max_d < to_call + base else 0.0
        # Legal-anchor count only (no brackets) — bit-identical to
        # anchor_grid_np(...).legal.sum() / ANCHOR_COUNT.
        out[_STK2_OFF + 5] = (
            float(n_legal_anchors_np(min_d, max_d, pot, to_call))
            / float(ANCHOR_COUNT)
        )

    # ---- STK-4: per-seat commitment ratio (8) @ _STK4_OFF, hero-rotated ----
    # RAW commit / EFFECTIVE stack, matching dim-1009's exact recipe; folded
    # and padded slots read 0, all-in live seats read 1.0 (eff==0).
    for k in range(num_seats):
        seat = (hero + k) % num_seats
        if folded[seat]:
            continue
        commit_s = float(total_commit[seat])
        denom4 = commit_s + eff_per_seat[seat]
        if denom4 > 0.0:
            out[_STK4_OFF + k] = commit_s / denom4

    # ---- STK-5: spr-after-action (4) @ _STK5_OFF ----
    out[_STK5_OFF + 0] = np.log1p(
        (hero_stack - eff_to_call) / max(pot + eff_to_call, 1.0)
    )
    out[_STK5_OFF + 1] = np.log1p((pot + eff_to_call) * inv_bb)
    if raise_legal:  # D == max_d (max-raise delta); same predicate as STK-2
        tot2 = pot + 2.0 * max_d - to_call
        out[_STK5_OFF + 2] = np.log1p((hero_stack - max_d) / max(tot2, 1.0))
        out[_STK5_OFF + 3] = np.log1p(tot2 * inv_bb)

    # ---- STK-6: geometric jam plan (2) @ _STK6_OFF ----
    # r = streets remaining incl. current (flop 3 / turn 2 / river 1);
    # preflop (only terminal, masked) would give r=4 — pinned identically.
    spr_e = hero_stack / pot_denom
    r = 4 - street_idx
    x6 = 1.0 + 2.0 * spr_e
    out[_STK6_OFF + 0] = min(max(np.ceil(np.log(x6) / np.log(3.0)), 0.0), 6.0)
    out[_STK6_OFF + 1] = min(max((np.power(x6, 1.0 / r) - 1.0) / 2.0, 0.0), 2.0)

    # ---- STK-7: pot-ceiling implied odds (2) @ _STK7_OFF ----
    # ceiling = pot + sum over live (non-folded) opponents of min(eff_opp,
    # eff_hero); all-in opponents contribute 0 (their eff==0). Seat-index
    # accumulation order matches the batched twin exactly.
    ceiling = pot
    for s in range(num_seats):
        if s == hero or folded[s]:
            continue
        ceiling = ceiling + min(eff_per_seat[s], hero_stack)
    out[_STK7_OFF + 0] = np.log1p(ceiling / pot_denom)
    if eff_to_call > 0.0:
        out[_STK7_OFF + 1] = eff_to_call / (ceiling + eff_to_call)

    # ---- STK-8: side-pot eligibility (3) @ _STK8_OFF ----
    # RAW commits throughout (side-pot math is about chips actually in pot).
    hero_after = hero_commit + eff_to_call
    sum_now = 0.0
    sum_after = 0.0
    dead = 0.0
    for s in range(num_seats):
        cs = float(total_commit[s])
        sum_now = sum_now + min(cs, hero_commit)
        sum_after = sum_after + min(cs, hero_after)
        if folded[s]:
            dead = dead + cs
    out[_STK8_OFF + 0] = sum_now / pot_denom
    out[_STK8_OFF + 1] = sum_after / pot_denom
    out[_STK8_OFF + 2] = dead / pot_denom

    # ---- STK-9: call-risk fraction (2) @ _STK9_OFF ----
    out[_STK9_OFF + 0] = eff_to_call / max(hero_stack, 1.0)
    out[_STK9_OFF + 1] = eff_to_call / max(hero_commit + hero_stack, 1.0)

    # ---- STK-10: ante-pot bloat (2) @ _STK10_OFF ----
    # pure config arithmetic; frozen, history-independent (truncation-immune).
    ante_i = int(config.ante)
    pot_at_flop = 0
    for s in range(num_seats):
        pot_at_flop += min(ante_i, int(config.resolved_stacks[s]))
    paf = float(pot_at_flop)
    out[_STK10_OFF + 0] = ante_i * inv_bb
    out[_STK10_OFF + 1] = np.log1p(max(pot - paf, 0.0) / max(paf, 1.0))

    # ---- STK-11: per-seat price-to-continue (8) @ _STK11_OFF, hero-rotated ----
    # slot k = owed_k / (max(pot,1) + owed_k); hero (slot 0), folded, all-in
    # (eff==0 -> owed 0), and no-outstanding-bet all read 0.
    for k in range(1, num_seats):
        seat = (hero + k) % num_seats
        if folded[seat]:
            continue
        owed = min(max(btc - float(street_commit[seat]), 0.0), eff_per_seat[seat])
        out[_STK11_OFF + k] = owed / (pot_denom + owed)


def _encode_board_v3(
    out: np.ndarray,
    *,
    hole_list,
    board_a_list,
    board_b_list,
    visible_count,
    street_idx: int,
    hero_board_v3=None,
    board_draw_v3=None,
) -> None:
    """Board texture + hand-board combinatorics, dims 1061..1139.
    `hero_board_v3` is the engine's 8-int block [boat_a, boat_b, improve_a,
    improve_b, combos_a, combos_b, mask_a, mask_b] (None on pre-v7
    fixtures → BRD-7/BRD-12 stay zero; masks are consumed by the DUAL
    helper, not here).
    `board_draw_v3` is the engine's 7-int hot block
    [ds_a, ds_b, u_a, n_a, u_b, n_b, scoop] (None → Python BRD-5/6)."""
    # ---- Shared hero-side aggregates (board-agnostic) ----
    hole_rank_counts = [0] * 13
    hole_suit_counts = [0] * 4
    hole_ranks_set: set[int] = set()
    hole_max_per_suit = [-1, -1, -1, -1]
    for c in hole_list:
        r, s = c // 4, c % 4
        hole_rank_counts[r] += 1
        hole_suit_counts[s] += 1
        hole_ranks_set.add(r)
        if r > hole_max_per_suit[s]:
            hole_max_per_suit[s] = r

    vct = visible_count.sum(axis=1)              # (13,) global visible copies per rank
    visible_per_suit = visible_count.sum(axis=0)  # (4,) global visible per suit
    unseen_deck = 52 - int(visible_count.sum())

    # ---- BRD-7 / BRD-12: engine-emitted hero/board dims ----
    # Raw counts from GameState::hero_board_v3 (engine handles the FH+ gate,
    # the river-zeroing of out counts, and the global-unseen convention);
    # the encoder applies the spec normalizations: BRD-7 raw, BRD-12 outs
    # / actual unseen-deck size, combo redundancy / 10.
    if hero_board_v3 is not None:
        out[_BRD7_OFF + 0] = float(hero_board_v3[0])
        out[_BRD7_OFF + 1] = float(hero_board_v3[1])
        out[_BRD12_OFF + 0] = float(hero_board_v3[2]) / unseen_deck
        out[_BRD12_OFF + 1] = float(hero_board_v3[3]) / unseen_deck
        # /10 is pinned to PLO5's C(5,2) pairs. PLO4 tops out at 0.6 and
        # PLO6 can exceed 1.0 (15 pairs) — both variants are untrained;
        # revisit the divisor before ever training them at this layout.
        out[_BRD12_OFF + 2] = float(hero_board_v3[4]) / 10.0
        out[_BRD12_OFF + 3] = float(hero_board_v3[5]) / 10.0

    # Cross-board presence (BRD-10 key-card scan spans BOTH boards) + hero holdings.
    board_present_all = [[False] * 4 for _ in range(13)]
    for c in board_a_list + board_b_list:
        board_present_all[c // 4][c % 4] = True
    hero_present = [[False] * 4 for _ in range(13)]
    for c in hole_list:
        hero_present[c // 4][c % 4] = True

    is_river = street_idx == 3
    is_flop = street_idx == 1

    def _one_board(board_list, bi):
        board_rank_counts = [0] * 13
        board_suit_counts = [0] * 4
        board_ranks_per_suit = [set(), set(), set(), set()]
        for c in board_list:
            r, s = c // 4, c % 4
            board_rank_counts[r] += 1
            board_suit_counts[s] += 1
            board_ranks_per_suit[s].add(r)
        board_ranks_set = {r for r in range(13) if board_rank_counts[r] > 0}
        has_board = len(board_list) > 0

        # ---- BRD-1: rank ladder (5) ----
        o1 = _BRD1_OFF + bi * 5
        sorted_ranks = sorted((c // 4 for c in board_list), reverse=True)
        for i, rank in enumerate(sorted_ranks[:5]):
            out[o1 + i] = (rank + 1) / 13.0

        # ---- BRD-2: suit census (6) ----
        o2 = _BRD2_OFF + bi * 6
        for s in range(4):
            if board_suit_counts[s] == 2:
                out[o2 + s] = 1.0
        if any(bc == 4 for bc in board_suit_counts):
            out[o2 + 4] = 1.0
        if any(bc == 5 for bc in board_suit_counts):
            out[o2 + 5] = 1.0

        # ---- BRD-4: arrival volatility census (3), river-zeroed ----
        o4 = _BRD4_OFF + bi * 3
        if not is_river:
            pair_outs = sum(4 - int(vct[r]) for r in board_ranks_set)
            flush_adv = sum(
                13 - int(visible_per_suit[s])
                for s in range(4)
                if board_suit_counts[s] in (2, 3)
            )
            straight_adv = 0
            for r in range(13):
                if r in board_ranks_set:
                    continue
                unseen_r = 4 - int(vct[r])
                if unseen_r <= 0:
                    continue
                for W in _STRAIGHT_WINDOWS:
                    if r in W and len(W & board_ranks_set) == 2:
                        straight_adv += unseen_r
                        break
            out[o4 + 0] = pair_outs / unseen_deck
            out[o4 + 1] = flush_adv / unseen_deck
            out[o4 + 2] = straight_adv / unseen_deck

        # ---- BRD-5: hero vulnerability outs (3), river-zeroed ----
        # danger_flush / danger_pair stay pure-Python (cheap); danger_straight
        # uses the engine board_draw_v3 block when present (Rust hot path).
        o5 = _BRD5_OFF + bi * 3
        if not is_river:
            danger_flush = sum(
                13 - int(visible_per_suit[s])
                for s in range(4)
                if board_suit_counts[s] == 2 and hole_suit_counts[s] < 2
            )
            danger_pair = sum(
                4 - int(vct[r])
                for r in board_ranks_set
                if hole_rank_counts[r] == 0
            )
            if board_draw_v3 is not None:
                danger_straight = float(board_draw_v3[bi])  # ds_a / ds_b
            else:
                board_mask = 0
                for r in board_ranks_set:
                    board_mask |= 1 << r
                hole_mask = 0
                for r in hole_ranks_set:
                    hole_mask |= 1 << r
                unseen_per_rank = [4 - int(vct[r]) for r in range(13)]
                danger_straight = _danger_straight_outs(
                    board_mask, hole_mask, unseen_per_rank
                )
            out[o5 + 0] = danger_flush / unseen_deck
            out[o5 + 1] = danger_pair / unseen_deck
            out[o5 + 2] = danger_straight / unseen_deck

        # ---- BRD-6: straight out union (2), river-zeroed ----
        o6 = _BRD6_OFF + bi * 2
        if not is_river:
            if board_draw_v3 is not None:
                # layout: [ds_a, ds_b, u_a, n_a, u_b, n_b, scoop]
                union_outs = float(board_draw_v3[2 + bi * 2])
                nut_outs = float(board_draw_v3[3 + bi * 2])
            else:
                board_mask = 0
                for r in board_ranks_set:
                    board_mask |= 1 << r
                hole_mask = 0
                for r in hole_ranks_set:
                    hole_mask |= 1 << r
                unseen_per_rank = [4 - int(vct[r]) for r in range(13)]
                union_outs, nut_outs = _straight_out_union(
                    board_mask, hole_mask, unseen_per_rank
                )
            out[o6 + 0] = float(union_outs)
            out[o6 + 1] = float(nut_outs)

        # ---- BRD-8: fd rank quality (2) ----
        o8 = _BRD8_OFF + bi * 2
        draw_suit = -1
        best_max = -1
        for s in range(4):
            if hole_suit_counts[s] >= 2 and board_suit_counts[s] == 2:
                if hole_max_per_suit[s] > best_max:
                    best_max = hole_max_per_suit[s]
                    draw_suit = s
        if draw_suit >= 0:
            h1 = hole_max_per_suit[draw_suit]
            out[o8 + 0] = (h1 + 1) / 13.0
            out[o8 + 1] = float(
                sum(
                    1
                    for r in range(h1 + 1, 13)
                    if int(visible_count[r, draw_suit]) == 0
                )
            )

        # ---- BRD-9: backdoor draw census (2), flop-only ----
        o9 = _BRD9_OFF + bi * 2
        if is_flop:
            bdfd = sum(
                1
                for s in range(4)
                if hole_suit_counts[s] >= 2 and board_suit_counts[s] == 1
            )
            bdstr = 0
            for W in _STRAIGHT_WINDOWS:
                nH = len(W & hole_ranks_set)
                nL = len(W - board_ranks_set)
                missing_both = len(W - board_ranks_set - hole_ranks_set)
                if missing_both == 2 and nH >= 2 and nL <= 4:
                    bdstr += 1
            out[o9 + 0] = float(bdfd)
            out[o9 + 1] = float(bdstr)

        # ---- BRD-10: future nut-flush blocker (1), river-zeroed ----
        o10 = _BRD10_OFF + bi * 1
        if not is_river:
            cnt = 0
            for s in range(4):
                if board_suit_counts[s] != 2:
                    continue
                for r in range(12, -1, -1):
                    if board_present_all[r][s]:
                        continue
                    if hero_present[r][s]:
                        cnt += 1
                    break
            out[o10] = float(cnt)

        # ---- BRD-11: turn/river card identity (5 per slot; A-turn,A-river) ----
        o11t = _BRD11_OFF + bi * 10 + 0
        o11r = _BRD11_OFF + bi * 10 + 5
        if len(board_list) >= 4:
            c = board_list[3]
            out[o11t + 0] = (c // 4 + 1) / 13.0
            out[o11t + 1 + (c % 4)] = 1.0
        if len(board_list) >= 5:
            c = board_list[4]
            out[o11r + 0] = (c // 4 + 1) / 13.0
            out[o11r + 1 + (c % 4)] = 1.0

        # ---- BRD-13: board nut ceiling class (2) ----
        o13 = _BRD13_OFF + bi * 2
        if has_board:
            sf_possible = False
            for s in range(4):
                ranks_s = board_ranks_per_suit[s]
                if len(ranks_s) >= 3:
                    for W in _STRAIGHT_WINDOWS:
                        if len(W & ranks_s) >= 3:
                            sf_possible = True
                            break
                if sf_possible:
                    break
            paired = any(bc >= 2 for bc in board_rank_counts)
            flush_poss = any(bc >= 3 for bc in board_suit_counts)
            straight_poss = any(
                len(W & board_ranks_set) >= 3 for W in _STRAIGHT_WINDOWS
            )
            if sf_possible:
                ceil = 8
            elif paired:
                ceil = 7
            elif flush_poss:
                ceil = 5
            elif straight_poss:
                ceil = 4
            else:
                ceil = 3
            out[o13 + 0] = 1.0 if sf_possible else 0.0
            out[o13 + 1] = ceil / 8.0

    _one_board(board_a_list, 0)
    _one_board(board_b_list, 1)


def _encode_dual_v3(
    out: np.ndarray,
    *,
    hole_list,
    board_a_list,
    board_b_list,
    visible_count,
    per_board_outcome,
    pot: float,
    to_call: float,
    eff_to_call: float,
    hero_stack: float,
    config: GameConfig,
    hero_board_v3=None,
    share_bounds=None,
    board_draw_v3=None,
) -> None:
    """Double-board structure, dims 1139..1171. `hero_board_v3` carries the
    engine best-holding masks at [6]/[7] (DUAL-2); `share_bounds` is the
    fused pass's [g_min, g_max] (DUAL-4). `board_draw_v3` supplies the
    DUAL-5 scoop raw count at [6] when present. None → those blocks stay
    zero / Python fallback (pre-v7 fixtures)."""
    # ---- DUAL-1 (1139..1141): split-adjusted price ladder ----
    # Pot odds re-denominated to the win-one (0.5*pot) and quartered
    # (0.25*pot) split outcomes, stack-capped (eff_to_call already =
    # min(to_call, hero effective stack)). Both 0 when eff_to_call == 0.
    if eff_to_call > 0.0:
        out[_DUAL1_OFF + 0] = eff_to_call / (0.5 * pot + eff_to_call)
        out[_DUAL1_OFF + 1] = eff_to_call / (0.25 * pot + eff_to_call)

    # ---- DUAL-3 (1151..1157): nut-lock / freeroll flags ----
    # Exact ==0.0 thresholds of the k=2 per-board counters (per_board_outcome
    # layout [aheadA,tieA,behindA,aheadB,tieB,behindB,win-one,tie-both]).
    # Guarded by the activity gate (per-board fractions sum ~1 when the pass
    # ran, 0 preflop/terminal) so terminal all-zeros never read as "nuts".
    active = False
    if per_board_outcome is not None:
        ahead_a = float(per_board_outcome[0])
        tie_a = float(per_board_outcome[1])
        behind_a = float(per_board_outcome[2])
        tie_b = float(per_board_outcome[4])
        behind_b = float(per_board_outcome[5])
        active = (ahead_a + tie_a + behind_a) > 0.5
        if active:
            nut_or_chop_a = behind_a == 0.0
            nut_or_chop_b = behind_b == 0.0
            out[_DUAL3_OFF + 0] = 1.0 if (nut_or_chop_a and tie_a == 0.0) else 0.0
            out[_DUAL3_OFF + 1] = 1.0 if nut_or_chop_a else 0.0
            out[_DUAL3_OFF + 2] = 1.0 if (nut_or_chop_b and tie_b == 0.0) else 0.0
            out[_DUAL3_OFF + 3] = 1.0 if nut_or_chop_b else 0.0
            out[_DUAL3_OFF + 4] = 1.0 if (nut_or_chop_a and nut_or_chop_b) else 0.0
            out[_DUAL3_OFF + 5] = 1.0 if (nut_or_chop_a or nut_or_chop_b) else 0.0

    # ---- DUAL-4 (1157..1162): guaranteed pot share (k=2 g_min/g_max) ----
    # From the fused pass's appended trackers. [3] prices the CONTESTED
    # slice only (pot·(g_max−g_min)); [4] is the stack-vs-contested-pot
    # log ratio; both use the g_max==g_min sentinel (a fully-decided pot
    # has no contested slice — the [2] flag + DUAL-3 disambiguate).
    if share_bounds is not None and active:
        g_min = float(share_bounds[0])
        g_max = float(share_bounds[1])
        out[_DUAL4_OFF + 0] = g_min
        out[_DUAL4_OFF + 1] = g_max
        out[_DUAL4_OFF + 2] = 1.0 if g_min >= 0.5 else 0.0
        rng = g_max - g_min
        if rng > 0.0:
            if eff_to_call > 0.0:
                out[_DUAL4_OFF + 3] = eff_to_call / (pot * rng + eff_to_call)
            # max(pot, 1): identical for every reachable pot >= 1 chip, but
            # keeps the degenerate ante=0 UI config (pot 0 at the flop) from
            # dividing by zero (serial crash / batched inf divergence).
            out[_DUAL4_OFF + 4] = np.log1p(hero_stack / (max(pot, 1.0) * rng))

    # ---- DUAL-2 (1141..1151): best-holding card usage / coverage ----
    # Engine masks: bit i = the i-th hole card sorted by card index
    # DESCENDING is one of the exactly-2 cards of hero's best holding.
    if hero_board_v3 is not None:
        mask_a = int(hero_board_v3[6])
        mask_b = int(hero_board_v3[7])
        for i in range(5):
            if (mask_a >> i) & 1:
                out[_DUAL2_OFF + i] = 1.0
            if (mask_b >> i) & 1:
                out[_DUAL2_OFF + 5 + i] = 1.0

    # ---- DUAL-5 (1162..1171): villain cross-board coverage (board-only) ----
    # Hero-independent (villain scoop geometry). Per-suit both-boards>=2 (4),
    # both-boards>=3 (4), plus the count of 2-rank pairs completing a straight
    # on BOTH boards / 78. Empty boards yield zeros naturally.
    ba_suit = [0, 0, 0, 0]
    bb_suit = [0, 0, 0, 0]
    for c in board_a_list:
        ba_suit[c % 4] += 1
    for c in board_b_list:
        bb_suit[c % 4] += 1
    for s in range(4):
        if ba_suit[s] >= 2 and bb_suit[s] >= 2:
            out[_DUAL5_OFF + s] = 1.0
        if ba_suit[s] >= 3 and bb_suit[s] >= 3:
            out[_DUAL5_OFF + 4 + s] = 1.0
    if board_draw_v3 is not None:
        scoop_pairs = float(board_draw_v3[6])
    else:
        ba_mask = _ranks_mask(board_a_list)
        bb_mask = _ranks_mask(board_b_list)
        scoop_pairs = float(_scoop_pair_count(ba_mask, bb_mask))
    out[_DUAL5_OFF + 8] = scoop_pairs / 78.0


def encode_observation(obs: Mapping[str, Any], config: GameConfig) -> np.ndarray:
    """Encode a single observation dict into a (OBS_DIM,) float32 array."""
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
    # Pot before each visible action, reconstructed backwards: history
    # chips are per-action DELTAS (antes never recorded), so
    # pot_before(slot j) = current_pot − Σ chips of visible slots ≥ j.
    # Valid under truncation — truncated actions all precede the window.
    pot_now_chips = int(obs["pot"])
    pot_before = [0] * len(history)
    suffix = 0
    for j in range(len(history) - 1, -1, -1):
        suffix += int(history[j][2])
        pot_before[j] = pot_now_chips - suffix
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
        frac = float(chips) / float(max(pot_before[slot], 1))
        out[base + _HISTORY_FRAC_OFF_REL] = min(max(frac, 0.0), 2.0)

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

    # ---- obs v2 tail (V5_DESIGN.md §3.2) --------------------------------
    # Per-board current-rank decomposition (exhaustive k=2 universe, same
    # fused Rust pass as opp_outcome_fractions). Zeros preflop/terminal.
    pb = obs.get("per_board_outcome")
    if pb is not None:
        out[
            _PER_BOARD_OUTCOME_OFF : _PER_BOARD_OUTCOME_OFF + _PER_BOARD_OUTCOME_DIM
        ] = np.asarray(pb, dtype=np.float32)

    # Unconditional blockers-to-nuts per board.
    out[_BLOCKER_A_OFF : _BLOCKER_A_OFF + 4] = _blocker_features(
        hole_list, board_a_list
    )
    out[_BLOCKER_B_OFF : _BLOCKER_B_OFF + 4] = _blocker_features(
        hole_list, board_b_list
    )

    # Effective price: to_call capped by hero's EFFECTIVE remaining stack
    # — the uncapped _POT_ODDS_OFF overstates the price whenever a PL
    # pot-bet covers hero — plus commitment fraction and log1p money
    # companions. Effective (dead-chip-subtracted, same as _STACKS_OFF)
    # rather than raw: chips above max-other-reachable can never be bet,
    # and the whole encoding is invariant to them by contract.
    hero_stack = float(eff_per_seat[hero])
    eff_to_call = min(to_call, hero_stack)
    if eff_to_call > 0.0:
        out[_EFF_PRICE_OFF + 0] = eff_to_call / (pot + eff_to_call)
    if to_call > 0.0 and to_call >= hero_stack:
        out[_EFF_PRICE_OFF + 1] = 1.0
    hero_commit = float(total_commit[hero])
    commit_denom = hero_commit + hero_stack
    if commit_denom > 0.0:
        out[_EFF_PRICE_OFF + 2] = hero_commit / commit_denom
    out[_EFF_PRICE_OFF + 3] = np.log1p(eff_to_call * inv_bb)
    out[_EFF_PRICE_OFF + 4] = np.log1p(pot * inv_bb)

    # log1p effective SPR, UNCLIPPED (see _SPR_LOG_OFF comment).
    for k in range(num_seats):
        seat = (hero + k) % num_seats
        out[_SPR_LOG_OFF + k] = np.log1p(eff_per_seat[seat] / pot_safe)

    # ---- obs v3 batch-2 tail (stack + board + dual) ---------------------
    hero_board_v3 = obs.get("hero_board_v3")
    board_draw_v3 = obs.get("board_draw_v3")
    _encode_stack_v3(
        out,
        config=config,
        hero=hero,
        num_seats=num_seats,
        folded=folded,
        all_in=all_in,
        eff_per_seat=eff_per_seat,
        total_commit=total_commit,
        street_commit=street_commit,
        pot=pot,
        btc=btc,
        min_bet=float(obs["min_bet"]),
        max_bet=float(obs["max_bet"]),
        to_call=to_call,
        eff_to_call=eff_to_call,
        hero_stack=hero_stack,
        inv_bb=inv_bb,
        street_idx=int(obs["street"]),
        acted=obs.get("acted_this_street"),
    )
    _encode_board_v3(
        out,
        hole_list=hole_list,
        board_a_list=board_a_list,
        board_b_list=board_b_list,
        visible_count=visible_count,
        street_idx=int(obs["street"]),
        hero_board_v3=hero_board_v3,
        board_draw_v3=board_draw_v3,
    )
    _encode_dual_v3(
        out,
        hole_list=hole_list,
        board_a_list=board_a_list,
        board_b_list=board_b_list,
        visible_count=visible_count,
        per_board_outcome=pb,
        pot=pot,
        to_call=to_call,
        eff_to_call=eff_to_call,
        hero_stack=hero_stack,
        config=config,
        hero_board_v3=hero_board_v3,
        share_bounds=obs.get("share_bounds"),
        board_draw_v3=board_draw_v3,
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
    for k in range(hole.shape[1]):
        vh = hole_valid[:, k]
        if vh.any():
            np.add.at(
                hole_suit_counts,
                (np.nonzero(vh)[0], hole_suits[vh, k]),
                1,
            )
    for k in range(5):
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
    for k in range(hole.shape[1]):
        vh = hole_valid[:, k]
        if vh.any():
            rank_mask[np.nonzero(vh)[0], hole_ranks[vh, k]] = True
    for k in range(5):
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
    for k in range(hole.shape[1]):
        vh = hole_valid[:, k]
        if vh.any():
            np.add.at(
                hole_rank_counts,
                (np.nonzero(vh)[0], hole_ranks[vh, k]),
                1,
            )
    for k in range(5):
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
    for k in range(hole.shape[1]):
        vh = hole_valid[:, k]
        if vh.any():
            idx = np.nonzero(vh)[0]
            hole_rank_mask[idx, hole_ranks[vh, k]] = True
            hole_rank_suit[idx, hole_ranks[vh, k], hole_suits[vh, k]] = True
            np.add.at(hole_suit_counts, (idx, hole_suits[vh, k]), 1)
    for k in range(5):
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
    for k in range(hole.shape[1]):
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
    # one-hot + 1 chips/bb scalar + 1 chips/pot-before scalar. Gate derived
    # from (action, chips) per `_gate_from_action`: Fold→0,
    # CheckCall&chips==0→Check, CheckCall&chips>0→Call, anything else→Raise.
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

        # Pot before each visible action (per-action deltas; padded slots
        # are zero chips so the reversed cumsum is unaffected). Mirrors
        # the scalar path's suffix-sum reconstruction exactly.
        pot_chips_i64 = obs_arrays["pot"].astype(np.int64)
        suffix = np.cumsum(history_chips[:, ::-1], axis=1)[:, ::-1]
        pot_before = pot_chips_i64[:, None] - suffix
        pot_before_flat = pot_before[valid_slots]
        frac = chips_flat.astype(np.float64) / np.maximum(
            pot_before_flat, 1
        ).astype(np.float64)
        out[rows, base + _HISTORY_FRAC_OFF_REL] = np.clip(frac, 0.0, 2.0)

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
    for k in range(hole.shape[1]):
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
    # Hole and board widths differ under PLO6 (6 vs 5) — iterate separately.
    for k in range(hole.shape[1]):
        vh = hole_valid[:, k]
        if vh.any():
            hole_rank_mask[np.nonzero(vh)[0], hole_ranks_idx[vh, k]] = True
    for k in range(5):
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
        for k in range(src_arr.shape[1]):
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

    # ---- obs v2 tail (V5_DESIGN.md §3.2) — mirrors the scalar encoder ----
    pb = obs_arrays.get("per_board_outcome")
    if pb is not None:
        out[
            :, _PER_BOARD_OUTCOME_OFF : _PER_BOARD_OUTCOME_OFF + _PER_BOARD_OUTCOME_DIM
        ] = pb.astype(np.float32, copy=False)

    blk_a = _blocker_features_batch(hole, hole_valid, ba, ba_valid)
    blk_b = _blocker_features_batch(hole, hole_valid, bb, bb_valid)
    out[live_mask, _BLOCKER_A_OFF : _BLOCKER_A_OFF + 4] = blk_a[live_mask]
    out[live_mask, _BLOCKER_B_OFF : _BLOCKER_B_OFF + 4] = blk_b[live_mask]

    # Effective price (capped by hero's EFFECTIVE stack — dead-chip
    # invariance, see the scalar encoder) + commitment + log1p money.
    # effective_rot column 0 is hero's (rotation starts at hero).
    hero_stack = effective_rot[:, 0]
    eff_to_call = np.minimum(to_call, hero_stack)
    eff_denom = pot + eff_to_call
    eff_odds = np.where(
        eff_to_call > 0.0,
        eff_to_call / np.where(eff_denom > 0.0, eff_denom, 1.0),
        0.0,
    )
    out[live_mask, _EFF_PRICE_OFF + 0] = eff_odds[live_mask]
    allin_call = (to_call > 0.0) & (to_call >= hero_stack)
    out[live_mask, _EFF_PRICE_OFF + 1] = allin_call[live_mask].astype(np.float32)
    hero_commit = np.take_along_axis(total_commit, hero_idx[:, None], axis=1)[:, 0]
    commit_denom = hero_commit + hero_stack
    commit_frac = np.where(
        commit_denom > 0.0,
        hero_commit / np.where(commit_denom > 0.0, commit_denom, 1.0),
        0.0,
    )
    out[live_mask, _EFF_PRICE_OFF + 2] = commit_frac[live_mask]
    out[live_mask, _EFF_PRICE_OFF + 3] = np.log1p(eff_to_call * inv_bb)[live_mask]
    out[live_mask, _EFF_PRICE_OFF + 4] = np.log1p(pot * inv_bb)[live_mask]

    # log1p effective SPR, UNCLIPPED (see _SPR_LOG_OFF comment).
    out[live_mask, _SPR_LOG_OFF : _SPR_LOG_OFF + num_seats] = np.log1p(
        effective_rot / pot_safe[:, None]
    )[live_mask]

    # ---- obs v3 batch-2 tail (stack + board + dual) ---------------------
    # Twins of the serial helpers. All scalar arithmetic in f64, cast on
    # assignment (parity). effective (unrotated, f64, by seat), total_commit
    # (f64, by seat), street_commit_f64, pot/bet_to_call/to_call (f64) are
    # already built above. `street` (int64, by env) is the true obs street.
    _encode_stack_v3_batch(
        out,
        config=config,
        num_seats=num_seats,
        live_mask=live_mask,
        hero_idx=hero_idx,
        folded=folded,
        all_in=all_in,
        effective=effective,
        total_commit=total_commit,
        street_commit=street_commit_f64,
        pot=pot,
        bet_to_call=bet_to_call,
        min_bet=min_bet,
        max_bet=max_bet,
        to_call=to_call,
        inv_bb=inv_bb,
        street=street,
        acted=obs_arrays.get("acted_this_street"),
    )
    _hb_v3 = obs_arrays.get("hero_board_v3")
    _bd_v3 = obs_arrays.get("board_draw_v3")
    _encode_board_v3_batch(
        out,
        live_mask=live_mask,
        hole=hole,
        board_a=ba,
        board_b=bb,
        street=street,
        hero_board_v3=_hb_v3,
        board_draw_v3=_bd_v3,
    )
    _encode_dual_v3_batch(
        out,
        live_mask=live_mask,
        hero_idx=hero_idx,
        hole=hole,
        board_a=ba,
        board_b=bb,
        per_board_outcome=obs_arrays.get("per_board_outcome"),
        pot=pot,
        to_call=to_call,
        effective=effective,
        config=config,
        hero_board_v3=_hb_v3,
        share_bounds=obs_arrays.get("share_bounds"),
        board_draw_v3=_bd_v3,
    )

    return out


# ---- obs v3 batch-2 tail helpers (batched twins) ---------------------------
# Bit-exact twins of the serial _encode_*_v3 helpers. Same f64 arithmetic,
# cast-on-assignment. Write ONLY live_mask rows (terminal rows stay zero).
# [ENGINE] sub-blocks stay 0.0 until Chunk B. Bodies filled 2026-07-12.


def _encode_stack_v3_batch(
    out: np.ndarray,
    *,
    config: GameConfig,
    num_seats: int,
    live_mask: np.ndarray,
    hero_idx: np.ndarray,
    folded: np.ndarray,
    all_in: np.ndarray,
    effective: np.ndarray,
    total_commit: np.ndarray,
    street_commit: np.ndarray,
    pot: np.ndarray,
    bet_to_call: np.ndarray,
    min_bet: np.ndarray,
    max_bet: np.ndarray,
    to_call: np.ndarray,
    inv_bb: float,
    street: np.ndarray,
    acted=None,
) -> None:
    """Batched twin of _encode_stack_v3, dims 1020..1061. `acted` is the
    (N, S) engine acted_this_street array (None → STK-1 stays zero)."""
    n = out.shape[0]
    seats = np.arange(num_seats, dtype=np.int64)
    rot = (hero_idx[:, None] + seats[None, :]) % num_seats  # (n, num_seats)
    folded_rot = np.take_along_axis(folded, rot, axis=1)
    sc_rot = np.take_along_axis(street_commit, rot, axis=1)
    eff_rot = np.take_along_axis(effective, rot, axis=1)
    tc_rot = np.take_along_axis(total_commit, rot, axis=1)
    hero_sc = sc_rot[:, 0]
    hero_stack = eff_rot[:, 0]
    hero_commit = tc_rot[:, 0]
    eff_to_call = np.minimum(to_call, hero_stack)
    pot_denom = np.maximum(pot, 1.0)

    # ---- STK-1: money / raise-exposure still behind (4) ----
    # Same seat-index accumulation order as the serial loop (max is
    # order-free; the sum adds +0.0 for non-pending seats).
    if acted is not None:
        max_cap = np.zeros(n, dtype=np.float64)
        sum_cap = np.zeros(n, dtype=np.float64)
        max_eff = np.zeros(n, dtype=np.float64)
        any_pending = np.zeros(n, dtype=bool)
        for s in range(num_seats):
            pending = (
                (hero_idx != s)
                & ~folded[:, s]
                & ~all_in[:, s]
                & (~acted[:, s] | (street_commit[:, s] < bet_to_call))
            )
            eff_s = effective[:, s]
            owed = np.minimum(
                np.maximum(bet_to_call - street_commit[:, s], 0.0), eff_s
            )
            cap = np.maximum(0.0, eff_s - owed)
            max_cap = np.where(pending, np.maximum(max_cap, cap), max_cap)
            sum_cap = sum_cap + np.where(pending, cap, 0.0)
            max_eff = np.where(pending, np.maximum(max_eff, eff_s), max_eff)
            any_pending |= pending
        wl1 = live_mask & any_pending
        out[wl1, _STK1_OFF + 0] = np.log1p(max_cap / pot_denom)[wl1]
        out[wl1, _STK1_OFF + 1] = np.log1p(sum_cap / pot_denom)[wl1]
        out[wl1, _STK1_OFF + 2] = np.log1p(max_eff * inv_bb)[wl1]
        out[wl1, _STK1_OFF + 3] = (max_eff >= hero_stack).astype(np.float64)[wl1]

    # ---- STK-2: raise-ladder envelope (6) ----
    min_d = min_bet - hero_sc
    max_d = max_bet - hero_sc
    base = pot + to_call
    raise_legal = max_d > to_call
    base_safe = np.maximum(base, 1.0)
    eff_denom = np.maximum(hero_stack, 1.0)
    # Legal-anchor count only (no brackets) — bit-identical to
    # anchor_grid_np(...).legal.sum(-1) / ANCHOR_COUNT.
    n_legal = (
        n_legal_anchors_np(min_d, max_d, pot, to_call).astype(np.float64)
        / float(ANCHOR_COUNT)
    )
    wl2 = live_mask & raise_legal
    out[wl2, _STK2_OFF + 0] = np.clip((min_d - to_call) / base_safe, 0.0, 1.0)[wl2]
    out[wl2, _STK2_OFF + 1] = np.clip((max_d - to_call) / base_safe, 0.0, 1.0)[wl2]
    out[wl2, _STK2_OFF + 2] = np.clip(min_d / eff_denom, 0.0, 1.0)[wl2]
    out[wl2, _STK2_OFF + 3] = np.clip(max_d / eff_denom, 0.0, 1.0)[wl2]
    out[wl2, _STK2_OFF + 4] = (max_d < to_call + base).astype(np.float64)[wl2]
    out[wl2, _STK2_OFF + 5] = n_legal[wl2]

    # ---- STK-4: per-seat commitment ratio (8), hero-rotated ----
    denom4 = tc_rot + eff_rot
    ratio4 = np.where(
        denom4 > 0.0, tc_rot / np.where(denom4 > 0.0, denom4, 1.0), 0.0
    )
    ratio4 = np.where(folded_rot, 0.0, ratio4)
    out[live_mask, _STK4_OFF : _STK4_OFF + num_seats] = ratio4[live_mask]

    # ---- STK-5: spr-after-action (4) ----
    out[live_mask, _STK5_OFF + 0] = np.log1p(
        (hero_stack - eff_to_call) / np.maximum(pot + eff_to_call, 1.0)
    )[live_mask]
    out[live_mask, _STK5_OFF + 1] = np.log1p((pot + eff_to_call) * inv_bb)[live_mask]
    tot2 = pot + 2.0 * max_d - to_call
    den2 = np.maximum(tot2, 1.0)
    arg2 = np.where(raise_legal, (hero_stack - max_d) / den2, 0.0)
    tot2_safe = np.where(raise_legal, tot2, 0.0)
    wl5 = live_mask & raise_legal
    out[wl5, _STK5_OFF + 2] = np.log1p(arg2)[wl5]
    out[wl5, _STK5_OFF + 3] = np.log1p(tot2_safe * inv_bb)[wl5]

    # ---- STK-6: geometric jam plan (2) ----
    spr_e = hero_stack / pot_denom
    r = 4 - street
    x6 = 1.0 + 2.0 * spr_e
    with np.errstate(divide="ignore", invalid="ignore"):
        btj = np.ceil(np.log(x6) / np.log(3.0))
        gfrac = (np.power(x6, 1.0 / r) - 1.0) / 2.0
    out[live_mask, _STK6_OFF + 0] = np.clip(btj, 0.0, 6.0)[live_mask]
    out[live_mask, _STK6_OFF + 1] = np.clip(gfrac, 0.0, 2.0)[live_mask]

    # ---- STK-7: pot-ceiling implied odds (2) ----
    # Explicit seat-index accumulation to match the serial add order bit-exactly
    # (skipped seats contribute +0.0, transparent).
    ceiling = pot.copy()
    opp_min = np.minimum(effective, hero_stack[:, None])
    for s in range(num_seats):
        contrib = np.where((s != hero_idx) & (~folded[:, s]), opp_min[:, s], 0.0)
        ceiling = ceiling + contrib
    out[live_mask, _STK7_OFF + 0] = np.log1p(ceiling / pot_denom)[live_mask]
    den7 = ceiling + eff_to_call
    price7 = np.where(
        eff_to_call > 0.0, eff_to_call / np.where(den7 > 0.0, den7, 1.0), 0.0
    )
    out[live_mask, _STK7_OFF + 1] = price7[live_mask]

    # ---- STK-8: side-pot eligibility (3) ----
    hero_after = hero_commit + eff_to_call
    sum_now = np.zeros(n, dtype=np.float64)
    sum_after = np.zeros(n, dtype=np.float64)
    dead = np.zeros(n, dtype=np.float64)
    for s in range(num_seats):
        cs = total_commit[:, s]
        sum_now = sum_now + np.minimum(cs, hero_commit)
        sum_after = sum_after + np.minimum(cs, hero_after)
        dead = dead + np.where(folded[:, s], cs, 0.0)
    out[live_mask, _STK8_OFF + 0] = (sum_now / pot_denom)[live_mask]
    out[live_mask, _STK8_OFF + 1] = (sum_after / pot_denom)[live_mask]
    out[live_mask, _STK8_OFF + 2] = (dead / pot_denom)[live_mask]

    # ---- STK-9: call-risk fraction (2) ----
    out[live_mask, _STK9_OFF + 0] = (
        eff_to_call / np.maximum(hero_stack, 1.0)
    )[live_mask]
    out[live_mask, _STK9_OFF + 1] = (
        eff_to_call / np.maximum(hero_commit + hero_stack, 1.0)
    )[live_mask]

    # ---- STK-10: ante-pot bloat (2) ----
    ante_i = int(config.ante)
    pot_at_flop = 0
    for s in range(num_seats):
        pot_at_flop += min(ante_i, int(config.resolved_stacks[s]))
    paf = float(pot_at_flop)
    out[live_mask, _STK10_OFF + 0] = ante_i * inv_bb
    out[live_mask, _STK10_OFF + 1] = np.log1p(
        np.maximum(pot - paf, 0.0) / max(paf, 1.0)
    )[live_mask]

    # ---- STK-11: per-seat price-to-continue (8), hero-rotated ----
    owed = np.minimum(np.maximum(bet_to_call[:, None] - sc_rot, 0.0), eff_rot)
    ratio11 = owed / (pot_denom[:, None] + owed)
    ratio11 = np.where(folded_rot, 0.0, ratio11)
    ratio11[:, 0] = 0.0  # hero slot
    out[live_mask, _STK11_OFF : _STK11_OFF + num_seats] = ratio11[live_mask]


def _encode_board_v3_batch(
    out: np.ndarray,
    *,
    live_mask: np.ndarray,
    hole: np.ndarray,
    board_a: np.ndarray,
    board_b: np.ndarray,
    street: np.ndarray,
    hero_board_v3=None,
    board_draw_v3=None,
) -> None:
    """Batched twin of _encode_board_v3, dims 1061..1139. `hero_board_v3`
    is the (N, 8) engine block (None → BRD-7/BRD-12 stay zero).
    `board_draw_v3` is the (N, 7) engine hot block
    [ds_a, ds_b, u_a, n_a, u_b, n_b, scoop] (None → Python BRD-5/6)."""
    n = hole.shape[0]
    live = live_mask
    ranks13 = np.arange(13)
    wmat = _WINDOW_MATRIX                 # (10, 13) bool
    wmat_i = wmat.astype(np.int64)        # (10, 13)
    # Prefer engine hot path when present (training pack always supplies it).
    _bd = None if board_draw_v3 is None else np.asarray(board_draw_v3)

    # ---- Global visibility (hole + both boards), (N, 13, 4) 0/1 ----
    visible = np.zeros((n, 13, 4), dtype=np.int64)
    board_all = np.zeros((n, 13, 4), dtype=bool)   # both boards only (BRD-10 scan)
    hole_pres = np.zeros((n, 13, 4), dtype=bool)
    for src in (hole, board_a, board_b):
        sv = src < 52
        if sv.any():
            ei, si = np.nonzero(sv)
            cards = src[ei, si].astype(np.int64)
            visible[ei, cards >> 2, cards & 3] = 1
    for src in (board_a, board_b):
        sv = src < 52
        if sv.any():
            ei, si = np.nonzero(sv)
            cards = src[ei, si].astype(np.int64)
            board_all[ei, cards >> 2, cards & 3] = True
    sv = hole < 52
    if sv.any():
        ei, si = np.nonzero(sv)
        cards = hole[ei, si].astype(np.int64)
        hole_pres[ei, cards >> 2, cards & 3] = True

    vct = visible.sum(axis=2)                    # (N, 13) global copies visible per rank
    vps = visible.sum(axis=1)                    # (N, 4) global cards visible per suit
    unseen_deck = 52 - visible.sum(axis=(1, 2))  # (N,)
    unseen_r = 4 - vct                           # (N, 13) unseen copies per rank
    unseen_s = 13 - vps                          # (N, 4) unseen copies per suit
    unseen_ce = visible == 0                     # (N, 13, 4) not-visible-anywhere

    # ---- Hero-side aggregates ----
    hv = hole < 52
    hrc = np.zeros((n, 13), dtype=np.int64)      # hero rank counts
    hsc = np.zeros((n, 4), dtype=np.int64)       # hero suit counts
    hrm = np.zeros((n, 13), dtype=bool)          # hero rank presence
    hmax = -np.ones((n, 4), dtype=np.int64)      # hero max rank per suit (-1 = none)
    for k in range(hole.shape[1]):
        vk = hv[:, k]
        if vk.any():
            idx = np.nonzero(vk)[0]
            cards = hole[vk, k].astype(np.int64)
            rk = cards >> 2
            sk = cards & 3
            np.add.at(hrc, (idx, rk), 1)
            np.add.at(hsc, (idx, sk), 1)
            hrm[idx, rk] = True
            np.maximum.at(hmax, (idx, sk), rk)

    denom = unseen_deck.astype(np.float64)       # (N,) always > 0 for live rows
    not_river = street != 3
    is_flop = street == 1

    # ---- BRD-7 / BRD-12: engine-emitted hero/board dims ----
    # Same normalizations as the serial helper: BRD-7 raw counts, BRD-12
    # outs / actual unseen-deck size, combo redundancy / 10.
    if hero_board_v3 is not None:
        hb = hero_board_v3.astype(np.float64)
        out[live, _BRD7_OFF + 0] = hb[live, 0]
        out[live, _BRD7_OFF + 1] = hb[live, 1]
        out[live, _BRD12_OFF + 0] = (hb[:, 2] / denom)[live]
        out[live, _BRD12_OFF + 1] = (hb[:, 3] / denom)[live]
        out[live, _BRD12_OFF + 2] = (hb[:, 4] / 10.0)[live]
        out[live, _BRD12_OFF + 3] = (hb[:, 5] / 10.0)[live]

    def _board_block(board, bi):
        bv = board < 52
        brc = np.zeros((n, 13), dtype=np.int64)  # board rank counts
        bsc = np.zeros((n, 4), dtype=np.int64)   # board suit counts
        brm = np.zeros((n, 13), dtype=bool)      # board rank presence
        brs = np.zeros((n, 13, 4), dtype=bool)   # board (rank, suit) presence
        for k in range(5):
            vk = bv[:, k]
            if vk.any():
                idx = np.nonzero(vk)[0]
                cards = board[vk, k].astype(np.int64)
                rk = cards >> 2
                sk = cards & 3
                np.add.at(brc, (idx, rk), 1)
                np.add.at(bsc, (idx, sk), 1)
                brm[idx, rk] = True
                brs[idx, rk, sk] = True
        win_board_cnt = brm.astype(np.int64) @ wmat_i.T   # (N, 10) distinct board ranks per window

        # ---- BRD-1: rank ladder (5) ----
        o1 = _BRD1_OFF + bi * 5
        sort_key = np.where(bv, board.astype(np.int64) >> 2, -1)
        sorted_desc = -np.sort(-sort_key, axis=1)         # (N, 5) descending; -1 in tail
        valid_slot = sorted_desc >= 0
        ladder = np.where(
            valid_slot, (sorted_desc.astype(np.float64) + 1.0) / 13.0, 0.0
        )
        out[live, o1 : o1 + 5] = ladder[live]

        # ---- BRD-2: suit census (6) ----
        o2 = _BRD2_OFF + bi * 6
        out[live, o2 : o2 + 4] = (bsc[live] == 2).astype(np.float32)
        out[live, o2 + 4] = (bsc == 4).any(axis=1)[live].astype(np.float32)
        out[live, o2 + 5] = (bsc == 5).any(axis=1)[live].astype(np.float32)

        # ---- BRD-4: arrival volatility census (3), river-zeroed ----
        o4 = _BRD4_OFF + bi * 3
        pair_outs = (brm * unseen_r).sum(axis=1)
        adv_suit = (bsc == 2) | (bsc == 3)
        flush_adv = (adv_suit * unseen_s).sum(axis=1)
        window_has2 = win_board_cnt == 2                  # (N, 10)
        rank_in2win = (window_has2[:, :, None] & wmat[None, :, :]).any(axis=1)  # (N, 13)
        straight_adv = ((rank_in2win & ~brm) * unseen_r).sum(axis=1)
        b4 = np.stack([pair_outs, flush_adv, straight_adv], axis=1).astype(np.float64)
        b4 = np.where(not_river[:, None], b4 / denom[:, None], 0.0)
        out[live, o4 : o4 + 3] = b4[live]

        # ---- BRD-5: hero vulnerability outs (3), river-zeroed ----
        # Integer bitmask path: (N,) u16 board/hole masks + rank/window
        # loops over scalar ints. Bit-exact with the serial frozenset path
        # (and ~10× faster than the prior (N,13) bool-matrix formulation).
        o5 = _BRD5_OFF + bi * 3
        danger_flush = (((bsc == 2) & (hsc < 2)) * unseen_s).sum(axis=1)
        danger_pair = ((brm & (hrc == 0)) * unseen_r).sum(axis=1)
        if _bd is not None:
            # Engine hot path: raw danger_straight counts at cols 0/1.
            danger_straight = _bd[:, bi].astype(np.int64)
        else:
            board_bits = (brm.astype(np.uint16) * _RANK_WEIGHTS_13).sum(axis=1).astype(
                np.uint16
            )
            hole_bits = (hrm.astype(np.uint16) * _RANK_WEIGHTS_13).sum(axis=1).astype(
                np.uint16
            )
            danger_straight = np.zeros(n, dtype=np.int64)
            for rc in range(13):
                bit = np.uint16(1 << rc)
                new_board = board_bits | bit
                field = np.zeros(n, dtype=bool)
                makes_any = np.zeros(n, dtype=bool)
                for W in _WINDOW_BITS_13:
                    cov = np.bitwise_count((new_board & W).astype(np.uint16))
                    field |= cov >= 3
                    L = (W & ~new_board.astype(np.uint32)).astype(np.uint16) & np.uint16(
                        _MASK13
                    )
                    miss = (
                        L & ~hole_bits.astype(np.uint32)
                    ).astype(np.uint16) & np.uint16(_MASK13)
                    nL = np.bitwise_count(L)
                    nH = np.bitwise_count((W & hole_bits).astype(np.uint16))
                    makes_any |= (miss == 0) & (nH >= 2) & (nL <= 2)
                danger = field & ~makes_any
                danger_straight += np.where(danger, unseen_r[:, rc], 0)
        b5 = np.stack(
            [danger_flush, danger_pair, danger_straight], axis=1
        ).astype(np.float64)
        b5 = np.where(not_river[:, None], b5 / denom[:, None], 0.0)
        out[live, o5 : o5 + 3] = b5[live]

        # ---- BRD-6: straight out union (2), river-zeroed ----
        o6 = _BRD6_OFF + bi * 2
        if _bd is not None:
            # layout: [ds_a, ds_b, u_a, n_a, u_b, n_b, scoop]
            union_outs = _bd[:, 2 + bi * 2].astype(np.int64)
            nut_outs = _bd[:, 3 + bi * 2].astype(np.int64)
        else:
            board_bits = (brm.astype(np.uint16) * _RANK_WEIGHTS_13).sum(axis=1).astype(
                np.uint16
            )
            hole_bits = (hrm.astype(np.uint16) * _RANK_WEIGHTS_13).sum(axis=1).astype(
                np.uint16
            )
            union_mask = np.zeros(n, dtype=np.uint16)
            for W in _WINDOW_BITS_13:
                L = (W & ~board_bits.astype(np.uint32)).astype(np.uint16) & np.uint16(
                    _MASK13
                )
                M = (L & ~hole_bits.astype(np.uint32)).astype(np.uint16) & np.uint16(
                    _MASK13
                )
                nL = np.bitwise_count(L)
                nH = np.bitwise_count((W & hole_bits).astype(np.uint16))
                nM = np.bitwise_count(M)
                makes = (nM == 0) & (nH >= 2) & (nL <= 2)
                gate = (~makes) & (nH >= 2)
                cond1 = (nL == 3) & (nM == 0) & gate
                cond2 = (nM == 1) & (nL >= 1) & (nL <= 3) & gate
                union_mask = union_mask | np.where(cond1, L, np.uint16(0))
                union_mask = union_mask | np.where(cond2, M, np.uint16(0))
            union_outs = np.zeros(n, dtype=np.int64)
            for rc in range(13):
                in_u = (union_mask & np.uint16(1 << rc)) != 0
                union_outs += np.where(in_u, unseen_r[:, rc], 0)
            nut_outs = np.zeros(n, dtype=np.int64)
            for rc in range(13):
                in_u = (union_mask & np.uint16(1 << rc)) != 0
                if not in_u.any():
                    continue
                new_board = board_bits | np.uint16(1 << rc)
                h_max = np.full(n, -1, dtype=np.int64)
                for wi, W in enumerate(_WINDOW_BITS_13):
                    L = (
                        W & ~new_board.astype(np.uint32)
                    ).astype(np.uint16) & np.uint16(_MASK13)
                    miss = (
                        L & ~hole_bits.astype(np.uint32)
                    ).astype(np.uint16) & np.uint16(_MASK13)
                    nL = np.bitwise_count(L)
                    nH = np.bitwise_count((W & hole_bits).astype(np.uint16))
                    makes = (miss == 0) & (nH >= 2) & (nL <= 2)
                    h_max = np.where(makes & (wi > h_max), wi, h_max)
                any_make = h_max >= 0
                nut_dist = np.zeros(n, dtype=np.int64)
                for wi, W in enumerate(_WINDOW_BITS_13):
                    poss = np.bitwise_count((W & new_board).astype(np.uint16)) >= 3
                    nut_dist += (poss & (wi > h_max)).astype(np.int64)
                is_nut = any_make & (nut_dist == 0) & in_u
                nut_outs += np.where(is_nut, unseen_r[:, rc], 0)
        b6 = np.stack([union_outs, nut_outs], axis=1).astype(np.float64)
        b6 = np.where(not_river[:, None], b6, 0.0)
        out[live, o6 : o6 + 2] = b6[live]

        # ---- BRD-8: fd rank quality (2) ----
        o8 = _BRD8_OFF + bi * 2
        qual = (hsc >= 2) & (bsc == 2)                    # (N, 4)
        any_qual = qual.any(axis=1)
        hmax_masked = np.where(qual, hmax, -1)
        draw_suit = np.argmax(hmax_masked, axis=1)        # (N,) first-max = lowest suit idx
        h1 = hmax[np.arange(n), draw_suit]                # (N,)
        b8_0 = np.where(any_qual, (h1.astype(np.float64) + 1.0) / 13.0, 0.0)
        unseen_ds = unseen_ce[np.arange(n), :, draw_suit]  # (N, 13)
        above = ranks13[None, :] > h1[:, None]
        b8_1 = np.where(any_qual, (above & unseen_ds).sum(axis=1), 0)
        out[live, o8 + 0] = b8_0[live]
        out[live, o8 + 1] = b8_1[live].astype(np.float32)

        # ---- BRD-9: backdoor draw census (2), flop-only ----
        o9 = _BRD9_OFF + bi * 2
        bdfd = ((hsc >= 2) & (bsc == 1)).sum(axis=1)
        bdstr = np.zeros(n, dtype=np.int64)
        for w in range(10):
            Wm = wmat[w]
            inW = Wm[None, :]
            nH = (inW & hrm).sum(axis=1)
            nL = (inW & ~brm).sum(axis=1)
            missing_both = (inW & ~brm & ~hrm).sum(axis=1)
            bdstr += ((missing_both == 2) & (nH >= 2) & (nL <= 4)).astype(np.int64)
        b9 = np.stack([bdfd, bdstr], axis=1)
        b9 = np.where(is_flop[:, None], b9, 0)
        out[live, o9 : o9 + 2] = b9[live].astype(np.float32)

        # ---- BRD-10: future nut-flush blocker (1), river-zeroed ----
        o10 = _BRD10_OFF + bi * 1
        cnt = np.zeros(n, dtype=np.int64)
        for s in range(4):
            notb_desc = (~board_all[:, :, s])[:, ::-1]    # index 0 = rank 12
            key_rank = 12 - np.argmax(notb_desc, axis=1)
            has_key = hole_pres[np.arange(n), key_rank, s]
            cnt += ((bsc[:, s] == 2) & has_key).astype(np.int64)
        b10 = np.where(not_river, cnt, 0)
        out[live, o10] = b10[live].astype(np.float32)

        # ---- BRD-11: turn/river card identity (5 per slot) ----
        # (rank+1)/13 as ADD-then-DIVIDE to match the scalar path bit-exactly.
        o11t = _BRD11_OFF + bi * 10 + 0
        o11r = _BRD11_OFF + bi * 10 + 5
        for base_off, col in ((o11t, 3), (o11r, 4)):
            vcol = (board[:, col] < 52) & live
            rows = np.nonzero(vcol)[0]
            if rows.size:
                cards = board[rows, col].astype(np.int64)
                out[rows, base_off] = ((cards >> 2) + 1).astype(np.float64) / 13.0
                out[rows, base_off + 1 + (cards & 3)] = 1.0

        # ---- BRD-13: board nut ceiling class (2) ----
        o13 = _BRD13_OFF + bi * 2
        win_suit = np.tensordot(
            brs.astype(np.int64), wmat_i.T, axes=([1], [0])
        )  # (N, 4, 10)
        sf_possible = (win_suit >= 3).any(axis=(1, 2))
        paired = (brc >= 2).any(axis=1)
        flush_poss = (bsc >= 3).any(axis=1)
        straight_poss = (win_board_cnt >= 3).any(axis=1)
        has_board = bv.any(axis=1)
        ceil = np.full(n, 3, dtype=np.int64)
        ceil = np.where(straight_poss, 4, ceil)
        ceil = np.where(flush_poss, 5, ceil)
        ceil = np.where(paired, 7, ceil)
        ceil = np.where(sf_possible, 8, ceil)
        out[live, o13 + 0] = np.where(has_board, sf_possible, False)[live].astype(
            np.float32
        )
        out[live, o13 + 1] = np.where(has_board, ceil.astype(np.float64) / 8.0, 0.0)[
            live
        ]

    _board_block(board_a, 0)
    _board_block(board_b, 1)


def _encode_dual_v3_batch(
    out: np.ndarray,
    *,
    live_mask: np.ndarray,
    hero_idx: np.ndarray,
    hole: np.ndarray,
    board_a: np.ndarray,
    board_b: np.ndarray,
    per_board_outcome,
    pot: np.ndarray,
    to_call: np.ndarray,
    effective: np.ndarray,
    config: GameConfig,
    hero_board_v3=None,
    share_bounds=None,
    board_draw_v3=None,
) -> None:
    """Batched twin of _encode_dual_v3, dims 1139..1171. `hero_board_v3`
    (N, 8) carries the best-holding masks at cols 6/7 (DUAL-2);
    `share_bounds` (N, 2) is the fused pass's [g_min, g_max] (DUAL-4);
    `board_draw_v3` (N, 7) supplies DUAL-5 scoop raw count at col 6."""
    n = hole.shape[0]
    rows = np.arange(n)

    # ---- DUAL-1 (1139..1141): split-adjusted price ladder ----
    # eff_to_call = min(to_call, hero effective stack); f64 throughout,
    # cast on assignment (parity with the scalar path).
    hero_stack = effective[rows, hero_idx]
    eff_to_call = np.minimum(to_call, hero_stack)
    pos = eff_to_call > 0.0
    half = np.zeros(n, dtype=np.float64)
    quarter = np.zeros(n, dtype=np.float64)
    half[pos] = eff_to_call[pos] / (0.5 * pot[pos] + eff_to_call[pos])
    quarter[pos] = eff_to_call[pos] / (0.25 * pot[pos] + eff_to_call[pos])
    out[live_mask, _DUAL1_OFF + 0] = half[live_mask]
    out[live_mask, _DUAL1_OFF + 1] = quarter[live_mask]

    # ---- DUAL-3 (1151..1157): nut-lock / freeroll flags ----
    active = None
    if per_board_outcome is not None:
        pbo = np.asarray(per_board_outcome, dtype=np.float64)
        ahead_a = pbo[:, 0]
        tie_a = pbo[:, 1]
        behind_a = pbo[:, 2]
        tie_b = pbo[:, 4]
        behind_b = pbo[:, 5]
        active = (ahead_a + tie_a + behind_a) > 0.5
        nut_or_chop_a = active & (behind_a == 0.0)
        nut_or_chop_b = active & (behind_b == 0.0)
        pure_nut_a = nut_or_chop_a & (tie_a == 0.0)
        pure_nut_b = nut_or_chop_b & (tie_b == 0.0)
        locked_both = nut_or_chop_a & nut_or_chop_b
        locked_one = nut_or_chop_a | nut_or_chop_b
        dual3 = np.stack(
            [pure_nut_a, nut_or_chop_a, pure_nut_b, nut_or_chop_b, locked_both, locked_one],
            axis=1,
        ).astype(np.float64)
        out[live_mask, _DUAL3_OFF : _DUAL3_OFF + 6] = dual3[live_mask]

    # ---- DUAL-4 (1157..1162): guaranteed pot share (k=2 g_min/g_max) ----
    # Mirrors the serial helper: gated on the outcome block being active;
    # [3]/[4] use the g_max==g_min sentinel (no contested slice).
    if share_bounds is not None and active is not None:
        shb = np.asarray(share_bounds, dtype=np.float64)
        g_min = shb[:, 0]
        g_max = shb[:, 1]
        wl4 = live_mask & active
        out[wl4, _DUAL4_OFF + 0] = g_min[wl4]
        out[wl4, _DUAL4_OFF + 1] = g_max[wl4]
        out[wl4, _DUAL4_OFF + 2] = (g_min >= 0.5).astype(np.float64)[wl4]
        rng = g_max - g_min
        has_rng = rng > 0.0
        denom4 = pot * rng + eff_to_call
        price = np.where(
            has_rng & (eff_to_call > 0.0),
            eff_to_call / np.where(denom4 > 0.0, denom4, 1.0),
            0.0,
        )
        out[wl4, _DUAL4_OFF + 3] = price[wl4]
        # max(pot, 1): mirrors the serial guard — ante=0 (pot 0) states
        # would otherwise emit inf here while the scalar path crashes.
        pot_safe4 = np.maximum(pot, 1.0)
        ratio = np.where(
            has_rng,
            hero_stack / np.where(has_rng, pot_safe4 * rng, 1.0),
            0.0,
        )
        out[wl4, _DUAL4_OFF + 4] = np.log1p(ratio)[wl4]

    # ---- DUAL-2 (1141..1151): best-holding card usage / coverage ----
    if hero_board_v3 is not None:
        bits = np.arange(5)
        mask_a = hero_board_v3[:, 6].astype(np.int64)
        mask_b = hero_board_v3[:, 7].astype(np.int64)
        d2a = ((mask_a[:, None] >> bits[None, :]) & 1).astype(np.float64)
        d2b = ((mask_b[:, None] >> bits[None, :]) & 1).astype(np.float64)
        out[live_mask, _DUAL2_OFF : _DUAL2_OFF + 5] = d2a[live_mask]
        out[live_mask, _DUAL2_OFF + 5 : _DUAL2_OFF + 10] = d2b[live_mask]

    # ---- DUAL-5 (1162..1171): villain cross-board coverage (board-only) ----
    ba_valid = board_a < 52
    bb_valid = board_b < 52
    ba_suit = np.zeros((n, 4), dtype=np.int64)
    bb_suit = np.zeros((n, 4), dtype=np.int64)
    for arr, valid, dest in (
        (board_a, ba_valid, ba_suit),
        (board_b, bb_valid, bb_suit),
    ):
        suits = (arr & 3).astype(np.int64)
        for k in range(arr.shape[1]):
            vk = valid[:, k]
            if vk.any():
                np.add.at(dest, (np.nonzero(vk)[0], suits[vk, k]), 1)
    suit_ge2_both = (ba_suit >= 2) & (bb_suit >= 2)
    suit_ge3_both = (ba_suit >= 3) & (bb_suit >= 3)

    if board_draw_v3 is not None:
        scoop_pairs = np.asarray(board_draw_v3)[:, 6].astype(np.float64)
    else:
        ba_rank_mask = np.zeros((n, 13), dtype=bool)
        bb_rank_mask = np.zeros((n, 13), dtype=bool)
        for arr, valid, dest in (
            (board_a, ba_valid, ba_rank_mask),
            (board_b, bb_valid, bb_rank_mask),
        ):
            ranks = (arr >> 2).astype(np.int64)
            for k in range(arr.shape[1]):
                vk = valid[:, k]
                if vk.any():
                    dest[np.nonzero(vk)[0], ranks[vk, k]] = True
        ba_bits = (ba_rank_mask.astype(np.uint16) * _RANK_WEIGHTS_13).sum(
            axis=1
        ).astype(np.uint16)
        bb_bits = (bb_rank_mask.astype(np.uint16) * _RANK_WEIGHTS_13).sum(
            axis=1
        ).astype(np.uint16)
        made_a = np.zeros((n, _PAIR_BITS_13.shape[0]), dtype=bool)
        made_b = np.zeros((n, _PAIR_BITS_13.shape[0]), dtype=bool)
        for w_idx in range(_WINDOW_BITS_13.shape[0]):
            W = _WINDOW_BITS_13[w_idx]
            pair_in_W = (_PAIR_BITS_13 & W) == _PAIR_BITS_13
            if not pair_in_W.any():
                continue
            ba_W = (ba_bits & W).astype(np.uint16)
            bb_W = (bb_bits & W).astype(np.uint16)
            cov_a = np.bitwise_count(ba_W[:, None] | _PAIR_BITS_13[None, :])
            cov_b = np.bitwise_count(bb_W[:, None] | _PAIR_BITS_13[None, :])
            made_a |= (cov_a >= 5) & pair_in_W[None, :]
            made_b |= (cov_b >= 5) & pair_in_W[None, :]
        scoop_pairs = (made_a & made_b).sum(axis=1).astype(np.float64)

    dual5 = np.zeros((n, 9), dtype=np.float64)
    dual5[:, 0:4] = suit_ge2_both.astype(np.float64)
    dual5[:, 4:8] = suit_ge3_both.astype(np.float64)
    dual5[:, 8] = scoop_pairs / 78.0
    out[live_mask, _DUAL5_OFF : _DUAL5_OFF + 9] = dual5[live_mask]
