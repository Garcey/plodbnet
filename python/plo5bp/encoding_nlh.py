"""NLH single-board observation encoding.

Produces a fixed-length float32 vector from the dict emitted by
`PyGameState.observation_dict()` for the `nlh_single` variant
(augmented in `env._pack_obs` with the Rust-computed hand category).
A separate module from `encoding.py` because the layout is a different
game's: single board, any-combo hand features, a live preflop street,
and deep-stack scaling — sharing offsets with the PLO layout would be
a false economy. Rule-independent helpers (`_pair_features`,
`_gate_from_action`, the straight windows) are imported from
`encoding.py`; hand features whose *rule* differs (any 5 of hole+board,
hero-involved = uses ≥1 hole card, at most 2 hole cards) are
reimplemented here with the same per-dim semantics.

Layout (995 dims total):
  0..52     hero hole multi-hot (52) — 2 bits set
  52..104   board multi-hot (52)
  104..108  street one-hot (Preflop/Flop/Turn/River) — preflop is LIVE
  108..116  active mask, hero-rotated, padded to 8
  116..124  all-in mask, hero-rotated, padded to 8
  124..132  stacks / bb (effective, dead-capped like PLO), hero-rotated
  132..136  scalars: pot, bet_to_call, min_bet, max_bet — all / bb
  136..144  relative-position one-hot of actor
  144..864  history: last 40 actions oldest-first, each slot 18 dims
            (seat-one-hot 8 + gate one-hot 4 {Fold,Check,Call,Raise}
            + street one-hot 4 + chips/bb 1 + log1p(chips/pot-before) 1).
            Depth 40 (PLO uses 32): the preflop round adds ~6-10 actions
            per hand and depth can't grow later without discarding
            checkpoints. The pot-fraction dim is log1p-scaled instead of
            clip[0,2] — NL overbets and jams routinely exceed 2x pot and
            a clip cannot rank a 3x jam vs a 20x jam. Blinds and antes
            are forced posts, never history entries (visible through the
            street-commit block instead).
  864..872  SPR per seat: log1p(eff_stack / max(pot, 1)), hero-rotated.
            log1p, not clip[0,4]: preflop SPR at 100-250bb is 20-55 and
            a clip saturates the feature exactly where NLH needs it.
  872..873  pot odds (to_call / (pot + to_call), 0 if no bet to face)
  873..882  hero hand category one-hot (9, any-combo rule; env passes
            the engine's `hero_category(actor, 0)`)
  882..884  hero draw flags (flush, straight) — flush requires ≥1 hole
            card of the suit (4 total); straight is the coarse
            4-of-a-5-window union rule
  884..889  pair-with-board count (5 slots, board sorted rank desc;
            values 0..2 with two hole cards)
  889..893  board pair structure (paired, double_paired, tripled, quadded)
  893..906  hero rank histogram (13; a pocket pair reads as a 2)
  906..944  straight/flush/SF block (38, same sub-layout as the PLO
            board-A block, any-combo rules — see _sf_features_nlh)
  944..952  hero-rotated seat-exists mask
  952..960  per-seat hand-total commit, hero-rotated, /bb (raw)
  960..968  per-seat street commit, hero-rotated, /bb (raw; this is
            where the posted blinds are visible preflop)
  968..976  last-aggressor one-hot, hero-relative; all-zero when no
            raise yet (blinds are not aggression)
  976..984  hero distance to button one-hot
  984..987  opp-outcome: fraction of unseen 2-card combos currently
            [ahead of, tied with, behind] hero (exhaustive, Rust
            `nlh_opp_outcome_fractions`); zeros preflop
  987..988  bet-faced: log1p(to_call / max(pot - to_call, 1)); log1p for
            the same reason as the history frac — jams reach 20x+ pot
  988..993  hole-class: pocket_pair, suited, gap/12, hi_rank/12,
            lo_rank/12 — compact preflop identity (the 169-class one-hot
            alternative is deferred; category/draw blocks are silent
            preflop, this block is not)
  993..995  hero-is-SB, hero-is-BB (from the engine's stored blind
            seats — NOT re-derived, the walk skips sitting-out seats)
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from plo5bp.config import GameConfig
from plo5bp.encoding import (
    _GATE_CALL,
    _GATE_CHECK,
    _GATE_FOLD,
    _GATE_RAISE,
    _STRAIGHT_WINDOWS,
    _gate_from_action,
    _pair_features,
    _pair_features_batch,
)
from plo5bp.actions import CHECK_CALL, FOLD

OBS_DIM_NLH: int = 995

_HOLE_OFF = 0
_BOARD_OFF = 52
_STREET_OFF = 104
_ACTIVE_OFF = 108
_ALLIN_OFF = 116
_STACKS_OFF = 124
_SCALARS_OFF = 132
_REL_POS_OFF = 136
_HISTORY_OFF = 144
_HISTORY_DEPTH = 40
_HISTORY_SLOT_DIM = 18
_MAX_SEATS = 8
_NUM_STREET_ONEHOT = 4
_NUM_CATEGORIES = 9

_HISTORY_SEAT_OFF_REL = 0
_HISTORY_GATE_OFF_REL = 8
_HISTORY_STREET_OFF_REL = 12
_HISTORY_CHIPS_OFF_REL = 16
_HISTORY_FRAC_OFF_REL = 17

_SPR_OFF = 864
_POT_ODDS_OFF = 872
_CAT_OFF = 873
_DRAW_OFF = 882
_PAIR_COUNT_OFF = 884
_BOARD_STRUCT_OFF = 889
_HERO_RANK_HIST_OFF = 893
_SF_OFF = 906  # 38 dims, sub-layout identical to encoding.py's board-A block
_SEAT_EXISTS_OFF = 944
_TOTAL_COMMIT_OFF = 952
_STREET_COMMIT_OFF = 960
_LAST_AGGRESSOR_OFF = 968
_HERO_BTN_DIST_OFF = 976
_OPP_OUTCOME_OFF = 984  # 3 dims: [opp_ahead, tied, opp_behind]
_BET_PCT_POT_OFF = 987
_HOLE_CLASS_OFF = 988  # 5 dims
_BLIND_FLAGS_OFF = 993  # 2 dims

assert _BLIND_FLAGS_OFF + 2 == OBS_DIM_NLH
assert _SPR_OFF == _HISTORY_OFF + _HISTORY_DEPTH * _HISTORY_SLOT_DIM


def _draw_flags_nlh(hole_idx: list[int], board_idx: list[int]) -> tuple[float, float]:
    """(flush_draw, straight_draw) under NLH rules.

    Flush draw: some suit has ≥1 hole card and exactly 4 cards total
    across hole+board (one more completes a hero-involved 5-flush).
    Board-only 4-flushes are texture, not a hero draw — they live in
    the SF block's `flush_possible` dims.

    Straight draw: ≥4 ranks of some 5-rank straight window covered by
    hole ∪ board (coarse union rule, mirrors the PLO flag's altitude —
    made straights also read 1; the SF block carries the exact outs).
    """
    if not board_idx:
        return 0.0, 0.0
    hole_suits = [0, 0, 0, 0]
    board_suits = [0, 0, 0, 0]
    for c in hole_idx:
        hole_suits[c % 4] += 1
    for c in board_idx:
        board_suits[c % 4] += 1
    flush = 0.0
    for s in range(4):
        if hole_suits[s] >= 1 and hole_suits[s] + board_suits[s] == 4:
            flush = 1.0
            break

    ranks = {c // 4 for c in hole_idx} | {c // 4 for c in board_idx}
    straight = 0.0
    for W in _STRAIGHT_WINDOWS:
        if len(W & ranks) >= 4:
            straight = 1.0
            break
    return flush, straight


def _sf_features_nlh(
    hole_idx: list[int], board_idx: list[int], visible_count: np.ndarray
) -> np.ndarray:
    """Straight / flush / straight-flush block under NLH rules. (38,) f32,
    sub-layout identical to encoding.py's `_straight_flush_features`:

       0       flush_nut_distance
       1       straight_nut_distance
       2..12   straight_outs_per_window (slot 0=wheel, 9=broadway)
       12..22  straight_possible_per_window (board-only)
       22..26  flush_possible_per_suit (board-only)
       26..30  flush_draw_outs[s]
       30..34  nut_flush_draw_outs[s]
       34..38  straight_flush_draw_outs[s]

    Rule changes vs PLO: hero may use 0, 1, or 2 hole cards (never
    more), so "makes window W" is `(W − board) ⊆ hero ∧ |W − board| ≤ 2`
    with no ≥2-hole requirement; a made flush is any suit with
    hole+board ≥ 5 and ≥1 hole card (or a 5-flush board); draws are
    hero-involved (≥1 hole card of the suit, 4 total).
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

    vct = visible_count.sum(axis=1)  # (13,) global visibility per rank

    hole_max_per_suit = [-1, -1, -1, -1]
    for c in hole_idx:
        s = c % 4
        r = c // 4
        if r > hole_max_per_suit[s]:
            hole_max_per_suit[s] = r

    hole_ranks_per_suit: list[set[int]] = [set(), set(), set(), set()]
    board_ranks_per_suit: list[set[int]] = [set(), set(), set(), set()]
    for c in hole_idx:
        hole_ranks_per_suit[c % 4].add(c // 4)
    for c in board_idx:
        board_ranks_per_suit[c % 4].add(c // 4)

    # ---- Per-window straight features (any-combo, ≤2 from hole) ----
    makes_window = [False] * 10
    straight_outs = [0] * 10
    straight_possible = [0] * 10

    for w_i, W in enumerate(_STRAIGHT_WINDOWS):
        B_W = W & board_ranks_set
        H_W = W & hole_ranks_set
        L = W - B_W  # ranks the board lacks
        nL = len(L)
        straight_possible[w_i] = 1 if len(B_W) >= 3 else 0
        # NLH makes: the hole supplies every missing rank, and at most 2
        # of them (only 2 hole cards). nL == 0 is the board straight.
        makes = (L <= H_W) and (nL <= 2)
        makes_window[w_i] = makes
        if makes:
            continue
        # A board hit of rank r completes W iff the remaining missing
        # ranks are all in the hole and number ≤ 2.
        cands = {
            r for r in L
            if (L - {r}) <= H_W and len(L - {r}) <= 2
        }
        straight_outs[w_i] = sum(4 - int(vct[r]) for r in cands)

    if any(makes_window):
        h_max = max(i for i, m in enumerate(makes_window) if m)
        straight_nut_distance = sum(
            straight_possible[w] for w in range(h_max + 1, 10)
        )
    else:
        straight_nut_distance = 0

    # ---- Flush features ----
    flush_possible = [1.0 if board_suit_counts[s] >= 3 else 0.0 for s in range(4)]

    flush_nut_distance = 0
    for s in range(4):
        total_s = hole_suit_counts[s] + board_suit_counts[s]
        if total_s < 5:
            continue
        if hole_suit_counts[s] >= 1:
            h1 = hole_max_per_suit[s]
        else:
            # Board flush ("playing the board"): any unaccounted card of
            # the suit above the board's 5th-highest beats it.
            h1 = min(sorted(board_ranks_per_suit[s], reverse=True)[:5])
        for r in range(h1 + 1, 13):
            if visible_count[r, s] == 0:
                flush_nut_distance += 1
        break

    flush_draw_outs = [0] * 4
    nut_flush_draw_outs = [0] * 4
    sf_draw_outs = [0] * 4
    for s in range(4):
        if not (hole_suit_counts[s] >= 1
                and hole_suit_counts[s] + board_suit_counts[s] == 4):
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

        # SF outs: suited board hits (r, s) after which some window W is
        # fully covered by suited board + suited hole with ≤2 from hole.
        H_s = hole_ranks_per_suit[s]
        B_s = board_ranks_per_suit[s]
        sf_cands: set[int] = set()
        for W in _STRAIGHT_WINDOWS:
            L_s = W - B_s
            already = (L_s <= H_s) and (len(L_s) <= 2)
            if already:
                continue
            for r in L_s:
                rest = L_s - {r}
                if rest <= H_s and len(rest) <= 2:
                    sf_cands.add(r)
        sf_draw_outs[s] = sum(
            1 for r in sf_cands if visible_count[r, s] == 0
        )

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


def encode_observation_nlh(
    obs: Mapping[str, Any], config: GameConfig
) -> np.ndarray:
    """Encode a single NLH observation dict into a (995,) float32 array."""
    out = np.zeros(OBS_DIM_NLH, dtype=np.float32)
    num_seats = config.num_seats
    hero = obs["actor"]
    if hero is None:
        return out

    for idx in obs["hero_hole"]:
        out[_HOLE_OFF + int(idx)] = 1.0
    for idx in obs["board_a"]:
        out[_BOARD_OFF + int(idx)] = 1.0

    street_idx = int(obs["street"])
    if 0 <= street_idx < _NUM_STREET_ONEHOT:
        out[_STREET_OFF + street_idx] = 1.0

    folded = obs["folded"]
    all_in = obs["all_in"]
    stacks = obs["stacks"]
    eff_cap = obs["eff_stack_cap"]
    starting = config.resolved_stacks
    inv_bb = 1.0 / float(config.bb)
    # Effective remaining (dead chips above max-other-reachable removed),
    # same convention and rationale as the PLO encoder.
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

    out[_REL_POS_OFF + 0] = 1.0  # actor is hero by construction

    history = obs["history"]
    if len(history) > _HISTORY_DEPTH:
        history = history[-_HISTORY_DEPTH:]
    pot_now_chips = int(obs["pot"])
    pot_before = [0] * len(history)
    suffix = 0
    for j in range(len(history) - 1, -1, -1):
        suffix += int(history[j][2])
        pot_before[j] = pot_now_chips - suffix
    for slot, (seat, action, chips, h_street) in enumerate(history):
        base = _HISTORY_OFF + slot * _HISTORY_SLOT_DIM
        rel_seat = (seat - hero) % num_seats
        out[base + _HISTORY_SEAT_OFF_REL + rel_seat] = 1.0
        gate = _gate_from_action(int(action), int(chips))
        out[base + _HISTORY_GATE_OFF_REL + gate] = 1.0
        s_idx = int(h_street)
        if 0 <= s_idx < _NUM_STREET_ONEHOT:
            out[base + _HISTORY_STREET_OFF_REL + s_idx] = 1.0
        out[base + _HISTORY_CHIPS_OFF_REL] = float(chips) * inv_bb
        frac = float(chips) / float(max(pot_before[slot], 1))
        out[base + _HISTORY_FRAC_OFF_REL] = float(np.log1p(max(frac, 0.0)))

    pot_safe = max(pot, 1.0)
    for k in range(num_seats):
        seat = (hero + k) % num_seats
        spr = max(eff_per_seat[seat] / pot_safe, 0.0)
        out[_SPR_OFF + k] = float(np.log1p(spr))

    street_commit = obs.get("street_commit", [0] * num_seats)
    hero_street_commit = float(street_commit[hero]) if hero < len(street_commit) else 0.0
    to_call = max(btc - hero_street_commit, 0.0)
    if to_call > 0.0:
        out[_POT_ODDS_OFF] = to_call / (pot + to_call)
        pot_before_bet = max(pot - to_call, 1.0)
        out[_BET_PCT_POT_OFF] = float(np.log1p(to_call / pot_before_bet))

    cat = int(obs.get("hero_category_a", 0))
    if 0 <= cat < _NUM_CATEGORIES:
        out[_CAT_OFF + cat] = 1.0

    hole_list = [int(x) for x in obs["hero_hole"]]
    board_list = [int(x) for x in obs["board_a"]]
    f_d, s_d = _draw_flags_nlh(hole_list, board_list)
    out[_DRAW_OFF + 0] = f_d
    out[_DRAW_OFF + 1] = s_d

    counts, struct = _pair_features(hole_list, board_list)
    for i in range(5):
        out[_PAIR_COUNT_OFF + i] = counts[i]
    for i in range(4):
        out[_BOARD_STRUCT_OFF + i] = struct[i]

    for c in hole_list:
        out[_HERO_RANK_HIST_OFF + (c // 4)] += 1.0

    visible_count = np.zeros((13, 4), dtype=np.int32)
    for c in hole_list:
        visible_count[c // 4, c % 4] = 1
    for c in board_list:
        visible_count[c // 4, c % 4] = 1
    out[_SF_OFF:_SF_OFF + 38] = _sf_features_nlh(hole_list, board_list, visible_count)

    for k in range(num_seats):
        out[_SEAT_EXISTS_OFF + k] = 1.0

    total_commit = obs["total_commit"]
    for k in range(num_seats):
        seat = (hero + k) % num_seats
        out[_TOTAL_COMMIT_OFF + k] = float(total_commit[seat]) * inv_bb
        out[_STREET_COMMIT_OFF + k] = float(street_commit[seat]) * inv_bb

    last_agg = int(obs.get("last_aggressor", -1))
    if 0 <= last_agg < num_seats:
        out[_LAST_AGGRESSOR_OFF + (last_agg - hero) % num_seats] = 1.0

    button = int(obs["button"])
    out[_HERO_BTN_DIST_OFF + (button - hero) % num_seats] = 1.0

    opp = obs.get("nlh_opp_outcome")
    if opp is not None:
        vals = list(opp)[:3]
        for i, v in enumerate(vals):
            out[_OPP_OUTCOME_OFF + i] = float(v)

    # Hole-class block (compact preflop identity).
    if len(hole_list) == 2:
        r0, r1 = hole_list[0] // 4, hole_list[1] // 4
        hi, lo = max(r0, r1), min(r0, r1)
        out[_HOLE_CLASS_OFF + 0] = 1.0 if r0 == r1 else 0.0
        out[_HOLE_CLASS_OFF + 1] = (
            1.0 if hole_list[0] % 4 == hole_list[1] % 4 else 0.0
        )
        out[_HOLE_CLASS_OFF + 2] = (hi - lo) / 12.0
        out[_HOLE_CLASS_OFF + 3] = hi / 12.0
        out[_HOLE_CLASS_OFF + 4] = lo / 12.0

    sb_seat = obs.get("sb_seat")
    bb_seat = obs.get("bb_seat")
    if sb_seat is not None and int(sb_seat) == hero:
        out[_BLIND_FLAGS_OFF + 0] = 1.0
    if bb_seat is not None and int(bb_seat) == hero:
        out[_BLIND_FLAGS_OFF + 1] = 1.0

    return out


# ---------------------------------------------------------------------------
# Vectorized batch encoder (bit-exact vs encode_observation_nlh)
# ---------------------------------------------------------------------------

# (10, 13) bool matrix of the straight windows, row w = _STRAIGHT_WINDOWS[w]
# (slot 0 = wheel, 9 = broadway) — the batch analogue of the frozensets.
_WINDOWS_MAT = np.zeros((10, 13), dtype=bool)
for _w, _W in enumerate(_STRAIGHT_WINDOWS):
    for _r in _W:
        _WINDOWS_MAT[_w, _r] = True
del _w, _W, _r


def _draw_flags_nlh_batch(
    hole: np.ndarray, board: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized `_draw_flags_nlh`. `hole` (N, 2) u8, `board` (N, 5) u8,
    255 sentinels. Returns two (N,) float32 arrays (flush, straight)."""
    n = hole.shape[0]
    hole_valid = hole < 52
    board_valid = board < 52
    has_board = board_valid.any(axis=1)

    hole_suits = np.zeros((n, 4), dtype=np.int32)
    board_suits = np.zeros((n, 4), dtype=np.int32)
    ranks_union = np.zeros((n, 13), dtype=bool)
    for k in range(hole.shape[1]):
        v = hole_valid[:, k]
        if v.any():
            idx = np.nonzero(v)[0]
            np.add.at(hole_suits, (idx, (hole[v, k] & 3).astype(np.int64)), 1)
            ranks_union[idx, (hole[v, k] >> 2).astype(np.int64)] = True
    for k in range(5):
        v = board_valid[:, k]
        if v.any():
            idx = np.nonzero(v)[0]
            np.add.at(board_suits, (idx, (board[v, k] & 3).astype(np.int64)), 1)
            ranks_union[idx, (board[v, k] >> 2).astype(np.int64)] = True

    flush = ((hole_suits >= 1) & (hole_suits + board_suits == 4)).any(axis=1)
    window_cover = ranks_union[:, None, :] & _WINDOWS_MAT[None, :, :]
    straight = (window_cover.sum(axis=2) >= 4).any(axis=1)

    flush_f = np.where(has_board, flush, False).astype(np.float32)
    straight_f = np.where(has_board, straight, False).astype(np.float32)
    return flush_f, straight_f


def _sf_features_nlh_batch(
    hole: np.ndarray, board: np.ndarray, visible_count: np.ndarray
) -> np.ndarray:
    """Vectorized `_sf_features_nlh`. Returns (N, 38) float32 with the same
    sub-layout. `visible_count` is (N, 13, 4) int with 1 iff card (r, s) is
    in env n's hole or board."""
    n = hole.shape[0]
    out = np.zeros((n, 38), dtype=np.float32)
    hole_valid = hole < 52
    board_valid = board < 52
    has_board = board_valid.any(axis=1)
    if not has_board.any():
        return out

    arange_n = np.arange(n)
    ranks_arr = np.arange(13)

    # Presence masks + suit counts + per-suit max hole rank.
    H = np.zeros((n, 13), dtype=bool)
    B = np.zeros((n, 13), dtype=bool)
    H_rs = np.zeros((n, 13, 4), dtype=bool)
    B_rs = np.zeros((n, 13, 4), dtype=bool)
    hole_suit_counts = np.zeros((n, 4), dtype=np.int32)
    board_suit_counts = np.zeros((n, 4), dtype=np.int32)
    hole_max_per_suit = np.full((n, 4), -1, dtype=np.int64)
    for k in range(hole.shape[1]):
        v = hole_valid[:, k]
        if v.any():
            idx = np.nonzero(v)[0]
            r = (hole[v, k] >> 2).astype(np.int64)
            s = (hole[v, k] & 3).astype(np.int64)
            H[idx, r] = True
            H_rs[idx, r, s] = True
            np.add.at(hole_suit_counts, (idx, s), 1)
            np.maximum.at(hole_max_per_suit, (idx, s), r)
    for k in range(5):
        v = board_valid[:, k]
        if v.any():
            idx = np.nonzero(v)[0]
            r = (board[v, k] >> 2).astype(np.int64)
            s = (board[v, k] & 3).astype(np.int64)
            B[idx, r] = True
            B_rs[idx, r, s] = True
            np.add.at(board_suit_counts, (idx, s), 1)

    vct = visible_count.sum(axis=2)  # (N, 13) global visibility per rank
    unseen_rs = visible_count == 0  # (N, 13, 4)

    # ---- Straight block (any-combo, ≤2 hole ranks may fill a window) ----
    Wm = _WINDOWS_MAT[None, :, :]  # (1, 10, 13)
    Bw = B[:, None, :] & Wm  # window ranks on board
    Lw = Wm & ~B[:, None, :]  # window ranks the board lacks
    nB = Bw.sum(axis=2)  # (N, 10)
    nL = Lw.sum(axis=2)
    straight_possible = (nB >= 3).astype(np.int32)  # (N, 10)
    LnotH = Lw & ~H[:, None, :]
    moh = LnotH.sum(axis=2)  # |L − H| per window
    makes = (moh == 0) & (nL <= 2)  # (N, 10)

    # Candidate board hits: r ∈ L with (L−{r}) ⊆ H and |L−{r}| ≤ 2; only
    # counted for windows not already made (serial `continue`).
    rest_missing = moh[:, :, None] - LnotH.astype(np.int32)
    cand = Lw & (rest_missing == 0) & (nL[:, :, None] <= 3) & ~makes[:, :, None]
    straight_outs = np.where(cand, 4 - vct[:, None, :], 0).sum(axis=2)  # (N, 10)

    any_made = makes.any(axis=1)
    h_max = 9 - np.argmax(makes[:, ::-1], axis=1)  # valid only where any_made
    srev = np.cumsum(straight_possible[:, ::-1], axis=1)[:, ::-1]
    later = srev - straight_possible  # later[:, w] = Σ straight_possible[v > w]
    snd = np.where(
        any_made, later[arange_n, np.where(any_made, h_max, 0)], 0
    )

    # ---- Flush block ----
    flush_possible = (board_suit_counts >= 3).astype(np.float32)  # (N, 4)
    total_suit = hole_suit_counts + board_suit_counts
    made_mask = total_suit >= 5  # at most one suit per env (7 cards total)
    any_made_flush = made_mask.any(axis=1)
    made_suit = np.argmax(made_mask, axis=1)  # (N,)
    hole_in_made = (
        hole_suit_counts[arange_n, made_suit] >= 1
    )
    h1_hole = hole_max_per_suit[arange_n, made_suit]
    # Board flush ("playing the board"): h1 = lowest board rank of the suit
    # (board_suit_count is 5 exactly — the board is the flush).
    B_made = B_rs[arange_n, :, made_suit]  # (N, 13)
    h1_board = np.argmax(B_made, axis=1).astype(np.int64)
    h1_made = np.where(hole_in_made, h1_hole, h1_board)
    above_made = ranks_arr[None, :] > h1_made[:, None]
    unseen_made = unseen_rs[arange_n, :, made_suit]  # (N, 13)
    fnd = (above_made & unseen_made).sum(axis=1)
    fnd = np.where(any_made_flush, fnd, 0)

    # Hero-involved 4-flush draws.
    draw_mask = (hole_suit_counts >= 1) & (total_suit == 4)  # (N, 4)
    visible_per_suit = visible_count.sum(axis=1)  # (N, 4)
    fdo = np.where(draw_mask, 13 - visible_per_suit, 0)
    above_s = ranks_arr[None, :, None] > hole_max_per_suit[:, None, :]
    blockers = (above_s & unseen_rs).sum(axis=1)  # (N, 4)
    nfdo = np.where(blockers == 0, fdo, np.where(blockers == 1, 1, 0))
    nfdo = np.where(draw_mask, nfdo, 0)

    # SF outs: per suit, suited-board hits completing some window with
    # suited hole help (≤2 hole); rank set unioned across windows before
    # the visibility filter (serial builds one `sf_cands` set per suit).
    sf_outs = np.zeros((n, 4), dtype=np.int32)
    for s in range(4):
        H_s = H_rs[:, :, s]
        B_s = B_rs[:, :, s]
        Lw_s = _WINDOWS_MAT[None, :, :] & ~B_s[:, None, :]
        nL_s = Lw_s.sum(axis=2)
        LnotH_s = Lw_s & ~H_s[:, None, :]
        moh_s = LnotH_s.sum(axis=2)
        already_s = (moh_s == 0) & (nL_s <= 2)
        rest_s = moh_s[:, :, None] - LnotH_s.astype(np.int32)
        cand_s = (
            Lw_s
            & (rest_s == 0)
            & (nL_s[:, :, None] <= 3)
            & ~already_s[:, :, None]
        )
        cand_ranks = cand_s.any(axis=1)  # (N, 13) — set union across windows
        sf_outs[:, s] = (cand_ranks & unseen_rs[:, :, s]).sum(axis=1)
    sf_outs = np.where(draw_mask, sf_outs, 0)

    out[:, 0] = fnd.astype(np.float32)
    out[:, 1] = snd.astype(np.float32)
    out[:, 2:12] = straight_outs.astype(np.float32)
    out[:, 12:22] = straight_possible.astype(np.float32)
    out[:, 22:26] = flush_possible
    out[:, 26:30] = fdo.astype(np.float32)
    out[:, 30:34] = nfdo.astype(np.float32)
    out[:, 34:38] = sf_outs.astype(np.float32)

    # Serial early-returns zeros on an empty board.
    out[~has_board] = 0.0
    return out


def encode_observation_batch_nlh(
    obs_arrays: Mapping[str, np.ndarray],
    hero_category_a: np.ndarray,
    config: GameConfig,
) -> np.ndarray:
    """Vectorized NLH observation encoder. Returns (N, OBS_DIM_NLH) float32.

    `obs_arrays` is the dict from `PyBatchedEngine.observation_and_features_batch`
    for an `nlh_single`-variant engine (single board in `board_a`, blind
    seats in `sb_seat`/`bb_seat`, 40-slot history, exhaustive 3-dim
    `nlh_opp_outcome`). `hero_category_a` is the engine's any-combo
    category at the current actor (`hero_cat_b` is ignored — board B does
    not exist). Bit-exact vs `encode_observation_nlh`
    (tests/python/test_nlh_env_batched.py); all scalar arithmetic runs in
    f64 with the f32 cast on assignment, mirroring the scalar path.

    Terminal envs (actor == -1) produce all-zero rows.
    """
    actor = obs_arrays["actor"]
    n = actor.shape[0]
    num_seats = config.num_seats
    inv_bb = 1.0 / float(config.bb)

    out = np.zeros((n, OBS_DIM_NLH), dtype=np.float32)

    live_mask = actor != -1
    if not live_mask.any():
        return out

    hero_idx = np.where(live_mask, actor, 0).astype(np.int64)

    # Hole / board multi-hots.
    hole = obs_arrays["hero_hole"]
    board = obs_arrays["board_a"]
    for src, offset in ((hole, _HOLE_OFF), (board, _BOARD_OFF)):
        valid = (src < 52) & live_mask[:, None]
        if valid.any():
            rows = np.broadcast_to(np.arange(n)[:, None], src.shape)[valid]
            cols = src[valid].astype(np.int64) + offset
            out[rows, cols] = 1.0

    # Street one-hot (Preflop is LIVE for NLH).
    street = obs_arrays["street"].astype(np.int64)
    street_ok = (street < _NUM_STREET_ONEHOT) & live_mask
    if street_ok.any():
        rows = np.nonzero(street_ok)[0]
        out[rows, _STREET_OFF + street[rows]] = 1.0

    # Hero-rotated seat fields.
    rot = (
        hero_idx[:, None] + np.arange(num_seats, dtype=np.int64)[None, :]
    ) % num_seats
    folded = obs_arrays["folded"]
    all_in = obs_arrays["all_in"]
    folded_rot = np.take_along_axis(folded, rot, axis=1)
    all_in_rot = np.take_along_axis(all_in, rot, axis=1)
    out[live_mask, _ACTIVE_OFF : _ACTIVE_OFF + num_seats] = (
        (~folded_rot[live_mask]).astype(np.float32)
    )
    out[live_mask, _ALLIN_OFF : _ALLIN_OFF + num_seats] = all_in_rot[
        live_mask
    ].astype(np.float32)

    # Effective stacks (dead-capped), f64 arithmetic like the scalar path.
    stacks_f64 = obs_arrays["stacks"].astype(np.float64)
    eff_cap = obs_arrays["eff_stack_cap"].astype(np.float64)
    starting = np.asarray(config.resolved_stacks, dtype=np.float64)
    dead = np.maximum(0.0, starting[None, :] - eff_cap)
    effective = np.maximum(0.0, stacks_f64 - dead)
    effective_rot = np.take_along_axis(effective, rot, axis=1)
    out[live_mask, _STACKS_OFF : _STACKS_OFF + num_seats] = (
        effective_rot[live_mask] * inv_bb
    )

    # Scalars.
    pot = obs_arrays["pot"].astype(np.float64)
    bet_to_call = obs_arrays["bet_to_call"].astype(np.float64)
    min_bet = obs_arrays["min_bet"].astype(np.float64)
    max_bet = obs_arrays["max_bet"].astype(np.float64)
    out[live_mask, _SCALARS_OFF + 0] = pot[live_mask] * inv_bb
    out[live_mask, _SCALARS_OFF + 1] = bet_to_call[live_mask] * inv_bb
    out[live_mask, _SCALARS_OFF + 2] = min_bet[live_mask] * inv_bb
    out[live_mask, _SCALARS_OFF + 3] = max_bet[live_mask] * inv_bb

    out[live_mask, _REL_POS_OFF] = 1.0

    # History: last 40 slots, oldest-first (packer keeps exactly the last
    # `_HISTORY_DEPTH`); log1p pot-fraction instead of PLO's clip[0,2].
    history_seat = obs_arrays["history_seat"].astype(np.int64)
    history_action = obs_arrays["history_action"].astype(np.int64)
    history_chips = obs_arrays["history_chips"].astype(np.int64)
    history_street = obs_arrays["history_street"].astype(np.int64)
    history_len = obs_arrays["history_len"].astype(np.int64)
    depth = history_seat.shape[1]
    slot_idx = np.arange(depth, dtype=np.int64)[None, :]
    valid_slots = (slot_idx < history_len[:, None]) & live_mask[:, None]
    if valid_slots.any():
        rows = np.broadcast_to(np.arange(n)[:, None], (n, depth))[valid_slots]
        slots = np.broadcast_to(slot_idx, (n, depth))[valid_slots]
        rel_seats = (history_seat[valid_slots] - hero_idx[rows]) % num_seats
        actions_flat = history_action[valid_slots]
        chips_flat = history_chips[valid_slots]
        street_flat = history_street[valid_slots]
        base = _HISTORY_OFF + slots * _HISTORY_SLOT_DIM
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

        street_ok_h = (street_flat >= 0) & (street_flat < _NUM_STREET_ONEHOT)
        if street_ok_h.any():
            out[
                rows[street_ok_h],
                base[street_ok_h] + _HISTORY_STREET_OFF_REL + street_flat[street_ok_h],
            ] = 1.0

        out[rows, base + _HISTORY_CHIPS_OFF_REL] = (
            chips_flat.astype(np.float64) * inv_bb
        )

        pot_chips_i64 = obs_arrays["pot"].astype(np.int64)
        suffix = np.cumsum(history_chips[:, ::-1], axis=1)[:, ::-1]
        pot_before = pot_chips_i64[:, None] - suffix
        pot_before_flat = pot_before[valid_slots]
        frac = chips_flat.astype(np.float64) / np.maximum(
            pot_before_flat, 1
        ).astype(np.float64)
        out[rows, base + _HISTORY_FRAC_OFF_REL] = np.log1p(np.maximum(frac, 0.0))

    # SPR: log1p, not clip (deep-NL scales).
    pot_safe = np.maximum(pot, 1.0)
    spr = np.maximum(effective_rot / pot_safe[:, None], 0.0)
    out[live_mask, _SPR_OFF : _SPR_OFF + num_seats] = np.log1p(spr)[live_mask]

    # Pot odds + bet-faced (both only when facing chips).
    street_commit = obs_arrays["street_commit"]
    hero_street_commit = np.take_along_axis(
        street_commit, hero_idx[:, None], axis=1
    )[:, 0].astype(np.float64)
    to_call = np.maximum(bet_to_call - hero_street_commit, 0.0)
    facing = to_call > 0.0
    denom = pot + to_call
    pot_odds = np.where(facing, to_call / np.where(denom > 0.0, denom, 1.0), 0.0)
    out[live_mask, _POT_ODDS_OFF] = pot_odds[live_mask]
    pot_before_bet = np.maximum(pot - to_call, 1.0)
    bet_faced = np.where(facing, np.log1p(to_call / pot_before_bet), 0.0)
    out[live_mask, _BET_PCT_POT_OFF] = bet_faced[live_mask]

    # Hand-category one-hot (engine category 0 fires slot 0, matching the
    # scalar path — including preflop where the engine returns 0).
    cats = hero_category_a.astype(np.int64)
    cat_ok = (cats < _NUM_CATEGORIES) & live_mask
    if cat_ok.any():
        rows = np.nonzero(cat_ok)[0]
        out[rows, _CAT_OFF + cats[rows]] = 1.0

    # Draw flags + pair features + rank histogram.
    f_d, s_d = _draw_flags_nlh_batch(hole, board)
    out[live_mask, _DRAW_OFF + 0] = f_d[live_mask]
    out[live_mask, _DRAW_OFF + 1] = s_d[live_mask]

    counts, struct = _pair_features_batch(hole, board)
    out[live_mask, _PAIR_COUNT_OFF : _PAIR_COUNT_OFF + 5] = counts[live_mask]
    out[live_mask, _BOARD_STRUCT_OFF : _BOARD_STRUCT_OFF + 4] = struct[live_mask]

    hole_valid = hole < 52
    for k in range(hole.shape[1]):
        v = hole_valid[:, k] & live_mask
        if v.any():
            idx = np.nonzero(v)[0]
            np.add.at(
                out,
                (idx, _HERO_RANK_HIST_OFF + (hole[v, k] >> 2).astype(np.int64)),
                1.0,
            )

    # Straight/flush/SF block.
    visible_count = np.zeros((n, 13, 4), dtype=np.int32)
    for src in (hole, board):
        for k in range(src.shape[1]):
            v = src[:, k] < 52
            if v.any():
                idx = np.nonzero(v)[0]
                visible_count[
                    idx,
                    (src[v, k] >> 2).astype(np.int64),
                    (src[v, k] & 3).astype(np.int64),
                ] = 1
    sf = _sf_features_nlh_batch(hole, board, visible_count)
    out[live_mask, _SF_OFF : _SF_OFF + 38] = sf[live_mask]

    # Seat-exists + commits (raw, /bb).
    out[live_mask, _SEAT_EXISTS_OFF : _SEAT_EXISTS_OFF + num_seats] = 1.0
    total_commit = obs_arrays["total_commit"].astype(np.float64)
    street_commit_f64 = street_commit.astype(np.float64)
    total_rot = np.take_along_axis(total_commit, rot, axis=1)
    street_rot = np.take_along_axis(street_commit_f64, rot, axis=1)
    out[live_mask, _TOTAL_COMMIT_OFF : _TOTAL_COMMIT_OFF + num_seats] = (
        total_rot[live_mask] * inv_bb
    )
    out[live_mask, _STREET_COMMIT_OFF : _STREET_COMMIT_OFF + num_seats] = (
        street_rot[live_mask] * inv_bb
    )

    # Last-aggressor one-hot (blinds are not aggression → -1 stays silent).
    last_agg = obs_arrays["last_aggressor"].astype(np.int64)
    la_ok = (last_agg >= 0) & (last_agg < num_seats) & live_mask
    if la_ok.any():
        rows = np.nonzero(la_ok)[0]
        rel = (last_agg[rows] - hero_idx[rows]) % num_seats
        out[rows, _LAST_AGGRESSOR_OFF + rel] = 1.0

    # Button distance one-hot.
    button = obs_arrays["button"].astype(np.int64)
    rows = np.nonzero(live_mask)[0]
    btn_rel = (button[rows] - hero_idx[rows]) % num_seats
    out[rows, _HERO_BTN_DIST_OFF + btn_rel] = 1.0

    # Opp-outcome (exhaustive 3-dim; zeros preflop by the engine's guard).
    opp = obs_arrays["nlh_opp_outcome"].astype(np.float64)
    out[live_mask, _OPP_OUTCOME_OFF : _OPP_OUTCOME_OFF + 3] = opp[live_mask]

    # Hole-class block (both hole cards valid on live rows).
    both_valid = hole_valid.all(axis=1) & live_mask
    if both_valid.any():
        rows = np.nonzero(both_valid)[0]
        c0 = hole[rows, 0].astype(np.int64)
        c1 = hole[rows, 1].astype(np.int64)
        r0, r1 = c0 >> 2, c1 >> 2
        hi = np.maximum(r0, r1).astype(np.float64)
        lo = np.minimum(r0, r1).astype(np.float64)
        out[rows, _HOLE_CLASS_OFF + 0] = (r0 == r1).astype(np.float32)
        out[rows, _HOLE_CLASS_OFF + 1] = ((c0 & 3) == (c1 & 3)).astype(np.float32)
        out[rows, _HOLE_CLASS_OFF + 2] = (hi - lo) / 12.0
        out[rows, _HOLE_CLASS_OFF + 3] = hi / 12.0
        out[rows, _HOLE_CLASS_OFF + 4] = lo / 12.0

    # Blind flags (stored seats; -1 = variant has none).
    sb_seat = obs_arrays["sb_seat"].astype(np.int64)
    bb_seat = obs_arrays["bb_seat"].astype(np.int64)
    out[live_mask & (sb_seat == hero_idx) & (sb_seat >= 0), _BLIND_FLAGS_OFF + 0] = 1.0
    out[live_mask & (bb_seat == hero_idx) & (bb_seat >= 0), _BLIND_FLAGS_OFF + 1] = 1.0

    return out
