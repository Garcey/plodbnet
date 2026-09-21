"""All-in runout helpers: side-pot layers, winning combos, per-board equity.

Chip-layer math mirrors ``rust_engine/src/double_board.rs``: unique commit
levels, each layer split half/half across the two boards, ties split with
odd chips clockwise from the button. Deepest layer (fewest eligible
players) is first so the UI can animate side pots before the main pot.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Any

from plo5bp.ui.hand_describe import _eval5, best_combo

AWARD_SECS = 1.4


def pot_layers(
    total_commit: list[int], folded: list[bool]
) -> list[dict[str, Any]]:
    """Side-pot layers, deepest first.

    ``eligible`` are non-folded seats who put in at least this level.
    Folded chips still feed the layer size.
    """
    n = len(total_commit)
    levels = sorted({int(c) for c in total_commit if int(c) > 0})
    layers: list[dict[str, Any]] = []
    prev = 0
    for level in levels:
        contributors = sum(1 for c in total_commit if int(c) >= level)
        chips = (level - prev) * contributors
        prev = level
        if chips <= 0:
            continue
        eligible = [
            i
            for i in range(n)
            if int(total_commit[i]) >= level and not folded[i]
        ]
        layers.append(
            {
                "level": int(level),
                "chips": int(chips),
                "eligible": eligible,
            }
        )
    layers.reverse()
    return layers


def _distribute(amount: int, winners: list[int], button: int, n: int) -> dict[int, int]:
    if amount <= 0 or not winners:
        return {}
    share, rem = divmod(int(amount), len(winners))
    out = {w: share for w in winners}
    if rem:
        given = 0
        for step in range(1, 2 * n + 1):
            i = (int(button) + step) % n
            if i in out:
                out[i] += 1
                given += 1
                if given >= rem:
                    break
    return out


def _winners_on_board(
    holes: list[list[int] | None], eligible: list[int], board: list[int]
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    combos: dict[int, dict[str, Any]] = {}
    best: tuple[int, tuple] | None = None
    winners: list[int] = []
    for s in eligible:
        hole = holes[s] if s < len(holes) else None
        if not hole:
            continue
        combo = best_combo(hole, board)
        if combo is None:
            continue
        combos[s] = combo
        score = (int(combo["category"]), tuple(combo["tiebreak"]))
        if best is None or score > best:
            best = score
            winners = [s]
        elif score == best:
            winners.append(s)
    return winners, combos


def build_awards(
    holes: list[list[int] | None],
    folded: list[bool],
    total_commit: list[int],
    board_a: list[int],
    board_b: list[int],
    button: int,
) -> list[dict[str, Any]]:
    """Animation script: deepest side pot → main, board A then board B."""
    n = len(total_commit)
    alive = [i for i in range(n) if not folded[i]]
    awards: list[dict[str, Any]] = []
    if len(alive) <= 1:
        if alive:
            awards.append(
                {
                    "board": "a",
                    "eligible": alive,
                    "winners": alive,
                    "chips": int(sum(total_commit)),
                    "shares": {str(alive[0]): int(sum(total_commit))},
                    "combos": {},
                    "uncontested": True,
                }
            )
        return awards

    for layer in pot_layers(total_commit, folded):
        elig = layer["eligible"]
        chips = int(layer["chips"])
        if chips <= 0 or not elig:
            continue
        half_a = chips // 2
        half_b = chips - half_a
        for board_key, board, half in (("a", board_a, half_a), ("b", board_b, half_b)):
            if half <= 0:
                continue
            if len(elig) == 1:
                winners = list(elig)
                combos: dict[int, dict[str, Any]] = {}
                combo = best_combo(holes[elig[0]] or [], board) if holes[elig[0]] else None
                if combo:
                    combos[elig[0]] = combo
                uncontested = True
            else:
                winners, combos = _winners_on_board(holes, elig, board)
                uncontested = False
            if not winners:
                winners = list(elig)
            shares = _distribute(half, winners, button, n)
            awards.append(
                {
                    "board": board_key,
                    "eligible": list(elig),
                    "winners": winners,
                    "chips": half,
                    "shares": {str(k): int(v) for k, v in shares.items()},
                    "combos": {
                        str(k): {
                            "hole": v["hole"],
                            "board": v["board"],
                            "label": v["label"],
                        }
                        for k, v in combos.items()
                        if k in winners
                    },
                    "uncontested": uncontested or len(elig) == 1,
                }
            )
    return awards


# --- All-in equity ------------------------------------------------------------
#
# (review 2026-09-20 G3) The first version enumerated ORDERED PAIRS of board
# completions (~1,100 cases on a turn all-in, 250 Monte-Carlo samples on the
# flop) and called ``best_combo`` — 100 five-card evaluations — for every seat
# on every case: 0.5-2.9 s of pure Python, inside every poll, under the table
# lock. But each seat's equity is reported PER BOARD, and a board's completion
# is uniform over the m-subsets of the stub whatever the other board takes, so
# the exact per-board number needs only ONE enumeration per board (C(stub, m)
# cases, m <= 2 in a bomb pot: <= 630 heads-up on the flop, ~40 on the turn).
# On top of that the best (2 hole + 3 board) score is memoized per 3-card
# board subset — completions share almost all of them — and the 5-card rank is
# memoized on what it depends on (rank multiset + flushness).

_RANK_OF = tuple(c // 4 for c in range(52))
_SUIT_OF = tuple(c % 4 for c in range(52))
# (sorted ranks..., is_flush) -> (category, tiebreak). At most ~12k entries.
_SCORE_MEMO: dict[tuple, tuple[int, tuple]] = {}
#: Exact enumeration up to this many unknown cards per board; beyond it
#: (never in a bomb pot — hands start on the flop) sample instead.
ENUM_MAX_MISSING = 2


def _score5(c0: int, c1: int, c2: int, c3: int, c4: int) -> tuple[int, tuple]:
    """``_eval5``'s (category, tiebreak) for five distinct cards, memoized.

    Delegates to ``_eval5`` on a miss, so rankings can never drift from the
    describer that is pinned against the engine."""
    r = _RANK_OF
    su = _SUIT_OF
    s0 = su[c0]
    ranks = sorted((r[c0], r[c1], r[c2], r[c3], r[c4]))
    ranks.append(s0 == su[c1] == su[c2] == su[c3] == su[c4])
    key = tuple(ranks)
    hit = _SCORE_MEMO.get(key)
    if hit is None:
        e = _eval5([c0, c1, c2, c3, c4])
        hit = _SCORE_MEMO[key] = (int(e[0]), tuple(e[1]))
    return hit


def _board_shares(
    seats: list[int],
    pairs: dict[int, list[tuple[int, int]]],
    board: list[int],
    deck: list[int],
    samples: int,
    rng: random.Random,
) -> dict[int, float]:
    """Each seat's share (0..1) of ONE board: exact over every completion of
    the board from ``deck`` when <= ENUM_MAX_MISSING cards are missing,
    otherwise the mean over ``samples`` random completions."""
    missing = max(0, 5 - len(board))
    acc = {s: 0.0 for s in seats}
    if missing > len(deck):
        return acc
    if missing <= ENUM_MAX_MISSING:
        completions: Any = combinations(deck, missing)
    else:
        completions = (
            tuple(rng.sample(deck, missing)) for _ in range(max(1, int(samples)))
        )
    # seat -> {3 board cards -> best score over the seat's hole pairs}
    memo: dict[int, dict[tuple[int, ...], tuple[int, tuple]]] = {s: {} for s in seats}
    n_run = 0
    for extra in completions:
        n_run += 1
        triples = list(combinations((*board, *extra), 3))
        best: tuple[int, tuple] | None = None
        winners: list[int] = []
        for s in seats:
            seat_memo = memo[s]
            top: tuple[int, tuple] | None = None
            for b3 in triples:
                sc = seat_memo.get(b3)
                if sc is None:
                    b0, b1, b2 = b3
                    sc = max(_score5(h0, h1, b0, b1, b2) for h0, h1 in pairs[s])
                    seat_memo[b3] = sc
                if top is None or sc > top:
                    top = sc
            if top is None:
                continue
            if best is None or top > best:
                best, winners = top, [s]
            elif top == best:
                winners.append(s)
        if winners:
            share = 1.0 / len(winners)
            for s in winners:
                acc[s] += share
    if n_run:
        inv = 1.0 / n_run
        for s in seats:
            acc[s] *= inv
    return acc


def board_equities(
    holes: dict[int, list[int]],
    board_a: list[int],
    board_b: list[int],
    *,
    samples: int = 250,
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Per-seat share of each board (0..1), given known holes and boards.

    ``holes`` must be the ALIVE contenders only (never "whatever a viewer can
    see" — a folded viewer's own cards are not in the race). Dead cards =
    those holes + the dealt part of both boards; folded/unknown hole cards
    stay in the stub. Each board is evaluated on its own marginal (see the
    block comment above): exact for <= 2 missing cards, sampled beyond.
    """
    seats = sorted(holes)
    if not seats:
        return {}
    clean = {
        s: [int(c) for c in holes[s] if c is not None and int(c) >= 0] for s in seats
    }
    ba = [int(c) for c in board_a if c is not None and int(c) >= 0]
    bb = [int(c) for c in board_b if c is not None and int(c) >= 0]
    dead = set(ba) | set(bb)
    for h in clean.values():
        dead.update(h)
    deck = [c for c in range(52) if c not in dead]
    contenders = [s for s in seats if len(clean[s]) >= 2]
    pairs = {s: list(combinations(clean[s], 2)) for s in contenders}
    rng = random.Random(int(seed) & 0xFFFFFFFF)
    out: dict[int, dict[str, float]] = {s: {"a": 0.0, "b": 0.0} for s in seats}
    for key, board in (("a", ba), ("b", bb)):
        shares = _board_shares(contenders, pairs, board, deck, samples, rng)
        for s, v in shares.items():
            out[s][key] = round(v, 4)
    return out
