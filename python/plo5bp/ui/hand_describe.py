"""Best made-hand description for a PLO5 hand on one board (ClubGG-style).

Pure helper: given a seat's 5 hole-card indices and a board's dealt
community-card indices (3-5), enumerate the exactly-2-hole + 3-board PLO
combinations, pick the best 5-card hand, and format it the way ClubGG
labels it ("three of a kind, Qs", "a straight 10-A", "a flush A high",
"a pair of 8s", "four of a kind, 8s", "a straight flush, 9-K").

Card index convention matches the engine: ``rank = idx // 4`` (0='2' ..
12='A'), ``suit = idx % 4``. Category ordering matches rust_engine
hand_eval CAT_* (0=high card .. 8=straight flush); ``best_hand_category``
is pinned against the engine's ``hero_category`` over thousands of dealt
hands in tests/python/test_hand_describe.py, so a label can never disagree
with how a showdown actually ranks.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

# rank index 0..12 -> display string. "10" (not "T") to match ClubGG.
_RANK_DISP = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]


def _eval5(cards: list[int]) -> tuple[int, tuple, tuple]:
    """Rank an exact 5-card hand. Returns (category, tiebreak, info):
    higher (category, tiebreak) is the stronger hand; `info` carries the
    rank data the formatter needs. Tiebreak tuples are uniform within a
    category, so comparisons only ever line up like-with-like."""
    ranks = sorted((c // 4 for c in cards), reverse=True)
    suits = [c % 4 for c in cards]
    is_flush = len(set(suits)) == 1
    cnt = Counter(ranks)
    # groups: (rank, count) sorted by count desc then rank desc.
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    counts = sorted(cnt.values(), reverse=True)

    uniq = sorted(set(ranks))
    straight_high: int | None = None
    is_wheel = False
    if len(uniq) == 5:
        if uniq[-1] - uniq[0] == 4:
            straight_high = uniq[-1]
        elif uniq == [0, 1, 2, 3, 12]:  # A-2-3-4-5 (ace plays low)
            straight_high, is_wheel = 3, True
    is_straight = straight_high is not None

    if is_straight and is_flush:
        return 8, (straight_high,), ("sf", straight_high, is_wheel)
    if counts[0] == 4:
        return 7, (groups[0][0], groups[1][0]), ("quads", groups[0][0])
    if counts[0] == 3 and counts[1] >= 2:
        return 6, (groups[0][0], groups[1][0]), ("fh", groups[0][0], groups[1][0])
    if is_flush:
        return 5, tuple(ranks), ("flush", ranks[0])
    if is_straight:
        return 4, (straight_high,), ("straight", straight_high, is_wheel)
    if counts[0] == 3:
        return 3, (groups[0][0], *(g[0] for g in groups[1:])), ("trips", groups[0][0])
    if counts[0] == 2 and counts[1] == 2:
        return 2, (groups[0][0], groups[1][0], groups[2][0]), ("twopair", groups[0][0], groups[1][0])
    if counts[0] == 2:
        return 1, (groups[0][0], *(g[0] for g in groups[1:])), ("pair", groups[0][0])
    return 0, tuple(ranks), ("high", ranks[0])


def _best(hole, board) -> tuple[int, tuple, tuple] | None:
    hole = [int(c) for c in hole if c is not None]
    board = [int(c) for c in board if c is not None]
    if len(hole) < 2 or len(board) < 3:
        return None
    best: tuple[int, tuple, tuple] | None = None
    for h2 in combinations(hole, 2):
        for b3 in combinations(board, 3):
            e = _eval5([*h2, *b3])
            if best is None or (e[0], e[1]) > (best[0], best[1]):
                best = e
    return best


def best_hand_category(hole, board) -> int | None:
    """0..8 category of the best PLO hand (None if too few cards). Pinned
    against the engine's hero_category in tests."""
    b = _best(hole, board)
    return None if b is None else b[0]


def _fmt(info: tuple) -> str:
    d = _RANK_DISP
    kind = info[0]
    if kind == "pair":
        return f"a pair of {d[info[1]]}s"
    if kind == "twopair":
        return f"two pair, {d[info[1]]}s and {d[info[2]]}s"
    if kind == "trips":
        return f"three of a kind, {d[info[1]]}s"
    if kind == "straight":
        return "a straight A-5" if info[2] else f"a straight {d[info[1] - 4]}-{d[info[1]]}"
    if kind == "flush":
        return f"a flush {d[info[1]]} high"
    if kind == "fh":
        return f"a full house, {d[info[1]]}s full of {d[info[2]]}s"
    if kind == "quads":
        return f"four of a kind, {d[info[1]]}s"
    if kind == "sf":
        return "a straight flush, A-5" if info[2] else f"a straight flush, {d[info[1] - 4]}-{d[info[1]]}"
    return f"{d[info[1]]} high"


def describe_made_hand(hole, board) -> str | None:
    """ClubGG-style label of the best PLO hand on `board`, or None when
    fewer than 3 board cards / 2 hole cards are known."""
    b = _best(hole, board)
    return None if b is None else _fmt(b[2])


def _best_nlh(hole, board) -> tuple[int, tuple, tuple] | None:
    """Best 5-card hand under NLH rules: ANY 5 of hole+board (0, 1, or 2
    hole cards may play). None with fewer than 3 board / 2 hole cards."""
    hole = [int(c) for c in hole if c is not None]
    board = [int(c) for c in board if c is not None]
    if len(hole) < 2 or len(board) < 3:
        return None
    pool = [*hole, *board]
    best: tuple[int, tuple, tuple] | None = None
    for five in combinations(pool, 5):
        e = _eval5(list(five))
        if best is None or (e[0], e[1]) > (best[0], best[1]):
            best = e
    return best


def best_hand_category_nlh(hole, board) -> int | None:
    """0..8 category of the best NLH (any-combo) hand; None if too few
    cards. Mirrors the engine's `evaluate_nlh` category."""
    b = _best_nlh(hole, board)
    return None if b is None else b[0]


def describe_made_hand_nlh(hole, board) -> str | None:
    """ClubGG-style label of the best NLH hand (any 5 of hole+board)."""
    b = _best_nlh(hole, board)
    return None if b is None else _fmt(b[2])
