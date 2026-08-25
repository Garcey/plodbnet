"""Preflop 169-class helpers (must match ``rust_engine/src/cfr/preflop.rs``).

Card encoding: ``card = rank * 4 + suit`` with rank 0=2 .. 12=A, suit 0..3.
Class id layout: 0..12 pairs, 13..90 suited non-pairs, 91..168 offsuit.
"""

from __future__ import annotations

from typing import Iterable

NUM_PREFLOP_CLASSES = 169
_RANKS = "23456789TJQKA"


def preflop_class_from_cards(c0: int, c1: int) -> int:
    """Return class id 0..168 for two hole cards (matches Rust ``PreflopHandClass::id``)."""
    r0, r1 = int(c0) // 4, int(c1) // 4
    s0, s1 = int(c0) % 4, int(c1) % 4
    hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)
    if hi == lo:
        return hi
    suited = r0 != r1 and s0 == s1
    idx = 0
    for h in range(13):
        for l in range(h):
            if h == hi and l == lo:
                base = 13 + (0 if suited else 78)
                return base + idx
            idx += 1
    return 0


def preflop_class_from_id(class_id: int) -> tuple[int, int, bool]:
    """Return ``(hi_rank, lo_rank, suited)`` for class id 0..168."""
    cid = int(class_id)
    if cid < 0 or cid >= NUM_PREFLOP_CLASSES:
        raise ValueError(f"class_id {cid} out of 0..{NUM_PREFLOP_CLASSES - 1}")
    if cid < 13:
        return cid, cid, False
    suited = cid < 13 + 78
    rem = cid - 13 if suited else cid - 13 - 78
    for h in range(13):
        for l in range(h):
            if rem == 0:
                return h, l, suited
            rem -= 1
    return 12, 11, False


def preflop_class_label(class_id: int) -> str:
    hi, lo, suited = preflop_class_from_id(class_id)
    if hi == lo:
        return f"{_RANKS[hi]}{_RANKS[lo]}"
    return f"{_RANKS[hi]}{_RANKS[lo]}{'s' if suited else 'o'}"


def representative_hole(class_id: int, *, blocked: Iterable[int] = ()) -> list[int]:
    """One concrete (c0, c1) with c0 < c1 for the class, avoiding ``blocked`` cards.

    Used when PolicyNet needs a hole multi-hot but the solve only has a 169-class id.
    """
    blocked_set = {int(c) for c in blocked if 0 <= int(c) < 52}
    hi, lo, suited = preflop_class_from_id(class_id)
    if hi == lo:
        # Pair: two different suits of the same rank
        for s0 in range(4):
            for s1 in range(s0 + 1, 4):
                a, b = hi * 4 + s0, hi * 4 + s1
                if a not in blocked_set and b not in blocked_set:
                    return [a, b] if a < b else [b, a]
        # Fallback if board blocks all suits (rare)
        return [hi * 4, hi * 4 + 1]
    if suited:
        for s in range(4):
            a, b = hi * 4 + s, lo * 4 + s
            if a not in blocked_set and b not in blocked_set:
                return [a, b] if a < b else [b, a]
        a, b = hi * 4, lo * 4
        return [a, b] if a < b else [b, a]
    # Offsuit: different suits
    for s0 in range(4):
        for s1 in range(4):
            if s0 == s1:
                continue
            a, b = hi * 4 + s0, lo * 4 + s1
            if a not in blocked_set and b not in blocked_set:
                return [a, b] if a < b else [b, a]
    a, b = hi * 4, lo * 4 + 1
    return [a, b] if a < b else [b, a]


def combo_to_cards(combo_id: int) -> list[int]:
    """Combo id 0..1325 → ``[c0, c1]`` with c0 < c1 (matches Rust ``combo_cards``)."""
    cid = int(combo_id)
    if cid < 0 or cid >= 1326:
        raise ValueError(f"combo_id {cid} out of 0..1325")
    c1 = 1
    while (c1 + 1) * c1 // 2 <= cid and c1 < 51:
        c1 += 1
    base = c1 * (c1 - 1) // 2
    c0 = cid - base
    return [c0, c1]


def cards_to_combo(c0: int, c1: int) -> int:
    a, b = (int(c0), int(c1)) if int(c0) < int(c1) else (int(c1), int(c0))
    return b * (b - 1) // 2 + a
