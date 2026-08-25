"""Suit-isomorphism helpers (must match ``rust_engine/src/cfr/card_abs.rs``).

Teacher v1 does **not** train on iso-canonical cards. This module exists so
export/tests can detect the poison case (``iso_id`` treated as a raw hole).
"""

from __future__ import annotations

from typing import Sequence

from plo5bp.gto.preflop_class import cards_to_combo, combo_to_cards

# v1 teacher policy: solves emit raw combos; serve encodes raw 52-hot.
TEACHER_USE_ISOMORPHISM = False


def suit_map_for_board(board: Sequence[int]) -> list[int]:
    """Board first-seen suit → 0..3, remaining suits in order (Rust parity)."""
    mapping = [255] * 4
    nxt = 0
    for c in board:
        s = int(c) % 4
        if mapping[s] == 255:
            mapping[s] = nxt
            nxt += 1
    for s in range(4):
        if mapping[s] == 255:
            mapping[s] = nxt
            nxt += 1
    return mapping


def remap_card(card: int, mapping: Sequence[int]) -> int:
    rank = int(card) // 4
    suit = int(card) % 4
    return rank * 4 + int(mapping[suit])


def iso_combo_id(combo: int, board: Sequence[int]) -> int:
    """Canonical combo id for ``combo`` given ``board`` (Rust ``iso_combo_id``)."""
    c0, c1 = combo_to_cards(int(combo))
    mapping = suit_map_for_board(board)
    return cards_to_combo(remap_card(c0, mapping), remap_card(c1, mapping))


def teacher_solve_kwargs() -> dict:
    """Kwargs every teacher SolveConfig must include."""
    return {"use_isomorphism": TEACHER_USE_ISOMORPHISM}
