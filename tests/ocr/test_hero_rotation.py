"""Regression test for per-slot de-rotation of the fanned hero hole cards.

The 5 hole cards are dealt in a fan, so the edge cards are physically rotated;
`extract._classify_hero_hole` de-rotates each slot's crop before matching the
upright rank templates. These labeled frames pin the previously-misread edge
cases (8->9 leftmost, "10" unread rightmost, 8->6 rightmost) plus center cards,
and a folded/noise frame that must read nothing.

Skipped when OpenCV / the rank templates aren't available (mirrors the other
template-dependent card tests).
"""
from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from plo5bp.ocr import cards as card_mod
from plo5bp.ocr import extract as extract_mod
from plo5bp.ocr.types import RANK_CHARS, SUIT_CHARS

FIXTURES = Path(__file__).parent / "fixtures" / "rotation"

# filename -> the 5 hero hole cards (slot 0..4), as "<rank><suit>" strings.
# Hands confirmed from the live captures; slot 0 = leftmost (fan bottom),
# slot 4 = rightmost (fan top). The edge slots are the rotation-sensitive ones.
LABELED = {
    "debug_1780904885232.png": ["Ac", "Jh", "Td", "Tc", "8h"],  # rightmost 8 (was read 6)
    "debug_1780904108464.png": ["Kh", "Qs", "Qc", "Ts", "Tc"],  # rightmost 10 (was unread)
    "debug_1780877287582.png": ["8s", "6c", "3c", "2s", "2c"],  # leftmost 8 (was read 9)
    "debug_1780907909745.png": ["Ac", "Js", "Ts", "9h", "6h"],  # rightmost 6 (was read 5)
    "debug_1780824963891.png": ["As", "Kc", "7d", "5s", "4c"],  # all-correct baseline
    "debug_1780825775291.png": ["Kc", "Qs", "9d", "8s", "3d"],  # all-correct baseline
}
NOISE_FRAME = "debug_1779080738476.png"

_NO_TEMPLATES = not card_mod._load_templates()
_skip_no_templates = pytest.mark.skipif(
    _NO_TEMPLATES, reason="rank templates not available on this machine"
)


def _card_str(card) -> str | None:
    if card is None:
        return None
    return f"{RANK_CHARS[card.rank]}{SUIT_CHARS[card.suit]}"


@_skip_no_templates
@pytest.mark.parametrize("fname,expected", LABELED.items(), ids=list(LABELED))
def test_hero_hole_reads_after_derotation(fname: str, expected: list[str]):
    path = FIXTURES / fname
    if not path.exists():
        pytest.skip(f"fixture {fname} not present")
    img = cv2.imread(str(path))
    assert img is not None, f"could not read {path}"
    got = [_card_str(c) for c in extract_mod._classify_hero_hole(img)]
    assert got == expected, f"{fname}: expected {expected}, got {got}"


@_skip_no_templates
def test_noise_frame_reads_no_hero_cards():
    path = FIXTURES / NOISE_FRAME
    if not path.exists():
        pytest.skip(f"fixture {NOISE_FRAME} not present")
    img = cv2.imread(str(path))
    assert img is not None
    got = extract_mod._classify_hero_hole(img)
    assert all(c is None for c in got), f"noise frame produced reads: {got}"
