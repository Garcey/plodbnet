"""Regression: Ten of hearts at hero hole slot 0.

The slot-0 ROI is calibrated to a fan-offset narrow width (J/K-friendly).
The 2-character "10" is wider and used to fail the 0.55 template floor.
The fix combines a height-ratio merge guard in `_preprocess_rank_glyph`
with a wide-ROI fallback in `_classify_hero_hole`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from plo5bp.ocr.extract import extract_frame_state
from plo5bp.ocr.types import Card

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_slot0_ten_of_hearts():
    frame = REPO_ROOT / "screenrecords" / "frames" / "debug_1777231338393.png"
    if not frame.exists():
        pytest.skip(f"missing fixture {frame}")
    img = cv2.imread(str(frame))
    assert img is not None
    fs = extract_frame_state(img, num_seats=6)
    assert fs.hero_hole[0] == Card(rank=8, suit=2)
