"""ROI sanity tests: every rectangle must fit a 1920x1080 canvas and have
positive area. No overlap check -- some ROIs are deliberately adjacent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")

from plo5bp.ocr import rois


W, H = 1920, 1080


def _all_rois():
    yield from rois.BOARD_A
    yield from rois.BOARD_B
    yield from rois.HERO_HOLE
    yield rois.POT_BANNER
    for s in rois.seats(6):
        yield s.name_plate
        yield s.stack_label
        yield s.committed_label
        yield s.button_anchor
        yield s.cards_back


def test_rois_within_canvas():
    for roi in _all_rois():
        x1, y1, x2, y2 = roi.abs(W, H)
        assert 0 <= x1 < x2 <= W, f"x out of bounds: {roi}"
        assert 0 <= y1 < y2 <= H, f"y out of bounds: {roi}"


def test_rois_positive_area():
    for roi in _all_rois():
        assert roi.w > 0 and roi.h > 0, f"non-positive area: {roi}"


def test_board_rows_have_5_cards():
    assert len(rois.BOARD_A) == 5
    assert len(rois.BOARD_B) == 5
    assert len(rois.HERO_HOLE) == 5


def test_seat_count():
    assert len(rois.seats(6)) == 6
    with pytest.raises(NotImplementedError):
        rois.seats(4)
