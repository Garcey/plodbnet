"""ROI sanity tests: every rectangle must fit the calibration canvas
(1927x1391) and a 1920x1080 one, and have positive area — timer bars
included (TEST-033). No overlap check -- some ROIs are deliberately adjacent.
"""

from __future__ import annotations

import pytest

# No cv2 gate (review 2026-09-20 I12/J2): `rois` is pure Python, and since
# `plo5bp.ocr.__init__` stopped importing `extract` eagerly this module runs
# on machines without the `[ocr]` extras.
from plo5bp.ocr import rois


CANVASES = [rois.CALIBRATION_SIZE, (1920, 1080)]


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
        yield s.timer_bar_left
        yield s.timer_bar_band


@pytest.mark.parametrize("W, H", CANVASES)
def test_rois_within_canvas(W, H):
    for roi in _all_rois():
        x1, y1, x2, y2 = roi.abs(W, H)
        assert 0 <= x1 < x2 <= W, f"x out of bounds: {roi}"
        assert 0 <= y1 < y2 <= H, f"y out of bounds: {roi}"


@pytest.mark.parametrize("W, H", CANVASES)
def test_timer_detection_bands_tolerate_drift(W, H):
    """TOOL-012: the calibrated bar LINES are 1-2 px tall; what the detector
    reads is a band of several px centred on each line."""
    for s in rois.seats(6):
        _, ly1, _, ly2 = s.timer_bar_left.abs(W, H)
        _, by1, _, by2 = s.timer_bar_band.abs(W, H)
        assert by2 - by1 >= 7, (s.seat, by1, by2)
        assert by1 <= ly1 and ly2 <= by2, "the band must contain the line"


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


# --- TOOL-003: capture geometry -----------------------------------------------------


def test_frame_geometry_accepts_the_calibration_window():
    from plo5bp.ocr.rois import CALIBRATION_SIZE, frame_geometry

    assert frame_geometry(*CALIBRATION_SIZE) == ("ok", None)
    assert frame_geometry(1930, 1393)[0] == "ok"  # within 2%


def test_frame_geometry_rescales_the_same_shape_at_another_size():
    from plo5bp.ocr.rois import frame_geometry

    action, note = frame_geometry(1445, 1043)  # 75% of the calibration size
    assert action == "rescale" and "1445x1043" in note


def test_frame_geometry_refuses_another_aspect():
    from plo5bp.ocr.rois import frame_geometry

    action, msg = frame_geometry(1920, 1080)  # 16:9, not ClubGG's shape
    assert action == "refuse"
    assert "1920x1080" in msg and "1927x1391" in msg and "resize" in msg
    assert frame_geometry(0, 0)[0] == "refuse"
