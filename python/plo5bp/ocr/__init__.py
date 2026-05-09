"""Offline OCR for ClubGG PLO5 double-board bomb-pot session screen recordings.

Phase 1 scope: stateless per-frame extractor that turns an image (BGR numpy
array) into a `FrameState`. No timeline, no event diff, no UI wiring -- those
are Phase 2+.

Public entry points:
    from plo5bp.ocr import extract_frame_state
    from plo5bp.ocr.types import Card, SeatObs, FrameState
"""

from plo5bp.ocr.extract import extract_frame_state
from plo5bp.ocr.types import Card, FrameState, SeatObs

__all__ = ["extract_frame_state", "Card", "FrameState", "SeatObs"]
