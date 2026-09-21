"""Offline OCR for ClubGG PLO5 double-board bomb-pot session screen recordings.

Phase 1 scope: stateless per-frame extractor that turns an image (BGR numpy
array) into a `FrameState`. No timeline, no event diff, no UI wiring -- those
are Phase 2+.

Public entry points:
    from plo5bp.ocr import extract_frame_state
    from plo5bp.ocr.types import Card, SeatObs, FrameState

Import safety (review 2026-09-20 I12): this package is imported by code that
never touches a pixel -- the PokerNow DOM path (`pokernow`), the event
reconstructor (`events`) and the data contracts (`types`). Importing
`extract` here eagerly pulled in OpenCV, so a machine without the `[ocr]`
extras could not even `import plo5bp.ocr.events`. `extract_frame_state` is
therefore resolved lazily (PEP 562 module `__getattr__`): the public name
still works, but cv2 is only imported when a caller actually asks for the
pixel extractor. Keep the pure-Python modules free of top-level cv2 /
pytesseract imports.
"""

from plo5bp.ocr.types import Card, FrameState, SeatObs

__all__ = ["extract_frame_state", "Card", "FrameState", "SeatObs"]


def __getattr__(name: str):
    # Deliberately NOT cached in globals(): tests monkeypatch
    # `plo5bp.ocr.extract.extract_frame_state`, and resolving through the
    # submodule on every access keeps the package-level name in sync.
    if name == "extract_frame_state":
        from plo5bp.ocr.extract import extract_frame_state

        return extract_frame_state
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
