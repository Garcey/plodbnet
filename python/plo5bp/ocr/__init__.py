"""Live table reading for the local study build: ClubGG pixel OCR + PokerNow.

* `extract` turns one ClubGG frame (BGR numpy array) into a `FrameState`
  (cards, stacks, commits, button, pot, who is in the hand / to act);
* `events.EventReconstructor` diffs successive FrameStates against the
  engine and emits the actions / street reveals it can prove;
* `pokernow` maps a PokerNow DOM snapshot to the same FrameState;
* `live` captures the ClubGG window (Windows.Graphics.Capture).

The session wiring (runners, hand-start machine, routes) is
`plo5bp.ui.live`, mounted by the local build only.

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
