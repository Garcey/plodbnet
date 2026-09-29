"""Window-targeted screen capture for live ClubGG OCR.

Two capture paths:

* mss + GDI BitBlt (`capture` / `capture_by_match`) — for windows whose
  ``SetWindowDisplayAffinity`` is unset (it can NOT see a ClubGG table).
* Windows.Graphics.Capture (`start_wgc_capture`) — used by the live
  OCR runner. Replicates OBS's "Windows 10 (1903 and up)" capture
  source: WGC reads from the DWM compositor surface, which has the
  protected window's actual content. ClubGG sets WDA on its tables,
  so the WGC path is the only one that captures real content there.

Both return BGR numpy arrays at the window's native resolution; the live
runner brings them to the ROI calibration geometry first
(`extract.fit_to_calibration`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class WindowMatch:
    title: str
    hwnd: int
    left: int
    top: int
    width: int
    height: int

    @property
    def rect(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


class MultipleWindowsError(Exception):
    """Raised when the match substring resolves to more than one window."""

    def __init__(self, match: str, candidates: list[str]) -> None:
        super().__init__(
            f"{len(candidates)} windows match {match!r}: "
            + ", ".join(repr(c) for c in candidates)
        )
        self.match = match
        self.candidates = candidates


class NoWindowError(Exception):
    """Raised when no window title contains the match substring."""

    def __init__(self, match: str) -> None:
        super().__init__(f"no visible window titles contain {match!r}")
        self.match = match


def list_window_titles() -> list[str]:
    """Return non-empty titles of all top-level windows (best-effort)."""
    import pygetwindow as gw

    seen: list[str] = []
    for w in gw.getAllWindows():
        t = (getattr(w, "title", "") or "").strip()
        if t and t not in seen:
            seen.append(t)
    return seen


def is_window_alive(hwnd: int | None) -> bool:
    """True if `hwnd` is still a valid top-level window (False once closed).

    Stays True while the window is merely minimized/occluded — we only want
    to auto-stop OCR on actual destruction, not when the table is tucked away.
    ctypes is imported lazily (and only resolves on Windows, where this whole
    capture stack runs) to match the file's lazy-import idiom.
    """
    if not hwnd:
        return False
    import ctypes

    return bool(ctypes.windll.user32.IsWindow(int(hwnd)))


def _candidate_windows(match: str) -> list:
    import pygetwindow as gw

    needle = match.lower()
    out = []
    for w in gw.getAllWindows():
        title = (getattr(w, "title", "") or "").strip()
        if not title:
            continue
        if needle not in title.lower():
            continue
        # Skip minimized / zero-size windows — mss can't capture those.
        try:
            if getattr(w, "isMinimized", False):
                continue
            if (w.width or 0) <= 0 or (w.height or 0) <= 0:
                continue
        except Exception:
            continue
        out.append(w)
    return out


def find_window(match: str) -> WindowMatch:
    """Locate exactly one top-level window whose title contains `match`.

    Case-insensitive substring match, but with an exact-match override:
    if multiple titles contain the substring and exactly one of them
    equals it case-insensitively, that one wins. This keeps a generic
    needle like ``"testing"`` from raising ``MultipleWindowsError`` when
    a ClubGG table titled ``"testing"`` is open alongside e.g.
    ``"testing.py - VS Code"``. Otherwise raises NoWindowError /
    MultipleWindowsError so the caller can prompt the user to refine.
    """
    cands = _candidate_windows(match)
    if not cands:
        raise NoWindowError(match)
    if len(cands) > 1:
        needle = match.strip().lower()
        exact = [w for w in cands if (w.title or "").strip().lower() == needle]
        if len(exact) == 1:
            cands = exact
        else:
            raise MultipleWindowsError(match, [(w.title or "") for w in cands])
    w = cands[0]
    hwnd = int(getattr(w, "_hWnd", 0) or 0)
    return WindowMatch(
        title=(w.title or "").strip(),
        hwnd=hwnd,
        left=int(w.left),
        top=int(w.top),
        width=int(w.width),
        height=int(w.height),
    )


def capture(match: WindowMatch) -> "np.ndarray":
    """Grab the window's current pixels as a BGR numpy array.

    No rescale — output is the window's native size. Fractional ROIs are
    NOT aspect-invariant (the layout moves with the window's shape) and the
    glyph extractor has absolute pixel floors, so callers check the frame
    with `rois.frame_geometry` (a same-aspect frame of another size is
    rescaled to the calibration size; another aspect is refused).
    """
    import cv2
    import mss
    import numpy as np

    with mss.mss() as sct:
        shot = sct.grab(match.rect)
    arr = np.asarray(shot, dtype=np.uint8)  # BGRA, shape (H, W, 4)
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)


def capture_by_match(match_str: str) -> tuple[WindowMatch, "np.ndarray"]:
    """Convenience: resolve window by substring and capture in one call."""
    wm = find_window(match_str)
    return wm, capture(wm)


def start_wgc_capture(
    hwnd: int,
    on_frame: Callable[["np.ndarray"], None],
    *,
    min_interval_s: float = 0.0,
):
    """Start a free-threaded Windows.Graphics.Capture session against `hwnd`.

    Each delivered frame is copied to a contiguous BGR ndarray and
    passed to ``on_frame`` from the WGC binding's worker thread (so the
    callback must be thread-safe). Returns a ``CaptureControl``; call
    ``.stop()`` to end the session.

    ``min_interval_s``: frames arriving sooner than this after the last one
    handed on are dropped BEFORE the copy. ClubGG repaints at display rate
    (~8 MB per 1927x1391 frame) while the OCR loop reads one frame per poll,
    so the runner passes half its poll period.

    The WGC source surfaces frames from the DWM compositor, which holds
    the protected-window content even when ``SetWindowDisplayAffinity``
    is set — this is how OBS captures ClubGG tables and what we need to
    replicate. The ``windows-capture`` package owns the underlying
    BGRA buffer; we copy it before handing the slice off so callers can
    hold the ndarray past the callback's return.
    """
    import numpy as np
    from windows_capture import WindowsCapture

    capture = WindowsCapture(
        window_hwnd=int(hwnd),
        cursor_capture=False,
        draw_border=False,
    )

    last_kept = [float("-inf")]

    @capture.event
    def on_frame_arrived(frame, capture_control):  # noqa: ARG001
        now = time.monotonic()
        if now - last_kept[0] < min_interval_s:
            return
        last_kept[0] = now
        bgr = np.ascontiguousarray(frame.frame_buffer[:, :, :3])
        on_frame(bgr)

    @capture.event
    def on_closed():
        return None

    return capture.start_free_threaded()
