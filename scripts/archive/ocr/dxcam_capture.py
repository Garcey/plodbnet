r"""Desktop Duplication (DXGI/DDA) window-tracked capture for OCR.

A standalone, real-time screen-capture loop built on **dxcam** (Windows
Desktop Duplication API) that dynamically tracks the ClubGG poker window
and yields BGR numpy frames ready for OCR.

────────────────────────────────────────────────────────────────────────
⚠  READ THIS FIRST — DDA vs. ClubGG's display-affinity protection
────────────────────────────────────────────────────────────────────────
ClubGG calls ``SetWindowDisplayAffinity`` (WDA) on its table windows.
The Desktop Duplication API (what dxcam uses) RESPECTS WDA — so a
protected window is rendered as black, or you simply capture whatever
desktop/window is *behind* it. This is the documented reason the live
OCR runner in this repo uses Windows.Graphics.Capture (the ``windows_capture``
package) instead of DDA — WGC reads the DWM compositor surface and
bypasses WDA the way OBS's "Windows 10 (1903 and up)" source does.
See ``python/plo5bp/ocr/live.py`` and ``ui/server.py`` (OcrRunner docstring).

Consequences for THIS script:
  • For non-protected windows, dxcam is fast and great.
  • For the actual ClubGG table, the preview will very likely show the
    DESKTOP / window behind the table, or a black rectangle — NOT the
    cards. This script's purpose is therefore twofold:
      1. a clean DDA capture harness for any non-protected window, and
      2. an empirical re-test of whether WDA still blocks DDA on your
         ClubGG build (watch the preview; the built-in black-frame
         detector also prints a hint).
  • If the preview shows your desktop instead of the table, WDA is the
    cause and the production WGC path remains the only option.

This script is intentionally standalone: it does NOT modify or import the
working WGC pipeline.

────────────────────────────────────────────────────────────────────────
Installation  (this repo's venv)
────────────────────────────────────────────────────────────────────────
    .venv/Scripts/pip install dxcam pywin32
    # opencv-python and numpy are already project dependencies.

────────────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────────────
    # default: track "ClubGG", grab mode, live preview window
    .venv/Scripts/python scripts/dxcam_capture.py

    # tweak from the CLI (all override the CONFIG block below)
    .venv/Scripts/python scripts/dxcam_capture.py --window ClubGG --fps 30 --mode stream
    .venv/Scripts/python scripts/dxcam_capture.py --no-preview          # headless
    .venv/Scripts/python scripts/dxcam_capture.py --selftest            # find + 3 grabs, then exit

    # in the preview window:  q / ESC = quit,  p = toggle preview
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, Optional

import numpy as np

# ── third-party capture / window deps (fail loudly with an install hint) ──
try:
    import dxcam
    import win32api
    import win32con
    import win32gui
except ImportError as exc:  # pragma: no cover - environment guard
    sys.exit(
        f"missing dependency ({exc.name}). Install with:\n"
        "    .venv/Scripts/pip install dxcam pywin32"
    )

# cv2 is only needed for the optional preview window; import lazily so the
# capture core runs even in a minimal environment.
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ════════════════════════════════════════════════════════════════════════
# CONFIG  — edit these defaults (each is overridable via CLI; see main()).
# ════════════════════════════════════════════════════════════════════════
WINDOW_TITLE_SUBSTR = "ClubGG"   # partial, case-insensitive title match
TARGET_FPS = 60                  # capture/pacing target
CAPTURE_MODE = "grab"            # "grab" (per-frame region) | "stream" (start + get_latest_frame)
OUTPUT_COLOR = "BGR"             # dxcam channel order; BGR feeds OpenCV/OCR directly
MONITOR_IDX = 0                  # dxcam output index (0 = primary monitor)
GPU_IDX = 0                      # dxcam device index (0 = first GPU)
REGION_REFRESH_EVERY = 30        # re-query the window rect every N frames (tracks moves/resizes)
USE_DWM_FRAME_BOUNDS = True      # trim Win10/11 invisible resize borders (DWM extended frame bounds)
SHOW_PREVIEW = True              # show a downscaled OpenCV preview window
PREVIEW_MAX_W = 960              # max preview width (px); preserves aspect
PREVIEW_WINDOW = "DXcam capture (q=quit  p=toggle)"
FPS_PRINT_EVERY = 1.0            # seconds between FPS log lines
BLACK_FRAME_WARN = True          # print a WDA hint if the region stays near-black
BLACK_MEAN_THRESHOLD = 6.0       # mean pixel value below which a frame is "black"
BLACK_CONSEC_FRAMES = 15         # consecutive black frames before warning (avoids transients)
WINDOW_LOST_BACKOFF = 0.5        # seconds to wait when the window isn't capturable


# ════════════════════════════════════════════════════════════════════════
# DPI awareness — must run before any GetWindowRect so the rect we read is
# in true physical pixels that line up with the DDA framebuffer.
# ════════════════════════════════════════════════════════════════════════
def enable_dpi_awareness() -> str:
    """Make this process per-monitor DPI aware. Returns the level achieved."""
    import ctypes

    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2 (shcore, Win 8.1+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "per-monitor"
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # legacy system-DPI
            return "system"
        except Exception:
            return "none"


# ════════════════════════════════════════════════════════════════════════
# DWM helpers (ctypes) — cloaked-window filter + true visible bounds.
# ════════════════════════════════════════════════════════════════════════
_DWMWA_EXTENDED_FRAME_BOUNDS = 9
_DWMWA_CLOAKED = 14
_dwmapi = None


def _dwm():
    """Lazily bind dwmapi.DwmGetWindowAttribute with explicit signatures."""
    global _dwmapi
    if _dwmapi is None:
        import ctypes
        from ctypes import wintypes

        dll = ctypes.WinDLL("dwmapi")
        dll.DwmGetWindowAttribute.restype = ctypes.c_long  # HRESULT
        dll.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        _dwmapi = dll
    return _dwmapi


def _is_cloaked(hwnd: int) -> bool:
    """True for DWM-cloaked windows (ghost UWP/virtual-desktop entries)."""
    import ctypes
    from ctypes import wintypes

    val = wintypes.DWORD()
    hr = _dwm().DwmGetWindowAttribute(
        hwnd, _DWMWA_CLOAKED, ctypes.byref(val), ctypes.sizeof(val)
    )
    return hr == 0 and val.value != 0


def _dwm_extended_frame_bounds(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    """The window's *visible* bounds (excludes the invisible Win10/11 border).

    ``GetWindowRect`` includes a ~7px transparent resize margin on Win10/11;
    capturing that margin pulls in pixels from behind the window. The DWM
    extended frame bounds give the real on-screen edges. Returns None if the
    DWM call fails (older OS / non-composited)."""
    import ctypes
    from ctypes import wintypes

    rect = wintypes.RECT()
    hr = _dwm().DwmGetWindowAttribute(
        hwnd, _DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)
    )
    if hr != 0:
        return None
    return (rect.left, rect.top, rect.right, rect.bottom)


# ════════════════════════════════════════════════════════════════════════
# Window tracking  (pywin32: win32gui / win32con / win32api)
# ════════════════════════════════════════════════════════════════════════
def find_window_hwnd(title_substr: str = WINDOW_TITLE_SUBSTR) -> Optional[int]:
    """Return the HWND of the best visible window whose title contains
    ``title_substr`` (case-insensitive), or None if none match.

    When several windows match (e.g. a chat + the table), the largest by
    area wins — the poker table is the big one. Minimized and DWM-cloaked
    windows are skipped."""
    needle = title_substr.lower()
    matches: list[tuple[int, int]] = []  # (area, hwnd)

    def _enum(hwnd: int, _ctx) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title or needle not in title.lower():
            return True
        if _is_minimized(hwnd) or _is_cloaked(hwnd):
            return True
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        area = max(0, right - left) * max(0, bottom - top)
        if area > 0:
            matches.append((area, hwnd))
        return True

    win32gui.EnumWindows(_enum, None)
    if not matches:
        return None
    matches.sort(reverse=True)  # largest area first
    return matches[0][1]


def _is_minimized(hwnd: int) -> bool:
    """True if the window is minimized (placement showCmd == SW_SHOWMINIMIZED)."""
    try:
        placement = win32gui.GetWindowPlacement(hwnd)
        return placement[1] == win32con.SW_SHOWMINIMIZED
    except Exception:
        return False


def get_window_rect(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    """Screen-space (left, top, right, bottom) of the window's visible area,
    or None if it can't be captured (minimized)."""
    if _is_minimized(hwnd):
        return None
    if USE_DWM_FRAME_BOUNDS:
        bounds = _dwm_extended_frame_bounds(hwnd)
        if bounds is not None:
            return bounds
    return tuple(win32gui.GetWindowRect(hwnd))  # type: ignore[return-value]


# Cache the HWND so the common case (window present, just moved) skips a
# full EnumWindows every refresh. Revalidated each call.
_cached_hwnd: Optional[int] = None


def get_clubgg_region(
    title_substr: str = WINDOW_TITLE_SUBSTR,
) -> Optional[tuple[int, int, int, int]]:
    """Current (left, top, right, bottom) bounding box of the target window
    in screen coordinates, or None if the window isn't found / is minimized.

    This is the single source of truth for *where* to capture; the main loop
    calls it periodically so a moved or resized window is tracked live."""
    global _cached_hwnd

    hwnd = _cached_hwnd
    valid = (
        hwnd is not None
        and win32gui.IsWindow(hwnd)
        and win32gui.IsWindowVisible(hwnd)
        and title_substr.lower() in (win32gui.GetWindowText(hwnd) or "").lower()
    )
    if not valid:
        hwnd = find_window_hwnd(title_substr)
        _cached_hwnd = hwnd
    if hwnd is None:
        return None
    return get_window_rect(hwnd)


def describe_window_state(title_substr: str = WINDOW_TITLE_SUBSTR) -> str:
    """Human-readable status for log lines: ok / minimized / not found."""
    hwnd = find_window_hwnd(title_substr)
    if hwnd is None:
        # find_window_hwnd skips minimized windows, so distinguish that here.
        return "not found (or minimized)"
    return "ok"


# ════════════════════════════════════════════════════════════════════════
# Monitor → output-local region mapping.
# dxcam region coords are relative to the captured OUTPUT's top-left, while
# GetWindowRect is in virtual-desktop coords. For the primary monitor at the
# desktop origin they're identical; for others we subtract that monitor's
# origin. We also clamp to the output bounds.
# ════════════════════════════════════════════════════════════════════════
def _output_origin_for_point(x: int, y: int) -> tuple[int, int]:
    """(left, top) of the monitor containing screen point (x, y); (0, 0) if
    it can't be determined."""
    try:
        for _hmon, _hdc, rect in win32api.EnumDisplayMonitors():
            l, t, r, b = rect
            if l <= x < r and t <= y < b:
                return l, t
    except Exception:
        pass
    return 0, 0


def screen_to_output_region(
    rect: tuple[int, int, int, int], out_w: int, out_h: int
) -> Optional[tuple[int, int, int, int]]:
    """Convert a screen-space rect to an output-local, clamped dxcam region.

    Returns None if the window doesn't overlap the captured output (e.g. it's
    on a different monitor than MONITOR_IDX) — the caller treats that like a
    lost window."""
    l, t, r, b = rect
    ox, oy = _output_origin_for_point((l + r) // 2, (t + b) // 2)
    l, t, r, b = l - ox, t - oy, r - ox, b - oy
    # Clamp into [0, out_w] x [0, out_h].
    l = max(0, min(int(l), out_w))
    r = max(0, min(int(r), out_w))
    t = max(0, min(int(t), out_h))
    b = max(0, min(int(b), out_h))
    if r - l < 4 or b - t < 4:
        return None
    return (l, t, r, b)


# ════════════════════════════════════════════════════════════════════════
# Capture initialization.
# ════════════════════════════════════════════════════════════════════════
def create_camera(
    monitor_idx: int = MONITOR_IDX,
    gpu_idx: int = GPU_IDX,
    color: str = OUTPUT_COLOR,
):
    """Create a dxcam DXCamera for one output. dxcam allows only ONE camera
    per output, so callers must ``release()`` before recreating."""
    cam = dxcam.create(device_idx=gpu_idx, output_idx=monitor_idx, output_color=color)
    if cam is None:
        raise RuntimeError(
            f"dxcam.create failed for device={gpu_idx} output={monitor_idx}. "
            "Check the GPU/monitor indices."
        )
    return cam


def release_camera(cam) -> None:
    """Stop + release a camera, swallowing benign teardown errors."""
    if cam is None:
        return
    try:
        if cam.is_capturing:
            cam.stop()
    except Exception:
        pass
    try:
        cam.release()
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════
# Per-frame handling: black-frame/WDA detector, OCR hook, preview.
# ════════════════════════════════════════════════════════════════════════
class FrameProcessor:
    """Stateful per-frame sink: tracks the black-frame streak (for the WDA
    hint), forwards frames to an optional OCR callback, and renders the
    preview window."""

    def __init__(
        self,
        ocr_callback: Optional[Callable[[np.ndarray], None]] = None,
        show_preview: bool = SHOW_PREVIEW,
    ) -> None:
        self.ocr_callback = ocr_callback
        self.show_preview = show_preview and cv2 is not None
        self._black_streak = 0
        self._warned_black = False
        self._preview_open = False

    def handle(self, frame: np.ndarray) -> bool:
        """Process one frame. Returns False if the user asked to quit."""
        if BLACK_FRAME_WARN:
            self._check_black(frame)
        if self.ocr_callback is not None:
            # Hand the raw BGR frame to OCR. This is the integration point:
            # e.g. plo5bp.ocr.extract.extract_frame_state(frame).
            try:
                self.ocr_callback(frame)
            except Exception as exc:  # never let an OCR error kill capture
                print(f"  [ocr] callback error: {exc}")
        if self.show_preview:
            return self._render_preview(frame)
        return True

    def _check_black(self, frame: np.ndarray) -> None:
        """Maintain the near-black streak and emit a one-time WDA hint.

        Diagnostic only — does not touch the frame data. A protected window
        captured via DDA is often black; this turns that failure mode into a
        clear message instead of silent empty OCR."""
        if float(frame.mean()) < BLACK_MEAN_THRESHOLD:
            self._black_streak += 1
        else:
            self._black_streak = 0
            self._warned_black = False
            return
        if self._black_streak >= BLACK_CONSEC_FRAMES and not self._warned_black:
            self._warned_black = True
            print(
                "\n  ⚠  Region has been near-black for "
                f"{self._black_streak} frames.\n"
                "     Likely cause: the target window uses SetWindowDisplayAffinity\n"
                "     (WDA) protection, which the Desktop Duplication API respects.\n"
                "     ClubGG does this on its tables — DDA/dxcam cannot read them.\n"
                "     Use the Windows.Graphics.Capture path (windows_capture) instead.\n"
            )

    def _render_preview(self, frame: np.ndarray) -> bool:
        """Show a downscaled preview. Returns False on quit (q/ESC)."""
        h, w = frame.shape[:2]
        if w > PREVIEW_MAX_W:
            scale = PREVIEW_MAX_W / float(w)
            disp = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            disp = frame
        # frame is already BGR when OUTPUT_COLOR == "BGR"; convert if not.
        if OUTPUT_COLOR.upper() == "RGB":
            disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
        cv2.imshow(PREVIEW_WINDOW, disp)
        self._preview_open = True
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            return False
        if key == ord("p"):  # toggle preview off
            self.show_preview = False
            self.close_preview()
        return True

    def close_preview(self) -> None:
        if cv2 is not None and self._preview_open:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
            except Exception:
                pass
            self._preview_open = False


class FpsMeter:
    """Rolling FPS printer (every FPS_PRINT_EVERY seconds)."""

    def __init__(self) -> None:
        self._count = 0
        self._t0 = time.perf_counter()

    def tick(self, region: Optional[tuple], shape: Optional[tuple]) -> None:
        self._count += 1
        now = time.perf_counter()
        elapsed = now - self._t0
        if elapsed >= FPS_PRINT_EVERY:
            fps = self._count / elapsed
            dims = f"{shape[1]}x{shape[0]}" if shape else "—"
            print(f"  {fps:5.1f} fps | region={region} | frame={dims}")
            self._count = 0
            self._t0 = now


# ════════════════════════════════════════════════════════════════════════
# Main capture loops — two strategies, selectable via CAPTURE_MODE.
# ════════════════════════════════════════════════════════════════════════
def run_grab_mode(cam, processor: FrameProcessor, target_fps: int) -> None:
    """On-demand capture: call cam.grab(region=...) each iteration with a
    freshly-tracked region. Best for a window that moves, since the region is
    re-supplied per frame. ``new_frame_only=False`` returns the latest buffer
    even when the screen is static (poker tables sit still between actions)."""
    fps = FpsMeter()
    frame_interval = 1.0 / max(1, target_fps)
    next_tick = time.perf_counter()
    region: Optional[tuple[int, int, int, int]] = None
    last_frame: Optional[np.ndarray] = None
    i = 0

    while True:
        # Refresh the tracked region periodically (and on the first frame).
        if i % REGION_REFRESH_EVERY == 0 or region is None:
            screen_rect = get_clubgg_region()
            if screen_rect is None:
                print(f"  window {describe_window_state()!r}; retrying…")
                region = None
                time.sleep(WINDOW_LOST_BACKOFF)
                i = 0
                continue
            region = screen_to_output_region(screen_rect, cam.width, cam.height)
            if region is None:
                print("  window not on the captured monitor (set --monitor); retrying…")
                time.sleep(WINDOW_LOST_BACKOFF)
                continue

        try:
            frame = cam.grab(region=region, new_frame_only=False)
        except Exception as exc:
            print(f"  grab error: {exc}; recreating camera…")
            cam = _recreate(cam)
            region = None
            continue

        if frame is None:
            frame = last_frame  # nothing new yet; reuse previous
        else:
            last_frame = frame

        if frame is not None:
            if not processor.handle(frame):
                break
            fps.tick(region, frame.shape)

        i += 1
        # Pace to target FPS.
        next_tick += frame_interval
        sleep = next_tick - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_tick = time.perf_counter()  # fell behind; resync


def run_stream_mode(cam, processor: FrameProcessor, target_fps: int) -> None:
    """Continuous capture: cam.start(...) spins a background grab thread and
    get_latest_frame() returns the newest frame (video_mode=True re-delivers
    the last frame so a static table doesn't block). The region is fixed at
    start(), so we stop/restart when the window moves."""
    fps = FpsMeter()
    region: Optional[tuple[int, int, int, int]] = None
    i = 0

    def _start(reg) -> bool:
        try:
            cam.start(region=reg, target_fps=target_fps, video_mode=True)
            return True
        except Exception as exc:
            print(f"  start error: {exc}")
            return False

    try:
        while True:
            # (Re)acquire the region and (re)start capture if it changed.
            if i % REGION_REFRESH_EVERY == 0 or region is None:
                screen_rect = get_clubgg_region()
                new_region = (
                    screen_to_output_region(screen_rect, cam.width, cam.height)
                    if screen_rect is not None
                    else None
                )
                if new_region is None:
                    print(f"  window {describe_window_state()!r}; retrying…")
                    if cam.is_capturing:
                        cam.stop()
                    region = None
                    time.sleep(WINDOW_LOST_BACKOFF)
                    i = 0
                    continue
                if new_region != region:
                    if cam.is_capturing:
                        cam.stop()
                    if not _start(new_region):
                        time.sleep(WINDOW_LOST_BACKOFF)
                        continue
                    region = new_region

            try:
                frame = cam.get_latest_frame()  # blocks until a frame is ready
            except Exception as exc:
                print(f"  get_latest_frame error: {exc}; recreating camera…")
                cam = _recreate(cam)
                region = None
                continue

            if frame is not None:
                if not processor.handle(frame):
                    break
                fps.tick(region, frame.shape)
            i += 1
    finally:
        if cam.is_capturing:
            cam.stop()


def _recreate(cam):
    """Release a faulted camera and make a fresh one (device-lost recovery)."""
    release_camera(cam)
    time.sleep(0.2)
    return create_camera()


# ════════════════════════════════════════════════════════════════════════
# OCR integration point.
# ════════════════════════════════════════════════════════════════════════
def make_ocr_callback() -> Optional[Callable[[np.ndarray], None]]:
    """Return a callback that runs the project's frame extractor, or None if
    the package isn't importable. Wire real OCR in here.

    NOTE: extract_frame_state expects a full-window BGR frame at native
    resolution (ROIs are fractional). Under WDA this will see black/behind
    pixels — see the warning at the top of this file."""
    try:
        from plo5bp.ocr.extract import extract_frame_state
    except Exception as exc:
        print(f"  [ocr] extractor unavailable ({exc}); running capture only.")
        return None

    def _cb(frame: np.ndarray) -> None:
        fs = extract_frame_state(frame)
        # Replace with whatever downstream consumption you need.
        print(f"  [ocr] button={fs.button_seat} pot={fs.pot}")

    return _cb


# ════════════════════════════════════════════════════════════════════════
# Self-test: find the window, print region, grab a few frames, exit.
# Non-interactive — no preview, no infinite loop.
# ════════════════════════════════════════════════════════════════════════
def selftest(title_substr: str) -> int:
    print(f"DPI awareness: {enable_dpi_awareness()}")
    print(f"Searching for window containing {title_substr!r} …")
    hwnd = find_window_hwnd(title_substr)
    if hwnd is None:
        print(f"  NOT FOUND. Visible candidate titles:")
        for t in _list_titles():
            print(f"    - {t}")
        return 1
    print(f"  HWND={hwnd}  title={win32gui.GetWindowText(hwnd)!r}")
    region = get_clubgg_region(title_substr)
    print(f"  screen region (l,t,r,b) = {region}")

    cam = create_camera()
    print(f"  camera output = {cam.width}x{cam.height}")
    out_region = screen_to_output_region(region, cam.width, cam.height) if region else None
    print(f"  output-local region = {out_region}")
    try:
        for n in range(3):
            frame = cam.grab(region=out_region, new_frame_only=False)
            if frame is None:
                print(f"  grab #{n}: None (no frame yet)")
            else:
                m = float(frame.mean())
                flag = "  <-- near-black (WDA?)" if m < BLACK_MEAN_THRESHOLD else ""
                print(f"  grab #{n}: shape={frame.shape} mean={m:.1f}{flag}")
            time.sleep(0.1)
    finally:
        release_camera(cam)
    return 0


def _list_titles() -> list[str]:
    out: list[str] = []

    def _enum(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if t and t not in out:
                out.append(t)
        return True

    win32gui.EnumWindows(_enum, None)
    return out


# ════════════════════════════════════════════════════════════════════════
# CLI / entrypoint.
# ════════════════════════════════════════════════════════════════════════
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DXGI/DDA window-tracked capture (dxcam).")
    p.add_argument("--window", default=WINDOW_TITLE_SUBSTR, help="title substring to track")
    p.add_argument("--fps", type=int, default=TARGET_FPS, help="target FPS")
    p.add_argument("--mode", choices=("grab", "stream"), default=CAPTURE_MODE)
    p.add_argument("--monitor", type=int, default=MONITOR_IDX, help="dxcam output index")
    p.add_argument("--gpu", type=int, default=GPU_IDX, help="dxcam device index")
    p.add_argument("--no-preview", action="store_true", help="disable the OpenCV preview")
    p.add_argument("--no-dwm-trim", action="store_true", help="don't trim invisible borders")
    p.add_argument("--ocr", action="store_true", help="run the plo5bp OCR extractor per frame")
    p.add_argument("--selftest", action="store_true", help="find + grab 3 frames, then exit")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Fold CLI overrides into the module config the helpers read.
    global WINDOW_TITLE_SUBSTR, TARGET_FPS, CAPTURE_MODE, MONITOR_IDX, GPU_IDX
    global SHOW_PREVIEW, USE_DWM_FRAME_BOUNDS
    WINDOW_TITLE_SUBSTR = args.window
    TARGET_FPS = args.fps
    CAPTURE_MODE = args.mode
    MONITOR_IDX = args.monitor
    GPU_IDX = args.gpu
    SHOW_PREVIEW = not args.no_preview
    USE_DWM_FRAME_BOUNDS = not args.no_dwm_trim

    if args.selftest:
        return selftest(WINDOW_TITLE_SUBSTR)

    print(f"DPI awareness: {enable_dpi_awareness()}")
    if SHOW_PREVIEW and cv2 is None:
        print("  opencv not installed; preview disabled.")
        SHOW_PREVIEW = False

    ocr_cb = make_ocr_callback() if args.ocr else None
    processor = FrameProcessor(ocr_callback=ocr_cb, show_preview=SHOW_PREVIEW)

    cam = create_camera(MONITOR_IDX, GPU_IDX, OUTPUT_COLOR)
    print(
        f"dxcam ready: output {MONITOR_IDX} = {cam.width}x{cam.height} | "
        f"mode={CAPTURE_MODE} | target {TARGET_FPS} fps | tracking {WINDOW_TITLE_SUBSTR!r}\n"
        "Ctrl-C to stop."
    )
    try:
        if CAPTURE_MODE == "stream":
            run_stream_mode(cam, processor, TARGET_FPS)
        else:
            run_grab_mode(cam, processor, TARGET_FPS)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        processor.close_preview()
        release_camera(cam)
        if cv2 is not None:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
