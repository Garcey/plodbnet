"""Read-only diagnostic: log ClubGG (Unity) window identity + display-affinity
over time, to see whether/when capture-relevant state changes during a session.

Run this ALONGSIDE your OCR session and let it run through the freeze. When
capture stops, read the log:

  * AFFINITY CHANGE 0x00 -> 0x11 on the SAME hwnd  -> app set EXCLUDEFROMCAPTURE
  * GONE hwnd ... + NEW hwnd ...                   -> the table window was recreated
                                                       (WGC was bound to the dead HWND)
  * nothing logged at the freeze time              -> the freeze is in the WGC session
                                                       itself, look there

This tool only READS window state (EnumWindows / GetWindowDisplayAffinity). It
does not capture pixels and does not modify or bypass anything.

    .venv/Scripts/python scripts/window_affinity_monitor.py --match ClubGG --interval 2
"""
from __future__ import annotations

import argparse
import ctypes
import time
from ctypes import wintypes
from datetime import datetime

import win32gui

AFFINITY = {0x00: "WDA_NONE", 0x01: "WDA_MONITOR", 0x11: "WDA_EXCLUDEFROMCAPTURE"}

_user32 = ctypes.windll.user32
_user32.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]


def get_affinity(hwnd: int):
    """Current display-affinity DWORD for hwnd (cross-process read), or None."""
    val = wintypes.DWORD()
    ok = _user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(val))
    return val.value if ok else None


def find_windows(match: str):
    """All visible windows whose title contains `match` OR are Unity windows
    with a title (catches the lobby, the table, and any newly-spawned table)."""
    needle = match.lower()
    out = []

    def cb(h, _):
        if win32gui.IsWindowVisible(h):
            title = win32gui.GetWindowText(h) or ""
            cls = win32gui.GetClassName(h)
            if (needle in title.lower()) or (cls == "UnityWndClass" and title):
                out.append((h, title, cls))
        return True

    win32gui.EnumWindows(cb, None)
    return out


def _t() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser(description="Log window display-affinity over time.")
    ap.add_argument("--match", default="ClubGG", help="title substring to track")
    ap.add_argument("--interval", type=float, default=2.0, help="poll seconds")
    args = ap.parse_args()

    print(f"[{_t()}] tracking {args.match!r} / UnityWndClass every {args.interval}s. Ctrl-C to stop.\n")
    state: dict[int, tuple[str, object]] = {}  # hwnd -> (title, affinity)
    t0 = time.time()

    try:
        while True:
            seen = set()
            for hwnd, title, cls in find_windows(args.match):
                seen.add(hwnd)
                aff = get_affinity(hwnd)
                el = int(time.time() - t0)
                prev = state.get(hwnd)
                if prev is None:
                    print(f"[{_t()} +{el}s] NEW   hwnd={hwnd} cls={cls} "
                          f"aff={AFFINITY.get(aff, aff)} title={title!r}")
                elif prev[1] != aff:
                    print(f"[{_t()} +{el}s] ** AFFINITY CHANGE ** hwnd={hwnd} "
                          f"{AFFINITY.get(prev[1], prev[1])} -> {AFFINITY.get(aff, aff)} "
                          f"title={title!r}")
                state[hwnd] = (title, aff)
            # Report any window that disappeared (recreation / close).
            for hwnd in list(state):
                if hwnd not in seen:
                    el = int(time.time() - t0)
                    print(f"[{_t()} +{el}s] GONE  hwnd={hwnd} (was {state[hwnd][0]!r})")
                    del state[hwnd]
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n[{_t()}] stopped.")


if __name__ == "__main__":
    main()
