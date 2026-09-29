"""Capture a window via ``live.capture_by_match`` and save two PNGs:
the raw post-rescale frame, and a copy with every ROI rectangle
drawn on top. Used after ``remap_rois.py`` to eyeball alignment.

Run while ClubGG is foregrounded::

    .venv/Scripts/python scripts/save_capture_with_rois.py --window ClubGG
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from plo5bp.ocr import live
from plo5bp.ocr import rois as roi_mod

OUT_DIR = REPO_ROOT / "screenrecords" / "frames"

# BGR. Tuned so neighbors don't blend on ClubGG's green felt.
COLORS = {
    "board_a": (0, 255, 0),
    "board_b": (0, 200, 255),
    "hero":    (255, 255, 0),
    "pot":     (255, 0, 255),
    "stack":   (0, 255, 255),
    "commit":  (255, 0, 0),
    "button":  (0, 0, 255),
    "back":    (200, 200, 200),
    "name":    (128, 128, 0),
}


def draw(frame, roi: roi_mod.ROI, color: tuple[int, int, int], label: str) -> None:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = roi.abs(W, H)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(12, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--window",
        default="ClubGG",
        help="Window-title substring (default: ClubGG). Ignored if --frame is given.",
    )
    p.add_argument(
        "--frame",
        type=Path,
        default=None,
        help="Path to a saved 1920x1080 PNG. If given, draw on it instead of live-capturing.",
    )
    p.add_argument("--num-seats", type=int, default=6)
    args = p.parse_args()

    if args.frame is not None:
        frame = cv2.imread(str(args.frame))
        if frame is None:
            print(f"ERROR: could not read {args.frame}", file=sys.stderr)
            return 1
        wm = None
    else:
        try:
            wm, frame = live.capture_by_match(args.window)
        except (live.NoWindowError, live.MultipleWindowsError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    overlay = frame.copy()
    for i, r in enumerate(roi_mod.BOARD_A):
        draw(overlay, r, COLORS["board_a"], f"A{i}")
    for i, r in enumerate(roi_mod.BOARD_B):
        draw(overlay, r, COLORS["board_b"], f"B{i}")
    for i, r in enumerate(roi_mod.HERO_HOLE):
        draw(overlay, r, COLORS["hero"], f"H{i}")
    draw(overlay, roi_mod.POT_BANNER, COLORS["pot"], "pot")
    for s in roi_mod.seats(args.num_seats):
        draw(overlay, s.name_plate,      COLORS["name"],   f"s{s.seat}.name")
        draw(overlay, s.stack_label,     COLORS["stack"],  f"s{s.seat}.stk")
        draw(overlay, s.committed_label, COLORS["commit"], f"s{s.seat}.cmt")
        draw(overlay, s.button_anchor,   COLORS["button"], f"s{s.seat}.btn")
        draw(overlay, s.cards_back,      COLORS["back"],   f"s{s.seat}.bk")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    raw_path = OUT_DIR / f"clubgg_capture_{ts}.png"
    overlay_path = OUT_DIR / f"clubgg_overlay_{ts}.png"
    cv2.imwrite(str(raw_path), frame)
    cv2.imwrite(str(overlay_path), overlay)
    if wm is not None:
        print(f"window:  {wm.title!r} {wm.width}x{wm.height}")
    else:
        print(f"source:  {args.frame}")
    print(f"raw:     {raw_path}")
    print(f"overlay: {overlay_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
