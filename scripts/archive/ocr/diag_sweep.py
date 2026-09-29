"""Sweep all 8 debug_*.png frames through per-seat primitive signals.

Table-prints whether each frame yields readable signals on seat 1's
cards_back / banner / committed / stack — to confirm the Phase 1 finding
is consistent across the whole bug-frame set, not a one-off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from plo5bp.ocr import cards as card_mod
from plo5bp.ocr import text as text_mod
from plo5bp.ocr.rois import seats as seat_rois_for

FRAMES_DIR = REPO_ROOT / "screenrecords" / "frames"
DEBUG_FRAMES = sorted(FRAMES_DIR.glob("debug_*.png"))
SEAT = 1


def main() -> int:
    rois = seat_rois_for(6)[SEAT]
    print(f"{'frame':<30} | back | banner | commit        | stack")
    print("-" * 88)
    for fp in DEBUG_FRAMES:
        img = cv2.imread(str(fp))
        if img is None:
            print(f"{fp.name:<30} | (read fail)")
            continue
        cb = rois.cards_back.crop(img)
        co = rois.committed_label.crop(img)
        st = rois.stack_label.crop(img)
        has_back = card_mod.has_cards_back(cb)
        has_banner = card_mod.has_bet_banner(cb)
        commit = text_mod.read_seat_commit(co)
        stack = text_mod.read_chip_amount(st)
        print(f"{fp.name:<30} | {str(has_back):<4} | {str(has_banner):<6} | {str(commit):<13} | {stack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
