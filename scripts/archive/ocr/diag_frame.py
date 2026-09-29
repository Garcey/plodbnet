"""Phase 2 diagnostic: extract_frame_state on the post-bet bug frame.

Runs the full FrameState aggregator and dumps the serialized state so the
Phase 1 primitive-level finding (seat-1 committed_chips is None due to
misaligned ROI) can be confirmed at the FrameState level.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from plo5bp.ocr.extract import extract_frame_state

FRAME = REPO_ROOT / "screenrecords" / "frames" / "debug_1776889069.png"


def main() -> int:
    img = cv2.imread(str(FRAME))
    if img is None:
        print("read fail")
        return 1
    fs = extract_frame_state(img, num_seats=6)
    print(f"=== FrameState from {FRAME.name} ===")
    print(f"button_seat     = {fs.button_seat}")
    print(f"pot_total_chips = {fs.pot_total_chips}")
    print(f"hero_hole       = {fs.hero_hole}")
    print(f"board_a (flop)  = {fs.board_a}")
    print(f"board_b (flop)  = {fs.board_b}")
    print()
    print(f"{'seat':<4} | {'folded':<6} | {'banner':<6} | {'stack':<8} | commit")
    for s in fs.seats:
        print(
            f"{s.seat:<4} | "
            f"{str(s.folded):<6} | "
            f"{str(getattr(s, 'bet_banner', False)):<6} | "
            f"{str(s.stack_chips):<8} | "
            f"{s.committed_chips}"
        )

    print()
    print("JSON:")
    print(json.dumps(fs.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
