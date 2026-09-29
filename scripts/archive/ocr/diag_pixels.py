"""Phase 1 diagnostic: pixel / primitive signals on the Butt2Butt bug frame.

Loads a post-bet debug capture, crops seat 1's cards_back / committed_label /
stack_label ROIs, saves them to disk for visual inspection, and reports the
output of has_cards_back, has_bet_banner, read_seat_commit, read_chip_amount
plus raw HSV pixel ratios for the banner band.

Usage (from repo root):
    .venv/Scripts/python scripts/diag_pixels.py [path/to/frame.png]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from plo5bp.ocr import cards as card_mod
from plo5bp.ocr import text as text_mod
from plo5bp.ocr.rois import seats as seat_rois_for


DEFAULT_FRAME = REPO_ROOT / "screenrecords" / "frames" / "debug_1776889069.png"
SEAT_UNDER_TEST = 1  # Butt2Butt (CO / right side)

DEBUG_FRAMES = [
    "debug_1776865569.png",
    "debug_1776867418.png",
    "debug_1776870433.png",
    "debug_1776870443.png",
    "debug_1776878916.png",
    "debug_1776879944.png",
    "debug_1776879966.png",
    "debug_1776889069.png",
]


def hsv_ratio(bgr: np.ndarray, low: tuple[int, int, int], high: tuple[int, int, int]) -> float:
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
    return float(mask.sum() / 255) / mask.size


def dominant_hsv(bgr: np.ndarray) -> dict[str, float]:
    if bgr.size == 0:
        return {}
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].ravel()
    s = hsv[..., 1].ravel()
    v = hsv[..., 2].ravel()
    return {
        "H_mean": float(h.mean()),
        "H_med": float(np.median(h)),
        "S_mean": float(s.mean()),
        "S_med": float(np.median(s)),
        "V_mean": float(v.mean()),
        "V_med": float(np.median(v)),
    }


def main() -> int:
    frame_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FRAME
    print(f"[frame] {frame_path}")
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"  ERROR: cv2.imread returned None")
        return 1
    H, W = img.shape[:2]
    print(f"  shape: {W}x{H}")

    seat = seat_rois_for(6)[SEAT_UNDER_TEST]
    print(f"[seat {SEAT_UNDER_TEST} ROIs in pixels]")
    for name in ("cards_back", "committed_label", "stack_label", "name_plate"):
        roi = getattr(seat, name)
        print(f"  {name}: abs={roi.abs(W, H)}  frac=({roi.x:.3f},{roi.y:.3f},{roi.w:.3f},{roi.h:.3f})")

    cards_back_crop = seat.cards_back.crop(img)
    committed_crop = seat.committed_label.crop(img)
    stack_crop = seat.stack_label.crop(img)

    out_dir = REPO_ROOT
    cv2.imwrite(str(out_dir / "_diag_seat1_cards_back.png"), cards_back_crop)
    cv2.imwrite(str(out_dir / "_diag_seat1_committed.png"), committed_crop)
    cv2.imwrite(str(out_dir / "_diag_seat1_stack.png"), stack_crop)
    print(f"[crops saved] _diag_seat1_{{cards_back,committed,stack}}.png")

    print("\n[primitive signals on seat 1]")
    has_back = card_mod.has_cards_back(cards_back_crop)
    back_mask_ratio = hsv_ratio(cards_back_crop, (0, 0, 150), (179, 30, 220))
    print(f"  has_cards_back          = {has_back}   (threshold 0.15, measured {back_mask_ratio:.3f})")

    has_banner = card_mod.has_bet_banner(cards_back_crop)
    banner_ratio = hsv_ratio(cards_back_crop, (100, 150, 130), (115, 255, 255))
    print(f"  has_bet_banner          = {has_banner}   (threshold 0.03, measured {banner_ratio:.3f})")

    for low, high, label in [
        ((95, 100, 100), (120, 255, 255), "wide blue band"),
        ((85, 50, 100), (130, 255, 255), "very wide blue-cyan band"),
    ]:
        r = hsv_ratio(cards_back_crop, low, high)
        print(f"  [extra] {label} ratio = {r:.3f}   (low={low} high={high})")

    print(f"  [cards_back HSV stats] {dominant_hsv(cards_back_crop)}")

    commit_val = text_mod.read_seat_commit(committed_crop)
    print(f"\n  read_seat_commit        = {commit_val}   (expected 18000 for $180, or None)")
    commit_chip = text_mod.read_chip_amount(committed_crop)
    print(f"  read_chip_amount(commit)= {commit_chip}   (alt: chip-style OCR on same ROI)")
    stack_val = text_mod.read_chip_amount(stack_crop)
    print(f"  read_chip_amount(stack) = {stack_val}   (expected ~145513 for $1,455.13)")

    print("\n[also sampling adjacent ROI shifts for committed_label]")
    for dy_pct in (-0.02, -0.01, 0.0, 0.01):
        shifted = type(seat.committed_label)(
            x=seat.committed_label.x,
            y=seat.committed_label.y + dy_pct,
            w=seat.committed_label.w,
            h=seat.committed_label.h,
        )
        crop = shifted.crop(img)
        v = text_mod.read_seat_commit(crop)
        print(f"  dy={dy_pct:+.2f}  ROI y={shifted.y:.3f}  read_seat_commit={v}")

    print("\n[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
