"""Sweep seat 1's committed_label ROI in 2D, save crops + read each.

Determines whether the Butt2Butt "180" chip oval is reachable by any ROI
shift (→ ROI misalignment bug) or whether Tesseract simply can't parse
the oval glyph regardless of crop (→ OCR weakness bug).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from plo5bp.ocr import text as text_mod
from plo5bp.ocr.rois import seats as seat_rois_for, ROI

FRAME = REPO_ROOT / "screenrecords" / "frames" / "debug_1776889069.png"
OUT_DIR = REPO_ROOT / "_diag_roi_sweep"


def main() -> int:
    img = cv2.imread(str(FRAME))
    if img is None:
        print("read fail")
        return 1
    H, W = img.shape[:2]
    OUT_DIR.mkdir(exist_ok=True)

    base = seat_rois_for(6)[1].committed_label
    print(f"Base seat1 committed_label: x={base.x:.3f} y={base.y:.3f} w={base.w:.3f} h={base.h:.3f}")
    print(f"Base abs: {base.abs(W, H)}")
    print()

    # Sweep dy (up/down) and dx (left/right) in 0.01 fraction steps.
    print(f"{'dy':>6} {'dx':>6} | {'abs':<32} | read_seat_commit | read_chip_amount")
    print("-" * 90)
    for dy in [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.00, 0.01]:
        for dx in [-0.02, -0.01, 0.00, 0.01, 0.02]:
            shifted = ROI(x=base.x + dx, y=base.y + dy, w=base.w, h=base.h)
            crop = shifted.crop(img)
            sc = text_mod.read_seat_commit(crop)
            ca = text_mod.read_chip_amount(crop)
            tag = f"dy{dy:+.2f}_dx{dx:+.2f}"
            # Save noteworthy reads only for eyeball.
            if sc is not None or ca is not None:
                cv2.imwrite(str(OUT_DIR / f"seat1_commit_{tag}.png"), crop)
            print(f"{dy:>+.2f} {dx:>+.2f} | {str(shifted.abs(W, H)):<32} | {str(sc):<16} | {ca}")

    # Also try a WIDER + TALLER crop covering the whole chip-oval region.
    print()
    print("Wider crop attempts:")
    for (x, y, w, h, label) in [
        (0.660, 0.600, 0.100, 0.080, "wide_tall_centered"),
        (0.670, 0.605, 0.080, 0.070, "medium_centered"),
        (0.670, 0.610, 0.090, 0.065, "tight_on_oval"),
    ]:
        r = ROI(x=x, y=y, w=w, h=h)
        crop = r.crop(img)
        sc = text_mod.read_seat_commit(crop)
        ca = text_mod.read_chip_amount(crop)
        cv2.imwrite(str(OUT_DIR / f"seat1_commit_{label}.png"), crop)
        print(f"  {label:<22} x={x:.3f} y={y:.3f} w={w:.3f} h={h:.3f} -> seat_commit={sc} chip_amount={ca}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
