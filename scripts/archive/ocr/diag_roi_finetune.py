"""Fine-grained y sweep for seat 1 committed_label on the bug frame.

Picks the y that maximizes the OCR hit-rate in a 0.005-step grid
across the plausible range.
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
EXPECTED_CENTS = 18000


def main() -> int:
    img = cv2.imread(str(FRAME))
    if img is None:
        return 1

    base = seat_rois_for(6)[1].committed_label
    print(f"base y={base.y:.3f}  expected cents={EXPECTED_CENTS}")
    print(f"{'y':>7} | {'seat_commit':<13} | {'chip_amount':<13} | hit")
    print("-" * 60)

    hits = []
    for y_val in [0.590 + 0.005 * i for i in range(20)]:  # 0.590..0.685
        r = ROI(x=base.x, y=y_val, w=base.w, h=base.h)
        crop = r.crop(img)
        sc = text_mod.read_seat_commit(crop)
        ca = text_mod.read_chip_amount(crop)
        hit_sc = sc == EXPECTED_CENTS
        hit_ca = ca == EXPECTED_CENTS
        mark = "SC" if hit_sc else ("CA" if hit_ca else "  ")
        print(f"{y_val:>7.3f} | {str(sc):<13} | {str(ca):<13} | {mark}")
        if hit_sc or hit_ca:
            hits.append((y_val, hit_sc, hit_ca))

    print()
    print(f"hits: {len(hits)}")
    if hits:
        sc_hits = [h[0] for h in hits if h[1]]
        if sc_hits:
            # midpoint of contiguous seat_commit hits
            best = sum(sc_hits) / len(sc_hits)
            print(f"seat_commit hits at y={sc_hits}")
            print(f"suggested y = {best:.3f}  (centroid)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
