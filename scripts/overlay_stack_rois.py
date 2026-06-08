"""Overlay stack_label ROIs on a frame and OCR each crop. Diagnostic only."""

import sys
from pathlib import Path

import cv2

from plo5bp.ocr import text as text_mod
from plo5bp.ocr.rois import seats


def main(frame_path: str, out_path: str) -> None:
    img = cv2.imread(frame_path)
    if img is None:
        raise SystemExit(f"failed to read {frame_path}")
    H, W = img.shape[:2]
    overlay = img.copy()

    for sr in seats(num_seats=6):
        r = sr.stack_label
        x0 = int(r.x * W)
        y0 = int(r.y * H)
        x1 = int((r.x + r.w) * W)
        y1 = int((r.y + r.h) * H)
        crop = r.crop(img)
        val = text_mod.read_chip_amount(crop)
        val_str = f"${val/100:.2f}" if val is not None else "None"
        color = (0, 255, 0) if val is not None else (0, 0, 255)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        label = f"s{sr.seat}: {val_str}"
        ty = max(0, y0 - 8)
        cv2.putText(overlay, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
        crop_out = Path(out_path).parent / f"crop_seat{sr.seat}.png"
        cv2.imwrite(str(crop_out), crop)

    cv2.imwrite(out_path, overlay)
    print(f"wrote {out_path} ({W}x{H})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
