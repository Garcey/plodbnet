"""Phase 1 CLI: `python -m plo5bp.ocr <png_path>` -> FrameState JSON on stdout.

Used to eyeball ROI alignment and classifier output against saved frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from plo5bp.ocr.extract import extract_frame_state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="plo5bp.ocr")
    ap.add_argument("image", type=Path, help="Path to a PNG/JPEG frame")
    ap.add_argument(
        "--num-seats",
        type=int,
        default=6,
        help="Seat count for layout selection (Phase 1: 6 only)",
    )
    ap.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output",
    )
    args = ap.parse_args(argv)

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"error: failed to read {args.image}", file=sys.stderr)
        return 2

    state = extract_frame_state(img, num_seats=args.num_seats)
    json.dump(
        state.to_dict(),
        sys.stdout,
        indent=2 if args.pretty else None,
        sort_keys=False,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
