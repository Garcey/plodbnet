"""Build rank-NCC templates from a labeled frame.

Workflow:
    1. Pick a frame where you know the rank of each card in a specific ROI row
       (e.g. frame_0840 hero_hole: T, 9, 9, 5, 3).
    2. Call `build_templates(frame_path, {slot_name: rank_char, ...})`.
    3. PNGs are written to `python/plo5bp/ocr/templates/rank_<R>.png`.

Runs locally, does NOT load frames into Claude's context. Existing templates
for the same rank are NOT overwritten -- so the first sighting wins and later
runs can add missing ranks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from plo5bp.ocr import cards as card_mod
from plo5bp.ocr import rois as roi_mod
from plo5bp.ocr.types import RANK_CHARS


def _extract_glyph(card_bgr: np.ndarray) -> np.ndarray | None:
    return card_mod._preprocess_rank_glyph(card_bgr)


_ROI_GROUPS: dict[str, tuple[roi_mod.ROI, ...]] = {
    "board_a": roi_mod.BOARD_A,
    "board_b": roi_mod.BOARD_B,
    "hero_hole": roi_mod.HERO_HOLE,
}


def build_templates(
    frame_path: Path,
    group: str,
    rank_chars: list[str],
    overwrite: bool = False,
) -> list[Path]:
    """Extract rank templates from a labeled row.

    Writes one file per (rank, source-frame+group+slot) so the classifier
    can see the same rank rendered under different card-body colors /
    neighboring fan positions. File naming: ``rank_<R>_<tag>.png`` where
    tag is ``<frame-stem>_<group>_<slot>``. The legacy ``rank_<R>.png``
    is still picked up by the loader but no longer written.
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(f"could not read {frame_path}")
    rois = _ROI_GROUPS[group]
    if len(rank_chars) != len(rois):
        raise ValueError(f"expected {len(rois)} labels, got {len(rank_chars)}")
    card_mod.TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    frame_tag = frame_path.stem
    written: list[Path] = []
    for slot_idx, (roi, rc) in enumerate(zip(rois, rank_chars)):
        if not rc or rc == "-":
            continue
        if rc.upper() not in RANK_CHARS:
            raise ValueError(f"unknown rank char {rc!r}")
        rank = RANK_CHARS.index(rc.upper())
        tag = f"{frame_tag}_{group}_{slot_idx}"
        out_path = card_mod.TEMPLATES_DIR / f"rank_{rank}_{tag}.png"
        if out_path.exists() and not overwrite:
            continue
        crop = roi.crop(img)
        glyph = _extract_glyph(crop)
        if glyph is None:
            print(f"warn: could not extract glyph for {rc!r} at roi {roi}")
            continue
        cv2.imwrite(str(out_path), glyph)
        written.append(out_path)
    card_mod._RANK_TEMPLATES_CACHE = None  # invalidate cache
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="plo5bp.ocr.tools.label_cards")
    ap.add_argument("frame", type=Path)
    ap.add_argument(
        "--group",
        choices=sorted(_ROI_GROUPS.keys()),
        required=True,
    )
    ap.add_argument(
        "--labels",
        required=True,
        help="Comma-separated rank chars for the 5 slots, e.g. T,9,9,5,3 (use - for unrevealed)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    labels = [x.strip() for x in args.labels.split(",")]
    written = build_templates(args.frame, args.group, labels, overwrite=args.overwrite)
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
