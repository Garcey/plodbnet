"""Build the chip-amount digit templates (TOOL-033) from labelled ClubGG crops.

The in-process digit reader (``plo5bp.ocr.digits``) needs one template per
character of ClubGG's amount font. They are cut from crops whose text is known:

1. **Golden frames** (``--frames``, default ``tests/ocr/fixtures/frames``):
   every ``frame_*.png`` with a reviewed ``frame_*.state.json`` beside it (what
   ``POST /ocr/save_frame {"to_fixtures": true}`` writes — see that folder's
   README). Each seat's stack label, each non-zero bet badge and the pot banner
   is cropped with the live ROIs, preprocessed exactly as the live reader does
   (``text.prep_chip_crop`` / ``text.prep_commit_crop``) and paired with the
   amount in the JSON. The printed form (grouping commas, cents, a ``$``) is
   worked out from where the small ``.``/``,`` glyphs sit.
2. **Loose crops** (``--crops DIR``): PNGs named after the text they show,
   ``<text>__<anything>.png`` (e.g. ``1,755.59__seat2.png``); put commit-badge
   crops in a ``commit/`` subfolder (they are preprocessed like badges).

Run (the OCR extras — OpenCV — are needed to read the images)::

    .venv/Scripts/python -m plo5bp.ocr.tools.harvest_digits
    .venv/Scripts/python -m plo5bp.ocr.tools.harvest_digits --crops screenrecords/digit_crops --dry-run

It prints how many glyphs each character got, every crop it could not use and
why, and a self-check (each crop re-read with the new templates: right / left
to Tesseract / WRONG). The template file is written only when all ten digits
are covered (``--allow-incomplete`` writes anyway; the reader still ignores an
incomplete file). From then on the live OCR reads amounts in-process and falls
back to Tesseract only for glyphs it is unsure of.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

from plo5bp.ocr.digits import TEMPLATE_PATH, Sample, harvest, ink_mask

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FRAMES = REPO_ROOT / "tests" / "ocr" / "fixtures" / "frames"


def _imread(path: Path):
    import cv2  # OCR extras: only the image loading needs them

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img


def frame_samples(frames_dir: Path) -> Iterator[Sample]:
    """Labelled crops from golden frames (PNG + reviewed ``.state.json``)."""
    from plo5bp.ocr import rois as roi_mod
    from plo5bp.ocr import text as text_mod
    from plo5bp.ocr.types import FrameState

    for png in sorted(frames_dir.glob("*.png")):
        state_path = png.with_name(png.stem + ".state.json")
        if not state_path.is_file():
            continue
        fs = FrameState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
        img = _imread(png)
        seat_rois = roi_mod.seats(len(fs.seats) or 6)
        for i, (sr, seat) in enumerate(zip(seat_rois, fs.seats)):
            if seat.stack_chips is not None:
                prep = text_mod.prep_chip_crop(sr.stack_label.crop(img))
                yield Sample(ink_mask(prep, ink_dark=True), cents=seat.stack_chips, kind="stack",
                             source=f"{png.name} seat {i} stack")
            if seat.committed_chips:
                prep = text_mod.prep_commit_crop(sr.committed_label.crop(img))
                yield Sample(ink_mask(prep, ink_dark=True), cents=seat.committed_chips, kind="commit",
                             source=f"{png.name} seat {i} bet")
        if fs.pot_total_chips:
            crop = roi_mod.POT_BANNER.crop(img)
            if not text_mod._has_pot_chip_overlay(crop):
                prep = text_mod.prep_chip_crop(crop)
                yield Sample(ink_mask(prep, ink_dark=True), cents=fs.pot_total_chips, kind="pot",
                             source=f"{png.name} pot")


def crop_samples(crops_dir: Path) -> Iterator[Sample]:
    """Loose crops named ``<text>__<anything>.png`` (``commit/`` = bet badges)."""
    from plo5bp.ocr import text as text_mod

    for png in sorted(crops_dir.rglob("*.png")):
        label = png.stem.split("__", 1)[0].strip()
        if not label:
            continue
        kind = "commit" if "commit" in {p.lower() for p in png.relative_to(crops_dir).parts[:-1]} else "stack"
        img = _imread(png)
        prep = text_mod.prep_commit_crop(img) if kind == "commit" else text_mod.prep_chip_crop(img)
        yield Sample(ink_mask(prep, ink_dark=True), text=label, kind=kind, source=str(png.relative_to(crops_dir)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the chip-amount digit templates from labelled crops")
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES,
                    help="golden frames: frame_*.png + frame_*.state.json (default: %(default)s)")
    ap.add_argument("--crops", type=Path, default=None, help="loose crops named <text>__<anything>.png")
    ap.add_argument("--out", type=Path, default=TEMPLATE_PATH, help="template file (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write even when a digit is missing (the reader ignores such a file)")
    args = ap.parse_args(argv)

    try:
        samples = list(frame_samples(args.frames)) if args.frames.is_dir() else []
        if args.crops is not None:
            if not args.crops.is_dir():
                print(f"harvest_digits: no such folder {args.crops}", file=sys.stderr)
                return 2
            samples += list(crop_samples(args.crops))
    except ModuleNotFoundError as e:
        print(f"harvest_digits: {e} — install the OCR extras (pip install -e .[ocr])", file=sys.stderr)
        return 2
    if not samples:
        print(
            "harvest_digits: no labelled crops found. Save golden frames with POST /ocr/save_frame "
            '{"to_fixtures": true} (see tests/ocr/fixtures/frames/README.md) or pass --crops.',
            file=sys.stderr,
        )
        return 1
    result = harvest(samples)
    print(result.report())
    if args.dry_run:
        return 0 if result.templates.complete else 1
    if not result.templates.complete and not args.allow_incomplete:
        print("harvest_digits: not written — some digits have no glyph yet (add crops showing them)")
        return 1
    if result.self_check.get("wrong"):
        print(f"harvest_digits: WARNING — {result.self_check['wrong']} crop(s) read wrong in the self-check; "
              "check those labels before trusting the templates")
    path = result.templates.save(args.out)
    print(f"harvest_digits: wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
