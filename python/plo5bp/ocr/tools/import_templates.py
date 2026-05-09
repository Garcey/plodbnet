"""Postprocess user-cropped rank-glyph PNGs into the binary format
``_load_templates`` expects.

Workflow
========

1. User crops rank glyphs from a native-resolution ClubGG screenshot
   and saves them to ``python/plo5bp/ocr/templates/`` with names like
   ``5_a.png``, ``J_clubs.png``, ``A_spades_2.png``. The first
   underscore-delimited token must be a rank char from
   ``types.RANK_CHARS`` (``2 3 4 5 6 7 8 9 T J Q K A``); the remaining
   tag distinguishes multiple captures of the same rank.

2. Run::

       .venv/Scripts/python -m plo5bp.ocr.tools.import_templates

   The CLI scans the templates directory for PNGs whose filename does
   NOT already start with ``rank_`` (those are postprocessed outputs
   and should be skipped). For each raw crop it produces a binary
   tight-bbox sibling at ``rank_<R>_<tag>.png``.

The grayscale → threshold → tight-bbox pipeline mirrors
``cards._preprocess_rank_glyph`` so user-imported and runtime-extracted
glyphs end up under identical rules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from plo5bp.ocr import cards as card_mod
from plo5bp.ocr.types import RANK_CHARS

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"


def _postprocess(bgr: np.ndarray) -> np.ndarray:
    """Grayscale → threshold(150) → tight-bbox crop.

    Mirrors ``cards._preprocess_rank_glyph`` so imported templates and
    runtime-extracted glyphs use identical extraction rules. Raises
    ``ValueError`` if the crop is too sparse, too dense, or degenerate.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    _, binm = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    n_white = int(binm.sum() / 255)
    if n_white < 40:
        raise ValueError(
            f"<40 white pixels (got {n_white}) — crop too tight or threshold rejected the glyph"
        )
    # Note: the runtime preprocessor enforces a >50%-white upper bound, but
    # that's only sensible there because the runtime "corner" region is
    # mostly background. User-supplied crops are tight by design, so a high
    # white ratio is expected. Bbox-size guards below catch degenerate cases.
    ys, xs = np.where(binm > 0)
    y_lo, y_hi = int(ys.min()), int(ys.max()) + 1
    x_lo, x_hi = int(xs.min()), int(xs.max()) + 1
    if (y_hi - y_lo) < 12 or (x_hi - x_lo) < 6:
        raise ValueError(
            f"degenerate bbox {y_hi - y_lo}x{x_hi - x_lo} after thresholding"
        )
    return binm[y_lo:y_hi, x_lo:x_hi]


def import_one(path: Path) -> Path:
    """Postprocess a single raw crop. Returns the written output path."""
    stem = path.stem
    parts = stem.split("_", 1)
    rank_token = parts[0].upper()
    # "10" is what ClubGG actually renders on the card face; treat it as
    # an alias for the canonical "T" used in RANK_CHARS.
    rank_char = "T" if rank_token == "10" else rank_token
    if rank_char not in RANK_CHARS:
        raise ValueError(
            f"unknown rank char {rank_token!r} in {path.name}; expected one of {RANK_CHARS} (or '10')"
        )
    rank = RANK_CHARS.index(rank_char)
    tag = parts[1] if len(parts) > 1 else "0"

    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    glyph = _postprocess(bgr)
    out_path = TEMPLATES_DIR / f"rank_{rank}_{tag}.png"
    cv2.imwrite(str(out_path), glyph)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Postprocess raw rank-glyph crops in templates/ into binary rank_*.png templates."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=TEMPLATES_DIR,
        help=f"Templates directory (default: {TEMPLATES_DIR})",
    )
    args = parser.parse_args(argv)

    if not args.dir.exists():
        print(f"templates dir not found: {args.dir}", file=sys.stderr)
        return 2

    written: list[Path] = []
    errors: list[tuple[Path, str]] = []
    for p in sorted(args.dir.glob("*.png")):
        if p.stem.startswith("rank_"):
            continue
        try:
            out = import_one(p)
        except (ValueError, FileNotFoundError) as e:
            errors.append((p, str(e)))
            continue
        written.append(out)
        print(f"  {p.name} -> {out.name}")

    card_mod._RANK_TEMPLATES_CACHE = None

    print()
    print(f"wrote {len(written)} template(s)")
    if errors:
        print(f"{len(errors)} error(s):")
        for p, msg in errors:
            print(f"  {p.name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
