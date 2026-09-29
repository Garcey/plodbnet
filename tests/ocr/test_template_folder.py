"""The shipped rank-template folder holds exactly what the classifier loads
(TOOL-041): `rank_*.png` at the top level, reference crops under `raw/`, no
stale second template set."""

from __future__ import annotations

from pathlib import Path

import plo5bp.ocr

OCR_DIR = Path(plo5bp.ocr.__file__).parent


def test_only_rank_templates_ship_at_the_top_level():
    tops = [p.name for p in (OCR_DIR / "templates").iterdir() if p.is_file()]
    assert tops, "the classifier needs its templates"
    assert all(n.startswith("rank_") and n.endswith(".png") for n in tops), tops


def test_no_stale_template_set():
    assert not (OCR_DIR / "templates_old").exists()
