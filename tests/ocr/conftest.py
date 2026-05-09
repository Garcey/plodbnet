"""Pytest fixtures for OCR tests.

Skips the whole module if `opencv-python-headless` is not installed, and
bootstraps rank-NCC templates from the labeled fixture frames once per
session so the card classifier has something to match against.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")


REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = Path(__file__).parent / "fixtures" / "labels.json"


def _load_labels():
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)["fixtures"]


@pytest.fixture(scope="session")
def labels():
    return _load_labels()


@pytest.fixture(scope="session")
def rank_templates_bootstrapped(labels) -> bool:
    """Build rank templates from labeled rows if they don't already exist."""
    from plo5bp.ocr import cards as card_mod
    from plo5bp.ocr.tools import label_cards as labeler

    groups = {
        "board_a": "board_a",
        "board_b": "board_b",
        "hero_hole": "hero_hole",
    }

    for fx in labels:
        frame_path = REPO_ROOT / fx["frame"]
        if not frame_path.exists():
            continue
        state = fx["state"]
        for group in groups:
            labels_row = state[group]
            rank_chars = [s[0] if s else "-" for s in labels_row]
            try:
                labeler.build_templates(frame_path, group, rank_chars, overwrite=False)
            except Exception:
                pass  # tolerate partial template builds
    # Invalidate cache so subsequent tests re-read.
    card_mod._RANK_TEMPLATES_CACHE = None
    templates = card_mod._load_templates()
    return len(templates) >= 8
