"""Per-card classifier tests against labeled fixture frames.

Phase 1 targets a minimum accuracy; exact per-slot correctness on two
hand-labeled frames is aspirational and the user will refine ROIs /
templates as they add more fixtures via `plo5bp.ocr.tools.label_cards`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from plo5bp.ocr import cards as card_mod
from plo5bp.ocr import rois as roi_mod
from plo5bp.ocr.types import SUIT_CHARS, Card

REPO_ROOT = Path(__file__).resolve().parents[2]

_ROI_GROUPS = {
    "board_a": roi_mod.BOARD_A,
    "board_b": roi_mod.BOARD_B,
    "hero_hole": roi_mod.HERO_HOLE,
}

MIN_SUIT_ACCURACY = 0.50
MIN_RANK_ACCURACY = 0.40


def _iter_labeled_cards(labels, resolve):
    for fx in labels:
        frame_path = resolve(fx["frame"])
        if frame_path is None:
            continue
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        state = fx["state"]
        for group, rois in _ROI_GROUPS.items():
            for slot_idx, (roi, label) in enumerate(zip(rois, state[group])):
                crop = roi.crop(img)
                yield {
                    "frame": fx["frame"],
                    "group": group,
                    "slot": slot_idx,
                    "crop": crop,
                    "label": None if label is None else Card.parse(label),
                }


def _skip_if_no_ground_truth(total: int) -> None:
    """The labeled frames are LOCAL-ONLY (screenrecords/ is gitignored;
    the originals were lost in a 2026-06-26 cleanup). With zero collected
    crops there is nothing to measure — skip loudly rather than fail on
    0/0. Recovered or newly labeled frames dropped into
    tests/ocr/fixtures/frames/ (TRACKED) re-arm these tests."""
    if total == 0:
        pytest.skip(
            "no labeled fixture frames on disk — see "
            "tests/ocr/fixtures/frames/README.md to re-arm"
        )


def test_suit_accuracy_on_revealed_cards(
    labels, rank_templates_bootstrapped, frame_resolver
):
    correct = 0
    total = 0
    wrong: list[str] = []
    for item in _iter_labeled_cards(labels, frame_resolver):
        label = item["label"]
        if label is None:
            continue
        total += 1
        pred = card_mod.classify_suit(item["crop"])
        if pred == label.suit:
            correct += 1
        else:
            pred_s = SUIT_CHARS[pred] if pred is not None else "?"
            wrong.append(
                f"{item['frame']} {item['group']}[{item['slot']}]: "
                f"expected {SUIT_CHARS[label.suit]} got {pred_s}"
            )
    _skip_if_no_ground_truth(total)
    acc = correct / max(total, 1)
    print(f"\nsuit accuracy: {correct}/{total} = {acc:.2%}")
    for m in wrong[:20]:
        print(" ", m)
    assert acc >= MIN_SUIT_ACCURACY, f"suit accuracy {acc:.2%} < {MIN_SUIT_ACCURACY:.0%}"


def test_rank_accuracy_on_revealed_cards(
    labels, rank_templates_bootstrapped, frame_resolver
):
    if not rank_templates_bootstrapped:
        pytest.skip("not enough rank templates bootstrapped")
    correct = 0
    total = 0
    wrong: list[str] = []
    for item in _iter_labeled_cards(labels, frame_resolver):
        label = item["label"]
        if label is None:
            continue
        total += 1
        pred, _conf = card_mod.classify_rank(item["crop"])
        if pred == label.rank:
            correct += 1
        else:
            wrong.append(
                f"{item['frame']} {item['group']}[{item['slot']}]: "
                f"expected rank {label.rank} got {pred}"
            )
    _skip_if_no_ground_truth(total)
    acc = correct / max(total, 1)
    print(f"\nrank accuracy: {correct}/{total} = {acc:.2%}")
    for m in wrong[:20]:
        print(" ", m)
    assert acc >= MIN_RANK_ACCURACY, f"rank accuracy {acc:.2%} < {MIN_RANK_ACCURACY:.0%}"


# --- bet banner detector ------------------------------------------------


def _bet_banner_crop(blue_frac: float = 0.25) -> "np.ndarray":
    """Synthesize a card-back-sized BGR crop with a blue rectangle covering
    `blue_frac` of the image, mimicking the ClubGG 'Bet' overlay."""
    import numpy as np

    h, w = 120, 200
    img = np.full((h, w, 3), 60, dtype=np.uint8)  # dark-ish card-back bg
    if blue_frac <= 0:
        return img
    rect_w = int(w * (blue_frac ** 0.5))
    rect_h = int(h * (blue_frac ** 0.5))
    y0 = (h - rect_h) // 2
    x0 = (w - rect_w) // 2
    # ClubGG banner blue ≈ hue 105 in OpenCV; in BGR that's roughly
    # (220, 80, 30) — saturated blue.
    img[y0 : y0 + rect_h, x0 : x0 + rect_w] = (220, 80, 30)
    return img


def test_has_bet_banner_detects_solid_blue_overlay():
    crop = _bet_banner_crop(blue_frac=0.20)
    assert card_mod.has_bet_banner(crop) is True


def test_has_bet_banner_rejects_empty_cardback():
    crop = _bet_banner_crop(blue_frac=0.0)
    assert card_mod.has_bet_banner(crop) is False


def test_has_bet_banner_rejects_small_blue_logo():
    # A tiny blue accent (e.g., a suit pip on a deck pattern) must NOT
    # trigger the banner. 1% coverage is well below the 3% threshold.
    crop = _bet_banner_crop(blue_frac=0.01)
    assert card_mod.has_bet_banner(crop) is False


# --- active timer bar detector -----------------------------------------

# Each reference frame shows the yellow turn-timer bar under a single
# seat. The test verifies that across all 6 seat ROIs, exactly one
# fires positive — the basic invariant the live-capture pipeline
# relies on. Locks in the 0.25 threshold (lowered from 0.5 for live
# ROI-drift headroom): if a future tightening pushes a positive read
# below 0.25 the test catches it.
_TIMER_BAR_FRAMES = (
    "screenrecords/frames/debug_1777190042024.png",
    "screenrecords/frames/debug_1777190077543.png",
    "screenrecords/frames/debug_1777190110906.png",
    "screenrecords/frames/debug_1777190153322.png",
    "screenrecords/frames/debug_1777190192408.png",
    "screenrecords/frames/debug_1777190246300.png",
)


@pytest.mark.parametrize("frame_rel", _TIMER_BAR_FRAMES)
def test_has_active_timer_bar_one_seat_per_reference_frame(frame_rel):
    frame_path = REPO_ROOT / frame_rel
    if not frame_path.exists():
        pytest.skip(f"reference frame missing: {frame_rel}")
    img = cv2.imread(str(frame_path))
    assert img is not None, f"failed to read {frame_rel}"
    seats = roi_mod.seats(6)
    hits = [
        s.seat
        for s in seats
        if card_mod.has_active_timer_bar(s.timer_bar_left.crop(img))
    ]
    assert len(hits) == 1, (
        f"{frame_rel}: expected exactly 1 seat to register active "
        f"timer bar, got {hits}"
    )


# --- TOOL-012: timer bar read from a drift-tolerant band -----------------------------


def _band_with(rows_yellow: list[int], h: int = 10, w: int = 26) -> "np.ndarray":
    import numpy as np

    band = np.zeros((h, w, 3), dtype=np.uint8)
    band[:, :] = (180, 110, 40)  # blue felt (BGR)
    for r in rows_yellow:
        band[r, :] = (20, 200, 240)  # warm yellow (BGR)
    return band


@pytest.mark.parametrize("row", [0, 3, 6, 9])
def test_timer_bar_found_anywhere_in_the_band(row):
    assert card_mod.has_active_timer_bar(_band_with([row]))


def test_scattered_yellow_is_not_a_timer_bar():
    import numpy as np

    band = _band_with([])
    rng = np.random.default_rng(0)
    ys, xs = rng.integers(0, 10, 60), rng.integers(0, 26, 60)
    band[ys, xs] = (20, 200, 240)  # ~23% of pixels, no row mostly yellow
    assert not card_mod.has_active_timer_bar(band)
    assert not card_mod.has_active_timer_bar(_band_with([]))


def test_template_bootstrap_never_writes_into_the_package(rank_templates_bootstrapped):
    """TEST-032: the session's templates live in a temp folder; the shipped
    `plo5bp/ocr/templates/` is only ever read."""
    from pathlib import Path

    shipped = Path(card_mod.__file__).parent / "templates"
    assert Path(card_mod.TEMPLATES_DIR).resolve() != shipped.resolve()


# --- TOOL-014: precomputed template features, cached card reads --------------------


def test_precomputed_features_score_bit_identically():
    import numpy as np

    rng = np.random.default_rng(7)
    glyph = (rng.random((40, 28)) > 0.5).astype(np.uint8) * 255
    templates = {r: [(rng.random((44, 30)) > 0.5).astype(np.uint8) * 255 for _ in range(3)]
                 for r in range(4)}
    feats = card_mod._template_features(templates)
    g = card_mod._features(glyph)
    for rank, tmpls in templates.items():
        for t, tf in zip(tmpls, feats[rank]):
            assert card_mod._score_features(g, tf) == card_mod._shape_score(glyph, t)


def test_unchanged_card_crops_are_not_classified_again(monkeypatch):
    import numpy as np

    from plo5bp.ocr import extract as extract_mod

    calls: list[int] = []
    monkeypatch.setattr(card_mod, "classify_card", lambda crop: calls.append(1))
    img = np.zeros((1391, 1927, 3), dtype=np.uint8)
    cache: dict = {}
    extract_mod._classify_row(img, roi_mod.BOARD_A, cache, "board_a")
    extract_mod._classify_row(img, roi_mod.BOARD_A, cache, "board_a")
    assert len(calls) == len(roi_mod.BOARD_A)  # second pass: all cache hits
    img[:, :, 0] = 1  # pixels changed -> read again
    extract_mod._classify_row(img, roi_mod.BOARD_A, cache, "board_a")
    assert len(calls) == 2 * len(roi_mod.BOARD_A)


def test_hero_templates_are_harvested_upright(monkeypatch, tmp_path):
    """TOOL-041: the runtime de-rotates each fanned hole card before matching,
    so the labeler de-rotates it the same way before cutting the template."""
    import numpy as np

    from plo5bp.ocr import extract as extract_mod
    from plo5bp.ocr.tools import label_cards as labeler

    frame = tmp_path / "f.png"
    cv2.imwrite(str(frame), np.zeros((1391, 1927, 3), dtype=np.uint8))
    angles: list[float] = []
    real = extract_mod._rotate
    monkeypatch.setattr(extract_mod, "_rotate", lambda c, d: (angles.append(d), real(c, d))[1])
    labeler.build_templates(frame, "hero_hole", list("AKQJT"), out_dir=tmp_path / "t")
    assert angles == [r[0] for r in extract_mod._HERO_HOLE_ROTATIONS]
