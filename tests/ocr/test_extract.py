"""End-to-end extractor tests against labeled fixtures.

Strict: output structure (dataclass shapes), button_seat, pot_total within 5%.
Lenient: per-card exactness and per-seat stacks (baseline thresholds only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from plo5bp.ocr import cards as card_mod
from plo5bp.ocr import extract as extract_mod
from plo5bp.ocr import rois as roi_mod
from plo5bp.ocr import text as text_mod
from plo5bp.ocr.extract import _read_seat, extract_frame_state
from plo5bp.ocr.types import Card, FrameState

REPO_ROOT = Path(__file__).resolve().parents[2]

MIN_CARD_ACCURACY = 0.45


def _load_fixture(fx):
    frame_path = REPO_ROOT / fx["frame"]
    if not frame_path.exists():
        pytest.skip(f"fixture frame missing: {frame_path}")
    img = cv2.imread(str(frame_path))
    if img is None:
        pytest.skip(f"could not read {frame_path}")
    expected = FrameState.from_dict(fx["state"])
    return img, expected


def test_state_has_expected_shape(labels):
    """Pipeline always returns a well-formed FrameState, regardless of classification accuracy."""
    for fx in labels:
        img, expected = _load_fixture(fx)
        got = extract_frame_state(img, num_seats=fx["num_seats"])
        assert len(got.board_a) == 5
        assert len(got.board_b) == 5
        assert len(got.hero_hole) == 5
        assert len(got.seats) == fx["num_seats"]
        for c in got.board_a + got.board_b + got.hero_hole:
            assert c is None or isinstance(c, Card)


def test_cards_accuracy(labels, rank_templates_bootstrapped):
    if not rank_templates_bootstrapped:
        pytest.skip("not enough rank templates bootstrapped")
    total = 0
    correct = 0
    mismatches: list[str] = []
    for fx in labels:
        img, expected = _load_fixture(fx)
        got = extract_frame_state(img, num_seats=fx["num_seats"])
        for field in ("board_a", "board_b", "hero_hole"):
            exp_row = getattr(expected, field)
            got_row = getattr(got, field)
            for i, (g, e) in enumerate(zip(got_row, exp_row)):
                total += 1
                if g == e:
                    correct += 1
                else:
                    mismatches.append(f"{fx['frame']} {field}[{i}]: got {g} expected {e}")
    acc = correct / max(total, 1)
    print(f"\ncard-level accuracy: {correct}/{total} = {acc:.2%}")
    for m in mismatches[:15]:
        print(" ", m)
    assert acc >= MIN_CARD_ACCURACY, f"card accuracy {acc:.2%} < {MIN_CARD_ACCURACY:.0%}"


def test_button_seat_match(labels):
    wrong: list[str] = []
    for fx in labels:
        img, expected = _load_fixture(fx)
        got = extract_frame_state(img, num_seats=fx["num_seats"])
        if got.button_seat != expected.button_seat:
            wrong.append(
                f"{fx['frame']}: got button_seat={got.button_seat} expected {expected.button_seat}"
            )
    assert not wrong, "button_seat mismatches:\n" + "\n".join(wrong)


def test_pot_total_close(labels):
    """Tesseract can misread a digit; accept within 10% or exact match.

    Skips if the Tesseract binary isn't installed on the system (text.py
    returns None for every read in that case).
    """
    pytest.importorskip("pytesseract")
    any_read = False
    checked: list[tuple[str, int, int]] = []
    for fx in labels:
        img, expected = _load_fixture(fx)
        got = extract_frame_state(img, num_seats=fx["num_seats"])
        exp = expected.pot_total_chips
        if exp is None:
            continue
        if got.pot_total_chips is not None:
            any_read = True
            checked.append((fx["frame"], got.pot_total_chips, exp))
    if not any_read:
        pytest.skip("Tesseract binary not installed / no pots readable")
    for frame, g, e in checked:
        rel = abs(g - e) / max(e, 1)
        assert rel < 0.10, f"{frame}: pot_total got {g} expected ~{e}"


# --- multi-signal in_hand detection (Fix 1) -----------------------------

def _fake_seat_rois(seat: int = 1) -> roi_mod.SeatROIs:
    """Build a SeatROIs with the real geometry so `.crop` works on a
    fake image. We monkeypatch the detectors so the actual pixel values
    don't matter."""
    return roi_mod.seats(6)[seat]


def _blank_img() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _apply_stub_detectors(
    monkeypatch: pytest.MonkeyPatch,
    *,
    has_back: bool,
    has_banner: bool,
    stack: int | None = 34000,
    commit: int | None = None,
) -> None:
    monkeypatch.setattr(card_mod, "has_cards_back", lambda _: has_back)
    monkeypatch.setattr(card_mod, "has_bet_banner", lambda _: has_banner)
    monkeypatch.setattr(text_mod, "read_chip_amount", lambda _: stack)
    monkeypatch.setattr(text_mod, "read_seat_commit", lambda _: commit)
    # Also patch the detectors through the module alias `extract_mod`
    # uses (Python resolves to the same object, but be explicit).
    monkeypatch.setattr(extract_mod.card_mod, "has_cards_back", lambda _: has_back)
    monkeypatch.setattr(extract_mod.card_mod, "has_bet_banner", lambda _: has_banner)
    monkeypatch.setattr(extract_mod.text_mod, "read_chip_amount", lambda _: stack)
    monkeypatch.setattr(extract_mod.text_mod, "read_seat_commit", lambda _: commit)


def test_read_seat_in_hand_via_cards_back_only(monkeypatch):
    _apply_stub_detectors(monkeypatch, has_back=True, has_banner=False)
    obs = _read_seat(_blank_img(), _fake_seat_rois(1), hero_hole=(None,) * 5)
    assert obs.folded is False
    assert obs.bet_banner is False


def test_read_seat_in_hand_via_banner_only(monkeypatch):
    # Card-back pattern defeated by banner — banner alone keeps seat in.
    _apply_stub_detectors(monkeypatch, has_back=False, has_banner=True)
    obs = _read_seat(_blank_img(), _fake_seat_rois(1), hero_hole=(None,) * 5)
    assert obs.folded is False
    assert obs.bet_banner is True


def test_hero_banner_suppressed_when_hole_cards_face_up(monkeypatch):
    # Anti-collusion: hero's cards stay face-up after the first flop
    # turn. Their cards_back ROI then overlays the actual cards;
    # has_bet_banner false-triggers on card-edge blue + felt
    # bleed-through and pins bet_banner=True for the rest of the hand.
    # The structural impossibility (banner only renders OVER card-backs
    # during chip-settle) means we suppress the false positive at the
    # _build_seat_obs boundary for seat 0 whenever any hole card is
    # visible. The override prevents downstream walk false positives
    # (e.g. _any_remaining_delta misreads) without altering the
    # has_bet_banner detector.
    _apply_stub_detectors(monkeypatch, has_back=False, has_banner=True)
    hole = (Card(rank=10, suit=0), None, None, None, None)
    obs = _read_seat(_blank_img(), _fake_seat_rois(0), hero_hole=hole)
    assert obs.bet_banner is False
    # Hole-card visibility alone keeps the seat in_hand; banner is
    # not the only signal.
    assert obs.folded is False


def test_hero_banner_passes_through_when_hole_cards_hidden(monkeypatch):
    # Pre-anti-collusion-trigger (hand start, hero hasn't acted on
    # the flop yet): hero_hole is all-None, so the banner-suppression
    # override stays inert and the detector value passes through.
    _apply_stub_detectors(monkeypatch, has_back=False, has_banner=True)
    obs = _read_seat(_blank_img(), _fake_seat_rois(0), hero_hole=(None,) * 5)
    assert obs.bet_banner is True
    assert obs.folded is False


def test_read_seat_in_hand_via_committed_only(monkeypatch):
    # No cards_back, no banner, but a positive committed_chips read.
    _apply_stub_detectors(
        monkeypatch, has_back=False, has_banner=False, commit=18000
    )
    obs = _read_seat(_blank_img(), _fake_seat_rois(1), hero_hole=(None,) * 5)
    assert obs.folded is False
    assert obs.committed_chips == 18000


def test_read_seat_folded_when_all_signals_absent(monkeypatch):
    _apply_stub_detectors(
        monkeypatch, has_back=False, has_banner=False, commit=None
    )
    obs = _read_seat(_blank_img(), _fake_seat_rois(1), hero_hole=(None,) * 5)
    assert obs.folded is True
    assert obs.bet_banner is False


def test_read_seat_committed_passes_through_even_when_in_hand_signals_fail(monkeypatch):
    # Critical invariant: committed read must NOT be clamped to None
    # just because cards_back/banner failed. Dropping the clamp is what
    # lets the reconstructor see a bet amount when OCR caught the chip
    # oval but missed both visual signals.
    _apply_stub_detectors(
        monkeypatch, has_back=False, has_banner=False, commit=9000
    )
    obs = _read_seat(_blank_img(), _fake_seat_rois(1), hero_hole=(None,) * 5)
    assert obs.committed_chips == 9000
    assert obs.folded is False  # has_commit signal kicked in


# --- Fix S: Tesseract parallelization must be byte-identical to serial ---


def _pick_fixture_frame(labels):
    """Return the first labeled fixture whose frame file actually exists
    on disk, or skip if none are present."""
    for fx in labels:
        frame_path = REPO_ROOT / fx["frame"]
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            if img is not None:
                return img, fx
    pytest.skip("no labeled fixture frames available on disk")


def test_extract_frame_state_is_stable_across_repeated_calls(labels):
    """A thread-pool race (shared mutable state, non-deterministic
    ordering) would surface as non-deterministic output across repeated
    calls on the same frame. Five iterations is enough to catch any
    realistic race."""
    pytest.importorskip("pytesseract")
    img, fx = _pick_fixture_frame(labels)
    results = [
        extract_frame_state(img, num_seats=fx["num_seats"]) for _ in range(5)
    ]
    for fs in results[1:]:
        assert fs == results[0]


def test_extract_frame_state_matches_serial_mode(monkeypatch, labels):
    """With worker count forced to 1 (serial behavior via the pool) vs.
    8 (full parallelism), output must be byte-identical — the
    parallelization layer must not influence what gets read, only
    how fast."""
    pytest.importorskip("pytesseract")
    img, fx = _pick_fixture_frame(labels)

    monkeypatch.setenv("PLO5BP_OCR_WORKERS", "1")
    monkeypatch.setattr(extract_mod, "_OCR_POOL", None)
    serial = extract_frame_state(img, num_seats=fx["num_seats"])

    monkeypatch.setenv("PLO5BP_OCR_WORKERS", "8")
    monkeypatch.setattr(extract_mod, "_OCR_POOL", None)
    parallel = extract_frame_state(img, num_seats=fx["num_seats"])

    assert serial == parallel
