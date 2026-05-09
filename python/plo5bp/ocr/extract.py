"""Stateless per-frame state extractor.

Public entry point: `extract_frame_state(img: np.ndarray) -> FrameState`
where `img` is a BGR numpy array (as returned by `cv2.imread`).

Phase 1 is intentionally stateless: no cross-frame diff, no event detection.
Fields that can't be read confidently come back as None / False.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from plo5bp.ocr import cards as card_mod
from plo5bp.ocr import rois as roi_mod
from plo5bp.ocr import text as text_mod
from plo5bp.ocr.types import Card, FrameState, SeatObs


_OCR_POOL: ThreadPoolExecutor | None = None


def _get_ocr_pool() -> ThreadPoolExecutor:
    """Lazy module-level worker pool for Tesseract calls.

    Each pytesseract call spawns the Tesseract binary as a subprocess and
    releases the GIL during the wait, so a small thread pool genuinely
    overlaps the ~30-50ms per-call Windows process startup across the
    13 reads a tick issues (6 stacks + 6 commits + 1 pot).

    Size knob: ``PLO5BP_OCR_WORKERS`` env var. Defaults to
    ``min(13, os.cpu_count() or 4)``. Setting it to 1 collapses the pool
    to serial execution — the live-regression escape hatch.
    """
    global _OCR_POOL
    if _OCR_POOL is None:
        try:
            workers = int(os.environ.get("PLO5BP_OCR_WORKERS", "0"))
        except ValueError:
            workers = 0
        if workers <= 0:
            workers = min(13, os.cpu_count() or 4)
        _OCR_POOL = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="plo5bp-ocr"
        )
    return _OCR_POOL


def _safe_future_result(fut: Future) -> int | None:
    """Drain a Tesseract future, folding any worker-side exception into
    ``None`` so callers see the same null-return semantics the serial
    ``text.py`` wrappers already guarantee."""
    try:
        return fut.result()
    except Exception:
        return None


def _classify_row(img: np.ndarray, rois: tuple[roi_mod.ROI, ...]) -> tuple[Card | None, ...]:
    out: list[Card | None] = []
    for roi in rois:
        crop = roi.crop(img)
        out.append(card_mod.classify_card(crop))
    return tuple(out)


# Slot 0's calibrated ROI (`HERO_HOLE[0]`, w=0.0260) was tuned for J at the
# fan offset. The 2-character "10" is wider and gets clipped at the right
# edge, so T at slot 0 fails the 0.55 template floor. This wider variant
# captures the full "10" digit pair; templates for J in the
# fused-into-one-component rendering still match better at the narrow ROI,
# so we keep narrow primary and only fall back to wide when narrow returns
# None for slot 0.
_HERO_HOLE_SLOT0_WIDE = roi_mod.ROI(
    x=roi_mod.HERO_HOLE[0].x,
    y=roi_mod.HERO_HOLE[0].y,
    w=roi_mod.HERO_HOLE[0].w + 0.0030,
    h=roi_mod.HERO_HOLE[0].h,
)


def _classify_hero_hole(img: np.ndarray) -> tuple[Card | None, ...]:
    out: list[Card | None] = []
    for i, roi in enumerate(roi_mod.HERO_HOLE):
        card = card_mod.classify_card(roi.crop(img))
        if card is None and i == 0:
            card = card_mod.classify_card(_HERO_HOLE_SLOT0_WIDE.crop(img))
        out.append(card)
    return tuple(out)


def _build_seat_obs(
    seat_rois: roi_mod.SeatROIs,
    hero_hole: tuple[Card | None, ...],
    stack: int | None,
    committed: int | None,
    cards_back_crop: np.ndarray,
    timer_bar_left_crop: np.ndarray,
) -> SeatObs:
    """Assemble a ``SeatObs`` from resolved Tesseract reads + card-back crop.

    Shared by both serial (`_read_seat`) and parallel (`extract_frame_state`)
    paths so the multi-signal in-hand rule lives in exactly one place.

    Multi-signal "in hand" detection. ``has_cards_back`` alone is fragile
    when the blue "Bet" banner or other overlays partially occlude the
    silver-diamond pattern, pushing the pixel ratio below threshold. We
    OR three independent signals so a seat is still classified in-hand
    whenever ANY of them fires:
      - card-back silver-diamond pattern visible, OR
      - blue "Bet" banner rendered over cards (actively betting), OR
      - a positive ``committed_chips`` read (they have live chips on the
        table this street).
    """
    has_back = card_mod.has_cards_back(cards_back_crop)
    bet_banner = card_mod.has_bet_banner(cards_back_crop)
    is_actor = card_mod.has_active_timer_bar(timer_bar_left_crop)
    has_commit = committed is not None and int(committed) > 0
    if seat_rois.seat == 0:
        # ClubGG's anti-collusion rule hides hero's hole cards postflop
        # until it's hero's turn. In that state `hero_hole` classifies as
        # all-None (no face-up cards) but the silver-diamond cards_back
        # pattern is still rendered in the same ROI convention as other
        # seats. Use the same multi-signal OR so hero doesn't silently
        # drop out of `hand_in_hand_mask` when they're in the hand but
        # not yet facing action.
        hero_hole_visible = any(c is not None for c in hero_hole)
        if hero_hole_visible:
            # Once hero's cards flip face-up (anti-collusion rule:
            # face-up from first flop turn through hand end), the
            # cards_back ROI overlays the actual cards. The blue HSV
            # band that has_bet_banner detects (hue 100-115, S>150,
            # V>130) catches card-edge blue + felt bleed-through and
            # pins bet_banner=True for the rest of the hand. The bet
            # banner is rendered ON the card-back overlay during the
            # chip-settle animation — structurally impossible while
            # cards are face-up — so suppress the false positive at
            # the source rather than propagate it through the walk.
            bet_banner = False
        in_hand = hero_hole_visible or has_back or bet_banner or has_commit or is_actor
    else:
        in_hand = has_back or bet_banner or has_commit or is_actor
    return SeatObs(
        seat=seat_rois.seat,
        stack_chips=stack,
        committed_chips=committed,
        folded=not in_hand,
        bet_banner=bet_banner,
        is_actor=is_actor,
    )


def _read_seat(
    img: np.ndarray,
    seat_rois: roi_mod.SeatROIs,
    hero_hole: tuple[Card | None, ...],
) -> SeatObs:
    """Serial per-seat reader retained for tests that monkeypatch
    ``text_mod.read_chip_amount`` / ``read_seat_commit`` directly."""
    stack = text_mod.read_chip_amount(seat_rois.stack_label.crop(img))
    committed = text_mod.read_seat_commit(seat_rois.committed_label.crop(img))
    cards_back_crop = seat_rois.cards_back.crop(img)
    timer_bar_crop = seat_rois.timer_bar_left.crop(img)
    return _build_seat_obs(
        seat_rois, hero_hole, stack, committed, cards_back_crop, timer_bar_crop
    )


def _detect_button(img: np.ndarray, seat_rois: tuple[roi_mod.SeatROIs, ...]) -> int | None:
    # Score each seat by the largest connected amber blob in its anchor.
    # Using the biggest component (not raw pixel fraction) means a thin
    # actor-countdown ring can't outscore the compact D disc when both
    # sit in the same HSV band.
    import cv2

    best_seat: int | None = None
    best_score = 0.0
    lower = np.array([15, 120, 120], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)
    for s in seat_rois:
        crop = s.button_anchor.crop(img)
        if crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        largest = max(
            (int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)),
            default=0,
        )
        score = largest / (mask.size + 1e-9)
        if score > best_score:
            best_score = score
            best_seat = s.seat
    if best_score < 0.04:
        return None
    return best_seat


def extract_frame_state(img: np.ndarray, num_seats: int = 6) -> FrameState:
    """Parse a single BGR frame into a `FrameState`.

    Parameters
    ----------
    img : np.ndarray
        BGR image (H, W, 3), as returned by `cv2.imread`.
    num_seats : int
        Seat count for layout selection. Phase 1 supports 6 only.

    Tesseract reads (``num_seats`` stack labels + ``num_seats`` committed
    ovals + 1 pot banner) run on a shared worker pool so their per-call
    subprocess startup overlaps. Pure-CV2 work (card classification,
    button detection) and the final multi-signal in-hand rule stay on
    the caller thread — the pool is a read pipeline, not a reorderer.
    """
    if img is None or img.ndim != 3:
        raise ValueError("extract_frame_state expects a BGR image (H, W, 3)")

    board_a = _classify_row(img, roi_mod.BOARD_A)
    board_b = _classify_row(img, roi_mod.BOARD_B)
    hero_hole = _classify_hero_hole(img)

    seat_rois = roi_mod.seats(num_seats)

    # Crop phase (serial, cheap numpy slices).
    stack_crops = [sr.stack_label.crop(img) for sr in seat_rois]
    commit_crops = [sr.committed_label.crop(img) for sr in seat_rois]
    cards_back_crops = [sr.cards_back.crop(img) for sr in seat_rois]
    timer_bar_crops = [sr.timer_bar_left.crop(img) for sr in seat_rois]
    pot_crop = roi_mod.POT_BANNER.crop(img)

    # Warm the Tesseract config on the main thread before we submit any
    # futures. This is the same lazy init the first worker would do —
    # front-loading it eliminates a benign-but-ugly TOCTOU race on
    # `text._tesseract_configured` without needing a lock.
    try:
        text_mod._get_tesseract()
    except RuntimeError:
        # pytesseract missing — the read_* wrappers will return None for
        # every crop (same behavior as today).
        pass

    # Dispatch phase.
    pool = _get_ocr_pool()
    stack_futures = [
        pool.submit(text_mod.read_chip_amount, crop) for crop in stack_crops
    ]
    commit_futures = [
        pool.submit(text_mod.read_seat_commit, crop) for crop in commit_crops
    ]
    pot_future = pool.submit(text_mod.read_pot_amount, pot_crop)

    # Collect phase. Indexing keeps per-seat assembly order-independent
    # of completion order.
    seats: list[SeatObs] = []
    for i, sr in enumerate(seat_rois):
        stack = _safe_future_result(stack_futures[i])
        committed = _safe_future_result(commit_futures[i])
        seats.append(
            _build_seat_obs(
                sr, hero_hole, stack, committed,
                cards_back_crops[i], timer_bar_crops[i],
            )
        )

    button_seat = _detect_button(img, seat_rois)
    pot_total = _safe_future_result(pot_future)

    return FrameState(
        board_a=board_a,
        board_b=board_b,
        hero_hole=hero_hole,
        button_seat=button_seat,
        pot_total_chips=pot_total,
        seats=tuple(seats),
    )
