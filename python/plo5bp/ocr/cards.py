"""Card classifier.

Split design:
  suit -> HSV mean of the card body, nearest centroid in ClubGG 4-color palette
          (blue=diamonds, green=clubs, red=hearts, black=spades)
  rank -> normalized cross-correlation (NCC) against 13 rank templates

Templates are expected at `plo5bp/ocr/templates/rank_<R>.png` as small
grayscale PNGs. They are produced by `plo5bp.ocr.tools.label_cards`.
When templates are missing, rank classification returns None and callers
should treat the card as unknown.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from plo5bp.ocr.types import Card

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Hue windows for ClubGG 4-color suits (OpenCV H in 0..179).
# Red wraps, so hearts are matched by a union of two ranges.
_HUE_WINDOWS: dict[int, tuple[tuple[int, int], ...]] = {
    # clubs = green
    0: ((40, 80),),
    # diamonds = blue
    1: ((95, 130),),
    # hearts = red (wraps around 0/180)
    2: ((0, 15), (165, 180)),
}


def classify_suit(card_bgr: np.ndarray) -> int | None:
    """Return suit 0..3 from a cropped card BGR image, or None if low confidence.

    Sample the rank-corner region (top-left, around the glyph). In fanned
    hand layouts only the rank corner is guaranteed to be the current
    card; the body area bleeds into the next card. Excluding the rank
    glyph itself (white pixels), the remaining pixels give the card's
    body color.
    """
    if card_bgr.size == 0:
        return None
    h, w = card_bgr.shape[:2]
    sample = card_bgr[int(h * 0.02) : int(h * 0.45), int(w * 0.05) : int(w * 0.95)]
    if sample.size == 0:
        return None
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Exclude the rank glyph (white = high V, low S) from the sample; what's
    # left is the card background which carries the suit color.
    not_white = ~((s_ch < 60) & (v_ch > 180))
    body_mask = ((s_ch > 80) | (v_ch < 50)) & not_white
    n_body = int(body_mask.sum())
    if n_body < 30:
        return None

    # Spades: dominance of dark pixels (low S, low V).
    spade_mask = (s_ch < 60) & (v_ch < 60)
    if spade_mask.sum() / max(n_body, 1) > 0.40:
        return 3

    scores: dict[int, float] = {}
    for suit, windows in _HUE_WINDOWS.items():
        in_window = np.zeros_like(h_ch, dtype=bool)
        for lo, hi in windows:
            in_window |= (h_ch >= lo) & (h_ch <= hi)
        scores[suit] = float((in_window & body_mask).sum()) / n_body

    best_suit = max(scores, key=scores.get)
    best_score = scores[best_suit]
    if best_score < 0.30:
        return None
    return best_suit


_RANK_TEMPLATES_CACHE: dict[int, list[np.ndarray]] | None = None


def _load_templates() -> dict[int, list[np.ndarray]]:
    """Return list of templates per rank. Filenames: rank_<R>.png or rank_<R>_<tag>.png."""
    global _RANK_TEMPLATES_CACHE
    if _RANK_TEMPLATES_CACHE is not None:
        return _RANK_TEMPLATES_CACHE
    out: dict[int, list[np.ndarray]] = {}
    if TEMPLATES_DIR.exists():
        for p in TEMPLATES_DIR.glob("rank_*.png"):
            parts = p.stem.split("_")
            try:
                rank = int(parts[1])
            except (IndexError, ValueError):
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            out.setdefault(rank, []).append(img)
    _RANK_TEMPLATES_CACHE = out
    return out


_RANK_CROP_Y1 = 0.55  # default top-corner crop height (fraction of card).
_RANK_CROP_Y1_TALL = 0.72  # adaptive retry when the digit is bottom-clipped.


def _extract_rank_glyph(card_bgr: np.ndarray, y1_frac: float):
    """Binarize + isolate the top rank glyph from `card_bgr[:y1_frac*h]`.

    Returns ``(glyph_mask, clipped)`` or ``None``. `clipped` is True when the
    extracted glyph touches the bottom edge of the crop (i.e. the digit likely
    extends below the crop and was cut off).
    """
    h, w = card_bgr.shape[:2]
    # Crop generously; CC isolation handles the suit pip below the digit.
    y0, y1 = 0, int(h * y1_frac)
    x0, x1 = 0, int(w * 0.95)
    corner = card_bgr[y0:y1, x0:x1]
    if corner.size == 0:
        return None
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    # 150 catches antialiased edges on K/J but rejects felt-table reflections.
    _, binm = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    n_white = int(binm.sum() / 255)
    if n_white < 40 or n_white > corner.size // 2:
        return None

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    # Drop tiny noise; 5 px is below any real glyph stroke.
    comps = [
        (i, int(stats[i, 1]), int(stats[i, 1] + stats[i, 3]), int(stats[i, 3]))
        for i in range(1, n_lbl)
        if int(stats[i, 4]) >= 5
    ]
    if not comps:
        return None
    comps.sort(key=lambda c: c[1])
    selected = {comps[0][0]}
    cluster_bottom = comps[0][2]
    cluster_h = comps[0][3]
    # Greedy merge: components whose top-y is within 4 px of the cluster's
    # current bottom belong to the same glyph (e.g. "1" and "0" of "10").
    # The height-ratio guard separates real digit pairs (similar heights,
    # ~0.9 ratio) from a tall rank glyph absorbing a short fragment like
    # the bottom curl of J or a serif (~0.18 ratio). Without it, a wider
    # slot-0 ROI causes J to merge with its curl-fragment and miss the
    # 0.55 template floor.
    for cid, top_y, bot_y, h_comp in comps[1:]:
        if top_y - cluster_bottom <= 4 and h_comp >= 0.5 * cluster_h:
            selected.add(cid)
            cluster_bottom = max(cluster_bottom, bot_y)
            cluster_h = max(cluster_h, h_comp)
        else:
            break

    mask = np.zeros_like(binm)
    for i in selected:
        mask[labels == i] = 255
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    y_lo, y_hi = int(ys.min()), int(ys.max()) + 1
    x_lo, x_hi = int(xs.min()), int(xs.max()) + 1
    if (y_hi - y_lo) < 12 or (x_hi - x_lo) < 6:
        return None
    clipped = y_hi >= corner.shape[0]
    return mask[y_lo:y_hi, x_lo:x_hi], clipped


def _preprocess_rank_glyph(card_bgr: np.ndarray) -> np.ndarray | None:
    """Extract and binarize the top rank glyph from a card crop.

    Crops the top corner generously and uses connected-components to
    keep only the topmost cluster — the rank digit/letter. This drops
    the suit pip below the digit (ClubGG hero hole crops have aspect
    ~1.82 so the rank corner crop also catches the pip), making the
    extracted glyph suit-color-independent.

    Adaptive bottom-crop: the fanned bottom hole card (slot 0) sits lower in
    its ROI, so the fixed 0.55 crop clips the digit's bottom — an "8" loses its
    lower loop and reads as "9". When the extracted glyph touches the crop's
    bottom edge we re-extract from a taller crop; the suit pip below stays
    rejected by the height-ratio merge guard, so non-clipped glyphs (other
    slots, board cards) are unchanged.
    """
    if card_bgr.size == 0:
        return None
    res = _extract_rank_glyph(card_bgr, _RANK_CROP_Y1)
    if res is None:
        return None
    glyph, clipped = res
    if clipped:
        taller = _extract_rank_glyph(card_bgr, _RANK_CROP_Y1_TALL)
        if taller is not None:
            return taller[0]
    return glyph


_CANON_GLYPH_SIZE = (32, 48)  # (w, h)


def _canon(bin_img: np.ndarray) -> np.ndarray:
    h, w = bin_img.shape[:2]
    tw, th = _CANON_GLYPH_SIZE
    # INTER_AREA is only correct for downsampling; LINEAR handles both ways.
    interp = cv2.INTER_AREA if (h >= th and w >= tw) else cv2.INTER_LINEAR
    resized = cv2.resize(bin_img, _CANON_GLYPH_SIZE, interpolation=interp)
    # Re-binarize after interpolation so IoU/cosine score on crisp shapes.
    _, binm = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return binm


def _shape_score(glyph: np.ndarray, tmpl: np.ndarray) -> float:
    """Score glyph vs template on canonical-size binary overlap.

    Blends IoU (overall agreement) with a penalty on filled-area disagreement,
    which helps separate visually similar ranks like 3/5/6/8.
    """
    g = _canon(glyph).astype(np.float32) / 255.0
    t = _canon(tmpl).astype(np.float32) / 255.0
    inter = float(np.minimum(g, t).sum())
    union = float(np.maximum(g, t).sum())
    iou = inter / (union + 1e-9)
    # Cosine on centered bits
    gf = g.ravel() - g.mean()
    tf = t.ravel() - t.mean()
    gn = np.linalg.norm(gf)
    tn = np.linalg.norm(tf)
    cos = float(gf @ tf / (gn * tn + 1e-9))
    return 0.5 * iou + 0.5 * cos


def classify_rank(card_bgr: np.ndarray) -> tuple[int | None, float]:
    """Return (rank, confidence). Rank None if confidence below threshold.

    For each rank we may have multiple templates (captured from different
    frames); the rank's score is the MAX over its templates. Then we take
    the best rank and require a margin over second-best to avoid
    misclassifying visually ambiguous ranks (e.g. 3 vs 5).
    """
    templates = _load_templates()
    if not templates:
        return None, 0.0
    glyph = _preprocess_rank_glyph(card_bgr)
    if glyph is None:
        return None, 0.0
    per_rank: list[tuple[int, float]] = []
    for rank, tmpls in templates.items():
        best = max(_shape_score(glyph, t) for t in tmpls)
        per_rank.append((rank, best))
    per_rank.sort(key=lambda kv: -kv[1])
    best_rank, best_score = per_rank[0]
    runner_up = per_rank[1][1] if len(per_rank) > 1 else 0.0
    # Floor 0.55: real ClubGG cards score 0.59+ (lowest observed across 4
    # in-hand debug frames). The hero avatar art at idle hits the H4 ROI
    # and scores ~0.483 against the rank-J template — a stable false
    # positive that 0.40 lets through. 0.55 cleanly separates them.
    # (The fanned edge hole cards used to dip below this floor when tilted;
    # that is now handled at the source by per-slot de-rotation in
    # extract._classify_hero_hole, so the strict floor stands.)
    if best_score < 0.55:
        return None, best_score
    if best_score - runner_up < 0.03 and best_score < 0.65:
        return None, best_score
    return best_rank, best_score


def is_card_present(card_bgr: np.ndarray) -> bool:
    """Cheap pre-filter: a revealed card has a saturated body with a bright rank glyph.

    We only reject crops that are unambiguously NOT cards (table felt, empty slots).
    Borderline cases are left to `classify_card`, which requires BOTH rank-NCC
    and suit-HSV to succeed before returning a Card.
    """
    if card_bgr.size == 0:
        return False
    h, w = card_bgr.shape[:2]
    body = card_bgr[int(h * 0.35) :, int(w * 0.05) : int(w * 0.95)]
    if body.size == 0:
        return False
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # Card body is either high-saturation (blue/green/red) or low-saturation-low-V (black spade).
    has_colored_body = float((sat > 120).mean()) > 0.25
    has_black_body = float(((sat < 40) & (val < 50)).mean()) > 0.25
    return has_colored_body or has_black_body


def classify_card(card_bgr: np.ndarray) -> Card | None:
    """Classify a single card crop. Returns None for unrevealed / low-confidence."""
    if not is_card_present(card_bgr):
        return None
    suit = classify_suit(card_bgr)
    rank, _conf = classify_rank(card_bgr)
    if suit is None or rank is None:
        return None
    return Card(rank=rank, suit=suit)


def has_cards_back(bgr: np.ndarray) -> bool:
    """True when the ROI contains the silver-diamond card-back pattern.

    ClubGG card backs render as desaturated mid-tone gray (S<30, V∈[150,220])
    across most of the face. Avatars are either saturated (skin/flags/art) or
    fall outside that V window (dark panda fur / bright teal bg). Measured on
    live debug frames, in-hand seats score 0.21-0.23; empty seats score <0.07.
    """
    if bgr.size == 0:
        return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat < 30) & (val >= 150) & (val <= 220)
    return float(mask.sum()) / mask.size > 0.15


def has_active_timer_bar(bgr: np.ndarray) -> bool:
    """True when ClubGG's yellow turn-timer bar is rendered in the ROI.

    The bar is the most reliable on-screen signal for "this seat is the
    current actor". It's a yellow→green gradient that depletes
    right-to-left; the leftmost ~10-15% (which is what the
    `timer_bar_left` ROI captures, see rois.py) stays solidly yellow
    until the timer is nearly spent. We mask the yellow band and
    threshold on coverage.

    HSV bounds: hue 15-35 covers ClubGG's warm-yellow / orange band
    (measured H_median ≈ 23 across all 6 seats in the reference
    frames). The blue table felt at the same ROI when no bar is
    rendered scores H_median ≈ 100-106 — well outside the band.
    Active reads in the reference frames score ratio = 1.0; inactive
    reads score 0.0. 0.25 leaves headroom for ROI drift and
    anti-aliasing on the 26×2-pixel non-hero ROI; the gap to the 1.0
    positive / 0.0 negative reference is wide enough that the lower
    threshold doesn't compress the negative margin.
    """
    if bgr.size == 0:
        return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([15, 140, 140], dtype=np.uint8),
        np.array([35, 255, 255], dtype=np.uint8),
    )
    return float(mask.sum() / 255) / mask.size > 0.25


def has_bet_banner(bgr: np.ndarray) -> bool:
    """True when a blue "Bet" banner is overlaid on the seat's card-backs.

    ClubGG renders a solid bright-blue rectangle with white "Bet" text over
    the actor's cards while their chips are settling. The banner covers
    ~30-40% of the cards_back ROI; a pure HSV-mask pixel count is enough.

    Hue ~100-115 (OpenCV scale, 180-max), Sat>150, Val>130. The blue back
    pattern itself is desaturated (S<30) and won't trigger this band.
    Ratio > 0.03 tolerates slight ROI mis-alignment; measured on debug
    frames the active banner covers 0.15+ of the crop.
    """
    if bgr.size == 0:
        return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([100, 150, 130], dtype=np.uint8),
        np.array([115, 255, 255], dtype=np.uint8),
    )
    return float(mask.sum() / 255) / mask.size > 0.03
