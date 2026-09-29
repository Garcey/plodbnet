"""Text-OCR helpers (chip amounts, dealer-button marker).

pytesseract is a heavy optional dep; we import it lazily so that import of
`plo5bp.ocr` doesn't fail when OCR extras aren't installed (matters for tests
that only exercise `types`/`rois`).

OpenCV gets the same treatment (review 2026-09-20 I12): `cv2` is imported
inside the pixel helpers that need it, never at module top, so the pure
string parser (`_parse_chip_text`) and its tests run without the `[ocr]`
extras. `import cv2` inside a function is a `sys.modules` dict hit after the
first call -- negligible next to a Tesseract invocation.

(TOOL-033) Amounts are read by the in-process digit-template reader
(`digits.py`) first, whenever harvested templates exist
(`templates/digits.npz`, built by `python -m plo5bp.ocr.tools.harvest_digits`);
it answers only when every glyph matches confidently, and Tesseract reads the
rest — and everything, while there are no templates. `PLO5BP_OCR_DIGITS=0`
turns the template reader off. `reader_stats()` counts who read what.
"""

from __future__ import annotations

import os
import re
import shutil
import threading

import numpy as np

# One amount token: a run of digits/separators holding at least one digit.
# Leading separators are allowed so ``.50`` keeps its decimal point (the old
# ``[0-9][0-9,.]*`` started matching at the ``5`` and read $50).
_DIGIT_RE = re.compile(r"[0-9.,]*[0-9][0-9.,]*")

_TESSERACT_PATH_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
)

_tesseract_configured = False


def _get_tesseract():
    global _tesseract_configured
    try:
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is required for text OCR; install with `pip install -e .[ocr]` "
            "and ensure the Tesseract binary is on PATH"
        ) from exc
    if not _tesseract_configured:
        # Honor an explicit override first (tests, non-standard installs).
        env_path = os.environ.get("TESSERACT_CMD")
        if env_path and os.path.isfile(env_path):
            pytesseract.pytesseract.tesseract_cmd = env_path
        elif shutil.which("tesseract") is None:
            for p in _TESSERACT_PATH_CANDIDATES:
                if p and os.path.isfile(p):
                    pytesseract.pytesseract.tesseract_cmd = p
                    break
        _tesseract_configured = True
    return pytesseract


def _preprocess_chip_crop(bgr: np.ndarray) -> np.ndarray:
    """Upscale, gray-out, threshold to prep a chip-amount ROI for tesseract."""
    if bgr.size == 0:
        return bgr
    import cv2

    # Upscale 3x for small digits.
    scaled = cv2.resize(bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    # Chip numbers are bright-cyan on dark bg -> threshold to keep bright pixels.
    _, binm = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    # Tesseract prefers dark text on white -> invert so digits are black.
    return cv2.bitwise_not(binm)


_TESSERACT_CFG = "--psm 8 -c tessedit_char_whitelist=0123456789.,$"


def _parse_chip_text(raw: str) -> int | None:
    """Parse a Tesseract chip-amount string to integer cents.

    ClubGG renders stacks/pots in dollars with at most two decimals
    (e.g. ``580.03`` or ``1,755.59``). Tesseract often confuses ``,``
    and ``.`` at small scales — most notably reading the thousands
    separator in ``1,090`` as a decimal point and emitting ``1.090``.
    Naive ``dollars.cents`` parsing then truncates ``090`` to ``09``
    and the stack collapses to $1.09. Heuristic: when the post-decimal
    fragment has three or more digits, the ``.`` is a misread ``,`` —
    reinterpret the token as a comma-grouped integer dollar amount.

    One rule covers every case (review 2026-09-20 I11): after dropping
    TRAILING separators, the LAST separator is the decimal point iff
    exactly 1-2 digits follow it; every other separator is grouping.

    * Trailing ``.``/``,`` is Tesseract punctuation noise, not a decimal
      point. The old multi-dot collapse kept the LAST dot, so
      ``"450.5."`` became ``"4505."`` = $4505 and ``"70.05."`` = $7005 —
      a silent 10-100x stack (the intermittent "decimal-drop").
    * 3+ digits after the last separator = grouping (the ``1.090`` case
      above), unchanged.
    * ``,`` gets the same treatment as ``.``: a comma followed by 1-2
      final digits cannot be a thousands separator, so ``"450,5"`` is
      $450.50, not $4505.
    * ``".50"`` is 50 cents, not $50.

    Returns None when the text holds no digit at all; a genuine ``"0"``
    stays 0 — callers rely on the None-vs-0 distinction.
    """
    match = _DIGIT_RE.search(raw.replace(" ", ""))
    if not match:
        return None
    # The token holds a digit, so stripping trailing separators cannot
    # empty it.
    token = match.group(0).rstrip(".,")
    last = max(token.rfind("."), token.rfind(","))
    cents = ""
    if last >= 0 and 1 <= len(token) - last - 1 <= 2:
        token, cents = token[:last], token[last + 1 :]
    dollars = token.replace(".", "").replace(",", "")
    return int(dollars or "0") * 100 + int((cents + "00")[:2])


_UNSET = object()
_DIGIT_READER: object = _UNSET
_DIGIT_LOCK = threading.Lock()
_STATS = {"template": 0, "tesseract": 0, "unread": 0}


def _digit_reader():
    """(TOOL-033) The digit-template reader, or None — no complete template
    file yet, or ``PLO5BP_OCR_DIGITS=0``. Loaded once (the OCR pool calls this
    from several threads)."""
    global _DIGIT_READER
    if _DIGIT_READER is _UNSET:
        with _DIGIT_LOCK:
            if _DIGIT_READER is _UNSET:
                reader = None
                if os.environ.get("PLO5BP_OCR_DIGITS", "1").strip() != "0":
                    try:
                        from plo5bp.ocr.digits import DigitReader

                        reader = DigitReader.load()
                    except Exception:  # a damaged template file must not break OCR
                        reader = None
                _DIGIT_READER = reader
    return _DIGIT_READER


def _count(key: str) -> None:
    with _DIGIT_LOCK:
        _STATS[key] += 1


def reader_stats() -> dict[str, int]:
    """(TOOL-033) How many amounts each reader answered since start-up:
    ``template`` (in-process, exact), ``tesseract`` (fallback), ``unread``."""
    with _DIGIT_LOCK:
        return dict(_STATS)


def _ocr_chip_token(prep: np.ndarray) -> int | None:
    """Read a preprocessed binary image (black text on white) to chips.

    The digit-template reader answers first when templates exist; Tesseract
    reads what it is unsure of (TOOL-033). Returns None on empty input,
    missing tesseract, or unparseable output. Shared parse path: both stack/pot
    crops and seat-commit crops land here after their respective preprocessors.
    """
    if prep.size == 0:
        return None
    reader = _digit_reader()
    if reader is not None:
        text = reader.read_text(np.asarray(prep) < 128)
        value = _parse_chip_text(text) if text is not None else None
        if value is not None:
            _count("template")
            return value
    value = _tesseract_chip_token(prep)
    _count("tesseract" if value is not None else "unread")
    return value


def _tesseract_chip_token(prep: np.ndarray) -> int | None:
    try:
        tesseract = _get_tesseract()
    except RuntimeError:
        return None
    try:
        raw = tesseract.image_to_string(prep, config=_TESSERACT_CFG)
    except tesseract.TesseractNotFoundError:
        return None
    except Exception:
        return None
    return _parse_chip_text(raw)


def _cyan_text_bbox(bgr: np.ndarray, pad: int = 3) -> tuple[int, int, int, int] | None:
    """Bbox of bright-cyan stack-text pixels; None when the mask is empty.

    Tesseract hallucinates phantom leading/trailing digits when the crop carries
    unrelated bright pixels from neighbors (avatars, card edges). Shrinking the
    ROI to just the cyan digits before thresholding reliably eliminates those.
    """
    if bgr.size == 0 or bgr.ndim != 3:
        return None
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([80, 60, 140]), np.array([110, 255, 255]))
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    h, w = bgr.shape[:2]
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w, int(xs.max()) + pad),
        min(h, int(ys.max()) + pad),
    )


def read_chip_amount(bgr: np.ndarray) -> int | None:
    """Parse a chip-amount ROI (e.g. '580.03' or '1,755.59') to an int chips value.

    ClubGG displays stacks in dollars with two decimals; we treat 1 cent = 1 chip
    for training parity (100 chips = $1 = 1 bb at 3bb ante game). Returns None on
    empty / unparseable crops, or when the Tesseract binary is unavailable.
    """
    if bgr.size == 0:
        return None
    return _ocr_chip_token(prep_chip_crop(bgr))


def prep_chip_crop(bgr: np.ndarray) -> np.ndarray:
    """A stack / pot crop as the readers see it: tightened to the cyan digits,
    then `_preprocess_chip_crop` (the digit harvester uses this too, so the
    templates are cut from exactly what gets read)."""
    bbox = _cyan_text_bbox(bgr)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        tight = bgr[y0:y1, x0:x1]
        if tight.size > 0:
            bgr = tight
    return _preprocess_chip_crop(bgr)


def prep_commit_crop(bgr: np.ndarray) -> np.ndarray:
    """A seat-commit badge as its first read sees it (see `read_seat_commit`)."""
    return _preprocess_seat_commit_crop(bgr)


def _preprocess_seat_commit_crop(bgr: np.ndarray) -> np.ndarray:
    """Isolate the dark pill + white text on a seat-commit badge.

    Flame/frost decorations around decorated badges (e.g. seat-1 flame,
    seat-3 frost) are highly saturated; the digit pill itself is near-
    neutral. Zero saturated pixels before the bright-text threshold so
    halo noise doesn't survive to tesseract.
    """
    if bgr.size == 0:
        return bgr
    import cv2

    scaled = cv2.resize(bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
    neutral = hsv[:, :, 1] < 80
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    gray[~neutral] = 0
    _, binm = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(binm)


def read_seat_commit(bgr: np.ndarray) -> int | None:
    """Parse a seat's committed-this-street badge to an int chips value.

    Tries decoration-aware preprocessing first so flame/frost halos around
    some seats don't turn a persistent '0' into a false non-zero read.
    Falls back to the plain bright-text threshold when the decoration
    filter wipes too much of the pill; a successful fallback is strictly
    better than returning None (which would stall the action inferrer).
    """
    if bgr.size == 0:
        return None
    primary = _ocr_chip_token(_preprocess_seat_commit_crop(bgr))
    if primary is not None:
        return primary
    return _ocr_chip_token(_preprocess_chip_crop(bgr))


def _has_pot_chip_overlay(bgr: np.ndarray) -> bool:
    """Detect ClubGG's transient 'chips added' overlay on the pot banner.

    After a bet lands, ClubGG briefly renders a small white number above the
    pot total that overlaps the leading digits of the real amount. Any OCR
    pass on that state will misread the amount, so the caller should skip
    the frame and wait for the overlay to clear.
    """
    if bgr.size == 0 or bgr.shape[0] < 10:
        return False
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    top = hsv[: bgr.shape[0] // 5, :]
    white = (top[:, :, 1] < 50) & (top[:, :, 2] > 180)
    return float(white.sum()) / white.size > 0.03


def read_pot_amount(bgr: np.ndarray) -> int | None:
    """Read the pot-total banner, returning None while the chip-add overlay is up."""
    if _has_pot_chip_overlay(bgr):
        return None
    return read_chip_amount(bgr)

