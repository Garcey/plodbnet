"""ROI layout for ClubGG PLO5 double-board bomb-pot screenshots.

Coordinates are stored as fractions of the frame's width/height, calibrated
on a 1927x1391 ClubGG table window captured through WGC (`CALIBRATION_SIZE`).
Fractions do NOT make the layout aspect-invariant: a window of another aspect
ratio moves every ROI, and the card-glyph extractor uses absolute pixel floors.
`frame_geometry` says what a capture needs: same aspect (within 2%) and size
= use as is; same aspect, other size = rescale to the calibration size;
another aspect = refuse (TOOL-003).

Phase 1 hardcodes the 6-seat layout. Hero is always seat 0 (bottom-center).
Seats 1..5 wrap physical-clockwise (poker convention) from hero's right:
1=bottom-left, 2=top-left, 3=top-center, 4=top-right, 5=bottom-right.
This keeps increasing seat index aligned with the engine's
`(actor + 1) % n` advancement, so engine action flow matches
real-poker CW direction on screen.

Each `ROI` is a (x, y, w, h) fractional rectangle that converts to absolute
pixel coordinates via `ROI.abs(W, H) -> (x1, y1, x2, y2)`.
"""

from __future__ import annotations

from dataclasses import dataclass

#: (width, height) of the ClubGG table window the ROI table and the rank
#: templates were calibrated on.
CALIBRATION_SIZE = (1927, 1391)
#: Relative aspect / size mismatch tolerated before a capture is refused or
#: rescaled.
GEOMETRY_TOLERANCE = 0.02


def frame_geometry(width: int, height: int) -> tuple[str, str | None]:
    """What a ``width`` x ``height`` capture needs before extraction.

    Returns ``("ok", None)``, ``("rescale", note)`` (same aspect as the
    calibration, other size: resize it to `CALIBRATION_SIZE` so ROIs, glyph
    pixel floors and rank templates all see calibration-scale pixels) or
    ``("refuse", message)`` (another aspect ratio: every ROI would land on
    the wrong pixels, so nothing read from it can be trusted).
    """
    cw, ch = CALIBRATION_SIZE
    if width <= 0 or height <= 0:
        return "refuse", f"empty capture ({width}x{height})"
    aspect, cal = width / height, cw / ch
    if abs(aspect - cal) / cal > GEOMETRY_TOLERANCE:
        return "refuse", (
            f"the ClubGG window is {width}x{height} (aspect {aspect:.2f}); live "
            f"capture is calibrated for {cw}x{ch} (aspect {cal:.2f}) — resize "
            "the table window to that shape"
        )
    if abs(width - cw) / cw > GEOMETRY_TOLERANCE:
        return "rescale", f"capture {width}x{height} rescaled to {cw}x{ch}"
    return "ok", None


@dataclass(frozen=True)
class ROI:
    x: float
    y: float
    w: float
    h: float

    def abs(self, W: int, H: int) -> tuple[int, int, int, int]:
        x1 = int(round(self.x * W))
        y1 = int(round(self.y * H))
        x2 = int(round((self.x + self.w) * W))
        y2 = int(round((self.y + self.h) * H))
        return x1, y1, x2, y2

    def crop(self, img):
        import numpy as np

        H, W = img.shape[:2]
        x1, y1, x2, y2 = self.abs(W, H)
        x1 = max(0, min(W, x1))
        x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1))
        y2 = max(0, min(H, y2))
        return np.ascontiguousarray(img[y1:y2, x1:x2])


NUM_BOARD_CARDS = 5

_BOARD_LEFT = 0.3108
_BOARD_CARD_W = 0.0716
_BOARD_CARD_H = 0.0810
_BOARD_CARD_STEP = 0.0766
_BOARD_A_TOP = 0.3580
_BOARD_B_TOP = 0.4400
_HERO_CARDS_LEFT = 0.4178
_HERO_CARD_W = 0.0260
_HERO_CARD_H = 0.0654
_HERO_CARD_STEP = 0.0240
_HERO_TOP = 0.7434
# The fanned 5-card hand isn't strictly linear: card 0 (the bottom of
# the fan) sits ~11 px further left than `LEFT + 0*STEP` would predict,
# so the linear formula clips its rank glyph and any J/K/T at slot 0
# returns None from classify_rank. Per-slot x-offsets fix this without
# affecting cards 1-4. Calibrated by sweeping x_left on a real frame
# (debug_1777228968620.png) where 0.412 is the only value that captures
# the full J glyph at slot 0.
_HERO_CARD_X: tuple[float, ...] = (
    0.4120,
    _HERO_CARDS_LEFT + 1 * _HERO_CARD_STEP,
    _HERO_CARDS_LEFT + 2 * _HERO_CARD_STEP,
    _HERO_CARDS_LEFT + 3 * _HERO_CARD_STEP,
    _HERO_CARDS_LEFT + 4 * _HERO_CARD_STEP,
)
def _board_row(y_top: float) -> tuple[ROI, ...]:
    return tuple(
        ROI(
            x=_BOARD_LEFT + i * _BOARD_CARD_STEP,
            y=y_top,
            w=_BOARD_CARD_W,
            h=_BOARD_CARD_H,
        )
        for i in range(NUM_BOARD_CARDS)
    )


BOARD_A: tuple[ROI, ...] = _board_row(_BOARD_A_TOP)
BOARD_B: tuple[ROI, ...] = _board_row(_BOARD_B_TOP)

HERO_HOLE: tuple[ROI, ...] = tuple(
    ROI(
        x=_HERO_CARD_X[i],
        y=_HERO_TOP,
        w=_HERO_CARD_W,
        h=_HERO_CARD_H,
    )
    for i in range(5)
)

POT_BANNER = ROI(x=0.4447, y=0.3087, w=0.1133, h=0.0403)


@dataclass(frozen=True)
class SeatROIs:
    seat: int
    name_plate: ROI
    stack_label: ROI
    committed_label: ROI
    button_anchor: ROI
    cards_back: ROI
    # Leftmost slice of the yellow turn-timer bar that ClubGG renders
    # below the active seat's plate. Bar depletes right-to-left, so the
    # leftmost ~10-15% stays solidly yellow until the timer is nearly
    # spent — that gives us a stable HSV-yellow signal for "this seat
    # is the current actor" (see cards.has_active_timer_bar). This is the
    # calibrated LINE (2-4 px tall); detection reads `timer_bar_band`.
    timer_bar_left: ROI

    @property
    def timer_bar_band(self) -> ROI:
        """The region the timer detector reads: `timer_bar_left` widened to
        a `TIMER_BAND_H` band centred on the calibrated line (TOOL-012). The
        villain lines are 1 px tall at 1080 and 2 px at 1391, so any vertical
        drift made the detector miss the very read the timer-bar CHECK
        depends on; `cards.has_active_timer_bar` accepts the band when any
        single row of it is mostly yellow."""
        return timer_band(self.timer_bar_left)


#: Height of the timer-bar detection band: ~10 px at the calibration height.
TIMER_BAND_H = 10 / CALIBRATION_SIZE[1]


def timer_band(line: ROI) -> ROI:
    """``line`` grown vertically to `TIMER_BAND_H`, same centre and width."""
    h = max(line.h, TIMER_BAND_H)
    cy = line.y + line.h / 2
    return ROI(line.x, cy - h / 2, line.w, h)


_SEATS_6: tuple[SeatROIs, ...] = (
    # `committed_label` is aimed at the chip-on-table oval (~"180" badge)
    # rendered between the seat's cards and the pot, NOT the idle $0 pill
    # on the seat plate. When no chips are in play the ROI reads empty
    # and `committed_chips` comes back None — the reconstructor's
    # stack-delta / bet-banner fallbacks cover that state.
    # seat 0: bottom-center (hero).
    SeatROIs(
        seat=0,
        name_plate=ROI(0.4328, 0.8584, 0.1344, 0.0338),
        stack_label=ROI(0.4195, 0.8929, 0.1637, 0.0352),
        committed_label=ROI(0.4707, 0.6715, 0.0586, 0.0237),
        button_anchor=ROI(0.5378, 0.6562, 0.0504, 0.0604),
        cards_back=ROI(0.3943, 0.7368, 0.2140, 0.1108),
        # Hero's bar is wider/taller than the others (40x4 px vs 26x2 px
        # at the 1927x1391 reference). Pixel TL = (848, 1316).
        timer_bar_left=ROI(0.4401, 0.9461, 0.0208, 0.0029),
    ),
    # seat 1: bottom-left (one step physical-CW from hero).
    SeatROIs(
        seat=1,
        name_plate=ROI(0.0986, 0.7563, 0.1069, 0.0273),
        stack_label=ROI(0.0708, 0.7871, 0.1637, 0.0352),
        committed_label=ROI(0.2117, 0.6175, 0.0586, 0.0237),
        button_anchor=ROI(0.2118, 0.6491, 0.0504, 0.0604),
        cards_back=ROI(0.0859, 0.6260, 0.2140, 0.1108),
        # Pixel TL = (201, 1150), 26x2 at 1927x1391 reference.
        timer_bar_left=ROI(0.1043, 0.8267, 0.0135, 0.0014),
    ),
    # seat 2: top-left.
    SeatROIs(
        seat=2,
        name_plate=ROI(0.0986, 0.2983, 0.1069, 0.0273),
        stack_label=ROI(0.0708, 0.3289, 0.1637, 0.0352),
        committed_label=ROI(0.1904, 0.4119, 0.0586, 0.0237),
        button_anchor=ROI(0.1387, 0.3994, 0.0504, 0.0604),
        cards_back=ROI(0.0859, 0.1828, 0.2140, 0.1108),
        # Pixel TL = (201, 513), 26x2 at 1927x1391 reference.
        timer_bar_left=ROI(0.1043, 0.3688, 0.0135, 0.0014),
    ),
    # seat 3: top-center.
    SeatROIs(
        seat=3,
        name_plate=ROI(0.4468, 0.1963, 0.1069, 0.0273),
        stack_label=ROI(0.4195, 0.2282, 0.1637, 0.0352),
        # y moved from 0.3098 to 0.2500 to clear POT_BANNER (top=0.3087):
        # original ROI was fully nested inside POT_BANNER, causing
        # Tesseract to read pot text as a phantom seat-3 commit.
        committed_label=ROI(0.4707, 0.2500, 0.0586, 0.0237),
        button_anchor=ROI(0.3968, 0.2362, 0.0504, 0.0604),
        cards_back=ROI(0.3943, 0.0821, 0.2140, 0.1108),
        # Pixel TL = (871, 369), 26x2 at 1927x1391 reference.
        timer_bar_left=ROI(0.4520, 0.2653, 0.0135, 0.0014),
    ),
    # seat 4: top-right.
    SeatROIs(
        seat=4,
        name_plate=ROI(0.7950, 0.2983, 0.1069, 0.0273),
        stack_label=ROI(0.7682, 0.3238, 0.1637, 0.0352),
        committed_label=ROI(0.7509, 0.4119, 0.0586, 0.0237),
        button_anchor=ROI(0.8135, 0.3994, 0.0504, 0.0604),
        cards_back=ROI(0.7342, 0.1828, 0.2140, 0.1108),
        # Pixel TL = (1541, 513), 26x2 at 1927x1391 reference.
        timer_bar_left=ROI(0.7997, 0.3688, 0.0135, 0.0014),
    ),
    # seat 5: bottom-right (one step physical-CCW from hero, or
    # equivalently (n-1) steps physical-CW).
    SeatROIs(
        seat=5,
        name_plate=ROI(0.7950, 0.7563, 0.1069, 0.0273),
        stack_label=ROI(0.7682, 0.7871, 0.1637, 0.0352),
        committed_label=ROI(0.7307, 0.6175, 0.0586, 0.0237),
        button_anchor=ROI(0.7405, 0.6491, 0.0504, 0.0604),
        cards_back=ROI(0.7342, 0.6260, 0.2140, 0.1108),
        # Pixel TL = (1541, 1150), 26x2 at 1927x1391 reference.
        timer_bar_left=ROI(0.7997, 0.8267, 0.0135, 0.0014),
    ),
)


def seats(num_seats: int) -> tuple[SeatROIs, ...]:
    if num_seats != 6:
        raise NotImplementedError(
            f"only 6-seat layout is wired up in Phase 1 (got {num_seats})"
        )
    return _SEATS_6
