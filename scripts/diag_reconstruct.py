"""Phase 3 diagnostic: EventReconstructor forensics in isolation.

Feeds a synthesized (pre_bet, post_bet) FrameState pair through the
reconstructor with a hand-built EngineView to answer: *if the server
gave the reconstructor a well-formed baseline and engine snapshot,
would it emit the $180 raise on seat 1?*

Pre-bet is synthesized from post-bet because we never captured a
pre-commit frame — OCR started mid-hand. We set seat 1's
`stack_chips` to 163513 (the pre-bet $1,635.13 reading from the
user's eyeball confirmation) and clear its `committed_chips`.

Expected: `SeatAction(seat=1, gate="raise", chips=90000)` via the
Fix 5 stack-delta fallback. 18000 cents drop * 5.0 chips_per_cent =
90000 engine-chips.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from plo5bp.ocr.events import EngineView, EventReconstructor
from plo5bp.ocr.extract import extract_frame_state
from plo5bp.ocr.types import FrameState, SeatObs

FRAME = REPO_ROOT / "screenrecords" / "frames" / "debug_1776889069.png"

CHIPS_PER_CENT = 5.0
MIN_BET_CENTS = 2000  # 1bb floor at default config
SEAT_UNDER_TEST = 1  # Butt2Butt
PRE_BET_STACK_CENTS = 163513  # $1,635.13 pre-bet (user-confirmed eyeball)


def _cents_to_chips(cents: int | None) -> int:
    if cents is None:
        return 0
    return int(round(int(cents) * CHIPS_PER_CENT))


def _synthesize_pre_bet(post_bet: FrameState) -> FrameState:
    """Clone post_bet, roll seat 1 back to pre-bet stack + no commit."""
    new_seats = []
    for s in post_bet.seats:
        if s.seat == SEAT_UNDER_TEST:
            new_seats.append(
                replace(
                    s,
                    stack_chips=PRE_BET_STACK_CENTS,
                    committed_chips=None,
                    bet_banner=False,
                )
            )
        else:
            new_seats.append(s)
    return replace(post_bet, seats=tuple(new_seats))


def _print_seats(label: str, fs: FrameState) -> None:
    print(f"--- {label} ---")
    print(f"  button_seat={fs.button_seat}  pot={fs.pot_total_chips}")
    for s in fs.seats:
        print(
            f"  seat {s.seat}: folded={s.folded} banner={s.bet_banner} "
            f"stack={s.stack_chips} committed={s.committed_chips}"
        )


def main() -> int:
    img = cv2.imread(str(FRAME))
    if img is None:
        print(f"ERROR: couldn't read {FRAME}")
        return 1

    post_bet = extract_frame_state(img, num_seats=6)
    pre_bet = _synthesize_pre_bet(post_bet)

    _print_seats("POST_BET (extracted from frame)", post_bet)
    print()
    _print_seats("PRE_BET (synthesized: seat 1 stack=163513, commit=None)", pre_bet)
    print()

    # Button -> action order. On a 6-max bomb pot flop, first to act is
    # SB = (button + 1) % 6. If OCR missed the button, default to seat
    # 3 (arbitrary midfield) and loudly flag.
    if post_bet.button_seat is None:
        print("!!! button_seat is None — picking SB=3 as fallback; "
              "reconstructor walk will start there")
        button = 2
    else:
        button = int(post_bet.button_seat)
    sb = (button + 1) % 6

    # Build folded mask. hand_in_hand_mask logic: non-hero seats with
    # folded=True are out. Seat 0 (hero) treated as in-hand.
    folded_mask = tuple(
        (bool(s.folded) and s.seat != 0) for s in pre_bet.seats
    )
    # Stacks in engine-chips, derived from pre_bet (the baseline the
    # server should have locked in _begin_new_hand).
    stacks = tuple(_cents_to_chips(s.stack_chips) for s in pre_bet.seats)

    engine_view = EngineView(
        num_seats=6,
        current_actor=sb,
        street=1,  # flop
        awaiting_next_street=None,
        button_seat=button,
        committed_this_street=(0,) * 6,
        stacks=stacks,
        folded=folded_mask,
        all_in=(False,) * 6,
        bet_to_call=0,
        chips_per_cent=CHIPS_PER_CENT,
        min_bet_cents=MIN_BET_CENTS,
    )
    print("--- EngineView ---")
    print(f"  num_seats=6  current_actor={sb}  button_seat={button}")
    print(f"  street=1 (flop)  bet_to_call=0")
    print(f"  min_bet_cents={MIN_BET_CENTS}  chips_per_cent={CHIPS_PER_CENT}")
    print(f"  stacks={stacks}")
    print(f"  folded={folded_mask}")
    print()

    reconstructor = EventReconstructor(num_seats=6)
    reconstructor.rebaseline(pre_bet)
    print(f"rebaselined on pre_bet; reconstructor.last_fs set, "
          f"hero_hole_emitted={reconstructor.hero_hole_emitted}")
    print()

    events = reconstructor.step(post_bet, engine_view)
    print(f"--- step(post_bet) -> {len(events)} event(s) ---")
    for i, ev in enumerate(events):
        print(f"  [{i}] {type(ev).__name__}: {ev}")
    print()

    # Assertion: did the seat-1 raise fire?
    seat1_raise = None
    from plo5bp.ocr.events import SeatAction
    for ev in events:
        if (
            isinstance(ev, SeatAction)
            and ev.seat == SEAT_UNDER_TEST
            and ev.gate == "raise"
        ):
            seat1_raise = ev
            break

    print("=" * 60)
    if seat1_raise is not None:
        print(f"PASS: reconstructor emitted {seat1_raise}")
        print("     -> Phase 3 confirms the reconstructor is SOUND.")
        print("     -> Bug is upstream: the pixel-level ROI miss on")
        print("       seat 1's committed_label AND the missing pre-bet")
        print("       baseline (no stack_drop when last_fs is post-bet).")
        expected = 90000
        if seat1_raise.chips == expected:
            print(f"     -> chips={seat1_raise.chips} matches expected "
                  f"${expected/100/CHIPS_PER_CENT}")
        else:
            print(f"     !! chips={seat1_raise.chips}, expected {expected}")
    else:
        print("FAIL: no seat-1 raise in events.")
        print("     -> Walk broke before reaching seat 1.")
        print("     -> Trace: actor=3 folded=True (skip); actor=4 primary=None")
        print("       stack_drop=None banner=False -> fallback ladder exits")
        print("       via final `else: break` at events.py:352.")
        print()

    # Probe 2: force committed_chips=0 for silent seats (simulating
    # what'd happen if OCR returned 0 instead of None for empty chip
    # ovals). Tests whether the walk could coast past the quiet seats
    # to reach seat 1.
    print("=" * 60)
    print("PROBE 2: force committed_chips=0 on silent seats in BOTH frames")
    print("   (simulates OCR returning 0 for empty chip-oval ROIs)")
    print()
    from plo5bp.ocr.events import SeatAction as _SA

    def _force_zero_commits(fs: FrameState, keep_seat: int | None = None) -> FrameState:
        new_seats = []
        for s in fs.seats:
            if keep_seat is not None and s.seat == keep_seat:
                new_seats.append(s)  # leave seat 1 untouched
            elif s.committed_chips is None and not s.folded:
                new_seats.append(replace(s, committed_chips=0))
            else:
                new_seats.append(s)
        return replace(fs, seats=tuple(new_seats))

    pre_zero = _force_zero_commits(pre_bet, keep_seat=SEAT_UNDER_TEST)
    post_zero = _force_zero_commits(post_bet, keep_seat=SEAT_UNDER_TEST)
    reco2 = EventReconstructor(num_seats=6)
    reco2.rebaseline(pre_zero)
    events2 = reco2.step(post_zero, engine_view)
    print(f"   -> {len(events2)} event(s):")
    for i, ev in enumerate(events2):
        print(f"      [{i}] {type(ev).__name__}: {ev}")

    raise2 = next(
        (e for e in events2 if isinstance(e, _SA) and e.seat == 1 and e.gate == "raise"),
        None,
    )
    if raise2 is not None:
        print()
        print(f"   PROBE 2 RESULT: walk reaches seat 1 and raise fires ({raise2.chips} chips)")
        print("   -> Secondary finding: when committed_chips=None for silent seats,")
        print("      the fallback ladder's final `else: break` stops the walk at the")
        print("      FIRST silent seat. This compounds the ROI misalignment bug.")
        print("      Fix candidate: treat committed=None+no-drop+no-banner as `likely")
        print("      unchanged` and continue walking, OR have OCR return 0 not None")
        print("      when the ROI is empty.")
    else:
        print()
        print("   PROBE 2 RESULT: still no raise. The reconstructor has a deeper issue")
        print("   than the silent-seat walk-break. Inspect events2 above.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
