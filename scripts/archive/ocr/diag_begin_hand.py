"""Phase 6a offline diagnostic: simulate _begin_new_hand + _rebuild_env.

Bypasses the OCR runner and window capture — feeds the bug frame directly
through extract_frame_state and calls the server's _begin_new_hand path,
then builds the env and dumps the post-reset stacks. This validates the
Phase 6a hypothesis matrix without requiring a live ClubGG window.

The POST-bet frame (debug_1776889069.png) is what would realistically be
captured by the debouncer if the pre-bet tick wasn't stable. Seats 3 and
5 are sitting out in this frame.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

from plo5bp.ocr.extract import extract_frame_state
from plo5bp.ui import server as srv

FRAME = REPO_ROOT / "screenrecords" / "frames" / "debug_1776889069.png"


def main() -> int:
    img = cv2.imread(str(FRAME))
    if img is None:
        print("read fail")
        return 1

    fs = extract_frame_state(img, num_seats=6)

    # Simulate a glitched anchor: seat 1 stack read = 0 cents (banner
    # overlay during chip animation). Other seats unchanged.
    from dataclasses import replace
    glitched_seats = list(fs.seats)
    s1 = glitched_seats[1]
    glitched_seats[1] = replace(s1, stack_chips=0)
    fs = replace(fs, seats=tuple(glitched_seats))

    print("=== FrameState reads (seat 1 stack forced to 0) ===")
    for s in fs.seats:
        print(
            f"seat={s.seat} folded={s.folded} "
            f"banner={getattr(s, 'bet_banner', False)} "
            f"stack_cents={s.stack_chips} commit_cents={s.committed_chips}"
        )
    print(f"button={fs.button_seat}\n")

    # Mirror what _mirror_observable_state does: set hero hole + seat config
    srv.session.num_seats = 6
    srv.session.hero_seat = 0
    if fs.hero_hole is not None:
        hero_hole_indices = tuple(
            (int(c.rank) - 1) * 4 + int(c.suit) for c in fs.hero_hole if c is not None
        )
    else:
        hero_hole_indices = None

    # Simulate _begin_new_hand firing with this anchor
    print("=== _begin_new_hand (watch for anchor_seat_ocr logs) ===")
    srv._begin_new_hand(
        fs,
        button_seat=fs.button_seat if fs.button_seat is not None else 2,
        hero_hole_indices=hero_hole_indices,
    )
    cfg = srv.session.game_config
    print(f"\ncfg.resolved_stacks = {list(cfg.resolved_stacks)}")
    print(f"ante = {cfg.ante}")
    print(f"sitting_out_seats = {sorted(srv.session.sitting_out_seats)}")
    print(f"hand_in_hand_mask = {sorted(srv.session.hand_in_hand_mask)}")

    # Simulate the full _tick sequence. The reconstructor emits 3 events
    # per HANDOFF.md verification:
    #   SeatAction(seat=4, gate='check_call', chips=0)
    #   SeatAction(seat=0, gate='check_call', chips=0)
    #   SeatAction(seat=1, gate='raise', chips=90000)
    # The server's _tick appends these to action_log in order. _auto_fold
    # _sitting_out handles seats 3,5 during replay.
    from plo5bp.actions import GATE_RAISE, GATE_CHECK_CALL
    srv.session.action_log.append({"gate": int(GATE_CHECK_CALL), "chips": 0})
    srv.session.action_log.append({"gate": int(GATE_CHECK_CALL), "chips": 0})
    srv.session.action_log.append({"gate": int(GATE_RAISE), "chips": 90000})
    print(f"\naction_log: {srv.session.action_log}")

    print("\n=== _rebuild_env (watch for rebuild_env_stacks log) ===")
    try:
        srv._rebuild_env()
    except Exception as e:
        print(f"_rebuild_env failed: {type(e).__name__}: {e}")
        return 2

    env = srv.session.env
    assert env is not None
    raw = dict(env._rs.observation_dict())
    stacks = list(raw["stacks"])
    all_in = list(raw["all_in"])
    folded = list(raw["folded"])
    print("\n=== Post-reset env ===")
    print(f"{'seat':<4} | {'stack_chips':<12} | {'$display':<10} | {'all_in':<6} | folded")
    for i in range(6):
        dollars = stacks[i] / 500.0
        print(
            f"{i:<4} | {stacks[i]:<12} | ${dollars:<9.2f} | "
            f"{str(all_in[i]):<6} | {folded[i]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
