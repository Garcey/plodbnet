"""Diagnostic for OCR cold-start failure: pot=$0 + suppressed recommendation.

Walks a saved frame through the same pipeline the live OCR runner uses
(server.py:_tick): extract_frame_state -> _mirror_observable_state (4x to
clear the 2-tick debounce) -> _rebuild_env -> _compute_recommendation.
Prints debounce / mask / engine state at each step to pinpoint which of
the (A/B/C/D) hypotheses in
.claude/plans/the-last-thing-i-woolly-swing.md applies.

Usage:
    .venv/Scripts/python scripts/diag_ocr_frame.py [frame_path] [num_seats]

Defaults to the most recent debug_*.png in screenrecords/frames/, and
num_seats=6 (the only layout rois.py supports today).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from plo5bp.ocr.extract import extract_frame_state
from plo5bp.ui import server as srv


def _find_default_frame() -> Path:
    frames_dir = REPO_ROOT / "screenrecords" / "frames"
    candidates = sorted(
        frames_dir.glob("debug_*.png"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise SystemExit("No debug_*.png frames in screenrecords/frames/")
    return candidates[-1]


def _print_framestate(fs) -> None:
    print("=== FrameState ===")
    print(f"button_seat     = {fs.button_seat}")
    print(f"pot_total_chips = {fs.pot_total_chips}")
    print(f"hero_hole       = {fs.hero_hole}")
    print(f"board_a         = {fs.board_a}")
    print(f"board_b         = {fs.board_b}")
    print()
    print(
        f"{'seat':<4} | {'folded':<6} | {'banner':<6} | "
        f"{'stack_cents':<11} | commit_cents"
    )
    for s in fs.seats:
        print(
            f"{s.seat:<4} | "
            f"{str(s.folded):<6} | "
            f"{str(getattr(s, 'bet_banner', False)):<6} | "
            f"{str(s.stack_chips):<11} | "
            f"{s.committed_chips}"
        )
    print()


def _reset_session(num_seats: int) -> None:
    """Reset session to fresh-cold-start defaults (mirrors what an OCR
    runner sees on its very first tick after import)."""
    srv.session.num_seats = num_seats
    srv.session.hero_seat = 0
    srv.session.button_seat = 0
    srv.session.last_hero_hole = None
    srv.session.hand_in_hand_mask = frozenset()
    srv.session.folded_this_hand = frozenset()
    srv.session.sitting_out_seats = frozenset()
    srv.session._pending_button = None
    srv.session._pending_sitting_out = None
    srv.session._pending_stable_ticks = 0
    srv.session._pending_anchor_fs = None
    srv.session._pending_mask_additions = frozenset()
    srv.session._pending_mask_additions_ticks = 0
    srv.session.action_log = []
    srv.session.hero_hole = [None] * 5
    srv.session.flop_a = [None] * 3
    srv.session.flop_b = [None] * 3
    srv.session.turn_cards = [None, None]
    srv.session.river_cards = [None, None]
    srv.session.env = None
    srv.session.last_obs = None
    srv.session.last_info = None


def _patch_begin_new_hand(call_log: list) -> callable:
    original = srv._begin_new_hand

    def wrapped(*args, **kwargs):
        call_log.append(
            {
                "button_seat": kwargs.get("button_seat"),
                "hero_hole_indices": kwargs.get("hero_hole_indices"),
            }
        )
        return original(*args, **kwargs)

    srv._begin_new_hand = wrapped
    return original


def _print_tick_state(tick: int, fired: bool) -> None:
    s = srv.session
    anchor_set = s._pending_anchor_fs is not None
    if anchor_set:
        anchor_plausible = srv._anchor_fs_stacks_plausible(s._pending_anchor_fs)
    else:
        anchor_plausible = None
    committed_ready = (
        s._pending_stable_ticks >= srv._STABILITY_TICKS_REQUIRED
        and (not anchor_set or anchor_plausible)
    )
    pending_so = (
        sorted(s._pending_sitting_out)
        if s._pending_sitting_out is not None
        else None
    )
    print(f"--- Tick {tick} ---")
    print(f"  _pending_button       = {s._pending_button}")
    print(f"  _pending_sitting_out  = {pending_so}")
    print(f"  _pending_stable_ticks = {s._pending_stable_ticks}")
    print(f"  _pending_anchor_fs    = {'set' if anchor_set else 'None'}")
    print(f"  anchor_plausible      = {anchor_plausible}")
    print(f"  committed_ready       = {committed_ready}")
    print(f"  hand_in_hand_mask     = {sorted(s.hand_in_hand_mask)}")
    print(f"  sitting_out_seats     = {sorted(s.sitting_out_seats)}")
    print(f"  button_seat           = {s.button_seat}")
    print(f"  _begin_new_hand fired = {fired}")
    print()


def main(argv: list[str]) -> int:
    if len(argv) >= 2:
        frame = Path(argv[1])
    else:
        frame = _find_default_frame()
    num_seats = int(argv[2]) if len(argv) >= 3 else 6

    print(f"Frame:     {frame}")
    print(f"num_seats: {num_seats}")
    print()

    img = cv2.imread(str(frame))
    if img is None:
        print(f"cv2.imread failed for {frame}")
        return 1

    _reset_session(num_seats)

    fs = extract_frame_state(img, num_seats=num_seats)
    _print_framestate(fs)

    begin_calls: list[dict] = []
    original = _patch_begin_new_hand(begin_calls)
    try:
        for tick in range(1, 5):
            calls_before = len(begin_calls)
            srv._mirror_observable_state(fs)
            fired = len(begin_calls) > calls_before
            _print_tick_state(tick, fired)
    finally:
        srv._begin_new_hand = original

    print("=== _begin_new_hand call summary ===")
    print(f"calls = {len(begin_calls)}")
    for i, c in enumerate(begin_calls, 1):
        print(f"  call {i}: button_seat={c['button_seat']} "
              f"hero_hole={c['hero_hole_indices']}")
    print()

    print("=== _rebuild_env ===")
    try:
        srv._rebuild_env()
    except Exception as e:
        print(f"_rebuild_env failed: {type(e).__name__}: {e}")
        return 2
    env = srv.session.env
    if env is None:
        print("env is None after _rebuild_env")
        return 2

    raw = dict(env._rs.observation_dict())
    actor_raw = raw.get("actor")
    actor = int(actor_raw) if actor_raw is not None else None
    print(f"  env.pot               = {raw.get('pot')}")
    print(f"  env.current_actor()   = {actor}")
    print(f"  env.bet_to_call       = {raw.get('bet_to_call')}")
    print(f"  env.street            = {raw.get('street')}")
    print(f"  env.stacks            = {list(raw['stacks'])}")
    print(f"  env.folded            = {list(raw['folded'])}")
    print(f"  env.awaiting_next_street = {env.awaiting_next_street()}")
    print()

    print("=== _compute_recommendation gates ===")
    s = srv.session
    if env is None:
        print("  GATE: env is None -> return None")
    elif actor is None:
        print("  GATE: actor is None -> return None")
    elif actor != s.hero_seat:
        print(f"  GATE: actor ({actor}) != hero_seat ({s.hero_seat}) -> return None")
    elif not srv._hero_info_complete():
        print("  GATE: _hero_info_complete() False -> return None")
    elif s.last_obs is None or s.last_info is None:
        print("  GATE: last_obs or last_info None -> return None")
    else:
        rec = srv._compute_recommendation()
        print(f"  GATE: passed -> recommendation = {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
