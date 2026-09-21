"""Regression tests for the 2026-09-20 code review — the ClubGG-OCR side of
`plo5bp/ui/server.py` (hand-start machine, tick loop, fold reconcile).

Drives `_mirror_observable_state` / `OcrRunner._tick` with synthetic
FrameStates. No OpenCV needed: the one place the tick imports the pixel
extractor is fed a stand-in module.

Covers I1 (stale hero-hole baseline), I5 (stale EngineView on the hand-start
tick), I8 (hero can join the mask / is never auto-acted), F10 (debounce state
cleared on resets, no hand-start re-fire loop), F11 (2-tick fold reconcile),
H7 (OCR never locks a duplicate card) and I10 (tick idles outside PLO5).
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

fastapi = pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")

from plo5bp.config import GameConfig  # noqa: E402
from plo5bp.ocr.events import EventReconstructor  # noqa: E402
from plo5bp.ocr.types import Card, FrameState, SeatObs  # noqa: E402
from plo5bp.ui import server  # noqa: E402


def _c(rank: int, suit: int) -> Card:
    return Card(rank=rank, suit=suit)


H1 = (_c(8, 1), _c(7, 1), _c(7, 0), _c(3, 2), _c(1, 2))
H2 = (_c(12, 3), _c(11, 3), _c(10, 3), _c(9, 3), _c(0, 0))  # disjoint from H1
NONE5 = (None,) * 5
FLOP_A = (_c(10, 0), _c(8, 0), _c(11, 0), None, None)
FLOP_B = (_c(5, 1), _c(6, 2), _c(1, 1), None, None)


def _idx(cards) -> tuple[int, ...]:
    return tuple(int(c.rank) * 4 + int(c.suit) for c in cards)


def _seat(i, stack=100_000, commit=None, folded=False, banner=False):
    return SeatObs(seat=i, stack_chips=stack, committed_chips=commit,
                   folded=folded, bet_banner=banner)


def _fs(seats, *, button=2, hero=H1, flop=True, board_a=None, board_b=None):
    ba = board_a if board_a is not None else (FLOP_A if flop else NONE5)
    bb = board_b if board_b is not None else (FLOP_B if flop else NONE5)
    return FrameState(board_a=ba, board_b=bb, hero_hole=hero, button_seat=button,
                      pot_total_chips=0, seats=tuple(seats))


def _full(**kw):
    return [_seat(i, **kw) for i in range(6)]


@pytest.fixture(autouse=True)
def _clean_session():
    s = server.session
    s.variant = server.VARIANT_PLO5
    s.game_config = GameConfig()
    s.dollars_per_bb = 20.0
    s.num_seats = 6
    s.button_seat = 0
    s.hero_seat = 0
    s.simple_ocr_mode = True
    server._new_session_defaults()
    server._reset_live_tracking()
    server.ocr_runner._reconstructor = None
    server.ocr_runner.running = False
    server.ocr_runner.latest_frame = None
    server.ocr_runner.last_error = None
    server.ocr_runner._tick_lock = asyncio.Lock()
    server._set_active_reconstructor(None)
    server._rebuild_env()
    yield
    s.simple_ocr_mode = True
    s.variant = server.VARIANT_PLO5
    s.game_config = GameConfig(starting_stack=400000)
    s.dollars_per_bb = 2.0
    s.button_seat = 0
    server._new_session_defaults()
    server._reset_live_tracking()
    server.ocr_runner._reconstructor = None
    server.ocr_runner.latest_frame = None
    server._set_active_reconstructor(None)
    server._rebuild_env()


@pytest.fixture()
def begin_calls(monkeypatch):
    """Records every `_begin_new_hand` invocation (button, had hero cards)."""
    calls: list[tuple[int, bool]] = []
    real = server._begin_new_hand

    def spy(fs, **kw):
        calls.append((kw.get("button_seat"), kw.get("hero_hole_indices") is not None))
        return real(fs, **kw)

    monkeypatch.setattr(server, "_begin_new_hand", spy)
    return calls


@pytest.fixture()
def tick(monkeypatch):
    """Run `OcrRunner._tick` on a synthetic FrameState (full, non-simple mode)."""
    queue: list[FrameState] = []
    fake = types.ModuleType("plo5bp.ocr.extract")
    fake.extract_frame_state = lambda img, num_seats=6, cache=None: queue.pop(0)
    monkeypatch.setitem(sys.modules, "plo5bp.ocr.extract", fake)

    runner = server.ocr_runner
    runner._reconstructor = EventReconstructor(num_seats=6)
    server._set_active_reconstructor(runner._reconstructor)
    runner.latest_frame = object()  # non-None sentinel; extraction is faked
    server.session.simple_ocr_mode = False

    def run(fs: FrameState) -> None:
        queue.append(fs)
        runner._tick()

    return run


def _engine() -> dict:
    return dict(server.session.env._rs.observation_dict())


# --- I1: stale hero-hole baseline ------------------------------------------------


def test_i1_begin_new_hand_clears_an_unreadable_baseline():
    server.session.last_hero_hole = _idx(H1)
    server._begin_new_hand(_fs(_full(), hero=NONE5), button_seat=3, hero_hole_indices=None)
    assert server.session.last_hero_hole is None  # used to stay on H1


def test_i1_one_unreadable_button_frame_does_not_restart_the_hand(begin_calls):
    s = server.session
    for _ in range(3):
        server._mirror_observable_state(_fs(_full(stack=34000), button=2, hero=H1))
    assert begin_calls == [(2, True)] and s.last_hero_hole == _idx(H1)
    for _ in range(6):
        server._mirror_observable_state(_fs(_full(stack=34000), button=2, hero=H1))
    begin_calls.clear()

    # Hand 2: the button moves BEFORE hero's new cards are visible.
    for _ in range(4):
        server._mirror_observable_state(
            _fs(_full(stack=34000), button=3, hero=NONE5, flop=False)
        )
    assert begin_calls == [(3, False)]
    assert s.last_hero_hole is None, "baseline must not stay on the old hand"
    begin_calls.clear()

    # Hand 2 proceeds; hero's cards are revealed and ADOPTED as the baseline.
    for _ in range(12):
        server._mirror_observable_state(_fs(_full(stack=34000), button=3, hero=H2))
    assert s.last_hero_hole == _idx(H2)
    assert s._pending_hero_hole_rotation_ticks == 0, "rotation must not latch"
    s.action_log = [{"gate": int(server.GATE_RAISE), "chips": 60000, "seat": 4}]

    # ONE frame with the dealer button occluded used to wipe the live hand.
    server._mirror_observable_state(_fs(_full(stack=34000), button=None, hero=H2))
    assert begin_calls == []
    assert len(s.action_log) == 1
    assert s.button_seat == 3


def test_i1_rotation_never_fires_on_a_frame_without_a_button_read(begin_calls):
    """Even with a genuinely rotated hole, an unreadable button confirms
    nothing; the trigger waits for a positive read of the moved button."""
    s = server.session
    for _ in range(2 + server._LOCK_AFTER_TICKS):
        server._mirror_observable_state(_fs(_full(), button=2, hero=H1))
    begin_calls.clear()

    for _ in range(server._STABILITY_TICKS_REQUIRED_LOCKED + 4):
        server._mirror_observable_state(_fs(_full(), button=None, hero=H2))
    assert begin_calls == [] and s.button_seat == 2

    for _ in range(server._STABILITY_TICKS_REQUIRED_LOCKED):
        server._mirror_observable_state(_fs(_full(), button=4, hero=H2))
    assert begin_calls and begin_calls[0][0] == 4
    assert s.button_seat == 4 and s.last_hero_hole == _idx(H2)


def test_i1_a_baseline_that_was_stale_at_birth_is_replaced(begin_calls):
    """Hand 2 starts (button move) while hand 1's cards are still face-up, so
    the baseline is recorded from the WRONG hand. Once hero's real cards hold
    with the button positively read on its seat, they replace it — otherwise
    the latched rotation bypasses the button debounce and one misread button
    frame restarts the hand."""
    s = server.session
    for _ in range(2 + server._LOCK_AFTER_TICKS):
        server._mirror_observable_state(_fs(_full(), button=2, hero=H1))
    for _ in range(server._BUTTON_STABLE_TICKS_LOCKED):
        server._mirror_observable_state(_fs(_full(), button=3, hero=H1))
    assert s.button_seat == 3 and s.last_hero_hole == _idx(H1)  # stale at birth
    begin_calls.clear()

    for _ in range(server._LOCK_AFTER_TICKS + server._STABILITY_TICKS_REQUIRED_LOCKED + 2):
        server._mirror_observable_state(_fs(_full(), button=3, hero=H2))
    assert begin_calls == []
    assert s.last_hero_hole == _idx(H2)
    assert s._pending_hero_hole_rotation_ticks == 0

    s.action_log = [{"gate": int(server.GATE_CHECK_CALL), "chips": 0, "seat": 4}]
    server._mirror_observable_state(_fs(_full(), button=5, hero=H2))  # 1-frame misread
    assert begin_calls == [] and len(s.action_log) == 1 and s.button_seat == 3


def test_i1_baseline_is_not_adopted_before_a_hand_is_committed():
    """Debouncer invariant: nothing hand-scoped is written pre-commit."""
    server._mirror_observable_state(_fs(_full(), button=2, hero=H1))
    assert not server.session.hand_in_hand_mask
    assert server.session.last_hero_hole is None
    assert server.session.button_seat == 0  # not updated pre-commit
    assert server.session.sitting_out_seats == frozenset()


# --- I5: the EngineView on a hand-start tick describes the NEW hand ----------------


def test_i5_begin_new_hand_invalidates_the_env():
    assert server.session.env is not None
    seats = [_seat(0, 34000), _seat(1, 145513), _seat(2, 163513),
             _seat(3, None, folded=True), _seat(4, 24650), _seat(5, None, folded=True)]
    server._begin_new_hand(_fs(seats, button=2), button_seat=2, hero_hole_indices=None)
    assert server.session.env is None
    view = server._engine_view_from_session()  # rebuilds on demand
    assert view.button_seat == 2
    assert view.current_actor == 4  # left of button 2 with seat 3 out
    assert view.stacks[1] == 145513 * 5  # the new hand's seeded stack, post-ante


def test_i5_bet_inside_the_debounce_window_lands_on_the_bettor(tick):
    def seats(sb_stack):
        return [_seat(0, 34000), _seat(1, 145513), _seat(2, sb_stack),
                _seat(3, None, folded=True), _seat(4, 24650),
                _seat(5, None, folded=True)]

    pre = _fs(seats(163513), button=2, hero=NONE5)
    post = _fs(seats(145513), button=2, hero=NONE5)  # seat 2 bet $180
    for f in (pre, post, post):
        tick(f)

    s = server.session
    assert s.hand_in_hand_mask == frozenset({0, 1, 2, 4})
    raises = [e for e in s.action_log if e["gate"] == int(server.GATE_RAISE)]
    assert [(e["seat"], e["chips"]) for e in raises] == [(2, 90000)]
    raw = _engine()
    # The bet used to be replayed onto HERO (stale pre-hand EngineView).
    assert list(raw["street_commit"]) == [0, 0, 90000, 0, 0, 0]
    assert int(raw["actor"]) == 4
    assert server.ocr_runner.last_error is None


# --- I8: hero and the participant mask ---------------------------------------------


def _hero_missing_anchor():
    def seats(hero_folded):
        return [_seat(0, 34000, folded=hero_folded), _seat(1, 145513),
                _seat(2, 163513), _seat(3, None, folded=True), _seat(4, 24650),
                _seat(5, None, folded=True)]

    anchor = _fs(seats(True), button=2, hero=NONE5, flop=False)
    live = _fs(seats(False), button=2, hero=H1)
    return anchor, live


def test_i8_hero_joins_the_mask_through_expansion():
    s = server.session
    anchor, live = _hero_missing_anchor()
    server._mirror_observable_state(anchor)
    server._mirror_observable_state(anchor)
    assert s.hand_in_hand_mask == frozenset({1, 2, 4})  # anchor missed hero

    server._mirror_observable_state(live)
    assert 0 not in s.hand_in_hand_mask  # one tick is not enough
    server._mirror_observable_state(live)
    assert s.hand_in_hand_mask == frozenset({0, 1, 2, 4})
    assert s.sitting_out_seats == frozenset({3, 5})

    server._rebuild_env()
    raw = _engine()
    assert int(raw["pot"]) == 4 * s.game_config.ante  # 4 antes, not 6
    assert list(raw["folded"]) == [False, False, False, True, False, True]
    # Seat 4 checks → the engine now WAITS on hero (it used to auto-act him).
    s.action_log.append({"gate": int(server.GATE_CHECK_CALL), "chips": 0, "seat": 4})
    server._rebuild_env()
    raw = _engine()
    assert int(raw["actor"]) == 0
    assert [int(h[0]) for h in raw["history"]] == [4]


def test_i8_engine_view_never_reports_hero_as_sitting_out():
    s = server.session
    anchor, _ = _hero_missing_anchor()
    server._mirror_observable_state(anchor)
    server._mirror_observable_state(anchor)
    assert 0 in s.sitting_out_seats  # session bookkeeping is unchanged...
    view = server._engine_view_from_session()
    # ...but the walk must not skip a seat the engine is going to wait on.
    assert view.sitting_out == (False, False, False, True, False, True)


def test_i8_a_folded_hero_is_not_re_added():
    s = server.session
    _, live = _hero_missing_anchor()
    server._mirror_observable_state(live)
    server._mirror_observable_state(live)
    assert 0 in s.hand_in_hand_mask
    s.folded_this_hand = frozenset({0})
    s.hand_in_hand_mask = frozenset({1, 2, 4})  # pretend hero was never in
    for _ in range(4):
        server._mirror_observable_state(live)
    assert 0 not in s.hand_in_hand_mask


# --- F10: debounce state ------------------------------------------------------------


def test_f10_reset_drops_the_stale_anchor():
    s = server.session
    recon = EventReconstructor(num_seats=6)
    server._set_active_reconstructor(recon)

    start = _fs(_full(stack=40000), button=2)
    server._mirror_observable_state(start)
    server._mirror_observable_state(start)
    assert s.hand_in_hand_mask == frozenset(range(6))

    t1_seats = _full(stack=40000)
    t1_seats[3] = _seat(3, 40000, folded=True)
    stale = _fs(t1_seats, button=2)          # becomes the pending anchor
    server._mirror_observable_state(stale)
    now_seats = [_seat(0, 10000), _seat(1, 10000), _seat(2, 40000),
                 _seat(3, 40000, folded=True), _seat(4, 40000), _seat(5, 40000)]
    now = _fs(now_seats, button=2)
    for _ in range(10):
        server._mirror_observable_state(now)
    assert s._pending_anchor_fs is stale

    server.reset()
    assert s._pending_anchor_fs is None and s._pending_stable_ticks == 0
    assert s.last_hero_hole is None and not s.hand_in_hand_mask

    # The next hand debounces from scratch on FRESH frames...
    server._mirror_observable_state(now)
    assert not s.hand_in_hand_mask
    server._mirror_observable_state(now)
    assert s.hand_in_hand_mask == frozenset({0, 1, 2, 4, 5})
    # ...and is seeded / rebaselined from them, not from the stale anchor.
    want = server._ocr_cents_to_engine_chips(10000) + s.game_config.ante
    assert s.game_config.resolved_stacks[0] == want
    assert recon.last_fs is now


def test_f10_format_switch_drops_live_tracking():
    s = server.session
    frame = _fs(_full(), button=2)
    server._mirror_observable_state(frame)
    server._mirror_observable_state(frame)
    server._mirror_observable_state(_fs(_full(), button=4))  # new pending anchor
    assert s.hand_in_hand_mask and s._pending_anchor_fs is not None
    try:
        server.set_format(server.FormatRequest(format="nlh_single"))
        assert s._pending_anchor_fs is None and s._pending_button is None
        assert s.last_hero_hole is None and not s.hand_in_hand_mask
    finally:
        server.set_format(server.FormatRequest(format="plo5_double_bomb"))


def test_f10_no_hand_start_loop_while_nobody_is_in_the_hand(begin_calls):
    empty = _fs([_seat(i, folded=True) for i in range(6)], button=3,
                hero=NONE5, flop=False)
    for _ in range(10):
        server._mirror_observable_state(empty)
    # The button move itself is one legitimate hand boundary; the empty mask
    # it produced used to re-arm `first_commit` on every following tick (9×).
    assert begin_calls == [(3, False)]

    dealt = _fs(_full(), button=3, hero=NONE5, flop=False)
    server._mirror_observable_state(dealt)
    server._mirror_observable_state(dealt)
    assert server.session.hand_in_hand_mask == frozenset(range(6))
    assert len(begin_calls) == 2


# --- F11: fold reconcile needs two ticks --------------------------------------------

TURN_A = FLOP_A[:3] + (_c(2, 3), None)
TURN_B = FLOP_B[:3] + (_c(9, 2), None)


def _three_handed(folded2=False, **seat1):
    return [_seat(0), _seat(1, **seat1), _seat(2, folded=folded2),
            _seat(3, None, folded=True), _seat(4, None, folded=True),
            _seat(5, None, folded=True)]


def test_f11_single_frame_fold_flicker_on_the_reveal_tick_is_ignored(tick):
    s = server.session
    flop = _fs(_three_handed(), button=2)
    tick(flop)
    tick(flop)  # hand-start commit
    assert s.hand_in_hand_mask == frozenset({0, 1, 2})
    s.action_log.extend({"gate": int(server.GATE_CHECK_CALL), "chips": 0} for _ in range(3))
    server._rebuild_env()
    tick(flop)  # engine is on the (padded) turn
    assert int(_engine()["street"]) == 2

    # Turn reveal; seat 2's card backs are missed on exactly this frame.
    tick(_fs(_three_handed(folded2=True), button=2, board_a=TURN_A, board_b=TURN_B))
    assert s.folded_this_hand == frozenset()
    assert s._pending_reveal_folds == frozenset({2})
    assert all(e["gate"] != int(server.GATE_FOLD) for e in s.action_log)

    # Next tick: seat 2 reads in-hand again → the candidate is dropped.
    tick(_fs(_three_handed(), button=2, board_a=TURN_A, board_b=TURN_B))
    assert s.folded_this_hand == frozenset()
    assert s._pending_reveal_folds == frozenset()
    assert s._pending_reveal_target is None
    assert all(e["gate"] != int(server.GATE_FOLD) for e in s.action_log)
    assert server._engine_view_from_session().sitting_out[2] is False
    assert list(_engine()["folded"])[:3] == [False, False, False]


def test_f11_pending_candidates_do_not_survive_to_a_later_reveal():
    """A first sighting is confirmed by the NEXT tick only; it must not sit
    around and get 'confirmed' by an unrelated glitch a street later."""
    s = server.session
    s.hand_in_hand_mask = frozenset({0, 1, 2})
    glitch = _fs(_three_handed(folded2=True), button=2)
    assert server._reconcile_missed_folds_on_street_reveal(glitch) is False
    assert s._pending_reveal_folds == frozenset({2})
    clean = _fs(_three_handed(), button=2)
    assert server._reconcile_missed_folds_on_street_reveal(clean) is False
    assert s._pending_reveal_folds == frozenset()
    assert server._reconcile_missed_folds_on_street_reveal(glitch) is False
    assert s.action_log == [] and s.folded_this_hand == frozenset()


def test_f11_exact_sources_reconcile_immediately():
    s = server.session
    s.hand_in_hand_mask = frozenset({0, 1, 2})
    frame = _fs(_three_handed(folded2=True), button=2)
    assert server._reconcile_missed_folds_on_street_reveal(frame, exact=True) is True
    assert s.folded_this_hand == frozenset({2})
    assert s.action_log == [{"gate": int(server.GATE_FOLD), "chips": 0}]


# --- H7: the OCR mirror never locks a duplicate card ---------------------------------


def test_h7_mirror_does_not_lock_a_card_another_slot_holds():
    s = server.session
    dup = H1[0]
    board_a = (dup, _c(8, 0), _c(11, 0), None, None)  # flop_a[0] misread as hero's card
    frame = _fs(_full(), button=2, hero=H1, board_a=board_a)
    for _ in range(server._CARD_STABLE_TICKS + 2):
        server._mirror_observable_state(frame)
    assert s.hero_hole == list(_idx(H1))
    assert s.flop_a[0] is None and s._card_slot_locked["flop_a"][0] is False
    assert s.flop_a[1:] == [_idx([board_a[1]])[0], _idx([board_a[2]])[0]]
    server._rebuild_env()  # used to raise 400 "duplicate card" on every tick


# --- I10: the tick idles outside PLO5 -------------------------------------------------


def test_i10_tick_idles_when_the_format_is_not_plo5(tick):
    s = server.session
    try:
        server.set_format(server.FormatRequest(format="nlh_single"))
        s.simple_ocr_mode = False
        cfg, hole = s.game_config, list(s.hero_hole)
        tick(_fs(_full(), button=2))  # a 5-card / 2-board frame
        assert "PLO5" in (server.ocr_runner.last_error or "")
        assert s.game_config is cfg and s.hero_hole == hole
        assert not s.hand_in_hand_mask
    finally:
        server.set_format(server.FormatRequest(format="plo5_double_bomb"))
