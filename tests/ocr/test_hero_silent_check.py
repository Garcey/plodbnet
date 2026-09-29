"""Hero's silent CHECK (TOOL-002) — the open issue in python/plo5bp/ocr/CLAUDE.md.

Heads-up bomb pot, hero first to act on the flop, no bet to face: hero
CHECKs, no chips move. The only positive CHECK signal used to be "the timer
bar was on hero and moved off", which needs hero's own (1-2 px) bar to have
been read as the lit one — right after a hand-start the lock is cleared, so a
quick check was never detected until villain put chips in or the turn came.

Two ways to see it now (pure reconstructor, no pixels):
* the turn PASSED: the next live seat holds the bar on two positive reads;
* hero's cards turning face up (ClubGG does that when hero's turn comes)
  locks the turn on hero, so the bar moving on reads as the check.
"""

from __future__ import annotations

from plo5bp.ocr.events import EngineView, EventReconstructor, SeatAction
from plo5bp.ocr.types import Card, FrameState, SeatObs

FLOP_A = (Card(10, 0), Card(8, 0), Card(11, 0), None, None)
FLOP_B = (Card(5, 1), Card(6, 2), Card(1, 1), None, None)
HERO = (Card(8, 1), Card(7, 1), Card(7, 0), Card(3, 2), Card(1, 2))


def _view(*, actor=0, bet_to_call=0, committed=(0, 0)) -> EngineView:
    return EngineView(
        num_seats=2, current_actor=actor, street=1, awaiting_next_street=None,
        button_seat=1, committed_this_street=committed, stacks=(170000, 170000),
        folded=(False, False), all_in=(False, False), bet_to_call=bet_to_call,
        chips_per_cent=5.0, min_bet_cents=2000,
    )


def _fs(*, actor_seat=None, hero_stack=34000, hero=HERO) -> FrameState:
    seats = (
        SeatObs(seat=0, stack_chips=hero_stack, committed_chips=None,
                is_actor=actor_seat == 0),
        SeatObs(seat=1, stack_chips=34000, committed_chips=None,
                is_actor=actor_seat == 1),
    )
    return FrameState(board_a=FLOP_A, board_b=FLOP_B, hero_hole=hero,
                      button_seat=1, pot_total_chips=None, seats=seats)


def _checks(events) -> list[int]:
    return [e.seat for e in events
            if isinstance(e, SeatAction) and e.gate == "check_call"]


def _fresh(frame: FrameState) -> EventReconstructor:
    r = EventReconstructor(num_seats=2)
    r.rebaseline(frame)  # a hand-start: the timer lock is cleared
    return r


def test_turn_passed_to_the_next_seat_reads_as_heros_check():
    r = _fresh(_fs())
    # Hero checked before hero's bar was ever read; villain's bar is lit.
    assert _checks(r.step(_fs(actor_seat=1), _view())) == []  # one read: wait
    assert _checks(r.step(_fs(actor_seat=1), _view())) == [0]


def test_no_check_while_facing_a_bet():
    r = _fresh(_fs())
    view = _view(bet_to_call=20000, committed=(0, 20000))
    for _ in range(3):
        assert _checks(r.step(_fs(actor_seat=1), view)) == []


def test_no_check_when_heros_stack_is_unreadable():
    r = _fresh(_fs())
    for _ in range(3):
        assert _checks(r.step(_fs(actor_seat=1, hero_stack=None), _view())) == []


def test_no_check_while_hero_still_holds_the_turn():
    r = _fresh(_fs())
    for _ in range(3):
        assert _checks(r.step(_fs(actor_seat=0), _view())) == []


def test_heros_cards_turning_face_up_lock_the_turn_on_hero():
    r = _fresh(_fs(hero=(None,) * 5))  # dealt face down
    # Hero's turn: the cards flip, hero's own bar is not read.
    assert _checks(r.step(_fs(), _view())) == []
    # Hero checked: the bar is read on villain once — the lock on hero turns
    # that single read into the check (the `timer_moved` branch).
    assert _checks(r.step(_fs(actor_seat=1), _view())) == [0]
