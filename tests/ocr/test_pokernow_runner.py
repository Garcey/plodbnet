"""End-to-end tests for the PokerNow ingest runner.

Drives ``PokerNowRunner.handle_payload`` with ``pokernow.v1`` snapshots
(no websocket) and asserts the shared session pipeline ends up consistent:
seat-count reconfigured, hand-start fired at the flop, cards mirrored, and
flop actions inferred into ``action_log``.
"""

from __future__ import annotations

import pytest

from plo5bp.config import GameConfig
from plo5bp.ui import server


def _configure_session(bb_chips: int = 2, ante_chips: int = 12, dpb: float = 1.0) -> None:
    """Reset the global session to a clean state with a known unit mapping.

    With bb=2 chips and $1/bb, ``_ocr_cents_to_engine_chips`` resolves to
    0.02 engine-chips per cent, i.e. $1 → 2 chips. A $6 bomb-pot ante is
    12 chips (``ante_chips``).
    """
    s = server.session
    s.game_config = GameConfig(num_seats=2, starting_stack=400, ante=ante_chips, bb=bb_chips)
    s.num_seats = 2
    s.button_seat = 0
    s.hero_seat = 0
    s.dollars_per_bb = dpb
    s.last_hero_hole = None
    server._new_session_defaults()
    server.pokernow_runner._reconstructor = None
    server._set_active_reconstructor(None)
    server._rebuild_env()


def _seat(seat, name, *, hero=False, actor=False, angle, stack, cards, bet=None, bet_text=None, folded=False):
    return {
        "seat": seat, "name": name, "isHero": hero, "isActor": actor,
        "angleCW": angle, "stackDollars": stack, "folded": folded,
        "betText": bet_text, "betDollars": bet, "cards": cards,
    }


def _flop_payload(*, hero_bet=None, hero_actor=True, villain_actor=False,
                  hero_stack=86.0, villain_stack=62.0):
    """A bomb-pot flop frame: button at JJ (engine seat 1) so hero acts first."""
    return {
        "schema": "pokernow.v1", "variant": "plo5", "bombPot": True,
        "potDollars": 12.0, "button": {"seat": 6},
        "boards": [
            {"run": "1", "cards": ["4c", "Jc", "Ad"]},
            {"run": "2", "cards": ["9h", "2s", "Kd"]},
        ],
        "heroCards": ["Ts", "As", "4h", "3d", "2d"],
        "seats": [
            _seat(1, "Miles", hero=True, actor=hero_actor, angle=147,
                  stack=hero_stack, cards=["Ts", "As", "4h", "3d", "2d"],
                  bet=hero_bet, bet_text=("check" if hero_bet is None else None)),
            _seat(6, "JJ", actor=villain_actor, angle=328,
                  stack=villain_stack, cards=[None] * 5),
        ],
    }


def _river_payload():
    return {
        "schema": "pokernow.v1", "variant": "plo5", "bombPot": True,
        "potDollars": 12.0, "button": {"seat": 6},
        "boards": [
            {"run": "1", "cards": ["2c", "5h", "Ac", "6d", "3h"]},
            {"run": "2", "cards": ["7h", "Ks", "6s", "Kh", "Qs"]},
        ],
        "heroCards": ["Ts", "As", "4h", "3d", "2d"],
        "seats": [
            _seat(1, "Miles", hero=True, actor=True, angle=147, stack=74.0,
                  cards=["Ts", "As", "4h", "3d", "2d"]),
            _seat(6, "JJ", angle=328, stack=74.0, cards=[None] * 5),
        ],
    }


def test_bootstrap_from_full_board_frame():
    _configure_session()
    server.pokernow_runner.handle_payload(_river_payload())

    s = server.session
    assert s.num_seats == 2
    # Hand-start fired (first commit) on a flop-present frame.
    assert s.hand_in_hand_mask == frozenset({0, 1})
    # Button physical seat 6 → engine seat 1.
    assert s.button_seat == 1
    # Hero hole + both boards mirrored exactly.
    from plo5bp.ocr.types import Card
    assert s.hero_hole == [Card.parse(c).rank * 4 + Card.parse(c).suit
                           for c in ("Ts", "As", "4h", "3d", "2d")]
    assert all(c is not None for c in s.flop_a)
    assert all(c is not None for c in s.river_cards)
    assert s.env is not None
    assert server.pokernow_runner.last_error is None


def test_seat_count_reconfigures_from_default_six():
    server.session.num_seats = 6
    server.session.game_config = GameConfig(num_seats=6, starting_stack=400, ante=12, bb=2)
    server.session.dollars_per_bb = 1.0
    server._new_session_defaults()
    server.pokernow_runner._reconstructor = None

    server.pokernow_runner.handle_payload(_river_payload())
    # Two in-hand seats → engine reconfigured down to 2.
    assert server.session.num_seats == 2
    assert server.pokernow_runner._reconstructor.num_seats == 2


def test_flop_bet_infers_raise_action():
    _configure_session()
    r = server.pokernow_runner
    # Frame A: flop dealt, no bets, hero to act → fires hand-start, baseline.
    r.handle_payload(_flop_payload(hero_bet=None, hero_actor=True))
    assert server.session.hand_in_hand_mask == frozenset({0, 1})
    log_before = list(server.session.action_log)

    # Frame B: hero bets $9 (stack 86 → 77); villain now to act.
    r.handle_payload(_flop_payload(
        hero_bet=9.0, hero_actor=False, villain_actor=True, hero_stack=77.0,
    ))
    log_after = server.session.action_log
    assert len(log_after) > len(log_before)
    # The new entry is a raise (bet into an unraised flop).
    gate_raise = server._GATE_NAME_TO_IDX["raise"]
    assert any(e["gate"] == gate_raise for e in log_after[len(log_before):])
    assert r.last_error is None


def test_new_hand_fires_at_flop_without_action_or_button_move():
    """Regression: a fresh hand must show the flop + actor BEFORE anyone acts.

    The failing case was hero first-to-act: hand-start used to wait for the
    first action, so a first-to-act hero never saw a pre-action recommendation.
    Here the second hand shares a card with the first (so the old disjoint
    guard fails) and keeps the same button (so `button_changed` can't save it):
    only the hero-card-change signal can fire hand-start at the flop.
    """
    from plo5bp.ocr.types import Card

    _configure_session()
    r = server.pokernow_runner
    r.handle_payload(_flop_payload(hero_actor=True))
    assert server.session.hand_in_hand_mask == frozenset({0, 1})

    # Hand 2: new hero cards sharing "As" with hand 1, SAME button, no action.
    p2 = _flop_payload(hero_actor=True)
    p2["heroCards"] = ["As", "Kc", "Qd", "Jh", "9s"]
    p2["seats"][0]["cards"] = ["As", "Kc", "Qd", "Jh", "9s"]
    p2["boards"] = [
        {"run": "1", "cards": ["2c", "3c", "4c"]},
        {"run": "2", "cards": ["5d", "6d", "7d"]},
    ]
    r.handle_payload(p2)

    # Hand-start fired at the flop frame, pre-action: new hero cards are live,
    # the board is the new one, and no action was needed to get here.
    want = [Card.parse(c).rank * 4 + Card.parse(c).suit
            for c in ("As", "Kc", "Qd", "Jh", "9s")]
    assert server.session.hero_hole == want
    assert server.session.action_log == []
    st = server._state_dict()
    assert st["street"] == "flop"
    assert st["actor"] is not None  # hand is live; recommendation is reachable
    assert r.last_error is None


def test_street_advances_at_reveal_even_if_check_frames_were_dropped():
    """A checked street must advance the engine when the next card reveals,
    even if the intermediate check frames never arrived.

    The userscript coalesces snapshots, so the flop-check frames can be
    dropped and the server may jump straight from 'flop, hero to act' to
    'turn dealt'. The StreetReveal reconcile must still close the flop so the
    turn (card + recommendation) shows at deal time, not at the first action.
    """
    _configure_session()
    r = server.pokernow_runner
    hero = ["As", "Ks", "Qh", "Jd", "Tc"]

    def frame(b1, b2):
        return {
            "schema": "pokernow.v1", "variant": "plo5", "bombPot": True,
            "potDollars": 12.0, "button": {"seat": 6},
            "boards": [{"run": "1", "cards": b1}, {"run": "2", "cards": b2}],
            "heroCards": hero,
            "seats": [
                _seat(1, "Miles", hero=True, actor=True, angle=180, stack=80.0, cards=hero),
                _seat(6, "JJ", angle=0, stack=80.0, cards=[None] * 5),
            ],
        }

    # Flop (bootstrap hand-start), hero to act.
    r.handle_payload(frame(["2c", "3c", "4c"], ["5d", "6d", "7d"]))
    assert server.session.hand_in_hand_mask == frozenset({0, 1})
    assert server._state_dict()["street"] == "flop"

    # Turn deals — flop-check frames were dropped, so we jump straight here.
    r.handle_payload(frame(["2c", "3c", "4c", "8c"], ["5d", "6d", "7d", "9h"]))
    st = server._state_dict()
    assert st["street"] == "turn"            # reconcile closed the flop
    assert st["card_spec"]["turn"][0] is not None
    assert r.last_error is None


def test_closing_call_reconciled_at_reveal_when_signal_lost():
    """The call that closes a street must register even when its DOM signal
    is gone by the reveal frame.

    PokerNow sweeps the closing caller's chips to the pot and deals the next
    card in the same instant, so the committed amount is cleared and the stack
    delta landed in a (coalesced-away) earlier frame. The reveal-reconcile must
    still close the prior street by filling the call.
    """
    _configure_session()
    r = server.pokernow_runner
    hero = ["Ah", "Kh", "Qh", "Jh", "9h"]

    def frame(b1, b2, hero_bet=None, hero_actor=True, villain_actor=False,
              hero_stack=80.0, villain_stack=80.0):
        return {
            "schema": "pokernow.v1", "variant": "plo5", "bombPot": True,
            "potDollars": 12.0, "button": {"seat": 6},
            "boards": [{"run": "1", "cards": b1}, {"run": "2", "cards": b2}],
            "heroCards": hero,
            "seats": [
                _seat(1, "Miles", hero=True, actor=hero_actor, angle=180,
                      stack=hero_stack, cards=hero, bet=hero_bet),
                _seat(6, "JJ", actor=villain_actor, angle=0, stack=villain_stack,
                      cards=[None] * 5),
            ],
        }

    flop_b1, flop_b2 = ["2c", "3c", "4d"], ["5s", "6s", "7d"]
    # Flop hand-start, hero to act.
    r.handle_payload(frame(flop_b1, flop_b2))
    assert server.session.hand_in_hand_mask == frozenset({0, 1})
    # Hero bets $10 (captured); villain now to act.
    r.handle_payload(frame(flop_b1, flop_b2, hero_bet=10.0, hero_actor=False,
                           villain_actor=True, hero_stack=70.0))
    assert len(server.session.action_log) == 1  # hero's bet recorded

    # Turn deals. Villain's closing call signal is GONE: committed cleared and
    # stack unchanged (the drop was in a frame that got coalesced away).
    r.handle_payload(frame(flop_b1 + ["8c"], flop_b2 + ["Td"],
                           hero_bet=None, hero_actor=True, villain_actor=False,
                           hero_stack=70.0, villain_stack=80.0))
    st = server._state_dict()
    assert st["street"] == "turn"                 # prior street closed
    # Hero's bet + villain's reconciled call are both in the log.
    assert len(server.session.action_log) == 2
    assert server.session.action_log[1]["gate"] == server._GATE_NAME_TO_IDX["check_call"]
    assert r.last_error is None


def test_overbet_beyond_effective_stack_is_clamped_not_dropped():
    """A deep player betting more than a short opponent can cover must clamp,
    not vanish.

    Hero has $15, villain ~$100. PokerNow lets villain bet $50; the engine caps
    raises at the effective stack (~$15), so the raw bet is illegal. The replay
    must clamp it to the effective all-in instead of dropping it (which loses
    the bet and glitches the hand).
    """
    s = server.session
    s.game_config = GameConfig(num_seats=2, starting_stack=400, ante=2, bb=2)
    s.num_seats = 2
    s.button_seat = 0
    s.hero_seat = 0
    s.dollars_per_bb = 1.0
    s.last_hero_hole = None
    server._new_session_defaults()
    server.pokernow_runner._reconstructor = None
    server._set_active_reconstructor(None)
    server._rebuild_env()

    r = server.pokernow_runner
    hero = ["As", "Ks", "Qh", "Jd", "Tc"]
    b1, b2 = ["2c", "3c", "4d"], ["5s", "6s", "7d"]

    def frame(*, hero_check=False, friend_bet=None, hero_actor=False,
              friend_actor=False, hero_stack=15.0, friend_stack=100.0):
        return {
            "schema": "pokernow.v1", "variant": "plo5", "bombPot": True,
            "potDollars": 3.0, "button": {"seat": 6},  # button on villain → hero acts first
            "boards": [{"run": "1", "cards": b1}, {"run": "2", "cards": b2}],
            "heroCards": hero,
            "seats": [
                _seat(1, "Miles", hero=True, actor=hero_actor, angle=180,
                      stack=hero_stack, cards=hero,
                      bet_text="check" if hero_check else None),
                _seat(6, "JJ", actor=friend_actor, angle=0, stack=friend_stack,
                      cards=[None] * 5, bet=friend_bet),
            ],
        }

    r.handle_payload(frame(hero_actor=True))                       # hand-start, hero to act
    r.handle_payload(frame(hero_check=True, friend_actor=True))    # hero checks
    # Villain over-bets $50 into hero's $15 effective stack.
    r.handle_payload(frame(friend_bet=50.0, hero_actor=True, friend_stack=50.0))

    assert r.last_error is None
    gate_raise = server._GATE_NAME_TO_IDX["raise"]
    assert any(e["gate"] == gate_raise for e in server.session.action_log), \
        "villain's over-bet must be recorded (clamped), not dropped"
    st = server._state_dict()
    assert st["street"] == "flop"   # hero still to act facing the (clamped) bet
    assert st["actor"] == 0


def test_game_lock_ignores_foreign_game():
    """Frames from a second open PokerNow tab (different gameId) are ignored."""
    r = server.PokerNowRunner()
    assert r.accept_game("gameA") is True
    assert r.active_game_id == "gameA"
    assert r.accept_game("gameA") is True          # same game, fine
    assert r.accept_game("gameB") is False         # A is locked + fresh
    assert r.ignored_frames == 1
    assert r.accept_game(None) is True             # untagged → back-compat accept


def test_game_lock_switches_after_active_goes_quiet():
    """A new game may take over once the locked one is silent past the window."""
    r = server.PokerNowRunner()
    assert r.accept_game("gameA") is True
    # Simulate gameA going quiet longer than the switch window.
    r._active_game_at -= r._GAME_SWITCH_STALE + 1.0
    assert r.accept_game("gameB") is True
    assert r.active_game_id == "gameB"


@pytest.fixture(autouse=True)
def _restore_session():
    """Keep the module-global session from leaking across the wider suite.

    The runner reconfigures the session to a 2-seat table; restore the
    module's 6-seat default (config + env) so sibling test files that share
    the global ``server.session`` see a clean slate.
    """
    yield
    server._new_session_defaults()
    server.session.num_seats = 6
    server.session.button_seat = 0
    server.session.game_config = GameConfig(starting_stack=400000)
    server.session.dollars_per_bb = 2.0
    server.pokernow_runner._reconstructor = None
    server._set_active_reconstructor(None)
    server._rebuild_env()
