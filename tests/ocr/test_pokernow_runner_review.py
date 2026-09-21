"""Regression tests for the 2026-09-20 code review — the PokerNow side of
`plo5bp/ui/server.py`.

Drives `PokerNowRunner.handle_payload` with `pokernow.v1` snapshots. Covers
I5/F14 (stale env after a hand-start / seat-count change — ingest used to 500
forever), I9 (reconstructor registration, bare button correction, mid-street
stack seeding), the 2..8 engine seat ceiling (B8 contract), F10 (live-source
switch) and H7 (no duplicate card locks from the DOM mirror).
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")

from plo5bp.config import GameConfig  # noqa: E402
from plo5bp.ocr.events import EventReconstructor  # noqa: E402
from plo5bp.ui import server  # noqa: E402

H1 = ["Ts", "As", "4h", "3d", "2d"]
H2 = ["Qs", "Qd", "7h", "7d", "5c"]
B1, B2 = ["4c", "Jc", "Ad"], ["9h", "2s", "Kd"]

RAISE = int(server.GATE_RAISE)
CHECK = int(server.GATE_CHECK_CALL)


def _configure(n: int = 2) -> None:
    """bb=2 chips at $1/bb ⇒ $1 = 2 chips; the $6 bomb-pot ante is 12 chips."""
    s = server.session
    s.variant = server.VARIANT_PLO5
    s.game_config = GameConfig(num_seats=n, starting_stack=400, ante=12, bb=2)
    s.num_seats = n
    s.button_seat = 0
    s.hero_seat = 0
    s.dollars_per_bb = 1.0
    server._new_session_defaults()
    server._reset_live_tracking()
    server.pokernow_runner._reconstructor = None
    server.pokernow_runner.last_error = None
    server.ocr_runner._reconstructor = None
    server.ocr_runner.running = False
    server._set_active_reconstructor(None)
    server._rebuild_env()


@pytest.fixture(autouse=True)
def _session():
    _configure()
    yield
    s = server.session
    s.variant = server.VARIANT_PLO5
    s.num_seats = 6
    s.button_seat = 0
    s.game_config = GameConfig(starting_stack=400000)
    s.dollars_per_bb = 2.0
    server._new_session_defaults()
    server._reset_live_tracking()
    server.pokernow_runner._reconstructor = None
    server.pokernow_runner.last_error = None
    server.ocr_runner._reconstructor = None
    server._set_active_reconstructor(None)
    server._note_live_source(None)
    server._rebuild_env()


def _seat(no, name, *, hero=False, actor=False, angle, stack, cards,
          bet=None, bet_text=None, folded=False):
    return {"seat": no, "name": name, "isHero": hero, "isActor": actor,
            "angleCW": angle, "stackDollars": stack, "folded": folded,
            "betText": bet_text, "betDollars": bet, "cards": cards}


def _hu(*, hero_cards, button, flop=True, hero_bet=None, vil_bet=None,
        hero_actor=False, vil_actor=False, hero_stack=86.0, vil_stack=62.0,
        pot=12.0):
    """Heads-up frame: hero = physical seat 1 (engine 0), villain = physical 6
    (engine 1)."""
    boards = (
        [{"run": "1", "cards": B1}, {"run": "2", "cards": B2}] if flop else []
    )
    return {
        "schema": "pokernow.v1", "variant": "plo5", "bombPot": True,
        "potDollars": pot, "button": {"seat": button}, "boards": boards,
        "heroCards": hero_cards,
        "seats": [
            _seat(1, "Miles", hero=True, actor=hero_actor, angle=147,
                  stack=hero_stack, cards=hero_cards, bet=hero_bet),
            _seat(6, "JJ", actor=vil_actor, angle=328, stack=vil_stack,
                  cards=[None] * 5, bet=vil_bet),
        ],
    }


def _log() -> list[tuple[int, int]]:
    return [(e["gate"], e["chips"]) for e in server.session.action_log]


def _engine() -> dict:
    return dict(server.session.env._rs.observation_dict())


# --- I5 / F14: table size changes -----------------------------------------------------


def test_f14_seat_count_increase_does_not_wedge_the_ingest():
    r = server.pokernow_runner
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_actor=True))
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_bet=9.0, vil_actor=True,
                         hero_stack=77.0))
    assert len(_engine()["folded"]) == 2 and _engine()["actor"] == 1

    # Hand 2: a third player sat in. The env still had 2 seats while the view
    # was sized for 3 ⇒ IndexError on this and EVERY later frame.
    three = {
        "schema": "pokernow.v1", "variant": "plo5", "bombPot": True,
        "potDollars": 18.0, "button": {"seat": 6},
        "boards": [{"run": "1", "cards": ["2c", "3c", "4d"]},
                   {"run": "2", "cards": ["5s", "6s", "7d"]}],
        "heroCards": H2,
        "seats": [
            _seat(1, "Miles", hero=True, angle=180, stack=80.0, cards=H2),
            _seat(3, "New", actor=True, angle=270, stack=50.0, cards=[None] * 5),
            _seat(6, "JJ", angle=0, stack=60.0, cards=[None] * 5),
        ],
    }
    for k in range(3):
        three["potDollars"] = 18.0 + k
        r.handle_payload(three)  # must not raise
        assert r.last_error is None
        assert server.session.num_seats == 3
        assert len(_engine()["folded"]) == 3
    assert server.session.hand_in_hand_mask == frozenset({0, 1, 2})
    assert _log() == []


def test_f14_seat_count_decrease_does_not_wedge_the_ingest():
    _configure(n=3)
    r = server.pokernow_runner

    def three(actor_a, actor_b, a_stack=80, a_bet=None, pot=18):
        return {
            "schema": "pokernow.v1", "variant": "plo5", "bombPot": True,
            "potDollars": pot, "button": {"seat": 1},
            "boards": [{"run": "1", "cards": B1}, {"run": "2", "cards": B2}],
            "heroCards": H1,
            "seats": [
                _seat(1, "Miles", hero=True, angle=180, stack=80, cards=H1),
                _seat(3, "A", actor=actor_a, angle=270, stack=a_stack,
                      cards=[None] * 5, bet=a_bet),
                _seat(6, "B", actor=actor_b, angle=0, stack=80, cards=[None] * 5),
            ],
        }

    r.handle_payload(three(True, False))
    r.handle_payload(three(False, True, a_stack=70, a_bet=10, pot=28))
    assert _engine()["actor"] == 2  # stale actor index 2 is out of range below

    for k in range(2):
        r.handle_payload(_hu(hero_cards=H2, button=6, hero_actor=True,
                             hero_stack=70.0, vil_stack=90.0, pot=12.0 + k))
        assert r.last_error is None
        assert server.session.num_seats == 2 and len(_engine()["folded"]) == 2


def test_i5_engine_view_on_the_hand_start_tick_is_the_new_hand(monkeypatch):
    r = server.pokernow_runner
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_actor=True))
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_bet=9.0, vil_actor=True,
                         hero_stack=77.0))
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_bet=9.0, vil_bet=30.0,
                         hero_actor=True, hero_stack=77.0, vil_stack=32.0))
    assert _log() == [(RAISE, 18), (RAISE, 60)]

    views = []
    real = server._engine_view_from_session

    def spy(*a, **k):
        v = real(*a, **k)
        views.append(v)
        return v

    monkeypatch.setattr(server, "_engine_view_from_session", spy)
    # Hand 2's first flop frame: new cards, button moved to hero.
    r.handle_payload(_hu(hero_cards=H2, button=1, vil_actor=True,
                         hero_stack=71.0, vil_stack=77.0))
    v = views[0]
    # It used to describe hand 1: actor 0 facing 60 with commits (18, 60).
    assert v.button_seat == 0 and v.current_actor == 1
    assert v.bet_to_call == 0 and v.committed_this_street == (0, 0)
    assert v.stacks == (142, 154)
    assert _log() == [] and r.last_error is None


# --- I9: reconstructor registration ---------------------------------------------------


def test_i9_pokernow_reconstructor_is_registered_on_every_payload():
    r = server.pokernow_runner
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_actor=True))
    mine = r._reconstructor

    # A ClubGG session ran in between and left ITS reconstructor registered;
    # same seat count, so PokerNow does not build (and re-register) a new one.
    stale = EventReconstructor(num_seats=server.session.num_seats)
    server.ocr_runner._reconstructor = stale
    server._set_active_reconstructor(stale)

    # Hand 2: pre-flop frame (antes shown as bets), flop frame, villain bets.
    r.handle_payload(_hu(hero_cards=H2, button=1, flop=False, hero_bet=6.0,
                         vil_bet=6.0, hero_stack=80.0, vil_stack=56.0))
    r.handle_payload(_hu(hero_cards=H2, button=1, vil_actor=True,
                         hero_stack=80.0, vil_stack=56.0))
    assert r._reconstructor is mine
    assert server._active_reconstructor() is mine
    assert _log() == []  # used to be a phantom ante RAISE 12 + CALL
    r.handle_payload(_hu(hero_cards=H2, button=1, vil_bet=9.0, hero_actor=True,
                         hero_stack=80.0, vil_stack=47.0))
    assert _log() == [(RAISE, 18)]
    raw = _engine()
    assert list(raw["stacks"]) == [160, 94] and int(raw["pot"]) == 42


def test_i9_ocr_runner_unregisters_its_reconstructor_when_it_stops():
    runner = server.ocr_runner
    runner._reconstructor = EventReconstructor(num_seats=6)
    server._set_active_reconstructor(runner._reconstructor)
    runner._retire_reconstructor()
    assert server._LIVE_RECONSTRUCTOR is None
    assert server._active_reconstructor() is None

    # Never unregisters somebody else's.
    other = EventReconstructor(num_seats=2)
    server._set_active_reconstructor(other)
    runner._reconstructor = EventReconstructor(num_seats=6)
    runner._retire_reconstructor()
    assert server._active_reconstructor() is other


# --- I9: bare button change = correction, not a new hand --------------------------------


def test_i9_button_dom_lag_corrects_the_button_without_wiping_the_hand(monkeypatch):
    s = server.session
    r = server.pokernow_runner
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_actor=True))
    # Hand 2 starts on the hero-card signal while the button DOM still shows
    # the old seat (6 = villain); the TRUE button is hero, so villain acts first.
    r.handle_payload(_hu(hero_cards=H2, button=6, vil_actor=True,
                         hero_stack=80.0, vil_stack=56.0))
    assert s.button_seat == 1 and s.game_config.resolved_stacks == (172, 124)
    r.handle_payload(_hu(hero_cards=H2, button=6, vil_bet=9.0, hero_actor=True,
                         hero_stack=80.0, vil_stack=47.0))
    # Under the stale button the walk had to invent a hero check first.
    assert _log() == [(CHECK, 0), (RAISE, 18)]
    cards_before = (list(s.hero_hole), list(s.flop_a), list(s.flop_b))
    hole_baseline = s.last_hero_hole

    calls = []
    real = server._begin_new_hand
    monkeypatch.setattr(
        server, "_begin_new_hand", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    )
    # The dealer-button DOM catches up.
    r.handle_payload(_hu(hero_cards=H2, button=1, vil_bet=9.0, hero_actor=True,
                         hero_stack=80.0, vil_stack=47.0))

    assert calls == [], "a bare button change must not start a new hand"
    assert s.button_seat == 0
    assert s.game_config.resolved_stacks == (172, 124)  # was re-seeded to (172, 106)
    assert (list(s.hero_hole), list(s.flop_a), list(s.flop_b)) == cards_before
    assert s.hand_in_hand_mask == frozenset({0, 1})
    assert s.last_hero_hole == hole_baseline
    # The street is re-derived under the corrected acting order.
    assert [(e["gate"], e["chips"], e["seat"]) for e in s.action_log] == [(RAISE, 18, 1)]
    raw = _engine()
    assert int(raw["button"]) == 0 and int(raw["actor"]) == 0
    assert list(raw["stacks"]) == [160, 94]  # villain really has $47 behind
    assert list(raw["street_commit"]) == [0, 18] and int(raw["pot"]) == 42
    assert r.last_error is None


def test_i9_button_correction_before_any_action_keeps_everything(monkeypatch):
    s = server.session
    r = server.pokernow_runner
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_actor=True))
    r.handle_payload(_hu(hero_cards=H2, button=6, vil_actor=True,
                         hero_stack=80.0, vil_stack=56.0))
    # A slot the user overrode survives a correction (a hand-start wipes it).
    s._card_slot_locked["river_cards"][0] = True
    cfg = s.game_config
    calls = []
    real = server._begin_new_hand
    monkeypatch.setattr(
        server, "_begin_new_hand", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    )
    r.handle_payload(_hu(hero_cards=H2, button=1, vil_actor=True,
                         hero_stack=80.0, vil_stack=56.0, pot=12.5))
    assert calls == []
    assert s.button_seat == 0 and _log() == []
    assert s.game_config is cfg
    assert s._card_slot_locked["river_cards"][0] is True
    assert int(_engine()["actor"]) == 1  # villain first under the real button


def test_i9_button_change_with_unreadable_hero_cards_is_still_a_new_hand(monkeypatch):
    r = server.pokernow_runner
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_actor=True))
    calls = []
    real = server._begin_new_hand
    monkeypatch.setattr(
        server, "_begin_new_hand", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    )
    # Hero's cards are not readable ⇒ the button is the only new-hand signal.
    frame = _hu(hero_cards=[None] * 5, button=1, vil_actor=True)
    frame["heroCards"] = []
    r.handle_payload(frame)
    assert calls == [1]
    assert server.session.button_seat == 0


# --- I9: mid-street stack seeding ---------------------------------------------------------


def test_i9_hand_start_mid_street_accounts_for_visible_commits():
    """PokerNow attaches while villain's $9 bet is already on the table."""
    s = server.session
    r = server.pokernow_runner
    r.handle_payload(_hu(hero_cards=H2, button=1, vil_bet=9.0, hero_actor=True,
                         hero_stack=80.0, vil_stack=47.0, pot=21.0))
    # starting = behind + commit + ante: 94 + 18 + 12 (was 94 + 12 = 106).
    assert s.game_config.resolved_stacks == (172, 124)
    assert _log() == [(RAISE, 18)]
    raw = _engine()
    assert list(raw["stacks"]) == [160, 94]
    assert list(raw["street_commit"]) == [0, 18]
    assert int(raw["actor"]) == 0 and r.last_error is None


def test_i9_ante_bets_still_on_display_are_not_flop_bets():
    """Should the first flop frame still show the $6 ante bets, they are
    neither folded into the stacks nor re-derived as actions."""
    s = server.session
    r = server.pokernow_runner
    r.handle_payload(_hu(hero_cards=H2, button=1, hero_bet=6.0, vil_bet=6.0,
                         vil_actor=True, hero_stack=80.0, vil_stack=56.0))
    assert s.game_config.resolved_stacks == (172, 124)  # behind + ante only
    assert _log() == []
    raw = _engine()
    assert list(raw["stacks"]) == [160, 112] and int(raw["pot"]) == 24


def test_i9_clubgg_seeding_ignores_pixel_commit_reads():
    """OCR `committed_chips` is noisy; only exact sources fold it into a stack."""
    from plo5bp.ocr.types import FrameState, SeatObs

    _configure(n=2)
    fs = FrameState(
        board_a=(None,) * 5, board_b=(None,) * 5, hero_hole=(None,) * 5,
        button_seat=0, pot_total_chips=0,
        seats=(SeatObs(seat=0, stack_chips=8000, committed_chips=0, folded=False),
               SeatObs(seat=1, stack_chips=4700, committed_chips=900, folded=False)),
    )
    server._begin_new_hand(fs, button_seat=0, hero_hole_indices=None)
    assert server.session.game_config.resolved_stacks == (172, 106)


# --- engine seat ceiling (PokerNow seats up to 10) -------------------------------------------


def _ring(n: int) -> dict:
    hero = ["As", "Ks", "Qh", "Jd", "Tc"]
    seats = [_seat(1, "Hero", hero=True, actor=True, angle=180, stack=80.0, cards=hero)]
    for k in range(1, n):
        seats.append(_seat(1 + k, f"V{k}", angle=(180 + k * 360 / n) % 360,
                           stack=80.0, cards=[None] * 5))
    return {"schema": "pokernow.v1", "variant": "plo5", "bombPot": True,
            "potDollars": 6.0 * n, "button": {"seat": n},
            "boards": [{"run": "1", "cards": ["2c", "3c", "4d"]},
                       {"run": "2", "cards": ["5s", "6s", "7d"]}],
            "heroCards": hero, "seats": seats}


@pytest.mark.parametrize("n", [9, 10])
def test_b8_tables_above_the_engine_limit_are_refused_gracefully(n):
    s = server.session
    r = server.pokernow_runner
    cfg, env = s.game_config, s.env
    for _ in range(3):
        r.handle_payload(_ring(n))  # never raises, never 500-loops
    assert s.game_config is cfg and s.num_seats == 2 and s.env is env
    assert not s.hand_in_hand_mask and s.hero_hole == [None] * 5
    assert "at most 8" in (r.last_error or "")
    assert r.status()["table_seats"] == n

    # A hand the engine CAN represent recovers on its own.
    r.handle_payload(_ring(8))
    assert r.last_error is None
    assert s.num_seats == 8 and s.hand_in_hand_mask == frozenset(range(8))
    assert len(_engine()["folded"]) == 8


# --- F10: switching the live source ---------------------------------------------------------


def test_f10_first_pokernow_frame_after_clubgg_resets_the_hand_start_machine():
    s = server.session
    server._note_live_source("ocr")
    # What a ClubGG session leaves behind.
    s.hand_in_hand_mask = frozenset({0, 1})
    s.folded_this_hand = frozenset({1})
    s.sitting_out_seats = frozenset({1})
    s.last_hero_hole = (1, 2, 3, 4, 5)
    s._pending_anchor_fs = object()
    s._pending_stable_ticks = 9

    r = server.pokernow_runner
    r.handle_payload(_hu(hero_cards=H1, button=6, flop=False, hero_bet=6.0,
                         vil_bet=6.0))  # pre-flop: no hand-start yet
    assert server._LIVE_SOURCE == "pokernow"
    assert not s.hand_in_hand_mask and s.folded_this_hand == frozenset()
    assert s.last_hero_hole is None
    assert s._pending_anchor_fs is None and s._pending_stable_ticks == 0

    r.handle_payload(_hu(hero_cards=H1, button=6, hero_actor=True))
    assert s.hand_in_hand_mask == frozenset({0, 1})
    # Same source again: no reset (the live hand survives).
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_bet=9.0, vil_actor=True,
                         hero_stack=77.0))
    assert s.hand_in_hand_mask == frozenset({0, 1}) and _log() == [(RAISE, 18)]


def test_f10_a_stray_ocr_stop_does_not_reset_a_pokernow_hand():
    s = server.session
    r = server.pokernow_runner
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_actor=True))
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_bet=9.0, vil_actor=True,
                         hero_stack=77.0))
    assert _log() == [(RAISE, 18)]

    server.ocr_runner._retire_reconstructor()  # what /ocr/stop ends with
    assert server._LIVE_SOURCE == "pokernow"
    assert server._active_reconstructor() is r._reconstructor

    r.handle_payload(_hu(hero_cards=H1, button=6, hero_bet=9.0, vil_bet=30.0,
                         hero_actor=True, hero_stack=77.0, vil_stack=32.0))
    assert s.hand_in_hand_mask == frozenset({0, 1})
    assert _log() == [(RAISE, 18), (RAISE, 60)]


# --- H7: the DOM mirror never locks a duplicate ------------------------------------------------


def test_h7_dom_card_colliding_with_a_user_override_is_not_locked():
    from plo5bp.ocr.types import Card

    s = server.session
    r = server.pokernow_runner
    r.handle_payload(_hu(hero_cards=H1, button=6, hero_actor=True))
    turn_a = Card.parse("8c")
    turn_a_idx = turn_a.rank * 4 + turn_a.suit
    # The user typed the (future) board-A turn card into board B's turn slot.
    s.turn_cards = [None, turn_a_idx]
    s._card_slot_locked["turn_cards"] = [False, True]

    frame = _hu(hero_cards=H1, button=6, hero_actor=True, pot=13.0)
    frame["boards"] = [{"run": "1", "cards": B1 + ["8c"]},
                       {"run": "2", "cards": B2 + ["Td"]}]
    r.handle_payload(frame)
    assert s.turn_cards == [None, turn_a_idx]  # the duplicate was not committed
    assert s._card_slot_locked["turn_cards"] == [False, True]
    assert r.last_error is None  # used to be "rebuild failed: duplicate card"
