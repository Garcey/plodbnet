"""Record / replay of live sessions (TEST-028).

* A ClubGG and a PokerNow session recorded through the real runners replay
  tick-for-tick identically (events AND action log).
* Every recording kept under ``tests/ocr/fixtures/sessions/`` replays as it
  was recorded — drop a ``PLO5BP_LIVE_RECORD`` file there to pin a live bug
  (see that folder's README).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from plo5bp.config import GameConfig  # noqa: E402
from plo5bp.ocr.events import EventReconstructor  # noqa: E402
from plo5bp.ocr.types import Card, FrameState, SeatObs  # noqa: E402
from plo5bp.ui import server  # noqa: E402
from plo5bp.ui.live import clubgg, pokernow as pn_live, record, replay, tracking  # noqa: E402
from plo5bp.ui.live.state import live_state  # noqa: E402

SESSIONS = Path(__file__).parent / "fixtures" / "sessions"


@pytest.fixture(autouse=True)
def _restore_session():
    yield
    s = server.session
    s.variant = server.VARIANT_PLO5
    s.game_config = GameConfig(starting_stack=400000)
    s.num_seats, s.button_seat, s.hero_seat = 6, 0, 0
    s.dollars_per_bb = 2.0
    live_state.simple_ocr_mode = True
    server._new_session_defaults()
    tracking._reset_live_tracking()
    tracking._set_active_reconstructor(None)
    tracking._note_live_source(None)
    pn_live.pokernow_runner._reconstructor = None
    server._rebuild_env()


@pytest.fixture()
def recorder(tmp_path, monkeypatch):
    rec = record.LiveRecorder(tmp_path)
    monkeypatch.setattr(record, "RECORDER", rec)
    yield rec
    rec.close()


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


# --- ClubGG --------------------------------------------------------------------------

FLOP_A = (Card(10, 0), Card(8, 0), Card(11, 0), None, None)
FLOP_B = (Card(5, 1), Card(6, 2), Card(1, 1), None, None)


def _ocr_frame(sb_stack: int) -> FrameState:
    stacks = {0: 34000, 1: 145513, 2: sb_stack, 4: 24650}
    seats = tuple(
        SeatObs(seat=i, stack_chips=stacks.get(i), committed_chips=None,
                folded=i not in stacks)
        for i in range(6)
    )
    return FrameState(board_a=FLOP_A, board_b=FLOP_B, hero_hole=(None,) * 5,
                      button_seat=2, pot_total_chips=None, seats=seats)


def test_a_clubgg_session_replays_identically(recorder):
    s = server.session
    s.variant = server.VARIANT_PLO5
    s.game_config, s.dollars_per_bb = GameConfig(), 20.0
    s.num_seats, s.button_seat, s.hero_seat = 6, 0, 0
    server._new_session_defaults()
    tracking._reset_live_tracking()
    server._rebuild_env()
    runner = clubgg.OcrRunner()
    runner._reconstructor = EventReconstructor(num_seats=6)
    tracking._set_active_reconstructor(runner._reconstructor)
    live_state.simple_ocr_mode = False  # action tracking on
    path = recorder.open("ocr")
    for f in (_ocr_frame(163513), _ocr_frame(145513), _ocr_frame(145513)):
        runner.process_frame(f)  # seat 2 bets $180 inside the debounce window
    recorder.close()
    live_log = [dict(e) for e in s.action_log]
    assert [(e["seat"], e["chips"]) for e in live_log if e["gate"] == 2] == [(2, 90000)]

    lines = _lines(path)
    assert lines[0]["kind"] == "header" and lines[0]["source"] == "ocr"
    ticks = lines[1:]
    assert [t["i"] for t in ticks] == [0, 1, 2]
    assert all(t["frame"] and t["simple"] is False for t in ticks)
    assert any(e["type"] == "SeatAction" for t in ticks for e in t["events"])
    assert ticks[-1]["log"] == live_log

    first, recorded, replayed = replay.compare(path)
    assert first is None
    assert replayed[-1]["log"] == json.loads(json.dumps(live_log))


# --- PokerNow --------------------------------------------------------------------------

HERO = ["Ts", "As", "4h", "3d", "2d"]
B1, B2 = ["4c", "Jc", "Ad"], ["9h", "2s", "Kd"]


def _pn(*, vil_bet=None, hero_actor=False, vil_actor=False, vil_stack=62.0):
    def seat(no, name, *, hero, actor, angle, stack, cards, bet):
        return {"seat": no, "name": name, "isHero": hero, "isActor": actor,
                "angleCW": angle, "stackDollars": stack, "folded": False,
                "betText": None, "betDollars": bet, "cards": cards}
    return {
        "schema": "pokernow.v1", "variant": "plo5", "bombPot": True,
        "potDollars": 12.0, "button": {"seat": 6},
        "boards": [{"run": "1", "cards": B1}, {"run": "2", "cards": B2}],
        "heroCards": HERO,
        "seats": [
            seat(1, "Miles", hero=True, actor=hero_actor, angle=147,
                 stack=86.0, cards=HERO, bet=None),
            seat(6, "JJ", hero=False, actor=vil_actor, angle=328,
                 stack=vil_stack, cards=[None] * 5, bet=vil_bet),
        ],
    }


def test_a_pokernow_session_replays_identically(recorder):
    s = server.session
    s.variant = server.VARIANT_PLO5
    s.game_config = GameConfig(num_seats=2, starting_stack=400, ante=12, bb=2)
    s.num_seats, s.button_seat, s.hero_seat = 2, 0, 0
    s.dollars_per_bb = 1.0
    server._new_session_defaults()
    tracking._reset_live_tracking()
    server._rebuild_env()
    runner = pn_live.PokerNowRunner()
    for p in (_pn(hero_actor=True),
              _pn(vil_actor=True),                      # hero checked
              _pn(vil_bet=9.0, vil_stack=53.0, hero_actor=True)):
        runner.handle_payload(p)
    recorder.close()
    live_log = [dict(e) for e in s.action_log]
    assert [e["gate"] for e in live_log] == [1, 2]  # check, bet

    path = recorder.path
    lines = _lines(path)
    assert lines[0]["source"] == "pokernow"
    assert [t["payload"]["seats"][1]["betDollars"] for t in lines[1:]] == [None, None, 9.0]
    first, _, replayed = replay.compare(path)
    assert first is None
    assert replayed[-1]["log"] == json.loads(json.dumps(live_log))


def test_a_changed_pipeline_is_reported(recorder, tmp_path):
    s = server.session
    s.variant = server.VARIANT_PLO5
    s.game_config = GameConfig(num_seats=2, starting_stack=400, ante=12, bb=2)
    s.num_seats, s.dollars_per_bb = 2, 1.0
    server._new_session_defaults()
    tracking._reset_live_tracking()
    server._rebuild_env()
    runner = pn_live.PokerNowRunner()
    for p in (_pn(hero_actor=True), _pn(vil_actor=True)):
        runner.handle_payload(p)
    recorder.close()
    # Tamper with what was "recorded": the replay no longer matches tick 1.
    lines = _lines(recorder.path)
    lines[2]["log"] = []
    doctored = tmp_path / "doctored.jsonl"
    doctored.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    assert replay.main([str(doctored)]) == 1
    assert replay.compare(doctored)[0] == 1


def test_recording_is_off_unless_asked(monkeypatch):
    monkeypatch.setattr(record, "RECORDER", None)
    runner = pn_live.PokerNowRunner()
    s = server.session
    s.game_config = GameConfig(num_seats=2, starting_stack=400, ante=12, bb=2)
    s.num_seats, s.dollars_per_bb = 2, 1.0
    server._new_session_defaults()
    server._rebuild_env()
    runner.handle_payload(_pn(hero_actor=True))  # no recorder: nothing to do


# --- Owner recordings --------------------------------------------------------------------

_KEPT = sorted(SESSIONS.glob("*.jsonl")) if SESSIONS.exists() else []


@pytest.mark.parametrize("path", _KEPT or [None], ids=[p.name for p in _KEPT] or ["none"])
def test_kept_recordings_replay_as_recorded(path):
    if path is None:
        pytest.skip("no recordings in tests/ocr/fixtures/sessions/ yet")
    first, recorded, replayed = replay.compare(path)
    assert first is None, (
        f"{path.name}: tick {first} differs — recorded {recorded[first]}, "
        f"replayed {replayed[first]}"
    )
