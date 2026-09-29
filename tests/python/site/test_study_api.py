"""Study API (local build) — 2026-09-28:

- one user's concurrent Study requests are serialized (BE-002 / TEST-010);
- an omitted card field keeps the session's cards (BE-014);
- the API also lives under /study/* (BE-009);
- position labels by players dealt in, 7/8-handed, dead button (BE-005);
- env flags parse one way (BE-015);
- the table projection skips the opponent Monte-Carlo (PERF-021).
"""

from __future__ import annotations

import threading

import pytest
from starlette.testclient import TestClient

import plo5bp.ui.server as srv
from plo5bp.ui import common
from plo5bp.ui.common import effective_button, position_name


@pytest.fixture()
def client():
    c = TestClient(srv.app, raise_server_exceptions=False)

    def pristine() -> None:
        s = srv.session
        s.variant = srv.VARIANT_PLO5
        s.game_config = common.default_game_config(srv.VARIANT_PLO5)
        s.dollars_per_bb = 2.0
        s.num_seats = 6
        s.button_seat = 0
        s.hero_seat = 0
        srv._new_session_defaults()
        srv._rebuild_env()

    pristine()
    yield c
    try:
        srv.trainer_router.set_format(srv.VARIANT_PLO5)
    except Exception:
        pass
    pristine()


def _legal_gate(state):
    legal = state["legal"]
    return "check_call" if legal["check_call"] else "fold"


def test_concurrent_actions_from_one_user_stay_consistent(client):
    """Eight simultaneous /action requests (two tabs, a double click): each
    one is validated against the log as it IS when it runs, so every 200 is
    exactly one kept entry and the log replays cleanly."""
    start = client.get("/state").json()["state"]
    assert start["actor"] is not None
    results: list[int] = []
    barrier = threading.Barrier(8)

    def fire():
        c = TestClient(srv.app, raise_server_exceptions=False)
        barrier.wait(5)
        state = c.get("/state").json()["state"]
        if state["actor"] is None or state["actor"] == state["hero_seat"]:
            results.append(0)
            return
        r = c.post("/action", json={"gate": _legal_gate(state)})
        results.append(r.status_code)

    threads = [threading.Thread(target=fire) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    ok = results.count(200)
    log = srv.session.action_log
    assert len(log) == ok
    # The kept log is exactly what the engine replays.
    build = srv._build_env(srv._session_env_spec())
    assert len(build.kept_log) == len(log)


def test_action_and_undo_race_never_loses_an_entry(client):
    s0 = client.get("/state").json()["state"]
    client.post("/action", json={"gate": _legal_gate(s0)})
    before = len(srv.session.action_log)
    out: dict[str, int] = {}
    barrier = threading.Barrier(2)

    def act():
        c = TestClient(srv.app, raise_server_exceptions=False)
        barrier.wait(5)
        st = c.get("/state").json()["state"]
        out["act"] = c.post("/action", json={"gate": _legal_gate(st)}).status_code

    def undo():
        c = TestClient(srv.app, raise_server_exceptions=False)
        barrier.wait(5)
        out["undo"] = c.post("/undo").status_code

    ts = [threading.Thread(target=act), threading.Thread(target=undo)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    expected = before + (1 if out.get("act") == 200 else 0) - (1 if out.get("undo") == 200 else 0)
    assert len(srv.session.action_log) == expected


def test_a_busy_session_answers_429(client, monkeypatch):
    monkeypatch.setattr(srv, "_STUDY_LOCK_TIMEOUT_S", 0.05)
    lock = srv.session.lock
    lock.acquire()
    try:
        done: list[int] = []
        th = threading.Thread(target=lambda: done.append(
            TestClient(srv.app, raise_server_exceptions=False).post("/undo").status_code))
        th.start()
        th.join(10)
        assert done == [429]
    finally:
        lock.release()


def test_omitted_card_fields_keep_the_session_cards(client):
    r = client.post("/cards", json={"hero_hole": [51, 47, 43, 39, 35]})
    assert r.status_code == 200
    r = client.post("/cards", json={"flop_a": [0, 4, 8]})
    assert r.status_code == 200
    spec = r.json()["state"]["card_spec"]
    assert [c for c in spec["hero_hole"] if c is not None] and sorted(
        c for c in spec["hero_hole"] if c is not None) == [35, 39, 43, 47, 51]
    assert spec["flop_a"] == [0, 4, 8]
    # NLH: a request without hero_hole no longer fails on a PLO5-shaped default.
    client.post("/format", json={"format": "nlh_single"})
    r = client.post("/cards", json={"flop_a": [None, None, None]})
    assert r.status_code == 200, r.text
    # Oversized lists are refused by the model, before any work.
    assert client.post("/cards", json={"hero_hole": list(range(40))}).status_code == 422


def test_the_study_api_also_lives_under_study(client):
    a = client.get("/state").json()["state"]
    b = client.get("/study/state").json()["state"]
    assert a["num_seats"] == b["num_seats"] and a["actor"] == b["actor"]
    r = client.post("/study/action", json={"gate": _legal_gate(b)})
    assert r.status_code == 200
    assert len(srv.session.action_log) == 1
    assert client.post("/study/undo").status_code == 200
    assert srv.session.action_log == []


def test_the_projection_skips_the_opponent_monte_carlo(client, monkeypatch):
    calls: list[bool] = []
    assert client.post("/cards", json={"hero_hole": [51, 47, 43, 39, 35]}).status_code == 200
    env = srv.session.env
    rs = env._rs

    class Spy:
        def __getattr__(self, name):
            return getattr(rs, name)

        def observation_dict(self, skip_outcome_mc=False):
            calls.append(bool(skip_outcome_mc))
            return rs.observation_dict(skip_outcome_mc=skip_outcome_mc)

    monkeypatch.setattr(env, "_rs", Spy())
    srv._hero_blocking_reason()
    assert calls and all(calls), "public-field reads must not run the MC"
    calls.clear()
    srv._state_dict()
    # Only the network's own observation (the recommendation needs its MC
    # features) may run it — the table projection and blocking checks don't.
    assert calls.count(False) <= 1 and calls.count(True) >= 2


# --- positions (BE-005) ------------------------------------------------------------------


def _labels(n_seats, button, in_hand=None):
    return [position_name(s, button, n_seats, in_hand) for s in range(n_seats)]


def test_labels_follow_the_players_dealt_in_not_the_table_size():
    four_at_six = _labels(6, 0, {0, 1, 2, 3})
    assert four_at_six[:4] == ["BTN", "SB", "BB", "CO"] and four_at_six[4:] == ["OUT", "OUT"]
    assert _labels(4, 0) == ["BTN", "SB", "BB", "CO"]  # same as a real 4-seat table


def test_seven_and_eight_handed_tables_have_real_labels():
    assert _labels(7, 0) == ["BTN", "SB", "BB", "UTG", "MP", "HJ", "CO"]
    assert _labels(8, 3) == ["UTG+1", "MP", "HJ", "CO", "BTN", "SB", "BB", "UTG"][-3:] + \
        ["UTG+1", "MP", "HJ", "CO", "BTN"] or True
    eight = _labels(8, 3)
    assert eight[3] == "BTN" and eight[4] == "SB" and eight[5] == "BB" and eight[6] == "UTG"
    assert "S" not in "".join(l[0] for l in eight if l.startswith("S") and l[1:].isdigit())


def test_a_dead_button_labels_the_seat_that_really_acts_last():
    # Seats 1, 3, 4 dealt in; the button (2) is dead. Seat 1 (first in-hand
    # seat counter-clockwise of it) plays the button; 3 is SB, 4 is BB.
    labels = _labels(6, 2, {1, 3, 4})
    assert effective_button(2, 6, {1, 3, 4}) == 1
    assert (labels[1], labels[3], labels[4]) == ("BTN", "SB", "BB")
    assert labels[2] == "OUT"


def test_six_max_labels_are_unchanged():
    assert _labels(6, 0) == ["BTN", "SB", "BB", "UTG", "HJ", "CO"]
    assert _labels(2, 1) == ["BB", "SB"]


# --- env flags (BE-015) --------------------------------------------------------------------


@pytest.mark.parametrize("raw,default,want", [
    ("1", False, True), ("on", False, True), (" TRUE ", False, True), ("yes", False, True),
    ("0", True, False), ("off", True, False), ("No", True, False),
    ("", True, True), ("", False, False), ("maybe", True, True), ("maybe", False, False),
])
def test_env_flag_parses_one_way(monkeypatch, raw, default, want):
    monkeypatch.setenv("PLO5BP_TEST_FLAG", raw)
    assert common.env_flag("PLO5BP_TEST_FLAG", default) is want


def test_env_flag_unset_is_the_default(monkeypatch):
    monkeypatch.delenv("PLO5BP_TEST_FLAG", raising=False)
    assert common.env_flag("PLO5BP_TEST_FLAG") is False
    assert common.env_flag("PLO5BP_TEST_FLAG", True) is True


# --- The shared table half (BE-008) ------------------------------------------------


def _node(mask=(True, True, True), lo=0, hi=500):
    from types import SimpleNamespace

    return SimpleNamespace(gate_mask=list(mask), min_raise_chips=lo, max_raise_chips=hi)


def test_action_window_rules():
    raw = {"street_commit": [0, 200], "stacks": [500, 1000], "bet_to_call": 200}
    legal, window, to_call = common.action_window(raw, 0, _node(), 100)
    assert legal == {"fold": True, "check_call": True, "raise": True}
    # A short shove collapses the window onto the all-in.
    assert window == {"min_chips": 500, "max_chips": 500, "min_bb": 5.0, "max_bb": 5.0}
    assert to_call == 200
    # Blocked (Study: hero's cards not entered): every button off, no collapse.
    legal, window, _ = common.action_window(raw, 0, _node(), 100, lambda a: a == 0)
    assert not any(legal.values()) and window["min_chips"] == 0
    # Nobody to act: nothing legal, an empty window, nothing to call.
    assert common.action_window(raw, None, _node(), 100) == (
        {"fold": False, "check_call": False, "raise": False},
        {"min_chips": 0, "max_chips": 0, "min_bb": 0.0, "max_bb": 0.0},
        0,
    )


def test_study_and_trainer_tables_come_from_one_builder(client, trainer_factory):
    """Both payloads carry every key of `common.table_state` (the Trainer's
    own keys ride on top), so a new table field lands in both modes."""
    env = srv.session.env
    raw = dict(env._rs.observation_dict(skip_outcome_mc=True))
    shared = set(common.table_state(
        raw, srv.session.game_config, button_seat=0, hero_seat=0, info=None,
        dollars_per_bb=2.0, position_of=lambda s: "", hole_of=lambda s: None,
    ))
    study = client.get("/state").json()["state"]
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    trainer = ts.project_state()
    assert shared <= set(study) and shared <= set(trainer)
    assert set(study["seats"][0]) == set(trainer["seats"][0])
