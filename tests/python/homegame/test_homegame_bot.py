"""Home games — the network at a seat (2026-10-05; owner: "integrate the model into home
games ... This is something only I get access to do"): the site's owner lets the network
play their seat (``auto``) or suggest each move (``assist``), its favourite move or its
full strategy (``mix``); everybody at the table sees it (``seat.bot``, a line in the feed,
``action.bot`` in the history) and those decisions are never graded (``homegame_bot``).

Booted once for the module in PUBLIC mode against a temp DB. The network is the test
session's random placeholder model — it plays legal moves, which is what these check.
"""

from __future__ import annotations

import sys
import time

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"
NAMES = ["kai", "lee"]
OWNER_SEAT = 1


@pytest.fixture(scope="module")
def server(boot_public_server):
    srv = boot_public_server(PLO5BP_ADMIN_EMAILS=ADMIN_EMAIL)
    # (no checkpoint in the tests: the placeholder plays — production never lets it)
    sys.modules["plo5bp.ui.homegame"].set_model_provider(lambda: srv.MODEL)
    return srv


@pytest.fixture(scope="module")
def hg(server):
    return sys.modules["plo5bp.ui.homegame"]


@pytest.fixture(scope="module")
def cast(server):
    def login(email):
        cl = TestClient(server.app, raise_server_exceptions=False)
        assert cl.get("/auth/dev", params={"email": email}).status_code == 200
        return cl

    owner = login(ADMIN_EMAIL)
    players = [login(f"{n}@example.com") for n in NAMES]
    ids = {u["email"]: u["id"] for u in owner.get("/admin/api/users").json()["users"]}
    for email in [ADMIN_EMAIL] + [f"{n}@example.com" for n in NAMES]:
        owner.post("/admin/api/games_access", json={"user_id": ids[email], "action": "grant"})
    return {"owner": owner, "p": players, "owner_id": ids[ADMIN_EMAIL]}


@pytest.fixture(autouse=True)
def _quick(hg, monkeypatch):
    monkeypatch.setattr(hg, "BOT_DELAY_S", 0.0)  # (autopilot acts at the next tick)


def _post(cl, gid, what, body=None):
    return cl.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _state(cl, gid):
    return cl.get(f"/games/api/tables/{gid}").json()


def _table(cast, variant="plo5"):
    """kai hosts at seat 0, the owner sits at seat 1 (manual dealing: `run` deals hand 1)."""
    gid = cast["p"][0].post("/games/api/tables", json={
        "name": "network", "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 20000,
        "variant": variant}).json()["id"]
    assert _post(cast["owner"], gid, "sit", {"seat": OWNER_SEAT, "buyin_cents": 20000}).status_code == 200
    return gid


def _tick_until_played(hg, t, timeout=20.0):
    """The clock thread's autopilot step, until the network has played its turn."""
    with t.lock:
        before = (t.hand_no, t.action_seq)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with t.lock:
            if (t.hand_no, t.action_seq) != before:
                return
            hg._bot_tick_locked(t)
        time.sleep(0.02)
    raise AssertionError("the network never played its turn")


def _play_out(cast, hg, gid):
    """Play hand to its end: the network at the owner's seat, kai checks / calls. Returns
    the decisions the network made: [(key, decision)]."""
    t = hg.HUB.get(gid)
    made = []
    for _ in range(40):
        s = _state(cast["p"][0], gid)
        if s["phase"] != "in_hand" or s["actor"] is None:
            break
        if s["actor"] == OWNER_SEAT:
            with t.lock:
                key = (t.hand_no, t.action_seq)
            _tick_until_played(hg, t)
            with t.lock:
                made.append((key, dict(t.bot_cache[key]) if key in t.bot_cache else None))
        else:
            assert _post(cast["p"][0], gid, "act", {"gate": "check_call"}).status_code == 200
    with t.lock:  # (a checked-down showdown: skip its award animation, so the hand is published)
        if t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0
    return made


def test_only_the_sites_owner_can_let_the_network_play(cast, hg, monkeypatch):
    gid = _table(cast)
    assert _post(cast["p"][1], gid, "sit", {"seat": 2, "buyin_cents": 20000}).status_code == 200
    # a friend can't — and never sees the owner's settings in their view
    r = _post(cast["p"][1], gid, "bot", {"mode": "auto"})
    assert r.status_code == 403
    assert _state(cast["p"][1], gid)["bot"] is None
    mine = _state(cast["owner"], gid)["bot"]
    assert mine == {"available": True, "why": None, "mode": "", "mix": False}
    # not seated / no network loaded / not PLO5: refused, with the reason
    gid2 = cast["p"][0].post("/games/api/tables", json={"name": "x", "bb_cents": 100, "ante_cents": 300,
                                                        "default_buyin_cents": 20000}).json()["id"]
    assert _post(cast["owner"], gid2, "bot", {"mode": "auto"}).json()["detail"] == "Take a seat first."
    with monkeypatch.context() as m:
        m.setattr(hg, "_grading_model", lambda: None)
        r = _post(cast["owner"], gid, "bot", {"mode": "assist"})
        assert r.status_code == 400 and "isn't loaded" in r.json()["detail"]
    gid6 = _table(cast, "plo6")
    r = _post(cast["owner"], gid6, "bot", {"mode": "auto"})
    assert r.status_code == 400 and "PLO5" in r.json()["detail"]
    assert _state(cast["owner"], gid6)["bot"]["available"] is False


def test_autopilot_plays_the_owners_turns_and_everybody_sees_it(cast, hg):
    gid = _table(cast)
    kai = cast["p"][0]
    r = _post(cast["owner"], gid, "bot", {"mode": "auto", "mix": False})
    assert r.status_code == 200 and r.json()["bot"] == {"available": True, "why": None, "mode": "auto", "mix": False}
    s = _state(kai, gid)
    assert s["seats"][OWNER_SEAT]["bot"] == "auto" and s["seats"][0]["bot"] is None
    lines = [e["text"] for e in s["events"] if e["kind"] == "bot"]
    assert lines and lines[-1].startswith("The network is playing ") and lines[-1].endswith("'s seat.")
    # the table hears the mode only (owner, 2026-10-05): a change of strategy says nothing
    assert _post(cast["owner"], gid, "bot", {"mode": "auto", "mix": True}).status_code == 200
    assert [e["text"] for e in _state(kai, gid)["events"] if e["kind"] == "bot"] == lines
    assert not any(w in x for x in lines for w in ("favourite", "strategy", "mix"))
    assert _post(cast["owner"], gid, "bot", {"mode": "auto", "mix": False}).status_code == 200
    assert _post(kai, gid, "run", {"running": True}).status_code == 200
    # the owner's own view offers nothing to press while it plays — and a manual act is refused
    for _ in range(10):
        s = _state(cast["owner"], gid)
        if s["actor"] == OWNER_SEAT:
            break
        _post(kai, gid, "act", {"gate": "check_call"})
    assert s["actor"] == OWNER_SEAT and not any(s["legal"].values())
    r = _post(cast["owner"], gid, "act", {"gate": "check_call", "hand_no": s["hand_no"], "action_seq": s["action_seq"]})
    assert r.status_code == 409 and "network is playing" in r.json()["detail"]
    t = hg.HUB.get(gid)
    made = _play_out(cast, hg, gid)
    assert made and all(d is not None and "gate" in d for _k, d in made)
    with t.lock:  # (the network's moves: marked, never the player's own — never graded)
        mine = [(k, a) for k, a in enumerate(t.hand_actions) if a[0] == OWNER_SEAT]
        assert mine and all(a[3] is False and t.bot_marks.get(k) == "auto" for k, a in mine)
        assert [a[1] for _k, a in mine] == [d["gate"] for _key, d in made]
    det = kai.get(f"/games/api/tables/{gid}/hands/1").json()
    rows = [a for a in det["actions"] if a["seat"] == OWNER_SEAT]
    assert rows and all(a["bot"] == "auto" and a["auto"] is False for a in rows)
    assert all("bot" not in a for a in det["actions"] if a["seat"] != OWNER_SEAT)
    # off again: said to the table, and the owner's buttons are back
    _post(cast["owner"], gid, "bot", {"mode": "off"})
    s = _state(kai, gid)
    assert s["seats"][OWNER_SEAT]["bot"] is None
    assert any(e["kind"] == "bot" and "on their own again" in e["text"] for e in s["events"])


def test_assist_shows_the_networks_move_and_the_owner_plays_it(cast, hg):
    gid = _table(cast)
    kai, owner = cast["p"][0], cast["owner"]
    assert _post(owner, gid, "bot", {"mode": "assist"}).status_code == 200
    assert _state(kai, gid)["seats"][OWNER_SEAT]["bot"] == "assist"
    assert _post(kai, gid, "run", {"running": True}).status_code == 200
    for _ in range(10):
        s = _state(owner, gid)
        if s["actor"] == OWNER_SEAT:
            break
        _post(kai, gid, "act", {"gate": "check_call"})
    assert s["actor"] == OWNER_SEAT and any(s["legal"].values())  # (the owner still presses the button)
    url = f"/games/api/tables/{gid}/bot/suggest"
    assert kai.get(url).status_code == 403
    sug = owner.get(url).json()
    assert (sug["hand_no"], sug["action_seq"]) == (s["hand_no"], s["action_seq"])
    assert s["legal"][sug["gate"]] and abs(sum(sug["probs"].values()) - 1.0) < 1e-3 and sug["mix"] is False
    # its favourite move — the Trainer's recommendation at this very spot
    t = hg.HUB.get(gid)
    with t.lock:
        job = hg._bot_job_locked(t)
    from plo5bp.ui import trainer as tr
    _env, obs, info = hg._bot_replay(job)
    model = hg._grading_model()
    rec = tr.compute_node_distribution(model, next(model.parameters()).device, obs, info)
    assert sug["gate"] == {0: "fold", 1: "check_call", 2: "raise"}[rec["rec_gate"]]
    assert owner.get(url).json() == sug  # (one answer per decision)
    body = {"gate": sug["gate"], "hand_no": s["hand_no"], "action_seq": s["action_seq"]}
    if sug["gate"] == "raise":
        body["raise_to_chips"] = sug["raise_to_chips"]
    assert _post(owner, gid, "act", body).status_code == 200
    with t.lock:
        k = max(k for k, a in enumerate(t.hand_actions) if a[0] == OWNER_SEAT)
        assert t.bot_marks[k] == "assist" and t.hand_actions[k][3] is False  # (not graded)
    assert owner.get(url).status_code == 409  # (no longer their turn)


def test_its_full_strategy_draws_one_move_per_decision(cast, hg):
    gid = _table(cast)
    kai, owner = cast["p"][0], cast["owner"]
    assert _post(owner, gid, "bot", {"mode": "assist", "mix": True}).status_code == 200
    assert _post(kai, gid, "run", {"running": True}).status_code == 200
    for _ in range(10):
        s = _state(owner, gid)
        if s["actor"] == OWNER_SEAT:
            break
        _post(kai, gid, "act", {"gate": "check_call"})
    url = f"/games/api/tables/{gid}/bot/suggest"
    first = owner.get(url).json()
    assert first["mix"] is True and s["legal"][first["gate"]]
    assert all(owner.get(url).json() == first for _ in range(4))  # (a reload never redraws)
    # a new setting is a new draw
    assert _post(owner, gid, "bot", {"mode": "assist", "mix": False}).status_code == 200
    assert owner.get(url).json()["mix"] is False


def test_a_move_it_cannot_work_out_checks_or_folds_instead_of_stalling(cast, hg, monkeypatch):
    gid = _table(cast)
    kai, owner = cast["p"][0], cast["owner"]
    assert _post(owner, gid, "bot", {"mode": "auto"}).status_code == 200
    assert _post(kai, gid, "run", {"running": True}).status_code == 200

    def broken(job, model, mix):
        raise RuntimeError("a replay that diverged")

    monkeypatch.setattr(hg, "_bot_decide", broken)
    for _ in range(10):
        s = _state(owner, gid)
        if s["actor"] == OWNER_SEAT:
            break
        _post(kai, gid, "act", {"gate": "check_call"})
    t = hg.HUB.get(gid)
    with t.lock:
        free = t.info.gate_mask[1] and not (s["to_call_cents"] > 0)
    _tick_until_played(hg, t)
    with t.lock:
        k = max(k for k, a in enumerate(t.hand_actions) if a[0] == OWNER_SEAT)
        assert t.bot_marks[k] == "auto" and t.hand_actions[k][1] == (1 if free else 0)


def test_the_owners_view_holds_up_with_a_cold_user_cache(cast, hg, server):
    """(found in the preview) A live stream builds the view with no request in front of
    it: the sign-in check's user cache has run out — the owner check reads the database."""
    pub = sys.modules["plo5bp.ui.public"]
    gid = _table(cast)
    t = hg.HUB.get(gid)
    with pub._USER_CACHE_LOCK:
        pub._USER_CACHE.clear()
    hg.CTX.owner_cache.clear()
    with t.lock:
        mine = hg._view(t, cast["owner_id"])["bot"]
        theirs = hg._view(t, t.host_user_id)["bot"]
    assert mine is not None and mine["available"] is True and theirs is None


def test_switched_on_late_in_a_turn_it_plays_at_once(cast, hg, monkeypatch):
    """(found in the preview) Its thinking delay counts from the turn's start: switched on
    with the clock nearly out, it plays before the clock does."""
    monkeypatch.setattr(hg, "BOT_DELAY_S", 5.0)
    gid = cast["p"][0].post("/games/api/tables", json={
        "name": "late", "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 20000,
        "decision_secs": 30}).json()["id"]
    kai, owner = cast["p"][0], cast["owner"]
    assert _post(owner, gid, "sit", {"seat": OWNER_SEAT, "buyin_cents": 20000}).status_code == 200
    assert _post(kai, gid, "run", {"running": True}).status_code == 200
    for _ in range(10):
        s = _state(owner, gid)
        if s["actor"] == OWNER_SEAT:
            break
        _post(kai, gid, "act", {"gate": "check_call"})
    t = hg.HUB.get(gid)
    with t.lock:
        assert t.turn_started_mono is not None
        t.turn_started_mono -= 28.0  # (two seconds left on the clock)
    assert _post(owner, gid, "bot", {"mode": "auto"}).status_code == 200
    _tick_until_played(hg, t, timeout=4.0)  # (well inside the 5 s delay counted from now)
    with t.lock:
        k = max(k for k, a in enumerate(t.hand_actions) if a[0] == OWNER_SEAT)
        assert t.bot_marks[k] == "auto"
