"""Regression tests for the 2026-09-20 review — home games (G1-G14).

Each test names the finding it pins. The app is booted once for the module in
PUBLIC mode against a temp DB (`boot_public_server`, tests/python/conftest.py).
"""

from __future__ import annotations

import json
import random
import sys
import time

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"
NAMES = ["alice", "bob", "carol", "dave", "erin", "frank"]


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server()


@pytest.fixture(scope="module")
def hg(server):
    return sys.modules["plo5bp.ui.homegame"]


@pytest.fixture(scope="module")
def cast(server):
    """admin + six granted players + one granted spectator (never sits)."""
    def login(email):
        c = TestClient(server.app, raise_server_exceptions=False)
        assert c.get("/auth/dev", params={"email": email}).status_code == 200
        return c

    adm = login(ADMIN_EMAIL)
    players = [login(f"{n}@example.com") for n in NAMES]
    spec = login("spectator@example.com")
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for email in [f"{n}@example.com" for n in NAMES] + ["spectator@example.com"]:
        r = adm.post("/admin/api/games_access", json={"user_id": ids[email], "action": "grant"})
        assert r.status_code == 200
    by_uid = {ids[f"{n}@example.com"]: players[i] for i, n in enumerate(NAMES)}
    return {"adm": adm, "p": players, "spec": spec, "ids": ids, "by_uid": by_uid}


# --- helpers -------------------------------------------------------------------


def _create(client, **kw):
    body = {"name": "review table", "sb_cents": 50, "bb_cents": 100,
            "ante_cents": 300, "default_buyin_cents": 4000}
    body.update(kw)
    r = client.post("/games/api/tables", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _state(client, gid):
    r = client.get(f"/games/api/tables/{gid}")
    assert r.status_code == 200, r.text
    return r.json()


def _post(client, gid, what, body=None):
    return client.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _table(cast, n, **kw):
    """Table hosted by player 0 with players 0..n-1 in seats 0..n-1."""
    p = cast["p"]
    gid = _create(p[0], **kw)["id"]
    for i in range(1, n):
        assert _post(p[i], gid, "sit", {"seat": i, "buyin_cents": 4000}).status_code == 200
    return gid


def _start(cast, gid):
    r = _post(cast["p"][0], gid, "run", {"running": True})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["phase"] == "in_hand", s["phase"]
    return s


def _actor(cast, gid):
    """(client, state-as-seen-by-the-actor) or (None, state) when no hand."""
    s = _state(cast["p"][0], gid)
    if s["phase"] != "in_hand" or s["actor"] is None:
        return None, s
    cl = cast["by_uid"][s["seats"][s["actor"]]["user_id"]]  # seats != player index after re-sits
    return cl, _state(cl, gid)


def _act(cl, gid, **body):
    r = _post(cl, gid, "act", body)
    assert r.status_code == 200, r.text
    return r.json()


def _min_raise_to(s):
    return s["raise_bounds"]["min_chips"] + s["street_commit_chips"]


def _max_raise_to(s):
    return s["raise_bounds"]["max_chips"] + s["street_commit_chips"]


def _play_out(cast, gid, *, folders=(), jam=False, limit=80):
    """Drive the hand to its end. `folders` fold when facing a bet (check when
    free); everyone else jams (`jam`) or check/calls."""
    for _ in range(limit):
        cl, s = _actor(cast, gid)
        if cl is None:
            return s
        if s["actor"] in folders:
            gate = "fold" if s["legal"]["fold"] else "check_call"
            _act(cl, gid, gate=gate)
        elif jam and s["legal"]["raise"]:
            _act(cl, gid, gate="raise", raise_to_chips=_max_raise_to(s))
        else:
            _act(cl, gid, gate="check_call")
    raise AssertionError("hand did not end")


def _live(hg, gid):
    return hg.HUB.get(gid)


def _finish_runout(hg, gid):
    t = _live(hg, gid)
    with t.lock:
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0


def _nets(state):
    return {r["name"]: r["net_cents"] for r in state["ledger"]}


# --- G1: uncontested pots stay face-down ---------------------------------------------


def test_g1_fold_out_does_not_reveal_the_winner(cast, hg):
    gid = _table(cast, 2)
    _start(cast, gid)
    cl, s = _actor(cast, gid)
    winner = s["actor"]
    folder = 1 - winner
    _act(cl, gid, gate="raise", raise_to_chips=_min_raise_to(s))
    end = _act(cast["p"][folder], gid, gate="fold")
    assert end["phase"] == "showdown" and end["can_rabbit"] is True
    for viewer in (cast["p"][folder], cast["spec"]):
        seen = _state(viewer, gid)["seats"][winner]
        assert seen["hole"] == [-1] * 5, "a successful bluff must stay face-down"
        assert seen["hand_desc"] is None
    own = _state(cast["p"][winner], gid)["seats"][winner]
    assert own["hole"][0] >= 0 and own["hand_desc"]
    t = _live(hg, gid)
    assert t.showdown_reveal is False and all(h is None for h in t.last_holes)


def test_g1_real_showdown_still_reveals_live_hands_only(cast):
    gid = _table(cast, 3)
    _start(cast, gid)
    cl, s = _actor(cast, gid)
    bettor = s["actor"]
    _act(cl, gid, gate="raise", raise_to_chips=_min_raise_to(s))
    cl, s = _actor(cast, gid)
    folder = s["actor"]
    _act(cl, gid, gate="fold")
    end = _play_out(cast, gid)
    assert end["phase"] == "showdown"
    view = _state(cast["spec"], gid)
    for seat in view["seats"][:3]:
        if seat["seat"] == folder:
            assert seat["hole"] == [-1] * 5
        else:
            assert seat["hole"][0] >= 0 and seat["hand_desc"]
    assert bettor != folder


# --- G2: own cards = the user who was DEALT them -----------------------------------------


def test_g2_seated_but_not_dealt_player_sees_no_cards(cast, hg):
    gid = _table(cast, 3)
    assert _post(cast["p"][2], gid, "sit_out", {"on": True}).status_code == 200
    s = _start(cast, gid)
    assert [x["in_hand"] for x in s["seats"][:3]] == [True, True, False]
    carol = _state(cast["p"][2], gid)
    assert carol["seats"][2]["hole"] is None  # the engine dealt seat 2 five dead cards
    assert carol["seats"][2]["hand_desc"] is None
    assert carol["seats"][0]["hole"] == [-1] * 5
    dead = sorted(_live(hg, gid).env.all_hole_cards()[2])
    assert all(sorted(x["hole"] or []) != dead for x in carol["seats"])
    # Dealt players still see exactly their own hand.
    assert _state(cast["p"][0], gid)["seats"][0]["hole"][0] >= 0
    assert _state(cast["p"][0], gid)["seats"][1]["hole"] == [-1] * 5


def test_g2_seat_reuse_does_not_reveal_the_mucked_hand(cast, hg):
    gid = _table(cast, 3)
    _start(cast, gid)
    cl, s = _actor(cast, gid)
    _act(cl, gid, gate="raise", raise_to_chips=_min_raise_to(s))
    cl, s = _actor(cast, gid)
    folder = s["actor"]
    mucked = s["seats"][folder]["hole"]
    assert mucked[0] >= 0
    _act(cl, gid, gate="fold")
    _play_out(cast, gid)
    _finish_runout(hg, gid)
    assert _post(cast["p"][folder], gid, "leave").status_code == 200
    # From the rail the dealt player still sees their own (mucked) hand ...
    assert _state(cast["p"][folder], gid)["seats"][folder]["hole"] == mucked
    assert _state(cast["spec"], gid)["seats"][folder]["hole"] == [-1] * 5
    newcomer = cast["p"][5]
    r = _post(newcomer, gid, "sit", {"seat": folder, "buyin_cents": 4000})
    assert r.status_code == 200, r.text
    seen = r.json()["seats"][folder]
    assert seen["is_hero"] is True
    # ... and the newcomer, who used to be shown those five cards as their
    # "own", gets nothing at all — the seat is theirs, the hand is not.
    assert seen["hole"] is None and seen["hand_desc"] is None
    for viewer in (cast["p"][folder], cast["spec"], cast["p"][0]):
        assert _state(viewer, gid)["seats"][folder]["hole"] is None


# --- G3 / G11: all-in runout ------------------------------------------------------------------


@pytest.fixture()
def allin(cast, hg, monkeypatch):
    """3-handed: seat 0 folds, seats 1+2 get it in on the flop. Slow runout."""
    calls = []
    real = hg.board_equities

    def counting(holes, ba, bb, **kw):
        calls.append((sorted(holes), len(ba), len(bb)))
        return real(holes, ba, bb, **kw)

    monkeypatch.setattr(hg, "board_equities", counting)
    gid = _table(cast, 3)
    assert _post(cast["p"][0], gid, "street_pause", {"secs": 5}).status_code == 200
    before = _nets(_state(cast["p"][0], gid))
    _start(cast, gid)
    end = _play_out(cast, gid, folders=(0,), jam=True)
    assert end["phase"] == "showdown" and end["runout"]["active"] is True
    return {"gid": gid, "calls": calls, "before": before}


def test_g3_equities_once_per_hand_from_alive_holes_same_for_everyone(cast, hg, allin):
    gid, calls = allin["gid"], allin["calls"]
    t = _live(hg, gid)
    start = t.runout_start_len
    # ONE computation per street the runout shows — at capture time.
    assert [c[1] for c in calls] == list(range(start, 6))
    assert all(c[0] == [1, 2] for c in calls), "folded seat 0 must never be a contender"
    views = [_state(cl, gid) for cl in (cast["p"][0], cast["p"][1], cast["p"][2], cast["spec"])]
    for _ in range(3):  # alternating viewers used to thrash a per-viewer cache
        views += [_state(cast["p"][0], gid), _state(cast["p"][1], gid)]
    assert len(calls) == 6 - start, "polling must not recompute"
    eqs = [[(x["equity_a"], x["equity_b"]) for x in v["seats"][:3]] for v in views]
    assert all(e == eqs[0] for e in eqs), "every viewer sees the same numbers"
    assert eqs[0][0] == (None, None)  # folded viewer's seat has no equity
    assert abs(eqs[0][1][0] + eqs[0][2][0] - 1) < 1e-3
    assert abs(eqs[0][1][1] + eqs[0][2][1] - 1) < 1e-3
    # and a poll is cheap (it used to be 0.5-2.9 s under the table lock)
    t0 = time.perf_counter()
    _state(cast["p"][0], gid)
    assert time.perf_counter() - t0 < 0.5


def test_g11_nothing_in_the_payload_runs_ahead_of_the_runout(cast, hg, allin):
    gid = allin["gid"]
    t = _live(hg, gid)
    s = _state(cast["p"][0], gid)
    ro = s["runout"]
    assert ro["blocking"] is True and ro["shown_len"] == ro["start_len"] < 5
    hidden = set(t.rabbit_full_a[ro["shown_len"]:]) | set(t.rabbit_full_b[ro["shown_len"]:])
    assert hidden
    # Award script: count only (public: it follows from the commit levels).
    assert s["pot_awards"] == [] and ro["award_step"] is None and ro["award_count"] > 0
    shown = {
        c
        for b in ("a", "b")
        for c in s["board"][b]["flop"] + [s["board"][b]["turn"], s["board"][b]["river"]]
        if c is not None
    }
    assert not (hidden & shown)
    # Hole cards are the only other card-bearing fields; none is a board card.
    in_holes = {c for seat in s["seats"] for c in (seat["hole"] or []) if c >= 0}
    assert not (hidden & in_holes)
    # Deltas: only what each dealt-in seat PUT IN — nobody has won anything yet.
    deltas = s["hand_deltas_cents"]
    assert deltas[0] == -300 and deltas[1] < 0 and deltas[2] < 0
    assert all(d <= 0 for d in deltas)
    # Ledger: exactly as before the hand (final nets used to be there already).
    assert _nets(s) == allin["before"]
    assert sum(r["net_cents"] for r in s["ledger"]) == 0
    assert s["eligible_count"] == 3  # from hand-start stacks: no bust-out spoiler
    assert s["can_deal"] is False

    _finish_runout(hg, gid)
    done = _state(cast["p"][0], gid)
    assert done["runout"]["blocking"] is False
    assert len(done["pot_awards"]) == done["runout"]["award_count"]
    assert sum(done["hand_deltas_cents"]) == 0 and max(done["hand_deltas_cents"]) > 0
    assert _nets(done) != allin["before"]
    assert sum(r["net_cents"] for r in done["ledger"]) == 0


def test_g11_award_steps_are_released_one_at_a_time(cast, hg, allin):
    gid = allin["gid"]
    t = _live(hg, gid)
    with t.lock:  # jump to: river shown, first award step in progress
        span = (5 - t.runout_start_len) * t.street_pause_secs
        t.runout_started_mono = time.monotonic() - span - 0.2 * hg.AWARD_SECS
    s = _state(cast["p"][1], gid)
    assert s["runout"]["shown_len"] == 5 and s["runout"]["award_index"] == 0
    assert len(s["pot_awards"]) == 1 and s["runout"]["award_step"] == s["pot_awards"][0]
    assert s["runout"]["blocking"] is True
    assert _nets(s) == allin["before"]  # still not the final ledger


# --- G7: /deal is gated on the runout ----------------------------------------------------------


def test_g7_deal_is_refused_while_the_runout_is_revealing(cast, hg, allin):
    gid = allin["gid"]
    for cl in (cast["p"][1], cast["p"][0]):
        r = _post(cl, gid, "deal")
        assert r.status_code == 400 and "runout" in r.json()["detail"]
    assert _state(cast["p"][0], gid)["hand_no"] == 1
    # Pausing/unpausing mid-runout must not deal either.
    assert _post(cast["p"][0], gid, "run", {"running": False}).status_code == 200
    r = _post(cast["p"][0], gid, "run", {"running": True})
    assert r.status_code == 200 and r.json()["hand_no"] == 1
    # Sitting / rebuying waits for the hand to REALLY end as well.
    assert _post(cast["p"][1], gid, "rebuy", {"amount_cents": 4000}).status_code == 400
    _finish_runout(hg, gid)
    for cl in cast["p"][:3]:  # bust-outs rebuy so two players are eligible
        if _state(cl, gid)["seats"][_state(cl, gid)["my_seat"]]["stack_cents"] <= 300:
            assert _post(cl, gid, "rebuy", {"amount_cents": 4000}).status_code == 200
    r = _post(cast["p"][1], gid, "deal")
    assert r.status_code == 200 and r.json()["hand_no"] == 2


# --- G4: amounts ----------------------------------------------------------------------------------


@pytest.mark.parametrize("amount", [1e30, 9e15, 2_000_000_000, -5, "abc", None, [1], True, "nan", "inf"])
def test_g4_absurd_amounts_are_rejected_and_the_table_survives(cast, hg, amount):
    gid = _table(cast, 2)
    r = _post(cast["p"][1], gid, "rebuy", {"amount_cents": amount})
    assert r.status_code == 400, (amount, r.status_code, r.text[:120])
    t = _live(hg, gid)
    assert t.seats[1].stack_chips == 400000 and t.seats[1].buyin_cents == 4000
    assert _start(cast, gid)["phase"] == "in_hand"  # it used to be bricked


def test_g4_non_finite_json_numbers(cast, hg):
    gid = _table(cast, 2)
    host = cast["p"][0]
    for what, raw in (
        ("rebuy", b'{"amount_cents": 1e999}'),
        ("rebuy", b'{"amount_cents": NaN}'),
        ("street_pause", b'{"secs": NaN}'),
        ("street_pause", b'{"secs": Infinity}'),
        ("street_pause", b'{"secs": "fast"}'),
        ("decision_time", b'{"secs": NaN}'),
        ("decision_time", b'{"secs": 1e999}'),
        ("sit", b'{"seat": "x"}'),
        ("kick", b'{"user_id": null}'),
        ("sit_out_player", b'{"user_id": [1]}'),
        ("sit_out", b'{"on": "yes"}'),
    ):
        r = host.post(f"/games/api/tables/{gid}/{what}", content=raw,
                      headers={"content-type": "application/json"})
        assert r.status_code == 400, (what, raw, r.status_code)
    assert _live(hg, gid).street_pause_secs == 1.5
    assert _state(host, gid)["street_pause_secs"] == 1.5  # GET used to 500 forever
    _start(cast, gid)
    cl, _ = _actor(cast, gid)
    for body in ({"gate": "raise", "chips": "zz"}, {"gate": "raise", "raise_to_chips": {}},
                 {"gate": ["raise"]}, {"gate": None}):
        assert _post(cl, gid, "act", body).status_code == 400, body
    r = cast["p"][0].post("/games/api/tables", json={"bb_cents": 1e30})
    assert r.status_code == 400


def test_g4_failed_write_rolls_memory_back(cast, hg, monkeypatch):
    gid = _table(cast, 2)
    t = _live(hg, gid)
    pub = hg.pub
    ledger_n = lambda: pub.DB.one(  # noqa: E731
        "SELECT COUNT(*) c FROM homegame_ledger WHERE game_id=?", (gid,))["c"]
    before = (t.seats[1].stack_chips, t.seats[1].buyin_cents, ledger_n())

    real = hg._persist_player

    def boom(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(hg, "_persist_player", boom)
    assert _post(cast["p"][1], gid, "rebuy", {"amount_cents": 4000}).status_code == 500
    assert _post(cast["p"][1], gid, "sit_out", {"on": True}).status_code == 500
    assert _post(cast["p"][1], gid, "leave").status_code == 500
    assert _post(cast["p"][2], gid, "sit", {"seat": 3, "buyin_cents": 4000}).status_code == 500
    # Memory AND the DB are exactly as before: the ledger row written before
    # the failing statement was rolled back with it.
    assert (t.seats[1].stack_chips, t.seats[1].buyin_cents, ledger_n()) == before
    assert t.seats[1].sitting_out is False and t.seats[3] is None
    monkeypatch.setattr(hg, "_persist_player", real)
    assert _post(cast["p"][1], gid, "rebuy", {"amount_cents": 4000}).status_code == 200
    assert t.seats[1].buyin_cents == 8000
    assert _start(cast, gid)["phase"] == "in_hand"


def test_g4_engine_path_persist_failure_does_not_wedge(cast, hg, monkeypatch):
    gid = _table(cast, 2)
    _start(cast, gid)
    t = _live(hg, gid)
    real = hg._persist_seats
    monkeypatch.setattr(hg, "_persist_seats", lambda _t: (_ for _ in ()).throw(RuntimeError("locked")))
    cl, s = _actor(cast, gid)
    _act(cl, gid, gate="raise", raise_to_chips=_min_raise_to(s))
    cl, s = _actor(cast, gid)
    end = _act(cl, gid, gate="fold")  # hand finishes; the persist fails
    assert end["phase"] == "showdown" and t.persist_dirty is True
    monkeypatch.setattr(hg, "_persist_seats", real)
    with t.lock:
        t.persist_retry_mono = 0.0
        hg._retry_persist_locked(t)  # what the watchdog does
    assert t.persist_dirty is False
    row = hg.pub.DB.one(
        "SELECT stack_chips FROM homegame_players WHERE game_id=? AND seat=0", (gid,))
    assert row["stack_chips"] == t.seats[0].stack_chips


# --- G5: the shot clock belongs to the decision -----------------------------------------------


def test_g5_other_players_cannot_reset_the_actors_clock(cast, hg):
    gid = _table(cast, 3)
    s = _start(cast, gid)
    actor = s["actor"]
    others = [i for i in range(3) if i != actor]
    t = _live(hg, gid)
    with t.lock:
        t.turn_started_mono = time.monotonic() - 25.0  # 25 of 30 s burned
    for _ in range(2):
        assert _post(cast["p"][others[0]], gid, "sit_out", {"on": True}).status_code == 200
        assert _post(cast["p"][others[0]], gid, "sit_out", {"on": False}).status_code == 200
    # host-side away toggle and a kick of a NON-actor don't reset it either
    host, target = cast["p"][0], others[-1] if others[-1] != 0 else others[0]
    uid = s["seats"][target]["user_id"]
    if target != 0:
        assert _post(host, gid, "sit_out_player", {"user_id": uid, "on": True}).status_code == 200
    after = _state(cast["p"][actor], gid)
    assert after["actor"] == actor
    assert after["turn_remaining_secs"] <= 5.1, "the clock was restarted"
    # ...while a real new decision does get a fresh clock.
    cl, st = _actor(cast, gid)
    nxt = _act(cl, gid, gate="check_call")
    if nxt["phase"] == "in_hand":
        assert nxt["turn_remaining_secs"] > 25


# --- G6: a hand can always finish -----------------------------------------------------------------


def test_g6_new_tables_have_a_clock_and_zero_stays_selectable(cast):
    t = _create(cast["p"][0])
    assert t["decision_secs"] == 30
    r = _post(cast["p"][0], t["id"], "decision_time", {"secs": 0})
    assert r.status_code == 200 and r.json()["decision_secs"] == 0
    assert _create(cast["p"][0], decision_secs=0)["decision_secs"] == 0
    assert cast["p"][0].post("/games/api/tables", json={"decision_secs": 3}).status_code == 400


def test_g6_leave_mid_hand_folds_now_and_cashes_out_after(cast, hg):
    """``leave {now: true}`` = the G6 behaviour: out of the hand at once, cashed
    out when it ends. (A plain ``leave`` mid-hand now plays the hand out first —
    2026-09-22, see test_homegame_table_ux.py.)"""
    gid = _table(cast, 3)
    s = _start(cast, gid)
    leaver = (s["actor"] + 1) % 3  # NOT the actor
    r = _post(cast["p"][leaver], gid, "leave", {"now": True})
    assert r.status_code == 200, r.text  # used to be 400 "wait for the hand to finish"
    seat = r.json()["seats"][leaver]
    assert seat["empty"] is False and seat["pending_remove"] and seat["sitting_out"]
    # They cannot cancel it by coming back mid-hand.
    assert _post(cast["p"][leaver], gid, "sit_out", {"on": False}).status_code == 400
    end = _play_out(cast, gid)
    assert end["phase"] == "showdown"
    _finish_runout(hg, gid)
    done = _state(cast["p"][0], gid)
    assert done["seats"][leaver]["empty"] is True
    row = next(r for r in done["ledger"] if r["name"] == NAMES[leaver])
    assert row["seated"] is False
    assert sum(r["net_cents"] for r in done["ledger"]) == 0


def test_g6_actor_leaving_is_acted_for_immediately(cast, hg):
    gid = _table(cast, 2)
    s = _start(cast, gid)
    actor = s["actor"]
    r = _post(cast["p"][actor], gid, "leave", {"now": True})
    assert r.status_code == 200
    after = r.json()
    assert after["action_seq"] >= 1  # the away logic acted for them
    assert after["phase"] != "in_hand" or after["actor"] != actor


def test_g6_clockless_table_with_away_actor_is_moved_by_the_watchdog(cast, hg):
    gid = _table(cast, 3, decision_secs=0)
    s = _start(cast, gid)
    t = _live(hg, gid)
    with t.lock:
        # The state the old `num_seats + 4` cap could strand: an AWAY actor,
        # no clock. The watchdog used to return early on decision_secs == 0.
        t.seats[s["actor"]].sitting_out = True
        seq = t.action_seq
        hg._timeout_tick_locked(t)
        assert t.action_seq > seq
        assert t.phase != "in_hand" or t.env.current_actor() != s["actor"]


def test_g6_everyone_away_still_finishes_the_hand(cast):
    gid = _table(cast, 6, decision_secs=0)
    s = _start(cast, gid)
    order = [i for i in range(6) if i != s["actor"]] + [s["actor"]]
    for i in order:
        assert _post(cast["p"][i], gid, "sit_out", {"on": True}).status_code == 200
    end = _state(cast["p"][0], gid)
    assert end["phase"] == "showdown", (end["phase"], end["actor"], len(end["history"]))


def test_g6_host_fold_checks_when_folding_is_not_legal(cast):
    gid = _table(cast, 2, decision_secs=0)
    s = _start(cast, gid)
    assert s["to_call_cents"] == 0
    r = _post(cast["p"][0], gid, "host_fold")
    assert r.status_code == 200, r.text  # used to 400 "illegal action"
    assert r.json()["history"][0]["label"] == "Check"


# --- G8: stale actions ---------------------------------------------------------------------------------


def test_g8_act_and_deal_409_on_a_stale_decision(cast, hg):
    gid = _table(cast, 2)
    _start(cast, gid)
    cl, s = _actor(cast, gid)
    assert (s["hand_no"], s["action_seq"]) == (1, 0)
    for stale in ({"hand_no": 0}, {"hand_no": 2}, {"action_seq": 5},
                  {"hand_no": 1, "action_seq": 1}):
        r = _post(cl, gid, "act", {"gate": "check_call", **stale})
        assert r.status_code == 409, stale
    assert _state(cl, gid)["action_seq"] == 0  # nothing was applied
    r = _post(cl, gid, "act", {"gate": "check_call", "hand_no": 1, "action_seq": 0})
    assert r.status_code == 200 and r.json()["action_seq"] == 1
    # The same click again (a delayed duplicate) must NOT land on the next
    # decision — this is the "Call $1 calls a different bet" bug.
    r = _post(cl, gid, "act", {"gate": "check_call", "hand_no": 1, "action_seq": 0})
    assert r.status_code in (400, 409)
    end = _play_out(cast, gid)
    _finish_runout(hg, gid)
    assert _post(cast["p"][1], gid, "deal", {"hand_no": end["hand_no"] + 3}).status_code == 409
    r = _post(cast["p"][1], gid, "deal", {"hand_no": end["hand_no"]})
    assert r.status_code == 200 and r.json()["hand_no"] == end["hand_no"] + 1
    # a second auto-deal for the hand that is already over: refused
    assert _post(cast["p"][0], gid, "deal", {"hand_no": end["hand_no"]}).status_code == 409


def test_g8_state_carries_a_monotonic_revision(cast):
    gid = _table(cast, 2)
    a = _state(cast["p"][0], gid)
    s = _start(cast, gid)
    assert s["epoch"] == a["epoch"] and s["rev"] > a["rev"]
    cl, st = _actor(cast, gid)
    nxt = _act(cl, gid, gate="check_call")
    assert nxt["rev"] > st["rev"]
    assert _state(cl, gid)["rev"] == nxt["rev"]  # polling does not bump it


def test_g8_raise_to_is_converted_against_the_acting_node(cast):
    gid = _table(cast, 2)
    _start(cast, gid)
    cl, s = _actor(cast, gid)
    to = _min_raise_to(s) + 20000
    after = _act(cl, gid, gate="raise", raise_to_chips=to)
    assert after["history"][-1]["to_cents"] == to // 100
    cl2, s2 = _actor(cast, gid)
    re_to = _min_raise_to(s2)
    after = _act(cl2, gid, gate="raise", raise_to_chips=re_to)
    assert after["history"][-1]["to_cents"] == re_to // 100


# --- G9: history labels ---------------------------------------------------------------------------------


def test_g9_raise_to_is_a_per_street_total(cast):
    gid = _table(cast, 2, default_buyin_cents=20000)
    assert _post(cast["p"][1], gid, "rebuy", {"amount_cents": 16000}).status_code == 200
    _start(cast, gid)
    bets = {}
    for street in ("flop", "turn"):
        cl, s = _actor(cast, gid)
        assert s["street"] == street
        to = _min_raise_to(s)
        bets[street] = to // 100
        _act(cl, gid, gate="raise", raise_to_chips=to)
        cl, s = _actor(cast, gid)
        _act(cl, gid, gate="check_call")
    hist = _state(cast["p"][0], gid)["history"]
    turn_bet = next(h for h in hist if h["street"] == "turn" and h["action"] not in (0, 1))
    assert turn_bet["to_cents"] == bets["turn"], "flop action leaked into the turn label"
    assert turn_bet["label"] == f"Raise to ${bets['turn'] / 100:.2f}"


# --- G10: money ---------------------------------------------------------------------------------------------


def test_g10_apportion_cents(hg):
    ap = hg.apportion_cents
    assert ap(100, [1, 1, 1]) == [34, 33, 33]  # tie -> lower seat
    assert ap(7, [50, 150]) == [2, 5]  # 1.75 / 5.25 -> the larger remainder gets it
    assert ap(10, [0, 5, 5]) == [0, 5, 5]
    assert ap(5, [0, 0]) == [5, 0]
    assert ap(0, [3, 4]) == [0, 0] and ap(-3, [1]) == [0] and ap(9, []) == []
    rng = random.Random(3)
    for _ in range(300):
        chips = [rng.randrange(0, 10**7) for _ in range(rng.randrange(1, 9))]
        total = rng.randrange(0, 10**6)
        out = ap(total, chips)
        assert sum(out) == total and all(x >= 0 for x in out)
        assert ap(total, chips) == out  # deterministic
        if sum(chips):
            for c, x in zip(chips, out):
                assert abs(x - total * c / sum(chips)) < 1


def test_g10_conversions_are_exact_integers(hg):
    assert hg.chips_per_cent(100) == 100 and hg.chips_per_cent(25) == 400
    assert hg.chips_per_cent(300) == 0 and hg.chips_per_cent(0) == 0
    assert hg.cents_to_chips(4000, 100) == 400000
    assert hg.chips_to_cents(50, 100) == 1 and hg.chips_to_cents(49, 100) == 0  # half-up
    assert hg.chips_to_cents(-150, 100) == -2
    big = 10**9
    assert hg.chips_to_cents(hg.cents_to_chips(big, 1), 1) == big  # no float drift


def test_g10_raises_are_snapped_to_whole_cents(cast, hg):
    gid = _table(cast, 2)
    _start(cast, gid)
    cl, s = _actor(cast, gid)
    after = _act(cl, gid, gate="raise", raise_to_chips=_min_raise_to(s) + 12345)
    t = _live(hg, gid)
    committed = after["seats"][s["actor"]]["committed_this_street_chips"]
    assert committed % 100 == 0, committed  # bb = $1 -> 100 chips per cent
    assert abs(committed - (_min_raise_to(s) + 12345)) < 100
    assert t.action_seq == 1


def test_g10_ledger_is_exactly_zero_sum_through_random_play(cast, hg):
    rng = random.Random(11)
    gid = _table(cast, 4)
    t = _live(hg, gid)
    with t.lock:
        t.street_pause_secs = 0.3
    p = cast["p"]

    def check(where):
        s = _state(p[0], gid)
        net = sum(r["net_cents"] for r in s["ledger"])
        assert net == 0, (where, net, [(r["name"], r["net_cents"]) for r in s["ledger"]])
        return s

    _start(cast, gid)
    sub_cent_seen = False
    cash_outs = 0
    for hand in range(40):
        for _ in range(60):
            cl, s = _actor(cast, gid)
            if cl is None:
                break
            check("in hand")
            roll = rng.random()
            if s["legal"]["raise"] and roll < 0.45:
                lo, hi = _min_raise_to(s), _max_raise_to(s)
                to = hi if rng.random() < 0.3 else rng.randint(lo, hi)  # odd chip amounts
                _act(cl, gid, gate="raise", raise_to_chips=to)
            elif s["legal"]["fold"] and s["to_call_chips"] > 0 and roll < 0.6:
                _act(cl, gid, gate="fold")
            else:
                _act(cl, gid, gate="check_call")
        check("runout")
        _finish_runout(hg, gid)
        s = check("settled")
        with t.lock:
            stacks = [q.stack_chips for q in t.seats if q is not None]
            sub_cent_seen |= any(c % 100 for c in stacks)
            money = sum(r["buyin_cents"] - r["leftover_cents"] for r in s["ledger"])
            # No chip is ever created or destroyed by play, buy-ins or rebuys:
            # the table's chips are worth exactly the money on it (whole
            # cents), even when single stacks carry sub-cent remainders from
            # split pots. Only a cash-out can open a gap — it pays whole cents
            # for a stack that may hold a fraction of one — and the gap is
            # below a cent per cash-out (the ledger itself stays exact).
            drift = abs(sum(stacks) - money * 100)
            assert drift <= 99 * cash_outs, (hand, sum(stacks), money, cash_outs)
        # churn: rebuy the short, sometimes leave + re-sit
        for i in range(4):
            me = _state(p[i], gid)
            if me["my_seat"] is None:
                seat = next(x["seat"] for x in me["seats"] if x["empty"])
                assert _post(p[i], gid, "sit", {"seat": seat, "buyin_cents": rng.choice([800, 3000, 6000])}).status_code == 200
            elif me["seats"][me["my_seat"]]["stack_cents"] <= 400:
                assert _post(p[i], gid, "rebuy", {"amount_cents": rng.choice([1234, 3000])}).status_code == 200
            elif i and hand >= 15 and rng.random() < 0.12:  # exact until here
                assert _post(p[i], gid, "leave").status_code == 200
                cash_outs += 1
        check("after churn")
        r = _post(p[0], gid, "deal")
        if r.status_code != 200:
            assert "need at least 2" in r.text, r.text
            break
    assert sub_cent_seen, "the simulation never produced a sub-cent stack — not a test"
    # Everyone out: the last cent is accounted for.
    _play_out(cast, gid)
    _finish_runout(hg, gid)
    assert _post(p[0], gid, "run", {"running": False}).status_code == 200
    closed = _post(p[0], gid, "close")
    assert closed.status_code == 200, closed.text
    rows = closed.json()["ledger"]
    assert all(not r["seated"] and r["stack_cents"] == 0 for r in rows)
    assert sum(r["leftover_cents"] for r in rows) == sum(r["buyin_cents"] for r in rows)
    assert sum(r["net_cents"] for r in rows) == 0


def test_g10_auto_stack_carries_the_sub_cent_remainder(cast, hg):
    gid = _table(cast, 2)
    host = cast["p"][0]
    assert _post(host, gid, "auto_stack", {"mode": "host", "all_cents": 5000}).status_code == 200
    t = _live(hg, gid)
    with t.lock:  # what a split pot leaves behind: +/- sub-cent chips
        t.seats[0].stack_chips = 400_030
        t.seats[1].stack_chips = 399_970
    s = _start(cast, gid)
    # $10.00 top-ups each (a whole number of cents), remainders carried. The
    # stacks used to be SET to 500000: 30 chips destroyed, 30 created.
    assert t.hand_start_stacks[:2] == [500_030, 499_970]
    rows = {r["name"]: r for r in s["ledger"]}
    assert rows["alice"]["buyin_cents"] == 5000 and rows["bob"]["buyin_cents"] == 5000
    assert sum(r["net_cents"] for r in s["ledger"]) == 0
    assert sum(t.hand_start_stacks) == 10000 * 100


# --- G12: membership -------------------------------------------------------------------------------------------


def test_g12_no_emails_and_outsiders_cannot_write(cast, hg):
    gid = _table(cast, 2)
    spec = cast["spec"]
    s = _state(spec, gid)
    assert s["is_member"] is False
    assert "@" not in json.dumps(s), "an email address leaked into the table payload"
    assert all("email" not in row for row in s["ledger"])
    assert _post(spec, gid, "chat", {"text": "hi from a non-member"}).status_code == 403
    assert _post(cast["p"][0], gid, "chat", {"text": "hello"}).status_code == 200
    # fold-out -> rabbit available: members only
    _start(cast, gid)
    cl, st = _actor(cast, gid)
    _act(cl, gid, gate="raise", raise_to_chips=_min_raise_to(st))
    cl, st = _actor(cast, gid)
    assert _act(cl, gid, gate="fold")["can_rabbit"] is True
    assert _post(spec, gid, "rabbit").status_code == 403
    # a FORMER member (sat, then left) still counts
    assert _post(cast["p"][1], gid, "leave").status_code == 200
    assert _state(cast["p"][1], gid)["is_member"] is True
    assert _post(cast["p"][1], gid, "chat", {"text": "gg"}).status_code == 200
    assert _post(cast["p"][1], gid, "rabbit").status_code == 200


# --- G13: resources ------------------------------------------------------------------------------------------------


def test_g13_open_tables_per_host_are_capped(cast, hg, monkeypatch):
    host = cast["p"][4]
    monkeypatch.setattr(hg, "MAX_OPEN_TABLES_PER_USER", 2)
    first = _create(host)
    _create(host)
    r = host.post("/games/api/tables", json={"name": "one too many"})
    assert r.status_code == 429
    assert _post(host, first["id"], "close").status_code == 200
    assert host.post("/games/api/tables", json={"name": "fits again"}).status_code == 200


def test_g13_table_name_and_chat_limits(cast, hg):
    host = cast["p"][0]
    assert host.post("/games/api/tables", json={"name": "x" * 61}).status_code == 400
    assert host.post("/games/api/tables", json={"name": "x" * 200_000}).status_code == 400
    t = _create(host, name="  spaced \n  out  ")
    assert t["name"] == "spaced out"
    gid = t["id"]
    long = _post(host, gid, "chat", {"text": "y" * 5000})
    assert long.status_code == 200 and len(long.json()["chat"][-1]["text"]) == 240
    assert _post(host, gid, "chat", {"text": {"no": "strings"}}).status_code == 400
    codes = [_post(host, gid, "chat", {"text": f"m{i}"}).status_code for i in range(8)]
    assert codes.count(200) == hg.CHAT_RATE_MAX - 1 and codes[-1] == 429
    live = _live(hg, gid)
    with live.lock:  # the window passes
        uid = cast["ids"]["alice@example.com"]
        live.chat_times[uid].clear()
    assert _post(host, gid, "chat", {"text": "later"}).status_code == 200


def test_g13_hub_evicts_closed_and_idle_tables(cast, hg):
    host = cast["p"][3]
    idle = _create(host)["id"]
    closed = _create(host)["id"]
    busy = _create(host)["id"]
    assert _post(cast["p"][2], busy, "sit", {"seat": 1, "buyin_cents": 4000}).status_code == 200
    assert _post(host, busy, "run", {"running": True}).json()["phase"] == "in_hand"
    assert _post(host, closed, "close").status_code == 200
    now = time.monotonic()
    assert not {idle, closed, busy} & set(hg.HUB.evict_idle(now + 5))
    gone = set(hg.HUB.evict_idle(now + hg.HUB_CLOSED_EVICT_S + 30))
    assert closed in gone and idle not in gone and busy not in gone
    gone = set(hg.HUB.evict_idle(now + hg.HUB_IDLE_EVICT_S + 30))
    assert idle in gone and busy not in gone  # never mid-hand...
    assert busy in set(hg.HUB.evict_idle(now + hg.HUB_ABANDONED_EVICT_S + 30))  # ...unless abandoned
    assert idle not in hg.HUB._tables
    # an evicted table simply reloads from the DB
    assert _state(host, idle)["seats"][0]["stack_cents"] == 4000
    assert _state(host, closed)["status"] == "closed"


# --- G14: after a restart ---------------------------------------------------------------------------------------------


def test_g14_reloaded_table_is_paused_and_dealable(cast, hg):
    gid = _table(cast, 2)
    _start(cast, gid)
    _play_out(cast, gid)
    pub = hg.pub
    assert pub.DB.one("SELECT running FROM homegames WHERE id=?", (gid,))["running"] == 1
    hg.HUB.drop(gid)  # what a server restart does to the in-memory table
    s = _state(cast["p"][0], gid)
    # It used to come back phase="waiting" + running=True: the host saw only
    # "Pause", the client (which required phase=="showdown") offered no deal.
    assert (s["phase"], s["running"], s["can_deal"]) == ("waiting", False, False)
    assert pub.DB.one("SELECT running FROM homegames WHERE id=?", (gid,))["running"] == 0
    assert s["seats"][0]["stack_cents"] + s["seats"][1]["stack_cents"] == 8000
    r = _post(cast["p"][0], gid, "run", {"running": True})
    assert r.status_code == 200 and r.json()["phase"] == "in_hand"


def test_g14_waiting_table_offers_a_deal_once_two_can_play(cast):
    host = cast["p"][0]
    gid = _create(host)["id"]
    s = _post(host, gid, "run", {"running": True}).json()  # alone: nothing to deal
    assert (s["phase"], s["running"], s["can_deal"]) == ("waiting", True, False)
    s = _post(cast["p"][1], gid, "sit", {"seat": 1, "buyin_cents": 4000}).json()
    assert s["phase"] == "waiting" and s["can_deal"] is True  # the client keys on this
    r = _post(cast["p"][1], gid, "deal", {"hand_no": s["hand_no"]})
    assert r.status_code == 200 and r.json()["phase"] == "in_hand"


def test_games_js_client_contract(server):
    """The client half of G3/G6/G8/G14 (pins the wiring; the behaviour itself is
    driven in Node by test_review_homegame_client_js.py)."""
    from pathlib import Path

    js = (Path(server.STATIC_DIR) / "games.js").read_text(encoding="utf-8")
    assert "hand_no: s.hand_no, action_seq: s.action_seq" in js  # /act names its decision
    assert "{ hand_no: s.hand_no }" in js  # /deal names the hand it follows
    assert "function acceptState(s, seq)" in js and "s.rev < cur.rev" in js
    assert "if (G.pollBusy) return;" in js  # one poll in flight
    assert "e.status === 409" in js
    assert "G.preAction = null;\n    G.preActed = false;" in js.replace("\r\n", "\n")
    # G14, superseded 2026-09-21: the SERVER deals the next hand (a timer in
    # the host's browser stalled the table whenever the host looked away), so
    # the client must not schedule deals of its own any more.
    assert "scheduleAutoDeal" not in js and "autoDeal" not in js
    play = (Path(server.STATIC_DIR) / "games.play.js").read_text(encoding="utf-8")
    assert "s.can_deal" in play  # ... but a manual "Deal" is offered whenever the server allows one
    assert 's.can_deal && s.phase === "showdown"' not in play
    html = (Path(server.STATIC_DIR) / "games.html").read_text(encoding="utf-8")
    assert 'method="post" action="/auth/logout"' in html
