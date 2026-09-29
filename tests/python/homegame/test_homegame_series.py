"""Home games: the profit graph's data (2026-09-28, FEAT-012). GET /games/api/my/series
(and /players/{id}/series, for anyone whose numbers you may see) = the running net hand
by hand in the stats' own scope — one table, one game, one period — never a hand its
table is still revealing; a long history is thinned to 2 x SERIES_BUCKETS points and
keeps every bucket's high and low. It is summed the way the stats sum (OPS-015): a
session's hands in chips, a closed session's ledger — its last point IS the stats' net."""
from __future__ import annotations

import sys

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "admin@series.example"
NAMES = ["sal", "tam"]


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_ADMIN_EMAILS=ADMIN_EMAIL)


@pytest.fixture(scope="module")
def hg(server):
    return sys.modules["plo5bp.ui.homegame"]


@pytest.fixture(scope="module")
def cast(server):
    def login(email):
        c = TestClient(server.app, raise_server_exceptions=False)
        assert c.get("/auth/dev", params={"email": email}).status_code == 200
        return c

    adm = login(ADMIN_EMAIL)
    players = [login(f"{n}@series.example") for n in NAMES]
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for n in NAMES:
        assert adm.post("/admin/api/games_access", json={"user_id": ids[f"{n}@series.example"], "action": "grant"}).status_code == 200
    by_uid = {ids[f"{n}@series.example"]: players[i] for i, n in enumerate(NAMES)}
    return {"p": players, "by_uid": by_uid, "ids": [ids[f"{n}@series.example"] for n in NAMES]}


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _play_hands(cast, hg, gid, n):
    p = cast["p"]
    _ok(p[0].post(f"/games/api/tables/{gid}/run", json={"running": True}))
    for _ in range(n):
        for _ in range(80):
            s = _ok(p[0].get(f"/games/api/tables/{gid}"))
            if s["phase"] != "in_hand" or s["actor"] is None:
                break
            cl = cast["by_uid"][s["seats"][s["actor"]]["user_id"]]
            st = _ok(cl.get(f"/games/api/tables/{gid}"))
            _ok(cl.post(f"/games/api/tables/{gid}/act", json={"gate": "check_call", "hand_no": st["hand_no"], "action_seq": st["action_seq"]}))
        t = hg.HUB.get(gid)
        with t.lock:  # (the runout's reveal, fast-forwarded)
            if t.runout_active and t.runout_started_mono is not None:
                t.runout_started_mono -= 600.0
        s = _ok(p[0].get(f"/games/api/tables/{gid}"))
        if s["phase"] != "in_hand":
            r = p[0].post(f"/games/api/tables/{gid}/deal", json={})
            assert r.status_code in (200, 409), r.text
    _ok(p[0].post(f"/games/api/tables/{gid}/run", json={"running": False}))


def test_the_running_net_adds_up_the_hands_in_order(cast, hg):
    p = cast["p"]
    gid = _ok(p[0].post("/games/api/tables", json={"name": "series", "bb_cents": 100, "ante_cents": 300,
                                                    "default_buyin_cents": 4000, "num_seats": 2}))["id"]
    _ok(p[1].post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000}))
    _play_hands(cast, hg, gid, 4)
    hands = _ok(p[0].get("/games/api/my/hands", params={"sort": "time", "dir": "asc", "limit": 50}))["hands"]
    hands.sort(key=lambda h: h["hand_no"])  # (one table: time order = hand order; a fast test ends several in one second)
    sr = _ok(p[0].get("/games/api/my/series"))
    assert sr["hands"] == len(hands) >= 3
    assert sr["points"][0] == [0, 0]
    run = 0
    for k, h in enumerate(hands, 1):
        run += h["net_cents"]
        assert sr["points"][k] == [k, run]
    assert sr["net_cents"] == run
    assert sr["best_cents"] == max(x[1] for x in sr["points"][1:])
    assert sr["worst_cents"] == min(x[1] for x in sr["points"][1:])
    # one table only; the other player's own series, as the club lets you see it
    assert _ok(p[0].get("/games/api/my/series", params={"game": gid}))["hands"] == len(hands)
    assert _ok(p[0].get("/games/api/my/series", params={"game": "nope"}))["points"] == [[0, 0]]
    theirs = _ok(p[0].get(f"/games/api/players/{cast['ids'][1]}/series"))
    assert theirs["net_cents"] == -run and theirs["hands"] == len(hands)
    assert p[0].get("/games/api/players/999999/series").status_code == 404


def _play_odd_hands(cast, hg, gid, n, rng, buyin):
    """Hands with random raise sizes at a stake whose chips are not whole cents (a 3c big
    blind: raises are not snapped): results in fractions of a cent — the per-hand cents are
    rounded, the chips are exact."""
    p = cast["p"]
    _ok(p[0].post(f"/games/api/tables/{gid}/street_pause", json={"secs": 0.3}))
    _ok(p[0].post(f"/games/api/tables/{gid}/run", json={"running": True}))
    for k in range(n):
        for _ in range(80):
            s = _ok(p[0].get(f"/games/api/tables/{gid}"))
            if s["phase"] != "in_hand" or s["actor"] is None:
                break
            cl = cast["by_uid"][s["seats"][s["actor"]]["user_id"]]
            st = _ok(cl.get(f"/games/api/tables/{gid}"))
            if st["legal"]["raise"] and rng.random() < 0.5:
                lo = st["raise_bounds"]["min_chips"] + st["street_commit_chips"]
                hi = st["raise_bounds"]["max_chips"] + st["street_commit_chips"]
                body = {"gate": "raise", "raise_to_chips": rng.randint(lo, min(hi, lo + 3 * hg.BB_CHIPS))}
            else:
                body = {"gate": "check_call"}
            _ok(cl.post(f"/games/api/tables/{gid}/act", json={**body, "hand_no": st["hand_no"], "action_seq": st["action_seq"]}))
        t = hg.HUB.get(gid)
        with t.lock:  # (the runout's reveal, fast-forwarded)
            if t.runout_active and t.runout_started_mono is not None:
                t.runout_started_mono -= 600.0
            hg._settle_locked(t)
        for i, cl in enumerate(p):  # (a busted player reloads)
            me = _ok(cl.get(f"/games/api/tables/{gid}"))
            if me["my_seat"] is not None and me["seats"][me["my_seat"]]["stack_cents"] <= buyin // 10:
                _ok(cl.post(f"/games/api/tables/{gid}/rebuy", json={"amount_cents": buyin}))
        s = _ok(p[0].get(f"/games/api/tables/{gid}"))
        if s["phase"] != "in_hand" and k < n - 1:  # (the last one ends the session: nothing dealt after it)
            r = p[0].post(f"/games/api/tables/{gid}/deal", json={})
            assert r.status_code in (200, 409), r.text
    _ok(p[0].post(f"/games/api/tables/{gid}/run", json={"running": False}))


def test_the_graph_ends_exactly_on_the_stats_net(cast, hg):
    """OPS-015 for the graph: a session is summed the way the stats sum it — its hands in
    CHIPS, a closed session's ledger — so the last point IS the stats page's net, to the
    cent, never the per-hand rounded cents added up (they drift)."""
    import random

    p, ids = cast["p"], cast["ids"]
    bb, buyin = 3, 300
    gid = _ok(p[0].post("/games/api/tables", json={"name": "exact", "bb_cents": bb, "ante_cents": 2 * bb,
                                                    "default_buyin_cents": buyin, "num_seats": 2}))["id"]
    _ok(p[1].post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": buyin}))
    club = _ok(p[0].get(f"/games/api/tables/{gid}"))["club"]["id"]
    _play_odd_hands(cast, hg, gid, 12, random.Random(17), buyin)
    rows = hg.pub.DB.q("SELECT user_id, delta_chips, delta_cents FROM homegame_hand_results WHERE game_id=? "
                       "ORDER BY hand_no", (gid,))
    assert any(int(r["delta_chips"]) * bb % hg.BB_CHIPS for r in rows), "some result is a fraction of a cent"

    def check(label):
        for i, cl in enumerate(p):
            st = _ok(cl.get("/games/api/my/stats", params={"club": club}))
            sr = _ok(cl.get("/games/api/my/series", params={"club": club}))
            assert sr["points"][-1][1] == sr["net_cents"] == st["net_cents"], (label, i, sr["net_cents"], st["net_cents"])
            one = _ok(cl.get("/games/api/my/series", params={"club": club, "game": gid}))
            sess = next(x for x in st["sessions"] if x["id"] == gid)
            assert one["net_cents"] == sess["net_cents"], (label, i)
            # and as anyone else sees it
            other = _ok(p[1 - i].get(f"/games/api/players/{ids[i]}/series", params={"club": club, "game": gid}))
            assert other["net_cents"] == sess["net_cents"], (label, i)
            # every point on the way is the session's chips so far, turned into cents once
            chips, want = 0, []
            for r in rows:
                if int(r["user_id"]) == ids[i]:
                    chips += int(r["delta_chips"])
                    want.append(hg.chips_to_cents(chips, bb))
            want[-1] = sess["net_cents"]
            assert [v for _, v in one["points"][1:]] == want, (label, i)

    check("open")
    closed = _ok(p[0].post(f"/games/api/tables/{gid}/close", json={}))
    ledger = {r["user_id"]: r["net_cents"] for r in closed["ledger"]}
    check("closed")
    for i, cl in enumerate(p):  # (closed: the stats — and so the graph — are the ledger)
        assert _ok(cl.get("/games/api/my/series", params={"club": club, "game": gid}))["net_cents"] == ledger[ids[i]]


def test_a_long_history_is_thinned_but_keeps_its_swings(hg, monkeypatch):
    rows = []
    net = [((i * 7919) % 101) - 50 for i in range(3000)]  # a jagged walk
    net[1234] = 50000  # one huge win…
    net[2345] = -80000  # …and one huge loss: both must survive the thinning
    for i, d in enumerate(net):
        rows.append({"d": d, "dc": None, "bb": 100, "gid": "A" if i < 1500 else "B"})  # (hands stored before chips)
    monkeypatch.setattr(hg.pub.DB, "q", lambda *a, **k: rows)
    monkeypatch.setattr(hg, "_session_results", lambda *a, **k: [])
    sr = hg._my_series(1)
    run, cum = [], 0
    for d in net:
        cum += d
        run.append(cum)
    assert sr["hands"] == 3000 and sr["net_cents"] == run[-1]
    assert len(sr["points"]) <= 2 * hg.SERIES_BUCKETS + 2
    assert sr["points"][-1] == [3000, run[-1]]
    got = dict((k, v) for k, v in sr["points"])
    near = lambda a, b: [v for k, v in got.items() if a <= k <= b]  # noqa: E731
    assert max(near(1200, 1270)) - min(near(1200, 1270)) >= 49000, "the big win shows as a step"
    assert max(near(2310, 2380)) - min(near(2310, 2380)) >= 79000, "and so does the big loss"
    assert max(got.values()) == max(run) and min(got.values()) == min(run)
    assert sr["breaks"] == [1500]
    assert all(b[0] < a[0] for b, a in zip(sr["points"], sr["points"][1:])), "in hand order"
