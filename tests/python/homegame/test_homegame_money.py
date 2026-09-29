"""Home games — the money (improvements backlog, 2026-09-28, second shift).

OPS-015: a hand's stored cents add up to zero, and stats take a closed
session's net from its ledger (an open one's from its hands summed in CHIPS) —
no more drift between the stats and the ledger.
OPS-017: the ledger names each movement precisely and the hand it came after,
and ``reconcile_ledger`` proves the rows add up to every player's totals (at
start and in /games/api/health).
FEAT-001: the fewest payments that settle a session, on the table and per
finished session in the lobby.
FEAT-002: a receipt — your own money at a table, movement by movement (the
host may open anyone's).
TEST-005: stats net == ledger net after many random hands with every kind of
money movement.

Booted once for the module in PUBLIC mode against a temp DB
(`boot_public_server`, tests/python/conftest.py).
"""

from __future__ import annotations

import base64
import itertools
import random
import sys

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"
NAMES = ["quinn", "rae", "sol", "tia", "uri", "val"]


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server()


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
    players = [login(f"{n}@example.com") for n in NAMES]
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for n in NAMES:
        r = adm.post("/admin/api/games_access", json={"user_id": ids[f"{n}@example.com"], "action": "grant"})
        assert r.status_code == 200
    uid = {n: ids[f"{n}@example.com"] for n in NAMES}
    by_uid = {uid[n]: players[i] for i, n in enumerate(NAMES)}
    return {"p": players, "adm": adm, "uid": uid, "by_uid": by_uid, "app": server.app}


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _post(client, gid, what, body=None):
    return client.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _state(client, gid):
    return _ok(client.get(f"/games/api/tables/{gid}"))


def _table(cast, n, host=0, **kw):
    p = cast["p"]
    body = {"name": "money", "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 4000}
    body.update(kw)
    gid = _ok(p[host].post("/games/api/tables", json=body))["id"]
    for i in range(1, n):
        _ok(_post(p[host + i], gid, "sit", {"seat": i, "buyin_cents": 4000}))
    return gid


def _actor(cast, gid):
    s = _state(cast["p"][0], gid)
    if s["phase"] != "in_hand" or s["actor"] is None:
        return None, s
    cl = cast["by_uid"][s["seats"][s["actor"]]["user_id"]]
    return cl, _state(cl, gid)


def _skip_runout(hg, gid):
    t = hg.HUB.get(gid)
    with t.lock:
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0
        hg._settle_locked(t)


def _play_hand(cast, hg, gid, rng):
    """One hand of random actions (odd raise sizes: sub-cent stacks), to its end."""
    for _ in range(80):
        cl, s = _actor(cast, gid)
        if cl is None:
            break
        roll = rng.random()
        if s["legal"]["raise"] and roll < 0.4:
            lo = s["raise_bounds"]["min_chips"] + s["street_commit_chips"]
            hi = s["raise_bounds"]["max_chips"] + s["street_commit_chips"]
            to = hi if rng.random() < 0.3 else rng.randint(lo, hi)
            _ok(_post(cl, gid, "act", {"gate": "raise", "raise_to_chips": to}))
        elif s["legal"]["fold"] and s["to_call_chips"] > 0 and roll < 0.6:
            _ok(_post(cl, gid, "act", {"gate": "fold"}))
        else:
            _ok(_post(cl, gid, "act", {"gate": "check_call"}))
    _skip_runout(hg, gid)


# --- OPS-015: a hand's cents add up -------------------------------------------------------------


def test_one_hands_cents_sum_to_zero_and_each_is_within_a_cent(hg):
    rng = random.Random(3)
    for _ in range(3000):
        n = rng.randint(2, 8)
        d = [rng.randint(-10**7, 10**7) for _ in range(n - 1)]
        d.append(-sum(d))
        bb = rng.choice([100, 200, 50, 3, 7, 1000, 2500])
        c = hg.zero_sum_cents(d, bb)
        assert sum(c) == 0, (d, bb, c)
        for di, ci in zip(d, c):
            assert abs(ci * hg.BB_CHIPS - di * bb) < hg.BB_CHIPS, (di, bb, ci)  # floor or ceil
    assert hg.zero_sum_cents([50, 50, -100], 100) == [1, 0, -1]  # half-cents: the lower seat's goes up


# --- FEAT-001: the fewest payments ---------------------------------------------------------------


def _min_payments_bruteforce(vals):
    """n - (most zero-sum groups the players split into), by trying every ordering."""
    best = 0
    for perm in itertools.permutations(vals):
        run, groups = 0, 0
        for v in perm:
            run += v
            groups += run == 0
        best = max(best, groups)
    return len(vals) - best


def _apply(plan, nets):
    left = dict(nets)
    for a, b, c in plan:
        assert c > 0
        left[a] += c
        left[b] -= c
    return left


def test_settling_up_takes_the_fewest_payments(hg):
    rng = random.Random(5)
    for _ in range(300):
        n = rng.randint(2, 7)
        vals = [rng.choice([-1, 1]) * rng.choice([100, 250, 400, 650, 1000, 1210]) for _ in range(n - 1)]
        vals.append(-sum(vals))
        nets = {100 + i: v for i, v in enumerate(vals)}
        plan = hg._settle_plan(tuple(sorted(nets.items())))
        assert all(v == 0 for v in _apply(plan, nets).values())
        nonzero = [v for v in vals if v]
        assert len(plan) == (_min_payments_bruteforce(nonzero) if nonzero else 0), (vals, plan)
        assert plan == hg._settle_plan(tuple(sorted(nets.items()))), "deterministic"
    # two pairs that cancel each other: two payments, not three
    plan = hg._settle_plan(((1, -500), (2, 500), (3, -700), (4, 700)))
    assert sorted(plan) == [(1, 2, 500), (3, 4, 700)]
    # a big session: settled greedily, still at most n - 1 payments
    vals = [rng.randint(-5000, 5000) for _ in range(19)]
    vals.append(-sum(vals))
    nets = dict(enumerate(vals))
    plan = hg._settle_plan(tuple(sorted(nets.items())))
    assert len(plan) <= 19 and all(v == 0 for v in _apply(plan, nets).values())
    # rows that cannot be settled (money that does not exist) settle nothing — never an error
    assert hg.settle_up([{"user_id": 1, "name": "a", "net_cents": 5}]) == []


# --- OPS-017 / FEAT-002 / FEAT-001 / OPS-015 / TEST-005: a whole session -------------------------


def test_a_session_with_every_kind_of_money_movement_adds_up_everywhere(cast, hg):
    p, uid = cast["p"], cast["uid"]
    rng = random.Random(11)
    gid = _table(cast, 4, allow_rathole=True)
    _ok(_post(p[0], gid, "street_pause", {"secs": 0.3}))
    _ok(_post(p[0], gid, "auto_topup", {"mode": "player"}))
    _ok(_post(p[0], gid, "auto_stack", {"mode": "player"}))
    _ok(_post(p[1], gid, "auto_chips_self", {"kind": "topup", "target_cents": 5000, "below_cents": 3000}))
    _ok(_post(p[2], gid, "auto_chips_self", {"kind": "set", "target_cents": 4500}))
    club = _state(p[0], gid)["club"]["id"]
    _ok(_post(p[0], gid, "run", {"running": True}))
    left = False
    for hand in range(24):
        _play_hand(cast, hg, gid, rng)
        s = _state(p[0], gid)
        assert sum(r["net_cents"] for r in s["ledger"]) == 0
        # the payments on the table settle the ledger exactly
        nets = {r["user_id"]: r["net_cents"] for r in s["ledger"]}
        plan = [(x["from"], x["to"], x["cents"]) for x in s["settle_up"]]
        assert all(v == 0 for v in _apply(plan, nets).values())
        if hand == 5:
            _ok(_post(p[3], gid, "rebuy", {"amount_cents": 1234, "queue": True}))
        if hand == 8:  # chips off the table now (between hands) and once more after the next hand
            if _state(p[3], gid)["seats"][3]["stack_cents"] < 2000:
                _ok(_post(p[3], gid, "rebuy", {"amount_cents": 3000}))
            _ok(_post(p[3], gid, "remove_chips", {"amount_cents": 500}))
            _ok(_post(p[3], gid, "remove_chips", {"amount_cents": 500, "queue": True}))
        if hand == 12 and not left:
            _ok(_post(p[1], gid, "leave"))
            left = True
        if hand == 14:
            seat = next(x["seat"] for x in _state(p[1], gid)["seats"] if x["empty"])
            _ok(_post(p[1], gid, "sit", {"seat": seat, "buyin_cents": 3000}))
        for i in range(4):  # the busted reload
            me = _state(p[i], gid)
            if me["my_seat"] is not None and me["seats"][me["my_seat"]]["stack_cents"] <= 400:
                _ok(_post(p[i], gid, "rebuy", {"amount_cents": rng.choice([1234, 3000])}))
        r = _post(p[0], gid, "deal", {})
        if r.status_code != 200:
            assert "need at least 2" in r.text or "already" in r.text, r.text
            break
    _play_hand(cast, hg, gid, rng)

    # OPS-015: stats of the OPEN session are the hands in chips — within a cent of the ledger
    s = _state(p[0], gid)
    ledger = {r["user_id"]: r for r in s["ledger"]}
    for i in range(4):
        st = _ok(p[i].get("/games/api/my/stats", params={"club": club}))
        sess = next((x for x in st["sessions"] if x["id"] == gid), None)
        if sess is not None:
            assert abs(sess["net_cents"] - ledger[uid[NAMES[i]]]["net_cents"]) <= 1, (sess, ledger[uid[NAMES[i]]])

    # FEAT-002: every movement on the receipt, in the specific kinds, with hand numbers
    kinds = set()
    for i in range(4):
        rc = _ok(p[i].get(f"/games/api/tables/{gid}/ledger/me"))
        row = ledger[uid[NAMES[i]]]
        assert rc["in_cents"] == row["buyin_cents"] and rc["out_cents"] == row["leftover_cents"]
        assert rc["net_cents"] == row["net_cents"] and rc["stack_cents"] == row["stack_cents"]
        assert all(x["hand_no"] is not None and x["label"] for x in rc["items"])
        kinds |= {x["kind"] for x in rc["items"]}
    assert {"buyin", "rebuy", "topup", "cashout", "take_off"} <= kinds, kinds
    assert kinds & {"auto_topup", "stack_in", "stack_out"}, kinds
    other = uid[NAMES[1]]
    assert p[2].get(f"/games/api/tables/{gid}/ledger/{other}").status_code == 403
    assert _ok(p[0].get(f"/games/api/tables/{gid}/ledger/{other}"))["user_id"] == other  # the host may
    assert hg.reconcile_ledger(gid) == []

    # close: the session's stats ARE its ledger, the lobby shows who pays whom
    _ok(_post(p[0], gid, "run", {"running": False}))
    closed = _ok(_post(p[0], gid, "close"))
    nets = {r["user_id"]: r["net_cents"] for r in closed["ledger"]}
    assert sum(nets.values()) == 0
    plan = [(x["from"], x["to"], x["cents"]) for x in closed["settle_up"]]
    assert all(v == 0 for v in _apply(plan, nets).values())
    community = _ok(p[0].get("/games/api/community", params={"club": club}))
    for i in range(4):
        u = uid[NAMES[i]]
        st = _ok(p[i].get("/games/api/my/stats", params={"club": club}))
        assert next(x for x in st["sessions"] if x["id"] == gid)["net_cents"] == nets[u], "stats == ledger"
        lobby = _ok(p[i].get("/games/api/tables", params={"club": club}))
        sess = next(x for x in lobby["sessions"] if x["id"] == gid)
        assert sess["net_cents"] == nets[u]
        mine = [x for x in closed["settle_up"] if u in (x["from"], x["to"])]
        assert len(sess["settle"]) == len(mine)
        assert sum(x["cents"] if x["you"] == "get" else -x["cents"] for x in sess["settle"]) == nets[u]
    played = {x["user_id"]: x["net_cents"] for x in community["players"]}
    assert all(played[u] == nets[u] for u in nets if u in played)
    # OPS-015: every hand's stored cents add up to zero
    bad = hg.pub.DB.q("SELECT hand_no, SUM(delta_cents) s FROM homegame_hand_results WHERE game_id=? "
                      "GROUP BY hand_no HAVING SUM(delta_cents)<>0", (gid,))
    assert not bad, [dict(x) for x in bad]
    assert hg.reconcile_ledger(gid) == []


def test_reconciliation_reports_money_that_does_not_add_up(cast, hg):
    p, uid = cast["p"], cast["uid"]
    gid = _table(cast, 2, host=4)
    assert hg.reconcile_ledger(gid) == []
    health = _ok(cast["adm"].get("/games/api/health"))
    assert "ledger" in health
    hg.pub.DB.q("UPDATE homegame_ledger SET amount_cents=amount_cents+1 WHERE game_id=? AND user_id=?",
                (gid, uid[NAMES[5]]))
    try:
        probs = hg.reconcile_ledger(gid)
        assert len(probs) == 1 and probs[0]["user_id"] == uid[NAMES[5]] and probs[0]["problem"] == "totals"
        assert probs[0]["ledger_in_cents"] == probs[0]["buyin_cents"] + 1
        h = _ok(cast["adm"].get("/games/api/health"))["ledger"]
        assert h["ok"] is False and any(x["table"] == gid for x in h["first"])
    finally:
        hg.pub.DB.q("UPDATE homegame_ledger SET amount_cents=amount_cents-1 WHERE game_id=? AND user_id=?",
                    (gid, uid[NAMES[5]]))
    assert hg.reconcile_ledger(gid) == []
    cols = {r[1] for r in hg.pub.DB.q("PRAGMA table_info(homegame_ledger)")}
    assert "hand_no" in cols
    assert p[5].get(f"/games/api/tables/{gid}/ledger/999999").status_code == 403
    assert p[4].get(f"/games/api/tables/{gid}/ledger/999999").status_code == 404  # (the host: nobody there)


def test_the_sites_health_reports_the_home_games(cast, hg, server):
    from plo5bp.ui import middleware as mw

    assert "home_games" in mw.HEALTH_CHECKS
    hg._ledger_health()  # (the site's check reuses the last ledger check for a few minutes)
    line = mw.HEALTH_CHECKS["home_games"]()
    assert line["ok"] is True and line["clock"] is True and line["ledger_ok"] is True, line
    body = cast["adm"].get("/health").json()
    assert body["threads"]["home_games"]["ok"] is True


# --- SEC-009: request budgets -----------------------------------------------------------------


def test_request_budgets_answer_429_with_retry_after(cast, hg, monkeypatch):
    from plo5bp.ui.ratelimit import RateLimiter

    p, uid = cast["p"], cast["uid"]
    monkeypatch.setattr(hg, "API_RATE", RateLimiter(rate=0.01, burst=6))
    for _ in range(6):
        _ok(p[0].get("/games/api/clubs"))
    r = p[0].get("/games/api/clubs")
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    _ok(p[1].get("/games/api/clubs"))  # (per user)
    r = p[1].get("/games/api/community")  # (costs 5 of the 6 left: one more is too many)
    assert r.status_code in (200, 404)
    assert p[1].get("/games/api/community").status_code == 429
    monkeypatch.setattr(hg, "API_RATE", None)

    # live streams: at most MAX_STREAMS_PER_USER open at once, a finished one frees its slot
    gid = _table(cast, 2, host=4)
    with p[4].stream("GET", f"/games/api/tables/{gid}/stream", params={"max_events": 1}) as s:
        assert s.status_code == 200
        "".join(s.iter_text())
    assert hg.STREAMS.count(uid[NAMES[4]]) == 0
    for _ in range(hg.MAX_STREAMS_PER_USER):
        assert hg.STREAMS.enter(uid[NAMES[4]])
    try:
        r = p[4].get(f"/games/api/tables/{gid}/stream", params={"max_events": 1})
        assert r.status_code == 429
    finally:
        for _ in range(hg.MAX_STREAMS_PER_USER):
            hg.STREAMS.leave(uid[NAMES[4]])

    # join requests and new pictures have budgets of their own
    monkeypatch.setattr(hg, "JOIN_RATE", RateLimiter(rate=0.001, burst=1))
    clubs = [_ok(p[i].post("/games/api/clubs", json={"name": f"budget {i}"}))["id"] for i in (2, 3)]
    assert _ok(p[5].post(f"/games/api/clubs/{clubs[0]}/request"))["request"] == "pending"
    assert _ok(p[5].post(f"/games/api/clubs/{clubs[0]}/request"))["request"] == "pending", \
        "asking again while pending costs nothing"
    assert p[5].post(f"/games/api/clubs/{clubs[1]}/request").status_code == 429
    monkeypatch.setattr(hg, "AVATAR_RATE", RateLimiter(rate=0.001, burst=1))
    png = base64.b64encode(
        b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (64).to_bytes(4, "big") + (64).to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00" + b"\x00" * 16).decode()
    url = f"data:image/png;base64,{png}"
    _ok(p[5].post("/games/api/me/avatar", json={"data_url": url}))
    assert p[5].post("/games/api/me/avatar", json={"data_url": "data:image/png;base64,AAAA"}).status_code == 400, \
        "a rejected picture costs nothing"
    assert p[5].post("/games/api/me/avatar", json={"data_url": url}).status_code == 429
    p[5].delete("/games/api/me/avatar")


# --- PERF-010: the lobby's queries don't grow with its tables ---------------------------------


def test_the_lobby_runs_a_fixed_number_of_queries(cast, hg, monkeypatch):
    p = cast["p"]
    club = _ok(p[3].post("/games/api/clubs", json={"name": "Query count"}))["id"]
    calls = []
    real_q, real_one = hg.pub.DB.q, hg.pub.DB.one

    def count(fn):
        def inner(sql, *a, **kw):
            calls.append(sql)
            return fn(sql, *a, **kw)
        return inner

    def lobby_queries():
        calls.clear()
        monkeypatch.setattr(hg.pub.DB, "q", count(real_q))
        monkeypatch.setattr(hg.pub.DB, "one", count(real_one))
        try:
            _ok(p[3].get("/games/api/tables", params={"club": club}))
        finally:
            monkeypatch.setattr(hg.pub.DB, "q", real_q)
            monkeypatch.setattr(hg.pub.DB, "one", real_one)
        return len(calls)

    _ok(p[3].post("/games/api/tables", json={"name": "one", "club_id": club, "listed": False}))
    one = lobby_queries()
    for k in range(4):
        _ok(p[3].post("/games/api/tables", json={"name": f"more {k}", "club_id": club, "listed": k % 2 == 0}))
    many = lobby_queries()
    assert many == one, f"{one} queries with one table, {many} with five"


# --- HGB-003 / PERF-004: the view is built from a shared half + the viewer's own ---------------


def test_views_after_the_first_run_no_queries_and_proofs_never_build_a_view(cast, hg, monkeypatch):
    p, uid = cast["p"], cast["uid"]
    gid = _table(cast, 3)
    _ok(_post(p[0], gid, "run", {"running": True}))
    _ok(_post(p[1], gid, "chat", {"text": "gl"}))
    t = hg.HUB.get(gid)
    watchers = [p[0], p[1], p[2], p[3]]  # three seated players and a clubmate watching
    for c in watchers:
        _state(c, gid)
    calls = []
    real_q = hg.pub.DB.q

    def counting(sql, *a, **kw):
        calls.append(sql)
        return real_q(sql, *a, **kw)

    monkeypatch.setattr(hg.pub.DB, "q", counting)
    try:
        with t.lock:
            for c_uid in (uid[NAMES[0]], uid[NAMES[1]], uid[NAMES[2]], uid[NAMES[3]]):
                hg._view(t, c_uid)
    finally:
        monkeypatch.setattr(hg.pub.DB, "q", real_q)
    assert calls == [], f"a view queried the database: {calls}"
    # each viewer still sees only their own cards
    views = {n: _state(p[i], gid) for i, n in enumerate(NAMES[:4])}
    for i in range(3):
        mine = views[NAMES[i]]["seats"][i]["hole"]
        assert mine and mine[0] >= 0
        for j in range(3):
            if j != i and views[NAMES[i]]["phase"] == "in_hand":
                assert views[NAMES[i]]["seats"][j]["hole"][0] == -1
    assert all(x["hole"] is None or x["hole"][0] == -1 for x in views[NAMES[3]]["seats"])
    # the cards a viewer is shown are exactly the ones whose proofs they get
    with t.lock:
        for i in range(4):
            v = hg._view(t, uid[NAMES[i]])
            shown = [c for s in v["seats"] for c in (s["hole"] or []) if c >= 0]
            shown += [c for b in ("a", "b") for c in v["board"][b]["flop"] + [v["board"][b]["turn"], v["board"][b]["river"]]
                      if isinstance(c, int) and c >= 0] + v["burns"]
            assert sorted(hg._visible_cards(t, uid[NAMES[i]])) == sorted(shown)
    if hg.FAIR_ON:
        def no_view(*a, **k):
            raise AssertionError("a shuffle proof built a whole view")

        monkeypatch.setattr(hg, "_view", no_view)
        with t.lock:
            hg._fair_transcript(t, uid[NAMES[0]], t.hand_no) if t.fair_hand is not None else None


def test_a_removed_member_loses_the_table_at_once(cast, hg):
    p, uid = cast["p"], cast["uid"]
    club = _ok(p[4].post("/games/api/clubs", json={"name": "Revocable"}))["id"]
    code = _ok(p[4].get(f"/games/api/clubs/{club}"))["invite_code"]
    _ok(p[5].post(f"/games/api/invites/{code}/join"))
    gid = _ok(p[4].post("/games/api/tables", json={"name": "t", "club_id": club}))["id"]
    _state(p[5], gid)  # (their role is cached now)
    _ok(p[4].post(f"/games/api/clubs/{club}/members", json={"user_id": uid[NAMES[5]], "remove": True}))
    r = p[5].get(f"/games/api/tables/{gid}")
    assert r.status_code == 403 and r.json()["detail"]["error"] == "club"
