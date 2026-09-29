"""Home games — features and table rules from the improvements backlog (2026-09-28,
second shift): who may skip the countdown to the next hand (HGT-024), the view
saying the table's history is open (SEC-008), and the FEAT items built on top.

Booted once for the module in PUBLIC mode against a temp DB
(`boot_public_server`, tests/python/conftest.py).
"""

from __future__ import annotations

import sys
import time

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"
NAMES = ["kim", "lou", "max", "ned", "oli", "pam"]


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server()


@pytest.fixture(scope="module")
def hg(server):
    return sys.modules["plo5bp.ui.homegame"]


@pytest.fixture(scope="module")
def cast(server):
    def login(email, name=None):
        c = TestClient(server.app, raise_server_exceptions=False)
        params = {"email": email}
        if name is not None:
            params["name"] = name
        assert c.get("/auth/dev", params=params).status_code == 200
        return c

    adm = login(ADMIN_EMAIL)
    players = [login(f"{n}@example.com") for n in NAMES]
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for n in NAMES:
        r = adm.post("/admin/api/games_access", json={"user_id": ids[f"{n}@example.com"], "action": "grant"})
        assert r.status_code == 200
    uid = {n: ids[f"{n}@example.com"] for n in NAMES}
    by_uid = {uid[n]: players[i] for i, n in enumerate(NAMES)}
    return {"p": players, "adm": adm, "uid": uid, "by_uid": by_uid, "login": login, "app": server.app}


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _post(client, gid, what, body=None):
    return client.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _state(client, gid):
    return _ok(client.get(f"/games/api/tables/{gid}"))


def _table(cast, n, host=0, **kw):
    p = cast["p"]
    body = {"name": "features", "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 4000}
    body.update(kw)
    gid = _ok(p[host].post("/games/api/tables", json=body))["id"]
    for i in range(1, n):
        _ok(_post(p[host + i], gid, "sit", {"seat": i, "buyin_cents": 4000}))
    return gid


def _fold_out(cast, gid, host=0):
    """Play the hand in progress out by folding (checking when folding is not legal)."""
    for _ in range(60):
        s = _state(cast["p"][host], gid)
        if s["phase"] != "in_hand" or s["actor"] is None:
            return s
        cl = cast["by_uid"][s["seats"][s["actor"]]["user_id"]]
        me = _state(cl, gid)
        _ok(_post(cl, gid, "act", {"gate": "fold" if me["legal"]["fold"] else "check_call"}))
    raise AssertionError("hand did not end")


def _skip_runout(hg, gid):
    t = hg.HUB.get(gid)
    with t.lock:
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0


# --- HGT-024: skipping the countdown is the host's call -----------------------------------------


def test_only_the_host_can_skip_the_countdown_but_anyone_deals_a_manual_table(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2, deal_delay_secs=30)
    _ok(_post(p[0], gid, "run", {"running": True}))
    _fold_out(cast, gid)
    _skip_runout(hg, gid)
    s = None
    for _ in range(60):  # the clock thread starts the countdown within a tick or two
        _state(p[1], gid)
        s = _state(p[0], gid)
        if s["next_deal_in_secs"] is not None:
            break
        time.sleep(0.05)
    assert s["next_deal_in_secs"] is not None, "the server is counting down to the next hand"
    r = _post(p[1], gid, "deal", {"hand_no": s["hand_no"]})
    assert r.status_code == 403 and "host" in r.json()["detail"]
    assert _state(p[1], gid)["hand_no"] == s["hand_no"], "nothing was dealt"
    assert _ok(_post(p[0], gid, "deal", {"hand_no": s["hand_no"]}))["hand_no"] == s["hand_no"] + 1
    # a table dealt by hand: any seated player deals the next hand
    gid2 = _table(cast, 2, host=2)
    _ok(_post(p[2], gid2, "run", {"running": True}))
    end = _fold_out(cast, gid2, host=2)
    _skip_runout(hg, gid2)
    assert _state(p[3], gid2)["next_deal_in_secs"] is None
    assert _ok(_post(p[3], gid2, "deal", {"hand_no": end["hand_no"]}))["hand_no"] == end["hand_no"] + 1


# --- SEC-008: the view says the table's history is open to the viewer --------------------------


def test_a_club_member_who_never_sat_may_browse_the_tables_hands_and_proofs(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    _ok(_post(p[0], gid, "run", {"running": True}))
    end = _fold_out(cast, gid)
    _skip_runout(hg, gid)
    watcher = p[4]  # in the club, never sat here
    s = _state(watcher, gid)
    assert s["is_member"] is False and s["can_browse_hands"] is True
    hands = _ok(watcher.get(f"/games/api/tables/{gid}/hands"))["hands"]
    assert hands and hands[0]["hand_no"] == end["hand_no"]
    _ok(watcher.get(f"/games/api/tables/{gid}/hands/{end['hand_no']}"))
    if hg.FAIR_ON:
        _ok(_post(p[0], gid, "deal", {"hand_no": end["hand_no"]}))  # (the proof is of an OLDER hand)
        proof = _ok(watcher.get(f"/games/api/tables/{gid}/fair/{end['hand_no']}"))
        assert proof["hand_no"] == end["hand_no"]
        assert all(int(c) >= 0 for c in proof["open"]), "openings only of cards the viewer may see"


# --- FEAT-005: the table says why the next hand isn't coming -------------------------------------


def test_the_table_says_why_the_next_hand_is_not_coming(cast, hg, monkeypatch):
    p, uid = cast["p"], cast["uid"]
    gid = _table(cast, 3, deal_delay_secs=30)
    _ok(_post(p[0], gid, "run", {"running": True}))
    _fold_out(cast, gid)
    _skip_runout(hg, gid)
    t = hg.HUB.get(gid)
    with t.lock:  # two of the three have closed their tab a while ago
        for n in ("lou", "max"):
            t.seen[uid[n]] = time.monotonic() - hg.PRESENCE_WINDOW_S - 60
        hg._auto_deal_tick_locked(t)
        reason = hg._deal_blocked_reason(t)
    assert reason == "Waiting for lou and max to come back to the table."
    s = _state(p[0], gid)
    assert s["next_deal_in_secs"] is None and s["deal_blocked_reason"] == reason
    # someone out of chips (and no automatic chips) — named
    with t.lock:
        t.seats[1].stack_chips = t.ante_chips
        t.seats[2].sitting_out = True
        assert hg._deal_blocked_reason(t) == "lou needs chips to play the next hand."
        t.seats[1].stack_chips = hg.cents_to_chips(4000, 100)
        assert hg._deal_blocked_reason(t).startswith("Waiting for lou")
        t.seats[1].sitting_out = True
        assert hg._deal_blocked_reason(t) == "Waiting for another player — lou and max are sitting out."
        t.seats[1].sitting_out = t.seats[2].sitting_out = False
        for n in ("lou", "max"):
            t.seen[uid[n]] = time.monotonic()
        assert hg._deal_blocked_reason(t) is None  # (everyone's here: the countdown runs)

    # an automatic deal that fails says so — and is not retried in a loop
    calls = []

    def failing(tb):
        calls.append(1)
        raise hg.HTTPException(status_code=400, detail="need at least 2 players with more than the ante")

    monkeypatch.setattr(hg, "_deal_locked", failing)
    with t.lock:
        hg._auto_deal_tick_locked(t)  # starts the countdown
        t.next_deal_mono -= 60.0
        hg._auto_deal_tick_locked(t)  # the deal is due: it fails
        assert calls == [1]
        assert hg._deal_blocked_reason(t) == ("The next hand couldn't be dealt: need at least 2 players "
                                             "with more than the ante.")
        for _ in range(3):
            hg._auto_deal_tick_locked(t)
        assert t.next_deal_mono is None and calls == [1], "nothing changed: no new countdown, no retry"
    _ok(_post(p[1], gid, "chat", {"text": "sorry, topping up"}))  # anything at the table changes
    with t.lock:
        hg._auto_deal_tick_locked(t)
        assert t.next_deal_mono is not None, "tried again after a change"
    monkeypatch.undo()
    with t.lock:
        t.next_deal_mono -= 60.0
        hg._auto_deal_tick_locked(t)
        assert t.phase == "in_hand" and t.deal_error is None and hg._deal_blocked_reason(t) is None


# --- FEAT-003: download a session's hand history ------------------------------------------------


def _showdown_with_a_fold(cast, gid):
    """First actor bets the minimum, the next folds, the rest call and check it down."""
    p = cast["p"]
    bettor = folder = None
    for _ in range(80):
        s = _state(p[0], gid)
        if s["phase"] != "in_hand" or s["actor"] is None:
            return s, bettor, folder
        a = s["actor"]
        cl = cast["by_uid"][s["seats"][a]["user_id"]]
        me = _state(cl, gid)
        if bettor is None and me["legal"]["raise"]:
            bettor = a
            _ok(_post(cl, gid, "act", {"gate": "raise", "raise_to_chips": me["raise_bounds"]["min_chips"] + me["street_commit_chips"]}))
        elif folder is None and me["legal"]["fold"] and me["to_call_chips"] > 0:
            folder = a
            _ok(_post(cl, gid, "act", {"gate": "fold"}))
        else:
            _ok(_post(cl, gid, "act", {"gate": "check_call"}))
    raise AssertionError("hand did not end")


def test_a_sessions_hands_download_with_only_the_cards_you_could_see(cast, hg):
    import json as _json

    p = cast["p"]
    gid = _table(cast, 3)
    _ok(_post(p[0], gid, "run", {"running": True}))
    end, bettor, folder = _showdown_with_a_fold(cast, gid)
    _skip_runout(hg, gid)
    assert end["phase"] == "showdown" and folder is not None
    _ok(_post(p[0], gid, "deal", {"hand_no": end["hand_no"]}))  # hand #2 on the table: never exported
    folder_cl = cast["by_uid"][end["seats"][folder]["user_id"]]
    other = next(i for i in range(3) if i != folder)
    other_cl = cast["by_uid"][end["seats"][other]["user_id"]]

    r = other_cl.get(f"/games/api/tables/{gid}/hands/export", params={"format": "json"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert r.headers["content-disposition"].startswith('attachment; filename="wrapgto-features-')
    data = _json.loads(r.text)
    assert data["format"] == "wrapgto.homegame.hands" and data["table"]["id"] == gid
    assert [h["hand_no"] for h in data["hands"]] == [end["hand_no"]], "the hand being played is not in it"
    seats = {x["seat"]: x for x in data["hands"][0]["seats"]}
    assert seats[folder]["hole"] is None, "the folded hand stays face-down"
    assert seats[other]["hole"] and all(c >= 0 for c in seats[other]["hole"])
    assert all("avatar" not in x and "user_id" not in x for x in seats.values())

    txt = folder_cl.get(f"/games/api/tables/{gid}/hands/export", params={"format": "txt"})
    assert txt.status_code == 200 and txt.headers["content-type"].startswith("text/plain")
    body = txt.text
    assert f"Hand #{end['hand_no']}" in body and "*** FLOP ***" in body and "*** SUMMARY ***" in body
    folder_name = end["seats"][folder]["name"]
    assert f"Dealt to {folder_name}:" in body, "your own cards, folded or not"
    assert f"{folder_name}: fold" in body and "*** SHOWDOWN ***" in body
    live = [i for i in range(3) if i != folder]
    for i in live:
        assert f"{end['seats'][i]['name']}: shows [" in body, "tabled hands are in it"
    assert f"{folder_name}: shows" not in body

    # a clubmate who never sat gets the file too — with the tabled hands only
    watcher = _ok(p[4].get(f"/games/api/tables/{gid}/hands/export", params={"format": "json"}))
    wseats = {x["seat"]: x for x in watcher["hands"][0]["seats"]}
    assert wseats[folder]["hole"] is None and all(wseats[i]["hole"] for i in live)
    assert p[0].get(f"/games/api/tables/{gid}/hands/export", params={"format": "pdf"}).status_code == 400
    stranger = cast["login"]("stranger.feat003@example.com")
    assert stranger.get(f"/games/api/tables/{gid}/hands/export").status_code == 403


# --- FEAT-004: club stats by time period ------------------------------------------------------------


def test_stats_periods_parse(hg):
    assert hg._parse_since(None) is None and hg._parse_since("all") is None and hg._parse_since("") is None
    month = hg._parse_since("month")
    assert month.endswith("T00:00:00+00:00") and month[8:10] == "01"
    assert hg._parse_since("2026-09-01T04:00:00.000Z") == "2026-09-01T04:00:00+00:00"  # a browser's local midnight
    assert hg._parse_since("2026-09-01") == "2026-09-01T00:00:00+00:00"
    assert hg._parse_since("30d") < hg._parse_since("7d")
    with pytest.raises(hg.HTTPException):
        hg._parse_since("last tuesday")


def test_club_numbers_for_a_period(cast, hg):
    p, uid = cast["p"], cast["uid"]
    q = hg.pub.DB.q
    gid = _table(cast, 2)
    _ok(_post(p[0], gid, "run", {"running": True}))
    ends = []
    for _ in range(3):
        ends.append(_fold_out(cast, gid))
        _skip_runout(hg, gid)
        if len(ends) < 3:
            _ok(_post(p[0], gid, "deal", {"hand_no": ends[-1]["hand_no"]}))
    club = _state(p[0], gid)["club"]["id"]
    first = ends[0]["hand_no"]
    q("UPDATE homegame_hands SET ended_at='2025-01-05T20:00:00+00:00' WHERE game_id=? AND hand_no=?", (gid, first))
    _ok(_post(p[0], gid, "run", {"running": False}))
    _ok(_post(p[0], gid, "close"))
    delta = {r["user_id"]: r["delta_chips"] for r in q(
        "SELECT user_id, delta_chips FROM homegame_hand_results WHERE game_id=? AND hand_no=?", (gid, first))}

    allt = _ok(p[0].get("/games/api/my/stats", params={"club": club}))
    recent = _ok(p[0].get("/games/api/my/stats", params={"club": club, "since": "30d"}))
    assert recent["since"] and allt["since"] is None
    assert recent["hands"] == allt["hands"] - 1, "the old hand is outside the last 30 days"
    s_all = next(x for x in allt["sessions"] if x["id"] == gid)
    s_rec = next(x for x in recent["sessions"] if x["id"] == gid)
    # all time: the closed session's net is its ledger; the period cuts the session, so
    # it is the period's hands in chips instead (the ledger can't be split by time)
    assert s_rec["net_cents"] == s_all["net_cents"] - hg.chips_to_cents(delta[uid["kim"]], 100)
    hands = _ok(p[0].get("/games/api/my/hands", params={"club": club, "since": "30d"}))
    assert all(h["hand_no"] != first or h["game_id"] != gid for h in hands["hands"])
    com_all = _ok(p[0].get("/games/api/community", params={"club": club}))
    com_rec = _ok(p[0].get("/games/api/community", params={"club": club, "since": "30d"}))
    me_all = next(x for x in com_all["players"] if x["user_id"] == uid["kim"])
    me_rec = next(x for x in com_rec["players"] if x["user_id"] == uid["kim"])
    assert me_rec["hands"] == me_all["hands"] - 1 and com_rec["since"]
    # a period that holds none of it: nothing (and the section can switch back)
    future = _ok(p[0].get("/games/api/community", params={"club": club, "since": "2999-01-01"}))
    assert future["players"] == [] and all(g["hands"] == 0 for g in future["games"])
    assert p[0].get("/games/api/community", params={"club": club, "since": "soon"}).status_code == 400
    # a session wholly inside the period still reads its ledger
    old = _ok(p[0].get("/games/api/my/stats", params={"club": club, "since": "2025-01-01"}))
    assert next(x for x in old["sessions"] if x["id"] == gid)["net_cents"] == s_all["net_cents"]


# --- FEAT-008 / FEAT-006 / SEC-005: the name the table shows -------------------------------------


def test_players_choose_their_name_at_the_table_and_per_club_nicknames(cast, hg, server):
    p, uid = cast["p"], cast["uid"]
    gid = _table(cast, 2, host=4)  # oli hosts, pam sits
    oli, pam = p[4], p[5]
    # the dev login named them after their email: a name nobody chose — the client asks
    st = _ok(pam.get("/games/api/me/name"))
    assert st["name"] == "pam" and st["default"] is True and st["chosen"] is None
    assert _state(pam, gid)["my_name_default"] is True
    _ok(_post(pam, gid, "chat", {"text": "hi all"}))
    # FEAT-008: a name for every table — the seat, the chat, the ledger follow at once
    st = _ok(pam.post("/games/api/me/name", json={"name": "  Pamela​  "}))
    assert st == {"name": "Pamela", "chosen": "Pamela", "account_name": "pam", "default": False}
    s = _state(oli, gid)
    assert s["seats"][1]["name"] == "Pamela"
    assert [m["name"] for m in s["chat"] if m["user_id"] == uid["pam"]] == ["Pamela"]
    assert next(r for r in s["ledger"] if r["user_id"] == uid["pam"])["name"] == "Pamela"
    assert _state(pam, gid)["my_name"] == "Pamela" and _state(pam, gid)["my_name_default"] is False
    lobby = _ok(oli.get("/games/api/tables"))
    row = next(x for x in lobby["tables"] if x["id"] == gid)
    assert [x["name"] for x in row["players"]] == ["oli", "Pamela"]
    for bad in ("x" * 21, "pam@example.com", "   "):
        r = pam.post("/games/api/me/name", json={"name": bad})
        assert r.status_code == 400 if bad.strip() else r.status_code == 200
    assert _ok(pam.get("/games/api/me/name"))["name"] == "pam", "blank = back to the account's name"
    _ok(pam.post("/games/api/me/name", json={"name": "Pamela"}))

    # FEAT-006: a nickname in ONE club — its tables show it, other clubs don't
    club = _state(oli, gid)["club"]["id"]
    v = _ok(pam.post(f"/games/api/clubs/{club}/nickname", json={"nickname": "Pam the Shark"}))
    me = next(m for m in v["members"] if m["user_id"] == uid["pam"])
    assert me["name"] == "Pam the Shark" and me["nickname"] == "Pam the Shark" and me["base_name"] == "Pamela"
    assert _state(oli, gid)["seats"][1]["name"] == "Pam the Shark"
    other_club = _ok(oli.post("/games/api/clubs", json={"name": "Second club"}))["id"]
    code = _ok(oli.get(f"/games/api/clubs/{other_club}"))["invite_code"]
    _ok(pam.post(f"/games/api/invites/{code}/join"))
    gid2 = _ok(oli.post("/games/api/tables", json={"name": "elsewhere", "club_id": other_club,
                                                    "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 4000}))["id"]
    _ok(_post(pam, gid2, "sit", {"seat": 1, "buyin_cents": 4000}))
    assert _state(oli, gid2)["seats"][1]["name"] == "Pamela", "the nickname belongs to the first club only"
    # only the owner / admins name someone else
    assert pam.post(f"/games/api/clubs/{club}/nickname", json={"user_id": uid["oli"], "nickname": "Boss"}).status_code == 403
    v = _ok(oli.post(f"/games/api/clubs/{other_club}/nickname", json={"user_id": uid["pam"], "nickname": "P"}))
    assert next(m for m in v["members"] if m["user_id"] == uid["pam"])["name"] == "P"
    assert _state(pam, gid2)["seats"][1]["name"] == "P"
    _ok(oli.post(f"/games/api/clubs/{other_club}/nickname", json={"user_id": uid["pam"], "nickname": ""}))
    assert _state(pam, gid2)["seats"][1]["name"] == "Pamela"
    # a hand record keeps the name the seat had when the hand was played
    _ok(_post(oli, gid, "run", {"running": True}))
    _fold_out(cast, gid, host=4)
    _skip_runout(hg, gid)
    hands = _ok(oli.get(f"/games/api/tables/{gid}/hands"))["hands"]
    rec = _ok(oli.get(f"/games/api/tables/{gid}/hands/{hands[0]['hand_no']}"))
    assert "Pam the Shark" in {s["name"] for s in rec["seats"]}


def test_names_never_come_from_the_email(cast, hg, server):
    login = cast["login"]
    c = login("no.name.person@private-family.example")
    q = hg.pub.DB.q
    u = q("SELECT id FROM users WHERE email=?", ("no.name.person@private-family.example",))[0]["id"]
    q("UPDATE users SET name='' WHERE id=?", (u,))
    st = _ok(c.get("/games/api/me/name"))
    assert st["name"] == f"Player {u}" and st["default"] is True  # SEC-005: never "no.name.person"
    assert hg._mask_email("no.name.person@private-family.example") == "n•••@p•••.example"
    assert hg._mask_email("sam@gmail.com") == "s•••@gmail.com"
    # a signed-out table link names the host only by a name they chose (or Google's)
    host = cast["p"][4]
    gid = _table(cast, 1, host=4)
    hg.pub.DB.q("DELETE FROM homegame_names WHERE user_id=?", (cast["uid"]["oli"],))
    hg._forget_names(uid=cast["uid"]["oli"])
    anon = TestClient(cast["app"], raise_server_exceptions=False)
    page = anon.get(f"/games/t/{gid}", headers={"accept": "text/html"}).text
    assert "<b>A friend</b> is hosting" in page, "oli's only name is his email's local part"
    _ok(host.post("/games/api/me/name", json={"name": "Oliver"}))
    page = anon.get(f"/games/t/{gid}", headers={"accept": "text/html"}).text
    assert "<b>Oliver</b> is hosting" in page


# --- FEAT-007: archive a club ---------------------------------------------------------------------


def test_the_owner_archives_a_club_and_restores_it(cast, hg):
    p = cast["p"]
    owner, member = p[2], p[3]  # max, ned
    cid = _ok(owner.post("/games/api/clubs", json={"name": "Summer league"}))["id"]
    code = _ok(owner.get(f"/games/api/clubs/{cid}"))["invite_code"]
    _ok(member.post(f"/games/api/invites/{code}/join"))
    gid = _ok(owner.post("/games/api/tables", json={"name": "last game", "club_id": cid, "bb_cents": 100,
                                                     "ante_cents": 300, "default_buyin_cents": 4000}))["id"]
    _ok(_post(member, gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    assert member.post(f"/games/api/clubs/{cid}/archive", json={"on": True}).status_code == 403
    r = owner.post(f"/games/api/clubs/{cid}/archive", json={"on": True})
    assert r.status_code == 409 and "last game" in r.json()["detail"], "a table is still open"
    _ok(_post(owner, gid, "close"))
    out = _ok(owner.post(f"/games/api/clubs/{cid}/archive", json={"on": True}))
    assert out["archived"] is True and all(c["id"] != cid for c in out["clubs"])
    assert [c["id"] for c in out["archived_clubs"]] == [cid]
    # gone from everyone's lobby; its link invites nobody; nobody hosts in it
    for c in (owner, member):
        lobby = _ok(c.get("/games/api/tables"))
        assert all(x["id"] != cid for x in lobby["clubs"])
    assert _ok(owner.get("/games/api/tables"))["archived_clubs"][0]["id"] == cid
    assert _ok(member.get("/games/api/tables"))["archived_clubs"] == [], "only its owner sees it"
    assert member.get(f"/games/api/invites/{code}").status_code == 404
    assert p[4].post(f"/games/api/clubs/{cid}/request").status_code == 404
    assert owner.post("/games/api/tables", json={"name": "x", "club_id": cid}).status_code == 404
    # ... but nothing is deleted: the old table still opens, with its ledger
    s = _state(member, gid)
    assert s["status"] == "closed" and s["ledger"]
    # restore: back for everyone
    out = _ok(owner.post(f"/games/api/clubs/{cid}/archive", json={"on": False}))
    assert any(c["id"] == cid for c in out["clubs"]) and out["archived_clubs"] == []
    assert any(c["id"] == cid for c in _ok(member.get("/games/api/tables"))["clubs"])
    assert _ok(member.get(f"/games/api/invites/{code}"))["club"]["id"] == cid
    # the site's own main club can't be archived
    main = hg._main_club()
    r = cast["adm"].post(f"/games/api/clubs/{main}/archive", json={"on": True})
    assert r.status_code == 400
