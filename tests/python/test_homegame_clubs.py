"""Home games, clubs (2026-09-25). Home games used to be admin-granted per
account; now every signed-in user has them, and a CLUB is the private circle:
its tables, its members, its numbers. Anyone can start a club and host in it;
people join with the invite link (or ask first); a table's club alone sees,
sits at and watches it; and every ranking / profit / accuracy / head-to-head is
computed inside one club."""
from __future__ import annotations

import sys

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "admin@clubs.example"
HTML = {"accept": "text/html,application/xhtml+xml"}


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_ADMIN_EMAILS=ADMIN_EMAIL, PLO5BP_HOMEGAME_MAX_TABLES="1000")


@pytest.fixture(scope="module")
def hg(server):
    return sys.modules["plo5bp.ui.homegame"]


def _client(server, email=None, name=None):
    c = TestClient(server.app, raise_server_exceptions=False)
    if email:
        params = {"email": email}
        if name:
            params["name"] = name
        assert c.get("/auth/dev", params=params).status_code == 200
    return c


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _club(c, name):
    return _ok(c.post("/games/api/clubs", json={"name": name}))


def _table(c, club_id, name="t", **kw):
    body = {"name": name, "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 4000, "club_id": club_id, **kw}
    return _ok(c.post("/games/api/tables", json=body))


def _check_down(players, gid, first_raise=False):
    """Everyone checks / calls to the end of the hand (the first actor bets
    the minimum when ``first_raise``). ``players`` = clients by user id."""
    raised = not first_raise
    for _ in range(60):
        s = next(iter(players.values())).get(f"/games/api/tables/{gid}").json()
        if s["phase"] != "in_hand" or s["actor"] is None:
            return s
        cl = players[s["seats"][s["actor"]]["user_id"]]
        me = cl.get(f"/games/api/tables/{gid}").json()
        body = {"hand_no": me["hand_no"], "action_seq": me["action_seq"], "gate": "check_call"}
        if not raised and me["legal"]["raise"]:
            body.update(gate="raise", chips=me["raise_bounds"]["min_chips"])
            raised = True
        _ok(cl.post(f"/games/api/tables/{gid}/act", json=body))
    raise AssertionError("the hand did not end")


def _finish(hg, gid):
    t = hg.HUB.get(gid)
    with t.lock:
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0


def _ids(admin):
    return {u["email"]: u["id"] for u in _ok(admin.get("/admin/api/users"))["users"]}


# --- signed out -------------------------------------------------------------------


def test_signed_out_links_get_a_sign_in_page_that_comes_back(server):
    owner = _client(server, "owner1@clubs.example", "Oona")
    club = _club(owner, "Friday <Club>")
    t = _table(owner, club["id"], "Friday <PLO>")
    anon = _client(server)
    r = anon.get(f"/games/t/{t['id']}", headers=HTML)
    assert r.status_code == 200 and "You're invited to Friday &lt;PLO&gt;" in r.text
    assert "<b>Oona</b> is hosting" in r.text and "Friday &lt;Club&gt;" in r.text
    assert f'href="/auth/login?next=/games/t/{t["id"]}"' in r.text
    assert "default-src 'none'" in r.headers["content-security-policy"] and "<script" not in r.text
    r = anon.get(f"/games/join/{club['invite_code']}", headers=HTML)
    assert r.status_code == 200 and "Join Friday &lt;Club&gt;" in r.text
    assert f'next=/games/join/{club["invite_code"]}' in r.text
    r = anon.get("/games", headers=HTML)
    assert r.status_code == 200 and "Sign in with Google" in r.text
    # everything else stays the hidden 404: the API, the client, unknown / stale links
    for path in (f"/games/api/tables/{t['id']}", "/games/static/games.js", "/games/t/nope", "/games/join/nope-nope",
                 f"/games/api/invites/{club['invite_code']}"):
        assert anon.get(path, headers=HTML).status_code == 404, path
    assert anon.get(f"/games/t/{t['id']}", headers={"accept": "application/json"}).status_code == 404


def test_sign_in_comes_back_to_a_home_games_link_and_nowhere_else(server):
    c = TestClient(server.app, raise_server_exceptions=False, follow_redirects=False)
    for good in ("/games", "/games/t/abc123", "/games/join/abcdefgh"):
        r = c.get("/auth/dev", params={"email": "back@clubs.example", "next": good})
        assert r.status_code in (302, 303, 307) and r.headers["location"] == good, good
    for bad in ("https://evil.example/games", "//evil.example", "/admin", "/games/t/x/../../admin", "/games/join/x"):
        r = c.get("/auth/dev", params={"email": "back@clubs.example", "next": bad})
        assert r.headers["location"] == "/", bad


# --- starting a club, hosting, the invite link ------------------------------------------


def test_anyone_can_start_a_club_host_in_it_and_nobody_else_sees_it(server):
    ann = _client(server, "ann@clubs.example", "Ann")
    stranger = _client(server, "stranger@clubs.example", "Stan")
    assert _ok(ann.get("/games/api/tables"))["clubs"] == []
    # hosting needs a club (the lobby offers to start one)
    r = ann.post("/games/api/tables", json={"name": "x", "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 4000})
    assert r.status_code == 400 and "club" in r.text
    club = _club(ann, "  Ann's   crew  ")
    assert club["name"] == "Ann's crew" and club["role"] == "owner" and club["invite_code"]
    assert [m["role"] for m in club["members"]] == ["owner"]
    t = _table(ann, club["id"], "Ann's game")
    assert t["club"]["id"] == club["id"] and t["club"]["name"] == "Ann's crew" and t["club"]["role"] == "owner"
    lobby = _ok(ann.get(f"/games/api/tables?club={club['id']}"))
    assert [x["id"] for x in lobby["tables"]] == [t["id"]] and lobby["tables"][0]["club_name"] == "Ann's crew"
    assert lobby["clubs"][0]["open_tables"] == 1 and lobby["clubs"][0]["members"] == 1
    # a stranger: no clubs, no tables, a 403 naming the club on the table link, 404 on the club
    assert _ok(stranger.get("/games/api/tables"))["tables"] == []
    r = stranger.get(f"/games/api/tables/{t['id']}")
    assert r.status_code == 403
    d = r.json()["detail"]
    assert d["error"] == "club" and d["club"] == {"id": club["id"], "name": "Ann's crew"} and d["request"] is None
    for path in (f"/games/api/clubs/{club['id']}", f"/games/api/tables?club={club['id']}",
                 f"/games/api/community?club={club['id']}"):
        assert stranger.get(path).status_code == 404, path
    for what in ("sit", "chat", "act", "leave"):
        assert stranger.post(f"/games/api/tables/{t['id']}/{what}", json={"seat": 2, "text": "hi"}).status_code == 403
    assert stranger.get(f"/games/api/tables/{t['id']}/stream").status_code == 403
    # names are limited; five clubs a person
    assert ann.post("/games/api/clubs", json={"name": "x" * 41}).status_code == 400
    assert ann.post("/games/api/clubs", json={"name": "   "}).status_code == 400
    for k in range(4):
        _club(ann, f"more {k}")
    assert ann.post("/games/api/clubs", json={"name": "one too many"}).status_code == 429


def test_the_invite_link_joins_the_club_and_a_new_link_kills_the_old_one(server):
    ben = _client(server, "ben@clubs.example", "Ben")
    cal = _client(server, "cal@clubs.example", "Cal")
    dee = _client(server, "dee@clubs.example", "Dee")
    club = _club(ben, "Ben's")
    t = _table(ben, club["id"])
    code = club["invite_code"]
    info = _ok(cal.get(f"/games/api/invites/{code}"))
    assert info["club"]["name"] == "Ben's" and info["club"]["owner_name"] == "Ben" and info["member"] is False
    joined = _ok(cal.post(f"/games/api/invites/{code}/join"))
    assert joined["member"] is True
    assert _ok(cal.get("/games/api/clubs"))["clubs"][0]["role"] == "member"
    # now Cal sees the table and can sit
    _ok(cal.post(f"/games/api/tables/{t['id']}/sit", json={"seat": 2, "buyin_cents": 4000}))
    # a member does not see the invite code or the join requests
    view = _ok(cal.get(f"/games/api/clubs/{club['id']}"))
    assert view["invite_code"] is None and view["requests"] == []
    # a new link: the old one is dead, the new one works
    fresh = _ok(ben.post(f"/games/api/clubs/{club['id']}/invite"))["invite_code"]
    assert fresh != code
    assert dee.get(f"/games/api/invites/{code}").status_code == 404
    assert dee.post(f"/games/api/invites/{code}/join").status_code == 404
    assert _ok(dee.post(f"/games/api/invites/{fresh}/join"))["member"] is True
    # a member can't reset the link
    assert cal.post(f"/games/api/clubs/{club['id']}/invite").status_code == 403


# --- asking first ---------------------------------------------------------------------


def test_an_ask_first_club_and_a_table_link_request_wait_for_the_owner_or_an_admin(server, hg):
    eve = _client(server, "eve@clubs.example", "Eve")
    fay = _client(server, "fay@clubs.example", "Fay")
    gus = _client(server, "gus@clubs.example", "Gus")
    hal = _client(server, "hal@clubs.example", "Hal")
    club = _club(eve, "Eve's")
    _ok(eve.post(f"/games/api/clubs/{club['id']}/settings", json={"approve_joins": True}))
    t = _table(eve, club["id"])
    _ok(fay.post(f"/games/api/invites/{club['invite_code']}/join"))  # (asks first now)
    info = _ok(fay.get(f"/games/api/invites/{club['invite_code']}"))
    assert info["member"] is False and info["approve"] is True and info["request"] == "pending"
    assert fay.get(f"/games/api/tables/{t['id']}").json()["detail"]["request"] == "pending"
    # Gus asks from the table link
    r = gus.get(f"/games/api/tables/{t['id']}")
    assert r.status_code == 403
    asked = _ok(gus.post(f"/games/api/clubs/{r.json()['detail']['club']['id']}/request"))
    assert asked["request"] == "pending" and asked["member"] is False
    # the owner sees both, oldest first, with a masked email — in the lobby and at the table
    lobby = _ok(eve.get(f"/games/api/tables?club={club['id']}"))
    assert [(q["name"], q["email"]) for q in lobby["join_requests"]] == [("Fay", "f•••@clubs.example"), ("Gus", "g•••@clubs.example")]
    assert lobby["clubs"][0]["requests"] == 2
    hg.HUB.get(t["id"]).join_checked_mono = 0.0
    at_table = _ok(eve.get(f"/games/api/tables/{t['id']}"))["join_requests"]
    assert [q["name"] for q in at_table] == ["Fay", "Gus"] and "fay@clubs.example" not in str(at_table)
    # Hal joins as a member (Eve lets him in straight away) and becomes an admin
    _ok(hal.post(f"/games/api/invites/{club['invite_code']}/join"))
    ids = {m["name"]: m["user_id"] for m in _ok(eve.get(f"/games/api/clubs/{club['id']}"))["requests"]}
    _ok(eve.post(f"/games/api/clubs/{club['id']}/requests/decide", json={"user_id": _hal_id(eve, club, hal), "allow": True}))
    assert _ok(hal.get(f"/games/api/tables?club={club['id']}"))["join_requests"] == [], "a plain member decides nothing"
    assert hal.post(f"/games/api/clubs/{club['id']}/requests/decide", json={"user_id": ids["Fay"], "allow": True}).status_code == 403
    hal_id = _hal_id(eve, club, hal)
    _ok(eve.post(f"/games/api/clubs/{club['id']}/members", json={"user_id": hal_id, "role": "admin"}))
    assert len(_ok(hal.get(f"/games/api/tables?club={club['id']}"))["join_requests"]) == 2
    # the admin lets Fay in, declines Gus
    assert _ok(hal.post(f"/games/api/clubs/{club['id']}/requests/decide", json={"user_id": ids["Fay"], "allow": True}))["allowed"]
    assert fay.get(f"/games/api/tables/{t['id']}").status_code == 200
    assert hal.post(f"/games/api/clubs/{club['id']}/requests/decide", json={"user_id": ids["Fay"], "allow": True}).status_code == 409
    _ok(hal.post(f"/games/api/clubs/{club['id']}/requests/decide", json={"user_id": ids["Gus"], "allow": False}))
    d = gus.get(f"/games/api/tables/{t['id']}").json()["detail"]
    assert d["request"] == "declined" and d["retry_in"] > 0
    _ok(gus.post(f"/games/api/clubs/{club['id']}/request"))  # too soon: ignored
    assert gus.get(f"/games/api/tables/{t['id']}").json()["detail"]["request"] == "declined"
    hg.pub.DB.q("UPDATE homegame_club_requests SET decided_at='2000-01-01T00:00:00+00:00' WHERE user_id=?", (ids["Gus"],))
    assert _ok(gus.post(f"/games/api/clubs/{club['id']}/request"))["request"] == "pending"


def _hal_id(owner, club, hal):
    view = _ok(owner.get(f"/games/api/clubs/{club['id']}"))
    for m in view["members"]:
        if m["name"] == "Hal":
            return m["user_id"]
    for q in view["requests"]:
        if q["name"] == "Hal":
            return q["user_id"]
    raise AssertionError("no Hal")


# --- roles ---------------------------------------------------------------------------


def test_roles_removal_leaving_and_handing_the_club_over(server):
    ivy = _client(server, "ivy@clubs.example", "Ivy")
    jon = _client(server, "jon@clubs.example", "Jon")
    kai = _client(server, "kai@clubs.example", "Kai")
    lee = _client(server, "lee@clubs.example", "Lee")
    club = _club(ivy, "Ivy's")
    for c in (jon, kai, lee):
        _ok(c.post(f"/games/api/invites/{club['invite_code']}/join"))
    m = {x["name"]: x["user_id"] for x in _ok(ivy.get(f"/games/api/clubs/{club['id']}"))["members"]}
    cid = club["id"]
    # a member can do nothing to others
    assert kai.post(f"/games/api/clubs/{cid}/members", json={"user_id": m["Lee"], "remove": True}).status_code == 403
    assert kai.post(f"/games/api/clubs/{cid}/settings", json={"name": "mine"}).status_code == 403
    # an admin removes members, not admins or the owner, and changes no roles
    _ok(ivy.post(f"/games/api/clubs/{cid}/members", json={"user_id": m["Jon"], "role": "admin"}))
    _ok(ivy.post(f"/games/api/clubs/{cid}/members", json={"user_id": m["Kai"], "role": "admin"}))
    assert jon.post(f"/games/api/clubs/{cid}/members", json={"user_id": m["Kai"], "remove": True}).status_code == 403
    assert jon.post(f"/games/api/clubs/{cid}/members", json={"user_id": m["Ivy"], "remove": True}).status_code == 403
    assert jon.post(f"/games/api/clubs/{cid}/members", json={"user_id": m["Lee"], "role": "admin"}).status_code == 403
    # someone with a seat at the club's table can't be removed (or leave) until they stand up
    t = _table(ivy, cid)
    _ok(lee.post(f"/games/api/tables/{t['id']}/sit", json={"seat": 3, "buyin_cents": 4000}))
    r = jon.post(f"/games/api/clubs/{cid}/members", json={"user_id": m["Lee"], "remove": True})
    assert r.status_code == 409 and "leave the table first" in r.text
    assert lee.post(f"/games/api/clubs/{cid}/leave").status_code == 409
    _ok(lee.post(f"/games/api/tables/{t['id']}/leave", json={}))
    _ok(jon.post(f"/games/api/clubs/{cid}/members", json={"user_id": m["Lee"], "remove": True}))
    assert lee.get(f"/games/api/tables/{t['id']}").status_code == 403, "removed: the table is closed to them"
    assert lee.get(f"/games/api/clubs/{cid}").status_code == 404
    # the owner can't leave; handing the club over makes the old owner an admin
    assert ivy.post(f"/games/api/clubs/{cid}/leave").status_code == 400
    view = _ok(ivy.post(f"/games/api/clubs/{cid}/members", json={"user_id": m["Kai"], "role": "owner"}))
    roles = {x["name"]: x["role"] for x in view["members"]}
    assert roles["Kai"] == "owner" and roles["Ivy"] == "admin"
    assert _ok(kai.post(f"/games/api/clubs/{cid}/settings", json={"name": "Kai's now"}))["name"] == "Kai's now"
    assert ivy.post(f"/games/api/clubs/{cid}/settings", json={"name": "back"}).status_code == 403
    # an admin may leave
    _ok(jon.post(f"/games/api/clubs/{cid}/leave"))
    assert _ok(jon.get("/games/api/clubs"))["clubs"] == []


# --- the numbers stay in the club ---------------------------------------------------------


def test_every_statistic_stays_inside_its_club(server, hg):
    """Two clubs, the same two players, a hand in each: each club's rankings, the
    players' numbers there and the head-to-head show that club's hand only; an
    outsider can't read either club's numbers."""
    mo = _client(server, "mo@clubs.example", "Mo")
    nia = _client(server, "nia@clubs.example", "Nia")
    out = _client(server, "out@clubs.example", "Out")
    a, b = _club(mo, "Club A"), _club(mo, "Club B")
    for club in (a, b):
        _ok(nia.post(f"/games/api/invites/{club['invite_code']}/join"))
    ids = {x["name"]: x["user_id"] for x in _ok(mo.get(f"/games/api/clubs/{a['id']}"))["members"]}
    players = {ids["Mo"]: mo, ids["Nia"]: nia}
    hands = {}
    for club in (a, b):
        t = _table(mo, club["id"], f"table {club['name']}")
        _ok(nia.post(f"/games/api/tables/{t['id']}/sit", json={"seat": 1, "buyin_cents": 4000}))
        _ok(mo.post(f"/games/api/tables/{t['id']}/run", json={"running": True}))
        _check_down(players, t["id"], first_raise=club is a)
        _finish(hg, t["id"])
        _ok(mo.get(f"/games/api/tables/{t['id']}"))
        _ok(mo.post(f"/games/api/tables/{t['id']}/run", json={"running": False}))
        hands[club["id"]] = t["id"]
    for club in (a, b):
        cid, gid = club["id"], hands[club["id"]]
        comm = _ok(nia.get(f"/games/api/community?club={cid}"))
        assert {p["name"] for p in comm["players"]} == {"Mo", "Nia"}
        assert all(p["hands"] == 1 for p in comm["players"]), comm["players"]
        assert [s["id"] for s in comm["sessions"]] == [gid]
        mine = _ok(nia.get(f"/games/api/my/stats?club={cid}"))
        assert mine["hands"] == 1 and [s["id"] for s in mine["sessions"]] == [gid]
        theirs = _ok(nia.get(f"/games/api/players/{ids['Mo']}/stats?club={cid}"))
        assert theirs["hands"] == 1 and [s["id"] for s in theirs["sessions"]] == [gid]
        hl = _ok(nia.get(f"/games/api/players/{ids['Mo']}/hands?club={cid}"))
        assert {h["game_id"] for h in hl["hands"]} == {gid}
        assert out.get(f"/games/api/community?club={cid}").status_code == 404
        assert out.get(f"/games/api/players/{ids['Mo']}/stats?club={cid}").status_code == 404
        assert out.get(f"/games/api/players/{ids['Mo']}/hands?club={cid}").status_code == 404
    # own numbers without a club = everything I played; someone else's = only the clubs I share
    assert _ok(nia.get("/games/api/my/stats"))["hands"] == 2
    assert _ok(out.get(f"/games/api/players/{ids['Mo']}/stats"))["hands"] == 0
    assert _ok(out.get(f"/games/api/players/{ids['Mo']}/hands"))["hands"] == []
    # the lobby's sessions are the club's
    for club in (a, b):
        sess = _ok(mo.get(f"/games/api/tables?club={club['id']}"))["sessions"]
        assert all(s["id"] == hands[club["id"]] for s in sess)


def test_only_the_clubs_owner_can_take_a_session_out_of_the_stats(server, hg):
    pam = _client(server, "pam@clubs.example", "Pam")
    quin = _client(server, "quin@clubs.example", "Quin")
    club = _club(pam, "Pam's")
    _ok(quin.post(f"/games/api/invites/{club['invite_code']}/join"))
    t = _table(quin, club["id"], "Quin hosts")  # any member may host
    assert quin.post(f"/games/api/tables/{t['id']}/exclude", json={"on": True}).status_code == 403
    comm = _ok(quin.get(f"/games/api/community?club={club['id']}"))
    assert comm["can_manage"] is False
    assert _ok(pam.get(f"/games/api/community?club={club['id']}"))["can_manage"] is True
    _ok(pam.post(f"/games/api/tables/{t['id']}/exclude", json={"on": True}))
    assert hg.pub.DB.one("SELECT excluded FROM homegames WHERE id=?", (t["id"],))["excluded"] == 1
    assert t["id"] not in [s["id"] for s in _ok(quin.get(f"/games/api/community?club={club['id']}"))["sessions"]]
    assert t["id"] in [s["id"] for s in _ok(pam.get(f"/games/api/community?club={club['id']}"))["sessions"]]


def test_the_admin_switch_is_the_main_club(server):
    adm = _client(server, ADMIN_EMAIL, "Admin")
    rae = _client(server, "rae@clubs.example", "Rae")
    ids = _ids(adm)
    _ok(adm.post("/admin/api/games_access", json={"user_id": ids["rae@clubs.example"], "action": "grant"}))
    clubs = _ok(rae.get("/games/api/clubs"))["clubs"]
    assert len(clubs) == 1 and clubs[0]["is_main"] is True
    main = _ok(adm.get(f"/games/api/clubs/{clubs[0]['id']}"))
    assert main["role"] == "owner" and main["name"] == "Admin's club"
    # a table from someone in the main club with no club named lands there (older clients)
    r = rae.post("/games/api/tables", json={"name": "old client", "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 4000})
    assert _ok(r)["club"]["id"] == clubs[0]["id"]
    # still seated, then still the host of an open table: not yet
    gid = r.json()["id"]
    assert adm.post("/admin/api/games_access", json={"user_id": ids["rae@clubs.example"], "action": "revoke"}).status_code == 409
    _ok(rae.post(f"/games/api/tables/{gid}/leave", json={}))
    busy = adm.post("/admin/api/games_access", json={"user_id": ids["rae@clubs.example"], "action": "revoke"})
    assert busy.status_code == 409 and "hosts" in busy.text
    _ok(rae.post(f"/games/api/tables/{gid}/close", json={}))
    _ok(adm.post("/admin/api/games_access", json={"user_id": ids["rae@clubs.example"], "action": "revoke"}))
    assert _ok(rae.get("/games/api/clubs"))["clubs"] == []
