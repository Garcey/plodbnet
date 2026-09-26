"""A database created by the build that was live on 2026-09-02 must upgrade IN
PLACE when the current code starts on it (first production ship of the premium
tables / tracking / verifiable shuffle). A schema surprise here is not a failing
feature — it is a site that does not start.

``OLD_HOMEGAME_SCHEMA`` is frozen from the pre-review snapshot of homegame.py
(the closest copy of what was live); the ``oldest`` variant also drops the
columns that build added with ALTER, so both generations of rows are covered.
"""
from __future__ import annotations

import sqlite3
import sys

from starlette.testclient import TestClient

OLD_USERS_SCHEMA = """
CREATE TABLE users (
  id INTEGER PRIMARY KEY, google_sub TEXT UNIQUE, email TEXT UNIQUE NOT NULL,
  name TEXT DEFAULT '', picture TEXT DEFAULT '', created_at TEXT NOT NULL, last_login_at TEXT,
  sub_status TEXT NOT NULL DEFAULT 'none', sub_source TEXT NOT NULL DEFAULT '',
  stripe_customer_id TEXT, stripe_subscription_id TEXT, current_period_end TEXT,
  sub_checked_at TEXT, homegame_access INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE usage (user_id INTEGER NOT NULL, day TEXT NOT NULL, hands INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (user_id, day));
CREATE TABLE payments (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, stripe_ref TEXT UNIQUE NOT NULL,
  amount_cents INTEGER NOT NULL, currency TEXT NOT NULL DEFAULT 'usd', created_at TEXT NOT NULL);
CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

OLD_HOMEGAME_SCHEMA = """
CREATE TABLE homegames (
  id TEXT PRIMARY KEY, host_user_id INTEGER NOT NULL, name TEXT NOT NULL, num_seats INTEGER NOT NULL,
  sb_cents INTEGER NOT NULL, bb_cents INTEGER NOT NULL, ante_cents INTEGER NOT NULL,
  default_buyin_cents INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'open',
  {late_game_cols}
  button INTEGER NOT NULL DEFAULT 0, hand_no INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL, closed_at TEXT
);
CREATE TABLE homegame_players (
  game_id TEXT NOT NULL, user_id INTEGER NOT NULL, seat INTEGER, stack_chips INTEGER NOT NULL DEFAULT 0,
  sitting_out INTEGER NOT NULL DEFAULT 0, buyin_cents INTEGER NOT NULL DEFAULT 0,
  leftover_cents INTEGER NOT NULL DEFAULT 0{late_player_cols},
  PRIMARY KEY (game_id, user_id)
);
CREATE TABLE homegame_ledger (id INTEGER PRIMARY KEY, game_id TEXT NOT NULL, user_id INTEGER NOT NULL,
  kind TEXT NOT NULL, amount_cents INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE homegame_chat (id INTEGER PRIMARY KEY, game_id TEXT NOT NULL, user_id INTEGER NOT NULL,
  body TEXT NOT NULL, created_at TEXT NOT NULL);
"""
LATE_GAME_COLS = """running INTEGER NOT NULL DEFAULT 0, auto_stack_mode TEXT NOT NULL DEFAULT 'off',
  auto_stack_all_cents INTEGER NOT NULL DEFAULT 0, decision_secs INTEGER NOT NULL DEFAULT 0,
  street_pause_ms INTEGER NOT NULL DEFAULT 1500,"""
NOW = "2026-09-02T20:00:00+00:00"
PEOPLE = [("host@old.example", "Holly"), ("ray@old.example", "Ray"), ("sue@old.example", "Sue")]


def _old_database(path, oldest: bool) -> None:
    con = sqlite3.connect(path)
    con.executescript(OLD_USERS_SCHEMA)
    con.executescript(OLD_HOMEGAME_SCHEMA.format(
        late_game_cols="" if oldest else LATE_GAME_COLS,
        late_player_cols="" if oldest else ", auto_stack_cents INTEGER NOT NULL DEFAULT 0"))
    for i, (email, name) in enumerate(PEOPLE, start=1):
        con.execute("INSERT INTO users(id,email,name,created_at,homegame_access) VALUES(?,?,?,?,1)",
                    (i, email, name, NOW))
    # a table that was OPEN with money on it when the old build was stopped, and a closed one
    con.execute("INSERT INTO homegames(id,host_user_id,name,num_seats,sb_cents,bb_cents,ante_cents,"
                "default_buyin_cents,status,button,hand_no,created_at) VALUES"
                "('oldopen1',1,'Friday game',6,50,100,300,20000,'open',2,41,?)", (NOW,))
    con.execute("INSERT INTO homegames(id,host_user_id,name,num_seats,sb_cents,bb_cents,ante_cents,"
                "default_buyin_cents,status,button,hand_no,created_at,closed_at) VALUES"
                "('oldshut1',1,'Last week',6,50,100,300,20000,'closed',0,88,?,?)", (NOW, NOW))
    for uid, seat, chips in ((1, 0, 2_350_000), (2, 1, 1_800_000), (3, 3, 1_850_000)):
        con.execute("INSERT INTO homegame_players(game_id,user_id,seat,stack_chips,buyin_cents) "
                    "VALUES('oldopen1',?,?,?,20000)", (uid, seat, chips))
        con.execute("INSERT INTO homegame_ledger(game_id,user_id,kind,amount_cents,created_at) "
                    "VALUES('oldopen1',?,'buyin',20000,?)", (uid, NOW))
    con.execute("INSERT INTO homegame_players(game_id,user_id,seat,stack_chips,buyin_cents,leftover_cents) "
                "VALUES('oldshut1',1,NULL,0,20000,26150)")
    con.execute("INSERT INTO homegame_players(game_id,user_id,seat,stack_chips,buyin_cents,leftover_cents) "
                "VALUES('oldshut1',2,NULL,0,20000,13850)")
    con.execute("INSERT INTO homegame_chat(game_id,user_id,body,created_at) VALUES('oldopen1',2,'gg',?)", (NOW,))
    con.commit()
    con.close()


def test_the_september_database_upgrades_in_place_and_the_table_plays_on(tmp_path, boot_public_server):
    # The OLDEST shape is the superset case: every ALTER the September build did AND
    # every one added since has to happen (each is conditional on the column being
    # absent, so the as-live shape — verified by hand on 2026-09-21 — is a subset).
    db = tmp_path / "public.db"
    _old_database(db, oldest=True)
    server = boot_public_server(PLO5BP_DB=str(db), PLO5BP_HOMEGAME_GRADING="0")
    hg = sys.modules["plo5bp.ui.homegame"]

    def login(email, name):  # (a sign-in refreshes the display name, as Google's does)
        cl = TestClient(server.app, raise_server_exceptions=False)
        assert cl.get("/auth/dev", params={"email": email, "name": name}).status_code == 200
        return cl

    host, ray, sue = (login(e, n) for e, n in PEOPLE)
    # the lobby and the old table survive, money intact (chips -> cents at $1/bb, 10000 chips/bb)
    lobby = host.get("/games/api/tables")
    assert lobby.status_code == 200, lobby.text
    assert [t["id"] for t in lobby.json()["tables"]] == ["oldopen1"]
    assert any(s["id"] == "oldshut1" for s in lobby.json()["sessions"])
    s = host.get("/games/api/tables/oldopen1").json()
    assert s["hand_no"] == 41 and s["my_seat"] == 0
    assert [(x["name"], x["stack_cents"]) for x in s["seats"] if not x["empty"]] == [
        ("Holly", 23500), ("Ray", 18000), ("Sue", 18500)]
    assert sum(r["net_cents"] for r in s["ledger"]) == 0, "the ledger still balances"
    # columns the old rows never had come up with their documented defaults
    assert s["settings"]["approve_buyins"] is False and s["settings"]["show_grades"] is True
    assert s["settings"]["listed"] is True and s["settings"]["allow_rabbit"] is True
    assert s["fair"]["supported"] == hg.FAIR_ON and (not hg.FAIR_ON or s["fair"]["next"]["hand_no"] == 42)
    # … and the table simply plays on: start, deal hand 42, fold it out, history + club stats work
    r = host.post("/games/api/tables/oldopen1/run", json={"running": True})
    assert r.status_code == 200 and r.json()["phase"] == "in_hand" and r.json()["hand_no"] == 42
    by_seat = {0: host, 1: ray, 3: sue}
    for _ in range(12):
        st = host.get("/games/api/tables/oldopen1").json()
        if st["phase"] != "in_hand":
            break
        cl = by_seat[st["actor"]]
        mine = cl.get("/games/api/tables/oldopen1").json()
        gate = "fold" if mine["legal"]["fold"] else ("raise" if st["action_seq"] == 0 else "check_call")
        body = {"gate": gate, "hand_no": st["hand_no"], "action_seq": st["action_seq"]}
        if gate == "raise":
            body["raise_to_chips"] = mine["raise_bounds"]["min_chips"] + mine["street_commit_chips"]
        assert cl.post("/games/api/tables/oldopen1/act", json=body).status_code == 200
    t = hg.HUB.get("oldopen1")
    with t.lock:
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0
    assert host.get("/games/api/tables/oldopen1").json()["phase"] == "showdown"
    hist = host.get("/games/api/tables/oldopen1/hands").json()
    assert [h["hand_no"] for h in hist["hands"]] == [42], "hands from before the upgrade were never recorded; new ones are"
    assert host.get("/games/api/tables/oldopen1/hands/42").status_code == 200
    club = host.get("/games/api/community").json()
    assert {p["name"] for p in club["players"]} == {"Holly", "Ray", "Sue"}
    assert {x["id"] for x in club["sessions"]} >= {"oldopen1", "oldshut1"}
    # clubs (2026-09-25): the old circle became the MAIN club — every table that predates
    # clubs, everyone who played or had the old flag — so its numbers carry on unchanged
    clubs = ray.get("/games/api/clubs").json()["clubs"]
    assert len(clubs) == 1 and clubs[0]["is_main"] and clubs[0]["open_tables"] == 1
    main = host.get(f"/games/api/clubs/{clubs[0]['id']}").json()
    assert {(m["name"], m["role"]) for m in main["members"]} == {("Holly", "owner"), ("Ray", "member"), ("Sue", "member")}
    assert main["name"] == "Holly's club"
    # every new table / column exists exactly once, and a SECOND start on the same file is a no-op
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert {"homegame_hands", "homegame_hand_results", "homegame_flows", "homegame_fair",
            "homegame_clubs", "homegame_club_members", "homegame_club_requests"} <= tables
    cols = {r[1] for r in con.execute("pragma table_info(homegames)")}
    assert {"running", "deal_delay_ms", "time_bank_secs", "approve_buyins", "topup_mode", "show_grades", "excluded",
            "club_id"} <= cols
    assert {r[0] for r in con.execute("select club_id from homegames")} == {clubs[0]["id"]}
    assert {"auto_stack_cents", "trusted", "topup_target_cents"} <= {r[1] for r in con.execute("pragma table_info(homegame_players)")}
    con.close()
    hg._ensure_schema()
    assert hg.pub.DB.one("SELECT COUNT(*) c FROM homegame_clubs")["c"] == 1
