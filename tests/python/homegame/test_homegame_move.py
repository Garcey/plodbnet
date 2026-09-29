"""Home games: changing seats (2026-09-28, FEAT-013). A seated player who holds no cards
takes an empty seat with everything the seat carries — stack, ledger totals, automatic
chips — through POST /games/api/tables/{id}/move {"seat": n}. Someone in the hand moves
once it is over (the client sends the move then); a move never lands while the next
hand's shuffle is being confirmed, and a shuffle commitment made from the old seat (it
names that seat) is dropped so the device can commit again from the new one."""
from __future__ import annotations

import sys

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "admin@move.example"
NAMES = ["mo", "nia", "ola", "pip"]


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
    players = [login(f"{n}@move.example") for n in NAMES]
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for n in NAMES:
        assert adm.post("/admin/api/games_access", json={"user_id": ids[f"{n}@move.example"], "action": "grant"}).status_code == 200
    by_uid = {ids[f"{n}@move.example"]: players[i] for i, n in enumerate(NAMES)}
    return {"p": players, "by_uid": by_uid, "ids": [ids[f"{n}@move.example"] for n in NAMES]}


def _post(cl, gid, what, body=None):
    return cl.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _table(cast, n, **kw):
    body = {"name": "move", "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 4000, "num_seats": 6, **kw}
    gid = _ok(cast["p"][0].post("/games/api/tables", json=body))["id"]
    for i in range(1, n):
        _ok(_post(cast["p"][i], gid, "sit", {"seat": i, "buyin_cents": 4000 + 100 * i}))
    return gid


def _check_down(cast, gid):
    for _ in range(60):
        s = _ok(cast["p"][0].get(f"/games/api/tables/{gid}"))
        if s["phase"] != "in_hand" or s["actor"] is None:
            return s
        cl = cast["by_uid"][s["seats"][s["actor"]]["user_id"]]
        st = _ok(cl.get(f"/games/api/tables/{gid}"))
        _ok(_post(cl, gid, "act", {"gate": "check_call", "hand_no": st["hand_no"], "action_seq": st["action_seq"]}))
    raise AssertionError("hand did not end")


def _finish_runout(hg, gid):
    t = hg.HUB.get(gid)
    with t.lock:
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0


def test_a_player_moves_to_an_empty_seat_with_everything_the_seat_carries(cast, hg):
    p = cast["p"]
    gid = _table(cast, 3)
    before = _ok(p[1].get(f"/games/api/tables/{gid}"))
    s = _ok(_post(p[1], gid, "move", {"seat": 4}))
    assert s["my_seat"] == 4 and s["seats"][1]["empty"] is True
    moved = s["seats"][4]
    assert moved["user_id"] == cast["ids"][1] and moved["stack_cents"] == before["seats"][1]["stack_cents"] == 4100
    row = next(r for r in s["ledger"] if r["user_id"] == cast["ids"][1])
    assert row["buyin_cents"] == 4100 and row["net_cents"] == 0 and row["seated"] is True
    assert any(e["kind"] == "move" and e["text"].endswith("moved to seat 5") for e in s["events"])
    # (persisted: a reload of the table finds them in the new seat)
    db = hg.pub.DB.one("SELECT seat FROM homegame_players WHERE game_id=? AND user_id=?", (gid, cast["ids"][1]))
    assert db["seat"] == 4
    # the same seat again is nothing; a taken, a missing or a bad seat is refused
    assert _ok(_post(p[1], gid, "move", {"seat": 4}))["my_seat"] == 4
    assert _post(p[1], gid, "move", {"seat": 2}).status_code == 409
    assert _post(p[1], gid, "move", {"seat": 9}).status_code == 400
    assert _post(p[3], gid, "move", {"seat": 5}).status_code == 400  # (not seated)


def test_in_the_hand_you_move_after_it_and_a_player_out_of_it_moves_now(cast, hg):
    p = cast["p"]
    gid = _table(cast, 3)
    _ok(_post(p[2], gid, "sit_out", {"on": True}))  # ola is not dealt in
    _ok(_post(p[0], gid, "run", {"running": True}))
    s = _ok(p[1].get(f"/games/api/tables/{gid}"))
    assert s["phase"] == "in_hand" and s["seats"][1]["in_hand"] is True
    r = _post(p[1], gid, "move", {"seat": 5})
    assert r.status_code == 409 and "hand is over" in r.json()["detail"]
    ola = _ok(p[2].get(f"/games/api/tables/{gid}"))["seats"][2]["stack_cents"]
    out = _ok(_post(p[2], gid, "move", {"seat": 3}))  # (not in the hand: at once)
    assert out["my_seat"] == 3
    _ok(_post(p[0], gid, "run", {"running": False}))
    done = _check_down(cast, gid)
    _finish_runout(hg, gid)
    done = _ok(p[2].get(f"/games/api/tables/{gid}"))
    assert done["seats"][3]["stack_cents"] == ola, "the hand's end never touches a seat that was not in it"
    assert done["seats"][2]["empty"] is True
    assert sum(r["net_cents"] for r in done["ledger"]) == 0
    moved = _ok(_post(p[1], gid, "move", {"seat": 5}))  # (the hand is over now)
    assert moved["my_seat"] == 5


def test_a_move_drops_the_old_seats_shuffle_commitment_and_waits_for_a_deal(cast, hg):
    if not hg.FAIR_ON:
        pytest.skip("the verified shuffle is off on this engine")
    p = cast["p"]
    gid = _table(cast, 2)
    t = hg.HUB.get(gid)
    with t.lock:
        hg._fair_prepare_locked(t)
        nxt = t.fair_next
        nxt.commits[1] = "ab" * 32
        nxt.commit_users[1] = cast["ids"][1]
    _ok(_post(p[1], gid, "move", {"seat": 3}))
    with t.lock:
        assert 1 not in t.fair_next.commits and 1 not in t.fair_next.commit_users
        t.fair_next.pending = True  # (a deal is waiting on the shuffle, a while yet)
        t.fair_next.deadline_mono = __import__("time").monotonic() + 600
    r = _post(p[1], gid, "move", {"seat": 4})
    assert r.status_code == 409 and "being dealt" in r.json()["detail"]
    with t.lock:
        t.fair_next.pending = False
