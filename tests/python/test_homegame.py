"""Private PLO5 home-game table: access, sit/deal/act, hole privacy, ledger."""

from __future__ import annotations

import importlib
import os
import time

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"

_ENV = {
    "PLO5BP_PUBLIC": "1",
    "PLO5BP_DEV_LOGIN": "1",
    # TestClient's client host is "testclient": opt in explicitly (review F4).
    "PLO5BP_DEV_LOGIN_TESTCLIENT": "1",
    "PLO5BP_BASE_URL": "http://127.0.0.1:8770",
    "PLO5BP_ADMIN_EMAILS": ADMIN_EMAIL,
    # Every test hosts a fresh table as alice and never closes it; the
    # per-host cap (review G13) is tested in test_review_homegame_fixes.py.
    "PLO5BP_HOMEGAME_MAX_TABLES": "1000",
}


@pytest.fixture(scope="module")
def server(tmp_path_factory, ui_purge):
    tmp = tmp_path_factory.mktemp("homegame")
    keys = (*_ENV, "PLO5BP_DB", "PLO5BP_TRAINER_STATS")
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update(_ENV)
    os.environ["PLO5BP_DB"] = str(tmp / "public.db")
    os.environ["PLO5BP_TRAINER_STATS"] = str(tmp / "default_stats.json")
    # J3: pop the ui modules AND the stale `plo5bp.ui.<name>` package attrs.
    ui_purge()
    mod = importlib.import_module("plo5bp.ui.server")
    yield mod
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    ui_purge()


@pytest.fixture
def players(server):
    """Fresh signed-in alice, bob, admin — alice+bob granted games access."""
    a = TestClient(server.app)
    b = TestClient(server.app)
    adm = TestClient(server.app)
    assert a.get("/auth/dev", params={"email": "alice@example.com"}).status_code == 200
    assert b.get("/auth/dev", params={"email": "bob@example.com"}).status_code == 200
    assert adm.get("/auth/dev", params={"email": ADMIN_EMAIL}).status_code == 200
    users = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for email in ("alice@example.com", "bob@example.com"):
        r = adm.post(
            "/admin/api/games_access",
            json={"user_id": users[email], "action": "grant"},
        )
        assert r.status_code == 200
    return a, b, adm


def _create(client, **kwargs):
    body = {
        "name": "test table",
        "sb_cents": 50,
        "bb_cents": 100,
        "ante_cents": 300,
        "default_buyin_cents": 4000,
    }
    body.update(kwargs)
    r = client.post("/games/api/tables", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_create_sits_host_and_lists(players):
    a, b, _ = players
    t = _create(a)
    assert t["phase"] == "waiting"
    assert t["num_seats"] == 8
    assert t["is_host"] is True
    assert t["my_seat"] == 0
    assert t["seats"][0]["name"]
    assert t["seats"][0]["stack_cents"] == 4000
    assert t["running"] is False
    assert t["can_deal"] is False  # paused until host starts
    assert t["auto_stack"]["mode"] == "off"
    assert t["auto_stack"]["all_cents"] == 0
    assert t["seats"][0]["auto_stack_cents"] == 0
    # New tables ship with a shot clock (review 2026-09-20 G6 — this used to
    # pin the wedge-prone default of 0 = unlimited; 0 stays selectable).
    assert t["decision_secs"] == 30
    assert t["turn_remaining_secs"] is None  # no hand yet
    tables = a.get("/games/api/tables").json()["tables"]
    assert any(x["id"] == t["id"] for x in tables)
    # Bob can see the lobby too (granted).
    assert any(x["id"] == t["id"] for x in b.get("/games/api/tables").json()["tables"])


def test_sit_deal_check_down_ledger(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    r = b.post(
        f"/games/api/tables/{gid}/sit",
        json={"seat": 1, "buyin_cents": 4000},
    )
    assert r.status_code == 200
    s = r.json()
    assert s["seats"][1]["name"]
    assert s["running"] is False
    assert s["can_deal"] is False
    assert a.post(f"/games/api/tables/{gid}/deal").status_code == 400
    # Only the host can start; starting with 2 eligible deals immediately.
    assert b.post(f"/games/api/tables/{gid}/run", json={"running": True}).status_code == 400
    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    assert s["running"] is True
    assert s["phase"] in ("in_hand", "showdown")
    assert s["hand_no"] == 1
    assert s["street"] in ("flop", "turn", "river", "showdown")
    # Alice sees her hole cards, not Bob's (unless already showdown).
    alice_hole = s["seats"][0]["hole"]
    bob_from_alice = s["seats"][1]["hole"]
    assert alice_hole and alice_hole[0] >= 0
    assert alice_hole == sorted(alice_hole, reverse=True)
    desc = s["seats"][0]["hand_desc"]
    assert desc and len(desc) == 2
    assert any(desc)  # flop is always out, so at least one board labels
    if s["phase"] == "in_hand":
        assert bob_from_alice and bob_from_alice[0] < 0
        bob_view = b.get(f"/games/api/tables/{gid}").json()
        assert bob_view["seats"][1]["hole"][0] >= 0
        assert bob_view["seats"][0]["hole"][0] < 0

    for _ in range(40):
        sa = a.get(f"/games/api/tables/{gid}").json()
        if sa["phase"] != "in_hand":
            s = sa
            break
        acting = a if sa["my_seat"] == sa["actor"] else b
        st = acting.get(f"/games/api/tables/{gid}").json()
        assert st["legal"]["check_call"] or st["legal"]["fold"]
        gate = "check_call" if st["legal"]["check_call"] else "fold"
        r = acting.post(f"/games/api/tables/{gid}/act", json={"gate": gate})
        assert r.status_code == 200, r.text
    else:
        pytest.fail("hand did not terminate")

    s = a.get(f"/games/api/tables/{gid}").json()
    assert s["phase"] == "showdown"
    # Remaining players' holes are revealed.
    shown = [seat for seat in s["seats"] if seat["in_hand"] and not seat["folded"]]
    assert shown
    for seat in shown:
        assert seat["hole"] and seat["hole"][0] >= 0

    ledger = _ledger_by_name(s)
    assert ledger["alice"]["buyin_cents"] == 4000
    assert ledger["bob"]["buyin_cents"] == 4000
    # Zero-sum on the live stacks (antes stay in the pot then return as stacks).
    nets = [row["net_cents"] for row in s["ledger"] if row["seated"]]
    assert sum(nets) == 0


def test_rabbit_hunt_after_fold(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    b.post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000})
    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    if s["phase"] != "in_hand":
        pytest.skip("hand ended before action")
    acting = a if s["my_seat"] == s["actor"] else b
    other = b if acting is a else a
    st = acting.get(f"/games/api/tables/{gid}").json()
    if st["legal"]["raise"]:
        min_to = st["raise_bounds"]["min_chips"] + st["street_commit_chips"]
        r = acting.post(
            f"/games/api/tables/{gid}/act",
            json={"gate": "raise", "raise_to_chips": min_to},
        )
        assert r.status_code == 200, r.text
        st = other.get(f"/games/api/tables/{gid}").json()
        folder = other
    else:
        folder = acting
    assert st["legal"]["fold"]
    s = folder.post(f"/games/api/tables/{gid}/act", json={"gate": "fold"}).json()
    assert s["phase"] == "showdown"
    assert s["can_rabbit"] is True
    assert s["board"]["a"]["turn"] is None
    assert s["board"]["b"]["turn"] is None
    r = a.post(f"/games/api/tables/{gid}/rabbit")
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["can_rabbit"] is False
    assert s["rabbit_shown"] is True
    assert s["board"]["a"]["turn"] is not None
    assert s["board"]["a"]["river"] is not None
    assert s["board"]["b"]["turn"] is not None


def test_allin_runout_hides_future_streets_and_shows_equity(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    b.post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000})
    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    if s["phase"] != "in_hand":
        pytest.skip("hand ended before action")
    for _ in range(24):
        st = a.get(f"/games/api/tables/{gid}").json()
        if st["phase"] != "in_hand":
            s = st
            break
        acting = a if st["my_seat"] == st["actor"] else b
        cur = acting.get(f"/games/api/tables/{gid}").json()
        if cur["legal"]["raise"]:
            max_to = cur["raise_bounds"]["max_chips"] + cur["street_commit_chips"]
            r = acting.post(
                f"/games/api/tables/{gid}/act",
                json={"gate": "raise", "raise_to_chips": max_to},
            )
        elif cur["legal"]["check_call"]:
            r = acting.post(
                f"/games/api/tables/{gid}/act", json={"gate": "check_call"}
            )
        else:
            pytest.skip("could not jam")
        assert r.status_code == 200, r.text
        s = r.json()
    else:
        pytest.fail("hand did not terminate")
    assert s["phase"] == "showdown"
    assert s["runout"]["active"] is True
    assert s["runout"]["shown_len"] == s["runout"]["start_len"]
    if s["runout"]["shown_len"] < 5:
        assert s["board"]["a"]["river"] is None
        assert s["board"]["b"]["river"] is None
    assert s["can_rabbit"] is False
    shown = [seat for seat in s["seats"] if seat["in_hand"] and not seat["folded"]]
    assert len(shown) >= 2
    for seat in shown:
        assert seat["hole"] and seat["hole"][0] >= 0
        assert seat["hand_desc"] and any(seat["hand_desc"])
        assert seat["equity_a"] is not None
        assert 0 <= seat["equity_a"] <= 1
        assert 0 <= seat["equity_b"] <= 1
    eq_a = sum(seat["equity_a"] for seat in shown)
    assert abs(eq_a - 1) < 0.02
    # (review 2026-09-20 G11) This used to `assert s["pot_awards"]` — i.e. it
    # pinned the spoiler: the whole award script (winners + winning board
    # cards) shipped while the streets were still face-down. Award steps are
    # now released one at a time, only once all five cards are out.
    assert s["runout"]["award_count"] > 0
    if s["runout"]["award_index"] < 0:
        assert s["pot_awards"] == []
        assert s["runout"]["award_step"] is None
    assert s["can_deal"] is False


def test_leave_cashes_out(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    b.post(f"/games/api/tables/{gid}/sit", json={"seat": 2, "buyin_cents": 2000})
    s = b.post(f"/games/api/tables/{gid}/leave").json()
    bob = _ledger_by_name(s)["bob"]
    assert bob["seated"] is False
    assert bob["leftover_cents"] == 2000
    assert bob["net_cents"] == 0
    assert s["seats"][2]["empty"] is True


def test_chat_roundtrip(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    r = a.post(f"/games/api/tables/{gid}/chat", json={"text": "hello table"})
    assert r.status_code == 200, r.text
    msgs = r.json()["chat"]
    assert msgs and msgs[-1]["text"] == "hello table"
    b.post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000})
    got = b.get(f"/games/api/tables/{gid}").json()["chat"]
    assert any(m["text"] == "hello table" for m in got)


def test_games_assets_are_hidden(players):
    a, _, adm = players
    unsigned = TestClient(adm.app)
    for path in (
        "/static/games.js",
        "/static/games.css",
        "/static/games.html",
        "/games/static/games.js",
        "/games/static/games.css",
    ):
        assert unsigned.get(path).status_code == 404
    # Granted users load the assets from the gated /games/static/ route...
    for path in ("/games/static/games.js", "/games/static/games.css"):
        assert a.get(path).status_code == 200
    # ...and ONLY from there: since the 2026-09-20 review (F1) the public
    # /static mount denies the home-games files to everyone, on the resolved
    # file, so no path spelling reaches them. (This test used to expect 200
    # for a granted user on /static/games.*; nothing references those URLs.)
    for path in ("/static/games.js", "/static/games.css", "/static/games.html"):
        assert a.get(path).status_code == 404
    page = a.get("/games")
    assert page.status_code == 200
    # 2026-09-21: the client is split into modules, every one of them gated.
    modules = ("games.sound.js", "games.table.js", "games.ui.js", "games.play.js", "games.js")
    for name in modules:
        assert f"/games/static/{name}" in page.text
        assert a.get(f"/games/static/{name}").status_code == 200
        assert unsigned.get(f"/games/static/{name}").status_code == 404
        assert a.get(f"/static/{name}").status_code == 404
    assert page.text.index("games.ui.js") < page.text.index('/games/static/games.js"')  # core loads last
    for hook in ("chat-form", "hero-hand-labels", "award-caption", "actbar", "stage-box",
                 "drawer-root", "modal-root", "toast-root", 'method="post" action="/auth/logout"'):
        assert hook in page.text, hook
    js = {n: a.get(f"/games/static/{n}").text for n in modules}
    css = a.get("/games/static/games.css").text
    # the wiring each module owns (names the server API it drives)
    assert "Number.isInteger(s.my_seat)" in js["games.js"]
    for api in ("auto_stack", "auto_topup", "auto_chips_self", "street_pause", "settings",
                "transfer_host", "trust", '"request"', "kick", "sit_out_player", "host_fold",
                "close", "rebuy", "react", "/hands"):
        assert api in js["games.ui.js"], api
    for api in ("sit_out", "show", "rabbit", "run", "raise_to_chips"):
        assert api in js["games.play.js"], api
    assert "I'm back" in js["games.ui.js"] and "I'm back" in js["games.play.js"]
    assert "My stack each hand (host)" in js["games.ui.js"]
    assert "as-self-go" not in js["games.ui.js"]
    assert "EventSource" in js["games.js"] and "/stream" in js["games.js"]  # live push
    assert "startPoll" in js["games.js"]  # ... with the poll kept as the fallback
    for cls in ("actor-timer", "hg-card-bigrank"):
        assert cls in js["games.table.js"], cls
    assert "hero-hand-label" in js["games.play.js"]
    for cls in ("as-switch", "as-modes", "actor-timer", "actor-glow", "hhl-tag",
                "award-caption", "width: 100%", "prefers-reduced-motion"):
        assert cls in css, cls


def _ledger_by_name(state):
    """Ledger rows by display name (dev-login names = the email local part).
    Rows must not carry emails at all (review 2026-09-20 G12)."""
    assert all("email" not in row for row in state["ledger"])
    return {row["name"]: row for row in state["ledger"]}


ADMIN_NAME = ADMIN_EMAIL.split("@")[0]


def test_auto_stack_host_all_applies_before_ante(players):
    a, b, adm = players
    t = _create(a)
    gid = t["id"]
    alice_id = t["seats"][0]["user_id"]
    b.post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000})
    r = b.post(
        f"/games/api/tables/{gid}/auto_stack",
        json={"mode": "host", "all_cents": 10000},
    )
    assert r.status_code == 400
    r = a.post(
        f"/games/api/tables/{gid}/auto_stack",
        json={"mode": "host", "all_cents": 10000},
    )
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["auto_stack"]["mode"] == "host"
    assert s["auto_stack"]["all_cents"] == 10000
    assert s["seats"][0]["auto_stack_cents"] == 10000
    assert s["seats"][1]["auto_stack_cents"] == 10000
    # New sitters inherit the host's select-all amount.
    r = adm.post(
        f"/games/api/tables/{gid}/sit", json={"seat": 2, "buyin_cents": 4000}
    )
    assert r.status_code == 200, r.text
    assert r.json()["seats"][2]["auto_stack_cents"] == 10000
    # Host can still control one player independently.
    r = a.post(
        f"/games/api/tables/{gid}/auto_stack",
        json={"players": [{"user_id": alice_id, "cents": 5000}]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["seats"][0]["auto_stack_cents"] == 5000
    assert r.json()["seats"][1]["auto_stack_cents"] == 10000
    # Too small (ante is $3.00) is rejected.
    bad = a.post(
        f"/games/api/tables/{gid}/auto_stack",
        json={"players": [{"user_id": alice_id, "cents": 300}]},
    )
    assert bad.status_code == 400

    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    assert s["phase"] in ("in_hand", "showdown")
    if s["phase"] == "in_hand":
        # Start $50 / $100, then post $3 ante → $47 / $97 behind.
        assert s["seats"][0]["stack_cents"] == 4700
        assert s["seats"][1]["stack_cents"] == 9700
        assert s["seats"][2]["stack_cents"] == 9700
    led = _ledger_by_name(s)
    assert led["alice"]["buyin_cents"] == 5000
    assert led["bob"]["buyin_cents"] == 10000
    assert led[ADMIN_NAME]["buyin_cents"] == 10000
    assert led["alice"]["leftover_cents"] == 0
    assert led["bob"]["leftover_cents"] == 0


def test_auto_stack_surplus_cashout(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    b.post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000})
    assert b.post(
        f"/games/api/tables/{gid}/rebuy", json={"amount_cents": 4000}
    ).status_code == 200
    r = a.post(
        f"/games/api/tables/{gid}/auto_stack",
        json={"mode": "host", "all_cents": 4000},
    )
    assert r.status_code == 200, r.text
    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    led = _ledger_by_name(s)
    # Alice was already $40 — no ledger move. Bob $80 → $40, surplus cashed out.
    assert led["alice"]["buyin_cents"] == 4000
    assert led["alice"]["leftover_cents"] == 0
    assert led["bob"]["buyin_cents"] == 8000
    assert led["bob"]["leftover_cents"] == 4000
    nets = [row["net_cents"] for row in s["ledger"] if row["seated"]]
    # During the hand ledger stack is the pre-ante auto-stack amount, so
    # leftover + stack - buyin is still 0 before the pot settles.
    if s["phase"] == "in_hand":
        assert sum(nets) == 0
        assert s["seats"][1]["stack_cents"] == 3700


def test_auto_stack_mode_off_does_not_apply(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    b.post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000})
    a.post(
        f"/games/api/tables/{gid}/auto_stack",
        json={"mode": "host", "all_cents": 10000},
    )
    r = a.post(f"/games/api/tables/{gid}/auto_stack", json={"mode": "off"})
    assert r.status_code == 200, r.text
    assert r.json()["auto_stack"]["mode"] == "off"
    assert r.json()["seats"][0]["auto_stack_cents"] == 10000
    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    led = _ledger_by_name(s)
    assert led["alice"]["buyin_cents"] == 4000
    assert led["bob"]["buyin_cents"] == 4000
    if s["phase"] == "in_hand":
        assert s["seats"][0]["stack_cents"] == 3700
        assert s["seats"][1]["stack_cents"] == 3700


def test_auto_stack_player_toggle(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    b.post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000})
    assert b.post(
        f"/games/api/tables/{gid}/auto_stack_self", json={"cents": 8000}
    ).status_code == 400
    r = a.post(f"/games/api/tables/{gid}/auto_stack", json={"mode": "player"})
    assert r.status_code == 200, r.text
    assert r.json()["auto_stack"]["mode"] == "player"
    r = b.post(f"/games/api/tables/{gid}/auto_stack_self", json={"cents": 8000})
    assert r.status_code == 200, r.text
    assert r.json()["seats"][1]["auto_stack_cents"] == 8000
    assert r.json()["seats"][0]["auto_stack_cents"] == 0
    # Host is a player too — they can set their own amount in this mode.
    r = a.post(f"/games/api/tables/{gid}/auto_stack_self", json={"cents": 5000})
    assert r.status_code == 200, r.text
    assert r.json()["seats"][0]["auto_stack_cents"] == 5000
    r = a.post(f"/games/api/tables/{gid}/auto_stack_self", json={"cents": 0})
    assert r.status_code == 200, r.text
    assert r.json()["seats"][0]["auto_stack_cents"] == 0
    bad = b.post(f"/games/api/tables/{gid}/auto_stack_self", json={"cents": 100})
    assert bad.status_code == 400
    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    led = _ledger_by_name(s)
    assert led["alice"]["buyin_cents"] == 4000
    assert led["bob"]["buyin_cents"] == 8000
    if s["phase"] == "in_hand":
        assert s["seats"][0]["stack_cents"] == 3700
        assert s["seats"][1]["stack_cents"] == 7700


def test_auto_stack_skips_sitting_out(players):
    a, b, adm = players
    t = _create(a)
    gid = t["id"]
    b.post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000})
    adm.post(f"/games/api/tables/{gid}/sit", json={"seat": 2, "buyin_cents": 4000})
    a.post(
        f"/games/api/tables/{gid}/auto_stack",
        json={"mode": "host", "all_cents": 10000},
    )
    assert b.post(
        f"/games/api/tables/{gid}/sit_out", json={"on": True}
    ).status_code == 200
    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    led = _ledger_by_name(s)
    assert led["bob"]["buyin_cents"] == 4000
    assert led["alice"]["buyin_cents"] == 10000
    assert led[ADMIN_NAME]["buyin_cents"] == 10000
    if s["phase"] == "in_hand":
        assert s["seats"][1]["sitting_out"] is True
        assert s["seats"][1]["stack_cents"] == 4000


def _finish_runout(gid: str) -> None:
    """Fast-forward an all-in runout / showdown award animation."""
    import plo5bp.ui.homegame as hg

    t = hg.HUB.get(gid)
    with t.lock:
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0


def _expire_turn(gid: str) -> None:
    import plo5bp.ui.homegame as hg

    t = hg.HUB.get(gid)
    with t.lock:
        secs = int(t.decision_secs or 0) or 10
        t.decision_secs = secs
        t.turn_started_mono = time.monotonic() - float(secs) - 0.05


def test_decision_time_auto_check_and_fold(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    b.post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000})
    assert b.post(
        f"/games/api/tables/{gid}/decision_time", json={"secs": 15}
    ).status_code == 400
    bad = a.post(f"/games/api/tables/{gid}/decision_time", json={"secs": 3})
    assert bad.status_code == 400
    r = a.post(f"/games/api/tables/{gid}/decision_time", json={"secs": 15})
    assert r.status_code == 200, r.text
    assert r.json()["decision_secs"] == 15
    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    if s["phase"] != "in_hand":
        pytest.skip("hand ended before action")
    assert s["turn_remaining_secs"] is not None
    assert 0 < s["turn_remaining_secs"] <= 15
    _expire_turn(gid)
    s = a.get(f"/games/api/tables/{gid}").json()
    checks = [h for h in s["history"] if h["action"] == 1 and h["chips"] == 0]
    assert checks, s["history"]

    # New table: facing a bet, timeout folds.
    t2 = _create(a)
    gid = t2["id"]
    b.post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000})
    a.post(f"/games/api/tables/{gid}/decision_time", json={"secs": 15})
    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    if s["phase"] != "in_hand":
        pytest.skip("hand ended before action")
    acting = a if s["my_seat"] == s["actor"] else b
    other = b if acting is a else a
    st = acting.get(f"/games/api/tables/{gid}").json()
    if not st["legal"]["raise"]:
        pytest.skip("no raise available")
    min_to = st["raise_bounds"]["min_chips"] + st["street_commit_chips"]
    r = acting.post(
        f"/games/api/tables/{gid}/act",
        json={"gate": "raise", "raise_to_chips": min_to},
    )
    assert r.status_code == 200, r.text
    st = other.get(f"/games/api/tables/{gid}").json()
    if st["phase"] != "in_hand":
        pytest.skip("hand ended after raise")
    _expire_turn(gid)
    s = other.get(f"/games/api/tables/{gid}").json()
    assert any(h["action"] == 0 for h in s["history"]), s["history"]


def test_away_auto_acts_and_host_sit_out(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    sit = b.post(
        f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000}
    ).json()
    bob_id = sit["seats"][1]["user_id"]
    # Host can sit bob out between hands.
    r = a.post(
        f"/games/api/tables/{gid}/sit_out_player",
        json={"user_id": bob_id, "on": True},
    )
    assert r.status_code == 200, r.text
    assert r.json()["seats"][1]["sitting_out"] is True
    r = a.post(
        f"/games/api/tables/{gid}/sit_out_player",
        json={"user_id": bob_id, "on": False},
    )
    assert r.json()["seats"][1]["sitting_out"] is False
    assert b.post(
        f"/games/api/tables/{gid}/sit_out_player",
        json={"user_id": sit["seats"][0]["user_id"], "on": True},
    ).status_code == 400

    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    if s["phase"] != "in_hand":
        pytest.skip("hand ended before action")
    acting = a if s["my_seat"] == s["actor"] else b
    other = b if acting is a else a
    # Non-actor goes away; when the actor checks, away player auto-acts.
    r = other.post(f"/games/api/tables/{gid}/sit_out", json={"on": True})
    assert r.status_code == 200, r.text
    assert r.json()["seats"][other.get(f"/games/api/tables/{gid}").json()["my_seat"]]["sitting_out"] is True
    st = acting.get(f"/games/api/tables/{gid}").json()
    if not st["legal"]["check_call"]:
        pytest.skip("check not legal")
    s = acting.post(
        f"/games/api/tables/{gid}/act", json={"gate": "check_call"}
    ).json()
    # Away player should have been auto-checked or auto-folded in the same step.
    assert s["history"], s
    if s["phase"] == "in_hand":
        # Should not be waiting on the away player.
        away_seat = other.get(f"/games/api/tables/{gid}").json()["my_seat"]
        assert s["actor"] != away_seat


def test_away_as_actor_auto_acts(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    b.post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000})
    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    if s["phase"] != "in_hand":
        pytest.skip("hand ended before action")
    acting = a if s["my_seat"] == s["actor"] else b
    before = len(s["history"])
    r = acting.post(f"/games/api/tables/{gid}/sit_out", json={"on": True})
    assert r.status_code == 200, r.text
    s = r.json()
    assert len(s["history"]) > before or s["phase"] != "in_hand"


def test_host_kick_cashes_out(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    sit = b.post(
        f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 2000}
    ).json()
    bob_id = sit["seats"][1]["user_id"]
    alice_id = sit["seats"][0]["user_id"]
    assert b.post(
        f"/games/api/tables/{gid}/kick", json={"user_id": alice_id}
    ).status_code == 400
    assert a.post(
        f"/games/api/tables/{gid}/kick", json={"user_id": alice_id}
    ).status_code == 400
    r = a.post(f"/games/api/tables/{gid}/kick", json={"user_id": bob_id})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["seats"][1]["empty"] is True
    bob = _ledger_by_name(s)["bob"]
    assert bob["seated"] is False
    assert bob["leftover_cents"] == 2000
    assert bob["net_cents"] == 0


def test_host_kick_mid_hand_removes_after(players):
    a, b, _ = players
    t = _create(a)
    gid = t["id"]
    sit = b.post(
        f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000}
    ).json()
    bob_id = sit["seats"][1]["user_id"]
    s = a.post(f"/games/api/tables/{gid}/run", json={"running": True}).json()
    if s["phase"] != "in_hand":
        pytest.skip("hand ended before action")
    r = a.post(f"/games/api/tables/{gid}/kick", json={"user_id": bob_id})
    assert r.status_code == 200, r.text
    s = r.json()
    if s["phase"] == "in_hand":
        assert s["seats"][1]["sitting_out"] is True
        assert s["seats"][1]["pending_remove"] is True
        # Play it out; bob should be gone at showdown.
        for _ in range(40):
            st = a.get(f"/games/api/tables/{gid}").json()
            if st["phase"] != "in_hand":
                s = st
                break
            if st["seats"][1]["empty"]:
                s = st
                break
            acting = a if st["my_seat"] == st["actor"] else b
            cur = acting.get(f"/games/api/tables/{gid}").json()
            if cur["phase"] != "in_hand":
                s = cur
                break
            gate = "check_call" if cur["legal"]["check_call"] else "fold"
            acting.post(f"/games/api/tables/{gid}/act", json={"gate": gate})
        else:
            pytest.fail("hand did not terminate")
    # The removal is settled once the hand is REALLY over — after the
    # showdown's pot-award animation, not while it is still revealing
    # (review 2026-09-20 G11).
    _finish_runout(gid)
    s = a.get(f"/games/api/tables/{gid}").json()
    assert s["seats"][1]["empty"] is True
    bob = _ledger_by_name(s)["bob"]
    assert bob["seated"] is False


def test_subscriber_without_flag_still_404(server, players):
    _, _, adm = players
    c = TestClient(server.app)
    assert c.get("/auth/dev", params={"email": "subby@example.com"}).status_code == 200
    users = {u["email"]: u for u in adm.get("/admin/api/users").json()["users"]}
    uid = users["subby@example.com"]["id"]
    adm.post("/admin/api/grant", json={"user_id": uid, "action": "grant"})
    assert c.get("/state").status_code == 200
    assert c.get("/games").status_code == 404
    assert c.get("/static/games.js").status_code == 404
    assert "Home games" not in (c.get("/").text or "")
