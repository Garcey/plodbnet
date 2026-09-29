"""Accounts, sessions and the admin surface of the public build (2026-09-28):

- disable / "sign out everywhere" (SEC-016), Google-id-first account lookup
  and admin pinning (SEC-019), race-free first sign-in (BE-020), session-key
  rotation out of the database (SEC-020), the admin audit log (SEC-022);
- email sign-in links (ACC-008);
- export my data / delete my account, home games anonymized consistently
  (ACC-009 / SEC-017);
- the maintenance notice (FEAT-028), the System panel (FEAT-024), and live
  model reload / promote / rollback without a restart (OPS-027 / OPS-022);
- billing guards (BE-023 / BE-024 / SEC-031).
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timedelta, timezone

import pytest
import torch
from starlette.testclient import TestClient

ADMIN = "acct-admin@example.com"


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_ADMIN_EMAILS=ADMIN, PLO5BP_EMAIL_LOGIN="log")


@pytest.fixture(scope="module")
def pub(server):
    return sys.modules["plo5bp.ui.public"]


def _client(server, email=None):
    c = TestClient(server.app, raise_server_exceptions=False)
    if email:
        assert c.get("/auth/dev", params={"email": email}).status_code == 200
    return c


def _uid(pub, email):
    return int(pub.DB.one("SELECT id FROM users WHERE email=?", (email,))["id"])


@pytest.fixture(scope="module")
def admin(server):
    return _client(server, ADMIN)


# --- SEC-016 -------------------------------------------------------------------------


def test_disabling_an_account_signs_it_out_and_keeps_it_out(server, pub, admin):
    u = _client(server, "dis@example.com")
    assert u.get("/trainer/state").status_code == 200
    uid = _uid(pub, "dis@example.com")
    r = admin.post("/admin/api/users/action", json={"user_id": uid, "action": "disable"})
    assert r.status_code == 200 and r.json()["disabled"] is True
    r = u.get("/trainer/state")
    assert r.status_code == 403 and r.json()["code"] == "account_disabled"
    assert u.get("/me").json()["signed_in"] is False  # the cookie was dropped
    # A disabled account cannot sign back in either.
    assert _client(server).get("/auth/dev", params={"email": "dis@example.com"}).status_code == 403
    admin.post("/admin/api/users/action", json={"user_id": uid, "action": "enable"})
    again = _client(server, "dis@example.com")
    assert again.get("/trainer/state").status_code == 200


def test_sign_out_everywhere_ends_every_other_session(server, pub, admin):
    a1 = _client(server, "multi@example.com")
    a2 = _client(server, "multi@example.com")
    assert a2.get("/trainer/state").status_code == 200
    assert a1.post("/account/signout_everywhere").status_code == 200
    assert a1.get("/trainer/state").status_code == 200   # this browser stays in
    assert a2.get("/trainer/state").status_code == 401   # the other one is out
    # The admin can do it for a user (a leaked cookie).
    a3 = _client(server, "multi@example.com")
    uid = _uid(pub, "multi@example.com")
    admin.post("/admin/api/users/action", json={"user_id": uid, "action": "signout"})
    assert a3.get("/trainer/state").status_code == 401


def test_admins_are_not_disabled_from_the_panel(pub, admin):
    uid = _uid(pub, ADMIN)
    r = admin.post("/admin/api/users/action", json={"user_id": uid, "action": "disable"})
    assert r.status_code == 409


# --- SEC-019 / BE-020 --------------------------------------------------------------------


def test_accounts_are_found_by_google_id_first(pub):
    a = pub._upsert_user("sub-alpha", "alpha@example.com", "Alpha", "")
    # The Google account's email changed: same account, new address.
    b = pub._upsert_user("sub-alpha", "alpha2@example.com", "Alpha", "")
    assert b["id"] == a["id"] and b["email"] == "alpha2@example.com"
    # Another Google identity asserting that address is refused.
    with pytest.raises(pub.SignInConflict):
        pub._upsert_user("sub-other", "alpha2@example.com", "Mallory", "")
    # Email-only sign-ins (dev / email link) prove the inbox: same account.
    assert pub._upsert_user(None, "alpha2@example.com", "", "")["id"] == a["id"]


def test_admin_rights_can_be_pinned_to_google_ids(pub, monkeypatch):
    row = pub._upsert_user("sub-admin", ADMIN, "Admin", "")
    assert pub._is_admin(row)
    monkeypatch.setattr(pub, "ADMIN_SUBS", {"sub-somebody-else"})
    assert not pub._is_admin(row)
    monkeypatch.setattr(pub, "ADMIN_SUBS", {"sub-admin"})
    assert pub._is_admin(row)


def test_simultaneous_first_sign_ins_never_fail(pub):
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def go():
        try:
            barrier.wait(5)
            pub._upsert_user(None, "race@example.com", "Race", "")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors
    assert pub.DB.one("SELECT COUNT(*) c FROM users WHERE email='race@example.com'")["c"] == 1


# --- SEC-020 ---------------------------------------------------------------------------------


def test_session_key_moves_out_of_the_database(pub, monkeypatch):
    legacy = pub._session_secret()
    monkeypatch.delenv("PLO5BP_SESSION_SECRET", raising=False)
    assert pub._session_keys() == [legacy]
    monkeypatch.setenv("PLO5BP_SESSION_SECRET", "new-key, older-key")
    pub.DB.kv_delete("session_secret_env_since")
    # First cookie lifetime after the switch: the old key still verifies.
    assert pub._session_keys() == ["new-key", "older-key", legacy]
    old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat(timespec="seconds")
    pub.DB.kv_set("session_secret_env_since", old)
    assert pub._session_keys() == ["new-key", "older-key"]
    assert pub.DB.kv_get("session_secret") is None  # retired for good
    pub.DB.kv_set("session_secret", legacy)  # (restore for the rest of the module)


# --- SEC-022 -----------------------------------------------------------------------------------


def test_admin_actions_are_audited(server, pub, admin):
    _client(server, "audited@example.com")
    uid = _uid(pub, "audited@example.com")
    assert admin.post("/admin/api/grant", json={"user_id": uid, "action": "grant"}).status_code == 200
    assert admin.post("/admin/api/grant", json={"user_id": uid, "action": "revoke"}).status_code == 200
    entries = admin.get("/admin/api/audit").json()["entries"]
    top = [(e["action"], e["admin"], e["target"]) for e in entries[:2]]
    assert top == [("comp_revoke", ADMIN, "audited@example.com"),
                   ("comp_grant", ADMIN, "audited@example.com")]


# --- ACC-008 ------------------------------------------------------------------------------------


def test_email_sign_in_link(server, pub, monkeypatch):
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(pub, "_send_login_email", lambda email, link: sent.append((email, link)))
    c = _client(server)
    assert c.get("/me").json()["email_login"] is True
    form = c.get("/auth/email")
    assert form.status_code == 200 and 'action="/auth/email"' in form.text
    r = c.post("/auth/email", data={"email": "Link@Example.com"})
    assert r.status_code == 200 and "Check your email" in r.text
    assert sent and sent[-1][0] == "link@example.com"
    token = sent[-1][1].split("token=", 1)[1]
    # The link lands on a confirmation button (mail scanners open links).
    page = c.get(f"/auth/email/verify?token={token}")
    assert page.status_code == 200 and 'method="post"' in page.text
    assert c.get("/me").json()["signed_in"] is False
    done = c.post("/auth/email/verify", data={"token": token}, follow_redirects=False)
    assert done.status_code == 303
    me = c.get("/me").json()
    assert me["signed_in"] is True and me["email"] == "link@example.com"
    # Single use.
    again = _client(server).post("/auth/email/verify", data={"token": token})
    assert again.status_code == 400 and "expired" in again.text
    assert c.post("/auth/email", data={"email": "not-an-email"}).status_code == 400


def test_email_links_expire(server, pub, monkeypatch):
    sent: list[str] = []
    monkeypatch.setattr(pub, "_send_login_email", lambda email, link: sent.append(link))
    c = _client(server)
    c.post("/auth/email", data={"email": "late@example.com"})
    token = sent[-1].split("token=", 1)[1]
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(timespec="seconds")
    pub.DB.q("UPDATE email_tokens SET expires_at=? WHERE email='late@example.com'", (past,))
    assert c.post("/auth/email/verify", data={"token": token}).status_code == 400
    assert c.get("/me").json()["signed_in"] is False


# --- ACC-009 / SEC-017 ---------------------------------------------------------------------------


def _seed_home_game_rows(pub, uid, other):
    """A finished home-game hand between `uid` and `other`, a chat line, a
    picture and a club membership — written straight to the tables."""
    now = pub._now()
    q = pub.DB.q
    q("INSERT INTO homegame_clubs(id,name,owner_user_id,invite_code,created_at) "
      "VALUES('clubX','Club X',?, 'invX1234', ?)", (other, now))
    q("INSERT INTO homegame_club_members(club_id,user_id,role,joined_at) VALUES('clubX',?,'member',?)",
      (uid, now))
    q("INSERT INTO homegames(id,host_user_id,name,num_seats,sb_cents,bb_cents,ante_cents,"
      "default_buyin_cents,status,created_at,club_id) VALUES('gX',?,'Friday',6,50,100,300,10000,"
      "'closed',?,'clubX')", (other, now))
    summary = {"hand_no": 1, "seats": [
        {"seat": 0, "user_id": uid, "name": "Victim", "hole": [1, 2, 3, 4, 5], "delta_cents": -500},
        {"seat": 1, "user_id": other, "name": "Friend", "hole": [6, 7, 8, 9, 10], "delta_cents": 500},
    ], "winners": [["Friend", 500]]}
    q("INSERT INTO homegame_hands(game_id,hand_no,ended_at,pot_cents,summary) VALUES('gX',1,?,1000,?)",
      (now, json.dumps(summary)))
    q("INSERT INTO homegame_hand_results(game_id,hand_no,user_id,delta_cents) VALUES('gX',1,?,-500)", (uid,))
    q("INSERT INTO homegame_hand_results(game_id,hand_no,user_id,delta_cents) VALUES('gX',1,?,500)", (other,))
    q("INSERT INTO homegame_ledger(game_id,user_id,kind,amount_cents,created_at) VALUES('gX',?,'buyin',-10000,?)",
      (uid, now))
    q("INSERT INTO homegame_ledger(game_id,user_id,kind,amount_cents,created_at) VALUES('gX',?,'cashout',9500,?)",
      (uid, now))
    q("INSERT INTO homegame_chat(game_id,user_id,body,created_at) VALUES('gX',?,'gg',?)", (uid, now))
    q("INSERT INTO homegame_avatars(user_id,mime,data,version,updated_at) VALUES(?,'image/png',X'00','v1',?)",
      (uid, now))


def test_export_then_delete_my_account(server, pub, admin):
    u = _client(server, "victim@example.com")
    friend = _client(server, "friend@example.com")
    uid, other = _uid(pub, "victim@example.com"), _uid(pub, "friend@example.com")
    assert u.get("/trainer/state").status_code == 200
    _seed_home_game_rows(pub, uid, other)

    r = u.get("/account/export")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    data = r.json()
    assert data["account"]["email"] == "victim@example.com"
    hg = data["home_games"]
    assert hg["hands"][0]["your_cards"] == [1, 2, 3, 4, 5]
    assert [c["body"] for c in hg["chat"]] == ["gg"] and hg["has_picture"] is True
    assert "hole" not in json.dumps(hg["hands"][0]).replace("your_cards", "")  # only own cards

    assert u.post("/account/delete", json={"confirm": "wrong@example.com"}).status_code == 400
    assert u.post("/account/delete", json={"confirm": "victim@example.com"}).json() == {"deleted": True}
    assert u.get("/me").json()["signed_in"] is False
    row = pub._user_by_id(uid)
    assert row["email"] == f"deleted-{uid}@deleted.invalid" and row["name"] == "Deleted player"
    assert row["deleted_at"] and row["google_sub"] is None
    # Other players' history stays whole: their hand record, the zero-sum
    # results and the ledger rows keep the id; only the name is gone.
    rec = json.loads(pub.DB.one("SELECT summary FROM homegame_hands WHERE game_id='gX'")["summary"])
    assert [s["name"] for s in rec["seats"]] == ["Deleted player", "Friend"]
    net = pub.DB.one("SELECT SUM(delta_cents) s FROM homegame_hand_results WHERE game_id='gX'")["s"]
    assert net == 0
    assert pub.DB.one("SELECT COUNT(*) c FROM homegame_ledger WHERE user_id=?", (uid,))["c"] == 2
    # …while what was only theirs is gone.
    for table in ("homegame_chat", "homegame_avatars", "homegame_club_members"):
        assert pub.DB.one(f"SELECT COUNT(*) c FROM {table} WHERE user_id=?", (uid,))["c"] == 0
    emails = {x["email"] for x in admin.get("/admin/api/users").json()["users"]}
    assert "victim@example.com" not in emails and f"deleted-{uid}@deleted.invalid" not in emails
    # The address is free again: signing in makes a NEW account.
    assert _uid(pub, "friend@example.com") == other
    _client(server, "victim@example.com")
    assert _uid(pub, "victim@example.com") != uid


def test_delete_is_refused_while_seated_at_a_live_table(server, pub):
    u = _client(server, "seated@example.com")
    uid = _uid(pub, "seated@example.com")
    now = pub._now()
    pub.DB.q("INSERT INTO homegames(id,host_user_id,name,num_seats,sb_cents,bb_cents,ante_cents,"
             "default_buyin_cents,status,created_at) VALUES('gLive',0,'Live one',6,50,100,300,10000,'open',?)",
             (now,))
    pub.DB.q("INSERT INTO homegame_players(game_id,user_id,seat,stack_chips) VALUES('gLive',?,2,100000)", (uid,))
    r = u.post("/account/delete", json={"confirm": "seated@example.com"})
    assert r.status_code == 409 and "Live one" in r.json()["detail"]
    assert pub._user_by_id(uid)["deleted_at"] is None


# --- FEAT-028 / FEAT-024 --------------------------------------------------------------------------


def test_maintenance_notice_reaches_everyone(server, admin):
    anon, user = _client(server), _client(server, "notice@example.com")
    r = admin.post("/admin/api/maintenance", json={"message": "Restart at 10 pm", "in_minutes": 15})
    assert r.status_code == 200
    for c in (anon, user):
        m = c.get("/me").json()["maintenance"]
        assert m["message"] == "Restart at 10 pm" and m["at"]
    admin.post("/admin/api/maintenance", json={"clear": True})
    assert "maintenance" not in user.get("/me").json()


def test_system_panel(admin):
    s = admin.get("/admin/api/system").json()
    assert {"health", "formats", "http", "db", "config", "runtimes", "work_gate"} <= set(s)
    assert s["db"]["schema"]["public"] >= 7
    ids = {f["format"] for f in s["formats"]}
    assert "plo5_double_bomb" in ids
    assert s["health"]["http_status"] in (200, 503)


def test_the_admin_user_list_runs_a_fixed_number_of_queries(server, pub, admin, monkeypatch):
    """PERF-010: the list used to cost two home-games queries per user (plus two
    correlated usage lookups per row); now a handful in all, whatever its length."""
    calls: list[str] = []
    real_q, real_one = pub.DB.q, pub.DB.one

    def count(fn):
        def inner(sql, *a, **kw):
            calls.append(sql)
            return fn(sql, *a, **kw)
        return inner

    def listing() -> list[dict]:
        calls.clear()
        monkeypatch.setattr(pub.DB, "q", count(real_q))
        monkeypatch.setattr(pub.DB, "one", count(real_one))
        try:
            r = admin.get("/admin/api/users")
        finally:
            monkeypatch.setattr(pub.DB, "q", real_q)
            monkeypatch.setattr(pub.DB, "one", real_one)
        assert r.status_code == 200
        return r.json()["users"]

    member = _client(server, "club-member@example.com")
    assert member.post("/trainer/new_hand").status_code == 200  # a dealt hand is usage
    assert admin.post("/admin/api/games_access",
                      json={"user_id": _uid(pub, "club-member@example.com"), "action": "grant"}
                      ).status_code == 200
    few = listing()
    queries = len(calls)
    for k in range(6):
        _client(server, f"many-{k}@example.com")
    many = listing()
    assert len(many) == len(few) + 6
    assert len(calls) == queries, f"{queries} queries for {len(few)} users, {len(calls)} for {len(many)}"
    rows = {u["email"]: u for u in many}
    assert rows["club-member@example.com"]["homegame_access"] is True
    assert rows["many-0@example.com"]["homegame_access"] is False
    assert rows["club-member@example.com"]["hands_today"] == rows["club-member@example.com"]["hands_total"] >= 1


# --- OPS-027 / OPS-022 ------------------------------------------------------------------------------


def _net_ckpt(path, seed):
    from plo5bp.network import ActorCriticV2

    torch.manual_seed(seed)
    net = ActorCriticV2(hidden_dim=8, num_layers=1)
    torch.save({"model": net.state_dict(), "head_version": 2, "obs_rev": 2,
                "config": {"hidden_dim": 8, "num_layers": 1}}, path)
    return path


def test_models_reload_promote_and_roll_back_without_a_restart(server, pub, admin, tmp_path, monkeypatch):
    from plo5bp.ui import models

    stub = tmp_path / "stub.pt"
    monkeypatch.setenv("PLO5BP_CHECKPOINT", str(stub))
    saved = server.FORMATS["plo5_double_bomb"]
    player = _client(server, "reload-player@example.com")
    assert player.get("/trainer/state").status_code == 200
    try:
        _net_ckpt(stub, 1)
        a = admin.post("/admin/api/models", json={"action": "reload", "format": "plo5_double_bomb"})
        assert a.status_code == 200, a.text
        a = a.json()
        assert a["model_loaded"] is True and a["checkpoint"] == "stub.pt"
        assert a["sha256"] == models.file_facts(stub)["sha256"]
        assert server.MODEL is server.FORMATS["plo5_double_bomb"]["model"]
        # A trainer session picks the new model up at its NEXT hand.
        assert player.post("/trainer/new_hand").status_code == 200
        ts = pub._REGISTRY.peek(_uid(pub, "reload-player@example.com")).trainer
        assert ts.model is server.FORMATS["plo5_double_bomb"]["model"]

        # Promote a verified .new; the old file is kept as .prev.
        sha_a = a["sha256"]
        _net_ckpt(tmp_path / "stub.pt.new", 2)
        b = admin.post("/admin/api/models", json={"action": "promote", "format": "plo5_double_bomb"}).json()
        assert b["sha256"] != sha_a and not (tmp_path / "stub.pt.new").exists()
        assert models.file_facts(tmp_path / "stub.pt.prev")["sha256"] == sha_a
        assert b["version"] > a["version"]
        # A broken candidate never replaces the served model.
        (tmp_path / "stub.pt.new").write_bytes(b"not a checkpoint")
        bad = admin.post("/admin/api/models", json={"action": "promote", "format": "plo5_double_bomb"})
        assert bad.status_code == 409
        assert server.FORMATS["plo5_double_bomb"]["sha256"] == b["sha256"]
        # Roll back (and forth).
        c = admin.post("/admin/api/models", json={"action": "rollback", "format": "plo5_double_bomb"}).json()
        assert c["sha256"] == sha_a
        actions = [e["action"] for e in admin.get("/admin/api/audit").json()["entries"][:4]]
        assert actions[0] == "model_rollback" and "model_promote_failed" in actions
    finally:
        server.FORMATS["plo5_double_bomb"] = saved
        server._on_model_swap("plo5_double_bomb", saved)


# --- billing guards (BE-023 / BE-024 / SEC-031) --------------------------------------------------------


class _NoStripe:
    """Any Stripe call is a test failure."""

    def __getattr__(self, name):
        raise AssertionError(f"Stripe touched: {name}")


def test_checkout_refuses_a_second_subscription(server, pub, admin, monkeypatch):
    monkeypatch.setattr(pub, "FREE_FOR_ALL", False)
    monkeypatch.setattr(pub, "STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setattr(pub, "_stripe", lambda: _NoStripe())
    u = _client(server, "subbed@example.com")
    uid = _uid(pub, "subbed@example.com")
    admin.post("/admin/api/grant", json={"user_id": uid, "action": "grant"})
    r = u.post("/billing/checkout")
    assert r.status_code == 409
    # A comp never overwrites a live Stripe subscription.
    pub.DB.q("UPDATE users SET sub_status='active', sub_source='stripe', "
             "stripe_subscription_id='sub_1' WHERE id=?", (uid,))
    assert admin.post("/admin/api/grant", json={"user_id": uid, "action": "grant"}).status_code == 409


def test_live_keys_need_a_configured_price(server, pub, monkeypatch):
    monkeypatch.setattr(pub, "FREE_FOR_ALL", False)
    monkeypatch.setattr(pub, "STRIPE_SECRET_KEY", "sk_live_x")
    monkeypatch.setattr(pub, "STRIPE_PRICE_ID", "")
    monkeypatch.setattr(pub, "_stripe", lambda: _NoStripe())
    u = _client(server, "live-price@example.com")
    r = u.post("/billing/checkout")
    assert r.status_code == 503 and "STRIPE_PRICE_ID" in r.json()["detail"]


def test_a_live_key_never_runs_with_the_dev_login(pub, monkeypatch):
    monkeypatch.setattr(pub, "STRIPE_SECRET_KEY", "rk_live_abc")
    with pytest.raises(RuntimeError, match="LIVE"):
        pub._check_stripe_key_is_safe()
    monkeypatch.setattr(pub, "STRIPE_SECRET_KEY", "sk_test_abc")
    pub._check_stripe_key_is_safe()
