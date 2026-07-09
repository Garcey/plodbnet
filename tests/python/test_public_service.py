"""Public-build service tests: auth gate, free tier, per-user isolation, admin.

Boots the app with PLO5BP_PUBLIC=1 + dev login (loopback fake sign-in) against
a temp sqlite DB, then exercises the whole lifecycle with three users (free A,
free B, admin). Env must be set before plo5bp.ui.server imports, so the
module-scoped fixture purges and reimports the ui modules around the tests.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"
FREE_HANDS = 3  # lower than prod's 5 to keep the test fast

_ENV = {
    "PLO5BP_PUBLIC": "1",
    "PLO5BP_DEV_LOGIN": "1",
    "PLO5BP_FREE_HANDS": str(FREE_HANDS),
    "PLO5BP_ADMIN_EMAILS": ADMIN_EMAIL,
}
_UI_MODULES = ("plo5bp.ui.server", "plo5bp.ui.public", "plo5bp.ui.trainer")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("public_svc")
    old = {k: os.environ.get(k) for k in (*_ENV, "PLO5BP_DB", "PLO5BP_TRAINER_STATS")}
    os.environ.update(_ENV)
    os.environ["PLO5BP_DB"] = str(tmp / "public.db")
    os.environ["PLO5BP_TRAINER_STATS"] = str(tmp / "default_stats.json")
    for m in _UI_MODULES:
        sys.modules.pop(m, None)
    mod = importlib.import_module("plo5bp.ui.server")
    yield mod
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    for m in _UI_MODULES:
        sys.modules.pop(m, None)


@pytest.fixture(scope="module")
def clients(server):
    """(user_a, user_b, admin) — separate cookie jars over one app."""
    a = TestClient(server.app)
    b = TestClient(server.app)
    adm = TestClient(server.app)
    assert a.get("/auth/dev", params={"email": "alice@example.com"}).status_code == 200
    assert b.get("/auth/dev", params={"email": "bob@example.com"}).status_code == 200
    assert adm.get("/auth/dev", params={"email": ADMIN_EMAIL}).status_code == 200
    return a, b, adm


def test_unauthenticated_is_gated(server):
    c = TestClient(server.app)
    assert c.get("/trainer/state").status_code == 401
    assert c.get("/state").status_code == 401
    me = c.get("/me").json()
    assert me["signed_in"] is False and me["dev_login"] is True


def test_live_routes_absent(server):
    c = TestClient(server.app)
    assert c.get("/ocr/status").status_code in (401, 404)
    # signed-out gets 401 from middleware; confirm the route truly 404s when authed
    c.get("/auth/dev", params={"email": "probe@example.com"})
    assert c.get("/ocr/status").status_code == 404
    assert c.get("/pokernow/status").status_code == 404
    # The NLH range grid is local-only until validated: stripped like /ocr.
    r = c.post(
        "/ranges/query", json={"seats": 6, "stack_bb": 100.0, "line": []}
    )
    assert r.status_code == 404


def test_me_shape(clients):
    a, _, adm = clients
    me = a.get("/me").json()
    assert me["signed_in"] and me["email"] == "alice@example.com"
    assert me["sub"]["active"] is False
    assert me["free"]["limit"] == FREE_HANDS
    admin_me = adm.get("/me").json()
    assert admin_me["is_admin"] is True and admin_me["sub"]["active"] is True


def test_free_limit_and_headers(clients):
    a, _, _ = clients
    # implicit first hand via GET /trainer/state is not quota-counted
    assert a.get("/trainer/state").status_code == 200
    for want_left in range(FREE_HANDS - 1, -1, -1):
        r = a.post("/trainer/new_hand")
        assert r.status_code == 200
        assert r.headers.get("X-Free-Hands-Left") == str(want_left)
    r = a.post("/trainer/new_hand")
    assert r.status_code == 402
    body = r.json()
    assert body["error"] == "free_limit" and body["limit"] == FREE_HANDS
    # repeat of the current hand stays free
    assert a.post("/trainer/repeat").status_code == 200


def test_study_requires_subscription(clients):
    a, _, _ = clients
    r = a.get("/state")
    assert r.status_code == 402
    assert r.json()["error"] == "subscription_required"


def test_per_user_trainer_isolation(clients):
    a, b, _ = clients
    hand_a = a.get("/trainer/state").json()["state"]["trainer"]["hand_no"]
    hand_b0 = b.get("/trainer/state").json()["state"]["trainer"]["hand_no"]
    assert hand_a > 1  # alice burned her quota dealing hands
    assert hand_b0 == 1  # bob's runtime is untouched by alice's dealing
    b.post("/trainer/new_hand")
    assert a.get("/trainer/state").json()["state"]["trainer"]["hand_no"] == hand_a


def test_admin_gate_and_dashboard(clients):
    a, _, adm = clients
    assert a.get("/admin/api/users").status_code == 403
    users = adm.get("/admin/api/users").json()["users"]
    emails = {u["email"] for u in users}
    assert {"alice@example.com", "bob@example.com", ADMIN_EMAIL} <= emails
    alice = next(u for u in users if u["email"] == "alice@example.com")
    assert alice["hands_today"] >= FREE_HANDS
    metrics = adm.get("/admin/api/metrics").json()
    assert metrics["users"] >= 3
    assert metrics["mrr_cents"] == 0  # no stripe subs in test


def test_comp_grant_unlocks_and_revoke_relocks(clients):
    a, _, adm = clients
    users = adm.get("/admin/api/users").json()["users"]
    uid = next(u["id"] for u in users if u["email"] == "alice@example.com")

    r = adm.post("/admin/api/grant", json={"user_id": uid, "action": "grant"})
    assert r.status_code == 200
    me = a.get("/me").json()
    assert me["sub"]["active"] is True and me["sub"]["source"] == "comp"
    # past the free limit but entitled now
    assert a.post("/trainer/new_hand").status_code == 200
    assert a.get("/state").status_code == 200  # study unlocked

    r = adm.post("/admin/api/grant", json={"user_id": uid, "action": "revoke"})
    assert r.status_code == 200
    assert a.get("/me").json()["sub"]["active"] is False
    assert a.get("/state").status_code == 402
    assert a.post("/trainer/new_hand").status_code == 402


def test_billing_unconfigured_is_graceful(clients):
    a, _, _ = clients
    r = a.post("/billing/checkout")
    assert r.status_code == 503
    assert "not configured" in r.json()["detail"].lower()


def test_admin_page_served(clients):
    _, _, adm = clients
    r = adm.get("/admin")
    assert r.status_code == 200 and "Admin" in r.text


def test_active_users_counter(server, clients):
    """/admin/api/active: admin-gated; counts non-admin users seen inside the
    window (every earlier test kept alice/bob warm); the admin's own traffic
    is excluded from the headline count; stale entries expire."""
    import time as _time

    a, _, adm = clients
    c = TestClient(server.app)
    assert c.get("/admin/api/active").status_code == 401  # signed out
    assert a.get("/admin/api/active").status_code == 403  # signed in, not admin

    d = adm.get("/admin/api/active").json()
    assert d["window_seconds"] > 0
    assert {"alice@example.com", "bob@example.com"} <= set(d["emails"])
    assert ADMIN_EMAIL not in d["emails"]
    assert d["active_users"] == len(d["emails"])
    assert d["active_total"] >= d["active_users"] + 1  # admin in the total

    # Age alice out of the window: she drops from the count and is pruned.
    pub = sys.modules["plo5bp.ui.public"]
    users = adm.get("/admin/api/users").json()["users"]
    alice_id = next(u["id"] for u in users if u["email"] == "alice@example.com")
    pub._ACTIVITY[alice_id] = _time.time() - pub.ACTIVE_WINDOW_S - 1
    d = adm.get("/admin/api/active").json()
    assert "alice@example.com" not in d["emails"]
    assert alice_id not in pub._ACTIVITY  # pruned, not just filtered

    # One authenticated request to a gated route brings her back. (/me is an
    # OPEN route — signed-out landing needs it — so it deliberately does NOT
    # count as activity.)
    assert a.get("/trainer/state").status_code == 200
    d = adm.get("/admin/api/active").json()
    assert "alice@example.com" in d["emails"]


def test_session_instances_do_not_share_mutable_state(server):
    """Regression: each Session() must own its mutable state. In the public
    multi-user build every signed-in user gets their own Session(), so a
    shared class-level list/dict default would alias one user's action_log /
    cards / slot-locks into another's session (a cross-user leak)."""
    Session = server.Session
    a, b = Session(), Session()
    # Distinct container identities.
    assert a.action_log is not b.action_log
    assert a.hero_hole is not b.hero_hole
    assert a.game_config is not b.game_config
    assert a._card_slot_locked is not b._card_slot_locked
    assert a._card_slot_locked["hero_hole"] is not b._card_slot_locked["hero_hole"]
    assert a._card_slot_pending is not b._card_slot_pending
    # In-place mutation on one must not bleed into the other.
    a.action_log.append({"gate": 1, "chips": 999})
    a.hero_hole[0] = 42
    a._card_slot_locked["hero_hole"][0] = True
    assert b.action_log == []
    assert b.hero_hole[0] is None
    assert b._card_slot_locked["hero_hole"][0] is False


def test_webhook_requires_settled_payment(server, clients, monkeypatch):
    """Regression: the Stripe webhook must NOT grant a subscription for a
    checkout event whose payment hasn't settled (payment_status='unpaid', as
    async ACH / bank-transfer methods fire) — only once it is paid."""
    pub = sys.modules["plo5bp.ui.public"]
    _, _, adm = clients

    c = TestClient(server.app)
    assert c.get("/auth/dev", params={"email": "wh@example.com"}).status_code == 200
    uid = next(
        u["id"]
        for u in adm.get("/admin/api/users").json()["users"]
        if u["email"] == "wh@example.com"
    )
    assert c.get("/me").json()["sub"]["active"] is False

    # Fake Stripe: bypass signature verification and inject a crafted event.
    monkeypatch.setattr(pub, "STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setattr(pub, "STRIPE_WEBHOOK_SECRET", "whsec_x")
    injected: dict = {}

    class _FakeWebhook:
        @staticmethod
        def construct_event(payload, sig, secret):
            return injected

    class _FakeSub:
        @staticmethod
        def retrieve(sid):  # only hit if a stale refresh fires; keep it sane
            return {"status": "active", "current_period_end": 9999999999}

    class _FakeStripe:
        Webhook = _FakeWebhook
        Subscription = _FakeSub

    monkeypatch.setattr(pub, "_stripe", lambda: _FakeStripe)

    def fire(payment_status):
        injected.clear()
        injected.update(
            {
                "type": "checkout.session.completed",
                "data": {
                    "object": {
                        "client_reference_id": str(uid),
                        "payment_status": payment_status,
                        "subscription": "sub_test",
                        "customer": "cus_test",
                        "id": "cs_test",
                        "amount_total": 1000,
                        "currency": "usd",
                    }
                },
            }
        )
        return c.post(
            "/stripe/webhook",
            content=b"{}",
            headers={"stripe-signature": "t=1,v1=fake"},
        )

    # Unpaid async checkout "completed" → must NOT activate.
    assert fire("unpaid").status_code == 200
    assert c.get("/me").json()["sub"]["active"] is False

    # Settled payment → activates via stripe.
    assert fire("paid").status_code == 200
    me = c.get("/me").json()
    assert me["sub"]["active"] is True
    assert me["sub"]["source"] == "stripe"
