"""Regression tests for the 2026-09-20 review — Stripe billing (F2, F3).

Never talks to Stripe: `public._stripe` is replaced by a fake module object
(the same technique as test_public_service.py's webhook test).
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server()


@pytest.fixture(scope="module")
def pub(server):
    return sys.modules["plo5bp.ui.public"]


class FakeStripeError(Exception):
    """Shaped like stripe.InvalidRequestError (`code`, `http_status`)."""

    def __init__(self, message, code=None, http_status=None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status


class FakeStripe:
    """Just enough of the stripe module: checkout.Session.retrieve,
    Subscription.retrieve, Webhook.construct_event."""

    def __init__(self):
        self.sessions: dict[str, dict] = {}
        self.subs: dict[str, dict] = {}
        self.sub_error: Exception | None = None
        self.sub_calls = 0
        self.on_event_loop = 0  # retrievals made from the event-loop thread
        self.event: dict = {}
        outer = self

        class _Session:
            @staticmethod
            def retrieve(sid):
                outer._note_thread()
                if sid not in outer.sessions:
                    raise FakeStripeError(
                        f"No such checkout.session: '{sid}'", "resource_missing", 404
                    )
                return outer.sessions[sid]

        class _Checkout:
            Session = _Session

        class _Subscription:
            @staticmethod
            def retrieve(sub_id):
                outer.sub_calls += 1
                outer._note_thread()
                if outer.sub_error is not None:
                    raise outer.sub_error
                if sub_id not in outer.subs:
                    raise FakeStripeError(
                        f"No such subscription: '{sub_id}'", "resource_missing", 404
                    )
                return outer.subs[sub_id]

        class _Webhook:
            @staticmethod
            def construct_event(payload, sig, secret):
                return outer.event

        self.checkout = _Checkout
        self.Subscription = _Subscription
        self.Webhook = _Webhook

    def _note_thread(self):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self.on_event_loop += 1


def _ts(days: float) -> int:
    return int(time.time() + days * 86400)


def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat(timespec="seconds")


@pytest.fixture()
def stripe(pub, monkeypatch):
    fake = FakeStripe()
    monkeypatch.setattr(pub, "STRIPE_SECRET_KEY", "sk_test_fake")
    monkeypatch.setattr(pub, "STRIPE_WEBHOOK_SECRET", "whsec_fake")
    monkeypatch.setattr(pub, "_stripe", lambda: fake)
    pub._STRIPE_RETRY_AT.clear()
    yield fake
    pub._STRIPE_RETRY_AT.clear()


_N = [0]


def _user(server, pub, prefix: str):
    _N[0] += 1
    email = f"{prefix}{_N[0]}@example.com"
    c = TestClient(server.app, raise_server_exceptions=False)
    assert c.get("/auth/dev", params={"email": email}).status_code == 200
    uid = int(pub.DB.one("SELECT id FROM users WHERE email=?", (email,))["id"])
    return c, uid


def _make_stripe_sub(pub, uid, *, sub_id, period_end: timedelta | None, checked: timedelta):
    pub.DB.q(
        "UPDATE users SET sub_status='active', sub_source='stripe',"
        " stripe_customer_id='cus_x', stripe_subscription_id=?,"
        " current_period_end=?, sub_checked_at=? WHERE id=?",
        (sub_id, _iso(period_end) if period_end is not None else None, _iso(checked), uid),
    )


def _row(pub, uid):
    return dict(pub._user_by_id(uid))


# --- F2: re-validation ----------------------------------------------------------


def test_revalidation_never_runs_on_the_event_loop(server, pub, stripe):
    c, uid = _user(server, pub, "f2loop")
    stripe.subs["sub_loop"] = {
        "status": "active",
        "items": {"data": [{"current_period_end": _ts(20)}]},
    }
    _make_stripe_sub(pub, uid, sub_id="sub_loop", period_end=timedelta(days=-1),
                     checked=timedelta(hours=-30))
    # /state goes through the ASYNC middleware; /me is a sync endpoint.
    assert c.get("/state").status_code == 200
    assert stripe.sub_calls == 1
    assert stripe.on_event_loop == 0
    row = _row(pub, uid)
    assert row["sub_status"] == "active"
    assert pub._parse_iso(row["current_period_end"]) > datetime.now(timezone.utc)
    # Fresh now: further requests do not call Stripe at all.
    for _ in range(4):
        assert c.get("/state").status_code == 200
    assert stripe.sub_calls == 1


def test_no_such_subscription_means_inactive(server, pub, stripe):
    """A deleted sub / test->live key switch is an ANSWER, not an outage."""
    c, uid = _user(server, pub, "f2gone")
    _make_stripe_sub(pub, uid, sub_id="sub_gone", period_end=timedelta(days=-40),
                     checked=timedelta(days=-40))
    codes = [c.get("/state").status_code for _ in range(5)]
    assert codes == [402] * 5
    assert stripe.sub_calls == 1  # asked once; the row is 'none' afterwards
    assert _row(pub, uid)["sub_status"] == "none"
    assert c.get("/me").json()["sub"]["active"] is False


@pytest.mark.parametrize("exc", [
    FakeStripeError("No such subscription: 'sub_1'", code="resource_missing", http_status=404),
    RuntimeError("No such subscription: 'sub_1'"),
])
def test_stripe_missing_classifier(pub, exc):
    assert pub._stripe_missing(exc) is True


def test_stripe_missing_classifier_real_error_class(pub):
    stripe_mod = pytest.importorskip("stripe")
    e = stripe_mod.InvalidRequestError(
        "No such subscription: 'sub_1'", param="id", code="resource_missing",
        http_status=404,
    )
    assert pub._stripe_missing(e) is True
    assert pub._stripe_missing(stripe_mod.APIConnectionError("timeout")) is False
    assert pub._stripe_missing(TimeoutError("read timed out")) is False


def test_outage_inside_paid_period_keeps_access_with_backoff(server, pub, stripe):
    c, uid = _user(server, pub, "f2in")
    _make_stripe_sub(pub, uid, sub_id="sub_in", period_end=timedelta(days=10),
                     checked=timedelta(hours=-30))
    stripe.sub_error = TimeoutError("read timed out")
    codes = [c.get("/state").status_code for _ in range(6)]
    assert codes == [200] * 6  # paid through the period: fail open
    assert stripe.sub_calls == 1  # ...but ONE call, then back off
    assert stripe.on_event_loop == 0
    assert _row(pub, uid)["sub_status"] == "active"


def test_outage_after_period_end_is_capped_by_the_grace(server, pub, stripe):
    stripe.sub_error = TimeoutError("read timed out")
    # Period ended yesterday: inside the 3-day grace -> still served.
    c1, u1 = _user(server, pub, "f2grace")
    _make_stripe_sub(pub, u1, sub_id="sub_g1", period_end=timedelta(days=-1),
                     checked=timedelta(days=-2))
    assert [c1.get("/state").status_code for _ in range(4)] == [200] * 4
    calls_after_first_user = stripe.sub_calls
    assert calls_after_first_user == 1
    # Period ended 40 days ago: the old code served this forever AND called
    # Stripe on every request.
    c2, u2 = _user(server, pub, "f2expired")
    _make_stripe_sub(pub, u2, sub_id="sub_g2", period_end=timedelta(days=-40),
                     checked=timedelta(days=-40))
    assert [c2.get("/state").status_code for _ in range(5)] == [402] * 5
    assert stripe.sub_calls == calls_after_first_user + 1
    assert c2.get("/me").json()["sub"]["active"] is False
    # Not written off: the cached row stays 'active' so access returns by
    # itself once Stripe answers again.
    assert _row(pub, u2)["sub_status"] == "active"
    stripe.sub_error = None
    stripe.subs["sub_g2"] = {
        "status": "active", "items": {"data": [{"current_period_end": _ts(25)}]},
    }
    pub._STRIPE_RETRY_AT.clear()  # the 15-minute back-off elapsing
    assert c2.get("/state").status_code == 200


def test_verified_past_due_sub_is_not_rechecked_every_request(server, pub, stripe):
    c, uid = _user(server, pub, "f2pd")
    # Stripe says past_due and keeps the OLD period end: still entitled, and
    # still "past the period end" on every request.
    stripe.subs["sub_pd"] = {
        "status": "past_due", "items": {"data": [{"current_period_end": _ts(-2)}]},
    }
    _make_stripe_sub(pub, uid, sub_id="sub_pd", period_end=timedelta(days=-2),
                     checked=timedelta(hours=-1))
    assert [c.get("/state").status_code for _ in range(6)] == [200] * 6
    assert stripe.sub_calls == 1


def test_canceled_sub_is_noticed_on_the_daily_recheck(server, pub, stripe):
    c, uid = _user(server, pub, "f2cancel")
    stripe.subs["sub_c"] = {
        "status": "canceled", "items": {"data": [{"current_period_end": _ts(10)}]},
    }
    _make_stripe_sub(pub, uid, sub_id="sub_c", period_end=timedelta(days=10),
                     checked=timedelta(hours=-25))
    assert c.get("/state").status_code == 402
    assert _row(pub, uid)["sub_status"] == "none"


def test_stripe_client_gets_a_short_timeout(pub, monkeypatch):
    stripe_mod = pytest.importorskip("stripe")
    monkeypatch.setattr(pub, "_STRIPE_CLIENT_READY", False)
    monkeypatch.setattr(stripe_mod, "default_http_client", None, raising=False)
    s = pub._stripe()
    client = s.default_http_client
    assert client is not None
    assert getattr(client, "_timeout", None) == pub.STRIPE_TIMEOUT_S
    assert pub.STRIPE_TIMEOUT_S <= 15


# --- F3: checkout confirmation ---------------------------------------------------------


def _session(uid, *, sid="cs_1", mode="subscription", pay="paid", sub="sub_1", amount=1000):
    return {
        "id": sid, "mode": mode, "client_reference_id": str(uid),
        "payment_status": pay, "subscription": sub, "customer": "cus_1",
        "amount_total": amount, "currency": "usd",
    }


def _active_sub(days=30, status="active"):
    return {"status": status, "items": {"data": [{"current_period_end": _ts(days)}]}}


def test_confirm_activates_a_live_subscription(server, pub, stripe):
    c, uid = _user(server, pub, "f3ok")
    stripe.sessions["cs_ok"] = _session(uid, sid="cs_ok", sub="sub_ok")
    stripe.subs["sub_ok"] = _active_sub()
    assert c.get("/billing/confirm", params={"session_id": "cs_ok"}).json() == {"active": True}
    row = _row(pub, uid)
    assert (row["sub_status"], row["sub_source"]) == ("active", "stripe")
    assert row["stripe_subscription_id"] == "sub_ok"
    assert pub._parse_iso(row["current_period_end"]) > datetime.now(timezone.utc)
    assert c.get("/state").status_code == 200
    n = pub.DB.one("SELECT COUNT(*) c FROM payments WHERE user_id=?", (uid,))["c"]
    assert n == 1
    # Idempotent: confirming again neither errors nor double-books.
    assert c.get("/billing/confirm", params={"session_id": "cs_ok"}).json() == {"active": True}
    assert pub.DB.one("SELECT COUNT(*) c FROM payments WHERE user_id=?", (uid,))["c"] == 1


def test_replaying_an_old_session_does_not_reactivate_a_canceled_sub(server, pub, stripe):
    c, uid = _user(server, pub, "f3replay")
    stripe.sessions["cs_r"] = _session(uid, sid="cs_r", sub="sub_r")
    stripe.subs["sub_r"] = _active_sub()
    assert c.get("/billing/confirm", params={"session_id": "cs_r"}).json()["active"] is True
    # Canceled/refunded; the lazy re-check (25 h later) notices.
    stripe.subs["sub_r"] = _active_sub(status="canceled")
    pub.DB.q("UPDATE users SET sub_checked_at=? WHERE id=?",
             (_iso(timedelta(hours=-25)), uid))
    assert c.get("/state").status_code == 402
    # The receipt is still "paid" forever — replaying it used to buy another
    # 24 h of access each time.
    r = c.get("/billing/confirm", params={"session_id": "cs_r"}).json()
    assert r == {"active": False, "status": "subscription_canceled"}
    assert c.get("/state").status_code == 402
    assert c.get("/me").json()["sub"]["active"] is False
    assert _row(pub, uid)["sub_status"] == "none"


def test_confirm_rejects_non_subscription_sessions(server, pub, stripe):
    c, uid = _user(server, pub, "f3mode")
    stripe.sessions["cs_pay"] = _session(uid, sid="cs_pay", mode="payment", sub=None)
    r = c.get("/billing/confirm", params={"session_id": "cs_pay"}).json()
    assert r == {"active": False, "status": "not_a_subscription"}
    stripe.sessions["cs_nosub"] = _session(uid, sid="cs_nosub", sub=None)
    r = c.get("/billing/confirm", params={"session_id": "cs_nosub"}).json()
    assert r == {"active": False, "status": "not_a_subscription"}
    assert _row(pub, uid)["sub_status"] == "none"
    assert c.get("/state").status_code == 402


def test_confirm_payment_status_gate(server, pub, stripe):
    c, uid = _user(server, pub, "f3pay")
    stripe.subs["sub_p"] = _active_sub(status="trialing")
    stripe.sessions["cs_unpaid"] = _session(uid, sid="cs_unpaid", pay="unpaid", sub="sub_p")
    r = c.get("/billing/confirm", params={"session_id": "cs_unpaid"}).json()
    assert r == {"active": False, "status": "unpaid"}
    # 100% promo code / free trial: nothing to pay, subscription trialing.
    stripe.sessions["cs_free"] = _session(
        uid, sid="cs_free", pay="no_payment_required", sub="sub_p", amount=0
    )
    assert c.get("/billing/confirm", params={"session_id": "cs_free"}).json() == {"active": True}
    assert c.get("/state").status_code == 200


def test_confirm_cross_user_and_errors(server, pub, stripe):
    owner, owner_id = _user(server, pub, "f3owner")
    thief, thief_id = _user(server, pub, "f3thief")
    stripe.sessions["cs_own"] = _session(owner_id, sid="cs_own", sub="sub_own")
    stripe.subs["sub_own"] = _active_sub()
    assert thief.get("/billing/confirm", params={"session_id": "cs_own"}).status_code == 403
    assert _row(pub, thief_id)["sub_status"] == "none"
    assert thief.get("/billing/confirm", params={"session_id": "cs_nope"}).status_code == 404
    # Stripe down while verifying the subscription: 502, nothing activated.
    stripe.sub_error = TimeoutError("read timed out")
    assert owner.get("/billing/confirm", params={"session_id": "cs_own"}).status_code == 502
    assert _row(pub, owner_id)["sub_status"] == "none"


def _fire(client, stripe, etype, obj):
    stripe.event = {"type": etype, "data": {"object": obj}}
    return client.post(
        "/stripe/webhook", content=b"{}", headers={"stripe-signature": "t=1,v1=fake"}
    )


def test_webhook_applies_the_same_verification(server, pub, stripe):
    c, uid = _user(server, pub, "f3hook")
    # Subscription already canceled when the (re-delivered) event arrives.
    stripe.subs["sub_h"] = _active_sub(status="canceled")
    r = _fire(c, stripe, "checkout.session.completed", _session(uid, sub="sub_h"))
    assert r.status_code == 200
    assert _row(pub, uid)["sub_status"] == "none"
    # One-off payment checkout: never a subscription.
    r = _fire(c, stripe, "checkout.session.completed",
              _session(uid, mode="payment", sub=None))
    assert r.status_code == 200
    assert _row(pub, uid)["sub_status"] == "none"
    # Live subscription: activates, off the event loop.
    stripe.subs["sub_h"] = _active_sub()
    r = _fire(c, stripe, "checkout.session.async_payment_succeeded",
              _session(uid, sub="sub_h"))
    assert r.status_code == 200
    assert _row(pub, uid)["sub_status"] == "active"
    assert stripe.on_event_loop == 0
    # Lifecycle event flips it off again.
    r = _fire(c, stripe, "customer.subscription.deleted", {"id": "sub_h", "status": "canceled"})
    assert r.status_code == 200
    assert _row(pub, uid)["sub_status"] == "none"


def test_webhook_stripe_outage_asks_for_redelivery(server, pub, stripe):
    c, uid = _user(server, pub, "f3hook502")
    stripe.sub_error = TimeoutError("read timed out")
    r = _fire(c, stripe, "checkout.session.completed", _session(uid, sub="sub_x"))
    assert r.status_code >= 500  # Stripe retries the event later
    assert _row(pub, uid)["sub_status"] == "none"


def test_stripe_access_without_a_subscription_id_is_not_entitled(server, pub, stripe):
    """Residue of the old bug: mode=payment replays left 'stripe' rows with no
    subscription to ever re-verify."""
    c, uid = _user(server, pub, "f3residue")
    pub.DB.q(
        "UPDATE users SET sub_status='active', sub_source='stripe',"
        " stripe_subscription_id=NULL, sub_checked_at=? WHERE id=?",
        (_iso(timedelta(hours=-1)), uid),
    )
    assert c.get("/state").status_code == 402
    assert stripe.sub_calls == 0
