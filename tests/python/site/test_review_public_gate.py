"""Regression tests for the 2026-09-20 review — public access layer.

F1/F7 (middleware authorizes on a normalized path), F4 (dev login), F6
(implicit-deal free-tier bypass), F7 minors (`/format`, logout verb, garbage
ints, `email_verified`). Billing lives in test_review_public_billing.py.
"""

from __future__ import annotations

import sys

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"
FREE_HANDS = 2


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_FREE_HANDS=str(FREE_HANDS))


@pytest.fixture(scope="module")
def pub(server):
    return sys.modules["plo5bp.ui.public"]


def _login(server, email: str) -> TestClient:
    c = TestClient(server.app)
    assert c.get("/auth/dev", params={"email": email}).status_code == 200
    return c


def _raw(client: TestClient, method: str, raw_path: str, **kw):
    """Send `raw_path` byte-for-byte (the client would normalize `//`)."""
    req = client.build_request(method, "http://testserver/", **kw)
    req.url = req.url.copy_with(raw_path=raw_path.encode())
    return client.send(req, follow_redirects=False)


# --- path normalization -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,want",
    [
        ("/", "/"),
        ("", "/"),
        ("//", "/"),
        ("/state/", "/state"),
        ("/state//", "/state"),
        ("//state", "/state"),
        ("/static//games.js", "/static/games.js"),
        ("/static/games.js/", "/static/games.js"),
        ("/static/./games.js", "/static/games.js"),
        ("/static/x/../games.js", "/static/games.js"),
        ("/static\\games.js", "/static/games.js"),
        ("/static/../admin/api/users", "/admin/api/users"),
        ("/../../admin", "/admin"),
        ("/trainer//new_hand/", "/trainer/new_hand"),
    ],
)
def test_norm_path(pub, raw, want):
    assert pub._norm_path(raw) == want


def test_hidden_games_assets_cannot_be_reached_by_path_spelling(server):
    """F1: `/static//games.js` and `/static/games.js/` used to serve the hidden
    home-games client to ANONYMOUS visitors (the gate compared raw strings,
    StaticFiles normalizes)."""
    anon = TestClient(server.app)
    free = _login(server, "gate-free@example.com")
    for client in (anon, free):
        for raw in (
            "/static/games.js",
            "/static//games.js",
            "/static///games.js",
            "/static/games.js/",
            "/static//games.css",
            "/static//games.html",
            "/static/GAMES.JS",  # case-insensitive filesystems
        ):
            r = _raw(client, "GET", raw)
            assert r.status_code == 404, (raw, r.status_code)
            assert "use strict" not in r.text
    # Since clubs (2026-09-25) every SIGNED-IN user has the home-games pages; the
    # anonymous visitor still gets the hidden 404 on every spelling of them.
    for raw in ("/games/", "//games", "/games//api/tables"):
        r = _raw(anon, "GET", raw)
        assert r.status_code == 404, (raw, r.status_code)
        assert "use strict" not in r.text


def test_admin_and_study_gates_ignore_slash_tricks(server):
    free = _login(server, "gate-free2@example.com")
    anon = TestClient(server.app)
    for raw in ("/admin/", "/admin//api/users", "/admin/api/users/", "//admin/api/users"):
        assert _raw(free, "GET", raw).status_code in (403, 404), raw
        assert _raw(anon, "GET", raw).status_code in (401, 404), raw
    # The point: never a 200, and the canonical spellings are 403 / 402.
    assert _raw(free, "GET", "/admin/api/users/").status_code == 403
    for raw in ("/state", "/state/", "//state", "/state//"):
        r = _raw(free, "GET", raw)
        assert r.status_code in (402, 404), (raw, r.status_code)
    assert _raw(free, "GET", "/state/").status_code == 402


def test_format_is_a_study_route(server):
    """F7: a free user's POST /format used to 500 on the never-built study env."""
    free = _login(server, "gate-format@example.com")
    r = free.post("/format", json={"format": "plo5_double_bomb"})
    assert r.status_code == 402
    assert r.json()["error"] == "subscription_required"
    assert free.get("/formats").status_code == 200  # the listing stays open


def test_quota_cannot_be_dodged_with_a_trailing_slash(server):
    free = _login(server, "gate-quota@example.com")
    assert free.get("/trainer/state").status_code == 200
    # Under the limit the slash-spelling is metered like the real route, but
    # the router only answers it with a redirect: nothing dealt, so nothing
    # charged (the redirected request is what gets metered).
    r = _raw(free, "POST", "/trainer/new_hand/")
    assert r.status_code in (307, 308)
    assert free.get("/me").json()["free"]["used"] == 0
    for _ in range(FREE_HANDS):
        assert free.post("/trainer/new_hand").status_code == 200
    assert free.post("/trainer/new_hand").status_code == 402
    # The router redirects `/trainer/new_hand/` -> `/trainer/new_hand`; the
    # redirect itself must neither deal nor burn/refund quota incorrectly.
    r = _raw(free, "POST", "/trainer/new_hand/")
    assert r.status_code in (402, 307, 308)
    assert free.post("/trainer/new_hand").status_code == 402
    assert free.get("/me").json()["free"]["used"] == FREE_HANDS


# --- F6: implicit deals are metered ---------------------------------------------


def _trainer_session(pub, email):
    uid = pub.DB.one("SELECT id FROM users WHERE email=?", (email,))["id"]
    return uid, pub._REGISTRY.peek(uid).trainer


def test_first_implicit_hand_is_free_but_later_ones_are_metered(server, pub):
    email = "gate-implicit@example.com"
    free = _login(server, email)
    # Documented: the first implicit hand of a fresh session is not counted.
    assert free.get("/trainer/state").status_code == 200
    assert free.get("/me").json()["free"]["used"] == 0
    uid, ts = _trainer_session(pub, email)
    assert ts.hand is not None and ts.hand_no == 1

    # The session loses its hand after having dealt (what a format switch
    # does): the next /trainer/state deals implicitly -> metered now.
    with ts.lock:
        ts.hand = None
    r = free.get("/trainer/state")
    assert r.status_code == 200
    assert r.headers.get("X-Free-Hands-Left") == str(FREE_HANDS - 1)
    assert free.get("/me").json()["free"]["used"] == 1
    assert ts.hand is not None and ts.hand_no == 2

    # A live hand: /trainer/state is free again (no deal happens).
    assert free.get("/trainer/state").status_code == 200
    assert free.get("/me").json()["free"]["used"] == 1

    # Burn the rest, then the implicit path is paywalled like new_hand.
    for _ in range(FREE_HANDS - 1):
        assert free.post("/trainer/new_hand").status_code == 200
    with ts.lock:
        ts.hand = None
    for path, method in (("/trainer/state", "GET"), ("/trainer/act", "POST")):
        r = free.request(method, path, json={"gate": "fold"} if method == "POST" else None)
        assert r.status_code == 402, (path, r.status_code)
        assert r.json()["error"] == "free_limit"
    assert ts.hand is None  # nothing was dealt behind the paywall
    assert free.get("/me").json()["free"]["used"] == FREE_HANDS


def test_implicit_deal_is_not_metered_for_entitled_users(server, pub):
    adm = _login(server, ADMIN_EMAIL)
    assert adm.get("/trainer/state").status_code == 200
    uid, ts = _trainer_session(pub, ADMIN_EMAIL)
    before = pub._hands_today(uid)
    with ts.lock:
        ts.hand = None
    assert adm.get("/trainer/state").status_code == 200
    assert pub._hands_today(uid) == before


# --- F4: dev login -------------------------------------------------------------------


def test_dev_login_rejects_anything_proxied(server):
    for header in (
        {"X-Forwarded-For": "203.0.113.9"},
        {"CF-Connecting-IP": "203.0.113.9"},
        {"Forwarded": "for=203.0.113.9"},
        {"X-Real-IP": "203.0.113.9"},
    ):
        c = TestClient(server.app)
        r = c.get("/auth/dev", params={"email": ADMIN_EMAIL}, headers=header)
        assert r.status_code == 403, header
        assert c.get("/me").json()["signed_in"] is False
    # A public Host header (what a tunnel forwards) is not loopback either.
    c = TestClient(server.app)
    r = c.get("/auth/dev", params={"email": ADMIN_EMAIL}, headers={"host": "wrapgto.com"})
    assert r.status_code == 403
    # Non-loopback peer address.
    c = TestClient(server.app, client=("203.0.113.9", 5555))
    assert c.get("/auth/dev", params={"email": ADMIN_EMAIL}).status_code == 403


def test_dev_login_accepts_real_loopback(server):
    for peer, host in (("127.0.0.1", "127.0.0.1:8770"), ("::1", "localhost:8770")):
        c = TestClient(server.app, client=(peer, 5555))
        r = c.get(
            "/auth/dev", params={"email": "loop@example.com"}, headers={"host": host}
        )
        assert r.status_code == 200, (peer, host)
        assert c.get("/me", headers={"host": host}).json()["signed_in"] is True


def test_testclient_host_needs_the_explicit_opt_in(server, pub, monkeypatch):
    monkeypatch.setattr(pub, "DEV_LOGIN_TESTCLIENT", False)
    c = TestClient(server.app)  # client host "testclient"
    assert c.get("/auth/dev", params={"email": ADMIN_EMAIL}).status_code == 403
    assert c.get("/me").json()["dev_login"] is False


def test_me_discloses_dev_login_only_to_loopback(server):
    anon = TestClient(server.app)
    assert anon.get("/me").json()["dev_login"] is True  # opted-in test client
    assert anon.get("/me", headers={"X-Forwarded-For": "203.0.113.9"}).json()[
        "dev_login"
    ] is False
    remote = TestClient(server.app, client=("203.0.113.9", 5555))
    assert remote.get("/me").json()["dev_login"] is False


@pytest.mark.parametrize(
    "host,want",
    [
        ("127.0.0.1", True), ("127.8.9.1", True), ("localhost", True),
        ("LOCALHOST", True), ("::1", True), ("[::1]", True),
        ("", False), (None, False), ("0.0.0.0", False), ("10.0.0.5", False),
        ("wrapgto.com", False), ("localhost.evil.com", False), ("testclient", False),
    ],
)
def test_is_loopback_host(pub, host, want):
    assert pub._is_loopback_host(host) is want


# --- F7 minors ---------------------------------------------------------------------------


def test_logout_accepts_post_and_get(server):
    for method in ("POST", "GET"):
        c = _login(server, f"logout-{method.lower()}@example.com")
        assert c.get("/me").json()["signed_in"] is True
        r = c.request(method, "/auth/logout", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/"
        assert c.get("/me").json()["signed_in"] is False


def test_admin_garbage_ints_are_400_not_500(server):
    adm = TestClient(server.app, raise_server_exceptions=False)
    assert adm.get("/auth/dev", params={"email": ADMIN_EMAIL}).status_code == 200
    for bad in ("abc", None, [1], {"a": 1}, 1.5, True, "1e999", 10**30):
        for path in ("/admin/api/grant", "/admin/api/games_access"):
            r = adm.post(path, json={"user_id": bad, "action": "grant"})
            assert r.status_code == 400, (path, bad, r.status_code)
    r = adm.post("/admin/api/grant", json={"action": "grant"})
    assert r.status_code == 400
    r = adm.post("/admin/api/grant", json={"user_id": 999999, "action": "grant"})
    assert r.status_code == 404


def test_body_int(pub):
    from fastapi import HTTPException

    assert pub.body_int({"n": 5}, "n") == 5
    assert pub.body_int({"n": "7"}, "n") == 7
    assert pub.body_int({"n": 3.0}, "n") == 3
    assert pub.body_int({}, "n", 9) == 9
    assert pub.body_int({"n": None}, "n", 9) == 9
    for bad in ("x", 1.5, True, [1], float("nan"), float("inf"), 2**60):
        with pytest.raises(HTTPException) as e:
            pub.body_int({"n": bad}, "n")
        assert e.value.status_code == 400
    with pytest.raises(HTTPException):
        pub.body_int({}, "n")


def test_verified_email_helper(pub):
    ok = {"sub": "g", "email": " Someone@Example.COM ", "email_verified": True}
    assert pub._verified_email(ok) == "someone@example.com"
    assert pub._verified_email({**ok, "email_verified": "true"}) == "someone@example.com"
    # Claim ABSENT used to count as verified.
    assert pub._verified_email({"sub": "g", "email": "a@example.com"}) is None
    for bad in (False, None, 0, "false", "yes", 1, []):
        assert pub._verified_email({**ok, "email_verified": bad}) is None, bad
    assert pub._verified_email({"email_verified": True}) is None
    assert pub._verified_email(None) is None


def _oauth_callback(server, userinfo: dict):
    """Drive GET /auth/callback with a fake Authlib client that returns
    `userinfo` (the route closes over `oauth`, which is None without Google
    credentials — swap the closure cell for the duration)."""

    class _FakeGoogle:
        async def authorize_access_token(self, request):
            return {"userinfo": userinfo}

    class _FakeOAuth:
        google = _FakeGoogle()

    route = next(
        r for r in server.app.router.routes
        if getattr(r, "path", "") == "/auth/callback"
    )
    fn = route.endpoint
    cell = fn.__closure__[fn.__code__.co_freevars.index("oauth")]
    old = cell.cell_contents
    cell.cell_contents = _FakeOAuth()
    try:
        c = TestClient(server.app)
        r = c.get("/auth/callback", follow_redirects=False)
        return r.headers.get("location", ""), c.get("/me").json()["signed_in"]
    finally:
        cell.cell_contents = old


def test_oauth_callback_requires_the_verified_claim(server):
    loc, signed_in = _oauth_callback(server, {"sub": "g1", "email": "noclaim@example.com"})
    assert signed_in is False and "login=failed" in loc
    loc, signed_in = _oauth_callback(
        server, {"sub": "g2", "email": "unv@example.com", "email_verified": False}
    )
    assert signed_in is False and "login=failed" in loc
    loc, signed_in = _oauth_callback(
        server, {"sub": "g3", "email": "ver@example.com", "email_verified": True}
    )
    assert signed_in is True and loc == "/"
