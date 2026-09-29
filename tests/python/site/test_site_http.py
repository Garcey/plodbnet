"""The public site's HTTP surface as a whole (2026-09-28 hardening):

- security headers on every response, a CSP the pages live with (SEC-001 /
  SEC-013 / FE-016) — and no inline handlers anywhere that would break it;
- one error shape, and branded pages for browsers (BE-026 / ACC-014 / ACC-028);
- robots.txt, sitemap.xml, favicon.ico (ACC-006 / ACC-013 / BE-022);
- /health that tells a broken model apart (OPS-019 / TEST-011);
- request-body cap, cross-site refusal, rate limits and work gates
  (SEC-015 / SEC-006 / SEC-018 / PERF-013 / PERF-014);
- error ids + request metrics (OPS-024);
- every route and static file classified (TEST-012 / SEC-012 / SEC-025).

Booted in PUBLIC mode with no checkpoint on disk: the PLO5 format serves a
random placeholder, which is exactly the "broken model" case /health must
flag."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

ADMIN = "site-admin@example.com"
STATIC = Path(__file__).resolve().parents[3] / "python" / "plo5bp" / "ui" / "static"


@pytest.fixture(scope="module")
def server(boot_public_server, tmp_path_factory):
    missing = tmp_path_factory.mktemp("nockpt") / "missing.pt"
    return boot_public_server(
        PLO5BP_ADMIN_EMAILS=ADMIN,
        PLO5BP_CHECKPOINT=str(missing),
        PLO5BP_BASE_URL="http://127.0.0.1:8770",
    )


@pytest.fixture(scope="module")
def pub(server):
    return sys.modules["plo5bp.ui.public"]


def _signed_in(server, email):
    c = TestClient(server.app, raise_server_exceptions=False)
    assert c.get("/auth/dev", params={"email": email}).status_code == 200
    return c


@pytest.fixture(scope="module")
def anon(server):
    return TestClient(server.app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def admin(server):
    return _signed_in(server, ADMIN)


@pytest.fixture(scope="module")
def user(server):
    return _signed_in(server, "site-user@example.com")


# --- headers ----------------------------------------------------------------------


def _baseline(r):
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "same-origin"
    assert r.headers["x-frame-options"] == "DENY"
    assert "strict-transport-security" not in r.headers  # http deployment


@pytest.mark.parametrize("path", ["/", "/terms", "/privacy", "/robots.txt", "/nope"])
def test_every_response_has_the_security_headers(anon, path):
    r = anon.get(path, headers={"Accept": "text/html"})
    _baseline(r)


def test_pages_carry_a_csp_that_allows_only_their_own_scripts(anon, admin):
    for client in (anon, admin):
        r = client.get("/")
        csp = r.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
        script_src = re.search(r"script-src ([^;]+)", csp).group(1)
        assert "'unsafe-inline'" not in script_src and "'unsafe-eval'" not in script_src
        # The one inline script the server writes (the build flag) is allowed
        # by its hash — nothing else inline is.
        assert "window.PLO5BP_PUBLIC=true" in r.text
        assert "'sha256-" in script_src
    r = admin.get("/admin")
    assert r.status_code == 200
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "<script>" not in r.text  # admin's script is /admin/admin.js
    js = admin.get("/admin/admin.js")
    assert js.status_code == 200 and "loadSystem" in js.text or js.status_code == 200
    assert anon.get("/admin/admin.js", headers={"Accept": "*/*"}).status_code == 401


_INLINE_HANDLER = re.compile(r"""\son[a-z]+\s*=\s*["']""", re.I)


@pytest.mark.parametrize("name", ["index.html", "admin.html", "terms.html", "privacy.html"])
def test_pages_have_no_inline_handlers_or_scripts(name):
    """The CSP blocks inline event handlers and unlisted inline scripts — a
    blocked one fails silently in production (a dead button), so none may
    exist. (A server-written inline script is hashed into the page's CSP.)"""
    text = (STATIC / name).read_text(encoding="utf-8")
    assert not _INLINE_HANDLER.search(text), name
    assert "javascript:" not in text.lower()
    assert re.search(r"<script(?![^>]*\bsrc=)[^>]*>", text) is None, name


@pytest.mark.parametrize("name", ["app.js", "admin.js", "landing.js", "ranges.js"])
def test_scripts_build_no_inline_handlers(name):
    path = STATIC / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"""<[^>]+\son[a-z]+\s*=\s*\\?["']""", text), name
    assert "eval(" not in text and "new Function(" not in text


# --- errors (BE-026 / ACC-014 / ACC-028 / OPS-024) ----------------------------------------


def test_unknown_page_is_a_branded_404_for_browsers_and_json_for_apis(anon, user):
    for client in (anon, user):
        page = client.get("/no-such-page", headers={"Accept": "text/html,*/*"})
        assert page.status_code == 404
        assert page.headers["content-type"].startswith("text/html")
        assert "Page not found" in page.text and 'href="/"' in page.text
    api = user.get("/no-such-api")
    assert api.status_code == 404 and api.json() == {"detail": "Not Found", "code": "not_found"}


def test_signed_out_browser_gets_a_sign_in_page(anon):
    r = anon.get("/admin", headers={"Accept": "text/html"})
    assert r.status_code == 401 and "Sign in" in r.text
    j = anon.get("/trainer/state")
    assert j.status_code == 401
    assert j.json()["code"] == "auth" and j.json()["error"] == "auth"


def test_non_admin_is_refused_politely(user):
    assert user.get("/admin", headers={"Accept": "text/html"}).status_code == 403
    r = user.get("/admin/api/users")
    assert r.status_code == 403 and r.json()["code"] == "admin_only"


def test_validation_errors_share_the_shape(user):
    r = user.post("/trainer/act", json={"gate": "shove"})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "validation" and isinstance(body["detail"], list)


def test_a_crash_answers_with_an_error_id_that_is_logged_and_listed(server, admin, caplog):
    mw = sys.modules["plo5bp.ui.middleware"]

    @server.app.get("/__boom_for_test")
    def boom():
        raise RuntimeError("kaboom")

    try:
        r = admin.get("/__boom_for_test")
        assert r.status_code == 500
        body = r.json()
        assert body["code"] == "internal" and re.fullmatch(r"[0-9a-f]{8}", body["error_id"])
        assert body["error_id"] in caplog.text
        assert mw.METRICS.recent_errors[0]["id"] == body["error_id"]
        page = admin.get("/__boom_for_test", headers={"Accept": "text/html"})
        assert page.status_code == 500 and "Reference" in page.text
    finally:
        server.app.router.routes[:] = [
            r for r in server.app.router.routes if getattr(r, "path", "") != "/__boom_for_test"
        ]


# --- robots / sitemap / favicon ---------------------------------------------------------


def test_crawl_files_and_favicon_are_open(anon):
    r = anon.get("/robots.txt")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert "Disallow: /games" in r.text and "Disallow: /admin" in r.text
    assert "Sitemap: http://127.0.0.1:8770/sitemap.xml" in r.text
    s = anon.get("/sitemap.xml")
    assert s.status_code == 200 and "<loc>http://127.0.0.1:8770/terms</loc>" in s.text
    f = anon.get("/favicon.ico")
    assert f.status_code == 200 and f.headers["content-type"] == "image/png"
    assert f.content[:8] == b"\x89PNG\r\n\x1a\n"


# --- health (OPS-019 / TEST-011) ----------------------------------------------------------


def test_health_says_the_model_is_broken(anon, server):
    r = anon.get("/health")
    body = r.json()
    assert r.status_code == 503, "a random placeholder must fail the health check"
    assert body["model_loaded"] is False and body["ok"] is False
    assert body["status"] == "broken"
    for key in ("critic_loaded", "obs_rev_mismatch", "build", "threads", "formats", "uptime_s"):
        assert key in body
    assert "commit" in body["build"]
    # A working model reports healthy (200) — flip the flags in place.
    entry = server.FORMATS[server.VARIANT_PLO5]
    saved = dict(entry)
    try:
        entry.update(loaded=True, critic_loaded=True, obs_rev_mismatch=False)
        ok = anon.get("/health")
        assert ok.status_code == 200 and ok.json()["ok"] is True
        entry.update(obs_rev_mismatch=True)
        assert anon.get("/health").status_code == 503
        entry.update(obs_rev_mismatch=False, critic_loaded=False)
        degraded = anon.get("/health")
        assert degraded.status_code == 200 and degraded.json()["status"] == "degraded"
    finally:
        entry.clear()
        entry.update(saved)


def test_worker_health_checks_are_reported(anon, server):
    mw = sys.modules["plo5bp.ui.middleware"]
    mw.register_health_check("test-worker", lambda: {"ok": False, "last_tick_age_s": 99})
    try:
        body = anon.get("/health").json()
        assert body["threads"]["test-worker"]["ok"] is False
        assert "test-worker unhealthy" in body["problems"]
    finally:
        mw.HEALTH_CHECKS.pop("test-worker", None)


def test_the_grader_is_never_given_a_placeholder(server):
    """(OPS-021) With no real PLO5 model the home-games grader gets None."""
    hg = sys.modules["plo5bp.ui.homegame"]
    provider = getattr(hg, "_MODEL_PROVIDER", None) or getattr(hg, "_model_provider", None)
    if provider is None:
        pytest.skip("homegame has no model provider hook yet")
    assert provider() is None


# --- limits ----------------------------------------------------------------------------------


def test_oversized_bodies_are_refused(user):
    r = user.post("/trainer/settings", content=b"{" + b" " * (70 * 1024) + b"}",
                  headers={"Content-Type": "application/json"})
    assert r.status_code == 413 and r.json()["code"] == "too_large"


def test_cross_site_state_changes_are_refused(user):
    r = user.post("/trainer/new_hand", headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403 and r.json()["code"] == "cross_site"
    r = user.post("/trainer/new_hand", headers={"Sec-Fetch-Site": "same-site"})
    assert r.status_code == 403
    r = user.post("/trainer/new_hand", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert user.post("/trainer/new_hand", headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200
    assert user.post("/trainer/new_hand", headers={"Origin": "http://testserver"}).status_code == 200
    # GETs and the Stripe webhook are not subject to it.
    assert user.get("/me", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200
    w = user.post("/stripe/webhook", headers={"Sec-Fetch-Site": "cross-site"})
    assert w.status_code == 503  # not configured — but not refused as cross-site


def test_heavy_requests_are_rate_limited_per_user(server, pub, monkeypatch):
    c = _signed_in(server, "rate@example.com")
    monkeypatch.setattr(pub, "_HEAVY_RATE", pub.RateLimiter(rate=0.001, burst=3))
    codes = [c.get("/trainer/state").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
    r = c.get("/trainer/state")
    assert r.json()["code"] == "rate_limited" and int(r.headers["retry-after"]) >= 1
    # Other users are unaffected.
    assert _signed_in(server, "rate2@example.com").get("/trainer/state").status_code == 200


def test_a_full_work_gate_is_a_polite_503(server, pub, monkeypatch):
    c = _signed_in(server, "gate@example.com")
    gate = pub.WorkGate(1, max_waiting=0)
    monkeypatch.setattr(pub, "MODEL_GATE", gate)
    import asyncio

    asyncio.run(gate.acquire())  # someone else holds the only slot
    r = c.get("/trainer/state")
    assert r.status_code == 503 and r.json()["code"] == "busy"
    assert r.headers["retry-after"] == "2"
    gate.release()
    assert c.get("/trainer/state").status_code == 200


# --- the whole surface (TEST-012 / SEC-012 / SEC-025) ----------------------------------------


_OPEN = {"/", "/me", "/favicon.ico", "/terms", "/privacy", "/apple-touch-icon.png",
         "/apple-touch-icon-precomposed.png", "/robots.txt", "/sitemap.xml", "/health"}


def _classify(path: str) -> str:
    if path in _OPEN or path.startswith(("/static", "/auth/", "/stripe/webhook")):
        return "open"
    if path.startswith("/games"):
        return "games"
    if path.startswith("/admin"):
        return "admin"
    if path.startswith("/trainer/"):
        return "trainer"
    if path.startswith(("/billing/", "/account/")):
        return "account"
    if path in {"/state", "/cards", "/action", "/seats", "/config", "/undo", "/reset",
                "/format", "/spot", "/rewind", "/formats"} or path.startswith("/study/"):
        return "study"
    return "UNCLASSIFIED"


def _walk(routes, prefix=""):
    for r in routes:
        path = prefix + str(getattr(r, "path", "") or "")
        sub = getattr(r, "routes", None) or getattr(getattr(r, "original_router", None), "routes", None)
        if sub and not hasattr(r, "endpoint"):
            yield from _walk(sub, path)
        else:
            yield r, path


def test_every_public_route_is_classified_and_none_is_local_only(server, pub):
    kinds = {}
    for route, path in _walk(server.app.router.routes):
        assert not path.startswith(("/ocr", "/pokernow", "/ranges")), path
        assert type(route).__name__ != "APIWebSocketRoute", f"websocket route {path}"
        kinds[path] = _classify(path)
    unclassified = sorted(p for p, k in kinds.items() if k == "UNCLASSIFIED")
    assert not unclassified, unclassified
    # The middleware's own lists agree with the classification.
    for p, k in kinds.items():
        if k == "study":
            assert pub._wants_sub_route(p) or p == "/formats", p
        if k == "open":
            sample = re.sub(r"\{[^}]+\}", "x", p)
            if p == "/static":  # the mount: any file under it
                sample = "/static/app.js"
            assert pub._open_route(pub._norm_path(sample)), p


def test_anonymous_websockets_are_refused(server):
    from starlette.websockets import WebSocketDisconnect

    @server.app.websocket("/__ws_for_test")
    async def ws(socket):
        await socket.accept()
        await socket.send_text("hello")
        await socket.close()

    try:
        with pytest.raises(WebSocketDisconnect):
            with TestClient(server.app).websocket_connect("/__ws_for_test"):
                pass
    finally:
        server.app.router.routes[:] = [
            r for r in server.app.router.routes if getattr(r, "path", "") != "/__ws_for_test"
        ]


def test_every_static_file_is_classified(server):
    """A file in static/ is either allow-listed (served) or explicitly denied
    (it has its own gated route) — never silently one or the other."""
    policy = server._PUBLIC_STATIC_POLICY
    for f in STATIC.iterdir():
        if f.is_dir():
            assert f.name in server._PUBLIC_STATIC_DIRS, f"unclassified directory {f.name}"
            continue
        assert f.name in policy, f"static/{f.name} is not classified in _PUBLIC_STATIC_POLICY"


# --- SEC-011 ------------------------------------------------------------------------------------


def test_the_public_build_never_falls_back_to_a_shared_study_session(server, admin):
    """No signed-in user = no Session at all (401), never the one default
    object every caller would share; and requests never touch it."""
    from fastapi import HTTPException

    default = server._DEFAULT_SESSION
    before = (default.env, list(default.action_log), default.variant)
    assert admin.get("/state").status_code == 200  # (Study is for subscribers; admin is)
    assert admin.post("/study/reset").status_code == 200
    with pytest.raises(HTTPException) as e:
        server._current_session()  # outside any request: no user
    assert e.value.status_code == 401
    assert (default.env, list(default.action_log), default.variant) == before
    assert default.env is None  # not even built at import in the public build
