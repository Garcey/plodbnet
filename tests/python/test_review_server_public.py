"""Regression tests for the 2026-09-20 code review — PUBLIC build of
`plo5bp/ui/server.py`.

F1: the static mount applies the public file policy to the file a request
RESOLVES to, so no spelling of the path (`//`, trailing `/`, `.` segments,
case) reaches the unstripped live-capture client, the raw index markup or the
hidden home-games / admin assets; and the interactive docs / OpenAPI schema
are not served. F13: `POST /format` with the unchanged format on a fresh
per-user Session no longer 500s.

The app is booted with PLO5BP_PUBLIC=1 against a temp sqlite DB. Env must be
set before `plo5bp.ui.server` imports, so the fixture purges the ui modules
through the shared `ui_purge` helper (tests/python/conftest.py — it also
drops the `plo5bp.ui` PACKAGE ATTRIBUTES, otherwise `from plo5bp.ui import
public` resolves the stale module, review J3) and reimports. On teardown it
purges its public instance again and puts a previously-imported LOCAL-build
instance back, so local test modules keep resolving the objects they hold.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "review-admin@example.com"

_ENV = {
    "PLO5BP_PUBLIC": "1",
    "PLO5BP_DEV_LOGIN": "1",
    # Starlette's TestClient reports client host "testclient"; the dev login
    # only accepts it when a test opts in (review F4).
    "PLO5BP_DEV_LOGIN_TESTCLIENT": "1",
    "PLO5BP_BASE_URL": "http://127.0.0.1:8770",
    "PLO5BP_FREE_HANDS": "3",
    "PLO5BP_ADMIN_EMAILS": ADMIN_EMAIL,
}
_UI_MODULES = (
    "plo5bp.ui.server",
    "plo5bp.ui.public",
    "plo5bp.ui.trainer",
    "plo5bp.ui.homegame",
)


def _snapshot_local_ui_modules() -> dict:
    """The currently-imported ui modules, if they are a LOCAL-build instance
    (a public one is quiesced by the purge and must not be resurrected)."""
    current = sys.modules.get("plo5bp.ui.server")
    if current is None or getattr(current, "PLO5BP_PUBLIC", False):
        return {}
    return {n: sys.modules[n] for n in _UI_MODULES if n in sys.modules}


def _reinstate(modules: dict) -> None:
    pkg = importlib.import_module("plo5bp.ui")
    for name, module in modules.items():
        sys.modules[name] = module
        setattr(pkg, name.rsplit(".", 1)[1], module)


@pytest.fixture(scope="module")
def server(tmp_path_factory, ui_purge):
    tmp = tmp_path_factory.mktemp("review_public")
    env_keys = (*_ENV, "PLO5BP_DB", "PLO5BP_TRAINER_STATS")
    old_env = {k: os.environ.get(k) for k in env_keys}
    os.environ.update(_ENV)
    os.environ["PLO5BP_DB"] = str(tmp / "public.db")
    os.environ["PLO5BP_TRAINER_STATS"] = str(tmp / "default_stats.json")
    local_modules = _snapshot_local_ui_modules()
    ui_purge()
    try:
        mod = importlib.import_module("plo5bp.ui.server")
        assert mod.PLO5BP_PUBLIC is True
        yield mod
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ui_purge()
        _reinstate(local_modules)


@pytest.fixture(scope="module")
def anon(server):
    return TestClient(server.app)


@pytest.fixture(scope="module")
def admin(server):
    c = TestClient(server.app)
    assert c.get("/auth/dev", params={"email": ADMIN_EMAIL}).status_code == 200
    return c


@pytest.fixture(scope="module")
def free_user(server):
    c = TestClient(server.app)
    assert c.get("/auth/dev", params={"email": "review-free@example.com"}).status_code == 200
    return c


def _leaks_live_client(text: str) -> bool:
    # The stripped build legitimately keeps the neutral `window.__wgLive*`
    # hook call sites; what must never ship is the live code itself and the
    # region markers.
    return "pokernow" in text.lower() or "/ocr/" in text or "WGLIVE:" in text


# --- F1: stripped assets, whatever the spelling ---------------------------------

_APP_JS_SPELLINGS = (
    "/static/app.js",
    "/static//app.js",
    "/static///app.js",
    "/static/./app.js",
    "/static/brand/../app.js",
    "/static/app.js/",
    "/static//app.js//",
    "/static/%61pp.js",     # percent-encoded spelling
    "/static/app%2Ejs",
    "/static/APP.JS",       # resolves on case-insensitive filesystems only
    "/static/App.Js",
)


@pytest.mark.parametrize("path", _APP_JS_SPELLINGS)
def test_f1_app_js_is_never_served_unstripped(server, anon, path):
    raw = (server.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert _leaks_live_client(raw), "fixture sanity: the raw file has live code"

    r = anon.get(path)
    assert r.status_code in (200, 404)
    assert not _leaks_live_client(r.text)
    if r.status_code == 200:
        assert r.text == server._strip_wglive(raw)
        assert "no-store" in r.headers["cache-control"]
        assert r.headers["content-type"].startswith("text/javascript")


def test_f1_canonical_assets_are_served_stripped(server, anon):
    for name, ctype in (("app.js", "text/javascript"), ("style.css", "text/css")):
        raw = (server.STATIC_DIR / name).read_text(encoding="utf-8")
        r = anon.get(f"/static/{name}")
        assert r.status_code == 200
        assert r.text == server._strip_wglive(raw)
        assert r.headers["content-type"].startswith(ctype)
        assert "WGLIVE" not in r.text


@pytest.mark.parametrize(
    "path",
    ["/static//style.css", "/static/style.css/", "/static/./style.css"],
)
def test_f1_style_css_variants_are_stripped(server, anon, path):
    raw = (server.STATIC_DIR / "style.css").read_text(encoding="utf-8")
    r = anon.get(path)
    assert r.status_code == 200
    assert r.text == server._strip_wglive(raw)


_DENIED = (
    "games.js", "games.css", "games.html", "index.html", "ranges.js", "admin.html",
)


@pytest.mark.parametrize("name", _DENIED)
@pytest.mark.parametrize(
    "template",
    ["/static/{n}", "/static//{n}", "/static/{n}/", "/static/./{n}",
     "/static/brand/../{n}", "/static///{n}//"],
)
def test_f1_hidden_files_404_from_the_static_mount(server, anon, admin, name, template):
    assert (server.STATIC_DIR / name).is_file(), "fixture sanity"
    path = template.format(n=name)
    assert anon.get(path).status_code == 404
    # Not an auth decision: even the admin (who HAS home-games access) gets
    # these only through their gated routes, never from the static mount.
    assert admin.get(path).status_code == 404


@pytest.mark.parametrize("name", ["games.js", "index.html", "admin.html"])
def test_f1_hidden_files_404_under_case_variants(anon, name):
    r = anon.get(f"/static/{name.upper()}")
    assert r.status_code == 404


@pytest.mark.parametrize(
    "path", ["/static/g%61mes.js", "/static/games%2Ejs", "/static/%69ndex.html"]
)
def test_f1_hidden_files_404_under_percent_encoding(anon, path):
    assert anon.get(path).status_code == 404


def test_f1_gated_routes_still_serve_the_hidden_assets(admin):
    js = admin.get("/games/static/games.js")
    assert js.status_code == 200 and len(js.text) > 1000
    assert admin.get("/games/static/games.css").status_code == 200
    assert admin.get("/games").status_code == 200
    assert admin.get("/admin").status_code == 200


def test_f1_ordinary_static_files_are_untouched(server, anon):
    r = anon.get("/static/terms.html")
    assert r.status_code == 200
    assert r.content == (server.STATIC_DIR / "terms.html").read_bytes()
    assert "no-store" in r.headers["cache-control"]
    brand = sorted(p for p in (server.STATIC_DIR / "brand").iterdir() if p.is_file())
    assert brand, "fixture sanity"
    assert anon.get(f"/static/brand/{brand[0].name}").status_code == 200
    assert anon.get("/static/nope.js").status_code == 404
    assert anon.post("/static/app.js").status_code == 405


def test_f1_index_does_not_request_denied_assets(server, anon, admin):
    raw = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "/static/ranges.js" in raw and "WGLIVE" in raw, "fixture sanity"
    for client in (anon, admin):
        page = client.get("/")
        assert page.status_code == 200
        assert "/static/ranges.js" not in page.text
        assert "WGLIVE" not in page.text
        assert "/static/app.js" in page.text
        assert "window.PLO5BP_PUBLIC=true" in page.text
    # Signed-out visitors additionally lose the app chrome.
    assert "WGAPP" not in anon.get("/").text


# --- F1: no interactive docs / schema in the public build ------------------------


@pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"])
def test_f1_docs_and_schema_are_not_served(anon, free_user, admin, path):
    assert anon.get(path).status_code in (401, 404)
    for client in (free_user, admin):
        r = client.get(path)
        assert r.status_code == 404
        assert "/games/api" not in r.text and "/admin/api" not in r.text


def test_f1_app_has_no_openapi_routes(server):
    assert server.app.openapi_url is None
    assert server.app.docs_url is None and server.app.redoc_url is None
    paths = {getattr(r, "path", None) for r in server.app.router.routes}
    assert not ({"/openapi.json", "/docs", "/redoc"} & paths)


# --- F13 ---------------------------------------------------------------------------


def test_f13_same_format_post_on_a_fresh_user_session(server, admin):
    """The admin's per-user Session is brand new (env None); re-selecting the
    current format skipped the rebuild and asserted in `_state_dict` ⇒ 500."""
    r = admin.post("/format", json={"format": "plo5_double_bomb"})
    assert r.status_code == 200
    assert r.json()["state"]["format"] == "plo5_double_bomb"
    assert admin.get("/state").status_code == 200
