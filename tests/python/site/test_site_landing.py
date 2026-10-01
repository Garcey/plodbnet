"""The public landing page and sign-in plumbing (site ACC-016 / ACC-017 /
ACC-025 / PERF-002 / MOB-012 / ACC-007).

- A signed-out visitor gets a page that works without any script: the
  landing is not hidden, every "Sign in" is a real link, and the app bundle
  is not sent (only the tiny landing.js).
- The free-period copy and the paid-plan copy follow PLO5BP_FREE_FOR_ALL,
  and the paid copy's price / daily hands come from the settings.
- A failed Google sign-in says why (cancelled / expired / unverified).
"""

from __future__ import annotations

import re
import sys

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_FREE_FOR_ALL="1")


@pytest.fixture(scope="module")
def pub(server):
    return sys.modules["plo5bp.ui.public"]


def _anon(server):
    return TestClient(server.app, raise_server_exceptions=False)


def test_signed_out_landing_needs_no_script(server):
    html = _anon(server).get("/").text
    tag = re.search(r'<div id="login-overlay"[^>]*>', html).group(0)
    assert "display:none" not in tag.replace(" ", "")
    assert "hidden" not in tag
    # no app bundle for a marketing page; the landing extras only
    assert "/static/app.js" not in html
    assert "/static/landing.js" in html
    # sign-in is a plain link (works before/without JS)
    assert re.search(r'<a[^>]+class="gate-google"[^>]+href="/auth/login"|<a class="gate-google" href="/auth/login"', html)
    assert 'href="/auth/login"' in html
    # headline comes before any notice; one h1
    assert html.count("<h1") == 1
    assert "WGFREE" not in html and "WGPAID" not in html and "{{" not in html


def test_free_copy_follows_the_flag(server, pub, monkeypatch):
    html = _anon(server).get("/").text
    assert "Free while the models are in development" in html
    assert "a month" not in html
    raw = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    monkeypatch.setattr(pub, "FREE_FOR_ALL", False)
    monkeypatch.setattr(pub, "PRICE_CENTS", 1250)
    monkeypatch.setattr(pub, "FREE_HANDS_PER_DAY", 3)
    paid = server._select_pricing_copy(raw)
    assert "Free while the models are in development" not in paid
    assert "$12.50" in paid and "3 free trainer hands a day" in paid
    assert "WGPAID" not in paid and "WGFREE" not in paid and "{{" not in paid
    monkeypatch.setattr(pub, "PRICE_CENTS", 1000)
    assert "$10 a month" in server._select_pricing_copy(raw)


def test_signed_in_page_keeps_the_app_before_the_landing(server):
    c = TestClient(server.app, raise_server_exceptions=False)
    assert c.get("/auth/dev", params={"email": "landing@example.com"}).status_code == 200
    html = c.get("/").text
    assert "/static/app.js" in html
    # CSS hides the landing when the app chrome precedes it
    assert html.index('id="top-bar"') < html.index('id="login-overlay"')


def test_link_preview_tags(server):
    html = _anon(server).get("/").text
    for needle in ('property="og:title"', 'property="og:image"', 'name="twitter:card"',
                   'name="theme-color"', 'rel="canonical"', "viewport-fit=cover",
                   'rel="manifest"'):
        assert needle in html, needle
    r = _anon(server).get("/static/brand/manifest.webmanifest")
    assert r.status_code == 200
    assert r.json()["icons"][0]["src"].startswith("/static/brand/")


def test_login_failure_reasons(pub):
    class MismatchingStateError(Exception):
        pass

    class OAuthError(Exception):
        def __init__(self, error):
            super().__init__(error)
            self.error = error

    class Req:
        query_params: dict = {}

    assert pub._login_failure_reason(Req(), OAuthError("access_denied")) == "cancelled"
    assert pub._login_failure_reason(Req(), MismatchingStateError()) == "expired"
    assert pub._login_failure_reason(Req(), RuntimeError("boom")) == "error"


def test_invite_pages_have_link_previews(server):
    hg = sys.modules["plo5bp.ui.homegame"]
    html = hg._signin_html("Join Miles's club", "<b>Miles &amp; co</b> invited you", "/games/join/abc")
    assert '<meta property="og:title" content="Join Miles&#x27;s club">' in html
    assert '<meta property="og:description" content="Miles &amp; co invited you">' in html
    assert 'name="twitter:card" content="summary"' in html


def test_legal_pages_describe_the_site_as_it_is(server):
    anon = _anon(server)
    terms = anon.get("/terms").text
    privacy = anon.get("/privacy").text
    for page in (terms, privacy):
        assert "Effective September 28, 2026" in page
        assert "support@wrapgto.com" in page
        assert 'style="' not in page          # no inline styles (CSP-friendly)
    # free period, not a daily limit and $10 plans
    assert "currently free" in terms
    assert "$10" not in terms and "limited number of trainer hands" not in terms
    # home games: no money through the site, what others see, the shuffle's limits
    assert "no money is paid to, held by, or sent through WrapGTO" in terms
    assert "cannot prove that nobody at WrapGTO could see them" in terms
    # privacy covers pictures, chat, clubs, self-service export / deletion, backups
    for needle in ("profile picture you upload", "chat messages", "clubs you start or join",
                   "download a copy of your data", "delete your account", "up to two years"):
        assert needle in privacy, needle
    assert "encrypted-in-transit" not in privacy


def test_fonts_switch_to_self_hosted_files(server, monkeypatch):
    """FE-024: once both .woff2 files are in static/fonts/, every page loads
    the fonts from this site (versioned URLs, the preload and the @font-face
    naming the same one) and the privacy policy stops saying Google serves
    them; until then the pages keep the Google links. Markers never leak."""
    static = server.STATIC_DIR
    raw_index = (static / "index.html").read_text(encoding="utf-8")
    raw_privacy = (static / "privacy.html").read_text(encoding="utf-8")
    note = "loaded from Google's servers"
    for raw in (raw_index, raw_privacy, (static / "terms.html").read_text(encoding="utf-8")):
        google = server._font_links(raw, None)
        assert "fonts.googleapis.com/css2" in google and "WGFONT" not in google
        local = server._font_links(raw, ("a" * 16, "b" * 16))
        assert "fonts.googleapis.com" not in local and "fonts.gstatic.com" not in local
        assert "WGFONT" not in local
        assert 'rel="preload" href="/static/fonts/Geist-Variable.woff2?v=' + "a" * 16 in local
        assert 'url("/static/fonts/Geist-Variable.woff2?v=' + "a" * 16 + '")' in local
        assert 'url("/static/fonts/GeistMono-Variable.woff2?v=' + "b" * 16 + '")' in local
    assert note in server._font_links(raw_privacy, None)
    assert note not in server._font_links(raw_privacy, ("a" * 16, "b" * 16))

    anon = _anon(server)
    on_disk = server._font_versions(server._PAGES.versions)
    for path in ("/", "/terms", "/privacy"):
        html = anon.get(path).text
        assert "WGFONT" not in html
        assert ("fonts.googleapis.com" in html) == (on_disk is None)
    monkeypatch.setattr(server, "_font_versions", lambda _v: ("c" * 16, "d" * 16))
    for path in ("/", "/terms", "/privacy"):
        html = anon.get(path).text
        assert "fonts.googleapis.com" not in html
        assert "/static/fonts/Geist-Variable.woff2?v=" + "c" * 16 in html
    assert note not in anon.get("/privacy").text


def test_admin_page_follows_the_fonts_too(server, pub, monkeypatch):
    c = TestClient(server.app, raise_server_exceptions=False)
    admin = sorted(pub.ADMIN_EMAILS)[0]
    assert c.get("/auth/dev", params={"email": admin}).status_code == 200
    html = c.get("/admin").text
    assert "WGFONT" not in html and "Admin" in html
    monkeypatch.setattr(server, "_font_versions", lambda _v: ("e" * 16, "f" * 16))
    html = c.get("/admin").text
    assert "fonts.googleapis.com" not in html
    assert "/static/fonts/GeistMono-Variable.woff2?v=" + "f" * 16 in html


def test_signed_in_page_loads_every_client_part(server):
    """FE-019: the client is seven plain scripts. A signed-in page links every
    one, in order and versioned; the public build serves each (cached for a
    year at its version) with no live-capture code; a signed-out visitor gets
    none of them."""
    c = TestClient(server.app, raise_server_exceptions=False)
    assert c.get("/auth/dev", params={"email": "parts@example.com"}).status_code == 200
    html = c.get("/").text
    parts = re.findall(r'<script src="/static/(app(?:\.[a-z]+)?\.js)\?v=([0-9a-f]+)"', html)
    assert [name for name, _ in parts] == [
        "app.core.js", "app.table.js", "app.play.js", "app.study.js",
        "app.trainer.js", "app.topbar.js", "app.js",
    ]
    for name, version in parts:
        r = c.get(f"/static/{name}?v={version}")
        assert r.status_code == 200, name
        assert "immutable" in r.headers["cache-control"], name
        assert "WGLIVE" not in r.text and "pokernow" not in r.text.lower() and "/ocr/" not in r.text
    assert "/static/app." not in _anon(server).get("/").text


def test_signed_in_page_draws_the_home_games_table(server):
    """2026-10-01: Study and Trainer draw the home games' table. A signed-in page links
    its stylesheet and renderer from their gated home, /games/static (the /static
    mount refuses every games.* file), versioned, the stylesheet before style.css and
    the renderer before the client; a signed-out page links neither. Every element the
    renderer looks up is in the page."""
    c = TestClient(server.app, raise_server_exceptions=False)
    assert c.get("/auth/dev", params={"email": "felt@example.com"}).status_code == 200
    html = c.get("/").text
    css = re.search(r'<link rel="stylesheet" href="/games/static/games\.felt\.css\?v=([0-9a-f]+)"', html)
    js = re.search(r'<script src="/games/static/games\.table\.js\?v=([0-9a-f]+)"></script>', html)
    assert css and js
    assert html.index("games.felt.css") < html.index("/static/style.css")
    assert html.index("games.table.js") < html.index("/static/app.core.js")
    for name, m in (("games.felt.css", css), ("games.table.js", js)):
        assert c.get(f"/games/static/{name}?v={m.group(1)}").status_code == 200, name
        assert c.get(f"/static/{name}").status_code == 404, name
    anon = _anon(server).get("/").text
    assert "games.felt.css" not in anon and "games.table.js" not in anon
    renderer = (server.STATIC_DIR / "games.table.js").read_text(encoding="utf-8")
    page = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    ids = sorted(set(re.findall(r'\$\("([a-z0-9-]+)"\)', renderer)))
    assert len(ids) > 15
    for el_id in ids:
        assert f'id="{el_id}"' in page, el_id
