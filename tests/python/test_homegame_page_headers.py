"""Home games: the page's security headers (2026-09-25). Scripts may only come
from this site, the page can never be framed by another site, nothing is
MIME-sniffed — and the client carries nothing the policy would silently block
(an inline event handler or a javascript: URL would be a dead button in
production, with only a console line to show for it)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "admin@headers.example"
STATIC = Path(__file__).resolve().parents[2] / "python" / "plo5bp" / "ui" / "static"
CLIENT_FILES = [STATIC / "games.html", STATIC / "games.css", *sorted(STATIC.glob("games*.js"))]


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_ADMIN_EMAILS=ADMIN_EMAIL)


@pytest.fixture(scope="module")
def host(server):
    c = TestClient(server.app, raise_server_exceptions=False)
    assert c.get("/auth/dev", params={"email": ADMIN_EMAIL}).status_code == 200
    return c


def _check_page(r):
    assert r.status_code == 200, r.text
    csp = r.headers["content-security-policy"]
    directives = dict(d.strip().split(" ", 1) for d in csp.split(";"))
    assert directives["default-src"] == "'self'"
    assert directives["script-src"] == "'self'"  # no inline scripts, handlers or eval
    assert directives["object-src"] == "'none'"
    assert directives["frame-ancestors"] == "'none'"
    assert "unsafe-eval" not in csp
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "same-origin"
    assert r.headers["cross-origin-opener-policy"] == "same-origin-allow-popups"
    assert "no-store" in r.headers["cache-control"]


def test_the_lobby_and_every_table_page_carry_the_security_headers(host):
    _check_page(host.get("/games"))
    club = host.post("/games/api/clubs", json={"name": "Headers club"})
    assert club.status_code == 200, club.text
    _check_page(host.get(f"/games/join/{club.json()['invite_code']}"))
    r = host.post("/games/api/tables", json={
        "name": "headers", "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 20000, "num_seats": 6,
        "club_id": club.json()["id"],
    })
    assert r.status_code == 200, r.text
    _check_page(host.get(f"/games/t/{r.json()['id']}"))


def test_the_client_files_are_never_mime_sniffed(host, server):
    hg = sys.modules["plo5bp.ui.homegame"]
    for name, media in hg.GAMES_ASSETS.items():
        r = host.get(f"/games/static/{name}")
        assert r.status_code == 200, name
        assert r.headers["x-content-type-options"] == "nosniff", name
        assert r.headers["content-type"].split(";")[0] == media.split(";")[0], name


def test_the_client_has_nothing_the_policy_would_block():
    html = (STATIC / "games.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "an inline <script> in games.html"
    # an on...= attribute inside a tag, in the page or in any HTML the scripts build
    handler = re.compile(r"<[a-zA-Z][^>\n]*[\s\"'/]on[a-z]+\s*=")
    for f in CLIENT_FILES:
        text = f.read_text(encoding="utf-8")
        m = handler.search(text)
        assert m is None, f"an inline event handler in {f.name}: {m.group(0)[:80]!r} — attach a listener instead"
        assert "javascript:" not in text.lower(), f"a javascript: URL in {f.name}"


def test_every_outside_origin_the_client_names_is_allowed_by_the_policy():
    allowed = {"fonts.googleapis.com", "fonts.gstatic.com", "www.w3.org"}  # (w3.org: SVG namespace, never fetched)
    for f in CLIENT_FILES:
        for origin in re.findall(r"https?://([^/\s\"'`)<>]+)", f.read_text(encoding="utf-8")):
            assert origin in allowed, f"{f.name} names {origin}: add it to homegame.PAGE_CSP (and here) if it is fetched"
