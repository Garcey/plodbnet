"""The iPhone home-screen icon (2026-09-26). iOS asks the site ROOT for
``/apple-touch-icon.png`` (and ``-precomposed``) whenever a page names no icon —
both answered 401 to a signed-out phone, and only the landing page named one.
iOS also paints see-through pixels black before rounding the corners itself, so
the icon is square and opaque."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "admin@icon.example"
STATIC = Path(__file__).resolve().parents[2] / "python" / "plo5bp" / "ui" / "static"
ICON = STATIC / "brand" / "apple-touch-icon.png"
LINK = '<link rel="apple-touch-icon" sizes="180x180" href="/static/brand/apple-touch-icon.png"'


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_ADMIN_EMAILS=ADMIN_EMAIL)


def test_the_icon_is_a_square_opaque_180px_png():
    data = ICON.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR"
    width, height, depth, colour_type = struct.unpack(">IIBB", data[16:26])
    assert (width, height, depth) == (180, 180, 8)
    assert colour_type == 2  # RGB, no alpha channel: nothing for iOS to paint black


def test_the_root_paths_serve_it_to_a_signed_out_phone(server):
    anon = TestClient(server.app, raise_server_exceptions=False)
    for path in ("/apple-touch-icon.png", "/apple-touch-icon-precomposed.png",
                 "/static/brand/apple-touch-icon.png"):
        r = anon.get(path)
        assert r.status_code == 200, (path, r.status_code)
        assert r.headers["content-type"] == "image/png"
        assert r.content == ICON.read_bytes()


def test_every_page_names_the_icon(server):
    anon = TestClient(server.app, raise_server_exceptions=False)
    for path in ("/", "/terms", "/privacy", "/games"):  # /games signed out = the sign-in card
        r = anon.get(path, headers={"Accept": "text/html"})
        assert r.status_code == 200, (path, r.status_code)
        assert LINK in r.text, path
    user = TestClient(server.app, raise_server_exceptions=False)
    assert user.get("/auth/dev", params={"email": ADMIN_EMAIL}).status_code == 200
    for path in ("/", "/games", "/admin"):
        r = user.get(path, headers={"Accept": "text/html"})
        assert r.status_code == 200, (path, r.status_code)
        assert LINK in r.text, path
    # The home-screen name: games.js retitles the tab per table ("▶ Your turn · …").
    assert '<meta name="apple-mobile-web-app-title" content="Home games" />' in user.get("/games").text
    assert '<meta name="apple-mobile-web-app-title" content="WrapGTO" />' in user.get("/").text
