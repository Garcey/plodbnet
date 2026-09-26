"""Home games — the host's settings from last time and profile pictures
(owner, 2026-09-26).

Booted once for the module in PUBLIC mode against a temp DB
(`boot_public_server`, tests/python/conftest.py).
"""

from __future__ import annotations

import base64
import struct
import sys
import zlib

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "admin@profile.example"
NAMES = ["host", "guest", "stranger"]


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_ADMIN_EMAILS=ADMIN_EMAIL)


@pytest.fixture(scope="module")
def hg(server):
    return sys.modules["plo5bp.ui.homegame"]


@pytest.fixture(scope="module")
def cast(server):
    def login(email):
        c = TestClient(server.app, raise_server_exceptions=False)
        assert c.get("/auth/dev", params={"email": email}).status_code == 200
        return c

    adm = login(ADMIN_EMAIL)
    people = {n: login(f"{n}@example.com") for n in NAMES}
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for n in ("host", "guest"):  # the stranger stays out of the club
        r = adm.post("/admin/api/games_access", json={"user_id": ids[f"{n}@example.com"], "action": "grant"})
        assert r.status_code == 200
    return {**people, "ids": {n: ids[f"{n}@example.com"] for n in NAMES}}


def _png(w: int, h: int) -> bytes:
    raw = b"".join(b"\x00" + b"\x10\x20\x30" * w for _ in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


# --- the host's settings from last time --------------------------------------------------


def test_a_new_table_starts_from_the_hosts_last_settings(cast):
    host = cast["host"]
    assert host.get("/games/api/host_prefs").json() == {"prefs": None}, "never hosted: the defaults"
    r = host.post("/games/api/tables", json={
        "name": "friday", "bb_cents": 500, "ante_cents": 1000, "default_buyin_cents": 20000,
        "num_seats": 6, "decision_secs": 45, "allow_rathole": True, "approve_buyins": True,
    })
    assert r.status_code == 200, r.text
    gid = r.json()["id"]
    p = host.get("/games/api/host_prefs").json()["prefs"]
    assert (p["bb_cents"], p["ante_bb"], p["buyin_bb"], p["num_seats"]) == (500, 2.0, 40.0, 6)
    assert p["decision_secs"] == 45 and p["allow_rathole"] is True and p["approve_buyins"] is True
    # what the host changes later from Manage is remembered too
    assert host.post(f"/games/api/tables/{gid}/settings", json={"show_grades": False, "allow_rabbit": False}).status_code == 200
    assert host.post(f"/games/api/tables/{gid}/auto_topup", json={"mode": "player"}).status_code == 200
    p = host.get("/games/api/host_prefs").json()["prefs"]
    assert p["show_grades"] is False and p["allow_rabbit"] is False and p["topup_mode"] == "player"
    # the next table: the dialog sends its fields + remembered=true -> Manage's settings come along
    again = host.post("/games/api/tables", json={"name": "saturday", "bb_cents": 500, "ante_cents": 1000,
                                                 "default_buyin_cents": 20000, "remembered": True}).json()
    assert again["settings"]["show_grades"] is False and again["settings"]["allow_rabbit"] is False
    assert again["auto_topup"]["mode"] == "player"
    # a script / an old client that doesn't ask gets the plain defaults
    plain = host.post("/games/api/tables", json={"name": "script", "bb_cents": 100}).json()
    assert plain["settings"]["show_grades"] is True and plain["auto_topup"]["mode"] == "off"
    # prefs are per host
    assert cast["guest"].get("/games/api/host_prefs").json() == {"prefs": None}


# --- profile pictures ------------------------------------------------------------------


def test_image_headers_are_read_without_an_image_library(hg):
    assert hg._image_info(_png(40, 30)) == ("image/png", 40, 30)
    jpeg = (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xc0\x00\x11\x08\x00\x20\x00\x30\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01")
    assert hg._image_info(jpeg) == ("image/jpeg", 48, 32)
    vp8x = b"RIFF" + struct.pack("<I", 30) + b"WEBP" + b"VP8X" + struct.pack("<I", 10) + b"\x00" * 4 \
        + (255).to_bytes(3, "little") + (127).to_bytes(3, "little")
    assert hg._image_info(vp8x) == ("image/webp", 256, 128)
    assert hg._image_info(b"<html><script>alert(1)</script>") is None
    assert hg._image_info(b"GIF89a....") is None


def test_a_player_uploads_a_picture_and_the_club_sees_it(cast):
    host, guest, stranger = cast["host"], cast["guest"], cast["stranger"]
    r = guest.post("/games/api/me/avatar", json={"data_url": _data_url(_png(64, 64))})
    assert r.status_code == 200, r.text
    url = r.json()["avatar"]
    assert url.startswith(f"/games/api/avatars/{cast['ids']['guest']}?v=")
    assert guest.get("/games/api/tables").json()["my_avatar"] == url
    # at a table: on the seat, in the chat, on the club page
    gid = host.post("/games/api/tables", json={"name": "pics", "bb_cents": 100, "ante_cents": 300,
                                               "default_buyin_cents": 4000}).json()["id"]
    assert guest.post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000}).status_code == 200
    assert guest.post(f"/games/api/tables/{gid}/chat", json={"text": "hi"}).status_code == 200
    view = host.get(f"/games/api/tables/{gid}").json()
    assert view["seats"][1]["avatar"] == url and view["seats"][0]["avatar"] is None
    assert view["chat"][-1]["avatar"] == url
    club = host.get("/games/api/community").json()
    assert all("avatar" in p for p in club["players"])
    # the picture itself: for club members, with safe headers; hidden from outsiders
    img = host.get(url)
    assert img.status_code == 200 and img.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert img.headers["content-type"] == "image/png"
    assert img.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in img.headers["content-security-policy"]
    assert stranger.get(url).status_code == 404, "not in the club"
    assert guest.get(url).status_code == 200, "your own"
    # a new picture = a new URL; removing it clears it everywhere
    url2 = guest.post("/games/api/me/avatar", json={"data_url": _data_url(_png(80, 80))}).json()["avatar"]
    assert url2 != url
    assert guest.delete("/games/api/me/avatar").json() == {"avatar": None}
    assert host.get(f"/games/api/tables/{gid}").json()["seats"][1]["avatar"] is None
    assert host.get(url2).status_code == 404


@pytest.mark.parametrize("data_url", [
    "not a data url",
    "data:image/png;base64,@@@@",
    _data_url(b"<svg onload=alert(1)>", "image/png"),        # not a PNG at all
    _data_url(_png(20, 20), "image/jpeg"),                    # says JPEG, is a PNG
    _data_url(_png(8, 8)),                                    # too small
    _data_url(_png(1500, 16)),                                # too wide
    "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode(),
])
def test_anything_but_a_small_png_jpeg_or_webp_is_refused(cast, data_url):
    r = cast["guest"].post("/games/api/me/avatar", json={"data_url": data_url})
    assert r.status_code == 400, (data_url[:40], r.text)


def test_a_huge_upload_is_refused_before_decoding(cast, hg):
    big = "data:image/png;base64," + "A" * (hg.AVATAR_MAX_BYTES * 2)
    assert cast["guest"].post("/games/api/me/avatar", json={"data_url": big}).status_code == 400
