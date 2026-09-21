"""Public build — free while the models are in development (2026-09-22).

`PLO5BP_FREE_FOR_ALL` (default ON in production) makes every signed-in user
entitled: no daily trainer quota, Study unlocked, checkout closed. The rest of
the test session runs with it OFF (tests/python/conftest.py) so the paywall code
stays covered; this module boots the app the way production runs it.
"""

from __future__ import annotations

import sys

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_FREE_FOR_ALL="1", PLO5BP_FREE_HANDS="2")


@pytest.fixture(scope="module")
def pub(server):
    return sys.modules["plo5bp.ui.public"]


def _login(server, email):
    c = TestClient(server.app, raise_server_exceptions=False)
    assert c.get("/auth/dev", params={"email": email}).status_code == 200
    return c


def test_everyone_signed_in_is_entitled(server, pub):
    assert pub.FREE_FOR_ALL is True
    c = _login(server, "anyone@example.com")
    me = c.get("/me").json()
    assert me["signed_in"] is True and me["free_for_all"] is True
    assert me["sub"]["active"] is True and me["is_admin"] is False


def test_no_daily_quota_on_the_trainer(server):
    c = _login(server, "grinder@example.com")
    for _ in range(5):  # PLO5BP_FREE_HANDS=2 would have stopped this at the 3rd hand
        r = c.post("/trainer/new_hand", json={})
        assert r.status_code == 200, r.text


def test_study_is_unlocked(server):
    c = _login(server, "student@example.com")
    assert c.get("/state").status_code == 200
    assert c.post("/reset", json={}).status_code == 200
    assert c.get("/formats").status_code == 200


def test_checkout_is_closed_while_the_site_is_free(server):
    c = _login(server, "wallet@example.com")
    r = c.post("/billing/checkout")
    assert r.status_code == 409 and "free" in r.json()["detail"].lower()


def test_signed_out_visitors_still_have_to_sign_in(server):
    anon = TestClient(server.app, raise_server_exceptions=False)
    assert anon.get("/me").json()["signed_in"] is False
    assert anon.post("/trainer/new_hand", json={}).status_code in (401, 403)
    assert anon.get("/state").status_code in (401, 403)


def test_landing_page_says_it_is_free_and_in_development(server):
    anon = TestClient(server.app, raise_server_exceptions=False)
    html = anon.get("/").text.lower()
    assert "in development" in html
    assert "free" in html
