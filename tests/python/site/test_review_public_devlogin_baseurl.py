"""Review 2026-09-20 F4: the dev login is never mounted on a deployment whose
own BASE_URL is public, whatever PLO5BP_DEV_LOGIN says. (Own module: it needs
its own app boot, and `boot_public_server` allows one per module.)"""

from __future__ import annotations

import sys

from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"


def test_dev_login_route_absent_when_base_url_is_public(boot_public_server):
    """The env flag alone must never mount it on a tunnel/prod deployment."""
    srv = boot_public_server(PLO5BP_BASE_URL="https://wrapgto.example")
    pub = sys.modules["plo5bp.ui.public"]
    assert pub.DEV_LOGIN_REQUESTED is True and pub.DEV_LOGIN is False
    c = TestClient(srv.app, client=("127.0.0.1", 5555))
    r = c.get("/auth/dev", params={"email": ADMIN_EMAIL}, headers={"host": "127.0.0.1"})
    assert r.status_code == 404
    assert c.get("/me").json()["dev_login"] is False
