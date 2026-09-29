"""Several apps in one process (BE-007) — and the purge helper that is left.

Review 2026-09-20 J3: `test_homegame.py` + `test_public_service.py` passed alone
and failed together (7 failures): each purged `sys.modules` and re-imported
`plo5bp.ui.server` to get an app for ITS environment, and `from plo5bp.ui import
public` then resolved the STALE module through the package attribute. Since
BE-007 nothing is re-imported for that: `server.create_app()` builds an app from
the environment as it is when it is called, with its own database, caches and
home-games context, and makes it the current site. Pinned here: two public apps
built one after the other each see their own settings and database, the first
one retired (closed) by the second; the local site comes back; a retired or
closed app cannot be made current again; the module's names follow the current
site; importing `plo5bp.ui.public` opens no database. `conftest.purge_ui_modules`
remains for a test that genuinely needs a module's import re-executed.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

_ENV = {
    "PLO5BP_PUBLIC": "1",
    "PLO5BP_DEV_LOGIN": "1",
    "PLO5BP_DEV_LOGIN_TESTCLIENT": "1",
    "PLO5BP_BASE_URL": "http://127.0.0.1:8770",
}


@pytest.fixture()
def server():
    """The server module, with the current site put back after the test."""
    mod = importlib.import_module("plo5bp.ui.server")
    before = mod.current_site()
    try:
        yield mod
    finally:
        now = mod.current_site()
        if now is not before and now.public:
            now.close()
        mod.use_site(before)


def test_purge_drops_the_package_attribute(ui_purge):
    import plo5bp.ui as pkg

    first = importlib.import_module("plo5bp.ui.runout")
    assert pkg.runout is first
    ui_purge(("plo5bp.ui.runout",))
    assert "plo5bp.ui.runout" not in sys.modules
    assert "runout" not in vars(pkg)  # popping sys.modules alone leaves this behind
    from plo5bp.ui import runout as second

    assert second is not first and sys.modules["plo5bp.ui.runout"] is second


def test_two_public_apps_in_one_process_each_see_their_own_environment(server, monkeypatch, tmp_path):
    local = server.current_site()
    assert not local.public
    for k, v in _ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PLO5BP_TRAINER_STATS", str(tmp_path / "stats.json"))
    monkeypatch.setenv("PLO5BP_DB", str(tmp_path / "a.db"))
    monkeypatch.setenv("PLO5BP_FREE_HANDS", "5")
    monkeypatch.setenv("PLO5BP_HOMEGAME_MAX_TABLES", "7")
    first = server.create_app()
    pub = sys.modules["plo5bp.ui.public"]
    hg = sys.modules["plo5bp.ui.homegame"]
    assert server.app is first and server.PLO5BP_PUBLIC is True
    assert pub.FREE_HANDS_PER_DAY == 5 and pub.DB.path.name == "a.db"
    assert first.state.site.db is pub.DB and hg.CTX is first.state.site.homegames
    assert hg.MAX_OPEN_TABLES_PER_USER == 7
    hg.AVATAR_RATE.allow(1)  # a request budget spent in the first app (user ids are its database's)
    c1 = TestClient(first)
    assert c1.get("/auth/dev", params={"email": "one@example.com"}).status_code == 200
    assert c1.get("/me").json()["free"]["limit"] == 5

    monkeypatch.setenv("PLO5BP_DB", str(tmp_path / "b.db"))
    monkeypatch.setenv("PLO5BP_FREE_HANDS", "3")
    monkeypatch.setenv("PLO5BP_HOMEGAME_MAX_TABLES", "4")
    second = server.create_app()
    assert hg.MAX_OPEN_TABLES_PER_USER == 4 and len(hg.AVATAR_RATE) == 0
    # The same modules — nothing re-imported — now serving the second app.
    assert sys.modules["plo5bp.ui.public"] is pub and sys.modules["plo5bp.ui.homegame"] is hg
    assert hg.pub is pub
    assert pub.FREE_HANDS_PER_DAY == 3 and pub.DB.path.name == "b.db"
    assert hg.CTX is second.state.site.homegames is not first.state.site.homegames
    # The first app was retired: its home games stopped, its database closed.
    assert first.state.site.closed and first.state.site.homegames.watchdog_stop.is_set()
    c2 = TestClient(second)
    assert c2.get("/auth/dev", params={"email": "two@example.com"}).status_code == 200
    assert c2.get("/me").json()["free"]["limit"] == 3
    # A database of its own: the first app's user is not in it.
    assert {r["email"] for r in pub.DB.q("SELECT email FROM users")} == {"two@example.com"}
    with pytest.raises(RuntimeError, match="closed|retired"):
        server.use_site(first)

    server.use_site(local)
    assert server.app is local.app and server.PLO5BP_PUBLIC is False
    health = TestClient(local.app).get("/health").json()
    assert health["public"] is False and "home_games" not in health["threads"]
    assert TestClient(local.app).get("/state").status_code == 200
    second.state.site.close()


def test_the_module_names_follow_the_current_site(server, monkeypatch):
    monkeypatch.delenv("PLO5BP_PUBLIC", raising=False)
    first = server.current_site()
    other = server.create_app()
    site = other.state.site
    assert server.current_site() is site and site is not first
    assert server.app is other and server.FORMATS is site.formats
    assert server.MODEL is site.formats["plo5_double_bomb"]["model"]
    assert server.MODEL_DEVICE == site.device and server.trainer_router is site.trainer_router
    assert server._DEFAULT_SESSION is site.default_session is not first.default_session
    # A swapped model is what the names read (no alias to refresh).
    entry = dict(site.formats["plo5_double_bomb"])
    server._on_model_swap("plo5_double_bomb", entry)
    assert server.FORMATS["plo5_double_bomb"] is entry and server.MODEL is entry["model"]
    # The settable names write the site; the others refuse.
    host = object()
    monkeypatch.setattr(server, "GTO_HOST", host)
    assert site.gto_host is host and first.gto_host is not host
    with pytest.raises(AttributeError, match="read-only"):
        server.MODEL = None
    server.use_site(first)
    assert server.app is first.app and server.GTO_HOST is first.gto_host


def test_site_settings_are_one_parse_of_the_environment(server):
    parsed = server.SiteSettings.from_env({
        "PLO5BP_PUBLIC": " Yes ", "PLO5BP_BASE_URL": "https://wrapgto.example/ ",
        "PLO5BP_GTO_CHECKPOINT": "  ", "PLO5BP_TORCH_THREADS": "0",
    })
    assert parsed == server.SiteSettings(
        public=True, gto_checkpoint=None, base_url="https://wrapgto.example", torch_threads=1,
    )
    assert server.SiteSettings.from_env({}) == server.SiteSettings()


def test_create_app_refuses_settings_that_disagree_with_the_environment(server, monkeypatch):
    monkeypatch.delenv("PLO5BP_PUBLIC", raising=False)
    before = server.current_site()
    with pytest.raises(ValueError, match="PLO5BP_PUBLIC"):
        server.create_app(server.SiteSettings(public=True))
    assert server.current_site() is before


def test_importing_the_public_module_opens_no_database(tmp_path):
    db = tmp_path / "never.db"
    env = {**os.environ, "PLO5BP_DB": str(db)}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(Path(__file__).resolve().parents[3] / "python"), env.get("PYTHONPATH", "")) if p
    )
    code = (
        "import plo5bp.ui.public as p, plo5bp.ui.homegame as hg\n"
        "try:\n"
        "    p.DB.q('SELECT 1')\n"
        "except AttributeError as e:\n"
        "    print('not open:', 'create_app' in str(e))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=300, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "not open: True"
    assert not db.exists()
