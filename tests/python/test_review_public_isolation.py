"""Review 2026-09-20 J3: re-importing the public app inside one pytest process.

`test_homegame.py` + `test_public_service.py` passed alone and failed together
(7 failures): both purge `sys.modules` and reimport `plo5bp.ui.server`, but
`from plo5bp.ui import public` resolves through the PACKAGE ATTRIBUTE first, so
the second import got a fresh `server` wired to the STALE `public` (old DB, old
FREE_HANDS). `conftest.purge_ui_modules` also deletes the package attributes.
"""

from __future__ import annotations

import importlib
import sys

from starlette.testclient import TestClient

_ENV = {
    "PLO5BP_PUBLIC": "1",
    "PLO5BP_DEV_LOGIN": "1",
    "PLO5BP_DEV_LOGIN_TESTCLIENT": "1",
    "PLO5BP_BASE_URL": "http://127.0.0.1:8770",
}


def test_purge_drops_the_package_attribute(ui_purge):
    import plo5bp.ui as pkg

    first = importlib.import_module("plo5bp.ui.runout")
    assert pkg.runout is first
    ui_purge(("plo5bp.ui.runout",))
    assert "plo5bp.ui.runout" not in sys.modules
    assert "runout" not in vars(pkg)  # popping sys.modules alone leaves this behind
    from plo5bp.ui import runout as second

    assert second is not first and sys.modules["plo5bp.ui.runout"] is second


def test_second_import_in_one_process_sees_the_new_env(ui_purge, monkeypatch, tmp_path):
    for k, v in _ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PLO5BP_TRAINER_STATS", str(tmp_path / "stats.json"))
    try:
        monkeypatch.setenv("PLO5BP_DB", str(tmp_path / "a.db"))
        monkeypatch.setenv("PLO5BP_FREE_HANDS", "5")
        ui_purge()
        importlib.import_module("plo5bp.ui.server")
        pub1 = sys.modules["plo5bp.ui.public"]
        assert pub1.FREE_HANDS_PER_DAY == 5

        monkeypatch.setenv("PLO5BP_DB", str(tmp_path / "b.db"))
        monkeypatch.setenv("PLO5BP_FREE_HANDS", "3")
        ui_purge()
        server2 = importlib.import_module("plo5bp.ui.server")
        from plo5bp.ui import homegame as hg2
        from plo5bp.ui import public as pub2

        # This is exactly what failed before: the package attribute kept
        # handing out `pub1`, and `plo5bp.ui.public` was absent from
        # sys.modules after the "fresh" import.
        assert pub2 is sys.modules["plo5bp.ui.public"] and pub2 is not pub1
        assert pub2.FREE_HANDS_PER_DAY == 3 and pub2.DB_PATH.name == "b.db"
        assert hg2.pub is pub2
        c = TestClient(server2.app)
        assert c.get("/auth/dev", params={"email": "j3@example.com"}).status_code == 200
        assert c.get("/me").json()["free"]["limit"] == 3
    finally:
        ui_purge()
