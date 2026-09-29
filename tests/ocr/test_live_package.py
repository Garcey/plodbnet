"""The live-capture split (BE-001 / TOOL-034).

ClubGG OCR + PokerNow ingest live in `plo5bp.ui.live`, mounted by the local
build only. Pinned here:

* the PUBLIC build never imports the package (nor the local-only Ranges
  router), so none of that code ships there, and its study `Session` carries
  no live state;
* the LOCAL build mounts `/ocr/*` + `/pokernow/*` and exposes the live toggle
  in the study state;
* importing a live submodule FIRST (before the server) works — the package
  imports the study core before anything else (see `plo5bp/ui/live/__init__`).

Each build is checked in a fresh interpreter: the flag is read at import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

_PKG_PARENT = str(Path(__file__).resolve().parents[2] / "python")


def _run(code: str, **env_overrides: str) -> dict:
    env = dict(os.environ)
    env.pop("PLO5BP_PUBLIC", None)
    env.update(env_overrides)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (_PKG_PARENT, env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=600, env=env,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


_PROBE = r"""
import json, sys
from starlette.testclient import TestClient
{first_import}
import plo5bp.ui.server as s


def _state_keys(s):
    own = s.Session()
    prev = s._SESSION_RESOLVER
    s.set_session_resolver(lambda: own)
    try:
        return s._state_dict()
    finally:
        s.set_session_resolver(prev)


c = TestClient(s.app)
out = {{
    "live_modules": sorted(m for m in sys.modules if m.startswith("plo5bp.ui.live")),
    "ranges_imported": "plo5bp.ui.ranges" in sys.modules,
    "session_live": s.Session().live is None,
    # A user's own session, as the public build resolves one per sign-in
    # (it refuses the shared default session outside a signed-in request).
    "state_keys": sorted(_state_keys(s)),
    "ocr_status": c.get("/ocr/status").status_code,
    "pn_status": c.get("/pokernow/status").status_code,
}}
print(json.dumps(out))
"""


def test_public_build_never_imports_live_capture(tmp_path):
    # A throwaway database: the public build opens (and migrates) one, and the
    # real data/public.db may be in use by a running server.
    out = _run(
        _PROBE.format(first_import=""), PLO5BP_PUBLIC="1",
        PLO5BP_DB=str(tmp_path / "public.db"), PLO5BP_HOMEGAME_GRADING="0",
    )
    assert out["live_modules"] == []
    assert out["ranges_imported"] is False
    assert out["session_live"] is True
    assert "simple_ocr_mode" not in out["state_keys"]
    # Signed out, the service layer answers first; either way no live route.
    assert out["ocr_status"] in (401, 404) and out["pn_status"] in (401, 404)


def test_local_build_mounts_live_capture():
    out = _run(_PROBE.format(first_import=""))
    assert "plo5bp.ui.live.routes" in out["live_modules"]
    assert out["ranges_imported"] is True
    assert "simple_ocr_mode" in out["state_keys"]
    assert out["ocr_status"] == 200 and out["pn_status"] == 200


def test_importing_a_live_submodule_first_works():
    out = _run(_PROBE.format(first_import="import plo5bp.ui.live.tracking"))
    assert "plo5bp.ui.live.tracking" in out["live_modules"]
    assert out["ocr_status"] == 200 and out["pn_status"] == 200


def test_action_tracking_toggle_says_what_it_does():
    """TOOL-009: "Simple: On" hid that no action was tracked. The toggle now
    names the tracking itself (off by default = simple mode) and the status
    line says who enters the actions."""
    static = Path(__file__).resolve().parents[2] / "python" / "plo5bp" / "ui" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "app.js").read_text(encoding="utf-8")
    assert ">Track actions: Off</button>" in html and "Simple: On" not in html
    assert '"Track actions: On" : "Track actions: Off"' in js
    assert '"you enter actions · "' in js
