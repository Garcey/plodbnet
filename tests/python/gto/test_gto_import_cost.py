"""(TOOL-020) The CFR API and the desktop app import without torch.

``import plo5bp.gto.cfr_api`` used to take ~2 s and load torch: the ``plo5bp``
and ``plo5bp.gto`` packages imported the encoder / PolicyNet eagerly. The desktop
app paid it at start (against a readiness timeout) and every solve's spawn child
paid it again. Checked in a fresh interpreter so other tests' imports can't hide
a regression.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

_TORCH_FREE = [
    "plo5bp.gto.cfr_api",
    "plo5bp.gto.cfr_batch",
    "plo5bp.gto.teacher",
    "plo5bp.gto.roots",
    "plo5bp.gto.iso",
    "plo5bp.gto.preflop_class",
    "plo5bp.cfr_app.solve_worker",
    "plo5bp.cfr_app.session",
    "plo5bp.cfr_app.strategy_view",
    "plo5bp.cfr_app.tree_model",
    "plo5bp.cfr_app.ranges",
]


def _run(code: str) -> subprocess.CompletedProcess:
    env_path = str(REPO / "python")
    return subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {env_path!r}); " + code],
        capture_output=True, text=True, timeout=120,
    )


def test_cfr_modules_import_without_torch():
    code = (
        "import importlib\n"
        f"for m in {_TORCH_FREE!r}: importlib.import_module(m)\n"
        "heavy = sorted(k for k in ('torch', 'numpy') if k in sys.modules)\n"
        "print('HEAVY=' + ','.join(heavy))\n"
    )
    r = _run(code)
    assert r.returncode == 0, r.stderr
    assert "HEAVY=\n" in r.stdout or r.stdout.strip() == "HEAVY=", r.stdout


def test_cfr_app_server_imports_without_torch():
    pytest.importorskip("fastapi")
    r = _run("import plo5bp.cfr_app.server\nprint('TORCH=' + str('torch' in sys.modules))")
    assert r.returncode == 0, r.stderr
    assert "TORCH=False" in r.stdout, r.stdout


def test_package_level_names_still_resolve():
    import plo5bp
    import plo5bp.gto as gto
    from plo5bp.config import GameConfig
    from plo5bp.gto.cfr_api import solve
    from plo5bp.gto.policy_host import PolicyNetHost

    assert plo5bp.GameConfig is GameConfig
    from plo5bp import OBS_DIM, BombPotEnv  # noqa: F401 — the old eager names
    from plo5bp.encoding import OBS_DIM as REAL_OBS_DIM

    assert OBS_DIM == REAL_OBS_DIM
    assert gto.solve is solve and gto.PolicyNetHost is PolicyNetHost
    assert set(gto.__all__) <= set(dir(gto))
    with pytest.raises(AttributeError):
        gto.no_such_name  # noqa: B018
    with pytest.raises(AttributeError):
        plo5bp.no_such_name  # noqa: B018
    from plo5bp import _engine  # a real submodule still imports through the package

    assert hasattr(_engine, "cfr_solve")


def test_roots_anchor_spec_name_matches_sizing():
    from plo5bp.gto.roots import CLUBGG_NLH_ROOT, NLH_ANCHOR_SPEC_NAME
    from plo5bp.sizing import NLH_ANCHOR_SPEC

    assert NLH_ANCHOR_SPEC_NAME == NLH_ANCHOR_SPEC.name
    assert CLUBGG_NLH_ROOT.anchor_spec_name == NLH_ANCHOR_SPEC.name
