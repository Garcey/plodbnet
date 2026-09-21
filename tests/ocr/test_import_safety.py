"""`plo5bp.ocr` must import — and `tests/ocr` must collect — without OpenCV.

review 2026-09-20 I12 + J2:

* I12 — `plo5bp/ocr/__init__.py` eagerly imported `extract` → `cv2`, so the
  PokerNow DOM path (`pokernow`), the reconstructor (`events`) and even the
  data contracts (`types`) needed OpenCV just to import.
* J2 — `tests/ocr/conftest.py` ran `pytest.importorskip("cv2")` at module
  level. A `Skipped` raised while importing a *conftest* is a collection error,
  so `pytest tests/python/ tests/ocr/` ran ZERO tests on a machine without the
  `[ocr]` extras.

Both checks run in a subprocess with the OCR extras BLOCKED by a meta-path
finder, so they prove the property even on a machine that has cv2 installed
(where a plain import test would pass vacuously).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import plo5bp

REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_PARENT = str(Path(plo5bp.__file__).resolve().parents[1])

# Pretend the `[ocr]` extras are not installed.
_BLOCK_OCR_EXTRAS = """
import importlib.abc, sys

class _BlockOcrExtras(importlib.abc.MetaPathFinder):
    BLOCKED = {"cv2", "pytesseract", "mss", "pygetwindow", "windows_capture"}

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r} (blocked)", name=name)
        return None

for _m in list(sys.modules):
    if _m.split(".")[0] in _BlockOcrExtras.BLOCKED:
        del sys.modules[_m]
sys.meta_path.insert(0, _BlockOcrExtras())
"""


def _run(code: str, *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (_PKG_PARENT, env.get("PYTHONPATH", "")) if p
    )
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_OCR_EXTRAS + code],
        capture_output=True, text=True, timeout=300, env=env, cwd=cwd,
    )


def test_pure_python_ocr_modules_import_without_opencv():
    proc = _run(
        """
import plo5bp.ocr
import plo5bp.ocr.types, plo5bp.ocr.events, plo5bp.ocr.pokernow
import plo5bp.ocr.text, plo5bp.ocr.rois, plo5bp.ocr.live
from plo5bp.ocr import Card, FrameState, SeatObs
from plo5bp.ocr.events import EventReconstructor, EngineView, cents_to_engine_chips
from plo5bp.ocr.pokernow import map_payload, PokerNowPayloadError
from plo5bp.ocr.text import _parse_chip_text

assert _parse_chip_text("1,755.59.") == 175559
assert cents_to_engine_chips(18000, 5.0) == 90000
leaked = sorted(m for m in sys.modules if m.split(".")[0] in _BlockOcrExtras.BLOCKED)
assert not leaked, leaked

# The public name still exists; it just needs OpenCV when actually asked for.
assert "extract_frame_state" in plo5bp.ocr.__all__
assert "extract_frame_state" in dir(plo5bp.ocr)
try:
    plo5bp.ocr.extract_frame_state
except ModuleNotFoundError as e:
    assert e.name == "cv2", e
else:
    raise SystemExit("extract_frame_state resolved with cv2 blocked?!")
try:
    plo5bp.ocr.no_such_name
except AttributeError:
    pass
else:
    raise SystemExit("unknown attribute did not raise AttributeError")
print("IMPORT-OK")
"""
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "IMPORT-OK" in proc.stdout


def test_ocr_test_dir_collects_and_runs_without_opencv():
    """The J2 regression itself: with cv2 absent, a session that includes
    `tests/ocr` must still RUN the pure-Python tests and merely SKIP the
    pixel modules. Pre-fix this exited with a collection error and
    "no tests ran"."""
    proc = _run(
        """
import pytest
raise SystemExit(pytest.main([
    "tests/ocr/test_rois.py",          # pure Python: must run
    "tests/ocr/test_text_parser.py",   # pure Python: must run
    "tests/ocr/test_cards.py",         # pixel module: must be skipped, not error
    "tests/ocr/test_extract.py",
    "-q", "-p", "no:cacheprovider", "-rs",
]))
""",
        cwd=REPO_ROOT,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert " passed" in out, out
    assert "skipped" in out, out
    assert "error" not in out.lower().replace("no:cacheprovider", ""), out
