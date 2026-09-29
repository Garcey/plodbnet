"""(TEST-031) Skips are listed after every run and unexpected ones can fail it."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TESTS))

from skip_report import classify, load_policy  # noqa: E402


def test_policy_classifies_the_known_reasons():
    rules = load_policy()
    cases = {
        "could not import 'cv2': No module named 'cv2'": "optional package",
        "node is not installed": "tool",
        "this engine build has no explicit-deck deal": "engine build",
        "no golden frames in tests/ocr/fixtures/frames/ yet": "data not on disk",
        "hand ended before hero acted": "scenario not dealt",
    }
    for reason, cat in cases.items():
        assert classify(reason, rules) == cat, reason
    assert classify("TODO: flaky, look at it later", rules) is None


def _suite(tmp_path: Path) -> Path:
    (tmp_path / "conftest.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(TESTS)!r})\n"
        "from skip_report import SkipReport\n"
        "def pytest_configure(config):\n"
        "    config.pluginmanager.register(SkipReport(config), 'plo5-skip-report')\n",
        encoding="utf-8",
    )
    (tmp_path / "test_x.py").write_text(
        "import pytest\n"
        "def test_ok():\n    pass\n"
        "def test_env():\n    pytest.skip('node is not installed')\n"
        "def test_new():\n    pytest.skip('TODO: flaky, look at it later')\n",
        encoding="utf-8",
    )
    return tmp_path


def _run(suite: Path, strict: bool) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("CI", "PLO5BP_STRICT_SKIPS")}
    if strict:
        env["PLO5BP_STRICT_SKIPS"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(suite)],
        capture_output=True, text=True, env=env, cwd=str(suite), timeout=120,
    )


def test_quiet_runs_still_list_skips_and_flag_the_unexpected(tmp_path: Path):
    r = _run(_suite(tmp_path), strict=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2 skipped (tests/expected_skips.txt)" in r.stdout
    assert "[tool] node is not installed" in r.stdout
    assert "[UNEXPECTED] TODO: flaky" in r.stdout and "UNEXPECTED skip(s)" in r.stdout


def test_strict_mode_fails_on_an_unexpected_skip(tmp_path: Path):
    r = _run(_suite(tmp_path), strict=True)
    assert r.returncode == 1, r.stdout + r.stderr
