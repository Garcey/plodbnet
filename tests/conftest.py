"""Suite-wide pytest wiring (applies to tests/python and tests/ocr).

1. **Markers by file** — every test gets the marker of its area, so a subset runs
   without listing files:
       pytest -m "homegame or public or ui or ops"   the website, before a deploy
       pytest -m training                            engine / encoders / rollout / PPO
       pytest -m "not slow"                          a quick pass
   (`markers` and the rules live in pyproject.toml / below; a new test file is
   classified by its name, nothing to register.)
2. **Stale engine** — the Rust engine is a compiled module. When
   `rust_engine/src` changed after `_engine` was built, tests that probe for a
   feature (`hasattr(_RustGameState, "reset_with_deck")`) SKIP instead of failing
   and the run looks green. The header of every run says so loudly; with `CI` set
   the session stops. The engine carries a hash of its sources (`SOURCE_HASH`,
   baked in by rust_engine/build.rs); older builds are judged by file times.
3. **CI turns environment skips into failures** — "engine build has no …",
   "rebuild the extension", "node is not installed": on CI the engine is fresh and
   Node is installed, so such a skip is a real problem.
4. ``timeout`` (pytest-timeout, in the dev extra) is accepted even when the plugin
   is missing, instead of a warning on every run.
5. **Skips are listed** after every run, even with ``-q``, grouped by reason and
   checked against ``tests/expected_skips.txt`` (tests/skip_report.py); an
   unexpected one fails the run with ``--strict-skips`` / ``PLO5BP_STRICT_SKIPS=1``
   / on CI.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
ENGINE_SRC = REPO / "rust_engine"

# --- 1. markers by file ------------------------------------------------------------

_PREFIXES = {
    "homegame": ("test_homegame", "test_review_homegame"),
    "public": ("test_public", "test_review_public", "test_review_server_public", "test_touch_icon"),
    "cfr": ("test_cfr", "test_review_cfr", "test_gto", "test_review_gto", "test_teacher_floors",
            "test_iso_policy", "test_obs_from_label"),
    "ui": ("test_trainer", "test_review_trainer", "test_ui_", "test_review_server_study", "test_nlh_ui",
           "test_ranges", "test_hand_describe", "test_runout", "test_all_hole_cards"),
    "ops": ("test_deploy", "test_ops"),
}
#: Test files that run JavaScript under Node.js.
_NODE = re.compile(r"(_js|client_reconnect|client_ui|plo6|plo67|short_hands|test_ranges|trainer_review"
                   r"|pokernow_userscript)")
#: The slowest files — ≥ ~10 s each in the 2026-09-28 full run on the desktop
#: (`pytest --durations=40`); excluded by `-m "not slow"`. Re-measure now and then.
_SLOW = {
    "test_deploy_remote.py", "test_deploy_script.py", "test_exactness.py", "test_resume_seed.py",
    "test_review_homegame_fixes.py", "test_homegame_concurrency.py", "test_live_package.py",
}


def area_markers(path: Path) -> set[str]:
    """The markers a test file gets from its location and name."""
    try:
        rel = path.resolve().relative_to(TESTS)
    except ValueError:
        return set()
    name = rel.name
    out = set()
    if rel.parts[0] == "ocr":
        out.add("ocr")
    for marker, prefixes in _PREFIXES.items():
        if name.startswith(prefixes):
            out.add(marker)
    if rel.parts[0] == "python" and not out:
        out.add("training")  # the engine, encoders, envs, rollout, PPO, networks, probes
    if _NODE.search(path.stem):
        out.add("node")
    if name in _SLOW:
        out.add("slow")
    return out


def pytest_collection_modifyitems(config, items):
    cache: dict[Path, set[str]] = {}
    for item in items:
        p = Path(str(item.fspath))
        if p not in cache:
            cache[p] = area_markers(p)
        for m in cache[p]:
            item.add_marker(m)


# --- 2. stale engine -------------------------------------------------------------------

def engine_source_hash(root: Path = ENGINE_SRC) -> str:
    """FNV-1a 64 over rust_engine/src/** (+ the crate's Cargo.toml and the workspace
    Cargo.lock), path-sorted, CRLF normalised — the same bytes and order as
    rust_engine/build.rs, which bakes it into the module as ``SOURCE_HASH``."""
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    src = sorted(("src/" + p.relative_to(root / "src").as_posix(), p)
                 for p in (root / "src").rglob("*") if p.is_file())
    for rel, p in src + [("Cargo.toml", root / "Cargo.toml"), ("../Cargo.lock", root.parent / "Cargo.lock")]:
        if not p.is_file():
            continue
        for chunk in (rel.encode(), b"\0", p.read_bytes().replace(b"\r\n", b"\n"), b"\0"):
            for b in chunk:
                h = ((h ^ b) * prime) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def engine_staleness() -> str | None:
    """None when the compiled engine matches rust_engine/, else why not."""
    try:
        import plo5bp._engine as eng  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001 — no engine at all is reported by the tests themselves
        return f"the engine does not import ({type(e).__name__})"
    built = getattr(eng, "SOURCE_HASH", None)
    if not ENGINE_SRC.is_dir():
        return None
    if built:
        now = engine_source_hash()
        return None if now == built else f"built from sources {built}, the sources are now {now}"
    # A build from before SOURCE_HASH existed: compare file times instead.
    so = Path(eng.__file__)
    newest = max((p.stat().st_mtime, p) for p in (ENGINE_SRC / "src").rglob("*.rs"))
    if newest[0] > so.stat().st_mtime + 1:
        return f"{newest[1].relative_to(REPO).as_posix()} changed after {so.name} was built"
    return None


def pytest_configure(config):
    config._plo5_engine_stale = engine_staleness()
    # 5. skipped tests are listed after every run (TEST-031, tests/skip_report.py).
    # Loaded by PATH: a pytest run started from another rootdir / cwd (e.g. a
    # subprocess suite) does not have tests/ on sys.path.
    import importlib.util
    import sys

    plugin_file = TESTS / "skip_report.py"
    if not plugin_file.is_file():  # a copied / partial suite: no skip report
        return
    mod = sys.modules.get("plo5_skip_report")
    if mod is None:
        spec = importlib.util.spec_from_file_location("plo5_skip_report", plugin_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plo5_skip_report"] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not config.pluginmanager.has_plugin("plo5-skip-report"):
        config.pluginmanager.register(mod.SkipReport(config), "plo5-skip-report")


def pytest_report_header(config):
    why = getattr(config, "_plo5_engine_stale", None)
    if why is None:
        return None
    return [f"!!! STALE ENGINE: {why}. Engine tests may SKIP instead of fail.",
            "!!! Rebuild: .venv/Scripts/maturin develop --release   (from the repo root)"]


def pytest_sessionstart(session):
    why = getattr(session.config, "_plo5_engine_stale", None)
    if os.environ.get("CI") and why:
        pytest.exit(f"STALE ENGINE on CI: {why}", returncode=1)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    why = getattr(config, "_plo5_engine_stale", None)
    if why:
        terminalreporter.write_line(f"!!! STALE ENGINE: {why} - rebuild before trusting the skips.", red=True)


# --- 3. CI: environment skips are failures -----------------------------------------------

_ENV_SKIP = re.compile(r"engine build|rebuild (the )?extension|maturin develop|node is not installed", re.I)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if not os.environ.get("CI"):
        return
    rep = outcome.get_result()
    if rep.skipped and isinstance(rep.longrepr, tuple) and _ENV_SKIP.search(str(rep.longrepr[2])):
        rep.outcome = "failed"
        rep.longrepr = f"skipped on CI, which must have a fresh engine and Node: {rep.longrepr[2]}"


# --- 4. `timeout` without pytest-timeout ---------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--strict-skips", action="store_true", default=False,
                     help="fail the run on a skip not listed in tests/expected_skips.txt (TEST-031)")
    try:
        import pytest_timeout  # noqa: F401
    except ImportError:
        parser.addini("timeout", "per-test timeout (pytest-timeout; not installed here)", default=None)
