"""Repo-level guards: the production env reference, the test-suite wiring
(tests/conftest.py), .gitignore's secret patterns, the pre-commit secret hook and
the shell scripts' syntax."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bash_tools import BASH, posix as _posix

REPO = Path(__file__).resolve().parents[3]
GIT = shutil.which("git")

# Read only by the local build, training or the test suite — never by the website.
_NOT_SERVER = {
    "PLO5BP_ANNEAL_CONTROL", "PLO5BP_NUMPY_FLUSH", "PLO5BP_PIN_ROLLOUT", "PLO5BP_ROLLOUT_OVERLAP",
    "PLO5BP_STEP_TIMERS", "PLO5BP_STEP_TIMERS_OWNED", "PLO5BP_NO_FULL_PACKED", "PLO5_RUST_ENCODER",
    "PLO5BP_OCR_DEBUG_HANDSTART", "PLO5BP_OCR_DEBUG_TIMER", "PLO5BP_PN_DEBUG", "PLO5BP_LIVE_RECORD",
}


def test_env_example_lists_every_variable_the_website_reads():
    """ops/env.example is the production env reference (ops/SERVER_SETUP.md):
    a variable the web app reads but the reference omits is a setting nobody can
    find when rebuilding the server."""
    sources = list((REPO / "python" / "plo5bp" / "ui").rglob("*.py"))
    sources += [REPO / "python" / "plo5bp" / f for f in ("encoding.py", "encoding_nlh.py", "network.py")]
    names = set()
    for f in sources:
        if f.exists():
            names |= set(re.findall(r"[\"'](PLO5BP_[A-Z0-9_]+|STRIPE_[A-Z_]+|GOOGLE_[A-Z_]+)[\"']",
                                    f.read_text(encoding="utf-8", errors="replace")))
    documented = set(re.findall(r"\b([A-Z][A-Z0-9_]{3,})=", (REPO / "ops" / "env.example").read_text(encoding="utf-8")))
    missing = sorted(names - documented - _NOT_SERVER)
    assert not missing, f"add these to ops/env.example (name + one-line meaning): {missing}"


# --- tests/conftest.py ---------------------------------------------------------------

def _conftest():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tests_root_conftest", REPO / "tests" / "conftest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_area_markers_by_file():
    c = _conftest()
    t = REPO / "tests"
    assert c.area_markers(t / "python" / "test_homegame_plo67.py") == {"homegame", "node"}
    assert c.area_markers(t / "python" / "test_review_public_billing.py") == {"public"}
    assert c.area_markers(t / "python" / "test_trainer_scoring.py") == {"ui"}
    assert c.area_markers(t / "python" / "test_gto_policy.py") == {"cfr"}
    assert c.area_markers(t / "python" / "test_rollout_parity.py") == {"training"}
    assert c.area_markers(t / "python" / "test_compact_obs.py") == {"training"}
    assert c.area_markers(t / "python" / "test_exactness.py") == {"training", "slow"}
    assert c.area_markers(t / "python" / "test_deploy_remote.py") == {"ops", "slow"}
    assert c.area_markers(t / "ocr" / "test_events.py") == {"ocr"}
    assert c.area_markers(t / "ocr" / "test_pokernow_userscript.py") == {"ocr", "node"}


def test_engine_source_hash_is_deterministic_and_line_ending_blind(tmp_path):
    c = _conftest()
    crate = tmp_path / "rust_engine"
    (crate / "src" / "sub").mkdir(parents=True)
    (crate / "src" / "lib.rs").write_bytes(b"fn a() {}\nfn b() {}\n")
    (crate / "src" / "sub" / "m.rs").write_bytes(b"// m\n")
    (crate / "Cargo.toml").write_bytes(b"[package]\n")
    (tmp_path / "Cargo.lock").write_bytes(b"# lock\n")
    h1 = c.engine_source_hash(crate)
    (crate / "src" / "lib.rs").write_bytes(b"fn a() {}\r\nfn b() {}\r\n")   # a Windows checkout
    assert c.engine_source_hash(crate) == h1
    (crate / "src" / "sub" / "m.rs").write_bytes(b"// changed\n")
    assert c.engine_source_hash(crate) != h1
    assert re.fullmatch(r"[0-9a-f]{16}", h1)


def test_engine_source_hash_matches_the_build(tmp_path):
    """build.rs and tests/conftest.py must agree, or every run reports a stale engine."""
    import plo5bp._engine as eng
    built = getattr(eng, "SOURCE_HASH", None)
    if built is None:
        pytest.skip("this engine was built before SOURCE_HASH existed")
    c = _conftest()
    if c.engine_staleness() is not None:
        pytest.skip("the engine is older than its sources (rebuild to compare)")
    assert built == c.engine_source_hash()


def _run_pytest(tmp_path: Path, body: str, **env) -> subprocess.CompletedProcess:
    d = tmp_path / "suite"
    (d / "tests" / "python").mkdir(parents=True)
    shutil.copy(REPO / "tests" / "conftest.py", d / "tests" / "conftest.py")
    (d / "tests" / "python" / "test_probe.py").write_text(body)
    e = {k: v for k, v in os.environ.items() if k not in ("CI", "PYTEST_ADDOPTS")}
    e.update(env)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(d / "tests")],
                          cwd=d, env=e, capture_output=True, text=True, timeout=300)


def test_ci_turns_environment_skips_into_failures(tmp_path):
    body = ("import pytest\n"
            "def test_node():\n    pytest.skip('node is not installed')\n"
            "def test_engine():\n    pytest.skip('this engine build has no explicit-deck deal')\n"
            "def test_other():\n    pytest.skip('no GPU here')\n")
    local = _run_pytest(tmp_path / "a", body)
    assert local.returncode == 0 and "3 skipped" in local.stdout, local.stdout
    ci = _run_pytest(tmp_path / "b", body, CI="true")
    assert ci.returncode == 1, ci.stdout
    assert "2 failed" in ci.stdout and "1 skipped" in ci.stdout


# --- .gitignore and the secret hook ------------------------------------------------------

@pytest.mark.skipif(GIT is None, reason="needs git")
@pytest.mark.parametrize("path,ignored", [
    (".env", True), (".env.prod", True), (".env.public", True), ("client_secret_1.json", True),
    ("x.key", True), ("x.p12", True), ("x.pfx", True), ("a/public.db", True), ("public.db-wal", True),
    ("x.sqlite3", True), ("node_modules/x.js", True), (".grok/x", True), ("ssh/id_ed25519", True),
    ("ops/env.example", False), ("requirements/server.txt", False), ("python/plo5bp/x.py", False),
])
def test_gitignore_covers_secret_files(path, ignored):
    r = subprocess.run([GIT, "-C", str(REPO), "check-ignore", "-q", "--no-index", path])
    assert (r.returncode == 0) is ignored, path


@pytest.mark.skipif(GIT is None or BASH is None, reason="needs git and bash")
def test_pre_commit_hook_refuses_secrets(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    g = [GIT, "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(g + ["init", "-q"], check=True)
    hook = repo / "pre-commit"
    hook.write_bytes((REPO / "scripts" / "hooks" / "pre-commit").read_bytes().replace(b"\r\n", b"\n"))
    env = {**os.environ, "PATH": os.environ.get("PATH", "")}
    # Never the real gitleaks here: exercise the built-in fallback.
    env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if "gitleaks" not in p.lower())

    def hook_rc() -> int:
        return subprocess.run([BASH, _posix(hook)], cwd=repo, env=env, capture_output=True, text=True).returncode

    (repo / "ok.py").write_text("x = 1\n")
    subprocess.run(g + ["add", "ok.py"], check=True)
    assert hook_rc() == 0
    (repo / "keys.py").write_text('K = "sk_live_' + "Zq9Xw8Vu7Ts6Rq5Po4Nm" + '"\n')
    subprocess.run(g + ["add", "keys.py"], check=True)
    assert hook_rc() == 1
    subprocess.run(g + ["rm", "-q", "--cached", "keys.py"], check=True)
    (repo / ".env").write_text("A=1\n")
    subprocess.run(g + ["add", "-f", ".env"], check=True)
    assert hook_rc() == 1


# --- shell scripts ---------------------------------------------------------------------------

@pytest.mark.skipif(GIT is None or BASH is None, reason="needs git and bash")
def test_every_shell_script_parses():
    files = subprocess.run([GIT, "-C", str(REPO), "ls-files", "*.sh"], capture_output=True, text=True).stdout.split()
    files += [p.relative_to(REPO).as_posix() for d in ("ops/bin", "scripts/hooks") for p in (REPO / d).iterdir()]
    files += ["ops/deploy-remote.sh", "scripts/deploy_prod.sh", "scripts/check.sh"]
    bad = []
    for f in sorted(set(files)):
        p = REPO / f
        if not p.is_file():
            continue
        if p.read_bytes().startswith(b"\xef\xbb\xbf"):
            bad.append(f"{f}: starts with a byte-order mark (breaks the #! line)")
        r = subprocess.run([BASH, "-n", _posix(p)], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(f"{f}: {r.stderr.strip()[:200]}")
    assert not bad, bad
