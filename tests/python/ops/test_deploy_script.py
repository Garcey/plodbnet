"""scripts/deploy_prod.sh — the local half of the production deploy.

Runs the real script in a throwaway git repository (never this one) with
DRY_RUN=1 / `pack`, so nothing touches the network: what ships is exactly the
committed allowlist with LF line endings, secrets and user data are refused,
uncommitted changes are refused unless ALLOW_DIRTY=1, and every mode's dry run
builds a bundle whose server script passes `bash -n`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from bash_tools import BASH, posix as _posix

REPO = Path(__file__).resolve().parents[3]
GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(BASH is None or GIT is None, reason="needs git and a GNU bash")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run([GIT, "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "core.autocrlf=false", *args],
                          check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    for rel in ("scripts/deploy_prod.sh", "ops/deploy-remote.sh", "ops/deploytool.py",
                "ops/bin/wrapgto-backup", "ops/systemd/wrapgto.service"):
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_bytes((REPO / rel).read_bytes().replace(b"\r\n", b"\n"))
    (r / "python" / "plo5bp").mkdir(parents=True)
    (r / "python" / "plo5bp" / "__init__.py").write_bytes(b"A = 1\nB = 2\n")
    (r / "rust_engine" / "src").mkdir(parents=True)
    (r / "rust_engine" / "src" / "lib.rs").write_text("// engine\n")
    (r / "pyproject.toml").write_text("[project]\nname = 'plo5bp'\n")
    (r / "Cargo.toml").write_text("[workspace]\n")
    (r / "Cargo.lock").write_text("# lock\n")
    (r / "docs").mkdir()
    (r / "docs" / "notes.md").write_text("not shipped\n")
    (r / "tests").mkdir()
    (r / "tests" / "test_x.py").write_text("not shipped\n")
    (r / ".gitignore").write_text("*.pyd\n")
    _git(r, "init", "-q", "-b", "main")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "init")
    return r


def _run(repo: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update({"DRY_RUN": "1", "WRAPGTO_HOST": "test-host", "WRAPGTO_SSH": "false"})
    e.update(env)
    return subprocess.run([BASH, _posix(repo / "scripts" / "deploy_prod.sh"), *args], cwd=repo, env=e,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)


def _shipped(repo: Path, cp: subprocess.CompletedProcess, tmp_path: Path) -> dict[str, bytes]:
    """Re-create what `pack` built (same git archive command) and read it."""
    out = tmp_path / "ship.tar"
    tree = _git(repo, "rev-parse", "HEAD").strip()
    subprocess.run([GIT, "-C", str(repo), "-c", "core.autocrlf=false", "archive", "--format=tar", "-o", str(out),
                    tree, "--", "python", "rust_engine", "Cargo.toml", "Cargo.lock", "pyproject.toml"], check=True)
    with tarfile.open(out) as tf:
        return {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}


def test_pack_ships_only_the_committed_allowlist(repo, tmp_path):
    (repo / "python" / "plo5bp" / "untracked_new.py").write_text("x = 1\n")
    (repo / "python" / "plo5bp" / "_engine.pyd").write_text("ignored build output")
    (repo / "secret.env").write_text("NOT SHIPPED\n")
    cp = _run(repo, "pack")
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "top level: Cargo.lock Cargo.toml ops pyproject.toml python rust_engine" in cp.stdout
    assert "uncommitted change" in cp.stdout            # the untracked file is pointed out…
    files = _shipped(repo, cp, tmp_path)
    assert "python/plo5bp/untracked_new.py" not in files  # …but never shipped
    assert not any(n.startswith(("docs/", "tests/")) or n.endswith(".pyd") for n in files)


def test_committed_crlf_is_shipped_as_stored(repo, tmp_path):
    # git stores LF; a CRLF working copy (Windows autocrlf) must not leak into the archive
    p = repo / "python" / "plo5bp" / "__init__.py"
    p.write_bytes(b"A = 1\r\nB = 2\r\n")
    cp = _run(repo, "pack")
    assert cp.returncode == 0
    assert _shipped(repo, cp, tmp_path)["python/plo5bp/__init__.py"] == b"A = 1\nB = 2\n"


def test_key_shaped_secrets_are_refused(repo):
    (repo / "python" / "plo5bp" / "keys.py").write_text('KEY = "sk_live_' + "a1B2c3D4e5F6g7H8i9J0" + '"\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "oops")
    cp = _run(repo, "pack")
    assert cp.returncode == 1
    assert "refusing to ship" in cp.stdout and "python/plo5bp/keys.py" in cp.stdout
    assert "a1B2c3" not in cp.stdout  # names only, never the secret itself


def test_the_word_alone_is_not_a_secret(repo):
    (repo / "python" / "plo5bp" / "guard.py").write_text('LIVE_PREFIXES = ("sk_live_", "rk_live_")\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "guard")
    assert _run(repo, "pack").returncode == 0


def test_user_data_names_are_refused(repo):
    (repo / "python" / "data").mkdir()
    (repo / "python" / "data" / "public.db").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "oops")
    cp = _run(repo, "pack")
    assert cp.returncode == 1 and "public.db" in cp.stdout


def test_deploy_refuses_uncommitted_changes_unless_allowed(repo):
    (repo / "python" / "plo5bp" / "__init__.py").write_text("A = 99\n")
    real = {k: v for k, v in os.environ.items()}
    cp = _run(repo, "deploy", DRY_RUN="0")  # refused before any network use
    assert cp.returncode == 1
    assert "commit them first" in cp.stdout
    cp = _run(repo, "deploy", ALLOW_DIRTY="1")
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "DIRTY=1" in cp.stdout and "ALLOW_DIRTY=1: shipping them" in cp.stdout
    # the real index was not touched
    assert _git(repo, "diff", "--cached", "--name-only") == ""
    assert real == {k: v for k, v in os.environ.items()}


@pytest.mark.parametrize("args", [
    ("check",), ("deploy",), ("stage",), ("rollback",), ("rollback", "20260928-101500"),
    ("restore-db",), ("restore-db", "public-x.db.gz"), ("promote-undo",), ("freeze",),
    ("watch",), ("recover",), ("install-unit",),
])
def test_every_mode_dry_runs_without_the_network(repo, args):
    cp = _run(repo, *args)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "DRY_RUN — nothing was sent" in cp.stdout
    assert "test-host" in cp.stdout
    assert "under /opt/wrapgto/deploys" in cp.stdout   # uploads land root-only, never at fixed /tmp names


def test_the_settings_reach_the_server_side(repo):
    cp = _run(repo, "rollback", "20260928-101500", FORCE="1", ROLLBACK_ANYWAY="1")
    assert "FORCE=1" in cp.stdout and "ROLLBACK_ANYWAY=1" in cp.stdout
    cp = _run(repo, "deploy", SKIP_ENGINE="1", ALLOW_ENGINE_MISMATCH="1")
    assert "SKIP_ENGINE=1" in cp.stdout and "ALLOW_ENGINE_MISMATCH=1" in cp.stdout
    cp = _run(repo, "promote-undo", OBS_REV="2")
    assert "OBS_REV=2" in cp.stdout
    cp = _run(repo, "help")
    for word in ("ROLLBACK_ANYWAY", "OBS_REV", "ALLOW_ENGINE_MISMATCH", "recover", "watch", "install-unit"):
        assert word in cp.stdout, word


def _fake_ssh(tmp_path, body: str) -> str:
    p = tmp_path / "fake-ssh"
    p.write_text("#!/usr/bin/env bash\ncat >/dev/null\n" + body, newline="\n")
    p.chmod(0o755)
    return _posix(p)


def test_the_servers_advice_becomes_the_exact_command(repo, tmp_path):
    ck = tmp_path / "vSix6_1400.pt"
    ck.write_bytes(b"not really a checkpoint")
    ssh = _fake_ssh(tmp_path, "echo '!! trained on obs rev 2'; echo 'RERUN-WITH OBS_REV=2 RESTART=1'; exit 4\n")
    cp = _run(repo, "promote", _posix(ck), DRY_RUN="0", WRAPGTO_SSH=ssh, CONFIRM="PROMOTE")
    assert cp.returncode == 4, cp.stdout + cp.stderr
    assert f"OBS_REV=2 RESTART=1 bash scripts/deploy_prod.sh promote {_posix(ck)}" in cp.stdout


def test_a_dropped_connection_points_at_watch(repo, tmp_path):
    ssh = _fake_ssh(tmp_path, "exit 255\n")
    cp = _run(repo, "recover", DRY_RUN="0", WRAPGTO_SSH=ssh, CONFIRM="RECOVER")
    assert cp.returncode == 255
    assert "carries on with it by itself" in cp.stdout and "deploy_prod.sh watch" in cp.stdout


def test_dry_run_sanitises_what_reaches_the_server_shell(repo):
    cp = _run(repo, "rollback", "x;rm -rf /")
    assert cp.returncode == 0
    assert "TARGET=x_rm_-rf_/" in cp.stdout


def test_promote_dry_run(repo, tmp_path):
    ck = tmp_path / "vSix6_1400.pt"
    ck.write_bytes(b"not really a checkpoint")
    cp = _run(repo, "promote", _posix(ck))
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "owner's explicit OK" in cp.stdout
    assert "MODE=promote" in cp.stdout and "CKPT_NAME=vSix6_1400.pt" in cp.stdout


def test_unknown_mode_and_help(repo):
    assert _run(repo, "bogus").returncode == 2
    cp = _run(repo, "help")
    assert cp.returncode == 0 and "promote FILE" in cp.stdout
