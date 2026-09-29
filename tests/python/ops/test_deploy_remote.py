"""The production deploy's server half (ops/deploy-remote.sh), run for real
against a fake server.

``scripts/deploy_prod.sh`` runs its remote half as root on the live machine, so
it can never be tried out there. These tests run the very same script under
bash with a fake /opt/wrapgto tree: shims stand in for systemctl / sudo / chown /
journalctl (and flock where Git Bash has none), a small HTTP server plays the
app's ``/health`` (it reads the LIVE tree and the obs revision the fake service
started with, so a switch or a settings change is visible, and answers
``?deploy=1`` from a "memory" the test controls), and a fake ``plo5bp`` package
lets the pre-flight import "the app" and check its model and its engine (a fake
``_engine`` whose SOURCE_HASH comes from the engine file, so SKIP_ENGINE reuses
the LIVE engine's hash) the way it checks the real one.

Every change runs DETACHED here too (nohup / setsid) and the test reads what the
launcher followed. Covered: deploy, the automatic rollback, a failing
pre-flight, the in-memory "is anyone playing?" guard and FORCE=1 (the guard
only), rollback (ROLLBACK_ANYWAY, old copies holding server folders), promote /
promote-undo (with an obs-revision change written and reverted together),
restore-db, install-unit, the engine-vs-sources check, the lock, surviving the
connection, an interrupted change and `recover`, check, and the real backup job.
"""
from __future__ import annotations

import gzip
import hashlib
import http.server
import importlib.util
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bash_tools import find_bash, posix

REPO = Path(__file__).resolve().parents[3]
REMOTE = REPO / "ops" / "deploy-remote.sh"
TOOL = REPO / "ops" / "deploytool.py"
BACKUP = REPO / "ops" / "bin" / "wrapgto-backup"
UNIT = REPO / "ops" / "systemd" / "wrapgto.service"

BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="needs a GNU bash (Git Bash on Windows)")
_posix = posix

_SPEC = importlib.util.spec_from_file_location("deploytool_for_remote_tests", TOOL)
dt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dt)


SHIMS = {
    # sudo -u USER -H cmd… -> cmd…   (the fake server has one user)
    "sudo": 'while [ $# -gt 0 ]; do case "$1" in -u) shift 2;; -H) shift;; *) break;; esac; done\nexec "$@"\n',
    "chown": "exit 0\n",
    "journalctl": "exit 0\n",
    # The fake service: FAKE_STATE exists while it "runs" and records the obs revision it
    # started with (read from the env file at start, like the real process); a unit file
    # holding BROKEN refuses to start. `show` / `cat` print what the test put in
    # FAKE_UNIT_SHOW / FAKE_UNIT_CAT.
    "systemctl": (
        'act=""; val=0; props=""; prev=""\n'
        'for a in "$@"; do\n'
        '  case "$a" in --value) val=1;; -p) ;; -*) ;;\n'
        '    *) if [ "$prev" = "-p" ]; then props="$props $a"; elif [ -z "$act" ]; then act="$a"; fi;;\n'
        '  esac; prev="$a"; done\n'
        'case "$act" in\n'
        '  stop) rm -f "$FAKE_STATE"; echo stop >> "$FAKE_LOG";;\n'
        '  start|restart)\n'
        '    if [ -n "${FAKE_UNITFILE:-}" ] && grep -qs BROKEN "$FAKE_UNITFILE"; then echo "broken-start" >> "$FAKE_LOG"; rm -f "$FAKE_STATE"; exit 1; fi\n'
        '    rev="$(sed -n "s/^PLO5BP_OBS_REV=//p" "$FAKE_ENVFILE" 2>/dev/null | tail -1)"\n'
        '    echo "on rev=${rev:-2}" > "$FAKE_STATE"; echo "$act" >> "$FAKE_LOG";;\n'
        '  daemon-reload) echo daemon-reload >> "$FAKE_LOG";;\n'
        '  is-active) q=0; for a in "$@"; do [ "$a" = --quiet ] && q=1; done\n'
        '    if [ -f "$FAKE_STATE" ]; then [ $q = 1 ] || echo active; exit 0; fi\n'
        '    [ $q = 1 ] || echo inactive; exit 3;;\n'
        '  show) f="${FAKE_UNIT_SHOW:-/nonexistent}"; [ -f "$f" ] || exit 0\n'
        '    if [ -z "$props" ]; then cat "$f"; exit 0; fi\n'
        '    for p in $props; do\n'
        '      if [ $val = 1 ]; then sed -n "s/^$p=//p" "$f"; else grep "^$p=" "$f" || true; fi\n'
        '    done;;\n'
        '  cat) [ -f "${FAKE_UNIT_CAT:-/nonexistent}" ] && cat "$FAKE_UNIT_CAT"; exit 0;;\n'
        '  *) exit 0;;\n'
        "esac\n"
    ),
    # a stand-in `cargo` (only its presence is checked; maturin is faked in the venv)
    "cargo": "exit 0\n",
    # CI runners have a real one, which would judge the fake unit against the runner
    "systemd-analyze": "exit 0\n",
}

# Git Bash has no flock: `flock -n FD` locks the file the CALLER has open on FD, as a
# lock directory owned by the caller's pid (stale once that process is gone); -u frees it.
FLOCK_SHIM = """\
un=0; fd=""
for a in "$@"; do case "$a" in -u) un=1;; -*) ;; *) fd="$a";; esac; done
f="$(readlink "/proc/$PPID/fd/$fd")" || exit 1
d="$f.shimlock"
if [ "$un" = 1 ]; then
  if [ "$(cat "$d/pid" 2>/dev/null)" = "$PPID" ]; then rm -rf "$d"; fi
  exit 0
fi
if mkdir "$d" 2>/dev/null; then echo "$PPID" > "$d/pid"; exit 0; fi
old="$(cat "$d/pid" 2>/dev/null)"
if [ -z "$old" ] || kill -0 "$old" 2>/dev/null; then exit 1; fi
rm -rf "$d"; mkdir "$d" && echo "$PPID" > "$d/pid"
"""

# `mv` that fails for ONE source path (FAKE_MV_FAIL) — an undo that cannot move data back.
MV_FAIL_SHIM = """\
for a in "$@"; do
  case "$a" in -*) ;; *) if [ "$a" = "${FAKE_MV_FAIL:-}" ]; then echo "mv: cannot move '$a' (test)" >&2; exit 1; fi; break;; esac
done
for r in /usr/bin/mv /bin/mv; do if [ -x "$r" ]; then exec "$r" "$@"; fi; done
exit 127
"""

# The fake engine: its SOURCE_HASH is written into the engine FILE, so reusing the
# live engine (SKIP_ENGINE=1) carries the live engine's hash — as in real life.
FAKE_ENV = '''\
import glob, os, sys, types
_eng = types.ModuleType("plo5bp._engine")
for _so in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "_engine*.so"))):
    _text = open(_so).read()
    if "SOURCE_HASH=" in _text:
        _eng.SOURCE_HASH = _text.split("SOURCE_HASH=", 1)[1].split()[0]
sys.modules["plo5bp._engine"] = _eng
class _RustGameState:
    def reset_with_deck(self):
        pass
'''
FAKE_HOMEGAME = "FAIR_ON = True\n"
# The fake app's "model": checkpoints/stub.pt holds text. "good…" loads, "unhealthy…"
# loads but the (fake) running app then reports model_loaded false, anything else
# does not load. "rev1" in it = trained on obs rev 1 (else rev 2).
FAKE_SERVER = '''\
import os
from pathlib import Path
VARIANT_PLO5 = "plo5_double_bomb"
if os.environ.get("FAKE_IMPORT_ERROR") or Path(__file__).with_name("BROKEN").exists():
    raise ImportError("this release is broken")
_ck = Path(os.environ.get("PLO5BP_CHECKPOINT") or "checkpoints/stub.pt")
def _format_ckpt_path(variant):
    return _ck
_text = _ck.read_text() if _ck.exists() else ""
MODEL_LOADED = _text.startswith(("good", "unhealthy"))
MODEL_CRITIC = object() if MODEL_LOADED else None
_ck_rev = 1 if "rev1" in _text else 2
_rev = int(os.environ.get("PLO5BP_OBS_REV", "") or "2")
FORMATS = {VARIANT_PLO5: {"obs_rev": _ck_rev, "obs_rev_mismatch": MODEL_LOADED and _ck_rev != _rev}}
'''


def _write_release(root: Path, marker: str, *, health: dict | None = None, broken: bool = False,
                   rust: str = "same", engine_hash: str | None = "auto") -> None:
    pkg = root / "python" / "plo5bp"
    (pkg / "ui").mkdir(parents=True)
    (pkg / "__init__.py").write_text(f"MARKER = {marker!r}\n")
    (pkg / "env.py").write_text(FAKE_ENV)
    (pkg / "ui" / "__init__.py").write_text("")
    (pkg / "ui" / "homegame.py").write_text(FAKE_HOMEGAME)
    (pkg / "ui" / "server.py").write_text(FAKE_SERVER)
    if broken:
        (pkg / "ui" / "BROKEN").write_text("x")
    crate = root / "rust_engine"
    (crate / "src").mkdir(parents=True)
    (crate / "src" / "lib.rs").write_text(f"// engine sources: {rust}\n")
    (crate / "Cargo.toml").write_text("[package]\nname = 'plo5bp_engine'\n")
    (root / "Cargo.lock").write_text("# lock\n")
    h = dt.engine_source_hash(crate) if engine_hash == "auto" else engine_hash
    (pkg / "_engine.cpython-311-x86_64-linux-gnu.so").write_text(
        f"engine {marker}" + (f" SOURCE_HASH={h}" if h else ""))
    (root / "health.json").write_text(json.dumps(health if health is not None else {"ok": True}))


def _make_db(path: Path, *, hand_in_progress: bool = False, recent: bool = False) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE homegames (id TEXT PRIMARY KEY, name TEXT, status TEXT, hand_no INTEGER);"
        "CREATE TABLE homegame_hands (game_id TEXT, hand_no INTEGER, ended_at TEXT, summary TEXT,"
        " PRIMARY KEY (game_id, hand_no));"
        "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT);"
    )
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(timespec="seconds")
    new = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute("INSERT INTO homegames VALUES ('g1', 'Friday', 'open', ?)", (3 if hand_in_progress else 2,))
    con.execute("INSERT INTO homegame_hands VALUES ('g1', 1, ?, '{\"grades\": []}')", (old,))
    con.execute("INSERT INTO homegame_hands VALUES ('g1', 2, ?, '{\"grades\": []}')", (new if recent else old,))
    con.execute("INSERT INTO users VALUES (1, 'live@example.com')")
    con.commit()
    con.close()


def _replace_db(fake: "FakeServer", **kw) -> None:
    _make_db(fake.app / "data" / "public.db.new", **kw)
    os.replace(fake.app / "data" / "public.db.new", fake.app / "data" / "public.db")


class FakeServer:
    """/opt/wrapgto on disk + a fake service whose /health reads the live tree."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.base = tmp / "opt" / "wrapgto"
        self.app = self.base / "app"
        self.keep = self.base / "backups"
        self.deploys = self.base / "deploys"
        self.state = tmp / "service-active"
        self.log = tmp / "service.log"
        self.unitfile = tmp / "etc" / "wrapgto.service"
        self.shims = tmp / "shims"
        self.shims.mkdir(parents=True)
        shims = dict(SHIMS)
        if shutil.which("flock") is None:
            shims["flock"] = FLOCK_SHIM
        for name, body in shims.items():
            self.add_shim(name, body)
        # the live tree: old code + server-owned state + an older deploy's leftovers
        _write_release(self.app, "v1")
        (self.app / "junk.md").write_text("left by an older deploy")
        (self.app / "data").mkdir()
        _make_db(self.app / "data" / "public.db")
        (self.app / "checkpoints").mkdir()
        (self.app / "checkpoints" / "stub.pt").write_text("good rev1 model-A")
        venv_bin = self.app / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        py = venv_bin / "python"
        py.write_text(f'#!/usr/bin/env bash\nexec "{_posix(sys.executable)}" "$@"\n', newline="\n")
        py.chmod(0o755)
        self.keep.mkdir(parents=True)
        self.envfile = tmp / "env"
        self.envfile.write_text("PLO5BP_OBS_REV=1\nPLO5BP_PUBLIC=1\n")
        self.state.write_text("on rev=1")
        # the app's in-memory answer to /health?deploy=1 (None = an app too old to give one)
        self.memory: dict | None = {"hands_in_progress": [], "games_running": []}
        self.old_app = False           # /health = {"ok": true} only, like the pre-rewrite app
        self.healthy_after = 0.0       # time.time() before which /health says "starting"
        self._start_http()

    def add_shim(self, name: str, body: str) -> None:
        p = self.shims / name
        p.write_text("#!/usr/bin/env bash\n" + body, newline="\n")
        p.chmod(0o755)

    # -- the fake app's /health ---------------------------------------------------
    def _payload(self, deploy: bool):
        if not self.state.exists() or time.time() < self.healthy_after:
            return 503, {"ok": False, "status": "starting"}
        try:
            rev = int(self.state.read_text().split("rev=")[1].split()[0])
        except (IndexError, ValueError, OSError):
            rev = 2
        try:
            h = json.loads((self.app / "health.json").read_text())
            if self.old_app and not (self.app / "BUILD_INFO.json").exists():
                return 200, {"ok": True}   # the pre-rewrite app; a deployed release answers fully
            stub = self.app / "checkpoints" / "stub.pt"
            text = stub.read_text() if stub.exists() else ""
            loaded = text.startswith("good")
            h.setdefault("model_loaded", loaded)
            h.setdefault("critic_loaded", h["model_loaded"])
            ck_rev = 1 if "rev1" in text else 2
            h.setdefault("obs_rev", ck_rev)
            h["process_obs_rev"] = rev
            h.setdefault("obs_rev_mismatch", bool(loaded and ck_rev != rev))
            h["sha256"] = hashlib.sha256(text.encode()).hexdigest()[:16]
            dep = self.app / "BUILD_INFO.json"
            if dep.exists():
                h["build"] = {"commit": json.loads(dep.read_text())["commit"]}
            if deploy and self.memory is not None:
                h["deploy"] = {"home_games": self.memory}
            broken = h["model_loaded"] is not True or h["obs_rev_mismatch"]
            return (503 if broken else 200), h
        except (OSError, ValueError):
            return 503, {"ok": False}

    def _start_http(self):
        srv = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                code, body = srv._payload(q.get("deploy") == ["1"])
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    # -- running the remote script ---------------------------------------------------
    def bundle(self, *, release: Path | None = None, candidate: str | None = None,
               backup_script: str | None = None, unit_text: str | None = None) -> Path:
        w = self.tmp / f"bundle-{len(list(self.tmp.glob('bundle-*')))}"
        w.mkdir()
        shutil.copy(REMOTE, w / "remote.sh")
        shutil.copy(TOOL, w / "deploytool.py")
        (w / "wrapgto-backup").write_bytes(
            backup_script.encode() if backup_script is not None else BACKUP.read_bytes().replace(b"\r\n", b"\n"))
        (w / "wrapgto.service").write_text(unit_text if unit_text is not None else UNIT.read_text(), newline="\n")
        if release is not None:
            with tarfile.open(w / "app.tgz", "w:gz") as tf:
                for p in sorted(release.rglob("*")):
                    tf.add(p, arcname=p.relative_to(release).as_posix(), recursive=False)
        if candidate is not None:
            (w / "candidate.pt").write_text(candidate)
        return w

    def env(self, mode: str, **env: str) -> dict[str, str]:
        e = dict(os.environ)
        git_usr = Path(BASH).parents[1] / "usr" / "bin" if os.name == "nt" else None
        e["PATH"] = os.pathsep.join([str(self.shims)] + ([str(git_usr)] if git_usr else []) + [e.get("PATH", "")])
        e.update({
            "FAKE_SHIMS": _posix(self.shims),
            "MODE": mode, "APP": _posix(self.app), "FAKE_STATE": _posix(self.state), "FAKE_LOG": _posix(self.log),
            "FAKE_ENVFILE": _posix(self.envfile), "FAKE_UNITFILE": _posix(self.unitfile),
            "WRAPGTO_PORT": str(self.port), "WRAPGTO_ENVFILE": _posix(self.envfile),
            "WRAPGTO_PY": _posix(sys.executable), "WRAPGTO_UNITFILE": _posix(self.unitfile),
            "WRAPGTO_HEALTH_TIMEOUT": "3", "SKIP_ENGINE": "1", "TMPDIR": _posix(self.tmp),
            "WRAPGTO_DETACH": "setsid" if shutil.which("setsid") else "nohup",
            "WRAPGTO_FOLLOW_POLL": "0.2",
            # for the backup job
            "WRAPGTO_APP": _posix(self.app), "WRAPGTO_BACKUPS": _posix(self.keep),
            "WRAPGTO_BACKUP_CONF": _posix(self.tmp / "no-backup.env"),
        })
        e.update(env)
        return e

    # The shims must win over /usr/bin, which Git Bash's launcher puts first.
    # umask 077: what deploy_prod.sh's ssh command sets on the server.
    LAUNCH = 'umask 077; PATH="$FAKE_SHIMS:$PATH"; export PATH; exec bash "$0"'

    def run(self, mode: str, w: Path, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run([BASH, "-c", self.LAUNCH, _posix(w / "remote.sh")], env=self.env(mode, **env),
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)

    def popen(self, mode: str, w: Path, **env: str) -> subprocess.Popen:
        return subprocess.Popen([BASH, "-c", self.LAUNCH, _posix(w / "remote.sh")], env=self.env(mode, **env),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def marker(self, tree: Path | None = None) -> str:
        t = (tree or self.app) / "python" / "plo5bp" / "__init__.py"
        return t.read_text().split("'")[1]

    def write_state(self, **kv: str) -> None:
        """A journal as an interrupted change leaves it (ops/deploy-remote.sh state_save)."""
        self.deploys.mkdir(parents=True, exist_ok=True)
        (self.deploys / "state").write_text("".join(f"{k}={shlex.quote(v)}\n" for k, v in kv.items()),
                                            newline="\n")


@pytest.fixture
def fake(tmp_path):
    s = FakeServer(tmp_path)
    yield s
    s.close()


def _release(tmp: Path, marker: str, **kw) -> Path:
    r = tmp / f"release-{marker}-{len(list(tmp.glob('release-*')))}"
    _write_release(r, marker, **kw)
    return r


def _out(cp: subprocess.CompletedProcess) -> str:
    return cp.stdout + cp.stderr


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _gz_db_ok(path: Path) -> bool:
    raw = path.parent / (path.name + ".check.db")
    raw.write_bytes(gzip.decompress(path.read_bytes()))
    con = sqlite3.connect(raw)
    try:
        return (con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                and con.execute("SELECT COUNT(*) FROM users").fetchone()[0] >= 1)
    finally:
        con.close()
        raw.unlink()


def _emails(db: Path) -> set[str]:
    con = sqlite3.connect(db)
    try:
        return {r[0] for r in con.execute("SELECT email FROM users")}
    finally:
        con.close()


# --- deploy ----------------------------------------------------------------------------


def test_deploy_switches_trees_and_carries_server_state(fake, tmp_path):
    w = fake.bundle(release=_release(tmp_path, "v2"))
    cp = fake.run("deploy", w, COMMIT="c0ffee1234567890", DEPLOYER="test")
    assert cp.returncode == 0, _out(cp)
    assert "LIVE and healthy" in cp.stdout
    assert fake.marker() == "v2"
    # server-owned state moved across, not copied or lost
    assert (fake.app / "data" / "public.db").exists()
    assert (fake.app / "checkpoints" / "stub.pt").read_text() == "good rev1 model-A"
    assert (fake.app / ".venv" / "bin" / "python").exists()
    info = json.loads((fake.app / "BUILD_INFO.json").read_text())
    assert info["commit"] == "c0ffee1234567890" and info["engine"]["source"] == "reused"
    # the engine's sources were checked, and both hashes recorded (finding 5)
    assert info["engine"]["matches_sources"] is True
    assert info["engine"]["source_hash"] == info["engine"]["rust_sources_hash"]
    # the retired tree keeps only code (+ the old leftovers), ready for a rollback
    (retired,) = list((fake.base / "releases").iterdir())
    assert fake.marker(retired) == "v1"
    assert (retired / "junk.md").exists()
    for d in ("data", "checkpoints", ".venv"):
        assert not (retired / d).exists()
    assert not (fake.base / "ship-staging").exists()
    if os.name != "nt":  # modes the service account depends on, despite the root-only umask
        import stat
        mode = lambda p: stat.S_IMODE(p.stat().st_mode)  # noqa: E731
        assert mode(fake.base / "releases") == 0o755
        assert mode(fake.app) & 0o055 == 0o055 and mode(fake.app / "python" / "plo5bp") & 0o055 == 0o055
        assert mode(fake.app / "python" / "plo5bp" / "__init__.py") & 0o044 == 0o044
        assert mode(fake.app / "python" / "plo5bp" / "__init__.py") & 0o022 == 0  # read-only to others
        assert mode(fake.app / "BUILD_INFO.json") & 0o044 == 0o044  # /health must read it
    # the pre-flight imported the NEW code with the live model and a copy of the data
    assert "copy of the live database" in cp.stdout
    # a fresh backup was taken first — by the UPLOADED ops/bin/wrapgto-backup (finding 7)
    (snap,) = list(fake.keep.glob("public-*.db.gz"))
    assert list(fake.keep.glob("data-*.tgz")) and list(fake.keep.glob("manifest-*.txt"))
    assert _gz_db_ok(snap)
    # it ran detached, said so, and said what it was doing while the site was down (finding 1)
    assert "does NOT stop it" in cp.stdout and "stopping the app (" in cp.stdout
    assert "starting the app" in cp.stdout
    (log,) = list(fake.deploys.glob("*-deploy.log"))
    assert (fake.deploys / (log.name + ".rc")).read_text().strip() == "0"
    assert "LIVE and healthy" in log.read_text()
    assert not (fake.deploys / "state").exists() and not (fake.deploys / ".current").exists()
    assert " deploy c0ffee1234567890 ok" in (fake.base / "deploys.log").read_text()
    assert "cd " in cp.stdout  # a shell in the old folder is told to cd again (finding 8)


def test_unhealthy_release_is_rolled_back_exactly(fake, tmp_path):
    rel = _release(tmp_path, "v2", health={"ok": True, "model_loaded": False})
    cp = fake.run("deploy", fake.bundle(release=rel), COMMIT="abc1234")
    assert cp.returncode == 5, _out(cp)
    assert "rolled back" in cp.stdout
    assert fake.marker() == "v1"
    for d in ("data", "checkpoints", ".venv"):
        assert (fake.app / d).exists(), d
    assert (fake.app / "junk.md").exists()
    failed = [p for p in (fake.base / "releases").iterdir() if p.name.endswith("-failed")]
    assert len(failed) == 1 and fake.marker(failed[0]) == "v2"
    assert fake.state.exists()  # the service is running again
    assert not (fake.deploys / "state").exists()
    assert "failed-rolled-back" in (fake.base / "deploys.log").read_text()


def test_failing_preflight_changes_nothing(fake, tmp_path):
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2", broken=True)), COMMIT="abc1234")
    assert cp.returncode == 4, _out(cp)
    assert "does not start" in cp.stdout
    assert fake.marker() == "v1"
    assert not (fake.base / "releases").exists() or not list((fake.base / "releases").iterdir())
    assert not fake.log.exists()  # never stopped or restarted


def test_preflight_refuses_an_obs_rev_mismatch(fake, tmp_path):
    fake.envfile.write_text("PLO5BP_OBS_REV=2\n")  # the live model says rev1
    cp = fake.run("stage", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234")
    assert cp.returncode == 4, _out(cp)
    assert "OBS-REV MISMATCH" in cp.stdout


def test_stage_only_never_touches_the_live_site(fake, tmp_path):
    cp = fake.run("stage", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234")
    assert cp.returncode == 0, _out(cp)
    assert fake.marker() == "v1"
    assert fake.marker(fake.base / "ship-staging") == "v2"
    assert (fake.base / "ship-staging" / "BUILD_INFO.json").exists()
    assert not fake.log.exists()
    assert not list(fake.keep.glob("public-*"))  # no backup needed for a stage


def test_the_backup_is_the_uploaded_script_and_a_failure_stops_everything(fake, tmp_path):
    """(finding 7) Never the server's own older job: the bundle's wrapgto-backup runs."""
    failing = "#!/usr/bin/env bash\necho 'uploaded backup script ran'; exit 1\n"
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2"), backup_script=failing), COMMIT="abc1234")
    assert cp.returncode == 2, _out(cp)
    assert "uploaded backup script ran" in cp.stdout and "backup failed" in cp.stdout
    assert fake.marker() == "v1" and not fake.log.exists()


def test_not_enough_disk_space_stops_before_anything(fake, tmp_path):
    shim = ('if [ "$1" = -Pk ]; then printf "Filesystem 1024-blocks Used Available Capacity Mounted\\n'
            'fake 1000000 999000 1000 100%% /\\n"; exit 0; fi; exec /usr/bin/df "$@"\n')
    fake.add_shim("df", shim)
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234")
    assert cp.returncode == 2, _out(cp)
    assert "not enough free disk space" in cp.stdout
    assert fake.marker() == "v1" and not fake.log.exists()


# --- the home-games guard (findings 2, 3) -----------------------------------------------


def test_a_hand_in_memory_blocks_the_switch_unless_forced(fake, tmp_path):
    fake.memory = {"hands_in_progress": [{"id": "g1", "hand_no": 3, "present": 4}], "games_running": []}
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234")
    assert cp.returncode == 6, _out(cp)
    assert "HAND IN PROGRESS" in cp.stdout and "Friday" in cp.stdout and "running app" in cp.stdout
    assert fake.marker() == "v1" and not fake.log.exists()
    assert not (fake.deploys / "state").exists()
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234", FORCE="1")
    assert cp.returncode == 0, _out(cp)
    assert fake.marker() == "v2"
    assert "come back PAUSED" in cp.stdout


def test_a_running_game_in_memory_blocks(fake, tmp_path):
    fake.memory = {"hands_in_progress": [], "games_running": [{"id": "g1", "hand_no": 2, "present": 3}]}
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234")
    assert cp.returncode == 6, _out(cp)
    assert "game running" in cp.stdout and "3 players at the table" in cp.stdout


def test_a_hand_cut_short_long_ago_never_blocks(fake, tmp_path):
    """(finding 3) hand_no is saved at the deal and a voided hand is never recorded: the
    database says "in progress" forever. The app's memory knows better."""
    _replace_db(fake, hand_in_progress=True)
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234")
    assert cp.returncode == 0, _out(cp)
    assert "void, not blocking" in cp.stdout and "hand #3" in cp.stdout


def test_an_older_app_falls_back_to_the_database(fake, tmp_path):
    fake.old_app = True
    _replace_db(fake, recent=True)
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234")
    assert cp.returncode == 6, _out(cp)
    assert "game running" in cp.stdout and "too old to report" in cp.stdout
    # a stale unrecorded hand does not block even then — it is listed as void
    _replace_db(fake, hand_in_progress=True)
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234")
    assert cp.returncode == 0, _out(cp)
    assert "void, not blocking" in cp.stdout and "lobby shows who is playing" in cp.stdout


# --- the engine vs its Rust sources (finding 5) --------------------------------------------


def test_a_reused_engine_from_other_rust_sources_is_refused(fake, tmp_path):
    rel = _release(tmp_path, "v2", rust="changed")
    cp = fake.run("deploy", fake.bundle(release=rel), COMMIT="abc1234")
    assert cp.returncode == 4, _out(cp)
    assert "OTHER Rust sources" in cp.stdout
    assert fake.marker() == "v1" and not fake.log.exists()
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2", rust="changed")), COMMIT="abc1234",
                  ALLOW_ENGINE_MISMATCH="1")
    assert cp.returncode == 0, _out(cp)
    assert "ALLOW_ENGINE_MISMATCH=1" in cp.stdout
    eng = json.loads((fake.app / "BUILD_INFO.json").read_text())["engine"]
    assert eng["matches_sources"] is False and eng["source_hash"] != eng["rust_sources_hash"]


def test_an_engine_older_than_source_hash_cannot_be_reused_unchecked(fake, tmp_path):
    shutil.rmtree(fake.app / "python")
    _write_release_code_only(fake.app, "v1", engine_hash=None)
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234")
    assert cp.returncode == 4, _out(cp)
    assert "predates SOURCE_HASH" in cp.stdout


def _write_release_code_only(root: Path, marker: str, **kw) -> None:
    tmp = root.parent / f"tmp-{marker}"
    _write_release(tmp, marker, **kw)
    shutil.move(str(tmp / "python"), str(root / "python"))
    shutil.rmtree(tmp)


def test_engine_build_without_a_toolchain_pin(fake, tmp_path):
    """(finding 9) A tree with no rust-toolchain.toml used to stop the script silently."""
    maturin = fake.app / ".venv" / "bin" / "maturin"
    maturin.write_text(
        "#!/usr/bin/env bash\n"
        'out=""; while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done\n'
        f'"{_posix(sys.executable)}" - "$out/plo5bp_engine-0-cp311-linux_x86_64.whl" "$FAKE_ENGINE_HASH" <<\'PY\'\n'
        "import sys, zipfile\n"
        "with zipfile.ZipFile(sys.argv[1], 'w') as z:\n"
        "    z.writestr('plo5bp/_engine.cpython-311-x86_64-linux-gnu.so', 'engine built SOURCE_HASH=' + sys.argv[2])\n"
        "PY\n", newline="\n")
    maturin.chmod(0o755)
    rel = _release(tmp_path, "v2", rust="brand new")
    assert not (rel / "rust-toolchain.toml").exists()
    cp = fake.run("deploy", fake.bundle(release=rel), COMMIT="abc1234", SKIP_ENGINE="0",
                  FAKE_ENGINE_HASH=dt.engine_source_hash(rel / "rust_engine"))
    if "root has no cargo" in cp.stdout:
        pytest.skip("this bash's login shell does not see the cargo shim")
    assert cp.returncode == 0, _out(cp)
    assert "pins none" in cp.stdout and "(built)" in cp.stdout
    eng = json.loads((fake.app / "BUILD_INFO.json").read_text())["engine"]
    assert eng["source"] == "built" and eng["matches_sources"] is True


# --- rollback (findings 2, 11) ------------------------------------------------------------------


def test_rollback_puts_the_previous_release_back(fake, tmp_path):
    assert fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234").returncode == 0
    assert fake.marker() == "v2"
    time.sleep(1.1)  # release folders are named by the second
    cp = fake.run("rollback", fake.bundle())
    assert cp.returncode == 0, _out(cp)
    assert fake.marker() == "v1"
    assert (fake.app / "data" / "public.db").exists() and (fake.app / ".venv").exists()
    names = [fake.marker(p) for p in (fake.base / "releases").iterdir()]
    assert names == ["v2"]  # rolling back again would return to v2
    # v1 predates BUILD_INFO.json: the switch says what its health check cannot see
    assert "predates BUILD_INFO.json" in cp.stdout


def test_force_does_not_switch_to_a_version_whose_preflight_failed(fake, tmp_path):
    """(finding 2) FORCE=1 skips the home-games check only; ROLLBACK_ANYWAY=1 is its
    own, loudly-warned decision."""
    (fake.base / "releases").mkdir(parents=True)
    _write_release(fake.base / "releases" / "20260101-000000", "broken", broken=True)
    cp = fake.run("rollback", fake.bundle(), TARGET="20260101-000000", FORCE="1")
    assert cp.returncode == 4, _out(cp)
    assert "ROLLBACK_ANYWAY=1" in cp.stdout and "FORCE=1 does NOT" in cp.stdout
    assert fake.marker() == "v1" and not fake.log.exists()
    cp = fake.run("rollback", fake.bundle(), TARGET="20260101-000000", ROLLBACK_ANYWAY="1")
    assert cp.returncode == 0, _out(cp)
    assert "switching although the pre-flight FAILED" in cp.stdout
    assert fake.marker() == "broken"


def test_rollback_to_an_old_copy_leaves_out_what_belongs_to_the_server(fake, tmp_path):
    """(finding 11) An old deploy's app-before-*.tgz can hold runs/, logs/, data/…"""
    old = tmp_path / "old-tree"
    _write_release(old, "v0")
    for d in ("runs", "logs", "data", "screenrecords", ".venv"):
        (old / d).mkdir()
        (old / d / "x.txt").write_text("from the old copy")
    tgz = fake.keep / "app-before-20260901-000000.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        for p in sorted(old.rglob("*")):
            tf.add(p, arcname="./" + p.relative_to(old).as_posix(), recursive=False)
    cp = fake.run("rollback", fake.bundle(), TARGET=tgz.name)
    assert cp.returncode == 0, _out(cp)
    assert "left out of the old copy" in cp.stdout
    assert fake.marker() == "v0"
    assert (fake.app / "data" / "public.db").exists()            # the LIVE data, carried
    assert not (fake.app / "data" / "x.txt").exists()
    assert not (fake.app / "runs").exists()


def test_failed_releases_are_pruned_separately(fake, tmp_path):
    rel = fake.base / "releases"
    rel.mkdir(parents=True)
    for i in range(4):
        (rel / f"2026010{i}-000000-failed").mkdir()
    for i in range(3):
        _write_release(rel / f"2025010{i}-000000", f"old{i}")
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234")
    assert cp.returncode == 0, _out(cp)
    names = sorted(p.name for p in rel.iterdir())
    assert len([n for n in names if n.endswith("-failed")]) == 2
    assert len([n for n in names if not n.endswith("-failed")]) == 4  # 3 old + the one just retired


# --- promote (findings 4, 12) ----------------------------------------------------------------------


def test_promote_swaps_the_model_and_keeps_the_previous(fake):
    cp = fake.run("promote", fake.bundle(candidate="good rev1 model-B"), RESTART="1",
                  CKPT_SHA=_sha("good rev1 model-B"), CKPT_NAME="vSix6_1400.pt")
    assert cp.returncode == 0, _out(cp)
    ck = fake.app / "checkpoints"
    assert (ck / "stub.pt").read_text() == "good rev1 model-B"
    assert (ck / "stub.pt.prev").read_text() == "good rev1 model-A"
    assert "MODEL-LOG" in cp.stdout
    assert "vSix6_1400.pt" in (fake.base / "models.log").read_text()
    # …and promote-undo puts model A back (B becomes the .prev)
    cp = fake.run("promote-undo", fake.bundle())
    assert cp.returncode == 0, _out(cp)
    assert (ck / "stub.pt").read_text() == "good rev1 model-A"
    assert (ck / "stub.pt.prev").read_text() == "good rev1 model-B"
    assert sorted(p.name for p in ck.iterdir()) == ["stub.pt", "stub.pt.prev"]


def test_promote_stages_for_the_no_restart_swap_when_the_app_can_do_it(fake):
    """The live app's own model manager (/admin → System) promotes <model>.new with
    no restart — so promote only verifies and stages, unless RESTART=1."""
    ui = fake.app / "python" / "plo5bp" / "ui"
    (ui / "public.py").write_text('ROUTE = "/admin/api/models"\n')
    cp = fake.run("promote", fake.bundle(candidate="good rev1 model-B"), CKPT_SHA=_sha("good rev1 model-B"))
    assert cp.returncode == 0, _out(cp)
    assert "MODEL-STAGED" in cp.stdout and "NO restart" in cp.stdout
    ck = fake.app / "checkpoints"
    assert (ck / "stub.pt").read_text() == "good rev1 model-A"      # live model untouched
    assert (ck / "stub.pt.new").read_text() == "good rev1 model-B"
    assert not fake.log.exists()                                     # never restarted
    cp = fake.run("promote", fake.bundle(candidate="good rev1 model-B"), CKPT_SHA=_sha("good rev1 model-B"),
                  RESTART="1")
    assert cp.returncode == 0, _out(cp)
    assert (ck / "stub.pt").read_text() == "good rev1 model-B"
    assert not (ck / "stub.pt.new").exists()   # superseded by the restart swap


def test_promote_without_the_no_restart_manager_asks_for_restart(fake):
    cp = fake.run("promote", fake.bundle(candidate="good rev1 model-B"), CKPT_SHA=_sha("good rev1 model-B"))
    assert cp.returncode == 2, _out(cp)
    assert "RERUN-WITH RESTART=1" in cp.stdout
    assert sorted(p.name for p in (fake.app / "checkpoints").iterdir()) == ["stub.pt"]


def test_promote_refuses_a_model_the_live_code_cannot_serve(fake):
    cp = fake.run("promote", fake.bundle(candidate="garbage"), CKPT_SHA=_sha("garbage"), RESTART="1")
    assert cp.returncode == 4, _out(cp)
    assert "random, untrained network" in cp.stdout
    ck = fake.app / "checkpoints"
    assert (ck / "stub.pt").read_text() == "good rev1 model-A"
    assert sorted(p.name for p in ck.iterdir()) == ["stub.pt"]  # nothing staged or left behind
    assert not fake.log.exists()


def test_promote_reverts_when_the_site_comes_up_unhealthy(fake):
    cp = fake.run("promote", fake.bundle(candidate="unhealthy rev1"), CKPT_SHA=_sha("unhealthy rev1"), RESTART="1")
    assert cp.returncode == 5, _out(cp)
    assert (fake.app / "checkpoints" / "stub.pt").read_text() == "good rev1 model-A"
    assert not (fake.deploys / "state").exists()


def test_promote_rejects_a_corrupted_upload(fake):
    cp = fake.run("promote", fake.bundle(candidate="good rev1 model-B"), CKPT_SHA="0" * 64)
    assert cp.returncode == 2, _out(cp)
    assert (fake.app / "checkpoints" / "stub.pt").read_text() == "good rev1 model-A"


def test_a_refused_restart_promote_leaves_nothing_staged(fake):
    """(finding 12) The guard runs before anything is placed."""
    fake.memory = {"hands_in_progress": [{"id": "g1", "hand_no": 3}], "games_running": []}
    cp = fake.run("promote", fake.bundle(candidate="good rev1 model-B"), CKPT_SHA=_sha("good rev1 model-B"),
                  RESTART="1")
    assert cp.returncode == 6, _out(cp)
    assert sorted(p.name for p in (fake.app / "checkpoints").iterdir()) == ["stub.pt"]


def test_an_obs_rev_change_goes_live_together_with_the_model(fake):
    """(finding 4) OBS_REV writes PLO5BP_OBS_REV WITH the swap; the health check then
    wants the process on that revision."""
    before = fake.envfile.read_text()
    cp = fake.run("promote", fake.bundle(candidate="good rev2 model-B"), CKPT_SHA=_sha("good rev2 model-B"))
    assert cp.returncode == 4, _out(cp)
    assert "RERUN-WITH OBS_REV=2 RESTART=1" in cp.stdout      # the exact advice
    assert fake.envfile.read_text() == before
    cp = fake.run("promote", fake.bundle(candidate="good rev2 model-B"), CKPT_SHA=_sha("good rev2 model-B"),
                  OBS_REV="2", RESTART="1")
    assert cp.returncode == 0, _out(cp)
    assert dt.parse_envfile(fake.envfile.read_text()) == {"PLO5BP_OBS_REV": "2", "PLO5BP_PUBLIC": "1"}
    assert (fake.app / "checkpoints" / "stub.pt").read_text() == "good rev2 model-B"
    assert "rev=2" in fake.state.read_text()
    assert not list(fake.tmp.glob("env.before-*"))             # the backup goes once it is healthy


def test_an_obs_rev_change_is_reverted_with_the_model(fake):
    before = fake.envfile.read_text()
    cp = fake.run("promote", fake.bundle(candidate="unhealthy rev2"), CKPT_SHA=_sha("unhealthy rev2"),
                  OBS_REV="2", RESTART="1")
    assert cp.returncode == 5, _out(cp)
    assert fake.envfile.read_text() == before                  # the settings too, byte for byte
    assert (fake.app / "checkpoints" / "stub.pt").read_text() == "good rev1 model-A"
    assert "rev=1" in fake.state.read_text()
    assert not list(fake.tmp.glob("env.before-*")) and not (fake.deploys / "state").exists()


def test_the_no_restart_path_refuses_what_it_cannot_do(fake):
    (fake.app / "python" / "plo5bp" / "ui" / "public.py").write_text('ROUTE = "/admin/api/models"\n')
    cp = fake.run("promote", fake.bundle(candidate="good rev2 model-B"), CKPT_SHA=_sha("good rev2 model-B"),
                  OBS_REV="2")
    assert cp.returncode == 2, _out(cp)
    assert "needs a restart" in cp.stdout and "RERUN-WITH OBS_REV=2 RESTART=1" in cp.stdout
    # the env file changed but the process did not restart: /admin would serve the
    # model on the wrong revision
    fake.envfile.write_text("PLO5BP_OBS_REV=2\nPLO5BP_PUBLIC=1\n")
    cp = fake.run("promote", fake.bundle(candidate="good rev2 model-B"), CKPT_SHA=_sha("good rev2 model-B"))
    assert cp.returncode == 2, _out(cp)
    assert "changed without a restart" in cp.stdout
    assert sorted(p.name for p in (fake.app / "checkpoints").iterdir()) == ["stub.pt"]


# --- restore-db (finding 12) ------------------------------------------------------------------------


def test_restore_db_replaces_the_database_and_keeps_the_old_one(fake):
    backup = fake.keep / "public-older.db"
    _make_db(backup)
    con = sqlite3.connect(backup)
    con.execute("INSERT INTO users VALUES (2, 'from-backup@example.com')")
    con.commit()
    con.close()
    live = fake.app / "data" / "public.db"
    con = sqlite3.connect(live)
    con.execute("INSERT INTO users VALUES (3, 'only-in-live@example.com')")
    con.commit()
    con.close()
    # a leftover journal of the OLD database (empty: SQLite leaves it alone until then)
    (fake.app / "data" / "public.db-journal").write_bytes(b"")
    cp = fake.run("restore-db", fake.bundle(), TARGET="public-older.db")
    assert cp.returncode == 0, _out(cp)
    assert "from-backup@example.com" in _emails(live)
    # the old journal must never meet the restored database
    assert not (fake.app / "data" / "public.db-journal").exists()
    (replaced,) = list(fake.keep.glob("replaced-*"))
    assert (replaced / "raw" / "public.db").exists() and (replaced / "raw" / "public.db-journal").exists()
    # one self-contained copy to put back by hand, and how
    assert "only-in-live@example.com" in _emails(replaced / "public.db")
    assert "restore-db replaced-" in (replaced / "README.txt").read_text()
    # …and it is a restore-db target itself
    cp = fake.run("restore-db", fake.bundle(), TARGET=f"{replaced.name}/public.db")
    assert cp.returncode == 0, _out(cp)
    assert "only-in-live@example.com" in _emails(live)


def test_restore_db_refuses_a_broken_backup(fake):
    (fake.keep / "public-broken.db").write_bytes(b"not a database")
    cp = fake.run("restore-db", fake.bundle(), TARGET="public-broken.db")
    assert cp.returncode == 2, _out(cp)
    assert not fake.log.exists()


def test_restore_db_puts_the_original_back_when_unhealthy(fake):
    backup = fake.keep / "public-older.db"
    _make_db(backup)
    before = (fake.app / "data" / "public.db").read_bytes()
    (fake.app / "health.json").write_text(json.dumps({"ok": True}))
    fake.healthy_after = time.time() + 3600   # the app never answers healthy
    cp = fake.run("restore-db", fake.bundle(), TARGET="public-older.db", FORCE="1")
    fake.healthy_after = 0.0
    assert cp.returncode == 5, _out(cp)
    assert (fake.app / "data" / "public.db").read_bytes() == before
    assert not (fake.deploys / "state").exists()


# --- install-unit (finding 6) -------------------------------------------------------------------------


# Paths INSIDE files that deploytool (a native python) reads are written the native
# way; Git Bash converts only command-line arguments.
def _fake_unit(fake: FakeServer, extra: str = "") -> str:
    text = UNIT.read_text().replace("/opt/wrapgto/app", fake.app.as_posix()).replace("/etc/wrapgto/env",
                                                                                      fake.envfile.as_posix())
    return text + extra


def _unit_show(fake: FakeServer, *, environment: str = "PYTHONUNBUFFERED=1", dropins: str = "") -> None:
    fake.unitfile.parent.mkdir(parents=True, exist_ok=True)
    fake.unitfile.write_text("[Service]\nExecStart=/old/python -m uvicorn plo5bp.ui.server:app\n", newline="\n")
    show = fake.tmp / "unit-show.txt"
    show.write_text(
        f"FragmentPath={fake.unitfile.as_posix()}\nDropInPaths={dropins}\n"
        f"ExecStart={{ path=/old/python ; argv[]=/old/python -m uvicorn plo5bp.ui.server:app --workers 2 ; ignore_errors=no }}\n"
        f"Environment={environment}\n"
        f"EnvironmentFiles={fake.envfile.as_posix()} (ignore_errors=no)\n"
        f"User=wrapgto\nWorkingDirectory={fake.app.as_posix()}\nTimeoutStopUSec=30s\n", newline="\n")
    (fake.tmp / "unit-cat.txt").write_text(fake.unitfile.read_text())


def _unit_env(fake: FakeServer) -> dict[str, str]:
    return {"FAKE_UNIT_SHOW": _posix(fake.tmp / "unit-show.txt"), "FAKE_UNIT_CAT": _posix(fake.tmp / "unit-cat.txt")}


def test_install_unit_refuses_to_drop_settings_kept_in_the_old_unit(fake):
    _unit_show(fake, environment="PYTHONUNBUFFERED=1 PLO5BP_SECRET_ONLY_HERE=1")
    old = fake.unitfile.read_text()
    cp = fake.run("install-unit", fake.bundle(unit_text=_fake_unit(fake)), **_unit_env(fake))
    assert cp.returncode == 2, _out(cp)
    assert "would be LOST" in cp.stdout and "PLO5BP_SECRET_ONLY_HERE" in cp.stdout
    assert fake.unitfile.read_text() == old and not fake.log.exists()


def test_install_unit_installs_checks_and_keeps_the_old_one(fake):
    _unit_show(fake)
    old = fake.unitfile.read_text()
    new = _fake_unit(fake)
    cp = fake.run("install-unit", fake.bundle(unit_text=new), **_unit_env(fake))
    assert cp.returncode == 0, _out(cp)
    assert "LIVE and healthy" in cp.stdout
    assert fake.unitfile.read_text() == new
    assert "daemon-reload" in fake.log.read_text()
    (kept,) = list(fake.deploys.glob("unit-before-*.service"))
    assert kept.read_text() == old
    # the pre-flight/guard/backup side sees what the unit shows (finding 6: `check` prints it)
    cp = fake.run("check", fake.bundle(unit_text=new), **_unit_env(fake))
    assert "--workers" in cp.stdout and "runs as wrapgto" in cp.stdout


def test_install_unit_reverts_when_the_app_does_not_start(fake):
    _unit_show(fake)
    old = fake.unitfile.read_text()
    cp = fake.run("install-unit", fake.bundle(unit_text=_fake_unit(fake, "# BROKEN\n")), **_unit_env(fake))
    assert cp.returncode == 5, _out(cp)
    assert fake.unitfile.read_text() == old
    assert fake.state.exists()                  # running again, on the old definition
    assert not (fake.deploys / "state").exists()


# --- running detached, the lock, recover (findings 1, 10) --------------------------------------------


def _wait_for(pred, timeout=60.0, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def test_a_dropped_connection_never_stops_the_switch(fake, tmp_path):
    """The launcher (the ssh session) dies right after the site stopped; the change
    still finishes on its own and says so in its log."""
    fake.healthy_after = time.time() + 3600            # the new app takes a while to answer
    proc = fake.popen("deploy", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234",
                      WRAPGTO_HEALTH_TIMEOUT="60")
    _wait_for(lambda: any("switching the live site over" in p.read_text(errors="replace")
                          for p in fake.deploys.glob("*.log")), what="the switch to start")
    proc.kill()                                        # the window is closed
    proc.wait(timeout=30)
    fake.healthy_after = 0.0                           # …and the app comes up
    (log,) = list(fake.deploys.glob("*-deploy.log"))
    _wait_for(lambda: (fake.deploys / (log.name + ".rc")).exists(), what="the detached run to finish")
    assert (fake.deploys / (log.name + ".rc")).read_text().strip() == "0", log.read_text()
    assert fake.marker() == "v2" and "LIVE and healthy" in log.read_text()
    # `watch` shows the finished log afterwards
    cp = fake.run("watch", fake.bundle())
    assert cp.returncode == 0 and "LIVE and healthy" in cp.stdout and "finished (exit 0" in cp.stdout


def test_one_change_at_a_time(fake, tmp_path):
    fake.deploys.mkdir(parents=True)
    lockfile = _posix(fake.deploys / ".lock")
    holder = subprocess.Popen([BASH, "-c", 'PATH="$FAKE_SHIMS:$PATH"; exec 9>>"$1"; flock -n 9 || exit 3; '
                               'echo locked; exec sleep 60', "holder", lockfile],
                              env=fake.env("check"), stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v2")), COMMIT="abc1234")
        assert cp.returncode == 8, _out(cp)
        assert "another change is running" in cp.stdout
        assert fake.marker() == "v1" and not (fake.base / "ship-staging").exists()
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_an_interrupted_change_blocks_others_until_recovered(fake, tmp_path):
    """The process died after moving the live code aside, before the new code moved in."""
    rel = fake.base / "releases"
    rel.mkdir(parents=True)
    stage = fake.base / "ship-staging"
    _write_release(stage, "v2")
    os.rename(fake.app, rel / "20260928-100000")
    fake.state.unlink()                                # the site is down
    fake.write_state(S_KIND="switch", S_MODE="deploy", S_TS="20260928-100000", S_PHASE="moving-new",
                     S_NEW=_posix(stage), S_RETIRED=_posix(rel / "20260928-100000"), S_WANT="abc1234")
    cp = fake.run("deploy", fake.bundle(release=_release(tmp_path, "v3")), COMMIT="abc1234")
    assert cp.returncode == 9, _out(cp)
    assert "interrupted half-way" in cp.stdout and "recover" in cp.stdout
    cp = fake.run("check", fake.bundle())
    assert "INTERRUPTED" in cp.stdout
    cp = fake.run("recover", fake.bundle())
    assert cp.returncode == 0, _out(cp)
    assert fake.marker() == "v1" and (fake.app / "data" / "public.db").exists()
    assert fake.state.exists() and not (fake.deploys / "state").exists()
    assert "previous version is live" in cp.stdout


def test_recover_finishes_a_switch_whose_new_code_was_in_place(fake, tmp_path):
    rel = fake.base / "releases"
    rel.mkdir(parents=True)
    retired = rel / "20260928-100000"
    os.rename(fake.app, retired)
    _write_release(fake.app, "v2")
    (fake.app / "BUILD_INFO.json").write_text(json.dumps({"commit": "abc1234"}))
    os.rename(retired / "checkpoints", fake.app / "checkpoints")   # carried before the crash
    fake.state.unlink()
    fake.write_state(S_KIND="switch", S_MODE="deploy", S_TS="20260928-100000", S_PHASE="carrying",
                     S_NEW=_posix(fake.base / "ship-staging"), S_RETIRED=_posix(retired), S_WANT="abc1234")
    cp = fake.run("recover", fake.bundle())
    assert cp.returncode == 0, _out(cp)
    assert "the new version is LIVE and healthy" in cp.stdout
    assert fake.marker() == "v2"
    for d in ("data", "checkpoints", ".venv"):
        assert (fake.app / d).exists() and not (retired / d).exists(), d


def test_an_undo_that_cannot_move_data_back_never_starts_the_old_code_without_it(fake, tmp_path):
    """(finding 10) It stops with exact commands and leaves `recover` to finish."""
    fake.add_shim("mv", MV_FAIL_SHIM)
    rel = _release(tmp_path, "v2", health={"ok": True, "model_loaded": False})
    cp = fake.run("deploy", fake.bundle(release=rel), COMMIT="abc1234", FAKE_MV_FAIL=_posix(fake.app / "data"))
    assert cp.returncode == 7, _out(cp)
    assert "COULD NOT MOVE data BACK" in cp.stdout and "mv -T" in cp.stdout
    assert not fake.state.exists()                     # the old code was NOT started without its data
    assert fake.marker() == "v2" and (fake.app / "data" / "public.db").exists()
    assert (fake.deploys / "state").exists()
    cp = fake.run("recover", fake.bundle())            # the mv works again
    assert cp.returncode == 0, _out(cp)
    assert fake.marker() == "v1" and (fake.app / "data" / "public.db").exists()
    assert fake.state.exists() and not (fake.deploys / "state").exists()


def test_recover_with_nothing_to_do(fake):
    cp = fake.run("recover", fake.bundle())
    assert cp.returncode == 0, _out(cp)
    assert "nothing to recover" in cp.stdout


# --- check / list ---------------------------------------------------------------------------------------


def test_check_and_list_change_nothing(fake):
    before = sorted(p.relative_to(fake.tmp).as_posix() for p in fake.base.rglob("*"))
    cp = fake.run("check", fake.bundle())
    assert cp.returncode == 0, _out(cp)
    assert "nothing was changed" in cp.stdout
    assert "junk.md" in cp.stdout  # older deploys' leftovers are pointed out
    assert "asked the running app" in cp.stdout
    cp = fake.run("list", fake.bundle())
    assert cp.returncode == 0, _out(cp)
    after = sorted(p.relative_to(fake.tmp).as_posix() for p in fake.base.rglob("*")
                   if "__pycache__" not in p.as_posix())
    assert [p for p in before if "__pycache__" not in p] == after
    assert not fake.log.exists()


# --- the backup script ---------------------------------------------------------------------------------


def _run_backup(fake: FakeServer, *args: str, **env: str) -> subprocess.CompletedProcess:
    job = fake.tmp / "wrapgto-backup"
    job.write_bytes(BACKUP.read_bytes().replace(b"\r\n", b"\n"))
    e = dict(os.environ)
    e.update({"FAKE_SHIMS": _posix(fake.shims), "WRAPGTO_APP": _posix(fake.app),
              "WRAPGTO_BACKUPS": _posix(fake.keep), "WRAPGTO_ENVFILE": _posix(fake.envfile),
              "WRAPGTO_BACKUP_CONF": _posix(fake.tmp / "backup.env"), "WRAPGTO_PY": _posix(sys.executable),
              "TMPDIR": _posix(fake.tmp)})
    e.update(env)
    launch = 'PATH="$FAKE_SHIMS:$PATH"; export PATH; exec bash "$0" "$@"'
    return subprocess.run([BASH, "-c", launch, _posix(job), *args], env=e, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=120)


def test_backup_script_snapshot_manifest_and_retention(fake):
    old = fake.keep / "public-20200101-000000.db.gz"
    old.write_bytes(b"old")
    t = time.time() - 30 * 86400
    os.utime(old, (t, t))
    cp = _run_backup(fake)
    assert cp.returncode == 0, _out(cp)
    assert "off-site copy: not set up" in cp.stdout
    (snap,) = list(fake.keep.glob("public-*.db.gz"))           # the 30-day-old one was pruned
    assert _gz_db_ok(snap)
    (data,) = list(fake.keep.glob("data-*.tgz"))
    with tarfile.open(data) as tf:                              # data/ minus the live database files
        assert not any(n.endswith("public.db") for n in tf.getnames())
    manifest = next(fake.keep.glob("manifest-*.txt")).read_text()
    assert snap.name in manifest and "model stub.pt:" in manifest
    assert not list(fake.keep.glob(".*part"))


def test_backup_retention_comes_from_its_config_file(fake):
    """(finding 12) WRAPGTO_BACKUP_DAYS in backup.env used to be read before the file."""
    old = fake.keep / "public-20200101-000000.db.gz"
    old.write_bytes(b"old")
    t = time.time() - 30 * 86400
    os.utime(old, (t, t))
    (fake.tmp / "backup.env").write_text("WRAPGTO_BACKUP_DAYS=40\n")
    cp = _run_backup(fake)
    assert cp.returncode == 0, _out(cp)
    assert old.exists()


def test_backup_tolerates_files_changing_under_it(fake):
    """(finding 12) GNU tar exits 1 when a file changed or vanished while read — the
    archive is complete; only 2+ is a failure."""
    real = "/usr/bin/tar"
    fake.add_shim("tar", f'"{real}" "$@"; rc=$?; [ "$rc" = 0 ] && [ -n "${{FAKE_TAR_RC:-}}" ] && exit "$FAKE_TAR_RC"; exit $rc\n')
    assert _run_backup(fake, FAKE_TAR_RC="1").returncode == 0
    cp = _run_backup(fake, FAKE_TAR_RC="2")
    assert cp.returncode == 1 and "tar exit 2" in cp.stdout


def test_backup_uses_the_database_the_deploy_names(fake):
    other = fake.tmp / "elsewhere.db"
    _make_db(other)
    cp = _run_backup(fake, WRAPGTO_DB=_posix(other))
    assert cp.returncode == 0, _out(cp)
    assert "elsewhere.db" in next(fake.keep.glob("manifest-*.txt")).read_text()


def test_backup_script_check_and_drill(fake):
    (fake.app / "ops").mkdir()
    shutil.copy(TOOL, fake.app / "ops" / "deploytool.py")
    assert _run_backup(fake, "--check").returncode == 1        # nothing backed up yet
    assert _run_backup(fake, WRAPGTO_BACKUP_LOCAL_ONLY="1").returncode == 0
    cp = _run_backup(fake, "--check")
    assert cp.returncode == 0, _out(cp)
    cp = _run_backup(fake, "--drill")                           # no off-site copy configured
    assert cp.returncode == 1 and "nothing to drill" in cp.stdout
    assert _run_backup(fake, "--bogus").returncode == 2


def test_backup_script_fails_loudly_without_a_database(fake):
    (fake.app / "data" / "public.db").unlink()
    cp = _run_backup(fake)
    assert cp.returncode == 1 and "no database" in cp.stdout
    assert not list(fake.keep.glob("public-*"))


def test_backup_prunes_old_replaced_folders_but_keeps_the_newest(fake):
    t = time.time() - 60 * 86400
    for name in ("replaced-20200101-000000", "replaced-20200102-000000"):
        (fake.keep / name).mkdir()
        os.utime(fake.keep / name, (t, t))
    os.utime(fake.keep / "replaced-20200102-000000", (t + 10, t + 10))
    assert _run_backup(fake).returncode == 0
    assert sorted(p.name for p in fake.keep.glob("replaced-*")) == ["replaced-20200102-000000"]


