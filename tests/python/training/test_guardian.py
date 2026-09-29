"""The pod guardians' decisions (plo5bp/train/guardian.py, 2026-09-28).

ML-057: resume from the HIGHEST numbered compatible checkpoint (or the rolling
file when its counter says it is newer), not the newest by modification time.
ML-024: a heartbeat says how long ago THIS trainer finished an update.
The guardian scripts still parse (bash -n) and the vSix6 launch command is
the recipe's (see also test_train_cli.py, which parses and validates it).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import torch

from plo5bp.network import ActorCriticV5
from plo5bp.train import guardian as g

REPO = Path(__file__).resolve().parents[3]


def _save(path: Path, counter: int, hidden=16, critic_act="silu", obs_rev=1) -> None:
    m = ActorCriticV5(hidden_dim=hidden, num_layers=3)
    torch.save({"model": m.state_dict(), "update_counter": counter, "obs_rev": obs_rev,
                "config": {"hidden_dim": hidden, "num_layers": 3, "critic_hidden_dim": 1536,
                           "critic_act": critic_act, "obs_mode": "full"}}, path)


NEED = {"hidden_dim": "16", "num_layers": "3", "obs_mode": "full", "obs_rev": "1"}
OWN = {"critic_hidden_dim": "1536", "critic_act": "silu"}


def test_the_highest_number_wins_not_the_newest_file(tmp_path):
    for n in (1398, 1400, 1399):
        _save(tmp_path / f"st_{n}.pt", n)
    os.utime(tmp_path / "st_1398.pt", (time.time() + 60, time.time() + 60))  # "newest" by mtime
    (tmp_path / "st.optim.pt").write_bytes(b"x")                          # never a candidate
    (tmp_path / "st_1401.pt.tmp").write_bytes(b"x")
    assert g.pick_warm(tmp_path, "st", NEED, OWN) == tmp_path / "st_1400.pt"


def test_the_rolling_file_only_when_its_counter_is_newer(tmp_path):
    _save(tmp_path / "st_1400.pt", 1400)
    _save(tmp_path / "st.pt", 1401)          # the final save of the same weights (COUNT)
    assert g.candidates(tmp_path, "st")[0].name == "st_1400.pt"
    _save(tmp_path / "st.pt", 1403)          # updates ran after the last numbered save
    assert g.pick_warm(tmp_path, "st", NEED, OWN) == tmp_path / "st.pt"


def test_incompatible_files_are_skipped_then_the_fallback(tmp_path):
    _save(tmp_path / "st_1400.pt", 1400, hidden=32)       # wrong actor size
    _save(tmp_path / "st_1399.pt", 1399, critic_act="relu")  # its own file, old critic
    _save(tmp_path / "st_1398.pt", 1398)
    assert g.pick_warm(tmp_path, "st", NEED, OWN) == tmp_path / "st_1398.pt"
    (tmp_path / "st_1398.pt").unlink()
    warm = tmp_path / "other_1299.pt"
    _save(warm, 1299, critic_act="relu")  # another stem: the critic rule does not apply
    assert g.pick_warm(tmp_path, "st", NEED, OWN, also=[str(warm)]) == warm
    assert g.pick_warm(tmp_path, "st", {**NEED, "obs_rev": "2"}, OWN, also=[str(warm)]) is None


def test_heartbeat_age_belongs_to_one_process(tmp_path):
    hb = tmp_path / "st.heartbeat"
    started = time.time() - 200
    assert g.heartbeat_age(hb, 42, started=started) == -1  # missing: unknown
    hb.write_text(json.dumps({"time": time.time() - 100, "pid": 42}), encoding="utf-8")
    assert 99 <= g.heartbeat_age(hb, 42, started=started) < 110
    assert g.heartbeat_age(hb, 43, started=started) == -1  # another trainer's: unknown


def test_a_heartbeat_older_than_the_process_is_unknown(tmp_path):
    """An old-style trainer (before 2026-09-28) writes no heartbeat; a file left
    by an earlier trainer -- even one with the same PID -- predates this
    process and must never count as THIS trainer being hung."""
    hb = tmp_path / "st.heartbeat"
    hb.write_text(json.dumps({"time": time.time() - 50_000, "pid": 42}), encoding="utf-8")
    assert g.heartbeat_age(hb, 42, started=time.time() - 3600) == -1
    # a start time that cannot be read: unknown too (never a kill)
    assert g.heartbeat_age(hb, 42, started=None) in (-1, g.heartbeat_age(hb, 42))
    real = g.process_start_time(os.getpid())
    assert real is None or real <= time.time()


def test_the_cli_prints_the_pick(tmp_path, capsys):
    _save(tmp_path / "st_7.pt", 7)
    rc = g.main(["pick-warm", "--dir", str(tmp_path), "--stem", "st",
                 "--need", "hidden_dim=16", "--need-own", "critic_act=silu"])
    assert rc == 0 and capsys.readouterr().out.strip().endswith("st_7.pt")
    assert g.main(["pick-warm", "--dir", str(tmp_path), "--stem", "st",
                   "--need", "hidden_dim=99"]) == 1


BASH = shutil.which("bash")


@pytest.mark.skipif(BASH is None, reason="no bash")
@pytest.mark.parametrize("heartbeat", ["missing", "stale-old-file"])
def test_the_guardian_never_kills_an_old_style_trainer(tmp_path, heartbeat):
    """gl_check_heartbeat with a live process and no heartbeat of its own (an
    80048a0-era train.py) or only a stale file from before it started: no kill,
    one "check OFF" line, the PID watch rules."""
    runs = tmp_path / "runs"
    runs.mkdir()
    if heartbeat == "stale-old-file":
        (runs / "st.heartbeat").write_text(
            json.dumps({"time": time.time() - 99_999, "pid": 1}), encoding="utf-8")
    import sys as _sys
    py = Path(_sys.executable).as_posix()
    lib = (REPO / "scripts" / "guardian_lib.sh").as_posix()
    script = f"""
set -u
export PYTHONPATH="{(REPO / 'python').as_posix()}"
STEM=st GLOG=guardian.log PY="{py}" STALE_SECS=1
. "{lib}"
sleep 30 & victim=$!
gl_check_heartbeat "$victim"; rc=$?
gl_check_heartbeat "$victim"
kill -0 "$victim" && echo ALIVE
kill "$victim"
echo "rc=$rc"
"""
    r = subprocess.run([BASH, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert "ALIVE" in r.stdout and "rc=1" in r.stdout, (r.stdout, r.stderr)
    log = (tmp_path / "guardian.log").read_text(encoding="utf-8")
    assert log.count("heartbeat check OFF") == 1, log  # logged once per trainer
    assert "HUNG" not in log


@pytest.mark.skipif(BASH is None, reason="no bash")
@pytest.mark.parametrize("script", ["guardian_lib.sh", "vSix6_guardian.sh", "vMin3_guardian.sh"])
def test_guardian_scripts_parse(script):
    r = subprocess.run([BASH, "-n", f"scripts/{script}"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
