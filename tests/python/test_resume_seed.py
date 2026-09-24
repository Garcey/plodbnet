"""A resumed run draws its own random stream (2026-09-23).

train.py used to re-seed every run with the bare --seed, so each relaunch of
a guardian replayed the fresh run's first updates exactly: the same table
configs and the same card deals. A resume now seeds from (seed, resume
update): different from the fresh run's start, identical across two resumes
of the same checkpoint (still reproducible); fresh runs are untouched.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_ARGS = [
    "--variant", "plo5_double_bomb", "--v6", "--obs-mode", "minimal",
    "--hidden-dim", "16", "--num-layers", "3",
    "--critic-hidden-dim", "16", "--critic-num-blocks", "1",
    "--batched", "--device", "cpu", "--num-envs", "120", "--rollout-length", "1500",
    "--num-minibatches", "2", "--ppo-epochs", "1",
    "--mix-configs", "--configs-per-tier", "1", "--mix-tiers", "clubgg,clubgg_deep,deep",
    "--cpu-threads", "2", "--snapshot-every", "1", "--checkpoint-every", "1",
    "--seed", "3", "--num-updates", "1",
]


def _train(ckpt: Path, load: Path | None = None) -> str:
    cmd = [sys.executable, "-u", "scripts/train.py", *_ARGS, "--checkpoint", str(ckpt)]
    if load is not None:
        cmd += ["--load-checkpoint", str(load)]
    env = dict(os.environ, PLO5_RUST_ENCODER="1", PLO5BP_STEP_TIMERS="0")
    out = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    return out.stdout


def _first_stacks(log: str) -> str:
    m = re.search(r"update\s+0 .*?stacks_bb=(\[[^\]]*\])", log)
    assert m, log[-2000:]
    return m.group(1)


def test_resume_draws_a_fresh_but_reproducible_stream(tmp_path) -> None:
    fresh = _train(tmp_path / "a.pt")
    assert "[seed] resumed" not in fresh
    r1 = _train(tmp_path / "b.pt", load=tmp_path / "a.pt")
    r2 = _train(tmp_path / "c.pt", load=tmp_path / "a.pt")
    assert "[seed] resumed at update" in r1
    assert _first_stacks(r1) == _first_stacks(r2)      # reproducible
    assert _first_stacks(r1) != _first_stacks(fresh)   # not a replay of the start
