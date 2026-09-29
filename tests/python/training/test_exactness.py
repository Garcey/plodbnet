"""The owner's exactness rule, checked on every test run (ML-025 / TEST-017).

`plo5bp.exactness` trains a tiny recipe with two versions of the code and
compares the SHA-256 of every tensor of every checkpoint + optimizer sidecar.
Here:

- the digest logic itself (a one-bit change is caught, metadata is not);
- the "smoke" recipe (the vSix6 flag set at toy size, CPU): when training
  sources have uncommitted changes, the working tree must reproduce HEAD's
  checkpoints bit for bit; otherwise the working tree must reproduce itself
  (determinism). A change that is MEANT to move numbers belongs behind a
  default-off flag (docs/training.md "Second efficiency pass").
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from plo5bp import exactness as ex

REPO = Path(__file__).resolve().parents[3]


def _ckpt(scale: float = 1.0) -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        "model": {"w": torch.randn(4, 3, generator=g) * scale,
                  "b": torch.zeros(3, dtype=torch.bfloat16)},
        "optimizer_state": {0: {"step": torch.tensor(3.0),
                                "exp_avg": torch.ones(2)}},
        "pool": [torch.arange(5), np.arange(3, dtype=np.float32)],
        "update_counter": 7,
        "wall_clock": 12.5,
    }


def test_digests_cover_every_tensor_and_ignore_metadata() -> None:
    d = ex.tensor_digests(_ckpt())
    assert set(d) == {
        "model/w", "model/b", "optimizer_state/0/step",
        "optimizer_state/0/exp_avg", "pool[0]", "pool[1]",
    }
    other = _ckpt()
    other["wall_clock"] = 99.0          # metadata: ignored
    other["update_counter"] = 8
    assert ex.tensor_digests(other) == d


def test_a_one_bit_change_is_caught() -> None:
    a, b = _ckpt(), _ckpt()
    w = b["model"]["w"]
    bits = w.view(torch.int32)
    bits[1, 2] ^= 1                     # one ulp in one float
    diffs = ex.compare_digests({"x.pt": ex.tensor_digests(a)},
                               {"x.pt": ex.tensor_digests(b)})
    assert diffs == ["x.pt:model/w: differs"]
    # dtype and shape are part of the digest too
    c = _ckpt()
    c["model"]["b"] = torch.zeros(3, dtype=torch.float32)
    assert ex.tensor_digests(c)["model/b"] != ex.tensor_digests(a)["model/b"]


def test_missing_files_and_tensors_are_reported(tmp_path) -> None:
    ref = {"x.pt": {"a": "1", "b": "2"}, "x_1.pt": {"a": "1"}}
    new = {"x.pt": {"a": "1", "c": "3"}}
    diffs = ex.compare_digests(ref, new)
    assert "x.pt:b: missing in the new code" in diffs
    assert "x.pt:c: new tensor (not in the reference)" in diffs
    assert "x_1.pt: written by the reference only" in diffs


def test_max_diff_is_reported_from_the_files(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    torch.save(_ckpt(), tmp_path / "a" / "x.pt")
    torch.save(_ckpt(scale=1.5), tmp_path / "b" / "x.pt")
    ref = ex.checkpoint_digests(tmp_path / "a")
    new = ex.checkpoint_digests(tmp_path / "b")
    diffs = ex.compare_digests(ref, new, tmp_path / "a", tmp_path / "b")
    assert len(diffs) == 1 and diffs[0].startswith("x.pt:model/w: differs (max |diff| ")


def _git_ok() -> bool:
    try:
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                       capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def test_training_is_bit_exact(tmp_path) -> None:
    """HEAD vs the working tree when training code changed, else the working
    tree twice. ~10 s on CPU (the two runs go in parallel)."""
    if not ex.engine_binaries(REPO / "python" / "plo5bp"):
        pytest.skip("no compiled engine in python/plo5bp")
    changed = ex.training_changes(REPO) if _git_ok() else []
    if changed:
        try:
            ref = ex.export_git_tree("HEAD", tmp_path / "head")
        except (OSError, subprocess.CalledProcessError) as e:
            pytest.skip(f"cannot export HEAD: {e}")
        what = f"HEAD vs working tree (changed: {', '.join(changed)})"
    else:
        ref, what = REPO, "working tree twice (determinism)"
    try:
        res = ex.check("smoke", ref, REPO, tmp_path / "runs")
    except ex.RunFailed as e:
        if e.side == "ref":
            # HEAD's Python cannot run against this tree's engine binary (an
            # engine API change in flight): nothing to compare against.
            pytest.skip(f"reference run failed: {str(e)[-600:]}")
        raise
    assert res.identical, (
        f"{what}: the smoke recipe's checkpoints moved -- a default training "
        "path is no longer bit-exact. Put the change behind a default-off flag "
        "or confirm it with the owner. Differences:\n  " + "\n  ".join(res.differences)
    )
