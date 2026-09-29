"""--number-by-count (2026-09-28, ML-063): relaunch bookkeeping.

The legacy ("index") numbering stamps a numbered save with the 0-based index
of the update that just finished and the final save with the count, never
saves a launch's first update, and after a relaunch its numbers fall one
behind the weights (the pool snapshot of local update 0 carries the loaded
file's own tag with newer weights). --number-by-count stamps the COUNT
everywhere; a stem never mixes the two (a relaunch follows its file).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from plo5bp.train import loop
from plo5bp.train.checkpoint import (
    NUMBERING_COUNT,
    NUMBERING_INDEX,
    is_own_stem,
    resolve_numbering,
)

ARGS = [
    "--variant", "plo5_double_bomb", "--v6", "--obs-mode", "minimal",
    "--hidden-dim", "16", "--num-layers", "3",
    "--critic-hidden-dim", "16", "--critic-num-blocks", "1",
    "--no-grad-checkpoint", "--device", "cpu",
    "--num-envs", "40", "--rollout-length", "400",
    "--num-minibatches", "2", "--ppo-epochs", "1",
    "--mix-configs", "--configs-per-tier", "1", "--cpu-threads", "2",
    "--snapshot-every", "1", "--checkpoint-every", "1", "--seed", "3",
]


def _main(argv: list[str]) -> None:
    old = sys.argv
    sys.argv = ["train.py", *argv]
    try:
        loop.main()
    finally:
        sys.argv = old


def _load(p: Path) -> dict:
    return torch.load(p, map_location="cpu", weights_only=False)


def _same_weights(a: dict, b: dict) -> bool:
    return all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


def test_own_stem_and_numbering_resolution() -> None:
    assert is_own_stem("checkpoints/vSix6_1300.pt", "checkpoints/vSix6.pt")
    assert is_own_stem("/elsewhere/vSix6.pt", "checkpoints/vSix6.pt")
    assert not is_own_stem("checkpoints/vSix5_1290.pt", "checkpoints/vSix6.pt")
    assert not is_own_stem("checkpoints/vSix6x_12.pt", "checkpoints/vSix6.pt")
    # cold start / new stem: the flag decides
    assert resolve_numbering(True, None, "x.pt", None) is True
    assert resolve_numbering(False, "other_4.pt", "x.pt", NUMBERING_COUNT) is False
    # relaunch: the loaded file decides; switching mid-stem is refused
    assert resolve_numbering(False, "x_4.pt", "x.pt", NUMBERING_COUNT) is True
    assert resolve_numbering(False, "x_4.pt", "x.pt", None) is False
    with pytest.raises(SystemExit, match="stem boundary"):
        resolve_numbering(True, "x_4.pt", "x.pt", NUMBERING_INDEX)


def test_count_numbering_through_a_relaunch(tmp_path, capsys) -> None:
    ck = tmp_path / "c" / "x.pt"
    runs = ["--run-dir", str(tmp_path / "runs")]
    _main([*ARGS, "--number-by-count", "--num-updates", "3", "--checkpoint", str(ck), *runs])
    d = ck.parent
    # the first update is saved; file N = the weights after N updates
    assert sorted(p.name for p in d.glob("x_*.pt") if "optim" not in p.name) == [
        "x_1.pt", "x_2.pt", "x_3.pt"]
    for n in (1, 2, 3):
        c = _load(d / f"x_{n}.pt")
        assert c["update_counter"] == n and c["numbering"] == NUMBERING_COUNT
    final = _load(ck)
    assert final["update_counter"] == 3 and _same_weights(final, _load(d / "x_3.pt"))
    # every snapshot's tag names the file holding its weights
    assert final["pool_member_updates"] == [1, 2, 3]

    # A relaunch WITHOUT the flag follows the stem's numbering: the next files
    # are 4 and 5 (the old numbering would have skipped one and fallen behind),
    # and the sidecar written with x_3 restores its Adam moments.
    capsys.readouterr()
    _main([*ARGS, "--num-updates", "2", "--load-checkpoint", str(d / "x_3.pt"),
           "--checkpoint", str(ck), *runs])
    log = capsys.readouterr().out
    assert "[optim] restored Adam moments from x.optim.pt" in log, log[-2000:]
    assert "numbering by update COUNT" in log
    assert (d / "x_4.pt").exists() and (d / "x_5.pt").exists()
    final = _load(ck)
    assert final["update_counter"] == 5 and final["numbering"] == NUMBERING_COUNT
    assert _same_weights(final, _load(d / "x_5.pt"))
    assert final["pool_member_updates"][-2:] == [4, 5]


def test_an_index_numbered_stem_cannot_switch_midway(tmp_path) -> None:
    ck = tmp_path / "y.pt"
    runs = ["--run-dir", str(tmp_path / "runs")]
    _main([*ARGS, "--num-updates", "2", "--checkpoint", str(ck), *runs])
    assert _load(ck)["numbering"] == NUMBERING_INDEX
    assert (tmp_path / "y_1.pt").exists() and not (tmp_path / "y_0.pt").exists()
    with pytest.raises(SystemExit, match="stem boundary"):
        _main([*ARGS, "--number-by-count", "--num-updates", "1",
               "--load-checkpoint", str(tmp_path / "y_1.pt"), "--checkpoint", str(ck),
               *runs])
