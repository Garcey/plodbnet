"""Checkpoint versioning through scripts/train.py.

Runs the real main() in-process at miniature scale:
  - a cold v2 run saves head_version=2 + critic state and round-trips
    through a warm start;
  - a v1 checkpoint (no head_version) is refused with a clear message;
  - a v2 checkpoint missing the critic, or with a mismatched critic
    width, is refused.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from plo5bp.network import ActorCritic

_TRAIN_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
_SPEC = importlib.util.spec_from_file_location("plo5bp_train_script", _TRAIN_PATH)
train_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_mod)


def _run_train(ckpt_path: Path, extra: list[str] | None = None) -> None:
    argv = [
        "train.py",
        "--num-updates", "1",
        "--num-envs", "2",
        "--rollout-length", "32",
        "--hidden-dim", "16",
        "--num-layers", "1",
        "--critic-hidden-dim", "32",
        "--critic-num-blocks", "1",
        "--device", "cpu",
        "--no-batched",
        "--block-rotation", "",
        "--checkpoint-every", "0",
        "--snapshot-every", "1000",
        "--checkpoint", str(ckpt_path),
        *(extra or []),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        train_mod.main()
    finally:
        sys.argv = old_argv


def test_v2_checkpoint_roundtrip(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "anchor_test.pt"
    _run_train(ckpt_path)
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["head_version"] == 2
    assert "critic" in ckpt
    assert "anchor_head.weight" in ckpt["model"]
    assert "raise_head.weight" not in ckpt["model"]
    assert int(ckpt["config"]["critic_hidden_dim"]) == 32
    assert int(ckpt["config"]["critic_num_blocks"]) == 1

    # Warm start from the v2 checkpoint completes (model + critic load).
    warm_out = tmp_path / "anchor_warm.pt"
    _run_train(warm_out, extra=["--load-checkpoint", str(ckpt_path)])
    assert warm_out.exists()

    # Mismatched critic width is refused before training starts.
    with pytest.raises(SystemExit, match="critic_hidden_dim"):
        _run_train(
            tmp_path / "never.pt",
            extra=[
                "--load-checkpoint", str(ckpt_path),
                "--critic-hidden-dim", "64",
            ],
        )


def test_v1_checkpoint_refused(tmp_path: Path) -> None:
    v1 = ActorCritic(hidden_dim=16, num_layers=1)
    v1_path = tmp_path / "optimized_old.pt"
    torch.save(
        {
            "model": v1.state_dict(),
            "config": {"hidden_dim": 16, "num_layers": 1},
            "gate_count": 3,
        },
        v1_path,
    )
    with pytest.raises(SystemExit, match="head_version"):
        _run_train(tmp_path / "never.pt", extra=["--load-checkpoint", str(v1_path)])


def test_v2_checkpoint_without_critic_refused(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "anchor_base.pt"
    _run_train(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    del ckpt["critic"]
    stripped = tmp_path / "anchor_nocritic.pt"
    torch.save(ckpt, stripped)
    with pytest.raises(SystemExit, match="critic"):
        _run_train(tmp_path / "never.pt", extra=["--load-checkpoint", str(stripped)])
