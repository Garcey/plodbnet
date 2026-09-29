"""Opponent-pool anchors (--pool-anchors, 2026-09-28, ML-033).

The FIFO pool (8 snapshots every --snapshot-every updates) only remembers the
last ~40 updates, which lets self-play drift (the 2026-09-26 deep dive saw
the edge over OLDER references fall). Anchors add the run's numbered
checkpoints nearest a few ages back; the FIFO is untouched.
"""

from __future__ import annotations

import sys

import torch

from plo5bp.network import ActorCriticV5
from plo5bp.selfplay import OpponentPool, refresh_pool_anchors, select_anchor_updates
from plo5bp.train import loop


def test_anchor_selection() -> None:
    files = list(range(1, 400))
    # nearest to current - age on the snapshot grid, oldest first
    assert select_anchor_updates(files, 399, [10, 100, 300], 5) == [95, 295, 385]
    # never the FIFO's members, never twice, never the future
    assert select_anchor_updates(files, 399, [10], 5, exclude={385}) == [384]
    assert select_anchor_updates([1, 2, 3], 3, [1, 2, 5], 1) == [1, 2]
    # a young run has fewer anchors; no files -> none
    assert select_anchor_updates(files[:20], 20, [10, 100], 5) == [10]
    assert select_anchor_updates([], 50, [10], 5) == []


def test_anchors_ride_beside_the_fifo(tmp_path) -> None:
    ref = ActorCriticV5(hidden_dim=8, num_layers=1)
    for n in range(1, 13):
        m = ActorCriticV5(hidden_dim=8, num_layers=1)
        torch.save({"model": m.state_dict(), "variant": "plo5_double_bomb",
                    "head_version": 4, "obs_rev": 2}, tmp_path / f"x_{n}.pt")
    torch.save({"model": {"bad": torch.zeros(1)}, "variant": "plo5_double_bomb",
                "head_version": 4, "obs_rev": 2}, tmp_path / "x_20.pt")
    pool = OpponentPool(capacity=3)
    for tag in (10, 11, 12):
        pool.snapshot(ref, tag=tag)
    got = refresh_pool_anchors(
        pool, tmp_path / "x.pt", 12, [2, 5, 8], 1, "plo5_double_bomb", 4,
        ref.state_dict(), expected_obs_rev=2,
    )
    # ages 8 / 5 -> updates 4 / 7; age 2 -> 10 is in the FIFO, so the nearest
    # older member outside it, 9
    assert got == [4, 7, 9] and pool.anchor_tags == [4, 7, 9]
    assert pool.fifo_tags == [10, 11, 12] and pool.tags == [10, 11, 12, 4, 7, 9]
    assert len(pool) == len(pool.snapshots) == 6
    # the FIFO still evicts its own oldest only
    pool.snapshot(ref, tag=13)
    assert pool.fifo_tags == [11, 12, 13] and pool.anchor_tags == [4, 7, 9]
    # unchanged choice: the loaded weights are kept, not re-read
    kept = pool.snapshots[3]
    refresh_pool_anchors(pool, tmp_path / "x.pt", 12, [2, 5, 8], 1,
                         "plo5_double_bomb", 4, ref.state_dict(), expected_obs_rev=2)
    assert pool.snapshots[3] is kept
    # an unfit file (another shape) never joins
    refresh_pool_anchors(pool, tmp_path / "x.pt", 21, [1], 1,
                         "plo5_double_bomb", 4, ref.state_dict(), expected_obs_rev=2)
    assert pool.anchor_tags == []


def test_anchors_through_the_training_loop(tmp_path, capsys) -> None:
    argv = [
        "train.py", "--variant", "plo5_double_bomb", "--v6", "--obs-mode", "minimal",
        "--hidden-dim", "16", "--num-layers", "3",
        "--critic-hidden-dim", "16", "--critic-num-blocks", "1",
        "--no-grad-checkpoint", "--device", "cpu",
        "--num-envs", "40", "--rollout-length", "300",
        "--num-minibatches", "2", "--ppo-epochs", "1",
        "--mix-configs", "--configs-per-tier", "1", "--cpu-threads", "2",
        "--snapshot-every", "1", "--checkpoint-every", "1", "--seed", "3",
        "--number-by-count", "--pool-anchors", "8",
        "--num-updates", "10", "--checkpoint", str(tmp_path / "a.pt"),
        "--run-dir", str(tmp_path / "runs"),
    ]
    old = sys.argv
    sys.argv = argv
    try:
        loop.main()
    finally:
        sys.argv = old
    out = capsys.readouterr().out
    # count 9: FIFO 2..9, files 1..8 -> the anchor 8 back is update 1; count
    # 10: FIFO 3..10 -> update 2 (the snapshot comes before that update's save)
    assert "[pool] anchors: updates [1]" in out and "[pool] anchors: updates [2]" in out
    final = torch.load(tmp_path / "a.pt", map_location="cpu", weights_only=False)
    # the checkpoint records the FIFO only (a relaunch rebuilds it exactly)
    assert final["pool_member_updates"] == list(range(3, 11))
    assert final["config"]["pool_anchor_ages"] == (8,)
