"""Warm-start opponent-pool reconstruction.

The pool is ephemeral (state dicts never ride in checkpoints), so a
stop/resume used to lose every opponent. These tests pin the rebuild:
`select_warmstart_pool_updates` picks the members a never-stopped run
would hold (exact prior membership when recorded, natural snapshot grid
otherwise, walk-back fill when the disk cadence is coarse), and
`seed_pool_from_checkpoints` loads them from `<stem>_<N>.pt` siblings
while refusing incompatible files.
"""

import numpy as np
import torch

from plo5bp.network import ActorCriticV2
from plo5bp.selfplay import (
    OpponentPool,
    discover_checkpoint_family,
    seed_pool_from_checkpoints,
    select_warmstart_pool_updates,
)


# ---------------------------------------------------------------------------
# Pure selection logic
# ---------------------------------------------------------------------------


def test_selection_exact_grid_matches_never_stopped_pool():
    # The motivating example: pool of 8, snapshot every 5, resume at 500,
    # files at every multiple of 5 → exactly 465..500.
    available = list(range(0, 505, 5))
    got = select_warmstart_pool_updates(
        available, target_update=500, capacity=8, snapshot_every=5
    )
    assert got == [465, 470, 475, 480, 485, 490, 495, 500]


def test_selection_coarse_files_walk_back_to_fill():
    # Files every 50 but snapshots every 10: the natural grid (430..500)
    # only holds two distinct files (450, 500) — the walk continues back
    # so the pool still fills with the most-recent distinct files.
    available = list(range(0, 550, 50))
    got = select_warmstart_pool_updates(
        available, target_update=500, capacity=8, snapshot_every=10
    )
    assert got == [150, 200, 250, 300, 350, 400, 450, 500]


def test_selection_prefers_recorded_membership():
    available = list(range(0, 505, 5))
    preferred = [500, 490, 480, 470]
    got = select_warmstart_pool_updates(
        available, 500, capacity=6, snapshot_every=5, preferred=preferred
    )
    # The 4 recorded members are honored exactly; the grid fills the rest
    # with the nearest unused files.
    assert set(preferred).issubset(set(got))
    assert len(got) == 6
    assert got == sorted(got)


def test_selection_ignores_future_and_missing():
    # Updates past the target are another (newer) run's — never seeded.
    available = [480, 490, 500, 510, 520]
    got = select_warmstart_pool_updates(
        available, 500, capacity=8, snapshot_every=5
    )
    assert got == [480, 490, 500]
    assert select_warmstart_pool_updates([], 500, 8, 5) == []
    # Preferred entries with no file on disk fall through to the grid.
    got2 = select_warmstart_pool_updates(
        [490, 500], 500, capacity=2, snapshot_every=5, preferred=[499, 481]
    )
    assert got2 == [490, 500]


def test_selection_nearest_tie_prefers_newer():
    got = select_warmstart_pool_updates(
        [485, 495], 500, capacity=1, snapshot_every=10
    )
    # Grid point 490 is equidistant from 485 and 495 → newer wins.
    assert got == [495]


# ---------------------------------------------------------------------------
# Pool mechanics
# ---------------------------------------------------------------------------


def test_pool_seed_fifo_and_tags():
    pool = OpponentPool(capacity=3)
    for u in (10, 20, 30):
        pool.seed({"w": torch.zeros(1) + u}, tag=u)
    assert pool.tags == [10, 20, 30]
    pool.seed({"w": torch.zeros(1) + 40}, tag=40)
    assert pool.tags == [20, 30, 40], "capacity evicts the oldest"
    assert len(pool) == 3
    sd = pool.sample()
    assert sd is not None and "w" in sd


def test_pool_snapshot_tags_default_unknown():
    net = ActorCriticV2(obs_dim=32, hidden_dim=16, num_layers=1)
    pool = OpponentPool(capacity=2)
    pool.snapshot(net)
    pool.snapshot(net, tag=7)
    assert pool.tags == [-1, 7]


# ---------------------------------------------------------------------------
# End-to-end seeding from real checkpoint files
# ---------------------------------------------------------------------------


def _tiny_net(seed: int) -> ActorCriticV2:
    torch.manual_seed(seed)
    return ActorCriticV2(obs_dim=32, hidden_dim=16, num_layers=1)


def _write_ckpt(path, net, variant="plo5_double_bomb", head_version=2, update=0):
    torch.save(
        {
            "model": net.state_dict(),
            "variant": variant,
            "head_version": head_version,
            "update_counter": update,
        },
        path,
    )


def test_discover_family_parses_numbered_siblings(tmp_path):
    net = _tiny_net(0)
    for u in (5, 10, 15):
        _write_ckpt(tmp_path / f"run_{u}.pt", net, update=u)
    _write_ckpt(tmp_path / "other_5.pt", net, update=5)
    (tmp_path / "run_final.pt").write_bytes(b"not numbered")
    base, family = discover_checkpoint_family(tmp_path / "run_10.pt")
    assert base == "run"
    assert sorted(family) == [5, 10, 15]
    # A plain (un-numbered) checkpoint uses its whole stem as the base.
    base2, family2 = discover_checkpoint_family(tmp_path / "run.pt")
    assert base2 == "run" and sorted(family2) == [5, 10, 15]


def test_seed_pool_end_to_end_with_incompatible_files(tmp_path):
    ref = _tiny_net(0)
    # Compatible members at 5-update cadence.
    for u in (480, 485, 490, 495, 500):
        _write_ckpt(tmp_path / f"vX_{u}.pt", _tiny_net(u), update=u)
    # Poison: wrong variant, wrong head, wrong architecture — all skipped.
    _write_ckpt(tmp_path / "vX_475.pt", _tiny_net(1), variant="plo6_double_bomb")
    _write_ckpt(tmp_path / "vX_470.pt", _tiny_net(2), head_version=1)
    torch.manual_seed(3)
    wrong_arch = ActorCriticV2(obs_dim=48, hidden_dim=16, num_layers=1)
    _write_ckpt(tmp_path / "vX_465.pt", wrong_arch)

    pool = OpponentPool(capacity=8)
    seeded = seed_pool_from_checkpoints(
        pool,
        tmp_path / "vX_500.pt",
        target_update=500,
        snapshot_every=5,
        expected_variant="plo5_double_bomb",
        expected_head_version=2,
        reference_state_dict=ref.state_dict(),
    )
    assert seeded == [480, 485, 490, 495, 500]
    assert pool.tags == seeded
    assert len(pool) == 5

    # Members carry the actual file weights (spot-check one tensor).
    disk = torch.load(
        tmp_path / "vX_490.pt", map_location="cpu", weights_only=False
    )["model"]
    member = pool.snapshots[pool.tags.index(490)]
    k = next(iter(disk))
    assert torch.equal(member[k], disk[k])


def test_seed_pool_respects_capacity(tmp_path):
    ref = _tiny_net(0)
    for u in range(0, 105, 5):
        _write_ckpt(tmp_path / f"vY_{u}.pt", _tiny_net(u), update=u)
    pool = OpponentPool(capacity=4)
    seeded = seed_pool_from_checkpoints(
        pool,
        tmp_path / "vY_100.pt",
        target_update=100,
        snapshot_every=5,
        expected_variant="plo5_double_bomb",
        expected_head_version=2,
        reference_state_dict=ref.state_dict(),
    )
    assert seeded == [85, 90, 95, 100]
    assert len(pool) == 4
