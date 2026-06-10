"""CentralCritic input plumbing.

Pins the hero-rotation convention for opponent hole blocks — slot j =
seat (actor + 1 + j) % num_seats, 255 padding past num_seats - 1 —
which must match the encoder's active-seat rotation (relative seat
r = (seat - actor) % num_seats, so opponent slot j ↔ relative seat
j + 1). Also covers the (B, 5, 5) → (B, 260) multi-hot expansion and
critic forward shapes across seat counts.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.network import CentralCritic, opp_holes_multihot
from plo5bp.rollout import _rotate_opp_holes, _rotate_opp_holes_batch


@pytest.mark.parametrize("n_seats", [2, 3, 4, 5, 6])
def test_rotation_convention(n_seats: int) -> None:
    # holes[seat] = a distinctive 5-card row per seat.
    holes = np.arange(n_seats * 5, dtype=np.uint8).reshape(n_seats, 5)
    for actor in range(n_seats):
        out = _rotate_opp_holes(holes, actor)
        assert out.shape == (5, 5) and out.dtype == np.uint8
        for j in range(5):
            if j + 1 < n_seats:
                seat = (actor + 1 + j) % n_seats
                assert (out[j] == holes[seat]).all()
            else:
                assert (out[j] == 255).all()


@pytest.mark.parametrize("n_seats", [2, 3, 4, 5, 6])
def test_rotation_batch_matches_serial(n_seats: int) -> None:
    rng = np.random.default_rng(0)
    n_envs = 7
    cache = rng.integers(0, 52, size=(n_envs, n_seats, 5)).astype(np.uint8)
    env_idx = np.array([0, 2, 5, 6, 2], dtype=np.int64)
    actors = np.array(
        [0, n_seats - 1, 1 % n_seats, 0, n_seats // 2], dtype=np.int64
    )
    got = _rotate_opp_holes_batch(cache, env_idx, actors)
    assert got.shape == (5, 5, 5) and got.dtype == np.uint8
    for r in range(len(env_idx)):
        expect = _rotate_opp_holes(cache[int(env_idx[r])], int(actors[r]))
        assert (got[r] == expect).all()


def test_multihot_expansion() -> None:
    holes = torch.full((3, 5, 5), 255, dtype=torch.uint8)
    holes[0, 0] = torch.tensor([0, 13, 26, 39, 51], dtype=torch.uint8)
    holes[1, 2] = torch.tensor([5, 6, 7, 8, 9], dtype=torch.uint8)
    mh = opp_holes_multihot(holes)
    assert mh.shape == (3, 260)
    assert mh.dtype == torch.float32

    row0 = mh[0].view(5, 52)
    for c in (0, 13, 26, 39, 51):
        assert row0[0, c] == 1.0
    assert row0[0].sum() == 5.0
    assert row0[1:].sum() == 0.0  # padding slots set no bits

    row1 = mh[1].view(5, 52)
    assert row1[2].sum() == 5.0 and (row1[2, 5:10] == 1.0).all()

    assert mh[2].sum() == 0.0  # all-padding row → all zeros


@pytest.mark.parametrize("n_seats", [2, 3, 4, 5, 6])
def test_critic_forward_shapes(n_seats: int) -> None:
    torch.manual_seed(0)
    critic = CentralCritic(hidden_dim=64, num_blocks=1)
    b = 9
    obs = torch.randn(b, critic.torso[0][0].in_features - 260)
    holes = np.full((b, n_seats, 5), 255, dtype=np.uint8)
    rng = np.random.default_rng(1)
    holes[:, :, :] = rng.integers(0, 52, size=(b, n_seats, 5))
    rot = _rotate_opp_holes_batch(
        holes, np.arange(b, dtype=np.int64) % b,
        rng.integers(0, n_seats, size=b),
    )
    v = critic(obs, opp_holes_multihot(torch.from_numpy(rot)))
    assert v.shape == (b,)
    assert torch.isfinite(v).all()
