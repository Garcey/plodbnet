"""The obs-X tail the site serves (2026-10-05): the vSix7 lineage was trained with 75 extra
observation dims (the training worktree's batched encoder, groups RUN and LINE); serving it
means the site's single-table env must hand the network the very same numbers.

- fixtures/obs_x_trainer.npz holds the TRAINER's observations (full 1246-wide rows) along
  seeded hands played with random legal actions (fixtures/make_obs_x_fixture.py); replaying
  those hands through `BombPotEnv(obs_x_groups=5)` must give every value bit for bit -- the
  1171 base columns and the tail;
- off (the default) nothing changes: 1171 columns, the same values;
- the serving switch (`encoding.set_serving_obs_x`) is what the env follows when it isn't
  told, and only the engine's groups can be asked for;
- the model adapter: an obs-X network keeps its own groups' columns; an older 1171 network
  takes the prefix of a 1246 row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import plo5bp.encoding as E
import plo5bp.encoding_nlh as EN
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv

FIXTURE = Path(__file__).parent / "fixtures" / "obs_x_trainer.npz"


@pytest.fixture
def rev1(monkeypatch: pytest.MonkeyPatch) -> None:
    # (the fixture is at observation revision 1, production's; Rust reads it at construction)
    monkeypatch.setenv(E.OBS_REV_ENV, "1")
    monkeypatch.setattr(E, "OBS_SEMANTICS_REV", 1)
    monkeypatch.setattr(EN, "OBS_SEMANTICS_REV", 1)


@pytest.fixture(autouse=True)
def _serving_off():
    yield
    E.set_serving_obs_x(0)


def _hands(d):
    """The fixture's hands: (seats, stacks, seed, button, [(gate, chips, obs)]) in order."""
    out: list = []
    key = None
    for i in range(len(d["gate"])):
        k = (int(d["cfg"][i]), int(d["hand"][i]))
        if k != key:
            key = k
            n = int(d["seats"][k[0]])
            out.append((n, tuple(int(x) for x in d["stacks"][k[0]][:n]), int(d["seed"][i]),
                        int(d["button"][i]), []))
        out[-1][4].append((int(d["gate"][i]), int(d["chips"][i]), d["obs"][i]))
    return out


def test_the_site_encodes_the_tail_the_network_was_trained_on(rev1):
    d = np.load(FIXTURE)
    hands = _hands(d)
    assert len(hands) == 30 and sum(len(h[4]) for h in hands) == len(d["gate"])
    checked = 0
    for n, stacks, seed, button, steps in hands:
        env = BombPotEnv(GameConfig(num_seats=n, starting_stacks=stacks), obs_x_groups=5)
        obs, info = env.reset(seed, button)
        for gate, chips, want in steps:
            assert obs.shape == (E.OBS_DIM + E.OBS_X_DIM,) == (env.obs_dim,)
            bad = np.nonzero(obs.view(np.uint32) != want.view(np.uint32))[0]
            assert bad.size == 0, f"seed {seed} decision {checked}: columns {bad[:8].tolist()}"
            checked += 1
            obs, _r, _done, info = env.step_hybrid(gate, chips)
    assert checked == len(d["gate"])
    tails = d["obs"][:, E.OBS_DIM:]
    assert (tails[:, :7] != 0).mean() > 0.9 and (tails[:, 17:63] != 0).any()  # (both groups exercised)


def test_off_by_default_and_the_serving_switch(rev1):
    env = BombPotEnv(GameConfig())
    obs, _ = env.reset(7, 0)
    assert obs.shape == (E.OBS_DIM,) and env.obs_dim == E.OBS_DIM
    E.set_serving_obs_x(5)
    on, _ = BombPotEnv(GameConfig()).reset(7, 0)
    assert on.shape == (E.OBS_DIM + E.OBS_X_DIM,)
    assert np.array_equal(on[: E.OBS_DIM], obs)  # (the base row is the same)
    assert (on[E.OBS_DIM:E.OBS_DIM + 7] != 0).any()
    # an env told its groups ignores the switch; minimal / NLH envs never take the tail
    assert BombPotEnv(GameConfig(), obs_x_groups=0).reset(7, 0)[0].shape == (E.OBS_DIM,)
    assert BombPotEnv(GameConfig(), obs_mode="minimal").reset(7, 0)[0].shape == (E.OBS_DIM_MINIMAL,)
    with pytest.raises(ValueError):
        E.set_serving_obs_x(2)  # (RANGE: a trainer experiment the engine doesn't compute)
    with pytest.raises(ValueError):
        BombPotEnv(GameConfig(), obs_mode="minimal", obs_x_groups=5)


def test_the_adapter_reads_each_networks_own_columns():
    from plo5bp.network import ActorCriticV5, obs_adapter

    wide = ActorCriticV5(hidden_dim=16, num_layers=1, obs_dim=E.OBS_DIM + E.OBS_X_DIM)
    wide.obs_x_groups = 1  # (trained with RUN only)
    narrow = ActorCriticV5(hidden_dim=16, num_layers=1, obs_dim=E.OBS_DIM)
    row = np.arange(E.OBS_DIM + E.OBS_X_DIM, dtype=np.float32)[None] + 1.0
    got = obs_adapter(wide)(row)
    lo, hi = E.OBS_X_SLICES["line"]
    assert np.array_equal(got[0, : E.OBS_DIM + 7], row[0, : E.OBS_DIM + 7])
    assert not got[0, E.OBS_DIM + lo:E.OBS_DIM + hi].any()  # (LINE wasn't trained: zero)
    assert np.array_equal(obs_adapter(narrow)(row), row[:, : E.OBS_DIM])
    with pytest.raises(ValueError):
        obs_adapter(wide)(row[:, : E.OBS_DIM])  # (the tail can't be made up)
    with torch.no_grad():  # (and the shapes really fit the networks)
        wide.torso(torch.from_numpy(obs_adapter(wide)(row)))
        narrow.torso(torch.from_numpy(obs_adapter(narrow)(row)))
