"""UI model loading: head-version sniffing + obs-width adaptation.

The encoder now always emits OBS_DIM=991; v1-era checkpoints were
trained at 959. obs_adapter must hand those models the exact downgrade
projection while leaving current-width models (fresh v1 nets included)
untouched. _load_model must build the class the state dict was saved
from.
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.encoding import OBS_DIM, OBS_DIM_V1
from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.network import (
    ActorCritic,
    ActorCriticV2,
    CentralCritic,
    build_critic_from_state_dict,
    model_class_for_state_dict,
    obs_adapter,
)


def test_model_class_sniffing():
    v1 = ActorCritic(hidden_dim=8, num_layers=1)
    v2 = ActorCriticV2(hidden_dim=8, num_layers=1)
    assert model_class_for_state_dict(v1.state_dict()) is ActorCritic
    assert model_class_for_state_dict(v2.state_dict()) is ActorCriticV2


def test_obs_adapter_widths():
    obs = np.arange(OBS_DIM, dtype=np.float32)
    # Old checkpoint shape (959 inputs) → exact downgrade.
    old = ActorCritic(hidden_dim=8, obs_dim=OBS_DIM_V1, num_layers=1)
    out = obs_adapter(old)(obs)
    assert out.shape == (OBS_DIM_V1,)
    # Current-width models (v1 or v2) → identity.
    for m in (
        ActorCritic(hidden_dim=8, num_layers=1),
        ActorCriticV2(hidden_dim=8, num_layers=1),
        # num_layers >= 3 wraps the first Linear in a Sequential block.
        ActorCritic(hidden_dim=8, num_layers=3),
    ):
        assert obs_adapter(m)(obs).shape == (OBS_DIM,)
    # Residual-torso old checkpoint shape also detected.
    old3 = ActorCritic(hidden_dim=8, obs_dim=OBS_DIM_V1, num_layers=3)
    assert obs_adapter(old3)(obs).shape == (OBS_DIM_V1,)
    # Batch dims pass through.
    batch = np.zeros((4, OBS_DIM), dtype=np.float32)
    assert obs_adapter(old)(batch).shape == (4, OBS_DIM_V1)


def test_server_load_model_dual_path(tmp_path, monkeypatch):
    import plo5bp.ui.server as server

    v1 = ActorCritic(hidden_dim=16, num_layers=1)
    v2 = ActorCriticV2(hidden_dim=16, num_layers=1)
    p1 = tmp_path / "v1.pt"
    p2 = tmp_path / "v2.pt"
    torch.save(
        {"model": v1.state_dict(),
         "config": {"hidden_dim": 16, "num_layers": 1}},
        p1,
    )
    torch.save(
        {"model": v2.state_dict(), "head_version": 2,
         "critic": {"ignored": True},
         "config": {"hidden_dim": 16, "num_layers": 1}},
        p2,
    )

    monkeypatch.setenv("PLO5BP_CHECKPOINT", str(p1))
    m1, loaded1 = server._load_model()
    assert type(m1) is ActorCritic
    assert getattr(m1, "head_version", 1) == 1
    assert loaded1 is True

    monkeypatch.setenv("PLO5BP_CHECKPOINT", str(p2))
    m2, loaded2 = server._load_model()
    assert type(m2) is ActorCriticV2
    assert m2.head_version == 2
    assert loaded2 is True


def test_server_load_model_v1_era_959_checkpoint(tmp_path, monkeypatch):
    """A v1 checkpoint trained BEFORE the +32 pot-fraction dims (input
    width 959 — the live stub.pt) must load with its real weights, not
    silently fall back to random init because the constructor defaulted
    to the current OBS_DIM."""
    import plo5bp.ui.server as server

    old = ActorCritic(hidden_dim=16, obs_dim=OBS_DIM_V1, num_layers=1)
    p = tmp_path / "v1_959.pt"
    torch.save(
        {"model": old.state_dict(),
         "config": {"hidden_dim": 16, "num_layers": 1}},
        p,
    )
    monkeypatch.setenv("PLO5BP_CHECKPOINT", str(p))
    m, _loaded = server._load_model()
    w_loaded = m.torso[0].weight.detach()
    assert w_loaded.shape == (16, OBS_DIM_V1)
    assert torch.equal(w_loaded, old.torso[0].weight.detach())
    # And the matching adapter feeds it 959-wide observations.
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    assert obs_adapter(m)(obs).shape == (OBS_DIM_V1,)


def test_build_critic_from_state_dict_widths():
    """The builder reconstructs whatever width/depth the critic was saved
    at — PLO 991 and NLH 995 — with weights intact and a working forward."""
    for obs_dim, blocks in ((OBS_DIM, 2), (OBS_DIM_NLH, 1)):
        src = CentralCritic(obs_dim=obs_dim, hidden_dim=32, num_blocks=blocks)
        crit = build_critic_from_state_dict(src.state_dict())
        assert crit.obs_dim == obs_dim
        assert torch.equal(
            crit.torso[0][0].weight.detach(), src.torso[0][0].weight.detach()
        )
        out = crit(torch.zeros(3, obs_dim), torch.zeros(3, 5 * 52))
        assert out.shape == (3,)


def test_server_load_critic_variant_widths(tmp_path, monkeypatch):
    """Regression (prod hit 2026-07-04..05): the NLH stub's 995-wide critic
    shape-failed on every boot because _load_critic constructed at the PLO
    width; it must load now, and a genuine width/variant mismatch must
    degrade to None (true-EV off) rather than raise."""
    import plo5bp.ui.server as server

    nlh_critic = CentralCritic(obs_dim=OBS_DIM_NLH, hidden_dim=32, num_blocks=2)
    p = tmp_path / "nlh.pt"
    torch.save(
        {"head_version": 3, "critic": nlh_critic.state_dict(), "config": {}}, p
    )
    monkeypatch.setenv("PLO5BP_CHECKPOINT_NLH", str(p))
    crit = server._load_critic(torch.device("cpu"), server.VARIANT_NLH)
    assert crit is not None and crit.obs_dim == OBS_DIM_NLH
    assert torch.equal(
        crit.torso[0][0].weight.detach(),
        nlh_critic.torso[0][0].weight.detach(),
    )

    # The same file served as the PLO format trips the serve-width guard.
    monkeypatch.setenv("PLO5BP_CHECKPOINT", str(p))
    assert server._load_critic(torch.device("cpu"), server.VARIANT_PLO5) is None
