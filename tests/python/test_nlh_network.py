"""v4 head on the NLH anchor spec: shapes, act/evaluate parity, the
all-in atom's reachability, and checkpoint spec round-tripping."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.network import (
    ActorCriticV2,
    ActorCriticV4,
    build_actor_from_state_dict,
    state_dict_anchor_count,
)
from plo5bp.sizing import NLH_ANCHOR_SPEC, PLO_ANCHOR_SPEC


def _random_batch(n: int, obs_dim: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, obs_dim, generator=g)
    gate_mask = torch.ones(n, 3, dtype=torch.bool)
    # Deep-NLH-flavoured sizing rows: pot 45k, min-raise 20k, huge max.
    mn = torch.full((n,), 20_000, dtype=torch.int64)
    mx = torch.full((n,), 995_000, dtype=torch.int64)
    pot = torch.full((n,), 45_000, dtype=torch.int64)
    tc = torch.zeros(n, dtype=torch.int64)
    sizing = torch.stack([mn, mx, pot, tc], dim=-1)
    return obs, gate_mask, sizing


def test_v4_nlh_shapes_and_act_evaluate_parity():
    torch.manual_seed(1)
    model = ActorCriticV4(
        hidden_dim=64, obs_dim=OBS_DIM_NLH, num_layers=2,
        anchor_spec=NLH_ANCHOR_SPEC,
    )
    assert model._anchor_count == 12
    assert model.refine_head.out_features == (12 - 2) * 2

    obs, gate_mask, sizing = _random_batch(256, OBS_DIM_NLH)
    out = model.act(obs, gate_mask, sizing, deterministic=False)
    assert out.anchor.min() >= 0 and out.anchor.max() < 12

    log_prob, *_ = model.evaluate(
        obs, gate_mask, sizing, out.gate, out.anchor, out.refine_u
    )
    # PPO contract: evaluate of the stored actions reproduces the
    # sampling log-prob exactly (ratio 1.0 at epoch start).
    torch.testing.assert_close(log_prob, out.log_prob, rtol=0, atol=1e-6)


def test_v4_nlh_allin_atom_returns_max_raise_chips():
    torch.manual_seed(2)
    model = ActorCriticV4(
        hidden_dim=64, obs_dim=OBS_DIM_NLH, num_layers=2,
        anchor_spec=NLH_ANCHOR_SPEC,
    )
    obs, gate_mask, sizing = _random_batch(2048, OBS_DIM_NLH, seed=3)
    out = model.act(obs, gate_mask, sizing, deterministic=False)
    jam_rows = out.anchor == 11
    assert jam_rows.any(), "untrained net should reach the all-in atom sometimes"
    # The atom is not refinable: its chips are exactly max_raise.
    raise_jams = jam_rows & (out.gate == 2)
    if raise_jams.any():
        assert (out.chips[raise_jams] == 995_000).all()


def test_v2_default_spec_is_plo_unchanged():
    model = ActorCriticV2(hidden_dim=32, num_layers=1)
    assert model.anchor_spec is PLO_ANCHOR_SPEC
    assert model._anchor_count == 11
    assert model.anchor_head.out_features == 11
    assert model.refine_head.out_features == 18


def test_checkpoint_anchor_spec_roundtrip():
    src = ActorCriticV4(
        hidden_dim=64, obs_dim=OBS_DIM_NLH, num_layers=2,
        anchor_spec=NLH_ANCHOR_SPEC,
    )
    sd = src.state_dict()
    assert state_dict_anchor_count(sd) == 12
    rebuilt = build_actor_from_state_dict(sd, hidden_dim=64, num_layers=2)
    assert isinstance(rebuilt, ActorCriticV4)
    assert rebuilt.anchor_spec is NLH_ANCHOR_SPEC
    assert rebuilt._anchor_count == 12

    plo = ActorCriticV2(hidden_dim=32, num_layers=1)
    assert state_dict_anchor_count(plo.state_dict()) == 11
