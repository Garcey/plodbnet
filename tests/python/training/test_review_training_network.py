"""Regression tests for the 2026-09-20 code review, training workstream —
network.py items:

- A2  v4/v5 sizing head computes the discretized-logistic CDF math in fp32
      even under bf16 autocast (evaluate() == fp32 act() log-probs).
- A16 opp_holes_multihot never lets a 255 pad erase a genuine card 51.
- A19 build_critic_from_state_dict can rebuild with the TRAINED value support.
- A20 obs_adapter refuses widths that are neither the model's own nor a known
      superset layout of the SAME variant (no PLO obs sliced into an NLH model).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.actions import GATE_RAISE
from plo5bp.encoding import OBS_DIM, OBS_DIM_MINIMAL, OBS_DIM_V1, OBS_DIM_V2
from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.network import (
    ActorCritic,
    ActorCriticV2,
    ActorCriticV4,
    ActorCriticV5,
    CentralCritic,
    build_critic_from_state_dict,
    obs_adapter,
    opp_holes_multihot,
)
from plo5bp.sizing import NLH_ANCHOR_SPEC, anchor_grid_torch


# --------------------------------------------------------------------- A2
# Deep-ish spot (min 1bb, max 60bb, pot 18bb, to_call 0): all 11 PLO anchors
# legal, so the upper-tail anchors are reachable.
_SIZING_ALL_LEGAL = [10_000, 600_000, 180_000, 0]


def _pin_head(model, mu_raw, s_raw):
    """Zero the sizing head's weights and drive (mu_raw, s_raw) from the bias
    with bf16-EXACT values. The head Linear's output is then bit-identical in
    fp32 and under bf16 autocast, which isolates what A2 is about: the dtype
    of the logistic-CDF math DOWNSTREAM of the head (a generic head's bf16
    output quantization is a separate, symmetric, much smaller effect)."""
    with torch.no_grad():
        if isinstance(model, ActorCriticV5):
            k = model._mixture_k
            model.mix_head.weight.zero_()
            model.mix_head.bias.zero_()
            model.mix_head.bias[0:k] = torch.tensor(mu_raw[:k])
            model.mix_head.bias[k:2 * k] = s_raw
        else:
            model.size_head.weight.zero_()
            model.size_head.bias[0] = mu_raw[0]
            model.size_head.bias[1] = s_raw


@pytest.mark.parametrize("cls", [ActorCriticV4, ActorCriticV5])
@pytest.mark.parametrize("s_raw", [-8.0, -4.0, -2.0])  # s ~= 0.30 / 0.38 / 0.86
@pytest.mark.parametrize("mu_raw", [(-0.5, 0.25, 0.75), (0.25, -0.25, 0.5)])
def test_a2_evaluate_under_bf16_autocast_matches_fp32(cls, s_raw, mu_raw):
    torch.manual_seed(0)
    m = cls(hidden_dim=32, obs_dim=OBS_DIM, num_layers=2).eval()
    _pin_head(m, mu_raw, s_raw)
    count = m._anchor_count
    obs = torch.randn(count, OBS_DIM)
    gm = torch.ones(count, 3, dtype=torch.bool)
    sizing = torch.tensor([_SIZING_ALL_LEGAL]).repeat(count, 1)
    gates = torch.full((count,), GATE_RAISE, dtype=torch.long)
    anchors = torch.arange(count)  # EVERY anchor, incl. the upper tail
    u = torch.full((count,), 0.5)
    with torch.no_grad():
        ref = m.evaluate(obs, gm, sizing, gates, anchors, u)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ev = m.evaluate(obs, gm, sizing, gates, anchors, u)
    # The repro precondition: the head really does emit bf16 under autocast.
    assert ev[9].dtype == torch.bfloat16
    d = (ref[7] - ev[7].float()).abs()
    assert float(d.max()) < 1e-4, (
        f"anchor log-prob drifts under bf16 autocast: max|d|={float(d.max()):.3f} "
        f"per-anchor={d.tolist()}"
    )


@pytest.mark.parametrize("cls", [ActorCriticV4, ActorCriticV5])
def test_a2_sampled_act_logprobs_match_autocast_evaluate(cls):
    """The PPO ratio at ZERO policy change: fp32 act() (rollout) vs evaluate()
    under bf16 autocast (the CUDA update), sampled anchors incl. above-mu."""
    torch.manual_seed(1)
    m = cls(hidden_dim=32, obs_dim=OBS_DIM, num_layers=2).eval()
    _pin_head(m, (0.25, -0.5, 0.75), -2.0)
    b = 4096
    obs = torch.randn(b, OBS_DIM)
    gm = torch.ones(b, 3, dtype=torch.bool)
    sizing = torch.tensor([_SIZING_ALL_LEGAL]).repeat(b, 1)
    with torch.no_grad():
        out = m.act(obs, gm, sizing)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ev = m.evaluate(obs, gm, sizing, out.gate, out.anchor, out.refine_u)
    # Coverage guard: the sample must actually contain upper-tail anchors.
    assert int((out.anchor >= 8).sum()) > 50
    d = (out.anchor_log_prob - ev[7].float()).abs()
    assert float(d.max()) < 1e-4


def test_a2_anchor_dist_is_fp32_for_bf16_params():
    m4 = ActorCriticV4(hidden_dim=16, obs_dim=OBS_DIM, num_layers=2)
    m5 = ActorCriticV5(hidden_dim=16, obs_dim=OBS_DIM, num_layers=2)
    grid = anchor_grid_torch(torch.tensor([_SIZING_ALL_LEGAL]))
    p4 = m4._anchor_dist(torch.zeros(1, 2, dtype=torch.bfloat16), grid).probs
    p5 = m5._anchor_dist(torch.zeros(1, 9, dtype=torch.bfloat16), grid).probs
    assert p4.dtype == torch.float32 and p5.dtype == torch.float32
    mu, s, w = m5.mixture_params(torch.zeros(1, 9, dtype=torch.bfloat16))
    assert mu.dtype == s.dtype == w.dtype == torch.float32


# -------------------------------------------------------------------- A16
def _multihot_reference(holes: np.ndarray) -> np.ndarray:
    b = holes.shape[0]
    out = np.zeros((b, 5, 52), dtype=np.float32)
    for i in range(b):
        for j in range(5):
            for c in holes[i, j]:
                if c < 52:
                    out[i, j, int(c)] = 1.0
    return out.reshape(b, 260)


def test_a16_multihot_pad_does_not_erase_card_51():
    # NLH-style slot padded to width 5 (the old serial `_hole_cache`): the
    # genuine card 51 sits BEFORE the 255 pads, which clamped onto index 51
    # and overwrote it with 0.
    holes = np.full((3, 5, 5), 255, dtype=np.uint8)
    holes[0, 0, :2] = (51, 7)      # card 51 first, pads after
    holes[1, 2, :2] = (3, 51)      # card 51 last real card, pads after
    holes[2, 4, :4] = (51, 0, 50, 12)  # PLO4-style width-4 in a 5-wide slot
    out = opp_holes_multihot(torch.from_numpy(holes)).numpy()
    np.testing.assert_array_equal(out, _multihot_reference(holes))
    assert out[0, 0 * 52 + 51] == 1.0
    assert out[1, 2 * 52 + 51] == 1.0
    assert out[2, 4 * 52 + 51] == 1.0
    assert out.sum() == 8.0


@pytest.mark.parametrize("hole_w", [2, 4, 5, 6])
def test_a16_multihot_matches_reference_all_widths(hole_w):
    rng = np.random.default_rng(hole_w)
    b = 64
    holes = np.full((b, 5, hole_w), 255, dtype=np.uint8)
    for i in range(b):
        n_opp = int(rng.integers(1, 6))
        deck = rng.permutation(52)
        for j in range(n_opp):
            holes[i, j] = deck[j * hole_w : (j + 1) * hole_w]
    # Force card 51 into a few rows so the edge is always exercised.
    holes[0, 0, 0] = 51
    out = opp_holes_multihot(torch.from_numpy(holes))
    assert out.shape == (b, 260) and out.dtype == torch.float32
    np.testing.assert_array_equal(out.numpy(), _multihot_reference(holes))


# -------------------------------------------------------------------- A19
def _support_critic(**kw):
    torch.manual_seed(0)
    c = CentralCritic(
        obs_dim=30, hidden_dim=16, num_blocks=1, q_actions=13,
        value_bins=51, **kw,
    )
    with torch.no_grad():
        c.value_head.weight.normal_(0, 1.0)
        c.adv_head.weight.normal_(0, 0.1)
    return c


def test_a19_builder_rebuilds_trained_value_support():
    c = _support_critic(value_support=3000.0, hlgauss_sigma=0.5, q_base_raw=True)
    obs = torch.randn(64, 30)
    opp = opp_holes_multihot(torch.randint(0, 52, (64, 5, 5)).to(torch.uint8))
    rebuilt = build_critic_from_state_dict(
        c.state_dict(), q_base_raw=True,
        value_support=3000.0, value_hlgauss_sigma=0.5,
    )
    with torch.no_grad():
        v1, q1 = c.q_values(obs, opp)
        v2, q2 = rebuilt.q_values(obs, opp)
    assert torch.equal(v1, v2)
    assert torch.equal(q1, q2)
    assert torch.equal(c._raw_value_centers, rebuilt._raw_value_centers)
    assert rebuilt.hlgauss_sigma == c.hlgauss_sigma
    # And the HL-Gauss training loss agrees (sigma rides on the bin step).
    logits = torch.randn(8, 51)
    rets = torch.randn(8) * 50
    assert torch.equal(
        c.hlgauss_value_loss(logits, rets), rebuilt.hlgauss_value_loss(logits, rets)
    )


def test_a19_builder_defaults_unchanged_when_kwargs_omitted():
    # None == today's constructor defaults (1500 / 0.75): a default-support
    # checkpoint round-trips exactly with no kwargs, as before.
    c = _support_critic(q_base_raw=True)
    rebuilt = build_critic_from_state_dict(c.state_dict(), q_base_raw=True)
    assert torch.equal(c._raw_value_centers, rebuilt._raw_value_centers)
    assert rebuilt.hlgauss_sigma == c.hlgauss_sigma
    explicit = build_critic_from_state_dict(
        c.state_dict(), q_base_raw=True,
        value_support=1500.0, value_hlgauss_sigma=0.75,
    )
    assert torch.equal(explicit._raw_value_centers, rebuilt._raw_value_centers)


# -------------------------------------------------------------------- A20
def test_a20_obs_adapter_refuses_plo_obs_for_nlh_model():
    nlh = ActorCriticV4(
        hidden_dim=8, obs_dim=OBS_DIM_NLH, num_layers=2,
        anchor_spec=NLH_ANCHOR_SPEC,
    )
    adapt = obs_adapter(nlh)
    ok = np.zeros(OBS_DIM_NLH, dtype=np.float32)
    assert adapt(ok) is ok  # its own width -> identity
    with pytest.raises(ValueError, match="nlh"):
        adapt(np.zeros(OBS_DIM, dtype=np.float32))  # was: silent [:995] slice
    with pytest.raises(ValueError):
        adapt(np.zeros((4, OBS_DIM_V2), dtype=np.float32))


def test_a20_obs_adapter_plo_known_layouts_still_adapt():
    full = np.arange(OBS_DIM, dtype=np.float32)
    # 991 / 1020-wide PLO stems: exact prefix slices of the current layout.
    for w in (OBS_DIM_V2, 1020):
        m = ActorCriticV2(hidden_dim=8, obs_dim=w, num_layers=1)
        out = obs_adapter(m)(full)
        np.testing.assert_array_equal(out, full[:w])
        # already-adapted input of the model's own width passes through
        own = np.zeros((2, w), dtype=np.float32)
        assert obs_adapter(m)(own) is own
    # v1-era 959 + minimal 796 keep their named projections.
    assert obs_adapter(
        ActorCritic(hidden_dim=8, obs_dim=OBS_DIM_V1, num_layers=1)
    )(full).shape == (OBS_DIM_V1,)
    assert obs_adapter(
        ActorCriticV5(hidden_dim=8, obs_dim=OBS_DIM_MINIMAL, num_layers=2)
    )(full).shape == (OBS_DIM_MINIMAL,)
    # current-width model: identity, same object
    cur = ActorCriticV2(hidden_dim=8, num_layers=1)
    assert obs_adapter(cur)(full) is full


@pytest.mark.parametrize("bad_width", [OBS_DIM_NLH, OBS_DIM + 1, 500])
def test_a20_obs_adapter_refuses_unknown_widths_for_plo_model(bad_width):
    for m in (
        ActorCriticV2(hidden_dim=8, num_layers=1),                      # 1171
        ActorCriticV2(hidden_dim=8, obs_dim=OBS_DIM_V2, num_layers=1),   # 991
        ActorCritic(hidden_dim=8, obs_dim=OBS_DIM_V1, num_layers=1),     # 959
    ):
        with pytest.raises(ValueError):
            obs_adapter(m)(np.zeros(bad_width, dtype=np.float32))
