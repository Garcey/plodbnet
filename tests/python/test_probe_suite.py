"""scripts/probe_suite.py — the per-checkpoint probe suite.

Drives the module's node generator + per-family metric function + lock
split against tiny in-process nets (no real checkpoint on disk), and pins
the JSONL history schema. Fast: one small CLUBGG family, ~80 seeds.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from plo5bp.network import ActorCriticV2, CentralCritic

_PROBE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "probe_suite.py"
_SPEC = importlib.util.spec_from_file_location("plo5bp_probe_suite", _PROBE_PATH)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


_POLICY_KEYS = {"gate_p_fold", "gate_p_cc", "gate_p_raise",
                "sharp_med", "gate_h_mean"}
_CRITIC_KEYS = {"v_mean", "v_std", "fold_mean", "fold_std", "fold_gt1bb_pct",
                "fold_slope", "fold_resid", "a_call_med", "a_call_std",
                "gap_med", "gap_abs_med", "gap_abs_p95"}
_POOLED_KEYS = {"a_raise_med", "a_raise_std", "a_raise_p5", "a_raise_p95"}
_LOCK_KEYS = {"n", "n_lock", "pf_lock", "pr_lock", "val_lock",
              "n_trash", "pf_trash"}


@pytest.fixture(scope="module")
def tiny_nets():
    torch.manual_seed(0)
    actor = ActorCriticV2(hidden_dim=32).eval()
    critic = CentralCritic(
        hidden_dim=32, num_blocks=2, q_actions=3, value_bins=51
    ).eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    for p in critic.parameters():
        p.requires_grad_(False)
    return actor, critic


@pytest.fixture(scope="module")
def small_bank():
    # One small family in the exact gen_nodes procedure (CLUBGG flop vs
    # pot-bet), keyed by the real "shallow" lock tag so evaluate_checkpoint
    # exercises the lock split too.
    obs, opp, masks = probe.gen_nodes(probe.CLUBGG, 80, "flop", "pot", 0)
    return obs, opp, masks


def _all_finite(m):
    for k, v in m.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"{k} not finite: {v}"
        elif isinstance(v, list):
            assert all(np.isfinite(x) for x in v), f"{k} has non-finite"


def test_gen_nodes_shapes(small_bank):
    obs, opp, masks = small_bank
    assert obs.ndim == 2 and obs.shape[1] >= 1020
    assert obs.shape[0] > 0
    assert opp.shape[0] == obs.shape[0] and opp.shape[1] == 5
    assert masks.shape == (obs.shape[0], 3)
    assert obs.dtype == np.float32 and masks.dtype == bool


def test_family_metrics_keys_and_finite(tiny_nets, small_bank):
    actor, critic = tiny_nets
    obs, opp, masks = small_bank
    m = probe.family_metrics(actor, critic, obs, opp, masks)
    # policy + critic (pooled, since q_actions == 3) keys all present
    assert _POLICY_KEYS <= set(m)
    assert _CRITIC_KEYS <= set(m)
    assert _POOLED_KEYS <= set(m)
    _all_finite(m)
    # gate probs are a normalized distribution
    tot = m["gate_p_fold"] + m["gate_p_cc"] + m["gate_p_raise"]
    assert tot == pytest.approx(1.0, abs=1e-4)


def test_family_metrics_actor_only(tiny_nets, small_bank):
    actor, _critic = tiny_nets
    obs, opp, masks = small_bank
    m = probe.family_metrics(actor, None, obs, opp, masks)
    assert _POLICY_KEYS <= set(m)
    assert not (_CRITIC_KEYS & set(m))  # no Q metrics without a critic
    _all_finite(m)


def test_lock_split_keys(tiny_nets, small_bank):
    actor, _critic = tiny_nets
    obs, _opp, masks = small_bank
    ls = probe.lock_split(actor, obs, masks)
    assert set(ls) == _LOCK_KEYS
    for k in ("n", "n_lock", "n_trash"):
        assert isinstance(ls[k], int)
    assert ls["n"] == obs.shape[0]
    assert 0 <= ls["n_lock"] <= ls["n"]
    if ls["n_lock"] > 0:
        assert np.isfinite(ls["pf_lock"]) and np.isfinite(ls["pr_lock"])
        assert 0.0 <= ls["pf_lock"] <= 100.0


def test_parse_update():
    assert probe.parse_update("checkpoints/vSix1_460.pt") == 460
    assert probe.parse_update("/tmp/nlh1_20.pt") == 20
    assert probe.parse_update("stub.pt") is None


def test_jsonl_writer_schema(tiny_nets, small_bank, tmp_path):
    actor, critic = tiny_nets
    obs, opp, masks = small_bank
    tag = probe.LOCK_TAGS["shallow"]
    banks = {tag: (obs, opp, masks)}
    record = probe.evaluate_checkpoint(
        actor, critic, banks, ckpt_name="tiny_5.pt", update=5, do_print=False
    )
    assert set(record) >= {"ckpt", "update", "ts", "families", "locks"}
    assert record["ckpt"] == "tiny_5.pt" and record["update"] == 5
    assert tag in record["families"]
    assert "shallow" in record["locks"] and "deep" in record["locks"]

    out = tmp_path / "hist.jsonl"
    probe.append_jsonl(str(out), record)
    probe.append_jsonl(str(out), record)  # append, don't clobber
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])  # valid JSON, no NaN tokens
    assert set(parsed) >= {"ckpt", "update", "ts", "families", "locks"}
    assert _POLICY_KEYS <= set(parsed["families"][tag])


def test_load_checkpoint_rebuilds_critic_with_trained_q_semantics(tmp_path):
    """(review 2026-09-20 A19) The suite READS Q, so the critic must come back
    with the run's value support + q_base_raw / q_fold_zero from the
    checkpoint's config stamp. Rebuilt at the builder defaults, a support-3000
    q_base_raw critic read Q far off (V exact) and the flags were dropped."""
    from plo5bp.network import opp_holes_multihot

    torch.manual_seed(0)
    actor = ActorCriticV2(hidden_dim=32)
    critic = CentralCritic(
        hidden_dim=32, num_blocks=1, q_actions=3, value_bins=51,
        value_support=3000.0, hlgauss_sigma=0.5,
        q_base_raw=True, q_fold_zero=True,
    )
    with torch.no_grad():
        critic.value_head.weight.normal_(0, 1.0)
        critic.adv_head.weight.normal_(0, 0.1)
    path = tmp_path / "probe_me_7.pt"
    torch.save(
        {
            "model": actor.state_dict(),
            "critic": critic.state_dict(),
            "config": {
                "hidden_dim": 32, "num_layers": 2,
                "value_support": 3000.0, "value_hlgauss_sigma": 0.5,
                "q_base_raw": True, "q_fold_zero": True,
            },
        },
        path,
    )
    _actor, loaded = probe.load_checkpoint(str(path))
    assert loaded.q_base_raw and loaded.q_fold_zero
    obs = torch.randn(32, critic.obs_dim)
    opp = opp_holes_multihot(torch.randint(0, 52, (32, 5, 5)).to(torch.uint8))
    with torch.inference_mode():
        v0, q0 = critic.q_values(obs, opp)
        v1, q1 = loaded.q_values(obs, opp)
    assert torch.equal(v0, v1) and torch.equal(q0, q1)
    assert float(q1[:, 0].abs().max()) == 0.0  # fold column pinned

    # A checkpoint whose config predates the keys still loads (defaults).
    torch.save(
        {"model": actor.state_dict(),
         "critic": CentralCritic(hidden_dim=32, num_blocks=1).state_dict(),
         "config": {"hidden_dim": 32, "num_layers": 2}},
        tmp_path / "old_3.pt",
    )
    _a, old = probe.load_checkpoint(str(tmp_path / "old_3.pt"))
    assert old is not None and not old.q_base_raw
