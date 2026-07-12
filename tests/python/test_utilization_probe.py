"""Network-capacity utilization probe (scripts/utilization_probe.py) pins.

Exercises the probe's measurement + I/O helpers on tiny in-process nets and a
small batch of real flop nodes:

- gen_nodes yields well-shaped (obs, opp, masks).
- measure_actor / measure_critic report per-layer metrics in-range: dead% in
  [0,100] and eff_rank in (0, width] for torso linears, None dead-metrics for
  head linears, finite stable ranks everywhere, adv_head captured on a
  dueling critic.
- build_record / write_record round-trip a JSON history line with the exact
  per-layer schema keys, and parse the trailing _<N> update from the stem.

Kept tiny (hidden_dim=32, ~200 nodes) so the whole file runs in a few seconds.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from plo5bp.config import GameConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.network import ActorCriticV2, CentralCritic

_PROBE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "utilization_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "utilization_probe_under_test", _PROBE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load_probe()

_SCHEMA_KEYS = {
    "name", "width", "dead_pct", "near_dead_pct", "eff_rank", "rank99",
    "stable_rank",
}


def _probe_batch():
    """~200 real fold-legal flop nodes from the default 20bb table (tiny seed
    budget so the whole test stays fast)."""
    return probe.gen_nodes(
        GameConfig(), n_seeds=500, street="flop", bet="pot", max_nodes=250
    )


def _tiny_actor():
    torch.manual_seed(0)
    m = ActorCriticV2(hidden_dim=32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _tiny_critic():
    torch.manual_seed(1)
    m = CentralCritic(
        hidden_dim=32, num_blocks=2, q_actions=3, value_bins=51
    ).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def test_gen_nodes_shapes():
    obs, opp, masks = _probe_batch()
    n = int(obs.shape[0])
    assert n >= 50
    assert obs.shape == (n, OBS_DIM)
    assert obs.dtype == np.float32
    assert opp.shape[0] == n and opp.shape[1] == 5 and opp.shape[2] == 5
    assert masks.shape == (n, 3) and masks.dtype == bool


def _assert_common_layer(L):
    # Bounded + finite everywhere. eff_rank/stable_rank can legitimately be 0
    # for a degenerate layer (e.g. the zero-init dueling adv_head, where
    # Q == V at init) — the strict (0, width] / >0 checks live in the
    # torso-specific block below, where they always hold.
    w = L["width"]
    assert np.isfinite(L["eff_rank"])
    assert 0.0 <= L["eff_rank"] <= w + 1e-6           # eff_rank in [0, width]
    assert isinstance(L["rank99"], int) and 0 <= L["rank99"] <= w
    assert np.isfinite(L["stable_rank"]) and L["stable_rank"] >= 0.0


def test_actor_layer_metrics_ranges():
    obs, opp, masks = _probe_batch()
    actor = _tiny_actor()
    layers = probe.measure_actor(actor, obs, masks)
    assert len(layers) >= 2
    torso_seen = head_seen = False
    for L in layers:
        _assert_common_layer(L)
        w = L["width"]
        if L["is_torso"]:
            torso_seen = True
            assert 0.0 <= L["dead_pct"] <= 100.0
            assert 0.0 <= L["near_dead_pct"] <= 100.0
            assert 0.0 < L["eff_rank"] <= w + 1e-6    # (0, width] for torso
            assert L["stable_rank"] > 0.0             # torso W never zero-init
        else:
            head_seen = True
            assert L["dead_pct"] is None and L["near_dead_pct"] is None
    assert torso_seen and head_seen


def test_critic_layer_metrics_and_adv_head():
    obs, opp, masks = _probe_batch()
    critic = _tiny_critic()
    layers = probe.measure_critic(critic, obs, opp)
    names = {L["name"] for L in layers}
    # train_outputs exercises BOTH heads on the dueling critic
    assert "adv_head" in names and "value_head" in names
    for L in layers:
        _assert_common_layer(L)
        if L["is_torso"]:
            assert 0.0 <= L["dead_pct"] <= 100.0
            assert 0.0 < L["eff_rank"] <= L["width"] + 1e-6
            assert L["stable_rank"] > 0.0
    vh = next(L for L in layers if L["name"] == "value_head")
    assert vh["width"] == 51 and vh["dead_pct"] is None


def test_json_record_schema(tmp_path):
    obs, opp, masks = _probe_batch()
    a_layers = probe.measure_actor(_tiny_actor(), obs, masks)
    c_layers = probe.measure_critic(_tiny_critic(), obs, opp)
    rec = probe.build_record(
        "checkpoints/faketest_42.pt", int(obs.shape[0]), a_layers, c_layers
    )
    out = tmp_path / "utilization_history.jsonl"
    probe.write_record(rec, out)

    parsed = json.loads(out.read_text(encoding="utf-8").strip())
    for k in ("ckpt", "update", "ts", "rows", "actor", "critic"):
        assert k in parsed
    assert parsed["ckpt"] == "faketest_42.pt"
    assert parsed["update"] == 42
    assert parsed["rows"] == int(obs.shape[0])
    assert parsed["actor"]["layers"] and parsed["critic"]["layers"]
    for net in ("actor", "critic"):
        for layer in parsed[net]["layers"]:
            assert set(layer.keys()) == _SCHEMA_KEYS


def test_update_none_without_trailing_number():
    rec = probe.build_record("checkpoints/stub.pt", 10, [], [])
    assert rec["update"] is None
