"""Old-format checkpoints still resume exactly (2026-09-28, schema 2).

The checkpoint dict gained keys ("schema", "arch", "provenance") and
TrainingConfig gained fields (weight_decay, critic_q_norm_minibatch,
compile_critic_train). The live run resumes from files written by the OLD code,
so: build old-format files the old way (legacy keys only, old config dict),
resume from them through train.py's main(), and check that the weights, the
Adam moments (the optimizer sidecar) and the opponent pool come back exactly;
and that the self-describing loaders rebuild both formats identically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from plo5bp.network import (
    actor_arch,
    build_actor_from_checkpoint,
    build_critic_from_checkpoint,
    critic_arch,
)
from plo5bp.train import loop
from plo5bp.train.checkpoint import CHECKPOINT_SCHEMA, checkpoint_schema

# Every key the pre-2026-09-28 trainer wrote, in its order.
LEGACY_KEYS = (
    "model", "critic", "head_version", "config", "game_config", "gate_count",
    "variant", "anchor_count", "update_counter", "pool_member_updates",
    "anneal_tier_ent", "anneal_baseline", "anneal_block_acc", "mix_configs",
    "mix_tiers", "configs_per_tier", "model_ema", "anneal_control_applied",
    "drain_inflight", "obs_rev",
)
NEW_CONFIG_FIELDS = (
    "weight_decay", "critic_q_norm_minibatch", "compile_critic_train", "num_minibatches",
)
# Keys (and config fields) of retired features the old trainer also wrote
# (2026-09-28, ML-030: the block-rotation auto-anneal's state and the
# aggression bonuses): gone from new checkpoints, still accepted from old ones.
RETIRED_KEYS = {
    "anneal_baseline": {"clubgg": None, "clubgg_deep": None, "deep": None},
    "anneal_block_acc": {"bonus_steps": [0, 0, 0], "steps": [0, 0, 0], "tier": None},
}
RETIRED_CONFIG_FIELDS = {"aggression_bonus_c": 0.0, "retroactive_bonus_c": 0.0}

ARGS = [
    "--variant", "plo5_double_bomb", "--v6", "--obs-mode", "minimal",
    "--hidden-dim", "16", "--num-layers", "3",
    "--critic-hidden-dim", "16", "--critic-num-blocks", "1",
    "--critic-act", "silu", "--critic-in-norm", "--critic-v-raw",
    "--q-base-raw", "--q-fold-zero", "--critic-q-norm",
    "--critic-extra-epochs", "1", "--critic-minibatches", "2",
    "--no-grad-checkpoint", "--device", "cpu",
    "--num-envs", "60", "--rollout-length", "900",
    "--num-minibatches", "2", "--ppo-epochs", "1",
    "--mix-configs", "--configs-per-tier", "1", "--cpu-threads", "2",
    "--snapshot-every", "1", "--checkpoint-every", "1", "--seed", "7",
]


def _main(argv: list[str]) -> None:
    old = sys.argv
    sys.argv = ["train.py", *argv]
    try:
        loop.main()
    finally:
        sys.argv = old


def _to_old_format(ckpt: dict) -> dict:
    old = {k: (RETIRED_KEYS[k] if k in RETIRED_KEYS else ckpt[k]) for k in LEGACY_KEYS}
    old["config"] = {k: v for k, v in ckpt["config"].items() if k not in NEW_CONFIG_FIELDS}
    old["config"].update(RETIRED_CONFIG_FIELDS)
    return old


def _same_tensors(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("compat")
    _main([*ARGS, "--num-updates", "3", "--checkpoint", str(d / "src" / "x.pt"),
           "--run-dir", str(d / "runs")])
    return d


def test_new_checkpoints_are_schema_2_and_describe_themselves(trained):
    ck = torch.load(trained / "src" / "x_2.pt", map_location="cpu", weights_only=False)
    assert checkpoint_schema(ck) == CHECKPOINT_SCHEMA == 2
    # nothing renamed or removed but the retired features' keys
    assert all(k in ck for k in LEGACY_KEYS if k not in RETIRED_KEYS)
    assert not any(k in ck for k in RETIRED_KEYS)
    assert not any(k in ck["config"] for k in RETIRED_CONFIG_FIELDS)
    assert ck["provenance"]["argv"] and "engine" in ck["provenance"]
    actor = build_actor_from_checkpoint(ck)
    critic = build_critic_from_checkpoint(ck)
    assert actor_arch(actor) == ck["arch"]["actor"]
    assert critic_arch(critic) == ck["arch"]["critic"]
    # the same networks as sniffing an old-format dict
    old = _to_old_format(ck)
    assert checkpoint_schema(old) == 1
    _same_tensors(build_actor_from_checkpoint(old).state_dict(), actor.state_dict())
    old_critic = build_critic_from_checkpoint(old)
    assert critic_arch(old_critic) == critic_arch(critic)
    _same_tensors(old_critic.state_dict(), critic.state_dict())


def test_old_format_checkpoint_resumes_weights_adam_and_pool(trained, capsys):
    src, old_dir = trained / "src", trained / "old"
    old_dir.mkdir(exist_ok=True)
    for n in (1, 2):
        ck = torch.load(src / f"x_{n}.pt", map_location="cpu", weights_only=False)
        torch.save(_to_old_format(ck), old_dir / f"o_{n}.pt")
    # The sidecar format did not change: the old trainer wrote this very dict.
    side = torch.load(src / "x.optim.pt", map_location="cpu", weights_only=False)
    torch.save(side, old_dir / "o.optim.pt")
    capsys.readouterr()

    out_ck = trained / "resumed" / "r.pt"
    _main([*ARGS, "--num-updates", "0", "--load-checkpoint", str(old_dir / "o_2.pt"),
           "--checkpoint", str(out_ck), "--run-dir", str(trained / "runs")])
    log = capsys.readouterr().out
    assert "[optim] restored Adam moments from o.optim.pt" in log, log[-3000:]
    assert "[pool] warm-start seeded" in log, log[-3000:]

    old = torch.load(old_dir / "o_2.pt", map_location="cpu", weights_only=False)
    new = torch.load(out_ck, map_location="cpu", weights_only=False)
    _same_tensors(new["model"], old["model"])
    _same_tensors(new["critic"], old["critic"])
    assert new["update_counter"] == old["update_counter"] == 2
    # The recorded prior members that have a file on disk (no mid save exists
    # for update 0: numbered saves start at the second update).
    assert new["pool_member_updates"] == [u for u in old["pool_member_updates"] if u >= 1]
    assert checkpoint_schema(new) == CHECKPOINT_SCHEMA
    # The restored Adam moments are the ones saved beside the old checkpoint.
    new_side = torch.load(out_ck.with_name("r.optim.pt"), map_location="cpu", weights_only=False)
    valid_for = {side["update_counter"], *side.get("same_state_counters", [])}
    entry = side if 2 in valid_for else side["previous"]
    for idx, st in entry["optimizer_state"].items():
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(new_side["optimizer_state"][idx][key], st[key]), (idx, key)
