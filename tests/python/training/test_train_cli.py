"""train.py's command line and run files (2026-09-28).

- ML-022: the network size (actor AND critic) is required.
- ML-023: block rotation is off unless named.
- ML-006: flag combinations that would silently do nothing are refused; a
  --spec file supplies defaults and the command line wins.
- The live guardian's exact command line still parses and validates.
- ML-005 / ML-024 / ML-019 / ML-060: the per-run files (metrics JSONL,
  heartbeat, launch provenance) are named after the checkpoint stem; a
  rollback livelock exits with its own code.
- ML-001: non-finite statistics / weights are refused explicitly.
- ML-049: the rollout dump refuses a layout it does not describe.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from plo5bp.ppo import PPOStats, PPOTrainer
from plo5bp.train import loop
from plo5bp.train.checkpoint import NonFiniteCheckpointError, assert_finite_for_save
from plo5bp.train.cli import (
    _apply_v6_preset,
    build_parser,
    parse_args,
    validate_flag_combinations,
)
from plo5bp.train.diagnostics import _dump_batch_diagnostics
from plo5bp.train.metrics import read_metrics

REPO = Path(__file__).resolve().parents[3]
SIZES = ["--hidden-dim", "16", "--num-layers", "3",
         "--critic-hidden-dim", "16", "--critic-num-blocks", "1"]


def _resolved(argv: list[str]):
    args = parse_args(argv)
    _apply_v6_preset(args)
    validate_flag_combinations(args)
    return args


@pytest.mark.parametrize("missing", ["--hidden-dim", "--num-layers",
                                     "--critic-hidden-dim", "--critic-num-blocks"])
def test_every_size_flag_is_required(missing, capsys):
    argv = list(SIZES)
    i = argv.index(missing)
    del argv[i:i + 2]
    with pytest.raises(SystemExit):
        parse_args(argv)
    assert missing in capsys.readouterr().err


@pytest.mark.parametrize("flag", [
    ["--block-rotation", "deep:0.1"], ["--block-size", "50"], ["--anneal-entropy"],
    ["--anneal-step", "0.002"], ["--anneal-floor", "0"], ["--anneal-tolerance", "1"],
    ["--anneal-start-update", "600"], ["--aggression-bonus-c", "5"],
    ["--retroactive-bonus-c", "0.1"], ["--stack-dist", "agro_deep"],
    ["--stack-dist", "nlh_topoff"], ["--seats-dist", "nlh_ring"],
])
def test_retired_features_are_gone(flag, capsys):
    """Block rotation, the F/T/R auto-anneal, the aggression bonuses and the
    NLH-PPO tiers were retired 2026-09-28 (ML-030): a launch line still using
    one fails at once instead of training something else."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([*SIZES, *flag])


@pytest.mark.parametrize("extra, message", [
    (["--batch-on-host"], "--batch-on-host needs --mix-configs"),
    (["--clip-room-mid", "0.07"], "--clip-room-mid only acts"),
    (["--clip-room-ext", "0.2"], "--clip-room-ext only acts"),
    (["--kl-anchor-ema", "0.99"], "--kl-anchor-ema only acts"),
    (["--critic-minibatches", "8"], "--critic-minibatches only acts"),
    (["--critic-q-norm-minibatch"], "--critic-q-norm-minibatch only acts"),
    (["--minibatches-from-rows", "--batch-size", "512"], "--minibatches-from-rows sizes"),
    (["--critic-fresh"], "need --load-checkpoint"),
])
def test_silently_ignored_combinations_are_refused(extra, message):
    with pytest.raises(SystemExit, match=re.escape(message)):
        _resolved([*SIZES, *extra])


def test_valid_combinations_resolve_their_sentinels():
    args = _resolved([*SIZES, "--v6", "--clip-room-mid", "0.07", "--mix-configs",
                      "--batch-on-host", "--critic-extra-epochs", "1",
                      "--critic-minibatches", "8", "--critic-q-norm",
                      "--critic-q-norm-minibatch"])
    assert (args.clip_room_ext, args.clip_room_mid, args.kl_anchor_ema) == (0.10, 0.07, 0.999)
    plain = _resolved(SIZES)
    assert (plain.clip_room_ext, plain.clip_room_mid, plain.kl_anchor_ema) == (0.10, 0.05, 0.999)


def test_spec_file_supplies_defaults_and_the_command_line_wins(tmp_path):
    spec = tmp_path / "run.toml"
    spec.write_text(
        'hidden-dim = 64\nnum_layers = 3\ncritic-hidden-dim = 32\n'
        'critic-num-blocks = 2\nmix-configs = true\nlr = 1e-4\nentropy-coef = 0.045\n',
        encoding="utf-8",
    )
    args = parse_args(["--spec", str(spec), "--lr", "7.5e-5"])
    assert (args.hidden_dim, args.num_layers, args.critic_hidden_dim) == (64, 3, 32)
    assert args.mix_configs is True and args.entropy_coef == 0.045
    assert args.lr == 7.5e-5  # the command line wins
    js = tmp_path / "run.json"
    js.write_text(json.dumps({"hidden_dim": 8, "num-layers": 1, "critic_hidden_dim": 8,
                              "critic_num_blocks": 1}), encoding="utf-8")
    assert parse_args(["--spec", str(js)]).hidden_dim == 8
    bad = tmp_path / "bad.toml"
    bad.write_text('hidden-dim = 8\nno-such-flag = 1\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown flag"):
        parse_args(["--spec", str(bad)])


def _guardian_train_argv(path: Path) -> list[str]:
    """The train.py flags of a guardian's launch(), its ${VAR:-default}s filled."""
    text = path.read_text(encoding="utf-8")
    defaults = dict(re.findall(r'^(\w+)=\$\{\w+:-([^}]*)\}', text, flags=re.M))
    m = re.search(r"python -u scripts/train\.py \\\n(.*?)>> ", text, flags=re.S)
    assert m, f"no train.py command in {path}"
    cmd = m.group(1).replace("\\\n", " ")
    cmd = re.sub(r"\$load|\$first", "", cmd)
    cmd = re.sub(r'"?\$(\w+)"?', lambda v: defaults[v.group(1)], cmd)
    return shlex.split(cmd)


def test_the_vsix6_guardian_command_line_still_parses_and_validates():
    argv = _guardian_train_argv(REPO / "scripts" / "vSix6_guardian.sh")
    args = _resolved(argv)
    assert args.hidden_dim == 1024 and args.critic_hidden_dim == 1536
    assert args.v6 and args.mix_configs and args.batch_on_host
    assert args.clip_room_mid == 0.07 and args.critic_minibatches == 128


def _stats(**kw) -> PPOStats:
    base = dict(policy_loss=0.0, value_loss=1.0, entropy=1.0, approx_kl=0.0)
    base.update(kw)
    return PPOStats(**base)


def test_nonfinite_statistics_are_refused_explicitly():
    loop._check_stats_finite(_stats())
    with pytest.raises(RuntimeError, match="critic_value_loss"):
        loop._check_stats_finite(_stats(critic_value_loss=float("nan")))
    # diagnostics that are legitimately nan (no minibatch ran) are not checked
    loop._check_stats_finite(_stats(kl0=float("nan"), ratio_dev0=float("nan")))


def test_a_nonfinite_checkpoint_is_never_written():
    good = {"model": {"w": torch.ones(3)}, "critic": {"b": torch.zeros(2)}, "model_ema": None}
    assert_finite_for_save(good, "x.pt", ("model", "critic", "model_ema"))
    bad = {"model": {"w": torch.tensor([1.0, float("inf")])}, "critic": {}}
    with pytest.raises(NonFiniteCheckpointError, match="model/w"):
        assert_finite_for_save(bad, "x.pt", ("model", "critic", "model_ema"))
    # integer tensors (e.g. the critic's _arch buffer) are not float: fine
    assert_finite_for_save({"critic": {"_arch": torch.tensor([1, 0, 1])}}, "x.pt", ("critic",))


def test_the_dump_refuses_a_layout_it_does_not_describe(tmp_path):
    batch = SimpleNamespace(obs=torch.zeros(4, 995), values=torch.zeros(4))
    with pytest.raises(SystemExit, match="995-wide"):
        _dump_batch_diagnostics(batch, str(tmp_path / "d.npz"))


RUN = [*SIZES, "--variant", "plo5_double_bomb", "--v6", "--obs-mode", "minimal",
       "--device", "cpu", "--num-envs", "40", "--rollout-length", "600",
       "--num-minibatches", "2", "--ppo-epochs", "1", "--mix-configs",
       "--configs-per-tier", "1", "--cpu-threads", "2", "--seed", "3"]


def _main(argv: list[str]) -> None:
    old = sys.argv
    sys.argv = ["train.py", *argv]
    try:
        loop.main()
    finally:
        sys.argv = old


def test_run_files_are_named_after_the_stem(tmp_path):
    runs = tmp_path / "runs"
    _main([*RUN, "--num-updates", "2", "--checkpoint", str(tmp_path / "ck" / "st.pt"),
           "--run-dir", str(runs)])
    recs = read_metrics(runs / "st.metrics.jsonl")
    assert [r["update"] for r in recs] == [0, 1]
    for r in recs:
        assert r["rows"] > 0 and r["ppo"]["entropy"] > 0
        assert set(r["value_health"]["tier"]) == {"clubgg", "clubgg_deep", "deep"}
        assert r["pool"]["size"] >= 1
    beat = json.loads((runs / "st.heartbeat").read_text(encoding="utf-8"))
    assert beat["update"] == 1 and beat["state"] == "ok"
    launch = json.loads((runs / "st.launches.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert launch["provenance"]["argv"][0] == "train.py"
    assert launch["config"]["hidden_dim"] == 16
    assert not (runs / "st.resources.jsonl").exists()  # the sampler is opt-in


def test_a_rollback_livelock_exits_with_its_own_code(tmp_path, monkeypatch):
    real = PPOTrainer.update

    def always_rolled_back(self, *a, **k):
        s = real(self, *a, **k)
        s.rolled_back = True
        return s

    monkeypatch.setattr(PPOTrainer, "update", always_rolled_back)
    runs = tmp_path / "runs"
    with pytest.raises(SystemExit) as e:
        _main([*RUN, "--num-updates", "5", "--max-consecutive-rollbacks", "2",
               "--checkpoint", str(tmp_path / "ck" / "lv.pt"), "--run-dir", str(runs)])
    assert e.value.code == loop.LIVELOCK_EXIT_CODE == 3
    beat = json.loads((runs / "lv.heartbeat").read_text(encoding="utf-8"))
    assert beat["state"] == "livelock" and beat["consecutive_rollbacks"] == 2


def test_a_new_critic_through_main_keeps_the_frozen_actor(tmp_path, capsys):
    """TEST-019: --critic-fresh / --critic-init with --actor-freeze-updates, the
    way vSix6's first launch installed its critic, run end to end."""
    runs = tmp_path / "runs"
    src = tmp_path / "ck" / "a.pt"
    _main([*RUN, "--num-updates", "1", "--checkpoint", str(src), "--run-dir", str(runs)])
    a = torch.load(src, map_location="cpu", weights_only=False)
    out = tmp_path / "ck2" / "b.pt"
    _main([*RUN, "--num-updates", "1", "--load-checkpoint", str(src), "--critic-fresh",
           "--actor-freeze-updates", "1", "--checkpoint", str(out), "--run-dir", str(runs)])
    b = torch.load(out, map_location="cpu", weights_only=False)
    for k, v in a["model"].items():
        assert torch.equal(v, b["model"][k]), f"actor {k} moved while frozen"
    assert any(not torch.equal(v, b["critic"][k]) for k, v in a["critic"].items())
    log = capsys.readouterr().out
    assert "--critic-fresh" in log and "actor FROZEN" in log
    # --critic-init from a bare critic state dict (no critic needed in the
    # loaded checkpoint any more -- ML-006)
    bare = tmp_path / "critic_only.pt"
    torch.save(b["critic"], bare)
    no_critic = tmp_path / "ck3" / "nc.pt"
    no_critic.parent.mkdir()
    torch.save({k: v for k, v in a.items() if k != "critic"}, no_critic)
    _main([*RUN, "--num-updates", "1", "--load-checkpoint", str(no_critic), "--critic-init",
           str(bare), "--actor-freeze-updates", "1", "--checkpoint", str(tmp_path / "ck3" / "c.pt"),
           "--run-dir", str(runs)])
    assert "[critic] initialized from" in capsys.readouterr().out
