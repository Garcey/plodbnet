"""Regression tests for the 2026-09-20 code review, training workstream —
scripts/train.py items (pure helpers + a miniature in-process main()):

- A1  _sample_game_config never returns a config where <2 seats can act.
- A3  the rolling optimizer sidecar is written next to the checkpoints, its
      name does NOT match the guardians' `<stem>_*.pt` glob, and a warm start
      restores it only for the matching update_counter.
- A4  the applied control content is stamped and re-applied on relaunch.
- A7  checkpoint writes are atomic (tmp + os.replace).
- A11 the default --checkpoint is not a UI-served file; stub names refused.
- A12 --mix-tiers / unknown stack-dist names raise.
- A13 control-file reads never raise; A20 value hardening + entropy broadcast.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_TRAIN_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
_SPEC = importlib.util.spec_from_file_location("plo5bp_train_review", _TRAIN_PATH)
train = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train)

BB, ANTE = 10_000, 30_000


class _Trainer:
    target_kl = 0.5
    kl_hard = 10.0
    sizing_entropy_scale = 1.0
    _clip_room_mid = 0.05
    _clip_room_ext = 0.10
    _q_fold_sup = 0.0


def _apply(raw, tiers=None, last=None, trainer=None, **kw):
    tiers = {"clubgg": 0.45, "deep": 0.30} if tiers is None else tiers
    out = train._apply_anneal_control(
        raw, last, tiers, 0.002, 1.5e-4, 0.45, 0.45,
        trainer=trainer if trainer is not None else _Trainer(), **kw,
    )
    return out, tiers


# --------------------------------------------------------------------- A1
@pytest.mark.parametrize("variant", [train.VARIANT_PLO5, train.VARIANT_NLH])
def test_a1_sampled_configs_always_seat_two_live_players(variant):
    # The production-shaped draw: clubgg band (1-20bb @ 5%) inside 1:300bb,
    # 3bb ante, HU included — P(stack <= ante) ~ 0.5% per seat.
    rng = np.random.default_rng(0)
    is_nlh = variant == train.VARIANT_NLH
    ante = BB // 2 if is_nlh else ANTE
    floor = ante + (BB if is_nlh else 0)
    for _ in range(4000):
        cfg, _tier = train._sample_game_config(
            (2, 3), 1.0, 300.0, BB, ante, rng,
            stack_dist="clubgg", variant=variant, sb=BB // 2 if is_nlh else 0,
        )
        assert sum(s > floor for s in cfg.resolved_stacks) >= 2, cfg


def test_a1_first_draw_is_unchanged_for_playable_configs():
    # Same seed, one draw each: the resample loop must not perturb the stream.
    a = train._sample_game_config(
        (2, 3, 4, 5, 6), 1.0, 300.0, BB, ANTE, np.random.default_rng(7),
        stack_dist="full_mix", seats_dist="clubgg",
    )
    rng = np.random.default_rng(7)
    n = train._sample_clubgg_seats((2, 3, 4, 5, 6), rng)
    tier = str(rng.choice(("clubgg", "clubgg_deep", "deep")))
    assert a[0].num_seats == n and a[1] == tier


def test_a1_unplayable_range_is_clamped_loudly(capsys):
    train._CONTROL_WARNED.clear()
    cfg, _ = train._sample_game_config(
        (2,), 1.0, 2.0, BB, ANTE, np.random.default_rng(1), stack_dist="uniform"
    )
    assert sum(s > ANTE for s in cfg.resolved_stacks) == 2
    assert min(cfg.resolved_stacks) == ANTE + BB
    assert "CLAMPED" in capsys.readouterr().out


# -------------------------------------------------------------------- A12
def test_a12_unknown_tiers_raise():
    assert train._parse_mix_tiers("clubgg, clubgg_deep,deep") == [
        "clubgg", "clubgg_deep", "deep",
    ]
    with pytest.raises(SystemExit, match="clubg_deep"):
        train._parse_mix_tiers("clubgg,clubg_deep")
    with pytest.raises(ValueError, match="clubg_deep"):
        train._sample_game_config(
            (6,), 1.0, 300.0, BB, ANTE, np.random.default_rng(0),
            stack_dist="clubg_deep",
        )


# ---------------------------------------------------------------- A13 / A20
def test_a13_control_reads_never_raise(tmp_path, capsys):
    train._CONTROL_WARNED.clear()
    p = tmp_path / "anneal_control.json"
    p.write_bytes('{"lr": 1e-4}'.encode("utf-16"))  # PowerShell 5.1 `>`
    assert train._read_control_text(p) is None
    assert train._read_control_text(p) is None
    out = capsys.readouterr().out
    assert out.count("cannot read") == 1  # logged ONCE, not every update
    p.write_bytes(b"\xef\xbb\xbf" + b'{"lr": 1e-4}')  # UTF-8 BOM
    raw = train._read_control_text(p)
    assert json.loads(raw) == {"lr": 1e-4}  # BOM used to break the parse
    assert train._read_control_text(tmp_path / "missing.json") is None


@pytest.mark.parametrize(
    "raw",
    [
        '{"lr": true}',                   # JSON bool -> was lr 1.0
        '{"lr": -1e-4}',                  # negative
        '{"lr": 0}',                      # e.g. 1e-400 underflow
        '{"lr": 15}',                     # absurd (dropped exponent)
        '{"lr": NaN}',
        '{"target_kl": Infinity}',
        '{"step": 1' + "0" * 400 + "}",   # OverflowError, used to be uncaught
        '{"tier_ent": {"deep": -0.1}}',
        '{"tier_ent": {"deep": true}}',
        '{"clip_room_mid": 0}',
        '{"lr": 1e-4, "kl_hard": -5}',    # one bad key poisons the whole edit
        '{"entropy_coef": "0.3"}',
    ],
)
def test_a20_bad_values_reject_the_whole_edit(raw, capsys):
    train._CONTROL_WARNED.clear()
    tr = _Trainer()
    (step, last, lr, ent, entd), tiers = _apply(raw, trainer=tr)
    assert (step, last, lr, ent, entd) == (0.002, None, 1.5e-4, 0.45, 0.45)
    assert tiers == {"clubgg": 0.45, "deep": 0.30}
    assert (tr.target_kl, tr.kl_hard) == (0.5, 10.0)
    out = capsys.readouterr().out
    assert "IGNORED" in out
    _apply(raw, trainer=tr)  # re-read next update: no second log line
    assert capsys.readouterr().out == ""


def test_a20_unknown_keys_and_tiers_are_logged_once_rest_applies(capsys):
    train._CONTROL_WARNED.clear()
    raw = '{"entropy": 0.2, "lr": 1e-4, "tier_ent": {"DEEP": 0.2, "deep": 0.25}}'
    (step, last, lr, _e, _ed), tiers = _apply(raw)
    assert last == raw and lr == 1e-4 and tiers["deep"] == 0.25
    assert "DEEP" not in tiers
    out = capsys.readouterr().out
    assert "unknown key(s) ['entropy']" in out
    assert "unknown tier_ent tier(s) ['DEEP']" in out


def test_a20_malformed_json_is_logged_once_and_retried(capsys):
    train._CONTROL_WARNED.clear()
    half = '{"lr": 1e-'
    (_s, last, *_), _t = _apply(half)
    assert last is None  # NOT marked applied -> the finished save is picked up
    _apply(half)
    assert capsys.readouterr().out.count("malformed JSON") == 1
    (_s, last, lr, *_), _t = _apply('{"lr": 1e-4}')
    assert last == '{"lr": 1e-4}' and lr == 1e-4


def test_a20_entropy_coef_broadcasts_even_when_flat_value_is_stale():
    # Mix mode: the flat coef is still the launch value (0.45) while `deep`
    # was tuned to 0.30 per tier. Re-sending entropy_coef=0.45 used to compare
    # equal to the stale flat value and do nothing.
    (_s, _l, _lr, ent, _ed), tiers = _apply(
        '{"entropy_coef": 0.45}', broadcast_entropy=True
    )
    assert ent == 0.45 and tiers == {"clubgg": 0.45, "deep": 0.45}
    # a tier named in the same write keeps its explicit value
    (_s, _l, _lr, _e, _ed), tiers = _apply(
        '{"entropy_coef": 0.2, "tier_ent": {"deep": 0.3}}', broadcast_entropy=True
    )
    assert tiers == {"clubgg": 0.2, "deep": 0.3}
    # non-mix runs: the flat key never touches the tiers
    (_s, _l, _lr, ent, _ed), tiers = _apply('{"entropy_coef": 0.2}')
    assert ent == 0.2 and tiers == {"clubgg": 0.45, "deep": 0.30}


# --------------------------------------------------------------------- A7
def test_a7_atomic_save_leaves_no_partial_and_no_glob_match(tmp_path, monkeypatch):
    target = tmp_path / "vSix4_65.pt"
    train._atomic_torch_save({"ok": 1}, target)
    assert torch.load(target, weights_only=False) == {"ok": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["vSix4_65.pt"]

    # A save that dies mid-write must leave the previous file intact and
    # nothing a guardian's `ls -t vSix4_*.pt | head -1` could pick up.
    def boom(obj, f):
        Path(f).write_bytes(b"truncated")
        raise OSError("disk full")

    monkeypatch.setattr(train.torch, "save", boom)
    with pytest.raises(OSError):
        train._atomic_torch_save({"ok": 2}, target)
    monkeypatch.undo()
    assert torch.load(target, weights_only=False) == {"ok": 1}
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != target.name]
    assert leftovers == ["vSix4_65.pt.tmp"]
    assert not fnmatch.fnmatch(leftovers[0], "vSix4_*.pt")


def test_a3_sidecar_name_never_matches_the_checkpoint_globs():
    for ckpt in ("checkpoints/vSix4_65.pt", "checkpoints/vSix4.pt"):
        side = train._optimizer_sidecar_path(Path(ckpt))
        assert side == Path("checkpoints/vSix4.optim.pt")
        assert not fnmatch.fnmatch(side.name, "vSix4_*.pt")
        assert train.re.match(r"^.+_(\d+)\.pt$", side.name) is None


# ------------------------------------------------- in-process miniature runs
def _run_train(ckpt_path: Path, extra: "list[str] | None" = None) -> None:
    argv = [
        "train.py",
        "--num-envs", "2", "--rollout-length", "32",
        "--hidden-dim", "16", "--num-layers", "1",
        "--critic-hidden-dim", "32", "--critic-num-blocks", "1",
        "--device", "cpu", "--no-batched", "--block-rotation", "",
        "--snapshot-every", "1000",
        "--checkpoint", str(ckpt_path),
        *(extra or []),
    ]
    old = sys.argv
    sys.argv = argv
    try:
        train.main()
    finally:
        sys.argv = old


def test_a11_default_checkpoint_is_not_a_served_file(tmp_path):
    src = _TRAIN_PATH.read_text(encoding="utf-8")
    assert 'default=Path("checkpoints/train_run.pt")' in src
    assert 'default=Path("checkpoints/stub.pt")' not in src
    for name in ("stub.pt", "nlh_stub.pt"):
        with pytest.raises(SystemExit, match="UI-served"):
            _run_train(tmp_path / name, ["--num-updates", "1"])
        assert not (tmp_path / name).exists()


def test_a3_a4_sidecar_and_control_stamp_through_main(tmp_path, monkeypatch, capsys):
    """Cold run -> numbered checkpoint + final + ONE sidecar; the applied
    control content is stamped; a warm start from the numbered file restores
    Adam and RE-APPLIES the still-matching control file; a changed file is
    ignored loudly; --no-optimizer-sidecar is the legacy resume."""
    monkeypatch.chdir(tmp_path)  # main() reads runs/ relative to the cwd
    (tmp_path / "runs").mkdir()
    ctrl = tmp_path / "runs" / "anneal_control.json"
    ck = tmp_path / "ck" / "rv.pt"
    train._CONTROL_WARNED.clear()

    # --- cold run; pre-existing control file is a stale one -> ignored LOUDLY
    ctrl.write_text('{"lr": 9e-5}', encoding="utf-8")
    _run_train(ck, ["--num-updates", "2", "--checkpoint-every", "1"])
    out = capsys.readouterr().out
    assert "PRE-EXISTING" in out and "IGNORED" in out
    files = sorted(p.name for p in ck.parent.iterdir())
    assert files == ["rv.optim.pt", "rv.pt", "rv_1.pt"]  # no .tmp, ONE sidecar
    final = torch.load(ck, map_location="cpu", weights_only=False)
    assert final["anneal_control_applied"] is None  # ignored != applied
    side = torch.load(ck.parent / "rv.optim.pt", map_location="cpu", weights_only=False)
    # final save ran right after the rv_1 mid save: valid for BOTH stamps
    assert side["update_counter"] == 2 and side["same_state_counters"] == [1]
    assert "optimizer" not in final  # moments never ride in the checkpoints

    # --- a live edit during a run is applied AND stamped
    applied = '{"lr": 8e-5, "target_kl": 0.4}'

    real_update = train.PPOTrainer.update

    def edit_then_update(self, *a, **k):  # the operator edits mid-run
        ctrl.write_text(applied, encoding="utf-8")
        return real_update(self, *a, **k)

    monkeypatch.setattr(train.PPOTrainer, "update", edit_then_update)
    ck2 = tmp_path / "ck2" / "rv.pt"
    _run_train(ck2, ["--num-updates", "3", "--checkpoint-every", "1"])
    monkeypatch.setattr(train.PPOTrainer, "update", real_update)
    capsys.readouterr()
    mid = torch.load(ck2.parent / "rv_2.pt", map_location="cpu", weights_only=False)
    assert mid["anneal_control_applied"] == applied
    assert mid["update_counter"] == 2

    # --- relaunch from the numbered file, file unchanged -> re-applied + warm
    _run_train(
        tmp_path / "ck3" / "rv.pt",
        ["--num-updates", "1", "--load-checkpoint", str(ck2.parent / "rv_2.pt")],
    )
    out = capsys.readouterr().out
    assert "RE-APPLYING" in out
    assert "lr 0.0003 -> 8e-05" in out and "target_kl 0.5 -> 0.4" in out
    assert "restored Adam moments from rv.optim.pt" in out
    resumed = torch.load(tmp_path / "ck3" / "rv.pt", map_location="cpu", weights_only=False)
    assert resumed["anneal_control_applied"] == applied  # stamp carried on

    # --- an OLDER numbered file: sidecar counter mismatch -> cold, said so
    _run_train(
        tmp_path / "ck4" / "rv.pt",
        ["--num-updates", "1", "--load-checkpoint", str(ck2.parent / "rv_1.pt")],
    )
    out = capsys.readouterr().out
    assert "Adam starts COLD" in out and "restored Adam moments" not in out

    # --- file edited while the run was down -> NOT re-applied, loudly
    ctrl.write_text('{"lr": 1e-5}', encoding="utf-8")
    _run_train(
        tmp_path / "ck5" / "rv.pt",
        ["--num-updates", "1", "--load-checkpoint", str(ck2.parent / "rv_2.pt")],
    )
    out = capsys.readouterr().out
    assert "IGNORED" in out and "RE-APPLYING" not in out

    # --- legacy escape hatch: nothing read, nothing written
    ctrl.unlink()
    _run_train(
        tmp_path / "ck6" / "rv.pt",
        ["--num-updates", "1", "--no-optimizer-sidecar",
         "--load-checkpoint", str(ck2.parent / "rv_2.pt")],
    )
    out = capsys.readouterr().out
    assert "LEGACY resume" in out
    assert sorted(p.name for p in (tmp_path / "ck6").iterdir()) == ["rv.pt"]


def test_a13_undecodable_control_file_does_not_kill_the_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    train._CONTROL_WARNED.clear()
    ctrl = tmp_path / "runs" / "anneal_control.json"
    threads = tmp_path / "runs" / "threads.txt"
    real_update = train.PPOTrainer.update

    def corrupt_then_update(self, *a, **k):
        ctrl.write_bytes('{"lr": 1e-4}'.encode("utf-16"))
        threads.write_bytes("8".encode("utf-16"))
        return real_update(self, *a, **k)

    monkeypatch.setattr(train.PPOTrainer, "update", corrupt_then_update)
    _run_train(tmp_path / "ck" / "rv.pt", ["--num-updates", "3", "--checkpoint-every", "0"])
    out = capsys.readouterr().out
    assert (tmp_path / "ck" / "rv.pt").exists()  # ran to the final save
    assert out.count("cannot read") == 2  # one line per file, not per update


def test_obs_rev_is_stamped_and_guards_the_warm_start(tmp_path, monkeypatch, capsys):
    """2026-09-20 obs-SEMANTICS revision: stamped into both checkpoint kinds;
    a warm start across a rev change is refused (naming both revs and both
    ways forward) unless --allow-obs-rev-change; an unstamped checkpoint
    counts as rev 1; older-rev siblings are not seeded into the pool."""
    import plo5bp.encoding as enc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(enc, "OBS_SEMANTICS_REV", 2, raising=False)
    ck = tmp_path / "ck" / "rv.pt"
    _run_train(ck, ["--num-updates", "2", "--checkpoint-every", "1"])
    assert "observation semantics rev = 2" in capsys.readouterr().out
    final = torch.load(ck, map_location="cpu", weights_only=False)
    mid = torch.load(ck.parent / "rv_1.pt", map_location="cpu", weights_only=False)
    assert final["obs_rev"] == 2 and mid["obs_rev"] == 2

    load = ["--num-updates", "1", "--load-checkpoint", str(ck.parent / "rv_1.pt")]
    _run_train(tmp_path / "same" / "rv.pt", load)  # same rev: fine

    # --- the process is on rev 1, the checkpoint on rev 2 -> refused
    monkeypatch.setattr(enc, "OBS_SEMANTICS_REV", 1, raising=False)
    with pytest.raises(SystemExit) as ei:
        _run_train(tmp_path / "never" / "rv.pt", load)
    msg = str(ei.value)
    assert "obs_rev mismatch: checkpoint=2 vs this process=1" in msg
    assert "PLO5BP_OBS_REV=2" in msg and "--allow-obs-rev-change" in msg
    assert not (tmp_path / "never").exists()

    # --- deliberate migration proceeds, loudly; rev-2 siblings stay out of the pool
    capsys.readouterr()
    _run_train(tmp_path / "migrated" / "rv.pt", load + ["--allow-obs-rev-change"])
    out = capsys.readouterr().out
    assert "PRODUCTION BEHAVIOR CHANGE: migrating this stem from obs_rev 2 to 1" in out
    assert "[pool] skip rv_1.pt: obs_rev mismatch (file 2 vs run 1)" in out
    migrated = torch.load(tmp_path / "migrated" / "rv.pt", map_location="cpu", weights_only=False)
    assert migrated["obs_rev"] == 1

    # --- an UNSTAMPED checkpoint is rev 1
    del mid["obs_rev"]
    old = tmp_path / "old" / "legacy_1.pt"
    old.parent.mkdir()
    torch.save(mid, old)
    old_load = ["--num-updates", "1", "--load-checkpoint", str(old)]
    _run_train(tmp_path / "old_ok" / "rv.pt", old_load)  # process rev 1: fine
    monkeypatch.setattr(enc, "OBS_SEMANTICS_REV", 2, raising=False)
    with pytest.raises(SystemExit, match="checkpoint=1 vs this process=2"):
        _run_train(tmp_path / "never2" / "rv.pt", old_load)


def test_a3_clean_stop_keeps_the_numbered_checkpoints_moments(tmp_path, monkeypatch, capsys):
    """A graceful stop writes `<stem>.pt` a few updates PAST the newest
    numbered file, but the guardians resume from the NUMBERED file. The final
    save must not throw that file's Adam moments away, or every stop/restart
    would still be a cold-Adam start."""
    monkeypatch.chdir(tmp_path)
    ck = tmp_path / "ck" / "rv.pt"
    # mid save at update idx 2 (-> rv_2.pt), one more update, then the final.
    _run_train(ck, ["--num-updates", "4", "--checkpoint-every", "2"])
    capsys.readouterr()
    side = torch.load(ck.parent / "rv.optim.pt", map_location="cpu", weights_only=False)
    assert side["update_counter"] == 4 and side["same_state_counters"] == []
    assert side["previous"]["update_counter"] == 2
    a = side["optimizer_state"][0]["exp_avg"]
    b = side["previous"]["optimizer_state"][0]["exp_avg"]
    assert not torch.equal(a, b)  # genuinely two different optimizer states

    for src, counter in (("rv_2.pt", 2), ("rv.pt", 4)):
        _run_train(
            tmp_path / f"resume_{counter}" / "rv.pt",
            ["--num-updates", "1", "--load-checkpoint", str(ck.parent / src)],
        )
        out = capsys.readouterr().out
        assert f"restored Adam moments from rv.optim.pt (u{counter}," in out, out

    # the next MID save drops the carried entry again (one rolling state)
    _run_train(ck, ["--num-updates", "2", "--checkpoint-every", "1",
                    "--load-checkpoint", str(ck)])
    side = torch.load(ck.parent / "rv.optim.pt", map_location="cpu", weights_only=False)
    assert "previous" not in side

