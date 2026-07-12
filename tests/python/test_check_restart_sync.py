"""check_restart_sync.py: guardian flags vs anneal_control baseline.

The tool exists because anneal_control.json is a silent STARTUP BASELINE —
live-tuned values not baked back into guardian flags revert on relaunch.
Pins: flag parsing (both `--f v` and `--f=v`, line continuations, comments,
last-wins), the uniform-vs-divergent tier_ent rule, and the exit codes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_restart_sync import compare, main, parse_guardian_flags

GUARDIAN = """#!/bin/bash
# comment with a decoy: --lr 9.9
FLAGS="--v6 --hidden-dim 2048 --num-layers 4 \\
  --entropy-coef 0.25 --lr 1.5e-4 --lr-warmup-updates 75 \\
  --clip-room-mid 0.07 --target-kl 0.5"
python scripts/train.py $FLAGS --q-fold-sup-coef=15.0
"""


def test_parse_guardian_flags():
    flags = parse_guardian_flags(GUARDIAN)
    assert flags["--lr"] == 1.5e-4          # comment decoy ignored
    assert flags["--entropy-coef"] == 0.25  # across a continuation
    assert flags["--clip-room-mid"] == 0.07
    assert flags["--q-fold-sup-coef"] == 15.0  # `=` form


def test_last_occurrence_wins():
    flags = parse_guardian_flags("train --lr 1e-3\ntrain --lr 2e-3\n")
    assert flags["--lr"] == 2e-3


def _rows_ok(rows):
    return all(ok for _, _, _, ok in rows)


def test_compare_matching():
    control = {
        "lr": 1.5e-4,
        "tier_ent": {"clubgg": 0.25, "clubgg_deep": 0.25, "deep": 0.25},
        "clip_room_mid": 0.07,
        "q_fold_sup_coef": 15.0,
    }
    rows = compare(parse_guardian_flags(GUARDIAN), control)
    assert _rows_ok(rows)


def test_compare_mismatch_and_divergent_tiers():
    drifted = {
        "lr": 1.0e-3,  # live-tuned; guardian still 1.5e-4 -> mismatch
        "tier_ent": {"clubgg": 0.25, "deep": 0.15},  # divergent -> always flagged
    }
    rows = compare(parse_guardian_flags(GUARDIAN), drifted)
    by_name = {name: ok for name, _, _, ok in rows}
    assert by_name["--lr"] is False
    assert any("tiers differ" in name and not ok for name, _, _, ok in rows)
    # knobs absent from the control file are not enforced
    assert by_name["--clip-room-mid"] is True


def test_main_exit_codes(tmp_path):
    g = tmp_path / "guardian.sh"
    c = tmp_path / "anneal_control.json"
    g.write_text(GUARDIAN, encoding="utf-8")
    c.write_text(
        json.dumps({"lr": 1.5e-4, "clip_room_mid": 0.07}), encoding="utf-8"
    )
    assert main([str(g), str(c)]) == 0
    c.write_text(json.dumps({"lr": 3.0e-4}), encoding="utf-8")
    assert main([str(g), str(c)]) == 1
