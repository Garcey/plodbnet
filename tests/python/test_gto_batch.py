"""Batch factory + obs_from_label + probe (no live solve required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.gto.labels import (
    ActionProb,
    LabelRecord,
    make_smoke_label,
    write_jsonl,
)
from plo5bp.gto.obs_from_label import labels_to_supervised_rows, obs_from_label
from plo5bp.gto.policy_net import build_policy_net, save_policy_checkpoint
from plo5bp.gto.probe import probe_policy_vs_labels
from plo5bp.gto.train import TrainConfig, train_policy_net


def _combo_label(hole: list[int], gate: list[float]) -> LabelRecord:
    lab = make_smoke_label(seed=0, trash_fold=gate[0] > 0.5)
    lab.hero_hole = hole
    lab.gate_probs = gate
    lab.notes = {"aggregate": False}
    lab.board = [48, 44, 12, 8, 4]
    lab.street = 3
    lab.pot_chips = 100_000
    lab.to_call_chips = 0
    lab.min_raise_chips = 10_000
    lab.max_raise_chips = 500_000
    lab.stacks_chips = [500_000, 500_000]
    lab.action_probs = [
        ActionProb("check_call", None, None, 0, gate[1]),
        ActionProb("raise", 6, 1.0, 100_000, gate[2]),
    ]
    return lab


def test_obs_from_label_shape():
    lab = _combo_label([51, 50], [0.0, 0.7, 0.3])
    obs = obs_from_label(lab)
    assert obs is not None
    assert obs.shape == (OBS_DIM_NLH,)
    assert obs.dtype == np.float32
    # hole multi-hot set
    assert obs[51] == 1.0 and obs[50] == 1.0


def test_aggregate_skipped():
    lab = make_smoke_label(seed=1)
    lab.hero_hole = []
    lab.notes = {"aggregate": True}
    assert obs_from_label(lab) is None


def test_labels_to_rows_and_train(tmp_path: Path):
    labs = [
        _combo_label([51, 50], [0.0, 0.9, 0.1]),
        _combo_label([48, 47], [0.0, 0.2, 0.8]),
        _combo_label([12, 8], [0.05, 0.9, 0.05]),
        _combo_label([0, 4], [0.0, 0.6, 0.4]),
    ]
    # pad to a few more
    for i in range(20):
        labs.append(_combo_label([i % 50, (i + 3) % 51], [0.0, 0.8, 0.2]))
    rows = labels_to_supervised_rows(labs)
    assert len(rows) == len(labs)
    out = tmp_path / "pol.pt"
    r = train_policy_net(
        rows,
        out,
        cfg=TrainConfig(
            hidden_dim=64,
            num_layers=1,
            epochs=5,
            batch_size=8,
            lr=1e-3,
            log_every=999,
            value_coef=0.01,
        ),
        meta={"source": "unit"},
    )
    assert r.final_gate_kl < 0.5
    report = probe_policy_vs_labels(
        build_policy_net(hidden_dim=64, num_layers=1),
        labs,
    )
    # untrained tiny net — just ensure probe runs
    assert report.n == len(labs)

    # load trained
    from plo5bp.gto.policy_net import load_policy_checkpoint

    m, _ = load_policy_checkpoint(out)
    report2 = probe_policy_vs_labels(m, labs)
    assert report2.mean_gate_kl is not None
    assert report2.mean_gate_kl < 0.3


def test_write_read_shard(tmp_path: Path):
    labs = [_combo_label([51, 50], [0.0, 0.5, 0.5])]
    path = tmp_path / "x.jsonl"
    write_jsonl(path, labs)
    from plo5bp.gto.labels import read_jsonl

    back = list(read_jsonl(path))
    assert len(back) == 1
    assert back[0].hero_hole == [51, 50]
