"""Holdout probe gates + stamp + fail path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.gto.labels import (
    ActionProb,
    LABEL_SCHEMA_VERSION,
    LabelRecord,
    make_smoke_label,
    write_jsonl,
)
from plo5bp.gto.policy_net import (
    build_policy_net,
    is_validated_gto_checkpoint,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from plo5bp.gto.probe import (
    ProbeGates,
    evaluate_probe_gates,
    probe_checkpoint,
    probe_policy_vs_labels,
    stamp_probe_on_checkpoint,
)
from plo5bp.gto.train import TrainConfig, train_policy_net
from plo5bp.gto.dataset import SupervisedRow
from plo5bp.gto.obs_from_label import labels_to_supervised_rows
from plo5bp.sizing import NLH_ANCHOR_SPEC


def _pure_fold_label(seed: int = 0) -> LabelRecord:
    return make_smoke_label(seed=seed, trash_fold=True)


def _pure_jam_label(seed: int = 1) -> LabelRecord:
    return make_smoke_label(seed=seed, trash_fold=False)


def test_evaluate_gates_pass_and_fail():
    from plo5bp.gto.probe import ProbeReport

    good = ProbeReport(
        n=20, pure_n=10, pure_agree=0.95, mean_gate_kl=0.1, mean_gate_acc=0.9
    )
    r = evaluate_probe_gates(good, ProbeGates(min_n=5, min_pure_agree=0.9))
    assert r.passed
    assert not r.reasons

    bad_kl = ProbeReport(
        n=20, pure_n=10, pure_agree=0.95, mean_gate_kl=0.9, mean_gate_acc=0.5
    )
    r2 = evaluate_probe_gates(
        bad_kl, ProbeGates(min_n=5, max_mean_gate_kl=0.5, min_pure_agree=0.9)
    )
    assert not r2.passed
    assert any("mean_gate_kl" in x for x in r2.reasons)

    bad_pure = ProbeReport(
        n=20, pure_n=10, pure_agree=0.5, mean_gate_kl=0.1, mean_gate_acc=0.5
    )
    r3 = evaluate_probe_gates(
        bad_pure, ProbeGates(min_n=5, min_pure_agree=0.9)
    )
    assert not r3.passed
    assert any("pure_agree" in x for x in r3.reasons)


def test_probe_self_fit_passes(tmp_path: Path):
    """Train on pure canaries; probe same labels → should pass easy gates."""
    labels = [_pure_fold_label(i) for i in range(8)] + [
        _pure_jam_label(100 + i) for i in range(8)
    ]
    rows = labels_to_supervised_rows(labels)
    assert rows
    ckpt = tmp_path / "fit.pt"
    train_policy_net(
        rows,
        ckpt,
        cfg=TrainConfig(
            hidden_dim=64,
            num_layers=1,
            epochs=40,
            batch_size=16,
            lr=3e-3,
            log_every=9999,
            seed=0,
        ),
        meta={"source": "rust_cfr", "n_train": len(rows)},
    )
    holdout = tmp_path / "holdout.jsonl"
    write_jsonl(holdout, labels)
    result = probe_checkpoint(
        ckpt,
        holdout,
        gates=ProbeGates(
            min_n=4,
            min_pure_agree=0.75,
            max_mean_gate_kl=0.8,
            min_pure_n=2,
        ),
    )
    assert result.report.n >= 4
    assert result.passed, result.reasons
    stamp_probe_on_checkpoint(ckpt, result)
    assert is_validated_gto_checkpoint(ckpt)
    _, meta = load_policy_checkpoint(ckpt)
    assert meta.get("is_gto_validated") is True
    assert meta.get("probe", {}).get("passed") is True


def test_probe_random_model_fails_gates(tmp_path: Path):
    """Untrained net vs pure labels → high KL / low pure agree → FAIL."""
    labels = [_pure_fold_label(i) for i in range(6)] + [
        _pure_jam_label(50 + i) for i in range(6)
    ]
    holdout = tmp_path / "h.jsonl"
    write_jsonl(holdout, labels)
    m = build_policy_net(hidden_dim=64, num_layers=1)
    ckpt = tmp_path / "rand.pt"
    save_policy_checkpoint(
        ckpt, m, meta={"source": "rust_cfr", "n_train": 0}
    )
    result = probe_checkpoint(
        ckpt,
        holdout,
        gates=ProbeGates(
            min_n=4,
            min_pure_agree=0.90,
            max_mean_gate_kl=0.3,
            min_pure_n=2,
        ),
    )
    assert not result.passed
    assert result.reasons
    stamp_probe_on_checkpoint(ckpt, result)
    assert not is_validated_gto_checkpoint(ckpt)


def test_probe_empty_labels_fails(tmp_path: Path):
    m = build_policy_net(hidden_dim=32, num_layers=1)
    ckpt = tmp_path / "e.pt"
    save_policy_checkpoint(ckpt, m, meta={"source": "rust_cfr"})
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    result = probe_checkpoint(
        ckpt, empty, gates=ProbeGates(min_n=1)
    )
    assert not result.passed
