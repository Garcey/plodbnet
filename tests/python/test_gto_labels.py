"""Phase 0 GTO label factory / schema / metrics smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plo5bp.config import VARIANT_NLH
from plo5bp.gto.factory import run_factory
from plo5bp.gto.labels import (
    make_smoke_label,
    map_size_to_anchor,
    normalize_gate_probs,
    read_jsonl,
    write_jsonl,
)
from plo5bp.gto.metrics import (
    gate_kl,
    river_exact_available,
    score_model_gates,
    smoke_metric_pass,
)
from plo5bp.gto.roots import (
    CLUBGG_NLH_ROOT,
    iter_spr_grid,
    sample_train_roots,
)
from plo5bp.gto.cfr_api import rust_cfr_available
from plo5bp.sizing import NLH_ANCHOR_SPEC


def test_clubgg_root_locked():
    r = CLUBGG_NLH_ROOT
    assert r.name == "clubgg_5_10_5"
    assert r.bb == 10_000
    assert r.sb == 5_000
    assert r.ante == 5_000
    assert r.variant == VARIANT_NLH
    cfg = r.game_config(num_seats=6, stack_bb=100.0)
    assert cfg.sb == 5_000
    assert cfg.ante == 5_000
    assert cfg.bb == 10_000
    assert cfg.starting_stack == 1_000_000
    assert cfg.variant == VARIANT_NLH


def test_sample_train_roots_hu_heavy():
    roots = sample_train_roots(40, seed=0, seats=(2, 3, 6), streets=(1, 2, 3))
    assert len(roots) == 40
    hu = sum(1 for r in roots if r.num_seats == 2)
    assert hu >= 25  # ~85% HU
    assert all(r.root_name == "clubgg_5_10_5" for r in roots)
    assert all(r.street in (1, 2, 3) for r in roots)


def test_spr_grid_deterministic():
    a = list(iter_spr_grid(seed0=0))
    b = list(iter_spr_grid(seed0=0))
    assert a == b
    assert len(a) == 3 * 5  # streets × spr_points


def test_normalize_and_map_anchor():
    g = normalize_gate_probs(0.2, 0.2, 0.6)
    assert pytest.approx(sum(g), abs=1e-9) == 1.0
    # All-in chips should map to last atom
    k = map_size_to_anchor(
        500_000,
        min_raise=10_000,
        max_raise=500_000,
        pot=100_000,
        to_call=0,
    )
    assert k == NLH_ANCHOR_SPEC.count - 1


def test_smoke_label_jsonl_roundtrip(tmp_path: Path):
    labs = [make_smoke_label(seed=1, trash_fold=True), make_smoke_label(seed=2, trash_fold=False)]
    path = tmp_path / "x.jsonl"
    n = write_jsonl(path, labs)
    assert n == 2
    back = list(read_jsonl(path))
    assert len(back) == 2
    assert back[0].source == "synthetic_smoke"
    assert back[0].gate_probs[0] > 0.9
    assert back[1].gate_probs[2] > 0.9


def test_metrics_self_score_passes():
    labs = [make_smoke_label(seed=1), make_smoke_label(seed=2, trash_fold=False)]
    preds = [list(l.gate_probs) for l in labs]
    summary = score_model_gates(labs, preds)
    assert smoke_metric_pass(summary)
    assert summary.mean_gate_kl == pytest.approx(0.0, abs=1e-12)
    assert summary.pure_node_agree == 1.0


def test_gate_kl_positive_when_disagree():
    p = [0.9, 0.05, 0.05]
    q = [0.05, 0.9, 0.05]
    assert gate_kl(p, q) > 1.0


def test_factory_smoke_run(tmp_path: Path):
    report = run_factory(
        n_roots=4,
        seed=0,
        out_dir=tmp_path,
        include_smoke=True,
        use_grid=False,
    )
    assert report.n_labels >= 2  # smoke canaries at minimum
    assert report.smoke_used
    assert Path(report.out_path).is_file()
    manifest = json.loads((tmp_path / "manifest_seed0.json").read_text(encoding="utf-8"))
    assert manifest["root"] == "clubgg_5_10_5"
    assert report.metrics["mean_gate_kl"] == pytest.approx(0.0, abs=1e-9)
    assert report.metrics["pure_node_agree"] == 1.0


def test_river_exact_hook_safe():
    # Must not raise even when the rust_cfr extension is absent.
    assert isinstance(river_exact_available(), bool)
    assert isinstance(rust_cfr_available(), bool)
