"""Real-path tests for native NLH CFR (Python → Rust)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plo5bp.gto.cfr_api import (
    RootSpec,
    SolveConfig,
    induce_range,
    rust_cfr_available,
    solve,
    solve_kuhn,
)
from plo5bp.gto.cfr_batch import expand_river_grid, run_batch
from plo5bp.gto.cfr_export import export_dir, strategy_to_labels


pytestmark = pytest.mark.skipif(
    not rust_cfr_available(),
    reason="rebuild extension: maturin develop --release",
)


def test_rust_cfr_available():
    assert rust_cfr_available() is True


def test_kuhn_near_nash():
    rep = solve_kuhn(8000)
    assert abs(rep["value_p0"] - rep["nash_value"]) < 0.03
    assert rep["exploitability"] < 0.05


def test_river_hu_dcfr_produces_strategy():
    root = RootSpec.river_hu(
        [0, 5, 10, 15, 20],
        pot_bb=10.0,
        effective_stack_bb=20.0,
        size_preset="micro",
    )
    cfg = SolveConfig(max_iterations=80, seed=1, algorithm="dcfr", target_exploitability_bb=0.0)
    rep = solve(root, cfg)
    assert rep.status == "ok"
    assert rep.iterations_run == 80
    assert len(rep.strategy["infosets"]) > 0
    for iset in rep.strategy["infosets"][:5]:
        s = sum(iset["probs"])
        assert abs(s - 1.0) < 1e-5


def test_preflop_mccfr_smoke():
    root = RootSpec.preflop_hu(100.0)
    cfg = SolveConfig(max_iterations=100, algorithm="mccfr_es", seed=2)
    rep = solve(root, cfg)
    assert rep.status == "ok"
    assert len(rep.strategy["infosets"]) > 0


def test_multiway_smoke():
    root = RootSpec(
        street=3,
        pot_bb=12.0,
        effective_stack_bb=25.0,
        board=[0, 5, 10, 15, 20],
        num_seats=3,
        raise_sizes_pm=[500, 1000],
        root_id="mw3_test",
    )
    cfg = SolveConfig(max_iterations=40, algorithm="mccfr_es", seed=3)
    rep = solve(root, cfg)
    assert rep.status == "ok"
    assert any("sidepot" in n or "showdown" in n for n in rep.notes)


def test_flop_and_turn_solve():
    flop = RootSpec(
        street=1,
        pot_bb=8.0,
        effective_stack_bb=20.0,
        board=[0, 5, 10],
        raise_sizes_pm=[500, 1000],
        root_id="flop_t",
    )
    # Flop auto-upgrades to ochs buckets
    rep_f = solve(
        flop,
        SolveConfig(
            max_iterations=40,
            seed=4,
            target_exploitability_bb=0.0,
            card_abstraction="ochs",
        ),
    )
    assert rep_f.status == "ok"
    assert any("bucket" in n.lower() or "ochs" in n.lower() or "card_abs" in n for n in rep_f.notes)
    turn = RootSpec(
        street=2,
        pot_bb=10.0,
        effective_stack_bb=20.0,
        board=[0, 5, 10, 15],
        raise_sizes_pm=[500, 1000],
        root_id="turn_t",
    )
    rep_t = solve(turn, SolveConfig(max_iterations=40, seed=5, target_exploitability_bb=0.0))
    assert rep_t.status == "ok"


def test_pipeline_api():
    from plo5bp import _engine  # type: ignore

    rep = dict(
        _engine.cfr_pipeline(
            stack_bb=50.0,
            preflop_iters=60,
            postflop_iters=30,
            postflop_board=[0, 5, 10, 15, 20],
            pot_bb=12.0,
            postflop_stack_bb=30.0,
            seed=1,
        )
    )
    assert rep["preflop_status"] == "ok"
    assert rep["postflop_status"] == "ok"
    assert rep["preflop_infosets"] > 0


def test_multiway_unequal_stacks_and_real_expl():
    root = RootSpec(
        street=3,
        pot_bb=15.0,
        effective_stack_bb=30.0,
        board=[0, 5, 10, 15, 20],
        num_seats=3,
        raise_sizes_pm=[500, 1000],
        stacks_bb=[40.0, 25.0, 15.0],
        root_id="mw3_uneq",
    )
    rep = solve(root, SolveConfig(max_iterations=50, algorithm="mccfr_es", seed=8))
    assert rep.status == "ok"
    assert rep.exploitability_bb is not None
    assert any("unequal" in n for n in rep.notes)
    assert any("sidepot" in n for n in rep.notes)


def test_iso_and_thread_notes_on_river():
    root = RootSpec.river_hu(
        [0, 5, 10, 15, 20], pot_bb=10.0, effective_stack_bb=20.0, size_preset="micro"
    )
    cfg = SolveConfig(
        max_iterations=40,
        seed=2,
        target_exploitability_bb=0.0,
        use_isomorphism=True,
        thread_num=2,
    )
    rep = solve(root, cfg)
    assert rep.status == "ok"
    assert any("isomorphism=on" in n for n in rep.notes)
    assert any("rayon" in n or "thread_num" in n for n in rep.notes)


def test_multiway_preflop():
    # Push/fold keeps MC-BR expl tractable for unit tests.
    root = RootSpec.preflop_pushfold(num_seats=3, stack_bb=10.0, ante_chips=0)
    root.root_id = "mw3_pf_py"
    rep = solve(root, SolveConfig(max_iterations=60, algorithm="mccfr_es", seed=3))
    assert rep.status == "ok"
    assert len(rep.strategy["infosets"]) > 0
    assert any("multiway preflop" in n for n in rep.notes)
    assert any("mc_br_proxy" in n for n in rep.notes)
    assert rep.is_mc_br_proxy


def test_multiway_postflop_board_street_key_note():
    root = RootSpec(
        street=1,
        pot_bb=10.0,
        effective_stack_bb=20.0,
        board=[0, 5, 10],
        num_seats=3,
        raise_sizes_pm=[500, 1000],
        root_id="mw3_flop_key",
    )
    rep = solve(root, SolveConfig(max_iterations=30, algorithm="mccfr_es", seed=4))
    assert rep.status == "ok"
    assert any("board+street" in n for n in rep.notes)
    assert any("mc_br_proxy" in n for n in rep.notes)


def test_kuhn_tight_expl():
    rep = solve_kuhn(15000)
    assert abs(rep["value_p0"] - rep["nash_value"]) < 0.02
    assert rep["exploitability"] < 0.01


def test_range_parse_used():
    root = RootSpec.river_hu(
        [0, 5, 10, 15, 20],
        pot_bb=10.0,
        effective_stack_bb=15.0,
        size_preset="micro",
    )
    root.range_oop = "AA,KK"
    root.range_ip = "random"
    rep = solve(root, SolveConfig(max_iterations=30, seed=6, target_exploitability_bb=0.0))
    assert rep.status == "ok"
    assert any("parsed" in n or "ranges" in n for n in rep.notes)


def test_induce_range():
    prior = [0.5, 0.5]
    probs = [[1.0, 0.0], [0.0, 1.0]]
    post = induce_range(prior, probs, 0)
    assert abs(post[0] - 1.0) < 1e-9
    assert abs(post[1]) < 1e-9


def test_batch_dry_run_and_resume(tmp_path: Path):
    jobs = expand_river_grid(n_roots=2, seed=0, iters=30, size_preset="micro")
    out = tmp_path / "batch"
    man = run_batch(jobs, out, dry_run=True)
    assert len(man.jobs) == 2
    assert (out / "plan.json").exists()
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["dry_run"] is True

    man2 = run_batch(jobs, out, dry_run=False, resume=True, max_expl_bb=None)
    assert len(man2.completed) == 2
    assert not man2.failed

    # Resume skips completed
    man3 = run_batch(jobs, out, dry_run=False, resume=True)
    assert len(man3.skipped) == 2
    assert len(man3.completed) == 0


def test_export_labels(tmp_path: Path):
    root = RootSpec.river_hu(
        [1, 6, 11, 16, 21],
        pot_bb=8.0,
        effective_stack_bb=15.0,
        size_preset="micro",
    )
    rep = solve(root, SolveConfig(max_iterations=40, seed=9, target_exploitability_bb=0.0))
    strat_path = tmp_path / "s.json"
    rep.write_json(strat_path)
    labels = strategy_to_labels(rep.as_dict(), source="rust_cfr_river")
    assert len(labels) > 0
    assert labels[0].source == "rust_cfr_river"
    assert abs(sum(labels[0].gate_probs) - 1.0) < 1e-5 or sum(labels[0].gate_probs) > 0
    # Raise actions must map onto NLH anchors for PolicyNet
    for lab in labels[:5]:
        for a in lab.action_probs:
            if a.gate == "raise":
                assert a.anchor_k is not None
                assert 0 <= int(a.anchor_k) < 12

    out = tmp_path / "labels.jsonl"
    n = export_dir(strat_path, out, source="rust_cfr_river")
    assert n > 0
    assert n == len(labels)  # full export, no 64-cap
    line = out.read_text(encoding="utf-8").strip().splitlines()[0]
    d = json.loads(line)
    assert d["source"] == "rust_cfr_river"
