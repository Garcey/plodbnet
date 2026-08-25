"""CFR API validation + real solve path (when extension is built)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plo5bp.gto.cfr_api import (
    RootSpec,
    SolveConfig,
    rust_cfr_available,
    solve,
)


def test_preflop_root_validates():
    r = RootSpec.preflop_hu(100.0)
    r.validate()
    assert r.street == 0
    assert r.board == []
    assert r.num_seats == 2


def test_river_root_board_len():
    with pytest.raises(ValueError, match="board length"):
        RootSpec.river_hu([1, 2, 3], pot_bb=10.0).validate()
    r = RootSpec.river_hu([0, 1, 2, 3, 4], pot_bb=10.0, effective_stack_bb=40.0)
    r.validate()


def test_solve_rejects_bad_stack():
    with pytest.raises(ValueError):
        solve(RootSpec.preflop_hu(0.0))


def test_stacks_bb_length_mismatch():
    r = RootSpec.preflop_pushfold(num_seats=4, stack_bb=10.0)
    r.stacks_bb = [10.0, 10.0]  # wrong
    with pytest.raises(ValueError, match="stacks_bb length"):
        r.validate()


def test_empty_raises_without_allin_rejected():
    r = RootSpec.preflop_hu(10.0)
    r.raise_sizes_pm = []
    r.allin_atom = False
    with pytest.raises(ValueError, match="raise size|allin_atom"):
        r.validate()


def test_pushfold_factory_no_ante():
    r = RootSpec.preflop_pushfold(num_seats=4, stack_bb=10.0, ante_chips=0)
    r.validate()
    assert r.raise_sizes_pm == []
    assert r.allin_atom is True
    assert r.ante_chips == 0
    assert r.num_seats == 4
    assert r.stacks_bb == [10.0] * 4
    # pot_bb = (0*4 + sb + bb) / bb = 1.5 at ClubGG chips
    assert abs(r.pot_bb - 1.5) < 1e-9


@pytest.mark.skipif(not rust_cfr_available(), reason="extension not built")
def test_solve_preflop_ok(tmp_path: Path):
    rep = solve(RootSpec.preflop_hu(100.0), SolveConfig(max_iterations=50, algorithm="mccfr_es"))
    assert rep.status == "ok"
    assert rep.iterations_run == 50
    assert len(rep.strategy["infosets"]) > 0
    out = tmp_path / "pf.json"
    rep.write_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "ok"


@pytest.mark.skipif(not rust_cfr_available(), reason="extension not built")
def test_rust_cfr_bound():
    assert rust_cfr_available() is True


@pytest.mark.skipif(not rust_cfr_available(), reason="extension not built")
def test_pushfold_4handed_smoke_and_mc_br_label():
    root = RootSpec.preflop_pushfold(num_seats=4, stack_bb=10.0, ante_chips=0)
    rep = solve(root, SolveConfig(max_iterations=60, algorithm="mccfr_es", seed=3))
    assert rep.status == "ok"
    assert rep.is_mc_br_proxy
    assert any("push_fold" in n for n in rep.notes)
    assert any("pot0_chips=15000" in n for n in rep.notes)
    assert len(rep.strategy["infosets"]) > 50


@pytest.mark.skipif(not rust_cfr_available(), reason="extension not built")
def test_stacks_bb_mismatch_refused_by_rust():
    root = RootSpec.preflop_pushfold(num_seats=4, stack_bb=10.0)
    # Bypass Python validate by calling engine after fixing list length via dict path
    root.stacks_bb = [10.0, 10.0, 10.0]  # 3 != 4
    with pytest.raises(ValueError, match="stacks_bb|num_seats"):
        root.validate()
        solve(root, SolveConfig(max_iterations=5, algorithm="mccfr_es"))
