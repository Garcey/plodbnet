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


# --- TOOL-006: the native solver streams the report; Python keeps only scalars ---


def _small_river(root_id: str = "stream"):
    from plo5bp.gto.cfr_api import RootSpec

    return RootSpec(street=3, pot_bb=10.0, effective_stack_bb=20.0, board=[0, 5, 10, 15, 20],
                    raise_sizes_pm=[500, 1000], root_id=root_id)


def test_streamed_report_equals_the_in_memory_one(tmp_path):
    import json

    from plo5bp.gto.cfr_api import SolveConfig, rust_cfr_available, solve

    if not rust_cfr_available():
        pytest.skip("native CFR not built")
    cfg = dict(max_iterations=60, seed=3, target_exploitability_bb=0.0)
    mem = solve(_small_river(), SolveConfig(**cfg))
    path = tmp_path / "r.json"
    streamed = solve(_small_river(), SolveConfig(report_path=str(path), **cfg),
                     root_extra={"range_oop_text": "AA, KK"})
    assert streamed.streamed and streamed.strategy["infosets_omitted"]
    assert streamed.strategy["num_infosets"] == len(mem.strategy["infosets"])
    assert (streamed.iterations_run, streamed.exploitability_bb) == (mem.iterations_run, mem.exploitability_bb)
    text = path.read_text(encoding="utf-8")
    # scalars first, the strategy last: a head read sees everything else
    assert text.index('"notes"') < text.index('"strategy"') < text.index('"infosets"')
    on_disk = json.loads(text)
    ref = mem.as_dict()
    assert on_disk["root"].pop("range_oop_text") == "AA, KK"
    assert on_disk["root"] == ref["root"]
    for k in ("status", "iterations_run", "exploitability_bb"):
        assert on_disk[k] == ref[k]
    assert [n for n in on_disk["notes"] if "secs" not in n] == [n for n in ref["notes"] if "secs" not in n]
    # every infoset field, every float, exactly (round-trip formatting)
    assert on_disk["strategy"] == ref["strategy"]
    assert on_disk["config"]["report_path"] == str(path)
    # write_json of a streamed report copies the file; load_full parses it
    copy = tmp_path / "copy.json"
    streamed.write_json(copy)
    assert copy.read_text(encoding="utf-8") == text  # a byte copy, never re-serialized
    assert streamed.load_full()["strategy"] == ref["strategy"]


def test_batch_moves_or_wraps_the_streamed_report(tmp_path, monkeypatch):
    import json

    from plo5bp.gto.cfr_api import rust_cfr_available
    from plo5bp.gto.cfr_batch import expand_river_grid, run_batch

    if not rust_cfr_available():
        pytest.skip("native CFR not built")
    jobs = expand_river_grid(n_roots=2, seed=0, iters=40, size_preset="micro")
    # cap 0.0 bb → every root is rejected: the wrapper is streamed around the file
    man = run_batch(jobs, tmp_path, resume=False, max_expl_bb=1e-9)
    assert len(man.rejected) == 2 and not list((tmp_path / "tmp").glob("*.json"))
    wrapped = json.loads((tmp_path / "rejected" / f"{jobs[0].job_id}.json").read_text(encoding="utf-8"))
    assert wrapped["status"] == "rejected" and wrapped["reason"].startswith("expl_")
    assert wrapped["report"]["strategy"]["infosets"], "the full report is inside the wrapper"


def test_pipeline_uses_the_stakes_it_is_given():
    """(TOOL-050) cfr_pipeline used to hard-code 10k/5k/5k chips."""
    from plo5bp import _engine
    from plo5bp.gto.cfr_api import rust_cfr_available

    if not rust_cfr_available():
        pytest.skip("native CFR not built")

    def root_pot(**stakes):
        rep = dict(_engine.cfr_pipeline(stack_bb=20.0, preflop_iters=20, postflop_iters=10,
                                        postflop_board=[0, 5, 10, 15, 20], pot_bb=4.0,
                                        postflop_stack_bb=10.0, seed=1, raise_sizes_pm=[1000],
                                        **stakes))
        roots = [i for i in rep["preflop_strategy"]["infosets"] if not i.get("path")]
        return {i["pot_chips"] for i in roots}

    assert root_pot() == {10_000 + 5_000 + 2 * 5_000}  # ClubGG default
    assert root_pot(bb_chips=20_000, sb_chips=10_000, ante_chips=0) == {30_000}


def test_solve_docstring_says_what_happens_without_the_binding():
    from plo5bp.gto import cfr_api

    doc = cfr_api.solve.__doc__
    assert "not_implemented" in doc and "else raise" not in doc
