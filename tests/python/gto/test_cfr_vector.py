"""(TOOL-008) Full-range DCFR (``algorithm="dcfr_vector"``) through the Python API.

The Rust tests pin the solver itself (its best response equals ``dcfr.rs``'s
evaluator, parallel == serial, convergence). These check what Python callers
rely on: the tag is accepted, the report is the usual report (rows, dump fields,
EV / equity, an exact exploitability the teacher counts as verified), the memory
estimate knows the full tree, and the batch grids can ask for it.
"""

from __future__ import annotations

import pytest

from plo5bp.gto.cfr_api import RootSpec, SolveConfig, estimate_memory, rust_cfr_available, solve

pytestmark = pytest.mark.skipif(not rust_cfr_available(), reason="native CFR not built")


def _river(**kw) -> RootSpec:
    return RootSpec(street=3, pot_bb=10.0, effective_stack_bb=20.0, board=[3, 17, 22, 40, 51],
                    raise_sizes_pm=[500, 1000], root_id="vec_river", **kw)


def _needs_vector_engine() -> None:
    try:
        solve(_river(), SolveConfig(max_iterations=1, algorithm="dcfr_vector", target_exploitability_bb=0.0))
    except ValueError as e:  # an engine built before TOOL-008
        if "unknown algorithm" in str(e):
            pytest.skip("engine predates dcfr_vector — rebuild it")
        raise


def test_full_range_river_solve_is_exact_and_fast():
    _needs_vector_engine()
    rep = solve(_river(), SolveConfig(max_iterations=150, algorithm="dcfr_vector",
                                      target_exploitability_bb=0.0, use_isomorphism=False))
    assert rep.status == "ok" and rep.iterations_run == 150
    assert rep.exploitability_bb is not None and rep.exploitability_bb < 0.05
    assert any(n.startswith("expl_kind=exact_infoset") for n in rep.notes)
    assert rep.notes[0].startswith("DCFR-vector River")
    rows = rep.strategy["infosets"]
    assert rows and all(r["visits"] == 150 for r in rows)
    assert all(0.0 <= r["equity"] <= 1.0 and r["ev_bb"] is not None for r in rows)
    assert all(r["private_kind"] == "combo" and r["raw_combo"] == r["private_id"] for r in rows)

    from plo5bp.gto.teacher import expl_provenance

    assert expl_provenance(rep).verified


def test_target_stop_is_still_a_verified_final_number():
    _needs_vector_engine()
    rep = solve(_river(), SolveConfig(max_iterations=2000, algorithm="dcfr_vector",
                                      target_exploitability_bb=0.2))
    assert rep.iterations_run < 2000
    assert "early_stop=target_exploitability" in rep.notes
    assert rep.exploitability_bb <= 0.2

    from plo5bp.gto.teacher import expl_provenance

    assert expl_provenance(rep).verified  # the exact best response ran at the stop


def test_ranges_and_turn_roots():
    _needs_vector_engine()
    root = RootSpec(street=2, pot_bb=10.0, effective_stack_bb=10.0, board=[0, 5, 10, 15],
                    raise_sizes_pm=[], range_oop="AA,KK,JTs", range_ip="TT,99,AKs,87s", root_id="vec_turn")
    rep = solve(root, SolveConfig(max_iterations=120, algorithm="dcfr_vector",
                                  target_exploitability_bb=0.0, thread_num=4))
    assert rep.exploitability_bb < 0.05
    assert any("parallel=river subtrees" in n for n in rep.notes)
    turn_rows = [r for r in rep.strategy["infosets"] if r["street"] == 2]
    assert turn_rows and all(r["equity"] is not None for r in turn_rows)


def test_unsuitable_roots_are_refused_with_the_algorithm_to_use():
    _needs_vector_engine()
    flop = RootSpec(street=1, pot_bb=10.0, effective_stack_bb=20.0, board=[0, 5, 10], raise_sizes_pm=[500])
    with pytest.raises(ValueError, match="heads-up river and turn"):
        solve(flop, SolveConfig(max_iterations=5, algorithm="dcfr_vector"))


def test_the_estimate_knows_the_full_tree():
    _needs_vector_engine()
    vec = estimate_memory(_river(), SolveConfig(algorithm="dcfr_vector"))
    assert vec["algorithm"] == "dcfr_vector" and vec["refuse_reason"] is None
    rep = solve(_river(), SolveConfig(max_iterations=1, algorithm="dcfr_vector", target_exploitability_bb=0.0))
    # full ranges: every hand reaches every node, so the row bound is exact
    assert vec["est_infosets"] == len(rep.strategy["infosets"])
    flop = RootSpec(street=1, pot_bb=10.0, effective_stack_bb=20.0, board=[0, 5, 10], raise_sizes_pm=[500])
    assert "heads-up river and turn" in estimate_memory(flop, SolveConfig(algorithm="dcfr_vector"))["refuse_reason"]
    tight = estimate_memory(_river(), SolveConfig(algorithm="dcfr_vector", ram_budget_mb=1))
    assert "memory budget" in tight["refuse_reason"]


def test_the_estimate_counts_the_hands_in_the_ranges():
    _needs_vector_engine()
    for street, board in ((3, [3, 17, 22, 40, 51]), (2, [3, 17, 22, 40])):
        root = RootSpec(street=street, pot_bb=10.0, effective_stack_bb=10.0, board=board,
                        raise_sizes_pm=[500], range_oop="AA,KK,QJs,76s", range_ip="TT+,AKo,98s")
        est = estimate_memory(root, SolveConfig(algorithm="dcfr_vector"))
        rep = solve(root, SolveConfig(max_iterations=1, algorithm="dcfr_vector", target_exploitability_bb=0.0))
        # one iteration: every hand in the range has reached every node
        assert est["est_infosets"] == len(rep.strategy["infosets"]), street
        full = estimate_memory(RootSpec(street=street, pot_bb=10.0, effective_stack_bb=10.0, board=board,
                                        raise_sizes_pm=[500]), SolveConfig(algorithm="dcfr_vector"))
        assert est["est_mb"] < full["est_mb"] / 10


def test_batch_grids_can_ask_for_it():
    from plo5bp.gto.cfr_batch import expand_river_grid, expand_river_spr_grid

    jobs = expand_river_grid(n_roots=1, iters=50, streets=[1, 2, 3], algorithm="dcfr_vector")
    assert [j.config.algorithm for j in jobs] == ["dcfr", "dcfr_vector", "dcfr_vector"]
    assert all(j.config.algorithm == "dcfr" for j in expand_river_grid(n_roots=1, streets=[3]))
    spr = expand_river_spr_grid(n_boards=1, spr_points=(1.0,), iters=100, algorithm="dcfr_vector")
    assert spr[0].config.algorithm == "dcfr_vector"
