"""CFR desktop app — the algorithm menu, Validate's memory estimate, live checks.

- TOOL-008: full-range DCFR (``dcfr_vector``) is offered for heads-up river /
  turn roots, the river presets use it, and a solve through the API finishes with
  an exact exploitability.
- TOOL-017: every algorithm says what the "threads" number means for it, and a
  request the root cannot use falls back to the algorithm that root always ran.
- TOOL-030: the config carries ``expl_check_secs`` to the solver.
- TOOL-032: ``/api/estimate`` answers what a solve will need, or why the solver
  would refuse it, before anything runs.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from plo5bp.cfr_app import server
from plo5bp.cfr_app.server import algorithm_for_root, app
from plo5bp.cfr_app.session import SolveSession, _config_from_dict, root_presets
from plo5bp.gto.cfr_api import rust_cfr_available

RIVER = {"street": 3, "pot_bb": 10, "effective_stack_bb": 20, "board": [12, 28, 38, 41, 45],
         "num_seats": 2, "raise_sizes_pm": [500, 1000]}


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CFR_APP_DATA_DIR", str(tmp_path / "cfr_data"))
    monkeypatch.setenv("CFR_APP_ALLOWED_HOSTS", "testserver")
    monkeypatch.setattr(server, "session", SolveSession())
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize(
    ("street", "seats", "wanted", "runs"),
    [
        (3, 2, "dcfr_vector", "dcfr_vector"),
        (2, 2, "dcfr_vector", "dcfr_vector"),
        (1, 2, "dcfr_vector", "dcfr"),  # flops need buckets: sampled DCFR
        (3, 3, "dcfr_vector", "dcfr"),  # multiway: what "dcfr" always ran there
        (0, 2, "dcfr_vector", "mccfr_es"),
        (0, 2, "dcfr", "mccfr_es"),  # the old preflop rule
        (0, 2, "DCFR", "mccfr_es"),
        (3, 2, "dcfr", "dcfr"),
        (3, 3, "dcfr", "dcfr"),  # unchanged: multiway postflop keeps its tag
        (0, 4, "mccfr_es", "mccfr_es"),
        (3, 2, None, "dcfr"),
    ],
)
def test_the_algorithm_a_root_runs(street, seats, wanted, runs):
    assert algorithm_for_root(street, seats, wanted) == runs


def test_meta_lists_every_algorithm_with_where_it_applies(client):
    meta = client.get("/api/meta").json()
    assert meta["algorithms"] == ["dcfr_vector", "dcfr", "mccfr_es"]
    info = {a["id"]: a for a in meta["algorithm_info"]}
    assert info["dcfr_vector"]["streets"] == [2, 3] and info["dcfr_vector"]["hu_only"]
    assert "Deals per iteration" in info["dcfr"]["threads"]
    assert info["mccfr_es"]["threads"] is None  # the box is disabled for MCCFR


def test_river_presets_use_full_range_dcfr_and_turn_has_one():
    presets = {p["id"]: p for p in root_presets()}
    assert presets["river_hu_standard"]["algorithm"] == "dcfr_vector"
    assert presets["river_hu_micro"]["algorithm"] == "dcfr_vector"
    turn = presets["turn_hu_ranges"]
    assert turn["street"] == 2 and turn["algorithm"] == "dcfr_vector" and len(turn["board"]) == 4
    from plo5bp.cfr_app.ranges import parse_range

    for side in ("range_oop", "range_ip"):  # the app's own parser accepts them
        assert 150 < parse_range(turn[side], turn["board"]).summary()["combos"] < 400
    # the app boots on presets[2]: still the standard river
    assert root_presets()[2]["id"] == "river_hu_standard"


def test_config_carries_the_live_check_interval():
    assert _config_from_dict({"expl_check_secs": 10}).expl_check_secs == 10.0
    assert _config_from_dict({}).expl_check_secs == 0.0
    assert server.ConfigBody().expl_check_secs == 0.0


@pytest.mark.skipif(not rust_cfr_available(), reason="native CFR not built")
def test_estimate_says_what_a_solve_needs(client):
    r = client.post("/api/estimate", json={"root": RIVER, "config": {"algorithm": "dcfr_vector"}}).json()
    if not r.get("ok") and "no cfr_estimate_memory" in str(r.get("error")):
        pytest.skip("engine predates the estimate binding")
    assert r["ok"] and r["algorithm"] == "dcfr_vector" and r["refuse_reason"] is None
    assert 0 < r["est_mb"] < r["budget_mb"]
    assert r["est_infosets"] > 1000 and r["public_nodes"] > 1
    # A full-range request on a flop is estimated as the solve that would run.
    flop = dict(RIVER, street=1, board=[12, 28, 38])
    f = client.post("/api/estimate", json={"root": flop, "config": {"algorithm": "dcfr_vector"}}).json()
    assert f["ok"] and f["algorithm"] == "dcfr" and f["card_abstraction"] == "ochs"
    # A root the app refuses says so the same way Validate does.
    bad = client.post("/api/estimate", json={"root": dict(RIVER, board=[1, 2]), "config": {}}).json()
    assert bad["ok"] is False and bad["error"]
    bad_range = client.post("/api/estimate", json={"root": dict(RIVER, range_oop="AA,ZZ"), "config": {}}).json()
    assert bad_range["ok"] is False and "ZZ" in bad_range["error"]


@pytest.mark.skipif(not rust_cfr_available(), reason="native CFR not built")
def test_estimate_passes_the_solvers_refusal(client, monkeypatch):
    monkeypatch.setenv("CFR_RAM_BUDGET_MB", "64")  # the smallest budget the env can set
    big = dict(RIVER, effective_stack_bb=50, raise_sizes_pm=[330, 500, 750, 1000, 1500])
    r = client.post("/api/estimate", json={"root": big, "config": {"algorithm": "dcfr_vector"}}).json()
    assert r["ok"] and r["refuse_reason"] and "budget" in r["refuse_reason"]
    assert r["est_mb"] > r["budget_mb"] == 64


@pytest.mark.skipif(not rust_cfr_available(), reason="native CFR not built")
def test_a_full_range_solve_through_the_app(client):
    body = {"root": RIVER, "config": {"algorithm": "dcfr_vector", "max_iterations": 60,
                                       "target_exploitability_bb": 0, "expl_check_secs": 0.5}}
    job = client.post("/api/solve", json=body)
    assert job.status_code == 200, job.text
    job_id = job.json()["job_id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        j = server.session.get_job(job_id, full=False)
        if j["status"] not in ("queued", "running", "paused"):
            break
        time.sleep(0.05)
    if "unknown algorithm" in str(j.get("error")):
        pytest.skip("engine predates dcfr_vector — rebuild it")
    assert j["status"] == "done", j
    assert j["config"]["algorithm"] == "dcfr_vector"
    assert j["config"]["expl_check_secs"] == 0.5
    assert j["expl_kind"] == "exact_infoset"
    assert j["exploitability_bb"] < 0.2


def test_the_viewer_carries_ev_and_equity():
    """(TOOL-035) rows keep the solver's per-hand EV / equity; a 13x13 cell shows
    the class's reach-weighted value (the weights of its action mix)."""
    from plo5bp.cfr_app.strategy_view import build_class_matrix_from_combos, infoset_row, quality_summary
    from plo5bp.gto.preflop_class import cards_to_combo

    board = [0, 5, 10, 15, 20]

    def raw(c0, c1, ev, eq, mass):
        combo = cards_to_combo(c0, c1)
        return {"infoset_id": f"p0_h1_c{combo}", "actor": 0, "path": [], "private_kind": "combo",
                "private_id": combo, "raw_combo": combo, "board": board, "street": 3,
                "actions": ["CHECK_CALL", "RAISE_500"], "probs": [0.25, 0.75],
                "visit_mass": mass, "ev_bb": ev, "equity": eq}

    aces = [raw(48, 49, 4.0, 0.9, 3.0), raw(50, 51, 2.0, 0.7, 1.0)]  # AcAd, AhAs
    rows = [infoset_row(r, root_board_len=5) for r in aces]
    assert rows[0]["ev_bb"] == 4.0 and rows[0]["equity"] == 0.9
    assert infoset_row({**aces[0], "ev_bb": float("nan"), "equity": None}, root_board_len=5)["ev_bb"] is None
    cells = [c for line in build_class_matrix_from_combos(rows)["cells"] for c in line if c]
    assert len(cells) == 1 and cells[0]["label"] == "AA"
    assert cells[0]["ev_bb"] == pytest.approx((3 * 4.0 + 1 * 2.0) / 4)  # reach-weighted
    assert cells[0]["equity"] == pytest.approx((3 * 0.9 + 1 * 0.7) / 4)
    q = quality_summary(rows, {"notes": []})
    assert q["ev_rows"] == 2 and "Per-hand EV" in q["notes"][0]
    assert "No per-hand EV" in quality_summary([{**rows[0], "ev_bb": None}], {"notes": []})["notes"][0]
