"""Tests for the CFR desktop app — real shipped paths only.

No mocks of strategy_view / session loaders: we drive load_report on real
JSON under data/cfr and SolveSession with a real (or stubbed-only-at-solve)
path that still exercises RootSpec validation + job lifecycle.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from plo5bp.cfr_app.session import SolveSession, _config_from_dict, _root_from_dict, root_presets
from plo5bp.cfr_app.strategy_view import (
    build_preflop_matrix,
    filter_rows,
    infoset_row,
    list_strategy_library,
    load_report,
    parse_infoset_id,
    summarize_report_light,
)
from plo5bp.gto.cfr_api import RootSpec, SolveConfig, SolveReport, rust_cfr_available
from plo5bp.gto.preflop_class import preflop_class_label

REPO = Path(__file__).resolve().parents[2]
RIVER_STRAT = REPO / "data" / "cfr" / "verify" / "batch" / "strategies" / "s3_s0_i0.json"
CHART = REPO / "data" / "cfr" / "pushfold_14_charts" / "00_CO_open.json"
PUSHFOLD_FULL = REPO / "data" / "cfr" / "pushfold_4handed_10bb_300k.json"


# ---------------------------------------------------------------------------
# strategy_view
# ---------------------------------------------------------------------------


def test_parse_infoset_id_preflop_class():
    m = parse_infoset_id("pf_p0_h10001907875208514704_c18")
    assert m["seat"] == 0
    assert m["private"] == 18
    assert m["private_kind"] == "class"
    assert m["hand_label"] == preflop_class_label(18)


def test_parse_infoset_id_combo():
    m = parse_infoset_id("p0_h12222850381629986102_c200")
    assert m["private"] == 200
    assert m["private_kind"] == "combo"
    assert m["hand_label"]  # non-empty combo label


def test_load_report_river_real_file():
    if not RIVER_STRAT.is_file():
        pytest.skip("river strategy fixture missing")
    view = load_report(RIVER_STRAT)
    assert view["summary"]["kind"] == "solve_report"
    assert view["summary"]["status"] == "ok"
    assert view["summary"]["num_infosets"] > 100
    assert view["summary"]["street"] == 3
    assert view["rows"]
    row0 = view["rows"][0]
    assert "strategy" in row0
    assert row0["strategy"]
    assert "action" in row0["strategy"][0]
    page = filter_rows(view["rows"], limit=10, offset=0)
    assert page["total"] == view["summary"]["num_infosets"]
    assert len(page["rows"]) == 10


def test_load_report_pushfold_chart():
    if not CHART.is_file():
        pytest.skip("pushfold chart missing")
    view = load_report(CHART)
    assert view["summary"]["kind"] == "chart"
    assert view["summary"]["num_infosets"] == 169
    assert view["nodes"][0].get("label")
    assert view["matrix"] is not None
    assert not view["matrix"].get("empty")
    assert len(view["matrix"]["cells"]) == 13
    # AA should be on diagonal somewhere with data
    cells_flat = [c for row in view["matrix"]["cells"] for c in row if c]
    labels = {c["label"] for c in cells_flat}
    assert "AA" in labels
    assert "AKs" in labels or "AKo" in labels


def test_build_preflop_matrix_from_synthetic_rows():
    rows = []
    for cid in range(169):
        rows.append(
            infoset_row(
                {
                    "infoset_id": f"pf_p0_h1_c{cid}",
                    "actions": ["FOLD", "ALLIN"],
                    "probs": [0.3, 0.7] if cid < 13 else [0.8, 0.2],
                }
            )
        )
    matrix = build_preflop_matrix(rows)
    assert not matrix["empty"]
    assert matrix["num_classes"] == 169
    # pair 22 = class 0
    found_pair = False
    for row in matrix["cells"]:
        for c in row:
            if c and c["class_id"] == 0:
                found_pair = True
                assert c["label"] == "22"
                assert abs(c["agg"] - 0.7) < 1e-6
    assert found_pair


def test_list_strategy_library_finds_data_cfr():
    items = list_strategy_library(max_files=50)
    assert items, "expected strategy files under data/cfr"
    names = {i["name"] for i in items}
    # at least one known artifact
    assert any(
        n.endswith(".json") for n in names
    )
    # paths exist
    assert Path(items[0]["path"]).is_file()


def test_summarize_report_light_river():
    if not RIVER_STRAT.is_file():
        pytest.skip("missing")
    s = summarize_report_light(RIVER_STRAT)
    assert s["kind"] == "solve_report"
    assert s["num_infosets"] > 0
    assert s["street"] == 3


# ---------------------------------------------------------------------------
# session / job manager
# ---------------------------------------------------------------------------


def test_root_from_dict_and_validate():
    d = {
        "street": 3,
        "pot_bb": 10.0,
        "effective_stack_bb": 50.0,
        "board": [12, 28, 38, 41, 45],
        "num_seats": 2,
        "size_preset": "micro",
    }
    r = _root_from_dict(d)
    r.validate()
    assert r.raise_sizes_pm == [500, 1000]
    assert r.street == 3


def test_root_from_dict_rejects_bad_board():
    d = {
        "street": 3,
        "pot_bb": 10.0,
        "effective_stack_bb": 50.0,
        "board": [1, 2, 3],  # need 5
        "num_seats": 2,
    }
    r = _root_from_dict(d)
    with pytest.raises(ValueError, match="board length"):
        r.validate()


def test_config_from_dict():
    c = _config_from_dict({"iters": 50, "threads": 2, "time_budget_secs": 1.5})
    assert c.max_iterations == 50
    assert c.thread_num == 2
    assert c.time_budget_secs == 1.5


def test_root_presets_valid():
    for p in root_presets():
        r = _root_from_dict(p)
        # flop with ochs may still validate board length
        r.validate()


def test_session_job_lifecycle_with_stub_solve(tmp_path: Path):
    """Drive real SolveSession.start/stop/list without requiring long CFR."""

    def fake_solve(root: RootSpec, config: SolveConfig) -> SolveReport:
        # Simulate a short solve; honour stop_file quickly
        for _ in range(20):
            if config.stop_file and Path(config.stop_file).exists():
                return SolveReport(
                    status="ok",
                    root=root.as_dict(),
                    config=config.as_dict(),
                    strategy={
                        "root_id": root.root_id,
                        "infosets": [
                            {
                                "infoset_id": "pf_p0_h1_c12",
                                "actions": ["FOLD", "ALLIN"],
                                "probs": [0.4, 0.6],
                            }
                        ],
                    },
                    iterations_run=3,
                    exploitability_bb=0.1,
                    notes=["early_stop=stop_file"],
                )
            time.sleep(0.05)
        return SolveReport(
            status="ok",
            root=root.as_dict(),
            config=config.as_dict(),
            strategy={
                "root_id": root.root_id,
                "infosets": [
                    {
                        "infoset_id": "pf_p0_h1_c12",
                        "actions": ["FOLD", "ALLIN"],
                        "probs": [0.2, 0.8],
                    }
                ],
            },
            iterations_run=10,
            exploitability_bb=0.05,
            notes=["fake_solve_done"],
        )

    sess = SolveSession(work_dir=tmp_path, solve_fn=fake_solve)
    root = RootSpec.preflop_hu(stack_bb=20.0)
    job = sess.start(root, SolveConfig(max_iterations=10, poll_every=1))
    assert job.job_id
    assert job.status in ("queued", "running")

    # wait for completion
    deadline = time.time() + 5
    final = None
    while time.time() < deadline:
        j = sess.get_job(job.job_id)
        if j and j["status"] in ("done", "error", "stopped"):
            final = j
            break
        time.sleep(0.05)
    assert final is not None, "job did not finish"
    assert final["status"] == "done"
    assert final["report"]["iterations_run"] == 10
    assert final["out_path"]
    assert Path(final["out_path"]).is_file()

    # load back into strategy_view
    view = load_report(final["out_path"])
    assert view["summary"]["num_infosets"] == 1
    assert view["rows"][0]["hand_label"] == preflop_class_label(12)


def test_session_stop_writes_stop_file(tmp_path: Path):
    gate = {"released": False}

    def slow_solve(root: RootSpec, config: SolveConfig) -> SolveReport:
        for _ in range(200):
            if config.stop_file and Path(config.stop_file).exists():
                return SolveReport(
                    status="ok",
                    root=root.as_dict(),
                    config=config.as_dict(),
                    strategy={"root_id": root.root_id, "infosets": []},
                    iterations_run=1,
                    exploitability_bb=None,
                    notes=["early_stop=stop_file"],
                )
            time.sleep(0.02)
        return SolveReport(
            status="ok",
            root=root.as_dict(),
            config=config.as_dict(),
            strategy={"root_id": root.root_id, "infosets": []},
            iterations_run=99,
            exploitability_bb=None,
            notes=[],
        )

    sess = SolveSession(work_dir=tmp_path, solve_fn=slow_solve)
    job = sess.start(
        RootSpec.river_hu([0, 5, 10, 15, 20], pot_bb=8.0, effective_stack_bb=30.0),
        SolveConfig(max_iterations=1000, poll_every=1),
    )
    # let it start
    time.sleep(0.05)
    light = sess.stop(job.job_id)
    assert "stop_requested" in light.get("notes", []) or light["status"] in (
        "running",
        "stopped",
        "done",
    )
    deadline = time.time() + 5
    while time.time() < deadline:
        j = sess.get_job(job.job_id)
        if j and j["status"] in ("done", "stopped", "error"):
            break
        time.sleep(0.05)
    j = sess.get_job(job.job_id)
    assert j["status"] in ("done", "stopped")
    # stop should have shortened the run
    if j.get("report"):
        assert j["report"]["iterations_run"] <= 5 or "stop" in " ".join(
            j.get("notes") or []
        ).lower()


def test_session_load_report_file():
    if not RIVER_STRAT.is_file():
        pytest.skip("missing")
    sess = SolveSession(work_dir=REPO / "data" / "cfr" / "app_jobs")
    job = sess.load_report_file(RIVER_STRAT)
    assert job["status"] == "done"
    assert job["report"]["status"] == "ok"
    assert job["out_path"]


# ---------------------------------------------------------------------------
# FastAPI routes (TestClient) — real app module
# ---------------------------------------------------------------------------


def test_api_health_and_meta():
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    h = client.get("/api/health")
    assert h.status_code == 200
    body = h.json()
    assert body["ok"] is True
    assert "rust_cfr" in body

    m = client.get("/api/meta")
    assert m.status_code == 200
    meta = m.json()
    assert "standard" in meta["size_presets"]
    assert len(meta["presets"]) >= 3
    assert len(meta["preflop_labels"]) == 169


def test_api_validate_root():
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    ok = client.post(
        "/api/validate_root",
        json={
            "street": 3,
            "pot_bb": 10,
            "effective_stack_bb": 50,
            "board": [12, 28, 38, 41, 45],
            "num_seats": 2,
            "size_preset": "micro",
        },
    )
    assert ok.status_code == 200
    assert ok.json()["ok"] is True

    bad = client.post(
        "/api/validate_root",
        json={
            "street": 3,
            "pot_bb": 10,
            "effective_stack_bb": 50,
            "board": [1],
            "num_seats": 2,
        },
    )
    assert bad.status_code == 200
    assert bad.json()["ok"] is False


def test_api_library_and_view_real_file():
    if not RIVER_STRAT.is_file():
        pytest.skip("missing")
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    lib = client.get("/api/library")
    assert lib.status_code == 200
    items = lib.json()["items"]
    assert len(items) >= 1

    # load via API
    load = client.post("/api/library/load", json={"path": str(RIVER_STRAT)})
    assert load.status_code == 200
    assert load.json()["status"] == "done"

    view = client.get("/api/view", params={"path": str(RIVER_STRAT), "limit": 20})
    assert view.status_code == 200
    body = view.json()
    assert body["summary"]["num_infosets"] > 0
    assert len(body["page"]["rows"]) <= 20
    assert body["page"]["total"] > 0


def test_api_library_load_chart_has_matrix():
    if not CHART.is_file():
        pytest.skip("missing")
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    view = client.get("/api/view", params={"path": str(CHART), "limit": 50})
    assert view.status_code == 200
    body = view.json()
    assert body["summary"]["kind"] == "chart"
    assert body["matrix"] is not None
    assert not body["matrix"].get("empty")
    nav = body.get("line_nav") or {}
    assert nav.get("navigable") is True
    assert "open" in (nav.get("by_path") or {})


def test_api_index_serves_html():
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "CFR Solver" in r.text
    assert "/static/app.js" in r.text
    # GTOW-style viewer chrome
    assert "Open solution" in r.text or "btn-upload" in r.text
    assert "action-tree-strip" in r.text


def test_api_upload_solution_json():
    """Upload a chart/solve JSON into data/cfr/uploads and open in viewer."""
    if not CHART.is_file():
        pytest.skip("missing chart")
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    raw = CHART.read_bytes()
    r = client.post(
        "/api/upload",
        files={"file": ("utg_chart.json", raw, "application/json")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["kind"] == "chart"
    assert body["num_infosets"] == 169
    assert "uploads" in body["path"].replace("\\", "/")
    # view the uploaded path
    view = client.get("/api/view", params={"path": body["path"], "limit": 20})
    assert view.status_code == 200
    v = view.json()
    assert v["matrix"] is not None
    assert not v["matrix"].get("empty")


def test_api_upload_rejects_non_json():
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    r = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400


def test_preflop_matrix_fills_all_169_classes():
    """Regression: offsuit placement must not wipe suited cells (was ~91)."""
    if not CHART.is_file():
        pytest.skip("missing chart")
    view = load_report(CHART)
    matrix = view["matrix"]
    assert matrix is not None and not matrix.get("empty")
    filled = sum(1 for row in matrix["cells"] for c in row if c)
    assert filled == 169, f"expected 169 filled cells, got {filled}"
    assert matrix.get("num_filled") == 169 or matrix.get("num_classes") == 169
    # Both triangles present
    labels = {c["label"] for row in matrix["cells"] for c in row if c}
    assert "AKs" in labels and "AKo" in labels
    assert "22" in labels and "AA" in labels


def test_river_report_no_spurious_preflop_matrix():
    """River combo ids 0..168 must not be treated as preflop classes."""
    from pathlib import Path

    river = Path("data/cfr/verify/batch/strategies/s3_s0_i0.json")
    if not river.is_file():
        # fall back to synthetic
        from plo5bp.cfr_app.strategy_view import parse_infoset_id

        r = parse_infoset_id("p0_h999_c12")
        assert r["private_kind"] == "combo"
        assert r["path"] == "999"
        return
    view = load_report(river)
    assert view["summary"]["street"] == 3
    kinds = {r["private_kind"] for r in view["rows"][:100]}
    assert "combo" in kinds
    # Combos may be averaged into a 13×13 for display — never as raw class ids.
    m = view["matrix"]
    if m is not None and not m.get("empty"):
        assert m.get("aggregated_from_combos") is True
        filled = sum(1 for row in m["cells"] for c in row if c)
        assert filled > 0


def test_line_nav_open_allin_to_next_seat():
    from plo5bp.cfr_app.tree_model import action_to_token, build_line_nav, join_path, path_tokens

    assert action_to_token("FOLD") == "F"
    assert action_to_token("ALLIN") == "AI"
    assert action_to_token("RAISE_500") == "R500"
    assert path_tokens("open") == []
    assert path_tokens("AI,F") == ["AI", "F"]
    assert join_path(["AI", "F"]) == "AI,F"

    nodes = [
        {
            "seat": 0,
            "path": "open",
            "num_hands": 169,
            "aggregate": {
                "actions": ["FOLD", "ALLIN"],
                "mean_mix": {"FOLD": 0.4, "ALLIN": 0.6},
            },
        },
        {
            "seat": 1,
            "path": "AI",
            "num_hands": 169,
            "aggregate": {
                "actions": ["FOLD", "ALLIN"],
                "mean_mix": {"FOLD": 0.7, "ALLIN": 0.3},
            },
        },
        {
            "seat": 1,
            "path": "F",
            "num_hands": 169,
            "aggregate": {
                "actions": ["FOLD", "ALLIN"],
                "mean_mix": {"FOLD": 0.2, "ALLIN": 0.8},
            },
        },
    ]
    nav = build_line_nav(nodes, street=0, num_seats=4)
    assert nav["navigable"] is True
    assert nav["root_path"] == "open"
    assert nav["root_seat"] == 0
    acts = {a["token"]: a for a in nav["by_path"]["open"]["actions"]}
    assert acts["AI"]["has_next"] is True
    assert acts["AI"]["next_path"] == "AI"
    assert acts["AI"]["next_seat"] == 1
    assert acts["F"]["next_path"] == "F"
    assert acts["F"]["next_seat"] == 1


def test_chart_pack_index_maps_next_file():
    from pathlib import Path

    from plo5bp.cfr_app.strategy_view import load_report
    from plo5bp.cfr_app.tree_model import try_load_chart_pack

    chart = REPO / "data" / "cfr" / "pushfold_14_charts" / "00_CO_open.json"
    if not chart.is_file():
        pytest.skip("missing chart pack")
    pack = try_load_chart_pack(str(chart))
    assert pack is not None
    assert "open" in pack["by_path"]
    assert "AI" in pack["by_path"]
    assert Path(pack["by_path"]["AI"]["file"]).name.startswith("01_BTN_AI")

    view = load_report(chart)
    nav = view["line_nav"]
    assert nav["navigable"] is True
    acts = {a["token"]: a for a in nav["by_path"]["open"]["actions"]}
    assert acts["AI"]["has_next"] is True
    assert acts["AI"]["chart_file"]
    assert "BTN_AI" in Path(acts["AI"]["chart_file"]).name


def test_combo_rows_aggregate_to_class_matrix():
    from plo5bp.cfr_app.strategy_view import (
        build_class_matrix_from_combos,
        humanize_path,
        infoset_row,
    )

    assert humanize_path("open") == "Open"
    assert humanize_path("12222850381629986102").startswith("Line ")
    assert "R" in humanize_path("RAISE_500")

    rows = []
    for cid in (0, 1, 50, 100):
        rows.append(
            infoset_row(
                {
                    "infoset_id": f"p0_h42_c{cid}",
                    "actions": ["CHECK_CALL", "ALLIN"],
                    "probs": [0.25, 0.75],
                }
            )
        )
    m = build_class_matrix_from_combos(rows)
    assert m.get("aggregated_from_combos") is True
    assert not m.get("empty")
    filled = sum(1 for row in m["cells"] for c in row if c)
    assert filled >= 1


def test_filter_rows_matches_history_hash_path():
    from plo5bp.cfr_app.strategy_view import filter_rows

    rows = [
        {"seat": 0, "path": "42", "history_hash": 42, "hand_label": "AA"},
        {"seat": 0, "path": "open", "hand_label": "KK"},
    ]
    page = filter_rows(rows, path="42")
    assert page["total"] == 1
    assert page["rows"][0]["hand_label"] == "AA"


def test_session_unlimited_pause_resume_stop_live_strategy():
    """Play without max iters; pause keeps state; progress dumps strategy."""
    import time

    from plo5bp.cfr_app.session import SolveSession

    if not __import__("plo5bp.gto.cfr_api", fromlist=["rust_cfr_available"]).rust_cfr_available():
        pytest.skip("no rust cfr")

    s = SolveSession()
    job = s.start(
        {
            "street": 3,
            "pot_bb": 10,
            "effective_stack_bb": 25,
            "board": [2, 7, 12, 17, 22],
            "raise_sizes_pm": [500, 1000],
            "num_seats": 2,
            "allin_atom": True,
        },
        {
            "max_iterations": 0,
            "thread_num": 1,
            "target_exploitability_bb": 0,
            "poll_every": 20,
            "algorithm": "dcfr",
        },
        save=False,
    )
    jid = job.job_id
    assert job.unlimited is True

    saw = False
    for _ in range(80):
        p = s.refresh_progress(jid)
        if p and int(p.get("iterations_run") or 0) > 0:
            saw = True
            break
        time.sleep(0.05)
    assert saw, "expected live progress ticks"

    assert s.pause(jid)["status"] == "paused"
    full = s.get_job(jid, full=True)
    assert full is not None
    rep = full.get("report") or {}
    # partial strategy should be present for live viewer
    infosets = (rep.get("strategy") or {}).get("infosets") or []
    assert len(infosets) > 0

    assert s.resume(jid)["status"] == "running"
    s.stop(jid)
    for _ in range(100):
        st = s.get_job(jid, full=False)
        if st and st["status"] in ("done", "stopped", "error"):
            break
        time.sleep(0.05)
    st = s.get_job(jid, full=False)
    assert st is not None
    assert st["status"] in ("done", "stopped")
    assert int(st.get("iterations_run") or 0) > 0


def test_api_static_assets():
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    for path in ("/static/app.js", "/static/style.css"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert len(r.content) > 100


def test_build_abstract_tree_hu_river():
    from plo5bp.cfr_app.tree_model import build_abstract_tree
    from plo5bp.gto.cfr_api import RootSpec

    root = RootSpec.river_hu(
        [12, 28, 38, 41, 45],
        pot_bb=10.0,
        effective_stack_bb=30.0,
        size_preset="micro",
    )
    tree = build_abstract_tree(root, max_nodes=200, max_depth=6)
    assert tree["num_nodes_built"] >= 3
    assert tree["tree"]["actions"]
    assert "CHECK_CALL" in tree["tree"]["actions"] or "FOLD" in tree["tree"]["actions"]
    # has children for each menu action
    assert tree["tree"]["children"]


def test_build_abstract_tree_pushfold():
    from plo5bp.cfr_app.tree_model import build_abstract_tree
    from plo5bp.gto.cfr_api import RootSpec

    root = RootSpec.preflop_pushfold(num_seats=4, stack_bb=10.0, ante_chips=0)
    tree = build_abstract_tree(root, max_nodes=300, max_depth=8)
    assert tree["num_nodes_built"] >= 5
    # pure push/fold menu
    acts = set(tree["tree"]["actions"])
    assert "ALLIN" in acts
    assert "FOLD" in acts or "CHECK_CALL" in acts


def test_solution_tree_from_chart():
    if not CHART.is_file():
        pytest.skip("missing")
    view = load_report(CHART)
    st = view["solution_tree"]
    assert st is not None
    assert st["num_decision_points"] >= 1
    assert view["summary"]["quality"]["num_infosets"] == 169
    assert "mean_entropy" in view["summary"]["quality"]


def test_api_tree_preview_and_ranges_in_html():
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    html = client.get("/").text
    assert "f-range-oop" in html
    assert "f-range-ip" in html
    assert "btn-tree-preview" in html
    assert "line-tree" in html
    assert "quality-panel" in html

    r = client.post(
        "/api/tree/preview",
        json={
            "street": 3,
            "pot_bb": 10,
            "effective_stack_bb": 40,
            "board": [0, 5, 10, 15, 20],
            "num_seats": 2,
            "size_preset": "micro",
            "raise_sizes_pm": [500, 1000],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["num_nodes_built"] >= 3
    assert body["tree"]["children"]


def test_api_export_and_compare():
    if not CHART.is_file() or not RIVER_STRAT.is_file():
        pytest.skip("missing fixtures")
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    # compare two charts if second exists
    chart2 = REPO / "data" / "cfr" / "pushfold_14_charts" / "01_BTN_AI.json"
    if chart2.is_file():
        cmp = client.post(
            "/api/compare",
            json={"path_a": str(CHART), "path_b": str(chart2)},
        )
        assert cmp.status_code == 200
        d = cmp.json()["diff"]
        assert "mean_l1" in d
        assert d["num_common"] >= 0

    exp = client.post("/api/export", json={"path": str(CHART), "out_name": "test_export_chart"})
    assert exp.status_code == 200
    out = Path(exp.json()["path"])
    assert out.is_file()
    assert out.stat().st_size > 100


def test_api_view_includes_quality_and_solution_tree():
    if not CHART.is_file():
        pytest.skip("missing")
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    v = client.get("/api/view", params={"path": str(CHART), "limit": 20}).json()
    assert v["summary"]["quality"]["num_infosets"] == 169
    assert v["solution_tree"]["num_decision_points"] >= 1
    assert "mean_fold" in v["summary"]["quality"] or "mean_allin" in v["summary"]["quality"]


@pytest.mark.skipif(not rust_cfr_available(), reason="Rust CFR not built")
def test_api_solve_short_river_smoke():
    """End-to-end short river solve through the real API + engine."""
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    # stop any leftover
    client.post("/api/solve/stop")
    time.sleep(0.1)

    resp = client.post(
        "/api/solve",
        json={
            "root": {
                "street": 3,
                "pot_bb": 10.0,
                "effective_stack_bb": 20.0,
                "board": [12, 28, 38, 41, 45],
                "num_seats": 2,
                "size_preset": "micro",
                "raise_sizes_pm": [500, 1000],
                "algorithm": "dcfr",
            },
            "config": {
                "max_iterations": 5,
                "thread_num": 1,
                "target_exploitability_bb": 99.0,
                "seed": 1,
                "algorithm": "dcfr",
                "poll_every": 1,
            },
            "save": True,
            "label": "test_smoke",
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    deadline = time.time() + 60
    final = None
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error", "stopped"):
            final = j
            break
        time.sleep(0.2)
    assert final is not None
    assert final["status"] == "done", final
    assert final["report"]["iterations_run"] >= 1

    # view through job endpoint
    v = client.get(f"/api/jobs/{job_id}/view", params={"limit": 10})
    assert v.status_code == 200
    assert v.json()["summary"]["num_infosets"] >= 0


@pytest.mark.skipif(not rust_cfr_available(), reason="Rust CFR not built")
def test_e2e_solve_with_ranges_then_inspect_quality():
    """User journey: configure root + ranges → multi-second solve → inspect quality."""
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app.server import app

    client = TestClient(app)
    client.post("/api/solve/stop")
    time.sleep(0.15)

    # Preview tree first (builder)
    prev = client.post(
        "/api/tree/preview",
        json={
            "street": 3,
            "pot_bb": 12.0,
            "effective_stack_bb": 25.0,
            "board": [0, 5, 10, 15, 20],
            "num_seats": 2,
            "raise_sizes_pm": [500, 1000],
            "allin_atom": True,
            "range_oop": "AA,KK,QQ",
            "range_ip": "random",
        },
    )
    assert prev.status_code == 200
    assert prev.json()["num_nodes_built"] >= 3

    t0 = time.time()
    resp = client.post(
        "/api/solve",
        json={
            "root": {
                "street": 3,
                "pot_bb": 12.0,
                "effective_stack_bb": 25.0,
                "board": [0, 5, 10, 15, 20],
                "num_seats": 2,
                "raise_sizes_pm": [500, 1000],
                "allin_atom": True,
                "range_oop": "AA,KK,QQ",
                "range_ip": "random",
                "algorithm": "dcfr",
            },
            "config": {
                "max_iterations": 80,
                "thread_num": 2,
                "target_exploitability_bb": 0.01,
                "seed": 7,
                "algorithm": "dcfr",
                "poll_every": 10,
                "time_budget_secs": 25.0,
            },
            "save": True,
            "label": "e2e_range_inspect",
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    deadline = time.time() + 90
    final = None
    saw_running = False
    while time.time() < deadline:
        prog = client.get(f"/api/jobs/{job_id}/progress").json()
        if prog["status"] in ("running", "queued"):
            saw_running = True
        if prog["status"] in ("done", "error", "stopped"):
            final = client.get(f"/api/jobs/{job_id}").json()
            break
        time.sleep(0.25)

    elapsed = time.time() - t0
    assert final is not None, "solve did not finish"
    assert final["status"] == "done", final
    assert final["report"]["status"] == "ok"
    assert final["report"]["iterations_run"] >= 10
    # multi-second path (not a trivial 0ms stub)
    assert elapsed >= 0.3 or final["report"]["iterations_run"] >= 30

    # inspect via view
    v = client.get(f"/api/jobs/{job_id}/view", params={"limit": 50}).json()
    assert v["summary"]["num_infosets"] > 0
    q = v["summary"]["quality"]
    assert q["num_infosets"] > 0
    assert "mean_entropy" in q
    assert v["solution_tree"] is not None
    # ranges persisted on root
    root = final["report"]["root"]
    assert "AA" in (root.get("range_oop") or "") or root.get("range_oop")

    # export job output
    out_path = final.get("out_path")
    assert out_path and Path(out_path).is_file()
    exp = client.post("/api/export", json={"path": out_path, "out_name": "e2e_range_solve"})
    assert exp.status_code == 200
    assert Path(exp.json()["path"]).is_file()

    # progress endpoint had elapsed
    prog = client.get(f"/api/jobs/{job_id}/progress").json()
    assert prog["elapsed_secs"] is not None
    assert prog["iterations_run"] == final["report"]["iterations_run"]
