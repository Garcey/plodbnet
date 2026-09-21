"""Regression tests for the 2026-09-20 review, CFR desktop app — E4 (polling cost) + E12.

The live pollers hit /api/jobs and /progress every 800 ms. They must never
parse or copy the strategy; only /view may, outside the lock, once per dump.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from plo5bp.cfr_app import session as session_mod
from plo5bp.cfr_app.session import JobState, SolveSession, light_report, read_progress_counters
from plo5bp.gto.cfr_api import RootSpec, SolveConfig, SolveReport

RIVER_BOARD = [12, 28, 38, 41, 45]


@pytest.fixture(autouse=True)
def _isolated_cfr_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Never touch the real data/cfr (review J4)."""
    from plo5bp.cfr_app import server

    monkeypatch.setenv("CFR_APP_DATA_DIR", str(tmp_path / "cfr_data"))
    monkeypatch.setenv("CFR_APP_ALLOWED_HOSTS", "testserver")  # TestClient's Host
    monkeypatch.setattr(server, "session", SolveSession(solve_fn=lambda r, c: None))
    monkeypatch.setitem(server._view_cache, "key", None)
    monkeypatch.setitem(server._view_cache, "view", None)
    yield


def _infoset(i: int) -> dict:
    return {
        "infoset_id": f"p0_h4242424242_c{i}",
        "actions": ["CHECK_CALL", "ALLIN"],
        "probs": [0.25, 0.75],
    }


def _progress_json(iters: int, n: int, expl: float | None = 1.25) -> str:
    """Same head-first layout SolveConfig::write_progress emits."""
    return json.dumps(
        {
            "status": "running",
            "iterations_run": iters,
            "num_infosets": n,
            "root_id": "t",
            "exploitability_bb": expl,
            "strategy": {"root_id": "t", "schema_version": 2, "infosets": [_infoset(i) for i in range(n)]},
        },
        indent=2,
    )


def _live_job(sess: SolveSession, pf: Path, **kw) -> JobState:
    job = JobState(
        job_id="live", status="running", created_at=time.time(), started_at=time.time(),
        root={"street": 3, "board": RIVER_BOARD}, progress_file=str(pf), **kw,
    )
    sess._jobs["live"] = job
    sess._active_id = "live"
    return job


def _wait_terminal(sess: SolveSession, job_id: str, secs: float = 10.0) -> dict:
    deadline = time.time() + secs
    while time.time() < deadline:
        j = sess.get_job(job_id, full=False)
        if j and j["status"] not in ("queued", "running", "paused"):
            return j
        time.sleep(0.02)
    raise AssertionError("job never finished")


# --------------------------------------------------------------------------- E4


def test_e4_counters_come_from_the_file_head_without_parsing(tmp_path: Path):
    pf = tmp_path / "x.progress.json"
    # A body that is NOT valid JSON proves nothing past the head is parsed.
    pf.write_text(
        '{\n  "status": "running",\n  "iterations_run": 4200,\n  "num_infosets": 150000,\n'
        '  "root_id": "t",\n  "exploitability_bb": 1.2345,\n  "strategy": {\n    "infosets": ['
        + "<<<not json, stands in for 150 MB of infosets>>>" * 200,
        encoding="utf-8",
    )
    assert read_progress_counters(pf) == {
        "iterations_run": 4200, "num_infosets": 150000, "exploitability_bb": 1.2345,
    }


def test_e4_counters_never_come_from_the_strategy_body(tmp_path: Path):
    pf = tmp_path / "x.progress.json"
    pf.write_text('{"strategy": {"iterations_run": 999, "infosets": []}}', encoding="utf-8")
    assert read_progress_counters(pf) is None
    assert read_progress_counters(tmp_path / "missing.json") is None
    # null exploitability (no estimate yet) is reported as absent, not 0
    pf.write_text('{"iterations_run": 1, "num_infosets": 7, "exploitability_bb": null}', encoding="utf-8")
    assert read_progress_counters(pf) == {"iterations_run": 1, "num_infosets": 7}


def test_e4_refresh_progress_does_not_json_parse_or_stash_the_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pf = tmp_path / "live.progress.json"
    pf.write_text(_progress_json(300, 50), encoding="utf-8")
    sess = SolveSession(work_dir=tmp_path, solve_fn=lambda r, c: None)
    job = _live_job(sess, pf)

    def no_parse(*a, **k):
        raise AssertionError("refresh_progress must not json-parse the progress file")

    monkeypatch.setattr(session_mod.json, "loads", no_parse)
    light = sess.refresh_progress("live")
    assert (light["iterations_run"], light["num_infosets"], light["exploitability_bb"]) == (300, 50, 1.25)
    assert job.report is None  # no per-poll stash of the parsed snapshot


def test_e4_light_dicts_never_copy_the_report(tmp_path: Path):
    sess = SolveSession(work_dir=tmp_path, solve_fn=lambda r, c: None)
    # A lock cannot be deep-copied: the old `asdict(self)` raised TypeError here.
    poison = threading.Lock()
    job = JobState(
        job_id="big", status="done", created_at=time.time(),
        report={"status": "ok", "iterations_run": 9, "notes": ["n"],
                "strategy": {"root_id": "r", "infosets": [_infoset(0), {"poison": poison}]}},
    )
    sess._jobs["big"] = job

    light = sess.list_jobs()[0]
    assert light["report"]["strategy"] == {"root_id": "r", "num_infosets": 2, "infosets_omitted": True}
    assert light["report"]["iterations_run"] == 9
    assert sess.get_job("big", full=False)["report"]["strategy"]["infosets_omitted"] is True
    # full=True hands back the report itself (shared, read-only) — still no copy.
    assert sess.get_job("big", full=True)["report"] is job.report


def test_e4_light_report_handles_missing_and_partial_reports():
    assert light_report(None) is None
    assert light_report({"status": "ok"})["strategy"] == {
        "root_id": None, "num_infosets": 0, "infosets_omitted": True,
    }


def test_e4_full_report_parses_outside_the_lock_and_caches_by_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pf = tmp_path / "live.progress.json"
    pf.write_text(_progress_json(100, 3), encoding="utf-8")
    sess = SolveSession(work_dir=tmp_path, solve_fn=lambda r, c: None)
    _live_job(sess, pf)

    parses = {"n": 0, "lock_held": []}
    real_loads = json.loads

    def counting_loads(s, *a, **k):
        parses["n"] += 1
        # RLock: a non-blocking acquire from ANOTHER thread fails iff it is held.
        # Acquire AND release inside the probe thread — an RLock left owned by a
        # dead thread would deadlock the session.
        got: list[bool] = []

        def probe() -> None:
            ok = sess._lock.acquire(blocking=False)
            if ok:
                sess._lock.release()
            got.append(ok)

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        parses["lock_held"].append(not got[0])
        return real_loads(s, *a, **k)

    monkeypatch.setattr(session_mod.json, "loads", counting_loads)

    rep1, sig1 = sess.full_report("live")
    rep2, sig2 = sess.full_report("live")
    assert parses["n"] == 1, "second call must be a cache hit"
    assert parses["lock_held"] == [False], "the big parse must not hold the session lock"
    assert rep1 is rep2 and sig1 == sig2 and sig1[0] == "live"
    assert len(rep1["strategy"]["infosets"]) == 3
    assert rep1["iterations_run"] == 100 and rep1["root"]["board"] == RIVER_BOARD
    assert any(n.startswith("live_snapshot_iter=100") for n in rep1["notes"])

    time.sleep(0.02)
    pf.write_text(_progress_json(200, 5), encoding="utf-8")  # a new dump
    rep3, sig3 = sess.full_report("live")
    assert parses["n"] == 2 and sig3 != sig1
    assert len(rep3["strategy"]["infosets"]) == 5 and rep3["iterations_run"] == 200


def test_e4_counters_only_dump_keeps_serving_the_last_snapshot(tmp_path: Path):
    """Forward-compat with time-based strategy dumps requested from the Rust side."""
    pf = tmp_path / "live.progress.json"
    pf.write_text(_progress_json(100, 3), encoding="utf-8")
    sess = SolveSession(work_dir=tmp_path, solve_fn=lambda r, c: None)
    _live_job(sess, pf)
    first, _ = sess.full_report("live")

    time.sleep(0.02)
    pf.write_text('{"status": "running", "iterations_run": 150, "num_infosets": 3}', encoding="utf-8")
    again, _ = sess.full_report("live")
    assert again is first
    assert sess.refresh_progress("live")["iterations_run"] == 150  # counters still advance


def test_e4_loaded_files_are_not_retained_in_memory(tmp_path: Path):
    f = tmp_path / "saved.json"
    f.write_text(
        json.dumps({"status": "ok", "root": {"street": 3, "board": RIVER_BOARD}, "config": {},
                    "iterations_run": 77, "exploitability_bb": 0.5,
                    "strategy": {"root_id": "r", "infosets": [_infoset(i) for i in range(4)]}}),
        encoding="utf-8",
    )
    sess = SolveSession(work_dir=tmp_path, solve_fn=lambda r, c: None)
    ret = sess.load_report_file(f)
    assert len(ret["report"]["strategy"]["infosets"]) == 4  # return value: still the full report

    kept = sess._jobs[ret["job_id"]]
    assert "infosets" not in kept.report["strategy"]
    assert kept.report["strategy"]["num_infosets"] == 4
    assert (kept.iterations_run, kept.exploitability_bb, kept.num_infosets) == (77, 0.5, 4)
    # …and the viewer still gets everything, lazily, from the file.
    full = sess.get_job(ret["job_id"], full=True)
    assert len(full["report"]["strategy"]["infosets"]) == 4
    assert sess.full_report(ret["job_id"])[1][0] == "final"


def test_e4_view_is_rebuilt_only_when_the_report_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app import server

    pf = tmp_path / "live.progress.json"
    pf.write_text(_progress_json(100, 6), encoding="utf-8")
    _live_job(server.session, pf)

    builds = {"n": 0}
    real = server.load_report

    def counting(*a, **k):
        builds["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(server, "load_report", counting)
    client = TestClient(server.app)
    for params in ({}, {"seat": 0}, {"limit": 2, "offset": 2}):  # live tick, filter, paging
        r = client.get("/api/jobs/live/view", params=params)
        assert r.status_code == 200, r.text
        assert r.json()["job"]["live"] is True
    assert builds["n"] == 1

    time.sleep(0.02)
    pf.write_text(_progress_json(200, 8), encoding="utf-8")
    assert client.get("/api/jobs/live/view").json()["summary"]["num_infosets"] == 8
    assert builds["n"] == 2


# --------------------------------------------------------------------------- E12


def test_e12_refresh_progress_never_overwrites_a_finished_jobs_counters(tmp_path: Path):
    pf = tmp_path / "done.progress.json"
    pf.write_text(_progress_json(100, 40, expl=5.27), encoding="utf-8")  # stale last poll tick
    sess = SolveSession(work_dir=tmp_path, solve_fn=lambda r, c: None)
    job = _live_job(sess, pf, iterations_run=120, exploitability_bb=1.99, num_infosets=55)
    job.status = "done"
    job.finished_at = time.time()
    job.progress_message = "done"

    out = sess.refresh_progress("live")
    assert (out["iterations_run"], out["exploitability_bb"], out["num_infosets"]) == (120, 1.99, 55)
    assert out["status"] == "done" and out["progress_message"] == "done"


def test_e12_progress_file_is_deleted_when_the_job_finishes(tmp_path: Path):
    def solve_fn(root: RootSpec, config: SolveConfig) -> SolveReport:
        Path(config.progress_file).write_text(_progress_json(3, 1), encoding="utf-8")
        return SolveReport(
            status="ok", root=root.as_dict(), config=config.as_dict(),
            strategy={"root_id": root.root_id, "infosets": [_infoset(0)]},
            iterations_run=5, exploitability_bb=0.1, notes=[],
        )

    sess = SolveSession(work_dir=tmp_path, solve_fn=solve_fn)
    root = RootSpec.river_hu(RIVER_BOARD, pot_bb=10.0, effective_stack_bb=50.0)
    job = sess.start(root, SolveConfig(max_iterations=5), save=True)
    done = _wait_terminal(sess, job.job_id)

    assert done["status"] == "done" and done["iterations_run"] == 5
    assert not Path(done["progress_file"]).exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{job.job_id}.json"]
    # A late poll (the UI fires one after "done") changes nothing.
    assert sess.refresh_progress(job.job_id)["iterations_run"] == 5
    # Saved report is what the viewer reads back.
    assert len(sess.get_job(job.job_id, full=True)["report"]["strategy"]["infosets"]) == 1
