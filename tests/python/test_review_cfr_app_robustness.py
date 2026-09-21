"""Regression tests for the 2026-09-20 review, CFR desktop app — section E1/E2/E10/E12.

"The app must never die or wedge": roots that crash the native solver are
rejected up front, and a job ALWAYS reaches a terminal state.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from plo5bp.cfr_app.session import (
    SolveSession,
    _root_from_dict,
    root_presets,
    validate_root_for_app,
)
from plo5bp.gto.cfr_api import RootSpec, SolveConfig, SolveReport, rust_cfr_available

RIVER_BOARD = [12, 28, 38, 41, 45]


@pytest.fixture(autouse=True)
def _isolated_cfr_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Never touch the real data/cfr (review J4)."""
    from plo5bp.cfr_app import server

    monkeypatch.setenv("CFR_APP_DATA_DIR", str(tmp_path / "cfr_data"))
    monkeypatch.setenv("CFR_APP_ALLOWED_HOSTS", "testserver")  # TestClient's Host
    monkeypatch.setattr(server, "session", SolveSession())
    yield


def _wait_terminal(sess: SolveSession, job_id: str, secs: float = 30.0) -> dict:
    deadline = time.time() + secs
    while time.time() < deadline:
        j = sess.get_job(job_id, full=False)
        if j and j["status"] not in ("queued", "running", "paused"):
            return j
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached a terminal state")


def _hu_preflop(stack_bb: float, **kw) -> RootSpec:
    return _root_from_dict(
        {"street": 0, "pot_bb": 2.5, "effective_stack_bb": stack_bb, "board": [], "num_seats": 2, **kw}
    )


# --------------------------------------------------------------------------- E1


@pytest.mark.parametrize("stack_bb", [0.5, 1.0, 1.5])
def test_e1_preflop_stack_must_exceed_bb_plus_ante(stack_bb: float):
    # Default stake: bb 10000 + ante 5000 → 1.5bb. RootSpec.validate() accepts
    # all of these; stack=1.0 then killed the process (exit 0xC00000FD).
    root = _hu_preflop(stack_bb)
    root.validate()  # upstream check still passes — that is the bug being guarded
    with pytest.raises(ValueError, match="big blind \\+ ante"):
        validate_root_for_app(root)


def test_e1_preflop_threshold_follows_the_ante():
    validate_root_for_app(_hu_preflop(2.0))
    # No ante → only the 1bb blind must be covered.
    validate_root_for_app(_hu_preflop(1.2, pot_bb=1.5, ante_chips=0))
    with pytest.raises(ValueError, match="big blind \\+ ante"):
        validate_root_for_app(_hu_preflop(1.0, pot_bb=1.5, ante_chips=0))


def test_e1_every_multiway_stack_is_checked():
    root = _root_from_dict(
        {
            "street": 0, "pot_bb": 1.5, "effective_stack_bb": 10.0, "board": [],
            "num_seats": 3, "raise_sizes_pm": [], "allin_atom": True, "ante_chips": 0,
            "stacks_bb": [10.0, 0.9, 10.0],
        }
    )
    with pytest.raises(ValueError, match=r"stacks_bb\[1\]"):
        validate_root_for_app(root)


@pytest.mark.parametrize(
    "field,value",
    [("pot_bb", 0.00004), ("effective_stack_bb", 0.00004)],
)
def test_e1_amounts_that_round_to_zero_chips_are_rejected(field: str, value: float):
    d = {"street": 3, "pot_bb": 10.0, "effective_stack_bb": 50.0, "board": RIVER_BOARD, "num_seats": 2}
    d[field] = value
    root = _root_from_dict(d)
    root.validate()  # > 0, so upstream accepts it; Rust then panicked on 0 chips
    with pytest.raises(ValueError, match="rounds to 0 chips"):
        validate_root_for_app(root)


def test_e1_shipped_presets_still_validate():
    for p in root_presets():
        validate_root_for_app(_root_from_dict(p))


def test_e1_api_validate_and_solve_agree():
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app import server

    client = TestClient(server.app)
    bad = {"street": 0, "pot_bb": 2.5, "effective_stack_bb": 1.0, "board": [], "num_seats": 2}

    v = client.post("/api/validate_root", json=bad)
    assert v.status_code == 200
    assert v.json()["ok"] is False
    assert "big blind + ante" in v.json()["error"]

    s = client.post("/api/solve", json={"root": bad, "config": {"max_iterations": 5}})
    assert s.status_code == 400
    assert "big blind + ante" in s.json()["detail"]
    # Rejected before anything was queued or spawned.
    assert client.get("/api/jobs").json() == {"jobs": [], "active": None}

    # The stack_bb alias is validated too (Validate must check what Solve runs).
    alias = {"street": 0, "pot_bb": 2.5, "stack_bb": 1.0, "board": [], "num_seats": 2}
    assert client.post("/api/validate_root", json=alias).json()["ok"] is False


# --------------------------------------------------------------------------- E2


class _FakePanic(BaseException):
    """Stands in for pyo3_runtime.PanicException — a BaseException, NOT an Exception."""


def test_e2_baseexception_moves_job_to_error_and_frees_the_session(tmp_path: Path):
    calls = {"n": 0}

    def solve_fn(root: RootSpec, config: SolveConfig) -> SolveReport:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FakePanic("called `Result::unwrap()` on an `Err` value")
        return SolveReport(
            status="ok", root=root.as_dict(), config=config.as_dict(),
            strategy={"root_id": root.root_id, "infosets": []},
            iterations_run=1, exploitability_bb=None, notes=[],
        )

    sess = SolveSession(work_dir=tmp_path, solve_fn=solve_fn)
    root = RootSpec.river_hu(RIVER_BOARD, pot_bb=10.0, effective_stack_bb=50.0)

    j = _wait_terminal(sess, sess.start(root, SolveConfig(max_iterations=1), save=False).job_id)
    assert j["status"] == "error"
    assert j["finished_at"] is not None
    assert "_FakePanic" in j["error"]

    # Before the fix the job stayed "running" and this raised RuntimeError (→ 409) forever.
    j2 = _wait_terminal(sess, sess.start(root, SolveConfig(max_iterations=1), save=False).job_id)
    assert j2["status"] == "done"


def test_e2_kuhn_job_also_survives_a_baseexception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from plo5bp.cfr_app import session as session_mod

    def boom(iterations: int = 0):
        raise _FakePanic("kuhn panic")

    monkeypatch.setattr(session_mod, "solve_kuhn", boom)
    sess = SolveSession(work_dir=tmp_path, solve_fn=lambda r, c: None)
    j = _wait_terminal(sess, sess.start_kuhn(iterations=10).job_id)
    assert j["status"] == "error"
    assert j["finished_at"] is not None


# ------------------------------------------------------------- worker / E10 / E12


def test_worker_reports_python_failures_through_the_error_file(tmp_path: Path):
    from plo5bp.cfr_app.solve_worker import run_solve

    result, error = tmp_path / "r.json", tmp_path / "e.json"
    with pytest.raises(SystemExit) as ei:
        # street 7 is invalid → ValueError inside the worker
        run_solve({"street": 7, "pot_bb": 10, "effective_stack_bb": 50}, {}, str(result), str(error))
    assert ei.value.code == 1
    assert not result.exists()
    assert "ValueError" in json.loads(error.read_text(encoding="utf-8"))["worker_error"]


def test_sanitize_json_turns_nan_and_inf_into_null():
    from plo5bp.cfr_app.solve_worker import sanitize_json

    out = sanitize_json({"a": float("nan"), "b": [1.0, float("inf"), {"c": float("-inf")}], "d": "x", "e": 3})
    assert out == {"a": None, "b": [1.0, None, {"c": None}], "d": "x", "e": 3}
    json.dumps(out, allow_nan=False)  # strict JSON — what Starlette's JSONResponse requires


# --- E10: a child that will not stop / dies without a trace -----------------
# Module-level so the spawn child can import them by name (the parent's sys.path,
# which includes this directory under pytest, is handed to the child).


def _child_that_ignores_stop(root_d, config_d, result_path, error_path):  # pragma: no cover - runs in the child
    """Stands in for ONE native iteration that runs for minutes: writes a live
    snapshot, then never looks at the stop file or the clock again."""
    Path(config_d["progress_file"]).write_text(
        json.dumps({"status": "running", "iterations_run": 7, "num_infosets": 1,
                    "strategy": {"infosets": [{"infoset_id": "p0_h4242424242_c5",
                                               "actions": ["CHECK_CALL", "ALLIN"], "probs": [0.5, 0.5]}]}}),
        encoding="utf-8",
    )
    time.sleep(300)


def _child_that_crashes_natively(root_d, config_d, result_path, error_path):  # pragma: no cover
    import os

    os._exit(3)  # no result file, no error file — like a native stack overflow


RIVER_ROOT = {"street": 3, "pot_bb": 10, "effective_stack_bb": 20, "board": RIVER_BOARD,
              "num_seats": 2, "raise_sizes_pm": [1000]}


def _wait_for(predicate, secs: float = 30.0) -> None:
    deadline = time.time() + secs
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition never became true")


def test_e10_stop_kills_a_child_that_never_reaches_an_iteration_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from plo5bp.cfr_app import session as session_mod
    from plo5bp.cfr_app import solve_worker

    monkeypatch.setattr(solve_worker, "run_solve", _child_that_ignores_stop)
    monkeypatch.setattr(session_mod, "STOP_KILL_GRACE_SECS", 0.5)
    sess = SolveSession(work_dir=tmp_path, use_subprocess=True)
    job = sess.start(RIVER_ROOT, {"max_iterations": 0}, save=True)
    _wait_for(lambda: Path(job.progress_file).is_file())  # the child is up and "solving"
    proc = sess._proc

    t0 = time.time()
    sess.stop(job.job_id)
    j = _wait_terminal(sess, job.job_id, secs=20)
    assert time.time() - t0 < 10, "Stop must not wait for the stuck child"
    assert j["status"] == "stopped" and j["finished_at"] is not None
    assert "killed=stop" in j["notes"]
    assert "no final strategy" in j["progress_message"] + " ".join(j["notes"])
    proc.join(5)
    assert not proc.is_alive(), "the child process must actually be gone"

    # The last live snapshot is all the user has left — it stays, and is viewable.
    rep, sig = sess.full_report(job.job_id)
    assert sig[0] == "live" and len(rep["strategy"]["infosets"]) == 1 and rep["status"] == "stopped"
    assert sorted(p.name for p in tmp_path.iterdir()) == [Path(job.progress_file).name]

    # …and the session is free again.
    monkeypatch.setattr(solve_worker, "run_solve", _child_that_crashes_natively)
    j2 = _wait_terminal(sess, sess.start(RIVER_ROOT, {"max_iterations": 1}, save=False).job_id, secs=30)
    assert j2["status"] == "error"


def test_e10_time_budget_overrun_kills_the_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from plo5bp.cfr_app import session as session_mod
    from plo5bp.cfr_app import solve_worker

    monkeypatch.setattr(solve_worker, "run_solve", _child_that_ignores_stop)
    monkeypatch.setattr(session_mod, "BUDGET_KILL_GRACE_SECS", 0.5)
    sess = SolveSession(work_dir=tmp_path, use_subprocess=True)
    job = sess.start(RIVER_ROOT, {"max_iterations": 0, "time_budget_secs": 0.5}, save=False)
    j = _wait_terminal(sess, job.job_id, secs=30)
    assert j["status"] == "stopped" and "killed=time_budget" in j["notes"]


def test_e1_native_crash_of_the_child_is_an_error_not_the_end_of_the_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from plo5bp.cfr_app import solve_worker

    monkeypatch.setattr(solve_worker, "run_solve", _child_that_crashes_natively)
    sess = SolveSession(work_dir=tmp_path, use_subprocess=True)
    j = _wait_terminal(sess, sess.start(RIVER_ROOT, {"max_iterations": 1}, save=True).job_id, secs=30)
    assert j["status"] == "error" and j["finished_at"] is not None
    assert "solver process crashed (exit code 3" in j["error"]
    assert list(tmp_path.iterdir()) == []  # nothing left behind


@pytest.mark.skipif(not rust_cfr_available(), reason="Rust CFR not built")
def test_child_process_solve_round_trips_and_cleans_up(tmp_path: Path):
    """The real solver runs in a spawn child; result comes back by file."""
    sess = SolveSession(work_dir=tmp_path)  # default solve_fn → child process
    assert sess._use_subprocess is True
    job = sess.start(
        {"street": 3, "pot_bb": 10, "effective_stack_bb": 20, "board": RIVER_BOARD,
         "num_seats": 2, "raise_sizes_pm": [1000]},
        {"max_iterations": 5, "thread_num": 1, "target_exploitability_bb": 0, "poll_every": 1},
        save=True,
    )
    j = _wait_terminal(sess, job.job_id, secs=90)
    assert j["status"] == "done", j
    assert j["iterations_run"] == 5
    assert j["num_infosets"] > 0

    out = Path(j["out_path"])
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8"))["iterations_run"] == 5
    # (E12) only the saved report survives: no stop/pause/progress/result/error/tmp files.
    assert sorted(p.name for p in tmp_path.iterdir()) == [out.name]
