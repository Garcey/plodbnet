"""scripts/cfr_app.py: closing mid-solve (TOOL-031) and the --host guard (TOOL-058)."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("cfr_app_launcher", REPO / "scripts" / "cfr_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class FakeSession:
    def __init__(self, status="running"):
        self.status = status
        self.stopped = []

    def active_job(self):
        return {"job_id": "j1", "status": self.status, "iterations_run": 1234}

    def stop(self, job_id):
        self.stopped.append(job_id)
        threading.Timer(0.3, lambda: setattr(self, "status", "stopped")).start()

    def get_job(self, job_id, full=False):
        return {"job_id": job_id, "status": self.status}


def test_idle_window_closes_without_asking(launcher):
    asked = []
    assert launcher.close_decision(FakeSession("done"), lambda j: asked.append(j)) is True
    assert asked == []


def test_cancel_keeps_solving_and_close_now_does_not_stop(launcher):
    s = FakeSession()
    assert launcher.close_decision(s, lambda j: launcher.ANSWER_CANCEL) is False
    assert launcher.close_decision(s, lambda j: launcher.ANSWER_CLOSE) is True
    assert s.stopped == []


def test_save_stops_gracefully_then_closes_from_the_background(launcher):
    s = FakeSession()
    closed = threading.Event()
    t0 = time.time()
    assert launcher.close_decision(s, lambda j: launcher.ANSWER_SAVE, on_saved=closed.set) is False
    assert s.stopped == ["j1"]  # graceful stop requested; the window stays open meanwhile
    assert closed.wait(5), "the window must close once the solve has saved"
    assert s.status == "stopped" and time.time() - t0 >= 0.25


@pytest.mark.parametrize("host,ok", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("127.0.0.2", True),
    ("0.0.0.0", False), ("192.168.1.5", False), ("example.com", False),
])
def test_loopback_detection(launcher, host, ok):
    assert launcher._is_loopback(host) is ok


def test_a_network_host_is_refused_without_allow_remote(launcher, capsys):
    assert launcher.main(["--browser", "--host", "0.0.0.0"]) == 2
    assert "--allow-remote" in capsys.readouterr().err


def test_server_guard_in_remote_mode_is_same_origin_only(monkeypatch):
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app import server

    c = TestClient(server.app, base_url="http://192.168.1.5:8766")
    monkeypatch.setenv("CFR_APP_ALLOWED_HOSTS", "")
    assert c.get("/api/health").status_code == 403  # loopback-only by default
    monkeypatch.setenv("CFR_APP_ALLOWED_HOSTS", "*")
    assert c.get("/api/health").status_code == 200
    same = {"origin": "http://192.168.1.5:8766"}
    assert c.post("/api/range/parse", json={"text": "AA"}, headers=same).status_code == 200
    evil = {"origin": "http://evil.example"}
    assert c.post("/api/range/parse", json={"text": "AA"}, headers=evil).status_code == 403
