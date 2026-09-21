"""Regression tests for the 2026-09-20 review, CFR desktop app — local API surface.

The server binds 127.0.0.1, but the user's browser can reach that too: any open
web page could fire cross-origin POSTs at it, and a DNS-rebinding page could
read every response. Plus: path endpoints read any JSON under the repo.

These tests send REAL loopback Host headers — "testserver" is deliberately NOT
opted in here, so this is the production configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

LOCAL = {"Host": "127.0.0.1:8766"}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app import server
    from plo5bp.cfr_app.session import SolveSession

    monkeypatch.setenv("CFR_APP_DATA_DIR", str(tmp_path / "cfr_data"))  # never the real data/cfr (J4)
    monkeypatch.delenv("CFR_APP_ALLOWED_HOSTS", raising=False)
    monkeypatch.setattr(server, "session", SolveSession(solve_fn=lambda r, c: None))
    monkeypatch.setitem(server._view_cache, "key", None)
    return TestClient(server.app)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.1:8766", "localhost:51234", "LOCALHOST", "[::1]:8766"])
def test_loopback_hosts_are_served(client, host: str):
    assert client.get("/api/health", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize(
    "host", ["evil.example", "evil.example:8766", "127.0.0.1.evil.example", "testserver", "192.168.1.20:8766", ""]
)
def test_dns_rebinding_hosts_are_refused_everywhere(client, host: str):
    for path in ("/", "/api/health", "/api/jobs", "/static/app.js"):
        r = client.get(path, headers={"Host": host})
        assert r.status_code == 403, (host, path)
        assert "Host" in r.json()["detail"]


def test_cross_origin_requests_are_refused(client):
    for origin in ("http://evil.example", "https://127.0.0.1.evil.example", "null"):
        r = client.post("/api/solve/stop", json={}, headers={**LOCAL, "Origin": origin})
        assert r.status_code == 403 and "cross-origin" in r.json()["detail"], origin
    same = client.post("/api/solve/stop", json={}, headers={**LOCAL, "Origin": "http://127.0.0.1:8766"})
    assert same.status_code == 404  # reached the handler: "no active job"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # bodiless POST — what <form> / fetch(no-cors) can send
        {"content": "x=1", "headers": {"Content-Type": "application/x-www-form-urlencoded"}},
        {"content": '{"a":1}', "headers": {"Content-Type": "text/plain"}},  # JSON smuggled as text/plain
    ],
)
def test_simple_cross_site_style_posts_never_reach_a_handler(client, kwargs: dict):
    headers = {**LOCAL, **kwargs.pop("headers", {})}
    for path in ("/api/solve/stop", "/api/solve/pause", "/api/solve/kuhn", "/api/library/load", "/api/solve"):
        r = client.post(path, headers=headers, **kwargs)
        assert r.status_code == 403, path
        assert "application/json" in r.json()["detail"]


def test_json_posts_and_token_posts_are_accepted(client):
    from plo5bp.cfr_app import server

    assert client.post("/api/solve/stop", json={}, headers=LOCAL).status_code == 404  # handler: no job
    with_token = client.post("/api/solve/stop", headers={**LOCAL, "X-CFR-Token": server.API_TOKEN})
    assert with_token.status_code == 404
    wrong = client.post("/api/solve/stop", headers={**LOCAL, "X-CFR-Token": "nope"})
    assert wrong.status_code == 403
    # GETs are read-only and unreadable cross-origin (no CORS headers) → no token needed.
    assert client.get("/api/jobs", headers=LOCAL).status_code == 200


def test_upload_needs_the_token_because_multipart_is_a_simple_content_type(client):
    from plo5bp.cfr_app import server

    chart = {"node_id": "t", "seat_index": 0, "path": "open",
             "hands": [{"hand": "AA", "class_id": 12, "actions": ["FOLD", "ALLIN"], "probs": [0.0, 1.0]}]}
    files = {"file": ("chart.json", json.dumps(chart).encode(), "application/json")}
    assert client.post("/api/upload", files=files, headers=LOCAL).status_code == 403
    ok = client.post("/api/upload", files=files, headers={**LOCAL, "X-CFR-Token": server.API_TOKEN})
    assert ok.status_code == 200, ok.text
    assert ok.json()["kind"] == "chart"


def test_the_page_gets_the_token_and_no_cors_headers_are_sent(client):
    from plo5bp.cfr_app import server

    r = client.get("/", headers={**LOCAL, "Origin": "http://127.0.0.1:8766"})
    assert r.status_code == 200
    assert f"window.CFR_TOKEN = {json.dumps(server.API_TOKEN)};" in r.text
    assert r.text.index("window.CFR_TOKEN") < r.text.index("/static/app.js")  # set before app.js runs
    assert r.headers.get("cache-control") == "no-store"
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
    assert len(server.API_TOKEN) >= 24


def test_path_endpoints_only_read_strategy_directories(client, tmp_path: Path):
    good = tmp_path / "cfr_data" / "app_export" / "ok.json"
    good.parent.mkdir(parents=True)
    good.write_text(json.dumps({"status": "ok", "root": {"street": 3, "board": [0, 5, 10, 15, 20]},
                                "strategy": {"infosets": []}}), encoding="utf-8")
    (good.parent / "notes.txt").write_text("{}", encoding="utf-8")

    assert client.post("/api/library/load", json={"path": str(good)}, headers=LOCAL).status_code == 200
    assert client.get("/api/view", params={"path": str(good)}, headers=LOCAL).status_code == 200

    # Any other JSON in the checkout used to load and come back through /api/jobs.
    for target in ("tests/ocr/fixtures/labels.json", "pyproject.toml", "python/plo5bp/cfr_app/server.py",
                   "../outside.json", "C:/Windows/win.ini", "/etc/passwd"):
        for r in (
            client.post("/api/library/load", json={"path": target}, headers=LOCAL),
            client.get("/api/library/peek", params={"path": target}, headers=LOCAL),
            client.get("/api/view", params={"path": target}, headers=LOCAL),
            client.post("/api/export", json={"path": target}, headers=LOCAL),
            client.post("/api/compare", json={"path_a": target, "path_b": str(good)}, headers=LOCAL),
        ):
            assert r.status_code == 403, (target, r.request.url.path, r.status_code)
    # Inside an allowed directory, still .json only; a missing file is a plain 404.
    assert client.get("/api/library/peek", params={"path": str(good.parent / "notes.txt")}, headers=LOCAL).status_code == 403
    assert client.get("/api/library/peek", params={"path": str(good.parent / "gone.json")}, headers=LOCAL).status_code == 404
    assert client.get("/api/jobs", headers=LOCAL).json()["jobs"][0]["out_path"] == str(good)
