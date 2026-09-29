"""`GET /health?deploy=1` — what a restart would interrupt, from the app's memory
(the production deploy's "is anyone playing?" check, ops/deploytool.py `status`).

Only a request made ON the server gets it: cloudflared also connects from
127.0.0.1, so a forwarding header or a public Host header means "came through the
tunnel" and the answer is plain /health."""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope="module")
def server(boot_public_server, tmp_path_factory):
    missing = tmp_path_factory.mktemp("nockpt") / "missing.pt"
    return boot_public_server(PLO5BP_CHECKPOINT=str(missing), PLO5BP_BASE_URL="http://127.0.0.1:8770")


def _local(server):
    return TestClient(server.app, base_url="http://127.0.0.1:8770", client=("127.0.0.1", 50123),
                      raise_server_exceptions=False)


def test_the_server_itself_gets_the_deploy_block(server):
    body = _local(server).get("/health?deploy=1").json()
    hg = body["deploy"]["home_games"]
    assert hg["hands_in_progress"] == [] and hg["games_running"] == []
    assert "process_obs_rev" in body and "model_loaded" in body   # still the whole /health


@pytest.mark.parametrize("headers", [
    {"CF-Connecting-IP": "203.0.113.9"}, {"X-Forwarded-For": "203.0.113.9"}, {"Host": "wrapgto.com"},
    {"Forwarded": "for=203.0.113.9"}, {"CF-Ray": "abc"},
])
def test_anything_through_the_tunnel_gets_plain_health(server, headers):
    r = _local(server).get("/health?deploy=1", headers=headers)
    assert "deploy" not in r.json()


def test_a_remote_client_gets_plain_health(server):
    anon = TestClient(server.app, raise_server_exceptions=False)   # client "testclient", Host testserver
    assert "deploy" not in anon.get("/health?deploy=1").json()
    assert "deploy" not in _local(server).get("/health").json()      # only when asked


def _table(gid, *, phase="waiting", running=False, present=0, seated=3, status="open", hand_no=4):
    now = time.monotonic()
    seats = [SimpleNamespace(user_id=i) for i in range(seated)] + [None]
    seen = {i: now for i in range(present)}
    return SimpleNamespace(game_id=gid, status=status, phase=phase, running=running, seats=seats, seen=seen,
                           hand_no=hand_no, runout_active=False, pot_awards=[])


def test_deploy_status_reads_hands_and_running_games_from_memory(server):
    hg = sys.modules["plo5bp.ui.homegame"]
    st = hg.deploy_status([
        _table("hand", phase="in_hand", hand_no=9, present=1),        # a hand in play, whoever is looking
        _table("game", running=True, present=2),                       # dealing to two people at the table
        _table("alone", running=True, present=1),                      # nobody to deal to
        _table("paused", present=3),                                   # between hands, not running
        _table("closed", phase="in_hand", status="closed"),
    ])
    assert st["hands_in_progress"] == [{"id": "hand", "hand_no": 9, "present": 1}]
    assert [t["id"] for t in st["games_running"]] == ["game"]
    assert st["loaded_tables"] == 5
