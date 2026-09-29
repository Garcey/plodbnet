"""Study spots in one call (site FEAT-016 / FEAT-026 / FEAT-017 / FEAT-020).

`POST /spot` loads a whole spot — table, cards, action log — with
validate-then-commit semantics; `GET /spot` returns the current one in the
same shape (share links round-trip through it); `POST /rewind` keeps a log
prefix and returns what it took off (Redo). Local build, no live capture.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

import plo5bp.ui.server as srv
from plo5bp.config import GameConfig

PLO5 = "plo5_double_bomb"


@pytest.fixture()
def client():
    c = TestClient(srv.app, raise_server_exceptions=False)

    def pristine() -> None:
        s = srv.session
        s.variant = srv.VARIANT_PLO5
        s.game_config = GameConfig(starting_stack=400000)
        s.num_seats = 6
        s.button_seat = 0
        s.hero_seat = 0
        srv._new_session_defaults()
        srv._rebuild_env()

    pristine()
    yield c
    pristine()


def _spot(**over):
    spot = {
        "format": PLO5,
        "num_seats": 4,
        "button_seat": 1,
        "starting_stacks": [300000, 250000, 400000, 200000],
        "ante_chips": 30000,
        "bb_chips": 10000,
        "hero_hole": [51, 47, 43, 39, 35],
        "flop_a": [0, 4, 8],
        "flop_b": [1, 5, 9],
        "turn": [None, None],
        "river": [None, None],
        "actions": [{"gate": "check_call"}, {"gate": "check_call"}],
    }
    spot.update(over)
    return spot


def test_spot_loads_table_cards_and_actions(client):
    r = client.post("/spot", json=_spot())
    assert r.status_code == 200, r.text
    st = r.json()["state"]
    assert st["num_seats"] == 4 and st["button_seat"] == 1 and st["hero_seat"] == 0
    assert st["starting_stacks_chips"] == [300000, 250000, 400000, 200000]
    assert st["chip_scale"]["ante_chips"] == 30000
    assert st["card_spec"]["hero_hole"] == [51, 47, 43, 39, 35]
    assert [h["action"] for h in st["history"]] == ["CheckCall", "CheckCall"]


def test_get_spot_round_trips(client):
    client.post("/spot", json=_spot())
    got = client.get("/spot").json()["spot"]
    assert got["actions"] == [
        {"gate": "check_call", "chips": 0}, {"gate": "check_call", "chips": 0},
    ]
    before = client.get("/state").json()["state"]
    client.post("/reset", json={})
    r = client.post("/spot", json=got)
    assert r.status_code == 200, r.text
    after = r.json()["state"]
    for key in ("num_seats", "button_seat", "starting_stacks_chips", "card_spec", "history", "pot_chips"):
        assert after[key] == before[key], key


def test_bad_spot_leaves_the_session_unchanged(client):
    client.post("/spot", json=_spot())
    before = client.get("/spot").json()["spot"]
    # an illegal action (fold with nothing to call) -> 400, nothing committed
    r = client.post("/spot", json=_spot(num_seats=3, starting_stacks=[1, 2, 3],
                                        actions=[{"gate": "fold"}]))
    assert r.status_code == 400
    assert "isn't legal" in r.json()["detail"]
    assert client.get("/spot").json()["spot"] == before
    # a duplicate card -> 400, nothing committed
    r = client.post("/spot", json=_spot(flop_a=[51, 4, 8]))
    assert r.status_code == 400
    assert client.get("/spot").json()["spot"] == before


def test_spot_for_another_format_is_refused(client):
    r = client.post("/spot", json=_spot(format="nlh_single"))
    assert r.status_code == 409


def test_rewind_returns_what_it_took_off(client):
    client.post("/spot", json=_spot())
    r = client.post("/rewind", json={"length": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["state"]["history"] == []
    assert body["removed"] == [
        {"gate": "check_call", "chips": 0}, {"gate": "check_call", "chips": 0},
    ]
    # nothing to take off: no-op
    assert client.post("/rewind", json={"length": 5}).json()["removed"] == []


def test_spot_routes_are_study_routes():
    from plo5bp.ui import public

    assert {"/spot", "/rewind"} <= public.STUDY_PATHS


def test_the_client_prefix_serves_the_same_study(client):
    """The Study client calls /study/* (BE-009 gates the Study API by that
    prefix); the root paths stay as aliases for a page still running the
    previous script."""
    r = client.post("/study/spot", json=_spot())
    assert r.status_code == 200, r.text
    assert client.get("/study/spot").json() == client.get("/spot").json()
    r = client.post("/study/rewind", json={"length": 1})
    assert r.status_code == 200 and len(r.json()["removed"]) == 1
    assert [h["action"] for h in client.get("/study/state").json()["state"]["history"]] == ["CheckCall"]
    paths = {getattr(rt, "path", None) for rt in srv.app.router.routes}
    for p in ("/study/cards", "/study/seats", "/study/action", "/study/reset",
              "/study/config", "/study/format", "/study/compare"):
        assert p in paths, p
