"""NLH format in the study/trainer UI server.

Pins the format-switch surface added 2026-07-03: the /formats registry,
POST /format resets into the right game (preflop, blinds, card-spec
shapes), NLH recommendations ride the 12-anchor ladder (ALL-IN atom
included), street entry walks preflop→flop→turn→river through the
single-board setters, and the PLO5 path is unchanged after switching
back. The NLH model is a random-init placeholder unless
checkpoints/nlh_stub.pt exists — recommendations must still render
(flagged model_loaded=False), because the promote flow fills the real
weights in later.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

import plo5bp.ui.server as srv


@pytest.fixture()
def client():
    c = TestClient(srv.app)
    # Always leave the module-global session back on PLO5 for other tests.
    yield c
    c.post("/format", json={"format": "plo5_double_bomb"})


def test_formats_registry(client):
    r = client.get("/formats")
    assert r.status_code == 200
    body = r.json()
    ids = {f["id"] for f in body["formats"]}
    assert ids == {"plo5_double_bomb", "nlh_single"}
    assert body["active"] in ids


def test_format_switch_resets_to_nlh_preflop(client):
    s = client.post("/format", json={"format": "nlh_single"}).json()["state"]
    assert s["format"] == "nlh_single"
    assert s["street"] == "preflop"
    # 6 × $5 antes + $5 SB + $10 BB at bb=10000 → 45,000 chips ("$45").
    assert s["pot_chips"] == 45_000
    assert s["bet_to_call_chips"] == 10_000
    spec = s["card_spec"]
    assert len(spec["hero_hole"]) == 2
    assert spec["flop_b"] == []
    assert len(spec["turn"]) == 1 and len(spec["river"]) == 1
    # Unknown format is rejected.
    assert client.post("/format", json={"format": "plo7"}).status_code == 422


def test_nlh_recommendation_uses_12_anchor_ladder(client):
    client.post("/format", json={"format": "nlh_single"})
    r = client.post("/cards", json={
        "hero_hole": [51, 47], "flop_a": [None] * 3, "flop_b": [],
        "turn": [None], "river": [None],
    })
    assert r.status_code == 200
    # Walk non-hero actors (check/call) until hero acts.
    s = r.json()["state"]
    for _ in range(12):
        if s["actor"] == s["hero_seat"] or s["terminal"]:
            break
        s = client.post("/action", json={"gate": "check_call"}).json()["state"]
    assert s["actor"] == s["hero_seat"]
    rec = s["recommendation"]
    assert rec is not None
    assert rec["anchor_count"] == 12
    labels = [a["label"] for a in rec["anchors"]]
    assert labels[0] == "min" and labels[-1] == "ALL-IN"
    assert "275%" in labels, "overbet ladder must be exposed"
    allin = rec["anchors"][-1]
    assert allin["frac"] is None
    assert "model_loaded" in rec


def test_nlh_street_walk_to_showdown(client):
    client.post("/format", json={"format": "nlh_single"})
    # Full spec upfront (hero can't act on a street whose cards are
    # missing — the blocking guard covers NLH streets too).
    client.post("/cards", json={
        "hero_hole": [51, 47], "flop_a": [0, 5, 10], "flop_b": [],
        "turn": [15], "river": [20],
    })
    s = client.get("/state").json()["state"]
    for _ in range(40):
        if s["terminal"]:
            break
        s = client.post("/action", json={"gate": "check_call"}).json()["state"]
    assert s["terminal"] == "showdown"
    streets = {h["street"] for h in s["history"]}
    assert "preflop" in streets and "river" in streets
    # Board B never materializes for NLH.
    assert s["card_spec"]["flop_b"] == []


def test_nlh_flop_cards_apply_via_single_board_setters(client):
    client.post("/format", json={"format": "nlh_single"})
    client.post("/cards", json={
        "hero_hole": [51, 47], "flop_a": [None] * 3, "flop_b": [],
        "turn": [None], "river": [None],
    })
    s = client.get("/state").json()["state"]
    # Close preflop: everyone calls/checks.
    for _ in range(12):
        if s["street"] != "preflop" or s["terminal"]:
            break
        s = client.post("/action", json={"gate": "check_call"}).json()["state"]
    assert s["street"] == "flop"
    r = client.post("/cards", json={
        "hero_hole": [51, 47], "flop_a": [0, 5, 10], "flop_b": [],
        "turn": [None], "river": [None],
    })
    assert r.status_code == 200
    s = r.json()["state"]
    assert s["card_spec"]["flop_a"] == [0, 5, 10]
    # Duplicate with the hero hole is rejected.
    r = client.post("/cards", json={
        "hero_hole": [51, 47], "flop_a": [51, 5, 10], "flop_b": [],
        "turn": [None], "river": [None],
    })
    assert r.status_code == 400


def test_switch_back_to_plo5_is_clean(client):
    client.post("/format", json={"format": "nlh_single"})
    s = client.post("/format", json={"format": "plo5_double_bomb"}).json()["state"]
    assert s["format"] == "plo5_double_bomb"
    assert s["street"] == "flop"
    spec = s["card_spec"]
    assert len(spec["hero_hole"]) == 5
    assert len(spec["flop_b"]) == 3
    assert len(spec["turn"]) == 2
    rec_ready_state = client.get("/state").json()["state"]
    assert rec_ready_state["pot_chips"] > 0


def test_trainer_follows_format(client):
    client.post("/format", json={"format": "nlh_single"})
    t = client.post("/trainer/new_hand").json()["state"]
    assert t["format"] == "nlh_single"
    assert t["street"] in ("preflop", "flop", "turn", "river", "showdown")
    assert len(t["card_spec"]["hero_hole"]) == 2
    assert t["card_spec"]["flop_b"] == []
    # Single made-hand label slot (board B is None).
    assert len(t["hero_hand_desc"]) == 2 and t["hero_hand_desc"][1] is None
    # Play the hand out with checks/calls; scoring + terminal must work.
    for _ in range(60):
        if t["terminal"]:
            break
        if t["actor"] == t["hero_seat"]:
            t = client.post(
                "/trainer/act", json={"gate": "check_call"}
            ).json()["state"]
        else:
            break
    # Switching back re-arms PLO5 deals.
    client.post("/format", json={"format": "plo5_double_bomb"})
    t2 = client.post("/trainer/new_hand").json()["state"]
    assert t2["format"] == "plo5_double_bomb"
    assert len(t2["card_spec"]["hero_hole"]) == 5
