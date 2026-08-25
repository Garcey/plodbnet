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
def client(tmp_path):
    # Reset the module-global trainer session to fresh per-format defaults and
    # an isolated stats file. These tests switch formats and one persists NLH
    # settings, so without the reset they read order-dependent stale state and
    # without isolation they overwrite the real checkpoints/trainer_stats.json
    # (which then self-poisons the next run).
    from plo5bp.ui.trainer import _default_settings, VARIANT_NLH, VARIANT_PLO5

    ts = srv.trainer_router.trainer_session
    ts.stats_path = tmp_path / "trainer_stats.json"
    from plo5bp.ui.trainer import FORMAT_EXPERIMENTAL
    ts.settings_by_variant = {
        VARIANT_PLO5: _default_settings(VARIANT_PLO5),
        VARIANT_NLH: _default_settings(VARIANT_NLH),
        FORMAT_EXPERIMENTAL: _default_settings(FORMAT_EXPERIMENTAL),
    }
    ts.variant = VARIANT_PLO5
    ts.settings = ts.settings_by_variant[VARIANT_PLO5]
    c = TestClient(srv.app)
    # Always leave the module-global session back on PLO5 for other tests.
    yield c
    c.post("/format", json={"format": "plo5_double_bomb"})


def test_formats_registry(client):
    r = client.get("/formats")
    assert r.status_code == 200
    body = r.json()
    ids = {f["id"] for f in body["formats"]}
    assert ids == {"plo5_double_bomb", "nlh_single", "experimental"}
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


def test_trainer_settings_are_per_format(client):
    """Each format owns its settings object: NLH edits must never leak
    into PLO5 (a shared object once persisted NLH's 0.5bb ante and
    reloaded it under PLO5 — the 1bb-pot heads-up 'Bet $10' bug)."""
    # PLO5 baseline: factory defaults.
    t = client.get("/trainer/state").json()["state"]
    assert t["trainer"]["settings"]["ante_bb"] == 3.0
    assert t["trainer"]["settings"]["dollars_per_bb"] == 20.0

    # NLH: its own defaults; then customize its fixed stack.
    client.post("/format", json={"format": "nlh_single"})
    t = client.get("/trainer/state").json()["state"]
    nlh_settings = dict(t["trainer"]["settings"])
    assert nlh_settings["ante_bb"] == 0.5
    assert nlh_settings["stack_bb"] == 100.0
    assert nlh_settings["dollars_per_bb"] == 10.0
    nlh_settings["stack_bb"] = 200.0
    r = client.post("/trainer/settings", json=nlh_settings)
    assert r.status_code == 200

    # Back to PLO5: untouched factory defaults, not NLH residue.
    client.post("/format", json={"format": "plo5_double_bomb"})
    t = client.get("/trainer/state").json()["state"]
    s = t["trainer"]["settings"]
    assert s["ante_bb"] == 3.0 and s["stack_bb"] == 20.0
    assert s["dollars_per_bb"] == 20.0

    # And NLH kept the user's edit.
    client.post("/format", json={"format": "nlh_single"})
    t = client.get("/trainer/state").json()["state"]
    assert t["trainer"]["settings"]["stack_bb"] == 200.0


def test_trainer_legacy_settings_file_healed(tmp_path):
    """A v1 stats file (single 'settings' key — possibly polluted with
    the other format's stakes) is discarded on load: both formats
    restart at their own defaults, lifetime stats survive."""
    import json
    import torch

    from plo5bp.network import ActorCriticV2
    from plo5bp.ui.trainer import TrainerSession

    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({
        "version": 1,
        "lifetime": {"hands": 42, "decisions": 100, "sum_score": 9000.0,
                     "ev_loss_bb": 1.5, "cat_counts": {}},
        "settings": {"ante_bb": 0.5, "stack_bb": 100.0,
                     "dollars_per_bb": 10.0},
    }))
    torch.manual_seed(0)
    net = ActorCriticV2(obs_dim=991, hidden_dim=16, num_layers=1)
    ts = TrainerSession(net, torch.device("cpu"), stats_path=stats)
    assert ts.settings.ante_bb == 3.0, "polluted legacy settings discarded"
    assert ts.settings.stack_bb == 20.0
    assert ts.lifetime_stats.hands == 42, "lifetime stats preserved"
    # v2 round-trip: per-format entries persist and reload.
    ts.settings_by_variant["nlh_single"].mc_rollouts  # exists
    ts._persist()
    reloaded = TrainerSession(net, torch.device("cpu"), stats_path=stats)
    assert reloaded.settings_by_variant["nlh_single"].ante_bb == 0.5
    assert reloaded.settings_by_variant["plo5_double_bomb"].ante_bb == 3.0


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
