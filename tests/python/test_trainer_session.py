"""TrainerSession integration: dealing, hiding, settings, stats, endpoints."""

from __future__ import annotations

import pytest
import torch

from plo5bp.ui.trainer import TrainerSettings

# Keys the study projection returns (server.py:_state_dict). The trainer
# projection must be a superset so the frontend renderers work unchanged.
STUDY_STATE_KEYS = {
    "num_seats", "button_seat", "hero_seat", "actor", "seats", "card_spec",
    "hero_info_complete", "hero_blocking_reason", "modified_cards",
    "pot_chips", "pot_bb", "bet_to_call_chips", "bet_to_call_bb",
    "to_call_chips", "to_call_bb", "street", "history", "legal",
    "raise_bounds", "terminal", "terminal_message", "awaiting_next_street",
    "recommendation", "can_undo", "chip_scale", "starting_stacks_chips",
    "starting_stacks_bb", "simple_ocr_mode",
}


def test_state_shape_superset_of_study(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    s = ts.project_state()
    missing = STUDY_STATE_KEYS - set(s.keys())
    assert not missing, f"trainer state missing study keys: {missing}"
    assert "trainer" in s


def test_hidden_info_pre_terminal(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=5, mc_rollouts=0)
    ts.new_hand()
    s = ts.project_state()
    assert s["recommendation"] is None
    for seat in s["seats"]:
        if seat["is_hero"]:
            assert seat["hole"] is not None and len(seat["hole"]) == 5
        else:
            assert seat["hole"] is None
    # Hero is the actor whenever the state is awaiting input.
    if not ts.hand.terminal:
        assert s["actor"] == s["hero_seat"]


def test_play_to_terminal_reveals_and_reviews(trainer_factory, play_to_terminal):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    play_to_terminal(ts)
    s = ts.project_state()
    assert s["terminal"] in ("fold_out", "showdown")
    assert s["terminal_message"]
    assert all(x["hole"] is not None for x in s["seats"])
    # Chip conservation across seats.
    assert sum(ts.hand.rewards_bb) == pytest.approx(0.0, abs=1e-6)
    if ts.hand.decisions:
        rv = s["trainer"]["review"]
        assert rv is not None and rv["num_decisions"] == len(ts.hand.decisions)
        assert s["trainer"]["feedback"]["decision_idx"] == len(ts.hand.decisions) - 1


def test_settings_fixed_seats_and_kth_position(trainer_factory):
    ts = trainer_factory(
        seats_mode="fixed", seats_fixed=3,
        hero_position_mode="kth", hero_kth=1,
        stacks_mode="fixed", stack_bb=100.0,  # deep so nobody starts all-in
        mc_rollouts=0,
    )
    for _ in range(5):
        ts.new_hand()
        h = ts.hand
        assert h.config.num_seats == 3
        # First to act on the flop is one seat clockwise of the button.
        assert h.hero_seat == (h.button + 1) % 3
        if not h.terminal:
            assert h.last_info.actor == h.hero_seat
            assert len(h.action_log) == 0  # nobody acted before hero


def test_settings_random_seats_and_stack_ranges(trainer_factory):
    ts = trainer_factory(
        seats_mode="random", seats_min=2, seats_max=4,
        stacks_mode="random", stack_min_bb=15.0, stack_max_bb=30.0,
        mc_rollouts=0,
    )
    seen = set()
    for _ in range(15):
        ts.new_hand()
        n = ts.hand.config.num_seats
        assert 2 <= n <= 4
        seen.add(n)
        for st in ts.hand.config.resolved_stacks:
            assert 15.0 * 10_000 <= st <= 30.0 * 10_000
    assert len(seen) > 1, "seat count never varied"


def test_settings_per_seat_stacks(trainer_factory):
    per_seat = [(10.0, 10.0), (20.0, 25.0), (40.0, 40.0),
                (20.0, 20.0), (20.0, 20.0), (20.0, 20.0)]
    ts = trainer_factory(
        seats_mode="fixed", seats_fixed=3,
        stacks_mode="per_seat", stacks_per_seat_bb=per_seat,
        mc_rollouts=0,
    )
    ts.new_hand()
    stacks = ts.hand.config.resolved_stacks
    assert stacks[0] == 100_000
    assert 200_000 <= stacks[1] <= 250_000
    assert stacks[2] == 400_000


def test_stats_accumulate_persist_and_reset(trainer_factory, play_to_terminal):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=0)
    ts.new_hand()
    play_to_terminal(ts)
    moves = len(ts.hand.decisions)
    assert ts.session_stats.hands == 1
    assert ts.session_stats.moves == moves
    assert ts.lifetime_stats.moves == moves
    assert sum(ts.session_stats.cat_counts.values()) == moves

    # Repeat hands are excluded from stats.
    ts.new_hand(repeat=True)
    play_to_terminal(ts)
    assert ts.session_stats.hands == 1
    assert ts.session_stats.moves == moves

    # Lifetime persists into a fresh session at the same path; session resets.
    ts2 = trainer_factory(mc_rollouts=0)
    assert ts2.lifetime_stats.moves == moves
    assert ts2.session_stats.moves == 0

    ts2.lifetime_stats = type(ts2.lifetime_stats)()
    ts2._persist()
    ts3 = trainer_factory(mc_rollouts=0)
    assert ts3.lifetime_stats.moves == 0


def test_settings_persist_with_stats(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=2, mc_rollouts=7)
    ts._persist()
    ts2 = trainer_factory()
    assert ts2.settings.seats_fixed == 2
    assert ts2.settings.mc_rollouts == 7


def test_hero_fold_runs_hand_to_terminal(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    found = False
    for _ in range(10):
        ts.new_hand()
        if ts.hand.terminal:
            continue
        s = ts.project_state()
        if not s["legal"]["fold"]:
            continue
        ts.act("fold", None)
        found = True
        assert ts.hand.terminal, "hand must auto-run to terminal after hero folds"
        raw = ts.hand.env._rs.observation_dict()
        assert bool(raw["folded"][ts.hand.hero_seat])
        break
    assert found, "never reached a foldable hero node"


def test_act_returns_animation_frames(trainer_factory):
    """Each opponent action yields a frame snapshot tagged with
    trainer.anim_action; history grows monotonically across frames and
    hidden info stays hidden pre-terminal."""
    ts = trainer_factory(seats_mode="fixed", seats_fixed=5, mc_rollouts=0)
    deal_frames = ts.new_hand()
    assert len(deal_frames) >= 1
    guard = 0
    saw_opp_frame = False
    while not ts.hand.terminal and guard < 40:
        s = ts.project_state()
        legal = s["legal"]
        if legal["check_call"]:
            frames = ts.act("check_call", None)
        elif legal["fold"]:
            frames = ts.act("fold", None)
        else:
            frames = ts.act("raise", s["raise_bounds"]["min_chips"])
        assert len(frames) >= 1
        # Hero's own frame first (no anim_action), then opponents'.
        assert "anim_action" not in frames[0]["trainer"] \
            or frames[0]["trainer"].get("anim_action") is None
        hist_lens = [len(f["history"]) for f in frames]
        assert hist_lens == sorted(hist_lens)
        for f in frames[1:]:
            a = f["trainer"]["anim_action"]
            saw_opp_frame = True
            assert a["gate"] in ("fold", "check_call", "raise")
            assert a["seat"] != ts.hand.hero_seat
            if f["terminal"] is None:
                for seat in f["seats"]:
                    if not seat["is_hero"]:
                        assert seat["hole"] is None
        guard += 1
    assert ts.hand.terminal
    assert saw_opp_frame, "no opponent frames produced across a whole hand"


def test_hole_cards_display_sorted(trainer_factory, play_to_terminal):
    """Trainer hole cards display rank high->low (descending card index),
    consistently for hero and revealed opponents."""
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    s = ts.project_state()
    hh = s["card_spec"]["hero_hole"]
    assert hh == sorted(hh, reverse=True)
    play_to_terminal(ts)
    s = ts.project_state()
    for seat in s["seats"]:
        assert seat["hole"] == sorted(seat["hole"], reverse=True)
    # The dealt order is preserved separately for bit-exact replays.
    assert sorted(ts.hand.all_holes_dealt[ts.hand.hero_seat], reverse=True) == \
        ts.hand.all_holes[ts.hand.hero_seat]


def test_settings_validation():
    with pytest.raises(Exception):
        TrainerSettings(seats_min=5, seats_max=3)
    with pytest.raises(Exception):
        TrainerSettings(stack_min_bb=50, stack_max_bb=10)
    with pytest.raises(Exception):
        TrainerSettings(stacks_per_seat_bb=[(30.0, 20.0)] * 6)


def test_router_endpoints(trainer_factory, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from plo5bp.network import ActorCritic
    from plo5bp.ui.trainer import create_trainer_router

    monkeypatch.setenv("PLO5BP_TRAINER_STATS", str(tmp_path / "ep_stats.json"))
    torch.manual_seed(0)
    model = ActorCritic(hidden_dim=32, num_layers=1).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    app = FastAPI()
    router = create_trainer_router(model, torch.device("cpu"))
    router.trainer_session.settings = router.trainer_session.settings.model_copy(
        update={"seats_mode": "fixed", "seats_fixed": 3, "mc_rollouts": 0}
    )
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/trainer/state")
    assert r.status_code == 200
    s = r.json()["state"]
    assert s["trainer"]["hand_active"] in (True, False)

    # Review before terminal -> 400.
    if s["trainer"]["hand_active"]:
        assert client.get("/trainer/review?decision=0").status_code == 400

    # Drive to terminal through the endpoint.
    guard = 0
    while s["terminal"] is None and guard < 60:
        legal = s["legal"]
        if legal["check_call"]:
            body = {"gate": "check_call", "chips": None}
        elif legal["fold"]:
            body = {"gate": "fold", "chips": None}
        else:
            body = {"gate": "raise", "chips": s["raise_bounds"]["min_chips"]}
        r = client.post("/trainer/act", json=body)
        assert r.status_code == 200, r.text
        s = r.json()["state"]
        guard += 1
    assert s["terminal"] is not None

    # Acting at terminal -> 400.
    assert client.post(
        "/trainer/act", json={"gate": "check_call", "chips": None}
    ).status_code == 400

    # Bad gate name -> 422 (pydantic pattern).
    assert client.post(
        "/trainer/act", json={"gate": "limp", "chips": None}
    ).status_code == 422

    # Review now works and clamps out-of-range indices.
    if s["trainer"]["review"]:
        assert client.get("/trainer/review?decision=999").status_code == 200

    # New hand resets; raise out of range -> 400 (when raise is legal).
    s = client.post("/trainer/new_hand", json={}).json()["state"]
    if s["legal"]["raise"]:
        bad = s["raise_bounds"]["max_chips"] + 50_000
        assert client.post(
            "/trainer/act", json={"gate": "raise", "chips": bad}
        ).status_code == 400

    # Stats reset endpoints.
    assert client.post(
        "/trainer/stats/reset", json={"scope": "session"}
    ).status_code == 200
    assert client.post(
        "/trainer/stats/reset", json={"scope": "lifetime"}
    ).status_code == 200
