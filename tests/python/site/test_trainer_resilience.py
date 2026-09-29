"""Trainer robustness and cost guards (2026-09-28):

- a failing Monte-Carlo estimate or opponent step never wedges a hand
  (BE-004 / TEST-015);
- the finished hand's review is built once (PERF-019);
- one network forward per decision node (PERF-016 / TEST-016 — counting
  forwards is stable where wall-clock timing is not);
- a GET of /trainer/state that the app did not make does not deal (BE-013).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from starlette.testclient import TestClient

from plo5bp.network import ActorCriticV2
from plo5bp.ui import trainer as T

CPU = torch.device("cpu")


def _session(tmp_path, *, mc=0, seed=5):
    torch.manual_seed(0)
    model = ActorCriticV2(hidden_dim=16, num_layers=1).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ts = T.TrainerSession(model, CPU, stats_path=tmp_path / "stats.json")
    ts.set_settings(T.TrainerSettings(**{
        **ts.settings.model_dump(), "mc_rollouts": mc, "seats_mode": "fixed", "seats_fixed": 4,
    }))
    ts.rng = np.random.default_rng(seed)
    return ts, model


def _hero_turn(ts, tries=40):
    for _ in range(tries):
        ts.new_hand()
        h = ts.hand
        if not h.terminal and h.last_info.actor == h.hero_seat:
            return h
    pytest.skip("no hero decision dealt")


def _any_legal(h):
    gm = h.last_info.gate_mask
    return "check_call" if gm[T.GATE_CHECK_CALL] else "fold"


def test_a_failing_ev_estimate_never_blocks_the_move(tmp_path, monkeypatch, caplog):
    ts, _ = _session(tmp_path, mc=4)
    h = _hero_turn(ts)
    n_log = len(h.action_log)

    def boom(*a, **k):
        raise RuntimeError("MC exploded")

    monkeypatch.setattr(ts, "_rollout_ev", boom)
    # Deviate from the recommendation so the estimate really runs.
    dist = ts._node_dist(h.last_obs, h.last_info)
    gm = h.last_info.gate_mask
    choice = next(g for g in (T.GATE_FOLD, T.GATE_CHECK_CALL) if gm[g] and g != dist["rec_gate"]) \
        if any(gm[g] and g != dist["rec_gate"] for g in (T.GATE_FOLD, T.GATE_CHECK_CALL)) else None
    slug = T.GATE_SLUGS[choice] if choice is not None else _any_legal(h)
    frames = ts.act(slug, None)
    assert frames, "the move went through"
    d = h.decisions[-1]
    assert d.ev_loss_bb is None and d.ev_loss_se_bb is None  # "EV loss unavailable"
    assert len(h.action_log) > n_log
    assert "EV-loss estimate failed" in caplog.text


def test_an_opponent_failure_is_repaired_on_the_next_state_poll(tmp_path, monkeypatch):
    ts, _ = _session(tmp_path)
    for seed in range(40):
        ts.rng = np.random.default_rng(seed)
        h = _hero_turn(ts)
        real = ts._sample_action
        monkeypatch.setattr(ts, "_sample_action", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        ts.act(_any_legal(h), None)
        monkeypatch.setattr(ts, "_sample_action", real)
        info = h.last_info
        stuck = not h.terminal and info.actor is not None and info.actor != h.hero_seat
        if stuck:
            break
    else:
        pytest.skip("never left a hand waiting on an opponent")
    # /trainer/state (via `resume`) moves it on: back to the hero or finished.
    assert ts.resume() is True
    assert h.terminal or h.last_info.actor == h.hero_seat


def test_the_finished_hand_review_is_built_once(tmp_path, monkeypatch):
    ts, _ = _session(tmp_path)
    for seed in range(60):
        ts.rng = np.random.default_rng(seed)
        ts.new_hand()
        h = ts.hand
        while not h.terminal and h.last_info.actor == h.hero_seat:
            ts.act(_any_legal(h), None)
        if h.terminal and h.decisions:
            break
    else:
        pytest.skip("no finished hand with a hero decision")
    calls = {"n": 0}
    real = ts._node_view

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ts, "_node_view", counting)
    first = ts.project_state()["trainer"]["review"]
    for _ in range(3):
        assert ts.project_state()["trainer"]["review"] == first
    assert calls["n"] <= 1  # built at most once (0 if act's frame already built it)


def test_one_forward_per_decision_node(tmp_path, monkeypatch):
    """(TEST-016) The hero's node is scored with ONE forward (it used to be
    forward + act = two), and each sampled opponent action costs one."""
    ts, model = _session(tmp_path)
    h = _hero_turn(ts)
    calls = {"n": 0}
    real = model.forward

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(model, "forward", counting)
    before = len(h.action_log)
    ts.act(_any_legal(h), None)
    after = h.action_log[before + 1:]
    sampled = sum(1 for a in after if not a.get("auto"))
    finished_review = 1 if h.terminal and h.decisions else 0  # the review's node view
    assert calls["n"] == 1 + sampled + finished_review
    calls["n"] = 0
    T.compute_node_distribution(model, CPU, h.last_obs, h.last_info)
    assert calls["n"] == 1


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("PLO5BP_PUBLIC", raising=False)
    monkeypatch.setattr(T, "_SESSION_RESOLVER", None)
    torch.manual_seed(0)
    model = ActorCriticV2(hidden_dim=16, num_layers=1).eval()
    router = T.create_trainer_router(model, CPU)
    router.trainer_session.stats_path = tmp_path / "s.json"
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), router.trainer_session


def test_a_navigation_or_prefetch_never_deals(client):
    c, ts = client
    assert ts.hand is None
    r = c.get("/trainer/state", headers={"Sec-Fetch-Mode": "navigate"})
    assert r.status_code == 409 and ts.hand is None
    r = c.get("/trainer/state", headers={"Sec-Purpose": "prefetch"})
    assert r.status_code == 409 and ts.hand is None
    # The app's own fetch() deals the first hand, as before.
    r = c.get("/trainer/state", headers={"Sec-Fetch-Mode": "cors"})
    assert r.status_code == 200 and ts.hand is not None
    # …and an explicit POST works too.
    ts.hand = None
    assert c.post("/trainer/state").status_code == 200 and ts.hand is not None


def test_a_busy_session_answers_429_instead_of_parking_a_thread(client, monkeypatch):
    c, ts = client
    monkeypatch.setattr(T, "SESSION_LOCK_TIMEOUT_S", 0.05)
    ts.lock.acquire()
    try:
        r = c.post("/trainer/new_hand")
        assert r.status_code == 429 and r.headers["retry-after"] == "1"
    finally:
        ts.lock.release()
    assert c.post("/trainer/new_hand").status_code == 200


# --- engine access and the build flag (BE-007 add-ons) ---------------------------------


def test_the_trainer_reads_the_engine_only_through_the_env():
    """(HGB-023 for the trainer) payouts and the raw state come from BombPotEnv's
    accessors, never the private engine handle: an env change breaks at import
    and in tests, not in a hand."""
    from pathlib import Path

    assert "._rs" not in Path(T.__file__).read_text(encoding="utf-8")


def test_the_public_payload_follows_the_build_flag_when_it_runs(tmp_path, monkeypatch):
    """The public build's state carries no live-capture key. The flag is read when
    the payload is built (an import-time copy would be the flag of whichever build
    imported the trainer first — the app factory builds several in one process)."""
    ts, _ = _session(tmp_path)
    ts.new_hand()
    monkeypatch.delenv("PLO5BP_PUBLIC", raising=False)
    assert ts.project_state()["simple_ocr_mode"] is False
    monkeypatch.setenv("PLO5BP_PUBLIC", "1")
    assert "simple_ocr_mode" not in ts.project_state()
