"""Study and the Trainer offer the covering bet into a short stack's last chips
(2026-10-03; owner, on a Trainer river against one opponent with less than 1bb left: "in a
real poker app, I would be able to bet $20+ and the opponent just calls for their remaining
chips"). The rule itself — `GameConfig.cover_short_bets` — is pinned in
tests/python/engine/test_cover_short_bets.py; here: every table the site deals has it on,
and Study's dock gets the bet (one amount: what they have left)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from starlette.testclient import TestClient

import plo5bp.ui.server as srv
from plo5bp.network import ActorCriticV2
from plo5bp.ui import trainer as T
from plo5bp.ui.common import default_game_config

PLO5 = "plo5_double_bomb"
NLH = "nlh_single"
BB = 10_000


def test_every_site_table_starts_with_it():
    assert default_game_config(PLO5).cover_short_bets and default_game_config(NLH).cover_short_bets
    torch.manual_seed(0)
    model = ActorCriticV2(hidden_dim=16, num_layers=1).eval()
    ts = T.TrainerSession(model, torch.device("cpu"), stats_path=None)
    ts.set_settings(T.TrainerSettings(**{**ts.settings.model_dump(), "mc_rollouts": 0}))
    ts.rng = np.random.default_rng(3)
    for _ in range(5):
        ts.new_hand()
        assert ts.hand.config.cover_short_bets


@pytest.fixture()
def study():
    """A TestClient over a Study session that starts the way a real one does."""
    c = TestClient(srv.app, raise_server_exceptions=False)
    saved = srv.session.game_config
    srv.session.variant = srv.VARIANT_PLO5
    srv.session.game_config = default_game_config(PLO5)
    srv._new_session_defaults()
    srv._rebuild_env()
    yield c
    srv.session.game_config = saved
    srv._new_session_defaults()
    srv._rebuild_env()


def test_study_offers_the_bet_against_a_stack_under_one_big_blind(study):
    ante = srv.session.game_config.ante
    r = study.post("/seats", json={"num_seats": 2, "button_seat": 1, "stacks_are_starting": True,
                                   "starting_stacks": [100 * BB, ante + BB // 2]})
    assert r.status_code == 200, r.text
    r = study.post("/cards", json={"hero_hole": [0, 1, 2, 3, 4], "flop_a": [5, 6, 7], "flop_b": [8, 9, 10],
                                   "turn": [None, None], "river": [None, None]})
    assert r.status_code == 200, r.text
    s = study.get("/state").json()["state"]
    assert s["actor"] == s["hero_seat"] == 0
    assert s["legal"]["raise"] is True  # (it used to be check only)
    assert (s["raise_bounds"]["min_chips"], s["raise_bounds"]["max_chips"]) == (BB // 2, BB // 2)
    r = study.post("/action", json={"gate": "raise", "chips": BB // 2})
    assert r.status_code == 200, r.text
    s = r.json()["state"]
    assert s["actor"] == 1 and s["legal"]["check_call"] is True  # (they call it all in, or fold)
