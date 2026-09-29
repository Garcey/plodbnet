"""The standing robustness track: rating against a fixed panel (2026-09-28, ML-039).

plo5bp/evaluation/panel.py + scripts/panel_eval.py: a panel's round robin is
played once and cached; a candidate plays every member on the same deals and
gets a rating on the panel's scale (the panel's mean rating = 0, in the point
fit and every bootstrap fit), an interval, and non-transitivity numbers.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from plo5bp import encoding
from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.evaluation import panel as P
from plo5bp.evaluation.league import bootstrap_league
from plo5bp.evaluation.tables import play_duplicate
from plo5bp.network import ActorCriticV5


def test_baseline_policies_play_legal_actions() -> None:
    gm = torch.tensor([[True, True, True], [True, True, False], [False, True, True]])
    sizing = torch.tensor([[20_000, 90_000, 60_000, 0]] * 3, dtype=torch.int64)
    call = P.BaselineActor("call").act(None, gm, sizing)
    assert call.gate.tolist() == [GATE_CHECK_CALL] * 3 and not call.chips.any()
    pot = P.BaselineActor("pot").act(None, gm, sizing)
    assert pot.gate.tolist() == [GATE_RAISE, GATE_CHECK_CALL, GATE_RAISE]
    assert pot.chips.tolist() == [90_000, 0, 90_000]
    torch.manual_seed(0)
    rnd = P.BaselineActor("random").act(None, gm.repeat(200, 1), sizing.repeat(200, 1))
    assert gm.repeat(200, 1)[torch.arange(600), rnd.gate].all()  # always legal
    raised = rnd.gate == GATE_RAISE
    assert ((rnd.chips[raised] >= 20_000) & (rnd.chips[raised] <= 90_000)).all()
    assert set(rnd.gate.tolist()) == {GATE_FOLD, GATE_CHECK_CALL, GATE_RAISE}


def test_the_scale_is_anchored_to_the_panel() -> None:
    rng = np.random.default_rng(0)
    per = {(i, j): [rng.normal(0.3 * (j - i), 1.0, 40) for _ in range(6)]
           for i in range(4) for j in range(i + 1, 4)}
    res = bootstrap_league(4, per, samples=50, anchor=[0, 1, 2])
    assert abs(float(np.mean(res["ratings"][:3]))) < 1e-9
    assert (res["ci_low"] <= res["ratings"]).all() and (res["ratings"] <= res["ci_high"]).all()


def _ckpt(path, seed: int) -> None:
    torch.manual_seed(seed)
    m = ActorCriticV5(hidden_dim=8, obs_dim=OBS_DIM_MINIMAL, num_layers=3, torso_layernorm=True)
    torch.save({
        "model": m.state_dict(), "config": {"obs_mode": "minimal"},
        "obs_rev": int(encoding.OBS_SEMANTICS_REV), "update_counter": seed,
        "variant": "plo5_double_bomb", "head_version": 4,
    }, path)


def test_a_candidate_is_rated_and_the_round_robin_cached(tmp_path) -> None:
    for s in (1, 2, 3, 4):
        _ckpt(tmp_path / f"m{s}.pt", s)
    spec = tmp_path / "panel.json"
    spec.write_text(json.dumps({
        "name": "t", "deals": 8, "configs_per_tier": 1, "modes": ["argmax"],
        "ev_samples": 0,
        "members": [["m1", str(tmp_path / "m1.pt")], ["m2", str(tmp_path / "m2.pt")],
                    ["call", "baseline:call"]],
    }))
    panel = P.Panel.load(spec)
    pairs = []

    def counting_play(cfg, models, *a, **k):
        pairs.append(tuple(type(m).__name__ for m in models))
        return play_duplicate(cfg, models, *a, **k)

    cache = tmp_path / "cache.json"
    rec = P.rate_candidate(panel, str(tmp_path / "m3.pt"), "cpu", cache,
                           bootstrap=20, play=counting_play, log=lambda *_: None)
    r = rec["modes"]["argmax"]
    assert np.isfinite(r["rating"]) and r["ci"][0] <= r["rating"] <= r["ci"][1]
    assert set(r["edges"]) == {"m1", "m2", "call"}
    assert abs(sum(r["panel_ratings"].values())) < 1e-6
    assert np.isfinite(r["candidate_residual_rms"]) and 0.0 <= r["p_nontransitive"] <= 1.0
    assert rec["update"] == 3 and rec["panel_key"] == panel.key() and cache.exists()
    n_first = len(pairs)  # 3 member pairs x 3 configs + 3 candidate pairs x 3
    pairs.clear()
    P.rate_candidate(panel, str(tmp_path / "m4.pt"), "cpu", cache,
                     bootstrap=20, play=counting_play, log=lambda *_: None)
    assert len(pairs) == n_first // 2  # the round robin came from the cache
