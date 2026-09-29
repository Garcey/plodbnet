"""The shared evaluation library (plo5bp.evaluation, 2026-09-28).

ML-017 / ML-020: actors rebuilt from the checkpoint itself, obs-rev refused.
ML-021: round summaries keyed by opponent. ML-052: league ratings with
intervals and a non-transitivity test. ML-053: sharpness caches know what they
hold. ML-056: derived checkpoints drop the source run's bookkeeping.
ML-034: env.sizing() and the duplicate-deal match.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp import encoding as _encoding
from plo5bp.config import GameConfig
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.env_batched import BatchedBombPotEnv
from plo5bp.evaluation import ObsRevMismatch, checkpoint_meta, load_actor
from plo5bp.evaluation.league import bootstrap_league, fit_ratings
from plo5bp.evaluation.rounds import latest_results, round_means
from plo5bp.evaluation.sharpness import (
    StatesMismatch,
    check_readable,
    load_states,
    save_states,
)
from plo5bp.evaluation.tables import play_duplicate, summarize
from plo5bp.network import ActorCriticV5, build_actor_from_state_dict
from plo5bp.train.checkpoint import derived_checkpoint

REV = int(_encoding.OBS_SEMANTICS_REV)


def _ckpt(obs_dim=OBS_DIM_MINIMAL, hidden=16, layers=3, rev=REV, ema=False, **extra):
    torch.manual_seed(0)
    m = ActorCriticV5(hidden_dim=hidden, num_layers=layers, obs_dim=obs_dim, torso_layernorm=True)
    ck = {"model": m.state_dict(), "config": {"obs_mode": "minimal"}, "head_version": 4,
          "variant": "plo5_double_bomb", "update_counter": 12, "obs_rev": rev, **extra}
    if ema:
        ck["model_ema"] = {k: v.clone() for k, v in m.state_dict().items()}
    return ck


def test_the_actor_size_comes_from_the_checkpoint():
    ck = _ckpt(hidden=24, layers=4)
    ck["config"] = {}  # no size in the config: it used to default to 128x2
    actor, meta = load_actor(ck)
    assert (meta["hidden_dim"], meta["num_layers"]) == (24, 4)
    assert meta["obs_mode"] == "minimal" and meta["update"] == 12
    with pytest.raises(ValueError, match="hidden_dim=128"):
        build_actor_from_state_dict(ck["model"], 128, 2)
    assert not any(p.requires_grad for p in actor.parameters())


def test_ema_actor_when_present_else_the_last_iterate(capsys):
    _a, meta = load_actor(_ckpt(ema=True), ema=True)
    assert meta["ema"] is True
    _a, meta = load_actor(_ckpt(), ema=True)
    assert meta["ema"] is False and "no EMA actor" in capsys.readouterr().out


def test_an_obs_revision_mismatch_is_refused():
    other = 1 if REV == 2 else 2
    with pytest.raises(ObsRevMismatch, match=f"PLO5BP_OBS_REV={other}"):
        load_actor(_ckpt(rev=other))
    unstamped = _ckpt()
    del unstamped["obs_rev"]
    assert checkpoint_meta(unstamped)["obs_rev"] == 1  # unstamped = trained on rev 1
    load_actor(_ckpt(rev=other), check_rev=False)  # h2h_cross serves each its own rev


def _row(a, b, edge, greedy=False):
    return {"a": {"path": f"checkpoints/{a}.pt"}, "b": {"path": f"checkpoints/{b}.pt"},
            "edge_bb": edge, "greedy_a": greedy, "greedy_b": greedy,
            "tiers": {"deep": {"edge_bb": edge}}}


def test_round_summary_keys_by_opponent_and_refuses_a_mix():
    rows = [_row("r5a_1395", "vSix6_1390", 0.10), _row("r5a_1395", "vSix5_1248", 0.50),
            _row("r5a_1397", "vSix6_1390", 0.20)]
    latest = latest_results(rows)
    assert len(latest) == 3  # the same checkpoint vs two references: two keys
    with pytest.raises(ValueError, match="different references"):
        round_means(latest, [], 0)
    acc = round_means(latest_results(rows, ref="vSix6_1390"), [], 0)
    assert np.allclose(acc[("r5a", "sampled")]["ALL"], [0.10, 0.20])


def test_league_recovers_a_transitive_table_and_flags_a_cycle():
    truth = np.array([0.3, 0.0, -0.3])
    rng = np.random.default_rng(0)
    per = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        per[(i, j)] = [truth[i] - truth[j] + rng.normal(0, 0.5, 200) for _ in range(12)]
    res = bootstrap_league(3, per, samples=100, seed=1)
    assert np.allclose(res["ratings"], truth, atol=0.03)
    # intervals bracket the estimate with the width the noise implies
    # (each rating's sd here ~0.005: 12 configs x 200 pairs at sd 0.5)
    width = res["ci_high"] - res["ci_low"]
    assert np.all(res["ci_low"] < res["ratings"]) and np.all(res["ratings"] < res["ci_high"])
    assert np.all((width > 0.004) & (width < 0.06)), width
    assert res["p_nontransitive"] > 0.05
    # rock-paper-scissors: 0 > 1 > 2 > 0 by 0.5
    cyc = {(0, 1): [0.5 + rng.normal(0, 0.1, 200) for _ in range(12)],
           (1, 2): [0.5 + rng.normal(0, 0.1, 200) for _ in range(12)],
           (0, 2): [-0.5 + rng.normal(0, 0.1, 200) for _ in range(12)]}
    res = bootstrap_league(3, cyc, samples=100, seed=1)
    assert res["p_nontransitive"] < 0.05
    r, resid = fit_ratings(3, {(0, 1): (0.3, 0.1), (1, 2): (0.3, 0.1), (0, 2): (0.6, 0.1)})
    assert np.allclose(r, [0.3, 0.0, -0.3], atol=1e-4) and max(map(abs, resid.values())) < 1e-4


def test_summary_reports_a_config_level_error():
    rng = np.random.default_rng(0)
    # 10 configs whose true edges differ a lot, many pairs each: the pair-level
    # se is tiny, the config-level one is not
    parts = [rng.normal(mu, 0.1, 500) for mu in rng.normal(0, 1.0, 10)]
    rep = summarize({"deep": parts})
    assert rep["se_config"] > 5 * rep["se"]
    assert rep["tiers"]["deep"]["configs"] == 10


def test_sharpness_caches_know_what_they_hold(tmp_path):
    states = (np.zeros((4, OBS_DIM_MINIMAL), np.float32), np.ones((4, 3), bool),
              np.zeros((4, 4), np.int64), np.zeros(4, np.int8))
    cache = tmp_path / "s.npz"
    save_states(cache, states, states_from="x.pt", obs_mode="minimal")
    _, info = load_states(cache)
    assert info["obs_rev"] == REV and info["obs_dim"] == OBS_DIM_MINIMAL
    actor, meta = load_actor(_ckpt())
    check_readable(info, meta, actor)
    other = 1 if REV == 2 else 2
    with pytest.raises(StatesMismatch, match="obs rev"):
        check_readable(info, {**meta, "obs_rev": other}, actor)
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, obs=states[0], masks=states[1], sizing=states[2], tiers=states[3],
             states_from="x.pt")
    with pytest.raises(StatesMismatch, match="--cache-rev"):
        load_states(legacy)
    assert load_states(legacy, assume_rev=REV)[1]["obs_rev"] == REV


def test_derived_checkpoints_drop_the_source_runs_bookkeeping():
    src = _ckpt(pool_member_updates=[5, 10], anneal_control_applied='{"lr": 1e-4}',
                anneal_tier_ent={"deep": 0.1}, ema=True)
    out = derived_checkpoint(src, "average", ["a.pt", "b.pt"], model=src["model"])
    for gone in ("pool_member_updates", "anneal_control_applied", "anneal_tier_ent"):
        assert gone not in out
    assert out["model_ema"] is None and out["update_counter"] == 12
    assert out["derived_from"] == {"how": "average", "sources": ["a.pt", "b.pt"],
                                   "source_update_counter": 12}
    load_actor(out)


def test_env_sizing_and_a_duplicate_match():
    cfg = GameConfig(num_seats=3, starting_stack=40 * 10_000, ante=30_000, bb=10_000)
    env = BatchedBombPotEnv(16, cfg, obs_mode="minimal", opp_outcome_mc=0)
    env.reset_batch(np.arange(16, dtype=np.uint64), (np.arange(16) % 3).astype(np.uint8))
    rows = np.arange(16)
    actor = np.where(env._actors >= 0, env._actors, 0).astype(np.intp)
    want_to_call = np.maximum(env._bet_to_call.astype(np.int64)
                              - env._street_commit[rows, actor].astype(np.int64), 0)
    sz = env.sizing()
    assert sz.shape == (16, 4) and np.array_equal(sz[:, 3], want_to_call)
    assert np.array_equal(sz[:, 0], env._min_raise.astype(np.int64))
    a, _ = load_actor(_ckpt())
    runs = []
    for _ in range(2):
        torch.manual_seed(3)
        net, seats, steps = play_duplicate(cfg, (a, a), 8, torch.device("cpu"),
                                           np.random.default_rng(4), ev_samples=8,
                                           obs_mode="minimal")
        runs.append(net)
        assert seats == 3 and steps > 0 and net.size <= 8
    assert np.array_equal(runs[0], runs[1])  # a seed reproduces the match
