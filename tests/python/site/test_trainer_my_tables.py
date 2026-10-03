"""The Trainer's "My tables" (2026-10-03; owner: "make it so that the clubgg conditions mode
automatically updates the seat and stack size distributions based on the individual user's
hand histories … the distribution should have typical stack sizes for opponents and for
themselves" — represented, not copied: "I would prefer that you represent over pulling
exact configurations").

`TrainerSettings.tables == "mine"` deals from the profile Hand review keeps of the player's
latest hands (`handreview.table_profile` / `draw_table`: players, ante, the player's OWN
stack and the opponents' stacks, each its own distribution — recurring amounts as point
masses, every other stack between quantiles) and, without one (no subscription, too few
hands, the local build), typical ClubGG tables (the training tier `clubgg_real`). The
store's half (uploads, the cache, the route) is in `test_public_hand_review.py`. Every
hand here is synthetic."""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pytest
import torch

from plo5bp.config import VARIANT_NLH
from plo5bp.network import ActorCriticV2
from plo5bp.ui import handreview as hr
from plo5bp.ui import trainer as T

CPU = torch.device("cpu")
BB = T.BB_CHIPS
SEAT_SHARES = {6: 0.45, 5: 0.30, 4: 0.20, 3: 0.05}


def _hands(seed=1, n=600):
    """Synthetic hands at $10/$20 (a 2000-cent bb, 3bb ante). YOUR stack: $400 (auto
    top-up) 55% of the time, $385 3% (short by less than a bb: no top-up), else what you
    won (log-uniform $410-$6,000). The opponents': a $400 buy-in 15%, else log-uniform
    $30-$6,000."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = int(rng.choice(list(SEAT_SHARES), p=list(SEAT_SHARES.values())))
        r = rng.random()
        hero = 40000 if r < 0.55 else 38500 if r < 0.58 else int(np.exp(rng.uniform(np.log(41000), np.log(600000))))
        players = [(hero, True)] + [
            (40000 if rng.random() < 0.15 else int(np.exp(rng.uniform(np.log(3000), np.log(600000)))), False)
            for _ in range(k - 1)
        ]
        out.append((2000, 6000, players))
    return out


def _draws(prof, n, seed=5):
    rng = np.random.default_rng(seed)
    return [hr.draw_table(prof, rng) for _ in range(n)]


def test_a_profile_is_a_picture_of_your_tables_not_a_copy_of_them():
    hands = _hands()
    prof = hr.table_profile(hands)
    assert prof["hands"] == 600 and prof["antes"] == [[3.0, 1.0]]
    assert abs(sum(p for _n, p in prof["seats"]) - 1) < 1e-3
    hero_atoms = dict(map(tuple, prof["hero"]["atoms"]))
    assert set(hero_atoms) == {19.25, 20.0} and abs(hero_atoms[20.0] - 0.55) < 0.05
    assert 20.0 in dict(map(tuple, prof["opponents"]["atoms"]))
    for side in ("hero", "opponents"):
        k = prof[side]["knots"]
        assert len(k) == hr.PROFILE_KNOTS and all(a <= b for a, b in zip(k, k[1:]))

    tables = _draws(prof, 20000)
    seats = collections.Counter(n for n, *_ in tables)
    for n, share in SEAT_SHARES.items():
        assert abs(seats[n] / len(tables) - share) < 0.025
    assert all(len(others) == n - 1 and ante == 3.0 for n, _h, others, ante in tables)
    hero = np.array([h for _n, h, _o, _a in tables])
    opp = np.array([x for *_x, others, _a in tables for x in others])
    real_hero = np.array([s / 2000 for _b, _a, ps in hands for s, me in ps if me])
    real_opp = np.array([s / 2000 for _b, _a, ps in hands for s, me in ps if not me])
    # YOUR stack: the top-up amount as often as in your hands, the raked $385 now and then,
    # never anything shorter than you ever started with
    assert abs(np.mean(hero == 20.0) - np.mean(real_hero == 20.0)) < 0.02
    assert abs(np.mean(hero == 19.25) - np.mean(real_hero == 19.25)) < 0.01
    assert hero.min() == 19.25 and not np.any((hero > 19.25) & (hero < 20.0))
    assert hero.max() <= real_hero.max() + 1e-9
    # the opponents: their own picture — short stacks included
    assert opp.min() >= real_opp.min() - 1e-9 and opp.max() <= real_opp.max() + 1e-9
    assert abs(np.mean(opp < 20) - np.mean(real_opp < 20)) < 0.02
    for q in (10, 25, 50, 75, 90):
        assert abs(np.percentile(opp, q) / np.percentile(real_opp, q) - 1) < 0.08
        assert abs(np.percentile(hero, q) / np.percentile(real_hero, q) - 1) < 0.08
    # represented, not pulled: hardly any drawn table is one of the real ones
    real_tables = {tuple(sorted(round(s / 2000, 2) for s, _me in ps)) for _b, _a, ps in hands}
    same = sum(tuple(sorted([round(h, 2), *[round(x, 2) for x in o]])) in real_tables for _n, h, o, _a in tables)
    assert same / len(tables) < 0.01


def test_small_and_odd_profiles():
    assert hr.table_profile([])["hands"] == 0
    # one buy-in everywhere: every stack is that amount, every time
    flat = hr.table_profile([(2000, 6000, [(40000, True), (40000, False), (40000, False)])] * 10)
    assert flat["hero"]["knots"] == [] and flat["hero"]["atoms"] == [[20.0, 1.0]]
    assert {h for _n, h, _o, _a in _draws(flat, 50)} == {20.0}
    # a rare ante is left out; the most common one always stays
    mixed = [(2000, 6000, [(40000, True), (50000, False)])] * 97 + [(2000, 4000, [(40000, True), (50000, False)])] * 3
    assert hr.table_profile(mixed)["antes"] == [[3.0, 1.0]]
    # tables the Trainer can't seat (7+ players) count for nothing
    big = [(2000, 6000, [(40000, True)] + [(40000, False)] * 6)]
    assert hr.table_profile(big)["hands"] == 0


def _session(tmp_path, tables="mine", seed=11):
    torch.manual_seed(0)
    model = ActorCriticV2(hidden_dim=16, num_layers=1).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ts = T.TrainerSession(model, CPU, stats_path=tmp_path / "stats.json")
    ts.set_settings(T.TrainerSettings(**{**ts.settings.model_dump(), "mc_rollouts": 0, "tables": tables}))
    ts.rng = np.random.default_rng(seed)
    return ts


def _tables(ts, n):
    out = []
    for _ in range(n):
        ts.new_hand()
        h = ts.hand
        out.append((h.config, h.hero_seat))
    return out


def test_my_tables_deal_from_your_profile(tmp_path, monkeypatch):
    prof = hr.table_profile(_hands())
    monkeypatch.setattr(T, "_MY_TABLES", lambda: {"paid": True, "min_hands": 50, **prof})
    view = T.my_tables_view()
    assert view["source"] == "mine" and view["hands"] == 600 and view["review"] and view["paid"]
    assert view["antes"] == [[3.0, 1.0]] and view["hero"]["median"] == prof["hero"]["median"]
    assert "knots" not in view["hero"]  # (the dialog gets the numbers to show, not the profile)

    deals = _tables(_session(tmp_path), 300)
    assert {cfg.num_seats for cfg, _ in deals} <= set(SEAT_SHARES)
    assert all(cfg.ante == 3 * BB and cfg.sb == 0 for cfg, _ in deals)
    hero = np.array([cfg.starting_stacks[h] / BB for cfg, h in deals])
    others = np.array([s / BB for cfg, h in deals for i, s in enumerate(cfg.starting_stacks) if i != h])
    assert hero.min() >= 19.25 and 0.45 < np.mean(hero == 20.0) < 0.65  # (yours: the auto top-up)
    assert np.mean(others < 19.25) > 0.15                                # (theirs: short stacks too)


@pytest.mark.parametrize("raw, why", [
    (None, "the local build: no Hand review"),
    ({"paid": False, "hands": 0, "min_hands": 50}, "no subscription"),
    ({"paid": True, "hands": 12, "min_hands": 50}, "too few hands"),
])
def test_without_your_hands_my_tables_are_typical_clubgg_tables(tmp_path, monkeypatch, raw, why):
    if raw is not None:
        monkeypatch.setattr(T, "_MY_TABLES", lambda: raw)
    else:
        monkeypatch.setattr(T, "_MY_TABLES", None)
    view = T.my_tables_view()
    assert view["source"] == "typical", why
    assert view["review"] is (raw is not None) and view["paid"] is bool(raw and raw["paid"])
    assert view["hands"] == (raw or {}).get("hands", 0) and "hero" not in view
    deals = _tables(_session(tmp_path), 300)
    seats = collections.Counter(cfg.num_seats for cfg, _ in deals)
    assert seats[5] + seats[6] > 0.6 * len(deals) and seats[2] + seats[3] < 0.15 * len(deals)
    assert all(cfg.ante == 3 * BB for cfg, _ in deals)
    hero = np.array([cfg.starting_stacks[h] / BB for cfg, h in deals])
    assert np.mean(hero < 19.5) > 0.05  # (no floor: your stack is drawn like everyone's)


def test_a_failing_profile_deals_typical_tables_instead_of_an_error(tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("database is locked")

    monkeypatch.setattr(T, "_MY_TABLES", boom)
    assert T.my_tables_view()["source"] == "typical"
    assert len(_tables(_session(tmp_path), 3)) == 3


def test_the_setting_round_trips_and_custom_is_the_old_deal(tmp_path):
    assert T.TrainerSettings().tables == "custom"
    ts = _session(tmp_path, tables="custom", seed=3)
    ts.set_settings(T.TrainerSettings(**{**ts.settings.model_dump(), "seats_mode": "fixed", "seats_fixed": 4,
                                         "stacks_mode": "fixed", "stack_bb": 37.0, "ante_bb": 2.0}))
    for cfg, _ in _tables(ts, 20):
        assert cfg.num_seats == 4 and set(cfg.starting_stacks) == {37 * BB} and cfg.ante == 2 * BB
    with pytest.raises(ValueError):
        T.TrainerSettings(tables="pokerstars")


def test_nlh_ignores_my_tables(tmp_path):
    ts = _session(tmp_path)
    assert T._engine_variant(VARIANT_NLH) == VARIANT_NLH
    ts.variant = VARIANT_NLH  # (the format switch would also swap models; the deal only reads the variant)
    ts.settings = T.TrainerSettings(**{**T._default_settings(VARIANT_NLH).model_dump(), "tables": "mine",
                                       "seats_mode": "fixed", "seats_fixed": 3, "mc_rollouts": 0})
    cfg = ts._deal_one(False) and ts.hand.config
    assert cfg.num_seats == 3 and cfg.sb == BB // 2  # (NLH's own table, not the bomb pot's)


def test_the_client_offers_it():
    static = Path(T.__file__).resolve().parent / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "app.trainer.js").read_text(encoding="utf-8")
    assert 'id="ts-tables"' in html and '<option value="mine">My tables</option>' in html
    assert 'id="trainer-mytables-btn"' in html and ">My tables</button>" in html
    assert "trainer-clubgg-btn" not in html and 'value="clubgg"' not in html
    assert "tables:" in js and "toggleMyTables" in js and "renderMyTablesToggle(s)" in js
    assert 'getJSON("/trainer/my_tables")' in js
