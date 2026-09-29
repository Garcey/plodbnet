"""The serial env's one-row ENGINE encode (2026-09-28, ML-008).

`BombPotEnv` (the site's Study / Trainer, eval, the exploit probe) encodes
with `_engine.encode_game_state` -- the engine encoder training uses -- for
the PLO variants with fixed-size hands. The scalar numpy encoders are the
frozen oracle: every decision of many random hands (2-6 seats, PLO4/5/6, flop
to river, both observation-semantics revisions, full and minimal layouts)
must encode to the SAME BITS both ways, so Study / Trainer see exactly the
observations they saw before the switch. The env builds the raw dict with the
opp-outcome MC (as it always did) and hands that block to the engine encode,
so the MC still runs once per decision; it encodes at the revision the Python
side reads at call time, as the numpy encoders did.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp import _engine
from plo5bp import encoding as E
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv

VARIANTS = ("plo5_double_bomb", "plo4_double_bomb", "plo6_double_bomb")


def _oracle(env: BombPotEnv) -> np.ndarray:
    """The scalar numpy encoder on the env's current state (the old path)."""
    mode = env._obs_mode
    raw = dict(env._rs.observation_dict(skip_outcome_mc=(mode == "minimal")))
    actor = raw["actor"]
    if actor is not None:
        raw["hero_category_a"] = int(env._rs.hero_category(actor, 0))
        raw["hero_category_b"] = int(env._rs.hero_category(actor, 1))
    enc = E.encode_observation_minimal if mode == "minimal" else E.encode_observation
    return enc(raw, env.config)


def _random_legal(info, rng):
    legal = np.flatnonzero(info.gate_mask)
    gate = int(rng.choice(legal))
    chips = 0
    if gate == 2:
        lo, hi = int(info.min_raise_chips), int(info.max_raise_chips)
        chips = lo if hi <= lo else int(rng.integers(lo, hi + 1))
    return gate, chips


@pytest.mark.parametrize("rev", [2, 1])
@pytest.mark.parametrize("mode", ["full", "minimal"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_engine_encode_is_the_numpy_oracle_bit_for_bit(variant, mode, rev, monkeypatch) -> None:
    monkeypatch.setattr(E, "OBS_SEMANTICS_REV", rev)
    rng = np.random.default_rng(hash((variant, mode, rev)) % 2**32)
    checked = 0
    for hand in range(30):
        seats = int(rng.integers(2, 7)) if variant != "plo6_double_bomb" else int(rng.integers(2, 8))
        stacks = tuple(int(rng.integers(4, 300)) * 10_000 for _ in range(seats))
        cfg = GameConfig(num_seats=seats, starting_stack=stacks[0], ante=30_000,
                         bb=10_000, starting_stacks=stacks, variant=variant)
        env = BombPotEnv(cfg, obs_mode=mode)
        env._rs = _engine.GameState(
            num_seats=seats, starting_stack=0, ante=30_000, bb=10_000,
            starting_stacks=np.asarray(stacks, dtype=np.uint64), variant=variant, obs_rev=rev,
        )
        obs, info = env.reset(int(rng.integers(0, 2**63 - 1)), int(rng.integers(0, seats)))
        while not info.terminal:
            want = _oracle(env)
            got = np.asarray(_engine.encode_game_state(env._rs, mode), dtype=np.float32)
            assert got.shape == want.shape
            # ... and it IS what the env hands out
            assert np.array_equal(np.asarray(obs).view(np.uint32), want.view(np.uint32))
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (
                f"{variant} {mode} rev {rev} hand {hand}: columns "
                f"{np.flatnonzero(got.view(np.uint32) != want.view(np.uint32))[:10]}"
            )
            checked += 1
            obs, _r, done, info = env.step_hybrid(*_random_legal(info, rng))
            if done:
                break
    assert checked > 60


@pytest.mark.parametrize("mode", ["full", "minimal"])
def test_study_mode_nodes_match_too(mode) -> None:
    """The Study tab's states (reset_study with the user's cards, villains on
    placeholder cards, streets set by hand -- street boundaries included)."""
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(12):
        seats = int(rng.integers(2, 7))
        cfg = GameConfig(num_seats=seats, starting_stack=int(rng.integers(10, 200)) * 10_000,
                         ante=30_000, bb=10_000)
        env = BombPotEnv(cfg, obs_mode=mode)
        deck = [int(c) for c in rng.permutation(52)]
        hero = int(rng.integers(0, seats))
        _obs, info = env.reset_study(int(rng.integers(0, seats)), hero, deck[:5],
                                     deck[5:8], deck[8:11])
        streets = iter([(deck[11], deck[12]), (deck[13], deck[14])])
        for _step in range(40):
            want = _oracle(env)
            got = np.asarray(_engine.encode_game_state(env._rs, mode), dtype=np.float32)
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
            checked += 1
            if env.awaiting_next_street() is not None:
                nxt = next(streets, None)
                if nxt is None:
                    break
                setter = env.set_turn if env.awaiting_next_street() == 2 else env.set_river
                _obs, info = setter(*nxt)
                continue
            if info.terminal or info.actor is None:
                break
            _obs, _r, done, info = env.step_hybrid(*_random_legal(info, rng))
            if done:
                break
    assert checked > 40


def test_what_the_engine_encode_refuses() -> None:
    nlh = _engine.GameState(num_seats=2, variant="nlh_single", sb=5_000)
    nlh.reset(1, 0)
    with pytest.raises(RuntimeError, match="PLO-only"):
        _engine.encode_game_state(nlh, "full")
    plo = _engine.GameState(num_seats=3)
    with pytest.raises(RuntimeError, match="reset"):
        _engine.encode_game_state(plo, "full")  # no hand dealt
    plo.reset(1, 0)
    with pytest.raises(ValueError, match="layout"):
        _engine.encode_game_state(plo, "wide")
    with pytest.raises(ValueError, match="minimal layout has none"):
        _engine.encode_game_state(plo, "minimal", outcome=[0.0] * 22)
    with pytest.raises(ValueError, match="22 floats"):
        _engine.encode_game_state(plo, "full", outcome=[0.0] * 20)
    with pytest.raises(ValueError, match="obs_rev"):
        _engine.encode_game_state(plo, "full", obs_rev=3)


def _mc_block(raw: dict) -> list[float]:
    return (list(raw["opp_outcome_fractions"]) + list(raw["per_board_outcome"])
            + list(raw["share_bounds"]))


def test_a_given_outcome_block_is_the_block_the_encode_computes() -> None:
    """`outcome=` (the dict's MC, what the env passes) and the encode's own MC
    give the same bits, and that is the numpy oracle's."""
    rng = np.random.default_rng(11)
    checked = 0
    for hand in range(20):
        seats = int(rng.integers(2, 7))
        env = BombPotEnv(GameConfig(num_seats=seats))
        obs, info = env.reset(int(rng.integers(0, 2**62)), int(rng.integers(0, seats)))
        while not info.terminal:
            raw = dict(env._rs.observation_dict())
            own = np.asarray(_engine.encode_game_state(env._rs, "full"))
            given = np.asarray(_engine.encode_game_state(env._rs, "full", outcome=_mc_block(raw)))
            want = _oracle(env)
            assert np.array_equal(own.view(np.uint32), want.view(np.uint32))
            assert np.array_equal(given.view(np.uint32), want.view(np.uint32)), hand
            assert np.array_equal(np.asarray(obs).view(np.uint32), want.view(np.uint32))
            checked += 1
            obs, _r, done, info = env.step_hybrid(*_random_legal(info, rng))
            if done:
                break
    assert checked > 40


def test_the_env_hands_the_engine_its_own_mc_block(monkeypatch) -> None:
    """The MC runs once per decision: the full layout's raw dict carries it
    at full fidelity and the engine encode receives exactly that block."""
    import plo5bp.env as env_mod

    seen: list[dict] = []
    real = env_mod._engine_encode_state

    def spy(state, layout="full", **kw):
        seen.append({"layout": layout, **kw})
        return real(state, layout, **kw)

    monkeypatch.setattr(env_mod, "_engine_encode_state", spy)
    env = BombPotEnv(GameConfig())
    _obs, info = env.reset(7, 0)
    direct = dict(env._rs.observation_dict())
    assert np.asarray(direct["opp_outcome_fractions"])[4:].any()
    for key in ("opp_outcome_fractions", "per_board_outcome", "share_bounds"):
        assert info.raw_obs[key] == direct[key]
    assert len(seen) == 1 and seen[0]["outcome"] == _mc_block(direct)
    assert seen[0]["obs_rev"] == E.OBS_SEMANTICS_REV
    env_min = BombPotEnv(GameConfig(), obs_mode="minimal")
    env_min.reset(7, 0)
    assert seen[1] == {"layout": "minimal", "opp_outcome_mc": 0, "obs_rev": E.OBS_SEMANTICS_REV}


@pytest.mark.parametrize("mode", ["full", "minimal"])
def test_the_revision_is_read_at_call_time(mode, monkeypatch) -> None:
    """An env built under one revision and used after the Python side is
    re-pinned encodes at the NEW one, as the numpy encoders did (the engine
    state keeps the revision it was built with; the encode is told)."""
    here = int(E.OBS_SEMANTICS_REV)
    other = 1 if here == 2 else 2
    env = BombPotEnv(GameConfig(num_seats=3, starting_stacks=(40_000, 250_000, 90_000)),
                     obs_mode=mode)
    obs_here, _ = env.reset(21, 1)
    assert int(env._rs.obs_rev()) == here
    monkeypatch.setattr(E, "OBS_SEMANTICS_REV", other)
    rng = np.random.default_rng(3)
    obs, info = env.reset(21, 1)
    differs = not np.array_equal(np.asarray(obs), np.asarray(obs_here))
    for _ in range(12):
        want = _oracle(env)
        assert np.array_equal(np.asarray(obs).view(np.uint32), want.view(np.uint32))
        obs, _r, done, info = env.step_hybrid(*_random_legal(info, rng))
        if done:
            break
    assert differs  # a legal-raise-window dim is revision-gated on this deal


def test_a_stand_in_state_keeps_the_numpy_encoder() -> None:
    """A forwarding proxy in place of the engine state (the review tests'
    spies) is encoded by the numpy oracle -- the same bits."""

    class Proxy:
        def __init__(self, rs):
            self._inner = rs

        def __getattr__(self, name):
            return getattr(self._inner, name)

    a = BombPotEnv(GameConfig())
    b = BombPotEnv(GameConfig())
    obs_a, info = a.reset(5, 2)
    b.reset(5, 2)
    b._rs = Proxy(b._rs)
    obs_b, _ = b.reset(5, 2)
    rng = np.random.default_rng(5)
    for _ in range(10):
        assert np.array_equal(np.asarray(obs_a).view(np.uint32), np.asarray(obs_b).view(np.uint32))
        g, c = _random_legal(info, rng)
        obs_a, _r, done, info = a.step_hybrid(g, c)
        obs_b, _r, _d, _i = b.step_hybrid(g, c)
        if done:
            break
