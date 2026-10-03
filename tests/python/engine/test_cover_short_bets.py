"""The covering bet into a short stack's last chips (2026-10-03; owner: "Got to the river
where the only opponent has less than 1bb … in a real poker app, I would be able to bet
$20+ and the opponent just calls for their remaining chips").

Under the rule every network trains on (a bet is capped at what the deepest opponent can
still put in), the only bet left against opponents with less than one big blind behind is
the COVERING bet: min = max = what they have (the engine's cover-short clamp). The Raise
gate's dust screen (`actions.gate_mask_from_bounds`) hid it, so the deep player could only
check. `GameConfig.cover_short_bets` (the website: Study, the Trainer, the graders) offers
it — only true dust under bb/100 stays screened. Training (the default, False) is unchanged,
the observation is the same either way, and the batched training engine refuses the flag."""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.env_batched import BatchedBombPotEnv

BB = 10_000
ANTE = 3 * BB


def _spot(behind, **kw):
    """Heads-up on the flop: you (seat 0, 100bb) act first; the other seat has `behind`
    chips left after the ante."""
    cfg = GameConfig(num_seats=2, starting_stack=0, starting_stacks=(100 * BB, ANTE + behind),
                     ante=ANTE, bb=BB, **kw)
    env = BombPotEnv(cfg, ev_runout_samples=0)
    obs, info = env.reset(7, 1)
    assert info.actor == 0
    return env, obs, info


def test_training_screens_the_covering_bet_as_before():
    _env, _obs, info = _spot(BB // 2)
    assert (info.min_raise_chips, info.max_raise_chips) == (BB // 2, BB // 2)  # (the engine's covering bet)
    assert list(info.gate_mask) == [False, True, False]  # check only: the rule the networks learned


def test_the_site_offers_it_and_the_short_stack_calls_all_in():
    env, _obs, info = _spot(BB // 2, cover_short_bets=True)
    assert list(info.gate_mask) == [False, True, True]
    assert (info.min_raise_chips, info.max_raise_chips) == (BB // 2, BB // 2)
    _obs, _r, done, info = env.step_hybrid(GATE_RAISE, BB // 2)
    assert not done and info.actor == 1
    assert list(info.gate_mask) == [True, True, False]  # fold, or call it all in
    _obs, _r, done, _info = env.step_hybrid(GATE_CHECK_CALL, 0)
    assert done  # (all in: the boards run out)
    pay = env.terminal_rewards()
    assert pay.sum() == 0 and abs(pay[0]) <= ANTE + BB // 2  # (nobody put in more than the short stack had)


@pytest.mark.parametrize("behind, offered", [(BB // 100 - 1, False), (BB // 100, True), (BB - 1, True), (BB, True)])
def test_only_dust_stays_screened(behind, offered):
    _env, _obs, info = _spot(behind, cover_short_bets=True)
    assert bool(info.gate_mask[2]) is offered
    assert info.max_raise_chips == behind  # (the engine offers the cover either way)


def test_the_observation_is_the_same_either_way():
    for behind in (BB // 2, BB - 1, 3 * BB):
        _e, plain, _i = _spot(behind)
        _e, site, _i = _spot(behind, cover_short_bets=True)
        assert np.array_equal(plain, site), behind


def test_the_batched_training_engine_refuses_it():
    with pytest.raises(ValueError, match="covering bet"):
        BatchedBombPotEnv(num_envs=2, config=GameConfig(cover_short_bets=True))
