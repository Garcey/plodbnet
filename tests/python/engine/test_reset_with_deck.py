"""`BombPotEnv.reset_with_deck` — dealing from an EXPLICIT deck order (the home
games' verifiable shuffle needs the deck the players' devices helped permute).

The contract: it is the SAME deal as the seeded one. A hand dealt from
`shuffled_deck(seed)` is bit-identical to a hand dealt from `seed` — every
observation, every legal mask, the payouts — and the slot map is public:
seat s's k-th hole card = deck[5s + k] (every seat index, dealt in or not),
board A = deck[5n : 5n+5], board B = deck[5n+5 : 5n+10].
"""
from __future__ import annotations

import numpy as np
import pytest

from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv


def _cfg(n=6):
    return GameConfig(num_seats=n, starting_stack=400_000, ante=30_000, bb=10_000)


@pytest.mark.parametrize("seed,button,mask", [
    (1, 0, None),
    (99, 3, [True, True, False, True, True, True]),
    (2**62 + 17, 5, [True, False, False, True, False, False]),
])
def test_a_hand_from_the_seeds_deck_is_bit_identical_to_the_seeded_hand(seed, button, mask):
    a, b = BombPotEnv(_cfg()), BombPotEnv(_cfg())
    deck = BombPotEnv.shuffled_deck(seed)
    assert sorted(deck) == list(range(52))
    oa, ia = a.reset(seed, button, in_hand_mask=mask)
    ob, ib = b.reset_with_deck(deck, button, in_hand_mask=mask)
    assert np.array_equal(oa, ob) and ia.actor == ib.actor
    assert np.array_equal(ia.gate_mask, ib.gate_mask)
    rng = np.random.default_rng(seed % 1000)
    for _ in range(80):
        if a.is_terminal():
            break
        legal = [g for g in range(3) if ia.gate_mask[g]]
        gate = int(rng.choice(legal))
        chips = int(ia.min_raise_chips) if gate == 2 else 0
        oa, _, da, ia = a.step_hybrid(gate, chips)
        ob, _, db, ib = b.step_hybrid(gate, chips)
        assert np.array_equal(oa, ob) and da == db
    assert a.is_terminal() and b.is_terminal()
    assert np.array_equal(a.terminal_rewards(), b.terminal_rewards())
    assert [list(h) for h in a.all_hole_cards()] == [list(h) for h in b.all_hole_cards()]


def test_the_slot_map_is_public_and_does_not_depend_on_who_sits_out():
    deck = BombPotEnv.shuffled_deck(7)
    n = 6
    for mask in (None, [True, False, True, False, True, False]):
        env = BombPotEnv(_cfg(n))
        env.reset_with_deck(deck, 0, in_hand_mask=mask)
        holes = [[int(c) for c in h] for h in env.all_hole_cards()]
        for s in range(n):
            assert holes[s] == deck[5 * s: 5 * s + 5]
        for _ in range(60):  # check it down to the river: both full boards are on the table
            if env.is_terminal():
                break
            env.step_hybrid(1, 0)
        raw = env._rs.observation_dict()
        assert [int(c) for c in raw["board_a"]] == deck[5 * n: 5 * n + 5]
        assert [int(c) for c in raw["board_b"]] == deck[5 * n + 5: 5 * n + 10]


def test_anything_but_a_real_52_card_deck_is_refused():
    env = BombPotEnv(_cfg())
    good = BombPotEnv.shuffled_deck(3)
    for bad in (good[:51], good + [0], [good[1]] + good[1:], [52] + good[1:]):
        with pytest.raises((ValueError, OverflowError)):
            env.reset_with_deck(bad, 0)
    with pytest.raises(ValueError):
        env.reset_with_deck(good, 0, in_hand_mask=[True, False, False, False, False, False])
    with pytest.raises(ValueError):  # no seed to run the training-time EV runout from
        BombPotEnv(_cfg(), ev_runout_samples=8).reset_with_deck(good, 0)
