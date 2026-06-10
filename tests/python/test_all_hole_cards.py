"""The all_hole_cards() engine accessor (trainer reveal path)."""

from __future__ import annotations

from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv


def test_shape_and_range():
    for n in (2, 4, 6):
        env = BombPotEnv(GameConfig(num_seats=n))
        env.reset(99, 0)
        holes = env.all_hole_cards()
        assert len(holes) == n
        assert all(len(h) == 5 for h in holes)
        flat = [c for h in holes for c in h]
        assert all(0 <= c < 52 for c in flat)
        assert len(set(flat)) == len(flat), "duplicate cards across seats"


def test_actor_row_matches_observation():
    env = BombPotEnv(GameConfig(num_seats=4))
    obs, info = env.reset(7, 2)
    holes = env.all_hole_cards()
    assert list(info.raw_obs["hero_hole"]) == holes[info.actor]
    # Step once; the next actor's observation row must match too.
    obs, _, done, info = env.step_hybrid(1, 0)
    if not done and info.actor is not None:
        assert list(info.raw_obs["hero_hole"]) == holes[info.actor]


def test_deterministic_redeal():
    a = BombPotEnv(GameConfig(num_seats=5))
    b = BombPotEnv(GameConfig(num_seats=5))
    a.reset(123456, 3)
    b.reset(123456, 3)
    assert a.all_hole_cards() == b.all_hole_cards()
    c = BombPotEnv(GameConfig(num_seats=5))
    c.reset(123457, 3)
    assert c.all_hole_cards() != a.all_hole_cards()


def test_disjoint_from_boards():
    env = BombPotEnv(GameConfig(num_seats=3, starting_stack=2_000_000))
    obs, info = env.reset(42, 0)
    # Check everyone down to the river so both boards fully deal.
    guard = 0
    done = False
    while not done and guard < 30:
        obs, _, done, info = env.step_hybrid(1, 0)
        guard += 1
    raw = env._rs.observation_dict()
    board = set(raw["board_a"]) | set(raw["board_b"])
    flat = {c for h in env.all_hole_cards() for c in h}
    assert not (board & flat), "hole cards overlap board cards"
