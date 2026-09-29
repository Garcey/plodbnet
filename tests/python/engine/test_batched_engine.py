"""Bit-exact parity of PyBatchedEngine vs PyGameState (Phase A golden test).

Runs the same scripted (reset-seed, button, action-sequence) through both
engines for every (num_seats, starting_stack, batch_size, seed) cell in a
small matrix. Asserts equality at every step for:

- legal_action_mask (N, 8)
- current actor
- is_terminal flag
- observation_arrays scalar/vector fields (pot, street, stacks, bet_to_call,
  history, board_a, board_b, hero_hole, min/max bet, folded, all_in)
- payouts_batch and payouts_ev_batch at terminal

Each env within a batch is driven by a distinct pinned NumPy Generator so
action choices are independent across envs — this exercises heterogeneous
state within one batched call.

Coverage comes from the product of the cells; keeping each cell small
(short action sequences via random-legal play) keeps total runtime under
a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp._engine import BatchedEngine, GameState  # type: ignore[attr-defined]


def _random_legal_action(state: GameState, rng: np.random.Generator) -> int:
    mask = np.asarray(state.legal_action_mask(), dtype=bool)
    legal = np.flatnonzero(mask)
    assert legal.size > 0
    return int(rng.choice(legal))


def _drive_parity(
    num_seats: int,
    starting_stack: int,
    batch_size: int,
    base_seed: int,
) -> None:
    ante = 30000
    bb = 10000
    rng = np.random.default_rng(base_seed)
    # Per-env RNG so actions differ across the batch (exercises hetero state).
    env_rngs = [np.random.default_rng(base_seed * 1_000 + i) for i in range(batch_size)]
    seeds = rng.integers(0, 2**63 - 1, size=batch_size, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, num_seats, size=batch_size, dtype=np.int64).astype(np.uint8)

    # Reference: serial engines.
    serial: list[GameState] = []
    for i in range(batch_size):
        gs = GameState(num_seats, starting_stack, ante, bb)
        gs.reset(int(seeds[i]), int(buttons[i]))
        serial.append(gs)

    # Batched.
    be = BatchedEngine(
        batch_size,
        num_seats=num_seats,
        starting_stack=starting_stack,
        ante=ante,
        bb=bb,
    )
    be.reset_batch(seeds, buttons)

    step = 0
    while not be.is_terminal_batch().all():
        # Compare masks before stepping.
        mb = be.legal_mask_batch()
        actors = be.actor_batch()
        term = be.is_terminal_batch()
        for i in range(batch_size):
            if serial[i].is_terminal():
                assert term[i], f"env {i} step {step}: batched not terminal"
                assert actors[i] == -1
                assert not mb[i].any(), f"env {i} step {step}: mask non-empty on terminal"
            else:
                assert not term[i], f"env {i} step {step}: batched says terminal"
                assert int(actors[i]) == serial[i].current_actor(), (
                    f"env {i} step {step}: actor batched={actors[i]} "
                    f"serial={serial[i].current_actor()}"
                )
                ms = np.asarray(serial[i].legal_action_mask(), dtype=bool)
                assert np.array_equal(mb[i], ms), (
                    f"env {i} step {step}: mask mismatch batched={mb[i]} serial={ms}"
                )

        # Compare observation arrays field-by-field against serial dicts.
        obs = be.observation_arrays()
        for i in range(batch_size):
            if serial[i].is_terminal():
                continue
            d = dict(serial[i].observation_dict())
            hole = list(d["hero_hole"])
            assert list(obs["hero_hole"][i][: len(hole)]) == list(hole), (
                f"env {i} step {step}: hero_hole mismatch"
            )
            ba = list(d["board_a"])
            bb_ = list(d["board_b"])
            assert int(obs["board_a_len"][i]) == len(ba)
            assert int(obs["board_b_len"][i]) == len(bb_)
            assert list(obs["board_a"][i][: len(ba)]) == list(ba), (
                f"env {i} step {step}: board_a mismatch"
            )
            assert list(obs["board_b"][i][: len(bb_)]) == list(bb_), (
                f"env {i} step {step}: board_b mismatch"
            )
            assert int(obs["street"][i]) == int(d["street"])
            assert int(obs["pot"][i]) == int(d["pot"])
            assert int(obs["bet_to_call"][i]) == int(d["bet_to_call"])
            assert int(obs["min_bet"][i]) == int(d["min_bet"])
            assert int(obs["max_bet"][i]) == int(d["max_bet"])
            assert int(obs["button"][i]) == int(d["button"])
            assert list(obs["stacks"][i]) == list(d["stacks"])
            assert list(obs["folded"][i]) == [bool(x) for x in d["folded"]]
            assert list(obs["all_in"][i]) == [bool(x) for x in d["all_in"]]
            assert list(obs["street_commit"][i]) == list(d["street_commit"])
            assert list(obs["total_commit"][i]) == list(d["total_commit"])
            hist = list(d["history"])[-32:]
            hlen = int(obs["history_len"][i])
            assert hlen == len(hist), (
                f"env {i} step {step}: history_len mismatch"
            )
            for slot, (seat, act, _chips, _street) in enumerate(hist):
                assert int(obs["history_seat"][i][slot]) == int(seat)
                assert int(obs["history_action"][i][slot]) == int(act)

        # Pick one action per env (random-legal for live envs, 0 for terminal).
        actions = np.zeros(batch_size, dtype=np.uint8)
        for i in range(batch_size):
            if not serial[i].is_terminal():
                actions[i] = _random_legal_action(serial[i], env_rngs[i])

        # Apply in parallel.
        terminal_flags = be.apply_action_batch(actions)
        for i in range(batch_size):
            if not serial[i].is_terminal():
                serial[i].apply_action(int(actions[i]))
        # terminal_flags[i] true exactly when env transitioned this step.

        step += 1
        if step > 400:
            raise AssertionError("hand did not terminate within 400 steps")

    # All envs terminal — compare payouts.
    pb = be.payouts_batch()
    for i in range(batch_size):
        ps = np.asarray(serial[i].payouts(), dtype=np.int64)
        assert np.array_equal(pb[i], ps), (
            f"env {i}: payouts batched={pb[i]} serial={ps}"
        )
        assert pb[i].sum() == 0, f"env {i}: not zero-sum"

    # EV payouts: compare at a scripted per-env seed. Only needs matching
    # when the terminal state is a run-out (close_len < 5); otherwise
    # payouts_ev delegates to payouts and they match trivially.
    ev_seeds = (seeds ^ np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
    ev_b = be.payouts_ev_batch(32, ev_seeds)
    for i in range(batch_size):
        ev_s = np.asarray(
            serial[i].payouts_ev(32, int(ev_seeds[i])), dtype=np.int64
        )
        assert np.array_equal(ev_b[i], ev_s), (
            f"env {i}: payouts_ev batched={ev_b[i]} serial={ev_s}"
        )


@pytest.mark.parametrize("num_seats", [2, 3, 4, 6])
@pytest.mark.parametrize("starting_stack", [200000, 1000000])  # 20bb, 100bb
@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("base_seed", [0, 1, 2])
def test_batched_vs_serial_parity(
    num_seats: int, starting_stack: int, batch_size: int, base_seed: int
) -> None:
    _drive_parity(num_seats, starting_stack, batch_size, base_seed)


def test_large_batch_256() -> None:
    """One bigger batch to stress the shape-256 path."""
    _drive_parity(num_seats=6, starting_stack=200000, batch_size=256, base_seed=7)


def test_category_batch_parity() -> None:
    """hero_category_batch must match serial per-env."""
    num_seats = 6
    n = 16
    seeds = np.arange(n, dtype=np.uint64) + 1000
    buttons = np.zeros(n, dtype=np.uint8)

    be = BatchedEngine(n, num_seats=num_seats, starting_stack=200000, ante=30000, bb=10000)
    be.reset_batch(seeds, buttons)

    serial = []
    for i in range(n):
        gs = GameState(num_seats, 200000, 30000, 10000)
        gs.reset(int(seeds[i]), 0)
        serial.append(gs)

    # Sample per-env seats/boards to probe.
    rng = np.random.default_rng(42)
    seats = rng.integers(0, num_seats, size=n).astype(np.uint8)
    boards = rng.integers(0, 2, size=n).astype(np.uint8)

    cat_b = be.hero_category_batch(seats, boards)

    for i in range(n):
        cat_s = int(serial[i].hero_category(int(seats[i]), int(boards[i])))
        assert int(cat_b[i]) == cat_s, f"env {i}: category mismatch"


def test_reset_terminal_batch_leaves_non_masked_envs_untouched() -> None:
    n = 4
    seeds = np.arange(n, dtype=np.uint64) + 50
    buttons = np.zeros(n, dtype=np.uint8)
    be = BatchedEngine(n, num_seats=6, starting_stack=200000, ante=30000, bb=10000)
    be.reset_batch(seeds, buttons)
    # Step env 0 once (CheckCall) but not others.
    actions = np.ones(n, dtype=np.uint8)
    be.apply_action_batch(actions)
    obs_before = be.observation_arrays()
    pot_before = obs_before["pot"].copy()
    actor_before = be.actor_batch().copy()

    # Now reset-terminal for env 0 only (mask=[1,0,0,0]).
    mask = np.array([True, False, False, False], dtype=bool)
    new_seeds = np.array([999, 0, 0, 0], dtype=np.uint64)
    new_buttons = np.array([3, 0, 0, 0], dtype=np.uint8)
    be.reset_terminal_batch(new_seeds, new_buttons, mask)

    obs_after = be.observation_arrays()
    # Env 0 should now reflect button=3 → actor=4.
    assert be.actor_batch()[0] == 4
    # Envs 1-3 untouched.
    for i in range(1, n):
        assert obs_after["pot"][i] == pot_before[i]
        assert be.actor_batch()[i] == actor_before[i]
