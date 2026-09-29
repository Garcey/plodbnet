"""The serial GameState's observation dict and heavy calls (PERF-027 / PERF-028).

- `observation_dict(skip_outcome_mc=True)` -- the bookkeeping / table-view dict
  -- carries the same public fields as the full one and computes none of the
  full-encoder-only features (they are zeros);
- `total_commit()` is the dict's `total_commit` without building a dict;
- the heavy calls (the full dict's 1024-sample Monte Carlo, EV runouts) run on a
  snapshot of the hand with the GIL released: other Python threads keep running
  meanwhile, and a thread driving the same object never trips "Already borrowed".
"""

from __future__ import annotations

import threading
import time

import numpy as np

from plo5bp._engine import GameState

# The features only the full-layout encoder reads.
FEATURES = (
    "opp_outcome_fractions",
    "per_board_outcome",
    "share_bounds",
    "hero_board_v3",
    "board_draw_v3",
    "nlh_opp_outcome",
)


def _state(seed: int, actions: int, variant: str = "plo5_double_bomb") -> GameState:
    kw = {"sb": 5000} if variant == "nlh_single" else {}
    gs = GameState(num_seats=6, starting_stack=1_000_000, ante=30_000, bb=10_000, variant=variant, **kw)
    gs.reset(seed, seed % 6)
    # NLH: limp around to the flop first (its sweep is all-zero preflop).
    while variant == "nlh_single" and gs.observation_dict(skip_outcome_mc=True)["street"] == 0:
        gs.apply_action(1)
    rng = np.random.default_rng(seed)
    for _ in range(actions):
        if gs.is_terminal():
            break
        legal = np.flatnonzero(gs.legal_action_mask())
        gs.apply_action(int(rng.choice(legal)))
    return gs


def test_the_lean_dict_has_the_public_fields_and_computes_no_features() -> None:
    computed = {k: False for k in FEATURES}
    for seed in range(40):
        variant = "nlh_single" if seed % 4 == 3 else "plo5_double_bomb"
        gs = _state(seed, seed % 9, variant)
        full, lean = gs.observation_dict(), gs.observation_dict(skip_outcome_mc=True)
        assert full.keys() == lean.keys()
        for k in full:
            if k in FEATURES:
                assert not any(lean[k]), (seed, k)
                computed[k] |= any(full[k])
            else:
                assert lean[k] == full[k], (seed, k)
        assert list(gs.total_commit()) == list(full["total_commit"])
    # ...and the full dict really computes every one of them somewhere.
    assert all(computed.values()), computed


def test_the_full_dict_matches_the_single_feature_calls() -> None:
    for seed in range(6):
        gs = _state(seed, 3)
        d = gs.observation_dict()
        mc = gs.outcome_features_mc(1024)
        assert d["opp_outcome_fractions"] + d["per_board_outcome"] + d["share_bounds"] == mc
        assert d == gs.observation_dict()  # deterministic


def test_heavy_calls_let_other_threads_run() -> None:
    gs = _state(7, 0)
    window: list[float] = []
    done = threading.Event()

    def work() -> None:
        t0 = time.perf_counter()
        gs.outcome_features_mc(200_000)
        window.extend([t0, time.perf_counter()])
        done.set()

    stamps: list[float] = []
    t = threading.Thread(target=work)
    t.start()
    while not done.is_set():
        stamps.append(time.perf_counter())
    t.join()
    t0, t1 = window
    # With the GIL held for the whole call this thread could not run a single
    # line of Python between t0 and t1.
    assert sum(t0 < s < t1 for s in stamps) > 0, (t1 - t0, len(stamps))


def test_a_second_thread_on_the_same_state_never_trips_the_borrow() -> None:
    gs = _state(3, 2)
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                gs.observation_dict()
                gs.payouts_ev(16, 1)
                gs.outcome_features_mc(256)
        except BaseException as e:  # noqa: BLE001 -- collected for the assert
            errors.append(e)

    def writer() -> None:
        try:
            rng = np.random.default_rng(0)
            for i in range(300):
                gs.reset(i, i % 6)
                while not gs.is_terminal():
                    gs.apply_action(int(rng.choice(np.flatnonzero(gs.legal_action_mask()))))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            stop.set()

    threads = [threading.Thread(target=reader), threading.Thread(target=writer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
