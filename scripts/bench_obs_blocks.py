"""Diagnose per-block cost of the v7 obs encoder + engine dims.

Breaks down a postflop observation into:
  - engine: outcome_features_mc (opp_outcome + per_board + share_bounds)
  - engine: hero_board_v3 (BRD-7/12 + DUAL-2)
  - engine: full observation_dict
  - python: encode_observation full
  - python: encode with each v3 block disabled (via monkey-patch stubs)
  - python: encode_observation_batch (N=64/256)

Usage:
  .venv/Scripts/python scripts/bench_obs_blocks.py
  .venv/Scripts/python scripts/bench_obs_blocks.py --n 2000 --batch 256
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp import encoding as enc


def timeit(fn, n: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def land_postflop(seed: int = 12345):
    env = BombPotEnv(GameConfig())
    obs, info = env.reset(seed=seed, button=0)
    # Bomb pot starts at flop; confirm outcome block is live.
    fr = np.asarray(env._rs.opp_outcome_fractions(), dtype=np.float32)
    assert fr.any(), "expected live postflop node"
    return env, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--street", choices=["flop", "turn", "river"], default="flop")
    args = ap.parse_args()
    N = args.n

    env, _ = land_postflop()
    rs = env._rs
    cfg = env.config

    # Advance to requested street if possible (cheap: check/call through).
    if args.street != "flop":
        target = {"turn": 2, "river": 3}[args.street]
        for _ in range(40):
            st = rs.observation_dict()
            if st.get("street") is None:
                break
            street = int(st["street"]) if not isinstance(st["street"], str) else {
                "Flop": 1, "Turn": 2, "River": 3
            }.get(st["street"], 1)
            # observation_dict street is often int index; fall back via board len
            ba = list(st["board_a"])
            street_from_board = {3: 1, 4: 2, 5: 3}.get(len(ba), 1)
            if street_from_board >= target:
                break
            # check/call when legal
            try:
                env.step(1)  # CHECK_CALL
            except Exception:
                break
        env2, _ = land_postflop(seed=999)
        # keep original if advance failed; otherwise re-land is fine
        del env2

    raw = dict(rs.observation_dict())
    print(f"street board lens: A={len(raw['board_a'])} B={len(raw['board_b'])}")
    print(f"actor={raw.get('actor')}  opp_outcome any={np.any(raw.get('opp_outcome_fractions'))}")
    print(f"hero_board_v3={list(raw.get('hero_board_v3', []))}")
    print(f"share_bounds={list(raw.get('share_bounds', []))}")
    print()

    # ---- Engine microbench ----
    print(f"=== ENGINE (avg of {N}, single node) ===")
    t_out = timeit(lambda: rs.outcome_features_mc(384), N)  # train budget
    t_out1024 = timeit(lambda: rs.outcome_features_mc(1024), N)
    t_hb = timeit(lambda: rs.observation_dict() and None, 1)  # warm
    # Direct methods if exposed
    try:
        t_hb = timeit(lambda: list(rs.observation_dict()["hero_board_v3"]), N)
        # Better: call through outcome + dict split
    except Exception:
        t_hb = float("nan")

    # Time hero_board_v3 by differencing observation_dict with/without is hard;
    # call outcome_features_mc + estimate hero_board via full dict.
    t_dict = timeit(lambda: rs.observation_dict(), N)
    t_dict_only_feats = t_out  # lower bound of dict cost from outcome

    # Try binding if present
    t_hb_direct = None
    if hasattr(rs, "hero_board_v3"):
        t_hb_direct = timeit(lambda: rs.hero_board_v3(), N)
    elif hasattr(rs, "outcome_features_mc"):
        # estimate: observation_dict includes outcome(1024) + hero_board + rest
        pass

    print(f"  outcome_features_mc( 384 train): {t_out*1e6:9.1f} us")
    print(f"  outcome_features_mc(1024 serial): {t_out1024*1e6:9.1f} us")
    print(f"  observation_dict (full)        : {t_dict*1e6:9.1f} us")
    if t_hb_direct is not None:
        print(f"  hero_board_v3()                : {t_hb_direct*1e6:9.1f} us")
    else:
        # Estimate hero_board cost: call outcome once outside, then measure
        # a pure python loop that re-fetches hero_board from a cached dict is
        # free. Instead re-call observation_dict and subtract known parts.
        residual = max(t_dict - t_out1024, 0.0)
        print(f"  observation_dict residual      : {residual*1e6:9.1f} us  "
              f"(dict - outcome@1024; includes hero_board_v3 + packing)")

    # ---- Python encoder microbench ----
    print(f"\n=== PYTHON encode_observation (avg of {N}) ===")
    t_full = timeit(lambda: enc.encode_observation(raw, cfg), N)
    print(f"  full encode_observation        : {t_full*1e6:9.1f} us")

    # Disable each v3 block via monkeypatch
    stubs = {
        "_encode_stack_v3": ("stack_v3 (STK*)", enc._encode_stack_v3),
        "_encode_board_v3": ("board_v3 (BRD*)", enc._encode_board_v3),
        "_encode_dual_v3": ("dual_v3 (DUAL*)", enc._encode_dual_v3),
        "_blocker_features": ("blockers (obs-v2)", enc._blocker_features),
        "_straight_flush_features": ("sf_features (legacy)", enc._straight_flush_features),
        "_cross_board_features": ("cross_board (legacy)", enc._cross_board_features),
        "_pair_features": ("pair_features (legacy)", enc._pair_features),
    }

    def noop(*a, **k):
        if a and isinstance(a[0], np.ndarray) and a[0].ndim == 1:
            return None
        return np.zeros(4, dtype=np.float32)

    def noop_sf(*a, **k):
        return np.zeros(38, dtype=np.float32)

    def noop_cross(*a, **k):
        return np.zeros(28, dtype=np.float32)

    def noop_pair(*a, **k):
        return (np.zeros(5, dtype=np.float32), np.zeros(4, dtype=np.float32))

    costs = {}
    for name, (label, orig) in stubs.items():
        if name == "_straight_flush_features":
            stub = noop_sf
        elif name == "_cross_board_features":
            stub = noop_cross
        elif name == "_pair_features":
            stub = noop_pair
        elif name == "_blocker_features":
            stub = lambda *a, **k: np.zeros(4, dtype=np.float32)
        else:
            stub = noop
        setattr(enc, name, stub)
        try:
            t_off = timeit(lambda: enc.encode_observation(raw, cfg), N)
        finally:
            setattr(enc, name, orig)
        delta = t_full - t_off
        costs[label] = delta
        print(f"  without {label:28s}: {t_off*1e6:9.1f} us  "
              f"(saves {delta*1e6:7.1f} us = {100*delta/max(t_full,1e-12):5.1f}%)")

    # All three v3 blocks off
    for name in ("_encode_stack_v3", "_encode_board_v3", "_encode_dual_v3"):
        setattr(enc, name, noop)
    try:
        t_no_v3 = timeit(lambda: enc.encode_observation(raw, cfg), N)
    finally:
        for name, (_, orig) in stubs.items():
            if name.startswith("_encode_"):
                setattr(enc, name, orig)
    print(f"  without ALL v3 blocks         : {t_no_v3*1e6:9.1f} us  "
          f"(saves {(t_full-t_no_v3)*1e6:7.1f} us = "
          f"{100*(t_full-t_no_v3)/max(t_full,1e-12):5.1f}%)")

    # ---- Batched encoder ----
    print(f"\n=== PYTHON encode_observation_batch (N={args.batch}) ===")
    # Build a fake batch by tiling the single dict arrays.
    # Prefer real env_batched if available; else synthesize.
    try:
        from plo5bp.env_batched import BatchedBombPotEnv
        benv = BatchedBombPotEnv(
            num_envs=args.batch, config=cfg, opp_outcome_mc=384
        )
        seeds = np.arange(args.batch, dtype=np.uint64) + 7
        buttons = np.zeros(args.batch, dtype=np.uint8)
        benv.reset_batch(seeds, buttons)
        # observation_arrays lives on the Rust BatchedEngine
        obs_arrays = benv._be.observation_arrays()
        arr = {k: np.asarray(v) for k, v in dict(obs_arrays).items()}
        n_batch_iters = max(30, N // 10)
        t_b = timeit(lambda: enc.encode_observation_batch(arr, cfg), n_batch_iters)
        per = t_b / args.batch
        print(f"  batch encode total             : {t_b*1e6:9.1f} us  "
              f"({per*1e6:7.1f} us/row)")

        b_stubs = {
            "_encode_stack_v3_batch": enc._encode_stack_v3_batch,
            "_encode_board_v3_batch": enc._encode_board_v3_batch,
            "_encode_dual_v3_batch": enc._encode_dual_v3_batch,
        }

        def bnoop(*a, **k):
            return None

        for name, orig in b_stubs.items():
            setattr(enc, name, bnoop)
            try:
                t_off = timeit(
                    lambda: enc.encode_observation_batch(arr, cfg),
                    n_batch_iters,
                )
            finally:
                setattr(enc, name, orig)
            delta = t_b - t_off
            print(f"  without {name:28s}: {t_off*1e6:9.1f} us  "
                  f"(saves {delta*1e6:7.1f} us = {100*delta/max(t_b,1e-12):5.1f}%) "
                  f"/ row {(t_off/args.batch)*1e6:6.1f} us")

        for name in b_stubs:
            setattr(enc, name, bnoop)
        try:
            t_off = timeit(
                lambda: enc.encode_observation_batch(arr, cfg),
                n_batch_iters,
            )
        finally:
            for name, orig in b_stubs.items():
                setattr(enc, name, orig)
        print(f"  without ALL v3 batch blocks    : {t_off*1e6:9.1f} us  "
              f"(saves {(t_b-t_off)*1e6:7.1f} us = "
              f"{100*(t_b-t_off)/max(t_b,1e-12):5.1f}%)")

        t_pack = timeit(lambda: benv._rs.observation_arrays(), n_batch_iters)
        print(f"  observation_arrays (engine)    : {t_pack*1e6:9.1f} us  "
              f"({(t_pack/args.batch)*1e6:7.1f} us/row)")
    except Exception as e:
        import traceback
        print(f"  batched path skipped: {type(e).__name__}: {e}")
        traceback.print_exc()

    # ---- Summary ranking ----
    print("\n=== SERIAL PYTHON BLOCK COST RANKING ===")
    for label, delta in sorted(costs.items(), key=lambda kv: -kv[1]):
        print(f"  {label:32s}  {delta*1e6:8.1f} us  "
              f"({100*delta/max(t_full,1e-12):5.1f}% of encode)")
    print(f"\n  note: opp_outcome lives in the ENGINE observation_dict, not "
          f"encode_observation. Compare engine residual vs python board_v3.")


if __name__ == "__main__":
    main()
