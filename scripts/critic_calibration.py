#!/usr/bin/env python
"""Critic calibration from a rollout sample (2026-09-26, the regression diagnosis).

Reads the npz that `PLO5BP_DUMP_BATCH=<file> scripts/train.py ... --gae-lambda
1.0` writes (see train.py `_dump_batch_diagnostics`): with lambda = 1 the stored
returns are the Monte-Carlo returns, so for every group of rows

  bias = mean(return - V)                     (0 for a calibrated critic)
  EV   = 1 - Var(return - V) / Var(return)    (share of the return's variance
                                               the critic explains; the rest is
                                               luck still to come: cards, and
                                               everybody's sampled actions)

by street, by pot size, by stack depth and by the critic's own value decile,
plus the advantage scale per gate action.

    .venv/Scripts/python scripts/critic_calibration.py runs/diag/batch_u1290.npz
"""

from __future__ import annotations

import argparse

import numpy as np

STREETS = ("preflop", "flop", "turn", "river")
GATES = ("fold", "check/call", "raise")


def stats(r, v):
    d = r - v
    var_r = float(r.var())
    return {
        "n": int(r.size),
        "mean_R": float(r.mean()),
        "mean_V": float(v.mean()),
        "bias": float(d.mean()),
        "se_bias": float(d.std() / np.sqrt(max(r.size, 1))),
        "sd_R": float(np.sqrt(var_r)),
        "sd_resid": float(d.std()),
        "EV": float(1.0 - d.var() / var_r) if var_r > 0 else float("nan"),
    }


def row(label, s):
    return (f"  {label:<22} n {s['n']:>8}  R {s['mean_R']:+8.2f}  V {s['mean_V']:+8.2f}  "
            f"bias {s['bias']:+7.2f} (+-{s['se_bias']:.2f})  sdR {s['sd_R']:7.2f}  "
            f"sd(R-V) {s['sd_resid']:7.2f}  EV {s['EV']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("npz")
    args = ap.parse_args()
    z = np.load(args.npz)
    r, v = z["returns"].astype(np.float64), z["values"].astype(np.float64)
    street, pot = z["street"], z["pot_bb"]
    stack = z["hero_stack_bb"]
    print(f"{args.npz}: {r.size} rows (of {int(z['rows_total'][0])})")
    print(row("ALL", stats(r, v)))
    print("by street:")
    for k in range(4):
        m = street == k
        if m.any():
            print(row(STREETS[k], stats(r[m], v[m])))
    print("by pot (obs column 184, as encoded):")
    qs = np.quantile(pot, [0, 0.25, 0.5, 0.75, 0.9, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (pot >= lo) & (pot <= hi)
        print(row(f"pot {lo:.2f}..{hi:.2f}", stats(r[m], v[m])))
    print("by hero stack (obs column 176, as encoded):")
    qs = np.quantile(stack, [0, 0.33, 0.67, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (stack >= lo) & (stack <= hi)
        print(row(f"stack {lo:.2f}..{hi:.2f}", stats(r[m], v[m])))
    print("calibration by the critic's value decile (mean return per bin):")
    qs = np.quantile(v, np.linspace(0, 1, 11))
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (v >= lo) & (v <= hi)
        print(row(f"V {lo:+.1f}..{hi:+.1f}", stats(r[m], v[m])))
    adv, ga = z["advantages"].astype(np.float64), z["gate_actions"]
    print("advantage scale vs pot (does a big pot's noise drown the small pots?):")
    qs = np.quantile(pot, [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    tot = float((adv * adv).sum())
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (pot >= lo) & (pot <= hi)
        a = adv[m]
        print(f"  pot {lo:8.2f}..{hi:8.2f}  n {m.sum():>8}  sd(A) {a.std():8.3f}  "
              f"share of sum A^2 {float((a * a).sum()) / tot:6.3f}  sd(R) {r[m].std():8.2f}")
    print("advantages (as stored: per-config normalized) by gate action:")
    for g in range(3):
        m = ga == g
        if m.any():
            a = adv[m]
            print(f"  {GATES[g]:<11} n {m.sum():>8}  mean {a.mean():+8.3f}  sd {a.std():8.3f}  "
                  f"p1/p50/p99 {np.quantile(a, 0.01):+8.2f} {np.quantile(a, 0.5):+8.2f} {np.quantile(a, 0.99):+8.2f}")


if __name__ == "__main__":
    main()
