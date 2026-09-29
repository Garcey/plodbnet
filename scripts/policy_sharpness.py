#!/usr/bin/env python
"""Policy sharpness / collapse probe (2026-09-24 hyperparameter tuning).

How deterministic is a checkpoint's policy, measured on ONE fixed set of
decision states so every checkpoint is compared on the same spots? The states
come from the self-play of a reference checkpoint (`--states-from`) over
training-like table configs (10 configs per training tier) and are cached in
`--cache` with their observation revision / layout (2026-09-28): later calls
reuse them, and a checkpoint that cannot read them (another revision, a
layout its adapter refuses) is refused. Per tier and overall it reports

  - gate entropy over the legal gate actions: mean and p10 / p50 / p90
  - `rare_gate<1e-2` / `<1e-3`: share of states whose LEAST likely legal gate
    action has probability below 1% / 0.1% -- an action the policy has all but
    stopped trying there (the collapse the owner's high-entropy phase guards
    against; at 0 it can never be re-learned)
  - raise-size entropy (the legal-anchor distribution) over raise-legal states,
    and `rare_anchor<1e-3`: the share of legal anchors below 0.1%
  - `p_raise`: the mean raise probability where raising is legal

It prints a table and appends one JSON line per checkpoint to `--out`.

    .venv/bin/python scripts/policy_sharpness.py \
        --states-from checkpoints/swA32_40.pt --cache runs/sharpness_states.npz \
        checkpoints/t1lr150_59.pt checkpoints/t2ent15_59.pt

Library: plo5bp.evaluation.sharpness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from plo5bp.evaluation import ObsRevMismatch, load_actor
from plo5bp.evaluation.sharpness import (
    ANCHOR_NAMES,
    StatesMismatch,
    check_readable,
    ensure_states,
    measure,
)
from plo5bp.evaluation.tables import TIERS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--states-from", default="checkpoints/swA32_40.pt",
                    help="checkpoint whose self-play generates the probe states (first call)")
    ap.add_argument("--cache", default="runs/sharpness_states.npz")
    ap.add_argument("--cache-rev", type=int, default=None,
                    help="the obs revision a cache written before 2026-09-28 holds (no stamp)")
    ap.add_argument("--rows", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="runs/sharpness_history.jsonl")
    args = ap.parse_args(argv)
    torch.set_num_threads(int(args.threads))
    try:
        (obs, masks, sizing, tiers), info = ensure_states(
            args.cache, args.states_from, args.rows, args.cache_rev
        )
    except (StatesMismatch, ObsRevMismatch) as e:
        sys.exit(str(e))
    print(f"states: {obs.shape[0]} from {info['path']} (rev {info['obs_rev']}, "
          f"{info['obs_dim']}-wide, made by {info['states_from']})")

    def f(x, fmt="%.3f"):
        return "  -  " if x is None else fmt % x

    print(f"\n{'checkpoint':28s} {'tier':12s} {'gateH':>6s} {'p10/p50/p90':>17s} "
          f"{'rare<1%':>8s} {'rare<.1%':>8s} {'pRaise':>7s} {'sizeH':>6s} {'rareSz':>7s}")
    with open(args.out, "a") as fh:
        for path in args.ckpts:
            try:
                actor, meta = load_actor(path, check_rev=False)
                adapt = check_readable(info, meta, actor)
            except StatesMismatch as e:
                sys.exit(str(e))
            res = measure(actor, obs, masks, sizing, tiers, adapt=adapt)
            for k in ("ALL",) + TIERS:
                b = res[k]
                q = b["gate_h_p10_p50_p90"]
                qs = "  -  " if q is None else "%.2f/%.2f/%.2f" % tuple(q)
                print(f"{Path(path).name:28s} {k:12s} {f(b['gate_h']):>6s} {qs:>17s} "
                      f"{f(b['rare_gate_1e2']):>8s} {f(b['rare_gate_1e3']):>8s} "
                      f"{f(b['p_raise']):>7s} {f(b['anchor_h']):>6s} {f(b['rare_anchor_1e3']):>7s}")
            am = res["ALL"]["anchor_mean"]
            if am is not None:
                print(f"{'':28s} {'sizes':12s} top-size prob {res['ALL']['anchor_top']:.3f} | mean prob "
                      + " ".join("%s %.3f" % (n, x) for n, x in zip(ANCHOR_NAMES, am)))
            fh.write(json.dumps({"ckpt": str(path), "update": meta["update"],
                                 "states": str(info["path"]), "states_rev": info["obs_rev"],
                                 "result": res}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
