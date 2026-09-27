#!/usr/bin/env python
"""Round-robin head to head of N checkpoints + one rating per checkpoint
(2026-09-26, the regression diagnosis).

Every pair plays the SAME deals (the table configs and each config's deals are
drawn from fixed seeds, so the whole league shares common random numbers), in
h2h_eval.py's duplicate format (seats swapped, all-in hands paid their runout
EV), sampled and/or argmax-vs-argmax. Ratings: the least-squares fit of
edge(A, B) ~ r_A - r_B (inverse-variance weighted, ratings sum to 0) in bb per
seat-hand; the RESIDUALS say how far the pairwise results are from one
consistent ranking — large residuals = non-transitive play (A > B > C > A), the
signature of self-play cycling.

All checkpoints must read the same observation layout at this process's obs
rev (full-obs rev-1 lineages: run with PLO5BP_OBS_REV=1).

    PLO5BP_OBS_REV=1 .venv/bin/python scripts/h2h_league.py A.pt B.pt C.pt ... \\
        [--deals 1024] [--configs-per-tier 6] [--modes sampled,argmax]
"""

from __future__ import annotations

import os

os.environ.setdefault("PLO5_RUST_ENCODER", "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2h_eval import TIERS, _train_module, load_actor, play_config  # noqa: E402

from plo5bp import encoding as _encoding  # noqa: E402


def fit_ratings(names, results):
    """results: {(i, j): (edge, se)} with edge = A_i's edge over A_j. Returns
    (ratings, residuals {(i, j): edge - (r_i - r_j)})."""
    n = len(names)
    rows, y, w = [], [], []
    for (i, j), (e, se) in results.items():
        row = np.zeros(n)
        row[i], row[j] = 1.0, -1.0
        rows.append(row)
        y.append(e)
        w.append(1.0 / max(se, 1e-6) ** 2)
    rows.append(np.ones(n))  # sum-to-zero constraint (heavily weighted)
    y.append(0.0)
    w.append(1e6)
    a = np.asarray(rows) * np.sqrt(np.asarray(w))[:, None]
    b = np.asarray(y) * np.sqrt(np.asarray(w))
    r, *_ = np.linalg.lstsq(a, b, rcond=None)
    resid = {k: e - (r[k[0]] - r[k[1]]) for k, (e, _se) in results.items()}
    return r, resid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--deals", type=int, default=1024, help="deals per table config")
    ap.add_argument("--configs-per-tier", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="sampled,argmax")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ev-samples", type=int, default=64)
    ap.add_argument("--out", default="runs/h2h_league.jsonl")
    args = ap.parse_args()

    device = torch.device(args.device)
    models, metas = [], []
    for p in args.ckpts:
        m, meta = load_actor(p, device, ema=False)
        rev = meta["obs_rev"] if meta["obs_rev"] is not None else 1
        if int(rev) != int(_encoding.OBS_SEMANTICS_REV):
            sys.exit(f"{p}: obs rev {rev}, this process encodes rev {_encoding.OBS_SEMANTICS_REV}")
        models.append(m)
        metas.append(meta)
    if len({m["obs_mode"] for m in metas}) != 1:
        sys.exit("all checkpoints must share one obs_mode")
    obs_mode = metas[0]["obs_mode"]
    names = [Path(p).stem for p in args.ckpts]
    train = _train_module()
    crng = np.random.default_rng(args.seed)
    configs = []
    for tier in TIERS:
        for _ in range(args.configs_per_tier):
            cfg, _ = train._sample_game_config(
                (2, 3, 4, 5, 6), 1.0, 300.0, 10_000, 30_000, crng,
                stack_dist=tier, seats_dist="uniform", variant=metas[0]["variant"], sb=0,
            )
            configs.append((tier, cfg))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        greedy = mode == "argmax"
        results = {}
        t0 = time.time()
        for i, j in itertools.combinations(range(len(models)), 2):
            torch.manual_seed(args.seed)
            per = []
            for k, (tier, cfg) in enumerate(configs):
                rng = np.random.default_rng((args.seed, k))  # the SAME deals for every pair
                pair_net, n_seats, _ = play_config(
                    cfg, (models[i], models[j]), args.deals, obs_mode, device, rng,
                    args.ev_samples, greedy=(greedy, greedy),
                )
                per.append(pair_net / 10_000 / n_seats)
            x = np.concatenate(per)
            e, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))
            results[(i, j)] = (e, se)
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"mode": mode, "a": names[i], "b": names[j], "edge_bb": e, "se": se,
                                     "pairs": int(x.size), "deals": args.deals,
                                     "configs_per_tier": args.configs_per_tier, "seed": args.seed}) + "\n")
            print(f"[{mode}] {names[i]:>12} vs {names[j]:<12} {e:+.3f} +- {se:.3f}   ({time.time() - t0:.0f}s)", flush=True)
        r, resid = fit_ratings(names, results)
        order = np.argsort(-r)
        print(f"\n== {mode}: ratings (bb/seat-hand, sum 0) ==")
        for k in order:
            print(f"   {names[k]:>14} {r[k]:+.3f}")
        rms = float(np.sqrt(np.mean([v * v for v in resid.values()])))
        mean_se = float(np.mean([se for _e, se in results.values()]))
        worst = sorted(resid.items(), key=lambda kv: -abs(kv[1]))[:5]
        print(f"   residual RMS {rms:.3f} (mean pair se {mean_se:.3f}; RMS >> se = non-transitive)")
        for (i, j), v in worst:
            print(f"     {names[i]} vs {names[j]}: observed {results[(i, j)][0]:+.3f}, ratings say {r[i] - r[j]:+.3f} ({v:+.3f})")
        print(flush=True)


if __name__ == "__main__":
    main()
