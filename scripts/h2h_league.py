#!/usr/bin/env python
"""Round-robin head to head of N checkpoints + one rating per checkpoint
(2026-09-26, the regression diagnosis).

Every pair plays the SAME deals (the table configs and each config's deals are
drawn from fixed seeds, so the whole league shares common random numbers), in
h2h_eval.py's duplicate format (seats swapped, all-in hands paid their runout
EV), sampled and/or argmax-vs-argmax. Ratings: the least-squares fit of
edge(A, B) ~ r_A - r_B (inverse-variance weighted, ratings sum to 0) in bb per
seat-hand, with a 95% interval from a bootstrap over table CONFIGS (the same
resample for every pair, which keeps their shared-deal correlation; 2026-09-28
ML-052). The RESIDUALS say how far the pairwise results are from one
consistent ranking: their RMS is compared with its null (the RMS a perfectly
transitive table shows under the same noise) -- p small = non-transitive play
(A > B > C > A), the signature of self-play cycling.

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

from plo5bp.evaluation import ObsRevMismatch, load_actor  # noqa: E402
from plo5bp.evaluation.league import bootstrap_league, fit_ratings  # noqa: E402,F401
from plo5bp.evaluation.tables import (  # noqa: E402
    BB,
    TIERS,
    play_duplicate,
    sample_table,
    tier_spec_version,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--deals", type=int, default=1024, help="deals per table config")
    ap.add_argument("--configs-per-tier", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="sampled,argmax")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ev-samples", type=int, default=64)
    ap.add_argument("--bootstrap", type=int, default=200, help="config resamples for the intervals")
    ap.add_argument("--out", default="runs/h2h_league.jsonl")
    args = ap.parse_args()

    device = torch.device(args.device)
    models, metas = [], []
    for p in args.ckpts:
        try:
            m, meta = load_actor(p, device, ema=False)
        except ObsRevMismatch as e:
            sys.exit(str(e))
        models.append(m)
        metas.append(meta)
    if len({m["obs_mode"] for m in metas}) != 1:
        sys.exit("all checkpoints must share one obs_mode")
    obs_mode = metas[0]["obs_mode"]
    names = [Path(p).stem for p in args.ckpts]
    crng = np.random.default_rng(args.seed)
    configs = [
        (tier, sample_table(tier, crng, metas[0]["variant"]))
        for tier in TIERS for _ in range(args.configs_per_tier)
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tag = tier_spec_version()
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        greedy = mode == "argmax"
        per_config: dict[tuple[int, int], list[np.ndarray]] = {}
        t0 = time.time()
        for i, j in itertools.combinations(range(len(models)), 2):
            torch.manual_seed(args.seed)
            per = []
            for k, (_tier, cfg) in enumerate(configs):
                rng = np.random.default_rng((args.seed, k))  # the SAME deals for every pair
                pair_net, n_seats, _ = play_duplicate(
                    cfg, (models[i], models[j]), args.deals, device, rng,
                    args.ev_samples, greedy=(greedy, greedy), obs_mode=obs_mode,
                )
                per.append(pair_net / BB / n_seats)
            per_config[(i, j)] = per
            x = np.concatenate(per)
            e, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "mode": mode, "a": names[i], "b": names[j], "edge_bb": e, "se": se,
                    "pairs": int(x.size), "deals": args.deals,
                    "configs_per_tier": args.configs_per_tier, "seed": args.seed,
                    "config_means": [float(c.mean()) if c.size else None for c in per],
                    "tier_spec": tag,
                }) + "\n")
            print(f"[{mode}] {names[i]:>12} vs {names[j]:<12} {e:+.3f} +- {se:.3f}   ({time.time() - t0:.0f}s)", flush=True)
        res = bootstrap_league(len(models), per_config, samples=args.bootstrap, seed=args.seed)
        r = res["ratings"]
        order = np.argsort(-r)
        print(f"\n== {mode}: ratings (bb/seat-hand, sum 0) with 95% intervals over configs ==")
        for k in order:
            print(f"   {names[k]:>14} {r[k]:+.3f}  [{res['ci_low'][k]:+.3f}, {res['ci_high'][k]:+.3f}]")
        mean_se = float(np.mean([se for _e, se in res["edges"].values()]))
        print(f"   residual RMS {res['residual_rms']:.3f} vs a transitive table's "
              f"{res['null_rms_median']:.3f} (95% {res['null_rms_high']:.3f}); "
              f"p(non-transitive by chance) {res['p_nontransitive']:.2f}; mean pair se {mean_se:.3f}")
        worst = sorted(res["residuals"].items(), key=lambda kv: -abs(kv[1]))[:5]
        for (i, j), v in worst:
            print(f"     {names[i]} vs {names[j]}: observed {res['edges'][(i, j)][0]:+.3f}, "
                  f"ratings say {r[i] - r[j]:+.3f} ({v:+.3f})")
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "mode": mode, "summary": True, "names": names,
                "ratings": [float(x) for x in r],
                "ci_low": [float(x) for x in res["ci_low"]],
                "ci_high": [float(x) for x in res["ci_high"]],
                "residual_rms": res["residual_rms"], "null_rms_median": res["null_rms_median"],
                "null_rms_high": res["null_rms_high"], "p_nontransitive": res["p_nontransitive"],
                "tier_spec": tag,
            }) + "\n")
        print(flush=True)


if __name__ == "__main__":
    main()
