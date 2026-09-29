#!/usr/bin/env python
"""Head-to-head strength of two checkpoints -- the network-size sweep's
strength test (2026-09-23).

Duplicate format, batched: every deal is played TWICE on the same cards and
button with the seats swapped -- each seat belongs to A in exactly one of
the two passes -- and every hand that ends all-in before the river pays its
64-runout Monte-Carlo EV, so neither card luck nor runout luck is left in the
comparison, only the two policies (each sampling its own mixed strategy, as
in training). Table configs come from the same tiers train.py mixes
(plo5bp.train.tiers: seats 2-6, clubgg / clubgg_deep / deep stacks).

Reported: A's edge in bb per seat-hand (both passes, A's seats summed,
divided by the seats played) with two standard errors -- `se` over deal
pairs (treats the table configs as fixed) and `se_config` over the per-config
means (includes config-to-config variance: the honest one; if it is much
larger than `se`, play more configs with fewer deals, e.g.
`--configs-per-tier 30 --deals 1024`) -- and the same per tier. Positive = A
stronger. Both checkpoints must read the same observation layout (obs_mode)
and this process's PLO5BP_OBS_REV must match their obs_rev (refused
otherwise -- scripts/h2h_cross.py compares different layouts/revisions). A
series of checkpoints compared with ONE --seed plays the same tables and deals
(paired comparisons).

    .venv/bin/python scripts/h2h_eval.py A.pt B.pt [--deals 4096]
        [--configs-per-tier 10] [--device cuda] [--seed 0] [--ema]

Library: plo5bp.evaluation (loader + tables).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PLO5_RUST_ENCODER", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plo5bp.evaluation import ObsRevMismatch, load_actor  # noqa: E402
from plo5bp.evaluation.tables import (  # noqa: E402,F401  (TIERS: re-export)
    BB,
    TIERS,
    play_duplicate,
    sample_table,
    summarize,
    tier_spec_version,
)


def print_report(report: dict, tag_a: str, tag_b: str) -> None:
    print(f"A = {tag_a}\nB = {tag_b}")
    for tier, r in report["tiers"].items():
        print(f"  {tier:12s} A edge {r['edge_bb']:+.4f} bb/seat-hand  (se {r['se']:.4f}, "
              f"config se {r['se_config']:.4f}, {r['pairs']} pairs)")
    print(f"  {'ALL':12s} A edge {report['edge_bb']:+.4f} bb/seat-hand  (se {report['se']:.4f}, "
          f"config se {report['se_config']:.4f}, "
          f"z {report['edge_bb'] / max(report['se'], 1e-12):+.2f}, {report['pairs']} pairs, "
          f"{report['seconds']}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--deals", type=int, default=4096, help="deals per table config")
    ap.add_argument("--configs-per-tier", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ema", action="store_true", help="play the EMA actors")
    ap.add_argument("--ev-samples", type=int, default=64)
    ap.add_argument("--out", default="runs/h2h_history.jsonl")
    ap.add_argument(
        "--greedy-a", action="store_true",
        help="A plays its most likely action (argmax gate / sizing mode) instead of "
        "sampling: what A has LEARNED to prefer, apart from how much it still mixes "
        "(compare runs trained at different entropy coefficients this way)",
    )
    ap.add_argument(
        "--greedy-b", action="store_true",
        help="B plays its most likely action too. With --greedy-a: argmax vs argmax, "
        "which keeps separating two runs after A's argmax has saturated against a "
        "sampling B (a sampling opponent is much weaker than its own argmax)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    try:
        model_a, meta_a = load_actor(args.a, device, args.ema)
        model_b, meta_b = load_actor(args.b, device, args.ema)
    except ObsRevMismatch as e:
        sys.exit(str(e))
    if meta_a["obs_mode"] != meta_b["obs_mode"]:
        sys.exit(f"obs_mode differs: {meta_a['obs_mode']} vs {meta_b['obs_mode']} "
                 "(use scripts/h2h_cross.py)")
    obs_mode = meta_a["obs_mode"]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    per_tier: dict[str, list[np.ndarray]] = {t: [] for t in TIERS}
    for tier in TIERS:
        for _ in range(args.configs_per_tier):
            cfg = sample_table(tier, rng, meta_a["variant"])
            pair_net, n_seats, _steps = play_duplicate(
                cfg, (model_a, model_b), args.deals, device, rng, args.ev_samples,
                greedy=(bool(args.greedy_a), bool(args.greedy_b)), obs_mode=obs_mode,
            )
            per_tier[tier].append(pair_net / BB / n_seats)   # bb per A seat-hand
    report = {"a": meta_a, "b": meta_b, "deals_per_config": args.deals,
              "configs_per_tier": args.configs_per_tier, "seed": args.seed,
              "ema": bool(args.ema), "greedy_a": bool(args.greedy_a),
              "greedy_b": bool(args.greedy_b), "tier_spec": tier_spec_version()}
    report.update(summarize(per_tier))
    report["seconds"] = round(time.time() - t0, 1)
    tag = lambda m: f"{Path(m['path']).name} ({m['hidden_dim']}x{m['num_layers']}, u{m['update']})"
    print_report(report, tag(meta_a), tag(meta_b))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
