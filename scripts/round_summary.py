#!/usr/bin/env python
"""Summarize a recipe-search round (2026-09-26): per run, the mean edge over
its checkpoints from --from-update on (sampled and argmax rows of an
h2h_cross.py jsonl), overall and per tier, with the spread across checkpoints.
Single checkpoints swing +-0.1-0.2 bb/seat-hand, so rounds are judged on these
means, never one file.

    .venv/Scripts/python scripts/round_summary.py runs/diag/round_eval.jsonl \
        --stems r1e06,r2a,r2b,r2c,r2d --from-update 1293
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jsonl")
    ap.add_argument("--stems", default="")
    ap.add_argument("--from-update", type=int, default=0)
    args = ap.parse_args()
    want = [s for s in args.stems.split(",") if s]
    rows = [json.loads(line) for line in open(args.jsonl, encoding="utf-8") if line.strip()]
    latest = {}
    for r in rows:  # the last result per (checkpoint, mode) wins
        name = os.path.basename(r["a"]["path"]).replace(".pt", "")
        mode = ("argmax" if r.get("greedy_b") else "top-vs-sampled") if r.get("greedy_a") else "sampled"
        latest[(name, mode)] = r
    acc = defaultdict(lambda: defaultdict(list))
    for (name, mode), r in latest.items():
        if "_" not in name:
            continue
        stem, u = name.rsplit("_", 1)
        if not u.isdigit() or int(u) < args.from_update or (want and stem not in want):
            continue
        acc[(stem, mode)]["ALL"].append(r["edge_bb"])
        for t, v in (r.get("tiers") or {}).items():
            acc[(stem, mode)][t].append(v["edge_bb"])
    order = want or sorted({s for s, _ in acc})
    for mode in ("sampled", "top-vs-sampled", "argmax"):
        print(f"== {mode}")
        for stem in order:
            d = acc.get((stem, mode))
            if not d:
                continue
            a = np.asarray(d["ALL"])
            tiers = "  ".join(f"{t} {np.mean(v):+.3f}" for t, v in d.items() if t != "ALL")
            print(f"   {stem:>8}  mean {a.mean():+.3f}  (n {a.size}, min {a.min():+.3f}, max {a.max():+.3f})   {tiers}")


if __name__ == "__main__":
    main()
