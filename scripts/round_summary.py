#!/usr/bin/env python
"""Summarize a recipe-search round (2026-09-26): per run, the mean edge over
its checkpoints from --from-update on (sampled and argmax rows of an
h2h_cross.py jsonl), overall and per tier, with the spread across checkpoints.
Single checkpoints swing +-0.1-0.2 bb/seat-hand, so rounds are judged on these
means, never one file.

Results are keyed by candidate, mode AND opponent: a file that scores the same
runs against two references needs `--ref <name>` (refused otherwise, instead
of silently mixing the two).

    .venv/Scripts/python scripts/round_summary.py runs/diag/round_eval.jsonl \
        --stems r1e06,r2a,r2b,r2c,r2d --from-update 1293 [--ref vSix6_1390]
"""

from __future__ import annotations

import argparse
import json
import sys

from plo5bp.evaluation.rounds import latest_results, round_means


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jsonl")
    ap.add_argument("--stems", default="")
    ap.add_argument("--from-update", type=int, default=0)
    ap.add_argument("--ref", default="", help="only rows whose opponent's name contains this")
    args = ap.parse_args()
    want = [s for s in args.stems.split(",") if s]
    rows = [json.loads(line) for line in open(args.jsonl, encoding="utf-8") if line.strip()]
    try:
        acc = round_means(latest_results(rows, args.ref or None), want, args.from_update)
    except ValueError as e:
        sys.exit(str(e))
    order = want or sorted({s for s, _ in acc})
    for mode in ("sampled", "top-vs-sampled", "argmax"):
        print(f"== {mode}")
        for stem in order:
            d = acc.get((stem, mode))
            if not d:
                continue
            a = d["ALL"]
            tiers = "  ".join(f"{t} {v.mean():+.3f}" for t, v in d.items() if t != "ALL")
            print(f"   {stem:>8}  mean {a.mean():+.3f}  (n {a.size}, min {a.min():+.3f}, max {a.max():+.3f})   {tiers}")


if __name__ == "__main__":
    main()
