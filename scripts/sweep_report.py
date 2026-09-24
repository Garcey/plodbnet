#!/usr/bin/env python
"""Network-size sweep: one table of every head-to-head and utilization probe
so far (runs/h2h_history.jsonl, runs/utilization_selfplay.jsonl).

    .venv/bin/python scripts/sweep_report.py

Strength rows: candidate vs reference at the same update count k, edge in
bb/seat-hand (+ = candidate stronger) with its standard error and z. Only
EVALUATION noise is in that se; run-to-run training noise shows up as the
spread between same-size runs (vMin2 vs the sw128 control).
Utilization rows (self-play states): per torso layer, dead units / width and
rank99 (dimensions holding 99% of the activation variance).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _stem(path: str) -> str:
    name = Path(path).name
    return name.rsplit("_", 1)[0] if "_" in name else name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2h", default="runs/h2h_history.jsonl")
    ap.add_argument("--util", default="runs/utilization_selfplay.jsonl")
    ap.add_argument("--window", default="30:39",
                    help="FROM:TO update range averaged per pair (every checkpoint in it)")
    args = ap.parse_args()
    w_lo, w_hi = (int(x) for x in args.window.split(":"))

    h2h_path = REPO / args.h2h
    rows = [json.loads(line) for line in h2h_path.read_text().splitlines() if line.strip()] \
        if h2h_path.exists() else []
    table: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for r in rows:
        a, b = r["a"], r["b"]
        if a.get("update") is None:
            continue
        # A fixed reference (e.g. a tuning wave's common warm start) is labelled
        # with its update; a greedy-A comparison gets its own row.
        ref = _stem(b["path"])
        if b.get("update") != a.get("update"):
            ref = f"{ref}@u{b.get('update')}"
        if r.get("greedy_a"):
            ref += " (A greedy)"
        table[(_stem(a["path"]), ref)][int(a["update"])] = r

    print("== strength: candidate vs reference at equal updates (bb/seat-hand, + = candidate stronger)")
    for (cand, ref), by_k in sorted(table.items()):
        cells = []
        for k in sorted(by_k):
            if w_lo <= k <= w_hi and k % 10 != 9:
                continue  # the dense window is summarised below
            r = by_k[k]
            z = r["edge_bb"] / r["se"] if r["se"] > 0 else 0.0
            label = f"u{k}" if k >= 0 else "  "
            cells.append(f"{label} {r['edge_bb']:+.3f}+-{r['se']:.3f} (z{z:+.1f})")
        print(f"  {cand:>10} vs {ref:<10} " + "   ".join(cells))

    print(f"\n== mean over every checkpoint u{w_lo}..u{w_hi} (se_eval from the deals; "
          "se_spread = std across checkpoints / sqrt(n), includes checkpoint noise)")
    for (cand, ref), by_k in sorted(table.items()):
        ks = [k for k in sorted(by_k) if w_lo <= k <= w_hi]
        if len(ks) < 3:
            continue
        edges = [by_k[k]["edge_bb"] for k in ks]
        n = len(edges)
        mean = sum(edges) / n
        se_eval = (sum(by_k[k]["se"] ** 2 for k in ks) ** 0.5) / n
        var = sum((e - mean) ** 2 for e in edges) / (n - 1)
        se_spread = (var / n) ** 0.5
        print(f"  {cand:>10} vs {ref:<10} n={n:2d}  mean {mean:+.3f}  se_eval {se_eval:.3f}"
              f"  se_spread {se_spread:.3f}  (z {mean / max(se_spread, se_eval, 1e-9):+.1f})")

    util_path = REPO / args.util
    if not util_path.exists():
        return
    urows = [json.loads(line) for line in util_path.read_text().splitlines() if line.strip()]
    print("\n== utilization on self-play states: torso layers dead/width, rank99")
    last: dict[tuple[str, int], dict] = {}
    for r in urows:
        last[(_stem(r["ckpt"]), int(r["update"]))] = r
    for (stem, k), r in sorted(last.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        parts = []
        for net in ("actor", "critic"):
            layers = [lay for lay in r[net]["layers"] if lay.get("dead_pct") is not None]
            cells = []
            for lay in layers:
                dead = round(lay["dead_pct"] * lay["width"] / 100.0)
                cells.append(f"{dead}/{lay['width']} r{lay['rank99']}")
            parts.append(f"{net} " + " ".join(cells))
        print(f"  u{k:<3} {stem:>8}  " + "   ".join(parts))


if __name__ == "__main__":
    main()
