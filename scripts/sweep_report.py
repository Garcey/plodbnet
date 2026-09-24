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
    args = ap.parse_args()

    h2h_path = REPO / args.h2h
    rows = [json.loads(line) for line in h2h_path.read_text().splitlines() if line.strip()] \
        if h2h_path.exists() else []
    table: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for r in rows:
        a, b = r["a"], r["b"]
        if a.get("update") != b.get("update"):
            key = (f"{_stem(a['path'])}@{a.get('update')}", f"{_stem(b['path'])}@{b.get('update')}")
            table[key][-1] = r
            continue
        table[(_stem(a["path"]), _stem(b["path"]))][int(a["update"])] = r

    print("== strength: candidate vs reference at equal updates (bb/seat-hand, + = candidate stronger)")
    for (cand, ref), by_k in sorted(table.items()):
        cells = []
        for k in sorted(by_k):
            r = by_k[k]
            z = r["edge_bb"] / r["se"] if r["se"] > 0 else 0.0
            label = f"u{k}" if k >= 0 else "  "
            cells.append(f"{label} {r['edge_bb']:+.3f}+-{r['se']:.3f} (z{z:+.1f})")
        print(f"  {cand:>10} vs {ref:<10} " + "   ".join(cells))

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
