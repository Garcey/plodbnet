"""Recipe-round summaries over an h2h jsonl (scripts/round_summary.py).

Rows are keyed by (candidate, mode, OPPONENT) (2026-09-28, ML-021): the same
checkpoint scored against two references used to collapse into one key, the
last line silently winning and a round's means mixing references.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np


def _stem_name(path: str) -> str:
    return os.path.basename(path).replace(".pt", "")


def row_mode(r: dict) -> str:
    if r.get("greedy_a"):
        return "argmax" if r.get("greedy_b") else "top-vs-sampled"
    return "sampled"


def latest_results(rows: "list[dict]", ref: "str | None" = None) -> dict:
    """{(candidate, mode, opponent): the LAST row} -- `ref` keeps only rows
    whose opponent file name contains it."""
    latest = {}
    for r in rows:
        if "a" not in r or "b" not in r:
            continue
        opp = _stem_name(r["b"]["path"])
        if ref and ref not in opp:
            continue
        latest[(_stem_name(r["a"]["path"]), row_mode(r), opp)] = r
    return latest


def round_means(latest: dict, stems: "list[str]", from_update: int) -> dict:
    """{(stem, mode): {"ALL": [...], tier: [...]}} over the numbered
    checkpoints `<stem>_<N>` with N >= from_update. Raises ValueError when a
    stem's rows were scored against more than one opponent."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    opponents: dict = defaultdict(set)
    for (name, mode, opp), r in latest.items():
        if "_" not in name:
            continue
        stem, u = name.rsplit("_", 1)
        if not u.isdigit() or int(u) < from_update or (stems and stem not in stems):
            continue
        opponents[stem].add(opp)
        acc[(stem, mode)]["ALL"].append(r["edge_bb"])
        for t, v in (r.get("tiers") or {}).items():
            acc[(stem, mode)][t].append(v["edge_bb"])
    mixed = {s: sorted(o) for s, o in opponents.items() if len(o) > 1}
    if mixed:
        raise ValueError(
            "rows score the same run against different references "
            f"{mixed} -- pass --ref to pick one"
        )
    return {k: {t: np.asarray(v) for t, v in d.items()} for k, d in acc.items()}
