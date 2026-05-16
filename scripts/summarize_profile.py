"""Summarize a Chrome trace from `--profile-one-update`.

Reads `runs/profile_update0.json`, groups user-annotation spans by name,
and for each span sums durations of CUDA kernels whose timestamps fall
inside the span's [ts, ts+dur] interval. Prints a sorted table.

Usage:
    python scripts/summarize_profile.py runs/profile_update0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    args = ap.parse_args()

    if not args.trace.exists():
        print(f"trace not found: {args.trace}", file=sys.stderr)
        return 1

    data = json.loads(args.trace.read_text())
    events = data["traceEvents"] if isinstance(data, dict) else data

    user_spans = []
    kernels = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat == "user_annotation":
            user_spans.append((e["name"], e["ts"], e["dur"]))
        elif cat in ("kernel", "Kernel", "cuda_runtime"):
            if cat == "cuda_runtime":
                continue
            kernels.append((e["ts"], e["dur"]))

    kernels.sort()
    kernel_ts = [k[0] for k in kernels]

    import bisect

    grouped: dict[str, list[float]] = defaultdict(list)
    cuda_per_name: dict[str, float] = defaultdict(float)
    for name, ts, dur in user_spans:
        grouped[name].append(dur)
        lo = bisect.bisect_left(kernel_ts, ts)
        hi = bisect.bisect_right(kernel_ts, ts + dur)
        for i in range(lo, hi):
            cuda_per_name[name] += kernels[i][1]

    rows = []
    for name, durs in grouped.items():
        n = len(durs)
        cpu_total_ms = sum(durs) / 1000.0
        cpu_mean_ms = cpu_total_ms / n
        cuda_total_ms = cuda_per_name.get(name, 0.0) / 1000.0
        rows.append((name, n, cpu_total_ms, cuda_total_ms, cpu_mean_ms))

    rows.sort(key=lambda r: -r[2])

    print(f"{'step':<32} {'n_calls':>8} {'cpu_ms':>12} {'cuda_ms':>12} {'cpu_mean_ms':>12}")
    print("-" * 80)
    cpu_total = 0.0
    cuda_total = 0.0
    for name, n, cpu_ms, cuda_ms, cpu_mean in rows:
        print(f"{name:<32} {n:>8d} {cpu_ms:>12.1f} {cuda_ms:>12.1f} {cpu_mean:>12.3f}")
        cpu_total += cpu_ms
        cuda_total += cuda_ms
    print("-" * 80)
    print(f"{'TOTAL (annotated)':<32} {'':>8} {cpu_total:>12.1f} {cuda_total:>12.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
