#!/usr/bin/env python
"""Throughput / memory probe for the rollout-length and num_envs decisions
(2026-09-23). Runs short warm-started trainings of the vMin2 recipe at each
grid point -- a scratch checkpoint path, never a real stem -- and reports the
steady-state seconds per update (updates after the first, which pays
compilation and first-touch costs), rows per second, and peak GPU memory.

    .venv/bin/python scripts/throughput_probe.py --warm checkpoints/sw64_59.pt \\
        --hidden-dim 64 --num-layers 3 --critic-hidden-dim 64 --critic-num-blocks 2 \\
        --grid "num_envs=110000,220000,440000" --rollout-length 44000000 --updates 3

`--grid` is "name=v1,v2,..." over num_envs, rollout_length or
micro_batch_rows (one axis per probe); the other two come from the flags.
A point that fails (e.g. CUDA out of memory) is reported, not fatal. Results
append to runs/throughput_probe.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_PHASE = re.compile(r"\[phase\] update=(\d+)\s+rollout=([\d.]+)s\s+optimize=([\d.]+)s\s+total=([\d.]+)s")
_VRAM = re.compile(r"\[vram\] peak alloc=([\d.]+) GiB\s+reserved=([\d.]+) GiB\s+\(rollout=([\d,]+) rows\)")


def run_point(args, num_envs: int, rollout: int, micro: int, tag: str) -> dict:
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    ckpt = scratch / f"probe_{tag}.pt"
    cmd = [
        sys.executable, "-u", "scripts/train.py",
        "--variant", "plo5_double_bomb", "--v6", "--obs-mode", "minimal",
        "--hidden-dim", str(args.hidden_dim), "--num-layers", str(args.num_layers),
        "--critic-hidden-dim", str(args.critic_hidden_dim),
        "--critic-num-blocks", str(args.critic_num_blocks),
        "--batched", "--device", "cuda",
        "--num-envs", str(num_envs), "--rollout-length", str(rollout),
        "--num-minibatches", str(args.num_minibatches), "--ppo-epochs", "2",
        "--mix-configs", "--configs-per-tier", "10",
        "--mix-tiers", "clubgg,clubgg_deep,deep",
        "--entropy-coef", "0.25", "--sizing-entropy-scale", "1.0",
        "--lr", "1.5e-4", "--lr-warmup-updates", "0", "--clip-room-mid", "0.07",
        "--target-kl", "0.5", "--kl-hard", "10.0", "--adv-clip", "8",
        "--cpu-threads", "24", "--snapshot-every", "5", "--checkpoint-every", "0",
        "--micro-batch-rows", str(micro),
        "--checkpoint", str(ckpt), "--num-updates", str(args.updates),
    ]
    if args.warm:
        cmd += ["--load-checkpoint", args.warm]
    if args.gpu_lock:
        cmd += ["--gpu-lock", args.gpu_lock]
    env = dict(
        os.environ,
        PLO5_RUST_ENCODER="1",
        PLO5BP_STEP_TIMERS="0",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        NUMPY_MADVISE_HUGEPAGE="0",
        MALLOC_MMAP_THRESHOLD_="33554432",
        MALLOC_TRIM_THRESHOLD_="17179869184",
        MALLOC_TOP_PAD_="67108864",
    )
    log_path = scratch / f"probe_{tag}.log"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    phases = [tuple(float(x) for x in m.groups()) for m in _PHASE.finditer(log)]
    vrams = [(float(a), float(b), int(c.replace(",", ""))) for a, b, c in _VRAM.findall(log)]
    res = {
        "tag": tag, "num_envs": num_envs, "rollout_length": rollout,
        "micro_batch_rows": micro, "rc": rc, "wall_s": round(time.time() - t0, 1),
        "hidden_dim": args.hidden_dim, "critic_hidden_dim": args.critic_hidden_dim,
        "log": str(log_path),
    }
    if "out of memory" in log.lower():
        res["oom"] = True
    steady = [p for p in phases if p[0] >= 1]
    if steady:
        res["total_s"] = statistics.median(p[3] for p in steady)
        res["rollout_s"] = statistics.median(p[1] for p in steady)
        res["optimize_s"] = statistics.median(p[2] for p in steady)
    if vrams:
        res["peak_alloc_gib"] = max(v[0] for v in vrams)
        res["peak_reserved_gib"] = max(v[1] for v in vrams)
        res["rows"] = statistics.median(v[2] for v in vrams)
        if "total_s" in res:
            res["rows_per_s"] = round(res["rows"] / res["total_s"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", default="", help="checkpoint to warm-start from")
    ap.add_argument("--hidden-dim", type=int, required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--critic-hidden-dim", type=int, required=True)
    ap.add_argument("--critic-num-blocks", type=int, required=True)
    ap.add_argument("--num-envs", type=int, default=220000)
    ap.add_argument("--rollout-length", type=int, default=44000000)
    ap.add_argument("--micro-batch-rows", type=int, default=0)
    ap.add_argument("--num-minibatches", type=int, default=16)
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--grid", required=True, help='e.g. "num_envs=110000,220000"')
    ap.add_argument("--scratch", default="/root/probe_scratch")
    ap.add_argument("--gpu-lock", default="")
    ap.add_argument("--out", default="runs/throughput_probe.jsonl")
    args = ap.parse_args()

    axis, values = args.grid.split("=", 1)
    axis = axis.strip().replace("-", "_")
    if axis not in ("num_envs", "rollout_length", "micro_batch_rows"):
        sys.exit(f"unknown grid axis {axis!r}")
    rows = []
    for v in (int(x) for x in values.split(",") if x.strip()):
        point = {
            "num_envs": args.num_envs,
            "rollout_length": args.rollout_length,
            "micro_batch_rows": args.micro_batch_rows,
        }
        point[axis] = v
        tag = f"{axis}_{v}"
        print(f"[probe] {tag} ...", flush=True)
        res = run_point(args, point["num_envs"], point["rollout_length"],
                        point["micro_batch_rows"], tag)
        rows.append(res)
        print(f"[probe] {json.dumps(res)}", flush=True)
        with open(REPO / args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res) + "\n")
    print("\n  {:>22} {:>9} {:>10} {:>11} {:>10} {:>10}".format(
        axis, "total_s", "rollout_s", "rows/s", "peak_GiB", "status"))
    for r in rows:
        status = "OOM" if r.get("oom") else ("ok" if r["rc"] == 0 else f"rc={r['rc']}")
        print("  {:>22} {:>9} {:>10} {:>11} {:>10} {:>10}".format(
            r[axis], r.get("total_s", "-"), r.get("rollout_s", "-"),
            r.get("rows_per_s", "-"), r.get("peak_alloc_gib", "-"), status))


if __name__ == "__main__":
    main()
