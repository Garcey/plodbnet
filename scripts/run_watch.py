#!/usr/bin/env python
"""Progress + health watch for a long training run (2026-09-24, for vMin3).

Runs next to the trainer on the pod. Every `--every` updates it takes the
run's numbered checkpoint `<stem>_<N>.pt` and appends ONE line to
`runs/<stem>_watch.log`:

  - strength: argmax vs argmax (`h2h_eval.py --greedy-a --greedy-b`, the
    tuning's learning measure) against each fixed `--ref` -- by default the
    run's own starting point and vMin2 u150 (a separate lineage) -- in bb per
    seat-hand, + = the run is ahead. Both should climb over time.
  - collapse / sizing (`policy_sharpness.py` on its cached states): gate
    entropy, `rare<.1%` = share of decisions with a nearly dropped action,
    raise-size entropy, the most likely size's probability, and the min-raise
    / pot shares.
  - the trainer's latest logged numbers (critic loss v, entropy H, approx KL).

Warning signs: the strength numbers falling for several lines in a row,
`rare<.1%` climbing well past ~25% (0.07 settled at ~17% in tuning), or gate
entropy sliding toward 0.

    setsid nohup .venv/bin/python -u scripts/run_watch.py --stem vMin3 \\
        --ref checkpoints/t3ent07.pt --ref checkpoints/vMin2_150.pt \\
        > /dev/null 2>&1 < /dev/null &

It holds little memory (two small networks + the probe states) and one h2h at
a time, so it fits beside a 330M-row run. `--once` processes what exists and
exits (used to check it on finished runs).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_UPDATE_LINE = re.compile(
    r"update\s+(\d+)\s+pi=\S+\s+v=(\S+)\s+vd=\S+\s+H=(\S+)\s+\S+\s+kl=(\S+)"
)


def _load_sharpness():
    spec = importlib.util.spec_from_file_location(
        "_policy_sharpness", REPO / "scripts" / "policy_sharpness.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _numbered(stem: str) -> dict[int, Path]:
    out = {}
    for p in (REPO / "checkpoints").glob(f"{stem}_*.pt"):
        tail = p.stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            out[int(tail)] = p
    return out


def _h2h(cand: Path, ref: Path, deals: int, seed: int, out: Path, device: str) -> tuple[float, float] | None:
    cmd = [sys.executable, "scripts/h2h_eval.py", str(cand), str(ref), "--deals", str(deals),
           "--device", device, "--seed", str(seed), "--greedy-a", "--greedy-b", "--out", str(out)]
    before = out.read_text().count("\n") if out.exists() else 0
    env = dict(os.environ, PLO5_RUST_ENCODER=os.environ.get("PLO5_RUST_ENCODER", "1"))
    rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL).returncode
    if rc != 0 or not out.exists():
        return None
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    if len(lines) <= before:
        return None
    rec = json.loads(lines[-1])
    return float(rec["edge_bb"]), float(rec["se"])


def _trainer_now(stem: str) -> str:
    log = REPO / "runs" / f"{stem}.log"
    if not log.exists():
        return "trainer: no log"
    last = None
    with open(log, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 400_000))
        for line in fh.read().decode("utf-8", "replace").splitlines():
            m = _UPDATE_LINE.search(line)
            if m:
                last = m
    if last is None:
        return "trainer: no update yet"
    return f"trainer v={last.group(2)} H={last.group(3)} kl={last.group(4)}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", default="vMin3")
    ap.add_argument("--ref", action="append", default=None,
                    help="fixed reference checkpoint (repeatable); default: t3ent07.pt + vMin2_150.pt")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--from-update", type=int, default=0)
    ap.add_argument("--deals", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="runs/sharpness_states.npz")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--poll", type=float, default=600.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    refs = [Path(r) for r in (args.ref or ["checkpoints/t3ent07.pt", "checkpoints/vMin2_150.pt"])]
    for r in refs:
        if not (REPO / r).exists():
            sys.exit(f"reference not found: {r}")
    watch_log = REPO / "runs" / f"{args.stem}_watch.log"
    done_path = REPO / "runs" / f"{args.stem}_watch_done.json"
    h2h_out = REPO / "runs" / f"{args.stem}_watch_h2h.jsonl"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()

    import torch
    torch.set_num_threads(int(args.threads))
    sharp = _load_sharpness()
    z = np.load(REPO / args.cache)
    states = (z["obs"], z["masks"], z["sizing"], z["tiers"])

    while True:
        todo = sorted(n for n in _numbered(args.stem)
                      if n >= args.from_update and n % args.every == 0 and n not in done)
        for n in todo:
            cand = _numbered(args.stem)[n]
            parts = [f"u{n}"]
            for r in refs:
                res = _h2h(cand, REPO / r, args.deals, 7000 + n, h2h_out, args.device)
                label = Path(r).stem
                parts.append(f"vs {label} " + ("failed" if res is None else f"{res[0]:+.3f}+-{res[1]:.3f}"))
            actor, _mode, _u = sharp._load_actor(str(cand))
            b = sharp.measure(actor, *states)["ALL"]
            am = b["anchor_mean"] or [float("nan")] * 11
            parts.append(
                f"gateH {b['gate_h']:.3f} rare<.1% {100 * b['rare_gate_1e3']:.1f}% "
                f"sizeH {b['anchor_h']:.3f} top {b['anchor_top']:.2f} "
                f"min {100 * am[0]:.0f}% pot {100 * am[-1]:.0f}%"
            )
            parts.append(_trainer_now(args.stem))
            line = f"[{time.strftime('%m-%d %H:%M', time.gmtime())}] " + " | ".join(parts)
            print(line, flush=True)
            with open(watch_log, "a") as fh:
                fh.write(line + "\n")
            done.add(n)
            done_path.write_text(json.dumps(sorted(done)))
        if args.once:
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
