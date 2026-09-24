#!/usr/bin/env python
"""Network-size sweep evaluator (2026-09-23): every `--every` updates, compare
each sweep stem with the reference stem at the SAME number of updates.

For every checkpoint index N = every-1, 2*every-1, ... (i.e. after `every`,
2*every, ... updates) at which both `<ref>_N.pt` and `<stem>_N.pt` exist:
  - strength: scripts/h2h_eval.py <stem>_N.pt <ref>_N.pt  (-> runs/h2h_history.jsonl)
  - utilization: scripts/utilization_probe.py on each, fixed flop states
    (-> runs/utilization_history.jsonl) and self-play states
    (-> runs/utilization_selfplay.jsonl)
Finished (stem, N) pairs are remembered in runs/sweep_eval_done.json, so the
driver can be re-run or left looping (--loop SECONDS) while the runs train.
A second driver against another reference needs its own `--done` file (the
keys do not name the reference).

    .venv/bin/python scripts/sweep_eval.py --ref vMin2 --stems sw64,sw32 --loop 600
    .venv/bin/python scripts/sweep_eval.py --ref sw128 --stems sw64,sw32,vMin2         --done runs/sweep_eval_done_sw128.json --loop 600
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    print("[sweep-eval] $", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=REPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="vMin2")
    ap.add_argument("--stems", required=True, help="comma list, e.g. sw64,sw32")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--max-updates", type=int, default=1000)
    ap.add_argument("--deals", type=int, default=4096)
    ap.add_argument("--loop", type=float, default=0.0, help="re-check every N s (0 = once)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--done", default="runs/sweep_eval_done.json")
    args = ap.parse_args()
    stems = [s for s in args.stems.split(",") if s]

    def probe(ckpt: Path) -> int:
        # The original narrow probe (fixed flop nodes) AND the self-play one
        # (the states the policy actually meets) -- see utilization_probe.py.
        rc = run([py, "scripts/utilization_probe.py", str(ckpt)])
        return rc | run([py, "scripts/utilization_probe.py", str(ckpt),
                         "--states", "selfplay",
                         "--out", "runs/utilization_selfplay.jsonl"])
    done_path = REPO / args.done
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    py = sys.executable
    while True:
        worked = False
        for n in range(args.every - 1, args.max_updates, args.every):
            ref = REPO / "checkpoints" / f"{args.ref}_{n}.pt"
            if not ref.exists():
                continue
            for stem in stems:
                cand = REPO / "checkpoints" / f"{stem}_{n}.pt"
                key = f"{stem}@{n}"
                if key in done or not cand.exists():
                    continue
                rc = run([py, "scripts/h2h_eval.py", str(cand), str(ref),
                          "--deals", str(args.deals), "--device", args.device,
                          "--seed", str(n)])
                rc |= probe(cand)
                ref_key = f"{args.ref}@{n}"
                if ref_key not in done:
                    if probe(ref) == 0:
                        done.add(ref_key)
                if rc == 0:
                    done.add(key)
                    done_path.write_text(json.dumps(sorted(done)))
                worked = True
        if args.loop <= 0:
            break
        if not worked:
            time.sleep(args.loop)


if __name__ == "__main__":
    main()
