#!/usr/bin/env python
"""Progress + health watch for a long training run (2026-09-24; any stem since
2026-09-28).

Runs next to the trainer on the pod. Every `--every` updates it takes the
run's numbered checkpoint `<stem>_<N>.pt` and appends ONE line to
`runs/<stem>_watch.log`:

  - strength: argmax vs argmax (`h2h_cross.py --greedy-a --greedy-b`, the
    tuning's learning measure; any layouts / obs revisions) against each fixed
    reference, in bb per seat-hand, + = the run is ahead. Every checkpoint
    plays the SAME tables and deals (one seed: paired points). They should climb.
  - collapse / sizing (plo5bp.evaluation.sharpness on the run's own cached
    self-play states): gate entropy, `rare<.1%` = share of decisions with a
    nearly dropped action, raise-size entropy, the most likely size's
    probability, and the min-raise / pot shares.
  - the trainer's latest numbers from `runs/<stem>.metrics.jsonl` (critic
    loss v, policy entropy H, approx KL, critic EV) -- the log line's regex
    only for runs older than the metrics file.

The references and states live in an optional per-stem file
`runs/<stem>.watch.json`, e.g.

    {"refs": ["checkpoints/vSix6_1390.pt", "checkpoints/vSix5_1248.pt"],
     "states_from": "checkpoints/vSix6_1390.pt"}

(`--ref` / `--states-from` / `--cache` override it; the cache defaults to
`runs/<stem>.sharpness_states.npz`, built from `states_from` -- default the
first checkpoint watched).

    setsid nohup .venv/bin/python -u scripts/run_watch.py --stem vSix6 \\
        > /dev/null 2>&1 < /dev/null &

Warning signs: the strength numbers falling for several lines in a row,
`rare<.1%` climbing well past ~25%, or gate entropy sliding toward 0.
`--once` processes what exists and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Tolerant of fields added between H= and kl= (it used to require exactly one).
_UPDATE_LINE = re.compile(r"update\s+(\d+)\s+pi=\S+\s+v=(\S+)\s.*?\bH=(\S+)\s.*?\bkl=(\S+)")


def _numbered(stem: str) -> dict[int, Path]:
    out = {}
    for p in (REPO / "checkpoints").glob(f"{stem}_*.pt"):
        tail = p.stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            out[int(tail)] = p
    return out


def _h2h(cand: Path, ref: Path, deals: int, seed: int, out: Path, device: str) -> tuple[float, float] | None:
    cmd = [sys.executable, "scripts/h2h_cross.py", str(cand), str(ref), "--deals", str(deals),
           "--device", device, "--seed", str(seed), "--greedy-a", "--greedy-b", "--out", str(out)]
    before = out.read_text().count("\n") if out.exists() else 0
    env = dict(os.environ, PLO5_RUST_ENCODER=os.environ.get("PLO5_RUST_ENCODER", "1"))
    rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL).returncode
    if rc != 0 or not out.exists():
        return None
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    if len(lines) <= before:
        return None
    rec = json.loads(lines[-1])
    return float(rec["edge_bb"]), float(rec["se"])


def trainer_now(stem: str) -> str:
    """The trainer's latest numbers: the metrics JSONL, else the log line."""
    metrics = REPO / "runs" / f"{stem}.metrics.jsonl"
    if metrics.exists():
        from plo5bp.train.metrics import read_metrics

        recs = read_metrics(metrics)
        if recs:
            r = recs[-1]
            ppo = r.get("ppo") or {}
            ev = ((r.get("value_health") or {}).get("all") or {}).get("ev")

            def f(x, fmt):
                return "-" if x is None else fmt % x

            return (f"trainer u{r.get('update')} v={f(ppo.get('value_loss'), '%.4f')} "
                    f"H={f(ppo.get('entropy'), '%.3f')} kl={f(ppo.get('approx_kl'), '%+.4f')} "
                    f"EV={f(ev, '%+.3f')}")
    log = REPO / "runs" / f"{stem}.log"
    if not log.exists():
        return "trainer: no metrics / log"
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


def watch_config(stem: str, args) -> dict:
    path = REPO / "runs" / f"{stem}.watch.json"
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    refs = args.ref or cfg.get("refs") or []
    if not refs:
        sys.exit(f"no references: pass --ref (repeatable) or write {path} "
                 '({"refs": ["checkpoints/<ref>.pt", ...]})')
    return {
        "refs": [Path(r) for r in refs],
        "states_from": args.states_from or cfg.get("states_from"),
        "cache": args.cache or cfg.get("cache") or f"runs/{stem}.sharpness_states.npz",
        "cache_rev": args.cache_rev if args.cache_rev is not None else cfg.get("cache_rev"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--ref", action="append", default=None,
                    help="fixed reference checkpoint (repeatable; else runs/<stem>.watch.json)")
    ap.add_argument("--states-from", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--cache-rev", type=int, default=None,
                    help="the obs revision of a states cache written before 2026-09-28")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--from-update", type=int, default=0)
    ap.add_argument("--deals", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7000, help="the h2h seed of every checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--poll", type=float, default=600.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    wc = watch_config(args.stem, args)
    for r in wc["refs"]:
        if not (REPO / r).exists():
            sys.exit(f"reference not found: {r}")
    watch_log = REPO / "runs" / f"{args.stem}_watch.log"
    done_path = REPO / "runs" / f"{args.stem}_watch_done.json"
    h2h_out = REPO / "runs" / f"{args.stem}_watch_h2h.jsonl"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()

    import torch

    from plo5bp.evaluation import load_actor
    from plo5bp.evaluation.sharpness import check_readable, ensure_states, measure

    torch.set_num_threads(int(args.threads))
    states = info = None

    while True:
        todo = sorted(n for n in _numbered(args.stem)
                      if n >= args.from_update and n % args.every == 0 and n not in done)
        for n in todo:
            cand = _numbered(args.stem)[n]
            if states is None:
                states, info = ensure_states(
                    REPO / wc["cache"], str(wc["states_from"] or cand),
                    assume_rev=wc["cache_rev"],
                )
            parts = [f"u{n}"]
            for r in wc["refs"]:
                res = _h2h(cand, REPO / r, args.deals, args.seed, h2h_out, args.device)
                parts.append(f"vs {Path(r).stem} " + ("failed" if res is None else f"{res[0]:+.3f}+-{res[1]:.3f}"))
            actor, meta = load_actor(str(cand), check_rev=False)
            b = measure(actor, *states, adapt=check_readable(info, meta, actor))["ALL"]
            am = b["anchor_mean"] or [float("nan")] * 11
            parts.append(
                f"gateH {b['gate_h']:.3f} rare<.1% {100 * b['rare_gate_1e3']:.1f}% "
                f"sizeH {b['anchor_h']:.3f} top {b['anchor_top']:.2f} "
                f"min {100 * am[0]:.0f}% pot {100 * am[-1]:.0f}%"
            )
            parts.append(trainer_now(args.stem))
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
