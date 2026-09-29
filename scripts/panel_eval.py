#!/usr/bin/env python
"""Rate checkpoints against a FIXED reference panel -- the standing robustness
track (2026-09-28, ML-039; library: python/plo5bp/evaluation/panel.py).

One or two references cannot tell learning from cycling; a panel of 6-8 --
eras, lineages and simple baseline policies -- can. The panel's round robin is
played once and cached; each checkpoint then plays every member on the same
table configs and deals and gets one rating on the panel's scale (the panel's
mean = 0) with a 95% interval over table configs, plus how non-transitive its
results are (its residual RMS against the panel, and the league's p-value).

    # rate given checkpoints
    PLO5BP_OBS_REV=1 .venv/bin/python scripts/panel_eval.py \\
        --panel scripts/panels/plo5_full_rev1.json checkpoints/vSix6_1400.pt
    # every 20th numbered checkpoint of a stem not yet rated (a watcher can run
    # this on a timer)
    PLO5BP_OBS_REV=1 .venv/bin/python scripts/panel_eval.py \\
        --panel scripts/panels/plo5_full_rev1.json --stem checkpoints/vSix6 --every 20

Output: one JSON line per (checkpoint, mode) in runs/<panel>.panel.jsonl (the
cached round robin: runs/<panel>.panel.cache.json, rebuilt when the panel spec,
a member file or the table sampler changes). Panels: scripts/panels/*.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("PLO5_RUST_ENCODER", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from plo5bp.evaluation import ObsRevMismatch  # noqa: E402
from plo5bp.evaluation.panel import Panel, load_member, rate_candidate  # noqa: E402
from plo5bp.selfplay import discover_checkpoint_family  # noqa: E402


def _rated(out: Path, panel_key: str) -> set[str]:
    done: set[str] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("panel_key") == panel_key:
                done.add(str(Path(rec.get("path", "")).resolve()))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpts", nargs="*", help="checkpoints to rate")
    ap.add_argument("--panel", type=Path, required=True, help="panel spec (JSON)")
    ap.add_argument("--stem", default=None,
                    help="rate this stem's numbered checkpoints (<stem>_<N>.pt)")
    ap.add_argument("--every", type=int, default=20,
                    help="with --stem: only updates N divisible by this")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSONL output (default runs/<panel name>.panel.jsonl)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="round-robin cache (default runs/<panel name>.panel.cache.json)")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    panel = Panel.load(args.panel)
    out = args.out or Path("runs") / f"{panel.name}.panel.jsonl"
    cache = args.cache or Path("runs") / f"{panel.name}.panel.cache.json"
    device = torch.device(args.device)
    key = panel.key()
    todo = [str(p) for p in args.ckpts]
    if args.stem:
        _, family = discover_checkpoint_family(Path(f"{args.stem}.pt"))
        done = _rated(out, key)
        todo += [
            str(p) for n, p in sorted(family.items())
            if n % max(1, args.every) == 0 and str(Path(p).resolve()) not in done
        ]
    if not todo:
        print("[panel] nothing to rate")
        return 0
    try:
        members = [load_member(p, device) for _n, p in panel.members]
    except ObsRevMismatch as e:
        sys.exit(str(e))
    out.parent.mkdir(parents=True, exist_ok=True)
    for path in todo:
        try:
            rec = rate_candidate(panel, path, device, cache, args.bootstrap, _members=members)
        except ObsRevMismatch as e:
            print(f"[panel] skip {path}: {e}")
            continue
        with open(out, "a", encoding="utf-8") as fh:
            for mode, r in rec["modes"].items():
                fh.write(json.dumps({k: v for k, v in rec.items() if k != "modes"}
                                    | {"mode": mode, **r}) + "\n")
                print(
                    f"{rec['candidate']:>16} [{mode}] rating {r['rating']:+.3f} "
                    f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  "
                    f"non-transitivity: resid {r['candidate_residual_rms']:.3f}, "
                    f"league p {r['p_nontransitive']:.2f}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
