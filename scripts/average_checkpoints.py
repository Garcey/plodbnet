#!/usr/bin/env python
"""Average the weights of several checkpoints of ONE run (stochastic weight
averaging; 2026-09-26, the regression diagnosis).

Neighbouring checkpoints of one run sit in the same basin, so their average is a
valid network. If the average plays better than the checkpoints it came from,
the run's update-to-update movement is mostly NOISE (a random walk around the
optimum) that averaging cancels.

    .venv/Scripts/python scripts/average_checkpoints.py OUT.pt A.pt B.pt ...

The actor ("model") and the critic are averaged tensor by tensor (float64, cast
back); everything else is copied from the LAST input. Shapes must match.
"""

from __future__ import annotations

import argparse
import sys

import torch


def average(paths: list[str]) -> dict:
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    out = dict(cks[-1])
    for key in ("model", "critic"):
        if not all(isinstance(c.get(key), dict) for c in cks):
            continue
        ref = cks[-1][key]
        avg = {}
        for name, t in ref.items():
            if not torch.is_tensor(t) or not t.is_floating_point():
                avg[name] = t
                continue
            acc = torch.zeros_like(t, dtype=torch.float64)
            for c in cks:
                v = c[key][name]
                if v.shape != t.shape:
                    sys.exit(f"{key}.{name}: shape {tuple(v.shape)} vs {tuple(t.shape)}")
                acc += v.to(torch.float64)
            avg[name] = (acc / len(cks)).to(t.dtype)
        out[key] = avg
    out["model_ema"] = None
    out["averaged_from"] = list(paths)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()
    torch.save(average(args.inputs), args.out)
    print(f"wrote {args.out} = mean of {len(args.inputs)} checkpoints")


if __name__ == "__main__":
    main()
