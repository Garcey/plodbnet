#!/usr/bin/env python
"""Signal-to-noise of ONE training update (2026-09-26, the regression diagnosis).

Two runs start from the same checkpoint and optimizer state and take one update
each, differing only in `--seed` (other deals, table configs and pool draws).
Whatever the two updates share is the learning SIGNAL; whatever differs is
sampling NOISE. With delta_X = theta_X - theta_base:

  - parameter level: cos(delta_A, delta_B). If delta = s + n with independent
    noise of equal size in both runs, cos = |s|^2 / (|s|^2 + |n|^2), so the
    per-update SNR^2 (signal energy / noise energy) = cos / (1 - cos);
  - policy level, on one fixed set of decision states: the change of every legal
    gate log-probability in run A vs run B (correlation = the same ratio, seen
    through the policy), and KL(base->A), KL(base->B), KL(A->B) (noise-dominated
    updates: KL(A,B) ~ KL(base,A) + KL(base,B); signal-dominated: KL(A,B) << both).

    .venv/bin/python scripts/update_snr.py BASE.pt A.pt B.pt \
        [--states runs/snr_states.npz --rows 60000]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def _sharp():
    spec = importlib.util.spec_from_file_location("_sharp", REPO / "scripts" / "policy_sharpness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def param_deltas(base: dict, a: dict, b: dict, key: str):
    """Per-tensor and overall cosine of the two updates' parameter deltas."""
    rows = []
    dot = na = nb = 0.0
    for name, t0 in base[key].items():
        if not torch.is_tensor(t0) or not t0.is_floating_point():
            continue
        da = (a[key][name].double() - t0.double()).flatten()
        db = (b[key][name].double() - t0.double()).flatten()
        d, x, y = float(da @ db), float(da @ da), float(db @ db)
        dot, na, nb = dot + d, na + x, nb + y
        if x > 0 and y > 0:
            rows.append((name, d / np.sqrt(x * y), np.sqrt(x), np.sqrt(y), float(t0.double().norm())))
    cos = dot / np.sqrt(na * nb) if na > 0 and nb > 0 else float("nan")
    return cos, np.sqrt(na), np.sqrt(nb), rows


def gate_logp(actor, obs, masks, batch=8192):
    out = []
    with torch.inference_mode():
        for i in range(0, obs.shape[0], batch):
            o = torch.from_numpy(obs[i:i + batch]).float()
            m = torch.from_numpy(masks[i:i + batch]).bool()
            logits = actor(o, m)[0]
            out.append(torch.log_softmax(logits.float(), dim=-1))
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--states", default="runs/snr_states.npz")
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--out", default="runs/update_snr.jsonl")
    args = ap.parse_args()

    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in (args.base, args.a, args.b)]
    rec = {"base": args.base, "a": args.a, "b": args.b}
    for key in ("model", "critic"):
        if not all(isinstance(c.get(key), dict) for c in cks):
            continue
        cos, na, nb, rows = param_deltas(*cks, key)
        snr2 = cos / (1 - cos) if cos < 1 else float("inf")
        print(f"[{key}] cos(dA, dB) = {cos:+.4f}  |dA| {na:.4f}  |dB| {nb:.4f}  -> SNR^2 per update ~ {snr2:.3f}")
        rows.sort(key=lambda r: -r[2])
        for name, c, x, y, w in rows[:12]:
            print(f"     {name:<40} cos {c:+.3f}  |dA| {x:.4f}  |dB| {y:.4f}  |w| {w:.2f}")
        rec[key] = {"cos": cos, "norm_a": na, "norm_b": nb, "snr2": snr2,
                    "per_tensor": {r[0]: r[1] for r in rows}}

    sharp = _sharp()
    base_actor, obs_mode, _ = sharp._load_actor(args.base)
    st = Path(args.states)
    if st.exists():
        z = np.load(st)
        obs, masks = z["obs"], z["masks"]
    else:
        obs, masks, sizing, tiers = sharp.build_states(base_actor, args.rows, obs_mode)
        st.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(st, obs=obs, masks=masks, sizing=sizing, tiers=tiers)
    lp = [gate_logp(sharp._load_actor(p)[0], obs, masks) for p in (args.base, args.a, args.b)]
    legal = masks.astype(bool)
    p0, pa, pb = (np.exp(x) for x in lp)

    def kl(p, lq, lr):
        return float(np.where(legal, p * (lq - lr), 0.0).sum(-1).mean())

    da = np.where(legal, lp[1] - lp[0], 0.0)
    db = np.where(legal, lp[2] - lp[0], 0.0)
    # weight each state's legal actions by the base policy (what is played)
    w = np.where(legal, p0, 0.0)
    corr = float((w * da * db).sum() / np.sqrt((w * da * da).sum() * (w * db * db).sum()))
    k0a, k0b, kab = kl(p0, lp[0], lp[1]), kl(p0, lp[0], lp[2]), kl(pa, lp[1], lp[2])
    print(f"[policy] {obs.shape[0]} states: corr(dlogp_A, dlogp_B) = {corr:+.4f}  "
          f"KL(base,A) {k0a:.5f}  KL(base,B) {k0b:.5f}  KL(A,B) {kab:.5f}  "
          f"(noise-only: KL(A,B) ~ {k0a + k0b:.5f})")
    rec["policy"] = {"states": int(obs.shape[0]), "corr": corr, "kl_base_a": k0a,
                     "kl_base_b": k0b, "kl_a_b": kab}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("PLO5_RUST_ENCODER", "1")
    sys.exit(main())
