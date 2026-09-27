#!/usr/bin/env python
"""Offline critic experiments on a rollout dump (2026-09-26, the regression
diagnosis / critic redesign).

Input: the files `PLO5BP_DUMP_BATCH=<path> scripts/train.py ... --gae-lambda 1.0`
writes: <path> (npz: Monte-Carlo returns, the rollout critic's values, gate
actions, opponents' hole cards, ...) and <path>.obs16.npy (the observations,
float16). Rows are split into train / held-out by blocks of 2048 consecutive
rows (a block holds whole hands, so no hand is on both sides).

1. The CURRENT critic (a checkpoint's "critic") on the held-out rows, read two
   ways: V = symexp(E[symlog bins]) (what training uses) and the raw-space mean
   sum_i p_i symexp(c_i). Per street: bias, RMSE, explained variance.
2. Candidate critics trained on the train rows (HL-Gauss, the same 51-bin
   symlog support), compared on the held-out rows:
     scratch-relu      the current architecture from scratch
     scratch-relu+str  + every player's current made-hand strength per board
     scratch-silu+str  SiLU + LayerNorm'd input block + strengths
     finetune          the current critic, trained further on the same rows
     finetune+str      the current critic with the strength inputs added
                       (zero-initialised columns: starts as the same function)

    .venv/bin/python scripts/critic_offline.py /root/diag_batch_u1290.npz \
        --ckpt checkpoints/vSix5.pt [--epochs 3] [--only finetune,finetune+str]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plo5bp import _engine
from plo5bp.network import _symexp, _symlog, build_critic_from_state_dict, opp_holes_multihot

STREETS = {1: "flop", 2: "turn", 3: "river"}


# ----------------------------------------------------------------- features
def cards_from_multihot(mh: np.ndarray, width: int) -> np.ndarray:
    """(N, 52) 0/1 -> (N, width) card indices ascending, 255-padded."""
    n = mh.shape[0]
    out = np.full((n, width), 255, dtype=np.uint8)
    rows, cols = np.nonzero(mh > 0.5)  # row-major: ascending columns per row
    k = np.arange(rows.size) - np.searchsorted(rows, np.arange(n))[rows]
    keep = k < width
    out[rows[keep], k[keep]] = cols[keep]
    return out


def strength_features(obs: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """(N, 12) float32: made-hand strength / 7462 of hero then opponent slots
    1..5 (hero-rotated, zero where the slot is empty) on board A and B."""
    n = obs.shape[0]
    hero = cards_from_multihot(obs[:, 0:52], 5)
    ba = cards_from_multihot(obs[:, 52:104], 5)
    bb = cards_from_multihot(obs[:, 104:156], 5)
    la = (ba != 255).sum(1).astype(np.uint8)
    lb = (bb != 255).sum(1).astype(np.uint8)
    feats = np.zeros((n, 12), dtype=np.float32)
    seats = [hero] + [np.ascontiguousarray(opp[:, j, :]) for j in range(opp.shape[1])]
    for s, holes in enumerate(seats):
        for b, (board, blen) in enumerate(((ba, la), (bb, lb))):
            st = np.asarray(_engine.plo_board_strength_batch(holes, board, blen))
            feats[:, 2 * s + b] = st.astype(np.float32) / 7462.0
    return feats


# --------------------------------------------------------------- the critic
class Critic(nn.Module):
    """Input block Linear [-> LayerNorm] -> act, then `blocks` residual blocks
    x + act(Linear(LayerNorm(x))), then the 51-bin HL-Gauss value head over the
    same symlog support as CentralCritic."""

    def __init__(self, in_dim, hidden=1536, blocks=2, act="relu", in_norm=False,
                 bins=51, support=1500.0, sigma=0.75):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.in_norm = nn.LayerNorm(hidden) if in_norm else None
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(blocks)])
        self.lins = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(blocks)])
        self.act = {"relu": F.relu, "silu": F.silu, "gelu": F.gelu}[act]
        self.head = nn.Linear(hidden, bins)
        hi = float(_symlog(torch.tensor(float(support))))
        centers = torch.linspace(-hi, hi, bins)
        step = float(centers[1] - centers[0])
        self.register_buffer("centers", centers)
        self.register_buffer("edges", torch.cat([centers[:1] - step / 2, (centers[:-1] + centers[1:]) / 2,
                                                 centers[-1:] + step / 2]))
        self.sigma = sigma * step

    def logits(self, x):
        h = self.inp(x)
        if self.in_norm is not None:
            h = self.in_norm(h)
        h = self.act(h)
        for ln, lin in zip(self.norms, self.lins):
            h = h + self.act(lin(ln(h)))
        return self.head(h)


def from_central(cc, extra: int) -> Critic:
    """The checkpoint critic as a `Critic` with `extra` zero-initialised input
    columns inserted between obs and the opponent multi-hots (same function)."""
    w0 = cc.torso[0][0].weight.data
    hidden = w0.shape[0]
    obs_dim = w0.shape[1] - 260
    blocks = len(cc.torso) - 1
    c = Critic(obs_dim + extra + 260, hidden, blocks, "relu", False, cc.value_bins)
    with torch.no_grad():
        c.inp.weight.zero_()
        c.inp.weight[:, :obs_dim] = w0[:, :obs_dim]
        c.inp.weight[:, obs_dim + extra:] = w0[:, obs_dim:]
        c.inp.bias.copy_(cc.torso[0][0].bias)
        for k in range(blocks):
            blk = cc.torso[k + 1]
            c.norms[k].load_state_dict(blk.norm.state_dict())
            c.lins[k].load_state_dict(blk.linear.state_dict())
        c.head.load_state_dict(cc.value_head.state_dict())
        c.centers.copy_(cc._value_centers)
        c.edges.copy_(cc._value_edges)
    c.sigma = cc.hlgauss_sigma
    return c


def hl_loss(c: Critic, logits, ret):
    y = _symlog(ret.float())[:, None]
    inv = 1.0 / (c.sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf((c.edges[None, :] - y) * inv))
    tgt = cdf[:, 1:] - cdf[:, :-1]
    tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-8)
    return -(tgt * F.log_softmax(logits.float(), -1)).sum(-1)


def readouts(c: Critic, logits):
    p = F.softmax(logits.float(), -1)
    return _symexp((p * c.centers).sum(-1)), (p * _symexp(c.centers)).sum(-1)


# ---------------------------------------------------------------- metrics
def metrics(v, r, street):
    out = {}
    for k, name in [(None, "ALL")] + list(STREETS.items()):
        m = np.ones_like(r, dtype=bool) if k is None else street == k
        if not m.any():
            continue
        d = r[m] - v[m]
        out[name] = {"n": int(m.sum()), "bias": float(d.mean()), "rmse": float(np.sqrt((d * d).mean())),
                     "ev": float(1 - d.var() / r[m].var())}
    return out


def fmt(mt):
    return "  ".join(f"{k} bias {x['bias']:+6.2f} rmse {x['rmse']:6.2f} EV {x['ev']:.3f}" for k, x in mt.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("npz")
    ap.add_argument("--ckpt", default="checkpoints/vSix5.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="runs/critic_offline.jsonl")
    ap.add_argument("--save-dir", default="", help="save each trained candidate here")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    z = np.load(args.npz)
    obs16 = np.load(args.npz + ".obs16.npy", mmap_mode="r")
    n = obs16.shape[0]
    ret = z["returns"][:n].astype(np.float32)
    street = z["obs_street"][:n] if "obs_street" in z else z["street"][:n]
    opp = z["opp_holes"][:n]
    t0 = time.time()
    obs = np.asarray(obs16, dtype=np.float32)
    feats = strength_features(obs, opp)
    print(f"{n} rows; strength features {time.time() - t0:.0f}s", flush=True)
    blk = np.arange(n) // 2048
    test = np.random.default_rng(0).random(blk.max() + 1)[blk] < 0.2
    tr, te = np.nonzero(~test)[0], np.nonzero(test)[0]

    X = torch.from_numpy(obs).to(dev, torch.float16)
    FE = torch.from_numpy(feats).to(dev)
    OPP = opp_holes_multihot(torch.from_numpy(opp).to(dev)).to(torch.float16)
    R = torch.from_numpy(ret).to(dev)

    def inputs(rows, extra):
        parts = [X[rows].float()]
        if extra:
            parts.append(FE[rows])
        parts.append(OPP[rows].float())
        return torch.cat(parts, -1)

    def evaluate(c, extra):
        c.eval()
        vs, vr, ce = [], [], []
        with torch.no_grad():
            for i in range(0, len(te), 65536):
                rows = torch.from_numpy(te[i:i + 65536]).to(dev)
                lg = c.logits(inputs(rows, extra))
                a, b = readouts(c, lg)
                vs.append(a.cpu()); vr.append(b.cpu()); ce.append(hl_loss(c, lg, R[rows]).cpu())
        vs, vr = torch.cat(vs).numpy(), torch.cat(vr).numpy()
        rr, stt = ret[te], street[te]
        return {"ce": float(torch.cat(ce).mean()), "symlog_mean": metrics(vs, rr, stt),
                "raw_mean": metrics(vr, rr, stt)}

    def train(c, extra, lr, epochs):
        opt = torch.optim.AdamW(c.parameters(), lr=lr, weight_decay=0.0)
        g = torch.Generator(device="cpu").manual_seed(1)
        for ep in range(epochs):
            c.train()
            perm = tr[torch.randperm(len(tr), generator=g).numpy()]
            tot, cnt = 0.0, 0
            for i in range(0, len(perm), args.batch):
                rows = torch.from_numpy(perm[i:i + args.batch]).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lg = c.logits(inputs(rows, extra))
                loss = hl_loss(c, lg, R[rows]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(c.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(rows); cnt += len(rows)
            ev = evaluate(c, extra)
            print(f"    epoch {ep + 1}: train CE {tot / cnt:.4f}  held-out CE {ev['ce']:.4f}  "
                  f"raw-mean {fmt(ev['raw_mean'])}", flush=True)
        return ev

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    cc = build_critic_from_state_dict(ck["critic"], bool(cfg.get("q_fold_zero")), bool(cfg.get("q_base_raw")),
                                      value_support=cfg.get("value_support"),
                                      value_hlgauss_sigma=cfg.get("value_hlgauss_sigma"))
    base = from_central(cc, 0).to(dev)
    res = {"npz": args.npz, "ckpt": args.ckpt, "rows": n, "train": len(tr), "test": len(te)}
    ev = evaluate(base, 0)
    # The dueling Q of the TAKEN gate action (pooled head: 0 fold, 1 call,
    # 2 raise) -- what the VRPO advantage is built from.
    if cc.q_actions > 0:
        cc = cc.to(dev).eval()
        ga = torch.from_numpy(z["gate_actions"][:n].astype(np.int64)).to(dev)
        qs = []
        with torch.no_grad():
            for i in range(0, len(te), 65536):
                rows = torch.from_numpy(te[i:i + 65536]).to(dev)
                _v, q = cc.q_values(X[rows].float(), OPP[rows].float())
                col = ga[rows] if q.shape[-1] > 3 or True else ga[rows]
                qs.append(q.gather(-1, col.clamp(max=q.shape[-1] - 1)[:, None]).squeeze(-1).cpu())
        ev["q_taken"] = metrics(torch.cat(qs).numpy(), ret[te], street[te])
    res["current"] = ev
    print(f"CURRENT critic ({args.ckpt}): held-out CE {ev['ce']:.4f}")
    print(f"   V = symexp(E[symlog])  {fmt(ev['symlog_mean'])}")
    print(f"   V = raw-space mean     {fmt(ev['raw_mean'])}")
    if "q_taken" in ev:
        print(f"   Q(s, taken gate)       {fmt(ev['q_taken'])}", flush=True)
    obs_dim = obs.shape[1]
    cands = {
        "finetune": lambda: (from_central(cc, 0), 0, args.ft_lr),
        "finetune+str": lambda: (from_central(cc, 12), 12, args.ft_lr),
        "scratch-relu": lambda: (Critic(obs_dim + 260), 0, args.lr),
        "scratch-relu+str": lambda: (Critic(obs_dim + 12 + 260), 12, args.lr),
        "scratch-silu+str": lambda: (Critic(obs_dim + 12 + 260, act="silu", in_norm=True), 12, args.lr),
    }
    # Sized SiLU + LayerNorm'd-input critics without the strength inputs:
    # "silu-<width>x<blocks>", e.g. silu-1536x2, silu-3072x3.
    for spec in [s for s in args.only.split(",") if re.fullmatch(r"(silu|relu)-\d+x\d+", s)]:
        a, wb = spec.split("-")
        w, b = (int(x) for x in wb.split("x"))
        cands[spec] = (lambda a=a, w=w, b=b: (Critic(obs_dim + 260, w, b, act=a, in_norm=(a == "silu")), 0, args.lr))
    only = [s for s in args.only.split(",") if s]
    for name, make in cands.items():
        if only and name not in only:
            continue
        c, extra, lr = make()
        c = c.to(dev)
        print(f"\n[{name}] lr {lr}", flush=True)
        res[name] = train(c, extra, lr, args.epochs)
        if args.save_dir:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"name": name, "extra": extra, "state_dict": c.state_dict(),
                        "held_out": res[name]}, Path(args.save_dir) / f"{name}.pt")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
