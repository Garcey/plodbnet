#!/usr/bin/env python
"""Actor size study by DISTILLATION (2026-09-26 network redesign).

Can a smaller actor represent the policy the 2048x4 actor has learned? Train
student actors of several sizes to imitate a teacher checkpoint on the decision
states of a rollout dump (`PLO5BP_DUMP_BATCH=<path>` in train.py: <path> npz
with gate_masks + sizing, <path>.obs16.npy with the observations), then score
each student against the teacher (scripts/h2h_eval.py-style duplicate deals).

Loss per state (all KL(teacher || student), the teacher's play weighting):
  gate      KL over the legal fold / check-call / raise
  sizing    p_T(raise) * KL over the legal anchors
  refine    p_T(raise) * sum_k p_T(anchor k) * KL(Beta_T,k || Beta_S,k)
  value     1e-4 * (V_S - V_T)^2  (the actor's display value head)

Students are saved as full checkpoints (the teacher's file with the student's
"model" and hidden_dim / num_layers), so train.py can warm-start from them
(--hidden-dim / --num-layers must then match the student).

    .venv/bin/python scripts/distill_size.py /root/diag_batch_u1290.npz \
        --teacher checkpoints/vSix5.pt --sizes 256x3,512x3,1024x3 --epochs 8 \
        --out-dir checkpoints/distill
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from plo5bp.network import ActorCriticV5, actor_arch
from plo5bp.evaluation import load_actor
from plo5bp.train.checkpoint import derived_checkpoint
from plo5bp.sizing import anchor_grid_torch


def load_teacher(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    t, _meta = load_actor(ck, dev, check_rev=False)  # the states carry their own rev
    return ck, t


def heads(model, obs, gm, sizing):
    """(gate log-probs (B,3), anchor probs (B,A), refine (B,A-2,2), value (B,))."""
    gate_logits, mix, refine, value = model(obs, gm)
    grid = anchor_grid_torch(sizing, model.anchor_spec)
    anc = model._anchor_dist(mix.float(), grid).probs
    return F.log_softmax(gate_logits.float(), -1), anc, refine.float(), value.float(), grid


def distill_loss(s_out, t_out, gm):
    ls, as_, rs, vs, _ = s_out
    lt, at, rt, vt, grid = t_out
    pt = lt.exp()
    legal = gm.bool()
    gate_kl = torch.where(legal, pt * (lt - ls), torch.zeros_like(pt)).sum(-1)
    p_raise = pt[:, 2]
    leg_a = grid.legal
    anc_kl = torch.where(leg_a, at * (torch.log(at.clamp_min(1e-12)) - torch.log(as_.clamp_min(1e-12))),
                         torch.zeros_like(at)).sum(-1)
    # interior anchors 1..A-2 carry a Beta refinement
    bt = torch.distributions.Beta(rt[..., 0].clamp_min(1e-3), rt[..., 1].clamp_min(1e-3))
    bs = torch.distributions.Beta(rs[..., 0].clamp_min(1e-3), rs[..., 1].clamp_min(1e-3))
    ref_kl = (torch.distributions.kl_divergence(bt, bs) * at[:, 1:-1]).sum(-1)
    raise_ok = legal[:, 2].float()
    size_term = raise_ok * p_raise * (anc_kl + ref_kl)
    v_term = 1e-4 * (vs - vt).pow(2)
    return gate_kl, anc_kl * raise_ok * p_raise, ref_kl * raise_ok * p_raise, gate_kl + size_term + v_term


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("npz")
    ap.add_argument("--teacher", default="checkpoints/vSix5.pt")
    ap.add_argument("--sizes", default="256x3,512x3,1024x3")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", default="checkpoints/distill")
    ap.add_argument("--log", default="runs/distill_size.jsonl")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    z = np.load(args.npz)
    obs16 = np.load(args.npz + ".obs16.npy", mmap_mode="r")
    n = obs16.shape[0]
    X = torch.from_numpy(np.asarray(obs16)).to(dev)                 # f16
    GM = torch.from_numpy(z["gate_masks"][:n]).to(dev).bool()
    SZ = torch.from_numpy(z["sizing"][:n].astype(np.int64)).to(dev)
    blk = np.arange(n) // 2048
    test = np.random.default_rng(0).random(blk.max() + 1)[blk] < 0.1
    tr, te = np.nonzero(~test)[0], np.nonzero(test)[0]
    ck, teacher = load_teacher(args.teacher, dev)
    tcfg = ck.get("config") or {}
    print(f"{n} states ({len(tr)} train / {len(te)} held out); teacher {args.teacher} "
          f"{tcfg.get('hidden_dim')}x{tcfg.get('num_layers')}", flush=True)

    def teacher_out(rows):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            return heads(teacher, X[rows].float(), GM[rows], SZ[rows])

    def evaluate(student):
        student.eval()
        acc = np.zeros(4)
        cnt = 0
        with torch.no_grad():
            for i in range(0, len(te), 32768):
                rows = torch.from_numpy(te[i:i + 32768]).to(dev)
                t_out = teacher_out(rows)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    s_out = heads(student, X[rows].float(), GM[rows], SZ[rows])
                g, a, r, tot = distill_loss(s_out, t_out, GM[rows])
                acc += np.array([float(g.sum()), float(a.sum()), float(r.sum()), float(tot.sum())])
                cnt += len(rows)
        return dict(zip(("gate_kl", "anchor_kl", "refine_kl", "total"), (acc / cnt).tolist()))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in [s for s in args.sizes.split(",") if s]:
        h, l = (int(x) for x in spec.split("x"))
        student = ActorCriticV5(
            hidden_dim=h, obs_dim=X.shape[1], num_layers=l,
            anchor_spec=teacher.anchor_spec, mixture_k=teacher._mixture_k,
            torso_layernorm=bool(tcfg.get("torso_layernorm", True)),
        ).to(dev)
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
        steps = args.epochs * math.ceil(len(tr) / args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
        g = torch.Generator().manual_seed(1)
        t0 = time.time()
        print(f"\n[{spec}] {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M params", flush=True)
        for ep in range(args.epochs):
            student.train()
            perm = tr[torch.randperm(len(tr), generator=g).numpy()]
            for i in range(0, len(perm), args.batch):
                rows = torch.from_numpy(perm[i:i + args.batch]).to(dev)
                t_out = teacher_out(rows)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    s_out = heads(student, X[rows].float(), GM[rows], SZ[rows])
                loss = distill_loss(s_out, t_out, GM[rows])[3].mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                sched.step()
            ev = evaluate(student)
            print(f"    epoch {ep + 1}: held-out gate KL {ev['gate_kl']:.5f}  anchor KL {ev['anchor_kl']:.5f}  "
                  f"refine KL {ev['refine_kl']:.5f}  total {ev['total']:.5f}  ({time.time() - t0:.0f}s)", flush=True)
        cfg = dict(tcfg)
        cfg["hidden_dim"], cfg["num_layers"] = h, l
        arch = {"actor": actor_arch(student)}
        if isinstance(ck.get("arch"), dict) and "critic" in ck["arch"]:
            arch["critic"] = ck["arch"]["critic"]
        # The teacher's networks-describing keys, not its run's bookkeeping
        # (ML-056): a warm start from a student is a new lineage.
        out = derived_checkpoint(
            ck, "distill", [args.teacher],
            model={k: v.detach().cpu() for k, v in student.state_dict().items()},
            config=cfg, arch=arch,
            distilled_from={"teacher": args.teacher, "states": args.npz, "held_out": ev},
        )
        path = out_dir / f"distill_{spec}.pt"
        torch.save(out, path)
        with open(args.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"size": spec, "teacher": args.teacher, "held_out": ev, "path": str(path)}) + "\n")
        print(f"    saved {path}", flush=True)


if __name__ == "__main__":
    main()
