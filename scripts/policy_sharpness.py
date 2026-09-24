#!/usr/bin/env python
"""Policy sharpness / collapse probe (2026-09-24 hyperparameter tuning).

How deterministic is a checkpoint's policy, measured on ONE fixed set of
decision states so every checkpoint is compared on the same spots? The states
come from the self-play of a reference checkpoint (`--states-from`) over
training-like table configs (the utilization probe's `--states selfplay`
recipe: 10 configs per training tier) and are cached in `--cache`, so later
calls reuse them. Per tier and overall it reports

  - gate entropy over the legal gate actions: mean and p10 / p50 / p90
  - `rare_gate<1e-2` / `<1e-3`: share of states whose LEAST likely legal gate
    action has probability below 1% / 0.1% -- an action the policy has all but
    stopped trying there (the collapse the owner's high-entropy phase guards
    against; at 0 it can never be re-learned)
  - raise-size entropy (the legal-anchor distribution) over raise-legal states,
    and `rare_anchor<1e-3`: the share of legal anchors below 0.1%
  - `p_raise`: the mean raise probability where raising is legal

It prints a table and appends one JSON line per checkpoint to `--out`.

    .venv/bin/python scripts/policy_sharpness.py \
        --states-from checkpoints/swA32_40.pt --cache runs/sharpness_states.npz \
        checkpoints/t1lr150_59.pt checkpoints/t2ent15_59.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from plo5bp.actions import GATE_RAISE
from plo5bp.network import build_actor_from_state_dict
from plo5bp.sizing import anchor_grid_torch

REPO = Path(__file__).resolve().parents[1]
BB = 10_000
TIERS = ("clubgg", "clubgg_deep", "deep")
_ANCHOR_NAMES = ("min", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "pot")


def _load_actor(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    actor = build_actor_from_state_dict(
        ckpt["model"], int(cfg.get("hidden_dim", 128)), int(cfg.get("num_layers", 2))
    ).eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    obs_mode = str(cfg.get("obs_mode", "full") or "full").strip().lower()
    return actor, obs_mode, ckpt.get("update")


def build_states(actor, rows: int, obs_mode: str, seed: int = 0, tables: int = 256):
    """Self-play decision states of `actor` (sampling its policy) on 30 table
    configs, 10 per tier, drawn like training; a uniform sample of `rows`."""
    from plo5bp.env_batched import BatchedBombPotEnv

    spec = importlib.util.spec_from_file_location("_train", REPO / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    obs_l, mask_l, siz_l, tier_l = [], [], [], []
    for ti, tier in enumerate(TIERS):
        for _ in range(10):
            cfg, _ = train._sample_game_config(
                (2, 3, 4, 5, 6), 1.0, 300.0, BB, 3 * BB, rng,
                stack_dist=tier, seats_dist="uniform", variant="plo5_double_bomb", sb=0,
            )
            env = BatchedBombPotEnv(tables, cfg, opp_outcome_mc=0, obs_mode=obs_mode)
            env.reset_batch(
                rng.integers(0, 2**63 - 1, size=tables, dtype=np.int64).astype(np.uint64),
                rng.integers(0, cfg.num_seats, size=tables).astype(np.uint8),
            )
            all_rows = np.arange(tables)
            while not env._dones.all():
                live = np.nonzero(~env._dones)[0]
                safe = np.where(env._actors >= 0, env._actors, 0).astype(np.intp)
                to_call = np.maximum(
                    env._bet_to_call.astype(np.int64)
                    - env._street_commit[all_rows, safe].astype(np.int64), 0,
                )
                sizing = np.stack(
                    [env._min_raise.astype(np.int64), env._max_raise.astype(np.int64),
                     env._pot.astype(np.int64), to_call], axis=-1,
                )
                obs_l.append(env._obs[live].copy())
                mask_l.append(env._gate_mask[live].copy())
                siz_l.append(sizing[live].copy())
                tier_l.append(np.full(live.size, ti, dtype=np.int8))
                gates = np.zeros(tables, dtype=np.uint8)
                chips = np.zeros(tables, dtype=np.uint64)
                with torch.no_grad():
                    out = actor.act(
                        torch.from_numpy(env._obs[live]),
                        torch.from_numpy(env._gate_mask[live]),
                        torch.from_numpy(sizing[live]),
                    )
                gates[live] = out.gate.numpy().astype(np.uint8)
                chips[live] = np.maximum(out.chips.numpy(), 0).astype(np.uint64)
                env.step_hybrid_batch(gates, chips)
    obs = np.concatenate(obs_l)
    pick = np.sort(rng.permutation(obs.shape[0])[: int(rows)])
    print(f"  self-play: {obs.shape[0]} decision states over 30 configs; sampled {pick.size}")
    return (obs[pick], np.concatenate(mask_l)[pick], np.concatenate(siz_l)[pick],
            np.concatenate(tier_l)[pick])


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(1e-30))).sum(-1)


def measure(actor, obs, masks, sizing, tiers, batch: int = 8192) -> dict:
    gate_h, rare2, rare3, p_raise, raise_ok = [], [], [], [], []
    anch_h, anch_rare, anch_legal, anch_p, anch_ok = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, obs.shape[0], batch):
            o = torch.from_numpy(obs[s:s + batch])
            m = torch.from_numpy(masks[s:s + batch])
            z = torch.from_numpy(sizing[s:s + batch])
            gate_logits, head_out, _refine, _v = actor(o, m)
            gp = torch.softmax(gate_logits.float(), dim=-1) * m
            gp = gp / gp.sum(-1, keepdim=True)
            gate_h.append(_entropy(gp))
            legal_min = torch.where(m, gp, torch.ones_like(gp)).min(-1).values
            n_legal = m.sum(-1)
            rare2.append((legal_min < 1e-2) & (n_legal > 1))
            rare3.append((legal_min < 1e-3) & (n_legal > 1))
            ok = m[:, GATE_RAISE]
            raise_ok.append(ok)
            p_raise.append(gp[:, GATE_RAISE])
            grid = anchor_grid_torch(z, actor.anchor_spec)
            ap = actor._anchor_dist(head_out, grid).probs.float()
            anch_h.append(_entropy(ap))
            anch_rare.append(((ap < 1e-3) & grid.legal).sum(-1).float())
            anch_legal.append(grid.legal.sum(-1).float())
            anch_p.append(ap)
            anch_ok.append(grid.legal)
    gate_h = torch.cat(gate_h).numpy()
    rare2 = torch.cat(rare2).numpy()
    rare3 = torch.cat(rare3).numpy()
    raise_ok = torch.cat(raise_ok).numpy()
    p_raise = torch.cat(p_raise).numpy()
    anch_h = torch.cat(anch_h).numpy()
    anch_rare = torch.cat(anch_rare).numpy()
    anch_legal = torch.cat(anch_legal).numpy()
    anch_p = torch.cat(anch_p).numpy()
    anch_ok = torch.cat(anch_ok).numpy()
    multi = (masks.sum(-1) > 1)

    def block(sel):
        g = sel & multi
        r = sel & raise_ok & (anch_legal > 1)
        return {
            "n": int(sel.sum()),
            "gate_h": float(gate_h[g].mean()) if g.any() else None,
            "gate_h_p10_p50_p90": [float(x) for x in np.percentile(gate_h[g], [10, 50, 90])] if g.any() else None,
            "rare_gate_1e2": float(rare2[g].mean()) if g.any() else None,
            "rare_gate_1e3": float(rare3[g].mean()) if g.any() else None,
            "p_raise": float(p_raise[sel & raise_ok].mean()) if (sel & raise_ok).any() else None,
            "anchor_h": float(anch_h[r].mean()) if r.any() else None,
            "rare_anchor_1e3": float((anch_rare[r] / anch_legal[r]).mean()) if r.any() else None,
            # mean probability of each anchor (min, 10%, ..., pot) where raising
            # with 2+ legal sizes, and how often each anchor is legal there
            "anchor_mean": [float(x) for x in anch_p[r].mean(0)] if r.any() else None,
            "anchor_legal": [float(x) for x in anch_ok[r].mean(0)] if r.any() else None,
            "anchor_top": float(anch_p[r].max(-1).mean()) if r.any() else None,
        }

    out = {"ALL": block(np.ones_like(multi))}
    for ti, tier in enumerate(TIERS):
        out[tier] = block(tiers == ti)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--states-from", default="checkpoints/swA32_40.pt",
                    help="checkpoint whose self-play generates the probe states (first call)")
    ap.add_argument("--cache", default="runs/sharpness_states.npz")
    ap.add_argument("--rows", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="runs/sharpness_history.jsonl")
    args = ap.parse_args(argv)
    torch.set_num_threads(int(args.threads))
    cache = Path(args.cache)
    if cache.exists():
        z = np.load(cache)
        obs, masks, sizing, tiers = z["obs"], z["masks"], z["sizing"], z["tiers"]
        print(f"states: {obs.shape[0]} from {cache}")
    else:
        t0 = time.time()
        ref, obs_mode, _ = _load_actor(args.states_from)
        print(f"building {args.rows} states from {args.states_from} (obs_mode={obs_mode}) ...")
        obs, masks, sizing, tiers = build_states(ref, args.rows, obs_mode)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, obs=obs, masks=masks, sizing=sizing, tiers=tiers,
                            states_from=str(args.states_from))
        print(f"  cached -> {cache} ({time.time() - t0:.0f}s)")

    def f(x, fmt="%.3f"):
        return "  -  " if x is None else fmt % x

    print(f"\n{'checkpoint':28s} {'tier':12s} {'gateH':>6s} {'p10/p50/p90':>17s} "
          f"{'rare<1%':>8s} {'rare<.1%':>8s} {'pRaise':>7s} {'sizeH':>6s} {'rareSz':>7s}")
    with open(args.out, "a") as fh:
        for path in args.ckpts:
            actor, _mode, upd = _load_actor(path)
            res = measure(actor, obs, masks, sizing, tiers)
            for k in ("ALL",) + TIERS:
                b = res[k]
                q = b["gate_h_p10_p50_p90"]
                qs = "  -  " if q is None else "%.2f/%.2f/%.2f" % tuple(q)
                print(f"{Path(path).name:28s} {k:12s} {f(b['gate_h']):>6s} {qs:>17s} "
                      f"{f(b['rare_gate_1e2']):>8s} {f(b['rare_gate_1e3']):>8s} "
                      f"{f(b['p_raise']):>7s} {f(b['anchor_h']):>6s} {f(b['rare_anchor_1e3']):>7s}")
            am = res["ALL"]["anchor_mean"]
            if am is not None:
                print(f"{'':28s} {'sizes':12s} top-size prob {res['ALL']['anchor_top']:.3f} | mean prob "
                      + " ".join("%s %.3f" % (n, x) for n, x in zip(_ANCHOR_NAMES, am)))
            fh.write(json.dumps({"ckpt": str(path), "update": upd, "states": str(cache),
                                 "result": res}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
