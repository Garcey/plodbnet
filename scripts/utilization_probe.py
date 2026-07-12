#!/usr/bin/env python
"""Network-capacity utilization probe (V7_DESIGN.md workstream 2.1).

Question answered: is the 2048x4 actor / 1536x2 CentralCritic UNDER-utilized
(oversized -> a smaller net would train as well) or SATURATED (possibly
capacity-bound -> v7 wants a bigger net / new features)? The probe measures,
per nn.Linear, on a batch of real decision nodes:

  torso linears (post-ReLU activation is the unit signal):
    - dead_pct       units with 0 activation on EVERY probe row
    - near_dead_pct  units active (>0) on < 0.1% of rows
    - eff_rank       participation ratio (sum l)^2 / sum l^2, l = squared
                     singular values of the row-centered activation matrix
    - rank99         # singular values covering 99% of sum(s^2)
  head linears (gate/anchor/size/mix/refine/value/adv - no ReLU after):
    - eff_rank / rank99 on the RAW output (no dead metric)
  every linear:
    - stable_rank    ||W||_F^2 / sigma_max(W)^2   (a weight-matrix diagnostic)

This is a NEUTRAL methodology tool: it prints numbers and appends one JSON
line of history. It does NOT recommend a resize -- the canonical read happens
later on a healthy checkpoint.

Usage:
    .venv/Scripts/python scripts/utilization_probe.py CKPT \
        [--rows 8192] [--threads 8] [--out runs/utilization_history.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.network import (
    build_actor_from_state_dict,
    build_critic_from_state_dict,
    opp_holes_multihot,
)
from plo5bp.rollout import _rotate_opp_holes

GATE_FOLD, GATE_CC, GATE_RAISE = 0, 1, 2
BB = 10_000

# Row-subsample cap before every SVD (keeps a 2048-wide torso SVD to a couple
# of seconds); float32 throughout.
_SVD_MAX_ROWS = 4096

# The seven fields persisted per layer in the JSON history line.
_SCHEMA_KEYS = (
    "name", "width", "dead_pct", "near_dead_pct", "eff_rank", "rank99",
    "stable_rank",
)


# --------------------------------------------------------------------------
# Probe batch: real decision nodes (ported from q_head_audit_pooled.py)
# --------------------------------------------------------------------------
def gen_nodes(cfg, n_seeds, street="flop", bet="pot", seed0=0, max_nodes=None):
    """Deal `n_seeds` hands, advance to a chosen street, apply ONE pot/min
    raise, and keep the resulting node when the next actor exists and can
    fold. Returns (obs (N, OBS) f32, opp (N, 5, hole_w) u8, masks (N, 3) bool).

    Per-seed logic is identical to the q-head audit's `gen_nodes`; `max_nodes`
    only lets the caller stop early once it has enough nodes."""
    env = BombPotEnv(cfg)
    obs_rows, opp_rows, mask_rows = [], [], []
    for seed in range(seed0, seed0 + n_seeds):
        if max_nodes is not None and len(obs_rows) >= max_nodes:
            break
        obs, info = env.reset(seed=seed, button=seed % cfg.num_seats)
        if info.actor is None:
            continue
        if street == "turn":
            ok = True
            for _ in range(cfg.num_seats):
                obs, _r, done, info = env.step_hybrid(GATE_CC, 0)
                if done or info.actor is None:
                    ok = False
                    break
            if not ok:
                continue
        chips = info.max_raise_chips if bet == "pot" else info.min_raise_chips
        if chips is None or int(chips) <= 0:
            continue
        obs, _r, done, info = env.step_hybrid(GATE_RAISE, int(chips))
        if done or info.actor is None or not bool(info.gate_mask[GATE_FOLD]):
            continue
        holes = np.asarray(env.all_hole_cards(), dtype=np.uint8)
        obs_rows.append(obs)
        mask_rows.append(np.asarray(info.gate_mask, dtype=bool).copy())
        opp_rows.append(_rotate_opp_holes(holes, info.actor))
    return (
        np.stack(obs_rows).astype(np.float32),
        np.stack(opp_rows),
        np.stack(mask_rows),
    )


def build_probe_batch(rows):
    """Assemble ~rows/3 fold-legal flop nodes from each of three tables:
    default 20bb 6-max, deep (150bb) 6-max, and deep (150bb) heads-up.
    Total is capped at `rows`."""
    per = max(1, rows // 3)
    specs = [
        (GameConfig(), 0),
        (GameConfig(num_seats=6, starting_stack=150 * BB, ante=3 * BB, bb=BB),
         100_000),
        (GameConfig(num_seats=2, starting_stack=150 * BB, ante=3 * BB, bb=BB),
         150_000),
    ]
    obs_list, opp_list, mask_list = [], [], []
    for cfg, seed0 in specs:
        o, h, m = gen_nodes(
            cfg, n_seeds=per * 2 + 1000, street="flop", bet="pot",
            seed0=seed0, max_nodes=per,
        )
        o, h, m = o[:per], h[:per], m[:per]
        obs_list.append(o)
        opp_list.append(h)
        mask_list.append(m)
        print(f"  {cfg.num_seats}-seat stack={cfg.starting_stack // BB}bb "
              f"seed0={seed0}: {len(o)} nodes")
    obs = np.concatenate(obs_list)[:rows]
    opp = np.concatenate(opp_list)[:rows]
    masks = np.concatenate(mask_list)[:rows]
    return obs, opp, masks


# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------
def _dead_metrics(act_relu):
    """(dead_pct, near_dead_pct, dead_count) on a post-ReLU (B, W) matrix.

    dead      = unit is 0 on every row.
    near_dead = unit is active (>0) on < 0.1% of rows (includes dead)."""
    active = act_relu > 0                       # (B, W) bool
    width = int(act_relu.shape[1])
    ever = active.any(dim=0)                     # (W,)
    dead_count = int((~ever).sum().item())
    active_frac = active.float().mean(dim=0)     # (W,)
    near_dead = int((active_frac < 0.001).sum().item())
    return 100.0 * dead_count / width, 100.0 * near_dead / width, dead_count


def _eff_rank_and_rank99(feat, gen=None):
    """(eff_rank, rank99) of a (B, W) feature matrix.

    eff_rank = participation ratio (sum l)^2 / sum l^2 of the row-centered
    matrix's squared singular values l = s^2; rank99 = # singular values
    covering 99% of sum l. Rows are subsampled to <= _SVD_MAX_ROWS first."""
    x = feat.float()
    b = int(x.shape[0])
    if b > _SVD_MAX_ROWS:
        if gen is not None:
            idx = torch.randperm(b, generator=gen)[:_SVD_MAX_ROWS]
        else:
            idx = torch.arange(_SVD_MAX_ROWS)
        x = x[idx]
    x = x - x.mean(dim=0, keepdim=True)          # center each column
    s = torch.linalg.svdvals(x)                  # descending
    lam = s * s
    total = float(lam.sum())
    if total <= 0.0:
        return 0.0, 0
    eff_rank = float((total * total) / float((lam * lam).sum()))
    csum = torch.cumsum(lam, dim=0)
    rank99 = int((csum < 0.99 * total).sum().item()) + 1
    return eff_rank, min(rank99, int(lam.numel()))


def weight_stable_rank(weight):
    """||W||_F^2 / sigma_max(W)^2 -- an intrinsic-dimensionality proxy for a
    weight matrix (1 for a rank-1 matrix, up to min(shape) for a flat
    spectrum)."""
    w = weight.detach().float()
    fro2 = float((w * w).sum())
    smax = float(torch.linalg.svdvals(w).max())
    if smax <= 0.0:
        return 0.0
    return fro2 / (smax * smax)


# --------------------------------------------------------------------------
# Per-net measurement
# --------------------------------------------------------------------------
def _capture_linear_outputs(model, run_forward):
    """Run `run_forward()` (a no-arg closure that triggers a model forward)
    with a forward hook on every nn.Linear, returning {name: output (B, W)}."""
    captured = {}
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            def _hook(m, i, o, n=name):
                captured[n] = o.detach().clone()
            handles.append(mod.register_forward_hook(_hook))
    try:
        with torch.no_grad():
            run_forward()
    finally:
        for h in handles:
            h.remove()
    return captured


def measure_net(model, run_forward, gen=None):
    """Measure every nn.Linear in `model`. Torso linears (name starts with
    'torso.') report on the post-ReLU activation with dead metrics; head
    linears report eff_rank/rank99 on the raw output. Returns a list of layer
    dicts in torso-then-head order."""
    captured = _capture_linear_outputs(model, run_forward)
    modules = dict(model.named_modules())
    linear_names = [n for n, m in modules.items() if isinstance(m, nn.Linear)]
    torso_names = [n for n in linear_names if n.startswith("torso.")]
    head_names = [n for n in linear_names if not n.startswith("torso.")]

    layers = []
    for name in torso_names + head_names:
        mod = modules[name]
        out = captured[name].float()
        width = int(mod.out_features)
        is_torso = name.startswith("torso.")
        if is_torso:
            act = F.relu(out)
            dead_pct, near_dead_pct, dead_count = _dead_metrics(act)
            eff_rank, rank99 = _eff_rank_and_rank99(act, gen=gen)
        else:
            dead_pct = near_dead_pct = None
            dead_count = 0
            eff_rank, rank99 = _eff_rank_and_rank99(out, gen=gen)
        layers.append(dict(
            name=name, width=width, is_torso=is_torso,
            dead_pct=dead_pct, near_dead_pct=near_dead_pct,
            dead_count=dead_count, eff_rank=eff_rank, rank99=rank99,
            stable_rank=weight_stable_rank(mod.weight),
        ))
    return layers


def measure_actor(actor, obs, masks, gen=None):
    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))
    masks_t = torch.from_numpy(np.asarray(masks, dtype=bool))
    return measure_net(actor, lambda: actor(obs_t, masks_t), gen=gen)


def measure_critic(critic, obs, opp, gen=None):
    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))
    opp_t = opp_holes_multihot(torch.from_numpy(np.asarray(opp)))

    def run():
        # train_outputs runs the torso once AND both value + dueling-adv
        # heads (when present) in a single pass, so all head linears are
        # captured; falls back to forward()'s value-only path otherwise.
        if getattr(critic, "q_actions", 0) > 0:
            critic.train_outputs(obs_t, opp_t)
        else:
            critic(obs_t, opp_t)

    return measure_net(critic, run, gen=gen)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def _fmt_opt(v, fmt):
    return "   -  " if v is None else format(v, fmt)


def print_table(title, layers):
    print(f"\n=== {title} ===")
    hdr = (f"{'layer':<20} {'width':>6} {'dead%':>7} {'near%':>7} "
           f"{'eff_rank':>9} {'rank99':>7} {'effr/w':>7} {'stable':>9}")
    print(hdr)
    print("-" * len(hdr))
    for L in layers:
        effr_w = L["eff_rank"] / L["width"] if L["width"] else 0.0
        print(f"{L['name']:<20} {L['width']:>6} "
              f"{_fmt_opt(L['dead_pct'], '>7.2f')} "
              f"{_fmt_opt(L['near_dead_pct'], '>7.2f')} "
              f"{L['eff_rank']:>9.1f} {L['rank99']:>7d} "
              f"{effr_w:>7.3f} {L['stable_rank']:>9.1f}")


def _torso(layers):
    return [L for L in layers if L["is_torso"]]


def print_summary(actor_layers, critic_layers):
    print("\n=== summary ===")
    lines = []
    floors = {}
    for tag, layers in (("actor", actor_layers), ("critic", critic_layers)):
        t = _torso(layers)
        ratios = [L["eff_rank"] / L["width"] for L in t]
        rmin, rmed = float(np.min(ratios)), float(np.median(ratios))
        floors[tag] = rmin
        dead_units = sum(L["dead_count"] for L in t)
        total_units = sum(L["width"] for L in t)
        lines.append(
            f"{tag:<7} torso eff_rank/width: min {rmin:.3f}  median {rmed:.3f}"
            f"   dead units {dead_units}/{total_units}")
    for ln in lines:
        print(ln)
    print("reading: "
          f"actor torso eff-rank floor = {floors['actor'] * 100:.0f}% of width; "
          f"critic torso eff-rank floor = {floors['critic'] * 100:.0f}% of width")


def _layers_for_json(layers):
    return [{k: L[k] for k in _SCHEMA_KEYS} for L in layers]


def build_record(ckpt_path, rows, actor_layers, critic_layers):
    stem = Path(ckpt_path).stem
    m = re.search(r"_(\d+)$", stem)
    return {
        "ckpt": os.path.basename(ckpt_path),
        "update": int(m.group(1)) if m else None,
        "ts": time.time(),
        "rows": int(rows),
        "actor": {"layers": _layers_for_json(actor_layers)},
        "critic": {"layers": _layers_for_json(critic_layers)},
    }


def write_record(record, out_path):
    p = Path(out_path)
    if p.parent != Path(""):
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------
# Loading + driver
# --------------------------------------------------------------------------
def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfgb = ckpt.get("config", {}) or {}
    actor = build_actor_from_state_dict(
        ckpt["model"], int(cfgb.get("hidden_dim", 128)),
        int(cfgb.get("num_layers", 2)),
    ).eval()
    critic = build_critic_from_state_dict(ckpt["critic"]).eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    for p in critic.parameters():
        p.requires_grad_(False)
    return actor, critic


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt")
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="runs/utilization_history.jsonl")
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))
    gen = torch.Generator().manual_seed(0)

    t0 = time.time()
    print(f"loading {args.ckpt} ...")
    actor, critic = load_ckpt(args.ckpt)
    print(f"actor: {type(actor).__name__}  hidden={actor.torso[0][0].out_features}"
          f"  torso-linears={len(_torso_linear_names(actor))}")
    print(f"critic: {type(critic).__name__}  "
          f"hidden={critic.torso[0][0].out_features}  "
          f"q_actions={getattr(critic, 'q_actions', 0)}  "
          f"value_bins={getattr(critic, 'value_bins', 0)}")

    print(f"building probe batch (target {args.rows} rows) ...")
    obs, opp, masks = build_probe_batch(args.rows)
    n_rows = int(obs.shape[0])
    print(f"probe rows: {n_rows}")

    actor_layers = measure_actor(actor, obs, masks, gen=gen)
    critic_layers = measure_critic(critic, obs, opp, gen=gen)

    print_table(
        f"ACTOR  {type(actor).__name__}  hidden="
        f"{actor.torso[0][0].out_features}", actor_layers)
    print_table(
        f"CRITIC  {type(critic).__name__}  hidden="
        f"{critic.torso[0][0].out_features}", critic_layers)
    print_summary(actor_layers, critic_layers)

    record = build_record(args.ckpt, n_rows, actor_layers, critic_layers)
    write_record(record, args.out)
    print(f"\nappended history line -> {args.out}")
    print(f"wall time: {time.time() - t0:.1f}s")
    return 0


def _torso_linear_names(model):
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.startswith("torso.")]


if __name__ == "__main__":
    raise SystemExit(main())
