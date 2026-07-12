#!/usr/bin/env python
"""Per-checkpoint automated probe suite (V7_DESIGN.md workstream 3).

Unifies two proven scratchpad probes into one tracked CLI tool so the
per-checkpoint health signals are reproducible and logged to a JSONL
history instead of re-derived by hand each session:

  1. Q-calibration audit  (port of scratchpad q_head_audit_pooled.py) —
     the pooled-era acceptance gate. Per node family it reports the
     critic's V distribution, the fold-column canary (Q[FOLD], ground
     truth exactly 0) with its V-regression slope/resid, the call/raise
     advantages (width-aware: 3-col pooled vs 13-col legacy head), and
     the policy-weighted E_pi[Q]-V calibration gap.

  2. Lock-fold probe  (port of scratchpad probe_lock_folds.py) — did the
     policy learn to FOLD locked hands (hero ahead of 100% of k=2 opp
     combos on >=1 board) and is that unwinding? Reports P(fold)/P(raise)
     at locked nodes plus a trash (behind >=90% on both boards) control.

Node banks are generated ONCE (env replays are the expensive part) and
reused across every checkpoint on the command line, so multi-checkpoint
runs cost one node-gen pass plus one forward per checkpoint per family.

    python scripts/probe_suite.py CKPT [CKPT ...] \
        [--fast] [--threads 8] [--out runs/probe_history.jsonl]

--fast scales every family's seed count by 0.1 (smoke/testing). Each
checkpoint appends one JSON line to --out; the human-readable stdout ends
with a cross-checkpoint GATE SUMMARY + LOCKS summary table.

The metric math is ported verbatim from the two scratchpad probes; keep
it bit-for-bit identical when editing (the whole point is comparability
with the hand-run history — see the acceptance ranges in the module the
suite replaces). New vs the scratchpads: gate sharpness (actor-only, so
it survives an actor-only checkpoint), the JSONL history, and the
multi-checkpoint reuse of node banks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.network import (
    build_actor_from_state_dict,
    build_critic_from_state_dict,
    opp_holes_multihot,
)
from plo5bp.rollout import _rotate_opp_holes

GATE_CC = GATE_CHECK_CALL  # scratchpad probes name the check/call gate GATE_CC
BB = 10_000

CLUBGG = GameConfig()
DEEP6 = GameConfig(num_seats=6, starting_stack=150 * BB, ante=3 * BB, bb=BB)
DEEPHU = GameConfig(num_seats=2, starting_stack=150 * BB, ante=3 * BB, bb=BB)

# (letter, tag, cfg, n_seeds, street, bet, seed0). Families A and C are
# supersets of the 2026-07-09 audit (4000/6000 vs 2000 seeds) so the same
# banks feed the lock split — do NOT shrink them without re-checking both
# probes' acceptance ranges. Seeds/seed0 are chosen disjoint per family.
FAMILIES = [
    ("A", "A 20bb 6max FLOP vs pot-bet", CLUBGG, 4000, "flop", "pot", 0),
    ("B", "B 20bb 6max FLOP vs min-bet", CLUBGG, 1000, "flop", "min", 50_000),
    ("C", "C 150bb 6max FLOP vs pot-bet", DEEP6, 6000, "flop", "pot", 100_000),
    ("D", "D 150bb HU FLOP vs pot-bet", DEEPHU, 1000, "flop", "pot", 150_000),
    ("E", "E 20bb 6max TURN vs pot-bet", CLUBGG, 1000, "turn", "pot", 200_000),
    ("F", "F 150bb 6max TURN vs pot-bet", DEEP6, 1000, "turn", "pot", 250_000),
]
# Lock/trash split runs on the two flop-vs-pot supersets only.
LOCK_TAGS = {"shallow": "A 20bb 6max FLOP vs pot-bet",
             "deep": "C 150bb 6max FLOP vs pot-bet"}


# ------------------------------------------------------------------ nodes
def gen_nodes(cfg, n_seeds, street="flop", bet="pot", seed0=0):
    """Ported from q_head_audit_pooled.gen_nodes (the superset that ALSO
    collects hero-rotated opponent holes for the critic). Procedure:
    reset(seed, button=seed%num_seats); optionally check the whole table
    around to the turn; then one pot-sized (or min) raise; keep the node
    iff the next actor exists and fold is legal there. Returns
    (obs (N, OBS), opp (N, 5, hole_w) u8, masks (N, 3) bool)."""
    env = BombPotEnv(cfg)
    obs_rows, opp_rows, mask_rows = [], [], []
    for seed in range(seed0, seed0 + n_seeds):
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
    if not obs_rows:
        raise RuntimeError(
            f"gen_nodes produced 0 nodes (cfg seats={cfg.num_seats} "
            f"n_seeds={n_seeds} street={street} bet={bet} seed0={seed0})"
        )
    return (
        np.stack(obs_rows).astype(np.float32),
        np.stack(opp_rows),
        np.stack(mask_rows),
    )


def build_banks(fast=False):
    """Generate every family's node bank once. --fast scales seeds ×0.1."""
    scale = 0.1 if fast else 1.0
    banks = {}
    print("generating node families "
          f"({'FAST ×0.1' if fast else 'full'} seed counts)...")
    for _letter, tag, cfg, n_seeds, street, bet, seed0 in FAMILIES:
        n = max(1, int(round(n_seeds * scale)))
        obs, opp, masks = gen_nodes(cfg, n, street, bet, seed0)
        banks[tag] = (obs, opp, masks)
        print(f"  {tag}: {len(obs)} nodes")
    return banks


# ------------------------------------------------------------------ load
def load_checkpoint(path):
    """(actor, critic) — critic is None for an actor-only checkpoint (no
    'critic' block). Both nets are eval / frozen / inference-only."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfgb = ckpt.get("config", {}) or {}
    actor = build_actor_from_state_dict(
        ckpt["model"],
        int(cfgb.get("hidden_dim", 128)),
        int(cfgb.get("num_layers", 2)),
    ).eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    critic = None
    if "critic" in ckpt and ckpt["critic"] is not None:
        critic = build_critic_from_state_dict(ckpt["critic"]).eval()
        for p in critic.parameters():
            p.requires_grad_(False)
    return actor, critic


# --------------------------------------------------------------- metrics
def _pct(x, p):
    return float(np.percentile(x, p))


def _use_critic(critic):
    """The Q-calibration block needs the dueling adv head; a critic
    without one (q_actions == 0) is treated like actor-only for Q."""
    return critic is not None and int(getattr(critic, "q_actions", 0)) > 0


def family_metrics(actor, critic, obs, opp, masks):
    """Per-family probe metrics. Policy metrics (gate probs + sharpness)
    are always present; the Q-calibration block is added only when a
    usable dueling critic is supplied. Math ported verbatim from
    q_head_audit_pooled.audit (3-col pooled vs 13-col width-aware
    branches kept identical). All values are plain floats/lists."""
    obs_t = torch.from_numpy(obs)
    masks_t = torch.from_numpy(masks)
    with torch.inference_mode():
        g_logits, _a, _r, _vd = actor(obs_t, masks_t)
        gate_p = torch.softmax(g_logits.float(), -1)
    gp = gate_p.numpy()

    m = {}
    gp_mean = gp.mean(0)
    m["gate_p_fold"] = float(gp_mean[0])
    m["gate_p_cc"] = float(gp_mean[1])
    m["gate_p_raise"] = float(gp_mean[2])

    # Gate sharpness (new; actor-only, works without a critic).
    sharp = gp.max(1)
    m["sharp_med"] = float(np.median(sharp))
    with np.errstate(divide="ignore", invalid="ignore"):
        # gate logits are already masked to -1e9 on illegal gates, so the
        # softmax is a distribution over the LEGAL gates and its entropy is
        # the legal-gate entropy directly (illegal p == 0 -> 0*log0 := 0).
        ent = -np.where(gp > 0, gp * np.log(gp), 0.0).sum(1)
    m["gate_h_mean"] = float(ent.mean())

    if not _use_critic(critic):
        return m

    with torch.inference_mode():
        v_t, q_t = critic.q_values(
            obs_t, opp_holes_multihot(torch.from_numpy(opp))
        )
    v = v_t.float().numpy()
    q = q_t.float().numpy()
    ncols = q.shape[1]
    adv = q - v[:, None]
    err_f = q[:, 0]                       # truth: exactly 0
    slope, icept = np.polyfit(v, err_f, 1)
    resid = err_f - (icept + slope * v)
    A_r = adv[:, 2:]                      # (N, ncols-2)
    q_r_mean = q[:, 2:].mean(1)           # exact under pooling
    e_pi_q = gp[:, 0] * q[:, 0] + gp[:, 1] * q[:, 1] + gp[:, 2] * q_r_mean
    gap = e_pi_q - v

    m["v_mean"] = float(v.mean())
    m["v_std"] = float(v.std())
    m["fold_mean"] = float(err_f.mean())
    m["fold_std"] = float(err_f.std())
    m["fold_gt1bb_pct"] = float(np.mean(np.abs(err_f) > 1) * 100)
    m["fold_slope"] = float(slope)
    m["fold_resid"] = float(resid.std())
    m["a_call_med"] = float(np.median(adv[:, 1]))
    m["a_call_std"] = float(adv[:, 1].std())
    if ncols == 3:
        m["a_raise_med"] = float(np.median(A_r[:, 0]))
        m["a_raise_std"] = float(A_r[:, 0].std())
        m["a_raise_p5"] = float(_pct(A_r[:, 0], 5))
        m["a_raise_p95"] = float(_pct(A_r[:, 0], 95))
    else:
        med = np.median(A_r, 0)
        spread = A_r.max(1) - A_r.min(1)
        m["a_raise_percol_med"] = [float(x) for x in med]
        m["xcol_spread_med"] = float(np.median(spread))
        m["xcol_spread_p95"] = float(_pct(spread, 95))
    m["gap_med"] = float(np.median(gap))
    m["gap_abs_med"] = float(np.median(np.abs(gap)))
    m["gap_abs_p95"] = float(_pct(np.abs(gap), 95))
    return m


def lock_split(actor, obs, masks):
    """Ported from probe_lock_folds. Lock = hero ahead of 100% of k=2 opp
    combos on >=1 board (obs[:,991] aheadA or obs[:,994] aheadB >= .999).
    Trash = behind >=90% on both (obs[:,993] & obs[:,996] >= .9). Reports
    P(fold)/P(raise)/actor-value at locks and P(fold) at trash. Requires
    obs width >= 1020 (the outcome features live at 991..996); caller
    guards. Empty splits -> NaN (sanitized to null on JSON write)."""
    obs_t = torch.from_numpy(obs)
    masks_t = torch.from_numpy(masks)
    with torch.inference_mode():
        g, _a, _r, val = actor(obs_t, masks_t)
        gp = torch.softmax(g.float(), -1).numpy()
        v = val.float().numpy()
    lock = (obs[:, 991] >= 0.999) | (obs[:, 994] >= 0.999)
    trash = (obs[:, 993] >= 0.9) & (obs[:, 996] >= 0.9)
    n_lock = int(lock.sum())
    n_trash = int(trash.sum())
    return {
        "n": int(len(obs)),
        "n_lock": n_lock,
        "pf_lock": float(gp[lock, 0].mean() * 100) if n_lock else float("nan"),
        "pr_lock": float(gp[lock, 2].mean() * 100) if n_lock else float("nan"),
        "val_lock": float(v[lock].mean()) if n_lock else float("nan"),
        "n_trash": n_trash,
        "pf_trash": (
            float(gp[trash, 0].mean() * 100) if n_trash else float("nan")
        ),
    }


# -------------------------------------------------------------- evaluate
def _fmt_gate(m):
    return (f"[{m['gate_p_fold']:.3f} {m['gate_p_cc']:.3f} "
            f"{m['gate_p_raise']:.3f}]")


def print_family(tag, m, use_critic):
    print(f"\n### {tag}")
    if use_critic:
        print(f"V: mean {m['v_mean']:+.2f} std {m['v_std']:.2f}   "
              f"gate_p(f/c/r): {_fmt_gate(m)}")
        print(f"FOLD err (truth 0): mean {m['fold_mean']:+.3f}  "
              f"std {m['fold_std']:.3f}  |>1bb| {m['fold_gt1bb_pct']:.0f}%  "
              f"[slope {m['fold_slope']:+.2f} resid {m['fold_resid']:.3f}]")
        print(f"A_call: med {m['a_call_med']:+.2f} std {m['a_call_std']:.2f}")
        if "a_raise_med" in m:
            print(f"A_raise (pooled): med {m['a_raise_med']:+.2f} "
                  f"std {m['a_raise_std']:.2f}  p5 {m['a_raise_p5']:+.2f} "
                  f"p95 {m['a_raise_p95']:+.2f}")
        else:
            med = np.array(m["a_raise_percol_med"])
            print("A_raise per-col med: "
                  f"{np.array2string(med, precision=1, floatmode='fixed')}")
            print("same-action cross-col spread: med "
                  f"{m['xcol_spread_med']:.2f} p95 {m['xcol_spread_p95']:.2f}")
        print(f"E_pi[Q]-V: med {m['gap_med']:+.2f}  |gap| med "
              f"{m['gap_abs_med']:.2f}  p95 {m['gap_abs_p95']:.2f}")
    else:
        print(f"gate_p(f/c/r): {_fmt_gate(m)}   (actor-only: Q metrics skipped)")
    print(f"gate sharpness: max-p med {m['sharp_med']:.3f}  "
          f"entropy(legal) mean {m['gate_h_mean']:.3f}")


def _print_lock(key, tag, ls):
    print(f"LOCKS {key:7s} ({tag.split()[0]})  n={ls['n']}  "
          f"lock={ls['n_lock']}: "
          f"P(fold)={ls['pf_lock']:5.1f}%  P(raise)={ls['pr_lock']:5.1f}%  "
          f"actor-value={ls['val_lock']:+.1f}bb   | "
          f"trash={ls['n_trash']} P(fold)={ls['pf_trash']:5.1f}%")


def evaluate_checkpoint(actor, critic, banks, ckpt_name, update,
                        do_print=True):
    """Compute the full per-checkpoint record from prebuilt node banks.
    Returns the JSON record dict (raw floats; NaN sanitized only at write
    time). Pure compute + optional stdout — no file I/O, no model load, so
    the unit test can drive it with tiny in-process nets."""
    use_critic = _use_critic(critic)
    if do_print:
        print(f"\n================ {ckpt_name} (update={update}) "
              "================")
        if use_critic:
            w = critic.adv_head.weight.detach()
            rn = w.norm(dim=1)
            raise_rn = rn[2:].mean() if w.shape[0] > 2 else rn[-1]
            print(f"adv_head width {w.shape[0]}  row-norms: fold {rn[0]:.3f} "
                  f"call {rn[1]:.3f} raise {raise_rn:.3f}")
        else:
            print("actor-only critic metrics (no usable dueling head)")

    families = {}
    for tag, (obs, opp, masks) in banks.items():
        m = family_metrics(actor, critic, obs, opp, masks)
        if do_print:
            print_family(tag, m, use_critic)
        families[tag] = m

    locks = {}
    for key, tag in LOCK_TAGS.items():
        bank = banks.get(tag)
        if bank is None:
            locks[key] = None
            continue
        obs = bank[0]
        if obs.shape[1] < 1020:
            locks[key] = None  # outcome features absent; skip lock split
            continue
        ls = lock_split(actor, obs, masks=bank[2])
        locks[key] = ls

    if do_print and any(locks.values()):
        print("\n--- lock/trash splits ---")
        for key, tag in LOCK_TAGS.items():
            if locks.get(key):
                _print_lock(key, tag, locks[key])

    return {
        "ckpt": ckpt_name,
        "update": update,
        "ts": time.time(),
        "families": families,
        "locks": locks,
    }


# ------------------------------------------------------------------- I/O
def parse_update(path):
    """Trailing _<N> in the stem -> int, else None (e.g. vSix1_460.pt -> 460)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    mo = re.search(r"_(\d+)$", stem)
    return int(mo.group(1)) if mo else None


def _json_safe(o):
    """Recursively replace non-finite floats with None so the line is valid
    JSON (and re-parseable without allow_nan)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def append_jsonl(path, record):
    """Append one sanitized JSON line to the history file (parent created)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(record)) + "\n")


# -------------------------------------------------------------- summaries
def _short(record):
    up = record.get("update")
    if up is not None:
        return f"u{up}"
    return os.path.splitext(record["ckpt"])[0]


def print_summaries(records, banks):
    print("\n\n=========== GATE SUMMARY "
          "(fold-err std / |E_pi[Q]-V| p95, bb) ===========")
    for tag in banks:
        cells = []
        for rec in records:
            fam = rec["families"].get(tag, {})
            name = _short(rec)
            if "fold_std" in fam and "gap_abs_p95" in fam:
                cells.append(
                    f"{name}: {fam['fold_std']:.2f}/{fam['gap_abs_p95']:.1f}"
                )
            else:
                cells.append(f"{name}: n/a")
        print(f"{tag:34s} {'  '.join(cells)}")

    print("\n=========== LOCKS (P(fold)% / P(raise)% at locked nodes) "
          "===========")
    for key in ("deep", "shallow"):
        cells = []
        for rec in records:
            name = _short(rec)
            ls = (rec.get("locks") or {}).get(key)
            if (ls and ls.get("n_lock", 0) > 0
                    and ls.get("pf_lock") is not None
                    and math.isfinite(ls["pf_lock"])):
                cells.append(f"{name}: {ls['pf_lock']:.1f}/{ls['pr_lock']:.1f}")
            else:
                cells.append(f"{name}: n/a")
        label = f"{key} (family {'C' if key == 'deep' else 'A'})"
        print(f"{label:20s} {'  '.join(cells)}")


# -------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Per-checkpoint Q-calibration + lock-fold probe suite."
    )
    ap.add_argument("ckpts", nargs="+", help="checkpoint .pt paths")
    ap.add_argument("--fast", action="store_true",
                    help="scale every family's seed count by 0.1 (smoke)")
    ap.add_argument("--threads", type=int, default=8,
                    help="torch.set_num_threads (default 8)")
    ap.add_argument("--out", default="runs/probe_history.jsonl",
                    help="JSONL history file (one line appended per ckpt)")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))

    banks = build_banks(fast=args.fast)

    records = []
    for path in args.ckpts:
        if not os.path.exists(path):
            print(f"\n[skip] {path}: not found", file=sys.stderr)
            continue
        t0 = time.time()
        actor, critic = load_checkpoint(path)
        record = evaluate_checkpoint(
            actor, critic, banks,
            ckpt_name=os.path.basename(path),
            update=parse_update(path),
            do_print=True,
        )
        append_jsonl(args.out, record)
        records.append(record)
        print(f"\n[{os.path.basename(path)}] wall {time.time() - t0:.1f}s "
              f"-> appended to {args.out}")

    if records:
        print_summaries(records, banks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
