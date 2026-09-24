#!/usr/bin/env python
"""Head-to-head strength of two checkpoints -- the network-size sweep's
strength test (2026-09-23).

Duplicate format, batched: every deal is played TWICE on the same cards and
button with the seats swapped -- each seat belongs to A in exactly one of
the two passes -- and every hand that ends all-in before the river pays its
64-runout Monte-Carlo EV, so neither card luck nor runout luck is left in the
comparison, only the two policies (each sampling its own mixed strategy, as
in training). Table configs come from the same tiers train.py mixes
(`_sample_game_config`: seats 2-6, clubgg / clubgg_deep / deep stacks).

Reported: A's edge in bb per seat-hand (both passes, A's seats summed,
divided by the seats played), its standard error over deal pairs, and the
same per tier. Positive = A stronger. Both checkpoints must read the same
observation layout (obs_mode) and the process's PLO5BP_OBS_REV must match
their stamped obs_rev.

    .venv/bin/python scripts/h2h_eval.py A.pt B.pt [--deals 4096]
        [--configs-per-tier 10] [--device cuda] [--seed 0] [--ema]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PLO5_RUST_ENCODER", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plo5bp import encoding as _encoding  # noqa: E402
from plo5bp.env_batched import BatchedBombPotEnv  # noqa: E402
from plo5bp.network import build_actor_from_state_dict  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TIERS = ("clubgg", "clubgg_deep", "deep")


def _train_module():
    spec = importlib.util.spec_from_file_location("_train", REPO / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module level only defines things
    return mod


def load_actor(path: str, device: torch.device, ema: bool):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    sd = ckpt["model_ema"] if (ema and ckpt.get("model_ema")) else ckpt["model"]
    model = build_actor_from_state_dict(
        sd, int(cfg["hidden_dim"]), int(cfg["num_layers"])
    ).to(device).eval()
    meta = {
        "path": path,
        "hidden_dim": int(cfg["hidden_dim"]),
        "num_layers": int(cfg["num_layers"]),
        "obs_mode": str(cfg.get("obs_mode", "full")),
        "obs_rev": ckpt.get("obs_rev"),
        "update": ckpt.get("update_counter", ckpt.get("update")),
        "variant": ckpt.get("variant", "plo5_double_bomb"),
    }
    return model, meta


def play_config(game_cfg, models, deals, obs_mode, device, rng, ev_samples, greedy=(False, False)):
    """One table config: `deals` deals x 2 passes. Returns per-pair A net
    (chips, summed over A's seats in both passes) and seats played."""
    n_seats = game_cfg.num_seats
    n = 2 * deals
    env = BatchedBombPotEnv(
        n, game_cfg, ev_runout_samples=ev_samples, opp_outcome_mc=0, obs_mode=obs_mode
    )
    seeds = rng.integers(0, 2**63 - 1, size=deals, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, n_seats, size=deals).astype(np.uint8)
    env.reset_batch(np.concatenate([seeds, seeds]), np.concatenate([buttons, buttons]))
    # Pass 0: A on alternating seats (random per-deal offset); pass 1: the rest.
    offset = rng.integers(0, 2, size=deals)
    seat = np.arange(n_seats)
    a0 = ((seat[None, :] + offset[:, None]) % 2) == 0          # (deals, S)
    a_mask = np.concatenate([a0, ~a0], axis=0)                  # (n, S)
    live_at_deal = ~env._dones.copy()
    net = np.zeros(n, dtype=np.float64)
    rows_all = np.arange(n)
    steps = 0
    while not env._dones.all():
        steps += 1
        actors = env._actors
        live = ~env._dones
        safe = np.where(actors >= 0, actors, 0).astype(np.intp)
        is_a = live & a_mask[rows_all, safe]
        to_call = np.maximum(
            env._bet_to_call.astype(np.int64)
            - env._street_commit[rows_all, safe].astype(np.int64), 0,
        )
        sizing = np.stack(
            [env._min_raise.astype(np.int64), env._max_raise.astype(np.int64),
             env._pot.astype(np.int64), to_call], axis=-1,
        )
        gates = np.zeros(n, dtype=np.uint8)
        chips = np.zeros(n, dtype=np.uint64)
        for model, rows_mask, det in (
            (models[0], is_a, greedy[0]), (models[1], live & ~is_a, greedy[1])
        ):
            rows = np.nonzero(rows_mask)[0]
            if rows.size == 0:
                continue
            o = torch.from_numpy(env._obs[rows]).to(device)
            m = torch.from_numpy(env._gate_mask[rows]).to(device)
            b = torch.from_numpy(sizing[rows]).to(device)
            with torch.inference_mode():
                out = model.act(o, m, b, deterministic=det)
            gates[rows] = out.gate.cpu().numpy().astype(np.uint8)
            chips[rows] = np.maximum(out.chips.cpu().numpy(), 0).astype(np.uint64)
        st = env.step_hybrid_batch(gates, chips)
        term = np.nonzero(st.newly_terminal)[0]
        if term.size:
            r = st.rewards[term].astype(np.float64)               # chip deltas
            net[term] += (r * a_mask[term]).sum(axis=1)
    pair_net = net[:deals] + net[deals:]
    keep = live_at_deal[:deals] & live_at_deal[deals:]
    return pair_net[keep], n_seats, steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--deals", type=int, default=4096, help="deals per table config")
    ap.add_argument("--configs-per-tier", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ema", action="store_true", help="play the EMA actors")
    ap.add_argument("--ev-samples", type=int, default=64)
    ap.add_argument("--out", default="runs/h2h_history.jsonl")
    ap.add_argument(
        "--greedy-a", action="store_true",
        help="A plays its most likely action (argmax gate / sizing mode) instead of "
        "sampling: what A has LEARNED to prefer, apart from how much it still mixes "
        "(compare runs trained at different entropy coefficients this way)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    model_a, meta_a = load_actor(args.a, device, args.ema)
    model_b, meta_b = load_actor(args.b, device, args.ema)
    for meta in (meta_a, meta_b):
        rev = meta["obs_rev"]
        if rev is not None and int(rev) != int(_encoding.OBS_SEMANTICS_REV):
            sys.exit(
                f"{meta['path']}: trained on obs rev {rev}, this process encodes rev "
                f"{_encoding.OBS_SEMANTICS_REV} -- set PLO5BP_OBS_REV={rev}"
            )
    if meta_a["obs_mode"] != meta_b["obs_mode"]:
        sys.exit(f"obs_mode differs: {meta_a['obs_mode']} vs {meta_b['obs_mode']}")
    obs_mode = meta_a["obs_mode"]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train = _train_module()
    bb, ante = 10_000, 30_000
    t0 = time.time()
    per_tier: dict[str, list[np.ndarray]] = {t: [] for t in TIERS}
    seats_played: dict[str, list[int]] = {t: [] for t in TIERS}
    for tier in TIERS:
        for _ in range(args.configs_per_tier):
            cfg, _ = train._sample_game_config(
                (2, 3, 4, 5, 6), 1.0, 300.0, bb, ante, rng,
                stack_dist=tier, seats_dist="uniform", variant=meta_a["variant"], sb=0,
            )
            pair_net, n_seats, _steps = play_config(
                cfg, (model_a, model_b), args.deals, obs_mode, device, rng, args.ev_samples,
                greedy=(bool(args.greedy_a), False),
            )
            per_tier[tier].append(pair_net / bb / n_seats)   # bb per A seat-hand
    report = {"a": meta_a, "b": meta_b, "deals_per_config": args.deals,
              "configs_per_tier": args.configs_per_tier, "seed": args.seed,
              "ema": bool(args.ema), "greedy_a": bool(args.greedy_a), "tiers": {}}
    everything = []
    for tier in TIERS:
        x = np.concatenate(per_tier[tier]) if per_tier[tier] else np.zeros(0)
        everything.append(x)
        if x.size:
            report["tiers"][tier] = {
                "edge_bb": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(x.size)),
                "pairs": int(x.size),
            }
    x = np.concatenate(everything)
    report["edge_bb"] = float(x.mean())
    report["se"] = float(x.std(ddof=1) / np.sqrt(x.size))
    report["pairs"] = int(x.size)
    report["seconds"] = round(time.time() - t0, 1)
    tag = lambda m: f"{Path(m['path']).name} ({m['hidden_dim']}x{m['num_layers']}, u{m['update']})"
    print(f"A = {tag(meta_a)}\nB = {tag(meta_b)}")
    for tier, r in report["tiers"].items():
        print(f"  {tier:12s} A edge {r['edge_bb']:+.4f} bb/seat-hand  (se {r['se']:.4f}, {r['pairs']} pairs)")
    print(f"  {'ALL':12s} A edge {report['edge_bb']:+.4f} bb/seat-hand  (se {report['se']:.4f}, "
          f"z {report['edge_bb'] / max(report['se'], 1e-12):+.2f}, {report['pairs']} pairs, {report['seconds']}s)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
