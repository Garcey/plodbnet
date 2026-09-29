#!/usr/bin/env python
"""Head-to-head of two checkpoints that read DIFFERENT observations -- e.g. the
live site's full-obs, obs-rev-1 vSix4 against a minimal-obs, obs-rev-2 vMin3
(2026-09-26). `h2h_eval.py` needs both to share a layout and a revision.

Same duplicate format as h2h_eval.py (every deal twice on the same cards with
the seats swapped, all-in hands paid their runout EV, table configs from the
train tiers), but each model is served its observation the way the SITE serves
it: the full observation encoded at the model's own obs-semantics revision,
then `network.obs_adapter` (a prefix slice for an older full layout, the
minimal projection for a minimal-obs model). A checkpoint without an `obs_rev`
stamp was trained on rev 1 (override with --rev-a / --rev-b).

How (2026-09-28, ML-055): the table is MIRRORED onto a Rust engine built at
each other revision the players need (every reset and action is forwarded, so
the engines hold identical states); a player whose revision is this process's
reads the table's own observations. No module global is swapped and no encoder
is replaced. `--selfcheck` first proves the mirror exact: a shadow built at
this process's revision reproduces the table's observations bit for bit.

    .venv/Scripts/python scripts/h2h_cross.py A.pt B.pt [--deals 512]
        [--configs-per-tier 6] [--greedy-a --greedy-b] [--selfcheck]
"""

from __future__ import annotations

import os

os.environ.setdefault("PLO5_RUST_ENCODER", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plo5bp import encoding as _encoding  # noqa: E402
from plo5bp import env_batched as _env_batched  # noqa: E402
from plo5bp.env_batched import BatchedBombPotEnv  # noqa: E402
from plo5bp.evaluation import load_actor  # noqa: E402
from plo5bp.evaluation.tables import (  # noqa: E402
    BB,
    TIERS,
    play_duplicate,
    sample_table,
    summarize,
    tier_spec_version,
)
from plo5bp.network import obs_adapter  # noqa: E402
from plo5bp.rollout import TRAIN_OPP_OUTCOME_MC  # noqa: E402


class _Mirror:
    """Stands in for the table's Rust engine: forwards everything, and repeats
    every reset and action on the shadow engines, so they hold the same
    states."""

    def __init__(self, be, shadows):
        self._be, self.shadows = be, list(shadows)

    def reset_batch(self, seeds, buttons, *a, **k):
        for s in self.shadows:
            s.reset_batch(seeds, buttons)
        return self._be.reset_batch(seeds, buttons, *a, **k)

    def apply_hybrid_batch(self, gates, chips, *a, **k):
        for s in self.shadows:
            s.apply_hybrid_batch(gates, chips)
        return self._be.apply_hybrid_batch(gates, chips, *a, **k)

    def __getattr__(self, name):
        return getattr(self._be, name)


def make_table(n: int, game_cfg, ev_samples: int, shadow_revs=()):
    """(the table env at this process's revision, {rev: shadow engine})."""
    env = BatchedBombPotEnv(
        n, game_cfg, ev_runout_samples=ev_samples, opp_outcome_mc=TRAIN_OPP_OUTCOME_MC,
        obs_mode="full",
    )
    stacks = np.asarray(game_cfg.resolved_stacks, dtype=np.uint64)
    shadows = {
        int(rev): _env_batched.BatchedEngine(
            n, num_seats=game_cfg.num_seats, starting_stack=0, ante=game_cfg.ante,
            bb=game_cfg.bb, starting_stacks=stacks, opp_outcome_mc=TRAIN_OPP_OUTCOME_MC,
            variant=game_cfg.variant, sb=game_cfg.sb, obs_rev=int(rev),
        )
        for rev in sorted(set(shadow_revs))
    }
    env._be = _Mirror(env._be, shadows.values())
    return env, shadows


def full_obs(env, shadows: dict, rev: int, rows: np.ndarray) -> np.ndarray:
    """The full observation rows at revision `rev`."""
    if rev not in shadows:  # this process's revision: the table's own rows
        return env._obs[rows]
    sub = shadows[rev].observation_encoded_subset_batch(np.ascontiguousarray(rows, dtype=np.int64))
    return np.asarray(sub["obs"], dtype=np.float32)


def play_config(game_cfg, players, deals, device, rng, ev_samples, greedy):
    """`players` = ((model, adapter, rev), (model, adapter, rev)) for A and B."""
    here = int(_encoding.OBS_SEMANTICS_REV)
    env, shadows = make_table(
        2 * deals, game_cfg, ev_samples,
        shadow_revs=[rev for _m, _a, rev in players if int(rev) != here],
    )

    def observe(k, env_, rows):
        _model, adapt, rev = players[k]
        return adapt(full_obs(env_, shadows, int(rev), rows))

    pair_net, _n_seats, _steps = play_duplicate(
        game_cfg, (players[0][0], players[1][0]), deals, device, rng, ev_samples,
        greedy=greedy, obs_mode="full", env=env, observe=observe,
    )
    return pair_net


def selfcheck(rng, n: int = 256, steps: int = 6) -> None:
    """A shadow built at this process's revision, mirrored, must reproduce the
    table's own observations exactly on states from several streets."""
    here = int(_encoding.OBS_SEMANTICS_REV)
    worst = 0.0
    for tier in TIERS:
        cfg = sample_table(tier, rng)
        env, shadows = make_table(n, cfg, 0, shadow_revs=(here,))
        env.reset_batch(rng.integers(0, 2**62, size=n, dtype=np.int64).astype(np.uint64),
                        rng.integers(0, cfg.num_seats, size=n).astype(np.uint8))
        for _ in range(steps):
            live = np.nonzero(~env._dones)[0]
            if live.size == 0:
                break
            got = full_obs(env, shadows, here, live)
            want = env._obs[live]
            diff = float(np.abs(want - got).max())
            worst = max(worst, diff)
            if diff > 0.0:
                bad = np.nonzero(np.abs(want - got).max(axis=0) > 0)[0]
                sys.exit(f"SELF-CHECK FAILED ({tier}): the mirrored engine differs in dims {bad[:20]}")
            g = np.where(env._gate_mask[:, 2] & (rng.random(n) < 0.3), 2,
                         np.where(env._gate_mask[:, 1], 1, 0)).astype(np.uint8)
            c = np.where(g == 2, env._min_raise, 0).astype(np.uint64)
            env.step_hybrid_batch(g, c)
    print(f"self-check OK: the mirrored rev-{here} engine reproduces the table exactly (max diff {worst:.1e})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--deals", type=int, default=512, help="deals per table config")
    ap.add_argument("--configs-per-tier", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ev-samples", type=int, default=64)
    ap.add_argument("--rev-a", type=int, default=None, help="override A's obs revision")
    ap.add_argument("--rev-b", type=int, default=None, help="override B's obs revision")
    ap.add_argument("--greedy-a", action="store_true", help="A plays its argmax (see h2h_eval.py)")
    ap.add_argument("--greedy-b", action="store_true", help="B plays its argmax too")
    ap.add_argument("--selfcheck", action="store_true", help="verify the mirrored engines first")
    ap.add_argument("--out", default="runs/h2h_cross.jsonl")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.selfcheck:
        selfcheck(np.random.default_rng(12345))
    players, metas = [], []
    for path, rev_override in ((args.a, args.rev_a), (args.b, args.rev_b)):
        model, meta = load_actor(path, device, ema=False, check_rev=False)
        rev = rev_override if rev_override is not None else int(meta["obs_rev"])
        meta["serve_rev"] = rev
        players.append((model, obs_adapter(model), rev))
        metas.append(meta)
    if metas[0]["variant"] != metas[1]["variant"]:
        sys.exit("the two checkpoints play different games")
    t0 = time.time()
    per_tier: dict[str, list[np.ndarray]] = {t: [] for t in TIERS}
    for tier in TIERS:
        for _ in range(args.configs_per_tier):
            cfg = sample_table(tier, rng, metas[0]["variant"])
            pair_net = play_config(cfg, players, args.deals, device, rng, args.ev_samples,
                                   (bool(args.greedy_a), bool(args.greedy_b)))
            per_tier[tier].append(pair_net / BB / cfg.num_seats)
    report = {"a": metas[0], "b": metas[1], "deals_per_config": args.deals,
              "configs_per_tier": args.configs_per_tier, "seed": args.seed,
              "greedy_a": bool(args.greedy_a), "greedy_b": bool(args.greedy_b),
              "tier_spec": tier_spec_version()}
    report.update(summarize(per_tier))
    report["seconds"] = round(time.time() - t0, 1)
    tag = lambda m: (f"{Path(m['path']).name} ({m['obs_mode']} obs rev {m['serve_rev']}, "
                     f"{m['hidden_dim']}x{m['num_layers']}, u{m['update']})")
    print(f"A = {tag(metas[0])}\nB = {tag(metas[1])}")
    for tier, r in report["tiers"].items():
        print(f"  {tier:12s} A edge {r['edge_bb']:+.4f} bb/seat-hand  (se {r['se']:.4f}, "
              f"config se {r['se_config']:.4f}, {r['pairs']} pairs)")
    print(f"  {'ALL':12s} A edge {report['edge_bb']:+.4f} bb/seat-hand  (se {report['se']:.4f}, "
          f"config se {report['se_config']:.4f}, "
          f"z {report['edge_bb'] / max(report['se'], 1e-12):+.2f}, {report['pairs']} pairs, {report['seconds']}s)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
