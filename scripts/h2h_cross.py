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

How: the numpy encoders read `encoding.OBS_SEMANTICS_REV` when they RUN, so
one packed table state (captured from the env's own refresh) is encoded once
per model, each under its own revision. `--selfcheck` first proves that this
equals the Rust encoder of an engine BUILT at each revision, row for row.

    .venv/Scripts/python scripts/h2h_cross.py A.pt B.pt [--deals 512]
        [--configs-per-tier 6] [--greedy-a --greedy-b] [--selfcheck]
"""

from __future__ import annotations

import os

os.environ["PLO5_RUST_ENCODER"] = "0"  # the numpy path: its packed state is what we re-encode

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2h_eval import TIERS, _train_module, load_actor  # noqa: E402

from plo5bp import encoding as _encoding  # noqa: E402
from plo5bp import env_batched as _env_batched  # noqa: E402
from plo5bp.encoding import OBS_DIM, encode_observation_batch  # noqa: E402
from plo5bp.env_batched import BatchedBombPotEnv  # noqa: E402
from plo5bp.network import obs_adapter  # noqa: E402
from plo5bp.rollout import TRAIN_OPP_OUTCOME_MC  # noqa: E402


@contextlib.contextmanager
def obs_rev(rev: int):
    old = _encoding.OBS_SEMANTICS_REV
    _encoding.OBS_SEMANTICS_REV = int(rev)
    try:
        yield
    finally:
        _encoding.OBS_SEMANTICS_REV = old


class _Capture:
    """Stands in for the env's Rust engine: forwards everything, keeps the last
    packed table state, and (self-check) mirrors resets/actions onto shadow
    engines built at other revisions."""

    def __init__(self, be, shadows=()):
        self._be, self.shadows, self.last = be, list(shadows), None

    def observation_and_features_batch(self, *a, **k):
        self.last = self._be.observation_and_features_batch(*a, **k)
        return self.last

    def reset_batch(self, seeds, buttons):
        for s in self.shadows:
            s.reset_batch(seeds, buttons)
        return self._be.reset_batch(seeds, buttons)

    def apply_hybrid_batch(self, gates, chips):
        for s in self.shadows:
            s.apply_hybrid_batch(gates, chips)
        return self._be.apply_hybrid_batch(gates, chips)

    def __getattr__(self, name):
        return getattr(self._be, name)


def _rows(bundle: dict, idx: np.ndarray, n: int) -> dict:
    out = {}
    for k, v in dict(bundle).items():
        arr = np.asarray(v)
        out[k] = arr[idx] if arr.ndim >= 1 and arr.shape[0] == n else arr
    return out


def encode_full(bundle: dict, idx: np.ndarray, n: int, config, rev: int) -> np.ndarray:
    sub = _rows(bundle, idx, n)
    with obs_rev(rev):
        return encode_observation_batch(
            sub, np.asarray(sub["hero_cat_a"]), np.asarray(sub["hero_cat_b"]), config
        )


def _no_env_encode(bundle, cat_a, cat_b, config):
    # the env's own dense rows are never read here -- skip that encode
    return np.zeros((len(np.asarray(cat_a)), OBS_DIM), dtype=np.float32)


def make_env(n: int, game_cfg, ev_samples: int, shadows_for_revs=()):
    env = BatchedBombPotEnv(
        n, game_cfg, ev_runout_samples=ev_samples, opp_outcome_mc=TRAIN_OPP_OUTCOME_MC,
        obs_mode="full",
    )
    shadows = []
    for rev in shadows_for_revs:
        stacks = np.asarray(game_cfg.resolved_stacks, dtype=np.uint64)
        shadows.append((rev, _env_batched.BatchedEngine(
            n, num_seats=game_cfg.num_seats, starting_stack=0, ante=game_cfg.ante,
            bb=game_cfg.bb, starting_stacks=stacks, opp_outcome_mc=TRAIN_OPP_OUTCOME_MC,
            variant=game_cfg.variant, sb=game_cfg.sb, obs_rev=int(rev),
        )))
    env._be = _Capture(env._be, [s for _, s in shadows])
    return env, shadows


def play_config(game_cfg, players, deals, device, rng, ev_samples, greedy):
    """`players` = ((model, adapter, rev), (model, adapter, rev)) for A and B."""
    n_seats, n = game_cfg.num_seats, 2 * deals
    env, _ = make_env(n, game_cfg, ev_samples)
    seeds = rng.integers(0, 2**63 - 1, size=deals, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, n_seats, size=deals).astype(np.uint8)
    env.reset_batch(np.concatenate([seeds, seeds]), np.concatenate([buttons, buttons]))
    offset = rng.integers(0, 2, size=deals)
    seat = np.arange(n_seats)
    a0 = ((seat[None, :] + offset[:, None]) % 2) == 0
    a_mask = np.concatenate([a0, ~a0], axis=0)
    live_at_deal = ~env._dones.copy()
    net = np.zeros(n, dtype=np.float64)
    rows_all = np.arange(n)
    while not env._dones.all():
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
        bundle = env._be.last
        for (model, adapt, rev), rows_mask, det in (
            (players[0], is_a, greedy[0]), (players[1], live & ~is_a, greedy[1])
        ):
            rows = np.nonzero(rows_mask)[0]
            if rows.size == 0:
                continue
            o = torch.from_numpy(np.ascontiguousarray(adapt(encode_full(bundle, rows, n, env.config, rev)))).to(device)
            m = torch.from_numpy(env._gate_mask[rows]).to(device)
            b = torch.from_numpy(sizing[rows]).to(device)
            with torch.inference_mode():
                out = model.act(o, m, b, deterministic=det)
            gates[rows] = out.gate.cpu().numpy().astype(np.uint8)
            chips[rows] = np.maximum(out.chips.cpu().numpy(), 0).astype(np.uint64)
        st = env.step_hybrid_batch(gates, chips)
        term = np.nonzero(st.newly_terminal)[0]
        if term.size:
            r = st.rewards[term].astype(np.float64)
            net[term] += (r * a_mask[term]).sum(axis=1)
    pair_net = net[:deals] + net[deals:]
    keep = live_at_deal[:deals] & live_at_deal[deals:]
    return pair_net[keep]


def selfcheck(train, rng, n: int = 256, steps: int = 6) -> None:
    """The re-encode under a flipped revision == the Rust encoder of an engine
    BUILT at that revision, on states from several streets (both revisions)."""
    worst = 0.0
    for tier in TIERS:
        cfg, _ = train._sample_game_config(
            (2, 3, 4, 5, 6), 1.0, 300.0, 10_000, 30_000, rng,
            stack_dist=tier, seats_dist="uniform", variant="plo5_double_bomb", sb=0,
        )
        env, shadows = make_env(n, cfg, 0, shadows_for_revs=(1, 2))
        env.reset_batch(rng.integers(0, 2**62, size=n, dtype=np.int64).astype(np.uint64),
                        rng.integers(0, cfg.num_seats, size=n).astype(np.uint8))
        for _ in range(steps):
            live = np.nonzero(~env._dones)[0]
            if live.size == 0:
                break
            for rev, eng in shadows:
                want = np.asarray(eng.observation_encoded_batch()["obs"], dtype=np.float32)[live]
                got = encode_full(env._be.last, live, n, env.config, rev)
                diff = float(np.abs(want - got).max()) if live.size else 0.0
                worst = max(worst, diff)
                if diff > 1e-5:
                    bad = np.nonzero(np.abs(want - got).max(axis=0) > 1e-5)[0]
                    sys.exit(f"SELF-CHECK FAILED ({tier}, rev {rev}): max diff {diff} in dims {bad[:20]}")
            # random legal actions (fold rarely, raise sometimes)
            g = np.where(env._gate_mask[:, 2] & (rng.random(n) < 0.3), 2,
                         np.where(env._gate_mask[:, 1], 1, 0)).astype(np.uint8)
            c = np.where(g == 2, env._min_raise, 0).astype(np.uint64)
            env.step_hybrid_batch(g, c)
    print(f"self-check OK: re-encode == Rust encoder built at rev 1 and rev 2 (max diff {worst:.2e})")


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
    ap.add_argument("--selfcheck", action="store_true", help="verify the per-revision re-encode first")
    ap.add_argument("--out", default="runs/h2h_cross.jsonl")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train = _train_module()
    _env_batched.encode_observation_batch = _no_env_encode
    if args.selfcheck:
        selfcheck(train, np.random.default_rng(12345))
    players, metas = [], []
    for path, rev_override in ((args.a, args.rev_a), (args.b, args.rev_b)):
        model, meta = load_actor(path, device, ema=False)
        rev = rev_override if rev_override is not None else int(meta["obs_rev"] or 1)
        meta["serve_rev"] = rev
        players.append((model, obs_adapter(model), rev))
        metas.append(meta)
    if metas[0]["variant"] != metas[1]["variant"]:
        sys.exit("the two checkpoints play different games")
    bb, ante = 10_000, 30_000
    t0 = time.time()
    per_tier: dict[str, list[np.ndarray]] = {t: [] for t in TIERS}
    for tier in TIERS:
        for _ in range(args.configs_per_tier):
            cfg, _ = train._sample_game_config(
                (2, 3, 4, 5, 6), 1.0, 300.0, bb, ante, rng,
                stack_dist=tier, seats_dist="uniform", variant=metas[0]["variant"], sb=0,
            )
            pair_net = play_config(cfg, players, args.deals, device, rng, args.ev_samples,
                                   (bool(args.greedy_a), bool(args.greedy_b)))
            per_tier[tier].append(pair_net / bb / cfg.num_seats)
    report = {"a": metas[0], "b": metas[1], "deals_per_config": args.deals,
              "configs_per_tier": args.configs_per_tier, "seed": args.seed,
              "greedy_a": bool(args.greedy_a), "greedy_b": bool(args.greedy_b), "tiers": {}}
    everything = []
    for tier in TIERS:
        x = np.concatenate(per_tier[tier]) if per_tier[tier] else np.zeros(0)
        everything.append(x)
        if x.size:
            report["tiers"][tier] = {"edge_bb": float(x.mean()),
                                     "se": float(x.std(ddof=1) / np.sqrt(x.size)), "pairs": int(x.size)}
    x = np.concatenate(everything)
    report.update(edge_bb=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(x.size)),
                  pairs=int(x.size), seconds=round(time.time() - t0, 1))
    tag = lambda m: (f"{Path(m['path']).name} ({m['obs_mode']} obs rev {m['serve_rev']}, "
                     f"{m['hidden_dim']}x{m['num_layers']}, u{m['update']})")
    print(f"A = {tag(metas[0])}\nB = {tag(metas[1])}")
    for tier, r in report["tiers"].items():
        print(f"  {tier:12s} A edge {r['edge_bb']:+.4f} bb/seat-hand  (se {r['se']:.4f}, {r['pairs']} pairs)")
    print(f"  {'ALL':12s} A edge {report['edge_bb']:+.4f} bb/seat-hand  (se {report['se']:.4f}, "
          f"z {report['edge_bb'] / max(report['se'], 1e-12):+.2f}, {report['pairs']} pairs, {report['seconds']}s)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
