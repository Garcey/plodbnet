"""PPO training driver.

Two budget modes:
  - `--num-updates N` (default): stop after N PPO updates.
  - `--train-seconds S` (takes precedence when > 0): stop once wall-clock
    elapsed exceeds S seconds. Intended for time-boxed runs where we'd
    rather measure cost as "5 hours" than as "how many updates".

Heterogeneous configs (sampled per rollout so every batch stays
shape-homogeneous — seats and stacks vary across rollouts, not across
envs within a rollout):
  - `--num-seats-range "2,3,4,5,6"` — uniform choice per rollout.
  - `--stack-range "10:200"` — per-seat uniform(min, max) in bb each
    rollout; converted to chips via `round(depth * bb)`.

Time-based persistence (coexist with update-count flags):
  - `--snapshot-every-sec S` pushes to the opponent pool every S seconds.
  - `--checkpoint-every-sec S` saves a mid-run `<stem>_<updates>.pt`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from pathlib import Path

import numpy as np
import torch

from plo5bp.actions import GATE_ACTIONS
from plo5bp.config import (
    VARIANT_NLH,
    VARIANT_PLO4,
    VARIANT_PLO5,
    VARIANT_PLO6,
    GameConfig,
    TrainingConfig,
)
from plo5bp.encoding import OBS_DIM
from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.network import ActorCriticV2, ActorCriticV4, CentralCritic
from plo5bp.sizing import NLH_ANCHOR_SPEC, PLO_ANCHOR_SPEC
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import (
    collect_rollout,
    collect_rollout_batched,
    collect_rollout_multiconfig,
)
from plo5bp.selfplay import (
    OpponentPool,
    discover_checkpoint_family,
    seed_pool_from_checkpoints,
)


def _parse_seats_range(spec: str) -> tuple[int, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out = tuple(int(p) for p in parts)
    if not out or any(n < 2 for n in out):
        raise SystemExit(f"--num-seats-range must list ints ≥ 2, got {spec!r}")
    return out


def _lr_warmup_scale(update: int, warmup_updates: int) -> float:
    """Linear LR ramp over the first `warmup_updates` updates: scale
    runs from 1/warmup_updates up to 1.0, then stays at 1.0. 0 disables
    (always 1.0). Pure function of the global update index, so resumes
    are deterministic."""
    if warmup_updates <= 0 or update >= warmup_updates:
        return 1.0
    return (update + 1) / warmup_updates


def _parse_stack_range(spec: str) -> tuple[float, float]:
    if ":" not in spec:
        raise SystemExit(f"--stack-range must be 'min:max' in bb, got {spec!r}")
    lo_s, hi_s = spec.split(":", 1)
    lo, hi = float(lo_s), float(hi_s)
    if lo <= 0 or hi < lo:
        raise SystemExit(f"invalid --stack-range {spec!r}")
    return lo, hi


_VALID_STACK_DISTS = (
    "uniform", "clubgg", "clubgg_deep", "clubgg_mix",
    "agro_deep", "deep", "full_mix",
)


def _parse_block_rotation(spec: str) -> list[tuple[str, float]]:
    if not spec:
        return []
    blocks: list[tuple[str, float]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise SystemExit(
                f"--block-rotation token must be 'tier:ent_coef', got {token!r}"
            )
        tier, ent_str = token.split(":", 1)
        tier = tier.strip()
        if tier not in _VALID_STACK_DISTS:
            raise SystemExit(
                f"--block-rotation tier {tier!r} not in {_VALID_STACK_DISTS}"
            )
        blocks.append((tier, float(ent_str)))
    if not blocks:
        raise SystemExit(f"--block-rotation parsed to empty list from {spec!r}")
    return blocks


# Training always grades early all-ins by EXPECTED value over board
# runouts (engine `payouts_ev`) instead of the one sampled runout —
# unconditional, not a flag: there is no training regime where realized
# runout luck in the reward is preferable. 64 samples cuts runout
# variance ~64x; profiled at ~+45% ENGINE time in a 30%-shove stress
# test (a few percent of real update time, where the network dominates).
# Fold-outs and river-closes short-circuit to exact payouts in Rust.
# The TrainingConfig default stays 0 so UI/eval/parity paths keep
# realized payouts.
EV_RUNOUT_SAMPLES = 64


def _anneal_due(update: int, block_size: int, start_update: int) -> bool:
    """Whether the block ending at `update` should run an anneal decision.

    Blocks that finish at or before `start_update` are warmup — the
    strategy gets time to converge before any baseline is recorded or
    any entropy is lowered."""
    return (update + 1) % block_size == 0 and (update + 1) > start_update


def _apply_anneal_control(
    raw: str | None,
    last_raw: str | None,
    tier_ent: dict[str, float],
    step: float,
    live_lr: float,
    live_ent: float,
    live_ent_deep: float,
    trainer=None,
) -> tuple[float, str | None, float, float, float]:
    """Apply a live `runs/anneal_control.json` edit without pausing
    training. Returns (anneal_step, applied_content, live_lr, live_ent,
    live_ent_deep); mutates `tier_ent` in place (and `trainer.target_kl`
    / `trainer.kl_hard` when given). Read for EVERY run since 2026-07-04
    (previously block-rotation/mix-configs only — NLH runs needed a
    restart per entropy step). Re-applies only when the file CONTENT
    changes:

      {"step": 0.003}                     — change the per-block decrement
      {"tier_ent": {"deep": 0.08}}        — manually set a tier's coef
      {"entropy_coef": 0.38}              — FLAT coef (non-tier runs: NLH /
                                            plain --stack-dist; ignored by
                                            the tiered branches)
      {"entropy_coef_deep": 0.1}          — the deep-dist flat variant
      {"target_kl": 2.0}                  — retune the soft KL early-stop
      {"kl_hard": 12.0}                   — retune the hard rollback level
      {"lr": 1e-4}                        — retune the base learning rate
      {"sizing_entropy_scale": 2.5}       — scale the sizing-head entropy
      {"step": 0.003, "tier_ent": {...}}  — any combination

    A manual tier_ent set is one-shot: the anneal keeps lowering from
    the new level afterwards. The lr set is the BASE lr — the per-update
    warmup scale still multiplies it. Malformed JSON is ignored (and
    retried on the next loop, so a half-written save is harmless)."""
    if raw is None or raw == last_raw:
        return step, last_raw, live_lr, live_ent, live_ent_deep
    try:
        ctrl = json.loads(raw)
        new_step = float(ctrl["step"]) if "step" in ctrl else step
        new_tiers = {
            tier: float(v)
            for tier, v in (ctrl.get("tier_ent") or {}).items()
            if tier in tier_ent
        }
        new_target_kl = (
            float(ctrl["target_kl"]) if "target_kl" in ctrl else None
        )
        new_kl_hard = float(ctrl["kl_hard"]) if "kl_hard" in ctrl else None
        new_lr = float(ctrl["lr"]) if "lr" in ctrl else None
        new_sizing_scale = (
            float(ctrl["sizing_entropy_scale"])
            if "sizing_entropy_scale" in ctrl
            else None
        )
        new_ent = (
            float(ctrl["entropy_coef"]) if "entropy_coef" in ctrl else None
        )
        new_ent_deep = (
            float(ctrl["entropy_coef_deep"])
            if "entropy_coef_deep" in ctrl
            else None
        )
    except (ValueError, TypeError):
        return step, last_raw, live_lr, live_ent, live_ent_deep
    if new_step != step:
        print(f"[anneal-control] step {step} -> {new_step}")
    for tier, v in new_tiers.items():
        if tier_ent[tier] != v:
            print(f"[anneal-control] tier_ent[{tier}] {tier_ent[tier]} -> {v}")
        tier_ent[tier] = v
    if new_target_kl is not None and trainer is not None:
        if trainer.target_kl != new_target_kl:
            print(
                f"[anneal-control] target_kl {trainer.target_kl} -> {new_target_kl}"
            )
        trainer.target_kl = new_target_kl
    if new_kl_hard is not None and trainer is not None:
        if trainer.kl_hard != new_kl_hard:
            print(f"[anneal-control] kl_hard {trainer.kl_hard} -> {new_kl_hard}")
        trainer.kl_hard = new_kl_hard
    if new_sizing_scale is not None and trainer is not None:
        if trainer.sizing_entropy_scale != new_sizing_scale:
            print(
                "[anneal-control] sizing_entropy_scale "
                f"{trainer.sizing_entropy_scale} -> {new_sizing_scale}"
            )
        trainer.sizing_entropy_scale = new_sizing_scale
    out_lr = live_lr
    if new_lr is not None:
        if live_lr != new_lr:
            print(f"[anneal-control] lr {live_lr} -> {new_lr}")
        out_lr = new_lr
    out_ent = live_ent
    if new_ent is not None:
        if live_ent != new_ent:
            print(f"[anneal-control] entropy_coef {live_ent} -> {new_ent}")
        out_ent = new_ent
    out_ent_deep = live_ent_deep
    if new_ent_deep is not None:
        if live_ent_deep != new_ent_deep:
            print(
                f"[anneal-control] entropy_coef_deep {live_ent_deep} -> {new_ent_deep}"
            )
        out_ent_deep = new_ent_deep
    return new_step, raw, out_lr, out_ent, out_ent_deep


def _anneal_decision(
    now_ftr: tuple[float, float, float],
    baseline: tuple[float, float, float] | None,
    ent: float,
    step: float,
    floor: float,
    tol: float,
) -> tuple[float, tuple[float, float, float] | None, str]:
    """Decide a tier's next entropy coef from this block's F/T/R vs its baseline.

    Pure function (no I/O) so it is unit-testable. `now_ftr`/`baseline` are
    (flop%, turn%, river%) aggression rates. Returns
    ``(new_ent, new_baseline, action)``:

      - No baseline yet (first block of this tier): record it, leave ent.
      - All three streets HELD within `tol` (each >= baseline - tol): lower ent
        by `step` (clamped at `floor`) and ADVANCE the baseline to now_ftr — so
        each successive cut must keep paying for the aggression it had.
      - Any street DROPPED: HOLD ent and KEEP the old baseline. The next block
        must recover to the pre-drop level before lowering resumes; this stops
        the anneal from chasing F/T/R downward into passivity.
    """
    if baseline is None:
        return ent, (now_ftr[0], now_ftr[1], now_ftr[2]), "record-baseline"
    held = all(now_ftr[s] >= baseline[s] - tol for s in range(3))
    if held:
        if ent > floor:
            new_ent = max(floor, ent - step)
            return new_ent, (now_ftr[0], now_ftr[1], now_ftr[2]), "lowered"
        return floor, (now_ftr[0], now_ftr[1], now_ftr[2]), "held@floor"
    return ent, baseline, "drop:hold"


# ClubGG-realistic per-seat stack bands (bb). Weights sum to 1.
# Reflects table conditions at $20/bb: most stacks hover 20-40bb after
# a few orbits; deep stacks (75bb+) present in ~50% of hands by
# independent per-seat sampling.
_CLUBGG_STACK_BANDS: tuple[tuple[float, float, float], ...] = (
    (1.0, 20.0, 0.05),    # Short: 1-20 bb
    (20.0, 40.0, 0.50),   # Hover: 20-40 bb (dominant)
    (40.0, 75.0, 0.18),   # Warm: 40-75 bb
    (75.0, 150.0, 0.17),  # Big: 75-150 bb
    (150.0, 300.0, 0.10), # Monster: 150-300 bb
)

# ClubGG "deep" per-seat stack bands (bb). Weights sum to 1.
# Models the $80-buy-in / $0.80-ante game, ~2x deeper than the $0.60
# game. Probability concentrated on 30-65bb (63%); 65-80bb seats
# expected ~1.3 per 6-handed table; minimal weight on <20bb; small
# 20-30bb tail for seats that lost a few hands without auto top-up.
_CLUBGG_DEEP_STACK_BANDS: tuple[tuple[float, float, float], ...] = (
    (1.0, 20.0, 0.02),    # Short: 1-20 bb
    (20.0, 30.0, 0.06),   # Lost-a-few: 20-30 bb
    (30.0, 40.0, 0.16),
    (40.0, 50.0, 0.22),
    (50.0, 65.0, 0.25),   # Mode
    (65.0, 80.0, 0.22),
    (80.0, 120.0, 0.07),
)

# ClubGG-realistic seat-count weights.
_CLUBGG_SEAT_WEIGHTS: dict[int, float] = {
    6: 0.30,
    5: 0.25,
    4: 0.25,
    3: 0.15,
    2: 0.10,
}

# NLH ring-game seat weights (user-described 2026-07-04): "slightly more
# emphasis on 5-6 handed, the rest split evenly" — 5/6 get 1.25x the
# 2/3/4 weight (≈22.7% each vs ≈18.2% each after normalization).
_NLH_RING_SEAT_WEIGHTS: dict[int, float] = {
    6: 1.25,
    5: 1.25,
    4: 1.0,
    3: 1.0,
    2: 1.0,
}


def _sample_nlh_topoff_stack_bb(rng: np.random.Generator) -> float:
    """Per-seat stack depth for the live 5/10($5) NLH table's top-off
    culture (user-described 2026-07-04): most players auto top off to
    100bb, so hand-start stacks cluster there; 1-2 (occasionally 3) of
    ~6 seats sit below 100bb (non-topped, stuck); the rest drift
    100-150bb; 1-2 winners hold 150-400bb ($1.5k-4k at $10/bb).

    Mixture: 40% pinned at exactly 100bb, 25% short Uniform(30, 100),
    20% Uniform(100, 150), 15% Uniform(150, 400). At 6 seats that's
    ≈1.5 short / ≈3.6 at-or-near 100-150 / ≈0.9 deep — matching the
    described table.
    """
    r = rng.random()
    if r < 0.40:
        return 100.0
    if r < 0.65:
        return float(rng.uniform(30.0, 100.0))
    if r < 0.85:
        return float(rng.uniform(100.0, 150.0))
    return float(rng.uniform(150.0, 400.0))


def _sample_clubgg_stack_bb(
    stack_lo_bb: float,
    stack_hi_bb: float,
    rng: np.random.Generator,
    bands: tuple[tuple[float, float, float], ...] = _CLUBGG_STACK_BANDS,
) -> float:
    # Pick a band by weight, then uniform within the band. Bands are
    # clipped to the [stack_lo_bb, stack_hi_bb] range; bands that fall
    # entirely outside the range contribute zero weight.
    weights = []
    ranges = []
    for lo, hi, w in bands:
        c_lo = max(lo, stack_lo_bb)
        c_hi = min(hi, stack_hi_bb)
        if c_hi > c_lo:
            weights.append(w)
            ranges.append((c_lo, c_hi))
    if not weights:
        return stack_lo_bb
    total = sum(weights)
    probs = [w / total for w in weights]
    idx = int(rng.choice(len(ranges), p=probs))
    lo, hi = ranges[idx]
    return float(rng.uniform(lo, hi))


def _sample_clubgg_seats(
    seats_choices: tuple[int, ...],
    rng: np.random.Generator,
    weights: dict[int, float] | None = None,
) -> int:
    # Restrict to the intersection of the weight table and user-supplied
    # seat range; renormalize. Seats not in the table fall back to
    # uniform probability across the remaining weighted seats so we
    # never silently drop them. Default table = ClubGG PLO weights;
    # `nlh_ring` passes its own.
    table = _CLUBGG_SEAT_WEIGHTS if weights is None else weights
    weights_l = [table.get(n, 0.0) for n in seats_choices]
    total = sum(weights_l)
    if total <= 0.0:
        return int(rng.choice(seats_choices))
    probs = [w / total for w in weights_l]
    return int(rng.choice(seats_choices, p=probs))


def _sample_game_config(
    seats_choices: tuple[int, ...],
    stack_lo_bb: float,
    stack_hi_bb: float,
    bb: int,
    ante: int,
    rng: np.random.Generator,
    stack_dist: str = "uniform",
    seats_dist: str = "uniform",
    variant: str = VARIANT_PLO5,
    sb: int = 0,
) -> tuple[GameConfig, str]:
    if seats_dist == "clubgg":
        n_seats = _sample_clubgg_seats(seats_choices, rng)
    elif seats_dist == "nlh_ring":
        n_seats = _sample_clubgg_seats(
            seats_choices, rng, weights=_NLH_RING_SEAT_WEIGHTS
        )
    else:
        n_seats = int(rng.choice(seats_choices))

    effective_stack_dist = stack_dist
    if stack_dist == "clubgg_mix":
        effective_stack_dist = "clubgg_deep" if rng.random() < 0.5 else "clubgg"
    elif stack_dist == "full_mix":
        effective_stack_dist = str(
            rng.choice(("clubgg", "clubgg_deep", "deep"))
        )

    if effective_stack_dist == "clubgg":
        depths_bb = np.array(
            [_sample_clubgg_stack_bb(stack_lo_bb, stack_hi_bb, rng) for _ in range(n_seats)]
        )
    elif effective_stack_dist == "clubgg_deep":
        depths_bb = np.array(
            [
                _sample_clubgg_stack_bb(
                    stack_lo_bb, stack_hi_bb, rng, bands=_CLUBGG_DEEP_STACK_BANDS
                )
                for _ in range(n_seats)
            ]
        )
    elif effective_stack_dist == "nlh_topoff":
        depths_bb = np.array(
            [_sample_nlh_topoff_stack_bb(rng) for _ in range(n_seats)]
        )
    elif effective_stack_dist in ("agro_deep", "deep"):
        depths_bb = rng.uniform(100.0, 250.0, size=n_seats)
    elif stack_lo_bb == stack_hi_bb:
        depths_bb = np.full(n_seats, stack_lo_bb)
    else:
        depths_bb = rng.uniform(stack_lo_bb, stack_hi_bb, size=n_seats)
    stacks = tuple(int(round(float(d) * bb)) for d in depths_bb)
    cfg = GameConfig(
        num_seats=n_seats,
        starting_stack=stacks[0],
        ante=ante,
        bb=bb,
        starting_stacks=stacks,
        sb=sb,
        variant=variant,
    )
    return cfg, effective_stack_dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-updates", type=int, default=100_000_000)
    parser.add_argument(
        "--train-seconds",
        type=float,
        default=0.0,
        help="Wall-clock budget in seconds; overrides --num-updates when > 0.",
    )
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--sizing-head",
        choices=["anchor", "logistic"],
        default="anchor",
        help="Sizing-head architecture. 'anchor' = v2 flat 11-way categorical "
        "(head_version 2). 'logistic' = v4 ordinal discretized-logistic over the "
        "same 11 anchors (head_version 3): location+scale, stable under PPO, with "
        "the min/pot end anchors tail-absorbed so they stay hittable.",
    )
    parser.add_argument("--num-envs", type=int, default=1536)
    parser.add_argument("--rollout-length", type=int, default=262_144)
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=32,
        help="PPO minibatches per epoch. batch_size is derived as "
        "ceil(rollout_length / num_minibatches) when --batch-size is not set. "
        "Default 32 auto-scales across rollout sizes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="PPO minibatch size override. When unset, derived from "
        "--num-minibatches and --rollout-length.",
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument(
        "--snapshot-every-sec",
        type=float,
        default=0.0,
        help="If > 0, push opponent-pool snapshots on this wall-clock cadence "
        "(coexists with --snapshot-every).",
    )
    parser.add_argument("--pool-mix-prob", type=float, default=0.5)
    parser.add_argument("--pool-opp-seats", type=int, default=2)
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.1,
        help="PPO entropy bonus coefficient. Default 0.1 to keep the "
        "Beta raise-size head from saturating on the 2048x4 architecture.",
    )
    parser.add_argument(
        "--entropy-coef-deep",
        type=float,
        default=None,
        help="Optional per-tier entropy coefficient applied only when the "
        "rollout's effective stack distribution is 'deep' (very-deep "
        "100-250bb). Defaults to --entropy-coef when unset. Active only "
        "with --stack-dist full_mix or deep, where the deep tier shows "
        "low H under the global 0.1 default.",
    )
    parser.add_argument(
        "--aggression-bonus-c",
        type=float,
        default=None,
        help="Pot-fraction aggression bonus coefficient (bb units). Each "
        "GATE_RAISE step (which now also covers stack-bound short shoves) "
        "adds c * min(1, aggressive_chips / pre_step_pot) to the "
        "forward-EV per-step reward. 0.0 disables the bonus. Default 0.0, "
        "except --stack-dist agro_deep defaults to 5.0.",
    )
    parser.add_argument(
        "--retroactive-bonus-c",
        type=float,
        default=0.0,
        help="Retroactive aggression bonus coefficient (bb of bonus per "
        "bb of pot). At end-of-hand, each learner-seat trajectory "
        "receives `c * pot_bb_at_decision` on qualifying steps based "
        "on hero pot share: >50%% → GATE_RAISE only; ==50%% → "
        "GATE_RAISE + GATE_CHECK_CALL with chips>0; <50%% → no bonus. "
        "Folds and pure checks never get bonus. Pot-relative scaling "
        "makes the bonus louder on big-pot streets (river) and "
        "quieter on small-pot streets (flop). Independent of "
        "--aggression-bonus-c. 0.0 disables.",
    )
    parser.add_argument(
        "--num-seats-range",
        type=str,
        default="2,3,4,5,6",
        help='Comma list of seat counts to sample per rollout, e.g. "2,3,4,5,6".',
    )
    parser.add_argument(
        "--stack-range",
        type=str,
        default="1:300",
        help='Per-seat stack range in bb, "min:max"; each seat sampled '
        "uniformly within the range every rollout.",
    )
    parser.add_argument(
        "--stack-dist",
        type=str,
        choices=(
            "uniform", "clubgg", "clubgg_deep", "clubgg_mix", "agro_deep",
            "deep", "full_mix", "nlh_topoff",
        ),
        default="uniform",
        help="'uniform' samples within --stack-range; 'clubgg' uses piecewise "
        "weighted bands (Short 5%%, Hover 50%%, Warm 18%%, Big 17%%, Monster 10%%) "
        "clipped to --stack-range; 'clubgg_deep' targets the $0.80-ante game "
        "(1-20:2%%, 20-30:6%%, 30-40:16%%, 40-50:22%%, 50-65:25%%, 65-80:22%%, 80-120:7%%); "
        "'clubgg_mix' picks 50/50 between clubgg and clubgg_deep per config; "
        "'agro_deep' samples each seat uniformly in 100-250bb (ignores --stack-range) "
        "and defaults --aggression-bonus-c to 5.0; "
        "'deep' is identical sampling to agro_deep but doesn't auto-set the bonus; "
        "'full_mix' picks 1/3 each between clubgg, clubgg_deep, and deep per config.",
    )
    parser.add_argument(
        "--block-rotation",
        type=str,
        default="clubgg:0.1,clubgg_deep:0.1,deep:0.2",
        help="If set, rotates between (tier, entropy_coef) blocks every "
        "--block-size updates instead of sampling per --stack-dist. "
        "Format: 'clubgg:0.1,clubgg_deep:0.1,deep:0.2'. Pool, optimizer, "
        "and model state persist across blocks. Overrides --stack-dist "
        "and --entropy-coef/--entropy-coef-deep when active. Pass an "
        "empty string to disable.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=50,
        help="Updates per block when --block-rotation is set.",
    )
    parser.add_argument(
        "--mix-configs",
        action="store_true",
        help="vThree mode: each update mixes --configs-per-tier (seats,stacks) "
        "draws from EACH of --mix-tiers (default all three stack tiers), instead "
        "of one config per update + a 50-update block. The gradient averages "
        "over all N configs, removing the consecutive-shallow exposure that "
        "saturates the gate. Forces blocks off; requires --batched.",
    )
    parser.add_argument(
        "--configs-per-tier",
        type=int,
        default=10,
        help="With --mix-configs: distinct (seats,stacks) draws per tier per update.",
    )
    parser.add_argument(
        "--mix-tiers",
        type=str,
        default="clubgg,clubgg_deep,deep",
        help="With --mix-configs: comma-separated stack tiers mixed per update.",
    )
    parser.add_argument(
        "--anneal-entropy",
        action="store_true",
        help="Automatically lower each tier's block-rotation entropy coef by "
        "--anneal-step whenever that tier's F/T/R (per-street aggression) held "
        "or rose vs its previous same-tier block. Annealed floors, per-tier "
        "F/T/R baselines, the in-block accumulator, and the update counter are "
        "persisted in the checkpoint and restored on warm-start (so block "
        "position + annealed floors survive a relaunch). When absent, behavior "
        "is identical to static --block-rotation.",
    )
    parser.add_argument(
        "--anneal-step",
        type=float,
        default=0.002,
        help="Entropy-coef decrement per successful block (default 0.002). "
        "Live-tunable without pausing training via runs/anneal_control.json "
        '— e.g. {"step": 0.003}. The same file can manually set any '
        'tier\'s coef: {"tier_ent": {"deep": 0.08}} (one-shot; the anneal '
        "continues from the new level). Applied whenever the file content "
        "changes.",
    )
    parser.add_argument(
        "--anneal-floor",
        type=float,
        default=0.0,
        help="Minimum entropy coef the anneal will reach (default 0.0).",
    )
    parser.add_argument(
        "--anneal-tolerance",
        type=float,
        default=1.0,
        help="F/T/R points a street may slip vs its baseline and still "
        "count as 'held' (default 1.0 — e.g. 30/30/30 -> 29/29/29 still "
        "lowers entropy). Soaks up the block-to-block variance from "
        "sampled seat counts / stack configs so one unusually aggressive "
        "block doesn't set an unreachable bar.",
    )
    parser.add_argument(
        "--anneal-start-update",
        type=int,
        default=600,
        help="No anneal decisions (no baseline recording, no lowering) "
        "until this many updates have completed — gives the strategy "
        "time to converge to something reasonable before entropy starts "
        "coming down (default 600). Counted on the persisted update "
        "counter, so warm-started stems past the threshold anneal "
        "immediately.",
    )
    parser.add_argument(
        "--seats-dist",
        type=str,
        choices=("uniform", "clubgg", "nlh_ring"),
        default="uniform",
        help="'uniform' samples from --num-seats-range equiprobably; 'clubgg' "
        "weights 6:30/5:25/4:25/3:15/2:10 (normalized) restricted to "
        "--num-seats-range; 'nlh_ring' slightly favors 5-6 handed "
        "(1.25x the 2/3/4 weight — ~22.7% each vs ~18.2%).",
    )
    parser.add_argument(
        "--bb",
        type=int,
        default=10000,
        help="Chips per bb (default 10000 → cent precision at $20/bb).",
    )
    parser.add_argument(
        "--ante",
        type=int,
        default=None,
        help="Per-player ante in chips. Default: 3bb for the bomb pot "
        "(the historical 30000), 0.5bb for NLH (the 5/10(5) structure).",
    )
    parser.add_argument(
        "--variant",
        choices=[VARIANT_PLO5, VARIANT_PLO4, VARIANT_PLO6, VARIANT_NLH],
        default=VARIANT_PLO5,
        help="Game variant. 'plo4/plo6_double_bomb' = the PLO5 bomb pot "
        "with 4/6 hole cards (same obs layout + 11-anchor PL head). "
        "'nlh_single' = no-limit hold'em: 2 hole cards, single board, "
        "SB/BB + per-player ante, preflop street, and the 12-anchor NLH "
        "sizing ladder. All variants support --batched. Checkpoints are "
        "variant-specific: every variant trains from scratch (no "
        "cross-variant warm-start).",
    )
    parser.add_argument(
        "--sb",
        type=int,
        default=None,
        help="Small blind in chips (NLH only). Default bb/2. Ignored for "
        "the bomb-pot variant.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Optional warm-start: load model weights from this .pt before training.",
    )
    parser.add_argument(
        "--warmstart-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On --load-checkpoint, reconstruct the opponent pool from the "
        "checkpoint's numbered siblings (<stem>_<N>.pt) — the members a "
        "never-stopped run would hold: the exact prior membership when "
        "the checkpoint recorded pool_member_updates, else the nearest "
        "files to the natural snapshot grid. Without it a resumed run "
        "plays pure self-play until the first snapshot tick. "
        "--no-warmstart-pool restores the old empty-pool resume.",
    )
    parser.add_argument(
        "--warmstart-pool-dir",
        type=Path,
        default=None,
        help="Directory to scan for pool-seed checkpoints (default: the "
        "--load-checkpoint file's own directory).",
    )
    parser.add_argument(
        "--critic-hidden-dim",
        type=int,
        default=1536,
        help="Hidden width of the centralized critic (training-only value "
        "net that sees all hole cards).",
    )
    parser.add_argument(
        "--critic-num-blocks",
        type=int,
        default=2,
        help="Residual blocks in the centralized critic torso.",
    )
    parser.add_argument(
        "--kl-anchor-coef",
        type=float,
        default=0.0,
        help="KL-to-EMA-reference regularizer coefficient. 0 disables "
        "(no EMA model is built). The reference is NOT persisted in "
        "checkpoints — on (re)start it re-initializes to the current "
        "weights and ramps in over ~1/(1-ema) updates.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=TrainingConfig.lr,
        help="Base Adam learning rate before the --lr-warmup-updates ramp "
        "and any live anneal_control.json {\"lr\": ...} retune. Lower it for "
        "warm restarts whose gate is fragile under the full default rate.",
    )
    parser.add_argument(
        "--lr-warmup-updates",
        type=int,
        default=0,
        help="Linear LR warmup over the first N GLOBAL updates (cold "
        "starts only in practice — warm restarts past N run at full LR). "
        "0 disables. Cold-start Adam steps at full LR moved the policy "
        "by KL 1-20 per minibatch, tripping the KL guard at mb1-2 and "
        "starving the critic (vTwo1 2026-06-11); small early steps let "
        "the full inner loop run.",
    )
    parser.add_argument(
        "--adv-clip",
        type=float,
        default=8.0,
        help="Clamp normalized advantages to ±N σ before the PPO loss "
        "(0 disables). PPO clips the ratio, not the advantage weight; "
        "deep-stack all-in pots produce 30σ+ samples that carry 30x "
        "gradient weight and drove the post-block-transition violence.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.5,
        help="SOFT KL guard (early-stop): when a minibatch's |approx_kl| "
        "exceeds this, stop the PPO inner loop but KEEP the minibatches "
        "already applied this update. Standard PPO early-stopping. 0 "
        "disables. Live-tunable via runs/anneal_control.json {\"target_kl\"}.",
    )
    parser.add_argument(
        "--kl-hard",
        type=float,
        default=10.0,
        help="HARD KL guard (full rollback): when a minibatch's "
        "|approx_kl| exceeds this, restore params + optimizer state and "
        "discard the WHOLE update. Reserved for catastrophe (vTwo2 hit "
        "approx_kl ~ +2417 at update 173). Should be >= --target-kl. 0 "
        "disables hard rollback. Live-tunable via anneal_control.json.",
    )
    parser.add_argument(
        "--sizing-entropy-scale",
        type=float,
        default=1.0,
        help="v2 sizing-entropy scale: multiplies the anchor+beta "
        "(sizing-head) entropy bonus relative to the gate. 1.0 = off. >1 "
        "resists the anchor/beta over-sharpening that drives v2 saturation "
        "collapse without loosening the gate. Live-tunable via "
        'runs/anneal_control.json {"sizing_entropy_scale": X}.',
    )
    parser.add_argument(
        "--kl-anchor-ema",
        type=float,
        default=0.999,
        help="EMA decay of the KL reference model.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="0 disables; else save checkpoint_{update}.pt every N updates",
    )
    parser.add_argument(
        "--checkpoint-every-sec",
        type=float,
        default=0.0,
        help="If > 0, save a mid-run <stem>_<updates>.pt on this wall-clock cadence.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/stub.pt")
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print a per-update line every N updates (default 1).",
    )
    parser.add_argument(
        "--batched",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use collect_rollout_batched (Phase A-D speedup path) instead of the serial collector. "
        "Pass --no-batched to use the serial collector. Default: batched "
        "for every variant (NLH gained its batched packer + encoder "
        "2026-07-03).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cpu", "cuda"),
        help="Torch device for the learner + rollout buffers.",
    )
    parser.add_argument(
        "--profile-one-update",
        action="store_true",
        help="Wrap update 0 in torch.profiler, export Chrome trace to "
        "runs/profile_update0.json, print a top-40 summary, and exit. "
        "Adds ~10-30%% overhead; use only when diagnosing per-step CPU/GPU "
        "attribution.",
    )
    args = parser.parse_args()

    # Variant resolution: NLH defaults to the 5/10(5)-style structure
    # (sb = bb/2, ante = bb/2 per player); the bomb pot keeps its
    # historical 3bb ante and no blinds.
    is_nlh = args.variant == VARIANT_NLH
    if args.ante is None:
        args.ante = args.bb // 2 if is_nlh else 3 * args.bb
    if args.sb is None:
        args.sb = args.bb // 2 if is_nlh else 0
    if not is_nlh:
        args.sb = 0
    if args.batched is None:
        args.batched = True

    if args.aggression_bonus_c is None:
        args.aggression_bonus_c = 5.0 if args.stack_dist == "agro_deep" else 0.0

    if args.batch_size is None:
        if args.num_minibatches <= 0:
            raise SystemExit("--num-minibatches must be > 0")
        args.batch_size = max(
            1,
            (args.rollout_length + args.num_minibatches - 1) // args.num_minibatches,
        )
        print(
            f"[batch-size] derived {args.batch_size} from "
            f"rollout_length={args.rollout_length} / num_minibatches={args.num_minibatches}"
        )

    # The block-rotation default cycles PLO-named stack tiers (clubgg
    # bands, PLO-tuned entropy seeds). Running those against an NLH
    # table would be the silent-wrong-default failure mode again (cf.
    # the 128x2 hidden-dim incident), so NLH disables block rotation
    # unless the user explicitly overrides the cycle — plain
    # --stack-dist + --entropy-coef govern instead.
    _BLOCK_ROTATION_DEFAULT = "clubgg:0.1,clubgg_deep:0.1,deep:0.2"
    if is_nlh and args.block_rotation == _BLOCK_ROTATION_DEFAULT:
        print(
            "[variant] nlh_single: block-rotation default (PLO tiers) "
            "disabled; sampling via --stack-dist "
            f"{args.stack_dist!r} at --entropy-coef {args.entropy_coef}"
        )
        args.block_rotation = ""

    blocks = _parse_block_rotation(args.block_rotation)
    if args.mix_configs:
        blocks = []  # mix mode replaces block-rotation (summary printed below)
    if blocks and args.block_size <= 0:
        raise SystemExit("--block-size must be > 0 when --block-rotation is set")
    if blocks:
        rotation_summary = " -> ".join(
            f"{tier}@ent={ent:.3f}" for tier, ent in blocks
        )
        print(
            f"[block-rotation] N={args.block_size} per block, "
            f"cycle: {rotation_summary}"
        )

    mix_tiers = [t.strip() for t in args.mix_tiers.split(",") if t.strip()]
    if args.mix_configs:
        if not args.batched:
            raise SystemExit("--mix-configs requires --batched")
        if not mix_tiers:
            raise SystemExit("--mix-tiers parsed to empty")
        if args.configs_per_tier <= 0:
            raise SystemExit("--configs-per-tier must be > 0")
        print(
            f"[mix-configs] {args.configs_per_tier} configs/tier x "
            f"{len(mix_tiers)} tiers ({','.join(mix_tiers)}) = "
            f"{args.configs_per_tier * len(mix_tiers)} configs/update"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled torch wheel (see CLAUDE.md) or pass --device cpu."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    threads_file = Path("runs/threads.txt")
    current_threads: int | None = None
    if args.device == "cpu":
        # Initial thread count: prefer OMP_NUM_THREADS env var on launch
        # (matches BLAS threadpool); torch's intra-op pool is independent
        # and needs explicit set_num_threads.
        omp_env = os.environ.get("OMP_NUM_THREADS", "").strip()
        if omp_env.isdigit() and int(omp_env) > 0:
            torch.set_num_threads(int(omp_env))
        current_threads = torch.get_num_threads()
        print(f"[threads] initial torch threads = {current_threads}")

    seats_choices = _parse_seats_range(args.num_seats_range)
    stack_lo, stack_hi = _parse_stack_range(args.stack_range)

    train_cfg = TrainingConfig(
        lr=args.lr,
        num_updates=args.num_updates,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        snapshot_every=args.snapshot_every,
        ev_runout_samples=EV_RUNOUT_SAMPLES,
        pool_mix_prob=args.pool_mix_prob,
        pool_opp_seats=args.pool_opp_seats,
        entropy_coef=args.entropy_coef,
        aggression_bonus_c=args.aggression_bonus_c,
        retroactive_bonus_c=args.retroactive_bonus_c,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_blocks=args.critic_num_blocks,
        kl_anchor_coef=args.kl_anchor_coef,
        kl_anchor_ema=args.kl_anchor_ema,
        target_kl=args.target_kl,
        kl_hard=args.kl_hard,
        sizing_entropy_scale=args.sizing_entropy_scale,
        adv_clip=args.adv_clip,
        device=args.device,
    )

    model_cls = ActorCriticV4 if args.sizing_head == "logistic" else ActorCriticV2
    obs_dim = OBS_DIM_NLH if is_nlh else OBS_DIM
    anchor_spec = NLH_ANCHOR_SPEC if is_nlh else PLO_ANCHOR_SPEC
    model = model_cls(
        hidden_dim=train_cfg.hidden_dim,
        num_layers=train_cfg.num_layers,
        obs_dim=obs_dim,
        anchor_spec=anchor_spec,
    )
    print(
        f"[head] sizing-head={args.sizing_head} "
        f"(head_version={model.head_version}) variant={args.variant} "
        f"obs_dim={obs_dim} anchors={anchor_spec.count} ({anchor_spec.name})"
    )
    model.to(train_cfg.device)
    critic = CentralCritic(
        obs_dim=obs_dim,
        hidden_dim=train_cfg.critic_hidden_dim,
        num_blocks=train_cfg.critic_num_blocks,
    )
    critic.to(train_cfg.device)
    print(f"[device] learner on {train_cfg.device}")
    # Annealing state restored from the checkpoint (None when absent / cold).
    restored_update: int | None = None
    restored_tier_ent: dict | None = None
    restored_baseline: dict | None = None
    restored_block_acc: dict | None = None
    restored_pool_updates: list | None = None
    if args.load_checkpoint is not None:
        ckpt = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        ckpt_variant = str(ckpt.get("variant", VARIANT_PLO5))
        if ckpt_variant != args.variant:
            # Unconditional: even dims-identical pairs (plo4/plo5/plo6
            # share OBS_DIM 991 + the 11-anchor head) are refused. The
            # games' equities and minimum made-hand strengths differ so
            # much by hole-card count that transferred weights are a
            # confused prior, not a head start — every variant trains
            # from scratch (decision 2026-07-03).
            raise SystemExit(
                f"variant mismatch: checkpoint={ckpt_variant} vs "
                f"--variant={args.variant}. Cross-variant warm-starts are "
                "refused: each variant trains from scratch."
            )
        ckpt_head = int(ckpt.get("head_version", 1))
        if ckpt_head != model.head_version:
            raise SystemExit(
                f"head_version mismatch: checkpoint={ckpt_head} vs model="
                f"{model.head_version} (selected by --sizing-head). Warm-start "
                "requires a checkpoint of the same sizing-head version; start "
                "cold or point --load-checkpoint at a matching-family checkpoint."
            )
        if "critic" not in ckpt:
            raise SystemExit(
                "v2 checkpoint is missing the 'critic' state dict — cannot "
                "warm-start the centralized critic."
            )
        ckpt_cfg = ckpt.get("config") or {}
        ckpt_hidden = int(ckpt_cfg.get("hidden_dim", train_cfg.hidden_dim))
        ckpt_layers = int(ckpt_cfg.get("num_layers", train_cfg.num_layers))
        ckpt_critic_hidden = int(
            ckpt_cfg.get("critic_hidden_dim", train_cfg.critic_hidden_dim)
        )
        ckpt_critic_blocks = int(
            ckpt_cfg.get("critic_num_blocks", train_cfg.critic_num_blocks)
        )
        ckpt_gate_count = ckpt.get("gate_count")
        if ckpt_critic_hidden != train_cfg.critic_hidden_dim:
            raise SystemExit(
                f"critic_hidden_dim mismatch: checkpoint={ckpt_critic_hidden} "
                f"vs --critic-hidden-dim={train_cfg.critic_hidden_dim}"
            )
        if ckpt_critic_blocks != train_cfg.critic_num_blocks:
            raise SystemExit(
                f"critic_num_blocks mismatch: checkpoint={ckpt_critic_blocks} "
                f"vs --critic-num-blocks={train_cfg.critic_num_blocks}"
            )
        if ckpt_hidden != train_cfg.hidden_dim:
            raise SystemExit(
                f"hidden_dim mismatch: checkpoint={ckpt_hidden} vs --hidden-dim={train_cfg.hidden_dim}"
            )
        if ckpt_layers != train_cfg.num_layers:
            raise SystemExit(
                f"num_layers mismatch: checkpoint={ckpt_layers} vs --num-layers={train_cfg.num_layers}"
            )
        if ckpt_gate_count is not None and int(ckpt_gate_count) != GATE_ACTIONS:
            raise SystemExit(
                f"gate_count mismatch: checkpoint={ckpt_gate_count} vs current={GATE_ACTIONS}. "
                "This checkpoint was trained with a different gate-head width and cannot be warm-started."
            )
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        critic.load_state_dict(ckpt["critic"])
        prior_game = ckpt.get("game_config")
        print(f"warm-started from {args.load_checkpoint} (prior game_config: {prior_game})")
        restored_update = ckpt.get("update_counter")
        restored_tier_ent = ckpt.get("anneal_tier_ent")
        restored_baseline = ckpt.get("anneal_baseline")
        restored_block_acc = ckpt.get("anneal_block_acc")
        restored_pool_updates = ckpt.get("pool_member_updates")

    # Per-tier entropy-anneal state. `tier_ent` is always seeded from the
    # --block-rotation initial values and is what the loop reads for the entropy
    # coef (so with --anneal-entropy OFF it stays static == today's behavior).
    # Restore from the checkpoint only when annealing, so an anneal-off run is
    # byte-for-byte unchanged.
    tier_ent: dict[str, float] = {tier: ent for tier, ent in blocks}
    tier_baseline: dict[str, tuple[float, float, float] | None] = {
        tier: None for tier, _ in blocks
    }
    block_acc: dict = {"bonus_steps": [0, 0, 0], "steps": [0, 0, 0], "tier": None}
    if args.mix_configs:
        # Mix mode has no per-tier blocks; use one live-tunable entropy coef
        # (all tiers equal). anneal_control's {"tier_ent": {...}} still tunes
        # it live; {"lr": X} still tunes LR. Auto-anneal (F/T/R-driven) stays
        # off since `blocks` is empty.
        tier_ent = {t: args.entropy_coef for t in mix_tiers}
        tier_baseline = {t: None for t in mix_tiers}
    if args.anneal_entropy:
        if restored_tier_ent:
            tier_ent.update(
                {k: float(v) for k, v in restored_tier_ent.items() if k in tier_ent}
            )
        if restored_baseline:
            tier_baseline.update(
                {
                    k: (tuple(v) if v is not None else None)
                    for k, v in restored_baseline.items()
                    if k in tier_baseline
                }
            )
        if restored_block_acc:
            block_acc = {
                "bonus_steps": list(restored_block_acc.get("bonus_steps", [0, 0, 0])),
                "steps": list(restored_block_acc.get("steps", [0, 0, 0])),
                "tier": restored_block_acc.get("tier"),
            }
        print(
            f"[anneal] enabled step={args.anneal_step} floor={args.anneal_floor} "
            f"tol={args.anneal_tolerance} anneal_after={args.anneal_start_update} "
            f"start_update="
            f"{restored_update if restored_update is not None else 0} "
            f"tier_ent={tier_ent} baselines={tier_baseline}"
        )

    trainer = PPOTrainer(model, train_cfg, critic=critic)
    pool = OpponentPool(capacity=train_cfg.opponent_pool_size)
    rng = np.random.default_rng(args.seed)

    # Warm-start pool reconstruction: refill the (ephemeral) opponent
    # pool from the loaded checkpoint's numbered siblings so a resumed
    # run faces the same opponents a never-stopped one would, instead of
    # pure self-play until the first snapshot tick.
    if args.load_checkpoint is not None and args.warmstart_pool:
        ws_target = restored_update
        if ws_target is None:
            m = re.match(r"^.+_(\d+)\.pt$", args.load_checkpoint.name)
            ws_target = int(m.group(1)) if m else None
        if ws_target is None:
            _, family = discover_checkpoint_family(
                args.load_checkpoint, args.warmstart_pool_dir
            )
            ws_target = max(family) if family else None
        if ws_target is None:
            print(
                "[pool] warm-start seeding skipped: source update unknown "
                "(no update_counter in the checkpoint, no _<N> filename, "
                "no numbered siblings on disk)"
            )
        else:
            seeded = seed_pool_from_checkpoints(
                pool,
                args.load_checkpoint,
                int(ws_target),
                train_cfg.snapshot_every,
                args.variant,
                model.head_version,
                model.state_dict(),
                preferred=restored_pool_updates,
                directory=args.warmstart_pool_dir,
            )
            if seeded:
                print(
                    f"[pool] warm-start seeded {len(seeded)}/{pool.capacity} "
                    f"members from updates {seeded} "
                    f"(target u{ws_target}, snapshot_every={train_cfg.snapshot_every}"
                    + (", exact prior membership honored"
                       if restored_pool_updates else "")
                    + ")"
                )
            else:
                print(
                    "[pool] warm-start seeding found no compatible sibling "
                    "checkpoints — pool starts empty (pure self-play until "
                    "the first snapshot)"
                )

    # Live anneal control (step changes + manual tier-coef overrides)
    # without pausing training — see --anneal-step help.
    anneal_control_file = Path("runs/anneal_control.json")
    live_anneal_step = float(args.anneal_step)
    live_lr = float(train_cfg.lr)
    # Flat (non-tier) entropy coefs — what NLH / plain --stack-dist runs
    # consume each update. Live-tunable via {"entropy_coef": X} /
    # {"entropy_coef_deep": X}; tier runs keep using tier_ent. (Same
    # expression as the later `entropy_coef_deep` local — that one is
    # defined further down in main.)
    live_entropy_coef = float(args.entropy_coef)
    live_entropy_coef_deep = float(
        args.entropy_coef if args.entropy_coef_deep is None else args.entropy_coef_deep
    )
    # Seed the baseline with any PRE-EXISTING anneal_control.json so a stale
    # file left from a prior run/session is treated as ALREADY-APPLIED — not as
    # a fresh edit that silently overrides THIS run's launch args (--lr,
    # --entropy-coef via tier_ent, --target-kl, ...). Only edits made AFTER
    # startup take effect. A stale {"lr": 3e-4} once forced 3e-4 onto three runs
    # that launched with a lower --lr before this guard (2026-06-24).
    last_anneal_control: str | None = None
    if anneal_control_file.exists():
        try:
            last_anneal_control = anneal_control_file.read_text()
        except OSError:
            last_anneal_control = None

    collector = collect_rollout_batched if args.batched else collect_rollout
    time_budget = float(args.train_seconds)
    use_time_budget = time_budget > 0.0
    t_start = time.time()
    last_snapshot_sec = t_start
    last_ckpt_sec = t_start

    def _save_mid(update_idx: int) -> None:
        game_cfg_snap = sampled_game_cfg.__dict__
        mid_path = args.checkpoint.with_name(f"{args.checkpoint.stem}_{update_idx}.pt")
        mid_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "critic": critic.state_dict(),
                "head_version": model.head_version,
                "config": train_cfg.__dict__,
                "game_config": game_cfg_snap,
                "gate_count": GATE_ACTIONS,
                "variant": args.variant,
                "anchor_count": model._anchor_count,
                "update_counter": update_idx,
                # Metadata only (update indices, not weights): lets a
                # warm-start reconstruct the exact pool membership from
                # the sibling files still on disk.
                "pool_member_updates": list(pool.tags),
                "anneal_tier_ent": tier_ent,
                "anneal_baseline": tier_baseline,
                "anneal_block_acc": block_acc,
            },
            mid_path,
        )

    # Restore the update counter only when annealing, so the block cycle
    # continues across a relaunch instead of resetting to block 1 (which
    # under-trains the deep tier). Anneal-off keeps today's reset-to-0.
    update = (
        int(restored_update)
        if (args.anneal_entropy and restored_update is not None)
        else 0
    )
    sampled_game_cfg, sampled_eff_dist = _sample_game_config(
        seats_choices,
        stack_lo,
        stack_hi,
        args.bb,
        args.ante,
        rng,
        stack_dist=args.stack_dist,
        seats_dist=args.seats_dist,
        variant=args.variant,
        sb=args.sb,
    )
    entropy_coef_deep = (
        args.entropy_coef if args.entropy_coef_deep is None else args.entropy_coef_deep
    )

    # Graceful shutdown: SIGINT (Ctrl+C) / SIGTERM / (Windows) SIGBREAK
    # set the flag; loop checks it at the top of each iteration and
    # falls through to the final save so partial work is persisted.
    stop_requested = {"flag": False}

    def _request_stop(signum: int, _frame: object) -> None:
        stop_requested["flag"] = True
        print(f"\n[signal {signum}] stop requested — saving and exiting after current update")

    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_stop)

    while True:
        if stop_requested["flag"]:
            break

        # Live thread-count adjustment: edit runs/threads.txt to
        # change torch's intra-op threadpool without restarting the
        # run. Malformed reads are ignored. CUDA runs skip this —
        # set_num_threads is a CPU-pool concept.
        if train_cfg.device == "cpu" and threads_file.exists():
            try:
                desired = int(threads_file.read_text().strip())
                if desired > 0 and desired != current_threads:
                    torch.set_num_threads(desired)
                    current_threads = desired
                    print(f"[threads] set torch threads -> {desired}")
            except (ValueError, OSError):
                pass

        if use_time_budget:
            if time.time() - t_start >= time_budget:
                break
        else:
            if update >= train_cfg.num_updates:
                break

        # Live tuning for EVERY run (was block/mix-configs only until
        # 2026-07-04, which made NLH entropy steps require a restart).
        # The startup-seeded baseline still guards against stale files.
        if anneal_control_file.exists():
            try:
                control_raw = anneal_control_file.read_text()
            except OSError:
                control_raw = None
            (
                live_anneal_step,
                last_anneal_control,
                live_lr,
                live_entropy_coef,
                live_entropy_coef_deep,
            ) = _apply_anneal_control(
                control_raw, last_anneal_control, tier_ent, live_anneal_step,
                live_lr, live_entropy_coef, live_entropy_coef_deep,
                trainer=trainer,
            )

        if args.mix_configs:
            # vThree: every update mixes `configs_per_tier` (seats,stacks) draws
            # from each mix tier (no block-rotation), so the gradient averages
            # over all N configs — no consecutive-tier saturation.
            mix_cfgs = [
                _sample_game_config(
                    seats_choices, stack_lo, stack_hi, args.bb, args.ante, rng,
                    stack_dist=tier, seats_dist=args.seats_dist,
                    variant=args.variant, sb=args.sb,
                )[0]
                for tier in mix_tiers
                for _ in range(args.configs_per_tier)
            ]
            block_idx = -1
            active_tier = "mix"
            sampled_game_cfg, sampled_eff_dist = mix_cfgs[0], "mix"
        elif blocks:
            block_idx = (update // args.block_size) % len(blocks)
            active_tier = blocks[block_idx][0]
            sampled_game_cfg, sampled_eff_dist = _sample_game_config(
                seats_choices, stack_lo, stack_hi, args.bb, args.ante, rng,
                stack_dist=active_tier, seats_dist=args.seats_dist,
                variant=args.variant, sb=args.sb,
            )
        else:
            block_idx = -1
            active_tier = args.stack_dist
            sampled_game_cfg, sampled_eff_dist = _sample_game_config(
                seats_choices, stack_lo, stack_hi, args.bb, args.ante, rng,
                stack_dist=active_tier, seats_dist=args.seats_dist,
                variant=args.variant, sb=args.sb,
            )
        _profile_this = args.profile_one_update and update == 0
        _prof = None
        if _profile_this:
            _prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=False,
                record_shapes=False,
            )
            _prof.__enter__()

        if args.mix_configs:
            batch = collect_rollout_multiconfig(
                model, pool, mix_cfgs, train_cfg, rng, critic=critic
            )
            # One coef for the mixed update (all mix tiers seeded equal);
            # live-tunable via anneal_control {"tier_ent": {...}}.
            update_entropy_coef = tier_ent.get(mix_tiers[0], args.entropy_coef)
        elif blocks:
            batch = collector(model, pool, sampled_game_cfg, train_cfg, rng, critic=critic)
            # tier_ent[tier] == the static block value when --anneal-entropy is
            # off (it is never mutated then), so this is identical to today.
            update_entropy_coef = tier_ent[active_tier]
        else:
            batch = collector(model, pool, sampled_game_cfg, train_cfg, rng, critic=critic)
            update_entropy_coef = (
                live_entropy_coef_deep
                if sampled_eff_dist == "deep"
                else live_entropy_coef
            )
        # Cold-start LR warmup: small early steps keep per-minibatch KL
        # inside the guard's trust region, so all minibatches apply and
        # the critic actually trains (a tripped update aborts the critic
        # too — huge advantages then keep the next step violent). Uses
        # the GLOBAL update index, so warm restarts past the window run
        # at full LR from the first update.
        lr_scale = _lr_warmup_scale(update, args.lr_warmup_updates)
        for _pg in trainer.optimizer.param_groups:
            _pg["lr"] = live_lr * lr_scale
        stats = trainer.update(batch, rng, entropy_coef=update_entropy_coef)

        # Accumulate this update's per-street aggression counts into the current
        # block's bucket (reset whenever a new tier's block begins).
        if blocks and args.anneal_entropy:
            if block_acc["tier"] != active_tier:
                block_acc = {"bonus_steps": [0, 0, 0], "steps": [0, 0, 0], "tier": active_tier}
            for s in range(3):
                block_acc["bonus_steps"][s] += int(batch.aggr_bonus_steps_by_street[s])
                block_acc["steps"][s] += int(batch.aggr_steps_total_by_street[s])

        assert not np.isnan(stats.policy_loss), "NaN in policy loss"
        assert not np.isnan(stats.value_loss), "NaN in value loss"

        now = time.time()

        # Snapshot on update count, then also on wall-clock if configured.
        if update % train_cfg.snapshot_every == 0:
            pool.snapshot(model, tag=update)
        if args.snapshot_every_sec > 0 and now - last_snapshot_sec >= args.snapshot_every_sec:
            pool.snapshot(model, tag=update)
            last_snapshot_sec = now

        if _prof is not None:
            _prof.__exit__(None, None, None)
            trace_path = Path("runs/profile_update0.json")
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            _prof.export_chrome_trace(str(trace_path))
            print(_prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=40
            ))
            print(f"[profile] chrome trace -> {trace_path}")
            stop_requested["flag"] = True

        # Mid-run checkpoints: update-count + wall-clock variants.
        if args.checkpoint_every > 0 and update > 0 and update % args.checkpoint_every == 0:
            _save_mid(update)
        if args.checkpoint_every_sec > 0 and now - last_ckpt_sec >= args.checkpoint_every_sec:
            _save_mid(update)
            last_ckpt_sec = now

        if update % args.log_every == 0:
            elapsed = now - t_start
            stacks_bb = [round(s / args.bb, 1) for s in sampled_game_cfg.resolved_stacks]
            bonus_mean = batch.aggr_bonus_total_bb / max(1, batch.aggr_steps_total)
            steps_by_street = batch.aggr_steps_total_by_street
            bonus_by_street = batch.aggr_bonus_steps_by_street
            bonus_pct_flop = 100.0 * bonus_by_street[0] / max(1, steps_by_street[0])
            bonus_pct_turn = 100.0 * bonus_by_street[1] / max(1, steps_by_street[1])
            bonus_pct_river = 100.0 * bonus_by_street[2] / max(1, steps_by_street[2])
            print(
                f"[{elapsed:7.1f}s] update {update:5d}  "
                f"pi={stats.policy_loss:+.4f}  "
                f"v={stats.value_loss:.4f}  "
                f"vd={stats.display_loss:.4f}  "
                f"H={stats.entropy:.3f}  "
                f"Hg/Ha/Hb={stats.gate_entropy:.2f}/{stats.anchor_entropy:.2f}/"
                f"{stats.beta_entropy:.2f}  "
                f"kl={stats.approx_kl:+.4f}  "
                f"klG/klA/klB={stats.gate_kl:+.3f}/{stats.anchor_kl:+.3f}/"
                f"{stats.beta_kl:+.3f}  "
                + (f"klanc={stats.kl_anchor:.4f}  " if args.kl_anchor_coef > 0 else "")
                + (
                    (
                        f"KLROLLBACK@mb{stats.kl_stopped_at}"
                        if stats.rolled_back
                        else f"KLSTOP@mb{stats.kl_stopped_at}"
                    )
                    + f"(kl={stats.kl_stop:+.2f})  "
                    if stats.kl_stopped_at >= 0 else ""
                )
                +
                f"bonus={bonus_mean:+.4f}  "
                f"bonus%(F/T/R)={bonus_pct_flop:4.1f}/{bonus_pct_turn:4.1f}/"
                f"{bonus_pct_river:4.1f}  "
                f"pool={len(pool)}  "
                f"seats={sampled_game_cfg.num_seats}  "
                f"stacks_bb={stacks_bb}  "
                f"ent={update_entropy_coef:.3f}"
                + (f"  lr×{lr_scale:.2f}" if lr_scale < 1.0 else "")
                + (f"  block={block_idx + 1}/{len(blocks)}({active_tier})" if blocks else "")
            )

        # End-of-block entropy anneal: this tier's 50-update block just finished.
        if blocks and args.anneal_entropy and (update + 1) % args.block_size == 0:
            if not _anneal_due(update, args.block_size, args.anneal_start_update):
                # Warmup: discard the block accumulator without recording a
                # baseline or touching coefs — the strategy gets
                # --anneal-start-update updates to converge first.
                print(
                    f"[anneal] tier={active_tier} warmup "
                    f"({update + 1}/{args.anneal_start_update} updates) — "
                    "no baseline, no change"
                )
            else:
                st = block_acc["steps"]
                bn = block_acc["bonus_steps"]
                now_ftr = (
                    100.0 * bn[0] / max(1, st[0]),
                    100.0 * bn[1] / max(1, st[1]),
                    100.0 * bn[2] / max(1, st[2]),
                )
                base = tier_baseline.get(active_tier)
                new_ent, new_base, action = _anneal_decision(
                    now_ftr,
                    base,
                    tier_ent[active_tier],
                    live_anneal_step,
                    args.anneal_floor,
                    args.anneal_tolerance,
                )
                tier_ent[active_tier] = new_ent
                tier_baseline[active_tier] = new_base
                base_str = (
                    "--/--/--" if base is None
                    else f"{base[0]:.1f}/{base[1]:.1f}/{base[2]:.1f}"
                )
                print(
                    f"[anneal] tier={active_tier} "
                    f"F/T/R={now_ftr[0]:.1f}/{now_ftr[1]:.1f}/{now_ftr[2]:.1f} "
                    f"base={base_str} -> {action} ent={new_ent:.4f}"
                )
            block_acc = {"bonus_steps": [0, 0, 0], "steps": [0, 0, 0], "tier": None}

        update += 1

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "critic": critic.state_dict(),
            "head_version": model.head_version,
            "config": train_cfg.__dict__,
            "game_config": sampled_game_cfg.__dict__,
            "gate_count": GATE_ACTIONS,
            "variant": args.variant,
            "anchor_count": model._anchor_count,
            "update_counter": update,
            "pool_member_updates": list(pool.tags),
            "anneal_tier_ent": tier_ent,
            "anneal_baseline": tier_baseline,
            "anneal_block_acc": block_acc,
        },
        args.checkpoint,
    )
    elapsed = time.time() - t_start
    print(
        f"Saved checkpoint to {args.checkpoint} after {update} updates "
        f"({elapsed:.1f}s wall-clock)"
    )
    if train_cfg.device == "cuda" and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"[cuda] peak memory allocated: {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
