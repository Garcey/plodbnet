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
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from plo5bp.actions import GATE_ACTIONS
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.network import ActorCriticV2, CentralCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout, collect_rollout_batched
from plo5bp.selfplay import OpponentPool


def _parse_seats_range(spec: str) -> tuple[int, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out = tuple(int(p) for p in parts)
    if not out or any(n < 2 for n in out):
        raise SystemExit(f"--num-seats-range must list ints ≥ 2, got {spec!r}")
    return out


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
    seats_choices: tuple[int, ...], rng: np.random.Generator
) -> int:
    # Restrict to the intersection of clubgg weights and user-supplied
    # seat range; renormalize. Seats not in _CLUBGG_SEAT_WEIGHTS fall
    # back to uniform probability across the remaining clubgg-weighted
    # seats so we never silently drop them.
    weights = [_CLUBGG_SEAT_WEIGHTS.get(n, 0.0) for n in seats_choices]
    total = sum(weights)
    if total <= 0.0:
        return int(rng.choice(seats_choices))
    probs = [w / total for w in weights]
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
) -> tuple[GameConfig, str]:
    if seats_dist == "clubgg":
        n_seats = _sample_clubgg_seats(seats_choices, rng)
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
        choices=("uniform", "clubgg", "clubgg_deep", "clubgg_mix", "agro_deep", "deep", "full_mix"),
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
        help="Entropy-coef decrement per successful block (default 0.002).",
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
        default=0.5,
        help="F/T/R points a street may slip and still count as 'held' "
        "(default 0.5; block F/T/R is large-sample and stable).",
    )
    parser.add_argument(
        "--seats-dist",
        type=str,
        choices=("uniform", "clubgg"),
        default="uniform",
        help="'uniform' samples from --num-seats-range equiprobably; 'clubgg' "
        "weights 6:30/5:25/4:25/3:15/2:10 (normalized) restricted to "
        "--num-seats-range.",
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
        default=30000,
        help="Ante in chips (default 30000 → 3bb).",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Optional warm-start: load model weights from this .pt before training.",
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
        default=True,
        help="Use collect_rollout_batched (Phase A-D speedup path) instead of the serial collector. "
        "Pass --no-batched to use the serial collector.",
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

    blocks = _parse_block_rotation(args.block_rotation)
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
        device=args.device,
    )

    model = ActorCriticV2(
        hidden_dim=train_cfg.hidden_dim, num_layers=train_cfg.num_layers
    )
    model.to(train_cfg.device)
    critic = CentralCritic(
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
    if args.load_checkpoint is not None:
        ckpt = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        ckpt_head = int(ckpt.get("head_version", 1))
        if ckpt_head != 2:
            raise SystemExit(
                f"head_version mismatch: checkpoint={ckpt_head} (v1 Beta "
                "sizing head) cannot warm-start the v2 anchor-head trainer. "
                "Start cold or point --load-checkpoint at an anchor-family "
                "checkpoint."
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
            f"tol={args.anneal_tolerance} start_update="
            f"{restored_update if restored_update is not None else 0} "
            f"tier_ent={tier_ent} baselines={tier_baseline}"
        )

    trainer = PPOTrainer(model, train_cfg, critic=critic)
    pool = OpponentPool(capacity=train_cfg.opponent_pool_size)
    rng = np.random.default_rng(args.seed)

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
                "head_version": 2,
                "config": train_cfg.__dict__,
                "game_config": game_cfg_snap,
                "gate_count": GATE_ACTIONS,
                "update_counter": update_idx,
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

        if blocks:
            block_idx = (update // args.block_size) % len(blocks)
            active_tier = blocks[block_idx][0]
        else:
            block_idx = -1
            active_tier = args.stack_dist

        sampled_game_cfg, sampled_eff_dist = _sample_game_config(
            seats_choices,
            stack_lo,
            stack_hi,
            args.bb,
            args.ante,
            rng,
            stack_dist=active_tier,
            seats_dist=args.seats_dist,
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

        batch = collector(model, pool, sampled_game_cfg, train_cfg, rng, critic=critic)
        if blocks:
            # tier_ent[tier] == the static block value when --anneal-entropy is
            # off (it is never mutated then), so this is identical to today.
            update_entropy_coef = tier_ent[active_tier]
        else:
            update_entropy_coef = (
                entropy_coef_deep if sampled_eff_dist == "deep" else args.entropy_coef
            )
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
            pool.snapshot(model)
        if args.snapshot_every_sec > 0 and now - last_snapshot_sec >= args.snapshot_every_sec:
            pool.snapshot(model)
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
                + (f"klA={stats.kl_anchor:.4f}  " if args.kl_anchor_coef > 0 else "")
                +
                f"bonus={bonus_mean:+.4f}  "
                f"bonus%(F/T/R)={bonus_pct_flop:4.1f}/{bonus_pct_turn:4.1f}/"
                f"{bonus_pct_river:4.1f}  "
                f"pool={len(pool)}  "
                f"seats={sampled_game_cfg.num_seats}  "
                f"stacks_bb={stacks_bb}  "
                f"ent={update_entropy_coef:.3f}"
                + (f"  block={block_idx + 1}/{len(blocks)}({active_tier})" if blocks else "")
            )

        # End-of-block entropy anneal: this tier's 50-update block just finished.
        if blocks and args.anneal_entropy and (update + 1) % args.block_size == 0:
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
                args.anneal_step,
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
            "head_version": 2,
            "config": train_cfg.__dict__,
            "game_config": sampled_game_cfg.__dict__,
            "gate_count": GATE_ACTIONS,
            "update_counter": update,
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
