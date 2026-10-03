"""train.py's command line: the parser, the --v6 preset and the flag parsers."""

from __future__ import annotations

import argparse
from pathlib import Path

from plo5bp.config import (
    VARIANT_NLH,
    VARIANT_PLO4,
    VARIANT_PLO5,
    VARIANT_PLO6,
    GameConfig,
    TrainingConfig,
)
from plo5bp.train.tiers import _KNOWN_STACK_DISTS, _VALID_STACK_DISTS


def _parse_seats_range(spec: str, variant: str = VARIANT_PLO5) -> tuple[int, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out = tuple(int(p) for p in parts)
    if not out or any(n < 2 for n in out):
        raise SystemExit(f"--num-seats-range must list ints ≥ 2, got {spec!r}")
    # The obs layout has 8 hero-rotated seat slots and the deck must cover
    # every seat's hole cards plus the boards (PLO6: 7 seats max). GameConfig
    # raises on these too (review 2026-09-20 B8) — fail at arg-parse time with
    # the flag named instead of mid-run on the first unlucky seat draw.
    probe = GameConfig(num_seats=2, variant=variant)
    boards = 5 if variant == VARIANT_NLH else 10
    max_seats = min(8, (52 - boards) // probe.hole_count)
    if any(n > max_seats for n in out):
        raise SystemExit(
            f"--num-seats-range: {variant} supports at most {max_seats} seats, "
            f"got {spec!r}"
        )
    return out


def _parse_stack_range(spec: str) -> tuple[float, float]:
    if ":" not in spec:
        raise SystemExit(f"--stack-range must be 'min:max' in bb, got {spec!r}")
    lo_s, hi_s = spec.split(":", 1)
    lo, hi = float(lo_s), float(hi_s)
    if lo <= 0 or hi < lo:
        raise SystemExit(f"invalid --stack-range {spec!r}")
    return lo, hi


def _parse_mix_tiers(spec: str) -> list[str]:
    tiers = [t.strip() for t in spec.split(",") if t.strip()]
    bad = [t for t in tiers if t not in _KNOWN_STACK_DISTS]
    if bad:
        raise SystemExit(
            f"--mix-tiers: unknown tier(s) {bad} — valid: {_KNOWN_STACK_DISTS}"
        )
    return tiers


# Training always grades early all-ins by EXPECTED value over board
# runouts (engine `payouts_ev`) instead of the one sampled runout: there is
# no training regime where realized runout luck in the reward is
# preferable. --ev-runout-samples sets how many runouts (default below): 64
# cuts runout variance ~64x; profiled at ~+45% ENGINE time in a 30%-shove
# stress test (a few percent of real update time, where the network
# dominates). Fold-outs and river-closes short-circuit to exact payouts in
# Rust. The TrainingConfig default stays 0 so UI/eval/parity paths keep
# realized payouts.
EV_RUNOUT_SAMPLES = 64


# ---- --v6 preset (C2) ------------------------------------------------------
# attr -> (legacy_default, v6_value). The covered flags use default=None
# sentinels in argparse so "flag not passed" is distinguishable from
# "explicitly passed at the default value" — the old parser.get_default
# comparison could not tell those apart and silently overrode explicit
# ablation flags (`--v6 --advantage-estimator gae` trained vrpo;
# `--v6 --q-aux-coef 0` trained the Q head at 0.5). TrainingConfig dataclass
# defaults are deliberately NOT the mechanism (breaks live stems + parity).
_V6_PRESET: "dict[str, tuple[object, object]]" = {
    "sizing_head": ("anchor", "mixture"),
    "advantage_estimator": ("gae", "vrpo"),
    "q_aux_coef": (0.0, 0.5),
    # 2026-07-09 Q-head audit revision: pooled raise column + dense fold
    # supervision (fold forward-return == 0, free labels) so the VRPO Q
    # surface can actually calibrate; adv_head is AGC-exempt (ppo.py).
    # Coef 15.0 since 2026-07-11 (audit #2): at 1.0 the fold term was ~4%
    # of the q gradient (raw-bb² scale mismatch vs the taken-action MSE)
    # and the known-truth anchor lost — fold column drifted to tight
    # −3/−16bb family offsets. 15 ≈ gradient parity. Live-tunable via
    # anneal_control {"q_fold_sup_coef": X}.
    "q_pooled": (False, True),
    "q_fold_sup_coef": (0.0, 15.0),
    "torso_norm": (False, True),
    "l2_init_coef": (0.0, 1e-4),
    "agc_clip": (0.0, 0.1),
    "grad_checkpoint": (False, True),
    "value_bins": (0, 51),
    "clip_prob_dependent": (False, True),
}


def _apply_v6_preset(args) -> "tuple[dict, dict]":
    """Resolve the None-sentinel flags covered by the --v6 preset.

    None (flag not passed) -> the v6 value when --v6 is on, else the legacy
    default. Any non-None value was passed EXPLICITLY — even one equal to a
    default — and always wins ("your flags win", including the --no-<flag>
    boolean forms). Runs on EVERY invocation; non-v6 runs just get the
    legacy defaults filled in. Returns (applied, kept_overrides) for the
    launch log. Tests: tests/python/training/test_v6_preset.py."""
    applied: dict = {}
    kept: dict = {}
    for attr, (legacy_default, v6_value) in _V6_PRESET.items():
        cur = getattr(args, attr)
        if cur is None:
            setattr(args, attr, v6_value if args.v6 else legacy_default)
            if args.v6:
                applied[attr] = v6_value
        elif args.v6:
            kept[attr] = cur
    return applied, kept


# Sentinel flags (default None) resolved after parsing: the flag -> the value
# it means when not passed.
_SENTINEL_DEFAULTS = {
    "clip_room_ext": 0.10,
    "clip_room_mid": 0.05,
    "kl_anchor_ema": 0.999,
}


def validate_flag_combinations(args) -> None:
    """Refuse flag combinations that would silently do nothing (2026-09-28,
    ML-006), then fill the sentinel defaults. Runs after `_apply_v6_preset`
    (which decides clip_prob_dependent)."""
    errors: list[str] = []
    if getattr(args, "batch_on_host", False) and not args.mix_configs:
        errors.append("--batch-on-host needs --mix-configs (only the mixed-config "
                      "collector keeps the batch on the host)")
    for flag in ("clip_room_ext", "clip_room_mid"):
        if getattr(args, flag) is not None and not args.clip_prob_dependent:
            errors.append(f"--{flag.replace('_', '-')} only acts with "
                          "--clip-prob-dependent (or --v6)")
    if getattr(args, "kl_anchor_ema") is not None and not args.kl_anchor_coef > 0.0:
        errors.append("--kl-anchor-ema only acts with --kl-anchor-coef > 0")
    critic_passes = int(args.critic_extra_epochs) > 0 or int(args.actor_freeze_updates) > 0
    if int(args.critic_minibatches) > 0 and not critic_passes:
        errors.append("--critic-minibatches only acts with --critic-extra-epochs > 0 "
                      "or --actor-freeze-updates > 0")
    if getattr(args, "crn_streams", False) and args.batched is False:
        errors.append("--crn-streams needs the batched collector (drop --no-batched)")
    if getattr(args, "minibatches_from_rows", False) and args.batch_size is not None:
        errors.append("--minibatches-from-rows sizes the minibatches itself; "
                      "drop --batch-size")
    if args.critic_q_norm_minibatch and not args.critic_q_norm:
        errors.append("--critic-q-norm-minibatch only acts with --critic-q-norm")
    if (args.critic_fresh or args.critic_init is not None) and args.load_checkpoint is None:
        errors.append("--critic-fresh / --critic-init need --load-checkpoint")
    if errors:
        raise SystemExit("error: " + "\n       ".join(errors))
    for flag, default in _SENTINEL_DEFAULTS.items():
        if getattr(args, flag) is None:
            setattr(args, flag, default)


def _spec_value(action, value):
    """A spec-file value as argparse would have produced it from the CLI."""
    if isinstance(action, argparse._StoreTrueAction):
        if not isinstance(value, bool):
            raise SystemExit(f"--spec: {action.dest} must be true/false, got {value!r}")
        return value
    if isinstance(action, argparse.BooleanOptionalAction):
        if not isinstance(value, bool):
            raise SystemExit(f"--spec: {action.dest} must be true/false, got {value!r}")
        return value
    conv = action.type or (lambda x: x)
    return conv(value) if value is not None else None


def load_spec(path: Path, parser: argparse.ArgumentParser) -> dict:
    """{dest: value} from a run-spec file (TOML via tomllib, or JSON),
    validated against the parser's flags (keys by flag name, '-' or '_')."""
    raw = Path(path).read_bytes()
    if Path(path).suffix.lower() == ".json":
        import json

        table = json.loads(raw.decode("utf-8"))
    else:
        try:
            import tomllib
        except ImportError as e:  # Python < 3.11
            raise SystemExit(f"--spec {path}: TOML needs Python >= 3.11 ({e}); use a .json spec") from e
        table = tomllib.loads(raw.decode("utf-8"))
    actions = {a.dest: a for a in parser._actions if a.dest not in ("help", "spec")}
    out: dict = {}
    unknown = []
    for key, value in table.items():
        dest = str(key).lstrip("-").replace("-", "_")
        if dest not in actions:
            unknown.append(key)
            continue
        out[dest] = _spec_value(actions[dest], value)
    if unknown:
        raise SystemExit(f"--spec {path}: unknown flag(s) {unknown}")
    return out


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    """Parse train.py's command line. With --spec FILE, the file's values
    become the defaults and the command line still wins; a required flag
    (the network size) may come from either."""
    parser = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--spec", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    if known.spec is not None:
        values = load_spec(known.spec, parser)
        for action in parser._actions:
            if action.dest in values:
                action.required = False
        parser.set_defaults(**values)
    return parser.parse_args(argv)


def build_parser() -> argparse.ArgumentParser:
    """train.py's argument parser (every flag and its help text)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-updates", type=int, default=100_000_000)
    parser.add_argument(
        "--train-seconds",
        type=float,
        default=0.0,
        help="Wall-clock budget in seconds; overrides --num-updates when > 0.",
    )
    # The network size is REQUIRED (2026-09-28, ML-022): "never train a size by
    # accident" (CLAUDE.md) -- an unflagged run once trained a default for
    # days and was mistaken for another size. Actor AND critic.
    parser.add_argument(
        "--hidden-dim", type=int, required=True,
        help="Actor hidden width (required; e.g. 1024 for vSix6, 32 for vMin3).",
    )
    parser.add_argument(
        "--num-layers", type=int, required=True,
        help="Actor depth: >= 3 = input layer + residual blocks (required).",
    )
    parser.add_argument(
        "--spec", type=Path, default=None,
        help="A run spec: a TOML (or JSON) table of flag values, keyed by flag "
        "name (`hidden-dim = 1024` or `hidden_dim = 1024`; booleans true/false). "
        "The file supplies the defaults and the command line still wins, so "
        "one readable file per stem holds its recipe (ML-006). Unknown keys "
        "are refused.",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs"),
        help="Directory of this run's own files, all named after the "
        "--checkpoint stem: <stem>.metrics.jsonl (one JSON object per update), "
        "<stem>.heartbeat, <stem>.launches.jsonl (provenance of every launch), "
        "<stem>.threads.txt (live torch thread count), <stem>.resources.jsonl "
        "(--resource-sampler). Default runs/ (relative to the working dir).",
    )
    parser.add_argument(
        "--max-consecutive-rollbacks", type=int, default=20,
        help="Exit with code 3 (a LIVELOCK) after this many consecutive updates "
        "were rolled back or refused: no parameter has moved since, so the "
        "guardian's crash logic (relaunch, then stop after repeated crashes) "
        "takes over instead of the run burning the GPU forever. 0 = never.",
    )
    parser.add_argument(
        "--resource-sampler", action="store_true",
        help="Run the 1 Hz CPU/RAM/GPU sampler for the whole run (always on "
        "with --profile-one-update); writes <run-dir>/<stem>.resources.jsonl.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="AdamW decoupled weight decay on every tensor. 0.01 = AdamW's own "
        "default, what every stem to date trained with (ML-051). A stem-boundary "
        "decision.",
    )
    parser.add_argument(
        "--critic-q-norm-minibatch", action="store_true",
        help="With --critic-q-norm and --micro-batch-rows: normalize by the WHOLE "
        "minibatch's return variance (one number per step) instead of each "
        "chunk's own, so a micro-batched step is the same gradient as the "
        "unchunked one (ML-011). Changes the numerics of micro-batched q-norm "
        "runs slightly -> enable at a relaunch.",
    )
    parser.add_argument(
        "--compile-critic-train", action="store_true",
        help="CUDA + Triton: torch.compile the critic's training forwards "
        "(train_outputs / q_values), like evaluate() already is (ML-013). Not "
        "bit-exact vs the eager kernels -> enable at a relaunch; measure with "
        "PLO5BP_STEP_TIMERS=1.",
    )
    parser.add_argument(
        "--obs-mode",
        choices=["full", "minimal"],
        default="full",
        help="Observation layout. full=OBS_DIM 1171 (default). minimal=bare table-visible 796 (cards, street, active/all-in, stacks, pot/to_call/min/max, commits, seat-exists, button, history). Cold-start only; no warm-start from full-obs checkpoints. Skips opp-outcome MC for speed.",
    )
    parser.add_argument(
        "--obs-real-f16", action="store_true",
        help="Store the compact rollout rows' real columns as float16 (~1.9x "
        "the rows per GiB on the full layout; not bit-exact -- see "
        "TrainingConfig.obs_real_f16). For the longest rollouts.",
    )
    parser.add_argument(
        "--no-compact-obs",
        action="store_true",
        help="Store rollout observations as dense float32 rows instead of the "
        "compact layout (0/1 columns as bits, the rest verbatim f32 — "
        "plo5bp/compact_obs.py). Compact storage is bit-exact (training is "
        "unchanged) and ~7x smaller on --obs-mode minimal (2.4x full), which "
        "lets --rollout-length grow; this flag exists for A/B and debugging.",
    )
    parser.add_argument(
        "--no-batched-opponents",
        action="store_true",
        help="Call each opponent-pool snapshot separately every rollout step "
        "instead of ONE stacked forward + ONE sampling pass for all of them "
        "(rollout._StackedOpponents). Same per-row policy either way; the "
        "stacked call saves up to pool-size x the fixed GPU launch cost per "
        "step. For A/B and debugging.",
    )
    parser.add_argument(
        "--sizing-head",
        choices=["anchor", "logistic", "mixture"],
        default=None,  # C2 sentinel — resolved by _apply_v6_preset (legacy "anchor")
        help="Sizing-head architecture. 'anchor' = v2 flat 11-way categorical "
        "(head_version 2). 'logistic' = v4 ordinal discretized-logistic over the "
        "same 11 anchors (head_version 3): location+scale, stable under PPO, with "
        "the min/pot end anchors tail-absorbed so they stay hittable. "
        "'mixture' = v5 K-component mixture of discretized logistics "
        "(head_version 4): multi-modal solver-style size menus, exact "
        "closed-form marginal (V5_DESIGN.md §2).",
    )
    parser.add_argument(
        "--mixture-k",
        type=int,
        default=3,
        help="Component count for --sizing-head mixture (ignored otherwise).",
    )
    parser.add_argument(
        "--value-clip",
        type=float,
        default=0.2,
        help="PPO clipped-value-loss radius in RAW bb (V5_DESIGN.md B4: 0.2 "
        "against ±250bb returns rate-limits the critic; A/B {2, 10, 1e9} on "
        "a throwaway stem before changing production runs). <= 0 DISABLES "
        "value clipping (plain MSE) — before 2026-09-20 a literal 0 was a "
        "zero-width radius that froze the critic. Ignored by the "
        "distributional head (--value-bins > 0), which has no clip.",
    )
    parser.add_argument(
        "--q-aux-coef",
        type=float,
        default=None,  # C2 sentinel — resolved by _apply_v6_preset (legacy 0.0)
        help="Coefficient for the critic's auxiliary Q(s,a) regression "
        "(dueling head, v5 stems). 0 = head exists (mixture runs) but "
        "untrained; the Expected-SARSA advantage flip (VRPO, W2.5) needs "
        "it warmed first.",
    )
    parser.add_argument(
        "--q-pooled",
        action=argparse.BooleanOptionalAction,
        default=None,  # C2 sentinel — resolved by _apply_v6_preset (legacy False)
        help="Pool the dueling head's per-anchor raise columns into ONE "
        "raise column (q_actions=3: Fold/CheckCall/Raise). 2026-07-09 Q-head "
        "audit: the 11 anchor columns saw ~3%% of rows each and dominated "
        "the VRPO advantage noise; pooling gives the raise Q 11x the "
        "training density. Warm-starting across widths drops adv_head to "
        "fresh zero-init (Q==V; VRPO==GAE until retrained).",
    )
    parser.add_argument(
        "--q-fold-sup-coef",
        type=float,
        default=None,  # C2 sentinel — resolved by _apply_v6_preset (legacy 0.0)
        help="Dense fold-column supervision weight inside the q-aux loss: "
        "fold's forward return is EXACTLY 0 (per-step-cost rewards, sunk "
        "chips excluded), so q[FOLD] regresses to 0 on every fold-LEGAL "
        "row — free perfect labels, ~3x the fold-column data. 0 = off.",
    )
    parser.add_argument(
        "--q-fold-zero",
        action=argparse.BooleanOptionalAction,
        default=False,  # v7 candidate (V7_DESIGN.md WS1.1) — NOT in --v6
        help="Pin Q[FOLD] to its known truth (exactly 0) by construction "
        "instead of supervising it there. Kills the terminal fold-subsidy "
        "class outright; costs an init-era transient (E_pi[Q(s')] under-"
        "reads V^pi by ~pi_fold*V until the sibling columns specialize). "
        "Fresh stems / deliberate experiments only; warm-starts across a "
        "flip are refused.",
    )
    parser.add_argument(
        "--q-base-raw",
        action=argparse.BooleanOptionalAction,
        default=False,  # v7 candidate (V7_DESIGN.md WS1.2) — NOT in --v6
        help="Compose the dueling base in RAW-return space (sum p_i*"
        "symexp(c_i) over the HL-Gauss bins) instead of the display V "
        "(symexp of the symlog-space mean). Removes the estimator-space "
        "Jensen gap that surfaced as the July family offsets. Requires "
        "--value-bins>0; warm-starts across a flip are refused.",
    )
    parser.add_argument(
        "--advantage-estimator",
        choices=["gae", "vrpo"],
        default=None,  # C2 sentinel — resolved by _apply_v6_preset (legacy "gae")
        help="Policy-gradient advantage estimator. 'gae' (default) = V-based "
        "GAE(lambda), unchanged. 'vrpo' = Expected-SARSA(lambda) off the "
        "critic's dueling Q head (VRPO, Fan & Farina 2026; V5_DESIGN.md W2.5) "
        "— analytically averages out future-action-sampling variance at mixed "
        "nodes. Requires --sizing-head mixture AND --q-aux-coef>0 (warm the Q "
        "head first); at the zero-init head it reduces exactly to GAE.",
    )
    parser.add_argument(
        "--torso-norm",
        action=argparse.BooleanOptionalAction,
        default=None,  # C2 sentinel — resolved by _apply_v6_preset (legacy False)
        help="Insert pre-activation LayerNorm into the residual torso of BOTH "
        "actor and critic (v6 plasticity, V6_RESEARCH.md #4). Fresh stem only "
        "(not function-preserving; needs --num-layers>=3). Pair with "
        "--l2-init-coef>0 — LayerNorm-solo can hurt generalization (Nauman 2024).",
    )
    parser.add_argument(
        "--l2-init-coef",
        type=float,
        default=None,  # C2 sentinel — resolved by _apply_v6_preset (legacy 0.0)
        help="Weight-decay-to-init coefficient: L2 penalty pulling the trunk "
        "weight matrices toward their run-start values (the required companion "
        "for --torso-norm). 0 = off.",
    )
    parser.add_argument(
        "--adam-b2",
        type=float,
        default=0.999,
        help="AdamW second-moment beta2 (V6 internals). Sweep {0.98,0.99,0.999} "
        "against heavy-tailed policy-ratio spikes; 0.999 = current default.",
    )
    parser.add_argument(
        "--agc-clip",
        type=float,
        default=None,  # C2 sentinel — resolved by _apply_v6_preset (legacy 0.0)
        help="Stateless per-tensor adaptive gradient-clip coefficient (NFNet "
        "AGC): clip each param's grad to agc_clip*||param||. 0 = off; "
        "rollback-safe (no running state).",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,  # C2 sentinel — resolved by _apply_v6_preset (legacy False)
        help="Recompute torso activations in backward (identical math, less "
        "memory) to buy back rollout headroom. Trains slower per step.",
    )
    parser.add_argument(
        "--value-bins",
        type=int,
        default=None,  # C2 sentinel — resolved by _apply_v6_preset (legacy 0)
        help="Distributional/HL-Gauss critic value head with this many bins "
        "over a symlog support (V6 keystone). 0 = scalar MSE head (default). "
        "Try 51. Fresh critic value head on warm-start.",
    )
    parser.add_argument(
        "--value-support",
        type=float,
        default=1500.0,
        help="Max |value| in bb the distributional support covers (via symlog).",
    )
    parser.add_argument(
        "--value-hlgauss-sigma",
        type=float,
        default=0.75,
        help="HL-Gauss Gaussian sigma in bin-widths (->0 = hard two-hot; A/B "
        "small first).",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="Weight on the critic value loss in the total loss (0.5 = the old "
        "hardcoded value). Re-tune for the distributional head (cross-entropy "
        "!= MSE magnitude); also the critic-weight-lift A/B.",
    )
    parser.add_argument(
        "--clip-prob-dependent",
        action=argparse.BooleanOptionalAction,
        default=None,  # C2 sentinel — resolved by _apply_v6_preset (legacy False)
        help="v6 probability-dependent GATE clip (Over-mixing §6, generalized "
        "Clip-Higher): widen the clip band for RARE gate actions (fast recovery "
        "of a suppressed-but-correct check/bet) and tighten it near 50/50 (less "
        "thrash at genuinely-mixed nodes), keyed on the gate's old prob. Scoped "
        "to the gate so the sizing menu isn't over-loosened. Off = flat --clip.",
    )
    parser.add_argument(
        "--clip-room-ext",
        type=float,
        default=None,  # sentinel: 0.10, and an error without --clip-prob-dependent
        help="Target absolute prob-movement room at the gate extremes (p->0/1) "
        "for --clip-prob-dependent. Default 0.10 = ~10 points/update.",
    )
    parser.add_argument(
        "--clip-room-mid",
        type=float,
        default=None,  # sentinel: 0.05, and an error without --clip-prob-dependent
        help="Target absolute prob-movement room at a 50/50 gate for "
        "--clip-prob-dependent. Default 0.05 = ~5 points/update (tighter than "
        "the extremes -> the symmetric U).",
    )
    parser.add_argument(
        "--clip-prob-floor",
        type=float,
        default=1e-3,
        help="Floor on the gate prob in R/p for --clip-prob-dependent; caps the "
        "max ratio at ~1 + clip_room_ext/floor.",
    )
    parser.add_argument(
        "--v6",
        action="store_true",
        help="V6 PRESET: turn the whole v6 feature kit ON together (sizing-head "
        "mixture, advantage-estimator vrpo + q-aux, torso LayerNorm + l2-init, "
        "distributional value head, AGC, grad-checkpoint, probability-dependent "
        "gate clip). Sets each only where you did NOT pass it explicitly (your "
        "flags win — INCLUDING flags passed at their default value, and the "
        "booleans accept --no-<flag> to force a feature off under --v6); "
        "prints the resolved set. Fresh cold-start stem (not "
        "function-preserving). Use for v6 launches so no feature is silently "
        "left off (cf. the 2048x4 rule in CLAUDE.md).",
    )
    parser.add_argument(
        "--ev-runout-samples",
        type=int,
        default=EV_RUNOUT_SAMPLES,
        help="MC runout samples for the terminal-reward EV when a hand "
        "closes before the river with 2+ live seats (cuts runout luck "
        f"from the reward). Default {EV_RUNOUT_SAMPLES}; higher = lower "
        "reward variance at more engine time (measure — engine is a few %% "
        "of update wall-clock).",
    )
    parser.add_argument("--num-envs", type=int, default=1536)
    parser.add_argument("--rollout-length", type=int, default=262_144)
    parser.add_argument(
        "--micro-batch-rows",
        type=int,
        default=0,
        help="PPO micro-batching: process each minibatch in chunks of at most "
        "this many rows, accumulating gradients (per-row means weighted by the "
        "chunk's share), so --rollout-length can grow past what one "
        "minibatch's working set allows in GPU memory at a fixed "
        "--num-minibatches. The same gradient mathematically, not bit-identical "
        "(float summation order). 0 = off (the exact original path).",
    )
    parser.add_argument(
        "--batch-on-host",
        action="store_true",
        help="Keep the collected batch in host RAM for PPO: each minibatch "
        "(or --micro-batch-rows chunk) is gathered on the CPU into pinned "
        "staging and copied to the GPU. Bit-identical updates; the rollout "
        "is then bounded by host RAM instead of GPU memory. Needs "
        "--mix-configs.",
    )
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
    parser.add_argument(
        "--pool-anchors",
        type=str,
        default="",
        help="Comma list of update AGES (e.g. 25,50,100,200): besides the "
        "FIFO opponent pool (8 snapshots every --snapshot-every updates = a "
        "~40-update memory, which lets self-play drift), the run's numbered "
        "checkpoint nearest each age back joins the pool as an ANCHOR "
        "(refreshed on the snapshot grid; a young run has fewer). Needs the "
        "numbered checkpoints on disk (--checkpoint-every). Default off = the "
        "FIFO alone. A training behavior change: new stems / experiments.",
    )
    parser.add_argument(
        "--crn-streams",
        action="store_true",
        help="Common random numbers for paired comparisons (recipe waves, "
        "tuning runs): the update's table configs, its PPO shuffles and, per "
        "(env, hand), the deals / buttons / opponent assignments each come "
        "from their own stream keyed by (--seed, update[, sub-rollout]), so "
        "candidates that differ only in one knob play the same tables and "
        "hands whatever their hand lengths (the shared stream decorrelates "
        "them from the first finished hand on). Needs --batched. Not "
        "bit-exact with the default stream; use it for NEW runs (pair it with "
        "--number-by-count so a relaunch never reuses an update's key).",
    )
    parser.add_argument(
        "--number-by-count",
        action="store_true",
        help="Number checkpoint files, update_counter stamps, pool snapshot "
        "tags, the heartbeat and metrics by the COUNT of updates done (file "
        "N = the weights after N updates), and save / snapshot on that "
        "global grid: the first update after a relaunch is saved, the "
        "counter never falls one behind per relaunch, and a snapshot's tag "
        "names the file holding its weights. Stamped into checkpoints "
        "(`numbering`); a relaunch from the stem's own file follows that "
        "file's numbering, so a stem never mixes the two. Default off = the "
        "index numbering every stem so far used; start it with a NEW stem.",
    )
    parser.add_argument(
        "--minibatches-from-rows",
        action="store_true",
        help="Size each PPO minibatch from the rows actually COLLECTED -- "
        "ceil(rows / --num-minibatches) -- instead of from the --rollout-"
        "length target, so every epoch is exactly --num-minibatches steps "
        "whatever the drain overshoot (~8-9%%: vSix6 takes ~17 steps, not "
        "16, and the count moves with hand lengths). Also the critic-only "
        "passes' fallback size. Default off = the current numerics; not "
        "bit-exact -- switch at a stem boundary.",
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
    parser.add_argument(
        "--gpu-lock",
        type=str,
        default="",
        help="Share one GPU between concurrent runs (the network-size sweep): "
        "an exclusive flock on this file is held from the moment a finished "
        "rollout batch is copied to the GPU until the update is done and its "
        "memory handed back (torch.cuda.empty_cache), so no two runs' batches "
        "and PPO working sets are ever resident together. Waiting never "
        "changes a value. Linux; empty = off.",
    )
    parser.add_argument("--pool-mix-prob", type=float, default=0.5)
    parser.add_argument("--pool-opp-seats", type=int, default=2)
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.1,
        help="PPO entropy bonus coefficient (every tier's under --mix-"
        "configs). Default 0.1; vSix6 trains at 0.045, vMin3 at 0.07.",
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
        choices=_KNOWN_STACK_DISTS,
        default="uniform",
        help="'uniform' samples within --stack-range; 'clubgg' uses piecewise "
        "weighted bands (Short 5%%, Hover 50%%, Warm 18%%, Big 17%%, Monster 10%%) "
        "clipped to --stack-range; 'clubgg_deep' targets the $0.80-ante game "
        "(1-20:2%%, 20-30:6%%, 30-40:16%%, 40-50:22%%, 50-65:25%%, 65-80:22%%, 80-120:7%%); "
        "'clubgg_mix' picks 50/50 between clubgg and clubgg_deep per config; "
        "'deep' samples each seat uniformly in 100-250bb (ignores --stack-range); "
        "'full_mix' picks 1/3 each between clubgg, clubgg_deep, and deep per config; "
        "'clubgg_real' = the owner's own ClubGG tables measured from their hand histories "
        "(2026-10-03): per-seat bands with the 20bb buy-in spike, clipped to --stack-range, "
        "AND the real seat counts (6:41.5/5:31.2/4:20.9/3:5.2/2:1.2 %%, --seats-dist does "
        "not apply to it).",
    )
    parser.add_argument(
        "--mix-configs",
        action="store_true",
        help="vThree mode: each update mixes --configs-per-tier (seats,stacks) "
        "draws from EACH of --mix-tiers (default all three stack tiers), instead "
        "of one config per update. The gradient averages over all N configs, "
        "removing the consecutive-shallow exposure that saturated the gate "
        "under the retired block rotation. Requires --batched.",
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
        help="Chips per bb (default 10000 -> cent precision at $20/bb).",
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
        "--allow-obs-rev-change",
        action="store_true",
        help="Permit --load-checkpoint across an observation-SEMANTICS "
        "revision change (checkpoint `obs_rev` != this process's "
        "plo5bp.encoding.OBS_SEMANTICS_REV, env PLO5BP_OBS_REV; an unstamped "
        "checkpoint counts as rev 1 = trained before the 2026-09-20 feature "
        "fixes). PRODUCTION BEHAVIOR CHANGE: the stem migrates to the new "
        "feature values at unchanged widths — dims 186/187, 800/802, "
        "999-1006, 1024-1029, 1040-1041 change meaning under the loaded "
        "weights; expect a transient. Refused without this flag; to continue "
        "a stem byte-compatibly set PLO5BP_OBS_REV=<checkpoint rev> instead. "
        "Older-rev siblings are not seeded into the warm-start pool.",
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
        required=True,
        help="Hidden width of the centralized critic (training-only value "
        "net that sees all hole cards). Required, like the actor's size.",
    )
    parser.add_argument(
        "--critic-num-blocks",
        type=int,
        required=True,
        help="Residual blocks in the centralized critic torso (required).",
    )
    parser.add_argument(
        "--critic-act", choices=("relu", "silu", "gelu"), default="relu",
        help="Critic torso activation (2026-09-26 redesign: silu cannot die the "
        "way the vSix5 critic's ReLUs did).",
    )
    parser.add_argument(
        "--critic-in-norm", action=argparse.BooleanOptionalAction, default=False,
        help="LayerNorm between the critic's input Linear and its activation.",
    )
    parser.add_argument(
        "--critic-v-raw", action=argparse.BooleanOptionalAction, default=False,
        help="Read the distributional critic's V as the mean of its predicted "
        "return distribution (unbiased) instead of symexp(E[symlog]) (reads "
        "high-variance states too low).",
    )
    parser.add_argument(
        "--critic-extra-epochs", type=int, default=0,
        help="Critic-only passes over each rollout after the PPO epochs.",
    )
    parser.add_argument(
        "--critic-minibatches", type=int, default=0,
        help="Minibatches per critic-only epoch (0 = --num-minibatches): more, "
        "smaller critic steps per rollout.",
    )
    parser.add_argument(
        "--critic-q-norm", action=argparse.BooleanOptionalAction, default=False,
        help="Divide the Q regression losses by the minibatch return variance "
        "(+1) so they do not drown the value cross-entropy in the critic torso.",
    )
    parser.add_argument(
        "--critic-fresh", action="store_true",
        help="With --load-checkpoint: do not load the checkpoint's critic (start "
        "the critic from random init; the actor's Adam moments still restore).",
    )
    parser.add_argument(
        "--critic-init", type=Path, default=None,
        help="With --load-checkpoint: load the critic from this file instead "
        "(a checkpoint's 'critic' or a bare critic state dict).",
    )
    parser.add_argument(
        "--actor-freeze-updates", type=int, default=0,
        help="The first N updates of this run train ONLY the critic (the actor "
        "is frozen): a new critic learns before its advantages steer the actor.",
    )
    parser.add_argument(
        "--kl-anchor-coef",
        type=float,
        default=0.0,
        help="KL-to-EMA-reference regularizer coefficient. 0 disables "
        "(no EMA model is built). The reference IS persisted: every "
        "checkpoint carries it as `model_ema` and a warm start restores it, "
        "so the magnet's memory survives relaunches (it only re-initializes "
        "to the loaded weights, ramping in over ~1/(1-ema) updates, when the "
        "checkpoint predates the key). PLO5BP_SERVE_EMA=1 serves it in the UI.",
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
        "--critic-lr",
        type=float,
        default=0.0,
        help="Learning rate of the centralized critic in its own AdamW param "
        "group (follows the warmup ramp and live lr edits at the same ratio "
        "to --lr). 0 = the critic trains at --lr in the actor's group (the "
        "exact original path).",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=TrainingConfig.lam,
        help="lambda of the advantage estimator (GAE(lambda), or VRPO's "
        "Expected-SARSA(lambda) under --v6): how far back a later reward is "
        "credited. 1.0 = the full hand's outcome, lower = lean on the critic "
        "sooner. Default %(default)s (every stem to date).",
    )
    parser.add_argument(
        "--lr-warmup-updates",
        type=int,
        default=0,
        help="Linear LR warmup over the first N updates of EACH launch (the "
        "loop counter restarts at every relaunch, so a warm restart ramps "
        "again; pass 0 there -- the production guardians do). 0 disables. "
        "Cold-start Adam steps at full LR moved the policy "
        "by KL 1-20 per minibatch, tripping the KL guard at mb1-2 and "
        "starving the critic (vTwo1 2026-06-11); small early steps let "
        "the full inner loop run.",
    )
    parser.add_argument(
        "--adv-clip",
        type=float,
        default=8.0,
        help="Clamp normalized advantages to ±N std before the PPO loss "
        "(0 disables). PPO clips the ratio, not the advantage weight; "
        "deep-stack all-in pots produce 30-std+ samples that carry 30x "
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
        help="Sizing-entropy scale (every anchor head, v2..v5): multiplies the "
        "sizing heads' (anchor + refine) entropy bonus relative to the gate's. "
        "1.0 = the same coefficient as the gate; < 1 lets raise sizes "
        "differentiate (vMin3 0.1, vSix6 0.3), > 1 resists over-sharpening. "
        "The log's H= is the policy's own entropy either way; Hbonus= the "
        "bonus-weighted one. Live-tunable via the control file "
        '{"sizing_entropy_scale": X}.',
    )
    parser.add_argument(
        "--kl-anchor-ema",
        type=float,
        default=None,  # sentinel: 0.999, and an error without --kl-anchor-coef
        help="EMA decay of the KL reference model (default 0.999; needs "
        "--kl-anchor-coef > 0).",
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
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/train_run.pt"),
        help="Final-save path; mid-run saves are <stem>_<update>.pt beside it "
        "and the optimizer sidecar <stem>.optim.pt. Default "
        "checkpoints/train_run.pt — it used to be checkpoints/stub.pt, the "
        "file the UI SERVES, so a bare smoke / --profile-one-update run "
        "overwrote the live PLO5 model at its final save (review 2026-09-20 "
        "A11). Promote deliberately: cp <run>.pt checkpoints/stub.pt.",
    )
    parser.add_argument(
        "--allow-overwrite-stub",
        action="store_true",
        help="Permit --checkpoint to name a UI-served file (stub.pt / "
        "nlh_stub.pt). Refused otherwise.",
    )
    parser.add_argument(
        "--optimizer-sidecar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist Adam moments + the l2-init reference tensors in ONE "
        "rolling <stem>.optim.pt next to the checkpoints (rewritten at every "
        "save) and restore them on --load-checkpoint, so a relaunch resumes "
        "with warm Adam instead of a first step of ~lr*sign(g), and "
        "decay-to-init keeps pulling toward the stem's ORIGINAL init. Moments "
        "restore only when the sidecar's update_counter equals the loaded "
        "checkpoint's. --no-optimizer-sidecar = the pre-2026-09-20 behavior "
        "(nothing written, nothing read: cold Adam, init = relaunch point).",
    )
    parser.add_argument(
        "--drain-inflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Once the rollout row target is reached, stop re-dealing and "
        "play every in-flight hand to completion so it lands in the batch "
        "(unbiased in hand length). The batch then runs ~n_envs x one hand's "
        "rows PAST --rollout-length (+2.5-8%% at the vSix4 ratio) — size GPU "
        "memory / --rollout-length accordingly. --no-drain-inflight = the "
        "pre-2026-09-20 collector, byte-identical: exit at the target and "
        "DROP the in-flight hands (long hands under-sampled by ~len/W).",
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
    parser.add_argument(
        "--profile-at-update",
        type=int,
        default=0,
        help="With --profile-one-update, which update index to profile. >0 skips "
        "the one-time torch.compile/Inductor JIT (fires on the first forward/"
        "backward) so the trace is STEADY-STATE, not compilation. Warmup updates "
        "run normally, then the chosen update is profiled and the process exits.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Torch intra-op thread count for host-side rollout/post-rollout CPU "
        "work. The _concat_batches staging copy is memory-bandwidth-bound and runs "
        "~5x slower at the 192-thread default (host cores oversubscribing the "
        "pod's ~40-vCPU quota); ~8-32 is optimal. 0 = OMP_NUM_THREADS if set, "
        "else min(32, cpu_count) on CUDA / torch default on CPU. Applies on cpu "
        "AND cuda; also live-tunable via <run-dir>/<stem>.threads.txt.",
    )
    parser.add_argument(
        "--rayon-threads",
        type=int,
        default=0,
        help="Thread count for the Rust engine's rayon pool (opp-outcome MC + "
        "obs encoder). 0 = leave rayon's default / any pre-set RAYON_NUM_THREADS "
        "untouched. MEASURED by the owner on the pod: a cap of 40 vs unset made "
        "no meaningful difference (marginally slower) — rayon's default already "
        "follows the container's CPU quota (Rust's available_parallelism is "
        "cgroup-aware), so it never oversubscribed. Changes no training numbers "
        "(per-env deterministic MC seeds).",
    )
    return parser
