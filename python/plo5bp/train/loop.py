"""train.py's main(): setup, warm start, the update loop and the final save."""

from __future__ import annotations

import dataclasses
import math
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
    VARIANT_PLO5,
    TrainingConfig,
)
import plo5bp.encoding as _encoding
from plo5bp.encoding import OBS_DIM, OBS_DIM_MINIMAL
from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.network import (
    ActorCriticV2,
    ActorCriticV4,
    ActorCriticV5,
    CentralCritic,
)
from plo5bp.sizing import NLH_ANCHOR_SPEC, PLO_ANCHOR_SPEC
from plo5bp.network import actor_arch, critic_arch
from plo5bp.ppo import PPOStats, PPOTrainer, value_health
from plo5bp.rollout import (
    collect_rollout,
    collect_rollout_batched,
    collect_rollout_multiconfig,
)
from plo5bp.selfplay import (
    OpponentPool,
    discover_checkpoint_family,
    refresh_pool_anchors,
    seed_pool_from_checkpoints,
)
from plo5bp.train.checkpoint import (
    NUMBERING_COUNT,
    NUMBERING_INDEX,
    _UI_SERVED_CHECKPOINTS,
    _atomic_torch_save,
    _restore_optimizer_sidecar,
    _save_optimizer_sidecar,
    append_launch_record,
    assert_finite_for_save,
    build_checkpoint,
    resolve_numbering,
    run_provenance,
)
from plo5bp.train.cli import (
    _apply_v6_preset,
    _parse_mix_tiers,
    _parse_seats_range,
    _parse_stack_range,
    parse_args,
    validate_flag_combinations,
)
from plo5bp.train.control import (
    _apply_anneal_control,
    _read_control_text,
    _warn_once,
)
from plo5bp.train.diagnostics import _dump_batch_diagnostics
from plo5bp.train.metrics import Heartbeat, MetricsWriter, _ResourceSampler
from plo5bp.train.tiers import _sample_game_config


# Consecutive rolled-back updates before (and between) livelock alarms (A3).
_ROLLBACK_ALARM_AFTER = 5

def _parse_anchor_ages(spec: str) -> tuple:
    """--pool-anchors "25,50,100" -> (25, 50, 100); "" -> ()."""
    try:
        ages = tuple(sorted({int(t) for t in str(spec).split(",") if t.strip()}))
    except ValueError:
        raise SystemExit(f"--pool-anchors must be a comma list of update ages, got {spec!r}")
    if any(a <= 0 for a in ages):
        raise SystemExit(f"--pool-anchors ages must be > 0, got {spec!r}")
    return ages


def _lr_warmup_scale(update: int, warmup_updates: int) -> float:
    """Linear LR ramp over the first `warmup_updates` updates: scale
    runs from 1/warmup_updates up to 1.0, then stays at 1.0. 0 disables
    (always 1.0). A pure function of `update` -- which the loop passes as
    its LOOP counter (0 at every launch, see base_update), so a warm
    relaunch ramps again; the production guardians pass 0."""
    if warmup_updates <= 0 or update >= warmup_updates:
        return 1.0
    return (update + 1) / warmup_updates


class _GpuPhaseLock:
    """--gpu-lock: cross-process exclusive flock around the GPU-heavy part of
    an update (batch copied to the GPU -> PPO -> batch dropped). `acquire` is
    the rollout module's GPU_PHASE_HOOK (idempotent); `release` returns the
    cached GPU memory to the driver before letting the next run in."""

    def __init__(self, path: str) -> None:
        import fcntl

        self._fcntl = fcntl
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a+")
        self.held = False

    def acquire(self) -> None:
        if self.held:
            return
        t0 = time.perf_counter()
        self._fcntl.flock(self._fh.fileno(), self._fcntl.LOCK_EX)
        self.held = True
        waited = time.perf_counter() - t0
        if waited > 1.0:
            print(f"        [gpu-lock] waited {waited:.1f}s for {self._path}", flush=True)

    def release(self) -> None:
        if not self.held:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._fcntl.flock(self._fh.fileno(), self._fcntl.LOCK_UN)
        self.held = False


# Exit status of a run stopped as a rollback LIVELOCK (--max-consecutive-
# rollbacks): distinct from a crash (1) so a guardian can tell them apart.
LIVELOCK_EXIT_CODE = 3

# PPOStats fields that average APPLIED steps only: the guards refuse every
# non-finite step, so a non-finite value here means a bug, not bad luck.
_FINITE_STATS = (
    "policy_loss", "value_loss", "display_loss", "entropy", "entropy_bonus",
    "approx_kl", "kl_k3", "kl_anchor", "q_loss", "critic_value_loss",
    "critic_q_loss",
)


def _check_stats_finite(stats: PPOStats) -> None:
    """ML-001: an explicit check over every averaged PPOStats field (the old
    `assert`s read only two, and vanish under `python -O`)."""
    bad = {
        f: getattr(stats, f) for f in _FINITE_STATS
        if not math.isfinite(float(getattr(stats, f)))
    }
    if bad:
        raise RuntimeError(
            f"non-finite update statistics {bad}: a non-finite value reached an "
            "APPLIED step. Stopping before anything is saved; the guardian "
            "resumes from the last good checkpoint."
        )


def _health_line(stats: PPOStats, vh: dict) -> str:
    """One log line of per-update health numbers (ML-003 / ML-010 / ML-012)."""
    a = vh.get("all", {})
    parts = [f"EV={a.get('ev', float('nan')):+.3f} bias={a.get('bias', float('nan')):+.2f}bb"]
    st = vh.get("street") or {}
    if st:
        parts.append("EV(F/T/R)=" + "/".join(
            f"{st[k]['ev']:+.2f}" if k in st else "--" for k in ("flop", "turn", "river")
        ))
    parts.append(f"clip={100.0 * stats.clip_frac:.1f}%")
    parts.append(f"k3={stats.kl_k3:.4f}")
    parts.append(f"kl0={stats.kl0:+.1e} |r-1|0={stats.ratio_dev0:.1e}")
    parts.append(
        f"g a/c/d={stats.grad_norm_actor:.3g}/{stats.grad_norm_critic:.3g}/"
        f"{stats.grad_norm_display:.3g} clipped a/c="
        f"{100.0 * stats.grad_clip_actor:.0f}/{100.0 * stats.grad_clip_critic:.0f}%"
    )
    if stats.critic_steps or stats.critic_skipped:
        parts.append(
            f"crit={stats.critic_steps} steps v={stats.critic_value_loss:.4f}"
            + (f" SKIPPED={stats.critic_skipped}" if stats.critic_skipped else "")
        )
    if stats.nonfinite_grad:
        parts.append("NONFINITE-GRAD")
    parts.append(f"shuffle={stats.shuffle_s:.1f}s")
    return "        [health] " + "  ".join(parts)


def main(argv: "list[str] | None" = None) -> None:
    args = parse_args(argv)

    # A11: never let a training run's FINAL save land on a file the UI
    # serves unless that is explicitly what was asked for.
    if args.checkpoint.name in _UI_SERVED_CHECKPOINTS and not args.allow_overwrite_stub:
        raise SystemExit(
            f"error: --checkpoint {args.checkpoint} is a UI-served model file "
            f"({sorted(_UI_SERVED_CHECKPOINTS)}); the final save would overwrite "
            "it. Train to another path and promote with `cp`, or pass "
            "--allow-overwrite-stub if you really mean it."
        )

    # --v6 preset resolution (C2): sentinel defaults + _apply_v6_preset (module
    # level, above main) — explicit flags win FOR REAL now, including ones
    # passed at their default value and the --no-<flag> boolean forms (the old
    # parser.get_default comparison couldn't see "explicitly passed the
    # default" and silently overrode ablation flags). Runs on every
    # invocation; non-v6 runs just get the legacy defaults filled in.
    _v6_applied, _v6_kept = _apply_v6_preset(args)
    if args.v6:
        print(f"[v6] preset ON - applied: {_v6_applied}")
        if _v6_kept:
            print(f"[v6] kept your explicit overrides: {_v6_kept}")
    # Refuse flag combinations that would silently do nothing (ML-006).
    validate_flag_combinations(args)

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

    if args.batch_size is None:
        if args.num_minibatches <= 0:
            raise SystemExit("--num-minibatches must be > 0")
        args.batch_size = max(
            1,
            (args.rollout_length + args.num_minibatches - 1) // args.num_minibatches,
        )
        if args.minibatches_from_rows:
            print(
                f"[batch-size] ceil(collected rows / {args.num_minibatches}) per "
                f"update (--minibatches-from-rows; ~{args.batch_size} at the "
                f"{args.rollout_length}-row target)"
            )
        else:
            print(
                f"[batch-size] derived {args.batch_size} from "
                f"rollout_length={args.rollout_length} / num_minibatches={args.num_minibatches}"
            )

    # Validated even when --mix-configs is off: a typo is a typo (A12).
    mix_tiers = _parse_mix_tiers(args.mix_tiers)
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

    # This run's own files are named after the checkpoint stem (ML-060): two
    # trainers on one pod no longer share (and truncate) each other's files.
    run_dir = Path(args.run_dir)
    stem = args.checkpoint.stem
    threads_file = run_dir / f"{stem}.threads.txt"
    current_threads: int | None = None
    # Torch's intra-op thread count governs the host-side rollout AND
    # post-rollout CPU work (the _concat_batches staging copy, h2d assembly)
    # on BOTH cpu and CUDA runs — the whole rollout is CPU-side even when the
    # learner is on cuda. This was formerly gated on device=="cpu", which
    # pinned CUDA runs at the 192-thread default; the _concat_batches obs copy
    # is memory-bandwidth-bound and ran ~5x slower at 192 threads than at its
    # ~8-32-thread optimum (NUMA oversubscription; profiled 2026-07-08). Honor
    # --cpu-threads, else OMP_NUM_THREADS, regardless of device.
    _cpu_threads = int(args.cpu_threads or 0)
    _thread_src = "--cpu-threads" if _cpu_threads > 0 else ""
    if _cpu_threads <= 0:
        omp_env = os.environ.get("OMP_NUM_THREADS", "").strip()
        if omp_env.isdigit() and int(omp_env) > 0:
            _cpu_threads = int(omp_env)
            _thread_src = "OMP_NUM_THREADS"
    if _cpu_threads <= 0 and args.device == "cuda":
        # P4: nothing specified on a CUDA run — apply the measured quota-safe
        # default (32) instead of leaving torch at the host's physical-core count
        # (~192 on the pod). The pod's ~40-vCPU cgroup quota then ~4.7x
        # oversubscribes that, and the memory-bandwidth-bound _concat_batches copy
        # ran ~5x slower (36s@192 vs 6.9s@32, profiled 2026-07-08). min() keeps a
        # smaller CUDA box sane. Override via --cpu-threads / OMP_NUM_THREADS /
        # runs/threads.txt. Thread count changes no training numbers; CPU-only
        # runs are left alone (their compute IS on the CPU pool).
        _cpu_threads = min(32, os.cpu_count() or 32)
        _thread_src = "cuda default (P4)"
    if _cpu_threads > 0:
        torch.set_num_threads(_cpu_threads)
    current_threads = torch.get_num_threads()
    print(
        f"[threads] initial torch threads = {current_threads}"
        + (f" (via {_thread_src})" if _thread_src else "")
        + f"; live-tunable via {threads_file}"
    )
    if args.device == "cuda" and current_threads > 64:
        print(
            f"[threads] WARNING: {current_threads} torch threads on a CUDA run "
            "oversubscribes the pod's ~40-vCPU quota; the host-side rollout + "
            "_concat_batches copy is memory-bandwidth-bound and ~5x slower wide. "
            "Pass --cpu-threads 32 (or edit runs/threads.txt) unless deliberate."
        )

    # P12: cap the Rust engine's rayon pool (opp-outcome MC + obs encoder). Rayon
    # reads RAYON_NUM_THREADS lazily at its first par_iter (the first rollout,
    # after this startup), so setting it here takes effect; Python os.environ
    # writes reach Rust's std::env in-process. Default 0 leaves rayon's default /
    # any pre-set env untouched (byte-identical). Thread count changes no training
    # numbers (per-env deterministic outcome_seed + disjoint-row par writes).
    # Pod A/B (owner): 40 vs unset ≈ no difference — leave it unset by default.
    if int(args.rayon_threads or 0) > 0:
        os.environ["RAYON_NUM_THREADS"] = str(int(args.rayon_threads))
        print(
            f"[threads] rayon threads -> {int(args.rayon_threads)} "
            "(via --rayon-threads)"
        )
    else:
        _rayon_env = os.environ.get("RAYON_NUM_THREADS", "").strip()
        print(
            "[threads] rayon threads = "
            + (
                f"{_rayon_env} (via RAYON_NUM_THREADS env)"
                if _rayon_env.isdigit()
                else "default (~host logical cores)"
            )
        )

    # P13: disable torch.distributions argument/support validation process-wide.
    # Each Categorical/Beta construct + log_prob otherwise runs constraint checks
    # ending in `.all()` -> bool() on a CUDA tensor = a forced stream sync; ~7-9
    # per act() x ~50-100k act() calls/update land on the CPU-bound collection
    # path (GPU ~86% idle). Validation is read-only, so outputs/samples/RNG are
    # byte-identical (the test suite keeps validation ON and still passes). A NaN
    # logit, formerly caught here, now surfaces at the policy/value-loss NaN
    # asserts a few lines downstream.
    torch.distributions.Distribution.set_default_validate_args(False)

    seats_choices = _parse_seats_range(args.num_seats_range, args.variant)
    stack_lo, stack_hi = _parse_stack_range(args.stack_range)

    # A6 drain_inflight: a TrainingConfig field (it rides in every
    # checkpoint's `config` stamp), also handed to the collectors as an
    # explicit kwarg and stamped top-level (`drain_inflight`).
    drain_inflight = bool(args.drain_inflight)
    if not drain_inflight:
        print(
            "[rollout] --no-drain-inflight: LEGACY collection — hands in "
            "flight at the row target are dropped (long hands under-sampled)"
        )

    train_cfg = TrainingConfig(
        drain_inflight=drain_inflight,
        lr=args.lr,
        critic_lr=float(args.critic_lr),
        lam=float(args.gae_lambda),
        num_updates=args.num_updates,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        obs_mode=args.obs_mode,
        compact_obs=not args.no_compact_obs,
        obs_real_f16=args.obs_real_f16,
        micro_batch_rows=int(args.micro_batch_rows),
        batch_on_host=bool(args.batch_on_host),
        batched_opponents=not args.no_batched_opponents,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches if args.minibatches_from_rows else 0,
        crn_streams=bool(args.crn_streams),
        pool_anchor_ages=_parse_anchor_ages(args.pool_anchors),
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        snapshot_every=args.snapshot_every,
        ev_runout_samples=args.ev_runout_samples,
        pool_mix_prob=args.pool_mix_prob,
        pool_opp_seats=args.pool_opp_seats,
        entropy_coef=args.entropy_coef,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_blocks=args.critic_num_blocks,
        critic_act=args.critic_act,
        critic_in_norm=args.critic_in_norm,
        critic_v_raw=args.critic_v_raw,
        critic_extra_epochs=args.critic_extra_epochs,
        critic_minibatches=args.critic_minibatches,
        critic_q_norm=args.critic_q_norm,
        kl_anchor_coef=args.kl_anchor_coef,
        kl_anchor_ema=args.kl_anchor_ema,
        target_kl=args.target_kl,
        kl_hard=args.kl_hard,
        sizing_entropy_scale=args.sizing_entropy_scale,
        adv_clip=args.adv_clip,
        value_clip=args.value_clip,
        q_aux_coef=args.q_aux_coef,
        q_pooled=args.q_pooled,
        q_fold_sup_coef=args.q_fold_sup_coef,
        q_fold_zero=args.q_fold_zero,
        q_base_raw=args.q_base_raw,
        advantage_estimator=args.advantage_estimator,
        torso_layernorm=args.torso_norm,
        l2_init_coef=args.l2_init_coef,
        adam_b2=args.adam_b2,
        agc_clip=args.agc_clip,
        weight_decay=float(args.weight_decay),
        critic_q_norm_minibatch=bool(args.critic_q_norm_minibatch),
        compile_critic_train=bool(args.compile_critic_train),
        grad_checkpoint=args.grad_checkpoint,
        value_bins=args.value_bins,
        value_support=args.value_support,
        value_hlgauss_sigma=args.value_hlgauss_sigma,
        value_loss_coef=args.value_loss_coef,
        clip_prob_dependent=args.clip_prob_dependent,
        clip_room_ext=args.clip_room_ext,
        clip_room_mid=args.clip_room_mid,
        clip_prob_floor=args.clip_prob_floor,
        device=args.device,
    )

    if is_nlh and args.obs_mode == "minimal":
        raise SystemExit("error: --obs-mode minimal is PLO-only")
    if is_nlh:
        obs_dim = OBS_DIM_NLH
    elif args.obs_mode == "minimal":
        obs_dim = OBS_DIM_MINIMAL
    else:
        obs_dim = OBS_DIM
    anchor_spec = NLH_ANCHOR_SPEC if is_nlh else PLO_ANCHOR_SPEC
    head_kwargs: dict = {}
    if args.sizing_head == "mixture":
        model_cls = ActorCriticV5
        head_kwargs["mixture_k"] = int(args.mixture_k)
    elif args.sizing_head == "logistic":
        model_cls = ActorCriticV4
    else:
        model_cls = ActorCriticV2
    model = model_cls(
        hidden_dim=train_cfg.hidden_dim,
        num_layers=train_cfg.num_layers,
        obs_dim=obs_dim,
        anchor_spec=anchor_spec,
        torso_layernorm=train_cfg.torso_layernorm,
        **head_kwargs,
    )
    print(
        f"[head] sizing-head={args.sizing_head} "
        f"(head_version={model.head_version}) variant={args.variant} "
        f"obs_dim={obs_dim} obs_mode={args.obs_mode} anchors={anchor_spec.count} ({anchor_spec.name})"
    )
    # Observation-SEMANTICS revision (2026-09-20): same widths, different
    # feature VALUES. Env PLO5BP_OBS_REV via plo5bp.encoding — unset/2 = the
    # corrected features, 1 = the exact pre-fix values.
    obs_rev = int(_encoding.OBS_SEMANTICS_REV)
    # Rollout observation STORAGE (not a feature change — bit-exact on unpack).
    from plo5bp import rollout as _rollout_mod
    gpu_lock = _GpuPhaseLock(args.gpu_lock) if args.gpu_lock else None
    if gpu_lock is not None:
        _rollout_mod.GPU_PHASE_HOOK = gpu_lock.acquire
        print(f"[gpu-lock] GPU phase of every update serialized on {args.gpu_lock}")
    _obs_layout = _rollout_mod._resolve_obs_layout(train_cfg, args.variant)
    if _obs_layout is None:
        print(f"[obs-storage] dense float32: {4 * obs_dim:,} B per stored observation")
    else:
        print(
            f"[obs-storage] compact ({_obs_layout.name}): {_obs_layout.n_flag} 0/1 "
            f"columns as bits + {_obs_layout.n_real} verbatim {'f16' if _obs_layout.real_dtype == 'float16' else 'f32'} = "
            f"{_obs_layout.row_bytes:,} B per stored observation (dense "
            f"{4 * obs_dim:,} B, {4 * obs_dim / _obs_layout.row_bytes:.1f}x smaller)"
        )
    print(
        f"[obs-rev] observation semantics rev = {obs_rev} "
        f"(PLO5BP_OBS_REV={os.environ.get('PLO5BP_OBS_REV', '')!r}; "
        "stamped into every checkpoint as `obs_rev`)"
    )
    model.to(train_cfg.device)
    # v5 stems build the critic WITH the dueling Q head from day one
    # (zero-init; Q == V until --q-aux-coef trains it) so the VRPO
    # advantage flip later is a code change, not a checkpoint break.
    # --q-pooled collapses the per-anchor raise columns to one (audit
    # 2026-07-09: 11 starving columns dominated the VRPO noise).
    if args.sizing_head == "mixture":
        critic_q_actions = 3 if args.q_pooled else 2 + anchor_spec.count
    else:
        critic_q_actions = 0
    if args.torso_norm and args.l2_init_coef <= 0.0:
        print(
            "[warn] --torso-norm without --l2-init-coef>0: LayerNorm-solo can "
            "hurt generalization (Nauman 2024). Strongly consider a companion, "
            "e.g. --l2-init-coef 1e-4."
        )
    if args.advantage_estimator == "vrpo":
        if critic_q_actions <= 0:
            raise SystemExit(
                "error: --advantage-estimator vrpo requires --sizing-head "
                "mixture (it reads the critic's dueling Q head)."
            )
        if args.q_aux_coef <= 0.0:
            raise SystemExit(
                "error: --advantage-estimator vrpo requires --q-aux-coef > 0 "
                "so the Q head is trained first; at the untrained head the flip "
                "is identical to GAE (V5_DESIGN.md W2.5)."
            )
    critic = CentralCritic(
        obs_dim=obs_dim,
        hidden_dim=train_cfg.critic_hidden_dim,
        num_blocks=train_cfg.critic_num_blocks,
        q_actions=critic_q_actions,
        torso_layernorm=train_cfg.torso_layernorm,
        value_bins=train_cfg.value_bins,
        value_support=train_cfg.value_support,
        hlgauss_sigma=train_cfg.value_hlgauss_sigma,
        q_fold_zero=train_cfg.q_fold_zero,
        q_base_raw=train_cfg.q_base_raw,
        act=train_cfg.critic_act,
        in_norm=train_cfg.critic_in_norm,
        v_raw=train_cfg.critic_v_raw,
    )
    critic.to(train_cfg.device)
    print(f"[device] learner on {train_cfg.device}")
    # Run state restored from the checkpoint (None when absent / cold).
    restored_update: int | None = None
    restored_pool_updates: list | None = None
    restored_control_applied: str | None = None
    restored_numbering: str | None = None
    if args.load_checkpoint is not None:
        ckpt = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        ckpt_variant = str(ckpt.get("variant", VARIANT_PLO5))
        if ckpt_variant != args.variant:
            # Unconditional: even dims-identical pairs (plo4/plo5/plo6
            # share OBS_DIM + the 11-anchor head) are refused. The
            # games' equities and minimum made-hand strengths differ so
            # much by hole-card count that transferred weights are a
            # confused prior, not a head start — every variant trains
            # from scratch (decision 2026-07-03).
            raise SystemExit(
                f"variant mismatch: checkpoint={ckpt_variant} vs "
                f"--variant={args.variant}. Cross-variant warm-starts are "
                "refused: each variant trains from scratch."
            )
        ckpt_obs_mode = str(
            (ckpt.get("config") or {}).get("obs_mode", "full")
            if isinstance(ckpt.get("config"), dict)
            else ckpt.get("obs_mode", "full")
        )
        if ckpt_obs_mode != args.obs_mode:
            raise SystemExit(
                f"obs_mode mismatch: checkpoint={ckpt_obs_mode!r} vs "
                f"--obs-mode={args.obs_mode!r}. Minimal/full layouts are not "
                "warm-start compatible (different obs width + features)."
            )
        ckpt_head = int(ckpt.get("head_version", 1))
        if ckpt_head != model.head_version:
            raise SystemExit(
                f"head_version mismatch: checkpoint={ckpt_head} vs model="
                f"{model.head_version} (selected by --sizing-head). Warm-start "
                "requires a checkpoint of the same sizing-head version; start "
                "cold or point --load-checkpoint at a matching-family checkpoint."
            )
        # A NEW critic (--critic-fresh / --critic-init) does not come from
        # this checkpoint, so only a critic taken FROM it must be there.
        _new_critic = bool(args.critic_fresh or args.critic_init is not None)
        if "critic" not in ckpt and not _new_critic:
            raise SystemExit(
                "v2 checkpoint is missing the 'critic' state dict — cannot "
                "warm-start the centralized critic (--critic-init FILE or "
                "--critic-fresh start a new one)."
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
        # A NEW critic (--critic-fresh / --critic-init) does not come from this
        # checkpoint: its shape / Q-semantics guards do not apply. Otherwise the
        # redesign's choices must match (act / v_raw leave no shape trace, so a
        # silent mismatch would load cleanly and compute something else).
        if not _new_critic:
            for _ck, _cur in (("critic_act", train_cfg.critic_act),
                              ("critic_in_norm", train_cfg.critic_in_norm),
                              ("critic_v_raw", train_cfg.critic_v_raw)):
                _was = ckpt_cfg.get(_ck, {"critic_act": "relu"}.get(_ck, False))
                if _was != _cur:
                    raise SystemExit(
                        f"{_ck} mismatch: checkpoint={_was!r} vs flags={_cur!r}. "
                        "Pass --critic-fresh (or --critic-init) to start a new critic."
                    )
        if ckpt_critic_hidden != train_cfg.critic_hidden_dim and not _new_critic:
            raise SystemExit(
                f"critic_hidden_dim mismatch: checkpoint={ckpt_critic_hidden} "
                f"vs --critic-hidden-dim={train_cfg.critic_hidden_dim}"
            )
        if ckpt_critic_blocks != train_cfg.critic_num_blocks and not _new_critic:
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
        # v7 Q-surface semantics guards (V7_DESIGN.md WS1): q_fold_zero /
        # q_base_raw leave no trace in the state dict, but flipping either
        # reinterprets the whole learned Q surface (the adv rows absorbed
        # the old base), so a silent warm-start across a flip would train
        # against shifted targets. Old checkpoints lack the keys -> False.
        for _qk in ("q_fold_zero", "q_base_raw"):
            if _new_critic:
                break
            if bool(ckpt_cfg.get(_qk, False)) != bool(getattr(train_cfg, _qk)):
                raise SystemExit(
                    f"{_qk} mismatch: checkpoint="
                    f"{bool(ckpt_cfg.get(_qk, False))} vs current="
                    f"{bool(getattr(train_cfg, _qk))}. These flags change the "
                    "Q surface's meaning; warm-starting across a flip is "
                    "refused — start a fresh stem (or deliberately convert)."
                )
        # obs_rev guard — AFTER the structural guards above (a checkpoint of
        # the wrong variant/head/shape should say so, not talk about revs).
        # The layout (widths) is identical across revs, so
        # NOTHING downstream would notice weights trained on rev-1 feature
        # values being fed rev-2 ones. Absent stamp = 1 (every checkpoint
        # written before the 2026-09-20 fixes).
        ckpt_obs_rev = int(ckpt.get("obs_rev", 1))
        if ckpt_obs_rev != obs_rev:
            if not args.allow_obs_rev_change:
                raise SystemExit(
                    f"obs_rev mismatch: checkpoint={ckpt_obs_rev} vs this "
                    f"process={obs_rev} (plo5bp.encoding.OBS_SEMANTICS_REV). "
                    "The observation WIDTH is the same but the feature values "
                    "at dims 186/187, 800/802, 999-1006, 1024-1029, 1040-1041 "
                    "differ, so these weights would read inputs they were not "
                    f"trained on. Either set PLO5BP_OBS_REV={ckpt_obs_rev} to "
                    "continue this stem byte-compatibly, or pass "
                    "--allow-obs-rev-change to DELIBERATELY migrate it to rev "
                    f"{obs_rev} (a production behavior change — expect a "
                    "transient)."
                )
            print(
                f"!!! [obs-rev] PRODUCTION BEHAVIOR CHANGE: migrating this "
                f"stem from obs_rev {ckpt_obs_rev} to {obs_rev} "
                "(--allow-obs-rev-change) — dims 186/187, 800/802, 999-1006, "
                "1024-1029, 1040-1041 change meaning under the loaded weights; "
                "expect a transient. Older-rev siblings are NOT seeded into "
                "the opponent pool."
            )
        # A19: the distributional head's grid rides in the critic state dict
        # (`_value_centers` / `_value_edges` are persisted buffers), so the
        # CHECKPOINT's support always wins for V no matter what the flags
        # say — but `_raw_value_centers` (the q_base_raw dueling base) and
        # `hlgauss_sigma` are derived from the constructor's arguments. A
        # relaunch with a different --value-support / --value-hlgauss-sigma
        # would train Q and the HL-Gauss targets on a grid that disagrees
        # with V's. Rebuild the critic at the trained values instead.
        if train_cfg.value_bins > 0:
            _trained = {
                k: float(ckpt_cfg[k])
                for k in ("value_support", "value_hlgauss_sigma")
                if ckpt_cfg.get(k) is not None
                and float(ckpt_cfg[k]) != float(getattr(train_cfg, k))
            }
            if _trained:
                print(
                    f"[critic] checkpoint was trained with {_trained}; the "
                    "flags differ — keeping the CHECKPOINT's values (its "
                    "persisted value grid wins regardless)"
                )
                train_cfg = dataclasses.replace(train_cfg, **_trained)
                critic = CentralCritic(
                    obs_dim=obs_dim,
                    hidden_dim=train_cfg.critic_hidden_dim,
                    num_blocks=train_cfg.critic_num_blocks,
                    q_actions=critic_q_actions,
                    torso_layernorm=train_cfg.torso_layernorm,
                    value_bins=train_cfg.value_bins,
                    value_support=train_cfg.value_support,
                    hlgauss_sigma=train_cfg.value_hlgauss_sigma,
                    q_fold_zero=train_cfg.q_fold_zero,
                    q_base_raw=train_cfg.q_base_raw,
                    act=train_cfg.critic_act,
                    in_norm=train_cfg.critic_in_norm,
                    v_raw=train_cfg.critic_v_raw,
                )
                critic.to(train_cfg.device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        crit_sd = ckpt.get("critic")
        # Critic redesign (2026-09-26): --critic-fresh keeps the NEW critic's
        # random init; --critic-init loads a critic from another file (a
        # checkpoint's "critic", or a bare critic state dict). The actor's
        # Adam moments still restore (PPOTrainer.allow_actor_only_moments).
        if args.critic_init is not None:
            _ci = torch.load(args.critic_init, map_location="cpu", weights_only=False)
            crit_sd = _ci.get("critic", _ci) if isinstance(_ci, dict) else _ci
            print(f"[critic] initialized from {args.critic_init}")
        elif args.critic_fresh:
            crit_sd = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
            print("[critic] --critic-fresh: the checkpoint's critic is NOT loaded (new random init)")
        ck_adv = crit_sd.get("adv_head.weight")
        if (
            critic.q_actions > 0
            and ck_adv is not None
            and tuple(ck_adv.shape) != tuple(critic.adv_head.weight.shape)
        ):
            # Dueling-head width changed (e.g. --q-pooled 13->3): keep the
            # torso + value head, drop the old adv_head — it re-enters at
            # zero-init, so Q == V and the VRPO advantage is exactly GAE
            # until the (pooled) head retrains. Everything else is strict.
            crit_sd = {
                k: v for k, v in crit_sd.items()
                if not k.startswith("adv_head.")
            }
            missing, unexpected = critic.load_state_dict(crit_sd, strict=False)
            assert not unexpected, f"unexpected critic keys: {unexpected}"
            assert all(k.startswith("adv_head.") for k in missing), (
                f"non-adv_head keys missing from checkpoint critic: {missing}"
            )
            print(
                f"[q-pooled] checkpoint adv_head {tuple(ck_adv.shape)} != "
                f"built {tuple(critic.adv_head.weight.shape)} — dropped; "
                "fresh zero-init head (Q==V; VRPO==GAE until retrained)"
            )
        else:
            critic.load_state_dict(crit_sd)
        prior_game = ckpt.get("game_config")
        print(f"warm-started from {args.load_checkpoint} (prior game_config: {prior_game})")
        restored_update = ckpt.get("update_counter")
        restored_pool_updates = ckpt.get("pool_member_updates")
        restored_control_applied = ckpt.get("anneal_control_applied")
        restored_numbering = str(ckpt.get("numbering", NUMBERING_INDEX))

    # Per-tier entropy coefs (--mix-configs): every tier starts at
    # --entropy-coef; the control file's {"tier_ent": {...}} / {"entropy_coef":
    # X} retune them live, and each sub-rollout's rows carry their tier's coef
    # (Batch.ent_coef_rows). Stamped into checkpoints (`anneal_tier_ent`) as a
    # record. (2026-09-28, ML-030: the block rotation and the F/T/R-driven
    # auto-anneal that also read this are retired.)
    tier_ent: dict[str, float] = (
        {t: args.entropy_coef for t in mix_tiers} if args.mix_configs else {}
    )

    trainer = PPOTrainer(model, train_cfg, critic=critic)
    # The fully resolved training config, once (it is also stamped into every
    # checkpoint's `config` and the launch record): after the --v6 preset,
    # the sentinel defaults and any checkpoint-support override (ML-006).
    print(
        "[config] resolved: "
        + ", ".join(f"{k}={v}" for k, v in sorted(train_cfg.__dict__.items()))
    )
    if train_cfg.clip_prob_dependent:
        print(
            f"[clip] prob-dependent U: ext={args.clip_room_ext} "
            f"mid={args.clip_room_mid} floor={args.clip_prob_floor} "
            "(live-tunable via anneal_control clip_room_mid/clip_room_ext)"
        )
    # KL-anchor EMA magnet persistence: restore the reference from the
    # checkpoint so the pull-toward-history survives relaunches (absent
    # the key it re-initializes to the loaded weights and ramps in).
    if args.load_checkpoint is not None and trainer._ref is not None:
        ema_sd = ckpt.get("model_ema")
        if ema_sd:
            trainer.load_ref_state_dict(ema_sd)
            print("[kl-anchor] restored EMA reference from checkpoint")
    # A3 + A14: warm Adam + the original l2-init references from the rolling
    # sidecar (PRODUCTION BEHAVIOR CHANGE — see _restore_optimizer_sidecar).
    if args.load_checkpoint is not None:
        if args.optimizer_sidecar:
            # A NEW critic (--critic-fresh / --critic-init): the actor's Adam
            # moments + l2-init references still restore, the critic's start cold.
            _restore_optimizer_sidecar(
                trainer, args.load_checkpoint, restored_update,
                allow_actor_only=bool(args.critic_fresh or args.critic_init is not None),
            )
        else:
            print(
                "[optim] --no-optimizer-sidecar: LEGACY resume — Adam starts "
                "COLD, l2-init re-anchors at the loaded weights"
            )
    # A RESUMED run draws its own stream, derived from (seed, resume update).
    # Re-seeding with the bare --seed made every relaunch replay the fresh
    # run's first updates exactly -- the same table configs AND the same card
    # deals, update after update (guardian restarts are routine). Fresh runs
    # keep default_rng(seed) / the seed set above, byte-identical.
    if restored_update is not None:
        _resume_ss = np.random.SeedSequence([int(args.seed), int(restored_update)])
        run_seed = int(_resume_ss.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))
        torch.manual_seed(run_seed)
        rng = np.random.default_rng(_resume_ss)
        print(
            f"[seed] resumed at update {restored_update}: stream seeded from "
            f"(seed {args.seed}, update {restored_update})"
        )
    else:
        run_seed = int(args.seed)
        rng = np.random.default_rng(args.seed)
    pool = OpponentPool(capacity=train_cfg.opponent_pool_size, seed=run_seed)

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
                expected_obs_rev=obs_rev,
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

    # Live control (entropy / lr / KL-guard / clip / tier-coef retunes)
    # without pausing training -- see control._apply_anneal_control. The
    # control file is per run: <run-dir>/<stem>.control.json by default (2026-09-28, ML-060;
    # it used to default to the pod-wide runs/anneal_control.json, which every
    # trainer without the env override read). PLO5BP_ANNEAL_CONTROL still wins
    # (the vSix6 guardian sets runs/vSix6.control.json -- the same file).
    anneal_control_file = Path(
        os.environ.get("PLO5BP_ANNEAL_CONTROL", "").strip()
        or (run_dir / f"{stem}.control.json")
    )
    print(f"[control] live control file: {anneal_control_file}")
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
    #
    # ...EXCEPT the one case where ignoring it silently loses tuning (review
    # 2026-09-20 A4): a crash + guardian relaunch of a run that HAD applied
    # this very content. Nothing applied used to be persisted, so the relaunch
    # fell back to the launch flags (lr, target_kl, kl_hard, clip rooms,
    # q_fold_sup_coef, sizing_entropy_scale, entropy coefs) while the file
    # still showed the tuned values — with no log line. Every checkpoint now
    # stamps the last APPLIED content (`anneal_control_applied`):
    #   file == the loaded checkpoint's stamp -> RE-APPLY it (logged);
    #   file exists but differs / no stamp    -> ignored as before, LOUDLY.
    last_anneal_control: str | None = None
    # What THIS lineage has actually applied — the checkpoint stamp. Distinct
    # from `last_anneal_control` (the change-detection baseline), which also
    # holds an IGNORED stale file's text: stamping that would get it "re"-
    # applied on the next relaunch although it never took effect.
    anneal_control_applied: str | None = None
    if anneal_control_file.exists():
        _pre_raw = _read_control_text(anneal_control_file)
        if _pre_raw is not None and _pre_raw == restored_control_applied:
            print(
                f"[anneal-control] {anneal_control_file} matches the loaded "
                "checkpoint's applied stamp — RE-APPLYING it (live tuning "
                f"survives the relaunch): {_pre_raw.strip()[:300]}"
            )
            (
                last_anneal_control,
                live_lr,
                live_entropy_coef,
                live_entropy_coef_deep,
            ) = _apply_anneal_control(
                _pre_raw, None, tier_ent,
                live_lr, live_entropy_coef, live_entropy_coef_deep,
                trainer=trainer, broadcast_entropy=args.mix_configs,
            )
            if last_anneal_control is None:
                last_anneal_control = _pre_raw  # rejected now: don't retry forever
            else:
                anneal_control_applied = _pre_raw
        else:
            last_anneal_control = _pre_raw
            print(
                f"[anneal-control] !!! PRE-EXISTING {anneal_control_file} IGNORED "
                "— this run starts on its LAUNCH FLAGS. "
                + (
                    "Its content differs from the loaded checkpoint's applied "
                    "stamp"
                    if restored_control_applied is not None
                    else "No applied-control stamp to match it against (cold "
                    "start, or a checkpoint older than the stamp)"
                )
                + ". To apply it, re-save the file with ANY content change "
                "(e.g. add a space) after startup. Content: "
                + ("<unreadable>" if _pre_raw is None else _pre_raw.strip()[:300])
            )

    collector = collect_rollout_batched if args.batched else collect_rollout
    time_budget = float(args.train_seconds)
    use_time_budget = time_budget > 0.0
    t_start = time.time()
    last_snapshot_sec = t_start
    last_ckpt_sec = t_start
    # Loop index of the last update a numbered checkpoint was written for
    # (None = none yet) — lets the final save tell the sidecar when it holds
    # the very same optimizer state as that numbered file.
    last_mid_saved_update: int | None = None

    # Computed once per launch; rides in every checkpoint (ML-019).
    provenance = run_provenance()

    def _checkpoint_payload(update_counter: int, game_config: dict) -> dict:
        """The checkpoint dict -- ONE builder for the numbered saves and the
        final save (ML-016). Legacy keys first, unchanged; the schema-2 keys
        (`schema`, `arch`, `provenance`) are additions older loaders ignore."""
        return build_checkpoint(
            model=model.state_dict(),
            critic=critic.state_dict(),
            head_version=model.head_version,
            config=train_cfg.__dict__,
            game_config=game_config,
            gate_count=GATE_ACTIONS,
            variant=args.variant,
            anchor_count=model._anchor_count,
            update_counter=int(update_counter),
            # Metadata only (update indices, not weights): lets a warm-start
            # reconstruct the exact pool membership from the sibling files
            # still on disk.
            pool_member_updates=pool.fifo_tags,
            anneal_tier_ent=tier_ent,
            # Truthful regimen stamp under --mix-configs (game_config is just
            # the first sub-rollout's draw — B9).
            mix_configs=bool(args.mix_configs),
            mix_tiers=list(mix_tiers) if args.mix_configs else None,
            configs_per_tier=int(args.configs_per_tier) if args.mix_configs else None,
            # KL-anchor EMA reference (None when the magnet is off); restored
            # on warm-start. Doubles as the smoother serving actor.
            model_ema=trainer.ref_state_dict(),
            # A4: raw text of the last APPLIED control file (None = none); a
            # relaunch re-applies the file only when it still matches this.
            anneal_control_applied=anneal_control_applied,
            drain_inflight=drain_inflight,
            # Observation-semantics revision these weights trained on (the
            # warm-start guard + pool seeding key off it).
            obs_rev=obs_rev,
            schema_fields={
                "arch": {"actor": actor_arch(model), "critic": critic_arch(critic)},
                "provenance": provenance,
                "numbering": NUMBERING_COUNT if count_numbering else NUMBERING_INDEX,
            },
        )

    def _save_checkpoint(payload: dict, path: Path) -> None:
        # Never write NaN/Inf weights (ML-001 B): the guardian would resume
        # every relaunch from the poisoned file.
        assert_finite_for_save(payload, str(path), ("model", "critic", "model_ema"))
        _atomic_torch_save(payload, path)

    def _save_mid(update_idx: int) -> None:
        nonlocal last_mid_saved_update
        # `update_idx` is the LOCAL loop counter; numbered checkpoints are named
        # and stamped on the GLOBAL update axis (`_stamp`: base_update + local,
        # + 1 under --number-by-count). Without this, a warm relaunch (which
        # resets the loop counter to 0) would rewrite a prior segment's
        # <stem>_5.pt/_10.pt... over the originals AND stamp update_counter=5,
        # poisoning the next warm-start's pool seeding (ws_target reads that
        # counter). See base_update below.
        global_idx = _stamp(update_idx)
        game_cfg_snap = sampled_game_cfg.__dict__
        mid_path = args.checkpoint.with_name(f"{args.checkpoint.stem}_{global_idx}.pt")
        if (
            args.load_checkpoint is not None
            and mid_path.resolve() == Path(args.load_checkpoint).resolve()
        ):
            # C3 (narrowed after adversarial review): never overwrite THE
            # checkpoint this run warm-started from. Under the legacy index
            # numbering a relaunch's first update carries the loaded file's own
            # number (the wall-clock cadence can fire there) -- saving would
            # rewrite the exact restore point just loaded with weights carrying
            # one extra PPO update, destroying the clean-recovery file the
            # collapse playbook depends on. Skip; the next cadence tick writes a
            # fresh number. Guarding ONLY the loaded file (not blanket
            # write-once) preserves last-write-wins for every legitimate
            # collision: orchestrations that re-run a phase from a fixed source
            # must refresh their outputs.
            print(
                f"[ckpt] skip: {mid_path.name} is this run's warm-start source "
                "(never overwritten)"
            )
            return
        # NOTE (review 2026-09-20 A20): under the legacy index numbering a
        # MID save stamps the 0-based index of the update that just finished,
        # the FINAL save stamps the COUNT of updates done — so the same weights
        # read N here and N+1 there. Consumers (pool seeding, the optimizer
        # sidecar match) treat both as "this file's update"; changing it would
        # shift an existing stem's numbering, so the fix is opt-in per NEW
        # stem: --number-by-count stamps the count everywhere (ML-063).
        _save_checkpoint(_checkpoint_payload(global_idx, game_cfg_snap), mid_path)
        if args.optimizer_sidecar:
            _save_optimizer_sidecar(trainer, args.checkpoint, global_idx)
        last_mid_saved_update = update_idx

    # A relaunch restarts the LOOP counter at 0 (the LR-warmup ramp resumes
    # with its gentle-restart semantics); `base_update` carries the true
    # cumulative update index onto which checkpoint filenames / update_counter /
    # pool snapshot tags are stamped, so a same-stem relaunch never clobbers
    # prior <stem>_<N>.pt files and the counter a later warm-start reads stays
    # truthful (fixes silent checkpoint overwrite + pool-seeding poison). (The
    # retired --anneal-entropy continued the loop counter instead, so its
    # block cycle survived a relaunch.)
    _restored = int(restored_update) if restored_update is not None else 0
    update = 0
    base_update = _restored
    count_numbering = resolve_numbering(
        args.number_by_count, args.load_checkpoint, args.checkpoint, restored_numbering
    )
    if count_numbering:
        print("[ckpt] numbering by update COUNT (file N = the weights after N updates)")

    def _stamp(local_idx: int) -> int:
        """The update number stamped for the update at loop index
        `local_idx`: its 0-based global INDEX (the legacy numbering), or the
        COUNT of updates done once it finished (--number-by-count, ML-063)."""
        return base_update + local_idx + (1 if count_numbering else 0)
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

    # Optional 1 Hz CPU/GPU/RAM/VRAM sampler: --profile-one-update runs and
    # --resource-sampler (2026-09-28, ML-018: it used to start with every
    # PLO5BP_STEP_TIMERS=1 run -- every guardian -- spawning nvidia-smi each
    # second for weeks and keeping every sample in RAM; step timers and the
    # sampler are separate needs). Phase wall times are always logged.
    resource_sampler: _ResourceSampler | None = None
    if args.profile_one_update or args.resource_sampler:
        resource_sampler = _ResourceSampler(
            out_path=run_dir / (
                f"{stem}.resources_u{args.profile_at_update}.jsonl"
                if args.profile_one_update else f"{stem}.resources.jsonl"
            ),
            interval_s=1.0,
        )
        resource_sampler.start()

    # Per-update metrics, the heartbeat and this launch's provenance
    # (ML-005 / ML-024 / ML-019).
    metrics = MetricsWriter(run_dir / f"{stem}.metrics.jsonl")
    heartbeat = Heartbeat(run_dir / f"{stem}.heartbeat")
    print(
        f"[run-files] {metrics.path} (per-update metrics), {heartbeat.path}, "
        f"{run_dir / (stem + '.launches.jsonl')}"
    )
    append_launch_record(
        run_dir / f"{stem}.launches.jsonl",
        {
            "provenance": provenance,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "config": dict(train_cfg.__dict__),
            "resumed_from": str(args.load_checkpoint) if args.load_checkpoint else None,
            "resumed_update": restored_update,
            "obs_rev": obs_rev,
        },
    )

    consecutive_rollbacks = 0
    _update0 = update  # this run's first loop update (--actor-freeze-updates)
    heartbeat.beat(update=base_update + update, consecutive_rollbacks=0, state="starting")
    while True:
        if stop_requested["flag"]:
            break

        # Live thread-count adjustment: edit runs/threads.txt to change
        # torch's intra-op threadpool without restarting the run. Malformed
        # reads are ignored. Applies on CUDA too — the whole rollout +
        # _concat_batches copy is CPU-side even when the learner is on GPU
        # (8becc82 de-gated it; the cap matters MOST on CUDA).
        if threads_file.exists():
            _threads_raw = _read_control_text(threads_file)
            try:
                desired = int((_threads_raw or "").strip())
            except ValueError:
                desired = 0
                if _threads_raw is not None:
                    _warn_once(
                        f"[threads] {threads_file} is not an integer — "
                        f"IGNORED: {_threads_raw.strip()[:80]!r}"
                    )
            if desired > 0 and desired != current_threads:
                torch.set_num_threads(desired)
                current_threads = desired
                print(f"[threads] set torch threads -> {desired}")

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
            control_raw = _read_control_text(anneal_control_file)
            _prev_control = last_anneal_control
            (
                last_anneal_control,
                live_lr,
                live_entropy_coef,
                live_entropy_coef_deep,
            ) = _apply_anneal_control(
                control_raw, last_anneal_control, tier_ent,
                live_lr, live_entropy_coef, live_entropy_coef_deep,
                trainer=trainer,
                # Mix-configs consumes tier_ent (per-row coefs), not the flat
                # coef: `entropy_coef` broadcasts to the tiers (inside the
                # helper since 2026-09-20 — see its comment).
                broadcast_entropy=args.mix_configs,
            )
            if last_anneal_control != _prev_control:
                # The helper only advances this on a fully APPLIED edit.
                anneal_control_applied = last_anneal_control

        # Common random numbers (--crn-streams, ML-004): this update's table
        # configs, its PPO shuffles and its collection each draw from their
        # own stream keyed by (seed, update); otherwise the one shared `rng`.
        if train_cfg.crn_streams:
            _u_key = _stamp(update)
            cfg_rng = np.random.default_rng([int(args.seed), _u_key, 0])
            ppo_rng = np.random.default_rng([int(args.seed), _u_key, 1])
            crn_key = (int(args.seed), _u_key)
        else:
            cfg_rng = ppo_rng = rng
            crn_key = None
        if args.mix_configs:
            # vThree: every update mixes `configs_per_tier` (seats,stacks) draws
            # from each mix tier (no block-rotation), so the gradient averages
            # over all N configs — no consecutive-tier saturation.
            mix_cfgs = [
                _sample_game_config(
                    seats_choices, stack_lo, stack_hi, args.bb, args.ante, cfg_rng,
                    stack_dist=tier, seats_dist=args.seats_dist,
                    variant=args.variant, sb=args.sb,
                )[0]
                for tier in mix_tiers
                for _ in range(args.configs_per_tier)
            ]
            mix_cfg_tiers = [
                tier
                for tier in mix_tiers
                for _ in range(args.configs_per_tier)
            ]
            sampled_game_cfg, sampled_eff_dist = mix_cfgs[0], "mix"
        else:
            sampled_game_cfg, sampled_eff_dist = _sample_game_config(
                seats_choices, stack_lo, stack_hi, args.bb, args.ante, cfg_rng,
                stack_dist=args.stack_dist, seats_dist=args.seats_dist,
                variant=args.variant, sb=args.sb,
            )
        # Skip torch.profiler when using lightweight step timers — key_averages
        # on a full 9M-step update allocates 100GB+ and never finishes.
        _use_torch_prof = (
            args.profile_one_update
            and os.environ.get("PLO5BP_STEP_TIMERS", "").strip().lower()
            not in ("1", "true", "yes", "on")
        )
        _profile_this = _use_torch_prof and update == args.profile_at_update
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

        if resource_sampler is not None:
            resource_sampler.set_phase("rollout", update=update)
        _t_rollout0 = time.perf_counter()
        if args.mix_configs:
            # Per-tier coefs ride the batch as per-row ent_coef_rows
            # (V5_DESIGN.md B5): each tier's transitions are paid that
            # tier's own rate, so `{"tier_ent": {"deep": X}}` control
            # edits now genuinely apply under mixing. The scalar below is
            # only the ppo fallback + the log-line display value.
            batch = collect_rollout_multiconfig(
                model, pool, mix_cfgs, train_cfg, rng, critic=critic,
                config_tiers=mix_cfg_tiers, tier_ent=tier_ent,
                drain_inflight=drain_inflight, crn_key=crn_key,
            )
            update_entropy_coef = tier_ent.get(mix_tiers[0], args.entropy_coef)
        else:
            batch = collector(
                model, pool, sampled_game_cfg, train_cfg, rng, critic=critic,
                drain_inflight=drain_inflight,
                **({"crn_key": (*crn_key, 0)} if crn_key is not None else {}),
            )
            update_entropy_coef = (
                live_entropy_coef_deep
                if sampled_eff_dist == "deep"
                else live_entropy_coef
            )
        _t_rollout1 = time.perf_counter()
        # Cold-start LR warmup: small early steps keep per-minibatch KL
        # inside the guard's trust region, so all minibatches apply and
        # the critic actually trains (a tripped update aborts the critic
        # too — huge advantages then keep the next step violent). Counter
        # semantics (see base_update above): with --anneal-entropy the
        # loop counter continues from the checkpoint, so a resumed run
        # past the window is at full LR immediately; anneal-off relaunches
        # reset the loop counter and DELIBERATELY re-run the warmup ramp
        # (gentle-restart semantics — every vFour collapse recovery relied
        # on it). Checkpoint names/counters stay on the global axis either
        # way via base_update.
        lr_scale = _lr_warmup_scale(update, args.lr_warmup_updates)
        trainer.set_lr(live_lr * lr_scale)
        if resource_sampler is not None:
            resource_sampler.set_phase("optimize", update=update)
        _t_opt0 = time.perf_counter()
        _dump = os.environ.get("PLO5BP_DUMP_BATCH", "").strip()
        if _dump:  # diagnostics: save a sample of this rollout, skip the update
            _dump_batch_diagnostics(batch, _dump)
            print(f"[diag] rollout sample written to {_dump}; exiting before the update")
            return
        actor_frozen = (update - _update0) < int(args.actor_freeze_updates)
        if actor_frozen:
            print(f"[critic-warmup] update {update}: actor FROZEN, critic-only passes "
                  f"({train_cfg.ppo_epochs + train_cfg.critic_extra_epochs} epochs)")
        stats = trainer.update(
            batch, ppo_rng, entropy_coef=update_entropy_coef, actor_frozen=actor_frozen
        )
        _t_opt1 = time.perf_counter()
        if resource_sampler is not None:
            resource_sampler.set_phase("post", update=update)
        _rollout_s = _t_rollout1 - _t_rollout0
        _opt_s = _t_opt1 - _t_opt0
        _total_s = _t_opt1 - _t_rollout0
        print(
            f"        [phase] update={update}  "
            f"rollout={_rollout_s:.1f}s  optimize={_opt_s:.1f}s  "
            f"total={_total_s:.1f}s  "
            f"rollout%={100.0 * _rollout_s / max(1e-9, _total_s):.1f}  "
            f"optimize%={100.0 * _opt_s / max(1e-9, _total_s):.1f}"
        )

        _check_stats_finite(stats)

        # Livelock alarm (review 2026-09-20 A3). A hard rollback restores the
        # pre-update params AND Adam state, so if the FIRST minibatch of every
        # update clears kl_hard (cold Adam after a relaunch: the first step is
        # ~lr*sign(g)) or is non-finite (A8), every update is a no-op and the
        # run burns GPU forever while the guardians — which watch only PID +
        # entropy — report "ok". Entropy does not move in that state either.
        if stats.rolled_back or not math.isfinite(stats.kl_stop):
            consecutive_rollbacks += 1
            if consecutive_rollbacks >= _ROLLBACK_ALARM_AFTER and (
                consecutive_rollbacks % _ROLLBACK_ALARM_AFTER == 0
            ):
                print(
                    f"!!! [ALARM] {consecutive_rollbacks} CONSECUTIVE updates "
                    "rolled back / refused (last: "
                    f"mb{stats.kl_stopped_at}, kl={stats.kl_stop:+.3g}) — NO "
                    "parameter has moved since; this is a LIVELOCK, not "
                    "training. Fix live via runs/anneal_control.json (lower "
                    '{"lr": ...} or raise {"kl_hard": ...}), or relaunch '
                    "with --lr-warmup-updates > 0."
                )
        else:
            consecutive_rollbacks = 0
        if 0 < int(args.max_consecutive_rollbacks) <= consecutive_rollbacks:
            heartbeat.beat(
                update=_stamp(update), consecutive_rollbacks=consecutive_rollbacks,
                state="livelock",
            )
            print(
                f"!!! [LIVELOCK] {consecutive_rollbacks} consecutive updates rolled "
                "back / refused (--max-consecutive-rollbacks) — exiting with code "
                f"{LIVELOCK_EXIT_CODE}; nothing moved since the last checkpoint.",
                flush=True,
            )
            raise SystemExit(LIVELOCK_EXIT_CODE)

        now = time.time()

        # Snapshot on update count, then also on wall-clock if configured.
        # Tag on the GLOBAL axis so pool_member_updates stays consistent with
        # the numbered checkpoint filenames a warm-start reads. Legacy
        # numbering snapshots on the LOCAL grid (local update 0 included, its
        # tag = the loaded file's own number); --number-by-count on the
        # global count grid.
        snap_due = (
            _stamp(update) % train_cfg.snapshot_every == 0
            if count_numbering
            else update % train_cfg.snapshot_every == 0
        )
        if snap_due:
            pool.snapshot(model, tag=_stamp(update))
            if train_cfg.pool_anchor_ages:
                # Anchors (ML-033): re-pointed on the snapshot grid, after
                # the update like every pool change (frozen within one).
                anchors = refresh_pool_anchors(
                    pool, args.checkpoint, _stamp(update),
                    list(train_cfg.pool_anchor_ages), train_cfg.snapshot_every,
                    args.variant, model.head_version, model.state_dict(),
                    expected_obs_rev=obs_rev,
                )
                if anchors:
                    print(f"[pool] anchors: updates {anchors}")
        if args.snapshot_every_sec > 0 and now - last_snapshot_sec >= args.snapshot_every_sec:
            pool.snapshot(model, tag=_stamp(update))
            last_snapshot_sec = now

        if _prof is not None:
            _prof.__exit__(None, None, None)
            # Tables only by default: chrome traces at 9M-step scale grow to
            # multi-GB and can hang for 30+ min writing profile_updateN.json.tmp
            # before key_averages ever print. Set PLO5BP_CHROME_TRACE=1 to
            # re-enable (not recommended on full rollouts).
            _ka = _prof.key_averages()
            _tag = "(post-compile)" if update > 0 else "(incl. one-time compile)"
            print(f"\n===== profiled update {update} {_tag} — SELF CUDA =====")
            print(_ka.table(sort_by="self_cuda_time_total", row_limit=50))
            print(f"\n===== profiled update {update} {_tag} — SELF CPU =====")
            print(_ka.table(sort_by="self_cpu_time_total", row_limit=50))
            # Rank record_function buckets (step1a/refresh, step3/forward, ...)
            print(f"\n===== profiled update {update} {_tag} — CUDA total (incl. children) =====")
            print(_ka.table(sort_by="cuda_time_total", row_limit=40))
            print(f"\n===== profiled update {update} {_tag} — CPU total (incl. children) =====")
            print(_ka.table(sort_by="cpu_time_total", row_limit=40))
            if os.environ.get("PLO5BP_CHROME_TRACE", "").strip() in ("1", "true", "yes"):
                trace_path = Path(f"runs/profile_update{update}.json")
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[profile] exporting chrome trace (PLO5BP_CHROME_TRACE=1) -> {trace_path}")
                _prof.export_chrome_trace(str(trace_path))
                print(f"[profile] chrome trace -> {trace_path}")
            else:
                print("[profile] chrome trace SKIPPED (set PLO5BP_CHROME_TRACE=1 to enable)")
            if resource_sampler is not None:
                resource_sampler.stop()
                resource_sampler.summarize()
                resource_sampler = None
            stop_requested["flag"] = True

        # Mid-run checkpoints: update-count + wall-clock variants. Legacy
        # numbering never saves a launch's first update (local index 0 --
        # after a relaunch it would be the loaded file's own number);
        # --number-by-count saves on the global count grid, the first update
        # included.
        if args.checkpoint_every > 0 and (
            _stamp(update) % args.checkpoint_every == 0
            if count_numbering
            else update > 0 and update % args.checkpoint_every == 0
        ):
            _save_mid(update)
        if args.checkpoint_every_sec > 0 and now - last_ckpt_sec >= args.checkpoint_every_sec:
            _save_mid(update)
            last_ckpt_sec = now

        elapsed = now - t_start
        stacks_bb = [round(s / args.bb, 1) for s in sampled_game_cfg.resolved_stacks]
        steps_by_street = batch.aggr_steps_total_by_street
        bonus_by_street = batch.aggr_bonus_steps_by_street
        # Share of learner steps per street that were WINNING aggression (a
        # raise with > 50% of the final pot; at exactly 50% also a call) --
        # "aggr%", printed as "bonus%" before 2026-09-28 although it counts
        # steps, not bonuses (ML-061); the bonuses themselves are retired.
        aggr_pct = [
            100.0 * bonus_by_street[s_] / max(1, steps_by_street[s_]) for s_ in range(3)
        ]
        tier_aggr = {
            _t: [100.0 * _bonus[s_] / max(1, _steps[s_]) for s_ in range(3)]
            for _t, (_bonus, _steps) in (getattr(batch, "tier_ftr", None) or {}).items()
        }
        n_rows = int(batch.obs.shape[0])
        # The critic's value health on this rollout (ML-003): a strided sample,
        # no random draw -- reporting only.
        vh = value_health(batch)
        # P11 measurement: peak GPU memory of this update. `del batch` (below)
        # frees the prior update's batch before the next collection allocates
        # its own; this shows where the true per-update peak lands so the
        # rollout can be grown to fit. Bit-exact — reporting only.
        vram = None
        if args.device == "cuda":
            vram = {
                "peak_alloc_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
            }
            torch.cuda.reset_peak_memory_stats()
        metrics.write({
            "update": _stamp(update),
            "loop_update": update,
            "time": now,
            "elapsed_s": elapsed,
            "rollout_s": _rollout_s,
            "optimize_s": _opt_s,
            "update_s": _total_s,
            "rows": n_rows,
            "rows_per_s": n_rows / max(1e-9, _rollout_s),
            "lr": live_lr * lr_scale,
            "lr_scale": lr_scale,
            "entropy_coef": update_entropy_coef,
            "tier_ent": dict(tier_ent),
            "sizing_entropy_scale": trainer.sizing_entropy_scale,
            "ppo": dataclasses.asdict(stats),
            "value_health": vh,
            "aggr_pct": aggr_pct,
            "tier_aggr_pct": tier_aggr,
            "pool": {"size": len(pool), "tags": list(pool.tags)},
            "consecutive_rollbacks": consecutive_rollbacks,
            "configs": len(mix_cfgs) if args.mix_configs else 1,
            "seats": sampled_game_cfg.num_seats,
            "stacks_bb": stacks_bb,
            "vram": vram,
            "threads": current_threads,
        })
        heartbeat.beat(
            update=_stamp(update), consecutive_rollbacks=consecutive_rollbacks,
            rows=n_rows, update_s=_total_s, state="ok",
        )

        if update % args.log_every == 0:
            print(
                f"[{elapsed:7.1f}s] update {update:5d}  "
                f"pi={stats.policy_loss:+.4f}  "
                f"v={stats.value_loss:.4f}  "
                f"vd={stats.display_loss:.4f}  "
                # H = the policy's own entropy (since 2026-09-28, ML-040;
                # before, the bonus-weighted one -- see Hbonus).
                f"H={stats.entropy:.3f}  "
                f"Hg/Ha/Hb={stats.gate_entropy:.2f}/{stats.anchor_entropy:.2f}/"
                f"{stats.beta_entropy:.2f}  "
                f"kl={stats.approx_kl:+.4f}  "
                f"klG/klA/klB={stats.gate_kl:+.3f}/{stats.anchor_kl:+.3f}/"
                f"{stats.beta_kl:+.3f}  "
                + (
                    f"Hbonus={stats.entropy_bonus:.3f}  "
                    if trainer.sizing_entropy_scale != 1.0 else ""
                )
                + (f"klanc={stats.kl_anchor:.4f}  " if args.kl_anchor_coef > 0 else "")
                + (f"q={stats.q_loss:.4f}  " if args.q_aux_coef > 0 else "")
                # Fold-column canary (audit 2026-07-11): mean Q[FOLD] over
                # fold-LEGAL rows. Ground truth is exactly 0 — sustained
                # drift = the Q surface acquiring a systematic offset.
                + (f"qF={stats.q_fold_err:+.2f}  " if args.q_aux_coef > 0 else "")
                # Terminal-boundary canary (V7_DESIGN.md WS1.3): mean
                # (return − Q) over non-fold TERMINAL rows — the exact δ at
                # hand boundaries, where bias can't cancel against a next
                # state. Persistent positive = hand-ending actions collect
                # fake advantage (the July fold-subsidy mechanism).
                + (f"qT={stats.q_term_err:+.2f}  " if args.q_aux_coef > 0 else "")
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
                f"aggr%(F/T/R)={aggr_pct[0]:4.1f}/{aggr_pct[1]:4.1f}/"
                f"{aggr_pct[2]:4.1f}  "
                f"pool={len(pool)}  "
                f"seats={sampled_game_cfg.num_seats}  "
                f"stacks_bb={stacks_bb}  "
                f"ent={update_entropy_coef:.3f}"
                + (f"  lr×{lr_scale:.2f}" if lr_scale < 1.0 else "")
            )
            print(_health_line(stats, vh))
            # Per-tier F/T/R under mix-configs (V5_DESIGN.md B5): the
            # pooled line above can't drive the per-tier stop-loss; this
            # one can. Same semantics as aggr%(F/T/R), bucketed by tier.
            if tier_aggr:
                print("        [ftr-tier] " + "  ".join(
                    f"{_t}={f_:4.1f}/{t_:4.1f}/{r_:4.1f}"
                    f"(ent {tier_ent.get(_t, update_entropy_coef):.3f})"
                    for _t, (f_, t_, r_) in tier_aggr.items()
                ))
            if vram is not None:
                print(
                    f"        [vram] peak alloc={vram['peak_alloc_gib']:.1f} GiB  "
                    f"reserved={vram['peak_reserved_gib']:.1f} GiB  "
                    f"(rollout={n_rows:,} rows)"
                )

        # P11: drop this update's ~45GB batch (all fields on CUDA) now that its
        # last reads (the log/anneal blocks above) are done. Python otherwise
        # keeps `batch` bound until the next iteration's `batch = collect_...`
        # RHS finishes — i.e. through the whole next collection — so the old and
        # new batches sit co-resident (the 2x-batch VRAM peak = the measured
        # 78GiB@10M ceiling). Freeing here lets the caching allocator reuse the
        # blocks for the next collection. Bit-exact: nothing reads `batch` after
        # this point (the final save uses model/critic only).
        del batch
        if gpu_lock is not None:
            gpu_lock.release()  # frees this update's GPU memory first

        update += 1

    if resource_sampler is not None:
        resource_sampler.stop()
        resource_sampler.summarize()
        resource_sampler = None

    # The final save stamps the COUNT of updates done — one more than the
    # index a mid save of the same weights stamps (see the note in _save_mid).
    _save_checkpoint(
        _checkpoint_payload(base_update + update, sampled_game_cfg.__dict__),
        args.checkpoint,
    )
    if args.optimizer_sidecar:
        # When no update ran since the last numbered save, that file holds
        # these exact weights under the mid-save stamp (one lower): the
        # sidecar is valid for it too — which is what a guardian resumes from.
        _mid = (
            None if last_mid_saved_update is None
            else _stamp(last_mid_saved_update)
        )
        _current = _mid is not None and last_mid_saved_update == update - 1
        _save_optimizer_sidecar(
            trainer, args.checkpoint, base_update + update,
            same_state_counters=(_mid,) if _current else (),
            # Updates ran since that numbered save: keep ITS moments too.
            keep_counter=None if (_current or _mid is None) else _mid,
        )
    elapsed = time.time() - t_start
    print(
        f"Saved checkpoint to {args.checkpoint} after {update} updates this run "
        f"(global u{base_update + update}, {elapsed:.1f}s wall-clock)"
    )
    if train_cfg.device == "cuda" and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"[cuda] peak memory allocated: {peak_mb:.0f} MB")
