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
import dataclasses
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

class _ResourceSampler:
    """Background 1 Hz samples of CPU / RAM / GPU util / VRAM during a run.

    Writes JSONL to runs/profile_resources.jsonl. Phase labels via set_phase
    so each sample is attributable to rollout vs optimize. Uses psutil when
    available; falls back to /proc + nvidia-smi.
    """

    def __init__(
        self,
        out_path: str | Path = "runs/profile_resources.jsonl",
        interval_s: float = 1.0,
    ) -> None:
        self.out_path = Path(out_path)
        self.interval_s = float(interval_s)
        self._phase = "init"
        self._update = -1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._psutil = None
        self._proc = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)
            psutil.cpu_percent(None)
        except Exception:
            self._psutil = None
            self._proc = None
        self._n_logical = os.cpu_count() or 1
        self._cgroup_quota_cpus = self._read_cgroup_quota_cpus()
        self.samples: list[dict] = []

    @staticmethod
    def _read_cgroup_quota_cpus() -> float | None:
        for path in ("/sys/fs/cgroup/cpu.max",):
            try:
                raw = Path(path).read_text().strip().split()
                if len(raw) >= 2 and raw[0] != "max":
                    quota, period = float(raw[0]), float(raw[1])
                    if period > 0:
                        return quota / period
            except (OSError, ValueError):
                pass
        try:
            q = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            p = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0 and p > 0:
                return q / p
        except (OSError, ValueError):
            pass
        return None

    def set_phase(self, phase: str, update: int | None = None) -> None:
        self._phase = phase
        if update is not None:
            self._update = int(update)

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text("")
        self._thread = threading.Thread(
            target=self._loop, name="resource-sampler", daemon=True
        )
        self._thread.start()
        print(
            f"[profile-resources] sampling every {self.interval_s:.1f}s -> "
            f"{self.out_path}  "
            f"(cgroup_quota_cpus={self._cgroup_quota_cpus!s}, "
            f"logical_cpus={self._n_logical})"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2.0)
            self._thread = None

    def _sample_gpu(self) -> tuple[float | None, float | None, float | None]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2.0,
            ).strip().splitlines()
            if not out:
                return None, None, None
            parts = [p.strip() for p in out[0].split(",")]
            return float(parts[0]), float(parts[1]), float(parts[2])
        except Exception:
            return None, None, None

    def _sample_cpu_ram(self) -> dict:
        row: dict = {}
        if self._psutil is not None and self._proc is not None:
            proc_cpu_1core = float(self._proc.cpu_percent(None))
            host_cpu = float(self._psutil.cpu_percent(None))
            row["proc_cpu_pct_1core"] = round(proc_cpu_1core, 2)
            row["proc_cpu_pct_of_logical"] = round(
                proc_cpu_1core / max(1, self._n_logical), 2
            )
            if self._cgroup_quota_cpus and self._cgroup_quota_cpus > 0:
                row["proc_cpu_pct_of_quota"] = round(
                    proc_cpu_1core / self._cgroup_quota_cpus, 2
                )
            row["host_cpu_pct"] = round(host_cpu, 2)
            mem = self._proc.memory_info()
            row["proc_rss_gb"] = round(mem.rss / (1024 ** 3), 3)
            vm = self._psutil.virtual_memory()
            row["host_ram_used_gb"] = round(vm.used / (1024 ** 3), 3)
            row["host_ram_total_gb"] = round(vm.total / (1024 ** 3), 3)
            row["host_ram_pct"] = round(vm.percent, 2)
        else:
            try:
                with open("/proc/self/status", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            kb = float(line.split()[1])
                            row["proc_rss_gb"] = round(kb / (1024 ** 2), 3)
                            break
            except OSError:
                pass
        if self._cgroup_quota_cpus is not None:
            row["cgroup_quota_cpus"] = round(self._cgroup_quota_cpus, 2)
        row["logical_cpus"] = self._n_logical
        return row

    def _sample_torch_vram(self) -> dict:
        row: dict = {}
        if torch.cuda.is_available():
            try:
                row["torch_alloc_gb"] = round(
                    torch.cuda.memory_allocated() / (1024 ** 3), 3
                )
                row["torch_reserved_gb"] = round(
                    torch.cuda.memory_reserved() / (1024 ** 3), 3
                )
            except Exception:
                pass
        return row

    def _loop(self) -> None:
        t0 = time.time()
        while not self._stop.is_set():
            ts = time.time()
            gpu_util, gpu_used, gpu_total = self._sample_gpu()
            row = {
                "t": round(ts - t0, 3),
                "wall": ts,
                "phase": self._phase,
                "update": self._update,
            }
            row.update(self._sample_cpu_ram())
            row.update(self._sample_torch_vram())
            if gpu_util is not None:
                row["gpu_util_pct"] = gpu_util
            if gpu_used is not None:
                row["gpu_mem_used_mib"] = gpu_used
            if gpu_total is not None:
                row["gpu_mem_total_mib"] = gpu_total
            self.samples.append(row)
            try:
                with self.out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except OSError:
                pass
            self._stop.wait(self.interval_s)

    def summarize(self) -> None:
        if not self.samples:
            print("[profile-resources] no samples collected")
            return
        by_phase: dict[str, list[dict]] = {}
        for s in self.samples:
            by_phase.setdefault(str(s.get("phase", "?")), []).append(s)

        def _stats(vals: list[float]) -> str:
            if not vals:
                return "n/a"
            vals = sorted(vals)
            n = len(vals)
            mean = sum(vals) / n
            p50 = vals[n // 2]
            p95 = vals[min(n - 1, int(n * 0.95))]
            return f"mean={mean:5.1f}  p50={p50:5.1f}  p95={p95:5.1f}  n={n}"

        print("\n===== resource sampler summary (per phase) =====")
        for phase, rows in by_phase.items():
            print(f"  phase={phase!r}  samples={len(rows)}")
            for key, label in (
                ("proc_cpu_pct_of_quota", "CPU % of cgroup quota"),
                ("proc_cpu_pct_of_logical", "CPU % of logical cores"),
                ("host_cpu_pct", "host CPU %"),
                ("gpu_util_pct", "GPU util %"),
                ("gpu_mem_used_mib", "GPU mem MiB"),
                ("torch_alloc_gb", "torch alloc GiB"),
                ("torch_reserved_gb", "torch reserved GiB"),
                ("proc_rss_gb", "proc RSS GiB"),
            ):
                vals = [
                    float(r[key])
                    for r in rows
                    if key in r and r[key] is not None
                ]
                if vals:
                    print(f"    {label:28s}  {_stats(vals)}")
        print(f"[profile-resources] raw JSONL -> {self.out_path}")


from plo5bp.actions import GATE_ACTIONS
from plo5bp.config import (
    VARIANT_NLH,
    VARIANT_PLO4,
    VARIANT_PLO5,
    VARIANT_PLO6,
    GameConfig,
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
# Every name `_sample_game_config` implements (the --stack-dist choices).
# --mix-tiers is validated against this, and the sampler itself raises on
# anything else (review 2026-09-20 A12): an unknown name used to fall through
# to the uniform --stack-range branch, so a `clubg_deep` typo silently trained
# on uniform 1-300bb (median 155bb instead of 52bb).
_KNOWN_STACK_DISTS = _VALID_STACK_DISTS + ("nlh_topoff",)


def _parse_mix_tiers(spec: str) -> list[str]:
    tiers = [t.strip() for t in spec.split(",") if t.strip()]
    bad = [t for t in tiers if t not in _KNOWN_STACK_DISTS]
    if bad:
        raise SystemExit(
            f"--mix-tiers: unknown tier(s) {bad} — valid: {_KNOWN_STACK_DISTS}"
        )
    return tiers


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


# ---- checkpoint I/O (review 2026-09-20 A7 / A3 / A14) ----------------------
def _atomic_torch_save(obj: object, path: Path) -> None:
    """torch.save to `<name>.tmp` in the SAME directory, then os.replace.

    A disk-full / OOM / kill mid-`torch.save` used to leave a truncated
    NEWEST file; the guardians resume from `ls -t <stem>_*.pt | head -1`, so
    every relaunch loaded it, crashed, and burned MAX_RESTARTS. os.replace is
    atomic on POSIX and Windows (same volume), so a reader sees the old file
    or the complete new one, never a partial. `<name>.pt.tmp` does not match
    the guardians' `<stem>_*.pt` glob or `discover_checkpoint_family`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _optimizer_sidecar_path(ckpt_path: Path) -> Path:
    """`<dir>/<family base>.optim.pt` for a checkpoint of that family
    (`vSix4_65.pt` and `vSix4.pt` -> `vSix4.optim.pt`).

    Deliberately NOT `<base>_optim.pt`: that matches `<base>_*.pt`, which is
    how the guardians pick the newest checkpoint to resume from, how
    vMin1's pick_warm and pod_prune_disk.sh enumerate a stem, and how the UI
    resolves the experimental format's model (`_latest_vmin1_ckpt`) — the
    sidecar is rewritten at every save, so it would usually BE the newest
    match and get warm-loaded / served as a model."""
    m = re.match(r"^(?P<base>.+)_\d+\.pt$", ckpt_path.name)
    base = m.group("base") if m else ckpt_path.stem
    return ckpt_path.with_name(f"{base}.optim.pt")


_SIDECAR_ENTRY_KEYS = (
    "optimizer_state", "param_shapes", "update_counter", "same_state_counters",
)


def _save_optimizer_sidecar(
    trainer: PPOTrainer,
    checkpoint_path: Path,
    update_counter: int,
    same_state_counters: "tuple[int, ...]" = (),
    keep_counter: "int | None" = None,
) -> None:
    """Rewrite the ONE rolling sidecar for this stem: Adam moments + the
    l2-init reference tensors + the `update_counter` of the checkpoint it was
    written with (`same_state_counters`: other checkpoints stamped from this
    exact optimizer state).

    `keep_counter` (the FINAL save only): carry the entry the file currently
    holds for that numbered checkpoint along as `previous`. A clean stop
    writes `<stem>.pt` a few updates past the newest `<stem>_<N>.pt`, but the
    guardians resume from the NUMBERED file — overwriting its moments here
    would make every stop/restart a cold-Adam start, the very thing the
    sidecar exists to prevent. The next mid save drops `previous` again."""
    path = _optimizer_sidecar_path(checkpoint_path)
    side = trainer.optimizer_sidecar_state()
    side["update_counter"] = int(update_counter)
    side["same_state_counters"] = [int(c) for c in same_state_counters]
    if keep_counter is not None and path.exists():
        try:
            prev = torch.load(path, map_location="cpu", weights_only=False)
            if int(prev["update_counter"]) == int(keep_counter):
                side["previous"] = {k: prev[k] for k in _SIDECAR_ENTRY_KEYS}
        except Exception as e:  # noqa: BLE001 - best effort, never block a save
            print(f"[optim] could not carry the u{keep_counter} sidecar entry: {e!r}")
    _atomic_torch_save(side, path)


def _restore_optimizer_sidecar(
    trainer: PPOTrainer, ckpt_path: Path, ckpt_update: "int | None"
) -> "dict[str, bool]":
    """PRODUCTION BEHAVIOR CHANGE — resume dynamics (review 2026-09-20 A3 +
    A14). On a warm start, restore from the loaded checkpoint's sidecar:

      - Adam MOMENTS, only when the sidecar was written with THIS checkpoint
        (its update_counter matches) and every shape matches; moments from a
        different point of the run are not applied to these weights.
      - the l2-init REFERENCES, whenever names+shapes match, regardless of
        the counter: they are the stem's original init and never change, so
        a rollback to an older checkpoint still decays toward the true init.

    Every outcome prints exactly one line per item — a cold Adam start or a
    re-anchored init is never silent. Expected effect: the first update after
    a relaunch takes a normal step (measured KL ~0.04) instead of a fresh-Adam
    ~lr*sign(g) step (KL ~1.08 -> KLSTOP, and a rollback livelock if it clears
    kl_hard), and decay-to-init stops drifting to the relaunch point."""
    out = {"moments": False, "l2_init": False}
    path = _optimizer_sidecar_path(ckpt_path)
    if not path.exists():
        print(
            f"[optim] no sidecar {path.name} — Adam starts COLD"
            + (
                "; l2-init re-anchors at the loaded weights"
                if trainer._l2_init_pairs else ""
            )
        )
        return out
    try:
        side = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(side, dict):
            raise ValueError("not a sidecar dict")
    except Exception as e:  # noqa: BLE001 - a bad sidecar must never kill a resume
        print(f"[optim] sidecar {path.name} unreadable ({e!r}) — Adam starts COLD")
        return out
    # The entry written WITH the loaded checkpoint: the file's own, or the
    # numbered-save entry a final save carried along as `previous`.
    entry = next(
        (
            e for e in (side, side.get("previous"))
            if isinstance(e, dict) and ckpt_update is not None
            and int(ckpt_update)
            in {e.get("update_counter"), *e.get("same_state_counters", [])}
        ),
        None,
    )
    if entry is None:
        print(
            f"[optim] sidecar {path.name} is from update "
            f"{side.get('update_counter')}, the loaded checkpoint from "
            f"{ckpt_update} — Adam starts COLD (moments not applied)"
        )
    else:
        ok, why = trainer.load_optimizer_moments(entry)
        out["moments"] = ok
        print(
            f"[optim] restored Adam moments from {path.name} (u{ckpt_update}, {why})"
            if ok else
            f"[optim] sidecar {path.name} refused ({why}) — Adam starts COLD"
        )
    if trainer._l2_init_pairs:
        ok, why = trainer.load_l2_init_refs(side)
        out["l2_init"] = ok
        print(
            f"[optim] restored the ORIGINAL l2-init references from {path.name} ({why})"
            if ok else
            f"[optim] l2-init references NOT restored ({why}) — decay-to-init "
            "re-anchors at the loaded weights"
        )
    return out


# ---- live control file hardening (review 2026-09-20 A13 / A20) -------------
# One message per DISTINCT problem: the control file is re-read every update,
# so anything keyed on its content would otherwise spam the log forever.
_CONTROL_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _CONTROL_WARNED:
        _CONTROL_WARNED.add(msg)
        print(msg)


def _read_control_text(path: Path) -> str | None:
    """Text of a live-control file (anneal_control.json / threads.txt), or
    None when it cannot be read. NEVER raises: the read sites used to catch
    only OSError, so a UTF-16 file (what Windows PowerShell 5.1 `>` /
    Out-File writes) raised UnicodeDecodeError — a ValueError — killed the
    run, and crash-looped every guardian relaunch on the same file.
    `utf-8-sig` also strips a UTF-8 BOM, which used to make the JSON
    unparseable and the file silently ignored forever."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, ValueError) as e:  # UnicodeDecodeError is a ValueError
        _warn_once(
            f"[control] cannot read {path}: {e!r} — IGNORED (save it as "
            "UTF-8; PowerShell 5.1 `>`/Out-File writes UTF-16)"
        )
        return None


# key -> (lo, hi, lo_is_exclusive). Outside the range = a typo, not a tuning
# choice (`true` -> lr 1.0, a dropped exponent, a sign slip): reject it.
_CONTROL_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "step": (0.0, 1.0, False),
    "target_kl": (0.0, 1e6, False),            # 0 = guard off
    "kl_hard": (0.0, 1e6, False),              # 0 = guard off
    "lr": (0.0, 0.1, True),
    "sizing_entropy_scale": (0.0, 1e3, False),
    "entropy_coef": (0.0, 10.0, False),
    "entropy_coef_deep": (0.0, 10.0, False),
    "clip_room_mid": (0.0, 1.0, True),
    "clip_room_ext": (0.0, 1.0, True),
    "q_fold_sup_coef": (0.0, 1e6, False),
}
_TIER_ENT_BOUNDS = (0.0, 10.0, False)


def _control_number(
    key: str, value: object, bounds: tuple[float, float, bool]
) -> tuple[float | None, str | None]:
    """(validated float, None) or (None, why-not). JSON `true` is a Python
    bool — an int subclass, so float(True) == 1.0 used to become lr 1.0."""
    lo, hi, lo_open = bounds
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{key}={value!r} is not a number"
    try:
        x = float(value)
    except OverflowError:  # a 400-digit int literal
        return None, f"{key} overflows a float"
    if not math.isfinite(x):
        return None, f"{key}={x} is not finite"
    if x < lo or x > hi or (lo_open and x == lo):
        rng = f"{'(' if lo_open else '['}{lo:g}, {hi:g}]"
        return None, f"{key}={x:g} is outside {rng}"
    return x, None


def _apply_anneal_control(
    raw: str | None,
    last_raw: str | None,
    tier_ent: dict[str, float],
    step: float,
    live_lr: float,
    live_ent: float,
    live_ent_deep: float,
    trainer=None,
    broadcast_entropy: bool = False,
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
                                            plain --stack-dist). Under
                                            --mix-configs
                                            (`broadcast_entropy`) it sets
                                            EVERY tier not named by a
                                            `tier_ent` in the same write
      {"entropy_coef_deep": 0.1}          — the deep-dist flat variant
      {"target_kl": 2.0}                  — retune the soft KL early-stop
      {"kl_hard": 12.0}                   — retune the hard rollback level
      {"lr": 1e-4}                        — retune the base learning rate
      {"sizing_entropy_scale": 2.5}       — scale the sizing-head entropy
      {"clip_room_mid": 0.07}             — prob-dependent clip: mid-band
                                            probability room (the per-update
                                            policy quota at ~50/50 gates)
      {"clip_room_ext": 0.12}             — same, the rare-gate ends of the U
      {"q_fold_sup_coef": 15.0}           — fold-column supervision weight
                                            inside the q-aux loss (qF canary)
      {"step": 0.003, "tier_ent": {...}}  — any combination

    A manual tier_ent set is one-shot: the anneal keeps lowering from
    the new level afterwards. The lr set is the BASE lr — the per-update
    warmup scale still multiplies it.

    An edit is applied ATOMICALLY or not at all, and never silently
    (review 2026-09-20 A13 / A20): malformed JSON / a non-object / a
    non-object `tier_ent` / any value that is a bool, non-numeric, negative,
    non-finite or out of `_CONTROL_BOUNDS` makes the WHOLE edit a no-op
    (returned content unchanged, so a half-written save is simply retried
    on the next loop) with ONE log line per distinct content. Unknown keys
    and unknown tier names are logged once and skipped; the rest applies."""
    if raw is None or raw == last_raw:
        return step, last_raw, live_lr, live_ent, live_ent_deep
    unchanged = (step, last_raw, live_lr, live_ent, live_ent_deep)
    snippet = raw.strip().replace("\n", " ")[:160]
    try:
        ctrl = json.loads(raw)
    except ValueError:
        _warn_once(f"[anneal-control] malformed JSON IGNORED: {snippet}")
        return unchanged
    tiers_raw = ctrl.get("tier_ent") if isinstance(ctrl, dict) else None
    if not isinstance(ctrl, dict) or not isinstance(tiers_raw, (dict, type(None))):
        # C1: valid JSON but not an object (a list, bare string, or number
        # from a live-tune typo), or a non-object tier_ent. Treat as
        # malformed and ignore, per the docstring.
        _warn_once(f"[anneal-control] not a JSON object — IGNORED: {snippet}")
        return unchanged

    errors: list[str] = []
    vals: dict[str, float] = {}
    for key, bounds in _CONTROL_BOUNDS.items():
        if key in ctrl:
            x, why = _control_number(key, ctrl[key], bounds)
            if why is not None:
                errors.append(why)
            else:
                vals[key] = x
    new_tiers: dict[str, float] = {}
    unknown_tiers: list[str] = []
    for tier, v in (tiers_raw or {}).items():
        if tier not in tier_ent:
            unknown_tiers.append(str(tier))
            continue
        x, why = _control_number(f"tier_ent[{tier}]", v, _TIER_ENT_BOUNDS)
        if why is not None:
            errors.append(why)
        else:
            new_tiers[tier] = x
    if errors:
        _warn_once(
            "[anneal-control] edit IGNORED, NOTHING applied — "
            + "; ".join(errors) + f" | content: {snippet}"
        )
        return unchanged
    unknown_keys = sorted(set(ctrl) - set(_CONTROL_BOUNDS) - {"tier_ent"})
    if unknown_keys:
        _warn_once(
            f"[anneal-control] unknown key(s) {unknown_keys} skipped "
            f"(valid: {sorted(_CONTROL_BOUNDS) + ['tier_ent']})"
        )
    if unknown_tiers:
        _warn_once(
            f"[anneal-control] unknown tier_ent tier(s) {sorted(unknown_tiers)} "
            f"skipped (this run's tiers: {sorted(tier_ent)})"
        )

    new_step = vals.get("step", step)
    new_target_kl = vals.get("target_kl")
    new_kl_hard = vals.get("kl_hard")
    new_lr = vals.get("lr")
    new_sizing_scale = vals.get("sizing_entropy_scale")
    new_ent = vals.get("entropy_coef")
    new_ent_deep = vals.get("entropy_coef_deep")
    new_clip_mid = vals.get("clip_room_mid")
    new_clip_ext = vals.get("clip_room_ext")
    new_q_fold_sup = vals.get("q_fold_sup_coef")
    if new_step != step:
        print(f"[anneal-control] step {step} -> {new_step}")
    for tier, v in new_tiers.items():
        if tier_ent[tier] != v:
            print(f"[anneal-control] tier_ent[{tier}] {tier_ent[tier]} -> {v}")
        tier_ent[tier] = v
    # Mix-configs consumes tier_ent (per-row coefs), not the flat coef — an
    # `entropy_coef` edit used to be a silent no-op there (V5_DESIGN.md B5).
    # Broadcast it to every tier so the natural key works in both modes; a
    # tier NAMED by `tier_ent` in this same write keeps its explicit value.
    # Keyed on the key's PRESENCE, not on the flat value changing (A20): the
    # flat coef goes stale under mixing (nothing reads it), so re-sending the
    # launch value to undo per-tier edits compared equal and did nothing.
    if broadcast_entropy and new_ent is not None:
        for tier in tier_ent:
            if tier in new_tiers:
                continue
            if tier_ent[tier] != new_ent:
                print(
                    f"[anneal-control] tier_ent[{tier}] {tier_ent[tier]} "
                    f"-> {new_ent} (entropy_coef broadcast)"
                )
            tier_ent[tier] = new_ent
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
    # Prob-dependent clip rooms: _gate_clip_bounds reads these attributes
    # per minibatch, so mutating them live-retunes the per-update policy
    # quota (2026-07-11: the mid band IS the KL ceiling at mixed gates —
    # LR past saturation can't raise it, only this can). No-op unless the
    # run was built with clip_prob_dependent.
    if new_clip_mid is not None and trainer is not None:
        if trainer._clip_room_mid != new_clip_mid:
            print(
                f"[anneal-control] clip_room_mid "
                f"{trainer._clip_room_mid} -> {new_clip_mid}"
            )
        trainer._clip_room_mid = new_clip_mid
    if new_clip_ext is not None and trainer is not None:
        if trainer._clip_room_ext != new_clip_ext:
            print(
                f"[anneal-control] clip_room_ext "
                f"{trainer._clip_room_ext} -> {new_clip_ext}"
            )
        trainer._clip_room_ext = new_clip_ext
    # Fold-supervision weight: read per-minibatch in _q_fold_sup_term, so
    # a live edit rebalances the q gradient without a restart (audit #2:
    # the qF canary is the readout).
    if new_q_fold_sup is not None and trainer is not None:
        if trainer._q_fold_sup != new_q_fold_sup:
            print(
                f"[anneal-control] q_fold_sup_coef "
                f"{trainer._q_fold_sup} -> {new_q_fold_sup}"
            )
        trainer._q_fold_sup = new_q_fold_sup
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
        by `step` (clamped at `floor`) and RATCHET the baseline UP — per
        street `max(old, now)` — so each successive cut must keep paying for
        the best aggression the tier has shown.
      - Any street DROPPED: HOLD ent and KEEP the old baseline. The next block
        must recover to the pre-drop level before lowering resumes; this stops
        the anneal from chasing F/T/R downward into passivity.

    PRODUCTION BEHAVIOR CHANGE, block-anneal mode only (review 2026-09-20
    A15): a held block used to REPLACE the baseline with now_ftr, so a slip
    inside the tolerance also lowered the bar for the next block. F/T/R
    sliding 0.9/block (< tol 1.0) read as "held" forever: 30 -> 20.1 over 11
    blocks with entropy cut on every one — exactly the downward chase the
    drop rule exists to stop. The tolerance now absorbs block-to-block NOISE
    around a bar that only moves up. Expected effect: fewer entropy cuts on
    tiers whose aggression is drifting down; identical on tiers that hold.
    """
    if baseline is None:
        return ent, (now_ftr[0], now_ftr[1], now_ftr[2]), "record-baseline"
    held = all(now_ftr[s] >= baseline[s] - tol for s in range(3))
    if held:
        ratchet = (
            max(baseline[0], now_ftr[0]),
            max(baseline[1], now_ftr[1]),
            max(baseline[2], now_ftr[2]),
        )
        if ent > floor:
            new_ent = max(floor, ent - step)
            return new_ent, ratchet, "lowered"
        return floor, ratchet, "held@floor"
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


# Stack re-draws before `_sample_game_config` gives up and clamps (A1).
_STACK_RESAMPLE_TRIES = 32

# Consecutive rolled-back updates before (and between) livelock alarms (A3).
_ROLLBACK_ALARM_AFTER = 5

# The files python/plo5bp/ui/server.py serves by default (FORMATS registry).
_UI_SERVED_CHECKPOINTS = frozenset({"stub.pt", "nlh_stub.pt"})


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

    if stack_dist not in _KNOWN_STACK_DISTS:
        # A12: never fall through to the uniform branch on a typo.
        raise ValueError(
            f"unknown stack_dist {stack_dist!r} — valid: {_KNOWN_STACK_DISTS}"
        )
    effective_stack_dist = stack_dist
    if stack_dist == "clubgg_mix":
        effective_stack_dist = "clubgg_deep" if rng.random() < 0.5 else "clubgg"
    elif stack_dist == "full_mix":
        effective_stack_dist = str(
            rng.choice(("clubgg", "clubgg_deep", "deep"))
        )

    def _draw_depths_bb() -> np.ndarray:
        if effective_stack_dist == "clubgg":
            return np.array(
                [_sample_clubgg_stack_bb(stack_lo_bb, stack_hi_bb, rng) for _ in range(n_seats)]
            )
        if effective_stack_dist == "clubgg_deep":
            return np.array(
                [
                    _sample_clubgg_stack_bb(
                        stack_lo_bb, stack_hi_bb, rng, bands=_CLUBGG_DEEP_STACK_BANDS
                    )
                    for _ in range(n_seats)
                ]
            )
        if effective_stack_dist == "nlh_topoff":
            return np.array(
                [_sample_nlh_topoff_stack_bb(rng) for _ in range(n_seats)]
            )
        if effective_stack_dist in ("agro_deep", "deep"):
            return rng.uniform(100.0, 250.0, size=n_seats)
        if stack_lo_bb == stack_hi_bb:
            return np.full(n_seats, stack_lo_bb)
        return rng.uniform(stack_lo_bb, stack_hi_bb, size=n_seats)

    # A hand needs at least TWO seats that can still act after posting, or
    # there is nothing to decide: a seat with stack <= ante is all-in on the
    # ante (NLH: <= ante + bb — it may also owe the big blind). With fewer
    # than two such seats the hand runs out at deal, and a config where that
    # is ALWAYS so gives the collector no row, ever — the batched loop span
    # forever on it (review 2026-09-20 A1; ~6e-5 per update with the clubgg
    # 1-20bb band, i.e. ~6% per 1,000 updates, silent under the guardians).
    # Exactly one live seat is the same problem in a milder form: a whole
    # sub-rollout of forced single-action rows. Resample the stacks (the
    # tier draw above is kept); the FIRST draw is the pre-fix one, so every
    # config that was already playable is byte-identical.
    live_floor = ante + (bb if variant == VARIANT_NLH else 0)
    for _ in range(_STACK_RESAMPLE_TRIES):
        depths_bb = _draw_depths_bb()
        stacks = tuple(int(round(float(d) * bb)) for d in depths_bb)
        if sum(s > live_floor for s in stacks) >= 2:
            break
    else:
        # The distribution itself sits at/below the ante (e.g. a fixed
        # --stack-range under 3bb): resampling cannot help. Lift the two
        # deepest seats to one bb behind after posting, loudly, rather than
        # hand the collector a config it must refuse.
        lift = sorted(range(n_seats), key=lambda i: stacks[i], reverse=True)[:2]
        stacks = tuple(
            max(s, live_floor + bb) if i in lift else s
            for i, s in enumerate(stacks)
        )
        _warn_once(
            f"[config] stack_dist={effective_stack_dist!r} range "
            f"{stack_lo_bb:g}:{stack_hi_bb:g}bb cannot seat two players with "
            f"more than {live_floor / bb:g}bb (ante"
            + ("+bb" if variant == VARIANT_NLH else "")
            + f") after {_STACK_RESAMPLE_TRIES} draws — CLAMPED the two "
            f"deepest seats to {(live_floor + bb) / bb:g}bb"
        )
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
    launch log. Tests: tests/python/test_v6_preset.py."""
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
        "--obs-mode",
        choices=["full", "minimal"],
        default="full",
        help="Observation layout. full=OBS_DIM 1171 (default). minimal=bare table-visible 796 (cards, street, active/all-in, stacks, pot/to_call/min/max, commits, seat-exists, button, history). Cold-start only; no warm-start from full-obs checkpoints. Skips opp-outcome MC for speed.",
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
        help="HL-Gauss Gaussian sigma in bin-widths (→0 = hard two-hot; A/B "
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
        default=0.10,
        help="Target absolute prob-movement room at the gate extremes (p->0/1) "
        "for --clip-prob-dependent. 0.10 = ~10 points/update.",
    )
    parser.add_argument(
        "--clip-room-mid",
        type=float,
        default=0.05,
        help="Target absolute prob-movement room at a 50/50 gate for "
        "--clip-prob-dependent. 0.05 = ~5 points/update (tighter than the "
        "extremes -> the symmetric U).",
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
        "(1.25x the 2/3/4 weight — ~22.7%% each vs ~18.2%%).",
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
        "AND cuda; also live-tunable via runs/threads.txt.",
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
    args = parser.parse_args()

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

    threads_file = Path("runs/threads.txt")
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

    # A6 drain_inflight plumbing. The flag's home is a TrainingConfig field
    # (so it rides in every checkpoint's `config` stamp), but config.py is
    # owned by another workstream in the 2026-09-20 fix pass: set the field
    # only if the dataclass HAS it, and ALWAYS hand the value to the
    # collectors as an explicit kwarg (they resolve kwarg > config field >
    # True), so --no-drain-inflight works either way. The checkpoint also
    # stamps it top-level (`drain_inflight`).
    drain_inflight = bool(args.drain_inflight)
    _tc_extra: dict = {}
    if "drain_inflight" in {f.name for f in dataclasses.fields(TrainingConfig)}:
        _tc_extra["drain_inflight"] = drain_inflight
    if not drain_inflight:
        print(
            "[rollout] --no-drain-inflight: LEGACY collection — hands in "
            "flight at the row target are dropped (long hands under-sampled)"
        )

    train_cfg = TrainingConfig(
        **_tc_extra,
        lr=args.lr,
        num_updates=args.num_updates,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        obs_mode=args.obs_mode,
        compact_obs=not args.no_compact_obs,
        batched_opponents=not args.no_batched_opponents,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        snapshot_every=args.snapshot_every,
        ev_runout_samples=args.ev_runout_samples,
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
    # corrected features, 1 = the exact pre-fix values. Read through getattr
    # so this works on a tree where the switch has not landed yet (-> 2).
    obs_rev = int(getattr(_encoding, "OBS_SEMANTICS_REV", 2))
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
            f"columns as bits + {_obs_layout.n_real} verbatim f32 = "
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
    )
    critic.to(train_cfg.device)
    print(f"[device] learner on {train_cfg.device}")
    # Annealing state restored from the checkpoint (None when absent / cold).
    restored_update: int | None = None
    restored_tier_ent: dict | None = None
    restored_baseline: dict | None = None
    restored_block_acc: dict | None = None
    restored_pool_updates: list | None = None
    restored_control_applied: str | None = None
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
        # v7 Q-surface semantics guards (V7_DESIGN.md WS1): q_fold_zero /
        # q_base_raw leave no trace in the state dict, but flipping either
        # reinterprets the whole learned Q surface (the adv rows absorbed
        # the old base), so a silent warm-start across a flip would train
        # against shifted targets. Old checkpoints lack the keys -> False.
        for _qk in ("q_fold_zero", "q_base_raw"):
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
                )
                critic.to(train_cfg.device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        crit_sd = ckpt["critic"]
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
        restored_tier_ent = ckpt.get("anneal_tier_ent")
        restored_baseline = ckpt.get("anneal_baseline")
        restored_block_acc = ckpt.get("anneal_block_acc")
        restored_pool_updates = ckpt.get("pool_member_updates")
        restored_control_applied = ckpt.get("anneal_control_applied")

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
            _restore_optimizer_sidecar(trainer, args.load_checkpoint, restored_update)
        else:
            print(
                "[optim] --no-optimizer-sidecar: LEGACY resume — Adam starts "
                "COLD, l2-init re-anchors at the loaded weights"
            )
    pool = OpponentPool(capacity=train_cfg.opponent_pool_size, seed=args.seed)
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
                live_anneal_step,
                last_anneal_control,
                live_lr,
                live_entropy_coef,
                live_entropy_coef_deep,
            ) = _apply_anneal_control(
                _pre_raw, None, tier_ent, live_anneal_step,
                live_lr, live_entropy_coef, live_entropy_coef_deep,
                trainer=trainer, broadcast_entropy=args.mix_configs,
            )
            if last_anneal_control is None:
                last_anneal_control = _pre_raw  # rejected now: don't retry forever
            else:
                anneal_control_applied = _pre_raw
            if args.anneal_entropy and restored_tier_ent:
                # A manual tier_ent set is ONE-SHOT: the anneal kept lowering
                # from it, and those annealed coefs rode in the checkpoint.
                # They are newer than the file's — put them back on top.
                tier_ent.update(
                    {k: float(v) for k, v in restored_tier_ent.items() if k in tier_ent}
                )
                print(f"[anneal-control] annealed tier_ent kept: {tier_ent}")
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

    def _save_mid(update_idx: int) -> None:
        nonlocal last_mid_saved_update
        # `update_idx` is the LOCAL loop counter; numbered checkpoints are named
        # and stamped on the GLOBAL update axis (base_update + local). Without
        # this, an anneal-off warm relaunch (which resets the loop counter to 0)
        # would rewrite a prior segment's nlh4_5.pt/_10.pt... over the originals
        # AND stamp update_counter=5, poisoning the next warm-start's pool
        # seeding (ws_target reads that counter). See base_update below.
        global_idx = base_update + update_idx
        game_cfg_snap = sampled_game_cfg.__dict__
        mid_path = args.checkpoint.with_name(f"{args.checkpoint.stem}_{global_idx}.pt")
        if (
            args.load_checkpoint is not None
            and mid_path.resolve() == Path(args.load_checkpoint).resolve()
        ):
            # C3 (narrowed after adversarial review): never overwrite THE
            # checkpoint this run warm-started from. An anneal-ON resume
            # continues the loop counter from the restored update, so its first
            # iteration lands back on the loaded file's own grid index — saving
            # would rewrite the exact restore point just loaded with weights
            # carrying one extra PPO update (either cadence branch can fire),
            # destroying the clean-recovery file the collapse playbook depends
            # on. Skip; the next cadence tick writes a fresh number. Guarding
            # ONLY the loaded file (not blanket write-once) preserves
            # last-write-wins for every legitimate collision: orchestrations
            # that re-run a phase from a fixed source must refresh their
            # outputs, and an anneal-ON resume must checkpoint its new lineage
            # over the old segment's later files.
            print(
                f"[ckpt] skip: {mid_path.name} is this run's warm-start source "
                "(never overwritten)"
            )
            return
        _atomic_torch_save(
            {
                "model": model.state_dict(),
                "critic": critic.state_dict(),
                "head_version": model.head_version,
                "config": train_cfg.__dict__,
                "game_config": game_cfg_snap,
                "gate_count": GATE_ACTIONS,
                "variant": args.variant,
                "anchor_count": model._anchor_count,
                # NOTE (review 2026-09-20 A20, documented, deliberately left
                # alone): a MID save stamps the 0-based index of the update
                # that just finished, the FINAL save stamps the COUNT of
                # updates done — so the same weights read N here and N+1
                # there. Consumers (pool seeding, the anneal resume, the
                # optimizer sidecar match) treat both as "this file's update";
                # unifying them would shift every stem's numbering.
                "update_counter": global_idx,
                # Metadata only (update indices, not weights): lets a
                # warm-start reconstruct the exact pool membership from
                # the sibling files still on disk.
                "pool_member_updates": list(pool.tags),
                "anneal_tier_ent": tier_ent,
                "anneal_baseline": tier_baseline,
                "anneal_block_acc": block_acc,
                # Truthful regimen stamp under --mix-configs (game_config
                # above is just the first sub-rollout's draw — B9).
                "mix_configs": bool(args.mix_configs),
                "mix_tiers": list(mix_tiers) if args.mix_configs else None,
                "configs_per_tier": (
                    int(args.configs_per_tier) if args.mix_configs else None
                ),
                # KL-anchor EMA reference (None when the magnet is off);
                # restored on warm-start so the pull-toward-history
                # survives relaunches. Doubles as the smoother serving
                # actor (promote model_ema instead of the last iterate).
                "model_ema": trainer.ref_state_dict(),
                # A4: raw text of the last APPLIED runs/anneal_control.json
                # (None = none applied); a relaunch re-applies the file only
                # when it still matches this.
                "anneal_control_applied": anneal_control_applied,
                "drain_inflight": drain_inflight,
                # Observation-semantics revision these weights trained on
                # (the warm-start guard + pool seeding key off it).
                "obs_rev": obs_rev,
            },
            mid_path,
        )
        if args.optimizer_sidecar:
            _save_optimizer_sidecar(trainer, args.checkpoint, global_idx)
        last_mid_saved_update = update_idx

    # Restore the update counter only when annealing, so the block cycle
    # continues across a relaunch instead of resetting to block 1 (which
    # under-trains the deep tier). Anneal-off keeps today's reset-to-0 for the
    # LOOP counter so the LR-warmup ramp and block-cycle position resume with
    # their current gentle-restart semantics. `base_update` carries the true
    # cumulative update index onto which checkpoint filenames / update_counter /
    # pool snapshot tags are stamped, so a same-stem relaunch never clobbers
    # prior nlh4_<N>.pt files and the counter a later warm-start reads stays
    # truthful (fixes silent checkpoint overwrite + pool-seeding poison).
    _restored = int(restored_update) if restored_update is not None else 0
    if args.anneal_entropy and restored_update is not None:
        update = _restored  # loop counter continues; offset already folded in
        base_update = 0
    else:
        update = 0
        base_update = _restored
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

    # Optional 1 Hz CPU/GPU/RAM/VRAM sampler (profile runs and always-on phase
    # wall timers). Always log phase wall times; start the sampler whenever
    # --profile-one-update is set so the under-saturation story is on disk.
    resource_sampler: _ResourceSampler | None = None
    # Resource sampler for --profile-one-update OR lightweight step-timer runs
    # (PLO5BP_STEP_TIMERS=1): full torch.profiler at 9M rows OOMs/hangs on
    # key_averages; step timers + 1Hz samples are the supported path.
    _want_sampler = bool(args.profile_one_update) or (
        os.environ.get("PLO5BP_STEP_TIMERS", "").strip().lower()
        in ("1", "true", "yes", "on")
    )
    if _want_sampler:
        _samp_u = args.profile_at_update if args.profile_one_update else 0
        resource_sampler = _ResourceSampler(
            out_path=Path(f"runs/profile_resources_u{_samp_u}.jsonl"),
            interval_s=1.0,
        )
        resource_sampler.start()

    consecutive_rollbacks = 0
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
                live_anneal_step,
                last_anneal_control,
                live_lr,
                live_entropy_coef,
                live_entropy_coef_deep,
            ) = _apply_anneal_control(
                control_raw, last_anneal_control, tier_ent, live_anneal_step,
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
            mix_cfg_tiers = [
                tier
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
                drain_inflight=drain_inflight,
            )
            update_entropy_coef = tier_ent.get(mix_tiers[0], args.entropy_coef)
        elif blocks:
            batch = collector(
                model, pool, sampled_game_cfg, train_cfg, rng, critic=critic,
                drain_inflight=drain_inflight,
            )
            # tier_ent[tier] == the static block value when --anneal-entropy is
            # off (it is never mutated then), so this is identical to today.
            update_entropy_coef = tier_ent[active_tier]
        else:
            batch = collector(
                model, pool, sampled_game_cfg, train_cfg, rng, critic=critic,
                drain_inflight=drain_inflight,
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
        for _pg in trainer.optimizer.param_groups:
            _pg["lr"] = live_lr * lr_scale
        if resource_sampler is not None:
            resource_sampler.set_phase("optimize", update=update)
        _t_opt0 = time.perf_counter()
        stats = trainer.update(batch, rng, entropy_coef=update_entropy_coef)
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

        now = time.time()

        # Snapshot on update count, then also on wall-clock if configured.
        # Tag on the GLOBAL axis so pool_member_updates stays consistent with
        # the numbered checkpoint filenames a warm-start reads (base_update
        # is 0 on a fresh/annealing run, so this is a no-op there).
        if update % train_cfg.snapshot_every == 0:
            pool.snapshot(model, tag=base_update + update)
        if args.snapshot_every_sec > 0 and now - last_snapshot_sec >= args.snapshot_every_sec:
            pool.snapshot(model, tag=base_update + update)
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
            # Per-tier F/T/R under mix-configs (V5_DESIGN.md B5): the
            # pooled line above can't drive the per-tier stop-loss; this
            # one can. Same semantics as bonus%(F/T/R), bucketed by tier.
            if getattr(batch, "tier_ftr", None):
                parts = []
                for _t, (_bonus, _steps) in batch.tier_ftr.items():
                    f_, t_, r_ = (
                        100.0 * _bonus[s] / max(1, _steps[s]) for s in range(3)
                    )
                    parts.append(
                        f"{_t}={f_:4.1f}/{t_:4.1f}/{r_:4.1f}"
                        f"(ent {tier_ent.get(_t, update_entropy_coef):.3f})"
                    )
                print("        [ftr-tier] " + "  ".join(parts))

            # P11 measurement: peak GPU memory over this log window. `del batch`
            # (below) frees the prior update's batch before the next collection
            # allocates its own, removing the collect-end 2x-batch spike; this
            # line shows where the true per-update peak now lands so the rollout
            # can be grown to fit. Bit-exact — reporting only.
            if args.device == "cuda":
                _peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
                _peak_resv = torch.cuda.max_memory_reserved() / (1024 ** 3)
                print(
                    f"        [vram] peak alloc={_peak_alloc:.1f} GiB  "
                    f"reserved={_peak_resv:.1f} GiB  "
                    f"(rollout={batch.obs.shape[0]:,} rows)"
                )
                torch.cuda.reset_peak_memory_stats()

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

    _atomic_torch_save(
        {
            "model": model.state_dict(),
            "critic": critic.state_dict(),
            "head_version": model.head_version,
            "config": train_cfg.__dict__,
            "game_config": sampled_game_cfg.__dict__,
            "gate_count": GATE_ACTIONS,
            "variant": args.variant,
            "anchor_count": model._anchor_count,
            # COUNT of updates done — one more than the index a mid save of
            # the same weights stamps (see the note in _save_mid).
            "update_counter": base_update + update,
            "pool_member_updates": list(pool.tags),
            "anneal_tier_ent": tier_ent,
            "anneal_baseline": tier_baseline,
            "anneal_block_acc": block_acc,
            "mix_configs": bool(args.mix_configs),
            "mix_tiers": list(mix_tiers) if args.mix_configs else None,
            "configs_per_tier": (
                int(args.configs_per_tier) if args.mix_configs else None
            ),
            "model_ema": trainer.ref_state_dict(),
            "anneal_control_applied": anneal_control_applied,
            "drain_inflight": drain_inflight,
            "obs_rev": obs_rev,
        },
        args.checkpoint,
    )
    if args.optimizer_sidecar:
        # When no update ran since the last numbered save, that file holds
        # these exact weights under the mid-save stamp (one lower): the
        # sidecar is valid for it too — which is what a guardian resumes from.
        _mid = (
            None if last_mid_saved_update is None
            else base_update + last_mid_saved_update
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


if __name__ == "__main__":
    main()
