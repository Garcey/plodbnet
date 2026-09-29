"""Checkpoint I/O: the checkpoint format, atomic saves, the rolling optimizer
sidecar, the finiteness guard and the run's provenance."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from plo5bp.ppo import PPOTrainer


# The files python/plo5bp/ui/server.py serves by default (FORMATS registry).
_UI_SERVED_CHECKPOINTS = frozenset({"stub.pt", "nlh_stub.pt"})


# ---- the checkpoint dict (2026-09-28, ML-016 / ML-019 / ML-031) -------------
# `schema` versions the DICT a training checkpoint holds. 1 = every file
# written before the key existed (a missing key reads as 1); 2 = adds
# "schema", "arch" (network.actor_arch / critic_arch: every constructor
# argument) and "provenance" (run_provenance: command line, code, engine,
# versions, environment). Keys are only ever ADDED -- none is renamed or
# removed -- so every older checkpoint still loads, warm-starts, seeds the
# pool and serves, and older loaders ignore the new keys.
CHECKPOINT_SCHEMA = 2


def checkpoint_schema(ckpt: dict) -> int:
    """The schema of a loaded checkpoint dict (1 when unstamped)."""
    return int(ckpt.get("schema", 1)) if isinstance(ckpt, dict) else 1


# ---- update numbering (2026-09-28, ML-063) -----------------------------------
# "index" (every stem so far; the default): a numbered save stamps the 0-based
# index of the update that just finished and the final save the COUNT, a
# launch's first update is never saved, and a relaunch's numbers fall one
# behind its weights. "count" (--number-by-count): everything stamps the count
# of updates done. Stamped as ckpt["numbering"] (absent = "index"); a stem never
# mixes the two.
NUMBERING_INDEX = "index"
NUMBERING_COUNT = "count"

_NUMBERED = re.compile(r"^(?P<stem>.+)_\d+$")


def is_own_stem(load_path: "str | Path", checkpoint: "str | Path") -> bool:
    """Whether `load_path` is one of `checkpoint`'s own files (`<stem>.pt` or
    `<stem>_<N>.pt`, any directory) -- a relaunch, not a new stem."""
    name, stem = Path(load_path).stem, Path(checkpoint).stem
    m = _NUMBERED.match(name)
    return name == stem or (m is not None and m.group("stem") == stem)


def resolve_numbering(
    want_count: bool,
    load_path: "str | Path | None",
    checkpoint: "str | Path",
    loaded_numbering: "str | None",
) -> bool:
    """True = count numbering for this launch. A relaunch (loading the stem's
    own file) follows that file's numbering -- a stem never switches midway;
    a cold start or a warm start from another stem takes the flag."""
    if load_path is not None and is_own_stem(load_path, checkpoint):
        count = (loaded_numbering or NUMBERING_INDEX) == NUMBERING_COUNT
        if want_count and not count:
            raise SystemExit(
                f"--number-by-count: {Path(load_path).name} is this stem's own, "
                "index-numbered file -- the numbering can only change at a stem "
                "boundary (a new --checkpoint name), or its numbers would jump "
                "by one mid-stem"
            )
        return count
    return bool(want_count)


def build_checkpoint(*, schema_fields: dict, **fields) -> dict:
    """The one place a training checkpoint dict is assembled (the mid saves
    and the final save used to build it separately, so a new key had to be
    added twice). `fields` = the legacy keys, in their historical order;
    `schema_fields` = the schema-2 additions."""
    out = dict(fields)
    out["schema"] = CHECKPOINT_SCHEMA
    out.update(schema_fields)
    return out


# What a DERIVED checkpoint (an average, a distilled student -- made by a tool,
# not by a training run) carries over from its source: the networks, what
# describes them, and the lineage's cumulative `update_counter` (a stem
# continued from a derived file keeps counting: vSix6 went on from its distilled
# teacher's u1290). Not the source RUN's state -- opponent-pool membership,
# applied live-control text, anneal state, EMA -- which a warm start from the
# derived file used to pick up as if it were that run (2026-09-28, ML-056).
DERIVED_KEEP_KEYS = (
    "model", "critic", "head_version", "config", "game_config", "gate_count",
    "variant", "anchor_count", "mix_configs", "mix_tiers", "configs_per_tier",
    "drain_inflight", "obs_rev", "schema", "update_counter",
)


def derived_checkpoint(source: dict, how: str, sources: "list[str]", **fields) -> dict:
    """A checkpoint dict for a tool-made network: the allow-listed keys of
    `source`, then `fields` (e.g. the new "model" / "config" / "arch"), plus
    `derived_from` = {how, sources, source_update_counter}. `arch` is kept
    only when passed (a student's size differs from its teacher's; loaders
    sniff the weights when it is absent)."""
    out = {k: source[k] for k in DERIVED_KEEP_KEYS if k in source}
    out["model_ema"] = None
    out.update(fields)
    out["derived_from"] = {
        "how": how,
        "sources": [str(s) for s in sources],
        "source_update_counter": source.get("update_counter"),
    }
    return out


class NonFiniteCheckpointError(RuntimeError):
    """A save was refused: the weights hold NaN/Inf (ML-001)."""


def nonfinite_tensors(obj: object, prefix: str = "") -> list[str]:
    """Paths of the floating-point tensors inside `obj` (nested dicts / lists)
    that hold a NaN or an Inf."""
    bad: list[str] = []
    if torch.is_tensor(obj):
        if obj.is_floating_point() and not bool(torch.isfinite(obj).all()):
            bad.append(prefix or ".")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad += nonfinite_tensors(v, f"{prefix}/{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += nonfinite_tensors(v, f"{prefix}[{i}]")
    return bad


def assert_finite_for_save(payload: dict, what: str, keys: "tuple[str, ...]") -> None:
    """Last line of defence (ML-001 B): refuse to write `what` when a tensor
    under `payload[key]` for any of `keys` is not finite. A poisoned file on
    disk is worse than a crash: the guardian would resume every relaunch
    from it. Raising leaves the previous files as the newest good ones."""
    bad: list[str] = []
    for key in keys:
        if payload.get(key) is not None:
            bad += nonfinite_tensors(payload[key], key)
    if bad:
        shown = ", ".join(bad[:8]) + (" ..." if len(bad) > 8 else "")
        raise NonFiniteCheckpointError(
            f"refusing to write {what}: {len(bad)} tensor(s) hold NaN/Inf ({shown})"
        )


def _json_safe(x: object) -> object:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return _json_safe(float(x))
    return str(x)


_PROVENANCE_ENV_PREFIXES = (
    "PLO5", "TORCHINDUCTOR", "RAYON_", "OMP_", "PYTORCH_", "CUDA_VISIBLE",
    "MALLOC_", "NUMPY_", "PYTHONHASHSEED",
)


def _git_state(repo: Path) -> dict:
    def run(*args: str) -> "str | None":
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None}
    status = run("status", "--porcelain", "--untracked-files=no") or ""
    changed = [ln[3:] for ln in status.splitlines() if ln.strip()]
    return {"commit": commit.strip(), "dirty": bool(changed), "changed": changed[:40]}


def _engine_fingerprint() -> dict:
    try:
        import plo5bp._engine as eng

        path = Path(eng.__file__)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        st = path.stat()
        return {
            "file": path.name, "bytes": st.st_size, "sha256": h.hexdigest()[:16],
            "mtime_utc": _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc)
            .isoformat(timespec="seconds"),
        }
    except Exception as e:  # noqa: BLE001 -- provenance is best effort
        return {"error": repr(e)}


def run_provenance(argv: "list[str] | None" = None) -> dict:
    """Where a run came from (ML-019): its command line, the code (git commit,
    dirty files), the engine binary, library versions, the environment
    variables that change training, the host. Computed once per launch,
    stamped into every checkpoint and appended to runs/<stem>.launches.jsonl,
    so "which code and flags made this checkpoint?" never depends on a
    guardian script that was edited since."""
    import plo5bp

    repo = Path(plo5bp.__file__).resolve().parents[2]
    return {
        "argv": list(sys.argv if argv is None else argv),
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "git": _git_state(repo),
        "engine": _engine_fingerprint(),
        "env": {
            k: os.environ[k] for k in sorted(os.environ)
            if k.startswith(_PROVENANCE_ENV_PREFIXES)
        },
    }


def append_launch_record(path: Path, record: dict) -> None:
    """One JSON line per launch (runs/<stem>.launches.jsonl). Best effort:
    a full disk must not stop a run from starting."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
    except OSError as e:
        print(f"[provenance] could not append to {path}: {e!r}")


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
    assert_finite_for_save(side, f"the optimizer sidecar {path.name}", ("optimizer_state", "l2_init"))
    _atomic_torch_save(side, path)


def _restore_optimizer_sidecar(
    trainer: PPOTrainer,
    ckpt_path: Path,
    ckpt_update: "int | None",
    allow_actor_only: bool = False,
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
        ok, why = trainer.load_optimizer_moments(entry, allow_actor_only=allow_actor_only)
        out["moments"] = ok
        print(
            f"[optim] restored Adam moments from {path.name} (u{ckpt_update}, {why})"
            if ok else
            f"[optim] sidecar {path.name} refused ({why}) — Adam starts COLD"
        )
    if trainer._l2_init_pairs:
        ok, why = trainer.load_l2_init_refs(side, allow_actor_only=allow_actor_only)
        out["l2_init"] = ok
        print(
            f"[optim] restored the ORIGINAL l2-init references from {path.name} ({why})"
            if ok else
            f"[optim] l2-init references NOT restored ({why}) — decay-to-init "
            "re-anchors at the loaded weights"
        )
    return out
