"""Serving models for the study / trainer / home-games UI.

One place for: resolving each format's checkpoint, loading it ONCE (actor and
critic from the same read, PERF-023) with safe unpickling (SEC-023), the
observation-semantics bookkeeping, the per-format registry entries
(``FormatEntry``, BE-011), and live model management without a restart
(OPS-027 / OPS-022): reload a checkpoint from disk, promote ``<file>.new``
over it (keeping ``<file>.prev``), or roll back — every candidate is loaded
and smoke-tested on the side BEFORE the served entry is swapped, and the swap
is one dict assignment, so no request ever sees half a model.

The ``experimental`` format id is the admin-only CANDIDATE slot (FEAT-025 /
BE-012): ``PLO5BP_CHECKPOINT_CANDIDATE`` (or the legacy
``PLO5BP_CHECKPOINT_EXPERIMENTAL``) names a PLO5 checkpoint to try on real
Study spots and trainer hands before promoting it. Unset, the slot is not
offered at all (it used to serve the newest retired ``vMin1_*.pt`` by file
date — on production, a random placeholder).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypedDict

import torch
import torch.nn as nn

import plo5bp.encoding as _encoding  # module handle: OBS_SEMANTICS_REV is read late
from plo5bp.config import VARIANT_NLH, VARIANT_PLO5
from plo5bp.encoding import OBS_DIM, OBS_DIM_MINIMAL
from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.network import (
    ActorCritic,
    ActorCriticV4,
    ActorCriticV5,
    CentralCritic,
    build_actor_from_state_dict,
    build_critic_from_state_dict,
    obs_adapter,
)
from plo5bp.sizing import NLH_ANCHOR_SPEC, PLO_ANCHOR_SPEC
from plo5bp.ui.common import FORMAT_EXPERIMENTAL, env_flag

logger = logging.getLogger("plo5bp.ui")


class FormatEntry(TypedDict, total=False):
    """One served format (a plain dict at runtime — consumers index it)."""

    label: str
    model: nn.Module
    critic: CentralCritic | None
    adapter: Callable[[Any], Any]
    loaded: bool            # False = a random-init placeholder answers
    critic_loaded: bool
    engine_variant: str
    obs_rev: int
    obs_rev_mismatch: bool
    checkpoint: str | None  # the file served (None: none / placeholder)
    sha256: str | None
    mtime: float | None
    size: int | None
    loaded_at: float
    version: int            # bumps on every reload of this slot
    admin_only: bool        # never listed for users the format gate locks out
    available: bool         # offered in the format list at all
    critic_q: dict[str, Any] | None  # the critic's Q head, when it has one
    _ckpt_path: str


# --- Paths ---------------------------------------------------------------------------

#: format -> (env var, default path). The candidate slot has no default.
FORMAT_CKPTS: dict[str, tuple[str, str]] = {
    VARIANT_PLO5: ("PLO5BP_CHECKPOINT", "checkpoints/stub.pt"),
    VARIANT_NLH: ("PLO5BP_CHECKPOINT_NLH", "checkpoints/nlh_stub.pt"),
    FORMAT_EXPERIMENTAL: ("PLO5BP_CHECKPOINT_CANDIDATE", ""),
}
_LEGACY_CANDIDATE_ENV = "PLO5BP_CHECKPOINT_EXPERIMENTAL"


def format_ckpt_path(fmt: str) -> Path | None:
    """The checkpoint a format serves (None: the candidate slot is unset)."""
    env_key, default = FORMAT_CKPTS[fmt]
    override = os.environ.get(env_key, "").strip()
    if not override and fmt == FORMAT_EXPERIMENTAL:
        override = os.environ.get(_LEGACY_CANDIDATE_ENV, "").strip()
    if override:
        return Path(override)
    return Path(default) if default else None


def resolve_device() -> str:
    """Pick the inference device. PLO5BP_DEVICE=cuda promotes when
    available; otherwise log and fall back to CPU."""
    requested = os.environ.get("PLO5BP_DEVICE", "cpu").strip().lower() or "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("PLO5BP_DEVICE=cuda but cuda unavailable — falling back to cpu")
    return "cpu"


def random_init_model(variant: str) -> nn.Module:
    """Placeholder actor when no checkpoint exists for the format. The
    PLO fallback keeps the historical v1 128×2 shape; NLH needs a
    correctly-shaped v4 (995-dim obs, 12-anchor ladder) so the format is
    still explorable before the first promote — flagged un-loaded so the
    UI can badge the recommendations as untrained. The candidate slot's
    placeholder is a small minimal-obs mixture net (never offered)."""
    if variant == VARIANT_NLH:
        return ActorCriticV4(
            hidden_dim=128, obs_dim=OBS_DIM_NLH, num_layers=2, anchor_spec=NLH_ANCHOR_SPEC,
        )
    if variant == FORMAT_EXPERIMENTAL:
        return ActorCriticV5(
            hidden_dim=128, obs_dim=OBS_DIM_MINIMAL, num_layers=2, anchor_spec=PLO_ANCHOR_SPEC,
        )
    return ActorCritic(hidden_dim=128, num_layers=2)


# --- Reading checkpoints (SEC-023 / PERF-023) -------------------------------------------


class UnsafeCheckpoint(RuntimeError):
    """The file needs full (code-executing) unpickling."""


def _allow_numpy_globals() -> None:
    """numpy types some older checkpoints carry, allow-listed for the safe
    loader (they construct data, they cannot run code)."""
    try:
        import numpy as np

        extra: list[Any] = [np.dtype, np.ndarray]
        try:
            from numpy.core.multiarray import _reconstruct, scalar  # type: ignore
            extra += [_reconstruct, scalar]
        except Exception:  # noqa: BLE001 — numpy 2 moved them
            try:
                from numpy._core.multiarray import _reconstruct, scalar  # type: ignore
                extra += [_reconstruct, scalar]
            except Exception:  # noqa: BLE001
                pass
        for name in ("Float64DType", "Float32DType", "Int64DType", "Int32DType", "BoolDType"):
            t = getattr(getattr(np, "dtypes", None), name, None)
            if t is not None:
                extra.append(t)
        torch.serialization.add_safe_globals(extra)
    except Exception:  # noqa: BLE001 — best effort
        pass


_allow_numpy_globals()


def read_checkpoint(path: Path, *, trust: bool | None = None) -> Any:
    """``torch.load`` with ``weights_only=True``: a checkpoint is DATA, and a
    tampered or wrong file in ``checkpoints/`` must not run code as the
    service user. A file that only loads with full unpickling is refused
    (``UnsafeCheckpoint``) unless ``trust`` — by default the local build, or
    ``PLO5BP_TRUST_CHECKPOINTS=1`` — in which case it loads with a loud
    warning."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        message = str(e)
        if "weights_only" not in message and "Unsupported global" not in message:
            raise
        if trust is None:
            trust = env_flag("PLO5BP_TRUST_CHECKPOINTS") or not env_flag("PLO5BP_PUBLIC")
        if not trust:
            raise UnsafeCheckpoint(
                f"{path} needs full unpickling (it can run code): refused. Re-save it as"
                " plain tensors/dicts, or set PLO5BP_TRUST_CHECKPOINTS=1 if you trust it."
            ) from e
        logger.warning(
            "checkpoint %s is not loadable with weights_only=True — loading it with FULL"
            " unpickling because it is trusted (local build / PLO5BP_TRUST_CHECKPOINTS)", path,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def file_facts(path: Path) -> dict[str, Any]:
    """sha256 / size / mtime of a checkpoint (what the System panel shows)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    st = path.stat()
    return {"sha256": h.hexdigest(), "size": int(st.st_size), "mtime": float(st.st_mtime)}


# --- Observation-semantics revision -----------------------------------------------------
# (review 2026-09-20) The observation feature fixes from that review change
# VALUES, not the layout: same width, different numbers in the corrected
# dims. `plo5bp.encoding.OBS_SEMANTICS_REV` (env `PLO5BP_OBS_REV`; 2 = the
# corrected features, 1 = the exact pre-review values) says which semantics
# this PROCESS encodes; train.py stamps `obs_rev` into new checkpoints, and a
# checkpoint without the key was trained on rev 1. A model served on the
# other revision gets inputs it never saw, silently — so a mismatch is
# reported loudly and surfaced per format (and by /health), but never
# refused: the operator picks the revision, the UI keeps serving.

#: variant -> {"obs_rev": int, "obs_rev_mismatch": bool} for the checkpoint
#: `load_model` last loaded for that format.
OBS_REV_INFO: dict[str, dict[str, Any]] = {}
#: (checkpoint path, checkpoint rev, process rev) already warned about.
_OBS_REV_WARNED: set[tuple[str, int, int]] = set()


def process_obs_rev() -> int:
    """The obs-semantics revision this process encodes. Read late (and with a
    default) so it works before the encoder-side switch lands."""
    return int(getattr(_encoding, "OBS_SEMANTICS_REV", 2))


def checkpoint_obs_rev(ckpt: Any) -> int:
    raw = ckpt.get("obs_rev", 1) if isinstance(ckpt, dict) else 1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 1


def note_checkpoint_obs_rev(variant: str, ckpt: Any, ckpt_path: Path | None, loaded: bool) -> None:
    process_rev = process_obs_rev()
    if not loaded:
        # A random-init placeholder was trained on nothing: never a mismatch.
        OBS_REV_INFO[variant] = {"obs_rev": process_rev, "obs_rev_mismatch": False}
        return
    ckpt_rev = checkpoint_obs_rev(ckpt)
    mismatch = ckpt_rev != process_rev
    OBS_REV_INFO[variant] = {"obs_rev": ckpt_rev, "obs_rev_mismatch": mismatch}
    key = (str(ckpt_path), ckpt_rev, process_rev)
    if mismatch and key not in _OBS_REV_WARNED:
        _OBS_REV_WARNED.add(key)
        logger.warning(
            "OBS-REV MISMATCH: checkpoint %s (%s) was trained on observation "
            "semantics rev %d, but this process encodes rev %d — the model is "
            "being fed feature values it never saw. Still serving it; set "
            "PLO5BP_OBS_REV=%d and restart to serve it exactly as trained.",
            ckpt_path, variant, ckpt_rev, process_rev, ckpt_rev,
        )


def obs_rev_entry(variant: str) -> dict[str, Any]:
    """The `FORMATS` fields describing the checkpoint just loaded for
    ``variant`` (call right after `load_model`)."""
    return dict(
        OBS_REV_INFO.get(
            variant, {"obs_rev": process_obs_rev(), "obs_rev_mismatch": False}
        )
    )


# --- Actor / critic ---------------------------------------------------------------------------


def actor_from_checkpoint(ckpt: Any, variant: str, source: Any = "?") -> tuple[nn.Module, bool, int, int]:
    """(actor, loaded, hidden_dim, num_layers) from an already-read checkpoint.

    Dual path: v2+ anchor-head checkpoints carry 'anchor_head.weight', v1
    Beta-head ones 'raise_head.weight'; the trained obs width and anchor spec
    are sniffed from the state dict. Optional EMA serving
    (``PLO5BP_SERVE_EMA=1``): serve the slow EMA-of-past-iterates actor
    (``model_ema``) when present — a smoother, less exploitable policy, the
    same architecture; the critic is never EMA'd."""
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        cfg_block = ckpt.get("config", {}) or {}
        hidden_dim = int(cfg_block.get("hidden_dim", 128))
        num_layers = int(cfg_block.get("num_layers", 2))
        if os.environ.get("PLO5BP_SERVE_EMA") == "1":
            ema = ckpt.get("model_ema")
            if ema:
                state_dict = ema
                logger.info("serving EMA actor (PLO5BP_SERVE_EMA=1) for %s", variant)
            else:
                logger.info(
                    "PLO5BP_SERVE_EMA=1 but %s has no model_ema — serving last iterate", source,
                )
    else:
        state_dict = ckpt
        hidden_dim = 128
        num_layers = 2
    try:
        model = build_actor_from_state_dict(state_dict, hidden_dim, num_layers)
        return model, True, hidden_dim, num_layers
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "checkpoint %s is incompatible with current network (%s) — using random init",
            source, e,
        )
        return random_init_model(variant), False, hidden_dim, num_layers


#: Distributional-critic readout hyperparameters that are NOT sniffable from
#: the state dict (no parameters — pure forward-path semantics) but change
#: the served V / Q.
CRITIC_VALUE_KWARGS = ("value_support", "value_hlgauss_sigma")
CRITIC_Q_FLAGS = ("q_fold_zero", "q_base_raw")


def critic_value_kwargs(ckpt: Any) -> dict[str, Any]:
    """Keyword args for ``build_critic_from_state_dict`` taken from the
    checkpoint's ``config`` block: the value support / HL-Gauss sigma the run
    trained with, and its Q-head semantics flags (``q_fold_zero`` /
    ``q_base_raw`` — they change Q, which the trainer's instant EV loss
    reads). Absent keys keep the builder's defaults."""
    cfg_block = ckpt.get("config") if isinstance(ckpt, dict) else None
    if not isinstance(cfg_block, dict):
        return {}
    out: dict[str, Any] = {}
    for key in CRITIC_VALUE_KWARGS:
        value = cfg_block.get(key)
        if value is not None:
            out[key] = float(value)
    for key in CRITIC_Q_FLAGS:
        value = cfg_block.get(key)
        if value is not None:
            out[key] = bool(value)
    return out


def serve_obs_dim(variant: str, model: nn.Module | None = None) -> int:
    """The observation width a format's critic must take."""
    if variant == VARIANT_NLH:
        return OBS_DIM_NLH
    if variant == FORMAT_EXPERIMENTAL:
        # The candidate may be any PLO5 net: its critic pairs with ITS actor.
        width = getattr(model, "obs_dim", None)
        return int(width) if isinstance(width, int) else OBS_DIM
    return OBS_DIM


def critic_from_checkpoint(
    ckpt: Any, device: torch.device, variant: str, source: Any = "?",
    model: nn.Module | None = None,
) -> CentralCritic | None:
    """The centralized critic bundled in a checkpoint (v2+ only;
    ckpt['critic'] + head_version>=2), or None (v1 / random-init / missing /
    incompatible) — then the trainer review shows only the actor's own
    (blind) value estimate."""
    if not (
        isinstance(ckpt, dict)
        and int(ckpt.get("head_version", 1)) >= 2
        and "critic" in ckpt
    ):
        return None
    try:
        # Dims (obs width, hidden, residual depth) are sniffed from the
        # state dict — NLH critics are 995-wide, PLO 991+, and constructing
        # at the PLO default used to shape-fail every NLH load.
        critic = build_critic_from_state_dict(ckpt["critic"], **critic_value_kwargs(ckpt))
    except Exception as e:  # noqa: BLE001
        logger.warning("checkpoint %s critic incompatible (%s) — true-EV disabled", source, e)
        return None
    want = serve_obs_dim(variant, model)
    if critic.obs_dim != want:
        logger.warning(
            "checkpoint %s critic obs width %d != %s serve width %d — true-EV disabled",
            source, critic.obs_dim, variant, want,
        )
        return None
    critic.to(device).eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    logger.info(
        "loaded centralized critic (obs_dim=%d, hidden_dim=%d, device=%s)",
        critic.obs_dim, critic.value_head.in_features, device,
    )
    return critic


def critic_q_info(critic: CentralCritic | None, ckpt: Any) -> dict[str, Any] | None:
    """What the critic's dueling Q head can say (FEAT-023): None without one.
    ``pooled`` = one column for every raise size (only gate choices differ)."""
    q_actions = int(getattr(critic, "q_actions", 0) or 0)
    if critic is None or q_actions <= 0:
        return None
    cfg_block = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    trained = float((cfg_block or {}).get("q_aux_coef", 0.0) or 0.0) > 0.0
    return {"columns": q_actions, "pooled": q_actions == 3, "trained": trained}


# --- Entries --------------------------------------------------------------------------------

_LABELS = {
    VARIANT_PLO5: "PLO5 · Double-board bomb pot",
    VARIANT_NLH: "No-Limit Hold'em",
    FORMAT_EXPERIMENTAL: "Candidate model",
}
_ENGINE = {VARIANT_PLO5: VARIANT_PLO5, VARIANT_NLH: VARIANT_NLH, FORMAT_EXPERIMENTAL: VARIANT_PLO5}
_VERSIONS: dict[str, int] = {}
_VERSION_LOCK = threading.Lock()


def _next_version(fmt: str) -> int:
    with _VERSION_LOCK:
        _VERSIONS[fmt] = _VERSIONS.get(fmt, 0) + 1
        return _VERSIONS[fmt]


def load_model(variant: str = VARIANT_PLO5, path: Path | None = None) -> tuple[nn.Module, bool]:
    """Load a format's actor. Returns (model, loaded) — loaded=False means
    a random-init placeholder is being served."""
    model, loaded, _ckpt = _load_actor(variant, path)
    return model, loaded


def _load_actor(variant: str, path: Path | None) -> tuple[nn.Module, bool, Any]:
    device = resolve_device()
    ckpt_path = path if path is not None else format_ckpt_path(variant)
    if ckpt_path is None or not ckpt_path.exists():
        if ckpt_path is not None:
            logger.warning(
                "checkpoint %s not found — using random-init model (%s)", ckpt_path, variant,
            )
        note_checkpoint_obs_rev(variant, None, ckpt_path, loaded=False)
        return random_init_model(variant).to(device).eval(), False, None
    try:
        ckpt = read_checkpoint(ckpt_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("failed to load %s (%s) — using random init", ckpt_path, e)
        note_checkpoint_obs_rev(variant, None, ckpt_path, loaded=False)
        return random_init_model(variant).to(device).eval(), False, None
    model, loaded, hidden_dim, num_layers = actor_from_checkpoint(ckpt, variant, ckpt_path)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(
        "loaded checkpoint %s (%s, hidden_dim=%d, num_layers=%d, device=%s)",
        ckpt_path, type(model).__name__, hidden_dim, num_layers, device,
    )
    note_checkpoint_obs_rev(variant, ckpt, ckpt_path, loaded)
    return model, loaded, ckpt


def build_entry(fmt: str, path: Path | None = None) -> FormatEntry:
    """A complete, NOT-yet-served entry for ``fmt``: actor + critic from ONE
    read of the checkpoint, its facts (sha256, size, mtime), obs rev."""
    ckpt_path = path if path is not None else format_ckpt_path(fmt)
    model, loaded, ckpt = _load_actor(fmt, ckpt_path)
    device = next(model.parameters()).device
    critic = critic_from_checkpoint(ckpt, device, fmt, ckpt_path, model) if loaded else None
    facts: dict[str, Any] = {"sha256": None, "size": None, "mtime": None}
    if loaded and ckpt_path is not None:
        try:
            facts = file_facts(ckpt_path)
        except OSError:
            pass
    name = ckpt_path.name if ckpt_path is not None else None
    label = _LABELS.get(fmt, fmt)
    if fmt == FORMAT_EXPERIMENTAL and loaded and ckpt_path is not None:
        label = f"Candidate · {ckpt_path.stem}"
    entry: FormatEntry = {
        "label": label,
        "model": model,
        "critic": critic,
        "adapter": obs_adapter(model),
        "loaded": loaded,
        "critic_loaded": critic is not None,
        "engine_variant": _ENGINE.get(fmt, fmt),
        "checkpoint": name if loaded else None,
        "sha256": facts["sha256"],
        "mtime": facts["mtime"],
        "size": facts["size"],
        "loaded_at": time.time(),
        "version": _next_version(fmt),
        "admin_only": fmt == FORMAT_EXPERIMENTAL,
        "available": fmt != FORMAT_EXPERIMENTAL or loaded,
        "critic_q": critic_q_info(critic, ckpt) if loaded else None,
        "_ckpt_path": str(ckpt_path) if ckpt_path is not None else "",
        **obs_rev_entry(fmt),
    }
    return entry


def entry_summary(fmt: str, entry: dict[str, Any]) -> dict[str, Any]:
    """The JSON-safe facts about a served format (health + admin panel)."""
    return {
        "format": fmt,
        "label": entry.get("label"),
        "model_loaded": bool(entry.get("loaded")),
        "critic_loaded": bool(entry.get("critic_loaded", entry.get("critic") is not None)),
        "obs_rev": entry.get("obs_rev"),
        "obs_rev_mismatch": bool(entry.get("obs_rev_mismatch", False)),
        "checkpoint": entry.get("checkpoint"),
        "sha256": entry.get("sha256"),
        "size": entry.get("size"),
        "mtime": entry.get("mtime"),
        "loaded_at": entry.get("loaded_at"),
        "version": entry.get("version"),
        "available": bool(entry.get("available", True)),
        "admin_only": bool(entry.get("admin_only", False)),
        "model_class": type(entry.get("model")).__name__ if entry.get("model") is not None else None,
        "critic_q": entry.get("critic_q"),
    }


def smoke_test(fmt: str, entry: dict[str, Any]) -> None:
    """One real decision node through the entry's actor (and critic): the
    shapes, the obs adapter and the heads all work before anyone is served
    by it. Raises on any failure."""
    from plo5bp.config import GameConfig
    from plo5bp.env import BombPotEnv
    from plo5bp.sizing import sizing_from_info

    engine = entry.get("engine_variant", fmt)
    cfg = GameConfig.nlh_default() if engine == VARIANT_NLH else GameConfig()
    env = BombPotEnv(cfg)
    obs, info = env.reset(12345, 0)
    model = entry["model"]
    device = next(model.parameters()).device
    x = torch.from_numpy(entry["adapter"](obs)).unsqueeze(0).to(device)
    gm = torch.from_numpy(info.gate_mask).unsqueeze(0).to(device)
    sizing = torch.from_numpy(sizing_from_info(info)[None, :]).to(device)
    with torch.inference_mode():
        out = model.act(x, gm, sizing, deterministic=True)
    if not torch.isfinite(out.chips.float()).all():
        raise RuntimeError("the model produced a non-finite action")
    critic = entry.get("critic")
    if critic is not None:
        from plo5bp.rollout import _critic_values, _rotate_opp_holes
        import numpy as np

        holes = np.asarray(env.all_hole_cards(), dtype=np.uint8)
        opp = _rotate_opp_holes(holes, int(info.actor))[None]
        v = _critic_values(critic, device, np.asarray(entry["adapter"](obs), dtype=np.float32)[None], opp)
        if not np.isfinite(v).all():
            raise RuntimeError("the critic produced a non-finite value")


# --- Live model management (OPS-027 / OPS-022) ---------------------------------------------------


class ModelAdmin:
    """Reload / promote / roll back one format's checkpoint in a running
    server. ``formats`` is the live registry dict (a swap is one assignment
    into it); the optional ``on_swap(fmt, entry)`` hears about every swap.
    (The server's model names read the registry when they are used, so it
    passes none — BE-007.)"""

    def __init__(
        self,
        formats: dict[str, Any],
        on_swap: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.formats = formats
        self.on_swap = on_swap
        self._lock = threading.Lock()

    def _path(self, fmt: str) -> Path:
        if fmt not in self.formats:
            raise ValueError(f"unknown format {fmt!r}")
        path = format_ckpt_path(fmt)
        if path is None:
            raise ValueError(
                f"{fmt}: no checkpoint configured (set {FORMAT_CKPTS[fmt][0]})"
            )
        return path

    def _verified(self, fmt: str, path: Path) -> FormatEntry:
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        entry = build_entry(fmt, path)
        if not entry["loaded"]:
            raise RuntimeError(f"{path.name} did not load as a {fmt} model (see the log)")
        smoke_test(fmt, entry)
        return entry

    def _install(self, fmt: str, entry: FormatEntry, served_path: Path) -> dict[str, Any]:
        entry["_ckpt_path"] = str(served_path)
        entry["checkpoint"] = served_path.name
        if fmt == FORMAT_EXPERIMENTAL:
            entry["label"] = f"Candidate · {served_path.stem}"
        self.formats[fmt] = entry  # one assignment: readers see old or new, never a mix
        if self.on_swap is not None:
            self.on_swap(fmt, entry)
        logger.info(
            "format %s now serves %s (sha256 %s, version %s)",
            fmt, served_path, (entry.get("sha256") or "")[:12], entry.get("version"),
        )
        return entry_summary(fmt, entry)

    def __call__(self, action: str, fmt: str) -> dict[str, Any]:
        with self._lock:
            path = self._path(fmt)
            if action == "reload":
                return self._install(fmt, self._verified(fmt, path), path)
            if action == "promote":
                new = path.with_name(path.name + ".new")
                entry = self._verified(fmt, new)
                prev = path.with_name(path.name + ".prev")
                if path.exists():  # keep the outgoing file (a COPY: the live
                    tmp = path.with_name(path.name + ".prev.tmp")  # path never disappears)
                    shutil.copy2(path, tmp)
                    os.replace(tmp, prev)
                os.replace(new, path)
                return self._install(fmt, entry, path)
            if action == "rollback":
                prev = path.with_name(path.name + ".prev")
                entry = self._verified(fmt, prev)
                if path.exists():  # swap: rolling back twice returns here
                    tmp = path.with_name(path.name + ".swap.tmp")
                    shutil.copy2(path, tmp)
                    os.replace(prev, path)
                    os.replace(tmp, prev)
                else:
                    shutil.copy2(prev, path)
                return self._install(fmt, entry, path)
            raise ValueError("action must be reload|promote|rollback")
