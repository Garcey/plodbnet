"""GTO PolicyNet — supervised π* over the NLH gate × anchor menu.

Architecture is the v2 torso + flat anchor categorical (same act path as
ActorCriticV2) so Trainer / Study / Ranges reuse existing scoring without
a second code path. Checkpoints stamp ``kind=gto_policy_net`` for the
host load path. **Badge "GTO AI"** requires validated meta (native
``rust_cfr`` source + holdout probe pass) — see :func:`is_validated_gto_meta`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.network import ActorCriticV2, build_actor_from_state_dict, model_class_for_state_dict
from plo5bp.sizing import NLH_ANCHOR_SPEC

POLICY_KIND = "gto_policy_net"
DEFAULT_HIDDEN = 512
DEFAULT_LAYERS = 2


def build_policy_net(
    *,
    hidden_dim: int = DEFAULT_HIDDEN,
    num_layers: int = DEFAULT_LAYERS,
    obs_dim: int = OBS_DIM_NLH,
    torso_layernorm: bool = False,
) -> ActorCriticV2:
    """Construct an NLH-sized PolicyNet (ActorCriticV2 + NLH ladder)."""
    return ActorCriticV2(
        hidden_dim=hidden_dim,
        obs_dim=obs_dim,
        num_layers=num_layers,
        anchor_spec=NLH_ANCHOR_SPEC,
        torso_layernorm=torso_layernorm,
    )


def save_policy_checkpoint(
    path: Path | str,
    model: nn.Module,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """Save PolicyNet weights + meta.

    Always stamps ``kind=gto_policy_net``. ``is_gto`` in the file means
    "this is a PolicyNet artifact" (loadable by the host), **not** that
    the UI may show "GTO AI". Serving uses :func:`is_validated_gto_meta`
    (``rust_cfr`` source + probe.passed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first = model.torso[0]
    lin = first[0] if isinstance(first, nn.Sequential) else first
    payload: dict[str, Any] = {
        "kind": POLICY_KIND,
        "is_gto": True,  # PolicyNet family stamp (not badge validation)
        "is_gto_validated": False,
        "model": model.state_dict(),
        "obs_dim": int(lin.in_features),
        "hidden_dim": int(lin.out_features),
        "num_layers": _infer_num_layers(model),
        "anchor_spec": getattr(
            getattr(model, "anchor_spec", None), "name", NLH_ANCHOR_SPEC.name
        ),
        "head_version": int(getattr(model, "head_version", 2)),
        "root": "clubgg_5_10_5",
    }
    if meta:
        # meta may set source / n_train / probe / is_gto_validated
        payload.update(meta)
        payload["kind"] = POLICY_KIND  # never let meta clobber kind
    # Derive validated flag (inline — avoid circular import with probe.py)
    payload["is_gto_validated"] = _meta_claims_gto(payload)
    torch.save(payload, path)


def source_is_gto_teacher(source: str | None) -> bool:
    """True when labels came from the native rust_cfr solver.

    Bootstrap / smoke / rule priors are never a GTO teacher.
    """
    if not source:
        return False
    s = str(source).lower().strip()
    if "bootstrap" in s or "smoke" in s or s.startswith("rule_"):
        return False
    return s.startswith("rust_cfr")


def _meta_claims_gto(meta: dict[str, Any]) -> bool:
    """rust_cfr source + probe.passed (or explicit is_gto_validated)."""
    if not source_is_gto_teacher(str(meta.get("source") or "")):
        return False
    probe = meta.get("probe")
    if isinstance(probe, dict) and probe.get("passed") is True:
        return True
    return bool(meta.get("is_gto_validated"))


def _infer_num_layers(model: nn.Module) -> int:
    """Best-effort layer count from torso structure (for rebuild)."""
    t = model.torso
    # Residual: Sequential(input, block, block, ...) → 1 + n_blocks
    if len(t) >= 1 and isinstance(t[0], nn.Sequential):
        return len(t)  # input + residual blocks
    # Flat: Linear, ReLU, Linear, ReLU, ... → n_linear
    n_lin = sum(1 for m in t if isinstance(m, nn.Linear))
    return max(1, n_lin)


def load_policy_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
    map_location: str | None = None,
) -> tuple[ActorCriticV2, dict[str, Any]]:
    """Load a GTO PolicyNet checkpoint. Raises if not a policy_net kind.

    Also accepts a raw PPO actor state_dict nested under ``model`` when
    ``kind`` is missing but the head is v2+ with NLH anchor count — used
    only for teacher loading, not for serving as GTO.
    """
    path = Path(path)
    ckpt = torch.load(
        path,
        map_location=map_location or "cpu",
        weights_only=False,
    )
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: expected dict checkpoint")

    kind = ckpt.get("kind")
    state = ckpt.get("model") or ckpt.get("actor") or ckpt
    if not isinstance(state, dict):
        raise ValueError(f"{path}: no model state_dict")

    # Nested training checkpoints (train.py) store actor under "model"
    if "torso.0.weight" not in state and "torso.0.0.weight" not in state:
        # Maybe full train ckpt: try nested
        for key in ("model", "actor", "policy"):
            if key in ckpt and isinstance(ckpt[key], dict):
                inner = ckpt[key]
                if "torso.0.weight" in inner or "torso.0.0.weight" in inner:
                    state = inner
                    break

    hidden = int(ckpt.get("hidden_dim") or _hidden_from_state(state))
    layers = int(ckpt.get("num_layers") or 2)
    obs_dim = int(ckpt.get("obs_dim") or _obs_from_state(state))

    if kind == POLICY_KIND or ckpt.get("is_gto"):
        model = build_policy_net(
            hidden_dim=hidden,
            num_layers=layers,
            obs_dim=obs_dim,
            torso_layernorm=any(".norm.weight" in k for k in state),
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"{path}: missing keys {missing[:5]}...")
    else:
        # PPO teacher load path
        model = build_actor_from_state_dict(state, hidden, layers)  # type: ignore[assignment]
        model.load_state_dict(state, strict=False)

    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k not in ("model", "actor", "policy")}
    meta["path"] = str(path)
    meta["kind"] = kind or "unknown"
    return model, meta  # type: ignore[return-value]


def _hidden_from_state(state: dict) -> int:
    w = state.get("torso.0.weight")
    if w is None:
        w = state["torso.0.0.weight"]
    return int(w.shape[0])


def _obs_from_state(state: dict) -> int:
    w = state.get("torso.0.weight")
    if w is None:
        w = state["torso.0.0.weight"]
    return int(w.shape[1])


def is_gto_checkpoint(path: Path | str) -> bool:
    """True if file is a PolicyNet artifact (loadable as T1 host).

    Does **not** mean the badge may say GTO AI — use
    :func:`is_validated_gto_checkpoint` for that.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(ckpt, dict):
        return False
    return ckpt.get("kind") == POLICY_KIND or bool(ckpt.get("is_gto"))


def is_validated_gto_meta(meta: dict[str, Any] | None) -> bool:
    """Badge-level: rust_cfr source + recorded probe pass."""
    if not meta:
        return False
    return _meta_claims_gto(dict(meta))


def is_validated_gto_checkpoint(path: Path | str) -> bool:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(ckpt, dict):
        return False
    return is_validated_gto_meta(ckpt)
