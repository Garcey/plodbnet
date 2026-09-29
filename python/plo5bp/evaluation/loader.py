"""Load checkpoints for evaluation: the one place every tool rebuilds an actor
(or critic) from a file."""

from __future__ import annotations

from pathlib import Path

import torch

from plo5bp import encoding as _encoding
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.network import (
    build_actor_from_checkpoint,
    build_critic_from_checkpoint,
    state_dict_actor_size,
    state_dict_obs_dim,
)
from plo5bp.train.checkpoint import checkpoint_schema


class ObsRevMismatch(ValueError):
    """A checkpoint trained on one observation-semantics revision would be fed
    another: same widths, different feature values -- the results would be
    silently wrong."""


def load_checkpoint(path: "str | Path") -> dict:
    """The checkpoint dict, on the CPU."""
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_meta(ckpt: dict, path: "str | Path | None" = None) -> dict:
    """What a checkpoint is, from the checkpoint itself: the actor's size (read
    off the weights -- never a default), observation layout and revision
    (unstamped = rev 1, trained before the 2026-09-20 feature fixes), update,
    variant, schema, whether it carries an EMA actor."""
    sd = ckpt.get("model", ckpt)
    hid, nl = state_dict_actor_size(sd)
    width = state_dict_obs_dim(sd)
    cfg = ckpt.get("config") or {}
    return {
        "path": None if path is None else str(path),
        "hidden_dim": hid,
        "num_layers": nl,
        "obs_dim": width,
        "obs_mode": "minimal" if width == OBS_DIM_MINIMAL else str(cfg.get("obs_mode", "full")),
        "obs_rev": int(ckpt.get("obs_rev", 1)),
        "obs_rev_stamped": "obs_rev" in ckpt,
        "update": ckpt.get("update_counter", ckpt.get("update")),
        "variant": ckpt.get("variant", "plo5_double_bomb"),
        "head_version": int(ckpt.get("head_version", 1)),
        "schema": checkpoint_schema(ckpt),
        "has_ema": bool(ckpt.get("model_ema")),
    }


def check_obs_rev(meta: dict) -> None:
    """Refuse a checkpoint whose training revision is not this process's."""
    cur = int(_encoding.OBS_SEMANTICS_REV)
    if int(meta["obs_rev"]) != cur:
        raise ObsRevMismatch(
            f"{meta['path']}: trained on obs rev {meta['obs_rev']}"
            + ("" if meta["obs_rev_stamped"] else " (unstamped = rev 1)")
            + f", this process encodes obs rev {cur}. Run with "
            f"PLO5BP_OBS_REV={meta['obs_rev']} -- or compare different revisions "
            "with scripts/h2h_cross.py."
        )


def load_actor(
    path_or_ckpt: "str | Path | dict",
    device: "str | torch.device" = "cpu",
    ema: bool = False,
    check_rev: bool = True,
) -> "tuple[torch.nn.Module, dict]":
    """(actor in eval mode, no grads, on `device`; its `checkpoint_meta`).
    `ema=True` plays the KL-anchor EMA actor when the checkpoint has one
    (meta["ema"] says which was loaded). `check_rev` refuses an obs-revision
    mismatch (ML-020)."""
    if isinstance(path_or_ckpt, dict):
        ckpt, path = path_or_ckpt, None
    else:
        ckpt, path = load_checkpoint(path_or_ckpt), path_or_ckpt
    meta = checkpoint_meta(ckpt, path)
    if check_rev:
        check_obs_rev(meta)
    use_ema = bool(ema) and meta["has_ema"]
    actor = build_actor_from_checkpoint(ckpt, ema=use_ema).to(device).eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    meta["ema"] = use_ema
    if ema and not use_ema:
        print(f"[load_actor] {path}: no EMA actor in this checkpoint -- the last iterate plays")
    return actor, meta


def load_critic(
    path_or_ckpt: "str | Path | dict", device: "str | torch.device" = "cpu"
):
    """The checkpoint's centralized critic (eval mode, no grads), built with
    the Q-surface flags and value support it trained with."""
    ckpt = path_or_ckpt if isinstance(path_or_ckpt, dict) else load_checkpoint(path_or_ckpt)
    critic = build_critic_from_checkpoint(ckpt).to(device).eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    return critic
