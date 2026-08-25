"""PolicyNetHost — T1 StrategyBackend serving a supervised GTO PolicyNet.

Same act / node_distribution surface as PpoSolverHost so Trainer swaps
without UX changes.

Badge honesty:
  - ``is_gto=True`` / label **"GTO AI"** only when checkpoint meta has
    ``source`` starting with ``rust_cfr`` **and** a recorded holdout probe pass.
  - Bootstrap / synthetic / unprobed PolicyNet → "Curriculum" or
    "Policy net (unvalidated)" — never silent GTO claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plo5bp.env import StepInfo
from plo5bp.gto.backend import NodeDist
from plo5bp.gto.policy_net import (
    POLICY_KIND,
    is_gto_checkpoint,
    is_validated_gto_meta,
    load_policy_checkpoint,
)
from plo5bp.network import ActorCritic, obs_adapter
from plo5bp.sizing import NLH_ANCHOR_SPEC, sizing_from_info


def _badge_label(meta: dict[str, Any] | None, *, validated: bool) -> tuple[str, str]:
    """Return (label, note) for coverage_badge."""
    source = str((meta or {}).get("source") or "")
    if validated:
        return (
            "GTO AI",
            "Supervised PolicyNet trained on native rust_cfr labels; "
            "holdout probe passed. Equilibrium estimate / population play "
            "— not multiway Nash.",
        )
    src_l = source.lower()
    if "bootstrap" in src_l or src_l.startswith("rule_"):
        return (
            "Curriculum",
            "Rule-bootstrap prior only — not solver GTO. "
            f"source={source or 'unknown'}",
        )
    if src_l.startswith("rust_cfr"):
        return (
            "Policy net (unvalidated)",
            "rust_cfr-trained weights without a recorded holdout probe pass. "
            "Run scripts/gto_probe.py before promoting.",
        )
    if source:
        return (
            "Policy net (unvalidated)",
            f"PolicyNet source={source}; not validated as GTO.",
        )
    return (
        "Policy net (unvalidated)",
        "Checkpoint missing validated GTO meta (source + probe).",
    )


@dataclass
class PolicyNetHost:
    """T1 host: supervised PolicyNet sample + score."""

    model: ActorCritic
    device: torch.device
    name: str = "policy_net"
    mode: str = "policy_net"
    coverage: str = "gto_estimate"
    is_gto: bool = False  # default honest; set from validated meta
    ckpt_path: str | None = None
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        from plo5bp.ui.trainer import compute_node_distribution

        self._compute_node_distribution = compute_node_distribution
        self._adapt = obs_adapter(self.model)
        if not hasattr(self.model, "anchor_spec"):
            self.model.anchor_spec = NLH_ANCHOR_SPEC  # type: ignore[attr-defined]
        # Reconcile is_gto with meta if meta present
        if self.meta is not None:
            self.is_gto = is_validated_gto_meta(self.meta)
            if self.is_gto:
                self.coverage = "gto_validated"
            else:
                self.coverage = "policy_estimate"

    def act(
        self,
        obs: np.ndarray,
        info: StepInfo,
        *,
        deterministic: bool = False,
        rng_seed: int | None = None,
    ) -> tuple[int, int]:
        if rng_seed is not None:
            torch.manual_seed(int(rng_seed) & 0x7FFFFFFFFFFFFFFF)
        with torch.no_grad():
            o = torch.from_numpy(self._adapt(obs)).unsqueeze(0).to(self.device)
            m = torch.from_numpy(info.gate_mask).unsqueeze(0).to(self.device)
            b = torch.from_numpy(sizing_from_info(info)[None, :]).to(self.device)
            out = self.model.act(o, m, b, deterministic=deterministic)
        return int(out.gate.item()), int(out.chips.item())

    def node_distribution(self, obs: np.ndarray, info: StepInfo) -> NodeDist:
        raw = self._compute_node_distribution(
            self.model, self.device, self._adapt(obs), info
        )
        nd = NodeDist.from_dict(raw, backend_name=self.name)
        nd.mode = self.mode
        nd.coverage = self.coverage
        return nd

    def supports(self, *, seats: int, street: int) -> bool:
        return 2 <= int(seats) <= 6 and 0 <= int(street) <= 3

    def coverage_badge(self) -> dict[str, Any]:
        validated = bool(self.is_gto) and is_validated_gto_meta(self.meta)
        # Force honest: never show GTO if meta fails even if is_gto was set True
        show_gto = validated
        label, note = _badge_label(self.meta, validated=show_gto)
        probe = (self.meta or {}).get("probe")
        return {
            "backend": self.name,
            "mode": self.mode,
            "label": label,
            "is_gto": show_gto,
            "seats": "2-6",
            "note": note,
            "ckpt": self.ckpt_path,
            "source": (self.meta or {}).get("source"),
            "n_train": (self.meta or {}).get("n_train"),
            "probe_passed": (
                probe.get("passed") if isinstance(probe, dict) else None
            ),
            "is_gto_validated": (self.meta or {}).get("is_gto_validated"),
        }

    def rebind(self, model: ActorCritic, device: torch.device | None = None) -> None:
        self.model = model
        if device is not None:
            self.device = device
        self._adapt = obs_adapter(self.model)


def load_policy_host(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> PolicyNetHost:
    """Load a PolicyNet checkpoint into a T1 host.

    ``is_gto`` is True only for validated rust_cfr + probe meta.
    """
    path = Path(path)
    model, meta = load_policy_checkpoint(path, device=device)
    # kind must be policy net family for this host
    kind_ok = meta.get("kind") == POLICY_KIND or bool(meta.get("is_gto"))
    if not kind_ok:
        # Still load if weights work — badge will be unvalidated
        pass
    validated = is_validated_gto_meta(meta)
    return PolicyNetHost(
        model=model,  # type: ignore[arg-type]
        device=torch.device(device),
        is_gto=validated,
        ckpt_path=str(path),
        meta=meta,
    )


def try_load_gto_host(
    path: Path | str | None,
    *,
    device: torch.device | str = "cpu",
) -> PolicyNetHost | None:
    """Return a PolicyNet host if path is a gto_policy_net artifact.

    Host may still have is_gto=False (unvalidated). Callers that need
    validated GTO only should check ``host.is_gto``.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    if not is_gto_checkpoint(p):
        return None
    return load_policy_host(p, device=device)
