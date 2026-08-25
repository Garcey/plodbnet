"""StrategyBackend: single host for Trainer + Study + Ranges.

Trainer today hardwires PPO via ``model_policy`` / ``compute_node_distribution``.
This module abstracts that surface so T1 can swap in a GTO PolicyNetHost
without rewriting the UX loop.

Adapters:
  PpoSolverHost   — current PPO ActorCritic (T0; prove the seam)
  PolicyNetHost   — supervised π* (T1; Phase 1–2a)
  StreetCacheHost — Mode 2 then sample (T3; Phase 4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from plo5bp.env import StepInfo
from plo5bp.eval import model_policy
from plo5bp.network import ActorCritic


@dataclass
class NodeDist:
    """Policy mass + recommendation at one decision node.

    Mirrors the dict shape of ``compute_node_distribution`` so existing
    scorers (``score_move`` / ``score_move_v2``) and UI payloads keep working.
    Extra fields (mode, coverage, think_ms) are GTO-host honesty metadata.
    """

    head_version: int
    gate_probs: list[float]
    rec_gate: int
    rec_chips: int
    value_bb: float
    min_chips: int
    max_chips: int
    # v2+ anchor ladder
    anchor_probs: list[float] | None = None
    anchor_chips: list[int] | None = None
    anchor_legal: list[bool] | None = None
    anchor_lo: list[int] | None = None
    anchor_hi: list[int] | None = None
    refine_ok: list[bool] | None = None
    refine_params: list[list[float]] | None = None
    rec_anchor: int | None = None
    pot_ref_chips: int | None = None
    mixture: dict[str, Any] | None = None
    # v1 Beta head
    alpha: float | None = None
    beta: float | None = None
    # Host metadata (Trainer badges / Study deep-solve UI)
    mode: str = "ppo"
    coverage: str = "self-play"
    think_ms: float = 0.0
    backend_name: str = "ppo"

    def as_dict(self) -> dict[str, Any]:
        """Dict compatible with trainer scoring + review payloads."""
        d: dict[str, Any] = {
            "head_version": self.head_version,
            "gate_probs": list(self.gate_probs),
            "rec_gate": int(self.rec_gate),
            "rec_chips": int(self.rec_chips),
            "value_bb": float(self.value_bb),
            "min_chips": int(self.min_chips),
            "max_chips": int(self.max_chips),
            "mode": self.mode,
            "coverage": self.coverage,
            "think_ms": float(self.think_ms),
            "backend_name": self.backend_name,
        }
        if self.head_version >= 2:
            d["anchor_probs"] = list(self.anchor_probs or [])
            d["anchor_chips"] = list(self.anchor_chips or [])
            d["anchor_legal"] = list(self.anchor_legal or [])
            d["anchor_lo"] = list(self.anchor_lo or [])
            d["anchor_hi"] = list(self.anchor_hi or [])
            d["refine_ok"] = list(self.refine_ok or [])
            d["refine_params"] = list(self.refine_params or [])
            d["rec_anchor"] = self.rec_anchor
            d["pot_ref_chips"] = int(self.pot_ref_chips or 0)
            d["mixture"] = self.mixture
        else:
            d["alpha"] = float(self.alpha if self.alpha is not None else 1.0)
            d["beta"] = float(self.beta if self.beta is not None else 1.0)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, backend_name: str = "ppo") -> "NodeDist":
        hv = int(d.get("head_version", 1))
        return cls(
            head_version=hv,
            gate_probs=list(d["gate_probs"]),
            rec_gate=int(d["rec_gate"]),
            rec_chips=int(d["rec_chips"]),
            value_bb=float(d["value_bb"]),
            min_chips=int(d["min_chips"]),
            max_chips=int(d["max_chips"]),
            anchor_probs=d.get("anchor_probs"),
            anchor_chips=d.get("anchor_chips"),
            anchor_legal=d.get("anchor_legal"),
            anchor_lo=d.get("anchor_lo"),
            anchor_hi=d.get("anchor_hi"),
            refine_ok=d.get("refine_ok"),
            refine_params=d.get("refine_params"),
            rec_anchor=d.get("rec_anchor"),
            pot_ref_chips=d.get("pot_ref_chips"),
            mixture=d.get("mixture"),
            alpha=d.get("alpha"),
            beta=d.get("beta"),
            mode=str(d.get("mode", "ppo")),
            coverage=str(d.get("coverage", "self-play")),
            think_ms=float(d.get("think_ms", 0.0)),
            backend_name=str(d.get("backend_name", backend_name)),
        )


@runtime_checkable
class StrategyBackend(Protocol):
    """Host contract for Trainer opp acts, hero score, Study, Ranges."""

    name: str
    mode: str  # "ppo" | "policy_net" | "river_exact" | "street_cache" | ...

    def act(
        self,
        obs: np.ndarray,
        info: StepInfo,
        *,
        deterministic: bool = False,
        rng_seed: int | None = None,
    ) -> tuple[int, int]:
        """Sample or argmax (gate, chips). rng_seed re-seeds torch when set."""
        ...

    def node_distribution(self, obs: np.ndarray, info: StepInfo) -> NodeDist:
        """Full policy mass + recommendation at this node."""
        ...

    def supports(self, *, seats: int, street: int) -> bool:
        """Whether this host can serve the requested table/street."""
        ...

    def coverage_badge(self) -> dict[str, Any]:
        """Honest UI badge payload (never silent PPO-as-GTO)."""
        ...


@dataclass
class PpoSolverHost:
    """T0 host: existing PPO ``ActorCritic`` via trainer helpers.

    Proves the StrategyBackend seam without changing play quality.
    After T1 ships this remains the fallback ("Self-play (untrained GTO)").
    """

    model: ActorCritic
    device: torch.device
    name: str = "ppo"
    mode: str = "ppo"
    coverage: str = "self-play"

    def __post_init__(self) -> None:
        # Lazy import avoids circular import at package load
        # (trainer imports backend; backend's node_distribution uses trainer helpers).
        from plo5bp.ui.trainer import compute_node_distribution

        self._compute_node_distribution = compute_node_distribution
        self._policy_stochastic = model_policy(self.model, deterministic=False)
        self._policy_deterministic = model_policy(self.model, deterministic=True)

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
        pol = self._policy_deterministic if deterministic else self._policy_stochastic
        actor = int(info.actor) if info.actor is not None else 0
        return pol(obs, actor, info)

    def node_distribution(self, obs: np.ndarray, info: StepInfo) -> NodeDist:
        raw = self._compute_node_distribution(self.model, self.device, obs, info)
        nd = NodeDist.from_dict(raw, backend_name=self.name)
        nd.mode = self.mode
        nd.coverage = self.coverage
        return nd

    def supports(self, *, seats: int, street: int) -> bool:
        return 2 <= int(seats) <= 6 and 0 <= int(street) <= 3

    def coverage_badge(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "mode": self.mode,
            "label": "Self-play (untrained GTO)",
            "is_gto": False,
            "seats": "2-6",
            "note": "PPO self-play frequencies — not equilibrium labels",
        }

    def rebind(self, model: ActorCritic, device: torch.device | None = None) -> None:
        """Swap the underlying checkpoint (format switch / reload)."""
        self.model = model
        if device is not None:
            self.device = device
        self._policy_stochastic = model_policy(self.model, deterministic=False)
        self._policy_deterministic = model_policy(self.model, deterministic=True)


def make_ppo_host(model: ActorCritic, device: torch.device) -> PpoSolverHost:
    return PpoSolverHost(model=model, device=device)
