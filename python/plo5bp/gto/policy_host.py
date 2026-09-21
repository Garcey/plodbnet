"""PolicyNetHost — T1 StrategyBackend serving a supervised GTO PolicyNet.

Same act / node_distribution surface as PpoSolverHost so Trainer swaps
without UX changes.

Badge honesty:
  - ``is_gto=True`` / label **"GTO AI"** only when checkpoint meta has
    ``source`` starting with ``rust_cfr``, record-derived label provenance
    **and** a recorded holdout probe pass.
  - Bootstrap / synthetic / unprobed PolicyNet → "Curriculum" or
    "Policy net (unvalidated)" — never silent GTO claim.

Serve == train (review 2026-09-20):

  - **D3** The net was trained on the solver's CANONICAL obs (synthetic
    street root: current-street history only, ``total_commit ==
    street_commit``, no blind flags). A played hand's obs carries prior-street
    history / hand-total commits / blind flags whose first-layer weights never
    saw a gradient, so postflop nodes are re-encoded through
    :func:`plo5bp.gto.obs_from_label.canonical_serve_obs` before the forward.
  - **D5** The refine head is never trained, so a sampled / mean refinement
    slid 53% of interior-anchor jams short of all-in. The host serves the
    anchor's GRID chips (every anchor is an atom) and snaps any size within
    :data:`JAM_DUST_BB` of the stack to exactly ``max_raise``.
  - **F11** ``supports()`` answers from the checkpoint's recorded training
    coverage instead of claiming every seat count and street.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plo5bp.actions import GATE_RAISE
from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.env import StepInfo
from plo5bp.gto.backend import NodeDist
from plo5bp.gto.obs_from_label import canonical_serve_obs
from plo5bp.gto.obs_rev import obs_rev_mismatch, stamped_obs_rev
from plo5bp.gto.policy_net import (
    POLICY_KIND,
    is_gto_checkpoint,
    is_validated_gto_meta,
    load_policy_checkpoint,
    source_is_gto_teacher,
)
from plo5bp.gto.roots import CLUBGG_NLH_ROOT
from plo5bp.network import ActorCritic, obs_adapter
from plo5bp.sizing import NLH_ANCHOR_SPEC, anchor_grid_np, sizing_from_info

logger = logging.getLogger(__name__)

# Leaving less than this behind is not a bet size, it is a jam.
JAM_DUST_BB = 0.5
# ``obs[_POT_BB_DIM] == pot / bb`` in the NLH layout (encoding_nlh scalars).
_POT_BB_DIM = 132
# v1 teacher scope — what a checkpoint WITHOUT recorded coverage may claim.
_LEGACY_COVERAGE = {"streets": [3], "seats": [2], "streets_exact": [3]}


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
            "rust_cfr-trained weights without a recorded holdout probe pass "
            "and record-derived label provenance. "
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


def _range_label(values: list[int]) -> str:
    if not values:
        return "?"
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f"{lo}-{hi}"


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
    # Live table big blind (chips). Optional: when unset it is recovered from
    # the (obs, raw_obs) pair the caller passes — see :meth:`_bb_chips`.
    bb_chips: int | None = None

    def __post_init__(self) -> None:
        from plo5bp.ui.trainer import compute_node_distribution

        self._compute_node_distribution = compute_node_distribution
        self._adapt = obs_adapter(self.model)
        if not hasattr(self.model, "anchor_spec"):
            self.model.anchor_spec = NLH_ANCHOR_SPEC  # type: ignore[attr-defined]
        # Reconcile is_gto with meta if meta present
        if self.meta is not None:
            stale = obs_rev_mismatch(self.meta)
            if stale is not None:
                logger.warning(
                    "PolicyNetHost %s: %s", self.ckpt_path or self.name, stale
                )
            self.is_gto = is_validated_gto_meta(self.meta)
            if self.is_gto:
                self.coverage = "gto_validated"
            else:
                self.coverage = "policy_estimate"

    # -- serve-side canonicalization (review 2026-09-20 D3) -------------------

    def _bb_chips(self, obs: np.ndarray, raw: dict[str, Any]) -> int:
        """Big blind of the LIVE table.

        The host is handed ``(obs, info)`` but no ``GameConfig``. The live obs
        encodes ``pot / bb`` exactly, so bb is recovered from it; an explicit
        ``bb_chips`` wins, and the locked ClubGG stake is the last resort.
        """
        if self.bb_chips:
            return int(self.bb_chips)
        try:
            flat = np.asarray(obs).reshape(-1)
            pot = float(raw.get("pot") or 0.0)
            # Only the NLH layout puts pot/bb at this offset.
            pot_bb = float(flat[_POT_BB_DIM]) if flat.shape[0] == OBS_DIM_NLH else 0.0
            if pot > 0.0 and pot_bb > 0.0 and np.isfinite(pot_bb):
                bb = int(round(pot / pot_bb))
                if bb > 0:
                    return bb
        except (IndexError, TypeError, ValueError):
            pass
        return int(CLUBGG_NLH_ROOT.bb)

    @property
    def serves_canonical_obs(self) -> bool:
        """Was this net trained on the solver-root CANONICAL obs?

        New checkpoints record ``obs_forms`` (derived from the rows). Older
        ones are inferred from the source: rust_cfr label rows were always
        canonical, bootstrap / distill rows were live env obs. A host built
        without meta is assumed to hold a label-trained net.
        """
        if self.meta is None:
            return True
        forms = self.meta.get("obs_forms")
        if isinstance(forms, dict) and forms:
            return "canonical" in forms
        return source_is_gto_teacher(str(self.meta.get("source") or ""))

    def canonical_obs(self, obs: np.ndarray, info: StepInfo) -> np.ndarray:
        """The obs the net was TRAINED on for this node.

        Postflop: the label's canonical form rebuilt from the live raw obs.
        Preflop / no raw obs: the live obs unchanged (preflop labels are a
        hand-built synthetic dict with no exact serve-side twin). Nets trained
        on live obs (legacy bootstrap / distill) keep the live obs everywhere.
        """
        if not self.serves_canonical_obs:
            return obs
        raw = info.raw_obs or {}
        try:
            canon = canonical_serve_obs(raw, bb=self._bb_chips(obs, raw))
        except (KeyError, TypeError, ValueError, IndexError) as e:
            # Never crash the UI over it — but never hide it either: the net is
            # now being fed an obs distribution it was not trained on.
            if not getattr(self, "_warned_canon", False):
                self._warned_canon = True
                logger.warning(
                    "PolicyNetHost: could not canonicalize the live obs (%s: %s) "
                    "— serving the raw obs for this node",
                    type(e).__name__,
                    e,
                )
            canon = None
        return obs if canon is None else canon

    # -- sizing (review 2026-09-20 D5) ----------------------------------------

    def _dust_chips(self, obs: np.ndarray, info: StepInfo) -> int:
        return max(1, int(JAM_DUST_BB * self._bb_chips(obs, info.raw_obs or {})))

    def _anchor_chips(self, anchor: int, obs: np.ndarray, info: StepInfo) -> int:
        """Grid chips of ``anchor`` — never the (untrained) refinement — with
        anything within ``dust`` of the stack snapped to exactly ``max_raise``."""
        mn, mx, pot, tc = (int(x) for x in sizing_from_info(info))
        spec = getattr(self.model, "anchor_spec", NLH_ANCHOR_SPEC)
        chips = int(anchor_grid_np(mn, mx, pot, tc, spec).chips[int(anchor)])
        if mx > 0 and chips >= mx - self._dust_chips(obs, info):
            chips = mx
        return chips

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
            o = (
                torch.from_numpy(self._adapt(self.canonical_obs(obs, info)))
                .unsqueeze(0)
                .to(self.device)
            )
            m = torch.from_numpy(info.gate_mask).unsqueeze(0).to(self.device)
            b = torch.from_numpy(sizing_from_info(info)[None, :]).to(self.device)
            out = self.model.act(o, m, b, deterministic=deterministic)
        gate = int(out.gate.item())
        if gate != GATE_RAISE:
            return gate, 0
        return gate, self._anchor_chips(int(out.anchor.item()), obs, info)

    def node_distribution(self, obs: np.ndarray, info: StepInfo) -> NodeDist:
        raw = self._compute_node_distribution(
            self.model, self.device, self._adapt(self.canonical_obs(obs, info)), info
        )
        nd = NodeDist.from_dict(raw, backend_name=self.name)
        nd.mode = self.mode
        nd.coverage = self.coverage
        if nd.head_version >= 2 and nd.anchor_chips is not None:
            # Every anchor is an ATOM for a GTO net: no trained refinement, so
            # no bracket to score a user's size against and no refined rec.
            dust = self._dust_chips(obs, info)
            mx = int(nd.max_chips)
            nd.anchor_chips = [
                mx if (mx > 0 and int(c) >= mx - dust) else int(c)
                for c in nd.anchor_chips
            ]
            nd.refine_ok = [False] * len(nd.anchor_chips)
            nd.anchor_lo = list(nd.anchor_chips)
            nd.anchor_hi = list(nd.anchor_chips)
            nd.anchor_lo_raw = list(nd.anchor_chips)
            nd.anchor_hi_raw = list(nd.anchor_chips)
            if nd.rec_gate == GATE_RAISE and nd.rec_anchor is not None:
                nd.rec_chips = int(nd.anchor_chips[int(nd.rec_anchor)])
        if not self.value_head_trained:
            # (review 2026-09-20 F11) No value targets were trained: the head is
            # still at init, so its output is noise, not an EV.
            nd.value_bb = 0.0
        return nd

    @property
    def value_head_trained(self) -> bool:
        """False only when meta SAYS no row carried a value target."""
        n = (self.meta or {}).get("n_value_targets")
        return True if n is None else int(n) > 0

    def trained_coverage(self) -> dict[str, list[int]]:
        """Streets / seat counts present in the training rows (from meta)."""
        cov = (self.meta or {}).get("coverage")
        if isinstance(cov, dict) and cov.get("streets") and cov.get("seats"):
            streets = sorted(int(x) for x in cov["streets"])
            exact = cov.get("streets_exact")
            return {
                "streets": streets,
                "seats": sorted(int(x) for x in cov["seats"]),
                "streets_exact": (
                    streets if exact is None else sorted(int(x) for x in exact)
                ),
            }
        return {k: list(v) for k, v in _LEGACY_COVERAGE.items()}

    def supports(self, *, seats: int, street: int) -> bool:
        """True only for table shapes the net was actually TRAINED on.

        (review 2026-09-20 F11) Used to claim 2-6 seats on every street for a
        teacher validated on HU rivers. Now: the seat count must appear in the
        training rows and the street must be one whose rows ALL had an exact
        obs (``coverage.streets_exact``) — preflop CFR labels use a synthetic
        obs with no exact serve-side twin, so they never count.
        """
        cov = self.trained_coverage()
        return int(seats) in cov["seats"] and int(street) in cov["streets_exact"]

    def coverage_badge(self) -> dict[str, Any]:
        validated = bool(self.is_gto) and is_validated_gto_meta(self.meta)
        # Force honest: never show GTO if meta fails even if is_gto was set True
        show_gto = validated
        label, note = _badge_label(self.meta, validated=show_gto)
        probe = (self.meta or {}).get("probe")
        cov = self.trained_coverage()
        return {
            "backend": self.name,
            "mode": self.mode,
            "label": label,
            "is_gto": show_gto,
            "seats": _range_label(cov["seats"]),
            "streets": cov["streets_exact"],
            "note": note,
            "ckpt": self.ckpt_path,
            "source": (self.meta or {}).get("source"),
            "n_train": (self.meta or {}).get("n_train"),
            "probe_passed": (
                probe.get("passed") if isinstance(probe, dict) else None
            ),
            "is_gto_validated": (self.meta or {}).get("is_gto_validated"),
            "value_head_trained": self.value_head_trained,
            "badge_note": (self.meta or {}).get("gto_badge_note"),
            "obs_rev": None if self.meta is None else stamped_obs_rev(self.meta),
            "obs_rev_mismatch": (
                None if self.meta is None else obs_rev_mismatch(self.meta)
            ),
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
