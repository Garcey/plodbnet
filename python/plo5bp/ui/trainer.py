"""Trainer mode: GTO-Wizard-Practice-style drilling against the network.

The server deals fully random bomb-pot hands (`BombPotEnv.reset`), every
non-hero seat acts by *sampling* the network's mixed strategy
(`model_policy(deterministic=False)`), and each hero decision is scored
against the network's distribution at that node (gate probabilities +
Beta size density). A Monte-Carlo estimate of the EV lost by the user's
action vs the network's deterministic action is computed by rolling out
network-vs-network continuations of the same deal.

State/replay invariants:
- A hand is fully determined by `(config, seed, button)` — `reset(seed,
  button)` re-deals identically (ChaCha8), and the engine deals streets
  itself in non-study mode. Review/repeat/EV-loss all rebuild by replay;
  nothing snapshots live engine objects.
- This module is the ONLY consumer of torch's global RNG in the UI
  process (the study path always calls `act(deterministic=True)`).
  Opponent actions are re-seeded per node from `(hand seed, action
  prefix length)`, so opponent behavior is a pure function of the action
  prefix — MC rollouts in between don't perturb it, and "Repeat hand"
  reproduces opponent lines until the user deviates.
- `study_terminal` / `awaiting_next_street` are study-mode-only engine
  fields; trainer terminal state is synthesized from the `done` flag.
- Moot nodes (nothing to call, every other live seat all-in) are
  auto-checked by `_advance` for hero and opponents alike — an all-in
  hand runs out to showdown without drilling the user on meaningless
  check nodes. Auto-checks consume no model RNG and are recorded in
  `action_log` like any action (tagged `auto`), so replay determinism is
  unchanged.
- Hidden information (review 2026-09-20 H2): the MC EV-loss replays the
  REAL deal (villain cards + the real future board), so it is computed
  and stored on the `DecisionRecord` at act time but only SHOWN once the
  hand is terminal — live frames carry `feedback.ev_loss_bb = None` +
  `ev_loss_hidden`, and the stats blocks are committed per completed
  hand (`_commit_hand_stats`), never per move. Score / category are
  observation-only and show immediately. A hand abandoned mid-way
  contributes nothing to the stats.
- Every seed→sample region holds `_TORCH_RNG_LOCK` and ONLY those
  regions do: env replays / steps never run under it (`_rollout_ev`
  resumes its own RNG stream per depth via snapshot/restore).
- One host per NODE (`_backend_for`): a non-PPO backend (the GTO
  PolicyNetHost) serves a node only where its `supports(seats, street)`
  says its training covers it; every other node is served by the format's
  PPO host and tagged `backend: "ppo_fallback"`. Opponent sampling,
  recommendations / scoring, review and the MC continuations all go
  through that same per-node host, so the policy that is graded is the
  policy that plays. The choice reads public node state only.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from plo5bp.actions import (
    GATE_CHECK_CALL,
    GATE_FOLD,
    GATE_NAMES,
    GATE_RAISE,
)
from plo5bp.config import GameConfig, VARIANT_NLH, VARIANT_PLO5
from plo5bp.env import BombPotEnv, StepInfo
from plo5bp.eval import model_policy
from plo5bp.gto.backend import PpoSolverHost, StrategyBackend, make_ppo_host
from plo5bp.gto.policy_host import try_load_gto_host
from plo5bp.network import ActorCritic, CentralCritic, obs_adapter
from plo5bp.rollout import _critic_values, _rotate_opp_holes
from plo5bp.sizing import (
    ANCHOR_COUNT,
    PLO_ANCHOR_SPEC,
    anchor_grid_np,
    anchor_grid_torch,
    sizing_from_info,
)
from plo5bp.ui.common import (
    STREET_NAMES,
    anchor_label,
    anchor_label_spec,
    chips_to_bb,
    history_entries,
    position_name,
    validate_card_list,
)
from plo5bp.ui.hand_describe import describe_made_hand, describe_made_hand_nlh

# UI format id (mirrors server.FORMAT_EXPERIMENTAL). Same PLO5 engine rules;
# only the served checkpoint / obs projection differ.
FORMAT_EXPERIMENTAL = "experimental"


def _engine_variant(fmt_id: str) -> str:
    """Map a UI format id onto the engine GameConfig.variant string."""
    if fmt_id == FORMAT_EXPERIMENTAL:
        return VARIANT_PLO5
    return fmt_id


logger = logging.getLogger("plo5bp.ui.trainer")

def _public_mode() -> bool:
    """PLO5BP_PUBLIC, read at CALL time. The security-relevant guards (the
    `mc_rollouts` cap, `_ts` failing closed) use this rather than the
    import-time `_PUBLIC` constant so they hold no matter which module was
    imported first or how a test toggles the flag."""
    return os.environ.get("PLO5BP_PUBLIC", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Mirrors server.PLO5BP_PUBLIC (avoids a circular import): in the public
# build the state payload must carry no trace of the live-capture fields.
_PUBLIC = _public_mode()

BB_CHIPS = 10000

# Trainer sampling is a pure function of (hand seed, action prefix) via
# torch.manual_seed on the GLOBAL torch RNG. With per-user TrainerSessions
# (public build) two requests may interleave between seed and sample, so
# every seed→sample region takes this lock. Uncontended in the local build.
_TORCH_RNG_LOCK = threading.Lock()


def _snapshot_rng_state(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The global torch RNG stream position (CPU + the sampling device's
    CUDA generator). Call while holding `_TORCH_RNG_LOCK`."""
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.get_rng_state(), cuda


def _restore_rng_state(
    state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device
) -> None:
    """Resume a stream captured by `_snapshot_rng_state` (lock held): lets a
    multi-draw seed→sample sequence release the lock between draws and still
    see exactly the uninterrupted stream."""
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state(cuda, device)


# `_persist`: transient os.replace PermissionError retries (Windows).
_PERSIST_RETRIES = 5
_PERSIST_RETRY_SLEEP_S = 0.02

# Node-level `backend` tag when the ACTIVE (non-PPO) host declined a node via
# `supports()` and the format's PPO host served it instead.
BACKEND_PPO_FALLBACK = "ppo_fallback"

GATE_SLUGS = {GATE_FOLD: "fold", GATE_CHECK_CALL: "check_call", GATE_RAISE: "raise"}
GATE_NAME_TO_IDX = {v: k for k, v in GATE_SLUGS.items()}

CATEGORIES = ("best", "correct", "inaccuracy", "wrong", "blunder")
CATEGORY_MARKS = {
    "best": "✓✓",
    "correct": "✓",
    "inaccuracy": "~",
    "wrong": "✗",
    "blunder": "✗✗",
}

# All scoring tunables in one place. Categories are score bands except
# the absolute-probability blunder override; `best` additionally
# requires picking the argmax gate.
SCORING = {
    "size_floor": 0.4,         # raise-size quality floor in size_factor
    "best_min": 85.0,
    "correct_min": 60.0,
    "inaccuracy_min": 30.0,
    "wrong_min": 10.0,         # below -> blunder
    "blunder_gate_prob": 0.02, # P[user gate] below -> blunder regardless
    "u_eps": 1e-4,
}


def score_move(
    gate_probs: list[float],
    alpha: float,
    beta: float,
    min_chips: int,
    max_chips: int,
    user_gate: int,
    user_chips: int,
) -> dict[str, Any]:
    """Score one decision against the network's node distribution.

    `gate_ratio` compares the user's gate probability to the argmax
    gate's. For raises in a non-degenerate range, `size_q` is the Beta
    density at the user's normalized size relative to the density at the
    mode (normalization constants cancel). Degenerate ranges — short
    shove (min==0) or single-point (min==max) — have no size choice, so
    size_q is 1.
    """
    g_star = max(range(len(gate_probs)), key=lambda i: gate_probs[i])
    p_user = float(gate_probs[user_gate])
    p_best = float(gate_probs[g_star])
    gate_ratio = (p_user / p_best) if p_best > 0 else 0.0

    size_q = 1.0
    if user_gate == GATE_RAISE and min_chips > 0 and max_chips > min_chips:
        eps = SCORING["u_eps"]
        u = (user_chips - min_chips) / (max_chips - min_chips)
        u = min(1.0 - eps, max(eps, u))
        # Reference the Beta MEAN (= the size the deterministic policy bets
        # and the UI shows as "the network's choice"), clamped to <=1, so
        # playing the recommended size earns full size credit. Density peaks
        # at the mode, not the mean, so a size between mean and mode also
        # reads as full quality; only the tails are penalized. (The earlier
        # mode reference scored the recommended mean < 100% on skewed Betas.)
        ref = alpha / (alpha + beta)
        ref = min(1.0 - eps, max(eps, ref))

        def logpdf(x: float) -> float:
            return (alpha - 1.0) * math.log(x) + (beta - 1.0) * math.log(1.0 - x)

        size_q = min(1.0, math.exp(logpdf(u) - logpdf(ref)))

    size_factor = SCORING["size_floor"] + (1.0 - SCORING["size_floor"]) * size_q
    score = 100.0 * gate_ratio * (size_factor if user_gate == GATE_RAISE else 1.0)
    score = max(0.0, min(100.0, score))

    if p_user < SCORING["blunder_gate_prob"] or score < SCORING["wrong_min"]:
        category = "blunder"
    elif score < SCORING["inaccuracy_min"]:
        category = "wrong"
    elif score < SCORING["correct_min"]:
        category = "inaccuracy"
    elif user_gate == g_star and score >= SCORING["best_min"]:
        category = "best"
    else:
        category = "correct"
    return {
        "score": score,
        "category": category,
        "gate_ratio": gate_ratio,
        "size_q": size_q,
    }


def unclamped_brackets(
    min_raise: int, max_raise: int, pot: int, to_call: int, spec: Any,
) -> tuple[list[int], list[int]]:
    """Per-anchor refinement bracket in chips WITHOUT the
    [min_raise, max_raise] clip — the chip axis the POLICY maps `u` over.

    (review 2026-09-20 H1) `sizing.refine_chips_*` computes
    ``chips = clip(round(to_call + frac(u) * base), min_raise, max_raise)``
    with ``frac(u)`` spanning the spec's per-mille bracket, i.e. `u` lives
    on the UNCLAMPED bracket and only the resulting chips are clamped.
    `AnchorGrid.lo/hi` are the CLAMPED bounds, so inverting chips→u over
    them misplaces every size whenever min-raise lands inside a bracket
    (or max-raise inside the top one). Same integer half-up rounding as
    `sizing.anchor_grid_np`, minus the clip (sizing.py owns the canonical
    math; this is its inverse-side twin for the scorer only). The ALL-IN
    atom (NLH) has no fraction: its "bracket" is the single point
    `max_raise`, as in the grid.
    """
    base = int(pot) + int(to_call)
    tc = int(to_call)
    lo = [tc + (int(pm) * base + 500) // 1000 for pm in spec.bracket_lo_pm]
    hi = [tc + (int(pm) * base + 500) // 1000 for pm in spec.bracket_hi_pm]
    if spec.allin_atom:
        lo.append(int(max_raise))
        hi.append(int(max_raise))
    return lo, hi


def attach_unclamped_brackets(
    dist: dict[str, Any], info: StepInfo, spec: Any,
) -> dict[str, Any]:
    """Ensure a v2+ node-distribution dict carries `anchor_lo_raw` /
    `anchor_hi_raw`. `compute_node_distribution` exports them, but
    `gto.backend.NodeDist` (not this module's) round-trips a fixed field
    list and drops them — so the session re-derives them from the node's
    own `StepInfo` after `as_dict()`. No-op for v1 dicts, for dicts that
    already carry the keys, and when `spec` does not match the dict's
    ladder length. (review 2026-09-20 H1)"""
    if int(dist.get("head_version", 1)) < 2 or "anchor_lo_raw" in dist:
        return dist
    if spec is None or spec.count != len(dist.get("anchor_chips") or ()):
        return dist
    sizing = sizing_from_info(info)
    lo, hi = unclamped_brackets(
        int(sizing[0]), int(sizing[1]), int(sizing[2]), int(sizing[3]), spec
    )
    dist["anchor_lo_raw"], dist["anchor_hi_raw"] = lo, hi
    return dist


def refine_u_interval(
    dist: dict[str, Any], k: int, chips: int,
) -> tuple[float, float] | None:
    """Inverse image `[u_lo, u_hi]` of `chips` under the policy's
    (anchor k, u) -> chips map; None when anchor `k` has no slider (atom /
    collapsed bracket).

    (review 2026-09-20 H1) `u` is inverted over the UNCLAMPED bracket
    (`anchor_lo_raw/hi_raw`) - the axis the policy samples on - falling
    back to the clamped `anchor_lo/hi` only for hand-built dicts that lack
    the raw keys. Away from the clip the image is the single point
    `(chips - lo) / (hi - lo)`. Where the clip is active it is an INTERVAL:
    every u below the min-raise crossing collapses onto min_raise
    (`[0, u_pt]`), every u above the max-raise crossing onto max_raise
    (`[u_pt, 1]`).
    """
    if not dist["refine_ok"][k]:
        return None
    lo_raw = dist.get("anchor_lo_raw")
    hi_raw = dist.get("anchor_hi_raw")
    if lo_raw is not None and hi_raw is not None:
        lo, hi = int(lo_raw[k]), int(hi_raw[k])
    else:
        lo, hi = int(dist["anchor_lo"][k]), int(dist["anchor_hi"][k])
    if hi <= lo:
        return None
    u_pt = (chips - lo) / (hi - lo)
    u_lo = u_hi = u_pt
    max_chips = int(dist["max_chips"])
    min_chips = min(int(dist["min_chips"]), max_chips)
    if chips <= min_chips and lo < chips:
        u_lo = 0.0           # lower clip active: u in [0, u_pt] -> min_raise
    if chips >= max_chips and hi > chips:
        u_hi = 1.0           # upper clip active: u in [u_pt, 1] -> max_raise
    return u_lo, u_hi


def _refine_pdf_ratio(
    dist: dict[str, Any], k: int, chips: int,
) -> float:
    """Beta-density ratio (vs the Beta MEAN reference) of playing `chips`
    inside refinable anchor `k`'s bracket; 1.0 for atoms / collapsed
    brackets. Over a clipped interval (see `refine_u_interval`) the point
    closest to the reference is scored: a min-raise that the anchor's own
    mean would also have produced earns full size credit instead of being
    read as an off-centre `u`. (review 2026-09-20 H1)"""
    interval = refine_u_interval(dist, k, chips)
    if interval is None:
        return 1.0
    u_lo, u_hi = interval
    eps = SCORING["u_eps"]
    alpha, beta = dist["refine_params"][k - 1]
    # Reference the Beta MEAN (= the size the deterministic policy bets and
    # the UI shows as the recommendation), clamped to <=1, so betting the
    # recommended size earns full size credit. The earlier mode reference
    # penalized the recommended mean on any skewed Beta (a matched bet
    # could score well under 100%).
    ref = alpha / (alpha + beta)
    ref = min(1.0 - eps, max(eps, ref))
    u = min(max(ref, u_lo), u_hi)
    u = min(1.0 - eps, max(eps, u))

    def logpdf(x: float) -> float:
        return (alpha - 1.0) * math.log(x) + (beta - 1.0) * math.log(1.0 - x)

    return min(1.0, math.exp(logpdf(u) - logpdf(ref)))


def snap_to_anchor(dist: dict[str, Any], chips: int) -> tuple[int | None, float]:
    """Map a raise of `chips` onto the v2 anchor ladder: returns
    `(anchor, refinement-pdf-ratio)`.

    (review 2026-09-20 H1) Candidates are the LEGAL anchors that can
    actually PRODUCE `chips` under the policy's own (anchor, u) → chips
    map: refinable anchors whose bracket contains the chips (for a legal
    raise, clamped-bracket containment == unclamped-bracket containment)
    and atoms / collapsed anchors whose chips equal them. Among candidates
    the one with the highest `P(anchor) × pdf-ratio` wins (ties → the
    recommended anchor, then the nearer anchor, then the lower index) — a
    min-raise is simultaneously "the min atom" and "the low end of the
    first clipped bracket", and the user is credited with whichever
    reading the network likes more. The pre-fix nearest-by-chips snap sent
    a clipped recommendation to the wrong anchor (repro t5_snap). Chips no
    anchor can produce (the gaps of the NLH ladder) fall back to the
    nearest legal anchor by chip distance (tie → lower), as before.
    """
    legal_ks = [
        k for k in range(len(dist["anchor_legal"])) if dist["anchor_legal"][k]
    ]
    if not legal_ks:
        return None, 1.0
    a_chips = dist["anchor_chips"]
    a_probs = dist["anchor_probs"]
    rec_anchor = dist.get("rec_anchor")

    candidates: list[int] = []
    for k in legal_ks:
        if dist["refine_ok"][k]:
            if int(dist["anchor_lo"][k]) <= chips <= int(dist["anchor_hi"][k]):
                candidates.append(k)
        elif int(a_chips[k]) == chips:
            candidates.append(k)
    if not candidates:
        k_near = min(legal_ks, key=lambda k: (abs(chips - int(a_chips[k])), k))
        return k_near, _refine_pdf_ratio(dist, k_near, chips)

    ratios = {k: _refine_pdf_ratio(dist, k, chips) for k in candidates}
    best = max(
        candidates,
        key=lambda k: (
            float(a_probs[k]) * ratios[k],
            k == rec_anchor,
            -abs(chips - int(a_chips[k])),
            -k,
        ),
    )
    return best, ratios[best]


def score_move_v2(
    dist: dict[str, Any],
    user_gate: int,
    user_chips: int,
) -> dict[str, Any]:
    """Score one decision against a v2 (anchor head) node distribution.

    `gate_ratio` is unchanged from v1. For raises, the user's chips snap
    to `user_anchor` via `snap_to_anchor` (the legal anchor whose bracket
    can produce them; nearest-by-chips only as a fallback) and
    `size_q = P(user_anchor)/P(best_anchor) × refinement-pdf-ratio`,
    where the pdf ratio compares the user anchor's Beta density at the
    user's position on the UNCLAMPED bracket vs at its mean. Atoms,
    short-shove and collapsed brackets have no within-anchor size choice
    → pdf ratio 1. Categories, the size floor, and the blunder override
    match v1.

    (review 2026-09-20 H1) Playing EXACTLY the recommendation's chips is
    the recommended (anchor, u) by definition — full size credit, no
    inversion — so the exact rec always grades as the top category
    regardless of how min/max-raise clip the rec anchor's bracket.
    """
    gate_probs = dist["gate_probs"]
    g_star = max(range(len(gate_probs)), key=lambda i: gate_probs[i])
    p_user = float(gate_probs[user_gate])
    p_best = float(gate_probs[g_star])
    gate_ratio = (p_user / p_best) if p_best > 0 else 0.0

    size_q = 1.0
    user_anchor: int | None = None
    if user_gate == GATE_RAISE:
        # Spec-length lists (PLO 11 anchors, NLH 12 with the ALL-IN atom).
        legal_ks = [
            k for k in range(len(dist["anchor_legal"]))
            if dist["anchor_legal"][k]
        ]
        rec_anchor = dist.get("rec_anchor")
        if (
            int(dist.get("rec_gate", -1)) == GATE_RAISE
            and int(user_chips) == int(dist.get("rec_chips", -1))
            and rec_anchor is not None
            and bool(dist["anchor_legal"][int(rec_anchor)])
        ):
            user_anchor, pdf_ratio = int(rec_anchor), 1.0
        else:
            user_anchor, pdf_ratio = snap_to_anchor(dist, int(user_chips))
        if user_anchor is not None:  # (a legal raise always has >=1 legal anchor)
            a_probs = dist["anchor_probs"]
            k_best = max(legal_ks, key=lambda k: a_probs[k])
            p_ku = float(a_probs[user_anchor])
            p_kb = float(a_probs[k_best])
            anchor_ratio = (p_ku / p_kb) if p_kb > 0 else 0.0
            size_q = anchor_ratio * pdf_ratio

    size_factor = SCORING["size_floor"] + (1.0 - SCORING["size_floor"]) * size_q
    score = 100.0 * gate_ratio * (size_factor if user_gate == GATE_RAISE else 1.0)
    score = max(0.0, min(100.0, score))

    if p_user < SCORING["blunder_gate_prob"] or score < SCORING["wrong_min"]:
        category = "blunder"
    elif score < SCORING["inaccuracy_min"]:
        category = "wrong"
    elif score < SCORING["correct_min"]:
        category = "inaccuracy"
    elif user_gate == g_star and score >= SCORING["best_min"]:
        category = "best"
    else:
        category = "correct"
    return {
        "score": score,
        "category": category,
        "gate_ratio": gate_ratio,
        "size_q": size_q,
        "user_anchor": user_anchor,
    }


def compute_node_distribution(
    model: ActorCritic,
    device: torch.device,
    obs_np: np.ndarray,
    info: StepInfo,
) -> dict[str, Any]:
    """One forward at a decision node, plus the deterministic
    recommendation (argmax gate; v1: Beta-mean chips with short-shove
    redirect; v2: argmax legal anchor with refinement-mean chips).

    v1 dicts carry (alpha, beta); v2 dicts carry the masked anchor
    distribution, per-anchor chips/legality/brackets, and the (9, 2)
    refinement params. Both carry "head_version" so scorers branch.
    """
    obs_t = torch.from_numpy(
        obs_adapter(model)(obs_np)
    ).unsqueeze(0).to(device)
    gm_t = torch.from_numpy(info.gate_mask).unsqueeze(0).to(device)
    raise_max = int(info.max_raise_chips)
    raise_min = min(int(info.min_raise_chips), raise_max)

    if getattr(model, "head_version", 1) >= 2:
        # Sizing math under the MODEL'S OWN ladder (PLO 11-anchor pot
        # grid or NLH 12-anchor overbet grid with the ALL-IN atom).
        spec = getattr(model, "anchor_spec", PLO_ANCHOR_SPEC)
        sizing = sizing_from_info(info)
        sizing_t = torch.from_numpy(sizing[None, :]).to(device)
        with torch.no_grad():
            gate_logits, anchor_head_out, refine, value = model(obs_t, gm_t)
            gate_probs = F.softmax(gate_logits, dim=-1).squeeze(0).tolist()
            _act_out = model.act(obs_t, gm_t, sizing_t, deterministic=True)
            # Anchor histogram via the model's own (head-agnostic) anchor
            # distribution: flat softmax for v2, discretized-logistic for v4.
            anchor_probs = (
                model._anchor_dist(anchor_head_out, anchor_grid_torch(sizing_t, spec))
                .probs.squeeze(0).float().cpu().numpy()
            )
            refine_np = refine.squeeze(0).float().cpu().numpy()  # (interior, 2)
            # v5 mixture head: per-component (mu, s, w) so the client can
            # annotate the size menu — parity with the study recommendation
            # (_recommendation_v2), which already exposes this block.
            mixture = None
            if hasattr(model, "mixture_params"):
                mu_t, s_t, w_t = model.mixture_params(anchor_head_out)
                mixture = {
                    "mu": [round(float(x), 4) for x in mu_t.squeeze(0).tolist()],
                    "s": [round(float(x), 4) for x in s_t.squeeze(0).tolist()],
                    "w": [round(float(x), 4) for x in w_t.squeeze(0).tolist()],
                }
        grid = anchor_grid_np(sizing[0], sizing[1], sizing[2], sizing[3], spec)
        # (review 2026-09-20 H1) The scorer inverts chips -> u over the
        # UNCLAMPED bracket (the policy's own u axis); grid.lo/hi are clipped.
        lo_raw, hi_raw = unclamped_brackets(
            int(sizing[0]), int(sizing[1]), int(sizing[2]), int(sizing[3]), spec
        )
        return {
            "head_version": model.head_version,
            "gate_probs": [float(p) for p in gate_probs],
            "anchor_probs": [float(p) for p in anchor_probs],
            "anchor_chips": [int(c) for c in grid.chips],
            "anchor_legal": [bool(b) for b in grid.legal],
            "anchor_lo": [int(c) for c in grid.lo],
            "anchor_hi": [int(c) for c in grid.hi],
            "anchor_lo_raw": lo_raw,
            "anchor_hi_raw": hi_raw,
            "refine_ok": [bool(b) for b in grid.refine_ok],
            "refine_params": [[float(a), float(b)] for a, b in refine_np],
            "rec_anchor": int(_act_out.anchor.item()),
            "min_chips": int(info.min_raise_chips),
            "max_chips": raise_max,
            "pot_ref_chips": int(sizing[2]) + 2 * int(sizing[3]),
            "rec_gate": int(_act_out.gate.item()),
            "rec_chips": int(_act_out.chips.item()),
            "value_bb": float(value.squeeze(0).item()),
            "mixture": mixture,
        }

    bounds_t = torch.tensor(
        [[raise_min, raise_max]], dtype=torch.long, device=device
    )
    with torch.no_grad():
        gate_logits, raise_params, value = model(obs_t, gm_t)
        gate_probs = F.softmax(gate_logits, dim=-1).squeeze(0).tolist()
        _act_out = model.act(
            obs_t, gm_t, bounds_t, deterministic=True
        )
    rec_gate = int(_act_out.gate.item())
    rec_chips = int(_act_out.chips.item())
    if rec_gate == GATE_RAISE and int(info.min_raise_chips) == 0 and raise_max > 0:
        rec_chips = raise_max
    return {
        "head_version": 1,
        "gate_probs": [float(p) for p in gate_probs],
        "alpha": float(raise_params[0, 0].item()),
        "beta": float(raise_params[0, 1].item()),
        "min_chips": int(info.min_raise_chips),
        "max_chips": raise_max,
        "rec_gate": rec_gate,
        "rec_chips": rec_chips,
        "value_bb": float(value.squeeze(0).item()),
    }


def _rec_refine_params(dist: dict[str, Any]) -> tuple[float, float]:
    """(alpha, beta) of the rec anchor's refinement slider; (1.0, 1.0)
    when the rec anchor is an atom / has a collapsed bracket."""
    ra = int(dist["rec_anchor"])
    if dist["refine_ok"][ra]:
        a, b = dist["refine_params"][ra - 1]
        return float(a), float(b)
    return 1.0, 1.0


def _anchors_payload(
    anchor_probs: list[float],
    anchor_chips: list[int],
    anchor_legal: list[bool],
    bb: int,
    spec: Any = PLO_ANCHOR_SPEC,
) -> list[dict[str, Any]]:
    """Legal-only anchor histogram rows for client rendering. The lists
    are spec-length (PLO 11 / NLH 12 incl. the ALL-IN atom)."""
    return [
        {
            "k": int(k),
            "label": anchor_label_spec(spec, int(k)),
            # Pot fraction (None for the ALL-IN atom, whose chips are max_raise
            # not a pot fraction). WITHOUT this the client's hi.frac is
            # undefined and betCurveSVG falls into idxMode (the NLH all-in
            # ladder layout) — skipping the chips-space axis AND the mixture
            # size labels. Parity with the study rec (_recommendation_v2).
            "frac": (
                None if (spec.allin_atom and int(k) == spec.count - 1)
                else spec.fracs_pm[int(k)] / 1000.0
            ),
            "prob": round(float(anchor_probs[k]), 4),
            "chips": int(anchor_chips[k]),
            "chips_bb": round(chips_to_bb(int(anchor_chips[k]), bb), 4),
        }
        for k in range(len(anchor_legal))
        if bool(anchor_legal[k])
    ]


def _stable_seed(*parts: int) -> int:
    """Deterministic 63-bit mix of ints (process-restart stable, unlike
    Python's salted hash())."""
    acc = 0x9E3779B97F4A7C15
    for p in parts:
        acc ^= (int(p) & 0xFFFFFFFFFFFFFFFF) * 0x100000001B3
        acc &= 0xFFFFFFFFFFFFFFFF
        acc = (acc * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF
        acc ^= acc >> 29
    return acc & 0x7FFFFFFFFFFFFFFF


# --- Settings / request models ----------------------------------------------


class TrainerSettings(BaseModel):
    seats_mode: Literal["random", "fixed"] = "random"
    seats_fixed: int = Field(6, ge=2, le=6)
    seats_min: int = Field(2, ge=2, le=6)
    seats_max: int = Field(6, ge=2, le=6)
    stacks_mode: Literal["fixed", "random", "per_seat"] = "fixed"
    stack_bb: float = Field(20.0, ge=1.0, le=1000.0)
    stack_min_bb: float = Field(10.0, ge=1.0, le=1000.0)
    stack_max_bb: float = Field(50.0, ge=1.0, le=1000.0)
    # Per-seat [lo, hi] in bb; lo == hi means fixed. Entries beyond the
    # drawn seat count are ignored.
    stacks_per_seat_bb: list[tuple[float, float]] = Field(
        default_factory=lambda: [(20.0, 20.0)] * 6
    )
    hero_position_mode: Literal["random", "kth"] = "random"
    hero_kth: int = Field(1, ge=1, le=6)  # 1 = first to act (left of BTN)
    ante_bb: float = Field(3.0, ge=0.0, le=100.0)
    # Monte-Carlo rollouts per EV-loss candidate; 0 disables EV loss.
    # 16 keeps a deviating /trainer/act under ~1s with the 2048x4 net on CPU.
    # The PUBLIC build clamps this to MC_ROLLOUTS_PUBLIC_CAP (see
    # `_cap_settings`); 256 is the local-build ceiling.
    mc_rollouts: int = Field(16, ge=0, le=256)
    # Display conversion: dollars per 1bb when the UI is in $ mode.
    dollars_per_bb: float = Field(20.0, gt=0.0, le=100000.0)

    @model_validator(mode="after")
    def _check_ranges(self) -> "TrainerSettings":
        if self.seats_min > self.seats_max:
            raise ValueError("seats_min must be <= seats_max")
        if self.stack_min_bb > self.stack_max_bb:
            raise ValueError("stack_min_bb must be <= stack_max_bb")
        if len(self.stacks_per_seat_bb) != 6:
            raise ValueError("stacks_per_seat_bb must have 6 entries")
        for i, (lo, hi) in enumerate(self.stacks_per_seat_bb):
            if not (1.0 <= lo <= hi <= 1000.0):
                raise ValueError(
                    f"stacks_per_seat_bb[{i}]: need 1 <= lo <= hi <= 1000"
                )
        return self


# (review 2026-09-20 F5/F7) Public-build ceiling on `mc_rollouts`. Each MC
# arm costs ~n env replays + a batched forward per depth on a CPU 2048x4
# net, shared by every signed-in user; a free user could set 256 and stall
# the box. 32 keeps a deviating /trainer/act at ~2x the default cost. The
# local build keeps the model's own 0..256 range.
MC_ROLLOUTS_PUBLIC_CAP = 32


def mc_rollouts_cap() -> int:
    """Largest `mc_rollouts` the running build honours."""
    return MC_ROLLOUTS_PUBLIC_CAP if _public_mode() else 256


def _cap_settings(settings: TrainerSettings) -> TrainerSettings:
    """Clamp (not reject) `mc_rollouts` to the build's cap: a settings save
    from a client that still offers 256 — or a stats file persisted before
    the cap existed — keeps working, and the stored/echoed value is the one
    actually used."""
    cap = mc_rollouts_cap()
    if settings.mc_rollouts > cap:
        return settings.model_copy(update={"mc_rollouts": cap})
    return settings


def _default_settings(variant: str) -> TrainerSettings:
    """The format's factory settings. PLO5 = the TrainerSettings field
    defaults (20bb bomb pot, 3bb ante, $20/bb). NLH = the 5/10($5)
    table: 100bb top-off baseline, 100-250bb random band, 0.5bb ante,
    $10/bb. Each format OWNS its settings object — switching formats
    swaps objects, never overwrites values (a shared object once leaked
    NLH's 0.5bb ante into PLO5 hands across a restart, 2026-07-04)."""
    if variant == VARIANT_NLH:
        return TrainerSettings(
            stack_bb=100.0,
            stack_min_bb=100.0,
            stack_max_bb=250.0,
            stacks_per_seat_bb=[(100.0, 100.0)] * 6,
            ante_bb=0.5,
            dollars_per_bb=10.0,
        )
    return TrainerSettings()


class TrainerActRequest(BaseModel):
    gate: str = Field(..., pattern=r"^(fold|check_call|raise)$")
    chips: int | None = Field(default=None, ge=0)


class WhatifRequest(BaseModel):
    decision: int = Field(..., ge=0)
    hero_hole: list[int | None] | None = None
    flop_a: list[int | None] | None = None
    flop_b: list[int | None] | None = None
    turn: list[int | None] | None = None
    river: list[int | None] | None = None


class StatsResetRequest(BaseModel):
    scope: Literal["session", "lifetime"]


# --- Records ------------------------------------------------------------------


@dataclass
class DecisionRecord:
    decision_idx: int
    street: int
    action_log_idx: int  # len(action_log) BEFORE the hero action
    gate_probs: list[float]
    alpha: float
    beta: float
    min_chips: int  # raw engine min (0 in the short-shove regime)
    max_chips: int
    rec_gate: int
    rec_chips: int
    value_bb: float
    pot_chips: int
    to_call_chips: int
    user_gate: int
    user_chips: int
    gate_ratio: float
    size_q: float
    score: float
    category: str
    ev_user_bb: float | None = None
    ev_best_bb: float | None = None
    # Per-decision DISPLAY value: max(0, best - user). The aggregate stats
    # accumulate `ev_loss_signed_bb` instead (review 2026-09-20 H4): the MC
    # estimate is noisy (sd ~4-5bb at n=16), so clamping each decision at
    # zero before summing biased every total upward.
    ev_loss_bb: float | None = None
    ev_loss_signed_bb: float | None = None
    # Hero's total committed chips AT the decision node (= chips forfeited
    # on a fold). Rebases the EV components to forward-facing (fold = 0).
    hero_committed_chips: int = 0
    # Hero's STREET commit at the decision node, BEFORE the action. Engine
    # raise chips (`user_chips` / `rec_chips`) are raise-BY deltas; the
    # client shows raise-TO totals = delta + this. Sent as
    # `actor_commit_chips` so the client never has to derive it (a walk over
    # the action list is inexact for a blind that posted short).
    actor_commit_chips: int = 0
    # v2 (anchor head) extras; None/1 on v1 records. For v2, (alpha,
    # beta) above hold the REC anchor's refinement params (1.0/1.0 when
    # the rec anchor is an atom).
    head_version: int = 1
    anchor_probs: list[float] | None = None
    anchor_chips: list[int] | None = None
    anchor_legal: list[bool] | None = None
    rec_anchor: int | None = None
    user_anchor: int | None = None
    # 100%-pot bet reference (chips) for the sizing-curve axis; 0 on v1.
    pot_ref_chips: int = 0
    # Host that served this node: the active backend's name, or
    # BACKEND_PPO_FALLBACK when it did not cover the node (see `_backend_for`).
    backend: str = "ppo"


@dataclass
class HandRecord:
    hand_no: int
    seed: int
    button: int
    hero_seat: int
    config: GameConfig
    env: BombPotEnv
    last_obs: np.ndarray | None = None
    last_info: StepInfo | None = None
    # Every action by every seat, in order — the replay source of truth.
    action_log: list[dict[str, int]] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)
    opp_actions_since_hero: list[dict[str, Any]] = field(default_factory=list)
    terminal: bool = False
    rewards_bb: list[float] | None = None
    # Display order (rank high->low, suit-tiebroken). Replays that need
    # bit-exact observations use `all_holes_dealt` — the engine's
    # opp-equity float summation is sensitive to hole input order.
    all_holes: list[list[int]] = field(default_factory=list)
    all_holes_dealt: list[list[int]] = field(default_factory=list)
    is_repeat: bool = False
    # Set once this hand's decisions were folded into the stats blocks
    # (`_commit_hand_stats`, at hand end) — guards against a double commit.
    stats_committed: bool = False


def _finite(x: Any, default: float = 0.0) -> float:
    """float(x) if finite, else `default` — a hand-edited / corrupted stats
    file must not poison every later aggregate with NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


@dataclass
class StatsBlock:
    hands: int = 0
    moves: int = 0
    score_sum: float = 0.0
    cat_counts: dict[str, int] = field(
        default_factory=lambda: {c: 0 for c in CATEGORIES}
    )
    # (review 2026-09-20 H4) EV loss is kept as TWO sums:
    # - `ev_loss_sum_bb`: FROZEN legacy history. Until 2026-09-20 each
    #   decision was clamped at zero before it was added here, so this sum is
    #   already-clamped (upward-biased) and can't be un-biased after the fact.
    #   Old stats files load into it unchanged; nothing adds to it any more.
    # - `ev_loss_signed_sum_bb`: the SIGNED per-decision estimates
    #   (best - user, negative when noise favours the user's action), over
    #   `ev_loss_n` MC-estimated decisions. Zero-mean noise cancels in the
    #   sum; the clamp at zero happens once, on the aggregate, in `project`.
    ev_loss_sum_bb: float = 0.0
    ev_loss_signed_sum_bb: float = 0.0
    ev_loss_n: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hands": self.hands,
            "moves": self.moves,
            "score_sum": self.score_sum,
            "cat_counts": dict(self.cat_counts),
            "ev_loss_sum_bb": self.ev_loss_sum_bb,
            "ev_loss_signed_sum_bb": self.ev_loss_signed_sum_bb,
            "ev_loss_n": self.ev_loss_n,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StatsBlock":
        counts = {c: 0 for c in CATEGORIES}
        counts.update({k: int(v) for k, v in dict(d.get("cat_counts", {})).items()
                       if k in counts})
        return cls(
            hands=int(d.get("hands", 0)),
            moves=int(d.get("moves", 0)),
            score_sum=_finite(d.get("score_sum", 0.0)),
            cat_counts=counts,
            # Pre-H4 files carry only this key: already-clamped history.
            ev_loss_sum_bb=max(0.0, _finite(d.get("ev_loss_sum_bb", 0.0))),
            ev_loss_signed_sum_bb=_finite(d.get("ev_loss_signed_sum_bb", 0.0)),
            ev_loss_n=int(d.get("ev_loss_n", 0)),
        )

    @property
    def ev_loss_total_bb(self) -> float:
        """Aggregate EV loss for display: legacy (already-clamped) history
        plus the signed sum clamped at zero ONCE, as an aggregate."""
        return self.ev_loss_sum_bb + max(0.0, self.ev_loss_signed_sum_bb)

    def add_decision(self, d: "DecisionRecord") -> None:
        self.moves += 1
        self.score_sum += d.score
        self.cat_counts[d.category] += 1
        if d.ev_loss_signed_bb is not None:
            self.ev_loss_signed_sum_bb += d.ev_loss_signed_bb
            self.ev_loss_n += 1

    def project(self) -> dict[str, Any]:
        gto = (self.score_sum / self.moves) if self.moves > 0 else None
        total = self.ev_loss_total_bb
        per_hand = (total / self.hands) if self.hands > 0 else None
        return {
            "hands": self.hands,
            "moves": self.moves,
            "gto_score": round(gto, 1) if gto is not None else None,
            "cat_counts": dict(self.cat_counts),
            "ev_loss_total_bb": round(total, 2),
            "ev_loss_per_hand_bb": round(per_hand, 3) if per_hand is not None else None,
        }


# --- Session ------------------------------------------------------------------


class TrainerSession:
    def __init__(
        self,
        model: ActorCritic,
        device: torch.device,
        stats_path: Path | None = None,
        critic: CentralCritic | None = None,
        backend: StrategyBackend | None = None,
    ):
        self.model = model
        self.device = device
        # Centralized critic (sees all hole cards) for the review's
        # "true EV" readout. None on v1 / when the checkpoint lacks one
        # → review shows only the actor's own (blind) value estimate.
        self.critic = critic
        # StrategyBackend: T0 = PpoSolverHost (default). T1 swaps in
        # PolicyNetHost without rewriting act / score / advance.
        self.backend: StrategyBackend = backend or make_ppo_host(model, device)
        # The FORMAT's own PPO actor and its host. Under a non-PPO backend
        # this is the per-node fallback for every node that backend does not
        # cover (`_backend_for`); under the PPO backend it IS the backend.
        self._format_model = model
        self._ppo_host: PpoSolverHost | None = None
        self._refresh_ppo_host()
        # Active game format. `set_format` swaps model/critic to the new
        # format's pair and points `self.settings` at that format's OWN
        # settings object (see settings_by_variant).
        self.variant = VARIANT_PLO5
        self.lock = threading.Lock()
        self.settings_by_variant: dict[str, TrainerSettings] = {
            VARIANT_PLO5: _default_settings(VARIANT_PLO5),
            VARIANT_NLH: _default_settings(VARIANT_NLH),
            FORMAT_EXPERIMENTAL: _default_settings(FORMAT_EXPERIMENTAL),
        }
        self.settings = self.settings_by_variant[self.variant]
        self.hand: HandRecord | None = None
        self.hand_no = 0
        self.session_stats = StatsBlock()
        self.lifetime_stats = StatsBlock()
        self.rng = np.random.default_rng()
        self._policy = model_policy(model, deterministic=False)
        self._obs_adapt = obs_adapter(model)
        # The critic is paired with the FORMAT's PPO actor (they were trained
        # together and share an obs width) — NOT with whatever network the
        # active backend serves. Its obs projection therefore follows
        # `model` here / in `set_format`, never `set_backend`.
        # (review 2026-09-20 H3)
        self._critic_obs_adapt = obs_adapter(model)
        self._sync_from_backend()
        env_path = os.environ.get("PLO5BP_TRAINER_STATS")
        self.stats_path = (
            stats_path
            if stats_path is not None
            else Path(env_path) if env_path
            else Path("checkpoints/trainer_stats.json")
        )
        self._load_persisted()

    def _sync_from_backend(self) -> None:
        """`self.model` / `_policy` / `_obs_adapt` ALWAYS mirror the network
        the ACTIVE backend serves (anchor-spec labels, direct callers).

        (review 2026-09-20 H3) `set_backend` did this; `set_format(...,
        backend=gto_host)` did not, so under the GTO PolicyNetHost the MC
        sampled continuations from the format's PPO net (random-init when no
        nlh_stub.pt exists) while every other number came from the PolicyNet.
        Since the coverage follow-up the MC no longer reads `self.model` at
        all: `_rollout_ev` samples each node through the host that serves it
        (`_backend_for`). Hosts without a `.model` leave the PPO model in
        place."""
        m = getattr(self.backend, "model", None)
        if not isinstance(m, torch.nn.Module):
            return
        self.model = m
        self.device = getattr(self.backend, "device", self.device)
        self._policy = model_policy(m, deterministic=False)
        self._obs_adapt = obs_adapter(m)

    def _refresh_ppo_host(self) -> None:
        """Keep `_ppo_host` = a PPO host over the FORMAT's actor. Called
        before `_sync_from_backend` (which repoints `self.device`)."""
        if isinstance(self.backend, PpoSolverHost):
            self._ppo_host = self.backend
        elif self._ppo_host is None or self._ppo_host.model is not self._format_model:
            self._ppo_host = make_ppo_host(self._format_model, self.device)
        self._ppo_obs_adapt = obs_adapter(self._ppo_host.model)

    def _backend_for(self, info: StepInfo) -> tuple[StrategyBackend, str]:
        """`(host, tag)` that serves THIS node.

        (review 2026-09-20, GTO coverage) A teacher net is only valid on the
        table shapes / streets it was trained on (`PolicyNetHost.supports`
        answers from recorded coverage; preflop is never covered). The active
        backend serves a node only if it has no `supports` or says yes;
        otherwise the format's PPO host does and the node is tagged
        BACKEND_PPO_FALLBACK — a river-only teacher never drives preflop or
        multiway nodes, and the payload says which host produced the numbers.
        The choice depends only on PUBLIC node state (seat count, street), so
        opponents stay a pure function of the action prefix. `seats` is the
        TABLE's seat count: a 6-max hand that got heads-up is still a 6-seat
        observation, not the 2-seat form a HU teacher was trained on."""
        be = self.backend
        name = str(getattr(be, "name", "backend"))
        if isinstance(be, PpoSolverHost):
            return be, name
        supports = getattr(be, "supports", None)
        if not callable(supports):
            return be, name
        raw = info.raw_obs or {}
        try:
            ok = bool(supports(
                seats=len(raw.get("stacks") or ()), street=int(raw.get("street", 0))
            ))
        except Exception:  # noqa: BLE001 — a broken host must not kill the hand
            logger.exception("backend %s.supports() failed — using the PPO host", name)
            ok = False
        if ok:
            return be, name
        assert self._ppo_host is not None
        return self._ppo_host, BACKEND_PPO_FALLBACK

    def set_backend(self, backend: StrategyBackend) -> None:
        """Hot-swap the strategy host (T0 PPO ↔ T1 PolicyNet). The critic
        (and its obs projection) stay paired with the format's PPO actor."""
        self.backend = backend
        self._refresh_ppo_host()
        self._sync_from_backend()

    def set_format(
        self,
        variant: str,
        model: ActorCritic,
        critic: CentralCritic | None,
        backend: StrategyBackend | None = None,
    ) -> None:
        """Switch the trainer's game format: swap the served model/critic
        pair, drop the live hand (it belongs to the other game), and
        point `self.settings` at the format's OWN settings object.
        Formats never share or overwrite each other's settings; each
        keeps whatever the user last configured for it. Session/lifetime
        stats keep accumulating across formats.

        `model`/`critic` are the format's PPO pair; `backend` (optional) is
        the host that actually serves the format (e.g. the GTO PolicyNet for
        NLH). A no-op when neither the format nor the host changes, so a
        repeated POST /format can't drop (and re-deal) the live hand."""
        if variant == self.variant and (backend is None or backend is self.backend):
            return
        self.variant = variant
        self.model = model
        self._format_model = model
        self.critic = critic
        self._critic_obs_adapt = obs_adapter(model)
        self._policy = model_policy(model, deterministic=False)
        self._obs_adapt = obs_adapter(model)
        if backend is not None:
            self.backend = backend
        elif isinstance(self.backend, PpoSolverHost):
            self.backend.rebind(model, self.device)
        else:
            # Non-PPO host cannot serve a different format's PPO weights —
            # fall back to a fresh PPO host for this format.
            self.backend = make_ppo_host(model, self.device)
        self._refresh_ppo_host()
        self._sync_from_backend()
        self.hand = None
        self.settings = self.settings_by_variant[variant]

    def set_settings(self, settings: TrainerSettings) -> None:
        """Replace the ACTIVE format's settings (and keep the per-format
        registry in sync — `self.settings` must always be the same object
        as its registry entry)."""
        settings = _cap_settings(settings)
        self.settings_by_variant[self.variant] = settings
        self.settings = settings

    def _anchor_spec(self) -> Any:
        return getattr(self.model, "anchor_spec", PLO_ANCHOR_SPEC)

    def _node_dist(self, obs_np: np.ndarray, info: StepInfo) -> dict[str, Any]:
        """The node distribution of the host that SERVES this node
        (`_backend_for`) as a scorer-ready dict: carries the unclamped
        brackets (H1; re-derived when a producer omits them) and a `backend`
        tag (host name, or BACKEND_PPO_FALLBACK)."""
        host, tag = self._backend_for(info)
        dist = host.node_distribution(obs_np, info).as_dict()
        dist["backend"] = tag
        return attach_unclamped_brackets(dist, info, self._anchor_spec())

    # -- persistence ----------------------------------------------------------

    def _load_persisted(self) -> None:
        try:
            data = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception as e:
            logger.warning("trainer stats file %s unreadable (%s) — starting fresh",
                           self.stats_path, e)
            return
        try:
            self.lifetime_stats = StatsBlock.from_dict(data.get("lifetime", {}))
            # v2 schema: settings stored PER FORMAT. The legacy v1
            # single "settings" key is deliberately DISCARDED (not
            # migrated): a shared-object bug once persisted NLH stakes
            # into it and re-loaded them under PLO5 (0.5bb-ante bomb
            # pots, 2026-07-04) — legacy content can't be trusted to
            # belong to either format, so both restart at defaults.
            by_fmt = data.get("settings_by_format")
            if isinstance(by_fmt, dict):
                # Every format `_persist` writes — incl. the experimental
                # one, which a hardcoded (PLO5, NLH) pair silently dropped on
                # reload. (review 2026-09-20 F10)
                for variant in list(self.settings_by_variant):
                    if isinstance(by_fmt.get(variant), dict):
                        self.settings_by_variant[variant] = _cap_settings(
                            TrainerSettings.model_validate(by_fmt[variant])
                        )
                self.settings = self.settings_by_variant[self.variant]
        except Exception as e:
            logger.warning("trainer stats file %s malformed (%s) — starting fresh",
                           self.stats_path, e)

    def _persist(self) -> None:
        payload = {
            "version": 2,
            "lifetime": self.lifetime_stats.to_dict(),
            "settings_by_format": {
                variant: s.model_dump()
                for variant, s in self.settings_by_variant.items()
            },
        }
        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.stats_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            # Windows: os.replace raises PermissionError while another handle
            # (antivirus / indexer / a concurrent reader) briefly holds the
            # destination. It clears in milliseconds — retry instead of
            # dropping the write. (review 2026-09-20, latent)
            for attempt in range(_PERSIST_RETRIES):
                try:
                    os.replace(tmp, self.stats_path)
                    break
                except PermissionError:
                    if attempt == _PERSIST_RETRIES - 1:
                        raise
                    time.sleep(_PERSIST_RETRY_SLEEP_S)
        except Exception as e:
            logger.warning("failed to persist trainer stats to %s: %s",
                           self.stats_path, e)

    # -- dealing ---------------------------------------------------------------

    def _draw_stacks(self, n: int) -> tuple[int, ...]:
        s = self.settings
        if s.stacks_mode == "fixed":
            return (int(round(s.stack_bb * BB_CHIPS)),) * n
        if s.stacks_mode == "random":
            return tuple(
                int(round(float(self.rng.uniform(s.stack_min_bb, s.stack_max_bb))
                          * BB_CHIPS))
                for _ in range(n)
            )
        # per_seat
        out = []
        for i in range(n):
            lo, hi = s.stacks_per_seat_bb[i]
            v = lo if lo >= hi else float(self.rng.uniform(lo, hi))
            out.append(int(round(v * BB_CHIPS)))
        return tuple(out)

    def new_hand(self, repeat: bool = False) -> list[dict[str, Any]]:
        s = self.settings
        if repeat:
            if self.hand is None:
                raise ValueError("no hand to repeat")
            seed, button = self.hand.seed, self.hand.button
            hero_seat, config = self.hand.hero_seat, self.hand.config
        else:
            if s.seats_mode == "fixed":
                n = s.seats_fixed
            else:
                n = int(self.rng.integers(s.seats_min, s.seats_max + 1))
            hero_seat = int(self.rng.integers(0, n))
            if s.hero_position_mode == "kth":
                k = min(s.hero_kth, n)
                button = (hero_seat - k) % n
            else:
                button = int(self.rng.integers(0, n))
            eng = _engine_variant(self.variant)
            config = GameConfig(
                num_seats=n,
                starting_stack=int(round(s.stack_bb * BB_CHIPS)),
                ante=int(round(s.ante_bb * BB_CHIPS)),
                bb=BB_CHIPS,
                starting_stacks=self._draw_stacks(n),
                # NLH: the 5/10 structure — sb = bb/2, live preflop.
                sb=BB_CHIPS // 2 if eng == VARIANT_NLH else 0,
                variant=eng,
            )
            seed = int(self.rng.integers(0, 2**63 - 1))

        env = BombPotEnv(config)
        obs, info = env.reset(seed, button)
        self.hand_no += 1
        self.hand = HandRecord(
            hand_no=self.hand_no,
            seed=seed,
            button=button,
            hero_seat=hero_seat,
            config=config,
            env=env,
            last_obs=obs,
            last_info=info,
            # Descending card index = rank desc, suit desc within rank.
            all_holes=[sorted(h, reverse=True) for h in env.all_hole_cards()],
            all_holes_dealt=env.all_hole_cards(),
            is_repeat=repeat,
        )
        frames: list[dict[str, Any]] = []
        if env.is_terminal():
            # Everyone all-in from antes at deal time (stack <= ante):
            # the engine runs the boards out during reset.
            self._finalize(np.asarray(env._rs.payouts(), dtype=np.float32))
            frames.append(self.project_state())
        else:
            frames.append(self.project_state())  # fresh deal, pre-action
            self._advance(frames)
        return frames

    def _opp_seed(self, h: HandRecord) -> int:
        return _stable_seed(h.seed, len(h.action_log), 0x5DEECE66D)

    def _advance(self, frames: list[dict[str, Any]] | None = None) -> None:
        """Opponents act (sampled) until it's hero's turn or the hand ends.

        When `frames` is given, a state snapshot (tagged with the action
        that just happened via ``trainer.anim_action``) is appended after
        every opponent step so the client can animate the sequence.
        """
        h = self.hand
        assert h is not None
        while not h.terminal:
            info = h.last_info
            assert info is not None
            if info.actor is None:
                if h.env.is_terminal():
                    self._finalize(
                        np.asarray(h.env._rs.payouts(), dtype=np.float32)
                    )
                break
            actor = int(info.actor)
            moot = _betting_moot(info.raw_obs, actor)
            if actor == h.hero_seat and not moot:
                break
            street = int(info.raw_obs["street"])
            to_call_pre = _to_call_chips(info.raw_obs, actor)
            if moot:
                # All-in runout: no live opponent can respond, so check
                # is the only meaningful action. Auto-check (hero
                # included) rather than asking/sampling; no model RNG
                # is consumed, so behavior stays a pure function of the
                # action prefix.
                gate, chips = GATE_CHECK_CALL, 0
            else:
                # Re-seed per node: opponent behavior is a pure function
                # of the action prefix (see module docstring). Goes
                # through StrategyBackend so T1 PolicyNet swaps cleanly —
                # via the host that COVERS this node (`_backend_for`).
                host, _tag = self._backend_for(info)
                with _TORCH_RNG_LOCK:
                    gate, chips = host.act(
                        h.last_obs,
                        info,
                        deterministic=False,
                        rng_seed=self._opp_seed(h),
                    )
            obs, rewards, done, info2 = h.env.step_hybrid(gate, chips)
            entry = {"seat": actor, "gate": int(gate), "chips": int(chips),
                     "street": street}
            if moot:
                # Engine-forced check at a moot node, NOT a decision: the
                # flag lets the client tell hero's auto-checks from opponent
                # actions and lets review skip grading them (there is no
                # DecisionRecord behind a hero auto-check). Replay reads only
                # gate/chips, so determinism is untouched.
                entry["auto"] = True
            h.action_log.append(entry)
            h.opp_actions_since_hero.append(entry)
            h.last_obs, h.last_info = obs, info2
            if done:
                self._finalize(rewards)
            if frames is not None:
                frame = self.project_state()
                frame["trainer"]["anim_action"] = {
                    "seat": actor,
                    "position": self._position_of(actor),
                    "is_hero": actor == h.hero_seat,
                    "auto": bool(moot),
                    "gate": GATE_SLUGS[int(gate)],
                    "chips": int(chips),
                    "to_call": int(to_call_pre),
                    "street": STREET_NAMES.get(street, str(street)),
                }
                frames.append(frame)

    def _finalize(self, rewards: np.ndarray) -> None:
        h = self.hand
        assert h is not None
        h.terminal = True
        h.rewards_bb = [round(float(r) / h.config.bb, 4) for r in rewards]
        if not h.is_repeat:
            self._commit_hand_stats(h)
            self._persist()

    def _commit_hand_stats(self, h: HandRecord) -> None:
        """Fold a COMPLETED hand into the session + lifetime blocks.

        (review 2026-09-20 H2/H4) Stats are committed per HAND, at hand end,
        not per move:
        - the EV-loss estimate replays the real deal (villain cards + the
          real future board), so a running `ev_loss_total_bb` that ticked
          after each move leaked it mid-hand through the stats panel;
        - a hand abandoned mid-way (New hand / Repeat / format switch while
          live) used to leave its moves + EV loss in the sums without ever
          counting as a hand, inflating `ev_loss_per_hand_bb`. POLICY: an
          abandoned hand contributes NOTHING — hands, moves, score, category
          counts and EV loss always describe the same set of completed
          hands. (Repeat hands stay excluded entirely, as before.)
        """
        if h.stats_committed:
            return
        h.stats_committed = True
        for block in (self.session_stats, self.lifetime_stats):
            block.hands += 1
            for d in h.decisions:
                block.add_decision(d)

    # -- acting / scoring --------------------------------------------------------

    def act(self, gate_slug: str, chips_req: int | None) -> list[dict[str, Any]]:
        h = self.hand
        if h is None:
            # 409, not an implicit deal: the client's action was chosen for a
            # hand this session no longer has (restart / LRU eviction /
            # format switch). (review 2026-09-20 F9)
            raise HTTPException(status_code=409, detail="no active hand")
        if h.terminal:
            raise HTTPException(status_code=400, detail="hand is over")
        info = h.last_info
        assert info is not None and h.last_obs is not None
        if info.actor != h.hero_seat:
            raise HTTPException(status_code=400, detail="not hero's turn")

        gate_idx = GATE_NAME_TO_IDX[gate_slug]
        if not bool(info.gate_mask[gate_idx]):
            raise HTTPException(status_code=400, detail=f"gate {gate_slug!r} not legal")
        chips = 0
        if gate_idx == GATE_RAISE:
            if chips_req is None:
                raise HTTPException(status_code=400,
                                    detail="chips required for gate 'raise'")
            chips = int(chips_req)
            lo = int(info.min_raise_chips)
            hi = int(info.max_raise_chips)
            if not (lo <= chips <= hi):
                raise HTTPException(
                    status_code=400,
                    detail=f"chips {chips} out of raise range [{lo}, {hi}]",
                )

        dist = self._node_dist(h.last_obs, info)
        if dist["head_version"] >= 2:
            sc = score_move_v2(dist, gate_idx, chips)
            rec_alpha, rec_beta = _rec_refine_params(dist)
        else:
            sc = score_move(
                dist["gate_probs"], dist["alpha"], dist["beta"],
                dist["min_chips"], dist["max_chips"], gate_idx, chips,
            )
            rec_alpha, rec_beta = dist["alpha"], dist["beta"]
        raw = info.raw_obs
        decision = DecisionRecord(
            decision_idx=len(h.decisions),
            street=int(raw["street"]),
            action_log_idx=len(h.action_log),
            gate_probs=dist["gate_probs"],
            alpha=rec_alpha,
            beta=rec_beta,
            min_chips=dist["min_chips"],
            max_chips=dist["max_chips"],
            rec_gate=dist["rec_gate"],
            rec_chips=dist["rec_chips"],
            value_bb=dist["value_bb"],
            pot_chips=int(raw["pot"]),
            to_call_chips=_to_call_chips(raw, h.hero_seat),
            hero_committed_chips=int(raw["total_commit"][h.hero_seat]),
            actor_commit_chips=int(raw["street_commit"][h.hero_seat]),
            user_gate=gate_idx,
            user_chips=chips,
            gate_ratio=sc["gate_ratio"],
            size_q=sc["size_q"],
            score=sc["score"],
            category=sc["category"],
            head_version=dist["head_version"],
            anchor_probs=dist.get("anchor_probs"),
            anchor_chips=dist.get("anchor_chips"),
            anchor_legal=dist.get("anchor_legal"),
            rec_anchor=dist.get("rec_anchor"),
            user_anchor=sc.get("user_anchor"),
            pot_ref_chips=int(dist.get("pot_ref_chips", 0)),
            backend=str(dist.get("backend", "ppo")),
        )

        street = int(raw["street"])
        obs, rewards, done, info2 = h.env.step_hybrid(gate_idx, chips)
        h.action_log.append({"seat": int(h.hero_seat), "gate": int(gate_idx),
                             "chips": int(chips), "street": street})
        h.opp_actions_since_hero = []
        h.last_obs, h.last_info = obs, info2
        # Record the decision (+ its EV-loss estimate and feedback) BEFORE
        # building any terminal frame. The terminal frame's review block is
        # gated on the decision being in h.decisions; a multiway all-in run-out
        # snapshots its terminal frame INSIDE _advance — i.e. before the old
        # post-branch append — so it shipped `review: null`, and the review pane
        # only ever opened via the post-animation applyState(final). On long
        # run-out animations the client's animSeq abort skips that settle and
        # the review never appeared. Recording first makes the terminal frame
        # self-sufficient. (Non-terminal frames stay review-less — the review
        # block is also gated on h.terminal.) It must also precede _finalize:
        # the hand's stats are committed there from h.decisions.
        self._estimate_ev_loss(decision)
        h.decisions.append(decision)
        frames: list[dict[str, Any]] = []
        if done:
            self._finalize(rewards)
            frames.append(self.project_state())
        else:
            frames.append(self.project_state())  # hero's action landed
            self._advance(frames)
        # Stats are committed per hand in _finalize (see _commit_hand_stats).
        return frames

    def _feedback_payload(
        self, d: DecisionRecord, reveal_ev: bool = True
    ) -> dict[str, Any]:
        """Flash payload for decision `d`.

        (review 2026-09-20 H2) The EV-loss estimate replays the REAL deal —
        villains' hole cards and the real future board — so while the hand is
        live it is a hidden-information channel (deviate, read the number,
        infer who is strong / what is coming). `reveal_ev=False` withholds it:
        `ev_loss_bb` is None and `ev_loss_hidden` says a value exists and will
        show at hand end. Score / category / labels are observation-only and
        are never withheld."""
        bb = BB_CHIPS
        has_ev = d.ev_loss_bb is not None
        return {
            "decision_idx": d.decision_idx,
            "category": d.category,
            "marks": CATEGORY_MARKS[d.category],
            "score": round(d.score, 1),
            "label": _action_label(d.user_gate, d.user_chips, d.to_call_chips),
            "rec_label": _action_label(d.rec_gate, d.rec_chips, d.to_call_chips),
            "rec_chips_bb": round(chips_to_bb(d.rec_chips, bb), 2)
            if d.rec_gate == GATE_RAISE else None,
            # Raw gate slugs + chip amounts so the client can format the
            # labels in the active display unit ($ vs bb); the *_label
            # strings above are bb-only fallbacks.
            "user_gate": GATE_SLUGS[d.user_gate],
            "user_chips": int(d.user_chips) if d.user_gate == GATE_RAISE else None,
            "rec_gate": GATE_SLUGS[d.rec_gate],
            "rec_chips": int(d.rec_chips) if d.rec_gate == GATE_RAISE else None,
            "to_call_chips": int(d.to_call_chips),
            # Hero's street commit before the action (raise-TO = chips + this)
            # and the decision's street name — both observation-only.
            "actor_commit_chips": int(d.actor_commit_chips),
            "street": STREET_NAMES.get(d.street, str(d.street)),
            # Which host graded this decision ("ppo_fallback" = the active GTO
            # host does not cover this node; the PPO net served it).
            "backend": d.backend,
            "ev_loss_bb": round(d.ev_loss_bb, 3) if (has_ev and reveal_ev) else None,
            "ev_loss_hidden": bool(has_ev and not reveal_ev),
        }

    # -- Monte-Carlo EV loss -------------------------------------------------------

    def _candidates_equal(self, d: DecisionRecord) -> bool:
        if d.user_gate != d.rec_gate:
            return False
        if d.user_gate != GATE_RAISE:
            return True
        if d.head_version >= 2:
            # Same chosen anchor + chips within tolerance. Short-shove /
            # single-anchor nodes collapse to the same anchor naturally.
            if d.user_anchor != d.rec_anchor:
                return False
            tol = max(1, (d.max_chips - d.min_chips) // 100)
            return abs(d.user_chips - d.rec_chips) <= tol
        if d.min_chips == 0 or d.min_chips >= d.max_chips:
            return True  # chips moot (short shove / single point)
        tol = max(1, (d.max_chips - d.min_chips) // 100)
        return abs(d.user_chips - d.rec_chips) <= tol

    def _estimate_ev_loss(self, d: DecisionRecord) -> None:
        h = self.hand
        assert h is not None
        # Effective rollout count: the public build caps it server-side even
        # if a larger value is somehow stored. (review 2026-09-20 F5/F7)
        n = min(int(self.settings.mc_rollouts), mc_rollouts_cap())
        if n <= 0:
            return
        if self._candidates_equal(d):
            d.ev_loss_bb = 0.0
            d.ev_loss_signed_bb = 0.0
            return
        node_seed = _stable_seed(h.seed, d.action_log_idx, 0xEC0FFEE)
        prefix = h.action_log[: d.action_log_idx]
        ev_user = self._rollout_ev(h, prefix, d.user_gate, d.user_chips, n, node_seed)
        ev_best = self._rollout_ev(h, prefix, d.rec_gate, d.rec_chips, n, node_seed)
        # Forward-facing EV: rebase by the chips already committed at this
        # node (sunk), so a fold reads as 0 EV instead of -(committed). The
        # offset is identical for both candidates, so the loss is unchanged.
        committed_bb = chips_to_bb(d.hero_committed_chips, h.config.bb)
        d.ev_user_bb = round(ev_user + committed_bb, 4)
        d.ev_best_bb = round(ev_best + committed_bb, 4)
        # Signed estimate feeds the aggregates; the per-decision display value
        # stays clamped at zero. (review 2026-09-20 H4)
        d.ev_loss_signed_bb = round(ev_best - ev_user, 4)
        d.ev_loss_bb = round(max(0.0, ev_best - ev_user), 4)

    def _rollout_ev(
        self,
        h: HandRecord,
        prefix: list[dict[str, int]],
        gate: int,
        chips: int,
        n: int,
        node_seed: int,
    ) -> float:
        """Mean hero payoff (bb) over `n` network-vs-network continuations
        of this deal after `prefix` + the candidate action. Same
        `node_seed` for both candidates = common random numbers.

        (review 2026-09-20 F7) `_TORCH_RNG_LOCK` is process-global — every
        user's opponent sampling queues behind it — and used to be held for
        the WHOLE arm, including the n env resets / prefix replays /
        candidate steps and every continuation env step, none of which touch
        torch's RNG. Now only the seed→sample regions hold it: the env work
        runs first, and each lockstep depth takes the lock just for
        (restore our RNG stream → sample → snapshot the stream). Restoring
        the snapshot makes the draws exactly the uninterrupted
        `manual_seed(node_seed)` stream no matter what other threads seed in
        between, so values are bit-identical to the hold-the-lock version
        (pinned by tests/python/test_review_trainer_session.py).

        (review 2026-09-20, GTO serve==train) Continuations are sampled by
        the host that SERVES each node (`_backend_for`), the same one that
        recommends and drives the live opponents there:
        - PPO-served nodes (the PPO backend, or the fallback for nodes a GTO
          host does not cover) go through ONE batched `model.act` per depth —
          the path pinned bit-identical above;
        - nodes served by a non-PPO host go through `host.act(...)` one env
          at a time, so the host's own serving rules apply (PolicyNetHost:
          canonical re-encoded obs, grid chips, jam snap). Sampling the raw
          network on the live obs here played a DIFFERENT policy from the one
          that produced the recommendation being graded.
        Within a depth the draw order is fixed (the PPO batch, then the hosted
        envs in rollout order), all inside one lock region."""
        hero = h.hero_seat
        total = 0.0
        live: list[list[Any]] = []  # [env, obs, info]
        for i in range(n):
            # EV runouts: grade all-in continuations by expected value over
            # board runouts instead of one sampled runout — same rollout
            # count, much less estimator noise. (The live hand's displayed
            # result stays realized; only this estimator uses EV.)
            env = BombPotEnv(h.config, ev_runout_samples=32)
            obs, info = env.reset(h.seed, h.button)
            for a in prefix:
                obs, _, _, info = env.step_hybrid(a["gate"], a["chips"])
            obs, rewards, done, info = env.step_hybrid(gate, chips)
            if not done:
                live.append([env, obs, info])
                continue
            total += float(rewards[hero])
            if i == 0:
                # The replay is a pure function of (config, seed, button,
                # prefix, candidate): if the candidate ends the hand (a fold,
                # a closing call) all n copies end identically — add the same
                # reward n-1 more times instead of replaying them.
                for _ in range(n - 1):
                    total += float(rewards[hero])
                break

        # Lockstep: one sampling region per depth across all live rollouts.
        rng_state: tuple[torch.Tensor, torch.Tensor | None] | None = None
        rng_device = self.device
        while live:
            hosts = [self._backend_for(x[2])[0] for x in live]
            batch_idx = [
                i for i, hst in enumerate(hosts) if isinstance(hst, PpoSolverHost)
            ]
            hosted_idx = [
                i for i, hst in enumerate(hosts) if not isinstance(hst, PpoSolverHost)
            ]
            batch_in = None
            if batch_idx:
                # `_backend_for` only ever yields ONE PPO host: `_ppo_host`
                # (the PPO backend itself, or the format's fallback host).
                ppo = self._ppo_host
                assert ppo is not None and hosts[batch_idx[0]] is ppo
                rows = [live[i] for i in batch_idx]
                batch_in = (
                    ppo.model,
                    torch.from_numpy(
                        self._ppo_obs_adapt(np.stack([x[1] for x in rows]))
                    ).to(ppo.device),
                    torch.from_numpy(
                        np.stack([x[2].gate_mask for x in rows])
                    ).to(ppo.device),
                    # (B, 4) sizing context — v1 models slice [..., :2], v2
                    # needs all four columns for the anchor grid.
                    torch.from_numpy(
                        np.stack([sizing_from_info(x[2]) for x in rows])
                    ).to(ppo.device),
                )
            actions: list[tuple[int, int] | None] = [None] * len(live)
            with _TORCH_RNG_LOCK:
                if rng_state is None:
                    torch.manual_seed(node_seed)
                else:
                    _restore_rng_state(rng_state, rng_device)
                if batch_in is not None:
                    model_b, obs_b, gm_b, sizing_b = batch_in
                    with torch.no_grad():
                        _mc_out = model_b.act(
                            obs_b, gm_b, sizing_b, deterministic=False
                        )
                    for j, g, c in zip(
                        batch_idx, _mc_out.gate.tolist(), _mc_out.chips.tolist()
                    ):
                        actions[j] = (int(g), int(c))
                for j in hosted_idx:
                    # rng_seed=None: continue OUR stream, never re-seed.
                    g, c = hosts[j].act(
                        live[j][1], live[j][2], deterministic=False, rng_seed=None
                    )
                    actions[j] = (int(g), int(c))
                rng_state = _snapshot_rng_state(rng_device)
            nxt: list[list[Any]] = []
            for x, action in zip(live, actions):
                assert action is not None
                obs2, rewards, done, info2 = x[0].step_hybrid(action[0], action[1])
                if done:
                    total += float(rewards[hero])
                else:
                    nxt.append([x[0], obs2, info2])
            live = nxt
        return total / n / h.config.bb

    # -- review / what-if ------------------------------------------------------------

    def _replay_to_node(
        self, node_idx: int
    ) -> tuple[BombPotEnv, np.ndarray, StepInfo]:
        """Replay the recorded action_log up to (not including) `node_idx`
        — leaving the engine at the node whose actor is
        `action_log[node_idx]["seat"]`. The encoder is actor-rotated, so
        the returned obs is that actor's observation."""
        h = self.hand
        assert h is not None
        env = BombPotEnv(h.config)
        obs, info = env.reset(h.seed, h.button)
        for a in h.action_log[:node_idx]:
            obs, _, _, info = env.step_hybrid(a["gate"], a["chips"])
        return env, obs, info

    def _replay_to_decision(
        self, d: DecisionRecord
    ) -> tuple[BombPotEnv, np.ndarray, StepInfo]:
        return self._replay_to_node(d.action_log_idx)

    def _decision_for(self, idx: int) -> DecisionRecord:
        h = self.hand
        if h is None or not h.terminal:
            raise HTTPException(status_code=400,
                                detail="review available after the hand ends")
        if not h.decisions:
            raise HTTPException(status_code=400, detail="no decisions this hand")
        if not (0 <= idx < len(h.decisions)):
            idx = max(0, min(len(h.decisions) - 1, idx))
        return h.decisions[idx]

    def _hero_by_node(self) -> dict[int, DecisionRecord]:
        """Map action_log index -> hero DecisionRecord. Keys are unique:
        `action_log_idx` is captured as len(action_log) right before each
        hero action is appended."""
        h = self.hand
        assert h is not None
        return {d.action_log_idx: d for d in h.decisions}

    def _node_index(self) -> list[dict[str, Any]]:
        """One lightweight row per action_log entry (every seat's
        decision, in order) — drives the review stepper and the pill ->
        node mapping."""
        h = self.hand
        assert h is not None
        hero_by_node = self._hero_by_node()
        rows: list[dict[str, Any]] = []
        for i, a in enumerate(h.action_log):
            seat = int(a["seat"])
            d = hero_by_node.get(i)
            rows.append({
                "node_idx": i,
                "seat": seat,
                "position": self._position_of(seat),
                "is_hero": seat == h.hero_seat,
                "street": STREET_NAMES.get(int(a["street"]), str(a["street"])),
                "actual_gate": GATE_SLUGS[int(a["gate"])],
                "actual_chips": int(a["chips"]),
                "decision_idx": d.decision_idx if d is not None else None,
                # Engine-forced check at a moot node (hero or villain): not
                # a decision — nothing to grade, no DecisionRecord.
                "auto": bool(a.get("auto", False)),
            })
        return rows

    def _node_view(
        self, node_idx: int, obs_np: np.ndarray, info: StepInfo
    ) -> dict[str, Any]:
        """Detailed view of one decision node (hero OR villain): the
        model's policy + both EV estimates + the actor's actual action.
        Same key shape as `_hero_current` so the frontend renders both
        uniformly. Hero nodes additionally overlay the stored record's
        scoring / MC EV-loss (a fresh forward can't reproduce the MC)."""
        h = self.hand
        assert h is not None
        bb = BB_CHIPS
        a = h.action_log[node_idx]
        actor = int(info.actor) if info.actor is not None else int(a["seat"])
        raw = info.raw_obs
        to_call = _to_call_chips(raw, actor)
        dist = self._node_dist(obs_np, info)

        # own EV = the actor's observation-only value head (blind to
        # opponents' cards); true EV = the centralized critic (sees all
        # hole cards), built with the EXACT training convention via the
        # canonical rollout helpers so the number is meaningful.
        value_true_bb: float | None = None
        if self.critic is not None and dist["head_version"] >= 2:
            holes = np.asarray(h.all_holes_dealt, dtype=np.uint8)  # (S, 5)
            opp = _rotate_opp_holes(holes, actor)[None]            # (1, 5, 5)
            # Match the critic's trained obs width (full / prefix / minimal).
            # Env always emits full OBS_DIM; without the adapter a 796-d
            # experimental critic gets 1171+opp and matmul-crashes (500).
            # The CRITIC's adapter (paired with the format's PPO actor), not
            # the served actor's: under a GTO host the two can differ (H3).
            crit_obs = np.asarray(
                self._critic_obs_adapt(obs_np), dtype=np.float32
            )
            value_true_bb = round(float(_critic_values(
                self.critic, self.device,
                crit_obs[None], opp,
            )[0]), 4)

        actual_gate = int(a["gate"])
        actual_chips = int(a["chips"])
        nc: dict[str, Any] = {
            "node_idx": node_idx,
            "seat": actor,
            "position": self._position_of(actor),
            "is_hero": actor == h.hero_seat,
            "street": STREET_NAMES.get(int(raw["street"]), str(raw["street"])),
            "gate_probs": [round(p, 4) for p in dist["gate_probs"]],
            "rec_gate": GATE_SLUGS[dist["rec_gate"]],
            "rec_chips": dist["rec_chips"] if dist["rec_gate"] == GATE_RAISE else None,
            "rec_chips_bb": round(chips_to_bb(dist["rec_chips"], bb), 4)
            if dist["rec_gate"] == GATE_RAISE else None,
            "rec_label": _action_label(dist["rec_gate"], dist["rec_chips"], to_call),
            "value_bb": round(dist["value_bb"], 4),
            "value_true_bb": value_true_bb,
            "to_call_chips": to_call,
            # The ACTING seat's street commit before its action, read off the
            # replayed engine state (exact for any seat, incl. short blinds).
            "actor_commit_chips": int(raw["street_commit"][actor]),
            "pot_chips": int(raw["pot"]),
            "actual_gate": GATE_SLUGS[actual_gate],
            "actual_chips": actual_chips if actual_gate == GATE_RAISE else None,
            "actual_label": _action_label(actual_gate, actual_chips, to_call),
            "auto": bool(a.get("auto", False)),
            # Host whose policy/EV this view shows: the active backend's
            # name, or "ppo_fallback" when it does not cover this node.
            "backend": dist.get("backend"),
        }
        if dist["head_version"] >= 2:
            nc["head_version"] = 2
            nc["pot_ref_chips"] = dist.get("pot_ref_chips")
            nc["anchors"] = _anchors_payload(
                dist["anchor_probs"], dist["anchor_chips"],
                dist["anchor_legal"], bb,
                spec=getattr(self.model, "anchor_spec", PLO_ANCHOR_SPEC),
            )
            nc["rec_anchor"] = dist["rec_anchor"]
            nc["mixture"] = dist.get("mixture")
            # Mark where the actor's ACTUAL raise landed (the ● on the EQ
            # bars) with the scorer's own snap, so hero and villain markers
            # follow one rule (the anchor that can produce those chips).
            if actual_gate == GATE_RAISE:
                nc["user_anchor"] = snap_to_anchor(dist, actual_chips)[0]
            else:
                nc["user_anchor"] = None

        d = self._hero_by_node().get(node_idx)
        if d is not None:
            nc.update({
                "decision_idx": d.decision_idx,
                "user_gate": GATE_SLUGS[d.user_gate],
                "user_chips": d.user_chips if d.user_gate == GATE_RAISE else None,
                "user_chips_bb": round(chips_to_bb(d.user_chips, bb), 4)
                if d.user_gate == GATE_RAISE else None,
                "user_label": _action_label(
                    d.user_gate, d.user_chips, d.to_call_chips
                ),
                "score": round(d.score, 1),
                "category": d.category,
                "marks": CATEGORY_MARKS[d.category],
                "gate_ratio": round(d.gate_ratio, 4),
                "size_q": round(d.size_q, 4),
                "ev_user_bb": d.ev_user_bb,
                "ev_best_bb": d.ev_best_bb,
                "ev_loss_bb": d.ev_loss_bb,
                "ev_loss_signed_bb": d.ev_loss_signed_bb,
            })
            if d.head_version >= 2:
                nc["user_anchor"] = d.user_anchor
        elif actor == h.hero_seat:
            # Hero node with NO DecisionRecord = a moot auto-check (`auto`).
            # Send the graded keys as explicit nulls so the client can render
            # "auto-check" instead of dereferencing a missing category.
            nc.update({
                "decision_idx": None,
                "user_gate": GATE_SLUGS[actual_gate],
                "user_chips": None,
                "user_chips_bb": None,
                "user_label": nc["actual_label"],
                "score": None,
                "category": None,
                "marks": None,
                "gate_ratio": None,
                "size_q": None,
                "ev_user_bb": None,
                "ev_best_bb": None,
                "ev_loss_bb": None,
                "ev_loss_signed_bb": None,
            })
        return nc

    def _hero_current(self, current_idx: int) -> dict[str, Any] | None:
        """The detailed view of hero decision `current_idx` (the pill
        detail panel). None when the hand had no hero decisions."""
        h = self.hand
        assert h is not None
        if not h.decisions:
            return None
        bb = BB_CHIPS
        d = h.decisions[current_idx]
        current = {
            "decision_idx": d.decision_idx,
            "street": STREET_NAMES.get(d.street, str(d.street)),
            "gate_probs": [round(p, 4) for p in d.gate_probs],
            "beta_alpha": round(d.alpha, 4),
            "beta_beta": round(d.beta, 4),
            "min_chips": d.min_chips,
            "max_chips": d.max_chips,
            "rec_gate": GATE_SLUGS[d.rec_gate],
            "rec_chips": d.rec_chips if d.rec_gate == GATE_RAISE else None,
            "rec_chips_bb": round(chips_to_bb(d.rec_chips, bb), 4)
            if d.rec_gate == GATE_RAISE else None,
            "rec_label": _action_label(d.rec_gate, d.rec_chips, d.to_call_chips),
            "value_bb": round(d.value_bb, 4),
            "user_gate": GATE_SLUGS[d.user_gate],
            "user_chips": d.user_chips if d.user_gate == GATE_RAISE else None,
            "user_chips_bb": round(chips_to_bb(d.user_chips, bb), 4)
            if d.user_gate == GATE_RAISE else None,
            "user_label": _action_label(d.user_gate, d.user_chips, d.to_call_chips),
            "score": round(d.score, 1),
            "category": d.category,
            "marks": CATEGORY_MARKS[d.category],
            "gate_ratio": round(d.gate_ratio, 4),
            "size_q": round(d.size_q, 4),
            "to_call_chips": d.to_call_chips,
            "actor_commit_chips": d.actor_commit_chips,
            "backend": d.backend,
            "pot_chips": d.pot_chips,
            "ev_user_bb": d.ev_user_bb,
            "ev_best_bb": d.ev_best_bb,
            "ev_loss_bb": d.ev_loss_bb,
            "ev_loss_signed_bb": d.ev_loss_signed_bb,
        }
        if d.head_version >= 2:
            current["head_version"] = 2
            current["pot_ref_chips"] = d.pot_ref_chips
            current["anchors"] = _anchors_payload(
                d.anchor_probs, d.anchor_chips, d.anchor_legal, bb,
                spec=getattr(self.model, "anchor_spec", PLO_ANCHOR_SPEC),
            )
            current["rec_anchor"] = d.rec_anchor
            current["user_anchor"] = d.user_anchor
        return current

    def review_block(
        self, current_idx: int, whatif: dict[str, Any] | None = None,
        *, node_idx: int | None = None,
        node_current: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        h = self.hand
        assert h is not None
        decisions = [
            {
                "decision_idx": d.decision_idx,
                "node_idx": d.action_log_idx,
                "street": STREET_NAMES.get(d.street, str(d.street)),
                "category": d.category,
                "score": round(d.score, 1),
                "user_label": _action_label(d.user_gate, d.user_chips,
                                            d.to_call_chips),
                # Exact sizing context for the pill label (raise-TO total =
                # the node's raise-BY chips + actor_commit_chips).
                "to_call_chips": int(d.to_call_chips),
                "actor_commit_chips": int(d.actor_commit_chips),
                "backend": d.backend,
                "ev_loss_bb": round(d.ev_loss_bb, 3)
                if d.ev_loss_bb is not None else None,
            }
            for d in h.decisions
        ]
        scores = [d.score for d in h.decisions]
        hand_score = round(sum(scores) / len(scores), 1) if scores else None
        current = self._hero_current(current_idx)
        nodes = self._node_index()
        # `node` is the shared cursor (action_log index). Pills and arrows
        # both drive it; the hero path syncs it to the current decision.
        if node_idx is not None:
            node_val = node_idx
        elif h.decisions:
            node_val = h.decisions[current_idx].action_log_idx
        else:
            node_val = 0
        return {
            "active": True,
            "decision": current_idx,
            "num_decisions": len(h.decisions),
            "hand_score": hand_score,
            "decisions": decisions,
            "current": current,
            "whatif": whatif,
            "nodes": nodes,
            "num_nodes": len(nodes),
            "node": node_val,
            "node_current": node_current,
        }

    def review_at_node(self, node_idx: int) -> dict[str, Any]:
        """Project the hand at decision node `node_idx` (any seat). The
        review carries `node_current` (the node detail) and keeps the
        hero pill list in sync via the shared `node` cursor."""
        h = self.hand
        if h is None or not h.terminal:
            raise HTTPException(status_code=400,
                                detail="review available after the hand ends")
        if not h.action_log:
            raise HTTPException(status_code=400, detail="no decisions this hand")
        node_idx = max(0, min(len(h.action_log) - 1, int(node_idx)))
        env, obs, info = self._replay_to_node(node_idx)
        nc = self._node_view(node_idx, obs, info)
        cur_dec = self._hero_by_node().get(node_idx)
        if cur_dec is not None:
            current_idx = cur_dec.decision_idx
        else:
            current_idx = len(h.decisions) - 1 if h.decisions else 0
        review = self.review_block(
            current_idx, node_idx=node_idx, node_current=nc,
        )
        return self.project_state(
            env=env, info=info, reveal=True, review=review,
        )

    def _original_card_spec(self, d: DecisionRecord) -> dict[str, list[int | None]]:
        """Cards as visible at decision `d`'s node (turn/river None until
        revealed). Sliced from the terminal boards — boards only grow."""
        h = self.hand
        assert h is not None
        raw = dict(h.env._rs.observation_dict())
        ba = [int(c) for c in raw["board_a"]]
        bb_ = [int(c) for c in raw["board_b"]]
        street = d.street
        if _engine_variant(self.variant) == VARIANT_NLH:
            return {
                # Dealt order, not display order: an unmodified what-if
                # must reproduce the original observation bit-exactly.
                "hero_hole": list(h.all_holes_dealt[h.hero_seat]),
                "flop_a": ba[:3] if street >= 1 else [None, None, None],
                "flop_b": [],
                "turn": [ba[3] if street >= 2 and len(ba) > 3 else None],
                "river": [ba[4] if street >= 3 and len(ba) > 4 else None],
            }
        return {
            # Dealt order, not display order: an unmodified what-if must
            # reproduce the original observation bit-exactly.
            "hero_hole": list(h.all_holes_dealt[h.hero_seat]),
            "flop_a": ba[:3],
            "flop_b": bb_[:3],
            "turn": [
                ba[3] if street >= 2 and len(ba) > 3 else None,
                bb_[3] if street >= 2 and len(bb_) > 3 else None,
            ],
            "river": [
                ba[4] if street >= 3 and len(ba) > 4 else None,
                bb_[4] if street >= 3 and len(bb_) > 4 else None,
            ],
        }

    def whatif(self, req: WhatifRequest) -> dict[str, Any]:
        """Recompute the network's view of decision `req.decision` under a
        modified card spec. Returns a full state dict (review projection
        with the modified cards on the table)."""
        h = self.hand
        d = self._decision_for(req.decision)
        assert h is not None
        spec = self._original_card_spec(d)

        is_nlh = _engine_variant(self.variant) == VARIANT_NLH
        overrides = {
            "hero_hole": (req.hero_hole, 2 if is_nlh else 5),
            "flop_a": (req.flop_a, 3),
            "flop_b": (req.flop_b, 0 if is_nlh else 3),
            "turn": (req.turn, 1 if is_nlh else 2),
            "river": (req.river, 1 if is_nlh else 2),
        }
        for key, (xs, length) in overrides.items():
            if xs is None:
                continue
            try:
                cleaned = validate_card_list(xs, length, key)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            for i, c in enumerate(cleaned):
                if c is None:
                    continue
                if spec[key][i] is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{key}[{i}] is not revealed at this decision",
                    )
                spec[key][i] = c

        seen: set[int] = set()
        for key in ("hero_hole", "flop_a", "flop_b", "turn", "river"):
            for c in spec[key]:
                if c is None:
                    continue
                if c in seen:
                    raise HTTPException(status_code=400,
                                        detail=f"duplicate card {c} in what-if spec")
                seen.add(c)

        env = BombPotEnv(h.config)
        obs, info = _study_replay(
            env, h.button, h.hero_seat, spec, h.action_log[: d.action_log_idx],
            variant=_engine_variant(self.variant),
        )
        if info.actor != h.hero_seat:
            raise HTTPException(
                status_code=500,
                detail="what-if replay did not reach hero's decision node",
            )
        dist = self._node_dist(obs, info)
        if dist["head_version"] >= 2:
            rescored = score_move_v2(dist, d.user_gate, d.user_chips)
        else:
            rescored = score_move(
                dist["gate_probs"], dist["alpha"], dist["beta"],
                dist["min_chips"], dist["max_chips"], d.user_gate, d.user_chips,
            )
        bb = BB_CHIPS
        rec_chips_out = dist["rec_chips"] if dist["rec_gate"] == GATE_RAISE else None
        recommendation: dict[str, Any] = {
            "gate": GATE_SLUGS[dist["rec_gate"]],
            "gate_name": GATE_NAMES[dist["rec_gate"]],
            "chips": rec_chips_out,
            "chips_bb": round(chips_to_bb(rec_chips_out, bb), 4)
            if rec_chips_out is not None else None,
            "value_bb": round(dist["value_bb"], 4),
            "gate_distribution": [round(p, 4) for p in dist["gate_probs"]],
            "backend": dist.get("backend"),
        }
        if dist["head_version"] >= 2:
            rec_alpha, rec_beta = _rec_refine_params(dist)
            recommendation["head_version"] = 2
            recommendation["pot_ref_chips"] = dist.get("pot_ref_chips")
            recommendation["anchors"] = _anchors_payload(
                dist["anchor_probs"], dist["anchor_chips"],
                dist["anchor_legal"], bb,
                spec=getattr(self.model, "anchor_spec", PLO_ANCHOR_SPEC),
            )
            recommendation["rec_anchor"] = dist["rec_anchor"]
            recommendation["mixture"] = dist.get("mixture")
            recommendation["refine"] = (
                {"alpha": round(rec_alpha, 4), "beta": round(rec_beta, 4)}
                if dist["refine_ok"][dist["rec_anchor"]] else None
            )
        else:
            recommendation["beta_alpha"] = round(dist["alpha"], 4)
            recommendation["beta_beta"] = round(dist["beta"], 4)
        # Display copy keeps the hole-card sort invariant; the replay above
        # used dealt order so an unmodified what-if stays bit-exact.
        display_spec = {k: list(v) for k, v in spec.items()}
        display_spec["hero_hole"] = sorted(
            display_spec["hero_hole"], key=lambda c: -1 if c is None else c,
            reverse=True,
        )
        whatif_block = {
            "card_spec": display_spec,
            "recommendation": recommendation,
            "rescored": {
                "score": round(rescored["score"], 1),
                "category": rescored["category"],
            },
        }
        review = self.review_block(d.decision_idx, whatif=whatif_block)
        return self.project_state(
            env=env, info=info, reveal=True, review=review,
            card_spec_override=display_spec,
        )

    # -- projection -------------------------------------------------------------------

    def _position_of(self, seat: int) -> str:
        h = self.hand
        assert h is not None
        return position_name(seat, h.button, h.config.num_seats, None)

    def project_state(
        self,
        env: BombPotEnv | None = None,
        info: StepInfo | None = None,
        reveal: bool | None = None,
        review: dict[str, Any] | None = None,
        card_spec_override: dict[str, list[int | None]] | None = None,
    ) -> dict[str, Any]:
        """Project the trainer state in the study `_state_dict` shape (so
        the existing frontend renderers work) plus a `trainer` block.

        With no args, projects the live hand frontier. Review/what-if
        callers pass a replayed env + the review block.
        """
        h = self.hand
        assert h is not None
        live = env is None
        if env is None:
            env = h.env
        if info is None:
            info = h.last_info
        cfg = h.config
        bb = cfg.bb
        raw = dict(env._rs.observation_dict())

        actor_raw = raw.get("actor")
        actor = int(actor_raw) if actor_raw is not None else None

        terminal: str | None = None
        terminal_message: str | None = None
        if live and h.terminal:
            alive = sum(1 for f in raw["folded"] if not f)
            terminal = "fold_out" if alive <= 1 else "showdown"
            hero_delta = (h.rewards_bb or [0.0] * cfg.num_seats)[h.hero_seat]
            if hero_delta > 0:
                result = f"hero won {hero_delta:.2f}bb"
            elif hero_delta < 0:
                result = f"hero lost {abs(hero_delta):.2f}bb"
            else:
                result = "hero broke even"
            terminal_message = (
                "All opponents folded — uncontested pot. "
                if terminal == "fold_out" else "Hand complete — "
            ) + result + "."

        if reveal is None:
            reveal = h.terminal

        seats: list[dict[str, Any]] = []
        for seat in range(cfg.num_seats):
            if seat == h.hero_seat:
                hole: list[int] | None = (
                    card_spec_override["hero_hole"]  # type: ignore[assignment]
                    if card_spec_override is not None
                    else list(h.all_holes[seat])
                )
            elif reveal:
                hole = list(h.all_holes[seat])
            else:
                hole = None
            seats.append({
                "seat": seat,
                "position": self._position_of(seat),
                "stack_chips": int(raw["stacks"][seat]),
                "stack_bb": round(chips_to_bb(int(raw["stacks"][seat]), bb), 4),
                "committed_this_street_bb": round(
                    chips_to_bb(int(raw["street_commit"][seat]), bb), 4
                ),
                "committed_total_bb": round(
                    chips_to_bb(int(raw["total_commit"][seat]), bb), 4
                ),
                "committed_this_street_chips": int(raw["street_commit"][seat]),
                "folded": bool(raw["folded"][seat]),
                "participant": True,
                "all_in": bool(raw["all_in"][seat]),
                "is_actor": actor is not None and seat == actor,
                "is_hero": seat == h.hero_seat,
                "hole": hole,
            })

        if actor is not None and info is not None and not info.terminal:
            gm = info.gate_mask
            legal = {
                "fold": bool(gm[GATE_FOLD]),
                "check_call": bool(gm[GATE_CHECK_CALL]),
                "raise": bool(gm[GATE_RAISE]),
            }
            max_chips = int(info.max_raise_chips)
            min_chips = min(int(info.min_raise_chips), max_chips)
            # Short-shove: collapse to a single point so the UI renders an
            # All-in button (matches the study projection).
            if legal["raise"] and min_chips == 0 and max_chips > 0:
                min_chips = max_chips
            raise_bounds = {
                "min_chips": min_chips,
                "max_chips": max_chips,
                "min_bb": round(chips_to_bb(min_chips, bb), 4),
                "max_bb": round(chips_to_bb(max_chips, bb), 4),
            }
            to_call = _to_call_chips(raw, actor)
        else:
            legal = {k: False for k in ("fold", "check_call", "raise")}
            raise_bounds = {"min_chips": 0, "max_chips": 0,
                            "min_bb": 0.0, "max_bb": 0.0}
            to_call = 0

        is_nlh = _engine_variant(self.variant) == VARIANT_NLH
        if card_spec_override is not None:
            card_spec = {k: list(v) for k, v in card_spec_override.items()}
        else:
            ba = [int(c) for c in raw["board_a"]]
            bb_ = [int(c) for c in raw["board_b"]]
            if is_nlh:
                card_spec = {
                    "hero_hole": list(h.all_holes[h.hero_seat]),
                    "flop_a": (ba[:3] + [None] * 3)[:3],
                    "flop_b": [],
                    "turn": [ba[3] if len(ba) > 3 else None],
                    "river": [ba[4] if len(ba) > 4 else None],
                }
            else:
                card_spec = {
                    "hero_hole": list(h.all_holes[h.hero_seat]),
                    "flop_a": (ba[:3] + [None] * 3)[:3],
                    "flop_b": (bb_[:3] + [None] * 3)[:3],
                    "turn": [ba[3] if len(ba) > 3 else None,
                             bb_[3] if len(bb_) > 3 else None],
                    "river": [ba[4] if len(ba) > 4 else None,
                              bb_[4] if len(bb_) > 4 else None],
                }

        # Hero's best made hand per board, ClubGG-style ("#1 .. / #2 .."),
        # derived from the DISPLAYED card_spec (so a what-if swap relabels
        # too). Surfaces a stealth set/straight that's easy to fold by
        # reflex. describe_made_hand() is None with too few cards dealt.
        # NLH: single board, any-combo rule; the UI shows one unnumbered
        # label (board B slot is None).
        _hh = [c for c in card_spec["hero_hole"] if c is not None]
        _bd_a = [c for c in (list(card_spec["flop_a"])
                             + [card_spec["turn"][0], card_spec["river"][0]])
                 if c is not None]
        if is_nlh:
            hero_hand_desc = [describe_made_hand_nlh(_hh, _bd_a), None]
        else:
            _bd_b = [c for c in (list(card_spec["flop_b"])
                                 + [card_spec["turn"][1], card_spec["river"][1]])
                     if c is not None]
            hero_hand_desc = [describe_made_hand(_hh, _bd_a),
                              describe_made_hand(_hh, _bd_b)]

        # Live terminal state carries the review block (last decision) so
        # the frontend can open the review pane immediately.
        if review is None and live and h.terminal and h.decisions:
            # Open review on the last hero decision, but attach its
            # node_current so the dual-EV node view is populated from the
            # first frame (one extra replay+forward+critic at hand-end).
            last = h.decisions[-1]
            _e, _o, _i = self._replay_to_node(last.action_log_idx)
            review = self.review_block(
                len(h.decisions) - 1,
                node_idx=last.action_log_idx,
                node_current=self._node_view(last.action_log_idx, _o, _i),
            )

        state = {
            "num_seats": cfg.num_seats,
            "button_seat": h.button,
            "hero_seat": h.hero_seat,
            "actor": actor,
            "seats": seats,
            "card_spec": card_spec,
            "hero_hand_desc": hero_hand_desc,
            "hero_info_complete": True,
            "hero_blocking_reason": None,
            "modified_cards": [],
            "pot_chips": int(raw["pot"]),
            "pot_bb": round(chips_to_bb(int(raw["pot"]), bb), 4),
            # Settled pot ("Pot") = pot minus this street's live commits;
            # pot_chips is the grand "Total Pot".
            "settled_pot_chips": int(raw["pot"]) - sum(int(x) for x in raw["street_commit"]),
            "settled_pot_bb": round(
                chips_to_bb(
                    int(raw["pot"]) - sum(int(x) for x in raw["street_commit"]), bb
                ), 4
            ),
            "bet_to_call_chips": int(raw["bet_to_call"]),
            "bet_to_call_bb": round(chips_to_bb(int(raw["bet_to_call"]), bb), 4),
            "to_call_chips": int(to_call),
            "to_call_bb": round(chips_to_bb(int(to_call), bb), 4),
            "street": STREET_NAMES.get(int(raw["street"]), "flop"),
            "history": history_entries(raw, self._position_of, bb),
            "legal": legal,
            "raise_bounds": raise_bounds,
            "terminal": terminal,
            "terminal_message": terminal_message,
            "awaiting_next_street": None,
            "recommendation": None,
            "can_undo": False,
            "chip_scale": {
                "bb_chips": int(bb),
                "ante_chips": int(cfg.ante),
                "dollars_per_bb": float(self.settings.dollars_per_bb),
            },
            "starting_stacks_chips": [int(s) for s in cfg.resolved_stacks],
            "starting_stacks_bb": [
                round(chips_to_bb(int(s), bb), 4) for s in cfg.resolved_stacks
            ],
            **({} if _PUBLIC else {"simple_ocr_mode": False}),
            "format": self.variant,
            "trainer": {
                "settings": self.settings.model_dump(),
                # Server-side ceiling on settings.mc_rollouts for this build
                # (32 in the public build) so the client can bound its input.
                "mc_rollouts_max": mc_rollouts_cap(),
                "hand_no": h.hand_no,
                "hand_active": not h.terminal,
                "rewards_bb": h.rewards_bb,
                # Last hero decision's flash. Built per projection (not
                # stored) so the EV-loss number is withheld in every frame
                # while the hand is live and revealed once it is terminal
                # (review 2026-09-20 H2).
                "feedback": (
                    self._feedback_payload(h.decisions[-1], reveal_ev=h.terminal)
                    if h.decisions else None
                ),
                "backend": self.backend.coverage_badge(),
                # Host serving the PROJECTED decision node (public info only:
                # seat count + street): the active backend's name, or
                # "ppo_fallback" when the badge's host does not cover it.
                "node_backend": (
                    self._backend_for(info)[1]
                    if (actor is not None and info is not None
                        and not info.terminal)
                    else None
                ),
                "opp_actions": [
                    {
                        "seat": a["seat"],
                        "position": self._position_of(a["seat"]),
                        # `auto` = engine-forced check at a moot node; with
                        # `is_hero` it marks HERO's auto-checks, which ride in
                        # this list but are not opponent actions.
                        "is_hero": a["seat"] == h.hero_seat,
                        "auto": bool(a.get("auto", False)),
                        "gate": GATE_SLUGS[a["gate"]],
                        "chips": a["chips"],
                        "chips_bb": round(chips_to_bb(a["chips"], bb), 4),
                        "street": STREET_NAMES.get(a["street"], str(a["street"])),
                    }
                    for a in h.opp_actions_since_hero
                ],
                "stats": {
                    "session": self.session_stats.project(),
                    "lifetime": self.lifetime_stats.project(),
                },
                "review": review,
            },
        }
        return state


def _study_replay(
    env: BombPotEnv,
    button: int,
    hero_seat: int,
    spec: dict[str, list[int | None]],
    actions: list[dict[str, int]],
    variant: str = VARIANT_PLO5,
) -> tuple[np.ndarray, StepInfo]:
    """reset_study + replay `actions`, feeding (possibly modified)
    street cards whenever the study engine awaits one. Mirrors the study
    server's `_rebuild_env` advance pattern. NLH enters at the preflop
    (2-card hole, no flops at reset) and feeds flop/turn/river through
    the single-board setters."""
    is_nlh = variant == VARIANT_NLH

    def advance_streets() -> tuple[np.ndarray, StepInfo] | None:
        out = None
        while True:
            awaiting = env.awaiting_next_street()
            if is_nlh and awaiting == 1:
                f = spec["flop_a"]
                if any(c is None for c in f):
                    raise HTTPException(status_code=400,
                                        detail="flop cards required for replay")
                out = env.set_flop_nlh(int(f[0]), int(f[1]), int(f[2]))
            elif awaiting == 2:
                t = spec["turn"]
                if is_nlh:
                    if t[0] is None:
                        raise HTTPException(status_code=400,
                                            detail="turn card required for replay")
                    out = env.set_turn_nlh(int(t[0]))
                else:
                    if t[0] is None or t[1] is None:
                        raise HTTPException(status_code=400,
                                            detail="turn cards required for replay")
                    out = env.set_turn(int(t[0]), int(t[1]))
            elif awaiting == 3:
                r = spec["river"]
                if is_nlh:
                    if r[0] is None:
                        raise HTTPException(status_code=400,
                                            detail="river card required for replay")
                    out = env.set_river_nlh(int(r[0]))
                else:
                    if r[0] is None or r[1] is None:
                        raise HTTPException(status_code=400,
                                            detail="river cards required for replay")
                    out = env.set_river(int(r[0]), int(r[1]))
            else:
                return out

    hero_hole = [int(c) for c in spec["hero_hole"]]  # type: ignore[arg-type]
    if is_nlh:
        obs, info = env.reset_study_nlh(button, hero_seat, hero_hole)
    else:
        flop_a = [int(c) for c in spec["flop_a"]]  # type: ignore[arg-type]
        flop_b = [int(c) for c in spec["flop_b"]]  # type: ignore[arg-type]
        obs, info = env.reset_study(button, hero_seat, hero_hole, flop_a, flop_b)
    for a in actions:
        adv = advance_streets()
        if adv is not None:
            obs, info = adv
        obs, _, _, info = env.step_hybrid(a["gate"], a["chips"])
    adv = advance_streets()
    if adv is not None:
        obs, info = adv
    return obs, info


def _to_call_chips(obs: dict[str, Any], actor: int) -> int:
    current_commit = int(obs["street_commit"][actor])
    stack = int(obs["stacks"][actor])
    bet_to_call = int(obs["bet_to_call"])
    return min(max(0, bet_to_call - current_commit), stack)


# A stack at or below this is "dust": chips that can neither make nor
# meaningfully call a bet (0.02bb — a few cents at any real chip scale,
# 1/50th of the 1bb minimum bet). A seat that called off all but a sliver
# is treated like an all-in seat. This is load-bearing because the engine
# does NOT set all_in on a *call* that merely empties a stack to a tiny
# residual: it keeps offering betting rounds whose only "bet" is that
# sub-cent remainder, stranding an all-in hand on a meaningless hero check
# instead of running it out to showdown. (Observed: a ~15-chip / 0.0015bb
# residual showing "$0.00", which the old 0.001bb floor missed.) Pinned by
# test_trainer_runout.py.
_DUST_CHIPS = max(1, BB_CHIPS // 50)


def _betting_moot(obs: dict[str, Any], actor: int) -> bool:
    """True when the actor has no real betting decision: at most dust
    to call, and every other live seat is all-in or down to dust, so
    no meaningful bet could ever be made or called. The engine still
    asks for an action on each remaining street; `_advance` auto
    check/calls through these nodes so an all-in hand runs out to
    showdown instead of stopping on hero."""
    if _to_call_chips(obs, actor) > _DUST_CHIPS:
        return False
    folded = obs["folded"]
    all_in = obs["all_in"]
    stacks = obs["stacks"]
    return all(
        bool(folded[s]) or bool(all_in[s]) or int(stacks[s]) <= _DUST_CHIPS
        for s in range(len(folded))
        if s != actor
    )


def _action_label(gate: int, chips: int, to_call: int) -> str:
    if gate == GATE_FOLD:
        return "Fold"
    if gate == GATE_CHECK_CALL:
        return "Call" if to_call > 0 else "Check"
    verb = "Raise" if to_call > 0 else "Bet"
    return f"{verb} {chips_to_bb(chips, BB_CHIPS):.2f}bb"


# --- Router ---------------------------------------------------------------------


# Public-build hook: when installed (plo5bp.ui.public), returns the signed-in
# user's own TrainerSession. In the LOCAL build None (or no hook) falls back
# to the router's single default session, keeping it byte-identical; in the
# PUBLIC build that fallback is refused (see `_ts`).
_SESSION_RESOLVER: Callable[[], "TrainerSession | None"] | None = None


def set_session_resolver(fn: Callable[[], "TrainerSession | None"] | None) -> None:
    global _SESSION_RESOLVER
    _SESSION_RESOLVER = fn


def create_trainer_router(
    model: ActorCritic,
    device: torch.device,
    critic: CentralCritic | None = None,
    formats: dict[str, dict[str, Any]] | None = None,
    gto_checkpoint: str | Path | None = None,
) -> APIRouter:
    # Optional T1 host: env PLO5BP_GTO_CHECKPOINT or explicit path.
    gto_path = gto_checkpoint or os.environ.get("PLO5BP_GTO_CHECKPOINT", "").strip() or None
    gto_host = try_load_gto_host(gto_path, device=device) if gto_path else None
    default_ts = TrainerSession(model, device, critic=critic)
    if gto_host is not None and default_ts.variant == VARIANT_NLH:
        default_ts.set_backend(gto_host)
    router = APIRouter(prefix="/trainer")
    router.trainer_session = default_ts  # type: ignore[attr-defined]  # test hook
    # Per-format (model, critic) registry mirroring server.FORMATS; used
    # by set_format to swap what the CURRENT trainer session serves.
    router_formats = formats or {}

    def _ts() -> TrainerSession:
        if _SESSION_RESOLVER is not None:
            resolved = _SESSION_RESOLVER()
            if resolved is not None:
                return resolved
        if _public_mode():
            # FAIL CLOSED (review 2026-09-20, latent): the default session is
            # ONE object shared by every caller. In the public build a request
            # that reaches here without a per-user session (auth middleware
            # bypassed / resolver not installed) must never read or drive it.
            if _SESSION_RESOLVER is None:
                raise HTTPException(
                    status_code=500,
                    detail="trainer session resolver not installed",
                )
            raise HTTPException(status_code=401, detail="sign in required")
        return default_ts

    def set_format(variant: str) -> None:
        """Switch the resolved trainer session's game format (called by
        the study server's POST /format so both tabs track one game)."""
        entry = router_formats.get(variant)
        if entry is None:
            raise ValueError(f"no model registered for format {variant!r}")
        ts = _ts()
        with ts.lock:
            # T1: NLH + GTO checkpoint → PolicyNetHost; else PPO host for format.
            be = None
            if variant == VARIANT_NLH and gto_host is not None:
                be = gto_host
            ts.set_format(variant, entry["model"], entry["critic"], backend=be)

    router.set_format = set_format  # type: ignore[attr-defined]

    def _ensure_hand(ts: TrainerSession) -> None:
        if ts.hand is None:
            ts.new_hand()

    @router.get("/state")
    def trainer_state() -> dict[str, Any]:
        ts = _ts()
        with ts.lock:
            _ensure_hand(ts)
            return {"state": ts.project_state()}

    @router.post("/settings")
    def trainer_settings(req: TrainerSettings) -> dict[str, Any]:
        ts = _ts()
        with ts.lock:
            # Applies to the ACTIVE format only (each format owns its
            # settings; set_settings keeps the per-format registry in sync).
            ts.set_settings(req)
            ts._persist()
            _ensure_hand(ts)
            return {"state": ts.project_state()}

    @router.post("/new_hand")
    def trainer_new_hand() -> dict[str, Any]:
        ts = _ts()
        with ts.lock:
            frames = ts.new_hand()
            return {"state": ts.project_state(), "frames": frames}

    @router.post("/repeat")
    def trainer_repeat() -> dict[str, Any]:
        ts = _ts()
        with ts.lock:
            if ts.hand is None:
                raise HTTPException(status_code=400, detail="no hand to repeat")
            frames = ts.new_hand(repeat=True)
            return {"state": ts.project_state(), "frames": frames}

    @router.post("/act")
    def trainer_act(req: TrainerActRequest) -> dict[str, Any]:
        ts = _ts()
        with ts.lock:
            # No `_ensure_hand` here (review 2026-09-20 F9 / F6): with no live
            # hand (restart, LRU eviction, format switch) the old path dealt a
            # fresh — unmetered — hand and applied the client's stale action to
            # it blindly. `ts.act` answers 409; the client re-syncs via
            # GET /trainer/state or deals through POST /trainer/new_hand.
            frames = ts.act(req.gate, req.chips)
            return {"state": ts.project_state(), "frames": frames}

    @router.get("/review")
    def trainer_review(
        decision: int = 0, node: int | None = None
    ) -> dict[str, Any]:
        ts = _ts()
        with ts.lock:
            if node is not None:
                return {"state": ts.review_at_node(node)}
            d = ts._decision_for(decision)
            env, _obs, info = ts._replay_to_decision(d)
            review = ts.review_block(d.decision_idx)
            return {
                "state": ts.project_state(
                    env=env, info=info, reveal=True, review=review
                )
            }

    @router.post("/whatif")
    def trainer_whatif(req: WhatifRequest) -> dict[str, Any]:
        ts = _ts()
        with ts.lock:
            return {"state": ts.whatif(req)}

    @router.post("/stats/reset")
    def trainer_stats_reset(req: StatsResetRequest) -> dict[str, Any]:
        ts = _ts()
        with ts.lock:
            if req.scope == "session":
                ts.session_stats = StatsBlock()
            else:
                ts.lifetime_stats = StatsBlock()
                ts._persist()
            _ensure_hand(ts)
            return {"state": ts.project_state()}

    return router
