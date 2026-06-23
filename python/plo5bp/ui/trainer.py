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
  `action_log` like any action, so replay determinism is unchanged.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv, StepInfo
from plo5bp.eval import model_policy
from plo5bp.network import ActorCritic, CentralCritic, obs_adapter
from plo5bp.rollout import _critic_values, _rotate_opp_holes
from plo5bp.sizing import (
    ANCHOR_COUNT,
    anchor_grid_np,
    anchor_grid_torch,
    sizing_from_info,
)
from plo5bp.ui.common import (
    STREET_NAMES,
    anchor_label,
    chips_to_bb,
    history_entries,
    position_name,
    validate_card_list,
)

logger = logging.getLogger("plo5bp.ui.trainer")

BB_CHIPS = 10000

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


def score_move_v2(
    dist: dict[str, Any],
    user_gate: int,
    user_chips: int,
) -> dict[str, Any]:
    """Score one decision against a v2 (anchor head) node distribution.

    `gate_ratio` is unchanged from v1. For raises, the user's chips snap
    to `user_anchor` — the nearest LEGAL anchor by chip distance (tie →
    lower anchor) — and
    `size_q = P(user_anchor)/P(best_anchor) × refinement-pdf-ratio`,
    where the pdf ratio compares the user anchor's Beta density at the
    user's in-bracket position vs at its mode. Atoms, short-shove and
    collapsed brackets have no within-anchor size choice → pdf ratio 1.
    Categories, the size floor, and the blunder override match v1.
    """
    gate_probs = dist["gate_probs"]
    g_star = max(range(len(gate_probs)), key=lambda i: gate_probs[i])
    p_user = float(gate_probs[user_gate])
    p_best = float(gate_probs[g_star])
    gate_ratio = (p_user / p_best) if p_best > 0 else 0.0

    size_q = 1.0
    user_anchor: int | None = None
    if user_gate == GATE_RAISE:
        legal_ks = [
            k for k in range(ANCHOR_COUNT) if dist["anchor_legal"][k]
        ]
        user_anchor = min(
            legal_ks,
            key=lambda k: (abs(user_chips - dist["anchor_chips"][k]), k),
        )
        a_probs = dist["anchor_probs"]
        k_best = max(legal_ks, key=lambda k: a_probs[k])
        p_ku = float(a_probs[user_anchor])
        p_kb = float(a_probs[k_best])
        anchor_ratio = (p_ku / p_kb) if p_kb > 0 else 0.0

        pdf_ratio = 1.0
        if dist["refine_ok"][user_anchor]:
            lo = int(dist["anchor_lo"][user_anchor])
            hi = int(dist["anchor_hi"][user_anchor])
            if hi > lo:
                eps = SCORING["u_eps"]
                u = (user_chips - lo) / (hi - lo)
                u = min(1.0 - eps, max(eps, u))
                alpha, beta = dist["refine_params"][user_anchor - 1]
                # Reference the Beta MEAN (= the size the deterministic policy
                # bets and the UI shows as the recommendation), clamped to <=1,
                # so betting the recommended size earns full size credit. The
                # earlier mode reference penalized the recommended mean on any
                # skewed Beta (a matched bet could score well under 100%).
                ref = alpha / (alpha + beta)
                ref = min(1.0 - eps, max(eps, ref))

                def logpdf(x: float) -> float:
                    return (alpha - 1.0) * math.log(x) \
                        + (beta - 1.0) * math.log(1.0 - x)

                pdf_ratio = min(1.0, math.exp(logpdf(u) - logpdf(ref)))
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
        sizing = sizing_from_info(info)
        sizing_t = torch.from_numpy(sizing[None, :]).to(device)
        with torch.no_grad():
            gate_logits, anchor_head_out, refine, value = model(obs_t, gm_t)
            gate_probs = F.softmax(gate_logits, dim=-1).squeeze(0).tolist()
            _act_out = model.act(obs_t, gm_t, sizing_t, deterministic=True)
            # Anchor histogram via the model's own (head-agnostic) anchor
            # distribution: flat softmax for v2, discretized-logistic for v4.
            anchor_probs = (
                model._anchor_dist(anchor_head_out, anchor_grid_torch(sizing_t))
                .probs.squeeze(0).float().cpu().numpy()
            )
            refine_np = refine.squeeze(0).float().cpu().numpy()  # (9, 2)
        grid = anchor_grid_np(sizing[0], sizing[1], sizing[2], sizing[3])
        return {
            "head_version": model.head_version,
            "gate_probs": [float(p) for p in gate_probs],
            "anchor_probs": [float(p) for p in anchor_probs],
            "anchor_chips": [int(c) for c in grid.chips],
            "anchor_legal": [bool(b) for b in grid.legal],
            "anchor_lo": [int(c) for c in grid.lo],
            "anchor_hi": [int(c) for c in grid.hi],
            "refine_ok": [bool(b) for b in grid.refine_ok],
            "refine_params": [[float(a), float(b)] for a, b in refine_np],
            "rec_anchor": int(_act_out.anchor.item()),
            "min_chips": int(info.min_raise_chips),
            "max_chips": raise_max,
            "rec_gate": int(_act_out.gate.item()),
            "rec_chips": int(_act_out.chips.item()),
            "value_bb": float(value.squeeze(0).item()),
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
) -> list[dict[str, Any]]:
    """Legal-only anchor histogram rows for client rendering."""
    return [
        {
            "k": int(k),
            "label": anchor_label(int(k)),
            "prob": round(float(anchor_probs[k]), 4),
            "chips": int(anchor_chips[k]),
            "chips_bb": round(chips_to_bb(int(anchor_chips[k]), bb), 4),
        }
        for k in range(ANCHOR_COUNT)
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
    ev_loss_bb: float | None = None
    # Hero's total committed chips AT the decision node (= chips forfeited
    # on a fold). Rebases the EV components to forward-facing (fold = 0).
    hero_committed_chips: int = 0
    # v2 (anchor head) extras; None/1 on v1 records. For v2, (alpha,
    # beta) above hold the REC anchor's refinement params (1.0/1.0 when
    # the rec anchor is an atom).
    head_version: int = 1
    anchor_probs: list[float] | None = None
    anchor_chips: list[int] | None = None
    anchor_legal: list[bool] | None = None
    rec_anchor: int | None = None
    user_anchor: int | None = None


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
    feedback: dict[str, Any] | None = None


@dataclass
class StatsBlock:
    hands: int = 0
    moves: int = 0
    score_sum: float = 0.0
    cat_counts: dict[str, int] = field(
        default_factory=lambda: {c: 0 for c in CATEGORIES}
    )
    ev_loss_sum_bb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hands": self.hands,
            "moves": self.moves,
            "score_sum": self.score_sum,
            "cat_counts": dict(self.cat_counts),
            "ev_loss_sum_bb": self.ev_loss_sum_bb,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StatsBlock":
        counts = {c: 0 for c in CATEGORIES}
        counts.update({k: int(v) for k, v in dict(d.get("cat_counts", {})).items()
                       if k in counts})
        return cls(
            hands=int(d.get("hands", 0)),
            moves=int(d.get("moves", 0)),
            score_sum=float(d.get("score_sum", 0.0)),
            cat_counts=counts,
            ev_loss_sum_bb=float(d.get("ev_loss_sum_bb", 0.0)),
        )

    def project(self) -> dict[str, Any]:
        gto = (self.score_sum / self.moves) if self.moves > 0 else None
        per_hand = (self.ev_loss_sum_bb / self.hands) if self.hands > 0 else None
        return {
            "hands": self.hands,
            "moves": self.moves,
            "gto_score": round(gto, 1) if gto is not None else None,
            "cat_counts": dict(self.cat_counts),
            "ev_loss_total_bb": round(self.ev_loss_sum_bb, 2),
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
    ):
        self.model = model
        self.device = device
        # Centralized critic (sees all hole cards) for the review's
        # "true EV" readout. None on v1 / when the checkpoint lacks one
        # → review shows only the actor's own (blind) value estimate.
        self.critic = critic
        self.lock = threading.Lock()
        self.settings = TrainerSettings()
        self.hand: HandRecord | None = None
        self.hand_no = 0
        self.session_stats = StatsBlock()
        self.lifetime_stats = StatsBlock()
        self.rng = np.random.default_rng()
        self._policy = model_policy(model, deterministic=False)
        self._obs_adapt = obs_adapter(model)
        env_path = os.environ.get("PLO5BP_TRAINER_STATS")
        self.stats_path = (
            stats_path
            if stats_path is not None
            else Path(env_path) if env_path
            else Path("checkpoints/trainer_stats.json")
        )
        self._load_persisted()

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
            if isinstance(data.get("settings"), dict):
                self.settings = TrainerSettings.model_validate(data["settings"])
        except Exception as e:
            logger.warning("trainer stats file %s malformed (%s) — starting fresh",
                           self.stats_path, e)

    def _persist(self) -> None:
        payload = {
            "version": 1,
            "lifetime": self.lifetime_stats.to_dict(),
            "settings": self.settings.model_dump(),
        }
        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.stats_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self.stats_path)
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
            config = GameConfig(
                num_seats=n,
                starting_stack=int(round(s.stack_bb * BB_CHIPS)),
                ante=int(round(s.ante_bb * BB_CHIPS)),
                bb=BB_CHIPS,
                starting_stacks=self._draw_stacks(n),
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
                # of the action prefix (see module docstring).
                torch.manual_seed(self._opp_seed(h))
                gate, chips = self._policy(h.last_obs, info.actor, info)
            obs, rewards, done, info2 = h.env.step_hybrid(gate, chips)
            entry = {"seat": actor, "gate": int(gate), "chips": int(chips),
                     "street": street}
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
            self.session_stats.hands += 1
            self.lifetime_stats.hands += 1
            self._persist()

    # -- acting / scoring --------------------------------------------------------

    def act(self, gate_slug: str, chips_req: int | None) -> list[dict[str, Any]]:
        h = self.hand
        if h is None:
            raise HTTPException(status_code=400, detail="no active hand")
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

        dist = compute_node_distribution(self.model, self.device, h.last_obs, info)
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
        )

        street = int(raw["street"])
        obs, rewards, done, info2 = h.env.step_hybrid(gate_idx, chips)
        h.action_log.append({"seat": int(h.hero_seat), "gate": int(gate_idx),
                             "chips": int(chips), "street": street})
        h.opp_actions_since_hero = []
        h.last_obs, h.last_info = obs, info2
        frames: list[dict[str, Any]] = []
        if done:
            self._finalize(rewards)
            frames.append(self.project_state())
        else:
            frames.append(self.project_state())  # hero's action landed
            self._advance(frames)

        self._estimate_ev_loss(decision)
        h.decisions.append(decision)
        h.feedback = self._feedback_payload(decision)

        if not h.is_repeat:
            for block in (self.session_stats, self.lifetime_stats):
                block.moves += 1
                block.score_sum += decision.score
                block.cat_counts[decision.category] += 1
                if decision.ev_loss_bb is not None:
                    block.ev_loss_sum_bb += decision.ev_loss_bb
            self._persist()
        return frames

    def _feedback_payload(self, d: DecisionRecord) -> dict[str, Any]:
        bb = BB_CHIPS
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
            "ev_loss_bb": round(d.ev_loss_bb, 3) if d.ev_loss_bb is not None else None,
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
        n = int(self.settings.mc_rollouts)
        if n <= 0:
            return
        if self._candidates_equal(d):
            d.ev_loss_bb = 0.0
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
        `node_seed` for both candidates = common random numbers."""
        torch.manual_seed(node_seed)
        total = 0.0
        live: list[list[Any]] = []  # [env, obs, info]
        for _ in range(n):
            # EV runouts: grade all-in continuations by expected value over
            # board runouts instead of one sampled runout — same rollout
            # count, much less estimator noise. (The live hand's displayed
            # result stays realized; only this estimator uses EV.)
            env = BombPotEnv(h.config, ev_runout_samples=32)
            obs, info = env.reset(h.seed, h.button)
            for a in prefix:
                obs, _, _, info = env.step_hybrid(a["gate"], a["chips"])
            obs, rewards, done, info = env.step_hybrid(gate, chips)
            if done:
                total += float(rewards[h.hero_seat])
            else:
                live.append([env, obs, info])
        # Lockstep: one batched forward per depth across all live rollouts.
        while live:
            obs_b = torch.from_numpy(
                self._obs_adapt(np.stack([x[1] for x in live]))
            ).to(self.device)
            gm_b = torch.from_numpy(
                np.stack([x[2].gate_mask for x in live])
            ).to(self.device)
            # (B, 4) sizing context — v1 models slice [..., :2], v2 needs
            # all four columns for the anchor grid.
            sizing_b = torch.from_numpy(
                np.stack([sizing_from_info(x[2]) for x in live])
            ).to(self.device)
            with torch.no_grad():
                _mc_out = self.model.act(
                    obs_b, gm_b, sizing_b, deterministic=False
                )
            nxt: list[list[Any]] = []
            for i, x in enumerate(live):
                obs2, rewards, done, info2 = x[0].step_hybrid(
                    int(_mc_out.gate[i].item()), int(_mc_out.chips[i].item())
                )
                if done:
                    total += float(rewards[h.hero_seat])
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
        dist = compute_node_distribution(self.model, self.device, obs_np, info)

        # own EV = the actor's observation-only value head (blind to
        # opponents' cards); true EV = the centralized critic (sees all
        # hole cards), built with the EXACT training convention via the
        # canonical rollout helpers so the number is meaningful.
        value_true_bb: float | None = None
        if self.critic is not None and dist["head_version"] >= 2:
            holes = np.asarray(h.all_holes_dealt, dtype=np.uint8)  # (S, 5)
            opp = _rotate_opp_holes(holes, actor)[None]            # (1, 5, 5)
            value_true_bb = round(float(_critic_values(
                self.critic, self.device,
                obs_np[None].astype(np.float32), opp,
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
            "pot_chips": int(raw["pot"]),
            "actual_gate": GATE_SLUGS[actual_gate],
            "actual_chips": actual_chips if actual_gate == GATE_RAISE else None,
            "actual_label": _action_label(actual_gate, actual_chips, to_call),
        }
        if dist["head_version"] >= 2:
            nc["head_version"] = 2
            nc["anchors"] = _anchors_payload(
                dist["anchor_probs"], dist["anchor_chips"],
                dist["anchor_legal"], bb,
            )
            nc["rec_anchor"] = dist["rec_anchor"]
            # Mark where the actor's ACTUAL raise landed (the ● on the EQ
            # bars): nearest legal anchor by chip distance, tie -> lower.
            if actual_gate == GATE_RAISE:
                legal_ks = [
                    k for k in range(ANCHOR_COUNT) if dist["anchor_legal"][k]
                ]
                nc["user_anchor"] = min(
                    legal_ks,
                    key=lambda k: (abs(actual_chips - dist["anchor_chips"][k]), k),
                ) if legal_ks else None
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
            })
            if d.head_version >= 2:
                nc["user_anchor"] = d.user_anchor
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
            "pot_chips": d.pot_chips,
            "ev_user_bb": d.ev_user_bb,
            "ev_best_bb": d.ev_best_bb,
            "ev_loss_bb": d.ev_loss_bb,
        }
        if d.head_version >= 2:
            current["head_version"] = 2
            current["anchors"] = _anchors_payload(
                d.anchor_probs, d.anchor_chips, d.anchor_legal, bb
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

        overrides = {
            "hero_hole": (req.hero_hole, 5),
            "flop_a": (req.flop_a, 3),
            "flop_b": (req.flop_b, 3),
            "turn": (req.turn, 2),
            "river": (req.river, 2),
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
            env, h.button, h.hero_seat, spec, h.action_log[: d.action_log_idx]
        )
        if info.actor != h.hero_seat:
            raise HTTPException(
                status_code=500,
                detail="what-if replay did not reach hero's decision node",
            )
        dist = compute_node_distribution(self.model, self.device, obs, info)
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
        }
        if dist["head_version"] >= 2:
            rec_alpha, rec_beta = _rec_refine_params(dist)
            recommendation["head_version"] = 2
            recommendation["anchors"] = _anchors_payload(
                dist["anchor_probs"], dist["anchor_chips"],
                dist["anchor_legal"], bb,
            )
            recommendation["rec_anchor"] = dist["rec_anchor"]
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

        if card_spec_override is not None:
            card_spec = {k: list(v) for k, v in card_spec_override.items()}
        else:
            ba = [int(c) for c in raw["board_a"]]
            bb_ = [int(c) for c in raw["board_b"]]
            card_spec = {
                "hero_hole": list(h.all_holes[h.hero_seat]),
                "flop_a": (ba[:3] + [None] * 3)[:3],
                "flop_b": (bb_[:3] + [None] * 3)[:3],
                "turn": [ba[3] if len(ba) > 3 else None,
                         bb_[3] if len(bb_) > 3 else None],
                "river": [ba[4] if len(ba) > 4 else None,
                          bb_[4] if len(bb_) > 4 else None],
            }

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
            "simple_ocr_mode": False,
            "trainer": {
                "settings": self.settings.model_dump(),
                "hand_no": h.hand_no,
                "hand_active": not h.terminal,
                "rewards_bb": h.rewards_bb,
                "feedback": h.feedback,
                "opp_actions": [
                    {
                        "seat": a["seat"],
                        "position": self._position_of(a["seat"]),
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
) -> tuple[np.ndarray, StepInfo]:
    """reset_study + replay `actions`, feeding (possibly modified)
    turn/river cards whenever the study engine awaits a street. Mirrors
    the study server's `_rebuild_env` advance pattern."""

    def advance_streets() -> tuple[np.ndarray, StepInfo] | None:
        out = None
        while True:
            awaiting = env.awaiting_next_street()
            if awaiting == 2:
                t = spec["turn"]
                if t[0] is None or t[1] is None:
                    raise HTTPException(status_code=400,
                                        detail="turn cards required for replay")
                out = env.set_turn(int(t[0]), int(t[1]))
            elif awaiting == 3:
                r = spec["river"]
                if r[0] is None or r[1] is None:
                    raise HTTPException(status_code=400,
                                        detail="river cards required for replay")
                out = env.set_river(int(r[0]), int(r[1]))
            else:
                return out

    hero_hole = [int(c) for c in spec["hero_hole"]]  # type: ignore[arg-type]
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
# meaningfully call a bet (0.001bb — fractions of a cent at any real
# chip scale). A seat that called off all but a few chips is treated
# like an all-in seat; without this, an exact-ish all-in call leaves
# the engine walking betting rounds where the only available bet is
# the opponent's sub-cent remainder ("Bet $0.00" in the UI).
_DUST_CHIPS = max(1, BB_CHIPS // 1000)


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


def create_trainer_router(
    model: ActorCritic,
    device: torch.device,
    critic: CentralCritic | None = None,
) -> APIRouter:
    ts = TrainerSession(model, device, critic=critic)
    router = APIRouter(prefix="/trainer")
    router.trainer_session = ts  # type: ignore[attr-defined]  # test hook

    def _ensure_hand() -> None:
        if ts.hand is None:
            ts.new_hand()

    @router.get("/state")
    def trainer_state() -> dict[str, Any]:
        with ts.lock:
            _ensure_hand()
            return {"state": ts.project_state()}

    @router.post("/settings")
    def trainer_settings(req: TrainerSettings) -> dict[str, Any]:
        with ts.lock:
            ts.settings = req
            ts._persist()
            _ensure_hand()
            return {"state": ts.project_state()}

    @router.post("/new_hand")
    def trainer_new_hand() -> dict[str, Any]:
        with ts.lock:
            frames = ts.new_hand()
            return {"state": ts.project_state(), "frames": frames}

    @router.post("/repeat")
    def trainer_repeat() -> dict[str, Any]:
        with ts.lock:
            if ts.hand is None:
                raise HTTPException(status_code=400, detail="no hand to repeat")
            frames = ts.new_hand(repeat=True)
            return {"state": ts.project_state(), "frames": frames}

    @router.post("/act")
    def trainer_act(req: TrainerActRequest) -> dict[str, Any]:
        with ts.lock:
            _ensure_hand()
            frames = ts.act(req.gate, req.chips)
            return {"state": ts.project_state(), "frames": frames}

    @router.get("/review")
    def trainer_review(
        decision: int = 0, node: int | None = None
    ) -> dict[str, Any]:
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
        with ts.lock:
            return {"state": ts.whatif(req)}

    @router.post("/stats/reset")
    def trainer_stats_reset(req: StatsResetRequest) -> dict[str, Any]:
        with ts.lock:
            if req.scope == "session":
                ts.session_stats = StatsBlock()
            else:
                ts.lifetime_stats = StatsBlock()
                ts._persist()
            _ensure_hand()
            return {"state": ts.project_state()}

    return router
