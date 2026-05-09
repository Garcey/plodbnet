"""Evaluation: match a model against simple baselines over many hands.

Policies return a `(gate, chips)` tuple in the hybrid action space
(Fold/CheckCall/Raise, with chips in [min_raise, max_raise] when
gate == Raise, else 0). Stack-bound short shoves are encoded as Raise
at u=1 — the engine's `max_raise_chips` already clamps to stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from plo5bp.actions import (
    GATE_CHECK_CALL,
    GATE_FOLD,
    GATE_RAISE,
)
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv, StepInfo
from plo5bp.network import ActorCritic


Policy = Callable[[np.ndarray, int, StepInfo], tuple[int, int]]


def _first_legal_gate(gate_mask: np.ndarray) -> int:
    return int(np.flatnonzero(gate_mask)[0])


def random_legal_policy(rng: np.random.Generator) -> Policy:
    def act(obs: np.ndarray, actor: int, info: StepInfo) -> tuple[int, int]:
        legal = np.flatnonzero(info.gate_mask)
        gate = int(rng.choice(legal))
        chips = 0
        if gate == GATE_RAISE:
            lo = int(info.min_raise_chips)
            hi = int(info.max_raise_chips)
            chips = int(rng.integers(lo, hi + 1))
        return gate, chips

    return act


def always_call_policy() -> Policy:
    def act(obs: np.ndarray, actor: int, info: StepInfo) -> tuple[int, int]:
        gm = info.gate_mask
        if gm[GATE_CHECK_CALL]:
            return GATE_CHECK_CALL, 0
        return _first_legal_gate(gm), 0

    return act


def always_pot_bet_policy() -> Policy:
    """Raise to max chips when legal (engine clamps to stack for short shoves),
    else call, else fold."""

    def act(obs: np.ndarray, actor: int, info: StepInfo) -> tuple[int, int]:
        gm = info.gate_mask
        if gm[GATE_RAISE]:
            return GATE_RAISE, int(info.max_raise_chips)
        if gm[GATE_CHECK_CALL]:
            return GATE_CHECK_CALL, 0
        if gm[GATE_FOLD]:
            return GATE_FOLD, 0
        return _first_legal_gate(gm), 0

    return act


def model_policy(model: ActorCritic, deterministic: bool = False) -> Policy:
    device = next(model.parameters()).device

    def act(obs: np.ndarray, actor: int, info: StepInfo) -> tuple[int, int]:
        with torch.no_grad():
            o = torch.from_numpy(obs).unsqueeze(0).to(device)
            m = torch.from_numpy(info.gate_mask).unsqueeze(0).to(device)
            b = torch.tensor(
                [[info.min_raise_chips, info.max_raise_chips]],
                dtype=torch.long,
                device=device,
            )
            gate, chips, _, _ = model.act(o, m, b, deterministic=deterministic)
        return int(gate.item()), int(chips.item())

    return act


@dataclass
class MatchStats:
    hero_reward_mean: float
    hero_reward_std: float
    num_hands: int


def run_match(
    hero_policy: Policy,
    opponent_policy: Policy,
    game_config: GameConfig,
    num_hands: int,
    seed: int = 0,
) -> MatchStats:
    """Play `num_hands` hands. Hero seat rotates; return hero's mean chip delta
    in bb."""
    env = BombPotEnv(game_config)
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    reward_norm = 1.0 / float(game_config.bb)
    for h in range(num_hands):
        hero_seat = h % game_config.num_seats
        button = int(rng.integers(0, game_config.num_seats))
        hand_seed = int(rng.integers(0, 2**63 - 1))
        obs, info = env.reset(hand_seed, button)
        while not info.terminal:
            actor = info.actor
            pol = hero_policy if actor == hero_seat else opponent_policy
            gate, chips = pol(obs, actor, info)
            obs, rs, done, info = env.step_hybrid(gate, chips)
            if done:
                rewards.append(float(rs[hero_seat]) * reward_norm)
                break
    arr = np.asarray(rewards, dtype=np.float64)
    return MatchStats(
        hero_reward_mean=float(arr.mean()),
        hero_reward_std=float(arr.std()),
        num_hands=len(arr),
    )
