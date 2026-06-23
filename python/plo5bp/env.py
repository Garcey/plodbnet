"""Multi-agent bomb-pot env.

Each `step` applies one action for `current_actor` and returns the next
observation (for the new actor, or a zero vector when terminal), a
per-seat reward vector (zero mid-hand, chip delta at terminal), a done
flag, and an info dict.

The hybrid continuous-sizing policy calls `step_hybrid(gate, chips)`
instead of `step(action_idx)`: gate 2 (Raise) routes to the engine's
`apply_raise_chips(chips)` path; gates 0/1 map to the discrete
Fold/CheckCall actions. Stack-bound short shoves are encoded as Raise
to the clamped max — the engine's `max_raise_chips` already clamps to
stack and `apply_raise_chips` routes short shoves correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plo5bp._engine import GameState as _RustGameState  # type: ignore[attr-defined]
from plo5bp.actions import (
    ALL_IN,
    CHECK_CALL,
    FOLD,
    GATE_ACTIONS,
    GATE_CHECK_CALL,
    GATE_FOLD,
    GATE_RAISE,
    NUM_ACTIONS,
    gate_mask_from_bounds,
)
from plo5bp.config import GameConfig
from plo5bp.encoding import OBS_DIM, encode_observation


@dataclass
class StepInfo:
    legal_mask: np.ndarray              # (8,) discrete action mask (UI/legacy)
    gate_mask: np.ndarray               # (3,) hybrid-gate legality
    min_raise_chips: int                # 0 when Raise gate is illegal
    max_raise_chips: int                # 0 when Raise gate is illegal
    actor: int | None
    terminal: bool
    raw_obs: dict[str, Any] = field(default_factory=dict)
    # Per-seat chips committed AT THIS STEP (post_total_commit −
    # pre_total_commit). Always ≥ 0; non-zero only for the seat that
    # just acted (the actor). Powers the forward-EV reward signal:
    # rollout uses −commit_delta as the per-step training reward for
    # the actor, so the value head learns chip change from the decision
    # point forward (sunk costs excluded).
    commit_delta: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    # Per-seat cumulative chips committed THIS HAND, post-step. At
    # terminal, rollout adds this back to `payouts` to recover the
    # gross-win component (`won = payouts + total_commit`) which is
    # then credited as the trajectory's terminal reward.
    total_commit: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )


class BombPotEnv:
    """Multi-seat env. Actions are applied for the current actor; rewards
    are per-seat, delivered at terminal as the chip delta vs starting stack.
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        ev_runout_samples: int = 0,
    ):
        self.config = config or GameConfig()
        stacks = np.asarray(self.config.resolved_stacks, dtype=np.uint64)
        self._rs = _RustGameState(
            num_seats=self.config.num_seats,
            starting_stack=0,
            ante=self.config.ante,
            bb=self.config.bb,
            starting_stacks=stacks,
        )
        self._last_obs_vec = np.zeros(OBS_DIM, dtype=np.float32)
        self._last_mask = np.zeros(NUM_ACTIONS, dtype=bool)
        self._ev_runout_samples = int(ev_runout_samples)
        self._reset_seed: int = 0

    def reset(
        self,
        seed: int,
        button: int,
        in_hand_mask: list[bool] | None = None,
    ) -> tuple[np.ndarray, StepInfo]:
        if in_hand_mask is None:
            self._rs.reset(seed, button)
        else:
            self._rs.reset(seed, button, list(in_hand_mask))
        self._reset_seed = int(seed)
        return self._pack_obs()

    def reset_study(
        self,
        button: int,
        hero_seat: int,
        hero_hole: list[int],
        flop_a: list[int],
        flop_b: list[int],
        in_hand_mask: list[bool] | None = None,
    ) -> tuple[np.ndarray, StepInfo]:
        if in_hand_mask is None:
            self._rs.reset_study(
                button, hero_seat, list(hero_hole), list(flop_a), list(flop_b)
            )
        else:
            self._rs.reset_study(
                button,
                hero_seat,
                list(hero_hole),
                list(flop_a),
                list(flop_b),
                list(in_hand_mask),
            )
        return self._pack_obs()

    def set_turn(self, card_a: int, card_b: int) -> tuple[np.ndarray, StepInfo]:
        self._rs.set_turn(int(card_a), int(card_b))
        return self._pack_obs()

    def set_river(self, card_a: int, card_b: int) -> tuple[np.ndarray, StepInfo]:
        self._rs.set_river(int(card_a), int(card_b))
        return self._pack_obs()

    def awaiting_next_street(self) -> int | None:
        return self._rs.awaiting_next_street()

    def study_terminal(self) -> int | None:
        return self._rs.study_terminal()

    def _read_total_commit(self) -> np.ndarray:
        return np.asarray(
            self._rs.observation_dict()["total_commit"], dtype=np.int64
        )

    def _finalize_step(
        self, pre_total_commit: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, StepInfo]:
        """Shared tail for `step` and `step_hybrid`: pack payouts/obs
        and emit the done flag."""
        post_total_commit = self._read_total_commit()
        commit_delta = post_total_commit - pre_total_commit
        done = bool(self._rs.is_terminal())
        if done:
            if self._ev_runout_samples > 0:
                ev_seed = (
                    self._reset_seed ^ 0x9E3779B97F4A7C15
                ) & ((1 << 64) - 1)
                rewards = np.asarray(
                    self._rs.payouts_ev(self._ev_runout_samples, ev_seed),
                    dtype=np.float32,
                )
            else:
                rewards = np.asarray(self._rs.payouts(), dtype=np.float32)
            obs_vec = np.zeros(OBS_DIM, dtype=np.float32)
            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            gate_mask = np.zeros(GATE_ACTIONS, dtype=bool)
            raw = dict(self._rs.observation_dict())
            info = StepInfo(
                legal_mask=mask,
                gate_mask=gate_mask,
                min_raise_chips=0,
                max_raise_chips=0,
                actor=None,
                terminal=True,
                raw_obs=raw,
                commit_delta=commit_delta,
                total_commit=post_total_commit,
            )
            self._last_obs_vec = obs_vec
            self._last_mask = mask
            return obs_vec, rewards, True, info
        rewards = np.zeros(self.config.num_seats, dtype=np.float32)
        obs_vec, info = self._pack_obs()
        info.commit_delta = commit_delta
        info.total_commit = post_total_commit
        return obs_vec, rewards, False, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, np.ndarray, bool, StepInfo]:
        pre = self._read_total_commit()
        self._rs.apply_action(int(action))
        return self._finalize_step(pre)

    def step_hybrid(
        self, gate: int, raise_chips: int = 0
    ) -> tuple[np.ndarray, np.ndarray, bool, StepInfo]:
        """Apply a gate-space action. `raise_chips` is required for
        `gate == GATE_RAISE` and ignored otherwise.

        Short-shove redirect: when the engine zeros `min_raise_chips`
        but `legal[ALL_IN]` is set (sub-min-raise stack), GATE_RAISE
        dispatches `apply(ALL_IN)` instead of `apply_raise_chips`
        (which would reject any chips when `min == 0`). The chip
        amount is ignored in that regime."""
        pre = self._read_total_commit()
        g = int(gate)
        if g == GATE_FOLD:
            self._rs.apply_action(FOLD)
        elif g == GATE_CHECK_CALL:
            self._rs.apply_action(CHECK_CALL)
        elif g == GATE_RAISE:
            if (
                self._rs.min_raise_chips() == 0
                and self._last_mask is not None
                and bool(self._last_mask[ALL_IN])
            ):
                self._rs.apply_action(ALL_IN)
            else:
                self._rs.apply_raise_chips(int(raise_chips))
        else:
            raise ValueError(f"invalid gate {g}; must be 0..=2")
        return self._finalize_step(pre)

    @property
    def num_seats(self) -> int:
        return self.config.num_seats

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    def current_actor(self) -> int | None:
        return self._rs.current_actor()

    def is_terminal(self) -> bool:
        return bool(self._rs.is_terminal())

    def all_hole_cards(self) -> list[list[int]]:
        """Every seat's 5 hole cards as raw indices. Trainer-only reveal
        accessor — never feed into observations mid-hand."""
        return [[int(c) for c in hole] for hole in self._rs.all_hole_cards()]

    def legal_action_mask(self) -> np.ndarray:
        return np.asarray(self._rs.legal_action_mask(), dtype=bool)

    def _pack_obs(self) -> tuple[np.ndarray, StepInfo]:
        raw = dict(self._rs.observation_dict())
        mask = np.asarray(self._rs.legal_action_mask(), dtype=bool)
        min_raise = int(raw.get("min_raise", 0))
        max_raise = int(raw.get("max_raise", 0))
        gate_mask = gate_mask_from_bounds(mask, max_raise, self.config.bb)
        actor = raw["actor"]
        if actor is not None:
            raw["hero_category_a"] = int(self._rs.hero_category(actor, 0))
            raw["hero_category_b"] = int(self._rs.hero_category(actor, 1))
        vec = encode_observation(raw, self.config)
        self._last_obs_vec = vec
        self._last_mask = mask
        n = self.config.num_seats
        total_commit = np.asarray(raw.get("total_commit", []), dtype=np.int64)
        if total_commit.size != n:
            total_commit = np.zeros(n, dtype=np.int64)
        info = StepInfo(
            legal_mask=mask,
            gate_mask=gate_mask,
            min_raise_chips=min_raise,
            max_raise_chips=max_raise,
            actor=actor,
            terminal=False,
            raw_obs=raw,
            commit_delta=np.zeros(n, dtype=np.int64),
            total_commit=total_commit,
        )
        return vec, info
