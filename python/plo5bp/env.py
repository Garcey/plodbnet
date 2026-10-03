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
from plo5bp.config import VARIANT_NLH, GameConfig
from plo5bp.encoding import (
    OBS_DIM,
    OBS_DIM_MINIMAL,
    OBS_SEMANTICS_REV,
    encode_observation_minimal,
    encode_observation,
)
from plo5bp.encoding_nlh import OBS_DIM_NLH, encode_observation_nlh
from plo5bp import encoding as _encoding  # module handle: OBS_SEMANTICS_REV is read late
from plo5bp.engine_abi import functions as _engine_functions
from plo5bp.engine_abi import require as _engine_require

# The per-step bookkeeping reads total_commit through its own getter (PERF-027).
_engine_require("GameState.total_commit")

# The engine's one-row observation encode (ML-008): what training encodes with,
# bit-identical to the scalar numpy encoders (tests/python/engine/test_serial_encode.py).
# An engine built before ML-008 has none: engine_abi refuses it at import.
(_engine_encode_state,) = _engine_functions("encode_game_state")
# Variants whose hands keep their size -- the engine encoder's layouts. NLH
# (numpy-only features) and PLO67 (hands grow on red burns) keep numpy.
_ENGINE_ENCODED_VARIANTS = frozenset(
    {"plo4_double_bomb", "plo5_double_bomb", "plo6_double_bomb"}
)


def _engine_obs_rev_kwargs(engine_cls) -> dict[str, int]:
    """Pin the Rust engine to the SAME observation-semantics revision the
    Python encoders read at import (review 2026-09-20): both sides parse
    `PLO5BP_OBS_REV`, but the engine reads it at construction, so an env var
    changed after import would otherwise let the fused Rust encoder and the
    numpy encoders disagree silently. (An engine built before the switch has
    no `obs_rev` argument; encoding.py refuses it at import -- engine_abi.)"""
    return {"obs_rev": int(OBS_SEMANTICS_REV)}


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
        obs_mode: str = "full",
    ):
        self.config = config or GameConfig()
        mode = str(obs_mode or "full").strip().lower()
        if mode not in ("full", "minimal"):
            raise ValueError(
                f"obs_mode must be 'full' or 'minimal', got {obs_mode!r}"
            )
        if mode == "minimal" and self.config.variant == VARIANT_NLH:
            raise ValueError("obs_mode=minimal is PLO-only (not NLH)")
        self._obs_mode = mode
        stacks = np.asarray(self.config.resolved_stacks, dtype=np.uint64)
        self._rs = _RustGameState(
            num_seats=self.config.num_seats,
            starting_stack=0,
            ante=self.config.ante,
            bb=self.config.bb,
            starting_stacks=stacks,
            variant=self.config.variant,
            sb=self.config.sb,
            reach_cap=bool(self.config.reach_cap),
            **_engine_obs_rev_kwargs(_RustGameState),
        )
        # Per-variant observation layout: PLO full 1171 / minimal 796, NLH 995.
        if self.config.variant == VARIANT_NLH:
            self._obs_dim = OBS_DIM_NLH
            self._encode = encode_observation_nlh
        elif mode == "minimal":
            self._obs_dim = OBS_DIM_MINIMAL
            self._encode = encode_observation_minimal
        else:
            self._obs_dim = OBS_DIM
            self._encode = encode_observation
        # The engine encodes the PLO layouts (ML-008); `_encode` (numpy) stays
        # for NLH / PLO67 and as the tests' oracle.
        self._engine_encode = self.config.variant in _ENGINE_ENCODED_VARIANTS
        self._last_obs_vec = np.zeros(self._obs_dim, dtype=np.float32)
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

    def reset_with_deck(
        self,
        deck: "list[int] | bytes",
        button: int,
        in_hand_mask: list[bool] | None = None,
    ) -> tuple[np.ndarray, StepInfo]:
        """:meth:`reset` from an EXPLICIT deck order (52 distinct card indices,
        first-dealt first) instead of a seed. Same public deal contract: five
        cards per seat index (every index, dealt in or not), seat 0 first, then
        full board A, then full board B. The home games deal this way so the
        deck can be sealed and re-permuted by the players' devices
        (``plo5bp.ui.fairdeal``). Actual payouts only — there is no seed for the
        training-time EV runout (``ev_runout_samples`` must be 0)."""
        if self._ev_runout_samples > 0:
            raise ValueError("reset_with_deck has no seed for ev_runout_samples > 0")
        order = [int(c) for c in deck]
        if in_hand_mask is None:
            self._rs.reset_with_deck(order, button)
        else:
            self._rs.reset_with_deck(order, button, list(in_hand_mask))
        self._reset_seed = 0
        return self._pack_obs()

    @staticmethod
    def shuffled_deck(seed: int) -> list[int]:
        """The deck order :meth:`reset` deals from for ``seed`` (parity tests)."""
        return [int(c) for c in _RustGameState.shuffled_deck(int(seed))]

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

    def reset_study_nlh(
        self, button: int, hero_seat: int, hero_hole: list[int]
    ) -> tuple[np.ndarray, StepInfo]:
        """NLH study entry: 2-card hero hole, hand starts PREFLOP with
        blinds posted. Streets arrive via the *_nlh setters."""
        self._rs.reset_study_nlh(button, hero_seat, list(hero_hole))
        return self._pack_obs()

    def set_flop_nlh(
        self, c0: int, c1: int, c2: int
    ) -> tuple[np.ndarray, StepInfo]:
        self._rs.set_flop_nlh(int(c0), int(c1), int(c2))
        return self._pack_obs()

    def set_turn_nlh(self, card: int) -> tuple[np.ndarray, StepInfo]:
        self._rs.set_turn_nlh(int(card))
        return self._pack_obs()

    def set_river_nlh(self, card: int) -> tuple[np.ndarray, StepInfo]:
        self._rs.set_river_nlh(int(card))
        return self._pack_obs()

    def pack_range_nlh(self, holes: np.ndarray) -> dict[str, np.ndarray]:
        """Range-grid support: pack the CURRENT decision node once per
        candidate actor hole (N, 2 card indices), in the batched-packer
        layout `encode_observation_batch_nlh` consumes. NLH-only; the
        observation is villain-blind so only the hole-derived fields
        differ across rows."""
        return dict(
            self._rs.pack_range_nlh(np.ascontiguousarray(holes, dtype=np.uint8))
        )

    def awaiting_next_street(self) -> int | None:
        return self._rs.awaiting_next_street()

    def study_terminal(self) -> int | None:
        return self._rs.study_terminal()

    def _read_total_commit(self) -> np.ndarray:
        # Bookkeeping read (twice per step): the engine's getter, not a whole
        # observation dict (PERF-027) — the same numbers as its total_commit.
        return np.asarray(self._rs.total_commit(), dtype=np.int64)

    def _finalize_step(
        self, pre_total_commit: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, StepInfo]:
        """Shared tail for `step` and `step_hybrid`: pack payouts/obs
        and emit the done flag."""
        post_total_commit = self._read_total_commit()
        commit_delta = post_total_commit - pre_total_commit
        done = bool(self._rs.is_terminal())
        if done:
            rewards = self.terminal_rewards()
            obs_vec = np.zeros(self._obs_dim, dtype=np.float32)
            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            gate_mask = np.zeros(GATE_ACTIONS, dtype=bool)
            # Terminal: no actor, so the feature slots are zeros with or
            # without the flag (same keys, same values) — it just keeps the
            # nothing-to-compute intent explicit.
            raw = dict(self._rs.observation_dict(skip_outcome_mc=True))
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

    def terminal_rewards(self) -> np.ndarray:
        """Per-seat chip deltas of the finished hand (EV-marginalised over
        the undealt runout when `ev_runout_samples > 0`, same seed contract
        as `_finalize_step`). Public because a hand can be terminal AT DEAL
        (fewer than two seats able to act — review 2026-09-20 C1), where no
        `step` ever returns the rewards. Zeros while the hand is live."""
        if self._ev_runout_samples > 0:
            ev_seed = (
                self._reset_seed ^ 0x9E3779B97F4A7C15
            ) & ((1 << 64) - 1)
            return np.asarray(
                self._rs.payouts_ev(self._ev_runout_samples, ev_seed),
                dtype=np.float32,
            )
        return np.asarray(self._rs.payouts(), dtype=np.float32)

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
        return self._obs_dim

    def current_actor(self) -> int | None:
        return self._rs.current_actor()

    def is_terminal(self) -> bool:
        return bool(self._rs.is_terminal())

    def all_hole_cards(self) -> list[list[int]]:
        """Every seat's hole cards (variant hole count) as raw indices. Trainer-only reveal
        accessor — never feed into observations mid-hand."""
        return [[int(c) for c in hole] for hole in self._rs.all_hole_cards()]

    # --- read-only accessors for the home games (HGB-023) ----------------------
    # homegame.py used to reach through the private ``_rs`` handle for these, so
    # an env change broke the tables at runtime instead of at import. Exact
    # pass-throughs: nothing is computed here.

    def observation_dict(self) -> dict:
        """The engine's raw state as a plain dict (the opp-outcome Monte Carlo
        skipped — it only feeds network observations)."""
        return dict(self._rs.observation_dict(skip_outcome_mc=True))

    def payouts(self) -> list[int]:
        """Exact per-seat chip deltas of the finished hand vs its start (won −
        committed; they sum to zero). Zeros while the hand is live."""
        return [int(x) for x in self._rs.payouts()]

    def all_burns(self) -> list[int]:
        """PLO67: all three burn cards of the hand — a reveal accessor, like
        ``all_hole_cards`` (the table shows them street by street). [] for the
        variants that burn nothing."""
        return [int(c) for c in self._rs.all_burns()]

    def hole_count_on(self, seat: int, street: int) -> int:
        """PLO67: the hole cards ``seat`` held on ``street`` (1 flop, 2 turn,
        3 river) — its extras are a prefix of the red burns."""
        return int(self._rs.hole_count_on(int(seat), int(street)))

    def legal_action_mask(self) -> np.ndarray:
        return np.asarray(self._rs.legal_action_mask(), dtype=bool)

    def _pack_obs(self) -> tuple[np.ndarray, StepInfo]:
        # The full layout's raw dict carries the opp-outcome MC (the ONE run
        # per decision: the engine encode below reuses it); the minimal layout
        # has no MC dims.
        minimal = getattr(self, "_obs_mode", "full") == "minimal"
        raw = dict(self._rs.observation_dict(skip_outcome_mc=minimal))
        mask = np.asarray(self._rs.legal_action_mask(), dtype=bool)
        min_raise = int(raw.get("min_raise", 0))
        max_raise = int(raw.get("max_raise", 0))
        # (the website offers the covering bet into a short stack's last chips — only
        # dust under bb/100 stays screened; training screens anything under 1bb)
        screen = max(1, self.config.bb // 100) if self.config.cover_short_bets else self.config.bb
        gate_mask = gate_mask_from_bounds(mask, max_raise, screen)
        actor = raw["actor"]
        if actor is not None:
            raw["hero_category_a"] = int(self._rs.hero_category(actor, 0))
            # Board B is empty for single-board variants; the engine
            # returns 0 before evaluating, so no variant branch needed.
            raw["hero_category_b"] = int(self._rs.hero_category(actor, 1))
        if self._engine_encode and isinstance(self._rs, _RustGameState):
            vec = self._engine_obs(raw, minimal)
        else:
            # NLH / PLO67, or a stand-in for the engine state (a test's
            # forwarding proxy): the numpy encoder, the engine's oracle.
            vec = self._encode(raw, self.config)
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
            # A hand can be terminal AT DEAL (fewer than two seats able to act
            # after posting — review 2026-09-20 C1): no actor and not a study
            # street boundary. Every `while not info.terminal:` driver relies
            # on this; rewards for such a hand come from `terminal_rewards()`.
            terminal=actor is None and raw.get("awaiting_next_street") is None,
            raw_obs=raw,
            commit_delta=np.zeros(n, dtype=np.int64),
            total_commit=total_commit,
        )
        return vec, info

    def _engine_obs(self, raw: dict[str, Any], minimal: bool) -> np.ndarray:
        """The engine's encode of the current node (ML-008), at the revision
        the Python side reads NOW (as the numpy encoders did — tests re-pin
        it). The full layout reuses the opp-outcome block `raw` already holds
        instead of running the MC a second time."""
        rev = int(_encoding.OBS_SEMANTICS_REV)
        if minimal:
            vec = _engine_encode_state(self._rs, "minimal", opp_outcome_mc=0, obs_rev=rev)
        else:
            outcome = (
                list(raw["opp_outcome_fractions"])
                + list(raw["per_board_outcome"])
                + list(raw["share_bounds"])
            )
            vec = _engine_encode_state(self._rs, "full", outcome=outcome, obs_rev=rev)
        return np.asarray(vec, dtype=np.float32)
