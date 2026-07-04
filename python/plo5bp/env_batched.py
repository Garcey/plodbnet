"""Batched multi-seat env (PLO double-bomb family + NLH).

Array-shaped analogue of `BombPotEnv`. Owns a single `PyBatchedEngine` and
returns observations / rewards / legal masks as stacked NumPy arrays for
all N envs at once. Does *not* auto-reset terminal envs — the caller uses
`reset_terminal_batch` to re-seed the envs whose `dones[i]` came back true.
The observation layout follows the config's variant (991-dim PLO or
995-dim NLH), same as the scalar env.

Design goals:
  - Bit-exact parity with `BombPotEnv` on (obs, legal_mask, reward, done)
    for matched (seed, button, action-sequence). Enforced by
    `tests/python/test_env_batched.py`.
  - Single FFI call per "step" group: one `apply_action_batch` plus one
    `hero_category_batch` (×2 boards), one `legal_mask_batch`, one
    `actor_batch`, one `observation_arrays`.
  - Vectorized encoder (`encode_observation_batch`) — no per-env Python
    loop anywhere on the hot path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from torch.profiler import record_function

from plo5bp._engine import BatchedEngine  # type: ignore[attr-defined]
from plo5bp.actions import (
    ALL_IN,
    GATE_ACTIONS,
    GATE_RAISE,
    NUM_ACTIONS,
    gate_mask_from_bounds,
)
from plo5bp.config import GameConfig, VARIANT_NLH
from plo5bp.encoding import OBS_DIM, encode_observation_batch
from plo5bp.encoding_nlh import OBS_DIM_NLH, encode_observation_batch_nlh


@dataclass
class BatchedStep:
    """Return value of `step_batch`. All fields are fixed-shape arrays."""

    obs: np.ndarray             # (N, OBS_DIM) float32 — zeros for terminal rows
    rewards: np.ndarray         # (N, num_seats) float32 — per-seat chip delta
    dones: np.ndarray           # (N,) bool — env is currently terminal
    newly_terminal: np.ndarray  # (N,) bool — env transitioned this step
    legal_mask: np.ndarray      # (N, NUM_ACTIONS) bool — discrete/legacy
    gate_mask: np.ndarray       # (N, GATE_ACTIONS) bool — hybrid-head legality
    min_raise: np.ndarray       # (N,) u64 — continuous-raise lower bound
    max_raise: np.ndarray       # (N,) u64 — continuous-raise upper bound
    actors: np.ndarray          # (N,) int8 — -1 for terminal


class BatchedBombPotEnv:
    def __init__(
        self,
        num_envs: int,
        config: GameConfig | None = None,
        ev_runout_samples: int = 0,
        opp_outcome_mc: int = 1024,
    ):
        self.n = int(num_envs)
        self.config = config or GameConfig()
        # Per-variant observation layout, mirroring the scalar env: the
        # PLO double-bomb family shares the 991-dim layout; NLH is 995.
        self._is_nlh = self.config.variant == VARIANT_NLH
        self._obs_dim = OBS_DIM_NLH if self._is_nlh else OBS_DIM
        # k=3/k=4 Monte-Carlo budget for the opp-outcome obs feature.
        # 1024 (serial/UI/eval fidelity) by default; training rollout
        # passes a lower count for speed. See rollout.TRAIN_OPP_OUTCOME_MC.
        # NLH ignores it (its 3-dim opp-outcome block is exhaustive).
        self._opp_outcome_mc = int(opp_outcome_mc)
        # Default-off switch for the Rust observation encoder. When set, the
        # finished obs comes straight from the engine (one FFI call, no numpy
        # assembly); otherwise the numpy `encode_observation_batch` reference
        # path runs. Bit-exact equivalent (test_encoding_batch.py); the switch
        # exists so a bit-mismatch surfacing in a long run can be reverted
        # instantly with `PLO5_RUST_ENCODER=0` and a restart — no rebuild.
        # Falls back to numpy if the rebuilt engine lacks the method.
        # PLO-only: the Rust encoder implements the 991-dim PLO layout
        # (the engine refuses it for NLH), so NLH always encodes in numpy.
        self._use_rust_encoder = (
            bool(int(os.environ.get("PLO5_RUST_ENCODER", "0")))
            and hasattr(BatchedEngine, "observation_encoded_batch")
            and not self._is_nlh
        )
        stacks = np.asarray(self.config.resolved_stacks, dtype=np.uint64)
        self._be = BatchedEngine(
            self.n,
            num_seats=self.config.num_seats,
            starting_stack=0,
            ante=self.config.ante,
            bb=self.config.bb,
            starting_stacks=stacks,
            opp_outcome_mc=self._opp_outcome_mc,
            variant=self.config.variant,
            sb=self.config.sb,
        )
        self._ev_runout_samples = int(ev_runout_samples)
        self._reset_seeds = np.zeros(self.n, dtype=np.uint64)

        # Cached "current" arrays; refreshed after every reset/step.
        self._obs = np.zeros((self.n, self._obs_dim), dtype=np.float32)
        self._legal = np.zeros((self.n, NUM_ACTIONS), dtype=bool)
        self._gate_mask = np.zeros((self.n, GATE_ACTIONS), dtype=bool)
        self._min_raise = np.zeros(self.n, dtype=np.uint64)
        self._max_raise = np.zeros(self.n, dtype=np.uint64)
        self._actors = np.full(self.n, -1, dtype=np.int8)
        self._dones = np.ones(self.n, dtype=bool)
        # (N, num_seats) i64 — cumulative chips committed per seat per
        # env this hand. Refreshed every `_refresh()`. Rollout reads pre
        # and post step to derive `commit_delta` for the forward-EV
        # per-step training reward.
        self._total_commit = np.zeros(
            (self.n, self.config.num_seats), dtype=np.int64
        )
        # (N,) u64 — current max street_commit per env. Used by rollout
        # to compute the actor's pre-step amount-to-call for the
        # aggression-bonus reward shaping.
        self._bet_to_call = np.zeros(self.n, dtype=np.uint64)
        # (N, num_seats) u64 — current per-seat street_commit per env.
        # `bet_to_call - street_commit[actor]` is the actor's call portion;
        # any chips committed beyond that are voluntary aggression.
        self._street_commit = np.zeros(
            (self.n, self.config.num_seats), dtype=np.uint64
        )
        # (N,) u8 — current street index per env (0=preflop, 1=flop,
        # 2=turn, 3=river, 4=showdown). Bomb pots start at 1; rollout
        # consults this to bucket per-street aggression-bonus diagnostics.
        self._street = np.zeros(self.n, dtype=np.uint8)

    @property
    def num_envs(self) -> int:
        return self.n

    @property
    def num_seats(self) -> int:
        return self.config.num_seats

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def reset_batch(
        self, seeds: np.ndarray, buttons: np.ndarray
    ) -> BatchedStep:
        """Reset all envs. Returns the initial observation batch with
        `rewards=0` and `newly_terminal=False` for every env."""
        seeds_u64 = np.ascontiguousarray(seeds, dtype=np.uint64)
        buttons_u8 = np.ascontiguousarray(buttons, dtype=np.uint8)
        if seeds_u64.shape != (self.n,) or buttons_u8.shape != (self.n,):
            raise ValueError(
                f"reset_batch expects shape ({self.n},); got "
                f"seeds={seeds_u64.shape}, buttons={buttons_u8.shape}"
            )
        self._be.reset_batch(seeds_u64, buttons_u8)
        self._reset_seeds = seeds_u64.copy()
        self._refresh()
        return self._snapshot(
            rewards=np.zeros((self.n, self.num_seats), dtype=np.float32),
            newly_terminal=np.zeros(self.n, dtype=bool),
        )

    def reset_terminal_batch(
        self,
        seeds: np.ndarray,
        buttons: np.ndarray,
        mask: np.ndarray,
    ) -> BatchedStep:
        """Re-seed the envs where `mask[i]` is true. Non-masked envs are
        untouched. Returns the refreshed full-batch view; `rewards=0` and
        `newly_terminal=False` because this is a reset, not a step."""
        mask_b = np.ascontiguousarray(mask, dtype=bool)
        seeds_u64 = np.ascontiguousarray(seeds, dtype=np.uint64)
        buttons_u8 = np.ascontiguousarray(buttons, dtype=np.uint8)
        if (
            seeds_u64.shape != (self.n,)
            or buttons_u8.shape != (self.n,)
            or mask_b.shape != (self.n,)
        ):
            raise ValueError(
                f"reset_terminal_batch expects shape ({self.n},)"
            )
        self._be.reset_terminal_batch(seeds_u64, buttons_u8, mask_b)
        # Update stored reset seeds only for masked envs (others unchanged).
        self._reset_seeds = np.where(mask_b, seeds_u64, self._reset_seeds)
        self._refresh()
        return self._snapshot(
            rewards=np.zeros((self.n, self.num_seats), dtype=np.float32),
            newly_terminal=np.zeros(self.n, dtype=bool),
        )

    def step_batch(self, actions: np.ndarray) -> BatchedStep:
        """Apply one discrete action per env. Terminal envs are no-op.
        Rewards populated only on the step that transitions an env to
        terminal; mid-hand steps yield zeros."""
        actions_u8 = np.ascontiguousarray(actions, dtype=np.uint8)
        if actions_u8.shape != (self.n,):
            raise ValueError(
                f"actions expected shape ({self.n},); got {actions_u8.shape}"
            )
        newly_terminal = np.asarray(
            self._be.apply_action_batch(actions_u8), dtype=bool
        )
        return self._post_apply(newly_terminal)

    def step_hybrid_batch(
        self, gates: np.ndarray, raise_chips: np.ndarray
    ) -> BatchedStep:
        """Apply a hybrid (gate, raise_chips) pair per env. `gates[i]` in
        `{0=Fold, 1=CheckCall, 2=Raise}`; `raise_chips[i]` is only read
        when `gates[i] == 2`. Stack-bound short shoves are encoded as
        Raise — the engine's `max_raise_chips` already clamps to stack.

        Short-shove redirect: rows where `gates == GATE_RAISE`,
        `min_raise == 0`, and `legal[..., ALL_IN]` are remapped to the
        Rust dispatcher's AllIn arm (gate=3). The chip amount is ignored
        for those rows; the Rust AllIn arm reads `legal[AllIn]` and
        emits `apply(Action::AllIn)` directly. This mirrors the scalar
        env's `step_hybrid` redirect and keeps sub-min-raise shoves
        reachable from the network's 3-gate space.
        """
        gates_u8 = np.ascontiguousarray(gates, dtype=np.uint8)
        chips_u64 = np.ascontiguousarray(raise_chips, dtype=np.uint64)
        if gates_u8.shape != (self.n,) or chips_u64.shape != (self.n,):
            raise ValueError(
                f"step_hybrid_batch expects shape ({self.n},); got "
                f"gates={gates_u8.shape}, raise_chips={chips_u64.shape}"
            )
        short_shove = (
            (gates_u8 == GATE_RAISE)
            & (self._min_raise == np.uint64(0))
            & self._legal[:, ALL_IN]
        )
        if short_shove.any():
            gates_u8 = gates_u8.copy()
            gates_u8[short_shove] = 3
        newly_terminal = np.asarray(
            self._be.apply_hybrid_batch(gates_u8, chips_u64), dtype=bool
        )
        return self._post_apply(newly_terminal)

    def _post_apply(self, newly_terminal: np.ndarray) -> BatchedStep:
        rewards = np.zeros((self.n, self.num_seats), dtype=np.float32)
        if newly_terminal.any():
            if self._ev_runout_samples > 0:
                ev_seeds = self._reset_seeds ^ np.uint64(0x9E3779B97F4A7C15)
                payouts = self._be.payouts_ev_batch(
                    self._ev_runout_samples, ev_seeds
                )
            else:
                payouts = self._be.payouts_batch()
            payouts_f32 = np.asarray(payouts, dtype=np.float32)
            rows = np.nonzero(newly_terminal)[0]
            rewards[rows] = payouts_f32[rows]

        self._refresh()
        return self._snapshot(rewards=rewards, newly_terminal=newly_terminal)

    def _snapshot(
        self, rewards: np.ndarray, newly_terminal: np.ndarray
    ) -> BatchedStep:
        return BatchedStep(
            obs=self._obs.copy(),
            rewards=rewards,
            dones=self._dones.copy(),
            newly_terminal=newly_terminal,
            legal_mask=self._legal.copy(),
            gate_mask=self._gate_mask.copy(),
            min_raise=self._min_raise.copy(),
            max_raise=self._max_raise.copy(),
            actors=self._actors.copy(),
        )

    def _refresh(self) -> None:
        """Re-encode observations and refresh cached arrays from the engine."""
        if self._use_rust_encoder:
            with record_function("step1a_bundle/obs_features_batch"):
                bundle = self._be.observation_encoded_batch()
            self._obs = np.asarray(bundle["obs"], dtype=np.float32)
        else:
            with record_function("step1a_bundle/obs_features_batch"):
                bundle = self._be.observation_and_features_batch()
            with record_function("step1a_unpack/actor"):
                cat_a = np.asarray(bundle["hero_cat_a"])
                cat_b = np.asarray(bundle["hero_cat_b"])
            with record_function("step1/encoder"):
                if self._is_nlh:
                    self._obs = encode_observation_batch_nlh(
                        bundle, cat_a, self.config
                    )
                else:
                    self._obs = encode_observation_batch(
                        bundle, cat_a, cat_b, self.config
                    )
        with record_function("step1a_unpack/post"):
            actors = np.asarray(bundle["actor"], dtype=np.int8)
            dones = actors == -1
            self._legal = np.asarray(bundle["legal_mask"], dtype=bool)
            self._min_raise = np.asarray(bundle["min_raise"], dtype=np.uint64)
            self._max_raise = np.asarray(bundle["max_raise"], dtype=np.uint64)
            self._gate_mask = gate_mask_from_bounds(
                self._legal, self._max_raise, self.config.bb
            )
            # Terminal envs: zero everything so downstream code can rely on
            # "dones → no legal action".
            self._gate_mask[dones] = False
            self._actors = actors
            self._dones = dones
            self._total_commit = np.asarray(bundle["total_commit"], dtype=np.int64)
            self._bet_to_call = np.asarray(bundle["bet_to_call"], dtype=np.uint64)
            self._street_commit = np.asarray(bundle["street_commit"], dtype=np.uint64)
            self._street = np.asarray(bundle["street"], dtype=np.uint8)
            self._pot = np.asarray(bundle["pot"], dtype=np.uint64)

    def _refresh_subset(self, mask: np.ndarray) -> None:
        """Partial `_refresh`: re-pack + re-encode ONLY the envs selected by
        `mask`, scattering the results into the cached arrays in place and
        leaving every other env's cached state untouched.

        This is bit-exact-equivalent to a full `_refresh()` whenever the
        non-masked envs' engine state is unchanged since the previous full
        refresh — which is exactly the situation after
        `reset_terminal_batch(mask)`: that call mutates only the masked
        (terminal) envs, and the rollout loop performs no engine mutation of
        non-masked envs between the two refreshes. The whole-batch
        `encode_observation_batch` is purely per-row, so encoding the masked
        subset and scattering gives results identical to encoding the full
        batch and slicing. Parity is asserted in
        `tests/python/test_refresh_subset_parity.py`.

        Skips all work when `mask` selects nothing.
        """
        idx = np.nonzero(np.ascontiguousarray(mask, dtype=bool))[0]
        if idx.size == 0:
            return
        idx_i64 = idx.astype(np.int64)
        if self._use_rust_encoder:
            with record_function("step1a_bundle/obs_features_subset"):
                bundle = self._be.observation_encoded_subset_batch(idx_i64)
            obs_sub = np.asarray(bundle["obs"], dtype=np.float32)
            actors_sub = np.asarray(bundle["actor"], dtype=np.int8)
        else:
            with record_function("step1a_bundle/obs_features_subset"):
                bundle = self._be.observation_and_features_subset_batch(idx_i64)
            with record_function("step1a_unpack/actor"):
                actors_sub = np.asarray(bundle["actor"], dtype=np.int8)
                cat_a = np.asarray(bundle["hero_cat_a"])
                cat_b = np.asarray(bundle["hero_cat_b"])
            with record_function("step1/encoder"):
                if self._is_nlh:
                    obs_sub = encode_observation_batch_nlh(
                        bundle, cat_a, self.config
                    )
                else:
                    obs_sub = encode_observation_batch(
                        bundle, cat_a, cat_b, self.config
                    )

        with record_function("step1a_unpack/post"):
            dones_sub = actors_sub == -1
            legal_sub = np.asarray(bundle["legal_mask"], dtype=bool)
            min_raise_sub = np.asarray(bundle["min_raise"], dtype=np.uint64)
            max_raise_sub = np.asarray(bundle["max_raise"], dtype=np.uint64)
            gate_sub = gate_mask_from_bounds(
                legal_sub, max_raise_sub, self.config.bb
            )
            # Mirror the full-refresh terminal rule on the subset rows.
            gate_sub[dones_sub] = False

            # Scatter compact (k-row) results back into the full cached
            # arrays at the masked indices. Non-masked rows are left as-is
            # (still valid from the preceding full refresh).
            self._obs[idx] = obs_sub
            self._legal[idx] = legal_sub
            self._gate_mask[idx] = gate_sub
            self._min_raise[idx] = min_raise_sub
            self._max_raise[idx] = max_raise_sub
            self._actors[idx] = actors_sub
            self._dones[idx] = dones_sub
            self._total_commit[idx] = np.asarray(bundle["total_commit"], dtype=np.int64)
            self._bet_to_call[idx] = np.asarray(bundle["bet_to_call"], dtype=np.uint64)
            self._street_commit[idx] = np.asarray(
                bundle["street_commit"], dtype=np.uint64
            )
            self._street[idx] = np.asarray(bundle["street"], dtype=np.uint8)
            self._pot[idx] = np.asarray(bundle["pot"], dtype=np.uint64)

    # ------------------------------------------------------------------
    # Read-only accessors mirroring the scalar env's helpers.
    # ------------------------------------------------------------------

    def legal_action_mask(self) -> np.ndarray:
        return self._legal.copy()

    def gate_mask(self) -> np.ndarray:
        return self._gate_mask.copy()

    def raise_bounds(self) -> np.ndarray:
        """`(N, 2)` u64 — (min_raise_chips, max_raise_chips) per env."""
        return np.stack([self._min_raise, self._max_raise], axis=-1)

    def current_actors(self) -> np.ndarray:
        return self._actors.copy()

    def is_terminal(self) -> np.ndarray:
        return self._dones.copy()

    def observation(self) -> np.ndarray:
        return self._obs.copy()
