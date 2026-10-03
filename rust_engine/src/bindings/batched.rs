//! `BatchedEngine` (PyO3): N independent hands behind one FFI call per
//! step for the training rollouts. The one `#[pymethods]` block lives here;
//! the packers, encoders and their plumbing are in pack.rs / encode*.rs.

use super::*;

/// Minimum envs per rayon leaf for the cheap per-env passes (apply /
/// validation): a few leaves keep the handful of woken workers busy instead
/// of waking every worker for sub-microsecond slices of work.
pub(super) const APPLY_MIN_LEN: usize = 512;

/// Deal a new hand into an env slot: re-dealt in place when the slot holds a
/// previous hand (its buffers reused -- `GameState::redeal`, PERF-034), a
/// fresh state otherwise. Either way exactly `GameState::new_hand`.
pub(super) fn deal_into(st: &mut Option<GameState>, config: &GameConfig, seed: u64, button: usize) {
    match st {
        Some(g) => g.redeal(config, seed, button),
        None => *st = Some(GameState::new_hand(config.clone(), seed, button)),
    }
}

/// [`validate_hybrid_action`] for a discrete action index (the 8-action
/// space): `Some((env_not_reset, message))` for the first problem, `None` when
/// the action is legal or the env is already terminal.
pub(super) fn validate_discrete_action(
    i: usize,
    state: Option<&GameState>,
    a_idx: u8,
) -> Option<(bool, String)> {
    let Some(state) = state else {
        return Some((
            true,
            format!("env {i} not reset; call reset_batch/reset_terminal_batch first"),
        ));
    };
    if state.is_terminal() {
        return None;
    }
    match Action::from_index(a_idx) {
        None => Some((false, format!("invalid action index {a_idx} at env {i}"))),
        Some(a) if !state.legal_action_mask()[a.index() as usize] => {
            Some((false, format!("action {a_idx} illegal at env {i}")))
        }
        Some(_) => None,
    }
}

/// `apply_hybrid_batch`'s legality check for one env: `None` when env `i` may
/// take `gate` (with `chips` for a raise) or is already terminal, else
/// `(true = not reset -> RuntimeError, message)`.
pub(super) fn validate_hybrid_action(
    i: usize,
    state: Option<&GameState>,
    gate: u8,
    chips: u64,
) -> Option<(bool, String)> {
    let state = match state {
        Some(s) => s,
        None => {
            return Some((
                true,
                format!(
                    "env {} not reset; call reset_batch/reset_terminal_batch first",
                    i
                ),
            ))
        }
    };
    if state.is_terminal() {
        return None;
    }
    match gate {
        0 => (!state.fold_is_legal()).then(|| (false, format!("gate Fold illegal at env {}", i))),
        1 => (!state.check_call_is_legal())
            .then(|| (false, format!("gate CheckCall illegal at env {}", i))),
        2 => {
            let min = state.min_raise_chips();
            let max = state.max_raise_chips();
            (min == 0 || chips < min || chips > max).then(|| {
                (
                    false,
                    format!(
                        "gate Raise chips {} out of range [{}, {}] at env {}",
                        chips, min, max, i
                    ),
                )
            })
        }
        3 => {
            (!state.all_in_is_legal()).then(|| (false, format!("gate AllIn illegal at env {}", i)))
        }
        other => Some((
            false,
            format!("invalid gate {} at env {} (must be 0..=3)", other, i),
        )),
    }
}

// ============================================================================
// Batched engine for fast training rollouts.
//
// Holds `num_envs` independent GameStates sharing a GameConfig. All public
// methods take / return stacked NumPy arrays so the Python side can drive N
// envs through a single FFI call. The inner loops release the GIL via
// `py.detach`, so a caller can overlap other Python work with Rust
// engine computation on a worker thread.
//
// Per-env determinism matches the serial engine bit-for-bit: each call takes
// seeds/buttons/actions as arrays, and the engine uses the same
// `ChaCha8Rng::seed_from_u64` derivations (for deck shuffling, equity MC, and
// EV runout sampling) as `PyGameState`. This is what lets Phase A be an
// optimization, not a behavior change.
// ============================================================================

/// NLH's history depth (`encoding_nlh._HISTORY_DEPTH`: the preflop round adds
/// actions). NLH encodes through numpy only, so no Rust encoder reads it.
pub(super) const NLH_HISTORY_DEPTH: usize = 40;

/// Batched-packer history width per variant. Must equal the variant's
/// batch-encoder slot count EXACTLY: the packer keeps the LAST `cap`
/// records oldest-first from slot 0, so a buffer wider than the encoder's
/// depth would hand it the oldest records instead of the newest. PLO: the
/// Rust encoders' own `obs_core::HISTORY_DEPTH` (32, = `encoding._HISTORY_DEPTH`)
/// — one constant, so the packer and the encoders cannot disagree (ENG-016).
pub(super) fn history_cap(variant: Variant) -> usize {
    match variant {
        Variant::NlhSingle => NLH_HISTORY_DEPTH,
        _ => obs_core::HISTORY_DEPTH,
    }
}

/// Python-facing batched game engine. Construct with `(num_envs, config)`,
/// then `reset_batch(seeds, buttons)` to seed all envs. `apply_action_batch`
/// steps every env; terminal envs stay terminal until `reset_terminal_batch`.
#[pyclass(name = "BatchedEngine")]
pub struct PyBatchedEngine {
    pub(super) states: Vec<Option<GameState>>,
    pub(super) config: GameConfig,
    /// k=3/k=4 Monte-Carlo budget for `opp_outcome_fractions` on this
    /// engine. Serial/UI/eval use 1024; batched TRAINING sets it lower
    /// (the rollouts pass `rollout.py`'s budget, 384 at the time of writing)
    /// to cut the dominant per-decision encode cost.
    pub(super) opp_outcome_mc: usize,
    /// Memoization of the opp-outcome MC output (the 22-dim fused pass), one
    /// slot per (env, SEAT), keyed on `GameState::outcome_seed` = hash of
    /// exactly the MC's inputs (street + the actor's hole + both boards). The
    /// MC is a pure function of those, so a seat that acts AGAIN on the same
    /// street (facing a raise after it already acted) reuses its result; a
    /// street advance or a new hand changes the seed and recomputes.
    /// (review 2026-09-20 C6) This used to be ONE slot per env, which never
    /// hit in a rollout: the key includes the actor's hole and the actor
    /// changes on every action, so each pack evicted the previous seat's
    /// entry (0 / 51,200 hits measured).
    /// Bit-exact vs always-recompute (pinned by test_encoding_rust: cached
    /// batched == fresh serial). Accessed only serially (locked outside the
    /// parallel MC), so the Mutex adds no contention and keeps the pyclass Sync.
    /// Every lock of it (and of `board_tables`) recovers a poisoned mutex
    /// (`unwrap_or_else(|e| e.into_inner())`): the memo is plain data, and a
    /// panic elsewhere must not break every later call on this engine for
    /// good (ENG-029).
    pub(super) outcome_cache: std::sync::Mutex<OutcomeCache>,
    /// Per-env board pair-rank table of the opp-outcome MC
    /// (`GameState::board_pair_table`, 2026-09-26): computed once per street
    /// and shared by every seat that acts on it, instead of re-evaluating the
    /// same ~1,000 two-card holdings for each seat. Keyed on the boards, so a
    /// stale table is never used (`outcome_features_mc_shared` checks the key).
    pub(super) board_tables: Vec<std::sync::Mutex<Option<crate::engine::BoardPairTable>>>,
    /// Observation-semantics revision the fused encoders emit, fixed at
    /// construction (see `OBS_REV_ENV`).
    pub(super) obs_rev: u8,
}

/// See `PyBatchedEngine::outcome_cache`. `slots[env * num_seats + seat]`.
pub(super) struct OutcomeCache {
    pub(super) slots: Vec<Option<(u64, RowFeatures)>>,
    /// Lifetime counters behind `outcome_cache_stats()` — the dead single-slot
    /// cache went unnoticed precisely because nothing reported its hit rate.
    pub(super) lookups: u64,
    pub(super) hits: u64,
}

impl OutcomeCache {
    pub(super) fn new(num_envs: usize, num_seats: usize) -> Self {
        OutcomeCache {
            slots: vec![None; num_envs * num_seats],
            lookups: 0,
            hits: 0,
        }
    }

    /// Drop every seat's entry for one env (its hand was re-dealt).
    pub(super) fn clear_env(&mut self, env: usize, num_seats: usize) {
        for slot in &mut self.slots[env * num_seats..(env + 1) * num_seats] {
            *slot = None;
        }
    }
}

#[pymethods]
impl PyBatchedEngine {
    #[new]
    #[pyo3(signature = (num_envs, num_seats=6, starting_stack=200000, ante=30000, bb=10000, starting_stacks=None, opp_outcome_mc=1024, variant="plo5_double_bomb", sb=0, obs_rev=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        num_seats: usize,
        starting_stack: u64,
        ante: u64,
        bb: u64,
        starting_stacks: Option<PyReadonlyArray1<'_, u64>>,
        opp_outcome_mc: usize,
        variant: &str,
        sb: u64,
        obs_rev: Option<u8>,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be >= 1"));
        }
        let variant = parse_variant(variant)?;
        validate_table(num_seats, variant, bb)?;
        if !variant.supports_batched() {
            // The packers lay hole cards out at a fixed width, and PLO67 hands
            // grow mid-hand (face-up burns deal extra cards): serial only.
            return Err(PyValueError::new_err(format!(
                "{} is not supported by the batched engine (nothing trains it yet)",
                variant.name()
            )));
        }
        let obs_rev = resolve_obs_rev(obs_rev)?;
        // opp_outcome_mc == 0 is allowed: skips the fused outcome_features_mc
        // pass entirely (zeros the opp-outcome / per-board / share-bound
        // slots). Used by obs_mode=minimal training which never consumes
        // those features.
        let stacks = resolve_starting_stacks(num_seats, starting_stack, starting_stacks)?;
        let config = GameConfig {
            num_seats,
            starting_stacks: stacks,
            ante,
            bb,
            sb,
            variant,
            reach_cap: true,
        };
        Ok(PyBatchedEngine::with_config(
            config,
            num_envs,
            opp_outcome_mc,
            obs_rev,
        ))
    }

    /// Observation-semantics revision the fused encoders of this engine emit.
    fn obs_rev(&self) -> u8 {
        self.obs_rev
    }

    /// Same as the module-level `obs_semantics_rev()` (the revision
    /// `PLO5BP_OBS_REV` selects right now).
    #[staticmethod]
    #[pyo3(name = "obs_semantics_rev")]
    fn obs_semantics_rev_static() -> PyResult<u8> {
        obs_rev_from_env()
    }

    fn num_envs(&self) -> usize {
        self.states.len()
    }

    /// Update chip-level config (stacks / ante / bb / sb) without reallocating
    /// the per-env state vector. `num_seats` and `variant` must match the
    /// construction-time values (they size hole arrays and history). Used by
    /// multiconfig rollout to reuse one `BatchedEngine` across stack samples
    /// at the same seat count. Clears live hands + the outcome MC cache so
    /// the next `reset_batch` starts clean under the new config.
    #[pyo3(signature = (starting_stacks, ante, bb, sb=0))]
    fn reconfigure(
        &mut self,
        starting_stacks: PyReadonlyArray1<'_, u64>,
        ante: u64,
        bb: u64,
        sb: u64,
    ) -> PyResult<()> {
        // num_seats / variant are fixed at construction; re-check the rest.
        validate_table(self.config.num_seats, self.config.variant, bb)?;
        let stacks = resolve_starting_stacks(
            self.config.num_seats,
            /*starting_stack=*/ 0,
            Some(starting_stacks),
        )?;
        self.config.starting_stacks = stacks;
        self.config.ante = ante;
        self.config.bb = bb;
        self.config.sb = sb;
        // Drop live hands — caller must reset_batch before the next step.
        for st in self.states.iter_mut() {
            *st = None;
        }
        let cache = self
            .outcome_cache
            .get_mut()
            .unwrap_or_else(|e| e.into_inner());
        for slot in cache.slots.iter_mut() {
            *slot = None;
        }
        Ok(())
    }

    /// `(lookups, hits)` of the opp-outcome MC memo since construction. A
    /// lookup is one live PLO row with a flop on both boards; a hit reused
    /// the cached 22-dim result instead of re-running the MC.
    fn outcome_cache_stats(&self) -> (u64, u64) {
        let cache = self.outcome_cache.lock().unwrap_or_else(|e| e.into_inner());
        (cache.lookups, cache.hits)
    }

    fn num_seats(&self) -> usize {
        self.config.num_seats
    }

    /// (N, num_seats, hole_count) u8 hole-card indices for every seat in every
    /// env; rows for dead (never-reset) envs are 255-filled. Holes are
    /// static per hand, so callers cache this once per reset wave —
    /// it feeds the centralized critic's opponent-hole inputs.
    fn all_hole_cards_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let n = self.states.len();
        let s = self.config.num_seats;
        let hole_w = self.config.variant.hole_count();
        let mut arr = numpy::ndarray::Array3::<u8>::from_elem((n, s, hole_w), 255u8);
        for (i, st) in self.states.iter().enumerate() {
            if let Some(g) = st {
                for seat in 0..s {
                    for c in 0..hole_w {
                        arr[[i, seat, c]] = g.hole_cards[seat][c].index();
                    }
                }
            }
        }
        Ok(arr.into_pyarray(py))
    }

    /// Subset variant of `all_hole_cards_batch`: returns compact
    /// `(k, num_seats, hole_w)` rows for the envs in `indices` only.
    /// Used by the rollout after `reset_terminal_batch` to refresh the
    /// per-hand hole cache for re-dealt envs without re-walking every
    /// table (holes are static within a hand).
    fn all_hole_cards_subset_batch<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let idx = checked_env_indices(
            indices.as_slice()?,
            self.states.len(),
            "all_hole_cards_subset_batch",
        )?;
        let k = idx.len();
        let s = self.config.num_seats;
        let hole_w = self.config.variant.hole_count();
        let mut arr = numpy::ndarray::Array3::<u8>::from_elem((k, s, hole_w), 255u8);
        for (j, &ei) in idx.iter().enumerate() {
            if let Some(g) = self.states[ei].as_ref() {
                for seat in 0..s {
                    for c in 0..hole_w {
                        arr[[j, seat, c]] = g.hole_cards[seat][c].index();
                    }
                }
            }
        }
        Ok(arr.into_pyarray(py))
    }

    /// Reset every env with the paired (seed, button). Arrays must both be
    /// length `num_envs`.
    fn reset_batch(
        &mut self,
        py: Python<'_>,
        seeds: PyReadonlyArray1<'_, u64>,
        buttons: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<()> {
        let n = self.states.len();
        let seeds_slice = seeds.as_slice()?;
        let buttons_slice = buttons.as_slice()?;
        if seeds_slice.len() != n || buttons_slice.len() != n {
            return Err(PyValueError::new_err(format!(
                "seeds/buttons length must equal num_envs={}",
                n
            )));
        }
        let num_seats = self.config.num_seats;
        for (i, &b) in buttons_slice.iter().enumerate() {
            if (b as usize) >= num_seats {
                return Err(PyValueError::new_err(format!(
                    "button {} out of range at env {}",
                    b, i
                )));
            }
        }
        // Deal every env IN PLACE, in parallel, into the previous hand's
        // buffers (`deal_into`) -- like `reset_terminal_batch` (PERF-032).
        let config = &self.config;
        let states = &mut self.states;
        py.detach(|| {
            states
                .par_iter_mut()
                .enumerate()
                .with_min_len(64)
                .for_each(|(i, st)| {
                    deal_into(st, config, seeds_slice[i], buttons_slice[i] as usize);
                });
        });
        let cache = self
            .outcome_cache
            .get_mut()
            .unwrap_or_else(|e| e.into_inner());
        for i in 0..n {
            cache.clear_env(i, num_seats);
        }
        Ok(())
    }

    /// For every env where `mask[i]` is true, reset it with `seeds[i]` and
    /// `buttons[i]`. Envs where the mask is false are untouched.
    fn reset_terminal_batch(
        &mut self,
        py: Python<'_>,
        seeds: PyReadonlyArray1<'_, u64>,
        buttons: PyReadonlyArray1<'_, u8>,
        mask: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<()> {
        let n = self.states.len();
        let seeds_slice = seeds.as_slice()?;
        let buttons_slice = buttons.as_slice()?;
        let mask_slice = mask.as_slice()?;
        if seeds_slice.len() != n || buttons_slice.len() != n || mask_slice.len() != n {
            return Err(PyValueError::new_err(format!(
                "seeds/buttons/mask length must equal num_envs={}",
                n
            )));
        }
        let num_seats = self.config.num_seats;
        for (i, &b) in buttons_slice.iter().enumerate() {
            if mask_slice[i] && (b as usize) >= num_seats {
                return Err(PyValueError::new_err(format!(
                    "button {} out of range at env {}",
                    b, i
                )));
            }
        }
        // Deal the masked envs IN PLACE, in parallel, each into the finished
        // hand's own buffers (`deal_into`, PERF-034). The old version built and
        // moved a num_envs-long Vec<Option<GameState>> on every call -- the
        // same deals, at several times the cost when ~1 table in 8 is re-dealt.
        let config = &self.config;
        let states = &mut self.states;
        py.detach(|| {
            states
                .par_iter_mut()
                .enumerate()
                .with_min_len(64)
                .for_each(|(i, st)| {
                    if mask_slice[i] {
                        deal_into(st, config, seeds_slice[i], buttons_slice[i] as usize);
                    }
                });
        });
        let cache = self
            .outcome_cache
            .get_mut()
            .unwrap_or_else(|e| e.into_inner());
        for (i, &m) in mask_slice.iter().enumerate() {
            if m {
                cache.clear_env(i, num_seats);
            }
        }
        Ok(())
    }

    /// Apply `actions[i]` in env `i`. Envs already terminal are skipped
    /// (their `actions[i]` entry is ignored). Returns a `(N,)` bool array
    /// where `true` means the env transitioned to terminal on this call.
    ///
    /// Errors if any action is illegal or out-of-range; in that case no
    /// env state is modified.
    fn apply_action_batch<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let n = self.states.len();
        let actions_slice = actions.as_slice()?;
        if actions_slice.len() != n {
            return Err(PyValueError::new_err(format!(
                "actions length {} != num_envs {}",
                actions_slice.len(),
                n
            )));
        }
        // Pre-validate every non-terminal env -- in parallel, reporting the
        // LOWEST offending env index exactly as a serial loop would, and
        // before any state is touched (all-or-nothing). PERF-032: this was a
        // serial loop, then a serial apply.
        let states_ref = &self.states;
        let bad = py.detach(|| {
            states_ref
                .par_iter()
                .enumerate()
                .with_min_len(APPLY_MIN_LEN)
                .find_map_first(|(i, st)| {
                    validate_discrete_action(i, st.as_ref(), actions_slice[i])
                })
        });
        if let Some((not_reset, msg)) = bad {
            return Err(if not_reset {
                PyRuntimeError::new_err(msg)
            } else {
                PyValueError::new_err(msg)
            });
        }
        let term_vec: Vec<bool> = py.detach(|| {
            self.states
                .par_iter_mut()
                .enumerate()
                .with_min_len(APPLY_MIN_LEN)
                .map(|(i, state_opt)| {
                    let state = state_opt.as_mut().expect("validated above");
                    if state.is_terminal() {
                        return false;
                    }
                    state.apply(Action::from_index(actions_slice[i]).expect("validated above"));
                    state.is_terminal()
                })
                .collect()
        });
        Ok(Array1::from_vec(term_vec).into_pyarray(py))
    }

    /// Hybrid dispatch per env: `gates[i]` in `{0=Fold, 1=CheckCall,
    /// 2=Raise, 3=AllIn}`. When `gates[i] == 2`, `raise_chips[i]` is the
    /// chip delta (must be in `[min_raise_chips, max_raise_chips]`);
    /// otherwise `raise_chips[i]` is ignored. Terminal envs are skipped.
    /// Returns a `(N,)` bool array where `true` means the env transitioned
    /// to terminal on this call.
    fn apply_hybrid_batch<'py>(
        &mut self,
        py: Python<'py>,
        gates: PyReadonlyArray1<'_, u8>,
        raise_chips: PyReadonlyArray1<'_, u64>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let n = self.states.len();
        let gates_slice = gates.as_slice()?;
        let chips_slice = raise_chips.as_slice()?;
        if gates_slice.len() != n || chips_slice.len() != n {
            return Err(PyValueError::new_err(format!(
                "gates/raise_chips length must equal num_envs={}",
                n
            )));
        }
        // Pre-validate every non-terminal env -- in parallel, reporting the
        // LOWEST offending env index exactly as the old serial loop did, and
        // before any state is touched (all-or-nothing).
        let states_ref = &self.states;
        let bad = py.detach(|| {
            states_ref
                .par_iter()
                .enumerate()
                .with_min_len(APPLY_MIN_LEN)
                .find_map_first(|(i, st)| {
                    validate_hybrid_action(i, st.as_ref(), gates_slice[i], chips_slice[i])
                })
        });
        if let Some((not_reset, msg)) = bad {
            return Err(if not_reset {
                PyRuntimeError::new_err(msg)
            } else {
                PyValueError::new_err(msg)
            });
        }
        // Per-env work here is sub-microsecond: coarse leaves (few rayon
        // wake-ups) beat splitting 7k tables across every worker.
        let term_vec: Vec<bool> = py.detach(|| {
            self.states
                .par_iter_mut()
                .enumerate()
                .with_min_len(APPLY_MIN_LEN)
                .map(|(i, state_opt)| {
                    let state = state_opt.as_mut().expect("validated above");
                    if state.is_terminal() {
                        return false;
                    }
                    match gates_slice[i] {
                        0 => state.apply(Action::Fold),
                        1 => state.apply(Action::CheckCall),
                        2 => state
                            .apply_raise_chips(chips_slice[i])
                            .expect("validated above"),
                        3 => state.apply(Action::AllIn),
                        _ => unreachable!(),
                    }
                    state.is_terminal()
                })
                .collect()
        });
        let terminal = Array1::from_vec(term_vec);
        Ok(terminal.into_pyarray(py))
    }

    /// `(N, NUM_ACTIONS)` bool legal-action mask. Terminal envs return
    /// an all-false row.
    fn legal_mask_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let n = self.states.len();
        let arr: Array2<bool> = py.detach(|| {
            let mut arr = Array2::<bool>::default((n, NUM_ACTIONS));
            for i in 0..n {
                if let Some(state) = self.states[i].as_ref() {
                    if !state.is_terminal() {
                        let mask = state.legal_action_mask();
                        for j in 0..NUM_ACTIONS {
                            arr[[i, j]] = mask[j];
                        }
                    }
                }
            }
            arr
        });
        Ok(arr.into_pyarray(py))
    }

    /// `(N,)` i8: current actor per env, or -1 if terminal / unset.
    fn actor_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i8>>> {
        let n = self.states.len();
        let arr: Array1<i8> = py.detach(|| {
            let mut arr = Array1::<i8>::from_elem(n, -1i8);
            for i in 0..n {
                if let Some(state) = self.states[i].as_ref() {
                    if let Some(a) = state.current_actor() {
                        arr[i] = a as i8;
                    }
                }
            }
            arr
        });
        Ok(arr.into_pyarray(py))
    }

    /// `(N,)` bool: true where env is terminal (or not yet reset). Test helper:
    /// the rollout reads terminality from `actor_batch` / the apply calls.
    fn is_terminal_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let n = self.states.len();
        let arr: Array1<bool> = py.detach(|| {
            let mut arr = Array1::<bool>::default(n);
            for i in 0..n {
                arr[i] = match self.states[i].as_ref() {
                    Some(s) => s.is_terminal(),
                    None => true,
                };
            }
            arr
        });
        Ok(arr.into_pyarray(py))
    }

    /// `(N, num_seats)` i64 chip-delta payouts. Zeros for non-terminal envs.
    fn payouts_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let n = self.states.len();
        let s = self.config.num_seats;
        let states_ref = &self.states;
        let arr: Array2<i64> = py.detach(|| {
            let rows: Vec<Vec<i64>> = states_ref
                .par_iter()
                .map(|state_opt| match state_opt.as_ref() {
                    Some(state) if state.is_terminal() => state.payouts(),
                    _ => vec![0i64; s],
                })
                .collect();
            let mut arr = Array2::<i64>::zeros((n, s));
            for (i, row) in rows.iter().enumerate() {
                for k in 0..s {
                    arr[[i, k]] = row[k];
                }
            }
            arr
        });
        Ok(arr.into_pyarray(py))
    }

    /// `(N, num_seats)` i64 EV payouts. Non-terminal envs get zeros.
    /// Each env uses `seeds[i]` for its MC sampler (pinned `ChaCha8Rng`),
    /// matching `PyGameState::payouts_ev` exactly.
    fn payouts_ev_batch<'py>(
        &self,
        py: Python<'py>,
        num_samples: u32,
        seeds: PyReadonlyArray1<'_, u64>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let n = self.states.len();
        let s = self.config.num_seats;
        let seeds_slice = seeds.as_slice()?;
        if seeds_slice.len() != n {
            return Err(PyValueError::new_err(format!(
                "seeds length {} != num_envs {}",
                seeds_slice.len(),
                n
            )));
        }
        let seeds_vec: Vec<u64> = seeds_slice.to_vec();
        let idx: Vec<usize> = (0..n).collect();
        let mut arr = Array2::<i64>::zeros((n, s));
        {
            let out = arr
                .as_slice_mut()
                .expect("freshly allocated Array2 is contiguous");
            py.detach(|| self.payouts_ev_rows(num_samples, &idx, &seeds_vec, out));
        }
        Ok(arr.into_pyarray(py))
    }

    /// `(k, num_seats)` i64 EV payouts for the envs in `indices` ONLY: row j
    /// is env `indices[j]` sampled with `seeds[j]` (zeros if that env is not
    /// terminal) -- identical to `payouts_ev_batch(num_samples, batch_seeds)
    /// [indices]` whenever `seeds[j] == batch_seeds[indices[j]]`. The rollout
    /// reads only the newly-terminal rows, and in the drain phase every env
    /// that finished on an EARLIER step is still terminal: the whole-batch
    /// call re-ran all of their runouts on every drain step.
    fn payouts_ev_subset<'py>(
        &self,
        py: Python<'py>,
        num_samples: u32,
        seeds: PyReadonlyArray1<'_, u64>,
        indices: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let n = self.states.len();
        let s = self.config.num_seats;
        let idx = checked_env_indices(indices.as_slice()?, n, "payouts_ev_subset")?;
        let seeds_vec: Vec<u64> = seeds.as_slice()?.to_vec();
        if seeds_vec.len() != idx.len() {
            return Err(PyValueError::new_err(format!(
                "payouts_ev_subset: {} seeds for {} indices",
                seeds_vec.len(),
                idx.len()
            )));
        }
        let mut arr = Array2::<i64>::zeros((idx.len(), s));
        {
            let out = arr
                .as_slice_mut()
                .expect("freshly allocated Array2 is contiguous");
            py.detach(|| self.payouts_ev_rows(num_samples, &idx, &seeds_vec, out));
        }
        Ok(arr.into_pyarray(py))
    }

    /// `(N,)` u8 hand category per env for the given `(seat, board)`.
    fn hero_category_batch<'py>(
        &self,
        py: Python<'py>,
        seats: PyReadonlyArray1<'_, u8>,
        boards: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let n = self.states.len();
        let seats_slice = seats.as_slice()?;
        let boards_slice = boards.as_slice()?;
        if seats_slice.len() != n || boards_slice.len() != n {
            return Err(PyValueError::new_err(format!(
                "seats/boards length must equal num_envs={}",
                n
            )));
        }
        let num_seats = self.config.num_seats;
        for i in 0..n {
            if (seats_slice[i] as usize) >= num_seats {
                return Err(PyValueError::new_err(format!(
                    "seat {} out of range at env {}",
                    seats_slice[i], i
                )));
            }
            if boards_slice[i] > 1 {
                return Err(PyValueError::new_err(format!(
                    "board {} must be 0 or 1 at env {}",
                    boards_slice[i], i
                )));
            }
        }
        let seats_vec: Vec<u8> = seats_slice.to_vec();
        let boards_vec: Vec<u8> = boards_slice.to_vec();
        let arr: Array1<u8> = py.detach(move || {
            let mut arr = Array1::<u8>::zeros(n);
            for i in 0..n {
                if let Some(state) = self.states[i].as_ref() {
                    arr[i] = state.hero_category(seats_vec[i] as usize, boards_vec[i]);
                }
            }
            arr
        });
        Ok(arr.into_pyarray(py))
    }

    /// Stacked view of every field the vectorized (numpy) encoder needs, one
    /// row per env (the keys of [`PackedObservation::into_dict`]):
    ///
    /// - `hero_hole`         (N, hole_count) u8 — the actor's hole; 255 when terminal.
    /// - `board_a` / `board_b` (N, 5) u8   — 255-padded past the visible cards.
    /// - `board_a_len` / `board_b_len` (N,) u8
    /// - `street` (N,) u8; `pot` (N,) u64; `bet_to_call` (N,) u64
    /// - `stacks` / `street_commit` / `total_commit` / `eff_stack_cap` (N, S) u64
    /// - `folded` / `all_in` / `acted_this_street` (N, S) bool
    /// - `min_bet` / `max_bet` (N,) u64 — `min/max_bet_total()` (rev-1 scalars).
    /// - `min_raise` / `max_raise` (N,) u64 — legal raise DELTAS (0 if illegal).
    /// - `actor` (N,) i8 (-1 when terminal); `button` (N,) u8;
    ///   `last_aggressor` (N,) i8 (-1 before any raise this street).
    /// - `history_seat` / `history_action` / `history_street` (N, H) i8 and
    ///   `history_chips` (N, H) u64 — the newest H records oldest-first, -1 / 0
    ///   in empty slots; `history_len` (N,) u8. H = 32 (PLO) / 40 (NLH).
    /// - `opp_outcome_fractions` (N, 12), `per_board_outcome` (N, 8),
    ///   `share_bounds` (N, 2) f32 — the PLO MC block (zero pre-flop / terminal).
    /// - `hero_board_v3` (N, 8), `board_draw_v3` (N, 7) u8 — the v7 scans.
    /// - `sb_seat` / `bb_seat` (N,) i8 — blind seats, -1 when the variant has none.
    /// - `nlh_opp_outcome` (N, 3) f32 — NLH [opp_ahead, tied, opp_behind];
    ///   all-zero for PLO variants.
    fn observation_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let idx: Vec<usize> = (0..self.states.len()).collect();
        let packed = py.detach(|| self.pack_full(&idx));
        packed.obs.into_dict(py)
    }

    /// `observation_arrays()` + `legal_mask_batch()` + the actor's
    /// `hero_category_batch(actor, 0|1)` in ONE FFI crossing (the numpy
    /// encoder path). Extra keys: `legal_mask` (N, 8) bool, `hero_cat_a` /
    /// `hero_cat_b` (N,) u8. Terminal envs get all-False masks and zero
    /// categories — the vectorized encoder ignores those rows anyway.
    fn observation_and_features_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let idx: Vec<usize> = (0..self.states.len()).collect();
        self.features_dict(py, &idx)
    }

    /// `observation_and_features_batch` for ONLY the envs in `indices`
    /// (compact k-row arrays, same keys and dtypes): the rollout refreshes just
    /// the re-dealt envs with it.
    fn observation_and_features_subset_batch<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let idx = checked_env_indices(
            indices.as_slice()?,
            self.states.len(),
            "observation_and_features_subset_batch",
        )?;
        self.features_dict(py, &idx)
    }

    /// The one observation entry point of the Rust encoders (ENG-012):
    ///
    /// ```text
    /// encode(layout="full"|"minimal", indices=None, out=None, encode_mask=None,
    ///        flag_cols=None, real_cols=None, out_bits=None, out_real=None) -> dict
    /// ```
    ///
    /// - `indices` (int64): encode only these envs (row j = env `indices[j]`);
    ///   default every env.
    /// - Without `out` and packed outputs, the dict carries a fresh `(k, dim)`
    ///   float32 `"obs"`.
    /// - `out`: a C-contiguous `(num_envs, dim)` float32 buffer (the env's own
    ///   obs cache); row j is written to `out[indices[j]]` in place and every
    ///   other row is left untouched. `indices` must then be strictly
    ///   increasing (e.g. from `np.nonzero`).
    /// - `encode_mask` (bool, length num_envs): envs whose entry is False get an
    ///   all-zero row.
    /// - `flag_cols` / `real_cols` / `out_bits` / `out_real` (all four; the
    ///   compact layout of python/plo5bp/compact_obs.py): every row is also
    ///   packed right after it is encoded — the bytes `pack_obs_rows` would
    ///   write. With `out=None` the call is PACKED-ONLY (no dense rows at all).
    ///
    /// Always returned: `actor`, `legal_mask`, `min_raise`, `max_raise`,
    /// `total_commit`, `bet_to_call`, `street_commit`, `street`, `pot` for the
    /// k encoded envs. PLO only (NLH encodes through numpy). Bit-exact with the
    /// numpy encoders (`encode_observation_batch[_minimal]`) and with every
    /// `observation_encoded_*` method below, which are thin aliases of it.
    #[pyo3(signature = (layout="full", indices=None, out=None, encode_mask=None, flag_cols=None, real_cols=None, out_bits=None, out_real=None))]
    #[allow(clippy::too_many_arguments)]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        layout: &str,
        indices: Option<PyReadonlyArray1<'_, i64>>,
        out: Option<PyReadwriteArray2<'_, f32>>,
        encode_mask: Option<PyReadonlyArray1<'_, bool>>,
        flag_cols: Option<PyReadonlyArray1<'_, i64>>,
        real_cols: Option<PyReadonlyArray1<'_, i64>>,
        out_bits: Option<PyReadwriteArray2<'_, u8>>,
        out_real: Option<PyReadwriteArray2<'_, f32>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let layout = Layout::parse(layout)?;
        let in_place = out.is_some()
            || out_bits.is_some()
            || out_real.is_some()
            || flag_cols.is_some()
            || real_cols.is_some();
        let mode = if in_place {
            EncodeMode::InPlace
        } else {
            EncodeMode::Return
        };
        self.encode_impl(
            py,
            "encode",
            layout,
            mode,
            indices,
            out,
            encode_mask,
            flag_cols,
            real_cols,
            out_bits,
            out_real,
        )
    }

    /// Alias: `encode("full")`.
    fn observation_encoded_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let what = "observation_encoded_batch";
        self.encode_impl(
            py,
            what,
            Layout::Full,
            EncodeMode::Return,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Alias: `encode("full", indices)` (compact k rows; any index order).
    fn observation_encoded_subset_batch<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let what = "observation_encoded_subset_batch";
        self.encode_impl(
            py,
            what,
            Layout::Full,
            EncodeMode::Return,
            Some(indices),
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Alias: `encode("full", out=out, encode_mask=..., <packed outputs>)` —
    /// the in-place full encoder (2026-09-26); `out=None` needs the packed
    /// outputs. Returns the aux dict without "obs".
    #[pyo3(signature = (out, encode_mask=None, flag_cols=None, real_cols=None, out_bits=None, out_real=None))]
    #[allow(clippy::too_many_arguments)]
    fn observation_encoded_into<'py>(
        &self,
        py: Python<'py>,
        out: Option<PyReadwriteArray2<'_, f32>>,
        encode_mask: Option<PyReadonlyArray1<'_, bool>>,
        flag_cols: Option<PyReadonlyArray1<'_, i64>>,
        real_cols: Option<PyReadonlyArray1<'_, i64>>,
        out_bits: Option<PyReadwriteArray2<'_, u8>>,
        out_real: Option<PyReadwriteArray2<'_, f32>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let what = "observation_encoded_into";
        self.encode_impl(
            py,
            what,
            Layout::Full,
            EncodeMode::InPlace,
            None,
            out,
            encode_mask,
            flag_cols,
            real_cols,
            out_bits,
            out_real,
        )
    }

    /// Alias: `encode("full", indices, out=out, <packed outputs>)` — the
    /// in-place full SUBSET encoder (indices strictly increasing).
    #[pyo3(signature = (indices, out=None, flag_cols=None, real_cols=None, out_bits=None, out_real=None))]
    #[allow(clippy::too_many_arguments)]
    fn observation_encoded_subset_into<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'_, i64>,
        out: Option<PyReadwriteArray2<'_, f32>>,
        flag_cols: Option<PyReadonlyArray1<'_, i64>>,
        real_cols: Option<PyReadonlyArray1<'_, i64>>,
        out_bits: Option<PyReadwriteArray2<'_, u8>>,
        out_real: Option<PyReadwriteArray2<'_, f32>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let what = "observation_encoded_subset_into";
        self.encode_impl(
            py,
            what,
            Layout::Full,
            EncodeMode::InPlace,
            Some(indices),
            out,
            None,
            flag_cols,
            real_cols,
            out_bits,
            out_real,
        )
    }

    /// Alias: `encode("minimal")` — the bare-visibility (796) layout: no MC,
    /// no hero categories, no v7 board scans. Bit-exact with
    /// `encode_observation_batch_minimal` (python/plo5bp/encoding.py).
    fn observation_encoded_minimal_batch<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let what = "observation_encoded_minimal_batch";
        self.encode_impl(
            py,
            what,
            Layout::Minimal,
            EncodeMode::Return,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Alias: `encode("minimal", indices)` (compact k rows).
    fn observation_encoded_minimal_subset_batch<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let what = "observation_encoded_minimal_subset_batch";
        self.encode_impl(
            py,
            what,
            Layout::Minimal,
            EncodeMode::Return,
            Some(indices),
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Alias: `encode("minimal", out=out, encode_mask=..., <packed outputs>)`
    /// — the in-place minimal encoder (2026-09-23; packed outputs 2026-09-24).
    #[pyo3(signature = (out, encode_mask=None, flag_cols=None, real_cols=None, out_bits=None, out_real=None))]
    #[allow(clippy::too_many_arguments)]
    fn observation_encoded_minimal_into<'py>(
        &self,
        py: Python<'py>,
        out: Option<PyReadwriteArray2<'_, f32>>,
        encode_mask: Option<PyReadonlyArray1<'_, bool>>,
        flag_cols: Option<PyReadonlyArray1<'_, i64>>,
        real_cols: Option<PyReadonlyArray1<'_, i64>>,
        out_bits: Option<PyReadwriteArray2<'_, u8>>,
        out_real: Option<PyReadwriteArray2<'_, f32>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let what = "observation_encoded_minimal_into";
        self.encode_impl(
            py,
            what,
            Layout::Minimal,
            EncodeMode::InPlace,
            None,
            out,
            encode_mask,
            flag_cols,
            real_cols,
            out_bits,
            out_real,
        )
    }

    /// Alias: `encode("minimal", indices, out=out, <packed outputs>)` (indices
    /// strictly increasing; other rows of `out` untouched).
    #[pyo3(signature = (indices, out, flag_cols=None, real_cols=None, out_bits=None, out_real=None))]
    #[allow(clippy::too_many_arguments)]
    fn observation_encoded_minimal_subset_into<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'_, i64>,
        out: Option<PyReadwriteArray2<'_, f32>>,
        flag_cols: Option<PyReadonlyArray1<'_, i64>>,
        real_cols: Option<PyReadonlyArray1<'_, i64>>,
        out_bits: Option<PyReadwriteArray2<'_, u8>>,
        out_real: Option<PyReadwriteArray2<'_, f32>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let what = "observation_encoded_minimal_subset_into";
        self.encode_impl(
            py,
            what,
            Layout::Minimal,
            EncodeMode::InPlace,
            Some(indices),
            out,
            None,
            flag_cols,
            real_cols,
            out_bits,
            out_real,
        )
    }
}

impl PyBatchedEngine {
    /// The dict of `observation_and_features_[subset_]batch` for envs `idx`:
    /// every packed field + `legal_mask` + `hero_cat_a` / `hero_cat_b`, from
    /// ONE parallel pack (PERF-032: this used to add a serial legality and
    /// category loop on the numpy encoder's path).
    fn features_dict<'py>(&self, py: Python<'py>, idx: &[usize]) -> PyResult<Bound<'py, PyDict>> {
        let packed = py.detach(|| self.pack_full(idx));
        let d = packed.obs.into_dict(py)?;
        d.set_item("legal_mask", packed.legal.into_pyarray(py))?;
        d.set_item(
            "hero_cat_a",
            Array1::from_vec(packed.cat_a).into_pyarray(py),
        )?;
        d.set_item(
            "hero_cat_b",
            Array1::from_vec(packed.cat_b).into_pyarray(py),
        )?;
        Ok(d)
    }

    /// `num_envs` not-yet-dealt envs over an already-validated `config` — the
    /// engine `new` builds after its Python-facing checks, and what the Rust
    /// tests build engines with.
    pub(super) fn with_config(
        config: GameConfig,
        num_envs: usize,
        opp_outcome_mc: usize,
        obs_rev: u8,
    ) -> Self {
        let num_seats = config.num_seats;
        PyBatchedEngine {
            states: (0..num_envs).map(|_| None).collect(),
            config,
            opp_outcome_mc,
            outcome_cache: std::sync::Mutex::new(OutcomeCache::new(num_envs, num_seats)),
            board_tables: (0..num_envs).map(|_| std::sync::Mutex::new(None)).collect(),
            obs_rev,
        }
    }

    /// Row j of `out` (num_seats wide) = EV payouts of env `idx[j]` sampled
    /// with `seeds[j]`; rows of non-terminal envs are left as they are (the
    /// callers pass zeros). Only terminal hands are dispatched, ONE hand per
    /// rayon task: the hands that need a runout (all-in before the river,
    /// `num_samples` full evaluations each) are few, expensive and scattered
    /// through the batch, so coarse chunks of the whole batch left most
    /// workers idle behind whichever chunk held several of them.
    pub(super) fn payouts_ev_rows(
        &self,
        num_samples: u32,
        idx: &[usize],
        seeds: &[u64],
        out: &mut [i64],
    ) {
        let s = self.config.num_seats;
        let states = &self.states;
        let term: Vec<(usize, usize)> = idx
            .iter()
            .enumerate()
            .filter(|&(_, &i)| states[i].as_ref().is_some_and(|st| st.is_terminal()))
            .map(|(j, &i)| (j, i))
            .collect();
        let rows: Vec<Vec<i64>> = term
            .par_iter()
            .with_max_len(1)
            .map(|&(j, i)| {
                states[i]
                    .as_ref()
                    .expect("filtered to Some above")
                    .payouts_ev(num_samples, seeds[j])
            })
            .collect();
        for (&(j, _), row) in term.iter().zip(rows) {
            out[j * s..(j + 1) * s].copy_from_slice(&row);
        }
    }
}
