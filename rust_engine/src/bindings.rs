//! PyO3 wrapper around [`GameState`]. Exposes a minimal surface for the
//! Python environment and training loop.

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::actions::{Action, NUM_ACTIONS};
use crate::cards::Card;
use crate::state::{GameConfig, GameState, StudyTerminal};

/// Python-facing `GameState`. Construct with config, then `reset(seed, button)`
/// to deal a hand. Subsequent calls drive the state machine.
#[pyclass(name = "GameState")]
pub struct PyGameState {
    inner: Option<GameState>,
    config: GameConfig,
}

#[pymethods]
impl PyGameState {
    #[new]
    #[pyo3(signature = (num_seats=6, starting_stack=200000, ante=30000, bb=10000, starting_stacks=None))]
    fn new(
        num_seats: usize,
        starting_stack: u64,
        ante: u64,
        bb: u64,
        starting_stacks: Option<PyReadonlyArray1<'_, u64>>,
    ) -> PyResult<Self> {
        let stacks = resolve_starting_stacks(num_seats, starting_stack, starting_stacks)?;
        Ok(PyGameState {
            inner: None,
            config: GameConfig {
                num_seats,
                starting_stacks: stacks,
                ante,
                bb,
            },
        })
    }

    #[pyo3(signature = (seed, button, in_hand_mask=None))]
    fn reset(
        &mut self,
        seed: u64,
        button: usize,
        in_hand_mask: Option<Vec<bool>>,
    ) -> PyResult<()> {
        if button >= self.config.num_seats {
            return Err(PyValueError::new_err("button out of range"));
        }
        if let Some(ref m) = in_hand_mask {
            if m.len() != self.config.num_seats {
                return Err(PyValueError::new_err(
                    "in_hand_mask length must equal num_seats",
                ));
            }
            if m.iter().filter(|&&b| b).count() < 2 {
                return Err(PyValueError::new_err(
                    "in_hand_mask must include at least 2 seats",
                ));
            }
        }
        self.inner = Some(GameState::new_hand_with_mask(
            self.config.clone(),
            seed,
            button,
            in_hand_mask,
        ));
        Ok(())
    }

    /// Deal a study-mode hand at the flop with user-supplied cards.
    /// Card inputs are raw indices in `0..=51`.
    /// `in_hand_mask` (optional, length `num_seats`) restricts the hand to
    /// the seats marked `true`; sitting-out seats post no ante and are
    /// pre-folded. Hero seat must be in-hand.
    #[pyo3(signature = (button, hero_seat, hero_hole, flop_a, flop_b, in_hand_mask=None))]
    fn reset_study(
        &mut self,
        button: usize,
        hero_seat: usize,
        hero_hole: Vec<u8>,
        flop_a: Vec<u8>,
        flop_b: Vec<u8>,
        in_hand_mask: Option<Vec<bool>>,
    ) -> PyResult<()> {
        let hero_hole = cards_from_indices::<5>(&hero_hole, "hero_hole")?;
        let flop_a = cards_from_indices::<3>(&flop_a, "flop_a")?;
        let flop_b = cards_from_indices::<3>(&flop_b, "flop_b")?;
        match GameState::new_study_with_mask(
            self.config.clone(),
            button,
            hero_seat,
            hero_hole,
            flop_a,
            flop_b,
            in_hand_mask,
        ) {
            Ok(g) => {
                self.inner = Some(g);
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }

    fn set_turn(&mut self, card_a: u8, card_b: u8) -> PyResult<()> {
        let g = self.get_mut()?;
        g.set_turn(card_from_index(card_a)?, card_from_index(card_b)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn set_river(&mut self, card_a: u8, card_b: u8) -> PyResult<()> {
        let g = self.get_mut()?;
        g.set_river(card_from_index(card_a)?, card_from_index(card_b)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `None` while no UI input is expected, else `2` (Turn) or `3` (River)
    /// matching [`Street::index`] for the street whose cards are awaited.
    fn awaiting_next_street(&self) -> PyResult<Option<u8>> {
        let g = self.get()?;
        Ok(g.awaiting_next_street.map(|s| s.index() as u8))
    }

    /// `None` while the hand is live, else `0=FoldOut`, `1=RunOut`, `2=Showdown`.
    fn study_terminal(&self) -> PyResult<Option<u8>> {
        let g = self.get()?;
        Ok(g.study_terminal.map(|t| match t {
            StudyTerminal::FoldOut => 0u8,
            StudyTerminal::RunOut => 1u8,
            StudyTerminal::Showdown => 2u8,
        }))
    }

    fn legal_action_mask(&self) -> PyResult<Vec<bool>> {
        let g = self.get()?;
        Ok(g.legal_action_mask().to_vec())
    }

    fn apply_action(&mut self, action_idx: u8) -> PyResult<()> {
        let action = Action::from_index(action_idx)
            .ok_or_else(|| PyValueError::new_err(format!("invalid action index {action_idx}")))?;
        let g = self.get_mut()?;
        let mask = g.legal_action_mask();
        if !mask[action.index() as usize] {
            return Err(PyValueError::new_err(format!(
                "action {action_idx} illegal in current state"
            )));
        }
        g.apply(action);
        Ok(())
    }

    /// Continuous-sizing raise. `chips` is the chip delta the current
    /// actor adds to the pot (not a target total). Must be in
    /// `[min_raise_chips(), max_raise_chips()]`.
    fn apply_raise_chips(&mut self, chips: u64) -> PyResult<()> {
        let g = self.get_mut()?;
        g.apply_raise_chips(chips)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn is_terminal(&self) -> PyResult<bool> {
        Ok(self.get()?.is_terminal())
    }

    fn current_actor(&self) -> PyResult<Option<usize>> {
        Ok(self.get()?.current_actor())
    }

    fn payouts(&self) -> PyResult<Vec<i64>> {
        Ok(self.get()?.payouts())
    }

    /// Expected chip delta per seat, averaged over `num_samples`
    /// Monte-Carlo runouts of the community cards undealt at the street
    /// where action closed. Delegates to `payouts` when sampling is a
    /// no-op (fold-out, river-close, or `num_samples == 0`).
    fn payouts_ev(&self, num_samples: u32, seed: u64) -> PyResult<Vec<i64>> {
        Ok(self.get()?.payouts_ev(num_samples, seed))
    }

    fn num_seats(&self) -> usize {
        self.config.num_seats
    }

    fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }

    fn min_bet_total(&self) -> PyResult<u64> {
        Ok(self.get()?.min_bet_total())
    }

    fn max_bet_total(&self) -> PyResult<u64> {
        Ok(self.get()?.max_bet_total())
    }

    /// Smallest legal chip delta for a raise/bet. Zero when Raise isn't
    /// available (not facing bet + can't open, or stack too small).
    fn min_raise_chips(&self) -> PyResult<u64> {
        Ok(self.get()?.min_raise_chips())
    }

    /// Largest legal chip delta for a raise/bet (PL-capped, stack-capped).
    fn max_raise_chips(&self) -> PyResult<u64> {
        Ok(self.get()?.max_raise_chips())
    }

    /// Hand category index (0..=8) of seat's best PLO5 hand on `board`.
    fn hero_category(&self, seat: usize, board: u8) -> PyResult<u8> {
        let g = self.get()?;
        if seat >= self.config.num_seats {
            return Err(PyValueError::new_err("seat out of range"));
        }
        Ok(g.hero_category(seat, board))
    }

    /// 12 fractions in `[0, 1]`, row-major `[k=2,3,4][outcome]`,
    /// outcome enum: 0=scoop_opp, 1=quarter_opp, 2=scoop_hero,
    /// 3=quarter_hero. See `GameState::opp_outcome_fractions`.
    fn opp_outcome_fractions(&self) -> PyResult<Vec<f32>> {
        Ok(self.get()?.opp_outcome_fractions())
    }

    /// Dict-shaped observation. See module doc for keys.
    fn observation_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let g = self.get()?;
        let d = PyDict::new(py);

        let hero_hole: Vec<u8> = match g.current_actor() {
            Some(a) => g.hole_cards[a].iter().map(|c| c.index()).collect(),
            None => Vec::new(),
        };
        d.set_item("hero_hole", hero_hole)?;

        let board_a: Vec<u8> = g.board_a.iter().map(|c| c.index()).collect();
        let board_b: Vec<u8> = g.board_b.iter().map(|c| c.index()).collect();
        d.set_item("board_a", board_a)?;
        d.set_item("board_b", board_b)?;

        d.set_item("street", g.street.index() as i64)?;
        d.set_item("pot", g.pot)?;
        d.set_item("stacks", g.stacks.clone())?;
        d.set_item("folded", g.folded.clone())?;
        d.set_item("all_in", g.all_in.clone())?;
        d.set_item("bet_to_call", g.bet_to_call)?;
        d.set_item("street_commit", g.street_commit.clone())?;
        d.set_item("total_commit", g.total_commit.clone())?;
        d.set_item("min_bet", g.min_bet_total())?;
        d.set_item("max_bet", g.max_bet_total())?;
        d.set_item("min_raise", g.min_raise_chips())?;
        d.set_item("max_raise", g.max_raise_chips())?;
        d.set_item("eff_stack_cap", g.eff_stack_cap_at_hand_start.clone())?;
        d.set_item("actor", g.actor)?;
        d.set_item("button", g.button)?;
        d.set_item(
            "last_aggressor",
            g.last_aggressor.map(|s| s as i64).unwrap_or(-1),
        )?;

        let history: Vec<(usize, u8, u64, u8)> = g
            .history
            .iter()
            .map(|r| (r.seat, r.action.index(), r.chips, r.street.index() as u8))
            .collect();
        d.set_item("history", history)?;

        d.set_item("opp_outcome_fractions", g.opp_outcome_fractions())?;

        let awaiting = g.awaiting_next_street.map(|s| s.index() as u8);
        d.set_item("awaiting_next_street", awaiting)?;
        let terminal = g.study_terminal.map(|t| match t {
            StudyTerminal::FoldOut => 0u8,
            StudyTerminal::RunOut => 1u8,
            StudyTerminal::Showdown => 2u8,
        });
        d.set_item("study_terminal", terminal)?;

        Ok(d)
    }
}

/// Diagnostic hook: compute layered side-pot payouts for an arbitrary
/// commit / hole-card / board configuration. Wraps the pure
/// [`crate::double_board::double_board_payout`] function so Python can
/// verify side-pot handling without driving a real GameState.
///
/// `hole_cards` must be `5 * num_seats` card indices, flat-packed by seat
/// (seat 0's 5 cards, then seat 1's 5 cards, ...). `board_a` / `board_b`
/// are 5 indices each. Returns chips *won* per seat (sum equals
/// `total_commit.sum()`).
#[pyfunction]
pub fn compute_double_board_payout(
    hole_cards: Vec<u8>,
    folded: Vec<bool>,
    total_commit: Vec<u64>,
    board_a: Vec<u8>,
    board_b: Vec<u8>,
    button: usize,
) -> PyResult<Vec<u64>> {
    let n = folded.len();
    if total_commit.len() != n {
        return Err(PyValueError::new_err(
            "total_commit length must equal num_seats",
        ));
    }
    if hole_cards.len() != 5 * n {
        return Err(PyValueError::new_err(
            "hole_cards must have 5 * num_seats indices",
        ));
    }
    if button >= n {
        return Err(PyValueError::new_err("button out of range"));
    }
    let mut holes: Vec<[Card; 5]> = Vec::with_capacity(n);
    for s in 0..n {
        let slice = &hole_cards[5 * s..5 * (s + 1)];
        holes.push(cards_from_indices::<5>(slice, "hole_cards")?);
    }
    let board_a = cards_from_indices::<5>(&board_a, "board_a")?;
    let board_b = cards_from_indices::<5>(&board_b, "board_b")?;
    Ok(crate::double_board::double_board_payout(
        &holes,
        &folded,
        &total_commit,
        &board_a,
        &board_b,
        button,
    ))
}

fn resolve_starting_stacks(
    num_seats: usize,
    starting_stack: u64,
    starting_stacks: Option<PyReadonlyArray1<'_, u64>>,
) -> PyResult<Vec<u64>> {
    match starting_stacks {
        Some(arr) => {
            let slice = arr.as_slice()?;
            if slice.len() != num_seats {
                return Err(PyValueError::new_err(format!(
                    "starting_stacks length {} != num_seats {}",
                    slice.len(),
                    num_seats
                )));
            }
            Ok(slice.to_vec())
        }
        None => Ok(vec![starting_stack; num_seats]),
    }
}

fn card_from_index(i: u8) -> PyResult<Card> {
    if i >= 52 {
        return Err(PyValueError::new_err(format!("card index {i} out of range")));
    }
    Ok(Card::from_index(i))
}

fn cards_from_indices<const N: usize>(indices: &[u8], label: &str) -> PyResult<[Card; N]> {
    if indices.len() != N {
        return Err(PyValueError::new_err(format!(
            "{label} must have {N} indices, got {}",
            indices.len()
        )));
    }
    let mut out = [Card(0); N];
    for (i, &idx) in indices.iter().enumerate() {
        out[i] = card_from_index(idx)?;
    }
    Ok(out)
}

impl PyGameState {
    fn get(&self) -> PyResult<&GameState> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("GameState not reset; call reset() first"))
    }

    fn get_mut(&mut self) -> PyResult<&mut GameState> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("GameState not reset; call reset() first"))
    }
}

// ============================================================================
// Batched engine for fast training rollouts.
//
// Holds `num_envs` independent GameStates sharing a GameConfig. All public
// methods take / return stacked NumPy arrays so the Python side can drive N
// envs through a single FFI call. The inner loops release the GIL via
// `py.allow_threads`, so a caller can overlap other Python work with Rust
// engine computation on a worker thread.
//
// Per-env determinism matches the serial engine bit-for-bit: each call takes
// seeds/buttons/actions as arrays, and the engine uses the same
// `ChaCha8Rng::seed_from_u64` derivations (for deck shuffling, equity MC, and
// EV runout sampling) as `PyGameState`. This is what lets Phase A be an
// optimization, not a behavior change.
// ============================================================================

const HISTORY_CAP: usize = 32;

/// Python-facing batched game engine. Construct with `(num_envs, config)`,
/// then `reset_batch(seeds, buttons)` to seed all envs. `apply_action_batch`
/// steps every env; terminal envs stay terminal until `reset_terminal_batch`.
#[pyclass(name = "BatchedEngine")]
pub struct PyBatchedEngine {
    states: Vec<Option<GameState>>,
    config: GameConfig,
}

#[pymethods]
impl PyBatchedEngine {
    #[new]
    #[pyo3(signature = (num_envs, num_seats=6, starting_stack=200000, ante=30000, bb=10000, starting_stacks=None))]
    fn new(
        num_envs: usize,
        num_seats: usize,
        starting_stack: u64,
        ante: u64,
        bb: u64,
        starting_stacks: Option<PyReadonlyArray1<'_, u64>>,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be >= 1"));
        }
        if num_seats < 2 {
            return Err(PyValueError::new_err("num_seats must be >= 2"));
        }
        let stacks = resolve_starting_stacks(num_seats, starting_stack, starting_stacks)?;
        Ok(PyBatchedEngine {
            states: (0..num_envs).map(|_| None).collect(),
            config: GameConfig {
                num_seats,
                starting_stacks: stacks,
                ante,
                bb,
            },
        })
    }

    fn num_envs(&self) -> usize {
        self.states.len()
    }

    fn num_seats(&self) -> usize {
        self.config.num_seats
    }

    fn num_actions(&self) -> usize {
        NUM_ACTIONS
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
        let config = self.config.clone();
        let seeds_vec: Vec<u64> = seeds_slice.to_vec();
        let buttons_vec: Vec<u8> = buttons_slice.to_vec();
        let new_states: Vec<GameState> = py.allow_threads(move || {
            (0..n)
                .map(|i| GameState::new_hand(config.clone(), seeds_vec[i], buttons_vec[i] as usize))
                .collect()
        });
        for (i, s) in new_states.into_iter().enumerate() {
            self.states[i] = Some(s);
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
        let config = self.config.clone();
        let seeds_vec: Vec<u64> = seeds_slice.to_vec();
        let buttons_vec: Vec<u8> = buttons_slice.to_vec();
        let mask_vec: Vec<bool> = mask_slice.to_vec();
        let new_states: Vec<Option<GameState>> = py.allow_threads(move || {
            (0..n)
                .map(|i| {
                    if mask_vec[i] {
                        Some(GameState::new_hand(
                            config.clone(),
                            seeds_vec[i],
                            buttons_vec[i] as usize,
                        ))
                    } else {
                        None
                    }
                })
                .collect()
        });
        for (i, s) in new_states.into_iter().enumerate() {
            if let Some(state) = s {
                self.states[i] = Some(state);
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
        // Pre-validate: all non-terminal envs must have a legal action.
        for i in 0..n {
            let state = match self.states[i].as_ref() {
                Some(s) => s,
                None => {
                    return Err(PyRuntimeError::new_err(format!(
                        "env {} not reset; call reset_batch/reset_terminal_batch first",
                        i
                    )));
                }
            };
            if state.is_terminal() {
                continue;
            }
            let a_idx = actions_slice[i];
            let action = Action::from_index(a_idx).ok_or_else(|| {
                PyValueError::new_err(format!("invalid action index {} at env {}", a_idx, i))
            })?;
            let mask = state.legal_action_mask();
            if !mask[action.index() as usize] {
                return Err(PyValueError::new_err(format!(
                    "action {} illegal at env {}",
                    a_idx, i
                )));
            }
        }
        // All actions validated — mutate.
        let actions_vec: Vec<u8> = actions_slice.to_vec();
        let terminal: Array1<bool> = py.allow_threads(|| {
            let mut term = Array1::<bool>::default(n);
            for i in 0..n {
                let state = self.states[i].as_mut().expect("validated above");
                if state.is_terminal() {
                    term[i] = false;
                    continue;
                }
                let action = Action::from_index(actions_vec[i]).expect("validated above");
                state.apply(action);
                if state.is_terminal() {
                    term[i] = true;
                }
            }
            term
        });
        Ok(terminal.into_pyarray(py))
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
        // Pre-validate all non-terminal envs.
        for i in 0..n {
            let state = match self.states[i].as_ref() {
                Some(s) => s,
                None => {
                    return Err(PyRuntimeError::new_err(format!(
                        "env {} not reset; call reset_batch/reset_terminal_batch first",
                        i
                    )));
                }
            };
            if state.is_terminal() {
                continue;
            }
            let g = gates_slice[i];
            match g {
                0 => {
                    let m = state.legal_action_mask();
                    if !m[Action::Fold as usize] {
                        return Err(PyValueError::new_err(format!(
                            "gate Fold illegal at env {}",
                            i
                        )));
                    }
                }
                1 => {
                    let m = state.legal_action_mask();
                    if !m[Action::CheckCall as usize] {
                        return Err(PyValueError::new_err(format!(
                            "gate CheckCall illegal at env {}",
                            i
                        )));
                    }
                }
                2 => {
                    let min = state.min_raise_chips();
                    let max = state.max_raise_chips();
                    let c = chips_slice[i];
                    if min == 0 || c < min || c > max {
                        return Err(PyValueError::new_err(format!(
                            "gate Raise chips {} out of range [{}, {}] at env {}",
                            c, min, max, i
                        )));
                    }
                }
                3 => {
                    let m = state.legal_action_mask();
                    if !m[Action::AllIn as usize] {
                        return Err(PyValueError::new_err(format!(
                            "gate AllIn illegal at env {}",
                            i
                        )));
                    }
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "invalid gate {} at env {} (must be 0..=3)",
                        other, i
                    )));
                }
            }
        }
        let gates_vec: Vec<u8> = gates_slice.to_vec();
        let chips_vec: Vec<u64> = chips_slice.to_vec();
        let terminal: Array1<bool> = py.allow_threads(|| {
            let mut term = Array1::<bool>::default(n);
            for i in 0..n {
                let state = self.states[i].as_mut().expect("validated above");
                if state.is_terminal() {
                    term[i] = false;
                    continue;
                }
                match gates_vec[i] {
                    0 => state.apply(Action::Fold),
                    1 => state.apply(Action::CheckCall),
                    2 => state
                        .apply_raise_chips(chips_vec[i])
                        .expect("validated above"),
                    3 => state.apply(Action::AllIn),
                    _ => unreachable!(),
                }
                if state.is_terminal() {
                    term[i] = true;
                }
            }
            term
        });
        Ok(terminal.into_pyarray(py))
    }

    /// Apply continuous-sizing raises to every non-terminal env.
    /// `chips[i]` must be in `[min_raise_chips, max_raise_chips]` for env `i`.
    /// Terminal envs are skipped (their `chips[i]` entry is ignored).
    /// Returns a `(N,)` bool array where `true` means the env transitioned
    /// to terminal on this call.
    fn apply_raise_chips_batch<'py>(
        &mut self,
        py: Python<'py>,
        chips: PyReadonlyArray1<'_, u64>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let n = self.states.len();
        let chips_slice = chips.as_slice()?;
        if chips_slice.len() != n {
            return Err(PyValueError::new_err(format!(
                "chips length {} != num_envs {}",
                chips_slice.len(),
                n
            )));
        }
        // Pre-validate: every non-terminal env must have chips in legal range.
        for i in 0..n {
            let state = match self.states[i].as_ref() {
                Some(s) => s,
                None => {
                    return Err(PyRuntimeError::new_err(format!(
                        "env {} not reset; call reset_batch/reset_terminal_batch first",
                        i
                    )));
                }
            };
            if state.is_terminal() {
                continue;
            }
            let min = state.min_raise_chips();
            let max = state.max_raise_chips();
            let c = chips_slice[i];
            if min == 0 || c < min || c > max {
                return Err(PyValueError::new_err(format!(
                    "raise chips {} out of range [{}, {}] at env {}",
                    c, min, max, i
                )));
            }
        }
        let chips_vec: Vec<u64> = chips_slice.to_vec();
        let terminal: Array1<bool> = py.allow_threads(|| {
            let mut term = Array1::<bool>::default(n);
            for i in 0..n {
                let state = self.states[i].as_mut().expect("validated above");
                if state.is_terminal() {
                    term[i] = false;
                    continue;
                }
                state
                    .apply_raise_chips(chips_vec[i])
                    .expect("validated above");
                if state.is_terminal() {
                    term[i] = true;
                }
            }
            term
        });
        Ok(terminal.into_pyarray(py))
    }

    /// `(N, NUM_ACTIONS)` bool legal-action mask. Terminal envs return
    /// an all-false row.
    fn legal_mask_batch<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let n = self.states.len();
        let arr: Array2<bool> = py.allow_threads(|| {
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
        let arr: Array1<i8> = py.allow_threads(|| {
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

    /// `(N,)` bool: true where env is terminal (or not yet reset).
    fn is_terminal_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let n = self.states.len();
        let arr: Array1<bool> = py.allow_threads(|| {
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
        let arr: Array2<i64> = py.allow_threads(|| {
            let mut arr = Array2::<i64>::zeros((n, s));
            for i in 0..n {
                if let Some(state) = self.states[i].as_ref() {
                    if state.is_terminal() {
                        let p = state.payouts();
                        for k in 0..s {
                            arr[[i, k]] = p[k];
                        }
                    }
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
        let arr: Array2<i64> = py.allow_threads(move || {
            let mut arr = Array2::<i64>::zeros((n, s));
            for i in 0..n {
                if let Some(state) = self.states[i].as_ref() {
                    if state.is_terminal() {
                        let p = state.payouts_ev(num_samples, seeds_vec[i]);
                        for k in 0..s {
                            arr[[i, k]] = p[k];
                        }
                    }
                }
            }
            arr
        });
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
        let arr: Array1<u8> = py.allow_threads(move || {
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

    /// Stacked view of every field the vectorized encoder needs. Keys:
    ///
    /// - `hero_hole`         (N, 5)   u8   — hole of current actor per env;
    ///                                        255 sentinel when terminal.
    /// - `board_a` / `board_b` (N, 5) u8   — padded with 255 sentinels for
    ///                                        unrevealed cards.
    /// - `board_a_len` / `board_b_len` (N,) u8
    /// - `street`            (N,)     u8
    /// - `pot`               (N,)     u64
    /// - `stacks`            (N, S)   u64
    /// - `folded`            (N, S)   bool
    /// - `all_in`            (N, S)   bool
    /// - `bet_to_call`       (N,)     u64
    /// - `street_commit`     (N, S)   u64
    /// - `total_commit`      (N, S)   u64
    /// - `min_bet`           (N,)     u64
    /// - `max_bet`           (N,)     u64
    /// - `min_raise`         (N,)     u64  — chip delta (0 if Raise illegal).
    /// - `max_raise`         (N,)     u64  — chip delta, PL- and stack-capped.
    /// - `actor`             (N,)     i8   — -1 when terminal.
    /// - `button`            (N,)     u8
    /// - `last_aggressor`    (N,)     i8   — -1 when no aggressor (no raise yet).
    /// - `history_seat`      (N, 32)  i8   — -1 for empty slots.
    /// - `history_action`    (N, 32)  i8   — -1 for empty slots.
    /// - `history_chips`     (N, 32)  u64  — 0 for empty slots and for fold/check.
    /// - `history_street`    (N, 32)  i8   — -1 for empty slots; otherwise street idx.
    /// - `history_len`       (N,)     u8
    /// - `opp_outcome_fractions` (N, 12) f32 — row-major [k=2,3,4][outcome]
    ///                                          (scoop_opp, quarter_opp,
    ///                                           scoop_hero, quarter_hero).
    ///                                          All-zero pre-flop / terminal.
    fn observation_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let n = self.states.len();
        let s = self.config.num_seats;
        let packed: PackedObservation = py.allow_threads(|| self.pack_observation(n, s));

        let d = PyDict::new(py);
        d.set_item("hero_hole", packed.hero_hole.into_pyarray(py))?;
        d.set_item("board_a", packed.board_a.into_pyarray(py))?;
        d.set_item("board_b", packed.board_b.into_pyarray(py))?;
        d.set_item("board_a_len", packed.board_a_len.into_pyarray(py))?;
        d.set_item("board_b_len", packed.board_b_len.into_pyarray(py))?;
        d.set_item("street", packed.street.into_pyarray(py))?;
        d.set_item("pot", packed.pot.into_pyarray(py))?;
        d.set_item("stacks", packed.stacks.into_pyarray(py))?;
        d.set_item("folded", packed.folded.into_pyarray(py))?;
        d.set_item("all_in", packed.all_in.into_pyarray(py))?;
        d.set_item("bet_to_call", packed.bet_to_call.into_pyarray(py))?;
        d.set_item("street_commit", packed.street_commit.into_pyarray(py))?;
        d.set_item("total_commit", packed.total_commit.into_pyarray(py))?;
        d.set_item("min_bet", packed.min_bet.into_pyarray(py))?;
        d.set_item("max_bet", packed.max_bet.into_pyarray(py))?;
        d.set_item("min_raise", packed.min_raise.into_pyarray(py))?;
        d.set_item("max_raise", packed.max_raise.into_pyarray(py))?;
        d.set_item("eff_stack_cap", packed.eff_stack_cap.into_pyarray(py))?;
        d.set_item("actor", packed.actor.into_pyarray(py))?;
        d.set_item("button", packed.button.into_pyarray(py))?;
        d.set_item("last_aggressor", packed.last_aggressor.into_pyarray(py))?;
        d.set_item("history_seat", packed.history_seat.into_pyarray(py))?;
        d.set_item("history_action", packed.history_action.into_pyarray(py))?;
        d.set_item("history_chips", packed.history_chips.into_pyarray(py))?;
        d.set_item("history_street", packed.history_street.into_pyarray(py))?;
        d.set_item("history_len", packed.history_len.into_pyarray(py))?;
        d.set_item(
            "opp_outcome_fractions",
            packed.opp_outcome_fractions.into_pyarray(py),
        )?;
        Ok(d)
    }

    /// Phase D — one FFI crossing that returns everything the vectorized
    /// encoder + rollout driver need. Equivalent to calling:
    ///   observation_arrays() + legal_mask_batch() +
    ///   hero_category_batch(actor, 0|1) (×2)
    /// in one GIL-released block. Saves several FFI crossings per step.
    ///
    /// Keys added on top of `observation_arrays()`:
    ///   - `legal_mask`  (N, NUM_ACTIONS) bool
    ///   - `hero_cat_a`  (N,) u8
    ///   - `hero_cat_b`  (N,) u8
    ///
    /// Terminal envs get all-false masks and zero categories — the
    /// vectorized encoder ignores those rows anyway.
    fn observation_and_features_batch<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n = self.states.len();
        let s = self.config.num_seats;

        let (packed, legal_mask, cat_a, cat_b) = py.allow_threads(|| {
            let packed = self.pack_observation(n, s);
            let mut legal_mask = Array2::<bool>::default((n, NUM_ACTIONS));
            let mut cat_a = Array1::<u8>::zeros(n);
            let mut cat_b = Array1::<u8>::zeros(n);
            for i in 0..n {
                let state = match self.states[i].as_ref() {
                    Some(s) => s,
                    None => continue,
                };
                if !state.is_terminal() {
                    let mask = state.legal_action_mask();
                    for j in 0..NUM_ACTIONS {
                        legal_mask[[i, j]] = mask[j];
                    }
                }
                if let Some(a) = state.current_actor() {
                    cat_a[i] = state.hero_category(a, 0);
                    cat_b[i] = state.hero_category(a, 1);
                }
            }
            (packed, legal_mask, cat_a, cat_b)
        });

        let d = PyDict::new(py);
        d.set_item("hero_hole", packed.hero_hole.into_pyarray(py))?;
        d.set_item("board_a", packed.board_a.into_pyarray(py))?;
        d.set_item("board_b", packed.board_b.into_pyarray(py))?;
        d.set_item("board_a_len", packed.board_a_len.into_pyarray(py))?;
        d.set_item("board_b_len", packed.board_b_len.into_pyarray(py))?;
        d.set_item("street", packed.street.into_pyarray(py))?;
        d.set_item("pot", packed.pot.into_pyarray(py))?;
        d.set_item("stacks", packed.stacks.into_pyarray(py))?;
        d.set_item("folded", packed.folded.into_pyarray(py))?;
        d.set_item("all_in", packed.all_in.into_pyarray(py))?;
        d.set_item("bet_to_call", packed.bet_to_call.into_pyarray(py))?;
        d.set_item("street_commit", packed.street_commit.into_pyarray(py))?;
        d.set_item("total_commit", packed.total_commit.into_pyarray(py))?;
        d.set_item("min_bet", packed.min_bet.into_pyarray(py))?;
        d.set_item("max_bet", packed.max_bet.into_pyarray(py))?;
        d.set_item("min_raise", packed.min_raise.into_pyarray(py))?;
        d.set_item("max_raise", packed.max_raise.into_pyarray(py))?;
        d.set_item("eff_stack_cap", packed.eff_stack_cap.into_pyarray(py))?;
        d.set_item("actor", packed.actor.into_pyarray(py))?;
        d.set_item("button", packed.button.into_pyarray(py))?;
        d.set_item("last_aggressor", packed.last_aggressor.into_pyarray(py))?;
        d.set_item("history_seat", packed.history_seat.into_pyarray(py))?;
        d.set_item("history_action", packed.history_action.into_pyarray(py))?;
        d.set_item("history_chips", packed.history_chips.into_pyarray(py))?;
        d.set_item("history_street", packed.history_street.into_pyarray(py))?;
        d.set_item("history_len", packed.history_len.into_pyarray(py))?;
        d.set_item(
            "opp_outcome_fractions",
            packed.opp_outcome_fractions.into_pyarray(py),
        )?;
        d.set_item("legal_mask", legal_mask.into_pyarray(py))?;
        d.set_item("hero_cat_a", cat_a.into_pyarray(py))?;
        d.set_item("hero_cat_b", cat_b.into_pyarray(py))?;
        Ok(d)
    }
}

struct PackedObservation {
    hero_hole: Array2<u8>,
    board_a: Array2<u8>,
    board_b: Array2<u8>,
    board_a_len: Array1<u8>,
    board_b_len: Array1<u8>,
    street: Array1<u8>,
    pot: Array1<u64>,
    stacks: Array2<u64>,
    folded: Array2<bool>,
    all_in: Array2<bool>,
    bet_to_call: Array1<u64>,
    street_commit: Array2<u64>,
    total_commit: Array2<u64>,
    min_bet: Array1<u64>,
    max_bet: Array1<u64>,
    min_raise: Array1<u64>,
    max_raise: Array1<u64>,
    eff_stack_cap: Array2<u64>,
    actor: Array1<i8>,
    button: Array1<u8>,
    last_aggressor: Array1<i8>,
    history_seat: Array2<i8>,
    history_action: Array2<i8>,
    history_chips: Array2<u64>,
    history_street: Array2<i8>,
    history_len: Array1<u8>,
    opp_outcome_fractions: Array2<f32>,
}

impl PyBatchedEngine {
    fn pack_observation(&self, n: usize, s: usize) -> PackedObservation {
        let mut hero_hole = Array2::<u8>::from_elem((n, 5), 255u8);
        let mut board_a = Array2::<u8>::from_elem((n, 5), 255u8);
        let mut board_b = Array2::<u8>::from_elem((n, 5), 255u8);
        let mut board_a_len = Array1::<u8>::zeros(n);
        let mut board_b_len = Array1::<u8>::zeros(n);
        let mut street = Array1::<u8>::zeros(n);
        let mut pot = Array1::<u64>::zeros(n);
        let mut stacks = Array2::<u64>::zeros((n, s));
        let mut folded = Array2::<bool>::default((n, s));
        let mut all_in = Array2::<bool>::default((n, s));
        let mut bet_to_call = Array1::<u64>::zeros(n);
        let mut street_commit = Array2::<u64>::zeros((n, s));
        let mut total_commit = Array2::<u64>::zeros((n, s));
        let mut min_bet = Array1::<u64>::zeros(n);
        let mut max_bet = Array1::<u64>::zeros(n);
        let mut min_raise = Array1::<u64>::zeros(n);
        let mut max_raise = Array1::<u64>::zeros(n);
        let mut eff_stack_cap = Array2::<u64>::zeros((n, s));
        let mut actor = Array1::<i8>::from_elem(n, -1i8);
        let mut button = Array1::<u8>::zeros(n);
        let mut last_aggressor = Array1::<i8>::from_elem(n, -1i8);
        let mut history_seat = Array2::<i8>::from_elem((n, HISTORY_CAP), -1i8);
        let mut history_action = Array2::<i8>::from_elem((n, HISTORY_CAP), -1i8);
        let mut history_chips = Array2::<u64>::zeros((n, HISTORY_CAP));
        let mut history_street = Array2::<i8>::from_elem((n, HISTORY_CAP), -1i8);
        let mut history_len = Array1::<u8>::zeros(n);
        let mut opp_outcome_fractions = Array2::<f32>::zeros((n, 12));

        // Compute opp_outcome_fractions in parallel — this is the
        // expensive per-env work (k=2/3 exhaustive + k=4 MC=1024 hand
        // evaluations). The remaining per-env writes below are cheap
        // memcpy and stay serial.
        let opp_fr_per_env: Vec<[f32; 12]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut out = [0.0f32; 12];
                if let Some(state) = self.states[i].as_ref() {
                    let fr = state.opp_outcome_fractions();
                    for j in 0..12 {
                        out[j] = fr[j];
                    }
                }
                out
            })
            .collect();
        for i in 0..n {
            for j in 0..12 {
                opp_outcome_fractions[[i, j]] = opp_fr_per_env[i][j];
            }
        }

        for i in 0..n {
            let state = match self.states[i].as_ref() {
                Some(s) => s,
                None => continue,
            };
            street[i] = state.street.index() as u8;
            pot[i] = state.pot;
            bet_to_call[i] = state.bet_to_call;
            min_bet[i] = state.min_bet_total();
            max_bet[i] = state.max_bet_total();
            min_raise[i] = state.min_raise_chips();
            max_raise[i] = state.max_raise_chips();
            button[i] = state.button as u8;
            last_aggressor[i] = state.last_aggressor.map(|s| s as i8).unwrap_or(-1);
            let a_opt = state.current_actor();
            if let Some(a) = a_opt {
                actor[i] = a as i8;
                for (j, c) in state.hole_cards[a].iter().enumerate() {
                    hero_hole[[i, j]] = c.index();
                }
            }
            let la = state.board_a.len().min(5);
            let lb = state.board_b.len().min(5);
            board_a_len[i] = la as u8;
            board_b_len[i] = lb as u8;
            for j in 0..la {
                board_a[[i, j]] = state.board_a[j].index();
            }
            for j in 0..lb {
                board_b[[i, j]] = state.board_b[j].index();
            }
            for k in 0..s {
                stacks[[i, k]] = state.stacks[k];
                folded[[i, k]] = state.folded[k];
                all_in[[i, k]] = state.all_in[k];
                street_commit[[i, k]] = state.street_commit[k];
                total_commit[[i, k]] = state.total_commit[k];
                eff_stack_cap[[i, k]] = state.eff_stack_cap_at_hand_start[k];
            }
            // Keep the last HISTORY_CAP entries oldest-first, matching the
            // scalar encoder's slice semantics.
            let hist_len = state.history.len();
            let start = if hist_len > HISTORY_CAP {
                hist_len - HISTORY_CAP
            } else {
                0
            };
            let kept = hist_len - start;
            history_len[i] = kept as u8;
            for (slot, rec) in state.history[start..].iter().enumerate() {
                history_seat[[i, slot]] = rec.seat as i8;
                history_action[[i, slot]] = rec.action.index() as i8;
                history_chips[[i, slot]] = rec.chips;
                history_street[[i, slot]] = rec.street.index() as i8;
            }
        }

        PackedObservation {
            hero_hole,
            board_a,
            board_b,
            board_a_len,
            board_b_len,
            street,
            pot,
            stacks,
            folded,
            all_in,
            bet_to_call,
            street_commit,
            total_commit,
            min_bet,
            max_bet,
            min_raise,
            max_raise,
            eff_stack_cap,
            actor,
            button,
            last_aggressor,
            history_seat,
            history_action,
            history_chips,
            history_street,
            history_len,
            opp_outcome_fractions,
        }
    }
}
