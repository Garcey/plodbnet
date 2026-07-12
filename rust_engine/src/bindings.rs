//! PyO3 wrapper around [`GameState`]. Exposes a minimal surface for the
//! Python environment and training loop.

use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::actions::{Action, NUM_ACTIONS};
use crate::cards::Card;
use crate::state::{GameConfig, GameState, StudyTerminal, Variant};

/// Parse the Python-facing variant string. Kept as strings (not an
/// exported enum class) so the Python config layer stays a plain
/// dataclass field.
fn parse_variant(s: &str) -> PyResult<Variant> {
    match s {
        "plo4_double_bomb" => Ok(Variant::Plo4DoubleBomb),
        "plo5_double_bomb" => Ok(Variant::Plo5DoubleBomb),
        "plo6_double_bomb" => Ok(Variant::Plo6DoubleBomb),
        "nlh_single" => Ok(Variant::NlhSingle),
        _ => Err(PyValueError::new_err(format!(
            "unknown variant '{s}' (expected 'plo4_double_bomb', 'plo5_double_bomb', \
             'plo6_double_bomb', or 'nlh_single')"
        ))),
    }
}

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
    #[pyo3(signature = (num_seats=6, starting_stack=200000, ante=30000, bb=10000, starting_stacks=None, variant="plo5_double_bomb", sb=0))]
    fn new(
        num_seats: usize,
        starting_stack: u64,
        ante: u64,
        bb: u64,
        starting_stacks: Option<PyReadonlyArray1<'_, u64>>,
        variant: &str,
        sb: u64,
    ) -> PyResult<Self> {
        let stacks = resolve_starting_stacks(num_seats, starting_stack, starting_stacks)?;
        let variant = parse_variant(variant)?;
        Ok(PyGameState {
            inner: None,
            config: GameConfig {
                num_seats,
                starting_stacks: stacks,
                ante,
                bb,
                sb,
                variant,
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

    /// NLH study-mode entry: user-supplied 2-card hero hole, hand starts
    /// at the PREFLOP with blinds posted. Streets are supplied via
    /// `set_flop_nlh` / `set_turn_nlh` / `set_river_nlh` as rounds close.
    fn reset_study_nlh(
        &mut self,
        button: usize,
        hero_seat: usize,
        hero_hole: Vec<u8>,
    ) -> PyResult<()> {
        let hero_hole = cards_from_indices::<2>(&hero_hole, "hero_hole")?;
        match GameState::new_study_nlh(self.config.clone(), button, hero_seat, hero_hole) {
            Ok(g) => {
                self.inner = Some(g);
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }

    fn set_flop_nlh(&mut self, c0: u8, c1: u8, c2: u8) -> PyResult<()> {
        let g = self.get_mut()?;
        g.set_flop_nlh([
            card_from_index(c0)?,
            card_from_index(c1)?,
            card_from_index(c2)?,
        ])
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn set_turn_nlh(&mut self, card: u8) -> PyResult<()> {
        let g = self.get_mut()?;
        g.set_turn_nlh(card_from_index(card)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn set_river_nlh(&mut self, card: u8) -> PyResult<()> {
        let g = self.get_mut()?;
        g.set_river_nlh(card_from_index(card)?)
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

    /// 20-dim superset: the 12 joint outcome fractions + the 8-dim
    /// per-board decomposition (obs v2 P1), one fused pass. See
    /// `GameState::outcome_features_mc`.
    fn outcome_features_mc(&self, mc_samples: usize) -> PyResult<Vec<f32>> {
        Ok(self.get()?.outcome_features_mc(mc_samples))
    }

    /// Like `opp_outcome_fractions` but with an explicit k=3/k=4 MC
    /// sample budget (the no-arg form uses 1024). For tests / benchmarks
    /// of the training-vs-UI fidelity split.
    fn opp_outcome_fractions_mc(&self, mc_samples: usize) -> PyResult<Vec<f32>> {
        Ok(self.get()?.opp_outcome_fractions_mc(mc_samples))
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
        d.set_item("sb_seat", g.sb_seat)?;
        d.set_item("bb_seat", g.bb_seat)?;
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

        // One fused pass computes the 12 joint fractions, the 8-dim
        // per-board decomposition (obs v2 P1), and the k=2 share bounds
        // (v7 DUAL-4) — same cost as the old opp_outcome_fractions call.
        let outcome_feats = g.outcome_features_mc(1024);
        d.set_item("opp_outcome_fractions", outcome_feats[..12].to_vec())?;
        d.set_item("per_board_outcome", outcome_feats[12..20].to_vec())?;
        d.set_item("share_bounds", outcome_feats[20..22].to_vec())?;
        // v7 batch-2 engine dims (STK-1 / BRD-7 / BRD-12 / DUAL-2).
        d.set_item("acted_this_street", g.acted_this_street.clone())?;
        d.set_item("hero_board_v3", g.hero_board_v3().to_vec())?;
        // NLH 3-dim [opp_ahead, tied, opp_behind]; cheap zeros for other
        // variants (the method's variant guard returns before any eval).
        d.set_item("nlh_opp_outcome", g.nlh_opp_outcome_fractions())?;

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

    /// Range-grid packer: this NLH decision node packed once per candidate
    /// actor hole, in the exact `observation_and_features_batch` layout
    /// (`nlh_single`) consumed by `encode_observation_batch_nlh`. The
    /// observation is villain-blind, so every field is identical across
    /// rows except the actor's hole cards and the two hole-derived inputs
    /// (`hero_cat_a`, `nlh_opp_outcome`), recomputed per combo in
    /// parallel. `holes` is (N, 2) card indices; each combo must be two
    /// distinct cards, none on the board (villain-placeholder collisions
    /// are fine — placeholders never enter the observation). Requires a
    /// live actor (not terminal, not awaiting a street card).
    fn pack_range_nlh<'py>(
        &self,
        py: Python<'py>,
        holes: PyReadonlyArray2<'_, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let g = self.get()?;
        if g.config.variant != Variant::NlhSingle {
            return Err(PyValueError::new_err("pack_range_nlh is NLH-only"));
        }
        if g.current_actor().is_none() {
            return Err(PyValueError::new_err(
                "no current actor (terminal or awaiting a street card)",
            ));
        }
        let actor = g.current_actor().unwrap();
        let holes = holes.as_array();
        if holes.ncols() != 2 {
            return Err(PyValueError::new_err("holes must have shape (N, 2)"));
        }
        let n = holes.nrows();
        let s = self.config.num_seats;

        let mut on_board = [false; 52];
        for c in g.board_a.iter() {
            on_board[c.index() as usize] = true;
        }
        let mut combos: Vec<[Card; 2]> = Vec::with_capacity(n);
        for i in 0..n {
            let (c0, c1) = (holes[[i, 0]], holes[[i, 1]]);
            if c0 >= 52 || c1 >= 52 || c0 == c1 {
                return Err(PyValueError::new_err(format!(
                    "invalid combo at row {i}: ({c0}, {c1})"
                )));
            }
            if on_board[c0 as usize] || on_board[c1 as usize] {
                return Err(PyValueError::new_err(format!(
                    "combo at row {i} collides with the board"
                )));
            }
            combos.push([Card::from_index(c0), Card::from_index(c1)]);
        }

        let hist_cap = history_cap(Variant::NlhSingle);
        let (packed, hero_cat_a) = py.allow_threads(move || {
            // Per-combo hole-derived features — the only expensive part
            // (exhaustive opp-outcome sweep per combo postflop).
            let per_combo: Vec<(u8, [f32; 3])> = combos
                .par_iter()
                .map(|h| {
                    (
                        crate::engine::nlh_category_for(h, &g.board_a),
                        crate::engine::nlh_opp_outcome_for(h, &g.board_a),
                    )
                })
                .collect();

            let mut hero_hole = Array2::<u8>::from_elem((n, 2), 255u8);
            let mut board_a = Array2::<u8>::from_elem((n, 5), 255u8);
            let board_b = Array2::<u8>::from_elem((n, 5), 255u8);
            let la = g.board_a.len().min(5);
            let board_a_len = Array1::<u8>::from_elem(n, la as u8);
            let board_b_len = Array1::<u8>::zeros(n);
            let street = Array1::<u8>::from_elem(n, g.street.index() as u8);
            let pot = Array1::<u64>::from_elem(n, g.pot);
            let bet_to_call = Array1::<u64>::from_elem(n, g.bet_to_call);
            let min_bet = Array1::<u64>::from_elem(n, g.min_bet_total());
            let max_bet = Array1::<u64>::from_elem(n, g.max_bet_total());
            let min_raise = Array1::<u64>::from_elem(n, g.min_raise_chips());
            let max_raise = Array1::<u64>::from_elem(n, g.max_raise_chips());
            let actor_arr = Array1::<i8>::from_elem(n, actor as i8);
            let button = Array1::<u8>::from_elem(n, g.button as u8);
            let last_aggressor = Array1::<i8>::from_elem(
                n,
                g.last_aggressor.map(|x| x as i8).unwrap_or(-1),
            );
            let sb_seat =
                Array1::<i8>::from_elem(n, g.sb_seat.map(|x| x as i8).unwrap_or(-1));
            let bb_seat =
                Array1::<i8>::from_elem(n, g.bb_seat.map(|x| x as i8).unwrap_or(-1));

            let mut stacks = Array2::<u64>::zeros((n, s));
            let mut folded = Array2::<bool>::default((n, s));
            let mut all_in = Array2::<bool>::default((n, s));
            let mut street_commit = Array2::<u64>::zeros((n, s));
            let mut total_commit = Array2::<u64>::zeros((n, s));
            let mut eff_stack_cap = Array2::<u64>::zeros((n, s));
            let mut acted_this_street = Array2::<bool>::default((n, s));
            let mut history_seat = Array2::<i8>::from_elem((n, hist_cap), -1i8);
            let mut history_action = Array2::<i8>::from_elem((n, hist_cap), -1i8);
            let mut history_chips = Array2::<u64>::zeros((n, hist_cap));
            let mut history_street = Array2::<i8>::from_elem((n, hist_cap), -1i8);

            let hist_len = g.history.len();
            let start = hist_len.saturating_sub(hist_cap);
            let kept = hist_len - start;
            let history_len = Array1::<u8>::from_elem(n, kept as u8);

            for i in 0..n {
                for j in 0..la {
                    board_a[[i, j]] = g.board_a[j].index();
                }
                for k in 0..s {
                    stacks[[i, k]] = g.stacks[k];
                    folded[[i, k]] = g.folded[k];
                    all_in[[i, k]] = g.all_in[k];
                    street_commit[[i, k]] = g.street_commit[k];
                    total_commit[[i, k]] = g.total_commit[k];
                    eff_stack_cap[[i, k]] = g.eff_stack_cap_at_hand_start[k];
                    acted_this_street[[i, k]] = g.acted_this_street[k];
                }
                for (slot, rec) in g.history[start..].iter().enumerate() {
                    history_seat[[i, slot]] = rec.seat as i8;
                    history_action[[i, slot]] = rec.action.index() as i8;
                    history_chips[[i, slot]] = rec.chips;
                    history_street[[i, slot]] = rec.street.index() as i8;
                }
            }

            let mut cat = Array1::<u8>::zeros(n);
            let mut nlh_opp_outcome = Array2::<f32>::zeros((n, 3));
            for (i, (c, fr)) in per_combo.iter().enumerate() {
                hero_hole[[i, 0]] = combos[i][0].index();
                hero_hole[[i, 1]] = combos[i][1].index();
                cat[i] = *c;
                for j in 0..3 {
                    nlh_opp_outcome[[i, j]] = fr[j];
                }
            }

            let packed = PackedObservation {
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
                actor: actor_arr,
                button,
                last_aggressor,
                history_seat,
                history_action,
                history_chips,
                history_street,
                history_len,
                opp_outcome_fractions: Array2::<f32>::zeros((n, 12)),
                per_board_outcome: Array2::<f32>::zeros((n, 8)),
                share_bounds: Array2::<f32>::zeros((n, 2)),
                acted_this_street,
                hero_board_v3: Array2::<u8>::zeros((n, 8)),
                sb_seat,
                bb_seat,
                nlh_opp_outcome,
            };
            (packed, cat)
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
        d.set_item(
            "per_board_outcome",
            packed.per_board_outcome.into_pyarray(py),
        )?;
        d.set_item("share_bounds", packed.share_bounds.into_pyarray(py))?;
        d.set_item(
            "acted_this_street",
            packed.acted_this_street.into_pyarray(py),
        )?;
        d.set_item("hero_board_v3", packed.hero_board_v3.into_pyarray(py))?;
        d.set_item("sb_seat", packed.sb_seat.into_pyarray(py))?;
        d.set_item("bb_seat", packed.bb_seat.into_pyarray(py))?;
        d.set_item("nlh_opp_outcome", packed.nlh_opp_outcome.into_pyarray(py))?;
        d.set_item("hero_cat_a", hero_cat_a.into_pyarray(py))?;
        d.set_item("hero_cat_b", Array1::<u8>::zeros(n).into_pyarray(py))?;
        Ok(d)
    }

    /// All seats' hole cards as raw indices, 5 per seat. Trainer-only
    /// accessor for opponent reveal at hand end; never feed into
    /// observations mid-hand.
    fn all_hole_cards(&self) -> PyResult<Vec<Vec<u8>>> {
        let g = self.get()?;
        Ok(g.hole_cards
            .iter()
            .map(|h| h.iter().map(|c| c.index()).collect())
            .collect())
    }
}

/// Mirror of `python/plo5bp/rollout.py:_aggression_bonus_bb`. Pot-fraction
/// bonus on voluntary aggression: chips committed beyond the actor's
/// amount-to-call, capped at the pre-step pot. Zero for non-RAISE gates
/// or non-positive `c`.
const GATE_RAISE_U8: u8 = 2;

#[inline]
fn aggression_bonus_bb_inner(
    gate: u8,
    commit_delta_chips: i64,
    bet_to_call_chips: i64,
    street_commit_actor_chips: i64,
    pot_chips_pre: i64,
    c: f64,
) -> f64 {
    if c <= 0.0 || gate != GATE_RAISE_U8 {
        return 0.0;
    }
    let call_chips = (bet_to_call_chips - street_commit_actor_chips).max(0);
    let aggressive = (commit_delta_chips - call_chips).max(0);
    if aggressive <= 0 || pot_chips_pre <= 0 {
        return 0.0;
    }
    let mut ratio = aggressive as f64 / pot_chips_pre as f64;
    if ratio > 1.0 {
        ratio = 1.0;
    }
    c * ratio
}

/// Per-env aggression-bonus computation for the batched rollout driver,
/// parallelized via rayon with the GIL released. Replaces the linear
/// per-env Python loop in `collect_rollout_batched`.
///
/// Inputs (all length `N` along axis 0):
/// - `actors`: current actor seat (-1 for terminal).
/// - `dones`: env terminated flag.
/// - `learner_mask`: `(N, num_seats)` — true where seat is a learner seat
///   in env `i`.
/// - `gates`: emitted hybrid gate per env.
/// - `pre_total_commit` / `post_total_commit`: `(N, num_seats)` cumulative
///   per-seat chip commit, snapshotted before/after the current step.
/// - `pre_bet_to_call`: env-level max street_commit before the step.
/// - `pre_street_commit`: `(N, num_seats)` per-seat street_commit before.
/// - `pre_street`: street index before the step (0=preflop ... 3=river).
/// - `c`: aggression-bonus coefficient.
/// - `reward_norm`: `1 / bb`, applied to per-step delta and pot to keep
///   the trajectory in bb units.
///
/// Returns a dict with per-env arrays plus pre-reduced scalar diagnostics:
/// - `valid`: `(N,)` bool — true iff the env contributed a learner step.
/// - `cost_increment`: `(N,)` f64 — `-(delta_chips * reward_norm) + bonus_bb`.
/// - `pot_pre_bb`: `(N,)` f64 — pre-step pot in bb.
/// - `street_pre`: `(N,)` i8 — engine street index, copied from input.
/// - `bonus_bb`: `(N,)` f64 — per-env bonus contribution (zero where
///   `!valid`).
/// - `total_bonus_bb`, `total_steps`, `bonus_steps`: serial reductions.
/// - `steps_by_street` / `bonus_steps_by_street`: 3-elem u64 arrays
///   bucketed by `(street_pre - 1)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn compute_aggression_bonus_batch<'py>(
    py: Python<'py>,
    actors: PyReadonlyArray1<'_, i8>,
    dones: PyReadonlyArray1<'_, bool>,
    learner_mask: PyReadonlyArray2<'_, bool>,
    gates: PyReadonlyArray1<'_, u8>,
    pre_total_commit: PyReadonlyArray2<'_, i64>,
    post_total_commit: PyReadonlyArray2<'_, i64>,
    pre_bet_to_call: PyReadonlyArray1<'_, u64>,
    pre_street_commit: PyReadonlyArray2<'_, u64>,
    pre_street: PyReadonlyArray1<'_, u8>,
    c: f64,
    reward_norm: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let actors_s = actors.as_slice()?;
    let dones_s = dones.as_slice()?;
    let gates_s = gates.as_slice()?;
    let pre_btc_s = pre_bet_to_call.as_slice()?;
    let pre_street_s = pre_street.as_slice()?;
    let n = actors_s.len();
    if dones_s.len() != n
        || gates_s.len() != n
        || pre_btc_s.len() != n
        || pre_street_s.len() != n
    {
        return Err(PyValueError::new_err(
            "1-D input lengths must all equal num_envs",
        ));
    }

    let lm_view = learner_mask.as_array();
    let pre_tc_view = pre_total_commit.as_array();
    let post_tc_view = post_total_commit.as_array();
    let pre_sc_view = pre_street_commit.as_array();
    if pre_tc_view.shape()[0] != n {
        return Err(PyValueError::new_err(
            "pre_total_commit shape[0] must equal num_envs",
        ));
    }
    let num_seats = pre_tc_view.shape()[1];
    if lm_view.shape() != [n, num_seats]
        || post_tc_view.shape() != [n, num_seats]
        || pre_sc_view.shape() != [n, num_seats]
    {
        return Err(PyValueError::new_err(
            "2-D input shapes must all equal (num_envs, num_seats)",
        ));
    }

    // Materialise owned copies so the parallel section can run with the
    // GIL released. The arrays are small relative to the rollout work.
    let actors_v = actors_s.to_vec();
    let dones_v = dones_s.to_vec();
    let gates_v = gates_s.to_vec();
    let pre_btc_v = pre_btc_s.to_vec();
    let pre_street_v = pre_street_s.to_vec();
    let lm_owned = lm_view.to_owned();
    let pre_tc_owned = pre_tc_view.to_owned();
    let post_tc_owned = post_tc_view.to_owned();
    let pre_sc_owned = pre_sc_view.to_owned();

    // Per-env outputs computed in parallel; the serial reduction below
    // is O(N) and cheap relative to the per-env arithmetic + i64 row sum.
    let per_env: Vec<(bool, f64, f64, i8, f64)> = py.allow_threads(|| {
        (0..n)
            .into_par_iter()
            .map(|i| {
                if dones_v[i] {
                    return (false, 0.0, 0.0, -1i8, 0.0);
                }
                let a = actors_v[i];
                if a < 0 {
                    return (false, 0.0, 0.0, -1i8, 0.0);
                }
                let actor = a as usize;
                if actor >= num_seats || !lm_owned[[i, actor]] {
                    return (false, 0.0, 0.0, -1i8, 0.0);
                }

                let pre_a = pre_tc_owned[[i, actor]];
                let post_a = post_tc_owned[[i, actor]];
                let delta = (post_a - pre_a).max(0);

                let mut pot_pre: i64 = 0;
                for s in 0..num_seats {
                    pot_pre += pre_tc_owned[[i, s]];
                }
                let pot_pre = pot_pre.max(0);

                let bet_to_call_pre = pre_btc_v[i] as i64;
                let sc_actor_pre = pre_sc_owned[[i, actor]] as i64;
                let gate = gates_v[i];

                let bonus_bb = aggression_bonus_bb_inner(
                    gate,
                    delta,
                    bet_to_call_pre,
                    sc_actor_pre,
                    pot_pre,
                    c,
                );
                let cost_inc = -(delta as f64) * reward_norm + bonus_bb;
                let pot_pre_bb = pot_pre as f64 * reward_norm;
                let street_pre = pre_street_v[i] as i8;
                (true, cost_inc, pot_pre_bb, street_pre, bonus_bb)
            })
            .collect()
    });

    let mut valid_v: Vec<bool> = Vec::with_capacity(n);
    let mut cost_inc_v: Vec<f64> = Vec::with_capacity(n);
    let mut pot_pre_bb_v: Vec<f64> = Vec::with_capacity(n);
    let mut street_pre_v: Vec<i8> = Vec::with_capacity(n);
    let mut bonus_bb_v: Vec<f64> = Vec::with_capacity(n);
    let mut total_bonus_bb: f64 = 0.0;
    let mut total_steps: u64 = 0;
    let mut bonus_steps: u64 = 0;
    let mut steps_by_street: [u64; 3] = [0, 0, 0];
    let mut bonus_steps_by_street: [u64; 3] = [0, 0, 0];
    for (v, ci, pp, sp, bb) in per_env.into_iter() {
        valid_v.push(v);
        cost_inc_v.push(ci);
        pot_pre_bb_v.push(pp);
        street_pre_v.push(sp);
        bonus_bb_v.push(bb);
        if v {
            total_steps += 1;
            total_bonus_bb += bb;
            if bb > 0.0 {
                bonus_steps += 1;
            }
            let bucket = sp as i64 - 1;
            if (0..3).contains(&bucket) {
                steps_by_street[bucket as usize] += 1;
                if bb > 0.0 {
                    bonus_steps_by_street[bucket as usize] += 1;
                }
            }
        }
    }

    let d = PyDict::new(py);
    d.set_item("valid", Array1::from_vec(valid_v).into_pyarray(py))?;
    d.set_item(
        "cost_increment",
        Array1::from_vec(cost_inc_v).into_pyarray(py),
    )?;
    d.set_item(
        "pot_pre_bb",
        Array1::from_vec(pot_pre_bb_v).into_pyarray(py),
    )?;
    d.set_item(
        "street_pre",
        Array1::from_vec(street_pre_v).into_pyarray(py),
    )?;
    d.set_item("bonus_bb", Array1::from_vec(bonus_bb_v).into_pyarray(py))?;
    d.set_item("total_bonus_bb", total_bonus_bb)?;
    d.set_item("total_steps", total_steps)?;
    d.set_item("bonus_steps", bonus_steps)?;
    d.set_item(
        "steps_by_street",
        Array1::from_vec(steps_by_street.to_vec()).into_pyarray(py),
    )?;
    d.set_item(
        "bonus_steps_by_street",
        Array1::from_vec(bonus_steps_by_street.to_vec()).into_pyarray(py),
    )?;
    Ok(d)
}

/// Diagnostic hook: compute layered side-pot payouts for an arbitrary
/// commit / hole-card / board configuration. Wraps the pure
/// [`crate::double_board::double_board_payout`] function so Python can
/// verify side-pot handling without driving a real GameState.
///
/// `hole_cards` must be `hole_count * num_seats` card indices (5 or 6 per
/// seat), flat-packed by seat
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
    let hole_w = if n > 0 && hole_cards.len() == 6 * n {
        6
    } else if n > 0 && hole_cards.len() == 4 * n {
        4
    } else {
        5
    };
    if hole_cards.len() != hole_w * n {
        return Err(PyValueError::new_err(
            "hole_cards must have 4*num_seats (PLO4), 5*num_seats (PLO5), \
             or 6*num_seats (PLO6) indices",
        ));
    }
    if button >= n {
        return Err(PyValueError::new_err("button out of range"));
    }
    let mut holes: Vec<Vec<Card>> = Vec::with_capacity(n);
    for s in 0..n {
        let slice = &hole_cards[hole_w * s..hole_w * (s + 1)];
        let mut hole = Vec::with_capacity(hole_w);
        for &ix in slice {
            if ix >= 52 {
                return Err(PyValueError::new_err("hole_cards index out of range"));
            }
            hole.push(Card::from_index(ix));
        }
        holes.push(hole);
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

/// Batched-packer history width per variant. Must equal the variant's
/// batch-encoder slot count EXACTLY: the packer keeps the LAST `cap`
/// records oldest-first from slot 0, so a buffer wider than the encoder's
/// depth would hand it the oldest records instead of the newest. PLO
/// encodes 32 slots (`encoding._HISTORY_DEPTH`); NLH encodes 40
/// (`encoding_nlh._HISTORY_DEPTH` — the preflop round adds actions).
fn history_cap(variant: Variant) -> usize {
    match variant {
        Variant::NlhSingle => 40,
        _ => HISTORY_CAP,
    }
}

/// Python-facing batched game engine. Construct with `(num_envs, config)`,
/// then `reset_batch(seeds, buttons)` to seed all envs. `apply_action_batch`
/// steps every env; terminal envs stay terminal until `reset_terminal_batch`.
#[pyclass(name = "BatchedEngine")]
pub struct PyBatchedEngine {
    states: Vec<Option<GameState>>,
    config: GameConfig,
    /// k=3/k=4 Monte-Carlo budget for `opp_outcome_fractions` on this
    /// engine. Serial/UI/eval use 1024; batched TRAINING sets it lower
    /// (256) to cut the dominant per-decision encode cost.
    opp_outcome_mc: usize,
    /// Per-env memoization of the opp-outcome MC output (the 20-dim fused
    /// pass), keyed on `GameState::outcome_seed` = hash of exactly the MC's
    /// inputs (hero seat + street + hero hole + both boards). The MC is a pure
    /// function of those, so within a street (board unchanged across the
    /// street's actions) it recomputes identically every step; this reuses it
    /// and recomputes only when the seed changes (street advance / new hand).
    /// Bit-exact vs always-recompute (pinned by test_encoding_rust: cached
    /// batched == fresh serial). Accessed only serially (locked outside the
    /// parallel MC), so the Mutex adds no contention and keeps the pyclass Sync.
    outcome_cache: std::sync::Mutex<Vec<Option<(u64, [f32; 22])>>>,
}

#[pymethods]
impl PyBatchedEngine {
    #[new]
    #[pyo3(signature = (num_envs, num_seats=6, starting_stack=200000, ante=30000, bb=10000, starting_stacks=None, opp_outcome_mc=1024, variant="plo5_double_bomb", sb=0))]
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
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be >= 1"));
        }
        if num_seats < 2 {
            return Err(PyValueError::new_err("num_seats must be >= 2"));
        }
        if opp_outcome_mc == 0 {
            return Err(PyValueError::new_err("opp_outcome_mc must be >= 1"));
        }
        let variant = parse_variant(variant)?;
        let stacks = resolve_starting_stacks(num_seats, starting_stack, starting_stacks)?;
        Ok(PyBatchedEngine {
            states: (0..num_envs).map(|_| None).collect(),
            config: GameConfig {
                num_seats,
                starting_stacks: stacks,
                ante,
                bb,
                sb,
                variant,
            },
            opp_outcome_mc,
            outcome_cache: std::sync::Mutex::new(vec![None; num_envs]),
        })
    }

    fn num_envs(&self) -> usize {
        self.states.len()
    }

    fn num_seats(&self) -> usize {
        self.config.num_seats
    }

    /// (N, num_seats, 5) u8 hole-card indices for every seat in every
    /// env; rows for dead (never-reset) envs are 255-filled. Holes are
    /// static per hand, so callers cache this once per reset wave —
    /// it feeds the centralized critic's opponent-hole inputs.
    fn all_hole_cards_batch<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
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
                .into_par_iter()
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
        let term_vec: Vec<bool> = py.allow_threads(|| {
            self.states
                .par_iter_mut()
                .enumerate()
                .map(|(i, state_opt)| {
                    let state = state_opt.as_mut().expect("validated above");
                    if state.is_terminal() {
                        return false;
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
                    state.is_terminal()
                })
                .collect()
        });
        let terminal = Array1::from_vec(term_vec);
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
        let states_ref = &self.states;
        let arr: Array2<i64> = py.allow_threads(|| {
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
        let states_ref = &self.states;
        let arr: Array2<i64> = py.allow_threads(move || {
            let rows: Vec<Vec<i64>> = states_ref
                .par_iter()
                .enumerate()
                .map(|(i, state_opt)| match state_opt.as_ref() {
                    Some(state) if state.is_terminal() => {
                        state.payouts_ev(num_samples, seeds_vec[i])
                    }
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
    /// - `hero_hole`         (N, hole_count) u8 — hole of current actor per env;
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
        d.set_item(
            "per_board_outcome",
            packed.per_board_outcome.into_pyarray(py),
        )?;
        d.set_item("share_bounds", packed.share_bounds.into_pyarray(py))?;
        d.set_item(
            "acted_this_street",
            packed.acted_this_street.into_pyarray(py),
        )?;
        d.set_item("hero_board_v3", packed.hero_board_v3.into_pyarray(py))?;
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
        d.set_item(
            "per_board_outcome",
            packed.per_board_outcome.into_pyarray(py),
        )?;
        d.set_item("share_bounds", packed.share_bounds.into_pyarray(py))?;
        d.set_item(
            "acted_this_street",
            packed.acted_this_street.into_pyarray(py),
        )?;
        d.set_item("hero_board_v3", packed.hero_board_v3.into_pyarray(py))?;
        d.set_item("sb_seat", packed.sb_seat.into_pyarray(py))?;
        d.set_item("bb_seat", packed.bb_seat.into_pyarray(py))?;
        d.set_item("nlh_opp_outcome", packed.nlh_opp_outcome.into_pyarray(py))?;
        d.set_item("legal_mask", legal_mask.into_pyarray(py))?;
        d.set_item("hero_cat_a", cat_a.into_pyarray(py))?;
        d.set_item("hero_cat_b", cat_b.into_pyarray(py))?;
        Ok(d)
    }

    /// Subset variant of `observation_and_features_batch`: packs + features
    /// ONLY the envs in `indices`, returning compact (k = indices.len()) rows
    /// in the SAME dict layout / dtypes. The Python rollout uses this to
    /// refresh just the reset-terminal envs after `reset_terminal_batch`,
    /// since every other env's state is byte-identical to the preceding full
    /// refresh. Keys and dtypes mirror `observation_and_features_batch`
    /// exactly — keep the two in lockstep if either gains a field.
    fn observation_and_features_subset_batch<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let idx: Vec<usize> = indices.as_slice()?.iter().map(|&x| x as usize).collect();
        let k = idx.len();
        let s = self.config.num_seats;

        let (packed, legal_mask, cat_a, cat_b) = py.allow_threads(|| {
            let packed = self.pack_observation_indexed(&idx, s);
            let mut legal_mask = Array2::<bool>::default((k, NUM_ACTIONS));
            let mut cat_a = Array1::<u8>::zeros(k);
            let mut cat_b = Array1::<u8>::zeros(k);
            for j in 0..k {
                let state = match self.states[idx[j]].as_ref() {
                    Some(s) => s,
                    None => continue,
                };
                if !state.is_terminal() {
                    let mask = state.legal_action_mask();
                    for t in 0..NUM_ACTIONS {
                        legal_mask[[j, t]] = mask[t];
                    }
                }
                if let Some(a) = state.current_actor() {
                    cat_a[j] = state.hero_category(a, 0);
                    cat_b[j] = state.hero_category(a, 1);
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
        d.set_item(
            "per_board_outcome",
            packed.per_board_outcome.into_pyarray(py),
        )?;
        d.set_item("share_bounds", packed.share_bounds.into_pyarray(py))?;
        d.set_item(
            "acted_this_street",
            packed.acted_this_street.into_pyarray(py),
        )?;
        d.set_item("hero_board_v3", packed.hero_board_v3.into_pyarray(py))?;
        d.set_item("sb_seat", packed.sb_seat.into_pyarray(py))?;
        d.set_item("bb_seat", packed.bb_seat.into_pyarray(py))?;
        d.set_item("nlh_opp_outcome", packed.nlh_opp_outcome.into_pyarray(py))?;
        d.set_item("legal_mask", legal_mask.into_pyarray(py))?;
        d.set_item("hero_cat_a", cat_a.into_pyarray(py))?;
        d.set_item("hero_cat_b", cat_b.into_pyarray(py))?;
        Ok(d)
    }

    /// One-FFI encoder: returns the finished (N, OBS_DIM) f32 observation plus
    /// the aux fields the rollout reads, replacing pack + 4 feature calls +
    /// the numpy assembly. Bit-exact with `encode_observation_batch`.
    fn observation_encoded_batch<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let idx: Vec<usize> = (0..self.states.len()).collect();
        self.encode_indexed(py, &idx)
    }

    /// Subset variant: encodes ONLY the envs in `indices` into compact
    /// (k = indices.len()) rows. Used by the rollout's post-reset refresh to
    /// re-encode just the reset-terminal envs.
    fn observation_encoded_subset_batch<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let idx: Vec<usize> = indices.as_slice()?.iter().map(|&x| x as usize).collect();
        self.encode_indexed(py, &idx)
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
    /// Per-board hero ahead/tie/behind + win-one/tie-both fractions
    /// (obs v2 P1; k=2 exhaustive, same fused pass). All-zero for NLH.
    per_board_outcome: Array2<f32>,
    /// (n, 2) k=2 guaranteed-pot-share bounds [g_min, g_max] (v7 DUAL-4;
    /// same fused pass, dims 20/21). All-zero for NLH / inactive states.
    share_bounds: Array2<f32>,
    /// (n, s) engine acted_this_street bits (v7 STK-1 pending-set input).
    acted_this_street: Array2<bool>,
    /// (n, 8) v7 hero/board engine dims [boat_a, boat_b, improve_a,
    /// improve_b, combos_a, combos_b, mask_a, mask_b] (BRD-7 / BRD-12 /
    /// DUAL-2; see GameState::hero_board_v3). All-zero for NLH.
    hero_board_v3: Array2<u8>,
    /// Blind seats (-1 when the variant has none). NLH batch encoder input.
    sb_seat: Array1<i8>,
    bb_seat: Array1<i8>,
    /// NLH 3-dim [opp_ahead, tied, opp_behind] exhaustive fractions;
    /// all-zero rows for PLO variants (mirrors `nlh_opp_outcome_fractions`'s
    /// variant guard on the serial path).
    nlh_opp_outcome: Array2<f32>,
}

impl PyBatchedEngine {
    /// Pack the full batch (identity index). Thin wrapper over the indexed
    /// core so the per-env packing logic lives in exactly one place.
    fn pack_observation(&self, n: usize, s: usize) -> PackedObservation {
        let idx: Vec<usize> = (0..n).collect();
        self.pack_observation_indexed(&idx, s)
    }

    /// Pack ONLY the envs listed in `idx` into compact (k = idx.len()) rows.
    /// Output row j is built from `self.states[idx[j]]`. Used by
    /// `observation_and_features_subset_batch` to re-pack just the envs that
    /// changed (e.g. the reset-terminal subset) without touching the rest.
    fn pack_observation_indexed(&self, idx: &[usize], s: usize) -> PackedObservation {
        let n = idx.len();
        let hole_w = self.config.variant.hole_count();
        let hist_cap = history_cap(self.config.variant);
        let is_nlh = matches!(self.config.variant, Variant::NlhSingle);
        let mut hero_hole = Array2::<u8>::from_elem((n, hole_w), 255u8);
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
        let mut history_seat = Array2::<i8>::from_elem((n, hist_cap), -1i8);
        let mut history_action = Array2::<i8>::from_elem((n, hist_cap), -1i8);
        let mut history_chips = Array2::<u64>::zeros((n, hist_cap));
        let mut history_street = Array2::<i8>::from_elem((n, hist_cap), -1i8);
        let mut history_len = Array1::<u8>::zeros(n);
        let mut opp_outcome_fractions = Array2::<f32>::zeros((n, 12));
        let mut per_board_outcome = Array2::<f32>::zeros((n, 8));
        let mut share_bounds = Array2::<f32>::zeros((n, 2));
        let mut acted_this_street = Array2::<bool>::default((n, s));
        let mut hero_board_v3 = Array2::<u8>::zeros((n, 8));
        let mut sb_seat = Array1::<i8>::from_elem(n, -1i8);
        let mut bb_seat = Array1::<i8>::from_elem(n, -1i8);
        let mut nlh_opp_outcome = Array2::<f32>::zeros((n, 3));

        // Compute the variant's opp-outcome block in parallel — this is
        // the expensive per-env work. PLO: k=2 exhaustive + k=3/k=4 MC
        // at `self.opp_outcome_mc` draws (12 dims). NLH: exhaustive
        // 2-card unseen sweep (3 dims); the other block stays zeros,
        // mirroring the serial `observation_dict` (both keys always
        // present, only the variant's own is populated). The remaining
        // per-env writes below are cheap memcpy and stay serial.
        if is_nlh {
            let nlh_fr_per_env: Vec<[f32; 3]> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut out = [0.0f32; 3];
                    if let Some(state) = self.states[idx[i]].as_ref() {
                        let fr = state.nlh_opp_outcome_fractions();
                        for j in 0..3 {
                            out[j] = fr[j];
                        }
                    }
                    out
                })
                .collect();
            for i in 0..n {
                for j in 0..3 {
                    nlh_opp_outcome[[i, j]] = nlh_fr_per_env[i][j];
                }
            }
        } else {
            let opp_outcome_mc = self.opp_outcome_mc;
            let states = &self.states;
            // Per-env deterministic outcome seed (cheap hash of hero + street +
            // both boards). The fused MC is a pure function of exactly those, so
            // it is constant across a street's actions; recompute only when the
            // seed changes (street advance / new hand) and reuse the cached
            // 20-dim result otherwise. Bit-exact vs always-recompute — pinned by
            // test_encoding_rust (cached batched == fresh serial). Non-actor /
            // <3-board rows have seed None and stay all-zeros, matching the
            // early return in `outcome_features_mc`.
            let seeds: Vec<Option<u64>> = (0..n)
                .into_par_iter()
                .map(|i| states[idx[i]].as_ref().and_then(|st| st.outcome_seed()))
                .collect();
            // Decide which envs need a fresh MC (seed changed, or never cached).
            let recompute: Vec<usize> = {
                let cache = self.outcome_cache.lock().unwrap();
                (0..n)
                    .filter(|&i| match (seeds[i], &cache[idx[i]]) {
                        (Some(sd), Some((csd, _))) => sd != *csd,
                        (Some(_), None) => true,
                        (None, _) => false,
                    })
                    .collect()
            };
            // Expensive fused pass — only for the changed envs (12 joint
            // fractions + 8 per-board dims, obs v2 P1).
            let fresh: Vec<(usize, [f32; 22])> = recompute
                .par_iter()
                .map(|&i| {
                    let mut out = [0.0f32; 22];
                    if let Some(state) = states[idx[i]].as_ref() {
                        let fr = state.outcome_features_mc(opp_outcome_mc);
                        out.copy_from_slice(&fr[..22]);
                    }
                    (i, out)
                })
                .collect();
            // Store fresh results, then assemble every env's vector from the
            // cache (unchanged envs reuse; None-seed envs stay all-zeros).
            let opp_fr_per_env: Vec<[f32; 22]> = {
                let mut cache = self.outcome_cache.lock().unwrap();
                for &(i, out) in &fresh {
                    if let Some(sd) = seeds[i] {
                        cache[idx[i]] = Some((sd, out));
                    }
                }
                (0..n)
                    .map(|i| match (seeds[i], &cache[idx[i]]) {
                        (Some(_), Some((_, out))) => *out,
                        _ => [0.0f32; 22],
                    })
                    .collect()
            };
            for i in 0..n {
                for j in 0..12 {
                    opp_outcome_fractions[[i, j]] = opp_fr_per_env[i][j];
                }
                for j in 0..8 {
                    per_board_outcome[[i, j]] = opp_fr_per_env[i][12 + j];
                }
                for j in 0..2 {
                    share_bounds[[i, j]] = opp_fr_per_env[i][20 + j];
                }
            }
        }

        // Per-env packing in parallel: each row pulls scalars + s seat
        // fields + up to HISTORY_CAP history entries from its state and
        // writes directly into the output ndarrays via raw pointers.
        // Each iteration owns a disjoint row slice (row stride = s for
        // seat fields, HISTORY_CAP for history, 5 for cards), so the
        // writes never alias. No intermediate Vec allocations.
        struct OutPtrs {
            hero_hole: *mut u8,
            board_a: *mut u8,
            board_b: *mut u8,
            board_a_len: *mut u8,
            board_b_len: *mut u8,
            street: *mut u8,
            pot: *mut u64,
            stacks: *mut u64,
            folded: *mut bool,
            all_in: *mut bool,
            bet_to_call: *mut u64,
            street_commit: *mut u64,
            total_commit: *mut u64,
            min_bet: *mut u64,
            max_bet: *mut u64,
            min_raise: *mut u64,
            max_raise: *mut u64,
            eff_stack_cap: *mut u64,
            actor: *mut i8,
            button: *mut u8,
            last_aggressor: *mut i8,
            history_seat: *mut i8,
            history_action: *mut i8,
            history_chips: *mut u64,
            history_street: *mut i8,
            history_len: *mut u8,
            sb_seat: *mut i8,
            bb_seat: *mut i8,
            acted_this_street: *mut bool,
            hero_board_v3: *mut u8,
        }
        unsafe impl Send for OutPtrs {}
        unsafe impl Sync for OutPtrs {}

        let ptrs = OutPtrs {
            hero_hole: hero_hole.as_mut_ptr(),
            board_a: board_a.as_mut_ptr(),
            board_b: board_b.as_mut_ptr(),
            board_a_len: board_a_len.as_mut_ptr(),
            board_b_len: board_b_len.as_mut_ptr(),
            street: street.as_mut_ptr(),
            pot: pot.as_mut_ptr(),
            stacks: stacks.as_mut_ptr(),
            folded: folded.as_mut_ptr(),
            all_in: all_in.as_mut_ptr(),
            bet_to_call: bet_to_call.as_mut_ptr(),
            street_commit: street_commit.as_mut_ptr(),
            total_commit: total_commit.as_mut_ptr(),
            min_bet: min_bet.as_mut_ptr(),
            max_bet: max_bet.as_mut_ptr(),
            min_raise: min_raise.as_mut_ptr(),
            max_raise: max_raise.as_mut_ptr(),
            eff_stack_cap: eff_stack_cap.as_mut_ptr(),
            actor: actor.as_mut_ptr(),
            button: button.as_mut_ptr(),
            last_aggressor: last_aggressor.as_mut_ptr(),
            history_seat: history_seat.as_mut_ptr(),
            history_action: history_action.as_mut_ptr(),
            history_chips: history_chips.as_mut_ptr(),
            history_street: history_street.as_mut_ptr(),
            history_len: history_len.as_mut_ptr(),
            sb_seat: sb_seat.as_mut_ptr(),
            bb_seat: bb_seat.as_mut_ptr(),
            acted_this_street: acted_this_street.as_mut_ptr(),
            hero_board_v3: hero_board_v3.as_mut_ptr(),
        };

        (0..n).into_par_iter().for_each(|i| {
            // Force the closure to capture `ptrs` as a whole binding
            // rather than as disjoint fields. Otherwise Rust 2021's
            // disjoint capture rules pick up each `*mut T` field
            // individually, which is `!Sync` despite the unsafe Sync
            // impl on the wrapping struct.
            let ptrs = &ptrs;
            let state = match self.states[idx[i]].as_ref() {
                Some(state) => state,
                None => return,
            };
            // SAFETY: every write below indexes into a disjoint slice
            // of its target array (row i for 1D arrays; row stride
            // {5, s, hist_cap} for 2D arrays). No two parallel
            // iterations touch the same byte. All output buffers are
            // C-contiguous (default ndarray layout).
            unsafe {
                *ptrs.street.add(i) = state.street.index() as u8;
                *ptrs.pot.add(i) = state.pot;
                *ptrs.bet_to_call.add(i) = state.bet_to_call;
                *ptrs.min_bet.add(i) = state.min_bet_total();
                *ptrs.max_bet.add(i) = state.max_bet_total();
                *ptrs.min_raise.add(i) = state.min_raise_chips();
                *ptrs.max_raise.add(i) = state.max_raise_chips();
                *ptrs.button.add(i) = state.button as u8;
                *ptrs.last_aggressor.add(i) =
                    state.last_aggressor.map(|s| s as i8).unwrap_or(-1);
                *ptrs.sb_seat.add(i) = state.sb_seat.map(|x| x as i8).unwrap_or(-1);
                *ptrs.bb_seat.add(i) = state.bb_seat.map(|x| x as i8).unwrap_or(-1);

                if let Some(a) = state.current_actor() {
                    *ptrs.actor.add(i) = a as i8;
                    let hole_base = i * hole_w;
                    for (j, c) in state.hole_cards[a].iter().enumerate() {
                        *ptrs.hero_hole.add(hole_base + j) = c.index();
                    }
                }

                let la = state.board_a.len().min(5);
                let lb = state.board_b.len().min(5);
                *ptrs.board_a_len.add(i) = la as u8;
                *ptrs.board_b_len.add(i) = lb as u8;
                let ba_base = i * 5;
                for j in 0..la {
                    *ptrs.board_a.add(ba_base + j) = state.board_a[j].index();
                }
                for j in 0..lb {
                    *ptrs.board_b.add(ba_base + j) = state.board_b[j].index();
                }

                let seat_base = i * s;
                for k in 0..s {
                    *ptrs.stacks.add(seat_base + k) = state.stacks[k];
                    *ptrs.folded.add(seat_base + k) = state.folded[k];
                    *ptrs.all_in.add(seat_base + k) = state.all_in[k];
                    *ptrs.street_commit.add(seat_base + k) = state.street_commit[k];
                    *ptrs.total_commit.add(seat_base + k) = state.total_commit[k];
                    *ptrs.eff_stack_cap.add(seat_base + k) =
                        state.eff_stack_cap_at_hand_start[k];
                    *ptrs.acted_this_street.add(seat_base + k) =
                        state.acted_this_street[k];
                }

                // v7 hero/board engine dims (BRD-7/BRD-12/DUAL-2): the
                // improve/boat scans are the priciest per-env packing work;
                // they run inside this par_iter. Self-guarded to zeros for
                // NLH / no-actor / short boards.
                let hbv = state.hero_board_v3();
                let hb_base = i * 8;
                for (j, v) in hbv.iter().enumerate() {
                    *ptrs.hero_board_v3.add(hb_base + j) = *v;
                }

                let hist_len = state.history.len();
                let start = hist_len.saturating_sub(hist_cap);
                let kept = hist_len - start;
                *ptrs.history_len.add(i) = kept as u8;
                let hist_base = i * hist_cap;
                for (slot, rec) in state.history[start..].iter().enumerate() {
                    *ptrs.history_seat.add(hist_base + slot) = rec.seat as i8;
                    *ptrs.history_action.add(hist_base + slot) = rec.action.index() as i8;
                    *ptrs.history_chips.add(hist_base + slot) = rec.chips;
                    *ptrs.history_street.add(hist_base + slot) = rec.street.index() as i8;
                }
            }
        });

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
            per_board_outcome,
            share_bounds,
            acted_this_street,
            hero_board_v3,
            sb_seat,
            bb_seat,
            nlh_opp_outcome,
        }
    }

    /// Full Rust port of `encode_observation_batch` (python/plo5bp/encoding.py).
    /// Packs the indexed envs, computes legal masks + hero categories, then
    /// builds the finished (k, OBS_DIM) f32 observation in one rayon pass —
    /// collapsing pack + 4 feature FFI calls + the numpy assembly into a
    /// single FFI crossing. Returns a dict with `obs` plus the aux fields the
    /// rollout reads (actor, legal_mask, min/max_raise, total_commit,
    /// bet_to_call, street_commit, street). Bit-exact with the numpy encoder
    /// (asserted by tests/python/test_encoding_batch.py); the feature blocks
    /// reuse the same inner fns the numpy path's Rust helpers call.
    fn encode_indexed<'py>(
        &self,
        py: Python<'py>,
        idx: &[usize],
    ) -> PyResult<Bound<'py, PyDict>> {
        // The Rust encoder implements the 991-dim PLO layout only
        // (obs_layout + encode_obs_row assume dual boards, the 12-dim
        // opp-outcome block, and 32 history slots). NLH batches encode
        // via the numpy `encode_observation_batch_nlh` path.
        if matches!(self.config.variant, Variant::NlhSingle) {
            return Err(PyRuntimeError::new_err(
                "observation_encoded_batch is PLO-only; NLH uses the \
                 numpy batch encoder (observation_and_features_batch)",
            ));
        }
        let n = idx.len();
        let s = self.config.num_seats;
        let bb = self.config.bb;
        let starting = self.config.starting_stacks.clone();

        let (obs_vec, packed, legal_mask) = py.allow_threads(|| {
            let packed = self.pack_observation_indexed(idx, s);

            // Legal mask + hero categories (serial; cheap per env). Mirrors
            // observation_and_features_batch's loop.
            let mut legal_mask = Array2::<bool>::default((n, NUM_ACTIONS));
            let mut cat_a = vec![0u8; n];
            let mut cat_b = vec![0u8; n];
            for j in 0..n {
                let state = match self.states[idx[j]].as_ref() {
                    Some(st) => st,
                    None => continue,
                };
                if !state.is_terminal() {
                    let mask = state.legal_action_mask();
                    for t in 0..NUM_ACTIONS {
                        legal_mask[[j, t]] = mask[t];
                    }
                }
                if let Some(a) = state.current_actor() {
                    cat_a[j] = state.hero_category(a, 0);
                    cat_b[j] = state.hero_category(a, 1);
                }
            }

            // Per-row encode in parallel. Each row is a disjoint OBS_DIM slice.
            let inv_bb = 1.0f64 / (bb as f64);
            let mut obs_vec = vec![0f32; n * obs_layout::OBS_DIM];
            obs_vec
                .par_chunks_exact_mut(obs_layout::OBS_DIM)
                .enumerate()
                .for_each(|(j, row)| {
                    encode_obs_row(&packed, j, s, cat_a[j], cat_b[j], inv_bb, &starting, row);
                });

            (obs_vec, packed, legal_mask)
        });

        let obs_arr = Array2::from_shape_vec((n, obs_layout::OBS_DIM), obs_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let d = PyDict::new(py);
        d.set_item("obs", obs_arr.into_pyarray(py))?;
        d.set_item("actor", packed.actor.into_pyarray(py))?;
        d.set_item("legal_mask", legal_mask.into_pyarray(py))?;
        d.set_item("min_raise", packed.min_raise.into_pyarray(py))?;
        d.set_item("max_raise", packed.max_raise.into_pyarray(py))?;
        d.set_item("total_commit", packed.total_commit.into_pyarray(py))?;
        d.set_item("bet_to_call", packed.bet_to_call.into_pyarray(py))?;
        d.set_item("street_commit", packed.street_commit.into_pyarray(py))?;
        d.set_item("street", packed.street.into_pyarray(py))?;
        d.set_item("pot", packed.pot.into_pyarray(py))?;
        Ok(d)
    }
}

// =============================================================================
// Observation layout (single source of truth; mirrors the offset constants in
// python/plo5bp/encoding.py lines 101-179). A wrong value here silently
// misplaces a whole feature block, so keep in lockstep with the Python side.
// =============================================================================
mod obs_layout {
    pub const OBS_DIM: usize = 1020;
    pub const HOLE_OFF: usize = 0;
    pub const BOARD_A_OFF: usize = 52;
    pub const BOARD_B_OFF: usize = 104;
    pub const STREET_OFF: usize = 156;
    pub const ACTIVE_OFF: usize = 160;
    pub const ALLIN_OFF: usize = 168;
    pub const STACKS_OFF: usize = 176;
    pub const SCALARS_OFF: usize = 184;
    pub const REL_POS_OFF: usize = 188;
    pub const HISTORY_OFF: usize = 196;
    pub const HISTORY_DEPTH: usize = 32;
    pub const HISTORY_SLOT_DIM: usize = 18;
    pub const HISTORY_SEAT_OFF_REL: usize = 0;
    pub const HISTORY_GATE_OFF_REL: usize = 8;
    pub const HISTORY_STREET_OFF_REL: usize = 12;
    pub const HISTORY_CHIPS_OFF_REL: usize = 16;
    pub const HISTORY_FRAC_OFF_REL: usize = 17;
    pub const NUM_STREET_ONEHOT: usize = 4;
    pub const NUM_CATEGORIES: usize = 9;
    pub const SPR_OFF: usize = 772;
    pub const POT_ODDS_OFF: usize = 780;
    pub const CAT_A_OFF: usize = 781;
    pub const CAT_B_OFF: usize = 790;
    pub const DRAW_A_OFF: usize = 799;
    pub const DRAW_B_OFF: usize = 801;
    pub const PAIR_COUNT_A_OFF: usize = 803;
    pub const PAIR_COUNT_B_OFF: usize = 808;
    pub const BOARD_STRUCT_A_OFF: usize = 813;
    pub const BOARD_STRUCT_B_OFF: usize = 817;
    pub const HERO_RANK_HIST_OFF: usize = 821;
    pub const FLUSH_NUT_DIST_A_OFF: usize = 834;
    pub const FLUSH_NUT_DIST_B_OFF: usize = 872;
    pub const SEAT_EXISTS_OFF: usize = 910;
    pub const TOTAL_COMMIT_OFF: usize = 918;
    pub const STREET_COMMIT_OFF: usize = 926;
    pub const LAST_AGGRESSOR_OFF: usize = 934;
    pub const HERO_BTN_DIST_OFF: usize = 942;
    pub const SHARED_RANKS_OFF: usize = 950;
    pub const FLUSH_MADE_BOTH_OFF: usize = 963;
    pub const FLUSH_DRAW_BOTH_OFF: usize = 967;
    pub const FLUSH_MIXED_OFF: usize = 971;
    pub const STRAIGHT_MADE_BOTH_OFF: usize = 975;
    pub const STRAIGHT_DRAW_BOTH_OFF: usize = 976;
    pub const STRAIGHT_MIXED_OFF: usize = 977;
    pub const OPP_OUTCOME_OFF: usize = 978;
    pub const OPP_OUTCOME_DIM: usize = 12;
    pub const BET_PCT_POT_OFF: usize = 990;
    // obs v2 tail (V5_DESIGN.md §3.2, dims 991..1020) — a pure append after the
    // 991-dim v1 core. Offsets are byte-identical to the numpy encoder's tail
    // (_PER_BOARD_OUTCOME_OFF etc. in python/plo5bp/encoding.py).
    pub const PER_BOARD_OUTCOME_OFF: usize = 991; // 8: hero ahead/tie/behind per board
    pub const BLOCKER_A_OFF: usize = 999; // 4: unconditional blockers-to-nuts, board A
    pub const BLOCKER_B_OFF: usize = 1003; // 4: unconditional blockers-to-nuts, board B
    pub const EFF_PRICE_OFF: usize = 1007; // 5: eff price + commit frac + log1p money
    pub const SPR_LOG_OFF: usize = 1012; // 8: log1p effective SPR, unclipped
}

/// Encode one env's observation into `out` (length OBS_DIM, pre-zeroed). A
/// bit-exact per-env port of the scalar `encode_observation`
/// (python/plo5bp/encoding.py:652) — the ground-truth reference the numpy
/// batch encoder is validated against. Reads row `j` of the packed arrays.
/// Terminal rows (actor < 0) are left all-zero, matching the scalar early
/// return.
///
/// Bit-exactness discipline: all scalar arithmetic is done in f64 and cast to
/// f32 ONLY at the store (`(x as f64 * inv_bb) as f32`), matching the numpy
/// path which keeps `inv_bb` in f64 and casts on assignment. Hero rotation
/// uses non-negative modulo to match numpy/Python `%`.
#[allow(clippy::too_many_arguments)]
fn encode_obs_row(
    packed: &PackedObservation,
    j: usize,
    num_seats: usize,
    cat_a: u8,
    cat_b: u8,
    inv_bb: f64,
    starting: &[u64],
    out: &mut [f32],
) {
    use obs_layout::*;

    let hero_i = packed.actor[j];
    if hero_i < 0 {
        return; // terminal env -> all-zero row
    }
    let hero = hero_i as usize;
    let ns_i = num_seats as i64;
    let rel = |x: i64| -> usize { (x - hero as i64).rem_euclid(ns_i) as usize };

    // Row slices for cards.
    let hole_view = packed.hero_hole.row(j);
    let hole_slice = hole_view.as_slice().unwrap();
    let ba_view = packed.board_a.row(j);
    let ba_slice = ba_view.as_slice().unwrap();
    let bb_view = packed.board_b.row(j);
    let bb_slice = bb_view.as_slice().unwrap();

    // --- Card multi-hots (hole / board A / board B). ---
    // Hole width is variant-dependent (PLO4=4, PLO5=5, PLO6=6), so the
    // hole loop is driven by `hole_slice.len()`; boards are always 5
    // wide and stay `0..5`. Do NOT merge these — a `0..5` hole loop
    // reads out of bounds on PLO4 and silently drops the 6th card on
    // PLO6 (CLAUDE.md variant-encoding gotcha).
    for &c in hole_slice {
        if c < 52 {
            out[HOLE_OFF + c as usize] = 1.0;
        }
    }
    for slot in 0..5 {
        let ca = ba_slice[slot];
        if ca < 52 {
            out[BOARD_A_OFF + ca as usize] = 1.0;
        }
        let cb = bb_slice[slot];
        if cb < 52 {
            out[BOARD_B_OFF + cb as usize] = 1.0;
        }
    }

    // --- Street one-hot. ---
    let street = packed.street[j] as usize;
    if street < NUM_STREET_ONEHOT {
        out[STREET_OFF + street] = 1.0;
    }

    // --- Effective stack per seat (dead-chips chain), absolute seat index. ---
    // dead = max(0, starting - eff_cap); eff = max(0, stacks - dead). All f64;
    // chip magnitudes are exact integers in f64 range.
    let mut eff_per_seat = [0f64; 8];
    for seat in 0..num_seats {
        let st = starting[seat] as f64;
        let ec = packed.eff_stack_cap[[j, seat]] as f64;
        let dead = (st - ec).max(0.0);
        let stk = packed.stacks[[j, seat]] as f64;
        eff_per_seat[seat] = (stk - dead).max(0.0);
    }

    // --- Active / all-in / stacks (hero-rotated). ---
    for k in 0..num_seats {
        let seat = (hero + k) % num_seats;
        if !packed.folded[[j, seat]] {
            out[ACTIVE_OFF + k] = 1.0;
        }
        if packed.all_in[[j, seat]] {
            out[ALLIN_OFF + k] = 1.0;
        }
        out[STACKS_OFF + k] = (eff_per_seat[seat] * inv_bb) as f32;
    }

    // --- Scalars: pot, bet_to_call, min_bet, max_bet (all / bb). ---
    let pot = packed.pot[j] as f64;
    let btc = packed.bet_to_call[j] as f64;
    out[SCALARS_OFF] = (pot * inv_bb) as f32;
    out[SCALARS_OFF + 1] = (btc * inv_bb) as f32;
    out[SCALARS_OFF + 2] = (packed.min_bet[j] as f64 * inv_bb) as f32;
    out[SCALARS_OFF + 3] = (packed.max_bet[j] as f64 * inv_bb) as f32;

    // --- Relative position one-hot: actor is always slot 0 (hero == actor). ---
    out[REL_POS_OFF] = 1.0;

    // --- History (oldest-first, already truncated to last HISTORY_DEPTH). ---
    let hlen = (packed.history_len[j] as usize).min(HISTORY_DEPTH);
    // Pot before each visible action: history chips are per-action
    // DELTAS (antes never recorded), so pot_before(slot) = current_pot −
    // Σ chips of visible slots ≥ slot. Valid under truncation — dropped
    // actions all precede the window. Integer math mirrors numpy/scalar.
    let pot_now_chips = packed.pot[j] as i64;
    let mut pot_before = [0i64; HISTORY_DEPTH];
    let mut chips_suffix: i64 = 0;
    for slot in (0..hlen).rev() {
        chips_suffix += packed.history_chips[[j, slot]] as i64;
        pot_before[slot] = pot_now_chips - chips_suffix;
    }
    for slot in 0..hlen {
        let base = HISTORY_OFF + slot * HISTORY_SLOT_DIM;
        let hseat = packed.history_seat[[j, slot]] as i64;
        out[base + HISTORY_SEAT_OFF_REL + rel(hseat)] = 1.0;
        let action = packed.history_action[[j, slot]];
        let chips = packed.history_chips[[j, slot]];
        // Gate (matches _gate_from_action): FOLD(0)->0; CHECK_CALL(1) & chips==0
        // ->Check(1), &chips>0 ->Call(2); anything else ->Raise(3).
        let gate = if action == 0 {
            0
        } else if action == 1 {
            if chips == 0 {
                1
            } else {
                2
            }
        } else {
            3
        };
        out[base + HISTORY_GATE_OFF_REL + gate] = 1.0;
        let s_idx = packed.history_street[[j, slot]];
        if s_idx >= 0 && (s_idx as usize) < NUM_STREET_ONEHOT {
            out[base + HISTORY_STREET_OFF_REL + s_idx as usize] = 1.0;
        }
        out[base + HISTORY_CHIPS_OFF_REL] = (chips as f64 * inv_bb) as f32;
        let frac = chips as f64 / pot_before[slot].max(1) as f64;
        out[base + HISTORY_FRAC_OFF_REL] = frac.clamp(0.0, 2.0) as f32;
    }

    // --- SPR per seat (hero-rotated), clip [0, 4]. ---
    let pot_safe = pot.max(1.0);
    for k in 0..num_seats {
        let seat = (hero + k) % num_seats;
        let spr = eff_per_seat[seat] / pot_safe;
        out[SPR_OFF + k] = spr.max(0.0).min(4.0) as f32;
    }

    // --- Pot odds + bet-faced-as-fraction-of-pot. ---
    let hero_street_commit = packed.street_commit[[j, hero]] as f64;
    let to_call = (btc - hero_street_commit).max(0.0);
    if to_call > 0.0 {
        out[POT_ODDS_OFF] = (to_call / (pot + to_call)) as f32;
        let pot_before_bet = (pot - to_call).max(1.0);
        out[BET_PCT_POT_OFF] = (to_call / pot_before_bet).min(4.0) as f32;
    }

    // --- Hand-category one-hots. ---
    if (cat_a as usize) < NUM_CATEGORIES {
        out[CAT_A_OFF + cat_a as usize] = 1.0;
    }
    if (cat_b as usize) < NUM_CATEGORIES {
        out[CAT_B_OFF + cat_b as usize] = 1.0;
    }

    // --- Hole-derived summaries reused by feature blocks. ---
    let mut hole_suit_count = [0u8; 4];
    let mut hole_rank_mask: u16 = 0;
    let mut hole_rank_counts = [0u8; 13];
    for &c in hole_slice {
        if c < 52 {
            hole_suit_count[(c & 3) as usize] += 1;
            hole_rank_mask |= 1u16 << (c >> 2);
            hole_rank_counts[(c >> 2) as usize] += 1;
        }
    }

    // --- Draw flags (per board). ---
    let (fa, sa) = draw_flags_one_board(&hole_suit_count, hole_rank_mask, ba_slice);
    let (fb, sb) = draw_flags_one_board(&hole_suit_count, hole_rank_mask, bb_slice);
    out[DRAW_A_OFF] = fa;
    out[DRAW_A_OFF + 1] = sa;
    out[DRAW_B_OFF] = fb;
    out[DRAW_B_OFF + 1] = sb;

    // --- Pair-with-board counts + board pair structure. ---
    let mut counts_a = [0f32; 5];
    let mut struct_a = [0f32; 4];
    let mut counts_b = [0f32; 5];
    let mut struct_b = [0f32; 4];
    pair_features_one_board(&hole_rank_counts, ba_slice, &mut counts_a, &mut struct_a);
    pair_features_one_board(&hole_rank_counts, bb_slice, &mut counts_b, &mut struct_b);
    for i in 0..5 {
        out[PAIR_COUNT_A_OFF + i] = counts_a[i];
        out[PAIR_COUNT_B_OFF + i] = counts_b[i];
    }
    for i in 0..4 {
        out[BOARD_STRUCT_A_OFF + i] = struct_a[i];
        out[BOARD_STRUCT_B_OFF + i] = struct_b[i];
    }

    // --- Hero rank histogram (board-agnostic). ---
    for &c in hole_slice {
        if c < 52 {
            out[HERO_RANK_HIST_OFF + (c >> 2) as usize] += 1.0;
        }
    }

    // --- Straight / flush / SF block: needs global (hole+A+B) visibility. ---
    let mut seen_per_suit = [0u16; 4];
    for slice in [hole_slice, ba_slice, bb_slice] {
        for &c in slice {
            if c < 52 {
                seen_per_suit[(c & 3) as usize] |= 1u16 << (c >> 2);
            }
        }
    }
    let mut vct = [0u8; 13];
    for r in 0..13 {
        let mut cnt = 0u8;
        for s in seen_per_suit.iter() {
            if (s >> r) & 1 == 1 {
                cnt += 1;
            }
        }
        vct[r] = cnt;
    }
    let mut unseen_suit = [0u16; 4];
    let mut visible_per_suit = [0u8; 4];
    for s in 0..4 {
        unseen_suit[s] = !seen_per_suit[s] & 0x1FFF;
        visible_per_suit[s] = seen_per_suit[s].count_ones() as u8;
    }
    let (h_rm, h_rs, h_sc, h_mps) = sf_derive_card_state(hole_slice);
    sf_compute_board(
        h_rm, &h_rs, &h_sc, &h_mps, ba_slice, &vct, &unseen_suit, &visible_per_suit,
        &mut out[FLUSH_NUT_DIST_A_OFF..FLUSH_NUT_DIST_A_OFF + 38],
    );
    sf_compute_board(
        h_rm, &h_rs, &h_sc, &h_mps, bb_slice, &vct, &unseen_suit, &visible_per_suit,
        &mut out[FLUSH_NUT_DIST_B_OFF..FLUSH_NUT_DIST_B_OFF + 38],
    );

    // --- Structural seat-exists mask. ---
    for k in 0..num_seats {
        out[SEAT_EXISTS_OFF + k] = 1.0;
    }

    // --- Per-seat commits (hero-rotated). ---
    for k in 0..num_seats {
        let seat = (hero + k) % num_seats;
        out[TOTAL_COMMIT_OFF + k] = (packed.total_commit[[j, seat]] as f64 * inv_bb) as f32;
        out[STREET_COMMIT_OFF + k] = (packed.street_commit[[j, seat]] as f64 * inv_bb) as f32;
    }

    // --- Last aggressor (hero-relative one-hot). ---
    let la = packed.last_aggressor[j];
    if la >= 0 && (la as usize) < num_seats {
        out[LAST_AGGRESSOR_OFF + rel(la as i64)] = 1.0;
    }

    // --- Hero distance to button. ---
    out[HERO_BTN_DIST_OFF + rel(packed.button[j] as i64)] = 1.0;

    // --- Cross-board interactions. ---
    let mut ba_rank_mask: u16 = 0;
    let mut bb_rank_mask: u16 = 0;
    let mut ba_suit = [0u8; 4];
    let mut bb_suit = [0u8; 4];
    for &c in ba_slice {
        if c < 52 {
            ba_rank_mask |= 1u16 << (c >> 2);
            ba_suit[(c & 3) as usize] += 1;
        }
    }
    for &c in bb_slice {
        if c < 52 {
            bb_rank_mask |= 1u16 << (c >> 2);
            bb_suit[(c & 3) as usize] += 1;
        }
    }
    let shared = ba_rank_mask & bb_rank_mask;
    for r in 0..13 {
        if (shared >> r) & 1 == 1 {
            out[SHARED_RANKS_OFF + r] = 1.0;
        }
    }
    for s in 0..4 {
        if hole_suit_count[s] < 2 {
            continue;
        }
        let a3 = ba_suit[s] >= 3;
        let b3 = bb_suit[s] >= 3;
        let a2 = ba_suit[s] == 2;
        let b2 = bb_suit[s] == 2;
        if a3 && b3 {
            out[FLUSH_MADE_BOTH_OFF + s] = 1.0;
        } else if a2 && b2 {
            out[FLUSH_DRAW_BOTH_OFF + s] = 1.0;
        } else if (a3 && b2) || (a2 && b3) {
            out[FLUSH_MIXED_OFF + s] = 1.0;
        }
    }
    let boards_visible = ba_rank_mask != 0 && bb_rank_mask != 0;
    let (cm, cd, cx) =
        cross_board_straight_per_env(hole_rank_mask, ba_rank_mask, bb_rank_mask, boards_visible);
    out[STRAIGHT_MADE_BOTH_OFF] = cm;
    out[STRAIGHT_DRAW_BOTH_OFF] = cd;
    out[STRAIGHT_MIXED_OFF] = cx;

    // --- Opp-outcome fractions (already f32; live rows only). ---
    for t in 0..OPP_OUTCOME_DIM {
        out[OPP_OUTCOME_OFF + t] = packed.opp_outcome_fractions[[j, t]];
    }

    // ===== obs v2 tail (V5_DESIGN.md §3.2, dims 991..1020) =====
    // Per-board hero ahead/tie/behind + win-one/tie-both — already f32 from the
    // fused MC pass (packed by pack_observation_indexed); all-zero preflop/terminal.
    for m in 0..8 {
        out[PER_BOARD_OUTCOME_OFF + m] = packed.per_board_outcome[[j, m]];
    }
    // Unconditional blockers-to-nuts per board (4 dims each).
    blocker_features_one_board(hole_slice, ba_slice, &mut out[BLOCKER_A_OFF..BLOCKER_A_OFF + 4]);
    blocker_features_one_board(hole_slice, bb_slice, &mut out[BLOCKER_B_OFF..BLOCKER_B_OFF + 4]);
    // Effective price: to_call capped by hero's EFFECTIVE remaining stack, plus
    // commitment fraction and log1p money companions. f64 throughout, cast on store.
    let hero_stack = eff_per_seat[hero];
    let eff_to_call = to_call.min(hero_stack);
    if eff_to_call > 0.0 {
        out[EFF_PRICE_OFF] = (eff_to_call / (pot + eff_to_call)) as f32;
    }
    if to_call > 0.0 && to_call >= hero_stack {
        out[EFF_PRICE_OFF + 1] = 1.0;
    }
    let hero_commit = packed.total_commit[[j, hero]] as f64;
    let commit_denom = hero_commit + hero_stack;
    if commit_denom > 0.0 {
        out[EFF_PRICE_OFF + 2] = (hero_commit / commit_denom) as f32;
    }
    out[EFF_PRICE_OFF + 3] = (eff_to_call * inv_bb).ln_1p() as f32;
    out[EFF_PRICE_OFF + 4] = (pot * inv_bb).ln_1p() as f32;
    // log1p effective SPR, unclipped (hero-rotated).
    for k in 0..num_seats {
        let seat = (hero + k) % num_seats;
        out[SPR_LOG_OFF + k] = (eff_per_seat[seat] / pot_safe).ln_1p() as f32;
    }
}

/// Unconditional blockers-to-nuts for ONE board (obs v2 P3, 4 dims). Bit-exact
/// port of `_blocker_features` (python/plo5bp/encoding.py): flush-suit top-card
/// blocker + top-3 held, nut-straight window blockers, top board-pair blocker.
/// `out` is a 4-wide pre-zeroed slice. Divisions are done in f64 then cast to
/// f32 (matching numpy's `held / 3.0` → f32-array assignment).
fn blocker_features_one_board(hole: &[u8], board: &[u8], out: &mut [f32]) {
    let mut board_rank_counts = [0i32; 13];
    let mut board_suit_counts = [0i32; 4];
    let mut board_suit_ranks = [0u16; 4]; // rank bitmask per suit
    let mut nboard = 0usize;
    for &c in board {
        if c < 52 {
            nboard += 1;
            board_rank_counts[(c >> 2) as usize] += 1;
            board_suit_counts[(c & 3) as usize] += 1;
            board_suit_ranks[(c & 3) as usize] |= 1u16 << (c >> 2);
        }
    }
    if nboard < 3 {
        return;
    }
    let mut hero_rank_counts = [0i32; 13];
    let mut hero_cards: u64 = 0;
    for &c in hole {
        if c < 52 {
            hero_rank_counts[(c >> 2) as usize] += 1;
            hero_cards |= 1u64 << c;
        }
    }

    // Flush blockers: first suit with >= 3 board cards (two can't coexist on 5).
    for s in 0..4usize {
        if board_suit_counts[s] >= 3 {
            let mut missing: Vec<usize> = Vec::with_capacity(13);
            for r in (0..13usize).rev() {
                if board_suit_ranks[s] & (1u16 << r) == 0 {
                    missing.push(r);
                }
            }
            if let Some(&top) = missing.first() {
                if hero_cards & (1u64 << (top * 4 + s)) != 0 {
                    out[0] = 1.0;
                }
            }
            let held = missing
                .iter()
                .take(3)
                .filter(|&&r| hero_cards & (1u64 << (r * 4 + s)) != 0)
                .count();
            out[1] = (held as f64 / 3.0) as f32;
            break;
        }
    }

    // Nut-straight blockers: highest qualifying window (broadway-first scan).
    // Windows match _STRAIGHT_WINDOWS: slot 0 = wheel, slot 9 = broadway.
    const STRAIGHT_WINDOWS: [u16; 10] = [
        (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3), // wheel A2345
        (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4),
        (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5),
        (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6),
        (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7),
        (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8),
        (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9),
        (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10),
        (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11),
        (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | (1 << 12), // broadway TJQKA
    ];
    let mut board_rank_set: u16 = 0;
    for r in 0..13usize {
        if board_rank_counts[r] > 0 {
            board_rank_set |= 1u16 << r;
        }
    }
    for wi in (0..10usize).rev() {
        let w = STRAIGHT_WINDOWS[wi];
        if (w & board_rank_set).count_ones() >= 3 {
            let missing = w & !board_rank_set;
            let mut blockers = 0i32;
            for r in 0..13usize {
                if missing & (1u16 << r) != 0 {
                    blockers += hero_rank_counts[r];
                }
            }
            out[2] = (blockers.min(4) as f64 / 4.0) as f32;
            break;
        }
    }

    // Board-pair blockers: highest paired rank.
    for r in (0..13usize).rev() {
        if board_rank_counts[r] >= 2 {
            out[3] = (hero_rank_counts[r].min(2) as f64 / 2.0) as f32;
            break;
        }
    }
}

// =============================================================================
// Straight / flush / SF features (vectorized port of
// `_straight_flush_features_batch` in python/plo5bp/encoding.py).
// =============================================================================

const SF_RANK_MASK_13: u16 = 0x1FFF;

// 10 straight windows; each is a 13-bit mask over ranks. Window 0 is the
// wheel (A-2-3-4-5 = ranks {12,0,1,2,3}); window 9 is the broadway
// (T-J-Q-K-A = ranks {8,9,10,11,12}).
const SF_W_MASKS: [u16; 10] = [
    0x100F, 0x001F, 0x003E, 0x007C, 0x00F8, 0x01F0, 0x03E0, 0x07C0, 0x0F80, 0x1F00,
];

#[inline]
fn sf_ranks_above(h_max: i8) -> u16 {
    // 13-bit mask of ranks r where r > h_max.
    if h_max < 0 {
        SF_RANK_MASK_13
    } else if h_max >= 12 {
        0
    } else {
        let cutoff = (h_max + 1) as u32;
        SF_RANK_MASK_13 & !((1u16 << cutoff) - 1)
    }
}

#[inline]
fn sf_derive_card_state(row: &[u8]) -> (u16, [u16; 4], [u8; 4], [i8; 4]) {
    // Returns (rank_mask, rank_suit_per_suit, suit_count, max_rank_per_suit).
    let mut rank_mask: u16 = 0;
    let mut rank_suit: [u16; 4] = [0; 4];
    let mut suit_count: [u8; 4] = [0; 4];
    let mut max_per_suit: [i8; 4] = [-1, -1, -1, -1];
    for &c in row {
        if c < 52 {
            let r = (c >> 2) as usize; // 0..13
            let s = (c & 3) as usize;  // 0..4
            rank_mask |= 1u16 << r;
            rank_suit[s] |= 1u16 << r;
            suit_count[s] += 1;
            if (r as i8) > max_per_suit[s] {
                max_per_suit[s] = r as i8;
            }
        }
    }
    (rank_mask, rank_suit, suit_count, max_per_suit)
}

#[inline]
fn sf_compute_board(
    hole_rank_mask: u16,
    hole_rank_suit: &[u16; 4],
    hole_suit_count: &[u8; 4],
    hole_max_per_suit: &[i8; 4],
    board_row: &[u8],
    vct: &[u8; 13],
    unseen_suit: &[u16; 4],
    visible_per_suit: &[u8; 4],
    out: &mut [f32],
) {
    debug_assert!(out.len() >= 38);

    let (board_rm, board_rs, board_sc, _) = sf_derive_card_state(board_row);

    let mut makes_window: [bool; 10] = [false; 10];
    let mut straight_outs: [u32; 10] = [0; 10];
    let mut straight_possible: [u32; 10] = [0; 10];
    let mut sf_cand: [u16; 4] = [0; 4];

    for (w_i, &w_mask) in SF_W_MASKS.iter().enumerate() {
        // Suit-agnostic straight in this window.
        let b_w = board_rm & w_mask;
        let h_w = hole_rank_mask & w_mask;
        let l_mask = w_mask & !board_rm;
        let m_mask = l_mask & !hole_rank_mask;
        let n_b_w = b_w.count_ones();
        let n_h_w = h_w.count_ones();
        let n_l = l_mask.count_ones();
        let n_m = m_mask.count_ones();

        let makes = n_m == 0 && n_h_w >= 2 && n_l <= 2;
        makes_window[w_i] = makes;
        if n_b_w >= 3 {
            straight_possible[w_i] = 1;
        }

        let gate = !makes && n_h_w >= 2;
        let cond_3_0 = n_l == 3 && n_m == 0 && gate;
        let cond_m1 = n_m == 1 && (1..=3).contains(&n_l) && gate;
        if cond_3_0 {
            let mut total: u32 = 0;
            let mut bits = l_mask;
            while bits != 0 {
                let r = bits.trailing_zeros() as usize;
                total += 4u32 - vct[r] as u32;
                bits &= bits - 1;
            }
            straight_outs[w_i] = total;
        } else if cond_m1 {
            let r = m_mask.trailing_zeros() as usize;
            straight_outs[w_i] = 4u32 - vct[r] as u32;
        }

        // Suit-restricted (SF) per-suit candidates for this window.
        for s in 0..4 {
            let h_s = hole_rank_suit[s] & w_mask;
            let l_s = w_mask & !board_rs[s];
            let m_s = l_s & !hole_rank_suit[s];
            let n_h_s = h_s.count_ones();
            let n_l_s = l_s.count_ones();
            let n_m_s = m_s.count_ones();
            let already_s = n_m_s == 0 && n_h_s >= 2 && n_l_s <= 2;
            let gate_s = !already_s && n_h_s >= 2;
            let cond_3_0_s = n_l_s == 3 && n_m_s == 0 && gate_s;
            let cond_m1_s = n_m_s == 1 && (1..=3).contains(&n_l_s) && gate_s;
            if cond_3_0_s {
                sf_cand[s] |= l_s;
            } else if cond_m1_s {
                sf_cand[s] |= m_s;
            }
        }
    }

    // Straight nut distance: only counted when hero has a made straight.
    let mut h_max_straight: i32 = -1;
    for w in 0..10 {
        if makes_window[w] {
            h_max_straight = w as i32;
        }
    }
    let any_made_straight = h_max_straight >= 0;
    let mut straight_nut_dist: u32 = 0;
    if any_made_straight {
        for w in 0..10 {
            if (w as i32) > h_max_straight && straight_possible[w] != 0 {
                straight_nut_dist += 1;
            }
        }
    }

    // Flush features.
    let mut flush_possible: [u32; 4] = [0; 4];
    let mut flush_draw_mask: [bool; 4] = [false; 4];
    let mut flush_draw_outs: [u32; 4] = [0; 4];
    let mut nut_flush_draw_outs: [u32; 4] = [0; 4];
    for s in 0..4 {
        if board_sc[s] >= 3 {
            flush_possible[s] = 1;
        }
        let is_draw = hole_suit_count[s] >= 2 && board_sc[s] == 2;
        flush_draw_mask[s] = is_draw;
        if is_draw {
            flush_draw_outs[s] = 13u32 - visible_per_suit[s] as u32;
            let above_mask = sf_ranks_above(hole_max_per_suit[s]);
            let blockers = (unseen_suit[s] & above_mask).count_ones();
            nut_flush_draw_outs[s] = match blockers {
                0 => flush_draw_outs[s],
                1 => 1,
                _ => 0,
            };
        }
    }

    // Made-flush nut distance: first suit where hero has >=2 + board has >=3.
    // At most one suit per env can satisfy (each player holds 5 cards but the
    // numpy oracle takes argmax = first True so we preserve that ordering).
    let mut flush_nut_dist: u32 = 0;
    for s in 0..4 {
        if board_sc[s] >= 3 && hole_suit_count[s] >= 2 {
            let above_mask = sf_ranks_above(hole_max_per_suit[s]);
            flush_nut_dist = (unseen_suit[s] & above_mask).count_ones();
            break;
        }
    }

    // SF outs per suit (gated by flush draw on that suit).
    let mut sf_outs_per_suit: [u32; 4] = [0; 4];
    for s in 0..4 {
        if flush_draw_mask[s] {
            sf_outs_per_suit[s] = (sf_cand[s] & unseen_suit[s]).count_ones();
        }
    }

    out[0] = flush_nut_dist as f32;
    out[1] = straight_nut_dist as f32;
    for w in 0..10 {
        out[2 + w] = straight_outs[w] as f32;
        out[12 + w] = straight_possible[w] as f32;
    }
    for s in 0..4 {
        out[22 + s] = flush_possible[s] as f32;
        out[26 + s] = flush_draw_outs[s] as f32;
        out[30 + s] = nut_flush_draw_outs[s] as f32;
        out[34 + s] = sf_outs_per_suit[s] as f32;
    }
}

#[pyfunction]
pub fn straight_flush_features_batch<'py>(
    py: Python<'py>,
    hole: PyReadonlyArray2<'_, u8>,
    board_a: PyReadonlyArray2<'_, u8>,
    board_b: PyReadonlyArray2<'_, u8>,
    visible_count: PyReadonlyArray3<'_, i8>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let hole_v = hole.as_array();
    let ba_v = board_a.as_array();
    let bb_v = board_b.as_array();
    let vc_v = visible_count.as_array();

    let n = hole_v.shape()[0];
    let hole_w = if n > 0 { hole_v.shape()[1] } else { 5 };
    if !(4..=6).contains(&hole_w) || hole_v.shape() != [n, hole_w] {
        return Err(PyValueError::new_err(
            "hole shape must be (N, 4), (N, 5), or (N, 6)",
        ));
    }
    if ba_v.shape() != [n, 5] || bb_v.shape() != [n, 5] {
        return Err(PyValueError::new_err(
            "board_a and board_b shapes must be (N, 5) matching hole",
        ));
    }
    if vc_v.shape() != [n, 13, 4] {
        return Err(PyValueError::new_err(
            "visible_count shape must be (N, 13, 4)",
        ));
    }

    // Materialize owned copies so the parallel section runs GIL-free.
    let hole_owned = hole_v.to_owned();
    let ba_owned = ba_v.to_owned();
    let bb_owned = bb_v.to_owned();
    let vc_owned = vc_v.to_owned();

    let mut sf_a = vec![0f32; n * 38];
    let mut sf_b = vec![0f32; n * 38];

    py.allow_threads(|| {
        sf_a.par_chunks_exact_mut(38)
            .zip(sf_b.par_chunks_exact_mut(38))
            .enumerate()
            .for_each(|(i, (a_out, b_out))| {
                let hole_row = hole_owned.row(i);
                let ba_row = ba_owned.row(i);
                let bb_row = bb_owned.row(i);
                let hole_slice = hole_row.as_slice().unwrap();
                let ba_slice = ba_row.as_slice().unwrap();
                let bb_slice = bb_row.as_slice().unwrap();

                let (h_rm, h_rs, h_sc, h_mps) = sf_derive_card_state(hole_slice);

                // Per-env visibility summaries derived from vc_owned[i].
                let mut vct: [u8; 13] = [0; 13];
                let mut unseen_suit: [u16; 4] = [0; 4];
                let mut visible_per_suit: [u8; 4] = [0; 4];
                for r in 0..13 {
                    for s in 0..4 {
                        if vc_owned[[i, r, s]] == 0 {
                            unseen_suit[s] |= 1u16 << r;
                        } else {
                            vct[r] += 1;
                            visible_per_suit[s] += 1;
                        }
                    }
                }

                sf_compute_board(
                    h_rm, &h_rs, &h_sc, &h_mps,
                    ba_slice,
                    &vct, &unseen_suit, &visible_per_suit,
                    a_out,
                );
                sf_compute_board(
                    h_rm, &h_rs, &h_sc, &h_mps,
                    bb_slice,
                    &vct, &unseen_suit, &visible_per_suit,
                    b_out,
                );
            });
    });

    let sf_a_arr = Array2::from_shape_vec((n, 38), sf_a)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let sf_b_arr = Array2::from_shape_vec((n, 38), sf_b)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((sf_a_arr.into_pyarray(py), sf_b_arr.into_pyarray(py)))
}

#[cfg(test)]
mod sf_tests {
    use super::*;

    fn build_vc(hole: &[u8], boards: &[&[u8]]) -> [[i8; 4]; 13] {
        let mut vc = [[0i8; 4]; 13];
        for &c in hole.iter().chain(boards.iter().flat_map(|b| b.iter())) {
            if c < 52 {
                let r = (c >> 2) as usize;
                let s = (c & 3) as usize;
                vc[r][s] = 1;
            }
        }
        vc
    }

    fn summarize_vc(vc: &[[i8; 4]; 13]) -> ([u8; 13], [u16; 4], [u8; 4]) {
        let mut vct = [0u8; 13];
        let mut unseen = [0u16; 4];
        let mut vps = [0u8; 4];
        for r in 0..13 {
            for s in 0..4 {
                if vc[r][s] == 0 {
                    unseen[s] |= 1u16 << r;
                } else {
                    vct[r] += 1;
                    vps[s] += 1;
                }
            }
        }
        (vct, unseen, vps)
    }

    fn run_one(hole: &[u8], board_a: &[u8], board_b: &[u8]) -> ([f32; 38], [f32; 38]) {
        let vc = build_vc(hole, &[board_a, board_b]);
        let (vct, unseen, vps) = summarize_vc(&vc);
        let mut hole_arr = [255u8; 5];
        let mut ba_arr = [255u8; 5];
        let mut bb_arr = [255u8; 5];
        for (i, &c) in hole.iter().enumerate().take(5) { hole_arr[i] = c; }
        for (i, &c) in board_a.iter().enumerate().take(5) { ba_arr[i] = c; }
        for (i, &c) in board_b.iter().enumerate().take(5) { bb_arr[i] = c; }
        let (h_rm, h_rs, h_sc, h_mps) = sf_derive_card_state(&hole_arr);
        let mut out_a = [0f32; 38];
        let mut out_b = [0f32; 38];
        sf_compute_board(h_rm, &h_rs, &h_sc, &h_mps, &ba_arr, &vct, &unseen, &vps, &mut out_a);
        sf_compute_board(h_rm, &h_rs, &h_sc, &h_mps, &bb_arr, &vct, &unseen, &vps, &mut out_b);
        (out_a, out_b)
    }

    // Card encoding helper: rank * 4 + suit. Ranks 0..13 (2=0, A=12).
    fn card(r: u8, s: u8) -> u8 { r * 4 + s }

    #[test]
    fn empty_inputs_produce_zero_rows() {
        let (out_a, out_b) = run_one(&[], &[], &[]);
        assert!(out_a.iter().all(|&x| x == 0.0));
        assert!(out_b.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn ranks_above_edges() {
        assert_eq!(sf_ranks_above(-1), 0x1FFF);
        assert_eq!(sf_ranks_above(12), 0);
        // h_max = 8 -> ranks 9..12 set: bits 9,10,11,12 = 0x1E00
        assert_eq!(sf_ranks_above(8), 0x1E00);
        // h_max = 0 -> ranks 1..12 set: bits 1..12 = 0x1FFE
        assert_eq!(sf_ranks_above(0), 0x1FFE);
    }

    #[test]
    fn broadway_made_straight_no_higher_possible() {
        // Hole: Ac Kc (12,11 of suit 0). Board A: Q,J,T of suit 0. Board B empty.
        // This is also a made flush — but we're checking straight features here.
        // Straight window 9 (T-J-Q-K-A): hero has 2 (K,A), board has 3 (T,J,Q).
        // makes_window[9] = true. Higher windows: none. straight_nut_dist = 0.
        let hole = [card(12, 0), card(11, 0)];
        let ba = [card(10, 0), card(9, 0), card(8, 0)]; // T,J,Q
        let (out_a, _) = run_one(&hole, &ba, &[]);
        // index 1 = straight_nut_dist
        assert_eq!(out_a[1], 0.0);
        // index 12+9 = 21 = straight_possible[9]
        assert_eq!(out_a[21], 1.0);
    }

    #[test]
    fn flush_draw_with_nut_blockers_reported_separately() {
        // Hole: Th, 2h (suit 1). Board A: Ah, 5h, 9c.
        // Hole suit-1 count = 2. Board suit-1 count = 2 -> flush draw.
        // Highest hero heart = T (rank 8). Above-h_max unseen suit-1 cards:
        // J,Q,K,A of hearts. Board contains Ah (rank 12, suit 1). So visible
        // includes Ah; unseen above T are J,Q,K = 3 blockers.
        // flush_draw_outs = 13 - visible_per_suit[1] = 13 - (Th + 2h + Ah + 5h) = 13 - 4 = 9.
        // nut_flush_draw_outs: blockers = 3 -> 0.
        let hole = [card(8, 1), card(0, 1)]; // Th, 2h
        let ba = [card(12, 1), card(3, 1), card(7, 2)]; // Ah, 5h, 9c (rank 7 = 9, suit 2 = c)
        let (out_a, _) = run_one(&hole, &ba, &[]);
        // index 26+1 = 27 = flush_draw_outs[1]
        assert_eq!(out_a[27], 9.0);
        // index 30+1 = 31 = nut_flush_draw_outs[1]
        assert_eq!(out_a[31], 0.0);
    }

    #[test]
    fn made_flush_nut_distance_counts_higher_unseen() {
        // Hole: Kh, Qh. Board A: 2h, 5h, 9h (three of hearts).
        // hole suit-1 count = 2, board suit-1 count = 3 -> made flush.
        // h_max for suit 1 = K (rank 11). Above-K unseen suit-1 cards:
        // only Ah (rank 12). visible suit-1: Kh, Qh, 2h, 5h, 9h. Ah is unseen.
        // flush_nut_dist = 1.
        let hole = [card(11, 1), card(10, 1)]; // Kh, Qh
        let ba = [card(0, 1), card(3, 1), card(7, 1)]; // 2h, 5h, 9h
        let (out_a, _) = run_one(&hole, &ba, &[]);
        // index 0 = flush_nut_dist
        assert_eq!(out_a[0], 1.0);
        // flush_possible[1] at index 22+1=23 should be 1
        assert_eq!(out_a[23], 1.0);
    }
}

// =============================================================================
// Cross-board straight features (vectorized port of
// `_cross_board_straight_batch` in python/plo5bp/encoding.py).
// =============================================================================

const fn build_pair_bits_78() -> [u16; 78] {
    let mut out = [0u16; 78];
    let mut idx = 0;
    let mut r1: u16 = 0;
    while r1 < 13 {
        let mut r2: u16 = r1 + 1;
        while r2 < 13 {
            out[idx] = (1u16 << r1) | (1u16 << r2);
            idx += 1;
            r2 += 1;
        }
        r1 += 1;
    }
    out
}

static PAIR_BITS_78: [u16; 78] = build_pair_bits_78();

const fn build_pair_in_w() -> [u128; 10] {
    let mut out = [0u128; 10];
    let pairs = build_pair_bits_78();
    let mut w_idx = 0;
    while w_idx < 10 {
        let w = SF_W_MASKS[w_idx];
        let mut bits: u128 = 0;
        let mut p = 0;
        while p < 78 {
            if pairs[p] & w == pairs[p] {
                bits |= 1u128 << p;
            }
            p += 1;
        }
        out[w_idx] = bits;
        w_idx += 1;
    }
    out
}

static PAIR_IN_W: [u128; 10] = build_pair_in_w();

#[inline]
fn pack_rank_mask(row: &[bool]) -> u16 {
    let mut bits: u16 = 0;
    for (r, &b) in row.iter().enumerate().take(13) {
        if b {
            bits |= 1u16 << r;
        }
    }
    bits
}

#[inline]
fn cross_board_straight_per_env(
    hero_bits: u16,
    ba_bits: u16,
    bb_bits: u16,
    valid: bool,
) -> (f32, f32, f32) {
    if !valid {
        return (0.0, 0.0, 0.0);
    }

    let mut hero_has_pair: u128 = 0;
    for p in 0..78 {
        let pb = PAIR_BITS_78[p];
        if hero_bits & pb == pb {
            hero_has_pair |= 1u128 << p;
        }
    }
    if hero_has_pair == 0 {
        return (0.0, 0.0, 0.0);
    }

    let mut made_a: u128 = 0;
    let mut draw_a: u128 = 0;
    let mut made_b: u128 = 0;
    let mut draw_b: u128 = 0;

    for w_idx in 0..10 {
        let w = SF_W_MASKS[w_idx];
        let gate = hero_has_pair & PAIR_IN_W[w_idx];
        if gate == 0 {
            continue;
        }
        let ba_w = ba_bits & w;
        let bb_w = bb_bits & w;

        let mut bits = gate;
        while bits != 0 {
            let p = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let pb = PAIR_BITS_78[p];
            let cov_a = (ba_w | pb).count_ones();
            let cov_b = (bb_w | pb).count_ones();
            let pmask = 1u128 << p;
            if cov_a >= 5 {
                made_a |= pmask;
            } else if cov_a == 4 {
                draw_a |= pmask;
            }
            if cov_b >= 5 {
                made_b |= pmask;
            } else if cov_b == 4 {
                draw_b |= pmask;
            }
        }
    }

    let draw_only_a = draw_a & !made_a;
    let draw_only_b = draw_b & !made_b;
    let made_both = (made_a & made_b) != 0;
    let draw_both = (draw_only_a & draw_only_b) != 0;
    let mixed = ((made_a & draw_only_b) | (made_b & draw_only_a)) != 0;

    (
        if made_both { 1.0 } else { 0.0 },
        if draw_both { 1.0 } else { 0.0 },
        if mixed { 1.0 } else { 0.0 },
    )
}

#[pyfunction]
pub fn cross_board_straight_batch<'py>(
    py: Python<'py>,
    hole_rank_mask: PyReadonlyArray2<'_, bool>,
    ba_rank_mask: PyReadonlyArray2<'_, bool>,
    bb_rank_mask: PyReadonlyArray2<'_, bool>,
    valid: PyReadonlyArray1<'_, bool>,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let hole_v = hole_rank_mask.as_array();
    let ba_v = ba_rank_mask.as_array();
    let bb_v = bb_rank_mask.as_array();
    let valid_v = valid.as_array();

    let n = hole_v.shape()[0];
    if hole_v.shape() != [n, 13] {
        return Err(PyValueError::new_err("hole_rank_mask shape must be (N, 13)"));
    }
    if ba_v.shape() != [n, 13] || bb_v.shape() != [n, 13] {
        return Err(PyValueError::new_err(
            "ba_rank_mask and bb_rank_mask shapes must be (N, 13) matching hole",
        ));
    }
    if valid_v.shape() != [n] {
        return Err(PyValueError::new_err("valid shape must be (N,)"));
    }

    let hole_owned = hole_v.to_owned();
    let ba_owned = ba_v.to_owned();
    let bb_owned = bb_v.to_owned();
    let valid_owned = valid_v.to_owned();

    let mut made_both = vec![0f32; n];
    let mut draw_both = vec![0f32; n];
    let mut mixed = vec![0f32; n];

    py.allow_threads(|| {
        made_both
            .par_iter_mut()
            .zip(draw_both.par_iter_mut())
            .zip(mixed.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((md, dr), mx))| {
                let h_bits = pack_rank_mask(hole_owned.row(i).as_slice().unwrap());
                let a_bits = pack_rank_mask(ba_owned.row(i).as_slice().unwrap());
                let b_bits = pack_rank_mask(bb_owned.row(i).as_slice().unwrap());
                let v = valid_owned[i];
                let (m, d, x) = cross_board_straight_per_env(h_bits, a_bits, b_bits, v);
                *md = m;
                *dr = d;
                *mx = x;
            });
    });

    Ok((
        Array1::from_vec(made_both).into_pyarray(py),
        Array1::from_vec(draw_both).into_pyarray(py),
        Array1::from_vec(mixed).into_pyarray(py),
    ))
}

// =============================================================================
// Draw flags (vectorized port of `_draw_flags_batch` in encoding.py).
// =============================================================================

#[inline]
fn draw_flags_one_board(
    hole_suit_count: &[u8; 4],
    hole_rank_mask: u16,
    board_row: &[u8],
) -> (f32, f32) {
    let mut board_suit_count: [u8; 4] = [0; 4];
    let mut board_rank_mask: u16 = 0;
    let mut board_has_cards = false;
    for &c in board_row {
        if c < 52 {
            let r = (c >> 2) as usize;
            let s = (c & 3) as usize;
            board_suit_count[s] += 1;
            board_rank_mask |= 1u16 << r;
            board_has_cards = true;
        }
    }
    if !board_has_cards {
        return (0.0, 0.0);
    }

    let mut flush = false;
    for s in 0..4 {
        if hole_suit_count[s] >= 2 && board_suit_count[s] == 2 {
            flush = true;
            break;
        }
    }

    let rank_mask = hole_rank_mask | board_rank_mask;
    // Ace-low shadow: if bit 12 (ace) is set, also set bit 13. Window check
    // looks for 4 consecutive ranks across bits {0..13}, matching the
    // scalar `_draw_flags` (popcount>=4 over a 4-bit window).
    let ace_bit = (rank_mask >> 12) & 1;
    let extended = rank_mask | (ace_bit << 13);
    let mut straight = false;
    for start in 0..11u32 {
        if (extended >> start) & 0b1111 == 0b1111 {
            straight = true;
            break;
        }
    }

    (
        if flush { 1.0 } else { 0.0 },
        if straight { 1.0 } else { 0.0 },
    )
}

#[pyfunction]
pub fn draw_flags_batch<'py>(
    py: Python<'py>,
    hole: PyReadonlyArray2<'_, u8>,
    board_a: PyReadonlyArray2<'_, u8>,
    board_b: PyReadonlyArray2<'_, u8>,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let hole_v = hole.as_array();
    let ba_v = board_a.as_array();
    let bb_v = board_b.as_array();

    let n = hole_v.shape()[0];
    let hole_w = if n > 0 { hole_v.shape()[1] } else { 5 };
    if !(4..=6).contains(&hole_w) || hole_v.shape() != [n, hole_w] {
        return Err(PyValueError::new_err(
            "hole shape must be (N, 4), (N, 5), or (N, 6)",
        ));
    }
    if ba_v.shape() != [n, 5] || bb_v.shape() != [n, 5] {
        return Err(PyValueError::new_err(
            "board_a and board_b shapes must be (N, 5) matching hole",
        ));
    }

    let hole_owned = hole_v.to_owned();
    let ba_owned = ba_v.to_owned();
    let bb_owned = bb_v.to_owned();

    let mut flush_a = vec![0f32; n];
    let mut straight_a = vec![0f32; n];
    let mut flush_b = vec![0f32; n];
    let mut straight_b = vec![0f32; n];

    py.allow_threads(|| {
        flush_a
            .par_iter_mut()
            .zip(straight_a.par_iter_mut())
            .zip(flush_b.par_iter_mut())
            .zip(straight_b.par_iter_mut())
            .enumerate()
            .for_each(|(i, (((fa, sa), fb), sb))| {
                let hole_slice = hole_owned.row(i);
                let hole_slice = hole_slice.as_slice().unwrap();
                let mut hole_suit_count: [u8; 4] = [0; 4];
                let mut hole_rank_mask: u16 = 0;
                for &c in hole_slice {
                    if c < 52 {
                        let r = (c >> 2) as usize;
                        let s = (c & 3) as usize;
                        hole_suit_count[s] += 1;
                        hole_rank_mask |= 1u16 << r;
                    }
                }

                let ba_slice = ba_owned.row(i);
                let bb_slice = bb_owned.row(i);
                let (fa_v, sa_v) =
                    draw_flags_one_board(&hole_suit_count, hole_rank_mask, ba_slice.as_slice().unwrap());
                let (fb_v, sb_v) =
                    draw_flags_one_board(&hole_suit_count, hole_rank_mask, bb_slice.as_slice().unwrap());
                *fa = fa_v;
                *sa = sa_v;
                *fb = fb_v;
                *sb = sb_v;
            });
    });

    Ok((
        Array1::from_vec(flush_a).into_pyarray(py),
        Array1::from_vec(straight_a).into_pyarray(py),
        Array1::from_vec(flush_b).into_pyarray(py),
        Array1::from_vec(straight_b).into_pyarray(py),
    ))
}

// =============================================================================
// Pair features (vectorized port of `_pair_features_batch` in encoding.py).
// =============================================================================

#[inline]
fn pair_features_one_board(
    hole_rank_counts: &[u8; 13],
    board_row: &[u8],
    counts_out: &mut [f32],   // length 5
    struct_out: &mut [f32],   // length 4
) {
    let mut board_rank_counts: [u8; 13] = [0; 13];
    let mut board_ranks: [u8; 5] = [0; 5];
    let mut valid_count: usize = 0;
    for &c in board_row.iter().take(5) {
        if c < 52 {
            let r = c >> 2;
            board_ranks[valid_count] = r;
            board_rank_counts[r as usize] += 1;
            valid_count += 1;
        }
    }

    // Pre-zero outputs (caller may reuse buffers).
    for slot in 0..5 {
        counts_out[slot] = 0.0;
    }
    for slot in 0..4 {
        struct_out[slot] = 0.0;
    }

    if valid_count == 0 {
        return;
    }

    // Sort the valid prefix descending.
    board_ranks[..valid_count].sort_unstable_by(|a, b| b.cmp(a));

    for slot in 0..valid_count {
        counts_out[slot] = hole_rank_counts[board_ranks[slot] as usize] as f32;
    }

    let mut paired = false;
    let mut tripled = false;
    let mut quadded = false;
    let mut pairs: u32 = 0;
    for &c in &board_rank_counts {
        if c >= 2 {
            paired = true;
            pairs += 1;
        }
        if c >= 3 {
            tripled = true;
        }
        if c >= 4 {
            quadded = true;
        }
    }
    struct_out[0] = if paired { 1.0 } else { 0.0 };
    struct_out[1] = if pairs >= 2 { 1.0 } else { 0.0 };
    struct_out[2] = if tripled { 1.0 } else { 0.0 };
    struct_out[3] = if quadded { 1.0 } else { 0.0 };
}

#[pyfunction]
pub fn pair_features_batch<'py>(
    py: Python<'py>,
    hole: PyReadonlyArray2<'_, u8>,
    board_a: PyReadonlyArray2<'_, u8>,
    board_b: PyReadonlyArray2<'_, u8>,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
)> {
    let hole_v = hole.as_array();
    let ba_v = board_a.as_array();
    let bb_v = board_b.as_array();

    let n = hole_v.shape()[0];
    let hole_w = if n > 0 { hole_v.shape()[1] } else { 5 };
    if !(4..=6).contains(&hole_w) || hole_v.shape() != [n, hole_w] {
        return Err(PyValueError::new_err(
            "hole shape must be (N, 4), (N, 5), or (N, 6)",
        ));
    }
    if ba_v.shape() != [n, 5] || bb_v.shape() != [n, 5] {
        return Err(PyValueError::new_err(
            "board_a and board_b shapes must be (N, 5) matching hole",
        ));
    }

    let hole_owned = hole_v.to_owned();
    let ba_owned = ba_v.to_owned();
    let bb_owned = bb_v.to_owned();

    let mut counts_a = vec![0f32; n * 5];
    let mut struct_a = vec![0f32; n * 4];
    let mut counts_b = vec![0f32; n * 5];
    let mut struct_b = vec![0f32; n * 4];

    py.allow_threads(|| {
        counts_a
            .par_chunks_exact_mut(5)
            .zip(struct_a.par_chunks_exact_mut(4))
            .zip(counts_b.par_chunks_exact_mut(5))
            .zip(struct_b.par_chunks_exact_mut(4))
            .enumerate()
            .for_each(|(i, (((ca, sa), cb), sb))| {
                let hole_row = hole_owned.row(i);
                let hole_slice = hole_row.as_slice().unwrap();
                let mut hole_rank_counts: [u8; 13] = [0; 13];
                for &c in hole_slice {
                    if c < 52 {
                        hole_rank_counts[(c >> 2) as usize] += 1;
                    }
                }

                let ba_row = ba_owned.row(i);
                let bb_row = bb_owned.row(i);
                pair_features_one_board(
                    &hole_rank_counts,
                    ba_row.as_slice().unwrap(),
                    ca,
                    sa,
                );
                pair_features_one_board(
                    &hole_rank_counts,
                    bb_row.as_slice().unwrap(),
                    cb,
                    sb,
                );
            });
    });

    let counts_a_arr = Array2::from_shape_vec((n, 5), counts_a)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let struct_a_arr = Array2::from_shape_vec((n, 4), struct_a)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let counts_b_arr = Array2::from_shape_vec((n, 5), counts_b)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let struct_b_arr = Array2::from_shape_vec((n, 4), struct_b)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        counts_a_arr.into_pyarray(py),
        struct_a_arr.into_pyarray(py),
        counts_b_arr.into_pyarray(py),
        struct_b_arr.into_pyarray(py),
    ))
}

#[cfg(test)]
mod encoder_port_tests {
    use super::*;

    fn card(r: u8, s: u8) -> u8 {
        r * 4 + s
    }

    fn pad5(cards: &[u8]) -> [u8; 5] {
        let mut out = [255u8; 5];
        for (i, &c) in cards.iter().enumerate().take(5) {
            out[i] = c;
        }
        out
    }

    fn rank_mask(cards: &[u8]) -> [bool; 13] {
        let mut out = [false; 13];
        for &c in cards {
            if c < 52 {
                out[(c >> 2) as usize] = true;
            }
        }
        out
    }

    // ---- cross_board_straight ----

    #[test]
    fn cross_board_empty_inputs() {
        let (md, dr, mx) =
            cross_board_straight_per_env(0, 0, 0, true);
        assert_eq!(md, 0.0);
        assert_eq!(dr, 0.0);
        assert_eq!(mx, 0.0);
    }

    #[test]
    fn cross_board_invalid_returns_zero() {
        // valid=false short-circuits even if everything else fires.
        let hero = rank_mask(&[card(10, 0), card(11, 0)]);
        let ba = rank_mask(&[card(8, 0), card(9, 0), card(12, 0)]);
        let bb = rank_mask(&[card(8, 1), card(9, 1), card(12, 1)]);
        let hb = pack_rank_mask(&hero);
        let ab = pack_rank_mask(&ba);
        let bb_ = pack_rank_mask(&bb);
        let (md, dr, mx) = cross_board_straight_per_env(hb, ab, bb_, false);
        assert_eq!(md, 0.0);
        assert_eq!(dr, 0.0);
        assert_eq!(mx, 0.0);
    }

    #[test]
    fn cross_board_made_both_broadway() {
        // Hero K,Q. Boards both A,J,T. Window 9 (T-J-Q-K-A): pair {K,Q}
        // in window; cov = popcount(board & W | pair). board has A,J,T
        // = bits 12,9,8; pair = bits 11,10. Union = 0x1F00 (5 bits) on
        // each board → cov_a == 5 → made; cov_b == 5 → made. made_both.
        let hero = rank_mask(&[card(11, 0), card(10, 0)]); // K=11, Q=10
        let ba = rank_mask(&[card(12, 0), card(9, 0), card(8, 0)]); // A,J,T
        let bb = rank_mask(&[card(12, 1), card(9, 1), card(8, 1)]); // A,J,T
        let (md, dr, mx) = cross_board_straight_per_env(
            pack_rank_mask(&hero),
            pack_rank_mask(&ba),
            pack_rank_mask(&bb),
            true,
        );
        assert_eq!(md, 1.0);
        assert_eq!(dr, 0.0);
        assert_eq!(mx, 0.0);
    }

    #[test]
    fn cross_board_mixed() {
        // Hero K,Q. Board A has A,J,T (made). Board B has A,J (draw only:
        // cov = 4). → mixed.
        let hero = rank_mask(&[card(11, 0), card(10, 0)]); // K, Q
        let ba = rank_mask(&[card(12, 0), card(9, 0), card(8, 0)]); // A,J,T (5 covered)
        let bb = rank_mask(&[card(12, 1), card(9, 1)]); // A,J (4 covered with pair)
        let (md, dr, mx) = cross_board_straight_per_env(
            pack_rank_mask(&hero),
            pack_rank_mask(&ba),
            pack_rank_mask(&bb),
            true,
        );
        assert_eq!(md, 0.0);
        assert_eq!(dr, 0.0);
        assert_eq!(mx, 1.0);
    }

    // ---- draw_flags ----

    #[test]
    fn draw_flags_empty_board() {
        let hole = pad5(&[card(11, 0), card(10, 0)]);
        let empty_board = pad5(&[]);
        let mut hsc: [u8; 4] = [0; 4];
        let mut hrm: u16 = 0;
        for &c in &hole {
            if c < 52 {
                hsc[(c & 3) as usize] += 1;
                hrm |= 1u16 << (c >> 2);
            }
        }
        let (f, s) = draw_flags_one_board(&hsc, hrm, &empty_board);
        assert_eq!(f, 0.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn draw_flags_flush_draw_fires() {
        // Hero: Ah, Kh (suit 1, ranks 12, 11). Board: 2h, 5h, 9c
        // (suit 1 has 2 cards on board). hero_suit[1] = 2, board_suit[1] = 2 → flush.
        let hole = pad5(&[card(12, 1), card(11, 1)]);
        let board = pad5(&[card(0, 1), card(3, 1), card(7, 2)]);
        let mut hsc: [u8; 4] = [0; 4];
        let mut hrm: u16 = 0;
        for &c in &hole {
            if c < 52 {
                hsc[(c & 3) as usize] += 1;
                hrm |= 1u16 << (c >> 2);
            }
        }
        let (f, _) = draw_flags_one_board(&hsc, hrm, &board);
        assert_eq!(f, 1.0);
    }

    #[test]
    fn draw_flags_straight_draw_fires() {
        // Hero+board union covers ranks 4,5,6,7 (consecutive 4) → straight draw.
        // Hero: 6c, 7c (ranks 4,5, suit 2). Board: 8d, 9d, 2c (ranks 6,7,0).
        let hole = pad5(&[card(4, 2), card(5, 2)]);
        let board = pad5(&[card(6, 3), card(7, 3), card(0, 2)]);
        let mut hsc: [u8; 4] = [0; 4];
        let mut hrm: u16 = 0;
        for &c in &hole {
            if c < 52 {
                hsc[(c & 3) as usize] += 1;
                hrm |= 1u16 << (c >> 2);
            }
        }
        let (_, s) = draw_flags_one_board(&hsc, hrm, &board);
        assert_eq!(s, 1.0);
    }

    // ---- pair_features ----

    #[test]
    fn pair_features_empty_board() {
        let hole = pad5(&[card(11, 0), card(10, 0)]);
        let mut hrc: [u8; 13] = [0; 13];
        for &c in &hole {
            if c < 52 {
                hrc[(c >> 2) as usize] += 1;
            }
        }
        let mut counts = [0f32; 5];
        let mut struct_out = [0f32; 4];
        pair_features_one_board(&hrc, &pad5(&[]), &mut counts, &mut struct_out);
        assert!(counts.iter().all(|&x| x == 0.0));
        assert!(struct_out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn pair_features_top_pair() {
        // Hero: Kh, Qd (ranks 11, 10). Board: Kc, 9h, 4s (ranks 11, 7, 2).
        // Sorted descending: 11, 7, 2. counts: hero K count = 1, hero 9 = 0, hero 4 = 0.
        let hole = pad5(&[card(11, 1), card(10, 3)]);
        let board = pad5(&[card(11, 2), card(7, 1), card(2, 0)]);
        let mut hrc: [u8; 13] = [0; 13];
        for &c in &hole {
            if c < 52 {
                hrc[(c >> 2) as usize] += 1;
            }
        }
        let mut counts = [0f32; 5];
        let mut struct_out = [0f32; 4];
        pair_features_one_board(&hrc, &board, &mut counts, &mut struct_out);
        assert_eq!(counts[0], 1.0);  // K-slot
        assert_eq!(counts[1], 0.0);  // 9-slot
        assert_eq!(counts[2], 0.0);  // 4-slot
        assert_eq!(counts[3], 0.0);  // unused
        assert_eq!(counts[4], 0.0);  // unused
        // No pairs on board.
        assert_eq!(struct_out[0], 0.0);  // paired
    }

    #[test]
    fn pair_features_paired_board_repeats_slots() {
        // Hero: Kh, Qd. Board: Kc, Kd, 4s (K paired). Sorted: 11,11,2.
        // counts: 1, 1, 0.
        let hole = pad5(&[card(11, 1), card(10, 3)]);
        let board = pad5(&[card(11, 2), card(11, 3), card(2, 0)]);
        let mut hrc: [u8; 13] = [0; 13];
        for &c in &hole {
            if c < 52 {
                hrc[(c >> 2) as usize] += 1;
            }
        }
        let mut counts = [0f32; 5];
        let mut struct_out = [0f32; 4];
        pair_features_one_board(&hrc, &board, &mut counts, &mut struct_out);
        assert_eq!(counts[0], 1.0);
        assert_eq!(counts[1], 1.0);
        assert_eq!(counts[2], 0.0);
        assert_eq!(struct_out[0], 1.0);  // paired
        assert_eq!(struct_out[1], 0.0);  // double_paired
        assert_eq!(struct_out[2], 0.0);  // tripled
        assert_eq!(struct_out[3], 0.0);  // quadded
    }

    #[test]
    fn pair_features_quadded_board() {
        // Board: four Kings + 4. Struct: paired, tripled, quadded all 1.
        // double_paired = 0 (only one rank has count>=2).
        let hole = pad5(&[card(10, 0), card(9, 0)]);
        let board = pad5(&[card(11, 0), card(11, 1), card(11, 2), card(11, 3), card(2, 0)]);
        let mut hrc: [u8; 13] = [0; 13];
        for &c in &hole {
            if c < 52 {
                hrc[(c >> 2) as usize] += 1;
            }
        }
        let mut counts = [0f32; 5];
        let mut struct_out = [0f32; 4];
        pair_features_one_board(&hrc, &board, &mut counts, &mut struct_out);
        assert_eq!(struct_out[0], 1.0);  // paired
        assert_eq!(struct_out[1], 0.0);  // double_paired (only Ks have c>=2)
        assert_eq!(struct_out[2], 1.0);  // tripled
        assert_eq!(struct_out[3], 1.0);  // quadded
    }
}
