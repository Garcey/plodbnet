//! Serial `GameState` API (PyO3): one hand, driven one action at a time by
//! the UI, the study / trainer / home-game paths and the serial env.

use super::*;

/// Python-facing `GameState`. Construct with config, then `reset(seed, button)`
/// to deal a hand. Subsequent calls drive the state machine.
#[pyclass(name = "GameState")]
pub struct PyGameState {
    pub(super) inner: Option<GameState>,
    pub(super) config: GameConfig,
    /// Observation-semantics revision fixed at construction (see
    /// `OBS_REV_ENV`). The serial dict / range packers emit raw fields only,
    /// so nothing here branches on it yet; it is validated and exposed so the
    /// Python env can check it against `encoding.OBS_SEMANTICS_REV`.
    pub(super) obs_rev: u8,
}

#[pymethods]
impl PyGameState {
    /// `reach_cap=False`: the home games' betting rule — a bet is capped at the
    /// pot limit and the bettor's own stack only, never at what the shorter
    /// stacks can call (`GameConfig::reach_cap`; every network trains with it on).
    #[new]
    #[pyo3(signature = (num_seats=6, starting_stack=200000, ante=30000, bb=10000, starting_stacks=None, variant="plo5_double_bomb", sb=0, obs_rev=None, reach_cap=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_seats: usize,
        starting_stack: u64,
        ante: u64,
        bb: u64,
        starting_stacks: Option<PyReadonlyArray1<'_, u64>>,
        variant: &str,
        sb: u64,
        obs_rev: Option<u8>,
        reach_cap: bool,
    ) -> PyResult<Self> {
        let variant = parse_variant(variant)?;
        validate_table(num_seats, variant, bb)?;
        let obs_rev = resolve_obs_rev(obs_rev)?;
        let stacks = resolve_starting_stacks(num_seats, starting_stack, starting_stacks)?;
        Ok(PyGameState {
            inner: None,
            config: GameConfig {
                num_seats,
                starting_stacks: stacks,
                ante,
                bb,
                sb,
                variant,
                reach_cap,
            },
            obs_rev,
        })
    }

    /// Whether bets are also capped at what the deepest opponent can still put
    /// in (`True`, the trained rule) or only at the pot limit and the bettor's
    /// stack (`False`, the home games).
    fn reach_cap(&self) -> bool {
        self.config.reach_cap
    }

    /// Observation-semantics revision this state was constructed under.
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

    #[pyo3(signature = (seed, button, in_hand_mask=None))]
    fn reset(&mut self, seed: u64, button: usize, in_hand_mask: Option<Vec<bool>>) -> PyResult<()> {
        self.check_deal_args(button, in_hand_mask.as_deref())?;
        self.inner = Some(GameState::new_hand_with_mask(
            self.config.clone(),
            seed,
            button,
            in_hand_mask,
        ));
        Ok(())
    }

    /// [`Self::reset`] from an EXPLICIT deck order (52 distinct indices,
    /// first-dealt first) instead of a seed — the home games' verifiable
    /// shuffle deals the deck the players' devices helped permute. Same deal
    /// contract as `reset`: `hole_slots` cards per seat index (every index,
    /// dealt in or not; PLO67's slots past the first four are the extras its
    /// red burns hand out), seat 0 first, then full board A, then full board
    /// B, then the burns (PLO67 only).
    #[pyo3(signature = (deck, button, in_hand_mask=None))]
    fn reset_with_deck(
        &mut self,
        deck: Vec<u8>,
        button: usize,
        in_hand_mask: Option<Vec<bool>>,
    ) -> PyResult<()> {
        self.check_deal_args(button, in_hand_mask.as_deref())?;
        let deck = crate::cards::Deck::from_order(&deck).map_err(PyValueError::new_err)?;
        self.inner = Some(GameState::new_hand_from_deck(
            self.config.clone(),
            deck,
            button,
            in_hand_mask,
        ));
        Ok(())
    }

    /// The deck order `reset(seed, ..)` deals from (parity tests: a hand dealt
    /// from `shuffled_deck(seed)` is bit-identical to one dealt from `seed`).
    #[staticmethod]
    fn shuffled_deck(seed: u64) -> Vec<u8> {
        crate::cards::Deck::new_shuffled(seed).order().to_vec()
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

    /// Build a live NLH postflop engine node from a CFR solver root + path.
    ///
    /// `street`: 1=flop 2=turn 3=river. `path` is dump action labels
    /// (FOLD / CHECK_CALL / RAISE_pm / ALLIN). Ante is already inside `pot`.
    #[pyo3(signature = (pot, stacks, board, street, hero_seat, hero_hole, path, bb=None))]
    #[allow(clippy::too_many_arguments)]
    fn reset_nlh_cfr_node(
        &mut self,
        pot: u64,
        stacks: Vec<u64>,
        board: Vec<u8>,
        street: u8,
        hero_seat: usize,
        hero_hole: Vec<u8>,
        path: Vec<String>,
        bb: Option<u64>,
    ) -> PyResult<()> {
        use crate::cfr::engine_bridge::game_state_from_cfr_label;
        use crate::state::Street;
        if hero_hole.len() != 2 {
            return Err(PyValueError::new_err("hero_hole must be 2 cards"));
        }
        // Validate up front (review 2026-09-20 C4): the bridge indexes a
        // 52-slot table by raw card value and deals a fresh hand for
        // `stacks.len()` seats, so a bad card / seat count used to panic.
        // The seat count must also equal the wrapper's, or `num_seats()`,
        // `hero_category` and `pack_range_nlh` would disagree with the state.
        if stacks.len() != self.config.num_seats {
            return Err(PyValueError::new_err(format!(
                "stacks has {} entries but this GameState was built with num_seats={}",
                stacks.len(),
                self.config.num_seats
            )));
        }
        let bb = bb.unwrap_or(self.config.bb);
        validate_table(stacks.len(), Variant::NlhSingle, bb)?;
        let mut seen = CardMask::EMPTY;
        for &c in board.iter().chain(hero_hole.iter()) {
            if c >= 52 {
                return Err(PyValueError::new_err(format!(
                    "card index {c} out of range"
                )));
            }
            if !seen.insert(Card(c)) {
                return Err(PyValueError::new_err(format!(
                    "duplicate card {c} across board / hero_hole"
                )));
            }
        }
        let street = match street {
            1 => Street::Flop,
            2 => Street::Turn,
            3 => Street::River,
            _ => {
                return Err(PyValueError::new_err(
                    "reset_nlh_cfr_node street must be 1..=3 (postflop)",
                ))
            }
        };
        let hole = [hero_hole[0], hero_hole[1]];
        match game_state_from_cfr_label(pot, &stacks, &board, bb, street, hero_seat, hole, &path) {
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

    /// Chip delta per seat. All zeros while the hand is still live — same
    /// contract as `payouts_batch` (review 2026-09-20 C7): the engine's
    /// `payouts` is only valid at terminal, and on a live hand it used to
    /// return the showdown over the PRE-DEALT full boards, leaking the
    /// undealt turn/river (and villain holes) through a plain getter.
    fn payouts(&self) -> PyResult<Vec<i64>> {
        let g = self.get()?;
        if !g.is_terminal() {
            return Ok(vec![0i64; g.config.num_seats]);
        }
        Ok(g.payouts())
    }

    /// Expected chip delta per seat, averaged over `num_samples`
    /// Monte-Carlo runouts of the community cards undealt at the street
    /// where action closed. Delegates to `payouts` when sampling is a
    /// no-op (fold-out, river-close, or `num_samples == 0`). All zeros
    /// while the hand is still live (see `payouts`).
    ///
    /// The runouts run on a snapshot of the hand with the GIL released
    /// (PERF-028), so other Python threads keep going meanwhile.
    fn payouts_ev(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        num_samples: u32,
        seed: u64,
    ) -> PyResult<Vec<i64>> {
        let g = slf.get()?;
        if !g.is_terminal() {
            return Ok(vec![0i64; g.config.num_seats]);
        }
        let g = g.clone();
        drop(slf);
        Ok(py.detach(move || g.payouts_ev(num_samples, seed)))
    }

    /// Chips each seat has put in this hand (antes, blinds, bets) -- the
    /// dict's `total_commit` without building the dict (PERF-027: the env
    /// reads it twice per step).
    fn total_commit(&self) -> PyResult<Vec<u64>> {
        Ok(self.get()?.total_commit.clone())
    }

    fn num_seats(&self) -> usize {
        self.config.num_seats
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

    /// Hand category index (0..=8) of seat's best hand on `board` under the
    /// variant's rule (PLO: exactly 2 hole + 3 board; NLH: any 5 of 7);
    /// 0 = board A, 1 = board B.
    fn hero_category(&self, seat: usize, board: u8) -> PyResult<u8> {
        let g = self.get()?;
        // The STATE's seat count bounds `hole_cards` (review 2026-09-20 C4).
        if seat >= g.config.num_seats {
            return Err(PyValueError::new_err("seat out of range"));
        }
        // Same contract as `hero_category_batch`: any other value used to be
        // read as board B silently.
        if board > 1 {
            return Err(PyValueError::new_err(format!(
                "board {board} must be 0 or 1"
            )));
        }
        Ok(g.hero_category(seat, board))
    }

    /// 12 fractions in `[0, 1]`, row-major `[k=2,3,4][outcome]`,
    /// outcome enum: 0=scoop_opp, 1=quarter_opp, 2=scoop_hero,
    /// 3=quarter_hero. See `GameState::opp_outcome_fractions`.
    fn opp_outcome_fractions(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Vec<f32>> {
        let g = slf.get()?.clone();
        drop(slf);
        Ok(py.detach(move || g.opp_outcome_fractions()))
    }

    /// 22-dim superset: the 12 joint outcome fractions, the 8-dim per-board
    /// decomposition (obs v2 P1) and the 2 k=2 pot-share bounds (v7
    /// DUAL-4), one fused pass. See `GameState::outcome_features_mc`.
    fn outcome_features_mc(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        mc_samples: usize,
    ) -> PyResult<Vec<f32>> {
        let g = slf.get()?.clone();
        drop(slf);
        Ok(py.detach(move || g.outcome_features_mc(mc_samples).to_vec()))
    }

    /// Like `opp_outcome_fractions` but with an explicit k=3/k=4 MC
    /// sample budget (the no-arg form uses 1024). For tests / benchmarks
    /// of the training-vs-UI fidelity split.
    fn opp_outcome_fractions_mc(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        mc_samples: usize,
    ) -> PyResult<Vec<f32>> {
        let g = slf.get()?.clone();
        drop(slf);
        Ok(py.detach(move || g.opp_outcome_fractions_mc(mc_samples)))
    }

    /// Dict-shaped observation. See module doc for keys.
    ///
    /// The features only the full-layout encoder reads -- the 1024-sample
    /// opp-outcome MC (`opp_outcome_fractions` / `per_board_outcome` /
    /// `share_bounds`), `hero_board_v3`, `board_draw_v3` and NLH's
    /// `nlh_opp_outcome` sweep -- cost up to a few hundred microseconds, so:
    /// - `skip_outcome_mc=true` (bookkeeping, table views, the minimal
    ///   encoder) returns them as zeros without computing any (PERF-027);
    /// - otherwise they are computed on a snapshot of the hand with the GIL
    ///   released (PERF-028): other Python threads keep running, and a
    ///   concurrent call on this object sees the snapshot, never an
    ///   "Already borrowed" error.
    #[pyo3(signature = (skip_outcome_mc=false))]
    fn observation_dict<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        skip_outcome_mc: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        if skip_outcome_mc {
            return observation_dict_of(py, slf.get()?, &ObsFeatures::zeros());
        }
        let g = slf.get()?.clone();
        drop(slf);
        let feats = py.detach(|| ObsFeatures::compute(&g));
        observation_dict_of(py, &g, &feats)
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
        let holes = holes.as_array();
        if holes.ncols() != 2 {
            return Err(PyValueError::new_err("holes must have shape (N, 2)"));
        }
        let n = holes.nrows();
        // The STATE's seat count sizes its per-seat vectors (review
        // 2026-09-20 C4; equal to the wrapper's by construction).
        let s = g.config.num_seats;

        let on_board = CardMask::of(g.board_a.iter());
        let mut combos: Vec<[Card; 2]> = Vec::with_capacity(n);
        for i in 0..n {
            let (c0, c1) = (holes[[i, 0]], holes[[i, 1]]);
            if c0 >= 52 || c1 >= 52 || c0 == c1 {
                return Err(PyValueError::new_err(format!(
                    "invalid combo at row {i}: ({c0}, {c1})"
                )));
            }
            if on_board.contains(Card(c0)) || on_board.contains(Card(c1)) {
                return Err(PyValueError::new_err(format!(
                    "combo at row {i} collides with the board"
                )));
            }
            combos.push([Card::from_index(c0), Card::from_index(c1)]);
        }

        let hist_cap = history_cap(Variant::NlhSingle);
        let (packed, hero_cat_a) = py.detach(move || {
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
            // Every row is the SAME node with a different hero hole: the
            // shared core writer fills each row from the state (the batched
            // packers' exact layout), then the hole and its features are
            // swapped in.
            let mut packed = PackedObservation::alloc(n, s, 2, hist_cap);
            {
                let core = packed.core.rows();
                for i in 0..n {
                    // SAFETY: a serial loop writes each row exactly once.
                    unsafe { core.write(i, g) };
                }
            }
            packed.board_a_len.fill(g.board_a.len().min(5) as u8);
            packed.board_b_len.fill(g.board_b.len().min(5) as u8);
            packed
                .last_aggressor
                .fill(g.last_aggressor.map_or(-1, |x| x as i8));
            packed.sb_seat.fill(g.sb_seat.map_or(-1, |x| x as i8));
            packed.bb_seat.fill(g.bb_seat.map_or(-1, |x| x as i8));
            for mut row in packed.acted_this_street.rows_mut() {
                row.as_slice_mut()
                    .unwrap()
                    .copy_from_slice(&g.acted_this_street);
            }
            let mut cat = Array1::<u8>::zeros(n);
            for (i, (c, fr)) in per_combo.iter().enumerate() {
                packed.core.hero_hole[[i, 0]] = combos[i][0].index();
                packed.core.hero_hole[[i, 1]] = combos[i][1].index();
                cat[i] = *c;
                for j in 0..3 {
                    packed.nlh_opp_outcome[[i, j]] = fr[j];
                }
            }
            (packed, cat)
        });

        let d = packed.into_dict(py)?;
        d.set_item("hero_cat_a", hero_cat_a.into_pyarray(py))?;
        d.set_item("hero_cat_b", Array1::<u8>::zeros(n).into_pyarray(py))?;
        Ok(d)
    }

    /// All seats' hole cards as raw indices — the variant's hole count per
    /// seat (PLO67: what each seat holds NOW, 4-7). Trainer-only accessor
    /// for opponent reveal at hand end; never feed into observations
    /// mid-hand.
    fn all_hole_cards(&self) -> PyResult<Vec<Vec<u8>>> {
        let g = self.get()?;
        Ok(g.hole_cards
            .iter()
            .map(|h| h.iter().map(|c| c.index()).collect())
            .collect())
    }

    /// PLO67: ALL three pre-dealt burns, turned up or not — a reveal
    /// accessor like `all_hole_cards` (the rabbit hunt after a fold-out
    /// shows the burns that would have come). Never an observation.
    fn all_burns(&self) -> PyResult<Vec<u8>> {
        Ok(self.get()?.full_burns.iter().map(|c| c.index()).collect())
    }

    /// PLO67: the hole cards `seat` held on `street` (1 = flop, 2 = turn,
    /// 3 = river, 4 = showdown): a seat's extras are a prefix of the red
    /// burns it was in the hand for. Every other variant: its hole count.
    fn hole_count_on(&self, seat: usize, street: u8) -> PyResult<usize> {
        let g = self.get()?;
        if seat >= g.config.num_seats {
            return Err(PyValueError::new_err("seat out of range"));
        }
        let st = match street {
            0 => crate::state::Street::Preflop,
            1 => crate::state::Street::Flop,
            2 => crate::state::Street::Turn,
            3 | 4 => crate::state::Street::River,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "street {street} must be 0..=4"
                )))
            }
        };
        Ok(g.hole_count_on(seat, st))
    }
}

/// PLO67 all-in runout equities for the home games' felt: each contender's
/// share of board A and board B (see `engine::plo67_runout_equities`).
/// `holes` = the contenders' hole cards NOW (4-7 each), `dead` = the burns
/// turned up. Returns `[(share_a, share_b), ...]` in `holes` order.
#[pyfunction]
#[pyo3(signature = (holes, board_a, board_b, dead, samples=3000, seed=0))]
pub fn plo67_runout_equities(
    py: Python<'_>,
    holes: Vec<Vec<u8>>,
    board_a: Vec<u8>,
    board_b: Vec<u8>,
    dead: Vec<u8>,
    samples: u32,
    seed: u64,
) -> PyResult<Vec<(f64, f64)>> {
    let bad = |v: &[u8]| v.iter().any(|&c| c >= 52);
    if holes.iter().any(|h| bad(h)) || bad(&board_a) || bad(&board_b) || bad(&dead) {
        return Err(PyValueError::new_err("card index out of range"));
    }
    let cards = |v: &[u8]| v.iter().map(|&c| Card::from_index(c)).collect::<Vec<_>>();
    let holes: Vec<Vec<Card>> = holes.iter().map(|h| cards(h)).collect();
    let (a, b, d) = (cards(&board_a), cards(&board_b), cards(&dead));
    let out = py
        .detach(move || crate::engine::plo67_runout_equities(&holes, &a, &b, &d, samples, seed))
        .map_err(PyValueError::new_err)?;
    Ok(out.into_iter().map(|e| (e[0], e[1])).collect())
}

/// The full-layout-only features of `observation_dict` (see there).
pub(super) struct ObsFeatures {
    outcome: [f32; 22],
    hero_board_v3: [u8; 8],
    board_draw_v3: [u8; 7],
    nlh_opp_outcome: Vec<f32>,
}

impl ObsFeatures {
    fn zeros() -> ObsFeatures {
        ObsFeatures {
            outcome: [0.0; 22],
            hero_board_v3: [0; 8],
            board_draw_v3: [0; 7],
            nlh_opp_outcome: vec![0.0; 3],
        }
    }

    fn compute(g: &GameState) -> ObsFeatures {
        ObsFeatures {
            // One fused pass: the 1024-sample serial / UI budget.
            outcome: g.outcome_features_mc(1024),
            hero_board_v3: g.hero_board_v3(),
            board_draw_v3: g.board_draw_v3(),
            // Zeros (no evaluation) for the PLO variants.
            nlh_opp_outcome: g.nlh_opp_outcome_fractions(),
        }
    }
}

/// `observation_dict`'s dict: the state's public fields plus `f`.
fn observation_dict_of<'py>(
    py: Python<'py>,
    g: &GameState,
    f: &ObsFeatures,
) -> PyResult<Bound<'py, PyDict>> {
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
    // PLO67: the burn cards turned face up so far (one per street
    // reached); empty for every other variant.
    let burns: Vec<u8> = g.burns.iter().map(|c| c.index()).collect();
    d.set_item("burns", burns)?;

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

    // The fused pass's 12 joint fractions, the 8-dim per-board
    // decomposition (obs v2 P1) and the k=2 share bounds (v7 DUAL-4).
    d.set_item("opp_outcome_fractions", f.outcome[..12].to_vec())?;
    d.set_item("per_board_outcome", f.outcome[12..20].to_vec())?;
    d.set_item("share_bounds", f.outcome[20..22].to_vec())?;
    // v7 batch-2 engine dims (STK-1 / BRD-7 / BRD-12 / DUAL-2).
    d.set_item("acted_this_street", g.acted_this_street.clone())?;
    d.set_item("hero_board_v3", f.hero_board_v3.to_vec())?;
    d.set_item("board_draw_v3", f.board_draw_v3.to_vec())?;
    // NLH 3-dim [opp_ahead, tied, opp_behind]; zeros for other variants.
    d.set_item("nlh_opp_outcome", f.nlh_opp_outcome.clone())?;

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

impl PyGameState {
    /// The deal arguments `reset` / `reset_with_deck` share (ENG-003: this
    /// was written out twice): a seat's button and, optionally, the in-hand
    /// mask (one entry per seat, at least two seats in).
    pub(super) fn check_deal_args(
        &self,
        button: usize,
        in_hand_mask: Option<&[bool]>,
    ) -> PyResult<()> {
        if button >= self.config.num_seats {
            return Err(PyValueError::new_err("button out of range"));
        }
        if let Some(m) = in_hand_mask {
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
        Ok(())
    }

    pub(super) fn get(&self) -> PyResult<&GameState> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("GameState not reset; call reset() first"))
    }

    pub(super) fn get_mut(&mut self) -> PyResult<&mut GameState> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("GameState not reset; call reset() first"))
    }
}
