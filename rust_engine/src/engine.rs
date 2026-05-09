//! Game state machine: deal, apply action, advance street, settle payouts.

use crate::actions::{Action, NUM_ACTIONS};
use crate::cards::{Card, Deck};
use crate::double_board::double_board_payout;
use crate::state::{ActionRecord, GameConfig, GameState, Street, StudyError, StudyTerminal};

impl GameState {
    /// Deal a fresh hand: shuffle deck with `ChaCha8Rng::seed_from_u64(seed)`,
    /// post antes, deal 5 hole cards per seat, pre-deal both full boards,
    /// reveal flop on both, and set actor to first active seat left of button.
    pub fn new_hand(config: GameConfig, seed: u64, button: usize) -> Self {
        Self::new_hand_with_mask(config, seed, button, None)
    }

    /// Like [`Self::new_hand`] but only seats `i` with `in_hand_mask[i] == true`
    /// are dealt into the hand. Sitting-out seats post no ante, are marked
    /// folded from start, and do not appear as actors. With `None`, behaves
    /// identically to [`Self::new_hand`] (all seats in hand).
    ///
    /// The mask must have length `num_seats` if provided. At least two seats
    /// must be in-hand.
    pub fn new_hand_with_mask(
        config: GameConfig,
        seed: u64,
        button: usize,
        in_hand_mask: Option<Vec<bool>>,
    ) -> Self {
        let n = config.num_seats;
        assert!(n >= 2, "need at least 2 seats");
        assert!(button < n, "button out of range");
        if let Some(ref m) = in_hand_mask {
            assert_eq!(m.len(), n, "in_hand_mask length must equal num_seats");
            assert!(
                m.iter().filter(|&&b| b).count() >= 2,
                "in_hand_mask must include at least 2 seats"
            );
        }

        let mut deck = Deck::new_shuffled(seed);

        let mut hole_cards: Vec<[Card; 5]> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut h = [Card(0); 5];
            for c in h.iter_mut() {
                *c = deck.deal_one();
            }
            hole_cards.push(h);
        }

        let mut full_board_a = [Card(0); 5];
        for c in full_board_a.iter_mut() {
            *c = deck.deal_one();
        }
        let mut full_board_b = [Card(0); 5];
        for c in full_board_b.iter_mut() {
            *c = deck.deal_one();
        }

        assert_eq!(
            config.starting_stacks.len(),
            n,
            "starting_stacks length must equal num_seats"
        );
        let mut stacks = config.starting_stacks.clone();
        let mut total_commit = vec![0u64; n];
        let mut all_in = vec![false; n];
        let mut folded = vec![false; n];
        let mut pot: u64 = 0;
        for i in 0..n {
            let in_hand = in_hand_mask.as_ref().map_or(true, |m| m[i]);
            if !in_hand {
                folded[i] = true;
                continue;
            }
            let paid = stacks[i].min(config.ante);
            stacks[i] -= paid;
            total_commit[i] = paid;
            pot += paid;
            if stacks[i] == 0 {
                all_in[i] = true;
            }
        }

        let board_a: Vec<Card> = full_board_a[0..3].to_vec();
        let board_b: Vec<Card> = full_board_b[0..3].to_vec();
        let street_commit = vec![0u64; n];
        let acted_this_street = vec![false; n];

        let eff_stack_cap_at_hand_start =
            compute_eff_stack_cap(&config.starting_stacks, &folded);

        let mut state = GameState {
            config,
            button,
            street: Street::Flop,
            pot,
            stacks,
            folded,
            all_in,
            hole_cards,
            board_a,
            board_b,
            full_board_a,
            full_board_b,
            street_commit,
            total_commit,
            bet_to_call: 0,
            last_raise_size: 0, // set below from bb
            last_aggression_was_full_raise: true,
            actor: None,
            last_aggressor: None,
            acted_this_street,
            history: Vec::new(),
            study_mode: false,
            awaiting_next_street: None,
            study_terminal: None,
            study_hero_seat: None,
            action_close_board_len: None,
            eff_stack_cap_at_hand_start,
        };
        state.last_raise_size = state.config.bb;

        state.actor = state.first_to_act_postflop();

        // If nobody can voluntarily act (e.g., 5 of 6 went all-in on antes —
        // impossible at 20bb/3bb but keep robust), auto-run to showdown.
        if state.actor.is_none() {
            state.run_out_to_showdown();
        }

        state
    }

    /// Deal a study-mode hand at the flop with user-supplied cards.
    ///
    /// Unlike [`Self::new_hand`], turn/river are *not* pre-dealt — the UI
    /// supplies them via [`Self::set_turn`] / [`Self::set_river`] after the
    /// flop round closes. Non-hero seats are dealt placeholder holes
    /// deterministically from a seed derived from the user inputs, so the
    /// spot is fully reproducible but opponent cards are never surfaced to
    /// the UI (payouts at non-fold terminals return zeros).
    ///
    /// Errors:
    /// - `SeatOutOfRange`: `button` or `hero_seat` ≥ `num_seats`.
    /// - `DuplicateCard`: any repeat among the 11 user-supplied indices
    ///   (5 hero hole + 3 flop A + 3 flop B).
    pub fn new_study(
        config: GameConfig,
        button: usize,
        hero_seat: usize,
        hero_hole: [Card; 5],
        flop_a: [Card; 3],
        flop_b: [Card; 3],
    ) -> Result<Self, StudyError> {
        Self::new_study_with_mask(config, button, hero_seat, hero_hole, flop_a, flop_b, None)
    }

    /// Like [`Self::new_study`] but only seats `i` with `in_hand_mask[i] == true`
    /// are dealt into the hand. Sitting-out seats post no ante, are marked
    /// folded from start, and do not appear as actors. Hero seat must be
    /// in-hand.
    pub fn new_study_with_mask(
        config: GameConfig,
        button: usize,
        hero_seat: usize,
        hero_hole: [Card; 5],
        flop_a: [Card; 3],
        flop_b: [Card; 3],
        in_hand_mask: Option<Vec<bool>>,
    ) -> Result<Self, StudyError> {
        let n = config.num_seats;
        if n < 2 || button >= n || hero_seat >= n {
            return Err(StudyError::SeatOutOfRange);
        }
        if let Some(ref m) = in_hand_mask {
            if m.len() != n {
                return Err(StudyError::SeatOutOfRange);
            }
            if !m[hero_seat] {
                return Err(StudyError::SeatOutOfRange);
            }
            if m.iter().filter(|&&b| b).count() < 2 {
                return Err(StudyError::SeatOutOfRange);
            }
        }

        // Validate 11 distinct card indices across hero hole + both flops.
        let mut used = [false; 52];
        for c in hero_hole.iter().chain(flop_a.iter()).chain(flop_b.iter()) {
            let i = c.index() as usize;
            if i >= 52 || used[i] {
                return Err(StudyError::DuplicateCard);
            }
            used[i] = true;
        }

        // Seed a deck from a hash of all user inputs for deterministic opp hole draws.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hasher::write_u8(&mut hasher, button as u8);
        std::hash::Hasher::write_u8(&mut hasher, hero_seat as u8);
        for c in hero_hole.iter().chain(flop_a.iter()).chain(flop_b.iter()) {
            std::hash::Hasher::write_u8(&mut hasher, c.index());
        }
        let seed = std::hash::Hasher::finish(&hasher);

        // Build the unseen deck (excluding the 11 user-supplied cards) and
        // shuffle it via ChaCha8Rng for bit-exact reproducibility.
        let unseen: Vec<Card> = (0..52u8)
            .filter(|&i| !used[i as usize])
            .map(Card::from_index)
            .collect();
        let mut deck_order = unseen;
        {
            use rand::seq::SliceRandom;
            use rand_chacha::ChaCha8Rng;
            use rand_chacha::rand_core::SeedableRng;
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            deck_order.shuffle(&mut rng);
        }
        let mut next = 0usize;

        // Deal 5 placeholder holes per non-hero seat; hero seat uses hero_hole.
        let mut hole_cards: Vec<[Card; 5]> = Vec::with_capacity(n);
        for seat in 0..n {
            if seat == hero_seat {
                hole_cards.push(hero_hole);
            } else {
                let mut h = [Card(0); 5];
                for c in h.iter_mut() {
                    *c = deck_order[next];
                    next += 1;
                }
                hole_cards.push(h);
            }
        }

        // Post antes exactly as new_hand. Sitting-out seats (mask[i]==false)
        // post no ante and are marked folded so they never become actors.
        if config.starting_stacks.len() != n {
            return Err(StudyError::SeatOutOfRange);
        }
        let mut stacks = config.starting_stacks.clone();
        let mut total_commit = vec![0u64; n];
        let mut all_in = vec![false; n];
        let mut folded = vec![false; n];
        let mut pot: u64 = 0;
        for i in 0..n {
            let in_hand = in_hand_mask.as_ref().map_or(true, |m| m[i]);
            if !in_hand {
                folded[i] = true;
                continue;
            }
            let paid = stacks[i].min(config.ante);
            stacks[i] -= paid;
            total_commit[i] = paid;
            pot += paid;
            if stacks[i] == 0 {
                all_in[i] = true;
            }
        }

        // full_board_* carry flop in [0..3] and Card(0) sentinels in [3..5].
        let mut full_board_a = [Card(0); 5];
        let mut full_board_b = [Card(0); 5];
        for i in 0..3 {
            full_board_a[i] = flop_a[i];
            full_board_b[i] = flop_b[i];
        }
        let board_a: Vec<Card> = flop_a.to_vec();
        let board_b: Vec<Card> = flop_b.to_vec();

        let street_commit = vec![0u64; n];
        let acted_this_street = vec![false; n];

        let eff_stack_cap_at_hand_start =
            compute_eff_stack_cap(&config.starting_stacks, &folded);

        let mut state = GameState {
            config,
            button,
            street: Street::Flop,
            pot,
            stacks,
            folded,
            all_in,
            hole_cards,
            board_a,
            board_b,
            full_board_a,
            full_board_b,
            street_commit,
            total_commit,
            bet_to_call: 0,
            last_raise_size: 0,
            last_aggression_was_full_raise: true,
            actor: None,
            last_aggressor: None,
            acted_this_street,
            history: Vec::new(),
            study_mode: true,
            awaiting_next_street: None,
            study_terminal: None,
            study_hero_seat: Some(hero_seat),
            action_close_board_len: None,
            eff_stack_cap_at_hand_start,
        };
        state.last_raise_size = state.config.bb;
        state.actor = state.first_to_act_postflop();
        Ok(state)
    }

    /// Advance from flop to turn with user-supplied cards (study mode only).
    ///
    /// Requires `study_mode == true` and `awaiting_next_street == Some(Turn)`.
    /// `card_a` / `card_b` must not collide with any card already in use
    /// (hero hole, board A, board B). On success, boards are extended, the
    /// per-street state is reset, and `actor` is set to the first eligible
    /// seat clockwise from button.
    pub fn set_turn(&mut self, card_a: Card, card_b: Card) -> Result<(), StudyError> {
        self.set_next_street(Street::Turn, card_a, card_b)
    }

    /// Advance from turn to river with user-supplied cards (study mode only).
    ///
    /// Same contract as [`Self::set_turn`] but requires `awaiting_next_street == Some(River)`.
    pub fn set_river(&mut self, card_a: Card, card_b: Card) -> Result<(), StudyError> {
        self.set_next_street(Street::River, card_a, card_b)
    }

    fn set_next_street(
        &mut self,
        expected: Street,
        card_a: Card,
        card_b: Card,
    ) -> Result<(), StudyError> {
        if !self.study_mode {
            return Err(StudyError::WrongState);
        }
        if self.awaiting_next_street != Some(expected) {
            return Err(StudyError::WrongState);
        }

        // Validate no collision with any card already in play (hero hole +
        // both boards, using the progressive views). Non-hero holes are
        // placeholder draws not visible to the user, so we ignore them.
        let hero_seat = self.study_hero_seat.ok_or(StudyError::WrongState)?;
        let mut used = [false; 52];
        for c in self.hole_cards[hero_seat].iter() {
            used[c.index() as usize] = true;
        }
        for c in self.board_a.iter().chain(self.board_b.iter()) {
            used[c.index() as usize] = true;
        }
        let ia = card_a.index() as usize;
        let ib = card_b.index() as usize;
        if ia >= 52 || ib >= 52 || ia == ib || used[ia] || used[ib] {
            return Err(StudyError::DuplicateCard);
        }

        let idx = match expected {
            Street::Turn => 3,
            Street::River => 4,
            _ => return Err(StudyError::WrongState),
        };
        self.full_board_a[idx] = card_a;
        self.full_board_b[idx] = card_b;
        self.board_a.push(card_a);
        self.board_b.push(card_b);
        self.street = expected;

        let n = self.config.num_seats;
        self.street_commit = vec![0u64; n];
        self.bet_to_call = 0;
        self.last_raise_size = self.config.bb;
        self.last_aggression_was_full_raise = true;
        self.last_aggressor = None;
        self.acted_this_street = vec![false; n];
        self.awaiting_next_street = None;
        self.actor = self.first_to_act_postflop();
        Ok(())
    }

    /// Length-8 boolean mask; true = legal.
    pub fn legal_action_mask(&self) -> [bool; NUM_ACTIONS] {
        let mut mask = [false; NUM_ACTIONS];
        let actor = match self.actor {
            Some(a) => a,
            None => return mask,
        };
        let stack = self.stacks[actor];
        let current_commit = self.street_commit[actor];
        let facing_bet = self.bet_to_call > current_commit;

        // Fold: legal only when facing a bet (avoids meaningless fold-on-check).
        if facing_bet {
            mask[Action::Fold as usize] = true;
        }
        // CheckCall: always legal for a seat that has a turn.
        mask[Action::CheckCall as usize] = true;

        if stack == 0 {
            // No chips to bet with. Only Fold (if facing bet) / Check (trivial) remain.
            return mask;
        }

        // Short-all-in rule: if the most recent aggression was a sub-min-raise
        // shove AND this seat has already acted this street, raising is
        // forbidden — only Fold/CheckCall remain. See state.rs doc + memory
        // `feedback_short_allin_rule`.
        if facing_bet
            && self.acted_this_street[actor]
            && !self.last_aggression_was_full_raise
        {
            return mask;
        }

        let min_total = self.min_bet_total();
        let max_total = self.max_bet_total();
        let to_call = self.bet_to_call.saturating_sub(current_commit).min(stack);

        // Pre-compute chip deltas for each sizing + shove.
        let checkcall_chips = to_call;
        let mut sizing_chips = [0u64; 5]; // BetPct10, 25, 50, 75, 100
        let mut sizing_feasible = [false; 5];
        let sizings = [
            Action::BetPct10,
            Action::BetPct25,
            Action::BetPct50,
            Action::BetPct75,
            Action::BetPct100,
        ];
        for (i, &act) in sizings.iter().enumerate() {
            let chips = self.compute_sizing_chips(act, actor);
            let target_total = current_commit + chips;
            if target_total >= min_total && target_total <= max_total && chips > 0 {
                sizing_chips[i] = chips;
                sizing_feasible[i] = true;
            }
        }

        let shove_chips = stack;
        let shove_total = current_commit + shove_chips;
        // AllIn is legal when (a) the shove is at least a full raise
        // (meets min_total), (b) facing a bet, the shove is strictly
        // above the call but below min (short raise — doesn't reopen),
        // or (c) not facing a bet but stack can't reach 1bb (short open
        // — sub-min bet, doesn't reset the min-raise floor). Cases (b)
        // and (c) flow through apply()'s short-shove branch when
        // raise_delta < prev_raise_size.
        let allin_is_full = shove_total >= min_total;
        let allin_is_short_raise =
            facing_bet && shove_total > self.bet_to_call && shove_total < min_total;
        let allin_is_short_open =
            !facing_bet && shove_total > 0 && shove_total < min_total;
        let allin_feasible = shove_chips > 0
            && shove_total <= max_total
            && (allin_is_full || allin_is_short_raise || allin_is_short_open);

        // Mask sizings: each is legal iff feasible AND chips differs from
        // (a) CheckCall, (b) all lower-indexed feasible sizings, (c) AllIn (if feasible).
        for i in 0..5 {
            if !sizing_feasible[i] {
                continue;
            }
            let chips = sizing_chips[i];
            let mut dup = chips == checkcall_chips;
            for j in 0..i {
                if sizing_feasible[j] && sizing_chips[j] == chips {
                    dup = true;
                    break;
                }
            }
            if allin_feasible && chips == shove_chips {
                dup = true;
            }
            if !dup {
                mask[sizings[i].index() as usize] = true;
            }
        }

        if allin_feasible {
            mask[Action::AllIn as usize] = true;
        }

        mask
    }

    /// Chip amount a given action contributes. `None` if illegal.
    pub fn action_to_chips(&self, action: Action) -> Option<u64> {
        let mask = self.legal_action_mask();
        if !mask[action.index() as usize] {
            return None;
        }
        let actor = self.actor?;
        let current_commit = self.street_commit[actor];
        let stack = self.stacks[actor];
        let to_call = self.bet_to_call.saturating_sub(current_commit).min(stack);
        Some(match action {
            Action::Fold => 0,
            Action::CheckCall => to_call,
            Action::AllIn => stack,
            Action::BetPct10
            | Action::BetPct25
            | Action::BetPct50
            | Action::BetPct75
            | Action::BetPct100 => self.compute_sizing_chips(action, actor),
        })
    }

    /// Apply a legal action. Advances state; may auto-run through remaining
    /// streets when fewer than two voluntary actors remain.
    pub fn apply(&mut self, action: Action) {
        let actor = self.actor.expect("apply called on terminal state");
        let chips = self
            .action_to_chips(action)
            .expect("illegal action passed to apply");

        match action {
            Action::Fold => {
                self.folded[actor] = true;
                self.history.push(ActionRecord {
                    seat: actor,
                    action,
                    chips: 0,
                    street: self.street,
                });
            }
            _ => {
                self.stacks[actor] -= chips;
                self.pot += chips;
                self.street_commit[actor] += chips;
                self.total_commit[actor] += chips;

                let new_commit = self.street_commit[actor];
                if new_commit > self.bet_to_call {
                    let raise_delta = new_commit - self.bet_to_call;
                    let prev_raise_size = self.last_raise_size;
                    self.bet_to_call = new_commit;
                    if raise_delta >= prev_raise_size {
                        // Full raise (or opening bet ≥ 1bb): reopens action.
                        self.last_raise_size = raise_delta;
                        self.last_aggression_was_full_raise = true;
                    } else {
                        // Short shove below the min-raise floor: bet_to_call
                        // advances but floor is preserved and already-acted
                        // seats cannot re-raise.
                        self.last_aggression_was_full_raise = false;
                    }
                    self.last_aggressor = Some(actor);
                }
                if self.stacks[actor] == 0 {
                    self.all_in[actor] = true;
                }
                self.history.push(ActionRecord {
                    seat: actor,
                    action,
                    chips,
                    street: self.street,
                });
            }
        }

        self.acted_this_street[actor] = true;

        // Fold-out: single non-folded seat wins.
        let alive: Vec<usize> = (0..self.config.num_seats)
            .filter(|&i| !self.folded[i])
            .collect();
        if alive.len() == 1 {
            if self.study_mode {
                self.study_terminal = Some(StudyTerminal::FoldOut);
            }
            self.finalize_terminal();
            return;
        }

        // Find next voluntary actor. If none, close the round.
        match self.find_next_actor(actor) {
            Some(next) => self.actor = Some(next),
            None => self.close_round_or_run_out(),
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.actor.is_none()
    }

    pub fn current_actor(&self) -> Option<usize> {
        self.actor
    }

    /// Chip delta per seat. Sum == 0. Only valid when `is_terminal()`.
    ///
    /// In study mode: `FoldOut` computes normally (uncontested pot; no card
    /// evaluation needed). `RunOut` and `Showdown` return all zeros because
    /// opponent cards are unknown and unsampled runout cards are sentinels.
    pub fn payouts(&self) -> Vec<i64> {
        let n = self.config.num_seats;
        if self.study_mode {
            match self.study_terminal {
                Some(StudyTerminal::FoldOut) => {
                    // Fall through to double_board_payout; it short-circuits
                    // on single-survivor and never inspects board cards.
                }
                Some(StudyTerminal::RunOut) | Some(StudyTerminal::Showdown) => {
                    return vec![0i64; n];
                }
                None => return vec![0i64; n],
            }
        }
        let won = double_board_payout(
            &self.hole_cards,
            &self.folded,
            &self.total_commit,
            &self.full_board_a,
            &self.full_board_b,
            self.button,
        );
        (0..n)
            .map(|i| won[i] as i64 - self.total_commit[i] as i64)
            .collect()
    }

    /// Expected chip delta per seat, marginalised over unsampled community
    /// cards at the street where action closed. Opponent holes are held
    /// fixed (they were dealt at `new_hand` time and represent the actual
    /// hands in this rollout; resampling them would skew the estimate).
    ///
    /// Delegates to [`Self::payouts`] when sampling would be a no-op:
    /// study mode, fold-out (card-agnostic), river-close, or
    /// `num_samples == 0`.
    ///
    /// Sum is zero-sum up to integer-division rounding (at most `num_seats`
    /// chips of rounding slack).
    pub fn payouts_ev(&self, num_samples: u32, seed: u64) -> Vec<i64> {
        use rand::Rng;
        use rand_chacha::ChaCha8Rng;
        use rand_chacha::rand_core::SeedableRng;
        let n = self.config.num_seats;
        if self.study_mode || num_samples == 0 {
            return self.payouts();
        }
        let alive_count = (0..n).filter(|&i| !self.folded[i]).count();
        if alive_count < 2 {
            return self.payouts();
        }
        let close_len = match self.action_close_board_len {
            Some(l) if l < 5 => l as usize,
            _ => return self.payouts(),
        };
        let missing = 5 - close_len;
        let draw_per_sample = missing * 2;

        // Unseen deck: 52 − all hole cards − known prefix of both boards.
        let mut used = [false; 52];
        for hole in &self.hole_cards {
            for c in hole.iter() {
                used[c.index() as usize] = true;
            }
        }
        for c in &self.full_board_a[..close_len] {
            used[c.index() as usize] = true;
        }
        for c in &self.full_board_b[..close_len] {
            used[c.index() as usize] = true;
        }
        let mut deck: Vec<Card> = (0..52u8)
            .filter(|&i| !used[i as usize])
            .map(Card::from_index)
            .collect();
        let deck_size = deck.len();

        let mut full_a = [Card(0); 5];
        let mut full_b = [Card(0); 5];
        for i in 0..close_len {
            full_a[i] = self.full_board_a[i];
            full_b[i] = self.full_board_b[i];
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut totals: Vec<i128> = vec![0i128; n];

        for _ in 0..num_samples {
            // Partial Fisher-Yates: shuffle only the first `draw_per_sample` cards.
            for i in 0..draw_per_sample {
                let j = rng.gen_range(i..deck_size);
                deck.swap(i, j);
            }
            for i in 0..missing {
                full_a[close_len + i] = deck[i];
            }
            for i in 0..missing {
                full_b[close_len + i] = deck[missing + i];
            }
            let won = double_board_payout(
                &self.hole_cards,
                &self.folded,
                &self.total_commit,
                &full_a,
                &full_b,
                self.button,
            );
            for i in 0..n {
                totals[i] += won[i] as i128;
            }
        }

        let denom = num_samples as i128;
        (0..n)
            .map(|i| {
                let mean_won = (totals[i] / denom) as i64;
                mean_won - self.total_commit[i] as i64
            })
            .collect()
    }

    /// Minimum legal total street_commit for a bet/raise.
    pub fn min_bet_total(&self) -> u64 {
        if self.bet_to_call == 0 {
            self.config.bb
        } else {
            self.bet_to_call + self.last_raise_size
        }
    }

    /// Largest total street_commit any alive opponent can still reach
    /// (their current street_commit plus remaining stack). Used to cap
    /// the actor's raise/bet at the effective stack — chips above this
    /// have no caller and are uncontested. Zero when no actor or no
    /// alive opponents.
    pub fn max_other_reachable_total(&self) -> u64 {
        let actor = match self.actor {
            Some(a) => a,
            None => return 0,
        };
        (0..self.stacks.len())
            .filter(|&j| j != actor && !self.folded[j])
            .map(|j| self.street_commit[j] + self.stacks[j])
            .max()
            .unwrap_or(0)
    }

    /// Pot-limit cap on total street_commit, further capped at the
    /// maximum total any alive opponent can match. Above the latter,
    /// chips are dead money under side-pot rules.
    pub fn max_bet_total(&self) -> u64 {
        let actor = match self.actor {
            Some(a) => a,
            None => return 0,
        };
        let current_commit = self.street_commit[actor];
        let stack = self.stacks[actor];
        let to_call = self.bet_to_call.saturating_sub(current_commit).min(stack);
        let pl_total = self.bet_to_call + self.pot + to_call;
        pl_total.min(self.max_other_reachable_total())
    }

    /// True when the current actor is locked out of raising by the
    /// short-shove reopen rule: already acted this street and facing a
    /// raise that didn't meet the min-raise floor.
    fn short_shove_lockout(&self) -> bool {
        let actor = match self.actor {
            Some(a) => a,
            None => return false,
        };
        let facing_bet = self.bet_to_call > self.street_commit[actor];
        facing_bet && self.acted_this_street[actor] && !self.last_aggression_was_full_raise
    }

    /// Smallest chip delta the current actor can add to make a legal
    /// raise/bet. Zero if Raise is unavailable — actor can't afford the
    /// floor, is terminal, or locked out by the short-shove rule.
    ///
    /// Cover-short clamp: when the 1 BB floor exceeds what the deepest
    /// alive opponent can reach, the floor collapses to that effective
    /// cap. Result: `min == max == cap_delta`, a single-amount raise
    /// (covering the short opponent, sub-1BB but legal in real poker).
    pub fn min_raise_chips(&self) -> u64 {
        let actor = match self.actor {
            Some(a) => a,
            None => return 0,
        };
        if self.short_shove_lockout() {
            return 0;
        }
        let current_commit = self.street_commit[actor];
        let stack = self.stacks[actor];
        let min_total = self.min_bet_total();
        if min_total <= current_commit {
            return 0;
        }
        let delta = min_total - current_commit;
        // Cover-short: when no alive opponent can reach `min_total`,
        // collapse the floor to the effective-cap delta — but only if
        // the resulting commit *strictly exceeds* `bet_to_call`. If the
        // cap merely matches an existing bet, there's no raise to make
        // (it would be a flat call), so no Raise is legal.
        let max_other = self.max_other_reachable_total();
        if max_other <= self.bet_to_call {
            return 0;
        }
        let cap_delta = max_other.saturating_sub(current_commit);
        if cap_delta == 0 {
            return 0;
        }
        // Affordability is checked against the (possibly clamped) commit.
        // The deep actor may not afford the full 1 BB floor but can still
        // afford the covering bet that caps at the short opp's reach.
        let clamped = delta.min(cap_delta);
        if clamped > stack {
            return 0;
        }
        clamped
    }

    /// Largest chip delta the current actor can add as a raise/bet,
    /// capped by the PL maximum, the actor's stack, and the deepest
    /// alive opponent's reachable total. Zero when no raise is legal.
    ///
    /// In the cover-short regime (`min_bet_total > max_other_reachable`)
    /// this returns the effective-cap delta — `min_raise_chips` clamps
    /// to the same value, so the legal range is a single point.
    pub fn max_raise_chips(&self) -> u64 {
        let actor = match self.actor {
            Some(a) => a,
            None => return 0,
        };
        if self.short_shove_lockout() {
            return 0;
        }
        let current_commit = self.street_commit[actor];
        let stack = self.stacks[actor];
        let max_total = self.max_bet_total();
        let cap_total = max_total.min(current_commit + stack);
        // Raise must strictly exceed `bet_to_call` — a cap that only
        // matches the call is a flat call, not a raise.
        if cap_total <= current_commit || cap_total <= self.bet_to_call {
            return 0;
        }
        cap_total - current_commit
    }

    /// Continuous-sizing raise entry point. `chips` is the chip delta the
    /// actor contributes to the pot (not a target total). Mirrors
    /// [`Self::apply`] for a BetPctN / AllIn action but lets the caller
    /// pick any amount in `[min_raise_chips(), max_raise_chips()]`.
    ///
    /// Records the resulting `ActionRecord` with `Action::BetPct100` as a
    /// transitional sentinel; the authoritative amount lives in the
    /// `chips` field. Phase 4 replaces pct-discrete history encoding with
    /// a chip-amount representation.
    ///
    /// Errors:
    /// - `WrongState`: terminal, folded, or all-in actor.
    /// - `InvalidAmount`: chips out of `[min_raise_chips, max_raise_chips]`.
    pub fn apply_raise_chips(&mut self, chips: u64) -> Result<(), StudyError> {
        let actor = self.actor.ok_or(StudyError::WrongState)?;
        if self.folded[actor] || self.all_in[actor] {
            return Err(StudyError::WrongState);
        }
        let min = self.min_raise_chips();
        let max = self.max_raise_chips();
        if min == 0 || chips < min || chips > max {
            return Err(StudyError::InvalidAmount);
        }
        self.commit_chips_as_raise(actor, chips, Action::BetPct100);

        // Round-close + next-actor logic identical to `apply`.
        let alive: Vec<usize> = (0..self.config.num_seats)
            .filter(|&i| !self.folded[i])
            .collect();
        if alive.len() == 1 {
            if self.study_mode {
                self.study_terminal = Some(StudyTerminal::FoldOut);
            }
            self.finalize_terminal();
            return Ok(());
        }
        match self.find_next_actor(actor) {
            Some(next) => self.actor = Some(next),
            None => self.close_round_or_run_out(),
        }
        Ok(())
    }

    fn commit_chips_as_raise(&mut self, actor: usize, chips: u64, label: Action) {
        self.stacks[actor] -= chips;
        self.pot += chips;
        self.street_commit[actor] += chips;
        self.total_commit[actor] += chips;

        let new_commit = self.street_commit[actor];
        if new_commit > self.bet_to_call {
            let raise_delta = new_commit - self.bet_to_call;
            let prev_raise_size = self.last_raise_size;
            self.bet_to_call = new_commit;
            if raise_delta >= prev_raise_size {
                self.last_raise_size = raise_delta;
                self.last_aggression_was_full_raise = true;
            } else {
                self.last_aggression_was_full_raise = false;
            }
            self.last_aggressor = Some(actor);
        }
        if self.stacks[actor] == 0 {
            self.all_in[actor] = true;
        }
        self.history.push(ActionRecord {
            seat: actor,
            action: label,
            chips,
            street: self.street,
        });
        self.acted_this_street[actor] = true;
    }

    /// Hand category index (0..=8 per `CAT_*` constants) of seat's current
    /// best PLO5 hand on `board` (0=A, 1=B). Returns 0 if board has <3 cards.
    pub fn hero_category(&self, seat: usize, board: u8) -> u8 {
        let b = if board == 0 { &self.board_a } else { &self.board_b };
        if b.len() < 3 {
            return 0;
        }
        let hole = &self.hole_cards[seat];
        let rank = crate::hand_eval::evaluate_plo5_partial(hole, b);
        (rank >> 20) as u8
    }

    /// Fraction of unseen-deck k-card opponent hands that produce each of
    /// 4 outcomes vs the hero (current actor) on both boards, for
    /// k ∈ {2, 3, 4}. Returns a length-12 `Vec<f32>` in row-major
    /// `[k][outcome]` order:
    /// - Outcome 0: opp scoops (opp wins both boards).
    /// - Outcome 1: opp quarters hero (opp wins one, ties the other).
    /// - Outcome 2: hero scoops (hero wins both).
    /// - Outcome 3: hero quarters opp (hero wins one, ties the other).
    ///
    /// Chops, splits, and double-ties are intentionally excluded; the
    /// network can infer them as residuals.
    ///
    /// Evaluation rule (current-rank dominance): each k-card opp hand is
    /// evaluated on the *visible* board under PLO5 rules (exactly 2 from
    /// k + 3 from board). No runout sampling on flop/turn.
    ///
    /// Sampling: k=2 and k=3 are exhaustive; k=4 uses 1024 MC samples.
    /// PRNG seeded deterministically from the immutable observation
    /// state so the feature is reproducible (parity tests survive).
    ///
    /// Returns all-zero before the flop or when the hand is terminal.
    pub fn opp_outcome_fractions(&self) -> Vec<f32> {
        const N_OUT: usize = 12;
        const SCOOP_OPP: usize = 0;
        const QUARTER_OPP: usize = 1;
        const SCOOP_HERO: usize = 2;
        const QUARTER_HERO: usize = 3;
        const MC_THRESHOLD: usize = 10_000;
        const MC_SAMPLES_K4: usize = 1024;

        let hero_seat = match self.actor {
            Some(s) => s,
            None => return vec![0.0; N_OUT],
        };
        if self.board_a.len() < 3 || self.board_b.len() < 3 {
            return vec![0.0; N_OUT];
        }

        let hero_hole = &self.hole_cards[hero_seat];
        let hero_a = crate::hand_eval::evaluate_plo5_partial(hero_hole, &self.board_a);
        let hero_b = crate::hand_eval::evaluate_plo5_partial(hero_hole, &self.board_b);

        // Build unseen deck (52 minus hero hole minus visible board on
        // both boards).
        let mut used = [false; 52];
        for c in hero_hole.iter() {
            used[c.index() as usize] = true;
        }
        for c in self.board_a.iter().chain(self.board_b.iter()) {
            used[c.index() as usize] = true;
        }
        let unseen: Vec<Card> = (0..52u8)
            .filter(|&i| !used[i as usize])
            .map(Card::from_index)
            .collect();
        let n_unseen = unseen.len();

        // Deterministic seed from the observation-visible state.
        let seed: u64 = {
            use std::hash::Hasher;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            hasher.write_u8(hero_seat as u8);
            hasher.write_u8(self.street.index() as u8);
            for c in hero_hole.iter() {
                hasher.write_u8(c.index());
            }
            for c in self.board_a.iter().chain(self.board_b.iter()) {
                hasher.write_u8(c.index());
            }
            hasher.finish()
        };

        use rand_chacha::ChaCha8Rng;
        use rand_chacha::rand_core::{RngCore, SeedableRng};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut out = vec![0.0f32; N_OUT];
        let mut opp_buf: Vec<Card> = Vec::with_capacity(4);

        let classify = |opp: &[Card],
                        board_a: &[Card],
                        board_b: &[Card],
                        counters: &mut [u32; 4]| {
            let opp_a = crate::hand_eval::evaluate_plo5_k_partial(opp, board_a);
            let opp_b = crate::hand_eval::evaluate_plo5_k_partial(opp, board_b);
            // Higher HandRank = stronger.
            let cmp_a: i8 = if opp_a > hero_a {
                1
            } else if opp_a < hero_a {
                -1
            } else {
                0
            };
            let cmp_b: i8 = if opp_b > hero_b {
                1
            } else if opp_b < hero_b {
                -1
            } else {
                0
            };
            match (cmp_a, cmp_b) {
                (1, 1) => counters[SCOOP_OPP] += 1,
                (-1, -1) => counters[SCOOP_HERO] += 1,
                (1, 0) | (0, 1) => counters[QUARTER_OPP] += 1,
                (-1, 0) | (0, -1) => counters[QUARTER_HERO] += 1,
                _ => {}
            }
        };

        for (idx_k, &k) in [2usize, 3, 4].iter().enumerate() {
            let total = n_choose_k(n_unseen, k);
            let mut counters = [0u32; 4];
            let mut samples: u32 = 0;

            // k=2,3 exhaustive at any street (within MC_THRESHOLD).
            // k=4 falls back to MC.
            if total <= MC_THRESHOLD {
                let mut idx: Vec<usize> = (0..k).collect();
                loop {
                    opp_buf.clear();
                    for &i in idx.iter() {
                        opp_buf.push(unseen[i]);
                    }
                    classify(&opp_buf, &self.board_a, &self.board_b, &mut counters);
                    samples += 1;
                    // Advance to next combination.
                    let mut pos = k;
                    let advanced = loop {
                        if pos == 0 {
                            break false;
                        }
                        pos -= 1;
                        if idx[pos] < n_unseen - (k - pos) {
                            idx[pos] += 1;
                            for j in (pos + 1)..k {
                                idx[j] = idx[j - 1] + 1;
                            }
                            break true;
                        }
                    };
                    if !advanced {
                        break;
                    }
                }
            } else {
                // MC: rejection sampling with a 64-bit mask
                // (n_unseen ≤ 41 ≤ 64).
                debug_assert!(n_unseen <= 64);
                for _ in 0..MC_SAMPLES_K4 {
                    let mut mask: u64 = 0;
                    opp_buf.clear();
                    let mut written = 0;
                    while written < k {
                        let i = (rng.next_u32() as usize) % n_unseen;
                        let bit = 1u64 << i;
                        if mask & bit == 0 {
                            mask |= bit;
                            opp_buf.push(unseen[i]);
                            written += 1;
                        }
                    }
                    classify(&opp_buf, &self.board_a, &self.board_b, &mut counters);
                    samples += 1;
                }
            }

            if samples > 0 {
                let inv = 1.0f32 / samples as f32;
                let base = idx_k * 4;
                out[base + SCOOP_OPP] = counters[SCOOP_OPP] as f32 * inv;
                out[base + QUARTER_OPP] = counters[QUARTER_OPP] as f32 * inv;
                out[base + SCOOP_HERO] = counters[SCOOP_HERO] as f32 * inv;
                out[base + QUARTER_HERO] = counters[QUARTER_HERO] as f32 * inv;
            }
        }
        out
    }

    // ---- Internal helpers ----

    fn compute_sizing_chips(&self, action: Action, actor: usize) -> u64 {
        let current_commit = self.street_commit[actor];
        let stack = self.stacks[actor];
        let (num, den) = match action.pot_fraction() {
            Some(f) => f,
            None => return 0,
        };
        let to_call_raw = self.bet_to_call.saturating_sub(current_commit);

        let raise_over = if self.bet_to_call == 0 {
            self.pot * num / den
        } else {
            let pot_after_call = self.pot + to_call_raw;
            pot_after_call * num / den
        };
        let target_total = if self.bet_to_call == 0 {
            raise_over
        } else {
            self.bet_to_call + raise_over
        };
        let min_total = self.min_bet_total();
        let max_total = self.max_bet_total();
        let clamped = target_total.max(min_total).min(max_total);
        let chips_want = clamped.saturating_sub(current_commit);
        chips_want.min(stack)
    }

    /// First seat-to-act for a postflop betting round. Walks clockwise
    /// starting at `(button + 1) % num_seats` and returns the first seat
    /// that is neither folded nor all-in. Returns `None` if no eligible
    /// seat exists (caller should then close / run out / finalize).
    ///
    /// This is an **eligibility walk**, not a fixed offset: if e.g. SB
    /// check-folded on the flop, SB is skipped on the turn and the helper
    /// returns BB (or the next eligible seat clockwise).
    fn first_to_act_postflop(&self) -> Option<usize> {
        let n = self.config.num_seats;
        let start = (self.button + 1) % n;
        for i in 0..n {
            let s = (start + i) % n;
            if !self.folded[s] && !self.all_in[s] {
                return Some(s);
            }
        }
        None
    }

    fn find_next_actor(&self, after: usize) -> Option<usize> {
        let n = self.config.num_seats;
        let start = (after + 1) % n;
        for i in 0..n {
            let s = (start + i) % n;
            if self.folded[s] || self.all_in[s] {
                continue;
            }
            // Either they haven't acted this street, or they face a bet above
            // their current commit (i.e., a raise reopened their turn).
            if !self.acted_this_street[s] || self.street_commit[s] < self.bet_to_call {
                return Some(s);
            }
        }
        None
    }

    /// Called when the current betting round has closed (no voluntary actor
    /// remains). In production, either advance to the next street and start
    /// a new round, or run out to showdown if fewer than two voluntary
    /// actors remain. In study mode, halts at street boundaries so the UI
    /// can supply next-street cards, and never evaluates a showdown.
    fn close_round_or_run_out(&mut self) {
        let n = self.config.num_seats;

        // Snapshot the street at which this round closed. By terminal time
        // this reflects the last street where a betting round closed —
        // i.e., the "all-in street" for run-outs. Consumed by `payouts_ev`.
        self.action_close_board_len = Some(self.board_a.len() as u8);

        // Fold-out handling is identical in both modes — single survivor.
        let alive: Vec<usize> = (0..n).filter(|&i| !self.folded[i]).collect();
        if alive.len() == 1 {
            if self.study_mode {
                self.study_terminal = Some(StudyTerminal::FoldOut);
            }
            self.finalize_terminal();
            return;
        }

        if self.study_mode {
            self.close_round_study();
            return;
        }

        loop {
            // If only one non-folded seat, finalize immediately.
            let alive: Vec<usize> = (0..n).filter(|&i| !self.folded[i]).collect();
            if alive.len() == 1 {
                self.finalize_terminal();
                return;
            }

            // Advance one street (or finalize at showdown).
            let next = match self.street {
                Street::Preflop => Street::Flop,
                Street::Flop => Street::Turn,
                Street::Turn => Street::River,
                Street::River => Street::Showdown,
                Street::Showdown => {
                    self.finalize_terminal();
                    return;
                }
            };
            self.street = next;
            match next {
                Street::Flop => {
                    self.board_a = self.full_board_a[0..3].to_vec();
                    self.board_b = self.full_board_b[0..3].to_vec();
                }
                Street::Turn => {
                    self.board_a.push(self.full_board_a[3]);
                    self.board_b.push(self.full_board_b[3]);
                }
                Street::River => {
                    self.board_a.push(self.full_board_a[4]);
                    self.board_b.push(self.full_board_b[4]);
                }
                Street::Showdown => {
                    self.finalize_terminal();
                    return;
                }
                Street::Preflop => unreachable!(),
            }

            // Reset per-street state.
            self.street_commit = vec![0u64; n];
            self.bet_to_call = 0;
            self.last_raise_size = self.config.bb;
            self.last_aggression_was_full_raise = true;
            self.last_aggressor = None;
            self.acted_this_street = vec![false; n];

            // Does a new round start? Need >=2 non-folded non-all-in seats.
            let can_act: Vec<usize> = (0..n)
                .filter(|&i| !self.folded[i] && !self.all_in[i])
                .collect();
            if can_act.len() >= 2 {
                self.actor = self.first_to_act_postflop();
                return;
            }
            // else continue looping — reveal next street, run out.
        }
    }

    /// Study-mode round-close: three outcomes.
    ///
    /// - `can_act.len() >= 2` and there's a next undealt street →
    ///   `awaiting_next_street = Some(next)`, `actor = None`. UI supplies cards.
    /// - `can_act.len() >= 2` and current street is River →
    ///   `study_terminal = Some(Showdown)`, `actor = None`, `street = Showdown`.
    /// - `can_act.len() < 2` (run-out case) →
    ///   `study_terminal = Some(RunOut)`, `actor = None`. No auto-run-out.
    fn close_round_study(&mut self) {
        let n = self.config.num_seats;
        let can_act: Vec<usize> = (0..n)
            .filter(|&i| !self.folded[i] && !self.all_in[i])
            .collect();
        if can_act.len() < 2 {
            self.study_terminal = Some(StudyTerminal::RunOut);
            self.actor = None;
            return;
        }
        match self.street {
            Street::Flop => {
                self.awaiting_next_street = Some(Street::Turn);
                self.actor = None;
            }
            Street::Turn => {
                self.awaiting_next_street = Some(Street::River);
                self.actor = None;
            }
            Street::River => {
                self.study_terminal = Some(StudyTerminal::Showdown);
                self.actor = None;
                self.street = Street::Showdown;
            }
            Street::Preflop | Street::Showdown => {
                // Unreachable in study mode (start at Flop, end at Showdown).
                self.actor = None;
            }
        }
    }

    /// Helper: run out all remaining streets to showdown (used when a hand
    /// begins in an already-run-out state, e.g. everyone all-in after antes).
    fn run_out_to_showdown(&mut self) {
        self.close_round_or_run_out();
    }

    fn finalize_terminal(&mut self) {
        // In production, reveal the full pre-dealt boards for observability.
        // In study mode, turn/river may be undealt (`Card(0)` sentinels) so
        // leave the progressive view intact — payouts for FoldOut are
        // card-agnostic (uncontested pot) and RunOut/Showdown return zeros.
        if !self.study_mode {
            if self.board_a.len() < 5 {
                self.board_a = self.full_board_a.to_vec();
            }
            if self.board_b.len() < 5 {
                self.board_b = self.full_board_b.to_vec();
            }
        }
        self.street = Street::Showdown;
        self.actor = None;
    }
}

/// Compute `C(n, k)`. Returns 0 when `k > n`. Saturates at `usize::MAX`
/// on overflow (only relevant for very large k that we don't use here).
fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

/// Per-seat hand-start effective-stack cap. For seat `i`:
/// `min(starting_stacks[i], max(starting_stacks[j] for j != i and !folded[j]))`.
/// At construction `folded[i]` is true iff seat `i` is sitting out, so
/// the `max_other` reduction excludes sit-outs. Returns own stack when
/// no other in-hand seat exists.
pub fn compute_eff_stack_cap(starting_stacks: &[u64], folded: &[bool]) -> Vec<u64> {
    let n = starting_stacks.len();
    let mut cap = vec![0u64; n];
    for i in 0..n {
        let max_other = (0..n)
            .filter(|&j| j != i && !folded[j])
            .map(|j| starting_stacks[j])
            .max()
            .unwrap_or(starting_stacks[i]);
        cap[i] = starting_stacks[i].min(max_other);
    }
    cap
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> GameConfig {
        GameConfig::default_6max_20bb()
    }

    #[test]
    fn eff_stack_cap_freezes_at_hand_start_3_handed_unequal() {
        // User's example scenario: 3-handed stacks 20/30/40 bb. Pre-ante
        // hand-start cap per seat is min(own, max_other). Verify the cap
        // doesn't shrink mid-hand as opponents fold.
        let bb = 10_000u64;
        let mut cfg = GameConfig::new_uniform(3, 0, 30_000, bb);
        cfg.starting_stacks = vec![20 * bb, 30 * bb, 40 * bb];
        let g = GameState::new_hand(cfg, 7, 0);
        // Caps: A=min(20, max(30,40))=20, B=min(30, max(20,40))=30,
        // C=min(40, max(20,30))=30. Frozen at hand start.
        assert_eq!(g.eff_stack_cap_at_hand_start, vec![20 * bb, 30 * bb, 30 * bb]);
        // Cap is read-only; it doesn't change as stacks shrink or seats fold.
        let mut g2 = g.clone();
        g2.folded[2] = true;
        g2.stacks[0] = 12 * bb;
        g2.stacks[1] = 22 * bb;
        assert_eq!(
            g2.eff_stack_cap_at_hand_start,
            vec![20 * bb, 30 * bb, 30 * bb],
            "cap must remain frozen mid-hand"
        );
    }

    #[test]
    fn min_raise_chips_zero_when_no_opp_can_match() {
        // Setup where the actor's min legal raise total exceeds the
        // largest reachable total of any alive opponent. Use heads-up
        // with extreme stack disparity. Hero (88bb) faces a tiny opp
        // (2bb stack). Min raise must go to 0 so Raise gate is illegal.
        let bb = 10_000u64;
        let mut cfg = GameConfig::new_uniform(2, 0, 0, bb); // no antes for clarity
        cfg.starting_stacks = vec![88 * bb, 2 * bb];
        let mut g = GameState::new_hand(cfg, 1, 0);
        // Force a minimal preexisting bet that triggers a min-raise total
        // larger than the short opp can match. Simplest: hero is actor,
        // with no facing bet, min_total = bb = 10000. Opp's reachable =
        // 0 + 2*bb = 20000. min_total <= opp_cap, so raise should be legal.
        // To make the cap bind, set a partial state by hand: pretend opp
        // already committed 2*bb (their entire stack as ante-equiv);
        // their reachable now = 2*bb + 0 = 20000, while min_total stays
        // at 10000 with bet_to_call=0. Still legal.
        // Instead: set bet_to_call = 5*bb (opp open) and last_raise_size = 5*bb.
        // But opp's stack is only 2*bb so they couldn't have made that bet.
        // Simulate by directly mutating state for the test.
        g.bet_to_call = 5 * bb;
        g.last_raise_size = 5 * bb;
        g.street_commit[0] = 0;
        g.street_commit[1] = 5 * bb; // hypothetical opp bet
        g.stacks[1] = 0; // opp all-in
        g.all_in[1] = true;
        // min_bet_total = bet_to_call + last_raise_size = 10*bb.
        // Hero's stack is 88*bb so they can afford it. But opp_cap (only
        // alive opp from hero's POV is opp who is all-in for 5*bb): 5*bb
        // + 0 = 5*bb. min_total (10*bb) > opp_cap (5*bb), so min_raise = 0.
        // Note: actor==0 (hero); folded[0]=false. opp(1) is all_in but
        // not folded, so it counts in max_other_reachable_total.
        g.actor = Some(0);
        assert_eq!(g.min_raise_chips(), 0);
        assert_eq!(g.max_raise_chips(), 0);
    }

    #[test]
    fn new_hand_posts_antes_and_reveals_flop() {
        let g = GameState::new_hand(default_config(), 42, 0);
        assert_eq!(g.pot, 6 * 30000);
        assert!(g.stacks.iter().all(|&s| s == 170000));
        assert_eq!(g.board_a.len(), 3);
        assert_eq!(g.board_b.len(), 3);
        assert_eq!(g.street, Street::Flop);
        assert!(g.actor.is_some());
        // First to act is seat left of button (button=0 → actor=1).
        assert_eq!(g.actor, Some(1));
        for h in &g.hole_cards {
            assert_eq!(h.len(), 5);
        }
    }

    #[test]
    fn reproducibility_from_seed() {
        let a = GameState::new_hand(default_config(), 12345, 2);
        let b = GameState::new_hand(default_config(), 12345, 2);
        assert_eq!(a.hole_cards, b.hole_cards);
        assert_eq!(a.full_board_a, b.full_board_a);
        assert_eq!(a.full_board_b, b.full_board_b);
    }

    #[test]
    fn check_around_advances_street() {
        let mut g = GameState::new_hand(default_config(), 1, 0);
        // 6 checks → street closes (Flop → Turn).
        for _ in 0..6 {
            assert_eq!(g.street, Street::Flop);
            g.apply(Action::CheckCall);
        }
        assert_eq!(g.street, Street::Turn);
        assert_eq!(g.board_a.len(), 4);
        assert_eq!(g.board_b.len(), 4);
        assert_eq!(g.pot, 180000);
    }

    #[test]
    fn check_check_check_to_showdown() {
        let mut g = GameState::new_hand(default_config(), 1, 0);
        for _ in 0..18 {
            if g.is_terminal() {
                break;
            }
            g.apply(Action::CheckCall);
        }
        assert!(g.is_terminal());
        let p = g.payouts();
        assert_eq!(p.len(), 6);
        assert_eq!(p.iter().sum::<i64>(), 0);
    }

    #[test]
    fn one_bettor_all_fold_terminates() {
        let mut g = GameState::new_hand(default_config(), 5, 0);
        g.apply(Action::BetPct50);
        for _ in 0..5 {
            g.apply(Action::Fold);
        }
        assert!(g.is_terminal());
        let p = g.payouts();
        assert_eq!(p.iter().sum::<i64>(), 0);
        assert!(p[1] > 0);
        for &seat in &[0usize, 2, 3, 4, 5] {
            assert_eq!(p[seat], -30000);
        }
    }

    #[test]
    fn min_bet_first_bet_is_one_bb() {
        let g = GameState::new_hand(default_config(), 0, 0);
        assert_eq!(g.min_bet_total(), 10000);
    }

    #[test]
    fn pl_cap_initial_bet_equals_pot_capped_by_opp_stack() {
        // Default 6-handed 20bb stacks (200000 chips) post 3bb antes
        // → 170000 stack each, pot 180000. PL formula gives 180000 but
        // each opponent only has 170000 to put in, so the
        // effective-stack cap binds and max_bet_total is 170000.
        let g = GameState::new_hand(default_config(), 0, 0);
        let pl_formula = g.pot;
        let opp_cap = g.max_other_reachable_total();
        assert!(
            opp_cap < pl_formula,
            "scenario assumes opp cap binds: opp_cap={opp_cap}, pl={pl_formula}"
        );
        assert_eq!(g.max_bet_total(), opp_cap);
    }

    #[test]
    fn pl_cap_after_bet_matches_min_of_pl_and_opp_stack() {
        // After a 50%-pot bet, the PL formula returns 450000 but no
        // alive opponent has more than 170000 in (commit + stack).
        // max_bet_total is the min — the new effective-stack cap.
        let mut g = GameState::new_hand(default_config(), 0, 0);
        g.apply(Action::BetPct50);
        let actor = g.actor.unwrap();
        let to_call = g.bet_to_call - g.street_commit[actor];
        let pl_formula = g.bet_to_call + g.pot + to_call;
        let opp_cap = g.max_other_reachable_total();
        assert_eq!(g.max_bet_total(), pl_formula.min(opp_cap));
    }

    #[test]
    fn min_raise_enforcement_masks_undersized_sizings() {
        let mut g = GameState::new_hand(default_config(), 0, 0);
        g.apply(Action::BetPct50);
        let mask = g.legal_action_mask();
        assert!(mask[Action::CheckCall as usize]);
        assert!(mask[Action::Fold as usize]);
        // Sanity: min_bet_total is now above the tiny 10% sizing, which
        // forces clamping and produces duplicates with other legal actions.
        let mbt = g.min_bet_total();
        let btc = g.bet_to_call;
        assert!(mbt > btc);
    }

    #[test]
    fn short_open_shove_below_min_bet_legal() {
        // Custom low-stack config where shove is below 1bb min.
        // Each seat behind 50 after ante. Pot = 600. Min bet = 100.
        // Sub-1bb open shove must be a legal action — any all-in is
        // legal regardless of bet floor.
        let cfg = GameConfig::new_uniform(2, 350, 300, 100);
        let g = GameState::new_hand(cfg, 0, 0);
        let mask = g.legal_action_mask();
        assert!(mask[Action::AllIn as usize]);
    }

    #[test]
    fn short_open_shove_preserves_min_raise_floor() {
        // Mixed stacks: deep seat 0, short seat 1. After 300-chip
        // ante: seat 0 has 1000 behind, seat 1 has 50. button=0 so
        // first_to_act_postflop = seat 1 (the short). Sub-1bb open
        // shove must advance bet_to_call but NOT reset
        // last_raise_size — same short-shove path as facing-bet shorts.
        let mut cfg = GameConfig::new_uniform(2, 0, 300, 100);
        cfg.starting_stacks = vec![1300, 350];
        let mut g = GameState::new_hand(cfg, 0, 0);
        assert_eq!(g.actor, Some(1));
        let stack_before = g.stacks[1];
        assert_eq!(stack_before, 50);
        g.apply(Action::AllIn);
        assert_eq!(g.bet_to_call, 50);
        assert_eq!(g.last_raise_size, 100, "1bb floor preserved");
        assert!(!g.last_aggression_was_full_raise);
    }

    #[test]
    fn short_shove_does_not_reopen_raising_for_previously_acted_seats() {
        // 3-seat mixed stacks. Button=2 → flop order is 0, 1, 2.
        // After 300-chip ante: seat0=19_700, seat1=800, seat2=49_700. Pot=900.
        // Seat 0 bets B50 (=450). btc=450, last_raise_size=450, min-raise
        // total=900. Seat 1 shoves remaining 800: total_commit=800>450 btc
        // but <900 min — short raise. Seat 2 (not yet acted) still has raise
        // options. Then seat 2 CheckCalls and action returns to seat 0,
        // who has already acted this street and faces the non-full short
        // shove — mask must be exactly {Fold, CheckCall}.
        let mut cfg = GameConfig::new_uniform(3, 0, 300, 100);
        cfg.starting_stacks = vec![20_000u64, 1_100u64, 50_000u64];
        let mut g = GameState::new_hand(cfg, 42, 2);
        // Seat 0 opens B50 (half-pot after ante: 0.5 * 900 = 450).
        assert_eq!(g.actor, Some(0));
        let btc_before = g.bet_to_call;
        g.apply(Action::BetPct50);
        assert_eq!(g.bet_to_call, 450);
        assert_eq!(g.last_raise_size, 450);
        assert!(g.last_aggression_was_full_raise);
        assert_eq!(btc_before, 0);
        // Seat 1 (short) is next. Seat 1 has 800 behind ante. CheckCall
        // requires 450; let's have seat 1 shove (AllIn for 800, all chips).
        assert_eq!(g.actor, Some(1));
        g.apply(Action::AllIn);
        // Seat 1's shove advances bet_to_call to 800 but does NOT reopen —
        // delta 350 < 450 floor.
        assert_eq!(g.bet_to_call, 800);
        assert_eq!(g.last_raise_size, 450, "floor preserved after short shove");
        assert!(
            !g.last_aggression_was_full_raise,
            "short shove must flag non-full aggression"
        );
        // Seat 2 (not yet acted this street) faces the short shove. Raising
        // is still legal for seat 2 because the guard applies only to
        // already-acted seats.
        assert_eq!(g.actor, Some(2));
        let mask2 = g.legal_action_mask();
        assert!(mask2[Action::Fold as usize]);
        assert!(mask2[Action::CheckCall as usize]);
        assert!(
            mask2.iter().skip(2).any(|&b| b),
            "seat 2 (not yet acted) must still have raise/AllIn options"
        );
        // Have seat 2 just call — no raise.
        g.apply(Action::CheckCall);
        // Now action returns to seat 0, who has acted this street and faces
        // the short shove. Seat 0's mask must be exactly {Fold, CheckCall}.
        assert_eq!(g.actor, Some(0));
        let mask0 = g.legal_action_mask();
        assert!(mask0[Action::Fold as usize]);
        assert!(mask0[Action::CheckCall as usize]);
        for a in [
            Action::BetPct10,
            Action::BetPct25,
            Action::BetPct50,
            Action::BetPct75,
            Action::BetPct100,
            Action::AllIn,
        ] {
            assert!(
                !mask0[a as usize],
                "already-acted seat 0 must not have {:?} after short shove",
                a
            );
        }
        // Continuous-raise entry point must also honor the lockout: bounds
        // report 0 and apply_raise_chips rejects any amount.
        assert_eq!(g.min_raise_chips(), 0, "lockout must zero min_raise_chips");
        assert_eq!(g.max_raise_chips(), 0, "lockout must zero max_raise_chips");
        assert_eq!(
            g.apply_raise_chips(1),
            Err(StudyError::InvalidAmount),
            "lockout must reject apply_raise_chips"
        );
    }

    #[test]
    fn deep_actor_can_bet_to_cover_sub_1bb_short() {
        // HU bomb-pot, flop. Cover-short regime: deep BB facing covered
        // sub-1BB SB. After 300-chip ante: SB=75 (sub-1BB), BB=1700.
        // BB's 1 BB lead-bet floor (100) exceeds SB's effective stack
        // reach (75). The cover-short clamp must collapse min_raise to
        // 75 (matching max_raise) so BB can make the covering bet.
        // button=0 → seat 1 (BB) acts first postflop.
        let mut cfg = GameConfig::new_uniform(2, 0, 300, 100);
        cfg.starting_stacks = vec![375, 2000];
        let mut g = GameState::new_hand(cfg, 0, 0);
        assert_eq!(g.actor, Some(1), "BB acts first postflop in HU");
        assert_eq!(g.stacks[0], 75, "SB sub-1BB after ante");
        assert_eq!(g.stacks[1], 1700, "BB deep after ante");
        assert_eq!(
            g.min_raise_chips(),
            75,
            "cover-short clamp collapses 1bb floor to opp's reach"
        );
        assert_eq!(
            g.max_raise_chips(),
            75,
            "max collapses to short opp's reach"
        );
        // BB is NOT going all-in — covering bet flows through Raise path.
        let mask = g.legal_action_mask();
        assert!(mask[Action::CheckCall as usize]);
        assert!(
            !mask[Action::AllIn as usize],
            "BB's 75-chip covering bet leaves 1625 behind — not all-in"
        );
        // Apply the covering bet via the continuous-raise entry point.
        g.apply_raise_chips(75).unwrap();
        assert_eq!(g.bet_to_call, 75);
        assert_eq!(g.street_commit[1], 75);
        assert_eq!(g.stacks[1], 1625, "BB still has 1625 left");
        // SB is next; facing 75 with stack 75 — call clamps to full stack.
        assert_eq!(g.actor, Some(0));
        let mask0 = g.legal_action_mask();
        assert!(mask0[Action::Fold as usize]);
        assert!(mask0[Action::CheckCall as usize]);
    }

    #[test]
    fn deep_actor_can_cover_after_3way_short_shove() {
        // 3-way bomb-pot, flop. SB shoves all-in for a full open. BB's
        // min-raise floor (= 2 × shove) exceeds BB's stack — but the
        // deepest non-allin opp (BTN) can be covered by less than BB's
        // stack. Cover-short clamp must collapse min/max to BTN's reach.
        // button=2 → flop order 0 (SB), 1 (BB), 2 (BTN, hero).
        // Pre-ante stacks scaled to bb=100, ante=300:
        //   SB=1150 (= $230), BB=1450 (= $290), BTN=1350 (= $270).
        let mut cfg = GameConfig::new_uniform(3, 0, 300, 100);
        cfg.starting_stacks = vec![1150, 1450, 1350];
        let mut g = GameState::new_hand(cfg, 0, 2);
        // Post-ante: stacks = [850, 1150, 1050], pot = 900.
        assert_eq!(g.actor, Some(0), "SB acts first postflop in 3-way (button=2)");
        assert_eq!(g.stacks, vec![850, 1150, 1050]);
        // SB shoves all-in for 850 (full open: 8.5 BB ≥ 1 BB floor).
        g.apply(Action::AllIn);
        assert_eq!(g.bet_to_call, 850);
        assert_eq!(g.last_raise_size, 850, "full open resets the floor");
        assert_eq!(g.actor, Some(1), "BB acts next");
        // BB's min_total = 850 + 850 = 1700, > BB's stack (1150).
        // max_other_reachable = max(SB=850, BTN=0+1050) = 1050.
        // cap_delta = 1050; clamped = min(1700, 1050) = 1050; 1050 ≤ 1150.
        assert_eq!(
            g.min_raise_chips(),
            1050,
            "min must collapse to BTN's reach despite delta > stack"
        );
        assert_eq!(
            g.max_raise_chips(),
            1050,
            "max collapses to BTN's reach via max_other_reachable"
        );
        // BB applies the covering raise — 100 chips left after.
        g.apply_raise_chips(1050).unwrap();
        assert_eq!(g.bet_to_call, 1050);
        assert_eq!(g.street_commit[1], 1050);
        assert_eq!(g.stacks[1], 100, "BB has 100 chips left after cover");
        // BTN is next, facing 1050 with stack 1050 — call clamps to all-in.
        assert_eq!(g.actor, Some(2));
        let mask = g.legal_action_mask();
        assert!(mask[Action::Fold as usize]);
        assert!(mask[Action::CheckCall as usize]);
    }

    #[test]
    fn bet_pct_100_masked_as_allin_dupe_at_20bb_flop() {
        // Flop fresh 20bb/3bb: pot=180000, stack=170000. BetPct100 targets
        // 180000 but clamps to stack=170000, which equals AllIn exactly.
        // Duplicate-masking must hide BetPct100 so the action space at
        // this state is { Fold, CheckCall, B10(18000), B25(45000), B50(90000),
        // B75(135000), AllIn(170000) } — AllIn is the only "pot-or-bigger"
        // option the model sees.
        let g = GameState::new_hand(default_config(), 0, 0);
        assert_eq!(g.pot, 180000);
        let actor = g.actor.unwrap();
        assert_eq!(g.stacks[actor], 170000);
        let mask = g.legal_action_mask();
        assert!(!mask[Action::BetPct100 as usize], "B100 should be masked as AllIn dupe");
        assert!(mask[Action::AllIn as usize]);
        for a in [
            Action::BetPct10,
            Action::BetPct25,
            Action::BetPct50,
            Action::BetPct75,
        ] {
            assert!(mask[a as usize], "{a:?} should be legal");
        }
    }

    #[test]
    fn bet_pct_100_distinct_when_pl_cap_masks_allin() {
        // Deeper stacks (3000 starting, 300 ante, 6-max): flop-open has
        // pot=1800, stack-behind=2700. PL cap = pot = 1800, so the shove
        // (2700) exceeds the cap and AllIn is masked. BetPct100 targets
        // 1800 (within stack) and survives as the distinct pot-bet
        // sizing. This is the state where slot 6 earns its keep.
        let cfg = GameConfig::new_uniform(6, 3000, 300, 100);
        let g = GameState::new_hand(cfg, 0, 0);
        let actor = g.actor.unwrap();
        assert_eq!(g.pot, 1800);
        assert_eq!(g.stacks[actor], 2700);
        let mask = g.legal_action_mask();
        assert!(mask[Action::BetPct100 as usize], "B100 should be legal");
        assert!(!mask[Action::AllIn as usize], "AllIn should be masked by PL cap");
        let chips = g.action_to_chips(Action::BetPct100).unwrap();
        assert_eq!(chips, 1800);
    }

    #[test]
    fn pl_cap_vs_stack_deep_stack_shove_masked() {
        // Deep stack where stack > PL cap → AllIn masked, sub-pot sizings legal.
        let cfg = GameConfig::new_uniform(2, 100_000, 300, 100);
        let g = GameState::new_hand(cfg, 0, 0);
        // Pot = 600; stack behind = 99_700. Max bet = 600 (pot). Shove = 99_700
        // total, way above cap → masked.
        let mask = g.legal_action_mask();
        assert!(!mask[Action::AllIn as usize]);
        assert!(mask[Action::BetPct75 as usize]);
    }

    #[test]
    fn all_in_and_called_runs_out_to_showdown() {
        // Heads-up. Seat 1 shoves, seat 0 calls → both all-in → run out.
        let cfg = GameConfig::new_uniform(2, 2000, 300, 100);
        let mut g = GameState::new_hand(cfg, 0, 0);
        // Pot after antes = 600. Seat 1 bet_to_call = 0. PL cap = 600.
        // BetPct75 = 450. Then seat 0 faces 450 to call. Seat 0 can call or
        // raise. Let's have seat 0 reraise all-in.
        g.apply(Action::BetPct75); // seat 1 bets 450
        g.apply(Action::AllIn); // seat 0 shoves
        g.apply(Action::CheckCall); // seat 1 calls the shove for remaining stack
        assert!(g.is_terminal());
        assert_eq!(g.board_a.len(), 5);
        assert_eq!(g.board_b.len(), 5);
        let p = g.payouts();
        assert_eq!(p.iter().sum::<i64>(), 0);
    }

    #[test]
    fn chip_conservation_random_legal_hand() {
        // Simulate a hand with scripted actions, verify sum of payouts == 0.
        let mut g = GameState::new_hand(default_config(), 99, 2);
        let mut steps = 0;
        while !g.is_terminal() && steps < 200 {
            let mask = g.legal_action_mask();
            let pick = (0..NUM_ACTIONS as u8)
                .find(|&i| mask[i as usize])
                .expect("some action must be legal");
            g.apply(Action::from_index(pick).unwrap());
            steps += 1;
        }
        assert!(g.is_terminal());
        assert_eq!(g.payouts().iter().sum::<i64>(), 0);
    }

    #[test]
    fn fold_action_illegal_when_no_bet() {
        let g = GameState::new_hand(default_config(), 0, 0);
        let mask = g.legal_action_mask();
        assert!(!mask[Action::Fold as usize]);
        assert!(mask[Action::CheckCall as usize]);
    }

    // ---- Study-mode tests ----

    fn cards(idxs: &[u8]) -> Vec<Card> {
        idxs.iter().map(|&i| Card::from_index(i)).collect()
    }

    fn hole5(idxs: [u8; 5]) -> [Card; 5] {
        let v = cards(&idxs);
        [v[0], v[1], v[2], v[3], v[4]]
    }
    fn flop3(idxs: [u8; 3]) -> [Card; 3] {
        let v = cards(&idxs);
        [v[0], v[1], v[2]]
    }

    #[test]
    fn new_study_rejects_duplicate_cards() {
        // Put Ac (index 48) in both hero hole and flop A → DuplicateCard.
        let cfg = default_config();
        let hero_hole = hole5([48, 0, 1, 2, 3]);
        let flop_a = flop3([48, 4, 5]); // duplicates Ac
        let flop_b = flop3([6, 7, 8]);
        let res = GameState::new_study(cfg, 0, 1, hero_hole, flop_a, flop_b);
        assert_eq!(res.err(), Some(StudyError::DuplicateCard));
    }

    #[test]
    fn new_study_legal_mask_matches_production_flop() {
        // Study hand at flop should offer the same legal action set as a
        // fresh production hand (same pot, same stacks, same board-length=3,
        // first-to-act = seat 1 with button 0).
        let cfg = default_config();
        let hero_hole = hole5([0, 1, 2, 3, 4]);
        let flop_a = flop3([5, 6, 7]);
        let flop_b = flop3([8, 9, 10]);
        let study = GameState::new_study(cfg.clone(), 0, 1, hero_hole, flop_a, flop_b).unwrap();
        assert_eq!(study.street, Street::Flop);
        assert_eq!(study.pot, 6 * 30000);
        assert!(study.stacks.iter().all(|&s| s == 170000));
        assert_eq!(study.actor, Some(1));
        let s_mask = study.legal_action_mask();
        let prod = GameState::new_hand(cfg, 0, 0);
        let p_mask = prod.legal_action_mask();
        assert_eq!(s_mask, p_mask);
    }

    #[test]
    fn study_mode_flop_check_around_halts_at_turn_awaiting() {
        // 6 checks on the flop must close the round and leave the hand
        // awaiting turn cards — not auto-advance.
        let cfg = default_config();
        let hero_hole = hole5([0, 1, 2, 3, 4]);
        let flop_a = flop3([5, 6, 7]);
        let flop_b = flop3([8, 9, 10]);
        let mut g = GameState::new_study(cfg, 0, 1, hero_hole, flop_a, flop_b).unwrap();
        for _ in 0..6 {
            g.apply(Action::CheckCall);
        }
        assert_eq!(g.awaiting_next_street, Some(Street::Turn));
        assert!(g.actor.is_none());
        assert_eq!(g.board_a.len(), 3);
        assert_eq!(g.street, Street::Flop); // unchanged until set_turn
        assert!(g.study_terminal.is_none());
    }

    #[test]
    fn set_turn_rejects_when_not_awaiting() {
        let cfg = default_config();
        let hero_hole = hole5([0, 1, 2, 3, 4]);
        let flop_a = flop3([5, 6, 7]);
        let flop_b = flop3([8, 9, 10]);
        let mut g = GameState::new_study(cfg, 0, 1, hero_hole, flop_a, flop_b).unwrap();
        // Directly call set_turn before flop action closes — should error.
        let res = g.set_turn(Card::from_index(11), Card::from_index(12));
        assert_eq!(res, Err(StudyError::WrongState));
    }

    #[test]
    fn set_turn_rejects_duplicate_card() {
        let cfg = default_config();
        let hero_hole = hole5([0, 1, 2, 3, 4]);
        let flop_a = flop3([5, 6, 7]);
        let flop_b = flop3([8, 9, 10]);
        let mut g = GameState::new_study(cfg, 0, 1, hero_hole, flop_a, flop_b).unwrap();
        for _ in 0..6 {
            g.apply(Action::CheckCall);
        }
        // Try to set turn A = hero hole card 0 → duplicate.
        let res = g.set_turn(Card::from_index(0), Card::from_index(11));
        assert_eq!(res, Err(StudyError::DuplicateCard));
        // Or turn B duplicates flop A.
        let res = g.set_turn(Card::from_index(11), Card::from_index(5));
        assert_eq!(res, Err(StudyError::DuplicateCard));
        // Or card_a == card_b.
        let res = g.set_turn(Card::from_index(11), Card::from_index(11));
        assert_eq!(res, Err(StudyError::DuplicateCard));
    }

    #[test]
    fn study_fold_out_computes_uncontested_payout() {
        // Seat 1 bets, all others fold → fold-out. Uncontested pot to seat 1.
        let cfg = default_config();
        let hero_hole = hole5([0, 1, 2, 3, 4]);
        let flop_a = flop3([5, 6, 7]);
        let flop_b = flop3([8, 9, 10]);
        let mut g = GameState::new_study(cfg, 0, 1, hero_hole, flop_a, flop_b).unwrap();
        g.apply(Action::BetPct50);
        for _ in 0..5 {
            g.apply(Action::Fold);
        }
        assert!(g.is_terminal());
        assert_eq!(g.study_terminal, Some(StudyTerminal::FoldOut));
        let p = g.payouts();
        assert_eq!(p.iter().sum::<i64>(), 0);
        assert!(p[1] > 0);
        for &seat in &[0usize, 2, 3, 4, 5] {
            assert_eq!(p[seat], -30000);
        }
    }

    #[test]
    fn study_run_out_sets_run_out_terminal_no_payout() {
        // Heads-up study hand: both shove on the flop and call → both all-in
        // but cards remain undealt (turn unknown). Study mode must flag
        // RunOut and return zero payouts.
        let cfg = GameConfig::new_uniform(2, 2000, 300, 100);
        let hero_hole = hole5([0, 1, 2, 3, 4]);
        let flop_a = flop3([5, 6, 7]);
        let flop_b = flop3([8, 9, 10]);
        let mut g = GameState::new_study(cfg, 0, 1, hero_hole, flop_a, flop_b).unwrap();
        g.apply(Action::BetPct75); // seat 1 bets 450
        g.apply(Action::AllIn); // seat 0 shoves
        g.apply(Action::CheckCall); // seat 1 calls → both all-in
        assert!(g.is_terminal());
        assert_eq!(g.study_terminal, Some(StudyTerminal::RunOut));
        assert_eq!(g.payouts(), vec![0i64, 0]);
        assert_eq!(g.board_a.len(), 3); // turn/river were never dealt
    }

    #[test]
    fn study_showdown_after_river_check_through_returns_zeros() {
        // Flop check around → set_turn → turn check around → set_river →
        // river check around → Showdown terminal, zero payouts.
        let cfg = default_config();
        let hero_hole = hole5([0, 1, 2, 3, 4]);
        let flop_a = flop3([5, 6, 7]);
        let flop_b = flop3([8, 9, 10]);
        let mut g = GameState::new_study(cfg, 0, 1, hero_hole, flop_a, flop_b).unwrap();
        for _ in 0..6 {
            g.apply(Action::CheckCall);
        }
        g.set_turn(Card::from_index(11), Card::from_index(12)).unwrap();
        for _ in 0..6 {
            g.apply(Action::CheckCall);
        }
        g.set_river(Card::from_index(13), Card::from_index(14)).unwrap();
        for _ in 0..6 {
            g.apply(Action::CheckCall);
        }
        assert!(g.is_terminal());
        assert_eq!(g.study_terminal, Some(StudyTerminal::Showdown));
        assert_eq!(g.street, Street::Showdown);
        assert_eq!(g.payouts().iter().sum::<i64>(), 0);
        assert!(g.payouts().iter().all(|&x| x == 0));
    }

    #[test]
    fn set_turn_skips_folded_seat_when_setting_first_actor() {
        // 3-seat study hand. Button = 0, SB = 1, BB = 2.
        // On the flop: first-to-act is SB (seat 1). SB folds. Next actor is
        // BB (seat 2). BB checks. Then button (seat 0) checks → round closes.
        // After set_turn, first-to-act must be **BB (seat 2)** — not SB,
        // which is folded. This proves the eligibility walk skips folded seats.
        let cfg = GameConfig::new_uniform(3, 2000, 300, 100);
        let hero_hole = hole5([0, 1, 2, 3, 4]);
        let flop_a = flop3([5, 6, 7]);
        let flop_b = flop3([8, 9, 10]);
        let mut g = GameState::new_study(cfg, 0, 2, hero_hole, flop_a, flop_b).unwrap();
        assert_eq!(g.actor, Some(1));
        // We need a bet for Fold to be legal. Have BB/BTN bet/call scenario?
        // Easier: have SB bet, then BTN fold, then BB fold — but that's fold-out.
        // Instead: let SB check, BB bet, BTN fold, SB fold — two folds, single
        // survivor. Again fold-out. We need 2+ survivors after the flop.
        //
        // Scenario: SB checks, BB bets, BTN calls, SB folds → round closes with
        // BB + BTN left. On the turn, first-to-act is BB (seat 2), skipping
        // folded SB (seat 1).
        g.apply(Action::CheckCall); // SB checks
        assert_eq!(g.actor, Some(2));
        g.apply(Action::BetPct50); // BB bets
        assert_eq!(g.actor, Some(0));
        g.apply(Action::CheckCall); // BTN calls
        assert_eq!(g.actor, Some(1));
        g.apply(Action::Fold); // SB folds
        assert_eq!(g.awaiting_next_street, Some(Street::Turn));
        g.set_turn(Card::from_index(11), Card::from_index(12)).unwrap();
        // First-to-act on turn: skip SB (folded), so actor = BB (seat 2).
        assert_eq!(g.actor, Some(2));
    }

    #[test]
    fn hero_hole_cards_are_five_unique() {
        let g = GameState::new_hand(default_config(), 7, 0);
        // Every seat has 5 unique cards.
        for h in &g.hole_cards {
            let mut seen = [false; 52];
            for c in h {
                let i = c.index() as usize;
                assert!(!seen[i]);
                seen[i] = true;
            }
        }
        // And no overlap across seats + boards.
        let mut seen = [false; 52];
        for h in &g.hole_cards {
            for c in h {
                assert!(!seen[c.index() as usize]);
                seen[c.index() as usize] = true;
            }
        }
        for c in &g.full_board_a {
            assert!(!seen[c.index() as usize]);
            seen[c.index() as usize] = true;
        }
        for c in &g.full_board_b {
            assert!(!seen[c.index() as usize]);
            seen[c.index() as usize] = true;
        }
        // 6*5 + 5 + 5 = 40 cards, within deck.
        let count = seen.iter().filter(|&&x| x).count();
        assert_eq!(count, 40);
    }

    // --- payouts_ev -----------------------------------------------------

    /// Drive a hand through a no-bet river showdown so that action closes
    /// on the river and all community cards are dealt.
    fn play_check_check_to_showdown(seed: u64) -> GameState {
        let mut g = GameState::new_hand(default_config(), seed, 0);
        // Everyone checks every street.
        while !g.is_terminal() {
            let mask = g.legal_action_mask();
            assert!(mask[Action::CheckCall as usize]);
            g.apply(Action::CheckCall);
        }
        g
    }

    #[test]
    fn payouts_ev_delegates_on_river_close() {
        let g = play_check_check_to_showdown(12345);
        assert_eq!(g.action_close_board_len, Some(5));
        let realized = g.payouts();
        let ev = g.payouts_ev(100, 9999);
        assert_eq!(ev, realized, "river-close: EV must equal realized");
    }

    #[test]
    fn payouts_ev_delegates_on_fold_out() {
        // First actor bets to open action; everyone else folds to produce a
        // single-survivor fold-out. Fold is illegal while bet_to_call == 0,
        // so an opening bet is required before folds become legal.
        let mut g = GameState::new_hand(default_config(), 42, 0);
        g.apply(Action::BetPct50);
        while !g.is_terminal() {
            g.apply(Action::Fold);
        }
        let alive = (0..g.config.num_seats).filter(|&i| !g.folded[i]).count();
        assert_eq!(alive, 1);
        let realized = g.payouts();
        let ev = g.payouts_ev(500, 7);
        assert_eq!(ev, realized, "fold-out: EV must equal realized (card-agnostic)");
    }

    #[test]
    fn payouts_ev_zero_samples_delegates() {
        let g = play_check_check_to_showdown(7);
        let ev = g.payouts_ev(0, 0);
        assert_eq!(ev, g.payouts());
    }

    /// Drive a flop all-in: first-to-act shoves, all remaining call or
    /// fold such that ≥2 seats are all-in. After this, engine auto-runs
    /// out turn+river internally and terminal is reached with
    /// `action_close_board_len == Some(3)`.
    fn play_flop_all_in(seed: u64) -> GameState {
        let mut g = GameState::new_hand(default_config(), seed, 0);
        // First to act: shove.
        assert!(g.legal_action_mask()[Action::AllIn as usize]);
        g.apply(Action::AllIn);
        // Everyone else: call (so we get 6-way all-in on flop).
        while !g.is_terminal() {
            let mask = g.legal_action_mask();
            if mask[Action::CheckCall as usize] {
                g.apply(Action::CheckCall);
            } else {
                g.apply(Action::Fold);
            }
        }
        g
    }

    #[test]
    fn payouts_ev_deterministic_same_seed() {
        let g = play_flop_all_in(31415);
        assert_eq!(g.action_close_board_len, Some(3));
        let ev1 = g.payouts_ev(200, 2024);
        let ev2 = g.payouts_ev(200, 2024);
        assert_eq!(ev1, ev2, "same seed must produce bitwise-identical EV");
    }

    #[test]
    fn payouts_ev_converges_on_flop_allin() {
        // Flop all-in: EV marginalises turn+river on both boards. Two
        // large-sample runs with different seeds should land very close
        // to each other (variance ~ 1/sqrt(N)). Tolerance scaled to
        // chip units; default config has starting_stack=200000.
        let g = play_flop_all_in(99);
        assert_eq!(g.action_close_board_len, Some(3));
        let n = g.config.num_seats;
        let ev_a = g.payouts_ev(2000, 111);
        let ev_b = g.payouts_ev(2000, 222);
        // Zero-sum invariant (within rounding).
        let sum_a: i64 = ev_a.iter().sum();
        assert!(sum_a.abs() <= n as i64, "EV must be zero-sum, got {sum_a}");
        // Per-seat estimates close between independent seeds.
        for i in 0..n {
            let diff = (ev_a[i] - ev_b[i]).abs();
            assert!(
                diff <= 20000,
                "seat {i}: ev_a={} ev_b={} diff={} (tolerance 20000)",
                ev_a[i],
                ev_b[i],
                diff
            );
        }
    }

    fn play_four_way_flop_allin_unequal(seed: u64) -> GameState {
        // 4 seats with heterogeneous stacks — chosen so every seat can
        // go all-in on the flop given the pot-limit cap. Pot opens at
        // 12bb (4 × 3bb ante); after a pot-sized open from first actor,
        // subsequent seats can shove up to ~48bb as a single action.
        // Stacks [20, 25, 35, 45] bb produce commits [17, 22, 32, 42] bb
        // — four distinct side-pot layers, the structure under test.
        let bb = 10_000u64;
        let mut cfg = GameConfig::new_uniform(4, 0, 30_000, bb);
        cfg.starting_stacks = vec![20 * bb, 25 * bb, 35 * bb, 45 * bb];
        let mut g = GameState::new_hand(cfg, seed, 0);
        while !g.is_terminal() {
            let mask = g.legal_action_mask();
            if mask[Action::AllIn as usize] {
                g.apply(Action::AllIn);
            } else if mask[Action::BetPct100 as usize] {
                // First actor can't shove directly (stack > pot-limit cap);
                // a pot-sized bet escalates the cap enough for everyone
                // behind to fold all-in into the pot.
                g.apply(Action::BetPct100);
            } else if mask[Action::CheckCall as usize] {
                g.apply(Action::CheckCall);
            } else {
                g.apply(Action::Fold);
            }
        }
        g
    }

    #[test]
    fn payouts_ev_four_way_unequal_stack_flop_allin() {
        // Engine prerequisite for ClubGG-realistic training: 4-way
        // unequal-stack flop with shoves must zero-sum and produce
        // bitwise-identical EV for the same seed. Stacks
        // [20, 25, 35, 45]bb post-ante give [17, 22, 32, 42]bb. Under
        // the effective-stack cap the deepest seat (45bb) is bounded
        // at 32bb (third-deepest's stack) and retains its excess —
        // chips above the deepest opponent stack are uncontested by
        // construction. Three distinct side-pot layers result.
        let g = play_four_way_flop_allin_unequal(42);
        assert_eq!(
            g.action_close_board_len,
            Some(3),
            "flop must close all-in (action closed on flop)"
        );
        let n = g.config.num_seats;
        // The three smaller stacks must drain; the deepest seat
        // retains the difference between its starting stack and the
        // third-deepest opponent's starting stack (excess uncontested
        // chips kept by the cap).
        let mut starts: Vec<u64> = g.config.starting_stacks.clone();
        starts.sort();
        let deepest_start = *starts.last().unwrap();
        let third_deepest_start = starts[starts.len() - 2];
        let cap_excess = deepest_start - third_deepest_start;
        let total_drained: u64 = g.stacks.iter().filter(|&&s| s == 0).count() as u64;
        assert_eq!(
            total_drained, 3,
            "exactly three seats should be all-in; the deepest retains excess"
        );
        let deepest_seat = g
            .config
            .starting_stacks
            .iter()
            .position(|&s| s == deepest_start)
            .unwrap();
        assert_eq!(
            g.stacks[deepest_seat], cap_excess,
            "deepest seat retains its uncontested excess"
        );
        let total_pot: i64 = g.config.starting_stacks.iter().sum::<u64>() as i64;
        let ev = g.payouts_ev(2000, 111);
        // Zero-sum within chip-rounding slack.
        let sum: i64 = ev.iter().sum();
        assert!(
            sum.abs() <= (n * 2) as i64,
            "EV must be zero-sum, got {sum}"
        );
        // No seat can lose more than it contributed, nor win more than
        // the rest of the pot.
        for i in 0..n {
            let starting = g.config.starting_stacks[i] as i64;
            let loss = -ev[i];
            assert!(
                loss <= starting,
                "seat {i}: loss {loss} exceeds starting stack {starting}"
            );
            assert!(
                ev[i] <= total_pot - starting,
                "seat {i}: gain {} exceeds pot minus own contribution {}",
                ev[i],
                total_pot - starting
            );
        }
        // Deterministic MC: same seed → bitwise-identical EV.
        let ev2 = g.payouts_ev(2000, 111);
        assert_eq!(ev, ev2, "same seed must produce bitwise-identical EV");
    }

    #[test]
    fn apply_raise_chips_matches_discrete_bet_pct() {
        // Parity: apply_raise_chips(compute_sizing_chips(BetPctN)) must
        // produce the same GameState as apply_action(BetPctN) for all
        // game-logic fields (pot, stacks, commits, btc, last_raise_size,
        // aggression flag, acted flag, actor transitions). History label
        // diverges by design — continuous path uses BetPct100 as a
        // transitional sentinel until Phase 4 redesigns history encoding.
        for &pct in &[Action::BetPct25, Action::BetPct50, Action::BetPct75] {
            let cfg = GameConfig::new_uniform(4, 200_000, 30_000, 10_000);
            let mut g_disc = GameState::new_hand(cfg.clone(), 999, 1);
            let mut g_cont = GameState::new_hand(cfg, 999, 1);
            let mask = g_disc.legal_action_mask();
            assert!(mask[pct as usize], "pct {pct:?} must be legal on fresh flop");
            let actor = g_disc.actor.unwrap();
            let chips = g_disc.compute_sizing_chips(pct, actor);
            g_disc.apply(pct);
            g_cont
                .apply_raise_chips(chips)
                .expect("chips from compute_sizing_chips must be in range");

            assert_eq!(g_disc.pot, g_cont.pot, "pct={pct:?} pot mismatch");
            assert_eq!(g_disc.stacks, g_cont.stacks, "pct={pct:?} stacks mismatch");
            assert_eq!(
                g_disc.street_commit, g_cont.street_commit,
                "pct={pct:?} street_commit mismatch"
            );
            assert_eq!(
                g_disc.total_commit, g_cont.total_commit,
                "pct={pct:?} total_commit mismatch"
            );
            assert_eq!(
                g_disc.bet_to_call, g_cont.bet_to_call,
                "pct={pct:?} bet_to_call mismatch"
            );
            assert_eq!(
                g_disc.last_raise_size, g_cont.last_raise_size,
                "pct={pct:?} last_raise_size mismatch"
            );
            assert_eq!(
                g_disc.last_aggression_was_full_raise,
                g_cont.last_aggression_was_full_raise,
                "pct={pct:?} full-raise flag mismatch"
            );
            assert_eq!(
                g_disc.acted_this_street, g_cont.acted_this_street,
                "pct={pct:?} acted flags mismatch"
            );
            assert_eq!(g_disc.actor, g_cont.actor, "pct={pct:?} next actor mismatch");
            assert_eq!(
                g_disc.last_aggressor, g_cont.last_aggressor,
                "pct={pct:?} last_aggressor mismatch"
            );
            // Historical chip amounts should match even though the Action
            // label differs for the continuous path.
            assert_eq!(
                g_disc.history.last().unwrap().chips,
                g_cont.history.last().unwrap().chips,
                "pct={pct:?} history chips mismatch"
            );
        }
    }

    #[test]
    fn apply_raise_chips_rejects_out_of_range() {
        let cfg = GameConfig::new_uniform(3, 200_000, 30_000, 10_000);
        let mut g = GameState::new_hand(cfg, 7, 0);
        let min = g.min_raise_chips();
        let max = g.max_raise_chips();
        assert!(min > 0, "raise must be available on fresh flop");
        assert!(max >= min);
        assert_eq!(
            g.apply_raise_chips(min - 1),
            Err(StudyError::InvalidAmount),
            "below min must reject"
        );
        assert_eq!(
            g.apply_raise_chips(max + 1),
            Err(StudyError::InvalidAmount),
            "above max must reject"
        );
        // Exact bounds succeed.
        assert!(g.apply_raise_chips(min).is_ok());
    }
}
