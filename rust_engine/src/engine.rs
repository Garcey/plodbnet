//! Game state machine: deal, apply action, advance street, settle payouts.

use crate::actions::{Action, NUM_ACTIONS};
use crate::cards::{Card, Deck};
use crate::double_board::{double_board_payout, single_board_payout};
use crate::state::{
    ActionRecord, GameConfig, GameState, Street, StudyError, StudyTerminal, Variant,
};

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
        let hole_count = config.variant.hole_count();
        let num_boards = config.variant.num_boards();

        // Deal order (determinism contract): `hole_count` cards per seat,
        // seat 0 first, then full board A, then full board B (two-board
        // variants only).
        let mut hole_cards: Vec<Vec<Card>> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut h = Vec::with_capacity(hole_count);
            for _ in 0..hole_count {
                h.push(deck.deal_one());
            }
            hole_cards.push(h);
        }

        let mut full_board_a = [Card(0); 5];
        for c in full_board_a.iter_mut() {
            *c = deck.deal_one();
        }
        // Single-board variants leave board B as never-read sentinels.
        let mut full_board_b = [Card(0); 5];
        if num_boards == 2 {
            for c in full_board_b.iter_mut() {
                *c = deck.deal_one();
            }
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

        // Blinds (variants with a preflop round): posted LIVE into
        // street_commit after the dead antes. Posting is not an action —
        // no history record, `acted_this_street` stays false, so the BB
        // option (round can't close until the BB acts) falls out of the
        // existing round-close machinery. Short posts go all-in;
        // `bet_to_call` stays at the NOMINAL bb so callers owe the full
        // blind and side pots absorb any shortfall.
        let mut street_commit = vec![0u64; n];
        let mut bet_to_call = 0u64;
        let mut blind_seats: Option<(usize, usize)> = None;
        if config.variant.has_preflop() {
            let (sb_seat, bb_seat) = nlh_blind_seats(n, button, &folded);
            for (seat, amount) in [(sb_seat, config.sb), (bb_seat, config.bb)] {
                let paid = stacks[seat].min(amount);
                stacks[seat] -= paid;
                street_commit[seat] += paid;
                total_commit[seat] += paid;
                pot += paid;
                if stacks[seat] == 0 {
                    all_in[seat] = true;
                }
            }
            bet_to_call = config.bb;
            blind_seats = Some((sb_seat, bb_seat));
        }

        // Preflop variants reveal nothing until the first round closes;
        // bomb pots start with both flops exposed.
        let (street, board_a, board_b) = if config.variant.has_preflop() {
            (Street::Preflop, Vec::new(), Vec::new())
        } else {
            (
                Street::Flop,
                full_board_a[0..3].to_vec(),
                full_board_b[0..3].to_vec(),
            )
        };
        let acted_this_street = vec![false; n];
        let street_level_acted = vec![0u64; n];

        let eff_stack_cap_at_hand_start =
            compute_eff_stack_cap(&config.starting_stacks, &folded);

        let mut state = GameState {
            config,
            button,
            sb_seat: blind_seats.map(|(sb, _)| sb),
            bb_seat: blind_seats.map(|(_, bb)| bb),
            street,
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
            bet_to_call,
            last_raise_size: 0, // set below from bb
            last_aggression_was_full_raise: true,
            street_level_acted,
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

        state.actor = match blind_seats {
            Some((_, bb_seat)) => state.first_to_act_preflop(bb_seat),
            None => state.first_to_act_postflop(),
        };

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
        // Study mode is PLO5-double-board-only for now: its card-entry
        // surface (5-card hero hole, paired flops, set_turn/set_river
        // taking two cards) is dual-board-shaped. NLH study support is a
        // separate workstream — reject rather than half-behave.
        if config.variant != Variant::Plo5DoubleBomb {
            return Err(StudyError::WrongState);
        }
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
        let mut hole_cards: Vec<Vec<Card>> = Vec::with_capacity(n);
        for seat in 0..n {
            if seat == hero_seat {
                hole_cards.push(hero_hole.to_vec());
            } else {
                let mut h = Vec::with_capacity(5);
                for _ in 0..5 {
                    h.push(deck_order[next]);
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
        let street_level_acted = vec![0u64; n];

        let eff_stack_cap_at_hand_start =
            compute_eff_stack_cap(&config.starting_stacks, &folded);

        let mut state = GameState {
            config,
            button,
            sb_seat: None,
            bb_seat: None,
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
            street_level_acted,
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
        // Dual-board setter: NLH study (single board) uses the *_nlh
        // setters — two cards here would corrupt board B.
        if self.config.variant == Variant::NlhSingle {
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
        self.study_advance_street(expected);
        Ok(())
    }

    /// Shared tail of every study street setter: enter `street`, reset the
    /// per-street betting state, and seat the first postflop actor.
    fn study_advance_street(&mut self, street: Street) {
        self.street = street;
        let n = self.config.num_seats;
        self.street_commit = vec![0u64; n];
        self.bet_to_call = 0;
        self.last_raise_size = self.config.bb;
        self.last_aggression_was_full_raise = true;
        self.street_level_acted = vec![0u64; n];
        self.last_aggressor = None;
        self.acted_this_street = vec![false; n];
        self.awaiting_next_street = None;
        self.actor = self.first_to_act_postflop();
    }

    /// Deal an NLH study-mode hand at the PREFLOP with a user-supplied
    /// 2-card hero hole. Antes post dead, blinds post live (exactly as
    /// [`Self::new_hand`]); no board exists yet — the UI supplies streets
    /// via [`Self::set_flop_nlh`] / [`Self::set_turn_nlh`] /
    /// [`Self::set_river_nlh`] as each round closes. Non-hero seats get
    /// deterministic placeholder holes (never surfaced; non-fold terminals
    /// pay zeros), mirroring the PLO study contract.
    pub fn new_study_nlh(
        config: GameConfig,
        button: usize,
        hero_seat: usize,
        hero_hole: [Card; 2],
    ) -> Result<Self, StudyError> {
        if config.variant != Variant::NlhSingle {
            return Err(StudyError::WrongState);
        }
        let n = config.num_seats;
        if n < 2 || button >= n || hero_seat >= n {
            return Err(StudyError::SeatOutOfRange);
        }
        if config.starting_stacks.len() != n {
            return Err(StudyError::SeatOutOfRange);
        }

        let mut used = [false; 52];
        for c in hero_hole.iter() {
            let i = c.index() as usize;
            if i >= 52 || used[i] {
                return Err(StudyError::DuplicateCard);
            }
            used[i] = true;
        }

        // Deterministic placeholder holes from a hash of the user inputs
        // (same recipe as the PLO study constructor).
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hasher::write_u8(&mut hasher, button as u8);
        std::hash::Hasher::write_u8(&mut hasher, hero_seat as u8);
        for c in hero_hole.iter() {
            std::hash::Hasher::write_u8(&mut hasher, c.index());
        }
        let seed = std::hash::Hasher::finish(&hasher);
        let mut deck_order: Vec<Card> = (0..52u8)
            .filter(|&i| !used[i as usize])
            .map(Card::from_index)
            .collect();
        {
            use rand::seq::SliceRandom;
            use rand_chacha::ChaCha8Rng;
            use rand_chacha::rand_core::SeedableRng;
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            deck_order.shuffle(&mut rng);
        }
        let mut next = 0usize;
        let mut hole_cards: Vec<Vec<Card>> = Vec::with_capacity(n);
        for seat in 0..n {
            if seat == hero_seat {
                hole_cards.push(hero_hole.to_vec());
            } else {
                let mut h = Vec::with_capacity(2);
                for _ in 0..2 {
                    h.push(deck_order[next]);
                    next += 1;
                }
                hole_cards.push(h);
            }
        }

        // Antes (dead) then blinds (live) — the exact new_hand sequence.
        let mut stacks = config.starting_stacks.clone();
        let mut total_commit = vec![0u64; n];
        let mut all_in = vec![false; n];
        let folded = vec![false; n];
        let mut pot: u64 = 0;
        for i in 0..n {
            let paid = stacks[i].min(config.ante);
            stacks[i] -= paid;
            total_commit[i] = paid;
            pot += paid;
            if stacks[i] == 0 {
                all_in[i] = true;
            }
        }
        let mut street_commit = vec![0u64; n];
        let (sb_seat, bb_seat) = nlh_blind_seats(n, button, &folded);
        for (seat, amount) in [(sb_seat, config.sb), (bb_seat, config.bb)] {
            let paid = stacks[seat].min(amount);
            stacks[seat] -= paid;
            street_commit[seat] += paid;
            total_commit[seat] += paid;
            pot += paid;
            if stacks[seat] == 0 {
                all_in[seat] = true;
            }
        }
        let bet_to_call = config.bb;

        let eff_stack_cap_at_hand_start =
            compute_eff_stack_cap(&config.starting_stacks, &folded);

        let mut state = GameState {
            config,
            button,
            sb_seat: Some(sb_seat),
            bb_seat: Some(bb_seat),
            street: Street::Preflop,
            pot,
            stacks,
            folded,
            all_in,
            hole_cards,
            board_a: Vec::new(),
            board_b: Vec::new(),
            full_board_a: [Card(0); 5],
            full_board_b: [Card(0); 5],
            street_commit,
            total_commit,
            bet_to_call,
            last_raise_size: 0,
            last_aggression_was_full_raise: true,
            street_level_acted: vec![0u64; n],
            actor: None,
            last_aggressor: None,
            acted_this_street: vec![false; n],
            history: Vec::new(),
            study_mode: true,
            awaiting_next_street: None,
            study_terminal: None,
            study_hero_seat: Some(hero_seat),
            action_close_board_len: None,
            eff_stack_cap_at_hand_start,
        };
        state.last_raise_size = state.config.bb;
        state.actor = state.first_to_act_preflop(bb_seat);
        Ok(state)
    }

    /// NLH study: supply the 3-card flop after the preflop round closes.
    pub fn set_flop_nlh(&mut self, cards: [Card; 3]) -> Result<(), StudyError> {
        self.set_next_street_nlh(Street::Flop, &cards)
    }

    /// NLH study: supply the turn card. Single board — one card.
    pub fn set_turn_nlh(&mut self, card: Card) -> Result<(), StudyError> {
        self.set_next_street_nlh(Street::Turn, &[card])
    }

    /// NLH study: supply the river card.
    pub fn set_river_nlh(&mut self, card: Card) -> Result<(), StudyError> {
        self.set_next_street_nlh(Street::River, &[card])
    }

    fn set_next_street_nlh(
        &mut self,
        expected: Street,
        cards: &[Card],
    ) -> Result<(), StudyError> {
        if !self.study_mode || self.config.variant != Variant::NlhSingle {
            return Err(StudyError::WrongState);
        }
        if self.awaiting_next_street != Some(expected) {
            return Err(StudyError::WrongState);
        }
        let hero_seat = self.study_hero_seat.ok_or(StudyError::WrongState)?;
        let mut used = [false; 52];
        for c in self.hole_cards[hero_seat].iter().chain(self.board_a.iter()) {
            used[c.index() as usize] = true;
        }
        for c in cards {
            let i = c.index() as usize;
            if i >= 52 || used[i] {
                return Err(StudyError::DuplicateCard);
            }
            used[i] = true;
        }
        let base = self.board_a.len();
        for (j, c) in cards.iter().enumerate() {
            self.full_board_a[base + j] = *c;
            self.board_a.push(*c);
        }
        self.study_advance_street(expected);
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

        // Short-all-in reopen rule (per seat): a seat that already acted may
        // re-raise only if the bet has grown by at least one full raise
        // since its last action. See `short_shove_lockout`.
        if self.short_shove_lockout() {
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
        self.street_level_acted[actor] = self.bet_to_call;

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
        let won = match self.config.variant.num_boards() {
            2 => double_board_payout(
                &self.hole_cards,
                &self.folded,
                &self.total_commit,
                &self.full_board_a,
                &self.full_board_b,
                self.button,
            ),
            _ => single_board_payout(
                &self.hole_cards,
                &self.folded,
                &self.total_commit,
                &self.full_board_a,
                self.button,
            ),
        };
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
        let num_boards = self.config.variant.num_boards();
        let draw_per_sample = missing * num_boards;

        // Unseen deck: 52 − all hole cards − known prefix of the board(s).
        let mut used = [false; 52];
        for hole in &self.hole_cards {
            for c in hole.iter() {
                used[c.index() as usize] = true;
            }
        }
        for c in &self.full_board_a[..close_len] {
            used[c.index() as usize] = true;
        }
        if num_boards == 2 {
            for c in &self.full_board_b[..close_len] {
                used[c.index() as usize] = true;
            }
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
            let won = if num_boards == 2 {
                for i in 0..missing {
                    full_b[close_len + i] = deck[missing + i];
                }
                double_board_payout(
                    &self.hole_cards,
                    &self.folded,
                    &self.total_commit,
                    &full_a,
                    &full_b,
                    self.button,
                )
            } else {
                single_board_payout(
                    &self.hole_cards,
                    &self.folded,
                    &self.total_commit,
                    &full_a,
                    self.button,
                )
            };
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

    /// Variant betting cap on total street_commit, further capped at the
    /// maximum total any alive opponent can match. Above the latter,
    /// chips are dead money under side-pot rules. Pot-limit variants cap
    /// at the PL total; no-limit variants have no size cap of their own
    /// (the actor's stack is applied by `max_raise_chips`).
    pub fn max_bet_total(&self) -> u64 {
        let actor = match self.actor {
            Some(a) => a,
            None => return 0,
        };
        let reachable = self.max_other_reachable_total();
        if !self.config.variant.pot_limit() {
            return reachable;
        }
        let current_commit = self.street_commit[actor];
        let stack = self.stacks[actor];
        let to_call = self.bet_to_call.saturating_sub(current_commit).min(stack);
        let pl_total = self.bet_to_call + self.pot + to_call;
        pl_total.min(reachable)
    }

    /// True when the current actor is locked out of raising by the
    /// short-all-in reopen rule (per seat, TDA-style): the seat already
    /// acted this street AND the bet has not grown by at least one full
    /// raise (`last_raise_size`, which short shoves never lower) since
    /// that action. A short all-in advances `bet_to_call` without
    /// reopening seats that already responded to the prior bet, but a
    /// seat whose last action was at a lower level (e.g. a check at 0)
    /// is reopened as soon as the cumulative increase since its action
    /// reaches a full raise — including via multiple short all-ins.
    fn short_shove_lockout(&self) -> bool {
        let actor = match self.actor {
            Some(a) => a,
            None => return false,
        };
        let facing_bet = self.bet_to_call > self.street_commit[actor];
        facing_bet
            && self.acted_this_street[actor]
            && self.bet_to_call
                < self.street_level_acted[actor] + self.last_raise_size
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
        // Dust guard: continuous (Beta-sampled) sizings frequently land a
        // few chips shy of all-in, leaving an absurd sub-display "live"
        // stack that forces extra degenerate streets (cover-short bets of
        // a few chips) instead of a clean run-out. When the raise would
        // leave at most bb/100 behind and the full shove is within the
        // legal max, commit the full stack instead. Deterministic, so
        // action-log replays re-snap identically.
        let stack = self.stacks[actor];
        let dust_eps = (self.config.bb / 100).max(1);
        let chips = if chips < stack && stack - chips <= dust_eps && stack <= max {
            stack
        } else {
            chips
        };
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
        self.street_level_acted[actor] = self.bet_to_call;
    }

    /// Hand category index (0..=8 per `CAT_*` constants) of seat's current
    /// best hand on `board` (0=A, 1=B) under the variant's evaluation
    /// rule. Returns 0 if board has <3 cards (always for board B on
    /// single-board variants — its progressive view stays empty).
    pub fn hero_category(&self, seat: usize, board: u8) -> u8 {
        let b = if board == 0 { &self.board_a } else { &self.board_b };
        if b.len() < 3 {
            return 0;
        }
        let hole = &self.hole_cards[seat];
        let rank = match self.config.variant {
            Variant::Plo4DoubleBomb | Variant::Plo5DoubleBomb | Variant::Plo6DoubleBomb => {
                crate::hand_eval::evaluate_plo5_partial(hole, b)
            }
            Variant::NlhSingle => crate::hand_eval::evaluate_nlh(hole, b),
        };
        (rank >> 20) as u8
    }

    /// v7 obs batch-2 hero/board engine dims for the CURRENT actor
    /// (V7_OBS_CANDIDATES.md BRD-7 / BRD-12 / DUAL-2):
    /// `[boat_a, boat_b, improve_a, improve_b, combos_a, combos_b,
    ///   mask_a, mask_b]` — boat-or-better outs, strict-category-improve
    /// outs, best-category combo counts (all raw counts; the encoders
    /// normalize), and the best-holding 5-bit hole masks (bit i = i-th
    /// hole card sorted by card index DESCENDING). The unseen deck for
    /// the out counts is GLOBAL (hole + BOTH boards), matching the
    /// encoder's cross-board visibility convention. All-zero when there
    /// is no actor, for NLH (any-combo eval — these are PLO semantics),
    /// or before both boards have flops.
    pub fn hero_board_v3(&self) -> [u8; 8] {
        let mut out = [0u8; 8];
        if self.config.variant == Variant::NlhSingle {
            return out;
        }
        let hero = match self.actor {
            Some(s) => s,
            None => return out,
        };
        if self.board_a.len() < 3 || self.board_b.len() < 3 {
            return out;
        }
        let hole = &self.hole_cards[hero];
        let mut used = [false; 52];
        for c in hole
            .iter()
            .chain(self.board_a.iter())
            .chain(self.board_b.iter())
        {
            used[c.index() as usize] = true;
        }
        // Fused per-board: one pair_best_cks + one unseen scan each
        // (byte-identical to the three free fns called separately).
        let (ba, ia, ca, ma) =
            crate::hand_eval::hero_board_one(hole, &self.board_a, &used);
        let (bb, ib, cb, mb) =
            crate::hand_eval::hero_board_one(hole, &self.board_b, &used);
        out[0] = ba;
        out[1] = bb;
        out[2] = ia;
        out[3] = ib;
        out[4] = ca;
        out[5] = cb;
        out[6] = ma;
        out[7] = mb;
        out
    }

    /// v7 BRD-5/BRD-6/DUAL-5 hot block:
    /// [ds_a, ds_b, u_a, n_a, u_b, n_b, scoop] raw counts; encoder
    /// normalizes. All-zero for NLH / no-actor / short boards.
    pub fn board_draw_v3(&self) -> [u8; 7] {
        let zero = [0u8; 7];
        if self.config.variant == Variant::NlhSingle {
            return zero;
        }
        let hero = match self.actor {
            Some(s) => s,
            None => return zero,
        };
        if self.board_a.len() < 3 || self.board_b.len() < 3 {
            return zero;
        }
        let hole = &self.hole_cards[hero];
        let mut used = [false; 52];
        for c in hole
            .iter()
            .chain(self.board_a.iter())
            .chain(self.board_b.iter())
        {
            used[c.index() as usize] = true;
        }
        crate::hand_eval::board_draw_v3(hole, &self.board_a, &self.board_b, &used)
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
    /// Sampling: k=2 is exhaustive; k=3 and k=4 use `mc_samples` MC
    /// draws each. PRNG seeded deterministically from the immutable
    /// observation state so the feature is reproducible (parity tests
    /// survive at a fixed sample count).
    ///
    /// Returns all-zero before the flop or when the hand is terminal.
    ///
    /// Serial / UI / eval callers use the 1024-sample
    /// `opp_outcome_fractions` wrapper; batched training passes a
    /// smaller `mc_samples` (e.g. 256) — this feature is ~94% of the
    /// per-decision encode cost, so halving the MC budget roughly
    /// doubles obs-build throughput at a benign ~1-3% extra noise.
    pub fn opp_outcome_fractions_mc(&self, mc_samples: usize) -> Vec<f32> {
        self.outcome_features_mc(mc_samples)[..12].to_vec()
    }

    /// Superset of [`Self::opp_outcome_fractions_mc`]: the 12 joint
    /// outcome fractions PLUS an 8-dim PER-BOARD decomposition (obs v2,
    /// V5_DESIGN.md P1), all from the SAME single pass — the extra dims
    /// are counter increments inside the existing k=2 exhaustive loop
    /// (no extra evals, no extra RNG draws, so dims 0..12 stay
    /// bit-identical to the pre-v5 feature).
    ///
    /// Dims 12..20, hero-centric, k=2 EXHAUSTIVE universe only (exact,
    /// deterministic):
    /// - 12/13/14: board A — fraction of combos hero currently beats /
    ///   ties / is behind (sums to 1 when active).
    /// - 15/16/17: board B — same.
    /// - 18: win-exactly-one (hero ahead on one board, behind on the
    ///   other, either direction) — the modal double-board outcome the
    ///   12-dim block folds into its residual.
    /// - 19: tie on BOTH boards.
    /// Deterministic key for the opp-outcome MC — a hash of exactly the inputs
    /// the MC depends on (hero seat, street, hero hole, both boards). Returns
    /// `None` in precisely the cases `outcome_features_mc` returns all-zeros
    /// (no actor, or fewer than 3 board cards on either board). Used as the
    /// per-env cache key in the batched pack: equal key => the MC would produce
    /// the identical 20-dim output, so the cached value can be reused.
    ///
    /// MUST hash the same fields, in the same order, with the same hasher as
    /// the inline seed in `outcome_features_mc` below — keep them in lockstep.
    pub fn outcome_seed(&self) -> Option<u64> {
        let hero_seat = self.actor?;
        if self.board_a.len() < 3 || self.board_b.len() < 3 {
            return None;
        }
        let hero_hole = &self.hole_cards[hero_seat];
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
        Some(hasher.finish())
    }

    pub fn outcome_features_mc(&self, mc_samples: usize) -> Vec<f32> {
        // N_OUT 20 → 22 (2026-07-12, DUAL-4): dims 20/21 append the k=2
        // guaranteed-pot-share bounds g_min/g_max. Dims 0..20 stay
        // byte-identical to the pre-append body — the P1 pin test compares
        // them against the frozen reference; the share trackers add no
        // evals, no RNG draws, and no reordering.
        const N_OUT: usize = 22;
        // mc_samples == 0: caller does not need outcome features (e.g.
        // obs_mode=minimal). Skip all evals / deck work and return zeros.
        if mc_samples == 0 {
            return vec![0.0; N_OUT];
        }
        const SCOOP_OPP: usize = 0;
        const QUARTER_OPP: usize = 1;
        const SCOOP_HERO: usize = 2;
        const QUARTER_HERO: usize = 3;
        const PER_BOARD_OFF: usize = 12;
        const SHARE_OFF: usize = 20;

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

        // Per-holding board comparisons; +1 = opp ahead, 0 = tie, -1 = opp
        // behind (higher HandRank = stronger). Joint/per-board tallying
        // happens at the call sites.
        let cmp_ranks = |opp_a: u32, opp_b: u32| -> (i8, i8) {
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
            (cmp_a, cmp_b)
        };
        let tally_joint = |counters: &mut [u32; 4], cmp_a: i8, cmp_b: i8| {
            match (cmp_a, cmp_b) {
                (1, 1) => counters[SCOOP_OPP] += 1,
                (-1, -1) => counters[SCOOP_HERO] += 1,
                (1, 0) | (0, 1) => counters[QUARTER_OPP] += 1,
                (-1, 0) | (0, -1) => counters[QUARTER_HERO] += 1,
                _ => {}
            }
        };

        // P1: per-board pair-rank scratch tables, filled by the k=2 exhaustive
        // pass and reused by the k=3/4 MC arms. A PLO holding must use EXACTLY
        // 2 hole cards, so a k-card holding's rank factorizes as max over its
        // C(k,2) pairs of that pair's rank — and every MC-drawable pair is
        // enumerated by the k=2 pass (same unseen deck), so the MC arms become
        // table lookups instead of full evaluate_plo5_k_partial calls (~81-84%
        // of this block's hand-eval work at 384 samples). Degenerate pairs
        // (study-mode duplicate cards; every combo ck==0-filtered) store
        // ck_to_hand_rank(7462) == 0 == the u32 order bottom, exactly
        // mirroring the k-level per-combo skip — the identity holds for every
        // reachable state, duplicates included. Indexed by unseen-deck
        // POSITIONS (lo*STRIDE + hi, lo<hi); stride 52 covers every variant
        // and degenerate study state (PLO4 flop = 42 unseen; duplicated
        // hole/board cards push n_unseen higher still). Byte-identity vs the
        // frozen pre-P1 body is pinned by outcome_mc_p1_tests.
        const PAIR_STRIDE: usize = 52;
        debug_assert!(n_unseen <= PAIR_STRIDE);
        let mut tab_a = [0u32; PAIR_STRIDE * PAIR_STRIDE];
        let mut tab_b = [0u32; PAIR_STRIDE * PAIR_STRIDE];

        for (idx_k, &k) in [2usize, 3, 4].iter().enumerate() {
            let mut counters = [0u32; 4];
            let mut samples: u32 = 0;

            // k=2 exhaustive (C(<=41, 2) <= 820 is cheap); k=3,4 always
            // MC (`mc_samples` draws each). Previously k=3 was exhaustive
            // at turn+river (C(39,3) and C(37,3) both <= 10k), but that's
            // ~18k evals/env vs ~2*mc_samples for MC — dominated bundle cost.
            if k == 2 {
                // Per-board counters (obs v2 P1): [ahead, tie, behind] per
                // board from HERO's perspective + win-exactly-one + tie-both.
                let mut pb = [0u32; 8];
                // DUAL-4: hero's per-combo pot share s = 0.5·[wins A] +
                // 0.25·[ties A] + 0.5·[wins B] + 0.25·[ties B]; track the
                // min/max over the exhaustive k=2 universe. Values land
                // exactly on {0, .25, .5, .75, 1}. Free riders on the
                // existing loop — no extra evals, no RNG.
                let mut g_min = f32::MAX;
                let mut g_max = f32::MIN;
                let mut idx: Vec<usize> = (0..k).collect();
                loop {
                    opp_buf.clear();
                    for &i in idx.iter() {
                        opp_buf.push(unseen[i]);
                    }
                    let opp_a =
                        crate::hand_eval::evaluate_plo5_k_partial(&opp_buf, &self.board_a);
                    let opp_b =
                        crate::hand_eval::evaluate_plo5_k_partial(&opp_buf, &self.board_b);
                    // P1: record this pair's per-board ranks for the k=3/4
                    // MC arms (idx[0] < idx[1] by the combination enumerator).
                    tab_a[idx[0] * PAIR_STRIDE + idx[1]] = opp_a;
                    tab_b[idx[0] * PAIR_STRIDE + idx[1]] = opp_b;
                    let (cmp_a, cmp_b) = cmp_ranks(opp_a, opp_b);
                    tally_joint(&mut counters, cmp_a, cmp_b);
                    match cmp_a {
                        -1 => pb[0] += 1, // hero ahead on A
                        0 => pb[1] += 1,
                        _ => pb[2] += 1,
                    }
                    match cmp_b {
                        -1 => pb[3] += 1, // hero ahead on B
                        0 => pb[4] += 1,
                        _ => pb[5] += 1,
                    }
                    if (cmp_a == -1 && cmp_b == 1) || (cmp_a == 1 && cmp_b == -1) {
                        pb[6] += 1; // win exactly one
                    }
                    if cmp_a == 0 && cmp_b == 0 {
                        pb[7] += 1; // tie both
                    }
                    let share = 0.5 * ((cmp_a == -1) as u32 as f32)
                        + 0.25 * ((cmp_a == 0) as u32 as f32)
                        + 0.5 * ((cmp_b == -1) as u32 as f32)
                        + 0.25 * ((cmp_b == 0) as u32 as f32);
                    if share < g_min {
                        g_min = share;
                    }
                    if share > g_max {
                        g_max = share;
                    }
                    samples += 1;
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
                if samples > 0 {
                    let inv = 1.0f32 / samples as f32;
                    for j in 0..8 {
                        out[PER_BOARD_OFF + j] = pb[j] as f32 * inv;
                    }
                    out[SHARE_OFF] = g_min;
                    out[SHARE_OFF + 1] = g_max;
                }
            } else {
                debug_assert!(n_unseen <= 64);
                let mut pos = [0usize; 4];
                for _ in 0..mc_samples {
                    let mut mask: u64 = 0;
                    let mut written = 0;
                    while written < k {
                        let i = (rng.next_u32() as usize) % n_unseen;
                        let bit = 1u64 << i;
                        if mask & bit == 0 {
                            mask |= bit;
                            // P1: record the drawn POSITION. The draw loop
                            // itself (RNG call count, modulo, rejection mask)
                            // is untouched — the sample stream stays
                            // byte-identical to the pre-P1 body.
                            pos[written] = i;
                            written += 1;
                        }
                    }
                    // P1: holding rank = max over its C(k,2) pairs of the
                    // stored pair ranks (exactly-2-of-k factorization; see
                    // the table comment above). Drawn positions are unordered
                    // while the table is filled for lo < hi only.
                    let mut opp_a = 0u32;
                    let mut opp_b = 0u32;
                    for p0 in 0..k {
                        for p1 in (p0 + 1)..k {
                            let (lo, hi) = if pos[p0] < pos[p1] {
                                (pos[p0], pos[p1])
                            } else {
                                (pos[p1], pos[p0])
                            };
                            let ra = tab_a[lo * PAIR_STRIDE + hi];
                            if ra > opp_a {
                                opp_a = ra;
                            }
                            let rb = tab_b[lo * PAIR_STRIDE + hi];
                            if rb > opp_b {
                                opp_b = rb;
                            }
                        }
                    }
                    let (cmp_a, cmp_b) = cmp_ranks(opp_a, opp_b);
                    tally_joint(&mut counters, cmp_a, cmp_b);
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

    /// 1024-sample MC convenience wrapper (serial / UI / eval path).
    /// See [`Self::opp_outcome_fractions_mc`].
    pub fn opp_outcome_fractions(&self) -> Vec<f32> {
        self.opp_outcome_fractions_mc(1024)
    }

    /// NLH single-board opponent-outcome fractions: the share of
    /// unseen-deck 2-card opponent combos currently AHEAD of / TIED with
    /// / BEHIND the hero (current actor) at the visible board, evaluated
    /// exhaustively (≤ C(47, 2) = 1081 combos) under the any-combo NLH
    /// rule. Current-rank dominance, no runout sampling — the same
    /// convention as the PLO opp-outcome feature. Exhaustive enumeration
    /// makes it exactly reproducible with no seed.
    ///
    /// Returns `[opp_ahead, tied, opp_behind]`. All-zero preflop, on
    /// terminal states, and for non-NLH variants.
    pub fn nlh_opp_outcome_fractions(&self) -> Vec<f32> {
        const N_OUT: usize = 3;
        if self.config.variant != Variant::NlhSingle {
            return vec![0.0; N_OUT];
        }
        let hero_seat = match self.actor {
            Some(s) => s,
            None => return vec![0.0; N_OUT],
        };
        nlh_opp_outcome_for(&self.hole_cards[hero_seat], &self.board_a).to_vec()
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

    /// First seat-to-act for the preflop round: first non-folded,
    /// non-all-in seat walking clockwise from the seat after the big
    /// blind. Heads-up this lands on the button/SB (the only other
    /// in-hand seat), which is correct — the button acts first preflop.
    /// The walk wraps all the way to the BB itself, so a BB who is the
    /// only seat with chips behind still gets their option.
    fn first_to_act_preflop(&self, bb_seat: usize) -> Option<usize> {
        let n = self.config.num_seats;
        let start = (bb_seat + 1) % n;
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
            let two_boards = self.config.variant.num_boards() == 2;
            match next {
                Street::Flop => {
                    self.board_a = self.full_board_a[0..3].to_vec();
                    if two_boards {
                        self.board_b = self.full_board_b[0..3].to_vec();
                    }
                }
                Street::Turn => {
                    self.board_a.push(self.full_board_a[3]);
                    if two_boards {
                        self.board_b.push(self.full_board_b[3]);
                    }
                }
                Street::River => {
                    self.board_a.push(self.full_board_a[4]);
                    if two_boards {
                        self.board_b.push(self.full_board_b[4]);
                    }
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
            self.street_level_acted = vec![0u64; n];
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
            Street::Preflop => {
                // NLH study starts preflop; the UI supplies the flop next.
                self.awaiting_next_street = Some(Street::Flop);
                self.actor = None;
            }
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
            Street::Showdown => {
                // Unreachable in study mode.
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
            if self.config.variant.num_boards() == 2 && self.board_b.len() < 5 {
                self.board_b = self.full_board_b.to_vec();
            }
        }
        self.street = Street::Showdown;
        self.actor = None;
    }
}

/// Small/big-blind seats for a variant with blinds. Walks in-hand
/// (non-folded-at-deal) seats clockwise. Heads-up (exactly 2 seats in
/// hand): the button — or the first in-hand seat at/after it when the
/// nominal button seat sits out — is the SB. 3+ handed: SB is the first
/// in-hand seat strictly after the button, BB the next.
fn nlh_blind_seats(n: usize, button: usize, folded: &[bool]) -> (usize, usize) {
    let in_hand: Vec<usize> = (0..n).filter(|&i| !folded[i]).collect();
    assert!(in_hand.len() >= 2, "blinds need at least 2 seats in hand");
    if in_hand.len() == 2 {
        let sb = (0..n)
            .map(|i| (button + i) % n)
            .find(|&s| !folded[s])
            .expect("at least 2 in-hand seats");
        let bb = in_hand.into_iter().find(|&s| s != sb).unwrap();
        (sb, bb)
    } else {
        let mut walk = (1..=n).map(|i| (button + i) % n).filter(|&s| !folded[s]);
        let sb = walk.next().expect("at least 2 in-hand seats");
        let bb = walk.next().expect("at least 2 in-hand seats");
        (sb, bb)
    }
}

/// The (hole, board)-parameterized core of `nlh_opp_outcome_fractions`,
/// shared with the range-grid packer where candidate holes belong to no
/// engine state: the share of unseen-deck 2-card opponent combos AHEAD
/// of / TIED with / BEHIND `hole` at `board` under the any-combo NLH
/// rule (exhaustive, ≤ C(47, 2)). All-zero when the board has fewer
/// than 3 cards, matching the state method's preflop guard bit-exactly.
pub fn nlh_opp_outcome_for(hole: &[Card], board: &[Card]) -> [f32; 3] {
    const N_OUT: usize = 3;
    if board.len() < 3 {
        return [0.0; N_OUT];
    }
    let hero_rank = crate::hand_eval::evaluate_nlh(hole, board);

    let mut used = [false; 52];
    for c in hole.iter().chain(board.iter()) {
        used[c.index() as usize] = true;
    }
    let unseen: Vec<Card> = (0..52u8)
        .filter(|&i| !used[i as usize])
        .map(Card::from_index)
        .collect();
    let m = unseen.len();
    let mut counters = [0u64; N_OUT];
    let mut total = 0u64;
    for i in 0..m {
        for j in (i + 1)..m {
            let opp = [unseen[i], unseen[j]];
            let opp_rank = crate::hand_eval::evaluate_nlh(&opp, board);
            let k = if opp_rank > hero_rank {
                0
            } else if opp_rank == hero_rank {
                1
            } else {
                2
            };
            counters[k] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return [0.0; N_OUT];
    }
    let inv = 1.0f32 / total as f32;
    [
        counters[0] as f32 * inv,
        counters[1] as f32 * inv,
        counters[2] as f32 * inv,
    ]
}

/// Any-combo NLH hand-category (0..=8) for an arbitrary (hole, board);
/// 0 when the board has fewer than 3 cards — the (hole, board) core of
/// `hero_category` for the NLH variant (same `rank >> 20` extraction).
pub fn nlh_category_for(hole: &[Card], board: &[Card]) -> u8 {
    if board.len() < 3 {
        return 0;
    }
    (crate::hand_eval::evaluate_nlh(hole, board) >> 20) as u8
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
    fn checker_is_reopened_after_full_bet_plus_short_shove() {
        // Per-seat reopen rule: seat 0 CHECKS (acted at level 0), seat 1
        // makes a full bet, seat 2 short-shoves (sub-min-raise), seat 3
        // calls. Action returns to seat 0, whose level has grown by a
        // full raise since its check — seat 0 MAY raise. Seat 1 (who bet)
        // saw only the sub-min increase and stays locked.
        // 4 seats, button=3 → flop order 0,1,2,3. Ante 300 → pot 1200.
        // Stacks after ante: s0=19_700, s1=19_700, s2=700, s3=19_700.
        let mut cfg = GameConfig::new_uniform(4, 0, 300, 100);
        cfg.starting_stacks = vec![20_000u64, 20_000u64, 1_000u64, 20_000u64];
        let mut g = GameState::new_hand(cfg, 7, 3);

        assert_eq!(g.actor, Some(0));
        g.apply(Action::CheckCall); // seat 0 checks at level 0

        assert_eq!(g.actor, Some(1));
        g.apply(Action::BetPct50); // 0.5 * 1200 = 600 — full bet
        assert_eq!(g.bet_to_call, 600);
        assert_eq!(g.last_raise_size, 600);

        assert_eq!(g.actor, Some(2));
        g.apply(Action::AllIn); // 700 total: > 600 call, < 1200 min — short
        assert_eq!(g.bet_to_call, 700);
        assert_eq!(g.last_raise_size, 600, "floor preserved");
        assert!(!g.last_aggression_was_full_raise);

        assert_eq!(g.actor, Some(3));
        g.apply(Action::CheckCall); // flat-call 700

        // Seat 0 checked at level 0 and now faces 700 ≥ 0 + 600: reopened.
        assert_eq!(g.actor, Some(0));
        let mask0 = g.legal_action_mask();
        assert!(
            mask0.iter().skip(2).any(|&b| b),
            "checker must be reopened by full bet + short shove"
        );
        // Min raise = call 700 + full 600 increment (short shove doesn't
        // lower the floor).
        assert_eq!(g.min_raise_chips(), 1300);
        assert!(g.max_raise_chips() >= g.min_raise_chips());

        // Seat 0 just calls; seat 1 (bet at level 600, faces +100 < 600)
        // must stay locked to Fold/CheckCall.
        g.apply(Action::CheckCall);
        assert_eq!(g.actor, Some(1));
        let mask1 = g.legal_action_mask();
        assert!(mask1[Action::Fold as usize]);
        assert!(mask1[Action::CheckCall as usize]);
        assert!(
            mask1.iter().skip(2).all(|&b| !b),
            "original bettor must stay locked after sub-min increase"
        );
        assert_eq!(g.min_raise_chips(), 0);
        assert_eq!(g.max_raise_chips(), 0);
    }

    #[test]
    fn dust_raise_snaps_to_all_in_and_runs_out() {
        // 3 seats, button=2 → order 0,1,2. Ante 300 → pot 900.
        // Behind after ante: s0=19_700, s1=2_700, s2=19_700.
        let mut cfg = GameConfig::new_uniform(3, 0, 300, 100);
        cfg.starting_stacks = vec![20_000u64, 3_000u64, 20_000u64];
        let mut g = GameState::new_hand(cfg, 42, 2);

        g.apply_raise_chips(900).unwrap(); // seat 0 pots it
        // Seat 1: PL max delta 3600 > stack 2700 → stack-capped max 2700.
        // Raising 2699 would leave 1 chip (≤ bb/100 = 1) → snapped to 2700.
        assert_eq!(g.actor, Some(1));
        assert_eq!(g.max_raise_chips(), 2_700);
        g.apply_raise_chips(2_699).unwrap();
        assert_eq!(g.stacks[1], 0, "dust raise must snap to full stack");
        assert!(g.all_in[1], "snapped raise must set all_in");
        assert_eq!(g.bet_to_call, 2_700);

        g.apply(Action::Fold); // seat 2
        // Seat 0 calls the all-in: only one live-with-chips seat remains →
        // streets run out and the hand finalizes at showdown.
        assert_eq!(g.actor, Some(0));
        g.apply(Action::CheckCall);
        assert!(g.is_terminal(), "all-in call must run out to showdown");
        assert_eq!(g.board_a.len(), 5, "board A fully dealt");
        assert_eq!(g.board_b.len(), 5, "board B fully dealt");
        let payouts = g.payouts();
        assert_eq!(payouts.iter().sum::<i64>(), 0);
    }

    #[test]
    fn non_dust_short_stack_raise_is_not_snapped() {
        // Same shape, but the raise leaves 10 chips (> bb/100 = 1): a real
        // (if tiny) stack stays live — no snap.
        let mut cfg = GameConfig::new_uniform(3, 0, 300, 100);
        cfg.starting_stacks = vec![20_000u64, 3_000u64, 20_000u64];
        let mut g = GameState::new_hand(cfg, 42, 2);
        g.apply_raise_chips(900).unwrap();
        g.apply_raise_chips(2_690).unwrap();
        assert_eq!(g.stacks[1], 10);
        assert!(!g.all_in[1]);
    }

    #[test]
    fn cumulative_short_shoves_reopen_when_full_raise_reached() {
        // Multiple short all-ins that cumulatively amount to a full raise
        // reopen seats that acted before them (TDA rule).
        // 4 seats, button=3 → order 0,1,2,3. Ante 300 → pot 1200.
        // Behind after ante: s0=19_700, s1=1_000, s2=1_450, s3=19_700.
        let mut cfg = GameConfig::new_uniform(4, 0, 300, 100);
        cfg.starting_stacks = vec![20_000u64, 1_300u64, 1_750u64, 20_000u64];
        let mut g = GameState::new_hand(cfg, 7, 3);

        assert_eq!(g.actor, Some(0));
        g.apply(Action::BetPct50); // 600 — full bet, level 600
        assert_eq!(g.actor, Some(1));
        g.apply(Action::AllIn); // 1000: short (+400 < 600)
        assert_eq!(g.bet_to_call, 1000);
        assert_eq!(g.actor, Some(2));
        g.apply(Action::AllIn); // 1450: short again (+450 < 600)
        assert_eq!(g.bet_to_call, 1450);
        assert_eq!(g.last_raise_size, 600, "floor never lowered by shorts");
        assert_eq!(g.actor, Some(3));
        g.apply(Action::CheckCall); // deep seat flat-calls 1450

        // Seat 0 bet at level 600; cumulative increase 850 ≥ 600 — reopened.
        assert_eq!(g.actor, Some(0));
        let mask0 = g.legal_action_mask();
        assert!(
            mask0.iter().skip(2).any(|&b| b),
            "cumulative short shoves reaching a full raise must reopen"
        );
        assert_eq!(g.min_raise_chips(), 1450); // to 2050 total from 600 commit
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

    // ---- NLH variant (blinds, preflop street, NL cap, single board) ----

    /// User-facing default stake: $5/$10 with a $5 per-player ante at
    /// 1bb = 10000 chips ($10) → sb 5000, bb 10000, ante 5000.
    fn nlh_cfg(num_seats: usize, stack: u64) -> GameConfig {
        GameConfig::new_nlh_uniform(num_seats, stack, 5_000, 10_000, 5_000)
    }

    #[test]
    fn nlh_new_hand_posts_antes_and_blinds() {
        let g = GameState::new_hand(nlh_cfg(6, 1_000_000), 42, 0);
        // 6 × 5000 antes + 5000 SB + 10000 BB = 45000 — the "$45 pre-pot"
        // from the ClubGG 5/10(5) reference table.
        assert_eq!(g.pot, 45_000);
        assert_eq!(g.street, Street::Preflop);
        assert!(g.board_a.is_empty(), "no board revealed preflop");
        assert!(g.board_b.is_empty(), "single-board variant never fills B");
        // Button 0 → SB seat 1, BB seat 2, UTG (first actor) seat 3.
        assert_eq!(g.street_commit[1], 5_000);
        assert_eq!(g.street_commit[2], 10_000);
        assert_eq!(g.bet_to_call, 10_000);
        assert_eq!(g.actor, Some(3));
        for s in 0..6 {
            assert_eq!(g.hole_cards[s].len(), 2, "2 hole cards in NLH");
            // Antes are dead (in total, not street); blinds live.
            assert_eq!(g.total_commit[s], 5_000 + g.street_commit[s]);
        }
        assert_eq!(g.stacks[1], 1_000_000 - 5_000 - 5_000);
        assert_eq!(g.stacks[2], 1_000_000 - 5_000 - 10_000);
        // Blinds are forced posts, not actions: no history, nobody acted.
        assert!(g.history.is_empty());
        assert!(!g.acted_this_street.iter().any(|&b| b));
    }

    #[test]
    fn nlh_no_limit_cap_exceeds_pot_limit() {
        let g = GameState::new_hand(nlh_cfg(6, 1_000_000), 42, 0);
        // UTG min open = raise-to 2bb → delta 20000.
        assert_eq!(g.min_raise_chips(), 20_000);
        // NL cap: UTG's whole post-ante stack is a legal raise delta —
        // far above the PL total (btc 10000 + pot 45000 + call 10000).
        assert_eq!(g.max_raise_chips(), 995_000);
    }

    #[test]
    fn nlh_min_reraise_uses_last_raise_increment() {
        let mut g = GameState::new_hand(nlh_cfg(6, 1_000_000), 42, 0);
        assert!(g.apply_raise_chips(30_000).is_ok()); // UTG opens to 3bb
        assert_eq!(g.bet_to_call, 30_000);
        assert_eq!(g.last_raise_size, 20_000);
        // Next seat's min re-raise: to 5bb (30000 + 20000).
        assert_eq!(g.min_bet_total(), 50_000);
        assert_eq!(g.min_raise_chips(), 50_000);
    }

    #[test]
    fn nlh_limp_around_bb_option_then_flop() {
        let mut g = GameState::new_hand(nlh_cfg(6, 1_000_000), 42, 0);
        // UTG(3), 4, 5, button(0) call; SB(1) completes.
        for _ in 0..5 {
            g.apply(Action::CheckCall);
        }
        // The BB posted but has not ACTED — round must stay open.
        assert_eq!(g.actor, Some(2), "BB gets the option after limps");
        assert_eq!(g.street, Street::Preflop);
        // BB is not facing a bet, so Fold is masked (check is free).
        assert!(!g.legal_action_mask()[Action::Fold as usize]);
        g.apply(Action::CheckCall); // BB checks
        assert_eq!(g.street, Street::Flop);
        assert_eq!(g.board_a.len(), 3);
        assert!(g.board_b.is_empty());
        assert_eq!(g.actor, Some(1), "SB first to act postflop");
        assert_eq!(g.bet_to_call, 0);
        assert_eq!(g.pot, 90_000); // 45000 pre-pot + 6 × 10000 - blinds already in
    }

    #[test]
    fn nlh_bb_raise_reopens_limpers() {
        let mut g = GameState::new_hand(nlh_cfg(6, 1_000_000), 42, 0);
        for _ in 0..5 {
            g.apply(Action::CheckCall); // limps to BB
        }
        assert_eq!(g.actor, Some(2));
        assert!(g.apply_raise_chips(20_000).is_ok()); // BB raises to 3bb
        assert_eq!(g.bet_to_call, 30_000);
        assert_eq!(g.street, Street::Preflop, "raise keeps the round open");
        assert_eq!(g.actor, Some(3), "action returns to the first limper");
    }

    #[test]
    fn nlh_fold_to_bb_walk() {
        let mut g = GameState::new_hand(nlh_cfg(6, 1_000_000), 42, 0);
        for _ in 0..5 {
            g.apply(Action::Fold); // UTG..SB all fold
        }
        assert!(g.is_terminal(), "walk ends the hand preflop");
        let p = g.payouts();
        assert_eq!(p.iter().sum::<i64>(), 0);
        // BB (seat 2) collects the antes + SB, net of its own 15000 in.
        assert_eq!(p[2], 30_000);
        assert_eq!(p[1], -10_000, "SB loses ante + blind");
        assert_eq!(p[0], -5_000, "non-blind seats lose the ante only");
    }

    #[test]
    fn nlh_heads_up_button_is_sb_and_acts_first() {
        let mut g = GameState::new_hand(nlh_cfg(2, 1_000_000), 7, 0);
        assert_eq!(g.street_commit[0], 5_000, "button posts the SB heads-up");
        assert_eq!(g.street_commit[1], 10_000);
        assert_eq!(g.actor, Some(0), "button/SB acts first preflop");
        g.apply(Action::CheckCall); // SB completes
        assert_eq!(g.actor, Some(1), "BB has the option");
        g.apply(Action::CheckCall); // BB checks
        assert_eq!(g.street, Street::Flop);
        assert_eq!(g.actor, Some(1), "BB (non-button) acts first postflop");
    }

    #[test]
    fn nlh_preflop_allin_call_runs_out_single_board() {
        let mut g = GameState::new_hand(nlh_cfg(6, 1_000_000), 123, 0);
        let max = g.max_raise_chips();
        assert!(g.apply_raise_chips(max).is_ok()); // UTG jams
        g.apply(Action::Fold); // seat 4
        g.apply(Action::Fold); // seat 5
        g.apply(Action::Fold); // button 0
        g.apply(Action::Fold); // SB 1
        g.apply(Action::CheckCall); // BB calls all-in
        assert!(g.is_terminal());
        assert_eq!(g.board_a.len(), 5, "single board runs out");
        assert!(g.board_b.is_empty(), "board B never dealt in NLH");
        let p = g.payouts();
        assert_eq!(p.iter().sum::<i64>(), 0, "payouts are zero-sum");
        assert_eq!(p[0], -5_000);
        assert_eq!(p[1], -10_000);
    }

    #[test]
    fn nlh_short_bb_post_keeps_nominal_call() {
        // BB (seat 2 for button 0) can post only 3000 of the 10000 blind
        // after the 5000 ante. Callers still owe the FULL nominal bb;
        // side pots absorb the shortfall at settlement.
        let mut stacks = vec![1_000_000u64; 6];
        stacks[2] = 8_000;
        let cfg = GameConfig {
            num_seats: 6,
            starting_stacks: stacks,
            ante: 5_000,
            bb: 10_000,
            sb: 5_000,
            variant: Variant::NlhSingle,
        };
        let mut g = GameState::new_hand(cfg, 9, 0);
        assert_eq!(g.street_commit[2], 3_000, "short post");
        assert!(g.all_in[2]);
        assert_eq!(g.bet_to_call, 10_000, "nominal bb");
        assert_eq!(g.actor, Some(3));
        g.apply(Action::CheckCall);
        assert_eq!(g.street_commit[3], 10_000, "caller owes the full blind");
    }

    #[test]
    fn nlh_hand_is_replayable_from_seed() {
        let a = GameState::new_hand(nlh_cfg(6, 1_000_000), 777, 3);
        let b = GameState::new_hand(nlh_cfg(6, 1_000_000), 777, 3);
        let idx = |g: &GameState| -> Vec<Vec<u8>> {
            g.hole_cards
                .iter()
                .map(|h| h.iter().map(|c| c.index()).collect())
                .collect()
        };
        assert_eq!(idx(&a), idx(&b));
        let board = |g: &GameState| -> Vec<u8> {
            g.full_board_a.iter().map(|c| c.index()).collect()
        };
        assert_eq!(board(&a), board(&b));
        assert_eq!(a.actor, b.actor);
    }

    #[test]
    fn nlh_check_through_to_showdown_zero_sum() {
        // Full passive hand: limps + checks on every street; exercises
        // Preflop → Flop → Turn → River → Showdown with the single-board
        // payout path.
        let mut g = GameState::new_hand(nlh_cfg(3, 1_000_000), 55, 0);
        let mut guard = 0;
        while !g.is_terminal() {
            g.apply(Action::CheckCall);
            guard += 1;
            assert!(guard < 40, "hand must terminate");
        }
        assert_eq!(g.street, Street::Showdown);
        assert_eq!(g.board_a.len(), 5);
        let p = g.payouts();
        assert_eq!(p.iter().sum::<i64>(), 0);
    }

    #[test]
    fn nlh_study_mode_rejected() {
        let cfg = nlh_cfg(6, 1_000_000);
        let r = GameState::new_study(
            cfg,
            0,
            0,
            [Card::from_index(0); 5],
            [Card::from_index(10), Card::from_index(11), Card::from_index(12)],
            [Card::from_index(20), Card::from_index(21), Card::from_index(22)],
        );
        assert_eq!(r.err(), Some(StudyError::WrongState));
    }
}

#[cfg(test)]
mod outcome_mc_p1_tests {
    //! P1 bit-exactness harness. `outcome_features_mc_reference` is a FROZEN
    //! copy of the pre-pair-table function body (as of 2026-07-09). Do NOT
    //! "sync" it with the live function — its entire purpose is to pin that
    //! the P1 pair-table rewrite produces byte-identical output on every
    //! reachable state class: all PLO variants, all streets, rotated heroes,
    //! and the degenerate study-mode duplicate-card states that exercise the
    //! all-combos-filtered pair fallback (HandRank 0) and n_unseen > 41.
    use super::*;

    #[allow(clippy::needless_range_loop)]
    fn outcome_features_mc_reference(g: &GameState, mc_samples: usize) -> Vec<f32> {
        const N_OUT: usize = 20;
        const SCOOP_OPP: usize = 0;
        const QUARTER_OPP: usize = 1;
        const SCOOP_HERO: usize = 2;
        const QUARTER_HERO: usize = 3;
        const PER_BOARD_OFF: usize = 12;

        let hero_seat = match g.actor {
            Some(s) => s,
            None => return vec![0.0; N_OUT],
        };
        if g.board_a.len() < 3 || g.board_b.len() < 3 {
            return vec![0.0; N_OUT];
        }

        let hero_hole = &g.hole_cards[hero_seat];
        let hero_a = crate::hand_eval::evaluate_plo5_partial(hero_hole, &g.board_a);
        let hero_b = crate::hand_eval::evaluate_plo5_partial(hero_hole, &g.board_b);

        let mut used = [false; 52];
        for c in hero_hole.iter() {
            used[c.index() as usize] = true;
        }
        for c in g.board_a.iter().chain(g.board_b.iter()) {
            used[c.index() as usize] = true;
        }
        let unseen: Vec<Card> = (0..52u8)
            .filter(|&i| !used[i as usize])
            .map(Card::from_index)
            .collect();
        let n_unseen = unseen.len();

        let seed: u64 = {
            use std::hash::Hasher;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            hasher.write_u8(hero_seat as u8);
            hasher.write_u8(g.street.index() as u8);
            for c in hero_hole.iter() {
                hasher.write_u8(c.index());
            }
            for c in g.board_a.iter().chain(g.board_b.iter()) {
                hasher.write_u8(c.index());
            }
            hasher.finish()
        };

        use rand_chacha::ChaCha8Rng;
        use rand_chacha::rand_core::{RngCore, SeedableRng};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut out = vec![0.0f32; N_OUT];
        let mut opp_buf: Vec<Card> = Vec::with_capacity(4);

        let outcomes = |opp: &[Card]| -> (i8, i8) {
            let opp_a = crate::hand_eval::evaluate_plo5_k_partial(opp, &g.board_a);
            let opp_b = crate::hand_eval::evaluate_plo5_k_partial(opp, &g.board_b);
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
            (cmp_a, cmp_b)
        };
        let tally_joint = |counters: &mut [u32; 4], cmp_a: i8, cmp_b: i8| {
            match (cmp_a, cmp_b) {
                (1, 1) => counters[SCOOP_OPP] += 1,
                (-1, -1) => counters[SCOOP_HERO] += 1,
                (1, 0) | (0, 1) => counters[QUARTER_OPP] += 1,
                (-1, 0) | (0, -1) => counters[QUARTER_HERO] += 1,
                _ => {}
            }
        };

        for (idx_k, &k) in [2usize, 3, 4].iter().enumerate() {
            let mut counters = [0u32; 4];
            let mut samples: u32 = 0;

            if k == 2 {
                let mut pb = [0u32; 8];
                let mut idx: Vec<usize> = (0..k).collect();
                loop {
                    opp_buf.clear();
                    for &i in idx.iter() {
                        opp_buf.push(unseen[i]);
                    }
                    let (cmp_a, cmp_b) = outcomes(&opp_buf);
                    tally_joint(&mut counters, cmp_a, cmp_b);
                    match cmp_a {
                        -1 => pb[0] += 1,
                        0 => pb[1] += 1,
                        _ => pb[2] += 1,
                    }
                    match cmp_b {
                        -1 => pb[3] += 1,
                        0 => pb[4] += 1,
                        _ => pb[5] += 1,
                    }
                    if (cmp_a == -1 && cmp_b == 1) || (cmp_a == 1 && cmp_b == -1) {
                        pb[6] += 1;
                    }
                    if cmp_a == 0 && cmp_b == 0 {
                        pb[7] += 1;
                    }
                    samples += 1;
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
                if samples > 0 {
                    let inv = 1.0f32 / samples as f32;
                    for j in 0..8 {
                        out[PER_BOARD_OFF + j] = pb[j] as f32 * inv;
                    }
                }
            } else {
                debug_assert!(n_unseen <= 64);
                for _ in 0..mc_samples {
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
                    let (cmp_a, cmp_b) = outcomes(&opp_buf);
                    tally_joint(&mut counters, cmp_a, cmp_b);
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

    fn assert_bit_identical(g: &GameState, mc_samples: usize, tag: &str) {
        let new = g.outcome_features_mc(mc_samples);
        let reference = outcome_features_mc_reference(g, mc_samples);
        // The live fn appends the DUAL-4 share bounds (dims 20/21,
        // 2026-07-12); the frozen reference stays 20-wide by design. The
        // pin's purpose is unchanged: dims 0..20 byte-identical.
        assert_eq!(new.len(), 22, "{tag}: live output must be 22-wide");
        assert_eq!(reference.len(), 20, "{tag}: frozen reference is 20-wide");
        for (i, (x, y)) in new.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{tag}: dim {i} differs (new {x} vs ref {y})"
            );
        }
        // Appended share bounds: either both zero (inactive / no combos)
        // or a valid quantized min<=max pair.
        let (g_min, g_max) = (new[20], new[21]);
        assert!(
            (g_min == 0.0 && g_max == 0.0) || g_min <= g_max,
            "{tag}: share bounds invalid ({g_min}, {g_max})"
        );
        for v in [g_min, g_max] {
            assert!(
                [0.0, 0.25, 0.5, 0.75, 1.0].contains(&v),
                "{tag}: share bound {v} not on the quarter grid"
            );
        }
    }

    fn reveal_turn(g: &mut GameState) {
        g.board_a.push(g.full_board_a[3]);
        g.board_b.push(g.full_board_b[3]);
        g.street = Street::Turn;
    }

    fn reveal_river(g: &mut GameState) {
        g.board_a.push(g.full_board_a[4]);
        g.board_b.push(g.full_board_b[4]);
        g.street = Street::River;
    }

    fn variant_cfg(variant: Variant, num_seats: usize) -> GameConfig {
        GameConfig {
            num_seats,
            starting_stacks: vec![200_000; num_seats],
            ante: 30_000,
            bb: 10_000,
            sb: 0,
            variant,
        }
    }

    #[test]
    fn pair_table_matches_reference() {
        // Production-shaped states: every PLO variant x street x seeds x seat
        // counts, with the hero rotated across every seat (the actor is the
        // only seat the function reads).
        let variants = [
            Variant::Plo5DoubleBomb,
            Variant::Plo4DoubleBomb,
            Variant::Plo6DoubleBomb,
        ];
        for (vi, &variant) in variants.iter().enumerate() {
            for &num_seats in &[2usize, 6] {
                for seed in 0..3u64 {
                    for street in 0..3usize {
                        let mut g = GameState::new_hand(
                            variant_cfg(variant, num_seats),
                            seed * 7919 + street as u64,
                            0,
                        );
                        if street >= 1 {
                            reveal_turn(&mut g);
                        }
                        if street >= 2 {
                            reveal_river(&mut g);
                        }
                        for hero in 0..num_seats {
                            g.actor = Some(hero);
                            assert_bit_identical(
                                &g,
                                64,
                                &format!(
                                    "variant#{vi} seats={num_seats} seed={seed} \
                                     street={street} hero={hero}"
                                ),
                            );
                        }
                    }
                }
            }
        }
        // Deployed sample count on one full-size case, plus mc_samples edges
        // (0 exercises the samples==0 normalize guard).
        let g = GameState::new_hand(variant_cfg(Variant::Plo5DoubleBomb, 6), 12345, 2);
        assert_bit_identical(&g, 384, "plo5 6max flop mc=384");
        assert_bit_identical(&g, 0, "mc=0");
        assert_bit_identical(&g, 1, "mc=1");
    }

    #[test]
    fn pair_table_matches_reference_degenerate_study_states() {
        // Study-mode duplicate-card states: hole cards colliding with board
        // cards and boards sharing cards shrink the `used` union (n_unseen
        // past the 41 production ceiling, up to 46+) and exercise the
        // degenerate-pair fallback (all combos ck==0-filtered -> HandRank 0).
        // Production deals never reach these; the serial observation_dict /
        // study path can.
        let mut g = GameState::new_hand(variant_cfg(Variant::Plo5DoubleBomb, 6), 99, 0);
        let hero = g.actor.unwrap();

        // Hero hole card duplicated onto board_a: at the flop there is exactly
        // one board triple, so every combo using that hole card degenerates.
        let mut g1 = g.clone();
        g1.board_a[0] = g1.hole_cards[hero][0];
        assert_bit_identical(&g1, 64, "hero hole card duplicated on board_a");

        // Boards sharing a card.
        let mut g2 = g.clone();
        g2.board_b[1] = g2.board_a[1];
        assert_bit_identical(&g2, 64, "board_a/board_b share a card");

        // Pathological mass duplication: several hero cards on both boards +
        // a cross-board duplicate (n_unseen well past 41).
        let mut g3 = g.clone();
        g3.board_a[0] = g3.hole_cards[hero][0];
        g3.board_a[1] = g3.hole_cards[hero][1];
        g3.board_b[0] = g3.hole_cards[hero][2];
        g3.board_b[1] = g3.hole_cards[hero][3];
        g3.board_b[2] = g3.board_a[2];
        assert_bit_identical(&g3, 64, "mass-duplicate study state");

        // INTRA-board duplicate: the only state class where an OPP pair's
        // evals can ALL degenerate (opp cards come from the unseen deck, so
        // they never collide with board cards — a 5-card combo can only
        // contain a duplicate if the board TRIPLE itself does). At the flop
        // board_a has exactly one triple, and it contains the dup, so every
        // opp pair's tab_a entry takes the all-combos-filtered fallback
        // (u16::MAX -> 7462 -> HandRank 0) — pinning the max-fold identity's
        // hardest case for real. Unreachable via study input validation
        // (DuplicateCard guard); pinned at the function level regardless.
        let mut g4 = g.clone();
        g4.board_a[1] = g4.board_a[0];
        assert_bit_identical(&g4, 64, "intra-board duplicate (all-degenerate pairs)");

        // Turn-street collision: 4-card board, so the colliding pair keeps
        // some valid triples (partial-degeneracy path).
        reveal_turn(&mut g);
        g.board_a[0] = g.hole_cards[hero][0];
        assert_bit_identical(&g, 64, "turn-street hole/board collision");
    }
}

#[cfg(test)]
mod plo6_tests {
    use super::*;

    fn plo6_cfg(num_seats: usize, stack: u64) -> GameConfig {
        GameConfig {
            num_seats,
            starting_stacks: vec![stack; num_seats],
            ante: 30_000,
            bb: 10_000,
            sb: 0,
            variant: Variant::Plo6DoubleBomb,
        }
    }

    #[test]
    fn plo6_deals_six_unique_cards_per_seat_full_ring() {
        let g = GameState::new_hand(plo6_cfg(6, 200_000), 42, 0);
        let mut seen = [false; 52];
        for h in &g.hole_cards {
            assert_eq!(h.len(), 6, "PLO6 deals 6 hole cards");
            for card in h {
                assert!(!seen[card.index() as usize], "duplicate hole card");
                seen[card.index() as usize] = true;
            }
        }
        for card in g.full_board_a.iter().chain(g.full_board_b.iter()) {
            assert!(!seen[card.index() as usize], "board reuses a hole card");
            seen[card.index() as usize] = true;
        }
        assert_eq!(seen.iter().filter(|&&x| x).count(), 6 * 6 + 10);
    }

    #[test]
    fn plo6_replayable_from_seed() {
        let a = GameState::new_hand(plo6_cfg(4, 500_000), 777, 2);
        let b = GameState::new_hand(plo6_cfg(4, 500_000), 777, 2);
        let idx = |g: &GameState| -> Vec<Vec<u8>> {
            g.hole_cards
                .iter()
                .map(|h| h.iter().map(|c| c.index()).collect())
                .collect()
        };
        assert_eq!(idx(&a), idx(&b));
    }

    #[test]
    fn plo6_starts_on_flop_and_checkdown_conserves_chips() {
        let mut g = GameState::new_hand(plo6_cfg(6, 200_000), 7, 3);
        assert_eq!(g.street, Street::Flop, "bomb pot starts on the flop");
        let mut guard = 0;
        while !g.is_terminal() && guard < 200 {
            g.apply(Action::CheckCall);
            guard += 1;
        }
        assert!(g.is_terminal(), "checkdown must reach showdown");
        let payouts = g.payouts();
        let net: i64 = payouts.iter().sum();
        assert_eq!(net, 0, "zero-sum payouts");
        assert!(payouts.iter().any(|&p| p > 0), "someone wins the antes");
    }

    #[test]
    fn plo6_pot_limit_cap_matches_plo5_math() {
        // Same stacks/antes → identical pot-limit max on the flop
        // regardless of hole-card count.
        let g5 = GameState::new_hand(GameConfig::new_uniform(6, 200_000, 30_000, 10_000), 11, 0);
        let g6 = GameState::new_hand(plo6_cfg(6, 200_000), 11, 0);
        assert_eq!(g5.max_raise_chips(), g6.max_raise_chips());
        assert_eq!(g5.min_bet_total(), g6.min_bet_total());
    }

    #[test]
    fn plo6_hero_category_uses_six_cards() {
        // Deterministic construction via seeds is opaque; instead assert
        // the category call runs and returns a valid index for every seat.
        let g = GameState::new_hand(plo6_cfg(6, 200_000), 99, 1);
        for seat in 0..6 {
            let cat_a = g.hero_category(seat, 0);
            let cat_b = g.hero_category(seat, 1);
            assert!(cat_a <= 8 && cat_b <= 8);
        }
    }

    #[test]
    fn plo6_study_mode_rejected() {
        let cfg = plo6_cfg(6, 200_000);
        let hero = [
            Card::from_index(0),
            Card::from_index(4),
            Card::from_index(8),
            Card::from_index(12),
            Card::from_index(16),
        ];
        let fa = [Card::from_index(20), Card::from_index(24), Card::from_index(28)];
        let fb = [Card::from_index(32), Card::from_index(36), Card::from_index(40)];
        let r = GameState::new_study(cfg, 0, 0, hero, fa, fb);
        assert!(r.is_err(), "study mode is PLO5-only until the UI phase");
    }
}

#[cfg(test)]
mod plo4_tests {
    use super::*;

    fn plo4_cfg(num_seats: usize, stack: u64) -> GameConfig {
        GameConfig {
            num_seats,
            starting_stacks: vec![stack; num_seats],
            ante: 30_000,
            bb: 10_000,
            sb: 0,
            variant: Variant::Plo4DoubleBomb,
        }
    }

    #[test]
    fn plo4_deals_four_unique_cards_per_seat_full_ring() {
        let g = GameState::new_hand(plo4_cfg(6, 200_000), 42, 0);
        let mut seen = [false; 52];
        for h in &g.hole_cards {
            assert_eq!(h.len(), 4, "PLO4 deals 4 hole cards");
            for card in h {
                assert!(!seen[card.index() as usize], "duplicate hole card");
                seen[card.index() as usize] = true;
            }
        }
        for card in g.full_board_a.iter().chain(g.full_board_b.iter()) {
            assert!(!seen[card.index() as usize], "board reuses a hole card");
            seen[card.index() as usize] = true;
        }
        assert_eq!(seen.iter().filter(|&&x| x).count(), 6 * 4 + 10);
    }

    #[test]
    fn plo4_replayable_from_seed() {
        let a = GameState::new_hand(plo4_cfg(4, 500_000), 777, 2);
        let b = GameState::new_hand(plo4_cfg(4, 500_000), 777, 2);
        let idx = |g: &GameState| -> Vec<Vec<u8>> {
            g.hole_cards
                .iter()
                .map(|h| h.iter().map(|c| c.index()).collect())
                .collect()
        };
        assert_eq!(idx(&a), idx(&b));
    }

    #[test]
    fn plo4_starts_on_flop_and_checkdown_conserves_chips() {
        let mut g = GameState::new_hand(plo4_cfg(6, 200_000), 7, 3);
        assert_eq!(g.street, Street::Flop, "bomb pot starts on the flop");
        let mut guard = 0;
        while !g.is_terminal() && guard < 200 {
            g.apply(Action::CheckCall);
            guard += 1;
        }
        assert!(g.is_terminal(), "checkdown must reach showdown");
        let payouts = g.payouts();
        let net: i64 = payouts.iter().sum();
        assert_eq!(net, 0, "zero-sum payouts");
        assert!(payouts.iter().any(|&p| p > 0), "someone wins the antes");
    }

    #[test]
    fn plo4_pot_limit_cap_matches_plo5_math() {
        // Same stacks/antes → identical pot-limit max on the flop
        // regardless of hole-card count.
        let g5 = GameState::new_hand(GameConfig::new_uniform(6, 200_000, 30_000, 10_000), 11, 0);
        let g4 = GameState::new_hand(plo4_cfg(6, 200_000), 11, 0);
        assert_eq!(g5.max_raise_chips(), g4.max_raise_chips());
        assert_eq!(g5.min_bet_total(), g4.min_bet_total());
    }

    #[test]
    fn plo4_hero_category_valid_all_seats_both_boards() {
        let g = GameState::new_hand(plo4_cfg(6, 200_000), 99, 1);
        for seat in 0..6 {
            let cat_a = g.hero_category(seat, 0);
            let cat_b = g.hero_category(seat, 1);
            assert!(cat_a <= 8 && cat_b <= 8);
        }
    }

    #[test]
    fn plo4_study_mode_rejected() {
        let cfg = plo4_cfg(6, 200_000);
        let hero = [
            Card::from_index(0),
            Card::from_index(4),
            Card::from_index(8),
            Card::from_index(12),
            Card::from_index(16),
        ];
        let fa = [Card::from_index(20), Card::from_index(24), Card::from_index(28)];
        let fb = [Card::from_index(32), Card::from_index(36), Card::from_index(40)];
        let r = GameState::new_study(cfg, 0, 0, hero, fa, fb);
        assert!(r.is_err(), "study mode is PLO5-only until the UI phase");
    }
}

#[cfg(test)]
mod nlh_study_tests {
    use super::*;

    fn cfg(num_seats: usize) -> GameConfig {
        GameConfig::new_nlh_uniform(num_seats, 1_000_000, 5_000, 10_000, 5_000)
    }

    fn hero2(a: u8, b: u8) -> [Card; 2] {
        [Card::from_index(a), Card::from_index(b)]
    }

    #[test]
    fn nlh_study_starts_preflop_with_blinds() {
        let g = GameState::new_study_nlh(cfg(6), 0, 3, hero2(51, 47)).unwrap();
        assert_eq!(g.street, Street::Preflop);
        assert!(g.board_a.is_empty() && g.board_b.is_empty());
        // 6 antes + SB + BB = 45,000 (the 5/10(5) $45 pre-pot).
        assert_eq!(g.pot, 45_000);
        assert_eq!((g.sb_seat, g.bb_seat), (Some(1), Some(2)));
        assert_eq!(g.street_commit[1], 5_000);
        assert_eq!(g.street_commit[2], 10_000);
        assert_eq!(g.bet_to_call, 10_000);
        assert_eq!(g.actor, Some(3), "UTG first preflop");
        assert!(g.study_mode);
        assert_eq!(g.hole_cards[3].len(), 2);
        for (seat, h) in g.hole_cards.iter().enumerate() {
            assert_eq!(h.len(), 2, "seat {seat} must hold 2 cards");
        }
    }

    #[test]
    fn nlh_study_full_hand_walkthrough() {
        // 3-max: button 0 = first preflop actor is button (SB=1, BB=2 →
        // UTG=0). Limp around → flop; bet/call → turn; check around →
        // river; check around → Showdown terminal with zero payouts.
        let mut g = GameState::new_study_nlh(cfg(3), 0, 0, hero2(51, 47)).unwrap();
        assert_eq!(g.actor, Some(0));
        for _ in 0..3 {
            g.apply(Action::CheckCall);
        }
        assert_eq!(g.awaiting_next_street, Some(Street::Flop));
        assert_eq!(g.actor, None);
        g.set_flop_nlh([
            Card::from_index(0),
            Card::from_index(5),
            Card::from_index(10),
        ])
        .unwrap();
        assert_eq!(g.street, Street::Flop);
        assert_eq!(g.board_a.len(), 3);
        assert!(g.board_b.is_empty(), "single board never fills B");
        assert_eq!(g.actor, Some(1), "SB first postflop");
        // SB bets 20k, others call.
        assert!(g.apply_raise_chips(20_000).is_ok());
        g.apply(Action::CheckCall);
        g.apply(Action::CheckCall);
        assert_eq!(g.awaiting_next_street, Some(Street::Turn));
        g.set_turn_nlh(Card::from_index(15)).unwrap();
        assert_eq!(g.board_a.len(), 4);
        for _ in 0..3 {
            g.apply(Action::CheckCall);
        }
        assert_eq!(g.awaiting_next_street, Some(Street::River));
        g.set_river_nlh(Card::from_index(20)).unwrap();
        assert_eq!(g.board_a.len(), 5);
        for _ in 0..3 {
            g.apply(Action::CheckCall);
        }
        assert_eq!(g.study_terminal, Some(StudyTerminal::Showdown));
        assert!(g.payouts().iter().all(|&p| p == 0), "opp cards unknown");
    }

    #[test]
    fn nlh_study_preflop_foldout_pays_hero() {
        let mut g = GameState::new_study_nlh(cfg(2), 0, 0, hero2(51, 47)).unwrap();
        // HU: button/SB acts first preflop; SB raises, BB folds.
        assert_eq!(g.actor, Some(0));
        assert!(g.apply_raise_chips(30_000).is_ok());
        g.apply(Action::Fold);
        assert_eq!(g.study_terminal, Some(StudyTerminal::FoldOut));
        let p = g.payouts();
        assert!(p[0] > 0 && p[1] < 0, "uncontested pot goes to hero");
    }

    #[test]
    fn nlh_study_card_validation() {
        // Duplicate hero cards rejected.
        assert!(GameState::new_study_nlh(cfg(3), 0, 0, hero2(5, 5)).is_err());
        // Flop colliding with hero hole rejected; PLO dual-board setter
        // rejected on an NLH state.
        let mut g = GameState::new_study_nlh(cfg(3), 0, 0, hero2(51, 47)).unwrap();
        for _ in 0..3 {
            g.apply(Action::CheckCall);
        }
        assert_eq!(g.awaiting_next_street, Some(Street::Flop));
        let r = g.set_flop_nlh([
            Card::from_index(51),
            Card::from_index(1),
            Card::from_index(2),
        ]);
        assert_eq!(r, Err(StudyError::DuplicateCard));
        let r2 = g.set_turn(Card::from_index(1), Card::from_index(2));
        assert_eq!(r2, Err(StudyError::WrongState));
        // Wrong-order setter (turn before flop) rejected.
        let r3 = g.set_turn_nlh(Card::from_index(1));
        assert_eq!(r3, Err(StudyError::WrongState));
        // Valid flop still works after the failed attempts.
        assert!(g
            .set_flop_nlh([
                Card::from_index(0),
                Card::from_index(1),
                Card::from_index(2),
            ])
            .is_ok());
    }

    #[test]
    fn nlh_study_rejected_for_plo() {
        let plo = GameConfig::new_uniform(6, 200_000, 30_000, 10_000);
        let r = GameState::new_study_nlh(plo, 0, 0, hero2(0, 4));
        assert_eq!(r.err(), Some(StudyError::WrongState));
    }

    #[test]
    fn nlh_hole_feature_free_fns_match_state_methods() {
        // The range-grid packer computes per-combo features via the free
        // fns; pin them bit-exact against the state methods for the
        // ACTUAL actor hole, preflop (both zero) and on a flop.
        let mut g = GameState::new_study_nlh(cfg(3), 0, 0, hero2(51, 47)).unwrap();
        let a = g.actor.unwrap();
        assert_eq!(
            nlh_opp_outcome_for(&g.hole_cards[a], &g.board_a),
            [0.0f32; 3],
            "preflop opp-outcome must be zeros"
        );
        assert_eq!(nlh_category_for(&g.hole_cards[a], &g.board_a), 0);
        for _ in 0..3 {
            g.apply(Action::CheckCall);
        }
        g.set_flop_nlh([
            Card::from_index(0),
            Card::from_index(5),
            Card::from_index(10),
        ])
        .unwrap();
        let a = g.actor.unwrap();
        let free = nlh_opp_outcome_for(&g.hole_cards[a], &g.board_a);
        let method = g.nlh_opp_outcome_fractions();
        assert_eq!(free.to_vec(), method);
        assert!(free.iter().sum::<f32>() > 0.99, "flop fractions populated");
        assert_eq!(
            nlh_category_for(&g.hole_cards[a], &g.board_a),
            g.hero_category(a, 0)
        );
        // An arbitrary non-actor combo also works (no state required).
        let combo = [Card::from_index(30), Card::from_index(31)];
        let fr = nlh_opp_outcome_for(&combo, &g.board_a);
        assert!((fr.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
