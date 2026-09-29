//! Game state machine: deal, apply action, advance street, settle payouts.

use crate::actions::{Action, NUM_ACTIONS};
use crate::cards::{Card, CardMask, Deck, NO_CARD};
use crate::double_board::{
    double_board_payout, double_board_payout_runout, single_board_payout,
    single_board_payout_layers, PotLayers, RunoutRanker,
};
// The observation-feature half of the old impl lives in obs_features.rs
// (ENG-010); these stay reachable at their old paths.
#[cfg(test)]
use crate::obs_features::outcome_mc_seed;
pub use crate::obs_features::{nlh_category_for, nlh_opp_outcome_for, BoardPairTable};
use crate::state::{
    ActionRecord, GameConfig, GameState, Street, StudyError, StudyTerminal, Variant,
};

impl GameState {
    /// Deal a fresh hand: shuffle deck with `ChaCha8Rng::seed_from_u64(seed)`,
    /// deal every seat its `hole_count` hole cards (PLO67: plus its reserved
    /// extras), pre-deal the full board(s) and burns, post the antes (and
    /// blinds), reveal the flops (bomb pots) and seat the first actor.
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
        Self::new_hand_from_deck(config, Deck::new_shuffled(seed), button, in_hand_mask)
    }

    /// [`Self::new_hand_with_mask`] from an EXPLICIT deck order instead of a
    /// seed (the home games' verifiable shuffle: the deck is sealed, re-permuted
    /// by the players' devices and dealt exactly as it lies). The deal order is
    /// the same public contract: `hole_slots` cards per seat index — EVERY
    /// seat index, dealt in or not, so a card's slot never depends on who sat
    /// out — seat 0 first, then full board A, then full board B, then the
    /// burns (PLO67 only: a seat's slots beyond `hole_count` are its reserved
    /// extra cards, handed out in slot order on red burns).
    /// `new_hand_with_mask(seed)` is exactly this with `Deck::new_shuffled(seed)`.
    pub fn new_hand_from_deck(
        config: GameConfig,
        deck: Deck,
        button: usize,
        in_hand_mask: Option<Vec<bool>>,
    ) -> Self {
        let mut state = GameState::blank(config);
        state.deal(deck, button, in_hand_mask.as_deref());
        state
    }

    /// Re-deal this table for a new hand: exactly
    /// `new_hand(config.clone(), seed, button)` -- every seat in, the same
    /// deal order, posts and first actor -- written into the previous hand's
    /// buffers instead of ~20 fresh allocations (PERF-034: the batched engine
    /// re-deals about one table in eight every step).
    pub fn redeal(&mut self, config: &GameConfig, seed: u64, button: usize) {
        self.config.clone_from(config);
        self.deal(Deck::new_shuffled(seed), button, None);
    }

    /// The deal behind [`Self::new_hand_from_deck`] and [`Self::redeal`]:
    /// the cards in the public deal order, every other field at its deal-time
    /// value ([`Self::reset_table`]), antes, blinds, the flop's burn and the
    /// first actor -- into this state's own buffers.
    fn deal(&mut self, mut deck: Deck, button: usize, in_hand_mask: Option<&[bool]>) {
        let n = self.config.num_seats;
        assert!(n >= 2, "need at least 2 seats");
        // Named failure instead of the deck's index-out-of-bounds (PLO5 at
        // 9 seats, PLO6 at 8). The Python `GameConfig` and the PyO3
        // constructors reject these up front (review 2026-09-20 C4/B8).
        assert!(
            n <= self.config.variant.max_seats(),
            "deck cannot cover {n} seats for this variant (max {})",
            self.config.variant.max_seats()
        );
        assert!(button < n, "button out of range");
        if let Some(m) = in_hand_mask {
            assert_eq!(m.len(), n, "in_hand_mask length must equal num_seats");
            assert!(
                m.iter().filter(|&&b| b).count() >= 2,
                "in_hand_mask must include at least 2 seats"
            );
        }
        assert_eq!(
            self.config.starting_stacks.len(),
            n,
            "starting_stacks length must equal num_seats"
        );

        let variant = self.config.variant;
        let (hole_count, hole_slots) = (variant.hole_count(), variant.hole_slots());

        // Deal order (determinism contract): `hole_slots` cards per seat,
        // seat 0 first, then full board A, then full board B (two-board
        // variants only), then the burns (PLO67 only). Every variant but
        // PLO67 has `hole_slots == hole_count` and no burns: the same cards
        // in the same order as before PLO67 existed.
        self.hole_cards.resize_with(n, Vec::new);
        self.extra_holes.resize_with(n, Vec::new);
        for (h, extra) in self.hole_cards.iter_mut().zip(self.extra_holes.iter_mut()) {
            h.clear();
            for _ in 0..hole_count {
                h.push(deck.deal_one());
            }
            extra.clear();
            for _ in hole_count..hole_slots {
                extra.push(deck.deal_one());
            }
        }
        for c in self.full_board_a.iter_mut() {
            *c = deck.deal_one();
        }
        // Single-board variants leave board B undealt.
        self.full_board_b = [NO_CARD; 5];
        if variant.num_boards() == 2 {
            for c in self.full_board_b.iter_mut() {
                *c = deck.deal_one();
            }
        }
        self.full_burns.clear();
        for _ in 0..variant.burn_count() {
            self.full_burns.push(deck.deal_one());
        }

        // Preflop variants reveal nothing until the first round closes;
        // bomb pots start with both flops exposed.
        let preflop = variant.has_preflop();
        let street = if preflop {
            Street::Preflop
        } else {
            Street::Flop
        };
        self.board_a.clear();
        self.board_b.clear();
        if !preflop {
            self.board_a.extend_from_slice(&self.full_board_a[..3]);
            self.board_b.extend_from_slice(&self.full_board_b[..3]);
        }
        self.reset_table(button, street);
        self.post_antes(in_hand_mask);
        let blind_seats = preflop.then(|| self.post_blinds());
        // PLO67: the flop's burn is turned up before the flops come, and a
        // red one deals every seat in the hand its fifth card.
        if street == Street::Flop {
            self.reveal_burn();
        }

        self.actor = match blind_seats {
            Some((_, bb_seat)) => self.first_to_act_preflop(bb_seat),
            None => self.first_to_act_postflop(),
        };

        // If nobody can voluntarily act, auto-run to showdown: every seat
        // all-in from the antes / blinds, or a LONE seat with chips whose
        // opponents are all all-in for no more than it has already posted
        // (the actor walk skips it — `nothing_to_contest`, review
        // 2026-09-20 C1; it used to get a forced check node). The hand is
        // then terminal at deal.
        if self.actor.is_none() {
            self.run_out_to_showdown();
        }
    }

    /// A just-dealt hand before any chips go in (ENG-003: the one struct
    /// literal the three deal constructors share): every field at its
    /// deal-time value — stacks at the starting stacks, nothing committed or
    /// folded, no actor, not a study hand. The constructors then post the
    /// antes ([`Self::post_antes`]) and blinds ([`Self::post_blinds`]) in
    /// that order.
    fn skeleton(
        config: GameConfig,
        button: usize,
        hole_cards: Vec<Vec<Card>>,
        extra_holes: Vec<Vec<Card>>,
        (full_board_a, full_board_b): ([Card; 5], [Card; 5]),
        full_burns: Vec<Card>,
        (street, board_a, board_b): (Street, Vec<Card>, Vec<Card>),
    ) -> Self {
        let mut g = GameState::blank(config);
        g.hole_cards = hole_cards;
        g.extra_holes = extra_holes;
        g.full_board_a = full_board_a;
        g.full_board_b = full_board_b;
        g.full_burns = full_burns;
        g.board_a = board_a;
        g.board_b = board_b;
        g.reset_table(button, street);
        g
    }

    /// A state holding nothing yet: the one struct literal (every field named,
    /// so a new field is a compile error here), filled by [`Self::deal`] or
    /// [`Self::skeleton`].
    fn blank(config: GameConfig) -> Self {
        GameState {
            button: 0,
            sb_seat: None,
            bb_seat: None,
            street: Street::Flop,
            pot: 0,
            stacks: Vec::new(),
            folded: Vec::new(),
            all_in: Vec::new(),
            hole_cards: Vec::new(),
            extra_holes: Vec::new(),
            full_burns: Vec::new(),
            burns: Vec::new(),
            board_a: Vec::new(),
            board_b: Vec::new(),
            full_board_a: [NO_CARD; 5],
            full_board_b: [NO_CARD; 5],
            street_commit: Vec::new(),
            total_commit: Vec::new(),
            bet_to_call: 0,
            last_raise_size: 0,
            last_aggression_was_full_raise: true,
            street_level_acted: Vec::new(),
            actor: None,
            last_aggressor: None,
            acted_this_street: Vec::new(),
            history: Vec::new(),
            study_mode: false,
            awaiting_next_street: None,
            study_terminal: None,
            study_hero_seat: None,
            action_close_board_len: None,
            eff_stack_cap_at_hand_start: Vec::new(),
            config,
        }
    }

    /// Every field but the cards and the config at its deal-time value
    /// (ENG-003's one list): stacks at the starting stacks, nothing committed
    /// or folded, no actor, not a study hand -- in place, reusing the buffers
    /// (PERF-034). The destructure is exhaustive: a new field is a compile
    /// error here until it gets its deal-time value.
    fn reset_table(&mut self, button: usize, street: Street) {
        fn refill<T: Clone>(v: &mut Vec<T>, n: usize, x: T) {
            v.clear();
            v.resize(n, x);
        }
        let n = self.config.num_seats;
        let GameState {
            button: button_field,
            sb_seat,
            bb_seat,
            street: street_field,
            pot,
            stacks,
            folded,
            all_in,
            hole_cards: _,
            extra_holes: _,
            full_burns: _,
            burns,
            board_a: _,
            board_b: _,
            full_board_a: _,
            full_board_b: _,
            street_commit,
            total_commit,
            bet_to_call,
            last_raise_size,
            last_aggression_was_full_raise,
            street_level_acted,
            actor,
            last_aggressor,
            acted_this_street,
            history,
            study_mode,
            awaiting_next_street,
            study_terminal,
            study_hero_seat,
            action_close_board_len,
            eff_stack_cap_at_hand_start,
            config,
        } = self;
        *button_field = button;
        *sb_seat = None;
        *bb_seat = None;
        *street_field = street;
        *pot = 0;
        stacks.clone_from(&config.starting_stacks);
        refill(folded, n, false);
        refill(all_in, n, false);
        burns.clear();
        refill(street_commit, n, 0);
        refill(total_commit, n, 0);
        *bet_to_call = 0;
        *last_raise_size = config.bb;
        *last_aggression_was_full_raise = true;
        refill(street_level_acted, n, 0);
        *actor = None;
        *last_aggressor = None;
        refill(acted_this_street, n, false);
        history.clear();
        *study_mode = false;
        *awaiting_next_street = None;
        *study_terminal = None;
        *study_hero_seat = None;
        *action_close_board_len = None;
        eff_stack_cap_at_hand_start.clear();
    }

    /// Post the antes (dead): every dealt-in seat pays `min(stack, ante)` and
    /// is all-in if that was everything; a seat outside `in_hand` sits out —
    /// folded from the start, posting nothing. Then freeze the per-seat
    /// effective-stack caps (they depend on who is dealt in).
    fn post_antes(&mut self, in_hand: Option<&[bool]>) {
        let ante = self.config.ante;
        for i in 0..self.config.num_seats {
            if !in_hand.is_none_or(|m| m[i]) {
                self.folded[i] = true;
                continue;
            }
            let paid = self.stacks[i].min(ante);
            self.stacks[i] -= paid;
            self.total_commit[i] = paid;
            self.pot += paid;
            if self.stacks[i] == 0 {
                self.all_in[i] = true;
            }
        }
        eff_stack_cap_into(
            &mut self.eff_stack_cap_at_hand_start,
            &self.config.starting_stacks,
            &self.folded,
        );
    }

    /// Post the blinds of a variant with a preflop round, LIVE into
    /// `street_commit`, after the dead antes. Posting is not an action — no
    /// history record, `acted_this_street` stays false, so the BB option
    /// (the round can't close until the BB acts) falls out of the existing
    /// round-close machinery. Short posts go all-in; `bet_to_call` stays at
    /// the NOMINAL bb so callers owe the full blind and side pots absorb any
    /// shortfall. Returns and stores (sb seat, bb seat).
    fn post_blinds(&mut self) -> (usize, usize) {
        let (sb_seat, bb_seat) = nlh_blind_seats(self.config.num_seats, self.button, &self.folded);
        for (seat, amount) in [(sb_seat, self.config.sb), (bb_seat, self.config.bb)] {
            let paid = self.stacks[seat].min(amount);
            self.stacks[seat] -= paid;
            self.street_commit[seat] += paid;
            self.total_commit[seat] += paid;
            self.pot += paid;
            if self.stacks[seat] == 0 {
                self.all_in[seat] = true;
            }
        }
        self.bet_to_call = self.config.bb;
        self.sb_seat = Some(sb_seat);
        self.bb_seat = Some(bb_seat);
        (sb_seat, bb_seat)
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
    /// - `TooManySeats`: more seats than one deck deals through the river
    ///   (`Variant::max_seats`, 8 for PLO5).
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
            return Err(StudyError::UnsupportedVariant);
        }
        let n = config.num_seats;
        if n < 2 || button >= n || hero_seat >= n {
            return Err(StudyError::SeatOutOfRange);
        }
        if config.starting_stacks.len() != n {
            return Err(StudyError::StackCountMismatch);
        }
        // The placeholder deal needs 5 cards per non-hero seat on top of
        // the hero hole and both full boards; 10 seats used to index past
        // the 41-card unseen deck and panic (review 2026-09-20 C4).
        if n > config.variant.max_seats() {
            return Err(StudyError::TooManySeats);
        }
        if let Some(ref m) = in_hand_mask {
            if m.len() != n || !m[hero_seat] || m.iter().filter(|&&b| b).count() < 2 {
                return Err(StudyError::BadMask);
            }
        }

        // Validate 11 distinct card indices across hero hole + both flops.
        let mut used = CardMask::EMPTY;
        for &c in hero_hole.iter().chain(flop_a.iter()).chain(flop_b.iter()) {
            if !used.insert(c) {
                return Err(StudyError::DuplicateCard);
            }
        }

        // Seed a deck from a hash of all user inputs for deterministic opp hole draws.
        let seed = study_deal_seed(button, hero_seat, &[&hero_hole, &flop_a, &flop_b]);

        // Build the unseen deck (excluding the 11 user-supplied cards) and
        // shuffle it via ChaCha8Rng for bit-exact reproducibility.
        let unseen: Vec<Card> = used.unseen().collect();
        let mut deck_order = unseen;
        {
            use rand::seq::SliceRandom;
            use rand_chacha::rand_core::SeedableRng;
            use rand_chacha::ChaCha8Rng;
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

        // full_board_* carry the flop in [0..3] and NO_CARD in [3..5] (the UI
        // supplies the turn and river later).
        let mut full_board_a = [NO_CARD; 5];
        let mut full_board_b = [NO_CARD; 5];
        full_board_a[..3].copy_from_slice(&flop_a);
        full_board_b[..3].copy_from_slice(&flop_b);
        let mut state = GameState::skeleton(
            config,
            button,
            hole_cards,
            vec![Vec::new(); n],
            (full_board_a, full_board_b),
            Vec::new(),
            (Street::Flop, flop_a.to_vec(), flop_b.to_vec()),
        );
        state.study_mode = true;
        state.study_hero_seat = Some(hero_seat);
        // Antes exactly as new_hand: sitting-out seats post none, folded.
        state.post_antes(in_hand_mask.as_deref());
        state.actor = state.first_to_act_postflop();
        // Nobody can act at construction (every seat all-in from the
        // antes, or a lone seat with chips and nothing to contest): close
        // the round so the hand is CLASSIFIED — `study_terminal = RunOut`
        // — instead of sitting terminal with `study_terminal = None`,
        // which the UI cannot tell from a live hand (review 2026-09-20 C3).
        if state.actor.is_none() {
            state.close_round_or_run_out();
        }
        Ok(state)
    }

    /// Advance from flop to turn with user-supplied cards (study mode only).
    ///
    /// Requires `study_mode == true` and `awaiting_next_street == Some(Turn)`.
    /// `card_a` / `card_b` must not collide with any card already in use
    /// (hero hole, board A, board B); a collision with a hidden non-hero
    /// placeholder hole is not an error — that placeholder card is redrawn
    /// (`redraw_colliding_placeholders`). On success, boards are extended,
    /// the per-street state is reset, and `actor` is set to the first
    /// eligible seat clockwise from button.
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
        // placeholder draws not visible to the user, so they are not
        // grounds for rejection — `redraw_colliding_placeholders` moves
        // them out of the way instead.
        let hero_seat = self.study_hero_seat.ok_or(StudyError::WrongState)?;
        let mut used = CardMask::of(
            self.hole_cards[hero_seat]
                .iter()
                .chain(self.board_a.iter())
                .chain(self.board_b.iter()),
        );
        if !used.insert(card_a) || !used.insert(card_b) {
            return Err(StudyError::DuplicateCard);
        }

        let idx = match expected {
            Street::Turn => 3,
            Street::River => 4,
            _ => return Err(StudyError::WrongState),
        };
        self.redraw_colliding_placeholders(hero_seat, &[card_a, card_b])?;
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

    /// Study mode: a user-entered street card may coincide with a non-hero
    /// seat's PLACEHOLDER hole card — placeholders are hidden draws the
    /// user can neither see nor avoid. Left alone, that seat's hole and
    /// the board share a card, so its observation evaluates 5-card hands
    /// holding the same card twice — and the evaluator's `ck == 0` filter
    /// only catches the single-suit case (a duplicate inside a mixed-suit
    /// hand scores as a real pair). Fixed at the source: before
    /// `new_cards` land on the board, each colliding placeholder card is
    /// redrawn from the deck that is still unused once they do (hero
    /// hole, boards and every other placeholder excluded). The pick is a
    /// pinned hash of (street, card, seat, slot) over that unused deck —
    /// a pure function of the state, so action-log replays redraw
    /// identically. Hero and board cards are user-controlled and stay
    /// validated by the callers. (review 2026-09-20 C8)
    fn redraw_colliding_placeholders(
        &mut self,
        hero_seat: usize,
        new_cards: &[Card],
    ) -> Result<(), StudyError> {
        let mut used = CardMask::of(
            self.hole_cards
                .iter()
                .flatten()
                .chain(self.board_a.iter())
                .chain(self.board_b.iter())
                .chain(new_cards.iter()),
        );
        // Decide every replacement before touching the state, so an error
        // leaves the hand exactly as it was.
        let mut redraws: Vec<(usize, usize, Card)> = Vec::new();
        for &card in new_cards {
            for (seat, hole) in self.hole_cards.iter().enumerate() {
                if seat == hero_seat {
                    continue;
                }
                for (slot, &held) in hole.iter().enumerate() {
                    if held != card {
                        continue;
                    }
                    let unused: Vec<Card> = used.unseen().collect();
                    // Unreachable past the constructors' `max_seats`
                    // check (the full hand fits one deck); never index an
                    // empty deck regardless.
                    if unused.is_empty() {
                        return Err(StudyError::TooManySeats);
                    }
                    let mut mixer = SeedMixer::new();
                    mixer.write_u8(self.street.index() as u8);
                    mixer.write_u8(card.index());
                    mixer.write_u8(seat as u8);
                    mixer.write_u8(slot as u8);
                    let pick = unused[(mixer.finish() % unused.len() as u64) as usize];
                    used.add(pick);
                    redraws.push((seat, slot, pick));
                }
            }
        }
        for (seat, slot, pick) in redraws {
            self.hole_cards[seat][slot] = pick;
        }
        Ok(())
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
            return Err(StudyError::UnsupportedVariant);
        }
        let n = config.num_seats;
        if n < 2 || button >= n || hero_seat >= n {
            return Err(StudyError::SeatOutOfRange);
        }
        if config.starting_stacks.len() != n {
            return Err(StudyError::StackCountMismatch);
        }
        // Same deck bound as the PLO study constructor (review 2026-09-20 C4).
        if n > config.variant.max_seats() {
            return Err(StudyError::TooManySeats);
        }

        let mut used = CardMask::EMPTY;
        for &c in hero_hole.iter() {
            if !used.insert(c) {
                return Err(StudyError::DuplicateCard);
            }
        }

        // Deterministic placeholder holes from a hash of the user inputs
        // (same recipe as the PLO study constructor).
        let seed = study_deal_seed(button, hero_seat, &[&hero_hole]);
        let mut deck_order: Vec<Card> = used.unseen().collect();
        {
            use rand::seq::SliceRandom;
            use rand_chacha::rand_core::SeedableRng;
            use rand_chacha::ChaCha8Rng;
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

        let mut state = GameState::skeleton(
            config,
            button,
            hole_cards,
            vec![Vec::new(); n],
            ([NO_CARD; 5], [NO_CARD; 5]),
            Vec::new(),
            (Street::Preflop, Vec::new(), Vec::new()),
        );
        state.study_mode = true;
        state.study_hero_seat = Some(hero_seat);
        // Antes (dead) then blinds (live) — the exact new_hand sequence.
        state.post_antes(None);
        let (_, bb_seat) = state.post_blinds();
        state.actor = state.first_to_act_preflop(bb_seat);
        // Same construction-time classification as the PLO study
        // constructor: nobody can act → `study_terminal = RunOut`
        // (review 2026-09-20 C3).
        if state.actor.is_none() {
            state.close_round_or_run_out();
        }
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

    fn set_next_street_nlh(&mut self, expected: Street, cards: &[Card]) -> Result<(), StudyError> {
        if !self.study_mode || self.config.variant != Variant::NlhSingle {
            return Err(StudyError::WrongState);
        }
        if self.awaiting_next_street != Some(expected) {
            return Err(StudyError::WrongState);
        }
        let hero_seat = self.study_hero_seat.ok_or(StudyError::WrongState)?;
        let mut used = CardMask::of(self.hole_cards[hero_seat].iter().chain(self.board_a.iter()));
        for &c in cards {
            if !used.insert(c) {
                return Err(StudyError::DuplicateCard);
            }
        }
        self.redraw_colliding_placeholders(hero_seat, cards)?;
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
            let chips = self.sizing_chips(act, actor, min_total, max_total);
            let target_total = current_commit + chips;
            if target_total >= min_total && target_total <= max_total && chips > 0 {
                sizing_chips[i] = chips;
                sizing_feasible[i] = true;
            }
        }

        let shove_chips = stack;
        let allin_feasible = self.all_in_feasible(actor, min_total, max_total);

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

    /// `legal_action_mask()[Fold]` without building the mask: a seat is to
    /// act and faces a bet.
    #[inline]
    /// Whether the actor may shove, given the wager bounds: (a) the shove is
    /// at least a full raise (meets `min_total`), (b) facing a bet, it is
    /// strictly above the call but below the minimum (a short raise that
    /// doesn't reopen), or (c) not facing a bet, the stack can't reach 1bb (a
    /// short open that doesn't reset the min-raise floor). Cases (b) and (c)
    /// flow through `commit_chips`' short-shove branch. The caller has
    /// already ruled out an empty stack and the short-shove lockout.
    fn all_in_feasible(&self, actor: usize, min_total: u64, max_total: u64) -> bool {
        let shove_chips = self.stacks[actor];
        let shove_total = self.street_commit[actor] + shove_chips;
        let facing_bet = self.bet_to_call > self.street_commit[actor];
        let allin_is_full = shove_total >= min_total;
        let allin_is_short_raise =
            facing_bet && shove_total > self.bet_to_call && shove_total < min_total;
        let allin_is_short_open = !facing_bet && shove_total > 0 && shove_total < min_total;
        shove_chips > 0
            && shove_total <= max_total
            && (allin_is_full || allin_is_short_raise || allin_is_short_open)
    }

    /// `legal_action_mask()[AllIn]` without the five pot-fraction sizings the
    /// mask also works out (PERF-031: the hybrid path's gate 3 needs only
    /// this bit).
    pub fn all_in_is_legal(&self) -> bool {
        match self.actor {
            Some(a) if self.stacks[a] > 0 && !self.short_shove_lockout() => {
                self.all_in_feasible(a, self.min_bet_total(), self.max_bet_total())
            }
            _ => false,
        }
    }

    pub fn fold_is_legal(&self) -> bool {
        match self.actor {
            Some(a) => self.bet_to_call > self.street_commit[a],
            None => false,
        }
    }

    /// `legal_action_mask()[CheckCall]` without building the mask: any seat
    /// to act may check / call.
    #[inline]
    pub fn check_call_is_legal(&self) -> bool {
        self.actor.is_some()
    }

    /// Chip amount a given action contributes. `None` if illegal.
    pub fn action_to_chips(&self, action: Action) -> Option<u64> {
        // Fold / CheckCall legality is two field reads (the mask's own
        // rules for those bits); only sized actions need the full mask.
        let legal = match action {
            Action::Fold => self.fold_is_legal(),
            Action::CheckCall => self.check_call_is_legal(),
            _ => self.legal_action_mask()[action.index() as usize],
        };
        if !legal {
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

        if action == Action::Fold {
            self.folded[actor] = true;
            self.history.push(ActionRecord {
                seat: actor,
                action,
                chips: 0,
                street: self.street,
            });
            self.acted_this_street[actor] = true;
            self.street_level_acted[actor] = self.bet_to_call;
        } else {
            self.commit_chips(actor, chips, action);
        }
        self.advance_after_action(actor);
    }

    /// After `actor`'s action: a fold-out ends the hand; otherwise the next
    /// seat with a decision acts, or the betting round closes. The one tail of
    /// [`Self::apply`] and [`Self::apply_raise_chips`] (ENG-014).
    fn advance_after_action(&mut self, actor: usize) {
        // Fold-out: single non-folded seat wins.
        if self.alive_count() == 1 {
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

    /// Number of seats that have not folded (no allocation: this runs on
    /// every action of every hand).
    #[inline]
    fn alive_count(&self) -> usize {
        self.folded.iter().filter(|&&f| !f).count()
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
    /// study mode, fold-out (card-agnostic), river-close, `num_samples == 0`,
    /// and PLO67 (its undealt burns change the HANDS, so the actual deal is
    /// the answer — see `payouts` / `plo67_runout_equities`).
    ///
    /// Sum is zero-sum up to integer-division rounding (at most `num_seats`
    /// chips of rounding slack).
    pub fn payouts_ev(&self, num_samples: u32, seed: u64) -> Vec<i64> {
        use rand::Rng;
        use rand_chacha::rand_core::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let n = self.config.num_seats;
        // PLO67: the undealt burns decide how many cards each hand holds, so
        // resampling only the boards would score hands that were never dealt.
        // Nothing trains PLO67 (yet): the actual deal is the answer.
        if self.study_mode || num_samples == 0 || self.config.variant.burn_count() > 0 {
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
        let mut used = CardMask::of(self.hole_cards.iter().flatten());
        for &c in &self.full_board_a[..close_len] {
            used.add(c);
        }
        if num_boards == 2 {
            for &c in &self.full_board_b[..close_len] {
                used.add(c);
            }
        }
        let mut deck: Vec<Card> = used.unseen().collect();
        let deck_size = deck.len();

        let mut full_a = [NO_CARD; 5];
        let mut full_b = [NO_CARD; 5];
        full_a[..close_len].copy_from_slice(&self.full_board_a[..close_len]);
        full_b[..close_len].copy_from_slice(&self.full_board_b[..close_len]);

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut totals: Vec<i128> = vec![0i128; n];
        // The side-pot structure is the same for every sampled runout: build
        // it once, and reuse one output buffer (double-board variants). So
        // are each hand's encoded hole pairs and its score on the board
        // cards already out: the ranker holds both (identical ranks).
        let layers = PotLayers::new(&self.folded, &self.total_commit);
        // Next-card tables once the samples deal at least a deck's worth of
        // cards (they cost one evaluation pass per deck card; same ranks).
        let tables = (num_samples as usize) * missing >= deck_size;
        let ranker = (num_boards == 2).then(|| {
            RunoutRanker::new(
                &self.hole_cards,
                &self.folded,
                &full_a,
                &full_b,
                close_len,
                &deck,
                tables,
            )
        });
        let mut won_buf = vec![0u64; n];

        for _ in 0..num_samples {
            // Partial Fisher-Yates: shuffle only the first `draw_per_sample` cards.
            for i in 0..draw_per_sample {
                let j = rng.gen_range(i..deck_size);
                deck.swap(i, j);
            }
            full_a[close_len..close_len + missing].copy_from_slice(&deck[..missing]);
            if let Some(ranker) = ranker.as_ref() {
                full_b[close_len..close_len + missing].copy_from_slice(&deck[missing..2 * missing]);
                double_board_payout_runout(
                    &layers,
                    ranker,
                    &self.hole_cards,
                    &self.folded,
                    &self.total_commit,
                    &full_a,
                    &full_b,
                    self.button,
                    &mut won_buf,
                );
            } else {
                single_board_payout_layers(
                    &layers,
                    &self.hole_cards,
                    &self.folded,
                    &self.total_commit,
                    &full_a,
                    self.button,
                    &mut won_buf,
                );
            }
            for i in 0..n {
                totals[i] += won_buf[i] as i128;
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
        match self.actor {
            Some(a) => self.max_other_reachable_for(a),
            None => 0,
        }
    }

    /// [`Self::max_other_reachable_total`] for an arbitrary `seat` (the
    /// actor walks need it for seats that are not the actor yet).
    fn max_other_reachable_for(&self, seat: usize) -> u64 {
        (0..self.stacks.len())
            .filter(|&j| j != seat && !self.folded[j])
            .map(|j| self.street_commit[j] + self.stacks[j])
            .max()
            .unwrap_or(0)
    }

    /// True when `seat` has nothing left to contest on this street: no
    /// alive opponent can ever commit more than `seat` already has in
    /// (`street_commit + stack` is a seat's ceiling for the street, and
    /// the max over opponents only shrinks as seats fold — so once true
    /// it stays true until the street ends). Such a seat can neither
    /// face a real bet nor find a caller for a raise; the actor walks
    /// skip it and the hand runs out.
    ///
    /// (review 2026-09-20 C1) Without this, NLH's NOMINAL `bet_to_call`
    /// offered Fold against a phantom bet: HU [100bb, 0.8bb] with a 3000
    /// all-in BB post, the covering SB could "fold to 5000 more" and
    /// forfeit chips nobody had matched. The same walk also handed a lone
    /// seat with chips (every opponent all-in from the antes / blinds) a
    /// forced single-action check node at hand start. PLO mid-hand is
    /// unaffected: `bet_to_call` is always a real alive seat's commit and
    /// a street only opens with >= 2 seats holding chips.
    /// PRODUCTION BEHAVIOR CHANGE: those forced nodes no longer exist, so
    /// a deal where fewer than two seats can act is terminal AT DEAL
    /// (`actor == None` straight out of `new_hand`).
    fn nothing_to_contest(&self, seat: usize) -> bool {
        self.max_other_reachable_for(seat) <= self.street_commit[seat]
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
            && self.bet_to_call < self.street_level_acted[actor] + self.last_raise_size
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
        self.commit_chips(actor, chips, Action::BetPct100);
        self.advance_after_action(actor);
        Ok(())
    }

    /// `actor` puts `chips` in -- a call, a bet, a raise or an all-in, whatever
    /// the amount makes it (the history records it as `label`): stacks, pot and
    /// commits, the bet to call, the raise floor and the last aggressor, the
    /// all-in flag, and the seat's acted-this-street marks.
    fn commit_chips(&mut self, actor: usize, chips: u64, label: Action) {
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
                // Full raise (or opening bet >= 1bb): reopens action.
                self.last_raise_size = raise_delta;
                self.last_aggression_was_full_raise = true;
            } else {
                // Short shove below the min-raise floor: bet_to_call
                // advances but the floor is preserved and already-acted
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
            action: label,
            chips,
            street: self.street,
        });
        self.acted_this_street[actor] = true;
        self.street_level_acted[actor] = self.bet_to_call;
    }

    // ---- Internal helpers ----

    fn compute_sizing_chips(&self, action: Action, actor: usize) -> u64 {
        self.sizing_chips(action, actor, self.min_bet_total(), self.max_bet_total())
    }

    /// [`Self::compute_sizing_chips`] with the wager bounds already worked
    /// out (the legal mask sizes five actions against the same bounds).
    fn sizing_chips(&self, action: Action, actor: usize, min_total: u64, max_total: u64) -> u64 {
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
        let clamped = target_total.max(min_total).min(max_total);
        let chips_want = clamped.saturating_sub(current_commit);
        chips_want.min(stack)
    }

    /// First seat-to-act for a postflop betting round. Walks clockwise
    /// starting at `(button + 1) % num_seats` and returns the first seat
    /// that is neither folded nor all-in and still has something to
    /// contest (see [`Self::nothing_to_contest`]). Returns `None` if no
    /// eligible seat exists (caller should then close / run out /
    /// finalize).
    ///
    /// This is an **eligibility walk**, not a fixed offset: if e.g. SB
    /// check-folded on the flop, SB is skipped on the turn and the helper
    /// returns BB (or the next eligible seat clockwise).
    fn first_to_act_postflop(&self) -> Option<usize> {
        let n = self.config.num_seats;
        let start = (self.button + 1) % n;
        for i in 0..n {
            let s = (start + i) % n;
            if !self.folded[s] && !self.all_in[s] && !self.nothing_to_contest(s) {
                return Some(s);
            }
        }
        None
    }

    /// First seat-to-act for the preflop round: first non-folded,
    /// non-all-in seat with something to contest, walking clockwise from
    /// the seat after the big blind. Heads-up this lands on the button/SB
    /// (the only other in-hand seat), which is correct — the button acts
    /// first preflop. The walk wraps all the way to the BB itself, so the
    /// BB gets the option whenever an opponent can still respond to a
    /// raise. A blind who already covers every opponent's reach (all of
    /// them all-in for less) is skipped — there is nothing to call and
    /// nobody to raise — and the hand runs out (review 2026-09-20 C1).
    fn first_to_act_preflop(&self, bb_seat: usize) -> Option<usize> {
        let n = self.config.num_seats;
        let start = (bb_seat + 1) % n;
        for i in 0..n {
            let s = (start + i) % n;
            if !self.folded[s] && !self.all_in[s] && !self.nothing_to_contest(s) {
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
            // Nothing left to contest (review 2026-09-20 C1): every alive
            // opponent's reach is already covered by this seat's street
            // commit, so any `bet_to_call` above it is NLH's nominal bb,
            // not a real bet. No turn — the round closes and runs out.
            if self.nothing_to_contest(s) {
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
        if self.alive_count() == 1 {
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
            if self.alive_count() == 1 {
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
                    // PLO67: the turn's burn comes up (and may deal every
                    // live seat a card) before the turn cards.
                    self.reveal_burn();
                    self.board_a.push(self.full_board_a[3]);
                    if two_boards {
                        self.board_b.push(self.full_board_b[3]);
                    }
                }
                Street::River => {
                    self.reveal_burn();
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

            // Reset per-street state (in place: every per-seat vector is
            // num_seats long for the whole hand).
            debug_assert!(self.street_commit.len() == n && self.acted_this_street.len() == n);
            self.street_commit.fill(0);
            self.bet_to_call = 0;
            self.last_raise_size = self.config.bb;
            self.last_aggression_was_full_raise = true;
            self.street_level_acted.fill(0);
            self.last_aggressor = None;
            self.acted_this_street.fill(false);

            // Does a new round start? Need >=2 non-folded non-all-in seats.
            let can_act = (0..n)
                .filter(|&i| !self.folded[i] && !self.all_in[i])
                .count();
            if can_act >= 2 {
                self.actor = self.first_to_act_postflop();
                return;
            }
            // else continue looping — reveal next street, run out.
        }
    }

    /// Study-mode round-close: three outcomes.
    ///
    /// - Current street is River (however many seats can still act) →
    ///   `study_terminal = Some(Showdown)`, `actor = None`, `street = Showdown`.
    /// - `can_act.len() >= 2` and there's a next undealt street →
    ///   `awaiting_next_street = Some(next)`, `actor = None`. UI supplies cards.
    /// - `can_act.len() < 2` before the river (run-out case) →
    ///   `study_terminal = Some(RunOut)`, `actor = None`. No auto-run-out.
    fn close_round_study(&mut self) {
        // River first: every card is out, so a closed river round is a
        // showdown no matter how it closed. A river all-in + call leaves
        // fewer than two seats able to act and used to fall into the
        // RunOut arm below — the UI then claimed "turn/river cards were
        // not entered" on a complete board (review 2026-09-20 C3).
        if self.street == Street::River {
            self.study_terminal = Some(StudyTerminal::Showdown);
            self.actor = None;
            self.street = Street::Showdown;
            return;
        }
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
            Street::River | Street::Showdown => {
                // River returned above; Showdown is unreachable in study mode.
                self.actor = None;
            }
        }
    }

    /// Helper: run out all remaining streets to showdown (used when a hand
    /// begins in an already-run-out state, e.g. everyone all-in after antes).
    fn run_out_to_showdown(&mut self) {
        self.close_round_or_run_out();
    }

    /// PLO67: turn the next burn card face up; a red one deals every seat
    /// still in the hand (all-in included, folded / sitting-out not) its
    /// next reserved hole card. A no-op once every burn is up and for every
    /// variant without burns. Runs BEFORE the street's board cards appear,
    /// in every mode that reaches the street (played, run out, or at the
    /// deal for the flop).
    fn reveal_burn(&mut self) {
        let k = self.burns.len();
        let Some(&burn) = self.full_burns.get(k) else {
            return;
        };
        self.burns.push(burn);
        if !Variant::burn_is_red(burn) {
            return;
        }
        for seat in 0..self.config.num_seats {
            if self.folded[seat] {
                continue;
            }
            let hc = self.config.variant.hole_count();
            let got = self.hole_cards[seat].len() - hc;
            if let Some(&card) = self.extra_holes[seat].get(got) {
                self.hole_cards[seat].push(card);
            }
        }
    }

    /// PLO67: how many hole cards `seat` held on `street` (Showdown counts
    /// as the river). A seat receives an extra card at every red burn while
    /// it is in the hand, so its extras are a PREFIX of the red burns: on a
    /// street it holds `hole_count + min(red burns up to that street, extras
    /// it ever received)`. Every other variant: `hole_count` throughout.
    pub fn hole_count_on(&self, seat: usize, street: Street) -> usize {
        let hc = self.config.variant.hole_count();
        let upto = match street {
            Street::Preflop => 0,
            Street::Flop => 1,
            Street::Turn => 2,
            Street::River | Street::Showdown => 3,
        };
        let red = self
            .burns
            .iter()
            .take(upto)
            .filter(|&&c| Variant::burn_is_red(c))
            .count();
        let received = self.hole_cards[seat].len().saturating_sub(hc);
        hc + red.min(received)
    }

    fn finalize_terminal(&mut self) {
        // In production, reveal the full pre-dealt boards for observability.
        // In study mode, turn/river may be undealt (`NO_CARD`) so
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

/// PLO67 all-in runout equities: each contender's expected share of board A
/// and of board B (0..=1, a tie splits), given what the table can SEE —
/// the contenders' hole cards NOW, both boards so far and `dead` (the burns
/// turned up). Everything else, folded hands included, is unknown and
/// equally likely: every sample deals the rest of the hand exactly as the
/// game does — per street a burn, then (if it is red) one card to every
/// contender, then one card per board. Exact once both boards are complete.
/// Monte Carlo over `samples` runouts from `seed` (deterministic).
pub fn plo67_runout_equities(
    holes: &[Vec<Card>],
    board_a: &[Card],
    board_b: &[Card],
    dead: &[Card],
    samples: u32,
    seed: u64,
) -> Result<Vec<[f64; 2]>, String> {
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let n = holes.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if board_a.len() != board_b.len() || !(3..=5).contains(&board_a.len()) {
        return Err("both boards need the same 3..=5 cards".into());
    }
    let mut used = CardMask::EMPTY;
    for &c in holes
        .iter()
        .flatten()
        .chain(board_a)
        .chain(board_b)
        .chain(dead)
    {
        if !used.insert(c) {
            return Err(format!(
                "card {} is out of range or appears twice",
                c.index()
            ));
        }
    }
    if holes
        .iter()
        .any(|h| !(4..=crate::hand_eval::MAX_PLO_HOLE).contains(&h.len()))
    {
        return Err("every hand holds 4..=7 cards".into());
    }
    let missing = 5 - board_a.len();
    let mut stub: Vec<Card> = used.unseen().collect();
    // the most a runout can take: per street a burn, one card per hand, two board cards
    let need = missing * (3 + n);
    if need > stub.len() {
        return Err(format!(
            "{} unseen cards cannot run out {missing} streets for {n} hands",
            stub.len()
        ));
    }
    let score = |h: &[Card], a: &[Card; 5], b: &[Card; 5]| {
        [
            crate::hand_eval::evaluate_plo(h, a),
            crate::hand_eval::evaluate_plo(h, b),
        ]
    };
    // one runout's result: each board's best hand(s) take a share of 1
    fn add_shares(acc: &mut [[f64; 2]], ranks: &[[u32; 2]]) {
        for k in 0..2 {
            let best = ranks.iter().map(|r| r[k]).max().unwrap_or(0);
            let winners = ranks.iter().filter(|r| r[k] == best).count() as f64;
            for (s, r) in ranks.iter().enumerate() {
                if r[k] == best {
                    acc[s][k] += 1.0 / winners;
                }
            }
        }
    }
    let mut acc = vec![[0f64; 2]; n];
    let mut full_a = [NO_CARD; 5];
    let mut full_b = [NO_CARD; 5];
    full_a[..board_a.len()].copy_from_slice(board_a);
    full_b[..board_b.len()].copy_from_slice(board_b);
    let mut ranks: Vec<[u32; 2]> = vec![[0, 0]; n];
    if missing == 0 {
        for (s, h) in holes.iter().enumerate() {
            ranks[s] = score(h, &full_a, &full_b);
        }
        add_shares(&mut acc, &ranks);
        return Ok(acc);
    }
    let samples = samples.max(1);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut hands: Vec<Vec<Card>> = holes.to_vec();
    for _ in 0..samples {
        let mut next = 0usize;
        let mut draw = |stub: &mut Vec<Card>| {
            let j = rng.gen_range(next..stub.len());
            stub.swap(next, j);
            next += 1;
            stub[next - 1]
        };
        for (h, base) in hands.iter_mut().zip(holes) {
            h.truncate(base.len());
        }
        for street in 0..missing {
            let burn = draw(&mut stub);
            if Variant::burn_is_red(burn) {
                for h in hands.iter_mut() {
                    if h.len() < crate::hand_eval::MAX_PLO_HOLE {
                        h.push(draw(&mut stub));
                    }
                }
            }
            full_a[board_a.len() + street] = draw(&mut stub);
            full_b[board_b.len() + street] = draw(&mut stub);
        }
        for (s, h) in hands.iter().enumerate() {
            ranks[s] = score(h, &full_a, &full_b);
        }
        add_shares(&mut acc, &ranks);
    }
    let inv = 1.0 / samples as f64;
    for a in acc.iter_mut() {
        a[0] *= inv;
        a[1] *= inv;
    }
    Ok(acc)
}

/// Pinned 64-bit mixer behind every engine-derived RNG seed: FNV-1a over
/// the bytes written, finished with the splitmix64 output function.
/// Hand-written on purpose — `std`'s `DefaultHasher` (the previous seed
/// source) is explicitly unspecified across Rust releases, so a toolchain
/// bump could silently change every seeded draw. Pinned by
/// `b4_seed_mixer_known_answers`; do not "upgrade" it without versioning
/// the observation. (review 2026-09-20 B4)
pub(crate) struct SeedMixer(u64);

impl SeedMixer {
    pub(crate) fn new() -> Self {
        SeedMixer(0xCBF2_9CE4_8422_2325) // FNV-1a 64-bit offset basis
    }

    pub(crate) fn write_u8(&mut self, byte: u8) {
        self.0 = (self.0 ^ byte as u64).wrapping_mul(0x0000_0100_0000_01B3); // FNV prime
    }

    /// Write `cards` as a SET: length byte, then the indices in the
    /// project's canonical multiset order (card index descending), so the
    /// hash cannot see deal / entry order.
    pub(crate) fn write_card_set(&mut self, cards: &[Card]) {
        // A stack buffer (PERF-030): this runs for every row's MC seed / cache
        // key. Hole cards (<= 7) and boards (<= 5) always fit.
        let mut buf = [0u8; 16];
        let idx = &mut buf[..cards.len()];
        for (x, c) in idx.iter_mut().zip(cards) {
            *x = c.index();
        }
        idx.sort_unstable_by(|a, b| b.cmp(a));
        self.write_u8(idx.len() as u8);
        for &i in idx.iter() {
            self.write_u8(i);
        }
    }

    pub(crate) fn finish(self) -> u64 {
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Seed of a study hand's placeholder (non-hero) hole deal: button, hero
/// seat and the user-entered card sets through the pinned mixer, so the
/// same spot deals the same placeholders on every toolchain and whatever
/// order the cards were typed in. Placeholders never reach a model (study
/// recommendations are hero-only, the hero's observation is villain-blind),
/// so re-seeding them changes nothing a checkpoint sees.
/// (review 2026-09-20 B4)
fn study_deal_seed(button: usize, hero_seat: usize, card_sets: &[&[Card]]) -> u64 {
    let mut mixer = SeedMixer::new();
    mixer.write_u8(button as u8);
    mixer.write_u8(hero_seat as u8);
    for set in card_sets {
        mixer.write_card_set(set);
    }
    mixer.finish()
}

/// Small/big-blind seats for a variant with blinds, from the in-hand
/// (non-folded-at-deal) seats in clockwise order starting at the seat
/// after the button. 3+ handed: SB is the first, BB the next. Heads-up
/// (exactly 2 seats in hand): the BB is the first and the SB the second —
/// the second seat is the button itself, or, when the nominal button seat
/// sits out, the seat that inherits its position (last to act postflop).
///
/// (review 2026-09-20 C2) The heads-up arm used to take the first in-hand
/// seat AT/after the button as the SB. With the button on a sitting-out
/// seat that is the seat `first_to_act_postflop` also starts from, so the
/// SB acted first on every street. Identical whenever the button is in
/// the hand — and no caller deals NLH with an in-hand mask today, so
/// nothing trained or served changes.
fn nlh_blind_seats(n: usize, button: usize, folded: &[bool]) -> (usize, usize) {
    let order: Vec<usize> = (1..=n)
        .map(|i| (button + i) % n)
        .filter(|&s| !folded[s])
        .collect();
    assert!(order.len() >= 2, "blinds need at least 2 seats in hand");
    if order.len() == 2 {
        (order[1], order[0])
    } else {
        (order[0], order[1])
    }
}

/// Per-seat hand-start effective-stack cap. For seat `i`:
/// `min(starting_stacks[i], max(starting_stacks[j] for j != i and !folded[j]))`.
/// At construction `folded[i]` is true iff seat `i` is sitting out, so
/// the `max_other` reduction excludes sit-outs. Returns own stack when
/// no other in-hand seat exists.
pub fn compute_eff_stack_cap(starting_stacks: &[u64], folded: &[bool]) -> Vec<u64> {
    let mut cap = Vec::new();
    eff_stack_cap_into(&mut cap, starting_stacks, folded);
    cap
}

/// [`compute_eff_stack_cap`] into `out` (overwritten; its buffer reused).
fn eff_stack_cap_into(out: &mut Vec<u64>, starting_stacks: &[u64], folded: &[bool]) {
    let n = starting_stacks.len();
    out.clear();
    out.extend((0..n).map(|i| {
        let max_other = (0..n)
            .filter(|&j| j != i && !folded[j])
            .map(|j| starting_stacks[j])
            .max()
            .unwrap_or(starting_stacks[i]);
        starting_stacks[i].min(max_other)
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{flop3, hole5};

    fn default_config() -> GameConfig {
        GameConfig::default_6max_20bb()
    }

    /// The verifiable-shuffle entry point is the SAME deal as the seeded one:
    /// dealing from `Deck::new_shuffled(seed)`'s order reproduces the seeded
    /// hand card for card, and a card's slot is `5*seat + k` / `5n + m` /
    /// `5n + 5 + m` whoever is sitting out.
    /// PERF-034: re-dealing a used table is the fresh deal, field for field
    /// (Debug prints every field), whatever the previous hand left behind --
    /// another variant, seat count, stacks, a study hand, a finished hand.
    #[test]
    fn redeal_is_exactly_a_new_hand() {
        use crate::test_util::{flop3, hole5, nlh, plo};
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let configs = [
            plo(Variant::Plo5DoubleBomb, &[400_000; 6], 30_000, 10_000),
            plo(
                Variant::Plo4DoubleBomb,
                &[25_000, 900_000, 60_000],
                30_000,
                10_000,
            ),
            plo(Variant::Plo6DoubleBomb, &[200_000; 7], 30_000, 10_000),
            plo(
                Variant::Plo67DoubleBomb,
                &[300_000, 30_000, 500_000, 90_000],
                30_000,
                10_000,
            ),
            nlh(&[1_000_000, 8_000, 2_000_000], 5_000, 5_000, 10_000),
            plo(Variant::Plo5DoubleBomb, &[30_000, 30_000], 30_000, 10_000),
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(34);
        let mut g = GameState::new_study(
            plo(Variant::Plo5DoubleBomb, &[400_000; 6], 30_000, 10_000),
            0,
            0,
            hole5([0, 1, 2, 3, 4]),
            flop3([10, 11, 12]),
            flop3([20, 21, 22]),
        )
        .unwrap();
        for round in 0..300 {
            let cfg = &configs[round % configs.len()];
            let (seed, button) = (rng.gen::<u64>(), round % cfg.num_seats);
            g.redeal(cfg, seed, button);
            let want = GameState::new_hand(cfg.clone(), seed, button);
            assert_eq!(format!("{g:?}"), format!("{want:?}"), "round {round}");
            // Leave some state behind for the next re-deal.
            for _ in 0..rng.gen_range(0..12) {
                if g.is_terminal() {
                    break;
                }
                let mask = g.legal_action_mask();
                let legal: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
                g.apply(Action::from_index(legal[rng.gen_range(0..legal.len())] as u8).unwrap());
            }
        }
    }

    #[test]
    fn hand_from_explicit_deck_matches_the_seeded_deal() {
        for seed in [0u64, 7, 2026, u64::MAX >> 1] {
            let mask = Some(vec![true, false, true, true, false, true]);
            let a = GameState::new_hand_with_mask(default_config(), seed, 2, mask.clone());
            let order = Deck::new_shuffled(seed).order();
            let deck = Deck::from_order(&order).unwrap();
            let b = GameState::new_hand_from_deck(default_config(), deck, 2, mask);
            assert_eq!(a.hole_cards, b.hole_cards);
            assert_eq!(a.full_board_a, b.full_board_a);
            assert_eq!(a.full_board_b, b.full_board_b);
            assert_eq!(a.stacks, b.stacks);
            assert_eq!(a.actor, b.actor);
            let n = a.config.num_seats;
            for s in 0..n {
                for k in 0..5 {
                    assert_eq!(b.hole_cards[s][k].index(), order[5 * s + k]);
                }
            }
            for m in 0..5 {
                assert_eq!(b.full_board_a[m].index(), order[5 * n + m]);
                assert_eq!(b.full_board_b[m].index(), order[5 * n + 5 + m]);
            }
        }
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
        assert_eq!(
            g.eff_stack_cap_at_hand_start,
            vec![20 * bb, 30 * bb, 30 * bb]
        );
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
        assert_eq!(
            g.actor,
            Some(0),
            "SB acts first postflop in 3-way (button=2)"
        );
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
        assert!(
            !mask[Action::BetPct100 as usize],
            "B100 should be masked as AllIn dupe"
        );
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
        assert!(
            !mask[Action::AllIn as usize],
            "AllIn should be masked by PL cap"
        );
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
        g.set_turn(Card::from_index(11), Card::from_index(12))
            .unwrap();
        for _ in 0..6 {
            g.apply(Action::CheckCall);
        }
        g.set_river(Card::from_index(13), Card::from_index(14))
            .unwrap();
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
        g.set_turn(Card::from_index(11), Card::from_index(12))
            .unwrap();
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
        assert_eq!(
            ev, realized,
            "fold-out: EV must equal realized (card-agnostic)"
        );
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
            assert!(
                mask[pct as usize],
                "pct {pct:?} must be legal on fresh flop"
            );
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
                g_disc.last_aggression_was_full_raise, g_cont.last_aggression_was_full_raise,
                "pct={pct:?} full-raise flag mismatch"
            );
            assert_eq!(
                g_disc.acted_this_street, g_cont.acted_this_street,
                "pct={pct:?} acted flags mismatch"
            );
            assert_eq!(
                g_disc.actor, g_cont.actor,
                "pct={pct:?} next actor mismatch"
            );
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
        let board =
            |g: &GameState| -> Vec<u8> { g.full_board_a.iter().map(|c| c.index()).collect() };
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
            [
                Card::from_index(10),
                Card::from_index(11),
                Card::from_index(12),
            ],
            [
                Card::from_index(20),
                Card::from_index(21),
                Card::from_index(22),
            ],
        );
        // ENG-024: named for what it is (was the generic WrongState).
        assert_eq!(r.err(), Some(StudyError::UnsupportedVariant));
    }
}

#[cfg(test)]
mod plo6_tests {
    use super::*;

    fn plo6_cfg(num_seats: usize, stack: u64) -> GameConfig {
        crate::test_util::plo(
            Variant::Plo6DoubleBomb,
            &vec![stack; num_seats],
            30_000,
            10_000,
        )
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
        let fa = [
            Card::from_index(20),
            Card::from_index(24),
            Card::from_index(28),
        ];
        let fb = [
            Card::from_index(32),
            Card::from_index(36),
            Card::from_index(40),
        ];
        let r = GameState::new_study(cfg, 0, 0, hero, fa, fb);
        assert!(r.is_err(), "study mode is PLO5-only until the UI phase");
    }
}

#[cfg(test)]
mod plo4_tests {
    use super::*;

    fn plo4_cfg(num_seats: usize, stack: u64) -> GameConfig {
        crate::test_util::plo(
            Variant::Plo4DoubleBomb,
            &vec![stack; num_seats],
            30_000,
            10_000,
        )
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
        let fa = [
            Card::from_index(20),
            Card::from_index(24),
            Card::from_index(28),
        ];
        let fb = [
            Card::from_index(32),
            Card::from_index(36),
            Card::from_index(40),
        ];
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
    fn study_errors_name_the_problem() {
        // ENG-024: mask and stack-count problems used to read "seat out of range".
        let cfg = GameConfig::new_uniform(4, 200_000, 30_000, 10_000);
        let (h, fa, fb) = (
            [0u8, 1, 2, 3, 4].map(Card::from_index),
            [10u8, 11, 12].map(Card::from_index),
            [20u8, 21, 22].map(Card::from_index),
        );
        for mask in [
            vec![true; 3],
            vec![false, true, true, true],
            vec![true, false, false, false],
        ] {
            let r = GameState::new_study_with_mask(cfg.clone(), 0, 0, h, fa, fb, Some(mask));
            assert_eq!(r.err(), Some(StudyError::BadMask));
        }
        let mut short = cfg.clone();
        short.starting_stacks.pop();
        let r = GameState::new_study(short, 0, 0, h, fa, fb);
        assert_eq!(r.err(), Some(StudyError::StackCountMismatch));
        let mut nlh = GameConfig::new_nlh_uniform(3, 1_000_000, 5_000, 10_000, 0);
        nlh.starting_stacks.push(1);
        let r = GameState::new_study_nlh(nlh, 0, 0, hero2(0, 4));
        assert_eq!(r.err(), Some(StudyError::StackCountMismatch));
        assert!(StudyError::BadMask.to_string().contains("mask"));
    }

    #[test]
    fn nlh_study_rejected_for_plo() {
        let plo = GameConfig::new_uniform(6, 200_000, 30_000, 10_000);
        let r = GameState::new_study_nlh(plo, 0, 0, hero2(0, 4));
        assert_eq!(r.err(), Some(StudyError::UnsupportedVariant));
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

#[cfg(test)]
mod review_2026_09_20_tests {
    //! Regression tests for the 2026-09-20 code review: C1 (NLH phantom
    //! bet / lone actor), C2 (HU dead button), C3 (study terminal
    //! classification), C4 (seat-count panics), C8 (study placeholder
    //! collisions), B4 (opp-outcome MC seed).
    use super::*;
    use crate::test_util::{cards, flop3, hole5, nlh, plo};
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    const BB: u64 = 10_000;

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Every card in play (all holes + both progressive boards) is unique.
    fn assert_all_cards_distinct(g: &GameState, tag: &str) {
        let mut seen = [false; 52];
        for c in g
            .hole_cards
            .iter()
            .flatten()
            .chain(g.board_a.iter())
            .chain(g.board_b.iter())
        {
            assert!(!seen[c.index() as usize], "{tag}: card {c} appears twice");
            seen[c.index() as usize] = true;
        }
    }

    /// Settlement invariants of a terminal (non-study) hand: zero-sum;
    /// nobody loses more than they put in; nobody wins more than the other
    /// seats MATCHED against their own commit; a folded seat loses its
    /// commit — minus anything above every alive seat's commit, which
    /// nobody ever matched.
    pub(super) fn assert_settlement(g: &GameState, tag: &str) {
        assert!(g.is_terminal(), "{tag}: hand must be terminal");
        let n = g.config.num_seats;
        let p = g.payouts();
        assert_eq!(p.iter().sum::<i64>(), 0, "{tag}: payouts must be zero-sum");
        let alive_max = (0..n)
            .filter(|&i| !g.folded[i])
            .map(|i| g.total_commit[i])
            .max()
            .expect("at least one alive seat");
        for i in 0..n {
            let tc = g.total_commit[i];
            let matched: u64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| g.total_commit[j].min(tc))
                .sum();
            assert!(
                p[i] <= matched as i64,
                "{tag}: seat {i} won {} but only {matched} was matched (commits {:?})",
                p[i],
                g.total_commit
            );
            assert!(
                p[i] >= -(tc as i64),
                "{tag}: seat {i} lost more than it put in"
            );
            if g.folded[i] {
                assert_eq!(
                    p[i],
                    -(tc.min(alive_max) as i64),
                    "{tag}: folded seat {i} must lose exactly its matched commit"
                );
            }
        }
    }

    // ---- C1: NLH phantom bet / lone actor ----

    #[test]
    fn c1_covering_sb_is_not_offered_a_fold_vs_short_all_in_bb() {
        // Review scenario A. HU, button 0 = SB. After the 5000 antes the BB
        // has 3000 left and posts it all-in; the SB's 5000 post already
        // covers it. `bet_to_call` stays at the nominal 10000, and the SB
        // used to get Fold / "call 5000" — folding paid [-10000, +10000]
        // with only 8000 ever matched. Now: no node, the hand runs out.
        let g = GameState::new_hand(nlh(&[1_000_000, 8_000], 5_000, 5_000, 10_000), 11, 0);
        assert_eq!(g.total_commit, vec![10_000, 8_000]);
        assert!(g.all_in[1] && !g.all_in[0]);
        assert_eq!(g.actor, None, "nothing to contest: terminal at deal");
        assert_eq!(g.street, Street::Showdown);
        assert_eq!(g.board_a.len(), 5, "board runs out");
        assert!(g.history.is_empty(), "no forced action was recorded");
        assert_eq!(g.action_close_board_len, Some(0), "closed preflop");
        assert_settlement(&g, "scenario A");
        // The SB's uncalled 2000 comes back whatever the runout: it wins or
        // loses exactly the 8000 the BB matched (or chops).
        let p = g.payouts();
        assert!(
            p == vec![8_000, -8_000] || p == vec![-8_000, 8_000] || p == vec![0, 0],
            "unexpected payouts {p:?}"
        );
        // The EV path (5-card runout sampling from the deal) obeys the
        // same bound.
        let ev = g.payouts_ev(64, 99);
        assert!(ev[0].abs() <= 8_000 && ev[1].abs() <= 8_000, "ev {ev:?}");
    }

    #[test]
    fn c1_uncalled_sb_chips_are_not_a_pot_for_the_other_seats() {
        // Review scenario B. 3-way [51, 10000, 51], ante 50, blinds 50/100,
        // button 0 → SB seat 1 (deep), BB seat 2 (1 chip behind, all-in).
        // Once the BTN is all-in for its last chip (or folds), nobody can
        // reach the SB's 50: it must not be asked to fold to the nominal
        // 100, and its uncalled 49 must come back.
        let cfg = nlh(&[51, 10_000, 51], 50, 50, 100);

        // Line 1: BTN calls all-in for 1.
        let mut g = GameState::new_hand(cfg.clone(), 3, 0);
        assert_eq!(g.actor, Some(0), "BTN faces a REAL bet (the SB's 50)");
        g.apply(Action::CheckCall);
        assert!(g.is_terminal(), "SB has nothing to contest → run-out");
        assert_eq!(g.total_commit, vec![51, 100, 51]);
        assert!(!g.folded[1], "the SB was never offered a fold");
        assert_settlement(&g, "scenario B / BTN calls");
        let p = g.payouts();
        assert!(
            p[1] >= -51,
            "SB can lose only the 51 that was matched: {p:?}"
        );
        assert!(p[0] <= 102 && p[2] <= 102, "{p:?}");

        // Line 2: BTN folds.
        let mut g = GameState::new_hand(cfg, 3, 0);
        g.apply(Action::Fold);
        assert!(g.is_terminal());
        assert!(!g.folded[1]);
        assert_settlement(&g, "scenario B / BTN folds");
        let p = g.payouts();
        assert_eq!(p[0], -50, "BTN forfeits its ante");
        assert!(p[1] >= -51 && p[2] <= 101, "{p:?}");
    }

    #[test]
    fn c1_real_bet_from_a_short_all_in_blind_still_gets_a_response() {
        // The skip is about REACH, not about being all-in: a short BB whose
        // all-in post exceeds the SB's 5000 is a real 2000 raise the SB
        // must answer (call or fold), with no raise available.
        let mut g = GameState::new_hand(nlh(&[1_000_000, 12_000], 5_000, 5_000, 10_000), 5, 0);
        assert_eq!(g.street_commit, vec![5_000, 7_000]);
        assert_eq!(g.actor, Some(0));
        let mask = g.legal_action_mask();
        assert!(mask[Action::Fold as usize] && mask[Action::CheckCall as usize]);
        assert_eq!((g.min_raise_chips(), g.max_raise_chips()), (0, 0));
        g.apply(Action::Fold);
        assert_settlement(&g, "short BB real bet / SB folds");
        assert_eq!(g.payouts(), vec![-10_000, 10_000]);
    }

    #[test]
    fn c1_lone_stack_at_hand_start_runs_out() {
        // PLO: seats 0/1 are all-in from the ante, seat 2 has 47bb behind
        // and nobody to bet against. It used to get a forced check node.
        let cfg = plo(
            Variant::Plo5DoubleBomb,
            &[30_000, 30_000, 500_000],
            30_000,
            BB,
        );
        let g = GameState::new_hand(cfg, 5, 0);
        assert_eq!(g.actor, None, "lone stack: terminal at deal");
        assert_eq!(g.street, Street::Showdown);
        assert_eq!((g.board_a.len(), g.board_b.len()), (5, 5));
        assert!(g.history.is_empty());
        assert_eq!(g.action_close_board_len, Some(3), "closed on the flop");
        assert_eq!(g.stacks[2], 470_000, "the lone stack never moved");
        assert_settlement(&g, "PLO lone stack");

        // Two stacks behind → a normal hand.
        let cfg = plo(
            Variant::Plo5DoubleBomb,
            &[30_000, 500_000, 500_000],
            30_000,
            BB,
        );
        assert_eq!(GameState::new_hand(cfg, 5, 0).actor, Some(1));
    }

    #[test]
    fn c1_bb_option_survives_when_someone_can_respond() {
        // The skip must not eat a live BB option: limped pot, everyone
        // deep — the BB still acts last preflop.
        let mut g = GameState::new_hand(nlh(&[1_000_000; 3], 5_000, 5_000, 10_000), 1, 0);
        g.apply(Action::CheckCall); // BTN limps
        g.apply(Action::CheckCall); // SB completes
        assert_eq!(g.actor, Some(2), "BB option");
        assert_eq!(g.street, Street::Preflop);
    }

    /// Drive one hand with random legal actions, checking the node
    /// invariants the C1 fix is responsible for; returns the terminal state.
    pub(super) fn play_random_hand(
        cfg: GameConfig,
        seed: u64,
        button: usize,
        mask: Option<Vec<bool>>,
        rng: &mut ChaCha8Rng,
        tag: &str,
    ) -> GameState {
        let n = cfg.num_seats;
        let pot_limit = cfg.variant.pot_limit();
        let mut g = GameState::new_hand_with_mask(cfg, seed, button, mask);
        // At most one seat can ever be skipped for having nothing to
        // contest, so two live stacks always produce an actor; bomb pots
        // (no blinds) need exactly that.
        let live = (0..n).filter(|&i| !g.folded[i] && !g.all_in[i]).count();
        if live >= 2 {
            assert!(g.actor.is_some(), "{tag}: >= 2 live stacks but no actor");
        }
        if pot_limit {
            assert_eq!(g.actor.is_some(), live >= 2, "{tag}: PLO deal-time actor");
        }
        let mut steps = 0;
        while let Some(actor) = g.actor {
            assert!(
                !g.folded[actor] && !g.all_in[actor] && g.stacks[actor] > 0,
                "{tag}: actor {actor} cannot act"
            );
            let legal = g.legal_action_mask();
            assert!(
                legal[Action::CheckCall as usize],
                "{tag}: check/call illegal"
            );
            // The mask-free fast paths agree with the mask at every node.
            assert_eq!(
                g.fold_is_legal(),
                legal[Action::Fold as usize],
                "{tag}: fold fast path"
            );
            assert_eq!(
                g.check_call_is_legal(),
                legal[Action::CheckCall as usize],
                "{tag}: check/call fast path"
            );
            assert_eq!(
                g.all_in_is_legal(),
                legal[Action::AllIn as usize],
                "{tag}: all-in fast path"
            );
            if legal[Action::Fold as usize] {
                // No phantom bets: Fold is only ever offered against chips
                // an alive opponent actually has in front of them.
                let real_bet = (0..n).any(|j| {
                    j != actor && !g.folded[j] && g.street_commit[j] > g.street_commit[actor]
                });
                assert!(
                    real_bet,
                    "{tag}: seat {actor} offered Fold with no real bet (btc {} commits {:?})",
                    g.bet_to_call, g.street_commit
                );
            }
            let (min, max) = (g.min_raise_chips(), g.max_raise_chips());
            let discrete: Vec<u8> = (0..NUM_ACTIONS as u8)
                .filter(|&i| legal[i as usize])
                .collect();
            let can_size = min > 0 && max >= min;
            let roll = rng.gen_range(0..discrete.len() + if can_size { 2 } else { 0 });
            if roll >= discrete.len() {
                let chips = match rng.gen_range(0..3) {
                    0 => min,
                    1 => max,
                    _ => rng.gen_range(min..=max),
                };
                g.apply_raise_chips(chips).expect("in-range raise");
            } else {
                g.apply(Action::from_index(discrete[roll]).unwrap());
            }
            steps += 1;
            assert!(steps < 400, "{tag}: hand did not terminate");
        }
        let legal = g.legal_action_mask();
        assert_eq!(
            g.fold_is_legal(),
            legal[Action::Fold as usize],
            "{tag}: terminal fold"
        );
        assert_eq!(
            g.check_call_is_legal(),
            legal[Action::CheckCall as usize],
            "{tag}: terminal call"
        );
        assert!(!g.all_in_is_legal(), "{tag}: terminal all-in");
        g
    }

    #[test]
    fn c1_chip_conservation_property_micro_stacks() {
        // Heterogeneous stacks down to a single chip — below the ante and
        // the blinds, the regime where NLH's nominal `bet_to_call` diverges
        // from the real bets. Zero violations expected on every variant;
        // the pre-fix engine trips the phantom-bet and win<=matched checks
        // on `nlh_single` only.
        let mut rng = ChaCha8Rng::seed_from_u64(0xC1);
        let variants = [
            Variant::NlhSingle,
            Variant::NlhSingle,
            Variant::Plo5DoubleBomb,
            Variant::Plo4DoubleBomb,
            Variant::Plo6DoubleBomb,
        ];
        for hand in 0..4_000u64 {
            let variant = variants[rng.gen_range(0..variants.len())];
            let n = rng.gen_range(2..=6usize);
            let stacks: Vec<u64> = (0..n)
                .map(|_| match rng.gen_range(0..4) {
                    0 => rng.gen_range(1..=3 * BB),
                    1 | 2 => rng.gen_range(3 * BB..=40 * BB),
                    _ => rng.gen_range(40 * BB..=300 * BB),
                })
                .collect();
            let cfg = if variant == Variant::NlhSingle {
                let ante = if rng.gen_bool(0.5) { 5_000 } else { 0 };
                nlh(&stacks, ante, 5_000, BB)
            } else {
                let ante = if rng.gen_bool(0.5) { 30_000 } else { 10_000 };
                plo(variant, &stacks, ante, BB)
            };
            let mask = if n >= 3 && rng.gen_bool(0.3) {
                let mut m = vec![true; n];
                for _ in 0..rng.gen_range(1..=n - 2) {
                    let i = rng.gen_range(0..n);
                    m[i] = false;
                }
                Some(m)
            } else {
                None
            };
            let button = rng.gen_range(0..n);
            let seed = rng.gen::<u64>();
            let tag = format!(
                "hand {hand} {variant:?} stacks {stacks:?} button {button} mask {mask:?} \
                 seed {seed}"
            );
            let g = play_random_hand(cfg, seed, button, mask.clone(), &mut rng, &tag);
            assert_settlement(&g, &tag);
            assert_eq!(g.pot, g.total_commit.iter().sum::<u64>(), "{tag}: pot");
            for i in 0..n {
                let dealt = mask.as_ref().is_none_or(|m| m[i]);
                assert_eq!(
                    g.stacks[i] + g.total_commit[i],
                    stacks[i],
                    "{tag}: seat {i} chips"
                );
                if !dealt {
                    assert!(
                        g.folded[i] && g.total_commit[i] == 0,
                        "{tag}: sit-out seat {i}"
                    );
                }
                // Nobody folds while out-committing every alive seat — that
                // is what a fold to a phantom bet looks like in the books.
                if g.folded[i] {
                    let alive_max = (0..n)
                        .filter(|&j| !g.folded[j])
                        .map(|j| g.total_commit[j])
                        .max()
                        .unwrap();
                    assert!(
                        g.total_commit[i] <= alive_max,
                        "{tag}: folded seat {i} out-committed every alive seat"
                    );
                }
            }
        }
    }

    // ---- C2: heads-up NLH with the button on a sitting-out seat ----

    #[test]
    fn c2_heads_up_dead_button_bb_acts_first_postflop() {
        // 3 seats, button on seat 0 which sits out → HU between 1 and 2.
        // Clockwise from button+1 the order is [1, 2]: seat 1 is the BB,
        // seat 2 inherits the button (SB: first preflop, LAST postflop).
        let cfg = nlh(&[1_000_000; 3], 5_000, 5_000, 10_000);
        let mut g = GameState::new_hand_with_mask(cfg, 7, 0, Some(vec![false, true, true]));
        assert_eq!((g.sb_seat, g.bb_seat), (Some(2), Some(1)));
        assert_eq!(g.street_commit, vec![0, 10_000, 5_000]);
        assert_eq!(g.actor, Some(2), "SB acts first preflop");
        g.apply(Action::CheckCall); // SB completes
        assert_eq!(g.actor, Some(1), "BB option");
        g.apply(Action::CheckCall); // BB checks
        assert_eq!(g.street, Street::Flop);
        assert_eq!(g.actor, g.bb_seat, "BB acts first postflop");
        g.apply(Action::CheckCall);
        assert_eq!(g.actor, g.sb_seat, "SB (dead-button seat) acts last");

        // 6-max, button 3 sits out, only seats 0 and 4 dealt in.
        let cfg = nlh(&[1_000_000; 6], 5_000, 5_000, 10_000);
        let in_hand = vec![true, false, false, false, true, false];
        let mut g = GameState::new_hand_with_mask(cfg, 7, 3, Some(in_hand));
        assert_eq!((g.sb_seat, g.bb_seat), (Some(0), Some(4)));
        assert_eq!(g.actor, Some(0));
        g.apply(Action::CheckCall);
        g.apply(Action::CheckCall);
        assert_eq!(g.street, Street::Flop);
        assert_eq!(g.actor, Some(4), "BB acts first postflop");
    }

    #[test]
    fn c2_blind_seats_unchanged_when_the_button_is_in_the_hand() {
        let none = [false; 6];
        // Heads-up: the button is the SB.
        assert_eq!(nlh_blind_seats(2, 0, &none[..2]), (0, 1));
        assert_eq!(nlh_blind_seats(2, 1, &none[..2]), (1, 0));
        // HU inside a bigger table, button dealt in.
        let folded = [true, false, true, true, false, true];
        assert_eq!(nlh_blind_seats(6, 4, &folded), (4, 1));
        assert_eq!(nlh_blind_seats(6, 1, &folded), (1, 4));
        // 3+ handed: first two in-hand seats after the button.
        assert_eq!(nlh_blind_seats(6, 0, &none), (1, 2));
        assert_eq!(nlh_blind_seats(6, 5, &none), (0, 1));
        let folded = [false, true, false, false, true, false];
        assert_eq!(nlh_blind_seats(6, 0, &folded), (2, 3));
        assert_eq!(
            nlh_blind_seats(6, 4, &folded),
            (5, 0),
            "dead button, 4-handed"
        );
    }

    // ---- C3: study terminal classification ----

    /// HU PLO5 study hand: seat 0 has 3bb behind after the ante, seat 1 is
    /// deep and first to act.
    fn short_vs_deep_study() -> GameState {
        GameState::new_study(
            plo(Variant::Plo5DoubleBomb, &[60_000, 900_000], 30_000, BB),
            0,
            1,
            hole5([0, 1, 2, 3, 4]),
            flop3([5, 6, 7]),
            flop3([8, 9, 10]),
        )
        .unwrap()
    }

    #[test]
    fn c3_study_river_all_in_and_call_is_a_showdown() {
        // Checked to the river; seat 1 checks, seat 0 shoves, seat 1 calls.
        // Only one seat can still act, but every card is out: Showdown.
        let mut g = short_vs_deep_study();
        g.apply(Action::CheckCall);
        g.apply(Action::CheckCall);
        g.set_turn(Card::from_index(11), Card::from_index(12))
            .unwrap();
        g.apply(Action::CheckCall);
        g.apply(Action::CheckCall);
        g.set_river(Card::from_index(13), Card::from_index(14))
            .unwrap();
        g.apply(Action::CheckCall); // seat 1 checks
        g.apply(Action::AllIn); // seat 0 shoves 30000
        g.apply(Action::CheckCall); // seat 1 calls with chips behind
        assert!(g.is_terminal());
        assert!(g.all_in[0] && !g.all_in[1]);
        assert_eq!(g.study_terminal, Some(StudyTerminal::Showdown));
        assert_eq!(g.street, Street::Showdown);
        assert_eq!(g.awaiting_next_street, None);

        // The same line one street earlier is still a genuine run-out.
        let mut g = short_vs_deep_study();
        g.apply(Action::CheckCall);
        g.apply(Action::CheckCall);
        g.set_turn(Card::from_index(11), Card::from_index(12))
            .unwrap();
        g.apply(Action::CheckCall);
        g.apply(Action::AllIn);
        g.apply(Action::CheckCall);
        assert_eq!(g.study_terminal, Some(StudyTerminal::RunOut));
        assert_eq!(g.street, Street::Turn);
    }

    #[test]
    fn c3_nlh_study_river_all_in_and_call_is_a_showdown() {
        let cfg = nlh(&[60_000, 900_000], 5_000, 5_000, 10_000);
        let hero = [Card::from_index(51), Card::from_index(47)];
        let mut g = GameState::new_study_nlh(cfg, 0, 1, hero).unwrap();
        g.apply(Action::CheckCall);
        g.apply(Action::CheckCall);
        g.set_flop_nlh([
            Card::from_index(0),
            Card::from_index(5),
            Card::from_index(10),
        ])
        .unwrap();
        g.apply(Action::CheckCall);
        g.apply(Action::CheckCall);
        g.set_turn_nlh(Card::from_index(15)).unwrap();
        g.apply(Action::CheckCall);
        g.apply(Action::CheckCall);
        g.set_river_nlh(Card::from_index(20)).unwrap();
        g.apply(Action::CheckCall); // BB checks
        let shove = g.max_raise_chips();
        assert_eq!(shove, 45_000, "seat 0 has 4.5bb behind");
        g.apply_raise_chips(shove).unwrap();
        g.apply(Action::CheckCall);
        assert!(g.all_in[0] && !g.all_in[1]);
        assert_eq!(g.study_terminal, Some(StudyTerminal::Showdown));
        assert_eq!(g.street, Street::Showdown);
    }

    #[test]
    fn c3_study_hand_nobody_can_act_in_is_classified_at_construction() {
        let build = |stacks: &[u64]| {
            GameState::new_study(
                plo(Variant::Plo5DoubleBomb, stacks, 30_000, BB),
                0,
                1,
                hole5([0, 1, 2, 3, 4]),
                flop3([5, 6, 7]),
                flop3([8, 9, 10]),
            )
            .unwrap()
        };
        // Everyone all-in from the ante: terminal, and SAID to be.
        let g = build(&[30_000, 30_000, 30_000]);
        assert!(g.is_terminal());
        assert_eq!(g.study_terminal, Some(StudyTerminal::RunOut));
        assert_eq!(g.awaiting_next_street, None);
        assert_eq!(g.street, Street::Flop, "study never auto-deals streets");
        assert!(g.payouts().iter().all(|&x| x == 0));
        // A lone stack has nothing to contest either (C1).
        let g = build(&[30_000, 30_000, 500_000]);
        assert_eq!(g.actor, None);
        assert_eq!(g.study_terminal, Some(StudyTerminal::RunOut));
        // Control: two stacks behind → a live hand, unclassified.
        let g = build(&[30_000, 500_000, 500_000]);
        assert!(g.actor.is_some());
        assert_eq!(g.study_terminal, None);

        // NLH study: both seats all-in from the ante (blinds post 0).
        let g = GameState::new_study_nlh(
            nlh(&[5_000, 5_000], 5_000, 5_000, 10_000),
            0,
            0,
            [Card::from_index(51), Card::from_index(47)],
        )
        .unwrap();
        assert!(g.is_terminal());
        assert_eq!(g.study_terminal, Some(StudyTerminal::RunOut));
        assert_eq!(g.street, Street::Preflop);
    }

    // ---- C4: seat counts past the deck ----

    #[test]
    fn c4_max_seats_is_the_deck_bound() {
        assert_eq!(Variant::Plo4DoubleBomb.max_seats(), 10);
        assert_eq!(Variant::Plo5DoubleBomb.max_seats(), 8);
        assert_eq!(Variant::Plo6DoubleBomb.max_seats(), 7);
        assert_eq!(Variant::NlhSingle.max_seats(), 23);
        // The bound is tight: `max_seats` still deals, every card distinct.
        for variant in [
            Variant::Plo4DoubleBomb,
            Variant::Plo5DoubleBomb,
            Variant::Plo6DoubleBomb,
            Variant::NlhSingle,
        ] {
            let n = variant.max_seats();
            let mut cfg = plo(variant, &vec![200_000; n], 30_000, BB);
            cfg.sb = 5_000;
            let g = GameState::new_hand(cfg, 1, 0);
            let board_b = if variant.num_boards() == 2 { 5 } else { 0 };
            let mut seen = [false; 52];
            for c in g
                .hole_cards
                .iter()
                .flatten()
                .chain(g.full_board_a.iter())
                .chain(g.full_board_b.iter().take(board_b))
            {
                assert!(!seen[c.index() as usize], "{variant:?}: duplicate {c}");
                seen[c.index() as usize] = true;
            }
        }
    }

    #[test]
    #[should_panic(expected = "deck cannot cover 8 seats")]
    fn c4_new_hand_names_the_deck_overrun() {
        let cfg = plo(Variant::Plo6DoubleBomb, &[200_000; 8], 30_000, BB);
        let _ = GameState::new_hand(cfg, 1, 0);
    }

    #[test]
    fn c4_study_with_too_many_seats_is_an_error_not_a_panic() {
        for n in [9usize, 10, 12] {
            let r = GameState::new_study(
                plo(Variant::Plo5DoubleBomb, &vec![200_000; n], 30_000, BB),
                0,
                0,
                hole5([0, 1, 2, 3, 4]),
                flop3([5, 6, 7]),
                flop3([8, 9, 10]),
            );
            assert_eq!(r.err(), Some(StudyError::TooManySeats), "{n} seats");
        }
        let r = GameState::new_study_nlh(
            nlh(&[1_000_000; 24], 5_000, 5_000, 10_000),
            0,
            0,
            [Card::from_index(51), Card::from_index(47)],
        );
        assert_eq!(r.err(), Some(StudyError::TooManySeats));
        // The largest legal table still builds.
        assert!(GameState::new_study(
            plo(Variant::Plo5DoubleBomb, &[200_000; 8], 30_000, BB),
            0,
            0,
            hole5([0, 1, 2, 3, 4]),
            flop3([5, 6, 7]),
            flop3([8, 9, 10]),
        )
        .is_ok());
    }

    // ---- C8: study placeholder collisions ----

    #[test]
    fn c8_turn_and_river_cards_colliding_with_placeholders_are_redrawn() {
        let build = || {
            let mut g = GameState::new_study(
                GameConfig::default_6max_20bb(),
                0,
                1,
                hole5([0, 1, 2, 3, 4]),
                flop3([5, 6, 7]),
                flop3([8, 9, 10]),
            )
            .unwrap();
            for _ in 0..6 {
                g.apply(Action::CheckCall);
            }
            g
        };
        let mut g = build();
        assert_all_cards_distinct(&g, "study flop");
        // The user cannot see villain placeholders; entering two of them
        // as the turn cards is legal and must not leave duplicates behind.
        let (x, y) = (g.hole_cards[0][0], g.hole_cards[3][2]);
        let before = g.hole_cards.clone();
        g.set_turn(x, y).unwrap();
        assert_eq!((g.board_a[3], g.board_b[3]), (x, y));
        assert_all_cards_distinct(&g, "turn collision");
        for seat in 0..6 {
            for slot in 0..5 {
                let moved = g.hole_cards[seat][slot] != before[seat][slot];
                assert_eq!(
                    moved,
                    (seat, slot) == (0, 0) || (seat, slot) == (3, 2),
                    "only the colliding slots are redrawn (seat {seat} slot {slot})"
                );
            }
        }
        // A pure function of the state: a replay redraws identically.
        let mut replay = build();
        replay.set_turn(x, y).unwrap();
        assert_eq!(replay.hole_cards, g.hole_cards);

        // River: both cards collide with the SAME villain hand.
        for _ in 0..6 {
            g.apply(Action::CheckCall);
        }
        let (x, y) = (g.hole_cards[2][0], g.hole_cards[2][1]);
        g.set_river(x, y).unwrap();
        assert_all_cards_distinct(&g, "river double collision");
        // Every seat's features now come from duplicate-free 5-card hands:
        // the exhaustive per-board ahead/tie/behind split sums to 1.
        for seat in 0..6 {
            g.actor = Some(seat);
            let f = g.outcome_features_mc(8);
            let per_board: f32 = f[12..15].iter().sum();
            assert!((per_board - 1.0).abs() < 1e-5, "seat {seat}: {f:?}");
        }

        // Hero / board collisions are still the user's error.
        let mut g = build();
        assert_eq!(
            g.set_turn(Card::from_index(0), Card::from_index(20)),
            Err(StudyError::DuplicateCard)
        );
    }

    #[test]
    fn c8_nlh_street_cards_colliding_with_placeholders_are_redrawn() {
        let mut g = GameState::new_study_nlh(
            nlh(&[1_000_000; 6], 5_000, 5_000, 10_000),
            0,
            3,
            [Card::from_index(51), Card::from_index(47)],
        )
        .unwrap();
        for _ in 0..6 {
            g.apply(Action::CheckCall);
        }
        assert_eq!(g.awaiting_next_street, Some(Street::Flop));
        let flop = [g.hole_cards[0][0], g.hole_cards[0][1], g.hole_cards[5][1]];
        g.set_flop_nlh(flop).unwrap();
        assert_eq!(g.board_a, flop.to_vec());
        assert_all_cards_distinct(&g, "nlh flop collision");
        assert_eq!(g.hole_cards[3], cards(&[51, 47]), "hero untouched");
        for _ in 0..6 {
            g.apply(Action::CheckCall);
        }
        let turn = g.hole_cards[4][0];
        g.set_turn_nlh(turn).unwrap();
        assert_all_cards_distinct(&g, "nlh turn collision");
    }

    // ---- B4: opp-outcome MC seed ----

    #[test]
    fn b4_seed_mixer_known_answers() {
        // Values from an independent Python implementation (FNV-1a 64 →
        // splitmix64 finalizer). If this fails the mixer changed, and with
        // it every seeded MC draw and placeholder deal: that is an
        // observation-versioning event, not a constant to refresh.
        let run = |bytes: &[u8]| {
            let mut m = SeedMixer::new();
            for &b in bytes {
                m.write_u8(b);
            }
            m.finish()
        };
        assert_eq!(run(b""), 0xF52A_15E9_A9B5_E89B);
        assert_eq!(run(b"a"), 0x02C0_BDBF_4814_20F8);
        assert_eq!(run(b"foobar"), 0x404D_A9E3_B740_78C2);
        // Byte stream: street, then (len, indices descending) per set.
        let seed = outcome_mc_seed(
            Street::Flop,
            &cards(&[48, 44, 21, 10, 3]),
            &cards(&[50, 37, 8]),
            &cards(&[29, 17, 1]),
        );
        assert_eq!(seed, 0x8AC9_3A61_860F_F1BE);
        let shuffled = outcome_mc_seed(
            Street::Flop,
            &cards(&[3, 48, 10, 21, 44]),
            &cards(&[8, 50, 37]),
            &cards(&[17, 1, 29]),
        );
        assert_eq!(shuffled, seed, "card order must not reach the seed");
        // Swapping the boards is a different spot.
        let swapped = outcome_mc_seed(
            Street::Flop,
            &cards(&[48, 44, 21, 10, 3]),
            &cards(&[29, 17, 1]),
            &cards(&[50, 37, 8]),
        );
        assert_ne!(swapped, seed);
        assert_eq!(
            study_deal_seed(
                0,
                1,
                &[
                    &cards(&[0, 1, 2, 3, 4]),
                    &cards(&[5, 6, 7]),
                    &cards(&[8, 9, 10])
                ]
            ),
            0xFA2A_C0D0_F984_1FEA
        );
    }

    #[test]
    fn b4_outcome_mc_ignores_card_order_and_absolute_seat() {
        for variant in [
            Variant::Plo5DoubleBomb,
            Variant::Plo4DoubleBomb,
            Variant::Plo6DoubleBomb,
        ] {
            let n = 6;
            let mut g = GameState::new_hand(plo(variant, &[200_000; 6], 30_000, BB), 2024, 0);
            // Turn street, so the board set has a non-flop card to move.
            g.board_a.push(g.full_board_a[3]);
            g.board_b.push(g.full_board_b[3]);
            g.street = Street::Turn;
            let hero = g.actor.unwrap();
            let base = g.outcome_features_mc(64);
            let base_seed = g.outcome_seed();
            assert!(base_seed.is_some());
            assert!(
                base[4..12].iter().any(|&x| x > 0.0),
                "{variant:?}: MC arms must be live"
            );

            // Same cards, different deal / entry order.
            let mut permuted = g.clone();
            permuted.hole_cards[hero].reverse();
            permuted.board_a.swap(0, 3); // a flop card trades places with the turn
            permuted.board_b.rotate_left(1);
            assert_eq!(permuted.outcome_seed(), base_seed, "{variant:?}");
            assert_eq!(
                bits(&permuted.outcome_features_mc(64)),
                bits(&base),
                "{variant:?}: card order changed the MC dims"
            );

            // Same cards, every other absolute seat.
            for shift in 1..n {
                let other = (hero + shift) % n;
                let mut moved = g.clone();
                moved.hole_cards.swap(hero, other);
                moved.actor = Some(other);
                assert_eq!(moved.outcome_seed(), base_seed, "{variant:?}");
                assert_eq!(
                    bits(&moved.outcome_features_mc(64)),
                    bits(&base),
                    "{variant:?}: hero seat {other} changed the MC dims"
                );
            }

            // A different hole is a different stream.
            let mut villain = g.clone();
            villain.actor = Some((hero + 1) % n);
            assert_ne!(villain.outcome_seed(), base_seed, "{variant:?}");
        }
    }

    #[test]
    fn b4_study_placeholder_deal_ignores_card_entry_order() {
        let cfg = GameConfig::default_6max_20bb();
        let a = GameState::new_study(
            cfg.clone(),
            0,
            1,
            hole5([0, 1, 2, 3, 4]),
            flop3([5, 6, 7]),
            flop3([8, 9, 10]),
        )
        .unwrap();
        let b = GameState::new_study(
            cfg.clone(),
            0,
            1,
            hole5([4, 2, 0, 3, 1]),
            flop3([7, 5, 6]),
            flop3([10, 8, 9]),
        )
        .unwrap();
        for seat in [0usize, 2, 3, 4, 5] {
            assert_eq!(a.hole_cards[seat], b.hole_cards[seat], "seat {seat}");
        }
        assert_all_cards_distinct(&a, "study deal");
        // ... while a different spot deals different placeholders.
        let c = GameState::new_study(
            cfg,
            0,
            1,
            hole5([0, 1, 2, 3, 11]),
            flop3([5, 6, 7]),
            flop3([8, 9, 10]),
        )
        .unwrap();
        assert_ne!(a.hole_cards[0], c.hole_cards[0]);
    }
}

#[cfg(test)]
mod plo67_tests {
    //! PLO67 (2026-09-27): four hole cards, the three burns dealt FACE UP,
    //! and every red burn deals each seat still in the hand one more card.
    use super::review_2026_09_20_tests::{assert_settlement, play_random_hand};
    use super::*;
    use crate::hand_eval::{evaluate_5, evaluate_plo, HandRank};
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    const BB: u64 = 10_000;
    const ANTE: u64 = 30_000;
    // card index = rank * 4 + suit (suit 1 = diamonds, 2 = hearts: red)
    const RED: [u8; 3] = [1, 2, 5]; // 2d 2h 3d
    const BLACK: [u8; 3] = [0, 3, 4]; // 2c 2s 3c

    fn cfg(stacks: &[u64]) -> GameConfig {
        crate::test_util::plo(Variant::Plo67DoubleBomb, stacks, ANTE, BB)
    }

    /// A shuffled deck with `burns` in the three burn slots (7n+10..7n+13);
    /// every other card keeps its shuffled order.
    fn deck_with_burns(n: usize, burns: [u8; 3], seed: u64) -> Vec<u8> {
        let order = crate::cards::Deck::new_shuffled(seed).order();
        let mut rest: Vec<u8> = order
            .iter()
            .copied()
            .filter(|c| !burns.contains(c))
            .collect();
        for (k, &b) in burns.iter().enumerate() {
            rest.insert(7 * n + 10 + k, b);
        }
        assert_eq!(rest.len(), 52);
        rest
    }

    fn deal(stacks: &[u64], order: &[u8]) -> GameState {
        let deck = crate::cards::Deck::from_order(order).unwrap();
        GameState::new_hand_from_deck(cfg(stacks), deck, 0, None)
    }

    fn idx(cards: &[Card]) -> Vec<u8> {
        cards.iter().map(|c| c.index()).collect()
    }

    fn counts(g: &GameState, seat: usize) -> [usize; 3] {
        [
            g.hole_count_on(seat, Street::Flop),
            g.hole_count_on(seat, Street::Turn),
            g.hole_count_on(seat, Street::River),
        ]
    }

    fn brute(hole: &[Card], board: &[Card; 5]) -> HandRank {
        let mut best = 0;
        for i in 0..hole.len() {
            for j in (i + 1)..hole.len() {
                for a in 0..5 {
                    for b in (a + 1)..5 {
                        for c in (b + 1)..5 {
                            best = best.max(evaluate_5(&[
                                hole[i], hole[j], board[a], board[b], board[c],
                            ]));
                        }
                    }
                }
            }
        }
        best
    }

    #[test]
    fn deck_budget_is_five_seats_and_other_variants_are_unchanged() {
        let v = Variant::Plo67DoubleBomb;
        assert_eq!((v.hole_count(), v.hole_slots(), v.burn_count()), (4, 7, 3));
        assert_eq!(v.max_seats(), 5);
        assert_eq!(v.cards_needed(5), 48);
        assert!(v.cards_needed(6) > crate::cards::DECK_SIZE);
        assert!(v.pot_limit() && !v.has_preflop());
        assert_eq!(v.num_boards(), 2);
        for (v, hc, max) in [
            (Variant::Plo4DoubleBomb, 4, 10),
            (Variant::Plo5DoubleBomb, 5, 8),
            (Variant::Plo6DoubleBomb, 6, 7),
            (Variant::NlhSingle, 2, 23),
        ] {
            assert_eq!(
                (v.hole_slots(), v.burn_count(), v.max_seats()),
                (hc, 0, max),
                "{v:?}"
            );
        }
        for (red, card) in [
            (false, 0u8),
            (true, 1),
            (true, 2),
            (false, 3),
            (true, 50),
            (false, 51),
        ] {
            assert_eq!(
                Variant::burn_is_red(Card::from_index(card)),
                red,
                "card {card}"
            );
        }
    }

    #[test]
    fn the_deal_follows_the_public_slot_map() {
        for n in 2..=5usize {
            for burns in [RED, BLACK] {
                let order = deck_with_burns(n, burns, 70 + n as u64);
                let g = deal(&vec![1_000_000; n], &order);
                let held = if burns == RED { 5 } else { 4 };
                for s in 0..n {
                    assert_eq!(
                        idx(&g.hole_cards[s]),
                        order[7 * s..7 * s + held].to_vec(),
                        "n {n} seat {s}"
                    );
                    assert_eq!(idx(&g.extra_holes[s]), order[7 * s + 4..7 * s + 7].to_vec());
                }
                assert_eq!(idx(&g.full_board_a), order[7 * n..7 * n + 5].to_vec());
                assert_eq!(idx(&g.full_board_b), order[7 * n + 5..7 * n + 10].to_vec());
                assert_eq!(idx(&g.full_burns), burns.to_vec());
                // the flop's burn is face up from the deal; the others wait
                assert_eq!(idx(&g.burns), vec![burns[0]]);
                assert_eq!(
                    (g.street, g.board_a.len(), g.board_b.len()),
                    (Street::Flop, 3, 3)
                );
            }
        }
        // the seeded deal is the same contract: 48 distinct cards for 5 seats
        let g = GameState::new_hand(cfg(&[500_000; 5]), 12345, 2);
        let mut all: Vec<u8> = Vec::new();
        for s in 0..5 {
            // held + still reserved = the seat's seven slots
            let held = g.hole_cards[s].len();
            all.extend(idx(&g.hole_cards[s]));
            all.extend(idx(&g.extra_holes[s][held - 4..]));
        }
        all.extend(idx(&g.full_board_a));
        all.extend(idx(&g.full_board_b));
        all.extend(idx(&g.full_burns));
        assert_eq!(all.len(), 48);
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 48);
    }

    #[test]
    fn red_burns_deal_live_seats_and_all_in_seats_but_not_folded_ones() {
        // Button 0: seat 1 acts first. Seat 2 is short (70k behind the ante).
        let order = deck_with_burns(3, RED, 1);
        let mut g = deal(&[1_000_000, 1_000_000, 100_000], &order);
        assert!(
            g.hole_cards.iter().all(|h| h.len() == 5),
            "red flop burn: five cards each"
        );
        assert_eq!(g.actor, Some(1));
        g.apply(Action::CheckCall); // seat 1 checks
        g.apply(Action::AllIn); // seat 2 all-in (70k)
        g.apply(Action::CheckCall); // seat 0 calls
        g.apply(Action::Fold); // seat 1 folds
                               // one seat left with chips: the turn and river are run out, burns and all
        assert!(g.is_terminal());
        assert_eq!(idx(&g.burns), RED.to_vec());
        assert_eq!(
            g.hole_cards.iter().map(|h| h.len()).collect::<Vec<_>>(),
            vec![7, 5, 7]
        );
        assert_eq!(counts(&g, 0), [5, 6, 7]);
        assert_eq!(
            counts(&g, 1),
            [5, 5, 5],
            "folded before the turn burn: no more cards"
        );
        assert_eq!(
            counts(&g, 2),
            [5, 6, 7],
            "all-in seats keep receiving cards"
        );
        for s in [0usize, 2] {
            assert_eq!(idx(&g.hole_cards[s]), order[7 * s..7 * s + 7].to_vec());
        }
        // the showdown scores all seven cards: one layer of 230k (the folded
        // seat's ante is dead money in it), half per board
        let p = g.payouts();
        assert_eq!(p.iter().sum::<i64>(), 0);
        assert_eq!(p[1], -30_000);
        let mut won = [0i64; 3];
        for board in [&g.full_board_a, &g.full_board_b] {
            let (r0, r2) = (
                brute(&g.hole_cards[0], board),
                brute(&g.hole_cards[2], board),
            );
            assert_eq!(evaluate_plo(&g.hole_cards[0], board), r0);
            assert_eq!(evaluate_plo(&g.hole_cards[2], board), r2);
            match r0.cmp(&r2) {
                std::cmp::Ordering::Greater => won[0] += 115_000,
                std::cmp::Ordering::Less => won[2] += 115_000,
                std::cmp::Ordering::Equal => {
                    won[0] += 57_500;
                    won[2] += 57_500;
                }
            }
        }
        assert_eq!(p[0], won[0] - 100_000);
        assert_eq!(p[2], won[2] - 100_000);
        assert_settlement(&g, "red run-out");
    }

    #[test]
    fn black_burns_deal_nothing() {
        let order = deck_with_burns(3, BLACK, 2);
        let mut g = deal(&[1_000_000; 3], &order);
        for _ in 0..9 {
            g.apply(Action::CheckCall);
        }
        assert!(g.is_terminal());
        assert_eq!(idx(&g.burns), BLACK.to_vec());
        for s in 0..3 {
            assert_eq!(g.hole_cards[s].len(), 4);
            assert_eq!(counts(&g, s), [4, 4, 4]);
        }
        assert_settlement(&g, "black check-down");
    }

    #[test]
    fn a_seat_that_folds_misses_every_later_card() {
        // burns: black flop, red turn, red river
        let order = deck_with_burns(3, [BLACK[0], RED[0], RED[1]], 3);
        let mut g = deal(&[1_000_000; 3], &order);
        assert!(g.hole_cards.iter().all(|h| h.len() == 4));
        g.apply(Action::CheckCall); // seat 1 checks
        g.apply(Action::BetPct50); // seat 2 bets
        g.apply(Action::Fold); // seat 0 folds
        g.apply(Action::CheckCall); // seat 1 calls -> turn
        assert_eq!(g.street, Street::Turn);
        assert_eq!(g.burns.len(), 2);
        assert_eq!(
            g.hole_cards.iter().map(|h| h.len()).collect::<Vec<_>>(),
            vec![4, 5, 5]
        );
        g.apply(Action::CheckCall);
        g.apply(Action::CheckCall); // -> river
        assert_eq!(g.street, Street::River);
        assert_eq!(
            g.hole_cards.iter().map(|h| h.len()).collect::<Vec<_>>(),
            vec![4, 6, 6]
        );
        g.apply(Action::CheckCall);
        g.apply(Action::CheckCall);
        assert!(g.is_terminal());
        assert_eq!(counts(&g, 0), [4, 4, 4]);
        assert_eq!(counts(&g, 1), [4, 5, 6]);
        assert_eq!(counts(&g, 2), [4, 5, 6]);
        assert_settlement(&g, "fold before the red burns");
    }

    #[test]
    fn a_fold_out_turns_up_no_more_burns() {
        let order = deck_with_burns(2, RED, 4);
        let mut g = deal(&[1_000_000; 2], &order);
        // heads-up, button 0: seat 1 acts first
        g.apply(Action::BetPct50);
        g.apply(Action::Fold);
        assert!(g.is_terminal());
        assert_eq!(g.burns.len(), 1, "the hand ended on the flop");
        assert_eq!(g.full_burns.len(), 3, "the rabbit can still show the rest");
        assert_eq!(g.hole_cards[1].len(), 5);
        assert_settlement(&g, "fold-out");
    }

    #[test]
    fn everyone_all_in_from_the_ante_runs_out_every_burn() {
        let order = deck_with_burns(4, RED, 5);
        let g = deal(&[ANTE; 4], &order);
        assert!(g.is_terminal(), "nobody can act: terminal at the deal");
        assert_eq!(idx(&g.burns), RED.to_vec());
        assert!(g.hole_cards.iter().all(|h| h.len() == 7));
        assert_settlement(&g, "ante all-in");
    }

    #[test]
    fn ev_payouts_are_the_actual_deal() {
        let order = deck_with_burns(3, RED, 6);
        let mut g = deal(&[1_000_000, 1_000_000, 100_000], &order);
        g.apply(Action::CheckCall);
        g.apply(Action::AllIn);
        g.apply(Action::CheckCall);
        g.apply(Action::Fold);
        assert_eq!(g.payouts_ev(64, 9), g.payouts());
    }

    #[test]
    fn runout_equities_are_exact_on_the_river_and_sum_to_one() {
        let c = |i: u8| Card::from_index(i);
        // river: complete boards -> the actual result, shares sum to 1 per board
        let holes = vec![
            vec![c(48), c(49), c(0), c(4), c(8)],
            vec![c(44), c(45), c(1), c(5), c(9), c(13)],
        ];
        let a = [c(50), c(51), c(20), c(24), c(28)];
        let b = [c(40), c(41), c(21), c(25), c(29)];
        let eq = plo67_runout_equities(&holes, &a, &b, &[c(2)], 100, 1).unwrap();
        for k in 0..2 {
            let board = if k == 0 { &a } else { &b };
            let (r0, r1) = (
                evaluate_plo(&holes[0], board),
                evaluate_plo(&holes[1], board),
            );
            let want = match r0.cmp(&r1) {
                std::cmp::Ordering::Greater => [1.0, 0.0],
                std::cmp::Ordering::Less => [0.0, 1.0],
                std::cmp::Ordering::Equal => [0.5, 0.5],
            };
            assert_eq!([eq[0][k], eq[1][k]], want);
        }
        // flop all-in, three hands: shares are probabilities that sum to 1
        let holes = vec![
            vec![c(48), c(49), c(0), c(4)],
            vec![c(44), c(45), c(1), c(5), c(9)],
            vec![c(40), c(41), c(2), c(6)],
        ];
        let eq = plo67_runout_equities(
            &holes,
            &[c(50), c(51), c(20)],
            &[c(36), c(37), c(21)],
            &[c(3)],
            2000,
            7,
        )
        .unwrap();
        for k in 0..2 {
            let total: f64 = eq.iter().map(|e| e[k]).sum();
            assert!((total - 1.0).abs() < 1e-9, "board {k}: {total}");
        }
        // deterministic from the seed
        let again = plo67_runout_equities(
            &holes,
            &[c(50), c(51), c(20)],
            &[c(36), c(37), c(21)],
            &[c(3)],
            2000,
            7,
        )
        .unwrap();
        assert_eq!(eq, again);
        // a repeated card is refused, never a panic
        assert!(plo67_runout_equities(
            &holes,
            &[c(48), c(51), c(20)],
            &[c(36), c(37), c(21)],
            &[],
            10,
            1
        )
        .is_err());
    }

    #[test]
    fn runout_equities_match_the_engine_deal_distribution() {
        // The sampler must deal the rest of the hand the way the engine does:
        // compare its flop-all-in equity with the frequency the engine's own
        // run-outs produce over many shuffles of the unseen cards.
        let order = deck_with_burns(2, [BLACK[0], RED[0], RED[1]], 11);
        let g0 = deal(&[ANTE; 2], &order); // all-in from the ante: runs out at the deal
        assert!(g0.is_terminal());
        let holes: Vec<Vec<Card>> = (0..2)
            .map(|s| {
                order[7 * s..7 * s + 4]
                    .iter()
                    .map(|&i| Card::from_index(i))
                    .collect()
            })
            .collect();
        let (fa, fb): (Vec<Card>, Vec<Card>) =
            (g0.full_board_a[..3].to_vec(), g0.full_board_b[..3].to_vec());
        let burn0 = g0.full_burns[0];
        let eq = plo67_runout_equities(&holes, &fa, &fb, &[burn0], 20_000, 3).unwrap();
        // the engine: keep the visible cards, reshuffle every other card into its slots
        let mut visible: Vec<u8> = holes.iter().flatten().map(|c| c.index()).collect();
        visible.extend(fa.iter().chain(fb.iter()).map(|c| c.index()));
        visible.push(burn0.index());
        let unseen: Vec<u8> = (0..52u8).filter(|i| !visible.contains(i)).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mut won = [[0f64; 2]; 2];
        let trials = 20_000;
        for _ in 0..trials {
            let mut rest = unseen.clone();
            for i in (1..rest.len()).rev() {
                let j = rng.gen_range(0..=i);
                rest.swap(i, j);
            }
            let mut deck = vec![0u8; 52];
            let mut it = rest.into_iter();
            for s in 0..2usize {
                for k in 0..7 {
                    deck[7 * s + k] = if k < 4 {
                        holes[s][k].index()
                    } else {
                        it.next().unwrap()
                    };
                }
            }
            for m in 0..5 {
                deck[14 + m] = if m < 3 {
                    fa[m].index()
                } else {
                    it.next().unwrap()
                };
                deck[19 + m] = if m < 3 {
                    fb[m].index()
                } else {
                    it.next().unwrap()
                };
            }
            deck[24] = burn0.index();
            deck[25] = it.next().unwrap();
            deck[26] = it.next().unwrap();
            for (k, v) in deck.iter_mut().enumerate().skip(27) {
                *v = it.next().unwrap_or(k as u8);
            }
            // (the tail past the last slot is never dealt)
            let tail: Vec<u8> = (0..52u8).filter(|c| !deck[..27].contains(c)).collect();
            deck[27..].copy_from_slice(&tail);
            let g = deal(&[ANTE; 2], &deck);
            for (k, board) in [&g.full_board_a, &g.full_board_b].into_iter().enumerate() {
                let (r0, r1) = (
                    evaluate_plo(&g.hole_cards[0], board),
                    evaluate_plo(&g.hole_cards[1], board),
                );
                match r0.cmp(&r1) {
                    std::cmp::Ordering::Greater => won[0][k] += 1.0,
                    std::cmp::Ordering::Less => won[1][k] += 1.0,
                    std::cmp::Ordering::Equal => {
                        won[0][k] += 0.5;
                        won[1][k] += 0.5;
                    }
                }
            }
        }
        for s in 0..2 {
            for k in 0..2 {
                let engine = won[s][k] / trials as f64;
                assert!(
                    (engine - eq[s][k]).abs() < 0.025,
                    "seat {s} board {k}: engine {engine} sampler {}",
                    eq[s][k]
                );
            }
        }
    }

    #[test]
    fn random_hands_settle_and_hold_the_right_number_of_cards() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x67);
        for hand in 0..3_000u64 {
            let n = rng.gen_range(2..=5usize);
            let stacks: Vec<u64> = (0..n)
                .map(|_| match rng.gen_range(0..4) {
                    0 => rng.gen_range(1..=3 * BB),
                    1 | 2 => rng.gen_range(3 * BB..=40 * BB),
                    _ => rng.gen_range(40 * BB..=300 * BB),
                })
                .collect();
            let mask = if n >= 3 && rng.gen_bool(0.3) {
                let mut m = vec![true; n];
                m[rng.gen_range(0..n)] = false;
                Some(m)
            } else {
                None
            };
            let button = rng.gen_range(0..n);
            let seed = rng.gen::<u64>();
            let tag = format!("hand {hand} stacks {stacks:?} mask {mask:?} seed {seed}");
            let g = play_random_hand(cfg(&stacks), seed, button, mask.clone(), &mut rng, &tag);
            assert_settlement(&g, &tag);
            let red = g.burns.iter().filter(|&&c| Variant::burn_is_red(c)).count();
            for s in 0..n {
                let dealt = mask.as_ref().is_none_or(|m| m[s]);
                let have = g.hole_cards[s].len();
                if !dealt {
                    assert_eq!(have, 4, "{tag}: a sitting-out seat never gets extras");
                } else if !g.folded[s] {
                    assert_eq!(have, 4 + red, "{tag}: seat {s} in to the end");
                } else {
                    assert!(have <= 4 + red, "{tag}: folded seat {s}");
                }
                assert_eq!(g.hole_count_on(s, Street::River), have, "{tag}");
                let hc =
                    [Street::Flop, Street::Turn, Street::River].map(|st| g.hole_count_on(s, st));
                assert!(hc[0] <= hc[1] && hc[1] <= hc[2], "{tag}: counts only grow");
            }
            // every card on the table is distinct
            let mut all: Vec<u8> = g.hole_cards.iter().flat_map(|h| idx(h)).collect();
            all.extend(idx(&g.board_a));
            all.extend(idx(&g.board_b));
            all.extend(idx(&g.burns));
            let len = all.len();
            all.sort_unstable();
            all.dedup();
            assert_eq!(all.len(), len, "{tag}: duplicate card");
        }
    }
}
