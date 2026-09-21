//! Compact public betting state for CFR trees.
//!
//! Owns chip transitions without cloning [`crate::state::GameState`].
//! Rules mirror `engine.rs` integer math (min/max raise, short-shove reopen,
//! pot-after-call sizing) for NLH no-limit HU postflop roots.

use super::CfrError;

/// Maximum seats the compact public state supports (multiway up to 6).
pub const MAX_SEATS: usize = 6;

/// Compact public node used by the CFR tree (no hole cards, no history vec).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicState {
    pub num_seats: u8,
    pub pot: u64,
    pub stacks: [u64; MAX_SEATS],
    pub street_commit: [u64; MAX_SEATS],
    pub total_commit: [u64; MAX_SEATS],
    pub bet_to_call: u64,
    pub last_raise_size: u64,
    pub last_aggression_was_full_raise: bool,
    pub street_level_acted: [u64; MAX_SEATS],
    pub acted_this_street: [bool; MAX_SEATS],
    pub actor: Option<u8>,
    pub folded: [bool; MAX_SEATS],
    pub all_in: [bool; MAX_SEATS],
    /// Board card indices 0..51; `board_len` valid prefix.
    pub board: [u8; 5],
    pub board_len: u8,
    pub button: u8,
    pub bb: u64,
    pub last_aggressor: Option<u8>,
    /// 0=preflop 1=flop 2=turn 3=river 4=showdown/terminal
    pub street: u8,
}

impl PublicState {
    /// HU postflop solver root: fixed board (3/4/5), pot already in middle,
    /// equal remaining stacks, no street commits, OOP acts first.
    pub fn river_hu_root(
        pot_chips: u64,
        stack_chips: u64,
        board: &[u8],
        bb: u64,
    ) -> Result<Self, CfrError> {
        Self::hu_postflop_root(pot_chips, stack_chips, board, bb)
    }

    pub fn hu_postflop_root(
        pot_chips: u64,
        stack_chips: u64,
        board: &[u8],
        bb: u64,
    ) -> Result<Self, CfrError> {
        let blen = board.len();
        if !(3..=5).contains(&blen) {
            return Err(CfrError::InvalidRoot(format!(
                "postflop board must have 3..=5 cards, got {blen}"
            )));
        }
        if pot_chips == 0 || stack_chips == 0 || bb == 0 {
            return Err(CfrError::InvalidRoot(
                "pot, stack, bb must be > 0".into(),
            ));
        }
        let mut seen = [false; 52];
        let mut board_arr = [0u8; 5];
        for (i, &c) in board.iter().enumerate() {
            if c >= 52 {
                return Err(CfrError::InvalidRoot(format!("card {c} out of range")));
            }
            if seen[c as usize] {
                return Err(CfrError::InvalidRoot("duplicate board card".into()));
            }
            seen[c as usize] = true;
            board_arr[i] = c;
        }
        let street = match blen {
            3 => 1u8,
            4 => 2,
            _ => 3,
        };
        let button = 1u8;
        let mut s = Self {
            num_seats: 2,
            pot: pot_chips,
            stacks: [0; MAX_SEATS],
            street_commit: [0; MAX_SEATS],
            total_commit: [0; MAX_SEATS],
            bet_to_call: 0,
            last_raise_size: bb,
            last_aggression_was_full_raise: true,
            street_level_acted: [0; MAX_SEATS],
            acted_this_street: [false; MAX_SEATS],
            actor: Some(0),
            folded: [false; MAX_SEATS],
            all_in: [false; MAX_SEATS],
            board: board_arr,
            board_len: blen as u8,
            button,
            bb,
            last_aggressor: None,
            street,
        };
        s.stacks[0] = stack_chips;
        s.stacks[1] = stack_chips;
        Ok(s)
    }

    /// Multiway postflop root: `stacks[0..n]` remaining, pot in middle, OOP first.
    pub fn postflop_root(
        num_seats: u8,
        pot_chips: u64,
        stacks: &[u64],
        board: &[u8],
        bb: u64,
        street: u8,
    ) -> Result<Self, CfrError> {
        let n = num_seats as usize;
        if !(2..=MAX_SEATS).contains(&n) {
            return Err(CfrError::InvalidRoot(format!("num_seats {n} out of 2..{MAX_SEATS}")));
        }
        if stacks.len() != n {
            return Err(CfrError::InvalidRoot("stacks len != num_seats".into()));
        }
        if pot_chips == 0 || bb == 0 {
            return Err(CfrError::InvalidRoot("pot and bb must be > 0".into()));
        }
        let need = match street {
            1 => 3,
            2 => 4,
            3 => 5,
            _ => {
                return Err(CfrError::InvalidRoot(
                    "postflop_root street must be 1..3".into(),
                ))
            }
        };
        if board.len() != need {
            return Err(CfrError::InvalidRoot(format!(
                "board len {} != {need} for street {street}",
                board.len()
            )));
        }
        let mut seen = [false; 52];
        let mut board_arr = [0u8; 5];
        for (i, &c) in board.iter().enumerate() {
            if c >= 52 || seen[c as usize] {
                return Err(CfrError::InvalidRoot("bad board card".into()));
            }
            seen[c as usize] = true;
            board_arr[i] = c;
        }
        let button = (n as u8).wrapping_sub(1); // last seat is button
        let first = 0u8; // seat 0 = UTG / OOP for our convention
        let mut s = Self {
            num_seats,
            pot: pot_chips,
            stacks: [0; MAX_SEATS],
            street_commit: [0; MAX_SEATS],
            total_commit: [0; MAX_SEATS],
            bet_to_call: 0,
            last_raise_size: bb,
            last_aggression_was_full_raise: true,
            street_level_acted: [0; MAX_SEATS],
            acted_this_street: [false; MAX_SEATS],
            actor: Some(first),
            folded: [false; MAX_SEATS],
            all_in: [false; MAX_SEATS],
            board: board_arr,
            board_len: need as u8,
            button,
            bb,
            last_aggressor: None,
            street,
        };
        for i in 0..n {
            s.stacks[i] = stacks[i];
        }
        // A seat that arrives with no chips is already all-in (review 2026-09-20 E1).
        s.seal_root();
        Ok(s)
    }

    /// Preflop root: antes (dead) + SB/BB (live `street_commit`) posted, short
    /// posts go all-in, `bet_to_call` is the NOMINAL big blind (engine rule).
    ///
    /// Seat convention: HU → seat 0 = BB, seat 1 = BTN/SB (acts first);
    /// 3+ seats → seats `0..n-3` UTG.., `n-2` = SB, `n-1` = BB, seat 0 first.
    ///
    /// `extra_dead_chips` is added to the pot on top of the posts (HU preflop
    /// passes `pot_bb − posts`, normally 0).
    ///
    /// (review 2026-09-20 E1) One validated constructor instead of four
    /// hand-rolled copies that posted blinds with `saturating_sub` and no
    /// all-in flag.
    pub fn preflop_root(
        stacks: &[u64],
        bb: u64,
        sb: u64,
        ante: u64,
        extra_dead_chips: u64,
    ) -> Result<Self, CfrError> {
        let n = stacks.len();
        if !(2..=MAX_SEATS).contains(&n) {
            return Err(CfrError::InvalidRoot(format!(
                "num_seats {n} out of 2..{MAX_SEATS}"
            )));
        }
        if bb == 0 {
            return Err(CfrError::InvalidRoot("bb must be > 0".into()));
        }
        if stacks.iter().any(|&s| s == 0) {
            return Err(CfrError::InvalidRoot("preflop stacks must be > 0 chips".into()));
        }
        let (sb_seat, bb_seat, first, button) = if n == 2 {
            (1usize, 0usize, 1u8, 1u8)
        } else {
            (n - 2, n - 1, 0u8, (n - 3) as u8)
        };
        let mut s = Self {
            num_seats: n as u8,
            pot: extra_dead_chips,
            stacks: [0; MAX_SEATS],
            street_commit: [0; MAX_SEATS],
            total_commit: [0; MAX_SEATS],
            bet_to_call: bb,
            last_raise_size: bb,
            last_aggression_was_full_raise: true,
            street_level_acted: [0; MAX_SEATS],
            acted_this_street: [false; MAX_SEATS],
            actor: Some(first),
            folded: [false; MAX_SEATS],
            all_in: [false; MAX_SEATS],
            board: [0; 5],
            board_len: 0,
            button,
            bb,
            last_aggressor: None,
            street: 0,
        };
        for i in 0..n {
            let ante_pay = ante.min(stacks[i]);
            s.stacks[i] = stacks[i] - ante_pay;
            s.total_commit[i] = ante_pay;
            s.pot += ante_pay;
        }
        for (seat, blind) in [(sb_seat, sb), (bb_seat, bb)] {
            let pay = blind.min(s.stacks[seat]);
            s.stacks[seat] -= pay;
            s.street_commit[seat] = pay;
            s.total_commit[seat] += pay;
            s.pot += pay;
        }
        s.seal_root();
        Ok(s)
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.num_seats as usize
    }

    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.actor.is_none()
    }

    pub fn alive_count(&self) -> usize {
        (0..self.n()).filter(|&i| !self.folded[i]).count()
    }

    pub fn min_bet_total(&self) -> u64 {
        if self.bet_to_call == 0 {
            self.bb
        } else {
            self.bet_to_call + self.last_raise_size
        }
    }

    pub fn max_other_reachable_total(&self) -> u64 {
        let actor = match self.actor {
            Some(a) => a as usize,
            None => return 0,
        };
        (0..self.n())
            .filter(|&j| j != actor && !self.folded[j])
            .map(|j| self.street_commit[j] + self.stacks[j])
            .max()
            .unwrap_or(0)
    }

    /// NLH no-limit: cap is deepest opponent reachable total.
    pub fn max_bet_total(&self) -> u64 {
        self.max_other_reachable_total()
    }

    fn short_shove_lockout(&self) -> bool {
        let actor = match self.actor {
            Some(a) => a as usize,
            None => return false,
        };
        let facing_bet = self.bet_to_call > self.street_commit[actor];
        facing_bet
            && self.acted_this_street[actor]
            && self.bet_to_call < self.street_level_acted[actor] + self.last_raise_size
    }

    pub fn min_raise_chips(&self) -> u64 {
        let actor = match self.actor {
            Some(a) => a as usize,
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
        let max_other = self.max_other_reachable_total();
        if max_other <= self.bet_to_call {
            return 0;
        }
        let cap_delta = max_other.saturating_sub(current_commit);
        if cap_delta == 0 {
            return 0;
        }
        let clamped = delta.min(cap_delta);
        if clamped > stack {
            return 0;
        }
        clamped
    }

    pub fn max_raise_chips(&self) -> u64 {
        let actor = match self.actor {
            Some(a) => a as usize,
            None => return 0,
        };
        if self.short_shove_lockout() {
            return 0;
        }
        let current_commit = self.street_commit[actor];
        let stack = self.stacks[actor];
        let max_total = self.max_bet_total();
        let cap_total = max_total.min(current_commit + stack);
        if cap_total <= current_commit || cap_total <= self.bet_to_call {
            return 0;
        }
        cap_total - current_commit
    }

    /// Chip delta to call (0 if checking).
    pub fn to_call_chips(&self) -> u64 {
        let actor = match self.actor {
            Some(a) => a as usize,
            None => return 0,
        };
        let need = self.bet_to_call.saturating_sub(self.street_commit[actor]);
        need.min(self.stacks[actor])
    }

    /// Pot-fraction raise chip delta (per-mille), clamped to legal range.
    /// Same convention as engine `compute_sizing_chips` / pot-after-call.
    pub fn raise_chips_for_pm(&self, pm: u32) -> Option<u64> {
        let actor = self.actor? as usize;
        let current_commit = self.street_commit[actor];
        let stack = self.stacks[actor];
        let to_call_raw = self.bet_to_call.saturating_sub(current_commit);
        // Saturating like the engine's sizing (`engine_bridge::raise_pm_chips`):
        // an absurd per-mille from Python must not overflow-panic.
        let raise_over = if self.bet_to_call == 0 {
            self.pot.saturating_mul(pm as u64) / 1000
        } else {
            let pot_after_call = self.pot.saturating_add(to_call_raw);
            pot_after_call.saturating_mul(pm as u64) / 1000
        };
        let target_total = if self.bet_to_call == 0 {
            raise_over
        } else {
            self.bet_to_call.saturating_add(raise_over)
        };
        let min_total = self.min_bet_total();
        let max_total = self.max_bet_total();
        let clamped = target_total.max(min_total).min(max_total);
        let chips_want = clamped.saturating_sub(current_commit).min(stack);
        let min_r = self.min_raise_chips();
        let max_r = self.max_raise_chips();
        if min_r == 0 || chips_want < min_r || chips_want > max_r {
            return None;
        }
        Some(chips_want)
    }

    pub fn apply_fold(&mut self) {
        let actor = self.actor.expect("fold on terminal") as usize;
        self.folded[actor] = true;
        self.acted_this_street[actor] = true;
        self.street_level_acted[actor] = self.bet_to_call;
        self.after_action(actor);
    }

    pub fn apply_check_call(&mut self) {
        let actor = self.actor.expect("xc on terminal") as usize;
        let chips = self.to_call_chips();
        if chips > 0 {
            self.stacks[actor] -= chips;
            self.pot += chips;
            self.street_commit[actor] += chips;
            self.total_commit[actor] += chips;
            if self.stacks[actor] == 0 {
                self.all_in[actor] = true;
            }
        }
        self.acted_this_street[actor] = true;
        self.street_level_acted[actor] = self.bet_to_call;
        self.after_action(actor);
    }

    pub fn apply_raise(&mut self, chips: u64) -> Result<(), CfrError> {
        let actor = self.actor.ok_or_else(|| {
            CfrError::InvalidConfig("raise on terminal".into())
        })? as usize;
        let min = self.min_raise_chips();
        let max = self.max_raise_chips();
        if min == 0 || chips < min || chips > max {
            return Err(CfrError::InvalidConfig(format!(
                "raise chips {chips} not in [{min},{max}]"
            )));
        }
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
            self.last_aggressor = Some(actor as u8);
        }
        if self.stacks[actor] == 0 {
            self.all_in[actor] = true;
        }
        self.acted_this_street[actor] = true;
        self.street_level_acted[actor] = self.bet_to_call;
        self.after_action(actor);
        Ok(())
    }

    fn after_action(&mut self, actor: usize) {
        if self.alive_count() == 1 {
            self.actor = None;
            self.street = 4;
            return;
        }
        match self.find_next_actor(actor) {
            Some(next) => self.actor = Some(next as u8),
            None => {
                // Street closed. If not yet river with full board, mark
                // awaiting_runout (street stays, actor=None, board_len < 5).
                // River with board_len==5 → terminal showdown.
                if self.board_len >= 5 || self.street >= 3 {
                    self.actor = None;
                    self.street = 4;
                } else {
                    // Signal chance node: actor None but street < 4
                    self.actor = None;
                    // street stays (flop=1, turn=2); caller advances board
                }
            }
        }
    }

    /// True when betting closed mid-street and more board cards are needed.
    pub fn needs_runout(&self) -> bool {
        self.actor.is_none() && self.street < 4 && self.board_len < 5 && self.alive_count() >= 2
    }

    /// Advance board by one card (turn or river) and open a new betting round.
    pub fn deal_board_card(&mut self, card: u8) {
        if self.board_len >= 5 {
            return;
        }
        let i = self.board_len as usize;
        self.board[i] = card;
        self.board_len += 1;
        // Advance street: flop(1)+1card→turn(2), turn+1→river(3)
        if self.board_len == 4 {
            self.street = 2;
        } else if self.board_len == 5 {
            self.street = 3;
        }
        // Reset betting round
        self.street_commit = [0; MAX_SEATS];
        self.bet_to_call = 0;
        self.last_raise_size = self.bb;
        self.last_aggression_was_full_raise = true;
        self.street_level_acted = [0; MAX_SEATS];
        self.acted_this_street = [false; MAX_SEATS];
        self.last_aggressor = None;
        // First to act postflop: seat 0 (OOP) if not folded/all-in
        let n = self.n();
        let mut actor = None;
        for i in 0..n {
            if self.can_act(i) {
                actor = Some(i as u8);
                break;
            }
        }
        self.actor = actor;
        if actor.is_none() {
            // All all-in — if board incomplete keep dealing; else terminal
            if self.board_len >= 5 {
                self.street = 4;
            }
        }
    }

    /// Cards still available (not on board).
    pub fn unseen_cards(&self) -> Vec<u8> {
        let mut used = [false; 52];
        for i in 0..self.board_len as usize {
            used[self.board[i] as usize] = true;
        }
        (0..52u8).filter(|&c| !used[c as usize]).collect()
    }

    /// A seat that can still make a decision: not folded, not all-in, and
    /// holding chips.
    ///
    /// (review 2026-09-20 E1) `stacks == 0` counts as all-in even when the flag
    /// was never set. Solver roots build states by hand (blinds/antes posted
    /// with `saturating_sub`), and a chipless seat whose `street_commit` is
    /// below `bet_to_call` used to be handed the action forever: its only
    /// legal move is a 0-chip call that never matches the bet, so
    /// `find_next_actor` returned it again → unbounded recursion → native
    /// stack overflow (HU preflop with stack <= ante + bb killed the process).
    #[inline]
    pub fn can_act(&self, seat: usize) -> bool {
        !self.folded[seat] && !self.all_in[seat] && self.stacks[seat] > 0
    }

    /// Flag every chipless live seat all-in and, if the current actor cannot
    /// act, move the action to the next seat that can (or close the street).
    /// Call after hand-building a root (blinds/antes posted manually).
    pub fn seal_root(&mut self) {
        for i in 0..self.n() {
            if !self.folded[i] && self.stacks[i] == 0 {
                self.all_in[i] = true;
            }
        }
        if let Some(a) = self.actor {
            let a = a as usize;
            if !self.can_act(a) {
                // Same order as play: next seat clockwise that still owes a decision.
                let prev = (a + self.n() - 1) % self.n();
                self.actor = self.find_next_actor(prev).map(|s| s as u8);
            }
        }
    }

    fn find_next_actor(&self, after: usize) -> Option<usize> {
        let n = self.n();
        let start = (after + 1) % n;
        for i in 0..n {
            let s = (start + i) % n;
            if !self.can_act(s) {
                continue;
            }
            if !self.acted_this_street[s] || self.street_commit[s] < self.bet_to_call {
                return Some(s);
            }
        }
        None
    }

    /// Fold terminal: survivor gets pot (EV in chips for seat).
    pub fn fold_payout_chips(&self, seat: usize) -> i64 {
        let alive: Vec<usize> = (0..self.n()).filter(|&i| !self.folded[i]).collect();
        if alive.len() != 1 {
            return 0;
        }
        let winner = alive[0];
        // Net: winner gains pot - own total_commit; others lose total_commit.
        // pot = dead_pot + sum(total_commit). We treat initial pot as dead money
        // not in total_commit, so winner nets +pot - total_commit[winner] only if
        // we count dead pot as already "paid". Standard: payoff = chips_won - total_commit.
        // chips_won for fold: pot goes to survivor (including all commits + dead).
        let pot = self.pot as i64;
        let own = self.total_commit[seat] as i64;
        if seat == winner {
            pot - own
        } else {
            -own
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn river_root_oop_acts_first() {
        let s = PublicState::river_hu_root(100_000, 500_000, &[0, 1, 2, 3, 4], 10_000).unwrap();
        assert_eq!(s.actor, Some(0));
        assert_eq!(s.bet_to_call, 0);
        assert_eq!(s.min_raise_chips(), 10_000); // 1bb open
    }

    #[test]
    fn pot_fraction_half_pot_open() {
        let s = PublicState::river_hu_root(100_000, 500_000, &[0, 1, 2, 3, 4], 10_000).unwrap();
        // 500‰ of pot 100k = 50k
        assert_eq!(s.raise_chips_for_pm(500), Some(50_000));
    }

    #[test]
    fn check_check_terminals() {
        let mut s = PublicState::river_hu_root(100_000, 500_000, &[0, 1, 2, 3, 4], 10_000).unwrap();
        s.apply_check_call(); // OOP checks
        assert_eq!(s.actor, Some(1));
        s.apply_check_call(); // IP checks
        assert!(s.is_terminal());
    }

    #[test]
    fn fold_gives_pot_to_survivor() {
        let mut s = PublicState::river_hu_root(100_000, 500_000, &[0, 1, 2, 3, 4], 10_000).unwrap();
        s.apply_fold(); // OOP folds
        assert!(s.is_terminal());
        assert_eq!(s.fold_payout_chips(1), 100_000);
        assert_eq!(s.fold_payout_chips(0), 0);
    }

    /// Multiway 4-hand postflop root: unequal stacks accepted, seat0 acts first.
    #[test]
    fn multiway_four_hand_unequal_stacks() {
        let s = PublicState::postflop_root(
            4,
            20_000,
            &[100_000, 80_000, 60_000, 40_000],
            &[0, 5, 10, 15, 20],
            10_000,
            3,
        )
        .unwrap();
        assert_eq!(s.num_seats, 4);
        assert_eq!(s.actor, Some(0));
        assert_eq!(s.stacks[0], 100_000);
        assert_eq!(s.stacks[3], 40_000);
        assert_eq!(s.pot, 20_000);
    }

    /// (review 2026-09-20 E1) a chipless seat is never handed the action, even
    /// when nobody set its all-in flag — this was the unbounded recursion.
    #[test]
    fn chipless_seat_never_gets_the_action() {
        // Hand-built state exactly like the old HU preflop root at stack 1 bb:
        // both stacks saturate to 0, flags unset, SB owes half a blind.
        let mut s = PublicState::hu_postflop_root(25_000, 10_000, &[0, 1, 2], 10_000).unwrap();
        s.street = 0;
        s.board_len = 0;
        s.stacks = [0; MAX_SEATS];
        s.street_commit[0] = 10_000;
        s.street_commit[1] = 5_000;
        s.bet_to_call = 10_000;
        s.actor = Some(1);
        assert!(!s.can_act(0) && !s.can_act(1));
        // Old behaviour: apply_check_call() re-selected seat 1 forever.
        let mut steps = 0;
        while s.actor.is_some() && steps < 10 {
            s.apply_check_call();
            steps += 1;
        }
        assert!(s.actor.is_none(), "action kept cycling on a chipless seat");
        assert!(steps <= 1);
        // seal_root() does the same up front.
        let mut t = PublicState::hu_postflop_root(25_000, 10_000, &[0, 1, 2], 10_000).unwrap();
        t.stacks = [0; MAX_SEATS];
        t.seal_root();
        assert!(t.actor.is_none() && t.all_in[0] && t.all_in[1]);
    }

    #[test]
    fn preflop_root_posts_blinds_and_flags_short_all_ins() {
        // HU 100bb, ante 0.5bb: seat 0 = BB, seat 1 = BTN/SB first to act.
        let s = PublicState::preflop_root(&[1_000_000, 1_000_000], 10_000, 5_000, 5_000, 0).unwrap();
        assert_eq!((s.actor, s.button, s.street, s.board_len), (Some(1), 1, 0, 0));
        assert_eq!(s.pot, 25_000);
        assert_eq!(&s.stacks[..2], &[985_000, 990_000]);
        assert_eq!(&s.street_commit[..2], &[10_000, 5_000]);
        assert_eq!(&s.total_commit[..2], &[15_000, 10_000]);
        assert_eq!(s.to_call_chips(), 5_000);
        // 4-handed: UTG first, SB/BB on the last two seats, BTN = seat 1.
        let s = PublicState::preflop_root(&[100_000; 4], 10_000, 5_000, 0, 0).unwrap();
        assert_eq!((s.actor, s.button, s.pot), (Some(0), 1, 15_000));
        assert_eq!(&s.stacks[..4], &[100_000, 100_000, 95_000, 90_000]);
        // Short BB: posts what it has and is all-in; chips are conserved.
        let s = PublicState::preflop_root(&[100_000, 100_000, 6_000], 10_000, 5_000, 0, 0).unwrap();
        assert!(s.all_in[2] && s.stacks[2] == 0 && s.street_commit[2] == 6_000);
        assert_eq!(s.bet_to_call, 10_000, "callers still owe the nominal blind");
        assert_eq!(s.pot + s.stacks[..3].iter().sum::<u64>(), 206_000);
        assert!(PublicState::preflop_root(&[100_000, 0], 10_000, 5_000, 0, 0).is_err());
    }

    /// deal_board_card advances board_len and street on flop→turn.
    #[test]
    fn deal_board_advances_street() {
        let mut s = PublicState::postflop_root(
            2,
            50_000,
            &[200_000, 200_000],
            &[0, 5, 10],
            10_000,
            1,
        )
        .unwrap();
        assert_eq!(s.board_len, 3);
        assert_eq!(s.street, 1);
        s.deal_board_card(15);
        assert_eq!(s.board_len, 4);
        assert_eq!(s.board[3], 15);
        assert_eq!(s.street, 2); // turn
        s.deal_board_card(20);
        assert_eq!(s.board_len, 5);
        assert_eq!(s.street, 3); // river
    }
}
