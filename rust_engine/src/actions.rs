//! Discrete 8-action space for the PPO policy.

/// Number of actions in the discrete policy output.
pub const NUM_ACTIONS: usize = 8;

/// Action indices 0..=7. Index equals enum discriminant.
///
/// Bet/raise sizings are percentage-of-pot under standard poker convention
/// (see `engine.rs`): when facing a bet, the percentage applies to
/// (pot + amount_to_call) on top of the call.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Fold = 0,
    CheckCall = 1,
    BetPct10 = 2,
    BetPct25 = 3,
    BetPct50 = 4,
    BetPct75 = 5,
    /// Pot-sized bet. At shallow effective stacks (e.g. 20bb / 3bb-ante
    /// flop-open where pot=1800 and stack-behind=1700) this slot
    /// collapses to `AllIn` and is removed by duplicate-masking in
    /// `legal_action_mask`. At deeper effective stacks (post-call with
    /// a grown pot, or future configs with larger starting stacks) it
    /// remains distinct. Keeping the full `B10 / B25 / B50 / B75 / B100`
    /// enum keeps the action space forward-compatible.
    BetPct100 = 6,
    /// Canonical shove. Masked when shove < `min_bet_total` or
    /// > `max_bet_total` (PL cap) or actor has no chips behind.
    AllIn = 7,
}

impl Action {
    /// Construct from an index in `0..=7`. Returns `None` for out-of-range.
    pub fn from_index(i: u8) -> Option<Action> {
        match i {
            0 => Some(Action::Fold),
            1 => Some(Action::CheckCall),
            2 => Some(Action::BetPct10),
            3 => Some(Action::BetPct25),
            4 => Some(Action::BetPct50),
            5 => Some(Action::BetPct75),
            6 => Some(Action::BetPct100),
            7 => Some(Action::AllIn),
            _ => None,
        }
    }

    /// The discriminant index in `0..=7`.
    #[inline]
    pub fn index(self) -> u8 {
        self as u8
    }

    /// Fraction-of-pot numerator/denominator for percentage sizings.
    /// Returns `None` for non-sizing actions.
    pub fn pot_fraction(self) -> Option<(u64, u64)> {
        match self {
            Action::BetPct10 => Some((10, 100)),
            Action::BetPct25 => Some((25, 100)),
            Action::BetPct50 => Some((50, 100)),
            Action::BetPct75 => Some((75, 100)),
            Action::BetPct100 => Some((100, 100)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_round_trip() {
        for i in 0..8u8 {
            let a = Action::from_index(i).unwrap();
            assert_eq!(a.index(), i);
        }
        assert!(Action::from_index(8).is_none());
    }

    #[test]
    fn pot_fractions() {
        assert_eq!(Action::BetPct10.pot_fraction(), Some((10, 100)));
        assert_eq!(Action::BetPct25.pot_fraction(), Some((25, 100)));
        assert_eq!(Action::BetPct50.pot_fraction(), Some((50, 100)));
        assert_eq!(Action::BetPct75.pot_fraction(), Some((75, 100)));
        assert_eq!(Action::BetPct100.pot_fraction(), Some((100, 100)));
        assert_eq!(Action::Fold.pot_fraction(), None);
        assert_eq!(Action::AllIn.pot_fraction(), None);
    }
}
