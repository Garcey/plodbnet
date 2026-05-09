//! Game state types. No logic — see `engine.rs` for the state machine.

use crate::actions::Action;
use crate::cards::Card;

/// Street of the hand. `Preflop` is unused in bomb-pot format but kept for
/// generality (the encoding reserves a one-hot slot for it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
    Showdown,
}

/// How a study-mode hand ended. `None` while the hand is live.
///
/// Study mode disables auto-run-out + showdown evaluation because the user
/// may not have entered turn/river cards yet, and opponent hole cards are
/// always unknown. Terminal classification is explicit rather than implicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudyTerminal {
    /// One survivor; payouts valid (uncontested pot).
    FoldOut,
    /// Action closed with fewer than 2 voluntary actors but cards remain
    /// undealt. No showdown evaluated; payouts report zeros.
    RunOut,
    /// River action closed with ≥2 survivors. Opp cards unknown, so no
    /// evaluation; payouts report zeros.
    Showdown,
}

/// Errors raised by study-mode entry points. Surfaced to Python as `ValueError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudyError {
    DuplicateCard,
    SeatOutOfRange,
    WrongState,
    InvalidAmount,
}

impl std::fmt::Display for StudyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StudyError::DuplicateCard => write!(f, "duplicate card"),
            StudyError::SeatOutOfRange => write!(f, "seat out of range"),
            StudyError::WrongState => write!(f, "operation not valid in current state"),
            StudyError::InvalidAmount => write!(f, "chip amount out of legal range"),
        }
    }
}

impl std::error::Error for StudyError {}

impl Street {
    /// Index for one-hot encoding: Preflop=0, Flop=1, Turn=2, River=3, Showdown=4.
    pub fn index(self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 1,
            Street::Turn => 2,
            Street::River => 3,
            Street::Showdown => 4,
        }
    }
}

/// Static configuration for a hand: seats, per-seat starting stacks, ante, bb unit.
#[derive(Debug, Clone)]
pub struct GameConfig {
    pub num_seats: usize,
    pub starting_stacks: Vec<u64>,
    pub ante: u64,
    pub bb: u64,
}

impl GameConfig {
    /// Default 6-max 20bb / 3bb ante config, 1bb = 10000 chips (cent precision at $20/bb).
    pub fn default_6max_20bb() -> Self {
        GameConfig::new_uniform(6, 200000, 30000, 10000)
    }

    /// Uniform-stack constructor: every seat starts at `stack` chips.
    pub fn new_uniform(num_seats: usize, stack: u64, ante: u64, bb: u64) -> Self {
        GameConfig {
            num_seats,
            starting_stacks: vec![stack; num_seats],
            ante,
            bb,
        }
    }
}

/// One entry in the per-hand action history.
#[derive(Debug, Clone, Copy)]
pub struct ActionRecord {
    pub seat: usize,
    pub action: Action,
    /// Chips this action added to the pot (0 for fold, 0 for check).
    pub chips: u64,
    pub street: Street,
}

/// Mutable state of a single hand.
///
/// Hole cards and both full boards are dealt at construction; the engine
/// progressively *reveals* board segments on street transitions. This
/// keeps the state `Clone` and makes hands replayable from `(seed, button)`.
#[derive(Debug, Clone)]
pub struct GameState {
    pub config: GameConfig,
    pub button: usize,
    pub street: Street,
    pub pot: u64,
    pub stacks: Vec<u64>,
    pub folded: Vec<bool>,
    pub all_in: Vec<bool>,
    pub hole_cards: Vec<[Card; 5]>,
    pub board_a: Vec<Card>,
    pub board_b: Vec<Card>,
    /// Pre-dealt full board A; `board_a` above is a progressive view.
    pub full_board_a: [Card; 5],
    pub full_board_b: [Card; 5],
    /// Per-seat chips committed this street only.
    pub street_commit: Vec<u64>,
    /// Per-seat chips committed across the whole hand (includes ante).
    pub total_commit: Vec<u64>,
    /// Current maximum `street_commit` across seats.
    pub bet_to_call: u64,
    /// Size of the most recent raise-delta (for min-raise enforcement).
    pub last_raise_size: u64,
    /// True iff the most recent aggression this street was a full (min-raise-
    /// sized or larger) raise. False after a short-all-in that was below the
    /// PL min-raise floor. When false, seats that already acted this street
    /// cannot re-raise (Fold/CheckCall only). Reset to true on street
    /// transitions. See feedback_short_allin_rule memory.
    pub last_aggression_was_full_raise: bool,
    /// Seat to act next. `None` when the hand is terminal.
    pub actor: Option<usize>,
    /// Seat of the last aggressor this street (for round-close detection).
    pub last_aggressor: Option<usize>,
    /// Per-seat: has this seat acted at least once on the current street.
    /// Reset on street transitions. Used for round-close detection.
    pub acted_this_street: Vec<bool>,
    pub history: Vec<ActionRecord>,

    /// When true, the engine halts at street boundaries so the UI can
    /// supply the next street's cards. Also suppresses auto-run-out and
    /// showdown evaluation (opp cards are unknown).
    pub study_mode: bool,
    /// `Some(Turn)` or `Some(River)` when the current round has closed with
    /// ≥2 voluntary actors and the UI must call `set_turn` / `set_river`.
    pub awaiting_next_street: Option<Street>,
    /// Set when a study-mode hand ends. `None` while live.
    pub study_terminal: Option<StudyTerminal>,
    /// Hero seat in study mode. `None` outside study mode. Persisted so
    /// `set_turn`/`set_river` can validate new board cards against the
    /// hero's visible hole (opp holes are placeholder draws not visible
    /// to the UI and therefore not user-controlled).
    pub study_hero_seat: Option<usize>,

    /// Progressive board length (`board_a.len()`) captured at the top of
    /// every `close_round_or_run_out` call. `Some(3)` = action closed on
    /// the flop; `Some(4)` = turn; `Some(5)` = river. Used by
    /// `payouts_ev` to determine how many community cards still need
    /// Monte-Carlo sampling when action went to showdown with 2+ live
    /// seats but no more voluntary actors (run-out case).
    pub action_close_board_len: Option<u8>,

    /// Per-seat effective-stack cap captured at hand start, frozen for
    /// the duration of the hand. For seat `i`:
    /// `min(starting_stack[i], max(starting_stack[j] for j != i and j ∈ in_hand))`.
    /// Used by the observation encoder so the visible "stack" feature
    /// doesn't shrink retroactively when an opponent folds — a deep
    /// stack's prospective leverage on later streets must be preserved.
    /// Sitting-out seats (`!in_hand_mask`) are excluded from the
    /// `max_other` reduction.
    pub eff_stack_cap_at_hand_start: Vec<u64>,
}
