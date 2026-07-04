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

/// Game variant. Selects hole-card count, board count, betting cap, and
/// street structure. Every playable format is an explicit enum arm with
/// derived properties — never free-floating config flags — so the set of
/// supported rule combinations is closed and each is tested by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// PLO5 double-board bomb pot: 5 hole cards, two boards, pot-limit
    /// cap, ante-only (no blinds, no preflop betting round — hands start
    /// at the flop). The original format; all pre-variant behavior.
    Plo5DoubleBomb,
    /// PLO6 double-board bomb pot: identical to `Plo5DoubleBomb` in every
    /// rule (two boards, pot-limit cap, ante-only, hands start at the
    /// flop, exactly-2-hole + 3-board eval) except each seat is dealt 6
    /// hole cards. Deck feasibility: 6 seats × 6 + 10 board = 46 ≤ 52.
    Plo6DoubleBomb,
    /// PLO4 double-board bomb pot: identical to `Plo5DoubleBomb` in every
    /// rule except each seat is dealt 4 hole cards (classic Omaha hole
    /// width). Deck feasibility: 6 seats × 4 + 10 board = 34 ≤ 52.
    Plo4DoubleBomb,
    /// No-limit hold'em, single board: 2 hole cards, best-5-of-7
    /// any-combo eval, no-limit cap, SB/BB blinds + per-player ante,
    /// betting starts preflop.
    NlhSingle,
}

impl Variant {
    pub fn hole_count(self) -> usize {
        match self {
            Variant::Plo4DoubleBomb => 4,
            Variant::Plo5DoubleBomb => 5,
            Variant::Plo6DoubleBomb => 6,
            Variant::NlhSingle => 2,
        }
    }

    pub fn num_boards(self) -> usize {
        match self {
            Variant::Plo4DoubleBomb | Variant::Plo5DoubleBomb | Variant::Plo6DoubleBomb => 2,
            Variant::NlhSingle => 1,
        }
    }

    pub fn pot_limit(self) -> bool {
        matches!(
            self,
            Variant::Plo4DoubleBomb | Variant::Plo5DoubleBomb | Variant::Plo6DoubleBomb
        )
    }

    /// True when hands begin with a preflop betting round (blinds posted
    /// live, no board revealed until the round closes).
    pub fn has_preflop(self) -> bool {
        matches!(self, Variant::NlhSingle)
    }
}

/// Static configuration for a hand: seats, per-seat starting stacks, ante, bb unit.
#[derive(Debug, Clone)]
pub struct GameConfig {
    pub num_seats: usize,
    pub starting_stacks: Vec<u64>,
    pub ante: u64,
    pub bb: u64,
    /// Small blind in chips. Only meaningful for variants with blinds
    /// (`variant.has_preflop()`); 0 for bomb pots.
    pub sb: u64,
    pub variant: Variant,
}

impl GameConfig {
    /// Default 6-max 20bb / 3bb ante config, 1bb = 10000 chips (cent precision at $20/bb).
    pub fn default_6max_20bb() -> Self {
        GameConfig::new_uniform(6, 200000, 30000, 10000)
    }

    /// Uniform-stack constructor: every seat starts at `stack` chips.
    /// Bomb-pot variant (no blinds) — the pre-variant behavior.
    pub fn new_uniform(num_seats: usize, stack: u64, ante: u64, bb: u64) -> Self {
        GameConfig {
            num_seats,
            starting_stacks: vec![stack; num_seats],
            ante,
            bb,
            sb: 0,
            variant: Variant::Plo5DoubleBomb,
        }
    }

    /// Uniform-stack NLH constructor. `ante` is per player (every
    /// dealt-in seat posts it, dead, before the blinds).
    pub fn new_nlh_uniform(
        num_seats: usize,
        stack: u64,
        sb: u64,
        bb: u64,
        ante: u64,
    ) -> Self {
        GameConfig {
            num_seats,
            starting_stacks: vec![stack; num_seats],
            ante,
            bb,
            sb,
            variant: Variant::NlhSingle,
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
    /// Blind seats for variants with a preflop round; `None` for bomb
    /// pots. Stored (not re-derived) because the clockwise walk skips
    /// sitting-out seats — downstream consumers must not duplicate it.
    pub sb_seat: Option<usize>,
    pub bb_seat: Option<usize>,
    pub street: Street,
    pub pot: u64,
    pub stacks: Vec<u64>,
    pub folded: Vec<bool>,
    pub all_in: Vec<bool>,
    /// Per-seat hole cards, `config.variant.hole_count()` each. Sized by
    /// what was actually dealt so no reader can see phantom cards.
    pub hole_cards: Vec<Vec<Card>>,
    pub board_a: Vec<Card>,
    pub board_b: Vec<Card>,
    /// Pre-dealt full board A; `board_a` above is a progressive view.
    pub full_board_a: [Card; 5],
    /// Pre-dealt full board B. Single-board variants leave this as
    /// `Card(0)` sentinels and never reveal or read it.
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
    /// PL min-raise floor. Informational/diagnostic only — raise-reopen
    /// legality is per-seat via `street_level_acted`. Reset to true on
    /// street transitions.
    pub last_aggression_was_full_raise: bool,
    /// Street bet level (`bet_to_call`) as of each seat's most recent
    /// action this street; 0 if the seat hasn't acted yet. Drives the
    /// per-seat short-all-in reopen rule: a seat that has acted may
    /// re-raise only once `bet_to_call` has grown by at least one full
    /// raise (`last_raise_size`) since that action. A seat that only
    /// checked (level 0) is therefore reopened by any subsequent full
    /// bet even if a later short all-in froze the seats that had
    /// already responded to that bet. Reset on street transitions.
    pub street_level_acted: Vec<u64>,
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
