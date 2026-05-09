//! Card and deck representation.
//!
//! A [`Card`] is a `u8` in `0..=51`. Rank = `card / 4` (0..=12 mapping 2..=A).
//! Suit = `card % 4` (0..=3 mapping c/d/h/s).
//!
//! [`Deck`] uses a pinned `ChaCha8Rng` seeded from a `u64` so shuffles are
//! bit-exact reproducible across machines and Rust versions.

use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

/// Number of ranks (2..=A).
pub const NUM_RANKS: u8 = 13;
/// Number of suits (c/d/h/s).
pub const NUM_SUITS: u8 = 4;
/// Total cards in a deck.
pub const DECK_SIZE: usize = 52;

/// Playing card index in `0..=51`.
///
/// Encoding: `Card(rank * 4 + suit)`. Rank 0 is deuce, rank 12 is ace.
/// Suit 0 is clubs, 1 diamonds, 2 hearts, 3 spades.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Card(pub u8);

impl Card {
    /// Construct from explicit rank (0..=12) and suit (0..=3).
    #[inline]
    pub fn new(rank: u8, suit: u8) -> Self {
        debug_assert!(rank < NUM_RANKS);
        debug_assert!(suit < NUM_SUITS);
        Card(rank * NUM_SUITS + suit)
    }

    /// Construct from a raw `0..=51` index.
    #[inline]
    pub fn from_index(i: u8) -> Self {
        debug_assert!((i as usize) < DECK_SIZE);
        Card(i)
    }

    /// Rank in `0..=12` (2..=A).
    #[inline]
    pub fn rank(self) -> u8 {
        self.0 / NUM_SUITS
    }

    /// Suit in `0..=3` (c/d/h/s).
    #[inline]
    pub fn suit(self) -> u8 {
        self.0 % NUM_SUITS
    }

    /// Raw index `0..=51`.
    #[inline]
    pub fn index(self) -> u8 {
        self.0
    }
}

impl std::fmt::Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const RANK_CHARS: [char; 13] = [
            '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
        ];
        const SUIT_CHARS: [char; 4] = ['c', 'd', 'h', 's'];
        write!(
            f,
            "{}{}",
            RANK_CHARS[self.rank() as usize],
            SUIT_CHARS[self.suit() as usize]
        )
    }
}

/// Shuffled 52-card deck with a read pointer.
///
/// Deterministic from a `u64` seed (uses `ChaCha8Rng::seed_from_u64`).
/// Callers cannot substitute an RNG — this is intentional for reproducibility.
pub struct Deck {
    cards: [Card; DECK_SIZE],
    next: usize,
}

impl Deck {
    /// Construct a freshly shuffled deck from a seed.
    pub fn new_shuffled(seed: u64) -> Self {
        let mut cards = [Card(0); DECK_SIZE];
        for (i, c) in cards.iter_mut().enumerate() {
            *c = Card(i as u8);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        cards.shuffle(&mut rng);
        Deck { cards, next: 0 }
    }

    /// Deal one card. Panics if the deck is exhausted.
    #[inline]
    pub fn deal_one(&mut self) -> Card {
        let c = self.cards[self.next];
        self.next += 1;
        c
    }

    /// Deal `n` cards. Panics if fewer than `n` remain.
    pub fn deal(&mut self, n: usize) -> Vec<Card> {
        (0..n).map(|_| self.deal_one()).collect()
    }

    /// Remaining cards in the deck.
    #[inline]
    pub fn remaining(&self) -> usize {
        DECK_SIZE - self.next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_round_trip() {
        for i in 0..DECK_SIZE as u8 {
            let c = Card::from_index(i);
            assert_eq!(c.index(), i);
            assert_eq!(Card::new(c.rank(), c.suit()), c);
        }
    }

    #[test]
    fn rank_suit_bounds() {
        for i in 0..DECK_SIZE as u8 {
            let c = Card(i);
            assert!(c.rank() < NUM_RANKS);
            assert!(c.suit() < NUM_SUITS);
        }
    }

    #[test]
    fn display_examples() {
        assert_eq!(format!("{}", Card::new(0, 0)), "2c");
        assert_eq!(format!("{}", Card::new(8, 1)), "Td");
        assert_eq!(format!("{}", Card::new(12, 2)), "Ah");
        assert_eq!(format!("{}", Card::new(11, 3)), "Ks");
    }

    #[test]
    fn shuffle_determinism_same_seed() {
        let a = Deck::new_shuffled(42).cards;
        let b = Deck::new_shuffled(42).cards;
        assert_eq!(a, b);
    }

    #[test]
    fn shuffle_different_seeds_differ() {
        let a = Deck::new_shuffled(1).cards;
        let b = Deck::new_shuffled(2).cards;
        assert_ne!(a, b);
    }

    #[test]
    fn shuffle_contains_every_card_once() {
        let d = Deck::new_shuffled(7);
        let mut seen = [false; DECK_SIZE];
        for c in &d.cards {
            let i = c.index() as usize;
            assert!(!seen[i], "duplicate card {}", c);
            seen[i] = true;
        }
        assert!(seen.iter().all(|&x| x));
    }

    #[test]
    fn deal_advances_pointer() {
        let mut d = Deck::new_shuffled(0);
        assert_eq!(d.remaining(), 52);
        let first = d.deal_one();
        assert_eq!(d.remaining(), 51);
        let five = d.deal(5);
        assert_eq!(five.len(), 5);
        assert_eq!(d.remaining(), 46);
        assert_ne!(first, five[0]);
    }
}
