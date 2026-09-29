//! Card and deck representation.
//!
//! A [`Card`] is a `u8` in `0..=51`. Rank = `card / 4` (0..=12 mapping 2..=A).
//! Suit = `card % 4` (0..=3 mapping c/d/h/s).
//!
//! [`Deck`] uses a pinned `ChaCha8Rng` seeded from a `u64` so shuffles are
//! bit-exact reproducible across machines and Rust versions. The `rand` /
//! `rand_chacha` versions are pinned exactly in Cargo.toml, and
//! `seeded_streams_are_pinned` below holds known answers for every stream
//! primitive the engine uses (the seeded deal order is a public contract).

use rand::seq::SliceRandom;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

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

/// An undealt slot of a fixed-size board (a single-board game's board B, a
/// study hand's turn and river before they are entered, a runout buffer
/// before it is filled). Out of range for every card-indexed table, so code
/// that wrongly reads one fails loudly instead of scoring it as a real card --
/// `Card(0)`, the 2 of clubs, used to fill these slots (ENG-023).
pub const NO_CARD: Card = Card(255);

/// A set of cards, bit `i` = the card with index `i` (ENG-034): the "which
/// cards are out" bookkeeping of the deal checks, the study validation, the
/// hand features and the Monte Carlo, in one 64-bit word instead of a
/// hand-rolled `[bool; 52]` at every site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CardMask(u64);

impl CardMask {
    pub const EMPTY: CardMask = CardMask(0);

    /// The set of `cards` (a repeated card is simply in it).
    pub fn of<'a>(cards: impl IntoIterator<Item = &'a Card>) -> CardMask {
        let mut m = CardMask::EMPTY;
        for &c in cards {
            m.add(c);
        }
        m
    }

    /// Put `c` in the set (already in: no change). `c` must be a real card.
    #[inline]
    pub fn add(&mut self, c: Card) {
        debug_assert!((c.0 as usize) < DECK_SIZE, "not a card: {}", c.0);
        self.0 |= 1u64 << (c.0 & 63);
    }

    /// Put `c` in the set -- `false` (set unchanged) when `c` is not a real
    /// card or is already in: the deal checks' duplicate test.
    #[inline]
    #[must_use]
    pub fn insert(&mut self, c: Card) -> bool {
        if (c.0 as usize) >= DECK_SIZE || self.contains(c) {
            return false;
        }
        self.0 |= 1u64 << c.0;
        true
    }

    /// Whether `c` is in the set (never true for a non-card).
    #[inline]
    pub fn contains(self, c: Card) -> bool {
        (c.0 as usize) < DECK_SIZE && (self.0 >> c.0) & 1 == 1
    }

    /// How many cards of rank `r` (0 = deuce ..= 12 = ace) are in the set.
    #[inline]
    pub fn count_rank(self, r: u8) -> u32 {
        ((self.0 >> (4 * r as u32)) & 0xF).count_ones()
    }

    /// The cards NOT in the set, ascending by index -- the order every unseen
    /// deck is laid out in (the Monte Carlo and the EV runouts index it by
    /// position, so this order is part of their results).
    pub fn unseen(self) -> impl Iterator<Item = Card> {
        let mut rest = !self.0 & ((1u64 << DECK_SIZE) - 1);
        std::iter::from_fn(move || {
            (rest != 0).then(|| {
                let i = rest.trailing_zeros() as u8;
                rest &= rest - 1;
                Card(i)
            })
        })
    }
}

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

    /// A deck in a CALLER-SUPPLIED order (index 0 is dealt first): the home
    /// games' verifiable shuffle seals a deck, lets the players' devices
    /// re-permute it, and deals exactly that order. Must be a permutation of
    /// all 52 cards — anything else is an error, never a panic.
    pub fn from_order(order: &[u8]) -> Result<Self, String> {
        if order.len() != DECK_SIZE {
            return Err(format!(
                "deck must list {DECK_SIZE} cards, got {}",
                order.len()
            ));
        }
        let mut seen = [false; DECK_SIZE];
        let mut cards = [Card(0); DECK_SIZE];
        for (slot, &i) in order.iter().enumerate() {
            if (i as usize) >= DECK_SIZE {
                return Err(format!("card index {i} out of range at position {slot}"));
            }
            if seen[i as usize] {
                return Err(format!("card index {i} appears twice in the deck"));
            }
            seen[i as usize] = true;
            cards[slot] = Card(i);
        }
        Ok(Deck { cards, next: 0 })
    }

    /// The full order, first-dealt card first (tests / parity checks).
    pub fn order(&self) -> [u8; DECK_SIZE] {
        let mut out = [0u8; DECK_SIZE];
        for (o, c) in out.iter_mut().zip(self.cards.iter()) {
            *o = c.index();
        }
        out
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
    fn card_mask_is_a_set_of_real_cards() {
        let mut m = CardMask::EMPTY;
        assert!(m.insert(Card(0)) && m.insert(Card(51)) && m.insert(Card(20)));
        assert!(!m.insert(Card(20)), "a duplicate");
        assert!(!m.insert(NO_CARD) && !m.insert(Card(52)), "not cards");
        assert!(m.contains(Card(0)) && m.contains(Card(51)) && !m.contains(Card(1)));
        assert!(!m.contains(NO_CARD));
        assert_eq!(CardMask::of(&[Card(20), Card(0), Card(51), Card(20)]), m);
        // The unseen deck ascends by index: exactly the old filter over 0..52.
        let unseen: Vec<Card> = m.unseen().collect();
        let want: Vec<Card> = (0..52u8).map(Card).filter(|c| !m.contains(*c)).collect();
        assert_eq!(unseen, want);
        assert_eq!(CardMask::EMPTY.unseen().count(), 52);
        // Rank counts: the 2c (rank 0) and the As (51 = rank 12) and 20 (rank 5).
        assert_eq!(
            (
                m.count_rank(0),
                m.count_rank(5),
                m.count_rank(12),
                m.count_rank(1)
            ),
            (1, 1, 1, 0)
        );
        let full = CardMask::of(&(0..52u8).map(Card).collect::<Vec<_>>());
        assert_eq!(full.unseen().count(), 0);
        assert!((0..13).all(|r| full.count_rank(r) == 4));
    }

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

    /// ENG-006: the seeded deal order is a public contract (every seeded hand,
    /// the study placeholders, the EV runouts and the opp-outcome MC all read
    /// these streams), and `rand` / `rand_chacha` are pinned with `=` in
    /// Cargo.toml for it. These known answers fail loudly if an upgrade — or a
    /// change of `rand_core`'s `seed_from_u64` expansion — moves any of the
    /// primitives the engine uses; "same seed = same deck" alone cannot, since
    /// both sides would move together.
    #[test]
    fn seeded_streams_are_pinned() {
        use rand::{Rng, RngCore};
        // Deck::new_shuffled = `SliceRandom::shuffle` over ChaCha8 (every
        // seeded deal; the study placeholder deal shuffles the same way).
        assert_eq!(
            Deck::new_shuffled(42).order(),
            [
                16, 30, 21, 0, 51, 38, 19, 2, 22, 36, 27, 8, 17, 10, 28, 50, 4, 40, 23, 1, 43, 35,
                6, 39, 15, 25, 42, 12, 29, 5, 3, 18, 24, 41, 47, 32, 44, 48, 9, 33, 31, 26, 49, 45,
                14, 13, 20, 37, 46, 7, 34, 11
            ]
        );
        assert_eq!(
            Deck::new_shuffled(0).order()[..12],
            [46, 8, 30, 25, 16, 13, 7, 3, 15, 22, 12, 0]
        );
        assert_eq!(
            Deck::new_shuffled(u64::MAX).order()[..12],
            [24, 17, 40, 5, 29, 19, 39, 11, 25, 23, 16, 48]
        );
        // The raw primitives: `next_u32` (opp-outcome MC draws), `gen_range`
        // (EV runouts, PLO67 runout equities) and `shuffle`, in one stream.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let u: Vec<u32> = (0..4).map(|_| rng.next_u32()).collect();
        assert_eq!(u, [601_310_139, 677_729_076, 781_920_570, 721_508_819]);
        let g: Vec<usize> = (0..6).map(|i| rng.gen_range(i..45)).collect();
        assert_eq!(g, [31, 27, 17, 6, 44, 13]);
        let g2: Vec<usize> = (0..4).map(|i| rng.gen_range(0..=i * 7)).collect();
        assert_eq!(g2, [0, 4, 3, 9]);
        let mut v: Vec<u8> = (0..20).collect();
        v.shuffle(&mut rng);
        assert_eq!(
            v,
            [11, 19, 6, 12, 2, 16, 7, 8, 14, 4, 9, 18, 13, 3, 0, 17, 1, 5, 15, 10]
        );
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
    fn from_order_round_trips_and_rejects_bad_decks() {
        let shuffled = Deck::new_shuffled(11);
        let order = shuffled.order();
        let mut d = Deck::from_order(&order).expect("a real permutation");
        assert_eq!(d.order(), order);
        assert_eq!(d.deal_one().index(), order[0]);
        assert_eq!(d.deal_one().index(), order[1]);
        assert!(Deck::from_order(&order[..51]).is_err(), "51 cards");
        let mut dup = order;
        dup[7] = dup[3];
        assert!(Deck::from_order(&dup).is_err(), "a duplicate card");
        let mut oob = order;
        oob[0] = 52;
        assert!(Deck::from_order(&oob).is_err(), "index out of range");
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
