//! Helpers the engine's test modules share (TEST-025: each used to carry its own
//! copy). Test-only: `#[cfg(test)] mod test_util` in lib.rs.

use crate::cards::Card;
use crate::state::{GameConfig, Variant};

/// A card by rank (0 = deuce ..= 12 = ace) and suit (0 clubs, 1 diamonds,
/// 2 hearts, 3 spades).
pub(crate) fn c(rank: u8, suit: u8) -> Card {
    Card::new(rank, suit)
}

/// Cards by index.
pub(crate) fn cards(idxs: &[u8]) -> Vec<Card> {
    idxs.iter().map(|&i| Card::from_index(i)).collect()
}

/// A five-card hole by index.
pub(crate) fn hole5(idxs: [u8; 5]) -> [Card; 5] {
    idxs.map(Card::from_index)
}

/// A flop by index.
pub(crate) fn flop3(idxs: [u8; 3]) -> [Card; 3] {
    idxs.map(Card::from_index)
}

/// A PLO table (`sb` 0: bomb pots post antes only).
pub(crate) fn plo(variant: Variant, stacks: &[u64], ante: u64, bb: u64) -> GameConfig {
    GameConfig {
        num_seats: stacks.len(),
        starting_stacks: stacks.to_vec(),
        ante,
        bb,
        sb: 0,
        variant,
    }
}

/// An NLH table.
pub(crate) fn nlh(stacks: &[u64], ante: u64, sb: u64, bb: u64) -> GameConfig {
    GameConfig {
        num_seats: stacks.len(),
        starting_stacks: stacks.to_vec(),
        ante,
        bb,
        sb,
        variant: Variant::NlhSingle,
    }
}

/// splitmix64 -- the tests' own pinned stream (a policy or a case generator
/// must not move when the `rand` crate does; the deals are pinned separately,
/// see cards.rs).
pub(crate) struct Mix(pub(crate) u64);

impl Mix {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-ish in `0..n` (n > 0); modulo bias is irrelevant here.
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// A float in [-4, 4] on a 1/250 grid.
    pub(crate) fn f32(&mut self) -> f32 {
        (self.below(2001) as f32 - 1000.0) / 250.0
    }
}

/// xorshift64: `next(m)` draws in `0..m` -- the case generators' stream.
pub(crate) fn xorshift(seed: u64) -> impl FnMut(u64) -> u64 {
    let mut x = seed;
    move |m: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % m
    }
}
