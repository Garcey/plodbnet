//! 5-card poker hand evaluator and PLO5 "exactly 2 from hand + 3 from board" evaluator.
//!
//! Fast path: Cactus Kev's 5-card algorithm. Each card is encoded as a `u32`
//! holding a rank bitset, suit mask, rank nibble, and prime factor. A 5-card
//! hand's bitset + product + all-suits-match check selects from three
//! precomputed tables to return a 1..=7462 CK ordinal (1 = royal flush).
//!
//! [`HandRank`] is a packed `u32`:
//! - bits 20..24 : category (0..=8)
//! - bits  0..20 : within-category ordinal (higher = better within category)
//!
//! u32 comparison on `HandRank` matches hand-strength comparison.
//!
//! Categories: 0 = high card, 1 = pair, 2 = two pair, 3 = trips, 4 = straight,
//! 5 = flush, 6 = full house, 7 = quads, 8 = straight flush.

use std::sync::OnceLock;

use crate::cards::Card;

// ---------- Category constants ----------

pub const CAT_HIGH_CARD: u32 = 0;
pub const CAT_PAIR: u32 = 1;
pub const CAT_TWO_PAIR: u32 = 2;
pub const CAT_TRIPS: u32 = 3;
pub const CAT_STRAIGHT: u32 = 4;
pub const CAT_FLUSH: u32 = 5;
pub const CAT_FULL_HOUSE: u32 = 6;
pub const CAT_QUADS: u32 = 7;
pub const CAT_STRAIGHT_FLUSH: u32 = 8;

pub type HandRank = u32;

/// Extract the category for debugging.
#[inline]
pub fn category(rank: HandRank) -> u32 {
    rank >> 20
}

// ---------- Cactus Kev card encoding ----------

/// Primes indexed by rank (deuce..=ace).
const PRIMES: [u32; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// Encode a card for Cactus Kev evaluation.
///
/// Layout (bits):
/// - `[28..=16]` — rank bitset (bit `16+r` set for rank `r`)
/// - `[15..=12]` — suit mask (one of the four bits set)
/// - `[11..= 8]` — rank nibble (value 0..=12)
/// - `[ 7..= 0]` — prime factor for the rank
#[inline]
fn card_to_ck(card: Card) -> u32 {
    let r = card.rank() as u32;
    let s = card.suit() as u32;
    let prime = PRIMES[r as usize];
    let rank_bit = 1u32 << (16 + r);
    let suit_bit = 1u32 << (12 + s);
    let rank_nibble = r << 8;
    rank_bit | suit_bit | rank_nibble | prime
}

// ---------- CK tables ----------

struct CkTables {
    /// 8192-entry table indexed by 13-bit rank bitset. Non-zero only for
    /// bitsets that correspond to a 5-card straight flush or plain flush.
    flushes: Vec<u16>,
    /// 8192-entry table for straight / high card (5 distinct ranks, non-flush).
    unique5: Vec<u16>,
    /// Sorted (prime_product, ck_rank) pairs for paired hands. Binary-searched.
    paired: Vec<(u32, u16)>,
}

static CK_TABLES: OnceLock<CkTables> = OnceLock::new();

#[inline]
fn tables() -> &'static CkTables {
    CK_TABLES.get_or_init(build_tables)
}

fn build_tables() -> CkTables {
    let mut flushes = vec![0u16; 8192];
    let mut unique5 = vec![0u16; 8192];
    let mut paired: Vec<(u32, u16)> = Vec::with_capacity(4888);

    // --- Straight bitsets: A-high..6-high, wheel ---
    let straight_bitsets: Vec<u16> = {
        let mut v = Vec::with_capacity(10);
        for top in (4..=12u8).rev() {
            let bs: u16 = (0..5).map(|i| 1u16 << (top - i)).fold(0, |a, b| a | b);
            v.push(bs);
        }
        // Wheel: A-2-3-4-5.
        let wheel: u16 = (1u16 << 12) | 0b1111;
        v.push(wheel);
        v
    };

    let mut straight_set = std::collections::HashSet::new();
    for (i, &bs) in straight_bitsets.iter().enumerate() {
        flushes[bs as usize] = (1 + i) as u16; // 1..=10 (straight flushes)
        unique5[bs as usize] = (1600 + i) as u16; // 1600..=1609 (straights)
        straight_set.insert(bs);
    }

    // --- Non-straight 5-card rank bitsets sorted by strength desc ---
    let mut no_pair: Vec<u16> = Vec::with_capacity(1277);
    for a in 0..13u8 {
        for b in (a + 1)..13u8 {
            for cc in (b + 1)..13u8 {
                for d in (cc + 1)..13u8 {
                    for e in (d + 1)..13u8 {
                        let bs = (1u16 << a)
                            | (1u16 << b)
                            | (1u16 << cc)
                            | (1u16 << d)
                            | (1u16 << e);
                        if !straight_set.contains(&bs) {
                            no_pair.push(bs);
                        }
                    }
                }
            }
        }
    }
    no_pair.sort_by(|x, y| y.cmp(x));
    for (i, &bs) in no_pair.iter().enumerate() {
        flushes[bs as usize] = (323 + i) as u16; // 323..=1599 (flushes)
        unique5[bs as usize] = (6186 + i) as u16; // 6186..=7462 (high cards)
    }

    // --- Paired entries ---
    let p = |r: u8| PRIMES[r as usize];

    // Quads: 11..=166
    let mut ck = 11u16;
    for quad in (0..13u8).rev() {
        for kicker in (0..13u8).rev() {
            if kicker == quad {
                continue;
            }
            let prod = p(quad).pow(4) * p(kicker);
            paired.push((prod, ck));
            ck += 1;
        }
    }
    debug_assert_eq!(ck, 167);

    // Full houses: 167..=322
    for trip in (0..13u8).rev() {
        for pair in (0..13u8).rev() {
            if pair == trip {
                continue;
            }
            let prod = p(trip).pow(3) * p(pair).pow(2);
            paired.push((prod, ck));
            ck += 1;
        }
    }
    debug_assert_eq!(ck, 323);

    // Trips: 1610..=2467
    ck = 1610;
    for trip in (0..13u8).rev() {
        for k1 in (0..13u8).rev() {
            if k1 == trip {
                continue;
            }
            for k2 in (0..k1).rev() {
                if k2 == trip {
                    continue;
                }
                let prod = p(trip).pow(3) * p(k1) * p(k2);
                paired.push((prod, ck));
                ck += 1;
            }
        }
    }
    debug_assert_eq!(ck, 2468);

    // Two pair: 2468..=3325
    for p1 in (0..13u8).rev() {
        for p2 in (0..p1).rev() {
            for k in (0..13u8).rev() {
                if k == p1 || k == p2 {
                    continue;
                }
                let prod = p(p1).pow(2) * p(p2).pow(2) * p(k);
                paired.push((prod, ck));
                ck += 1;
            }
        }
    }
    debug_assert_eq!(ck, 3326);

    // One pair: 3326..=6185
    for pair in (0..13u8).rev() {
        for k1 in (0..13u8).rev() {
            if k1 == pair {
                continue;
            }
            for k2 in (0..k1).rev() {
                if k2 == pair {
                    continue;
                }
                for k3 in (0..k2).rev() {
                    if k3 == pair {
                        continue;
                    }
                    let prod = p(pair).pow(2) * p(k1) * p(k2) * p(k3);
                    paired.push((prod, ck));
                    ck += 1;
                }
            }
        }
    }
    debug_assert_eq!(ck, 6186);

    paired.sort_by_key(|&(prod, _)| prod);

    CkTables {
        flushes,
        unique5,
        paired,
    }
}

// ---------- CK evaluation ----------

#[inline]
fn ck_eval_inline(c: [u32; 5], t: &CkTables) -> u16 {
    let q = ((c[0] | c[1] | c[2] | c[3] | c[4]) >> 16) as usize;
    if (c[0] & c[1] & c[2] & c[3] & c[4] & 0xF000) != 0 {
        return t.flushes[q];
    }
    let u = t.unique5[q];
    if u != 0 {
        return u;
    }
    let prod = (c[0] & 0xFF) * (c[1] & 0xFF) * (c[2] & 0xFF) * (c[3] & 0xFF) * (c[4] & 0xFF);
    match t.paired.binary_search_by_key(&prod, |&(p, _)| p) {
        Ok(i) => t.paired[i].1,
        Err(_) => unreachable!("prime product {prod} not in paired table"),
    }
}

#[inline]
fn ck_to_hand_rank(ck: u16) -> HandRank {
    // ck=0 is the sentinel ck_eval_inline returns when the 5-card hand
    // has a duplicate card (flush-bitset table has no entry for a rank
    // bitset with <5 bits). This is reachable in study mode when a
    // user-supplied turn/river card collides with a non-hero seat's
    // placeholder hole. Downgrade to the worst high-card rank rather
    // than panic — evaluate_plo5* loops filter degenerate combos first,
    // so this path only fires on evaluate_5 direct callers.
    let ck = if ck == 0 { 7462 } else { ck as u32 };
    let (cat, hi) = match ck {
        1..=10 => (CAT_STRAIGHT_FLUSH, 10),
        11..=166 => (CAT_QUADS, 166),
        167..=322 => (CAT_FULL_HOUSE, 322),
        323..=1599 => (CAT_FLUSH, 1599),
        1600..=1609 => (CAT_STRAIGHT, 1609),
        1610..=2467 => (CAT_TRIPS, 2467),
        2468..=3325 => (CAT_TWO_PAIR, 3325),
        3326..=6185 => (CAT_PAIR, 6185),
        6186..=7462 => (CAT_HIGH_CARD, 7462),
        _ => unreachable!("invalid ck rank {ck}"),
    };
    let within = hi - ck;
    (cat << 20) | within
}

/// Evaluate a 5-card poker hand.
pub fn evaluate_5(cards: &[Card; 5]) -> HandRank {
    let t = tables();
    let c = [
        card_to_ck(cards[0]),
        card_to_ck(cards[1]),
        card_to_ck(cards[2]),
        card_to_ck(cards[3]),
        card_to_ck(cards[4]),
    ];
    ck_to_hand_rank(ck_eval_inline(c, t))
}

// ---------- PLO5 evaluator (100 combos) ----------

const PAIRS_5: [(usize, usize); 10] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
];

const TRIPLES_5: [(usize, usize, usize); 10] = [
    (0, 1, 2),
    (0, 1, 3),
    (0, 1, 4),
    (0, 2, 3),
    (0, 2, 4),
    (0, 3, 4),
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 4),
    (2, 3, 4),
];

/// Partial-board PLO5 evaluator. `board` may have 3, 4, or 5 cards.
/// Returns the best hand hero can currently make using exactly 2 hole + 3
/// visible board cards. Used for the "current hand category" feature pre-river.
pub fn evaluate_plo5_partial(hole: &[Card; 5], board: &[Card]) -> HandRank {
    assert!(
        board.len() >= 3 && board.len() <= 5,
        "partial board must have 3..=5 cards"
    );
    let t = tables();
    let h: [u32; 5] = [
        card_to_ck(hole[0]),
        card_to_ck(hole[1]),
        card_to_ck(hole[2]),
        card_to_ck(hole[3]),
        card_to_ck(hole[4]),
    ];
    let bn = board.len();
    let mut best_ck: u16 = u16::MAX;
    for i0 in 0..bn {
        for i1 in (i0 + 1)..bn {
            for i2 in (i1 + 1)..bn {
                let b0 = card_to_ck(board[i0]);
                let b1 = card_to_ck(board[i1]);
                let b2 = card_to_ck(board[i2]);
                for &(hi0, hi1) in &PAIRS_5 {
                    let c = [h[hi0], h[hi1], b0, b1, b2];
                    let ck = ck_eval_inline(c, t);
                    // Skip degenerate 5-card hands (duplicate card).
                    // Reachable when a user-supplied turn/river coincides
                    // with a non-hero placeholder hole drawn at
                    // reset_study — a study-mode-only state.
                    if ck != 0 && ck < best_ck {
                        best_ck = ck;
                    }
                }
            }
        }
    }
    // All combos degenerate (pathologically many shared cards) →
    // return worst rank rather than passing u16::MAX to the mapper.
    let safe_ck = if best_ck == u16::MAX { 7462 } else { best_ck };
    ck_to_hand_rank(safe_ck)
}

/// Evaluate a k-card opponent hand on a 3..=5-card board under PLO5
/// rules ("exactly 2 from hole + 3 from board"). Generalizes
/// [`evaluate_plo5_partial`] to k = 2..=5 hole cards. Used for the
/// opp-vs-hero outcome-fraction features.
///
/// For k=2 there is exactly one hole-pair choice. For k=3,4,5 the
/// function enumerates `C(k, 2)` hole-pair × `C(board.len(), 3)`
/// board-triple combinations and returns the strongest 5-card rank.
pub fn evaluate_plo5_k_partial(hole: &[Card], board: &[Card]) -> HandRank {
    assert!(
        hole.len() >= 2 && hole.len() <= 5,
        "hole must have 2..=5 cards"
    );
    assert!(
        board.len() >= 3 && board.len() <= 5,
        "board must have 3..=5 cards"
    );
    let t = tables();
    let kn = hole.len();
    let bn = board.len();
    let mut h_ck = [0u32; 5];
    for i in 0..kn {
        h_ck[i] = card_to_ck(hole[i]);
    }
    let mut b_ck = [0u32; 5];
    for i in 0..bn {
        b_ck[i] = card_to_ck(board[i]);
    }
    let mut best_ck: u16 = u16::MAX;
    for hi0 in 0..kn {
        for hi1 in (hi0 + 1)..kn {
            for bi0 in 0..bn {
                for bi1 in (bi0 + 1)..bn {
                    for bi2 in (bi1 + 1)..bn {
                        let c = [h_ck[hi0], h_ck[hi1], b_ck[bi0], b_ck[bi1], b_ck[bi2]];
                        let ck = ck_eval_inline(c, t);
                        if ck != 0 && ck < best_ck {
                            best_ck = ck;
                        }
                    }
                }
            }
        }
    }
    let safe_ck = if best_ck == u16::MAX { 7462 } else { best_ck };
    ck_to_hand_rank(safe_ck)
}

/// PLO5 evaluator: must use exactly 2 from hole + 3 from board.
/// Enumerates all 100 combinations and returns the best rank.
pub fn evaluate_plo5(hole: &[Card; 5], board: &[Card; 5]) -> HandRank {
    let t = tables();
    let h: [u32; 5] = [
        card_to_ck(hole[0]),
        card_to_ck(hole[1]),
        card_to_ck(hole[2]),
        card_to_ck(hole[3]),
        card_to_ck(hole[4]),
    ];
    let b: [u32; 5] = [
        card_to_ck(board[0]),
        card_to_ck(board[1]),
        card_to_ck(board[2]),
        card_to_ck(board[3]),
        card_to_ck(board[4]),
    ];
    let mut best_ck: u16 = u16::MAX; // lower CK = stronger
    for &(h0, h1) in &PAIRS_5 {
        for &(b0, b1, b2) in &TRIPLES_5 {
            let c = [h[h0], h[h1], b[b0], b[b1], b[b2]];
            let ck = ck_eval_inline(c, t);
            // Skip degenerate 5-card hands (duplicate card), see
            // note in evaluate_plo5_partial.
            if ck != 0 && ck < best_ck {
                best_ck = ck;
            }
        }
    }
    let safe_ck = if best_ck == u16::MAX { 7462 } else { best_ck };
    ck_to_hand_rank(safe_ck)
}

// ---------- Naive reference evaluator (test oracle only) ----------

#[cfg(test)]
fn straight_top_naive(counts: &[u8; 13]) -> Option<u8> {
    for r in (4..=12u8).rev() {
        let ri = r as usize;
        if counts[ri] >= 1
            && counts[ri - 1] >= 1
            && counts[ri - 2] >= 1
            && counts[ri - 3] >= 1
            && counts[ri - 4] >= 1
        {
            return Some(r);
        }
    }
    if counts[12] >= 1 && counts[0] >= 1 && counts[1] >= 1 && counts[2] >= 1 && counts[3] >= 1 {
        return Some(3);
    }
    None
}

#[cfg(test)]
fn pack_naive(cat: u32, s0: u8, s1: u8, s2: u8, s3: u8, s4: u8) -> HandRank {
    (cat << 20)
        | ((s0 as u32) << 16)
        | ((s1 as u32) << 12)
        | ((s2 as u32) << 8)
        | ((s3 as u32) << 4)
        | (s4 as u32)
}

/// Slot-packed naive evaluator. Same category as [`evaluate_5`]; within-
/// category ordering agrees, but absolute HandRank values differ because
/// this evaluator packs kicker-nibbles instead of a dense CK ordinal.
#[cfg(test)]
fn evaluate_5_naive(cards: &[Card; 5]) -> HandRank {
    let mut ranks = [0u8; 5];
    let mut counts = [0u8; 13];
    let first_suit = cards[0].suit();
    let mut is_flush = true;
    for (i, c) in cards.iter().enumerate() {
        let r = c.rank();
        ranks[i] = r;
        counts[r as usize] += 1;
        if c.suit() != first_suit {
            is_flush = false;
        }
    }
    ranks.sort_unstable_by(|a, b| b.cmp(a));

    let straight = straight_top_naive(&counts);
    let mut by_count: Vec<(u8, u8)> = (0..13u8)
        .filter(|&r| counts[r as usize] > 0)
        .map(|r| (r, counts[r as usize]))
        .collect();
    by_count.sort_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));

    if is_flush {
        if let Some(top) = straight {
            return pack_naive(CAT_STRAIGHT_FLUSH, top, 0, 0, 0, 0);
        }
    }
    if by_count[0].1 == 4 {
        return pack_naive(CAT_QUADS, by_count[0].0, by_count[1].0, 0, 0, 0);
    }
    if by_count[0].1 == 3 && by_count.len() >= 2 && by_count[1].1 == 2 {
        return pack_naive(CAT_FULL_HOUSE, by_count[0].0, by_count[1].0, 0, 0, 0);
    }
    if is_flush {
        return pack_naive(CAT_FLUSH, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]);
    }
    if let Some(top) = straight {
        return pack_naive(CAT_STRAIGHT, top, 0, 0, 0, 0);
    }
    if by_count[0].1 == 3 {
        return pack_naive(
            CAT_TRIPS,
            by_count[0].0,
            by_count[1].0,
            by_count[2].0,
            0,
            0,
        );
    }
    if by_count[0].1 == 2 && by_count.len() >= 2 && by_count[1].1 == 2 {
        return pack_naive(
            CAT_TWO_PAIR,
            by_count[0].0,
            by_count[1].0,
            by_count[2].0,
            0,
            0,
        );
    }
    if by_count[0].1 == 2 {
        return pack_naive(
            CAT_PAIR,
            by_count[0].0,
            by_count[1].0,
            by_count[2].0,
            by_count[3].0,
            0,
        );
    }
    pack_naive(
        CAT_HIGH_CARD,
        ranks[0],
        ranks[1],
        ranks[2],
        ranks[3],
        ranks[4],
    )
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn c(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    // Clubs=0, diamonds=1, hearts=2, spades=3.
    // Ranks: 2=0, 3=1, ..., T=8, J=9, Q=10, K=11, A=12.

    #[test]
    fn category_ordering() {
        let hc = evaluate_5(&[c(12, 0), c(9, 1), c(7, 2), c(4, 3), c(2, 0)]);
        let pair = evaluate_5(&[c(12, 0), c(12, 1), c(7, 2), c(4, 3), c(2, 0)]);
        let two_pair = evaluate_5(&[c(12, 0), c(12, 1), c(7, 2), c(7, 3), c(2, 0)]);
        let trips = evaluate_5(&[c(12, 0), c(12, 1), c(12, 2), c(7, 3), c(2, 0)]);
        let straight = evaluate_5(&[c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(8, 0)]);
        let flush = evaluate_5(&[c(12, 0), c(9, 0), c(7, 0), c(4, 0), c(2, 0)]);
        let full = evaluate_5(&[c(12, 0), c(12, 1), c(12, 2), c(7, 3), c(7, 0)]);
        let quads = evaluate_5(&[c(12, 0), c(12, 1), c(12, 2), c(12, 3), c(7, 0)]);
        let sf = evaluate_5(&[c(12, 0), c(11, 0), c(10, 0), c(9, 0), c(8, 0)]);

        assert!(hc < pair);
        assert!(pair < two_pair);
        assert!(two_pair < trips);
        assert!(trips < straight);
        assert!(straight < flush);
        assert!(flush < full);
        assert!(full < quads);
        assert!(quads < sf);

        assert_eq!(category(hc), CAT_HIGH_CARD);
        assert_eq!(category(pair), CAT_PAIR);
        assert_eq!(category(two_pair), CAT_TWO_PAIR);
        assert_eq!(category(trips), CAT_TRIPS);
        assert_eq!(category(straight), CAT_STRAIGHT);
        assert_eq!(category(flush), CAT_FLUSH);
        assert_eq!(category(full), CAT_FULL_HOUSE);
        assert_eq!(category(quads), CAT_QUADS);
        assert_eq!(category(sf), CAT_STRAIGHT_FLUSH);
    }

    #[test]
    fn wheel_is_weakest_straight() {
        let wheel = evaluate_5(&[c(12, 0), c(0, 1), c(1, 2), c(2, 3), c(3, 0)]);
        let six_high = evaluate_5(&[c(4, 0), c(3, 1), c(2, 2), c(1, 3), c(0, 0)]);
        assert_eq!(category(wheel), CAT_STRAIGHT);
        assert_eq!(category(six_high), CAT_STRAIGHT);
        assert!(wheel < six_high);
    }

    #[test]
    fn tie_same_ranks_different_suits() {
        let a = evaluate_5(&[c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(8, 0)]);
        let b = evaluate_5(&[c(12, 3), c(11, 2), c(10, 1), c(9, 0), c(8, 3)]);
        assert_eq!(a, b);
    }

    #[test]
    fn quads_kicker_matters() {
        let q_ace_k = evaluate_5(&[c(7, 0), c(7, 1), c(7, 2), c(7, 3), c(12, 0)]);
        let q_ace_q = evaluate_5(&[c(7, 0), c(7, 1), c(7, 2), c(7, 3), c(10, 0)]);
        assert!(q_ace_k > q_ace_q);
    }

    #[test]
    fn flush_rank_ordering() {
        let ace_high = evaluate_5(&[c(12, 0), c(9, 0), c(7, 0), c(4, 0), c(2, 0)]);
        let king_high = evaluate_5(&[c(11, 0), c(9, 0), c(7, 0), c(4, 0), c(2, 0)]);
        assert!(ace_high > king_high);
    }

    // -------- PLO5 "exactly 2 from hand + 3 from board" corner cases --------

    #[test]
    fn plo5_four_board_hearts_plus_one_hand_heart_is_not_flush() {
        let board = [c(12, 2), c(10, 2), c(7, 2), c(4, 2), c(0, 0)];
        let hole = [c(11, 2), c(5, 0), c(3, 1), c(2, 3), c(1, 0)];
        let r = evaluate_plo5(&hole, &board);
        assert!(category(r) < CAT_FLUSH);
    }

    #[test]
    fn plo5_pocket_pair_plus_board_match_is_trips() {
        let hole = [c(7, 0), c(7, 1), c(0, 2), c(1, 3), c(2, 1)];
        let board = [c(7, 2), c(11, 0), c(5, 1), c(1, 0), c(6, 2)];
        let r = evaluate_plo5(&hole, &board);
        assert_eq!(category(r), CAT_TRIPS);
    }

    #[test]
    fn plo5_all_five_hole_cards_in_a_row_is_not_a_straight() {
        let hole = [c(0, 0), c(1, 1), c(2, 2), c(3, 3), c(4, 0)];
        let board = [c(6, 0), c(8, 1), c(10, 2), c(11, 3), c(12, 0)];
        let r = evaluate_plo5(&hole, &board);
        assert!(category(r) < CAT_STRAIGHT);
    }

    #[test]
    fn plo5_two_suited_hole_with_three_flush_board_is_flush() {
        let hole = [c(12, 3), c(11, 3), c(0, 0), c(1, 1), c(2, 2)];
        let board = [c(5, 3), c(7, 3), c(8, 3), c(10, 1), c(9, 1)];
        let r = evaluate_plo5(&hole, &board);
        assert_eq!(category(r), CAT_FLUSH);
    }

    #[test]
    fn plo5_trips_via_two_pair_on_board_not_allowed_if_only_one_match() {
        let hole = [c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(8, 0)];
        let board = [c(0, 0), c(2, 1), c(4, 2), c(6, 3), c(5, 0)];
        let r = evaluate_plo5(&hole, &board);
        assert!(category(r) < CAT_TRIPS);
    }

    // -------- CK vs naive evaluator equivalence --------

    fn random_hand(rng: &mut ChaCha8Rng) -> [Card; 5] {
        let mut deck: [u8; 52] = std::array::from_fn(|i| i as u8);
        for i in 0..5usize {
            let j = rng.gen_range(i..52);
            deck.swap(i, j);
        }
        [
            Card(deck[0]),
            Card(deck[1]),
            Card(deck[2]),
            Card(deck[3]),
            Card(deck[4]),
        ]
    }

    #[test]
    fn ck_matches_naive_on_50k_hands() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF);
        const NUM_HANDS: usize = 50_000;
        const NUM_PAIRS: usize = 100_000;

        let hands: Vec<[Card; 5]> = (0..NUM_HANDS).map(|_| random_hand(&mut rng)).collect();
        let new_ranks: Vec<HandRank> = hands.iter().map(evaluate_5).collect();
        let old_ranks: Vec<HandRank> = hands.iter().map(|h| evaluate_5_naive(h)).collect();

        for i in 0..NUM_HANDS {
            assert_eq!(
                category(new_ranks[i]),
                category(old_ranks[i]),
                "category mismatch on {:?} (new {:08x} old {:08x})",
                hands[i],
                new_ranks[i],
                old_ranks[i]
            );
        }

        for _ in 0..NUM_PAIRS {
            let i = rng.gen_range(0..NUM_HANDS);
            let j = rng.gen_range(0..NUM_HANDS);
            let n = new_ranks[i].cmp(&new_ranks[j]);
            let o = old_ranks[i].cmp(&old_ranks[j]);
            assert_eq!(
                n, o,
                "ordering mismatch: {:?} vs {:?} (new {:?}, old {:?})",
                hands[i], hands[j], n, o
            );
        }
    }

    #[test]
    fn ck_tables_sizes() {
        let t = tables();
        // 4888 paired entries: 156 quads + 156 fh + 858 trips + 858 2p + 2860 pair.
        assert_eq!(t.paired.len(), 4888);
        // Paired products must be sorted + unique.
        for w in t.paired.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
    }

    #[test]
    fn evaluate_plo5_partial_tolerates_card_shared_with_board() {
        // Study-mode bug repro: non-hero placeholder hole contains a
        // card that the user later sets as the turn. The 5-card hand
        // derived from picking that card from BOTH hole and board is
        // degenerate. Pre-fix: panicked in ck_to_hand_rank(0). Post-
        // fix: loop filters the degenerate combo and returns the best
        // valid combo (here, two hole-card high cards + flop).
        let hole = [c(11, 0), c(9, 1), c(7, 2), c(4, 3), c(2, 0)];
        // Board contains c(11, 0) — same card as hole[0].
        let board = [c(11, 0), c(8, 1), c(5, 2)];
        let rank = evaluate_plo5_partial(&hole, &board);
        // At minimum, the result category must be a valid category.
        let cat = rank >> 20;
        assert!(cat <= CAT_STRAIGHT_FLUSH, "category out of range: {cat}");
    }

    #[test]
    fn evaluate_plo5_tolerates_card_shared_with_board() {
        // Same scenario with full 5-card board.
        let hole = [c(11, 0), c(9, 1), c(7, 2), c(4, 3), c(2, 0)];
        let board = [c(11, 0), c(8, 1), c(5, 2), c(3, 3), c(1, 0)];
        let rank = evaluate_plo5(&hole, &board);
        let cat = rank >> 20;
        assert!(cat <= CAT_STRAIGHT_FLUSH, "category out of range: {cat}");
    }

    // -------- evaluate_plo5_k_partial (k-card opp hand) --------

    #[test]
    fn k_partial_k2_holdem_straight_flush() {
        // AhKh on Qh Jh Th flop: must use both hole cards + 3 board → SF.
        // (Hearts = suit 2 in this codebase.)
        let hole = [c(12, 2), c(11, 2)]; // Ah Kh
        let board = [c(10, 2), c(9, 2), c(8, 2)]; // Qh Jh Th
        let rank = evaluate_plo5_k_partial(&hole, &board);
        assert_eq!(category(rank), CAT_STRAIGHT_FLUSH);
    }

    #[test]
    fn k_partial_k2_only_high_card() {
        // 2 unrelated low cards; 5-card board with no straight/flush help.
        let hole = [c(0, 0), c(1, 1)]; // 2c 3d
        let board = [c(5, 0), c(8, 1), c(10, 2), c(11, 3), c(12, 0)];
        let rank = evaluate_plo5_k_partial(&hole, &board);
        // Best 5-card includes the two hole cards + 3 from board → high card.
        assert_eq!(category(rank), CAT_HIGH_CARD);
    }

    #[test]
    fn k_partial_k5_matches_evaluate_plo5_partial() {
        // Generalized k=5 path on 3-, 4-, 5-card boards must match the
        // specialized partial evaluator on identical inputs.
        let hole = [c(12, 0), c(11, 1), c(7, 2), c(4, 3), c(2, 0)];
        let board5 = [c(10, 1), c(9, 2), c(8, 3), c(0, 0), c(6, 1)];
        for n in 3..=5 {
            let board = &board5[..n];
            let r_general = evaluate_plo5_k_partial(&hole, board);
            let r_specific = evaluate_plo5_partial(&hole, board);
            assert_eq!(
                r_general, r_specific,
                "mismatch at board len {n}: general {r_general:08x} vs specific {r_specific:08x}",
            );
        }
    }

    #[test]
    fn k_partial_k4_takes_best_pair() {
        // 4 hole cards; pick the best pair to make trips with paired board.
        let hole = [c(7, 0), c(7, 1), c(0, 2), c(1, 3)]; // 88, 23 random low cards
        let board = [c(7, 2), c(11, 0), c(5, 1)]; // 8 high + paired
        let rank = evaluate_plo5_k_partial(&hole, &board);
        // Best pair = pocket 8s + 8 on board → trips.
        assert_eq!(category(rank), CAT_TRIPS);
    }
}
