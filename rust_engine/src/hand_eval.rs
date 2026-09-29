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

use crate::cards::{Card, CardMask};

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

/// A [`HandRank`] in 16 bits: category in the top 4, rank-within-category in
/// the low 12 (the widest category, one pair, spans 2,860 < 4,096). Same order
/// as the `HandRank` (and 0 stays 0), for tables held per table in memory
/// (`BoardPairTable`, PERF-026). [`unpack_rank16`] is the exact inverse.
#[inline]
pub fn pack_rank16(rank: HandRank) -> u16 {
    debug_assert!(
        rank >> 20 < 16 && rank & 0xF_FFFF < 1 << 12,
        "not a HandRank: {rank:#x}"
    );
    (((rank >> 20) << 12) | (rank & 0xFFF)) as u16
}

/// The [`HandRank`] a [`pack_rank16`] value stands for.
#[inline]
pub fn unpack_rank16(packed: u16) -> HandRank {
    let p = packed as u32;
    ((p >> 12) << 20) | (p & 0xFFF)
}

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

/// P2: open-addressing table size for the paired-hand lookup (`paired_hash`).
/// 16384 slots for 4888 entries = ~30% load factor (~1.2 probes avg). Power of
/// two so the Fibonacci-hash index is a shift and the probe wrap is a mask.
const PAIRED_HASH_CAP: usize = 16384;

/// Entries of the two rank-bitset tables: every 13-bit rank set.
const RANK_SETS: usize = 1 << 13;

/// A zeroed boxed array, built on the heap (never a large temporary on the
/// stack).
fn boxed_array<T: Copy, const N: usize>(zero: T) -> Box<[T; N]> {
    vec![zero; N]
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| unreachable!("a vec of exactly N elements"))
}

struct CkTables {
    /// 8192-entry table indexed by 13-bit rank bitset. Non-zero only for
    /// bitsets that correspond to a 5-card straight flush or plain flush.
    flushes: Box<[u16; RANK_SETS]>,
    /// 8192-entry table for straight / high card (5 distinct ranks, non-flush).
    unique5: Box<[u16; RANK_SETS]>,
    /// Sorted (prime_product, ck_rank) pairs for paired hands: what
    /// `paired_hash` is built from, kept only for the parity tests
    /// (`ck_tables_sizes`, `paired_hash_matches_binary_search`).
    #[cfg(test)]
    paired: Vec<(u32, u16)>,
    /// P2: open-addressing hash of prime_product -> ck for paired hands. Slot
    /// key 0 = empty (a real product is a product of 5 primes >= 2, never 0).
    /// Single-probe replacement for the ~12-deep `paired` binary search.
    paired_hash: Box<[(u32, u16); PAIRED_HASH_CAP]>,
}

static CK_TABLES: OnceLock<CkTables> = OnceLock::new();

#[inline]
fn tables() -> &'static CkTables {
    CK_TABLES.get_or_init(build_tables)
}

fn build_tables() -> CkTables {
    let mut flushes: Box<[u16; RANK_SETS]> = boxed_array(0);
    let mut unique5: Box<[u16; RANK_SETS]> = boxed_array(0);
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
                        let bs =
                            (1u16 << a) | (1u16 << b) | (1u16 << cc) | (1u16 << d) | (1u16 << e);
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

    // P2: build the open-addressing paired lookup from the same (prod, ck)
    // pairs. Fibonacci hash + linear probe; slot key 0 = empty. Products are
    // distinct (unique prime factorizations) so there are no duplicate keys, and
    // 4888 entries in 16384 slots guarantees a terminating probe on insert.
    let mut paired_hash: Box<[(u32, u16); PAIRED_HASH_CAP]> = boxed_array((0, 0));
    for &(prod, ck) in &paired {
        let mut slot = (prod.wrapping_mul(0x9E3779B9) >> 18) as usize;
        while paired_hash[slot].0 != 0 {
            slot = (slot + 1) & (PAIRED_HASH_CAP - 1);
        }
        paired_hash[slot] = (prod, ck);
    }

    CkTables {
        flushes,
        unique5,
        #[cfg(test)]
        paired,
        paired_hash,
    }
}

// ---------- CK evaluation ----------

/// The CK rank of five encoded cards: [`ck_eval_parts`] of their OR, AND and
/// prime product.
#[inline]
fn ck_eval_inline(c: [u32; 5], t: &CkTables) -> u16 {
    ck_eval_parts(
        c[0] | c[1] | c[2] | c[3] | c[4],
        c[0] & c[1] & c[2] & c[3] & c[4],
        (c[0] & 0xFF) * (c[1] & 0xFF) * (c[2] & 0xFF) * (c[3] & 0xFF) * (c[4] & 0xFF),
        t,
    )
}

/// The weakest (highest) Cactus-Kev rank of each hand category, indexed by
/// category (high card 0 ..= straight flush 8): category c spans
/// `CAT_CK_HI[c + 1] + 1 ..= CAT_CK_HI[c]` (straight flush: 1 ..= 10).
/// `ck_to_hand_rank` is written out as a match for speed; the test
/// `cat_ck_bounds_match_ck_to_hand_rank` pins the two equal (ENG-016).
pub const CAT_CK_HI: [u32; 9] = [7462, 6185, 3325, 2467, 1609, 1599, 322, 166, 10];

/// The Cactus-Kev rank (1 = royal flush ..= 7462 = worst high card) of a
/// [`HandRank`] — the inverse of `ck_to_hand_rank`.
#[inline]
pub fn ck_of_rank(rank: HandRank) -> u32 {
    CAT_CK_HI[category(rank) as usize] - (rank & 0xF_FFFF)
}

#[inline]
fn ck_to_hand_rank(ck: u16) -> HandRank {
    // ck=0 is the sentinel ck_eval_inline returns for SOME 5-card hands
    // holding a duplicated card: single-suit ones (the flush table has no
    // entry for a rank bitset with <5 bits) and 5-of-a-rank multisets. It
    // is NOT a duplicate detector — a duplicate inside a mixed-suit hand
    // hits a real paired-table entry and scores as a genuine pair / trips
    // (review 2026-09-20 C8). Duplicate-free input is therefore the
    // CALLER's contract: production deals satisfy it by construction, and
    // study mode now does too (a user street card that collides with a
    // hidden placeholder hole redraws the placeholder — see
    // `GameState::redraw_colliding_placeholders`; it used to reach this).
    // Downgrade to the worst high-card rank rather than panic; the
    // evaluate_plo* loops skip ck == 0 combos first, so this path only
    // fires on evaluate_5 direct callers.
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

/// C(4,2) = 6 hole-pair index combinations for 4-card (PLO4) holes.
const PAIRS_4: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

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

/// C(6,2) = 15 hole-pair index combinations for 6-card (PLO6) holes.
const PAIRS_6: [(usize, usize); 15] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
    (4, 5),
];

/// C(7,2) = 21 hole-pair index combinations for 7-card holes (PLO67 on the
/// river after three red burns).
const PAIRS_7: [(usize, usize); 21] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 5),
    (4, 6),
    (5, 6),
];

/// Most hole cards a PLO hand holds (PLO67: 4 dealt + 3 red burns).
pub const MAX_PLO_HOLE: usize = 7;
/// C(MAX_PLO_HOLE, 2): the most exactly-2 hole pairs a PLO hand has.
const MAX_PLO_PAIRS: usize = 21;

/// The exactly-2 hole pairs of a `k`-card PLO hole (4..=7), in the fixed
/// lexicographic order every evaluator iterates.
#[inline]
fn plo_pairs(k: usize) -> &'static [(usize, usize)] {
    match k {
        4 => &PAIRS_4,
        5 => &PAIRS_5,
        6 => &PAIRS_6,
        7 => &PAIRS_7,
        _ => panic!("PLO hole must have 4..=7 cards, got {k}"),
    }
}

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

/// Partial-board PLO evaluator (4- to 7-card holes). `board` may have 3, 4, or 5 cards.
/// Returns the best hand hero can currently make using exactly 2 hole + 3
/// visible board cards. Used for the "current hand category" feature pre-river.
pub fn evaluate_plo_partial(hole: &[Card], board: &[Card]) -> HandRank {
    assert!(
        (4..=MAX_PLO_HOLE).contains(&hole.len()),
        "PLO hole must have 4..=7 cards (PLO4 / PLO67 4, PLO5 5, PLO6 6, PLO67 up to 7)"
    );
    assert!(
        board.len() >= 3 && board.len() <= 5,
        "partial board must have 3..=5 cards"
    );
    let t = tables();
    let mut h = [0u32; MAX_PLO_HOLE];
    for (i, c) in hole.iter().enumerate() {
        h[i] = card_to_ck(*c);
    }
    let pairs: &[(usize, usize)] = plo_pairs(hole.len());
    let bn = board.len();
    let mut best_ck: u16 = u16::MAX;
    for i0 in 0..bn {
        for i1 in (i0 + 1)..bn {
            for i2 in (i1 + 1)..bn {
                let b0 = card_to_ck(board[i0]);
                let b1 = card_to_ck(board[i1]);
                let b2 = card_to_ck(board[i2]);
                for &(hi0, hi1) in pairs {
                    let c = [h[hi0], h[hi1], b0, b1, b2];
                    let ck = ck_eval_inline(c, t);
                    // Skip the degenerate 5-card hands the sentinel CAN
                    // flag (single-suit duplicates; see ck_to_hand_rank —
                    // it is not a general duplicate filter). Best-effort
                    // guard only: no engine path feeds duplicates any
                    // more, study mode included (review 2026-09-20 C8).
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
/// [`evaluate_plo_partial`] to k = 2..=7 hole cards. Used for the
/// opp-vs-hero outcome-fraction features.
///
/// For k=2 there is exactly one hole-pair choice. For k=3,4,5 the
/// function enumerates `C(k, 2)` hole-pair × `C(board.len(), 3)`
/// board-triple combinations and returns the strongest 5-card rank.
pub fn evaluate_plo_k_partial(hole: &[Card], board: &[Card]) -> HandRank {
    assert!(
        hole.len() >= 2 && hole.len() <= MAX_PLO_HOLE,
        "hole must have 2..=7 cards"
    );
    assert!(
        board.len() >= 3 && board.len() <= 5,
        "board must have 3..=5 cards"
    );
    let t = tables();
    let kn = hole.len();
    let bn = board.len();
    let mut h_ck = [0u32; MAX_PLO_HOLE];
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

// ---------- v7 obs batch-2 engine dims (V7_OBS_CANDIDATES.md) ----------
//
// BRD-7 (boat_plus_outs), BRD-12 (improve_outs), DUAL-2 (best_pair_mask).
// All follow evaluate_plo_partial's conventions: exactly-2-hole + 3-board,
// lower ck = stronger, degenerate duplicate-card combos (ck == 0) skipped,
// hole widths 4..=7 supported via the PAIRS_* tables (`plo_pairs`).

/// BRD-7: count of unseen next cards that promote hero's best category on
/// `board` to full-house-or-better (quads and straight flushes included).
/// `used` marks every visible card anywhere (hole + BOTH boards) — the
/// unseen deck is global, matching the encoder's cross-board convention.
/// 0 when hero already holds FH+ on this board, and 0 at the river (a
/// 5-card board has no next card).
#[cfg(test)]
pub fn boat_plus_outs(hole: &[Card], board: &[Card], used: CardMask) -> u8 {
    hero_board_one(hole, board, used).boat
}

/// BRD-12: `(improve_outs, best_cat_combo_count)`.
/// - improve_outs: count of DISTINCT unseen next cards whose arrival
///   STRICTLY improves hero's best hand CATEGORY on this board (union
///   across all improvement types — trips→boat, set→quads, draw→flush, …).
///   0 at the river (no next card). Raw count; the encoder normalizes by
///   the actual unseen-deck size.
/// - best_cat_combo_count: how many of hero's exactly-2-hole pairs achieve
///   his CURRENT best category on this board (counterfeit redundancy;
///   well-defined at every street including the river). Raw count; the
///   encoder divides by 10.
#[cfg(test)]
pub fn improve_outs(hole: &[Card], board: &[Card], used: CardMask) -> (u8, u8) {
    let hb = hero_board_one(hole, board, used);
    (hb.improve, hb.combos)
}

/// DUAL-2: 5-bit mask (bit i = slot i) over hero's hole cards SORTED BY
/// CARD INDEX DESCENDING (the project's canonical multiset order) marking
/// the exactly-2 cards of hero's best holding on this board. Rank ties
/// break to the lexicographically smallest (min_card_idx, max_card_idx)
/// pair — deterministic and independent of hole storage order. Returns 0
/// for hole widths > 5 (PLO6 doesn't fit 5 slots), short boards, and
/// all-degenerate study states.
#[cfg(test)]
pub fn best_pair_mask(hole: &[Card], board: &[Card]) -> u8 {
    // used is unused for the mask path; pass a dummy (mask doesn't scan).
    hero_board_one(hole, board, CardMask::EMPTY).mask
}

/// One board's v7 hero dims (BRD-7 / BRD-12 / DUAL-2) and the hero's current
/// made-hand category there — see [`hero_board_one`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeroBoard {
    /// BRD-7 boat-or-better outs (`boat_plus_outs`).
    pub boat: u8,
    /// BRD-12 strict-category-improve outs (`improve_outs`).
    pub improve: u8,
    /// BRD-12 best-category combo count.
    pub combos: u8,
    /// DUAL-2 best-holding hole mask (`best_pair_mask`).
    pub mask: u8,
    /// `category(evaluate_plo_partial(hole, board))` — what
    /// `GameState::hero_category` returns; a free by-product here.
    pub category: u8,
}

/// Fused per-board block for [`crate::engine::GameState::hero_board_v3`]:
/// boat / improve outs, best-category combo count, best-pair mask and the
/// current category, from ONE pass over the (hole pair x board triple) combos
/// and one lookup per unseen card. All-zero for a board with < 3 cards.
///
/// The unseen deck for the out counts is `used`'s complement — GLOBAL (hole +
/// BOTH boards), the encoder's cross-board visibility convention.
///
/// PERF-029: the hole's C(n,2) pairs are Cactus-Kev encoded once
/// ([`PloPairs`]) and the out scan reads [`one_new_card_table`] — the min CK
/// over the combos USING each next card — instead of re-encoding the hole and
/// the board for every unseen card. A combo's CK is a function of its five
/// cards' OR / AND / prime product, which is exactly what the tables combine,
/// and a min is order-free: the same values bit for bit (golden digests + the
/// first-principles `v7_brute_force_tests`). Combos NOT using the new card
/// are the baseline, so comparing the table's category with `cat0` alone
/// gives the same counts as re-evaluating the whole board + card.
pub fn hero_board_one(hole: &[Card], board: &[Card], used: CardMask) -> HeroBoard {
    let bn = board.len();
    if bn < 3 {
        return HeroBoard::default();
    }
    let t = tables();
    let pairs = PloPairs::new(hole);
    let mut b = [0u32; 5];
    for (x, c) in b.iter_mut().zip(board) {
        *x = card_to_ck(*c);
    }
    // Per hole pair: the best (min) CK over the visible board triples.
    let mut pair_best = [u16::MAX; MAX_PLO_PAIRS];
    for i0 in 0..bn {
        for i1 in (i0 + 1)..bn {
            for i2 in (i1 + 1)..bn {
                let (to, ta) = (b[i0] | b[i1] | b[i2], b[i0] & b[i1] & b[i2]);
                let tp = (b[i0] & 0xFF) * (b[i1] & 0xFF) * (b[i2] & 0xFF);
                for (k, best) in pair_best[..pairs.n].iter_mut().enumerate() {
                    let ck =
                        ck_eval_parts(pairs.or[k] | to, pairs.and[k] & ta, pairs.prod[k] * tp, t);
                    if ck != 0 && ck < *best {
                        *best = ck;
                    }
                }
            }
        }
    }
    let n_pairs = pairs.n;

    // ---- best_pair_mask from the pair table ----
    let mask = if hole.len() > 5 {
        0u8
    } else {
        best_pair_mask_from_pairs(hole, &pair_best, n_pairs)
    };

    // ---- cat0 + combo redundancy ----
    let best_ck = pair_best[..n_pairs]
        .iter()
        .copied()
        .min()
        .unwrap_or(u16::MAX);
    let cat0 = category(plo_rank_from_ck(best_ck));
    let combos = pair_best[..n_pairs]
        .iter()
        .filter(|&&pb| pb != u16::MAX && category(ck_to_hand_rank(pb)) == cat0)
        .count() as u8;

    let mut out = HeroBoard {
        boat: 0,
        improve: 0,
        combos,
        mask,
        category: cat0 as u8,
    };
    // River: no next card, so no outs; combos, mask and category still live.
    if bn >= 5 {
        return out;
    }
    let mut deck = [Card(0); 52];
    let mut nd = 0;
    for c in used.unseen() {
        deck[nd] = c;
        nd += 1;
    }
    let deck = &deck[..nd];
    let table = one_new_card_table(&pairs, board, deck);
    // Already FH+: no boat outs, but improve still scans (quads / SF).
    let boat_gate = cat0 < CAT_FULL_HOUSE;
    for c in deck {
        let cat = category(plo_rank_from_ck(table[c.index() as usize]));
        if boat_gate && cat >= CAT_FULL_HOUSE {
            out.boat += 1;
        }
        if cat > cat0 {
            out.improve += 1;
        }
    }
    out
}

/// `best_pair_mask` body given the per-hole-pair best-CK table.
fn best_pair_mask_from_pairs(
    hole: &[Card],
    pair_best: &[u16; MAX_PLO_PAIRS],
    n_pairs: usize,
) -> u8 {
    let pairs: &[(usize, usize)] = plo_pairs(hole.len());
    // (best CK, the pair's card indices low-high, the pair's hole positions)
    type Holding = (u16, (u8, u8), (usize, usize));
    let mut best: Option<Holding> = None;
    for (pi, &pb) in pair_best[..n_pairs].iter().enumerate() {
        if pb == u16::MAX {
            continue;
        }
        let (hi0, hi1) = pairs[pi];
        let (i0, i1) = (hole[hi0].index(), hole[hi1].index());
        let key = (i0.min(i1), i0.max(i1));
        let better = match &best {
            None => true,
            Some((bck, bkey, _)) => pb < *bck || (pb == *bck && key < *bkey),
        };
        if better {
            best = Some((pb, key, (hi0, hi1)));
        }
    }
    let (_, _, (hi0, hi1)) = match best {
        Some(b) => b,
        None => return 0,
    };
    let mut order: Vec<usize> = (0..hole.len()).collect();
    order.sort_by(|&a, &b| hole[b].index().cmp(&hole[a].index()));
    let mut mask = 0u8;
    for win_pos in [hi0, hi1] {
        for (slot, &pos) in order.iter().enumerate() {
            if mask & (1 << slot) != 0 {
                continue;
            }
            if hole[pos].index() == hole[win_pos].index() {
                mask |= 1 << slot;
                break;
            }
        }
    }
    mask
}

// ---------- v7 obs BRD-5 / BRD-6 / DUAL-5 hot paths (Python parity) ----------
//
// Straight windows match encoding.py `_STRAIGHT_WINDOWS` bit-for-bit:
// slot 0 = wheel {A,2,3,4,5}, slots 1..9 = consecutive 5-rank windows
// starting at rank 0..8 (broadway last). Rank bitmasks are 13-bit u16.

/// 10 straight windows as 13-bit rank masks (wheel first → broadway last).
pub const WINDOW_BITS: [u16; 10] = [
    (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3), // wheel
    (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4),
    (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5),
    (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6),
    (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7),
    (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8),
    (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9),
    (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10),
    (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11),
    (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | (1 << 12), // broadway
];

#[inline]
fn ranks_mask(cards: &[Card]) -> u16 {
    let mut m = 0u16;
    for c in cards {
        m |= 1u16 << (c.rank() as u16);
    }
    m
}

/// Unseen copies of each rank given the global `used` mask (hole + both boards).
#[inline]
fn unseen_per_rank(used: CardMask) -> [u8; 13] {
    let mut u = [4u8; 13];
    for (r, x) in u.iter_mut().enumerate() {
        *x -= used.count_rank(r as u8) as u8;
    }
    u
}

/// BRD-5 danger_straight: count of unseen rank-copies that open a field
/// straight hero does not currently make. Bit-exact with the Python
/// frozenset / bitmask formulation in encoding.py.
pub fn danger_straight_outs(hole: &[Card], board: &[Card], used: CardMask) -> u8 {
    if board.len() < 3 || board.len() >= 5 {
        return 0; // river-zeroed by the encoder too; short board = 0
    }
    let board_mask = ranks_mask(board);
    let hole_mask = ranks_mask(hole);
    let unseen = unseen_per_rank(used);
    let mut danger = 0u8;
    for r in 0..13u16 {
        let ur = unseen[r as usize];
        if ur == 0 {
            continue;
        }
        let new_board = board_mask | (1u16 << r);
        let mut field = false;
        for &w in &WINDOW_BITS {
            if (w & new_board).count_ones() >= 3 {
                field = true;
                break;
            }
        }
        if !field {
            continue;
        }
        let mut hero_makes = false;
        for &w in &WINDOW_BITS {
            let l = w & !new_board;
            let n_l = l.count_ones();
            let n_h = (w & hole_mask).count_ones();
            if n_h >= 2 && n_l <= 2 && (l & !hole_mask) == 0 {
                hero_makes = true;
                break;
            }
        }
        if !hero_makes {
            danger = danger.saturating_add(ur);
        }
    }
    danger
}

/// BRD-6 `(union_outs, nut_outs)`. Bit-exact with encoding.py.
pub fn straight_out_union(hole: &[Card], board: &[Card], used: CardMask) -> (u8, u8) {
    if board.len() < 3 || board.len() >= 5 {
        return (0, 0);
    }
    let board_mask = ranks_mask(board);
    let hole_mask = ranks_mask(hole);
    let unseen = unseen_per_rank(used);
    let mut union_mask = 0u16;
    for &w in &WINDOW_BITS {
        let l = w & !board_mask;
        let n_l = l.count_ones();
        let n_h = (w & hole_mask).count_ones();
        let makes = (l & !hole_mask) == 0 && n_h >= 2 && n_l <= 2;
        if makes || n_h < 2 {
            continue;
        }
        let m = l & !hole_mask;
        let n_m = m.count_ones();
        if n_l == 3 && n_m == 0 {
            union_mask |= l;
        } else if n_m == 1 && (1..=3).contains(&n_l) {
            union_mask |= m;
        }
    }
    let mut union_outs = 0u8;
    let mut nut_outs = 0u8;
    for r in 0..13u16 {
        if (union_mask >> r) & 1 == 0 {
            continue;
        }
        let ur = unseen[r as usize];
        union_outs = union_outs.saturating_add(ur);
        let new_board = board_mask | (1u16 << r);
        let mut h_max: i8 = -1;
        for (wi, &w) in WINDOW_BITS.iter().enumerate() {
            let l = w & !new_board;
            let n_l = l.count_ones();
            let n_h = (w & hole_mask).count_ones();
            if (l & !hole_mask) == 0 && n_h >= 2 && n_l <= 2 {
                let wi_i = wi as i8;
                if wi_i > h_max {
                    h_max = wi_i;
                }
            }
        }
        if h_max < 0 {
            continue;
        }
        let mut nut_dist = 0u8;
        for wi in ((h_max as usize) + 1)..10 {
            if (WINDOW_BITS[wi] & new_board).count_ones() >= 3 {
                nut_dist += 1;
            }
        }
        if nut_dist == 0 {
            nut_outs = nut_outs.saturating_add(ur);
        }
    }
    (union_outs, nut_outs)
}

/// DUAL-5 scoop-pair count: # of 2-rank pairs that complete a straight on
/// BOTH boards. Bit-exact with encoding.py `_scoop_pair_count`.
pub fn scoop_pair_count(board_a: &[Card], board_b: &[Card]) -> u8 {
    let ba = ranks_mask(board_a);
    let bb = ranks_mask(board_b);
    let mut scoop = 0u8;
    for r1 in 0..13u16 {
        for r2 in (r1 + 1)..13u16 {
            let pair = (1u16 << r1) | (1u16 << r2);
            let mut made_a = false;
            let mut made_b = false;
            for &w in &WINDOW_BITS {
                if (pair & w) != pair {
                    continue;
                }
                if (pair | (ba & w)).count_ones() >= 5 {
                    made_a = true;
                }
                if (pair | (bb & w)).count_ones() >= 5 {
                    made_b = true;
                }
                if made_a && made_b {
                    break;
                }
            }
            if made_a && made_b {
                scoop += 1;
            }
        }
    }
    scoop
}

/// Fused board-draw hot block for the v7 encoder (BRD-5 danger_straight,
/// BRD-6 union/nut, DUAL-5 scoop). Returns
/// `[ds_a, ds_b, u_a, n_a, u_b, n_b, scoop]` — raw counts; the encoder
/// applies `/unseen_deck` and `/78` normalizations. All-zero on short
/// boards / river (for the river-zeroed dims) matching Python.
pub fn board_draw_v3(hole: &[Card], board_a: &[Card], board_b: &[Card], used: CardMask) -> [u8; 7] {
    let mut out = [0u8; 7];
    if board_a.len() < 3 || board_b.len() < 3 {
        return out;
    }
    // River: BRD-5/6 are river-zeroed; scoop still lives (board-only).
    let is_river = board_a.len() >= 5; // both boards advance together
    if !is_river {
        out[0] = danger_straight_outs(hole, board_a, used);
        out[1] = danger_straight_outs(hole, board_b, used);
        let (ua, na) = straight_out_union(hole, board_a, used);
        let (ub, nb) = straight_out_union(hole, board_b, used);
        out[2] = ua;
        out[3] = na;
        out[4] = ub;
        out[5] = nb;
    }
    out[6] = scoop_pair_count(board_a, board_b);
    out
}

/// NLH evaluator: best 5-card hand from ANY combination of hole + board
/// cards (0, 1, or 2 hole cards may play — "play the board" included).
/// `hole` must have exactly 2 cards; `board` 3..=5 (partial boards give
/// the current best made hand, mirroring [`evaluate_plo_partial`]).
/// Enumerates all C(hole+board, 5) five-card subsets of the pooled
/// cards — 1 at the flop, 6 at the turn, 21 at the river.
pub fn evaluate_nlh(hole: &[Card], board: &[Card]) -> HandRank {
    assert!(hole.len() == 2, "NLH hole must have exactly 2 cards");
    assert!(
        board.len() >= 3 && board.len() <= 5,
        "board must have 3..=5 cards"
    );
    let t = tables();
    let pn = hole.len() + board.len();
    let mut pool = [0u32; 7];
    for (i, c) in hole.iter().chain(board.iter()).enumerate() {
        pool[i] = card_to_ck(*c);
    }
    let mut best_ck: u16 = u16::MAX;
    for i0 in 0..pn {
        for i1 in (i0 + 1)..pn {
            for i2 in (i1 + 1)..pn {
                for i3 in (i2 + 1)..pn {
                    for i4 in (i3 + 1)..pn {
                        let c = [pool[i0], pool[i1], pool[i2], pool[i3], pool[i4]];
                        let ck = ck_eval_inline(c, t);
                        // Degenerate (duplicate-card) subsets eval to 0;
                        // unreachable in production deals but keep the
                        // same guard discipline as the PLO evaluators.
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

/// PLO evaluator (4- to 7-card holes): exactly 2 from hole + 3 from board.
/// Enumerates all C(hole, 2) × 10 combinations and returns the best rank.
pub fn evaluate_plo(hole: &[Card], board: &[Card; 5]) -> HandRank {
    assert!(
        (4..=MAX_PLO_HOLE).contains(&hole.len()),
        "PLO hole must have 4..=7 cards (PLO4 / PLO67 4, PLO5 5, PLO6 6, PLO67 up to 7)"
    );
    let t = tables();
    let mut h = [0u32; MAX_PLO_HOLE];
    for (i, c) in hole.iter().enumerate() {
        h[i] = card_to_ck(*c);
    }
    let pairs: &[(usize, usize)] = plo_pairs(hole.len());
    let b: [u32; 5] = [
        card_to_ck(board[0]),
        card_to_ck(board[1]),
        card_to_ck(board[2]),
        card_to_ck(board[3]),
        card_to_ck(board[4]),
    ];
    let mut best_ck: u16 = u16::MAX; // lower CK = stronger
    for &(h0, h1) in pairs {
        for &(b0, b1, b2) in &TRIPLES_5 {
            let c = [h[h0], h[h1], b[b0], b[b1], b[b2]];
            let ck = ck_eval_inline(c, t);
            // Skip degenerate 5-card hands (duplicate card), see
            // note in evaluate_plo_partial.
            if ck != 0 && ck < best_ck {
                best_ck = ck;
            }
        }
    }
    let safe_ck = if best_ck == u16::MAX { 7462 } else { best_ck };
    ck_to_hand_rank(safe_ck)
}

// ---------- PLO evaluation against many boards (EV runouts) ----------

/// One PLO hole's C(hole, 2) two-card halves, Cactus-Kev encoded ONCE for a
/// hand that is evaluated against many boards (an EV runout replays 64 board
/// pairs). A 5-card combo's rank bitset, flush test and prime product are the
/// OR / AND / product of its five encoded cards -- i.e. the pair's combined
/// with the board triple's -- so [`plo_best_ck`] returns exactly the minimum
/// CK that [`evaluate_plo`] finds. (Five primes <= 41 multiply to at most
/// 41^5 < 2^32: the regrouped u32 product is the same number.)
#[derive(Clone, Copy)]
pub struct PloPairs {
    or: [u32; MAX_PLO_PAIRS],
    and: [u32; MAX_PLO_PAIRS],
    prod: [u32; MAX_PLO_PAIRS],
    n: usize,
}

impl PloPairs {
    /// No pairs at all (a folded seat nobody ranks).
    pub const EMPTY: PloPairs = PloPairs {
        or: [0; MAX_PLO_PAIRS],
        and: [0; MAX_PLO_PAIRS],
        prod: [0; MAX_PLO_PAIRS],
        n: 0,
    };

    pub fn new(hole: &[Card]) -> Self {
        assert!(
            (4..=MAX_PLO_HOLE).contains(&hole.len()),
            "PLO hole must have 4..=7 cards (PLO4 / PLO67 4, PLO5 5, PLO6 6, PLO67 up to 7)"
        );
        let mut h = [0u32; MAX_PLO_HOLE];
        for (i, c) in hole.iter().enumerate() {
            h[i] = card_to_ck(*c);
        }
        let pairs: &[(usize, usize)] = plo_pairs(hole.len());
        let mut out = Self {
            n: pairs.len(),
            ..Self::EMPTY
        };
        for (k, &(a, b)) in pairs.iter().enumerate() {
            out.or[k] = h[a] | h[b];
            out.and[k] = h[a] & h[b];
            out.prod[k] = (h[a] & 0xFF) * (h[b] & 0xFF);
        }
        out
    }
}

/// The ten 3-card triples of a 5-card board (`TRIPLES_5` order), encoded
/// like [`PloPairs`].
#[derive(Clone, Copy)]
pub struct BoardTriples {
    or: [u32; 10],
    and: [u32; 10],
    prod: [u32; 10],
}

impl BoardTriples {
    /// The triples made only of the first `known` cards of `board` (the
    /// others stay zero -- select these with [`fixed_triples_mask`]); later
    /// positions are never read, so they may be undealt ([`crate::cards::NO_CARD`]).
    pub fn of_known(board: &[Card; 5], known: usize) -> Self {
        let mut b = [0u32; 5];
        for (x, c) in b.iter_mut().zip(&board[..known.min(5)]) {
            *x = card_to_ck(*c);
        }
        let mut out = Self {
            or: [0; 10],
            and: [0; 10],
            prod: [0; 10],
        };
        for (j, &(x, y, z)) in TRIPLES_5.iter().enumerate() {
            if z < known {
                out.or[j] = b[x] | b[y] | b[z];
                out.and[j] = b[x] & b[y] & b[z];
                out.prod[j] = (b[x] & 0xFF) * (b[y] & 0xFF) * (b[z] & 0xFF);
            }
        }
        out
    }

    pub fn new(board: &[Card; 5]) -> Self {
        let b = [
            card_to_ck(board[0]),
            card_to_ck(board[1]),
            card_to_ck(board[2]),
            card_to_ck(board[3]),
            card_to_ck(board[4]),
        ];
        let mut out = Self {
            or: [0; 10],
            and: [0; 10],
            prod: [0; 10],
        };
        for (j, &(x, y, z)) in TRIPLES_5.iter().enumerate() {
            out.or[j] = b[x] | b[y] | b[z];
            out.and[j] = b[x] & b[y] & b[z];
            out.prod[j] = (b[x] & 0xFF) * (b[y] & 0xFF) * (b[z] & 0xFF);
        }
        out
    }
}

/// Every one of the ten board triples.
pub const ALL_TRIPLES: u16 = (1 << 10) - 1;

/// Bit j set = `TRIPLES_5[j]` uses only board positions `0..known` -- the
/// triples every runout of a board with `known` cards out shares.
pub fn fixed_triples_mask(known: usize) -> u16 {
    let mut m = 0u16;
    for (j, &(x, y, z)) in TRIPLES_5.iter().enumerate() {
        if x < known && y < known && z < known {
            m |= 1 << j;
        }
    }
    m
}

/// The ten triples of a board with `known` cards out, split by how many of
/// their positions are still to come: (none, exactly one, two or more).
pub fn triple_masks(known: usize) -> (u16, u16, u16) {
    let (mut fixed, mut one, mut multi) = (0u16, 0u16, 0u16);
    for (j, &(x, y, z)) in TRIPLES_5.iter().enumerate() {
        match [x, y, z].iter().filter(|&&p| p >= known).count() {
            0 => fixed |= 1 << j,
            1 => one |= 1 << j,
            _ => multi |= 1 << j,
        }
    }
    (fixed, one, multi)
}

/// Per next card: the min CK over `pairs` x every triple made of two of the
/// `known` board cards plus that card (u16::MAX where no combo counts; only
/// the `deck` cards are filled). For any runout of the board, the min over
/// its exactly-one-new-card triples is the min of this table over the new
/// cards -- so a runout sample looks those up instead of evaluating them.
pub fn one_new_card_table(pairs: &PloPairs, known: &[Card], deck: &[Card]) -> [u16; 52] {
    let t = tables();
    assert!(known.len() <= 5, "a board has at most 5 cards");
    let mut kc = [0u32; 5];
    for (x, c) in kc.iter_mut().zip(known) {
        *x = card_to_ck(*c);
    }
    let kc = &kc[..known.len()];
    // The known board's C(k, 2) <= 10 two-card halves (fixed storage: this
    // runs per row in the encoders).
    let mut kp = [(0u32, 0u32, 0u32); 10];
    let mut nkp = 0;
    for a in 0..kc.len() {
        for b in (a + 1)..kc.len() {
            kp[nkp] = (
                kc[a] | kc[b],
                kc[a] & kc[b],
                (kc[a] & 0xFF) * (kc[b] & 0xFF),
            );
            nkp += 1;
        }
    }
    let mut table = [u16::MAX; 52];
    for &c in deck {
        let e = card_to_ck(c);
        let mut best = u16::MAX;
        for &(o, a, p) in &kp[..nkp] {
            let (to, ta, tp) = (o | e, a & e, p * (e & 0xFF));
            for k in 0..pairs.n {
                let ck = ck_eval_parts(pairs.or[k] | to, pairs.and[k] & ta, pairs.prod[k] * tp, t);
                if ck != 0 && ck < best {
                    best = ck;
                }
            }
        }
        table[c.index() as usize] = best;
    }
    table
}

/// Min CK (lower = stronger; `u16::MAX` = none yet) over `pairs` x the
/// board triples selected by `tmask`, folded into `best`. Degenerate combos
/// (ck 0) are skipped exactly as in [`evaluate_plo`], and a min is
/// order-free, so splitting the triples across calls gives the same value.
#[inline]
pub fn plo_best_ck(pairs: &PloPairs, triples: &BoardTriples, tmask: u16, mut best: u16) -> u16 {
    let t = tables();
    for k in 0..pairs.n {
        let (po, pa, pp) = (pairs.or[k], pairs.and[k], pairs.prod[k]);
        let mut bits = tmask;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let ck = ck_eval_parts(
                po | triples.or[j],
                pa & triples.and[j],
                pp * triples.prod[j],
                t,
            );
            if ck != 0 && ck < best {
                best = ck;
            }
        }
    }
    best
}

/// The [`HandRank`] of a [`plo_best_ck`] result (the tail of
/// [`evaluate_plo`]: worst rank when every combo was degenerate).
#[inline]
pub fn plo_rank_from_ck(best: u16) -> HandRank {
    ck_to_hand_rank(if best == u16::MAX { 7462 } else { best })
}

/// The Cactus-Kev lookup (the one copy; [`ck_eval_inline`] and the
/// pre-combined [`PloPairs`] / [`BoardTriples`] paths all end here): a 5-card
/// combo from the OR, AND and prime product of its five encoded cards.
#[inline]
fn ck_eval_parts(or: u32, and: u32, prod: u32, t: &CkTables) -> u16 {
    let q = (or >> 16) as usize & (RANK_SETS - 1);
    if (and & 0xF000) != 0 {
        return t.flushes[q];
    }
    let u = t.unique5[q];
    if u != 0 {
        return u;
    }
    // P2: single-probe open-addressing lookup, replacing a ~12-deep binary
    // search. Key-verified: probe until the stored product matches `prod`
    // (return its ck) or an empty slot (key 0) is reached. The empty-slot case
    // returns the 0 sentinel — only a multiset no real hand can form (5 of a
    // rank, e.g. 41^5, from duplicated cards) has no table entry; every deal
    // the engine produces is duplicate-free and can't reach it. The table is
    // <100% full, so a missing key always reaches an empty slot.
    let mut slot = (prod.wrapping_mul(0x9E3779B9) >> 18) as usize;
    loop {
        let (k, v) = t.paired_hash[slot];
        if k == prod {
            return v;
        }
        if k == 0 {
            return 0;
        }
        slot = (slot + 1) & (PAIRED_HASH_CAP - 1);
    }
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
        return pack_naive(CAT_TRIPS, by_count[0].0, by_count[1].0, by_count[2].0, 0, 0);
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
    use crate::test_util::c;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

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
        let r = evaluate_plo(&hole, &board);
        assert!(category(r) < CAT_FLUSH);
    }

    #[test]
    fn plo5_pocket_pair_plus_board_match_is_trips() {
        let hole = [c(7, 0), c(7, 1), c(0, 2), c(1, 3), c(2, 1)];
        let board = [c(7, 2), c(11, 0), c(5, 1), c(1, 0), c(6, 2)];
        let r = evaluate_plo(&hole, &board);
        assert_eq!(category(r), CAT_TRIPS);
    }

    #[test]
    fn plo5_all_five_hole_cards_in_a_row_is_not_a_straight() {
        let hole = [c(0, 0), c(1, 1), c(2, 2), c(3, 3), c(4, 0)];
        let board = [c(6, 0), c(8, 1), c(10, 2), c(11, 3), c(12, 0)];
        let r = evaluate_plo(&hole, &board);
        assert!(category(r) < CAT_STRAIGHT);
    }

    #[test]
    fn plo5_two_suited_hole_with_three_flush_board_is_flush() {
        let hole = [c(12, 3), c(11, 3), c(0, 0), c(1, 1), c(2, 2)];
        let board = [c(5, 3), c(7, 3), c(8, 3), c(10, 1), c(9, 1)];
        let r = evaluate_plo(&hole, &board);
        assert_eq!(category(r), CAT_FLUSH);
    }

    #[test]
    fn plo5_trips_via_two_pair_on_board_not_allowed_if_only_one_match() {
        let hole = [c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(8, 0)];
        let board = [c(0, 0), c(2, 1), c(4, 2), c(6, 3), c(5, 0)];
        let r = evaluate_plo(&hole, &board);
        assert!(category(r) < CAT_TRIPS);
    }

    // -------- Pair/triple (EV-runout) evaluator == evaluate_plo --------

    #[test]
    fn plo_best_ck_matches_evaluate_plo() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5EED_CAFE);
        for case in 0..30_000 {
            let hole_w = 4 + (case % 3);
            let mut deck: [u8; 52] = std::array::from_fn(|i| i as u8);
            for i in 0..(hole_w + 5) {
                let j = rng.gen_range(i..52);
                deck.swap(i, j);
            }
            let hole: Vec<Card> = deck[..hole_w].iter().map(|&c| Card(c)).collect();
            let board: [Card; 5] = std::array::from_fn(|k| Card(deck[hole_w + k]));
            let want = evaluate_plo(&hole, &board);
            let pairs = PloPairs::new(&hole);
            let tri = BoardTriples::new(&board);
            assert_eq!(
                plo_rank_from_ck(plo_best_ck(&pairs, &tri, ALL_TRIPLES, u16::MAX)),
                want,
                "case {case}"
            );
            // Split at every "cards already out" count: same best CK.
            for known in 0..=5 {
                let fixed = fixed_triples_mask(known);
                let pre = plo_best_ck(&pairs, &tri, fixed, u16::MAX);
                let got = plo_best_ck(&pairs, &tri, ALL_TRIPLES & !fixed, pre);
                assert_eq!(plo_rank_from_ck(got), want, "case {case} known {known}");
            }
        }
        // Fixed triples: 3 cards out -> only the flop triple; 4 -> four; 5 -> all.
        assert_eq!(fixed_triples_mask(2), 0);
        assert_eq!(fixed_triples_mask(3).count_ones(), 1);
        assert_eq!(fixed_triples_mask(4).count_ones(), 4);
        assert_eq!(fixed_triples_mask(5), ALL_TRIPLES);
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
        let old_ranks: Vec<HandRank> = hands.iter().map(evaluate_5_naive).collect();

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
    fn rank16_round_trips_every_rank_in_order() {
        // Every one of the 7,462 CK classes (the worst high card is rank 0,
        // the value an empty table slot holds): unpack(pack(r)) == r, and
        // packing keeps the order.
        let mut ranks: Vec<HandRank> = (1..=7462u16).map(ck_to_hand_rank).collect();
        assert!(ranks.contains(&0));
        ranks.sort_unstable();
        for w in ranks.windows(2) {
            assert!(
                pack_rank16(w[0]) < pack_rank16(w[1]),
                "{:#x} {:#x}",
                w[0],
                w[1]
            );
        }
        for &r in &ranks {
            assert_eq!(unpack_rank16(pack_rank16(r)), r, "{r:#x}");
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
    fn paired_hash_matches_binary_search() {
        // P2: the open-addressing paired lookup must return the identical ck to
        // the old sorted-Vec binary search for every product, and the 0
        // sentinel for any product not in the table.
        let t = tables();
        let probe = |prod: u32| -> u16 {
            let mut slot = (prod.wrapping_mul(0x9E3779B9) >> 18) as usize;
            loop {
                let (k, v) = t.paired_hash[slot];
                if k == prod {
                    return v;
                }
                if k == 0 {
                    return 0;
                }
                slot = (slot + 1) & (PAIRED_HASH_CAP - 1);
            }
        };
        // All 4888 real products resolve to the same ck as the binary search.
        for &(prod, ck) in &t.paired {
            assert_eq!(probe(prod), ck, "hash != stored ck for prod {prod}");
            let bs = match t.paired.binary_search_by_key(&prod, |&(p, _)| p) {
                Ok(i) => t.paired[i].1,
                Err(_) => 0,
            };
            assert_eq!(probe(prod), bs, "hash != binary search for prod {prod}");
        }
        // Not-in-table products return the 0 degenerate sentinel.
        assert_eq!(probe(41u32.pow(5)), 0, "five-of-a-kind (41^5) sentinel");
        assert_eq!(probe(2), 0);
        assert_eq!(probe(u32::MAX), 0);
    }

    #[test]
    fn evaluate_plo_partial_tolerates_card_shared_with_board() {
        // Study-mode bug repro: non-hero placeholder hole contains a
        // card that the user later sets as the turn. The 5-card hand
        // derived from picking that card from BOTH hole and board is
        // degenerate. Pre-fix: panicked in ck_to_hand_rank(0). Post-
        // fix: the evaluator never panics on it and returns a valid
        // rank. (Since review 2026-09-20 C8 study mode redraws the
        // colliding placeholder, so the engine no longer produces this
        // input; the no-panic guarantee stays pinned at this level.)
        let hole = [c(11, 0), c(9, 1), c(7, 2), c(4, 3), c(2, 0)];
        // Board contains c(11, 0) — same card as hole[0].
        let board = [c(11, 0), c(8, 1), c(5, 2)];
        let rank = evaluate_plo_partial(&hole, &board);
        // At minimum, the result category must be a valid category.
        let cat = rank >> 20;
        assert!(cat <= CAT_STRAIGHT_FLUSH, "category out of range: {cat}");
    }

    #[test]
    fn evaluate_plo_tolerates_card_shared_with_board() {
        // Same scenario with full 5-card board.
        let hole = [c(11, 0), c(9, 1), c(7, 2), c(4, 3), c(2, 0)];
        let board = [c(11, 0), c(8, 1), c(5, 2), c(3, 3), c(1, 0)];
        let rank = evaluate_plo(&hole, &board);
        let cat = rank >> 20;
        assert!(cat <= CAT_STRAIGHT_FLUSH, "category out of range: {cat}");
    }

    #[test]
    fn ck_sentinel_is_not_a_duplicate_detector() {
        // (review 2026-09-20 C8) Pins the LIMIT of the ck == 0 filter so
        // nobody leans on it: the sentinel fires for a duplicated card only
        // when all five cards share a suit (flush table miss). In a
        // mixed-suit hand the duplicate is scored as a genuine pair — which
        // is why study mode keeps duplicates out at the source (placeholder
        // redraw in engine.rs) instead of relying on the evaluator.
        let t = tables();
        let ck = |cards: [Card; 5]| ck_eval_inline(cards.map(card_to_ck), t);
        // Kh Kh 9h 6h 3h — single suit: flagged.
        assert_eq!(ck([c(11, 2), c(11, 2), c(7, 2), c(4, 2), c(1, 2)]), 0);
        // Kh Kh 9d 6c 3s — mixed suits: indistinguishable from Kh Ks 9d 6c 3s.
        let dup = ck([c(11, 2), c(11, 2), c(7, 1), c(4, 0), c(1, 3)]);
        let real = ck([c(11, 2), c(11, 3), c(7, 1), c(4, 0), c(1, 3)]);
        assert_ne!(dup, 0);
        assert_eq!(dup, real);
        assert_eq!(category(ck_to_hand_rank(dup)), CAT_PAIR);
    }

    // -------- evaluate_plo_k_partial (k-card opp hand) --------

    #[test]
    fn k_partial_k2_holdem_straight_flush() {
        // AhKh on Qh Jh Th flop: must use both hole cards + 3 board → SF.
        // (Hearts = suit 2 in this codebase.)
        let hole = [c(12, 2), c(11, 2)]; // Ah Kh
        let board = [c(10, 2), c(9, 2), c(8, 2)]; // Qh Jh Th
        let rank = evaluate_plo_k_partial(&hole, &board);
        assert_eq!(category(rank), CAT_STRAIGHT_FLUSH);
    }

    #[test]
    fn k_partial_k2_only_high_card() {
        // 2 unrelated low cards; 5-card board with no straight/flush help.
        let hole = [c(0, 0), c(1, 1)]; // 2c 3d
        let board = [c(5, 0), c(8, 1), c(10, 2), c(11, 3), c(12, 0)];
        let rank = evaluate_plo_k_partial(&hole, &board);
        // Best 5-card includes the two hole cards + 3 from board → high card.
        assert_eq!(category(rank), CAT_HIGH_CARD);
    }

    #[test]
    fn k_partial_k5_matches_evaluate_plo_partial() {
        // Generalized k=5 path on 3-, 4-, 5-card boards must match the
        // specialized partial evaluator on identical inputs.
        let hole = [c(12, 0), c(11, 1), c(7, 2), c(4, 3), c(2, 0)];
        let board5 = [c(10, 1), c(9, 2), c(8, 3), c(0, 0), c(6, 1)];
        for n in 3..=5 {
            let board = &board5[..n];
            let r_general = evaluate_plo_k_partial(&hole, board);
            let r_specific = evaluate_plo_partial(&hole, board);
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
        let rank = evaluate_plo_k_partial(&hole, &board);
        // Best pair = pocket 8s + 8 on board → trips.
        assert_eq!(category(rank), CAT_TRIPS);
    }

    // ---- NLH any-combo evaluator ----

    #[test]
    fn nlh_one_hole_card_flush() {
        // Illegal under PLO's exactly-2 rule; the whole point of NLH eval.
        let hole = [c(12, 2), c(0, 0)]; // Ah 2c
        let board = [c(11, 2), c(10, 2), c(9, 2), c(7, 2), c(1, 3)]; // Kh Qh Jh 9h 3s
        let rank = evaluate_nlh(&hole, &board);
        assert_eq!(category(rank), CAT_FLUSH);
    }

    #[test]
    fn nlh_zero_hole_cards_plays_the_board() {
        let hole = [c(0, 0), c(1, 1)]; // 2c 3d
        let board = [c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(8, 0)]; // broadway
        let rank = evaluate_nlh(&hole, &board);
        assert_eq!(category(rank), CAT_STRAIGHT);
        // Identical to another junk hand playing the same board.
        let other = evaluate_nlh(&[c(0, 2), c(1, 3)], &board);
        assert_eq!(rank, other);
    }

    #[test]
    fn nlh_two_hole_cards_when_best() {
        let hole = [c(12, 0), c(12, 1)]; // AcAd
        let board = [c(12, 2), c(7, 3), c(5, 1), c(2, 0), c(0, 2)];
        let rank = evaluate_nlh(&hole, &board);
        assert_eq!(category(rank), CAT_TRIPS);
    }

    #[test]
    fn nlh_partial_boards() {
        // Flop: pool of exactly 5 → the single possible hand.
        let hole = [c(12, 0), c(12, 1)];
        let flop = [c(12, 2), c(7, 3), c(5, 1)];
        assert_eq!(category(evaluate_nlh(&hole, &flop)), CAT_TRIPS);
        // Turn adds a pairing card → full house among C(6,5) subsets.
        let turn = [c(12, 2), c(7, 3), c(5, 1), c(7, 0)];
        assert_eq!(category(evaluate_nlh(&hole, &turn)), CAT_FULL_HOUSE);
    }

    #[test]
    fn nlh_matches_exhaustive_reference_on_random_deals() {
        // Cross-check the pooled-combination evaluator against a direct
        // "best evaluate_5 over C(7,5)" reference on random full boards.
        let mut rng = ChaCha8Rng::seed_from_u64(0xD1CE);
        for _ in 0..200 {
            // Draw 7 distinct cards.
            let mut idx: Vec<u8> = (0..52).collect();
            for i in 0..7 {
                let j = rng.gen_range(i..52);
                idx.swap(i, j);
            }
            let cards: Vec<Card> = idx[..7].iter().map(|&i| Card::from_index(i)).collect();
            let hole = [cards[0], cards[1]];
            let board = [cards[2], cards[3], cards[4], cards[5], cards[6]];
            let got = evaluate_nlh(&hole, &board);

            let mut best = 0u32;
            let pool = &cards[..7];
            for a in 0..7 {
                for b in (a + 1)..7 {
                    for cc in (b + 1)..7 {
                        for d in (cc + 1)..7 {
                            for e in (d + 1)..7 {
                                let r = evaluate_5(&[pool[a], pool[b], pool[cc], pool[d], pool[e]]);
                                if r > best {
                                    best = r;
                                }
                            }
                        }
                    }
                }
            }
            assert_eq!(got, best);
        }
    }
}

#[cfg(test)]
mod plo67_tests {
    //! PLO67 holds up to SEVEN hole cards (4 dealt + one per red face-up
    //! burn): every PLO evaluator must score them exactly like a brute
    //! force over all C(k, 2) x C(board, 3) five-card hands.
    use super::*;
    use rand::seq::SliceRandom;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn brute(hole: &[Card], board: &[Card]) -> HandRank {
        let mut best: Option<HandRank> = None;
        for i in 0..hole.len() {
            for j in (i + 1)..hole.len() {
                for a in 0..board.len() {
                    for b in (a + 1)..board.len() {
                        for c in (b + 1)..board.len() {
                            let r = evaluate_5(&[hole[i], hole[j], board[a], board[b], board[c]]);
                            best = Some(best.map_or(r, |x| x.max(r)));
                        }
                    }
                }
            }
        }
        best.unwrap()
    }

    fn deal(rng: &mut ChaCha8Rng, k: usize, b: usize) -> (Vec<Card>, Vec<Card>) {
        let mut deck: Vec<Card> = (0..52u8).map(Card::from_index).collect();
        deck.shuffle(rng);
        (deck[..k].to_vec(), deck[k..k + b].to_vec())
    }

    #[test]
    fn seven_card_holes_match_brute_force_every_evaluator() {
        let mut rng = ChaCha8Rng::seed_from_u64(67);
        for k in 4..=7usize {
            for _ in 0..400 {
                let (hole, board) = deal(&mut rng, k, 5);
                let full: [Card; 5] = board.clone().try_into().unwrap();
                let want = brute(&hole, &board);
                assert_eq!(evaluate_plo(&hole, &full), want, "evaluate_plo k={k}");
                assert_eq!(evaluate_plo_partial(&hole, &board), want, "partial k={k}");
                assert_eq!(
                    evaluate_plo_k_partial(&hole, &board),
                    want,
                    "k_partial k={k}"
                );
                // the EV-runout path: pairs encoded once, triples by mask
                let ck = plo_best_ck(
                    &PloPairs::new(&hole),
                    &BoardTriples::new(&full),
                    ALL_TRIPLES,
                    u16::MAX,
                );
                assert_eq!(plo_rank_from_ck(ck), want, "PloPairs k={k}");
                for known in [3usize, 4] {
                    assert_eq!(
                        evaluate_plo_partial(&hole, &board[..known]),
                        brute(&hole, &board[..known]),
                        "partial k={k} board {known}"
                    );
                }
            }
        }
    }

    /// The 7th card must be able to play: trip aces need both hole aces,
    /// dealt as the 6th and 7th cards (a pair only PAIRS_7 covers).
    #[test]
    fn the_seventh_card_plays() {
        let c = |r: u8, s: u8| Card::new(r, s);
        let hole = [
            c(0, 0),
            c(1, 1),
            c(5, 3),
            c(6, 2),
            c(3, 0),
            c(12, 3),
            c(12, 1),
        ];
        let board = [c(12, 0), c(11, 1), c(9, 2), c(2, 3), c(7, 1)];
        assert_eq!(category(evaluate_plo(&hole, &board)), CAT_TRIPS);
        assert_eq!(
            category(evaluate_plo_partial(&hole, &board[..3])),
            CAT_TRIPS
        );
        // ... and exactly two hole cards still play: five hearts in hand
        // with two on the board is no flush.
        let hearts = [
            c(12, 2),
            c(11, 2),
            c(9, 2),
            c(7, 2),
            c(3, 2),
            c(0, 0),
            c(1, 1),
        ];
        let two = [c(5, 2), c(6, 2), c(2, 3), c(4, 0), c(8, 1)];
        assert!(category(evaluate_plo(&hearts, &two)) < CAT_FLUSH);
    }

    #[test]
    fn hero_board_features_accept_seven_cards() {
        let mut rng = ChaCha8Rng::seed_from_u64(76);
        for _ in 0..50 {
            let (hole, board) = deal(&mut rng, 7, 4);
            let used = CardMask::of(hole.iter().chain(board.iter()));
            let HeroBoard { combos, mask, .. } = hero_board_one(&hole, &board, used);
            assert!(combos as usize <= 21);
            assert_eq!(mask, 0, "the 5-slot mask is PLO5-only");
        }
    }
}

#[cfg(test)]
mod plo6_tests {
    use super::*;
    use crate::test_util::c;

    /// The 6th hole card must participate: trip aces need BOTH hole aces,
    /// which sit at hole indices 4 and 5 — a pair only PAIRS_6 covers.
    #[test]
    fn plo6_sixth_card_pairs_into_trips() {
        let hole = [c(0, 0), c(1, 1), c(5, 3), c(6, 2), c(12, 3), c(12, 1)];
        let board = [c(12, 0), c(11, 1), c(9, 2)];
        let r = evaluate_plo_partial(&hole, &board);
        assert_eq!(category(r), CAT_TRIPS, "As+Ad (hole idx 4,5) + board Ac");
    }

    /// Exactly-2-hole rule survives 6-card holes: four hearts in hand +
    /// two on board is NOT a flush (needs exactly 2 hole + 3 board).
    #[test]
    fn plo6_exactly_two_hole_cards_rule() {
        let hole = [c(12, 2), c(11, 2), c(9, 2), c(7, 2), c(2, 0), c(3, 1)];
        let board = [c(5, 2), c(6, 2), c(1, 3)];
        let r = evaluate_plo_partial(&hole, &board);
        assert!(
            category(r) < CAT_FLUSH,
            "2 board hearts cannot complete a flush regardless of hole hearts"
        );
        // ...but three board hearts CAN.
        let board5 = [c(5, 2), c(6, 2), c(1, 2), c(0, 3), c(3, 3)];
        let r5 = evaluate_plo(&hole, &board5);
        assert_eq!(category(r5), CAT_FLUSH);
    }

    /// k_partial generic accepts 6-card holes (opp-outcome path).
    #[test]
    fn plo6_k_partial_six_cards() {
        let hole = [c(12, 0), c(12, 1), c(3, 2), c(4, 3), c(8, 0), c(9, 1)];
        let board = [c(12, 2), c(7, 3), c(2, 0)];
        let r = evaluate_plo_k_partial(&hole, &board);
        assert_eq!(category(r), CAT_TRIPS);
    }

    /// PLO5 result is identical through the widened evaluator (regression:
    /// the pairs-table selection must not disturb 5-card behavior).
    #[test]
    fn plo5_path_unchanged_by_widening() {
        let hole = [c(12, 0), c(11, 1), c(9, 2), c(7, 3), c(2, 0)];
        let board = [c(12, 2), c(11, 3), c(4, 1)];
        let r = evaluate_plo_partial(&hole, &board);
        assert_eq!(category(r), CAT_TWO_PAIR);
    }
}

#[cfg(test)]
mod plo4_tests {
    use super::*;
    use crate::test_util::c;

    /// PAIRS_4 must cover the trailing pair (hole idx 2,3): pocket aces
    /// there + a board ace = trips.
    #[test]
    fn plo4_trailing_pair_makes_trips() {
        let hole = [c(0, 0), c(5, 1), c(12, 3), c(12, 1)];
        let board = [c(12, 0), c(11, 1), c(9, 2)];
        let r = evaluate_plo_partial(&hole, &board);
        assert_eq!(category(r), CAT_TRIPS, "As+Ad (hole idx 2,3) + board Ac");
    }

    /// Exactly-2-hole rule with 4-card holes: three hearts in hand + two
    /// on board is NOT a flush; three board hearts complete it.
    #[test]
    fn plo4_exactly_two_hole_cards_rule() {
        let hole = [c(12, 2), c(11, 2), c(9, 2), c(2, 0)];
        let board = [c(5, 2), c(6, 2), c(1, 3)];
        let r = evaluate_plo_partial(&hole, &board);
        assert!(
            category(r) < CAT_FLUSH,
            "2 board hearts cannot complete a flush regardless of hole hearts"
        );
        let board5 = [c(5, 2), c(6, 2), c(1, 2), c(0, 3), c(3, 3)];
        let r5 = evaluate_plo(&hole, &board5);
        assert_eq!(category(r5), CAT_FLUSH);
    }

    /// The specialized 4-card path must agree with the generic k-partial
    /// evaluator on identical inputs (they enumerate the same 6 pairs).
    #[test]
    fn plo4_partial_matches_k_partial() {
        let hole = [c(12, 0), c(11, 1), c(7, 2), c(4, 3)];
        let board5 = [c(10, 1), c(9, 2), c(8, 3), c(0, 0), c(6, 1)];
        for n in 3..=5 {
            let board = &board5[..n];
            let r_specific = evaluate_plo_partial(&hole, board);
            let r_general = evaluate_plo_k_partial(&hole, board);
            assert_eq!(
                r_specific, r_general,
                "mismatch at board len {n}: specific {r_specific:08x} vs general {r_general:08x}",
            );
        }
    }

    /// PLO5 result is identical through the 4-card widening (regression).
    #[test]
    fn plo5_path_unchanged_by_plo4_widening() {
        let hole = [c(12, 0), c(11, 1), c(9, 2), c(7, 3), c(2, 0)];
        let board = [c(12, 2), c(11, 3), c(4, 1)];
        let r = evaluate_plo_partial(&hole, &board);
        assert_eq!(category(r), CAT_TWO_PAIR);
    }
}

#[cfg(test)]
mod v7_dim_tests {
    //! v7 obs batch-2 engine dims (BRD-7 / BRD-12 / DUAL-2) — hand-computed
    //! cases. Card index = rank*4 + suit (rank 0=deuce..12=ace).
    use super::*;
    use crate::test_util::c;

    fn used_mask(hole: &[Card], boards: &[&[Card]]) -> CardMask {
        CardMask::of(hole.iter().chain(boards.iter().flat_map(|b| b.iter())))
    }

    // Shared fixture: hole AA KK 2 (suits 0,1/2,3/0) on board A-7-3
    // (suits 2,0,1) — hero has top set
    // (trips). FH+ outs by hand: three 7s (AA+A77 boat), three 3s (AA+A33
    // boat), one A (AAAA quads) = 7. No flush/SF is reachable with one
    // card (rainbow board), so improve == boat here; only the AA pair
    // achieves the trips category.
    fn fixture() -> ([Card; 5], [Card; 3]) {
        (
            [c(12, 0), c(12, 1), c(11, 2), c(11, 3), c(0, 0)],
            [c(12, 2), c(5, 0), c(1, 1)],
        )
    }

    #[test]
    fn boat_plus_outs_top_set() {
        let (hole, board) = fixture();
        let used = used_mask(&hole, &[&board]);
        assert_eq!(boat_plus_outs(&hole, &board, used), 7);
    }

    #[test]
    fn boat_plus_outs_zero_when_already_boat_or_river() {
        // Hero already FH+: hole AA332 on board A33 → 33 + (3,3,x) = quad
        // threes (even stronger than the aces-full read of AA + A33) —
        // either way cat0 >= FULL_HOUSE, which is the gate under test.
        let hole = [c(12, 0), c(12, 1), c(1, 2), c(1, 3), c(0, 0)];
        let board = [c(12, 2), c(1, 1), c(1, 0)];
        let used = used_mask(&hole, &[&board]);
        assert!(category(evaluate_plo_partial(&hole, &board)) >= CAT_FULL_HOUSE);
        assert_eq!(boat_plus_outs(&hole, &board, used), 0);
        // River board (5 cards): no next card → 0 by construction.
        let (hole2, b3) = fixture();
        let board5 = [b3[0], b3[1], b3[2], c(8, 2), c(3, 3)];
        let used5 = used_mask(&hole2, &[&board5]);
        assert_eq!(boat_plus_outs(&hole2, &board5, used5), 0);
    }

    #[test]
    fn improve_outs_top_set() {
        let (hole, board) = fixture();
        let used = used_mask(&hole, &[&board]);
        let (outs, combos) = improve_outs(&hole, &board, used);
        assert_eq!(outs, 7, "improve == boat outs on this rainbow board");
        assert_eq!(combos, 1, "only the AA pair achieves trips");
    }

    #[test]
    fn improve_outs_river_keeps_combos() {
        let (hole, b3) = fixture();
        let board5 = [b3[0], b3[1], b3[2], c(8, 2), c(3, 3)];
        let used = used_mask(&hole, &[&board5]);
        let (outs, combos) = improve_outs(&hole, &board5, used);
        assert_eq!(outs, 0, "no next card at the river");
        assert_eq!(combos, 1, "combo redundancy stays defined at the river");
    }

    #[test]
    fn improve_outs_cross_board_used_mask_excludes_other_board() {
        // The unseen deck is GLOBAL: a 7 sitting on the OTHER board must
        // not count as an out. Same fixture, but a 7 visible on board B.
        let (hole, board) = fixture();
        let board_b = [c(5, 1), c(9, 2), c(2, 3)];
        let used = used_mask(&hole, &[&board, &board_b]);
        assert_eq!(boat_plus_outs(&hole, &board, used), 6);
        let (outs, _) = improve_outs(&hole, &board, used);
        assert_eq!(outs, 6);
    }

    #[test]
    fn best_pair_mask_top_set() {
        let (hole, board) = fixture();
        // Winning holding = the two aces, card indices 48 + 49. Hole
        // sorted by index DESC: [49, 48, 47, 46, 0] → slots 0/1 → mask 0b11.
        assert_eq!(best_pair_mask(&hole, &board), 0b0000_0011);
    }

    #[test]
    fn best_pair_mask_tie_break_lexicographic() {
        // Hole AAAA2 on K72: every AA pair ties at pair-of-aces (identical
        // 5-card hand). Tie-break → lexicographically smallest card-index
        // pair = card indices (48, 49). Sorted desc: [51, 50, 49, 48, 0] →
        // slots 2 and 3 → mask 0b1100.
        let hole = [c(12, 0), c(12, 1), c(12, 2), c(12, 3), c(0, 0)];
        let board = [c(11, 0), c(5, 1), c(0, 2)];
        assert_eq!(best_pair_mask(&hole, &board), 0b0000_1100);
    }

    #[test]
    fn best_pair_mask_exactly_two_bits() {
        // Property over a spread of deals: exactly 2 bits set whenever the
        // board exists (5-card PLO hole, non-degenerate).
        let hole = [c(3, 0), c(7, 1), c(9, 2), c(11, 3), c(0, 1)];
        for r in 0..10u8 {
            let board = [c(r, 3), c((r + 2) % 13, 0), c((r + 5) % 13, 1)];
            let mask = best_pair_mask(&hole, &board);
            assert_eq!(mask.count_ones(), 2, "board seed {r}: mask {mask:#b}");
        }
    }

    #[test]
    fn board_draw_v3_river_zeros_outs() {
        let (hole, b3) = fixture();
        let board5 = [b3[0], b3[1], b3[2], c(8, 2), c(3, 3)];
        let board_b = [c(5, 1), c(9, 2), c(2, 3), c(7, 0), c(4, 1)];
        let used = used_mask(&hole, &[&board5, &board_b]);
        let out = board_draw_v3(&hole, &board5, &board_b, used);
        // river → BRD-5/6 zeroed; scoop may still be non-zero
        assert_eq!(&out[..6], &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn scoop_pair_count_identical_boards() {
        // T-J-Q on both boards: the only windows holding all three board
        // ranks are 8-Q, 9-K and T-A, completed by 8+9, 9+K and K+A — three
        // pairs, each scooping (the boards are the same).
        let board = [c(8, 0), c(9, 1), c(10, 2)]; // T,J,Q
        assert_eq!(scoop_pair_count(&board, &board), 3);
        // A second board with no three-rank window scoops nothing.
        let low = [c(0, 0), c(4, 1), c(9, 2)]; // 2, 6, J
        assert_eq!(scoop_pair_count(&board, &low), 0);
    }
}

#[cfg(test)]
mod ck_bounds_tests {
    use super::*;

    #[test]
    fn cat_ck_bounds_match_ck_to_hand_rank() {
        for ck in 1..=7462u16 {
            let r = ck_to_hand_rank(ck);
            let c = category(r) as usize;
            assert!(
                ck as u32 <= CAT_CK_HI[c] && (c == 8 || ck as u32 > CAT_CK_HI[c + 1]),
                "ck {ck}"
            );
            assert_eq!(ck_of_rank(r), ck as u32);
        }
    }
}

#[cfg(test)]
mod v7_brute_force_tests {
    //! TEST-022: first-principles references for the v7 engine dims, checked
    //! over random deals for every hole width (4..=7). The references share
    //! nothing with the fast paths except the 5-card evaluator (`evaluate_5`,
    //! itself pinned against the naive evaluator): they enumerate explicit
    //! "2 hole + 3 board" hands, add each unseen card to the board and
    //! re-evaluate from scratch, and define straights on rank SETS by
    //! enumerating hole pairs and board triples instead of window bitmasks.
    use super::*;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Best "exactly 2 hole + 3 board" hand, by explicit enumeration.
    fn best(hole: &[Card], board: &[Card]) -> HandRank {
        let mut top = 0;
        for i in 0..hole.len() {
            for j in (i + 1)..hole.len() {
                top = top.max(best_of_pair(hole[i], hole[j], board));
            }
        }
        top
    }

    fn best_of_pair(a: Card, b: Card, board: &[Card]) -> HandRank {
        let mut top = 0;
        for x in 0..board.len() {
            for y in (x + 1)..board.len() {
                for z in (y + 1)..board.len() {
                    top = top.max(evaluate_5(&[a, b, board[x], board[y], board[z]]));
                }
            }
        }
        top
    }

    /// (boat_plus_outs, improve_outs, best_cat_combo_count, best_pair_mask)
    /// from their definitions.
    fn reference(hole: &[Card], board: &[Card], used: CardMask) -> (u8, u8, u8, u8) {
        let cat0 = category(best(hole, board));
        let mut combos = 0u8;
        // The best holding: highest rank, ties to the smallest (lo, hi) card
        // indices.
        type Holding = (HandRank, (u8, u8), (Card, Card));
        let mut win: Option<Holding> = None;
        for i in 0..hole.len() {
            for j in (i + 1)..hole.len() {
                let r = best_of_pair(hole[i], hole[j], board);
                if category(r) == cat0 {
                    combos += 1;
                }
                let (x, y) = (hole[i].index(), hole[j].index());
                let key = (x.min(y), x.max(y));
                let better = match win {
                    None => true,
                    Some((br, bk, _)) => r > br || (r == br && key < bk),
                };
                if better {
                    win = Some((r, key, (hole[i], hole[j])));
                }
            }
        }
        let mut mask = 0u8;
        if hole.len() <= 5 {
            let mut sorted: Vec<Card> = hole.to_vec();
            sorted.sort_by_key(|q| std::cmp::Reverse(q.index()));
            let (_, _, (a, b)) = win.unwrap();
            for (slot, c) in sorted.iter().enumerate() {
                if *c == a || *c == b {
                    mask |= 1 << slot;
                }
            }
        }
        if board.len() >= 5 {
            return (0, 0, combos, mask);
        }
        let (mut boat, mut improve) = (0u8, 0u8);
        for i in 0..52u8 {
            if used.contains(Card(i)) {
                continue;
            }
            let mut next = board.to_vec();
            next.push(Card(i));
            let cat = category(best(hole, &next));
            if cat0 < CAT_FULL_HOUSE && cat >= CAT_FULL_HOUSE {
                boat += 1;
            }
            if cat > cat0 {
                improve += 1;
            }
        }
        (boat, improve, combos, mask)
    }

    /// Rank sets as sorted distinct rank lists.
    fn ranks(cards: &[Card]) -> Vec<u8> {
        let mut r: Vec<u8> = cards.iter().map(|c| c.rank()).collect();
        r.sort_unstable();
        r.dedup();
        r
    }

    /// Five distinct ranks form a straight (ace plays low in the wheel).
    fn is_straight(mut r: [u8; 5]) -> bool {
        r.sort_unstable();
        (r.windows(2).all(|w| w[1] == w[0] + 1)) || r == [0, 1, 2, 3, 12]
    }

    /// Hero makes a straight on these board ranks: two hole cards of distinct
    /// ranks + three distinct board ranks.
    fn hero_makes(hole: &[u8], board: &[u8]) -> bool {
        for i in 0..hole.len() {
            for j in (i + 1)..hole.len() {
                for x in 0..board.len() {
                    for y in (x + 1)..board.len() {
                        for z in (y + 1)..board.len() {
                            let five = [hole[i], hole[j], board[x], board[y], board[z]];
                            let mut d = five.to_vec();
                            d.sort_unstable();
                            d.dedup();
                            if d.len() == 5 && is_straight(five) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Some two ranks complete a straight with three of these board ranks.
    fn field_straight(board: &[u8]) -> bool {
        (0..13u8).any(|a| (a + 1..13).any(|b| hero_makes(&[a, b], board)))
    }

    fn unseen_of_rank(used: CardMask, r: u8) -> u8 {
        (0..4).filter(|s| !used.contains(Card(r * 4 + s))).count() as u8
    }

    /// BRD-5 danger: unseen copies of ranks that open a field straight hero
    /// does not make. BRD-6: ranks that complete a straight window hero does
    /// not make now (one card), and those where hero's best straight is then
    /// the nuts among field straights.
    fn draws_reference(hole: &[Card], board: &[Card], used: CardMask) -> (u8, u8, u8) {
        if board.len() < 3 || board.len() >= 5 {
            return (0, 0, 0);
        }
        let hole_r: Vec<u8> = hole.iter().map(|c| c.rank()).collect();
        let hole_set = ranks(hole);
        let board_r = ranks(board);
        let (mut danger, mut union, mut nut) = (0u8, 0u8, 0u8);
        // A window = 5 consecutive ranks (index 0 = the wheel).
        let window = |wi: usize| -> Vec<u8> {
            if wi == 0 {
                vec![12, 0, 1, 2, 3]
            } else {
                ((wi - 1) as u8..(wi + 4) as u8).collect()
            }
        };
        let makes_window = |b: &[u8], wi: usize| -> bool {
            // two distinct hole ranks in the window, the other three on b
            let w = window(wi);
            for i in 0..hole_set.len() {
                for j in (i + 1)..hole_set.len() {
                    let (a, c) = (hole_set[i], hole_set[j]);
                    if w.contains(&a)
                        && w.contains(&c)
                        && w.iter()
                            .filter(|&&x| x != a && x != c)
                            .all(|x| b.contains(x))
                    {
                        return true;
                    }
                }
            }
            false
        };
        for r in 0..13u8 {
            let u = unseen_of_rank(used, r);
            if u == 0 {
                continue;
            }
            let mut nb = board_r.clone();
            if !nb.contains(&r) {
                nb.push(r);
                nb.sort_unstable();
            }
            if field_straight(&nb) && !hero_makes(&hole_r, &nb) {
                danger += u;
            }
            let completes = (0..10).any(|wi| !makes_window(&board_r, wi) && makes_window(&nb, wi));
            if completes {
                union += u;
                let h_max = (0..10).filter(|&wi| makes_window(&nb, wi)).max().unwrap();
                let beaten = (h_max + 1..10)
                    .any(|wi| window(wi).iter().filter(|x| nb.contains(x)).count() >= 3);
                if !beaten {
                    nut += u;
                }
            }
        }
        (danger, union, nut)
    }

    /// DUAL-5: rank pairs that complete a straight on BOTH boards.
    fn scoop_reference(a: &[Card], b: &[Card]) -> u8 {
        let (ra, rb) = (ranks(a), ranks(b));
        let mut n = 0;
        for x in 0..13u8 {
            for y in (x + 1)..13u8 {
                if hero_makes(&[x, y], &ra) && hero_makes(&[x, y], &rb) {
                    n += 1;
                }
            }
        }
        n
    }

    #[test]
    fn v7_dims_match_first_principles_on_random_deals() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x7E57_0022);
        let mut deck: Vec<Card> = (0..52).map(Card).collect();
        let mut seen_boat = 0;
        let mut seen_improve = 0;
        let mut seen_danger = 0;
        let mut seen_nut = 0;
        let mut seen_scoop = 0;
        for trial in 0..1500 {
            deck.shuffle(&mut rng);
            let hole_n = 4 + trial % 4; // 4..=7
            let board_n = [3usize, 4, 5][rng.gen_range(0..3)];
            let hole = &deck[..hole_n];
            let board_a = &deck[hole_n..hole_n + board_n];
            let board_b = &deck[hole_n + board_n..hole_n + 2 * board_n];
            let used = CardMask::of(hole.iter().chain(board_a).chain(board_b));
            for board in [board_a, board_b] {
                let hb = hero_board_one(hole, board, used);
                let got = (hb.boat, hb.improve, hb.combos, hb.mask);
                assert_eq!(
                    got,
                    reference(hole, board, used),
                    "hole {hole:?} board {board:?}"
                );
                assert_eq!(hb.category as u32, category(best(hole, board)));
                seen_boat += (got.0 > 0) as usize;
                seen_improve += (got.1 > got.0) as usize;
                let (d, u, nt) = draws_reference(hole, board, used);
                assert_eq!(
                    danger_straight_outs(hole, board, used),
                    d,
                    "{hole:?} {board:?}"
                );
                assert_eq!(
                    straight_out_union(hole, board, used),
                    (u, nt),
                    "{hole:?} {board:?}"
                );
                seen_danger += (d > 0) as usize;
                seen_nut += (nt > 0) as usize;
            }
            let scoop = scoop_pair_count(board_a, board_b);
            assert_eq!(
                scoop,
                scoop_reference(board_a, board_b),
                "{board_a:?} {board_b:?}"
            );
            seen_scoop += (scoop > 0) as usize;
            let draw = board_draw_v3(hole, board_a, board_b, used);
            assert_eq!(draw[6], scoop);
        }
        // The deals must actually exercise every branch.
        assert!(
            seen_boat > 50 && seen_improve > 50,
            "{seen_boat} {seen_improve}"
        );
        assert!(
            seen_danger > 50 && seen_nut > 20 && seen_scoop > 20,
            "{seen_danger} {seen_nut} {seen_scoop}"
        );
    }
}
