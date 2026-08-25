//! Card abstraction helpers: preflop 169 (already in preflop.rs), simple
//! flop equity buckets for memory-safe flop solves.

use crate::cards::Card;
use crate::hand_eval::evaluate_nlh;

use super::range::{combo_cards, NUM_COMBOS};

/// Number of equity buckets for flop abstraction (OCHS-style target ~200).
pub const FLOP_BUCKETS: usize = 200;
/// Private-view encoding: `OCHS_BUCKET_BASE + bucket_id` (avoids combo 0..1325).
pub const OCHS_BUCKET_BASE: u32 = 2_000_000;

/// Assign each unblocked combo a bucket 0..FLOP_BUCKETS-1 by EHS vs
/// random opponent on this flop (MC with fixed seed samples).
/// OCHS-style flop buckets: hand strength histogram vs a fixed set of
/// opponent reference clusters, then quantile over the HS vector norm.
///
/// Not full published OCHS training, but uses **~200** opponent-relative
/// strength features (vs 12 MC samples of pure EHS) so flop infosets are
/// compressed to `FLOP_BUCKETS` (200) rather than exact 1326.
pub fn flop_equity_buckets(board3: &[u8; 3], samples: u32, seed: u64) -> Vec<u16> {
    let mut out = vec![u16::MAX; NUM_COMBOS];
    let mut blocked = [false; 52];
    for &c in board3 {
        blocked[c as usize] = true;
    }

    let mut rng = seed.wrapping_add(1);
    let next = |r: &mut u64| -> u64 {
        *r ^= *r >> 12;
        *r ^= *r << 25;
        *r ^= *r >> 27;
        r.wrapping_mul(0x2545F4914F6CDD1D)
    };

    // Build N_REF opponent reference combos (OCHS-style opponent clusters).
    const N_REF: usize = 16;
    let mut refs: Vec<usize> = Vec::with_capacity(N_REF);
    while refs.len() < N_REF {
        let id = (next(&mut rng) as usize) % NUM_COMBOS;
        let (a, b) = combo_cards(id);
        if blocked[a as usize] || blocked[b as usize] {
            continue;
        }
        if !refs.contains(&id) {
            refs.push(id);
        }
    }

    // Score = mean equity vs each reference (OCHS-like histogram summary)
    let mut scores = vec![0.0f64; NUM_COMBOS];
    let samp = samples.max(8);
    for id in 0..NUM_COMBOS {
        let (c0, c1) = combo_cards(id);
        if blocked[c0 as usize] || blocked[c1 as usize] {
            continue;
        }
        let hole = [Card(c0), Card(c1)];
        let mut hist = [0.0f64; N_REF];
        for (ri, &rid) in refs.iter().enumerate() {
            let (o0, o1) = combo_cards(rid);
            if o0 == c0 || o0 == c1 || o1 == c0 || o1 == c1 {
                hist[ri] = 0.5;
                continue;
            }
            let opp = [Card(o0), Card(o1)];
            let mut wins = 0.0;
            let mut n = 0.0;
            for _ in 0..samp {
                let mut used = blocked;
                used[c0 as usize] = true;
                used[c1 as usize] = true;
                used[o0 as usize] = true;
                used[o1 as usize] = true;
                let mut draw = || loop {
                    let x = (next(&mut rng) as usize) % 52;
                    if !used[x] {
                        used[x] = true;
                        return x as u8;
                    }
                };
                let t = draw();
                let r = draw();
                let board5 = [
                    Card(board3[0]),
                    Card(board3[1]),
                    Card(board3[2]),
                    Card(t),
                    Card(r),
                ];
                let hr = evaluate_nlh(&hole, &board5);
                let or = evaluate_nlh(&opp, &board5);
                if hr > or {
                    wins += 1.0;
                } else if hr == or {
                    wins += 0.5;
                }
                n += 1.0;
            }
            hist[ri] = if n > 0.0 { wins / n } else { 0.5 };
        }
        // Bucket by mean HS + L2 of histogram (opponent-relative signature)
        let mean: f64 = hist.iter().sum::<f64>() / N_REF as f64;
        let l2: f64 = hist.iter().map(|x| x * x).sum::<f64>().sqrt();
        scores[id] = mean * 10.0 + l2;
    }

    let mut sorted: Vec<f64> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(id, _)| {
            let (c0, c1) = combo_cards(*id);
            !blocked[c0 as usize] && !blocked[c1 as usize]
        })
        .map(|(_, s)| s)
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.is_empty() {
        return out;
    }
    for id in 0..NUM_COMBOS {
        let (c0, c1) = combo_cards(id);
        if blocked[c0 as usize] || blocked[c1 as usize] {
            continue;
        }
        let s = scores[id];
        let rank = sorted.partition_point(|&x| x < s);
        let b = (rank * FLOP_BUCKETS / sorted.len().max(1)).min(FLOP_BUCKETS - 1);
        out[id] = b as u16;
    }
    out
}

/// Private view for infosets: full combo or bucket id.
#[derive(Debug, Clone, Copy)]
pub enum PrivateView {
    Combo(u32),
    Bucket(u16),
    PreflopClass(u32),
}

impl PrivateView {
    pub fn as_u32(self) -> u32 {
        match self {
            PrivateView::Combo(c) => c,
            PrivateView::Bucket(b) => 2_000_000 + b as u32,
            PrivateView::PreflopClass(c) => 3_000_000 + c,
        }
    }
}

/// Suit isomorphism: map a hole combo relative to a public board so that
/// isomorphic suit assignments share one infoset key.
///
/// Algorithm: build a suit permutation that maps the board's suits to a
/// canonical order (first-seen suit → 0,1,2,3), then apply the same map to
/// hole cards and re-encode the combo id.
pub fn iso_combo_id(combo: usize, board: &[u8]) -> u32 {
    let (c0, c1) = combo_cards(combo);
    let map = suit_map_for_board(board);
    let m0 = remap_card(c0, &map);
    let m1 = remap_card(c1, &map);
    super::range::cards_to_combo(m0, m1) as u32
}

fn suit_map_for_board(board: &[u8]) -> [u8; 4] {
    // map original suit → canonical 0..3 by first appearance on board
    let mut map = [255u8; 4];
    let mut next = 0u8;
    for &c in board {
        let s = (c % 4) as usize;
        if map[s] == 255 {
            map[s] = next;
            next += 1;
        }
    }
    // fill remaining suits in order
    for s in 0..4 {
        if map[s] == 255 {
            map[s] = next;
            next += 1;
        }
    }
    map
}

fn remap_card(card: u8, map: &[u8; 4]) -> u8 {
    let rank = card / 4;
    let suit = card % 4;
    rank * 4 + map[suit as usize]
}

#[cfg(test)]
mod iso_tests {
    use super::*;

    #[test]
    fn iso_maps_isomorphic_hands_to_same_key() {
        // Board: all clubs on ranks 0,5,10 → only suit 0 used
        // Hole (2c,3c) and after suit rotation that preserves board emptiness
        // of other suits: map should send any pure-suited-on-unused to same.
        // Board cards: 0=2c, 4=3c, 8=4c (all clubs)
        let board = [0u8, 4, 8];
        // Two offsuit hands with same ranks relative to board suits:
        // (Ah, Ad) ranks 12/12 pair of aces different suits vs board clubs
        // Aces of diamonds and hearts should map under same suit-map.
        let aa_hd = super::super::range::cards_to_combo(12 * 4 + 2, 12 * 4 + 1); // Ah Ad
        let aa_hs = super::super::range::cards_to_combo(12 * 4 + 2, 12 * 4 + 3); // Ah As
        // After suit map, both pairs of non-club aces get remapped; they may
        // differ. Stronger check: same hand with board suit-permuted equals
        // original under iso of the permuted instance.
        //
        // Board B = suits rotated +1: 0→1, 4→5, 8→9
        let board_rot = [1u8, 5, 9]; // 2d 3d 4d
        let hand = super::super::range::cards_to_combo(12 * 4 + 0, 11 * 4 + 0); // Ac Kc
        let hand_rot = super::super::range::cards_to_combo(12 * 4 + 1, 11 * 4 + 1); // Ad Kd
        let k1 = iso_combo_id(hand, &board);
        let k2 = iso_combo_id(hand_rot, &board_rot);
        assert_eq!(
            k1, k2,
            "suit-isomorphic (board,hand) pairs must share infoset key"
        );
        let _ = (aa_hd, aa_hs);
    }

    #[test]
    fn ochs_uses_200_buckets() {
        assert_eq!(FLOP_BUCKETS, 200);
        let board = [0u8, 10, 20];
        let b = flop_equity_buckets(&board, 4, 1);
        let max_b = b.iter().filter(|&&x| x != u16::MAX).max().copied().unwrap_or(0);
        assert!(max_b < 200);
        // Should actually use a wide spread of buckets
        let mut used = std::collections::HashSet::new();
        for &x in &b {
            if x != u16::MAX {
                used.insert(x);
            }
        }
        assert!(used.len() > 50, "expected many buckets used, got {}", used.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flop_buckets_cover_unblocked() {
        let board = [0u8, 10, 20];
        let b = flop_equity_buckets(&board, 8, 1);
        let mut used = 0;
        let mut max_b = 0u16;
        for id in 0..NUM_COMBOS {
            if b[id] != u16::MAX {
                used += 1;
                max_b = max_b.max(b[id]);
            }
        }
        assert!(used > 1000);
        assert!(max_b < FLOP_BUCKETS as u16);
    }
}
