//! Card abstraction helpers: preflop 169 (already in preflop.rs), simple
//! flop / turn equity buckets for memory-safe flop solves.

use rayon::prelude::*;

use super::range::{combo_cards, combo_ranks_on_board, combo_table, NUM_COMBOS};
use super::rng::CfrRng;

/// Number of equity buckets for flop abstraction (OCHS-style target ~200).
pub const FLOP_BUCKETS: usize = 200;
/// Private-view encoding: `OCHS_BUCKET_BASE + bucket_id` (avoids combo 0..1325).
pub const OCHS_BUCKET_BASE: u32 = 2_000_000;
/// Reference hands the bucket score measures equity against.
const N_REF: usize = 16;

/// Assign each unblocked combo a bucket 0..FLOP_BUCKETS-1 on this board (the
/// flop of a flop root, the 4-card board of a turn root; 5 cards also work).
///
/// What it really is (review 2026-09-20 F8 — the old text oversold it): a
/// QUANTILE of one scalar score per combo. The score is `10·mean + L2` of the
/// combo's equities against 16 random reference combos (drawn from `seed`). It
/// is "OCHS-flavoured" only in that the equities are opponent-relative; it is
/// not the published OCHS clustering, and there are 16 features, not ~200 —
/// `FLOP_BUCKETS` (200) is the number of BUCKETS.
///
/// (TOOL-018) The equities are EXACT — every runout of the board is enumerated
/// (990 turn+river pairs on a flop, 46 rivers on a turn), so the score has no
/// sampling noise. It used to be a Monte-Carlo estimate from 8 runouts per
/// reference: about ±0.4 of noise on a score whose 200 quantile buckets are
/// ~0.07 wide, so similar hands scattered across buckets. Turn roots used to be
/// bucketed from the flop alone and now use their own 4-card board. The result
/// depends only on the board and the seed's reference hands — not on the
/// thread count.
///
/// The bucket is computed ONCE at the root and the solver keeps using it on
/// later streets, so later-street infosets cannot see how the runout changed
/// the hand (a made flush and a missed draw share an infoset if they shared a
/// root bucket). Reports label this
/// `card_abs=… (… fixed at the <street> …; coarse abstraction)`.
pub fn equity_buckets(board: &[u8], seed: u64) -> Vec<u16> {
    let mut out = vec![u16::MAX; NUM_COMBOS];
    if !(3..=5).contains(&board.len()) {
        return out;
    }
    let mut blocked = [false; 52];
    for &c in board {
        blocked[c as usize] = true;
    }
    let unblocked = |id: usize| {
        let (a, b) = combo_cards(id);
        !blocked[a as usize] && !blocked[b as usize]
    };

    // N_REF distinct opponent reference combos (OCHS-style opponent clusters).
    let mut rng = CfrRng::new(seed);
    let mut refs: Vec<usize> = Vec::with_capacity(N_REF);
    while refs.len() < N_REF {
        let id = rng.below(NUM_COMBOS);
        if unblocked(id) && !refs.contains(&id) {
            refs.push(id);
        }
    }

    let eq = exact_ref_equities(board, &refs);
    let mut scores = vec![0.0f64; NUM_COMBOS];
    for id in (0..NUM_COMBOS).filter(|&id| unblocked(id)) {
        // A reference that shares a card with the combo counts as a coin flip.
        let hist: Vec<f64> = (0..N_REF)
            .map(|i| {
                let e = eq[id * N_REF + i];
                if e.is_nan() {
                    0.5
                } else {
                    e
                }
            })
            .collect();
        // Bucket by mean HS + L2 of histogram (opponent-relative signature)
        let mean: f64 = hist.iter().sum::<f64>() / N_REF as f64;
        let l2: f64 = hist.iter().map(|x| x * x).sum::<f64>().sqrt();
        scores[id] = mean * 10.0 + l2;
    }

    let mut sorted: Vec<f64> = (0..NUM_COMBOS)
        .filter(|&id| unblocked(id))
        .map(|id| scores[id])
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.is_empty() {
        return out;
    }
    for id in (0..NUM_COMBOS).filter(|&id| unblocked(id)) {
        let rank = sorted.partition_point(|&x| x < scores[id]);
        let b = (rank * FLOP_BUCKETS / sorted.len().max(1)).min(FLOP_BUCKETS - 1);
        out[id] = b as u16;
    }
    out
}

/// Exact equity of every combo against each reference hand over EVERY runout
/// of `board`: `eq[combo * N_REF + i]` (NaN when the combo is blocked by the
/// board or shares a card with reference i). Wins are counted in exact integer
/// half-points, so the parallel sum is the same in any order.
fn exact_ref_equities(board: &[u8], refs: &[usize]) -> Vec<f64> {
    let blen = board.len();
    let free: Vec<u8> = (0..52u8).filter(|c| !board.contains(c)).collect();
    let mut base = [0u8; 5];
    base[..blen].copy_from_slice(board);
    let mut runouts: Vec<[u8; 5]> = Vec::new();
    match blen {
        5 => runouts.push(base),
        4 => runouts.extend(free.iter().map(|&r| {
            let mut b = base;
            b[4] = r;
            b
        })),
        _ => {
            for (i, &t) in free.iter().enumerate() {
                for &r in &free[i + 1..] {
                    let mut b = base;
                    b[3] = t;
                    b[4] = r;
                    runouts.push(b);
                }
            }
        }
    }
    let table = combo_table();
    let cells = NUM_COMBOS * refs.len();
    let zero = || (vec![0u32; cells], vec![0u32; cells]);
    let (wins2, games) = runouts
        .par_chunks(32)
        .map(|chunk| {
            let (mut wins2, mut games) = zero();
            for b in chunk {
                let ranks = combo_ranks_on_board(b);
                for (i, &o) in refs.iter().enumerate() {
                    let ro = ranks[o];
                    if ro == 0 {
                        continue; // the runout uses a card of the reference hand
                    }
                    let (o0, o1) = table[o];
                    for c in 0..NUM_COMBOS {
                        let rc = ranks[c];
                        if rc == 0 {
                            continue; // blocked by the board or the runout
                        }
                        let (c0, c1) = table[c];
                        if c0 == o0 || c0 == o1 || c1 == o0 || c1 == o1 {
                            continue;
                        }
                        let k = c * refs.len() + i;
                        games[k] += 1;
                        wins2[k] += if rc > ro {
                            2
                        } else if rc == ro {
                            1
                        } else {
                            0
                        };
                    }
                }
            }
            (wins2, games)
        })
        .reduce(zero, |(mut w, mut g), (w2, g2)| {
            for k in 0..cells {
                w[k] += w2[k];
                g[k] += g2[k];
            }
            (w, g)
        });
    (0..cells)
        .map(|k| {
            if games[k] > 0 {
                wins2[k] as f64 / (2.0 * games[k] as f64)
            } else {
                f64::NAN
            }
        })
        .collect()
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

/// Suit relabelling: map a hole combo relative to a public board so that
/// suit-permuted (board, hand) PAIRS share one key across different boards.
///
/// Algorithm: build a suit permutation that maps the board's suits to a
/// canonical order (first-seen suit → 0,1,2,3), then apply the same map to
/// hole cards and re-encode the combo id.
///
/// (review 2026-09-20 D15) Two facts callers must respect:
/// - For any FIXED board the map is a permutation of the suits, hence a
///   bijection on combos: it merges NOTHING within one public board (a solve
///   of a fixed river board gets no infoset reduction from it — the reports
///   say `iso=noop_on_fixed_board`). Hands that are strategically identical on
///   a board (e.g. AdKh / AhKs on a mono-club flop) are NOT merged.
/// - `board` must be the PUBLIC board dealt so far. Passing cards that are not
///   public yet (a sampled future river) makes the key depend on hidden
///   information: the first-seen order of a new suit changes the map.
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
        let hand = super::super::range::cards_to_combo(12 * 4, 11 * 4); // Ac Kc
        let hand_rot = super::super::range::cards_to_combo(12 * 4 + 1, 11 * 4 + 1); // Ad Kd
        let k1 = iso_combo_id(hand, &board);
        let k2 = iso_combo_id(hand_rot, &board_rot);
        assert_eq!(
            k1, k2,
            "suit-isomorphic (board,hand) pairs must share infoset key"
        );
        let _ = (aa_hd, aa_hs);
    }

    /// (review 2026-09-20 D15) on one fixed board the relabel is a bijection —
    /// no two combos ever share a key, so it cannot shrink a fixed-board solve.
    #[test]
    fn iso_is_a_bijection_on_a_fixed_board() {
        for board in [vec![0u8, 4, 8], vec![0, 5, 10, 15], vec![3, 17, 22, 40, 51]] {
            let mut seen = std::collections::HashSet::new();
            for combo in 0..NUM_COMBOS {
                assert!(
                    seen.insert(iso_combo_id(combo, &board)),
                    "collision on {board:?}"
                );
            }
            assert_eq!(seen.len(), NUM_COMBOS);
        }
    }

    /// A not-yet-public card must not be passed in: a new suit on the river
    /// changes the map (this is exactly the leak D15 removed from the solver).
    #[test]
    fn iso_key_depends_on_every_card_it_is_given() {
        let combo = super::super::range::cards_to_combo(50, 47); // Ah Ks
        let turn = [0u8, 4, 9, 13]; // suits c, c, d, d
        let with_heart_river = iso_combo_id(combo, &[0, 4, 9, 13, 22]);
        let with_spade_river = iso_combo_id(combo, &[0, 4, 9, 13, 23]);
        assert_ne!(with_heart_river, with_spade_river);
        // The public-board key is one value, whatever comes later.
        assert_eq!(iso_combo_id(combo, &turn), iso_combo_id(combo, &turn));
    }

    #[test]
    fn ochs_uses_200_buckets() {
        assert_eq!(FLOP_BUCKETS, 200);
        let board = [0u8, 10, 20];
        let b = equity_buckets(&board, 1);
        let max_b = b
            .iter()
            .filter(|&&x| x != u16::MAX)
            .max()
            .copied()
            .unwrap_or(0);
        assert!(max_b < 200);
        // Should actually use a wide spread of buckets
        let mut used = std::collections::HashSet::new();
        for &x in &b {
            if x != u16::MAX {
                used.insert(x);
            }
        }
        assert!(
            used.len() > 50,
            "expected many buckets used, got {}",
            used.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flop_buckets_cover_unblocked() {
        let board = [0u8, 10, 20];
        let b = equity_buckets(&board, 1);
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

    /// (TOOL-018) The equities are exact: equal to a brute-force count over
    /// every runout with the hand evaluator.
    #[test]
    fn reference_equities_are_exact() {
        use crate::cards::Card;
        use crate::hand_eval::evaluate_nlh;
        let range_combo = super::super::range::cards_to_combo;
        for board in [vec![0u8, 21, 47], vec![0u8, 21, 47, 30]] {
            let refs = [range_combo(51, 50), range_combo(33, 29), range_combo(8, 12)];
            let eq = exact_ref_equities(&board, &refs);
            for &c in &[range_combo(31, 28), range_combo(46, 42), range_combo(1, 5)] {
                for (i, &o) in refs.iter().enumerate() {
                    let (c0, c1) = combo_cards(c);
                    let (o0, o1) = combo_cards(o);
                    let used: Vec<u8> = board.iter().copied().chain([c0, c1, o0, o1]).collect();
                    let free: Vec<u8> = (0..52u8).filter(|x| !used.contains(x)).collect();
                    let (mut w2, mut n) = (0u32, 0u32);
                    let mut score = |full: &[u8]| {
                        let b: Vec<Card> = full.iter().map(|&x| Card(x)).collect();
                        let b5 = [b[0], b[1], b[2], b[3], b[4]];
                        let (hc, ho) = (
                            evaluate_nlh(&[Card(c0), Card(c1)], &b5),
                            evaluate_nlh(&[Card(o0), Card(o1)], &b5),
                        );
                        n += 1;
                        w2 += if hc > ho {
                            2
                        } else if hc == ho {
                            1
                        } else {
                            0
                        };
                    };
                    if board.len() == 4 {
                        for &r in &free {
                            score(&[board[0], board[1], board[2], board[3], r]);
                        }
                    } else {
                        for (k, &t) in free.iter().enumerate() {
                            for &r in &free[k + 1..] {
                                score(&[board[0], board[1], board[2], t, r]);
                            }
                        }
                    }
                    assert_eq!(
                        eq[c * refs.len() + i],
                        w2 as f64 / (2.0 * n as f64),
                        "board {board:?} combo {c} ref {o}"
                    );
                }
            }
        }
    }

    /// (TOOL-018 C) A turn root is bucketed on its own board: on 2c 7d Ks the
    /// overpair AA is far ahead of the underpair 33; once the 3h turns, 33 is a
    /// set and passes AA — which only happens if the turn card is seen.
    #[test]
    fn turn_buckets_see_the_turn_card() {
        let combo = super::super::range::cards_to_combo;
        let (threes, aces) = (combo(7, 5), combo(49, 48));
        let flop = equity_buckets(&[0, 21, 47], 3);
        let turn = equity_buckets(&[0, 21, 47, 6], 3);
        assert!(
            flop[aces] >= flop[threes] + 30,
            "flop: AA {} 33 {}",
            flop[aces],
            flop[threes]
        );
        assert!(
            turn[threes] > turn[aces],
            "turn: AA {} 33 {}",
            turn[aces],
            turn[threes]
        );
        assert_eq!(turn[combo(6, 2)], u16::MAX, "the turn card blocks");
    }

    /// Buckets depend on the board and the seed only — not on the thread count.
    #[test]
    fn buckets_are_deterministic_across_thread_counts() {
        let board = [3u8, 17, 22];
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let a = one.install(|| equity_buckets(&board, 9));
        let b = equity_buckets(&board, 9);
        assert_eq!(a, b);
    }

    /// (TOOL-018) Evidence: the old 8-runout Monte-Carlo score vs the exact one
    /// on the same references — how far sampling noise moved combos between
    /// buckets — plus the exact version's cost (cargo test --profile fasttest
    /// --lib bucket_noise_report -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn bucket_noise_report() {
        use crate::cards::Card;
        use crate::hand_eval::evaluate_nlh;
        let board = [3u8, 17, 22];
        let t0 = std::time::Instant::now();
        let exact = equity_buckets(&board, 1);
        let t_par = t0.elapsed().as_secs_f64();
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let t0 = std::time::Instant::now();
        let _ = one.install(|| equity_buckets(&board, 1));
        let t_one = t0.elapsed().as_secs_f64();
        // The old estimator: same references, `samples` random runouts per ref.
        let mut rng = CfrRng::new(1);
        let mut refs = Vec::new();
        while refs.len() < N_REF {
            let id = rng.below(NUM_COMBOS);
            let (a, b) = combo_cards(id);
            if !board.contains(&a) && !board.contains(&b) && !refs.contains(&id) {
                refs.push(id);
            }
        }
        for samples in [8u32, 128] {
            let mut scores = vec![f64::NAN; NUM_COMBOS];
            for id in 0..NUM_COMBOS {
                let (c0, c1) = combo_cards(id);
                if board.contains(&c0) || board.contains(&c1) {
                    continue;
                }
                let hist: Vec<f64> = refs
                    .iter()
                    .map(|&o| {
                        let (o0, o1) = combo_cards(o);
                        if [o0, o1].iter().any(|x| *x == c0 || *x == c1) {
                            return 0.5;
                        }
                        let mut w = 0.0;
                        for _ in 0..samples {
                            let mut used: Vec<u8> =
                                vec![board[0], board[1], board[2], c0, c1, o0, o1];
                            let mut draw = || loop {
                                let x = rng.below(52) as u8;
                                if !used.contains(&x) {
                                    used.push(x);
                                    return x;
                                }
                            };
                            let (t, r) = (draw(), draw());
                            let b5 = [
                                Card(board[0]),
                                Card(board[1]),
                                Card(board[2]),
                                Card(t),
                                Card(r),
                            ];
                            let (hc, ho) = (
                                evaluate_nlh(&[Card(c0), Card(c1)], &b5),
                                evaluate_nlh(&[Card(o0), Card(o1)], &b5),
                            );
                            w += if hc > ho {
                                1.0
                            } else if hc == ho {
                                0.5
                            } else {
                                0.0
                            };
                        }
                        w / samples as f64
                    })
                    .collect();
                let mean = hist.iter().sum::<f64>() / N_REF as f64;
                scores[id] = mean * 10.0 + hist.iter().map(|x| x * x).sum::<f64>().sqrt();
            }
            let mut sorted: Vec<f64> = scores.iter().copied().filter(|x| !x.is_nan()).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (mut sum, mut far, mut n) = (0.0, 0, 0);
            for id in 0..NUM_COMBOS {
                if scores[id].is_nan() {
                    continue;
                }
                let rank = sorted.partition_point(|&x| x < scores[id]);
                let b = (rank * FLOP_BUCKETS / sorted.len()).min(FLOP_BUCKETS - 1) as i64;
                let d = (b - exact[id] as i64).abs();
                sum += d as f64;
                far += (d > 10) as usize;
                n += 1;
            }
            println!(
                "{samples:>4} sampled runouts per reference: mean |bucket - exact bucket| = {:.1}, {:.0}% of combos more than 10 buckets off",
                sum / n as f64,
                100.0 * far as f64 / n as f64
            );
        }
        println!("exact buckets: {t_par:.3}s (all threads), {t_one:.3}s (one thread)");
    }
}
