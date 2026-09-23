//! Layered side-pot payout distribution across two boards.
//!
//! Pot layers are built from unique sorted values in `total_commit`. For each
//! layer, eligible seats (`total_commit >= level && !folded`) contest that
//! layer's chips, with half going to the best PLO5 hand on each board. Ties
//! on a board half split that half; odd-chip remainders are awarded to the
//! first tied winner encountered clockwise from button. Chips no alive seat
//! ever matched — a layer whose contributors all folded, or a folded seat's
//! commit above a lone survivor's — go back to the seats that put them in
//! (review 2026-09-20 C1; unreachable on real lines, see the helpers).

use crate::cards::Card;
use crate::hand_eval::{evaluate_nlh, evaluate_plo5, HandRank};

/// Most seats a [`PotLayers`] bitmask holds (the engine's tables are far smaller).
const MAX_LAYER_SEATS: usize = 32;

/// The side-pot structure of a settled hand. It depends only on who folded and
/// how much each seat committed — not on the cards — so an EV runout
/// (`GameState::payouts_ev`) builds it ONCE and replays it for every sampled
/// board pair instead of re-sorting the commit levels per sample.
pub struct PotLayers {
    /// Fold-out: the lone survivor (no showdown, boards never inspected).
    single_survivor: Option<usize>,
    /// Contested (or orphaned) layers in ascending commit order.
    layers: Vec<PotLayer>,
}

#[derive(Clone, Copy)]
struct PotLayer {
    level: u64,
    layer_each: u64,
    chips: u64,
    /// Bit `i` set = seat `i` contests the layer (commit >= level, not
    /// folded). 0 = orphan layer (refunded to its contributors).
    eligible: u32,
}

impl PotLayers {
    /// Same layer construction as the original per-call loop: unique commit
    /// levels ascending, a 0 level skipped, `prev_level` advanced before an
    /// empty layer is skipped.
    pub fn new(folded: &[bool], total_commit: &[u64]) -> Self {
        let n = total_commit.len();
        assert_eq!(folded.len(), n);
        assert!(n <= MAX_LAYER_SEATS, "PotLayers supports at most {MAX_LAYER_SEATS} seats");
        let mut alive = (0..n).filter(|&i| !folded[i]);
        if let (Some(only), None) = (alive.next(), alive.next()) {
            return Self { single_survivor: Some(only), layers: Vec::new() };
        }
        let mut levels: Vec<u64> = total_commit.to_vec();
        levels.sort_unstable();
        levels.dedup();
        let mut layers = Vec::with_capacity(levels.len());
        let mut prev_level = 0u64;
        for &level in &levels {
            if level == 0 {
                continue;
            }
            let contributors: u64 = total_commit.iter().filter(|&&c| c >= level).count() as u64;
            let layer_each = level - prev_level;
            let chips = layer_each * contributors;
            prev_level = level;
            if chips == 0 {
                continue;
            }
            let mut eligible = 0u32;
            for i in 0..n {
                if total_commit[i] >= level && !folded[i] {
                    eligible |= 1 << i;
                }
            }
            layers.push(PotLayer { level, layer_each, chips, eligible });
        }
        Self { single_survivor: None, layers }
    }
}

/// Distribute chips across seats via side-pot layers + double-board split.
///
/// Returns chips *won* per seat (not deltas — caller computes deltas vs
/// `total_commit`). The sum of the return values equals the sum of
/// `total_commit` (zero-sum by construction).
pub fn double_board_payout(
    hole_cards: &[Vec<Card>],
    folded: &[bool],
    total_commit: &[u64],
    board_a: &[Card; 5],
    board_b: &[Card; 5],
    button: usize,
) -> Vec<u64> {
    let n = hole_cards.len();
    assert_eq!(folded.len(), n);
    assert_eq!(total_commit.len(), n);
    let layers = PotLayers::new(folded, total_commit);
    let mut result = vec![0u64; n];
    double_board_payout_layers(
        &layers, hole_cards, folded, total_commit, board_a, board_b, button, &mut result,
    );
    result
}

/// [`double_board_payout`] over a prebuilt [`PotLayers`], writing chips won
/// per seat into `out` (overwritten). Each alive seat's hand is evaluated at
/// most ONCE per board — lazily, the first time a layer on that board has
/// chips to award — and every layer re-selects its winners from those ranks.
/// The original evaluated every eligible seat again for every layer (up to
/// n(n+1)/2 PLO evaluations per board for n distinct stacks); winners,
/// odd-chip order and totals are identical.
#[allow(clippy::too_many_arguments)]
pub fn double_board_payout_layers(
    layers: &PotLayers,
    hole_cards: &[Vec<Card>],
    folded: &[bool],
    total_commit: &[u64],
    board_a: &[Card; 5],
    board_b: &[Card; 5],
    button: usize,
    out: &mut [u64],
) {
    let n = hole_cards.len();
    out.fill(0);
    if let Some(only) = layers.single_survivor {
        award_single_survivor(out, only, total_commit);
        return;
    }
    let mut ranks_a = [0 as HandRank; MAX_LAYER_SEATS];
    let mut ranks_b = [0 as HandRank; MAX_LAYER_SEATS];
    let (mut have_a, mut have_b) = (false, false);
    for layer in &layers.layers {
        if layer.eligible == 0 {
            refund_orphan_layer(out, total_commit, layer.level, layer.layer_each);
            continue;
        }
        let half_a = layer.chips / 2;
        let half_b = layer.chips - half_a;
        if half_a > 0 {
            if !have_a {
                rank_alive(&mut ranks_a, hole_cards, folded, board_a, n);
                have_a = true;
            }
            award_half_ranked(out, layer.eligible, &ranks_a, half_a, button);
        }
        if half_b > 0 {
            if !have_b {
                rank_alive(&mut ranks_b, hole_cards, folded, board_b, n);
                have_b = true;
            }
            award_half_ranked(out, layer.eligible, &ranks_b, half_b, button);
        }
    }
}

fn rank_alive(
    ranks: &mut [HandRank; MAX_LAYER_SEATS],
    hole_cards: &[Vec<Card>],
    folded: &[bool],
    board: &[Card; 5],
    n: usize,
) {
    for i in 0..n {
        if !folded[i] {
            ranks[i] = evaluate_plo5(&hole_cards[i], board);
        }
    }
}

/// `award_half` over precomputed ranks: `half` chips to the best hand(s)
/// among the `eligible` seats (bitmask).
fn award_half_ranked(
    out: &mut [u64],
    eligible: u32,
    ranks: &[HandRank; MAX_LAYER_SEATS],
    half: u64,
    button: usize,
) {
    let mut best: Option<HandRank> = None;
    let mut bits = eligible;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        best = Some(best.map_or(ranks[i], |b| b.max(ranks[i])));
        bits &= bits - 1;
    }
    let best = best.expect("award_half_ranked: no eligible seat");
    let mut winners = 0u32;
    let mut bits = eligible;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        if ranks[i] == best {
            winners |= 1 << i;
        }
        bits &= bits - 1;
    }
    distribute_evenly_mask(out, winners, half, button);
}

/// [`distribute_evenly`] for a bitmask of recipients: even split, odd-chip
/// remainder clockwise from `button + 1`.
fn distribute_evenly_mask(result: &mut [u64], recipients: u32, amount: u64, button: usize) {
    if amount == 0 || recipients == 0 {
        return;
    }
    let count = recipients.count_ones() as u64;
    let share = amount / count;
    let remainder = amount - share * count;
    let mut bits = recipients;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        result[i] += share;
        bits &= bits - 1;
    }
    if remainder == 0 {
        return;
    }
    let n = result.len();
    let mut given = 0u64;
    for step in 1..=(2 * n) {
        let i = (button + step) % n;
        if recipients & (1 << i) != 0 {
            result[i] += 1;
            given += 1;
            if given >= remainder {
                return;
            }
        }
    }
}

/// Distribute chips across seats via side-pot layers on a single board
/// under NLH rules (best 5 of hole + board, any combination). Same layer
/// construction as [`double_board_payout`], but each layer's chips go
/// entirely to the best hand(s) on the one board.
///
/// Returns chips *won* per seat; sum equals the sum of `total_commit`.
pub fn single_board_payout(
    hole_cards: &[Vec<Card>],
    folded: &[bool],
    total_commit: &[u64],
    board: &[Card; 5],
    button: usize,
) -> Vec<u64> {
    let n = hole_cards.len();
    assert_eq!(folded.len(), n);
    assert_eq!(total_commit.len(), n);
    let mut result = vec![0u64; n];

    // Fold-out: single survivor takes every matched chip (no showdown).
    let alive: Vec<usize> = (0..n).filter(|&i| !folded[i]).collect();
    if alive.len() == 1 {
        award_single_survivor(&mut result, alive[0], total_commit);
        return result;
    }

    // Rank every alive seat once; layers only re-select among eligible.
    let ranks: Vec<Option<HandRank>> = (0..n)
        .map(|i| {
            if folded[i] {
                None
            } else {
                Some(evaluate_nlh(&hole_cards[i], board))
            }
        })
        .collect();

    let mut levels: Vec<u64> = total_commit.iter().copied().collect();
    levels.sort_unstable();
    levels.dedup();

    let mut prev_level = 0u64;
    for &level in &levels {
        if level == 0 {
            continue;
        }
        let contributors: u64 = total_commit.iter().filter(|&&c| c >= level).count() as u64;
        let layer_each = level - prev_level;
        let layer_chips = layer_each * contributors;
        prev_level = level;
        if layer_chips == 0 {
            continue;
        }

        let eligible: Vec<usize> = (0..n)
            .filter(|&i| total_commit[i] >= level && !folded[i])
            .collect();

        if eligible.is_empty() {
            refund_orphan_layer(&mut result, total_commit, level, layer_each);
            continue;
        }

        let best = eligible
            .iter()
            .map(|&s| ranks[s].expect("eligible seat must have a rank"))
            .max()
            .unwrap();
        let winners: Vec<usize> = eligible
            .iter()
            .copied()
            .filter(|&s| ranks[s] == Some(best))
            .collect();
        distribute_evenly(&mut result, &winners, layer_chips, button);
    }

    result
}

/// Fold-out settlement: the lone survivor collects, from every seat, at
/// most its OWN total commit; whatever a folded seat put in above that
/// was never matched by anyone still in the hand and goes back to it.
///
/// Defence in depth (review 2026-09-20 C1). On every line the engine can
/// reach the survivor holds the largest commit — a seat only folds to a
/// bigger live bet — so this is the old "survivor takes the sum". The one
/// way around that was NLH's nominal `bet_to_call` (a covering SB folding
/// to a short all-in BB's phantom bet, paid `[-10000, +10000]` with only
/// 8000 ever matched), which the engine's actor walk no longer offers.
fn award_single_survivor(result: &mut [u64], survivor: usize, total_commit: &[u64]) {
    let cap = total_commit[survivor];
    for (i, &commit) in total_commit.iter().enumerate() {
        let matched = commit.min(cap);
        result[survivor] += matched;
        result[i] += commit - matched;
    }
}

/// Orphaned layer: every seat that paid into it has folded, so no alive
/// seat ever matched these chips — each contributor gets its `layer_each`
/// slice back. It used to be split across the alive seats, which let a
/// showdown LOSER collect chips nobody had called (NLH 3-way
/// [51, 10000, 51]: the folded SB's uncalled 49). Defence in depth, same
/// as [`award_single_survivor`]: unreachable on real lines, where the top
/// commit always belongs to an alive seat. (review 2026-09-20 C1)
fn refund_orphan_layer(result: &mut [u64], total_commit: &[u64], level: u64, layer_each: u64) {
    for (i, &commit) in total_commit.iter().enumerate() {
        if commit >= level {
            result[i] += layer_each;
        }
    }
}

/// Award `half` chips on one board to the best hand(s) among `eligible`
/// (the original per-layer evaluation; kept for the equivalence test).
#[cfg(test)]
fn award_half(
    result: &mut [u64],
    eligible: &[usize],
    hole_cards: &[Vec<Card>],
    board: &[Card; 5],
    half: u64,
    button: usize,
) {
    if half == 0 {
        return;
    }
    let ranks: Vec<HandRank> = eligible
        .iter()
        .map(|&s| evaluate_plo5(&hole_cards[s], board))
        .collect();
    let best = *ranks.iter().max().unwrap();
    let winners: Vec<usize> = eligible
        .iter()
        .zip(ranks.iter())
        .filter(|&(_, &r)| r == best)
        .map(|(&s, _)| s)
        .collect();
    distribute_evenly(result, &winners, half, button);
}

/// Split `amount` evenly across `recipients`; award odd-chip remainder to
/// recipients clockwise from `button + 1`.
fn distribute_evenly(result: &mut [u64], recipients: &[usize], amount: u64, button: usize) {
    if amount == 0 || recipients.is_empty() {
        return;
    }
    let count = recipients.len() as u64;
    let share = amount / count;
    let remainder = amount - share * count;
    for &r in recipients {
        result[r] += share;
    }
    if remainder == 0 {
        return;
    }
    // Walk clockwise from button+1; first `remainder` recipients get +1.
    let n = result.len();
    let mut given = 0u64;
    for step in 1..=(2 * n) {
        let i = (button + step) % n;
        if recipients.contains(&i) {
            result[i] += 1;
            given += 1;
            if given >= remainder {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::Card;

    /// The pre-2026-09-23 implementation, verbatim: per-call layer build,
    /// every eligible seat re-evaluated per layer. Reference for the
    /// equivalence test below.
    fn double_board_payout_reference(
        hole_cards: &[Vec<Card>],
        folded: &[bool],
        total_commit: &[u64],
        board_a: &[Card; 5],
        board_b: &[Card; 5],
        button: usize,
    ) -> Vec<u64> {
        let n = hole_cards.len();
        let mut result = vec![0u64; n];
        let alive: Vec<usize> = (0..n).filter(|&i| !folded[i]).collect();
        if alive.len() == 1 {
            award_single_survivor(&mut result, alive[0], total_commit);
            return result;
        }
        let mut levels: Vec<u64> = total_commit.iter().copied().collect();
        levels.sort_unstable();
        levels.dedup();
        let mut prev_level = 0u64;
        for &level in &levels {
            if level == 0 {
                continue;
            }
            let contributors: u64 = total_commit.iter().filter(|&&c| c >= level).count() as u64;
            let layer_each = level - prev_level;
            let layer_chips = layer_each * contributors;
            prev_level = level;
            if layer_chips == 0 {
                continue;
            }
            let eligible: Vec<usize> = (0..n)
                .filter(|&i| total_commit[i] >= level && !folded[i])
                .collect();
            if eligible.is_empty() {
                refund_orphan_layer(&mut result, total_commit, level, layer_each);
                continue;
            }
            let half_a = layer_chips / 2;
            let half_b = layer_chips - half_a;
            award_half(&mut result, &eligible, hole_cards, board_a, half_a, button);
            award_half(&mut result, &eligible, hole_cards, board_b, half_b, button);
        }
        result
    }

    #[test]
    fn rank_once_layers_match_the_reference_on_random_hands() {
        // Deterministic xorshift so the sweep is reproducible without rand.
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move |m: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % m
        };
        for case in 0..20_000 {
            let n = 2 + next(5) as usize; // 2..=6 seats
            let hole_w = 4 + next(3) as usize; // PLO4/5/6
            let mut deck: Vec<u8> = (0..52).collect();
            for i in (1..52).rev() {
                let j = next(i as u64 + 1) as usize;
                deck.swap(i, j);
            }
            let mut it = deck.into_iter().map(Card::from_index);
            let hole: Vec<Vec<Card>> = (0..n).map(|_| (&mut it).take(hole_w).collect()).collect();
            let mut board_a = [Card(0); 5];
            let mut board_b = [Card(0); 5];
            for k in 0..5 {
                board_a[k] = it.next().unwrap();
                board_b[k] = it.next().unwrap();
            }
            let mut folded: Vec<bool> = (0..n).map(|_| next(4) == 0).collect();
            if case % 7 == 0 {
                // fold-outs and single survivors too
                for f in folded.iter_mut() {
                    *f = true;
                }
                folded[next(n as u64) as usize] = false;
            }
            // few distinct levels, some shared, odd sizes, the odd zero
            let commit: Vec<u64> = (0..n)
                .map(|_| if next(10) == 0 { 0 } else { 1 + next(6) * 997 + next(3) })
                .collect();
            let button = next(n as u64) as usize;
            let want = double_board_payout_reference(&hole, &folded, &commit, &board_a, &board_b, button);
            let got = double_board_payout(&hole, &folded, &commit, &board_a, &board_b, button);
            assert_eq!(got, want, "case {case}: folded {folded:?} commit {commit:?} button {button}");
        }
    }

    fn c(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    /// Test fixtures use fixed-size arrays; the payout API takes
    /// variant-sized Vecs.
    fn holes(arr: &[[Card; 5]]) -> Vec<Vec<Card>> {
        arr.iter().map(|h| h.to_vec()).collect()
    }

    #[test]
    fn fold_out_single_survivor() {
        // 3 seats, seat 0 alive, others folded. Commits 500/500/500.
        let hole = vec![[c(0, 0); 5]; 3];
        let folded = vec![false, true, true];
        let commit = vec![500u64, 500, 500];
        let board = [c(0, 0); 5];
        let result = double_board_payout(&holes(&hole), &folded, &commit, &board, &board, 0);
        assert_eq!(result, vec![1500u64, 0, 0]);
    }

    #[test]
    fn scoop_both_boards() {
        // Seat 0: four aces + one low → always trips AAA minimum.
        // Seat 1: low cards → pair at best.
        let hole = vec![
            [c(12, 0), c(12, 1), c(12, 2), c(12, 3), c(0, 0)], // AAAA2c
            [c(1, 0), c(1, 1), c(2, 2), c(2, 3), c(3, 0)],     // 33445
        ];
        let folded = vec![false, false];
        let commit = vec![1000u64, 1000];
        let board_a = [c(11, 0), c(10, 1), c(8, 2), c(6, 3), c(4, 0)]; // K Q T 8 6
        let board_b = [c(9, 0), c(7, 1), c(5, 2), c(4, 3), c(0, 1)]; // J 9 7 6 2
        let result = double_board_payout(&holes(&hole), &folded, &commit, &board_a, &board_b, 0);
        assert_eq!(result, vec![2000u64, 0]);
    }

    #[test]
    fn split_different_winner_per_board() {
        // Seat 0 wins board A (AA beats KK on unpaired board).
        // Seat 1 wins board B (KK makes quads vs seat 0's two pair).
        let hole = vec![
            [c(12, 0), c(12, 1), c(0, 2), c(1, 3), c(2, 0)], // AAc AAd + junk
            [c(11, 2), c(11, 3), c(0, 1), c(1, 0), c(2, 1)], // KKh KKs + junk
        ];
        let folded = vec![false, false];
        let commit = vec![1000u64, 1000];
        // Board A: unpaired low, no flush/straight — AA beats KK pair.
        let board_a = [c(6, 0), c(5, 1), c(3, 2), c(1, 0), c(0, 3)]; // 8 7 5 3 2
        // Board B: Kh Ks + junk → seat 1 makes quad kings.
        let board_b = [c(11, 0), c(11, 1), c(5, 2), c(3, 3), c(0, 3)]; // KK 7 5 2
        let result = double_board_payout(&holes(&hole), &folded, &commit, &board_a, &board_b, 0);
        assert_eq!(result, vec![1000u64, 1000]);
    }

    #[test]
    fn chop_both_boards_mirrored_hands() {
        // Two seats with symmetric hands (same ranks, different suits).
        // Boards are rainbow with no flush potential → perfect tie per board.
        let hole = vec![
            [c(12, 0), c(12, 1), c(0, 0), c(1, 1), c(2, 2)],
            [c(12, 2), c(12, 3), c(0, 3), c(1, 2), c(2, 1)],
        ];
        let folded = vec![false, false];
        let commit = vec![1000u64, 1000];
        let board_a = [c(11, 0), c(8, 1), c(5, 2), c(3, 3), c(0, 3)]; // K T 7 5 2 rainbow
        let board_b = [c(10, 1), c(7, 2), c(4, 3), c(2, 0), c(1, 3)]; // Q 9 6 4 3 rainbow
        let result = double_board_payout(&holes(&hole), &folded, &commit, &board_a, &board_b, 0);
        assert_eq!(result, vec![1000u64, 1000], "perfect chop");
    }

    #[test]
    fn chop_one_board_clean_win_other() {
        // Mirrored hands split board A (AA + kickers tie).
        // Board B crafted so seat 0's clubs + board clubs make Q-high flush,
        // seat 1 has no flush.
        let hole = vec![
            // Seat 0 has 2 clubs (Jc Qc) + AcAd + junk.
            [c(12, 0), c(12, 1), c(10, 0), c(9, 0), c(0, 3)], // Ac Ad Qc Jc 2s
            // Seat 1 has AsAh + mixed.
            [c(12, 2), c(12, 3), c(10, 2), c(9, 3), c(0, 1)], // Ah As Qh Js 2d
        ];
        let folded = vec![false, false];
        let commit = vec![1000u64, 1000];
        // Board A: rainbow no flush; AA + K + T + 7 kickers same for both.
        let board_a = [c(11, 1), c(8, 2), c(5, 3), c(3, 0), c(1, 3)]; // Kd Th 7s 5c 3s
        // Board B: 3 clubs present → seat 0 uses Jc+Qc + board clubs → Q-high flush.
        let board_b = [c(7, 0), c(5, 0), c(3, 0), c(4, 1), c(0, 2)]; // 9c 7c 5c 6d 2h
        let result = double_board_payout(&holes(&hole), &folded, &commit, &board_a, &board_b, 0);
        // Board A chops (500 each). Board B to seat 0 (1000).
        // Seat 0: 500 + 1000 = 1500. Seat 1: 500.
        assert_eq!(result, vec![1500u64, 500]);
    }

    #[test]
    fn side_pot_three_seats_unequal_commit() {
        // Commits [500, 500, 1500] — seat 2 is deepest. None folded.
        // Both boards are high-card with a Q pair: every seat makes "two pair
        // with QQ + Jack kicker" using hole pair + board. Seat 0's AA-QQ-J
        // beats seat 2's KK-QQ-J beats seat 1's QQ-88-J. Seat 0 scoops main
        // pot on both boards. Seat 2 alone in the side pot.
        let hole = vec![
            [c(12, 0), c(12, 1), c(12, 2), c(12, 3), c(0, 0)], // AAAA 2c
            [c(6, 1), c(6, 2), c(0, 2), c(1, 1), c(2, 3)],     // 8d 8h 2h 3d 4s
            [c(11, 0), c(11, 1), c(11, 2), c(11, 3), c(3, 0)], // KKKK 5c
        ];
        let folded = vec![false, false, false];
        let commit = vec![500u64, 500, 1500];
        let board_a = [c(10, 0), c(10, 1), c(9, 2), c(8, 2), c(7, 3)]; // Qc Qd Jh Th 9s
        let board_b = [c(10, 2), c(10, 3), c(9, 1), c(8, 0), c(7, 0)]; // Qh Qs Jd Tc 9c
        let result = double_board_payout(&holes(&hole), &folded, &commit, &board_a, &board_b, 0);
        // Layer 1 (main, 500×3=1500): seat 0 scoops → 1500.
        // Layer 2 (side, 1000×1=1000): seat 2 alone → 1000.
        assert_eq!(result, vec![1500u64, 0, 1000]);
        assert_eq!(result.iter().sum::<u64>(), commit.iter().sum::<u64>());
    }

    #[test]
    fn side_pot_four_seats_unequal_commit_scoop() {
        // 4 seats, commits [200, 500, 1000, 2000] — 4 distinct side-pot
        // layers. Seat 3 (deepest) holds quad aces + 2h; seat 2 holds quad
        // kings + 2c; seats 0, 1 are unpaired junk with no straight/flush
        // potential against the boards. Both boards are QQ-paired with
        // safe kickers so seat 3's AA + QQ two pair wins every layer on
        // every board. Seat 3 collects every contested chip (layers 1-3)
        // plus a 1000-chip solo refund on layer 4.
        //
        //   Layer 1 (200 × 4 = 800)  : all eligible,   seat 3 scoops
        //   Layer 2 (300 × 3 = 900)  : seats 1,2,3,    seat 3 scoops
        //   Layer 3 (500 × 2 = 1000) : seats 2,3,      seat 3 scoops
        //   Layer 4 (1000 × 1 = 1000): seat 3 solo,    refund
        //   total seat 3 = 3700; others = 0.
        let hole = vec![
            // Seat 0: 3c 4d 5h 7s 9c — unpaired, no 5-run with boards.
            [c(1, 0), c(2, 1), c(3, 2), c(5, 3), c(7, 0)],
            // Seat 1: 2d 6h 8s Tc Jd — unpaired, no 5-run with boards.
            [c(0, 1), c(4, 2), c(6, 3), c(8, 0), c(9, 1)],
            // Seat 2: KKKK + 2c.
            [c(11, 0), c(11, 1), c(11, 2), c(11, 3), c(0, 0)],
            // Seat 3: AAAA + 2h.
            [c(12, 0), c(12, 1), c(12, 2), c(12, 3), c(0, 2)],
        ];
        let folded = vec![false, false, false, false];
        let commit = vec![200u64, 500, 1000, 2000];
        // Board A: Qc Qd 7c 9d 2s.
        let board_a = [c(10, 0), c(10, 1), c(5, 0), c(7, 1), c(0, 3)];
        // Board B: Qh Qs 8d 4c 3s.
        let board_b = [c(10, 2), c(10, 3), c(6, 1), c(2, 0), c(1, 3)];
        let result = double_board_payout(&holes(&hole), &folded, &commit, &board_a, &board_b, 0);
        assert_eq!(result, vec![0u64, 0, 0, 3700]);
        assert_eq!(result.iter().sum::<u64>(), commit.iter().sum::<u64>());
    }

    #[test]
    fn side_pot_plus_split_boards() {
        // Commits [500, 500, 1500]. Main pot splits: seat 0 wins board A,
        // seat 1 wins board B (trips 8s beats AA pair). Seat 2 alone in side pot.
        let hole = vec![
            [c(12, 0), c(12, 1), c(12, 2), c(12, 3), c(0, 0)], // AAAA 2c
            [c(6, 1), c(6, 2), c(0, 2), c(1, 1), c(2, 3)],     // 8d 8h 2h 3d 4s
            [c(11, 0), c(11, 1), c(11, 2), c(11, 3), c(3, 0)], // KKKK 5c
        ];
        let folded = vec![false, false, false];
        let commit = vec![500u64, 500, 1500];
        // Board A: Qc Qd Jh Th 9s → seat 0 wins with AA-QQ-J two pair.
        let board_a = [c(10, 0), c(10, 1), c(9, 2), c(8, 2), c(7, 3)];
        // Board B: 8c 7h 5d 2s 3c → seat 1 makes trips 8s (beats AA pair).
        let board_b = [c(6, 0), c(5, 2), c(3, 1), c(0, 3), c(1, 0)];
        let result = double_board_payout(&holes(&hole), &folded, &commit, &board_a, &board_b, 0);
        // Main pot = 1500, split across boards → 750 each.
        //   Board A: seat 0 wins 750.
        //   Board B: seat 1 wins 750.
        // Side pot = 1000, seat 2 alone across both boards → 1000.
        assert_eq!(result, vec![750u64, 750, 1000]);
        assert_eq!(result.iter().sum::<u64>(), commit.iter().sum::<u64>());
    }

    // ---- NLH single-board payout ----

    #[test]
    fn nlh_payout_one_hole_card_flush_beats_pocket_aces() {
        // Any-combo rule: seat 0 makes an A-high flush with ONE hole heart
        // (impossible under PLO's exactly-2 rule); seat 1's pocket aces
        // make only a pair on this board.
        let hole = vec![
            vec![c(12, 2), c(0, 0)],  // Ah 2c
            vec![c(12, 3), c(12, 1)], // As Ad
        ];
        let folded = vec![false, false];
        let commit = vec![1_000u64, 1_000];
        let board = [c(11, 2), c(10, 2), c(9, 2), c(7, 2), c(1, 3)]; // Kh Qh Jh 9h 3s
        let result = single_board_payout(&hole, &folded, &commit, &board, 0);
        assert_eq!(result, vec![2_000u64, 0]);
    }

    #[test]
    fn nlh_payout_play_the_board_chops() {
        // Board is a broadway straight; neither hole improves → both seats
        // play the board (ZERO hole cards) and chop.
        let hole = vec![
            vec![c(0, 0), c(1, 1)], // 2c 3d
            vec![c(0, 2), c(1, 3)], // 2h 3s
        ];
        let folded = vec![false, false];
        let commit = vec![1_000u64, 1_000];
        let board = [c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(8, 0)]; // A K Q J T rainbow
        let result = single_board_payout(&hole, &folded, &commit, &board, 0);
        assert_eq!(result, vec![1_000u64, 1_000]);
    }

    #[test]
    fn nlh_payout_side_pot_layers() {
        // Commits [500, 1000, 2000]. Seat 0 (short) has trips queens and
        // wins the main pot only; seat 2 (pair of queens, ace kicker)
        // beats seat 1 (unpaired) for the middle layer and collects the
        // uncalled top layer as a refund.
        let hole = vec![
            vec![c(10, 0), c(10, 1)], // Qc Qd → trips with board Qs
            vec![c(6, 0), c(1, 1)],   // 8c 3d → high card
            vec![c(12, 3), c(10, 2)], // As Qh → pair of queens, A kicker
        ];
        let folded = vec![false, false, false];
        let commit = vec![500u64, 1_000, 2_000];
        let board = [c(10, 3), c(5, 1), c(0, 2), c(7, 0), c(2, 3)]; // Qs 7d 2h 9c 4s
        let result = single_board_payout(&hole, &folded, &commit, &board, 0);
        // Layer 1 (500 × 3 = 1500): seat 0. Layer 2 (500 × 2 = 1000):
        // seat 2. Layer 3 (1000 × 1): seat 2 refund.
        assert_eq!(result, vec![1_500u64, 0, 2_000]);
        assert_eq!(result.iter().sum::<u64>(), commit.iter().sum::<u64>());
    }

    #[test]
    fn nlh_payout_fold_out_short_circuits() {
        let hole = vec![vec![c(0, 0), c(1, 0)], vec![c(2, 0), c(3, 0)]];
        let folded = vec![false, true];
        let commit = vec![700u64, 300];
        let board = [c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(8, 0)];
        let result = single_board_payout(&hole, &folded, &commit, &board, 0);
        assert_eq!(result, vec![1_000u64, 0]);
    }

    // ---- Unmatched chips (review 2026-09-20 C1, defence in depth) ----
    //
    // Neither state is reachable through the engine any more (the actor
    // walk no longer offers a fold to NLH's nominal bet), so they are
    // pinned here at the function level.

    #[test]
    fn fold_out_survivor_cannot_win_unmatched_chips() {
        // Review scenario A books: the covering SB (10000 in) "folded" to
        // the short all-in BB (8000 in). Only 8000 was ever matched — the
        // SB's other 2000 goes back to it. Used to pay [0, 18000].
        let folded = vec![true, false];
        let commit = vec![10_000u64, 8_000];
        let nlh_hole = vec![vec![c(0, 0), c(1, 0)], vec![c(2, 0), c(3, 0)]];
        let board = [c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(8, 0)];
        assert_eq!(
            single_board_payout(&nlh_hole, &folded, &commit, &board, 0),
            vec![2_000u64, 16_000]
        );
        let plo_hole = vec![[c(0, 0); 5]; 2];
        assert_eq!(
            double_board_payout(&holes(&plo_hole), &folded, &commit, &board, &board, 0),
            vec![2_000u64, 16_000]
        );
        // Multiway: every folded seat is capped at the survivor's commit.
        let folded = vec![true, false, true, true];
        let commit = vec![900u64, 400, 250, 400];
        let plo_hole = vec![[c(0, 0); 5]; 4];
        assert_eq!(
            double_board_payout(&holes(&plo_hole), &folded, &commit, &board, &board, 2),
            vec![500u64, 400 + 400 + 250 + 400, 0, 0]
        );
    }

    #[test]
    fn orphan_layer_is_refunded_to_its_contributors() {
        // Review scenario B books: commits [51, 100, 51], the deep SB
        // (seat 1) folded, seats 0 and 2 show down. Its top 49 chips were
        // matched by nobody — they used to be split 25/24 between the
        // alive seats, handing the showdown LOSER chips. Now: back to
        // seat 1, and only the 153 matched chips are contested.
        let folded = vec![false, true, false];
        let commit = vec![51u64, 100, 51];
        let hole = vec![
            vec![c(12, 0), c(12, 1)], // AA — wins
            vec![c(5, 0), c(3, 1)],
            vec![c(0, 0), c(1, 1)], // 32o — loses
        ];
        let board = [c(11, 2), c(9, 3), c(7, 2), c(4, 3), c(2, 1)]; // K J 9 6 4
        let result = single_board_payout(&hole, &folded, &commit, &board, 0);
        assert_eq!(result, vec![153u64, 49, 0]);

        let hole5 = vec![
            [c(12, 0), c(12, 1), c(12, 2), c(12, 3), c(0, 0)], // AAAA2
            [c(6, 1), c(6, 2), c(0, 2), c(1, 1), c(2, 3)],
            [c(1, 0), c(1, 2), c(2, 2), c(2, 1), c(3, 0)], // 33445
        ];
        let board_a = [c(11, 0), c(10, 1), c(8, 2), c(6, 3), c(4, 0)]; // K Q T 8 6
        let board_b = [c(9, 0), c(7, 1), c(5, 2), c(4, 3), c(0, 1)]; // J 9 7 6 2
        let result = double_board_payout(&holes(&hole5), &folded, &commit, &board_a, &board_b, 0);
        assert_eq!(result, vec![153u64, 49, 0]);
        assert_eq!(result.iter().sum::<u64>(), commit.iter().sum::<u64>());
    }
}
