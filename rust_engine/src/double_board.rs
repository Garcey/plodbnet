//! Layered side-pot payout distribution across two boards.
//!
//! Pot layers are built from unique sorted values in `total_commit`. For each
//! layer, eligible seats (`total_commit >= level && !folded`) contest that
//! layer's chips, with half going to the best PLO5 hand on each board. Ties
//! on a board half split that half; odd-chip remainders are awarded to the
//! first tied winner encountered clockwise from button.

use crate::cards::Card;
use crate::hand_eval::{evaluate_plo5, HandRank};

/// Distribute chips across seats via side-pot layers + double-board split.
///
/// Returns chips *won* per seat (not deltas — caller computes deltas vs
/// `total_commit`). The sum of the return values equals the sum of
/// `total_commit` (zero-sum by construction).
pub fn double_board_payout(
    hole_cards: &[[Card; 5]],
    folded: &[bool],
    total_commit: &[u64],
    board_a: &[Card; 5],
    board_b: &[Card; 5],
    button: usize,
) -> Vec<u64> {
    let n = hole_cards.len();
    assert_eq!(folded.len(), n);
    assert_eq!(total_commit.len(), n);
    let mut result = vec![0u64; n];

    // Fold-out: single survivor takes all chips (no showdown).
    let alive: Vec<usize> = (0..n).filter(|&i| !folded[i]).collect();
    if alive.len() == 1 {
        result[alive[0]] = total_commit.iter().sum();
        return result;
    }

    // Sorted unique commitment levels, ascending.
    let mut levels: Vec<u64> = total_commit.iter().copied().collect();
    levels.sort_unstable();
    levels.dedup();

    let mut prev_level = 0u64;
    for &level in &levels {
        if level == 0 {
            continue;
        }
        let contributors: u64 = total_commit.iter().filter(|&&c| c >= level).count() as u64;
        let layer_chips = (level - prev_level) * contributors;
        prev_level = level;
        if layer_chips == 0 {
            continue;
        }

        let eligible: Vec<usize> = (0..n)
            .filter(|&i| total_commit[i] >= level && !folded[i])
            .collect();

        if eligible.is_empty() {
            // Orphaned layer (shouldn't occur with ≥2 survivors at showdown).
            // Distribute evenly across alive seats as a safe fallback.
            distribute_evenly(&mut result, &alive, layer_chips, button);
            continue;
        }

        let half_a = layer_chips / 2;
        let half_b = layer_chips - half_a;
        award_half(&mut result, &eligible, hole_cards, board_a, half_a, button);
        award_half(&mut result, &eligible, hole_cards, board_b, half_b, button);
    }

    result
}

/// Award `half` chips on one board to the best hand(s) among `eligible`.
fn award_half(
    result: &mut [u64],
    eligible: &[usize],
    hole_cards: &[[Card; 5]],
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

    fn c(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn fold_out_single_survivor() {
        // 3 seats, seat 0 alive, others folded. Commits 500/500/500.
        let hole = vec![[c(0, 0); 5]; 3];
        let folded = vec![false, true, true];
        let commit = vec![500u64, 500, 500];
        let board = [c(0, 0); 5];
        let result = double_board_payout(&hole, &folded, &commit, &board, &board, 0);
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
        let result = double_board_payout(&hole, &folded, &commit, &board_a, &board_b, 0);
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
        let result = double_board_payout(&hole, &folded, &commit, &board_a, &board_b, 0);
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
        let result = double_board_payout(&hole, &folded, &commit, &board_a, &board_b, 0);
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
        let result = double_board_payout(&hole, &folded, &commit, &board_a, &board_b, 0);
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
        let result = double_board_payout(&hole, &folded, &commit, &board_a, &board_b, 0);
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
        let result = double_board_payout(&hole, &folded, &commit, &board_a, &board_b, 0);
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
        let result = double_board_payout(&hole, &folded, &commit, &board_a, &board_b, 0);
        // Main pot = 1500, split across boards → 750 each.
        //   Board A: seat 0 wins 750.
        //   Board B: seat 1 wins 750.
        // Side pot = 1000, seat 2 alone across both boards → 1000.
        assert_eq!(result, vec![750u64, 750, 1000]);
        assert_eq!(result.iter().sum::<u64>(), commit.iter().sum::<u64>());
    }
}
