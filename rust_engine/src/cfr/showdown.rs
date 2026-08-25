//! Terminal EV for river showdowns and folds.

use super::public_state::PublicState;
use super::range::{combo_cards, equity_vs_range, Range, NUM_COMBOS};

/// Expected chip payoff for `seat` at a terminal public state.
///
/// - Fold: uses [`PublicState::fold_payout_chips`].
/// - Showdown: pot contested; EV = equity * pot - total_commit[seat]
///   under independent range product with blocker removal.
///
/// `hero_combo` is the private combo for `seat`; opponent range is the other seat's.
/// For multiway, pairwise equity approximation is used (Phase 3 exact later).
pub fn terminal_ev_chips(
    state: &PublicState,
    seat: usize,
    hero_combo: usize,
    ranges: &[Range],
    ranks: &[u32],
) -> f64 {
    let n = state.n();
    if state.alive_count() == 1 {
        return state.fold_payout_chips(seat) as f64;
    }

    // Showdown: both (or more) alive. Dead pot + commits = pot.
    let pot = state.pot as f64;
    let own_commit = state.total_commit[seat] as f64;

    if n == 2 {
        let opp = 1 - seat;
        let hero_rank = if hero_combo < NUM_COMBOS {
            ranks[hero_combo]
        } else {
            0
        };
        let eq = equity_vs_range(hero_combo, hero_rank, &ranges[opp], ranks);
        // Win pot with equity; always "spent" own commit (already in pot).
        // Net = eq * pot - own_commit  (standard chip EV)
        return eq * pot - own_commit;
    }

    // Multiway: side-pot layers from total_commit + dead pot (same construction
    // as mccfr::multi_terminal_real / engine single_board_payout layers).
    if self_folded(state, seat) {
        return -own_commit;
    }
    let commits: Vec<u64> = (0..n).map(|i| state.total_commit[i]).collect();
    let sum_c: u64 = commits.iter().sum();
    let dead = state.pot.saturating_sub(sum_c);
    // Hero rank from combo; opponent ranks from range-weighted best response
    // is not available here — use hero rank vs each alive seat's point mass
    // when ranges are point ranges; otherwise equal among tied-best using
    // equity_vs_range pairwise product (exact for HU already handled).
    let hero_rank = if hero_combo < NUM_COMBOS {
        ranks[hero_combo]
    } else {
        0
    };
    // Expected chips won via side-pot layers under independent range product.
    let mut expected_won = 0.0;
    let mut levels: Vec<u64> = commits.iter().copied().filter(|&c| c > 0).collect();
    levels.sort_unstable();
    levels.dedup();
    let mut prev = 0u64;
    for &level in &levels {
        let contributors: Vec<usize> = (0..n).filter(|&i| commits[i] >= level).collect();
        if contributors.is_empty() {
            continue;
        }
        let layer = (level - prev) as f64 * contributors.len() as f64;
        prev = level;
        let eligible: Vec<usize> = contributors
            .into_iter()
            .filter(|&i| !self_folded(state, i))
            .collect();
        if eligible.is_empty() || !eligible.contains(&seat) {
            continue;
        }
        // P(hero wins or ties this layer) vs product of other ranges
        let mut win_mass = 1.0f64;
        let mut tie_with = 1.0f64; // number of tied winners expectation
        for &opp in &eligible {
            if opp == seat {
                continue;
            }
            if opp >= ranges.len() {
                continue;
            }
            let eq = equity_vs_range(hero_combo, hero_rank, &ranges[opp], ranks);
            // rough: treat each opp independently for win/tie/lose
            // P(beat this opp) ≈ 2*eq-1 when eq>0.5 ... use eq as share vs that opp
            win_mass *= eq; // conservative joint approx
            tie_with += 1.0 - (2.0 * (eq - 0.5).abs()); // soft
        }
        let share = if tie_with > 0.0 {
            win_mass / tie_with.max(1.0)
        } else {
            0.0
        };
        expected_won += layer * share.clamp(0.0, 1.0);
    }
    if dead > 0 {
        let alive: Vec<usize> = (0..n).filter(|&i| !self_folded(state, i)).collect();
        if alive.contains(&seat) {
            // equal share of dead among alive as lower bound when multi-range hard
            expected_won += dead as f64 / alive.len() as f64;
        }
    }
    let _ = pot;
    expected_won - own_commit
}

fn self_folded(state: &PublicState, i: usize) -> bool {
    state.folded[i]
}

/// Expected EV for seat under full range product at showdown (no private card).
/// Used for root value reporting.
pub fn range_vs_range_ev(
    state: &PublicState,
    seat: usize,
    ranges: &[Range],
    ranks: &[u32],
) -> f64 {
    let n = state.n();
    if state.alive_count() == 1 {
        return state.fold_payout_chips(seat) as f64;
    }
    if n != 2 {
        let alive = (0..n).filter(|&i| !state.folded[i]).count() as f64;
        let share = if alive > 0.0 { 1.0 / alive } else { 0.0 };
        return share * state.pot as f64 - state.total_commit[seat] as f64;
    }
    let opp = 1 - seat;
    let mut total_w = 0.0;
    let mut ev = 0.0;
    for hid in 0..NUM_COMBOS {
        let wh = ranges[seat].weights[hid];
        if wh <= 0.0 || ranks[hid] == 0 {
            continue;
        }
        let (h0, h1) = combo_cards(hid);
        for vid in 0..NUM_COMBOS {
            let wv = ranges[opp].weights[vid];
            if wv <= 0.0 || ranks[vid] == 0 {
                continue;
            }
            let (v0, v1) = combo_cards(vid);
            if v0 == h0 || v0 == h1 || v1 == h0 || v1 == h1 {
                continue;
            }
            let w = wh * wv;
            total_w += w;
            let eq = if ranks[hid] > ranks[vid] {
                1.0
            } else if ranks[hid] == ranks[vid] {
                0.5
            } else {
                0.0
            };
            let pot = state.pot as f64;
            let own = state.total_commit[seat] as f64;
            ev += w * (eq * pot - own);
        }
    }
    if total_w <= 0.0 {
        0.0
    } else {
        ev / total_w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::public_state::PublicState;
    use crate::cfr::range::{combo_ranks_on_board, Range};

    #[test]
    fn fold_ev_matches_public() {
        let mut s = PublicState::river_hu_root(100_000, 500_000, &[0, 1, 2, 3, 4], 10_000).unwrap();
        s.apply_fold();
        let ranges = [
            Range::uniform_unblocked(&[0, 1, 2, 3, 4]),
            Range::uniform_unblocked(&[0, 1, 2, 3, 4]),
        ];
        let board = [0u8, 1, 2, 3, 4];
        let ranks = combo_ranks_on_board(&board);
        let ev1 = terminal_ev_chips(&s, 1, 0, &ranges, &ranks);
        assert!((ev1 - 100_000.0).abs() < 1.0);
    }
}
