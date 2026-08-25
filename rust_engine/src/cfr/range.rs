//! HU range over 1326 hole combos (C(52,2)).

use crate::cards::Card;
use crate::hand_eval::evaluate_nlh;

/// Number of unordered 2-card combos from a 52-card deck.
pub const NUM_COMBOS: usize = 1326; // C(52,2)

/// Map combo id 0..1325 → (c0, c1) with c0 < c1.
#[inline]
pub fn combo_cards(id: usize) -> (u8, u8) {
    debug_assert!(id < NUM_COMBOS);
    // Inverse of triangular: find c1 such that C(c1,2) <= id < C(c1+1,2)
    // c1 from 1..51, base = c1*(c1-1)/2
    let mut c1 = 1u8;
    while {
        let next = (c1 as usize + 1) * (c1 as usize) / 2;
        id >= next && c1 < 51
    } {
        c1 += 1;
    }
    let base = (c1 as usize) * (c1 as usize - 1) / 2;
    let c0 = (id - base) as u8;
    (c0, c1)
}

/// Map (c0, c1) with c0 != c1 → combo id.
#[inline]
pub fn cards_to_combo(a: u8, b: u8) -> usize {
    let (c0, c1) = if a < b { (a, b) } else { (b, a) };
    let c1u = c1 as usize;
    c1u * (c1u - 1) / 2 + c0 as usize
}

/// Reach weights over combos (non-negative). Not necessarily normalized.
#[derive(Debug, Clone)]
pub struct Range {
    pub weights: Vec<f64>,
}

impl Range {
    pub fn uniform_unblocked(board: &[u8]) -> Self {
        let mut blocked = [false; 52];
        for &c in board {
            if (c as usize) < 52 {
                blocked[c as usize] = true;
            }
        }
        let mut weights = vec![0.0; NUM_COMBOS];
        for id in 0..NUM_COMBOS {
            let (c0, c1) = combo_cards(id);
            if !blocked[c0 as usize] && !blocked[c1 as usize] {
                weights[id] = 1.0;
            }
        }
        Self { weights }
    }

    pub fn total(&self) -> f64 {
        self.weights.iter().sum()
    }

    pub fn normalize(&mut self) {
        let t = self.total();
        if t > 0.0 {
            for w in &mut self.weights {
                *w /= t;
            }
        }
    }

    /// Zero combos that share a card with `hand` or board.
    pub fn block_cards(&mut self, cards: &[u8]) {
        let mut blocked = [false; 52];
        for &c in cards {
            if (c as usize) < 52 {
                blocked[c as usize] = true;
            }
        }
        for id in 0..NUM_COMBOS {
            let (c0, c1) = combo_cards(id);
            if blocked[c0 as usize] || blocked[c1 as usize] {
                self.weights[id] = 0.0;
            }
        }
    }

    /// Bayes update: multiply by strategy probability for action (per combo).
    pub fn multiply_action_probs(&mut self, probs: &[f64]) {
        assert_eq!(probs.len(), NUM_COMBOS);
        for id in 0..NUM_COMBOS {
            self.weights[id] *= probs[id];
        }
    }

    /// Parse range string into weights.
    ///
    /// Formats:
    /// - empty → uniform unblocked
    /// - `"combo:weight,combo:weight,..."` (combo id 0..1325)
    /// - `"AA,AKs,22"` preflop-style labels (expanded to matching combos)
    /// - `"random"` / `"100%"` → uniform unblocked
    pub fn parse(spec: &str, board: &[u8]) -> Self {
        let s = spec.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("random") || s == "100%" {
            return Self::uniform_unblocked(board);
        }
        let mut blocked = [false; 52];
        for &c in board {
            if (c as usize) < 52 {
                blocked[c as usize] = true;
            }
        }
        let mut weights = vec![0.0; NUM_COMBOS];
        let mut any = false;
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((id_s, w_s)) = part.split_once(':') {
                if let (Ok(id), Ok(w)) = (id_s.trim().parse::<usize>(), w_s.trim().parse::<f64>()) {
                    if id < NUM_COMBOS && w > 0.0 {
                        let (c0, c1) = combo_cards(id);
                        if !blocked[c0 as usize] && !blocked[c1 as usize] {
                            weights[id] += w;
                            any = true;
                        }
                    }
                }
            } else if let Ok(id) = part.parse::<usize>() {
                if id < NUM_COMBOS {
                    let (c0, c1) = combo_cards(id);
                    if !blocked[c0 as usize] && !blocked[c1 as usize] {
                        weights[id] += 1.0;
                        any = true;
                    }
                }
            } else {
                // Preflop label expansion e.g. AA, AKs, AKo
                let expanded = expand_preflop_label(part);
                for id in expanded {
                    let (c0, c1) = combo_cards(id);
                    if !blocked[c0 as usize] && !blocked[c1 as usize] {
                        weights[id] += 1.0;
                        any = true;
                    }
                }
            }
        }
        if !any {
            return Self::uniform_unblocked(board);
        }
        Self { weights }
    }
}

/// Expand a simple preflop label (AA, AKs, AKo, 22) to combo ids.
fn expand_preflop_label(label: &str) -> Vec<usize> {
    let label = label.trim();
    if label.len() < 2 {
        return vec![];
    }
    let chars: Vec<char> = label.chars().collect();
    let rank = |c: char| -> Option<u8> {
        match c {
            '2' => Some(0),
            '3' => Some(1),
            '4' => Some(2),
            '5' => Some(3),
            '6' => Some(4),
            '7' => Some(5),
            '8' => Some(6),
            '9' => Some(7),
            'T' | 't' => Some(8),
            'J' | 'j' => Some(9),
            'Q' | 'q' => Some(10),
            'K' | 'k' => Some(11),
            'A' | 'a' => Some(12),
            _ => None,
        }
    };
    let r0 = match rank(chars[0]) {
        Some(r) => r,
        None => return vec![],
    };
    let r1 = match rank(chars[1]) {
        Some(r) => r,
        None => return vec![],
    };
    let suited = chars.get(2).map(|c| *c == 's' || *c == 'S').unwrap_or(false);
    let offsuit = chars.get(2).map(|c| *c == 'o' || *c == 'O').unwrap_or(false);
    let mut out = Vec::new();
    for c0 in 0..52u8 {
        for c1 in (c0 + 1)..52u8 {
            let a = c0 / 4;
            let b = c1 / 4;
            let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
            if hi != r0.max(r1) || lo != r0.min(r1) {
                // For pairs r0==r1
                if r0 == r1 {
                    if a != r0 || b != r0 {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            if r0 == r1 {
                // pair
                if a == r0 && b == r0 {
                    out.push(cards_to_combo(c0, c1));
                }
            } else {
                let same_suit = c0 % 4 == c1 % 4;
                if suited && same_suit {
                    out.push(cards_to_combo(c0, c1));
                } else if offsuit && !same_suit {
                    out.push(cards_to_combo(c0, c1));
                } else if !suited && !offsuit {
                    // no suffix: all combos of those ranks
                    out.push(cards_to_combo(c0, c1));
                }
            }
        }
    }
    out
}

/// Precompute showdown strength ranks for all combos on a fixed 5-card board.
pub fn combo_ranks_on_board(board: &[u8; 5]) -> Vec<u32> {
    let board_cards: [Card; 5] = [
        Card(board[0]),
        Card(board[1]),
        Card(board[2]),
        Card(board[3]),
        Card(board[4]),
    ];
    let mut blocked = [false; 52];
    for &c in board {
        blocked[c as usize] = true;
    }
    let mut ranks = vec![0u32; NUM_COMBOS];
    for id in 0..NUM_COMBOS {
        let (c0, c1) = combo_cards(id);
        if blocked[c0 as usize] || blocked[c1 as usize] {
            ranks[id] = 0;
            continue;
        }
        let hole = [Card(c0), Card(c1)];
        ranks[id] = evaluate_nlh(&hole, &board_cards);
    }
    ranks
}

/// Equity of hero combo vs villain range on fixed board (already ranked).
/// Returns P(win) + 0.5*P(tie) in [0,1], or 0 if blocked.
pub fn equity_vs_range(
    hero_id: usize,
    hero_rank: u32,
    villain: &Range,
    ranks: &[u32],
) -> f64 {
    if hero_rank == 0 {
        return 0.0;
    }
    let (h0, h1) = combo_cards(hero_id);
    let mut win = 0.0;
    let mut tie = 0.0;
    let mut total = 0.0;
    for vid in 0..NUM_COMBOS {
        let w = villain.weights[vid];
        if w <= 0.0 {
            continue;
        }
        let vr = ranks[vid];
        if vr == 0 {
            continue;
        }
        let (v0, v1) = combo_cards(vid);
        if v0 == h0 || v0 == h1 || v1 == h0 || v1 == h1 {
            continue;
        }
        total += w;
        if hero_rank > vr {
            win += w;
        } else if hero_rank == vr {
            tie += w;
        }
    }
    if total <= 0.0 {
        return 0.5; // no mass — split assumption
    }
    (win + 0.5 * tie) / total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combo_roundtrip() {
        for id in 0..NUM_COMBOS {
            let (c0, c1) = combo_cards(id);
            assert!(c0 < c1);
            assert_eq!(cards_to_combo(c0, c1), id);
        }
    }

    #[test]
    fn uniform_blocks_board() {
        let r = Range::uniform_unblocked(&[0, 1, 2, 3, 4]);
        let (c0, c1) = combo_cards(0); // cards 0,1 — both on board if board starts 0,1
        // combo 0 is (0,1) which is blocked
        assert_eq!(c0, 0);
        assert_eq!(c1, 1);
        assert_eq!(r.weights[0], 0.0);
        assert!(r.total() > 1000.0);
    }

    #[test]
    fn num_combos_is_1326() {
        assert_eq!(NUM_COMBOS, 52 * 51 / 2);
    }
}
