//! Kuhn poker vanilla CFR — correctness gate with known Nash equilibrium.
//!
//! Cards: J=0, Q=1, K=2. Ante 1 each. Bet size 1.
//! Nash game value for the first player ≈ −1/18 ≈ −0.0556.
//!
//! Implementation follows the standard current-player utility convention
//! (Neller & Lanctot).

use std::collections::HashMap;

const N_ACTIONS: usize = 2;

fn infoset_key(card: usize, history: &str) -> String {
    format!("{card}{history}")
}

struct Node {
    regret_sum: [f64; N_ACTIONS],
    strategy_sum: [f64; N_ACTIONS],
}

impl Node {
    fn new() -> Self {
        Self {
            regret_sum: [0.0; N_ACTIONS],
            strategy_sum: [0.0; N_ACTIONS],
        }
    }

    fn strategy(&self) -> [f64; N_ACTIONS] {
        let mut norm = 0.0;
        let mut s = [0.0; N_ACTIONS];
        for a in 0..N_ACTIONS {
            s[a] = self.regret_sum[a].max(0.0);
            norm += s[a];
        }
        if norm > 0.0 {
            for a in 0..N_ACTIONS {
                s[a] /= norm;
            }
        } else {
            s = [1.0 / N_ACTIONS as f64; N_ACTIONS];
        }
        s
    }

    fn avg_strategy(&self) -> [f64; N_ACTIONS] {
        let norm: f64 = self.strategy_sum.iter().sum();
        if norm > 0.0 {
            [
                self.strategy_sum[0] / norm,
                self.strategy_sum[1] / norm,
            ]
        } else {
            [0.5, 0.5]
        }
    }
}

fn is_terminal(history: &str) -> bool {
    let n = history.len();
    if n > 1 {
        let bytes = history.as_bytes();
        let terminal_pass = bytes[n - 1] == b'p' && bytes[n - 2] == b'p';
        let double_bet = history.chars().filter(|&c| c == 'b').count() >= 2;
        terminal_pass || double_bet || (n >= 2 && bytes[n - 1] == b'p')
        // "bp", "pbp" are terminal (fold to bet); "pp","bb","pbb" terminal
    } else {
        false
    }
}

/// Payoff for the **current player to act's opponent perspective** —
/// actually returns utility for player 0 always for terminals via cards.
fn terminal_util_for_player(history: &str, cards: [usize; 2], player: usize) -> f64 {
    let plays = history.len();
    let opponent = 1 - player;
    let bytes = history.as_bytes();

    // Double pass showdown
    if plays > 1 && bytes[plays - 1] == b'p' && bytes[plays - 2] == b'p' {
        let winner = if cards[player] > cards[opponent] {
            1.0
        } else {
            -1.0
        };
        return winner;
    }
    // Last player folded (pass after bet)
    if bytes[plays - 1] == b'p' {
        // current player just... wait terminal is evaluated before player acts.
        // When we enter terminal, the player who would act doesn't.
        // Standard: if history ends with pass after a bet, the passer folded.
        // The player who folded is (plays-1)%2, winner is the other.
        let folder = (plays - 1) % 2;
        return if folder == player { -1.0 } else { 1.0 };
    }
    // Showdown after call (ends with bet and previous bet exists)
    let winner = if cards[player] > cards[opponent] {
        2.0
    } else {
        -2.0
    };
    winner
}

/// CFR returning utility for the **current player**.
fn cfr(
    nodes: &mut HashMap<String, Node>,
    cards: [usize; 2],
    history: &str,
    p0: f64,
    p1: f64,
) -> f64 {
    let plays = history.len();
    let player = plays % 2;
    let opponent = 1 - player;

    if is_terminal(history) {
        return terminal_util_for_player(history, cards, player);
    }

    let key = infoset_key(cards[player], history);
    let strategy = {
        let node = nodes.entry(key.clone()).or_insert_with(Node::new);
        node.strategy()
    };

    let mut util = [0.0f64; N_ACTIONS];
    let actions = [b'p', b'b'];
    for (a, &ch) in actions.iter().enumerate() {
        let mut next = history.to_string();
        next.push(ch as char);
        util[a] = if player == 0 {
            -cfr(nodes, cards, &next, p0 * strategy[a], p1)
        } else {
            -cfr(nodes, cards, &next, p0, p1 * strategy[a])
        };
    }

    let node_util: f64 = (0..N_ACTIONS).map(|a| strategy[a] * util[a]).sum();
    {
        let node = nodes.get_mut(&key).unwrap();
        for a in 0..N_ACTIONS {
            let regret = util[a] - node_util;
            if player == 0 {
                node.regret_sum[a] += p1 * regret;
            } else {
                node.regret_sum[a] += p0 * regret;
            }
        }
        for a in 0..N_ACTIONS {
            if player == 0 {
                node.strategy_sum[a] += p0 * strategy[a];
            } else {
                node.strategy_sum[a] += p1 * strategy[a];
            }
        }
    }
    let _ = opponent;
    node_util
}

pub fn solve_kuhn(iterations: u32) -> KuhnReport {
    let mut nodes: HashMap<String, Node> = HashMap::new();
    let mut util_sum = 0.0;
    let mut deals = 0u32;

    for _ in 0..iterations {
        for c0 in 0..3 {
            for c1 in 0..3 {
                if c0 == c1 {
                    continue;
                }
                util_sum += cfr(&mut nodes, [c0, c1], "", 1.0, 1.0);
                deals += 1;
            }
        }
    }

    let mut strategies = HashMap::new();
    for (k, node) in &nodes {
        // Map string key to numeric for report API stability
        let card = k.chars().next().unwrap().to_digit(10).unwrap_or(0);
        let hist = &k[1..];
        let hcode = match hist {
            "" => 0,
            "p" => 1,
            "b" => 2,
            "pb" => 3,
            _ => 9,
        };
        let id = card * 10 + hcode;
        strategies.insert(id, node.avg_strategy());
    }

    let value = util_sum / deals as f64;
    // Recompute value from average strategy for accuracy
    let value_avg = game_value_avg(&nodes);
    let expl = exploitability_avg(&nodes);

    KuhnReport {
        iterations,
        deals,
        value_p0: value_avg,
        strategies,
        exploitability: expl,
        training_value: value,
    }
}

fn avg_strat_at(nodes: &HashMap<String, Node>, card: usize, history: &str) -> [f64; 2] {
    let key = infoset_key(card, history);
    nodes
        .get(&key)
        .map(|n| n.avg_strategy())
        .unwrap_or([0.5, 0.5])
}

fn game_value_avg(nodes: &HashMap<String, Node>) -> f64 {
    let mut total = 0.0;
    let mut n = 0.0;
    for c0 in 0..3usize {
        for c1 in 0..3usize {
            if c0 == c1 {
                continue;
            }
            // Utility for P0 under average strategies
            total += eval_p0(nodes, [c0, c1], "");
            n += 1.0;
        }
    }
    total / n
}

fn eval_p0(nodes: &HashMap<String, Node>, cards: [usize; 2], history: &str) -> f64 {
    if is_terminal(history) {
        // P0 utility
        return terminal_util_for_player(history, cards, 0);
    }
    let player = history.len() % 2;
    let s = avg_strat_at(nodes, cards[player], history);
    let mut v = 0.0;
    for (a, ch) in [b'p', b'b'].iter().enumerate() {
        let mut next = history.to_string();
        next.push(*ch as char);
        v += s[a] * eval_p0(nodes, cards, &next);
    }
    v
}

/// Exploitability via infoset-consistent pure BR (not omniscient full-state BR).
/// For each infoset of the BR player, pick the action that maximizes expected
/// utility averaged over opponent cards (uniform over remaining cards).
fn exploitability_avg(nodes: &HashMap<String, Node>) -> f64 {
    let v = game_value_avg(nodes);
    let br0 = infoset_br_value(nodes, 0);
    let br1 = infoset_br_value(nodes, 1);
    // br_* are utilities for that player
    ((br0 - v) + (br1 - (-v))) / 2.0
}

fn infoset_br_value(nodes: &HashMap<String, Node>, br_player: usize) -> f64 {
    // Build pure BR policy: infoset_key -> action index
    let mut policy: HashMap<String, usize> = HashMap::new();
    // All infosets for br_player that appear under average play
    let infosets = collect_infosets(br_player);
    for key in infosets {
        let best_a = best_action_for_infoset(nodes, &key, br_player);
        policy.insert(key, best_a);
    }
    // Evaluate policy vs average opponent
    let mut total = 0.0;
    let mut n = 0.0;
    for c0 in 0..3usize {
        for c1 in 0..3usize {
            if c0 == c1 {
                continue;
            }
            let u0 = eval_with_br(nodes, &policy, [c0, c1], "", br_player);
            let up = if br_player == 0 { u0 } else { -u0 };
            total += up;
            n += 1.0;
        }
    }
    total / n
}

fn collect_infosets(player: usize) -> Vec<String> {
    // All card×history pairs where it's this player's turn
    let mut out = Vec::new();
    let histories: &[&str] = if player == 0 {
        &["", "pb"] // P0 acts at root and after check-bet
    } else {
        &["p", "b"] // P1 acts after check or bet
    };
    for card in 0..3usize {
        for &h in histories {
            out.push(infoset_key(card, h));
        }
    }
    out
}

fn best_action_for_infoset(
    nodes: &HashMap<String, Node>,
    key: &str,
    br_player: usize,
) -> usize {
    // key = "{card}{history}"
    let card = key.chars().next().unwrap().to_digit(10).unwrap() as usize;
    let history = &key[1..];
    let mut best_a = 0;
    let mut best_v = f64::NEG_INFINITY;
    for a in 0..N_ACTIONS {
        let mut ev = 0.0;
        let mut w = 0.0;
        for opp in 0..3usize {
            if opp == card {
                continue;
            }
            let cards = if br_player == 0 {
                [card, opp]
            } else {
                [opp, card]
            };
            let mut next = history.to_string();
            next.push(if a == 0 { 'p' } else { 'b' });
            // Continue with average strategy for both (BR only at this decision
            // for action ranking — full tree BR is composed via policy table)
            let u0 = eval_p0(nodes, cards, &next);
            let up = if br_player == 0 { u0 } else { -u0 };
            ev += up;
            w += 1.0;
        }
        let v = if w > 0.0 { ev / w } else { 0.0 };
        if v > best_v {
            best_v = v;
            best_a = a;
        }
    }
    best_a
}

fn eval_with_br(
    nodes: &HashMap<String, Node>,
    policy: &HashMap<String, usize>,
    cards: [usize; 2],
    history: &str,
    br_player: usize,
) -> f64 {
    // Returns P0 utility
    if is_terminal(history) {
        return terminal_util_for_player(history, cards, 0);
    }
    let player = history.len() % 2;
    if player == br_player {
        let key = infoset_key(cards[player], history);
        let a = policy.get(&key).copied().unwrap_or(0);
        let mut next = history.to_string();
        next.push(if a == 0 { 'p' } else { 'b' });
        eval_with_br(nodes, policy, cards, &next, br_player)
    } else {
        let s = avg_strat_at(nodes, cards[player], history);
        let mut v = 0.0;
        for (a, &ch) in [b'p', b'b'].iter().enumerate() {
            let mut next = history.to_string();
            next.push(ch as char);
            v += s[a] * eval_with_br(nodes, policy, cards, &next, br_player);
        }
        v
    }
}

#[derive(Debug, Clone)]
pub struct KuhnReport {
    pub iterations: u32,
    pub deals: u32,
    pub value_p0: f64,
    pub strategies: HashMap<u32, [f64; 2]>,
    pub exploitability: f64,
    pub training_value: f64,
}

pub const KUHN_NASH_VALUE: f64 = -1.0 / 18.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kuhn_converges_near_nash() {
        let rep = solve_kuhn(20_000);
        assert!(
            (rep.value_p0 - KUHN_NASH_VALUE).abs() < 0.03,
            "value {} vs nash {}",
            rep.value_p0,
            KUHN_NASH_VALUE
        );
        // Tight gate: after 20k iters Kuhn should be near 0 expl.
        assert!(
            rep.exploitability < 0.01,
            "exploitability {} value {}",
            rep.exploitability,
            rep.value_p0
        );
    }

    #[test]
    fn kuhn_king_bets_often() {
        let rep = solve_kuhn(10_000);
        // card 2 = K at root ""
        let key = 2 * 10 + 0;
        if let Some(s) = rep.strategies.get(&key) {
            assert!(s[1] > 0.4, "K bet freq {}", s[1]);
        } else {
            panic!("missing K root infoset");
        }
    }

    #[test]
    fn terminal_utils_sane() {
        // pp showdown K vs J: P0 wins +1
        assert!((terminal_util_for_player("pp", [2, 0], 0) - 1.0).abs() < 1e-9);
        // bp: P1 folds, P0 wins +1
        assert!((terminal_util_for_player("bp", [0, 1], 0) - 1.0).abs() < 1e-9);
        assert!((terminal_util_for_player("bp", [0, 1], 1) - (-1.0)).abs() < 1e-9);
    }
}
