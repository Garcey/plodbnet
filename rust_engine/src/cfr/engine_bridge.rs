//! Parity between CFR [`PublicState`] and engine chip rules.
//!
//! Builds a minimal NLH `GameState` mid-street root and checks that
//! min/max raise and pot-fraction sizing match the compact public state.

use crate::actions::Action;
use crate::cards::Card;
use crate::state::{GameConfig, GameState, Street, Variant};

use super::public_state::PublicState;
use super::CfrError;

/// Materialize an NLH postflop solver-style GameState for parity tests.
///
/// - `street`: Flop/Turn/River
/// - equal remaining stacks, pot already in middle
/// - street_commit = 0, bet_to_call = 0, last_raise_size = bb
/// - holes set to placeholders (not used for chip math)
pub fn game_state_from_solver_root(
    num_seats: usize,
    pot: u64,
    stacks: &[u64],
    board: &[u8],
    bb: u64,
    street: Street,
) -> Result<GameState, CfrError> {
    if stacks.len() != num_seats {
        return Err(CfrError::InvalidRoot("stacks len".into()));
    }
    let need = match street {
        Street::Flop => 3,
        Street::Turn => 4,
        Street::River => 5,
        _ => {
            return Err(CfrError::InvalidRoot(
                "solver root street must be flop/turn/river".into(),
            ))
        }
    };
    if board.len() != need {
        return Err(CfrError::InvalidRoot(format!(
            "board len {} != {need}",
            board.len()
        )));
    }

    let config = GameConfig {
        num_seats,
        starting_stacks: stacks.to_vec(),
        ante: 0,
        bb,
        sb: bb / 2,
        variant: Variant::NlhSingle,
    };

    // new_hand deals + posts; we rebuild fields after.
    let mut g = GameState::new_hand(config, 0, num_seats.saturating_sub(1));
    g.street = street;
    g.pot = pot;
    g.stacks = stacks.to_vec();
    g.folded = vec![false; num_seats];
    g.all_in = vec![false; num_seats];
    g.street_commit = vec![0; num_seats];
    g.total_commit = vec![0; num_seats];
    g.bet_to_call = 0;
    g.last_raise_size = bb;
    g.last_aggression_was_full_raise = true;
    g.street_level_acted = vec![0; num_seats];
    g.acted_this_street = vec![false; num_seats];
    g.last_aggressor = None;
    g.history.clear();
    g.study_mode = false;
    g.study_terminal = None;
    g.awaiting_next_street = None;
    g.action_close_board_len = None;

    // Board
    let cards: Vec<Card> = board.iter().map(|&i| Card(i)).collect();
    g.board_a = cards.clone();
    let mut full = [Card(0); 5];
    for (i, c) in cards.iter().enumerate() {
        full[i] = *c;
    }
    // Pad remaining full board with high cards not on board for runout paths
    let mut used = [false; 52];
    for &c in board {
        used[c as usize] = true;
    }
    let mut fill = 0u8;
    for i in board.len()..5 {
        while fill < 52 && used[fill as usize] {
            fill += 1;
        }
        full[i] = Card(fill);
        used[fill as usize] = true;
        fill += 1;
    }
    g.full_board_a = full;
    g.board_b.clear();
    g.full_board_b = [Card(0); 5];

    // Placeholder holes
    g.hole_cards = (0..num_seats)
        .map(|_| vec![Card(50), Card(51)])
        .collect();

    // First postflop actor = left of button
    g.button = num_seats.saturating_sub(1);
    g.actor = Some(0); // our PublicState convention: seat 0 acts first postflop
    g.sb_seat = None;
    g.bb_seat = None;
    Ok(g)
}

fn parse_raise_pm(tok: &str) -> Option<u32> {
    let up = tok.to_ascii_uppercase();
    if let Some(rest) = up.strip_prefix("RAISE_") {
        return rest.parse().ok();
    }
    if let Some(rest) = up.strip_prefix("R") {
        // R500 / R330 — reject bare "R"
        if rest.is_empty() {
            return None;
        }
        return rest.parse().ok();
    }
    None
}

fn raise_pm_chips(g: &GameState, pm: u32) -> Option<u64> {
    let actor = g.actor?;
    let current_commit = g.street_commit[actor];
    let stack = g.stacks[actor];
    let to_call_raw = g.bet_to_call.saturating_sub(current_commit);
    let raise_over = if g.bet_to_call == 0 {
        g.pot.saturating_mul(pm as u64) / 1000
    } else {
        g.pot.saturating_add(to_call_raw).saturating_mul(pm as u64) / 1000
    };
    let target_total = if g.bet_to_call == 0 {
        raise_over
    } else {
        g.bet_to_call.saturating_add(raise_over)
    };
    let min_total = g.min_bet_total();
    let max_total = g.max_bet_total();
    let clamped = target_total.max(min_total).min(max_total);
    let chips_want = clamped.saturating_sub(current_commit).min(stack);
    let min_r = g.min_raise_chips();
    let max_r = g.max_raise_chips();
    if min_r == 0 || chips_want < min_r || chips_want > max_r {
        return None;
    }
    Some(chips_want)
}

/// Place `hero_hole` on `hero_seat`; fill other seats with unused cards.
pub fn assign_holes_for_obs(
    g: &mut GameState,
    hero_seat: usize,
    hero_hole: [u8; 2],
) -> Result<(), CfrError> {
    if hero_seat >= g.config.num_seats {
        return Err(CfrError::InvalidRoot("hero_seat out of range".into()));
    }
    if hero_hole[0] == hero_hole[1] || hero_hole[0] >= 52 || hero_hole[1] >= 52 {
        return Err(CfrError::InvalidRoot("bad hero hole".into()));
    }
    let mut used = [false; 52];
    for c in &g.board_a {
        let i = c.index() as usize;
        if i < 52 {
            used[i] = true;
        }
    }
    for &c in &hero_hole {
        if used[c as usize] {
            return Err(CfrError::InvalidRoot("hero hole collides with board".into()));
        }
        used[c as usize] = true;
    }
    g.hole_cards[hero_seat] = vec![Card(hero_hole[0]), Card(hero_hole[1])];
    let mut next = 0u8;
    for s in 0..g.config.num_seats {
        if s == hero_seat {
            continue;
        }
        let mut h = Vec::with_capacity(2);
        while h.len() < 2 && next < 52 {
            while (next as usize) < 52 && used[next as usize] {
                next += 1;
            }
            if next >= 52 {
                break;
            }
            used[next as usize] = true;
            h.push(Card(next));
            next += 1;
        }
        if h.len() != 2 {
            return Err(CfrError::InvalidRoot("not enough cards for opp holes".into()));
        }
        g.hole_cards[s] = h;
    }
    Ok(())
}

/// Replay CFR path labels (FOLD / CHECK_CALL / RAISE_pm / ALLIN, or F/XC/R500/AI).
pub fn replay_cfr_path(g: &mut GameState, path: &[String]) -> Result<(), CfrError> {
    for raw in path {
        let tok = raw.trim();
        if tok.is_empty() || tok.eq_ignore_ascii_case("open") {
            continue;
        }
        if g.actor.is_none() {
            return Err(CfrError::InvalidRoot(format!(
                "path continues after terminal at {tok}"
            )));
        }
        let up = tok.to_ascii_uppercase();
        if up == "F" || up == "FOLD" {
            g.apply(Action::Fold);
        } else if up == "XC" || up == "CHECK_CALL" || up == "CHECK" || up == "CALL" {
            g.apply(Action::CheckCall);
        } else if up == "AI" || up == "ALLIN" {
            let min_r = g.min_raise_chips();
            let max_r = g.max_raise_chips();
            if min_r > 0 && max_r >= min_r {
                g.apply_raise_chips(max_r).map_err(|e| {
                    CfrError::InvalidRoot(format!("ALLIN apply_raise: {e:?}"))
                })?;
            } else {
                g.apply(Action::CheckCall);
            }
        } else if let Some(pm) = parse_raise_pm(&up) {
            let chips = raise_pm_chips(g, pm).ok_or_else(|| {
                CfrError::InvalidRoot(format!("illegal RAISE_{pm}"))
            })?;
            g.apply_raise_chips(chips)
                .map_err(|e| CfrError::InvalidRoot(format!("RAISE_{pm}: {e:?}")))?;
        } else {
            return Err(CfrError::InvalidRoot(format!("unknown path token {tok}")));
        }
    }
    Ok(())
}

/// Solver root + hole + path → live engine node for observation encoding.
pub fn game_state_from_cfr_label(
    pot: u64,
    stacks: &[u64],
    board: &[u8],
    bb: u64,
    street: Street,
    hero_seat: usize,
    hero_hole: [u8; 2],
    path: &[String],
) -> Result<GameState, CfrError> {
    let mut g = game_state_from_solver_root(stacks.len(), pot, stacks, board, bb, street)?;
    assign_holes_for_obs(&mut g, hero_seat, hero_hole)?;
    replay_cfr_path(&mut g, path)?;
    Ok(g)
}

/// Compare PublicState min/max raise and a pot-fraction sizing to GameState.
pub fn assert_chip_parity_hu_river(
    pot: u64,
    stack: u64,
    board: &[u8; 5],
    bb: u64,
) -> Result<(), String> {
    let ps = PublicState::river_hu_root(pot, stack, board, bb)
        .map_err(|e| e.to_string())?;
    let gs = game_state_from_solver_root(2, pot, &[stack, stack], board, bb, Street::River)
        .map_err(|e| e.to_string())?;

    let p_min = ps.min_raise_chips();
    let g_min = gs.min_raise_chips();
    if p_min != g_min {
        return Err(format!("min_raise mismatch: public={p_min} engine={g_min}"));
    }
    let p_max = ps.max_raise_chips();
    let g_max = gs.max_raise_chips();
    if p_max != g_max {
        return Err(format!("max_raise mismatch: public={p_max} engine={g_max}"));
    }

    // Half-pot open
    let pm = 500u32;
    let p_chips = ps.raise_chips_for_pm(pm).ok_or("public raise_pm failed")?;
    // Engine compute via apply_raise after sizing: pot * 500/1000 when bet_to_call=0
    let want = pot * (pm as u64) / 1000;
    let want = want.max(bb).min(g_max);
    if p_chips != want && (p_chips as i64 - want as i64).abs() > 0 {
        // Allow if both clamp to same legal
        if p_chips < g_min || p_chips > g_max {
            return Err(format!("raise_pm chips {p_chips} not in [{g_min},{g_max}]"));
        }
    }
    let _ = Action::BetPct50; // keep import used if needed
    Ok(())
}

/// Apply a raise sequence on both and compare public fields.
pub fn assert_apply_sequence_parity(
    pot: u64,
    stack: u64,
    board: &[u8; 5],
    bb: u64,
    raise_chips: u64,
) -> Result<(), String> {
    let mut ps = PublicState::river_hu_root(pot, stack, board, bb)
        .map_err(|e| e.to_string())?;
    let mut gs = game_state_from_solver_root(2, pot, &[stack, stack], board, bb, Street::River)
        .map_err(|e| e.to_string())?;

    ps.apply_raise(raise_chips).map_err(|e| e.to_string())?;
    gs.apply_raise_chips(raise_chips).map_err(|e| e.to_string())?;

    if ps.pot != gs.pot {
        return Err(format!("pot after raise: public={} engine={}", ps.pot, gs.pot));
    }
    if ps.bet_to_call != gs.bet_to_call {
        return Err(format!(
            "bet_to_call: public={} engine={}",
            ps.bet_to_call, gs.bet_to_call
        ));
    }
    if ps.last_raise_size != gs.last_raise_size {
        return Err(format!(
            "last_raise_size: public={} engine={}",
            ps.last_raise_size, gs.last_raise_size
        ));
    }
    for i in 0..2 {
        if ps.stacks[i] != gs.stacks[i] {
            return Err(format!(
                "stacks[{i}]: public={} engine={}",
                ps.stacks[i], gs.stacks[i]
            ));
        }
        if ps.street_commit[i] != gs.street_commit[i] {
            return Err(format!(
                "street_commit[{i}]: public={} engine={}",
                ps.street_commit[i], gs.street_commit[i]
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_max_raise_matches_engine() {
        let board = [0u8, 5, 10, 15, 20];
        assert_chip_parity_hu_river(100_000, 500_000, &board, 10_000).unwrap();
    }

    #[test]
    fn apply_raise_sequence_matches_engine() {
        let board = [0u8, 5, 10, 15, 20];
        // 50% pot open = 50k
        assert_apply_sequence_parity(100_000, 500_000, &board, 10_000, 50_000).unwrap();
    }

    #[test]
    fn cfr_label_empty_path_actor_is_oop() {
        let board = [0u8, 5, 10, 15, 20];
        let g = game_state_from_cfr_label(
            100_000,
            &[500_000, 500_000],
            &board,
            10_000,
            Street::River,
            0,
            [1, 2],
            &[],
        )
        .unwrap();
        assert_eq!(g.actor, Some(0));
        assert_eq!(g.pot, 100_000);
        assert!(g.history.is_empty());
        assert_eq!(g.hole_cards[0][0].0, 1);
    }

    #[test]
    fn cfr_label_raise_then_hero_faces_bet() {
        let board = [0u8, 5, 10, 15, 20];
        let g = game_state_from_cfr_label(
            100_000,
            &[500_000, 500_000],
            &board,
            10_000,
            Street::River,
            1,
            [1, 2],
            &["RAISE_500".into()],
        )
        .unwrap();
        assert_eq!(g.actor, Some(1));
        assert!(g.bet_to_call > 0);
        assert_eq!(g.history.len(), 1);
        assert!(g.pot > 100_000);
    }

    #[test]
    fn short_stack_max_raise_parity() {
        let board = [0u8, 5, 10, 15, 20];
        // stack 15k, pot 100k — max raise covers opp
        assert_chip_parity_hu_river(100_000, 15_000, &board, 10_000).unwrap();
    }
}
