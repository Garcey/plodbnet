//! Abstract CFR action menu: FOLD | CHECK_CALL | RAISE_pm | ALLIN.

use super::public_state::PublicState;

/// Abstract action in the solve ladder (not the PPO 8-action enum).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbstractAction {
    Fold,
    CheckCall,
    /// Raise by pot-fraction per-mille (chip delta computed at apply time).
    RaisePm(u32),
    AllIn,
}

impl AbstractAction {
    pub fn label(self) -> String {
        match self {
            AbstractAction::Fold => "FOLD".into(),
            AbstractAction::CheckCall => "CHECK_CALL".into(),
            AbstractAction::RaisePm(pm) => format!("RAISE_{pm}"),
            AbstractAction::AllIn => "ALLIN".into(),
        }
    }
}

/// Legal abstract actions at a public node, given the size menu.
///
/// **Push/fold mode** (`raise_sizes_pm` empty + `allin_atom`): only FOLD and
/// ALLIN (no limp / no partial call). Matches Monker all-in-or-fold trees.
pub fn legal_actions(state: &PublicState, raise_sizes_pm: &[u32], allin_atom: bool) -> Vec<AbstractAction> {
    if state.is_terminal() {
        return vec![];
    }
    let actor = state.actor.unwrap() as usize;
    let to_call = state.to_call_chips();
    let stack = state.stacks[actor];
    let min_r = state.min_raise_chips();
    let max_r = state.max_raise_chips();

    // Pure jam-or-fold tree: empty size menu + all-in atom.
    if raise_sizes_pm.is_empty() && allin_atom {
        let mut out = Vec::with_capacity(2);
        if to_call > 0 {
            out.push(AbstractAction::Fold);
            // Call all-in or reshove — both go through AllIn apply path.
            if stack > 0 {
                out.push(AbstractAction::AllIn);
            }
        } else if stack > 0 && min_r > 0 && max_r >= min_r && stack >= min_r {
            // Open shove (no check / limp)
            out.push(AbstractAction::AllIn);
        } else if stack > 0 && to_call == 0 {
            // Free option with no legal raise (e.g. short lockout) — check only
            out.push(AbstractAction::CheckCall);
        }
        return out;
    }

    let mut out = Vec::with_capacity(2 + raise_sizes_pm.len());

    // Fold only when facing a bet
    if to_call > 0 {
        out.push(AbstractAction::Fold);
    }
    // Check / call always available for voluntary actor
    out.push(AbstractAction::CheckCall);

    if min_r > 0 && max_r >= min_r {
        let mut seen_chips = Vec::new();
        for &pm in raise_sizes_pm {
            if let Some(chips) = state.raise_chips_for_pm(pm) {
                if !seen_chips.contains(&chips) {
                    seen_chips.push(chips);
                    out.push(AbstractAction::RaisePm(pm));
                }
            }
        }
        if allin_atom {
            // All-in as raise only when stack is a legal raise amount
            if stack >= min_r && stack <= max_r && !seen_chips.contains(&stack) {
                out.push(AbstractAction::AllIn);
            } else if stack > 0 && to_call > 0 && stack == to_call {
                // short call-all-in already covered by CheckCall
            } else if stack > max_r && max_r >= min_r {
                // stack exceeds max (shouldn't for NL with cover cap) — skip
            } else if stack >= min_r && stack <= max_r {
                out.push(AbstractAction::AllIn);
            }
        }
    } else if allin_atom {
        // No raise legal; call-all-in covered by CheckCall when facing bet.
        let _ = (stack, min_r, max_r, to_call);
    }

    out
}

/// Apply abstract action; returns chip delta contributed (0 for fold/check).
pub fn apply_abstract(
    state: &mut PublicState,
    action: AbstractAction,
) -> Result<u64, super::CfrError> {
    match action {
        AbstractAction::Fold => {
            state.apply_fold();
            Ok(0)
        }
        AbstractAction::CheckCall => {
            let c = state.to_call_chips();
            state.apply_check_call();
            Ok(c)
        }
        AbstractAction::RaisePm(pm) => {
            let chips = state
                .raise_chips_for_pm(pm)
                .ok_or_else(|| super::CfrError::InvalidConfig(format!("illegal RAISE_{pm}")))?;
            state.apply_raise(chips)?;
            Ok(chips)
        }
        AbstractAction::AllIn => {
            let actor = state.actor.ok_or_else(|| {
                super::CfrError::InvalidConfig("allin terminal".into())
            })? as usize;
            let stack = state.stacks[actor];
            let min_r = state.min_raise_chips();
            let max_r = state.max_raise_chips();
            if stack >= min_r && stack <= max_r && min_r > 0 {
                state.apply_raise(stack)?;
                Ok(stack)
            } else {
                // Fall back to call all-in
                let c = state.to_call_chips();
                state.apply_check_call();
                Ok(c)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::public_state::PublicState;

    #[test]
    fn opening_menu_has_check_and_raises() {
        let s = PublicState::river_hu_root(100_000, 500_000, &[0, 1, 2, 3, 4], 10_000).unwrap();
        let acts = legal_actions(&s, &[500, 1000], true);
        assert!(acts.contains(&AbstractAction::CheckCall));
        assert!(!acts.contains(&AbstractAction::Fold));
        assert!(acts.iter().any(|a| matches!(a, AbstractAction::RaisePm(_))));
    }

    #[test]
    fn facing_bet_has_fold() {
        let mut s = PublicState::river_hu_root(100_000, 500_000, &[0, 1, 2, 3, 4], 10_000).unwrap();
        apply_abstract(&mut s, AbstractAction::RaisePm(500)).unwrap();
        let acts = legal_actions(&s, &[500, 1000], true);
        assert!(acts.contains(&AbstractAction::Fold));
        assert!(acts.contains(&AbstractAction::CheckCall));
    }

    #[test]
    fn push_fold_menu_open_is_allin_only() {
        // Preflop-like: facing BB, stack can shove
        let mut s = PublicState::river_hu_root(15_000, 100_000, &[0, 1, 2, 3, 4], 10_000).unwrap();
        // Force facing bet: set pot/commits like preflop BB posted
        s.street = 0;
        s.bet_to_call = 10_000;
        s.street_commit[0] = 0;
        s.street_commit[1] = 10_000;
        s.stacks[0] = 100_000;
        s.stacks[1] = 90_000;
        s.actor = Some(0);
        let acts = legal_actions(&s, &[], true);
        assert_eq!(acts, vec![AbstractAction::Fold, AbstractAction::AllIn]);
        assert!(!acts.contains(&AbstractAction::CheckCall));
    }

    #[test]
    fn push_fold_open_shove_no_check() {
        let mut s = PublicState::river_hu_root(15_000, 100_000, &[0, 1, 2, 3, 4], 10_000).unwrap();
        s.street = 0;
        s.bet_to_call = 0;
        s.actor = Some(0);
        s.stacks[0] = 100_000;
        let acts = legal_actions(&s, &[], true);
        assert_eq!(acts, vec![AbstractAction::AllIn]);
    }

    /// BB facing a jam: FOLD | ALLIN only (no limp / partial call).
    #[test]
    fn push_fold_bb_vs_jam_menu() {
        let mut s = PublicState::river_hu_root(15_000, 100_000, &[0, 1, 2, 3, 4], 10_000).unwrap();
        s.street = 0;
        // BB already posted 10k; facing jam so bet_to_call is jam total
        s.bet_to_call = 100_000;
        s.street_commit[0] = 10_000; // BB seat as actor with 10k posted
        s.street_commit[1] = 100_000; // jammer
        s.stacks[0] = 90_000;
        s.stacks[1] = 0;
        s.actor = Some(0);
        s.all_in[1] = true;
        let acts = legal_actions(&s, &[], true);
        assert_eq!(acts, vec![AbstractAction::Fold, AbstractAction::AllIn]);
        assert!(!acts.contains(&AbstractAction::CheckCall));
    }
}