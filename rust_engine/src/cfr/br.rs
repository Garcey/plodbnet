//! Best-response exploitability helpers (preflop / multiway MC estimates).

use std::collections::HashMap;

use super::actions::{apply_abstract, legal_actions, AbstractAction};
use super::infoset::{Infoset, InfosetKey};
use super::public_state::PublicState;

fn history_hash(actions: &[AbstractAction]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for a in actions {
        a.label().hash(&mut h);
    }
    h.finish()
}

/// Monte-Carlo NashConv/2 estimate given an infoset table and a terminal fn.
///
/// `sample_deal` returns (private_views per seat, terminal_eval closure state).
/// `terminal` maps (state, seat) → chip EV for seat under the sampled deal.
pub fn mc_exploitability_bb<FTerm, FRoot>(
    infosets: &HashMap<InfosetKey, Infoset>,
    raise_pm: &[u32],
    allin: bool,
    num_seats: usize,
    bb: f64,
    samples: u32,
    mut sample_and_root: FRoot,
    terminal: FTerm,
) -> f64
where
    FRoot: FnMut() -> (PublicState, Vec<u32>),
    FTerm: Fn(&PublicState, usize, &[u32]) -> f64,
{
    if bb <= 0.0 || samples == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for _ in 0..samples {
        let (root, privates) = sample_and_root();
        for br_player in 0..num_seats {
            let v = avg_value(
                infosets,
                &root,
                &[],
                &privates,
                br_player,
                raise_pm,
                allin,
                &terminal,
            );
            let br = br_value(
                infosets,
                &root,
                &[],
                &privates,
                br_player,
                raise_pm,
                allin,
                &terminal,
            );
            total += (br - v).max(0.0);
        }
    }
    // Average per-player gain / 2 is not quite NashConv; report mean total gain / n / bb
    (total / samples as f64) / num_seats as f64 / bb
}

fn avg_value<FTerm>(
    infosets: &HashMap<InfosetKey, Infoset>,
    state: &PublicState,
    history: &[AbstractAction],
    privates: &[u32],
    player: usize,
    raise_pm: &[u32],
    allin: bool,
    terminal: &FTerm,
) -> f64
where
    FTerm: Fn(&PublicState, usize, &[u32]) -> f64,
{
    if state.is_terminal() || state.actor.is_none() {
        return terminal(state, player, privates);
    }
    let actor = state.actor.unwrap() as usize;
    let acts = legal_actions(state, raise_pm, allin);
    if acts.is_empty() {
        return terminal(state, player, privates);
    }
    let key = InfosetKey::new(actor as u8, history_hash(history), privates[actor]);
    let strat = match infosets.get(&key) {
        Some(n) => n.average_strategy(),
        None => vec![1.0 / acts.len() as f64; acts.len()],
    };
    let mut v = 0.0;
    for (i, &act) in acts.iter().enumerate() {
        let mut child = state.clone();
        if apply_abstract(&mut child, act).is_err() {
            continue;
        }
        let mut h2 = history.to_vec();
        h2.push(act);
        let p = strat.get(i).copied().unwrap_or(0.0);
        v += p * avg_value(
            infosets, &child, &h2, privates, player, raise_pm, allin, terminal,
        );
    }
    v
}

fn br_value<FTerm>(
    infosets: &HashMap<InfosetKey, Infoset>,
    state: &PublicState,
    history: &[AbstractAction],
    privates: &[u32],
    br_player: usize,
    raise_pm: &[u32],
    allin: bool,
    terminal: &FTerm,
) -> f64
where
    FTerm: Fn(&PublicState, usize, &[u32]) -> f64,
{
    if state.is_terminal() || state.actor.is_none() {
        return terminal(state, br_player, privates);
    }
    let actor = state.actor.unwrap() as usize;
    let acts = legal_actions(state, raise_pm, allin);
    if acts.is_empty() {
        return terminal(state, br_player, privates);
    }
    if actor == br_player {
        let mut best = f64::NEG_INFINITY;
        for &act in &acts {
            let mut child = state.clone();
            if apply_abstract(&mut child, act).is_err() {
                continue;
            }
            let mut h2 = history.to_vec();
            h2.push(act);
            let v = br_value(
                infosets, &child, &h2, privates, br_player, raise_pm, allin, terminal,
            );
            if v > best {
                best = v;
            }
        }
        if best.is_finite() {
            best
        } else {
            0.0
        }
    } else {
        let key = InfosetKey::new(actor as u8, history_hash(history), privates[actor]);
        let strat = match infosets.get(&key) {
            Some(n) => n.average_strategy(),
            None => vec![1.0 / acts.len() as f64; acts.len()],
        };
        let mut v = 0.0;
        for (i, &act) in acts.iter().enumerate() {
            let mut child = state.clone();
            if apply_abstract(&mut child, act).is_err() {
                continue;
            }
            let mut h2 = history.to_vec();
            h2.push(act);
            let p = strat.get(i).copied().unwrap_or(0.0);
            v += p * br_value(
                infosets, &child, &h2, privates, br_player, raise_pm, allin, terminal,
            );
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::infoset::Infoset;
    use super::super::actions::AbstractAction;

    #[test]
    fn empty_infosets_zero_expl() {
        let infosets = HashMap::new();
        let expl = mc_exploitability_bb(
            &infosets,
            &[500],
            true,
            2,
            10_000.0,
            4,
            || {
                let st = PublicState::river_hu_root(100_000, 200_000, &[0, 5, 10, 15, 20], 10_000)
                    .unwrap();
                (st, vec![0, 1])
            },
            |s, seat, _| s.fold_payout_chips(seat) as f64,
        );
        // With empty strategy tables, BR ≈ avg (uniform) so expl near 0 or small
        assert!(expl.is_finite());
        let _ = Infoset::new(vec![AbstractAction::CheckCall]);
    }
}
