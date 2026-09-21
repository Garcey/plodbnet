//! Known-spot correctness gates for real NLH (beyond Kuhn).
//!
//! Spot: HU river, board Kc Kd Kh 2c 3d, pot = 10 bb, stacks = 10 bb each,
//! actions = {CHECK_CALL, ALLIN} when unopened, {FOLD, CHECK_CALL} facing the
//! jam (jam/check tree).
//!
//! What any equilibrium of this tree must do (strict dominance, so CFR finds
//! it fast and the assertions do not depend on which equilibrium is reached):
//!
//! - the nuts (quads = any hand with the case king) facing a jam always CALLS;
//! - the nuts in position after a check always JAMS (checking back wins the
//!   pot and nothing more);
//! - the worst hand (5-4: it can only ever tie) facing a jam always FOLDS;
//! - air mostly checks (a pot-size jam supports 1 bluff per 2 value combos).
//!
//! OOP's nuts at the ROOT are *not* forced to jam (check-calling to induce is
//! an equilibrium line; measured jam frequency ≈ 0.5), so that is reported but
//! not asserted.
//!
//! (review 2026-09-20 F10) Before the push/fold "no check" rule was restricted
//! to preflop, this tree had NO check at all: every hand "jammed" 100%,
//! `jam_freq_mean > 0.05` was vacuous and the gate proved nothing.

use super::types::{RootSpec, SolveConfig, StreetRoot};
use super::{dcfr, CfrError};

/// Board: Kc=11*4+0=44, Kd=45, Kh=46, 2c=0, 3d=5 → use fixed indices.
/// Card encoding: rank*4+suit. K=11 → 44,45,46,47. 2c=0, 3d=1*4+1=5.
pub const SPOT_BOARD: [u8; 5] = [44, 45, 46, 0, 5];
/// The case king: any hand holding it has quads (the nuts).
pub const CASE_KING: u8 = 47;

#[derive(Debug, Clone)]
pub struct KnownSpotReport {
    pub iterations: u32,
    pub exploitability_bb: f64,
    pub n_infosets: usize,
    /// Mean ALLIN probability over every infoset that offers it.
    pub jam_freq_mean: f64,
    /// True when every unopened node offers exactly CHECK_CALL | ALLIN and
    /// every jam-facing node exactly FOLD | CHECK_CALL.
    pub tree_is_jam_or_check: bool,
    /// Quads facing a jam (either seat): mean CALL probability. Must → 1.
    pub call_quads_vs_jam: f64,
    /// Quads in position after a check: mean ALLIN probability. Must → 1.
    pub ip_jam_quads_after_check: f64,
    /// 5-4 facing a jam: mean FOLD probability. Must → 1.
    pub fold_worst_vs_jam: f64,
    /// Air (unpaired, no 2/3, ten-high or worse) unopened: mean ALLIN prob.
    pub jam_air_unopened: f64,
    /// OOP quads at the root: mean ALLIN probability (informational).
    pub root_jam_quads: f64,
    pub notes: Vec<String>,
}

#[derive(Default)]
struct Mean(f64, f64);

impl Mean {
    fn add(&mut self, x: f64) {
        self.0 += x;
        self.1 += 1.0;
    }
    fn get(&self) -> f64 {
        if self.1 > 0.0 {
            self.0 / self.1
        } else {
            f64::NAN
        }
    }
}

/// Solve the jam/check river spot and return metrics.
pub fn solve_jam_check_spot(iterations: u32, seed: u64) -> Result<KnownSpotReport, CfrError> {
    let root = RootSpec {
        num_seats: 2,
        street: StreetRoot::River,
        pot_bb: 10.0,
        effective_stack_bb: 10.0,
        bb_chips: 10_000,
        sb_chips: 5_000,
        ante_chips: 5_000,
        board: SPOT_BOARD.to_vec(),
        // Empty raise sizes + allin on a POSTFLOP root → CHECK_CALL + ALLIN
        // when unopened (the no-check push/fold rule is preflop-only).
        raise_sizes_pm: vec![], // only allin atom
        allin_atom: true,
        range_ip: String::new(),
        range_oop: String::new(),
        stacks_bb: vec![],
        root_id: "known_jam_check_river".into(),
    };
    let mut cfg = SolveConfig::default();
    cfg.max_iterations = iterations;
    cfg.seed = seed;
    cfg.target_exploitability_bb = 0.0; // run full
    cfg.use_isomorphism = true;
    cfg.algorithm = "dcfr".into();

    let rep = dcfr::solve_river_dcfr(&root, &cfg)?;
    let expl = rep.exploitability_bb.unwrap_or(f64::INFINITY);
    let mut jam_all = Mean::default();
    let mut call_quads = Mean::default();
    let mut ip_jam_quads = Mean::default();
    let mut fold_worst = Mean::default();
    let mut jam_air = Mean::default();
    let mut root_jam_quads = Mean::default();
    let mut tree_ok = true;
    for is in &rep.strategy.infosets {
        let prob = |label: &str| {
            is.actions
                .iter()
                .zip(is.probs.iter())
                .find(|(a, _)| *a == label)
                .map(|(_, p)| *p)
        };
        if let Some(p) = prob("ALLIN") {
            jam_all.add(p);
        }
        let facing_jam = is.to_call_chips.unwrap_or(0) > 0;
        let want: &[&str] = if facing_jam {
            &["FOLD", "CHECK_CALL"]
        } else {
            &["CHECK_CALL", "ALLIN"]
        };
        if is.actions != want {
            tree_ok = false;
        }
        // Only trained infosets say anything about the strategy.
        let Some(raw) = is.raw_combo else { continue };
        if is.visit_mass.unwrap_or(0.0) <= 0.0 {
            continue;
        }
        let (c0, c1) = super::range::combo_cards(raw as usize);
        let (lo, hi) = ((c0 / 4).min(c1 / 4), (c0 / 4).max(c1 / 4));
        let quads = c0 == CASE_KING || c1 == CASE_KING;
        // Unpaired, no deuce/trey (full houses), ten-high or worse.
        let air = !quads && lo != hi && lo > 1 && hi <= 8;
        let worst = (lo, hi) == (2, 3); // 5-4: plays KKK54, can only tie
        let path_len = is.path.as_ref().map_or(0, |p| p.len());
        if facing_jam {
            if quads {
                call_quads.add(prob("CHECK_CALL").unwrap_or(0.0));
            }
            if worst {
                fold_worst.add(prob("FOLD").unwrap_or(0.0));
            }
        } else {
            let jam = prob("ALLIN").unwrap_or(0.0);
            if air {
                jam_air.add(jam);
            }
            if quads && is.actor == Some(1) && path_len == 1 {
                ip_jam_quads.add(jam);
            }
            if quads && is.actor == Some(0) && path_len == 0 {
                root_jam_quads.add(jam);
            }
        }
    }
    Ok(KnownSpotReport {
        iterations,
        exploitability_bb: expl,
        n_infosets: rep.strategy.infosets.len(),
        jam_freq_mean: jam_all.get(),
        tree_is_jam_or_check: tree_ok,
        call_quads_vs_jam: call_quads.get(),
        ip_jam_quads_after_check: ip_jam_quads.get(),
        fold_worst_vs_jam: fold_worst.get(),
        jam_air_unopened: jam_air.get(),
        root_jam_quads: root_jam_quads.get(),
        notes: rep.notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nuts jam / call, the worst hand folds, air mostly checks — on a tree
    /// that really has a check.
    #[test]
    fn jam_check_spot_nuts_jam_air_checks() {
        let rep = solve_jam_check_spot(150_000, 7).expect("spot");
        eprintln!("known spot: {rep:?}");
        assert!(rep.tree_is_jam_or_check, "tree must be CHECK|ALLIN then FOLD|CALL");
        assert!(rep.notes.iter().any(|n| n.starts_with("expl_kind=exact_infoset")));
        assert!(
            rep.exploitability_bb < 0.5,
            "jam/check river exact expl {} bb (want < 0.5)",
            rep.exploitability_bb
        );
        assert!(rep.call_quads_vs_jam > 0.97, "nuts must call a jam: {}", rep.call_quads_vs_jam);
        assert!(
            rep.ip_jam_quads_after_check > 0.95,
            "nuts in position must jam after a check: {}",
            rep.ip_jam_quads_after_check
        );
        assert!(rep.fold_worst_vs_jam > 0.95, "5-4 must fold to a jam: {}", rep.fold_worst_vs_jam);
        assert!(rep.jam_air_unopened < 0.4, "air should mostly check: {}", rep.jam_air_unopened);
        // Not everything jams / nothing is degenerate.
        assert!(rep.jam_freq_mean > 0.1 && rep.jam_freq_mean < 0.7, "{}", rep.jam_freq_mean);
    }

    #[test]
    fn jam_check_expl_improves_with_iters() {
        let a = solve_jam_check_spot(2_000, 1).unwrap();
        let b = solve_jam_check_spot(60_000, 1).unwrap();
        assert!(
            b.exploitability_bb < 0.5 * a.exploitability_bb,
            "expl@2k={} expl@60k={}",
            a.exploitability_bb,
            b.exploitability_bb
        );
    }
}
