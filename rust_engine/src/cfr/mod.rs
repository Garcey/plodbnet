//! NLH CFR solver (preflop → river, multiway-capable), ClubGG chip rules.
//!
//! - DCFR for HU postflop (flop/turn with runouts, river exact)
//! - External-sampling MCCFR for preflop / multiway
//! - Compact [`public_state::PublicState`] (engine parity via engine_bridge)
//!
//! `exploitability_bb` is always labelled in the report notes
//! (`expl_kind=<kind>`, review 2026-09-20 D8/D10):
//!
//! | kind | where | what it is |
//! |---|---|---|
//! | `exact_infoset` | HU river; HU turn when it fits the time allowance | vectorized infoset best response over all combos (turn: exact expectation over rivers) |
//! | `sampled_runout_br` | HU flop; HU turn fallback | same best response against a sampled runout grid (upper-biased) |
//! | `mc_poll` | any HU postflop | the final pass ran out of time; value is the last in-loop poll |
//! | `mc_br_proxy` | HU preflop, multiway | perfect-information deal-BR proxy — not a Nash certificate |
//! | `none` | any | nothing could be computed in the allowance |

pub mod actions;
pub mod br;
pub mod card_abs;
pub mod dcfr;
pub mod engine_bridge;
pub mod infoset;
pub mod known_spot;
pub mod kuhn;
pub mod mccfr;
pub mod memory;
pub mod pipeline;
pub mod preflop;
pub mod public_state;
pub mod py_api;
pub mod range;
pub mod showdown;
pub mod types;

use types::{RootSpec, SolveConfig, SolveReport, StreetRoot};

/// Run CFR / MCCFR for ``root`` under ``config``.
pub fn solve(root: &RootSpec, config: &SolveConfig) -> Result<SolveReport, CfrError> {
    root.validate_for_solve()?;
    config.validate()?;

    // Auto card-abs for flop when left at none
    let mut cfg = config.clone();
    if root.street == StreetRoot::Flop
        && (cfg.card_abstraction == "none" || cfg.card_abstraction.is_empty())
    {
        cfg.card_abstraction = "ochs".into();
    }

    if root.num_seats > 2 {
        return match root.street {
            StreetRoot::Preflop => mccfr::solve_multiway_preflop_mccfr(root, &cfg),
            _ => mccfr::solve_multiway_mccfr(root, &cfg),
        };
    }

    match root.street {
        StreetRoot::River => dcfr::solve_river_dcfr(root, &cfg),
        StreetRoot::Flop | StreetRoot::Turn => dcfr::solve_postflop_with_runouts(root, &cfg),
        StreetRoot::Preflop => mccfr::solve_preflop_mccfr(root, &cfg),
    }
}

/// Kuhn poker gate (known NE).
pub fn solve_kuhn(iterations: u32) -> kuhn::KuhnReport {
    kuhn::solve_kuhn(iterations)
}

/// Full-hand pipeline: preflop → induce → postflop.
pub use pipeline::{pipeline_to_solve_report, solve_preflop_to_postflop, PipelineReport};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfrError {
    InvalidRoot(String),
    InvalidConfig(String),
    NotImplemented(&'static str),
}

impl std::fmt::Display for CfrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CfrError::InvalidRoot(s) => write!(f, "invalid root: {s}"),
            CfrError::InvalidConfig(s) => write!(f, "invalid config: {s}"),
            CfrError::NotImplemented(s) => write!(f, "not implemented: {s}"),
        }
    }
}

impl std::error::Error for CfrError {}

#[cfg(test)]
mod tests {
    use super::*;
    use types::DEFAULT_RAISE_SIZES_PM;

    #[test]
    fn solve_preflop_runs_mccfr() {
        let root = RootSpec::preflop_hu(100.0, 10_000, 5_000, 5_000);
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 50;
        cfg.algorithm = "mccfr_es".into();
        let rep = solve(&root, &cfg).expect("preflop");
        assert_eq!(rep.status, "ok");
        assert!(rep.iterations_run > 0);
    }

    #[test]
    fn solve_river_runs_dcfr() {
        let root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            15.0,
            vec![0, 5, 10, 15, 20],
            vec![500, 1000],
        );
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 80;
        cfg.seed = 7;
        let rep = solve(&root, &cfg).expect("river");
        assert_eq!(rep.status, "ok");
        assert!(!rep.strategy.infosets.is_empty());
    }

    #[test]
    fn solve_flop_with_runouts() {
        let root = RootSpec::postflop_hu(
            StreetRoot::Flop,
            8.0,
            20.0,
            vec![0, 5, 10],
            vec![500, 1000],
        );
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 60;
        cfg.seed = 11;
        cfg.target_exploitability_bb = 0.0;
        let rep = solve(&root, &cfg).expect("flop");
        assert_eq!(rep.status, "ok");
        assert!(!rep.strategy.infosets.is_empty());
    }

    #[test]
    fn solve_turn_with_runouts() {
        let root = RootSpec::postflop_hu(
            StreetRoot::Turn,
            10.0,
            20.0,
            vec![0, 5, 10, 15],
            vec![500, 1000],
        );
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 50;
        cfg.seed = 12;
        cfg.target_exploitability_bb = 0.0;
        let rep = solve(&root, &cfg).expect("turn");
        assert_eq!(rep.status, "ok");
    }

    #[test]
    fn solve_river_root_requires_five_board_cards() {
        let mut root = RootSpec::postflop_hu(
            StreetRoot::River,
            20.0,
            50.0,
            vec![0, 1, 2, 3],
            DEFAULT_RAISE_SIZES_PM.to_vec(),
        );
        assert!(solve(&root, &SolveConfig::default()).is_err());
        root.board = vec![0, 1, 2, 3, 4];
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 20;
        assert!(solve(&root, &cfg).is_ok());
    }

    #[test]
    fn invalid_stack_rejected() {
        let mut root = RootSpec::preflop_hu(100.0, 10_000, 5_000, 5_000);
        root.effective_stack_bb = 0.0;
        assert!(matches!(
            solve(&root, &SolveConfig::default()),
            Err(CfrError::InvalidRoot(_))
        ));
    }

    #[test]
    fn multiway_three_seat_smoke() {
        let mut root = RootSpec::postflop_hu(
            StreetRoot::River,
            12.0,
            25.0,
            vec![0, 5, 10, 15, 20],
            vec![500, 1000],
        );
        root.num_seats = 3;
        root.root_id = "mw3".into();
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 40;
        cfg.algorithm = "mccfr_es".into();
        let rep = solve(&root, &cfg).expect("mw");
        assert_eq!(rep.status, "ok");
    }

    #[test]
    fn multiway_preflop_smoke() {
        // Pure push/fold keeps the MC-BR expl tree tiny (FOLD|ALLIN only).
        // Raised-size multiway preflop BR is O(branch^depth) and is not a unit-test.
        let mut root = RootSpec::preflop_hu(10.0, 10_000, 5_000, 0);
        root.num_seats = 3;
        root.pot_bb = 1.5;
        root.root_id = "mw3_pf".into();
        root.raise_sizes_pm = vec![];
        root.allin_atom = true;
        root.stacks_bb = vec![10.0; 3];
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 40;
        cfg.algorithm = "mccfr_es".into();
        cfg.seed = 1;
        let rep = solve(&root, &cfg).expect("mw preflop");
        assert_eq!(rep.status, "ok");
        assert!(!rep.strategy.infosets.is_empty());
        assert!(rep.notes.iter().any(|n| n.contains("multiway preflop")));
        assert!(rep.notes.iter().any(|n| n.contains("mc_br_proxy")));
    }

    #[test]
    fn river_expl_finite_after_warmup() {
        let root = RootSpec::postflop_hu(
            StreetRoot::River,
            6.0,
            12.0,
            vec![0, 5, 10, 15, 20],
            vec![500, 1000],
        );
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 200;
        cfg.seed = 42;
        cfg.target_exploitability_bb = 0.0;
        let rep = solve(&root, &cfg).expect("river");
        assert_eq!(rep.status, "ok");
        let expl = rep.exploitability_bb.expect("expl");
        assert!(expl.is_finite() && expl >= 0.0, "expl={expl}");
        // After 200 iters on tiny SPR, expl should be better than random (~pot)
        assert!(expl < 50.0, "expl too high: {expl}");
    }

    #[test]
    fn known_jam_check_spot_tight() {
        // Exact infoset exploitability (was a noisy MC number with a 4 bb bar
        // on a tree that, before F10, had no check at all).
        let rep = known_spot::solve_jam_check_spot(40_000, 3).expect("spot");
        assert!(rep.tree_is_jam_or_check);
        assert!(
            rep.exploitability_bb < 1.0,
            "known spot expl {}",
            rep.exploitability_bb
        );
    }

    /// (review 2026-09-20 E1) HU preflop with a sub-blind stack used to recurse
    /// forever (native stack overflow, the whole process died). Every
    /// degenerate root is now an ordinary `InvalidRoot` error.
    #[test]
    fn sub_blind_hu_preflop_is_an_error_not_a_stack_overflow() {
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 20;
        cfg.algorithm = "mccfr_es".into();
        // Default blinds/ante: ante 0.5bb + bb 1bb ⇒ need > 1.5bb.
        for stack in [0.3, 0.5, 0.9, 1.0, 1.4, 1.5] {
            let mut root = RootSpec::preflop_hu(stack, 10_000, 5_000, 5_000);
            root.raise_sizes_pm = vec![1000];
            let err = solve(&root, &cfg).expect_err(&format!("stack {stack}bb must be refused"));
            assert!(matches!(err, CfrError::InvalidRoot(_)), "{stack}: {err}");
        }
        // No ante: need > 1bb.
        for stack in [0.4, 0.9, 1.0] {
            let mut root = RootSpec::preflop_hu(stack, 10_000, 5_000, 0);
            root.raise_sizes_pm = vec![1000];
            assert!(matches!(solve(&root, &cfg), Err(CfrError::InvalidRoot(_))), "{stack}");
        }
        // Just above the blind: a real (tiny) tree that terminates.
        for stack in [1.6, 2.0] {
            let mut root = RootSpec::preflop_hu(stack, 10_000, 5_000, 5_000);
            root.raise_sizes_pm = vec![1000];
            let rep = solve(&root, &cfg).expect("shallow but valid");
            assert_eq!(rep.status, "ok");
            assert!(!rep.strategy.infosets.is_empty());
        }
        // Multiway: stacks that do not cover the ante leave nobody to act.
        let mut mw = RootSpec::preflop_hu(0.2, 10_000, 5_000, 5_000);
        mw.num_seats = 3;
        mw.raise_sizes_pm = vec![];
        mw.stacks_bb = vec![0.2; 3];
        assert!(matches!(solve(&mw, &cfg), Err(CfrError::InvalidRoot(_))));
    }

    /// (review 2026-09-20 E1) `pot_bb` / `effective_stack_bb` that round to 0
    /// chips (or overflow) are errors — they used to hit `unwrap()` and surface
    /// in Python as a `PanicException`.
    #[test]
    fn zero_chip_roots_are_errors_not_panics() {
        let board = vec![12, 28, 38, 41, 45];
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 5;
        let river = |pot: f64, stack: f64| {
            RootSpec::postflop_hu(StreetRoot::River, pot, stack, board.clone(), vec![1000])
        };
        for (pot, stack) in [(0.00004, 50.0), (10.0, 0.00004), (1e15, 1e15), (10.0, f64::MAX)] {
            let err = solve(&river(pot, stack), &cfg).expect_err("degenerate chips");
            assert!(matches!(err, CfrError::InvalidRoot(_)), "{pot}/{stack}: {err}");
        }
        let mut tiny_bb = river(0.4, 50.0);
        tiny_bb.bb_chips = 1;
        assert!(matches!(solve(&tiny_bb, &cfg), Err(CfrError::InvalidRoot(_))));
        let mut mw = river(10.0, 0.00004);
        mw.num_seats = 3;
        cfg.algorithm = "mccfr_es".into();
        assert!(matches!(solve(&mw, &cfg), Err(CfrError::InvalidRoot(_))));
    }

    /// (review 2026-09-20 E1) `max_iterations=0` means "unlimited": refused
    /// unless a time budget or a stop file can end the run.
    #[test]
    fn unlimited_iterations_need_a_stop_condition() {
        let root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            10.0,
            vec![0, 5, 10, 15, 20],
            vec![],
        );
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 0;
        assert!(matches!(solve(&root, &cfg), Err(CfrError::InvalidConfig(_))));
        cfg.time_budget_secs = 0.05;
        cfg.target_exploitability_bb = 0.0;
        let rep = solve(&root, &cfg).expect("budgeted unlimited run");
        assert!(rep.notes.iter().any(|n| n == "early_stop=time_budget"));
    }

    #[test]
    fn range_string_used() {
        let mut root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            20.0,
            vec![0, 5, 10, 15, 20],
            vec![500, 1000],
        );
        // Restrict to a few combos.
        // (review 2026-09-20 D12) This test used IP ids 200..203 = (x, 20):
        // every one holds board card 20, so the whole IP range was blocked,
        // the parser silently fell back to UNIFORM and the assertion on
        // "parsed" still passed. Ids below avoid the board {0,5,10,15,20}.
        root.range_oop = "100:1,102:1,103:1,104:1".into();
        root.range_ip = "211:1,212:1,213:1,214:1".into();
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 40;
        cfg.seed = 5;
        cfg.target_exploitability_bb = 0.0;
        let rep = solve(&root, &cfg).expect("ranged");
        assert_eq!(rep.status, "ok");
        assert!(
            rep.notes.iter().any(|n| n == "ranges=parsed oop=parsed:4 ip=parsed:4"),
            "{:?}",
            rep.notes
        );
        // Only the named combos ever get an infoset.
        for is in &rep.strategy.infosets {
            let raw = is.raw_combo.expect("raw combo");
            let want: &[u32] = if is.actor == Some(0) {
                &[100, 102, 103, 104]
            } else {
                &[211, 212, 213, 214]
            };
            assert!(want.contains(&raw), "actor {:?} combo {raw}", is.actor);
        }

        // The old, fully board-blocked IP range is now an error.
        root.range_ip = "200:1,201:1,202:1,203:1".into();
        assert!(matches!(solve(&root, &cfg), Err(CfrError::InvalidRoot(_))));
        // No range given → labelled as the uniform fallback.
        root.range_ip = String::new();
        root.range_oop = String::new();
        let rep = solve(&root, &cfg).expect("uniform");
        assert!(rep.notes.iter().any(|n| n.starts_with("ranges=uniform_fallback")));
    }
}
