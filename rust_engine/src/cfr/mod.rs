//! NLH CFR solver (preflop → river, multiway-capable), ClubGG chip rules.
//!
//! - DCFR for HU postflop (flop/turn with runouts, river exact)
//! - External-sampling MCCFR for preflop / multiway
//! - Compact [`public_state::PublicState`] (engine parity via engine_bridge)

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
        let rep = known_spot::solve_jam_check_spot(600, 3).expect("spot");
        assert!(
            rep.exploitability_bb < 4.0,
            "known spot expl {}",
            rep.exploitability_bb
        );
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
        // Restrict to a few combos
        root.range_oop = "100:1,101:1,102:1,103:1".into();
        root.range_ip = "200:1,201:1,202:1,203:1".into();
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 40;
        cfg.seed = 5;
        cfg.target_exploitability_bb = 0.0;
        let rep = solve(&root, &cfg).expect("ranged");
        assert_eq!(rep.status, "ok");
        assert!(rep.notes.iter().any(|n| n.contains("parsed")));
    }
}
