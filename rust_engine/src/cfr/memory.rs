//! Memory estimate + refuse-to-OOM for solve configs.

use super::types::{RootSpec, SolveConfig, StreetRoot};
use super::CfrError;
use super::card_abs::FLOP_BUCKETS;

/// Rough bytes per infoset (regrets + strategy_sum for ~6 actions).
const BYTES_PER_INFOSET: u64 = 128;
/// Soft default RAM budget if not overridden (8 GB).
pub const DEFAULT_RAM_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Estimate infoset count and memory for a root under config.
pub fn estimate_solve_memory(root: &RootSpec, config: &SolveConfig) -> MemoryEstimate {
    let n = root.num_seats as u64;
    let actions = (2 + root.raise_sizes_pm.len() as u64 + if root.allin_atom { 1 } else { 0 }).max(2);
    // Public tree branching crude: ~ actions^depth; depth ~ 4-8 betting rounds
    let public_nodes = match root.street {
        StreetRoot::River => 200u64,
        StreetRoot::Turn => 800,
        StreetRoot::Flop => 3_000,
        StreetRoot::Preflop => 500,
    };
    let private_views = match (root.street, config.card_abstraction.as_str()) {
        (StreetRoot::River, _) => 1_000u64, // ~ unblocked combos scale, sampled
        (StreetRoot::Turn, "none") => 1_200,
        (StreetRoot::Flop, "none") => 1_326, // exact — will refuse
        (StreetRoot::Flop, _) => FLOP_BUCKETS as u64,
        (StreetRoot::Turn, _) => FLOP_BUCKETS as u64,
        (StreetRoot::Preflop, _) => 169,
    };
    let infosets = public_nodes * private_views * n.min(2);
    let bytes = infosets * BYTES_PER_INFOSET * actions / 4; // rough
    MemoryEstimate {
        est_infosets: infosets,
        est_bytes: bytes,
        private_views,
        card_abs: config.card_abstraction.clone(),
    }
}

#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub est_infosets: u64,
    pub est_bytes: u64,
    pub private_views: u64,
    pub card_abs: String,
}

impl MemoryEstimate {
    pub fn mb(&self) -> f64 {
        self.est_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Refuse exact flop (or other unsafe configs) before building the tree.
pub fn refuse_if_unsafe(root: &RootSpec, config: &SolveConfig) -> Result<MemoryEstimate, CfrError> {
    let est = estimate_solve_memory(root, config);

    // Hard rule from master plan: exact HU flop is refused
    if root.street == StreetRoot::Flop
        && root.num_seats == 2
        && (config.card_abstraction == "none" || config.card_abstraction.is_empty())
    {
        return Err(CfrError::InvalidConfig(format!(
            "exact flop refused (est {:.0} MB private views={}); set card_abstraction to \"ochs\" or \"buckets\"",
            est.mb(),
            est.private_views
        )));
    }

    // Multiway exact river with many seats + fine sizes: soft warn via large estimate
    if est.est_bytes > DEFAULT_RAM_BUDGET_BYTES {
        return Err(CfrError::InvalidConfig(format!(
            "estimated memory {:.1} GB exceeds budget {:.1} GB; use coarser sizes (micro) or card_abstraction=ochs",
            est.est_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            DEFAULT_RAM_BUDGET_BYTES as f64 / (1024.0 * 1024.0 * 1024.0)
        )));
    }
    Ok(est)
}

/// Default card abstraction for a street when user left "none" / empty.
pub fn default_card_abs_for_street(street: StreetRoot) -> &'static str {
    match street {
        StreetRoot::Flop => "ochs",
        StreetRoot::Turn => "none", // exact turn usually OK
        StreetRoot::River => "none",
        StreetRoot::Preflop => "preflop169",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::types::{RootSpec, SolveConfig, StreetRoot};

    #[test]
    fn refuse_exact_flop() {
        let root = RootSpec::postflop_hu(
            StreetRoot::Flop,
            10.0,
            50.0,
            vec![0, 1, 2],
            vec![500, 1000],
        );
        let mut cfg = SolveConfig::default();
        cfg.card_abstraction = "none".into();
        assert!(refuse_if_unsafe(&root, &cfg).is_err());
    }

    #[test]
    fn allow_bucketed_flop() {
        let root = RootSpec::postflop_hu(
            StreetRoot::Flop,
            10.0,
            50.0,
            vec![0, 1, 2],
            vec![500, 1000],
        );
        let mut cfg = SolveConfig::default();
        cfg.card_abstraction = "ochs".into();
        assert!(refuse_if_unsafe(&root, &cfg).is_ok());
    }

    #[test]
    fn allow_river_exact() {
        let root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            50.0,
            vec![0, 1, 2, 3, 4],
            vec![500, 1000],
        );
        let cfg = SolveConfig::default();
        assert!(refuse_if_unsafe(&root, &cfg).is_ok());
    }
}
