//! Memory estimate + refuse-to-OOM for solve configs.
//!
//! (review 2026-09-20 F14) The estimate used to be a table of constants
//! (200 "public nodes" for any river, 128 bytes per infoset) that ignored the
//! action menu and the stack depth: it said 97.7 MB for a solve that grew
//! ~8.5M infosets. It now walks the REAL public betting tree of the root
//! (menu, stacks, min-raise rules, runout streets) and counts decision nodes
//! per street; the infoset bound is `nodes × private views × boards`.

use super::actions::{apply_abstract, legal_actions};
use super::card_abs::FLOP_BUCKETS;
use super::public_state::PublicState;
use super::types::{RootSpec, SolveConfig, StreetRoot};
use super::CfrError;

/// Live bytes per stored infoset, excluding per-action / per-path-step parts:
/// `Infoset` (3 Vec headers + dump `Option`) + `InfosetDump` (5 Vec/String
/// headers and their small heap blocks) + the `HashMap` slot and key.
pub const BYTES_PER_INFOSET_BASE: u64 = 640;
/// action enum + regret + strategy_sum per action.
const BYTES_PER_ACTION: u64 = 24;
/// One `String` path label (header + heap) per history step in the dump.
const BYTES_PER_PATH_STEP: u64 = 40;
/// Soft default RAM budget if not overridden (8 GB).
pub const DEFAULT_RAM_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Stop walking the public tree past this many decision nodes (per street
/// totals are then lower bounds and the solve is refused as unbounded).
const TREE_WALK_NODE_CAP: u64 = 5_000_000;
/// Longest action line the walk follows (real trees are < 40 deep).
const TREE_WALK_MAX_DEPTH: u64 = 200;

/// Decision nodes of the public betting tree, per street index (0..=3), for
/// ONE runout (chance nodes are followed along a single representative card;
/// every card gives a structurally identical subtree).
#[derive(Debug, Clone, Default)]
pub struct TreeCount {
    pub nodes_by_street: [u64; 4],
    /// Σ over nodes of (BASE + actions + path) bytes, per street.
    pub bytes_by_street: [u64; 4],
    /// True when the walk hit [`TREE_WALK_NODE_CAP`] (counts are lower bounds).
    pub truncated: bool,
}

impl TreeCount {
    pub fn total_nodes(&self) -> u64 {
        self.nodes_by_street.iter().sum()
    }
}

fn walk(
    state: &PublicState,
    raise_pm: &[u32],
    allin: bool,
    follow_runouts: bool,
    depth: u64,
    out: &mut TreeCount,
) {
    if out.truncated {
        return;
    }
    if depth > TREE_WALK_MAX_DEPTH {
        // Min-raise ladders (tiny pot fractions on deep stacks) grow one level
        // per big blind; refuse instead of recursing without bound.
        out.truncated = true;
        return;
    }
    if follow_runouts && state.needs_runout() {
        // Representative next card: lowest index not on the board.
        let blen = state.board_len as usize;
        let card = (0..52u8).find(|c| !state.board[..blen].contains(c)).unwrap_or(0);
        let mut child = state.clone();
        child.deal_board_card(card);
        return walk(&child, raise_pm, allin, follow_runouts, depth, out);
    }
    if state.is_terminal() || state.actor.is_none() {
        return;
    }
    let acts = legal_actions(state, raise_pm, allin);
    if acts.is_empty() {
        return;
    }
    let street = (state.street as usize).min(3);
    out.nodes_by_street[street] += 1;
    out.bytes_by_street[street] +=
        BYTES_PER_INFOSET_BASE + BYTES_PER_ACTION * acts.len() as u64 + BYTES_PER_PATH_STEP * depth;
    if out.total_nodes() >= TREE_WALK_NODE_CAP {
        out.truncated = true;
        return;
    }
    for act in acts {
        let mut child = state.clone();
        if apply_abstract(&mut child, act).is_ok() {
            walk(&child, raise_pm, allin, follow_runouts, depth + 1, out);
        }
    }
}

/// Count the public tree of a root state. `follow_runouts = false` for
/// PREFLOP roots: the trained preflop game ends when the preflop round closes.
pub fn count_public_tree(
    root_state: &PublicState,
    raise_pm: &[u32],
    allin: bool,
    follow_runouts: bool,
) -> TreeCount {
    let mut out = TreeCount::default();
    walk(root_state, raise_pm, allin, follow_runouts, 0, &mut out);
    out
}

fn root_state_for(root: &RootSpec) -> Result<PublicState, CfrError> {
    match (root.street, root.num_seats) {
        (StreetRoot::Preflop, 2) => super::mccfr::hu_preflop_root(root),
        (StreetRoot::Preflop, _) => super::mccfr::mw_preflop_root(root),
        (_, 2) => PublicState::hu_postflop_root(
            root.pot_chips()?,
            root.effective_stack_chips()?,
            &root.board,
            root.bb_chips,
        ),
        _ => PublicState::postflop_root(
            root.num_seats,
            root.pot_chips()?,
            &root.seat_stacks_chips()?,
            &root.board,
            root.bb_chips,
            root.street as u8,
        ),
    }
}

/// True when the config asks for exact (1326-combo) private views.
fn exact_cards(config: &SolveConfig) -> bool {
    matches!(config.card_abstraction.as_str(), "" | "none" | "exact")
}

/// Estimate infoset count and memory for a root under config.
///
/// `est_infosets` is an UPPER bound on the table (every public node × every
/// private view × every runout), additionally capped by what `max_iterations`
/// sampled deals can touch when that is finite.
pub fn estimate_solve_memory(root: &RootSpec, config: &SolveConfig) -> MemoryEstimate {
    let tree = match root_state_for(root) {
        Ok(st) => count_public_tree(
            &st,
            &root.raise_sizes_pm,
            root.allin_atom,
            root.street != StreetRoot::Preflop,
        ),
        Err(_) => TreeCount::default(), // invalid roots are rejected elsewhere
    };
    let hu = root.num_seats == 2;
    // Private views per (player, public node) — mirrors `RiverSolver::new`.
    let wants_buckets = matches!(config.card_abstraction.as_str(), "ochs" | "buckets" | "ehs");
    let bucketed = hu
        && (root.street == StreetRoot::Flop || (root.street == StreetRoot::Turn && wants_buckets));
    let multiway_class_view =
        !hu && root.street != StreetRoot::River && config.card_abstraction != "none";
    let private_views: u64 = if root.street == StreetRoot::Preflop || multiway_class_view {
        169
    } else if bucketed {
        FLOP_BUCKETS as u64
    } else {
        // C(52 - board, 2) live combos on that street's board.
        1_326
    };
    let root_street = root.street as usize;
    let mut est_infosets = 0u64;
    let mut est_bytes = 0u64;
    let mut one_runout_bytes = 0u64;
    let mut per_runout_nodes = 0u64;
    for street in 0..4usize {
        let nodes = tree.nodes_by_street[street];
        if nodes == 0 {
            continue;
        }
        per_runout_nodes += nodes;
        // Distinct public boards this street can show below the root.
        let boards: u64 = if root.street == StreetRoot::Preflop {
            1 // preflop solves end at the close of the preflop round
        } else {
            let mut b = 1u64;
            let mut cards_left = 52 - root.board.len() as u64;
            for _ in root_street..street {
                b = b.saturating_mul(cards_left);
                cards_left -= 1;
            }
            b
        };
        let views = if street == 0 || multiway_class_view || bucketed {
            private_views
        } else {
            let live = 52 - (street as u64 + 2).min(5);
            live * (live - 1) / 2
        };
        let n = nodes.saturating_mul(boards).saturating_mul(views);
        est_infosets = est_infosets.saturating_add(n);
        let avg_bytes = tree.bytes_by_street[street] / nodes;
        est_bytes = est_bytes.saturating_add(n.saturating_mul(avg_bytes));
        one_runout_bytes = one_runout_bytes
            .saturating_add(nodes.saturating_mul(views).saturating_mul(avg_bytes));
    }
    // A finite run cannot create more infosets than it visits: each sampled
    // deal touches at most every decision node of ONE runout, once per seat
    // view. (HU DCFR runs `rayon_deals` deals per iteration.)
    if config.max_iterations > 0 && est_infosets > 0 {
        let deals = config.max_iterations as u64 * config.thread_num.clamp(1, 8) as u64;
        let visit_cap = deals
            .saturating_mul(per_runout_nodes)
            .saturating_mul(root.num_seats as u64);
        if visit_cap < est_infosets {
            let avg = est_bytes / est_infosets.max(1);
            est_infosets = visit_cap;
            est_bytes = visit_cap.saturating_mul(avg.max(BYTES_PER_INFOSET_BASE));
        }
    }
    MemoryEstimate {
        est_infosets,
        est_bytes,
        one_runout_bytes,
        private_views,
        card_abs: config.card_abstraction.clone(),
        public_nodes: tree.total_nodes(),
        tree_truncated: tree.truncated,
    }
}

#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Upper bound on the infoset table (capped by a finite `max_iterations`).
    pub est_infosets: u64,
    pub est_bytes: u64,
    /// Table bytes once a SINGLE runout is fully explored. On river roots this
    /// is the whole table; on flop/turn roots the table keeps growing with
    /// every new sampled runout (bounded at run time, see `refuse_if_unsafe`).
    pub one_runout_bytes: u64,
    pub private_views: u64,
    pub card_abs: String,
    /// Decision nodes of the public tree along one runout.
    pub public_nodes: u64,
    /// The tree walk hit its node cap: the counts are lower bounds.
    pub tree_truncated: bool,
}

impl MemoryEstimate {
    pub fn mb(&self) -> f64 {
        self.est_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Refuse exact flop (or other unsafe configs) before building the tree.
pub fn refuse_if_unsafe(root: &RootSpec, config: &SolveConfig) -> Result<MemoryEstimate, CfrError> {
    let est = estimate_solve_memory(root, config);

    // Hard rule from master plan: exact HU flop is refused.
    // (review 2026-09-20 F14) `card_abstraction="exact"` used to slip past
    // this check (it only looked for "none"/"") while `RiverSolver::new`
    // honoured it by turning the buckets OFF — an exact 1326-combo flop solve.
    if root.street == StreetRoot::Flop && root.num_seats == 2 && exact_cards(config) {
        return Err(CfrError::InvalidConfig(format!(
            "exact flop refused (est {:.0} MB private views={}); set card_abstraction to \"ochs\" or \"buckets\"",
            est.mb(),
            est.private_views
        )));
    }

    if est.tree_truncated {
        return Err(CfrError::InvalidConfig(format!(
            "public betting tree exceeds {TREE_WALK_NODE_CAP} decision nodes per runout; \
             memory cannot be bounded — use fewer raise sizes or shallower stacks"
        )));
    }
    // Refuse when even ONE fully explored runout cannot fit (on river roots
    // that is the exact table). Flop/turn tables grow with each newly sampled
    // runout and how many get visited depends on the run length, so those are
    // bounded at run time instead: the solve loop stops with
    // `early_stop=memory_budget` once the live table passes the budget.
    if est.one_runout_bytes > DEFAULT_RAM_BUDGET_BYTES {
        return Err(CfrError::InvalidConfig(format!(
            "estimated memory {:.1} GB ({} public nodes x {} private views) exceeds budget {:.1} GB; \
             use coarser sizes (micro) or shallower stacks",
            est.one_runout_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            est.public_nodes,
            est.private_views,
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
    use crate::cfr::types::{RootSpec, SolveConfig, StreetRoot, DEFAULT_RAISE_SIZES_PM};

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
        // (review 2026-09-20 F14) the "exact" spelling is refused too.
        cfg.card_abstraction = "exact".into();
        assert!(refuse_if_unsafe(&root, &cfg).is_err());
        assert!(crate::cfr::solve(&root, &cfg).is_err());
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

    /// (review 2026-09-20 F14) the estimate follows the real tree: more sizes
    /// and deeper stacks ⇒ more public nodes ⇒ more infosets; and it bounds
    /// what a solve actually allocates.
    #[test]
    fn estimate_tracks_the_real_tree() {
        let board = vec![0u8, 5, 10, 15, 20];
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 0;
        cfg.time_budget_secs = 1.0;
        let small = RootSpec::postflop_hu(StreetRoot::River, 10.0, 10.0, board.clone(), vec![]);
        let big = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            100.0,
            board.clone(),
            DEFAULT_RAISE_SIZES_PM.to_vec(),
        );
        let (es, eb) = (
            estimate_solve_memory(&small, &cfg),
            estimate_solve_memory(&big, &cfg),
        );
        // Jam/check river: x, x/x.., x/AI, AI → 4 decision nodes.
        assert_eq!(es.public_nodes, 4, "{es:?}");
        assert!(eb.public_nodes > 50 * es.public_nodes, "{eb:?}");
        assert!(eb.est_bytes > 50 * es.est_bytes);
        assert_eq!(es.est_infosets, 4 * (47 * 46 / 2));

        // The bound holds against a real solve.
        let mut run = SolveConfig::default();
        run.max_iterations = 400;
        run.target_exploitability_bb = 0.0;
        run.use_isomorphism = false;
        let root = RootSpec::postflop_hu(StreetRoot::River, 10.0, 20.0, board, vec![500, 1000]);
        let est = estimate_solve_memory(&root, &run);
        let rep = crate::cfr::solve(&root, &run).unwrap();
        assert!(
            rep.strategy.infosets.len() as u64 <= est.est_infosets,
            "solve made {} infosets, estimate bound {}",
            rep.strategy.infosets.len(),
            est.est_infosets
        );
    }

    /// Trees that cannot be bounded are refused instead of walked/solved:
    /// a min-raise ladder (0.1%-pot "raises" on 1000 bb) is one level per bb.
    #[test]
    fn unbounded_trees_are_refused() {
        let ladder = RootSpec::postflop_hu(
            StreetRoot::River,
            1.0,
            1000.0,
            vec![0, 5, 10, 15, 20],
            vec![1],
        );
        let cfg = SolveConfig::default();
        let est = estimate_solve_memory(&ladder, &cfg);
        assert!(est.tree_truncated, "{est:?}");
        assert!(refuse_if_unsafe(&ladder, &cfg).is_err());
        assert!(crate::cfr::solve(&ladder, &cfg).is_err());
        // A deep standard-size river exceeds the RAM budget outright.
        let deep = RootSpec::postflop_hu(
            StreetRoot::River,
            5.0,
            200.0,
            vec![0, 5, 10, 15, 20],
            DEFAULT_RAISE_SIZES_PM.to_vec(),
        );
        let err = refuse_if_unsafe(&deep, &cfg).unwrap_err();
        assert!(format!("{err}").contains("exceeds budget"), "{err}");
    }

    /// A finite iteration count caps the estimate (flop solves stay allowed).
    #[test]
    fn finite_iterations_cap_the_estimate() {
        let root = RootSpec::postflop_hu(
            StreetRoot::Flop,
            8.0,
            20.0,
            vec![0, 5, 10],
            vec![500, 1000],
        );
        let mut cfg = SolveConfig::default();
        cfg.card_abstraction = "ochs".into();
        cfg.max_iterations = 60;
        let capped = estimate_solve_memory(&root, &cfg);
        cfg.max_iterations = 0;
        cfg.time_budget_secs = 5.0;
        let unbounded = estimate_solve_memory(&root, &cfg);
        assert!(capped.est_infosets < unbounded.est_infosets);
        assert!(refuse_if_unsafe(&root, &{
            let mut c = cfg.clone();
            c.max_iterations = 60;
            c
        })
        .is_ok());
    }
}
