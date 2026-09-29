//! The poker engine behind WrapGTO: PLO4 / PLO5 / PLO6 / PLO67 double-board bomb
//! pots and single-board no-limit hold'em — rules, evaluation, settlement, the
//! observation encoders and rollout kernels the trainer drives, and the NLH CFR
//! solver — exposed to Python as `plo5bp._engine`.
//!
//! Modules:
//! - [`cards`]: card/deck representation with a pinned RNG for reproducibility.
//! - [`hand_eval`]: 5-card rank evaluator, the PLO "exactly 2 from hand + 3 from
//!   board" evaluators (4..=7 hole cards) and NLH's any-5-of-7.
//! - [`double_board`]: side-pot-layered payouts on one or two boards.
//! - [`state`]: game state types, variants and configs.
//! - [`actions`]: the discrete 8-action space.
//! - [`engine`]: the state machine (deal, apply action, advance street, payouts).
//! - [`obs_features`]: observation features of a state (categories, v7 dims,
//!   opp-outcome MC, NLH sweep).
//! - [`flush`]: rollout trajectory kernels (flush, per-step record, row gathers).
//! - [`cfr`]: the NLH preflop→river CFR solver (DCFR / MCCFR).
//! - [`bindings`]: the PyO3 classes and functions; the module itself is `_engine`
//!   below.

pub mod actions;
pub mod bindings;
pub mod cards;
pub mod cfr;
pub mod double_board;
pub mod engine;
pub mod flush;
pub mod hand_eval;
pub mod obs_features;
pub mod state;
#[cfg(test)]
pub(crate) mod test_util;

// The Python module: every class and function Python sees is exported here
// (ENG-022: PyO3's declarative module, no hand-written registration calls).
/// WrapGTO's poker engine (Rust): game rules and settlement, the observation
/// encoders and rollout kernels, and the NLH CFR solver.
#[pyo3::pymodule]
mod _engine {
    #[pymodule_export]
    use crate::bindings::{PyBatchedEngine, PyGameState};

    // Settlement, the PLO67 home-game runouts and the observation-semantics
    // revision this binary reads from PLO5BP_OBS_REV (encoding.py cross-checks
    // it at import).
    #[pymodule_export]
    use crate::bindings::{compute_double_board_payout, obs_semantics_rev, plo67_runout_equities};

    // Standalone feature kernels (the fused encoders call the same code).
    #[pymodule_export]
    use crate::bindings::{
        cross_board_straight_batch, draw_flags_batch, pair_features_batch,
        plo_board_strength_batch, straight_flush_features_batch,
    };

    // Rollout: aggression bonus, compact observation storage
    // (python/plo5bp/compact_obs.py), the trajectory flush and row gathers
    // (python/plo5bp/rollout.py).
    #[pymodule_export]
    use crate::bindings::{
        aggression_record_batch, compute_aggression_bonus_batch, pack_obs_rows, unpack_obs_rows,
    };
    #[pymodule_export]
    use crate::flush::{
        flush_trajectories, gather_rows_into, gather_rows_multi, record_learner_steps,
    };

    // The serial env's one-row observation encode (python/plo5bp/env.py, ML-008).
    #[pymodule_export]
    use crate::bindings::encode_game_state;

    // The NLH CFR solver (python/plo5bp/gto/).
    #[pymodule_export]
    use crate::cfr::py_api::{
        cfr_estimate_memory, cfr_induce_range, cfr_parse_range, cfr_pipeline, cfr_solve,
        cfr_solve_kuhn,
    };

    #[pymodule_init]
    fn init(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        use pyo3::types::PyModuleMethods;
        // A hash of the sources this module was built from (build.rs): the test
        // suite compares it with the sources on disk to catch a stale build
        // (TEST-036).
        m.add("SOURCE_HASH", env!("PLO5_SOURCE_HASH"))
    }
}
