//! PLO5 double-board bomb-pot game engine (+ NLH CFR scaffold).
//!
//! Modules:
//! - [`cards`]: card/deck representation with a pinned RNG for reproducibility.
//! - [`hand_eval`]: 5-card rank evaluator and PLO5 "2 from hand + 3 from board" evaluator.
//! - [`double_board`]: side-pot-layered payout distribution across two boards.
//! - [`state`]: game state types.
//! - [`actions`]: discrete 8-action space.
//! - [`engine`]: state machine (deal, apply action, advance street, payouts).
//! - [`cfr`]: NLH preflop→river CFR solver (DCFR / MCCFR).
//! - [`bindings`]: PyO3 wrapper.

pub mod actions;
pub mod bindings;
pub mod cards;
pub mod cfr;
pub mod double_board;
pub mod engine;
pub mod flush;
pub mod hand_eval;
pub mod state;

use pyo3::prelude::*;

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::bindings::PyGameState>()?;
    m.add_class::<crate::bindings::PyBatchedEngine>()?;
    m.add_function(pyo3::wrap_pyfunction!(
        crate::bindings::compute_double_board_payout,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        crate::bindings::compute_aggression_bonus_batch,
        m
    )?)?;
    // Compact rollout-observation storage (python/plo5bp/compact_obs.py).
    m.add_function(pyo3::wrap_pyfunction!(crate::bindings::pack_obs_rows, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(crate::bindings::unpack_obs_rows, m)?)?;
    // Rollout trajectory flush (python/plo5bp/rollout.py step9b-9d).
    m.add_function(pyo3::wrap_pyfunction!(crate::flush::flush_trajectories, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(crate::flush::record_learner_steps, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(crate::flush::gather_rows_into, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        crate::bindings::straight_flush_features_batch,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        crate::bindings::cross_board_straight_batch,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        crate::bindings::draw_flags_batch,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        crate::bindings::pair_features_batch,
        m
    )?)?;
    // Observation-semantics revision this binary reads from PLO5BP_OBS_REV
    // (review 2026-09-20); encoding.py cross-checks it at import.
    m.add_function(pyo3::wrap_pyfunction!(
        crate::bindings::obs_semantics_rev,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(crate::cfr::py_api::cfr_solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(crate::cfr::py_api::cfr_solve_kuhn, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        crate::cfr::py_api::cfr_induce_range,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(crate::cfr::py_api::cfr_pipeline, m)?)?;
    Ok(())
}
