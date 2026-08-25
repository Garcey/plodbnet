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
    m.add_function(pyo3::wrap_pyfunction!(crate::cfr::py_api::cfr_solve, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(crate::cfr::py_api::cfr_solve_kuhn, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        crate::cfr::py_api::cfr_induce_range,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(crate::cfr::py_api::cfr_pipeline, m)?)?;
    Ok(())
}
