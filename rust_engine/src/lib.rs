//! PLO5 double-board bomb-pot game engine.
//!
//! Modules:
//! - [`cards`]: card/deck representation with a pinned RNG for reproducibility.
//! - [`hand_eval`]: 5-card rank evaluator and PLO5 "2 from hand + 3 from board" evaluator.
//! - [`double_board`]: side-pot-layered payout distribution across two boards.
//! - [`state`]: game state types.
//! - [`actions`]: discrete 8-action space.
//! - [`engine`]: state machine (deal, apply action, advance street, payouts).
//! - [`bindings`]: PyO3 wrapper.

pub mod actions;
pub mod bindings;
pub mod cards;
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
    Ok(())
}
