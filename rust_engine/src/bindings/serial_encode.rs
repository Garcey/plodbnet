//! One-row observation encode for the SERIAL env (2026-09-28, ML-008).
//!
//! `BombPotEnv` (python/plo5bp/env.py: the site's Study / Trainer, eval, the
//! exploit probe) encoded every decision with the scalar numpy encoder -- a
//! THIRD implementation of every feature, next to the batched numpy one and
//! this engine's. `encode_game_state` runs THE engine encoder (the one
//! training uses, `BatchedEngine.encode`) on the serial state instead: a
//! one-env engine view over a clone of the state, at the serial 1024-sample
//! opp-outcome budget. Bit-exact with the scalar numpy encoder, which stays as
//! the test oracle (tests/python/test_serial_encode.py); a new feature now
//! needs one implementation here plus that parity test.

use super::*;

/// The full layout's opp-outcome block: 12 joint fractions, 8 per-board, 2
/// share bounds (`observation_dict`'s `opp_outcome_fractions`,
/// `per_board_outcome`, `share_bounds`, in that order).
const OUTCOME_BLOCK: usize = 22;

/// `encode_game_state(state, layout="full", opp_outcome_mc=1024, outcome=None,
/// obs_rev=None)` -> the current actor's observation, float32 `(dim,)`: the
/// "full" (1171) or "minimal" (796) layout.
///
/// - `outcome`: the full layout's opp-outcome block the caller already holds
///   -- `state.observation_dict()`'s `opp_outcome_fractions` (12) +
///   `per_board_outcome` (8) + `share_bounds` (2) -- used instead of running
///   the Monte Carlo again, so a caller that builds that dict anyway (the
///   serial env) runs it once per decision; `opp_outcome_mc` is then unused.
///   It IS the value this encode would compute (the same 1024-sample pass,
///   pinned equal).
/// - `obs_rev`: the observation-semantics revision to encode at; default the
///   state's own. The serial env passes the revision its Python side reads,
///   as its numpy encoders did.
///
/// PLO variants whose hands keep their size (PLO4/5/6); NLH encodes through
/// numpy and PLO67's hands grow mid-hand (both are refused). A state with no
/// actor (terminal, or a study street boundary) encodes as zeros, like the
/// batched encoders' finished rows.
#[pyfunction]
#[pyo3(signature = (state, layout="full", opp_outcome_mc=1024, outcome=None, obs_rev=None))]
pub fn encode_game_state<'py>(
    py: Python<'py>,
    state: PyRef<'py, PyGameState>,
    layout: &str,
    opp_outcome_mc: usize,
    outcome: Option<Vec<f32>>,
    obs_rev: Option<u8>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let layout = Layout::parse(layout)?;
    let g = state.get()?.clone();
    let config = state.config.clone();
    let obs_rev = match obs_rev {
        Some(rev) => check_obs_rev(rev)?,
        None => state.obs_rev,
    };
    drop(state);
    if !config.variant.supports_batched() {
        return Err(PyValueError::new_err(format!(
            "encode_game_state: {} hands grow mid-hand -- the engine encoder lays out \
             fixed-size hands only (the serial env keeps its numpy encoder there)",
            config.variant.name()
        )));
    }
    let outcome: Option<[f32; OUTCOME_BLOCK]> = match outcome {
        None => None,
        Some(_) if layout != Layout::Full => {
            return Err(PyValueError::new_err(
                "encode_game_state: `outcome` is the full layout's opp-outcome block; \
                 the minimal layout has none",
            ))
        }
        Some(v) => Some(v.as_slice().try_into().map_err(|_| {
            PyValueError::new_err(format!(
                "encode_game_state: `outcome` must hold {OUTCOME_BLOCK} floats (12 joint \
                 fractions + 8 per-board + 2 share bounds), got {}",
                v.len()
            ))
        })?),
    };
    let mc = if outcome.is_some() { 0 } else { opp_outcome_mc };
    let mut eng = PyBatchedEngine::with_config(config, 1, mc, obs_rev);
    eng.require_plo("encode_game_state")?;
    eng.states[0] = Some(g);
    let mut row = vec![0f32; layout.dim()];
    py.detach(|| {
        let mut packed = eng.pack_for(layout, &[0]);
        if let (Some(block), PackedRows::Full(f)) = (outcome.as_ref(), &mut packed) {
            // The pack ran no MC (mc = 0: zeros); these are the caller's.
            let parts = [
                (&mut f.obs.opp_outcome_fractions, 0..12),
                (&mut f.obs.per_board_outcome, 12..20),
                (&mut f.obs.share_bounds, 20..OUTCOME_BLOCK),
            ];
            for (dst, cols) in parts {
                for (d, s) in dst.row_mut(0).iter_mut().zip(&block[cols]) {
                    *d = *s;
                }
            }
        }
        eng.encode_rows(&packed, Some(vec![&mut row[..]]), None, |_| true)
    })
    .map_err(|(_, col, v)| {
        PyRuntimeError::new_err(format!("encode_game_state: column {col} holds {v}"))
    })?;
    Ok(Array1::from_vec(row).into_pyarray(py))
}
