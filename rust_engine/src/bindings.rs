//! PyO3 bindings: everything Python sees of the engine (`plo5bp._engine`).
//!
//! This file holds the shared helpers (variant parsing, the observation-
//! semantics revision switch, table / index / card validation); each Python
//! concept lives in its own file under `bindings/`:
//!
//! - [`serial`]: `GameState` — one hand, one action at a time (UI, study,
//!   trainer, home games, the serial env) + `plo67_runout_equities`;
//! - [`batched`]: `BatchedEngine` — N hands per FFI call for the rollouts
//!   (the one `#[pymethods]` block);
//! - [`pack`]: engine state -> the stacked arrays the encoders read;
//! - [`encode`]: plumbing shared by both observation layouts (raise windows,
//!   in-place / packed output rows, aux dicts, the batched encode drivers);
//! - [`encode_full`] / [`encode_minimal`]: the 1171- and 796-dim layouts and
//!   their per-row encoders;
//! - [`features`]: the standalone feature kernels + pyfunctions (straight /
//!   flush, cross-board, draw flags, pair structure, board strength, payouts);
//! - [`compact`]: compact (bit-packed) rollout-observation rows;
//! - [`rollout_ops`]: the aggression-bonus rollout kernels.
//!
//! Every child does `use super::*;` and this module glob-imports each child's
//! `pub(super)` items, so the files share one namespace exactly as the single
//! file did; the Python-facing items are re-exported under their old paths.

use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::actions::{Action, NUM_ACTIONS};
use crate::cards::{Card, CardMask};
use crate::state::{GameConfig, GameState, StudyTerminal, Variant};

/// A 1-D float32 numpy array handed back to Python.
pub(crate) type F32Vec<'py> = Bound<'py, PyArray1<f32>>;
/// A 2-D float32 numpy array handed back to Python.
pub(crate) type F32Mat<'py> = Bound<'py, PyArray2<f32>>;

/// Parse the Python-facing variant string. Kept as strings (not an
/// exported enum class) so the Python config layer stays a plain
/// dataclass field.
fn parse_variant(s: &str) -> PyResult<Variant> {
    Variant::from_name(s).ok_or_else(|| {
        let names: Vec<String> = Variant::ALL
            .iter()
            .map(|v| format!("'{}'", v.name()))
            .collect();
        PyValueError::new_err(format!(
            "unknown variant '{s}' (expected one of {})",
            names.join(", ")
        ))
    })
}

/// The hole widths the batched PLO kernels take: those of the PLO variants
/// the batched engine deals (4, 5 and 6 cards).
fn batched_plo_hole_width(w: usize) -> bool {
    Variant::ALL
        .into_iter()
        .any(|v| v.is_plo() && v.supports_batched() && v.hole_count() == w)
}

// ---- observation-SEMANTICS revision switch (PLO5BP_OBS_REV) ----------------
// Twin of `OBS_SEMANTICS_REV` in python/plo5bp/encoding.py — read that block
// comment first. The 2026-09-20 review fixed features whose VALUES were wrong
// while the layout stayed put; a checkpoint is only served / resumed exactly on
// the semantics it was trained on, so the old values stay selectable:
//
//   PLO5BP_OBS_REV unset or "2"  (DEFAULT) — the fixed semantics.
//   PLO5BP_OBS_REV=1             — the pre-2026-09-20 values, bit-exact. Set it
//                                  to serve or resume a checkpoint trained
//                                  before 2026-09-20.
//
// Gated here: B1 (STK-2, STK-5[2:4]), B2 (draw flags 800/802), B3 (min/max
// scalars, full + minimal), B5 (blocker flush dims). B7 is NLH (numpy-only).
// NOT gated: B6, the STK-6 comparison chain, C4/C6/C7.
//
// The variable is read when a `GameState` / `BatchedEngine` is CONSTRUCTED and
// stored in its `obs_rev` field (an explicit `obs_rev=` constructor argument
// overrides it); the standalone feature pyfunctions take `obs_rev` explicitly.
// Python reads the same variable once at import and cross-checks it against
// `obs_semantics_rev()`. train.py stamps `obs_rev` into checkpoints and refuses
// a mismatched warm start; the UI warns on a mismatch.
const OBS_REV_ENV: &str = "PLO5BP_OBS_REV";
const OBS_REV_LEGACY: u8 = 1;
const OBS_REV_CURRENT: u8 = 2;

fn check_obs_rev(rev: u8) -> PyResult<u8> {
    if rev == OBS_REV_LEGACY || rev == OBS_REV_CURRENT {
        Ok(rev)
    } else {
        Err(PyValueError::new_err(format!(
            "obs_rev must be {OBS_REV_CURRENT} (default, the 2026-09-20 fixed features) or \
             {OBS_REV_LEGACY} (pre-2026-09-20 values), got {rev}"
        )))
    }
}

/// `PLO5BP_OBS_REV` parsed exactly like `encoding._read_obs_semantics_rev`:
/// unset, empty or whitespace-only (`PLO5BP_OBS_REV=` is common in .env files)
/// -> 2; "1" / "2" (surrounding whitespace ignored); anything else is an error
/// rather than a silent default.
fn obs_rev_from_env() -> PyResult<u8> {
    let raw = match std::env::var(OBS_REV_ENV) {
        Ok(v) => v,
        Err(std::env::VarError::NotPresent) => return Ok(OBS_REV_CURRENT),
        Err(std::env::VarError::NotUnicode(v)) => {
            return Err(PyValueError::new_err(format!(
                "{OBS_REV_ENV}={v:?} is not a known observation-semantics revision"
            )))
        }
    };
    match raw.trim() {
        "" => Ok(OBS_REV_CURRENT),
        "1" => Ok(OBS_REV_LEGACY),
        "2" => Ok(OBS_REV_CURRENT),
        _ => Err(PyValueError::new_err(format!(
            "{OBS_REV_ENV}={raw:?} is not a known observation-semantics revision: use \
             {OBS_REV_CURRENT} (default, the 2026-09-20 fixed features) or {OBS_REV_LEGACY} \
             (pre-2026-09-20 values, to serve/resume a checkpoint trained before that date)"
        ))),
    }
}

/// Constructor argument wins; otherwise the environment decides.
fn resolve_obs_rev(explicit: Option<u8>) -> PyResult<u8> {
    match explicit {
        Some(rev) => check_obs_rev(rev),
        None => obs_rev_from_env(),
    }
}

/// The observation-semantics revision the engine reads from `PLO5BP_OBS_REV`
/// right now (1 or 2). Python asserts at import that it equals
/// `encoding.OBS_SEMANTICS_REV`. Module-level registration lives in lib.rs;
/// the same value is reachable as `GameState.obs_semantics_rev()`.
#[pyfunction]
pub fn obs_semantics_rev() -> PyResult<u8> {
    obs_rev_from_env()
}

/// Seat cap of the observation encoders: every hero-rotated block is padded to
/// 8 slots (`encoding._MAX_SEATS`; the `[f64; 8]` effective-stack scratch in
/// `encode_obs_row*`). A 9th seat's active flag would land in all-in slot 0.
const MAX_SEATS: usize = 8;

/// Constructor-time table validation (review 2026-09-20 C4). Each case used to
/// surface later as a Rust panic — a `PanicException`, which derives from
/// `BaseException` and so escapes Python's `except Exception`: 1 seat ("need
/// at least 2 seats"), a deck overrun at PLO5 >= 9 / PLO6 >= 8 seats, and
/// PLO4 at 9 seats overrunning the 8-slot encoder arrays.
fn validate_table(num_seats: usize, variant: Variant, bb: u64) -> PyResult<()> {
    if !(2..=MAX_SEATS).contains(&num_seats) {
        return Err(PyValueError::new_err(format!(
            "num_seats must be in 2..={MAX_SEATS}, got {num_seats}"
        )));
    }
    if bb == 0 {
        // Every encoder scales by 1/bb: bb == 0 yields a non-finite obs.
        return Err(PyValueError::new_err("bb must be >= 1"));
    }
    let needed = variant.cards_needed(num_seats);
    if needed > crate::cards::DECK_SIZE {
        return Err(PyValueError::new_err(format!(
            "{num_seats} seats need {needed} cards for this variant; the deck has {}",
            crate::cards::DECK_SIZE
        )));
    }
    Ok(())
}

/// Bounds-checked env indices for every `*_subset_batch(indices)` entry point
/// (review 2026-09-20 C4): a negative index used to wrap to a huge `usize` and
/// an out-of-range one indexed past `states`, both panicking mid-pack.
fn checked_env_indices(indices: &[i64], n: usize, what: &str) -> PyResult<Vec<usize>> {
    indices
        .iter()
        .map(|&x| {
            if x < 0 || x as usize >= n {
                Err(PyValueError::new_err(format!(
                    "{what}: index {x} out of range (num_envs={n})"
                )))
            } else {
                Ok(x as usize)
            }
        })
        .collect()
}

fn resolve_starting_stacks(
    num_seats: usize,
    starting_stack: u64,
    starting_stacks: Option<PyReadonlyArray1<'_, u64>>,
) -> PyResult<Vec<u64>> {
    match starting_stacks {
        Some(arr) => {
            let slice = arr.as_slice()?;
            if slice.len() != num_seats {
                return Err(PyValueError::new_err(format!(
                    "starting_stacks length {} != num_seats {}",
                    slice.len(),
                    num_seats
                )));
            }
            Ok(slice.to_vec())
        }
        None => Ok(vec![starting_stack; num_seats]),
    }
}

fn card_from_index(i: u8) -> PyResult<Card> {
    if i >= 52 {
        return Err(PyValueError::new_err(format!(
            "card index {i} out of range"
        )));
    }
    Ok(Card::from_index(i))
}

fn cards_from_indices<const N: usize>(indices: &[u8], label: &str) -> PyResult<[Card; N]> {
    if indices.len() != N {
        return Err(PyValueError::new_err(format!(
            "{label} must have {N} indices, got {}",
            indices.len()
        )));
    }
    let mut out = [Card(0); N];
    for (i, &idx) in indices.iter().enumerate() {
        out[i] = card_from_index(idx)?;
    }
    Ok(out)
}

mod batched;
/// Entry points for the Criterion benchmarks (rust_engine/bench), `bench` feature only.
#[cfg(feature = "bench")]
#[doc(hidden)]
pub mod bench_api;
mod compact;
mod encode;
mod encode_full;
mod encode_minimal;
mod features;
mod pack;
mod rollout_ops;
mod serial;
// The serial env's one-row engine encode (ML-008; the train agent's file).
mod serial_encode;

// One namespace for the binding files (see the module docs).
use batched::*;
use compact::*;
use encode::*;
use encode_full::*;
use encode_minimal::*;
use features::*;
use pack::*;

// The Python-facing items, at the paths lib.rs registers them from.
pub use batched::PyBatchedEngine;
pub use compact::{pack_obs_rows, unpack_obs_rows};
pub use features::{
    compute_double_board_payout, cross_board_straight_batch, draw_flags_batch, pair_features_batch,
    plo_board_strength_batch, straight_flush_features_batch,
};
pub use rollout_ops::{aggression_record_batch, compute_aggression_bonus_batch};
pub use serial::{plo67_runout_equities, PyGameState};
pub use serial_encode::encode_game_state;

#[cfg(test)]
mod encoder_tests;
// Golden digests of every engine output (TEST-021).
#[cfg(test)]
mod golden_tests;
