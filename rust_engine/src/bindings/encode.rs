//! Encoder plumbing shared by both layouts: raise windows, output-row
//! selection, packed (compact) sinks, aux dicts, and the batched encode drivers.

use super::*;

/// Disjoint mutable `width`-wide rows of `buf` at the strictly increasing
/// indices `idx` (one forward walk).
pub(super) fn pick_rows<'a, T>(buf: &'a mut [T], width: usize, idx: &[usize]) -> Vec<&'a mut [T]> {
    let mut rows: Vec<&mut [T]> = Vec::with_capacity(idx.len());
    let mut want = idx.iter().copied().peekable();
    for (r, row) in buf.chunks_exact_mut(width).enumerate() {
        match want.peek() {
            None => break,
            Some(&w) if w == r => {
                rows.push(row);
                want.next();
            }
            Some(_) => {}
        }
    }
    rows
}

/// How the in-place encoders pack each row (see [`pack_obs_row_runs`]).
pub(super) struct PackPlan {
    pub(super) flags: Vec<usize>,
    pub(super) reals: Vec<usize>,
    pub(super) flag_runs: Vec<(usize, usize, usize)>,
    pub(super) real_runs: Vec<(usize, usize, usize)>,
    pub(super) nb: usize,
    pub(super) nr: usize,
}

/// Validate the optional packed-output arguments of a `d`-wide layout: all
/// four or none; column indices inside the row; outputs C-contiguous (n, nb) /
/// (n, nr).
pub(super) fn pack_plan_for_width(
    what: &str,
    n: usize,
    d: usize,
    flag_cols: Option<PyReadonlyArray1<'_, i64>>,
    real_cols: Option<PyReadonlyArray1<'_, i64>>,
    out_bits: &Option<PyReadwriteArray2<'_, u8>>,
    out_real: &Option<PyReadwriteArray2<'_, f32>>,
) -> PyResult<Option<PackPlan>> {
    let (flag_cols, real_cols, out_bits, out_real) =
        match (flag_cols, real_cols, out_bits, out_real) {
            (None, None, None, None) => return Ok(None),
            (Some(f), Some(r), Some(b), Some(o)) => (f, r, b, o),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "{what}: flag_cols, real_cols, out_bits and out_real go together"
                )))
            }
        };
    let cols = |v: &[i64]| -> PyResult<Vec<usize>> {
        v.iter()
            .map(|&x| {
                if x < 0 || (x as usize) >= d {
                    Err(PyValueError::new_err(format!(
                        "{what}: column {x} out of range [0, {d})"
                    )))
                } else {
                    Ok(x as usize)
                }
            })
            .collect()
    };
    let flags = cols(flag_cols.as_slice()?)?;
    let reals = cols(real_cols.as_slice()?)?;
    let (nb, nr) = (flags.len().div_ceil(8), reals.len());
    if nb == 0 || nr == 0 {
        return Err(PyValueError::new_err(format!(
            "{what}: the layout needs at least one flag and one real column"
        )));
    }
    if !out_bits.is_c_contiguous() || out_bits.shape() != [n, nb] {
        return Err(PyValueError::new_err(format!(
            "{what}: out_bits must be a C-contiguous ({n}, {nb}) uint8 array; got {:?}",
            out_bits.shape()
        )));
    }
    if !out_real.is_c_contiguous() || out_real.shape() != [n, nr] {
        return Err(PyValueError::new_err(format!(
            "{what}: out_real must be a C-contiguous ({n}, {nr}) float32 array; got {:?}",
            out_real.shape()
        )));
    }
    Ok(Some(PackPlan {
        flag_runs: column_runs(&flags),
        real_runs: column_runs(&reals),
        flags,
        reals,
        nb,
        nr,
    }))
}

/// The packed-output buffers as mutable slices (None when not requested).
pub(super) fn packed_slices<'a>(
    out_bits: &'a mut Option<PyReadwriteArray2<'_, u8>>,
    out_real: &'a mut Option<PyReadwriteArray2<'_, f32>>,
) -> PyResult<Option<(&'a mut [u8], &'a mut [f32])>> {
    match (out_bits.as_mut(), out_real.as_mut()) {
        (Some(b), Some(r)) => Ok(Some((b.as_slice_mut()?, r.as_slice_mut()?))),
        _ => Ok(None),
    }
}

/// The in-place encoders need somewhere to write: the dense rows, the packed
/// copy, or both.
pub(super) fn require_some_output(what: &str, dense: bool, packed: bool) -> PyResult<()> {
    if dense || packed {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{what}: out=None needs the packed outputs (flag_cols / real_cols / out_bits / out_real)"
    )))
}

pub(super) fn pack_encoder_error(what: &str, env: usize, col: usize, v: f32) -> PyErr {
    PyValueError::new_err(format!(
        "{what}: env {env} obs column {col} holds {v:?}, not a 0/1 flag -- the compact \
         layout (python/plo5bp/compact_obs.py) no longer matches the encoder"
    ))
}

/// The aux fields every encoder returns next to (or instead of) "obs" (the
/// same for both layouts — both come from the packed core).
pub(super) fn aux_dict<'py>(
    py: Python<'py>,
    core: PackedCore,
    legal_mask: Array2<bool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("actor", core.actor.into_pyarray(py))?;
    d.set_item("legal_mask", legal_mask.into_pyarray(py))?;
    d.set_item("min_raise", core.min_raise.into_pyarray(py))?;
    d.set_item("max_raise", core.max_raise.into_pyarray(py))?;
    d.set_item("total_commit", core.total_commit.into_pyarray(py))?;
    d.set_item("bet_to_call", core.bet_to_call.into_pyarray(py))?;
    d.set_item("street_commit", core.street_commit.into_pyarray(py))?;
    d.set_item("street", core.street.into_pyarray(py))?;
    d.set_item("pot", core.pot.into_pyarray(py))?;
    Ok(d)
}

/// Which observation layout an encode call produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layout {
    /// The 1171-dim layout ([`encode_obs_row`]).
    Full,
    /// The 796-dim bare-visibility layout ([`encode_obs_row_minimal`]).
    Minimal,
}

impl Layout {
    pub(super) fn parse(s: &str) -> PyResult<Self> {
        match s {
            "full" => Ok(Layout::Full),
            "minimal" => Ok(Layout::Minimal),
            _ => Err(PyValueError::new_err(format!(
                "layout must be 'full' or 'minimal', got {s:?}"
            ))),
        }
    }

    pub(super) fn dim(self) -> usize {
        match self {
            Layout::Full => obs_layout::OBS_DIM,
            Layout::Minimal => obs_layout_minimal::OBS_DIM_MINIMAL,
        }
    }
}

/// A packed batch, ready for its layout's row encoder. Built once per encode
/// call and consumed at once, so the size gap between the variants costs one
/// move, not a heap allocation.
#[allow(clippy::large_enum_variant)]
pub(super) enum PackedRows {
    Full(PackedFull),
    Minimal(PackedCore, Array2<bool>),
}

impl PackedRows {
    /// The parts the aux dict is built from.
    fn into_aux(self) -> (PackedCore, Array2<bool>) {
        match self {
            PackedRows::Full(f) => (f.obs.core, f.legal),
            PackedRows::Minimal(core, legal) => (core, legal),
        }
    }
}

/// Where an encode call puts the rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EncodeMode {
    /// A fresh `(k, dim)` "obs" array in the returned dict.
    Return,
    /// Into the caller's `out` rows and/or packed outputs (at least one).
    InPlace,
}

impl PyBatchedEngine {
    /// Both layouts' Rust encoders are PLO-only; NLH encodes through numpy.
    pub(super) fn require_plo(&self, what: &str) -> PyResult<()> {
        if !self.config.variant.is_plo() {
            return Err(PyRuntimeError::new_err(format!(
                "{what} is PLO-only; NLH uses the numpy batch encoder (observation_and_features_batch)"
            )));
        }
        Ok(())
    }

    pub(super) fn pack_for(&self, layout: Layout, idx: &[usize]) -> PackedRows {
        match layout {
            Layout::Full => PackedRows::Full(self.pack_full(idx)),
            Layout::Minimal => {
                let (core, legal) = self.pack_minimal_with_legal(idx);
                PackedRows::Minimal(core, legal)
            }
        }
    }

    /// Zero, then encode, row j of `packed` for every j with `keep(j)` (the
    /// other rows are only zeroed — exactly what encoding into a fresh
    /// `vec![0f32; ..]` did, so the bits match whichever buffer the rows live
    /// in), into `rows[j]` and/or packed into the compact sinks while still in
    /// cache. Without `rows` (packed-only) each row is encoded into a per-task
    /// scratch row, never stored dense. Err((j, col, value)) when a flag
    /// column is not exactly 0/1.
    #[allow(clippy::type_complexity)]
    pub(super) fn encode_rows(
        &self,
        packed: &PackedRows,
        rows: Option<Vec<&mut [f32]>>,
        sinks: Option<(&PackPlan, Vec<&mut [u8]>, Vec<&mut [f32]>)>,
        keep: impl Fn(usize) -> bool + Sync,
    ) -> Result<(), (usize, usize, f32)> {
        let s = self.config.num_seats;
        let bb = self.config.bb;
        let ante = self.config.ante;
        let obs_rev = self.obs_rev;
        let starting = &self.config.starting_stacks;
        let inv_bb = 1.0f64 / (bb as f64);
        let dim = match packed {
            PackedRows::Full(_) => obs_layout::OBS_DIM,
            PackedRows::Minimal(..) => obs_layout_minimal::OBS_DIM_MINIMAL,
        };
        let enc = |j: usize, row: &mut [f32]| {
            row.fill(0.0);
            if keep(j) {
                match packed {
                    PackedRows::Full(f) => encode_obs_row(
                        &f.obs, j, s, f.cat_a[j], f.cat_b[j], inv_bb, bb, ante, starting, obs_rev,
                        row,
                    ),
                    PackedRows::Minimal(core, _) => {
                        encode_obs_row_minimal(core, j, s, inv_bb, bb, starting, obs_rev, row)
                    }
                }
            }
        };
        let pack = |plan: &PackPlan, j: usize, row: &[f32], b: &mut [u8], r: &mut [f32]| {
            pack_obs_row_runs(
                row,
                &plan.flag_runs,
                &plan.real_runs,
                &plan.flags,
                &plan.reals,
                b,
                r,
            )
            .map_err(|(c, v)| (j, c, v))
        };
        match (rows, sinks) {
            (None, None) => Ok(()),
            (Some(rows), None) => {
                rows.into_par_iter()
                    .enumerate()
                    .for_each(|(j, row)| enc(j, row));
                Ok(())
            }
            (None, Some((plan, bits, reals))) => bits
                .into_par_iter()
                .zip(reals.into_par_iter())
                .enumerate()
                .try_for_each_init(
                    || vec![0f32; dim],
                    |row, (j, (b, r))| {
                        enc(j, row);
                        pack(plan, j, row, b, r)
                    },
                ),
            (Some(rows), Some((plan, bits, reals))) => rows
                .into_par_iter()
                .zip(bits.into_par_iter())
                .zip(reals.into_par_iter())
                .enumerate()
                .try_for_each(|(j, ((row, b), r))| {
                    enc(j, row);
                    pack(plan, j, row, b, r)
                }),
        }
    }

    /// The one implementation behind `BatchedEngine.encode` and every legacy
    /// `observation_encoded_*` name (ENG-012). `what` names the Python method
    /// in error messages.
    ///
    /// - `indices`: the envs to encode (row j = env `indices[j]`); None = all.
    /// - [`EncodeMode::Return`]: a fresh `(k, dim)` "obs" in the dict.
    /// - [`EncodeMode::InPlace`]: row j goes to `out[indices[j]]` (a
    ///   C-contiguous `(num_envs, dim)` array) and/or the packed outputs'
    ///   rows; `indices` must then be strictly increasing, and other rows are
    ///   untouched.
    /// - `encode_mask` (length num_envs): envs whose entry is False get an
    ///   all-zero row (the rollout's skipped-row convention).
    /// - `flag_cols` / `real_cols` / `out_bits` / `out_real` (all four; the
    ///   compact layout of python/plo5bp/compact_obs.py): each row is also
    ///   packed right after it is encoded — the exact bytes `pack_obs_rows`
    ///   would write. With `out=None` that is PACKED-ONLY.
    ///
    /// The dict always carries the aux fields (actor, legal_mask, min/max
    /// raise, commits, street, pot) for the k encoded rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_impl<'py>(
        &self,
        py: Python<'py>,
        what: &str,
        layout: Layout,
        mode: EncodeMode,
        indices: Option<PyReadonlyArray1<'_, i64>>,
        mut out: Option<PyReadwriteArray2<'_, f32>>,
        encode_mask: Option<PyReadonlyArray1<'_, bool>>,
        flag_cols: Option<PyReadonlyArray1<'_, i64>>,
        real_cols: Option<PyReadonlyArray1<'_, i64>>,
        mut out_bits: Option<PyReadwriteArray2<'_, u8>>,
        mut out_real: Option<PyReadwriteArray2<'_, f32>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.require_plo(what)?;
        let n = self.states.len();
        let d = layout.dim();
        if mode == EncodeMode::Return && (out.is_some() || out_bits.is_some() || out_real.is_some())
        {
            return Err(PyValueError::new_err(format!(
                "{what} returns fresh rows; it takes no out / out_bits / out_real"
            )));
        }
        if let Some(o) = out.as_ref() {
            if !o.is_c_contiguous() || o.shape() != [n, d] {
                return Err(PyValueError::new_err(format!(
                    "{what}: out must be a C-contiguous ({n}, {d}) float32 array; got shape {:?}",
                    o.shape()
                )));
            }
        }
        let plan = pack_plan_for_width(what, n, d, flag_cols, real_cols, &out_bits, &out_real)?;
        if mode == EncodeMode::InPlace {
            require_some_output(what, out.is_some(), plan.is_some())?;
        }
        let idx: Vec<usize> = match indices.as_ref() {
            None => (0..n).collect(),
            Some(ix) => {
                let idx = checked_env_indices(ix.as_slice()?, n, what)?;
                if mode == EncodeMode::InPlace && idx.windows(2).any(|w| w[0] >= w[1]) {
                    return Err(PyValueError::new_err(format!(
                        "{what}: indices must be strictly increasing (unique, sorted)"
                    )));
                }
                idx
            }
        };
        let mask: Option<Vec<bool>> = match encode_mask {
            None => None,
            Some(m) => {
                let m = m.as_slice()?;
                if m.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "{what}: encode_mask has {} entries, expected {n}",
                        m.len()
                    )));
                }
                Some(m.to_vec())
            }
        };
        let keep = |j: usize| mask.as_ref().is_none_or(|m| m[idx[j]]);
        if mode == EncodeMode::Return {
            let k = idx.len();
            let (obs, packed) = py.detach(|| {
                let packed = self.pack_for(layout, &idx);
                let mut obs = vec![0f32; k * d];
                let rows: Vec<&mut [f32]> = obs.chunks_exact_mut(d).collect();
                self.encode_rows(&packed, Some(rows), None, keep)
                    .expect("no packing requested");
                (obs, packed)
            });
            let obs = Array2::from_shape_vec((k, d), obs)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let (core, legal) = packed.into_aux();
            let dict = aux_dict(py, core, legal)?;
            dict.set_item("obs", obs.into_pyarray(py))?;
            return Ok(dict);
        }
        let out_s = match out.as_mut() {
            Some(o) => Some(o.as_slice_mut()?),
            None => None,
        };
        let sinks = packed_slices(&mut out_bits, &mut out_real)?;
        let (packed, res) = py.detach(|| {
            let packed = self.pack_for(layout, &idx);
            let rows = out_s.map(|o| pick_rows(o, d, &idx));
            let sinks = sinks
                .zip(plan.as_ref())
                .map(|((b, r), pl)| (pl, pick_rows(b, pl.nb, &idx), pick_rows(r, pl.nr, &idx)));
            let res = self.encode_rows(&packed, rows, sinks, keep);
            (packed, res)
        });
        res.map_err(|(j, c, v)| pack_encoder_error(what, idx[j], c, v))?;
        let (core, legal) = packed.into_aux();
        aux_dict(py, core, legal)
    }
}

/// The blocks BOTH observation layouts encode identically (ENG-011): the
/// dims that sit at the same place in both, and the history slot format.
/// Twin of the shared `_*_OFF` / `_M_*_OFF` constants in
/// python/plo5bp/encoding.py.
pub(super) mod obs_core {
    pub const HOLE_OFF: usize = 0;
    pub const BOARD_A_OFF: usize = 52;
    pub const BOARD_B_OFF: usize = 104;
    pub const STREET_OFF: usize = 156;
    pub const NUM_STREET_ONEHOT: usize = 4;
    pub const ACTIVE_OFF: usize = 160;
    pub const ALLIN_OFF: usize = 168;
    pub const STACKS_OFF: usize = 176;
    pub const SCALARS_OFF: usize = 184;
    /// History slots the PLO encoders read — and therefore the PLO packer's
    /// width (`history_cap`): the packer keeps the NEWEST records from slot 0,
    /// so a mismatch would shift every history feature (ENG-016).
    pub const HISTORY_DEPTH: usize = 32;
    pub const HISTORY_SLOT_DIM: usize = 18;
    pub const HISTORY_SEAT_OFF_REL: usize = 0;
    pub const HISTORY_GATE_OFF_REL: usize = 8;
    pub const HISTORY_STREET_OFF_REL: usize = 12;
    pub const HISTORY_CHIPS_OFF_REL: usize = 16;
    pub const HISTORY_FRAC_OFF_REL: usize = 17;
}

/// Where a layout puts the core blocks that move between layouts (the full
/// layout has REL_POS at 188 and its feature blocks before the seat block).
pub(super) struct CoreOffsets {
    pub(super) history: usize,
    pub(super) seat_exists: usize,
    pub(super) total_commit: usize,
    pub(super) street_commit: usize,
    pub(super) hero_btn: usize,
}

/// What the core pass hands the full encoder's own blocks.
pub(super) struct CoreRow<'a> {
    pub(super) hero: usize,
    pub(super) hole: &'a [u8],
    pub(super) board_a: &'a [u8],
    pub(super) board_b: &'a [u8],
    pub(super) street: usize,
    /// Effective stack per ABSOLUTE seat (the dead-chips chain).
    pub(super) eff_per_seat: [f64; 8],
    pub(super) pot: f64,
    pub(super) btc: f64,
    pub(super) legal_window: RaiseWindow,
}

/// Encode row `j`'s core blocks into `out` (pre-zeroed; `at` says where the
/// moving blocks go) — the ONE implementation both layouts share, so a fix to
/// the stack, scalar or history math lands in both. Returns what the full
/// layout's extra blocks read; None for a terminal row (actor < 0), which
/// stays all-zero (the scalar encoder's early return).
///
/// Bit-exactness discipline: scalar arithmetic in f64, cast to f32 only at the
/// store (`(x as f64 * inv_bb) as f32`), as the numpy path does; hero rotation
/// is a non-negative modulo, like Python's `%`.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_core<'a>(
    core: &'a PackedCore,
    j: usize,
    at: &CoreOffsets,
    num_seats: usize,
    inv_bb: f64,
    bb: u64,
    starting: &[u64],
    obs_rev: u8,
    out: &mut [f32],
) -> Option<CoreRow<'a>> {
    use obs_core::*;

    let hero_i = core.actor[j];
    if hero_i < 0 {
        return None;
    }
    let hero = hero_i as usize;
    let ns_i = num_seats as i64;
    let rel = |x: i64| -> usize { (x - hero as i64).rem_euclid(ns_i) as usize };

    // --- Card multi-hots (hole / board A / board B). ---
    // Hole width is variant-dependent (PLO4=4, PLO5=5, PLO6=6), so the hole
    // loop is driven by the row's length; boards are always 5 wide. Do NOT
    // merge these loops — a `0..5` hole loop reads out of bounds on PLO4 and
    // silently drops the 6th card on PLO6 (CLAUDE.md variant-encoding gotcha).
    let hole = core.hero_hole.row(j).to_slice().unwrap();
    let board_a = core.board_a.row(j).to_slice().unwrap();
    let board_b = core.board_b.row(j).to_slice().unwrap();
    for &c in hole {
        if c < 52 {
            out[HOLE_OFF + c as usize] = 1.0;
        }
    }
    for slot in 0..5 {
        let ca = board_a[slot];
        if ca < 52 {
            out[BOARD_A_OFF + ca as usize] = 1.0;
        }
        let cb = board_b[slot];
        if cb < 52 {
            out[BOARD_B_OFF + cb as usize] = 1.0;
        }
    }

    // --- Street one-hot. ---
    let street = core.street[j] as usize;
    if street < NUM_STREET_ONEHOT {
        out[STREET_OFF + street] = 1.0;
    }

    // --- Effective stack per seat (dead-chips chain), absolute seat index. ---
    // dead = max(0, starting - eff_cap); eff = max(0, stacks - dead). All f64;
    // chip magnitudes are exact integers in f64 range.
    let mut eff_per_seat = [0f64; 8];
    for seat in 0..num_seats {
        let st = starting[seat] as f64;
        let ec = core.eff_stack_cap[[j, seat]] as f64;
        let dead = (st - ec).max(0.0);
        let stk = core.stacks[[j, seat]] as f64;
        eff_per_seat[seat] = (stk - dead).max(0.0);
    }

    // --- Active / all-in / stacks (hero-rotated). ---
    for k in 0..num_seats {
        let seat = (hero + k) % num_seats;
        if !core.folded[[j, seat]] {
            out[ACTIVE_OFF + k] = 1.0;
        }
        if core.all_in[[j, seat]] {
            out[ALLIN_OFF + k] = 1.0;
        }
        out[STACKS_OFF + k] = (eff_per_seat[seat] * inv_bb) as f32;
    }

    // --- Scalars: pot, bet_to_call, raise-window totals (all / bb). ---
    let pot = core.pot[j] as f64;
    let btc = core.bet_to_call[j] as f64;
    out[SCALARS_OFF] = (pot * inv_bb) as f32;
    out[SCALARS_OFF + 1] = (btc * inv_bb) as f32;
    let legal_window = legal_raise_window(
        core.min_raise[j],
        core.max_raise[j],
        core.stacks[[j, hero]],
        bb,
    );
    if obs_rev == OBS_REV_LEGACY {
        // Rev 1: min_bet_total()/max_bet_total() verbatim.
        out[SCALARS_OFF + 2] = (core.min_bet[j] as f64 * inv_bb) as f32;
        out[SCALARS_OFF + 3] = (core.max_bet[j] as f64 * inv_bb) as f32;
    } else if legal_window.legal {
        // PRODUCTION BEHAVIOR CHANGE (review 2026-09-20 B3, full dims 186/187,
        // minimal slots 2/3): the LEGAL raise window as street totals —
        // street_commit[hero] + the engine's min/max raise delta, 0/0 when
        // Raise is illegal. Was min_bet_total()/max_bet_total() (the deepest
        // opponent's raw reach leaked in).
        let hero_sc = core.street_commit[[j, hero]] as f64;
        out[SCALARS_OFF + 2] = ((hero_sc + legal_window.min_d) * inv_bb) as f32;
        out[SCALARS_OFF + 3] = ((hero_sc + legal_window.max_d) * inv_bb) as f32;
    }

    // --- History (oldest-first, already truncated to the newest HISTORY_DEPTH). ---
    // Pot before each visible action: history chips are per-action DELTAS
    // (antes never recorded), so pot_before(slot) = current_pot - sum of the
    // chips of visible slots >= slot. Valid under truncation — dropped actions
    // all precede the window. Integer math mirrors numpy / the scalar encoder.
    let hlen = (core.history_len[j] as usize).min(HISTORY_DEPTH);
    let pot_now_chips = core.pot[j] as i64;
    let mut pot_before = [0i64; HISTORY_DEPTH];
    let mut chips_suffix: i64 = 0;
    for slot in (0..hlen).rev() {
        chips_suffix += core.history_chips[[j, slot]] as i64;
        pot_before[slot] = pot_now_chips - chips_suffix;
    }
    for slot in 0..hlen {
        let base = at.history + slot * HISTORY_SLOT_DIM;
        let hseat = core.history_seat[[j, slot]] as i64;
        out[base + HISTORY_SEAT_OFF_REL + rel(hseat)] = 1.0;
        let action = core.history_action[[j, slot]];
        let chips = core.history_chips[[j, slot]];
        // Gate (matches _gate_from_action): FOLD(0)->0; CHECK_CALL(1) with
        // chips==0 -> Check(1), with chips>0 -> Call(2); anything else ->Raise(3).
        let gate = if action == 0 {
            0
        } else if action == 1 {
            if chips == 0 {
                1
            } else {
                2
            }
        } else {
            3
        };
        out[base + HISTORY_GATE_OFF_REL + gate] = 1.0;
        let s_idx = core.history_street[[j, slot]];
        if s_idx >= 0 && (s_idx as usize) < NUM_STREET_ONEHOT {
            out[base + HISTORY_STREET_OFF_REL + s_idx as usize] = 1.0;
        }
        out[base + HISTORY_CHIPS_OFF_REL] = (chips as f64 * inv_bb) as f32;
        let frac = chips as f64 / pot_before[slot].max(1) as f64;
        out[base + HISTORY_FRAC_OFF_REL] = frac.clamp(0.0, 2.0) as f32;
    }

    // --- Structural seat-exists mask. ---
    for k in 0..num_seats {
        out[at.seat_exists + k] = 1.0;
    }

    // --- Per-seat commits (hero-rotated). ---
    for k in 0..num_seats {
        let seat = (hero + k) % num_seats;
        out[at.total_commit + k] = (core.total_commit[[j, seat]] as f64 * inv_bb) as f32;
        out[at.street_commit + k] = (core.street_commit[[j, seat]] as f64 * inv_bb) as f32;
    }

    // --- Hero distance to button (hero-relative one-hot). ---
    out[at.hero_btn + rel(core.button[j] as i64)] = 1.0;

    Some(CoreRow {
        hero,
        hole,
        board_a,
        board_b,
        street,
        eff_per_seat,
        pot,
        btc,
        legal_window,
    })
}

/// The raise window an observation describes, as chip DELTAS the actor adds.
/// Twin of `_RaiseWindow` (python/plo5bp/encoding.py) — keep in lockstep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct RaiseWindow {
    /// Raise available: the STK-2 / STK-5[2:4] gate.
    pub(super) legal: bool,
    pub(super) min_d: f64,
    pub(super) max_d: f64,
    /// Min fed to the legal-anchor count (the max is `max_d`).
    pub(super) anchor_min: f64,
}

/// Rev 2: the LEGAL raise window. Twin of `_legal_raise_window`.
///
/// PRODUCTION BEHAVIOR CHANGE (review 2026-09-20 B1/B3; obs_rev 2): the
/// raise-window dims (scalars min/max, v7 STK-2, STK-5[2:4]) come from the
/// engine's legal `min_raise_chips()`/`max_raise_chips()`, not from
/// `min_bet_total()`/`max_bet_total()`: the totals are not capped by the
/// actor's own stack, ignore the short-shove lockout, and `max_bet_total()` is
/// the deepest opponent's RAW reach (breaks dead-chip invariance).
///
/// `legal` mirrors the env's Raise gate (`actions.gate_mask_from_bounds`):
/// `max_raise > 0`, and a sub-1bb raise only counts as hero's own all-in — with
/// `max_raise > 0`, `legal[AllIn]` holds exactly when hero's stack is the
/// binding cap (`max_raise == stack`); the other case is the cover-short DUST
/// the gate screens off. Short-shove regime (`min_raise == 0 < max_raise`): the
/// only legal size is `max_raise`, so `min_d == max_d`, while `anchor_min`
/// stays the RAW `min_raise` (0) — how the anchor count detects the regime.
#[inline]
pub(super) fn legal_raise_window(
    min_raise: u64,
    max_raise: u64,
    hero_stack_raw: u64,
    bb: u64,
) -> RaiseWindow {
    let anchor_min = min_raise as f64;
    if max_raise == 0 || (max_raise < bb && max_raise != hero_stack_raw) {
        return RaiseWindow {
            legal: false,
            min_d: 0.0,
            max_d: 0.0,
            anchor_min,
        };
    }
    let min_d = if min_raise > 0 { min_raise } else { max_raise };
    RaiseWindow {
        legal: true,
        min_d: min_d as f64,
        max_d: max_raise as f64,
        anchor_min,
    }
}

/// Rev 1 (pre-2026-09-20, kept bit-exact for old checkpoints): the window
/// recovered from the `min_bet_total()`/`max_bet_total()` TOTALS. Known-wrong —
/// see `legal_raise_window`. Twin of `_legacy_raise_window`.
#[inline]
pub(super) fn legacy_raise_window(
    min_bet: u64,
    max_bet: u64,
    hero_sc: f64,
    to_call: f64,
) -> RaiseWindow {
    let min_d = min_bet as f64 - hero_sc;
    let max_d = max_bet as f64 - hero_sc;
    RaiseWindow {
        legal: max_d > to_call,
        min_d,
        max_d,
        anchor_min: min_d,
    }
}
