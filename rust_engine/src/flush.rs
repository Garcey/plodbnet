//! Rollout trajectory flush (2026-09-23): the per-finished-hand block of
//! `python/plo5bp/rollout.py` (step9b retroactive bonus, step9c GAE / VRPO
//! backward scans, step9d gathers into the output slabs) as ONE parallel pass,
//! plus the per-step trajectory record and the byte-row gathers.
//!
//! Exactness contract: every float32 value is produced by the same IEEE
//! operations in the same order as the numpy code it replaces -- numpy's
//! elementwise ufuncs round each operation to f32 and never fuse, and Rust
//! never contracts `a * b + c` into an FMA, so the bits match. Rows are written
//! in numpy's order (finished hand t, seat s, trajectory slot l -- the C order
//! of the (T, S, L) flush window). Integer counters are exact. The bonus total
//! (a log-line diagnostic, exactly 0 with the bonus off) is summed in f64 in a
//! FIXED order -- per block of (hand, seat) pairs, then the blocks in order --
//! so it is identical from run to run, though not numpy's pairwise f32 order
//! (ENG-031: a rayon `reduce` used to make it depend on thread scheduling).
//!
//! Every kernel here is a plain function over slices (`flush_inner`,
//! `record_inner`, `gather_inner`) behind a thin pyfunction, with no `unsafe`:
//! the parallel passes hand each task its own sub-slices (ENG-013), and every
//! check runs before the first write, so an error leaves the outputs as they
//! were (ENG-030).

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

const GATE_CHECK_CALL: i8 = 1;
const GATE_RAISE: i8 = 2;

/// Read-only C-contiguous slice of `dict[key]` as `T`.
fn get_ro<'py, T: numpy::Element>(
    d: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<numpy::PyReadonlyArrayDyn<'py, T>> {
    let obj = d
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("flush_trajectories: missing '{key}'")))?;
    let arr: numpy::PyReadonlyArrayDyn<'py, T> = obj.extract().map_err(|_| {
        PyValueError::new_err(format!("flush_trajectories: '{key}' has the wrong dtype"))
    })?;
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "flush_trajectories: '{key}' must be C-contiguous"
        )));
    }
    Ok(arr)
}

fn get_rw<'py, T: numpy::Element>(
    d: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<numpy::PyReadwriteArrayDyn<'py, T>> {
    let obj = d
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("flush_trajectories: missing '{key}'")))?;
    let arr: numpy::PyReadwriteArrayDyn<'py, T> = obj.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "flush_trajectories: '{key}' has the wrong dtype or is not writeable"
        ))
    })?;
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "flush_trajectories: '{key}' must be C-contiguous"
        )));
    }
    Ok(arr)
}

/// IEEE-754 binary16 bits of `x`, rounded to nearest even: numpy's
/// `astype(np.float16)` bit for bit on every finite and infinite input
/// (subnormals, ties and overflow to inf included; pinned over all 2^32 inputs
/// against the x86 F16C instruction by `f16_conversion_is_exact`). A NaN stays
/// a NaN of the same sign with its payload's top 10 bits and the quiet bit set
/// -- what F16C (and numpy's SIMD casts) produce; numpy's portable C path keeps
/// a signaling NaN signaling instead. Rollout rows never hold a NaN (the
/// packed-only tripwire NaNs are never stored). Half-precision rollout storage
/// (2026-09-26, TrainingConfig.obs_real_f16).
#[inline]
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign
            | 0x7c00
            | if man != 0 {
                0x0200 | (man >> 13) as u16
            } else {
                0
            };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1;
    }
    sign | r as u16
}

// ---------------------------------------------------------------------------
// flush_trajectories
// ---------------------------------------------------------------------------

/// The flat trajectory arrays a flush reads (`rollout._flat_traj_view`): one
/// slot per (env, seat, step), `sizing` 4 wide.
pub(crate) struct TrajIn<'a> {
    pub(crate) obs_idx: &'a [i64],
    pub(crate) gate: &'a [i8],
    pub(crate) chips: &'a [i64],
    pub(crate) sizing: &'a [i64],
    pub(crate) anchor: &'a [i8],
    pub(crate) u: &'a [f32],
    pub(crate) log_p: &'a [f32],
    pub(crate) gate_lp: &'a [f32],
    pub(crate) anchor_lp: &'a [f32],
    pub(crate) value: &'a [f32],
    pub(crate) costs: &'a [f32],
    pub(crate) pots: &'a [f32],
    pub(crate) streets: &'a [i8],
    /// `(q_taken, vpi)` for the VRPO advantage; None = GAE.
    pub(crate) q_vpi: Option<(&'a [f32], &'a [f32])>,
}

/// The per-step observation pool rows the trajectory slots point at.
pub(crate) struct PoolIn<'a> {
    pub(crate) bits: &'a [u8],
    pub(crate) nb: usize,
    pub(crate) real: &'a [f32],
    pub(crate) nr: usize,
    pub(crate) gm: &'a [bool],
    pub(crate) gm_w: usize,
}

/// Everything a flush reads. The (T, S) arrays are row-major over (finished
/// hand t, seat s); `holes_rot` has `hole_rows` rows (num_envs * S) of `oh_w`.
pub(crate) struct FlushIn<'a> {
    pub(crate) term: &'a [i64],
    pub(crate) s_n: usize,
    pub(crate) lengths: &'a [i32],
    pub(crate) flush_mask: &'a [bool],
    pub(crate) won_bb: &'a [f32],
    pub(crate) share_gt: &'a [bool],
    pub(crate) share_eq: &'a [bool],
    pub(crate) traj: TrajIn<'a>,
    pub(crate) traj_cap: usize,
    pub(crate) pool: PoolIn<'a>,
    pub(crate) holes_rot: &'a [u8],
    pub(crate) hole_rows: usize,
    pub(crate) oh_w: usize,
    pub(crate) gamma: f32,
    pub(crate) lam: f32,
    pub(crate) retro_c: f32,
}

/// The float observation columns of the output slab: float32, or float16 bits.
pub(crate) enum RealOut<'a> {
    F32(&'a mut [f32]),
    F16(&'a mut [u16]),
}

/// The output slabs, exactly the rows being written (row widths: bits `nb`,
/// real `nr`, gm `gm_w`, sz 4, oh `oh_w`, the rest 1).
pub(crate) struct FlushOut<'a> {
    pub(crate) bits: &'a mut [u8],
    pub(crate) real: RealOut<'a>,
    pub(crate) gm: &'a mut [bool],
    pub(crate) ga: &'a mut [i64],
    pub(crate) rc: &'a mut [i64],
    pub(crate) sz: &'a mut [i64],
    pub(crate) an: &'a mut [i64],
    pub(crate) ru: &'a mut [f32],
    pub(crate) oh: &'a mut [u8],
    pub(crate) lp: &'a mut [f32],
    pub(crate) glp: &'a mut [f32],
    pub(crate) alp: &'a mut [f32],
    pub(crate) v: &'a mut [f32],
    pub(crate) ret: &'a mut [f32],
    pub(crate) adv: &'a mut [f32],
    pub(crate) last: &'a mut [bool],
}

/// Row widths of the output slabs.
#[derive(Clone, Copy)]
struct Widths {
    nb: usize,
    nr: usize,
    gm: usize,
    oh: usize,
}

/// `(head, tail)` of `s` at `n` elements, leaving `tail` in place.
fn take_front<'a, T>(s: &mut &'a mut [T], n: usize) -> &'a mut [T] {
    let (head, tail) = std::mem::take(s).split_at_mut(n);
    *s = tail;
    head
}

impl<'a> FlushOut<'a> {
    /// Every slab's (name, values, width) — for the size check.
    fn sizes(&self, w: Widths) -> [(&'static str, usize, usize); 16] {
        let real = match &self.real {
            RealOut::F32(a) => a.len(),
            RealOut::F16(a) => a.len(),
        };
        [
            ("obs_bits", self.bits.len(), w.nb),
            ("obs_real", real, w.nr),
            ("gm", self.gm.len(), w.gm),
            ("ga", self.ga.len(), 1),
            ("rc", self.rc.len(), 1),
            ("sz", self.sz.len(), 4),
            ("an", self.an.len(), 1),
            ("ru", self.ru.len(), 1),
            ("oh", self.oh.len(), w.oh),
            ("lp", self.lp.len(), 1),
            ("glp", self.glp.len(), 1),
            ("alp", self.alp.len(), 1),
            ("v", self.v.len(), 1),
            ("ret", self.ret.len(), 1),
            ("adv", self.adv.len(), 1),
            ("last", self.last.len(), 1),
        ]
    }

    /// The next `rows` rows of every slab, split off the front (ENG-013: the
    /// parallel pass hands each task its own rows this way — no raw pointers).
    fn split_off(&mut self, rows: usize, w: Widths) -> FlushOut<'a> {
        FlushOut {
            bits: take_front(&mut self.bits, rows * w.nb),
            real: match &mut self.real {
                RealOut::F32(a) => RealOut::F32(take_front(a, rows * w.nr)),
                RealOut::F16(a) => RealOut::F16(take_front(a, rows * w.nr)),
            },
            gm: take_front(&mut self.gm, rows * w.gm),
            ga: take_front(&mut self.ga, rows),
            rc: take_front(&mut self.rc, rows),
            sz: take_front(&mut self.sz, rows * 4),
            an: take_front(&mut self.an, rows),
            ru: take_front(&mut self.ru, rows),
            oh: take_front(&mut self.oh, rows * w.oh),
            lp: take_front(&mut self.lp, rows),
            glp: take_front(&mut self.glp, rows),
            alp: take_front(&mut self.alp, rows),
            v: take_front(&mut self.v, rows),
            ret: take_front(&mut self.ret, rows),
            adv: take_front(&mut self.adv, rows),
            last: take_front(&mut self.last, rows),
        }
    }
}

/// What a flush returns: rows written, qualifying bonus steps (total and per
/// street: flop, turn, river) and the retroactive-bonus total (bb).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FlushStats {
    pub(crate) rows: usize,
    pub(crate) bonus_steps: u64,
    pub(crate) by_street: [u64; 3],
    pub(crate) bonus_total: f64,
}

/// (hand, seat) pairs per parallel task: each task splits its rows off the
/// slabs once and walks its pairs in order.
const FLUSH_BLOCK: usize = 64;

/// Validate every input against the others, then flush (see the module docs
/// and the numpy path in rollout.py, which this reproduces bit for bit). Every
/// check runs before the first write: an `Err` leaves `out` untouched.
pub(crate) fn flush_inner(inp: &FlushIn<'_>, mut out: FlushOut<'_>) -> Result<FlushStats, String> {
    let (s_n, cap) = (inp.s_n, inp.traj_cap);
    let t_n = inp.term.len();
    let n_k = t_n * s_n;
    if s_n == 0 || cap == 0 {
        return Err("term_envs / lengths / traj_cap disagree".into());
    }
    for (name, len) in [
        ("lengths", inp.lengths.len()),
        ("flush_mask", inp.flush_mask.len()),
        ("won_bb", inp.won_bb.len()),
        ("share_gt", inp.share_gt.len()),
        ("share_eq", inp.share_eq.len()),
    ] {
        if len != n_k {
            return Err(format!(
                "{name} has {len} entries, expected {t_n} hands x {s_n} seats"
            ));
        }
    }
    let tr = &inp.traj;
    let m = tr.gate.len();
    for (name, len) in [
        ("obs_idx", tr.obs_idx.len()),
        ("chips", tr.chips.len()),
        ("anchor", tr.anchor.len()),
        ("u", tr.u.len()),
        ("log_p", tr.log_p.len()),
        ("gate_lp", tr.gate_lp.len()),
        ("anchor_lp", tr.anchor_lp.len()),
        ("value", tr.value.len()),
        ("costs", tr.costs.len()),
        ("pots", tr.pots.len()),
        ("streets", tr.streets.len()),
        ("sizing/4", tr.sizing.len() / 4),
    ] {
        if len != m {
            return Err(format!(
                "trajectory array '{name}' has {len} slots, gate has {m}"
            ));
        }
    }
    if tr.sizing.len() != 4 * m {
        return Err(format!(
            "trajectory 'sizing' has {} values, expected {m} x 4",
            tr.sizing.len()
        ));
    }
    if let Some((q, v)) = tr.q_vpi {
        if q.len() != m || v.len() != m {
            return Err("q_taken / vpi slot counts differ from gate".into());
        }
    }
    if inp.holes_rot.len() != inp.hole_rows * inp.oh_w {
        return Err("holes_rot is not hole_rows x oh_w".into());
    }
    if m != inp.hole_rows * cap {
        return Err(format!(
            "{m} trajectory slots != holes_rot rows {} x traj_cap {cap}",
            inp.hole_rows
        ));
    }
    let pool = &inp.pool;
    let p_rows = if pool.nb == 0 {
        0
    } else {
        pool.bits.len() / pool.nb
    };
    if pool.bits.len() != p_rows * pool.nb
        || pool.real.len() != p_rows * pool.nr
        || pool.gm.len() != p_rows * pool.gm_w
    {
        return Err("pool arrays disagree on rows".into());
    }
    for (k, &e) in inp.term.iter().enumerate() {
        if e < 0 || (e as usize + 1) * s_n > inp.hole_rows {
            return Err(format!("term_envs[{k}] = {e} out of range"));
        }
    }
    // Row offsets per (hand, seat): C order of the (T, S, L) window.
    let mut offsets = vec![0usize; n_k + 1];
    for k in 0..n_k {
        let len = if inp.flush_mask[k] {
            inp.lengths[k].max(0) as usize
        } else {
            0
        };
        if len > cap {
            return Err(format!(
                "trajectory length {} > traj_cap {cap}",
                inp.lengths[k]
            ));
        }
        offsets[k + 1] = offsets[k] + len;
        // Pool rows referenced by the slots being flushed must exist.
        let base = (inp.term[k / s_n] as usize * s_n + k % s_n) * cap;
        for &p in &tr.obs_idx[base..base + len] {
            if p < 0 || p as usize >= p_rows {
                return Err(format!("pool index {p} out of range ({p_rows} rows)"));
            }
        }
    }
    let n_new = offsets[n_k];
    let w = Widths {
        nb: pool.nb,
        nr: pool.nr,
        gm: pool.gm_w,
        oh: inp.oh_w,
    };
    for (name, len, width) in out.sizes(w) {
        if len != n_new * width {
            return Err(format!(
                "output '{name}' holds {len} values, expected {n_new} rows x {width}"
            ));
        }
    }

    // Hand every task the rows of its block of (hand, seat) pairs.
    let mut blocks = Vec::with_capacity(n_k.div_ceil(FLUSH_BLOCK));
    let mut lo = 0;
    while lo < n_k {
        let hi = (lo + FLUSH_BLOCK).min(n_k);
        blocks.push((lo, hi, out.split_off(offsets[hi] - offsets[lo], w)));
        lo = hi;
    }
    let per_block: Vec<(u64, [u64; 3], f64)> = blocks
        .into_par_iter()
        .map(|(lo, hi, mut rows)| {
            let mut acc = (0u64, [0u64; 3], 0f64);
            for k in lo..hi {
                let len = offsets[k + 1] - offsets[k];
                if len > 0 {
                    flush_one(inp, k, len, &mut rows.split_off(len, w), w, &mut acc);
                }
            }
            acc
        })
        .collect();
    let mut stats = FlushStats {
        rows: n_new,
        bonus_steps: 0,
        by_street: [0; 3],
        bonus_total: 0.0,
    };
    for (steps, streets, total) in per_block {
        stats.bonus_steps += steps;
        for (a, b) in stats.by_street.iter_mut().zip(streets) {
            *a += b;
        }
        stats.bonus_total += total;
    }
    Ok(stats)
}

/// Flush one (hand, seat) pair `k` (`len` > 0 slots) into its own `rows`.
fn flush_one(
    inp: &FlushIn<'_>,
    k: usize,
    len: usize,
    rows: &mut FlushOut<'_>,
    w: Widths,
    acc: &mut (u64, [u64; 3], f64),
) {
    let tr = &inp.traj;
    let pool = &inp.pool;
    let (t, s) = (k / inp.s_n, k % inp.s_n);
    let es = inp.term[t] as usize * inp.s_n + s;
    let base = es * inp.traj_cap;
    let (gt, eq, won_k) = (inp.share_gt[k], inp.share_eq[k], inp.won_bb[k]);
    let (gamma, retro_c) = (inp.gamma, inp.retro_c);
    // numpy: `gamma_f * lam_f * last_gae` evaluates (gamma_f * lam_f) first.
    let gl = gamma * inp.lam;
    // step9b: qualification + the (optional) retroactive bonus applied to a
    // COPY of the costs.
    let mut c_eff = [0f32; 256];
    let mut c_heap: Vec<f32>;
    let costs: &mut [f32] = if len <= c_eff.len() {
        &mut c_eff[..len]
    } else {
        c_heap = vec![0f32; len];
        &mut c_heap[..]
    };
    for l in 0..len {
        let g = tr.gate[base + l];
        let c = tr.costs[base + l];
        let is_raise = g == GATE_RAISE;
        let call_chips = g == GATE_CHECK_CALL && c < 0.0;
        let q = (gt && is_raise) || (eq && (is_raise || call_chips));
        if q {
            acc.0 += 1;
            let st = tr.streets[base + l];
            if (1..=3).contains(&st) {
                acc.1[(st - 1) as usize] += 1;
            }
        }
        costs[l] = if retro_c != 0.0 {
            let qf: f32 = if q { 1.0 } else { 0.0 };
            let b = qf * (retro_c * tr.pots[base + l]);
            acc.2 += b as f64;
            c + b
        } else {
            c
        };
    }
    // step9c: GAE and (VRPO) Expected-SARSA traces, backward; step9d: the
    // gathers into this pair's rows (row l = slot l).
    let mut last_gae = 0f32;
    let mut last_es = 0f32;
    let hole = &inp.holes_rot[es * w.oh..(es + 1) * w.oh];
    for l in (0..len).rev() {
        let is_last = l == len - 1;
        let reward = costs[l] + if is_last { won_k } else { 0.0 };
        let v_l = tr.value[base + l];
        let next_v = if is_last { 0.0 } else { tr.value[base + l + 1] };
        let delta = (reward + gamma * next_v) - v_l;
        last_gae = delta + gl * last_gae;
        let adv_l = match tr.q_vpi {
            Some((qs, vs)) => {
                let next_vpi = if is_last { 0.0 } else { vs[base + l + 1] };
                let q_l = qs[base + l];
                let d_es = (reward + gamma * next_vpi) - q_l;
                last_es = d_es + gl * last_es;
                (q_l - vs[base + l]) + last_es
            }
            None => last_gae,
        };
        let p = tr.obs_idx[base + l] as usize;
        rows.bits[l * w.nb..(l + 1) * w.nb].copy_from_slice(&pool.bits[p * w.nb..(p + 1) * w.nb]);
        let src = &pool.real[p * w.nr..(p + 1) * w.nr];
        match &mut rows.real {
            RealOut::F32(d) => d[l * w.nr..(l + 1) * w.nr].copy_from_slice(src),
            RealOut::F16(d) => {
                for (o, &x) in d[l * w.nr..(l + 1) * w.nr].iter_mut().zip(src) {
                    *o = f32_to_f16_bits(x);
                }
            }
        }
        rows.gm[l * w.gm..(l + 1) * w.gm].copy_from_slice(&pool.gm[p * w.gm..(p + 1) * w.gm]);
        rows.ga[l] = tr.gate[base + l] as i64;
        rows.rc[l] = tr.chips[base + l];
        rows.sz[l * 4..l * 4 + 4].copy_from_slice(&tr.sizing[(base + l) * 4..(base + l) * 4 + 4]);
        rows.an[l] = tr.anchor[base + l] as i64;
        rows.ru[l] = tr.u[base + l];
        rows.oh[l * w.oh..(l + 1) * w.oh].copy_from_slice(hole);
        rows.lp[l] = tr.log_p[base + l];
        rows.glp[l] = tr.gate_lp[base + l];
        rows.alp[l] = tr.anchor_lp[base + l];
        rows.v[l] = v_l;
        rows.ret[l] = last_gae + v_l;
        rows.adv[l] = adv_l;
        rows.last[l] = is_last;
    }
}

/// What `flush_trajectories` returns: (rows written, hands flushed, (hands
/// with a retroactive bonus, bonus rows, qualifying seats), the logged bonus
/// total).
type FlushSummary = (usize, u64, (u64, u64, u64), f64);

/// One row-gather job: (source rows, row width in bytes, destination rows).
type RowCopy<'a> = (&'a [u8], usize, &'a mut [u8]);

/// Flush the finished hands `term_envs` into the output slabs `out` (views of
/// exactly the n_new rows being written). See the module docs and the numpy
/// path in rollout.py, which this reproduces bit for bit.
///
/// Inputs: `lengths` / `flush_mask` / `won_bb` / `share_gt` / `share_eq` are
/// (T, S); `traj` holds the FLAT trajectory arrays (`rollout._flat_traj_view`:
/// obs_idx i64, gate i8, chips i64, sizing (M, 4) i64, anchor i8, u / log_p /
/// gate_lp / anchor_lp / value / costs / pots f32, streets i8, optionally
/// q_taken / vpi f32 for the VRPO advantage); `pools` holds the per-step obs
/// pool (obs_bits (P, nb) u8, obs_real (P, nr) f32, gm (P, 3) bool);
/// `holes_rot` is (n_envs * S, 5 * hole_w) u8; `out` carries "obs_real" (f32)
/// or "obs_real_f16" (the float16 slab as its u16 view). Returns (rows
/// written, qualifying bonus steps, per-street (flop, turn, river) counts,
/// bonus total). A `ValueError` leaves every output as it was.
#[pyfunction]
#[pyo3(signature = (term_envs, lengths, flush_mask, won_bb, share_gt, share_eq, traj, traj_cap, pools, holes_rot, gamma, lam, retro_c, out))]
#[allow(clippy::too_many_arguments)]
pub fn flush_trajectories<'py>(
    py: Python<'py>,
    term_envs: PyReadonlyArray1<'py, i64>,
    lengths: PyReadonlyArray2<'py, i32>,
    flush_mask: PyReadonlyArray2<'py, bool>,
    won_bb: PyReadonlyArray2<'py, f32>,
    share_gt: PyReadonlyArray2<'py, bool>,
    share_eq: PyReadonlyArray2<'py, bool>,
    traj: &Bound<'py, PyDict>,
    traj_cap: usize,
    pools: &Bound<'py, PyDict>,
    holes_rot: PyReadonlyArray2<'py, u8>,
    gamma: f32,
    lam: f32,
    retro_c: f32,
    out: &Bound<'py, PyDict>,
) -> PyResult<FlushSummary> {
    let err = |m: String| PyValueError::new_err(format!("flush_trajectories: {m}"));
    for (name, c) in [
        ("lengths", lengths.is_c_contiguous()),
        ("flush_mask", flush_mask.is_c_contiguous()),
        ("won_bb", won_bb.is_c_contiguous()),
        ("share_gt", share_gt.is_c_contiguous()),
        ("share_eq", share_eq.is_c_contiguous()),
        ("holes_rot", holes_rot.is_c_contiguous()),
    ] {
        if !c {
            return Err(err(format!("{name} must be C-contiguous")));
        }
    }
    let (t_n, s_n) = (lengths.shape()[0], lengths.shape()[1]);
    for (name, shape) in [
        ("flush_mask", flush_mask.shape()),
        ("won_bb", won_bb.shape()),
        ("share_gt", share_gt.shape()),
        ("share_eq", share_eq.shape()),
    ] {
        if shape != [t_n, s_n] {
            return Err(err(format!(
                "{name} shape {shape:?} != lengths ({t_n}, {s_n})"
            )));
        }
    }
    if term_envs.len() != t_n {
        return Err(err("term_envs / lengths / traj_cap disagree".into()));
    }
    let t_obs_idx = get_ro::<i64>(traj, "obs_idx")?;
    let t_gate = get_ro::<i8>(traj, "gate")?;
    let t_chips = get_ro::<i64>(traj, "chips")?;
    let t_sizing = get_ro::<i64>(traj, "sizing")?;
    let t_anchor = get_ro::<i8>(traj, "anchor")?;
    let t_u = get_ro::<f32>(traj, "u")?;
    let t_lp = get_ro::<f32>(traj, "log_p")?;
    let t_glp = get_ro::<f32>(traj, "gate_lp")?;
    let t_alp = get_ro::<f32>(traj, "anchor_lp")?;
    let t_val = get_ro::<f32>(traj, "value")?;
    let t_cost = get_ro::<f32>(traj, "costs")?;
    let t_pot = get_ro::<f32>(traj, "pots")?;
    let t_street = get_ro::<i8>(traj, "streets")?;
    let vrpo = traj.contains("q_taken")?;
    let t_q = if vrpo {
        Some(get_ro::<f32>(traj, "q_taken")?)
    } else {
        None
    };
    let t_vpi = if vrpo {
        Some(get_ro::<f32>(traj, "vpi")?)
    } else {
        None
    };
    // ENG-030: the sizing rows must be exactly 4 wide (a length check alone
    // passed any (M/k, 4k) array).
    if t_sizing.ndim() != 2 || t_sizing.shape()[1] != 4 {
        return Err(err(format!(
            "traj 'sizing' must be (M, 4); got {:?}",
            t_sizing.shape()
        )));
    }
    let p_bits = get_ro::<u8>(pools, "obs_bits")?;
    let p_real = get_ro::<f32>(pools, "obs_real")?;
    let p_gm = get_ro::<bool>(pools, "gm")?;
    if p_bits.ndim() != 2 || p_real.ndim() != 2 || p_gm.ndim() != 2 {
        return Err(err("pool arrays must be 2-D".into()));
    }
    let p_rows = p_bits.shape()[0];
    if p_real.shape()[0] != p_rows || p_gm.shape()[0] != p_rows {
        return Err(err("pool arrays disagree on rows".into()));
    }

    let mut o_bits = get_rw::<u8>(out, "obs_bits")?;
    // Half-precision storage: the caller passes the float16 slab as its u16
    // view under "obs_real_f16" instead of "obs_real".
    let half = out.get_item("obs_real_f16")?.is_some();
    let mut o_real = if half {
        None
    } else {
        Some(get_rw::<f32>(out, "obs_real")?)
    };
    let mut o_real16 = if half {
        Some(get_rw::<u16>(out, "obs_real_f16")?)
    } else {
        None
    };
    let mut o_gm = get_rw::<bool>(out, "gm")?;
    let mut o_ga = get_rw::<i64>(out, "ga")?;
    let mut o_rc = get_rw::<i64>(out, "rc")?;
    let mut o_sz = get_rw::<i64>(out, "sz")?;
    let mut o_an = get_rw::<i64>(out, "an")?;
    let mut o_ru = get_rw::<f32>(out, "ru")?;
    let mut o_oh = get_rw::<u8>(out, "oh")?;
    let mut o_lp = get_rw::<f32>(out, "lp")?;
    let mut o_glp = get_rw::<f32>(out, "glp")?;
    let mut o_alp = get_rw::<f32>(out, "alp")?;
    let mut o_v = get_rw::<f32>(out, "v")?;
    let mut o_ret = get_rw::<f32>(out, "ret")?;
    let mut o_adv = get_rw::<f32>(out, "adv")?;
    let mut o_last = get_rw::<bool>(out, "last")?;
    let q_vpi = match (&t_q, &t_vpi) {
        (Some(q), Some(v)) => Some((q.as_slice()?, v.as_slice()?)),
        _ => None,
    };
    let inp = FlushIn {
        term: term_envs.as_slice()?,
        s_n,
        lengths: lengths.as_slice()?,
        flush_mask: flush_mask.as_slice()?,
        won_bb: won_bb.as_slice()?,
        share_gt: share_gt.as_slice()?,
        share_eq: share_eq.as_slice()?,
        traj: TrajIn {
            obs_idx: t_obs_idx.as_slice()?,
            gate: t_gate.as_slice()?,
            chips: t_chips.as_slice()?,
            sizing: t_sizing.as_slice()?,
            anchor: t_anchor.as_slice()?,
            u: t_u.as_slice()?,
            log_p: t_lp.as_slice()?,
            gate_lp: t_glp.as_slice()?,
            anchor_lp: t_alp.as_slice()?,
            value: t_val.as_slice()?,
            costs: t_cost.as_slice()?,
            pots: t_pot.as_slice()?,
            streets: t_street.as_slice()?,
            q_vpi,
        },
        traj_cap,
        pool: PoolIn {
            bits: p_bits.as_slice()?,
            nb: p_bits.shape()[1],
            real: p_real.as_slice()?,
            nr: p_real.shape()[1],
            gm: p_gm.as_slice()?,
            gm_w: p_gm.shape()[1],
        },
        holes_rot: holes_rot.as_slice()?,
        hole_rows: holes_rot.shape()[0],
        oh_w: holes_rot.shape()[1],
        gamma,
        lam,
        retro_c,
    };
    let real = match (o_real.as_mut(), o_real16.as_mut()) {
        (Some(a), _) => RealOut::F32(a.as_slice_mut()?),
        (_, Some(a)) => RealOut::F16(a.as_slice_mut()?),
        _ => unreachable!("one of obs_real / obs_real_f16 was fetched"),
    };
    let o = FlushOut {
        bits: o_bits.as_slice_mut()?,
        real,
        gm: o_gm.as_slice_mut()?,
        ga: o_ga.as_slice_mut()?,
        rc: o_rc.as_slice_mut()?,
        sz: o_sz.as_slice_mut()?,
        an: o_an.as_slice_mut()?,
        ru: o_ru.as_slice_mut()?,
        oh: o_oh.as_slice_mut()?,
        lp: o_lp.as_slice_mut()?,
        glp: o_glp.as_slice_mut()?,
        alp: o_alp.as_slice_mut()?,
        v: o_v.as_slice_mut()?,
        ret: o_ret.as_slice_mut()?,
        adv: o_adv.as_slice_mut()?,
        last: o_last.as_slice_mut()?,
    };
    let st = py.detach(|| flush_inner(&inp, o)).map_err(err)?;
    Ok((
        st.rows,
        st.bonus_steps,
        (st.by_street[0], st.by_street[1], st.by_street[2]),
        st.bonus_total,
    ))
}

// ---------------------------------------------------------------------------
// record_learner_steps
// ---------------------------------------------------------------------------

/// Parallel `record_learner_steps` in tasks of this many rows...
const REC_MIN_LEN: usize = 1024;
/// ... and only from this many rows up: a step of a few thousand rows costs
/// less sequentially than waking the pool's workers.
const REC_PAR_MIN_ROWS: usize = 8192;

/// One learner step's values (from the per-env arrays).
pub(crate) struct RecIn<'a> {
    pub(crate) gates: &'a [u8],
    pub(crate) chips: &'a [u64],
    pub(crate) sizing: &'a [i64],
    pub(crate) anchors: &'a [i64],
    /// u / log_p / gate_lp / anchor_lp / value (+ q_taken / vpi).
    pub(crate) f32s: Vec<&'a [f32]>,
}

/// The flat trajectory arrays a record writes.
pub(crate) struct RecOut<'a> {
    pub(crate) obs_idx: &'a mut [i64],
    pub(crate) gate: &'a mut [i8],
    pub(crate) chips: &'a mut [i64],
    pub(crate) sizing: &'a mut [i64],
    pub(crate) anchor: &'a mut [i8],
    pub(crate) f32s: Vec<&'a mut [f32]>,
}

impl<'a> RecOut<'a> {
    /// The next `n` slots of every array, split off the front.
    fn split_off(&mut self, n: usize) -> RecOut<'a> {
        RecOut {
            obs_idx: take_front(&mut self.obs_idx, n),
            gate: take_front(&mut self.gate, n),
            chips: take_front(&mut self.chips, n),
            sizing: take_front(&mut self.sizing, 4 * n),
            anchor: take_front(&mut self.anchor, n),
            f32s: self.f32s.iter_mut().map(|a| take_front(a, n)).collect(),
        }
    }

    /// Row i (env `e`) into slot `sl` of these arrays.
    #[inline]
    fn put(&mut self, sl: usize, e: usize, obs: i64, inp: &RecIn<'_>) {
        self.obs_idx[sl] = obs;
        let g = inp.gates[e] as i8;
        self.gate[sl] = g;
        self.chips[sl] = if g == GATE_RAISE {
            inp.chips[e] as i64
        } else {
            0
        };
        self.sizing[sl * 4..sl * 4 + 4].copy_from_slice(&inp.sizing[e * 4..e * 4 + 4]);
        self.anchor[sl] = inp.anchors[e] as i8;
        for (d, s) in self.f32s.iter_mut().zip(&inp.f32s) {
            d[sl] = s[e];
        }
    }
}

/// Validate, then write row i (env `lidx[i]`) into flat slot `slot[i]` for
/// every row. Strictly increasing slots from `REC_PAR_MIN_ROWS` rows up are
/// written in parallel: each task gets the slots from its first row's slot up
/// to the next task's (distinct slots, so the same result as the sequential
/// loop). An `Err` leaves the arrays untouched.
pub(crate) fn record_inner(
    slot: &[i64],
    lidx: &[i64],
    pool_start: i64,
    inp: &RecIn<'_>,
    mut out: RecOut<'_>,
) -> Result<(), String> {
    if slot.len() != lidx.len() {
        return Err("slot / learner_idx lengths differ".into());
    }
    let n = inp.gates.len();
    if inp.chips.len() != n
        || inp.anchors.len() != n
        || inp.sizing.len() != 4 * n
        || inp.f32s.iter().any(|a| a.len() != n)
    {
        return Err(format!(
            "per-env gate / chips / anchor / sizing / float rows differ ({n} gates)"
        ));
    }
    let m = out.gate.len();
    if out.obs_idx.len() != m
        || out.chips.len() != m
        || out.anchor.len() != m
        || out.sizing.len() != 4 * m
        || out.f32s.iter().any(|a| a.len() != m)
        || out.f32s.len() != inp.f32s.len()
    {
        return Err("flat trajectory arrays disagree on slots".into());
    }
    for (&sl, &e) in slot.iter().zip(lidx) {
        if sl < 0 || sl as usize >= m || e < 0 || e as usize >= n {
            return Err(format!(
                "slot {sl} / env {e} out of range ({m} slots, {n} envs)"
            ));
        }
    }
    let rows = slot.len();
    let ascending = slot.windows(2).all(|w| w[0] < w[1]);
    if !ascending || rows < REC_PAR_MIN_ROWS {
        for (i, (&sl, &e)) in slot.iter().zip(lidx).enumerate() {
            out.put(sl as usize, e as usize, pool_start + i as i64, inp);
        }
        return Ok(());
    }
    // Task b owns rows [lo, hi) and the slots [slot[lo], slot[hi]) (the last
    // task: to the end) -- disjoint, since the slots ascend.
    let _ = out.split_off(slot[0] as usize);
    let mut tasks = Vec::with_capacity(rows.div_ceil(REC_MIN_LEN));
    let mut lo = 0;
    while lo < rows {
        let hi = (lo + REC_MIN_LEN).min(rows);
        let span = if hi < rows {
            (slot[hi] - slot[lo]) as usize
        } else {
            m - slot[lo] as usize
        };
        tasks.push((lo, hi, out.split_off(span)));
        lo = hi;
    }
    tasks.into_par_iter().for_each(|(lo, hi, mut part)| {
        let first = slot[lo] as usize;
        for i in lo..hi {
            part.put(
                slot[i] as usize - first,
                lidx[i] as usize,
                pool_start + i as i64,
                inp,
            );
        }
    });
    Ok(())
}

/// The rollout's per-step trajectory record (python/plo5bp/rollout.py
/// step3c/traj_writes) in one call: for every learner row i (env
/// `learner_idx[i]`, flat trajectory slot `slot[i]`), copy that env's step
/// values from the per-env arrays in `per_env` into the FLAT trajectory
/// arrays in `traj` -- exactly the numpy assignments it replaces:
/// obs_idx = pool_start + i, gate = u8 -> i8, chips = chips (u64 -> i64) on a
/// raise else 0, sizing row, anchor = i64 -> i8, and u / log_p / gate_lp /
/// anchor_lp / value (+ q_taken / vpi when `traj` has them) verbatim.
///
/// Rows are written in parallel when `slot` is strictly increasing (the
/// rollout's case: learner envs ascend and each env's (seat, slot) block is
/// its own) -- distinct slots, so the writes are disjoint and the result is
/// the sequential one; any other order takes the sequential loop.
#[pyfunction]
pub fn record_learner_steps<'py>(
    slot: PyReadonlyArray1<'py, i64>,
    learner_idx: PyReadonlyArray1<'py, i64>,
    pool_start: i64,
    traj: &Bound<'py, PyDict>,
    per_env: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let err = |m: String| PyValueError::new_err(format!("record_learner_steps: {m}"));
    let gates = get_ro::<u8>(per_env, "gate")?;
    let chips = get_ro::<u64>(per_env, "chips")?;
    let sizing = get_ro::<i64>(per_env, "sizing")?;
    let anchors = get_ro::<i64>(per_env, "anchor")?;
    if sizing.ndim() != 2 || sizing.shape()[1] != 4 {
        return Err(err(format!(
            "per-env 'sizing' must be (N, 4); got {:?}",
            sizing.shape()
        )));
    }
    let f32_keys = [
        "u",
        "log_p",
        "gate_lp",
        "anchor_lp",
        "value",
        "q_taken",
        "vpi",
    ];
    let vrpo = traj.contains("q_taken")?;
    let keys: &[&str] = if vrpo { &f32_keys } else { &f32_keys[..5] };
    let src = keys
        .iter()
        .map(|k| get_ro::<f32>(per_env, k))
        .collect::<PyResult<Vec<_>>>()?;
    let mut t_obs = get_rw::<i64>(traj, "obs_idx")?;
    let mut t_gate = get_rw::<i8>(traj, "gate")?;
    let mut t_chips = get_rw::<i64>(traj, "chips")?;
    let mut t_sizing = get_rw::<i64>(traj, "sizing")?;
    let mut t_anchor = get_rw::<i8>(traj, "anchor")?;
    let mut dst = keys
        .iter()
        .map(|k| get_rw::<f32>(traj, k))
        .collect::<PyResult<Vec<_>>>()?;
    let inp = RecIn {
        gates: gates.as_slice()?,
        chips: chips.as_slice()?,
        sizing: sizing.as_slice()?,
        anchors: anchors.as_slice()?,
        f32s: src.iter().map(|a| a.as_slice()).collect::<Result<_, _>>()?,
    };
    let out = RecOut {
        obs_idx: t_obs.as_slice_mut()?,
        gate: t_gate.as_slice_mut()?,
        chips: t_chips.as_slice_mut()?,
        sizing: t_sizing.as_slice_mut()?,
        anchor: t_anchor.as_slice_mut()?,
        f32s: dst
            .iter_mut()
            .map(|a| a.as_slice_mut())
            .collect::<Result<_, _>>()?,
    };
    record_inner(
        slot.as_slice()?,
        learner_idx.as_slice()?,
        pool_start,
        &inp,
        out,
    )
    .map_err(err)
}

// ---------------------------------------------------------------------------
// gather_rows_multi
// ---------------------------------------------------------------------------

/// Rows per rayon task in `gather_rows_multi`.
const GATHER_BLOCK_ROWS: usize = 2048;
/// Copies below this many bytes run on the calling thread: handing a job to
/// the thread pool from Python wakes its sleeping workers (~0.2-0.4 ms on the
/// pod), more than a small copy costs.
const GATHER_PAR_MIN_BYTES: usize = 4 << 20;

/// One array pair of a gather: `src_rows` source rows of `w` bytes and the
/// whole destination.
pub(crate) struct GatherPair<'a> {
    pub(crate) src: &'a [u8],
    pub(crate) w: usize,
    pub(crate) dst: &'a mut [u8],
}

/// `dst[dst_start + i] = src[rows[i]]` for every pair and row i (`rows` None:
/// `src[i]`, every source holding the same row count). Validates first; an
/// `Err` leaves the destinations untouched.
pub(crate) fn gather_inner(
    pairs: Vec<GatherPair<'_>>,
    dst_start: usize,
    rows: Option<&[i64]>,
) -> Result<(), String> {
    let mut k: Option<usize> = rows.map(|r| r.len());
    let mut total_w = 0usize;
    for p in &pairs {
        let w = p.w;
        if w == 0 || p.src.len() % w != 0 || p.dst.len() % w != 0 {
            return Err(format!(
                "row widths differ: src {} / dst {} bytes are not rows of {w}",
                p.src.len(),
                p.dst.len()
            ));
        }
        let (src_n, dst_n) = (p.src.len() / w, p.dst.len() / w);
        let kk = match (rows, k) {
            (Some(r), _) => {
                if let Some(&bad) = r.iter().find(|&&x| x < 0 || x as usize >= src_n) {
                    return Err(format!("row {bad} out of range ({src_n} rows)"));
                }
                r.len()
            }
            (None, Some(prev)) if prev != src_n => {
                return Err(format!("sources hold {prev} and {src_n} rows"));
            }
            (None, _) => src_n,
        };
        k = Some(kk);
        if dst_start.checked_add(kk).is_none_or(|end| end > dst_n) {
            return Err(format!(
                "{kk} rows at {dst_start} overflow dst ({dst_n} rows)"
            ));
        }
        total_w += w;
    }
    let k = k.unwrap_or(0);
    if k == 0 || total_w == 0 {
        return Ok(());
    }
    // Each pair's destination rows [dst_start, dst_start + k), then per block.
    let copy = |src: &[u8], w: usize, dst: &mut [u8], lo: usize| match rows {
        Some(r) => {
            for (i, d) in dst.chunks_exact_mut(w).enumerate() {
                let s = r[lo + i] as usize;
                d.copy_from_slice(&src[s * w..(s + 1) * w]);
            }
        }
        None => dst.copy_from_slice(&src[lo * w..lo * w + dst.len()]),
    };
    let targets: Vec<(&[u8], usize, &mut [u8])> = pairs
        .into_iter()
        .map(|p| {
            (
                p.src,
                p.w,
                &mut p.dst[dst_start * p.w..(dst_start + k) * p.w],
            )
        })
        .collect();
    if k * total_w < GATHER_PAR_MIN_BYTES || k <= GATHER_BLOCK_ROWS {
        for (src, w, dst) in targets {
            copy(src, w, dst, 0);
        }
        return Ok(());
    }
    // Block b = rows [b * BLOCK, ...) of every pair: its own sub-slices.
    let n_blocks = k.div_ceil(GATHER_BLOCK_ROWS);
    let mut blocks: Vec<Vec<RowCopy>> = (0..n_blocks).map(|_| Vec::new()).collect();
    for (src, w, dst) in targets {
        for (b, chunk) in dst.chunks_mut(GATHER_BLOCK_ROWS * w).enumerate() {
            blocks[b].push((src, w, chunk));
        }
    }
    blocks.into_par_iter().enumerate().for_each(|(b, jobs)| {
        for (src, w, dst) in jobs {
            copy(src, w, dst, b * GATHER_BLOCK_ROWS);
        }
    });
    Ok(())
}

/// `dsts[j][dst_start + i] = srcs[j][rows[i]]` for every pair j and row i
/// (`rows` None: `srcs[j][i]`, with every source holding the same row count),
/// rows of raw BYTES -- pass any C-contiguous 2-D array as `a.view(np.uint8)`.
/// The rollout's per-step host copies (packed observation rows into the
/// pinned upload slots and the trajectory pool, gate-mask and sizing rows)
/// were single-threaded numpy gathers, one call per array; this is the same
/// bytes for every array in one call, in parallel only when the copy is big.
#[pyfunction]
#[pyo3(signature = (srcs, dsts, dst_start, rows=None))]
pub fn gather_rows_multi<'py>(
    srcs: Vec<PyReadonlyArray2<'py, u8>>,
    mut dsts: Vec<PyReadwriteArray2<'py, u8>>,
    dst_start: usize,
    rows: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<()> {
    let err = |m: String| PyValueError::new_err(format!("gather_rows_multi: {m}"));
    if srcs.len() != dsts.len() {
        return Err(err(format!(
            "{} sources but {} destinations",
            srcs.len(),
            dsts.len()
        )));
    }
    let mut pairs = Vec::with_capacity(srcs.len());
    for (src, dst) in srcs.iter().zip(dsts.iter_mut()) {
        if !src.is_c_contiguous() || !dst.is_c_contiguous() {
            return Err(err("sources and destinations must be C-contiguous".into()));
        }
        let w = src.shape()[1];
        if dst.shape()[1] != w {
            return Err(err(format!(
                "row widths differ: src {w} bytes, dst {} bytes",
                dst.shape()[1]
            )));
        }
        pairs.push(GatherPair {
            src: src.as_slice()?,
            w,
            dst: dst.as_slice_mut()?,
        });
    }
    let rows_arr = match rows.as_ref() {
        Some(r) => Some(r.as_slice()?),
        None => None,
    };
    gather_inner(pairs, dst_start, rows_arr).map_err(err)
}

/// `gather_rows_multi` for one array pair: `dst[dst_start + i] = src[rows[i]]`
/// (`rows` None: `src[i]`). Test helper — the rollout calls
/// `gather_rows_multi` (tests/python/test_gather_rows.py pins the two equal).
#[pyfunction]
#[pyo3(signature = (src, dst, dst_start, rows=None))]
pub fn gather_rows_into<'py>(
    src: PyReadonlyArray2<'py, u8>,
    dst: PyReadwriteArray2<'py, u8>,
    dst_start: usize,
    rows: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<()> {
    gather_rows_multi(vec![src], vec![dst], dst_start, rows).map_err(|e| {
        PyValueError::new_err(
            e.to_string()
                .replace("gather_rows_multi", "gather_rows_into"),
        )
    })
}

#[cfg(test)]
#[path = "flush_tests.rs"]
mod tests;
