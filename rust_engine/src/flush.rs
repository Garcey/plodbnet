//! Rollout trajectory flush (2026-09-23): the per-finished-hand block of
//! `python/plo5bp/rollout.py` (step9b retroactive bonus, step9c GAE / VRPO
//! backward scans, step9d gathers into the output slabs) as ONE parallel pass.
//!
//! Exactness contract: every float32 value is produced by the same IEEE
//! operations in the same order as the numpy code it replaces -- numpy's
//! elementwise ufuncs round each operation to f32 and never fuse, and Rust
//! never contracts `a * b + c` into an FMA, so the bits match. Rows are written
//! in numpy's order (finished hand t, seat s, trajectory slot l -- the C order
//! of the (T, S, L) flush window). Integer counters are exact; the bonus
//! total (a log-line diagnostic) is summed in f64 instead of numpy's
//! pairwise f32 order.

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

/// Raw output pointers; every (hand, seat) writes a disjoint row block.
struct Out {
    bits: *mut u8,
    real: *mut f32,
    gm: *mut bool,
    ga: *mut i64,
    rc: *mut i64,
    sz: *mut i64,
    an: *mut i64,
    ru: *mut f32,
    oh: *mut u8,
    lp: *mut f32,
    glp: *mut f32,
    alp: *mut f32,
    v: *mut f32,
    ret: *mut f32,
    adv: *mut f32,
    last: *mut bool,
}
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

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
/// `holes_rot` is (n_envs * S, 5 * hole_w) u8. Returns (rows written,
/// qualifying bonus steps, per-street (flop, turn, river) counts, bonus total).
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
) -> PyResult<(usize, u64, (u64, u64, u64), f64)> {
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
    let term = term_envs.as_slice()?;
    let (t_n, s_n) = (lengths.shape()[0], lengths.shape()[1]);
    for (name, shape) in [
        ("flush_mask", flush_mask.shape()),
        ("won_bb", won_bb.shape()),
        ("share_gt", share_gt.shape()),
        ("share_eq", share_eq.shape()),
    ] {
        if shape != [t_n, s_n] {
            return Err(err(format!("{name} shape {shape:?} != lengths ({t_n}, {s_n})")));
        }
    }
    if term.len() != t_n || s_n == 0 || traj_cap == 0 {
        return Err(err("term_envs / lengths / traj_cap disagree".into()));
    }
    let lens = lengths.as_slice()?;
    let fmask = flush_mask.as_slice()?;
    let won = won_bb.as_slice()?;
    let sgt = share_gt.as_slice()?;
    let seq = share_eq.as_slice()?;
    let holes = holes_rot.as_slice()?;
    let oh_w = holes_rot.shape()[1];
    let n_env_seats = holes_rot.shape()[0];

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
    let t_q = if vrpo { Some(get_ro::<f32>(traj, "q_taken")?) } else { None };
    let t_vpi = if vrpo { Some(get_ro::<f32>(traj, "vpi")?) } else { None };
    let m = t_gate.len();
    for (name, len) in [
        ("obs_idx", t_obs_idx.len()),
        ("chips", t_chips.len()),
        ("anchor", t_anchor.len()),
        ("u", t_u.len()),
        ("log_p", t_lp.len()),
        ("gate_lp", t_glp.len()),
        ("anchor_lp", t_alp.len()),
        ("value", t_val.len()),
        ("costs", t_cost.len()),
        ("pots", t_pot.len()),
        ("streets", t_street.len()),
        ("sizing/4", t_sizing.len() / 4),
    ] {
        if len != m {
            return Err(err(format!("trajectory array '{name}' has {len} slots, gate has {m}")));
        }
    }
    if let (Some(q), Some(v)) = (&t_q, &t_vpi) {
        if q.len() != m || v.len() != m {
            return Err(err("q_taken / vpi slot counts differ from gate".into()));
        }
    }
    if m != n_env_seats * traj_cap {
        return Err(err(format!(
            "{m} trajectory slots != holes_rot rows {n_env_seats} x traj_cap {traj_cap}"
        )));
    }

    let p_bits = get_ro::<u8>(pools, "obs_bits")?;
    let p_real = get_ro::<f32>(pools, "obs_real")?;
    let p_gm = get_ro::<bool>(pools, "gm")?;
    if p_bits.ndim() != 2 || p_real.ndim() != 2 || p_gm.ndim() != 2 || t_sizing.ndim() != 2 {
        return Err(err("pool arrays and traj 'sizing' must be 2-D".into()));
    }
    let (nb, nr, gm_w) = (p_bits.shape()[1], p_real.shape()[1], p_gm.shape()[1]);
    let p_rows = p_bits.shape()[0];
    if p_real.shape()[0] != p_rows || p_gm.shape()[0] != p_rows {
        return Err(err("pool arrays disagree on rows".into()));
    }

    // Row offsets per (hand, seat): C order of the (T, S, L) window.
    let mut offsets = vec![0usize; t_n * s_n + 1];
    for k in 0..t_n * s_n {
        let len = if fmask[k] { lens[k].max(0) as usize } else { 0 };
        offsets[k + 1] = offsets[k] + len;
    }
    let n_new = offsets[t_n * s_n];
    for (k, &e) in term.iter().enumerate() {
        if e < 0 || (e as usize + 1) * s_n > n_env_seats {
            return Err(err(format!("term_envs[{k}] = {e} out of range")));
        }
    }
    for k in 0..t_n * s_n {
        if fmask[k] && lens[k] as usize > traj_cap {
            return Err(err(format!("trajectory length {} > traj_cap {traj_cap}", lens[k])));
        }
    }

    let mut o_bits = get_rw::<u8>(out, "obs_bits")?;
    let mut o_real = get_rw::<f32>(out, "obs_real")?;
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
    for (name, len, w) in [
        ("obs_bits", o_bits.len(), nb),
        ("obs_real", o_real.len(), nr),
        ("gm", o_gm.len(), gm_w),
        ("ga", o_ga.len(), 1),
        ("rc", o_rc.len(), 1),
        ("sz", o_sz.len(), 4),
        ("an", o_an.len(), 1),
        ("ru", o_ru.len(), 1),
        ("oh", o_oh.len(), oh_w),
        ("lp", o_lp.len(), 1),
        ("glp", o_glp.len(), 1),
        ("alp", o_alp.len(), 1),
        ("v", o_v.len(), 1),
        ("ret", o_ret.len(), 1),
        ("adv", o_adv.len(), 1),
        ("last", o_last.len(), 1),
    ] {
        if len != n_new * w {
            return Err(err(format!(
                "output '{name}' holds {len} values, expected {n_new} rows x {w}"
            )));
        }
    }
    let o = Out {
        bits: o_bits.as_slice_mut()?.as_mut_ptr(),
        real: o_real.as_slice_mut()?.as_mut_ptr(),
        gm: o_gm.as_slice_mut()?.as_mut_ptr(),
        ga: o_ga.as_slice_mut()?.as_mut_ptr(),
        rc: o_rc.as_slice_mut()?.as_mut_ptr(),
        sz: o_sz.as_slice_mut()?.as_mut_ptr(),
        an: o_an.as_slice_mut()?.as_mut_ptr(),
        ru: o_ru.as_slice_mut()?.as_mut_ptr(),
        oh: o_oh.as_slice_mut()?.as_mut_ptr(),
        lp: o_lp.as_slice_mut()?.as_mut_ptr(),
        glp: o_glp.as_slice_mut()?.as_mut_ptr(),
        alp: o_alp.as_slice_mut()?.as_mut_ptr(),
        v: o_v.as_slice_mut()?.as_mut_ptr(),
        ret: o_ret.as_slice_mut()?.as_mut_ptr(),
        adv: o_adv.as_slice_mut()?.as_mut_ptr(),
        last: o_last.as_slice_mut()?.as_mut_ptr(),
    };

    let (obs_idx, gate, chips, sizing, anchor) = (
        t_obs_idx.as_slice()?,
        t_gate.as_slice()?,
        t_chips.as_slice()?,
        t_sizing.as_slice()?,
        t_anchor.as_slice()?,
    );
    let (u, lp, glp, alp, val) = (
        t_u.as_slice()?,
        t_lp.as_slice()?,
        t_glp.as_slice()?,
        t_alp.as_slice()?,
        t_val.as_slice()?,
    );
    let (cost, pot, street) = (t_cost.as_slice()?, t_pot.as_slice()?, t_street.as_slice()?);
    let q_sl = match &t_q {
        Some(a) => Some(a.as_slice()?),
        None => None,
    };
    let vpi_sl = match &t_vpi {
        Some(a) => Some(a.as_slice()?),
        None => None,
    };
    let (pb, pr, pg) = (p_bits.as_slice()?, p_real.as_slice()?, p_gm.as_slice()?);
    // Pool rows referenced by the slots being flushed must exist.
    for k in 0..t_n * s_n {
        if !fmask[k] {
            continue;
        }
        let base = (term[k / s_n] as usize * s_n + k % s_n) * traj_cap;
        for l in 0..lens[k] as usize {
            let p = obs_idx[base + l];
            if p < 0 || p as usize >= p_rows {
                return Err(err(format!("pool index {p} out of range ({p_rows} rows)")));
            }
        }
    }
    // numpy: `gamma_f * lam_f * last_gae` evaluates (gamma_f * lam_f) first.
    let gl = gamma * lam;

    let (bonus_steps, by_street, bonus_total) = py.allow_threads(|| {
        (0..t_n * s_n)
            .into_par_iter()
            .map(|k| {
                let o = &o; // capture the Send/Sync wrapper whole, not its fields
                let mut acc = (0u64, [0u64; 3], 0f64);
                let len = if fmask[k] { lens[k].max(0) as usize } else { 0 };
                if len == 0 {
                    return acc;
                }
                let (t, s) = (k / s_n, k % s_n);
                let es = term[t] as usize * s_n + s;
                let base = es * traj_cap;
                let r0 = offsets[k];
                let (gt, eq, won_k) = (sgt[k], seq[k], won[k]);
                // step9b: qualification + the (optional) retroactive bonus
                // applied to a COPY of the costs.
                let mut c_eff = [0f32; 256];
                let mut c_heap: Vec<f32>;
                let costs: &mut [f32] = if len <= c_eff.len() {
                    &mut c_eff[..len]
                } else {
                    c_heap = vec![0f32; len];
                    &mut c_heap[..]
                };
                for l in 0..len {
                    let g = gate[base + l];
                    let c = cost[base + l];
                    let is_raise = g == GATE_RAISE;
                    let call_chips = g == GATE_CHECK_CALL && c < 0.0;
                    let q = (gt && is_raise) || (eq && (is_raise || call_chips));
                    if q {
                        acc.0 += 1;
                        let st = street[base + l];
                        if (1..=3).contains(&st) {
                            acc.1[(st - 1) as usize] += 1;
                        }
                    }
                    costs[l] = if retro_c != 0.0 {
                        let qf: f32 = if q { 1.0 } else { 0.0 };
                        let b = qf * (retro_c * pot[base + l]);
                        acc.2 += b as f64;
                        c + b
                    } else {
                        c
                    };
                }
                // step9c: GAE and (VRPO) Expected-SARSA traces, backward.
                let mut last_gae = 0f32;
                let mut last_es = 0f32;
                for l in (0..len).rev() {
                    let is_last = l == len - 1;
                    let reward = costs[l] + if is_last { won_k } else { 0.0 };
                    let v_l = val[base + l];
                    let next_v = if is_last { 0.0 } else { val[base + l + 1] };
                    let delta = (reward + gamma * next_v) - v_l;
                    last_gae = delta + gl * last_gae;
                    let adv_l = match (q_sl, vpi_sl) {
                        (Some(qs), Some(vs)) => {
                            let next_vpi = if is_last { 0.0 } else { vs[base + l + 1] };
                            let q_l = qs[base + l];
                            let d_es = (reward + gamma * next_vpi) - q_l;
                            last_es = d_es + gl * last_es;
                            (q_l - vs[base + l]) + last_es
                        }
                        _ => last_gae,
                    };
                    let r = r0 + l;
                    let p = obs_idx[base + l] as usize;
                    // SAFETY: rows r0..r0+len belong to this (hand, seat) only;
                    // every index was bounds-checked above.
                    unsafe {
                        std::ptr::copy_nonoverlapping(pb.as_ptr().add(p * nb), o.bits.add(r * nb), nb);
                        std::ptr::copy_nonoverlapping(pr.as_ptr().add(p * nr), o.real.add(r * nr), nr);
                        std::ptr::copy_nonoverlapping(pg.as_ptr().add(p * gm_w), o.gm.add(r * gm_w), gm_w);
                        *o.ga.add(r) = gate[base + l] as i64;
                        *o.rc.add(r) = chips[base + l];
                        std::ptr::copy_nonoverlapping(sizing.as_ptr().add((base + l) * 4), o.sz.add(r * 4), 4);
                        *o.an.add(r) = anchor[base + l] as i64;
                        *o.ru.add(r) = u[base + l];
                        std::ptr::copy_nonoverlapping(holes.as_ptr().add(es * oh_w), o.oh.add(r * oh_w), oh_w);
                        *o.lp.add(r) = lp[base + l];
                        *o.glp.add(r) = glp[base + l];
                        *o.alp.add(r) = alp[base + l];
                        *o.v.add(r) = v_l;
                        *o.ret.add(r) = last_gae + v_l;
                        *o.adv.add(r) = adv_l;
                        *o.last.add(r) = is_last;
                    }
                }
                acc
            })
            .reduce(
                || (0u64, [0u64; 3], 0f64),
                |a, b| (a.0 + b.0, [a.1[0] + b.1[0], a.1[1] + b.1[1], a.1[2] + b.1[2]], a.2 + b.2),
            )
    });
    Ok((n_new, bonus_steps, (by_street[0], by_street[1], by_street[2]), bonus_total))
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
    let (slot, lidx) = (slot.as_slice()?, learner_idx.as_slice()?);
    if slot.len() != lidx.len() {
        return Err(err("slot / learner_idx lengths differ".into()));
    }
    let gates = get_ro::<u8>(per_env, "gate")?;
    let chips = get_ro::<u64>(per_env, "chips")?;
    let sizing = get_ro::<i64>(per_env, "sizing")?;
    let anchors = get_ro::<i64>(per_env, "anchor")?;
    let n = gates.len();
    let f32_keys = ["u", "log_p", "gate_lp", "anchor_lp", "value", "q_taken", "vpi"];
    let vrpo = traj.contains("q_taken")?;
    let keys: &[&str] = if vrpo { &f32_keys } else { &f32_keys[..5] };
    let mut src = Vec::with_capacity(keys.len());
    for k in keys {
        let a = get_ro::<f32>(per_env, k)?;
        if a.len() != n {
            return Err(err(format!("per-env '{k}' has {} rows, gate has {n}", a.len())));
        }
        src.push(a);
    }
    if chips.len() != n || anchors.len() != n || sizing.len() != 4 * n {
        return Err(err("per-env gate / chips / anchor / sizing row counts differ".into()));
    }
    let mut t_obs = get_rw::<i64>(traj, "obs_idx")?;
    let mut t_gate = get_rw::<i8>(traj, "gate")?;
    let mut t_chips = get_rw::<i64>(traj, "chips")?;
    let mut t_sizing = get_rw::<i64>(traj, "sizing")?;
    let mut t_anchor = get_rw::<i8>(traj, "anchor")?;
    let mut dst = Vec::with_capacity(keys.len());
    for k in keys {
        dst.push(get_rw::<f32>(traj, k)?);
    }
    let m = t_gate.len();
    if t_obs.len() != m || t_chips.len() != m || t_anchor.len() != m || t_sizing.len() != 4 * m
        || dst.iter().any(|a| a.len() != m)
    {
        return Err(err("flat trajectory arrays disagree on slots".into()));
    }
    for (&sl, &e) in slot.iter().zip(lidx) {
        if sl < 0 || sl as usize >= m || e < 0 || e as usize >= n {
            return Err(err(format!("slot {sl} / env {e} out of range ({m} slots, {n} envs)")));
        }
    }
    let (g, c, sz, an) = (gates.as_slice()?, chips.as_slice()?, sizing.as_slice()?, anchors.as_slice()?);
    let (to, tg, tc, ts, ta) = (
        t_obs.as_slice_mut()?,
        t_gate.as_slice_mut()?,
        t_chips.as_slice_mut()?,
        t_sizing.as_slice_mut()?,
        t_anchor.as_slice_mut()?,
    );
    let mut srcs: Vec<&[f32]> = Vec::with_capacity(src.len());
    for a in src.iter() {
        srcs.push(a.as_slice()?);
    }
    let mut dsts: Vec<&mut [f32]> = Vec::with_capacity(dst.len());
    for a in dst.iter_mut() {
        dsts.push(a.as_slice_mut()?);
    }
    let ascending = slot.windows(2).all(|w| w[0] < w[1]);
    if !ascending || slot.len() < 2 * REC_MIN_LEN {
        for (i, (&sl, &e)) in slot.iter().zip(lidx).enumerate() {
            let (sl, e) = (sl as usize, e as usize);
            to[sl] = pool_start + i as i64;
            tg[sl] = g[e] as i8;
            tc[sl] = if g[e] as i8 == GATE_RAISE { c[e] as i64 } else { 0 };
            ts[sl * 4..sl * 4 + 4].copy_from_slice(&sz[e * 4..e * 4 + 4]);
            ta[sl] = an[e] as i8;
        }
        for (sv, dv) in srcs.iter().zip(dsts.iter_mut()) {
            for (&sl, &e) in slot.iter().zip(lidx) {
                dv[sl as usize] = sv[e as usize];
            }
        }
        return Ok(());
    }
    // Strictly increasing slots (all in range, checked above) are distinct:
    // every row owns its own element of every destination array.
    let o = RecOut {
        to: to.as_mut_ptr(),
        tg: tg.as_mut_ptr(),
        tc: tc.as_mut_ptr(),
        ts: ts.as_mut_ptr(),
        ta: ta.as_mut_ptr(),
        f: dsts.iter_mut().map(|d| d.as_mut_ptr()).collect(),
    };
    (0..slot.len()).into_par_iter().with_min_len(REC_MIN_LEN).for_each(|i| {
        let o = &o;
        let (sl, e) = (slot[i] as usize, lidx[i] as usize);
        // SAFETY: `sl` < every destination's length (checked) and unique to
        // row i (strictly increasing), so no two rows touch the same element.
        unsafe {
            *o.to.add(sl) = pool_start + i as i64;
            let gi = g[e] as i8;
            *o.tg.add(sl) = gi;
            *o.tc.add(sl) = if gi == GATE_RAISE { c[e] as i64 } else { 0 };
            std::ptr::copy_nonoverlapping(sz.as_ptr().add(e * 4), o.ts.add(sl * 4), 4);
            *o.ta.add(sl) = an[e] as i8;
            for (sv, dp) in srcs.iter().zip(o.f.iter()) {
                *dp.add(sl) = sv[e];
            }
        }
    });
    Ok(())
}

/// Parallel `record_learner_steps` below this many rows per task.
const REC_MIN_LEN: usize = 1024;

struct RecOut {
    to: *mut i64,
    tg: *mut i8,
    tc: *mut i64,
    ts: *mut i64,
    ta: *mut i8,
    f: Vec<*mut f32>,
}
unsafe impl Send for RecOut {}
unsafe impl Sync for RecOut {}

/// Rows per rayon task in `gather_rows_into`.
const GATHER_MIN_ROWS: usize = 512;

/// `dst[dst_start + i] = src[rows[i]]` for every i (`rows` None: `src[i]`),
/// rows of raw BYTES -- pass any C-contiguous 2-D array as `a.view(np.uint8)`.
/// The rollout's per-step host copies (packed observation rows into the
/// pinned upload slots and the trajectory pool, gate-mask rows) were
/// single-threaded numpy gathers; this is the same bytes, copied in parallel.
#[pyfunction]
#[pyo3(signature = (src, dst, dst_start, rows=None))]
pub fn gather_rows_into<'py>(
    src: PyReadonlyArray2<'py, u8>,
    mut dst: PyReadwriteArray2<'py, u8>,
    dst_start: usize,
    rows: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<()> {
    let err = |m: String| PyValueError::new_err(format!("gather_rows_into: {m}"));
    if !src.is_c_contiguous() || !dst.is_c_contiguous() {
        return Err(err("src and dst must be C-contiguous".into()));
    }
    let (src_n, w) = (src.shape()[0], src.shape()[1]);
    let (dst_n, dst_w) = (dst.shape()[0], dst.shape()[1]);
    if dst_w != w {
        return Err(err(format!("row widths differ: src {w} bytes, dst {dst_w} bytes")));
    }
    let rows_arr = match rows.as_ref() {
        Some(r) => Some(r.as_slice()?),
        None => None,
    };
    let k = rows_arr.map_or(src_n, |r| r.len());
    if dst_start.checked_add(k).map_or(true, |end| end > dst_n) {
        return Err(err(format!("{k} rows at {dst_start} overflow dst ({dst_n} rows)")));
    }
    if let Some(r) = rows_arr {
        if let Some(&bad) = r.iter().find(|&&x| x < 0 || x as usize >= src_n) {
            return Err(err(format!("row {bad} out of range ({src_n} rows)")));
        }
    }
    if k == 0 || w == 0 {
        return Ok(());
    }
    let sv = src.as_slice()?;
    let dv = &mut dst.as_slice_mut()?[dst_start * w..(dst_start + k) * w];
    match rows_arr {
        Some(r) => dv
            .par_chunks_mut(w)
            .with_min_len(GATHER_MIN_ROWS)
            .zip(r.par_iter())
            .for_each(|(d, &ri)| {
                let ri = ri as usize;
                d.copy_from_slice(&sv[ri * w..(ri + 1) * w]);
            }),
        None => dv
            .par_chunks_mut(w * GATHER_MIN_ROWS)
            .zip(sv[..k * w].par_chunks(w * GATHER_MIN_ROWS))
            .for_each(|(d, s)| d.copy_from_slice(s)),
    }
    Ok(())
}
