//! Rollout bookkeeping kernels: the aggression bonus and its trajectory record.
//!
//! Both kernels share ONE copy of the per-env arithmetic (`AggrIn::step`,
//! ENG-021), read their inputs as slices without copying them, and write the
//! trajectory through per-env sub-slices (no raw pointers, ENG-013).

use super::*;

/// Mirror of `python/plo5bp/rollout.py:_aggression_bonus_bb`. Pot-fraction
/// bonus on voluntary aggression: chips committed beyond the actor's
/// amount-to-call, capped at the pre-step pot. Zero for non-RAISE gates
/// or non-positive `c`.
pub(super) const GATE_RAISE_U8: u8 = 2;

#[inline]
pub(super) fn aggression_bonus_bb_inner(
    gate: u8,
    commit_delta_chips: i64,
    bet_to_call_chips: i64,
    street_commit_actor_chips: i64,
    pot_chips_pre: i64,
    c: f64,
) -> f64 {
    if c <= 0.0 || gate != GATE_RAISE_U8 {
        return 0.0;
    }
    let call_chips = (bet_to_call_chips - street_commit_actor_chips).max(0);
    let aggressive = (commit_delta_chips - call_chips).max(0);
    if aggressive <= 0 || pot_chips_pre <= 0 {
        return 0.0;
    }
    let mut ratio = aggressive as f64 / pot_chips_pre as f64;
    if ratio > 1.0 {
        ratio = 1.0;
    }
    c * ratio
}

/// One rollout step's inputs over the batch: 1-D arrays of `n` envs, 2-D ones
/// (`n`, `s_n`) row-major.
struct AggrIn<'a> {
    s_n: usize,
    actors: &'a [i8],
    dones: &'a [bool],
    learner_mask: &'a [bool],
    gates: &'a [u8],
    pre_tc: &'a [i64],
    post_tc: &'a [i64],
    pre_btc: &'a [u64],
    pre_sc: &'a [u64],
    pre_street: &'a [u8],
    c: f64,
    reward_norm: f64,
}

/// One learner step's bookkeeping.
#[derive(Clone, Copy, Debug)]
struct AggrStep {
    actor: usize,
    /// `-(delta_chips * reward_norm) + bonus_bb`.
    cost_inc: f64,
    pot_pre_bb: f64,
    street: i8,
    bonus_bb: f64,
}

impl AggrIn<'_> {
    /// Env `i`'s step; None when it contributed no learner step (done, no
    /// actor, or the actor is not a learner seat).
    #[inline]
    fn step(&self, i: usize) -> Option<AggrStep> {
        if self.dones[i] {
            return None;
        }
        let a = self.actors[i];
        if a < 0 || a as usize >= self.s_n {
            return None;
        }
        let (actor, row) = (a as usize, i * self.s_n);
        if !self.learner_mask[row + actor] {
            return None;
        }
        let delta = (self.post_tc[row + actor] - self.pre_tc[row + actor]).max(0);
        let pot_pre = self.pre_tc[row..row + self.s_n].iter().sum::<i64>().max(0);
        let bonus_bb = aggression_bonus_bb_inner(
            self.gates[i],
            delta,
            self.pre_btc[i] as i64,
            self.pre_sc[row + actor] as i64,
            pot_pre,
            self.c,
        );
        Some(AggrStep {
            actor,
            cost_inc: -(delta as f64) * self.reward_norm + bonus_bb,
            pot_pre_bb: pot_pre as f64 * self.reward_norm,
            street: self.pre_street[i] as i8,
            bonus_bb,
        })
    }
}

/// The serial, env-ordered reductions both kernels return: (total bonus bb,
/// steps, bonus steps, steps by street (flop, turn, river), bonus steps by
/// street).
type AggrTally = (f64, u64, u64, [u64; 3], [u64; 3]);

fn tally<'a>(steps: impl Iterator<Item = &'a Option<AggrStep>>) -> AggrTally {
    let mut t: AggrTally = (0.0, 0, 0, [0; 3], [0; 3]);
    for st in steps.flatten() {
        t.1 += 1;
        t.0 += st.bonus_bb;
        if st.bonus_bb > 0.0 {
            t.2 += 1;
        }
        let bucket = st.street as i64 - 1;
        if (0..3).contains(&bucket) {
            t.3[bucket as usize] += 1;
            if st.bonus_bb > 0.0 {
                t.4[bucket as usize] += 1;
            }
        }
    }
    t
}

/// `compute_aggression_bonus_batch` + the rollout's per-step record of its
/// results (python/plo5bp/rollout.py step8) in ONE call: for every valid env
/// (live, learner seat acting) the step's cost increment, pre-step pot (bb)
/// and street go into the flat trajectory arrays at that seat's current slot
/// (`(env * S + actor) * traj_cap + traj_lengths[env, actor]`), and the slot
/// count advances -- exactly the numpy assignments it replaces, including the
/// f64 -> f32 roundings (`as f32` rounds to nearest-even like numpy's cast).
/// Returns (total bonus bb, steps, bonus steps, steps by street (flop, turn,
/// river), bonus steps by street); the bonus total is summed in env order, as
/// the numpy path did.
///
/// A `ValueError` (bad shapes, or a slot count outside `[0, traj_cap)`) leaves
/// `traj_lengths` exactly as it was (ENG-030); trajectory values past the
/// lengths are scratch the rollout never reads.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn aggression_record_batch<'py>(
    py: Python<'py>,
    actors: PyReadonlyArray1<'_, i8>,
    dones: PyReadonlyArray1<'_, bool>,
    learner_mask: PyReadonlyArray2<'_, bool>,
    gates: PyReadonlyArray1<'_, u8>,
    pre_total_commit: PyReadonlyArray2<'_, i64>,
    post_total_commit: PyReadonlyArray2<'_, i64>,
    pre_bet_to_call: PyReadonlyArray1<'_, u64>,
    pre_street_commit: PyReadonlyArray2<'_, u64>,
    pre_street: PyReadonlyArray1<'_, u8>,
    c: f64,
    reward_norm: f64,
    mut costs: PyReadwriteArray1<'_, f32>,
    mut pots: PyReadwriteArray1<'_, f32>,
    mut streets: PyReadwriteArray1<'_, i8>,
    mut traj_lengths: PyReadwriteArray2<'_, i32>,
    traj_cap: usize,
) -> PyResult<AggrTallyPy> {
    let err = |m: String| PyValueError::new_err(format!("aggression_record_batch: {m}"));
    let n = actors.len();
    let s_n = pre_total_commit.shape()[1];
    for (name, shape, contig) in [
        (
            "learner_mask",
            learner_mask.shape(),
            learner_mask.is_c_contiguous(),
        ),
        (
            "pre_total_commit",
            pre_total_commit.shape(),
            pre_total_commit.is_c_contiguous(),
        ),
        (
            "post_total_commit",
            post_total_commit.shape(),
            post_total_commit.is_c_contiguous(),
        ),
        (
            "pre_street_commit",
            pre_street_commit.shape(),
            pre_street_commit.is_c_contiguous(),
        ),
        (
            "traj_lengths",
            traj_lengths.shape(),
            traj_lengths.is_c_contiguous(),
        ),
    ] {
        if shape != [n, s_n] || !contig {
            return Err(err(format!(
                "{name} must be a C-contiguous ({n}, {s_n}) array"
            )));
        }
    }
    let inp = AggrIn {
        s_n,
        actors: actors.as_slice()?,
        dones: dones.as_slice()?,
        learner_mask: learner_mask.as_slice()?,
        gates: gates.as_slice()?,
        pre_tc: pre_total_commit.as_slice()?,
        post_tc: post_total_commit.as_slice()?,
        pre_btc: pre_bet_to_call.as_slice()?,
        pre_sc: pre_street_commit.as_slice()?,
        pre_street: pre_street.as_slice()?,
        c,
        reward_norm,
    };
    if [
        inp.dones.len(),
        inp.gates.len(),
        inp.pre_btc.len(),
        inp.pre_street.len(),
    ] != [n; 4]
    {
        return Err(err("1-D inputs must all have num_envs entries".into()));
    }
    let m = costs.len();
    let block = s_n * traj_cap;
    if pots.len() != m || streets.len() != m || n * block > m {
        return Err(err(format!(
            "trajectory arrays hold {m} / {} / {} slots, need {} ({n} envs x {s_n} seats x {traj_cap})",
            pots.len(),
            streets.len(),
            n * block
        )));
    }
    // Env i owns trajectory slots [i * S * cap, (i + 1) * S * cap) and length
    // row i: split into those blocks and hand each env its own.
    let costs_s = &mut costs.as_slice_mut()?[..n * block];
    let pots_s = &mut pots.as_slice_mut()?[..n * block];
    let streets_s = &mut streets.as_slice_mut()?[..n * block];
    let lengths_s = traj_lengths.as_slice_mut()?;
    // Per env: (the step, whether its slot count was out of range).
    let per_env: Vec<(Option<AggrStep>, bool)> = py.detach(|| {
        if block == 0 {
            return (0..n)
                .map(|i| (inp.step(i), inp.step(i).is_some()))
                .collect();
        }
        costs_s
            .par_chunks_mut(block)
            .zip(pots_s.par_chunks_mut(block))
            .zip(streets_s.par_chunks_mut(block))
            .zip(lengths_s.par_chunks_mut(s_n))
            .enumerate()
            .with_min_len(APPLY_MIN_LEN)
            .map(|(i, (((cs, ps), ss), ls))| {
                let Some(st) = inp.step(i) else {
                    return (None, false);
                };
                let slot = ls[st.actor];
                if slot < 0 || slot as usize >= traj_cap {
                    return (Some(st), true);
                }
                let at = st.actor * traj_cap + slot as usize;
                cs[at] = st.cost_inc as f32;
                ps[at] = st.pot_pre_bb as f32;
                ss[at] = st.street;
                ls[st.actor] = slot + 1;
                (Some(st), false)
            })
            .collect()
    });
    if per_env.iter().any(|&(_, bad)| bad) {
        // Undo the slot counts this call advanced, so the error leaves them as
        // they were.
        for (i, &(st, bad)) in per_env.iter().enumerate() {
            if let (Some(st), false) = (st, bad) {
                lengths_s[i * s_n + st.actor] -= 1;
            }
        }
        return Err(err(format!("a trajectory slot is outside [0, {traj_cap})")));
    }
    let t = tally(per_env.iter().map(|(st, _)| st));
    Ok((
        t.0,
        t.1,
        t.2,
        (t.3[0], t.3[1], t.3[2]),
        (t.4[0], t.4[1], t.4[2]),
    ))
}

/// The Python view of an [`AggrTally`].
type AggrTallyPy = (f64, u64, u64, (u64, u64, u64), (u64, u64, u64));

/// Per-env aggression-bonus computation for the batched rollout driver,
/// parallelized via rayon with the GIL released (the numpy-flush fallback of
/// `aggression_record_batch`).
///
/// Inputs (all length `N` along axis 0; any memory layout):
/// - `actors`: current actor seat (-1 for terminal).
/// - `dones`: env terminated flag.
/// - `learner_mask`: `(N, num_seats)` — true where seat is a learner seat
///   in env `i`.
/// - `gates`: emitted hybrid gate per env.
/// - `pre_total_commit` / `post_total_commit`: `(N, num_seats)` cumulative
///   per-seat chip commit, snapshotted before/after the current step.
/// - `pre_bet_to_call`: env-level max street_commit before the step.
/// - `pre_street_commit`: `(N, num_seats)` per-seat street_commit before.
/// - `pre_street`: street index before the step (0=preflop ... 3=river).
/// - `c`: aggression-bonus coefficient.
/// - `reward_norm`: `1 / bb`, applied to per-step delta and pot to keep
///   the trajectory in bb units.
///
/// Returns a dict with per-env arrays plus pre-reduced scalar diagnostics:
/// - `valid`: `(N,)` bool — true iff the env contributed a learner step.
/// - `cost_increment`: `(N,)` f64 — `-(delta_chips * reward_norm) + bonus_bb`.
/// - `pot_pre_bb`: `(N,)` f64 — pre-step pot in bb.
/// - `street_pre`: `(N,)` i8 — engine street index, copied from input (-1
///   where `!valid`).
/// - `bonus_bb`: `(N,)` f64 — per-env bonus contribution (zero where
///   `!valid`).
/// - `total_bonus_bb`, `total_steps`, `bonus_steps`: serial reductions.
/// - `steps_by_street` / `bonus_steps_by_street`: 3-elem u64 arrays
///   bucketed by `(street_pre - 1)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn compute_aggression_bonus_batch<'py>(
    py: Python<'py>,
    actors: PyReadonlyArray1<'_, i8>,
    dones: PyReadonlyArray1<'_, bool>,
    learner_mask: PyReadonlyArray2<'_, bool>,
    gates: PyReadonlyArray1<'_, u8>,
    pre_total_commit: PyReadonlyArray2<'_, i64>,
    post_total_commit: PyReadonlyArray2<'_, i64>,
    pre_bet_to_call: PyReadonlyArray1<'_, u64>,
    pre_street_commit: PyReadonlyArray2<'_, u64>,
    pre_street: PyReadonlyArray1<'_, u8>,
    c: f64,
    reward_norm: f64,
) -> PyResult<Bound<'py, PyDict>> {
    // Borrowed as-is when already standard-layout (the rollout's case): no
    // copies (ENG-021 — they used to be materialised just to release the GIL).
    let (a1, d1, g1) = (actors.as_array(), dones.as_array(), gates.as_array());
    let (btc1, st1) = (pre_bet_to_call.as_array(), pre_street.as_array());
    let n = a1.len();
    if d1.len() != n || g1.len() != n || btc1.len() != n || st1.len() != n {
        return Err(PyValueError::new_err(
            "1-D input lengths must all equal num_envs",
        ));
    }
    let (lm, pre_tc, post_tc, pre_sc) = (
        learner_mask.as_array(),
        pre_total_commit.as_array(),
        post_total_commit.as_array(),
        pre_street_commit.as_array(),
    );
    if pre_tc.shape()[0] != n {
        return Err(PyValueError::new_err(
            "pre_total_commit shape[0] must equal num_envs",
        ));
    }
    let s_n = pre_tc.shape()[1];
    if lm.shape() != [n, s_n] || post_tc.shape() != [n, s_n] || pre_sc.shape() != [n, s_n] {
        return Err(PyValueError::new_err(
            "2-D input shapes must all equal (num_envs, num_seats)",
        ));
    }
    let (a1, d1, g1) = (
        a1.as_standard_layout(),
        d1.as_standard_layout(),
        g1.as_standard_layout(),
    );
    let (btc1, st1) = (btc1.as_standard_layout(), st1.as_standard_layout());
    let (lm, pre_tc) = (lm.as_standard_layout(), pre_tc.as_standard_layout());
    let (post_tc, pre_sc) = (post_tc.as_standard_layout(), pre_sc.as_standard_layout());
    let inp = AggrIn {
        s_n,
        actors: a1.as_slice().unwrap(),
        dones: d1.as_slice().unwrap(),
        learner_mask: lm.as_slice().unwrap(),
        gates: g1.as_slice().unwrap(),
        pre_tc: pre_tc.as_slice().unwrap(),
        post_tc: post_tc.as_slice().unwrap(),
        pre_btc: btc1.as_slice().unwrap(),
        pre_sc: pre_sc.as_slice().unwrap(),
        pre_street: st1.as_slice().unwrap(),
        c,
        reward_norm,
    };
    let per_env: Vec<Option<AggrStep>> = py.detach(|| {
        (0..n)
            .into_par_iter()
            .with_min_len(APPLY_MIN_LEN)
            .map(|i| inp.step(i))
            .collect()
    });
    let t = tally(per_env.iter());
    let col = |f: fn(&Option<AggrStep>) -> f64| -> Array1<f64> { per_env.iter().map(f).collect() };
    let d = PyDict::new(py);
    d.set_item(
        "valid",
        per_env
            .iter()
            .map(Option::is_some)
            .collect::<Array1<bool>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "cost_increment",
        col(|s| s.map_or(0.0, |s| s.cost_inc)).into_pyarray(py),
    )?;
    d.set_item(
        "pot_pre_bb",
        col(|s| s.map_or(0.0, |s| s.pot_pre_bb)).into_pyarray(py),
    )?;
    d.set_item(
        "street_pre",
        per_env
            .iter()
            .map(|s| s.map_or(-1, |s| s.street))
            .collect::<Array1<i8>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "bonus_bb",
        col(|s| s.map_or(0.0, |s| s.bonus_bb)).into_pyarray(py),
    )?;
    d.set_item("total_bonus_bb", t.0)?;
    d.set_item("total_steps", t.1)?;
    d.set_item("bonus_steps", t.2)?;
    d.set_item(
        "steps_by_street",
        Array1::from_vec(t.3.to_vec()).into_pyarray(py),
    )?;
    d.set_item(
        "bonus_steps_by_street",
        Array1::from_vec(t.4.to_vec()).into_pyarray(py),
    )?;
    Ok(d)
}
