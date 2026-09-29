//! Rust tests of the rollout kernels (TEST-023): the flush against an
//! independent reference on small hand-built inputs (GAE and VRPO, the
//! retroactive bonus, variable lengths, masked seats, float16 storage), every
//! error path leaving the outputs untouched, the parallel record / gather
//! paths against the sequential ones, and the float16 conversion over every
//! float32 bit pattern (ENG-028).

use super::*;

use crate::test_util::Mix;

/// A random flush problem: `envs` envs x `s` seats, trajectories up to `cap`.
struct Case {
    s: usize,
    cap: usize,
    term: Vec<i64>,
    lengths: Vec<i32>,
    mask: Vec<bool>,
    won: Vec<f32>,
    gt: Vec<bool>,
    eq: Vec<bool>,
    obs_idx: Vec<i64>,
    gate: Vec<i8>,
    chips: Vec<i64>,
    sizing: Vec<i64>,
    anchor: Vec<i8>,
    f: [Vec<f32>; 8], // u, log_p, gate_lp, anchor_lp, value, costs, pots, (unused)
    streets: Vec<i8>,
    q: Vec<f32>,
    vpi: Vec<f32>,
    bits: Vec<u8>,
    real: Vec<f32>,
    gm: Vec<bool>,
    holes: Vec<u8>,
    nb: usize,
    nr: usize,
    oh: usize,
}

fn case(seed: u64) -> Case {
    let mut r = Mix(seed);
    let (envs, s, cap) = (6usize, 3usize, 7usize);
    let m = envs * s * cap;
    let (nb, nr, oh, pool_rows) = (2usize, 3usize, 4usize, 40usize);
    let term: Vec<i64> = vec![4, 0, 5, 2];
    let t = term.len();
    let lengths: Vec<i32> = (0..t * s).map(|_| r.below(cap as u64 + 1) as i32).collect();
    let mask: Vec<bool> = (0..t * s).map(|_| r.below(4) != 0).collect();
    Case {
        s,
        cap,
        won: (0..t * s).map(|_| r.f32()).collect(),
        gt: (0..t * s).map(|_| r.below(3) == 0).collect(),
        eq: (0..t * s).map(|_| r.below(3) == 0).collect(),
        term,
        lengths,
        mask,
        obs_idx: (0..m).map(|_| r.below(pool_rows as u64) as i64).collect(),
        gate: (0..m).map(|_| r.below(4) as i8).collect(),
        chips: (0..m).map(|_| r.below(90_000) as i64).collect(),
        sizing: (0..4 * m).map(|_| r.below(1000) as i64 - 3).collect(),
        anchor: (0..m).map(|_| r.below(11) as i8).collect(),
        f: std::array::from_fn(|_| (0..m).map(|_| r.f32()).collect()),
        streets: (0..m).map(|_| r.below(5) as i8).collect(),
        q: (0..m).map(|_| r.f32()).collect(),
        vpi: (0..m).map(|_| r.f32()).collect(),
        bits: (0..pool_rows * nb).map(|_| r.below(256) as u8).collect(),
        real: (0..pool_rows * nr).map(|_| r.f32() * 1e3).collect(),
        gm: (0..pool_rows * 3).map(|_| r.below(2) == 1).collect(),
        holes: (0..envs * s * oh).map(|_| r.below(52) as u8).collect(),
        nb,
        nr,
        oh,
    }
}

impl Case {
    fn input(&self, vrpo: bool, gamma: f32, lam: f32, retro_c: f32) -> FlushIn<'_> {
        FlushIn {
            term: &self.term,
            s_n: self.s,
            lengths: &self.lengths,
            flush_mask: &self.mask,
            won_bb: &self.won,
            share_gt: &self.gt,
            share_eq: &self.eq,
            traj: TrajIn {
                obs_idx: &self.obs_idx,
                gate: &self.gate,
                chips: &self.chips,
                sizing: &self.sizing,
                anchor: &self.anchor,
                u: &self.f[0],
                log_p: &self.f[1],
                gate_lp: &self.f[2],
                anchor_lp: &self.f[3],
                value: &self.f[4],
                costs: &self.f[5],
                pots: &self.f[6],
                streets: &self.streets,
                q_vpi: vrpo.then_some((&self.q[..], &self.vpi[..])),
            },
            traj_cap: self.cap,
            pool: PoolIn {
                bits: &self.bits,
                nb: self.nb,
                real: &self.real,
                nr: self.nr,
                gm: &self.gm,
                gm_w: 3,
            },
            holes_rot: &self.holes,
            hole_rows: self.holes.len() / self.oh,
            oh_w: self.oh,
            gamma,
            lam,
            retro_c,
        }
    }

    fn rows(&self) -> usize {
        (0..self.lengths.len())
            .map(|k| {
                if self.mask[k] {
                    self.lengths[k].max(0) as usize
                } else {
                    0
                }
            })
            .sum()
    }
}

/// Owned output slabs (sentinel-filled so untouched rows show).
#[derive(Clone, PartialEq, Debug)]
struct Slabs {
    bits: Vec<u8>,
    real: Vec<f32>,
    real16: Vec<u16>,
    gm: Vec<bool>,
    ga: Vec<i64>,
    rc: Vec<i64>,
    sz: Vec<i64>,
    an: Vec<i64>,
    ru: Vec<f32>,
    oh: Vec<u8>,
    lp: Vec<f32>,
    glp: Vec<f32>,
    alp: Vec<f32>,
    v: Vec<f32>,
    ret: Vec<f32>,
    adv: Vec<f32>,
    last: Vec<bool>,
}

impl Slabs {
    fn new(rows: usize, c: &Case) -> Self {
        Slabs {
            bits: vec![0xAA; rows * c.nb],
            real: vec![-7.0; rows * c.nr],
            real16: vec![0xBEEF; rows * c.nr],
            gm: vec![true; rows * 3],
            ga: vec![-9; rows],
            rc: vec![-9; rows],
            sz: vec![-9; rows * 4],
            an: vec![-9; rows],
            ru: vec![-9.0; rows],
            oh: vec![0xEE; rows * c.oh],
            lp: vec![-9.0; rows],
            glp: vec![-9.0; rows],
            alp: vec![-9.0; rows],
            v: vec![-9.0; rows],
            ret: vec![-9.0; rows],
            adv: vec![-9.0; rows],
            last: vec![true; rows],
        }
    }

    fn out(&mut self, half: bool) -> FlushOut<'_> {
        FlushOut {
            bits: &mut self.bits,
            real: if half {
                RealOut::F16(&mut self.real16)
            } else {
                RealOut::F32(&mut self.real)
            },
            gm: &mut self.gm,
            ga: &mut self.ga,
            rc: &mut self.rc,
            sz: &mut self.sz,
            an: &mut self.an,
            ru: &mut self.ru,
            oh: &mut self.oh,
            lp: &mut self.lp,
            glp: &mut self.glp,
            alp: &mut self.alp,
            v: &mut self.v,
            ret: &mut self.ret,
            adv: &mut self.adv,
            last: &mut self.last,
        }
    }
}

/// The flush written out naively, pair by pair, straight from the rollout's
/// numpy formulas (independent of `flush_inner`'s block splitting).
fn reference(
    c: &Case,
    vrpo: bool,
    gamma: f32,
    lam: f32,
    retro_c: f32,
) -> (Slabs, u64, [u64; 3], f64) {
    let mut o = Slabs::new(c.rows(), c);
    let (mut steps, mut streets, mut total) = (0u64, [0u64; 3], 0f64);
    let mut r = 0usize;
    for k in 0..c.lengths.len() {
        if !c.mask[k] {
            continue;
        }
        let len = c.lengths[k].max(0) as usize;
        let es = c.term[k / c.s] as usize * c.s + k % c.s;
        let base = es * c.cap;
        let mut cost = vec![0f32; len];
        for l in 0..len {
            let g = c.gate[base + l];
            let raw = c.f[5][base + l];
            let q = (c.gt[k] && g == 2) || (c.eq[k] && (g == 2 || (g == 1 && raw < 0.0)));
            if q {
                steps += 1;
                let st = c.streets[base + l];
                if (1..=3).contains(&st) {
                    streets[(st - 1) as usize] += 1;
                }
            }
            cost[l] = if retro_c != 0.0 {
                let b = (if q { 1.0f32 } else { 0.0 }) * (retro_c * c.f[6][base + l]);
                total += b as f64;
                raw + b
            } else {
                raw
            };
        }
        let (mut gae, mut es_tr) = (0f32, 0f32);
        let mut adv = vec![0f32; len];
        let mut ret = vec![0f32; len];
        for l in (0..len).rev() {
            let last = l + 1 == len;
            let reward = cost[l] + if last { c.won[k] } else { 0.0 };
            let v = c.f[4][base + l];
            let nv = if last { 0.0 } else { c.f[4][base + l + 1] };
            gae = ((reward + gamma * nv) - v) + (gamma * lam) * gae;
            ret[l] = gae + v;
            adv[l] = if vrpo {
                let nvpi = if last { 0.0 } else { c.vpi[base + l + 1] };
                es_tr = ((reward + gamma * nvpi) - c.q[base + l]) + (gamma * lam) * es_tr;
                (c.q[base + l] - c.vpi[base + l]) + es_tr
            } else {
                gae
            };
        }
        for l in 0..len {
            let p = c.obs_idx[base + l] as usize;
            o.bits[r * c.nb..(r + 1) * c.nb].copy_from_slice(&c.bits[p * c.nb..(p + 1) * c.nb]);
            o.real[r * c.nr..(r + 1) * c.nr].copy_from_slice(&c.real[p * c.nr..(p + 1) * c.nr]);
            for j in 0..c.nr {
                o.real16[r * c.nr + j] = f32_to_f16_bits(c.real[p * c.nr + j]);
            }
            o.gm[r * 3..r * 3 + 3].copy_from_slice(&c.gm[p * 3..p * 3 + 3]);
            o.ga[r] = c.gate[base + l] as i64;
            o.rc[r] = c.chips[base + l];
            o.sz[r * 4..r * 4 + 4].copy_from_slice(&c.sizing[(base + l) * 4..(base + l) * 4 + 4]);
            o.an[r] = c.anchor[base + l] as i64;
            o.ru[r] = c.f[0][base + l];
            o.oh[r * c.oh..(r + 1) * c.oh].copy_from_slice(&c.holes[es * c.oh..(es + 1) * c.oh]);
            o.lp[r] = c.f[1][base + l];
            o.glp[r] = c.f[2][base + l];
            o.alp[r] = c.f[3][base + l];
            o.v[r] = c.f[4][base + l];
            o.ret[r] = ret[l];
            o.adv[r] = adv[l];
            o.last[r] = l + 1 == len;
            r += 1;
        }
    }
    (o, steps, streets, total)
}

fn bits32(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn flush_matches_the_reference_bit_for_bit() {
    for seed in 0..40u64 {
        let c = case(seed);
        for (vrpo, retro) in [(false, 0.0f32), (false, 0.35), (true, 0.0), (true, 0.2)] {
            let (want, steps, streets, total) = reference(&c, vrpo, 0.99, 0.95, retro);
            for half in [false, true] {
                let mut got = Slabs::new(c.rows(), &c);
                let st = flush_inner(&c.input(vrpo, 0.99, 0.95, retro), got.out(half)).unwrap();
                assert_eq!(st.rows, c.rows());
                assert_eq!(
                    (st.bonus_steps, st.by_street),
                    (steps, streets),
                    "seed {seed}"
                );
                assert_eq!(
                    st.bonus_total, total,
                    "seed {seed}: the bonus total's order is fixed"
                );
                if retro == 0.0 {
                    assert_eq!(st.bonus_total, 0.0);
                }
                // Every slab bit for bit (float slabs compared as bits).
                assert_eq!(got.bits, want.bits);
                if half {
                    assert_eq!(got.real16, want.real16);
                } else {
                    assert_eq!(bits32(&got.real), bits32(&want.real));
                }
                assert_eq!(
                    (&got.gm, &got.ga, &got.rc, &got.sz, &got.an),
                    (&want.gm, &want.ga, &want.rc, &want.sz, &want.an)
                );
                assert_eq!(got.oh, want.oh);
                assert_eq!(got.last, want.last);
                for (g, w) in [
                    (&got.ru, &want.ru),
                    (&got.lp, &want.lp),
                    (&got.glp, &want.glp),
                    (&got.alp, &want.alp),
                    (&got.v, &want.v),
                    (&got.ret, &want.ret),
                    (&got.adv, &want.adv),
                ] {
                    assert_eq!(bits32(g), bits32(w), "seed {seed} vrpo {vrpo}");
                }
            }
        }
    }
}

#[test]
fn a_rejected_flush_writes_nothing() {
    let c = case(7);
    let rows = c.rows();
    let fresh = Slabs::new(rows, &c);
    let check = |c: &Case, rows: usize, what: &str| {
        let mut got = Slabs::new(rows, c);
        let e = flush_inner(&c.input(true, 0.99, 0.95, 0.1), got.out(false)).unwrap_err();
        assert!(e.contains(what), "{e}");
        assert_eq!(
            got,
            Slabs::new(rows, c),
            "{what}: outputs must be untouched"
        );
    };
    // A length past the trajectory capacity (the last pair: every earlier one
    // would already have been written by a check-as-you-go loop).
    let mut bad = case(7);
    let last = bad.lengths.len() - 1;
    bad.mask[last] = true;
    bad.lengths[last] = bad.cap as i32 + 1;
    check(&bad, rows, "traj_cap");
    // A pool row that does not exist.
    let mut bad = case(7);
    let k = (0..bad.lengths.len())
        .rev()
        .find(|&k| bad.mask[k] && bad.lengths[k] > 0)
        .unwrap();
    let base = (bad.term[k / bad.s] as usize * bad.s + k % bad.s) * bad.cap;
    bad.obs_idx[base] = 999;
    check(&bad, rows, "pool index");
    // A finished env outside the table.
    let mut bad = case(7);
    bad.term[2] = 99;
    check(&bad, rows, "out of range");
    // Output slabs of the wrong size.
    let mut got = Slabs::new(rows + 1, &c);
    assert!(
        flush_inner(&c.input(false, 0.99, 0.95, 0.0), got.out(false))
            .unwrap_err()
            .contains("output")
    );
    assert_eq!(got, Slabs::new(rows + 1, &c));
    // A (T, S) array of the wrong length.
    let mut bad = case(7);
    bad.won.pop();
    check(&bad, rows, "won_bb");
    // The good case still flushes.
    let mut got = fresh.clone();
    assert!(flush_inner(&c.input(true, 0.99, 0.95, 0.1), got.out(false)).is_ok());
    assert_ne!(got, fresh);
}

#[test]
fn many_pairs_split_across_tasks_match_one_task() {
    // More (hand, seat) pairs than one FLUSH_BLOCK: the block splitting must
    // give each pair exactly its own rows.
    let mut c = case(11);
    let reps = 3 * FLUSH_BLOCK / (c.term.len() * c.s) + 1;
    let (term, lengths, mask, won, gt, eq) = (
        c.term.clone(),
        c.lengths.clone(),
        c.mask.clone(),
        c.won.clone(),
        c.gt.clone(),
        c.eq.clone(),
    );
    for _ in 1..reps {
        c.term.extend(&term);
        c.lengths.extend(&lengths);
        c.mask.extend(&mask);
        c.won.extend(&won);
        c.gt.extend(&gt);
        c.eq.extend(&eq);
    }
    let (want, steps, streets, total) = reference(&c, true, 0.97, 0.9, 0.25);
    let mut got = Slabs::new(c.rows(), &c);
    let st = flush_inner(&c.input(true, 0.97, 0.9, 0.25), got.out(false)).unwrap();
    assert_eq!(
        (st.bonus_steps, st.by_street, st.bonus_total),
        (steps, streets, total)
    );
    assert_eq!(bits32(&got.adv), bits32(&want.adv));
    assert_eq!(bits32(&got.ret), bits32(&want.ret));
    assert_eq!(got.bits, want.bits);
    assert_eq!(got.oh, want.oh);
}

fn rec_case(
    n_envs: usize,
    rows: usize,
    m: usize,
    ascending: bool,
    seed: u64,
) -> (Vec<i64>, Vec<i64>) {
    let mut r = Mix(seed);
    let mut slot: Vec<i64> = if ascending {
        let mut s: Vec<i64> = (0..m as i64).collect();
        for i in (1..s.len()).rev() {
            s.swap(i, r.below(i as u64 + 1) as usize);
        }
        let mut s = s[..rows].to_vec();
        s.sort_unstable();
        s
    } else {
        (0..rows).map(|_| r.below(m as u64) as i64).collect()
    };
    if !ascending && rows > 1 {
        slot.swap(0, 1);
    }
    let lidx = (0..rows).map(|_| r.below(n_envs as u64) as i64).collect();
    (slot, lidx)
}

#[test]
fn record_parallel_matches_sequential_and_errors_write_nothing() {
    let (n, m) = (3000usize, 40_000usize);
    let mut r = Mix(5);
    let gates: Vec<u8> = (0..n).map(|_| r.below(4) as u8).collect();
    let chips: Vec<u64> = (0..n).map(|_| r.below(1 << 40)).collect();
    let sizing: Vec<i64> = (0..4 * n).map(|_| r.below(999) as i64).collect();
    let anchors: Vec<i64> = (0..n).map(|_| r.below(11) as i64).collect();
    let fl: Vec<Vec<f32>> = (0..7).map(|_| (0..n).map(|_| r.f32()).collect()).collect();
    let inp = RecIn {
        gates: &gates,
        chips: &chips,
        sizing: &sizing,
        anchors: &anchors,
        f32s: fl.iter().map(|v| &v[..]).collect(),
    };
    let run = |slot: &[i64], lidx: &[i64]| -> (Result<(), String>, Vec<Vec<u64>>) {
        let (mut o, mut g, mut c, mut s, mut a) = (
            vec![-1i64; m],
            vec![-1i8; m],
            vec![-1i64; m],
            vec![-1i64; 4 * m],
            vec![-1i8; m],
        );
        let mut f: Vec<Vec<f32>> = (0..7).map(|_| vec![-1.0; m]).collect();
        let out = RecOut {
            obs_idx: &mut o,
            gate: &mut g,
            chips: &mut c,
            sizing: &mut s,
            anchor: &mut a,
            f32s: f.iter_mut().map(|v| &mut v[..]).collect(),
        };
        let res = record_inner(slot, lidx, 77, &inp, out);
        let mut all = vec![
            o.iter().map(|&x| x as u64).collect::<Vec<u64>>(),
            g.iter().map(|&x| x as u64).collect(),
            c.iter().map(|&x| x as u64).collect(),
            s.iter().map(|&x| x as u64).collect(),
            a.iter().map(|&x| x as u64).collect(),
        ];
        all.extend(
            f.iter()
                .map(|v| v.iter().map(|x| x.to_bits() as u64).collect()),
        );
        (res, all)
    };
    // Parallel path (ascending, >= REC_PAR_MIN_ROWS rows) == the sequential loop.
    let (slot, lidx) = rec_case(n, REC_PAR_MIN_ROWS + 3 * REC_MIN_LEN + 17, m, true, 9);
    let (res, par) = run(&slot, &lidx);
    res.unwrap();
    let mut want = run(&[], &[]).1;
    for (i, (&sl, &e)) in slot.iter().zip(&lidx).enumerate() {
        let (sl, e) = (sl as usize, e as usize);
        want[0][sl] = 77 + i as u64;
        want[1][sl] = gates[e] as i8 as u64;
        want[2][sl] = if gates[e] == 2 { chips[e] } else { 0 };
        for j in 0..4 {
            want[3][sl * 4 + j] = sizing[e * 4 + j] as u64;
        }
        want[4][sl] = anchors[e] as i8 as u64;
        for (t, v) in fl.iter().enumerate() {
            want[5 + t][sl] = v[e].to_bits() as u64;
        }
    }
    assert!(
        par == want,
        "parallel record differs from the sequential one"
    );
    // Unordered slots take the sequential loop (later rows win on repeats).
    let (slot, lidx) = rec_case(n, 500, m, false, 3);
    run(&slot, &lidx).0.unwrap();
    // A bad row anywhere: nothing written.
    let (mut slot, lidx) = rec_case(n, REC_PAR_MIN_ROWS + 5, m, true, 4);
    let blank = run(&[], &[]).1;
    *slot.last_mut().unwrap() = m as i64;
    let (res, got) = run(&slot, &lidx);
    assert!(res.unwrap_err().contains("out of range"));
    assert!(got == blank);
}

#[test]
fn gather_parallel_matches_sequential_and_errors_write_nothing() {
    let mut r = Mix(8);
    let k = 3 * GATHER_BLOCK_ROWS + 99;
    let (w1, w2) = (1500usize, 7usize); // k * (w1 + w2) > GATHER_PAR_MIN_BYTES
    let src1: Vec<u8> = (0..(k + 50) * w1).map(|_| r.below(256) as u8).collect();
    let src2: Vec<u8> = (0..(k + 50) * w2).map(|_| r.below(256) as u8).collect();
    let rows: Vec<i64> = (0..k).map(|_| r.below(k as u64 + 50) as i64).collect();
    let start = 13;
    let mut d1 = vec![0u8; (k + start + 5) * w1];
    let mut d2 = vec![0u8; (k + start + 5) * w2];
    gather_inner(
        vec![
            GatherPair {
                src: &src1,
                w: w1,
                dst: &mut d1,
            },
            GatherPair {
                src: &src2,
                w: w2,
                dst: &mut d2,
            },
        ],
        start,
        Some(&rows),
    )
    .unwrap();
    for (i, &s) in rows.iter().enumerate() {
        let s = s as usize;
        assert_eq!(
            &d1[(start + i) * w1..(start + i + 1) * w1],
            &src1[s * w1..(s + 1) * w1]
        );
        assert_eq!(
            &d2[(start + i) * w2..(start + i + 1) * w2],
            &src2[s * w2..(s + 1) * w2]
        );
    }
    assert!(
        d1[..start * w1].iter().all(|&x| x == 0) && d1[(start + k) * w1..].iter().all(|&x| x == 0)
    );
    // rows = None copies the sources in order.
    let mut d3 = vec![0u8; (k + 50) * w1];
    gather_inner(
        vec![GatherPair {
            src: &src1,
            w: w1,
            dst: &mut d3,
        }],
        0,
        None,
    )
    .unwrap();
    assert_eq!(d3, src1);
    // A bad row in the second pair: neither destination changes.
    let mut bad_rows = rows.clone();
    bad_rows[k - 1] = -1;
    let (mut e1, mut e2) = (vec![0u8; d1.len()], vec![0u8; d2.len()]);
    let err = gather_inner(
        vec![
            GatherPair {
                src: &src1,
                w: w1,
                dst: &mut e1,
            },
            GatherPair {
                src: &src2,
                w: w2,
                dst: &mut e2,
            },
        ],
        start,
        Some(&bad_rows),
    )
    .unwrap_err();
    assert!(err.contains("out of range"));
    assert!(e1.iter().all(|&x| x == 0) && e2.iter().all(|&x| x == 0));
}

/// A plainly-correct f32 -> f16 conversion for finite inputs: scale to the
/// half-precision quantum of the input's binade and round to nearest even in
/// f64 (exact: f32 has 24 significant bits).
fn f16_reference(x: f32) -> u16 {
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = (x as f64).abs();
    if a >= 65520.0 {
        return sign | 0x7c00; // rounds to the next binade = overflow
    }
    // Binade exponent, clamped at the subnormal range (quantum 2^-24).
    let e = if a == 0.0 {
        -14
    } else {
        (a.log2().floor() as i32).max(-14)
    };
    let q = 2f64.powi(e - 10);
    let n = a / q; // exact
    let fl = n.floor();
    let rn = if n - fl > 0.5 || (n - fl == 0.5 && fl % 2.0 == 1.0) {
        fl + 1.0
    } else {
        fl
    };
    let v = rn * q;
    if v == 0.0 {
        return sign;
    }
    // Encode v (normal or subnormal) exactly.
    let ve = (v.log2().floor() as i32).max(-15);
    if ve < -14 {
        return sign | (v / 2f64.powi(-24)) as u16;
    }
    let man = (v / 2f64.powi(ve) - 1.0) * 1024.0;
    sign | (((ve + 15) as u16) << 10) | man as u16
}

#[test]
fn f16_conversion_edge_classes() {
    let cases: [(u32, u16); 12] = [
        (0x0000_0000, 0x0000), // +0
        (0x8000_0000, 0x8000), // -0
        (0x3F80_0000, 0x3C00), // 1.0
        (0x477F_E000, 0x7BFF), // 65504 = max half
        (0x477F_F000, 0x7C00), // 65520 ties to even -> inf
        (0x477F_EFFF, 0x7BFF), // just below the tie
        (0x3380_0000, 0x0001), // 2^-24 = min subnormal
        (0x3300_0000, 0x0000), // 2^-25 ties to even (0)
        (0x3300_0001, 0x0001), // just above -> min subnormal
        (0x7F80_0000, 0x7C00), // +inf
        (0xFF80_0000, 0xFC00), // -inf
        (0x3F80_1000, 0x3C00), // 1 + 2^-11 ties to even (1.0)
    ];
    for (bits, want) in cases {
        assert_eq!(f32_to_f16_bits(f32::from_bits(bits)), want, "{bits:#010x}");
    }
    // NaN stays NaN (quiet, same sign), signaling or not.
    for bits in [0x7FC0_0000u32, 0x7F80_0001, 0xFF80_2000, 0x7FFF_FFFF] {
        let h = f32_to_f16_bits(f32::from_bits(bits));
        assert_eq!(h & 0x7C00, 0x7C00);
        assert_ne!(h & 0x03FF, 0);
        assert_eq!(h & 0x0200, 0x0200, "quiet bit");
        assert_eq!(h & 0x8000, ((bits >> 16) & 0x8000) as u16);
    }
    // The software reference on a spread of every class.
    let mut x = 0u32;
    while x < u32::MAX - 65_537 {
        let f = f32::from_bits(x);
        if !f.is_nan() {
            assert_eq!(f32_to_f16_bits(f), f16_reference(f), "{x:#010x}");
        }
        x += 65_537;
    }
}

/// ENG-028: every one of the 2^32 float32 bit patterns against the x86 F16C
/// conversion (round to nearest even; NaN -> quiet NaN with the top payload
/// bits, which `f32_to_f16_bits` reproduces too). Runs in full in optimised
/// builds without debug assertions (`cargo test --release`, ~1-3 s); a
/// strided pass otherwise.
#[cfg(target_arch = "x86_64")]
#[test]
fn f16_conversion_is_exact() {
    if !std::arch::is_x86_feature_detected!("f16c") {
        eprintln!("f16c not available: the edge-class test covers the conversion");
        return;
    }
    #[target_feature(enable = "f16c")]
    unsafe fn hw4(x: [f32; 4]) -> [u16; 4] {
        use std::arch::x86_64::*;
        let v = _mm_loadu_ps(x.as_ptr());
        let h = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
        let mut out = [0u16; 8];
        _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, h);
        [out[0], out[1], out[2], out[3]]
    }
    let stride: u64 = if cfg!(debug_assertions) { 97 } else { 1 };
    let chunks: u64 = 1 << 16;
    let bad: Option<u32> = (0..chunks).into_par_iter().find_map_any(|hi| {
        let mut lo = 0u64;
        while lo < (1 << 16) {
            let base = (hi << 16) | lo;
            let xs: [u32; 4] =
                std::array::from_fn(|i| (base + i as u64 * stride).min(u32::MAX as u64) as u32);
            // SAFETY: f16c was detected above.
            let hw = unsafe { hw4(xs.map(f32::from_bits)) };
            for (x, h) in xs.iter().zip(hw) {
                if f32_to_f16_bits(f32::from_bits(*x)) != h {
                    return Some(*x);
                }
            }
            lo += 4 * stride;
        }
        None
    });
    assert_eq!(bad, None, "f32_to_f16_bits disagrees with F16C");
}
