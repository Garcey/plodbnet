//! Compact rollout-observation rows: exact-0/1 flag columns as bits, the rest
//! verbatim f32 (python/plo5bp/compact_obs.py).

use super::*;

/// Bit pattern of 1.0f32 — the only non-zero value a compact-storage flag
/// column may hold (see [`pack_obs_rows`]).
pub(super) const F32_ONE_BITS: u32 = 0x3F80_0000;

/// One row of [`pack_obs_rows`]: the flag columns become MSB-first bits
/// (numpy `packbits` order: flag i -> byte i/8, bit 7 - i%8), the real
/// columns are copied verbatim. A flag must be exactly +0.0 or 1.0 by BIT
/// PATTERN (so -0.0 and NaN are rejected too); otherwise returns the first
/// offending (column, value).
pub(super) fn pack_obs_row(
    row: &[f32],
    flag_cols: &[usize],
    real_cols: &[usize],
    out_bits: &mut [u8],
    out_real: &mut [f32],
) -> Result<(), (usize, f32)> {
    out_bits.fill(0);
    for (i, &c) in flag_cols.iter().enumerate() {
        let v = row[c];
        match v.to_bits() {
            0 => {}
            F32_ONE_BITS => out_bits[i >> 3] |= 0x80u8 >> (i & 7),
            _ => return Err((c, v)),
        }
    }
    for (o, &c) in out_real.iter_mut().zip(real_cols.iter()) {
        *o = row[c];
    }
    Ok(())
}

/// Maximal runs of consecutive columns in a column list: (first column,
/// position of that column in the list, run length).
pub(super) fn column_runs(cols: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    for (i, &c) in cols.iter().enumerate() {
        match runs.last_mut() {
            Some((c0, _, len)) if *c0 + *len == c => *len += 1,
            _ => runs.push((c, i, 1)),
        }
    }
    runs
}

/// Four consecutive flag values: their bits MSB-first as a nibble (the first
/// value in the top bit), and whether all four are exactly +0.0 / 1.0 by bit
/// pattern. SSE2 -- part of every x86-64 CPU, so no runtime detection
/// (PERF-037).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn flag_nibble(x: &[f32]) -> (u64, bool) {
    use std::arch::x86_64::*;
    /// Lane i of a movemask is bit i; the stream wants lane 0 first (highest).
    const REV4: [u64; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
    assert!(x.len() >= 4);
    // SAFETY: SSE2 is part of the x86-64 baseline (every CPU this target runs
    // on has it), and the load reads four in-bounds f32s (checked above).
    let (ones, valid) = unsafe {
        let v = _mm_loadu_si128(x.as_ptr().cast());
        let is_one = _mm_cmpeq_epi32(v, _mm_set1_epi32(F32_ONE_BITS as i32));
        let is_zero = _mm_cmpeq_epi32(v, _mm_setzero_si128());
        (
            _mm_movemask_ps(_mm_castsi128_ps(is_one)) as usize,
            _mm_movemask_ps(_mm_castsi128_ps(_mm_or_si128(is_one, is_zero))),
        )
    };
    (REV4[ones], valid == 0xF)
}

/// [`pack_obs_row`] over precomputed column runs -- the same bytes, several
/// times faster: each run of consecutive flag columns streams its values
/// through a branch-free 64-bit accumulator (bit i of the flag stream lands
/// at byte i >> 3, bit 7 - (i & 7): numpy `packbits` order), four at a time
/// on x86-64 ([`flag_nibble`]), validity is folded into one flag, and real
/// columns are copied run by run. Any non-0/1 flag falls back to
/// [`pack_obs_row`], which reports the FIRST offending column exactly as
/// before.
pub(super) fn pack_obs_row_runs(
    row: &[f32],
    flag_runs: &[(usize, usize, usize)],
    real_runs: &[(usize, usize, usize)],
    flag_cols: &[usize],
    real_cols: &[usize],
    out_bits: &mut [u8],
    out_real: &mut [f32],
) -> Result<(), (usize, f32)> {
    let mut acc: u64 = 0;
    let mut nacc: u32 = 0;
    let mut byte = 0usize;
    let mut ok = true;
    for &(c0, _, len) in flag_runs {
        let run = &row[c0..c0 + len];
        let mut i = 0;
        while i < len {
            // Four at a time while four remain and the accumulator has room.
            #[cfg(target_arch = "x86_64")]
            if len - i >= 4 && nacc <= 60 {
                let (nibble, valid) = flag_nibble(&run[i..]);
                ok &= valid;
                acc = (acc << 4) | nibble;
                nacc += 4;
                i += 4;
                if nacc == 64 {
                    out_bits[byte..byte + 8].copy_from_slice(&acc.to_be_bytes());
                    byte += 8;
                    acc = 0;
                    nacc = 0;
                }
                continue;
            }
            let b = run[i].to_bits();
            ok &= (b == 0) | (b == F32_ONE_BITS);
            acc = (acc << 1) | u64::from(b == F32_ONE_BITS);
            nacc += 1;
            i += 1;
            if nacc == 64 {
                out_bits[byte..byte + 8].copy_from_slice(&acc.to_be_bytes());
                byte += 8;
                acc = 0;
                nacc = 0;
            }
        }
    }
    if nacc > 0 {
        let tail = (acc << (64 - nacc)).to_be_bytes();
        let nb = (nacc as usize).div_ceil(8);
        out_bits[byte..byte + nb].copy_from_slice(&tail[..nb]);
    }
    if !ok {
        return pack_obs_row(row, flag_cols, real_cols, out_bits, out_real);
    }
    for &(c0, i0, len) in real_runs {
        out_real[i0..i0 + len].copy_from_slice(&row[c0..c0 + len]);
    }
    Ok(())
}

/// Compact storage for rollout observations (python/plo5bp/compact_obs.py).
/// Packs rows `rows` of the dense (N, D) f32 observation matrix `obs` into the
/// caller's buffers at rows `out_offset .. out_offset + len(rows)`: the 0/1
/// `flag_cols` bit-packed into `out_bits` (ceil(F/8) bytes per row, numpy
/// `packbits` order) and the `real_cols` copied verbatim into `out_real`.
/// Storage only — unpacking reproduces every value bit-exactly. A flag column
/// holding anything but exactly 0.0 / 1.0 is a ValueError, so an encoder
/// change can never silently corrupt stored training rows.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn pack_obs_rows(
    obs: PyReadonlyArray2<'_, f32>,
    rows: PyReadonlyArray1<'_, i64>,
    flag_cols: PyReadonlyArray1<'_, i64>,
    real_cols: PyReadonlyArray1<'_, i64>,
    mut out_bits: PyReadwriteArray2<'_, u8>,
    mut out_real: PyReadwriteArray2<'_, f32>,
    out_offset: usize,
) -> PyResult<()> {
    // as_slice() also accepts FORTRAN-contiguous arrays, whose memory is
    // column-major: require row-major explicitly (inputs and outputs).
    if !obs.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "pack_obs_rows: obs must be C-contiguous",
        ));
    }
    let (n, d) = (obs.shape()[0], obs.shape()[1]);
    let obs_s = obs.as_slice()?;
    fn checked(v: &[i64], bound: usize, what: &str) -> PyResult<Vec<usize>> {
        v.iter()
            .map(|&x| {
                if x < 0 || (x as usize) >= bound {
                    Err(PyValueError::new_err(format!(
                        "pack_obs_rows: {what} index {x} out of range [0, {bound})"
                    )))
                } else {
                    Ok(x as usize)
                }
            })
            .collect()
    }
    let rows_v = checked(rows.as_slice()?, n, "row")?;
    let flags_v = checked(flag_cols.as_slice()?, d, "flag column")?;
    let reals_v = checked(real_cols.as_slice()?, d, "real column")?;
    let (nb, nr, k) = (flags_v.len().div_ceil(8), reals_v.len(), rows_v.len());
    if nb == 0 || nr == 0 {
        return Err(PyValueError::new_err(
            "pack_obs_rows: the layout needs at least one flag and one real column",
        ));
    }
    let (bits_shape, real_shape) = (out_bits.shape().to_vec(), out_real.shape().to_vec());
    if bits_shape[1] != nb || real_shape[1] != nr {
        return Err(PyValueError::new_err(format!(
            "pack_obs_rows: out_bits width {} / out_real width {} != layout {nb} / {nr}",
            bits_shape[1], real_shape[1]
        )));
    }
    if out_offset + k > bits_shape[0] || out_offset + k > real_shape[0] {
        return Err(PyValueError::new_err(format!(
            "pack_obs_rows: rows {out_offset}..{} overflow the output buffers ({} / {} rows)",
            out_offset + k,
            bits_shape[0],
            real_shape[0]
        )));
    }
    if !out_bits.is_c_contiguous() || !out_real.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "pack_obs_rows: out_bits and out_real must be C-contiguous",
        ));
    }
    if k == 0 {
        return Ok(());
    }
    let bits_s = out_bits
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("pack_obs_rows: out_bits must be C-contiguous"))?;
    let real_s = out_real
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("pack_obs_rows: out_real must be C-contiguous"))?;
    let bits_dst = &mut bits_s[out_offset * nb..(out_offset + k) * nb];
    let real_dst = &mut real_s[out_offset * nr..(out_offset + k) * nr];
    let (flag_runs, real_runs) = (column_runs(&flags_v), column_runs(&reals_v));
    bits_dst
        .par_chunks_mut(nb)
        .zip(real_dst.par_chunks_mut(nr))
        .zip(rows_v.par_iter())
        .with_min_len(256)
        .try_for_each(|((b, r), &row)| {
            pack_obs_row_runs(
                &obs_s[row * d..(row + 1) * d],
                &flag_runs,
                &real_runs,
                &flags_v,
                &reals_v,
                b,
                r,
            )
            .map_err(|(c, v)| (row, c, v))
        })
        .map_err(|(row, c, v)| {
            PyValueError::new_err(format!(
                "pack_obs_rows: obs row {row} column {c} holds {v:?}, not a 0/1 flag -- \
                 the compact layout (python/plo5bp/compact_obs.py) no longer matches the encoder"
            ))
        })
}

/// One row of [`unpack_obs_rows`]: the exact inverse of [`pack_obs_row`].
pub(super) fn unpack_obs_row(
    bits: &[u8],
    real: &[f32],
    flag_cols: &[usize],
    real_cols: &[usize],
    out: &mut [f32],
) {
    for (i, &c) in flag_cols.iter().enumerate() {
        out[c] = ((bits[i >> 3] >> (7 - (i & 7))) & 1) as f32;
    }
    for (&v, &c) in real.iter().zip(real_cols.iter()) {
        out[c] = v;
    }
}

/// Inverse of [`pack_obs_rows`] for CPU training (compact_obs.unpack): writes
/// the dense (k, D) f32 rows of `bits` / `real` into `out`. `flag_cols` and
/// `real_cols` must partition 0..D (every output column written exactly
/// once), so `out` may start uninitialized. Bit-exact: flags come back as
/// 0.0 / 1.0, real columns are copied.
#[pyfunction]
pub fn unpack_obs_rows(
    bits: PyReadonlyArray2<'_, u8>,
    real: PyReadonlyArray2<'_, f32>,
    flag_cols: PyReadonlyArray1<'_, i64>,
    real_cols: PyReadonlyArray1<'_, i64>,
    mut out: PyReadwriteArray2<'_, f32>,
) -> PyResult<()> {
    if !bits.is_c_contiguous() || !real.is_c_contiguous() || !out.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "unpack_obs_rows: bits, real and out must be C-contiguous",
        ));
    }
    let (k, d) = (out.shape()[0], out.shape()[1]);
    let mut seen = vec![false; d];
    let mut cols = |v: &[i64]| -> PyResult<Vec<usize>> {
        v.iter()
            .map(|&x| {
                if x < 0 || (x as usize) >= d || seen[x as usize] {
                    return Err(PyValueError::new_err(format!(
                        "unpack_obs_rows: column {x} out of range or listed twice"
                    )));
                }
                seen[x as usize] = true;
                Ok(x as usize)
            })
            .collect()
    };
    let flags_v = cols(flag_cols.as_slice()?)?;
    let reals_v = cols(real_cols.as_slice()?)?;
    if flags_v.len() + reals_v.len() != d {
        return Err(PyValueError::new_err(
            "unpack_obs_rows: flag_cols + real_cols must cover every output column",
        ));
    }
    let (nb, nr) = (flags_v.len().div_ceil(8), reals_v.len());
    if nb == 0 || nr == 0 {
        return Err(PyValueError::new_err(
            "unpack_obs_rows: the layout needs at least one flag and one real column",
        ));
    }
    if bits.shape() != [k, nb] || real.shape() != [k, nr] {
        return Err(PyValueError::new_err(format!(
            "unpack_obs_rows: bits {:?} / real {:?} do not match out ({k}, {d}) -> ({k}, {nb}) / ({k}, {nr})",
            bits.shape(),
            real.shape()
        )));
    }
    if k == 0 {
        return Ok(());
    }
    let (bits_s, real_s) = (bits.as_slice()?, real.as_slice()?);
    out.as_slice_mut()?
        .par_chunks_mut(d)
        .zip(bits_s.par_chunks(nb))
        .zip(real_s.par_chunks(nr))
        .for_each(|((o, b), r)| unpack_obs_row(b, r, &flags_v, &reals_v, o));
    Ok(())
}

#[cfg(test)]
mod pack_obs_tests {
    use super::{column_runs, pack_obs_row, pack_obs_row_runs, unpack_obs_row};

    /// The run-based packer writes exactly the bytes (and reports exactly the
    /// errors) of the reference packer, whatever the column layout.
    #[test]
    fn run_packer_matches_the_reference_packer() {
        let mut next = crate::test_util::xorshift(0x2545_F491_4F6C_DD1D);
        for case in 0..3_000 {
            let d = 1 + next(300) as usize;
            // Random partition of 0..d into flag / real columns, in a random
            // order half of the time (runs are only a speed-up).
            let mut flags: Vec<usize> = Vec::new();
            let mut reals: Vec<usize> = Vec::new();
            // Every third case: long runs (blocks of 1..40 columns) like the
            // real layouts', which the four-at-a-time path packs.
            let mut block = (0usize, false);
            for c in 0..d {
                let flag = if case % 3 == 0 {
                    if block.0 == 0 {
                        block = (1 + next(40) as usize, next(4) != 0);
                    }
                    block.0 -= 1;
                    block.1
                } else {
                    next(3) != 0
                };
                if flag {
                    flags.push(c)
                } else {
                    reals.push(c)
                }
            }
            if case % 2 == 1 {
                for i in (1..flags.len()).rev() {
                    let j = next(i as u64 + 1) as usize;
                    flags.swap(i, j);
                }
            }
            let mut row: Vec<f32> = (0..d)
                .map(|_| if next(2) == 0 { 0.0 } else { 1.0 })
                .collect();
            for &c in &reals {
                row[c] = (next(2000) as f32 - 1000.0) / 7.0;
            }
            if case % 5 == 0 && !flags.is_empty() {
                let bad = [0.5f32, -0.0, 2.0, f32::NAN, -1.0][next(5) as usize];
                row[flags[next(flags.len() as u64) as usize]] = bad;
            }
            let nb = flags.len().div_ceil(8);
            let (mut b1, mut r1) = (vec![0xAAu8; nb], vec![f32::NAN; reals.len()]);
            let (mut b2, mut r2) = (vec![0x55u8; nb], vec![f32::NAN; reals.len()]);
            let want = pack_obs_row(&row, &flags, &reals, &mut b1, &mut r1);
            let got = pack_obs_row_runs(
                &row,
                &column_runs(&flags),
                &column_runs(&reals),
                &flags,
                &reals,
                &mut b2,
                &mut r2,
            );
            match (want, got) {
                (Ok(()), Ok(())) => {
                    assert_eq!(b1, b2, "case {case}");
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&r1), bits(&r2), "case {case}");
                }
                (Err(a), Err(b)) => assert_eq!(a.0, b.0, "case {case}: first bad column"),
                (a, b) => panic!("case {case}: {a:?} vs {b:?}"),
            }
        }
    }

    #[test]
    fn unpack_is_the_exact_inverse_of_pack() {
        let row: Vec<f32> = vec![
            1.0, 0.0, 0.0, 1.0, 3.25, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, -7.5, 1.0,
        ];
        let flags = [0usize, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12];
        let reals = [4usize, 11];
        let (mut bits, mut vals) = ([0u8; 2], [0f32; 2]);
        pack_obs_row(&row, &flags, &reals, &mut bits, &mut vals).unwrap();
        let mut back = vec![f32::NAN; row.len()];
        unpack_obs_row(&bits, &vals, &flags, &reals, &mut back);
        let bits_of = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits_of(&back), bits_of(&row));
    }

    #[test]
    fn packs_flags_msb_first_and_copies_reals_verbatim() {
        // 10 flags (2 bytes, 6 pad bits) + 2 reals, interleaved in the row.
        let row: Vec<f32> = vec![
            1.0, 0.0, 0.0, 1.0, 3.25, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, -7.5, 1.0,
        ];
        let flags = [0usize, 1, 2, 3, 5, 6, 7, 8, 9, 10];
        let reals = [4usize, 11];
        let mut bits = [0xFFu8; 2];
        let mut out = [0f32; 2];
        pack_obs_row(&row, &flags, &reals, &mut bits, &mut out).unwrap();
        // flags: 1,0,0,1,1,1,0,0 | 0,1 -> 0b1001_1100, 0b0100_0000 (pad bits zeroed)
        assert_eq!(bits, [0b1001_1100, 0b0100_0000]);
        assert_eq!(out[0].to_bits(), 3.25f32.to_bits());
        assert_eq!(out[1].to_bits(), (-7.5f32).to_bits());
    }

    #[test]
    fn rejects_anything_but_exact_zero_or_one_in_a_flag_column() {
        let flags = [0usize, 1];
        let reals = [2usize];
        for bad in [0.5f32, -0.0, 2.0, f32::NAN, -1.0] {
            let row = [1.0f32, bad, 9.0];
            let (mut bits, mut out) = ([0u8; 1], [0f32; 1]);
            let err = pack_obs_row(&row, &flags, &reals, &mut bits, &mut out).unwrap_err();
            assert_eq!(err.0, 1, "value {bad:?} must be rejected at column 1");
        }
    }
}
