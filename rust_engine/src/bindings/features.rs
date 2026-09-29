//! Standalone observation-feature kernels (straight/flush, cross-board,
//! draw flags, pair structure, board strength, payouts) and their pyfunctions.
//! The fused encoders call the same inner functions.

use super::*;

/// Current made-hand STRENGTH of PLO holdings on one board, batched
/// (2026-09-26, critic hand-strength inputs): row i = the best hand
/// `holes[i]` can make with exactly 2 of its cards and 3 of the first
/// `board_len[i]` cards of `board[i]`, as 7463 - (Cactus-Kev rank), i.e.
/// 1 (worst high card) ..= 7462 (royal flush). 0 when the board has fewer
/// than 3 cards or the row holds a 255 (empty) hole card. Rows in parallel.
#[pyfunction]
pub fn plo_board_strength_batch<'py>(
    py: Python<'py>,
    holes: PyReadonlyArray2<'_, u8>,
    board: PyReadonlyArray2<'_, u8>,
    board_len: PyReadonlyArray1<'_, u8>,
) -> PyResult<Bound<'py, PyArray1<u16>>> {
    let hv = holes.as_array();
    let bv = board.as_array();
    let lv = board_len.as_array();
    let n = hv.shape()[0];
    let hw = hv.shape()[1];
    if !batched_plo_hole_width(hw) || bv.shape() != [n, 5] || lv.len() != n {
        return Err(PyValueError::new_err(
            "holes must be (N, 4|5|6), board (N, 5), board_len (N,)",
        ));
    }
    if hv.iter().chain(bv.iter()).any(|&c| c != 255 && c >= 52) {
        return Err(PyValueError::new_err(
            "card index out of range (0..52 or 255)",
        ));
    }
    // Borrowed as-is when already standard-layout.
    let (hs, bs, ls) = (
        hv.as_standard_layout(),
        bv.as_standard_layout(),
        lv.as_standard_layout(),
    );
    let (hs, bs, ls) = (
        hs.as_slice().unwrap(),
        bs.as_slice().unwrap(),
        ls.as_slice().unwrap(),
    );
    let out: Vec<u16> = py.detach(|| {
        (0..n)
            .into_par_iter()
            .map(|i| {
                board_strength(
                    &hs[i * hw..(i + 1) * hw],
                    &bs[i * 5..i * 5 + 5],
                    ls[i] as usize,
                )
            })
            .collect()
    });
    Ok(Array1::from_vec(out).into_pyarray(py))
}

/// One row of [`plo_board_strength_batch`]: `7463 - CK rank` of the best
/// exactly-2-hole + 3-board hand on the first `len` board cards; 0 for a
/// board shorter than 3 (or longer than 5) or any empty (255) card.
pub(super) fn board_strength(hole: &[u8], board: &[u8], len: usize) -> u16 {
    if !(3..=5).contains(&len) || hole.iter().chain(&board[..len]).any(|&c| c >= 52) {
        return 0;
    }
    let mut hole_buf = [Card::from_index(0); crate::hand_eval::MAX_PLO_HOLE];
    let mut board_buf = [Card::from_index(0); 5];
    for (x, &c) in hole_buf.iter_mut().zip(hole) {
        *x = Card::from_index(c);
    }
    for (x, &c) in board_buf.iter_mut().zip(&board[..len]) {
        *x = Card::from_index(c);
    }
    let rank = crate::hand_eval::evaluate_plo_partial(&hole_buf[..hole.len()], &board_buf[..len]);
    (7463 - crate::hand_eval::ck_of_rank(rank)) as u16
}

#[cfg(test)]
mod board_strength_tests {
    use super::*;

    /// TEST-024: the strength is 7463 - CK rank of the best explicit
    /// 2-hole + 3-board hand (evaluate_5 over every combo), for hole widths
    /// 4..=6 and boards of 3..=5; empty cards and short boards give 0.
    #[test]
    fn strength_is_the_inverted_rank_of_the_best_holding() {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(24);
        let mut deck: Vec<u8> = (0..52).collect();
        for trial in 0..600 {
            deck.shuffle(&mut rng);
            let hw = 4 + trial % 3;
            let len = 3 + trial % 3;
            let (hole, board) = (&deck[..hw], &deck[hw..hw + 5]);
            let mut best = 0;
            for i in 0..hw {
                for j in (i + 1)..hw {
                    for x in 0..len {
                        for y in (x + 1)..len {
                            for z in (y + 1)..len {
                                let five = [hole[i], hole[j], board[x], board[y], board[z]]
                                    .map(Card::from_index);
                                best = best.max(crate::hand_eval::evaluate_5(&five));
                            }
                        }
                    }
                }
            }
            let want = 7463 - crate::hand_eval::ck_of_rank(best);
            assert_eq!(
                board_strength(hole, board, len) as u32,
                want,
                "{hole:?} {board:?} {len}"
            );
            assert!((1..=7462).contains(&want));
        }
        let (hole, board) = ([0u8, 5, 9, 13, 17], [20u8, 24, 28, 255, 255]);
        assert_eq!(board_strength(&hole, &board, 2), 0, "short board");
        assert!(board_strength(&hole, &board, 3) > 0);
        assert_eq!(
            board_strength(&hole, &board, 4),
            0,
            "an empty board card in range"
        );
        assert_eq!(
            board_strength(&[0, 5, 9, 255, 17], &board, 3),
            0,
            "an empty hole card"
        );
    }
}

/// Diagnostic hook: the layered side-pot payouts of an arbitrary commit /
/// hole-card / board configuration. Wraps the pure
/// [`crate::double_board::double_board_payout`] so Python can verify
/// side-pot handling without driving a real GameState.
///
/// `hole_cards` must be `hole_count * num_seats` card indices (4, 5 or 6 per
/// seat for PLO4 / PLO5 / PLO6), flat-packed by seat (seat 0's cards, then
/// seat 1's, ...); `board_a` / `board_b` are 5 indices each; 2..=8 seats; all
/// cards distinct. Returns chips *won* per seat (sum equals
/// `total_commit.sum()`). Bad input is a `ValueError`, never a panic
/// (ENG-029: 33+ seats reached an `assert!`).
#[pyfunction]
pub fn compute_double_board_payout(
    hole_cards: Vec<u8>,
    folded: Vec<bool>,
    total_commit: Vec<u64>,
    board_a: Vec<u8>,
    board_b: Vec<u8>,
    button: usize,
) -> PyResult<Vec<u64>> {
    let n = folded.len();
    if !(2..=MAX_SEATS).contains(&n) {
        return Err(PyValueError::new_err(format!(
            "num_seats must be in 2..={MAX_SEATS}, got {n}"
        )));
    }
    if total_commit.len() != n {
        return Err(PyValueError::new_err(
            "total_commit length must equal num_seats",
        ));
    }
    let hole_w = if n > 0 && hole_cards.len() == 6 * n {
        6
    } else if n > 0 && hole_cards.len() == 4 * n {
        4
    } else {
        5
    };
    if hole_cards.len() != hole_w * n {
        return Err(PyValueError::new_err(
            "hole_cards must have 4*num_seats (PLO4), 5*num_seats (PLO5), \
             or 6*num_seats (PLO6) indices",
        ));
    }
    if button >= n {
        return Err(PyValueError::new_err("button out of range"));
    }
    let mut holes: Vec<Vec<Card>> = Vec::with_capacity(n);
    for s in 0..n {
        let slice = &hole_cards[hole_w * s..hole_w * (s + 1)];
        let mut hole = Vec::with_capacity(hole_w);
        for &ix in slice {
            if ix >= 52 {
                return Err(PyValueError::new_err("hole_cards index out of range"));
            }
            hole.push(Card::from_index(ix));
        }
        holes.push(hole);
    }
    let board_a = cards_from_indices::<5>(&board_a, "board_a")?;
    let board_b = cards_from_indices::<5>(&board_b, "board_b")?;
    let mut seen = CardMask::EMPTY;
    for &c in holes
        .iter()
        .flatten()
        .chain(board_a.iter())
        .chain(board_b.iter())
    {
        if !seen.insert(c) {
            return Err(PyValueError::new_err(format!(
                "card {} appears twice",
                c.index()
            )));
        }
    }
    Ok(crate::double_board::double_board_payout(
        &holes,
        &folded,
        &total_commit,
        &board_a,
        &board_b,
        button,
    ))
}

/// A 2-D pyfunction input with contiguous rows: the array itself when it is
/// already C-contiguous (no copy -- PERF-032), else a C-order copy. A strided
/// view's rows are NOT contiguous, so the per-row `as_slice().unwrap()` in the
/// feature pyfunctions panicked on e.g. `np.asfortranarray(hole)` (review
/// 2026-09-20 C4).
pub(super) fn c_order<'a, T: Clone>(
    v: numpy::ndarray::ArrayView2<'a, T>,
) -> numpy::ndarray::CowArray<'a, T, numpy::ndarray::Ix2> {
    if v.is_standard_layout() {
        v.into()
    } else {
        v.as_standard_layout().into_owned().into()
    }
}

// =============================================================================
// Straight / flush / SF features (vectorized port of
// `_straight_flush_features_batch` in python/plo5bp/encoding.py).
// =============================================================================

pub(super) const SF_RANK_MASK_13: u16 = 0x1FFF;

// 10 straight windows; each is a 13-bit mask over ranks. Window 0 is the
// wheel (A-2-3-4-5 = ranks {12,0,1,2,3}); window 9 is the broadway
// (T-J-Q-K-A = ranks {8,9,10,11,12}).
/// The ten straight windows (wheel first, broadway last): the one table in
/// `hand_eval` (ENG-016 — this was a hand-written copy).
pub(super) const SF_W_MASKS: [u16; 10] = crate::hand_eval::WINDOW_BITS;

#[inline]
pub(super) fn sf_ranks_above(h_max: i8) -> u16 {
    // 13-bit mask of ranks r where r > h_max.
    if h_max < 0 {
        SF_RANK_MASK_13
    } else if h_max >= 12 {
        0
    } else {
        let cutoff = (h_max + 1) as u32;
        SF_RANK_MASK_13 & !((1u16 << cutoff) - 1)
    }
}

#[inline]
pub(super) fn sf_derive_card_state(row: &[u8]) -> (u16, [u16; 4], [u8; 4], [i8; 4]) {
    // Returns (rank_mask, rank_suit_per_suit, suit_count, max_rank_per_suit).
    let mut rank_mask: u16 = 0;
    let mut rank_suit: [u16; 4] = [0; 4];
    let mut suit_count: [u8; 4] = [0; 4];
    let mut max_per_suit: [i8; 4] = [-1, -1, -1, -1];
    for &c in row {
        if c < 52 {
            let r = (c >> 2) as usize; // 0..13
            let s = (c & 3) as usize; // 0..4
            rank_mask |= 1u16 << r;
            rank_suit[s] |= 1u16 << r;
            suit_count[s] += 1;
            if (r as i8) > max_per_suit[s] {
                max_per_suit[s] = r as i8;
            }
        }
    }
    (rank_mask, rank_suit, suit_count, max_per_suit)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sf_compute_board(
    hole_rank_mask: u16,
    hole_rank_suit: &[u16; 4],
    hole_suit_count: &[u8; 4],
    hole_max_per_suit: &[i8; 4],
    board_row: &[u8],
    vct: &[u8; 13],
    unseen_suit: &[u16; 4],
    visible_per_suit: &[u8; 4],
    out: &mut [f32],
) {
    debug_assert!(out.len() >= 38);

    let (board_rm, board_rs, board_sc, _) = sf_derive_card_state(board_row);

    let mut makes_window: [bool; 10] = [false; 10];
    let mut straight_outs: [u32; 10] = [0; 10];
    let mut straight_possible: [u32; 10] = [0; 10];
    let mut sf_cand: [u16; 4] = [0; 4];

    for (w_i, &w_mask) in SF_W_MASKS.iter().enumerate() {
        // Suit-agnostic straight in this window.
        let b_w = board_rm & w_mask;
        let h_w = hole_rank_mask & w_mask;
        let l_mask = w_mask & !board_rm;
        let m_mask = l_mask & !hole_rank_mask;
        let n_b_w = b_w.count_ones();
        let n_h_w = h_w.count_ones();
        let n_l = l_mask.count_ones();
        let n_m = m_mask.count_ones();

        let makes = n_m == 0 && n_h_w >= 2 && n_l <= 2;
        makes_window[w_i] = makes;
        if n_b_w >= 3 {
            straight_possible[w_i] = 1;
        }

        let gate = !makes && n_h_w >= 2;
        let cond_3_0 = n_l == 3 && n_m == 0 && gate;
        let cond_m1 = n_m == 1 && (1..=3).contains(&n_l) && gate;
        if cond_3_0 {
            let mut total: u32 = 0;
            let mut bits = l_mask;
            while bits != 0 {
                let r = bits.trailing_zeros() as usize;
                total += 4u32 - vct[r] as u32;
                bits &= bits - 1;
            }
            straight_outs[w_i] = total;
        } else if cond_m1 {
            let r = m_mask.trailing_zeros() as usize;
            straight_outs[w_i] = 4u32 - vct[r] as u32;
        }

        // Suit-restricted (SF) per-suit candidates for this window.
        for s in 0..4 {
            let h_s = hole_rank_suit[s] & w_mask;
            let l_s = w_mask & !board_rs[s];
            let m_s = l_s & !hole_rank_suit[s];
            let n_h_s = h_s.count_ones();
            let n_l_s = l_s.count_ones();
            let n_m_s = m_s.count_ones();
            let already_s = n_m_s == 0 && n_h_s >= 2 && n_l_s <= 2;
            let gate_s = !already_s && n_h_s >= 2;
            let cond_3_0_s = n_l_s == 3 && n_m_s == 0 && gate_s;
            let cond_m1_s = n_m_s == 1 && (1..=3).contains(&n_l_s) && gate_s;
            if cond_3_0_s {
                sf_cand[s] |= l_s;
            } else if cond_m1_s {
                sf_cand[s] |= m_s;
            }
        }
    }

    // Straight nut distance: only counted when hero has a made straight.
    let mut h_max_straight: i32 = -1;
    for w in 0..10 {
        if makes_window[w] {
            h_max_straight = w as i32;
        }
    }
    let any_made_straight = h_max_straight >= 0;
    let mut straight_nut_dist: u32 = 0;
    if any_made_straight {
        for w in 0..10 {
            if (w as i32) > h_max_straight && straight_possible[w] != 0 {
                straight_nut_dist += 1;
            }
        }
    }

    // Flush features.
    let mut flush_possible: [u32; 4] = [0; 4];
    let mut flush_draw_mask: [bool; 4] = [false; 4];
    let mut flush_draw_outs: [u32; 4] = [0; 4];
    let mut nut_flush_draw_outs: [u32; 4] = [0; 4];
    for s in 0..4 {
        if board_sc[s] >= 3 {
            flush_possible[s] = 1;
        }
        let is_draw = hole_suit_count[s] >= 2 && board_sc[s] == 2;
        flush_draw_mask[s] = is_draw;
        if is_draw {
            flush_draw_outs[s] = 13u32 - visible_per_suit[s] as u32;
            let above_mask = sf_ranks_above(hole_max_per_suit[s]);
            let blockers = (unseen_suit[s] & above_mask).count_ones();
            nut_flush_draw_outs[s] = match blockers {
                0 => flush_draw_outs[s],
                1 => 1,
                _ => 0,
            };
        }
    }

    // Made-flush nut distance: first suit where hero has >=2 + board has >=3.
    // At most one suit per env can satisfy (each player holds 5 cards but the
    // numpy oracle takes argmax = first True so we preserve that ordering).
    let mut flush_nut_dist: u32 = 0;
    for s in 0..4 {
        if board_sc[s] >= 3 && hole_suit_count[s] >= 2 {
            let above_mask = sf_ranks_above(hole_max_per_suit[s]);
            flush_nut_dist = (unseen_suit[s] & above_mask).count_ones();
            break;
        }
    }

    // SF outs per suit (gated by flush draw on that suit).
    let mut sf_outs_per_suit: [u32; 4] = [0; 4];
    for s in 0..4 {
        if flush_draw_mask[s] {
            sf_outs_per_suit[s] = (sf_cand[s] & unseen_suit[s]).count_ones();
        }
    }

    out[0] = flush_nut_dist as f32;
    out[1] = straight_nut_dist as f32;
    for w in 0..10 {
        out[2 + w] = straight_outs[w] as f32;
        out[12 + w] = straight_possible[w] as f32;
    }
    for s in 0..4 {
        out[22 + s] = flush_possible[s] as f32;
        out[26 + s] = flush_draw_outs[s] as f32;
        out[30 + s] = nut_flush_draw_outs[s] as f32;
        out[34 + s] = sf_outs_per_suit[s] as f32;
    }
}

#[pyfunction]
pub fn straight_flush_features_batch<'py>(
    py: Python<'py>,
    hole: PyReadonlyArray2<'_, u8>,
    board_a: PyReadonlyArray2<'_, u8>,
    board_b: PyReadonlyArray2<'_, u8>,
    visible_count: PyReadonlyArray3<'_, i8>,
) -> PyResult<(F32Mat<'py>, F32Mat<'py>)> {
    let hole_v = hole.as_array();
    let ba_v = board_a.as_array();
    let bb_v = board_b.as_array();
    let vc_v = visible_count.as_array();

    let n = hole_v.shape()[0];
    // Width straight from the array: an EMPTY PLO4/PLO6 batch is (0, 4) /
    // (0, 6), which the old `n > 0 ? shape[1] : 5` guess rejected.
    let hole_w = hole_v.shape()[1];
    if !batched_plo_hole_width(hole_w) {
        return Err(PyValueError::new_err(
            "hole shape must be (N, 4), (N, 5), or (N, 6)",
        ));
    }
    if ba_v.shape() != [n, 5] || bb_v.shape() != [n, 5] {
        return Err(PyValueError::new_err(
            "board_a and board_b shapes must be (N, 5) matching hole",
        ));
    }
    if vc_v.shape() != [n, 13, 4] {
        return Err(PyValueError::new_err(
            "visible_count shape must be (N, 13, 4)",
        ));
    }

    // Rows as contiguous slices for the GIL-free parallel section: the
    // arrays themselves when already C-contiguous (see `c_order`).
    let hole_owned = c_order(hole_v);
    let ba_owned = c_order(ba_v);
    let bb_owned = c_order(bb_v);
    let vc_owned = vc_v; // indexed element-wise, any layout

    let mut sf_a = vec![0f32; n * 38];
    let mut sf_b = vec![0f32; n * 38];

    py.detach(|| {
        sf_a.par_chunks_exact_mut(38)
            .zip(sf_b.par_chunks_exact_mut(38))
            .enumerate()
            .for_each(|(i, (a_out, b_out))| {
                let hole_row = hole_owned.row(i);
                let ba_row = ba_owned.row(i);
                let bb_row = bb_owned.row(i);
                let hole_slice = hole_row.as_slice().unwrap();
                let ba_slice = ba_row.as_slice().unwrap();
                let bb_slice = bb_row.as_slice().unwrap();

                let (h_rm, h_rs, h_sc, h_mps) = sf_derive_card_state(hole_slice);

                // Per-env visibility summaries derived from vc_owned[i].
                let mut vct: [u8; 13] = [0; 13];
                let mut unseen_suit: [u16; 4] = [0; 4];
                let mut visible_per_suit: [u8; 4] = [0; 4];
                for r in 0..13 {
                    for s in 0..4 {
                        if vc_owned[[i, r, s]] == 0 {
                            unseen_suit[s] |= 1u16 << r;
                        } else {
                            vct[r] += 1;
                            visible_per_suit[s] += 1;
                        }
                    }
                }

                sf_compute_board(
                    h_rm,
                    &h_rs,
                    &h_sc,
                    &h_mps,
                    ba_slice,
                    &vct,
                    &unseen_suit,
                    &visible_per_suit,
                    a_out,
                );
                sf_compute_board(
                    h_rm,
                    &h_rs,
                    &h_sc,
                    &h_mps,
                    bb_slice,
                    &vct,
                    &unseen_suit,
                    &visible_per_suit,
                    b_out,
                );
            });
    });

    let sf_a_arr = Array2::from_shape_vec((n, 38), sf_a)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let sf_b_arr = Array2::from_shape_vec((n, 38), sf_b)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((sf_a_arr.into_pyarray(py), sf_b_arr.into_pyarray(py)))
}

#[cfg(test)]
mod sf_tests {
    use super::*;

    fn build_vc(hole: &[u8], boards: &[&[u8]]) -> [[i8; 4]; 13] {
        let mut vc = [[0i8; 4]; 13];
        for &c in hole.iter().chain(boards.iter().flat_map(|b| b.iter())) {
            if c < 52 {
                let r = (c >> 2) as usize;
                let s = (c & 3) as usize;
                vc[r][s] = 1;
            }
        }
        vc
    }

    fn summarize_vc(vc: &[[i8; 4]; 13]) -> ([u8; 13], [u16; 4], [u8; 4]) {
        let mut vct = [0u8; 13];
        let mut unseen = [0u16; 4];
        let mut vps = [0u8; 4];
        for r in 0..13 {
            for s in 0..4 {
                if vc[r][s] == 0 {
                    unseen[s] |= 1u16 << r;
                } else {
                    vct[r] += 1;
                    vps[s] += 1;
                }
            }
        }
        (vct, unseen, vps)
    }

    fn run_one(hole: &[u8], board_a: &[u8], board_b: &[u8]) -> ([f32; 38], [f32; 38]) {
        let vc = build_vc(hole, &[board_a, board_b]);
        let (vct, unseen, vps) = summarize_vc(&vc);
        let mut hole_arr = [255u8; 5];
        let mut ba_arr = [255u8; 5];
        let mut bb_arr = [255u8; 5];
        for (i, &c) in hole.iter().enumerate().take(5) {
            hole_arr[i] = c;
        }
        for (i, &c) in board_a.iter().enumerate().take(5) {
            ba_arr[i] = c;
        }
        for (i, &c) in board_b.iter().enumerate().take(5) {
            bb_arr[i] = c;
        }
        let (h_rm, h_rs, h_sc, h_mps) = sf_derive_card_state(&hole_arr);
        let mut out_a = [0f32; 38];
        let mut out_b = [0f32; 38];
        sf_compute_board(
            h_rm, &h_rs, &h_sc, &h_mps, &ba_arr, &vct, &unseen, &vps, &mut out_a,
        );
        sf_compute_board(
            h_rm, &h_rs, &h_sc, &h_mps, &bb_arr, &vct, &unseen, &vps, &mut out_b,
        );
        (out_a, out_b)
    }

    // Card encoding helper: rank * 4 + suit. Ranks 0..13 (2=0, A=12).
    fn card(r: u8, s: u8) -> u8 {
        r * 4 + s
    }

    #[test]
    fn empty_inputs_produce_zero_rows() {
        let (out_a, out_b) = run_one(&[], &[], &[]);
        assert!(out_a.iter().all(|&x| x == 0.0));
        assert!(out_b.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn ranks_above_edges() {
        assert_eq!(sf_ranks_above(-1), 0x1FFF);
        assert_eq!(sf_ranks_above(12), 0);
        // h_max = 8 -> ranks 9..12 set: bits 9,10,11,12 = 0x1E00
        assert_eq!(sf_ranks_above(8), 0x1E00);
        // h_max = 0 -> ranks 1..12 set: bits 1..12 = 0x1FFE
        assert_eq!(sf_ranks_above(0), 0x1FFE);
    }

    #[test]
    fn broadway_made_straight_no_higher_possible() {
        // Hole: Ac Kc (12,11 of suit 0). Board A: Q,J,T of suit 0. Board B empty.
        // This is also a made flush — but we're checking straight features here.
        // Straight window 9 (T-J-Q-K-A): hero has 2 (K,A), board has 3 (T,J,Q).
        // makes_window[9] = true. Higher windows: none. straight_nut_dist = 0.
        let hole = [card(12, 0), card(11, 0)];
        let ba = [card(10, 0), card(9, 0), card(8, 0)]; // T,J,Q
        let (out_a, _) = run_one(&hole, &ba, &[]);
        // index 1 = straight_nut_dist
        assert_eq!(out_a[1], 0.0);
        // index 12+9 = 21 = straight_possible[9]
        assert_eq!(out_a[21], 1.0);
    }

    #[test]
    fn flush_draw_with_nut_blockers_reported_separately() {
        // Hole: Th, 2h (suit 1). Board A: Ah, 5h, 9c.
        // Hole suit-1 count = 2. Board suit-1 count = 2 -> flush draw.
        // Highest hero heart = T (rank 8). Above-h_max unseen suit-1 cards:
        // J,Q,K,A of hearts. Board contains Ah (rank 12, suit 1). So visible
        // includes Ah; unseen above T are J,Q,K = 3 blockers.
        // flush_draw_outs = 13 - visible_per_suit[1] = 13 - (Th + 2h + Ah + 5h) = 13 - 4 = 9.
        // nut_flush_draw_outs: blockers = 3 -> 0.
        let hole = [card(8, 1), card(0, 1)]; // Th, 2h
        let ba = [card(12, 1), card(3, 1), card(7, 2)]; // Ah, 5h, 9c (rank 7 = 9, suit 2 = c)
        let (out_a, _) = run_one(&hole, &ba, &[]);
        // index 26+1 = 27 = flush_draw_outs[1]
        assert_eq!(out_a[27], 9.0);
        // index 30+1 = 31 = nut_flush_draw_outs[1]
        assert_eq!(out_a[31], 0.0);
    }

    #[test]
    fn made_flush_nut_distance_counts_higher_unseen() {
        // Hole: Kh, Qh. Board A: 2h, 5h, 9h (three of hearts).
        // hole suit-1 count = 2, board suit-1 count = 3 -> made flush.
        // h_max for suit 1 = K (rank 11). Above-K unseen suit-1 cards:
        // only Ah (rank 12). visible suit-1: Kh, Qh, 2h, 5h, 9h. Ah is unseen.
        // flush_nut_dist = 1.
        let hole = [card(11, 1), card(10, 1)]; // Kh, Qh
        let ba = [card(0, 1), card(3, 1), card(7, 1)]; // 2h, 5h, 9h
        let (out_a, _) = run_one(&hole, &ba, &[]);
        // index 0 = flush_nut_dist
        assert_eq!(out_a[0], 1.0);
        // flush_possible[1] at index 22+1=23 should be 1
        assert_eq!(out_a[23], 1.0);
    }
}

// =============================================================================
// Cross-board straight features (vectorized port of
// `_cross_board_straight_batch` in python/plo5bp/encoding.py).
// =============================================================================

pub(super) const fn build_pair_bits_78() -> [u16; 78] {
    let mut out = [0u16; 78];
    let mut idx = 0;
    let mut r1: u16 = 0;
    while r1 < 13 {
        let mut r2: u16 = r1 + 1;
        while r2 < 13 {
            out[idx] = (1u16 << r1) | (1u16 << r2);
            idx += 1;
            r2 += 1;
        }
        r1 += 1;
    }
    out
}

pub(super) static PAIR_BITS_78: [u16; 78] = build_pair_bits_78();

pub(super) const fn build_pair_in_w() -> [u128; 10] {
    let mut out = [0u128; 10];
    let pairs = build_pair_bits_78();
    let mut w_idx = 0;
    while w_idx < 10 {
        let w = SF_W_MASKS[w_idx];
        let mut bits: u128 = 0;
        let mut p = 0;
        while p < 78 {
            if pairs[p] & w == pairs[p] {
                bits |= 1u128 << p;
            }
            p += 1;
        }
        out[w_idx] = bits;
        w_idx += 1;
    }
    out
}

pub(super) static PAIR_IN_W: [u128; 10] = build_pair_in_w();

#[inline]
pub(super) fn pack_rank_mask(row: &[bool]) -> u16 {
    let mut bits: u16 = 0;
    for (r, &b) in row.iter().enumerate().take(13) {
        if b {
            bits |= 1u16 << r;
        }
    }
    bits
}

#[inline]
pub(super) fn cross_board_straight_per_env(
    hero_bits: u16,
    ba_bits: u16,
    bb_bits: u16,
    valid: bool,
) -> (f32, f32, f32) {
    if !valid {
        return (0.0, 0.0, 0.0);
    }

    let mut hero_has_pair: u128 = 0;
    for p in 0..78 {
        let pb = PAIR_BITS_78[p];
        if hero_bits & pb == pb {
            hero_has_pair |= 1u128 << p;
        }
    }
    if hero_has_pair == 0 {
        return (0.0, 0.0, 0.0);
    }

    let mut made_a: u128 = 0;
    let mut draw_a: u128 = 0;
    let mut made_b: u128 = 0;
    let mut draw_b: u128 = 0;

    for w_idx in 0..10 {
        let w = SF_W_MASKS[w_idx];
        let gate = hero_has_pair & PAIR_IN_W[w_idx];
        if gate == 0 {
            continue;
        }
        let ba_w = ba_bits & w;
        let bb_w = bb_bits & w;

        let mut bits = gate;
        while bits != 0 {
            let p = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let pb = PAIR_BITS_78[p];
            let cov_a = (ba_w | pb).count_ones();
            let cov_b = (bb_w | pb).count_ones();
            let pmask = 1u128 << p;
            if cov_a >= 5 {
                made_a |= pmask;
            } else if cov_a == 4 {
                draw_a |= pmask;
            }
            if cov_b >= 5 {
                made_b |= pmask;
            } else if cov_b == 4 {
                draw_b |= pmask;
            }
        }
    }

    let draw_only_a = draw_a & !made_a;
    let draw_only_b = draw_b & !made_b;
    let made_both = (made_a & made_b) != 0;
    let draw_both = (draw_only_a & draw_only_b) != 0;
    let mixed = ((made_a & draw_only_b) | (made_b & draw_only_a)) != 0;

    (
        if made_both { 1.0 } else { 0.0 },
        if draw_both { 1.0 } else { 0.0 },
        if mixed { 1.0 } else { 0.0 },
    )
}

#[pyfunction]
pub fn cross_board_straight_batch<'py>(
    py: Python<'py>,
    hole_rank_mask: PyReadonlyArray2<'_, bool>,
    ba_rank_mask: PyReadonlyArray2<'_, bool>,
    bb_rank_mask: PyReadonlyArray2<'_, bool>,
    valid: PyReadonlyArray1<'_, bool>,
) -> PyResult<(F32Vec<'py>, F32Vec<'py>, F32Vec<'py>)> {
    let hole_v = hole_rank_mask.as_array();
    let ba_v = ba_rank_mask.as_array();
    let bb_v = bb_rank_mask.as_array();
    let valid_v = valid.as_array();

    let n = hole_v.shape()[0];
    if hole_v.shape() != [n, 13] {
        return Err(PyValueError::new_err(
            "hole_rank_mask shape must be (N, 13)",
        ));
    }
    if ba_v.shape() != [n, 13] || bb_v.shape() != [n, 13] {
        return Err(PyValueError::new_err(
            "ba_rank_mask and bb_rank_mask shapes must be (N, 13) matching hole",
        ));
    }
    if valid_v.shape() != [n] {
        return Err(PyValueError::new_err("valid shape must be (N,)"));
    }

    let hole_owned = c_order(hole_v);
    let ba_owned = c_order(ba_v);
    let bb_owned = c_order(bb_v);
    let valid_owned = valid_v;

    let mut made_both = vec![0f32; n];
    let mut draw_both = vec![0f32; n];
    let mut mixed = vec![0f32; n];

    py.detach(|| {
        made_both
            .par_iter_mut()
            .zip(draw_both.par_iter_mut())
            .zip(mixed.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((md, dr), mx))| {
                let h_bits = pack_rank_mask(hole_owned.row(i).as_slice().unwrap());
                let a_bits = pack_rank_mask(ba_owned.row(i).as_slice().unwrap());
                let b_bits = pack_rank_mask(bb_owned.row(i).as_slice().unwrap());
                let v = valid_owned[i];
                let (m, d, x) = cross_board_straight_per_env(h_bits, a_bits, b_bits, v);
                *md = m;
                *dr = d;
                *mx = x;
            });
    });

    Ok((
        Array1::from_vec(made_both).into_pyarray(py),
        Array1::from_vec(draw_both).into_pyarray(py),
        Array1::from_vec(mixed).into_pyarray(py),
    ))
}

// =============================================================================
// Draw flags (vectorized port of `_draw_flags_batch` in encoding.py).
// =============================================================================

#[inline]
pub(super) fn draw_flags_one_board(
    hole_suit_count: &[u8; 4],
    hole_rank_mask: u16,
    board_row: &[u8],
    obs_rev: u8,
) -> (f32, f32) {
    let mut board_suit_count: [u8; 4] = [0; 4];
    let mut board_rank_mask: u16 = 0;
    let mut board_has_cards = false;
    for &c in board_row {
        if c < 52 {
            let r = (c >> 2) as usize;
            let s = (c & 3) as usize;
            board_suit_count[s] += 1;
            board_rank_mask |= 1u16 << r;
            board_has_cards = true;
        }
    }
    if !board_has_cards {
        return (0.0, 0.0);
    }

    let mut flush = false;
    for s in 0..4 {
        if hole_suit_count[s] >= 2 && board_suit_count[s] == 2 {
            flush = true;
            break;
        }
    }

    let rank_mask = hole_rank_mask | board_rank_mask;
    // Rev 2: ranks shifted up one (rank r -> bit r+1) with the ace-low shadow
    // at bit 0, BELOW the deuce. Window check looks for 4 consecutive ranks
    // across bits {0..13} — A234 (start 0) through JQKA (start 10) —
    // matching the scalar `_draw_flags`.
    // PRODUCTION BEHAVIOR CHANGE (review 2026-09-20 B2, dims 800/802): in
    // rev 1 (kept for old checkpoints) the shadow sits at bit 13, ABOVE the
    // ace, so Q-K-A reads as 4-in-a-row (false positive) and A-2-3-4 never
    // fires (false negative).
    let ace_bit = (rank_mask >> 12) & 1;
    let extended = if obs_rev == OBS_REV_LEGACY {
        rank_mask | (ace_bit << 13)
    } else {
        (rank_mask << 1) | ace_bit
    };
    let mut straight = false;
    for start in 0..11u32 {
        if (extended >> start) & 0b1111 == 0b1111 {
            straight = true;
            break;
        }
    }

    (
        if flush { 1.0 } else { 0.0 },
        if straight { 1.0 } else { 0.0 },
    )
}

/// `obs_rev` selects the straight flag's ace handling (see
/// `draw_flags_one_board`); the numpy batch encoder passes
/// `encoding.OBS_SEMANTICS_REV` explicitly. Default = the current revision.
#[pyfunction]
#[pyo3(signature = (hole, board_a, board_b, obs_rev=OBS_REV_CURRENT))]
pub fn draw_flags_batch<'py>(
    py: Python<'py>,
    hole: PyReadonlyArray2<'_, u8>,
    board_a: PyReadonlyArray2<'_, u8>,
    board_b: PyReadonlyArray2<'_, u8>,
    obs_rev: u8,
) -> PyResult<(F32Vec<'py>, F32Vec<'py>, F32Vec<'py>, F32Vec<'py>)> {
    let obs_rev = check_obs_rev(obs_rev)?;
    let hole_v = hole.as_array();
    let ba_v = board_a.as_array();
    let bb_v = board_b.as_array();

    let n = hole_v.shape()[0];
    // Width straight from the array: an EMPTY PLO4/PLO6 batch is (0, 4) /
    // (0, 6), which the old `n > 0 ? shape[1] : 5` guess rejected.
    let hole_w = hole_v.shape()[1];
    if !batched_plo_hole_width(hole_w) {
        return Err(PyValueError::new_err(
            "hole shape must be (N, 4), (N, 5), or (N, 6)",
        ));
    }
    if ba_v.shape() != [n, 5] || bb_v.shape() != [n, 5] {
        return Err(PyValueError::new_err(
            "board_a and board_b shapes must be (N, 5) matching hole",
        ));
    }

    let hole_owned = c_order(hole_v);
    let ba_owned = c_order(ba_v);
    let bb_owned = c_order(bb_v);

    let mut flush_a = vec![0f32; n];
    let mut straight_a = vec![0f32; n];
    let mut flush_b = vec![0f32; n];
    let mut straight_b = vec![0f32; n];

    py.detach(|| {
        flush_a
            .par_iter_mut()
            .zip(straight_a.par_iter_mut())
            .zip(flush_b.par_iter_mut())
            .zip(straight_b.par_iter_mut())
            .enumerate()
            .for_each(|(i, (((fa, sa), fb), sb))| {
                let hole_slice = hole_owned.row(i);
                let hole_slice = hole_slice.as_slice().unwrap();
                let mut hole_suit_count: [u8; 4] = [0; 4];
                let mut hole_rank_mask: u16 = 0;
                for &c in hole_slice {
                    if c < 52 {
                        let r = (c >> 2) as usize;
                        let s = (c & 3) as usize;
                        hole_suit_count[s] += 1;
                        hole_rank_mask |= 1u16 << r;
                    }
                }

                let ba_slice = ba_owned.row(i);
                let bb_slice = bb_owned.row(i);
                let (fa_v, sa_v) = draw_flags_one_board(
                    &hole_suit_count,
                    hole_rank_mask,
                    ba_slice.as_slice().unwrap(),
                    obs_rev,
                );
                let (fb_v, sb_v) = draw_flags_one_board(
                    &hole_suit_count,
                    hole_rank_mask,
                    bb_slice.as_slice().unwrap(),
                    obs_rev,
                );
                *fa = fa_v;
                *sa = sa_v;
                *fb = fb_v;
                *sb = sb_v;
            });
    });

    Ok((
        Array1::from_vec(flush_a).into_pyarray(py),
        Array1::from_vec(straight_a).into_pyarray(py),
        Array1::from_vec(flush_b).into_pyarray(py),
        Array1::from_vec(straight_b).into_pyarray(py),
    ))
}

// =============================================================================
// Pair features (vectorized port of `_pair_features_batch` in encoding.py).
// =============================================================================

#[inline]
pub(super) fn pair_features_one_board(
    hole_rank_counts: &[u8; 13],
    board_row: &[u8],
    counts_out: &mut [f32], // length 5
    struct_out: &mut [f32], // length 4
) {
    let mut board_rank_counts: [u8; 13] = [0; 13];
    let mut board_ranks: [u8; 5] = [0; 5];
    let mut valid_count: usize = 0;
    for &c in board_row.iter().take(5) {
        if c < 52 {
            let r = c >> 2;
            board_ranks[valid_count] = r;
            board_rank_counts[r as usize] += 1;
            valid_count += 1;
        }
    }

    // Pre-zero outputs (caller may reuse buffers).
    for slot in 0..5 {
        counts_out[slot] = 0.0;
    }
    for slot in 0..4 {
        struct_out[slot] = 0.0;
    }

    if valid_count == 0 {
        return;
    }

    // Sort the valid prefix descending.
    board_ranks[..valid_count].sort_unstable_by(|a, b| b.cmp(a));

    for slot in 0..valid_count {
        counts_out[slot] = hole_rank_counts[board_ranks[slot] as usize] as f32;
    }

    let mut paired = false;
    let mut tripled = false;
    let mut quadded = false;
    let mut pairs: u32 = 0;
    for &c in &board_rank_counts {
        if c >= 2 {
            paired = true;
            pairs += 1;
        }
        if c >= 3 {
            tripled = true;
        }
        if c >= 4 {
            quadded = true;
        }
    }
    struct_out[0] = if paired { 1.0 } else { 0.0 };
    struct_out[1] = if pairs >= 2 { 1.0 } else { 0.0 };
    struct_out[2] = if tripled { 1.0 } else { 0.0 };
    struct_out[3] = if quadded { 1.0 } else { 0.0 };
}

#[pyfunction]
pub fn pair_features_batch<'py>(
    py: Python<'py>,
    hole: PyReadonlyArray2<'_, u8>,
    board_a: PyReadonlyArray2<'_, u8>,
    board_b: PyReadonlyArray2<'_, u8>,
) -> PyResult<(F32Mat<'py>, F32Mat<'py>, F32Mat<'py>, F32Mat<'py>)> {
    let hole_v = hole.as_array();
    let ba_v = board_a.as_array();
    let bb_v = board_b.as_array();

    let n = hole_v.shape()[0];
    // Width straight from the array: an EMPTY PLO4/PLO6 batch is (0, 4) /
    // (0, 6), which the old `n > 0 ? shape[1] : 5` guess rejected.
    let hole_w = hole_v.shape()[1];
    if !batched_plo_hole_width(hole_w) {
        return Err(PyValueError::new_err(
            "hole shape must be (N, 4), (N, 5), or (N, 6)",
        ));
    }
    if ba_v.shape() != [n, 5] || bb_v.shape() != [n, 5] {
        return Err(PyValueError::new_err(
            "board_a and board_b shapes must be (N, 5) matching hole",
        ));
    }

    let hole_owned = c_order(hole_v);
    let ba_owned = c_order(ba_v);
    let bb_owned = c_order(bb_v);

    let mut counts_a = vec![0f32; n * 5];
    let mut struct_a = vec![0f32; n * 4];
    let mut counts_b = vec![0f32; n * 5];
    let mut struct_b = vec![0f32; n * 4];

    py.detach(|| {
        counts_a
            .par_chunks_exact_mut(5)
            .zip(struct_a.par_chunks_exact_mut(4))
            .zip(counts_b.par_chunks_exact_mut(5))
            .zip(struct_b.par_chunks_exact_mut(4))
            .enumerate()
            .for_each(|(i, (((ca, sa), cb), sb))| {
                let hole_row = hole_owned.row(i);
                let hole_slice = hole_row.as_slice().unwrap();
                let mut hole_rank_counts: [u8; 13] = [0; 13];
                for &c in hole_slice {
                    if c < 52 {
                        hole_rank_counts[(c >> 2) as usize] += 1;
                    }
                }

                let ba_row = ba_owned.row(i);
                let bb_row = bb_owned.row(i);
                pair_features_one_board(&hole_rank_counts, ba_row.as_slice().unwrap(), ca, sa);
                pair_features_one_board(&hole_rank_counts, bb_row.as_slice().unwrap(), cb, sb);
            });
    });

    let counts_a_arr = Array2::from_shape_vec((n, 5), counts_a)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let struct_a_arr = Array2::from_shape_vec((n, 4), struct_a)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let counts_b_arr = Array2::from_shape_vec((n, 5), counts_b)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let struct_b_arr = Array2::from_shape_vec((n, 4), struct_b)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        counts_a_arr.into_pyarray(py),
        struct_a_arr.into_pyarray(py),
        counts_b_arr.into_pyarray(py),
        struct_b_arr.into_pyarray(py),
    ))
}
