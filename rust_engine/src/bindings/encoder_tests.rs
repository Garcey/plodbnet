//! Unit tests of the feature kernels and the fused row encoders.

use super::*;

fn card(r: u8, s: u8) -> u8 {
    r * 4 + s
}

fn pad5(cards: &[u8]) -> [u8; 5] {
    let mut out = [255u8; 5];
    for (i, &c) in cards.iter().enumerate().take(5) {
        out[i] = c;
    }
    out
}

fn rank_mask(cards: &[u8]) -> [bool; 13] {
    let mut out = [false; 13];
    for &c in cards {
        if c < 52 {
            out[(c >> 2) as usize] = true;
        }
    }
    out
}

// ---- cross_board_straight ----

#[test]
fn cross_board_empty_inputs() {
    let (md, dr, mx) = cross_board_straight_per_env(0, 0, 0, true);
    assert_eq!(md, 0.0);
    assert_eq!(dr, 0.0);
    assert_eq!(mx, 0.0);
}

#[test]
fn cross_board_invalid_returns_zero() {
    // valid=false short-circuits even if everything else fires.
    let hero = rank_mask(&[card(10, 0), card(11, 0)]);
    let ba = rank_mask(&[card(8, 0), card(9, 0), card(12, 0)]);
    let bb = rank_mask(&[card(8, 1), card(9, 1), card(12, 1)]);
    let hb = pack_rank_mask(&hero);
    let ab = pack_rank_mask(&ba);
    let bb_ = pack_rank_mask(&bb);
    let (md, dr, mx) = cross_board_straight_per_env(hb, ab, bb_, false);
    assert_eq!(md, 0.0);
    assert_eq!(dr, 0.0);
    assert_eq!(mx, 0.0);
}

#[test]
fn cross_board_made_both_broadway() {
    // Hero K,Q. Boards both A,J,T. Window 9 (T-J-Q-K-A): pair {K,Q}
    // in window; cov = popcount(board & W | pair). board has A,J,T
    // = bits 12,9,8; pair = bits 11,10. Union = 0x1F00 (5 bits) on
    // each board → cov_a == 5 → made; cov_b == 5 → made. made_both.
    let hero = rank_mask(&[card(11, 0), card(10, 0)]); // K=11, Q=10
    let ba = rank_mask(&[card(12, 0), card(9, 0), card(8, 0)]); // A,J,T
    let bb = rank_mask(&[card(12, 1), card(9, 1), card(8, 1)]); // A,J,T
    let (md, dr, mx) = cross_board_straight_per_env(
        pack_rank_mask(&hero),
        pack_rank_mask(&ba),
        pack_rank_mask(&bb),
        true,
    );
    assert_eq!(md, 1.0);
    assert_eq!(dr, 0.0);
    assert_eq!(mx, 0.0);
}

#[test]
fn cross_board_mixed() {
    // Hero K,Q. Board A has A,J,T (made). Board B has A,J (draw only:
    // cov = 4). → mixed.
    let hero = rank_mask(&[card(11, 0), card(10, 0)]); // K, Q
    let ba = rank_mask(&[card(12, 0), card(9, 0), card(8, 0)]); // A,J,T (5 covered)
    let bb = rank_mask(&[card(12, 1), card(9, 1)]); // A,J (4 covered with pair)
    let (md, dr, mx) = cross_board_straight_per_env(
        pack_rank_mask(&hero),
        pack_rank_mask(&ba),
        pack_rank_mask(&bb),
        true,
    );
    assert_eq!(md, 0.0);
    assert_eq!(dr, 0.0);
    assert_eq!(mx, 1.0);
}

// ---- draw_flags ----

#[test]
fn draw_flags_empty_board() {
    let hole = pad5(&[card(11, 0), card(10, 0)]);
    let empty_board = pad5(&[]);
    let mut hsc: [u8; 4] = [0; 4];
    let mut hrm: u16 = 0;
    for &c in &hole {
        if c < 52 {
            hsc[(c & 3) as usize] += 1;
            hrm |= 1u16 << (c >> 2);
        }
    }
    let (f, s) = draw_flags_one_board(&hsc, hrm, &empty_board, OBS_REV_CURRENT);
    assert_eq!(f, 0.0);
    assert_eq!(s, 0.0);
}

#[test]
fn draw_flags_flush_draw_fires() {
    // Hero: Ah, Kh (suit 1, ranks 12, 11). Board: 2h, 5h, 9c
    // (suit 1 has 2 cards on board). hero_suit[1] = 2, board_suit[1] = 2 → flush.
    let hole = pad5(&[card(12, 1), card(11, 1)]);
    let board = pad5(&[card(0, 1), card(3, 1), card(7, 2)]);
    let mut hsc: [u8; 4] = [0; 4];
    let mut hrm: u16 = 0;
    for &c in &hole {
        if c < 52 {
            hsc[(c & 3) as usize] += 1;
            hrm |= 1u16 << (c >> 2);
        }
    }
    let (f, _) = draw_flags_one_board(&hsc, hrm, &board, OBS_REV_CURRENT);
    assert_eq!(f, 1.0);
}

/// (hole_suit_count, hole_rank_mask) the way the encoders derive them.
fn hole_summary(hole: &[u8]) -> ([u8; 4], u16) {
    let mut hsc: [u8; 4] = [0; 4];
    let mut hrm: u16 = 0;
    for &c in hole {
        if c < 52 {
            hsc[(c & 3) as usize] += 1;
            hrm |= 1u16 << (c >> 2);
        }
    }
    (hsc, hrm)
}

/// review 2026-09-20 B2: in rev 2 the ace plays BOTH ends of the 4-run
/// ladder and nothing wraps around it; rev 1 keeps the old shadow-above-
/// the-ace values bit for bit.
#[test]
fn draw_flags_ace_low_shadow_sits_below_the_deuce() {
    let straight = |hole: &[u8], board: &[u8], rev: u8| {
        let (hsc, hrm) = hole_summary(hole);
        draw_flags_one_board(&hsc, hrm, &pad5(board), rev).1
    };
    let qka_hole = [card(10, 3), card(11, 1), card(0, 0), card(5, 2), card(5, 1)];
    let qka_board = [card(12, 2), card(1, 3), card(6, 0)];
    let wheel_hole = [card(12, 3), card(0, 1), card(7, 0), card(7, 2), card(11, 1)];
    let wheel_board = [card(1, 2), card(2, 3), card(9, 0)];
    // Q-K-A is three cards at the top, not a 4-run (rev-1 false positive).
    assert_eq!(straight(&qka_hole, &qka_board, OBS_REV_CURRENT), 0.0);
    assert_eq!(straight(&qka_hole, &qka_board, OBS_REV_LEGACY), 1.0);
    // A-2-3-4 wheel draw (rev-1 false negative).
    assert_eq!(straight(&wheel_hole, &wheel_board, OBS_REV_CURRENT), 1.0);
    assert_eq!(straight(&wheel_hole, &wheel_board, OBS_REV_LEGACY), 0.0);
    for rev in [OBS_REV_LEGACY, OBS_REV_CURRENT] {
        // J-Q-K-A fires and K-A-2-3 does not wrap, in both revisions.
        assert_eq!(
            straight(
                &[card(9, 3), card(10, 1)],
                &[card(11, 2), card(12, 3), card(6, 0)],
                rev
            ),
            1.0
        );
        assert_eq!(
            straight(
                &[card(11, 3), card(12, 1)],
                &[card(0, 2), card(1, 3), card(9, 0)],
                rev
            ),
            0.0
        );
    }
}

/// review 2026-09-20 B5: a card face-up on the OTHER board is in nobody's
/// hand, so it is not the "top missing" flush card.
#[test]
fn blocker_flush_dims_skip_the_other_board() {
    let hole = [card(10, 3), card(6, 0), card(6, 1), card(2, 2), card(1, 0)]; // Qs
    let board_a = pad5(&[card(11, 3), card(5, 3), card(0, 3)]); // Ks 7s 2s
    let board_b = pad5(&[card(12, 3), card(8, 1), card(3, 2)]); // As on B
    let mut out = [0f32; 4];
    blocker_features_one_board(&hole, &board_a, &board_b, &mut out);
    assert_eq!(out[0], 1.0); // Qs is the top HOLDABLE spade
    assert_eq!(out[1], (1.0f64 / 3.0) as f32); // of Qs/Js/Ts hero holds one
    let mut alone = [0f32; 4];
    blocker_features_one_board(&hole, &board_a, &pad5(&[]), &mut alone);
    assert_eq!(alone[0], 0.0); // without board B the As is still "missing"
}

/// review 2026-09-20 B1/B3: regimes of the legal (rev 2) raise window, and
/// the totals-derived rev-1 window it replaced.
#[test]
fn raise_window_regimes() {
    let bb = 10_000;
    let w = |legal, min_d, max_d, anchor_min| RaiseWindow {
        legal,
        min_d,
        max_d,
        anchor_min,
    };
    // Normal raise.
    assert_eq!(
        legal_raise_window(20_000, 70_000, 70_000, bb),
        w(true, 20_000.0, 70_000.0, 20_000.0)
    );
    // Short shove: min 0 < max — the only legal size is the all-in, and the
    // anchor count still sees the RAW 0.
    assert_eq!(
        legal_raise_window(0, 4_000, 4_000, bb),
        w(true, 4_000.0, 4_000.0, 0.0)
    );
    // No raise at all.
    assert_eq!(
        legal_raise_window(0, 0, 50_000, bb),
        w(false, 0.0, 0.0, 0.0)
    );
    // Cover-short DUST: a sub-1bb raise that is NOT hero's own all-in is
    // screened off by the env's Raise gate.
    assert_eq!(
        legal_raise_window(4_000, 4_000, 900_000, bb),
        w(false, 0.0, 0.0, 4_000.0)
    );
    // A cover-short raise of >= 1bb is an ordinary (single-size) raise.
    assert_eq!(
        legal_raise_window(15_000, 15_000, 900_000, bb),
        w(true, 15_000.0, 15_000.0, 15_000.0)
    );
    // Rev 1, the review's example: hero 7bb behind, min_bet 1bb / max_bet
    // 18bb (the pot) as TOTALS, nothing committed, nothing to call — a
    // "legal" 18bb raise out of a 7bb stack.
    assert_eq!(
        legacy_raise_window(10_000, 180_000, 0.0, 0.0),
        w(true, 10_000.0, 180_000.0, 10_000.0)
    );
    // ... and "illegal" only when the totals cap sits at/below the call.
    assert_eq!(
        legacy_raise_window(360_000, 180_000, 0.0, 180_000.0),
        w(false, 360_000.0, 180_000.0, 360_000.0)
    );
}

/// Pack + encode env 0 of a one-env engine built around `state` (pure
/// Rust: no Python objects are touched).
fn encode_single(state: GameState, config: GameConfig, obs_rev: u8) -> Vec<f32> {
    let s = config.num_seats;
    let mut engine = PyBatchedEngine::with_config(config.clone(), 1, 0, obs_rev);
    engine.states[0] = Some(state);
    let packed = engine.pack_full(&[0]).obs;
    let mut row = vec![0f32; obs_layout::OBS_DIM];
    encode_obs_row(
        &packed,
        0,
        s,
        0,
        0,
        1.0 / config.bb as f64,
        config.bb,
        config.ante,
        &config.starting_stacks,
        obs_rev,
        &mut row,
    );
    row
}

/// review 2026-09-20 B6: STK-10 sums antes over the DEALT-IN seats. The
/// batched engine never deals masked hands, so this is the only place the
/// Rust twin of that rule is exercised.
#[test]
fn stk10_pot_at_flop_skips_sitting_out_seats() {
    let config = GameConfig {
        num_seats: 6,
        starting_stacks: vec![200_000; 6],
        ante: 30_000,
        bb: 10_000,
        sb: 0,
        variant: Variant::Plo5DoubleBomb,
    };
    let mask = vec![true, true, false, true, false, false];
    let mut g = GameState::new_hand_with_mask(config.clone(), 5, 0, Some(mask));
    assert_eq!(g.pot, 90_000); // three antes, not six
    let bet = g.max_raise_chips();
    assert!(bet > 0);
    g.apply_raise_chips(bet).unwrap();
    // Chips on top of the antes actually posted: bet / (3 antes). Six
    // antes would have read max(90k + bet - 180k, 0) / 180k = 0.
    let expected = (bet as f64 / 90_000.0).ln_1p() as f32;
    assert!(expected > 0.0);
    // B6 is NOT gated by the semantics revision.
    for rev in [OBS_REV_LEGACY, OBS_REV_CURRENT] {
        let row = encode_single(g.clone(), config.clone(), rev);
        assert_eq!(row[obs_layout::STK10_OFF + 1], expected);
    }
}

/// The semantics switch end to end through the fused row encoder, on the
/// review's B1 example (hero 7bb behind in an 18bb pot, first to act).
#[test]
fn obs_rev_switches_the_gated_dims_only() {
    let config = GameConfig {
        num_seats: 6,
        starting_stacks: vec![100_000, 500_000, 500_000, 500_000, 500_000, 500_000],
        ante: 30_000,
        bb: 10_000,
        sb: 0,
        variant: Variant::Plo5DoubleBomb,
    };
    let g = (0..6)
        .map(|button| GameState::new_hand(config.clone(), 123, button))
        .find(|g| g.current_actor() == Some(0))
        .expect("some button puts seat 0 first to act");
    assert_eq!((g.min_raise_chips(), g.max_raise_chips()), (10_000, 70_000));
    let new = encode_single(g.clone(), config.clone(), OBS_REV_CURRENT);
    let old = encode_single(g, config, OBS_REV_LEGACY);
    use obs_core::SCALARS_OFF;
    use obs_layout::*;
    // Rev 2: the legal window — 1bb..7bb, stack-capped, 5 of 11 anchors.
    assert_eq!(new[SCALARS_OFF + 2], 1.0);
    assert_eq!(new[SCALARS_OFF + 3], 7.0);
    assert_eq!(new[STK2_OFF + 1], (70_000.0f64 / 180_000.0) as f32);
    assert_eq!(new[STK2_OFF + 4], 1.0);
    assert_eq!(new[STK2_OFF + 5], (5.0f64 / 11.0) as f32);
    assert_eq!(new[STK5_OFF + 2], 0.0);
    // Rev 1: min_bet_total()/max_bet_total() — a pot-sized 11-anchor ladder
    // out of a 7bb stack, and a NEGATIVE spr-after-max-raise.
    assert_eq!(old[SCALARS_OFF + 2], 1.0);
    assert_eq!(old[SCALARS_OFF + 3], 18.0);
    assert_eq!(old[STK2_OFF + 1], 1.0);
    assert_eq!(old[STK2_OFF + 4], 0.0);
    assert_eq!(old[STK2_OFF + 5], 1.0);
    assert!(old[STK5_OFF + 2] < 0.0);
    // Nothing outside the gated dims moves.
    let gated = |d: usize| {
        (SCALARS_OFF + 2..SCALARS_OFF + 4).contains(&d)
            || d == DRAW_A_OFF + 1
            || d == DRAW_B_OFF + 1
            || (BLOCKER_A_OFF..BLOCKER_A_OFF + 2).contains(&d)
            || (BLOCKER_B_OFF..BLOCKER_B_OFF + 2).contains(&d)
            || (STK2_OFF..STK2_OFF + 6).contains(&d)
            || (STK5_OFF + 2..STK5_OFF + 4).contains(&d)
    };
    for d in 0..OBS_DIM {
        if !gated(d) {
            assert_eq!(new[d], old[d], "ungated dim {d} differs between revisions");
        }
    }
}

#[test]
fn draw_flags_straight_draw_fires() {
    // Hero+board union covers ranks 4,5,6,7 (consecutive 4) → straight draw.
    // Hero: 6c, 7c (ranks 4,5, suit 2). Board: 8d, 9d, 2c (ranks 6,7,0).
    let hole = pad5(&[card(4, 2), card(5, 2)]);
    let board = pad5(&[card(6, 3), card(7, 3), card(0, 2)]);
    let mut hsc: [u8; 4] = [0; 4];
    let mut hrm: u16 = 0;
    for &c in &hole {
        if c < 52 {
            hsc[(c & 3) as usize] += 1;
            hrm |= 1u16 << (c >> 2);
        }
    }
    let (_, s) = draw_flags_one_board(&hsc, hrm, &board, OBS_REV_CURRENT);
    assert_eq!(s, 1.0);
}

// ---- pair_features ----

#[test]
fn pair_features_empty_board() {
    let hole = pad5(&[card(11, 0), card(10, 0)]);
    let mut hrc: [u8; 13] = [0; 13];
    for &c in &hole {
        if c < 52 {
            hrc[(c >> 2) as usize] += 1;
        }
    }
    let mut counts = [0f32; 5];
    let mut struct_out = [0f32; 4];
    pair_features_one_board(&hrc, &pad5(&[]), &mut counts, &mut struct_out);
    assert!(counts.iter().all(|&x| x == 0.0));
    assert!(struct_out.iter().all(|&x| x == 0.0));
}

#[test]
fn pair_features_top_pair() {
    // Hero: Kh, Qd (ranks 11, 10). Board: Kc, 9h, 4s (ranks 11, 7, 2).
    // Sorted descending: 11, 7, 2. counts: hero K count = 1, hero 9 = 0, hero 4 = 0.
    let hole = pad5(&[card(11, 1), card(10, 3)]);
    let board = pad5(&[card(11, 2), card(7, 1), card(2, 0)]);
    let mut hrc: [u8; 13] = [0; 13];
    for &c in &hole {
        if c < 52 {
            hrc[(c >> 2) as usize] += 1;
        }
    }
    let mut counts = [0f32; 5];
    let mut struct_out = [0f32; 4];
    pair_features_one_board(&hrc, &board, &mut counts, &mut struct_out);
    assert_eq!(counts[0], 1.0); // K-slot
    assert_eq!(counts[1], 0.0); // 9-slot
    assert_eq!(counts[2], 0.0); // 4-slot
    assert_eq!(counts[3], 0.0); // unused
    assert_eq!(counts[4], 0.0); // unused
                                // No pairs on board.
    assert_eq!(struct_out[0], 0.0); // paired
}

#[test]
fn pair_features_paired_board_repeats_slots() {
    // Hero: Kh, Qd. Board: Kc, Kd, 4s (K paired). Sorted: 11,11,2.
    // counts: 1, 1, 0.
    let hole = pad5(&[card(11, 1), card(10, 3)]);
    let board = pad5(&[card(11, 2), card(11, 3), card(2, 0)]);
    let mut hrc: [u8; 13] = [0; 13];
    for &c in &hole {
        if c < 52 {
            hrc[(c >> 2) as usize] += 1;
        }
    }
    let mut counts = [0f32; 5];
    let mut struct_out = [0f32; 4];
    pair_features_one_board(&hrc, &board, &mut counts, &mut struct_out);
    assert_eq!(counts[0], 1.0);
    assert_eq!(counts[1], 1.0);
    assert_eq!(counts[2], 0.0);
    assert_eq!(struct_out[0], 1.0); // paired
    assert_eq!(struct_out[1], 0.0); // double_paired
    assert_eq!(struct_out[2], 0.0); // tripled
    assert_eq!(struct_out[3], 0.0); // quadded
}

#[test]
fn pair_features_quadded_board() {
    // Board: four Kings + 4. Struct: paired, tripled, quadded all 1.
    // double_paired = 0 (only one rank has count>=2).
    let hole = pad5(&[card(10, 0), card(9, 0)]);
    let board = pad5(&[
        card(11, 0),
        card(11, 1),
        card(11, 2),
        card(11, 3),
        card(2, 0),
    ]);
    let mut hrc: [u8; 13] = [0; 13];
    for &c in &hole {
        if c < 52 {
            hrc[(c >> 2) as usize] += 1;
        }
    }
    let mut counts = [0f32; 5];
    let mut struct_out = [0f32; 4];
    pair_features_one_board(&hrc, &board, &mut counts, &mut struct_out);
    assert_eq!(struct_out[0], 1.0); // paired
    assert_eq!(struct_out[1], 0.0); // double_paired (only Ks have c>=2)
    assert_eq!(struct_out[2], 1.0); // tripled
    assert_eq!(struct_out[3], 1.0); // quadded
}
