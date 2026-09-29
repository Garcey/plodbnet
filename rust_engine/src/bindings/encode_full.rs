//! The full (1171-dim) observation layout and its per-row encoder, including
//! the v7 tail (dims 1020..1171; formerly `include!`d from obs_v7_inc.rs).

use super::*;
use obs_layout::*;

// =============================================================================
// Observation layout (single source of truth; mirrors the offset constants in
// python/plo5bp/encoding.py's `_*_OFF` constants). A wrong value here silently
// misplaces a whole feature block, so keep in lockstep with the Python side.
// =============================================================================
pub(super) mod obs_layout {
    // Dims 0..188 (hole, boards, street, active / all-in / stacks, scalars)
    // and the history slot format are the core blocks both layouts share:
    // `obs_core` in encode.rs (encoded once, by `encode_core`).
    pub const OBS_DIM: usize = 1171;
    // v7 batch-2 tail (V7_OBS_IMPL_PLAN.md, dims 1020..1171)
    pub const STK1_OFF: usize = 1020; // 4
    pub const STK2_OFF: usize = 1024; // 6
    pub const STK4_OFF: usize = 1030; // 8
    pub const STK5_OFF: usize = 1038; // 4
    pub const STK6_OFF: usize = 1042; // 2
    pub const STK7_OFF: usize = 1044; // 2
    pub const STK8_OFF: usize = 1046; // 3
    pub const STK9_OFF: usize = 1049; // 2
    pub const STK10_OFF: usize = 1051; // 2
    pub const STK11_OFF: usize = 1053; // 8
    pub const BRD1_OFF: usize = 1061; // 10
    pub const BRD2_OFF: usize = 1071; // 12
    pub const BRD4_OFF: usize = 1083; // 6
    pub const BRD5_OFF: usize = 1089; // 6
    pub const BRD6_OFF: usize = 1095; // 4
    pub const BRD7_OFF: usize = 1099; // 2
    pub const BRD8_OFF: usize = 1101; // 4
    pub const BRD9_OFF: usize = 1105; // 4
    pub const BRD10_OFF: usize = 1109; // 2
    pub const BRD11_OFF: usize = 1111; // 20
    pub const BRD12_OFF: usize = 1131; // 4
    pub const BRD13_OFF: usize = 1135; // 4
    pub const DUAL1_OFF: usize = 1139; // 2
    pub const DUAL2_OFF: usize = 1141; // 10
    pub const DUAL3_OFF: usize = 1151; // 6
    pub const DUAL4_OFF: usize = 1157; // 5
    pub const DUAL5_OFF: usize = 1162; // 9
    pub const ANCHOR_COUNT: usize = 11;
    pub const REL_POS_OFF: usize = 188;
    pub const HISTORY_OFF: usize = 196;
    pub const NUM_CATEGORIES: usize = 9;
    pub const SPR_OFF: usize = 772;
    pub const POT_ODDS_OFF: usize = 780;
    pub const CAT_A_OFF: usize = 781;
    pub const CAT_B_OFF: usize = 790;
    pub const DRAW_A_OFF: usize = 799;
    pub const DRAW_B_OFF: usize = 801;
    pub const PAIR_COUNT_A_OFF: usize = 803;
    pub const PAIR_COUNT_B_OFF: usize = 808;
    pub const BOARD_STRUCT_A_OFF: usize = 813;
    pub const BOARD_STRUCT_B_OFF: usize = 817;
    pub const HERO_RANK_HIST_OFF: usize = 821;
    pub const FLUSH_NUT_DIST_A_OFF: usize = 834;
    pub const FLUSH_NUT_DIST_B_OFF: usize = 872;
    pub const SEAT_EXISTS_OFF: usize = 910;
    pub const TOTAL_COMMIT_OFF: usize = 918;
    pub const STREET_COMMIT_OFF: usize = 926;
    pub const LAST_AGGRESSOR_OFF: usize = 934;
    pub const HERO_BTN_DIST_OFF: usize = 942;
    pub const SHARED_RANKS_OFF: usize = 950;
    pub const FLUSH_MADE_BOTH_OFF: usize = 963;
    pub const FLUSH_DRAW_BOTH_OFF: usize = 967;
    pub const FLUSH_MIXED_OFF: usize = 971;
    pub const STRAIGHT_MADE_BOTH_OFF: usize = 975;
    pub const STRAIGHT_DRAW_BOTH_OFF: usize = 976;
    pub const STRAIGHT_MIXED_OFF: usize = 977;
    pub const OPP_OUTCOME_OFF: usize = 978;
    pub const OPP_OUTCOME_DIM: usize = 12;
    pub const BET_PCT_POT_OFF: usize = 990;
    // obs v2 tail (V5_DESIGN.md §3.2, dims 991..1020) — a pure append after the
    // 991-dim v1 core. Offsets are byte-identical to the numpy encoder's tail
    // (_PER_BOARD_OUTCOME_OFF etc. in python/plo5bp/encoding.py).
    pub const PER_BOARD_OUTCOME_OFF: usize = 991; // 8: hero ahead/tie/behind per board
    pub const BLOCKER_A_OFF: usize = 999; // 4: unconditional blockers-to-nuts, board A
    pub const BLOCKER_B_OFF: usize = 1003; // 4: unconditional blockers-to-nuts, board B
    pub const EFF_PRICE_OFF: usize = 1007; // 5: eff price + commit frac + log1p money
    pub const SPR_LOG_OFF: usize = 1012; // 8: log1p effective SPR, unclipped
}

// v7 batch-2 obs tail (dims 1020..1171) for the Rust encoder.
// Bit-exact with python/plo5bp/encoding.py `_encode_stack_v3` /
// `_encode_board_v3` / `_encode_dual_v3`.

/// STK-6[0] "bets to jam" thresholds: 3^0..3^5 as exact f64 constants. The dim
/// is clip(ceil(log3(x6)), 0, 6); computed as `ceil(ln(x6) / ln(3))` it sits ON
/// an integer at x6 = 3, 9, 27, 81, 243, where a 1-ulp difference between this
/// `ln` and numpy's (SVML on AVX-512 Linux) flips the ceil. An exact comparison
/// chain has no such boundary (review 2026-09-20 STK-6; twin of `_POW3` /
/// `_bets_to_jam` in python/plo5bp/encoding.py — same values as the log form
/// gave at those five points on the reference build).
pub(super) const POW3: [f64; 6] = [1.0, 3.0, 9.0, 27.0, 81.0, 243.0];

/// PLO legal-anchor count (matches `n_legal_anchors_np` + PLO_ANCHOR_SPEC).
/// Rev 2 passes the engine's RAW legal chip deltas (min 0 in the short-shove
/// regime, which counts its single all-in atom); rev 1 the totals-derived
/// deltas — `RaiseWindow::anchor_min` / `max_d` either way.
pub(super) fn n_legal_anchors_plo(min_raise: f64, max_raise: f64, pot: f64, to_call: f64) -> usize {
    let mn = min_raise as i64;
    let mx = max_raise as i64;
    let pot_a = pot as i64;
    let tc = to_call as i64;
    let mr = mn.min(mx);
    let base = pot_a + tc;
    let mut chips = [0i64; 11];
    for k in 0..11 {
        let fr = 100i64 * k as i64;
        let c = tc + (fr * base + 500) / 1000;
        chips[k] = c.clamp(mr, mx);
    }
    let mut legal = [true; 11];
    for k in 1..11 {
        legal[k] = chips[k] > chips[k - 1];
    }
    if mn == 0 && mx > 0 {
        for k in 0..10 {
            legal[k] = false;
        }
        legal[10] = true;
    }
    legal.iter().filter(|&&x| x).count()
}

/// Everything the v7 tail reads besides the packed row itself (ENG-011: this
/// was a 20-argument list, two of them dead).
pub(super) struct TailCtx<'a> {
    pub(super) packed: &'a PackedObservation,
    pub(super) j: usize,
    pub(super) num_seats: usize,
    pub(super) hero: usize,
    pub(super) inv_bb: f64,
    pub(super) ante: u64,
    pub(super) starting: &'a [u64],
    pub(super) street: usize,
    pub(super) eff_per_seat: [f64; 8],
    pub(super) pot: f64,
    pub(super) btc: f64,
    pub(super) to_call: f64,
    pub(super) hero_stack: f64,
    pub(super) eff_to_call: f64,
    /// The raise window STK-2 / STK-5[2:4] describe (legal for rev 2, the
    /// totals-derived one for rev 1).
    pub(super) window: RaiseWindow,
    pub(super) hole: &'a [u8],
    pub(super) board_a: &'a [u8],
    pub(super) board_b: &'a [u8],
}

/// Encode dims 1020..1171 into `out` (pre-zeroed full OBS_DIM row).
pub(super) fn encode_v7_tail(ctx: &TailCtx<'_>, out: &mut [f32]) {
    let TailCtx {
        packed,
        j,
        num_seats,
        hero,
        inv_bb,
        ante,
        starting,
        street,
        ref eff_per_seat,
        pot,
        btc,
        to_call,
        hero_stack,
        eff_to_call,
        ref window,
        ..
    } = *ctx;
    let pot_denom = pot.max(1.0);
    let hero_commit = packed.core.total_commit[[j, hero]] as f64;
    let street_idx = street as i32;

    // ---- STK-1 ----
    {
        let mut max_cap = 0.0f64;
        let mut sum_cap = 0.0f64;
        let mut max_eff = 0.0f64;
        let mut any_pending = false;
        for s in 0..num_seats {
            if s == hero || packed.core.folded[[j, s]] || packed.core.all_in[[j, s]] {
                continue;
            }
            let acted = packed.acted_this_street[[j, s]];
            let sc_s = packed.core.street_commit[[j, s]] as f64;
            if acted && sc_s >= btc {
                continue;
            }
            any_pending = true;
            let eff_s = eff_per_seat[s];
            let owed = (btc - sc_s).max(0.0).min(eff_s);
            let cap = (eff_s - owed).max(0.0);
            if cap > max_cap {
                max_cap = cap;
            }
            sum_cap += cap;
            if eff_s > max_eff {
                max_eff = eff_s;
            }
        }
        if any_pending {
            out[STK1_OFF] = (max_cap / pot_denom).ln_1p() as f32;
            out[STK1_OFF + 1] = (sum_cap / pot_denom).ln_1p() as f32;
            out[STK1_OFF + 2] = (max_eff * inv_bb).ln_1p() as f32;
            out[STK1_OFF + 3] = if max_eff >= hero_stack { 1.0 } else { 0.0 };
        }
    }

    // ---- STK-2 ----
    // PRODUCTION BEHAVIOR CHANGE (review 2026-09-20 B1, dims 1024-1029 and
    // STK-5[2:4] = 1040-1041; obs_rev 2): min_d/max_d are the engine's LEGAL
    // raise deltas (== sizing_from_info's min/max_raise_chips) and the block is
    // zero when Raise is illegal. In rev 1 they are recovered from the
    // min_bet/max_bet totals, which are NOT capped by the actor's own stack and
    // ignore the short-shove lockout. The caller picks the window
    // (`legal_raise_window` / `legacy_raise_window` in bindings.rs); the
    // arithmetic below is shared.
    let RaiseWindow {
        legal: raise_legal,
        min_d,
        max_d,
        anchor_min,
    } = *window;
    let base = pot + to_call;
    if raise_legal {
        let base_safe = base.max(1.0);
        let eff_denom = hero_stack.max(1.0);
        out[STK2_OFF] = ((min_d - to_call) / base_safe).clamp(0.0, 1.0) as f32;
        out[STK2_OFF + 1] = ((max_d - to_call) / base_safe).clamp(0.0, 1.0) as f32;
        out[STK2_OFF + 2] = (min_d / eff_denom).clamp(0.0, 1.0) as f32;
        out[STK2_OFF + 3] = (max_d / eff_denom).clamp(0.0, 1.0) as f32;
        out[STK2_OFF + 4] = if max_d < to_call + base { 1.0 } else { 0.0 };
        out[STK2_OFF + 5] = (n_legal_anchors_plo(anchor_min, max_d, pot, to_call) as f64
            / ANCHOR_COUNT as f64) as f32;
    }

    // ---- STK-4 ----
    for k in 0..num_seats {
        let seat = (hero + k) % num_seats;
        if packed.core.folded[[j, seat]] {
            continue;
        }
        let commit_s = packed.core.total_commit[[j, seat]] as f64;
        let denom4 = commit_s + eff_per_seat[seat];
        if denom4 > 0.0 {
            out[STK4_OFF + k] = (commit_s / denom4) as f32;
        }
    }

    // ---- STK-5 ----
    out[STK5_OFF] = ((hero_stack - eff_to_call) / (pot + eff_to_call).max(1.0)).ln_1p() as f32;
    out[STK5_OFF + 1] = ((pot + eff_to_call) * inv_bb).ln_1p() as f32;
    if raise_legal {
        let tot2 = pot + 2.0 * max_d - to_call;
        out[STK5_OFF + 2] = ((hero_stack - max_d) / tot2.max(1.0)).ln_1p() as f32;
        out[STK5_OFF + 3] = (tot2 * inv_bb).ln_1p() as f32;
    }

    // ---- STK-6 ----
    {
        let spr_e = hero_stack / pot_denom;
        let r = 4 - street_idx;
        let x6 = 1.0 + 2.0 * spr_e;
        // Smallest k in 0..=6 with x6 <= 3^k == the count of powers strictly
        // below x6 (exact comparisons — see POW3).
        let btj = POW3.iter().filter(|&&p| x6 > p).count();
        out[STK6_OFF] = btj as f32;
        let gfrac = if r > 0 {
            (x6.powf(1.0 / r as f64) - 1.0) / 2.0
        } else {
            0.0
        };
        out[STK6_OFF + 1] = gfrac.clamp(0.0, 2.0) as f32;
    }

    // ---- STK-7 ----
    {
        let mut ceiling = pot;
        for s in 0..num_seats {
            if s == hero || packed.core.folded[[j, s]] {
                continue;
            }
            ceiling += eff_per_seat[s].min(hero_stack);
        }
        out[STK7_OFF] = (ceiling / pot_denom).ln_1p() as f32;
        if eff_to_call > 0.0 {
            out[STK7_OFF + 1] = (eff_to_call / (ceiling + eff_to_call)) as f32;
        }
    }

    // ---- STK-8 ----
    {
        let hero_after = hero_commit + eff_to_call;
        let mut sum_now = 0.0f64;
        let mut sum_after = 0.0f64;
        let mut dead = 0.0f64;
        for s in 0..num_seats {
            let cs = packed.core.total_commit[[j, s]] as f64;
            sum_now += cs.min(hero_commit);
            sum_after += cs.min(hero_after);
            if packed.core.folded[[j, s]] {
                dead += cs;
            }
        }
        out[STK8_OFF] = (sum_now / pot_denom) as f32;
        out[STK8_OFF + 1] = (sum_after / pot_denom) as f32;
        out[STK8_OFF + 2] = (dead / pot_denom) as f32;
    }

    // ---- STK-9 ----
    out[STK9_OFF] = (eff_to_call / hero_stack.max(1.0)) as f32;
    out[STK9_OFF + 1] = (eff_to_call / (hero_commit + hero_stack).max(1.0)) as f32;

    // ---- STK-10 ----
    // PRODUCTION BEHAVIOR CHANGE (review 2026-09-20 B6, dim 1052; SERVING
    // ONLY — training never passes an in_hand_mask): antes are summed over the
    // DEALT-IN seats. A sitting-out seat is pre-folded and posts no ante
    // (folded with nothing committed); it used to inflate pot_at_flop. With
    // every seat dealt in the sum is unchanged.
    {
        let ante_i = ante as i64;
        let mut pot_at_flop: i64 = 0;
        for s in 0..num_seats {
            if packed.core.folded[[j, s]] && packed.core.total_commit[[j, s]] == 0 {
                continue;
            }
            let st = starting.get(s).copied().unwrap_or(0) as i64;
            pot_at_flop += ante_i.min(st);
        }
        let paf = pot_at_flop as f64;
        out[STK10_OFF] = (ante_i as f64 * inv_bb) as f32;
        out[STK10_OFF + 1] = ((pot - paf).max(0.0) / paf.max(1.0)).ln_1p() as f32;
    }

    // ---- STK-11 ----
    for k in 1..num_seats {
        let seat = (hero + k) % num_seats;
        if packed.core.folded[[j, seat]] {
            continue;
        }
        let sc = packed.core.street_commit[[j, seat]] as f64;
        let owed = (btc - sc).max(0.0).min(eff_per_seat[seat]);
        out[STK11_OFF + k] = (owed / (pot_denom + owed)) as f32;
    }

    encode_board_dual(ctx, out);
}

/// The v7 BRD / DUAL blocks (dims 1061..1171) of the v7 tail.
pub(super) fn encode_board_dual(ctx: &TailCtx<'_>, out: &mut [f32]) {
    let TailCtx {
        packed,
        j,
        street,
        pot,
        eff_to_call,
        hero_stack,
        hole: hole_slice,
        board_a: ba_slice,
        board_b: bb_slice,
        ..
    } = *ctx;
    // Global visibility (hole + both boards)
    let mut vct = [0u8; 13];
    let mut vps = [0u8; 4];
    let mut visible_count = [[0u8; 4]; 13]; // [rank][suit]
    let mut board_all = [[false; 4]; 13];
    let mut hero_pres = [[false; 4]; 13];
    let mut hole_rank_counts = [0i32; 13];
    let mut hole_suit_counts = [0i32; 4];
    let mut hole_max_per_suit = [-1i32; 4];
    let mut hole_ranks_mask: u16 = 0;
    let mut n_vis = 0i32;

    for &c in hole_slice {
        if c < 52 {
            let r = (c >> 2) as usize;
            let s = (c & 3) as usize;
            visible_count[r][s] = 1;
            hero_pres[r][s] = true;
            hole_rank_counts[r] += 1;
            hole_suit_counts[s] += 1;
            hole_ranks_mask |= 1u16 << r;
            if r as i32 > hole_max_per_suit[s] {
                hole_max_per_suit[s] = r as i32;
            }
        }
    }
    for &c in ba_slice.iter().chain(bb_slice.iter()) {
        if c < 52 {
            let r = (c >> 2) as usize;
            let s = (c & 3) as usize;
            visible_count[r][s] = 1;
            board_all[r][s] = true;
        }
    }
    for r in 0..13 {
        for s in 0..4 {
            if visible_count[r][s] > 0 {
                vct[r] += 1;
                vps[s] += 1;
                n_vis += 1;
            }
        }
    }
    let unseen_deck = (52 - n_vis).max(1) as f64;
    let is_river = street == 3;
    let is_flop = street == 1;

    // Engine dims BRD-7 / BRD-12 / DUAL-2 from hero_board_v3
    let hb = [
        packed.hero_board_v3[[j, 0]],
        packed.hero_board_v3[[j, 1]],
        packed.hero_board_v3[[j, 2]],
        packed.hero_board_v3[[j, 3]],
        packed.hero_board_v3[[j, 4]],
        packed.hero_board_v3[[j, 5]],
        packed.hero_board_v3[[j, 6]],
        packed.hero_board_v3[[j, 7]],
    ];
    out[BRD7_OFF] = hb[0] as f32;
    out[BRD7_OFF + 1] = hb[1] as f32;
    out[BRD12_OFF] = (hb[2] as f64 / unseen_deck) as f32;
    out[BRD12_OFF + 1] = (hb[3] as f64 / unseen_deck) as f32;
    out[BRD12_OFF + 2] = (hb[4] as f64 / 10.0) as f32;
    out[BRD12_OFF + 3] = (hb[5] as f64 / 10.0) as f32;

    // board_draw_v3: [ds_a, ds_b, u_a, n_a, u_b, n_b, scoop]
    let bd = [
        packed.board_draw_v3[[j, 0]],
        packed.board_draw_v3[[j, 1]],
        packed.board_draw_v3[[j, 2]],
        packed.board_draw_v3[[j, 3]],
        packed.board_draw_v3[[j, 4]],
        packed.board_draw_v3[[j, 5]],
        packed.board_draw_v3[[j, 6]],
    ];

    for (bi, board) in [ba_slice, bb_slice].into_iter().enumerate() {
        let mut board_rank_counts = [0i32; 13];
        let mut board_suit_counts = [0i32; 4];
        let mut board_ranks_per_suit = [0u16; 4];
        let mut nboard = 0usize;
        let mut ranks_buf = [0i32; 5];
        for &c in board {
            if c < 52 {
                let r = (c >> 2) as i32;
                let s = (c & 3) as usize;
                board_rank_counts[r as usize] += 1;
                board_suit_counts[s] += 1;
                board_ranks_per_suit[s] |= 1u16 << r;
                ranks_buf[nboard] = r;
                nboard += 1;
            }
        }
        let sorted_ranks = &mut ranks_buf[..nboard];
        sorted_ranks.sort_by(|a, b| b.cmp(a));
        let mut board_ranks_mask: u16 = 0;
        for r in 0..13 {
            if board_rank_counts[r] > 0 {
                board_ranks_mask |= 1u16 << r;
            }
        }

        // BRD-1 rank ladder
        let o1 = BRD1_OFF + bi * 5;
        for (i, &rank) in sorted_ranks.iter().take(5).enumerate() {
            out[o1 + i] = ((rank + 1) as f64 / 13.0) as f32;
        }

        // BRD-2 suit census
        let o2 = BRD2_OFF + bi * 6;
        for s in 0..4 {
            if board_suit_counts[s] == 2 {
                out[o2 + s] = 1.0;
            }
        }
        if board_suit_counts.contains(&4) {
            out[o2 + 4] = 1.0;
        }
        if board_suit_counts.contains(&5) {
            out[o2 + 5] = 1.0;
        }

        // BRD-4 arrival volatility
        if !is_river && nboard > 0 {
            let o4 = BRD4_OFF + bi * 3;
            let mut pair_outs = 0i32;
            for r in 0..13 {
                if board_rank_counts[r] > 0 {
                    pair_outs += 4 - vct[r] as i32;
                }
            }
            let mut flush_adv = 0i32;
            for s in 0..4 {
                if board_suit_counts[s] == 2 || board_suit_counts[s] == 3 {
                    flush_adv += 13 - vps[s] as i32;
                }
            }
            // straight_adv: ranks not on board that complete a 2-rank board window
            let mut straight_adv = 0i32;
            for r in 0..13u16 {
                if board_ranks_mask & (1u16 << r) != 0 {
                    continue;
                }
                let ur = 4 - vct[r as usize] as i32;
                if ur <= 0 {
                    continue;
                }
                for &w in &crate::hand_eval::WINDOW_BITS {
                    if w & (1u16 << r) != 0 && (w & board_ranks_mask).count_ones() == 2 {
                        straight_adv += ur;
                        break;
                    }
                }
            }
            out[o4] = (pair_outs as f64 / unseen_deck) as f32;
            out[o4 + 1] = (flush_adv as f64 / unseen_deck) as f32;
            out[o4 + 2] = (straight_adv as f64 / unseen_deck) as f32;
        }

        // BRD-5: danger_flush/pair pure; danger_straight from board_draw
        if !is_river {
            let o5 = BRD5_OFF + bi * 3;
            let mut danger_flush = 0i32;
            for s in 0..4 {
                if board_suit_counts[s] == 2 && hole_suit_counts[s] < 2 {
                    danger_flush += 13 - vps[s] as i32;
                }
            }
            let mut danger_pair = 0i32;
            for r in 0..13 {
                if board_rank_counts[r] > 0 && hole_rank_counts[r] == 0 {
                    danger_pair += 4 - vct[r] as i32;
                }
            }
            let danger_straight = bd[bi] as f64;
            out[o5] = (danger_flush as f64 / unseen_deck) as f32;
            out[o5 + 1] = (danger_pair as f64 / unseen_deck) as f32;
            out[o5 + 2] = (danger_straight / unseen_deck) as f32;
        }

        // BRD-6 from board_draw
        if !is_river {
            let o6 = BRD6_OFF + bi * 2;
            out[o6] = bd[2 + bi * 2] as f32;
            out[o6 + 1] = bd[3 + bi * 2] as f32;
        }

        // BRD-8 fd rank quality
        {
            let o8 = BRD8_OFF + bi * 2;
            let mut draw_suit = -1i32;
            let mut best_max = -1i32;
            for s in 0..4 {
                if hole_suit_counts[s] >= 2
                    && board_suit_counts[s] == 2
                    && hole_max_per_suit[s] > best_max
                {
                    best_max = hole_max_per_suit[s];
                    draw_suit = s as i32;
                }
            }
            if draw_suit >= 0 {
                let h1 = hole_max_per_suit[draw_suit as usize];
                out[o8] = ((h1 + 1) as f64 / 13.0) as f32;
                let mut cnt = 0i32;
                for r in (h1 + 1)..13 {
                    if visible_count[r as usize][draw_suit as usize] == 0 {
                        cnt += 1;
                    }
                }
                out[o8 + 1] = cnt as f32;
            }
        }

        // BRD-9 backdoor (flop only)
        if is_flop {
            let o9 = BRD9_OFF + bi * 2;
            let mut bdfd = 0i32;
            for s in 0..4 {
                if hole_suit_counts[s] >= 2 && board_suit_counts[s] == 1 {
                    bdfd += 1;
                }
            }
            let mut bdstr = 0i32;
            for &w in &crate::hand_eval::WINDOW_BITS {
                let n_h = (w & hole_ranks_mask).count_ones();
                let n_l = (w & !board_ranks_mask).count_ones();
                let missing_both = (w & !board_ranks_mask & !hole_ranks_mask).count_ones();
                if missing_both == 2 && n_h >= 2 && n_l <= 4 {
                    bdstr += 1;
                }
            }
            out[o9] = bdfd as f32;
            out[o9 + 1] = bdstr as f32;
        }

        // BRD-10 future nut-flush blocker
        if !is_river {
            let o10 = BRD10_OFF + bi;
            let mut cnt = 0i32;
            for s in 0..4 {
                if board_suit_counts[s] != 2 {
                    continue;
                }
                for r in (0..13usize).rev() {
                    if board_all[r][s] {
                        continue;
                    }
                    if hero_pres[r][s] {
                        cnt += 1;
                    }
                    break;
                }
            }
            out[o10] = cnt as f32;
        }

        // BRD-11 turn/river card identity
        {
            let mut cards_buf = [0u8; 5];
            let mut n_cards = 0;
            for &c in board.iter().filter(|&&c| c < 52) {
                cards_buf[n_cards] = c;
                n_cards += 1;
            }
            let cards = &cards_buf[..n_cards];
            if cards.len() >= 4 {
                let c = cards[3];
                let o11t = BRD11_OFF + bi * 10;
                out[o11t] = (((c >> 2) + 1) as f64 / 13.0) as f32;
                out[o11t + 1 + (c & 3) as usize] = 1.0;
            }
            if cards.len() >= 5 {
                let c = cards[4];
                let o11r = BRD11_OFF + bi * 10 + 5;
                out[o11r] = (((c >> 2) + 1) as f64 / 13.0) as f32;
                out[o11r + 1 + (c & 3) as usize] = 1.0;
            }
        }

        // BRD-13 board nut ceiling
        if nboard > 0 {
            let o13 = BRD13_OFF + bi * 2;
            let mut sf_possible = false;
            for s in 0..4 {
                let ranks_s = board_ranks_per_suit[s];
                if ranks_s.count_ones() >= 3 {
                    for &w in &crate::hand_eval::WINDOW_BITS {
                        if (w & ranks_s).count_ones() >= 3 {
                            sf_possible = true;
                            break;
                        }
                    }
                }
                if sf_possible {
                    break;
                }
            }
            let paired = board_rank_counts.iter().any(|&c| c >= 2);
            let flush_poss = board_suit_counts.iter().any(|&c| c >= 3);
            let straight_poss = crate::hand_eval::WINDOW_BITS
                .iter()
                .any(|&w| (w & board_ranks_mask).count_ones() >= 3);
            let ceil = if sf_possible {
                8
            } else if paired {
                7
            } else if flush_poss {
                5
            } else if straight_poss {
                4
            } else {
                3
            };
            out[o13] = if sf_possible { 1.0 } else { 0.0 };
            out[o13 + 1] = (ceil as f64 / 8.0) as f32;
        }
    }

    // ---- DUAL-1 ----
    if eff_to_call > 0.0 {
        out[DUAL1_OFF] = (eff_to_call / (0.5 * pot + eff_to_call)) as f32;
        out[DUAL1_OFF + 1] = (eff_to_call / (0.25 * pot + eff_to_call)) as f32;
    }

    // ---- DUAL-3 from per_board_outcome ----
    let ahead_a = packed.per_board_outcome[[j, 0]] as f64;
    let tie_a = packed.per_board_outcome[[j, 1]] as f64;
    let behind_a = packed.per_board_outcome[[j, 2]] as f64;
    let tie_b = packed.per_board_outcome[[j, 4]] as f64;
    let behind_b = packed.per_board_outcome[[j, 5]] as f64;
    let active = (ahead_a + tie_a + behind_a) > 0.5;
    if active {
        let nut_or_chop_a = behind_a == 0.0;
        let nut_or_chop_b = behind_b == 0.0;
        out[DUAL3_OFF] = if nut_or_chop_a && tie_a == 0.0 {
            1.0
        } else {
            0.0
        };
        out[DUAL3_OFF + 1] = if nut_or_chop_a { 1.0 } else { 0.0 };
        out[DUAL3_OFF + 2] = if nut_or_chop_b && tie_b == 0.0 {
            1.0
        } else {
            0.0
        };
        out[DUAL3_OFF + 3] = if nut_or_chop_b { 1.0 } else { 0.0 };
        out[DUAL3_OFF + 4] = if nut_or_chop_a && nut_or_chop_b {
            1.0
        } else {
            0.0
        };
        out[DUAL3_OFF + 5] = if nut_or_chop_a || nut_or_chop_b {
            1.0
        } else {
            0.0
        };
    }

    // ---- DUAL-4 share_bounds ----
    if active {
        let g_min = packed.share_bounds[[j, 0]] as f64;
        let g_max = packed.share_bounds[[j, 1]] as f64;
        out[DUAL4_OFF] = g_min as f32;
        out[DUAL4_OFF + 1] = g_max as f32;
        out[DUAL4_OFF + 2] = if g_min >= 0.5 { 1.0 } else { 0.0 };
        let rng = g_max - g_min;
        if rng > 0.0 {
            if eff_to_call > 0.0 {
                out[DUAL4_OFF + 3] = (eff_to_call / (pot * rng + eff_to_call)) as f32;
            }
            out[DUAL4_OFF + 4] = (hero_stack / (pot.max(1.0) * rng)).ln_1p() as f32;
        }
    }

    // ---- DUAL-2 masks ----
    for i in 0..5 {
        if (hb[6] >> i) & 1 != 0 {
            out[DUAL2_OFF + i] = 1.0;
        }
        if (hb[7] >> i) & 1 != 0 {
            out[DUAL2_OFF + 5 + i] = 1.0;
        }
    }

    // ---- DUAL-5 ----
    {
        let mut ba_suit = [0i32; 4];
        let mut bb_suit = [0i32; 4];
        for &c in ba_slice {
            if c < 52 {
                ba_suit[(c & 3) as usize] += 1;
            }
        }
        for &c in bb_slice {
            if c < 52 {
                bb_suit[(c & 3) as usize] += 1;
            }
        }
        for s in 0..4 {
            if ba_suit[s] >= 2 && bb_suit[s] >= 2 {
                out[DUAL5_OFF + s] = 1.0;
            }
            if ba_suit[s] >= 3 && bb_suit[s] >= 3 {
                out[DUAL5_OFF + 4 + s] = 1.0;
            }
        }
        out[DUAL5_OFF + 8] = (bd[6] as f64 / 78.0) as f32;
    }
}

/// Where the full layout puts the core blocks that move between layouts.
pub(super) const FULL_CORE_AT: CoreOffsets = CoreOffsets {
    history: obs_layout::HISTORY_OFF,
    seat_exists: obs_layout::SEAT_EXISTS_OFF,
    total_commit: obs_layout::TOTAL_COMMIT_OFF,
    street_commit: obs_layout::STREET_COMMIT_OFF,
    hero_btn: obs_layout::HERO_BTN_DIST_OFF,
};

/// Encode one env's observation into `out` (length OBS_DIM, pre-zeroed). A
/// bit-exact per-env port of the scalar `encode_observation`
/// (python/plo5bp/encoding.py:652) — the ground-truth reference the numpy
/// batch encoder is validated against. Reads row `j` of the packed arrays.
/// Terminal rows (actor < 0) are left all-zero, matching the scalar early
/// return.
///
/// Bit-exactness discipline: all scalar arithmetic is done in f64 and cast to
/// f32 ONLY at the store (`(x as f64 * inv_bb) as f32`), matching the numpy
/// path which keeps `inv_bb` in f64 and casts on assignment. Hero rotation
/// uses non-negative modulo to match numpy/Python `%`.
///
/// Scope of "bit-exact" (ENG-005): everything here is integer or IEEE `+ - * /`
/// arithmetic — identical on every machine — except the `ln_1p` / `powf` dims
/// (EFF_PRICE 3-4, SPR_LOG, STK-1/5/6[1]/7/10[1], DUAL-4[4]), which call the
/// platform maths library (MSVC UCRT here, glibc on the pod / server, maybe SVML
/// in numpy) and can differ in the last bit ACROSS machines. The golden digests
/// in bindings/golden_tests.rs keep those dims in per-target sections.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_obs_row(
    packed: &PackedObservation,
    j: usize,
    num_seats: usize,
    cat_a: u8,
    cat_b: u8,
    inv_bb: f64,
    bb: u64,
    ante: u64,
    starting: &[u64],
    obs_rev: u8,
    out: &mut [f32],
) {
    use obs_layout::*;

    let Some(core_row) = encode_core(
        &packed.core,
        j,
        &FULL_CORE_AT,
        num_seats,
        inv_bb,
        bb,
        starting,
        obs_rev,
        out,
    ) else {
        return; // terminal env -> all-zero row
    };
    let CoreRow {
        hero,
        hole: hole_slice,
        board_a: ba_slice,
        board_b: bb_slice,
        street,
        eff_per_seat,
        pot,
        btc,
        legal_window,
    } = core_row;
    let ns_i = num_seats as i64;
    let rel = |x: i64| -> usize { (x - hero as i64).rem_euclid(ns_i) as usize };
    let legacy = obs_rev == OBS_REV_LEGACY;

    // --- Relative position one-hot: actor is always slot 0 (hero == actor). ---
    out[REL_POS_OFF] = 1.0;

    // --- SPR per seat (hero-rotated), clip [0, 4]. ---
    let pot_safe = pot.max(1.0);
    for k in 0..num_seats {
        let seat = (hero + k) % num_seats;
        let spr = eff_per_seat[seat] / pot_safe;
        #[allow(clippy::manual_clamp)] // max/min, not clamp: NaN / -0.0 bits unchanged
        let spr = spr.max(0.0).min(4.0);
        out[SPR_OFF + k] = spr as f32;
    }

    // --- Pot odds + bet-faced-as-fraction-of-pot. ---
    let hero_street_commit = packed.core.street_commit[[j, hero]] as f64;
    let to_call = (btc - hero_street_commit).max(0.0);
    if to_call > 0.0 {
        out[POT_ODDS_OFF] = (to_call / (pot + to_call)) as f32;
        let pot_before_bet = (pot - to_call).max(1.0);
        out[BET_PCT_POT_OFF] = (to_call / pot_before_bet).min(4.0) as f32;
    }

    // --- Hand-category one-hots. ---
    if (cat_a as usize) < NUM_CATEGORIES {
        out[CAT_A_OFF + cat_a as usize] = 1.0;
    }
    if (cat_b as usize) < NUM_CATEGORIES {
        out[CAT_B_OFF + cat_b as usize] = 1.0;
    }

    // --- Hole-derived summaries reused by feature blocks. ---
    let mut hole_suit_count = [0u8; 4];
    let mut hole_rank_mask: u16 = 0;
    let mut hole_rank_counts = [0u8; 13];
    for &c in hole_slice {
        if c < 52 {
            hole_suit_count[(c & 3) as usize] += 1;
            hole_rank_mask |= 1u16 << (c >> 2);
            hole_rank_counts[(c >> 2) as usize] += 1;
        }
    }

    // --- Draw flags (per board). ---
    let (fa, sa) = draw_flags_one_board(&hole_suit_count, hole_rank_mask, ba_slice, obs_rev);
    let (fb, sb) = draw_flags_one_board(&hole_suit_count, hole_rank_mask, bb_slice, obs_rev);
    out[DRAW_A_OFF] = fa;
    out[DRAW_A_OFF + 1] = sa;
    out[DRAW_B_OFF] = fb;
    out[DRAW_B_OFF + 1] = sb;

    // --- Pair-with-board counts + board pair structure. ---
    let mut counts_a = [0f32; 5];
    let mut struct_a = [0f32; 4];
    let mut counts_b = [0f32; 5];
    let mut struct_b = [0f32; 4];
    pair_features_one_board(&hole_rank_counts, ba_slice, &mut counts_a, &mut struct_a);
    pair_features_one_board(&hole_rank_counts, bb_slice, &mut counts_b, &mut struct_b);
    out[PAIR_COUNT_A_OFF..PAIR_COUNT_A_OFF + 5].copy_from_slice(&counts_a);
    out[PAIR_COUNT_B_OFF..PAIR_COUNT_B_OFF + 5].copy_from_slice(&counts_b);
    out[BOARD_STRUCT_A_OFF..BOARD_STRUCT_A_OFF + 4].copy_from_slice(&struct_a);
    out[BOARD_STRUCT_B_OFF..BOARD_STRUCT_B_OFF + 4].copy_from_slice(&struct_b);

    // --- Hero rank histogram (board-agnostic). ---
    for &c in hole_slice {
        if c < 52 {
            out[HERO_RANK_HIST_OFF + (c >> 2) as usize] += 1.0;
        }
    }

    // --- Straight / flush / SF block: needs global (hole+A+B) visibility. ---
    let mut seen_per_suit = [0u16; 4];
    for slice in [hole_slice, ba_slice, bb_slice] {
        for &c in slice {
            if c < 52 {
                seen_per_suit[(c & 3) as usize] |= 1u16 << (c >> 2);
            }
        }
    }
    let mut vct = [0u8; 13];
    for r in 0..13 {
        let mut cnt = 0u8;
        for s in seen_per_suit.iter() {
            if (s >> r) & 1 == 1 {
                cnt += 1;
            }
        }
        vct[r] = cnt;
    }
    let mut unseen_suit = [0u16; 4];
    let mut visible_per_suit = [0u8; 4];
    for s in 0..4 {
        unseen_suit[s] = !seen_per_suit[s] & 0x1FFF;
        visible_per_suit[s] = seen_per_suit[s].count_ones() as u8;
    }
    let (h_rm, h_rs, h_sc, h_mps) = sf_derive_card_state(hole_slice);
    sf_compute_board(
        h_rm,
        &h_rs,
        &h_sc,
        &h_mps,
        ba_slice,
        &vct,
        &unseen_suit,
        &visible_per_suit,
        &mut out[FLUSH_NUT_DIST_A_OFF..FLUSH_NUT_DIST_A_OFF + 38],
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
        &mut out[FLUSH_NUT_DIST_B_OFF..FLUSH_NUT_DIST_B_OFF + 38],
    );

    // --- Last aggressor (hero-relative one-hot). ---
    let la = packed.last_aggressor[j];
    if la >= 0 && (la as usize) < num_seats {
        out[LAST_AGGRESSOR_OFF + rel(la as i64)] = 1.0;
    }

    // --- Cross-board interactions. ---
    let mut ba_rank_mask: u16 = 0;
    let mut bb_rank_mask: u16 = 0;
    let mut ba_suit = [0u8; 4];
    let mut bb_suit = [0u8; 4];
    for &c in ba_slice {
        if c < 52 {
            ba_rank_mask |= 1u16 << (c >> 2);
            ba_suit[(c & 3) as usize] += 1;
        }
    }
    for &c in bb_slice {
        if c < 52 {
            bb_rank_mask |= 1u16 << (c >> 2);
            bb_suit[(c & 3) as usize] += 1;
        }
    }
    let shared = ba_rank_mask & bb_rank_mask;
    for r in 0..13 {
        if (shared >> r) & 1 == 1 {
            out[SHARED_RANKS_OFF + r] = 1.0;
        }
    }
    for s in 0..4 {
        if hole_suit_count[s] < 2 {
            continue;
        }
        let a3 = ba_suit[s] >= 3;
        let b3 = bb_suit[s] >= 3;
        let a2 = ba_suit[s] == 2;
        let b2 = bb_suit[s] == 2;
        if a3 && b3 {
            out[FLUSH_MADE_BOTH_OFF + s] = 1.0;
        } else if a2 && b2 {
            out[FLUSH_DRAW_BOTH_OFF + s] = 1.0;
        } else if (a3 && b2) || (a2 && b3) {
            out[FLUSH_MIXED_OFF + s] = 1.0;
        }
    }
    let boards_visible = ba_rank_mask != 0 && bb_rank_mask != 0;
    let (cm, cd, cx) =
        cross_board_straight_per_env(hole_rank_mask, ba_rank_mask, bb_rank_mask, boards_visible);
    out[STRAIGHT_MADE_BOTH_OFF] = cm;
    out[STRAIGHT_DRAW_BOTH_OFF] = cd;
    out[STRAIGHT_MIXED_OFF] = cx;

    // --- Opp-outcome fractions (already f32; live rows only). ---
    for t in 0..OPP_OUTCOME_DIM {
        out[OPP_OUTCOME_OFF + t] = packed.opp_outcome_fractions[[j, t]];
    }

    // ===== obs v2 tail (V5_DESIGN.md §3.2, dims 991..1020) =====
    // Per-board hero ahead/tie/behind + win-one/tie-both — already f32 from the
    // fused MC pass (packed by pack_observation_indexed); all-zero preflop/terminal.
    for m in 0..8 {
        out[PER_BOARD_OUTCOME_OFF + m] = packed.per_board_outcome[[j, m]];
    }
    // Unconditional blockers-to-nuts per board (4 dims each). Rev 2 also hands
    // over the OTHER board — its face-up cards are not holdable (review
    // 2026-09-20 B5); rev 1 keeps the board-local flush dims.
    let (other_a, other_b): (&[u8], &[u8]) = if legacy {
        (&[], &[])
    } else {
        (bb_slice, ba_slice)
    };
    blocker_features_one_board(
        hole_slice,
        ba_slice,
        other_a,
        &mut out[BLOCKER_A_OFF..BLOCKER_A_OFF + 4],
    );
    blocker_features_one_board(
        hole_slice,
        bb_slice,
        other_b,
        &mut out[BLOCKER_B_OFF..BLOCKER_B_OFF + 4],
    );
    // Effective price: to_call capped by hero's EFFECTIVE remaining stack, plus
    // commitment fraction and log1p money companions. f64 throughout, cast on store.
    let hero_stack = eff_per_seat[hero];
    let eff_to_call = to_call.min(hero_stack);
    if eff_to_call > 0.0 {
        out[EFF_PRICE_OFF] = (eff_to_call / (pot + eff_to_call)) as f32;
    }
    if to_call > 0.0 && to_call >= hero_stack {
        out[EFF_PRICE_OFF + 1] = 1.0;
    }
    let hero_commit = packed.core.total_commit[[j, hero]] as f64;
    let commit_denom = hero_commit + hero_stack;
    if commit_denom > 0.0 {
        out[EFF_PRICE_OFF + 2] = (hero_commit / commit_denom) as f32;
    }
    out[EFF_PRICE_OFF + 3] = (eff_to_call * inv_bb).ln_1p() as f32;
    out[EFF_PRICE_OFF + 4] = (pot * inv_bb).ln_1p() as f32;
    // log1p effective SPR, unclipped (hero-rotated).
    for k in 0..num_seats {
        let seat = (hero + k) % num_seats;
        out[SPR_LOG_OFF + k] = (eff_per_seat[seat] / pot_safe).ln_1p() as f32;
    }

    // ===== v7 batch-2 tail (dims 1020..1171) =====
    // The raise window STK-2 / STK-5[2:4] describe: legal (rev 2) or the
    // totals-derived one (rev 1).
    let window = if legacy {
        legacy_raise_window(
            packed.core.min_bet[j],
            packed.core.max_bet[j],
            hero_street_commit,
            to_call,
        )
    } else {
        legal_window
    };
    let ctx = TailCtx {
        packed,
        j,
        num_seats,
        hero,
        inv_bb,
        ante,
        starting,
        street,
        eff_per_seat,
        pot,
        btc,
        to_call,
        hero_stack,
        eff_to_call,
        window,
        hole: hole_slice,
        board_a: ba_slice,
        board_b: bb_slice,
    };
    encode_v7_tail(&ctx, out);
}

/// Unconditional blockers-to-nuts for ONE board (obs v2 P3, 4 dims). Bit-exact
/// port of `_blocker_features` (python/plo5bp/encoding.py): flush-suit top-card
/// blocker + top-3 held, nut-straight window blockers, top board-pair blocker.
/// `out` is a 4-wide pre-zeroed slice. Divisions are done in f64 then cast to
/// f32 (matching numpy's `held / 3.0` → f32-array assignment).
/// `other_board` is the OTHER board's row: its face-up cards are in nobody's
/// hand, so the flush dims skip them when ranking the "missing" suit cards.
pub(super) fn blocker_features_one_board(
    hole: &[u8],
    board: &[u8],
    other_board: &[u8],
    out: &mut [f32],
) {
    let mut board_rank_counts = [0i32; 13];
    let mut board_suit_counts = [0i32; 4];
    let mut faceup_suit_ranks = [0u16; 4]; // rank bitmask per suit, EITHER board
    let mut nboard = 0usize;
    for &c in board {
        if c < 52 {
            nboard += 1;
            board_rank_counts[(c >> 2) as usize] += 1;
            board_suit_counts[(c & 3) as usize] += 1;
            faceup_suit_ranks[(c & 3) as usize] |= 1u16 << (c >> 2);
        }
    }
    if nboard < 3 {
        return;
    }
    // PRODUCTION BEHAVIOR CHANGE (review 2026-09-20 B5, dims 999/1000 and
    // 1003/1004): "missing" excludes cards visible on the OTHER board too —
    // they used to count as holdable, so hero's Qs read 0 on a Ks-high spade
    // board although the As lay face-up on the other board.
    for &c in other_board {
        if c < 52 {
            faceup_suit_ranks[(c & 3) as usize] |= 1u16 << (c >> 2);
        }
    }
    let mut hero_rank_counts = [0i32; 13];
    let mut hero_cards: u64 = 0;
    for &c in hole {
        if c < 52 {
            hero_rank_counts[(c >> 2) as usize] += 1;
            hero_cards |= 1u64 << c;
        }
    }

    // Flush blockers: first suit with >= 3 board cards (two can't coexist on 5).
    for s in 0..4usize {
        if board_suit_counts[s] >= 3 {
            let mut missing_buf = [0usize; 13];
            let mut n_missing = 0;
            for r in (0..13usize).rev() {
                if faceup_suit_ranks[s] & (1u16 << r) == 0 {
                    missing_buf[n_missing] = r;
                    n_missing += 1;
                }
            }
            let missing = &missing_buf[..n_missing];
            if let Some(&top) = missing.first() {
                if hero_cards & (1u64 << (top * 4 + s)) != 0 {
                    out[0] = 1.0;
                }
            }
            let held = missing
                .iter()
                .take(3)
                .filter(|&&r| hero_cards & (1u64 << (r * 4 + s)) != 0)
                .count();
            out[1] = (held as f64 / 3.0) as f32;
            break;
        }
    }

    // Nut-straight blockers: highest qualifying window (broadway-first scan).
    // Windows match _STRAIGHT_WINDOWS: slot 0 = wheel, slot 9 = broadway —
    // `hand_eval::WINDOW_BITS`, the one copy of the table (ENG-016).
    use crate::hand_eval::WINDOW_BITS as STRAIGHT_WINDOWS;
    let mut board_rank_set: u16 = 0;
    for r in 0..13usize {
        if board_rank_counts[r] > 0 {
            board_rank_set |= 1u16 << r;
        }
    }
    for wi in (0..10usize).rev() {
        let w = STRAIGHT_WINDOWS[wi];
        if (w & board_rank_set).count_ones() >= 3 {
            let missing = w & !board_rank_set;
            let mut blockers = 0i32;
            for r in 0..13usize {
                if missing & (1u16 << r) != 0 {
                    blockers += hero_rank_counts[r];
                }
            }
            out[2] = (blockers.min(4) as f64 / 4.0) as f32;
            break;
        }
    }

    // Board-pair blockers: highest paired rank.
    for r in (0..13usize).rev() {
        if board_rank_counts[r] >= 2 {
            out[3] = (hero_rank_counts[r].min(2) as f64 / 2.0) as f32;
            break;
        }
    }
}
