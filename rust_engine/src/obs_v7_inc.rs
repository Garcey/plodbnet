// v7 batch-2 obs tail (dims 1020..1171) for the Rust encoder.
// Bit-exact with python/plo5bp/encoding.py `_encode_stack_v3` /
// `_encode_board_v3` / `_encode_dual_v3`.

use obs_layout::*;

/// PLO legal-anchor count (matches `n_legal_anchors_np` + PLO_ANCHOR_SPEC).
fn n_legal_anchors_plo(min_d: f64, max_d: f64, pot: f64, to_call: f64) -> usize {
    let mn = min_d as i64;
    let mx = max_d as i64;
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

/// Encode dims 1020..1171 into `out` (pre-zeroed full OBS_DIM row).
#[allow(clippy::too_many_arguments)]
fn encode_v7_tail(
    packed: &PackedObservation,
    j: usize,
    num_seats: usize,
    hero: usize,
    inv_bb: f64,
    ante: u64,
    starting: &[u64],
    street: usize,
    eff_per_seat: &[f64; 8],
    pot: f64,
    btc: f64,
    to_call: f64,
    _pot_safe: f64,
    hero_stack: f64,
    eff_to_call: f64,
    hole_slice: &[u8],
    ba_slice: &[u8],
    bb_slice: &[u8],
    out: &mut [f32],
) {
    let pot_denom = pot.max(1.0);
    let hero_sc = packed.street_commit[[j, hero]] as f64;
    let hero_commit = packed.total_commit[[j, hero]] as f64;
    let min_bet = packed.min_bet[j] as f64;
    let max_bet = packed.max_bet[j] as f64;
    let street_idx = street as i32;

    // ---- STK-1 ----
    {
        let mut max_cap = 0.0f64;
        let mut sum_cap = 0.0f64;
        let mut max_eff = 0.0f64;
        let mut any_pending = false;
        for s in 0..num_seats {
            if s == hero || packed.folded[[j, s]] || packed.all_in[[j, s]] {
                continue;
            }
            let acted = packed.acted_this_street[[j, s]];
            let sc_s = packed.street_commit[[j, s]] as f64;
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
    let min_d = min_bet - hero_sc;
    let max_d = max_bet - hero_sc;
    let base = pot + to_call;
    let raise_legal = max_d > to_call;
    if raise_legal {
        let base_safe = base.max(1.0);
        let eff_denom = hero_stack.max(1.0);
        out[STK2_OFF] = ((min_d - to_call) / base_safe).clamp(0.0, 1.0) as f32;
        out[STK2_OFF + 1] = ((max_d - to_call) / base_safe).clamp(0.0, 1.0) as f32;
        out[STK2_OFF + 2] = (min_d / eff_denom).clamp(0.0, 1.0) as f32;
        out[STK2_OFF + 3] = (max_d / eff_denom).clamp(0.0, 1.0) as f32;
        out[STK2_OFF + 4] = if max_d < to_call + base { 1.0 } else { 0.0 };
        out[STK2_OFF + 5] = (n_legal_anchors_plo(min_d, max_d, pot, to_call) as f64
            / ANCHOR_COUNT as f64) as f32;
    }

    // ---- STK-4 ----
    for k in 0..num_seats {
        let seat = (hero + k) % num_seats;
        if packed.folded[[j, seat]] {
            continue;
        }
        let commit_s = packed.total_commit[[j, seat]] as f64;
        let denom4 = commit_s + eff_per_seat[seat];
        if denom4 > 0.0 {
            out[STK4_OFF + k] = (commit_s / denom4) as f32;
        }
    }

    // ---- STK-5 ----
    out[STK5_OFF] =
        ((hero_stack - eff_to_call) / (pot + eff_to_call).max(1.0)).ln_1p() as f32;
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
        let btj = if x6 > 0.0 {
            (x6.ln() / 3.0f64.ln()).ceil()
        } else {
            0.0
        };
        out[STK6_OFF] = btj.clamp(0.0, 6.0) as f32;
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
            if s == hero || packed.folded[[j, s]] {
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
            let cs = packed.total_commit[[j, s]] as f64;
            sum_now += cs.min(hero_commit);
            sum_after += cs.min(hero_after);
            if packed.folded[[j, s]] {
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
    {
        let ante_i = ante as i64;
        let mut pot_at_flop: i64 = 0;
        for s in 0..num_seats {
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
        if packed.folded[[j, seat]] {
            continue;
        }
        let sc = packed.street_commit[[j, seat]] as f64;
        let owed = (btc - sc).max(0.0).min(eff_per_seat[seat]);
        out[STK11_OFF + k] = (owed / (pot_denom + owed)) as f32;
    }

    encode_board_dual(
        packed,
        j,
        street,
        pot,
        to_call,
        eff_to_call,
        hero_stack,
        hole_slice,
        ba_slice,
        bb_slice,
        out,
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_board_dual(
    packed: &PackedObservation,
    j: usize,
    street: usize,
    pot: f64,
    _to_call: f64,
    eff_to_call: f64,
    hero_stack: f64,
    hole_slice: &[u8],
    ba_slice: &[u8],
    bb_slice: &[u8],
    out: &mut [f32],
) {
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
        let mut sorted_ranks: Vec<i32> = Vec::with_capacity(5);
        for &c in board {
            if c < 52 {
                nboard += 1;
                let r = (c >> 2) as i32;
                let s = (c & 3) as usize;
                board_rank_counts[r as usize] += 1;
                board_suit_counts[s] += 1;
                board_ranks_per_suit[s] |= 1u16 << r;
                sorted_ranks.push(r);
            }
        }
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
        if board_suit_counts.iter().any(|&c| c == 4) {
            out[o2 + 4] = 1.0;
        }
        if board_suit_counts.iter().any(|&c| c == 5) {
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
                    if w & (1u16 << r) != 0
                        && (w & board_ranks_mask).count_ones() == 2
                    {
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
                if hole_suit_counts[s] >= 2 && board_suit_counts[s] == 2 {
                    if hole_max_per_suit[s] > best_max {
                        best_max = hole_max_per_suit[s];
                        draw_suit = s as i32;
                    }
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
            let cards: Vec<u8> = board.iter().copied().filter(|&c| c < 52).collect();
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
        out[DUAL3_OFF] = if nut_or_chop_a && tie_a == 0.0 { 1.0 } else { 0.0 };
        out[DUAL3_OFF + 1] = if nut_or_chop_a { 1.0 } else { 0.0 };
        out[DUAL3_OFF + 2] = if nut_or_chop_b && tie_b == 0.0 { 1.0 } else { 0.0 };
        out[DUAL3_OFF + 3] = if nut_or_chop_b { 1.0 } else { 0.0 };
        out[DUAL3_OFF + 4] = if nut_or_chop_a && nut_or_chop_b { 1.0 } else { 0.0 };
        out[DUAL3_OFF + 5] = if nut_or_chop_a || nut_or_chop_b { 1.0 } else { 0.0 };
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
                out[DUAL4_OFF + 3] =
                    (eff_to_call / (pot * rng + eff_to_call)) as f32;
            }
            out[DUAL4_OFF + 4] =
                (hero_stack / (pot.max(1.0) * rng)).ln_1p() as f32;
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
