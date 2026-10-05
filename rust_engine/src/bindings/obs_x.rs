//! The obs-X tail of the full layout (2026-10-05): optional dims after the 1171 that the
//! vSix7 lineage was trained with (the training worktree's 2026-09-29 experiment, whose engine
//! computed them in its batched encoder). The site's single-table env asks for them through
//! `encode_game_state(..., obs_x_groups=...)` when the served network reads them.
//!
//! The tail is always `OBS_X_DIM` wide in the trainer's column order; a group that is off
//! stays zero. This engine computes the groups the served lineage reads -- RUN (all-in equity
//! with runouts vs one / two random hands) and LINE (betting-line summaries) -- and refuses
//! the others (RANGE, POS, RUNN: experiments nobody serves). The values are the trainer's bit
//! for bit (tests/python/engine/test_obs_x_serving.py replays hands through both engines).

use super::*;

/// Width of the tail (all groups' columns, in the trainer's order).
pub(super) const OBS_X_DIM: usize = 75;
pub(super) const G_RUN: u32 = 1;
pub(super) const G_LINE: u32 = 4;
/// The groups this engine computes.
pub(super) const G_SERVED: u32 = G_RUN | G_LINE;
const RUN_OFF: usize = 0; // 7 columns
const LINE_OFF: usize = 17; // 46 columns (RANGE's 10 sit between, unserved)

/// `Ok` for a group mask this engine can encode (RUN 1, LINE 4); the trainer also knew RANGE
/// 2, POS 8 and RUNN 16.
pub(super) fn check_groups(groups: u32) -> PyResult<()> {
    if groups & !G_SERVED != 0 {
        return Err(PyValueError::new_err(format!(
            "obs-X groups {groups:#x}: this engine computes RUN (1) and LINE (4) only"
        )));
    }
    Ok(())
}

/// Encode the tail of row `j` of `obs` -- the packed row of `state`, the env being encoded --
/// into `x` (`OBS_X_DIM` wide, pre-zeroed). RUN is the state's `x_runout_equity`; LINE is a
/// pure function of the packed row (its newest `HISTORY_DEPTH` actions). No actor = all zero,
/// like the base row.
pub(super) fn encode_x_tail(
    state: &GameState,
    obs: &PackedObservation,
    j: usize,
    num_seats: usize,
    groups: u32,
    run_samples: usize,
    x: &mut [f32],
) {
    let c = &obs.core;
    let actor = c.actor[j];
    if actor < 0 || (actor as usize) >= num_seats {
        return;
    }
    let hero = actor as usize;
    let ns = num_seats as i64;
    let rel = |seat: i64| -> usize { (seat - hero as i64).rem_euclid(ns) as usize };
    if groups & G_RUN != 0 {
        x[RUN_OFF..RUN_OFF + 7].copy_from_slice(&state.x_runout_equity(run_samples));
    }
    if groups & G_LINE != 0 {
        // Per seat (hero-rotated): [raises this hand /3 (clip 1), raised this street, raised
        // last street, called this street, checked this street] (5 blocks of 8), then 6
        // street-level dims. From the packed history window.
        let street = c.street[j] as i64;
        let hlen = (c.history_len[j] as usize).min(c.history_seat.ncols());
        let mut raises_hand = [0u32; 8];
        let mut n_raise_st = 0u32;
        let mut n_check_st = 0u32;
        let mut calls_since_raise = 0u32;
        let mut raises_total = 0u32;
        let mut checked_st = [false; 8];
        let mut check_raise = false;
        for slot in 0..hlen {
            let seat = c.history_seat[[j, slot]] as i64;
            if seat < 0 || seat >= ns {
                continue;
            }
            let k = rel(seat);
            let action = c.history_action[[j, slot]];
            let chips = c.history_chips[[j, slot]];
            let st = c.history_street[[j, slot]] as i64;
            let is_raise = action >= 2;
            let is_call = action == 1 && chips > 0;
            let is_check = action == 1 && chips == 0;
            if is_raise {
                raises_hand[k] += 1;
                raises_total += 1;
            }
            if st == street {
                if is_raise {
                    x[LINE_OFF + 8 + k] = 1.0;
                    n_raise_st += 1;
                    calls_since_raise = 0;
                    if checked_st[k] {
                        check_raise = true;
                    }
                } else if is_call {
                    x[LINE_OFF + 24 + k] = 1.0;
                    calls_since_raise += 1;
                } else if is_check {
                    x[LINE_OFF + 32 + k] = 1.0;
                    checked_st[k] = true;
                    n_check_st += 1;
                }
            } else if st == street - 1 && is_raise {
                x[LINE_OFF + 16 + k] = 1.0;
            }
        }
        for k in 0..num_seats.min(8) {
            x[LINE_OFF + k] = (raises_hand[k] as f32 / 3.0).min(1.0);
        }
        x[LINE_OFF + 40] = (n_raise_st as f32 / 4.0).min(1.0);
        x[LINE_OFF + 41] = (calls_since_raise as f32 / 5.0).min(1.0);
        x[LINE_OFF + 42] = (n_check_st as f32 / 6.0).min(1.0);
        x[LINE_OFF + 43] = (raises_total as f32 / 8.0).min(1.0);
        x[LINE_OFF + 44] = if n_raise_st >= 2 { 1.0 } else { 0.0 };
        x[LINE_OFF + 45] = if check_raise { 1.0 } else { 0.0 };
    }
}
