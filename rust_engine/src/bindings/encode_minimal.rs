//! The minimal (796-dim, bare-visibility) observation layout and its encoder.

use super::*;

// =============================================================================
// Bare-visibility (minimal) layout — contiguous 796 dims matching
// python/plo5bp/encoding.py _M_* offsets / encode_observation_batch_minimal.
// =============================================================================
pub(super) mod obs_layout_minimal {
    // Dims 0..188 (hole, boards, street, active / all-in / stacks, scalars)
    // and the history slot format are the core blocks both layouts share:
    // `obs_core` in encode.rs (encoded once, by `encode_core`).
    pub const OBS_DIM_MINIMAL: usize = 796;
    // History starts at 188 in minimal (full layout has REL_POS at 188 and
    // history at 196 — REL_POS is dropped, so history slides left by 8).
    pub const HISTORY_OFF: usize = 188;
    pub const SEAT_EXISTS_OFF: usize = 764;
    pub const TOTAL_COMMIT_OFF: usize = 772;
    pub const STREET_COMMIT_OFF: usize = 780;
    pub const HERO_BTN_OFF: usize = 788;
}

/// Where the minimal layout puts the core blocks that move between layouts
/// (no REL_POS: the history starts at 188, 8 dims earlier than the full's).
pub(super) const MINIMAL_CORE_AT: CoreOffsets = CoreOffsets {
    history: obs_layout_minimal::HISTORY_OFF,
    seat_exists: obs_layout_minimal::SEAT_EXISTS_OFF,
    total_commit: obs_layout_minimal::TOTAL_COMMIT_OFF,
    street_commit: obs_layout_minimal::STREET_COMMIT_OFF,
    hero_btn: obs_layout_minimal::HERO_BTN_OFF,
};

/// Encode one env into bare-visibility `out` (length OBS_DIM_MINIMAL=796,
/// pre-zeroed): exactly the core blocks both layouts share
/// ([`encode_core`]). Bit-exact with python `encode_observation_minimal` /
/// `encode_observation_batch_minimal`. Terminal rows (actor < 0) stay zero.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_obs_row_minimal(
    packed: &PackedCore,
    j: usize,
    num_seats: usize,
    inv_bb: f64,
    bb: u64,
    starting: &[u64],
    obs_rev: u8,
    out: &mut [f32],
) {
    encode_core(
        packed,
        j,
        &MINIMAL_CORE_AT,
        num_seats,
        inv_bb,
        bb,
        starting,
        obs_rev,
        out,
    );
}
