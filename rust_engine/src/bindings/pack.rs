//! Packing engine state into the stacked arrays the encoders (Rust and numpy)
//! read: one row per env, one array per field.
//!
//! Both layouts read the same core fields ([`PackedCore`], the whole of what
//! the minimal packer produces); the full layout adds the MC outcome block,
//! the v7 hero/board scans, blind seats and the NLH sweep
//! ([`PackedObservation`]). One row writer ([`CoreRows::write`]) fills the core
//! for both packers, so the two can no longer drift apart (ENG-011).

use std::marker::PhantomData;

use super::*;

/// Row-disjoint parallel writes into one C-contiguous `(n, width)` buffer
/// (ENG-013). The packers fill ~30 arrays per row in ONE parallel pass, which
/// safe `par_chunks_mut` cannot express for more than a couple of arrays at a
/// time. The one unsafe operation — handing out row `i` — lives here: `i` is
/// bounds-checked and the row comes back as a `width`-long slice, so writing
/// past the row (a hole wider than the packed width, ENG-004) panics instead
/// of corrupting the next row or memory past the buffer. The obligation left
/// to callers is the parallel one: two threads must never hold the same row,
/// which every packer meets by touching row `i` only from iteration `i` of
/// `(0..n).into_par_iter()`.
pub(super) struct DisjointRows<'a, T> {
    ptr: *mut T,
    rows: usize,
    width: usize,
    _buf: PhantomData<&'a mut [T]>,
}

// SAFETY: the pointer comes from a `&'a mut [T]` held for 'a and is only ever
// turned into disjoint row slices (see `row`), exactly what `par_chunks_mut`
// hands to worker threads — hence the same `T: Send` bound.
unsafe impl<T: Send> Send for DisjointRows<'_, T> {}
unsafe impl<T: Send> Sync for DisjointRows<'_, T> {}

impl<'a, T> DisjointRows<'a, T> {
    pub(super) fn new(buf: &'a mut [T], width: usize) -> Self {
        assert!(
            width > 0 && buf.len().is_multiple_of(width),
            "a buffer of whole {width}-wide rows"
        );
        DisjointRows {
            ptr: buf.as_mut_ptr(),
            rows: buf.len() / width,
            width,
            _buf: PhantomData,
        }
    }

    /// The rows of a standard-layout 2-D array (a 1-D array is `width` 1).
    pub(super) fn of2<D>(a: &'a mut numpy::ndarray::Array<T, D>, width: usize) -> Self
    where
        D: numpy::ndarray::Dimension,
    {
        Self::new(
            a.as_slice_mut().expect("packer arrays are C-contiguous"),
            width,
        )
    }

    /// Row `i` as a `width`-long slice.
    ///
    /// # Safety
    /// No other reference to row `i` may be live: callers touch row `i` only
    /// from the one parallel iteration that owns index `i`.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub(super) unsafe fn row(&self, i: usize) -> &mut [T] {
        assert!(i < self.rows, "row {i} of {}", self.rows);
        // SAFETY: `i < rows`, so the row lies inside the buffer; exclusivity
        // is the caller's contract above.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(i * self.width), self.width) }
    }

    /// `row(i)[0] = v` for a 1-D array.
    ///
    /// # Safety
    /// As [`Self::row`].
    #[inline]
    pub(super) unsafe fn set(&self, i: usize, v: T) {
        unsafe { self.row(i)[0] = v };
    }
}

/// The stacked per-row fields BOTH observation layouts read — the whole of
/// the minimal packer's output. Row `j` is env `idx[j]`; rows of terminal /
/// never-dealt envs keep the sentinels (`actor` -1, cards 255).
pub(super) struct PackedCore {
    /// (n, hole_count) the actor's hole; 255 when there is no actor.
    pub(super) hero_hole: Array2<u8>,
    /// (n, 5) visible board cards, 255-padded.
    pub(super) board_a: Array2<u8>,
    pub(super) board_b: Array2<u8>,
    pub(super) street: Array1<u8>,
    pub(super) pot: Array1<u64>,
    pub(super) stacks: Array2<u64>,
    pub(super) folded: Array2<bool>,
    pub(super) all_in: Array2<bool>,
    pub(super) bet_to_call: Array1<u64>,
    pub(super) street_commit: Array2<u64>,
    pub(super) total_commit: Array2<u64>,
    /// `min_bet_total()` / `max_bet_total()`: read by the rev-1 scalars only.
    pub(super) min_bet: Array1<u64>,
    pub(super) max_bet: Array1<u64>,
    /// Legal raise deltas (`min_raise_chips()` / `max_raise_chips()`).
    pub(super) min_raise: Array1<u64>,
    pub(super) max_raise: Array1<u64>,
    pub(super) eff_stack_cap: Array2<u64>,
    pub(super) actor: Array1<i8>,
    pub(super) button: Array1<u8>,
    /// (n, history_cap) the NEWEST `history_cap` records, oldest first; -1 /
    /// 0 in empty slots.
    pub(super) history_seat: Array2<i8>,
    pub(super) history_action: Array2<i8>,
    pub(super) history_chips: Array2<u64>,
    pub(super) history_street: Array2<i8>,
    pub(super) history_len: Array1<u8>,
}

impl PackedCore {
    /// `n` rows of sentinels / zeros.
    pub(super) fn alloc(n: usize, s: usize, hole_w: usize, hist_cap: usize) -> Self {
        PackedCore {
            hero_hole: Array2::from_elem((n, hole_w), 255u8),
            board_a: Array2::from_elem((n, 5), 255u8),
            board_b: Array2::from_elem((n, 5), 255u8),
            street: Array1::zeros(n),
            pot: Array1::zeros(n),
            stacks: Array2::zeros((n, s)),
            folded: Array2::default((n, s)),
            all_in: Array2::default((n, s)),
            bet_to_call: Array1::zeros(n),
            street_commit: Array2::zeros((n, s)),
            total_commit: Array2::zeros((n, s)),
            min_bet: Array1::zeros(n),
            max_bet: Array1::zeros(n),
            min_raise: Array1::zeros(n),
            max_raise: Array1::zeros(n),
            eff_stack_cap: Array2::zeros((n, s)),
            actor: Array1::from_elem(n, -1i8),
            button: Array1::zeros(n),
            history_seat: Array2::from_elem((n, hist_cap), -1i8),
            history_action: Array2::from_elem((n, hist_cap), -1i8),
            history_chips: Array2::zeros((n, hist_cap)),
            history_street: Array2::from_elem((n, hist_cap), -1i8),
            history_len: Array1::zeros(n),
        }
    }

    pub(super) fn rows(&mut self) -> CoreRows<'_> {
        let hole_w = self.hero_hole.ncols();
        let s = self.stacks.ncols();
        let h = self.history_seat.ncols();
        CoreRows {
            hero_hole: DisjointRows::of2(&mut self.hero_hole, hole_w),
            board_a: DisjointRows::of2(&mut self.board_a, 5),
            board_b: DisjointRows::of2(&mut self.board_b, 5),
            street: DisjointRows::of2(&mut self.street, 1),
            pot: DisjointRows::of2(&mut self.pot, 1),
            stacks: DisjointRows::of2(&mut self.stacks, s),
            folded: DisjointRows::of2(&mut self.folded, s),
            all_in: DisjointRows::of2(&mut self.all_in, s),
            bet_to_call: DisjointRows::of2(&mut self.bet_to_call, 1),
            street_commit: DisjointRows::of2(&mut self.street_commit, s),
            total_commit: DisjointRows::of2(&mut self.total_commit, s),
            min_bet: DisjointRows::of2(&mut self.min_bet, 1),
            max_bet: DisjointRows::of2(&mut self.max_bet, 1),
            min_raise: DisjointRows::of2(&mut self.min_raise, 1),
            max_raise: DisjointRows::of2(&mut self.max_raise, 1),
            eff_stack_cap: DisjointRows::of2(&mut self.eff_stack_cap, s),
            actor: DisjointRows::of2(&mut self.actor, 1),
            button: DisjointRows::of2(&mut self.button, 1),
            history_seat: DisjointRows::of2(&mut self.history_seat, h),
            history_action: DisjointRows::of2(&mut self.history_action, h),
            history_chips: DisjointRows::of2(&mut self.history_chips, h),
            history_street: DisjointRows::of2(&mut self.history_street, h),
            history_len: DisjointRows::of2(&mut self.history_len, 1),
        }
    }
}

/// Row writers over every [`PackedCore`] array.
pub(super) struct CoreRows<'a> {
    hero_hole: DisjointRows<'a, u8>,
    board_a: DisjointRows<'a, u8>,
    board_b: DisjointRows<'a, u8>,
    street: DisjointRows<'a, u8>,
    pot: DisjointRows<'a, u64>,
    stacks: DisjointRows<'a, u64>,
    folded: DisjointRows<'a, bool>,
    all_in: DisjointRows<'a, bool>,
    bet_to_call: DisjointRows<'a, u64>,
    street_commit: DisjointRows<'a, u64>,
    total_commit: DisjointRows<'a, u64>,
    min_bet: DisjointRows<'a, u64>,
    max_bet: DisjointRows<'a, u64>,
    min_raise: DisjointRows<'a, u64>,
    max_raise: DisjointRows<'a, u64>,
    eff_stack_cap: DisjointRows<'a, u64>,
    actor: DisjointRows<'a, i8>,
    button: DisjointRows<'a, u8>,
    history_seat: DisjointRows<'a, i8>,
    history_action: DisjointRows<'a, i8>,
    history_chips: DisjointRows<'a, u64>,
    history_street: DisjointRows<'a, i8>,
    history_len: DisjointRows<'a, u8>,
}

impl CoreRows<'_> {
    /// Row `i` of every core array from `state`. The history keeps the NEWEST
    /// `history_cap` records oldest-first from slot 0 (the width MUST equal
    /// the encoder's depth — see `history_cap`).
    ///
    /// # Safety
    /// As [`DisjointRows::row`]: row `i` belongs to the calling iteration.
    pub(super) unsafe fn write(&self, i: usize, state: &GameState) {
        unsafe {
            self.street.set(i, state.street.index() as u8);
            self.pot.set(i, state.pot);
            self.bet_to_call.set(i, state.bet_to_call);
            self.min_bet.set(i, state.min_bet_total());
            self.max_bet.set(i, state.max_bet_total());
            self.min_raise.set(i, state.min_raise_chips());
            self.max_raise.set(i, state.max_raise_chips());
            self.button.set(i, state.button as u8);
            if let Some(a) = state.current_actor() {
                self.actor.set(i, a as i8);
                // A hole wider than the packed width panics here (ENG-004).
                let hole = &state.hole_cards[a];
                for (dst, c) in self.hero_hole.row(i)[..hole.len()].iter_mut().zip(hole) {
                    *dst = c.index();
                }
            }
            for (rows, board) in [
                (&self.board_a, &state.board_a),
                (&self.board_b, &state.board_b),
            ] {
                let row = rows.row(i);
                for (dst, c) in row.iter_mut().zip(board.iter()) {
                    *dst = c.index();
                }
            }
            self.stacks.row(i).copy_from_slice(&state.stacks);
            self.folded.row(i).copy_from_slice(&state.folded);
            self.all_in.row(i).copy_from_slice(&state.all_in);
            self.street_commit
                .row(i)
                .copy_from_slice(&state.street_commit);
            self.total_commit
                .row(i)
                .copy_from_slice(&state.total_commit);
            self.eff_stack_cap
                .row(i)
                .copy_from_slice(&state.eff_stack_cap_at_hand_start);
            let cap = self.history_seat.width;
            let start = state.history.len().saturating_sub(cap);
            let kept = &state.history[start..];
            self.history_len.set(i, kept.len() as u8);
            let (seat, action) = (self.history_seat.row(i), self.history_action.row(i));
            let (chips, street) = (self.history_chips.row(i), self.history_street.row(i));
            for (slot, rec) in kept.iter().enumerate() {
                seat[slot] = rec.seat as i8;
                action[slot] = rec.action.index() as i8;
                chips[slot] = rec.chips;
                street[slot] = rec.street.index() as i8;
            }
        }
    }
}

/// Everything the full layout (and the numpy encoders' dict) reads: the core
/// plus the full-only blocks.
pub(super) struct PackedObservation {
    pub(super) core: PackedCore,
    pub(super) board_a_len: Array1<u8>,
    pub(super) board_b_len: Array1<u8>,
    pub(super) last_aggressor: Array1<i8>,
    /// (n, 12) PLO opp-outcome fractions, row-major [k=2,3,4][outcome]
    /// (scoop_opp, quarter_opp, scoop_hero, quarter_hero); zero pre-flop /
    /// terminal / NLH / with the MC disabled.
    pub(super) opp_outcome_fractions: Array2<f32>,
    /// Per-board hero ahead/tie/behind + win-one/tie-both fractions (obs v2
    /// P1; k=2 exhaustive, same fused pass). All-zero for NLH.
    pub(super) per_board_outcome: Array2<f32>,
    /// (n, 2) k=2 guaranteed-pot-share bounds [g_min, g_max] (v7 DUAL-4;
    /// same fused pass, dims 20/21). All-zero for NLH / inactive states.
    pub(super) share_bounds: Array2<f32>,
    /// (n, s) engine acted_this_street bits (v7 STK-1 pending-set input).
    pub(super) acted_this_street: Array2<bool>,
    /// (n, 8) v7 hero/board engine dims [boat_a, boat_b, improve_a,
    /// improve_b, combos_a, combos_b, mask_a, mask_b] (BRD-7 / BRD-12 /
    /// DUAL-2; see GameState::hero_board_v3). All-zero for NLH.
    pub(super) hero_board_v3: Array2<u8>,
    /// (n, 7) v7 BRD-5/BRD-6/DUAL-5 hot counts:
    /// [ds_a, ds_b, u_a, n_a, u_b, n_b, scoop]. All-zero for NLH.
    pub(super) board_draw_v3: Array2<u8>,
    /// Blind seats (-1 when the variant has none). NLH batch encoder input.
    pub(super) sb_seat: Array1<i8>,
    pub(super) bb_seat: Array1<i8>,
    /// NLH 3-dim [opp_ahead, tied, opp_behind] exhaustive fractions;
    /// all-zero rows for PLO variants (mirrors `nlh_opp_outcome_fractions`'s
    /// variant guard on the serial path).
    pub(super) nlh_opp_outcome: Array2<f32>,
}

/// The legal-action mask of an env for the packers: all-False when terminal.
#[inline]
fn legal_row(state: &GameState) -> [bool; NUM_ACTIONS] {
    if state.is_terminal() {
        [false; NUM_ACTIONS]
    } else {
        state.legal_action_mask()
    }
}

impl PackedObservation {
    pub(super) fn alloc(n: usize, s: usize, hole_w: usize, hist_cap: usize) -> Self {
        PackedObservation {
            core: PackedCore::alloc(n, s, hole_w, hist_cap),
            board_a_len: Array1::zeros(n),
            board_b_len: Array1::zeros(n),
            last_aggressor: Array1::from_elem(n, -1i8),
            opp_outcome_fractions: Array2::zeros((n, 12)),
            per_board_outcome: Array2::zeros((n, 8)),
            share_bounds: Array2::zeros((n, 2)),
            acted_this_street: Array2::default((n, s)),
            hero_board_v3: Array2::zeros((n, 8)),
            board_draw_v3: Array2::zeros((n, 7)),
            sb_seat: Array1::from_elem(n, -1i8),
            bb_seat: Array1::from_elem(n, -1i8),
            nlh_opp_outcome: Array2::zeros((n, 3)),
        }
    }

    /// Every packed array as the dict the numpy encoders read (ENG-002: the
    /// one place its keys are listed). Callers add their extras (`legal_mask`,
    /// `hero_cat_a` / `hero_cat_b`, `obs`).
    pub(super) fn into_dict(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let c = self.core;
        let d = PyDict::new(py);
        d.set_item("hero_hole", c.hero_hole.into_pyarray(py))?;
        d.set_item("board_a", c.board_a.into_pyarray(py))?;
        d.set_item("board_b", c.board_b.into_pyarray(py))?;
        d.set_item("board_a_len", self.board_a_len.into_pyarray(py))?;
        d.set_item("board_b_len", self.board_b_len.into_pyarray(py))?;
        d.set_item("street", c.street.into_pyarray(py))?;
        d.set_item("pot", c.pot.into_pyarray(py))?;
        d.set_item("stacks", c.stacks.into_pyarray(py))?;
        d.set_item("folded", c.folded.into_pyarray(py))?;
        d.set_item("all_in", c.all_in.into_pyarray(py))?;
        d.set_item("bet_to_call", c.bet_to_call.into_pyarray(py))?;
        d.set_item("street_commit", c.street_commit.into_pyarray(py))?;
        d.set_item("total_commit", c.total_commit.into_pyarray(py))?;
        d.set_item("min_bet", c.min_bet.into_pyarray(py))?;
        d.set_item("max_bet", c.max_bet.into_pyarray(py))?;
        d.set_item("min_raise", c.min_raise.into_pyarray(py))?;
        d.set_item("max_raise", c.max_raise.into_pyarray(py))?;
        d.set_item("eff_stack_cap", c.eff_stack_cap.into_pyarray(py))?;
        d.set_item("actor", c.actor.into_pyarray(py))?;
        d.set_item("button", c.button.into_pyarray(py))?;
        d.set_item("last_aggressor", self.last_aggressor.into_pyarray(py))?;
        d.set_item("history_seat", c.history_seat.into_pyarray(py))?;
        d.set_item("history_action", c.history_action.into_pyarray(py))?;
        d.set_item("history_chips", c.history_chips.into_pyarray(py))?;
        d.set_item("history_street", c.history_street.into_pyarray(py))?;
        d.set_item("history_len", c.history_len.into_pyarray(py))?;
        d.set_item(
            "opp_outcome_fractions",
            self.opp_outcome_fractions.into_pyarray(py),
        )?;
        d.set_item("per_board_outcome", self.per_board_outcome.into_pyarray(py))?;
        d.set_item("share_bounds", self.share_bounds.into_pyarray(py))?;
        d.set_item("acted_this_street", self.acted_this_street.into_pyarray(py))?;
        d.set_item("hero_board_v3", self.hero_board_v3.into_pyarray(py))?;
        d.set_item("board_draw_v3", self.board_draw_v3.into_pyarray(py))?;
        // Every builder carries the NLH keys (review 2026-09-20 C4: a dict
        // without the blind seats could not feed encode_observation_batch_nlh).
        d.set_item("sb_seat", self.sb_seat.into_pyarray(py))?;
        d.set_item("bb_seat", self.bb_seat.into_pyarray(py))?;
        d.set_item("nlh_opp_outcome", self.nlh_opp_outcome.into_pyarray(py))?;
        Ok(d)
    }
}

/// Everything the full packer derives from (street, the actor's hole, both
/// boards) for a PLO row — the value of the per-(env, seat) memo
/// (`PyBatchedEngine::outcome_cache`), whose key hashes exactly those inputs.
#[derive(Clone, Copy, Debug)]
pub(super) struct RowFeatures {
    /// The opp-outcome MC block: 12 joint fractions, 8 per-board, 2 share bounds.
    pub(super) mc: [f32; 22],
    /// `GameState::hero_board_v3` (BRD-7 / BRD-12 / DUAL-2).
    pub(super) v3: [u8; 8],
    /// `GameState::board_draw_v3` (BRD-5 / BRD-6 / DUAL-5).
    pub(super) draw: [u8; 7],
    /// The actor's made-hand category on board A / B.
    pub(super) cats: [u8; 2],
}

impl RowFeatures {
    pub(super) const ZERO: RowFeatures = RowFeatures {
        mc: [0.0; 22],
        v3: [0; 8],
        draw: [0; 7],
        cats: [0; 2],
    };

    /// A PLO row's features with `mc` opp-outcome samples (0 = no MC block).
    /// `table`: the env's shared board pair-rank table, refreshed here when
    /// it belongs to other boards (its lock is uncontended: one env per row).
    pub(super) fn compute(
        state: &GameState,
        mc: usize,
        table: Option<&std::sync::Mutex<Option<crate::engine::BoardPairTable>>>,
    ) -> Self {
        let (v3, cats) = state.hero_board_features();
        let draw = state.board_draw_v3();
        let fr = match table {
            Some(t) => {
                let mut slot = t.lock().unwrap_or_else(|e| e.into_inner());
                let key = crate::engine::BoardPairTable::key_of(&state.board_a, &state.board_b);
                if slot.as_ref().is_none_or(|t| t.key != key) {
                    *slot = state.board_pair_table();
                }
                state.outcome_features_mc_shared(mc, slot.as_ref())
            }
            None => state.outcome_features_mc(mc),
        };
        RowFeatures {
            mc: fr,
            v3,
            draw,
            cats,
        }
    }
}

/// The full packer's output: every field, the legal masks and the actor's
/// made-hand category on each board (0 without an actor or a flop).
pub(super) struct PackedFull {
    pub(super) obs: PackedObservation,
    pub(super) legal: Array2<bool>,
    pub(super) cat_a: Vec<u8>,
    pub(super) cat_b: Vec<u8>,
}

impl PyBatchedEngine {
    /// Pack the envs `idx` (row j = env `idx[j]`) for the full layout, with
    /// legal masks and hero categories, in two parallel passes (PERF-035: this
    /// used to take five plus serial copy loops, and each worker wake-up costs
    /// 0.2-0.4 ms on the pod):
    ///
    /// 1. the opp-outcome MC cache keys (cheap hashes);
    /// 2. one pass per row: the MC block (fresh, or the (env, seat) memo's copy
    ///    — or NLH's exhaustive sweep), every packed field, the v7 scans, the
    ///    legal mask and the categories.
    ///
    /// Between them, serially, the memo decides which rows need a fresh MC;
    /// after, the fresh results are stored. Every value is a pure function of
    /// the env's state (the MC of its seed), so the pass structure cannot
    /// change a bit (golden digests).
    pub(super) fn pack_full(&self, idx: &[usize]) -> PackedFull {
        let n = idx.len();
        let s = self.config.num_seats;
        let hole_w = self.config.variant.hole_count();
        let is_nlh = !self.config.variant.is_plo();
        let mc = self.opp_outcome_mc;
        let states = &self.states;
        let mut out = PackedObservation::alloc(n, s, hole_w, history_cap(self.config.variant));
        let mut legal = Array2::<bool>::default((n, NUM_ACTIONS));
        let mut cat_a = vec![0u8; n];
        let mut cat_b = vec![0u8; n];

        // PLO per-(env, SEAT) memo, keyed on `outcome_seed` = a hash of
        // exactly the inputs of everything in a `RowFeatures` (street + the
        // actor's hole + both boards, as card SETS): the MC block, the v7
        // hero/board and draw dims and the hero's categories. A seat that acts
        // again on the same street reuses all of it (review 2026-09-20 C6 —
        // one slot per env never hit, the actor changes on every action;
        // PERF-029 — the v7 scans and categories ride along). Rows without an
        // actor or a flop on both boards have no key; their features are all
        // zero anyway. Bit-exact vs always-recompute (test_encoding_rust:
        // cached batched == fresh serial; the golden digests).
        let use_memo = !is_nlh && mc > 0;
        let keys: Vec<Option<(usize, u64)>> = if use_memo {
            (0..n)
                .into_par_iter()
                .map(|j| {
                    let st = states[idx[j]].as_ref()?;
                    let seed = st.outcome_seed()?;
                    Some((idx[j] * s + st.current_actor()?, seed))
                })
                .collect()
        } else {
            Vec::new()
        };
        // Per row: Some(features) on a memo hit, `fresh[j]` = compute + store.
        let (cached, fresh): (Vec<Option<RowFeatures>>, Vec<bool>) = if use_memo {
            let mut cache = self.outcome_cache.lock().unwrap_or_else(|e| e.into_inner());
            let mut hits = 0u64;
            let mut lookups = 0u64;
            let mut cached = vec![None; n];
            let mut fresh = vec![false; n];
            for j in 0..n {
                if let Some((slot, seed)) = keys[j] {
                    lookups += 1;
                    match cache.slots[slot] {
                        Some((s0, f)) if s0 == seed => {
                            hits += 1;
                            cached[j] = Some(f);
                        }
                        _ => fresh[j] = true,
                    }
                }
            }
            cache.lookups += lookups;
            cache.hits += hits;
            (cached, fresh)
        } else {
            (Vec::new(), Vec::new())
        };
        // The fresh rows' features, handed from the parallel pass to the store.
        let mut computed: Vec<RowFeatures> = vec![RowFeatures::ZERO; if use_memo { n } else { 0 }];

        {
            let core = out.core.rows();
            let board_a_len = DisjointRows::of2(&mut out.board_a_len, 1);
            let board_b_len = DisjointRows::of2(&mut out.board_b_len, 1);
            let last_aggressor = DisjointRows::of2(&mut out.last_aggressor, 1);
            let opp = DisjointRows::of2(&mut out.opp_outcome_fractions, 12);
            let per_board = DisjointRows::of2(&mut out.per_board_outcome, 8);
            let share = DisjointRows::of2(&mut out.share_bounds, 2);
            let acted = DisjointRows::of2(&mut out.acted_this_street, s);
            let hbv = DisjointRows::of2(&mut out.hero_board_v3, 8);
            let bdv = DisjointRows::of2(&mut out.board_draw_v3, 7);
            let sb_seat = DisjointRows::of2(&mut out.sb_seat, 1);
            let bb_seat = DisjointRows::of2(&mut out.bb_seat, 1);
            let nlh = DisjointRows::of2(&mut out.nlh_opp_outcome, 3);
            let legal_rows = DisjointRows::of2(&mut legal, NUM_ACTIONS);
            let ca = DisjointRows::new(&mut cat_a, 1);
            let cb = DisjointRows::new(&mut cat_b, 1);
            let computed_rows = DisjointRows::new(&mut computed, 1);
            let tables = &self.board_tables;
            (0..n).into_par_iter().for_each(|j| {
                let Some(state) = states[idx[j]].as_ref() else {
                    return;
                };
                // SAFETY: iteration j is the only one touching row j of any
                // of these buffers (DisjointRows' contract).
                unsafe {
                    core.write(j, state);
                    board_a_len.set(j, state.board_a.len().min(5) as u8);
                    board_b_len.set(j, state.board_b.len().min(5) as u8);
                    last_aggressor.set(j, state.last_aggressor.map_or(-1, |a| a as i8));
                    sb_seat.set(j, state.sb_seat.map_or(-1, |x| x as i8));
                    bb_seat.set(j, state.bb_seat.map_or(-1, |x| x as i8));
                    acted.row(j).copy_from_slice(&state.acted_this_street);
                    legal_rows.row(j).copy_from_slice(&legal_row(state));
                    if is_nlh {
                        // NLH: no v7 dims; categories + the exhaustive sweep.
                        if let Some(a) = state.current_actor() {
                            ca.set(j, state.hero_category(a, 0));
                            cb.set(j, state.hero_category(a, 1));
                        }
                        nlh.row(j)
                            .copy_from_slice(&state.nlh_opp_outcome_fractions()[..3]);
                        return;
                    }
                    let f = match cached.get(j).copied().flatten() {
                        Some(f) => f,
                        None => {
                            let table = (use_memo && fresh[j]).then(|| &tables[idx[j]]);
                            let f =
                                RowFeatures::compute(state, if use_memo { mc } else { 0 }, table);
                            if use_memo && fresh[j] {
                                computed_rows.set(j, f);
                            }
                            f
                        }
                    };
                    hbv.row(j).copy_from_slice(&f.v3);
                    bdv.row(j).copy_from_slice(&f.draw);
                    ca.set(j, f.cats[0]);
                    cb.set(j, f.cats[1]);
                    opp.row(j).copy_from_slice(&f.mc[..12]);
                    per_board.row(j).copy_from_slice(&f.mc[12..20]);
                    share.row(j).copy_from_slice(&f.mc[20..22]);
                }
            });
        }

        if use_memo {
            let mut cache = self.outcome_cache.lock().unwrap_or_else(|e| e.into_inner());
            for j in (0..n).filter(|&j| fresh[j]) {
                if let Some((slot, seed)) = keys[j] {
                    cache.slots[slot] = Some((seed, computed[j]));
                }
            }
        }
        PackedFull {
            obs: out,
            legal,
            cat_a,
            cat_b,
        }
    }

    /// Lean pack for the minimal layout: the core fields only (no MC, no v7
    /// scans, no categories) plus the legal masks, in ONE parallel pass — on a
    /// busy host the worker wake-ups, not the work, dominate these small passes.
    pub(super) fn pack_minimal_with_legal(&self, idx: &[usize]) -> (PackedCore, Array2<bool>) {
        let n = idx.len();
        let s = self.config.num_seats;
        let hole_w = self.config.variant.hole_count();
        let mut core = PackedCore::alloc(n, s, hole_w, history_cap(self.config.variant));
        let mut legal = Array2::<bool>::default((n, NUM_ACTIONS));
        {
            let rows = core.rows();
            let legal_rows = DisjointRows::of2(&mut legal, NUM_ACTIONS);
            let states = &self.states;
            (0..n).into_par_iter().for_each(|j| {
                let Some(state) = states[idx[j]].as_ref() else {
                    return;
                };
                // SAFETY: iteration j is the only one touching row j.
                unsafe {
                    rows.write(j, state);
                    legal_rows.row(j).copy_from_slice(&legal_row(state));
                }
            });
        }
        (core, legal)
    }
}
