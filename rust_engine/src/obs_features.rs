//! Observation features computed from a [`GameState`] (ENG-010): the actor's
//! made-hand categories, the v7 hero/board and board-draw dims, the PLO
//! opp-outcome Monte Carlo (+ its shared pair-rank table and seed) and the
//! NLH exhaustive opp-outcome sweep. These change with every observation
//! revision; the poker rules in engine.rs — stable and reviewed — do not.
//! Every method here is a pure function of the state it reads.

use crate::cards::{Card, CardMask};
use crate::engine::SeedMixer;
use crate::hand_eval::{pack_rank16, unpack_rank16};
use crate::state::{GameState, Street, Variant};

impl GameState {
    /// Hand category index (0..=8 per `CAT_*` constants) of seat's current
    /// best hand on `board` (0=A, 1=B) under the variant's evaluation
    /// rule. Returns 0 if board has <3 cards (always for board B on
    /// single-board variants — its progressive view stays empty).
    pub fn hero_category(&self, seat: usize, board: u8) -> u8 {
        let b = if board == 0 {
            &self.board_a
        } else {
            &self.board_b
        };
        if b.len() < 3 {
            return 0;
        }
        let hole = &self.hole_cards[seat];
        let rank = if self.config.variant.is_plo() {
            crate::hand_eval::evaluate_plo_partial(hole, b)
        } else {
            crate::hand_eval::evaluate_nlh(hole, b)
        };
        (rank >> 20) as u8
    }

    /// v7 obs batch-2 hero/board engine dims for the CURRENT actor
    /// (V7_OBS_CANDIDATES.md BRD-7 / BRD-12 / DUAL-2):
    /// `[boat_a, boat_b, improve_a, improve_b, combos_a, combos_b,
    ///   mask_a, mask_b]` — boat-or-better outs, strict-category-improve
    /// outs, best-category combo counts (all raw counts; the encoders
    /// normalize), and the best-holding 5-bit hole masks (bit i = i-th
    /// hole card sorted by card index DESCENDING). The unseen deck for
    /// the out counts is GLOBAL (hole + BOTH boards), matching the
    /// encoder's cross-board visibility convention. All-zero when there
    /// is no actor, for NLH (any-combo eval — these are PLO semantics),
    /// or before both boards have flops.
    pub fn hero_board_v3(&self) -> [u8; 8] {
        self.hero_board_features().0
    }

    /// [`Self::hero_board_v3`] plus the actor's made-hand category on each
    /// board (`[hero_category(actor, 0), hero_category(actor, 1)]`), from the
    /// same pass (PERF-029: the categories fall out of the per-pair table the
    /// v3 dims need, so the packers stop ranking the hero's hand again).
    /// PLO only: `([0; 8], [0, 0])` for NLH (use `hero_category` there), with
    /// no actor, or before both boards have flops (where `hero_category` is
    /// 0 too — PLO boards come out together).
    pub fn hero_board_features(&self) -> ([u8; 8], [u8; 2]) {
        let zero = ([0u8; 8], [0u8; 2]);
        if self.config.variant == Variant::NlhSingle {
            return zero;
        }
        let Some(hero) = self.actor else {
            return zero;
        };
        if self.board_a.len() < 3 || self.board_b.len() < 3 {
            return zero;
        }
        let hole = &self.hole_cards[hero];
        let used = CardMask::of(
            hole.iter()
                .chain(self.board_a.iter())
                .chain(self.board_b.iter()),
        );
        let a = crate::hand_eval::hero_board_one(hole, &self.board_a, used);
        let b = crate::hand_eval::hero_board_one(hole, &self.board_b, used);
        (
            [
                a.boat, b.boat, a.improve, b.improve, a.combos, b.combos, a.mask, b.mask,
            ],
            [a.category, b.category],
        )
    }

    /// v7 BRD-5/BRD-6/DUAL-5 hot block:
    /// [ds_a, ds_b, u_a, n_a, u_b, n_b, scoop] raw counts; encoder
    /// normalizes. All-zero for NLH / no-actor / short boards.
    pub fn board_draw_v3(&self) -> [u8; 7] {
        let zero = [0u8; 7];
        if self.config.variant == Variant::NlhSingle {
            return zero;
        }
        let hero = match self.actor {
            Some(s) => s,
            None => return zero,
        };
        if self.board_a.len() < 3 || self.board_b.len() < 3 {
            return zero;
        }
        let hole = &self.hole_cards[hero];
        let used = CardMask::of(
            hole.iter()
                .chain(self.board_a.iter())
                .chain(self.board_b.iter()),
        );
        crate::hand_eval::board_draw_v3(hole, &self.board_a, &self.board_b, used)
    }

    /// Fraction of unseen-deck k-card opponent hands that produce each of
    /// 4 outcomes vs the hero (current actor) on both boards, for
    /// k ∈ {2, 3, 4}. Returns a length-12 `Vec<f32>` in row-major
    /// `[k][outcome]` order:
    /// - Outcome 0: opp scoops (opp wins both boards).
    /// - Outcome 1: opp quarters hero (opp wins one, ties the other).
    /// - Outcome 2: hero scoops (hero wins both).
    /// - Outcome 3: hero quarters opp (hero wins one, ties the other).
    ///
    /// Chops, splits, and double-ties are intentionally excluded; the
    /// network can infer them as residuals.
    ///
    /// Evaluation rule (current-rank dominance): each k-card opp hand is
    /// evaluated on the *visible* board under PLO5 rules (exactly 2 from
    /// k + 3 from board). No runout sampling on flop/turn.
    ///
    /// Sampling: k=2 is exhaustive; k=3 and k=4 use `mc_samples` MC
    /// draws each. PRNG seeded deterministically from the immutable
    /// observation state so the feature is reproducible (parity tests
    /// survive at a fixed sample count).
    ///
    /// Returns all-zero before the flop or when the hand is terminal.
    ///
    /// Serial / UI / eval callers use the 1024-sample
    /// `opp_outcome_fractions` wrapper; batched training passes a smaller
    /// `mc_samples` (`rollout.py`: 384 at the time of writing) — the MC is
    /// the largest per-decision encode cost (its k=2 pass reads the shared
    /// pair table when the batched packer has one).
    pub fn opp_outcome_fractions_mc(&self, mc_samples: usize) -> Vec<f32> {
        self.outcome_features_mc(mc_samples)[..12].to_vec()
    }

    /// Deterministic key for the opp-outcome MC — a hash of exactly the inputs
    /// the MC depends on (street, hero hole, both boards — each as a card
    /// SET; see [`outcome_mc_seed`]). Returns `None` in precisely the cases
    /// `outcome_features_mc` returns all-zeros (no actor, or fewer than 3
    /// board cards on either board). Used as the per-env cache key in the
    /// batched pack: equal key => the MC would produce the identical output,
    /// so the cached value can be reused.
    ///
    /// Shares [`outcome_mc_seed`] with the seed inside `outcome_features_mc`
    /// below, so the two stay in lockstep by construction.
    pub fn outcome_seed(&self) -> Option<u64> {
        let hero_seat = self.actor?;
        if self.board_a.len() < 3 || self.board_b.len() < 3 {
            return None;
        }
        Some(outcome_mc_seed(
            self.street,
            &self.hole_cards[hero_seat],
            &self.board_a,
            &self.board_b,
        ))
    }

    /// Superset of [`Self::opp_outcome_fractions_mc`]: the 12 joint
    /// outcome fractions PLUS an 8-dim PER-BOARD decomposition (obs v2,
    /// V5_DESIGN.md P1), all from the SAME single pass — the extra dims
    /// are counter increments inside the existing k=2 exhaustive loop
    /// (no extra evals, no extra RNG draws, so dims 0..12 stay
    /// bit-identical to the pre-v5 feature).
    ///
    /// Dims 12..20, hero-centric, k=2 EXHAUSTIVE universe only (exact,
    /// deterministic):
    /// - 12/13/14: board A — fraction of combos hero currently beats /
    ///   ties / is behind (sums to 1 when active).
    /// - 15/16/17: board B — same.
    /// - 18: win-exactly-one (hero ahead on one board, behind on the
    ///   other, either direction) — the modal double-board outcome the
    ///   12-dim block folds into its residual.
    /// - 19: tie on BOTH boards.
    pub fn outcome_features_mc(&self, mc_samples: usize) -> [f32; 22] {
        self.outcome_features_mc_shared(mc_samples, None)
    }

    /// The pair-rank table `outcome_features_mc`'s k=2 exhaustive pass reads,
    /// for EVERY 2-card holding of cards on neither board (2026-09-26): it
    /// depends only on the two boards, so one table serves every seat that
    /// acts on this street -- the batched packer keeps one per env and the
    /// k=2 pass becomes lookups (the evaluations were most of the MC's cost,
    /// and they were repeated for each acting seat). `None` before both
    /// boards have a flop (the MC returns zeros there anyway).
    pub fn board_pair_table(&self) -> Option<BoardPairTable> {
        if self.board_a.len() < 3 || self.board_b.len() < 3 {
            return None;
        }
        let on_board = CardMask::of(self.board_a.iter().chain(self.board_b.iter()));
        let mut t = BoardPairTable {
            key: BoardPairTable::key_of(&self.board_a, &self.board_b),
            ranks: Box::new([[0u16; 2]; BoardPairTable::PAIRS]),
        };
        let mut pair = [Card::from_index(0); 2];
        for c0 in 0..52u8 {
            if on_board.contains(Card(c0)) {
                continue;
            }
            for c1 in (c0 + 1)..52u8 {
                if on_board.contains(Card(c1)) {
                    continue;
                }
                pair[0] = Card::from_index(c0);
                pair[1] = Card::from_index(c1);
                t.ranks[BoardPairTable::pair_index(c0, c1)] = [
                    pack_rank16(crate::hand_eval::evaluate_plo_k_partial(
                        &pair,
                        &self.board_a,
                    )),
                    pack_rank16(crate::hand_eval::evaluate_plo_k_partial(
                        &pair,
                        &self.board_b,
                    )),
                ];
            }
        }
        Some(t)
    }

    /// `outcome_features_mc` with the k=2 pass's pair ranks read from `shared`
    /// (a `board_pair_table()` of THESE boards) instead of evaluated: the same
    /// ranks, so the same output bit for bit (pinned by
    /// `shared_pair_table_matches_outcome_features_mc`). A table of other
    /// boards is ignored (falls back to evaluating).
    pub fn outcome_features_mc_shared(
        &self,
        mc_samples: usize,
        shared: Option<&BoardPairTable>,
    ) -> [f32; 22] {
        let shared =
            shared.filter(|t| t.key == BoardPairTable::key_of(&self.board_a, &self.board_b));
        // N_OUT 20 → 22 (2026-07-12, DUAL-4): dims 20/21 append the k=2
        // guaranteed-pot-share bounds g_min/g_max. Dims 0..20 stay
        // byte-identical to the pre-append body — the P1 pin test compares
        // them against the frozen reference; the share trackers add no
        // evals, no RNG draws, and no reordering.
        const N_OUT: usize = 22;
        // mc_samples == 0: caller does not need outcome features (e.g.
        // obs_mode=minimal). Skip all evals / deck work and return zeros.
        if mc_samples == 0 {
            return [0.0; N_OUT];
        }
        const SCOOP_OPP: usize = 0;
        const QUARTER_OPP: usize = 1;
        const SCOOP_HERO: usize = 2;
        const QUARTER_HERO: usize = 3;
        const PER_BOARD_OFF: usize = 12;
        const SHARE_OFF: usize = 20;

        let hero_seat = match self.actor {
            Some(s) => s,
            None => return [0.0; N_OUT],
        };
        if self.board_a.len() < 3 || self.board_b.len() < 3 {
            return [0.0; N_OUT];
        }

        let hero_hole = &self.hole_cards[hero_seat];
        let hero_a = crate::hand_eval::evaluate_plo_partial(hero_hole, &self.board_a);
        let hero_b = crate::hand_eval::evaluate_plo_partial(hero_hole, &self.board_b);

        // Build unseen deck (52 minus hero hole minus visible board on
        // both boards).
        let used = CardMask::of(
            hero_hole
                .iter()
                .chain(self.board_a.iter())
                .chain(self.board_b.iter()),
        );
        let mut unseen = [Card::from_index(0); 52];
        let mut n_unseen = 0;
        for c in used.unseen() {
            unseen[n_unseen] = c;
            n_unseen += 1;
        }

        // Deterministic seed from the observation-visible state (the same
        // derivation `outcome_seed` exposes as the batched cache key).
        let seed = outcome_mc_seed(self.street, hero_hole, &self.board_a, &self.board_b);

        use rand_chacha::rand_core::{RngCore, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut out = [0.0f32; N_OUT];

        // Per-holding board comparisons; +1 = opp ahead, 0 = tie, -1 = opp
        // behind (higher HandRank = stronger). Joint/per-board tallying
        // happens at the call sites.
        let cmp_ranks = |opp_a: u32, opp_b: u32| -> (i8, i8) {
            let cmp_a: i8 = if opp_a > hero_a {
                1
            } else if opp_a < hero_a {
                -1
            } else {
                0
            };
            let cmp_b: i8 = if opp_b > hero_b {
                1
            } else if opp_b < hero_b {
                -1
            } else {
                0
            };
            (cmp_a, cmp_b)
        };
        let tally_joint = |counters: &mut [u32; 4], cmp_a: i8, cmp_b: i8| match (cmp_a, cmp_b) {
            (1, 1) => counters[SCOOP_OPP] += 1,
            (-1, -1) => counters[SCOOP_HERO] += 1,
            (1, 0) | (0, 1) => counters[QUARTER_OPP] += 1,
            (-1, 0) | (0, -1) => counters[QUARTER_HERO] += 1,
            _ => {}
        };

        // P1: per-board pair-rank scratch tables, filled by the k=2 exhaustive
        // pass and reused by the k=3/4 MC arms. A PLO holding must use EXACTLY
        // 2 hole cards, so a k-card holding's rank factorizes as max over its
        // C(k,2) pairs of that pair's rank — and every MC-drawable pair is
        // enumerated by the k=2 pass (same unseen deck), so the MC arms become
        // table lookups instead of full evaluate_plo_k_partial calls (~81-84%
        // of this block's hand-eval work at 384 samples). Degenerate pairs
        // (duplicate-card states; every combo ck==0-filtered) store
        // ck_to_hand_rank(7462) == 0 == the u32 order bottom, exactly
        // mirroring the k-level per-combo skip — the identity holds for every
        // state, duplicates included. Indexed by unseen-deck POSITIONS
        // (lo*STRIDE + hi, lo<hi); stride 52 covers every variant and
        // duplicate-card state (PLO4 flop = 42 unseen; duplicated hole/board
        // cards push n_unseen higher still). No engine path produces
        // duplicates any more — study mode redraws colliding placeholders
        // (review 2026-09-20 C8) — but the function stays total over them.
        // Byte-identity vs the frozen pre-P1 body is pinned by
        // outcome_mc_p1_tests.
        debug_assert!(n_unseen <= PAIR_STRIDE);
        // PERF-033: this thread's reused scratch, not two freshly zeroed
        // 10.8 KB tables per call -- every entry a k=3/4 draw reads is written
        // by this call's k=2 pass first (it enumerates every position pair).
        PAIR_SCRATCH.with_borrow_mut(|tab| {
            for (idx_k, &k) in [2usize, 3, 4].iter().enumerate() {
                let mut counters = [0u32; 4];
                let mut samples: u32 = 0;

                // k=2 exhaustive (C(<=41, 2) <= 820 is cheap); k=3,4 always
                // MC (`mc_samples` draws each). Previously k=3 was exhaustive
                // at turn+river (C(39,3) and C(37,3) both <= 10k), but that's
                // ~18k evals/env vs ~2*mc_samples for MC — dominated bundle cost.
                if k == 2 {
                    // Per-board counters (obs v2 P1): [ahead, tie, behind] per
                    // board from HERO's perspective + win-exactly-one + tie-both.
                    let mut pb = [0u32; 8];
                    // DUAL-4: hero's per-combo pot share s = 0.5·[wins A] +
                    // 0.25·[ties A] + 0.5·[wins B] + 0.25·[ties B]; track the
                    // min/max over the exhaustive k=2 universe. Values land
                    // exactly on {0, .25, .5, .75, 1}. Free riders on the
                    // existing loop — no extra evals, no RNG.
                    let mut g_min = f32::MAX;
                    let mut g_max = f32::MIN;
                    // Every pair of unseen positions i0 < i1 (the tallies are
                    // counts, a min and a max: the order is immaterial).
                    for i0 in 0..n_unseen {
                        let c0 = unseen[i0];
                        // `unseen` ascends, so c0 < c1: the shared table's
                        // slots for c0 start at `row`.
                        let row = BoardPairTable::row_of(c0.index());
                        for i1 in (i0 + 1)..n_unseen {
                            let c1 = unseen[i1];
                            let (opp_a, opp_b) = match shared {
                                Some(t) => {
                                    let [a, b] =
                                        t.ranks[row + (c1.index() - c0.index() - 1) as usize];
                                    (unpack_rank16(a), unpack_rank16(b))
                                }
                                None => {
                                    let pair = [c0, c1];
                                    (
                                        crate::hand_eval::evaluate_plo_k_partial(
                                            &pair,
                                            &self.board_a,
                                        ),
                                        crate::hand_eval::evaluate_plo_k_partial(
                                            &pair,
                                            &self.board_b,
                                        ),
                                    )
                                }
                            };
                            // P1: record this pair's per-board ranks for the k=3/4
                            // MC arms (i0 < i1).
                            tab[i0 * PAIR_STRIDE + i1] = [opp_a, opp_b];
                            let (cmp_a, cmp_b) = cmp_ranks(opp_a, opp_b);
                            tally_joint(&mut counters, cmp_a, cmp_b);
                            match cmp_a {
                                -1 => pb[0] += 1, // hero ahead on A
                                0 => pb[1] += 1,
                                _ => pb[2] += 1,
                            }
                            match cmp_b {
                                -1 => pb[3] += 1, // hero ahead on B
                                0 => pb[4] += 1,
                                _ => pb[5] += 1,
                            }
                            if (cmp_a == -1 && cmp_b == 1) || (cmp_a == 1 && cmp_b == -1) {
                                pb[6] += 1; // win exactly one
                            }
                            if cmp_a == 0 && cmp_b == 0 {
                                pb[7] += 1; // tie both
                            }
                            let share = 0.5 * ((cmp_a == -1) as u32 as f32)
                                + 0.25 * ((cmp_a == 0) as u32 as f32)
                                + 0.5 * ((cmp_b == -1) as u32 as f32)
                                + 0.25 * ((cmp_b == 0) as u32 as f32);
                            if share < g_min {
                                g_min = share;
                            }
                            if share > g_max {
                                g_max = share;
                            }
                            samples += 1;
                        }
                    }
                    if samples > 0 {
                        let inv = 1.0f32 / samples as f32;
                        for j in 0..8 {
                            out[PER_BOARD_OFF + j] = pb[j] as f32 * inv;
                        }
                        out[SHARE_OFF] = g_min;
                        out[SHARE_OFF + 1] = g_max;
                    }
                } else {
                    debug_assert!(n_unseen <= 64);
                    let mut pos = [0usize; 4];
                    for _ in 0..mc_samples {
                        let mut mask: u64 = 0;
                        let mut written = 0;
                        while written < k {
                            let i = (rng.next_u32() as usize) % n_unseen;
                            let bit = 1u64 << i;
                            if mask & bit == 0 {
                                mask |= bit;
                                // P1: record the drawn POSITION. The draw loop
                                // itself (RNG call count, modulo, rejection mask)
                                // is untouched — the sample stream stays
                                // byte-identical to the pre-P1 body.
                                pos[written] = i;
                                written += 1;
                            }
                        }
                        // P1: holding rank = max over its C(k,2) pairs of the
                        // stored pair ranks (exactly-2-of-k factorization; see
                        // the table comment above). Drawn positions are unordered
                        // while the table is filled for lo < hi only.
                        let mut opp_a = 0u32;
                        let mut opp_b = 0u32;
                        for p0 in 0..k {
                            for p1 in (p0 + 1)..k {
                                let (lo, hi) = if pos[p0] < pos[p1] {
                                    (pos[p0], pos[p1])
                                } else {
                                    (pos[p1], pos[p0])
                                };
                                let [ra, rb] = tab[lo * PAIR_STRIDE + hi];
                                if ra > opp_a {
                                    opp_a = ra;
                                }
                                if rb > opp_b {
                                    opp_b = rb;
                                }
                            }
                        }
                        let (cmp_a, cmp_b) = cmp_ranks(opp_a, opp_b);
                        tally_joint(&mut counters, cmp_a, cmp_b);
                        samples += 1;
                    }
                }

                if samples > 0 {
                    let inv = 1.0f32 / samples as f32;
                    let base = idx_k * 4;
                    out[base + SCOOP_OPP] = counters[SCOOP_OPP] as f32 * inv;
                    out[base + QUARTER_OPP] = counters[QUARTER_OPP] as f32 * inv;
                    out[base + SCOOP_HERO] = counters[SCOOP_HERO] as f32 * inv;
                    out[base + QUARTER_HERO] = counters[QUARTER_HERO] as f32 * inv;
                }
            }
        });
        out
    }

    /// 1024-sample MC convenience wrapper (serial / UI / eval path).
    /// See [`Self::opp_outcome_fractions_mc`].
    pub fn opp_outcome_fractions(&self) -> Vec<f32> {
        self.opp_outcome_fractions_mc(1024)
    }

    /// NLH single-board opponent-outcome fractions: the share of
    /// unseen-deck 2-card opponent combos currently AHEAD of / TIED with
    /// / BEHIND the hero (current actor) at the visible board, evaluated
    /// exhaustively (≤ C(47, 2) = 1081 combos) under the any-combo NLH
    /// rule. Current-rank dominance, no runout sampling — the same
    /// convention as the PLO opp-outcome feature. Exhaustive enumeration
    /// makes it exactly reproducible with no seed.
    ///
    /// Returns `[opp_ahead, tied, opp_behind]`. All-zero preflop, on
    /// terminal states, and for non-NLH variants.
    pub fn nlh_opp_outcome_fractions(&self) -> Vec<f32> {
        const N_OUT: usize = 3;
        if self.config.variant != Variant::NlhSingle {
            return vec![0.0; N_OUT];
        }
        let hero_seat = match self.actor {
            Some(s) => s,
            None => return vec![0.0; N_OUT],
        };
        nlh_opp_outcome_for(&self.hole_cards[hero_seat], &self.board_a).to_vec()
    }
}

/// Seed of the opp-outcome MC (`GameState::outcome_features_mc`) and its
/// batched cache key (`GameState::outcome_seed`): street + hero hole +
/// visible board A + visible board B, each hashed as a card SET.
///
/// PRODUCTION BEHAVIOR CHANGE (review 2026-09-20 B4): the seed used to be
/// `DefaultHasher` over (hero SEAT, street, hole in DEAL order, boards in
/// DEAL order), so the k=3/k=4 MC dims (obs 982-989) moved with the
/// absolute seat, with card order (user entry order in study mode), and
/// potentially with the Rust toolchain — against the encoder's
/// canonical-ordering and hero-relative rules. The evaluation itself only
/// ever depended on the sets; now the draws do too. Same distribution,
/// different bits for a given state: every checkpoint sees fresh MC noise
/// on those 8 dims.
pub(crate) fn outcome_mc_seed(
    street: Street,
    hero_hole: &[Card],
    board_a: &[Card],
    board_b: &[Card],
) -> u64 {
    let mut mixer = SeedMixer::new();
    mixer.write_u8(street.index() as u8);
    mixer.write_card_set(hero_hole);
    mixer.write_card_set(board_a);
    mixer.write_card_set(board_b);
    mixer.finish()
}

/// The (hole, board)-parameterized core of `nlh_opp_outcome_fractions`,
/// shared with the range-grid packer where candidate holes belong to no
/// engine state: the share of unseen-deck 2-card opponent combos AHEAD
/// of / TIED with / BEHIND `hole` at `board` under the any-combo NLH
/// rule (exhaustive, ≤ C(47, 2)). All-zero when the board has fewer
/// than 3 cards, matching the state method's preflop guard bit-exactly.
pub fn nlh_opp_outcome_for(hole: &[Card], board: &[Card]) -> [f32; 3] {
    const N_OUT: usize = 3;
    if board.len() < 3 {
        return [0.0; N_OUT];
    }
    let hero_rank = crate::hand_eval::evaluate_nlh(hole, board);

    let used = CardMask::of(hole.iter().chain(board.iter()));
    let mut unseen = [Card::from_index(0); 52];
    let mut m = 0;
    for c in used.unseen() {
        unseen[m] = c;
        m += 1;
    }
    let mut counters = [0u64; N_OUT];
    let mut total = 0u64;
    for i in 0..m {
        for j in (i + 1)..m {
            let opp = [unseen[i], unseen[j]];
            let opp_rank = crate::hand_eval::evaluate_nlh(&opp, board);
            let k = if opp_rank > hero_rank {
                0
            } else if opp_rank == hero_rank {
                1
            } else {
                2
            };
            counters[k] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return [0.0; N_OUT];
    }
    let inv = 1.0f32 / total as f32;
    [
        counters[0] as f32 * inv,
        counters[1] as f32 * inv,
        counters[2] as f32 * inv,
    ]
}

/// Any-combo NLH hand-category (0..=8) for an arbitrary (hole, board);
/// 0 when the board has fewer than 3 cards — the (hole, board) core of
/// `hero_category` for the NLH variant (same `rank >> 20` extraction).
pub fn nlh_category_for(hole: &[Card], board: &[Card]) -> u8 {
    if board.len() < 3 {
        return 0;
    }
    (crate::hand_eval::evaluate_nlh(hole, board) >> 20) as u8
}

/// Position stride of the MC's pair scratch: a pair of unseen-deck positions
/// (lo, hi) lives at `lo * PAIR_STRIDE + hi`; 52 covers every deck state.
const PAIR_STRIDE: usize = 52;

thread_local! {
    /// The opp-outcome MC's per-pair rank scratch ([board A, board B] per
    /// position pair), one per thread and reused by every call (PERF-033).
    static PAIR_SCRATCH: std::cell::RefCell<[[u32; 2]; PAIR_STRIDE * PAIR_STRIDE]> =
        const { std::cell::RefCell::new([[0; 2]; PAIR_STRIDE * PAIR_STRIDE]) };
}

/// Board-only pair ranks for the opp-outcome MC (see
/// `GameState::board_pair_table`): `ranks[pair_index(c0, c1)]` (c0 < c1,
/// neither on a board) = that 2-card holding's best PLO rank on board A and
/// board B, as [`pack_rank16`] values (exact; 0 for pairs touching a board).
/// One 5.3 KB block per table (PERF-026: was two 52 x 52 u32 tables, 21.6 KB;
/// the batched engine keeps one per env).
#[derive(Clone, Debug)]
pub struct BoardPairTable {
    pub key: [u8; 12],
    pub ranks: Box<[[u16; 2]; BoardPairTable::PAIRS]>,
}

impl BoardPairTable {
    /// Two-card holdings of a 52-card deck: C(52, 2).
    pub const PAIRS: usize = 52 * 51 / 2;

    /// The slot of the holding `{c0, c1}` (`c0 < c1 < 52`): pairs in
    /// lexicographic order.
    #[inline]
    pub fn pair_index(c0: u8, c1: u8) -> usize {
        debug_assert!(c0 < c1 && c1 < 52);
        Self::row_of(c0) + (c1 - c0 - 1) as usize
    }

    /// The slot of `{c0, c0 + 1}`: the holdings with first card `c0` follow it.
    #[inline]
    pub fn row_of(c0: u8) -> usize {
        let a = c0 as usize;
        a * 51 - a * a.saturating_sub(1) / 2
    }

    /// Both boards' cards in deal order + lengths (255-padded).
    pub fn key_of(board_a: &[Card], board_b: &[Card]) -> [u8; 12] {
        let mut k = [255u8; 12];
        for (j, c) in board_a.iter().enumerate().take(5) {
            k[j] = c.index();
        }
        for (j, c) in board_b.iter().enumerate().take(5) {
            k[5 + j] = c.index();
        }
        k[10] = board_a.len() as u8;
        k[11] = board_b.len() as u8;
        k
    }
}

#[cfg(test)]
mod outcome_mc_p1_tests {
    //! P1 bit-exactness harness. `outcome_features_mc_reference` is a FROZEN
    //! copy of the pre-pair-table function body (as of 2026-07-09). Do NOT
    //! "sync" it with the live function — its entire purpose is to pin that
    //! the P1 pair-table rewrite produces byte-identical output on every
    //! reachable state class: all PLO variants, all streets, rotated heroes,
    //! and the degenerate study-mode duplicate-card states that exercise the
    //! all-combos-filtered pair fallback (HandRank 0) and n_unseen > 41.
    use super::*;
    use crate::state::GameConfig;

    fn outcome_features_mc_reference(g: &GameState, mc_samples: usize) -> Vec<f32> {
        const N_OUT: usize = 20;
        const SCOOP_OPP: usize = 0;
        const QUARTER_OPP: usize = 1;
        const SCOOP_HERO: usize = 2;
        const QUARTER_HERO: usize = 3;
        const PER_BOARD_OFF: usize = 12;

        let hero_seat = match g.actor {
            Some(s) => s,
            None => return vec![0.0; N_OUT],
        };
        if g.board_a.len() < 3 || g.board_b.len() < 3 {
            return vec![0.0; N_OUT];
        }

        let hero_hole = &g.hole_cards[hero_seat];
        let hero_a = crate::hand_eval::evaluate_plo_partial(hero_hole, &g.board_a);
        let hero_b = crate::hand_eval::evaluate_plo_partial(hero_hole, &g.board_b);

        let mut used = [false; 52];
        for c in hero_hole.iter() {
            used[c.index() as usize] = true;
        }
        for c in g.board_a.iter().chain(g.board_b.iter()) {
            used[c.index() as usize] = true;
        }
        let unseen: Vec<Card> = (0..52u8)
            .filter(|&i| !used[i as usize])
            .map(Card::from_index)
            .collect();
        let n_unseen = unseen.len();

        // The ONE deliberate edit to this frozen body (review 2026-09-20
        // B4): the seed derivation moved to the pinned set-based
        // `outcome_mc_seed`, and the reference has to draw from the same
        // stream to stay comparable. Everything downstream of the seed is
        // still the untouched pre-P1 code.
        let seed = outcome_mc_seed(g.street, hero_hole, &g.board_a, &g.board_b);

        use rand_chacha::rand_core::{RngCore, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut out = vec![0.0f32; N_OUT];
        let mut opp_buf: Vec<Card> = Vec::with_capacity(4);

        let outcomes = |opp: &[Card]| -> (i8, i8) {
            let opp_a = crate::hand_eval::evaluate_plo_k_partial(opp, &g.board_a);
            let opp_b = crate::hand_eval::evaluate_plo_k_partial(opp, &g.board_b);
            let cmp_a: i8 = if opp_a > hero_a {
                1
            } else if opp_a < hero_a {
                -1
            } else {
                0
            };
            let cmp_b: i8 = if opp_b > hero_b {
                1
            } else if opp_b < hero_b {
                -1
            } else {
                0
            };
            (cmp_a, cmp_b)
        };
        let tally_joint = |counters: &mut [u32; 4], cmp_a: i8, cmp_b: i8| match (cmp_a, cmp_b) {
            (1, 1) => counters[SCOOP_OPP] += 1,
            (-1, -1) => counters[SCOOP_HERO] += 1,
            (1, 0) | (0, 1) => counters[QUARTER_OPP] += 1,
            (-1, 0) | (0, -1) => counters[QUARTER_HERO] += 1,
            _ => {}
        };

        for (idx_k, &k) in [2usize, 3, 4].iter().enumerate() {
            let mut counters = [0u32; 4];
            let mut samples: u32 = 0;

            if k == 2 {
                let mut pb = [0u32; 8];
                let mut idx: Vec<usize> = (0..k).collect();
                loop {
                    opp_buf.clear();
                    for &i in idx.iter() {
                        opp_buf.push(unseen[i]);
                    }
                    let (cmp_a, cmp_b) = outcomes(&opp_buf);
                    tally_joint(&mut counters, cmp_a, cmp_b);
                    match cmp_a {
                        -1 => pb[0] += 1,
                        0 => pb[1] += 1,
                        _ => pb[2] += 1,
                    }
                    match cmp_b {
                        -1 => pb[3] += 1,
                        0 => pb[4] += 1,
                        _ => pb[5] += 1,
                    }
                    if (cmp_a == -1 && cmp_b == 1) || (cmp_a == 1 && cmp_b == -1) {
                        pb[6] += 1;
                    }
                    if cmp_a == 0 && cmp_b == 0 {
                        pb[7] += 1;
                    }
                    samples += 1;
                    let mut pos = k;
                    let advanced = loop {
                        if pos == 0 {
                            break false;
                        }
                        pos -= 1;
                        if idx[pos] < n_unseen - (k - pos) {
                            idx[pos] += 1;
                            for j in (pos + 1)..k {
                                idx[j] = idx[j - 1] + 1;
                            }
                            break true;
                        }
                    };
                    if !advanced {
                        break;
                    }
                }
                if samples > 0 {
                    let inv = 1.0f32 / samples as f32;
                    for j in 0..8 {
                        out[PER_BOARD_OFF + j] = pb[j] as f32 * inv;
                    }
                }
            } else {
                debug_assert!(n_unseen <= 64);
                for _ in 0..mc_samples {
                    let mut mask: u64 = 0;
                    opp_buf.clear();
                    let mut written = 0;
                    while written < k {
                        let i = (rng.next_u32() as usize) % n_unseen;
                        let bit = 1u64 << i;
                        if mask & bit == 0 {
                            mask |= bit;
                            opp_buf.push(unseen[i]);
                            written += 1;
                        }
                    }
                    let (cmp_a, cmp_b) = outcomes(&opp_buf);
                    tally_joint(&mut counters, cmp_a, cmp_b);
                    samples += 1;
                }
            }

            if samples > 0 {
                let inv = 1.0f32 / samples as f32;
                let base = idx_k * 4;
                out[base + SCOOP_OPP] = counters[SCOOP_OPP] as f32 * inv;
                out[base + QUARTER_OPP] = counters[QUARTER_OPP] as f32 * inv;
                out[base + SCOOP_HERO] = counters[SCOOP_HERO] as f32 * inv;
                out[base + QUARTER_HERO] = counters[QUARTER_HERO] as f32 * inv;
            }
        }
        out
    }

    fn assert_bit_identical(g: &GameState, mc_samples: usize, tag: &str) {
        let new = g.outcome_features_mc(mc_samples);
        let reference = outcome_features_mc_reference(g, mc_samples);
        // The live fn appends the DUAL-4 share bounds (dims 20/21,
        // 2026-07-12); the frozen reference stays 20-wide by design. The
        // pin's purpose is unchanged: dims 0..20 byte-identical.
        assert_eq!(new.len(), 22, "{tag}: live output must be 22-wide");
        assert_eq!(reference.len(), 20, "{tag}: frozen reference is 20-wide");
        for (i, (x, y)) in new.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{tag}: dim {i} differs (new {x} vs ref {y})"
            );
        }
        // Appended share bounds: either both zero (inactive / no combos)
        // or a valid quantized min<=max pair.
        let (g_min, g_max) = (new[20], new[21]);
        assert!(
            (g_min == 0.0 && g_max == 0.0) || g_min <= g_max,
            "{tag}: share bounds invalid ({g_min}, {g_max})"
        );
        for v in [g_min, g_max] {
            assert!(
                [0.0, 0.25, 0.5, 0.75, 1.0].contains(&v),
                "{tag}: share bound {v} not on the quarter grid"
            );
        }
    }

    fn reveal_turn(g: &mut GameState) {
        g.board_a.push(g.full_board_a[3]);
        g.board_b.push(g.full_board_b[3]);
        g.street = Street::Turn;
    }

    fn reveal_river(g: &mut GameState) {
        g.board_a.push(g.full_board_a[4]);
        g.board_b.push(g.full_board_b[4]);
        g.street = Street::River;
    }

    fn variant_cfg(variant: Variant, num_seats: usize) -> GameConfig {
        GameConfig {
            num_seats,
            starting_stacks: vec![200_000; num_seats],
            ante: 30_000,
            bb: 10_000,
            sb: 0,
            variant,
            reach_cap: true,
        }
    }

    #[test]
    fn shared_pair_table_matches_outcome_features_mc() {
        // The batched packer's shared board table (2026-09-26) must change
        // nothing: every variant x street x seat count x hero, one table per
        // (hand, street) shared by all heroes, compared bit for bit with the
        // self-evaluating path -- plus a STALE table (another street's), which
        // must be ignored.
        let variants = [
            Variant::Plo5DoubleBomb,
            Variant::Plo4DoubleBomb,
            Variant::Plo6DoubleBomb,
        ];
        for &variant in variants.iter() {
            for &num_seats in &[2usize, 3, 6] {
                for seed in 0..4u64 {
                    let mut g =
                        GameState::new_hand(variant_cfg(variant, num_seats), 1000 + seed, 1);
                    let mut stale: Option<BoardPairTable> = None;
                    for street in 0..3usize {
                        if street == 1 {
                            reveal_turn(&mut g);
                        }
                        if street == 2 {
                            reveal_river(&mut g);
                        }
                        let table = g.board_pair_table().expect("both boards have a flop");
                        for hero in 0..num_seats {
                            g.actor = Some(hero);
                            for &mc in &[1usize, 64, 384] {
                                let a = g.outcome_features_mc(mc);
                                let b = g.outcome_features_mc_shared(mc, Some(&table));
                                let bits =
                                    |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                                assert_eq!(bits(&a), bits(&b), "{variant:?} seats {num_seats} seed {seed} street {street} hero {hero} mc {mc}");
                                if let Some(t) = stale.as_ref() {
                                    let c = g.outcome_features_mc_shared(mc, Some(t));
                                    assert_eq!(bits(&a), bits(&c), "stale table used");
                                }
                            }
                        }
                        stale = Some(table);
                    }
                }
            }
        }
    }

    #[test]
    fn pair_table_matches_reference() {
        // Production-shaped states: every PLO variant x street x seeds x seat
        // counts, with the hero rotated across every seat (the actor is the
        // only seat the function reads).
        let variants = [
            Variant::Plo5DoubleBomb,
            Variant::Plo4DoubleBomb,
            Variant::Plo6DoubleBomb,
        ];
        for (vi, &variant) in variants.iter().enumerate() {
            for &num_seats in &[2usize, 6] {
                for seed in 0..3u64 {
                    for street in 0..3usize {
                        let mut g = GameState::new_hand(
                            variant_cfg(variant, num_seats),
                            seed * 7919 + street as u64,
                            0,
                        );
                        if street >= 1 {
                            reveal_turn(&mut g);
                        }
                        if street >= 2 {
                            reveal_river(&mut g);
                        }
                        for hero in 0..num_seats {
                            g.actor = Some(hero);
                            assert_bit_identical(
                                &g,
                                64,
                                &format!(
                                    "variant#{vi} seats={num_seats} seed={seed} \
                                     street={street} hero={hero}"
                                ),
                            );
                        }
                    }
                }
            }
        }
        // Deployed sample count on one full-size case, plus the mc_samples=1
        // edge.
        let g = GameState::new_hand(variant_cfg(Variant::Plo5DoubleBomb, 6), 12345, 2);
        assert_bit_identical(&g, 384, "plo5 6max flop mc=384");
        assert_bit_identical(&g, 1, "mc=1");
        // mc_samples == 0 is NOT a reference case: since 289bf87 it is the
        // minimal-obs "skip the whole pass" switch and returns zeros without
        // running even the exhaustive k=2 arm, which the frozen body still
        // does. The old bit-identity pin here was stale (review 2026-09-20
        // C5); pin the contract the callers actually rely on instead.
        let skipped = g.outcome_features_mc(0);
        assert_eq!(skipped.len(), 22, "mc=0: output stays 22-wide");
        assert!(
            skipped.iter().all(|x| x.to_bits() == 0),
            "mc=0 must return all +0.0, got {skipped:?}"
        );
    }

    #[test]
    fn pair_table_matches_reference_degenerate_study_states() {
        // Duplicate-card states: hole cards colliding with board cards and
        // boards sharing cards shrink the `used` union (n_unseen past the 41
        // production ceiling, up to 46+) and exercise the degenerate-pair
        // fallback (all combos ck==0-filtered -> HandRank 0). Production
        // deals never reach these; the study path could (a user street card
        // landing on a villain placeholder) until review 2026-09-20 C8 made
        // it redraw the placeholder. Hand-built here to keep the pair-table
        // identity pinned over the function's whole domain.
        let mut g = GameState::new_hand(variant_cfg(Variant::Plo5DoubleBomb, 6), 99, 0);
        let hero = g.actor.unwrap();

        // Hero hole card duplicated onto board_a: at the flop there is exactly
        // one board triple, so every combo using that hole card degenerates.
        let mut g1 = g.clone();
        g1.board_a[0] = g1.hole_cards[hero][0];
        assert_bit_identical(&g1, 64, "hero hole card duplicated on board_a");

        // Boards sharing a card.
        let mut g2 = g.clone();
        g2.board_b[1] = g2.board_a[1];
        assert_bit_identical(&g2, 64, "board_a/board_b share a card");

        // Pathological mass duplication: several hero cards on both boards +
        // a cross-board duplicate (n_unseen well past 41).
        let mut g3 = g.clone();
        g3.board_a[0] = g3.hole_cards[hero][0];
        g3.board_a[1] = g3.hole_cards[hero][1];
        g3.board_b[0] = g3.hole_cards[hero][2];
        g3.board_b[1] = g3.hole_cards[hero][3];
        g3.board_b[2] = g3.board_a[2];
        assert_bit_identical(&g3, 64, "mass-duplicate study state");

        // INTRA-board duplicate: the only state class where an OPP pair's
        // evals can ALL degenerate (opp cards come from the unseen deck, so
        // they never collide with board cards — a 5-card combo can only
        // contain a duplicate if the board TRIPLE itself does). At the flop
        // board_a has exactly one triple, and it contains the dup, so every
        // opp pair's tab_a entry takes the all-combos-filtered fallback
        // (u16::MAX -> 7462 -> HandRank 0) — pinning the max-fold identity's
        // hardest case for real. Unreachable via study input validation
        // (DuplicateCard guard); pinned at the function level regardless.
        let mut g4 = g.clone();
        g4.board_a[1] = g4.board_a[0];
        assert_bit_identical(&g4, 64, "intra-board duplicate (all-degenerate pairs)");

        // Turn-street collision: 4-card board, so the colliding pair keeps
        // some valid triples (partial-degeneracy path).
        reveal_turn(&mut g);
        g.board_a[0] = g.hole_cards[hero][0];
        assert_bit_identical(&g, 64, "turn-street hole/board collision");
    }
}
