//! Golden digests of everything the engine hands to Python (TEST-021, with the
//! cross-machine scope of ENG-005).
//!
//! One pure-Rust test deals ~2,000 seeded hands across every variant and table
//! shape the engine serves or trains on — PLO4/5/6 at 2..=8 seats with deep,
//! short, uneven and sub-ante stacks, sitting-out masks, explicit-deck deals,
//! NLH (heads-up and 6-max), PLO5 and NLH study hands, PLO67 — plays them out
//! with a pinned pseudo-random policy (every gate, the legacy discrete sizings,
//! min/max/random raise sizes, all-ins, folds) and folds EVERY bit it would
//! hand to Python into one FNV-1a digest per section:
//!
//! - `deal`: every seat's hole (+ PLO67 extras), both full boards, burns;
//! - `pack_full`: the full packer (`pack_full`: MC outcome block with its
//!   per-(env, seat) cache and the shared pair tables, hero_board_v3,
//!   board_draw_v3, history...) + legal masks + hero categories;
//! - `full_r1` / `full_r2`: the 1171-dim full encoder under obs rev 1 / 2
//!   (every dim except the libm ones below);
//! - `pack_min`, `min_r1`, `min_r2`: the same for the 796-dim minimal layout;
//! - `nlh_pack`: the NLH packer (exhaustive opp-outcome sweep, blind seats);
//! - `payouts`, `payouts_ev`: terminal settlement (EV runouts at 24 samples);
//! - `study`: PLO5 and NLH study hands (placeholder deals, the redraw of a
//!   placeholder hit by a user-entered street card, both encoders, study
//!   terminals and payouts);
//! - `plo67`: PLO67 serial hands: burns, per-street hole counts, payouts,
//!   `plo67_runout_equities`.
//!
//! A change meant to be bit-exact (splitting bindings.rs, merging the two
//! encoders, a faster evaluator, fewer allocations) must leave EVERY digest
//! unchanged; the failure names the sections that moved. A deliberate output
//! change (a new obs revision, a rules fix) re-records them: run
//! `cargo test --release --lib golden -- --nocapture` and paste the printed
//! table into `EXPECTED` / `EXPECTED_LIBM`, saying why in the commit.
//!
//! ## What "bit-exact" means across machines (ENG-005)
//!
//! Every output is integer arithmetic or IEEE-754 `+ - * /` and conversions,
//! which are exact and identical on every x86-64 machine (Rust never fuses or
//! reorders float operations) — EXCEPT 21 dims of the full layout computed with
//! the platform maths library: 13 `ln_1p` sites and one `powf` (`LIBM_DIMS`).
//! `f64::ln_1p` / `powf` call the C library (the MSVC UCRT on the desktop,
//! glibc on the pod and the production server; numpy may use SVML on AVX-512
//! Linux), and those may round differently in the last bit. So those dims get
//! their own `full_r1_libm` / `full_r2_libm` digests with expected values
//! recorded PER TARGET: the rest must match everywhere, while a mismatch in
//! only the libm sections on a new machine is maths-library drift, not a code
//! change — record that target's values in `EXPECTED_LIBM` (a target with no
//! entry prints the entry to add; CI's Linux run shows it on its Summary page).
//! (The STK-6 bets-to-jam dim used to depend on `ln` too; it is an exact
//! comparison chain now — see `POW3` in bindings/encode_full.rs.)

use super::obs_layout::*;
use super::*;
use crate::cards::Deck;
use crate::state::Street;
use crate::test_util::Mix;

/// FNV-1a 64 — hand-written (and so pinned), unlike `DefaultHasher`.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xCBF2_9CE4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0 ^ x as u64).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.bytes(&v.to_le_bytes());
    }
    fn u8s(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.bytes(v);
    }
    fn i8s(&mut self, v: &[i8]) {
        self.u64(v.len() as u64);
        for &x in v {
            self.bytes(&[x as u8]);
        }
    }
    fn u64s(&mut self, v: &[u64]) {
        self.u64(v.len() as u64);
        for &x in v {
            self.u64(x);
        }
    }
    fn i64s(&mut self, v: &[i64]) {
        self.u64(v.len() as u64);
        for &x in v {
            self.i64(x);
        }
    }
    fn bools(&mut self, v: &[bool]) {
        self.u64(v.len() as u64);
        for &x in v {
            self.bytes(&[x as u8]);
        }
    }
    fn f32s(&mut self, v: &[f32]) {
        self.u64(v.len() as u64);
        for &x in v {
            self.bytes(&x.to_bits().to_le_bytes());
        }
    }
    fn f64s(&mut self, v: &[f64]) {
        self.u64(v.len() as u64);
        for &x in v {
            self.bytes(&x.to_bits().to_le_bytes());
        }
    }
    fn cards(&mut self, v: &[Card]) {
        self.u64(v.len() as u64);
        for c in v {
            self.bytes(&[c.index()]);
        }
    }
}

/// The 21 full-layout dims that go through the platform maths library.
fn libm_dims() -> Vec<usize> {
    let mut d = vec![EFF_PRICE_OFF + 3, EFF_PRICE_OFF + 4];
    d.extend(SPR_LOG_OFF..SPR_LOG_OFF + 8);
    d.extend(STK1_OFF..STK1_OFF + 3);
    d.extend(STK5_OFF..STK5_OFF + 4);
    d.extend([STK6_OFF + 1, STK7_OFF, STK10_OFF + 1, DUAL4_OFF + 4]);
    d
}

#[derive(Default)]
struct Sections {
    deal: Option<Fnv>,
    pack_full: Option<Fnv>,
    full_r1: Option<Fnv>,
    full_r2: Option<Fnv>,
    full_r1_libm: Option<Fnv>,
    full_r2_libm: Option<Fnv>,
    pack_min: Option<Fnv>,
    min_r1: Option<Fnv>,
    min_r2: Option<Fnv>,
    nlh_pack: Option<Fnv>,
    payouts: Option<Fnv>,
    payouts_ev: Option<Fnv>,
    study: Option<Fnv>,
    plo67: Option<Fnv>,
    cov: Coverage,
}

/// What the golden run actually reached — printed with the digests and held to
/// minimums, so a policy or config change cannot quietly hollow the test out.
#[derive(Default, Debug)]
struct Coverage {
    /// Full + minimal PLO rows encoded, by street (flop, turn, river).
    plo_rows: [usize; 3],
    /// NLH rows packed, by street (preflop, flop, turn, river).
    nlh_rows: [usize; 4],
    /// Opp-outcome MC memo (lookups, hits) summed over the PLO engines.
    mc_cache: (u64, u64),
    /// Terminal hands whose action closed before the river with 2+ seats left
    /// (the EV runout path) / that folded out.
    runouts: usize,
    fold_outs: usize,
    /// Study rows (PLO5 + NLH) and placeholder cards the redraw replaced.
    study_rows: usize,
    study_redraws: usize,
    /// PLO67 hands in which a red burn dealt extra cards.
    plo67_red: usize,
}

fn h(slot: &mut Option<Fnv>) -> &mut Fnv {
    slot.get_or_insert_with(Fnv::new)
}

fn digest_packed(d: &mut Fnv, p: &PackedObservation) {
    d.u8s(p.core.hero_hole.as_slice().unwrap());
    d.u8s(p.core.board_a.as_slice().unwrap());
    d.u8s(p.core.board_b.as_slice().unwrap());
    d.u8s(p.board_a_len.as_slice().unwrap());
    d.u8s(p.board_b_len.as_slice().unwrap());
    d.u8s(p.core.street.as_slice().unwrap());
    d.u64s(p.core.pot.as_slice().unwrap());
    d.u64s(p.core.stacks.as_slice().unwrap());
    d.bools(p.core.folded.as_slice().unwrap());
    d.bools(p.core.all_in.as_slice().unwrap());
    d.u64s(p.core.bet_to_call.as_slice().unwrap());
    d.u64s(p.core.street_commit.as_slice().unwrap());
    d.u64s(p.core.total_commit.as_slice().unwrap());
    d.u64s(p.core.min_bet.as_slice().unwrap());
    d.u64s(p.core.max_bet.as_slice().unwrap());
    d.u64s(p.core.min_raise.as_slice().unwrap());
    d.u64s(p.core.max_raise.as_slice().unwrap());
    d.u64s(p.core.eff_stack_cap.as_slice().unwrap());
    d.i8s(p.core.actor.as_slice().unwrap());
    d.u8s(p.core.button.as_slice().unwrap());
    d.i8s(p.last_aggressor.as_slice().unwrap());
    d.i8s(p.core.history_seat.as_slice().unwrap());
    d.i8s(p.core.history_action.as_slice().unwrap());
    d.u64s(p.core.history_chips.as_slice().unwrap());
    d.i8s(p.core.history_street.as_slice().unwrap());
    d.u8s(p.core.history_len.as_slice().unwrap());
    d.f32s(p.opp_outcome_fractions.as_slice().unwrap());
    d.f32s(p.per_board_outcome.as_slice().unwrap());
    d.f32s(p.share_bounds.as_slice().unwrap());
    d.bools(p.acted_this_street.as_slice().unwrap());
    d.u8s(p.hero_board_v3.as_slice().unwrap());
    d.u8s(p.board_draw_v3.as_slice().unwrap());
    d.i8s(p.sb_seat.as_slice().unwrap());
    d.i8s(p.bb_seat.as_slice().unwrap());
    d.f32s(p.nlh_opp_outcome.as_slice().unwrap());
}

fn digest_packed_min(d: &mut Fnv, p: &PackedCore) {
    d.u8s(p.hero_hole.as_slice().unwrap());
    d.u8s(p.board_a.as_slice().unwrap());
    d.u8s(p.board_b.as_slice().unwrap());
    d.u8s(p.street.as_slice().unwrap());
    d.u64s(p.pot.as_slice().unwrap());
    d.u64s(p.stacks.as_slice().unwrap());
    d.bools(p.folded.as_slice().unwrap());
    d.bools(p.all_in.as_slice().unwrap());
    d.u64s(p.bet_to_call.as_slice().unwrap());
    d.u64s(p.street_commit.as_slice().unwrap());
    d.u64s(p.total_commit.as_slice().unwrap());
    d.u64s(p.min_bet.as_slice().unwrap());
    d.u64s(p.max_bet.as_slice().unwrap());
    d.u64s(p.min_raise.as_slice().unwrap());
    d.u64s(p.max_raise.as_slice().unwrap());
    d.u64s(p.eff_stack_cap.as_slice().unwrap());
    d.i8s(p.actor.as_slice().unwrap());
    d.u8s(p.button.as_slice().unwrap());
    d.i8s(p.history_seat.as_slice().unwrap());
    d.i8s(p.history_action.as_slice().unwrap());
    d.u64s(p.history_chips.as_slice().unwrap());
    d.i8s(p.history_street.as_slice().unwrap());
    d.u8s(p.history_len.as_slice().unwrap());
}

fn digest_deal(d: &mut Fnv, g: &GameState) {
    for seat in 0..g.config.num_seats {
        d.cards(&g.hole_cards[seat]);
        d.cards(&g.extra_holes[seat]);
    }
    d.cards(&g.full_board_a);
    // Only DEALT cards: a single board's board B is an undealt slot (ENG-023).
    if g.config.variant.num_boards() == 2 {
        d.cards(&g.full_board_b);
    }
    d.cards(&g.full_burns);
    d.cards(&g.board_a);
    d.cards(&g.board_b);
    d.u64s(&g.stacks);
    d.u64s(&g.total_commit);
    d.u64(g.pot);
    d.i64(g.actor.map_or(-1, |a| a as i64));
}

/// One decision with the pinned policy: every gate, the legacy pot-fraction
/// actions, min / max / random raise sizes and all-ins all get exercised.
fn act(g: &mut GameState, rng: &mut Mix) {
    let mask = g.legal_action_mask();
    let (fold_ok, call_ok) = (g.fold_is_legal(), g.check_call_is_legal());
    let (min, max) = (g.min_raise_chips(), g.max_raise_chips());
    let raise_ok = min > 0 && max >= min;
    let allin_ok = mask[Action::AllIn as usize];
    let r = rng.below(100);
    if r < 11 && fold_ok {
        g.apply(Action::Fold);
    } else if r < 33 && raise_ok {
        let chips = match rng.below(4) {
            0 => min,
            1 => max,
            _ => min + rng.below(max - min + 1),
        };
        g.apply_raise_chips(chips).expect("raise inside [min, max]");
    } else if r < 41 {
        // A legal legacy discrete sizing (BetPct10..BetPct100), else a call.
        let sizes: Vec<Action> = (2u8..=6)
            .filter(|&k| mask[k as usize])
            .map(|k| Action::from_index(k).unwrap())
            .collect();
        if sizes.is_empty() {
            g.apply(if call_ok {
                Action::CheckCall
            } else {
                Action::Fold
            });
        } else {
            g.apply(sizes[rng.below(sizes.len() as u64) as usize]);
        }
    } else if r < 47 && allin_ok {
        g.apply(Action::AllIn);
    } else if call_ok {
        g.apply(Action::CheckCall);
    } else if fold_ok {
        g.apply(Action::Fold);
    } else if raise_ok {
        g.apply_raise_chips(min).unwrap();
    } else {
        assert!(allin_ok, "no legal action in a non-terminal state");
        g.apply(Action::AllIn);
    }
}

/// Pack + encode the live envs `idx` exactly as a rollout step does, both
/// layouts, both semantics revisions.
fn digest_step(s: &mut Sections, eng: &mut PyBatchedEngine, idx: &[usize], libm: &[usize]) {
    let n = idx.len();
    let full = eng.pack_for(Layout::Full, idx);
    let PackedRows::Full(PackedFull {
        obs: packed,
        legal,
        cat_a,
        cat_b,
    }) = &full
    else {
        unreachable!()
    };
    for &st in packed.core.street.iter() {
        if (1..=3).contains(&st) {
            s.cov.plo_rows[st as usize - 1] += 1;
        }
    }
    {
        let d = h(&mut s.pack_full);
        digest_packed(d, packed);
        d.bools(legal.as_slice().unwrap());
        d.u8s(cat_a);
        d.u8s(cat_b);
    }
    let mut is_libm = vec![false; OBS_DIM];
    for &x in libm {
        is_libm[x] = true;
    }
    let mut out = vec![1.5f32; n * OBS_DIM]; // dirty on purpose: rows are zeroed first
    for rev in [OBS_REV_LEGACY, OBS_REV_CURRENT] {
        eng.obs_rev = rev;
        {
            let rows: Vec<&mut [f32]> = out.chunks_exact_mut(OBS_DIM).collect();
            eng.encode_rows(&full, Some(rows), None, |_| true)
                .expect("no packing requested");
        }
        let (exact, lm) = if rev == OBS_REV_LEGACY {
            (&mut s.full_r1, &mut s.full_r1_libm)
        } else {
            (&mut s.full_r2, &mut s.full_r2_libm)
        };
        let (exact, lm) = (h(exact), h(lm));
        for row in out.chunks_exact(OBS_DIM) {
            for (dim, v) in row.iter().enumerate() {
                let bits = v.to_bits().to_le_bytes();
                if is_libm[dim] {
                    lm.bytes(&bits);
                } else {
                    exact.bytes(&bits);
                }
            }
        }
    }
    let minimal = eng.pack_for(Layout::Minimal, idx);
    {
        let PackedRows::Minimal(pm, legal_m) = &minimal else {
            unreachable!()
        };
        let d = h(&mut s.pack_min);
        digest_packed_min(d, pm);
        d.bools(legal_m.as_slice().unwrap());
    }
    let dm = obs_layout_minimal::OBS_DIM_MINIMAL;
    let mut outm = vec![-2.0f32; n * dm];
    for rev in [OBS_REV_LEGACY, OBS_REV_CURRENT] {
        eng.obs_rev = rev;
        let rows: Vec<&mut [f32]> = outm.chunks_exact_mut(dm).collect();
        eng.encode_rows(&minimal, Some(rows), None, |_| true)
            .expect("no packing requested");
        let d = if rev == OBS_REV_LEGACY {
            h(&mut s.min_r1)
        } else {
            h(&mut s.min_r2)
        };
        d.f32s(&outm);
    }
    eng.obs_rev = OBS_REV_CURRENT;
}

fn config(variant: Variant, stacks: &[u64], ante: u64, bb: u64, sb: u64) -> GameConfig {
    GameConfig {
        num_seats: stacks.len(),
        starting_stacks: stacks.to_vec(),
        ante,
        bb,
        sb,
        variant,
    }
}

/// Deal kind for a batched case: plain seeded, with a sitting-out mask, or from
/// an explicit deck order (the home games' verifiable shuffle).
#[derive(Clone, Copy)]
enum DealKind {
    Seeded,
    Masked,
    FromDeck,
}

fn deal(cfg: &GameConfig, kind: DealKind, seed: u64, button: usize, rng: &mut Mix) -> GameState {
    let n = cfg.num_seats;
    match kind {
        DealKind::Seeded => GameState::new_hand(cfg.clone(), seed, button),
        DealKind::Masked => {
            let mut mask: Vec<bool> = (0..n).map(|_| rng.below(4) != 0).collect();
            // At least two seats in (the dealer's and the next one).
            mask[button] = true;
            mask[(button + 1) % n] = true;
            GameState::new_hand_with_mask(cfg.clone(), seed, button, Some(mask))
        }
        DealKind::FromDeck => {
            let mut order: Vec<u8> = (0..52).collect();
            for i in (1..52).rev() {
                let j = rng.below(i as u64 + 1) as usize;
                order.swap(i, j);
            }
            GameState::new_hand_from_deck(
                cfg.clone(),
                Deck::from_order(&order).unwrap(),
                button,
                None,
            )
        }
    }
}

/// Play `waves` rounds of `envs` hands on one batched engine (hands are
/// re-dealt as a rollout does: cache slots of a re-dealt env are cleared).
fn run_batched(
    s: &mut Sections,
    cfg: &GameConfig,
    kind: DealKind,
    envs: usize,
    waves: usize,
    case_seed: u64,
    libm: &[usize],
) {
    let mut rng = Mix(case_seed);
    let mut eng = PyBatchedEngine::with_config(cfg.clone(), envs, 96, OBS_REV_CURRENT);
    let n = cfg.num_seats;
    let is_nlh = cfg.variant == Variant::NlhSingle;
    for wave in 0..waves {
        for e in 0..envs {
            let seed = case_seed ^ ((wave * envs + e) as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
            let g = deal(cfg, kind, seed, (e + wave) % n, &mut rng);
            digest_deal(h(&mut s.deal), &g);
            eng.states[e] = Some(g);
            eng.outcome_cache.get_mut().unwrap().clear_env(e, n);
        }
        loop {
            let live: Vec<usize> = (0..envs)
                .filter(|&e| !eng.states[e].as_ref().unwrap().is_terminal())
                .collect();
            if live.is_empty() {
                break;
            }
            if is_nlh {
                let PackedFull {
                    obs: packed,
                    legal,
                    cat_a,
                    cat_b,
                } = eng.pack_full(&live);
                for &st in packed.core.street.iter() {
                    s.cov.nlh_rows[(st as usize).min(3)] += 1;
                }
                let d = h(&mut s.nlh_pack);
                digest_packed(d, &packed);
                d.bools(legal.as_slice().unwrap());
                d.u8s(&cat_a);
                d.u8s(&cat_b);
            } else {
                digest_step(s, &mut eng, &live, libm);
            }
            for &e in &live {
                act(eng.states[e].as_mut().unwrap(), &mut rng);
            }
        }
        for e in 0..envs {
            let g = eng.states[e].as_ref().unwrap();
            if g.folded.iter().filter(|&&f| !f).count() == 1 {
                s.cov.fold_outs += 1;
            } else if g.action_close_board_len.is_some_and(|l| l < 5) {
                s.cov.runouts += 1;
            }
            h(&mut s.payouts).i64s(&g.payouts());
            let ev_seed = (case_seed.wrapping_add(e as u64)) ^ 0x9E37_79B9_7F4A_7C15;
            h(&mut s.payouts_ev).i64s(&g.payouts_ev(24, ev_seed));
            h(&mut s.payouts_ev).u8s(&[g.street.index() as u8, g.board_a.len() as u8]);
        }
    }
    let cache = eng.outcome_cache.get_mut().unwrap();
    s.cov.mc_cache.0 += cache.lookups;
    s.cov.mc_cache.1 += cache.hits;
}

/// A card for a study street: anything not already visible to the user (hero
/// hole + boards so far). Hitting a hidden placeholder is allowed on purpose —
/// it exercises the placeholder redraw.
fn study_card(g: &GameState, hero: usize, extra: &[Card], rng: &mut Mix) -> Card {
    loop {
        let c = Card(rng.below(52) as u8);
        let used = g.hole_cards[hero].contains(&c)
            || g.board_a.contains(&c)
            || g.board_b.contains(&c)
            || extra.contains(&c);
        if !used {
            return c;
        }
    }
}

fn run_study(s: &mut Sections, libm: &[usize]) {
    let mut rng = Mix(0x5747_0D1E);
    // PLO5 study hands (4-6 seats, some with a sitting-out seat), encoded with
    // both layouts through a one-env engine at every decision.
    for hand in 0..64u64 {
        let n = 4 + (hand % 3) as usize;
        let stacks: Vec<u64> = (0..n).map(|_| 20_000 + rng.below(600_000)).collect();
        let cfg = config(Variant::Plo5DoubleBomb, &stacks, 30_000, 10_000, 0);
        let mut pick: Vec<u8> = (0..52).collect();
        for i in (1..52).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            pick.swap(i, j);
        }
        let hole: [Card; 5] = std::array::from_fn(|i| Card(pick[i]));
        let fa: [Card; 3] = std::array::from_fn(|i| Card(pick[5 + i]));
        let fb: [Card; 3] = std::array::from_fn(|i| Card(pick[8 + i]));
        let button = (hand as usize) % n;
        let hero = (hand as usize * 7 + 1) % n;
        let mask = (hand % 4 == 3).then(|| {
            let mut m = vec![true; n];
            m[(hero + 1) % n] = false;
            m
        });
        let mut g = GameState::new_study_with_mask(cfg.clone(), button, hero, hole, fa, fb, mask)
            .expect("valid study spot");
        let mut eng = PyBatchedEngine::with_config(cfg, 1, 96, OBS_REV_CURRENT);
        loop {
            if let Some(street) = g.awaiting_next_street {
                let a = study_card(&g, hero, &[], &mut rng);
                let b = study_card(&g, hero, &[a], &mut rng);
                let before = g.hole_cards.concat();
                match street {
                    Street::Turn => g.set_turn(a, b).unwrap(),
                    _ => g.set_river(a, b).unwrap(),
                }
                let after = g.hole_cards.concat();
                s.cov.study_redraws += before.iter().zip(&after).filter(|(x, y)| x != y).count();
                h(&mut s.study).cards(&after);
                continue;
            }
            if g.is_terminal() {
                break;
            }
            s.cov.study_rows += 1;
            eng.states[0] = Some(g.clone());
            // Study rows go through the same section digests as training rows
            // would, but into their own `study` section: a separate engine
            // per hand keeps the MC cache out of it.
            let mut sub = Sections::default();
            digest_step(&mut sub, &mut eng, &[0], libm);
            let d = h(&mut s.study);
            for sec in [
                sub.pack_full,
                sub.full_r1,
                sub.full_r2,
                sub.pack_min,
                sub.min_r1,
                sub.min_r2,
            ] {
                d.u64(sec.map_or(0, |f| f.0));
            }
            // libm dims of study rows join the per-target libm digests.
            for (dst, src) in [
                (&mut s.full_r1_libm, sub.full_r1_libm),
                (&mut s.full_r2_libm, sub.full_r2_libm),
            ] {
                h(dst).u64(src.map_or(0, |f| f.0));
            }
            act(&mut g, &mut rng);
        }
        let d = h(&mut s.study);
        d.bytes(&[g.study_terminal.map_or(0, |t| t as u8 + 1)]);
        d.i64s(&g.payouts());
        d.cards(&g.board_a);
        d.cards(&g.board_b);
    }
    // NLH study hands: preflop entry, user-supplied flop/turn/river.
    for hand in 0..48u64 {
        let n = 2 + (hand % 5) as usize;
        let stacks: Vec<u64> = (0..n).map(|_| 400_000 + rng.below(2_200_000)).collect();
        let cfg = config(Variant::NlhSingle, &stacks, 5_000, 10_000, 5_000);
        let a = Card(rng.below(52) as u8);
        let b = loop {
            let c = Card(rng.below(52) as u8);
            if c != a {
                break c;
            }
        };
        let button = (hand as usize) % n;
        let hero = (hand as usize * 3) % n;
        let mut g = GameState::new_study_nlh(cfg.clone(), button, hero, [a, b]).expect("valid");
        let mut eng = PyBatchedEngine::with_config(cfg, 1, 96, OBS_REV_CURRENT);
        loop {
            if let Some(street) = g.awaiting_next_street {
                let before = g.hole_cards.concat();
                match street {
                    Street::Flop => {
                        let c0 = study_card(&g, hero, &[], &mut rng);
                        let c1 = study_card(&g, hero, &[c0], &mut rng);
                        let c2 = study_card(&g, hero, &[c0, c1], &mut rng);
                        g.set_flop_nlh([c0, c1, c2]).unwrap();
                    }
                    Street::Turn => g.set_turn_nlh(study_card(&g, hero, &[], &mut rng)).unwrap(),
                    _ => g
                        .set_river_nlh(study_card(&g, hero, &[], &mut rng))
                        .unwrap(),
                }
                let after = g.hole_cards.concat();
                s.cov.study_redraws += before.iter().zip(&after).filter(|(x, y)| x != y).count();
                h(&mut s.study).cards(&after);
                continue;
            }
            if g.is_terminal() {
                break;
            }
            s.cov.study_rows += 1;
            eng.states[0] = Some(g.clone());
            let PackedFull {
                obs: packed,
                legal,
                cat_a,
                cat_b,
            } = eng.pack_full(&[0]);
            let d = h(&mut s.study);
            digest_packed(d, &packed);
            d.bools(legal.as_slice().unwrap());
            d.u8s(&cat_a);
            d.u8s(&cat_b);
            act(&mut g, &mut rng);
        }
        let d = h(&mut s.study);
        d.bytes(&[g.study_terminal.map_or(0, |t| t as u8 + 1)]);
        d.i64s(&g.payouts());
    }
}

fn run_plo67(s: &mut Sections) {
    let mut rng = Mix(0x0067_0067);
    for hand in 0..160u64 {
        let n = 2 + (hand % 4) as usize; // 2..=5 seats
        let stacks: Vec<u64> = (0..n).map(|_| 5_000 + rng.below(400_000)).collect();
        let cfg = config(Variant::Plo67DoubleBomb, &stacks, 10_000, 10_000, 0);
        let mut g = GameState::new_hand(
            cfg,
            hand.wrapping_mul(0x9E37_79B9) ^ 0x67,
            (hand as usize) % n,
        );
        digest_deal(h(&mut s.plo67), &g);
        let mut last_street = None;
        loop {
            let d = h(&mut s.plo67);
            d.cards(&g.burns);
            for seat in 0..n {
                d.cards(&g.hole_cards[seat]);
                for st in [Street::Flop, Street::Turn, Street::River] {
                    d.u64(g.hole_count_on(seat, st) as u64);
                }
            }
            if g.is_terminal() {
                break;
            }
            if last_street != Some(g.street) {
                // Street start: the all-in runout equities of the live hands.
                last_street = Some(g.street);
                let live: Vec<Vec<Card>> = (0..n)
                    .filter(|&i| !g.folded[i])
                    .map(|i| g.hole_cards[i].clone())
                    .collect();
                let eq = crate::engine::plo67_runout_equities(
                    &live,
                    &g.board_a,
                    &g.board_b,
                    &g.burns,
                    200,
                    hand ^ 0xE0,
                )
                .unwrap();
                for e in eq {
                    h(&mut s.plo67).f64s(&e);
                }
            }
            act(&mut g, &mut rng);
        }
        if g.hole_cards.iter().any(|hole| hole.len() > 4) {
            s.cov.plo67_red += 1;
        }
        let d = h(&mut s.plo67);
        d.i64s(&g.payouts());
        d.i64s(&g.payouts_ev(24, hand ^ 0x9E37_79B9_7F4A_7C15));
        d.cards(&g.board_a);
        d.cards(&g.board_b);
    }
}

fn compute() -> (Vec<(&'static str, u64)>, Coverage) {
    let libm = libm_dims();
    let mut s = Sections::default();
    let p5 = Variant::Plo5DoubleBomb;
    let cases: [(GameConfig, DealKind, usize, usize); 8] = [
        // The training default table.
        (
            config(p5, &[200_000; 6], 30_000, 10_000, 0),
            DealKind::Seeded,
            48,
            3,
        ),
        // Heads-up, very uneven.
        (
            config(p5, &[150_000, 900_000], 30_000, 10_000, 0),
            DealKind::Seeded,
            48,
            3,
        ),
        // 8-max with a sub-ante stack (all-in at the deal) and a spread.
        (
            config(
                p5,
                &[
                    40_000, 200_000, 1_000_000, 25_000, 300_000, 90_000, 500_000, 60_000,
                ],
                30_000,
                10_000,
                0,
            ),
            DealKind::Seeded,
            40,
            3,
        ),
        // Sitting-out masks (serving) and explicit-deck deals (home games).
        (
            config(
                p5,
                &[120_000, 400_000, 70_000, 250_000, 800_000],
                30_000,
                10_000,
                0,
            ),
            DealKind::Masked,
            40,
            3,
        ),
        (
            config(p5, &[200_000, 350_000, 50_000, 610_000], 20_000, 10_000, 0),
            DealKind::FromDeck,
            40,
            3,
        ),
        // PLO4 and PLO6 (hole widths 4 and 6), odd bb.
        (
            config(
                Variant::Plo4DoubleBomb,
                &[1_000_000, 300_000, 450_000, 90_000, 2_000_000],
                10_000,
                7_000,
                0,
            ),
            DealKind::Seeded,
            40,
            3,
        ),
        (
            config(
                Variant::Plo6DoubleBomb,
                &[200_000, 180_000, 90_000, 500_000, 260_000, 30_000, 400_000],
                30_000,
                10_000,
                0,
            ),
            DealKind::Seeded,
            40,
            3,
        ),
        (
            config(Variant::Plo6DoubleBomb, &[200_000; 3], 30_000, 10_000, 0),
            DealKind::Masked,
            40,
            2,
        ),
    ];
    for (k, (cfg, kind, envs, waves)) in cases.iter().enumerate() {
        run_batched(
            &mut s,
            cfg,
            *kind,
            *envs,
            *waves,
            0xC0DE_0000 + k as u64,
            &libm,
        );
    }
    let nlh = Variant::NlhSingle;
    let nlh_cases: [(GameConfig, DealKind); 3] = [
        (
            config(
                nlh,
                &[
                    1_000_000, 2_500_000, 1_800_000, 1_200_000, 3_000_000, 900_000,
                ],
                5_000,
                10_000,
                5_000,
            ),
            DealKind::Seeded,
        ),
        (
            config(nlh, &[1_500_000, 400_000], 5_000, 10_000, 5_000),
            DealKind::Seeded,
        ),
        // Short stacks: blinds / antes that put seats all-in at the deal.
        (
            config(
                nlh,
                &[8_000, 1_000_000, 12_000, 600_000],
                5_000,
                10_000,
                5_000,
            ),
            DealKind::Seeded,
        ),
    ];
    for (k, (cfg, kind)) in nlh_cases.iter().enumerate() {
        run_batched(&mut s, cfg, *kind, 48, 2, 0x0A1B_0000 + k as u64, &libm);
    }
    run_study(&mut s, &libm);
    run_plo67(&mut s);
    let v = |f: Option<Fnv>| f.map_or(0, |f| f.0);
    let digests = vec![
        ("deal", v(s.deal)),
        ("pack_full", v(s.pack_full)),
        ("full_r1", v(s.full_r1)),
        ("full_r2", v(s.full_r2)),
        ("pack_min", v(s.pack_min)),
        ("min_r1", v(s.min_r1)),
        ("min_r2", v(s.min_r2)),
        ("nlh_pack", v(s.nlh_pack)),
        ("payouts", v(s.payouts)),
        ("payouts_ev", v(s.payouts_ev)),
        ("study", v(s.study)),
        ("plo67", v(s.plo67)),
        ("full_r1_libm", v(s.full_r1_libm)),
        ("full_r2_libm", v(s.full_r2_libm)),
    ];
    (digests, s.cov)
}

/// Sections every machine must reproduce exactly.
const EXPECTED: &[(&str, u64)] = &[
    ("deal", 0xBAB43BE5E8759A47),
    ("pack_full", 0xFCC1933D0209E405),
    ("full_r1", 0xDB3D70105A9FE3CF),
    ("full_r2", 0x24B229E9F16B53A4),
    ("pack_min", 0xDD378A664F6395A7),
    ("min_r1", 0xFD4429BDB89440B9),
    ("min_r2", 0x703DC92C672676A9),
    ("nlh_pack", 0x5BF0425EF37D1013),
    ("payouts", 0xB6EE1F76B702E097),
    ("payouts_ev", 0xD5EE7DB6F3F9F10A),
    ("study", 0x24597BBC745FE2A4),
    ("plo67", 0x15D2076E823FF38C),
];

/// The libm-dependent sections, recorded per target (see the module docs):
/// `(target, [(section, digest)])`. A target with no entry passes and prints the
/// exact entry to add (under GitHub Actions also on the run's Summary page), so
/// CI's Linux run hands over the values the pod and the server should match.
const EXPECTED_LIBM: &[(&str, &[(&str, u64)])] = &[(
    "x86_64-windows-msvc",
    &[
        ("full_r1_libm", 0xC2206BE39B355517),
        ("full_r2_libm", 0x66BC9BF844F3702F),
    ],
)];

/// This build's target as `EXPECTED_LIBM` names it: the maths library follows
/// the architecture, the OS and the C runtime.
fn libm_target() -> String {
    let env = if cfg!(target_env = "msvc") {
        "msvc"
    } else if cfg!(target_env = "gnu") {
        "gnu"
    } else if cfg!(target_env = "musl") {
        "musl"
    } else {
        "other"
    };
    format!("{}-{}-{env}", std::env::consts::ARCH, std::env::consts::OS)
}

/// No libm digests recorded for this target: say exactly what to add, on stderr
/// and (under GitHub Actions) on the run's Summary page.
fn report_unrecorded_libm(target: &str, lookup: &dyn Fn(&str) -> u64) {
    let entry = format!(
        "(\"{target}\", &[(\"full_r1_libm\", 0x{:016X}), (\"full_r2_libm\", 0x{:016X})]),",
        lookup("full_r1_libm"),
        lookup("full_r2_libm")
    );
    eprintln!(
        "golden: no libm digests recorded for {target}. If this machine is trusted, add \
         this entry to EXPECTED_LIBM in rust_engine/src/bindings/golden_tests.rs:\n    {entry}"
    );
    if let Ok(summary) = std::env::var("GITHUB_STEP_SUMMARY") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(summary)
        {
            let _ = writeln!(
                f,
                "### Golden digests: libm values for `{target}`\n\nNo libm digests are \
                 recorded for this target yet (ENG-005). Add this entry to `EXPECTED_LIBM` \
                 in `rust_engine/src/bindings/golden_tests.rs`:\n\n```rust\n{entry}\n```\n"
            );
        }
    }
}

#[test]
fn golden_digests_of_every_engine_output() {
    let (got, cov) = compute();
    println!("golden coverage: {cov:?}");
    // The run must keep reaching every street, the EV runout path, the MC
    // memo, the study redraw and PLO67's red burns.
    assert!(cov.plo_rows.iter().all(|&r| r >= 500), "{cov:?}");
    assert!(cov.nlh_rows.iter().all(|&r| r >= 100), "{cov:?}");
    assert!(cov.mc_cache.1 >= 500, "{cov:?}");
    assert!(cov.runouts >= 50 && cov.fold_outs >= 50, "{cov:?}");
    assert!(cov.study_rows >= 300 && cov.study_redraws >= 5, "{cov:?}");
    assert!(cov.plo67_red >= 40, "{cov:?}");
    let table: String = got
        .iter()
        .map(|(k, v)| format!("    (\"{k}\", 0x{v:016X}),\n"))
        .collect();
    println!("golden digests (this machine):\n{table}");
    let lookup = |name: &str| {
        got.iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| *v)
            .unwrap()
    };
    let moved: Vec<&str> = EXPECTED
        .iter()
        .filter(|(k, v)| lookup(k) != *v)
        .map(|(k, _)| *k)
        .collect();
    let target = libm_target();
    let libm_moved: Vec<&str> = match EXPECTED_LIBM.iter().find(|(t, _)| *t == target) {
        Some((_, exp)) => exp
            .iter()
            .filter(|(k, v)| lookup(k) != *v)
            .map(|(k, _)| *k)
            .collect(),
        None => {
            report_unrecorded_libm(&target, &lookup);
            Vec::new()
        }
    };
    assert!(
        moved.is_empty() && libm_moved.is_empty(),
        "engine outputs changed in {moved:?}{}; if the change is deliberate, re-record \
         (see the module docs). Current values:\n{table}",
        if libm_moved.is_empty() {
            String::new()
        } else {
            format!(
                " and the libm dims {libm_moved:?} (if ONLY those moved on a new machine, \
                 that is maths-library drift, ENG-005 -- record this target)"
            )
        }
    );
}
