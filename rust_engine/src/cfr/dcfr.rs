//! Discounted CFR for HU postflop roots (river exact; flop/turn with sampled
//! runouts), chance-sampled over the hole-card deal.
//!
//! Each iteration samples ONE deal `(runout, c0, c1)` from the true joint
//! chance distribution and runs a full public-tree CFR pass for both players.
//! Exploitability is reported by a vectorized infoset best response
//! ([`RiverSolver::evaluate`]) — exact on river roots, an expectation over the
//! runout on turn/flop roots.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::actions::{apply_abstract, legal_actions, AbstractAction};
use super::infoset::{
    Infoset, InfosetDump, InfosetKey, PRIV_COMBO, PRIV_OCHS_BUCKET,
};
use super::public_state::PublicState;
use super::range::{combo_ranks_on_board, combo_table, Range, NUM_COMBOS};
use super::types::{
    Discounting, InfosetStrategy, RootSpec, SolveConfig, SolveReport, Strategy, StreetRoot,
};
use super::CfrError;

/// Hash public history from action sequence labels **and** public board.
/// Board must be in the key so flop/turn runouts get distinct infosets.
fn history_hash(actions: &[AbstractAction], board: &[u8], board_len: u8) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for a in actions {
        a.label().hash(&mut h);
    }
    board_len.hash(&mut h);
    for i in 0..board_len as usize {
        if i < board.len() {
            board[i].hash(&mut h);
        }
    }
    h.finish()
}

struct RiverSolver {
    raise_sizes_pm: Vec<u32>,
    allin_atom: bool,
    /// Full 5-card board when known at root (river); partial otherwise.
    board_root: Vec<u8>,
    /// Precomputed ranks only when board_root.len()==5.
    ranks: Option<Vec<u32>>,
    /// Validated once in `new` (review 2026-09-20 E1: `root_state()` used to
    /// `unwrap()` a constructor error → `PanicException` in Python).
    root: PublicState,
    pot0: u64,
    stack0: u64,
    bb: u64,
    infosets: HashMap<InfosetKey, Infoset>,
    /// Sampled full board for current iteration (len 5).
    sample_board: [u8; 5],
    sample_ranks: Vec<u32>,
    /// Optional flop/turn bucket map: combo → bucket id (u16::MAX = blocked).
    buckets: Option<Vec<u16>>,
    use_buckets: bool,
    use_isomorphism: bool,
}

impl RiverSolver {
    fn new(root: &RootSpec, config: &SolveConfig) -> Result<Self, CfrError> {
        let blen = root.board.len();
        if !(3..=5).contains(&blen) {
            return Err(CfrError::InvalidRoot(
                "postflop board must have 3..=5 cards".into(),
            ));
        }
        let board_root = root.board.clone();
        let ranks = if blen == 5 {
            let mut b = [0u8; 5];
            b.copy_from_slice(&root.board[..5]);
            Some(combo_ranks_on_board(&b))
        } else {
            None
        };
        // Checked conversions: a pot/stack that rounds to 0 chips is an error.
        let pot0 = root.pot_chips()?;
        let stack0 = root.effective_stack_chips()?;
        let root_state = PublicState::hu_postflop_root(pot0, stack0, &board_root, root.bb_chips)?;

        // Wire card abstraction for flop (and optional turn)
        let abs = config.card_abstraction.as_str();
        let want_buckets = matches!(abs, "ochs" | "buckets" | "ehs")
            || (root.street == StreetRoot::Flop && abs != "exact");
        // Keep MC samples modest so solve stays interactive (OCHS-style
        // histogram still uses 16 refs × samples runouts).
        let buckets = if want_buckets && blen == 3 {
            let mut b3 = [0u8; 3];
            b3.copy_from_slice(&root.board[..3]);
            Some(super::card_abs::flop_equity_buckets(&b3, 4, config.seed))
        } else if want_buckets && blen == 4 {
            let mut b3 = [0u8; 3];
            b3.copy_from_slice(&root.board[..3]);
            Some(super::card_abs::flop_equity_buckets(&b3, 3, config.seed))
        } else {
            None
        };
        let use_buckets = buckets.is_some();

        Ok(Self {
            raise_sizes_pm: root.raise_sizes_pm.clone(),
            allin_atom: root.allin_atom,
            board_root,
            ranks,
            root: root_state,
            pot0,
            stack0,
            bb: root.bb_chips,
            infosets: HashMap::new(),
            sample_board: [0; 5],
            sample_ranks: vec![0; NUM_COMBOS],
            buckets,
            use_buckets,
            use_isomorphism: config.use_isomorphism,
        })
    }

    /// Infoset private view of `combo` given the PUBLIC board dealt so far.
    ///
    /// (review 2026-09-20 D15) The suit canonicalization used to read the
    /// iteration's full *sampled* board, so on turn roots the suit of the
    /// not-yet-dealt river leaked into turn-street infoset keys (the same
    /// hand landed in different infosets depending on a future card). It
    /// must only ever see `public_board`.
    fn private_view(&self, combo: usize, public_board: &[u8]) -> u32 {
        if self.use_buckets {
            if let Some(ref b) = self.buckets {
                let bid = b.get(combo).copied().unwrap_or(u16::MAX);
                if bid != u16::MAX {
                    return super::card_abs::OCHS_BUCKET_BASE + bid as u32;
                }
            }
        }
        if self.use_isomorphism {
            return super::card_abs::iso_combo_id(combo, public_board);
        }
        combo as u32
    }

    /// (private_kind, raw_combo, iso_id) for the dump schema.
    fn private_meta(
        &self,
        combo: usize,
        priv_id: u32,
        public_board: &[u8],
    ) -> (&'static str, Option<u32>, Option<u32>) {
        let raw = Some(combo as u32);
        if self.use_buckets {
            let iso = if self.use_isomorphism {
                Some(super::card_abs::iso_combo_id(combo, public_board))
            } else {
                None
            };
            return (PRIV_OCHS_BUCKET, raw, iso);
        }
        if self.use_isomorphism {
            return (PRIV_COMBO, raw, Some(priv_id));
        }
        (PRIV_COMBO, raw, None)
    }

    fn root_state(&self) -> PublicState {
        self.root.clone()
    }

    fn set_sample_board(&mut self, full: [u8; 5]) {
        self.sample_board = full;
        if self.board_root.len() < 5 {
            self.sample_ranks = combo_ranks_on_board(&full);
        }
    }

    fn active_ranks(&self) -> &[u32] {
        if self.board_root.len() == 5 {
            self.ranks.as_ref().unwrap()
        } else {
            &self.sample_ranks
        }
    }

    /// Chip EV of `seat` at a terminal for a fixed deal (O(1); HU only).
    fn terminal_value(&self, state: &PublicState, seat: usize, combos: [usize; 2]) -> f64 {
        if state.alive_count() == 1 {
            return state.fold_payout_chips(seat) as f64;
        }
        let ranks = self.active_ranks();
        let (rh, ro) = (ranks[combos[seat]], ranks[combos[1 - seat]]);
        let eq = if rh > ro {
            1.0
        } else if rh == ro {
            0.5
        } else {
            0.0
        };
        eq * state.pot as f64 - state.total_commit[seat] as f64
    }

    /// CFR return: expected chip EV for the **traversing player** at this node,
    /// given fixed private combos for both seats and a fixed sample board.
    fn cfr(
        &mut self,
        state: &PublicState,
        history: &[AbstractAction],
        combos: [usize; 2],
        reach: [f64; 2],
        traverser: usize,
    ) -> f64 {
        // Chance runout: betting closed mid-street
        if state.needs_runout() {
            let mut child = state.clone();
            let card = self.sample_board[state.board_len as usize];
            debug_assert!(!combo_has_card(combos[0], card) && !combo_has_card(combos[1], card));
            child.deal_board_card(card);
            return self.cfr(&child, history, combos, reach, traverser);
        }

        if state.is_terminal() || state.actor.is_none() {
            return self.terminal_value(state, traverser, combos);
        }

        let actor = state.actor.unwrap() as usize;
        let acts = legal_actions(state, &self.raise_sizes_pm, self.allin_atom);
        if acts.is_empty() {
            return self.terminal_value(state, traverser, combos);
        }

        // Private view: exact combo on river; buckets on flop/turn when enabled.
        // History includes board so chance runouts create distinct infosets.
        let blen = state.board_len;
        let bslice = &state.board[..blen as usize];
        let priv_id = self.private_view(combos[actor], bslice);
        let key = InfosetKey::new(actor as u8, history_hash(history, bslice, blen), priv_id);
        if !self.infosets.contains_key(&key) {
            let (kind, raw, iso) = self.private_meta(combos[actor], priv_id, bslice);
            let dump = InfosetDump::from_state(state, history, kind, priv_id, raw, iso);
            self.infosets
                .insert(key, Infoset::new_with_dump(acts.clone(), dump));
        }
        let strategy = self.infosets.get(&key).unwrap().current_strategy();

        let mut action_utils = vec![0.0; acts.len()];
        let mut node_util = 0.0;

        for (i, &act) in acts.iter().enumerate() {
            let mut child = state.clone();
            if apply_abstract(&mut child, act).is_err() {
                continue;
            }
            let mut h2 = history.to_vec();
            h2.push(act);
            let mut r2 = reach;
            r2[actor] *= strategy[i];
            let u = self.cfr(&child, &h2, combos, r2, traverser);
            action_utils[i] = u;
            node_util += strategy[i] * u;
        }

        if actor == traverser {
            let opp_reach = reach[1 - traverser];
            let node = self.infosets.get_mut(&key).unwrap();
            for i in 0..acts.len() {
                let regret = action_utils[i] - node_util;
                node.regret[i] += opp_reach * regret;
            }
            let my_reach = reach[traverser];
            for i in 0..acts.len() {
                node.strategy_sum[i] += my_reach * strategy[i];
            }
        }

        node_util
    }

    fn discount_all(&mut self, iteration: u32, params: (f64, f64, f64)) {
        let (alpha, beta, gamma) = params;
        let (pos, neg, strat) = Infoset::discount_scales(iteration, alpha, beta, gamma);
        for node in self.infosets.values_mut() {
            node.apply_scales(pos, neg, strat);
        }
    }

    fn export_strategy(&self, root_id: &str) -> Strategy {
        let mut infosets = Vec::new();
        for (key, node) in &self.infosets {
            infosets.push(InfosetStrategy::from_node(
                format!("p{}_h{}_c{}", key.player, key.history, key.private),
                node,
            ));
        }
        // Stable order for determinism
        infosets.sort_by(|a, b| a.infoset_id.cmp(&b.infoset_id));
        Strategy::new(root_id.to_string(), infosets)
    }
}

#[inline]
fn combo_has_card(combo: usize, card: u8) -> bool {
    let (a, b) = combo_table()[combo];
    a == card || b == card
}

/// Minimal RNG (xorshift*) to avoid pulling rand into hot path tests.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

// ----------------------------------------------------------------------
// Deal sampling (review 2026-09-20 D7)
// ----------------------------------------------------------------------

/// Inverse-CDF sampler over one player's live combos (weight > 0, not blocked
/// by the ROOT board).
struct ComboSampler {
    ids: Vec<u32>,
    cdf: Vec<f64>,
}

impl ComboSampler {
    fn new(range: &Range, root_board: &[u8]) -> Self {
        let mut mask = 0u64;
        for &c in root_board {
            mask |= 1u64 << c;
        }
        let table = combo_table();
        let mut ids = Vec::new();
        let mut cdf = Vec::new();
        let mut acc = 0.0;
        for id in 0..NUM_COMBOS {
            let w = range.weights[id];
            let (a, b) = table[id];
            if w > 0.0 && mask & ((1u64 << a) | (1u64 << b)) == 0 {
                acc += w;
                ids.push(id as u32);
                cdf.push(acc);
            }
        }
        Self { ids, cdf }
    }

    fn total(&self) -> f64 {
        self.cdf.last().copied().unwrap_or(0.0)
    }

    fn sample(&self, rng: &mut Lcg) -> usize {
        let t = rng.next_f64() * self.total();
        let i = self.cdf.partition_point(|&c| c <= t).min(self.ids.len() - 1);
        self.ids[i] as usize
    }
}

/// Joint chance sampler: `P(c0, c1, runout) ∝ w0(c0)·w1(c1)·[all cards distinct]`.
///
/// (review 2026-09-20 D7) The solver used to draw `c0 ~ w0` and then
/// `c1 ~ w1 | c0`. That gives `P(c0) ∝ w0(c0)` instead of the true marginal
/// `∝ w0(c0)·W1(c0)` (`W1` = opponent mass NOT blocked by `c0`), so with
/// blocker-coupled ranges the solver trained on a different game (the toy case
/// {AcAd,KcKd} vs AA dealt AcAd 1/2 of the time instead of 1/7). Drawing both
/// combos independently and rejecting the PAIR on a collision samples the joint
/// exactly; the runout is then uniform over the remaining deck (every disjoint
/// pair leaves the same number of runouts, so no runout rejection is needed).
/// For uniform ranges the two schemes coincide (every combo blocks the same
/// number of opponent combos), so unranged solves are statistically unchanged.
struct DealSampler {
    root_board: Vec<u8>,
    s0: ComboSampler,
    s1: ComboSampler,
}

/// Independent-draw attempts before falling back to exact pair enumeration.
/// Both paths sample the same joint; the bound only caps wasted draws when
/// almost all of the mass collides.
const DEAL_REJECTION_TRIES: u32 = 64;

impl DealSampler {
    fn new(root_board: &[u8], r0: &Range, r1: &Range) -> Result<Self, CfrError> {
        let s = Self {
            root_board: root_board.to_vec(),
            s0: ComboSampler::new(r0, root_board),
            s1: ComboSampler::new(r1, root_board),
        };
        if s.s0.ids.is_empty() || s.s1.ids.is_empty() {
            return Err(CfrError::InvalidRoot(
                "range has no live combo on this board".into(),
            ));
        }
        if s.exact_pair(None).is_none() {
            return Err(CfrError::InvalidRoot(
                "range_oop and range_ip have no pair of combos without a shared card".into(),
            ));
        }
        Ok(s)
    }

    /// Exact draw from the joint by enumerating every disjoint pair.
    /// `rng == None` only checks that the joint has mass.
    fn exact_pair(&self, rng: Option<&mut Lcg>) -> Option<[usize; 2]> {
        let table = combo_table();
        let weight = |i: usize, j: usize| -> f64 {
            let (a, b) = table[self.s0.ids[i] as usize];
            let (c, d) = table[self.s1.ids[j] as usize];
            if a == c || a == d || b == c || b == d {
                return 0.0;
            }
            let w0 = self.s0.cdf[i] - if i > 0 { self.s0.cdf[i - 1] } else { 0.0 };
            let w1 = self.s1.cdf[j] - if j > 0 { self.s1.cdf[j - 1] } else { 0.0 };
            w0 * w1
        };
        let mut total = 0.0;
        for i in 0..self.s0.ids.len() {
            for j in 0..self.s1.ids.len() {
                total += weight(i, j);
            }
        }
        if total <= 0.0 {
            return None;
        }
        let rng = match rng {
            Some(r) => r,
            None => return Some([self.s0.ids[0] as usize, self.s1.ids[0] as usize]),
        };
        let mut t = rng.next_f64() * total;
        let mut last = None;
        for i in 0..self.s0.ids.len() {
            for j in 0..self.s1.ids.len() {
                let w = weight(i, j);
                if w <= 0.0 {
                    continue;
                }
                last = Some([self.s0.ids[i] as usize, self.s1.ids[j] as usize]);
                t -= w;
                if t <= 0.0 {
                    return last;
                }
            }
        }
        last
    }

    fn sample(&self, rng: &mut Lcg) -> ([u8; 5], [usize; 2]) {
        let table = combo_table();
        let mut pair = None;
        for _ in 0..DEAL_REJECTION_TRIES {
            let c0 = self.s0.sample(rng);
            let c1 = self.s1.sample(rng);
            let (a, b) = table[c0];
            let (c, d) = table[c1];
            if a != c && a != d && b != c && b != d {
                pair = Some([c0, c1]);
                break;
            }
        }
        let combos = pair
            .or_else(|| self.exact_pair(Some(rng)))
            .expect("joint mass checked in DealSampler::new");
        // Runout: uniform over the deck minus root board and both holes.
        let mut full = [0u8; 5];
        let mut used = [false; 52];
        for (i, &c) in self.root_board.iter().enumerate() {
            full[i] = c;
            used[c as usize] = true;
        }
        for &combo in &combos {
            let (a, b) = table[combo];
            used[a as usize] = true;
            used[b as usize] = true;
        }
        let mut i = self.root_board.len();
        while i < 5 {
            let c = (rng.next_u64() % 52) as u8;
            if !used[c as usize] {
                used[c as usize] = true;
                full[i] = c;
                i += 1;
            }
        }
        (full, combos)
    }
}

/// Solve HU river with chance-sampled DCFR.
pub fn solve_river_dcfr(root: &RootSpec, config: &SolveConfig) -> Result<SolveReport, CfrError> {
    root.validate()?;
    config.validate()?;
    match root.street {
        StreetRoot::River => {}
        _ => {
            return Err(CfrError::InvalidRoot(
                "solve_river_dcfr expects River street".into(),
            ));
        }
    }
    solve_postflop_dcfr(root, config)
}

/// HU flop/turn/river DCFR with external-sampled runouts when board < 5.
pub fn solve_postflop_with_runouts(
    root: &RootSpec,
    config: &SolveConfig,
) -> Result<SolveReport, CfrError> {
    root.validate()?;
    config.validate()?;
    solve_postflop_dcfr(root, config)
}

/// First exploitability poll; later polls are spaced geometrically (x1.25).
/// One exact poll costs on the order of 100 CFR iterations (tree x 1326 combos
/// vs tree x 1 deal), so this keeps polling at a few percent of the run while an
/// early stop overshoots its target by at most a quarter of the iterations.
const POLL_FIRST_ITER: u32 = 50;
/// Wall-clock allowance for the FINAL best response after a normal exit.
const FINAL_EVAL_MIN_SECS: f64 = 30.0;
/// ... and after a stop-file / time-budget exit, where the caller is waiting
/// (the desktop app kills the worker ~10 s after Stop, and exporting a large
/// strategy to Python takes seconds of that on its own).
const FINAL_EVAL_EARLY_STOP_SECS: f64 = 3.0;
/// Rough live bytes per stored infoset (regrets + sums + dump + map slot).
const RUNTIME_BYTES_PER_INFOSET: u64 = super::memory::BYTES_PER_INFOSET_BASE;

fn solve_postflop_dcfr(root: &RootSpec, config: &SolveConfig) -> Result<SolveReport, CfrError> {
    // Memory refuse + auto-upgrade flop to buckets
    let mut cfg = config.clone();
    if root.street == StreetRoot::Flop
        && (cfg.card_abstraction == "none" || cfg.card_abstraction.is_empty())
    {
        cfg.card_abstraction = "ochs".into();
    }
    let mem = super::memory::refuse_if_unsafe(root, &cfg)?;

    let mut solver = RiverSolver::new(root, &cfg)?;
    // (review 2026-09-20 D12) a range that does not parse is an error, never a
    // silent uniform fallback.
    let named = |which: &str, e: CfrError| match e {
        CfrError::InvalidRoot(msg) => CfrError::InvalidRoot(format!("{which}: {msg}")),
        other => other,
    };
    let range0 = Range::parse(&root.range_oop, &solver.board_root)
        .map_err(|e| named("range_oop", e))?;
    let range1 = Range::parse(&root.range_ip, &solver.board_root)
        .map_err(|e| named("range_ip", e))?;
    let dealer = DealSampler::new(&solver.board_root, &range0, &range1)?;

    let mut rng = Lcg::new(cfg.seed);
    let street_name = format!("{:?}", root.street);
    let bucket_note = if solver.use_buckets {
        // (review 2026-09-20 F8) honest label: NOT published OCHS, and the
        // bucket is frozen at the flop, so turn/river infosets cannot see how
        // later cards changed the hand.
        format!(
            "card_abs={} buckets_on (flop equity-quantile x{}, fixed at the flop: later \
             streets do not re-bucket; coarse abstraction)",
            cfg.card_abstraction,
            super::card_abs::FLOP_BUCKETS
        )
    } else {
        "card_abs=exact_combo".into()
    };
    // (review 2026-09-20 D15) the suit relabel is a bijection for any fixed
    // public board, so it never merges infosets — say so instead of implying
    // a reduction.
    let iso_note = if cfg.use_isomorphism {
        "isomorphism=on iso=noop_on_fixed_board (suit relabel only; no infoset reduction)"
    } else {
        "isomorphism=off"
    };
    let discounting = cfg.discounting();
    let algo_label = match discounting {
        Discounting::Dcfr => "DCFR",
        Discounting::Linear => "LinearCFR",
        Discounting::None => "CFR(no discount)",
    };
    let threads = cfg.thread_num.max(1);
    let batch = threads.min(8) as usize;
    let start = Instant::now();
    let mut stop_reason: Option<&'static str> = None;
    let mut iterations_run = 0u32;
    // (review 2026-09-20 D8c) no target ⇒ nobody reads the polls ⇒ skip them
    // (each one costs thousands of CFR iterations).
    let polls_on = cfg.target_exploitability_bb > 0.0;
    let mut next_poll = POLL_FIRST_ITER;
    let mut last_poll: Option<ExplEstimate> = None;
    let mut last_poll_iter = 0u32;
    let mem_budget = super::memory::DEFAULT_RAM_BUDGET_BYTES;

    let iter_limit = cfg.iter_limit();
    for it in 1..=iter_limit {
        if let Some(why) = cfg.should_stop(start, it) {
            stop_reason = Some(why);
            break;
        }
        // When thread_num==1, stay single-threaded (determinism for tests).
        if batch <= 1 {
            let (full, combos) = dealer.sample(&mut rng);
            solver.set_sample_board(full);
            let state = solver.root_state();
            for trav in 0..2 {
                solver.cfr(&state, &[], combos, [1.0, 1.0], trav);
            }
        } else {
            use rayon::prelude::*;
            let seed_base = cfg.seed.wrapping_add(it as u64 * 10007);
            let deals: Vec<_> = (0..batch)
                .into_par_iter()
                .map(|b| {
                    let mut local_rng = Lcg::new(seed_base.wrapping_add(b as u64));
                    dealer.sample(&mut local_rng)
                })
                .collect();
            // Sequential CFR on each deal (shared table); parallel deal sampling
            // is the Rayon win. Full lock-free CFR would need sharded tables.
            for (full, combos) in deals {
                solver.set_sample_board(full);
                let state = solver.root_state();
                for trav in 0..2 {
                    solver.cfr(&state, &[], combos, [1.0, 1.0], trav);
                }
            }
        }

        if let Some(params) = discounting.params() {
            solver.discount_all(it, params);
        }
        iterations_run = it;

        if polls_on && it >= next_poll {
            next_poll = (it + POLL_FIRST_ITER).max(it + it / 4);
            let deadline = Instant::now() + Duration::from_secs_f64(FINAL_EVAL_MIN_SECS);
            if let Some(est) = poll_exploitability(&solver, &range0, &range1, &mut rng, deadline) {
                let reached = est.expl_bb <= cfg.target_exploitability_bb;
                last_poll = Some(est);
                last_poll_iter = it;
                if reached {
                    stop_reason = Some("target_exploitability");
                }
            }
        }
        // Live progress dump for desktop UI (every poll_every iters).
        let poll = cfg.poll_every.max(1);
        if it % poll == 0 || it == 1 {
            let strat = if !cfg.progress_file.is_empty() {
                Some(solver.export_strategy(&root.root_id))
            } else {
                None
            };
            cfg.write_progress(
                it,
                last_poll.as_ref().map(|e| e.expl_bb),
                last_poll.as_ref().map(|e| e.kind.as_str()),
                solver.infosets.len(),
                strat.as_ref(),
                &root.root_id,
                false,
            );
            // (review 2026-09-20 F14) the up-front estimate is only a bound;
            // the table itself is the ground truth — stop before it OOMs.
            if solver.infosets.len() as u64 * RUNTIME_BYTES_PER_INFOSET > mem_budget {
                stop_reason = Some("memory_budget");
            }
        }
        if stop_reason.is_some() {
            break;
        }
    }

    // (review 2026-09-20 D8b) EVERY exit path — max_iterations, time budget,
    // stop file, target reached — reports the FINAL estimator. The in-loop
    // poll number is only ever returned when the final pass cannot finish in
    // its wall-clock allowance, and is then labelled `expl_kind=mc_poll`.
    let early = matches!(stop_reason, Some("time_budget") | Some("stop_file"));
    let allowance = if early {
        FINAL_EVAL_EARLY_STOP_SECS
    } else {
        FINAL_EVAL_MIN_SECS.max(0.25 * start.elapsed().as_secs_f64())
    };
    let final_started = Instant::now();
    let reuse_poll = last_poll
        .as_ref()
        .filter(|e| e.is_final_quality && last_poll_iter == iterations_run)
        .cloned();
    let final_est = reuse_poll.or_else(|| {
        final_exploitability(
            &solver,
            &range0,
            &range1,
            &mut rng,
            Instant::now() + Duration::from_secs_f64(allowance),
        )
    });
    let final_secs = final_started.elapsed().as_secs_f64();
    let (expl, expl_note) = match (final_est, last_poll) {
        (Some(est), _) => (
            Some(est.expl_bb),
            format!("expl_kind={} {} final_expl_secs={final_secs:.1}", est.kind, est.detail),
        ),
        (None, Some(poll)) => (
            Some(poll.expl_bb),
            format!(
                "expl_kind=mc_poll (final best response skipped: exceeded {allowance:.0}s; \
                 value is the iter-{last_poll_iter} poll: {} {})",
                poll.kind, poll.detail
            ),
        ),
        (None, None) => (
            None,
            format!("expl_kind=none (final best response skipped: exceeded {allowance:.0}s)"),
        ),
    };

    if iterations_run == 0 {
        iterations_run = 1;
    }

    let strategy = solver.export_strategy(&root.root_id);
    let n_infosets = solver.infosets.len();
    let uniform0 = Range::spec_is_uniform(&root.range_oop);
    let uniform1 = Range::spec_is_uniform(&root.range_ip);
    let side = |uniform: bool, r: &Range| {
        format!(
            "{}:{}",
            if uniform { "uniform" } else { "parsed" },
            r.live_combos()
        )
    };
    let mut notes = vec![
        format!("{algo_label} {street_name} HU (runouts sampled if board<5)"),
        format!("algorithm={} discount={}", cfg.algorithm, discounting.label()),
        format!("infosets={n_infosets}"),
        format!("pot_chips={} stack_chips={}", solver.pot0, solver.stack0),
        if uniform0 && uniform1 {
            // Only when the caller passed NO range (review 2026-09-20 D12).
            "ranges=uniform_fallback (no range given)".into()
        } else {
            format!(
                "ranges=parsed oop={} ip={}",
                side(uniform0, &range0),
                side(uniform1, &range1)
            )
        },
        "deal_sampling=joint (w0*w1*[disjoint], pair rejection)".into(),
        bucket_note,
        iso_note.into(),
        format!("thread_num={threads} rayon_deals={batch}"),
        format!("est_mem_mb={:.1} est_infosets={}", mem.mb(), mem.est_infosets),
        format!("wall_secs={:.1}", start.elapsed().as_secs_f64()),
        expl_note,
    ];
    if solver.use_buckets {
        notes.push(
            "expl_scope=abstract_game (best responder is restricted to the same buckets)".into(),
        );
    }
    if let Some(why) = stop_reason {
        notes.push(format!("early_stop={why}"));
    }
    Ok(SolveReport {
        status: "ok".into(),
        root: root.clone(),
        config: cfg.clone(),
        strategy,
        iterations_run,
        exploitability_bb: expl,
        notes,
    })
}

// ----------------------------------------------------------------------
// Exploitability (review 2026-09-20 D7 / D8)
// ----------------------------------------------------------------------

/// One exploitability estimate plus an honest description of what it is.
#[derive(Debug, Clone)]
struct ExplEstimate {
    expl_bb: f64,
    /// `exact_infoset` | `sampled_runout_br` | `mc_poll`
    kind: String,
    detail: String,
    /// True when this is what the final report would compute anyway.
    is_final_quality: bool,
}

/// How the evaluator expands chance (runout) nodes.
enum RunoutPlan {
    /// Every legal next card: the exact expectation over the runout.
    All,
    /// The empirical chance distribution of these sampled full boards.
    Sampled(Vec<[u8; 5]>),
}

/// Sampled rivers for a turn-root poll / fallback, and turn×river grids for
/// flop roots (final, poll).
const TURN_POLL_RIVERS: usize = 4;
const TURN_FALLBACK_RIVERS: usize = 8;
const FLOP_FINAL_GRID: (usize, usize) = (6, 4);
const FLOP_POLL_GRID: (usize, usize) = (3, 2);

/// `turns × rivers` distinct runouts extending `root_board` (turn roots use
/// `turns == 1` slot for the fixed turn card).
fn sample_runout_grid(
    root_board: &[u8],
    turns: usize,
    rivers: usize,
    rng: &mut Lcg,
) -> Vec<[u8; 5]> {
    let mut used = [false; 52];
    for &c in root_board {
        used[c as usize] = true;
    }
    let draw = |used: &mut [bool; 52], rng: &mut Lcg| -> u8 {
        loop {
            let c = (rng.next_u64() % 52) as u8;
            if !used[c as usize] {
                used[c as usize] = true;
                return c;
            }
        }
    };
    let mut out = Vec::new();
    let mut base = [0u8; 5];
    base[..root_board.len()].copy_from_slice(root_board);
    if root_board.len() == 4 {
        let mut u = used;
        for _ in 0..rivers.min(48) {
            let mut b = base;
            b[4] = draw(&mut u, rng);
            out.push(b);
        }
        return out;
    }
    let mut used_turn = used;
    for _ in 0..turns.min(49) {
        let t = draw(&mut used_turn, rng);
        let mut u = used;
        u[t as usize] = true;
        for _ in 0..rivers.min(48) {
            let mut b = base;
            b[3] = t;
            b[4] = draw(&mut u, rng);
            out.push(b);
        }
    }
    out
}

/// In-loop estimate used only for the target-exploitability early stop.
fn poll_exploitability(
    solver: &RiverSolver,
    r0: &Range,
    r1: &Range,
    rng: &mut Lcg,
    deadline: Instant,
) -> Option<ExplEstimate> {
    match solver.board_root.len() {
        5 => solver
            .evaluate(r0, r1, &RunoutPlan::All, Some(deadline))
            .map(|expl_bb| ExplEstimate {
                expl_bb,
                kind: "exact_infoset".into(),
                detail: "(vectorized infoset best response, all combos)".into(),
                is_final_quality: true,
            }),
        blen => {
            let (t, r) = if blen == 4 { (1, TURN_POLL_RIVERS) } else { FLOP_POLL_GRID };
            let plan = RunoutPlan::Sampled(sample_runout_grid(&solver.board_root, t, r, rng));
            solver
                .evaluate(r0, r1, &plan, Some(deadline))
                .map(|expl_bb| ExplEstimate {
                    expl_bb,
                    kind: "mc_poll".into(),
                    detail: format!("runouts_sampled={t}x{r} (upper-biased)"),
                    is_final_quality: false,
                })
        }
    }
}

/// The reported number: exact on river roots; on turn roots the exact
/// expectation over all rivers when it fits the allowance, else (and always on
/// flop roots) a best response against a sampled runout grid.
fn final_exploitability(
    solver: &RiverSolver,
    r0: &Range,
    r1: &Range,
    rng: &mut Lcg,
    deadline: Instant,
) -> Option<ExplEstimate> {
    let exact = |detail: &str| {
        solver
            .evaluate(r0, r1, &RunoutPlan::All, Some(deadline))
            .map(|expl_bb| ExplEstimate {
                expl_bb,
                kind: "exact_infoset".into(),
                detail: detail.into(),
                is_final_quality: true,
            })
    };
    let sampled = |t: usize, r: usize, rng: &mut Lcg| {
        let plan = RunoutPlan::Sampled(sample_runout_grid(&solver.board_root, t, r, rng));
        solver
            .evaluate(r0, r1, &plan, Some(deadline))
            .map(|expl_bb| ExplEstimate {
                expl_bb,
                kind: "sampled_runout_br".into(),
                detail: format!(
                    "runouts_sampled={t}x{r} (expectation over the sampled runouts, not \
                     clairvoyant; upper-biased: the responder fits the sample)"
                ),
                is_final_quality: true,
            })
    };
    // Turn/flop roots climb a ladder of ever larger runout samples under the
    // ONE shared deadline and keep the best level that finished, so a short
    // allowance (stop file / time budget on a big tree) still reports a
    // labelled estimate instead of nothing. Each rung costs ~4x the previous.
    match solver.board_root.len() {
        5 => exact("(vectorized infoset best response, all combos)"),
        4 => {
            let mut best = sampled(1, 2, rng)?;
            if let Some(better) = sampled(1, TURN_FALLBACK_RIVERS, rng) {
                best = better;
                if let Some(all) = exact("(expectation over all rivers)") {
                    best = all;
                }
            }
            Some(best)
        }
        _ => {
            let mut best = sampled(2, 2, rng)?;
            if let Some(better) = sampled(FLOP_FINAL_GRID.0, FLOP_FINAL_GRID.1, rng) {
                best = better;
            }
            Some(best)
        }
    }
}

/// Showdown order of one full board: live combos sorted by rank ascending.
struct ShowdownOrder {
    ranks: Vec<u32>,
    order: Vec<u32>,
}

impl ShowdownOrder {
    fn new(ranks: Vec<u32>) -> Self {
        let mut order: Vec<u32> = (0..NUM_COMBOS as u32)
            .filter(|&c| ranks[c as usize] > 0)
            .collect();
        order.sort_by_key(|&c| ranks[c as usize]);
        Self { ranks, order }
    }
}

/// `(total, per-card totals)` of a reach vector.
fn reach_totals(reach: &[f64]) -> (f64, [f64; 52]) {
    let table = combo_table();
    let mut total = 0.0;
    let mut by_card = [0.0f64; 52];
    for (c, &r) in reach.iter().enumerate() {
        if r > 0.0 {
            let (a, b) = table[c];
            total += r;
            by_card[a as usize] += r;
            by_card[b as usize] += r;
        }
    }
    (total, by_card)
}

struct EvalCtx<'a> {
    solver: &'a RiverSolver,
    hero: usize,
    hero_w: &'a [f64],
    plan: &'a RunoutPlan,
    deadline: Option<Instant>,
    orders: HashMap<[u8; 5], ShowdownOrder>,
    aborted: bool,
    nodes: u64,
}

/// Per hero combo `h`: unnormalized counterfactual value
/// `Σ_c reach_opp(c)·[h,c disjoint]·E_runout[u_hero(h,c)]` under
/// (best response, average strategy) for the hero.
type ValuePair = (Vec<f64>, Vec<f64>);

impl<'a> EvalCtx<'a> {
    fn zeros() -> ValuePair {
        (vec![0.0; NUM_COMBOS], vec![0.0; NUM_COMBOS])
    }

    fn eval(
        &mut self,
        state: &PublicState,
        history: &mut Vec<AbstractAction>,
        opp_reach: &[f64],
    ) -> ValuePair {
        self.nodes += 1;
        if self.nodes % 32 == 0 {
            if let Some(d) = self.deadline {
                if Instant::now() > d {
                    self.aborted = true;
                }
            }
        }
        if self.aborted {
            return Self::zeros();
        }
        if state.needs_runout() {
            return self.eval_chance(state, history, opp_reach);
        }
        let table = combo_table();
        let blen = state.board_len as usize;
        let mut board_mask = 0u64;
        for &c in &state.board[..blen] {
            board_mask |= 1u64 << c;
        }
        let live = |h: usize| {
            let (a, b) = table[h];
            board_mask & ((1u64 << a) | (1u64 << b)) == 0
        };

        let acts = if state.is_terminal() || state.actor.is_none() {
            vec![]
        } else {
            legal_actions(state, &self.solver.raise_sizes_pm, self.solver.allin_atom)
        };
        if acts.is_empty() {
            let v = self.terminal_values(state, opp_reach, &live);
            return (v.clone(), v);
        }

        let actor = state.actor.unwrap() as usize;
        let bslice = &state.board[..blen];
        let hhash = history_hash(history, bslice, state.board_len);
        let n_act = acts.len();
        // Average strategy of `actor` holding `combo` at this node (uniform
        // when the infoset was never visited), written into `out`.
        let strat_of = |combo: usize, out: &mut [f64]| {
            let key = InfosetKey::new(actor as u8, hhash, self.solver.private_view(combo, bslice));
            let sum_node = self.solver.infosets.get(&key).and_then(|n| {
                let s: f64 = n.strategy_sum.iter().sum();
                if s > 0.0 && n.strategy_sum.len() == n_act {
                    Some((n, s))
                } else {
                    None
                }
            });
            match sum_node {
                Some((n, s)) => {
                    for a in 0..n_act {
                        out[a] = n.strategy_sum[a] / s;
                    }
                }
                None => out.iter_mut().for_each(|x| *x = 1.0 / n_act as f64),
            }
        };

        if actor != self.hero {
            let mut child_reach = vec![vec![0.0; NUM_COMBOS]; n_act];
            let mut sigma = vec![0.0; n_act];
            for c in 0..NUM_COMBOS {
                let r = opp_reach[c];
                if r <= 0.0 {
                    continue;
                }
                strat_of(c, &mut sigma);
                for a in 0..n_act {
                    child_reach[a][c] = r * sigma[a];
                }
            }
            let mut out = Self::zeros();
            for (a, &act) in acts.iter().enumerate() {
                if child_reach[a].iter().all(|&x| x <= 0.0) {
                    continue;
                }
                let mut child = state.clone();
                if apply_abstract(&mut child, act).is_err() {
                    continue;
                }
                history.push(act);
                let (br, avg) = self.eval(&child, history, &child_reach[a]);
                history.pop();
                for h in 0..NUM_COMBOS {
                    out.0[h] += br[h];
                    out.1[h] += avg[h];
                }
            }
            return out;
        }

        // Hero node.
        let mut children: Vec<Option<ValuePair>> = Vec::with_capacity(n_act);
        for &act in &acts {
            let mut child = state.clone();
            if apply_abstract(&mut child, act).is_err() {
                children.push(None);
                continue;
            }
            history.push(act);
            children.push(Some(self.eval(&child, history, opp_reach)));
            history.pop();
        }
        let mut out = Self::zeros();
        // Average strategy.
        let mut sigma = vec![0.0; n_act];
        for h in 0..NUM_COMBOS {
            if self.hero_w[h] <= 0.0 || !live(h) {
                continue;
            }
            strat_of(h, &mut sigma);
            let mut v = 0.0;
            for (a, ch) in children.iter().enumerate() {
                if let Some((_, avg)) = ch {
                    v += sigma[a] * avg[h];
                }
            }
            out.1[h] = v;
        }
        // Best response: ONE action per hero infoset.
        if self.solver.use_buckets {
            // Infoset = bucket: maximize the prior-weighted sum over its combos.
            let mut sums: HashMap<u32, Vec<f64>> = HashMap::new();
            for h in 0..NUM_COMBOS {
                if self.hero_w[h] <= 0.0 || !live(h) {
                    continue;
                }
                let pv = self.solver.private_view(h, bslice);
                let e = sums.entry(pv).or_insert_with(|| vec![0.0; n_act]);
                for (a, ch) in children.iter().enumerate() {
                    if let Some((br, _)) = ch {
                        e[a] += self.hero_w[h] * br[h];
                    }
                }
            }
            let best: HashMap<u32, usize> = sums
                .into_iter()
                .map(|(pv, s)| {
                    let mut bi = usize::MAX;
                    for (a, ch) in children.iter().enumerate() {
                        if ch.is_some() && (bi == usize::MAX || s[a] > s[bi]) {
                            bi = a;
                        }
                    }
                    (pv, bi)
                })
                .collect();
            for h in 0..NUM_COMBOS {
                if self.hero_w[h] <= 0.0 || !live(h) {
                    continue;
                }
                let pv = self.solver.private_view(h, bslice);
                if let Some(&bi) = best.get(&pv) {
                    if let Some(Some((br, _))) = children.get(bi) {
                        out.0[h] = br[h];
                    }
                }
            }
        } else {
            // Infoset = combo (the iso relabel is a bijection): per-combo max.
            for h in 0..NUM_COMBOS {
                if self.hero_w[h] <= 0.0 || !live(h) {
                    continue;
                }
                let mut best = f64::NEG_INFINITY;
                for ch in children.iter().flatten() {
                    best = best.max(ch.0[h]);
                }
                out.0[h] = if best.is_finite() { best } else { 0.0 };
            }
        }
        out
    }

    /// (review 2026-09-20 D8d) Chance node: an EXPECTATION over the next
    /// public card. The old estimator fixed one sampled runout per deal and
    /// let the responder maximize along it, i.e. with the future board known
    /// (clairvoyant) — a large upward bias on flop/turn roots.
    fn eval_chance(
        &mut self,
        state: &PublicState,
        history: &mut Vec<AbstractAction>,
        opp_reach: &[f64],
    ) -> ValuePair {
        let blen = state.board_len as usize;
        let on_board = |c: u8| state.board[..blen].contains(&c);
        // True chance: uniform over the 52 - blen - 4 cards outside the board
        // and both hands.
        let free = (52 - blen - 4) as f64;
        let cards: Vec<(u8, f64)> = match self.plan {
            RunoutPlan::All => (0..52u8)
                .filter(|&c| !on_board(c))
                .map(|c| (c, 1.0 / free))
                .collect(),
            RunoutPlan::Sampled(boards) => {
                let mut counts = [0u32; 52];
                let mut n = 0u32;
                for b in boards {
                    if b[..blen] == state.board[..blen] {
                        counts[b[blen] as usize] += 1;
                        n += 1;
                    }
                }
                // Sampled uniformly from the 52 - blen non-board cards; cards
                // that hit a hand are masked below, hence the rescale.
                let scale = (52 - blen) as f64 / free;
                (0..52u8)
                    .filter(|&c| counts[c as usize] > 0)
                    .map(|c| (c, scale * counts[c as usize] as f64 / n as f64))
                    .collect()
            }
        };
        let table = combo_table();
        let mut out = Self::zeros();
        for (x, p) in cards {
            let mut r2 = opp_reach.to_vec();
            for (c, r) in r2.iter_mut().enumerate() {
                if *r > 0.0 && (table[c].0 == x || table[c].1 == x) {
                    *r = 0.0;
                }
            }
            let mut child = state.clone();
            child.deal_board_card(x);
            let (br, avg) = self.eval(&child, history, &r2);
            for h in 0..NUM_COMBOS {
                if table[h].0 == x || table[h].1 == x {
                    continue;
                }
                out.0[h] += p * br[h];
                out.1[h] += p * avg[h];
            }
        }
        out
    }

    fn terminal_values(
        &mut self,
        state: &PublicState,
        opp_reach: &[f64],
        live: &dyn Fn(usize) -> bool,
    ) -> Vec<f64> {
        let table = combo_table();
        let hero = self.hero;
        let mut v = vec![0.0; NUM_COMBOS];
        let (total, by_card) = reach_totals(opp_reach);
        if total <= 0.0 {
            return v;
        }
        let commit = state.total_commit[hero] as f64;
        if state.alive_count() == 1 {
            let pay = state.fold_payout_chips(hero) as f64;
            for h in 0..NUM_COMBOS {
                if self.hero_w[h] > 0.0 && live(h) {
                    let (a, b) = table[h];
                    let compat = total - by_card[a as usize] - by_card[b as usize] + opp_reach[h];
                    v[h] = pay * compat;
                }
            }
            return v;
        }
        if state.board_len < 5 {
            return v; // cannot happen: a showdown always has a full board
        }
        let mut board = [0u8; 5];
        board.copy_from_slice(&state.board[..5]);
        let solver = self.solver;
        let order = self.orders.entry(board).or_insert_with(|| {
            ShowdownOrder::new(match (&solver.ranks, solver.board_root.len()) {
                (Some(r), 5) => r.clone(),
                _ => combo_ranks_on_board(&board),
            })
        });
        let pot = state.pot as f64;
        // Sweep rank groups ascending; `cum*` = reach of strictly weaker combos.
        let (mut cum, mut cum_card) = (0.0f64, [0.0f64; 52]);
        let ord = &order.order;
        let mut i = 0;
        while i < ord.len() {
            let rank = order.ranks[ord[i] as usize];
            let mut j = i;
            let (mut grp, mut grp_card) = (0.0f64, [0.0f64; 52]);
            while j < ord.len() && order.ranks[ord[j] as usize] == rank {
                let c = ord[j] as usize;
                let r = opp_reach[c];
                if r > 0.0 {
                    let (a, b) = table[c];
                    grp += r;
                    grp_card[a as usize] += r;
                    grp_card[b as usize] += r;
                }
                j += 1;
            }
            for &hc in &ord[i..j] {
                let h = hc as usize;
                if self.hero_w[h] <= 0.0 {
                    continue;
                }
                let (a, b) = (table[h].0 as usize, table[h].1 as usize);
                let own = opp_reach[h]; // subtracted twice via both cards
                let lower = cum - cum_card[a] - cum_card[b];
                let tie = grp - grp_card[a] - grp_card[b] + own;
                let compat = total - by_card[a] - by_card[b] + own;
                v[h] = pot * (lower + 0.5 * tie) - commit * compat;
            }
            cum += grp;
            for k in 0..52 {
                cum_card[k] += grp_card[k];
            }
            i = j;
        }
        v
    }
}

impl RiverSolver {
    /// NashConv/2 in bb of the current AVERAGE strategy, by a vectorized
    /// infoset best response over all hole combos.
    ///
    /// (review 2026-09-20 D7) Hero combos are weighted by the TRUE marginal
    /// `w(h)·W_opp(h)`: the values are unnormalized counterfactual sums, so
    /// `Σ_h w(h)·V[h] / Z` with `Z = Σ w0·w1·[disjoint]` is the joint
    /// expectation. The old code averaged conditional values with `w(h)` only
    /// and reported 0.013 bb on a toy whose true exploitability was 0.35 bb.
    ///
    /// (review 2026-09-20 D8a) Both terms are exact expectations over the
    /// hole cards of the same (possibly runout-sampled) game, so
    /// `BR − value >= 0` by construction and is clamped ONCE at the end — no
    /// per-sample `max(0)` (which biased every poll upward: 5.27 polled vs
    /// 1.99 final).
    ///
    /// Returns `None` when `deadline` passes first.
    fn evaluate(
        &self,
        r0: &Range,
        r1: &Range,
        plan: &RunoutPlan,
        deadline: Option<Instant>,
    ) -> Option<f64> {
        let table = combo_table();
        let mut mask = 0u64;
        for &c in &self.board_root {
            mask |= 1u64 << c;
        }
        let masked = |r: &Range| -> Vec<f64> {
            (0..NUM_COMBOS)
                .map(|c| {
                    let (a, b) = table[c];
                    if mask & ((1u64 << a) | (1u64 << b)) == 0 {
                        r.weights[c].max(0.0)
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let w = [masked(r0), masked(r1)];
        // Z = joint mass of disjoint (c0, c1) pairs.
        let (tot1, by_card1) = reach_totals(&w[1]);
        let mut z = 0.0;
        for h in 0..NUM_COMBOS {
            if w[0][h] > 0.0 {
                let (a, b) = table[h];
                z += w[0][h] * (tot1 - by_card1[a as usize] - by_card1[b as usize] + w[1][h]);
            }
        }
        if z <= 0.0 {
            return Some(0.0);
        }
        let mut nashconv = 0.0;
        for hero in 0..2 {
            let mut ctx = EvalCtx {
                solver: self,
                hero,
                hero_w: &w[hero],
                plan,
                deadline,
                orders: HashMap::new(),
                aborted: false,
                nodes: 0,
            };
            let mut history = Vec::new();
            let (br, avg) = ctx.eval(&self.root, &mut history, &w[1 - hero]);
            if ctx.aborted {
                return None;
            }
            for h in 0..NUM_COMBOS {
                nashconv += w[hero][h] * (br[h] - avg[h]);
            }
        }
        Some((nashconv / z).max(0.0) / 2.0 / self.bb as f64)
    }
}

// ----------------------------------------------------------------------
// Test-only reference implementations (per-deal enumeration)
// ----------------------------------------------------------------------

#[cfg(test)]
mod reference {
    //! Slow, obviously-correct counterparts of [`RiverSolver::evaluate`] for
    //! combo infosets: every deal (hole cards AND runout) is enumerated.
    use super::*;
    use crate::cards::Card;
    use crate::hand_eval::evaluate_nlh;

    /// Chip EV of `seat` at a terminal whose board is complete (or a fold).
    fn terminal(state: &PublicState, seat: usize, combos: [usize; 2]) -> f64 {
        if state.alive_count() == 1 {
            return state.fold_payout_chips(seat) as f64;
        }
        assert_eq!(state.board_len, 5);
        let b: Vec<Card> = state.board.iter().map(|&c| Card(c)).collect();
        let board = [b[0], b[1], b[2], b[3], b[4]];
        let rank = |combo: usize| {
            let (x, y) = combo_table()[combo];
            evaluate_nlh(&[Card(x), Card(y)], &board)
        };
        let (rh, ro) = (rank(combos[seat]), rank(combos[1 - seat]));
        let eq = if rh > ro {
            1.0
        } else if rh == ro {
            0.5
        } else {
            0.0
        };
        eq * state.pot as f64 - state.total_commit[seat] as f64
    }

    /// Chance node: `(child state, probability)` for every next card that is
    /// not on the board or in a hand of `holes` (uniform over 52 - board - 4:
    /// the opponent's two cards are unseen but also out of the deck).
    fn runouts(state: &PublicState, holes: &[usize]) -> Vec<(PublicState, u8, f64)> {
        let blen = state.board_len as usize;
        let p = 1.0 / (52 - blen - 4) as f64;
        (0..52u8)
            .filter(|c| !state.board[..blen].contains(c))
            .filter(|&c| holes.iter().all(|&h| !combo_has_card(h, c)))
            .map(|c| {
                let mut child = state.clone();
                child.deal_board_card(c);
                (child, c, p)
            })
            .collect()
    }

    fn strategy_at(
        solver: &RiverSolver,
        state: &PublicState,
        history: &[AbstractAction],
        actor: usize,
        combo: usize,
        n_act: usize,
    ) -> Vec<f64> {
        let blen = state.board_len;
        let bslice = &state.board[..blen as usize];
        let key = InfosetKey::new(
            actor as u8,
            history_hash(history, bslice, blen),
            solver.private_view(combo, bslice),
        );
        match solver.infosets.get(&key) {
            Some(n) => n.average_strategy(),
            None => vec![1.0 / n_act as f64; n_act],
        }
    }

    /// On-policy value of `player` for one deal.
    pub fn avg_value(
        solver: &RiverSolver,
        state: &PublicState,
        history: &[AbstractAction],
        combos: [usize; 2],
        player: usize,
    ) -> f64 {
        if state.needs_runout() {
            return runouts(state, &combos)
                .iter()
                .map(|(child, _, p)| p * avg_value(solver, child, history, combos, player))
                .sum();
        }
        if state.is_terminal() || state.actor.is_none() {
            return terminal(state, player, combos);
        }
        let actor = state.actor.unwrap() as usize;
        let acts = legal_actions(state, &solver.raise_sizes_pm, solver.allin_atom);
        let strat = strategy_at(solver, state, history, actor, combos[actor], acts.len());
        let mut v = 0.0;
        for (i, &act) in acts.iter().enumerate() {
            let mut child = state.clone();
            apply_abstract(&mut child, act).unwrap();
            let mut h2 = history.to_vec();
            h2.push(act);
            v += strat[i] * avg_value(solver, &child, &h2, combos, player);
        }
        v
    }

    /// Perfect-information BR: the responder sees the opponent's hole cards.
    pub fn deal_br_value(
        solver: &RiverSolver,
        state: &PublicState,
        history: &[AbstractAction],
        combos: [usize; 2],
        br_player: usize,
    ) -> f64 {
        if state.needs_runout() {
            // Sees the opponent's cards, NOT the future board.
            return runouts(state, &combos)
                .iter()
                .map(|(child, _, p)| p * deal_br_value(solver, child, history, combos, br_player))
                .sum();
        }
        if state.is_terminal() || state.actor.is_none() {
            return terminal(state, br_player, combos);
        }
        let actor = state.actor.unwrap() as usize;
        let acts = legal_actions(state, &solver.raise_sizes_pm, solver.allin_atom);
        let strat = strategy_at(solver, state, history, actor, combos[actor], acts.len());
        let mut vals = Vec::new();
        for &act in &acts {
            let mut child = state.clone();
            apply_abstract(&mut child, act).unwrap();
            let mut h2 = history.to_vec();
            h2.push(act);
            vals.push(deal_br_value(solver, &child, &h2, combos, br_player));
        }
        if actor == br_player {
            vals.into_iter().fold(f64::NEG_INFINITY, f64::max)
        } else {
            vals.iter().zip(&strat).map(|(v, p)| v * p).sum()
        }
    }

    /// Infoset BR for ONE hero combo against reach-weighted opponent combos:
    /// returns the unnormalized counterfactual value `Σ_c w[c]·u(h, c)`.
    pub fn infoset_br_cf(
        solver: &RiverSolver,
        state: &PublicState,
        history: &[AbstractAction],
        br_player: usize,
        br_combo: usize,
        opp_w: &[(usize, f64)],
    ) -> f64 {
        if state.needs_runout() {
            // The next card is public: the responder re-optimizes per card,
            // facing only the opponent combos that do not hold it.
            return runouts(state, &[br_combo])
                .iter()
                .map(|(child, card, _)| {
                    let p = 1.0 / (52 - state.board_len as usize - 4) as f64;
                    let w2: Vec<(usize, f64)> = opp_w
                        .iter()
                        .copied()
                        .filter(|&(c, _)| !combo_has_card(c, *card))
                        .map(|(c, w)| (c, w * p))
                        .collect();
                    infoset_br_cf(solver, child, history, br_player, br_combo, &w2)
                })
                .sum();
        }
        if state.is_terminal() || state.actor.is_none() {
            return opp_w
                .iter()
                .map(|&(c, w)| {
                    let mut combos = [0usize; 2];
                    combos[br_player] = br_combo;
                    combos[1 - br_player] = c;
                    w * terminal(state, br_player, combos)
                })
                .sum();
        }
        let actor = state.actor.unwrap() as usize;
        let acts = legal_actions(state, &solver.raise_sizes_pm, solver.allin_atom);
        let mut vals = Vec::new();
        for (i, &act) in acts.iter().enumerate() {
            let mut child = state.clone();
            apply_abstract(&mut child, act).unwrap();
            let mut h2 = history.to_vec();
            h2.push(act);
            let w2: Vec<(usize, f64)> = if actor == br_player {
                opp_w.to_vec()
            } else {
                opp_w
                    .iter()
                    .map(|&(c, w)| {
                        (c, w * strategy_at(solver, state, history, actor, c, acts.len())[i])
                    })
                    .collect()
            };
            vals.push(infoset_br_cf(solver, &child, &h2, br_player, br_combo, &w2));
        }
        if actor == br_player {
            vals.into_iter().fold(f64::NEG_INFINITY, f64::max)
        } else {
            vals.iter().sum()
        }
    }

    /// `(infoset NashConv/2, perfect-information NashConv/2)` in bb, by
    /// enumerating every deal of the TRUE joint `w0·w1·[disjoint]`.
    pub fn brute_force_expl(solver: &RiverSolver, r0: &Range, r1: &Range) -> (f64, f64) {
        let table = combo_table();
        let on_board = |c: usize| solver.board_root.iter().any(|&b| combo_has_card(c, b));
        let live = |r: &Range| -> Vec<(usize, f64)> {
            (0..NUM_COMBOS)
                .filter(|&c| r.weights[c] > 0.0 && !on_board(c))
                .map(|c| (c, r.weights[c]))
                .collect()
        };
        let l = [live(r0), live(r1)];
        let disjoint = |x: usize, y: usize| {
            let ((a, b), (c, d)) = (table[x], table[y]);
            a != c && a != d && b != c && b != d
        };
        let root = solver.root_state();
        let (mut z, mut v, mut deal_br) = (0.0, [0.0; 2], [0.0; 2]);
        for &(c0, w0) in &l[0] {
            for &(c1, w1) in &l[1] {
                if !disjoint(c0, c1) {
                    continue;
                }
                let w = w0 * w1;
                z += w;
                for p in 0..2 {
                    v[p] += w * avg_value(solver, &root, &[], [c0, c1], p);
                    deal_br[p] += w * deal_br_value(solver, &root, &[], [c0, c1], p);
                }
            }
        }
        let mut br = [0.0; 2];
        for p in 0..2 {
            for &(h, wh) in &l[p] {
                let opp: Vec<(usize, f64)> = l[1 - p]
                    .iter()
                    .copied()
                    .filter(|&(c, _)| disjoint(h, c))
                    .collect();
                br[p] += wh * infoset_br_cf(solver, &root, &[], p, h, &opp);
            }
        }
        let bb = solver.bb as f64;
        (
            (br[0] + br[1] - v[0] - v[1]) / z / 2.0 / bb,
            (deal_br[0] + deal_br[1] - v[0] - v[1]) / z / 2.0 / bb,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::range::cards_to_combo;
    use crate::cfr::types::{RootSpec, SolveConfig, StreetRoot};

    /// Train `iters` single-threaded iterations; returns solver + ranges.
    fn train(root: &RootSpec, iters: u32, seed: u64) -> (RiverSolver, Range, Range) {
        let mut cfg = SolveConfig::default();
        cfg.seed = seed;
        cfg.use_isomorphism = false;
        let mut solver = RiverSolver::new(root, &cfg).expect("solver");
        let r0 = Range::parse(&root.range_oop, &solver.board_root).unwrap();
        let r1 = Range::parse(&root.range_ip, &solver.board_root).unwrap();
        let dealer = DealSampler::new(&solver.board_root, &r0, &r1).unwrap();
        let mut rng = Lcg::new(seed);
        for it in 1..=iters {
            let (full, combos) = dealer.sample(&mut rng);
            solver.set_sample_board(full);
            let state = solver.root_state();
            for trav in 0..2 {
                solver.cfr(&state, &[], combos, [1.0, 1.0], trav);
            }
            solver.discount_all(it, Discounting::Dcfr.params().unwrap());
        }
        (solver, r0, r1)
    }

    fn tiny_root() -> RootSpec {
        // 4 vs 4 unblocked combos (numeric machine format).
        let mut root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            10.0,
            vec![0, 5, 10, 15, 20],
            vec![],
        );
        root.range_oop = "2:1,9:1,27:1,44:1".into();
        root.range_ip = "77:1,104:1,152:1,189:1".into();
        root
    }

    /// The reviewer's blocker toy (exp_blockers.py): OOP {AcAd, KcKd} vs IP AA
    /// on 2c 7d 9h Jc Ks. AcAd blocks 5 of the 6 AA combos.
    fn blocker_root() -> (RootSpec, usize, usize) {
        let mut root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            10.0,
            vec![0, 21, 30, 36, 47],
            vec![500, 1000],
        );
        root.range_oop = "AcAd,KcKd".into();
        root.range_ip = "AA".into();
        (root, cards_to_combo(48, 49), cards_to_combo(44, 45))
    }

    #[test]
    fn river_dcfr_runs_and_produces_strategy() {
        let root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            20.0,
            vec![0, 5, 10, 15, 20],
            vec![500, 1000], // coarse for speed
        );
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 100;
        cfg.target_exploitability_bb = 0.0; // don't early-stop
        cfg.seed = 42;
        let rep = solve_river_dcfr(&root, &cfg).expect("solve");
        assert_eq!(rep.status, "ok");
        assert_eq!(rep.iterations_run, 100);
        assert!(!rep.strategy.infosets.is_empty());
        assert!(rep.strategy.schema_version >= 2);
        // Each infoset probs sum ~ 1; dump schema present; fold iff to_call > 0
        for is in &rep.strategy.infosets {
            let s: f64 = is.probs.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "probs sum {s}");
            assert_eq!(is.schema_version, 2);
            assert_eq!(is.private_kind.as_deref(), Some("combo"));
            assert!(is.visit_mass.is_some());
            assert!(is.pot_chips.is_some());
            assert!(is.to_call_chips.is_some());
            assert!(is.path.is_some());
            let tc = is.to_call_chips.unwrap();
            let has_fold = is.actions.iter().any(|a| a == "FOLD");
            assert_eq!(has_fold, tc > 0, "fold vs to_call mismatch id={}", is.infoset_id);
            if let Some(raw) = is.raw_combo {
                assert!(raw < 1326);
            }
        }
    }

    #[test]
    fn tiny_range_jamcheck_expl_under_one_bb() {
        let root = tiny_root();
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 3000;
        cfg.seed = 1;
        cfg.thread_num = 1;
        cfg.use_isomorphism = false;
        cfg.target_exploitability_bb = 0.0;
        let rep = solve_river_dcfr(&root, &cfg).expect("tiny");
        let expl = rep.exploitability_bb.expect("expl");
        assert!(
            expl < 1.0,
            "tiny jam/check infoset expl {expl} bb (want < 1.0)"
        );
        assert!(rep.notes.iter().any(|n| n.contains("expl_kind=exact_infoset")));
        assert!(rep.notes.iter().any(|n| n == "ranges=parsed oop=parsed:4 ip=parsed:4"));
    }

    #[test]
    fn quads_board_expl_under_one_bb() {
        // Quad 3s + 5c, 4 vs 4 kickers. Full-range deal-BR was ~4.8 bb at 10k.
        let mut root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            20.0,
            vec![8, 9, 10, 11, 16],
            vec![500, 1000],
        );
        root.range_oop = "0:1,5:1,14:1,27:1".into();
        root.range_ip = "90:1,119:1,170:1,209:1".into();
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 2500;
        cfg.seed = 3;
        cfg.thread_num = 1;
        cfg.use_isomorphism = false;
        cfg.target_exploitability_bb = 0.0;
        let rep = solve_river_dcfr(&root, &cfg).expect("quads");
        let expl = rep.exploitability_bb.expect("expl");
        assert!(
            expl < 1.0,
            "quads-on-board tiny-range infoset expl {expl} bb (want < 1.0)"
        );
    }

    /// The vectorized evaluator equals per-deal enumeration, and a
    /// perfect-information responder can only do better than an infoset one.
    #[test]
    fn vectorized_br_matches_brute_force_and_deal_br_dominates() {
        let (solver, r0, r1) = train(&tiny_root(), 2500, 1);
        let fast = solver.evaluate(&r0, &r1, &RunoutPlan::All, None).unwrap();
        let (slow, deal) = reference::brute_force_expl(&solver, &r0, &r1);
        assert!((fast - slow).abs() < 1e-9, "vectorized {fast} vs brute force {slow}");
        assert!(deal >= slow - 1e-9, "deal-BR {deal} below infoset BR {slow}");
        assert!(slow < 1.0, "infoset BR on solved 4x4 jam/check {slow} bb (want < 1.0)");
    }

    /// (review 2026-09-20 D7) Blocker-coupled ranges: the deal sampler draws
    /// the TRUE joint, and the reported exploitability is the true-game value.
    #[test]
    fn blocker_coupled_ranges_sample_and_report_the_true_game() {
        let (root, h1, _h2) = blocker_root();
        let board = root.board.clone();
        let r0 = Range::parse(&root.range_oop, &board).unwrap();
        let r1 = Range::parse(&root.range_ip, &board).unwrap();
        assert_eq!((r0.live_combos(), r1.live_combos()), (2, 6));
        // True joint: AcAd pairs with 1 AA combo, KcKd with all 6 ⇒ P(AcAd)=1/7
        // (sequential c0-then-c1 sampling dealt it 1/2 of the time).
        let dealer = DealSampler::new(&board, &r0, &r1).unwrap();
        let mut rng = Lcg::new(11);
        let n = 70_000;
        let hits = (0..n).filter(|_| dealer.sample(&mut rng).1[0] == h1).count();
        let p = hits as f64 / n as f64;
        assert!((p - 1.0 / 7.0).abs() < 0.006, "P(OOP=AcAd)={p}, want 1/7");

        // Reported == brute-force exploitability of the TRUE joint game, both
        // early (large) and after training (small).
        for iters in [10u32, 4000] {
            let (solver, r0, r1) = train(&root, iters, 3);
            let fast = solver.evaluate(&r0, &r1, &RunoutPlan::All, None).unwrap();
            let (slow, _) = reference::brute_force_expl(&solver, &r0, &r1);
            assert!((fast - slow).abs() < 1e-9, "iters={iters}: {fast} vs {slow}");
            if iters == 4000 {
                // The review measured a 0.35 bb plateau here (wrong game).
                assert!(fast < 0.1, "true-game expl after 4000 iters = {fast} bb");
            }
        }
    }

    /// Exact fallback path of the sampler: acceptance ~0.1% by construction.
    #[test]
    fn deal_sampler_exact_fallback_is_in_range() {
        let board = [0u8, 21, 30, 36, 47];
        let r0 = Range::parse("AcAd", &board).unwrap();
        let r1 = Range::parse("AcAh:1000,KcKd:1", &board).unwrap();
        let dealer = DealSampler::new(&board, &r0, &r1).unwrap();
        let mut rng = Lcg::new(5);
        for _ in 0..200 {
            let (_, combos) = dealer.sample(&mut rng);
            assert_eq!(combos, [cards_to_combo(48, 49), cards_to_combo(44, 45)]);
        }
        // Mutually exclusive ranges are refused up front.
        let r1 = Range::parse("AcAh", &board).unwrap();
        assert!(DealSampler::new(&board, &r0, &r1).is_err());
    }

    /// Runouts never collide with the sampled holes (no fix-up hack needed).
    #[test]
    fn sampled_runouts_avoid_holes() {
        let board = [0u8, 5, 10];
        let r = Range::uniform_unblocked(&board);
        let dealer = DealSampler::new(&board, &r, &r).unwrap();
        let mut rng = Lcg::new(9);
        for _ in 0..2000 {
            let (full, combos) = dealer.sample(&mut rng);
            let mut seen = [false; 52];
            for &c in &full {
                assert!(!seen[c as usize]);
                seen[c as usize] = true;
            }
            for &combo in &combos {
                let (a, b) = combo_table()[combo];
                assert!(!seen[a as usize] && !seen[b as usize]);
                seen[a as usize] = true;
                seen[b as usize] = true;
            }
        }
    }

    /// (review 2026-09-20 D8b) time-budget / stop-file exits report the FINAL
    /// estimator, never the poll.
    #[test]
    fn early_exit_paths_report_final_estimator() {
        let root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            10.0,
            vec![3, 17, 22, 40, 51],
            vec![500, 1000],
        );
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 0;
        cfg.time_budget_secs = 0.4;
        cfg.target_exploitability_bb = 0.0;
        cfg.seed = 1;
        let rep = solve_river_dcfr(&root, &cfg).expect("budgeted");
        assert!(rep.notes.iter().any(|n| n == "early_stop=time_budget"));
        assert!(rep.notes.iter().any(|n| n.starts_with("expl_kind=exact_infoset")));
        assert!(rep.exploitability_bb.is_some());

        let stop = std::env::temp_dir().join(format!("cfr_stop_{}.flag", std::process::id()));
        std::fs::write(&stop, b"x").unwrap();
        cfg.time_budget_secs = 0.0;
        cfg.stop_file = stop.to_string_lossy().into_owned();
        let rep = solve_river_dcfr(&root, &cfg).expect("stopped");
        let _ = std::fs::remove_file(&stop);
        assert!(rep.notes.iter().any(|n| n == "early_stop=stop_file"));
        assert!(rep.notes.iter().any(|n| n.starts_with("expl_kind=exact_infoset")));
    }

    /// (review 2026-09-20 D8) target-based early stop fires on a FULL-range
    /// river (it never could: the poll sat ~3 bb above the truth), and the
    /// reported value is the exact one that met the target.
    #[test]
    fn target_early_stop_fires_on_full_range_river() {
        let root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            10.0,
            vec![3, 17, 22, 40, 51],
            vec![],
        );
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 60_000;
        cfg.target_exploitability_bb = 1.0;
        cfg.seed = 2;
        let rep = solve_river_dcfr(&root, &cfg).expect("target");
        assert!(
            rep.notes.iter().any(|n| n == "early_stop=target_exploitability"),
            "notes: {:?}",
            rep.notes
        );
        assert!(rep.iterations_run < 60_000);
        assert!(rep.exploitability_bb.unwrap() <= 1.0);
    }

    /// (review 2026-09-20 D15) turn-street infoset keys must not depend on the
    /// (future) river card.
    #[test]
    fn turn_root_private_view_ignores_future_river() {
        // Turn board uses suits c,d only; the river suit (h vs s) used to
        // reorder the canonical suit map and re-key the same hand.
        let root = RootSpec::postflop_hu(
            StreetRoot::Turn,
            10.0,
            20.0,
            vec![0, 4, 9, 13],
            vec![1000],
        );
        let mut cfg = SolveConfig::default();
        cfg.use_isomorphism = true;
        let mut solver = RiverSolver::new(&root, &cfg).unwrap();
        let combo = cards_to_combo(50, 47); // Ah Ks
        let turn_board = [0u8, 4, 9, 13];
        solver.set_sample_board([0, 4, 9, 13, 22]); // river 7h
        let a = solver.private_view(combo, &turn_board);
        solver.set_sample_board([0, 4, 9, 13, 23]); // river 7s
        let b = solver.private_view(combo, &turn_board);
        assert_eq!(a, b, "future river suit leaked into the turn infoset key");
        // ...and the dump's iso_id agrees with the key.
        assert_eq!(solver.private_meta(combo, a, &turn_board).2, Some(a));
    }

    /// (review 2026-09-20 D8d) Turn root: the vectorized evaluator's chance
    /// node is the exact EXPECTATION over rivers — equal to brute-force
    /// enumeration of every (c0, c1, river) deal, where the responder
    /// re-optimizes per public river but never sees it in advance.
    #[test]
    fn turn_root_br_is_an_expectation_over_rivers() {
        let mut root = RootSpec::postflop_hu(
            StreetRoot::Turn,
            10.0,
            15.0,
            vec![0, 5, 10, 15],
            vec![1000],
        );
        root.range_oop = "AhKh,QsQd,7c6c,2d2h".into();
        root.range_ip = "AsAd,JhTh,9s8s,KcQc".into();
        let (solver, r0, r1) = train(&root, 400, 5);
        let fast = solver.evaluate(&r0, &r1, &RunoutPlan::All, None).unwrap();
        let (slow, deal) = reference::brute_force_expl(&solver, &r0, &r1);
        assert!((fast - slow).abs() < 1e-7, "vectorized {fast} vs brute force {slow}");
        assert!(deal >= slow - 1e-9);
        assert!(fast > 0.0);

        // A "sample" that lists every river exactly once IS the full chance node.
        let all: Vec<[u8; 5]> = (0..52u8)
            .filter(|c| !root.board.contains(c))
            .map(|c| [0, 5, 10, 15, c])
            .collect();
        assert_eq!(all.len(), 48);
        let sampled = solver
            .evaluate(&r0, &r1, &RunoutPlan::Sampled(all), None)
            .unwrap();
        assert!((sampled - fast).abs() < 1e-9, "sampled-all {sampled} vs exact {fast}");

        // A clairvoyant responder (told the river before acting on the turn)
        // does strictly better: that was the old estimator's bias.
        let mut clair = 0.0;
        for c in (0..52u8).filter(|c| !root.board.contains(c)) {
            let one = RunoutPlan::Sampled(vec![[0, 5, 10, 15, c]]);
            clair += solver.evaluate(&r0, &r1, &one, None).unwrap() / 48.0;
        }
        assert!(clair > fast + 1e-6, "clairvoyant {clair} should exceed exact {fast}");
    }

    /// (review 2026-09-20 D8d) turn roots: the final number is an expectation
    /// over rivers; a finite, labelled estimate also comes back for flop roots.
    #[test]
    fn turn_and_flop_roots_report_runout_expectation() {
        let mut root = RootSpec::postflop_hu(
            StreetRoot::Turn,
            10.0,
            10.0,
            vec![0, 5, 10, 15],
            vec![],
        );
        root.range_oop = "AA,KK,QQ,JTs".into();
        root.range_ip = "TT,99,AKs,87s".into();
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 300;
        cfg.target_exploitability_bb = 0.0;
        cfg.seed = 4;
        let rep = solve_postflop_with_runouts(&root, &cfg).expect("turn");
        let expl = rep.exploitability_bb.expect("turn expl");
        assert!(expl.is_finite() && expl >= 0.0);
        assert!(
            rep.notes.iter().any(|n| n.starts_with("expl_kind=exact_infoset")),
            "{:?}",
            rep.notes
        );

        let flop = RootSpec::postflop_hu(StreetRoot::Flop, 8.0, 8.0, vec![0, 5, 10], vec![]);
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 60;
        cfg.target_exploitability_bb = 0.0;
        let rep = solve_postflop_with_runouts(&flop, &cfg).expect("flop");
        assert!(rep.exploitability_bb.expect("flop expl").is_finite());
        assert!(rep
            .notes
            .iter()
            .any(|n| n.starts_with("expl_kind=sampled_runout_br")));
        assert!(rep.notes.iter().any(|n| n.starts_with("expl_scope=abstract_game")));
    }

    /// Ranges that do not parse are errors, not a uniform solve.
    #[test]
    fn bad_range_is_an_error() {
        let mut root = tiny_root();
        root.range_oop = "AKx".into();
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 5;
        let err = solve_river_dcfr(&root, &cfg).unwrap_err();
        assert!(format!("{err}").contains("range_oop"), "{err}");
    }

    /// Unknown algorithm tags are rejected; known ones are labelled truthfully.
    #[test]
    fn algorithm_labels_are_truthful() {
        let root = tiny_root();
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 20;
        cfg.target_exploitability_bb = 0.0;
        for (algo, want) in [
            ("dcfr", "DCFR River"),
            ("linear", "LinearCFR River"),
            ("vanilla", "CFR(no discount) River"),
            ("mccfr_es", "CFR(no discount) River"),
        ] {
            cfg.algorithm = algo.into();
            let rep = solve_river_dcfr(&root, &cfg).expect(algo);
            assert!(rep.notes[0].starts_with(want), "{algo}: {:?}", rep.notes[0]);
        }
        cfg.algorithm = "dcfr_typo".into();
        assert!(matches!(
            solve_river_dcfr(&root, &cfg),
            Err(CfrError::InvalidConfig(_))
        ));
    }
}
