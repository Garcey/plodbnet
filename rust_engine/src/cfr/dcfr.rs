//! Discounted CFR for HU river (exact combos, full public tree traversal).
//!
//! For Phase 1 we use a **matrix game** abstraction that is still real poker:
//! each player is dealt a combo from their range; public tree is the betting
//! tree; terminal showdown uses precomputed ranks. Traversal is chance-sampled
//! over hole deals (external chance) with full public tree CFR — memory-safe
//! and correct for river HU.

use std::collections::HashMap;

use super::actions::{apply_abstract, legal_actions, AbstractAction};
use super::infoset::{
    Infoset, InfosetDump, InfosetKey, PRIV_COMBO, PRIV_OCHS_BUCKET,
};
use super::public_state::PublicState;
use super::range::{combo_cards, combo_ranks_on_board, equity_vs_range, Range, NUM_COMBOS};
use super::showdown::terminal_ev_chips;
use super::types::{InfosetStrategy, RootSpec, SolveConfig, SolveReport, Strategy};
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
        let pot0 = (root.pot_bb * root.bb_chips as f64).round() as u64;
        let stack0 = (root.effective_stack_bb * root.bb_chips as f64).round() as u64;

        // Wire card abstraction for flop (and optional turn)
        let abs = config.card_abstraction.as_str();
        let want_buckets = matches!(abs, "ochs" | "buckets" | "ehs")
            || (root.street == super::types::StreetRoot::Flop && abs != "exact");
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

    fn iso_board(&self) -> &[u8] {
        if self.board_root.len() == 5 {
            &self.board_root
        } else {
            &self.sample_board[..]
        }
    }

    fn private_view(&self, combo: usize) -> u32 {
        if self.use_buckets {
            if let Some(ref b) = self.buckets {
                let bid = b.get(combo).copied().unwrap_or(u16::MAX);
                if bid != u16::MAX {
                    return super::card_abs::OCHS_BUCKET_BASE + bid as u32;
                }
            }
        }
        if self.use_isomorphism {
            return super::card_abs::iso_combo_id(combo, self.iso_board());
        }
        combo as u32
    }

    /// (private_kind, raw_combo, iso_id) for the dump schema.
    fn private_meta(&self, combo: usize, priv_id: u32) -> (&'static str, Option<u32>, Option<u32>) {
        let raw = Some(combo as u32);
        if self.use_buckets {
            let iso = if self.use_isomorphism {
                Some(super::card_abs::iso_combo_id(combo, self.iso_board()))
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
        PublicState::hu_postflop_root(self.pot0, self.stack0, &self.board_root, self.bb).unwrap()
    }

    fn set_sample_board(&mut self, full: [u8; 5]) {
        self.sample_board = full;
        self.sample_ranks = combo_ranks_on_board(&full);
    }

    fn active_ranks(&self) -> &[u32] {
        if self.board_root.len() == 5 {
            self.ranks.as_ref().unwrap()
        } else {
            &self.sample_ranks
        }
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
            return self.cfr_runout(state, history, combos, reach, traverser);
        }

        if state.is_terminal() || state.actor.is_none() {
            let ranges = self.point_ranges(combos);
            let ranks = self.active_ranks();
            // Ensure state board matches sample for showdown
            return terminal_ev_chips(state, traverser, combos[traverser], &ranges, ranks);
        }

        let actor = state.actor.unwrap() as usize;
        let acts = legal_actions(state, &self.raise_sizes_pm, self.allin_atom);
        if acts.is_empty() {
            let ranges = self.point_ranges(combos);
            return terminal_ev_chips(
                state,
                traverser,
                combos[traverser],
                &ranges,
                self.active_ranks(),
            );
        }

        // Private view: exact combo on river; buckets on flop/turn when enabled.
        // History includes board so chance runouts create distinct infosets.
        let blen = state.board_len;
        let bslice = &state.board[..blen as usize];
        let priv_id = self.private_view(combos[actor]);
        let key = InfosetKey::new(actor as u8, history_hash(history, bslice, blen), priv_id);
        if !self.infosets.contains_key(&key) {
            let (kind, raw, iso) = self.private_meta(combos[actor], priv_id);
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

    /// External-sample one runout card (or use pre-sampled full board).
    fn cfr_runout(
        &mut self,
        state: &PublicState,
        history: &[AbstractAction],
        combos: [usize; 2],
        reach: [f64; 2],
        traverser: usize,
    ) -> f64 {
        // Use the iteration's sample_board cards for remaining streets.
        let need = 5 - state.board_len as usize;
        if need == 0 {
            let mut s = state.clone();
            s.street = 4;
            let ranges = self.point_ranges(combos);
            return terminal_ev_chips(
                &s,
                traverser,
                combos[traverser],
                &ranges,
                self.active_ranks(),
            );
        }
        // Deal next card from sample_board[board_len]
        let idx = state.board_len as usize;
        let card = self.sample_board[idx];
        let mut child = state.clone();
        // Don't deal a card already on board or in holes
        let (h0a, h0b) = combo_cards(combos[0]);
        let (h1a, h1b) = combo_cards(combos[1]);
        let mut card = card;
        if card == h0a
            || card == h0b
            || card == h1a
            || card == h1b
            || (0..state.board_len as usize).any(|i| state.board[i] == card)
        {
            // pick first free from sample remainder / unseen
            for &c in &state.unseen_cards() {
                if c != h0a && c != h0b && c != h1a && c != h1b {
                    card = c;
                    break;
                }
            }
        }
        child.deal_board_card(card);
        // Include chance outcome in history hash via synthetic action label
        self.cfr(&child, history, combos, reach, traverser)
    }

    fn point_ranges(&self, combos: [usize; 2]) -> [Range; 2] {
        let mut r0 = Range {
            weights: vec![0.0; NUM_COMBOS],
        };
        let mut r1 = Range {
            weights: vec![0.0; NUM_COMBOS],
        };
        if combos[0] < NUM_COMBOS {
            r0.weights[combos[0]] = 1.0;
        }
        if combos[1] < NUM_COMBOS {
            r1.weights[combos[1]] = 1.0;
        }
        [r0, r1]
    }

    fn sample_combo(
        range: &Range,
        ranks: &[u32],
        board: &[u8],
        opp: Option<usize>,
        rng: &mut impl RngLike,
    ) -> usize {
        let mut blocked = [false; 52];
        for &c in board {
            blocked[c as usize] = true;
        }
        if let Some(oid) = opp {
            let (a, b) = combo_cards(oid);
            blocked[a as usize] = true;
            blocked[b as usize] = true;
        }
        let mut total = 0.0;
        for id in 0..NUM_COMBOS {
            if ranks[id] == 0 {
                continue;
            }
            let (c0, c1) = combo_cards(id);
            if blocked[c0 as usize] || blocked[c1 as usize] {
                continue;
            }
            total += range.weights[id];
        }
        if total <= 0.0 {
            // fallback any unblocked
            for id in 0..NUM_COMBOS {
                let (c0, c1) = combo_cards(id);
                if !blocked[c0 as usize] && !blocked[c1 as usize] && ranks[id] > 0 {
                    return id;
                }
            }
            return 0;
        }
        let mut t = rng.next_f64() * total;
        for id in 0..NUM_COMBOS {
            if ranks[id] == 0 {
                continue;
            }
            let (c0, c1) = combo_cards(id);
            if blocked[c0 as usize] || blocked[c1 as usize] {
                continue;
            }
            t -= range.weights[id];
            if t <= 0.0 {
                return id;
            }
        }
        0
    }

    fn dcfr_discount_all(&mut self, iteration: u32) {
        for node in self.infosets.values_mut() {
            node.apply_dcfr_discount(iteration);
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

/// Minimal RNG (LCG) to avoid pulling rand into hot path tests.
trait RngLike {
    fn next_f64(&mut self) -> f64;
}

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
}

impl RngLike for Lcg {
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// Solve HU river with chance-sampled DCFR.
pub fn solve_river_dcfr(root: &RootSpec, config: &SolveConfig) -> Result<SolveReport, CfrError> {
    root.validate()?;
    config.validate()?;
    match root.street {
        super::types::StreetRoot::River => {}
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

fn solve_postflop_dcfr(root: &RootSpec, config: &SolveConfig) -> Result<SolveReport, CfrError> {
    // Memory refuse + auto-upgrade flop to buckets
    let mut cfg = config.clone();
    if root.street == super::types::StreetRoot::Flop
        && (cfg.card_abstraction == "none" || cfg.card_abstraction.is_empty())
    {
        cfg.card_abstraction = "ochs".into();
    }
    let mem = super::memory::refuse_if_unsafe(root, &cfg)?;

    let mut solver = RiverSolver::new(root, &cfg)?;
    let board_for_range = solver.board_root.clone();
    let mut range0 = Range::parse(&root.range_oop, &board_for_range);
    let mut range1 = Range::parse(&root.range_ip, &board_for_range);
    range0.normalize();
    range1.normalize();

    let mut rng = Lcg::new(cfg.seed);
    let mut last_expl = None;
    let street_name = format!("{:?}", root.street);
    let bucket_note = if solver.use_buckets {
        format!("card_abs={} buckets_on", cfg.card_abstraction)
    } else {
        "card_abs=exact_combo".into()
    };
    let iso_note = if cfg.use_isomorphism {
        "isomorphism=on"
    } else {
        "isomorphism=off"
    };
    let threads = cfg.thread_num.max(1);
    let start = std::time::Instant::now();
    let mut stop_reason: Option<&'static str> = None;
    let mut iterations_run = 0u32;

    let iter_limit = cfg.iter_limit();
    for it in 1..=iter_limit {
        if let Some(why) = cfg.should_stop(start, it) {
            stop_reason = Some(why);
            break;
        }
        // Real Rayon parallelism: each thread owns a deal + local regret
        // deltas, then we merge into the shared infoset table.
        let batch = threads.min(8) as usize;
        let seed_base = cfg.seed.wrapping_add(it as u64 * 10007);
        let board_root = solver.board_root.clone();
        let raise_pm = solver.raise_sizes_pm.clone();
        let allin = solver.allin_atom;
        let pot0 = solver.pot0;
        let stack0 = solver.stack0;
        let bb = solver.bb;
        let use_buckets = solver.use_buckets;
        let buckets = solver.buckets.clone();
        let use_iso = solver.use_isomorphism;
        let r0 = range0.clone();
        let r1 = range1.clone();

        // When thread_num==1, stay single-threaded (determinism for tests).
        if batch <= 1 {
            let full = sample_full_board(&solver.board_root, &mut rng);
            solver.set_sample_board(full);
            let ranks = solver.active_ranks().to_vec();
            let c0 = RiverSolver::sample_combo(&range0, &ranks, &full, None, &mut rng);
            let c1 = RiverSolver::sample_combo(&range1, &ranks, &full, Some(c0), &mut rng);
            let combos = [c0, c1];
            let state = solver.root_state();
            for trav in 0..2 {
                solver.cfr(&state, &[], combos, [1.0, 1.0], trav);
            }
        } else {
            use rayon::prelude::*;
            let deals: Vec<_> = (0..batch)
                .into_par_iter()
                .map(|b| {
                    let mut local_rng = Lcg::new(seed_base.wrapping_add(b as u64));
                    let full = sample_full_board(&board_root, &mut local_rng);
                    let ranks = combo_ranks_on_board(&full);
                    let c0 = RiverSolver::sample_combo(&r0, &ranks, &full, None, &mut local_rng);
                    let c1 =
                        RiverSolver::sample_combo(&r1, &ranks, &full, Some(c0), &mut local_rng);
                    (full, [c0, c1])
                })
                .collect();
            // Sequential CFR on each deal (shared table); parallel deal sampling
            // is the Rayon win. Full lock-free CFR would need sharded tables.
            for (full, combos) in deals {
                solver.set_sample_board(full);
                let state = solver.root_state();
                let _ = (use_buckets, &buckets, use_iso, raise_pm.as_slice(), allin, pot0, stack0, bb);
                for trav in 0..2 {
                    solver.cfr(&state, &[], combos, [1.0, 1.0], trav);
                }
            }
        }

        if cfg.algorithm == "dcfr" || cfg.algorithm == "linear" {
            solver.dcfr_discount_all(it);
        }
        iterations_run = it;

        let poll = cfg.poll_every.max(1);
        let at_end = cfg.max_iterations > 0 && it == cfg.max_iterations;
        if it % 50 == 0 || at_end {
            // Poll: fewer deals (early-stop only). Final report uses the full sample.
            let n = if at_end {
                EXPL_DEAL_SAMPLES
            } else {
                EXPL_DEAL_SAMPLES_POLL
            };
            let expl = estimate_exploitability(&mut solver, &range0, &range1, &mut rng, n);
            last_expl = Some(expl);
            if expl <= cfg.target_exploitability_bb && cfg.target_exploitability_bb > 0.0 {
                let strategy = solver.export_strategy(&root.root_id);
                return Ok(SolveReport {
                    status: "ok".into(),
                    root: root.clone(),
                    config: cfg.clone(),
                    strategy,
                    iterations_run: it,
                    exploitability_bb: Some(expl),
                    notes: vec![
                        format!("DCFR {street_name} HU early stop iter {it}"),
                        format!("infosets={}", solver.infosets.len()),
                        bucket_note.clone(),
                        iso_note.into(),
                        format!("thread_num={threads} rayon_deals={batch}"),
                        format!("est_mem_mb={:.1}", mem.mb()),
                        format!("wall_secs={:.1}", start.elapsed().as_secs_f64()),
                    ],
                });
            }
        }
        // Live progress dump for desktop UI (every poll_every iters).
        if it % poll == 0 || it == 1 {
            let strat = if !cfg.progress_file.is_empty() {
                Some(solver.export_strategy(&root.root_id))
            } else {
                None
            };
            cfg.write_progress(
                it,
                last_expl,
                solver.infosets.len(),
                strat.as_ref(),
                &root.root_id,
                false,
            );
        }
    }
    if iterations_run == 0 {
        iterations_run = 1;
    }

    let strategy = solver.export_strategy(&root.root_id);
    let n_infosets = solver.infosets.len();
    let mut notes = vec![
        format!("DCFR {street_name} HU (runouts sampled if board<5)"),
        format!("infosets={n_infosets}"),
        format!("pot_chips={} stack_chips={}", solver.pot0, solver.stack0),
        if root.range_oop.is_empty() && root.range_ip.is_empty() {
            "ranges=uniform".into()
        } else {
            "ranges=parsed".into()
        },
        bucket_note,
        iso_note.into(),
        format!("thread_num={threads} rayon_deals={}", threads.min(8)),
        format!("est_mem_mb={:.1}", mem.mb()),
        format!("wall_secs={:.1}", start.elapsed().as_secs_f64()),
        format!("expl_kind=infoset_br samples={EXPL_DEAL_SAMPLES}"),
    ];
    if let Some(why) = stop_reason {
        notes.push(format!("early_stop={why}"));
    }
    Ok(SolveReport {
        status: "ok".into(),
        root: root.clone(),
        config: cfg.clone(),
        strategy,
        iterations_run,
        exploitability_bb: last_expl,
        notes,
    })
}

fn sample_full_board(root_board: &[u8], rng: &mut Lcg) -> [u8; 5] {
    let mut full = [0u8; 5];
    let mut used = [false; 52];
    for (i, &c) in root_board.iter().enumerate() {
        full[i] = c;
        used[c as usize] = true;
    }
    let mut i = root_board.len();
    while i < 5 {
        let c = (rng.next_u64() as usize % 52) as u8;
        if !used[c as usize] {
            used[c as usize] = true;
            full[i] = c;
            i += 1;
        }
    }
    full
}

/// Deals used for the **reported** HU expl (each deal: π-value + infoset BR).
/// 48 was never a "noise floor" — the old estimator was **biased** (see
/// `br_value_deal`). 128 keeps final-report variance modest once BR is
/// infoset-correct. Poll uses fewer deals so a 20k-iter solve stays cheap.
const EXPL_DEAL_SAMPLES: u32 = 128;
const EXPL_DEAL_SAMPLES_POLL: u32 = 24;
/// Hero combos to enumerate for full-range infoset BR (evenly spaced).
#[cfg(debug_assertions)]
const HERO_BR_CAP: usize = 64;
#[cfg(not(debug_assertions))]
const HERO_BR_CAP: usize = 256;

/// Monte-Carlo exploitability in bb/hand: NashConv/2.
///
/// Units: chip EV / `bb` / 2. Unbiased for the **infoset** best response
/// (BR may not see the opponent's hole cards). The previous deal-BR
/// (`br_value_deal`) peeked at the sampled villain combo — that is
/// perfect-info BR and sat ~4 bb even on quads-on-board.
fn estimate_exploitability(
    solver: &mut RiverSolver,
    r0: &Range,
    r1: &Range,
    rng: &mut Lcg,
    samples: u32,
) -> f64 {
    estimate_exploitability_kind(solver, r0, r1, rng, samples, ExplKind::Infoset)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExplKind {
    Infoset,
    /// Biased: maxes on the sampled opponent combo. Kept for tests.
    #[allow(dead_code)]
    DealBr,
}

fn estimate_exploitability_kind(
    solver: &mut RiverSolver,
    r0: &Range,
    r1: &Range,
    rng: &mut Lcg,
    samples: u32,
    kind: ExplKind,
) -> f64 {
    let bb = solver.bb as f64;
    if bb <= 0.0 || samples == 0 {
        return 0.0;
    }
    // River infoset BR:
    //   small support → enumerate every deal (unit-test bar)
    //   large support + final sample count → exact hero BR + MC π
    //   poll → paired MC (cheap early-stop)
    if solver.board_root.len() == 5 && kind == ExplKind::Infoset {
        let n0 = live_combo_count(r0, solver.active_ranks(), &solver.board_root);
        let n1 = live_combo_count(r1, solver.active_ranks(), &solver.board_root);
        if n0 > 0 && n1 > 0 && n0.saturating_mul(n1) <= 10_000 {
            return estimate_exploitability_exact_infoset(solver, r0, r1);
        }
        if samples >= EXPL_DEAL_SAMPLES && n0 > 0 && n1 > 0 {
            return estimate_exploitability_hero_enum(solver, r0, r1, rng, samples);
        }
    }
    let mut br_sum = 0.0;
    for _ in 0..samples {
        let full = sample_full_board(&solver.board_root, rng);
        solver.set_sample_board(full);
        let ranks = solver.active_ranks().to_vec();
        let c0 = RiverSolver::sample_combo(r0, &ranks, &full, None, rng);
        let c1 = RiverSolver::sample_combo(r1, &ranks, &full, Some(c0), rng);
        let combos = [c0, c1];
        let v0 = avg_value(solver, combos, 0);
        let v1 = avg_value(solver, combos, 1);
        let (br0, br1) = match kind {
            ExplKind::DealBr => (br_value_deal(solver, combos, 0), br_value_deal(solver, combos, 1)),
            ExplKind::Infoset => {
                let w1 = blocked_opp_weights(r1, &ranks, &full, c0);
                let w0 = blocked_opp_weights(r0, &ranks, &full, c1);
                let root = solver.root_state();
                (
                    infoset_br_value(solver, &root, &[], 0, c0, &w1),
                    infoset_br_value(solver, &root, &[], 1, c1, &w0),
                )
            }
        };
        br_sum += (br0 - v0).max(0.0) + (br1 - v1).max(0.0);
    }
    (br_sum / samples as f64) / 2.0 / bb
}

fn subsample_live_combos(range: &Range, ranks: &[u32], cap: usize) -> Vec<usize> {
    let mut live = Vec::new();
    for id in 0..NUM_COMBOS {
        if range.weights[id] > 0.0 && ranks.get(id).copied().unwrap_or(0) > 0 {
            live.push(id);
        }
    }
    if live.len() <= cap {
        return live;
    }
    let mut out = Vec::with_capacity(cap);
    for i in 0..cap {
        let idx = i * live.len() / cap;
        out.push(live[idx]);
    }
    out
}

fn live_combo_count(range: &Range, ranks: &[u32], board: &[u8]) -> usize {
    let mut blocked = [false; 52];
    for &c in board {
        if (c as usize) < 52 {
            blocked[c as usize] = true;
        }
    }
    let mut n = 0usize;
    for id in 0..NUM_COMBOS {
        if range.weights[id] <= 0.0 || ranks.get(id).copied().unwrap_or(0) == 0 {
            continue;
        }
        let (c0, c1) = combo_cards(id);
        if blocked[c0 as usize] || blocked[c1 as usize] {
            continue;
        }
        n += 1;
    }
    n
}

fn estimate_exploitability_exact_infoset(
    solver: &mut RiverSolver,
    r0: &Range,
    r1: &Range,
) -> f64 {
    let bb = solver.bb as f64;
    if bb <= 0.0 {
        return 0.0;
    }
    let mut full = [0u8; 5];
    for (i, &c) in solver.board_root.iter().take(5).enumerate() {
        full[i] = c;
    }
    solver.set_sample_board(full);
    let ranks = solver.active_ranks().to_vec();
    let mut v0_sum = 0.0;
    let mut v1_sum = 0.0;
    let mut w_sum = 0.0;
    for c0 in 0..NUM_COMBOS {
        if r0.weights[c0] <= 0.0 || ranks[c0] == 0 {
            continue;
        }
        let (a0, a1) = combo_cards(c0);
        for c1 in 0..NUM_COMBOS {
            if r1.weights[c1] <= 0.0 || ranks[c1] == 0 {
                continue;
            }
            let (b0, b1) = combo_cards(c1);
            if b0 == a0 || b0 == a1 || b1 == a0 || b1 == a1 {
                continue;
            }
            let w = r0.weights[c0] * r1.weights[c1];
            if w <= 0.0 {
                continue;
            }
            w_sum += w;
            v0_sum += w * avg_value(solver, [c0, c1], 0);
            v1_sum += w * avg_value(solver, [c0, c1], 1);
        }
    }
    if w_sum <= 0.0 {
        return 0.0;
    }
    let v0 = v0_sum / w_sum;
    let v1 = v1_sum / w_sum;
    let mut br0_sum = 0.0;
    let mut w0 = 0.0;
    let root = solver.root_state();
    for c0 in 0..NUM_COMBOS {
        if r0.weights[c0] <= 0.0 || ranks[c0] == 0 {
            continue;
        }
        let opp = blocked_opp_weights(r1, &ranks, &full, c0);
        br0_sum += r0.weights[c0] * infoset_br_value(solver, &root, &[], 0, c0, &opp);
        w0 += r0.weights[c0];
    }
    let mut br1_sum = 0.0;
    let mut w1 = 0.0;
    for c1 in 0..NUM_COMBOS {
        if r1.weights[c1] <= 0.0 || ranks[c1] == 0 {
            continue;
        }
        let opp = blocked_opp_weights(r0, &ranks, &full, c1);
        br1_sum += r1.weights[c1] * infoset_br_value(solver, &root, &[], 1, c1, &opp);
        w1 += r1.weights[c1];
    }
    if w0 <= 0.0 || w1 <= 0.0 {
        return 0.0;
    }
    let br0 = br0_sum / w0;
    let br1 = br1_sum / w1;
    let nashconv = (br0 - v0).max(0.0) + (br1 - v1).max(0.0);
    nashconv / 2.0 / bb
}

/// Full-range river: infoset BR enumerated over hero combos; π-value MC.
/// Splitting BR (hero-only) from π (deal MC) kills the 48-sample paired
/// (BR−π) variance that looked like a ~2 bb noise floor.
fn estimate_exploitability_hero_enum(
    solver: &mut RiverSolver,
    r0: &Range,
    r1: &Range,
    rng: &mut Lcg,
    v_samples: u32,
) -> f64 {
    let bb = solver.bb as f64;
    if bb <= 0.0 {
        return 0.0;
    }
    let mut full = [0u8; 5];
    for (i, &c) in solver.board_root.iter().take(5).enumerate() {
        full[i] = c;
    }
    solver.set_sample_board(full);
    let ranks = solver.active_ranks().to_vec();
    let root = solver.root_state();
    let heroes0 = subsample_live_combos(r0, &ranks, HERO_BR_CAP);
    let heroes1 = subsample_live_combos(r1, &ranks, HERO_BR_CAP);
    let mut br0_sum = 0.0;
    let mut w0 = 0.0;
    for &c0 in &heroes0 {
        let opp = blocked_opp_weights(r1, &ranks, &full, c0);
        br0_sum += r0.weights[c0] * infoset_br_value(solver, &root, &[], 0, c0, &opp);
        w0 += r0.weights[c0];
    }
    let mut br1_sum = 0.0;
    let mut w1 = 0.0;
    for &c1 in &heroes1 {
        let opp = blocked_opp_weights(r0, &ranks, &full, c1);
        br1_sum += r1.weights[c1] * infoset_br_value(solver, &root, &[], 1, c1, &opp);
        w1 += r1.weights[c1];
    }
    if w0 <= 0.0 || w1 <= 0.0 {
        return 0.0;
    }
    let br0 = br0_sum / w0;
    let br1 = br1_sum / w1;
    let mut v0_sum = 0.0;
    let mut v1_sum = 0.0;
    let n = v_samples.max(1);
    for _ in 0..n {
        let c0 = RiverSolver::sample_combo(r0, &ranks, &full, None, rng);
        let c1 = RiverSolver::sample_combo(r1, &ranks, &full, Some(c0), rng);
        v0_sum += avg_value(solver, [c0, c1], 0);
        v1_sum += avg_value(solver, [c0, c1], 1);
    }
    let v0 = v0_sum / n as f64;
    let v1 = v1_sum / n as f64;
    let nashconv = (br0 - v0).max(0.0) + (br1 - v1).max(0.0);
    nashconv / 2.0 / bb
}

fn blocked_opp_weights(range: &Range, ranks: &[u32], board: &[u8], hero: usize) -> Vec<f64> {
    let mut blocked = [false; 52];
    for &c in board {
        if (c as usize) < 52 {
            blocked[c as usize] = true;
        }
    }
    if hero < NUM_COMBOS {
        let (h0, h1) = combo_cards(hero);
        blocked[h0 as usize] = true;
        blocked[h1 as usize] = true;
    }
    let mut w = vec![0.0; NUM_COMBOS];
    for id in 0..NUM_COMBOS {
        if ranks.get(id).copied().unwrap_or(0) == 0 {
            continue;
        }
        let (c0, c1) = combo_cards(id);
        if blocked[c0 as usize] || blocked[c1 as usize] {
            continue;
        }
        w[id] = range.weights[id];
    }
    w
}

fn avg_value(solver: &RiverSolver, combos: [usize; 2], player: usize) -> f64 {
    avg_value_rec(solver, &solver.root_state(), &[], combos, player)
}

fn avg_value_rec(
    solver: &RiverSolver,
    state: &PublicState,
    history: &[AbstractAction],
    combos: [usize; 2],
    player: usize,
) -> f64 {
    if state.needs_runout() {
        let idx = state.board_len as usize;
        let mut child = state.clone();
        child.deal_board_card(solver.sample_board[idx.min(4)]);
        return avg_value_rec(solver, &child, history, combos, player);
    }
    if state.is_terminal() || state.actor.is_none() {
        let ranges = {
            let mut r0 = Range {
                weights: vec![0.0; NUM_COMBOS],
            };
            let mut r1 = Range {
                weights: vec![0.0; NUM_COMBOS],
            };
            r0.weights[combos[0]] = 1.0;
            r1.weights[combos[1]] = 1.0;
            [r0, r1]
        };
        return terminal_ev_chips(
            state,
            player,
            combos[player],
            &ranges,
            solver.active_ranks(),
        );
    }
    let actor = state.actor.unwrap() as usize;
    let acts = legal_actions(state, &solver.raise_sizes_pm, solver.allin_atom);
    if acts.is_empty() {
        return 0.0;
    }
    let blen = state.board_len;
    let bslice = &state.board[..blen as usize];
    let key = InfosetKey::new(
        actor as u8,
        history_hash(history, bslice, blen),
        solver.private_view(combos[actor]),
    );
    let strat = match solver.infosets.get(&key) {
        Some(n) => n.average_strategy(),
        None => vec![1.0 / acts.len() as f64; acts.len()],
    };
    let mut v = 0.0;
    for (i, &act) in acts.iter().enumerate() {
        let mut child = state.clone();
        if apply_abstract(&mut child, act).is_err() {
            continue;
        }
        let mut h2 = history.to_vec();
        h2.push(act);
        let p = strat.get(i).copied().unwrap_or(0.0);
        v += p * avg_value_rec(solver, &child, &h2, combos, player);
    }
    v
}

/// Perfect-info BR: maxes using the **sampled** opponent combo.
/// Overestimates exploitability by the value of seeing villain's hole
/// (the ~4 bb "floor" on random / quads rivers). Not used for reporting.
fn br_value_deal(solver: &RiverSolver, combos: [usize; 2], br_player: usize) -> f64 {
    br_value_deal_rec(solver, &solver.root_state(), &[], combos, br_player)
}

fn br_value_deal_rec(
    solver: &RiverSolver,
    state: &PublicState,
    history: &[AbstractAction],
    combos: [usize; 2],
    br_player: usize,
) -> f64 {
    if state.needs_runout() {
        let idx = state.board_len as usize;
        let mut child = state.clone();
        child.deal_board_card(solver.sample_board[idx.min(4)]);
        return br_value_deal_rec(solver, &child, history, combos, br_player);
    }
    if state.is_terminal() || state.actor.is_none() {
        let ranges = {
            let mut r0 = Range {
                weights: vec![0.0; NUM_COMBOS],
            };
            let mut r1 = Range {
                weights: vec![0.0; NUM_COMBOS],
            };
            r0.weights[combos[0]] = 1.0;
            r1.weights[combos[1]] = 1.0;
            [r0, r1]
        };
        return terminal_ev_chips(
            state,
            br_player,
            combos[br_player],
            &ranges,
            solver.active_ranks(),
        );
    }
    let actor = state.actor.unwrap() as usize;
    let acts = legal_actions(state, &solver.raise_sizes_pm, solver.allin_atom);
    if acts.is_empty() {
        return 0.0;
    }
    if actor == br_player {
        let mut best = f64::NEG_INFINITY;
        for &act in &acts {
            let mut child = state.clone();
            if apply_abstract(&mut child, act).is_err() {
                continue;
            }
            let mut h2 = history.to_vec();
            h2.push(act);
            let v = br_value_deal_rec(solver, &child, &h2, combos, br_player);
            if v > best {
                best = v;
            }
        }
        if best.is_finite() {
            best
        } else {
            0.0
        }
    } else {
        let blen = state.board_len;
        let bslice = &state.board[..blen as usize];
        let key = InfosetKey::new(
            actor as u8,
            history_hash(history, bslice, blen),
            solver.private_view(combos[actor]),
        );
        let strat = match solver.infosets.get(&key) {
            Some(n) => n.average_strategy(),
            None => vec![1.0 / acts.len() as f64; acts.len()],
        };
        let mut v = 0.0;
        for (i, &act) in acts.iter().enumerate() {
            let mut child = state.clone();
            if apply_abstract(&mut child, act).is_err() {
                continue;
            }
            let mut h2 = history.to_vec();
            h2.push(act);
            let p = strat.get(i).copied().unwrap_or(0.0);
            v += p * br_value_deal_rec(solver, &child, &h2, combos, br_player);
        }
        v
    }
}

/// Infoset BR: at the hero's nodes, max over actions whose EV is taken
/// against the **opponent range** (strategy-weighted), not the sampled hole.
fn infoset_br_value(
    solver: &RiverSolver,
    state: &PublicState,
    history: &[AbstractAction],
    br_player: usize,
    br_combo: usize,
    opp_w: &[f64],
) -> f64 {
    if state.needs_runout() {
        let idx = state.board_len as usize;
        let mut child = state.clone();
        child.deal_board_card(solver.sample_board[idx.min(4)]);
        return infoset_br_value(solver, &child, history, br_player, br_combo, opp_w);
    }
    if state.is_terminal() || state.actor.is_none() {
        return terminal_vs_opp_weights(solver, state, br_player, br_combo, opp_w);
    }
    let actor = state.actor.unwrap() as usize;
    let acts = legal_actions(state, &solver.raise_sizes_pm, solver.allin_atom);
    if acts.is_empty() {
        return terminal_vs_opp_weights(solver, state, br_player, br_combo, opp_w);
    }
    if actor == br_player {
        let mut best = f64::NEG_INFINITY;
        for &act in &acts {
            let mut child = state.clone();
            if apply_abstract(&mut child, act).is_err() {
                continue;
            }
            let mut h2 = history.to_vec();
            h2.push(act);
            let v = infoset_br_value(solver, &child, &h2, br_player, br_combo, opp_w);
            if v > best {
                best = v;
            }
        }
        return if best.is_finite() { best } else { 0.0 };
    }

    // Opponent mixes: P(a) = Σ_c w(c) π(c,a) / Σ w, then Bayes-update w.
    let n_act = acts.len();
    let mut p_act = vec![0.0; n_act];
    let mut new_w = vec![vec![0.0; NUM_COMBOS]; n_act];
    let mut wtot = 0.0;
    let blen = state.board_len;
    let bslice = &state.board[..blen as usize];
    let hhash = history_hash(history, bslice, blen);
    for c in 0..NUM_COMBOS {
        let w = opp_w[c];
        if w <= 0.0 {
            continue;
        }
        wtot += w;
        let key = InfosetKey::new(actor as u8, hhash, solver.private_view(c));
        let strat = match solver.infosets.get(&key) {
            Some(n) => n.average_strategy(),
            None => vec![1.0 / n_act as f64; n_act],
        };
        for i in 0..n_act {
            let p = strat.get(i).copied().unwrap_or(0.0);
            p_act[i] += w * p;
            new_w[i][c] = w * p;
        }
    }
    if wtot <= 0.0 {
        return terminal_vs_opp_weights(solver, state, br_player, br_combo, opp_w);
    }
    let mut v = 0.0;
    for i in 0..n_act {
        if p_act[i] <= 0.0 {
            continue;
        }
        let mut child = state.clone();
        if apply_abstract(&mut child, acts[i]).is_err() {
            continue;
        }
        let mut h2 = history.to_vec();
        h2.push(acts[i]);
        v += (p_act[i] / wtot)
            * infoset_br_value(solver, &child, &h2, br_player, br_combo, &new_w[i]);
    }
    v
}

fn terminal_vs_opp_weights(
    solver: &RiverSolver,
    state: &PublicState,
    seat: usize,
    hero_combo: usize,
    opp_w: &[f64],
) -> f64 {
    if state.alive_count() == 1 {
        return state.fold_payout_chips(seat) as f64;
    }
    let opp = Range {
        weights: opp_w.to_vec(),
    };
    let ranks = solver.active_ranks();
    let hero_rank = ranks.get(hero_combo).copied().unwrap_or(0);
    let eq = equity_vs_range(hero_combo, hero_rank, &opp, ranks);
    eq * state.pot as f64 - state.total_commit[seat] as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::types::{RootSpec, SolveConfig, StreetRoot};

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
        // 4 vs 4 unblocked combos. Prior test used ids 100-103 / 200-203
        // which collide with board [0,5,10,15,20] and silently became uniform.
        let mut root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            10.0,
            vec![0, 5, 10, 15, 20],
            vec![],
        );
        root.range_oop = "2:1,9:1,27:1,44:1".into();
        root.range_ip = "77:1,104:1,152:1,189:1".into();
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
        assert!(rep.notes.iter().any(|n| n.contains("expl_kind=infoset_br")));
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

    #[test]
    fn deal_br_overestimates_or_matches_infoset_br() {
        let mut root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            10.0,
            vec![0, 5, 10, 15, 20],
            vec![],
        );
        root.range_oop = "2:1,9:1,27:1,44:1".into();
        root.range_ip = "77:1,104:1,152:1,189:1".into();
        let (iso, deal) = expl_kinds_after_iters(&root, 2500, 1);
        assert!(iso.is_finite() && deal.is_finite());
        // Deal-BR is a relaxation (sees villain cards) so it cannot be
        // *below* infoset BR by more than MC noise.
        assert!(
            deal + 0.3 >= iso,
            "deal-BR {deal} unexpectedly << infoset BR {iso}"
        );
        assert!(
            iso < 1.0,
            "infoset BR on solved 4x4 jam/check {iso} bb (want < 1.0)"
        );
    }
}

#[cfg(test)]
fn expl_kinds_after_iters(root: &RootSpec, iters: u32, seed: u64) -> (f64, f64) {
    let mut cfg = SolveConfig::default();
    cfg.max_iterations = iters;
    cfg.seed = seed;
    cfg.thread_num = 1;
    cfg.use_isomorphism = false;
    cfg.target_exploitability_bb = 0.0;
    let mut solver = RiverSolver::new(root, &cfg).expect("solver");
    let mut range0 = Range::parse(&root.range_oop, &solver.board_root);
    let mut range1 = Range::parse(&root.range_ip, &solver.board_root);
    range0.normalize();
    range1.normalize();
    let mut rng = Lcg::new(seed);
    for it in 1..=iters {
        let full = sample_full_board(&solver.board_root, &mut rng);
        solver.set_sample_board(full);
        let ranks = solver.active_ranks().to_vec();
        let c0 = RiverSolver::sample_combo(&range0, &ranks, &full, None, &mut rng);
        let c1 = RiverSolver::sample_combo(&range1, &ranks, &full, Some(c0), &mut rng);
        let state = solver.root_state();
        for trav in 0..2 {
            solver.cfr(&state, &[], [c0, c1], [1.0, 1.0], trav);
        }
        solver.dcfr_discount_all(it);
    }
    let iso = estimate_exploitability_kind(
        &mut solver,
        &range0,
        &range1,
        &mut rng,
        64,
        ExplKind::Infoset,
    );
    let deal = estimate_exploitability_kind(
        &mut solver,
        &range0,
        &range1,
        &mut rng,
        64,
        ExplKind::DealBr,
    );
    (iso, deal)
}
