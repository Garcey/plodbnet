//! Vectorized (range-vs-range) DCFR for heads-up river and turn roots (TOOL-008).
//!
//! The chance-sampled solver in `dcfr.rs` deals ONE hand pair per iteration, so
//! a river root needs tens of thousands of iterations to get under 1 bb. This
//! solver is the standard public-tree design of commercial solvers: the public
//! betting tree is built once (for turn roots: one river subtree per river
//! card), every decision node holds regrets and strategy sums for all 1326 hands
//! of its actor, and one iteration walks the whole tree with 1326-wide reach and
//! counterfactual-value vectors — every hand pair and every river, exactly. DCFR
//! discounting (alpha 1.5, beta 0, gamma 2) with alternating updates, the same
//! update rule as the sampled solver in expectation.
//!
//! - Showdowns use the same sorted-rank sweep as the exact evaluator, so a
//!   terminal costs O(1326), not O(1326²).
//! - Turn roots walk their river subtrees in parallel (a pool of `thread_num`
//!   threads); per-card results are combined in card order, so the result does
//!   not depend on the thread count.
//! - Exploitability is an exact vectorized best response on the same tree
//!   (`expl_kind=exact_infoset`), pinned against `dcfr.rs`'s evaluator.
//! - The report has the same rows / ids / dump fields as the sampled solver, plus
//!   per-hand EV and equity at every decision node (TOOL-035).
//!
//! Selected with `algorithm = "dcfr_vector"`; flop roots stay with the sampled
//! bucketed solver (every turn and river under a flop tree is too large).

use std::time::{Duration, Instant};

use rayon::prelude::*;

use super::actions::{apply_abstract, legal_actions, AbstractAction};
use super::dcfr::{reach_totals, ShowdownOrder};
use super::hashing::{action_step, history_key, EMPTY_ACTIONS};
use super::infoset::{Infoset, InfosetDump, PRIV_COMBO};
use super::memory::{
    estimate_solve_memory, MemoryEstimate, VECTOR_BYTES_PER_ACTION, VECTOR_NODE_OVERHEAD_BYTES,
};
use super::public_state::PublicState;
use super::range::{combo_ranks_on_board, combo_table, Range, NUM_COMBOS};
use super::types::{
    InfosetStrategy, RootSpec, SnapshotPacer, SolveConfig, SolveReport, Strategy, StreetRoot,
};
use super::CfrError;

const N: usize = NUM_COMBOS;
/// An exploitability check every this many iterations when a target is set
/// (one exact check costs about as much as one iteration).
const TARGET_CHECK_EVERY: u32 = 10;

/// One street's betting tree on ONE board: every node shares the board, so the
/// showdown order and the live-hand masks are per tree. Chance nodes own one
/// subtree per next card.
struct VTree {
    nodes: Vec<VNode>,
    /// Showdown order of this tree's board (complete boards only).
    order: Option<ShowdownOrder>,
    /// `live[p][h]`: player p holds h with positive range weight, and h does
    /// not use a card of this board.
    live: [Vec<bool>; 2],
}

struct VNode {
    state: PublicState,
    history: Vec<AbstractAction>,
    ahash: u64,
    kind: VKind,
}

enum VKind {
    Terminal,
    Chance(Vec<(u8, VTree)>),
    Decision(Box<Decision>),
}

struct Decision {
    actor: usize,
    actions: Vec<AbstractAction>,
    /// Child node per action (None: the action could not be applied — a dead
    /// action worth 0, exactly as in the sampled solver).
    children: Vec<Option<usize>>,
    /// `regret[a * N + h]`
    regret: Vec<f64>,
    /// `sum[a * N + h]` — the average-strategy accumulator.
    sum: Vec<f64>,
    /// (TOOL-035) per actor hand: (EV bb, equity), filled by the final pass.
    ev: Option<Vec<(f64, f64)>>,
}

/// Read-only data the recursion needs.
struct Ctx {
    /// Range weights, 0 on hands that use a root-board card.
    w: [Vec<f64>; 2],
    parallel: bool,
    bb: f64,
    /// Turn roots: showdown order per river card (equity at turn nodes).
    river_orders: Vec<Option<ShowdownOrder>>,
}

impl VTree {
    fn new(state: &PublicState, root_live: &[Vec<bool>; 2]) -> Self {
        let blen = state.board_len as usize;
        let order = if blen == 5 {
            let mut b = [0u8; 5];
            b.copy_from_slice(&state.board[..5]);
            Some(ShowdownOrder::new(combo_ranks_on_board(&b)))
        } else {
            None
        };
        let table = combo_table();
        let mut mask = 0u64;
        for &c in &state.board[..blen] {
            mask |= 1u64 << c;
        }
        let live_for = |p: usize| -> Vec<bool> {
            (0..N)
                .map(|h| {
                    let (a, b) = table[h];
                    root_live[p][h] && mask & ((1u64 << a) | (1u64 << b)) == 0
                })
                .collect()
        };
        Self {
            nodes: Vec::new(),
            order,
            live: [live_for(0), live_for(1)],
        }
    }
}

struct Builder<'a> {
    raise_pm: &'a [u32],
    allin: bool,
    root_live: [Vec<bool>; 2],
    decision_nodes: u64,
    bytes: u64,
    budget: u64,
}

impl Builder<'_> {
    fn build(
        &mut self,
        tree: &mut VTree,
        state: PublicState,
        history: Vec<AbstractAction>,
        ahash: u64,
    ) -> Result<usize, CfrError> {
        let idx = tree.nodes.len();
        tree.nodes.push(VNode {
            state: state.clone(),
            history: history.clone(),
            ahash,
            kind: VKind::Terminal,
        });
        if state.needs_runout() {
            let blen = state.board_len as usize;
            let mut subs = Vec::new();
            for x in (0..52u8).filter(|c| !state.board[..blen].contains(c)) {
                let mut child = state.clone();
                child.deal_board_card(x);
                let mut sub = VTree::new(&child, &self.root_live);
                self.build(&mut sub, child, history.clone(), ahash)?;
                subs.push((x, sub));
            }
            tree.nodes[idx].kind = VKind::Chance(subs);
            return Ok(idx);
        }
        if state.is_terminal() || state.actor.is_none() {
            return Ok(idx);
        }
        let actions = legal_actions(&state, self.raise_pm, self.allin);
        if actions.is_empty() {
            return Ok(idx);
        }
        let n = actions.len();
        self.decision_nodes += 1;
        self.bytes += n as u64 * VECTOR_BYTES_PER_ACTION + VECTOR_NODE_OVERHEAD_BYTES;
        if self.bytes > self.budget {
            // The up-front estimate should have refused this already.
            return Err(CfrError::InvalidConfig(format!(
                "dcfr_vector: the tree needs more than {:.1} GB (budget {:.1} GB) — use fewer \
                 raise sizes, shallower stacks, or the sampled \"dcfr\" algorithm",
                self.bytes as f64 / (1u64 << 30) as f64,
                self.budget as f64 / (1u64 << 30) as f64
            )));
        }
        let mut children = Vec::with_capacity(n);
        for &act in &actions {
            let mut child = state.clone();
            if apply_abstract(&mut child, act).is_err() {
                children.push(None);
                continue;
            }
            let mut h2 = history.clone();
            h2.push(act);
            children.push(Some(self.build(
                tree,
                child,
                h2,
                action_step(ahash, act),
            )?));
        }
        let actor = state.actor.unwrap() as usize;
        tree.nodes[idx].kind = VKind::Decision(Box::new(Decision {
            actor,
            actions,
            children,
            regret: vec![0.0; n * N],
            sum: vec![0.0; n * N],
            ev: None,
        }));
        Ok(idx)
    }
}

/// Regret matching per hand (`live` hands; the rest keep uniform).
fn current_strategy(regret: &[f64], n: usize, live: &[bool]) -> Vec<f64> {
    let mut sigma = vec![1.0 / n as f64; n * N];
    for h in 0..N {
        if !live[h] {
            continue;
        }
        let mut s = 0.0;
        for a in 0..n {
            s += regret[a * N + h].max(0.0);
        }
        if s > 0.0 {
            for a in 0..n {
                sigma[a * N + h] = regret[a * N + h].max(0.0) / s;
            }
        }
    }
    sigma
}

/// Average strategy per hand (uniform when never reached — the evaluator's rule).
fn average_strategy(sum: &[f64], n: usize, live: &[bool]) -> Vec<f64> {
    let mut sigma = vec![1.0 / n as f64; n * N];
    for h in 0..N {
        if !live[h] {
            continue;
        }
        let s: f64 = (0..n).map(|a| sum[a * N + h]).sum();
        if s > 0.0 {
            for a in 0..n {
                sigma[a * N + h] = sum[a * N + h] / s;
            }
        }
    }
    sigma
}

/// Counterfactual values of `hero` at a terminal:
/// `v[h] = Σ_c opp(c)·[h,c disjoint]·u_hero(h, c)` (u = chips won − chips put in
/// since the root). The same math as `dcfr::EvalCtx::terminal_values`.
fn terminal_values(
    state: &PublicState,
    order: Option<&ShowdownOrder>,
    hero: usize,
    live: &[bool],
    opp: &[f64],
) -> Vec<f64> {
    let table = combo_table();
    let mut v = vec![0.0; N];
    let (total, by_card) = reach_totals(opp);
    if total <= 0.0 {
        return v;
    }
    if state.alive_count() == 1 {
        let pay = state.fold_payout_chips(hero) as f64;
        for h in 0..N {
            if live[h] {
                let (a, b) = table[h];
                v[h] = pay * (total - by_card[a as usize] - by_card[b as usize] + opp[h]);
            }
        }
        return v;
    }
    let Some(order) = order else {
        return v; // cannot happen: a showdown always has a complete board
    };
    let win = win_mass(order, opp, live);
    let (pot, commit) = (state.pot as f64, state.total_commit[hero] as f64);
    for h in 0..N {
        if live[h] && order.ranks[h] > 0 {
            let (a, b) = table[h];
            let compat = total - by_card[a as usize] - by_card[b as usize] + opp[h];
            v[h] = pot * win[h] - commit * compat;
        }
    }
    v
}

/// `Σ_c opp(c)·[h,c disjoint]·(1{h beats c} + ½·1{tie})` on one full board.
fn win_mass(order: &ShowdownOrder, opp: &[f64], live: &[bool]) -> Vec<f64> {
    let table = combo_table();
    let mut v = vec![0.0; N];
    let (mut cum, mut cum_card) = (0.0f64, [0.0f64; 52]);
    let ord = &order.order;
    let mut i = 0;
    while i < ord.len() {
        let rank = order.ranks[ord[i] as usize];
        let mut j = i;
        let (mut grp, mut grp_card) = (0.0f64, [0.0f64; 52]);
        while j < ord.len() && order.ranks[ord[j] as usize] == rank {
            let c = ord[j] as usize;
            let r = opp[c];
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
            if !live[h] {
                continue;
            }
            let (a, b) = (table[h].0 as usize, table[h].1 as usize);
            let own = opp[h];
            v[h] =
                (cum - cum_card[a] - cum_card[b]) + 0.5 * (grp - grp_card[a] - grp_card[b] + own);
        }
        cum += grp;
        for k in 0..52 {
            cum_card[k] += grp_card[k];
        }
        i = j;
    }
    v
}

/// `v` with every hand holding `card` zeroed.
fn without_card(v: &[f64], card: u8) -> Vec<f64> {
    let table = combo_table();
    v.iter()
        .enumerate()
        .map(|(h, &x)| {
            if table[h].0 == card || table[h].1 == card {
                0.0
            } else {
                x
            }
        })
        .collect()
}

/// Combine per-card subtree values at a chance node: the next card is uniform
/// over the `52 - board - 4` cards outside the board and both hands.
fn combine_chance(parts: Vec<(u8, Vec<f64>)>, blen: usize) -> Vec<f64> {
    let table = combo_table();
    let free = (52 - blen - 4) as f64;
    let mut out = vec![0.0; N];
    for (x, v) in parts {
        for h in 0..N {
            if table[h].0 != x && table[h].1 != x {
                out[h] += v[h] / free;
            }
        }
    }
    out
}

fn is_zero(v: &[f64]) -> bool {
    v.iter().all(|&x| x == 0.0)
}

/// One CFR pass for traverser `t` (reach `rs` = t's, `ro` = the opponent's);
/// returns t's counterfactual values at node `idx`.
fn cfr(tree: &mut VTree, idx: usize, t: usize, rs: &[f64], ro: &[f64], ctx: &Ctx) -> Vec<f64> {
    let blen = tree.nodes[idx].state.board_len as usize;
    let (actor, n, children) = match &mut tree.nodes[idx].kind {
        VKind::Terminal => {
            return terminal_values(
                &tree.nodes[idx].state,
                tree.order.as_ref(),
                t,
                &tree.live[t],
                ro,
            );
        }
        VKind::Chance(subs) => {
            let run = |(x, sub): &mut (u8, VTree)| {
                let (rs2, ro2) = (without_card(rs, *x), without_card(ro, *x));
                (*x, cfr(sub, 0, t, &rs2, &ro2, ctx))
            };
            let parts: Vec<(u8, Vec<f64>)> = if ctx.parallel {
                subs.par_iter_mut().map(run).collect()
            } else {
                subs.iter_mut().map(run).collect()
            };
            return combine_chance(parts, blen);
        }
        VKind::Decision(d) => (d.actor, d.actions.len(), d.children.clone()),
    };
    let sigma = match &tree.nodes[idx].kind {
        VKind::Decision(d) => current_strategy(&d.regret, n, &tree.live[actor]),
        _ => unreachable!(),
    };
    if actor != t {
        let mut out = vec![0.0; N];
        for (a, child) in children.iter().enumerate() {
            let Some(c) = child else { continue };
            let ro_a: Vec<f64> = (0..N).map(|h| ro[h] * sigma[a * N + h]).collect();
            if is_zero(&ro_a) && is_zero(rs) {
                continue; // nothing below can change
            }
            let v = cfr(tree, *c, t, rs, &ro_a, ctx);
            for h in 0..N {
                out[h] += v[h];
            }
        }
        return out;
    }
    let mut vals: Vec<Vec<f64>> = Vec::with_capacity(n);
    for (a, child) in children.iter().enumerate() {
        vals.push(match child {
            Some(c) => {
                let rs_a: Vec<f64> = (0..N).map(|h| rs[h] * sigma[a * N + h]).collect();
                cfr(tree, *c, t, &rs_a, ro, ctx)
            }
            None => vec![0.0; N],
        });
    }
    let VTree { nodes, live, .. } = tree;
    let live = &live[t];
    let mut node_v = vec![0.0; N];
    for h in 0..N {
        if live[h] {
            node_v[h] = (0..n).map(|a| sigma[a * N + h] * vals[a][h]).sum();
        }
    }
    if let VKind::Decision(d) = &mut nodes[idx].kind {
        for a in 0..n {
            let (reg, sum) = (
                &mut d.regret[a * N..(a + 1) * N],
                &mut d.sum[a * N..(a + 1) * N],
            );
            for h in 0..N {
                if live[h] {
                    reg[h] += vals[a][h] - node_v[h];
                    sum[h] += rs[h] * sigma[a * N + h];
                }
            }
        }
    }
    node_v
}

/// DCFR end-of-iteration discount on every decision node.
fn discount(tree: &mut VTree, pos: f64, neg: f64, strat: f64, parallel: bool) {
    for node in tree.nodes.iter_mut() {
        match &mut node.kind {
            VKind::Decision(d) => {
                for r in d.regret.iter_mut() {
                    *r *= if *r > 0.0 { pos } else { neg };
                }
                for s in d.sum.iter_mut() {
                    *s *= strat;
                }
            }
            VKind::Chance(subs) => {
                if parallel {
                    subs.par_iter_mut()
                        .for_each(|(_, sub)| discount(sub, pos, neg, strat, false));
                } else {
                    for (_, sub) in subs.iter_mut() {
                        discount(sub, pos, neg, strat, false);
                    }
                }
            }
            VKind::Terminal => {}
        }
    }
}

/// What [`best_response`] returns: the `(best response, on-policy)` per-hand
/// value vectors.
type BrValues = (Vec<f64>, Vec<f64>);

/// `(best response, on-policy)` counterfactual values of `hero` against the
/// other player's AVERAGE strategy. With `collect_ev`, every hero decision
/// node stores its hands' EV / equity (TOOL-035).
fn best_response(
    tree: &mut VTree,
    idx: usize,
    hero: usize,
    ro: &[f64],
    ctx: &Ctx,
    collect_ev: bool,
) -> BrValues {
    let blen = tree.nodes[idx].state.board_len as usize;
    let (actor, n, children) = match &mut tree.nodes[idx].kind {
        VKind::Terminal => {
            let v = terminal_values(
                &tree.nodes[idx].state,
                tree.order.as_ref(),
                hero,
                &tree.live[hero],
                ro,
            );
            return (v.clone(), v);
        }
        VKind::Chance(subs) => {
            let run = |(x, sub): &mut (u8, VTree)| {
                let ro2 = without_card(ro, *x);
                (*x, best_response(sub, 0, hero, &ro2, ctx, collect_ev))
            };
            let parts: Vec<(u8, BrValues)> = if ctx.parallel {
                subs.par_iter_mut().map(run).collect()
            } else {
                subs.iter_mut().map(run).collect()
            };
            let (br_parts, avg_parts): (Vec<_>, Vec<_>) = parts
                .into_iter()
                .map(|(x, (b, a))| ((x, b), (x, a)))
                .unzip();
            return (
                combine_chance(br_parts, blen),
                combine_chance(avg_parts, blen),
            );
        }
        VKind::Decision(d) => (d.actor, d.actions.len(), d.children.clone()),
    };
    let sigma = match &tree.nodes[idx].kind {
        VKind::Decision(d) => average_strategy(&d.sum, n, &tree.live[actor]),
        _ => unreachable!(),
    };
    if actor != hero {
        let (mut br, mut avg) = (vec![0.0; N], vec![0.0; N]);
        for (a, child) in children.iter().enumerate() {
            let Some(c) = child else { continue };
            let ro_a: Vec<f64> = (0..N).map(|h| ro[h] * sigma[a * N + h]).collect();
            if ro_a.iter().all(|&x| x <= 0.0) {
                continue; // unreached: worth 0 (the evaluator's rule)
            }
            let (b, v) = best_response(tree, *c, hero, &ro_a, ctx, collect_ev);
            for h in 0..N {
                br[h] += b[h];
                avg[h] += v[h];
            }
        }
        return (br, avg);
    }
    let mut kids: Vec<Option<(Vec<f64>, Vec<f64>)>> = Vec::with_capacity(n);
    for child in &children {
        kids.push(child.map(|c| best_response(tree, c, hero, ro, ctx, collect_ev)));
    }
    let VTree { nodes, live, order } = tree;
    let live = &live[hero];
    let (mut br, mut avg) = (vec![0.0; N], vec![0.0; N]);
    for h in 0..N {
        if !live[h] {
            continue;
        }
        let mut best = f64::NEG_INFINITY;
        for (a, k) in kids.iter().enumerate() {
            if let Some((b, v)) = k {
                avg[h] += sigma[a * N + h] * v[h];
                best = best.max(b[h]);
            }
        }
        br[h] = if best.is_finite() { best } else { 0.0 };
    }
    if collect_ev {
        let ev = node_ev(&nodes[idx].state, order.as_ref(), hero, live, ro, &avg, ctx);
        if let VKind::Decision(d) = &mut nodes[idx].kind {
            d.ev = Some(ev);
        }
    }
    (br, avg)
}

/// (TOOL-035) EV (bb) and equity per hero hand at a hero decision node: EV =
/// expected share of the final pot minus what the hand still puts in from here
/// (the formula of `dcfr::EvalCtx::record_ev`); NaN where no opponent hand
/// reaches the node.
fn node_ev(
    state: &PublicState,
    order: Option<&ShowdownOrder>,
    hero: usize,
    live: &[bool],
    ro: &[f64],
    avg: &[f64],
    ctx: &Ctx,
) -> Vec<(f64, f64)> {
    let table = combo_table();
    let mut out = vec![(f64::NAN, f64::NAN); N];
    let (total, by_card) = reach_totals(ro);
    if total <= 0.0 {
        return out;
    }
    let commit = state.total_commit[hero] as f64;
    let win: Option<Vec<f64>> = match (state.board_len, order) {
        (5, Some(o)) => Some(win_mass(o, ro, live)),
        (4, _) => {
            let free = (52 - 4 - 4) as f64;
            let mut acc = vec![0.0; N];
            for (x, o) in ctx.river_orders.iter().enumerate() {
                let Some(o) = o else { continue };
                let x = x as u8;
                let m = win_mass(o, &without_card(ro, x), live);
                for h in 0..N {
                    if table[h].0 != x && table[h].1 != x {
                        acc[h] += m[h] / free;
                    }
                }
            }
            Some(acc)
        }
        _ => None,
    };
    for h in 0..N {
        if !live[h] {
            continue;
        }
        let (a, b) = table[h];
        let compat = total - by_card[a as usize] - by_card[b as usize] + ro[h];
        if compat <= 1e-300 {
            continue;
        }
        let ev = (avg[h] / compat + commit) / ctx.bb;
        let eq = win
            .as_ref()
            .map(|m| (m[h] / compat).clamp(0.0, 1.0))
            .unwrap_or(f64::NAN);
        out[h] = (ev, eq);
    }
    out
}

/// NashConv/2 in bb of the average strategies — exact: every hand pair and
/// every runout (`dcfr::RiverSolver::evaluate`'s definition, with hero hands
/// weighted by their true marginal). Fills per-hand EV when asked.
fn exploitability(root: &mut VTree, ctx: &Ctx, collect_ev: bool) -> f64 {
    let table = combo_table();
    let (tot1, by_card1) = reach_totals(&ctx.w[1]);
    let mut z = 0.0;
    for h in 0..N {
        if ctx.w[0][h] > 0.0 {
            let (a, b) = table[h];
            z += ctx.w[0][h] * (tot1 - by_card1[a as usize] - by_card1[b as usize] + ctx.w[1][h]);
        }
    }
    if z <= 0.0 {
        return 0.0;
    }
    let mut nashconv = 0.0;
    for hero in 0..2 {
        let (br, avg) = best_response(root, 0, hero, &ctx.w[1 - hero], ctx, collect_ev);
        for h in 0..N {
            nashconv += ctx.w[hero][h] * (br[h] - avg[h]);
        }
    }
    (nashconv / z).max(0.0) / 2.0 / ctx.bb
}

/// Every decision node of the tree (all runouts) with its tree's live masks.
fn decision_nodes<'a>(
    tree: &'a VTree,
    out: &mut Vec<(&'a VNode, &'a Decision, &'a [Vec<bool>; 2])>,
) {
    for node in &tree.nodes {
        match &node.kind {
            VKind::Chance(subs) => {
                for (_, sub) in subs {
                    decision_nodes(sub, out);
                }
            }
            VKind::Decision(d) => out.push((node, d, &tree.live)),
            VKind::Terminal => {}
        }
    }
}

/// The rows of one decision node: every hand that reached it.
fn node_rows(
    node: &VNode,
    d: &Decision,
    live: &[Vec<bool>; 2],
    iso: bool,
    iterations: u32,
) -> Vec<InfosetStrategy> {
    let n = d.actions.len();
    let board = &node.state.board[..node.state.board_len as usize];
    let hist = history_key(node.ahash, board, node.state.street);
    let live = &live[d.actor];
    // The public part of the dump is the same for every hand of the node.
    let dump = InfosetDump::from_state(&node.state, &node.history, PRIV_COMBO, 0, None, None);
    let mut rows = Vec::new();
    for h in 0..N {
        if !live[h] {
            continue;
        }
        let mass: f64 = (0..n).map(|a| d.sum[a * N + h]).sum();
        if mass <= 0.0 {
            continue; // never reached with this hand: nothing to report
        }
        let pv = if iso {
            super::card_abs::iso_combo_id(h, board)
        } else {
            h as u32
        };
        let mut is = Infoset::new(d.actions.clone());
        for a in 0..n {
            is.regret[a] = d.regret[a * N + h];
            is.strategy_sum[a] = d.sum[a * N + h];
        }
        is.visits = iterations;
        let mut hand_dump = dump.clone();
        hand_dump.private_id = pv;
        hand_dump.raw_combo = Some(h as u32);
        hand_dump.iso_id = if iso { Some(pv) } else { None };
        is.dump = Some(hand_dump);
        let mut row = InfosetStrategy::from_node(format!("p{}_h{hist}_c{pv}", d.actor), &is);
        if let Some(ev) = &d.ev {
            let (e, q) = ev[h];
            if e.is_finite() {
                row.ev_bb = Some(e);
                row.equity = if q.is_finite() { Some(q) } else { None };
            }
        }
        rows.push(row);
    }
    rows
}

/// Every reached (node, hand) as a strategy row — the sampled solver's ids and
/// dump fields (the suit relabel is a bijection on a fixed board, so iso ids are
/// one per hand here too), sorted by id like the sampled solver's.
fn strategy_of(
    tree: &VTree,
    iso: bool,
    iterations: u32,
    root_id: &str,
    parallel: bool,
) -> Strategy {
    let mut jobs = Vec::new();
    decision_nodes(tree, &mut jobs);
    let build = |&(node, d, live): &(&VNode, &Decision, &[Vec<bool>; 2])| {
        node_rows(node, d, live, iso, iterations)
    };
    let parts: Vec<Vec<InfosetStrategy>> = if parallel {
        jobs.par_iter().map(build).collect()
    } else {
        jobs.iter().map(build).collect()
    };
    let mut rows = Vec::with_capacity(parts.iter().map(Vec::len).sum());
    for p in parts {
        rows.extend(p);
    }
    // Ids are unique, so an unstable sort is deterministic.
    if parallel {
        rows.par_sort_unstable_by(|a, b| a.infoset_id.cmp(&b.infoset_id));
    } else {
        rows.sort_unstable_by(|a, b| a.infoset_id.cmp(&b.infoset_id));
    }
    Strategy::new(root_id.to_string(), rows)
}

/// (decision nodes, live (node, hand) pairs) of a built tree.
fn count_rows(tree: &VTree) -> (u64, u64) {
    let (mut nodes, mut rows) = (0u64, 0u64);
    for node in &tree.nodes {
        match &node.kind {
            VKind::Chance(subs) => {
                for (_, sub) in subs {
                    let (a, b) = count_rows(sub);
                    nodes += a;
                    rows += b;
                }
            }
            VKind::Decision(d) => {
                nodes += 1;
                rows += tree.live[d.actor].iter().filter(|&&l| l).count() as u64;
            }
            VKind::Terminal => {}
        }
    }
    (nodes, rows)
}

/// Why `dcfr_vector` cannot solve `root` (street, seats, abstraction, or the
/// whole tree + report over the memory budget), checked BEFORE anything is
/// allocated; otherwise the memory estimate. Shared with `cfr_estimate_memory`,
/// so the app's Validate says exactly what a solve would.
pub fn check_root(root: &RootSpec, config: &SolveConfig) -> Result<MemoryEstimate, CfrError> {
    if root.num_seats != 2 || !matches!(root.street, StreetRoot::River | StreetRoot::Turn) {
        return Err(CfrError::InvalidConfig(
            "dcfr_vector solves heads-up river and turn roots; use \"dcfr\" for flops \
             (bucketed) and \"mccfr_es\" for preflop or multiway roots"
                .into(),
        ));
    }
    if !matches!(config.card_abstraction.as_str(), "" | "none" | "exact") {
        return Err(CfrError::InvalidConfig(
            "dcfr_vector works on exact hands (card_abstraction \"none\")".into(),
        ));
    }
    let est = estimate_solve_memory(root, config);
    let budget = config.ram_budget_bytes();
    if est.tree_truncated || est.est_bytes > budget {
        return Err(CfrError::InvalidConfig(format!(
            "dcfr_vector: the full tree and its report need ~{:.1} GB ({} decision nodes per \
             runout, every runout, 1326 hands each) — over the {:.1} GB memory budget; use \
             fewer raise sizes, shallower stacks, or the sampled \"dcfr\" algorithm",
            est.est_bytes as f64 / (1u64 << 30) as f64,
            est.public_nodes,
            budget as f64 / (1u64 << 30) as f64
        )));
    }
    Ok(est)
}

/// Solve a HU river or turn root with vectorized DCFR.
pub fn solve_vector_dcfr(root: &RootSpec, config: &SolveConfig) -> Result<SolveReport, CfrError> {
    root.validate()?;
    config.validate()?;
    check_root(root, config)?;
    if config.thread_num > 1 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.thread_num as usize)
            .build()
            .map_err(|e| CfrError::InvalidConfig(format!("thread pool: {e}")))?;
        pool.install(|| solve_inner(root, config, true))
    } else {
        solve_inner(root, config, false)
    }
}

fn solve_inner(
    root: &RootSpec,
    config: &SolveConfig,
    parallel: bool,
) -> Result<SolveReport, CfrError> {
    let start = Instant::now();
    let pot0 = root.pot_chips()?;
    let stack0 = root.effective_stack_chips()?;
    let root_state = PublicState::hu_postflop_root(pot0, stack0, &root.board, root.bb_chips)?;
    let named = |which: &str, e: CfrError| match e {
        CfrError::InvalidRoot(msg) => CfrError::InvalidRoot(format!("{which}: {msg}")),
        other => other,
    };
    let r0 = Range::parse(&root.range_oop, &root.board).map_err(|e| named("range_oop", e))?;
    let r1 = Range::parse(&root.range_ip, &root.board).map_err(|e| named("range_ip", e))?;
    let table = combo_table();
    let mut root_mask = 0u64;
    for &c in &root.board {
        root_mask |= 1u64 << c;
    }
    let clean = |r: &Range| -> Vec<f64> {
        (0..N)
            .map(|h| {
                let (a, b) = table[h];
                if root_mask & ((1u64 << a) | (1u64 << b)) == 0 {
                    r.weights[h].max(0.0)
                } else {
                    0.0
                }
            })
            .collect()
    };
    let w = [clean(&r0), clean(&r1)];
    let root_live = [
        w[0].iter().map(|&x| x > 0.0).collect::<Vec<bool>>(),
        w[1].iter().map(|&x| x > 0.0).collect::<Vec<bool>>(),
    ];
    let river_orders = if root.board.len() == 4 {
        (0..52u8)
            .map(|x| {
                (!root.board.contains(&x)).then(|| {
                    let b = [
                        root.board[0],
                        root.board[1],
                        root.board[2],
                        root.board[3],
                        x,
                    ];
                    ShowdownOrder::new(combo_ranks_on_board(&b))
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    let ctx = Ctx {
        w,
        parallel,
        bb: root.bb_chips as f64,
        river_orders,
    };

    let mut builder = Builder {
        raise_pm: &root.raise_sizes_pm,
        allin: root.allin_atom,
        root_live: root_live.clone(),
        decision_nodes: 0,
        bytes: 0,
        budget: config.ram_budget_bytes(),
    };
    let mut tree = VTree::new(&root_state, &root_live);
    builder.build(&mut tree, root_state, Vec::new(), EMPTY_ACTIONS)?;
    let (_, n_rows) = count_rows(&tree);
    let build_secs = start.elapsed().as_secs_f64();

    let params = config.discounting().params();
    let iter_limit = config.iter_limit();
    let polls_on = config.target_exploitability_bb > 0.0;
    let timed_checks = config.expl_check_secs > 0.0;
    let (mut last_expl, mut last_check_end, mut last_check_cost) =
        (None::<f64>, start, Duration::ZERO);
    let mut pacer = SnapshotPacer::default();
    let mut stop_reason: Option<&'static str> = None;
    let mut iterations_run = 0u32;
    let iso = config.use_isomorphism;
    let (rs0, rs1) = (ctx.w[0].clone(), ctx.w[1].clone());
    // An iteration can take a second on a big turn tree: the live counters go
    // out by wall clock too, and right after every exploitability check.
    let mut last_counters = start;
    // Pause is looked for every iteration (a file stat) — not only every
    // `poll_every` iterations, which can be many seconds on a turn tree.
    let mut stop_cfg = config.clone();
    stop_cfg.poll_every = 1;
    for it in 1..=iter_limit {
        if let Some(why) = stop_cfg.should_stop(start, it) {
            stop_reason = Some(why);
            break;
        }
        cfr(&mut tree, 0, 0, &rs0, &rs1, &ctx);
        cfr(&mut tree, 0, 1, &rs1, &rs0, &ctx);
        if let Some((a, b, g)) = params {
            let (pos, neg, strat) = Infoset::discount_scales(it, a, b, g);
            discount(&mut tree, pos, neg, strat, ctx.parallel);
        }
        iterations_run = it;
        // Exact exploitability (about one iteration's cost): every
        // TARGET_CHECK_EVERY iterations for a target; by wall clock (never over
        // ~20% of the run) for a live view (TOOL-030).
        let target_due = polls_on && it % TARGET_CHECK_EVERY == 0;
        let timed_due = timed_checks
            && last_check_end.elapsed()
                >= Duration::from_secs_f64(config.expl_check_secs).max(last_check_cost * 4);
        let checked_now = target_due || timed_due;
        if checked_now {
            let t0 = Instant::now();
            let e = exploitability(&mut tree, &ctx, false);
            last_expl = Some(e);
            last_check_cost = t0.elapsed();
            last_check_end = Instant::now();
            if polls_on && e <= config.target_exploitability_bb {
                stop_reason = Some("target_exploitability");
            }
        }
        let poll = config.poll_every.max(1);
        let counters_due = it % poll == 0
            || it == 1
            || stop_reason.is_some()
            || checked_now
            || last_counters.elapsed() >= Duration::from_millis(500);
        if !config.progress_file.is_empty() && counters_due {
            let kind = last_expl.map(|_| "exact_infoset");
            config.write_progress_counters(
                it,
                last_expl,
                kind,
                n_rows as usize,
                &root.root_id,
                false,
            );
            last_counters = Instant::now();
            if pacer.due(config) {
                let t0 = Instant::now();
                let strat = strategy_of(&tree, iso, it, &root.root_id, ctx.parallel);
                config.write_progress(
                    it,
                    last_expl,
                    kind,
                    strat.infosets.len(),
                    Some(&strat),
                    &root.root_id,
                    false,
                );
                pacer.record(t0);
            }
        }
        if stop_reason.is_some() {
            break;
        }
    }

    // The final best response is exact and costs about one iteration, so it
    // runs on every exit path (stop file and time budget included).
    let final_started = Instant::now();
    let expl = exploitability(&mut tree, &ctx, true);
    let final_secs = final_started.elapsed().as_secs_f64();
    let strategy = strategy_of(&tree, iso, iterations_run, &root.root_id, ctx.parallel);
    let with_ev = strategy
        .infosets
        .iter()
        .filter(|i| i.ev_bb.is_some())
        .count();
    let street_name = format!("{:?}", root.street);
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
        format!("DCFR-vector {street_name} HU (full ranges every iteration, every runout exact)"),
        format!("algorithm={} discount={}", config.algorithm, config.discounting().label()),
        format!("infosets={}", strategy.infosets.len()),
        format!("pot_chips={pot0} stack_chips={stack0}"),
        if uniform0 && uniform1 {
            "ranges=uniform_fallback (no range given)".into()
        } else {
            format!("ranges=parsed oop={} ip={}", side(uniform0, &r0), side(uniform1, &r1))
        },
        "card_abs=exact_combo".into(),
        if iso {
            "isomorphism=on iso=noop_on_fixed_board (suit relabel only; no infoset reduction)".into()
        } else {
            "isomorphism=off".into()
        },
        format!(
            "tree decision_nodes={} mem_mb={:.1} build_secs={build_secs:.1}",
            builder.decision_nodes,
            builder.bytes as f64 / (1024.0 * 1024.0)
        ),
        format!(
            "thread_num={} parallel={}",
            config.thread_num.max(1),
            match (ctx.parallel, root.board.len()) {
                (false, _) => "off",
                (true, 4) => "river subtrees + export",
                (true, _) => "export (one board)",
            }
        ),
        format!("wall_secs={:.1}", start.elapsed().as_secs_f64()),
        format!(
            "expl_kind=exact_infoset (vectorized best response, all combos{}) final_expl_secs={final_secs:.1}",
            if root.board.len() == 4 { ", expectation over all rivers" } else { "" }
        ),
    ];
    if with_ev > 0 {
        notes.push(format!(
            "ev=per-hand EV + equity on {with_ev} of {} infosets (average strategies)",
            strategy.infosets.len()
        ));
    }
    if let Some(why) = stop_reason {
        notes.push(format!("early_stop={why}"));
    }
    Ok(SolveReport {
        status: "ok".into(),
        root: root.clone(),
        config: config.clone(),
        strategy,
        iterations_run: iterations_run.max(1),
        exploitability_bb: Some(expl),
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(iters: u32) -> SolveConfig {
        let mut c = SolveConfig::default();
        c.algorithm = "dcfr_vector".into();
        c.max_iterations = iters;
        c.target_exploitability_bb = 0.0;
        c.use_isomorphism = false;
        c
    }

    /// A small river root with blockers between the ranges.
    fn river_small() -> RootSpec {
        let mut root = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            10.0,
            vec![0, 5, 10, 15, 20],
            vec![500],
        );
        root.range_oop = "AA,KK,QQ,JTs,T9s,65s".into();
        root.range_ip = "AK,KQs,99,88,A5s".into();
        root
    }

    fn turn_small() -> RootSpec {
        let mut root =
            RootSpec::postflop_hu(StreetRoot::Turn, 10.0, 10.0, vec![0, 5, 10, 15], vec![]);
        root.range_oop = "AA,KK,JTs".into();
        root.range_ip = "TT,99,AKs,87s".into();
        root
    }

    /// The vectorized best response equals `dcfr.rs`'s (brute-force-checked)
    /// evaluator on the same average strategy: the report's rows are loaded
    /// into a sampled-solver table and evaluated there.
    #[test]
    fn vector_exploitability_matches_the_reference_evaluator() {
        for (root, iters) in [(river_small(), 40u32), (turn_small(), 15)] {
            let rep = solve_vector_dcfr(&root, &cfg(iters)).unwrap();
            let reference = crate::cfr::dcfr::evaluate_rows_for_test(&root, &rep.strategy);
            let got = rep.exploitability_bb.unwrap();
            assert!(got > 0.0);
            assert!(
                (got - reference).abs() < 1e-9 * (1.0 + reference),
                "{:?}: vector {got} vs reference {reference}",
                root.street
            );
        }
    }

    /// Full-width DCFR converges in few iterations where sampling needs thousands.
    #[test]
    fn converges_on_river_and_turn_roots() {
        let rep = solve_vector_dcfr(&river_small(), &cfg(300)).unwrap();
        assert!(
            rep.exploitability_bb.unwrap() < 0.02,
            "small river {:?}",
            rep.exploitability_bb
        );
        let mut full = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            20.0,
            vec![3, 17, 22, 40, 51],
            vec![500, 1000],
        );
        full.root_id = "full".into();
        let rep = solve_vector_dcfr(&full, &cfg(200)).unwrap();
        assert!(
            rep.exploitability_bb.unwrap() < 0.05,
            "full river {:?}",
            rep.exploitability_bb
        );
        assert!(rep
            .notes
            .iter()
            .any(|n| n.starts_with("expl_kind=exact_infoset")));
        let rep = solve_vector_dcfr(&turn_small(), &cfg(150)).unwrap();
        assert!(
            rep.exploitability_bb.unwrap() < 0.05,
            "turn {:?}",
            rep.exploitability_bb
        );
    }

    /// Rows carry the sampled solver's ids, dumps and EV / equity.
    #[test]
    fn rows_have_ids_dumps_and_ev() {
        let root = river_small();
        let rep = solve_vector_dcfr(&root, &cfg(50)).unwrap();
        let rows = &rep.strategy.infosets;
        assert!(!rows.is_empty());
        let mut ids: Vec<&str> = rows.iter().map(|r| r.infoset_id.as_str()).collect();
        ids.dedup();
        assert_eq!(ids.len(), rows.len(), "ids are unique");
        for r in rows {
            assert_eq!(r.visits, Some(50));
            assert_eq!(r.private_kind.as_deref(), Some("combo"));
            assert_eq!(r.raw_combo, r.private_id);
            assert!((r.probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            let eq = r.equity.expect("river rows have equity");
            assert!((0.0..=1.0).contains(&eq));
            assert!(r.ev_bb.expect("ev").is_finite());
        }
        // The root has a row for every OOP hand in the range.
        let r0 = Range::parse(&root.range_oop, &root.board).unwrap();
        let root_rows = rows
            .iter()
            .filter(|r| r.path.as_ref().is_some_and(|p| p.is_empty()))
            .count();
        assert_eq!(root_rows, r0.live_combos());
    }

    /// The thread count changes the speed, never the result.
    #[test]
    fn parallel_turn_solve_is_identical() {
        let mut one = cfg(12);
        one.thread_num = 1;
        let mut four = cfg(12);
        four.thread_num = 4;
        let a = solve_vector_dcfr(&turn_small(), &one).unwrap();
        let b = solve_vector_dcfr(&turn_small(), &four).unwrap();
        assert_eq!(a.exploitability_bb, b.exploitability_bb);
        assert_eq!(a.strategy.infosets.len(), b.strategy.infosets.len());
        for (x, y) in a.strategy.infosets.iter().zip(&b.strategy.infosets) {
            assert_eq!(x.infoset_id, y.infoset_id);
            assert_eq!(x.probs, y.probs);
            assert_eq!(x.ev_bb, y.ev_bb);
        }
        assert!(b
            .notes
            .iter()
            .any(|n| n.contains("parallel=river subtrees")));
    }

    #[test]
    fn solve_dispatches_and_refuses_what_it_cannot_do() {
        let rep = crate::cfr::solve(&river_small(), &cfg(5)).unwrap();
        assert!(
            rep.notes[0].starts_with("DCFR-vector River"),
            "{:?}",
            rep.notes[0]
        );
        let flop = RootSpec::postflop_hu(StreetRoot::Flop, 10.0, 10.0, vec![0, 5, 10], vec![]);
        assert!(matches!(
            crate::cfr::solve(&flop, &cfg(5)),
            Err(CfrError::InvalidConfig(_))
        ));
        let mut mw = river_small();
        mw.num_seats = 3;
        assert!(matches!(
            solve_vector_dcfr(&mw, &cfg(5)),
            Err(CfrError::InvalidConfig(_))
        ));
        let mut tight = cfg(5);
        tight.ram_budget_mb = 64;
        let full = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            50.0,
            vec![12, 28, 38, 41, 45],
            vec![330, 500, 750, 1000, 1500],
        );
        let err = solve_vector_dcfr(&full, &tight).unwrap_err();
        assert!(format!("{err}").contains("budget"), "{err}");
        // Ranges shrink the report, so the same tree fits once they are narrow.
        let mut narrow = full.clone();
        narrow.range_oop = "AA,KK,QQ".into();
        narrow.range_ip = "AK,JJ".into();
        assert!(solve_vector_dcfr(&narrow, &tight).is_ok());
    }

    /// The estimate the app shows is what the solver allocates.
    #[test]
    fn memory_estimate_matches_the_built_tree() {
        for root in [river_small(), turn_small()] {
            let rep = solve_vector_dcfr(&root, &cfg(1)).unwrap();
            let note = rep
                .notes
                .iter()
                .find(|n| n.starts_with("tree decision_nodes="))
                .unwrap();
            let words: Vec<&str> = note.split(' ').collect();
            let nodes: u64 = words[1]["decision_nodes=".len()..].parse().unwrap();
            let mem_mb: f64 = words[2]["mem_mb=".len()..].parse().unwrap();
            let state = PublicState::hu_postflop_root(
                root.pot_chips().unwrap(),
                root.effective_stack_chips().unwrap(),
                &root.board,
                root.bb_chips,
            )
            .unwrap();
            let count = crate::cfr::memory::count_public_tree(
                &state,
                &root.raise_sizes_pm,
                root.allin_atom,
                true,
            );
            let (est_nodes, full_rows, tree_bytes) =
                crate::cfr::memory::vector_tree_size(&root, &count, None);
            assert_eq!(nodes, est_nodes, "{:?}", root.street);
            assert!(
                (tree_bytes as f64 / (1024.0 * 1024.0) - mem_mb).abs() <= 0.051,
                "{:?}",
                root.street
            );
            // With the ranges, the row count is exact: after one iteration every
            // hand in the range has reached every node (uniform strategies).
            let est = estimate_solve_memory(&root, &cfg(1));
            assert_eq!(
                est.est_infosets,
                rep.strategy.infosets.len() as u64,
                "{:?}",
                root.street
            );
            assert!(est.est_infosets < full_rows);
            assert!(est.est_bytes > tree_bytes);
        }
    }

    /// (TOOL-008) Benchmark: exact exploitability vs wall clock, sampled
    /// `dcfr` against `dcfr_vector`, on full-range roots (cargo test --profile
    /// fasttest --lib bench_vector_vs_sampled -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn bench_vector_vs_sampled() {
        let threads: u32 = std::env::var("BENCH_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let river = RootSpec::postflop_hu(
            StreetRoot::River,
            10.0,
            50.0,
            vec![12, 28, 38, 41, 45],
            vec![330, 500, 750, 1000, 1500],
        );
        let turn = RootSpec::postflop_hu(
            StreetRoot::Turn,
            10.0,
            20.0,
            vec![12, 28, 38, 41],
            vec![500, 1000],
        );
        for (name, root, sampled_iters, vector_iters) in [
            (
                "river 10bb/50bb 5 sizes",
                &river,
                &[2_000u32, 10_000, 50_000, 200_000][..],
                &[10u32, 25, 50, 100, 200, 400][..],
            ),
            (
                "turn 10bb/20bb 2 sizes",
                &turn,
                &[2_000u32, 10_000, 50_000][..],
                &[10u32, 25, 50, 100, 200][..],
            ),
        ] {
            let only = std::env::var("BENCH_ONLY").unwrap_or_default();
            for (algo, ladder) in [("dcfr", sampled_iters), ("dcfr_vector", vector_iters)] {
                if !only.is_empty() && only != algo {
                    continue;
                }
                for &iters in ladder {
                    let mut c = SolveConfig::default();
                    c.algorithm = algo.into();
                    c.max_iterations = iters;
                    c.target_exploitability_bb = 0.0;
                    c.seed = 7;
                    c.thread_num = if algo == "dcfr" { 1 } else { threads };
                    let t0 = Instant::now();
                    let rep = crate::cfr::solve(root, &c).unwrap();
                    let secs = t0.elapsed().as_secs_f64();
                    let kind = rep
                        .notes
                        .iter()
                        .find(|n| n.starts_with("expl_kind="))
                        .map(|n| n.split(' ').next().unwrap())
                        .unwrap_or("?");
                    let timing = rep
                        .notes
                        .iter()
                        .filter_map(|n| {
                            n.split(' ')
                                .find(|w| w.ends_with("_secs=") || w.contains("_secs="))
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    println!(
                        "{name:28} {algo:12} iters={iters:>7} wall={secs:>7.2}s expl={:.4} bb ({kind}) rows={} {timing}",
                        rep.exploitability_bb.unwrap_or(f64::NAN),
                        rep.strategy.infosets.len()
                    );
                }
            }
        }
    }
}
