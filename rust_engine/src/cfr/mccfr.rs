//! External-sampling MCCFR for preflop blueprint and multiway.

use std::collections::HashMap;

use crate::cards::Card;
use crate::hand_eval::evaluate_nlh;

use super::actions::{apply_abstract, legal_actions, AbstractAction};
use super::infoset::{
    Infoset, InfosetDump, InfosetKey, PRIV_CLASS, PRIV_COMBO,
};
use super::preflop::{deal_holes_hu, DealRng, PreflopHandClass, NUM_PREFLOP_CLASSES};
use super::public_state::{PublicState, MAX_SEATS};
use super::types::{InfosetStrategy, RootSpec, SolveConfig, SolveReport, Strategy, StreetRoot};
use super::CfrError;

/// Hash public history from action sequence **and** public board/street.
/// Board + street must be in the key so multiway flop→turn runouts (and
/// distinct boards after X/X) get distinct infosets — matches HU DCFR.
fn history_hash(
    actions: &[AbstractAction],
    board: &[u8],
    board_len: u8,
    street: u8,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for a in actions {
        a.label().hash(&mut h);
    }
    street.hash(&mut h);
    board_len.hash(&mut h);
    for i in 0..board_len as usize {
        if i < board.len() {
            board[i].hash(&mut h);
        }
    }
    h.finish()
}

/// Preflop / pure-action hash (board empty, street 0). Used when public
/// cards are not yet dealt.
#[inline]
fn history_hash_actions(actions: &[AbstractAction]) -> u64 {
    history_hash(actions, &[], 0, 0)
}

/// Hash from a live public state (postflop multiway / multi-street).
#[inline]
fn history_hash_state(actions: &[AbstractAction], state: &PublicState) -> u64 {
    history_hash(
        actions,
        &state.board,
        state.board_len,
        state.street,
    )
}

/// Human path label for push/fold trees: empty → "open", else "F" / "AI" joined by commas.
fn history_path_label(actions: &[AbstractAction]) -> String {
    if actions.is_empty() {
        return "open".into();
    }
    actions
        .iter()
        .map(|a| match a {
            AbstractAction::Fold => "F".to_string(),
            AbstractAction::AllIn => "AI".to_string(),
            AbstractAction::CheckCall => "XC".to_string(),
            AbstractAction::RaisePm(pm) => format!("R{pm}"),
        })
        .collect::<Vec<_>>()
        .join(",")
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
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

impl DealRng for Lcg {
    fn gen_range(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() as usize) % n
    }
}

/// External-sampling step at a **non-traverser** node: accumulate the ACTING
/// player's average strategy with its current strategy, then sample one action.
///
/// (review 2026-09-20 D6) The average strategy must be weighted by the acting
/// player's OWN reach. Under external sampling a non-traverser node is reached
/// with probability `π_actor · π_chance` (the traverser's actions are all
/// enumerated, so they contribute a t-independent multiplicity), so
/// `strategy_sum += σ` here is the own-reach-weighted update (Lanctot et al.
/// 2009, Alg. 1). The accumulation used to sit in the `actor == traverser`
/// branch, where the node is reached with the OPPONENTS' sampled reach — the
/// sum was opponent-reach-weighted, which is not the CFR average strategy and
/// stalls/diverges in any tree where a player acts twice.
///
/// With 3+ seats the node is also reached through the OTHER sampled seats'
/// actions, so the weight is own reach × their reach (the same convention as
/// OpenSpiel's ES-MCCFR); multiway has no convergence guarantee either way.
///
/// RNG consumption is unchanged (exactly one `next_f64` per sampled node), so
/// for a given seed the sampled deals/actions and the regrets are identical to
/// the pre-fix solver; only the accumulated averages differ.
fn es_sample_opponent_action(node: &mut Infoset, strategy: &[f64], rng: &mut Lcg) -> usize {
    for (s, &p) in node.strategy_sum.iter_mut().zip(strategy.iter()) {
        *s += p;
    }
    let mut t = rng.next_f64();
    let mut idx = 0;
    for (i, &p) in strategy.iter().enumerate() {
        t -= p;
        idx = i;
        if t <= 0.0 {
            break;
        }
    }
    idx
}

/// Sample a full 5-card board not colliding with hole cards.
fn sample_board5(rng: &mut Lcg, blocked: &[u8]) -> [u8; 5] {
    let mut used = [false; 52];
    for &c in blocked {
        used[c as usize] = true;
    }
    let mut board = [0u8; 5];
    let mut i = 0;
    while i < 5 {
        let c = rng.gen_range(52) as u8;
        if !used[c as usize] {
            used[c as usize] = true;
            board[i] = c;
            i += 1;
        }
    }
    board
}

/// Validated HU preflop root (seat 0 = BB, seat 1 = BTN/SB, BTN first).
///
/// (review 2026-09-20 E1) `effective_stack_bb <= ante + bb` used to build a
/// state where both seats held 0 chips without being flagged all-in; the BTN's
/// 0-chip "call" never matched the blind, so it was re-selected as the next
/// actor forever → unbounded recursion → native stack overflow that killed
/// the host process (stack 0.5/1.0 bb with default blinds/ante). Such sub-blind
/// stacks have no real preflop decision tree in this abstraction (blinds are
/// accounted at full size), so they are refused with an ordinary error.
/// `PublicState::can_act` separately guarantees a chipless seat can never be
/// handed the action, so no other hand-built root can recurse either.
pub(crate) fn hu_preflop_root(root: &RootSpec) -> Result<PublicState, CfrError> {
    let (bb, sb, ante) = (root.bb_chips, root.sb_chips, root.ante_chips);
    let pot0 = root.pot_chips()?;
    let stack0 = root.effective_stack_chips()?;
    if sb > bb {
        return Err(CfrError::InvalidRoot(format!(
            "sb_chips {sb} > bb_chips {bb}"
        )));
    }
    let need = ante.saturating_add(bb);
    if stack0 <= need {
        return Err(CfrError::InvalidRoot(format!(
            "HU preflop needs effective_stack_bb > ante + big blind \
             ({stack0} chips <= {need}): a sub-blind stack is all-in from the posts \
             and leaves no decision tree to solve"
        )));
    }
    let posted = ante.saturating_mul(2).saturating_add(sb).saturating_add(bb);
    if pot0 < posted {
        return Err(CfrError::InvalidRoot(format!(
            "pot_bb is {pot0} chips but the posted antes+blinds are {posted}; \
             HU preflop pot_bb must be >= (2*ante + sb + bb) / bb"
        )));
    }
    let st = PublicState::preflop_root(&[stack0, stack0], bb, sb, ante, pot0 - posted)?;
    require_root_decision(&st, &root.raise_sizes_pm, root.allin_atom)?;
    Ok(st)
}

/// Validated multiway preflop root (seats `0..n-3` UTG.., `n-2` SB, `n-1` BB).
/// The pot is rebuilt from the actual posts; `root.pot_bb` is ignored.
pub(crate) fn mw_preflop_root(root: &RootSpec) -> Result<PublicState, CfrError> {
    let stacks = root.seat_stacks_chips()?;
    if root.sb_chips > root.bb_chips {
        return Err(CfrError::InvalidRoot(format!(
            "sb_chips {} > bb_chips {}",
            root.sb_chips, root.bb_chips
        )));
    }
    let st = PublicState::preflop_root(&stacks, root.bb_chips, root.sb_chips, root.ante_chips, 0)?;
    require_root_decision(&st, &root.raise_sizes_pm, root.allin_atom)?;
    Ok(st)
}

/// Run-time refuse-to-OOM (review 2026-09-20 F14): sampled solvers cannot
/// bound their table up front (it grows with what gets visited), so the loop
/// stops with `early_stop=memory_budget` once the live table passes the budget.
fn table_over_memory_budget(n_infosets: usize) -> bool {
    n_infosets as u64 * super::memory::BYTES_PER_INFOSET_BASE
        > super::memory::DEFAULT_RAM_BUDGET_BYTES
}

/// A root with nobody to act (everyone all-in from the posts) is not a game.
/// It used to "solve" to 0 infosets and exploitability 0.0 (review 2026-09-20 E1).
fn require_root_decision(
    st: &PublicState,
    raise_pm: &[u32],
    allin: bool,
) -> Result<(), CfrError> {
    if st.actor.is_none() || legal_actions(st, raise_pm, allin).is_empty() {
        return Err(CfrError::InvalidRoot(
            "degenerate root: no seat has a decision (stacks do not cover the antes/blinds)".into(),
        ));
    }
    Ok(())
}

/// Preflop HU MCCFR: 169 classes, external sampling, **real equity** via
/// sampled runouts at showdown (MC equity, not class_id proxy).
pub fn solve_preflop_mccfr(root: &RootSpec, config: &SolveConfig) -> Result<SolveReport, CfrError> {
    root.validate()?;
    config.validate()?;
    if root.street != StreetRoot::Preflop {
        return Err(CfrError::InvalidRoot(
            "mccfr preflop expects Preflop street".into(),
        ));
    }

    let bb = root.bb_chips;
    let raise_pm = root.raise_sizes_pm.clone();
    let allin = root.allin_atom;
    // Faithful HU preflop: blinds in street_commit, BTN/SB (seat 1) first.
    // Built ONCE and validated (review 2026-09-20 E1) — see `hu_preflop_root`.
    let st = hu_preflop_root(root)?;
    let mut infosets: HashMap<InfosetKey, Infoset> = HashMap::new();
    let mut rng = Lcg::new(config.seed);
    let start = std::time::Instant::now();
    let mut stop_reason: Option<&'static str> = None;
    let mut iterations_run = 0u32;
    // Only "dcfr" discounts the sampled regrets (every 10th iteration); every
    // other tag is plain ES-MCCFR. Reported truthfully in the notes.
    let discount_dcfr = config.discounting() == super::types::Discounting::Dcfr;

    // Concrete holes for equity (not just class)
    let iter_limit = config.iter_limit();
    let poll = config.poll_every.max(1);
    for it in 1..=iter_limit {
        if let Some(why) = config.should_stop(start, it) {
            stop_reason = Some(why);
            break;
        }
        let (h0, h1) = deal_holes_hu(&mut rng);
        let holes = [h0, h1];
        let classes = [
            PreflopHandClass::from_cards(h0.0, h0.1).id(),
            PreflopHandClass::from_cards(h1.0, h1.1).id(),
        ];

        let board = sample_board5(&mut rng, &[h0.0, h0.1, h1.0, h1.1]);

        for trav in 0..2 {
            mccfr_traverse(
                &st,
                &[],
                classes,
                holes,
                board,
                trav,
                &raise_pm,
                allin,
                &mut infosets,
                &mut rng,
            );
        }

        if discount_dcfr && it % 10 == 0 {
            for node in infosets.values_mut() {
                node.apply_dcfr_discount(it);
            }
        }
        iterations_run = it;
        if it % poll == 0 && table_over_memory_budget(infosets.len()) {
            stop_reason = Some("memory_budget");
            break;
        }
        if (it % poll == 0 || it == 1) && !config.progress_file.is_empty() {
            // 169-class strategy is small — dump full average for live viewer.
            let mut snap = Vec::new();
            for (key, node) in &infosets {
                snap.push(InfosetStrategy::from_node(
                    format!("pf_p{}_h{}_c{}", key.player, key.history, key.private),
                    node,
                ));
            }
            snap.sort_by(|a, b| a.infoset_id.cmp(&b.infoset_id));
            let strat = Strategy::new(root.root_id.clone(), snap);
            config.write_progress(
                it,
                None,
                None,
                infosets.len(),
                Some(&strat),
                &root.root_id,
                false,
            );
        }
    }
    if iterations_run == 0 {
        iterations_run = 1; // avoid empty export edge case
    }

    // (review 2026-09-20 D10) This is a PERFECT-INFORMATION deal-BR: the
    // "best responder" sees the opponent's hole cards and the sampled board,
    // so the number is an upward-biased proxy that does not shrink with more
    // iterations (2.71 → 3.11 bb from 20k → 1M iters in the review). It is
    // reported for continuity but labelled `expl_kind=mc_br_proxy`, like the
    // multiway paths — never a Nash certificate.
    let expl = {
        let mut total = 0.0;
        let samples = 32u32;
        for _ in 0..samples {
            let (h0, h1) = deal_holes_hu(&mut rng);
            let holes = [h0, h1];
            let classes = [
                PreflopHandClass::from_cards(h0.0, h0.1).id(),
                PreflopHandClass::from_cards(h1.0, h1.1).id(),
            ];
            let board = sample_board5(&mut rng, &[h0.0, h0.1, h1.0, h1.1]);
            for br_player in 0..2 {
                let v = pf_avg_value(
                    &infosets, &st, &[], classes, holes, board, br_player, &raise_pm, allin,
                );
                let brv = pf_br_value(
                    &infosets, &st, &[], classes, holes, board, br_player, &raise_pm, allin,
                );
                total += (brv - v).max(0.0);
            }
        }
        (total / samples as f64) / 2.0 / bb as f64
    };

    let mut out = Vec::new();
    for (key, node) in &infosets {
        out.push(InfosetStrategy::from_node(
            format!("pf_p{}_h{}_c{}", key.player, key.history, key.private),
            node,
        ));
    }
    out.sort_by(|a, b| a.infoset_id.cmp(&b.infoset_id));

    let mut notes = vec![
        "External-sampling MCCFR preflop 169-class".into(),
        "showdown=real NLH equity via sampled board".into(),
        "preflop_tree=SB/BB street_commit + BTN first".into(),
        format!("infosets={}", infosets.len()),
        format!("classes={NUM_PREFLOP_CLASSES}"),
        format!(
            "algorithm={} discount={}",
            config.algorithm,
            if discount_dcfr { "dcfr_every_10_iters" } else { "none" }
        ),
        "avg_strategy=own_reach (accumulated at sampled opponent nodes)".into(),
        format!("expl_kind=mc_br_proxy samples=32 mc_br_proxy_bb={expl}"),
        "exploitability_bb field is a perfect-information deal-BR proxy (BR sees villain's \
         hole cards + board): upward-biased, NOT a Nash certificate and not expected to \
         shrink with more iterations"
            .into(),
        format!("wall_secs={:.1}", start.elapsed().as_secs_f64()),
    ];
    if let Some(why) = stop_reason {
        notes.push(format!("early_stop={why}"));
    }

    Ok(SolveReport {
        status: "ok".into(),
        root: root.clone(),
        config: config.clone(),
        strategy: Strategy::new(root.root_id.clone(), out),
        iterations_run,
        exploitability_bb: Some(expl),
        notes,
    })
}

/// Multiway **preflop** MCCFR: 169-class private views, micro/coarse sizes,
/// real sampled-board equity at showdown. Monker-class multiway starts here.
pub fn solve_multiway_preflop_mccfr(
    root: &RootSpec,
    config: &SolveConfig,
) -> Result<SolveReport, CfrError> {
    if root.num_seats < 3 {
        return Err(CfrError::InvalidRoot(
            "multiway preflop needs num_seats >= 3".into(),
        ));
    }
    if root.street != StreetRoot::Preflop {
        return Err(CfrError::InvalidRoot(
            "solve_multiway_preflop_mccfr expects Preflop".into(),
        ));
    }
    root.validate_for_solve()?;
    config.validate()?;

    let n = root.num_seats as usize;
    // Public preflop root: seats 0..n-3 UTG+, n-2 = SB, n-1 = BB, UTG first.
    // Pot = the actual posts (n*ante + sb + bb for covering stacks); pot_bb is
    // ignored. Built once + validated (review 2026-09-20 E1).
    let st = mw_preflop_root(root)?;
    let pot0 = st.pot;
    // Empty raise_sizes + allin_atom = pure push/fold (Monker AoF tree).
    // Validation already rejects empty menu without allin_atom — no silent default.
    let raise_pm = root.raise_sizes_pm.clone();
    let allin = root.allin_atom;
    let mut infosets: HashMap<InfosetKey, Infoset> = HashMap::new();
    // history_hash → human path label ("open", "F", "AI", "F,F", …)
    let mut path_labels: HashMap<u64, String> = HashMap::new();
    let mut rng = Lcg::new(config.seed);
    let start = std::time::Instant::now();
    let mut stop_reason: Option<&'static str> = None;
    let mut iterations_run = 0u32;
    let iter_limit = config.iter_limit();
    let poll = config.poll_every.max(1);

    for it in 1..=iter_limit {
        if let Some(why) = config.should_stop(start, it) {
            stop_reason = Some(why);
            break;
        }
        // Deal holes
        let mut used = [false; 52];
        let mut holes: Vec<(u8, u8)> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut draw = || loop {
                let c = rng.gen_range(52) as u8;
                if !used[c as usize] {
                    used[c as usize] = true;
                    return c;
                }
            };
            let a = draw();
            let b = draw();
            holes.push(if a < b { (a, b) } else { (b, a) });
        }
        let classes: Vec<u32> = holes
            .iter()
            .map(|&(a, b)| PreflopHandClass::from_cards(a, b).id())
            .collect();
        // Sample board for showdown equity
        let board = {
            let blocked: Vec<u8> = holes.iter().flat_map(|&(a, b)| [a, b]).collect();
            sample_board5(&mut rng, &blocked)
        };

        for trav in 0..n {
            mw_preflop_traverse(
                &st,
                &[],
                &classes,
                &holes,
                board,
                trav,
                &raise_pm,
                allin,
                &mut infosets,
                &mut path_labels,
                &mut rng,
            );
        }
        iterations_run = it;
        if it % poll == 0 && table_over_memory_budget(infosets.len()) {
            stop_reason = Some("memory_budget");
            break;
        }
        if (it % poll == 0 || it == 1) && !config.progress_file.is_empty() {
            // Multiway preflop: dump average strategy for live viewer.
            let mut snap = Vec::new();
            for (key, node) in &infosets {
                let path = path_labels
                    .get(&key.history)
                    .cloned()
                    .unwrap_or_else(|| format!("h{}", key.history));
                snap.push(InfosetStrategy::from_node(
                    format!("mwpf_p{}_path{}_c{}", key.player, path, key.private),
                    node,
                ));
            }
            snap.sort_by(|a, b| a.infoset_id.cmp(&b.infoset_id));
            let strat = Strategy::new(root.root_id.clone(), snap);
            config.write_progress(
                it,
                None,
                None,
                infosets.len(),
                Some(&strat),
                &root.root_id,
                false,
            );
        }
    }
    if iterations_run == 0 {
        iterations_run = 1;
    }

    let mut out = Vec::new();
    for (key, node) in &infosets {
        let path = path_labels
            .get(&key.history)
            .cloned()
            .unwrap_or_else(|| format!("h{}", key.history));
        out.push(InfosetStrategy::from_node(
            format!("mwpf_p{}_path{}_c{}", key.player, path, key.private),
            node,
        ));
    }
    out.sort_by(|a, b| a.infoset_id.cmp(&b.infoset_id));

    // Push/fold trees are small; raised-size multiway BR is exponential — cap samples.
    let expl_samples = if raise_pm.is_empty() && allin { 24 } else { 4 };
    let expl = mw_preflop_mc_expl(&infosets, &raise_pm, allin, &st, &mut rng, expl_samples);

    let mut notes_extra = vec![format!("wall_secs={:.1}", start.elapsed().as_secs_f64())];
    if let Some(why) = stop_reason {
        notes_extra.push(format!("early_stop={why}"));
    }

    Ok(SolveReport {
        status: "ok".into(),
        root: root.clone(),
        config: config.clone(),
        strategy: Strategy::new(root.root_id.clone(), out),
        iterations_run,
        exploitability_bb: Some(expl),
        notes: {
            let mut nts = vec![
                format!("MCCFR multiway preflop n={n} 169-class"),
                "showdown=real NLH equity sampled board".into(),
                "population_eq=honest (multiway NE non-unique)".into(),
                format!("mc_br_proxy_bb={expl}"),
                format!("expl_kind=mc_br_proxy samples={expl_samples}"),
                "exploitability_bb field is MC BR proxy (not tight multiway Nash cert)".into(),
                "avg_strategy=own_reach (accumulated at sampled opponent nodes)".into(),
                if raise_pm.is_empty() && allin {
                    "tree=push_fold (FOLD|ALLIN only)".into()
                } else {
                    format!("raise_pm={raise_pm:?} allin={allin}")
                },
                format!(
                    "pot0_chips={pot0} (n*ante+sb+bb; root.pot_bb ignored for multiway preflop)"
                ),
                "rake=none (not implemented)".into(),
                format!("infosets={}", infosets.len()),
            ];
            nts.extend(notes_extra);
            nts
        },
    })
}

fn mw_preflop_mc_expl(
    infosets: &HashMap<InfosetKey, Infoset>,
    raise_pm: &[u32],
    allin: bool,
    root_state: &PublicState,
    rng: &mut Lcg,
    samples: u32,
) -> f64 {
    let n = root_state.n();
    let bb = root_state.bb;
    if bb == 0 || samples == 0 {
        return 0.0;
    }
    let st = root_state;
    let mut total = 0.0;
    for _ in 0..samples {
        let mut used = [false; 52];
        let mut holes: Vec<(u8, u8)> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut draw = || loop {
                let c = rng.gen_range(52) as u8;
                if !used[c as usize] {
                    used[c as usize] = true;
                    return c;
                }
            };
            let a = draw();
            let b = draw();
            holes.push(if a < b { (a, b) } else { (b, a) });
        }
        let classes: Vec<u32> = holes
            .iter()
            .map(|&(a, b)| PreflopHandClass::from_cards(a, b).id())
            .collect();
        let blocked: Vec<u8> = holes.iter().flat_map(|&(a, b)| [a, b]).collect();
        let board = sample_board5(rng, &blocked);
        for br_player in 0..n {
            let v = mw_avg(
                infosets,
                st,
                &[],
                &classes,
                &holes,
                board,
                br_player,
                raise_pm,
                allin,
                false,
            );
            let br = mw_br(
                infosets,
                &st,
                &[],
                &classes,
                &holes,
                board,
                br_player,
                raise_pm,
                allin,
                false,
            );
            total += (br - v).max(0.0);
        }
    }
    (total / samples as f64) / n as f64 / bb as f64
}

fn mw_preflop_traverse(
    state: &PublicState,
    history: &[AbstractAction],
    classes: &[u32],
    holes: &[(u8, u8)],
    board: [u8; 5],
    traverser: usize,
    raise_pm: &[u32],
    allin: bool,
    infosets: &mut HashMap<InfosetKey, Infoset>,
    path_labels: &mut HashMap<u64, String>,
    rng: &mut Lcg,
) -> f64 {
    if state.is_terminal() || state.actor.is_none() {
        return multi_terminal_real(state, holes, board, traverser);
    }
    let actor = state.actor.unwrap() as usize;
    let acts = legal_actions(state, raise_pm, allin);
    if acts.is_empty() {
        return multi_terminal_real(state, holes, board, traverser);
    }
    // Preflop: board not public yet — action path only (street=0, board_len=0).
    let hhash = history_hash_actions(history);
    path_labels
        .entry(hhash)
        .or_insert_with(|| history_path_label(history));
    let key = InfosetKey::new(actor as u8, hhash, classes[actor]);
    if !infosets.contains_key(&key) {
        let raw = super::range::cards_to_combo(holes[actor].0, holes[actor].1) as u32;
        let dump = InfosetDump::from_state(
            state,
            history,
            PRIV_CLASS,
            classes[actor],
            Some(raw),
            None,
        );
        infosets.insert(key, Infoset::new_with_dump(acts.clone(), dump));
    }
    let strategy = infosets.get(&key).unwrap().current_strategy();
    if actor == traverser {
        let mut utils = vec![0.0; acts.len()];
        let mut node_util = 0.0;
        for (i, &act) in acts.iter().enumerate() {
            let mut child = state.clone();
            if apply_abstract(&mut child, act).is_err() {
                continue;
            }
            let mut h2 = history.to_vec();
            h2.push(act);
            utils[i] = mw_preflop_traverse(
                &child,
                &h2,
                classes,
                holes,
                board,
                traverser,
                raise_pm,
                allin,
                infosets,
                path_labels,
                rng,
            );
            node_util += strategy[i] * utils[i];
        }
        // Regrets only at traverser nodes; the average strategy is accumulated
        // at non-traverser nodes (review 2026-09-20 D6).
        let node = infosets.get_mut(&key).unwrap();
        for i in 0..acts.len() {
            node.regret[i] += utils[i] - node_util;
        }
        node_util
    } else {
        let idx = es_sample_opponent_action(infosets.get_mut(&key).unwrap(), &strategy, rng);
        let mut child = state.clone();
        let _ = apply_abstract(&mut child, acts[idx]);
        let mut h2 = history.to_vec();
        h2.push(acts[idx]);
        mw_preflop_traverse(
            &child,
            &h2,
            classes,
            holes,
            board,
            traverser,
            raise_pm,
            allin,
            infosets,
            path_labels,
            rng,
        )
    }
}

/// Multiway postflop MCCFR: real hole cards, side-pot showdown, ES sampling.
/// Supports unequal stacks via `root.stacks_bb` when non-empty.
pub fn solve_multiway_mccfr(root: &RootSpec, config: &SolveConfig) -> Result<SolveReport, CfrError> {
    if root.num_seats < 3 {
        return Err(CfrError::InvalidRoot("multiway needs num_seats >= 3".into()));
    }
    root.validate_for_solve()?;
    config.validate()?;

    let n = root.num_seats as usize;
    // (review 2026-09-20 E1) chips via the checked converters: a pot/stack that
    // rounds to 0 chips is an error, not an `unwrap()` panic / 0-stack "solve".
    let pot0 = root.pot_chips()?;
    let bb = root.bb_chips;
    let stacks = root.seat_stacks_chips()?;
    let board = root.board.clone();
    let street = root.street as u8;
    let use_combo_view = root.street == StreetRoot::River || config.card_abstraction == "none";
    // Identical every iteration — build (and validate) once.
    let st = PublicState::postflop_root(root.num_seats, pot0, &stacks, &board, bb, street)?;
    require_root_decision(&st, &root.raise_sizes_pm, root.allin_atom)?;

    let mut infosets: HashMap<InfosetKey, Infoset> = HashMap::new();
    let mut rng = Lcg::new(config.seed);
    // Empty raise_sizes + allin_atom = pure jam/check or push/fold menu.
    // Validation rejects empty menu without allin_atom — no silent default.
    let raise_pm = root.raise_sizes_pm.clone();
    let start = std::time::Instant::now();
    let mut stop_reason: Option<&'static str> = None;
    let mut iterations_run = 0u32;
    let iter_limit = config.iter_limit();
    let poll = config.poll_every.max(1);

    for it in 1..=iter_limit {
        if let Some(why) = config.should_stop(start, it) {
            stop_reason = Some(why);
            break;
        }
        let mut used = [false; 52];
        for &c in &board {
            if (c as usize) < 52 {
                used[c as usize] = true;
            }
        }
        let mut holes: Vec<(u8, u8)> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut draw = || loop {
                let c = rng.gen_range(52) as u8;
                if !used[c as usize] {
                    used[c as usize] = true;
                    return c;
                }
            };
            let a = draw();
            let b = draw();
            holes.push(if a < b { (a, b) } else { (b, a) });
        }
        let mut full = [0u8; 5];
        for (i, &c) in board.iter().enumerate() {
            full[i] = c;
        }
        let mut i = board.len();
        while i < 5 {
            let c = rng.gen_range(52) as u8;
            if !used[c as usize] {
                used[c as usize] = true;
                full[i] = c;
                i += 1;
            }
        }

        for trav in 0..n {
            multi_traverse_es(
                &st,
                &[],
                &holes,
                full,
                trav,
                &raise_pm,
                root.allin_atom,
                use_combo_view,
                &mut infosets,
                &mut rng,
            );
        }
        iterations_run = it;
        if it % poll == 0 && table_over_memory_budget(infosets.len()) {
            stop_reason = Some("memory_budget");
            break;
        }
        if (it % poll == 0 || it == 1) && !config.progress_file.is_empty() {
            // Multiway postflop: dump average strategy for live viewer.
            let mut snap = Vec::new();
            for (key, node) in &infosets {
                snap.push(InfosetStrategy::from_node(
                    format!("mw_p{}_h{}_c{}", key.player, key.history, key.private),
                    node,
                ));
            }
            snap.sort_by(|a, b| a.infoset_id.cmp(&b.infoset_id));
            let strat = Strategy::new(root.root_id.clone(), snap);
            config.write_progress(
                it,
                None,
                None,
                infosets.len(),
                Some(&strat),
                &root.root_id,
                false,
            );
        }
    }
    if iterations_run == 0 {
        iterations_run = 1;
    }

    // Real MC exploitability: re-deal holes+board and evaluate terminals with
    // multi_terminal_real (side pots + evaluate_nlh).
    let expl = multiway_mc_exploitability(
        &infosets,
        &raise_pm,
        root.allin_atom,
        &st,
        &board,
        use_combo_view,
        &mut rng,
        32,
    );

    let mut out = Vec::new();
    for (key, node) in &infosets {
        out.push(InfosetStrategy::from_node(
            format!("mw_p{}_h{}_c{}", key.player, key.history, key.private),
            node,
        ));
    }
    out.sort_by(|a, b| a.infoset_id.cmp(&b.infoset_id));

    let mut notes = vec![
        format!("wall_secs={:.1}", start.elapsed().as_secs_f64()),
    ];
    if let Some(why) = stop_reason {
        notes.push(format!("early_stop={why}"));
    }

    Ok(SolveReport {
        status: "ok".into(),
        root: root.clone(),
        config: config.clone(),
        strategy: Strategy::new(root.root_id.clone(), out),
        iterations_run,
        exploitability_bb: Some(expl),
        notes: {
            let mut nts = vec![
                format!("MCCFR multiway n={n} ES + sidepot showdown"),
                format!("infosets={}", infosets.len()),
                format!("mc_br_proxy_bb={expl}"),
                "expl_kind=mc_br_proxy samples=32".into(),
                "exploitability_bb field is MC BR proxy (not tight multiway Nash cert)".into(),
                "avg_strategy=own_reach (accumulated at sampled opponent nodes)".into(),
                "infoset_key=actions+board+street (distinct runouts)".into(),
                if use_combo_view {
                    "private_view=combo".into()
                } else {
                    // Honest label (review 2026-09-20 F8): the 169 preflop class is
                    // blind to suits-vs-board, so flop/turn infosets cannot tell a
                    // flush draw from air. Quality limitation, not redesigned here.
                    "private_view=preflop169 (board-blind: coarse on flop/turn)".into()
                },
                if root.stacks_bb.len() == n {
                    "unequal_stacks".into()
                } else {
                    "equal_stacks".into()
                },
            ];
            nts.extend(notes);
            nts
        },
    })
}

fn mccfr_traverse(
    state: &PublicState,
    history: &[AbstractAction],
    classes: [u32; 2],
    holes: [(u8, u8); 2],
    board: [u8; 5],
    traverser: usize,
    raise_pm: &[u32],
    allin: bool,
    infosets: &mut HashMap<InfosetKey, Infoset>,
    rng: &mut Lcg,
) -> f64 {
    if state.is_terminal() || (state.actor.is_none() && !state.needs_runout()) {
        return preflop_terminal_real(state, holes, board, traverser);
    }
    // Preflop single street: if actor none after close → showdown
    if state.actor.is_none() {
        return preflop_terminal_real(state, holes, board, traverser);
    }

    let actor = state.actor.unwrap() as usize;
    let acts = legal_actions(state, raise_pm, allin);
    if acts.is_empty() {
        return preflop_terminal_real(state, holes, board, traverser);
    }
    let key = InfosetKey::new(actor as u8, history_hash_actions(history), classes[actor]);
    if !infosets.contains_key(&key) {
        let raw = super::range::cards_to_combo(holes[actor].0, holes[actor].1) as u32;
        let dump = InfosetDump::from_state(
            state,
            history,
            PRIV_CLASS,
            classes[actor],
            Some(raw),
            None,
        );
        infosets.insert(key, Infoset::new_with_dump(acts.clone(), dump));
    }
    let strategy = infosets.get(&key).unwrap().current_strategy();

    if actor == traverser {
        let mut utils = vec![0.0; acts.len()];
        let mut node_util = 0.0;
        for (i, &act) in acts.iter().enumerate() {
            let mut child = state.clone();
            if apply_abstract(&mut child, act).is_err() {
                continue;
            }
            let mut h2 = history.to_vec();
            h2.push(act);
            utils[i] = mccfr_traverse(
                &child, &h2, classes, holes, board, traverser, raise_pm, allin, infosets, rng,
            );
            node_util += strategy[i] * utils[i];
        }
        // Regrets only here; average strategy accumulates at the sampled
        // (non-traverser) branch below (review 2026-09-20 D6).
        let node = infosets.get_mut(&key).unwrap();
        for i in 0..acts.len() {
            node.regret[i] += utils[i] - node_util;
        }
        node_util
    } else {
        // External sample opponent action (+ own-reach average-strategy update)
        let idx = es_sample_opponent_action(infosets.get_mut(&key).unwrap(), &strategy, rng);
        let act = acts[idx];
        let mut child = state.clone();
        let _ = apply_abstract(&mut child, act);
        let mut h2 = history.to_vec();
        h2.push(act);
        mccfr_traverse(
            &child, &h2, classes, holes, board, traverser, raise_pm, allin, infosets, rng,
        )
    }
}

fn multi_traverse_es(
    state: &PublicState,
    history: &[AbstractAction],
    holes: &[(u8, u8)],
    board: [u8; 5],
    traverser: usize,
    raise_pm: &[u32],
    allin: bool,
    use_combo_view: bool,
    infosets: &mut HashMap<InfosetKey, Infoset>,
    rng: &mut Lcg,
) -> f64 {
    if state.needs_runout() {
        let idx = state.board_len as usize;
        let mut child = state.clone();
        child.deal_board_card(board[idx.min(4)]);
        return multi_traverse_es(
            &child,
            history,
            holes,
            board,
            traverser,
            raise_pm,
            allin,
            use_combo_view,
            infosets,
            rng,
        );
    }
    if state.is_terminal() || state.actor.is_none() {
        return multi_terminal_real(state, holes, board, traverser);
    }

    let actor = state.actor.unwrap() as usize;
    let acts = legal_actions(state, raise_pm, allin);
    if acts.is_empty() {
        return multi_terminal_real(state, holes, board, traverser);
    }
    // River: exact combo; earlier streets / multiway flop: 169 class abs
    let private = if use_combo_view {
        super::range::cards_to_combo(holes[actor].0, holes[actor].1) as u32
    } else {
        PreflopHandClass::from_cards(holes[actor].0, holes[actor].1).id()
    };
    let key = InfosetKey::new(actor as u8, history_hash_state(history, state), private);
    if !infosets.contains_key(&key) {
        let raw = super::range::cards_to_combo(holes[actor].0, holes[actor].1) as u32;
        let kind = if use_combo_view { PRIV_COMBO } else { PRIV_CLASS };
        let dump = InfosetDump::from_state(state, history, kind, private, Some(raw), None);
        infosets.insert(key, Infoset::new_with_dump(acts.clone(), dump));
    }
    let strategy = infosets.get(&key).unwrap().current_strategy();

    if actor == traverser {
        let mut utils = vec![0.0; acts.len()];
        let mut node_util = 0.0;
        for (i, &act) in acts.iter().enumerate() {
            let mut child = state.clone();
            if apply_abstract(&mut child, act).is_err() {
                continue;
            }
            let mut h2 = history.to_vec();
            h2.push(act);
            utils[i] = multi_traverse_es(
                &child,
                &h2,
                holes,
                board,
                traverser,
                raise_pm,
                allin,
                use_combo_view,
                infosets,
                rng,
            );
            node_util += strategy[i] * utils[i];
        }
        // Regrets only here; average strategy accumulates at the sampled
        // (non-traverser) branch below (review 2026-09-20 D6).
        let node = infosets.get_mut(&key).unwrap();
        for i in 0..acts.len() {
            node.regret[i] += utils[i] - node_util;
        }
        node_util
    } else {
        // External sample (+ own-reach average-strategy update)
        let idx = es_sample_opponent_action(infosets.get_mut(&key).unwrap(), &strategy, rng);
        let mut child = state.clone();
        let _ = apply_abstract(&mut child, acts[idx]);
        let mut h2 = history.to_vec();
        h2.push(acts[idx]);
        multi_traverse_es(
            &child,
            &h2,
            holes,
            board,
            traverser,
            raise_pm,
            allin,
            use_combo_view,
            infosets,
            rng,
        )
    }
}

fn pf_avg_value(
    infosets: &HashMap<InfosetKey, Infoset>,
    state: &PublicState,
    history: &[AbstractAction],
    classes: [u32; 2],
    holes: [(u8, u8); 2],
    board: [u8; 5],
    player: usize,
    raise_pm: &[u32],
    allin: bool,
) -> f64 {
    if state.is_terminal() || state.actor.is_none() {
        return preflop_terminal_real(state, holes, board, player);
    }
    let actor = state.actor.unwrap() as usize;
    let acts = legal_actions(state, raise_pm, allin);
    if acts.is_empty() {
        return preflop_terminal_real(state, holes, board, player);
    }
    let key = InfosetKey::new(actor as u8, history_hash_actions(history), classes[actor]);
    let strat = match infosets.get(&key) {
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
        v += p * pf_avg_value(
            infosets, &child, &h2, classes, holes, board, player, raise_pm, allin,
        );
    }
    v
}

fn pf_br_value(
    infosets: &HashMap<InfosetKey, Infoset>,
    state: &PublicState,
    history: &[AbstractAction],
    classes: [u32; 2],
    holes: [(u8, u8); 2],
    board: [u8; 5],
    br_player: usize,
    raise_pm: &[u32],
    allin: bool,
) -> f64 {
    if state.is_terminal() || state.actor.is_none() {
        return preflop_terminal_real(state, holes, board, br_player);
    }
    let actor = state.actor.unwrap() as usize;
    let acts = legal_actions(state, raise_pm, allin);
    if acts.is_empty() {
        return preflop_terminal_real(state, holes, board, br_player);
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
            let v = pf_br_value(
                infosets, &child, &h2, classes, holes, board, br_player, raise_pm, allin,
            );
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
        let key = InfosetKey::new(actor as u8, history_hash_actions(history), classes[actor]);
        let strat = match infosets.get(&key) {
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
            v += p * pf_br_value(
                infosets, &child, &h2, classes, holes, board, br_player, raise_pm, allin,
            );
        }
        v
    }
}

fn preflop_terminal_real(
    state: &PublicState,
    holes: [(u8, u8); 2],
    board: [u8; 5],
    seat: usize,
) -> f64 {
    if state.alive_count() == 1 {
        return state.fold_payout_chips(seat) as f64;
    }
    let board_cards = [
        Card(board[0]),
        Card(board[1]),
        Card(board[2]),
        Card(board[3]),
        Card(board[4]),
    ];
    let h0 = [Card(holes[0].0), Card(holes[0].1)];
    let h1 = [Card(holes[1].0), Card(holes[1].1)];
    let r0 = evaluate_nlh(&h0, &board_cards);
    let r1 = evaluate_nlh(&h1, &board_cards);
    let pot = state.pot as f64;
    let own = state.total_commit[seat] as f64;
    let eq = if r0 == r1 {
        0.5
    } else if (seat == 0 && r0 > r1) || (seat == 1 && r1 > r0) {
        1.0
    } else {
        0.0
    };
    eq * pot - own
}

fn multi_terminal_real(
    state: &PublicState,
    holes: &[(u8, u8)],
    board: [u8; 5],
    seat: usize,
) -> f64 {
    if state.alive_count() == 1 {
        return state.fold_payout_chips(seat) as f64;
    }
    if state.folded[seat] {
        return -(state.total_commit[seat] as f64);
    }
    // Side-pot layers from total_commit + dead pot remainder.
    let n = state.n();
    let board_cards = [
        Card(board[0]),
        Card(board[1]),
        Card(board[2]),
        Card(board[3]),
        Card(board[4]),
    ];
    let mut ranks = vec![0u32; n];
    for i in 0..n {
        if state.folded[i] {
            continue;
        }
        let hole = [Card(holes[i].0), Card(holes[i].1)];
        ranks[i] = evaluate_nlh(&hole, &board_cards);
    }
    // total_commit only tracks street action; dead pot = pot - sum(total_commit)
    let commits: Vec<u64> = (0..n).map(|i| state.total_commit[i]).collect();
    let sum_c: u64 = commits.iter().sum();
    let dead = state.pot.saturating_sub(sum_c);
    let mut won = vec![0u64; n];
    let mut levels: Vec<u64> = commits.iter().copied().filter(|&c| c > 0).collect();
    levels.sort_unstable();
    levels.dedup();
    let mut prev = 0u64;
    for &level in &levels {
        let contributors: Vec<usize> = (0..n).filter(|&i| commits[i] >= level).collect();
        if contributors.is_empty() {
            continue;
        }
        let layer = (level - prev) * contributors.len() as u64;
        prev = level;
        let eligible: Vec<usize> = contributors
            .into_iter()
            .filter(|&i| !state.folded[i])
            .collect();
        if eligible.is_empty() {
            continue;
        }
        let best = eligible.iter().map(|&i| ranks[i]).max().unwrap_or(0);
        let winners: Vec<usize> = eligible.into_iter().filter(|&i| ranks[i] == best).collect();
        let share = layer / winners.len() as u64;
        let rem = layer % winners.len() as u64;
        for (k, &w) in winners.iter().enumerate() {
            won[w] += share + if (k as u64) < rem { 1 } else { 0 };
        }
    }
    if dead > 0 {
        let alive: Vec<usize> = (0..n).filter(|&i| !state.folded[i]).collect();
        if !alive.is_empty() {
            let best = alive.iter().map(|&i| ranks[i]).max().unwrap_or(0);
            let winners: Vec<usize> = alive.into_iter().filter(|&i| ranks[i] == best).collect();
            let share = dead / winners.len() as u64;
            let rem = dead % winners.len() as u64;
            for (k, &w) in winners.iter().enumerate() {
                won[w] += share + if (k as u64) < rem { 1 } else { 0 };
            }
        }
    }
    won[seat] as f64 - state.total_commit[seat] as f64
}

// silence unused MAX_SEATS if any
const _: usize = MAX_SEATS;

/// Multiway MC NashConv estimate with holes rebound per sample.
fn multiway_mc_exploitability(
    infosets: &HashMap<InfosetKey, Infoset>,
    raise_pm: &[u32],
    allin: bool,
    root_state: &PublicState,
    board: &[u8],
    use_combo_view: bool,
    rng: &mut Lcg,
    samples: u32,
) -> f64 {
    let n = root_state.n();
    let bb = root_state.bb;
    if bb == 0 || samples == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for _ in 0..samples {
        let mut used = [false; 52];
        for &c in board {
            if (c as usize) < 52 {
                used[c as usize] = true;
            }
        }
        let mut holes: Vec<(u8, u8)> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut draw = || loop {
                let c = rng.gen_range(52) as u8;
                if !used[c as usize] {
                    used[c as usize] = true;
                    return c;
                }
            };
            let a = draw();
            let b = draw();
            holes.push(if a < b { (a, b) } else { (b, a) });
        }
        let mut full = [0u8; 5];
        for (i, &c) in board.iter().enumerate() {
            full[i] = c;
        }
        let mut i = board.len();
        while i < 5 {
            let c = rng.gen_range(52) as u8;
            if !used[c as usize] {
                used[c as usize] = true;
                full[i] = c;
                i += 1;
            }
        }
        let root = root_state;
        let privates: Vec<u32> = holes
            .iter()
            .map(|&(a, b)| {
                if use_combo_view {
                    super::range::cards_to_combo(a, b) as u32
                } else {
                    PreflopHandClass::from_cards(a, b).id()
                }
            })
            .collect();
        for br_player in 0..n {
            let v = mw_avg(
                infosets,
                root,
                &[],
                &privates,
                &holes,
                full,
                br_player,
                raise_pm,
                allin,
                true,
            );
            let br = mw_br(
                infosets,
                root,
                &[],
                &privates,
                &holes,
                full,
                br_player,
                raise_pm,
                allin,
                true,
            );
            total += (br - v).max(0.0);
        }
    }
    (total / samples as f64) / n as f64 / bb as f64
}

/// On-policy (average strategy) value of `player` on one sampled deal.
///
/// `deal_runouts`: postflop roots deal the remaining board cards and keep
/// betting; PREFLOP roots must pass `false` — the trained preflop game ends at
/// the close of the preflop round (showdown on the sampled board). (review
/// 2026-09-20, found while fixing E1: the preflop evaluator used to walk into
/// phantom 1-/2-card "streets" that the solver never trained, forcing jams
/// under the push/fold menu whenever two covering stacks were still live.)
fn mw_avg(
    infosets: &HashMap<InfosetKey, Infoset>,
    state: &PublicState,
    history: &[AbstractAction],
    privates: &[u32],
    holes: &[(u8, u8)],
    board: [u8; 5],
    player: usize,
    raise_pm: &[u32],
    allin: bool,
    deal_runouts: bool,
) -> f64 {
    if deal_runouts && state.needs_runout() {
        let mut child = state.clone();
        child.deal_board_card(board[state.board_len as usize]);
        return mw_avg(
            infosets,
            &child,
            history,
            privates,
            holes,
            board,
            player,
            raise_pm,
            allin,
            deal_runouts,
        );
    }
    if state.is_terminal() || state.actor.is_none() {
        return multi_terminal_real(state, holes, board, player);
    }
    let actor = state.actor.unwrap() as usize;
    let acts = legal_actions(state, raise_pm, allin);
    if acts.is_empty() {
        return multi_terminal_real(state, holes, board, player);
    }
    let key = InfosetKey::new(actor as u8, history_hash_state(history, state), privates[actor]);
    let strat = match infosets.get(&key) {
        Some(node) => node.average_strategy(),
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
        v += p * mw_avg(
            infosets,
            &child,
            &h2,
            privates,
            holes,
            board,
            player,
            raise_pm,
            allin,
            deal_runouts,
        );
    }
    v
}

fn mw_br(
    infosets: &HashMap<InfosetKey, Infoset>,
    state: &PublicState,
    history: &[AbstractAction],
    privates: &[u32],
    holes: &[(u8, u8)],
    board: [u8; 5],
    br_player: usize,
    raise_pm: &[u32],
    allin: bool,
    deal_runouts: bool,
) -> f64 {
    if deal_runouts && state.needs_runout() {
        let mut child = state.clone();
        child.deal_board_card(board[state.board_len as usize]);
        return mw_br(
            infosets,
            &child,
            history,
            privates,
            holes,
            board,
            br_player,
            raise_pm,
            allin,
            deal_runouts,
        );
    }
    if state.is_terminal() || state.actor.is_none() {
        return multi_terminal_real(state, holes, board, br_player);
    }
    let actor = state.actor.unwrap() as usize;
    let acts = legal_actions(state, raise_pm, allin);
    if acts.is_empty() {
        return multi_terminal_real(state, holes, board, br_player);
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
            let v = mw_br(
                infosets,
                &child,
                &h2,
                privates,
                holes,
                board,
                br_player,
                raise_pm,
                allin,
                deal_runouts,
            );
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
        let key = InfosetKey::new(actor as u8, history_hash_state(history, state), privates[actor]);
        let strat = match infosets.get(&key) {
            Some(node) => node.average_strategy(),
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
            v += p * mw_br(
                infosets,
                &child,
                &h2,
                privates,
                holes,
                board,
                br_player,
                raise_pm,
                allin,
                deal_runouts,
            );
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::types::SolveConfig;

    #[test]
    fn preflop_mccfr_smoke() {
        let root = RootSpec::preflop_hu(100.0, 10_000, 5_000, 5_000);
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 200;
        cfg.algorithm = "mccfr_es".into();
        let rep = solve_preflop_mccfr(&root, &cfg).unwrap();
        assert_eq!(rep.status, "ok");
        assert!(!rep.strategy.infosets.is_empty());
        assert!(rep.notes.iter().any(|n| n.contains("real NLH equity")));
    }

    #[test]
    fn multiway_mccfr_smoke() {
        let mut root = RootSpec::postflop_hu(
            StreetRoot::River,
            15.0,
            30.0,
            vec![0, 5, 10, 15, 20],
            vec![500, 1000],
        );
        root.num_seats = 3;
        root.root_id = "mw3_river".into();
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 50;
        cfg.algorithm = "mccfr_es".into();
        let rep = solve_multiway_mccfr(&root, &cfg).unwrap();
        assert_eq!(rep.status, "ok");
        assert!(rep.notes.iter().any(|n| n.contains("sidepot") || n.contains("showdown")));
        assert!(rep.notes.iter().any(|n| n.contains("mc_br_proxy")));
        assert!(rep.notes.iter().any(|n| n.contains("infoset_key=actions+board+street")));
    }

    #[test]
    fn preflop_equity_aa_beats_72o() {
        // Direct terminal check: AA vs 72o on random board should favor AA
        let mut wins = 0;
        let mut rng = Lcg::new(1);
        // AA = ranks 12,12; 72o = 5,0
        let aa = (48u8, 49u8); // Ac Ad approx
        let weak = (0u8, 20u8); // 2c 7c if ranks work
        for _ in 0..100 {
            let board = sample_board5(&mut rng, &[aa.0, aa.1, weak.0, weak.1]);
            let r0 = evaluate_nlh(
                &[Card(aa.0), Card(aa.1)],
                &[
                    Card(board[0]),
                    Card(board[1]),
                    Card(board[2]),
                    Card(board[3]),
                    Card(board[4]),
                ],
            );
            let r1 = evaluate_nlh(
                &[Card(weak.0), Card(weak.1)],
                &[
                    Card(board[0]),
                    Card(board[1]),
                    Card(board[2]),
                    Card(board[3]),
                    Card(board[4]),
                ],
            );
            if r0 > r1 {
                wins += 1;
            }
        }
        assert!(wins > 70, "AA should crush weak hand, wins={wins}");
    }

    /// Golden: 4-handed 10bb no-ante push/fold root pot + stack conservation.
    #[test]
    fn pushfold_4handed_10bb_no_ante_root_chips() {
        let bb = 10_000u64;
        let sb = 5_000u64;
        let ante = 0u64;
        let n = 4usize;
        let stack_bb = 10.0;
        let stacks: Vec<u64> = vec![(stack_bb * bb as f64) as u64; n];
        let pot0 = n as u64 * ante + sb + bb; // 15_000
        assert_eq!(pot0, 15_000);
        // After posting: each loses ante; SB/BB lose blinds from remaining.
        let mut remaining = stacks.clone();
        for i in 0..n {
            remaining[i] = remaining[i].saturating_sub(ante);
        }
        let sb_seat = n - 2;
        let bb_seat = n - 1;
        remaining[sb_seat] -= sb;
        remaining[bb_seat] -= bb;
        // CO/BTN 10bb, SB 9.5, BB 9
        assert_eq!(remaining[0], 100_000);
        assert_eq!(remaining[1], 100_000);
        assert_eq!(remaining[2], 95_000);
        assert_eq!(remaining[3], 90_000);
        // Chip conservation: sum(remaining) + pot0 == n * start
        let total = remaining.iter().sum::<u64>() + pot0;
        assert_eq!(total, n as u64 * 100_000);
    }

    /// Empty raise menu without allin is rejected (no silent 50/100% default).
    #[test]
    fn empty_raise_menu_without_allin_rejected() {
        let mut root = RootSpec::preflop_hu(10.0, 10_000, 5_000, 0);
        root.num_seats = 4;
        root.raise_sizes_pm = vec![];
        root.allin_atom = false;
        assert!(root.validate_for_solve().is_err());
    }

    /// stacks_bb length mismatch is refused.
    #[test]
    fn stacks_bb_length_mismatch_rejected() {
        let mut root = RootSpec::preflop_hu(10.0, 10_000, 5_000, 0);
        root.num_seats = 4;
        root.stacks_bb = vec![10.0, 10.0]; // wrong len
        root.raise_sizes_pm = vec![];
        root.allin_atom = true;
        let err = root.validate_for_solve().unwrap_err();
        assert!(format!("{err}").contains("stacks_bb"));
    }

    /// Board+street in history hash: same action path on flop vs turn differs.
    #[test]
    fn history_hash_includes_board_and_street() {
        let acts = [AbstractAction::CheckCall, AbstractAction::CheckCall];
        let flop_board = [0u8, 5, 10, 0, 0];
        let turn_board = [0u8, 5, 10, 15, 0];
        let h_flop = history_hash(&acts, &flop_board, 3, 1);
        let h_turn = history_hash(&acts, &turn_board, 4, 2);
        let h_flop2 = history_hash(&acts, &flop_board, 3, 1);
        assert_eq!(h_flop, h_flop2);
        assert_ne!(
            h_flop, h_turn,
            "flop X/X and turn X/X must be distinct infosets"
        );
        // Different boards same street
        let other_flop = [1u8, 6, 11, 0, 0];
        let h_other = history_hash(&acts, &other_flop, 3, 1);
        assert_ne!(h_flop, h_other);
    }

    /// 3-way side pot: short stack all-in, two deep continue — short only contests main.
    #[test]
    fn three_way_side_pot_payout() {
        // seats: short 10k committed, A 30k, B 30k; pot = 70k (dead 0)
        let mut st = PublicState::postflop_root(
            3,
            70_000,
            &[0, 0, 0],
            &[0, 5, 10, 15, 20],
            10_000,
            3,
        )
        .unwrap();
        st.total_commit = [10_000, 30_000, 30_000, 0, 0, 0];
        st.stacks = [0, 0, 0, 0, 0, 0];
        st.all_in = [true, true, true, false, false, false];
        st.actor = None;
        st.folded = [false, false, false, false, false, false];
        // holes: seat0 weak, seat1 best, seat2 medium — use concrete cards
        // ranks via evaluate_nlh: give seat1 nuts-ish
        let holes = [
            (0u8, 1u8),   // 2c 2d weak
            (48u8, 49u8), // high pair-ish
            (4u8, 8u8),
        ];
        let board = [12u8, 16, 20, 24, 28]; // fixed board
        // Recompute pot consistency: pot should equal sum commits for this unit test
        st.pot = 70_000;
        let u0 = multi_terminal_real(&st, &holes, board, 0);
        let u1 = multi_terminal_real(&st, &holes, board, 1);
        let u2 = multi_terminal_real(&st, &holes, board, 2);
        // Net utilities sum to 0 (zero-sum over commits)
        let sum = u0 + u1 + u2;
        assert!(
            (sum).abs() < 1.0,
            "side-pot nets must be zero-sum, got {sum} ({u0},{u1},{u2})"
        );
        // Winner of main should not be the short stack with weakest cards usually;
        // at least short can only win up to the main pot share.
        assert!(u0 <= 20_000.0, "short stack net capped, got {u0}");
    }

    /// Push/fold 4-handed smoke: pure FOLD|ALLIN, labeled mc_br_proxy.
    #[test]
    fn pushfold_4handed_smoke() {
        let mut root = RootSpec::preflop_hu(10.0, 10_000, 5_000, 0);
        root.num_seats = 4;
        root.pot_bb = 1.5;
        root.raise_sizes_pm = vec![];
        root.allin_atom = true;
        root.stacks_bb = vec![10.0; 4];
        root.root_id = "pf4_test".into();
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 80;
        cfg.algorithm = "mccfr_es".into();
        cfg.seed = 9;
        let rep = solve_multiway_preflop_mccfr(&root, &cfg).unwrap();
        assert_eq!(rep.status, "ok");
        assert!(rep.notes.iter().any(|n| n.contains("push_fold")));
        assert!(rep.notes.iter().any(|n| n.contains("mc_br_proxy")));
        assert!(rep.notes.iter().any(|n| n.contains("pot0_chips=15000")));
        // All infosets should only offer FOLD and/or ALLIN
        for is in &rep.strategy.infosets {
            for a in &is.actions {
                assert!(
                    a == "FOLD" || a == "ALLIN" || a == "CHECK_CALL",
                    "unexpected action {a}"
                );
            }
        }
        // 14 public nodes × 169 ≈ 2366 at high iters; smoke may be less
        assert!(rep.strategy.infosets.len() > 100);
    }

    // ------------------------------------------------------------------
    // (review 2026-09-20 D6) exact-BR harness on a toy HU game driven
    // through the PRODUCTION traversal (`multi_traverse_es`).
    //
    // Chance is an explicit list of equiprobable deals (holes + full board),
    // so the test owns the game exactly; the betting tree is the real
    // PublicState tree and the solver is the production ES traversal.
    // ------------------------------------------------------------------
    #[derive(Clone, Copy)]
    struct ToyDeal {
        holes: [(u8, u8); 2],
        board: [u8; 5],
    }

    struct ToyGame {
        deals: Vec<ToyDeal>,
        root: PublicState,
        raise_pm: Vec<u32>,
        allin: bool,
    }

    const TOY_BB: u64 = 10_000;

    /// Leduc-shaped two-street game (turn root → river): three hands J/Q/K
    /// (kicker 4, never plays), dealt without replacement; the river either
    /// pairs one of the three ranks (that hand then wins) or is a blank.
    /// Both players act on both streets ⇒ own reach at river infosets varies
    /// strongly across iterations — the regime where the D6 bug bites.
    fn toy_leduc_like_game() -> ToyGame {
        // Turn: 2c 7d 3h 8s (no straight/flush reachable with these holes).
        let turn = [0u8, 21, 6, 27];
        // 4c Jc | 4d Qd | 4h Kh
        let hands = [(8u8, 36u8), (9u8, 41u8), (10u8, 46u8)];
        // Js, Qs, Ks, 2s (blank)
        let rivers = [39u8, 43, 47, 3];
        let mut deals = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                for &r in &rivers {
                    deals.push(ToyDeal {
                        holes: [hands[i], hands[j]],
                        board: [turn[0], turn[1], turn[2], turn[3], r],
                    });
                }
            }
        }
        let root =
            PublicState::postflop_root(2, 20_000, &[100_000, 100_000], &turn, TOY_BB, 2).unwrap();
        ToyGame {
            deals,
            root,
            raise_pm: vec![1000],
            allin: false,
        }
    }

    /// Preflop-style push/fold tree (each player acts exactly once): seat 1
    /// (SB) jams or folds into seat 0 (BB), who calls or folds. Six hands of
    /// strictly ordered strength on a fixed board, dealt without replacement.
    fn toy_push_fold_game() -> ToyGame {
        // Board 2c 3d 4h 5s 7c. T-high < Q-high < K-high < 88 < 99 < KK.
        let board = [0u8, 5, 10, 15, 20];
        let hands = [
            (25u8, 32u8),
            (37u8, 40u8),
            (41u8, 44u8),
            (26u8, 27u8),
            (30u8, 31u8),
            (46u8, 47u8),
        ];
        let mut deals = Vec::new();
        for i in 0..hands.len() {
            for j in 0..hands.len() {
                if i != j {
                    deals.push(ToyDeal {
                        holes: [hands[i], hands[j]],
                        board,
                    });
                }
            }
        }
        let mut root =
            PublicState::postflop_root(2, 15_000, &[60_000, 60_000], &board, TOY_BB, 3).unwrap();
        root.street = 0; // push/fold "no limp" rule is preflop-only
        root.stacks[0] = 50_000;
        root.stacks[1] = 55_000;
        root.street_commit[0] = 10_000;
        root.street_commit[1] = 5_000;
        root.total_commit[0] = 10_000;
        root.total_commit[1] = 5_000;
        root.bet_to_call = 10_000;
        root.actor = Some(1);
        ToyGame {
            deals,
            root,
            raise_pm: vec![],
            allin: true,
        }
    }

    /// Per-deal weighted values for `player`: best response (infoset-
    /// consistent, backward induction over reach-weighted deals) or the
    /// dumped average strategy. `w[d]` = chance × opponent average reach.
    fn toy_values(
        infosets: &HashMap<InfosetKey, Infoset>,
        game: &ToyGame,
        state: &PublicState,
        history: &[AbstractAction],
        player: usize,
        best_response: bool,
        w: &[f64],
    ) -> Vec<f64> {
        let nd = game.deals.len();
        if state.needs_runout() {
            // Chance node: split the deals by the next public card.
            let idx = state.board_len as usize;
            let mut cards: Vec<u8> = game.deals.iter().map(|dl| dl.board[idx]).collect();
            cards.sort_unstable();
            cards.dedup();
            let mut out = vec![0.0; nd];
            for c in cards {
                let mut child = state.clone();
                child.deal_board_card(c);
                let w2: Vec<f64> = (0..nd)
                    .map(|d| if game.deals[d].board[idx] == c { w[d] } else { 0.0 })
                    .collect();
                let vals = toy_values(infosets, game, &child, history, player, best_response, &w2);
                for d in 0..nd {
                    out[d] += vals[d];
                }
            }
            return out;
        }
        if state.is_terminal() || state.actor.is_none() {
            return (0..nd)
                .map(|d| {
                    let dl = &game.deals[d];
                    w[d] * multi_terminal_real(state, &dl.holes, dl.board, player)
                })
                .collect();
        }
        let actor = state.actor.unwrap() as usize;
        let acts = legal_actions(state, &game.raise_pm, game.allin);
        assert!(!acts.is_empty());
        let hhash = history_hash_state(history, state);
        let strat_of = |d: usize| -> Vec<f64> {
            let (a, b) = game.deals[d].holes[actor];
            let key = InfosetKey::new(
                actor as u8,
                hhash,
                crate::cfr::range::cards_to_combo(a, b) as u32,
            );
            infosets
                .get(&key)
                .map(|n| n.average_strategy())
                .unwrap_or_else(|| vec![1.0 / acts.len() as f64; acts.len()])
        };
        let mut child_vals: Vec<Vec<f64>> = Vec::with_capacity(acts.len());
        for (i, &act) in acts.iter().enumerate() {
            let mut child = state.clone();
            apply_abstract(&mut child, act).unwrap();
            let mut h2 = history.to_vec();
            h2.push(act);
            let w2: Vec<f64> = if actor == player {
                w.to_vec()
            } else {
                (0..nd).map(|d| w[d] * strat_of(d)[i]).collect()
            };
            child_vals.push(toy_values(
                infosets,
                game,
                &child,
                &h2,
                player,
                best_response,
                &w2,
            ));
        }
        let mut out = vec![0.0; nd];
        if actor != player {
            for cv in &child_vals {
                for d in 0..nd {
                    out[d] += cv[d];
                }
            }
            return out;
        }
        if !best_response {
            for d in 0..nd {
                let s = strat_of(d);
                for (i, cv) in child_vals.iter().enumerate() {
                    out[d] += s[i] * cv[d];
                }
            }
            return out;
        }
        // BR: one action per infoset (= own hole), maximizing the summed
        // counterfactual value of the deals in that infoset.
        let mut groups: HashMap<(u8, u8), Vec<usize>> = HashMap::new();
        for d in 0..nd {
            groups.entry(game.deals[d].holes[player]).or_default().push(d);
        }
        for members in groups.values() {
            let mut best_i = 0;
            let mut best_v = f64::NEG_INFINITY;
            for (i, cv) in child_vals.iter().enumerate() {
                let v: f64 = members.iter().map(|&d| cv[d]).sum();
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            for &d in members {
                out[d] = child_vals[best_i][d];
            }
        }
        out
    }

    /// Exact exploitability (NashConv/2) in bb.
    fn toy_exploitability_bb(infosets: &HashMap<InfosetKey, Infoset>, game: &ToyGame) -> f64 {
        let w0 = vec![1.0 / game.deals.len() as f64; game.deals.len()];
        let mut nashconv = 0.0;
        for p in 0..2 {
            let br: f64 = toy_values(infosets, game, &game.root, &[], p, true, &w0)
                .iter()
                .sum();
            let v: f64 = toy_values(infosets, game, &game.root, &[], p, false, &w0)
                .iter()
                .sum();
            assert!(br >= v - 1e-6, "BR {br} below on-policy value {v}");
            nashconv += br - v;
        }
        nashconv / 2.0 / TOY_BB as f64
    }

    /// Pre-fix ES traversal (average strategy accumulated at TRAVERSER
    /// nodes). Kept ONLY as the regression reference for D6.
    fn legacy_traverse_es(
        game: &ToyGame,
        state: &PublicState,
        history: &[AbstractAction],
        deal: &ToyDeal,
        traverser: usize,
        infosets: &mut HashMap<InfosetKey, Infoset>,
        rng: &mut Lcg,
    ) -> f64 {
        let holes = &deal.holes;
        if state.needs_runout() {
            let mut child = state.clone();
            child.deal_board_card(deal.board[state.board_len as usize]);
            return legacy_traverse_es(game, &child, history, deal, traverser, infosets, rng);
        }
        if state.is_terminal() || state.actor.is_none() {
            return multi_terminal_real(state, holes, deal.board, traverser);
        }
        let actor = state.actor.unwrap() as usize;
        let acts = legal_actions(state, &game.raise_pm, game.allin);
        let private = crate::cfr::range::cards_to_combo(holes[actor].0, holes[actor].1) as u32;
        let key = InfosetKey::new(actor as u8, history_hash_state(history, state), private);
        let strategy = infosets
            .entry(key)
            .or_insert_with(|| Infoset::new(acts.clone()))
            .current_strategy();
        if actor == traverser {
            let mut utils = vec![0.0; acts.len()];
            let mut node_util = 0.0;
            for (i, &act) in acts.iter().enumerate() {
                let mut child = state.clone();
                apply_abstract(&mut child, act).unwrap();
                let mut h2 = history.to_vec();
                h2.push(act);
                utils[i] = legacy_traverse_es(game, &child, &h2, deal, traverser, infosets, rng);
                node_util += strategy[i] * utils[i];
            }
            let node = infosets.get_mut(&key).unwrap();
            for i in 0..acts.len() {
                node.regret[i] += utils[i] - node_util;
                node.strategy_sum[i] += strategy[i]; // <- the D6 bug
            }
            node_util
        } else {
            let mut t = rng.next_f64();
            let mut idx = 0;
            for (i, &p) in strategy.iter().enumerate() {
                t -= p;
                idx = i;
                if t <= 0.0 {
                    break;
                }
            }
            let mut child = state.clone();
            apply_abstract(&mut child, acts[idx]).unwrap();
            let mut h2 = history.to_vec();
            h2.push(acts[idx]);
            legacy_traverse_es(game, &child, &h2, deal, traverser, infosets, rng)
        }
    }

    /// Run `iters` ES-MCCFR iterations; returns exact expl at each checkpoint.
    fn toy_run(game: &ToyGame, iters: u32, seed: u64, legacy: bool, checkpoints: &[u32]) -> Vec<f64> {
        let mut infosets: HashMap<InfosetKey, Infoset> = HashMap::new();
        let mut rng = Lcg::new(seed);
        let mut out = Vec::new();
        for it in 1..=iters {
            let deal = game.deals[rng.gen_range(game.deals.len())];
            for trav in 0..2 {
                if legacy {
                    legacy_traverse_es(game, &game.root, &[], &deal, trav, &mut infosets, &mut rng);
                } else {
                    multi_traverse_es(
                        &game.root,
                        &[],
                        &deal.holes,
                        deal.board,
                        trav,
                        &game.raise_pm,
                        game.allin,
                        true,
                        &mut infosets,
                        &mut rng,
                    );
                }
            }
            if checkpoints.contains(&it) {
                out.push(toy_exploitability_bb(&infosets, game));
            }
        }
        out
    }

    /// (review 2026-09-20 D6) In a tree where players act twice the fixed
    /// (own-reach) average converges; the pre-fix (opponent-reach) one does not.
    #[test]
    fn es_average_strategy_is_own_reach_weighted() {
        let game = toy_leduc_like_game();
        // Sanity: the tree really has second decisions (x/b → OOP again, and
        // a river street after the turn closes).
        let mut s = game.root.clone();
        apply_abstract(&mut s, AbstractAction::CheckCall).unwrap();
        apply_abstract(&mut s, AbstractAction::RaisePm(1000)).unwrap();
        assert_eq!(s.actor, Some(0), "OOP must act again after x/b");
        apply_abstract(&mut s, AbstractAction::CheckCall).unwrap();
        assert!(s.needs_runout(), "turn call must lead to a river street");
        let cps = [2_500u32, 40_000];
        for seed in [1u64, 2, 3] {
            let fixed = toy_run(&game, 40_000, seed, false, &cps);
            let legacy = toy_run(&game, 40_000, seed, true, &cps);
            eprintln!("seed {seed}: fixed={fixed:?} legacy={legacy:?}");
            assert!(
                fixed[1] < fixed[0],
                "seed {seed}: expl must decrease with iterations: {fixed:?}"
            );
            // Measured 2026-09-20 (2 bb pot): fixed 0.040-0.048 bb, legacy
            // 0.28-0.56 bb at 40k iterations.
            assert!(
                fixed[1] < 0.10,
                "seed {seed}: own-reach average should be near-Nash, got {} bb",
                fixed[1]
            );
            assert!(
                3.0 * fixed[1] < legacy[1],
                "seed {seed}: fixed {} bb should clearly beat legacy {} bb",
                fixed[1],
                legacy[1]
            );
        }
    }

    /// (review 2026-09-20 D6) All THREE production traversals: one traversal
    /// leaves the traverser's infosets without average-strategy mass and adds
    /// exactly one unit (Σσ = 1) at every sampled opponent infoset.
    #[test]
    fn es_average_accumulates_only_at_non_traverser_nodes() {
        fn check(infosets: &HashMap<InfosetKey, Infoset>, trav: usize, what: &str) {
            let (mut own, mut opp) = (0, 0);
            for (key, node) in infosets {
                let mass: f64 = node.strategy_sum.iter().sum();
                if key.player as usize == trav {
                    assert_eq!(mass, 0.0, "{what}: traverser node got average mass");
                    own += 1;
                } else {
                    assert!((mass - 1.0).abs() < 1e-12, "{what}: opponent node mass {mass}");
                    opp += 1;
                }
            }
            assert!(own > 0 && opp > 0, "{what}: own={own} opp={opp}");
        }
        let mut rng = Lcg::new(3);

        // HU preflop (`mccfr_traverse`).
        let root = RootSpec::preflop_hu(20.0, 10_000, 5_000, 5_000);
        let st = hu_preflop_root(&root).unwrap();
        let (h0, h1) = deal_holes_hu(&mut rng);
        let classes = [
            PreflopHandClass::from_cards(h0.0, h0.1).id(),
            PreflopHandClass::from_cards(h1.0, h1.1).id(),
        ];
        let board = sample_board5(&mut rng, &[h0.0, h0.1, h1.0, h1.1]);
        for trav in 0..2 {
            let mut infosets = HashMap::new();
            mccfr_traverse(
                &st, &[], classes, [h0, h1], board, trav, &root.raise_sizes_pm, true,
                &mut infosets, &mut rng,
            );
            check(&infosets, trav, "mccfr_traverse");
        }

        // Multiway preflop (`mw_preflop_traverse`), sized tree so seats act twice.
        let mut mw = RootSpec::preflop_hu(20.0, 10_000, 5_000, 0);
        mw.num_seats = 3;
        mw.raise_sizes_pm = vec![1000];
        let st = mw_preflop_root(&mw).unwrap();
        let holes = vec![(0u8, 1u8), (20, 21), (40, 41)];
        let classes: Vec<u32> = holes
            .iter()
            .map(|&(a, b)| PreflopHandClass::from_cards(a, b).id())
            .collect();
        let board = [8u8, 13, 26, 31, 50];
        for trav in 0..3 {
            let mut infosets = HashMap::new();
            let mut labels = HashMap::new();
            mw_preflop_traverse(
                &st, &[], &classes, &holes, board, trav, &mw.raise_sizes_pm, true,
                &mut infosets, &mut labels, &mut rng,
            );
            check(&infosets, trav, "mw_preflop_traverse");
        }

        // Multiway postflop (`multi_traverse_es`).
        let st = PublicState::postflop_root(
            3, 30_000, &[100_000, 100_000, 100_000], &board, 10_000, 3,
        )
        .unwrap();
        for trav in 0..3 {
            let mut infosets = HashMap::new();
            multi_traverse_es(
                &st, &[], &holes, board, trav, &[1000], true, true, &mut infosets, &mut rng,
            );
            check(&infosets, trav, "multi_traverse_es");
        }
    }

    /// (review 2026-09-20 D11) 3-way push/fold with stacks [30, 10, 10] bb: the
    /// covering UTG stack's ALLIN is a real raise to 10 bb in the solved tree
    /// (it used to be a limp, so SB "faced" a 0.5 bb completion).
    #[test]
    fn pushfold_three_way_covering_stack_jams() {
        let mut root = RootSpec::preflop_hu(10.0, 10_000, 5_000, 0);
        root.num_seats = 3;
        root.raise_sizes_pm = vec![];
        root.allin_atom = true;
        root.stacks_bb = vec![30.0, 10.0, 10.0];
        root.root_id = "pf3_cover".into();
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 300;
        cfg.algorithm = "mccfr_es".into();
        cfg.seed = 2;
        let rep = solve_multiway_preflop_mccfr(&root, &cfg).unwrap();
        let sb_vs_jam: Vec<_> = rep
            .strategy
            .infosets
            .iter()
            .filter(|i| i.actor == Some(1) && i.path.as_deref() == Some(&["ALLIN".to_string()][..]))
            .collect();
        assert!(!sb_vs_jam.is_empty());
        for is in sb_vs_jam {
            assert_eq!(is.actions, ["FOLD", "ALLIN"]);
            // Facing a jam to 10 bb with 0.5 bb posted: 9.5 bb to call, pot 11.5 bb.
            assert_eq!(is.to_call_chips, Some(95_000), "UTG's ALLIN was not a raise");
            assert_eq!(is.pot_chips, Some(115_000));
            assert_eq!(is.stacks_chips.as_deref(), Some(&[200_000u64, 95_000, 90_000][..]));
        }
        // The evaluator plays the same (single-street) game the solver trained.
        assert!(rep.exploitability_bb.unwrap().is_finite());
    }

    /// (review 2026-09-20 D6) Push/fold trees (each player acts once) are not
    /// hurt by the fix: the production rule still converges there.
    #[test]
    fn es_average_push_fold_tree_still_converges() {
        let game = toy_push_fold_game();
        let acts = legal_actions(&game.root, &game.raise_pm, game.allin);
        assert_eq!(acts, vec![AbstractAction::Fold, AbstractAction::AllIn]);
        for seed in [1u64, 2] {
            let fixed = toy_run(&game, 20_000, seed, false, &[20_000]);
            let legacy = toy_run(&game, 20_000, seed, true, &[20_000]);
            eprintln!("pushfold seed {seed}: fixed={fixed:?} legacy={legacy:?}");
            assert!(fixed[0] < 0.02, "push/fold expl {} bb", fixed[0]);
            assert!(
                fixed[0] <= legacy[0] + 0.02,
                "push/fold: fixed {} vs legacy {}",
                fixed[0],
                legacy[0]
            );
        }
    }

    /// Multiway flop solve notes board+street key (runout-safe).
    #[test]
    fn multiway_flop_distinct_street_keys() {
        let mut root = RootSpec::postflop_hu(
            StreetRoot::Flop,
            12.0,
            20.0,
            vec![0, 5, 10],
            vec![500, 1000],
        );
        root.num_seats = 3;
        root.root_id = "mw3_flop".into();
        let mut cfg = SolveConfig::default();
        cfg.max_iterations = 40;
        cfg.algorithm = "mccfr_es".into();
        cfg.seed = 11;
        let rep = solve_multiway_mccfr(&root, &cfg).unwrap();
        assert_eq!(rep.status, "ok");
        assert!(rep.notes.iter().any(|n| n.contains("board+street")));
    }
}
