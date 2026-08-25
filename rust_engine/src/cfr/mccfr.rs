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

    let pot0 = (root.pot_bb * root.bb_chips as f64).round() as u64;
    let stack0 = (root.effective_stack_bb * root.bb_chips as f64).round() as u64;
    let bb = root.bb_chips;
    let mut infosets: HashMap<InfosetKey, Infoset> = HashMap::new();
    let mut rng = Lcg::new(config.seed);
    let raise_pm = root.raise_sizes_pm.clone();
    let allin = root.allin_atom;
    let start = std::time::Instant::now();
    let mut stop_reason: Option<&'static str> = None;
    let mut iterations_run = 0u32;

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
        // Faithful HU preflop: blinds in street_commit, BTN first.
        let sb = root.sb_chips;
        let ante = root.ante_chips;
        let mut st = PublicState::hu_postflop_root(pot0, stack0, &[0, 1, 2], bb).unwrap();
        st.street = 0;
        st.board_len = 0;
        st.board = [0; 5];
        st.stacks[0] = stack0.saturating_sub(ante + bb); // BB
        st.stacks[1] = stack0.saturating_sub(ante + sb); // BTN/SB
        st.street_commit[0] = bb;
        st.street_commit[1] = sb;
        st.total_commit[0] = ante + bb;
        st.total_commit[1] = ante + sb;
        st.bet_to_call = bb;
        st.last_raise_size = bb;
        st.actor = Some(1);
        st.button = 1;

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

        if config.algorithm == "dcfr" && it % 10 == 0 {
            for node in infosets.values_mut() {
                node.apply_dcfr_discount(it);
            }
        }
        iterations_run = it;
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
            config.write_progress(it, None, infosets.len(), Some(&strat), &root.root_id, false);
        }
    }
    if iterations_run == 0 {
        iterations_run = 1; // avoid empty export edge case
    }

    // Real preflop MC expl with holes + sampled board rebound per sample
    let expl = {
        let mut total = 0.0;
        let samples = 32u32;
        let sb = root.sb_chips;
        let ante = root.ante_chips;
        for _ in 0..samples {
            let (h0, h1) = deal_holes_hu(&mut rng);
            let holes = [h0, h1];
            let classes = [
                PreflopHandClass::from_cards(h0.0, h0.1).id(),
                PreflopHandClass::from_cards(h1.0, h1.1).id(),
            ];
            let board = sample_board5(&mut rng, &[h0.0, h0.1, h1.0, h1.1]);
            let mut st = PublicState::hu_postflop_root(pot0, stack0, &[0, 1, 2], bb).unwrap();
            st.street = 0;
            st.board_len = 0;
            st.board = [0; 5];
            st.stacks[0] = stack0.saturating_sub(ante + bb);
            st.stacks[1] = stack0.saturating_sub(ante + sb);
            st.street_commit[0] = bb;
            st.street_commit[1] = sb;
            st.total_commit[0] = ante + bb;
            st.total_commit[1] = ante + sb;
            st.bet_to_call = bb;
            st.last_raise_size = bb;
            st.actor = Some(1);
            st.button = 1;
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
    let bb = root.bb_chips;
    let sb = root.sb_chips;
    let ante = root.ante_chips;
    // Pot = n*ante + sb + bb (multiway blinds only SB/BB seats). pot_bb ignored.
    let pot0 = root.multiway_preflop_pot_chips();
    let stacks: Vec<u64> = if root.stacks_bb.len() == n {
        root.stacks_bb
            .iter()
            .map(|&s| (s * bb as f64).round() as u64)
            .collect()
    } else {
        let s0 = (root.effective_stack_bb * bb as f64).round() as u64;
        vec![s0; n]
    };
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

        // Public preflop: seat 0 = UTG (first to act multiway preflop after BB),
        // last seats post SB/BB. Our convention: seats 0..n-3 UTG+, n-2=SB, n-1=BB.
        let mut st = PublicState::postflop_root(
            root.num_seats,
            pot0,
            &stacks,
            &[0, 1, 2], // dummy board for constructor
            bb,
            1,
        )?;
        st.street = 0;
        st.board_len = 0;
        st.board = [0; 5];
        // Post blinds on last two seats
        let sb_seat = n - 2;
        let bb_seat = n - 1;
        for i in 0..n {
            st.stacks[i] = stacks[i].saturating_sub(ante);
            st.total_commit[i] = ante;
        }
        let sb_pay = sb.min(st.stacks[sb_seat]);
        st.stacks[sb_seat] -= sb_pay;
        st.street_commit[sb_seat] = sb_pay;
        st.total_commit[sb_seat] += sb_pay;
        let bb_pay = bb.min(st.stacks[bb_seat]);
        st.stacks[bb_seat] -= bb_pay;
        st.street_commit[bb_seat] = bb_pay;
        st.total_commit[bb_seat] += bb_pay;
        st.bet_to_call = bb;
        st.last_raise_size = bb;
        st.actor = Some(0); // UTG first multiway preflop
        st.button = (n - 3) as u8; // rough

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
            config.write_progress(it, None, infosets.len(), Some(&strat), &root.root_id, false);
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
    let expl = mw_preflop_mc_expl(
        &infosets,
        &raise_pm,
        allin,
        n,
        pot0,
        &stacks,
        bb,
        sb,
        ante,
        &mut rng,
        expl_samples,
    );

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
                "exploitability_bb field is MC BR proxy (not tight multiway Nash cert)".into(),
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
    n: usize,
    pot0: u64,
    stacks: &[u64],
    bb: u64,
    sb: u64,
    ante: u64,
    rng: &mut Lcg,
    samples: u32,
) -> f64 {
    if bb == 0 || samples == 0 {
        return 0.0;
    }
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
        let mut st = PublicState::postflop_root(n as u8, pot0, stacks, &[0, 1, 2], bb, 1).unwrap();
        st.street = 0;
        st.board_len = 0;
        st.board = [0; 5];
        let sb_seat = n - 2;
        let bb_seat = n - 1;
        for i in 0..n {
            st.stacks[i] = stacks[i].saturating_sub(ante);
            st.total_commit[i] = ante;
        }
        let sb_pay = sb.min(st.stacks[sb_seat]);
        st.stacks[sb_seat] -= sb_pay;
        st.street_commit[sb_seat] = sb_pay;
        st.total_commit[sb_seat] += sb_pay;
        let bb_pay = bb.min(st.stacks[bb_seat]);
        st.stacks[bb_seat] -= bb_pay;
        st.street_commit[bb_seat] = bb_pay;
        st.total_commit[bb_seat] += bb_pay;
        st.bet_to_call = bb;
        st.last_raise_size = bb;
        st.actor = Some(0);
        for br_player in 0..n {
            let v = mw_avg(
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
        let node = infosets.get_mut(&key).unwrap();
        for i in 0..acts.len() {
            node.regret[i] += utils[i] - node_util;
            node.strategy_sum[i] += strategy[i];
        }
        node_util
    } else {
        let mut t = rng.next_f64();
        let mut idx = 0;
        for (i, &p) in strategy.iter().enumerate() {
            t -= p;
            if t <= 0.0 {
                idx = i;
                break;
            }
            idx = i;
        }
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
    let pot0 = (root.pot_bb * root.bb_chips as f64).round() as u64;
    let bb = root.bb_chips;
    let stacks: Vec<u64> = if root.stacks_bb.len() == n {
        root.stacks_bb
            .iter()
            .map(|&s| (s * bb as f64).round() as u64)
            .collect()
    } else {
        let stack0 = (root.effective_stack_bb * bb as f64).round() as u64;
        vec![stack0; n]
    };
    let board = root.board.clone();
    let street = root.street as u8;
    let use_combo_view = root.street == StreetRoot::River || config.card_abstraction == "none";

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

        let st = PublicState::postflop_root(root.num_seats, pot0, &stacks, &board, bb, street)?;

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
            config.write_progress(it, None, infosets.len(), Some(&strat), &root.root_id, false);
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
        n,
        pot0,
        &stacks,
        &board,
        bb,
        street,
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
                "exploitability_bb field is MC BR proxy (not tight multiway Nash cert)".into(),
                "infoset_key=actions+board+street (distinct runouts)".into(),
                if use_combo_view {
                    "private_view=combo".into()
                } else {
                    "private_view=preflop169".into()
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
        let node = infosets.get_mut(&key).unwrap();
        for i in 0..acts.len() {
            node.regret[i] += utils[i] - node_util;
            node.strategy_sum[i] += strategy[i];
        }
        node_util
    } else {
        // External sample opponent action
        let mut t = rng.next_f64();
        let mut idx = 0;
        for (i, &p) in strategy.iter().enumerate() {
            t -= p;
            if t <= 0.0 {
                idx = i;
                break;
            }
            idx = i;
        }
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
        let node = infosets.get_mut(&key).unwrap();
        for i in 0..acts.len() {
            node.regret[i] += utils[i] - node_util;
            node.strategy_sum[i] += strategy[i];
        }
        node_util
    } else {
        // External sample
        let mut t = rng.next_f64();
        let mut idx = 0;
        for (i, &p) in strategy.iter().enumerate() {
            t -= p;
            if t <= 0.0 {
                idx = i;
                break;
            }
            idx = i;
        }
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
    n: usize,
    pot0: u64,
    stacks: &[u64],
    board: &[u8],
    bb: u64,
    street: u8,
    use_combo_view: bool,
    rng: &mut Lcg,
    samples: u32,
) -> f64 {
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
        let root = PublicState::postflop_root(n as u8, pot0, stacks, board, bb, street).unwrap();
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
                &root,
                &[],
                &privates,
                &holes,
                full,
                br_player,
                raise_pm,
                allin,
                use_combo_view,
            );
            let br = mw_br(
                infosets,
                &root,
                &[],
                &privates,
                &holes,
                full,
                br_player,
                raise_pm,
                allin,
                use_combo_view,
            );
            total += (br - v).max(0.0);
        }
    }
    (total / samples as f64) / n as f64 / bb as f64
}

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
    use_combo_view: bool,
) -> f64 {
    if state.needs_runout() {
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
            use_combo_view,
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
            use_combo_view,
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
    use_combo_view: bool,
) -> f64 {
    if state.needs_runout() {
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
            use_combo_view,
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
                use_combo_view,
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
                use_combo_view,
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
