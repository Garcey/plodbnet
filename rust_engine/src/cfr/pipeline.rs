//! Full-hand glue: preflop blueprint → range induction → postflop resolve.

use super::actions::{apply_abstract, legal_actions};
use super::mccfr::{hu_preflop_root, solve_preflop_mccfr};
use super::preflop::{PreflopHandClass, NUM_PREFLOP_CLASSES};
use super::types::{RootSpec, SolveConfig, SolveReport, Strategy, StreetRoot};
use super::{dcfr, CfrError};

/// Result of a scripted preflop → postflop pipeline.
#[derive(Debug, Clone)]
pub struct PipelineReport {
    pub preflop: SolveReport,
    pub postflop: Option<SolveReport>,
    pub induced_oop: Vec<f64>,
    pub induced_ip: Vec<f64>,
    /// Preflop line (abstract-action labels) the ranges were induced along.
    pub line: Vec<String>,
    pub notes: Vec<String>,
}

/// Average strategy of `actor` at the public node reached by `path`, per
/// preflop class: `Some((actions, probs))`, or `None` for a class that never
/// visited the node.
///
/// (review 2026-09-20 D9) The node is selected by the dump's `path` + `actor`.
/// The old code took "the first infoset seen per class" while iterating
/// infosets sorted by their `pf_p{player}_h{HASH}_c{class}` id — i.e. an
/// arbitrary node of the tree (whichever history hash sorted first), so the
/// "induced" range was conditioned on a random decision, often deep in a
/// 3-bet line, instead of the root decision.
fn class_strategy_at(
    strategy: &Strategy,
    actor: u8,
    path: &[String],
) -> Vec<Option<(Vec<String>, Vec<f64>)>> {
    let mut by_class: Vec<Option<(Vec<String>, Vec<f64>)>> = vec![None; NUM_PREFLOP_CLASSES];
    for is in &strategy.infosets {
        if is.actor != Some(actor) || is.path.as_deref() != Some(path) {
            continue;
        }
        let class = match is.private_id {
            Some(c) if (c as usize) < NUM_PREFLOP_CLASSES => c as usize,
            _ => continue,
        };
        // Untouched infosets carry the 1/n default, not a learned strategy.
        if is.visit_mass.unwrap_or(0.0) <= 0.0 || is.actions.len() != is.probs.len() {
            continue;
        }
        by_class[class] = Some((is.actions.clone(), is.probs.clone()));
    }
    by_class
}

/// Bayes-update both players' 169-class weights along a preflop `line` of
/// abstract-action labels (e.g. `["RAISE_500", "CHECK_CALL"]`).
///
/// The line is replayed on the real preflop tree, so every step is checked for
/// legality and the acting seat comes from the game, not from guesswork.
/// Actions are aligned **by label**. Returns `(w_oop, w_ip, notes)` with
/// seat 0 = BB = postflop OOP and seat 1 = BTN/SB = postflop IP; each vector
/// is normalized to sum 1.
pub fn induce_ranges_along_line(
    preflop_root: &RootSpec,
    strategy: &Strategy,
    line: &[String],
) -> Result<(Vec<f64>, Vec<f64>, Vec<String>), CfrError> {
    let prior = 1.0 / NUM_PREFLOP_CLASSES as f64;
    let mut w = [
        vec![prior; NUM_PREFLOP_CLASSES],
        vec![prior; NUM_PREFLOP_CLASSES],
    ];
    let mut notes = Vec::new();
    let mut state = hu_preflop_root(preflop_root)?;
    let mut path: Vec<String> = Vec::new();
    for (k, label) in line.iter().enumerate() {
        let actor = state.actor.ok_or_else(|| {
            CfrError::InvalidConfig(format!(
                "preflop line step {k} ({label}): betting is already closed after {path:?}"
            ))
        })? as usize;
        let acts = legal_actions(&state, &preflop_root.raise_sizes_pm, preflop_root.allin_atom);
        let act = *acts.iter().find(|a| &a.label() == label).ok_or_else(|| {
            CfrError::InvalidConfig(format!(
                "preflop line step {k}: {label:?} is not legal after {path:?}; legal = {:?}",
                acts.iter().map(|a| a.label()).collect::<Vec<_>>()
            ))
        })?;
        let table = class_strategy_at(strategy, actor as u8, &path);
        let uniform = 1.0 / acts.len() as f64;
        let mut unvisited = 0usize;
        for class in 0..NUM_PREFLOP_CLASSES {
            let p = match &table[class] {
                Some((labels, probs)) => labels
                    .iter()
                    .position(|l| l == label)
                    .map(|i| probs[i])
                    .unwrap_or(0.0),
                None => {
                    unvisited += 1;
                    uniform
                }
            };
            w[actor][class] *= p;
        }
        if unvisited > 0 {
            notes.push(format!(
                "induce: step {k} seat {actor} {label}: {unvisited}/{NUM_PREFLOP_CLASSES} classes \
                 never visited this node (uniform 1/{} used)",
                acts.len()
            ));
        }
        apply_abstract(&mut state, act)?;
        path.push(label.clone());
    }
    notes.push(format!(
        "induce: line={line:?} closes_preflop={}",
        state.actor.is_none()
    ));
    for (seat, v) in w.iter_mut().enumerate() {
        let t: f64 = v.iter().sum();
        if !(t > 0.0) {
            return Err(CfrError::InvalidConfig(format!(
                "preflop line {line:?} has zero probability for every class of seat {seat}; \
                 the induced range would be empty"
            )));
        }
        v.iter_mut().for_each(|x| *x /= t);
    }
    let [w_oop, w_ip] = w;
    Ok((w_oop, w_ip, notes))
}

/// Legacy index form → label line. `ip_action` indexes the BTN/SB's ROOT
/// decision (path `[]`); `oop_action` indexes the BB's decision FACING that
/// action (path `[ip label]`), so it needs `ip_action`.
fn line_from_indices(
    preflop_root: &RootSpec,
    oop_action: Option<usize>,
    ip_action: Option<usize>,
) -> Result<Vec<String>, CfrError> {
    let mut line = Vec::new();
    let mut state = hu_preflop_root(preflop_root)?;
    let pick = |idx: usize, who: &str, state: &mut super::public_state::PublicState| {
        let acts = legal_actions(state, &preflop_root.raise_sizes_pm, preflop_root.allin_atom);
        let act = *acts.get(idx).ok_or_else(|| {
            CfrError::InvalidConfig(format!(
                "{who}_action index {idx} out of range; legal = {:?}",
                acts.iter().map(|a| a.label()).collect::<Vec<_>>()
            ))
        })?;
        apply_abstract(state, act)?;
        Ok::<String, CfrError>(act.label())
    };
    match (ip_action, oop_action) {
        (None, None) => {}
        (Some(i), oop) => {
            line.push(pick(i, "ip", &mut state)?);
            if let Some(j) = oop {
                if state.actor != Some(0) {
                    return Err(CfrError::InvalidConfig(
                        "oop_action given but the BB has no decision after ip_action".into(),
                    ));
                }
                line.push(pick(j, "oop", &mut state)?);
            }
        }
        (None, Some(_)) => {
            return Err(CfrError::InvalidConfig(
                "oop_action needs ip_action: the BB's decision is only defined against an \
                 explicit BTN/SB action (pass a label `line` for longer sequences)"
                    .into(),
            ));
        }
    }
    Ok(line)
}

/// Run preflop MCCFR, induce both ranges along a preflop action line, then
/// solve a postflop root with those ranges (class weights expanded to combos).
///
/// `oop_action` / `ip_action` are action INDICES at each player's root
/// decision (see [`line_from_indices`]); use
/// [`solve_preflop_to_postflop_line`] to pass labels / longer lines.
pub fn solve_preflop_to_postflop(
    preflop_root: &RootSpec,
    preflop_cfg: &SolveConfig,
    postflop_board: &[u8],
    postflop_street: StreetRoot,
    pot_bb: f64,
    stack_bb: f64,
    oop_action: Option<usize>,
    ip_action: Option<usize>,
    postflop_cfg: &SolveConfig,
) -> Result<PipelineReport, CfrError> {
    let line = line_from_indices(preflop_root, oop_action, ip_action)?;
    solve_preflop_to_postflop_line(
        preflop_root,
        preflop_cfg,
        postflop_board,
        postflop_street,
        pot_bb,
        stack_bb,
        &line,
        postflop_cfg,
    )
}

/// Label-line form of [`solve_preflop_to_postflop`]: `line` is the preflop
/// action sequence from the BTN/SB's first decision, e.g.
/// `["RAISE_500", "CHECK_CALL"]`. An empty line induces nothing (uniform).
#[allow(clippy::too_many_arguments)]
pub fn solve_preflop_to_postflop_line(
    preflop_root: &RootSpec,
    preflop_cfg: &SolveConfig,
    postflop_board: &[u8],
    postflop_street: StreetRoot,
    pot_bb: f64,
    stack_bb: f64,
    line: &[String],
    postflop_cfg: &SolveConfig,
) -> Result<PipelineReport, CfrError> {
    // Validate the line BEFORE spending the preflop budget on it.
    induce_ranges_along_line(preflop_root, &Strategy::default(), line)?;

    let mut pf_cfg = preflop_cfg.clone();
    if pf_cfg.algorithm == "dcfr" {
        pf_cfg.algorithm = "mccfr_es".into();
    }
    let preflop = solve_preflop_mccfr(preflop_root, &pf_cfg)?;

    let (induced_oop, induced_ip, mut notes) =
        induce_ranges_along_line(preflop_root, &preflop.strategy, line)?;

    // Expand class ranges → combo range strings for postflop. A player who did
    // not act on the line keeps the uniform prior: pass NO range, so the solve
    // reports `ranges=uniform…` instead of pretending it parsed one.
    let acted = |seat: usize| -> Result<bool, CfrError> {
        let mut state = hu_preflop_root(preflop_root)?;
        let mut any = false;
        for label in line {
            let acts = legal_actions(&state, &preflop_root.raise_sizes_pm, preflop_root.allin_atom);
            if state.actor == Some(seat as u8) {
                any = true;
            }
            if let Some(a) = acts.iter().find(|a| &a.label() == label) {
                apply_abstract(&mut state, *a)?;
            }
        }
        Ok(any)
    };
    let range_oop = if acted(0)? {
        class_weights_to_combo_range(&induced_oop, postflop_board)
    } else {
        String::new()
    };
    let range_ip = if acted(1)? {
        class_weights_to_combo_range(&induced_ip, postflop_board)
    } else {
        String::new()
    };

    let mut post_root = RootSpec::postflop_hu(
        postflop_street,
        pot_bb,
        stack_bb,
        postflop_board.to_vec(),
        preflop_root.raise_sizes_pm.clone(),
    );
    post_root.range_oop = range_oop;
    post_root.range_ip = range_ip;
    post_root.root_id = format!("pipe_{}_to_{:?}", preflop_root.root_id, postflop_street);

    let postflop = match postflop_street {
        StreetRoot::River => Some(dcfr::solve_river_dcfr(&post_root, postflop_cfg)?),
        StreetRoot::Flop | StreetRoot::Turn => {
            Some(dcfr::solve_postflop_with_runouts(&post_root, postflop_cfg)?)
        }
        StreetRoot::Preflop => None,
    };

    let oop_mass: f64 = induced_oop.iter().sum();
    let ip_mass: f64 = induced_ip.iter().sum();
    let mut all_notes = vec![
        "pipeline: preflop MCCFR -> Bayes induce -> postflop DCFR".into(),
        "induce: node selected by dump path (root decision / explicit line), actions by label"
            .into(),
        format!("oop mass={oop_mass:.4} ip mass={ip_mass:.4}"),
    ];
    all_notes.append(&mut notes);
    Ok(PipelineReport {
        preflop,
        postflop,
        induced_oop,
        induced_ip,
        line: line.to_vec(),
        notes: all_notes,
    })
}

/// Encode class weights as an unambiguous combo range string `#combo:w,...`.
fn class_weights_to_combo_range(class_w: &[f64], board: &[u8]) -> String {
    let mut blocked = [false; 52];
    for &c in board {
        if (c as usize) < 52 {
            blocked[c as usize] = true;
        }
    }
    let mut parts = Vec::new();
    for c0 in 0..52u8 {
        for c1 in (c0 + 1)..52u8 {
            if blocked[c0 as usize] || blocked[c1 as usize] {
                continue;
            }
            let class = PreflopHandClass::from_cards(c0, c1).id() as usize;
            let w = class_w.get(class).copied().unwrap_or(0.0);
            if w <= 0.0 {
                continue;
            }
            let combo = super::range::cards_to_combo(c0, c1);
            // Every combo of a class gets the class posterior: with a uniform
            // per-class prior that is exactly the per-combo posterior ∝ σ(class).
            // `#` = combo id (a bare `44` would read as pocket fours).
            parts.push(format!("#{combo}:{w:.9}"));
        }
    }
    parts.join(",")
}

/// Export pipeline as a combined strategy report (postflop preferred).
pub fn pipeline_to_solve_report(pipe: &PipelineReport) -> SolveReport {
    if let Some(ref post) = pipe.postflop {
        let mut r = post.clone();
        r.notes.extend(pipe.notes.clone());
        r.notes.push("includes preflop blueprint in notes".into());
        return r;
    }
    let mut r = pipe.preflop.clone();
    r.notes.extend(pipe.notes.clone());
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::types::InfosetStrategy;

    #[test]
    fn pipeline_preflop_to_river_smoke() {
        let pf = RootSpec::preflop_hu(50.0, 10_000, 5_000, 5_000);
        let mut pcfg = SolveConfig::default();
        pcfg.max_iterations = 80;
        pcfg.algorithm = "mccfr_es".into();
        let mut rcfg = SolveConfig::default();
        rcfg.max_iterations = 40;
        rcfg.seed = 3;
        rcfg.target_exploitability_bb = 0.0;
        let board = vec![0, 5, 10, 15, 20];
        // ip index 1 = BTN limp (CHECK_CALL), oop index 0 = BB checks back.
        let pipe = solve_preflop_to_postflop(
            &pf,
            &pcfg,
            &board,
            StreetRoot::River,
            12.0,
            40.0,
            Some(0),
            Some(1),
            &rcfg,
        )
        .expect("pipeline");
        assert_eq!(pipe.preflop.status, "ok");
        assert!(pipe.postflop.is_some());
        assert_eq!(pipe.postflop.as_ref().unwrap().status, "ok");
        assert_eq!(pipe.induced_oop.len(), 169);
        assert_eq!(pipe.line, vec!["CHECK_CALL".to_string(), "CHECK_CALL".to_string()]);
        assert!(pipe.notes.iter().any(|n| n.contains("closes_preflop=true")));
        let post = pipe.postflop.as_ref().unwrap();
        assert!(post.notes.iter().any(|n| n.starts_with("ranges=parsed")), "{:?}", post.notes);

        // oop_action without ip_action has no defined node → error.
        assert!(solve_preflop_to_postflop(
            &pf, &pcfg, &board, StreetRoot::River, 12.0, 40.0, Some(0), None, &rcfg
        )
        .is_err());
        // Illegal line step → error before any solving.
        assert!(solve_preflop_to_postflop_line(
            &pf, &pcfg, &board, StreetRoot::River, 12.0, 40.0, &["RAISE_77".to_string()], &rcfg
        )
        .is_err());
        // No line → nothing induced → postflop solve is labelled uniform.
        let pipe = solve_preflop_to_postflop(
            &pf, &pcfg, &board, StreetRoot::River, 12.0, 40.0, None, None, &rcfg,
        )
        .expect("no line");
        let post = pipe.postflop.as_ref().unwrap();
        assert!(post.notes.iter().any(|n| n.starts_with("ranges=uniform_fallback")));
    }

    fn fake_infoset(
        actor: u8,
        path: &[&str],
        class: u32,
        actions: &[&str],
        probs: &[f64],
        id: &str,
    ) -> InfosetStrategy {
        InfosetStrategy {
            infoset_id: id.into(),
            actions: actions.iter().map(|s| s.to_string()).collect(),
            probs: probs.to_vec(),
            schema_version: 2,
            actor: Some(actor),
            path: Some(path.iter().map(|s| s.to_string()).collect()),
            private_kind: Some("class".into()),
            private_id: Some(class),
            visit_mass: Some(10.0),
            ..Default::default()
        }
    }

    /// (review 2026-09-20 D9) the ROOT decision is selected by path, even when
    /// a deeper node of the same player sorts first by infoset id, and actions
    /// are matched by label, not by position.
    #[test]
    fn induction_reads_the_root_decision_by_path_and_label() {
        let mut pf = RootSpec::preflop_hu(50.0, 10_000, 5_000, 5_000);
        pf.raise_sizes_pm = vec![1000];
        pf.allin_atom = false;
        let aa = PreflopHandClass::from_cards(48, 49).id();
        let sevdeuce = PreflopHandClass::from_cards(0, 21).id();
        let raise = {
            let st = hu_preflop_root(&pf).unwrap();
            let acts = legal_actions(&st, &pf.raise_sizes_pm, pf.allin_atom);
            assert_eq!(acts[0].label(), "FOLD");
            acts.last().unwrap().label()
        };
        let infosets = vec![
            // Deep BTN node (facing a 3-bet) — sorts FIRST by id ("pf_p1_h1…").
            fake_infoset(1, &[&raise, &raise], aa, &["FOLD", "CHECK_CALL"], &[0.0, 1.0], "pf_p1_h1_c0"),
            fake_infoset(1, &[&raise, &raise], sevdeuce, &["FOLD", "CHECK_CALL"], &[1.0, 0.0], "pf_p1_h1_c1"),
            // BTN ROOT node, actions deliberately listed in a different order.
            fake_infoset(1, &[], aa, &[&raise, "CHECK_CALL", "FOLD"], &[0.9, 0.1, 0.0], "pf_p1_h9_c0"),
            fake_infoset(1, &[], sevdeuce, &[&raise, "CHECK_CALL", "FOLD"], &[0.1, 0.1, 0.8], "pf_p1_h9_c1"),
            // BB facing the raise.
            fake_infoset(0, &[&raise], aa, &["FOLD", "CHECK_CALL", &raise], &[0.0, 0.2, 0.8], "pf_p0_h5_c0"),
            fake_infoset(0, &[&raise], sevdeuce, &["FOLD", "CHECK_CALL", &raise], &[0.95, 0.05, 0.0], "pf_p0_h5_c1"),
        ];
        let strat = Strategy::new("t", infosets);
        let line = vec![raise.clone(), "CHECK_CALL".to_string()];
        let (w_oop, w_ip, _) = induce_ranges_along_line(&pf, &strat, &line).unwrap();
        // IP = BTN raised: P(raise|AA)=0.9 vs 0.1 for 72 (root node, by label).
        let r = w_ip[aa as usize] / w_ip[sevdeuce as usize];
        assert!((r - 9.0).abs() < 1e-9, "IP AA:72 ratio {r} (want 9)");
        // OOP = BB called the raise: 0.2 vs 0.05.
        let r = w_oop[aa as usize] / w_oop[sevdeuce as usize];
        assert!((r - 4.0).abs() < 1e-9, "OOP AA:72 ratio {r} (want 4)");
        assert!((w_ip.iter().sum::<f64>() - 1.0).abs() < 1e-9);

        // An action NO class ever takes → explicit error, not a silent uniform
        // range (the empty induced range used to parse as "no range given").
        let never: Vec<InfosetStrategy> = (0..NUM_PREFLOP_CLASSES as u32)
            .map(|c| {
                fake_infoset(1, &[], c, &["FOLD", "CHECK_CALL", &raise], &[0.5, 0.5, 0.0], "x")
            })
            .collect();
        let err = induce_ranges_along_line(&pf, &Strategy::new("t", never), &line[..1])
            .expect_err("empty induced range");
        assert!(format!("{err}").contains("zero probability"), "{err}");
    }

    #[test]
    fn combo_range_string_round_trips_through_the_parser() {
        let mut w = vec![0.0; NUM_PREFLOP_CLASSES];
        let fours = PreflopHandClass::from_cards(8, 9).id() as usize; // 4c4d
        w[fours] = 1.0;
        let s = class_weights_to_combo_range(&w, &[0, 5, 10, 15, 20]);
        let r = crate::cfr::range::Range::parse(&s, &[0, 5, 10, 15, 20]).unwrap();
        // 4h (card 10) is on the board ⇒ 3 of the 6 combos remain.
        assert_eq!(r.live_combos(), 3);
        assert!(s.starts_with('#'));
    }
}
