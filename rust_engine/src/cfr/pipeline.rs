//! Full-hand glue: preflop blueprint → range induction → postflop resolve.

use super::mccfr::solve_preflop_mccfr;
use super::preflop::{induce_range, PreflopHandClass, NUM_PREFLOP_CLASSES};
use super::types::{RootSpec, SolveConfig, SolveReport, Strategy, StreetRoot};
use super::{dcfr, CfrError};

/// Result of a scripted preflop → postflop pipeline.
#[derive(Debug, Clone)]
pub struct PipelineReport {
    pub preflop: SolveReport,
    pub postflop: Option<SolveReport>,
    pub induced_oop: Vec<f64>,
    pub induced_ip: Vec<f64>,
    pub notes: Vec<String>,
}

/// Extract average strategy matrix class → action probs from preflop report.
/// Returns (action_labels, probs[class][action]).
fn class_strategy_matrix(
    strategy: &Strategy,
    player: u8,
) -> (Vec<String>, Vec<Vec<f64>>) {
    // Infoset ids: "pf_p{player}_h{history}_c{class}"
    // Use root history (empty / first decision) — history hash varies.
    // Aggregate all infosets for this player at the earliest history by
    // taking the mode history with most classes, or average over histories.
    let mut by_class: Vec<Option<(Vec<String>, Vec<f64>)>> =
        vec![None; NUM_PREFLOP_CLASSES];
    let prefix = format!("pf_p{player}_");
    for is in &strategy.infosets {
        if !is.infoset_id.starts_with(&prefix) {
            continue;
        }
        // parse class from _cN
        let class = is
            .infoset_id
            .rsplit("_c")
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(NUM_PREFLOP_CLASSES);
        if class >= NUM_PREFLOP_CLASSES {
            continue;
        }
        // Keep first seen (root-ish) per class
        if by_class[class].is_none() {
            by_class[class] = Some((is.actions.clone(), is.probs.clone()));
        }
    }
    // Align action labels from first non-empty
    let labels = by_class
        .iter()
        .find_map(|x| x.as_ref().map(|(a, _)| a.clone()))
        .unwrap_or_default();
    let n_act = labels.len().max(1);
    let mut matrix = vec![vec![1.0 / n_act as f64; n_act]; NUM_PREFLOP_CLASSES];
    for (c, entry) in by_class.into_iter().enumerate() {
        if let Some((acts, probs)) = entry {
            if acts.len() == probs.len() && !probs.is_empty() {
                // Map into labels order
                let mut row = vec![0.0; n_act];
                for (a, p) in acts.iter().zip(probs.iter()) {
                    if let Some(idx) = labels.iter().position(|l| l == a) {
                        row[idx] = *p;
                    }
                }
                let s: f64 = row.iter().sum();
                if s > 0.0 {
                    for x in &mut row {
                        *x /= s;
                    }
                }
                matrix[c] = row;
            }
        }
    }
    (labels, matrix)
}

/// Run preflop MCCFR, induce ranges for a chosen abstract action index for
/// each player, then solve a postflop root with those ranges encoded as
/// weight strings (compact: "class:weight,..." for postflop we pass via
/// range_ip/oop as uniform for now and store induced in report — postflop
/// river uses combo ranges from class expansion).
pub fn solve_preflop_to_postflop(
    preflop_root: &RootSpec,
    preflop_cfg: &SolveConfig,
    postflop_board: &[u8],
    postflop_street: StreetRoot,
    pot_bb: f64,
    stack_bb: f64,
    // Action index observed for OOP (player 0) at first decision; None = no filter
    oop_action: Option<usize>,
    ip_action: Option<usize>,
    postflop_cfg: &SolveConfig,
) -> Result<PipelineReport, CfrError> {
    let mut pf_cfg = preflop_cfg.clone();
    if pf_cfg.algorithm == "dcfr" {
        pf_cfg.algorithm = "mccfr_es".into();
    }
    let preflop = solve_preflop_mccfr(preflop_root, &pf_cfg)?;

    let (_lab0, mat0) = class_strategy_matrix(&preflop.strategy, 0);
    let (_lab1, mat1) = class_strategy_matrix(&preflop.strategy, 1);

    let prior: Vec<f64> = vec![1.0 / NUM_PREFLOP_CLASSES as f64; NUM_PREFLOP_CLASSES];
    let induced_oop = if let Some(a) = oop_action {
        induce_range(&prior, &mat0, a)
    } else {
        prior.clone()
    };
    let induced_ip = if let Some(a) = ip_action {
        induce_range(&prior, &mat1, a)
    } else {
        prior
    };

    // Expand class ranges → combo range strings for postflop
    let range_oop = class_weights_to_combo_range(&induced_oop, postflop_board);
    let range_ip = class_weights_to_combo_range(&induced_ip, postflop_board);

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
    Ok(PipelineReport {
        preflop,
        postflop,
        induced_oop,
        induced_ip,
        notes: vec![
            "pipeline: preflop MCCFR -> Bayes induce -> postflop DCFR".into(),
            format!("oop mass={oop_mass:.4} ip mass={ip_mass:.4}"),
        ],
    })
}

/// Encode class weights as combo range string "combo:w,..." for range parser.
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
            // Weight shared among combos of class — approximate by raw class w
            parts.push(format!("{combo}:{w:.6}"));
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
        let pipe = solve_preflop_to_postflop(
            &pf,
            &pcfg,
            &board,
            StreetRoot::River,
            12.0,
            40.0,
            Some(0),
            Some(0),
            &rcfg,
        )
        .expect("pipeline");
        assert_eq!(pipe.preflop.status, "ok");
        assert!(pipe.postflop.is_some());
        assert_eq!(pipe.postflop.as_ref().unwrap().status, "ok");
        assert_eq!(pipe.induced_oop.len(), 169);
    }
}
