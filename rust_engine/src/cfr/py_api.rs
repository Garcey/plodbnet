//! PyO3 surface for CFR solve.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::types::{RootSpec, SolveConfig, StreetRoot};
use super::{solve, solve_kuhn, CfrError};

fn street_from_u8(v: u8) -> PyResult<StreetRoot> {
    StreetRoot::from_u8(v).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn map_err(e: CfrError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Solve a CFR root. Returns a dict matching Python ``SolveReport.as_dict()``.
///
/// Arguments mirror ``RootSpec`` / ``SolveConfig`` fields.
#[pyfunction]
#[pyo3(signature = (
    street,
    pot_bb,
    effective_stack_bb,
    board,
    raise_sizes_pm,
    max_iterations=200,
    target_exploitability_bb=0.5,
    thread_num=1,
    seed=0,
    algorithm="dcfr",
    card_abstraction="none",
    num_seats=2,
    bb_chips=10000,
    sb_chips=5000,
    ante_chips=5000,
    allin_atom=true,
    range_ip="",
    range_oop="",
    root_id="",
    use_isomorphism=true,
    stacks_bb=vec![],
    time_budget_secs=0.0,
    stop_file="",
    poll_every=500,
    pause_file="",
    progress_file="",
))]
#[allow(clippy::too_many_arguments)]
pub fn cfr_solve<'py>(
    py: Python<'py>,
    street: u8,
    pot_bb: f64,
    effective_stack_bb: f64,
    board: Vec<u8>,
    raise_sizes_pm: Vec<u32>,
    max_iterations: u32,
    target_exploitability_bb: f64,
    thread_num: u32,
    seed: u64,
    algorithm: &str,
    card_abstraction: &str,
    num_seats: u8,
    bb_chips: u64,
    sb_chips: u64,
    ante_chips: u64,
    allin_atom: bool,
    range_ip: &str,
    range_oop: &str,
    root_id: &str,
    use_isomorphism: bool,
    stacks_bb: Vec<f64>,
    time_budget_secs: f64,
    stop_file: &str,
    poll_every: u32,
    pause_file: &str,
    progress_file: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let street = street_from_u8(street)?;
    let rid = if root_id.is_empty() {
        format!("s{street:?}_pot{pot_bb}")
    } else {
        root_id.to_string()
    };
    let root = RootSpec {
        num_seats,
        street,
        pot_bb,
        effective_stack_bb,
        bb_chips,
        sb_chips,
        ante_chips,
        board,
        raise_sizes_pm,
        allin_atom,
        range_ip: range_ip.to_string(),
        range_oop: range_oop.to_string(),
        stacks_bb,
        root_id: rid,
    };
    let config = SolveConfig {
        max_iterations,
        target_exploitability_bb,
        thread_num,
        seed,
        use_isomorphism,
        algorithm: algorithm.to_string(),
        card_abstraction: card_abstraction.to_string(),
        time_budget_secs,
        stop_file: stop_file.to_string(),
        poll_every: poll_every.max(1),
        pause_file: pause_file.to_string(),
        progress_file: progress_file.to_string(),
    };
    // Release the GIL so the FastAPI / desktop event loop can keep serving
    // progress polls while a long solve runs (was the main "app freezes" bug).
    let report = py.allow_threads(|| solve(&root, &config)).map_err(map_err)?;

    let d = PyDict::new(py);
    d.set_item("status", report.status)?;
    d.set_item("iterations_run", report.iterations_run)?;
    match report.exploitability_bb {
        Some(e) => d.set_item("exploitability_bb", e)?,
        None => d.set_item("exploitability_bb", py.None())?,
    }
    d.set_item("notes", report.notes)?;

    let root_d = PyDict::new(py);
    root_d.set_item("street", report.root.street as u8)?;
    root_d.set_item("pot_bb", report.root.pot_bb)?;
    root_d.set_item("effective_stack_bb", report.root.effective_stack_bb)?;
    // Use list[int] not bytes (Vec<u8> → bytes is not JSON-serializable).
    let board_list = PyList::new(
        py,
        report
            .root
            .board
            .iter()
            .map(|&c| c as i32)
            .collect::<Vec<_>>(),
    )?;
    root_d.set_item("board", board_list)?;
    root_d.set_item("num_seats", report.root.num_seats)?;
    root_d.set_item("bb_chips", report.root.bb_chips)?;
    root_d.set_item("sb_chips", report.root.sb_chips)?;
    root_d.set_item("ante_chips", report.root.ante_chips)?;
    let raise_list = PyList::new(
        py,
        report
            .root
            .raise_sizes_pm
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>(),
    )?;
    root_d.set_item("raise_sizes_pm", raise_list)?;
    root_d.set_item("allin_atom", report.root.allin_atom)?;
    root_d.set_item("range_ip", report.root.range_ip.clone())?;
    root_d.set_item("range_oop", report.root.range_oop.clone())?;
    root_d.set_item("stacks_bb", report.root.stacks_bb.clone())?;
    root_d.set_item("root_id", report.root.root_id.clone())?;
    d.set_item("root", root_d)?;

    let cfg_d = PyDict::new(py);
    cfg_d.set_item("max_iterations", report.config.max_iterations)?;
    cfg_d.set_item(
        "target_exploitability_bb",
        report.config.target_exploitability_bb,
    )?;
    cfg_d.set_item("thread_num", report.config.thread_num)?;
    cfg_d.set_item("seed", report.config.seed)?;
    cfg_d.set_item("use_isomorphism", report.config.use_isomorphism)?;
    cfg_d.set_item("algorithm", report.config.algorithm.clone())?;
    cfg_d.set_item("card_abstraction", report.config.card_abstraction.clone())?;
    cfg_d.set_item("time_budget_secs", report.config.time_budget_secs)?;
    cfg_d.set_item("stop_file", report.config.stop_file.clone())?;
    cfg_d.set_item("poll_every", report.config.poll_every)?;
    cfg_d.set_item("pause_file", report.config.pause_file.clone())?;
    cfg_d.set_item("progress_file", report.config.progress_file.clone())?;
    d.set_item("config", cfg_d)?;

    let strat = strategy_to_pydict(py, &report.strategy)?;
    d.set_item("strategy", strat)?;
    Ok(d)
}

/// Kuhn poker solve gate. Returns dict with value_p0, exploitability, iterations.
#[pyfunction]
#[pyo3(signature = (iterations=5000))]
pub fn cfr_solve_kuhn(py: Python<'_>, iterations: u32) -> PyResult<Bound<'_, PyDict>> {
    let rep = solve_kuhn(iterations);
    let d = PyDict::new(py);
    d.set_item("iterations", rep.iterations)?;
    d.set_item("deals", rep.deals)?;
    d.set_item("value_p0", rep.value_p0)?;
    d.set_item("exploitability", rep.exploitability)?;
    d.set_item("training_value", rep.training_value)?;
    d.set_item("nash_value", super::kuhn::KUHN_NASH_VALUE)?;
    Ok(d)
}

/// Range induction helper: prior × action probs → posterior.
#[pyfunction]
pub fn cfr_induce_range(
    prior: Vec<f64>,
    action_probs_by_class: Vec<Vec<f64>>,
    action_idx: usize,
) -> PyResult<Vec<f64>> {
    if prior.len() != action_probs_by_class.len() {
        return Err(PyValueError::new_err(
            "prior and action_probs_by_class length mismatch",
        ));
    }
    Ok(super::preflop::induce_range(
        &prior,
        &action_probs_by_class,
        action_idx,
    ))
}

/// Preflop → induce → postflop pipeline (full-hand glue).
#[pyfunction]
#[pyo3(signature = (
    stack_bb=100.0,
    preflop_iters=200,
    postflop_iters=100,
    postflop_board=vec![0,5,10,15,20],
    pot_bb=12.0,
    postflop_stack_bb=40.0,
    oop_action=None,
    ip_action=None,
    seed=0,
    preflop_time_budget_secs=0.0,
    postflop_time_budget_secs=0.0,
    stop_file="",
    raise_sizes_pm=vec![330, 500, 1000, 1500],
))]
#[allow(clippy::too_many_arguments)]
pub fn cfr_pipeline<'py>(
    py: Python<'py>,
    stack_bb: f64,
    preflop_iters: u32,
    postflop_iters: u32,
    postflop_board: Vec<u8>,
    pot_bb: f64,
    postflop_stack_bb: f64,
    oop_action: Option<usize>,
    ip_action: Option<usize>,
    seed: u64,
    preflop_time_budget_secs: f64,
    postflop_time_budget_secs: f64,
    stop_file: &str,
    raise_sizes_pm: Vec<u32>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut pf = RootSpec::preflop_hu(stack_bb, 10_000, 5_000, 5_000);
    if !raise_sizes_pm.is_empty() {
        pf.raise_sizes_pm = raise_sizes_pm;
    }
    let mut pcfg = SolveConfig::default();
    pcfg.max_iterations = preflop_iters;
    pcfg.algorithm = "mccfr_es".into();
    pcfg.seed = seed;
    pcfg.time_budget_secs = preflop_time_budget_secs;
    pcfg.stop_file = stop_file.to_string();
    pcfg.poll_every = 1000;
    let mut rcfg = SolveConfig::default();
    rcfg.max_iterations = postflop_iters;
    rcfg.seed = seed.wrapping_add(1);
    rcfg.target_exploitability_bb = 0.0;
    rcfg.time_budget_secs = postflop_time_budget_secs;
    rcfg.stop_file = stop_file.to_string();
    rcfg.poll_every = 50;
    let street = match postflop_board.len() {
        3 => StreetRoot::Flop,
        4 => StreetRoot::Turn,
        5 => StreetRoot::River,
        _ => {
            return Err(PyValueError::new_err(
                "postflop_board must be 3, 4, or 5 cards",
            ))
        }
    };
    if street == StreetRoot::Flop {
        rcfg.card_abstraction = "ochs".into();
    }
    let pipe = super::pipeline::solve_preflop_to_postflop(
        &pf,
        &pcfg,
        &postflop_board,
        street,
        pot_bb,
        postflop_stack_bb,
        oop_action,
        ip_action,
        &rcfg,
    )
    .map_err(map_err)?;
    let d = PyDict::new(py);
    d.set_item("preflop_status", pipe.preflop.status.clone())?;
    d.set_item("preflop_infosets", pipe.preflop.strategy.infosets.len())?;
    d.set_item("preflop_iterations", pipe.preflop.iterations_run)?;
    d.set_item("preflop_exploitability_bb", pipe.preflop.exploitability_bb)?;
    d.set_item(
        "postflop_status",
        pipe.postflop
            .as_ref()
            .map(|r| r.status.clone())
            .unwrap_or_else(|| "none".into()),
    )?;
    d.set_item(
        "postflop_infosets",
        pipe.postflop
            .as_ref()
            .map(|r| r.strategy.infosets.len())
            .unwrap_or(0),
    )?;
    d.set_item("notes", pipe.notes.clone())?;
    d.set_item(
        "induced_oop_mass",
        pipe.induced_oop.iter().sum::<f64>(),
    )?;
    d.set_item("induced_ip_mass", pipe.induced_ip.iter().sum::<f64>())?;
    // Full strategies for overnight training dumps
    d.set_item(
        "preflop_strategy",
        strategy_to_pydict(py, &pipe.preflop.strategy)?,
    )?;
    if let Some(ref post) = pipe.postflop {
        d.set_item("postflop_exploitability_bb", post.exploitability_bb)?;
        d.set_item("postflop_iterations", post.iterations_run)?;
        d.set_item("postflop_strategy", strategy_to_pydict(py, &post.strategy)?)?;
        d.set_item("postflop_notes", post.notes.clone())?;
    }
    Ok(d)
}

fn strategy_to_pydict<'py>(
    py: Python<'py>,
    strat: &super::types::Strategy,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("root_id", strat.root_id.clone())?;
    d.set_item("schema_version", strat.schema_version)?;
    let infosets = PyList::empty(py);
    for is in &strat.infosets {
        infosets.append(infoset_to_pydict(py, is)?)?;
    }
    d.set_item("infosets", infosets)?;
    Ok(d)
}

fn infoset_to_pydict<'py>(
    py: Python<'py>,
    is: &super::types::InfosetStrategy,
) -> PyResult<Bound<'py, PyDict>> {
    let idict = PyDict::new(py);
    idict.set_item("infoset_id", is.infoset_id.clone())?;
    idict.set_item("actions", is.actions.clone())?;
    idict.set_item("probs", is.probs.clone())?;
    if let Some(v) = is.visit_mass {
        idict.set_item("visit_mass", v)?;
    }
    if is.schema_version > 0 {
        idict.set_item("schema_version", is.schema_version)?;
        if let Some(v) = is.street {
            idict.set_item("street", v)?;
        }
        if let Some(v) = is.actor {
            idict.set_item("actor", v)?;
        }
        if let Some(v) = is.pot_chips {
            idict.set_item("pot_chips", v)?;
        }
        if let Some(v) = is.to_call_chips {
            idict.set_item("to_call_chips", v)?;
        }
        if let Some(v) = is.min_raise_chips {
            idict.set_item("min_raise_chips", v)?;
        }
        if let Some(v) = is.max_raise_chips {
            idict.set_item("max_raise_chips", v)?;
        }
        if let Some(ref xs) = is.stacks_chips {
            idict.set_item("stacks_chips", xs.clone())?;
        }
        if let Some(ref xs) = is.folded {
            idict.set_item("folded", xs.clone())?;
        }
        if let Some(ref xs) = is.board {
            let board_list = PyList::new(py, xs.iter().map(|&c| c as i32).collect::<Vec<_>>())?;
            idict.set_item("board", board_list)?;
        }
        if let Some(ref xs) = is.path {
            idict.set_item("path", xs.clone())?;
        }
        if let Some(ref k) = is.private_kind {
            idict.set_item("private_kind", k.clone())?;
        }
        if let Some(v) = is.private_id {
            idict.set_item("private_id", v)?;
        }
        match is.raw_combo {
            Some(v) => idict.set_item("raw_combo", v)?,
            None => idict.set_item("raw_combo", py.None())?,
        }
        match is.iso_id {
            Some(v) => idict.set_item("iso_id", v)?,
            None => idict.set_item("iso_id", py.None())?,
        }
    }
    Ok(idict)
}
