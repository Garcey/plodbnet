//! CFR public types (Phase 0).
//!
//! Chip unit convention matches the rest of the project:
//! - 1 bb = 10_000 engine chips at ClubGG 5/10($5)
//! - Solver configs often speak in **bb**; engine apply uses integer chips.

use super::infoset::{Infoset, InfosetDump, DUMP_SCHEMA_VERSION};
use super::CfrError;

/// Which street the subgame root starts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreetRoot {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3,
}

impl StreetRoot {
    pub fn from_u8(v: u8) -> Result<Self, CfrError> {
        match v {
            0 => Ok(StreetRoot::Preflop),
            1 => Ok(StreetRoot::Flop),
            2 => Ok(StreetRoot::Turn),
            3 => Ok(StreetRoot::River),
            _ => Err(CfrError::InvalidRoot(format!("street {v}"))),
        }
    }

    pub fn expected_board_len(self) -> usize {
        match self {
            StreetRoot::Preflop => 0,
            StreetRoot::Flop => 3,
            StreetRoot::Turn => 4,
            StreetRoot::River => 5,
        }
    }
}

/// Default raise sizes as pot-fraction per-mille (same spirit as NLH_ANCHOR_SPEC).
/// Phase 1 may use a coarser ladder for speed.
pub const DEFAULT_RAISE_SIZES_PM: [u32; 5] = [330, 500, 750, 1000, 1500];

/// One solve root: public parameters + optional board + range hooks.
#[derive(Debug, Clone)]
pub struct RootSpec {
    /// Always 2 for v1 (HU).
    pub num_seats: u8,
    pub street: StreetRoot,
    /// Pot at root, in big blinds.
    pub pot_bb: f64,
    /// Effective stack each player can still put in, in big blinds
    /// (postflop remaining; preflop ≈ starting stack after blinds — see notes).
    pub effective_stack_bb: f64,
    /// Big blind in engine chips (ClubGG: 10_000).
    pub bb_chips: u64,
    pub sb_chips: u64,
    pub ante_chips: u64,
    /// Board card indices 0..51; length must match street.
    pub board: Vec<u8>,
    /// Raise sizes as pot-fraction per-mille for this tree.
    pub raise_sizes_pm: Vec<u32>,
    /// Include all-in as a discrete action when stack > pot-fraction menu.
    pub allin_atom: bool,
    /// Optional range strings (Phase 1+); empty = uniform over combos not on board.
    pub range_ip: String,
    pub range_oop: String,
    /// Optional per-seat stacks in bb (multiway unequal). Empty = use effective_stack_bb for all.
    pub stacks_bb: Vec<f64>,
    /// Provenance / batch id.
    pub root_id: String,
}

impl RootSpec {
    pub fn preflop_hu(stack_bb: f64, bb_chips: u64, sb_chips: u64, ante_chips: u64) -> Self {
        // Starting pot ≈ sb+bb+2*ante in bb units for HU.
        let pot_bb = (sb_chips + bb_chips + 2 * ante_chips) as f64 / bb_chips as f64;
        Self {
            num_seats: 2,
            street: StreetRoot::Preflop,
            pot_bb,
            effective_stack_bb: stack_bb,
            bb_chips,
            sb_chips,
            ante_chips,
            board: vec![],
            raise_sizes_pm: DEFAULT_RAISE_SIZES_PM.to_vec(),
            allin_atom: true,
            range_ip: String::new(),
            range_oop: String::new(),
            stacks_bb: vec![],
            root_id: "preflop_hu".into(),
        }
    }

    pub fn postflop_hu(
        street: StreetRoot,
        pot_bb: f64,
        effective_stack_bb: f64,
        board: Vec<u8>,
        raise_sizes_pm: Vec<u32>,
    ) -> Self {
        Self {
            num_seats: 2,
            street,
            pot_bb,
            effective_stack_bb,
            bb_chips: 10_000,
            sb_chips: 5_000,
            ante_chips: 5_000,
            board,
            raise_sizes_pm,
            allin_atom: true,
            range_ip: String::new(),
            range_oop: String::new(),
            stacks_bb: vec![],
            root_id: format!("postflop_{street:?}"),
        }
    }

    pub fn validate(&self) -> Result<(), CfrError> {
        self.validate_for_solve()
    }

    /// Validate root; allows multiway (2..=6 seats).
    pub fn validate_for_solve(&self) -> Result<(), CfrError> {
        if self.num_seats < 2 || self.num_seats > 6 {
            return Err(CfrError::InvalidRoot(
                "num_seats must be 2..=6".into(),
            ));
        }
        if self.pot_bb <= 0.0 || !self.pot_bb.is_finite() {
            return Err(CfrError::InvalidRoot("pot_bb must be positive finite".into()));
        }
        if self.effective_stack_bb <= 0.0 || !self.effective_stack_bb.is_finite() {
            return Err(CfrError::InvalidRoot(
                "effective_stack_bb must be positive finite".into(),
            ));
        }
        if self.bb_chips == 0 {
            return Err(CfrError::InvalidRoot("bb_chips must be > 0".into()));
        }
        let need = self.street.expected_board_len();
        if self.board.len() != need {
            return Err(CfrError::InvalidRoot(format!(
                "board len {} != {} for {:?}",
                self.board.len(),
                need,
                self.street
            )));
        }
        for &c in &self.board {
            if c >= 52 {
                return Err(CfrError::InvalidRoot(format!("card index {c} out of 0..51")));
            }
        }
        let mut seen = [false; 52];
        for &c in &self.board {
            if seen[c as usize] {
                return Err(CfrError::InvalidRoot("duplicate board card".into()));
            }
            seen[c as usize] = true;
        }
        if self.raise_sizes_pm.is_empty() && !self.allin_atom {
            return Err(CfrError::InvalidRoot(
                "need at least one raise size or allin_atom=true (empty menu is push/fold only)".into(),
            ));
        }
        // stacks_bb: empty = use effective_stack_bb for all seats; otherwise must match num_seats
        let n = self.num_seats as usize;
        if !self.stacks_bb.is_empty() {
            if self.stacks_bb.len() != n {
                return Err(CfrError::InvalidRoot(format!(
                    "stacks_bb length {} != num_seats {n}",
                    self.stacks_bb.len()
                )));
            }
            for (i, &s) in self.stacks_bb.iter().enumerate() {
                if s <= 0.0 || !s.is_finite() {
                    return Err(CfrError::InvalidRoot(format!(
                        "stacks_bb[{i}] must be positive finite, got {s}"
                    )));
                }
            }
        }
        // allin-only menus are legal (jam/check / push-fold trees)
        Ok(())
    }

    /// Starting pot in chips for multiway preflop: `n*ante + sb + bb`.
    /// Multiway preflop rebuilds pot from blinds/antes and **ignores** `pot_bb`
    /// (which often double-counts or desyncs from seat blinds).
    pub fn multiway_preflop_pot_chips(&self) -> u64 {
        let n = self.num_seats as u64;
        n * self.ante_chips + self.sb_chips + self.bb_chips
    }
}

/// Solver hyperparameters.
#[derive(Debug, Clone)]
pub struct SolveConfig {
    /// Max CFR iterations. **0 = unlimited** (run until stop file / time budget).
    pub max_iterations: u32,
    pub target_exploitability_bb: f64,
    pub thread_num: u32,
    pub seed: u64,
    pub use_isomorphism: bool,
    /// Discounted CFR (γ) or linear — string tag for now.
    pub algorithm: String,
    /// Card abstraction id: "none" | "ochs" | custom later.
    pub card_abstraction: String,
    /// Wall-clock budget in seconds (0 = unlimited). When exceeded, solver
    /// exports the current average strategy and returns status ok with a note.
    pub time_budget_secs: f64,
    /// If non-empty, poll this path every ~poll iterations; if the file exists,
    /// stop early (kill-safe overnight). Written by the batch runner / user.
    pub stop_file: String,
    /// How often to poll time budget / stop / pause file (iterations). Min 1.
    pub poll_every: u32,
    /// If non-empty and this path exists, the solver **pauses** (spin-sleep)
    /// until the file is removed or stop_file appears. Does not discard state.
    pub pause_file: String,
    /// If non-empty, write a partial SolveReport JSON here every `poll_every`
    /// iterations so a live UI can stream strategy without waiting for finish.
    pub progress_file: String,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            target_exploitability_bb: 0.5,
            thread_num: 1,
            seed: 0,
            use_isomorphism: true,
            algorithm: "dcfr".into(),
            card_abstraction: "none".into(),
            time_budget_secs: 0.0,
            stop_file: String::new(),
            poll_every: 500,
            pause_file: String::new(),
            progress_file: String::new(),
        }
    }
}

impl SolveConfig {
    pub fn validate(&self) -> Result<(), CfrError> {
        // max_iterations == 0 means unlimited — allowed.
        if self.thread_num == 0 {
            return Err(CfrError::InvalidConfig("thread_num must be > 0".into()));
        }
        if self.target_exploitability_bb < 0.0 {
            return Err(CfrError::InvalidConfig(
                "target_exploitability_bb must be >= 0".into(),
            ));
        }
        if self.time_budget_secs < 0.0 {
            return Err(CfrError::InvalidConfig(
                "time_budget_secs must be >= 0".into(),
            ));
        }
        if self.poll_every == 0 {
            return Err(CfrError::InvalidConfig("poll_every must be > 0".into()));
        }
        Ok(())
    }

    /// Effective iteration ceiling (0 → ~4e9, effectively unlimited).
    pub fn iter_limit(&self) -> u32 {
        if self.max_iterations == 0 {
            u32::MAX
        } else {
            self.max_iterations
        }
    }

    /// Handle pause file (spin) then check stop/time. Call every iteration.
    /// Returns Some(reason) when the outer loop should break.
    ///
    /// Wall-clock `time_budget` and `stop_file` are checked **every**
    /// iteration (cheap). Pause + progress cadence still use `poll_every`.
    pub fn should_stop(&self, start: std::time::Instant, iteration: u32) -> Option<&'static str> {
        if self.time_budget_secs > 0.0
            && start.elapsed().as_secs_f64() >= self.time_budget_secs
        {
            return Some("time_budget");
        }
        if !self.stop_file.is_empty() && std::path::Path::new(&self.stop_file).exists() {
            return Some("stop_file");
        }
        let every = self.poll_every.max(1);
        // Pause is polled (spin-sleep); don't stat the pause file every iter.
        if iteration % every != 0 && iteration > 1 {
            return None;
        }
        // Pause: spin until resume (file removed) or stop / budget.
        if !self.pause_file.is_empty() {
            while std::path::Path::new(&self.pause_file).exists() {
                if !self.stop_file.is_empty() && std::path::Path::new(&self.stop_file).exists() {
                    return Some("stop_file");
                }
                if self.time_budget_secs > 0.0
                    && start.elapsed().as_secs_f64() >= self.time_budget_secs
                {
                    return Some("time_budget");
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
        None
    }

    /// Minimal JSON string escape for progress dumps.
    pub(crate) fn json_escape_str(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 8);
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    /// Write a lightweight progress JSON (status + iters + optional strategy).
    /// Best-effort: IO errors are ignored so the solve never fails on a bad path.
    pub fn write_progress(
        &self,
        iterations_run: u32,
        exploitability_bb: Option<f64>,
        n_infosets: usize,
        strategy: Option<&Strategy>,
        root_id: &str,
        paused: bool,
    ) {
        if self.progress_file.is_empty() {
            return;
        }
        use std::io::Write;
        let status = if paused { "paused" } else { "running" };
        let mut body = format!(
            "{{\n  \"status\": \"{status}\",\n  \"iterations_run\": {iterations_run},\n  \"num_infosets\": {n_infosets},\n  \"root_id\": \"{}\"",
            Self::json_escape_str(root_id)
        );
        if let Some(e) = exploitability_bb {
            if e.is_finite() {
                body.push_str(&format!(",\n  \"exploitability_bb\": {e}"));
            } else {
                body.push_str(",\n  \"exploitability_bb\": null");
            }
        } else {
            body.push_str(",\n  \"exploitability_bb\": null");
        }
        if let Some(strat) = strategy {
            body.push_str(",\n  \"strategy\": {\n    \"root_id\": \"");
            body.push_str(&Self::json_escape_str(&strat.root_id));
            body.push_str("\",\n    \"schema_version\": ");
            body.push_str(&format!("{}", strat.schema_version));
            body.push_str(",\n    \"infosets\": [");
            for (i, is) in strat.infosets.iter().enumerate() {
                if i > 0 {
                    body.push(',');
                }
                body.push_str("\n      ");
                is.write_json_object(&mut body);
            }
            body.push_str("\n    ]\n  }");
        }
        body.push_str("\n}\n");
        // Atomic-ish: write temp then rename. On Windows, rename fails if dest
        // exists — remove first so live progress actually updates every poll.
        let path = std::path::Path::new(&self.progress_file);
        let tmp = path.with_extension("progress.tmp");
        if let Ok(mut f) = std::fs::File::create(&tmp) {
            let _ = f.write_all(body.as_bytes());
            let _ = f.sync_all();
            let _ = std::fs::remove_file(path);
            if std::fs::rename(&tmp, path).is_err() {
                // Last resort: direct overwrite
                let _ = std::fs::copy(&tmp, path);
                let _ = std::fs::remove_file(&tmp);
            }
        }
    }
}

/// Average strategy at one information set, plus dump-schema v2 fields.
#[derive(Debug, Clone, Default)]
pub struct InfosetStrategy {
    pub infoset_id: String,
    /// Action labels (e.g. "FOLD", "CHECK_CALL", "RAISE_500").
    pub actions: Vec<String>,
    /// Average strategy probabilities (same length as actions).
    pub probs: Vec<f64>,
    /// 0 = legacy (id/actions/probs only); 2 = public+private dump.
    pub schema_version: u32,
    pub street: Option<u8>,
    pub actor: Option<u8>,
    pub pot_chips: Option<u64>,
    pub to_call_chips: Option<u64>,
    pub min_raise_chips: Option<u64>,
    pub max_raise_chips: Option<u64>,
    pub stacks_chips: Option<Vec<u64>>,
    pub folded: Option<Vec<bool>>,
    pub board: Option<Vec<u8>>,
    pub path: Option<Vec<String>>,
    pub private_kind: Option<String>,
    pub private_id: Option<u32>,
    pub raw_combo: Option<u32>,
    pub iso_id: Option<u32>,
    /// Sum of ``strategy_sum``. 0 ⇒ untouched (average strategy is the 1/n default).
    pub visit_mass: Option<f64>,
}

impl InfosetStrategy {
    pub fn from_node(infoset_id: String, node: &Infoset) -> Self {
        let mut s = Self {
            infoset_id,
            actions: node.actions.iter().map(|a| a.label()).collect(),
            probs: node.average_strategy(),
            visit_mass: Some(node.strategy_sum.iter().sum()),
            ..Default::default()
        };
        if let Some(ref d) = node.dump {
            s.fill_dump(d);
        }
        s
    }

    pub fn fill_dump(&mut self, d: &InfosetDump) {
        self.schema_version = DUMP_SCHEMA_VERSION;
        self.street = Some(d.street);
        self.actor = Some(d.actor);
        self.pot_chips = Some(d.pot_chips);
        self.to_call_chips = Some(d.to_call_chips);
        self.min_raise_chips = Some(d.min_raise_chips);
        self.max_raise_chips = Some(d.max_raise_chips);
        self.stacks_chips = Some(d.stacks_chips.clone());
        self.folded = Some(d.folded.clone());
        self.board = Some(d.board.clone());
        self.path = Some(d.path.clone());
        self.private_kind = Some(d.private_kind.clone());
        self.private_id = Some(d.private_id);
        self.raw_combo = d.raw_combo;
        self.iso_id = d.iso_id;
    }

    /// Append one JSON object (no trailing comma) for progress dumps.
    pub fn write_json_object(&self, body: &mut String) {
        body.push_str("{\"infoset_id\": \"");
        body.push_str(&SolveConfig::json_escape_str(&self.infoset_id));
        body.push_str("\", \"actions\": [");
        for (j, a) in self.actions.iter().enumerate() {
            if j > 0 {
                body.push(',');
            }
            body.push('"');
            body.push_str(&SolveConfig::json_escape_str(a));
            body.push('"');
        }
        body.push_str("], \"probs\": [");
        for (j, p) in self.probs.iter().enumerate() {
            if j > 0 {
                body.push(',');
            }
            if p.is_finite() {
                body.push_str(&format!("{p:.8}"));
            } else {
                body.push('0');
            }
        }
        body.push(']');
        if self.schema_version >= DUMP_SCHEMA_VERSION {
            body.push_str(&format!(", \"schema_version\": {}", self.schema_version));
            if let Some(v) = self.street {
                body.push_str(&format!(", \"street\": {v}"));
            }
            if let Some(v) = self.actor {
                body.push_str(&format!(", \"actor\": {v}"));
            }
            if let Some(v) = self.pot_chips {
                body.push_str(&format!(", \"pot_chips\": {v}"));
            }
            if let Some(v) = self.to_call_chips {
                body.push_str(&format!(", \"to_call_chips\": {v}"));
            }
            if let Some(v) = self.min_raise_chips {
                body.push_str(&format!(", \"min_raise_chips\": {v}"));
            }
            if let Some(v) = self.max_raise_chips {
                body.push_str(&format!(", \"max_raise_chips\": {v}"));
            }
            if let Some(ref xs) = self.stacks_chips {
                body.push_str(", \"stacks_chips\": [");
                for (i, x) in xs.iter().enumerate() {
                    if i > 0 {
                        body.push(',');
                    }
                    body.push_str(&format!("{x}"));
                }
                body.push(']');
            }
            if let Some(ref xs) = self.folded {
                body.push_str(", \"folded\": [");
                for (i, x) in xs.iter().enumerate() {
                    if i > 0 {
                        body.push(',');
                    }
                    body.push_str(if *x { "true" } else { "false" });
                }
                body.push(']');
            }
            if let Some(ref xs) = self.board {
                body.push_str(", \"board\": [");
                for (i, x) in xs.iter().enumerate() {
                    if i > 0 {
                        body.push(',');
                    }
                    body.push_str(&format!("{x}"));
                }
                body.push(']');
            }
            if let Some(ref xs) = self.path {
                body.push_str(", \"path\": [");
                for (i, x) in xs.iter().enumerate() {
                    if i > 0 {
                        body.push(',');
                    }
                    body.push('"');
                    body.push_str(&SolveConfig::json_escape_str(x));
                    body.push('"');
                }
                body.push(']');
            }
            if let Some(ref k) = self.private_kind {
                body.push_str(", \"private_kind\": \"");
                body.push_str(&SolveConfig::json_escape_str(k));
                body.push('"');
            }
            if let Some(v) = self.private_id {
                body.push_str(&format!(", \"private_id\": {v}"));
            }
            match self.raw_combo {
                Some(v) => body.push_str(&format!(", \"raw_combo\": {v}")),
                None => body.push_str(", \"raw_combo\": null"),
            }
            match self.iso_id {
                Some(v) => body.push_str(&format!(", \"iso_id\": {v}")),
                None => body.push_str(", \"iso_id\": null"),
            }
        }
        if let Some(v) = self.visit_mass {
            if v.is_finite() {
                body.push_str(&format!(", \"visit_mass\": {v:.8}"));
            } else {
                body.push_str(", \"visit_mass\": 0");
            }
        }
        body.push('}');
    }
}

/// Full strategy dump for a solve.
#[derive(Debug, Clone, Default)]
pub struct Strategy {
    pub root_id: String,
    pub schema_version: u32,
    pub infosets: Vec<InfosetStrategy>,
}

impl Strategy {
    pub fn new(root_id: impl Into<String>, infosets: Vec<InfosetStrategy>) -> Self {
        let schema_version = if infosets
            .iter()
            .any(|i| i.schema_version >= DUMP_SCHEMA_VERSION)
        {
            DUMP_SCHEMA_VERSION
        } else {
            0
        };
        Self {
            root_id: root_id.into(),
            schema_version,
            infosets,
        }
    }

    pub fn empty(root: &RootSpec) -> Self {
        Self {
            root_id: root.root_id.clone(),
            schema_version: 0,
            infosets: vec![],
        }
    }
}

/// Result of [`super::solve`].
#[derive(Debug, Clone)]
pub struct SolveReport {
    pub status: String,
    pub root: RootSpec,
    pub config: SolveConfig,
    pub strategy: Strategy,
    pub iterations_run: u32,
    pub exploitability_bb: Option<f64>,
    pub notes: Vec<String>,
}

#[cfg(test)]
mod stop_tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn time_budget_fires_without_waiting_for_poll() {
        let mut cfg = SolveConfig::default();
        cfg.poll_every = 2000;
        cfg.time_budget_secs = 0.05;
        let start = Instant::now() - Duration::from_secs(1);
        // iteration 3 is not a poll tick; must still honour wall clock
        assert_eq!(cfg.should_stop(start, 3), Some("time_budget"));
    }
}
