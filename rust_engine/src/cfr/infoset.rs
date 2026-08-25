//! Tabular infoset regrets and average strategy (DCFR accumulators).

use super::actions::AbstractAction;
use super::public_state::PublicState;

/// Strategy-dump schema version. v2 carries public + private fields so
/// Python can rebuild live-matching obs without guessing `to_call`.
pub const DUMP_SCHEMA_VERSION: u32 = 2;

pub const PRIV_COMBO: &str = "combo";
pub const PRIV_CLASS: &str = "class";
pub const PRIV_OCHS_BUCKET: &str = "ochs_bucket";

/// Public + private snapshot stored on first visit to an infoset.
#[derive(Debug, Clone)]
pub struct InfosetDump {
    pub street: u8,
    pub actor: u8,
    pub pot_chips: u64,
    pub to_call_chips: u64,
    pub min_raise_chips: u64,
    pub max_raise_chips: u64,
    pub stacks_chips: Vec<u64>,
    pub folded: Vec<bool>,
    pub board: Vec<u8>,
    /// Full abstract-action labels in visit order (not a u64 hash).
    pub path: Vec<String>,
    pub private_kind: String,
    pub private_id: u32,
    pub raw_combo: Option<u32>,
    pub iso_id: Option<u32>,
}

impl InfosetDump {
    pub fn from_state(
        state: &PublicState,
        history: &[AbstractAction],
        private_kind: &str,
        private_id: u32,
        raw_combo: Option<u32>,
        iso_id: Option<u32>,
    ) -> Self {
        let n = state.num_seats as usize;
        Self {
            street: state.street,
            actor: state.actor.unwrap_or(0),
            pot_chips: state.pot,
            to_call_chips: state.to_call_chips(),
            min_raise_chips: state.min_raise_chips(),
            max_raise_chips: state.max_raise_chips(),
            stacks_chips: state.stacks[..n].to_vec(),
            folded: state.folded[..n].to_vec(),
            board: state.board[..state.board_len as usize].to_vec(),
            path: history.iter().map(|a| a.label()).collect(),
            private_kind: private_kind.to_string(),
            private_id,
            raw_combo,
            iso_id,
        }
    }
}

/// One information set: regrets + cumulative strategy for a fixed action list.
#[derive(Debug, Clone)]
pub struct Infoset {
    pub actions: Vec<AbstractAction>,
    pub regret: Vec<f64>,
    pub strategy_sum: Vec<f64>,
    /// DCFR positive-regret accumulator helpers (optional).
    pub iter_touched: u32,
    /// Public/private snapshot for strategy dumps (set on first insert).
    pub dump: Option<InfosetDump>,
}

impl Infoset {
    pub fn new(actions: Vec<AbstractAction>) -> Self {
        let n = actions.len();
        Self {
            actions,
            regret: vec![0.0; n],
            strategy_sum: vec![0.0; n],
            iter_touched: 0,
            dump: None,
        }
    }

    pub fn new_with_dump(actions: Vec<AbstractAction>, dump: InfosetDump) -> Self {
        let mut n = Self::new(actions);
        n.dump = Some(dump);
        n
    }

    /// Current regret-matching strategy (non-negative regrets normalized).
    pub fn current_strategy(&self) -> Vec<f64> {
        let n = self.actions.len();
        if n == 0 {
            return vec![];
        }
        let mut s = vec![0.0; n];
        let mut sum = 0.0;
        for i in 0..n {
            let r = self.regret[i].max(0.0);
            s[i] = r;
            sum += r;
        }
        if sum <= 0.0 {
            let u = 1.0 / n as f64;
            for x in &mut s {
                *x = u;
            }
        } else {
            for x in &mut s {
                *x /= sum;
            }
        }
        s
    }

    /// Average strategy from cumulative strategy_sum.
    pub fn average_strategy(&self) -> Vec<f64> {
        let n = self.actions.len();
        if n == 0 {
            return vec![];
        }
        let sum: f64 = self.strategy_sum.iter().sum();
        if sum <= 0.0 {
            return vec![1.0 / n as f64; n];
        }
        self.strategy_sum.iter().map(|x| x / sum).collect()
    }

    /// DCFR discount on positive regrets / strategy sum (Brown & Sandholm).
    /// α=1.5, β=0, γ=2 common defaults; applied once per iteration end.
    pub fn apply_dcfr_discount(&mut self, iteration: u32) {
        // iteration is 1-based
        let t = iteration as f64;
        let alpha = 1.5;
        let beta = 0.0;
        let gamma = 2.0;
        let pos_scale = t.powf(alpha) / (t.powf(alpha) + 1.0);
        let neg_scale = t.powf(beta) / (t.powf(beta) + 1.0);
        let strat_scale = (t / (t + 1.0)).powf(gamma);
        for r in &mut self.regret {
            if *r > 0.0 {
                *r *= pos_scale;
            } else {
                *r *= neg_scale;
            }
        }
        for s in &mut self.strategy_sum {
            *s *= strat_scale;
        }
    }
}

/// Key for infoset map: player + public history id + private view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InfosetKey {
    pub player: u8,
    pub history: u64,
    pub private: u32,
}

impl InfosetKey {
    pub fn new(player: u8, history: u64, private: u32) -> Self {
        Self {
            player,
            history,
            private,
        }
    }
}
