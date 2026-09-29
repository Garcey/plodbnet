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
    /// (TOOL-007) Lazy DCFR: the end-of-iteration discounts of iterations
    /// `1..=disc_mark` have been applied to `regret` / `strategy_sum`; later
    /// ones are applied by [`LazyDiscount::sync`] the next time the infoset is
    /// updated (or exported).
    pub disc_mark: u32,
    /// (TOOL-007) The positive-regret cumulative factor `P(disc_mark)` at the
    /// last sync (see [`LazyDiscount`]).
    pub pos_seen: f64,
    /// (TOOL-028) How many times the average strategy was accumulated here —
    /// a plain visit count, unlike `visit_mass` (reach-weighted and decayed by
    /// DCFR, so its scale depended on the iteration count and the depth).
    pub visits: u32,
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
            disc_mark: 0,
            pos_seen: 1.0,
            visits: 0,
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
            s.fill(u);
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
        self.apply_discount(iteration, 1.5, 0.0, 2.0);
    }

    /// General DCFR(α, β, γ) discount; Linear CFR is α=β=γ=1.
    /// (review 2026-09-20, latent: `algorithm="linear"` used to run the DCFR
    /// parameters under a "linear" label.)
    pub fn apply_discount(&mut self, iteration: u32, alpha: f64, beta: f64, gamma: f64) {
        let (pos, neg, strat) = Self::discount_scales(iteration, alpha, beta, gamma);
        self.apply_scales(pos, neg, strat);
    }

    /// `(positive-regret, negative-regret, strategy-sum)` multipliers for a
    /// 1-based `iteration`. They depend only on the iteration, so table-wide
    /// discounting computes them ONCE instead of 3 `powf` per infoset.
    pub fn discount_scales(iteration: u32, alpha: f64, beta: f64, gamma: f64) -> (f64, f64, f64) {
        let t = iteration as f64;
        (
            t.powf(alpha) / (t.powf(alpha) + 1.0),
            t.powf(beta) / (t.powf(beta) + 1.0),
            (t / (t + 1.0)).powf(gamma),
        )
    }

    /// Current regret-matching strategy written into `out` (no allocation).
    pub fn current_strategy_into(&self, out: &mut [f64]) {
        let n = self.actions.len();
        let mut sum = 0.0;
        for i in 0..n {
            let r = self.regret[i].max(0.0);
            out[i] = r;
            sum += r;
        }
        if sum <= 0.0 {
            let u = 1.0 / n as f64;
            out[..n].iter_mut().for_each(|x| *x = u);
        } else {
            out[..n].iter_mut().for_each(|x| *x /= sum);
        }
    }

    pub fn apply_scales(&mut self, pos_scale: f64, neg_scale: f64, strat_scale: f64) {
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

/// (TOOL-007) Lazy DCFR discounting.
///
/// DCFR multiplies EVERY stored infoset's positive regrets by `a(t)`, negative
/// regrets by `b(t)` and strategy sums by `g(t)` at the end of every iteration
/// `t`. Chance-sampled DCFR touches a few hundred infosets per iteration, so on
/// a table of a million infosets that table-wide pass dominated the solve. The
/// factors only multiply, never mix values, and never change a value's sign, so
/// an infoset can instead be brought up to date when it is next UPDATED (or
/// exported) by the product of the factors it missed:
///
/// - strategy sums: `prod_{k=m+1..n} (k/(k+1))^g = ((m+1)/(n+1))^g` (closed form);
/// - negative regrets: `b = 1/2` for DCFR (`beta = 0`) → `2^-(n-m)` exactly; for
///   Linear CFR (`beta = 1`) the closed form `(m+1)/(n+1)`;
/// - positive regrets: `a(k) = k^alpha/(k^alpha+1)` has no closed form, so the
///   cumulative product `P(t) = prod_{k<=t} a(k)` is kept globally and each
///   infoset remembers `P(m)` at its last sync: the missed factor is `P(n)/P(m)`
///   (`P` converges to a positive constant for `alpha > 1` and is `1/(t+1)` for
///   Linear CFR — it never underflows).
///
/// Reading a strategy never needs a sync: regret matching normalizes the
/// positive regrets (all scaled by one factor) and ignores the negative ones, and
/// the average strategy normalizes the sums. The results equal the eager pass up
/// to floating-point rounding (pinned by `dcfr::tests::lazy_discount_matches_eager`).
#[derive(Debug, Clone)]
pub struct LazyDiscount {
    alpha: f64,
    beta: f64,
    gamma: f64,
    /// Iterations whose end-of-iteration discount has been issued.
    issued: u32,
    /// `P(issued)`.
    pos_cum: f64,
}

impl LazyDiscount {
    pub fn new(alpha: f64, beta: f64, gamma: f64) -> Self {
        Self {
            alpha,
            beta,
            gamma,
            issued: 0,
            pos_cum: 1.0,
        }
    }

    /// Record that iteration `t` (== issued + 1) ended: its discount is owed
    /// by every infoset from now on.
    pub fn end_iteration(&mut self, t: u32) {
        debug_assert_eq!(t, self.issued + 1);
        let (pos, _, _) = Infoset::discount_scales(t, self.alpha, self.beta, self.gamma);
        self.pos_cum *= pos;
        self.issued = t;
    }

    pub fn issued(&self) -> u32 {
        self.issued
    }

    /// A brand-new infoset owes nothing: mark it as up to date.
    pub fn stamp(&self, node: &mut Infoset) {
        node.disc_mark = self.issued;
        node.pos_seen = self.pos_cum;
    }

    /// Apply every discount `node` missed since its last sync.
    pub fn sync(&self, node: &mut Infoset) {
        let (m, n) = (node.disc_mark, self.issued);
        if m >= n {
            return;
        }
        let ratio = (m as f64 + 1.0) / (n as f64 + 1.0);
        let strat = ratio.powf(self.gamma);
        let neg = if self.beta == 0.0 {
            // b(k) = 1/(1+1) = 1/2 exactly for every k.
            0.5f64.powi((n - m).min(i32::MAX as u32) as i32)
        } else if self.beta == 1.0 {
            ratio
        } else {
            (m + 1..=n)
                .map(|k| Infoset::discount_scales(k, self.alpha, self.beta, self.gamma).1)
                .product()
        };
        let pos = self.pos_cum / node.pos_seen;
        node.apply_scales(pos, neg, strat);
        node.disc_mark = n;
        node.pos_seen = self.pos_cum;
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
