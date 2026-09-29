//! Entry points for the Criterion benchmarks in rust_engine/bench (TEST-026):
//! the batched packer, both row encoders and the compact row packer, through
//! the same code a rollout step runs. Compiled only with the `bench` cargo
//! feature, so it is never part of the Python module.

use super::*;

/// A batched engine holding `envs` hands in progress: the per-step observation
/// work (pack + encode) on a realistic mix of streets and seats to act.
pub struct Table {
    eng: PyBatchedEngine,
    live: Vec<usize>,
}

impl Table {
    /// `envs` seeded hands of `config`, each advanced by up to `actions`
    /// pseudo-random legal actions (so the rows spread over the streets), with
    /// an opp-outcome Monte-Carlo budget of `mc` samples (training uses 384).
    pub fn new(config: GameConfig, envs: usize, actions: usize, mc: usize, seed: u64) -> Table {
        let n = config.num_seats;
        let mut eng = PyBatchedEngine::with_config(config.clone(), envs, mc, OBS_REV_CURRENT);
        let mut x = seed | 1;
        let mut next = move || {
            // xorshift64*: any fixed sequence will do.
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for e in 0..envs {
            let mut g = GameState::new_hand(config.clone(), next(), e % n);
            for _ in 0..(next() as usize % (actions + 1)) {
                if g.is_terminal() {
                    break;
                }
                let mask = g.legal_action_mask();
                let legal: Vec<u8> = (0..NUM_ACTIONS as u8)
                    .filter(|&a| mask[a as usize])
                    .collect();
                let a = legal[next() as usize % legal.len()];
                g.apply(Action::from_index(a).expect("a legal action index"));
            }
            eng.states[e] = Some(g);
        }
        let live = (0..envs)
            .filter(|&e| !eng.states[e].as_ref().is_some_and(|g| g.is_terminal()))
            .collect();
        Table { eng, live }
    }

    /// Rows a pack / encode call produces (the hands still in progress).
    pub fn rows(&self) -> usize {
        self.live.len()
    }

    /// Forget the MC memo and the shared pair tables, so the next pack pays
    /// for them as the first decision on a street does.
    pub fn clear_caches(&self) {
        let n = self.eng.config.num_seats;
        let mut cache = self
            .eng
            .outcome_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for e in 0..self.eng.states.len() {
            cache.clear_env(e, n);
        }
        for t in &self.eng.board_tables {
            *t.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }

    /// The full-layout packer (MC outcome block, v3 features, history...);
    /// returns something derived from the result so nothing is optimised away.
    pub fn pack_full(&self) -> usize {
        let p = self.eng.pack_full(&self.live);
        p.cat_a.len() + p.legal.len()
    }

    /// The minimal-layout packer.
    pub fn pack_minimal(&self) -> usize {
        let (core, legal) = self.eng.pack_minimal_with_legal(&self.live);
        core.street.len() + legal.len()
    }

    /// Pack, then encode every row into `out` (`rows() x dim` floats).
    pub fn encode(&self, full: bool, out: &mut [f32]) {
        let layout = if full { Layout::Full } else { Layout::Minimal };
        let dim = layout.dim();
        let packed = self.eng.pack_for(layout, &self.live);
        let rows: Vec<&mut [f32]> = out[..self.live.len() * dim].chunks_mut(dim).collect();
        self.eng
            .encode_rows(&packed, Some(rows), None, |_| true)
            .expect("the encoder writes exact 0/1 flags");
    }
}

/// The observation width of the full (`true`) or minimal layout.
pub fn obs_dim(full: bool) -> usize {
    if full {
        Layout::Full.dim()
    } else {
        Layout::Minimal.dim()
    }
}

/// Compact storage of `rows` (row-major, `dim` wide): every row's `flag_cols`
/// as bits and its `real_cols` verbatim — the per-row work of `pack_obs_rows`.
pub fn pack_rows(
    rows: &[f32],
    dim: usize,
    flag_cols: &[usize],
    real_cols: &[usize],
    out_bits: &mut [u8],
    out_real: &mut [f32],
) {
    let (flag_runs, real_runs) = (column_runs(flag_cols), column_runs(real_cols));
    let (nb, nr) = (flag_cols.len().div_ceil(8), real_cols.len());
    for ((row, b), r) in rows
        .chunks(dim)
        .zip(out_bits.chunks_mut(nb))
        .zip(out_real.chunks_mut(nr))
    {
        pack_obs_row_runs(row, &flag_runs, &real_runs, flag_cols, real_cols, b, r)
            .expect("flag columns hold exact 0/1 values");
    }
}
