//! Set-up shared by the engine benchmarks (benches/kernels.rs, TEST-026): the
//! table shapes and game states the kernels are timed on. Everything is seeded,
//! so two runs time exactly the same work.

use plo5bp_engine::actions::{Action, NUM_ACTIONS};
use plo5bp_engine::cards::{Card, Deck};
use plo5bp_engine::state::{GameConfig, GameState, Street};

/// Run the engine's rayon work on ONE thread: the numbers are per-core costs,
/// comparable between runs and machines (a batch's parallel speed-up is what
/// scripts/bench_subrollout.py measures). The first call wins.
pub fn one_thread() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();
}

/// xorshift64* — a fixed pseudo-random sequence for the set-up.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// The training table: 6-max PLO5 double-board bomb pot, 100 bb deep
/// (bb = 10,000 chips) with a 3 bb ante.
pub fn plo5_6max() -> GameConfig {
    GameConfig::new_uniform(6, 1_000_000, 30_000, 10_000)
}

/// `k` distinct cards from a seeded shuffle.
pub fn cards(seed: u64, k: usize) -> Vec<Card> {
    Deck::new_shuffled(seed).deal(k)
}

/// Play one pseudo-random legal action.
pub fn random_action(g: &mut GameState, rng: &mut Rng) {
    let mask = g.legal_action_mask();
    let legal: Vec<u8> = (0..NUM_ACTIONS as u8)
        .filter(|&a| mask[a as usize])
        .collect();
    let a = legal[rng.below(legal.len())];
    g.apply(Action::from_index(a).expect("a legal action index"));
}

/// `n` hands of `config` in progress: each played up to `actions` random
/// legal actions in (the street mix a rollout sees), re-dealt when that ends
/// the hand, so every state has a seat to act.
pub fn states_in_progress(
    config: &GameConfig,
    n: usize,
    actions: usize,
    seed: u64,
) -> Vec<GameState> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut g =
            GameState::new_hand(config.clone(), rng.next_u64(), out.len() % config.num_seats);
        for _ in 0..rng.below(actions + 1) {
            if g.is_terminal() {
                break;
            }
            random_action(&mut g, &mut rng);
        }
        if !g.is_terminal() {
            out.push(g);
        }
    }
    out
}

/// `n` hands checked down to `street` (every seat still in).
pub fn states_on(config: &GameConfig, n: usize, street: Street, seed: u64) -> Vec<GameState> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let mut g = GameState::new_hand(config.clone(), rng.next_u64(), i % config.num_seats);
            while g.street.index() < street.index() && !g.is_terminal() {
                g.apply(Action::CheckCall);
            }
            g
        })
        .collect()
}

/// `n` three-handed hands all-in on the flop (short stacks: 5 bb behind a 3 bb
/// ante), so settling one samples whole turn + river runouts on both boards.
pub fn flop_all_ins(n: usize, seed: u64) -> Vec<GameState> {
    let config = GameConfig::new_uniform(3, 50_000, 30_000, 10_000);
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let mut g = GameState::new_hand(config.clone(), rng.next_u64(), i % 3);
            g.apply(Action::AllIn);
            while !g.is_terminal() {
                g.apply(Action::CheckCall);
            }
            g
        })
        .collect()
}
