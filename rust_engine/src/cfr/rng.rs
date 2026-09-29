//! The solvers' one random-number generator (TOOL-046).
//!
//! The same xorshift64* (misnamed `Lcg`) used to be copied into dcfr.rs,
//! mccfr.rs and card_abs.rs, each seeded with `seed + constant`: the parallel
//! deal streams of one iteration got ADJACENT seeds (xorshift states that differ
//! in one low bit start out correlated), and exactly one seed value mapped to the
//! all-zero state, which xorshift never leaves (a constant stream).
//!
//! [`CfrRng`] seeds through splitmix64 (a bijection with full avalanche, so
//! nearby seeds give unrelated states) and never starts at zero; independent
//! streams for parallel work come from [`CfrRng::stream`].

use super::hashing::mix64;

/// xorshift64* seeded through splitmix64.
#[derive(Debug, Clone)]
pub struct CfrRng {
    state: u64,
}

impl CfrRng {
    pub fn new(seed: u64) -> Self {
        let s = mix64(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
        // xorshift's one fixed point; mix64 is a bijection, so exactly one
        // seed lands here — move it anywhere else.
        Self {
            state: if s == 0 { 0x6A09_E667_F3BC_C909 } else { s },
        }
    }

    /// Stream `index` of `seed`: seeds that are unrelated for every index.
    pub fn stream(seed: u64, index: u64) -> Self {
        Self::new(mix64(seed ^ 0xD1B5_4A32_D192_ED03).wrapping_add(mix64(index.wrapping_add(1))))
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Uniform in 0..n (multiply-high; n == 0 gives 0).
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

impl super::preflop::DealRng for CfrRng {
    fn gen_range(&mut self, n: usize) -> usize {
        self.below(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No seed gives the stuck all-zero stream, and nearby seeds / stream
    /// indices start far apart.
    #[test]
    fn no_seed_is_stuck_and_neighbours_are_unrelated() {
        // The seed that mix64 would map to 0 (mix64 is a bijection).
        for seed in [
            0u64,
            1,
            2,
            u64::MAX,
            0x9E37_79B9_7F4A_7C15u64.wrapping_neg(),
        ] {
            let mut r = CfrRng::new(seed);
            let draws: Vec<u64> = (0..4).map(|_| r.next_u64()).collect();
            assert!(draws.iter().all(|&x| x != 0), "seed {seed}: {draws:?}");
            assert!(draws.windows(2).all(|w| w[0] != w[1]));
        }
        let (a, b) = (CfrRng::new(41).next_u64(), CfrRng::new(42).next_u64());
        assert!(
            (a ^ b).count_ones() > 16,
            "adjacent seeds correlated: {a:x} {b:x}"
        );
        let (s0, s1) = (
            CfrRng::stream(7, 0).next_u64(),
            CfrRng::stream(7, 1).next_u64(),
        );
        assert!((s0 ^ s1).count_ones() > 16);
    }

    #[test]
    fn below_is_in_range_and_roughly_uniform() {
        let mut r = CfrRng::new(3);
        let mut hist = [0u32; 52];
        for _ in 0..52_000 {
            let x = r.below(52);
            hist[x] += 1;
        }
        assert!(hist.iter().all(|&c| (800..1200).contains(&c)), "{hist:?}");
        assert_eq!(CfrRng::new(1).below(0), 0);
        let f = CfrRng::new(9).next_f64();
        assert!((0.0..1.0).contains(&f));
    }
}
