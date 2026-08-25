//! Preflop 169 hand classes and range induction.

/// Number of canonical preflop classes (13 pairs + 78 suited + 78 offsuit).
pub const NUM_PREFLOP_CLASSES: usize = 169;

/// Canonical preflop hand class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreflopHandClass {
    /// High rank 0..12 (2..A)
    pub hi: u8,
    /// Low rank 0..12
    pub lo: u8,
    /// true = suited (not pair)
    pub suited: bool,
}

impl PreflopHandClass {
    pub fn from_cards(c0: u8, c1: u8) -> Self {
        let r0 = c0 / 4;
        let r1 = c1 / 4;
        let s0 = c0 % 4;
        let s1 = c1 % 4;
        let (hi, lo) = if r0 >= r1 { (r0, r1) } else { (r1, r0) };
        let suited = r0 != r1 && s0 == s1;
        Self { hi, lo, suited }
    }

    /// Stable id 0..168
    pub fn id(self) -> u32 {
        if self.hi == self.lo {
            return self.hi as u32;
        }
        let mut idx = 0u32;
        for h in 0..13u8 {
            for l in 0..h {
                if h == self.hi && l == self.lo {
                    let base = 13 + if self.suited { 0 } else { 78 };
                    return base + idx;
                }
                idx += 1;
            }
        }
        0
    }

    pub fn from_id(id: u32) -> Self {
        if id < 13 {
            return Self {
                hi: id as u8,
                lo: id as u8,
                suited: false,
            };
        }
        let suited = id < 13 + 78;
        let mut rem = if suited { id - 13 } else { id - 13 - 78 };
        for h in 0..13u8 {
            for l in 0..h {
                if rem == 0 {
                    return Self {
                        hi: h,
                        lo: l,
                        suited,
                    };
                }
                rem -= 1;
            }
        }
        Self {
            hi: 12,
            lo: 11,
            suited: false,
        }
    }

    pub fn label(self) -> String {
        const R: [char; 13] = [
            '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
        ];
        if self.hi == self.lo {
            format!("{}{}", R[self.hi as usize], R[self.lo as usize])
        } else if self.suited {
            format!("{}{}s", R[self.hi as usize], R[self.lo as usize])
        } else {
            format!("{}{}o", R[self.hi as usize], R[self.lo as usize])
        }
    }
}

/// RNG used when dealing hole cards for MCCFR.
pub trait DealRng {
    fn gen_range(&mut self, n: usize) -> usize;
}

/// Deal two distinct HU hands as ordered (lo, hi) card pairs.
pub fn deal_holes_hu(rng: &mut impl DealRng) -> ((u8, u8), (u8, u8)) {
    let mut used = [false; 52];
    let mut draw = || loop {
        let c = rng.gen_range(52) as u8;
        if !used[c as usize] {
            used[c as usize] = true;
            return c;
        }
    };
    let a = draw();
    let b = draw();
    let c = draw();
    let d = draw();
    let h0 = if a < b { (a, b) } else { (b, a) };
    let h1 = if c < d { (c, d) } else { (d, c) };
    (h0, h1)
}

/// Bayesian range induction: after observing `action_idx`, multiply class
/// weights by σ(action|class) and renormalize.
pub fn induce_range(
    prior: &[f64],
    action_probs_by_class: &[Vec<f64>],
    action_idx: usize,
) -> Vec<f64> {
    assert_eq!(prior.len(), action_probs_by_class.len());
    let mut out = vec![0.0; prior.len()];
    let mut t = 0.0;
    for (i, &w) in prior.iter().enumerate() {
        let p = action_probs_by_class
            .get(i)
            .and_then(|v| v.get(action_idx))
            .copied()
            .unwrap_or(0.0);
        out[i] = w * p;
        t += out[i];
    }
    if t > 0.0 {
        for x in &mut out {
            *x /= t;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_ids_cover_169() {
        let mut seen = vec![false; 169];
        for c0 in 0..52u8 {
            for c1 in (c0 + 1)..52u8 {
                let id = PreflopHandClass::from_cards(c0, c1).id() as usize;
                assert!(id < 169, "id {id}");
                seen[id] = true;
            }
        }
        assert_eq!(seen.iter().filter(|&&x| x).count(), 169);
    }

    #[test]
    fn induce_zeros_impossible() {
        let prior = vec![0.5, 0.5];
        let probs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let post = induce_range(&prior, &probs, 0);
        assert!((post[0] - 1.0).abs() < 1e-9);
        assert!(post[1].abs() < 1e-9);
    }
}
