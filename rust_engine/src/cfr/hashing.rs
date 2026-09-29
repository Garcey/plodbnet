//! Infoset history keys and the infoset-table hasher (TOOL-027).
//!
//! The solvers used to key a node by hashing the whole action history through
//! `AbstractAction::label()` Strings (one allocation per step, every node) with
//! std's `DefaultHasher` — whose algorithm Rust does not promise across releases
//! (CLAUDE.md flags it for the obs seed). Keys are now built from numeric action
//! codes with a fixed mixer (splitmix64's finalizer), so they are fast, allocation
//! free and identical on every toolchain:
//!
//! - [`action_step`] extends a parent's action hash by one action — the hot loops
//!   pass it down the recursion instead of re-hashing the history;
//! - [`history_key`] combines an action hash with the public board and street,
//!   exactly the inputs the old key covered (actions + board + board length [+
//!   street]), so the infoset PARTITION is unchanged and so are the strategies.
//!
//! [`InfosetMap`] is the `HashMap` the tables use: its keys are already well
//! mixed, so a multiply-fold hasher replaces SipHash.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use super::actions::AbstractAction;

/// splitmix64 finalizer: a bijective, avalanching 64-bit mixer.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Hash of the empty action history.
pub const EMPTY_ACTIONS: u64 = 0x243F_6A88_85A3_08D3; // pi

#[inline]
fn action_code(a: AbstractAction) -> u64 {
    match a {
        AbstractAction::Fold => 1,
        AbstractAction::CheckCall => 2,
        AbstractAction::AllIn => 3,
        AbstractAction::RaisePm(pm) => 0x1_0000_0000 | pm as u64,
    }
}

/// The action hash of `parent`'s history followed by `a`.
#[inline]
pub fn action_step(parent: u64, a: AbstractAction) -> u64 {
    mix64(parent ^ action_code(a).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// The action hash of a whole history (what [`action_step`] builds up).
pub fn actions_hash(actions: &[AbstractAction]) -> u64 {
    actions
        .iter()
        .fold(EMPTY_ACTIONS, |h, &a| action_step(h, a))
}

/// Infoset history key: the action hash + the public board dealt so far + the
/// street. Boards must be in the key so each runout gets its own infosets.
#[inline]
pub fn history_key(action_hash: u64, board: &[u8], street: u8) -> u64 {
    let mut h =
        mix64(action_hash ^ 0xA076_1D64_78BD_642F ^ ((board.len() as u64) << 56) ^ street as u64);
    for &c in board {
        h = mix64(h ^ (0x100 + c as u64));
    }
    h
}

/// `HashMap` hasher for keys that are already mixed (the history hash): fold the
/// written words with a multiply-rotate. Deterministic (no random state).
#[derive(Default, Clone, Copy)]
pub struct KeyHasher(u64);

impl Hasher for KeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(26) ^ i).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

/// The infoset table type.
pub type InfosetMap<K, V> = HashMap<K, V, BuildHasherDefault<KeyHasher>>;

#[cfg(test)]
mod tests {
    use super::*;

    fn all_histories(depth: usize) -> Vec<Vec<AbstractAction>> {
        let menu = [
            AbstractAction::Fold,
            AbstractAction::CheckCall,
            AbstractAction::AllIn,
            AbstractAction::RaisePm(330),
            AbstractAction::RaisePm(500),
            AbstractAction::RaisePm(1000),
        ];
        let mut out = vec![vec![]];
        let mut frontier = vec![vec![]];
        for _ in 0..depth {
            let mut next = Vec::new();
            for h in &frontier {
                for &a in &menu {
                    let mut x: Vec<AbstractAction> = h.clone();
                    x.push(a);
                    next.push(x);
                }
            }
            out.extend(next.iter().cloned());
            frontier = next;
        }
        out
    }

    /// The incremental hash equals the from-scratch one, and distinct histories
    /// (incl. different lengths / orders) never collide on a 56k-history sample.
    #[test]
    fn incremental_equals_scratch_and_is_collision_free() {
        let hs = all_histories(6);
        let mut seen = std::collections::HashSet::new();
        for h in &hs {
            let inc = h.iter().fold(EMPTY_ACTIONS, |acc, &a| action_step(acc, a));
            assert_eq!(inc, actions_hash(h));
            assert!(seen.insert(inc), "collision at {h:?}");
        }
        assert_eq!(seen.len(), hs.len());
    }

    /// Board and street are part of the key; the same action line on two
    /// runouts gets two keys.
    #[test]
    fn board_and_street_are_in_the_key() {
        let a = actions_hash(&[AbstractAction::CheckCall, AbstractAction::CheckCall]);
        let flop = history_key(a, &[0, 5, 10], 1);
        let turn = history_key(a, &[0, 5, 10, 15], 2);
        let other = history_key(a, &[0, 5, 11], 1);
        assert!(flop != turn && flop != other && turn != other);
        assert_eq!(flop, history_key(a, &[0, 5, 10], 1));
        assert_ne!(
            history_key(a, &[0, 5, 10], 1),
            history_key(a, &[5, 0, 10], 1)
        );
    }
}
