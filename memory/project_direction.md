---
name: Project direction — heterogeneous stacks + continuous bet sizing
description: Plodbbot is trending toward per-seat heterogeneous stacks and a continuous-sizing policy head; design decisions should stay forward-compatible
type: project
originSessionId: 0f4de9c9-28ef-47dd-90d1-414a1df544da
---
Plodbbot currently hard-codes 6-max / 20bb stacks / 3bb ante and a
five-slot pct bet enum. Two directional goals beyond that:

1. **Different player counts + heterogeneous per-seat stacks.** Not just
   uniform-stack 2–6 max, but each seat at its own stack depth (e.g. seat 0
   at 100bb, seat 3 at 25bb). Stated 2026-04-20. Requires the deferred
   engine/GameConfig vec<u64> plumbing + the short-all-in re-raise fix
   (both already in the plan file; see feedback_short_allin_rule).
2. **Continuous bet sizing.** Stated 2026-04-21 after the 40bb run: the
   eventual policy should output any sizing, not pick from the five
   discrete pct slots. Concrete design is open — candidates: separate
   continuous head (mean + stddev over [min_raise, stack], truncated
   normal), pointer/beta parametrization, or a hybrid where discrete
   slots remain for training stability but a continuous head augments
   them at inference. Masking + PL-floor clamping must still be honored.

**Why:** study-tool value grows when users can analyze bomb-pot spots at
varied seat counts, effective stack depths, and realistic sizings.
Discrete pcts miss the spots that matter most (e.g. a 33% bet in a
texture that wants smaller than 50%).

**How to apply when making design decisions today:**
- Action space / masking / encoding: prefer forward-compatible choices
  even if some option is redundant at the current default. Keep the
  full pct enum `B10 / B25 / B50 / B75 / B100`; rely on engine dup-
  masking to drop sizes that collapse to AllIn at shallow stacks —
  they'll be distinct deeper. (Decided 2026-04-20.)
- UI: don't invest heavily in per-stack-depth selectors when the real
  target is per-seat heterogeneous stacks. Stated 2026-04-21: user
  declined a 20/40/60bb UI switcher for the Phase A→B handoff because
  the switcher would be rebuilt once per-seat stacks land.
- Don't lock the action space into discrete pcts at the Rust-level
  type system; keep sizing representation as a chip-amount `u64` with
  a thin enum wrapper for today's training, so a continuous head can
  slot in without an engine rewrite.
