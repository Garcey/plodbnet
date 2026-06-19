---
name: Short all-in doesn't reopen raising (engine invariant for mixed stacks)
description: PL/NL rule — a short all-in that's less than a full min-raise over the standing bet does not give previously-acted callers the right to re-raise
type: feedback
originSessionId: 0f4de9c9-28ef-47dd-90d1-414a1df544da
---
Standard PL/NL poker rule: when a player shoves all-in for an amount
that is *less than a full raise* over the current `bet_to_call`, the
aggression does not reopen raising for players who already acted at
the previous bet_to_call. Those players may only call the incremental
amount or fold. Players not yet acted this street still get full
options (it's their first turn).

**Why:** user flagged this on 2026-04-20 as a prerequisite for the
deferred heterogeneous-stack training phase. Uniform stacks never
produce a short all-in (shoves from equal stacks are always full
raises), so the bug is invisible today, but mixed stacks will expose
it.

**How to apply:**
- Today's engine code at `rust_engine/src/engine.rs:433-438` updates
  `last_raise_size` unconditionally and `find_next_actor` at
  `engine.rs:797` reopens for any `street_commit < bet_to_call`.
  Both need fixing before mixed-stack training begins.
- Concrete fix spec is in the deferred "Heterogeneous per-seat
  stacks" section of the approved plan at
  `.claude/plans/i-m-starting-a-new-kind-dragonfly.md`. Summary:
  add `last_aggression_was_full_raise: bool` to `GameState`, only
  update `last_raise_size` on full raises, and suppress raise-style
  actions in `legal_action_mask` when `facing_bet &&
  acted_this_street[actor] && !last_aggression_was_full_raise`.
- Don't land the mixed-stacks engine plumbing without this fix —
  they go together.
- Existing uniform-stack training and tests are unaffected by the
  fix (short all-ins never occur with equal stacks), so the fix is
  safe to land proactively.
