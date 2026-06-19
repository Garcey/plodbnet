---
name: Network shouldn't recommend negative-EV actions in normal cases
description: Heuristic check on recommendation+value outputs; only specific scenarios legitimately produce negative EV
type: feedback
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
A trained network should not recommend an action whose EV is
negative, except in scenarios where every legal action has
negative EV.

**Why:** This is the user's correctness sanity check on the
recommendation surface. If the recommended-action's value comes
back negative when a $0-EV alternative existed (e.g., recommending
a check whose displayed EV is far below zero), something is broken
— either the value head, the input encoding, the unit conversion,
or the recommendation pipeline.

**Legitimate negative-EV scenarios** (don't flag these as bugs):
- Calling a bet with a hand that has positive equity but bad
  pot odds — EV(call) negative but EV(fold) = $0 already, so
  call may still be the least-bad if facing forced action.
- Bluffing/betting hands where opponent calls often enough that
  fold-equity gain is outweighed by losses to calls. Sometimes
  the best -EV bluff still is the best play in a multi-action
  comparison.

**How to apply:** When the UI shows a recommendation with
negative `value_bb` and a check/fold path was available with V≈0,
treat that as a diagnosis trigger. Trace input encoding, training
reward semantics, and unit conversion before assuming the model is
just under-trained. Don't be content with a "model extrapolated
badly" explanation if a structural cause (OOD obs features, unit
mismatch) is plausible.
