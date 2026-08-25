# Bootstrapping, TD errors, and GAE

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Every training update starts with a grading problem: for each decision the bot made, how good was it? There are two honest ways to answer, and they fail in opposite directions.

The first is to wait for the final result of the hand and grade the decision by the money. This is truthful on average — no assumptions, just outcomes. It is also drowning in luck. A perfect flop check-raise can still lose to a two-outer, and pure outcome-grading will dutifully mark it down.

The second is to trust your own estimate. The network carries a value head — its standing guess of what any situation is worth. So instead of waiting for the river, grade a decision by where it landed you: the chips picked up along the way, plus the estimated value of the new situation, compared against the estimated value of the one you left. This is bootstrapping — the learner pulling itself up by its own estimates. The grade is far less noisy, but it is only as good as the estimator, and early in training the value head is mostly wrong.

That step-by-step comparison has a name: the TD error (temporal-difference error). Reward received, plus the estimate of the next state, minus the estimate of this state. In plain English: did this step beat expectations? Think of a GPS on a road trip. It re-estimates your arrival time at every intersection, so a wrong turn is graded the moment you make it — the ETA jumps four minutes — not when you finally pull into the driveway. Each ETA jump is a TD error.

## The lambda dial

GAE (generalized advantage estimation) refuses to pick a side. It walks the rest of the trajectory summing TD errors, each decayed a bit more than the last, with a dial called lambda setting the decay. Lambda at zero is pure one-step trust: believe the estimator completely. Lambda at one reassembles the full actual outcome: believe only what happened. Anything in between blends the two — mostly low-noise estimates, with real outcomes still allowed to speak.

What comes out is the advantage: how much better this action was than par for the situation. That number, not the raw result, is what the policy learns from.

**In this project:**

- Gamma (the time-discount) is 1.0 — no discounting, because hands are short — and lambda is 0.95, leaning on the value head while keeping real outcomes in the mix.
- The value head's targets (the "returns") are always built the GAE way, even in experimental runs where the VRPO estimator supplies the advantages instead.
- This is the framing the whole course runs on: advantages built this way are the anti-results-oriented-thinking machine. They grade the decision, not the runout.
