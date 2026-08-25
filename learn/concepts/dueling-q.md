# Dueling Q heads (value plus a correction per action)

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

The critic's usual output answers "what is this spot worth if play continues normally?" A Q value answers a sharper question: what is the expected result of taking one specific action from this spot — this fold, this call, this raise? Having that number per action is the raw material for a stronger learning signal, because you can compare actions directly instead of inferring their quality from noisy played-out hands.

A dueling head is a particular way of building those Q values. Rather than learning a separate, independent prediction for every action from scratch, it learns the spot's overall value once, plus a small per-action correction: Q equals the state's value plus "how much this action helps or hurts relative to that baseline." Most of what makes a spot good or bad — stack depth, board texture, position — is shared across all the actions, so forcing each action's column to relearn it would be wasteful. The dueling split keeps the shared part in one place and reserves the per-action columns for what actually differs between actions.

## Zero start, zero risk

Here's the elegant part. Initialize every per-action correction at exactly zero, and the head says: each action is worth exactly what the spot is worth. Perfectly neutral — no invented opinions, no random noise masquerading as knowledge. It's like installing a new mixing board in a studio with every slider set flat: the day it arrives, the band sounds exactly as it did before, and each adjustment only gets dialed in once it's earned. That's what makes a dueling head a safe scaffold for upgrading the learning signal: you can bolt it onto a proven training run, and until the corrections learn something real, the new machinery behaves identically to the old one. There is no scary cliff-edge moment where you swap estimators and hope.

**In this project:**

- The critic's Q head is built as V plus a zero-initialized per-action correction, so the new (VRPO) advantage estimator starts out mathematically identical to the proven old one.
- A 2026-07 audit found the per-bet-size correction columns starving for data — each saw only about 3% of training rows — so the head was pooled from 13 columns down to 3: fold, check-call, raise.
- The fold column has a known true value of exactly zero (folding costs nothing further), which the project uses as free perfect training labels (weighted 15× after a 2026-07-11 audit found the anchor being drowned out) and as a standing calibration canary — the `qF=` number on every training log line.
- How the columns actually get trained — the homework-grading mechanism, the moving-target chase, and why the lean happened — is its own explainer: [how the Q head learns](q-head-learning.md).
