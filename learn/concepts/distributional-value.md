# Distributional value heads (HL-Gauss)

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Ask the critic "what is this spot worth?" and the obvious design is a single number: "+4 big blinds." A distributional value head refuses to compress that far. Instead it answers with a probability spread over outcome buckets: roughly 30% this lands around −20, 50% near zero, 20% around +40. Poker players already think this way — a spot isn't one fixed result, it's a set of possible results with probabilities attached — and it turns out neural networks learn better when they're allowed to say so too.

## Why distributions learn more calmly

When the network must produce one number and the real outcomes are wild — tiny losses, huge wins, six-way stacks flying in — every training example yanks that single number around. The target whipsaws, and the learning updates whipsaw with it. Predicting a distribution turns the problem into something closer to sorting: nudge up the probability of the buckets where outcomes actually land, nudge down the rest. Each example makes a small, bounded adjustment to a few probabilities rather than a violent tug on one scalar. When the outcome scale is extreme, that difference in temperament matters a lot.

Nothing is lost, either. Whenever the rest of the training loop needs the familiar single value, it's recovered as the distribution's average: weight each bucket's center by its probability and add them up. One head, both views.

## The HL-Gauss refinement

There's a subtlety in how you tell the head it was right. The naive target is a spike: the outcome fell in bucket 27, so bucket 27 gets 100% and every other bucket — including bucket 26, a hair away — gets zero. That treats a near-miss as exactly as wrong as a miss by a mile. HL-Gauss smears the target slightly, spreading a little bell-curve (Gaussian) weight onto the neighboring buckets — the way a good teacher gives partial credit for an answer that's off by rounding rather than off by reasoning. The smearing tells the network that distance between buckets means something, which smooths learning and stops it from treating adjacent buckets as unrelated categories.

**In this project:**

- The critic's value head outputs 51 buckets spanning roughly ±1,500 big blinds on a compressed (symlog) grid, so tiny pots and six-way all-in monsters fit on one scale with fine resolution near zero.
- Because the buckets themselves bound what the head can predict, the old value-clipping machinery became unnecessary.
