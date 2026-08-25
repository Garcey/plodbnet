# Cross-entropy loss

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Some network outputs aren't numbers meant to be close to a target — they're beliefs. When a network outputs a probability distribution ("30% this outcome, 50% that one, 20% the next"), grading it with squared error misses the point: squared error measures distances, and beliefs aren't graded on distance. They're graded on how much probability you placed on the thing that actually happened.

That's cross-entropy. Once the outcome is known, the loss looks up a single number: the probability the network had placed on that outcome. Lots of probability there means a small loss; very little means a large one. And the punishment is not linear. Sliding from 50% down to 5% on the true outcome hurts, but sliding from 5% toward zero hurts explosively. Confident wrongness is the cardinal sin. Calibrated confidence — being sure exactly when you're entitled to be — is what scores best over time. Hedging everything equally is safe but mediocre, and it loses to anyone whose confidence actually tracks reality.

Poker players already live under this loss. Put a river jam on "80% value, 20% bluff," and showdown grades you on the probability you gave the actual holding. Assign 20% to the bluff he shows and you take a real but survivable hit; assign 2% and it's brutal. The player whose certainty matches the truth beats both the always-hedger and the confidently wrong — cross-entropy is that grading, made exact.

## Soft labels

Usually "the thing that actually happened" is a single option, and the target is a spike: all probability on one answer. But the target can itself be a spread. When the possible outcomes are buckets along a scale, landing in bucket 23 is nearly the same event as landing in 22 or 24 — the bucket edges are arbitrary. Soft labels spread the target's probability across the true bucket's neighbors, asking the network to put its mass in the right neighborhood rather than on one exact slot. Training gets smoother, and near-misses stop being punished as though they were absurd.

**In this project:**

- The critic's value head predicts a probability distribution over 51 outcome buckets and trains with cross-entropy against soft labels — a small Gaussian smear around the true outcome rather than a single spike.
- The v= number on the training log line is this cross-entropy, measured in its own units — which is why it lives near 3.0 instead of looking like a chip amount.
