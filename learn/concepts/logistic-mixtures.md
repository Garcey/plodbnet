# Logistic curves and mixture distributions

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

A logistic bump is about the simplest opinion a network can draw: one smooth hump, described by two numbers. The center says where the hump sits; the width says how spread out it is. Narrow means "I want this size, specifically." Wide means "somewhere in this general neighborhood." That's the whole object — a where and a how-sure.

Bet sizes here aren't a smooth continuum, though; they're an ordered ladder of rungs. So the smooth bump gets discretized: lay the curve over the ladder and give each rung the slice of the bump sitting directly above it. A rung near the center collects a fat slice; rungs out on the fringes collect slivers. The two end rungs get a special job — they absorb the tails, all the probability stretching past the ladder in either direction. If the bump's center drifts beyond the top rung, that overflow lands on the top rung itself, which is a tidy way for "as big as allowed" to emerge naturally.

One bump has one built-in limitation: it can only want one thing. Its probability rises to a single peak and falls away. Strong poker is often not like that. Think of a polarized river strategy — sometimes a small blocking bet, sometimes a huge overbet, and almost never the middle. A single bump physically cannot represent that. Worse, if you force it to try, it plants its peak on the average of the two sizes: a medium bet that neither half of the strategy actually wants. Two good ideas, averaged into one bad one.

A mixture fixes this. Take several bumps, give each its own center and width, assign each a weight (the weights are themselves probabilities — they sum to 1), and add the weighted curves together. The combined shape can now hold two or three genuine peaks at once. The distribution can literally contain "small bet sometimes, overbet other times" as a single object, with the weights saying how often each mode gets used. Nothing is averaged; the modes coexist, and training can sharpen or fade each one independently.

**In this project:**

- v4's sizing head was a single discretized logistic over the 11 rungs — one center and one width per decision.
- v5/v6 upgraded it to a 3-component mixture, with each component guaranteed a fixed 3% minimum weight, so no bump can ever be squeezed entirely out of existence.
- The humps you can see on the study tab's bet-size chart are literally these components, drawn in chips-space.
