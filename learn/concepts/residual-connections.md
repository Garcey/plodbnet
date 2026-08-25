# Residual (skip) connections

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

In an ordinary deep network, each layer takes what the previous layer produced and replaces it. Whatever layer three outputs is all layer four ever sees. Every layer must re-express everything worth keeping, every single time — one clumsy layer in the middle and good information is gone for good.

A residual block changes the contract. Instead of replacing its input, it computes a correction and adds it on. The output is the input plus the block's adjustment. Nothing gets overwritten: a running summary flows through the network, and each block edits it rather than rewriting it.

It's the difference between how you actually build a read on an opponent and how a beginner does. You don't scrap your read every street and form a fresh one from the last thing you saw — you carry the running read forward and nudge it: a little more aggressive than I thought, discount that limp. Residual blocks give the network the same style. Keep the summary; apply the edit.

Going forward through the network, this keeps depth useful. A block with something to say makes a meaningful edit. A block with nothing to add can learn to add almost nothing, and the summary passes through unharmed. Depth stops being a chain where every link has to transform the signal perfectly, and becomes a series of optional refinements — which is why residual networks can be made much deeper without falling apart.

Going backward, it matters even more. Training works by sending a learning signal backward through the network: each layer is told how to adjust, based on how the final answer should have been different. Passed backward through an addition, that signal goes through untouched — the plus sign acts as a gradient highway, carrying the signal from the final output straight back to the earliest layers at full strength. Without skips, the signal has to squeeze through every intermediate layer's math, shrinking or distorting at each step, and the early layers — the ones reading the raw situation — may barely hear it at all. With skips, every layer hears clearly, so the whole stack learns together instead of just the last few floors.

That's the quiet trade at the heart of the design: each block gives up the freedom to rewrite everything, and in exchange the network gains depth that stays expressive and a learning signal that reaches all the way down.

**In this project:**

- The model's torso is one input layer followed by three residual blocks.
- The project's own code comment records that without residuals, the 2,048-wide, four-layer network fails to train at all.
