# Gradient checkpointing

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Training a network is a two-pass affair. The forward pass pushes a batch of situations through the layers and produces predictions. The backward pass then works out, layer by layer in reverse, how every weight should nudge to make those predictions better. The catch: the backward pass needs to see what happened during the forward pass. To know how a weight should change, you need the exact values that flowed through it.

The standard approach is to simply remember everything — every layer's intermediate output, for every situation in the batch, held in GPU memory until the backward pass has consumed it. For a small batch this is invisible. For a giant batch through a deep network, those stored intermediates (the "activations") swell into the largest thing on the GPU, far bigger than the network itself. Memory, not compute, becomes the ceiling on how much you can learn from at once.

Gradient checkpointing raises that ceiling with a simple bet: don't store the intermediates — re-derive them. Keep only a sparse set of waypoints from the forward pass and throw the rest away. When the backward pass reaches a stretch whose values were discarded, re-run just that slice of the forward pass from the nearest waypoint, use the regenerated values, and discard them again. The math is identical — the same gradients, down to the last bit — you've simply paid for parts of the forward pass twice. In practice the bill comes to roughly a third more compute in exchange for drastically less memory. It's the classic space-for-time trade, made where space is the scarce resource.

It's like hiking a long trail without photographing every step. You jot down the junctions; if you later need to retrace a stretch, you walk it again from the last junction. Retracing costs a little time, but your pack stays light the whole way.

Why is a third more compute so often worth it? Because on big training runs the GPU starves for memory long before it starves for arithmetic. Reclaiming the activation memory lets you run batches several times larger, and larger batches mean steadier, less noisy learning signals. Paying with compute — the thing you have — to buy memory — the thing you don't — is usually the right side of the trade.

**In this project:**

- v6 turns on checkpointing for both networks' torsos — the tall stacks of layers in the actor and the critic.
- The reclaimed activation memory is part of what lets each update process about 9 million decisions on a single GPU.
- An earlier generation had to shrink its batch from 11 million decisions down to 9 million partly because memory ran out — exactly the ceiling this technique attacks.
