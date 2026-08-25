# LayerNorm

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

LayerNorm is housekeeping with outsized consequences. Before a layer does its math, LayerNorm takes the list of numbers about to flow in, re-centers it so the average sits at zero, and re-scales it so the spread is standard — then hands the tidied list to the layer. A pair of learned dials lets the network choose whatever scale and offset it actually prefers, so nothing expressive is lost. Think of it as a thermostat on activations: it doesn't decide what the layer computes, it keeps the operating temperature steady so the layer is always working in the range it works best in.

Why does a network need a thermostat? Because long training runs drift. Over millions of updates, the sizes of the numbers flowing through the network creep — one layer's outputs run a little hot, which changes what the next layer sees, which compounds down the stack. Neurons slide into dead zones and go permanently silent. The effective size of each learning step quietly changes, because the same nudge means something different when the numbers it acts on have grown or shrunk. None of this shows up as a crash. The network still plays — it has just quietly lost the ability to keep learning. Researchers call that lost quality plasticity.

LayerNorm preserves plasticity by construction. Every layer keeps receiving inputs in the range it was built for, update after update, however long the run goes — so the millionth lesson lands as cleanly as the first.

Two honest costs come with it. First, LayerNorm changes what even a fresh network computes. It is not a transparent add-on: insert it into an already-trained model and the same weights now produce different answers, so you can't bolt it on mid-life and simply continue. Second, normalization removes the natural brake on weight growth. Ordinarily, weights that balloon produce wild outputs and training pushes back; but if the normalizer undoes any blow-up in scale, the weights can inflate indefinitely with nothing to stop them. So LayerNorm needs a partner regularizer — something that gently restrains the weights — to be safe on truly long runs.

The overall bargain is a good one: a small change in what the network computes on day one, in exchange for a network that stays trainable indefinitely.

**In this project:**

- v6 added LayerNorm inside every residual block, placed just before the block's linear layer.
- Because the change is not function-preserving, v6 had to start training from scratch rather than continue from v5's weights.
- It ships paired with l2-init — a gentle pull of the trunk's weights back toward their starting values — as the partner brake on weight growth.
