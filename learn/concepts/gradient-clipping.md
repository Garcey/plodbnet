# Gradient clipping (including AGC)

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Most training batches produce well-behaved gradients. Occasionally one doesn't. A single strange minibatch — a freak cluster of coolers, a run of bizarre all-in hands — can produce a gradient far larger than anything around it, and if the optimizer obeys it, weights get yanked hard in a direction justified by one weird sample. The damage isn't just one bad update: a policy knocked off balance can take many updates to recover, and in the worst case never does.

Clipping caps the damage while changing nothing else. Before the step is taken, the gradient's overall size is measured; if it exceeds a fixed cap, the whole thing is shrunk down to the cap — direction preserved exactly, magnitude limited. Ordinary batches pass through untouched. Wild ones still move the network the way they wanted to, just not as far.

## Two flavors of cap

The first is the classic global cap: one limit on the total size of the entire gradient, with everything scaled down together when it trips. The second is Adaptive Gradient Clipping — AGC — which works tensor by tensor. Each weight matrix gets its own allowance, proportional to its own size: a big torso matrix gets a big allowance, a small output head a small one. That stops a spike in one small corner from hiding inside a global measurement, and stops any single tensor from being moved wildly out of proportion to what it currently is.

Clipping is a stop-loss for the update: it caps how much any one session can move your bankroll without changing how you play a single street — but a stop-loss set too tight pulls you out of the best game you'll sit in all year. A clip threshold can misfire the same way, strangling a part of the network that legitimately needs to grow fast. AGC's proportional rule makes this failure sneaky in one case in particular: a tensor that starts near zero has an allowance near zero, so a head that begins at zero and is *supposed* to grow can find its gradient permanently choked by the very rule meant to protect it.

**In this project:**

- AGC caps each tensor's gradient at 10% of that tensor's own weight size.
- The actor and critic get separate global caps (0.5 each), so a chip-scale spike in the critic's loss can't throttle the actor's learning.
- The critic's Q head was recently exempted from AGC after an audit showed exactly the misfire above: the cap was strangling a zero-initialized head that needed to grow.
