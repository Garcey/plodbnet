# Adam: momentum and per-weight step sizes

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

After backpropagation hands every weight its gradient, something still has to decide how far each weight actually moves. The naive rule — step each weight directly on its latest gradient — has a flaw: any single batch of hands is noisy. One batch says "grow," the next says "shrink," and a network that obeys each verdict literally spends its energy chasing noise instead of the signal underneath.

Adam's fix is memory. For every individual weight, it maintains two running averages. The first is the recent trend of that weight's gradients — which way has the evidence been pointing lately, on balance? That's the momentum part. The second is the typical size of those gradients — how big are this weight's signals, noise included? Each update, the weight steps by roughly its trend divided by its typical size, times the learning rate.

That division is the clever bit. A weight whose gradients keep pointing the same way has a trend nearly as large as its typical size, so it takes close to full-size steps: steady evidence, confident movement. A weight whose gradients flip-flop has a trend near zero sitting on top of a large typical size, so it barely moves: conflicting evidence, cautious movement. Every weight travels at a pace matched to the reliability of its own evidence.

It's the discipline a good player applies to reads. One showdown where a new opponent turned over a bluff doesn't justify calling them down everywhere — that's a single noisy data point. Five sessions of consistent overbluffing justifies a confident, sizable adjustment. Adam applies exactly that judgment separately to every one of the network's millions of weights, on every update.

## The W in AdamW

AdamW adds one more ingredient: a mild weight decay — a gentle, steady pull on every weight back toward zero. It quietly discourages weights from staying large without ongoing evidence justifying their size, which helps keep the network from overcommitting to stale patterns.

**In this project:**

- The optimizer is AdamW. The trend (momentum) average updates with constant 0.9 and the typical-size average with 0.999 — the size memory is much longer than the trend memory.
- Those running averages are not saved in checkpoints. Every restart therefore begins with a blind optimizer that must rebuild its evidence from scratch — the operational reason restarts re-ramp the learning rate instead of resuming at full speed.
