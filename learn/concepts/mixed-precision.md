# Mixed precision (bf16 and fp32)

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Every number a computer stores has limited precision. The standard format for training, fp32 ("full precision"), spends 32 bits per number and keeps about seven significant digits. The half-size format, bf16, spends 16 bits and keeps roughly two to three. Nothing in the machine holds a number exactly; the only question is how coarse the rounding is.

Why would anyone pick the coarse format? Because it's a bargain. bf16 math runs about twice as fast on a modern GPU and takes half the memory, which translates directly into bigger batches and faster training. And for the bulk of what a neural network does — millions upon millions of multiply-and-add operations flowing through its layers — the coarseness genuinely doesn't matter. Each individual result only needs to be roughly right; the network's job is pattern-finding, not bookkeeping, and tiny rounding wobbles wash out.

There is one place coarseness turns poisonous: delicate accumulations. When you add a small number to a much larger one in bf16, the small number can round away to nothing — the running total is too coarse to register it. Picture keeping a running count of a tournament chip tray where you're only ever allowed to write three digits: once the board says 1,240,000, tossing in a 25-chip changes nothing you're allowed to write down. The chip vanishes from the books. One vanished chip is trivia; half a million vanished chips is an accounting scandal. That is exactly what happens when you sum half a million small values in bf16 — the answer isn't slightly off, it's garbage, because precision ran out long before the last addition.

Mixed precision is the sensible truce: run the bulk network math in bf16 for the speed and memory, and escape to full fp32 exactly where the arithmetic is load-bearing — losses, averages, and any sum over a huge number of items. Training frameworks make this convenient with an "autocast" region: inside it, operations run in bf16 where that's known to be safe, and the code explicitly lifts the delicate pieces back to full precision.

The skill isn't choosing one format over the other. It's knowing which ten lines out of ten thousand actually need the expensive one.

**In this project:**

- The training update runs the networks forward in bf16 inside an autocast region — the bulk math that dominates each update's cost.
- Value-loss calculations and the big reductions (large sums and averages) are kept in fp32, where coarseness would actually distort learning.
- A recent addition, the fold-column supervision, explicitly lifts a true/false mask to fp32 before summing across roughly 562,000 rows — a bf16 sum over that many values comes back as garbage.
