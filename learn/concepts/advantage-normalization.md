# Advantage normalization

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

By the time an update runs, every decision in the batch has been graded with an advantage: how much better or worse the action was than par for its situation. Those raw grades arrive in wildly different sizes. A batch that happened to include a few monster pots carries advantages worth hundreds of blinds; a quieter batch tops out at a handful. Fed in raw, they would make the effective learning step swing with the batch's luck — huge strides after a swingy batch, baby steps after a calm one.

So the trainer standardizes first. It shifts the advantages so their average is zero, rescales them so their spread is one, and clamps the freak outliers so no single hand can dominate the step. It's grading on a curve: what matters isn't your raw score but where you sit relative to the class. A grade of +2 after the rescale means "two spreads better than typical for this batch," whatever chip amounts were flying around. Update after update, the optimizer sees signal of a consistent size.

## The shared-divisor catch

The curve has a sneaky failure mode. The rescale divides everything by one shared spread measure computed across the whole batch. If one subset of the batch is inherently swingy, its huge advantages inflate that shared divisor — and everything else gets divided by it too. The calm subset's perfectly clean grades come out looking tiny. The optimizer, seeing near-zero signal there, barely moves. Nothing is wrong with the calm subset's data; it has been drowned out by a loud neighbor sharing the same denominator.

This matters whenever a batch deliberately mixes situations of different volatility — exactly what happens when one policy is trained across many table conditions at once. And the cure isn't free: normalize each subset fully on its own and you lose any sense of which subset's decisions mattered more. It's a real tradeoff, worth remembering whenever one tier of a mixed run goes mysteriously quiet.

**In this project:**

- Advantages are normalized per config sub-batch, then re-normalized globally across the whole 30-config batch, with outliers clamped at ±8 spreads.
- That global coupling was a real incident mechanism here: noisy deep-stack advantage estimates inflated the shared divisor and crushed the clean shallow-tier signal.
- The crushed signal contributed to a 200-update learning freeze.
