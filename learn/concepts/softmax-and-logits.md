# Logits and softmax

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

When the network looks at a poker spot, the last thing it computes is not a set of probabilities. It's a set of raw scores, one per option, called logits. A logit can be any number at all — 3.2, minus 40, a few hundred. There's no rule that logits must be positive or add up to anything. They're just the network's raw lean toward each option, on an open-ended scale.

Softmax is the conversion step that turns those raw scores into a proper probability list. After softmax, every option has a positive probability and the whole list sums to exactly 1. The option with the biggest logit gets the biggest share, and the relationship is exponential rather than linear: adding a fixed amount to a logit multiplies its share by a fixed factor. A score that's a couple of points ahead of its rivals doesn't get a couple of percent more probability — it gets several times as much.

If that sounds dramatic, think of a tournament payout ladder. Climbing one finishing place never adds a flat bonus; it roughly multiplies your prize, so small differences near the top turn into enormous differences in money. Softmax treats scores the same way: modest gaps between logits become lopsided gaps in probability.

## Masking: making "illegal" mean "impossible"

Here's the trick that matters most in this project. Suppose one of the options is against the rules right now. You could hope the network learns to score it low — but a low logit still leaves a sliver of probability, and a sliver, sampled millions of times during training, means the rule gets broken constantly.

The clean solution is masking: before softmax runs, overwrite the illegal option's logit with a number so negative it might as well be negative infinity. Softmax of a hugely negative score comes out as exactly zero — not tiny, zero. And because the mask is applied before the conversion, the remaining legal options simply absorb the freed-up share and still sum to 1.

That distinction — zero versus merely small — is the whole point. A discouraged action still happens occasionally. A masked action cannot happen, ever, no matter how confused the network is. The illegal move isn't penalized; it's removed from the menu.

**In this project:**

- The fold / check-call / raise gate is three logits, softmaxed into three probabilities.
- Illegal actions — like folding when nobody has bet — have their logit set to minus-a-billion before softmax, so their probability is exactly zero and the model literally cannot choose them.
- Because the mask enforces legality, the network never has to learn the rules of poker; all of its capacity goes into learning which legal action is best.
