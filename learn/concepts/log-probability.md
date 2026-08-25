# Log-probability (the receipt)

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Every time the policy acts during collect, the trainer stores not only
*what* it did but *how likely* that choice was under the weights at that
moment. That stored number is the **log-probability** of the taken action
— the "receipt." At update time the network recomputes the log-prob of the
*same* stored action under the *new* weights. The difference of those two
numbers is the entire fuel for the importance ratio:

```
ratio = exp(log π_new − log π_old)
```

Working in log space is numerical hygiene (products of small probabilities
become sums; under/overflow dies), but the concept is ordinary probability:
if the model said "30% raise pot" and took that raise, the receipt is
`log(0.30)`.

## Joint log-prob for a factored action

This project's action is a chain, so the receipt is a sum:

```
log π = log P(gate)
      + 1[raise] · ( log P(anchor | legal grid)
                   + 1[refine_ok] · log f_Beta(u) )
```

Fold and check/call have no sizing term. Raises that land on an atom
(min or pot) or a collapsed bracket skip the Beta term. Illegal gates and
anchors are masked to probability exactly 0 before this is computed, so
they never appear in a receipt.

## Why the receipt must be bit-exact

PPO compares `log π_new` (recomputed in `evaluate()`) against the stored
`log π_old` from `act()`. If the two code paths disagree by even one ULP of
rounding — different anchor-grid rounding, a drifted mask, a float32 vs
float64 slip — every ratio is silently wrong and the update optimizes
noise. That is why the sizing math lives in one module (`sizing.py`), why
numpy/torch twins are pinned bit-identical, and why observation encoding
is under bit-exact tests.

## In this project

- Stored on the batch as `log_probs` (joint) plus optional per-head
  breakdowns used for `klG` / `klA` / `klB` diagnostics.
- Recomputed every minibatch by `model.evaluate(...)` inside the PPO loop.
- Entropy is the expected *negative* log-prob under the distribution
  (how mixed the menu is); the receipt is the log-prob of the *one*
  action that was sampled.
