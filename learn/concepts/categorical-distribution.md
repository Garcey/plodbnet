# Categorical distributions

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

A categorical distribution is the simplest probability object there is: a finite menu of options, each with a probability, all of them adding up to 1. Fold 10%, check-call 60%, raise 30% — that's a categorical distribution over three options. No curves, no formulas, just a weighted list.

Poker players already think this way without using the vocabulary. A mixed strategy — "in this spot I bluff 30% of the time" — is a categorical distribution. And the pros who glance at the second hand of their watch, or the suit of a card, to decide whether this particular hand is a bluffing instance? That's sampling. The strategy is the list of frequencies; the watch is the random draw that turns the list into one concrete action.

Sampling matters because a distribution isn't a decision — it's a recipe for making decisions. Each time you sample, one option comes out, drawn in proportion to its probability. Do it many times and the intended frequencies emerge. The alternative reading is the argmax: take the single highest-probability option, every time, deterministically. Argmax throws away the mixing and keeps only the top choice. Both readings of the same distribution are useful, for different jobs: sampling explores, argmax commits.

## The log-probability receipt

There's one more piece the training loop leans on. Whenever the policy samples an action, it also writes down the log of the probability it assigned to that action at that moment. Think of it as a receipt stapled to the decision: "at the time, I gave this action such-and-such a chance."

Why keep receipts? Because training later needs to compare policies. After the weights change, the algorithm asks: under the new policy, how likely would that same action have been? Comparing the new probability against the receipt shows whether the update made that action more or less likely, and by how much — exactly the signal PPO uses to push good actions up and bad ones down without over-trusting stale data. Storing the log of the probability, rather than the probability itself, is just the numerically comfortable way to do the bookkeeping; the idea is unchanged.

**In this project:**

- Both the action gate (fold / check-call / raise) and the 11-rung bet-size ladder are categorical distributions.
- Training samples from them — that's exploration: second- and third-choice actions keep getting real table time so the model can learn whether they're better than they currently look.
- The study tab's recommendation takes the argmax instead: deterministic mode, always the single most-probable action.
