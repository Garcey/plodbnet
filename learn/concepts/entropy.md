# Entropy

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Entropy is a single number you can attach to any probability distribution to say how mixed it is. If the distribution always makes the same choice — 100% on one option — its entropy is 0. If it spreads itself perfectly evenly, coin-flipping among everything on the menu, entropy sits at the maximum possible for a menu that size. Everything in between scores in between. It doesn't care which options are favored, only how concentrated or spread out the favoritism is.

For a policy, entropy is a decisiveness meter read backwards. High entropy means the model is genuinely mixing — many actions get real probability. Low entropy means it has made up its mind and plays nearly the same way every time it sees a similar spot.

Why would training care? Because a policy that commits too early stops generating evidence about anything else. If it decides in week one that checking is best, it stops raising — so it never collects the data that might have proven raising better. Reinforcement learning's fix is the entropy bonus: a small reward paid to the policy simply for keeping its distributions mixed, folded into the same objective as the poker winnings. A coefficient sets how much that mixing pays.

The mental model that makes tuning intuitive: the entropy bonus is a tax on decisiveness. While the tax is high, sharpening onto one action costs more than it earns, so the policy stays deliberately flexible and keeps exploring. Lower the tax and commitment becomes affordable — the policy is finally allowed to cash in what it has learned and play sharply. Walk it down slowly and you get the classic arc: explore broadly early, commit gradually as evidence accumulates. Drop it near zero too soon, and the policy can slam onto one action before it knows anything — decisiveness bought with ignorance.

So on a training chart, falling entropy is not by itself a warning sign. It's often exactly the goal: the policy graduating from "trying everything" to "knowing what it wants." The real question is always whether it's committing on evidence or merely collapsing.

**In this project:**

- The training log's H splits into three parts: Hg for the fold / check-call / raise gate, Ha for the 11-rung size ladder, and Hb for the refine slider.
- The gate's maximum possible entropy is ln(3), about 1.10, and the current run sits around 0.85–0.89 — very mixed play, which is the current diagnosis.
- The entropy coefficient currently sits at 0.25; historical cold starts in this project only began to sharpen once the coefficient came down into the 0.10–0.18 range.
