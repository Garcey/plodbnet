# ReLU

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

ReLU is the simplest rule in the whole network. After a linear layer computes its numbers, ReLU walks down the list and applies one test to each: if the number is positive, keep it exactly as it is; if it's negative, replace it with zero. That's the entire operation.

It looks too trivial to matter. It's actually what makes deep networks deep.

Here's the problem it solves. A linear layer only re-weights and adds. Stack a second linear layer directly on the first and it re-weights and adds the results of re-weighting and adding — and sums of sums are still just sums. The whole stack collapses, mathematically, into one linear layer. Ten layers of pure weighted checklists can't compute anything a single checklist couldn't. Depth would be decoration.

ReLU breaks the collapse because zeroing things out is a decision, not arithmetic. Which numbers survive depends on the situation, so the next layer sees genuinely different information in different spots — and now stacking layers builds something new at each level: features made of features, reads made of reads.

It also gives each neuron an if-then character. The neuron's weighted checklist produces a score, and ReLU means the neuron only speaks when that score comes out positive — when the pattern it's tuned to is actually present. Otherwise it contributes exactly nothing downstream. It's like the tightest player at the table: silent orbit after orbit, but when they finally raise, it means something specific. The network's knowledge lives in millions of these quiet specialists, each firing only on its own pattern.

One failure mode is worth knowing: the dead ReLU. Suppose training drifts a neuron's weights to a place where its score is negative for every situation it ever encounters. Its output is then always zero — and a neuron that always outputs zero receives no learning signal, because nothing it does changes anything. It can't find its way back on its own. That's capacity paid for but permanently idle. A handful of dead units is normal; over very long training runs they can accumulate and quietly eat into what the network is able to learn.

That slow attrition is one of the reasons long-lived training setups add machinery to keep each layer's numbers centered and healthy, rather than trusting them to stay in range on their own.

**In this project:**

- ReLU follows every linear layer in the model's torso.
- The risk of units dying over very long training runs is part of why v6 added LayerNorm, which keeps each layer's inputs in a healthy range.
