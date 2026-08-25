# Gradients and backpropagation

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Every training update ends with a verdict. The network's decisions for a batch of hands are compared against what training says they should have been, and all that disagreement is compressed into a single number: the loss. Lower is better. The entire game of training is nudging millions of weights so this one number shrinks.

The gradient is how each weight learns its part in that. Every weight gets a personalized answer to one question: "If I grew slightly, would the loss rise or fall — and how steeply?" A weight whose answer is "the loss would drop sharply" is worth moving hard. A weight whose answer is "barely anything changes" can sit still. This isn't a guess or a sample; it is the exact sensitivity of the loss to that one weight, with everything else held where it is.

The remarkable part is the price. Computing millions of these personalized answers does not take millions of passes — backpropagation delivers all of them, exactly, in one backward sweep. Blame flows backward through the very same wiring the prediction flowed forward through.

## The bucket brigade

Think of it as a bucket brigade of responsibility running in reverse. The loss stands at the end of the line holding the full bucket. It hands blame to the last layer, which settles two accounts: how much of this belongs to my own weights, and how much belongs to the inputs I was handed? It keeps the first share — those are its gradients — and passes the second share up the line. The layer before it repeats the accounting, and so on back to the very first layer. When the brigade finishes, every weight is holding exactly its own share of the blame, nothing double-counted and nothing dropped.

That's why the forward and backward passes are two halves of one mechanism: the forward pass records how each output was built from its inputs, and the backward pass replays that record in reverse to divide up responsibility.

One wrinkle matters in deep networks. Blame that must pass through many layers in a row can fade, leaving the earliest layers with a weak, diluted signal — slow learners at the bottom of the stack. Skip connections fix this: the shortcuts that let a layer's input bypass a block and rejoin later also carry blame straight backward, so early layers hear the verdict at full strength.

**In this project:**

- The two networks (actor and critic) total 22,069,333 weights, and each one gets its own freshly computed gradient 32 times per training update.
- The torso's skip connections double as gradient highways: early layers receive their learning signal undiluted rather than faded through the full depth of the stack.
