# Linear layers and matrix multiplication

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Strip away the mystique and a neuron is a weighted checklist. It looks at every number coming in, multiplies each one by its own personal weight, adds the results together, adds one more number called a bias, and outputs the total. One number out — that's the whole job.

The weights say what the neuron cares about. A big positive weight means "this input pushes my answer up." A negative weight means "this pushes it down." A weight near zero means "I ignore this." The bias is a standing lean — where the neuron sits before any evidence arrives.

Think of how you decide whether to call a river bet. Pot odds weigh heavily. The blocker in your hand counts for something. That quick timing tell counts a little. "He's never bluffing here" counts hard against. You weigh each factor, sum the leanings, and act on the total. A neuron does exactly that, with actual numbers, millions of times a second.

A linear layer is nothing more than thousands of these neurons working side by side. Each one reads the same inputs but carries its own checklist — its own weights and its own bias — and each outputs its one number. So the layer as a whole turns one list of numbers into another list of numbers, one per neuron.

Here's where matrix multiplication comes in, and it's less scary than it sounds. Stack every neuron's weights as one row in a big grid — that grid is the matrix. "Multiplying" the matrix by the input list just means running every row's multiply-and-add against the inputs in a single sweep. It's not new math layered on top of the checklists; it's the same arithmetic written compactly so a computer can do all of it at once. This is also why graphics cards train networks so quickly — they're built to run enormous piles of multiply-adds in parallel.

The last piece is the most important one: nobody designs the weights. No engineer decided that a particular input deserves 0.3. Every weight starts as a small random number, and training nudges each one — a tiny step at a time — whenever the network's decisions turn out better or worse than expected. The checklists that end up mattering are discovered from experience, not written by hand. What you design is the shape of the network; what it learns is the weights.

**In this project:**

- The first layer of the poker model has 2,048 neurons, and each one reads all 1,171 observation numbers describing the spot — 2,398,208 weights (plus 2,048 biases → 2,400,256 params) in that single layer alone.
- Every "head" that reads a decision off the network is itself just one small linear layer sitting on top of the torso.
