# Weight initialization (and why some parts start at zero)

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Before a network has seen a single hand, every weight needs some starting value. The standard choice is small random numbers — and both words are doing real work.

Random, because of a trap called symmetry. If two neurons start with identical weights, they compute identical outputs, so they receive identical learning nudges — and after the update they are still identical. Clones stay clones forever, no matter how long you train. A layer of 2,048 identical neurons has the power of exactly one. Starting each weight at a slightly different random value breaks the symmetry: every neuron begins with its own faint leanings, training amplifies those differences, and the neurons specialize into genuinely different pattern-detectors.

Small, because nothing should dominate on day one. Big starting weights mean the network opens with strong random opinions — its outputs swing hard on noise, and the real learning signal has to fight through that. Small weights start the network near neutral: quiet, roughly uncommitted, easy for training to bend in any direction.

Then there's the deliberate special case: initializing a component to exactly zero. A zero-weighted part contributes literally nothing — perfectly neutral, invisible to the rest of the system. It isn't stuck there. The learning signal still reaches it (its inputs aren't zero, only its influence is), so training can grow it a role from nothing, and every bit of influence it ends up with was earned from experience.

That makes zero-initialization a safety pattern for upgrading systems that already work. Bolting a new component onto a trained, trusted model is risky if the new part arrives with random opinions — it perturbs everything the moment it's attached. Start it at zero and the guarantee is exact: on day one, the upgraded system behaves identically to the proven one, down to the last decision. It's like adding a new musician to a working band with their amp turned all the way down — tonight's show sounds exactly the same, and the volume only comes up as fast as they learn the songs.

So read a network's starting state as a set of commitments: random and small where you want diverse learners and no early favorites, exactly zero where a new part must first do no harm.

**In this project:**

- The critic's Q head is zero-initialized so that Q exactly equals V on day one — the new advantage estimator starts out identical to the proven old one.
- The v5 sizing head was zero-initialized so that a fresh model bets exactly the anchor sizes.
- With the refine slider untrained, its average lands exactly on each anchor — the neutral starting point is the anchor itself.
