# 20 · Inside the network

*Written early, by request — reads fine any time after chapter 3. A
little denser than usual (~12 minutes). Goal: follow the observation
through all four layers to the bet, then watch one weight decide its own
change during training. Every number in this chapter was read out of the
live architecture at OBS_DIM=1171 (vSix4-class).*

So far the network has been a box: 1,171 numbers in, probabilities out.
This chapter opens the box. It has two halves — the **anatomy** (the
forward pass) and the **mechanism of change** (how each of the
**22,069,333 weights** decides, every update, whether to grow or shrink).

---

## Part 1 — Anatomy: from 1,171 numbers to a bet

### The atom: one neuron

A neuron is embarrassingly simple: it holds a list of **weights** — one
per input — multiplies each input by its weight, adds the results plus a
small constant (the **bias**), and outputs that single sum.

The right mental model is a **learned checklist with importances**. A
first-layer neuron holds 1,171 weights, one per observation dim: maybe it
learned +2.3 on "flush possible on board A," −1.8 on "hero has the nut
blocker," near-0 on things it doesn't care about. Its output is "how
strongly does this spot match my pattern." Nobody assigns the patterns —
they emerge from training, and most defy clean English descriptions.

### The first layer

Layer 1 is **2,048 such neurons in parallel**, each with its own 1,171
weights — 2,398,208 weights doing one big matrix multiplication. In: the
observation. Out: 2,048 pattern-match scores — the spot re-described in
the network's own invented vocabulary instead of the encoder's.

Then comes **ReLU**, the simplest important function in deep learning:
*keep positives, zero out negatives.* Without something like it, stacked
layers would mathematically collapse into one layer (sums of sums are
just sums — the network could only ever draw straight lines through its
input space). ReLU gives neurons an if-then character: silent unless
their pattern is actually present. That tiny nonlinearity is what makes
depth meaningful at all.

### Layers 2–4: residual blocks

The next three layers (this is the `--num-layers 4` architecture: one
input layer + three blocks) are **residual blocks**, each 4,194,304
weights. A block does:

> take the 2,048-number summary **x** → normalize it (LayerNorm, below)
> → run it through a 2,048×2,048 neuron layer + ReLU → **add the result
> back onto x**.

That "add it back" — the **skip connection** — is the design choice that
matters. Each block doesn't *replace* the running summary; it computes a
**correction** and edits the summary in place. Layer 2 might sharpen
"multiway pot dynamics," layer 3 might combine "nut blocker + scare
turn" into something bluff-shaped, layer 4 refines further — but the
original signal always survives underneath the edits. This isn't
optional at this depth: the project's own code comment records that
without residuals, **the 2048×4 net fails to train at all**. (Why the
skip helps *training*, not just expression, becomes clear in Part 2.)

### Where LayerNorm sits, and why v6 added it

Inside each block, before the math, **LayerNorm** re-centers and
re-scales the 2,048 activations to a standard spread — a thermostat on
the block's inputs. Its job is *plasticity*: over billions of decisions,
activation scales drift, neurons drift toward permanently-silent (dead
ReLUs), and the effective step size of learning quietly decays.
Normalizing every block's input keeps the network operating in the same
regime at update 10,000 as at update 100 — it keeps an old network
*trainable*.

Two footnotes that connect to things you've lived: LayerNorm is **not
function-preserving** (it changes what even a fresh network computes),
which is a big part of why v6 had to be a cold start rather than a warm
continuation of v5. And it's deliberately paired with a partner
regularizer (l2-init — Part 2), because normalization alone removes the
natural brake on weight growth. Each LayerNorm also carries 4,096 small
learnable parameters of its own (a scale and shift per dimension).

### The heads: four tiny readers

After block 4, the torso's work is done: the spot now lives as one
2,048-number summary, call it **z**. Everything the model "understands"
about poker is in how the torso builds z. What remains is reading
decisions off it — and each reader is startlingly small, just one more
linear layer:

- **Gate head** — 3 neurons, 2,048 weights each (**6,147 params**
  total): three pattern-detectors on z, one per gate. Their three
  scores get the legality mask (−∞ on illegal), then softmax → the
  fold/call/raise probabilities of chapter 3.
- **Mix head** — 9 outputs (**18,441 params**): 3 components × (center,
  width, weight). Those spread over the 11 legal rungs to form the size
  distribution — *this head's output is literally the curve on the
  study tab's bet chart.*
- **Refine head** — 18 outputs (**36,882 params**): an (α, β) pair for
  each of the 9 interior rungs — the sliders.
- **Value head** — 1 output (**2,049 params**): the display EV you see
  in the UI.

Sit with the proportions for a second: 14.7M parameters build the
understanding; a few thousand read each decision off it. And all four
heads share the same z — the value head's learning pressure shapes
features the gate head also reads. One brain, four mouths.

### The second network

The critic — the all-seeing judge from chapter 1 — is a **separate**
network with the same recipe, different diet: its input is the same
1,171 dims **plus 260** more (a 5×52 checklist of every opponent's hole
cards), its torso is 1,536 wide with two residual blocks (~6.7M
params), and its heads output a 51-bin value *distribution* plus the
Q-scores-per-action head we rebuilt in the audit (13 columns in this
checkpoint; 3 in the pooled rebuild). Chapters 5 and 24 give it its due.

---

## Part 2 — The mechanism: how a weight decides its change

### One number to rule them all

Chapter 1's learn phase wants many things at once: raise probability on
positive-advantage actions, keep value predictions accurate, stay a bit
mixed (entropy), train the Q head, keep trunk weights near their start
(l2-init). Every one of those desires is written as a term in a single
number — **the loss** — and they're simply added up.

This has a consequence worth internalizing: **an individual weight never
knows which term is pulling it.** It feels only the net force. When the
Q-head fix added fold-column supervision, no weight was "assigned" to
it — the new term just changed the forces some weights feel.

### The gradient: 21.5 million personalized answers

For every single weight, training computes the answer to one question:

> *If this weight increased by a hair, would the loss go up or down —
> and how steeply?*

That per-weight answer is the **gradient**. Computing all 22,069,333 of
them takes one **backward pass** — backpropagation — and it isn't
approximate or mystical: the framework recorded every arithmetic step of
the forward pass, and blame flows backward through the same wiring.

The bucket-brigade picture: the loss tells the gate head "your fold
score was too high in these spots." The gate head, knowing its own
weights and inputs, splits that blame two ways: *how much was each of
my weights responsible* (those become its gradients) and *how much was
each element of z responsible* — which it passes back to block 4. Block
4 does the same and passes blame to block 3, and so on down to the
input layer. Every layer settles its own accounts and forwards the
rest.

Here's the payoff of the skip connections from Part 1: the "+x" in each
block is also a **gradient highway** — blame flows through the addition
untouched, so early layers receive it undiluted even through four
layers of math. Deep nets without residuals starve their early layers
of blame; that's precisely why the 2048×4 torso wouldn't train without
them.

### The gauntlet: from raw gradient to actual movement

A raw gradient does not simply get applied. In this project it runs a
four-stage gauntlet (this is the exact order in the code):

**1. AGC — the proportional speed limit.** Each tensor's gradient is
capped at 10% of that tensor's own size (weight norm). A big torso
matrix gets big allowances, a small head gets small ones — no single
minibatch can violently yank any one piece. (Fresh war story: the Q
head is now *exempt* — the audit found this cap was strangling a
zero-initialized head that needed to grow. Guards can misfire; ch. 26.)

**2. Split global clips.** The actor's gradients, taken together, get
clipped to a total budget; the critic's get their own separate budget.
Separate on purpose: the critic's loss lives at chip scale and can
spike, and a shared budget would let a critic spike throttle the
actor's gate learning in the same step.

**3. AdamW — per-weight adaptive steps.** The optimizer keeps **two
running averages for every weight**: the recent *direction* of its
gradients (momentum — the trend over roughly the last ten minibatches,
not the last noisy one) and the recent *magnitude* (how big this
weight's gradients typically are). The actual step is, in essence:

> learning rate × (trend) ÷ (typical magnitude)

So a weight with steady, consistent evidence takes confident steps,
while one getting large, conflicting signals takes small effective
steps. In poker terms: every weight sizes its adjustments to the
reliability of its reads. This per-weight normalization is why one
global learning rate (1.5e-4, times the warmup ramp) can serve 21.5M
weights with wildly different roles. (The "W" adds a mild weight decay.
And it's why restarts matter: those running averages aren't saved in
checkpoints — a restarted optimizer is briefly flying blind, which is
exactly why we re-ramp the learning rate after restarts.)

**4. Repeat ×32.** Sixteen minibatches, two epochs — the whole gauntlet
runs 32 times per update, each pass nudging all 21.5M weights.

### Where LayerNorm and l2-init fit in training

LayerNorm's training-time role is the flip side of its forward role:
because every block's inputs stay standard-scaled, gradient magnitudes
stay commensurate across layers and across months of training — the
learning machinery doesn't slowly detune as the network ages.

But normalization has a loophole: once activations are re-scaled anyway,
the raw *size* of trunk weights stops self-regulating, and growing norms
quietly shrink the effective learning rate. The counterweight is
**l2-init**: every trunk weight matrix feels a constant, gentle pull
back toward its value at run start (coefficient 0.0001; heads are
exempt). Implementation-wise it's elegant — just one more term added to
the loss, so the pull arrives at each weight through the same gradient
machinery as everything else. No special case, just another force in
the sum.

### The two guards that can veto everything

After each minibatch, the update's total movement (the `kl` you read on
log lines) is checked. Drift too far → **soft stop**: the offending
minibatch is never applied and the update ends early. Drift
catastrophically → **hard rollback**: weights *and* Adam's running
averages are restored from a snapshot taken at the update's start —
even the optimizer's memory is rewound, as if the update never
happened. Both exist because of specific disasters (chapter 8).

Two quiet enablers, one line each: **gradient checkpointing** throws
away the forward pass's intermediate activations and recomputes them
during backward — identical math, far less memory, and part of why a
9M-decision update fits on one GPU. And most forward math runs in
**half precision** (bf16), with the precision-critical sums kept in
full fp32.

---

## The map: where each piece lives

| Concept | Where in the code |
|---|---|
| Residual block (LN → Linear → ReLU → add) | `network.py` · `_ResidualBlock` |
| Torso assembly (input layer + 3 blocks) | `network.py` · `ActorCritic.__init__` |
| Gate/mix/refine/value heads | `network.py` · head `nn.Linear`s + `forward()` |
| Legality masking (−∞ before softmax) | `network.py` · `masked_fill` in `forward()` |
| Loss assembly (all terms summed) | `ppo.py` · `update()`, step12b block |
| l2-init pull (trunk-only) | `ppo.py` · `_l2_init_pairs` + its loss term |
| Backward pass | `ppo.py` · `loss.backward()` |
| AGC + the Q-head exemption | `ppo.py` · `_adaptive_grad_clip_`, `_agc_params` |
| Split actor/critic clips | `ppo.py` · the two `clip_grad_norm_` calls |
| AdamW construction | `ppo.py` · `optim.AdamW(...)` in `__init__` |
| KL guards + rollback snapshot | `ppo.py` · `kl_hard` / `target_kl` block |
| LR warmup ramp | `scripts/train.py` · `_lr_warmup_scale` |
| Gradient checkpointing | `network.py` · `_maybe_checkpoint` |

---

## Check yourself

1. A layer-1 neuron's checklist has 1,171 entries with obvious meanings
   (cards, pot, history). A block-3 neuron's checklist has 2,048
   entries. What is it a checklist *of*, and why is it harder to name
   its entries?
2. The gate head is 6,147 parameters — 0.04% of the actor. How can so
   few parameters be responsible for the fold/call/raise decision?
3. LayerNorm and l2-init ship as a mandatory pair in v6. What does each
   one do that creates the need for the other?

*(This chapter sits at position 20 in the syllabus — when you reach Part
V it'll already be an old friend. Next in reading order: chapter 4,
rewards and the accounting, which completes Part I and unlocks Lab A.)*
