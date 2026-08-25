# 1 · One lap around the training loop

*~10 minute read. Goal: by the end, you can read one line of the training
log and say what every field on it measures.*

Everything this project does during training is one loop, run over and
over. The loop has two phases:

1. **Collect** — play an enormous batch of poker and write everything down.
2. **Learn** — grade every recorded decision and nudge the weights.

One full lap is called an **update**. On the pod, one lap currently takes
about 11–13 minutes and contains roughly **9 million decisions**. The run
you've been watching (vSix4, and the parallel vMin1 experiment) is the
same loop — billions of decisions across successive stems. Everything else in this curriculum is detail hanging off this
one loop.

---

## Phase 1 — Collect

### 49,134 tables at once

Training doesn't play one hand at a time. It runs **49,134 tables
simultaneously**, all dealt and managed by the Rust engine. This isn't a
poker decision — it's a hardware one: the network grades and answers
decisions in giant batches, and a GPU answering 30,000 questions at once
takes barely longer than answering one. Volume is also what makes the
learning signal usable at all, for reasons you'll see in Phase 2.

The tables aren't identical. Each lap, the trainer samples **30 table
configurations** — different seat counts (2–6) and different stack depths,
drawn from three "tiers" (roughly: 20bb-ish ClubGG stacks, a deeper ClubGG
band, and a 100–250bb deep tier). The 49k tables are split across those 30
configs. This forces one model to be competent shallow and deep, shorthanded
and full ring, rather than overfitting to a single lineup. (Chapter 8 covers
this properly — including a trap in how it's logged.)

### What the network sees

When it's some seat's turn to act, the engine freezes the moment and
describes it as a list of **1,171 numbers** — called the **observation**
(full mode; the vMin1 experiment uses a 796-dim minimal subset).
Conceptually it contains: the hero's five cards, both boards, the pot and
stacks, who's still in, the action history of the hand, and a set of
precomputed poker facts (things like "how often does hero's hand end up
ahead on runouts" — the engine computes these by brute force so the network
doesn't have to rediscover hand strength from scratch).

Two things matter here. First, the network can only know what's in those
numbers — if a fact isn't encoded, it doesn't exist for the model. Second,
**opponents' hole cards are NOT in there**. The playing model is honest: it
sees exactly what a human in the seat would see. (Something else DOES get
to peek — Phase 2.)

### How it answers

The network's answer is a **two-stage decision**, mirroring how the game
actually works:

- **Stage 1 — the gate:** probabilities over *fold / check-call / raise*.
- **Stage 2 — the size:** if raising, probabilities over a ladder of **11
  anchor sizes** (min-raise, 10% pot, 20% … up to full pot), plus a small
  "refine" adjustment that can slide between neighboring anchors.

Illegal options are hard-masked to zero before anything else happens — the
engine tells the network what's legal (you can't fold when there's no bet;
you can't raise when the bet covers your stack), and the network literally
cannot choose an illegal action.

Then — important — the action is **sampled**, not just "pick the highest."
If the gate says 55% call / 30% raise / 15% fold, the trainer rolls those
dice. That's where exploration comes from: the model constantly tries its
second and third choices, which is the only way it can ever discover
they're undervalued. (This is entropy's territory — the knob you already
know — chapter 6.)

### The receipt

For every decision, the trainer records the observation, the action taken,
and one more thing that will matter enormously later: **the probability the
model assigned to that action at the moment it acted**. Think of it as a
receipt. During the Learn phase, the updated model gets compared against
these receipts — "you used to raise here 30% of the time, now you'd do it
34%" — and that comparison is the core of how PPO keeps updates safe
(chapter 5).

### Who it's playing against

Mostly itself — every seat at most tables is played by the current model —
but a fraction of tables seat **frozen snapshots** of the model from
earlier in the run (the "pool," currently 8 snapshots, refreshed every 5
updates). Training purely against your current self has a failure mode
where the model chases its own latest quirks in circles; keeping some
recent-past opponents in the mix dampens that. Chapter 7 is about this.

### When the hand ends: the reward

At the end of each hand, every seat gets a number: **chips won or lost,
measured in big blinds**. This is the *reward* — the only opinion the
training process ever gets about quality. Nobody labels bluffs as good or
bad; there is no poker knowledge anywhere in the loss. Just chips.

The accounting has one convention you already know, because you worked it
out yourself: **chips already in the pot are sunk**. Costs are charged at
the moment you put chips in, and winnings arrive at the end, so from any
decision point, folding has a forward value of *exactly zero* — the ante
you posted is gone regardless and isn't blamed on the fold. (Your
observation about this is now literally a term in the training loss —
chapter 9 tells that story.)

One refinement: if the hand ends all-in before the river, the reward is not
the one runout that happened to come. The engine deals **64 different
runouts** and averages them. A cooler river no longer teaches the model
that getting it in good was a mistake — that's variance reduction at the
source, and it's a theme you'll see everywhere in this project.

---

## Phase 2 — Learn

### The grading question

Nine million decisions are now sitting in memory with rewards attached.
The naive move would be: raise the probability of everything that won
money, lower everything that lost. That fails in poker, and you know
exactly why: **results are mostly luck**. Punishing a correct hero call
that ran into the top of villain's range is results-oriented thinking, and
it would teach garbage.

So the system asks a sharper question about every decision:

> **Did this work out better or worse than *expected* from that spot?**

The "expected" comes from a second network — the **critic** — whose whole
job is estimating the value of a situation. The difference between what
actually happened and what the critic expected is called the **advantage**.
Positive advantage → this action beat expectations → nudge its probability
up. Negative → nudge it down. The size of the nudge scales with the size of
the surprise.

This is the heart of the entire system, and it's worth saying in poker
terms: **the advantage is an anti-results-oriented-thinking machine.** The
critic supplies the "what was my EV here" baseline; subtracting it strips
out the luck you couldn't control, leaving mostly the part your *decision*
was responsible for. A winning call still grades negative if the spot was
even better than the result; a losing call grades positive if it lost less
than that spot loses on average.

One asymmetry that surprises people: **the critic gets to see everyone's
hole cards.** That's legal because the critic never plays a hand — it only
grades. Give the judge x-ray vision and its EV estimates get far more
accurate, which makes the advantages cleaner, which makes learning faster.
The playing model stays honest; only the grader cheats. (This is called a
*centralized critic* — chapter 4.)

### The nudge

Now the actual weight update — the part where your existing vocabulary
slots in. The 9M graded decisions are shuffled and split into **16
minibatches**, and the trainer makes **two passes** (epochs) over them: 32
weight nudges per lap. Each nudge pushes probabilities toward
positive-advantage actions and away from negative ones, scaled by the
**learning rate**, with the **entropy bonus** riding along as a gentle
counterweight against becoming too sure of anything too early.

PPO's contribution — the reason it's the industry default — is a set of
seatbelts on this nudge. The famous one is the **clip**: no matter how
excited the math is about some action, the policy may only move a bounded
distance from the receipts recorded during play. On top of that, this
project runs KL guards that abort or roll back an update that moves
suspiciously far (a hard-won feature — a 2026 run named vTwo2 destroyed
itself in one update before these existed). All of chapter 5 is this.

### Housekeeping, then go again

Lap ends: save a checkpoint (every 5th one is kept as a numbered file and
becomes a pool snapshot), print one line to the log, resample the 30 table
configs, and start the next lap. That's the whole life of the system —
about 120 laps a day, forever, until we stop it.

---

## Reading the log line

Here's a real line from this week (u8 of the current segment), broken down.
This is your chapter-1 exam:

```
[ 7202.0s] update 8  pi=-0.0032  v=3.1989  vd=2029.68  H=1.519
Hg/Ha/Hb=0.87/1.62/-0.01  kl=+0.0038  klG/klA/klB=+0.003/+0.001/-0.001
q=2380.14  bonus=+0.0000  bonus%(F/T/R)=14.4/22.7/32.4  pool=8
seats=3  stacks_bb=[35.8, 29.1, 98.9]  ent=0.250  lr×0.90
```

- **`[7202.0s]`** — seconds since this training process launched. (The
  update counter restarts from 0 when the process restarts; checkpoint
  *files* are numbered cumulatively.)
- **`pi=-0.0032`** — the policy loss: how much improving-direction the
  update found in the advantages. Consistently negative at a few
  thousandths = actively fitting real signal; hovering at ±0.001 with
  flipping signs = nothing to fit (we watched exactly this distinction
  this week).
- **`v=3.1989`** — the critic's prediction error, in its own scoring units.
  Watch its *trend*: falling = the judge is improving; stable = converged;
  climbing/NaN = trouble.
- **`vd=2029.68`** — same idea for the actor's own little value head (the
  one the UI displays). Measured in raw squared big blinds, hence the huge
  numbers. Bouncy is normal.
- **`H=1.519`** and **`Hg/Ha/Hb`** — entropy: how mixed the play is,
  split into gate / size-ladder / refine. **Hg is the one we stare at.**
  0 = always the same gate choice; 1.10 = coin-flipping all three. The
  0.87 here is the "frozen, too mixed" reading from this week's saga.
- **`kl=+0.0038`** — how far the policy actually moved this update
  (klG/klA/klB split it by head). Healthy laps live around 0.003–0.05;
  the guards trip near 0.5.
- **`q=2380.14`** — the Q head's prediction error (the component we just
  rebuilt — chapter 9).
- **`bonus=+0.0000` / `bonus%(F/T/R)`** — a leftover diagnostic. The
  aggression bonus reward is OFF (the 0.0000); the F/T/R percentages track,
  per street, how often a decision was a raise that *won* (everyone folded,
  or took ≥half the pot at showdown) or a call that won. Outcome-dependent,
  so it carries luck — read trends, never single lines.
- **`pool=8`** — opponent snapshots currently in the pool.
- **`seats=3 stacks_bb=[…]`** — ⚠️ the classic trap: this is **one sample**
  of the 30 configs this lap, printed for flavor. The stats on the line are
  aggregates over all 30. Never explain a stat by this field.
- **`ent=0.250`** — the entropy coefficient currently in force (your knob).
- **`lr×0.90`** — learning-rate warmup: this lap ran at 90% of full LR.
  The suffix disappears once warmup completes.

---

## Check yourself

1. A hand ends with hero folding the flop after posting a 3bb ante. What
   reward does that fold contribute, and why isn't it −3bb?
2. Why does the critic get to see every player's cards when the actor
   doesn't — and why doesn't that leak into how the model plays live?
3. `seats=6` shows on a log line with a bad-looking F/T/R. What's wrong
   with concluding "full-ring is dragging the stats down"?

*(Answers are all in the text. Next: chapter 2 — what's actually inside
those 1,171 numbers, and how the two-stage action head works.)*
