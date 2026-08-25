# 3 · What the model says

*~10 minute read. Goal: trace one raise from probabilities to actual
chips — through the gate, the ladder, and the slider — and know why
illegal actions are impossible rather than discouraged.*

Chapter 2 was the model's ears. This chapter is its mouth.

When the observation goes in, what comes out is not "an action." It's a
**chain of three small decisions**, each one a probability distribution,
sampled in order:

1. **The gate** — fold, check/call, or raise? (3 options)
2. **The rung** — if raising: which of 11 ladder sizes? (the "anchors")
3. **The slider** — a fine adjustment around that rung. (continuous)

Why a chain instead of one giant menu of every possible bet? Two reasons.
First, it mirrors how the decision actually decomposes — *what do I want
to do*, then *how much* — so each stage learns a cleaner question. Second,
it's a data-efficiency trick: 3 + 11 + a slider covers the entire
continuous space of legal bets with a handful of outputs, and every
decision the model makes pours learning into those same few outputs
instead of spreading it across a thousand rarely-used menu items.

---

## Stage 1 — the gate

The network produces three numbers that become probabilities over
**fold / check-call / raise**.

You asked the right question about this in chapter 2's quiz: why do
check and call share a slot when they're such different actions? Because
as *outputs* they never compete: if there's no bet facing you, "call"
doesn't exist; if there is one, "check" doesn't. Exactly one of them is
available at any moment, so one gate covers both and the context decides
which one it means. (As *inputs* — in the history — they're recorded
separately, because "he checked" and "he called a pot bet" are different
stories.)

**Legality is enforced before probability, not after.** The engine hands
the network a mask of which gates are legal right now — fold is illegal
when there's nothing to fold to; raise is illegal when a bet already
covers your stack, or when a short all-in didn't reopen the action. The
mask sets illegal options' scores to negative infinity *before* the
probabilities are formed, which makes their probability exactly zero.
The model doesn't learn "don't pick illegal things" — illegal things are
unpickable, by construction. All poker-rulebook knowledge lives in the
engine; the network never has to learn the rules, only the strategy.

## Stage 2 — the rung

If the gate says raise, the next question is size — asked in the
**pot-fraction language** you met in chapter 2's history block.

The ladder has **11 rungs**: a **min-raise rung**, then 10%, 20%, … up
to **100% of pot**. "Percent of pot" here means the pot-limit convention
you already play by: the reference is the pot *after* your call, and a
"full pot" raise is a call plus 100% of that — exactly the pot-limit
maximum. So in this game the ladder isn't an arbitrary menu; **it spans
the entire legal sizing space by definition** — the bottom rung is the
smallest legal raise, the top rung is the biggest one the rules allow.
(This is also why the top rung doubles as "all-in" when stacks are
short: at 20bb, pot-raise and jam are usually the same number.)

Each rung converts to chips by a fixed formula — call amount plus the
rung's fraction of the after-call pot, rounded half-up to a whole chip —
and then gets **clamped into the legal window** [min-raise, max-raise].
Clamping creates a phenomenon you've already seen on the UI's bet chart:
**rung collapse**. Face a pot-sized bet and compute the rungs: the 10%,
20%, 30%, 40%, even the 50% rung all land at or below the min-raise, so
they all clamp to the same number. The dedupe rule is simple — a rung is
legal only if its chips are *strictly greater* than the rung below it —
so facing a pot bet, the bottom half of the ladder literally vanishes
from the menu and the "min" rung *is* the 50%-ish raise. The ladder
always covers min-to-max; how many distinct rungs survive depends on
the spot.

One special regime: when your stack is too short to make even a legal
min-raise (but jamming is allowed), the engine zeroes out min-raise and
the *only* surviving rung is the top one, which now simply means
**all-in** — the environment redirects any "raise" straight to the
all-in action and ignores the chip amount. You met this exact situation
in the Q-head audit: 20bb, facing a pot bet, where "raise" wasn't even
on the menu.

The network puts a probability on every *legal* rung (illegal rungs are
masked to zero, same trick as the gate) and one is sampled. *How* the
network represents that distribution internally — this is where v1's
single Beta, v2's raw categorical, v4's ordinal logistic, and v5's
3-component mixture differ — is a story that gets its own chapter
(ch. 10). For now: some distribution over legal rungs, one rung sampled.

## Stage 3 — the slider

Rungs every 10% of pot are still coarse — sometimes 43% is the bet, not
40%. So every **interior** rung carries a fine-tuning slider: a number
**u between 0 and 1**, drawn from that rung's own little learned
distribution (a Beta — a flexible bump on the 0–1 interval).

The slider moves the bet inside a **bracket of ±5% pot around the rung**
— deliberately half the distance to the neighboring rungs, so a slider
can *never* wander into another rung's territory. The mapping is linear:
u = 0 is the bottom of the bracket, u = 1 the top, and **u = 0.5 lands
exactly on the rung**. That midpoint is a cute, deliberate detail: a
freshly initialized network's slider distribution is symmetric, its
average is 0.5 — so an untrained model bets *exactly* the anchor sizes,
and refinement away from them is something it has to *earn* through
learning.

Two rungs have no slider at all: **min and pot are atoms**. Their
meanings are absolute — the smallest and largest legal raise — and
sliding off them in either direction would be either illegal or a
different rung's job.

### A full trace, with real numbers

Six-handed bomb pot, 3bb antes: the flop arrives with **18bb** in the
pot, nobody has bet, hero acts first.

- Legal gates: check/call and raise (fold is masked — nothing to fold to).
  Suppose the gate comes out 55% check / 45% raise → the dice say **raise**.
- The ladder, with no bet to call: min-bet 1bb, then 1.8, 3.6, 5.4, 7.2,
  **9**, 10.8, 12.6, 14.4, 16.2, 18bb. All eleven distinct, all legal.
  The rung distribution is sampled → say the **50% rung (9bb)**.
- The 50% rung's slider bracket is 9bb ± 0.9bb (that ±5% of pot). Its
  Beta produces **u = 0.72** → chips = 8.1 + 0.72 × 1.8 ≈ **9.4bb**.
- The engine applies a 9.4bb bet. The next player's observation now
  contains a history record reading: *hero, raise, flop, 9.4bb — 52% of
  the pot at the time.* The circle closes: the sizing head speaks the
  same pot-fraction language the history block listens in.

And one number gets written on the receipt: the joint probability of
that exact action — P(raise) × P(50% rung | raise) × the slider's
density at 0.72 — recorded as one log-probability. That single number
is what PPO will later compare the updated model against (chapter 7).

---

## Sampled or chosen?

Everything above **sampled** — training rolls the dice at all three
stages, which is where exploration comes from. But the same network can
be run in **deterministic mode**: take the highest-probability gate, the
highest-probability rung, and the slider's average instead of a draw.
That's what the study tab does when it shows you *the* recommendation —
same brain, dice removed. (The white dot on the bet chart is exactly
this: argmax rung, average slider.)

A quiet engineering note that will matter later: the chips math above —
formula, clamping, brackets — is implemented **three times** (once in
the network's tensor code, once in the collector/UI's numpy code, and
once in tests) and the implementations are pinned to agree **to the
exact chip**. That's not perfectionism. During learning, the updated
model must re-evaluate the *stored* actions, and if its idea of "the
50% rung" differed from the collector's by even a rounding direction,
every receipt comparison would be silently corrupted. Bit-exactness is
load-bearing (chapter 11).

---

## Check yourself

1. Fold, check, and call are three different poker actions. Why does the
   action head only need a 3-way gate rather than a 4-way one?
2. You're facing a pot-sized bet. Explain why the 10%–50% rungs
   disappear from the sizing menu, and what the "min" rung's chips
   actually are in that spot.
3. A freshly initialized, completely untrained network always bets
   *exactly* 40% pot, 50% pot, etc., never 43%. Which design choice
   makes that true, and why is it a sensible starting point?

*(Next: chapter 4 — what a hand is worth: the reward accounting, sunk
chips, your fold-EV-zero rule, and why all-ins are graded on 64 rivers
instead of the one that came.)*
