# 2 · What the model sees

*~10 minute read. Goal: know what's inside the 1,171 numbers — and, just
as important, what's deliberately NOT inside them.*

The model has no eyes. It never sees cards, a table, or a pot. At every
decision point, the engine freezes the moment and translates it into a
fixed list of **1,171 numbers** — the **observation** — and that list *is*
the model's entire reality. Two rules govern everything in this chapter:

1. **If a fact isn't in the numbers, it does not exist for the model.**
2. **The same spot must always produce the exact same numbers.** The
   whole system — training stability, replay, the UI, the test suite —
   leans on this. (It's why the encoder is pinned by "bit-exact" tests:
   if dim #700 quietly changes meaning, every trained model starts
   reading gibberish at dim #700.)

Think of the encoder as the world's most disciplined hand-history
narrator: it describes any spot in exactly the same order, in exactly the
same units, every single time.

A useful way to hold the layout in your head: it's what *you'd* text a
friend about a hand, formalized — "my cards, the boards, the stacks, the
action so far" — plus a section you wouldn't text: precomputed coaching
notes about hand strength. Here's the tour, neighborhood by neighborhood.

---

## Cards: three 52-slot checklists (dims 0–156)

The first 156 numbers are three copies of the same simple idea: a
checklist with one slot per card in the deck. Slot = 1 if that card is
there, 0 if not. One checklist for your five hole cards (five 1s,
forty-seven 0s), one for board A, one for board B.

Notice what this *doesn't* encode: nothing says an ace beats a king.
There's no rank order, no suit hierarchy, no "this is a good hand" baked
in. The model meets A♠ as "slot 51 is on" and has to learn everything
about what that means from millions of hands. (It does get help — see the
coaching notes below.)

## Table state (dims 156–196)

The basics of the moment: which street (one-hot — one slot per street,
exactly one lit); who's still in the hand and who's all-in; each seat's
remaining stack **in big blinds**; the pot, the amount to call, and the
min/max bet (all in bb); and whose turn it is.

One design choice here quietly does a lot of work: **every per-seat list
is hero-rotated.** Seat 0 always means "me," seat 1 always means "the
next player after me," and so on around the table. Your actual chair
number never appears. Why: the strategy for "button vs blinds" is the
same whether you're physically in seat 2 or seat 5, and rotating
everything to be hero-relative means the model learns that situation
*once* instead of six times — a symmetry handled by data formatting
instead of wasted learning.

Also note the units: everything is big blinds, never dollars or raw
chips. A 20bb spot is the same spot at any stake — another symmetry the
encoding gives away for free.

## The story so far (dims 196–772)

Here's a surprise about proportions: **the action history is still the
largest single block** — 576 of 1,171 numbers (~49%; the v7 tails added
stack/board/dual coaching after it). The model's "read" of a hand isn't
a summary like "there was a bet and a raise"; it's the raw sequence, the
**last 32 actions, oldest first**, each one an 18-number record:

- **who** acted (hero-relative seat),
- **what** they did — fold, check, call, or raise,
- **which street** it happened on,
- **how much** — the chips added, in bb,
- **how much, relative** — the same chips as a *fraction of the pot at
  that moment*, the same "percent of pot" language the sizing head
  speaks.

A detail worth savoring: the history records **check and call as
different things**, even though the model's own action head lumps them
into one check/call gate. That's not an inconsistency — it's the
difference between input and output. As an *action*, check and call are
one choice (only one of them is ever legal at a given moment, so one gate
covers both). As *information about the past*, "he checked" and "he
called a pot-sized bet" are wildly different stories, so the input
spells them apart.

## The coaching notes (dims 772–978)

In principle, the model could rediscover all of poker hand-reading from
raw checklists. In practice, every update spent rediscovering "a flush
draw has nine outs" is an update not spent on strategy. So the engine
precomputes a thick block of poker facts — think of it as handing the
student a calculator so the course can be about the math, not the
arithmetic:

- **SPR per seat** (stack-to-pot ratio) and **pot odds**.
- **Made-hand category on each board** — nine classes from high card up
  to straight flush — plus flush/straight **draw flags**.
- **Board texture**: how each board pairs (paired, double-paired,
  trips, quads), and how your hole cards interact with each board's
  ranks. A separate **rank histogram** of your hole cards closes a blind
  spot: pocket pairs that missed the board entirely.
- **A deep straight/flush block per board**: distance-to-the-nuts for
  your made flushes and straights, straight outs per 5-rank window (all
  ten windows, wheel through broadway), which flushes are even possible
  on this board, flush-draw outs per suit, *nut*-flush-draw outs, even
  straight-flush-draw outs.
- **Commitment**: how much each seat has put in this hand, and this
  street. **Who the last aggressor was. Distance to the button.**

One honest caveat the project lives by: coaching notes are only worth
having if they're *exactly right*. A subtle bug in "flush outs" wouldn't
crash anything — it would just quietly poison every decision the model
ever makes. That's the real reason the encoder is armored with bit-exact
tests and treated as the most change-averse code in the repo.

## The double-board block (dims 950–978)

This game's whole identity is that two boards run at once and scooping
both halves is everything. A dedicated block encodes exactly that
geometry: which ranks appear on *both* boards, and — per suit — whether
you have a flush **made on both**, a **draw on both**, or **made on one
and drawing on the other**. Same three-way breakdown for straights using
the same two hole cards. These are your "am I playing for the whole pot
or half of it" sensors, and no generic poker feature would capture them.

## Ranking your hand against the universe (dims 978–990)

The last big base block answers the question a human asks constantly:
*how does my hand rank against what's out there?* For hypothetical
opponent hands of 2, 3, and 4 cards drawn from the unseen deck, the
engine computes the fraction of all such combos that would **scoop you**,
**quarter you**, get **scooped by you**, or get **quartered by you** at
the current board.

Read that carefully: this is *not* peeking at anyone's cards. It's
range-vs-universe math — "against everything the deck could be holding,
where does my hand sit" — the formalized version of the ranking a strong
player carries in their head. The 2- and 3-card versions are computed
exhaustively; the 4-card version samples 1,024 combos with a fixed seed
(rule #2: same spot, same numbers, always).

Plus one small, very human number at dim 990: **the bet you're facing as
a fraction of the pot it was bet into** — "he bet half pot" as a literal
input, so the model doesn't have to reverse-engineer sizing tells from
raw chip counts.

## The engineered tails (dims 991–1171) — and a war story

Two pure-append tails sit after the 991-dim body that v2/v4 trained on.

**Obs-v2 tail (991–1020, 29 dims)** — appended for v5/v6: per-board
**ahead/tied/behind** plus "win-exactly-one-board" and "tie-both";
**blockers-to-the-nuts** per board; an **effective-price block** where
the price of a bet is capped by the *effective* stack (chips that can't
ever be lost shouldn't change your price — "dead-chip invariance"); and
a re-done **SPR block**.

**V7 batch-2 tail (1020–1171, 151 dims)** — stack geometry (41), board
texture (78), and double-board specifics (32). Live full-obs stems
(vSix3+) train at the full **1,171**.

The war story is the SPR redo inside the v2 tail. The original SPR
feature was clipped at 4 — any value above 4 was recorded as exactly
4.0. Perfectly fine at 20bb. But at the deep tier, real flop SPRs run
about 5.4 to 13.9 — meaning **every deep spot read as "SPR = 4"**. The
model literally could not tell 100bb deep from 250bb deep at the flop;
the feature had *saturated*. The fix: store `log1p(SPR)` instead — a
gentle compression that keeps big values distinguishable instead of
chopping them off. The lesson generalizes: a feature that saturates
doesn't error, doesn't warn, and doesn't train — it just silently
deletes information. (You watched a cousin of this lesson play out in
the Q-head audit.)

Both tails are **pure appends**. Dims 0–991 are byte-identical to the
v2/v4 layout (and 0–1020 to the pre-v7 layout), so older checkpoints
keep working by slicing off the tail. Encoding upgrades in this project
are add-only, never rearrange — because rule #2 isn't just per-spot,
it's per-generation.

---

## What is deliberately NOT in the observation

The absences are as designed as the presences.

**Opponents' hole cards.** The playing model is honest — it sees exactly
what you'd see in the seat. (During *training*, a separate grading
network does see everyone's cards — that's chapter 5's story, and the
cards enter through a completely different door that never touches live
play.)

**Any memory across hands.** No opponent tendencies, no "he's been
raising a lot," no session history. Every hand begins with total
amnesia. This is a real design choice with real consequences: the model
cannot develop reads or exploit an individual — instead, it's pushed
toward strategies that hold up against the whole *population* it trains
against. In poker terms: it's being raised as a GTO-leaning amnesiac,
not an exploit artist. (Whether the population it trains against makes
that a *good* GTO approximation is the self-play question — Part IV.)

**Dollars, positions-as-chair-numbers, or anything stake-specific.**
Everything is big blinds and hero-relative. The model learns poker, not
one particular table.

---

## Check yourself

1. Why does the history block record "check" and "call" as different
   things when the action head treats them as one gate?
2. The old SPR feature was clipped at 4 and nothing crashed or errored
   for months. What was actually going wrong, and at which stack tier?
3. Villain's cards are nowhere in these 1,171 numbers — yet the training
   process *does* use them. Reconcile those two statements in one
   sentence.

*(Next: chapter 3 — the other side of the conversation: how the model's
answer becomes a legal poker action, from the 3-way gate through the
11-anchor size ladder to actual chips.)*

---

## Obs modes (full vs minimal)

Production **full** mode is 1,171 dims (everything above, including the
v2 and v7 engineered tails). Experimental **minimal** mode (`--obs-mode
minimal`, stem vMin1) keeps only table-visible state — cards, street,
active/all-in, stacks, pot scalars, 32-slot history, commits, seat-exists,
button — **796 dims**. No SPR, categories, draws, blockers, opp-outcome
MC, or v2/v7 tails. The question: does the coaching block teach strategy
faster, or does it over-constrain what the net can discover?
