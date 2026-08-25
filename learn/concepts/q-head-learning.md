# How the Q head learns

*Part of the [master doc](../MASTER.md) concept library. Plain-English
explainer. Companion to [dueling Q heads](dueling-q.md), which covers what
the head IS — this one covers how it learns.*

The Q head keeps three running estimates for every spot: what folding is
worth, what check/calling is worth, what raising is worth. Nobody ever
tells it these numbers directly. It learns them the way a student learns
from graded homework — and the grading works like this.

**One graded answer per hand, for one column only.** When a hand
finishes, every decision in it gets a final grade: the chips that
decision's seat eventually won or lost from that point on. That grade is
attached to the action *actually taken*. If the model called from some
spot and ended up −7bb, the *call* column gets nudged toward −7 for
spots like that one — and the fold and raise columns get nothing from
that hand, because those actions weren't played and produced no grade.
Each column only ever learns from the hands where its action was chosen.

Do this across millions of hands and each column settles toward the
*average* result of taking that action in that kind of spot. That's all
a Q value is: a long-run average of graded outcomes, sorted by action.
It's the same way a player's instinct for "calling here loses money"
forms — thousands of remembered results, compressed — except the model
needs no memory of individual hands, just the running average baked into
its weights.

**The one freebie.** There's a single action whose grade never needs
waiting for: folding is *always* worth exactly zero going forward (the
chips you already put in are gone either way — chapter 3's accounting).
So the fold column gets free, perfect homework at every spot where
folding was even *legal*, not just the spots where the model actually
folded. This "fold anchor" was recently turned up about 15× louder,
because it turned out to be whispering: the real-outcome homework comes
in huge, noisy chip amounts, and next to those the little "this should
be zero" corrections were only ~4% of the lesson. The `qF=` number now
on every log line is the running lie-detector — the fold column's
average score, which should hover at zero forever.

**Why it's harder than it sounds.** The head doesn't learn each column
from scratch — it learns each as a *correction* on top of the judge's
overall spot value (that's the dueling design). Two complications follow.
First, that baseline is itself still learning, so the corrections chase
a moving target. Second, the baseline and the homework grades are
measured on slightly different scales (the baseline uses a compressed
scale to handle huge pots gracefully; the grades are raw chips), and
the mismatch showed up as a systematic lean — every column reading a
few chips pessimistic, worse in deep games — which the fold column's
known-zero exposed during an audit. The louder fold anchor is the
current corrective; a deeper redesign is on the v7 list.

**Why mistakes here mattered so much:** these three columns aren't just
diagnostics — the v6 learning signal (VRPO) uses them as its baseline
for judging every decision. Noisy columns made the signal noisy (July's
starvation problem); leaning columns would bias it, except that a lean
shared equally by all three columns cancels out when actions are
*compared* — which is why training kept improving even while the lean
existed.

**In this project:**

- Grades are the same "returns" the value head trains on: chips won/lost
  from each decision forward, in big blinds, with all-ins averaged over
  64 runouts.
- The head is 3 columns (fold / check-call / raise) after the July
  pooling fix; each column's homework comes only from hands where its
  action was taken — plus the fold column's free zero-labels on every
  fold-legal row, now weighted 15× (live-tunable) with the `qF=` canary
  tracking it.
- New columns start at exactly zero correction, so the fancy learning
  signal begins life identical to the proven old one and only diverges
  as the columns earn real opinions.
