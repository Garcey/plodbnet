# Monte-Carlo estimates and variance

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Many of the numbers this project runs on aren't facts — they're averages over futures that haven't happened yet. What is this hand worth all-in on the turn? The true answer is the average payout across every river that could fall. Sometimes you can enumerate every possibility and take the exact average. Often that's too expensive: too many combinations, not enough time.

The Monte-Carlo move: don't enumerate, sample. Deal a random future and record the payout. Deal another. Average the results. With enough samples the average settles toward the true value — and it settles *around* the truth, not off to one side. Sampling adds noise, never a lean.

The noise has a stubborn price schedule: it shrinks with the square root of the sample count. Halving the noise takes 4 times the samples; quartering it takes 16 times. That's why nothing here uses a million runouts — the first few dozen samples buy most of the accuracy, and every further digit of precision costs an order of magnitude more compute.

## Running it twice, industrialized

Poker players already use this idea at the table: running it twice. When the money goes in and both players agree to deal two rivers for half the pot each, the EV of the spot doesn't change by a cent — but the swings shrink, because two samples average out more luck than one. Training does the same thing at scale: replace one realized outcome with the average over many simulated outcomes, and you remove luck without introducing bias. The number means exactly what it meant before; it's just quieter. Statisticians call this variance reduction, and training leans on it everywhere it can afford to.

One habit follows from all of this: any number still built on realized or sampled outcomes is a noisy reading of an underlying truth. Judge it over many prints, never off a single one.

**In this project:**

- Hands that get all-in before the river are graded by dealing 64 seeded runouts and averaging the payouts, instead of using the one river that actually came.
- The hand-strength features in the observation use exhaustive enumeration where it's cheap and 1,024 seeded samples where it isn't.
- The F/T/R diagnostic on the log line is outcome-dependent and carries showdown variance — which is why it's read as a trend, never as a single print.
