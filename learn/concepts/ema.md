# Exponential moving averages (EMA)

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

An exponential moving average is the simplest useful memory there is: keep one running blend, and at every step stir in a tiny fraction of the newest value. If the blend fraction is 1%, the new average is 99% of the old average plus 1% of what just happened. No history to store, nothing to replay — one running number that always leans toward the recent past.

Think of a perpetual stew. Each day the cook ladles a little out and stirs a little fresh in. Today's bowl mostly tastes of this week's cooking, faintly of last month's, and almost nothing of last year's — old contributions never fully vanish, they just fade away exponentially. The size of the daily ladle sets the memory: a big ladle and the stew chases every new ingredient; a tiny ladle and the stew barely changes for months.

That ladle size is the whole design decision. As a rule of thumb, blending in 0.1% per step gives a memory horizon of roughly a thousand steps — about how long it takes for old material to substantially wash out. And the smoothness you gain comes with a caution to respect: a very long horizon means the average is dominated by the distant past. If what you started with was garbage, the average stays mostly garbage for a very long time.

## EMAs of a whole model

The trick generalizes from single numbers to entire networks: keep an EMA of every weight in the model. The result is a smoother, steadier twin of the live network — it drifts along the center of the training trajectory instead of jumping with each update's noise. That twin is often the better one to show the world, precisely because it doesn't carry the latest update's twitchiness.

But the long-horizon caution bites hard on fresh runs. A slow EMA of a from-scratch model is, for its first many hundreds of updates, still mostly the random initialization — a smooth, steady, confident memory of knowing nothing. Pull the live model toward that and you're anchoring it to its own ignorance.

**In this project:**

- The (currently disabled) "magnet" regularizer pulled the policy toward an EMA of itself that updated 0.1% per update — a roughly 1,000-update memory. On a from-scratch run, that tethered the policy to nearly its random starting point, the root cause of a 200-update freeze.
- When the magnet runs, its EMA weights are saved in checkpoints and can optionally be served as the acting model — a smoother, less-exploitable actor.
