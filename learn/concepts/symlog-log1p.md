# Symlog and log1p: taming huge ranges

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Some numbers a poker network has to digest live on wildly different scales. A result might be a 2-big-blind scrap or a 1,500-big-blind mountain. Feed those in raw and the big values dominate everything: the gap between 1,400 and 1,500 takes up as much of the input's range as the gap between 2 and 102, while the small differences that decide ordinary hands all but vanish into rounding dust.

People don't perceive money on a raw scale either. The jump from a $10 pot to a $100 pot feels enormous; the jump from $1,000,010 to $1,000,100 feels like nothing — even though both are $90. Log-style compression builds that intuition into the encoding: it keeps small values clearly distinguishable and gently squeezes big ones closer together, spending resolution where decisions actually change.

Two flavors show up in this codebase. log1p means "take the log of one-plus-the-value," a small tweak whose whole point is handling zero cleanly — log1p of 0 is exactly 0, where a plain log of zero blows up. Symlog extends the same idea symmetrically around zero, compressing large negative values just like large positive ones — essential when the quantity is an outcome that can be a big loss as easily as a big win.

## The crude alternative: clipping

The other way to tame a huge range is a hard cap: pick a ceiling and record anything above it as the ceiling itself. That's clipping, and its failure mode is silent. Values past the cap don't get compressed — they get deleted. Everything above the line reads as exactly the same number, a state called saturation. The network isn't confused about those spots; it literally cannot see the differences between them, because the encoding threw the differences away before the network ever got a look.

That isn't hypothetical here. The stack-to-pot-ratio feature used to be clipped at 4, and the entire deep tier lived past the cap. Every deep-stacked spot arrived at the network reading exactly 4, so nothing downstream could distinguish moderately deep from very deep — until the feature was re-encoded without the cap, letting deep spots compress gracefully instead of flatlining.

**In this project:**

- The critic's 51 value buckets sit on a symlog grid, so ±1,500bb outcomes and 2bb outcomes share one scale.
- A real saturation bug: the stack-to-pot-ratio feature was clipped at 4, and every deep-stack spot (true SPR 5.4–13.9) read as exactly 4 — the model couldn't tell 100bb deep from 250bb deep until the feature was re-encoded as unclipped log1p.
