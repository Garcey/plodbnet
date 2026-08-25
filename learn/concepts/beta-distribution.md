# The Beta distribution

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Sometimes a decision isn't a menu — it's a slider. Not "which of these preset sizes," but "where exactly, between the two ends, do you want to land?" For that you need a probability shape that lives on a continuous stretch, and the Beta distribution is the standard tool: a flexible curve defined on the interval from 0 to 1, controlled by just two knobs, called alpha and beta.

Those two knobs give it a remarkable range of shapes. Set both to 1 and the curve is perfectly flat — every point on the slider equally likely. Raise both together and a hump grows in the middle: now the distribution has an opinion, centered at 0.5, and the higher the knobs, the narrower and more confident that hump becomes. Make the knobs unequal and the hump slides toward one end — more alpha leans it toward 1, more beta leans it toward 0.

The average is the tidy part: it's alpha divided by (alpha + beta) — alpha's share of the total. A useful picture is a tug-of-war. Alpha is a team pulling the marker toward 1, beta a team pulling toward 0, and the marker settles at alpha's fraction of the total pull. Add pullers to both sides equally and the settling point doesn't move, but the rope is held far more firmly — which is exactly what large, equal knobs do: same average, much less wobble.

Why is this the natural choice for "pick a point on a slider"? Because it's built for the job. It never proposes a point off the ends of the interval. It can express anything from "no idea, anywhere is fine" (flat) to "precisely here" (a tall, narrow hump). And the network only has to produce two numbers to say all of that.

One caution lives at the edges. If either knob drops below 1, the curve stops being a hump and starts piling probability into a spike at an end of the slider — mathematically legal, but jumpy and awkward to train against. Keeping both knobs at 1 or more guarantees a single well-behaved hump and calm behavior at the boundaries.

**In this project:**

- Every interior bet-size rung carries its own Beta-distributed refine slider, which fine-tunes the final bet within ±5% of pot around that rung.
- Both knobs are kept at 1 or above, so the shape always has a single hump and stays well-behaved at the slider's edges.
- A fresh, untrained network's slider averages 0.5 — and 0.5 is deliberately mapped to "exactly on the rung," so before any training the model's bets start centered on the rungs themselves.
