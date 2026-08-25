# KL divergence

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Two policies are two sets of probabilities over the same choices. KL divergence is the standard yardstick for how different they are — not "do they pick different favorites," but how far apart the whole distributions sit, weighted by what actually gets played.

The units are surprise. KL is measured in nats, and the intuition really is the everyday word: how surprised, on average, would you be watching one policy's choices while expecting the other's? Identical policies score zero — nothing ever surprises you. The more often the watched policy does things the expected one considered unlikely, the higher the number climbs.

Here's the poker version. You have two years of notes on a regular. Tonight, hand after hand, their play keeps breaking from your reads — a fold your notes say they never make, a jam your notes call a once-a-year play. Your accumulating disbelief, tallied hand by hand, is the KL between the player in your notes and the player in the seat. A small tally means the same player having a normal night. A large one means somebody new is wearing their avatar.

Notice the asymmetry: it matters which policy you expect and which you watch. A player who starts doing things you thought near-impossible generates enormous surprise; a player who merely stops doing something rare barely registers. Swap the two roles and you get a different number — which is why this is called a divergence rather than a distance.

## Why training watches it

Each PPO update turns an old policy into a new one, and the KL between them is the honest measure of how far the policy actually moved, whatever the loss numbers claim. Small, steady KL between successive policies means measured steps taken close to the data just learned from — stable learning. A KL explosion means the policy lurched somewhere its training batch never graded, and the next batch will be played by a stranger.

**In this project:**

- The `kl=` field on every log line is the per-update policy movement; healthy runs here sit roughly between 0.003 and 0.05.
- Two guards watch it: a soft stop at 0.5 that skips the offending minibatch, and a hard rollback at 10 that restores the weights and the optimizer's memory as if the update never happened.
- The guards exist because a 2026 run (vTwo2) died in a single update with a KL near +2417.
