# Seeds and determinism

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

A computer never actually rolls dice. When code asks for a "random" number, it gets the next value from a deterministic recipe — a generator that was started from one chosen number called the seed. Same seed, same sequence of "random" values, forever, on any machine. The randomness is a performance: convincing, statistically impeccable, and completely rerunnable.

Left unmanaged, that's trivia. Managed deliberately, it's a superpower, because seeding everything makes randomness replayable. If every random choice in the system — the shuffle, the opponents' sampled decisions, the simulated runouts — draws from generators you seeded on purpose, then any hand ever played can be reconstructed exactly from a few stored numbers. Not a similar hand. The hand: same cards, same order, same everything.

Duplicate bridge works this way on purpose. Tournaments pre-arrange the deals so every table plays the identical hands, which lets skill be compared because the luck repeats exactly. Seeding is duplicate bridge for an entire training system: the deals are still fair and unpredictable in the moment, but any of them can be dealt again, identically, on demand.

Three things fall out of this. Bugs stop being ghosts — if something weird happened on a particular hand, you replay that exact hand a hundred times while you hunt the cause, instead of praying it recurs. Tests get to pin behavior byte-for-byte, which this project calls "bit-exact": a test doesn't check that an answer looks roughly right, it checks that it is identical, to the last bit, to a known-good result, so the slightest unintended change trips an alarm. And whole features come nearly free — a replay tool doesn't need to save a snapshot of everything, it just saves the seed and deals the hand again.

Best of all, the discipline is free at runtime. A seeded generator runs exactly as fast as an unseeded one; the only cost is the up-front care of threading seeds through every corner that uses randomness. It pays for itself, usually with interest, the first time something inexplicable happens.

**In this project:**

- Every hand is a pure function of three things — the table config, a seed, and the dealer button. The UI's review and what-if features literally replay hands from those; nothing is snapshotted.
- The all-in equity grading derives its runout seed from the hand's own seed via a fixed formula, so graded rewards are reproducible.
- The observation encoder is pinned by bit-exact tests, because a silent one-bit drift in what the network sees would corrupt training invisibly.
