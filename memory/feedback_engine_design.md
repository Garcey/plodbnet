---
name: Engine and ML design preferences
description: Recurring design rules the user applies to game engines and RL feature encoding
type: feedback
originSessionId: 0f4de9c9-28ef-47dd-90d1-414a1df544da
---
**Rule 1: For permutation-invariant multi-set inputs (hole cards, held items, etc.) never encode in deal order. Either use a multi-hot aggregate OR sort canonically (e.g., by rank desc, then suit) before one-hot encoding per position.**
Why: The net otherwise has to learn that 5 different orderings of the same hand are equivalent, wasting capacity on a trivial symmetry. Multi-hot matches the true symmetry exactly.
How to apply: When proposing observation encodings, flag any dim-per-position scheme and choose multi-hot when possible; sort first when per-position encoding is genuinely needed (suit-shape features, combos).

**Rule 2: Pin specific RNG implementations (e.g., `rand_chacha::ChaCha8Rng::seed_from_u64(seed)`); don't accept generic `impl Rng`.**
Why: Reproducibility across machines and Rust versions. Generic Rng allows the caller to pass anything, which silently changes results.
How to apply: Engine entry points that take a seed should instantiate the concrete RNG internally from `u64`, not take a generic RNG by reference.

**Rule 3: In discrete-action spaces with multiple sizings that can collapse to the same chip amount, apply one consistent masking rule across all collapses.**
Why: Inconsistent masking (e.g., mask when collapsing to CheckCall but leave legal when collapsing to AllIn) wastes policy probability mass on redundant actions and creates subtle biases.
How to apply: Cleanest rule: mask any sizing whose clamped chip value equals the value of a lower-index legal action. Applies to all collapses uniformly.

**Rule 4: Every per-seat field in the observation must be hero-rotated, not absolute-indexed. Slot 0 = hero, slot k = seat `(hero + k) mod num_seats`.**
Why: Absolute seat indices make the same strategic situation look different depending on hero's seat that hand, forcing the net to learn N positional models instead of one. Consistency across *all* per-seat fields (active mask, all-in mask, stacks, street/total commitments, and the seat channel in action history) is strictly better than partial rotation — partial rotation forces the net to learn a re-indexing, which is harder than rotation-by-construction.
How to apply: Any observation component with a seat dimension gets rotated; zero cost, big generalization win. Applies equally to history sequences and per-seat static fields.

**Rule 5: All-in status is a first-class observation feature, not derivable from stacks.**
Why: At short stacks, all-in players fundamentally change the decision context (can't be bet into, can't fold, chips committed). Stacks=0 alone is ambiguous with busted/folded; the net needs an explicit all-in mask. Especially critical in short-stack formats (~1 SPR postflop) where a large fraction of decision points involve ≥1 all-in player.
How to apply: Add an explicit all_in_mask to observations alongside the active (not-folded) mask. Do not collapse the two.

**Rule 6: When editing plan/spec documents, propagate changes through every cross-reference, not just the primary section.**
Why: The user has flagged stale cross-references (OBS_DIM mismatch between step 10 and the encoding table; test recap missing newly-added test cases) during plan review. Inconsistency between sections signals sloppiness and forces the user to re-check every detail.
How to apply: After making a change in one section of a multi-section document, explicitly scan all other sections that might reference the same value or list (constants, implementation steps, test summaries, verification sections). Edit those too in the same turn.
