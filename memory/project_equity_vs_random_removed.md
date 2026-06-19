---
name: Equity-vs-random features removed from observation
description: 2026-05-06 production obs change — `hero_equity_a` / `hero_equity_b` (MC vs random hands) deleted from the encoder; OBS_DIM 741→739; underlying Rust hero_equity_mc bindings + engine method removed. Triggers retrain.
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
On 2026-05-06 the equity-vs-random features at obs offsets 739–740
(`_EQ_A_OFF`, `_EQ_B_OFF`) were deleted from the encoder. `OBS_DIM` is
now 739. Trailing trim — no other feature slot moves.

Removed end-to-end:
- `python/plo5bp/encoding.py`: `_EQ_A_OFF` / `_EQ_B_OFF` constants and
  the scalar / batched assignments. `encode_observation_batch` no
  longer takes `hero_equity_a` / `hero_equity_b` parameters.
- `python/plo5bp/env.py`: `_equity_cache` field, all `.clear()` calls,
  and the per-step `hero_equity_mc` calls in `_pack_obs`.
- `python/plo5bp/env_batched.py`: `_eq_cache`, the two-phase MC
  refresh, the `seeds_base` derivation.
- `python/plo5bp/rollout.py`: `env._eq_cache[i].clear()` in batched
  reset.
- `rust_engine/src/bindings.rs`: `PyGameState::hero_equity_mc` and
  `PyBatchedEngine::hero_equity_mc_batch`.
- `rust_engine/src/engine.rs`: `pub fn hero_equity_mc` and its two
  unit tests.
- Test `test_equity_in_unit_range` (test_encoding.py) and the
  `hero_equity_mc` parity assertions in test_batched_engine.py and
  test_encoding_batch.py.
- CLAUDE.md determinism-contract bullets that named the equity MC
  seed and per-hand `(seat, board) → equity` cache.

**Why:** PLO5 double-board bomb pots have extremely narrow continuing
ranges — a flop bet typically represents the nuts or near-nuts; turn
and river ranges with prior action are vanishingly small. Equity vs
*random hands* gives the network a number that is uninformative to
actively misleading (a 70% vs random hand can be ~5% vs the actual
continuing range). User: "I believe the equity vs random hands is
counterproductive to the model... Start by removing it." All training
checkpoints had already been deleted on 2026-05-06 ahead of this
representation overhaul.

**How to apply:**
- This is a production observation-semantics change. Retrain from
  scratch on the new 739-dim encoder; do not warm-start from any
  pre-2026-05-06 checkpoint (none exist on disk).
- If hand-strength-vs-range signals are reintroduced later, do them
  on the actual continuing range, not vs random — single-board PLO
  range-vs-range MC, or a learned hand-strength embedding. Don't
  resurrect `hero_equity_mc`.
- The EV runout seed (`base XOR 0x9E3779B97F4A7C15`) is the only
  remaining determinism contract for stochastic engine queries.
