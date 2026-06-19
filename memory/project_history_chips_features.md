---
name: History slots store gate + street + chips/bb (no more action bucket one-hot)
description: 2026-05-06 production obs change — per-slot 8-action one-hot replaced with 4-way gate + 4-way street + chips/bb scalar; OBS_DIM 846→878. Triggers retrain.
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
On 2026-05-06, immediately after the straight/flush/SF block
(`project_straight_flush_features_added.md`), the per-history-slot
encoding was rewritten. The old 16-dim layout dedicated 8 dims to a
one-hot of the discrete BetPctN action bucket — a relic from before
the gate + Beta(α,β) policy head shipped. The engine had been
storing chip amounts and street boundaries on every history record
all along; the encoder just discarded them.

**New per-slot layout (17 dims):**

```
0..8     hero-relative seat one-hot
8..12    gate one-hot {Fold=0, Check=1, Call=2, Raise=3}
12..16   street one-hot {preflop=0, flop=1, turn=2, river=3}
16       chips / cfg.bb (raw scalar, no clamp)
```

Per-slot dim grew 16 → 17. With 32 slots, the history block grew
512 → 544 dims. **OBS_DIM 846 → 878** (+32). All offsets at and below
`_SPR_OFF` shifted by +32.

**Encoder-side gate (4-way) is distinct from the policy gate (3-way).**
The policy gate stays {Fold, CheckCall, Raise}; the encoder splits
CheckCall into {Check, Call} so the network sees check-with-no-bet
explicitly. Derivation at encode time from the engine's 8-action
record:

```
action == Fold                       → Fold (0)
action == CheckCall, chips == 0      → Check (1)
action == CheckCall, chips > 0       → Call (2)
action ∈ {BetPct10..100, AllIn}      → Raise (3)
```

`chips` is the engine's stored field on the history record:
the seat's street-total commit at the moment of the action. Fold and
Check both write 0; call writes the call amount; raise writes the
new total street commit.

**Implementation surfaces:**
- `rust_engine/src/bindings.rs`: `PyBatchedEngine.observation_arrays()`
  now also emits `history_chips: (N, 32) u64` and
  `history_street: (N, 32) i8` (with `-1` sentinel for empty slots).
  `PackedObservation` got matching fields populated in
  `pack_observation`'s slot loop. The serial
  `PyGameState.observation_dict()` already returned the full
  `(seat, action, chips, street)` 4-tuple — no change.
- `python/plo5bp/encoding.py`: scalar history loop calls a new
  `_gate_from_action(action, chips)` helper; batched encoder reads
  `history_chips`/`history_street` and derives the gate via boolean
  masks. Bit-exact parity vs scalar enforced by the existing batch
  parity sweep.
- 7 new history-slot tests in `tests/python/test_encoding.py`
  exercise the four gate cases, the street one-hot across all 4
  streets, and chips/bb scaling.

**How to apply:**
- Production observation-semantics change. Retrain from scratch on
  the new 878-dim encoder; do not warm-start from any pre-2026-05-06
  checkpoint (none exist on disk by design).
- Closes the chip-amount blind spot: a 75% pot bet was previously
  identical regardless of pot size — now the network sees the
  actual chips committed. Also tags street boundaries explicitly,
  so the network no longer has to infer them from board-rank
  changes across history slots.
- The 8-action enum still drives the discrete-action mask path;
  encoder-side gate derivation is independent. Don't conflate them.
