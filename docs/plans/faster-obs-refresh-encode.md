# Plan: Attack #3 — Faster observation rebuild (`refresh`)

Reference: `memory/rollout_saturation_profile_2026-07-13.md` (~**104 s / ~23%** of rollout in `step1a/refresh`).  
**Not in scope:** cross-update stale self-play; changing OBS_DIM meaning without a design doc.

---

## Goal

Same observation **values and layout** (bit-exact with today’s numpy/scalar reference), **less wall-clock** to produce them after each apply/reset.

Keep training semantics: current learner, current hand truth, then optimize.

---

## Current architecture (facts)

### Call sites (batched rollout)

| When | Call | Encode scope |
|---|---|---|
| Post-`apply_hybrid` | `env._refresh(encode_mask=~newly_terminal)` | Skip encode for newly-terminal rows (zeros); **always** refresh commit/legal/actors for all |
| All terminal | `encode_mask=all-False` | Zeros only for obs |
| Post-`reset_terminal` | `env._refresh_subset(reset_mask)` | Only re-dealt rows |

Already good: terminal skip + subset after reset (`test_refresh_encode_mask.py`, `test_refresh_subset_parity.py`).

### Two encode backends (`env_batched.py`)

```text
_use_rust_encoder =
  PLO5_RUST_ENCODER=1
  and not NLH
  and OBS_DIM == _RUST_ENCODER_OBS_DIM   # currently 1171
```

| Path | What runs |
|---|---|
| **Rust ON** (full mask) | One FFI: `observation_encoded_batch()` → pack + parallel `encode_obs_row` (Rayon) + aux |
| **Rust ON** (partial mask) | **Full** `observation_and_features_batch()` **plus** `observation_encoded_subset_batch(idx)` → **double pack** of many envs |
| **Rust OFF** | Full `observation_and_features_batch()` + numpy `encode_observation_batch` (large pure-Python/numpy assembly in `encoding.py`) |

Rust already has **v7 tail** in `rust_engine/src/obs_v7_inc.rs` (dims 1020..1171) and `encode_indexed` uses `obs_layout::OBS_DIM` with `par_chunks_exact_mut`. Guardian sets `PLO5_RUST_ENCODER=1`. Width gate is **1171**, so training **should** use Rust when the extension is built.

**Critical inconsistency to verify first:** comments in `test_encoding_rust.py` still say “force-disabled / 1020”; `env_batched` width-gates at **1171**. Plan assumes we **measure** which path the pod actually takes.

### Profile implication

`step1a/refresh` ~14 ms × ~7.4k calls ≈ 104 s. That total includes pack + encode + unpack. Without sub-timers we don’t yet know the split.

MC for opp-outcome in training is already reduced: `TRAIN_OPP_OUTCOME_MC = 384` (not 1024).

---

## Root causes to attack (ordered)

### R1 — Confirm backend + split cost (Phase 0, mandatory)

On a short pod profile (`PLO5BP_STEP_TIMERS=1` + existing `record_function` names under refresh):

- Log once per process: `_use_rust_encoder`, `OBS_DIM`, `PLO5_RUST_ENCODER`
- Break refresh wall into: pack / encode / unpack (extend step timers or use existing `step1a_bundle/*`, `step1/encoder`)

**Decides** whether R2 (fix partial path) or R3 (re-enable/fix rust) is the main lever.

### R2 — Eliminate double work on partial `encode_mask` (high ROI, correctness-friendly)

**Bug/smell (code today, rust partial path):**

```text
bundle = observation_and_features_batch()     # pack+features ALL envs
enc    = observation_encoded_subset_batch(idx)  # pack+encode SUBSET again
```

Full pack is needed for `_unpack_post` (commits/legal/actors for **all** rows including terminal). Subset encode re-packs the kept rows.

**Target:**

| Step | API |
|---|---|
| 1 | One full **pack+aux** for caches (`observation_and_features_batch` or a thinner `pack_and_aux_batch` if we add it) |
| 2 | Encode **only** `idx` **without** a second full pack — either encode from already-packed full arrays by row index, or `observation_encoded_subset_batch` **only** if pack cost is proven small vs encode |

Preferred design (bit-exact):

1. Add Rust `encode_from_packed_indices(packed, idx)` **or** extend subset path to accept optional precomputed packed state.  
2. Simpler intermediate: **numpy partial path** already slices one full bundle and encodes subset — mirror that for Rust: pack once → encode rows `idx` in parallel from that pack → scatter obs; unpack caches from same pack.

Also apply the same “one pack” discipline if any numpy path still double-fetches.

**Tests:** `test_refresh_encode_mask.py` (zeros on skipped rows, full equality on kept rows), `test_encoding_rust.py` 3-way parity, `test_refresh_subset_parity.py`.

### R3 — Ensure full-batch Rust encoder is actually live

If Phase 0 shows `_use_rust_encoder=False` on the pod:

1. Fix gate / rebuild (`maturin develop --release`) so 1171 path works.  
2. Align comments/tests (`test_encoding_rust.py` skip flags, CLAUDE notes).  
3. Prefer **always** `observation_encoded_batch` / subset for PLO training (numpy remains reference for tests).

If already live, R3 is “confirm + document,” not a rewrite.

### R4 — Subset / dirty-row hygiene (medium)

Already: post-apply skip terminals; post-reset subset only.

Further (only if safe and measured):

- Skip encode for rows with `actor == -1` even when not in `newly_terminal` (should already be zeros).  
- Avoid reallocating `self._obs = zeros` every partial refresh — **in-place** zero skipped rows + write `idx` (less allocator traffic).  
- Ensure multiconfig env reuse doesn’t force full re-encode when only stacks reconfigure (already reconfigure path — verify no extra full refresh).

### R5 — Hot-spot encode math (only after R0–R2)

Inside `encode_obs_row` / numpy `encode_observation_batch`:

- Profile which blocks dominate (MC opp-outcome already capped at 384 for train; boards/stack/history).  
- Micro-opts in Rust (already Rayon per row).  
- **Do not** cut MC further without a quality gate.  
- **Do not** drop dims here (obs-v3 design is separate).

### R6 — Explicit non-goals for #3

- Cross-update rollout∥optimize  
- Changing OBS_DIM / feature definitions without design approval  
- Numba on the whole encoder if Rust path is the intended production path  
- “Approximate” equity for training without an A/B protocol  

---

## Implementation phases

### Phase 0 — Measure (short)

1. Add a one-line log in `BatchedBombPotEnv.__init__`:  
   `[obs-encoder] rust=… OBS_DIM=… opp_mc=…`  
2. Optional: wall timers for pack vs encode vs unpack under `PLO5BP_STEP_TIMERS` (or rely on existing `record_function` names in a tables-only profile).  
3. One 2-update profile on pod with production flags.

**Exit criterion:** know rust on/off and ~% pack vs encode.

### Phase 1 — Fix partial-mask double pack (main code change)

**Files:** `python/plo5bp/env_batched.py`, possibly `rust_engine/src/bindings.rs` (+ `obs_v7_inc.rs` only if encode API changes).

1. Refactor `_refresh` rust branch for partial `encode_mask`:
   - Single source of packed/aux data for full batch caches.  
   - Single encode pass for `idx` only.  
   - Zero non-`idx` obs rows (terminal convention).  
2. Prefer in-place `self._obs[idx] = …` and `self._obs[~idx] = 0` over full reallocate when shapes match.  
3. Keep `_refresh_subset` as the post-reset fast path (already good); ensure it doesn’t regress.

**Tests (local then pod smoke):**

```text
pytest tests/python/engine/test_refresh_encode_mask.py \
       tests/python/engine/test_refresh_subset_parity.py \
       tests/python/engine/test_encoding_rust.py \
       tests/python/engine/test_encoding_batch.py \
       tests/python/engine/test_env_batched.py -q
```

Bit-exact: kept rows == full encode; skipped rows == 0.

### Phase 2 — Rust path hygiene

1. Confirm `PLO5_RUST_ENCODER=1` + width gate 1171 on guardian/train.  
2. Fix stale comments/tests that claim force-disable.  
3. If any training path still hits numpy PLO encode, treat as regression.

### Phase 3 — Optional micro-opts (only if Phase 0 still shows encode-bound)

- Rayon chunk tuning / avoid extra copies in `encode_indexed`  
- Reduce Python dict churn in numpy fallback (not primary)  
- Dirty-row flags only if profiler shows redundant full encodes mid-hand  

### Phase 4 — Measure success

Same recipe as baseline profile, update 1:

| Metric | Baseline | Success |
|---|---|---|
| `step1a/refresh` | ~104 s | Clear drop (target **≥20–40%** on this bucket if double-pack fixed or rust fully used) |
| Rollout wall | ~425 s | Down in proportion |
| Obs parity tests | green | Required |
| Policy metrics | — | Noise only |

Append to `memory/rollout_saturation_profile_2026-07-13.md`.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Silent obs truncation (old 991/1020 vs 1171) | Keep width gate; never compare “first K dims only” in tests |
| Bit drift vs scalar encoder | Existing 3-way parity suite; run after every rust touch |
| Partial encode zeros wrong row | `test_refresh_encode_mask` |
| Subset ≠ full after reset | `test_refresh_subset_parity` |
| Maturin not rebuilt on pod | Phase 0 log + CI/docs; rebuild if methods missing |
| “Faster” by cutting MC | Forbidden without explicit A/B |

---

## Files to touch

| File | Likely change |
|---|---|
| `python/plo5bp/env_batched.py` | `_refresh` partial path; encoder log; in-place obs writes |
| `rust_engine/src/bindings.rs` | Optional pack-once / encode-from-pack API |
| `rust_engine/src/obs_v7_inc.rs` / obs_layout | Only if encode path changes |
| `tests/python/engine/test_encoding_rust.py` | Un-stale skip/comments; full 1171 parity |
| `tests/python/test_refresh_*.py` | Already cover mask/subset — keep green |
| `memory/rollout_saturation_profile_2026-07-13.md` | Results |
| Guardian already has `PLO5_RUST_ENCODER=1` | Verify; don’t remove |

---

## Suggested coding order

1. Phase 0 log + (optional) sub-timers  
2. Phase 1 partial-mask single-pack/encode fix  
3. Full encoding/refresh pytest suite  
4. Phase 2 comment/test hygiene + confirm pod rust=on  
5. Pod step-timer A/B  
6. Only then R5 micro-opts  

---

## Expected outcome (honest)

| If… | Then… |
|---|---|
| Partial path double-packs today | Fixing it can remove a large fraction of **pack** work on the common post-apply path (most steps have mixed terminal/non-terminal) |
| Rust was off by mistake | Re-enabling is a large win vs pure numpy |
| Rust already on + pack is small | Gains smaller; need encode micro-opts or accept #4 as next host win |
| Combined with #1/#2 | Shorter host pole → better dual util automatically |

**#3 alone** will not halve the update, but it is the largest **host FLOP** reduction left after scheduling work (#1/#2).

---

## Decision defaults

1. **Correctness first:** bit-exact obs on kept rows; zeros on skipped.  
2. **No MC cuts** without a separate experiment.  
3. **Measure Phase 0** before large rust refactors.  
4. Prefer **one pack + subset encode** over “encode everyone every time.”

---

## Ready to implement when approved

Same bar as #1/#2: green encode/refresh tests, memory note, no pod deploy until you say so.

---

## Update (user 2026-07-13): vSix4 **was** on the Rust encoder

Confirmed: production vSix4 ran with `PLO5_RUST_ENCODER=1` and the width gate
satisfied (`OBS_DIM == 1171`), so `_use_rust_encoder=True`.

### What this changes

| Item | Before | After confirmation |
|---|---|---|
| Phase 0 "is Rust on?" | Mandatory open question | **Closed** — treat Rust full-batch path as live |
| R3 re-enable Rust | Possible big win | **Mostly docs/test hygiene** (stale comments only) |
| **Primary lever** | Unclear | **R2: partial `encode_mask` double pack** |
| Full-batch encode | Maybe numpy | Already Rayon `observation_encoded_batch` |
| Expected win shape | Could be huge if rust-off | **Moderate–large** on the *common* post-apply path only |

### Why R2 still matters with Rust on

Post-apply refresh almost always uses a **partial** mask (`~newly_terminal`).
The Rust branch then does:

1. `observation_and_features_batch()` — pack (+ features) for **all** envs
2. `observation_encoded_subset_batch(idx)` — pack+encode **again** for kept rows

Full-mask and subset-after-reset paths are already the good one-FFI forms.
The plan's Phase 1 (single pack for caches + single encode for `idx`) remains
the main implementation target; Phase 0 shrinks to optional pack-vs-encode
split timing, not backend discovery.

### What does **not** change

- Bit-exact obs requirement
- No MC cuts without A/B
- `_refresh_subset` post-reset stays
- No cross-update stale self-play

