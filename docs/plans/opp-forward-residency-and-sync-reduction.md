# Plan: Attack #1 — Opponent forwards (~23% of rollout)

Reference: `memory/rollout_saturation_profile_2026-07-13.md`.

## Current code (facts)

**Already done (do not re-do):**
- Pool stores **CPU** `state_dict`s only (`OpponentPool.snapshot` → `.cpu()`).
- Multiconfig builds each frozen actor **once per update** via `snapshot_cache` + `_get_snapshot_model` → `_build_frozen_model(...).to(device)` — models stay on GPU for the whole update (~8 × small; comment says ~0.5 GB). Lifetime ends with the multiconfig call.
- Opp seats are grouped by snapshot index; one `model.act` per group per env-step.

**Where the time actually goes:**
```text
# collect_rollout_batched hot loop
learner:  _forward(learner, learner_idx, ...)     # H2D → act → blocking D2H
critic:   (learner only)
opps:     for sd_idx in unique(active opp snapshots):
              _forward(m_sd, group, ...)          # SAME path: H2D → act → blocking D2H
apply / refresh / terminal...
```

`_forward` (`rollout.py` ~1440) uses **one** shared `_PinnedStepH2D` buffer and ends every call with **blocking** `.cpu().numpy()` (coalesced ints + floats). That **forces a full CUDA sync** before the next group can safely reuse the staging buffer.

So each env-step pays roughly:
- 1× learner sync round-trip  
- **K× opp sync round-trips** (K ≈ distinct snapshots acting that step; profile ≈ **~6 groups/step** given `step3` n≈52k vs `refresh` n≈7.4k)

**Timer caveat (fix while implementing):**  
`step3/learner_forward` is nested inside both learner and opp `_forward`s, and `step4/opp_forwards` wraps the whole opp loop — nested wall is **double-counted** in `total_accounted`. True story:
- `step4` ≈ **106 s** = full opp-loop wall (real #1 target)
- Most of the 111 s under `step3` is **opp** `act`s, not learner-only  
Split timers as part of this work so the next profile is trustworthy.

**What #1 is *not*:** models are not being parked to CPU every step today. “Keep them on deck” is already true. The win is **fewer/later syncs and less serial queueing around opp groups**.

---

## Target behavior (after)

Same poker logic and **same `act()` call order** (CUDA RNG bit-exact), but:

1. For each env-step’s opponent phase: run **all** snapshot-group `act()`s (and their H2Ds) **before** any blocking D2H for those groups.
2. Then **one** (or few) host sync(s) to pull gates/chips for every opp group.
3. Optional small win: dedicated opp staging so we never contend with learner’s single buffer mid-step.
4. Clear timers: `step3a/learner_act` vs `step3b/opp_act` vs `step4/opp_loop` (loop overhead only) or similar.

**Before:**  
`opp A: H2D→act→SYNC → opp B: H2D→act→SYNC → …` (GPU idle between groups)

**After:**  
`opp A: H2D→act → opp B: H2D→act → … → SYNC once → scatter gates/chips`  
(GPU can stay in the compute queue; host waits once)

---

## Implementation phases

### Phase 1 — Split `_forward` / opp path (core win)

**File:** `python/plo5bp/rollout.py`

1. **Refactor `_forward` into:**
   - `_forward_device(...)` → H2D + `act` → returns **device** tensors (gate, chips, …) **without** `.cpu()`.
   - `_actions_to_host(...)` → stacked coalesced D2H (existing int/float stack pattern) → numpy.
   - Keep a thin `_forward(...)` = device + host for learner / serial callers so existing call sites stay simple.

2. **Opp loop rewrite** (`step4/opp_forwards` ~1614):
   ```text
   groups = []  # (sd_idx, env_idx array)
   for sd_idx in unique(...):
       groups.append(...)
   device_outs = []
   for sd_idx, group in groups:
       m = _get_snapshot_model(sd_idx)
       device_outs.append(_forward_device(m, group, ..., want_marginal=False))
   # single sync boundary:
   for (group, dev_out) in zip(...):
       g_np, c_np, ... = _actions_to_host(dev_out)
       gates_per_env[group] = ...
       chips_per_env[group] = ...
   ```
   - **Preserve loop order** of `np.unique` (or sort explicitly the same way) so RNG order is unchanged.
   - Opp still `want_marginal=False` (no extra work).

3. **Staging safety:**
   - **Minimum:** after device-only acts, do **not** refill `_PinnedStepH2D` until all opp H2Ds for this step are done — if one buffer, H2D must still be sequential **but** compute can queue; **D2H sync only at end**.
   - **Better (recommended):** small **double buffer** for step H2D (2 × `num_envs` rows) or “upload into per-call device tensors via `non_blocking` from pinned, keep two host slots” so group *i+1* H2D can overlap group *i* compute. Document non-overlap rule like existing `_PinnedStepH2D` docstring.
   - Learner path can keep today’s single-buffer + immediate D2H for the first patch (smaller diff); optional follow-up: learner device-out → critic reuse of `o_t` already exists (P9).

4. **Eager GPU residency (belt-and-suspenders):**
   - At start of `collect_rollout_batched` / multiconfig, after pool is known: optionally `for i in range(len(pool.snapshots)): _get_snapshot_model(i)` so first steps don’t pay build latency mid-hand. Multiconfig already builds on first assign; make this explicit + one-line log under step timers if useful.
   - **Do not** introduce a cross-update cache (explicitly rejected in P5 comments).

5. **Timers:**
   - Rename/split so nest double-count dies:
     - `step3/learner_forward` only around learner act  
     - `step4a/opp_act` around opp device acts  
     - `step4b/opp_d2h` around coalesced host pull  
     - `step4/opp_forwards` only if needed as parent (prefer non-overlapping names for `%` sum)

### Phase 2 — Tests (correctness gate)

Bit-exact / parity is non-negotiable.

| Test | Why |
|---|---|
| `tests/python/training/test_rollout_parity.py` | serial vs batched actions |
| `tests/python/training/test_rollout_batched.py` / `test_batched_rollout.py` | batched smoke + shapes |
| `tests/python/training/test_multiconfig_staging.py` | multiconfig shared staging bit-exact vs legacy |
| `tests/python/training/test_multiconfig_rollout.py` | mix path |
| Pool warmstart if touched | `test_warmstart_pool.py` (likely untouched) |

Add a **focused unit test** if easy:
- Tiny synthetic: 2 snapshots, fixed seed, capture `gates_per_env` / chips for N steps before vs after opp D2H deferral (or assert identical Batch fields on CPU).

Run (pod or local CPU as appropriate):
```text
.venv/Scripts/python -m pytest tests/python/training/test_rollout_parity.py tests/python/training/test_rollout_batched.py tests/python/training/test_multiconfig_staging.py tests/python/training/test_multiconfig_rollout.py -q
```

### Phase 3 — Measure (must beat baseline)

On pod, **same recipe as profile**, `PLO5BP_STEP_TIMERS=1`, **2 updates**, compare **update 1**:

| Metric | Baseline (2026-07-13 u1) | Success |
|---|---|---|
| `step4` / opp wall | ~106 s | **Clearly down** (target ≥15–25% on this bucket) |
| Rollout wall `[phase]` | ~425 s | Down in proportion |
| Rollout GPU util (1 Hz) | ~11% mean | **Up** (more continuous compute) |
| Training numbers | — | No intentional change; KL/H in the noise |

Artifacts: append short note to `memory/rollout_saturation_profile_2026-07-13.md` (“post-#1 results”).

### Phase 4 — Out of scope for #1 (do later)

- CUDA multi-stream opp groups (harder reasoning, marginal if single sync already queues well)
- Double-buffer **encode vs GPU** (attack list #2)
- `refresh` / `slab_copies` (attacks #3–4)
- Changing pool sampling, `pool_opp_seats`, or num snapshots

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| RNG / action bit drift | Keep **identical order** of `model.act` calls; only move D2H later |
| Staging buffer overwrite | Double buffer or “all H2D then no refill until D2H”; document invariant |
| VRAM | No extra full models; only small staging / transient act outputs |
| Timer confusion | Split non-overlapping names in same PR |
| Multiconfig cache lifetime | Keep `snapshot_cache` per-update only; no cross-update |

---

## Files to touch

| File | Change |
|---|---|
| `python/plo5bp/rollout.py` | Split `_forward`, rewrite opp loop, staging, timers |
| `tests/python/test_*.py` | Run existing; add small parity test if needed |
| `memory/rollout_saturation_profile_2026-07-13.md` | Record before/after after pod A/B |

No `train.py` / guardian changes required for the optimization itself.

---

## Suggested implementation order (coding)

1. Timer split only (no behavior change) — confirm next profile reads cleanly  
2. `_forward_device` + `_actions_to_host`; learner still sync-immediate  
3. Opp loop: multi-act then one D2H phase  
4. Double-buffer H2D if profiling shows H2D still serial-bound  
5. Pytest suite  
6. Pod step-timer A/B vs baseline table  

---

## Expected outcome (honest)

- **Direct:** cut a large fraction of the ~106 s opp-loop wall (sync tax + better GPU queueing).  
- **Update-level:** maybe **~5–12%** faster rollout if sync was a big share of the 106 s; more if double-buffer overlaps H2D with compute.  
- **Util:** higher GPU % during rollout; CPU may look similar or slightly busier preparing next group.  
- **Not** half the update by itself — still need #2–#4 for host encode/terminal.

## Decision needed before coding

Default recommendation: **Phase 1 as above (defer opp D2H + optional double-buffer)**.  
Skip “move models to GPU” as a feature — already true; only document it.
