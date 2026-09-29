# Plan: Attack #4 — Faster end-of-hand bookkeeping (terminal / slabs)

Reference: `memory/rollout_saturation_profile_2026-07-13.md` baseline update-1 timers:

| Bucket | ~sec | ~% of rollout timers |
|---|---|---|
| **`step9d/slab_copies`** | **45** | **9.8%** |
| **`step9f/reset_terminal`** | **28** | **6.1%** |
| **`step9a/payouts`** | **20** | **4.3%** |
| `step9e/pool_mix` | 7 | 1.5% |
| `step9c/gae_scan` | 3 | 0.6% |
| `step9b/retroactive_bonus` | 1 | 0.2% |

**Not in scope:** cross-update stale self-play; changing GAE/VRPO math; lowering EV runout quality without A/B.

---

## Goal

Same finished-hand **training rows** (obs, actions, logp, values, advantages, returns, terminal flags) and same engine reset semantics, **less wall-clock** on the terminal flush path.

Keep: current learner this update → full rollout → then optimize.

---

## Current architecture (facts)

Terminal flush lives in `collect_rollout_batched` after apply + post-apply `_refresh`:

```text
if newly_terminal.any():
  step9a  payouts_batch / payouts_ev_batch     (Rust Rayon over N)
  step9b  retroactive bonus                     (vectorized T,S,L)
  step9c  GAE / VRPO scan                       (vectorized over T*S; Python loop over L)
  step9d  slab_copies                           (gather traj → output slabs at wcursor)
  step9e  pool_mix re-assign                    (Python for term_envs)
  step9f  reset_terminal_batch + holes_subset
          + _refresh_subset                     (Rust Rayon + #3 encode path)
```

### Data layout (why slabs are hot)

**During hand:**
- Learner steps append obs/gm into flat `step_obs_pool` / `step_gm_pool`
- Per `(env, seat, t)` metadata in `traj_*` / `costs_arr` / … with `MAX_STEPS_PER_SEAT=192`
- `traj_obs_idx[env,seat,t]` = absolute row in `step_obs_pool`

**On flush:**
- Build `active_step` mask `(T,S,L)` for learner seats with length > 0
- `sel = nonzero(active_step.ravel())`
- `np.take(step_obs_pool, obs_idx, …)` → `all_obs_arr[wcursor:end]` (**1171-wide**, random gather)
- Many `.ravel()[sel]` copies for gate/chips/sizing/logp/value/adv/…
- Rebuild rotated opp-holes from `holes_cache` then index

Multiconfig already writes into shared host slabs (`out_slabs`) — good; #4 is about **flush gather cost**, not finalize H2D (already separate).

### What is already good

- Vectorized GAE/VRPO over seats (not per-env Python walks)
- `L = lengths.max()` not full 192 for flush temps
- Rust Rayon on `payouts_*_batch` and `reset_terminal_batch`
- Post-reset `_refresh_subset` only (not full batch encode)
- #2 can overlap *next* act with this host work when flag on — #4 still reduces the host pole

---

## Root causes / levers (ordered)

### R1 — `slab_copies` gather (primary)

**Hot pattern:** indirect `np.take` of ~`n_new` rows × `obs_dim` (~1171) with irregular `obs_idx`, plus many small field gathers.

**Directions (prefer measure-guided):**

| ID | Idea | Risk | Notes |
|---|---|---|---|
| **S1** | Fuse multiple small field writes; avoid repeated `traj_*[term_envs,:,:L].ravel()[sel]` materializations — one index plan, reuse | Low | Pure Python/numpy reshape |
| **S2** | Precompute / cache **rotated opp-holes per (env,seat)** at hand start (or first learner step) instead of rebuild every flush | Low–med | Memory: `(n_envs, seats, 5, hole)` small vs obs pool |
| **S3** | Contiguous packing of active `(env,seat,t)` when possible (sort `sel` for better take locality) — **only if bit-exact row order not required in slab** | Med | Batch row order may not need env-major order if PPO shuffles; **must not** change within-seat time order for GAE (GAE already done before gather — slab order is free as long as each row is correct) |
| **S4** | Write obs **directly into slab** at learner step time (no `step_obs_pool` indirection) for simpler gather | High effort | Big redesign; pool exists to support variable-length flush |
| **S5** | Rust/Cython gather kernel for obs+fields given `obs_idx` + traj slices | Med | Only if S1–S3 insufficient |

**Default plan:** implement **S1 + S2** first; consider **S3** (sort by `obs_idx` for take locality) if safe; defer S4/S5.

**Note on S3:** After GAE, each output row is independent. Reordering rows in the batch does not change learning if advantages/returns already attached. Prefer stable documentation + optional flag if worried about debug reproducibility.

### R2 — `pool_mix` Python loop (easy)

```text
for i_int in term_envs.tolist():
    _assign_pool_mix(int(i_int))
```

**Target:** vectorize or batch RNG for pool assignment where possible; at least avoid Python per-env when `pool_opp_seats==0` or empty pool (fast path already partial in `_assign_pool_mix`).

**Risk:** RNG stream order for seat assignment — keep **same sequence** as sequential `term_envs` order for bit-exact pool mixes, or document intentional change.

### R3 — `payouts` / `reset` (secondary)

Already Rayon. Possible polish:

| ID | Idea |
|---|---|
| **P1** | `payouts_batch` only for `term_envs` compact rows then scatter (less work when few terminals) |
| **P2** | Fuse `payouts + won = payouts + total_commit` on Rust side for terminal mask only |
| **P3** | `reset_terminal_batch` already parallel — avoid extra host copies of seeds/mask if any |

Only after S1/S2 if timers still show 9a/9f large.

### R4 — GAE loop over `L` (low priority)

~3s baseline. Python `for t in range(L-1,-1,-1)` over large `T*S` arrays — already numpy-heavy. Optional numba/Rust later; not first.

### R5 — Interaction with #2 / #3

- Do **not** break prefetch contract (terminal flush must still complete before Wave B).
- Faster flush improves #2 overlap quality (shorter host side).
- `_refresh_subset` cost sits inside 9f timer — already improved by #3; don't re-litigate encode here except if 9f still dominated by refresh.

### R6 — Non-goals

- Stale self-play pipeline  
- Reducing `ev_runout_samples` without A/B  
- Changing advantage estimator  
- Dropping traj fields  

---

## Implementation phases

### Phase 0 — Sub-timers (small)

Under `PLO5BP_STEP_TIMERS`, split `step9d` if useful:

- `step9d_obs_take` vs `step9d_meta` vs `step9d_opp_holes`

Or one profiled update after S1 to see residual. Optional if S1/S2 are clearly the copies.

### Phase 1 — Slab gather cleanup (main)

**File:** `python/plo5bp/rollout.py` (`step9d` block)

1. Build `sel` / `obs_idx` once; reuse for all fields.  
2. Minimize temporary `(T,S,L)` ravel copies — index once into flat views where dtype allows.  
3. **Opp-holes:** compute `rot_block` only for `term_envs` once; consider caching at deal time in `holes_rot_cache[env, seat]` filled on reset / first step.  
4. Ensure `wcursor` / overflow guards unchanged.  
5. Preserve **per-row** correctness (each slab row = same content as today). Row **order** may be documented as unspecified if we sort for locality (flag or comment).

**Tests:**

```text
pytest tests/python/training/test_rollout_parity.py \
       tests/python/training/test_rollout_batched.py \
       tests/python/training/test_multiconfig_staging.py \
       tests/python/training/test_vrpo_advantage.py -q
```

If order changes: add test that compares **sorted-by-hash** or multiset of rows, or keep order bit-exact for first PR (prefer **bit-exact order** first, locality sort as opt-in).

**Recommendation:** first PR **bit-exact** (same `sel` order as today); second PR optional sort-by-`obs_idx`.

### Phase 2 — pool_mix

- Fast path when no pool / zero opp seats.  
- If pool active: keep sequential assignment order for RNG; maybe pre-draw randoms in a vector then apply.

### Phase 3 — payouts compact (optional)

- Add or use terminal-index payout API if profiling still shows 9a large with few terminals.  
- Training often uses EV path (`ev_runout_samples`); keep seeds/determinism.

### Phase 4 — Measure

Pod, production recipe, `PLO5BP_STEP_TIMERS=1`, update 1:

| Metric | Baseline | Success |
|---|---|---|
| `step9d/slab_copies` | ~45s | Clear drop (target ≥25–40% on bucket) |
| `step9f` / `step9a` | ~28s / ~20s | Drop if Phase 2–3 landed |
| Rollout wall | ~425s | Down in proportion |
| Batch contents | — | Parity tests green |

Append to `memory/rollout_saturation_profile_2026-07-13.md`.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Wrong slab row (obs≠action) | Keep single `sel`/`obs_idx` plan; parity tests |
| Advantage mismatch | Don't touch GAE inputs; only gather after scan |
| Pool RNG drift | Preserve `term_envs` order for `_assign_pool_mix` |
| Multiconfig staging | Same `out_slabs` views; overflow guards |
| Memory regression | Opp-hole cache is small; don't grow MAX_STEPS temps |

---

## Files to touch

| File | Change |
|---|---|
| `python/plo5bp/rollout.py` | step9d gather; optional opp-hole cache; pool_mix; maybe payouts call site |
| `rust_engine/...` | Only if Phase 3 compact payouts |
| `tests/python/test_rollout_*.py`, `test_vrpo_advantage.py`, `test_multiconfig_staging.py` | Green |
| `memory/rollout_saturation_profile_2026-07-13.md` | Results |

---

## Suggested coding order

1. Phase 1 slab gather **bit-exact** (S1 + S2)  
2. Full rollout/VRPO/multiconfig tests  
3. Phase 2 pool_mix  
4. Pod timers  
5. Phase 3 payouts only if still hot  
6. Optional S3 locality sort behind clarity on row order  

---

## Expected outcome (honest)

| Work | Plausible effect |
|---|---|
| S1+S2 on slab_copies | Largest #4 win (~tens of seconds if take/opp-holes dominate) |
| pool_mix | Small |
| payouts compact | Small–medium when sparse terminals |
| #4 alone | Solid chunk of the ~20% terminal group; not half update |
| With #1–#3 | Compound; dual util improves as host pole shrinks |

---

## Decision defaults

1. **Bit-exact rows first**; locality reorder only as follow-up.  
2. **No EV sample cuts.**  
3. Prefer numpy gather cleanup over big “write obs straight to slab” redesign (S4).  
4. Measure after Phase 1 before Rust payouts work.

---

## Ready to implement when approved

Same bar as #1–#3: tests green, memory note, no pod deploy until you say so.
