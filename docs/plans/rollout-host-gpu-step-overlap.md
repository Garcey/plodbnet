# Plan: Attack #2 — Overlap host work with GPU (same update, same weights)

Reference: `memory/rollout_saturation_profile_2026-07-13.md`, attack list #2.  
**Not in scope:** pipelining update *u+1* rollout against update *u* optimize (rejected design).

---

## Goal

Reduce **idle waiting inside one update’s rollout** so CPU and GPU can be busy at the same time when their work does not depend on each other — **without** changing:

- which seats are learner vs pool  
- which weights the learner uses this update  
- `act()` sampling order / bit-exact trajectories  
- “rollout fully, then `trainer.update`”

---

## Current step (facts)

One env-step in `collect_rollout_batched` is roughly:

```text
read env caches (obs, masks, actors, dones)
classify learner vs opp + sizing_step
learner:  H2D → act → blocking D2H          [step2/3/5]
critic:   forward (needs o_t + often actions for VRPO)
opp:      H2D→act (×K, deferred D2H)        [step4a/4b]  ← #1 done
traj write (needs actions + values + obs)
apply_hybrid                                [~2%]
_refresh (encode)                           [~23%]
aggression bookkeeping
if terminals: payouts / GAE / slab_copies / pool_mix / reset / subset refresh  [~20%]
```

**Hard dependencies (cannot break):**

| Before | Needs |
|---|---|
| Next `act` | Fresh `obs` / masks / actors / dones for those envs |
| `apply` | Final gates/chips for **all** acting envs this step |
| Traj store | Learner actions + values (+ VRPO Q) |
| Terminal GAE/slabs | Post-apply commits, payouts, traj rows for finished hands |
| Re-dealt tables acting | `reset_terminal` + their encode **before** they can be in an act batch |

**Implication:** You cannot start a full next-step act for *re-dealt* tables until terminal reset finishes. You *can* act **non-terminal** tables using post-apply refresh while the host finishes terminal flush — that is the main structural overlap.

Baseline share (update 1 timers): inference wall ~50%, refresh ~23%, terminal/slabs ~20%, apply ~2%.

---

## Target behavior (after)

Same answers, better schedule:

1. **Intra-step:** queue as much device work as possible before any blocking D2H; do pure-host prep that does not need actions while GPU runs.  
2. **Inter-phase (main win):** after `apply` + non-terminal `refresh`, run **GPU act wave for still-active (non-done) envs** in parallel with **host terminal flush + reset + terminal encode**.  
3. Then a **small second act wave** only for envs that became live again via reset (if any need to act immediately), merge into `gates_per_env` / `chips_per_env`, continue.

```text
TODAY (serial):
  act_all → D2H → traj → apply → refresh → terminal_host → (repeat)

AFTER (overlap):
  act_all → D2H → traj → apply → refresh_nonterm
       ┌─────────────────────────────┐
       │ GPU: act wave A (active)    │
       │ CPU: terminal flush+reset   │
       └─────────────────────────────┘
  GPU: act wave B (re-dealt only, if needed)
  merge → traj pieces already done for wave A; handle B → apply → …
```

Still **one update**, **current learner**, then optimize.

---

## Implementation phases

### Phase 0 — Instrumentation (small, first)

Keep `PLO5BP_STEP_TIMERS=1` honest:

- Split timers so overlap is visible, e.g.  
  - `step2x/overlap_terminal_host`  
  - `step2x/act_wave_active`  
  - `step2x/act_wave_redealt`  
- Optional: count “host-only / gpu-only / both” wall with a simple side clock (not required if timers + resource sampler suffice).

No behavior change.

### Phase 1 — Intra-step device coalescing (safe, extends #1)

**File:** `python/plo5bp/rollout.py`

1. **Learner device path** (mirror opp):  
   - `_learner_upload_act` → H2D + `act` (+ optional marginal) **on device**  
   - Critic runs on device using `o_t`  
   - **Do not** `.cpu()` until after opp acts are queued (or immediately before traj write in one coalesced pull)

2. **Host work during learner GPU compute** (only if free):  
   - Build opp group lists / `sd_ids` (needs only masks, not actions)  
   - `_rotate_opp_holes_batch` for critic can run on CPU while learner `act` runs **if** critic is sequenced after that completes (holes prep overlaps learner act; critic still after act)

3. **Single/few D2H boundary** per env-step for all numpy needed by traj + apply.

4. **Preserve order:** learner `act` before opp `act`s (CUDA RNG / current semantics). Opp order still `unique(sd)`.

**Risk:** low if D2H values match today’s stacks.  
**Expected win:** modest (sync/latency); dual util up a bit during the act block.

### Phase 2 — Dual-wave act + terminal overlap (**core #2**)

**File:** `python/plo5bp/rollout.py` (hot loop restructuring)

**After** traj write + `apply_hybrid` + non-terminal-aware `_refresh` (same as today):

1. Snapshot what terminal flush needs from env **now** (commits, seeds, traj lengths, etc.) — same data as today’s terminal block.  
2. **Wave A (GPU):** classify actors among `~newly_terminal` (active envs only); run learner/opp/critic act path for those envs only; keep results on device or in side buffers keyed by env index.  
3. **Parallel host:** existing `step9a–9f` terminal path (payouts, retro, GAE, slab_copies, pool_mix, `reset_terminal`, subset refresh).  
   - Prefer **same thread, interleaved** only if we use async GPU: queue Wave A device work **without** D2H, run terminal host, then D2H Wave A.  
   - That is true overlap without Python threads:  
     `queue act wave A (no D2H) → terminal host CPU → D2H wave A`.  
4. **Wave B (GPU):** envs that were terminal and are now live after reset and have `actor >= 0`; act only those; D2H; merge into `gates_per_env` / `chips_per_env` / traj for those learner seats.  
5. Continue loop: next iteration’s “pre-step” state is already post-refresh; **careful** not to double-apply.

**Critical correctness rules:**

- Wave A must **not** include `newly_terminal` envs (they are done until reset).  
- Wave B only after reset + encode for those envs.  
- Traj for Wave A learner decisions must use the obs that Wave A acted on (post-step-t refresh), and must not clobber unfinished terminal flush indexing — use the same traj slot machinery as today, just possibly two writes per outer iteration.  
- `wcursor` / rollout_target accounting must count both waves’ learner steps.  
- Short-shove remap still applied before **each** apply (only one apply per outer iteration — see below).

**Apply timing choice (pick one, document in code):**

| Option | Behavior |
|---|---|
| **A (recommended)** | Outer loop still: act(full) → apply once. Overlap is **only** “queue next act while finishing previous terminal work” by restructuring so “full act” = wave A + wave B **before** apply. Terminal work from **previous** apply runs while wave A of **current** act is on GPU. |
| **B** | apply mid-loop as now; dual-wave is “prep next act” across iterations (more state machine). |

**Recommended Option A state machine:**

```text
# Invariant at loop head: env state is "ready to act" (all live tables encoded)

act_all (learner+opp+critic) for current ready state   # may be dual-wave only if we split ready set — actually at loop head all ready
D2H, traj, apply, refresh
queue device act for active-after-apply (wave A) without D2H
terminal host flush+reset+encode
D2H wave A
wave B act for re-dealt needing action
D2H wave B, traj for A/B learner rows
# Now we have actions for the "next" decision — either apply immediately
# or set as "pending" for top of loop

```

Cleaner **Option A'** (simplest dual-wave):

At loop head state is ready-to-act (today’s invariant).

```text
1. act complete (as today, with Phase 1 coalescing)
2. traj + apply + refresh
3. if any terminal:
     queue wave_A_act(active) on GPU (no D2H)
     run terminal host (reset+encode)
     D2H wave_A; store actions in pending_*
     wave_B_act(redealt_active); D2H; merge into pending_*
     traj write for pending learner rows
     # pending actions are for the NEW decisions — apply them next:
     apply(pending); refresh; ... 
```

That changes the loop to sometimes do **two applies per “iteration”** or folds into “end of iteration prepares next actions.”

**Simplest correct design to implement:**

**End-of-step prefetch:**

```text
while wcursor < target:
  # Enter with ready obs (and optional prefetched device acts from last iter)
  if not prefetch:
    act + D2H as today (Phase 1)
  else:
    D2H prefetch + fill gates/chips/values  # already computed

  traj; apply; refresh

  # Prefetch next act:
  if newly_terminal.any():
    start = queue act on ~newly_terminal (device only)
    terminal_host()  # reset+encode terminals
    # now all envs ready; if wave only had non-term, queue act on re-dealt actives
    finish remaining acts; keep on device as prefetch for next iter
  else:
    queue full act on device as prefetch (no terminal host to overlap)
    # optional: nothing to overlap — D2H immediately or keep prefetch
```

When there are **no** terminals, overlap is weak (Phase 1 only).  
When terminals are common (deep multiway), host ~20% overlaps with GPU act ~50% of non-term tables — real win.

### Phase 3 — Tests

| Test | Why |
|---|---|
| `test_rollout_parity.py` | serial vs batched actions/values |
| `test_rollout_batched.py` / `test_batched_rollout.py` | shapes / smoke |
| `test_multiconfig_staging.py` | bit-exact staging vs legacy |
| `test_multiconfig_rollout.py` | mix path |
| New focused test | Fixed seed, pool mix on, many short hands: compare Batch (or gates/chips stream) before/after #2 with flag off/on |

**Feature flag:** `PLO5BP_ROLLOUT_OVERLAP=1` (default **on** once green, or default off until pod A/B — recommend **default on** only after parity tests pass; until then env-gated).

### Phase 4 — Measure

Pod, same recipe, `PLO5BP_STEP_TIMERS=1`, 2 updates, trust update 1:

| Metric | Baseline | Success |
|---|---|---|
| Rollout wall `[phase]` | ~425 s | Clear drop |
| `step1a/refresh` + terminal timers | ~100 s + ~95 s | Less **exposed** wall (may same CPU time but overlapped) |
| Rollout GPU util | ~11% | Up during former host-only stretches |
| Resource CPU of 40.8 | ~21% | Up or stable |
| Policy metrics | — | Noise only |

Append results to `memory/rollout_saturation_profile_2026-07-13.md`.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Double-apply / missed apply | Single pending-action buffer; assert every active env got a gate before apply |
| Traj slot / wcursor double-count | One write path; waves only add rows once |
| Terminal env acted before reset | Wave A mask `~newly_terminal`; Wave B only after reset |
| RNG order change | Keep learner-then-opp, unique(sd) order; document wave B as extra acts in table order |
| Prefetch stale obs | Prefetch only after refresh that produced that obs; never reuse across apply |
| VRPO critic needs actions | Critic after learner act on device; D2H jointly |
| Harder debugging | Env flag to disable overlap; step timers for waves |

---

## Files to touch

| File | Change |
|---|---|
| `python/plo5bp/rollout.py` | Phase 1 coalescing; Phase 2 prefetch / dual-wave; timers; flag |
| `tests/python/test_rollout_*.py` / multiconfig | parity + optional new test |
| `memory/rollout_saturation_profile_2026-07-13.md` | post-#2 results |

No train.py algorithm changes. No pool semantics changes.

---

## Suggested coding order

1. Phase 0 timers (names only)  
2. Phase 1 learner device path + deferred D2H + host prep during learner act  
3. Pytest  
4. Phase 2 prefetch: queue next act after apply/refresh; run terminal host before D2H of that act; wave B for re-dealt  
5. Pytest + flag  
6. Pod step-timer A/B  

---

## Expected outcome (honest)

| Piece | Plausible effect |
|---|---|
| Phase 1 only | Small wall win; cleaner GPU queue |
| Phase 2 with frequent terminals | **Meaningful** fraction of min(GPU act, terminal host) removed from wall — maybe **~10–20%** rollout if overlap is good |
| #2 alone | Not half update; stacks with #1 |
| Dual util | Should rise in rollout (the point of #2) |

Still later: #3 refresh speed, #4 slab/terminal speed (reduce work, not only hide it).

---

## Explicit non-goals

- Rollout(u+1) concurrent with optim(u)  
- Changing entropy / PPO / pool sampling  
- Multi-process env workers (future structural option, not #2)  
- Multi-stream different models (optional later)

---

## Decision defaults (unless you override)

1. **Option:** end-of-step **prefetch next act** + terminal host under in-flight GPU (no extra Python threads).  
2. **Flag:** `PLO5BP_ROLLOUT_OVERLAP` for kill-switch during bring-up.  
3. **Parity first**, then pod A/B, then leave on.

---

## Ready to implement when approved

Same bar as #1: green parity/multiconfig tests, memory note, no deploy until you say so.
