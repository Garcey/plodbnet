# Rollout saturation profile findings (2026-07-13)

Persistent notes for attacking low CPU/GPU util during observation/rollout.
Re-read this after context compaction before implementing patches.

## Environment measured

- **Stem**: vSix4 (PLO5 double-bomb, v6 mixture + VRPO)
- **Pod**: RunPod, **cgroup quota = 40.8 vCPUs**, GPU not shared (~98 GiB VRAM)
- **Recipe**: `num_envs=49134`, `rollout_length=9_000_000`, mix-configs 10×3 tiers,
  `--cpu-threads 32`, `PLO5_RUST_ENCODER=1`, batched CUDA
- **Checkpoint at profile**: `checkpoints/vSix4.pt` global **u90** (resume from here, no LR ramp)
- **Do not** promote `checkpoints/vSix4_profile.pt` as production

## How we measured (what works / what doesn't)

| Method | Result |
|---|---|
| RunPod telemetry CPU % | **% of 40.8 vCPU allocation**, not of 384 host cores |
| 1 Hz resource sampler (`_ResourceSampler` in `scripts/train.py`) | Confirmed util + VRAM; JSONL under `runs/profile_resources_*.jsonl` |
| `[phase]` wall timers | rollout vs optimize split |
| Full `torch.profiler` + chrome export at 9M rows | **FAILS** — multi-GB trace hang; `key_averages` can balloon to ~100–160 GiB RSS and never finish |
| Lightweight step timers | **WORKS** — `PLO5BP_STEP_TIMERS=1`, `_TimedRF` / `_StepTimers` in `python/plo5bp/rollout.py` |

**Supported profile recipe:**

```bash
export PLO5BP_STEP_TIMERS=1
export PLO5_RUST_ENCODER=1
# same vSix4 train flags, --num-updates 2, NO --profile-one-update
# (or profile-one-update only with STEP_TIMERS so torch.profiler is skipped)
# Log: runs/vSix4_profile_steps.log
```

Chrome trace only if `PLO5BP_CHROME_TRACE=1` (not recommended at full rollout).

## Phase split (trust update 1 — steady state)

| Phase | Wall | Share of update |
|---|---|---|
| **Rollout** | **~425 s** | **~96%** |
| Optimize (PPO) | **~19 s** | **~4%** |
| **Total** | **~444 s** | |

Update 0 similar: rollout ~431 s (93%), optimize ~33 s (7%).

### Resource util (1 Hz, of **40.8 vCPU quota**)

| Phase | CPU mean (of 40.8) | GPU mean | Notes |
|---|---|---|---|
| **Rollout** | **~21–24%** (~9 vCPU) | **~8–12%** | Matches user telemetry 20–26% / 8–11% |
| **Optimize** | **~2%** | **p50 ~98%, p95 100%** | Already saturated GPU |

High VRAM during rollout (~40–82 GiB reserved) is **cached/reserved batch memory**, not SM busy work.

## Rollout step-timer breakdown (update 1, accounted ~464 s)

Source: `runs/vSix4_profile_steps.log` — `===== step timers (collect_rollout_multiconfig) =====`

| Rank | Bucket | Sec | % of timers | ms/call | n |
|---|---|---|---|---|---|
| 1 | **`step3/learner_forward`** | **111.0** | **23.9%** | 2.14 | 51926 |
| 2 | **`step4/opp_forwards`** | **106.4** | **22.9%** | 14.34 | 7418 |
| 3 | **`step1a/refresh`** | **104.4** | **22.5%** | 14.08 | 7418 |
| 4 | **`step9d/slab_copies`** | **45.3** | **9.8%** | 6.19 | 7318 |
| 5 | **`step9f/reset_terminal`** | **28.3** | **6.1%** | 3.86 | 7318 |
| 6 | **`step9a/payouts`** | **20.2** | **4.3%** | 2.76 | 7318 |
| 7 | `step6+7/rust_apply` | 9.5 | 2.0% | 1.28 | 7418 |
| 8 | `step3b/critic_forward` | 9.4 | 2.0% | 1.27 | 7418 |
| 9 | `step2/learner_h2d` | 8.3 | 1.8% | 0.16 | 51926 |
| 10 | `step9e/pool_mix` | 6.8 | 1.5% | 0.93 | 7318 |
| 11 | `step5/action_d2h` | 5.6 | 1.2% | 0.11 | 51926 |
| 12 | `step8/aggression_bonus` | 5.0 | 1.1% | 0.68 | 7418 |
| 13 | `step9c/gae_scan` | 2.9 | 0.6% | 0.40 | 7318 |
| 14 | `step9b/retroactive_bonus` | 1.0 | 0.2% | 0.14 | 7318 |

### Grouped

| Group | ~Share | Meaning |
|---|---|---|
| **Inference wall** (learner + opp + critic + H2D/D2H) | **~52%** | Real GPU work, but **serial bursts** → low average GPU util |
| **Obs encode** (`refresh`) | **~23%** | Host long pole #1 (pure CPU/Rust) |
| **Terminal path** (slab_copies + reset + payouts + GAE) | **~21%** | Host long pole #2 |
| **Apply / misc** | **~5%** | Already fine |

## Root cause (not “nothing is the bottleneck”)

Single-process **lockstep** rollout:

```
host classify → H2D → GPU act → blocking D2H → (opp groups serial)
→ host traj write → rust apply → refresh encode → terminal bookkeeping → repeat
```

- Wall time ≈ **sum** of stages, not max(cpu, gpu)
- GPU idle while host encodes / copies / resets
- Host idle (or under-fed) while waiting on CUDA sync after each forward
- Opponent pool: **one sequential GPU forward group per snapshot** per step
- `--cpu-threads 32` is intentional (NUMA; 192 threads made `_concat_batches` ~5× slower)

**Implication:** Filling GPU bubbles alone cannot cut update time in half. Half wall-clock requires making the **host path** much faster **and/or** overlapping host prep with GPU so wall ≈ max(host, gpu).

## Ranked attack list (implement in this order)

### 1. Opponent forwards (~23%) — best dual-util lever
- Keep pool models on GPU for whole rollout (avoid park/unpark thrash)
- Larger batches / fewer sequential snapshot groups
- Optional multi-stream for independent snapshots
- **Expected:** lower wall + higher GPU duty cycle during the step

### 2. Overlap host prep with GPU (learner ~24% + refresh ~23%)
- Double-buffer pinned staging (`_PinnedStepH2D` is single-buffer + blocking D2H by design today)
- While GPU runs step *t*, CPU prepares encode/classify for *t+1*
- Defer blocking `.cpu().numpy()` until host has no more independent work
- **Risk:** correctness / bit-exact parity; needs tests
- **Expected:** both util meters rise; wall ≈ max not sum

### 3. Faster `refresh` (~23%)
- Already `PLO5_RUST_ENCODER=1` and still ~14 ms × ~7.4k calls
- Optimize encode hot path; maximize subset-only encode; Rayon toward ~40.8 quota if encode-bound
- Do **not** blindly raise torch intra-op threads above 32

### 4. Slab / terminal path (~20% combined)
- `step9d/slab_copies` — pure host memcpy; fewer copies, pin, write final layout once
- Batch payouts / reset_terminal further if possible

### 5. Explicitly do **not** start with
- Raising `--cpu-threads` to 40+ for torch
- Cutting `num_envs` (same total rows → more steps → more sync tax)
- Full chrome `torch.profiler` on 9M-step updates
- Assuming high VRAM means GPU is working hard

## Realistic speedup expectations

| Change | Plausible wall win | Dual util? |
|---|---|---|
| Opp-forward residency/batching | meaningful on ~23% | GPU util up |
| Perfect host/GPU overlap only | ~5–15% if GPU was only idle wait | Yes |
| Host refresh 30% faster | ~7% of update | CPU up |
| Host 30% + opp + overlap | **~30–40%** update plausible | Yes |
| “Saturate both → half update” | Only if host path ~2× faster overall | Not automatic |
| Multi-process collectors + inference server | 50%+ possible | Design goal; large rewrite |

## Code touchpoints

| Piece | Path |
|---|---|
| Phase timers + resource sampler | `scripts/train.py` (`_ResourceSampler`, `[phase]`, chrome skip) |
| Step timers | `python/plo5bp/rollout.py` (`_StepTimers`, `_TimedRF`, `PLO5BP_STEP_TIMERS`) |
| Pinned step H2D (single buffer, blocking D2H) | `python/plo5bp/rollout.py` (`_PinnedStepH2D`) |
| Rollout loop buckets | `collect_rollout_batched` / multiconfig |
| Guardian / resume | `scripts/vSix4_guardian.sh`; stop via `runs/vSix4.stop` |
| Resume recipe | warm `checkpoints/vSix4.pt`, `--lr-warmup-updates 0` |

## Artifacts on pod (may age)

- `runs/vSix4_profile_steps.log` — step timer tables + phase lines
- `runs/profile_resources_u0.jsonl` — 1 Hz samples for step-timer run
- Earlier failed chrome attempt: do not rely on `profile_update1.json*`

## Decision log

- User correctly observed dual under-saturation during observation; telemetry is vs **allocated** CPU.
- Optimize phase already GPU-saturated; not the long pole.
- Next work: attack ranked list starting with **opp forwards** and **double-buffer overlap**, with **refresh** and **slab_copies** as host follow-ups.
- Re-profile after each meaningful patch with `PLO5BP_STEP_TIMERS=1` (2 updates, compare update 1 tables + resource means).

## Implementation log — attack #1 (2026-07-13)

**Status: coded + unit-tested locally; pod A/B not yet run.**

### What shipped (python/plo5bp/rollout.py)

1. **Opp D2H deferred:** all snapshot-group ct()s run first (step4a/opp_h2d_act), then **one** coalesced gate/chips .cpu() (step4b/opp_d2h). Pre-#1: H2D→act→sync per group.
2. **Independent opp H2D:** _opp_upload_act uses rom_numpy().to(device, non_blocking=True) + host keepalive list — does **not** reuse _PinnedStepH2D (avoids pin-buffer races when multi-group acts are in flight).
3. **Learner path unchanged:** still pinned H2D + immediate D2H; timers step2/3/5 learner-only (no nest under opp).
4. **Eager pool build:** or sd in range(len(pool.snapshots)): _get_snapshot_model(sd) before seat assign (still per-update cache only).
5. **Timer fix:** non-overlapping step4a / step4b so next profile % sums cleanly.

### Tests (local)

`
53 passed — test_rollout_parity, test_rollout_batched, test_batched_rollout,
            test_multiconfig_staging, test_multiconfig_rollout, test_env_batched
19 passed — test_rollout_v2, test_warmstart_pool
`

### Pod A/B still required

Same recipe, PLO5BP_STEP_TIMERS=1, 2 updates, compare update 1:
- baseline opp wall ~106s (step4/opp_forwards)
- expect lower step4a+step4b, higher rollout GPU util
- resume training from checkpoints/vSix4.pt u90 after measure if green

### Follow-up: double pin slots (same day)

_PinnedStepH2D now has **n_slots=2** (shape (2, cap, ...)):
- Opp groups alternate slots 0/1; wait_slot(s) only before reusing a slot (group *i* waits on *i-2* H2D, usually already done).
- Learner uses slot 0 + wait_slot(0) before upload (clears any leftover opp H2D on slot 0 from previous env-step).
- Pin memory ~2x step staging (still tiny vs rollout slabs).
- Tests: 62 passed after double-pin.

## Implementation log — attack #2 (2026-07-13)

**Status: coded + unit-tested locally (default OFF for Phase 2); pod A/B not yet run.**

### Phase 1 (always on, bit-exact)

- Learner ct stays on device (_learner_act_device)
- Opp acts still deferred D2H (#1)
- **One** coalesced step5/action_d2h for learner ActOut + critic V/Q + opp gates/chips
- Critic runs on device via critic.q_values / critic(...) with P9 o_t
- Host can prepare opp-hole rotation while learner kernels run

### Phase 2 (env-gated)

- Enable: PLO5BP_ROLLOUT_OVERLAP=1 (requires CUDA; default **OFF**)
- After apply/refresh, if some (not all) envs newly terminal and still collecting:
  1. step2x/act_wave_active: queue acts for ~newly_terminal (device, no D2H)
  2. Terminal host flush (payouts/GAE/slabs/reset/refresh_subset) as today
  3. step2x/act_wave_redealt: queue acts for re-dealt envs
  4. step2x/prefetch_d2h: merge + host arrays into _act_prefetch
  5. Next loop iter step2x/consume_prefetch: skip act, traj/apply as usual
- **Not** cross-update pipeline (still same update, same learner weights)

### Tests

- 62 passed default (Phase 1 only)
- 8 passed with PLO5BP_ROLLOUT_OVERLAP=1 on CPU (flag no-ops without CUDA)

### Still needed

- Pod A/B with PLO5BP_STEP_TIMERS=1 and PLO5BP_ROLLOUT_OVERLAP=1
- Compare update-1 rollout wall + GPU util vs baseline ~425s / ~11% GPU

## Implementation log — attack #3 (2026-07-13)

**Status: coded + unit-tested locally; pod A/B not yet run.**

### Change (python/plo5bp/env_batched.py)

**Rust _refresh partial-mask path (primary):**
- Before: observation_and_features_batch() (full pack) + observation_encoded_subset_batch(idx) (pack+encode again)
- After: **one** observation_encoded_batch() then in-place zero ~encode_mask
- Rationale: encoding terminal/ctor==-1 rows is cheap (zero early-out); double pack was the waste on the common post-apply path
- Full-mask path unchanged in spirit (still one FFI); now unified with partial

**Numpy path:** in-place ill(0) + scatter kept rows (less realloc)

**Diagnostics:** one-shot [obs-encoder] rust=… OBS_DIM=… log on first env construct

### Tests
- 97+ passed: refresh encode_mask, refresh subset, env_batched, encoding_batch (+ rollout_batched if re-run)

### Still needed
- Pod measure: step1a/refresh vs baseline ~104s with PLO5_RUST_ENCODER=1

## Deferred headroom (do not implement yet) — 2026-07-13

Ideas considered during #3 verification and explicitly **parked**. Revisit only
after a pod A/B of current #1+#2+#3 (especially `step1a/refresh` vs ~104s).

### A. Encode only "dirty" live rows each step

**Idea:** Skip re-encoding live envs whose packed state did not change this step.

**Why parked:**
- After `apply_hybrid_batch`, almost every non-terminal env advanced (new actor
  action) — dirty-row wins assume many *unchanged* live tables, which lockstep
  batched apply rarely provides.
- High risk of silent wrong obs if any field is missed in the dirty bit.
- Needs engine dirty flags + heavy parity tests.

**When to reopen:** Profiler shows many live encodes with identical pre/post packs.

### B. Pack-once / encode-live-only (partial pack)

**Idea:** Full pack for caches (commits/legal/actors for *all* envs, including
newly-terminal) but encode **only** live rows from that pack — not a second
subset pack+encode, and not encode-then-zero terminals.

**Why parked:**
- #3 already removed the **double pack** (the clear bug): one
  `observation_encoded_batch` + zero `~encode_mask`.
- Encoding `actor==-1` rows is already cheap (zero early-out); remaining win of
  "encode live only" may be small vs full Rayon encode.
- True pack-once/encode-subset needs a clean Rust API + bit-exact tests.

**Variants:**
| | |
|---|---|
| B1 | Pack all once, encode only `idx` from that pack (no second pack) |
| B2 | Two FFIs: light terminal pack for commits + pack+encode live (often worse) |

**When to reopen:** After #3, profile shows **encode** (not pack) still dominates
`step1a/refresh`.

### C. Thinner pack for cache-only needs

**Idea:** If payouts/aggression only need commits/actors, avoid packing full
observation feature blocks on all-terminal or cache-only paths.

**Why parked:** All-terminal path already uses `observation_and_features_batch`
without encode (#3 fix). Further thinning needs careful audit of every cache
field consumers read.

**When to reopen:** Pack time (not encode) is the measured long pole.

### D. #2 polish (not dirty encode)

Already noted under attack #2; still deferred:
- Prefetch when *all* tables terminal (currently skipped)
- Fuse Wave A/B D2H into one `.cpu()`
- Multi-stream / multi-process collectors (structural, not #3)

### E. Explicit non-goals (still)

- Cross-update rollout concurrent with PPO (stale self-play) — user rejected
- Cutting `TRAIN_OPP_OUTCOME_MC` without quality A/B
- Dropping OBS dims without design approval
- Approximate equity in obs for speed

### Decision rule

1. Ship/measure #1+#2+#3 first.
2. If `step1a/refresh` still large → open **B** (or **C** if pack-bound).
3. Open **A** only with evidence of redundant live encodes.

## Implementation log — attack #4 (2026-07-13)

**Status: coded + unit-tested locally; pod A/B not yet run.**

### Changes (python/plo5bp/rollout.py)

**S1 slab_copies:**
- Single (T,S,L) window into traj arrays; one sel/obs_idx plan
- Fewer repeated fancy-index+ravel chains
- Same sel order as pre-#4 (bit-exact row order)

**S2 opp-hole cache:**
- holes_rot_cache (n_envs, seats, 5, hole_w) filled on initial deal and after terminal re-deal
- Flush indexes cache instead of rebuilding rot_block every terminal wave

**pool_mix:**
- Fast path when pool empty / opp seats 0 / mix prob 0: vectorized mask clear
- Else sequential _assign_pool_mix (RNG order preserved)

### Deferred (per plan)
- S3 sort-by-obs_idx locality
- Compact payouts API
- Direct-write obs to slab redesign

### Tests
- 32 passed: rollout parity/batched/multiconfig/vrpo/v2

### Still needed
- Pod measure step9d/9f vs baseline ~45s / ~28s

## Deferred headroom after #4 (do not implement yet) — 2026-07-13

Parked after #4 verification. **Do not implement without a pod A/B** of
current #1–#4 (`PLO5BP_STEP_TIMERS=1`, trust update 1 vs baseline ~425s
rollout / 9d~45s / refresh~104s / opp~106s).

### Already shipped in #4 (not deferred)

- S1: single sel/obs_idx plan + fewer traj ravel chains (bit-exact order)
- S2: `holes_rot_cache` on deal/reset
- pool_mix fast path when pool inactive

### Still parked under #4 terminal/slab path

| ID | Idea | When to reopen |
|---|---|---|
| **S3** | Sort gather by `obs_idx` for `np.take` locality (row order free after GAE; PPO shuffles) | `step9d` still large after measure |
| **S4** | Write obs straight into output slab at learner-step time (drop `step_obs_pool` indirection) | Big redesign; only if gather layout is proven hopeless |
| **S5** | Rust/Cython fused gather of obs + meta given `obs_idx` | After S3 still insufficient |
| **P1** | Compact `payouts_*` only on `term_envs` (not full N) | `step9a` still hot |
| **P2** | Fuse payouts + `won = payouts + total_commit` on Rust for terminal mask | Same as P1 |
| **pool_mix on** | Vectorize RNG when pool active (preserve `term_envs` order) | Tiny; only if 9e shows up |
| **GAE over L** | Numba/Rust backward scan | ~3s baseline — low priority |

### Still parked under #3 (refresh) — same decision rule

| ID | Idea | When to reopen |
|---|---|---|
| **#3-B** | Pack-once / encode-live-only from one pack | **encode** (not pack) dominates `step1a/refresh` |
| **#3-C** | Thinner cache-only pack | **pack** dominates refresh |
| **#3-A** | Dirty live-row encode | Evidence of redundant live encodes |

### Still parked under #2

| ID | Idea | When to reopen |
|---|---|---|
| All-terminal prefetch | Currently skipped when `newly_terminal.all()` | Overlap on + measure shows gap |
| Fuse Wave A/B D2H | One `.cpu()` instead of two | Micro after overlap measured |
| Multi-stream / multi-process collectors | Structural | Separate project |

### Explicit non-goals (unchanged)

- Cross-update rollout concurrent with PPO (stale self-play) — user rejected
- Cutting `TRAIN_OPP_OUTCOME_MC` / `ev_runout_samples` without quality A/B
- Dropping OBS dims without design approval
- Approximate equity in obs for speed

### Decision rule (post #1–#4)

1. Deploy/measure #1–#4 on pod with step timers.
2. If `step9d` still large → **S3** then **S5** (not S4 first).
3. If `step1a/refresh` still large → **#3-B** or **#3-C** by pack vs encode split.
4. If GPU util still soft and opp/overlap timers hot → #2 flag on + polish.
5. If nothing dominates → stop optimizing; resume training.

