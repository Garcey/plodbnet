# Handoff — "double-refresh" rollout optimization (per-PPO-update wall-clock)

**Date:** 2026-05-28/29
**Goal:** reduce wall-clock per PPO update on the RunPod training run, with
TWO hard constraints from the user:
1. **No increase in GPU VRAM** (card is maxed: RTX PRO 6000 Blackwell, 96 GB,
   ~92/96 GiB used during the run).
2. **No reduction in training accuracy** — any change must be **bit-exact**
   (produces numerically identical training data / gradients) or have an
   airtight numerically-negligible argument.

The user has chosen to **implement the fix manually themselves** from this
point. This document is the complete spec + context so they (or a future
session) can finish it without re-deriving anything.

---

## TL;DR — what to build

In `collect_rollout_batched` (`python/plo5bp/rollout.py`) there are **two**
full `env._refresh()` calls per loop iteration. The **second** one (after
`reset_terminal_batch`) re-packs and re-encodes **all ~49,134 envs**, but
`reset_terminal_batch` only changed the small subset of envs that just ended
a hand. Recomputing the unchanged ~90%+ is pure waste.

**Fix:** make the second refresh update **only the reset envs** (the
`reset_mask` / `newly_terminal` rows), reusing the first refresh's cached
state for everything else.

**Expected win:** **~18–22% faster per update** (≈16 min → ≈13 min),
**bit-exact, zero extra VRAM** — IF done in full (requires a Rust subset-pack
method). A Python-only partial version is ~4–5% (skips only the numpy
re-encode, not the dominant Rust pack).

**This is safe because** re-dealing finished hands provably leaves every
other env's engine state byte-for-byte untouched, and nothing between the two
refreshes mutates non-terminal env state. So the unchanged envs already hold
correct data from refresh #1 — reusing it is numerically identical.

---

## Why we're confident (evidence)

### The profile (authoritative — ran on the real pod code, exact live config)
Collected via `scripts/train.py --profile-one-update` on the pod
(49,134 envs, rollout 6,266,880, 48 minibatches, 4 epochs, 2048×4, cuda,
**pool=0 self-play**). Trace at pod `runs/profile_update0.json` (~98 MB).
Summarized with `scripts/summarize_profile.py`:

| Span | calls | CPU ms | CUDA ms | Notes |
|---|---|---|---|---|
| `step1a_bundle/obs_features_batch` | 263 | 326,339 | 0 | Rust FFI `observation_and_features_batch`, **nested inside both refreshes** |
| `step9f/reset_terminal` | 129 | 213,569 | 0 | `reset_terminal_batch` **+ refresh #2** |
| `step1a/refresh` | 133 | 203,846 | 0 | refresh #1 |
| `step1/encoder` | 263 | 89,367 | 0 | numpy `encode_observation_batch`, nested in both refreshes |
| `step12/inner_loop` | 1 | 28,108 | 25,885 | **THE ENTIRE PPO UPDATE — only ~3%** |
| `step5/action_d2h` | 133 | 12,992 | 11,295 | per-step D2H syncs (~1.4%) |
| `step9d/slab_copies` | 129 | 13,041 | 0 | terminal flush gather |
| (others: rust_apply, gae_scan, h2d, payouts, bonus…) | | ~15,000 | | small, additive |
| **TOTAL annotated** | | **910,776** | 42,715 | inflated by nesting double-count |

**Key conclusions:**
- **Rollout ≈ 97% of wall-clock; the PPO update ≈ 3%.** This INVERTS the
  original static audit's guess. **Do NOT optimize the update side**
  (torch.compile is already active per the dynamo spans in the trace; fused
  AdamW already on). Update-side wins target only 3% — not worth it.
- The two refresh spans (#1 ~204s, reset+#2 ~214s) are the largest captured
  costs. Each refresh ≈ Rust pack (~163s) + numpy encoder (~44s) + unpack.
- **Per-call:** `obs_features_batch` ≈ 1240 ms, `encoder` ≈ 340 ms.
- **263 refreshes = 133 (refresh#1) + 130 (refresh#2)** — the second refresh
  fires on nearly every step.
- **NESTING CAVEAT:** these `record_function` spans are nested — the
  `obs_features_batch` and `encoder` totals are counted INSIDE both
  `step1a/refresh` and `step9f`. Do **not** sum them. Real wall-clock ~960s.

### Savings math
If refresh #2 processes only the reset fraction `f` of envs:
- Full (Rust subset pack + subset encode): saves ≈ 130 × (1240+340)×(1−f) ms.
  At small `f`, ≈ 130 × ~1480 ms ≈ **~192 s of ~960 s ≈ ~20%**.
- Python-only (subset encode, full Rust pack still runs): saves
  ≈ 130 × 340 ms ≈ 44 s ≈ **~4–5%**.

### The one unverified assumption — MEASURE THIS FIRST
The ~18–22% assumes the **per-step terminal fraction is small** (~2–10% of
envs reset per step). This is very likely (a bomb-pot hand spans multiple
action steps: flop/turn/river; rollout is ~127.5 transitions/env total) but
was **reasoned, not measured**. **Before writing the Rust change, confirm it:**
add one line to log `reset_mask.sum() / n_envs` averaged over one update.
If `f` is well under ~10%, the full fix is worth it. (If `f` were large, the
win shrinks proportionally.)

### Bit-exactness — verified by adversarial multi-agent review
- `reset_terminal_batch` mutates **only masked envs**; non-masked envs'
  engine state is untouched. (Confirm on the Rust side:
  `reset_terminal_batch` in `bindings.rs` — it iterates the mask.)
- Between refresh #1 and refresh #2, the loop only mutates **Python-side
  bookkeeping** (trajectory/cost arrays), **not engine state** of
  non-terminal envs. So non-terminal envs' packed obs after refresh #2 are
  byte-identical to refresh #1.
- `encode_observation_batch` is **purely per-row** (each env encoded
  independently from its own bundle data), so encoding a subset of rows gives
  results identical to encoding the full batch and slicing. Bit-exact.
- After `reset_terminal_batch(mask)`, ALL envs are non-terminal (reset deals
  fresh hands; non-masked were already non-terminal). So `_dones` for reset
  rows flips True→False; non-reset rows stay False from refresh #1. Correct.
- `reset_mask == newly_terminal` in the code (verified).

### Rejected alternatives (do NOT do these)
- **Cache `opp_outcome_fractions` by board:** NOT bit-exact — it depends on
  the **current actor's hole cards**, which rotate within a street, not just
  the board. Silent correctness bug. Rejected.
- **Cut Monte-Carlo samples 1024→512:** ~8% faster but changes training
  numbers. Violates no-accuracy-loss. Rejected.
- **Any update-side / torch.compile / GPU tuning:** update is only 3%.
  Rejected.

---

## Implementation plan (the FULL ~20% version)

### Files involved
- `python/plo5bp/rollout.py` — `collect_rollout_batched`. The two refresh
  sites: **refresh #1** ~line 914–915 (right after `apply_hybrid_batch`,
  span `step1a/refresh`); **refresh #2** ~line 1081–1086 (terminal block,
  inside span `step9f/reset_terminal`, right after `reset_terminal_batch`).
- `python/plo5bp/env_batched.py` — `BatchedBombPotEnv._refresh()` (~line
  251–278). Currently replaces ALL cached arrays wholesale: `_obs`, `_legal`,
  `_gate_mask`, `_min_raise`, `_max_raise`, `_actors`, `_dones`,
  `_total_commit`, `_bet_to_call`, `_street_commit`, `_street`.
- `rust_engine/src/bindings.rs` — `observation_and_features_batch` (~line
  1287–1385) and `pack_observation` (~line 1387–1625). The expensive
  per-env work (`opp_outcome_fractions`, k=2/3 exhaustive + k=4 MC=1024 hand
  evals, ~line 1420 `into_par_iter`) and the parallel memcpy pack (~line 1505
  `into_par_iter`) run for ALL envs every refresh.

### Step 1 — Measure `f` (do this first; cheap, risk-free)
Add a temporary counter in `collect_rollout_batched` summing
`reset_mask.sum()` and step count; print mean fraction at end. Run one update
on the pod (or laptop at smaller scale). Confirm `f` ≪ 10%. (Can reuse the
`--num-updates 1` profiling harness pattern from this session.)

### Step 2 — Rust: add a subset pack method
Add to `BatchedEngine` in `bindings.rs` a method, e.g.
`observation_and_features_subset_batch(indices: PyReadonlyArray1<i64>)` (or
take a bool mask), that:
- Packs ONLY the given env indices into **compact** arrays of size `k =
  len(indices)` (same per-env logic as `pack_observation`, just iterating the
  index list instead of `0..n`).
- Returns the same dict keys as `observation_and_features_batch`
  (`hero_hole`, `board_a/b`, …, `legal_mask`, `hero_cat_a/b`,
  `total_commit`, `bet_to_call`, `street_commit`, `street`, `actor`,
  `min_raise`, `max_raise`, etc.) but with `k` rows.
- Keep `py.allow_threads` + `into_par_iter` parallelism.
- Refactor `pack_observation` to share code with an indexed variant (e.g.
  `pack_observation_indexed(&self, idx: &[usize], s)`), so the per-env logic
  stays in one place (avoid divergence bugs).

### Step 3 — Python: add `BatchedBombPotEnv._refresh_subset(mask)`
In `env_batched.py`:
- `idx = np.nonzero(mask)[0]`; if `idx.size == 0`, return.
- `bundle = self._be.observation_and_features_subset_batch(idx)` (compact, k rows).
- Encode compact: `obs_sub = encode_observation_batch(bundle, cat_a_sub,
  cat_b_sub, self.config)` → (k, OBS_DIM).
- **Scatter into existing cached arrays** at `idx` (do NOT replace the whole
  array): `self._obs[idx] = obs_sub`, `self._legal[idx] = ...`,
  `self._min_raise[idx] = ...`, `self._max_raise[idx] = ...`,
  recompute `self._gate_mask[idx]` via `gate_mask_from_bounds`, set
  `self._actors[idx]`, `self._dones[idx]`, `self._total_commit[idx]`,
  `self._bet_to_call[idx]`, `self._street_commit[idx]`, `self._street[idx]`.
- Apply the `gate_mask[dones]=False` rule for the subset rows (reset envs
  won't be terminal, but keep the invariant).
- **Watch dtypes** — match `_refresh` exactly (`int8` actors, `uint64`
  min/max_raise, `int64` total_commit, etc.).

### Step 4 — Wire into rollout.py
Replace refresh #2 (the `env._refresh()` at ~line 1086, inside
`step9f/reset_terminal`) with `env._refresh_subset(reset_mask)`. Leave
refresh #1 (~915) untouched. Keep the `record_function` span wrappers.

### Step 5 — Parity test (the safety gate — REQUIRED)
Add a test (e.g. `tests/python/test_refresh_subset_parity.py`) asserting the
subset path is byte-identical to the full path:
1. Build two identical `BatchedBombPotEnv`s (same seeds/buttons), step both
   through several identical actions until some envs go terminal.
2. On env A: `reset_terminal_batch(mask)` then full `_refresh()`.
   On env B: `reset_terminal_batch(mask)` then `_refresh_subset(mask)`.
3. Assert ALL cached arrays equal: `_obs`, `_legal`, `_gate_mask`,
   `_min_raise`, `_max_raise`, `_actors`, `_dones`, `_total_commit`,
   `_bet_to_call`, `_street_commit`, `_street` (use `np.array_equal` /
   exact equality, not `allclose` — this must be bit-exact).
Also good: an end-to-end check that `collect_rollout_batched` with the
optimization produces an identical `Batch` to a reference run at the same
seed (the existing batched-env parity tests in `tests/python/` are the
template).

### Step 6 — Build & run tests (on the LAPTOP)
```bash
# from repo ROOT (not rust_engine/) — see CLAUDE.md
.venv/Scripts/maturin develop --release
.venv/Scripts/python -m pytest tests/python/ -q   # full suite must stay green
.venv/Scripts/python -m pytest tests/python/test_refresh_subset_parity.py -q
```
Per HANDOFF.md the laptop can build the Rust ext and the suite was last green
(383 pass, 2 pre-existing card-template failures unrelated to this).

### Step 7 (optional safe down-payment) — Python-only first
If you want a low-risk partial win before the Rust work: in `_refresh`, keep
the full Rust `observation_and_features_batch` but skip the numpy
`encode_observation_batch` for non-reset rows (it already has a `live_mask`
and is easy to subset; scatter only reset rows). ~4–5%, no Rust rebuild,
trivially bit-exact. The full Rust subset pack (Steps 2–4) is where the
remaining ~15% lives; the two are NOT additive (the Rust version subsumes the
Python one).

---

## Deployment notes (important — read before pushing to pod)
- **Code provenance (user asked):** analysis was done against the **laptop**
  repo, verified structurally to equal the pod (pod sits at bare
  `initial commit 5dc9f2b` with all training changes as **uncommitted
  working-tree edits**; the laptop has the **same changes committed**:
  commits `bb0a2d8`, `b0c9bdf`, `8b560b7`). ppo.py + train.py were
  spot-checked line-for-line and matched. The **profile ran on the pod's
  actual code**, so measurements are ground-truth.
- A **definitive diff** of the three files-to-edit (`rollout.py`,
  `env_batched.py`, `bindings.rs`) between laptop and pod was **in progress
  when this handoff was requested** — recommend completing it
  (`scp` pod files to `/tmp/pod_check/` and `diff --strip-trailing-cr`)
  before assuming laptop==pod for these specific files.
- **Deploying to the pod is delicate:** the pod's matching changes are
  UNCOMMITTED. Pushing a new laptop commit means reconciling the pod's
  working tree (it can't just `git pull` over uncommitted edits). Plan the
  pod sync explicitly; don't blow away the pod's working tree without
  confirming it equals the laptop first. **The pod is NOT currently training
  (we stopped it this session), so there's no live run to disturb.**

## Current pod/run state (as of this session)
- **Training STOPPED** cleanly at **update 296** (was on the phase-3 run,
  PID 21358). Final checkpoint saved: `checkpoints/optimized3.pt`. Last
  mid-checkpoint: `checkpoints/optimized3_290.pt`. (User had wanted to stop
  ~u300 to lower entropy for phase 4 — see main `HANDOFF.md` for the phase-4
  entropy plan: `--block-rotation clubgg:0.06,clubgg_deep:0.075,deep:0.10`.)
- **GPU free** (0 MiB). Throwaway profiling checkpoints cleaned up.
- **py-spy** was installed on the pod (v0.4.2) but proved unusable here:
  it falls behind badly walking the engine's ~190-thread rayon pool, even at
  rate 25 without `--native`. **Use `--profile-one-update` (torch.profiler),
  not py-spy**, for this codebase — it hooks ops, not threads. The pod's
  profiling log file was accidentally `rm`'d while open but the JSON
  (`runs/profile_update0.json`) is intact.
- **VRAM side-note (free win, separate from this):** peak CUDA allocated was
  only **~49.7 GB** though the dashboard shows ~92 GB — the gap is allocator
  reserved-but-unused slack. Setting env var
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (no code change, zero
  accuracy risk) may reclaim real headroom — useful later for raising
  envs/rollout (the lever the user is actually maxed on). Verify via
  `torch.cuda.memory_reserved()` vs `memory_allocated()`.

## Helper artifacts created this session
- `scripts/analyze_pyspy_folded.py` — parses py-spy `-f raw` collapsed
  stacks into rollout-vs-update buckets + hotspots. (Unused in the end since
  py-spy was abandoned for torch.profiler, but kept for reference.)
- `scripts/summarize_profile.py` — already existed; groups the
  torch.profiler trace by `record_function` span (this is what produced the
  table above). Run: `python scripts/summarize_profile.py runs/profile_update0.json`.

## Quick correctness checklist before declaring done
- [ ] `f` (reset fraction/step) measured and confirmed small.
- [ ] Rust subset pack shares per-env logic with `pack_observation` (no dup).
- [ ] `_refresh_subset` scatters into ALL cached arrays with matching dtypes.
- [ ] refresh #1 left untouched; only refresh #2 swapped.
- [ ] Parity test asserts byte-identical cached arrays (exact, not allclose).
- [ ] `maturin develop --release` from repo ROOT; full pytest green.
- [ ] Pod sync plan decided before deploy; pod working tree reconciled safely.
