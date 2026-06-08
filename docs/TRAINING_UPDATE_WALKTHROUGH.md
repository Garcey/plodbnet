# Anatomy of one training update (this project, step by step)

Plain-language, chronological walkthrough of **exactly what happens during a
single PPO update** in this codebase — what runs, in what order, what gets
stored in CPU RAM vs GPU VRAM, and when each is allocated and freed.

Written for someone who understands PPO at a big-picture level but wants the
project-specific mechanics. It follows the actual code:
`scripts/train.py` (the loop) → `python/plo5bp/rollout.py`
(`collect_rollout_batched`) → `python/plo5bp/ppo.py` (`PPOTrainer.update`),
plus `env_batched.py`, `selfplay.py`, and the Rust engine.

> **Two notes on precision.** (1) Memory *sizes* below are computed from the
> live config (49,134 envs, rollout 6,266,880, 48 minibatches, 2048×4 net) —
> they're accurate to the formula but the exact resident bytes depend on
> lazy paging and PyTorch's caching, which I flag where it matters. (2) The
> code-flow ordering is exact; the memory-behavior explanations are the
> mechanism, some of which is empirical (PyTorch allocator behavior), not
> literally written in this repo.

---

## 0. The shape of one update (the 30-second version)

One update = **two phases that run back-to-back, never overlapping:**

1. **Rollout (the "CPU phase").** Play ~6.27 million hands-worth of decisions
   by simulating ~49k poker tables in parallel, recording every learner
   decision. GPU is mostly idle here; CPU (the Rust engine + numpy) is busy.
   This is ~95%+ of the wall-clock.
2. **Optimize (the "GPU phase").** Take all those recorded decisions and do
   the PPO gradient updates on the neural network. CPU mostly idle; GPU at
   100%. This is ~3-5% of the wall-clock.

This is why your dashboard shows CPU and GPU taking turns, and why RAM rises
during phase 1 and falls during phase 2.

**What persists across all updates** (allocated once, at program start, lives
on the GPU the whole run):
- The **learner network** weights (~58 MB — that's why checkpoints are 58 MB).
- The **AdamW optimizer state** (two extra copies of every weight + the
  gradients) — roughly another ~170 MB.
- These never get freed between updates.

---

## Phase A — Setup at the top of the update
*(`scripts/train.py` main `while` loop)*

**A1. Check stop / budget.** Looks at the Ctrl-C flag and the update/time
budget. (On GPU runs it skips the `runs/threads.txt` live-thread-count check —
that's a CPU-only feature.) Cheap; no memory.

**A2. Pick the entropy tier for this update (block rotation).**
`block_idx = (update // block_size) % len(blocks)`. With your phase-4 setting
`clubgg:0.06, clubgg_deep:0.075, deep:0.10` and `block_size=50`, updates
0–49 use the `clubgg` tier at entropy 0.06, updates 50–99 use `clubgg_deep`
at 0.075, updates 100–149 use `deep` at 0.10, then it cycles. This picks
**both** the stack-distribution tier *and* the entropy coefficient for the
whole update.

**A3. Randomly sample the table configuration for this rollout.**
*(`_sample_game_config` in `train.py`)* — **this is the "randomly assigned
seats and stacks" you asked about.** Once per update:
- **Number of seats** is drawn (2–6) from the seat-count distribution.
- **Each seat's starting stack** is drawn from the active tier's stack bands
  (e.g. for `clubgg`: 50% chance 20–40 bb, etc.), converted to chips.

Important detail: **the config is sampled once and shared by all ~49k tables
in this update.** Seats/stacks vary *across* updates, not across tables within
one update — that keeps every table's data the same shape so it can be batched.
So "this update is a 5-handed game with stacks [209, 181, 240, 222, 100] bb"
is decided here, once. (Tiny memory: just a config object.)

---

## Phase B — Rollout (`collect_rollout_batched` in `rollout.py`)

This is where almost all time and almost all the RAM movement happens.

**B1. Build the batched environment.** `BatchedBombPotEnv(n_envs, config)`
constructs the Rust `BatchedEngine`, which allocates ~49k independent poker
game-states **in CPU RAM on the Rust side**, plus the Python-side cached numpy
arrays (observations, masks, actors, etc.). Modest RAM.

**B2. Assign opponents (pool mix).** For each table, decide which seats are
played by the **current learner** vs by a **frozen past snapshot** of itself
(self-play pool). At your settings, ~50% of tables get 2 seats handed to a
random pool snapshot. (This is `_assign_pool_mix`. The frozen opponent models
are fetched from a **cache on the pool** — built once and reused across
updates, then parked in CPU RAM — so they cost ~0 GPU memory during the
optimize phase. *This is one of this session's optimizations.*)

**B3. Seed + deal all tables.** Random per-table seeds and dealer buttons are
drawn, then `env.reset_batch(...)` has the Rust engine **deal every table's
hole cards and flop** at once. Bomb-pots start on the flop (everyone's already
posted the ante), so there's no preflop betting.

**B4. Encode the first observations.** `_refresh()` turns every table's raw
game-state into the **959-number observation vector** the network reads (card
one-hots, stacks, pot, betting history, hand-strength features, etc.). The
heavy hand-strength math runs in parallel Rust; the assembly runs in numpy.
The result is a `(49134, 959)` float32 array in **CPU RAM**.

**B5. Allocate the recording buffers (the big RAM commit).** Before the loop,
the rollout pre-allocates the arrays that will hold every recorded decision:
- **Per-table/per-seat trajectory arrays** (`traj_*`, `costs_arr`, etc.) —
  shape `(n_envs, n_seats, 32)`. A few hundred MB total. (The `32` is
  `MAX_STEPS_PER_SEAT` — the most actions one seat can take in a hand.)
- **A flat observation pool** (`step_obs_pool`): `(~7.8M, 959)` float32 ≈
  **~30 GB of address space**. It's plain numpy, so RAM is only physically
  committed as rows actually get written during the rollout (lazy paging).
- **The output "slabs"** (`all_obs_arr`, `all_ret_arr`, …): same `~7.8M`-row
  capacity. On a GPU run these are **pinned (page-locked) CPU memory** so the
  later copy to the GPU is fast. The obs slab alone is `~7.8M × 959 × 4 ≈
  30 GB`. PyTorch caches pinned memory, so this block is typically allocated
  on the **first** update and **reused** on later updates rather than
  re-allocated.

> **This is the main source of the RAM you see climb.** The obs-width buffers
> (the step pool + slabs) are tens of GB. The pinned slab tends to stay
> resident (reused each update); the plain-numpy step pool fills up during the
> rollout and is released when the rollout function returns — that
> fill-then-release is the per-update RAM oscillation you observe (the ~35 ↔
> ~50 GB swing). See the Memory Timeline at the bottom.

**B6. The per-step loop** *(repeats until ~6.27M learner decisions are
recorded — this is the bulk of the time)*. Each iteration advances every table
by one decision:

  - **B6a. Read current state** of all tables (observations, legal-action
    masks, raise bounds, whose turn it is, which tables are done).
  - **B6b. Snapshot pre-step numbers** (pot, amounts committed) for the reward
    bookkeeping. (numpy copies, RAM.)
  - **B6c. Split tables** into "learner is acting" vs "a frozen opponent is
    acting."
  - **B6d. Learner forward pass** (`_forward`): stack the acting tables'
    observations, **copy them to the GPU (H2D)**, run the network to get each
    table's action (fold/call/raise + raise size) and the critic's value, then
    **copy the results back to CPU (D2H)**. These transfers are small and
    transient — VRAM here is just the small batch of observations + activations
    for this one step, freed immediately after.
  - **B6e. Opponent forward passes.** Same thing, once per distinct pool
    snapshot in play. (Frozen models moved to GPU on use, parked back to CPU
    at the end of the rollout.)
  - **B6f. Record learner decisions.** Write each learner action's
    observation, chosen action, raise size, log-probability, and critic value
    into the trajectory arrays + the flat obs pool. (numpy writes, RAM.)
  - **B6g. Apply all actions** via the Rust engine (`apply_hybrid_batch`) —
    mutates every table's game-state.
  - **B6h. Re-encode observations** (`_refresh`) for the next step.
  - **B6i. Reward bookkeeping** (per-step chip cost; the aggression-bonus
    accounting, currently zero-coefficient).
  - **B6j. Handle finished hands.** For tables whose hand just ended:
    compute **payouts** (Rust), then for each learner seat run the
    **GAE-λ backward scan** to turn its sequence of rewards into "advantages"
    and "returns" (the learning targets), and **copy that finished
    trajectory** out of the work buffers into the output slabs. Then re-deal
    those tables (`reset_terminal_batch`) and re-encode **only the re-dealt
    tables** (`_refresh_subset`). *Re-encoding only the changed tables instead
    of all 49k is this session's biggest optimization, ~20% faster per
    update.*

**B7. Park the opponent models** back to CPU (frees their GPU memory before
the optimize phase).

**B8. Finalize — the big copy to the GPU (`_finalize_batch_arr`).** Slice the
output slabs down to exactly the number of recorded transitions (~6.27M),
then **copy the whole batch from CPU to GPU VRAM**: observations
(`~6.27M × 959 × 4 ≈ 24 GB`), plus actions, log-probs, values, returns,
advantages. Advantages are normalized (mean 0, std 1) on the GPU. The
function returns this `Batch` — **now living in VRAM** — and all the CPU-side
work buffers (step pool, slabs, traj arrays) go out of scope and are freed.

> **This 24 GB obs batch on the GPU is the dominant VRAM consumer of the
> whole update.** It's allocated at finalize and lives through the entire
> optimize phase.

---

## Phase C — Optimize (`PPOTrainer.update` in `ppo.py`)

Now the GPU phase. Everything operates on the `Batch` already in VRAM.

**C1. Loop over epochs × minibatches.** `ppo_epochs=4`, `num_minibatches=48`
→ the ~6.27M transitions are shuffled and sliced into 48 chunks of ~130,560
each, and the whole set is passed over 4 times = **192 gradient steps**. The
shuffle index is built on CPU and the per-minibatch slices are gathered on the
GPU.

For each of the 192 minibatches:
  - **C2. Forward pass under bf16 autocast** (`model.evaluate`, torch.compiled
    on CUDA): re-run the network on the minibatch's observations to get the
    *current* policy's log-probabilities, entropy, and value estimates.
    Allocates **activations in VRAM** for this minibatch (transient).
  - **C3. Compute the PPO loss:** the clipped policy-gradient term (using the
    ratio of new vs old action probabilities × advantages), the clipped value
    loss, and the entropy bonus (weighted by this update's entropy coef from
    A2). One number.
  - **C4. Backward pass:** `loss.backward()` computes gradients — allocates a
    **gradient buffer in VRAM** (~same size as the weights).
  - **C5. Optimizer step:** clip the gradient norm to 0.5, then `AdamW.step()`
    nudges the weights. Updates the persistent optimizer state in VRAM.
  - **C6. Stats:** accumulate policy/value/entropy/KL as GPU tensors (summed
    on-device, pulled to CPU once at the very end to avoid stalls).

**C7. Return stats** (`pi`, `v`, `H`, `kl` — the numbers in your log line).
When `update()` returns, the `Batch` is no longer referenced and its ~24 GB of
VRAM becomes free (but stays *reserved* by PyTorch — see VRAM note).

---

## Phase D — Wrap-up (`scripts/train.py`, after the update)

**D1. NaN check** on the losses (asserts training didn't diverge).
**D2. Pool snapshot.** Every `snapshot_every` (50) updates, copy the current
weights to CPU and add them to the opponent pool (capped at 16; oldest
dropped). This is why early updates are pool-light and the pool fills slowly.
**D3. Checkpoint save.** Every 5 updates, save weights to
`checkpoints/optimized4_<update>.pt` (and the final `optimized4.pt`).
**D4. Log line** (`pi=… v=… H=… kl=… seats=… stacks_bb=… ent=…`).
**D5. `update += 1`**, loop back to A1.

---

## Memory timeline — answering your RAM and VRAM questions directly

### CPU RAM (the ~35 ↔ ~50 GB oscillation you see)
RAM rises during the **rollout** and falls during the **optimize** phase, once
per update:

- **Rises at B5–B6:** the rollout allocates and fills the big **959-wide host
  buffers** — the flat observation pool (`step_obs_pool`, plain numpy, fills
  lazily to ~24 GB as decisions are recorded) and the pinned output slabs
  (~30 GB capacity for obs alone). The trajectory arrays add a few hundred MB.
- **Falls at B8 → C:** when `collect_rollout_batched` returns, the plain-numpy
  pool and the trajectory arrays are freed (Python reclaims them), so RAM
  drops back down going into the GPU optimize phase. The **pinned** slab is
  usually *kept cached by PyTorch and reused* next update, so it stays
  resident rather than bouncing — which is why your swing (~15 GB) is smaller
  than the raw buffer sizes (~30 GB+).
- **Net:** the oscillation is the rollout's host obs-buffers being
  filled-then-released each update. The steady ~35 GB floor includes the
  cached pinned slab + the engine + Python/torch baseline.

### GPU VRAM (why it climbs to ~92 GiB and stays) — and why it's NOT idle fluff
**The ~92 GiB is mostly REAL usage, not reserved slack.** (An earlier note in
this repo wrongly implied most of it was idle reserve; the OOM history below
disproves that — if 60+ GiB were truly free, a 1.5× rollout would have fit.)

- **Persistent (from program start):** network weights (~58 MB) + AdamW state
  + gradients (~170 MB). Small.
- **The dominant consumer — the resident rollout batch (allocated per update
  at B8, freed at C7):** the observation `Batch` = `rollout_length × 959 × 4`
  bytes ≈ **22 GiB** at the current 6.27M rollout. **This whole block stays
  resident in VRAM for the entire optimize phase** — all 4 epochs iterate over
  it — plus a few GB of returns/advantages/values.
- **Plus optimize-phase transients:** per-minibatch gathered slices +
  forward/backward activations + torch.compile/inductor workspaces. Several GB
  at the momentary peak.
- **Why it *climbs over ~dozens of updates* then plateaus:** PyTorch never
  returns freed GPU memory to the card — it keeps a **reserved pool** and
  grows it to the high-water mark it has ever needed. Fragmentation (and the
  dynamic last-minibatch, which is a different size each time) makes that
  high-water mark creep upward for the first updates until it covers the true
  peak, then it plateaus at your consistent ~92/96 GiB. So the climb is real
  peak + fragmentation accumulating, NOT idle padding. At 1.0× rollout the
  plateau sits ~4 GiB under the 96 GiB ceiling.

### ⚠️ Why bigger rollout OOMs — and why MORE MINIBATCHES DOESN'T FIX IT
A prior attempt at **1.5× rollout (9.4M) with more minibatches OOM'd around
update 18.** The reason is the single most important VRAM fact here:

- **The resident obs batch scales with `rollout_length` and is fully resident
  regardless of minibatch count.** 1.5× rollout → obs batch 22 GiB → **34 GiB
  (+11 GiB)**. Unavoidable.
- **`num_minibatches` only slices that already-resident block; it does NOT
  shrink it.** More minibatches = smaller *transient* activation slices (a few
  GB), but the 34 GiB resident batch is untouched. So increasing minibatches
  counteracts only the small activation part, never the large rollout-scaled
  part.
- At 1.0× the pool plateaus ~4 GiB under the ceiling. +11 GiB of resident
  batch, plus the reserved pool's update-by-update fragmentation climb, crosses
  96 GiB around update ~18 → OOM. (It survives to ~18 rather than dying at
  update 1 *because* the reserved high-water mark climbs gradually.)

**Practical ceiling (from the repo's VRAM note):** keep the obs batch ≲ ~28-30
GiB ⇒ **rollout ≲ ~7.5-8M** to fit alongside everything else on the 96 GiB
card. To go meaningfully bigger you'd need to *shrink the obs batch itself*
(fewer envs won't help — it's `rollout_length × 959 × 4`), e.g. lower
`rollout_length`, reduce `OBS_DIM`, or store the GPU obs batch in bf16 (~halves
it to ~11 GiB — but that changes the stored observations' precision vs what the
rollout used, a potential accuracy concern that would need its own bit-exact
check). None of these are minibatch-count changes.

---

## Where this session's optimizations sit in the flow
- **Double-refresh elimination (~20%, on):** B6j — re-encode only re-dealt
  tables, not all 49k.
- **Opponent-model cache (on):** B2/B6e — build frozen opponents once, reuse,
  park on CPU.
- **D2H coalescing, GAE-scan bound, inference_mode (on):** B6d / B6j — minor.
- **Rust encoder (built, default OFF):** would move B4/B6h/B8's encode work
  into the engine; measured ~neutral, so left off. numpy encode is the active
  path.

---

## One-line mental model
**Rollout = fill ~24 GB of decisions in CPU RAM by simulating 49k poker
tables; ship it to the GPU; Optimize = 192 gradient steps over it on the GPU;
free the CPU buffers; repeat.** RAM breathes with the rollout buffers each
update; VRAM climbs once to the reserved-pool ceiling and stays.
