# Attack #3 follow-through: residual `step1a/refresh`

## Current state (evidence)

| Item | Status | Evidence |
|---|---|---|
| **Double-pack fix** | **LANDED** | `env_batched._refresh` rust path: one `observation_encoded_batch()` + in-place zero `~em`; all-skip uses pack-only (`observation_and_features_batch`). Comment marks “Attack #3 (2026-07-13)”. |
| **Rust encoder live in prod** | **YES** | `OBS_DIM == 1171 == _RUST_ENCODER_OBS_DIM`; guardian `PLO5_RUST_ENCODER=1`; `[obs-encoder]` one-shot log; `test_encoding_batch` width gate. |
| **Terminal encode skip + subset post-reset** | **DONE** | Rollout `encode_mask=~newly_terminal`; `_refresh_subset`; `all_hole_cards_subset_batch` + `holes_rot_cache` (#4 S2). |
| **Multiconfig env recreate tax** | **ALREADY FIXED** | `env_cache` + `reconfigure` (P3) in `collect_rollout_multiconfig` / `collect_rollout_batched`. |
| **Numpy encode in PLO train** | **Should not run** | Only if flag off / NLH / width mismatch. Production path is Rust Rayon. |
| **#3 wall win on pod** | **INCONCLUSIVE / ~flat** | A/B with `OVERLAP=1`: refresh **104→108s**. #2 regressed rollout **+41%** — pollutes isolation. #4 slab **45→26s** looked real. |
| **Live train** | refresh still **~17–23%** of timers; rollout **~610–680s** | Matches residual host encode pole after hygiene. |

**Verdict:** #3 hygiene is **partially done**. The clear bug (double pack on partial mask) is fixed; **residual ~100s is mostly legitimate full-batch pack+encode**, not leftover double-pack. Next work is **measure-driven residual reduction**, not re-doing R2.

### What the hot path does today (prod, post-apply)

```text
apply_hybrid → _refresh(encode_mask=~newly_terminal)
  rust: observation_encoded_batch()   # pack ALL + Rayon encode ALL
        np.asarray(obs) + copyto(self._obs) + zero ~mask
        _unpack_post (legal/commits/actors/…)
```

`encode_obs_row` already early-outs `actor < 0` (cheap). Encoding terminals is not the waste; **packing + allocating/copying full (N, 1171) + full aux every call** is.

Multiconfig: `num_envs=49134`, 30 configs → **~1637 envs/sub**, ~7.4k refreshes/update ≈ **~14 ms/call × 7.4k ≈ 100s**.

---

## Ranked worklist

### P0 — Isolate + split cost (do first; no algo risk)

| Work | Why | Est. wall impact | Confidence |
|---|---|---|---|
| **P0a** Re-profile with **`PLO5BP_ROLLOUT_OVERLAP=0`** (unset), `STEP_TIMERS=1`, `RUST_ENCODER=1`, 2 updates, trust **u1** | Separates #1+#3+#4 from #2 disaster; establishes true residual refresh vs baseline 104s | Measurement only | High |
| **P0b** Nest **wall timers under `step1a/refresh`**: `refresh/pack_encode_ffi`, `refresh/obs_host_copy`, `refresh/unpack` (wire `_TimedRF` or equivalent — `record_function` alone does **not** feed `PLO5BP_STEP_TIMERS`) | Decides P1 (copy) vs P2 (encode/pack API) | Unblocks ROI ranking | High |
| **P0c** Confirm log line `[obs-encoder] rust=True OBS_DIM=1171 …` on pod | Guards against silent numpy fallback | 0 if green; **large** if rust=False | High |

**Exit:** know (1) clean refresh seconds, (2) % FFI vs host copy vs unpack.

### P1 — Cheap residual wins (after P0; prefer if host-copy or FFI alloc shows up)

| Work | Idea | Est. impact on refresh | Est. update wall | Risk | Confidence |
|---|---|---|---|---|---|
| **P1a** Reuse / write-into preallocated `self._obs` from Rust (or eliminate `asarray` + `copyto` double materialization) | Today every refresh builds a new (N,1171) f32 in PyO3 then copies. ~7–8 MB/sub × 7.4k ≈ tens of GB of traffic/update | **10–25%** of refresh if copy/alloc is material | **~2–5%** update | Low if bit-exact + same dtype/contig | Med |
| **P1b** Reuse aux buffers in `_unpack_post` (legal, commits, …) instead of fresh `np.asarray` every call | Same pattern for smaller arrays | **3–10%** refresh | **~1–2%** update | Low | Med-low |
| **P1c** Docs/test hygiene only | `test_encoding_rust.py` comments still claim force-disable / 1020 while `_RUST_ENCODER_CURRENT=True` and Rust `obs_layout::OBS_DIM=1171` | 0 wall | — | None | High |

**Do not** ship P1 without P0 split — if FFI encode is 90% of the 14 ms, copy opts are noise.

### P2 — Encode/pack depth (only if P0 says encode- or pack-bound)

| Work | Idea | Est. impact | Risk | When |
|---|---|---|---|---|
| **P2 / #3-B** Pack once for caches + encode **live indices only** from that pack (Rust API) | Avoids Rayon over terminal rows; early-out already cheap → **likely small** unless pack is fused wastefully | **5–15%** refresh if many terminals; else **&lt;5%** | Med (API + parity) | Encode dominates **and** terminal fraction high |
| **P2 / #3-C** Thinner cache-only pack (no full feature pack when encode skipped) | All-terminal path already pack-without-encode | Small–med on all-term waves only | Med | Pack dominates all-term |
| **P2d** Rayon / chunk micro-opts inside `encode_indexed` | After real encode profile | Unknown, usually small | Low–med | Last resort |
| **P2e** Cut `TRAIN_OPP_OUTCOME_MC` below 384 | **Forbidden** without quality A/B (memory non-goal) | — | High quality risk | Do not |

### P3 — Explicitly NOT now

| False lead | Why |
|---|---|
| Re-enable / re-port Rust encoder | Already on @ 1171 |
| Re-fix double-pack | Already fixed; pod flat |
| Dirty live-row skip-encode (#3-A) | Lockstep apply dirties almost all live rows; high silent-wrong-obs risk |
| Re-enable `PLO5BP_ROLLOUT_OVERLAP` | **+41%** wall; leave OFF |
| Expand full #4 (S3/S5/payouts) under this PR | #4 S1/S2 already shipped; reopen only if clean profile still shows 9d/9a hot |
| Multiconfig “stop recreating env” | Already `env_cache` + `reconfigure` |
| Raise `--cpu-threads` past 32 | Known `_concat_batches` regression history |
| Numba whole encoder | Prod path is Rust |
| Drop OBS dims / approximate equity | Design-gated |

---

## End-to-end math (honest)

Assume clean residual refresh **R ≈ 100–110 s**, rollout **~450–650 s** depending on isolation run.

| If refresh drops… | Δ refresh | Δ update (≈ R / total) |
|---|---|---|
| **10%** (copy hygiene only) | −10–11 s | **~2%** |
| **20–30%** (copy + real encode/pack win) | −20–33 s | **~4–7%** |
| **40%+** | Unlikely without quality cuts or structural redesign | — |

**#3 alone will not halve the update.** Combined with already-shipped #1/#4 (and #2 left OFF), host pole shrinks a few more percent. Inference groups (learner/opp) remain ~half of timers — separate from #3.

---

## Test guarantees (keep green)

| Test | Guarantees |
|---|---|
| `test_refresh_encode_mask.py` | Live rows bit-exact vs full `_refresh`; terminal obs all-zero; **all** non-obs caches match; rust path with `PLO5_RUST_ENCODER=1`; `all_hole_cards_subset` parity |
| `test_refresh_subset_parity.py` | Post-`reset_terminal` `_refresh_subset` ≡ full `_refresh` for every cache field |
| `test_encoding_rust.py` | 3-way scalar ↔ numpy ↔ rust full **1171** (enabled via `_RUST_ENCODER_CURRENT=True`; **comments stale** — fix in P1c) |
| `test_encoding_batch.py` | Width gate / refuse stale encoder |

Any Rust write-into or pack/encode API change: run full suite above + `test_env_batched.py` + `test_encoding_batch.py`.

---

## First experiment (exact)

**No code required for P0a** if timers already deployed; P0b needs a tiny instrumentation PR first (recommended).

### Pod profile (isolation)

```bash
# On pod, after stop flag / pause guardian as usual
export PLO5_RUST_ENCODER=1
export PLO5BP_STEP_TIMERS=1
unset PLO5BP_ROLLOUT_OVERLAP   # critical: must be OFF

# Same recipe as guardian (num_envs 49134, mix 10×3, cpu-threads 32),
# --num-updates 2, warm from current vSix4.pt (or profile stem),
# --lr-warmup-updates 0
# Log: runs/vSix4_profile_no_overlap.log
```

**Compare update-1 to baseline** `memory/rollout_saturation_profile_2026-07-13.md`:

| Metric | Baseline | Success for “#3 done enough” |
|---|---|---|
| `step1a/refresh` | ~104 s | Clear drop **or** proven encode-bound residual |
| Rollout wall | ~425 s (pre-#2) | No regression vs clean #1+#3+#4 |
| `[obs-encoder] rust=` | True @ 1171 | Must stay True |
| GPU/CPU util | ~11% / ~22% of 40.8 | Informational |

### First code PR (if P0b not already present)

1. Add nested `_TimedRF` under `_refresh` / `_refresh_subset` for pack_encode_ffi / host_obs / unpack (or fold into existing step timer names).
2. Fix stale comments in `test_encoding_rust.py` only (P1c can ride along).
3. **No** maturin required for timer-only PR.

### Second PR (only after P0 numbers)

- If **host copy ≥ ~20% of refresh** → P1a/P1b (may need Rust “encode into existing buffer” → **maturin rebuild + pod deploy**).
- If **FFI encode ≥ ~70%** → consider P2-B carefully; default bias is **accept residual** and move host effort to remaining #4 / inference, not dirty-encode.
- If **refresh already ≤ ~70s** on clean profile → **stop #3**; document and train.

---

## Blocked on maturin / pod deploy?

| Change | Maturin? | Pod deploy? |
|---|---|---|
| Nested step timers (Python only) | No | Yes (scp/restart) for measure |
| Comment hygiene | No | No |
| Write-into / buffer reuse Rust API | **Yes** | Yes |
| Pack-once encode-live-only API | **Yes** | Yes |
| Current double-pack fix (already in tree) | No (Python-only) | Must be on pod already for A/B; if pod Python lags local, scp `env_batched.py` |

---

## Success criteria

| Criterion | Pass |
|---|---|
| Clean OVERLAP=0 profile exists | Required |
| `rust=True`, OBS_DIM=1171 on that run | Required |
| Refresh vs baseline 104s | Target **≥15–25%** drop **if** a residual fix ships; else document “encode-bound residual, #3 closed” |
| Parity tests green | Required for any code change |
| No policy-metric shift beyond noise | Required |
| OVERLAP remains default OFF | Required |

---

## Recommended decision now

1. **Do not re-implement double-pack** — already done.  
2. **Do not touch #2.**  
3. **P0a immediately** (isolation profile). Prefer **P0b timers** in the same small PR if easy.  
4. Only then spend engineering on P1/P2; expected **honest #3 residual ceiling ~5–7% update** unless copy/FFI proves fat.  
5. If clean profile still has refresh ~100s and encode-bound with early-out terminals: **close #3 as done**, pivot host time to remaining terminal gather (S3) or inference batching — larger poles.

---

## Files (if implementing after measure)

| File | Touch |
|---|---|
| `python/plo5bp/env_batched.py` | Nested timers; optional buffer reuse call sites |
| `python/plo5bp/rollout.py` | Only if timer aggregation needs names |
| `rust_engine/src/bindings.rs` | Only for write-into / encode-from-pack API |
| `tests/python/engine/test_encoding_rust.py` | Un-stale comments |
| `memory/rollout_saturation_profile_2026-07-13.md` | Append clean A/B |

Ready to implement **P0 instrumentation** when approved; **no residual encode rewrite until numbers land.**
