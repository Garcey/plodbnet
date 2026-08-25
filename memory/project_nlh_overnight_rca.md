# Overnight CFR campaign RCA (2026-08-12)

**Symptom:** `data/cfr/overnight/` never produced strategies. `status.json`
stuck at `"phase": "start"`, `strategies/` empty, `markers/` empty,
`overnight.err.log` empty, `overnight.log` has one line:

```
[overnight] blueprint bp_hu100_coarse budget=5400s …
```

Grid: `data/cfr/overnight_grid.json` (`overnight_2026-07-16_native_fullhand`).
First job is HU 100bb preflop MCCFR, `max_iterations=2e9`,
`time_budget_secs=5400`.

## Cause (best supported)

The process entered `solve()` for job 1 and **never returned**. Evidence:

1. The start print is immediately before `solve()` in
   `python/plo5bp/gto/cfr_overnight.py` (`_run_blueprint`).
2. `status.json` is written as `phase=start` *before* the job loop.
   `phase=running` was only written **after** a job completed. Stuck
   `start` ⇒ first job never completed.
3. No strategy JSON, no marker, no stderr. Matches a **killed / crashed /
   abandoned in-flight solve**, not a path-bug that skipped work.

Why a kill left nothing (this is the ops bug, not just “user killed it”):

| Gap | Effect |
|-----|--------|
| `SolveConfig.progress_file` never set | Rust cannot dump a mid-solve snapshot. The “kill-safe overnight” claim was false except for a live `STOP` file. |
| `poll_every` default **2000** | `should_stop` only checked time-budget / STOP on iter 1 and every 2000 iters. A slow 100bb HU tree can spend hours inside that window. |
| No in-flight status | After a kill, the dir looks like “never started” instead of “died on `bp_hu100_coarse`”. |
| First job is the longest (1.5h) | One death wastes the night. |

Exact killer (session end, laptop sleep, Ctrl+C, OOM, reboot) is **UNKNOWN**.
Empty stderr argues against a Python exception. A hang on a single MCCFR
iteration is possible but less likely than an external kill of a silent
1.5h blocking call.

## Smallest fix (shipped)

1. **`should_stop`**: wall-clock `time_budget` and `stop_file` are checked
   **every iteration** (`rust_engine/src/cfr/types.rs`). Pause still uses
   `poll_every`.
2. Overnight blueprint jobs now set `progress_file` to
   `strategies/<job_id>.progress.json` and `poll_every=250`.
3. `status.json` goes to `phase=running, current_job=…` **before** `solve()`.
4. On resume, a leftover progress snapshot with infosets is **promoted** to
   a strategy + marker (`promote_progress_if_any`) so a morning kill still
   yields JSON.
5. Non-`ok` solve status raises (fail loudly; no silent `not_implemented`).

Do **not** re-run the 7h grid until dump schema + export gates (Steps 1–2)
are in. When you do, first job can stay 100bb HU but expect a progress file
within the first poll window.

## Verify the fix (no multi-hour run)

```
.venv/Scripts/python -m pytest tests/python/test_cfr_overnight.py -q
```
