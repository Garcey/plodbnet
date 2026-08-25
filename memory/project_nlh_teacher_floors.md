# NLH teacher floors (Step 5)

**Decision (2026-08-12):** teacher batches apply three quality floors
before PolicyNet training. Configurable; these are the defaults.

| Floor | Default | Where | Effect |
|---|---|---|---|
| Per-root exploitability | `TEACHER_MAX_EXPL_BB = 1.0` | `run_batch` + export | Missing / non-finite / `expl_bb > 1.0` → root **rejected**. Cap unchanged. After the 2026-08-12 infoset-BR fix this bar is reachable (s3_s3_i2 @ 20k = 0.67 bb). |
| Visit / reach | `TEACHER_MIN_VISIT_MASS = 1.0` | export `strategy_to_labels` | If dump `visit_mass` is present and `< 1.0`, DROP `low_visit`. `visit_mass = sum(strategy_sum)` ≈ one DCFR visit. Does **not** apply when `visit_mass` is absent (legacy). Zero-mass still DROP `unused_uniform`. Library default remains `0` (off). |
| Holdout split | `TEACHER_HOLDOUT_FRAC = 0.15`, seed `0` | export | SHA-256 of `f"{seed}\\0{root_id}"`. Writes train JSONL + `<stem>_holdout.jsonl` + `<stem>_split.json`. Probe takes the holdout JSONL. |

Iso policy is unchanged (OFF; raw 52-hot). Export still DROPs
`unused_uniform` / `illegal_fold` / `inconsistent_public` / `iso_without_raw`.

## Code

- Constants + hash split: `python/plo5bp/gto/teacher.py`
- Batch reject → `rejected/{id}.json` + marker `rejected`, not `strategies/`:
  `cfr_batch.run_batch(..., max_expl_bb=1.0)`. `max_expl_bb=None` disables
  (verify / resume-mechanics tests).
- Export: `export_dir` kwargs off by default; `export_teacher_dir` / CLIs on.
- CLIs: `scripts/cfr_batch.py`, `scripts/cfr_export_labels.py`,
  `scripts/train_policy_from_cfr.py`. Probe: `--holdout <stem>_holdout.jsonl`.
