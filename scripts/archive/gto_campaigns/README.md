# Archived NLH GTO campaign scripts

One-off drivers of the 2026-07/09 teacher campaigns (Step 6: the first HU-river
teacher batch at SPR 2; Step 7: river boards x SPR {1, 2, 3, 5}) and of the
4-handed push/fold study. They hard-code their output folders
(`data/cfr/step6_teacher`, `data/cfr/teacher_s7`, `data/cfr/pushfold_*`), job ids
(e.g. `s3_s3_i1`) and checkpoint names, so they are kept for the record and for
re-running THAT campaign — not maintained as general tools (TOOL-060).

They now use only the public `plo5bp.gto.cfr_batch` API (`run_job`,
`record_result`, `strategy_path` / `rejected_path` / …, `clear_job`,
`teacher_split`, `run_jobs_incremental`) and `plo5bp.gto.jsonio`; nothing
imports another script. Run them from the repo root, as before:
`.venv/Scripts/python scripts/archive/gto_campaigns/step7_teacher_campaign.py`.

| script | what it was for |
|---|---|
| `step6_teacher_run.py` | Step 6 batch: HU river grid, full budget, split printed |
| `step6_find_pass.py`, `step6_scout_jamcheck.py`, `expl_after_fix.py` | Step 6 probes: which iteration budget passes the cap; jam/check scouting; re-measure after the 2026-09-20 fixes |
| `step6_raise_one.py`, `step6_oneoff_accept.py` | re-solve one rejected root at a higher budget; the one-off 5 bb acceptance |
| `unstamp_step6_gto.py` | removed the GTO badge from the Step 6 checkpoint after the review |
| `step7_teacher_campaign.py` | Step 7 campaign (pass 1 + retry of rejected roots) |
| `step7_retry_close.py` | retry only close-to-cap rejected roots (holdout first) |
| `step7_export_train_probe.py` | export → train → probe → stamp for the Step 7 labels |
| `solve_pushfold_4handed.py`, `run_pushfold_300k.py` | the 4-handed 10 bb push/fold solves (20k / 300k iterations) |
| `summarize_pushfold.py` | per-node push/fold summary for a Monker comparison |

For a NEW teacher run use the maintained entry points — see
`docs/ops/GTO_PIPELINE.md` (batch → export → train → probe → serve).
`scripts/export_pushfold_14_charts.py` (the 14-chart exporter) stays in `scripts/`:
the CFR app tests build their push/fold fixture with it.
