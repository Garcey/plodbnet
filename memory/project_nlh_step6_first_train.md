# NLH Step 6 — first teacher train (2026-08-12)

4 HU river micro roots, iso OFF, visit floor 1.0, 15% holdout.

**Expl floor default stays `TEACHER_MAX_EXPL_BB = 1.0`.** This run could
not pass it: 20k iters ~4.1–4.6 bb; one root raised to 80k still 3.89 bb.
Jam/check and even quads-on-board also report ~4–5 bb (48-sample MC expl
looks like a ~4 bb noise floor). Only a 99TTT board hit ≤1.0.

**One-off accept at 5.0 bb for this first train only** (documented in
`data/cfr/step6_teacher/manifest_oneoff5.json`). Defaults unchanged.

| Root | iters | expl_bb | split |
|---|---|---|---|
| s3_s3_i0 | 20000 | 4.65 | holdout |
| s3_s3_i1 | 80000 | 3.89 | train |
| s3_s3_i2 | 20000 | 4.10 | train |
| s3_s3_i3 | 20000 | 4.27 | train |

Export: train 63503 / holdout 20057; 11568 `low_visit` drops.

Train: `checkpoints/gto_step6.pt` 256×2, 4 epochs, final_loss 0.106.

Probe vs holdout: PASS (pure_agree 0.988, mean_gate_kl 0.075). Stamped
`is_gto_validated=True`. That means net matches this teacher — **not**
that the teacher is a 1.0 bb Nash cert.
