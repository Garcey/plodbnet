# Teacher batch after infoset-BR (2026-08-13)

4 HU river micro roots, 20k iters, iso OFF, floors ON (expl 1.0 / visit 1.0 / 15% holdout). All four **completed**.

| Root | split | expl_bb |
|---|---|---|
| s3_s3_i0 | holdout | 0.776 |
| s3_s3_i1 | train | 0.832 |
| s3_s3_i2 | train | 0.673 |
| s3_s3_i3 | train | 0.9998 |

Export: train 60237 / holdout 20057; 14834 `low_visit`.  
Train: `checkpoints/gto_teacher_ibr.pt` 256×2, 4 ep, final_loss 0.067.  
Probe PASS (pure_agree 0.992, mean_gate_kl 0.075). Stamped — every trained root expl ≤ 1.0.
