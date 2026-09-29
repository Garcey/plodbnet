# Which model is live on wrapgto.com, and why

One row per change of the live site's PLO5 model (newest last). The live site changes
only with the owner's explicit OK, every time.

How a row gets here: `bash scripts/deploy_prod.sh promote checkpoints/<file>.pt` adds
it (verified with the live code and staged; then **Promote** in /admin → System swaps
it in with no restart — or `RESTART=1` swaps it in at once; a model trained on the other
observation revision takes `OBS_REV=<rev> RESTART=1`, which changes `PLO5BP_OBS_REV` with
it). Fill in the evidence column and commit. The server keeps its own log in `/opt/wrapgto/models.log`
(sha256, time, the pre-flight's findings), and `bash scripts/deploy_prod.sh check`
shows the sha256 of the model being served.

| When (UTC) | Checkpoint | sha256 (first 16) | How | Why / evidence |
|---|---|---|---|---|
| 2026-09-26 08:04 | `vSix5_1248.pt` | (not recorded) | scp + restart | +0.50 sampled / +0.13 argmax bb/seat-hand vs the live vSix4_1240 (h2h_cross, z 16 / 7); entropy 0.10. Backup on the server: `stub.pt.bak-pre-vSix5_1248` |
| 2026-09-26 23:42 | `vSix6_1300.pt` actor + the vSix5_1248 critic | (not recorded) | scp + restart | vs vSix5_1248: sampled +0.34 (z 13), argmax +0.08 (z 4). The old site code could not rebuild vSix6's SiLU critic, so the vSix5 critic serves the review's true EV. Promoted WITHOUT the owner's OK — the reason live changes now always need it. Backup: `stub.pt.bak-pre-vSix6_1300` |
