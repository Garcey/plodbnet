---
name: Train indefinitely when update count unspecified
description: When user says "start training" / "train on X" without specifying a number of updates, launch with effectively-unbounded `--num-updates` (e.g. 100000000) and only stop on unhealthy signals — don't let the default 1000 cap end the run quietly.
type: feedback
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
When the user asks to start a training run without specifying a number
of updates ("let's train on clubgg deep distribution", "kick off
training", etc.), pass `--num-updates 100000000` (or similar
effectively-unbounded value) so the run continues indefinitely.

Stop only when the run becomes unhealthy under the established
criteria — entropy fully collapsed (sustained near 0), KL violations
sustained over many updates, NaN/inf in losses. The /loop monitoring
prompt's stop criteria apply.

**Why:** On 2026-04-30 a clubgg_deep training launch used the default
`--num-updates 1000` and finished cleanly at update 1000 while the
metrics were still healthy. The user's intent was to keep training
running until something went wrong, not to stop at the script default.
The default cap silently ended a productive run.

**How to apply:**
- For unspecified-duration training requests: launch with
  `--num-updates 100000000` (or use `--budget-seconds` if a
  wall-clock budget is more appropriate to context).
- If the user specifies a number ("train for 500 updates"), honor
  that exactly.
- Continue /loop health-monitoring; only stop on the explicit
  unhealthy-criteria checklist (entropy collapsed, sustained KL
  violations, non-finite losses).
- Checkpoints save every 300s (engine default) — the longer-running
  run accumulates more snapshots to choose from when promoting.
