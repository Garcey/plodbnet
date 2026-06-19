---
name: Auto-promote good checkpoints to the UI
description: After a training run finishes, if the checkpoint looks healthy, copy it over checkpoints/stub.pt without asking
type: feedback
originSessionId: 0f4de9c9-28ef-47dd-90d1-414a1df544da
---
After a training run completes, if the checkpoint looks good, push it
to the UI automatically — don't ask first. The user stated this
directly after the 5k batched run on 2026-04-20.

**Why:** the user iterates by running checkpoints and playing against
them in the UI. Stopping to ask "should I promote?" every time adds
friction to that loop. They want the promotion to be the default
post-training step.

**How to apply:**
- "Looks good" = finite losses throughout, entropy dropping or flat
  (not blowing up toward ~2.2, not collapsing too fast), `approx_kl`
  bounded under ~0.05, `v_loss` stable. If any of those are off, flag
  it and do NOT promote — ask the user.
- Mechanism: `cp checkpoints/<run_name>.pt checkpoints/stub.pt`. The
  UI's `_load_model()` falls back to `checkpoints/stub.pt` when
  `$PLO5BP_CHECKPOINT` is unset (see
  `python/plo5bp/ui/server.py:58`).
- The server loads `MODEL` at import time — no hot reload. If a UI
  server is already running, tell the user it needs a restart; don't
  kill the process without asking.
- Full guidance also lives in the "Promote good checkpoints to the UI"
  section of `CLAUDE.md`.
