---
name: Stop-training protocol
description: When user says stop training, save current-state checkpoint, end the process, and if healthy promote to UI
type: feedback
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
When the user says "stop training" (or equivalent), do all of these without asking:

1. Save a checkpoint at the current update — try a graceful stop first so train.py's final-save block at `scripts/train.py:406-414` writes `args.checkpoint`. If graceful fails on Windows, kill and use the freshest mid-run `<stem>_<update>.pt` from `checkpoints/`.
2. End the training process.
3. Assess health from the log tail (finite losses, kl < 0.05, entropy not blown up to ~2.2 or stuck at 0, v_loss trend not exploding).
4. If healthy, promote the saved checkpoint by `cp <saved>.pt checkpoints/stub.pt` and restart the UI server (per the auto-restart-services rule).
5. If unhealthy, flag it and skip the promotion.

**Why:** User wants stopping training to be one command, not a multi-step dance. The save-first step preserves the training state at the moment of stop (not the last 5-min mid-run checkpoint), which matters when the user has been watching a specific window of progress they want to keep.

**How to apply:** Triggered by phrases like "stop the training", "kill training", "end the run", "stop training and promote". Don't ask "should I save?" or "should I promote?" — both are implied. Only ask if the health assessment is genuinely ambiguous (e.g., mixed signals where some metrics look bad but others look fine).
