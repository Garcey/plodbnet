---
name: GPU training is supported (network scaled past CPU sweet spot)
description: GPU/CUDA is the recommended training device for the 2048×4 + residuals architecture; the prior "CPU-only" stance applied while the network was 128×2
type: feedback
originSessionId: 0f4de9c9-28ef-47dd-90d1-414a1df544da
---
Training supports both CPU and CUDA via `--device {cpu,cuda}` on
`scripts/train.py`. CPU was the right default while the network was
128×2 (~112K params): host↔device transfer ate the batched-forward
win at that size. With the 2048×4 + residual-connection architecture
(~14M params), GPU forwards dominate and the migration was completed
on 2026-04-27 (cu128 wheel; RTX 3070).

**Why:** Inverted the prior "CPU-only by design" stance because the
network outgrew the regime where CPU was competitive. The Rust engine
is still CPU-bound (rollout step) but the learner forward + PPO update
is GPU-bound at the new scale. UI inference reads `PLO5BP_DEVICE` and
falls back to CPU automatically if CUDA isn't available, so CPU-only
contributors stay supported.

**How to apply:**
- Default to `--device cuda` for fresh training runs at 2048×4.
- Don't recommend pinning CUDA wheels in `pyproject.toml` — the
  install command for cu128 is in `CLAUDE.md`. CPU-only contributors
  should keep the default torch wheel.
- Don't propose mixed precision (`autocast`, `GradScaler`),
  multi-GPU/DDP, or cudnn-determinism flags without explicit
  user direction; those were all explicitly out of scope of the
  migration.
- The 128×2 net still trains fine on CPU and is a useful smoke
  target — keep it as a regression check.
