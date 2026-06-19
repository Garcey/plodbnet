---
name: All checkpoints deleted 2026-05-06; representation overhaul incoming
description: Every training run to date failed to converge to a decent policy; user is rewriting the game representation and accepts checkpoint incompatibility
type: project
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
On 2026-05-06 the user deleted everything in `checkpoints/` (1,184
files, ~39 GB) including `stub.pt`. Reason: no training
configuration explored so far has produced a usable policy, and
they are about to make large changes to the observation /
representation that will be incompatible with existing weights
anyway.

**Why:** "I want to make big changes to the game representation
because no training configuration seems to end with anywhere near
decent results. I understand that this means training will be
incompatible with old checkpoints."

**How to apply:**
- The UI will fail at `_load_model` until a new checkpoint is
  promoted to `stub.pt`. Don't try to recover deleted ones.
- Treat representation changes as production-behavior changes
  (per the flag-production-changes rule); they shift the obs
  layout and break weight compatibility.
- Forward-EV reward shipped earlier in `rollout.py` (per-step cost
  + terminal won) and is still the active reward semantics — that
  part of the project notes survives this reset. The earlier
  curriculum-stage notes (`run_forward_ev_hu20_6353.pt`,
  `run_forward_ev_2to6_20bb.pt`) are gone with the rest.
- Don't add a "but keep one for UI testing" caveat back unless
  the user re-introduces it.
