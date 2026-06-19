---
name: Auto-restart services after changes that require it
description: After promoting a checkpoint, editing server code, or anything else that needs a service restart to take effect, do the restart without asking
type: feedback
originSessionId: 0f4de9c9-28ef-47dd-90d1-414a1df544da
---
If an action I just took requires a local service restart to take
effect, do the restart — don't stop to ask. User stated this directly
on 2026-04-20, immediately after asking why I didn't restart the UI
after promoting the 5k checkpoint.

**Why:** the user's default-on expectation is "if you changed
something that needs a restart, you also do the restart." Stopping to
confirm defeats the automation.

**How to apply:**
- Applies to the UI server (`plo5bp.ui.server:app`, port 8765) and to
  any local dev service we own in this repo.
- Workflow: check if an instance is running (`netstat -ano | grep
  :8765` for the UI), stop it, start fresh in background. Report new
  PID / URL.
- Start the UI in background without `--reload` (the reloader spawns
  a subprocess that complicates clean shutdown):
  `.venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8765`
- This is NOT blanket authorization for destructive ops elsewhere
  (force-push, drop tables, kill unrelated processes). Scope is
  services owned by this repo whose restart is the next step of work
  the user just asked for.
- Full guidance also lives in the "Restart services yourself" section
  of `CLAUDE.md`.
