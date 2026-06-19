---
name: Always pass python -u when redirecting training stdout on Windows
description: Without `python -u` (or PYTHONUNBUFFERED=1), Windows redirected stdout block-buffers for many minutes; the log file stays empty even though training is running on GPU
type: feedback
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
When launching `scripts/train.py` (or any long-running Python
script) and redirecting stdout to a file on Windows, **always**
either invoke as `python -u` or set `PYTHONUNBUFFERED=1` in the
environment.

**Why:** CPython block-buffers stdout when it isn't a TTY. On
Windows + bash redirect (`> log.txt 2>&1`), the buffer can hold
several MB before flushing — meaning a training run that's
actively pegging the GPU writes nothing to the log for 5–15
minutes. This has burned us at least twice; previously the only
"fix" was killing the run and relaunching, which masked the real
cause. The actual fix is unbuffered output.

**How to apply:**
- Default form for background launches:
  ```
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 .venv/Scripts/python -u \
    scripts/train.py ... > logs/<name>.log 2>&1
  ```
- The `-u` flag is sufficient on its own; `PYTHONUNBUFFERED=1` is
  belt-and-suspenders. Use both — they are cheap.
- If you launch a script and the log file stays empty for >2
  minutes while the process is alive and consuming GPU/CPU, the
  cause is buffering. Don't kill-and-relaunch hoping for the best;
  add `-u` and relaunch.
- Diagnostic ladder:
  1. `wc -c <log>` shows 0 bytes after several minutes.
  2. `tasklist | grep python` confirms process alive.
  3. `nvidia-smi` shows GPU utilization > 0.
  4. → buffering, not a hang. Kill, add `-u`, relaunch.
