---
name: Verify external preconditions before destructive environment changes
description: Before uninstalling/replacing an environment dependency, run a one-shot probe of the system precondition the new install requires
type: feedback
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
Before any step that destructively replaces a working environment
dependency (uninstalling a torch wheel to install a CUDA build,
swapping a python version, replacing a Tesseract install), first run a
read-only probe that verifies the system-level precondition the new
install needs. If the probe fails, stop and report — don't proceed
into a half-migrated state.

Concrete example: GPU migration plan called for
`pip uninstall torch && pip install torch --index-url ...cu128`. User
required a Step 0: `nvidia-smi` first, gating the uninstall on
"command exits 0, RTX 3070 visible, driver/CUDA fields populated."
Reason: a broken NVIDIA driver leaves you with no working torch at
all once the CPU wheel is uninstalled — recovery is OS-level, not
something the agent can patch from inside the venv.

**Why:** Destructive environment ops are irreversible from inside
the same shell session. If the install fails, you can't reroll
because the prior wheel is gone. A 5-second read-only probe of the
underlying system (driver / kernel module / OS package) catches the
"environment isn't ready" case before you've broken the working
setup. The cost of probing is trivial; the cost of a half-migrated
state is the user has to fix system-level config under time pressure.

**How to apply:** Whenever a plan step is "uninstall X, install Y where
Y needs system feature Z," prepend a Step 0 that probes Z directly.
Examples:
- Replacing torch with a CUDA wheel → `nvidia-smi` succeeds + GPU listed
- Reinstalling Tesseract for a new language → existing binary present
  + `tesseract --list-langs` works
- Switching to a new python minor version → target version available on
  PATH OR system package manager confirms it can be installed
- Wiping a virtualenv → confirm the rebuild script's source-of-truth
  manifest (pyproject, requirements.txt) is committed/pushed first

The probe should be read-only and fast. If it fails, report the
specific failure mode and what needs to happen first; don't try to
fix the system condition from the agent.
