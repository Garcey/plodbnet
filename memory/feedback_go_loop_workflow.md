# Go-loop workflow (user preference)

When there is still unfinished work on an active plan (especially NLH GTO), **end the turn by proposing the single next concrete step** so the user can reply **"go"** / **"implement"** and keep looping until the plan is done.

## Rules

1. After meaningful progress, if work remains: state **what you will do next** in 1–3 bullets (files / commands / success criteria), not a long menu.
2. Do **not** wait for the user to re-derive the plan — you know it better; default to the plan’s next ordered workstream.
3. If the user says **"go"** or **"implement"** (or equivalent), execute that proposed next step without re-asking for confirmation of the whole plan.
4. If the user disagrees or steers elsewhere, follow the conversation normally — the go-loop is an optimization, not a jail.
5. Keep proposals **scoped to one workstream chunk** that can finish in a session (e.g. Workstream B code + tests), not “finish all of NLH.”
6. Locked constraints still apply (e.g. do not pause PLO5 runpod; NLH work is local CPU unless user frees GPU).

## Active plan pointer (NLH)

- Mode 0 plan: `.claude/plans/nlh-gto-complete-before-confident-training.md` (A+B done; **C = rust_cfr scale-train**)
- Teacher: **native rust_cfr only** (`project_nlh_gto_status.md`); PLO5 keeps training
- **Inspiration:** MonkerSolver-class (preflop + multiway); **our wedge:** batch + scripting on rust_engine
- Next default offer: **teacher-quality rust_cfr batch → PolicyNet → probe**

