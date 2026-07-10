# AUTOMATION.md — Claude self-directed operations catalog

Goal: maximize Claude usage for continuous improvement with minimal
human prompting. The user is time-limited and disorganized; Claude is
not. This doc is the standing catalog of automatable work, the
mechanisms that run it, and the invariants automated sessions must
respect. Future Claude sessions: treat this as the execution menu;
BACKLOG.md (once created) is the prioritized queue.

Status 2026-07-05: catalog written, nothing scheduled yet. Starter set
proposed at the bottom awaits user go-ahead.

---

## Mechanisms (how unattended work runs)

Every scheduled session is a fresh Claude session in this repo, so
CLAUDE.md + memory load automatically — cron prompts can be short
("check on training" already carries full meaning via memory).

| # | Mechanism | What it is | Right for | Limits |
|---|---|---|---|---|
| M1 | **Local scheduled tasks** (Claude Desktop) | Cron-style prompts run on this laptop. Set up by telling Claude "schedule X every N hours". | Anything needing local state: ssh keys (pod + Hetzner), `.venv`, local UI, checkpoint files. The ops workhorse. | Laptop must be on and awake. |
| M2 | **Cloud routines** (`/schedule` cloud agents) | Prompts run in Anthropic's cloud against the GitHub repo (`github.com/Garcey/plodbnet`); output = branches/PRs. | Pure code work: bug hunts, tests, refactors, docs. Also public-URL probing (wrapgto.com). Runs while laptop is off. | No laptop ssh keys → cannot touch pod or Hetzner. Cannot restart local services. |
| M3 | **/loop** | Self-paced recurring prompt inside an open interactive session. | Active-babysitting days (anneal in progress, launch day) while you're at the machine anyway. | Dies with the session. |
| M4 | **Hooks** (settings.json, via update-config skill) | Event-driven: after edits / on session end, run a command. | Auto-run pytest after Python edits; session-end HANDOFF.md reminder. | Fires commands, not judgment — keep them cheap. |
| M5 | **Server-side timers (Hetzner)** | One-time: Claude writes a systemd timer on the VPS. Zero Claude tokens, 24/7. | Uptime self-healing: probe wrapgto.com locally, restart service on failure, log for the daily triage to read. | Dumb by design; Claude reads its log later. |

Cross-cutting:

- **Multi-agent sweeps**: putting the keyword `ultracode` in a
  scheduled prompt opts that run into workflow orchestration (dozens of
  parallel agents) — use for bug hunts / audits, not routine checks.
- **Push notifications**: scheduled runs should notify ONLY on action
  taken or blockage, never "all healthy" spam. Digests go to files.
- **Permissions prep (prerequisite for all of M1/M2)**: headless runs
  stall forever on permission prompts. Run `/fewer-permission-prompts`
  and curate an allowlist (ssh to the two known hosts, pytest, cargo,
  maturin, scp to prod, curl to localhost/wrapgto) in
  `.claude/settings.json` before scheduling anything.

Safety defaults (loosen only by explicit user decision):

- Automated code changes land on **branches/PRs**, never direct to
  main. A daily interactive session (or the user) merges.
- **Outward-facing actions are draft-gated**: social posts, emails to
  users, Stripe live-mode changes, spending money → Claude drafts and
  queues, user approves. Training-ops actions (restart, promote,
  anneal) are pre-authorized per existing memory.

---

## 1. Training operations (RunPod)

Highest value; mostly pre-authorized. All M1 (needs pod ssh key).

- **1.1 Babysitter** — every 2–4h: ssh pod, tail guardian log, check
  ent/KL/v_loss/aggression vs protocol, verify anneal state, promote
  latest healthy checkpoint to local UI (per "check on training"
  memory), append one line to `runs/ops_digest.md`. Notify only on
  anomaly or corrective action.
  Prompt: `Check on training. Act autonomously per the collapse/anneal
  protocols. Promote latest healthy checkpoint. Append one line to
  runs/ops_digest.md. Push-notify ONLY if you acted or are blocked.`
- **1.2 Collapse auto-recovery** — already authorized (memory:
  autonomous warm-restart): detect collapse → warm-restart from latest
  clean checkpoint, bump stem, gentle resume, write postmortem. Lives
  inside 1.1.
- **1.3 Anneal driver** — automate the F/T/R stop-loss protocol:
  detect plateau → step entropy via `runs/anneal_control.json` →
  verify aggression holds over the next window → revert on regression.
  Today this is manual judgment per check; the babysitter can own it.
- **1.4 Exploitability cadence** — run the exploit probe against the
  latest checkpoint on a fixed cadence, log the trend to a CSV, flag
  regressions. (`tests/python/test_exploit_smoke.py` just landed —
  wire it in.)
- **1.5 Checkpoint league** — `scripts/evaluate.py` head-to-head: new
  checkpoint vs previous generation N-hand match. Promote-to-prod only
  on winrate evidence + 1.4 not regressing (gates 3.8 too).
- **1.6 Pod resource watch** — disk growth (NEVER prune vFour4_*),
  GPU util sanity, guardian/watchdog process alive. Credits balance:
  can't read it headlessly → notify user to check when runtime math
  says it's close.
- **1.7 Daily training digest** — one file per day summarizing
  updates/hr, entropy trajectory, promotions, incidents; morning
  standup (5.2) reads it.
- **1.8 Experiment queue** — user-approved queue of next configs
  (e.g. PLO4/PLO6 entropy-seed tuning when the pod frees up, nlh5
  hypers). When a run completes/pauses, babysitter launches the next
  entry instead of idling the pod. One family per pod rule applies.
- **1.9 Ops journal** — keep `runs/babysit_observations.md` anchored
  (already the pattern; make it automatic post-incident).

## 2. Codebase quality

Mostly M2 (cloud, runs while laptop is off) with M1 for anything
needing the local `.venv`/maturin build.

- **2.1 Nightly bug hunt** — rotate through subsystems (engine
  bindings, encoding, env_batched, OCR events, server/session, trainer,
  public.py, app.js). Multi-agent review (`ultracode`), adversarially
  verified findings → BACKLOG.md with repro steps.
- **2.2 Auto-fix tier** — the hunt may fix at most one low-risk
  CONFIRMED bug per night on a branch, with a test, as a PR. Everything
  else is backlog-only. (gh is installed; repo has origin.)
- **2.3 Test-gap filler** — weekly: coverage report → write parity
  tests for uncovered paths (fits the repo's bit-exact test culture).
- **2.4 Standing bug sessions** — scheduled deep-dives on HANDOFF.md
  issues, e.g. the open "hero silent CHECK not detected" OCR bug, until
  resolved. These are M1 (needs local fixtures/UI).
- **2.5 Full test matrix nightly** — full pytest (incl. slow/OCR) +
  `cargo test` + a maturin rebuild; bisect any new failure; notify only
  on red. Catches what the fast inner loop misses.
- **2.6 Perf benchmark tracking** — `scripts/profile_rollout.py` on a
  fixed config → CSV; alert on >10% regression.
- **2.7 Dependency & security audits** — weekly `pip-audit` /
  `cargo audit`; run the security-review skill on any diff touching
  public.py / server routes / Stripe code.
- **2.8 Docs drift** — after any substantial merge: does CLAUDE.md
  still match reality? HANDOFF.md stale? Monthly memory consolidation
  (consolidate-memory skill).
- **2.9 Commit hygiene** — end-of-day: draft commits for uncommitted
  working-tree changes (11 modified files sitting right now), grouped
  logically, pushed to a branch for review.

## 3. WrapGTO product & prod ops

- **3.1 Launch blockers (one-shots, do soon)** — ToS + privacy pages,
  Stripe live-mode key swap (user provides the rk_ key; the swap +
  verification is Claude work), exploitability QA pass. These are the
  three items memory says block full launch.
- **3.2 Growth backlog executor** — weekly "ship one growth idea"
  session off the 16-idea memory (suggested order: shareable hand
  permalinks → daily free spot → PWA). M2-able; merge locally.
- **3.3 Uptime self-healing** — M5 systemd timer on the Hetzner box:
  probe the local service, restart on failure, append to a log. Claude
  never burns tokens polling; daily triage (3.4) reads the log. Add a
  `/health` endpoint if one doesn't exist.
- **3.4 Prod error triage** — daily M1: ssh Hetzner, pull
  `journalctl -u wrapgto` since yesterday, classify tracebacks, fix
  trivial ones on a branch, backlog the rest, note restart events from
  3.3's log.
- **3.5 Business digest** — daily: signups, active users, trainer
  hands consumed, free→paid conversion, MRR from `data/public.db` over
  ssh (+ Stripe MCP once user authorizes it in claude.ai connector
  settings; until then sqlite + public.py revenue logic suffice).
- **3.6 Backup verification** — nightly backups already exist; weekly
  M1 task: pull latest backup, restore to a temp SQLite, integrity
  check + row counts. An unverified backup is a hope, not a backup.
- **3.7 UX audit loop** — browser-driven walkthrough of trainer/study
  flows (desktop + mobile viewport), screenshots, concrete polish
  proposals; implement approved ones. Feeds the PWA item.
- **3.8 Model QA harness (promotion gate)** — build once: a fixed
  suite of canonical spots (e.g. 20 hands) with assertion ranges on
  recommendations (no 3-bet-folding the nuts). Every prod promotion
  must pass it. Turns "looks good in the log" into a real gate.
- **3.9 Usage analytics** — which spots/streets users drill most,
  where free users hit the wall → feeds pricing, content (4.1), and
  product priorities.

## 4. Marketing & growth

All content-producing items are draft-gated: Claude writes, user
approves publication. Recommendation: keep it that way — poker
communities are hostile to obvious automation, and brand voice is worth
one minute of your review per piece.

- **4.1 Content engine** — weekly strategy article (PLO5 double-board
  bomb pots is a near-empty SEO niche) + "hand of the week" generated
  from the trainer with EV graphics. Prereq one-shot: build a /blog on
  wrapgto.com (static, server-rendered for SEO).
- **4.2 Social draft queue** — X/Reddit/2+2 post drafts written to a
  queue file with suggested subreddit/thread; you approve & post, or
  approve and Claude posts via browser with you watching.
- **4.3 SEO mechanics** — one-shot then quarterly: meta tags, OG
  cards, sitemap.xml, landing copy variants; monthly ranking spot-check
  via web search.
- **4.4 Competitor watch** — weekly sweep (GTO Wizard, Vision, PLO
  trainers, pricing pages) → diff digest + feature ideas into backlog.
- **4.5 Email drafts** — onboarding sequence + monthly changelog
  newsletter drafted via the connected Gmail (drafts only; sending is
  user-approved).
- **4.6 Changelog page** — auto-generated from commit history on each
  deploy (also one of the 16 growth ideas; cheap trust signal).
- **4.7 Mention monitoring** — weekly search for wrapgto mentions on
  reddit/2+2/Discord aggregators → digest, suggested replies (gated).

## 5. Meta / organization (the disorganization fix)

- **5.1 BACKLOG.md** — single prioritized queue at repo root. Every
  automated session appends findings and picks from the top; weekly
  grooming pass dedupes and re-ranks. Created as part of starter setup.
- **5.2 Morning standup digest** — daily 8am M1 task: training status,
  prod health + yesterday's numbers, what overnight automation did
  (PRs opened, bugs found), backlog top 5. Push-notify a 3-line
  summary; full digest to `runs/standup/<date>.md`.
- **5.3 Idea intake** — you brain-dump one-liners into `IDEAS.md` (or
  email yourself); a scheduled pass triages them into BACKLOG.md with
  effort/value estimates. Zero-friction capture for a disorganized
  human is the whole point.
- **5.4 Session handoff discipline** — session-end hook reminder to
  update HANDOFF.md when an issue is left open (M4).
- **5.5 Memory upkeep** — monthly consolidate-memory run.
- **5.6 Calendar nudges** — calendar MCP is connected; Claude can
  schedule "review Claude's queued drafts" blocks so gated work
  doesn't rot in the queue.

---

## Constraints & invariants for automated runs

- **Laptop-awake dependency**: M1 tasks silently don't run if the
  laptop sleeps. Put pure monitoring on Hetzner (M5), pure code work in
  the cloud (M2). Accept gaps in pod babysitting or keep the laptop on.
- **Pod ssh key rotates on pod restart** (memory: runpod ops). On auth
  failure: notify user for the new console string, do NOT retry-thrash.
- **One pod-touching task at a time** — stagger schedules; guardian
  pgrep rules mean overlapping sessions can misread each other.
- **Respect standing ops law** (all in CLAUDE.md/memory, auto-loaded):
  2048×4 always; no cross-variant warm-start; never prune vFour4_*;
  one watchdog family per pod; promote = read-the-number + SHA-verify;
  never POST to the live :8765 trainer (scratch :8766); UI restarts via
  detached Start-Process.
- **Stripe MCP is unauthorized** until you connect it (claude.ai
  connector settings, or /mcp in an interactive session). Revenue
  reads work via ssh + sqlite meanwhile. Live-mode key changes are
  always user-approved.
- **Usage**: effectively unlimited per user, but start with the
  starter set and scale — more crons = more overlapping context to
  keep coherent, and BACKLOG.md is the coherence mechanism.

## Recommended rollout

**Day 1 (one interactive session, ~1h):**
1. Permissions allowlist (`/fewer-permission-prompts` + manual curation).
2. Create BACKLOG.md, seed from memory (growth ideas, open OCR bug,
   launch blockers, deferred items).
3. Hetzner self-healing timer + /health endpoint (M5, one-shot).
4. Schedule: training babysitter (every 3h) + morning standup (daily).

**Week 1:** nightly bug hunt (M2), nightly full test matrix, launch
blockers 3.1 knocked out one by one.

**Week 2:** model QA harness (3.8), business digest (3.5), backup
verification (3.6), growth executor cadence (3.2).

**Week 3+:** blog + content engine, UX audit loop, competitor watch,
email drips, SEO pass.
