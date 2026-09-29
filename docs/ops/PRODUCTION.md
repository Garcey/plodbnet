# wrapgto.com — production runbook

**The rule: nothing changes on the live site — code, model or data — without the
owner's explicit OK, every time.** Claude prepares the change (commits, runs `pack`,
`check` and the tests, gathers the evidence) and hands over the one command; the
owner runs it.

- Server setup, rebuild and the files it runs: [ops/SERVER_SETUP.md](../../ops/SERVER_SETUP.md)
- Every setting the app reads: [ops/env.example](../../ops/env.example)
- Which model is live and why: [docs/models.md](../models.md)
- Setting up a PC or the training pod: [SETUP.md](../../SETUP.md)

## Where things run

wrapgto.com is one Hetzner VPS: the app (`wrapgto.service`, uvicorn on
127.0.0.1:8770, code in `/opt/wrapgto/app`), reached only through a Cloudflare tunnel
(`cloudflared.service`) — the server has no open web ports, and SSH (port 22) is the
only way in. Log in with the ssh alias `wrapgto-prod` (see "A new machine" below);
the address is in the Hetzner console and deliberately not in the repository.

**What is live right now:** `bash scripts/deploy_prod.sh check` (read-only) prints the
live commit, the served model's sha256, the app's own health verdict, games in
progress, the newest backup and its off-site copy, the rollback targets, the last
changes, and the service definition (what it runs, where its settings come from).
From anywhere: https://wrapgto.com/health (`build.commit`, `model_loaded`, …).

On Windows, run the commands below in **Git Bash** — or from PowerShell with Git
Bash's full path (there is no `bash` on PowerShell's PATH):
`& "C:\Program Files\Git\bin\bash.exe" /c/Users/themi/plodbnet/scripts/deploy_prod.sh check`

## Every change finishes on the server, whatever happens to your window

Once a deploy, rollback, restore or model swap has started, **the server does it by
itself**: your window only shows its log. Closing the window, Ctrl+C, a laptop going
to sleep or a dropped connection never stops it half-way — it finishes, or puts
everything back, on its own.

- See it again (or how it ended): `bash scripts/deploy_prod.sh watch`
- Only one change runs at a time; a second one says so and changes nothing.
- The log of every change stays in `/opt/wrapgto/deploys/` on the server, and one line
  per change in `/opt/wrapgto/deploys.log`.
- If the **server itself** crashed or rebooted in the middle of a change, every later
  change refuses to start and says so. Then run
  `bash scripts/deploy_prod.sh recover` (type RECOVER): it finishes the change if the
  site is healthy with it, or puts the previous state back — code, model, settings or
  database. `check` and the phone alerts also say when this is needed.

## Deploy code

1. Commit (the script refuses uncommitted changes in what ships — the live site is
   always a commit) and push; wait for CI to go green (the script refuses a commit
   whose CI failed when the GitHub CLI `gh` is installed).
2. Optional: `bash scripts/deploy_prod.sh pack` — offline, exactly what would ship;
   `DRY_RUN=1 bash scripts/deploy_prod.sh` — every local step, nothing sent.
3. `bash scripts/deploy_prod.sh` — on the server, in order: unpack into a staging
   folder · build the engine · pre-flight: start the NEW code as the service user
   with the production settings, the live model and a copy of the live database, and
   check the engine was built from this commit's Rust sources (anything wrong stops
   here, live site untouched) · stop if a home-game hand is being played or a game
   is running · fresh database backup · switch (one restart) · health check ·
   automatic rollback if unhealthy. Then it tags the commit `prod-<time>`.
   **How long:** the build takes a few minutes (the first one after a Rust change,
   or on a new server, 10+ minutes); the site itself is down only for the switch —
   about 1–2 minutes (the old app gets up to 30 s to close its live connections,
   then the new one loads torch and the models before it answers).
4. Glance at `bash scripts/deploy_prod.sh check`.

**After any restart** (deploy, rollback, restore, a model swap with `RESTART=1`),
home-game tables that were dealing come back **paused**: each host presses Start.
Everyone's open Study spot and trainer hand are forgotten. For a planned restart
during busy hours, post a notice first: /admin → System → Maintenance.

**Is anyone playing?** The deploy asks the running app, which knows exactly which
tables have a hand in play or a game running. It stops (and changes nothing) while
one does — try again when the game is over, or ask the host to pause the table.
Hands that were cut short by an earlier restart are listed as "void, not blocking".
(The very first deploy from the older app can only look at the database; it treats a
hand that finished in the last 15 minutes as a game in progress.)

Settings, written before the command (`FORCE=1 bash scripts/deploy_prod.sh`):

| Setting | What it does |
|---|---|
| `FORCE=1` | skip ONLY the "is anyone playing?" check: a hand being played is void (everyone keeps the chips they had before it) and running tables come back paused. Nothing else. |
| `SKIP_ENGINE=1` | reuse the live engine instead of building it — accepted only when it was built from this commit's Rust sources (a Python-only change). |
| `ENGINE_WHEEL=<file.whl>` | CI's Linux wheel instead of compiling on the server (must be built from this commit). |
| `ALLOW_ENGINE_MISMATCH=1` | accept an engine NOT built from this commit's Rust sources. The engine deals, applies the betting rules and pays out every pot — almost never right. |
| `ALLOW_DIRTY=1` · `ALLOW_RED=1` · `ALLOW_NO_CRITIC=1` | ship uncommitted changes · deploy although CI failed · accept a model whose critic does not load. |

## Put a new model live

1. Evidence first: head-to-head against the live model (`scripts/h2h_cross.py`,
   sampled and argmax, several checkpoints — one checkpoint swings ±0.1–0.2).
2. The owner's OK.
3. `bash scripts/deploy_prod.sh promote checkpoints/<file>.pt` — uploads it, checks
   its sha256, loads it with the LIVE code under the production settings (actor,
   critic, observation revision) and stages it as `stub.pt.new`. Then **/admin →
   System → Promote**: swapped in with no restart (the current one is kept as
   `stub.pt.prev`). `RESTART=1` in front swaps it in at once with a restart instead
   (home-games check, health check and automatic revert included).
4. Fill in the "why" of the row the script added to [docs/models.md](../models.md) and commit.

Undo: /admin → System → Roll back (no restart), or `bash scripts/deploy_prod.sh promote-undo`
(a restart).

**A model trained on the other observation revision** (the pre-flight says so and
prints the exact command): put `OBS_REV=<its revision> RESTART=1` in front, e.g.
`OBS_REV=2 RESTART=1 bash scripts/deploy_prod.sh promote checkpoints/<file>.pt`. It
changes `PLO5BP_OBS_REV` in `/etc/wrapgto/env` **together with** the model, restarts,
checks the app now runs that revision — and if the site is not healthy, puts BOTH back.
Do not edit `/etc/wrapgto/env` by hand for this. (`promote-undo` takes `OBS_REV` the same
way when the previous model was on the other revision.) The no-restart /admin path
refuses a revision change, and also refuses when `/etc/wrapgto/env` was edited without
a restart — the running app would serve the model on the wrong revision.

## Roll back code

`bash scripts/deploy_prod.sh rollback` lists the earlier releases with their commits
and puts the newest back (type ROLLBACK to confirm); `rollback <NAME>` picks one.
It pre-flights that version against today's model and data, then does the same
guarded switch with health check — and the version it replaces becomes a rollback
target itself. An old `app-before-*.tgz` copy works too (what belongs to the server —
data, checkpoints, runs, logs — is left out of it).

If that version's pre-flight FAILS, the rollback stops. `ROLLBACK_ANYWAY=1` switches
to it anyway — only when you know why it failed and that it does not matter
(`FORCE=1` does not do this; it only skips the home-games check). The oldest targets
predate `BUILD_INFO.json`: the health check after switching to one can only see that
the app answers — run `check` and open Study for a recommendation afterwards.

## Restore the database

`bash scripts/deploy_prod.sh restore-db` lists the backups;
`bash scripts/deploy_prod.sh restore-db <NAME>` checks it, backs up the current
database, swaps it in (type RESTORE), health-checks, and puts the previous one back
if the app does not come up. The replaced database stays in
`/opt/wrapgto/backups/replaced-<time>/`: `public.db` there is one self-contained copy
(its `README.txt` says how to put it back: `restore-db replaced-<time>/public.db`),
`raw/` the files exactly as they were. Everything since that backup is lost — only
for a real disaster. The off-site copy: ops/SERVER_SETUP.md, "Backups".

## Alerts and what to do

| Alert | Meaning | First steps |
|---|---|---|
| Uptime monitor: wrapgto.com down | nobody can reach the site | `check`; if the app is up, the tunnel or Cloudflare: `ssh wrapgto-prod systemctl status cloudflared` |
| ntfy "app service is not running" / "/health …" | the app crashed or serves a broken model | `ssh wrapgto-prod journalctl -u wrapgto -n 80`; a bad deploy → `rollback`; a bad model → `promote-undo` |
| ntfy "… interrupted half-way" | the server crashed or rebooted during a change | `bash scripts/deploy_prod.sh recover` |
| ntfy "disk … % full" | the disk is filling | `ssh wrapgto-prod du -sh /opt/wrapgto/* /var/log/journal`; old releases and backups are safe to trim |
| healthchecks.io "watch" late | the server stopped reporting — probably down | Hetzner console: is the server running? |
| healthchecks.io "backup"/"drill" failed | last night's backup or the monthly restore test failed | `ssh wrapgto-prod journalctl -u wrapgto-backup -n 50` |

(The watch stays quiet while a deploy is running — the app is down for a minute on
purpose — unless the deploy runs for over an hour.)

## A new machine (SSH access)

Every machine gets its OWN key — never copy a private key between machines; a lost
PC is then one line to revoke.

1. PowerShell: `ssh-keygen -t ed25519 -C "wrapgto-<machine>" -f "$env:USERPROFILE\.ssh\wrapgto_<machine>"` — with a passphrase.
2. Once, in an ADMIN PowerShell: `Get-Service ssh-agent | Set-Service -StartupType Automatic; Start-Service ssh-agent`;
   then (normal shell) `ssh-add "$env:USERPROFILE\.ssh\wrapgto_<machine>"` — the
   passphrase is typed this once.
3. From a machine that already has access, append the new `.pub` line to the
   server's `/root/.ssh/authorized_keys`.
4. `~/.ssh/config`: `Host wrapgto-prod` / `HostName <the server's address, from the Hetzner console>` /
   `User root` / `IdentityFile ~/.ssh/wrapgto_<machine>` / `IdentitiesOnly yes`.
5. `bash scripts/deploy_prod.sh check` must say "connected" and "app service: active".
   The script uses Windows' own `ssh.exe` (the one that talks to the agent) and
   ships everything as one checksummed bundle — PowerShell pipes are not binary-safe,
   so never `tar | ssh` from PowerShell.

To revoke a machine: delete its line from `authorized_keys`. The Hetzner firewall
should allow port 22 only from your own addresses.

## Production settings that matter

The full, commented list is [ops/env.example](../../ops/env.example) (in
`/etc/wrapgto/env` on the server — keep EVERY setting there: the deploy's checks, the
backup and the service all read it; `check` warns about settings kept anywhere else).
The ones to know:

- `PLO5BP_OBS_REV=1` — the served checkpoints predate the 2026-09-20 observation fix;
  it changes only together with a model (`OBS_REV=… RESTART=1 … promote`, above).
- `PLO5BP_FREE_FOR_ALL` (default 1) — the site is free for signed-in users while the
  models train; `0` brings the paywall and the free-trainer quota back.
- `PLO5BP_HOMEGAME_FAIR` (default 1) — the home games' verifiable shuffle; needs the
  engine's `reset_with_deck` (deploys build it).
- `PLO5BP_TRAINER_STATS=/opt/wrapgto/app/data/trainer_stats.json` — keeps every file
  the app writes inside `data/` (what the hardened unit allows).
- Never in production: `PLO5BP_DEV_LOGIN`, `PLO5BP_BUILD_COMMIT` (deploys refuse it: it
  would hide which code answers), a Stripe `sk_live_` key anywhere but here.
