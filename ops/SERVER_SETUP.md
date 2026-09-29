# Production server — from a fresh machine to the live site

Everything the server runs is in this folder, so the server can be rebuilt (or
reviewed) from the repository alone. Day-to-day operations — deploy, promote a
model, roll back, restore — are in [docs/ops/PRODUCTION.md](../docs/ops/PRODUCTION.md).

| File | Goes to | What it is |
|---|---|---|
| `systemd/wrapgto.service` | `/etc/systemd/system/` | the web app, sandboxed |
| `env.example` | `/etc/wrapgto/env` (600) | the app's settings — every variable it reads |
| `cloudflared-config.example.yml` | `/etc/cloudflared/config.yml` | the tunnel (the only way in) |
| `bin/wrapgto-backup` | `/usr/local/sbin/` | nightly backup + encrypted off-site copy + restore drill |
| `backup.env.example` | `/etc/wrapgto/backup.env` (600) | off-site repository + backup monitor |
| `systemd/wrapgto-backup.{service,timer}` | `/etc/systemd/system/` | the nightly backup |
| `systemd/wrapgto-backup-drill.{service,timer}` | `/etc/systemd/system/` | the monthly restore drill |
| `bin/wrapgto-watch` | `/usr/local/sbin/` | health / disk / backup checks → phone alerts |
| `alerts.env.example` | `/etc/wrapgto/alerts.env` (600) | where alerts go |
| `systemd/wrapgto-watch.{service,timer}` | `/etc/systemd/system/` | the checks, every 5 minutes |
| `deploy-remote.sh`, `deploytool.py` | — (uploaded by every deploy) | the deploy's server half |

Commands below run as root on the server (`ssh wrapgto-prod`). `<…>` marks a value
you choose or copy from an account page. **Every path is absolute on purpose:** a
deploy swaps `/opt/wrapgto/app` for a new folder, so a shell that was sitting in it is
left inside the retired copy (`/opt/wrapgto/releases/<time>`) — after a deploy, run
`cd /opt/wrapgto/app` again before any command of your own.

Every deploy puts this `ops/` folder in `/opt/wrapgto/app/ops/`. Before the first
deploy of a brand-new server, copy it from your PC to a place deploys never touch:
`scp -r ops wrapgto-prod:/root/wrapgto-ops` (and `scp requirements/server.txt
wrapgto-prod:/root/wrapgto-ops/`).

## 1. A fresh server (Hetzner Cloud, Debian 12)

A CCX13 (2 dedicated vCPU, 8 GB) is enough; CCX23 when the site grows (resize in the
Hetzner console, ~1 min downtime, nothing else changes).

1. **Firewall first** (Hetzner console → Firewalls): inbound **TCP 22 only, and only
   from your own addresses** (home, phone hotspot…). Nothing else: the website comes in
   through the Cloudflare tunnel, which dials OUT. Attach it to the server.
2. **SSH**: add your machine's public key when creating the server, then
   ```bash
   printf 'PasswordAuthentication no\nKbdInteractiveAuthentication no\nPermitRootLogin prohibit-password\n' > /etc/ssh/sshd_config.d/10-wrapgto.conf
   systemctl reload ssh
   apt-get update && apt-get -y full-upgrade
   apt-get -y install unattended-upgrades fail2ban python3-venv python3-dev build-essential curl git restic sqlite3
   dpkg-reconfigure -plow unattended-upgrades
   ```
   Every machine that deploys gets its OWN key (docs/ops/PRODUCTION.md, "A new
   machine") — never copy a private key between machines.
3. **The service account and folders**
   ```bash
   useradd --system --home-dir /var/lib/wrapgto --shell /usr/sbin/nologin wrapgto
   mkdir -p /opt/wrapgto/app/data /opt/wrapgto/app/checkpoints /opt/wrapgto/backups /etc/wrapgto
   chown wrapgto:wrapgto /opt/wrapgto/app/data /opt/wrapgto/app/checkpoints
   chmod 700 /opt/wrapgto/backups /etc/wrapgto
   ```
4. **Python environment** (Debian 12 ships Python 3.11), made from `/usr/bin/python3`
   — never from a python under `/root` or `/home` (the hardened unit hides those, and
   the app would not start). The CPU build of torch — the default PyPI wheel pulls in
   gigabytes of CUDA libraries:
   ```bash
   /usr/bin/python3 -m venv /opt/wrapgto/app/.venv
   /opt/wrapgto/app/.venv/bin/pip install -U pip
   /opt/wrapgto/app/.venv/bin/pip install -r /root/wrapgto-ops/server.txt
   echo /opt/wrapgto/app/python > "$(/opt/wrapgto/app/.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/wrapgto-app.pth"
   ```
   (`requirements/server.txt` is the live server's exact package list; if it does not
   exist yet, install from `requirements/server.in` and run
   `bash scripts/deploy_prod.sh freeze` from your PC once the site is up.)
5. **The Rust engine** is built during each deploy. Either give root a toolchain
   (`curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal`, then
   `/opt/wrapgto/app/.venv/bin/pip install maturin`), or deploy with
   `ENGINE_WHEEL=<the Linux wheel CI builds>` and keep compilers off the server. A
   deploy needs about 3 GB free for a build (the pinned toolchain + an optimised build)
   and checks before it starts.
6. **Settings**: `install -m 600 /root/wrapgto-ops/env.example /etc/wrapgto/env`, then
   fill in the Google keys (and Stripe, when the paywall returns). Keep EVERY setting in
   this file — never in the unit's `Environment=` lines: the deploy's checks, the backup
   and `check` read this file.
7. **The app**:
   `cp /root/wrapgto-ops/systemd/wrapgto.service /etc/systemd/system/ && systemctl daemon-reload && systemctl enable wrapgto`.
   Put the model in place first — the deploy refuses to go live without one — from
   your PC: `scp checkpoints/<file>.pt wrapgto-prod:/opt/wrapgto/app/checkpoints/stub.pt`,
   then on the server `chown wrapgto:wrapgto /opt/wrapgto/app/checkpoints/stub.pt`.
   The code arrives with a deploy from your PC: `bash scripts/deploy_prod.sh` (it needs
   `/opt/wrapgto/app` and its `.venv`, which steps 3–4 made).
8. **The tunnel**: install cloudflared (Cloudflare's apt repository, see their docs),
   then follow the comments in `/opt/wrapgto/app/ops/cloudflared-config.example.yml`.
   Check: `systemctl status cloudflared` and https://wrapgto.com/health.
9. **Backups and monitoring**: sections 3 and 4 below.

`systemd-analyze security wrapgto` rates the sandbox (lower is better; the unit
explains the two settings deliberately left out).

## 2. Moving the current server over to this layout

The live server predates this folder. Once, in this order (each step is safe on its
own, and each says how to undo it):

1. `bash scripts/deploy_prod.sh check` from your PC — read-only. Read its "== service"
   part: what the unit runs, where its settings come from (anything kept in the unit's
   `Environment=` lines or in other files is listed with a warning), drop-ins, and which
   python the venv really uses.
2. Keep a copy of the old pieces:
   `cp "$(systemctl show -p FragmentPath --value wrapgto)" /root/wrapgto.service.old; cp /etc/cron.daily/wrapgto-backup /root/wrapgto-backup.old`
   (the second one only if it exists).
3. Settings, safely (a `>>` onto a last line without a newline would glue two settings
   together):
   `grep -q '^PLO5BP_TRAINER_STATS=' /etc/wrapgto/env || printf '\nPLO5BP_TRAINER_STATS=/opt/wrapgto/app/data/trainer_stats.json\n' >> /etc/wrapgto/env`
   — it takes effect at the next restart (step 4). If step 1 listed settings kept in
   the unit itself, add each `NAME=value` line to `/etc/wrapgto/env` the same way (see
   their values with `systemctl show -p Environment wrapgto`).
4. Deploy the current code with the new script: `bash scripts/deploy_prod.sh`. From now
   on the code is root-owned and read-only to the app, and every deploy is a clean,
   detached switch. (It runs the backup script it uploads, so the old daily job no
   longer matters to deploys.) This first build takes longer; see PRODUCTION.md.
5. Install the service definition: `bash scripts/deploy_prod.sh install-unit` (type
   INSTALL). It first checks what would break and refuses — changing nothing — with the
   exact fix if a setting lives only in the old unit, a drop-in in
   `/etc/systemd/system/wrapgto.service.d/` would override the new unit, the venv's python
   is under `/root` or `/home` (hidden by `ProtectHome=yes`), or a setting points where
   the hardened unit cannot write (anything outside `/opt/wrapgto/app/data` and
   `/opt/wrapgto/app/checkpoints`). Then: home-games check, backup, install, one restart,
   health check, and the old definition back automatically if the site is not healthy.
   The new unit runs exactly ONE uvicorn process (never `--workers`: the live home-game
   tables are in that process's memory) with `--timeout-graceful-shutdown 3`.
   To go back to the old definition later, by hand:
   `cp /opt/wrapgto/deploys/unit-before-<time>.service /etc/systemd/system/wrapgto.service && systemctl daemon-reload && systemctl restart wrapgto`
   (install-unit prints the exact file name). If the old unit lived elsewhere (step 1
   shows its path), undo with
   `rm /etc/systemd/system/wrapgto.service && systemctl daemon-reload && systemctl restart wrapgto`.
6. Backups and monitoring as below; then `rm /etc/cron.daily/wrapgto-backup` (the timer
   replaces it).
7. `bash scripts/deploy_prod.sh freeze` from your PC → commit `requirements/server.txt`.

## 3. Backups

The database holds users, subscriptions, every home-game hand and ledger, the
cookie-signing secret and the shuffle keys. Local snapshots alone die with the disk,
the server or the hosting account — so a second, **encrypted** copy goes to another
provider, and a monthly drill proves it restores.

1. Install the script and its timers (after the first deploy, which puts `ops/` in place):
   ```bash
   install -m 755 /opt/wrapgto/app/ops/bin/wrapgto-backup /usr/local/sbin/wrapgto-backup
   cp /opt/wrapgto/app/ops/systemd/wrapgto-backup.service /opt/wrapgto/app/ops/systemd/wrapgto-backup.timer \
      /opt/wrapgto/app/ops/systemd/wrapgto-backup-drill.service /opt/wrapgto/app/ops/systemd/wrapgto-backup-drill.timer \
      /etc/systemd/system/ && systemctl daemon-reload
   systemctl enable --now wrapgto-backup.timer
   /usr/local/sbin/wrapgto-backup          # the first run, by hand
   ```
2. **Off-site (Backblaze B2, ~free at this size)**: create a B2 account, a private
   bucket, and an application key limited to that bucket. Create the password:
   `python3 -c "import secrets; print(secrets.token_urlsafe(32))" > /etc/wrapgto/restic-password;
   chmod 600 /etc/wrapgto/restic-password` — and **store the same password in your
   password manager** (without it nobody can read the copy, you included).
   `install -m 600 /opt/wrapgto/app/ops/backup.env.example /etc/wrapgto/backup.env`, fill
   it in, then `set -a; . /etc/wrapgto/backup.env; set +a; restic init` and run
   `/usr/local/sbin/wrapgto-backup` again: it ends with "off-site copy: ok".
   (A Hetzner Storage Box over sftp works too, but it is the same account as the server.)
3. **The drill**: `systemctl enable --now wrapgto-backup-drill.timer`; try it now with
   `/usr/local/sbin/wrapgto-backup --drill` → "restore drill: ok".
4. **Know when it fails**: at https://healthchecks.io make two checks — "backup"
   (period 1 day, grace 2 hours) and "drill" (period 31 days) — and put their ping URLs
   in `/etc/wrapgto/backup.env`. Missing or failed runs then email you.

After a change to `ops/bin/wrapgto-backup` in the repo, re-run the `install` line above
after the next deploy (deploys themselves always run the copy they upload).

Restoring: `bash scripts/deploy_prod.sh restore-db` lists the local backups and puts
one back safely. From the off-site copy (the server is gone), on the new server with
the same `/etc/wrapgto/backup.env`, BEFORE the app's first start:
```bash
set -a; . /etc/wrapgto/backup.env; set +a
restic restore latest --target /root/restore
gunzip -c "$(ls /root/restore/opt/wrapgto/backups/public-*.db.gz | tail -1)" > /opt/wrapgto/app/data/public.db
tar -xzf "$(ls /root/restore/opt/wrapgto/backups/data-*.tgz | tail -1)" -C /opt/wrapgto/app/data
chown -R wrapgto:wrapgto /opt/wrapgto/app/data     # a root-run restore leaves files root-owned
chmod 600 /opt/wrapgto/app/data/public.db
```

## 4. Monitoring

1. **From outside** (catches everything, including the whole server or Cloudflare
   being down): a free UptimeRobot (or Better Stack) HTTP monitor on
   `https://wrapgto.com/health`, every 5 minutes, alerting your email / phone. The app
   answers 503 when its model is broken, so the monitor catches a bad model too.
2. **On the server**:
   ```bash
   install -m 755 /opt/wrapgto/app/ops/bin/wrapgto-watch /usr/local/sbin/wrapgto-watch
   cp /opt/wrapgto/app/ops/systemd/wrapgto-watch.service /opt/wrapgto/app/ops/systemd/wrapgto-watch.timer \
      /etc/systemd/system/ && systemctl daemon-reload
   ```
   Install the ntfy app on your phone, subscribe to a long random topic,
   `install -m 600 /opt/wrapgto/app/ops/alerts.env.example /etc/wrapgto/alerts.env` and
   put that topic in it (and a healthchecks.io "watch" check: period 5 min, grace 10 min),
   then `/usr/local/sbin/wrapgto-watch --test` (your phone buzzes) and
   `systemctl enable --now wrapgto-watch.timer`. It alerts when something breaks (disk
   over 85 %, service or tunnel down, /health unhappy, backups stale, a change
   interrupted half-way), again every 6 h while it stays broken, and once when it is
   fixed. It stays quiet while a deploy is running.
3. **Cloudflare**: Notifications → "Tunnel health" for the `wrapgto` tunnel.

## 5. What is where (quick reference)

- App `/opt/wrapgto/app` (code: root-owned, read-only to `wrapgto`; `data/`,
  `checkpoints/`: owned by `wrapgto`); env `/etc/wrapgto/env`; logs
  `journalctl -u wrapgto` (capped by journald).
- Earlier releases `/opt/wrapgto/releases/<UTC time>` (rollback targets, newest 5, plus
  the newest 2 that failed their health check); environments built for a changed
  `requirements/server.txt` `/opt/wrapgto/venvs/`.
- Every change's log `/opt/wrapgto/deploys/<time>-<mode>.log` (root-only; the lock and
  the journal of a change in progress live there too); one line per change in
  `/opt/wrapgto/deploys.log`; models `/opt/wrapgto/models.log`.
- Backups `/opt/wrapgto/backups` (root-only, 14 days) + the off-site restic repository;
  databases a restore replaced: `/opt/wrapgto/backups/replaced-<time>/` (newest 3).
- Security updates: unattended-upgrades. Reboot-safe: every service is enabled (a
  change the reboot interrupted is finished or undone by `deploy_prod.sh recover`).
