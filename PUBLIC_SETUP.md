# Public build — setup & operations

## PRODUCTION (current): Hetzner VPS

wrapgto.com is served by a Hetzner CCX13 (2 vCPU/8GB) at **87.99.132.209**
(`ssh root@87.99.132.209`, laptop key). The laptop is DEV ONLY now — its
`run_public.ps1` no longer starts a tunnel, and its `.env.public` is the
dev config (localhost base URL + dev login).

- App: `/opt/wrapgto/app` (repo), venv at `.venv`, runs as user `wrapgto`
  via **`wrapgto.service`** (uvicorn 127.0.0.1:8770; `systemctl
  status|restart wrapgto`). Env: `/etc/wrapgto/env` (prod config: https
  base URL, dev login OFF, Google + Stripe keys).
- Tunnel: **`cloudflared.service`**, config `/etc/cloudflared/config.yml`
  (apex + www → :8770). No inbound ports open except SSH (Hetzner firewall).
- Logs: `journalctl -u wrapgto -f` (capped 200M). Security updates:
  unattended-upgrades. Reboot-safe (verified): both services auto-start.
- Backups: `/etc/cron.daily/wrapgto-backup` → `/opt/wrapgto/backups/`
  (14 daily SQLite snapshots, WAL-safe backup API).
- **Promote a checkpoint to prod** (from the laptop repo):

```bash
scp checkpoints/stub.pt root@87.99.132.209:/opt/wrapgto/app/checkpoints/stub.pt
ssh root@87.99.132.209 systemctl restart wrapgto
```

- Deploy code changes: **`scripts/deploy_prod.sh`** (from the laptop — the
  only key the server accepts). It is the tar-over-ssh ship (excl. .git/.venv/
  checkpoints/data/secrets), then it REBUILDS THE RUST ENGINE on the server
  (`.venv/bin/maturin develop --release` as user `wrapgto`), restarts
  `wrapgto` and checks `/health`. The engine rebuild matters since
  2026-09-25: the home games' verifiable shuffle needs the engine's
  `reset_with_deck`; without it the site still runs and the tables deal the
  old way, shown as "Unverified shuffle" (`SKIP_ENGINE=1` skips the rebuild).
  Kill switch: `PLO5BP_HOMEGAME_FAIR=0` in `/etc/wrapgto/env`.
- **Deploying from a new machine** (one-time, ~10 min). Every machine gets its
  OWN key — never copy a private key between machines; a lost PC is then one
  line to revoke.
  1. On the new machine (PowerShell): `ssh-keygen -t ed25519 -C "wrapgto-<machine>"
     -f "$env:USERPROFILE\.ssh\wrapgto_<machine>"` — give it a passphrase.
  2. Once, in an ADMIN PowerShell: `Get-Service ssh-agent | Set-Service
     -StartupType Automatic; Start-Service ssh-agent`, then (normal shell)
     `ssh-add "$env:USERPROFILE\.ssh\wrapgto_<machine>"`. Windows keeps the
     unlocked key for your account, so the passphrase is typed this once.
  3. From a machine that already has access, append the new `.pub` line to the
     server's `/root/.ssh/authorized_keys`.
  4. `~/.ssh/config` on the new machine: `Host wrapgto-prod` / `HostName <the
     address above>` / `User root` / `IdentityFile ~/.ssh/wrapgto_<machine>` /
     `IdentitiesOnly yes`.
  5. `scripts/deploy_prod.sh check` (read-only) must say "connected", "app
     service: active" and cargo/maturin "ok". The script uses Windows' own
     `ssh.exe` when it exists (that is the one that talks to the agent) and ships
     through a temp archive — PowerShell 5.1 pipes are not binary-safe, so never
     `tar | ssh` from PowerShell. `scripts/deploy_prod.sh pack` shows, offline,
     exactly what would ship.
  To revoke a machine: delete its line from `authorized_keys`.
- Scale-up path: Hetzner console → resize to CCX23 (4 vCPU/16GB), ~1 min
  downtime, nothing else changes.

The public build (`PLO5BP_PUBLIC=1`) serves the trainer + study tabs behind
Google sign-in with a $10/mo Stripe subscription, a 5-hands/day free trainer
tier, and an admin dashboard at `/admin` (user list, comp grants, revenue).
No OCR / live capture is mounted. Everything lives in the same codebase; the
local build (flag unset) is untouched by any of this.

## Run it (laptop)

```bash
PLO5BP_PUBLIC=1 PLO5BP_DEV_LOGIN=1 \
  .venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8770
```

- `PLO5BP_DEV_LOGIN=1` enables a **loopback-only** fake sign-in (email box on
  the landing card) so you can use the app before Google/Stripe are
  configured. It refuses non-127.0.0.1 clients, but still: **never expose a
  tunnel while dev login is on** — anyone could sign in as any email,
  including the admin's. Drop the env var once Google OAuth works.
- Data: SQLite at `data/public.db` (override `PLO5BP_DB`), per-user trainer
  stats at `data/trainer_stats/u<id>.json`. Delete the DB to reset everything.
- The model served is `checkpoints/stub.pt` (same promote flow as always).
- Your admin account: sign in with the Google account for
  `themilesgarcia@icloud.com` (override list via `PLO5BP_ADMIN_EMAILS`,
  comma-separated). Admins bypass the paywall and see the Admin button.

## One-time: Google sign-in credentials

1. https://console.cloud.google.com/ → create (or pick) a project.
2. **APIs & Services → OAuth consent screen**: External, app name, your
   email; add scopes `openid`, `email`, `profile` (non-sensitive). While the
   app is in "Testing" status, add your + your friends' Gmail addresses under
   Test users (or publish the app to allow anyone).
3. **APIs & Services → Credentials → Create credentials → OAuth client ID**:
   Application type **Web application**. Authorized redirect URIs — add:
   - `http://127.0.0.1:8770/auth/callback`
   - `http://localhost:8770/auth/callback`
   (When you later host/tunnel, add `https://<your-domain>/auth/callback`.)
4. Export before launching:

```bash
export GOOGLE_CLIENT_ID="...apps.googleusercontent.com"
export GOOGLE_CLIENT_SECRET="..."
```

The landing card then shows "Sign in with Google" (dev login row disappears
unless `PLO5BP_DEV_LOGIN=1`).

## One-time: Stripe

1. https://dashboard.stripe.com/ → create account. Use **Test mode** first
   (toggle top-right); test-mode keys start `sk_test_`.
2. Developers → API keys → copy the **Secret key**:

```bash
export STRIPE_SECRET_KEY="sk_test_..."
```

3. That's it for the happy path — on first checkout the server auto-creates
   the "PLO5 Bomb-Pot Trainer — Monthly" $10 price and caches its id in the
   DB (or pin one yourself via `STRIPE_PRICE_ID`). Change the amount with
   `PLO5BP_PRICE_CENTS` (default 1000) BEFORE the first checkout.
4. Test a purchase with card `4242 4242 4242 4242`, any future expiry/CVC.
5. Webhooks are OPTIONAL on a laptop: subscription status is lazily
   re-verified against Stripe after the cached period ends (and daily), so
   cancellations are picked up without a public URL. For instant lifecycle
   events later:
   `stripe listen --forward-to 127.0.0.1:8770/stripe/webhook` and export the
   printed `STRIPE_WEBHOOK_SECRET`.
6. Go live: flip to Live mode keys, and manage/cancel real subs from the
   Stripe dashboard. **Revenue truth lives in Stripe**; the admin page's
   revenue/MRR are convenience mirrors of locally recorded events.

## Letting friends in (before Stripe / instead of paying)

They sign in with Google once → they appear in `/admin` → click **Grant
comp**. Comp = full access, no billing. **Revoke comp** takes it back.
Stripe-sourced subs can't be revoked from the dashboard (cancel in Stripe) —
that button never touches money.

## Free tier

- Signed-in non-subscribers: **5 trainer hands/day** (UTC reset), counted on
  New Hand (`PLO5BP_FREE_HANDS` to change). Repeat/review/what-if of those
  hands is free. The remaining count shows in the top-bar pill.
- Study mode is subscriber-only (server-enforced 402 + upgrade modal).

## Exposing it beyond the laptop (later)

Quickest: `cloudflared tunnel --url http://127.0.0.1:8770` (or ngrok). Then:
set `PLO5BP_BASE_URL=https://<tunnel-host>` (OAuth redirects + Stripe return
URLs derive from it), add that `/auth/callback` to the Google client, and
REMOVE `PLO5BP_DEV_LOGIN`. Cookies switch to `Secure` automatically when the
base URL is https. A real deploy (Docker etc.) is the next slice.

## Env reference

| Var | Default | Meaning |
|---|---|---|
| `PLO5BP_PUBLIC` | unset | 1 = public build (auth+billing on, live capture off) |
| `PLO5BP_BASE_URL` | `http://127.0.0.1:8770` | External URL for OAuth/Stripe redirects |
| `PLO5BP_DB` | `data/public.db` | SQLite path |
| `PLO5BP_ADMIN_EMAILS` | `themilesgarcia@icloud.com` | Comma-separated admin allowlist |
| `PLO5BP_FREE_FOR_ALL` | `1` | **1 = the whole site is free** for every signed-in user while the models are in development (no quota, Study unlocked, checkout closed). Set `0` to bring the paywall back |
| `PLO5BP_HOMEGAME_FAIR` | `1` | Home games' verifiable shuffle (sealed deck + the players' cut). `0` = deal the old way. Also off by itself when the engine on this machine predates `reset_with_deck` — rebuild it (`scripts/deploy_prod.sh`) |
| `PLO5BP_HOMEGAME_GRADING` | `1` | Background network grading of every home-game action (`0` = off) |
| `PLO5BP_FREE_HANDS` | `5` | Free trainer hands per UTC day (only when `PLO5BP_FREE_FOR_ALL=0`) |
| `PLO5BP_PRICE_CENTS` | `1000` | Monthly price (before first checkout) |
| `PLO5BP_DEV_LOGIN` | unset | 1 = loopback fake sign-in (testing only). The route is only registered when `PLO5BP_BASE_URL`'s host is loopback, and it rejects any request carrying a forwarding header (XFF, CF-Connecting-IP, Forwarded, …) |
| `PLO5BP_DEV_LOGIN_TESTCLIENT` | unset | 1 = also accept Starlette's `testclient` host (the test fixtures set it; never in a real deployment) |
| `PLO5BP_STRIPE_TIMEOUT` | `8` | Seconds before a Stripe status re-check gives up (runs in a threadpool, never on the event loop) |
| `PLO5BP_STRIPE_GRACE_DAYS` | `3` | On a Stripe ERROR, keep access only until `current_period_end` + this many days ("No such subscription" is INACTIVE immediately) |
| `PLO5BP_STRIPE_RETRY_S` | `900` | Minimum seconds between Stripe re-checks per user while the status is uncertain |
| `PLO5BP_HOMEGAME_MAX_TABLES` | `5` | Open home-game tables per host |
| `PLO5BP_OBS_REV` | `2` | Observation-semantics revision. Set `1` while the served checkpoint was trained before 2026-09-20 (the server logs `OBS-REV MISMATCH` and `/formats` reports `obs_rev_mismatch` when it disagrees with the checkpoint's stamp) |
| `GOOGLE_CLIENT_ID/SECRET` | unset | Google OAuth (sign-in disabled without) |
| `STRIPE_SECRET_KEY` | unset | Stripe (checkout 503s without) |
| `STRIPE_WEBHOOK_SECRET` | unset | Only if running `stripe listen` / hosted webhook |
| `STRIPE_PRICE_ID` | auto-created | Pin an existing Stripe price |
| `PLO5BP_MAX_RUNTIMES` | `300` | LRU cap on in-memory per-user states |
| `PLO5BP_ACTIVE_WINDOW` | `300` | Seconds a user counts as "active" (admin top-bar counter) |
