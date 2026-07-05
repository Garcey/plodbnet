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

- Deploy code changes: re-run the tar-over-ssh ship (excl. .git/.venv/
  checkpoints/data), then `systemctl restart wrapgto`.
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
| `PLO5BP_FREE_HANDS` | `5` | Free trainer hands per UTC day |
| `PLO5BP_PRICE_CENTS` | `1000` | Monthly price (before first checkout) |
| `PLO5BP_DEV_LOGIN` | unset | 1 = loopback fake sign-in (testing only) |
| `GOOGLE_CLIENT_ID/SECRET` | unset | Google OAuth (sign-in disabled without) |
| `STRIPE_SECRET_KEY` | unset | Stripe (checkout 503s without) |
| `STRIPE_WEBHOOK_SECRET` | unset | Only if running `stripe listen` / hosted webhook |
| `STRIPE_PRICE_ID` | auto-created | Pin an existing Stripe price |
| `PLO5BP_MAX_RUNTIMES` | `300` | LRU cap on in-memory per-user states |
| `PLO5BP_ACTIVE_WINDOW` | `300` | Seconds a user counts as "active" (admin top-bar counter) |
