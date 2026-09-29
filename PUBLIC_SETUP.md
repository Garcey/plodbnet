# The public build — running it locally, and the one-time account setup

The public build (`PLO5BP_PUBLIC=1`) is the website at wrapgto.com: Study, the
Trainer and the private home games behind a sign-in, with an admin page at `/admin`
(users, comps, the System panel: served models, health, maintenance notice). It has
no live table capture — `/ocr`, `/pokernow` and `/ranges` are stripped. The same
codebase without the flag is the full local build.

**While the models are in development the whole site is free** for signed-in users
(`PLO5BP_FREE_FOR_ALL`, default on): no daily trainer quota, Study unlocked, checkout
closed. The Stripe subscription and the 5-hands-a-day free tier are built and tested
and come back with `PLO5BP_FREE_FOR_ALL=0`.

Production (the server, deploys, models, backups, alerts) is in
[docs/ops/PRODUCTION.md](docs/ops/PRODUCTION.md) — this page is about your own PC.

## Run it on your PC

```powershell
.\run_public.ps1            # reads .env.public; Ctrl+C stops
.\run_public.ps1 -Detached  # in the background
```

or by hand (Git Bash):

```bash
PLO5BP_PUBLIC=1 PLO5BP_DEV_LOGIN=1 \
  .venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8770
```

- `.env.public` (git-ignored) holds your local settings: keep `PLO5BP_BASE_URL=http://127.0.0.1:8770`,
  `PLO5BP_DEV_LOGIN=1`, and only Stripe **test** keys (`sk_test_…` — the app refuses to
  start with a live key next to the dev login or a loopback URL).
- `PLO5BP_DEV_LOGIN=1` = a **loopback-only** fake sign-in (an email box on the landing
  card) so you can use the app without Google. Never start a tunnel from a dev
  machine: a second connector on the `wrapgto` tunnel would take a share of the live
  traffic, and anyone could sign in as anyone.
- Data: SQLite at `data/public.db` (`PLO5BP_DB` to move it), per-user trainer stats
  in `data/trainer_stats/`. Delete the folder to start over.
- The model served is `checkpoints/stub.pt` (no file = a random-init placeholder,
  flagged "untrained" in the UI).
- Admins: `PLO5BP_ADMIN_EMAILS` (comma-separated; the default is the owner's
  address). Admins see the Admin button.

## One-time: Google sign-in

1. https://console.cloud.google.com/ → create (or pick) a project.
2. **APIs & Services → OAuth consent screen**: External, app name, your email; scopes
   `openid`, `email`, `profile`. While the app is in "Testing", add the testers'
   Gmail addresses (or publish it to allow anyone).
3. **Credentials → Create credentials → OAuth client ID** → Web application.
   Authorized redirect URIs: `https://wrapgto.com/auth/callback` (production) and, for
   local testing with real Google sign-in, `http://127.0.0.1:8770/auth/callback`.
4. Put `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` in `.env.public` (local) or
   `/etc/wrapgto/env` (production). The landing card then shows "Sign in with Google".

## One-time: Stripe (only when the paywall is back)

1. https://dashboard.stripe.com/ → an account. Start in **Test mode** (keys `sk_test_…`).
2. Developers → API keys → the **Secret key** → `STRIPE_SECRET_KEY`.
3. On the first checkout the server creates the "$10 / month" price and remembers it
   (or pin one with `STRIPE_PRICE_ID`; change the amount with `PLO5BP_PRICE_CENTS`
   BEFORE the first checkout).
4. Test card `4242 4242 4242 4242`, any future expiry / CVC.
5. Webhooks are optional: subscription status is re-checked with Stripe after the
   paid period ends (and daily), so cancellations are picked up without one. For
   instant updates: `stripe listen --forward-to 127.0.0.1:8770/stripe/webhook` and
   `STRIPE_WEBHOOK_SECRET`.
6. Going live: live-mode keys go ONLY in `/etc/wrapgto/env` on the server. Revenue
   truth lives in Stripe; the admin page mirrors what the app recorded.

## Letting friends in without paying

They sign in once → they appear in `/admin` → **Grant comp** (full access, no
billing); **Revoke comp** takes it back. Stripe subscriptions are cancelled in Stripe
— that button never touches money.

## Free tier (only with `PLO5BP_FREE_FOR_ALL=0`)

- Signed-in non-subscribers: **5 trainer hands a day** (UTC reset, `PLO5BP_FREE_HANDS`),
  counted on New Hand; repeat / review / what-if of those hands are free.
- Study is for subscribers (the server answers 402 and the page offers the upgrade).

## Settings for local runs

| Variable | Default | Meaning |
|---|---|---|
| `PLO5BP_PUBLIC` | unset | 1 = the public build |
| `PLO5BP_BASE_URL` | `http://127.0.0.1:8770` | the site's own URL (sign-in redirects, Stripe return URLs; https ⇒ Secure cookies) |
| `PLO5BP_DEV_LOGIN` | unset | 1 = loopback-only fake sign-in; only registered when the base URL is loopback, and refuses forwarded requests |
| `PLO5BP_DEV_LOGIN_TESTCLIENT` | unset | tests only: also accept Starlette's test client |
| `PLO5BP_DB` | `data/public.db` | the SQLite database |
| `PLO5BP_CHECKPOINT` | `checkpoints/stub.pt` | the PLO5 model |
| `PLO5BP_OBS_REV` | `2` | set `1` to serve a checkpoint trained before 2026-09-20 exactly as trained (`OBS-REV MISMATCH` in the log / `/health` otherwise) |
| `PLO5BP_FREE_FOR_ALL` | `1` | `0` = the paywall and the free-tier quota |

Every other setting — models, home games, limits, e-mail sign-in, Stripe timeouts —
is listed with its default in [ops/env.example](ops/env.example).
