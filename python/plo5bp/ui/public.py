"""Public-build service layer: auth, per-user state, free tier, billing, admin.

Installed by ``server.py`` ONLY when ``PLO5BP_PUBLIC`` is truthy (the same flag
that strips the live-capture routes). The local build never imports this.

What it adds around the existing app:

- **Google sign-in** (Authlib). No passwords stored — the only identity is
  Google's verified email. A loopback-only dev login (``PLO5BP_DEV_LOGIN=1``)
  exists so the flow can be exercised before OAuth credentials exist; the
  route is only registered when ``PLO5BP_BASE_URL`` is itself a loopback
  URL, and it refuses non-loopback clients and anything that arrived
  through a proxy/tunnel (forwarding headers, non-loopback Host).
- **Per-user state.** The study ``Session`` and the ``TrainerSession`` become
  per-user (LRU registry, capacity-capped). Wired via the resolver hooks in
  ``server.py`` / ``trainer.py``; a ContextVar carries the user through the
  request (anyio propagates it into threadpool endpoints).
- **Free tier**: N trainer hands per UTC day (default 5) for signed-in
  non-subscribers, enforced in middleware on ``POST /trainer/new_hand``.
  Study mode requires a subscription. Admins bypass everything.
- **Stripe subscriptions** ($10/mo default). Checkout is verified BOTH via
  the success-redirect (``/billing/confirm``) and the optional webhook;
  status is lazily re-verified against Stripe after period end, so laptop
  hosting works without a public webhook URL.
- **Admin** (email allowlist): user list + usage, comp-subscription
  grant/revoke, revenue metrics. Stripe-sourced subs are managed in Stripe,
  not here — revoking a comp never touches money.

Unit note: money is stored in cents; days are UTC ``YYYY-MM-DD`` strings.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import posixpath
import re
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

logger = logging.getLogger("plo5bp.ui.public")

# --- Config (env) -----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

BASE_URL = os.environ.get("PLO5BP_BASE_URL", "http://127.0.0.1:8770").rstrip("/")
DB_PATH = Path(os.environ.get("PLO5BP_DB", str(REPO_ROOT / "data" / "public.db")))
ADMIN_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("PLO5BP_ADMIN_EMAILS", "themilesgarcia@icloud.com").split(",")
    if e.strip()
}
FREE_HANDS_PER_DAY = int(os.environ.get("PLO5BP_FREE_HANDS", "5"))
# (2026-09-22) The models are still in development, so for now the whole site
# is FREE: every signed-in user is entitled — no daily hand quota, Study is
# unlocked, checkout is closed. Sign-in stays (per-user sessions, abuse
# control). PLO5BP_FREE_FOR_ALL=0 brings the paywall back unchanged; the
# billing code and its tests are all still here.
FREE_FOR_ALL = os.environ.get("PLO5BP_FREE_FOR_ALL", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
PRICE_CENTS = int(os.environ.get("PLO5BP_PRICE_CENTS", "1000"))  # $10/mo


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _is_loopback_host(host: str | None) -> bool:
    """True for localhost / 127.0.0.0/8 / ::1 (brackets and case ignored)."""
    h = (host or "").strip().strip("[]").lower()
    if not h:
        return False
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


# Dev login (review 2026-09-20 F4). The env flag alone is NOT enough: the
# fake sign-in lets anyone become any email (the admin's included), so the
# route only exists when the deployment's own BASE_URL is a loopback URL — a
# prod/tunnel config (https://wrapgto.com) can never mount it even if the
# flag leaks into its env file. Per-request checks are in `_dev_login_request_ok`.
DEV_LOGIN_REQUESTED = _env_flag("PLO5BP_DEV_LOGIN")
DEV_LOGIN = DEV_LOGIN_REQUESTED and _is_loopback_host(urlsplit(BASE_URL).hostname)
# Starlette's TestClient reports client host "testclient" / Host "testserver".
# Those are only accepted when a test fixture opts in explicitly.
DEV_LOGIN_TESTCLIENT = _env_flag("PLO5BP_DEV_LOGIN_TESTCLIENT")
# Any of these means the request crossed a proxy / tunnel — never loopback.
_FORWARDING_HEADERS = (
    "x-forwarded-for",
    "x-forwarded-host",
    "x-real-ip",
    "forwarded",
    "cf-connecting-ip",
    "cf-ray",
    "true-client-ip",
)

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")
# Stripe re-validation policy (review 2026-09-20 F2).
#: Per-request network budget for a Stripe call (the library default is 80 s).
STRIPE_TIMEOUT_S = float(os.environ.get("PLO5BP_STRIPE_TIMEOUT", "8"))
#: Fail-open cap: when Stripe cannot be reached, a cached "active" status is
#: honoured only until ``current_period_end`` + this grace.
STRIPE_GRACE = timedelta(days=float(os.environ.get("PLO5BP_STRIPE_GRACE_DAYS", "3")))
#: Minimum spacing of re-checks for one user once the period has ended (or
#: after a failed check) — never one Stripe call per request.
STRIPE_RETRY_S = float(os.environ.get("PLO5BP_STRIPE_RETRY_S", "900"))
#: A verified status is trusted this long while inside the paid period.
STRIPE_RECHECK = timedelta(hours=24)
#: Stripe subscription statuses that carry paid access.
_STRIPE_ACTIVE = ("active", "trialing", "past_due")

MAX_USER_RUNTIMES = int(os.environ.get("PLO5BP_MAX_RUNTIMES", "300"))
# A signed-in user counts as "active" for this many seconds after their last
# authenticated request (the admin top-bar deploy-safety counter).
ACTIVE_WINDOW_S = int(os.environ.get("PLO5BP_ACTIVE_WINDOW", "300"))

# Study-mode routes: subscription required (the full product). The trainer
# tree is the free-tier surface. `/format` switches the STUDY session's game
# (and 500'd on a free user's never-built env) — review 2026-09-20 F7.
STUDY_PATHS = {
    "/state", "/cards", "/action", "/seats", "/config", "/undo", "/reset",
    "/format",
}
# No auth at all:
OPEN_PREFIXES = ("/static/", "/auth/", "/stripe/webhook", "/health")
OPEN_EXACT = {
    "/", "/me", "/favicon.ico", "/terms", "/privacy",
    "/apple-touch-icon.png", "/apple-touch-icon-precomposed.png",
}
# Free-tier metering (see AccessMiddleware): the explicit deal route, plus
# the trainer routes that deal IMPLICITLY when the session has no live hand.
NEW_HAND_PATH = "/trainer/new_hand"
IMPLICIT_DEAL_PATHS = frozenset({
    "/trainer/state", "/trainer/settings", "/trainer/act", "/trainer/stats/reset",
})

_MULTI_SLASH = re.compile(r"/{2,}")


def _norm_path(raw: str | None) -> str:
    """Canonical form of a request path for AUTHORIZATION decisions.

    (review 2026-09-20 F1/F7) The gate used to compare the raw path against
    exact strings, while the layers below it are more forgiving: StaticFiles
    normalizes ``/static//games.js`` and ``/static/games.js/`` to the same
    file, and the router redirects ``/state/`` to ``/state``. Every check in
    the middleware therefore runs on this form: backslashes → slashes,
    repeated slashes collapsed, ``.``/``..`` segments resolved, trailing
    slash dropped (except for "/"). Routing itself is untouched."""
    p = _MULTI_SLASH.sub("/", (raw or "/").replace("\\", "/"))
    if not p.startswith("/"):
        p = "/" + p
    p = posixpath.normpath(p)
    # normpath keeps a leading "//" (POSIX quirk) — already collapsed above.
    return p or "/"

# --- DB ----------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY,
  google_sub TEXT UNIQUE,
  email TEXT UNIQUE NOT NULL,
  name TEXT DEFAULT '',
  picture TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  last_login_at TEXT,
  sub_status TEXT NOT NULL DEFAULT 'none',   -- none | active
  sub_source TEXT NOT NULL DEFAULT '',        -- '' | comp | stripe
  stripe_customer_id TEXT,
  stripe_subscription_id TEXT,
  current_period_end TEXT,                    -- iso, stripe subs only
  sub_checked_at TEXT,
  homegame_access INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage (
  user_id INTEGER NOT NULL,
  day TEXT NOT NULL,
  hands INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (user_id, day)
);
CREATE TABLE IF NOT EXISTS payments (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  stripe_ref TEXT UNIQUE NOT NULL,
  amount_cents INTEGER NOT NULL,
  currency TEXT NOT NULL DEFAULT 'usd',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


class Db:
    """Small thread-safe sqlite wrapper (single connection + lock).

    Every ``q()`` commits on its own unless it runs inside ``transaction()``,
    which makes a group of statements all-or-nothing (review 2026-09-20 G4:
    a home-game mutation must never be half-persisted)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Re-entrant: q() is called from inside transaction() on one thread.
        self._lock = threading.RLock()
        self._tx_depth = 0
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            cols = {
                r[1]
                for r in self._conn.execute("PRAGMA table_info(users)").fetchall()
            }
            if "homegame_access" not in cols:
                self._conn.execute(
                    "ALTER TABLE users ADD COLUMN homegame_access "
                    "INTEGER NOT NULL DEFAULT 0"
                )
            self._conn.commit()

    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            try:
                cur = self._conn.execute(sql, args)
                rows = cur.fetchall()
            except BaseException:
                if self._tx_depth == 0:
                    self._conn.rollback()
                raise
            if self._tx_depth == 0:
                self._conn.commit()
            return rows

    @contextmanager
    def transaction(self):
        """All-or-nothing group of ``q()`` calls (nestable; the outermost
        level commits or rolls back). Holds the connection lock throughout,
        so keep the body short and free of network I/O."""
        with self._lock:
            self._tx_depth += 1
            try:
                yield self
            except BaseException:
                self._tx_depth -= 1
                if self._tx_depth == 0:
                    self._conn.rollback()
                raise
            else:
                self._tx_depth -= 1
                if self._tx_depth == 0:
                    self._conn.commit()

    def one(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def kv_get(self, key: str) -> str | None:
        r = self.one("SELECT value FROM kv WHERE key=?", (key,))
        return r["value"] if r else None

    def kv_set(self, key: str, value: str) -> None:
        self.q(
            "INSERT INTO kv(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


DB = Db(DB_PATH)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _session_secret() -> str:
    s = DB.kv_get("session_secret")
    if not s:
        s = secrets.token_urlsafe(48)
        DB.kv_set("session_secret", s)
    return s


# --- Users -------------------------------------------------------------------


def _upsert_user(google_sub: str | None, email: str, name: str, picture: str) -> sqlite3.Row:
    email = email.strip().lower()
    row = DB.one("SELECT * FROM users WHERE email=?", (email,))
    if row is None:
        DB.q(
            "INSERT INTO users(google_sub,email,name,picture,created_at,last_login_at)"
            " VALUES(?,?,?,?,?,?)",
            (google_sub, email, name, picture, _now(), _now()),
        )
    else:
        DB.q(
            "UPDATE users SET google_sub=COALESCE(?,google_sub), name=?, picture=?,"
            " last_login_at=? WHERE id=?",
            (google_sub, name or row["name"], picture or row["picture"], _now(), row["id"]),
        )
    return DB.one("SELECT * FROM users WHERE email=?", (email,))


def _user_by_id(uid: int) -> sqlite3.Row | None:
    return DB.one("SELECT * FROM users WHERE id=?", (uid,))


def _is_admin(user: sqlite3.Row | None) -> bool:
    return bool(user) and user["email"].lower() in ADMIN_EMAILS


def _homegame_access(user: sqlite3.Row | None) -> bool:
    """The home-games pages. Independent of subscription.

    Since clubs (2026-09-25) every signed-in user has them: what a user may SEE
    there — a club's tables, members and numbers — is the club's business
    (homegame.py). Signed out is denial (a sign-in page / the hidden 404).
    """
    return user is not None


# homegame.install: /admin's home-games switch now means "a member of the MAIN
# club" (the site's original circle). (admin_uid, uid, grant) -> member after;
# (uid) -> member?
_GAMES_ACCESS_HOOK: Any = None
_GAMES_MEMBER_HOOK: Any = None


def _games_path(path: str) -> bool:
    """`path` must already be normalized (`_norm_path`)."""
    return path == "/games" or path.startswith("/games/")


# Served from the public StaticFiles mount, so they would otherwise be
# world-readable via the /static/ open prefix and advertise the page.
_GAMES_ASSETS = frozenset({
    "/static/games.js",
    "/static/games.css",
    "/static/games.html",
})


def _games_asset(path: str) -> bool:
    """Normalized path names one of the hidden home-games static files.

    Case-folded: a case-insensitive filesystem (Windows/macOS dev boxes)
    serves ``/static/GAMES.JS`` as the same file."""
    return path.lower() in _GAMES_ASSETS


# A table link is itself the secret. For someone WITHOUT home-games access,
# homegame.install sets this to a function (request, path, user) -> Response | None
# that answers a VALID table link with an invite page (sign in / ask to join /
# waiting) instead of the hidden 404 — every other /games path stays a 404.
_GAMES_INVITE_HOOK: Any = None

# Where a sign-in may send the browser afterwards: the home-games lobby, a table
# link or a club invite link only (an open redirect would let a phishing page
# borrow this site's name).
_NEXT_RE = re.compile(r"^/games(?:/t/[A-Za-z0-9_-]{1,40}|/join/[A-Za-z0-9_-]{6,40})?$")


def _safe_next(raw: Any) -> str | None:
    return raw if isinstance(raw, str) and _NEXT_RE.match(raw) else None


def _hidden_not_found(request: Request) -> HTMLResponse | JSONResponse:
    """404 with no feature name — the home-games surface must not advertise
    itself to anyone without access, including a 401/403 distinction."""
    accept = request.headers.get("accept", "")
    headers = {"Cache-Control": "no-store, must-revalidate"}
    if "text/html" in accept:
        return HTMLResponse(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Not Found</title></head><body><h1>Not Found</h1></body></html>",
            status_code=404,
            headers=headers,
        )
    return JSONResponse({"detail": "Not Found"}, status_code=404, headers=headers)


_STRIPE_CLIENT_READY = False


def _verified_email(info: Any) -> str | None:
    """The sign-in identity from OIDC userinfo: the lowercased email, and only
    when the IdP asserts it is verified.

    (review 2026-09-20 F7) The check used to be ``info.get("email_verified",
    True)`` — an ABSENT claim counted as verified. The email is the account
    key here (it carries the subscription and the admin allowlist), so the
    default is closed. Google sends a JSON boolean; the string form some IdPs
    use is tolerated."""
    if not isinstance(info, dict):
        return None
    email = str(info.get("email") or "").strip().lower()
    verified = info.get("email_verified", False)
    if isinstance(verified, str):
        verified = verified.strip().lower() == "true"
    if not email or verified is not True:
        return None
    return email


def _dev_login_request_ok(request: Request) -> bool:
    """Request-level loopback proof for the dev login (review 2026-09-20 F4).

    ``request.client`` alone depends on proxy config: a tunnel that sends no
    X-Forwarded-For, ``--forwarded-allow-ips=*``, or a proxy connecting over
    ``::1`` all make a remote visitor look like 127.0.0.1. So ALSO require
    that nothing about the request says "proxied": no forwarding headers, and
    a loopback Host header (a tunnel forwards the public hostname)."""
    if not DEV_LOGIN:
        return False
    headers = request.headers
    if any(h in headers for h in _FORWARDING_HEADERS):
        return False
    client = request.client.host if request.client else ""
    try:
        host = urlsplit("//" + headers.get("host", "")).hostname
    except ValueError:
        return False
    if DEV_LOGIN_TESTCLIENT and client == "testclient":
        # Starlette's TestClient; only when a test fixture opted in.
        return host == "testserver" or _is_loopback_host(host)
    return _is_loopback_host(client) and _is_loopback_host(host)


def _stripe():
    """The configured stripe module. Blocking network I/O — every caller must
    be off the event loop (sync endpoint, or ``run_in_threadpool``)."""
    global _STRIPE_CLIENT_READY
    import stripe as _s

    _s.api_key = STRIPE_SECRET_KEY
    if not _STRIPE_CLIENT_READY:
        # (review 2026-09-20 F2) The library default is an 80 s timeout with
        # retries: one slow Stripe could pin a worker thread per request.
        try:
            _s.default_http_client = _s.new_default_http_client(
                timeout=STRIPE_TIMEOUT_S
            )
            _s.max_network_retries = 1
        except Exception as e:  # noqa: BLE001 — e.g. a client without `timeout`
            logger.warning("could not set stripe timeout: %s", e)
        _STRIPE_CLIENT_READY = True
    return _s


def body_int(
    body: Any, key: str, default: int | None = None, *, limit: int = 2**53
) -> int:
    """Integer field of a JSON body, or HTTP 400 — never a 500.

    (review 2026-09-20 F7/G-minor) ``int(body.get(...))`` raised ValueError /
    TypeError / OverflowError on ``"abc"``, ``None``, ``[1]``, ``1e999``.
    Accepts ints and integral floats/strings; rejects bools, NaN/inf and
    anything past ±``limit``. ``default`` (when given) covers a missing/null
    field. Shared with homegame.py."""
    v = body.get(key) if isinstance(body, dict) else None
    if v is None:
        if default is None:
            raise HTTPException(status_code=400, detail=f"missing {key}")
        return int(default)
    try:
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise ValueError(key)
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")) or f != int(f):
            raise ValueError(key)
        n = int(v) if isinstance(v, int) else int(f)
    except (TypeError, ValueError, OverflowError) as e:
        raise HTTPException(status_code=400, detail=f"invalid {key}") from e
    if abs(n) > limit:
        raise HTTPException(status_code=400, detail=f"invalid {key}")
    return n


def _sv(obj: Any, key: str, default: Any = None) -> Any:
    """Safe field access for Stripe objects AND plain dicts (webhooks).

    stripe-python v15 StripeObjects support ``obj[key]`` / ``obj.key`` but no
    longer inherit dict — ``.get()`` raises AttributeError. This is the one
    accessor used for every Stripe payload field below."""
    try:
        return obj[key]
    except (KeyError, IndexError, TypeError):
        return default


def _parse_iso(value: Any) -> datetime | None:
    """Stored UTC iso timestamp → aware datetime (None when absent/garbage)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _sub_period_end(sub: Any) -> str | None:
    """`current_period_end` of a Stripe subscription as a stored iso string.
    Newer API versions carry it on the first item, older ones on the sub."""
    try:
        items = _sv(_sv(sub, "items", {}), "data", []) or []
        end_ts = _sv(items[0], "current_period_end") if items else None
        if not end_ts:
            end_ts = _sv(sub, "current_period_end")
        if end_ts:
            return datetime.fromtimestamp(int(end_ts), tz=timezone.utc).isoformat(
                timespec="seconds"
            )
    except Exception:  # noqa: BLE001 — a missing period end is survivable
        pass
    return None


def _stripe_missing(exc: BaseException) -> bool:
    """Stripe's definitive "No such subscription" (``resource_missing`` /
    ``InvalidRequestError``): the sub was deleted, or the key was switched
    test→live. That is an answer — INACTIVE — not an outage."""
    if getattr(exc, "code", None) == "resource_missing":
        return True
    if type(exc).__name__ == "InvalidRequestError":
        return True
    return "no such subscription" in str(exc).lower()


# uid -> monotonic time before which Stripe is not asked again for that user.
# In-memory on purpose: it is a rate limit, and a restart may re-ask once.
_STRIPE_RETRY_AT: dict[int, float] = {}
_STRIPE_RETRY_LOCK = threading.Lock()


def _stripe_managed(user: sqlite3.Row) -> bool:
    return bool(STRIPE_SECRET_KEY and user["stripe_subscription_id"])


def _stripe_check_due(user: sqlite3.Row, now: datetime | None = None) -> bool:
    """Is the cached stripe status stale enough to re-verify? Cheap + pure.

    ``sub_checked_at`` is the last time Stripe gave a DEFINITIVE answer.
    Inside the paid period that answer is trusted for 24 h; once the period
    has ended it is re-checked, but at most every STRIPE_RETRY_S — not on
    every request (a verified ``past_due`` sub keeps an old period end)."""
    now = now or datetime.now(timezone.utc)
    checked = _parse_iso(user["sub_checked_at"])
    if checked is None:
        return True
    age = now - checked
    if age > STRIPE_RECHECK:
        return True
    raw_end = user["current_period_end"]
    end = _parse_iso(raw_end)
    past_end = (end is not None and now > end) or (bool(raw_end) and end is None)
    return past_end and age > timedelta(seconds=STRIPE_RETRY_S)


def _within_stripe_grace(user: sqlite3.Row, now: datetime | None = None) -> bool:
    """Fail-open cap for an UNVERIFIED cached 'active': honoured only until
    the paid period's end + STRIPE_GRACE. Without a period end, the anchor is
    the moment the last verification went stale."""
    now = now or datetime.now(timezone.utc)
    end = _parse_iso(user["current_period_end"])
    if end is None:
        checked = _parse_iso(user["sub_checked_at"])
        if checked is None:
            return False
        end = checked + STRIPE_RECHECK
    return now < end + STRIPE_GRACE


def _refresh_stripe_status(user: sqlite3.Row) -> sqlite3.Row | None:
    """Re-verify a stripe-sourced sub against Stripe (BLOCKING network call —
    never on the event loop; see ``AccessMiddleware``).

    Returns the re-read user row when Stripe gave a definitive answer
    (including "no such subscription" → inactive), or ``None`` when the
    status stays unverified: Stripe errored/timed out, or this user is
    inside the retry back-off / another request is already asking.

    (review 2026-09-20 F2) The old version ran on the event loop with an
    80 s timeout, kept access forever on ANY exception, and re-called Stripe
    on every request once the period had ended."""
    uid = int(user["id"])
    now_m = time.monotonic()
    with _STRIPE_RETRY_LOCK:
        if _STRIPE_RETRY_AT.get(uid, 0.0) > now_m:
            return None
        # Claim the slot before the call: single-flight per user, and a
        # failure below leaves the back-off in place.
        _STRIPE_RETRY_AT[uid] = now_m + STRIPE_RETRY_S
    try:
        sub = _stripe().Subscription.retrieve(user["stripe_subscription_id"])
        active = _sv(sub, "status") in _STRIPE_ACTIVE
        period_end = _sub_period_end(sub)
    except Exception as e:  # noqa: BLE001
        if not _stripe_missing(e):
            logger.warning(
                "stripe status refresh failed for user %s (unverified, retry in"
                " %.0fs): %s", uid, STRIPE_RETRY_S, e,
            )
            return None
        logger.warning("stripe subscription gone for user %s: %s", uid, e)
        active, period_end = False, user["current_period_end"]
    DB.q(
        "UPDATE users SET sub_status=?, sub_source='stripe', current_period_end=?,"
        " sub_checked_at=? WHERE id=?",
        ("active" if active else "none", period_end, _now(), uid),
    )
    with _STRIPE_RETRY_LOCK:
        _STRIPE_RETRY_AT.pop(uid, None)
    return _user_by_id(uid)


def _entitlement_needs_stripe(user: sqlite3.Row | None) -> bool:
    """True when ``_entitled(user)`` would (try to) call Stripe. Lets the
    async middleware keep the common path inline and move only the network
    call to a worker thread."""
    return (
        user is not None
        and not _is_admin(user)
        and user["sub_status"] == "active"
        and user["sub_source"] != "comp"
        and _stripe_managed(user)
        and _stripe_check_due(user)
    )


def _entitled(user: sqlite3.Row | None) -> bool:
    """Full access: admin, comp grant, or active stripe subscription.

    May block on Stripe (see ``_entitlement_needs_stripe``)."""
    if user is None:
        return False
    if FREE_FOR_ALL or _is_admin(user):
        return True
    if user["sub_status"] != "active":
        return False
    if user["sub_source"] == "comp":
        return True
    if (
        user["sub_source"] == "stripe"
        and STRIPE_SECRET_KEY
        and not user["stripe_subscription_id"]
    ):
        # (review 2026-09-20 F3) "stripe" access with no subscription to
        # verify against — only a non-subscription checkout replay made these.
        return False
    if not _stripe_managed(user) or not _stripe_check_due(user):
        return True
    fresh = _refresh_stripe_status(user)
    if fresh is not None:
        return fresh["sub_status"] == "active"
    # Unverified (Stripe down / backing off): cached status, capped.
    return _within_stripe_grace(user)


# --- Usage (free tier) --------------------------------------------------------


def _hands_today(uid: int) -> int:
    r = DB.one("SELECT hands FROM usage WHERE user_id=? AND day=?", (uid, _today()))
    return int(r["hands"]) if r else 0


def _record_hand(uid: int) -> int:
    DB.q(
        "INSERT INTO usage(user_id,day,hands) VALUES(?,?,1)"
        " ON CONFLICT(user_id,day) DO UPDATE SET hands = hands + 1",
        (uid, _today()),
    )
    return _hands_today(uid)


def _refund_hand(uid: int) -> None:
    DB.q(
        "UPDATE usage SET hands = MAX(0, hands - 1) WHERE user_id=? AND day=?",
        (uid, _today()),
    )


def _resets_at() -> str:
    tomorrow = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(days=1)
    return tomorrow.isoformat(timespec="seconds")


# --- Per-user runtimes ---------------------------------------------------------

_CURRENT_USER_ID: ContextVar[int | None] = ContextVar("plo5bp_user_id", default=None)


class _Runtime:
    __slots__ = ("study", "trainer", "last_seen")

    def __init__(self, study: Any, trainer: Any):
        self.study = study
        self.trainer = trainer
        self.last_seen = time.time()


class Registry:
    """user_id -> per-user (study Session, TrainerSession), LRU-capped.

    Eviction is safe: study rebuilds from user input; trainer lifetime stats
    persist per-user (each TrainerSession writes its own stats file), only an
    in-flight hand is lost — same as a browser refresh after eviction."""

    def __init__(self, study_factory: Callable[[], Any], trainer_factory: Callable[[int], Any]):
        self._study_factory = study_factory
        self._trainer_factory = trainer_factory
        self._map: OrderedDict[int, _Runtime] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, uid: int) -> _Runtime:
        with self._lock:
            rt = self._map.get(uid)
            if rt is None:
                rt = _Runtime(self._study_factory(), self._trainer_factory(uid))
                self._map[uid] = rt
            rt.last_seen = time.time()
            self._map.move_to_end(uid)
            while len(self._map) > MAX_USER_RUNTIMES:
                evicted_uid, evicted = self._map.popitem(last=False)
                logger.info("evicted runtime for user %s (LRU cap)", evicted_uid)
            return rt

    def peek(self, uid: int) -> _Runtime | None:
        """The user's runtime if one exists. Never creates or LRU-touches."""
        with self._lock:
            return self._map.get(uid)


_REGISTRY: Registry | None = None


def _current_runtime() -> _Runtime | None:
    uid = _CURRENT_USER_ID.get()
    if uid is None or _REGISTRY is None:
        return None
    return _REGISTRY.get(uid)


# --- Active-user tracking --------------------------------------------------------

# uid -> last authenticated-request time. In-memory on purpose: it feeds the
# admin "safe to deploy?" counter, and a restart (the event it protects
# against) legitimately resets it.
_ACTIVITY: dict[int, float] = {}
_ACTIVITY_LOCK = threading.Lock()


def _record_activity(uid: int) -> None:
    with _ACTIVITY_LOCK:
        _ACTIVITY[uid] = time.time()


def _active_uids() -> list[int]:
    """Users seen within ACTIVE_WINDOW_S; prunes expired entries."""
    cutoff = time.time() - ACTIVE_WINDOW_S
    with _ACTIVITY_LOCK:
        for uid in [u for u, t in _ACTIVITY.items() if t < cutoff]:
            del _ACTIVITY[uid]
        return list(_ACTIVITY)


# --- Middleware ----------------------------------------------------------------


def _wants_sub_route(path: str) -> bool:
    return path in STUDY_PATHS


def _open_route(path: str) -> bool:
    return path in OPEN_EXACT or path.startswith(OPEN_PREFIXES)


def _implicit_deal_session(uid: int) -> Any | None:
    """The user's TrainerSession when the NEXT trainer request would deal a
    hand implicitly *and that deal must be metered*; else None.

    trainer.py's ``_ensure_hand`` deals whenever ``ts.hand is None``. The very
    first implicit hand of a fresh session (``hand_no == 0``: first load, or a
    runtime rebuilt after a restart / LRU eviction) is documented as free.
    But a session that has ALREADY dealt (``hand_no > 0``) and lost its hand —
    today only via a format switch — would otherwise turn
    ``POST /format`` + ``GET /trainer/state`` into an unmetered free-hand
    loop (review 2026-09-20 F6)."""
    rt = _REGISTRY.peek(uid) if _REGISTRY is not None else None
    ts = getattr(rt, "trainer", None)
    if ts is None:
        return None
    try:
        if ts.hand is None and int(ts.hand_no) > 0:
            return ts
    except (AttributeError, TypeError, ValueError):
        return None
    return None


class AccessMiddleware(BaseHTTPMiddleware):
    """Auth + entitlement + free-quota enforcement for the public build."""

    async def dispatch(self, request: Request, call_next):
        # Authorize on the NORMALIZED ASGI path (review 2026-09-20 F1/F7):
        # `scope["path"]` is what the router matches (request.url is rebuilt
        # from the Host header), and `_norm_path` closes the `//` and
        # trailing-slash spellings the layers below treat as equivalent.
        path = _norm_path(request.scope.get("path"))
        uid = request.session.get("uid")
        user = _user_by_id(int(uid)) if uid is not None else None

        # Home games: signed out = the sign-in page for a browser, the hidden
        # 404 for everything else (clubs, 2026-09-25: every signed-in user has
        # them). Do this BEFORE the /static open-prefix short-circuit and
        # BEFORE the generic 401 so a signed-out probe of /games or
        # /static/games.js looks like a missing page.
        if _games_path(path) or _games_asset(path):
            if not _homegame_access(user):
                hook = _GAMES_INVITE_HOOK
                invite = hook(request, path, user) if hook is not None else None
                return invite if invite is not None else _hidden_not_found(request)

        if _open_route(path) and not _games_path(path):
            return await call_next(request)

        if user is None:
            return JSONResponse(
                {"detail": "auth required", "error": "auth"}, status_code=401
            )

        # (review 2026-09-20 F2) Entitlement is inline and cheap EXCEPT when a
        # Stripe re-validation is due — that blocking call must never run on
        # the event loop, where it froze every user for up to 80 s.
        if _entitlement_needs_stripe(user):
            entitled = await run_in_threadpool(_entitled, user)
        else:
            entitled = _entitled(user)
        admin = _is_admin(user)
        user_id = int(user["id"])
        _record_activity(user_id)

        if path == "/admin" or path.startswith("/admin/"):
            if not admin:
                return JSONResponse({"detail": "admin only"}, status_code=403)

        if _wants_sub_route(path) and not entitled:
            return JSONResponse(
                {
                    "detail": "Study mode requires a subscription.",
                    "error": "subscription_required",
                },
                status_code=402,
            )

        # Free-tier metering. The explicit deal counts for everyone (admin
        # metrics); an implicit deal counts for non-entitled users only.
        consumed = False
        implicit_ts = None
        implicit_hand_no = 0
        if path == NEW_HAND_PATH and request.method == "POST":
            consumed = True
        elif not entitled and path in IMPLICIT_DEAL_PATHS:
            implicit_ts = _implicit_deal_session(user_id)
            if implicit_ts is not None:
                implicit_hand_no = int(implicit_ts.hand_no)
                consumed = True
        if consumed:
            used = _record_hand(user_id)
            if not entitled and used > FREE_HANDS_PER_DAY:
                _refund_hand(user_id)
                return JSONResponse(
                    {
                        "detail": (
                            f"Free limit reached: {FREE_HANDS_PER_DAY} trainer hands"
                            " per day. Subscribe for unlimited."
                        ),
                        "error": "free_limit",
                        "used": used - 1,
                        "limit": FREE_HANDS_PER_DAY,
                        "resets_at": _resets_at(),
                    },
                    status_code=402,
                )

        token = _CURRENT_USER_ID.set(user_id)
        try:
            response = await call_next(request)
        finally:
            _CURRENT_USER_ID.reset(token)

        if consumed:
            # No hand was dealt: an error, a redirect (the router's
            # trailing-slash 307 — the follow-up request is metered), or an
            # implicit-deal candidate that ended up not dealing (e.g. 409).
            no_deal = response.status_code >= 300 or (
                implicit_ts is not None
                and int(implicit_ts.hand_no) == implicit_hand_no
            )
            if no_deal:
                _refund_hand(user_id)
            elif not entitled:
                left = max(0, FREE_HANDS_PER_DAY - _hands_today(user_id))
                response.headers["X-Free-Hands-Left"] = str(left)
        return response


# --- Install -------------------------------------------------------------------


def install(
    app: FastAPI,
    *,
    study_session_factory: Callable[[], Any],
    set_study_resolver: Callable[[Callable[[], Any] | None], None],
    trainer_session_factory: Callable[[Path], Any],
    set_trainer_resolver: Callable[[Callable[[], Any] | None], None],
    static_dir: Path,
    set_format_gate: Callable[[Any], None] | None = None,
) -> None:
    """Wire auth/billing/admin into the public app. Called from server.py."""
    global _REGISTRY

    stats_dir = DB_PATH.parent / "trainer_stats"
    _REGISTRY = Registry(
        study_session_factory,
        lambda uid: trainer_session_factory(stats_dir / f"u{uid}.json"),
    )

    def _study_for_request():
        rt = _current_runtime()
        return rt.study if rt else None

    def _trainer_for_request():
        rt = _current_runtime()
        return rt.trainer if rt else None

    set_study_resolver(_study_for_request)
    set_trainer_resolver(_trainer_for_request)

    if set_format_gate is not None:
        def _format_gate(fmt_id: str) -> bool:
            """Non-default formats are admin-only on the public site
            while their models train — everyone else sees the dropdown
            entry greyed out as "coming soon!". PLO5 (the launched
            product) is never gated."""
            if fmt_id == "plo5_double_bomb":
                return False
            uid = _CURRENT_USER_ID.get()
            user = _user_by_id(int(uid)) if uid is not None else None
            return not _is_admin(user)

        set_format_gate(_format_gate)

    # OAuth (Authlib). Configured lazily so the app boots without credentials
    # (dev login covers local testing until Google creds exist).
    oauth = None
    if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
        from authlib.integrations.starlette_client import OAuth

        oauth = OAuth()
        oauth.register(
            name="google",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )

    # --- Auth routes ---------------------------------------------------------

    @app.get("/auth/login")
    async def auth_login(request: Request):
        if oauth is None:
            return JSONResponse(
                {
                    "detail": (
                        "Google OAuth is not configured. Set GOOGLE_CLIENT_ID /"
                        " GOOGLE_CLIENT_SECRET (see PUBLIC_SETUP.md)"
                        + (
                            " — or use the dev login."
                            if _dev_login_request_ok(request)
                            else "."
                        )
                    )
                },
                status_code=503,
            )
        redirect_uri = f"{BASE_URL}/auth/callback"
        nxt = _safe_next(request.query_params.get("next"))
        if nxt:
            request.session["next"] = nxt  # back to the table link after sign-in
        else:
            request.session.pop("next", None)
        return await oauth.google.authorize_redirect(request, redirect_uri)

    @app.get("/auth/callback")
    async def auth_callback(request: Request):
        if oauth is None:
            raise HTTPException(status_code=503, detail="OAuth not configured")
        try:
            token = await oauth.google.authorize_access_token(request)
        except Exception as e:  # noqa: BLE001 — OAuth dance failures → login page
            logger.warning("oauth callback failed: %s", e)
            return RedirectResponse(url="/?login=failed")
        info = token.get("userinfo") or {}
        email = _verified_email(info)
        if not email:
            return RedirectResponse(url="/?login=failed")
        user = _upsert_user(
            info.get("sub"), email, info.get("name") or "", info.get("picture") or ""
        )
        request.session["uid"] = int(user["id"])
        return RedirectResponse(url=_safe_next(request.session.pop("next", None)) or "/")

    if DEV_LOGIN_REQUESTED and not DEV_LOGIN:
        logger.warning(
            "PLO5BP_DEV_LOGIN is set but PLO5BP_BASE_URL (%s) is not a loopback"
            " URL — dev login stays DISABLED.", BASE_URL,
        )

    if DEV_LOGIN:

        @app.get("/auth/dev")
        def auth_dev(request: Request, email: str, name: str = "", next: str = ""):  # noqa: A002
            """Loopback-only fake sign-in for local testing without OAuth.

            NEVER expose a tunnel with PLO5BP_DEV_LOGIN=1 — anyone could sign
            in as any email, including the admin's. Defence in depth: the
            route is not even registered unless BASE_URL is loopback, and
            `_dev_login_request_ok` rejects anything proxied."""
            if not _dev_login_request_ok(request):
                raise HTTPException(status_code=403, detail="dev login is loopback-only")
            user = _upsert_user(None, email, name or email.split("@")[0], "")
            request.session["uid"] = int(user["id"])
            return RedirectResponse(url=_safe_next(next) or "/")

    # (review 2026-09-20 F7) Logout is state-changing, so POST is the real
    # verb. GET stays because app.js still navigates to it
    # (`window.location.href = "/auth/logout"`); drop it once that moves.
    @app.api_route("/auth/logout", methods=["GET", "POST"])
    def auth_logout(request: Request):
        request.session.clear()
        # 303: a POST must be followed by a GET of "/".
        return RedirectResponse(url="/", status_code=303)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/me")
    def me(request: Request):
        uid = request.session.get("uid")
        user = _user_by_id(int(uid)) if uid is not None else None
        if user is None:
            return {
                "signed_in": False,
                "auth_configured": oauth is not None,
                # (review 2026-09-20 F4) Only a request that could actually
                # USE the dev login learns that it exists.
                "dev_login": _dev_login_request_ok(request),
            }
        entitled = _entitled(user)
        user = _user_by_id(user["id"])  # re-read (refresh may have written)
        used = _hands_today(user["id"])
        payload: dict[str, Any] = {
            "signed_in": True,
            "email": user["email"],
            "name": user["name"],
            "picture": user["picture"],
            "is_admin": _is_admin(user),
            "sub": {
                "active": entitled,
                "source": user["sub_source"] if user["sub_status"] == "active" else (
                    "admin" if _is_admin(user) else ""
                ),
                "period_end": user["current_period_end"],
            },
            "free": {
                "used": used,
                "limit": FREE_HANDS_PER_DAY,
                "left": max(0, FREE_HANDS_PER_DAY - used),
                "resets_at": _resets_at(),
            },
            "billing_configured": bool(STRIPE_SECRET_KEY),
            "price_cents": PRICE_CENTS,
            # Everyone has full access while the models are in development.
            "free_for_all": FREE_FOR_ALL,
        }
        # Only present when granted so a /me dump from a normal subscriber
        # does not advertise that a private games page exists. The href +
        # label ride along so the frontend can build the tab WITHOUT shipping
        # those literals to every visitor (review 2026-09-20 F1).
        if _homegame_access(user):
            payload["homegame"] = {"href": "/games", "label": "Home games"}
        return payload

    # --- Billing (Stripe) -----------------------------------------------------

    def _price_id() -> str:
        if STRIPE_PRICE_ID:
            return STRIPE_PRICE_ID
        cached = DB.kv_get("stripe_price_id")
        if cached:
            return cached
        s = _stripe()
        product = s.Product.create(name="WrapGTO — Monthly")
        price = s.Price.create(
            product=product["id"],
            unit_amount=PRICE_CENTS,
            currency="usd",
            recurring={"interval": "month"},
        )
        DB.kv_set("stripe_price_id", price["id"])
        return price["id"]

    def _require_user(request: Request) -> sqlite3.Row:
        uid = request.session.get("uid")
        user = _user_by_id(int(uid)) if uid is not None else None
        if user is None:
            raise HTTPException(status_code=401, detail="auth required")
        return user

    @app.post("/billing/checkout")
    def billing_checkout(request: Request):
        if FREE_FOR_ALL:
            # Nobody should be able to start paying for something that is free.
            raise HTTPException(
                status_code=409,
                detail="WrapGTO is free while the models are in development — there is nothing to buy right now.",
            )
        user = _require_user(request)
        if not STRIPE_SECRET_KEY:
            raise HTTPException(
                status_code=503,
                detail="Billing is not configured yet (STRIPE_SECRET_KEY unset).",
            )
        s = _stripe()
        kwargs: dict[str, Any] = {
            "mode": "subscription",
            "line_items": [{"price": _price_id(), "quantity": 1}],
            "success_url": f"{BASE_URL}/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            "cancel_url": f"{BASE_URL}/?checkout=cancel",
            "client_reference_id": str(user["id"]),
            "allow_promotion_codes": True,
        }
        if user["stripe_customer_id"]:
            kwargs["customer"] = user["stripe_customer_id"]
        else:
            kwargs["customer_email"] = user["email"]
        sess = s.checkout.Session.create(**kwargs)
        return {"url": sess["url"]}

    def _obj_id(v: Any) -> str | None:
        """Stripe reference field: a bare id, or an expanded object."""
        if isinstance(v, str):
            return v or None
        ident = _sv(v or {}, "id")
        return str(ident) if ident else None

    def _activate_from_checkout(sess: Any) -> tuple[bool, str]:
        """Activate the checkout's user IFF it bought a LIVE subscription.

        (review 2026-09-20 F3) A Checkout Session is a receipt that stays
        "paid" forever; replaying your own old ``session_id`` used to
        re-activate a canceled/refunded subscription, and a ``mode=payment``
        session granted access with no subscription to ever re-verify. So
        the session must be a subscription checkout, settled, AND its
        subscription must be active/trialing at Stripe *right now*. Shared
        by /billing/confirm and the webhook. Returns (activated, status)."""
        uid = int(_sv(sess, "client_reference_id") or 0)
        user = _user_by_id(uid)
        if user is None:
            logger.warning(
                "checkout for unknown user ref %r", _sv(sess, "client_reference_id")
            )
            return False, "unknown_user"
        sub_id = _obj_id(_sv(sess, "subscription"))
        if _sv(sess, "mode") != "subscription" or not sub_id:
            return False, "not_a_subscription"
        pay = _sv(sess, "payment_status")
        # "no_payment_required" = 100% promo code / free trial checkout.
        if pay not in ("paid", "no_payment_required"):
            return False, str(pay or "unpaid")
        sub = _stripe().Subscription.retrieve(sub_id)
        sub_status = _sv(sub, "status")
        if sub_status not in ("active", "trialing"):
            return False, f"subscription_{sub_status}"
        DB.q(
            "UPDATE users SET sub_status='active', sub_source='stripe',"
            " stripe_customer_id=?, stripe_subscription_id=?,"
            " current_period_end=?, sub_checked_at=? WHERE id=?",
            (_obj_id(_sv(sess, "customer")), sub_id, _sub_period_end(sub), _now(), uid),
        )
        with _STRIPE_RETRY_LOCK:
            _STRIPE_RETRY_AT.pop(uid, None)
        ref = _sv(sess, "payment_intent") or _sv(sess, "invoice") or _sv(sess, "id")
        amount = int(_sv(sess, "amount_total") or 0)
        if ref and amount:
            try:
                DB.q(
                    "INSERT OR IGNORE INTO payments(user_id,stripe_ref,amount_cents,"
                    "currency,created_at) VALUES(?,?,?,?,?)",
                    (uid, str(_obj_id(ref) or ref), amount,
                     _sv(sess, "currency") or "usd", _now()),
                )
            except Exception:  # noqa: BLE001
                logger.exception("payment record failed")
        return True, "active"

    @app.get("/billing/confirm")
    def billing_confirm(request: Request, session_id: str):
        """Success-redirect verification — the no-webhook activation path."""
        user = _require_user(request)
        if not STRIPE_SECRET_KEY:
            raise HTTPException(status_code=503, detail="billing not configured")
        try:
            sess = _stripe().checkout.Session.retrieve(session_id)
        except Exception as e:  # noqa: BLE001
            if _stripe_missing(e):
                raise HTTPException(status_code=404, detail="no such checkout session")
            logger.warning("checkout session lookup failed: %s", e)
            raise HTTPException(
                status_code=502, detail="could not reach Stripe — try again shortly"
            )
        if str(_sv(sess, "client_reference_id")) != str(user["id"]):
            raise HTTPException(status_code=403, detail="session belongs to another user")
        try:
            active, status = _activate_from_checkout(sess)
        except Exception as e:  # noqa: BLE001 — subscription lookup failed
            logger.warning("subscription verification failed: %s", e)
            raise HTTPException(
                status_code=502, detail="could not reach Stripe — try again shortly"
            )
        return {"active": True} if active else {"active": False, "status": status}

    @app.post("/billing/portal")
    def billing_portal(request: Request):
        user = _require_user(request)
        if not (STRIPE_SECRET_KEY and user["stripe_customer_id"]):
            raise HTTPException(status_code=400, detail="no Stripe customer on file")
        sess = _stripe().billing_portal.Session.create(
            customer=user["stripe_customer_id"], return_url=BASE_URL + "/"
        )
        return {"url": sess["url"]}

    @app.post("/stripe/webhook")
    async def stripe_webhook(request: Request):
        """Optional: full lifecycle sync (renewals, cancellations). Laptop
        testing can run `stripe listen --forward-to <base>/stripe/webhook`;
        without it, lazy revalidation covers correctness."""
        if not (STRIPE_SECRET_KEY and STRIPE_WEBHOOK_SECRET):
            raise HTTPException(status_code=503, detail="webhook not configured")
        payload = await request.body()
        sig = request.headers.get("stripe-signature", "")
        s = _stripe()
        try:
            event = s.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
        except Exception as e:  # noqa: BLE001 — bad signature
            raise HTTPException(status_code=400, detail=f"invalid webhook: {e}")
        # The handler talks to Stripe (subscription verification) and sqlite:
        # blocking work, so off the event loop (review 2026-09-20 F2/F3). An
        # exception → 500 → Stripe redelivers the event later.
        await run_in_threadpool(_handle_stripe_event, event)
        return {"received": True}

    def _handle_stripe_event(event: Any) -> None:
        etype = event["type"]
        obj = event["data"]["object"]
        if etype in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
            # Only activate once payment has actually settled. For async
            # payment methods (ACH / bank transfer) `checkout.session.completed`
            # fires immediately with payment_status="unpaid"; activating then
            # would grant paid access before money clears. The matching
            # `async_payment_succeeded` event fires when it does. Same gate as
            # /billing/confirm: subscription mode, "paid"/"no_payment_required",
            # and the subscription live at Stripe (`_activate_from_checkout`).
            activated, status = _activate_from_checkout(obj)
            if not activated:
                logger.info("webhook %s did not activate: %s", etype, status)
        elif etype in ("customer.subscription.updated", "customer.subscription.deleted"):
            row = DB.one(
                "SELECT * FROM users WHERE stripe_subscription_id=?", (obj["id"],)
            )
            if row is not None:
                active = (
                    etype != "customer.subscription.deleted"
                    and obj["status"] in _STRIPE_ACTIVE
                )
                DB.q(
                    "UPDATE users SET sub_status=?, current_period_end=COALESCE(?,"
                    " current_period_end), sub_checked_at=? WHERE id=?",
                    (
                        "active" if active else "none",
                        _sub_period_end(obj),
                        _now(),
                        row["id"],
                    ),
                )
        elif etype == "invoice.paid":
            row = DB.one(
                "SELECT * FROM users WHERE stripe_customer_id=?", (_sv(obj, "customer"),)
            )
            if row is not None and _sv(obj, "id"):
                DB.q(
                    "INSERT OR IGNORE INTO payments(user_id,stripe_ref,amount_cents,"
                    "currency,created_at) VALUES(?,?,?,?,?)",
                    (
                        row["id"],
                        str(obj["id"]),
                        int(_sv(obj, "amount_paid") or 0),
                        _sv(obj, "currency") or "usd",
                        _now(),
                    ),
                )

    # --- Admin -----------------------------------------------------------------

    @app.get("/admin")
    def admin_page():
        page = static_dir / "admin.html"
        return HTMLResponse(
            page.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, must-revalidate"},
        )

    @app.get("/admin/api/users")
    def admin_users():
        rows = DB.q(
            """
            SELECT u.*,
                   COALESCE((SELECT hands FROM usage WHERE user_id=u.id AND day=?), 0)
                     AS hands_today,
                   COALESCE((SELECT SUM(hands) FROM usage WHERE user_id=u.id), 0)
                     AS hands_total
            FROM users u ORDER BY u.created_at DESC
            """,
            (_today(),),
        )
        return {
            "users": [
                {
                    "id": r["id"],
                    "email": r["email"],
                    "name": r["name"],
                    "created_at": r["created_at"],
                    "last_login_at": r["last_login_at"],
                    "sub_status": r["sub_status"],
                    "sub_source": r["sub_source"],
                    "is_admin": r["email"].lower() in ADMIN_EMAILS,
                    "hands_today": r["hands_today"],
                    "hands_total": r["hands_total"],
                    "period_end": r["current_period_end"],
                    # (in the main club — see _GAMES_MEMBER_HOOK)
                    "homegame_access": bool(_GAMES_MEMBER_HOOK(int(r["id"]))) if _GAMES_MEMBER_HOOK else False,
                }
                for r in rows
            ]
        }

    @app.post("/admin/api/grant")
    def admin_grant(body: dict):
        uid = body_int(body, "user_id")
        action = body.get("action", "")
        user = _user_by_id(uid)
        if user is None:
            raise HTTPException(status_code=404, detail="no such user")
        if action == "grant":
            DB.q(
                "UPDATE users SET sub_status='active', sub_source='comp' WHERE id=?",
                (uid,),
            )
        elif action == "revoke":
            if user["sub_source"] == "stripe":
                raise HTTPException(
                    status_code=400,
                    detail="Stripe subscription — cancel via Stripe, not here.",
                )
            DB.q(
                "UPDATE users SET sub_status='none', sub_source='' WHERE id=?", (uid,)
            )
        else:
            raise HTTPException(status_code=400, detail="action must be grant|revoke")
        return {"ok": True, "user_id": uid, "action": action}

    @app.post("/admin/api/games_access")
    def admin_games_access(body: dict):
        """Add someone to / remove them from the MAIN home-games club (the site's
        original private circle; made on the first grant, owned by the granting
        admin). Everyone signed in can use home games; this is only the club.
        The old flag is still written so the history of who was in stays."""
        uid = body_int(body, "user_id")
        action = body.get("action", "")
        user = _user_by_id(uid)
        if user is None:
            raise HTTPException(status_code=404, detail="no such user")
        if action not in ("grant", "revoke"):
            raise HTTPException(status_code=400, detail="action must be grant|revoke")
        admin_uid = _CURRENT_USER_ID.get()
        member = False
        if _GAMES_ACCESS_HOOK is not None:
            member = bool(_GAMES_ACCESS_HOOK(int(admin_uid) if admin_uid is not None else uid, uid, action == "grant"))
        DB.q("UPDATE users SET homegame_access=? WHERE id=?", (1 if action == "grant" else 0, uid))
        return {"ok": True, "user_id": uid, "action": action, "homegame_access": member}

    @app.get("/admin/api/metrics")
    def admin_metrics():
        users_n = DB.one("SELECT COUNT(*) c FROM users")["c"]
        stripe_active = DB.one(
            "SELECT COUNT(*) c FROM users WHERE sub_status='active' AND sub_source='stripe'"
        )["c"]
        comp_active = DB.one(
            "SELECT COUNT(*) c FROM users WHERE sub_status='active' AND sub_source='comp'"
        )["c"]
        revenue = DB.one("SELECT COALESCE(SUM(amount_cents),0) c FROM payments")["c"]
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        signups = DB.q(
            "SELECT substr(created_at,1,10) day, COUNT(*) n FROM users"
            " WHERE substr(created_at,1,10) >= ? GROUP BY day ORDER BY day",
            (cutoff,),
        )
        hands = DB.q(
            "SELECT day, SUM(hands) n FROM usage WHERE day >= ? GROUP BY day ORDER BY day",
            (cutoff,),
        )
        payments = DB.q(
            "SELECT p.*, u.email FROM payments p JOIN users u ON u.id=p.user_id"
            " ORDER BY p.created_at DESC LIMIT 20"
        )
        return {
            "users": users_n,
            "active_stripe_subs": stripe_active,
            "active_comp_subs": comp_active,
            "mrr_cents": stripe_active * PRICE_CENTS,
            "revenue_cents_total": revenue,
            "price_cents": PRICE_CENTS,
            "free_hands_per_day": FREE_HANDS_PER_DAY,
            "signups_by_day": [{"day": r["day"], "n": r["n"]} for r in signups],
            "hands_by_day": [{"day": r["day"], "n": r["n"]} for r in hands],
            "recent_payments": [
                {
                    "email": r["email"],
                    "amount_cents": r["amount_cents"],
                    "currency": r["currency"],
                    "created_at": r["created_at"],
                    "ref": r["stripe_ref"],
                }
                for r in payments
            ],
        }

    @app.get("/admin/api/active")
    def admin_active():
        """Deploy-safety signal: who is on the site right now. The headline
        count excludes admins so the asking admin's own browsing never makes
        the site look busy."""
        rows = [r for r in (_user_by_id(u) for u in _active_uids()) if r is not None]
        non_admin = [r for r in rows if not _is_admin(r)]
        return {
            "window_seconds": ACTIVE_WINDOW_S,
            "active_users": len(non_admin),
            "active_total": len(rows),
            "emails": sorted(r["email"] for r in non_admin),
        }

    # Middleware LAST (add_middleware prepends: Session must wrap Access).
    app.add_middleware(AccessMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=_session_secret(),
        max_age=30 * 24 * 3600,
        same_site="lax",
        https_only=BASE_URL.startswith("https"),
    )

    from plo5bp.ui import homegame as _homegame

    _homegame.install(app, static_dir=static_dir)

    logger.info(
        "PUBLIC service installed: base=%s db=%s admins=%s oauth=%s stripe=%s dev_login=%s",
        BASE_URL,
        DB_PATH,
        ",".join(sorted(ADMIN_EMAILS)),
        "on" if oauth else "OFF",
        "on" if STRIPE_SECRET_KEY else "OFF",
        DEV_LOGIN,
    )
