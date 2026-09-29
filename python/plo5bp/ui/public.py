"""Public-build service layer: auth, per-user state, access control, billing, admin.

Installed by ``server.py`` ONLY when ``PLO5BP_PUBLIC`` is truthy (the same flag
that keeps the live-capture routes out). The local build never imports this.

What it adds around the existing app:

- **Sign-in.** Google (Authlib; the account chooser is always shown) and, when
  an email provider is configured, a one-time email link (``/auth/email``). No
  passwords are stored. Accounts are found by Google's stable id first, then
  by email. A loopback-only dev login (``PLO5BP_DEV_LOGIN=1``) exists so the
  flow can be exercised before OAuth credentials exist; the route is only
  registered when ``PLO5BP_BASE_URL`` is itself a loopback URL, and it refuses
  non-loopback clients and anything that arrived through a proxy/tunnel.
- **Sessions.** Signed cookies (key from ``PLO5BP_SESSION_SECRET``, rotation
  supported; the legacy database key is honoured for one cookie lifetime).
  Each cookie carries the account's session version, so "sign out everywhere"
  and disabling an account take effect on the next request.
- **Per-user state.** The study ``Session`` and the ``TrainerSession`` are
  per-user (LRU registry, capacity-capped, idle runtimes swept). The access
  middleware resolves the user ONCE per request and the runtime at most once
  (``_REQUEST_RUNTIME``), so a handler can never read one object and write
  another after an eviction.
- **Access middleware** (pure ASGI — HTTP and websockets alike): request-body
  size limit, cross-site request refusal for state-changing methods, auth,
  admin/subscription gates, free-tier metering, per-user rate limits and
  queuing of CPU-heavy Study/Trainer work behind a site-wide work gate, and
  friendly HTML errors for browser navigations.
- **Billing (dormant while** ``FREE_FOR_ALL``**)**: while the models are in
  development every signed-in user is entitled and checkout is closed. With
  ``PLO5BP_FREE_FOR_ALL=0`` the paywall returns: N free trainer hands per UTC
  day, Study for subscribers, Stripe subscriptions ($10/mo default) verified
  via the success redirect and the optional webhook, lazily re-verified.
- **Admin** (email allowlist, optionally pinned to Google ids): users, comp
  grants, disable / sign-out-everywhere / export / delete, the audit log,
  revenue, the System panel (served models, health, timings, recent errors),
  model reload / promote / rollback and the maintenance notice.
- **Account self-service**: export my data, delete my account (anonymized so
  home-game ledgers and other players' histories stay consistent).

Unit note: money is stored in cents; days are UTC ``YYYY-MM-DD`` strings.
"""

from __future__ import annotations

import hashlib
import html as _html
import ipaddress
import json
import logging
import os
import posixpath
import re
import secrets
import smtplib
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.middleware.sessions import SessionMiddleware
from starlette.routing import Match

from plo5bp.ui import middleware as mw
from plo5bp.ui.common import env_flag, model_slots
from plo5bp.ui.ratelimit import KeyedCounter, KeyedGates, RateLimiter, WorkGate

logger = logging.getLogger("plo5bp.ui.public")

# --- Config (env) -----------------------------------------------------------
# Every environment setting of the service is read by `_read_settings()`: once
# at import, and again whenever an app installs this layer (`install`). The app
# factory (`server.create_app`, BE-007) may build several apps in one process —
# each from the environment as it is at that moment; the server builds one,
# right after import, so its values are the ones the process started with.
# The module names are the settings in force: the code reads them when it runs
# (so tests can still monkeypatch them).

REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_flag(name: str) -> bool:
    """Kept for callers of the old helper; the one parser is `common.env_flag`."""
    return env_flag(name)


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
#: The name the auto-created test-mode product gets.
STRIPE_PRODUCT_NAME = "WrapGTO Monthly"
#: A verified Stripe status is trusted this long while inside the paid period.
STRIPE_RECHECK = timedelta(hours=24)
#: Stripe subscription statuses that carry paid access.
_STRIPE_ACTIVE = ("active", "trialing", "past_due")
#: Signed session cookies live this long.
SESSION_MAX_AGE_S = 30 * 24 * 3600
_BODY_LIMITS = {
    "/games/api/me/avatar": 512 * 1024,  # a <=200 KB picture as a data URL
    "/stripe/webhook": 1024 * 1024,
}
EMAIL_TOKEN_TTL = timedelta(minutes=15)
_EMAIL_RE = re.compile(r"^[^@\s<>\"',;]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,63}$")


def _read_settings(environ: Mapping[str, str] | None = None) -> None:
    """(Re)read every environment setting of the service into the module names
    below — ONE parse, used at import and by ``install`` (BE-007)."""
    global BASE_URL, DB_PATH, ADMIN_EMAILS, ADMIN_SUBS, FREE_HANDS_PER_DAY, FREE_FOR_ALL
    global PRICE_CENTS, DEV_LOGIN_REQUESTED, BASE_IS_LOOPBACK, DEV_LOGIN, DEV_LOGIN_TESTCLIENT
    global GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET
    global STRIPE_PRICE_ID, STRIPE_TIMEOUT_S, STRIPE_GRACE, STRIPE_RETRY_S
    global MAX_USER_RUNTIMES, RUNTIME_IDLE_S, ACTIVE_WINDOW_S, SESSION_COOKIE
    global MAX_BODY_BYTES, RATE_PER_S, RATE_BURST, MODEL_SLOTS, MODEL_WAIT_S, USER_QUEUE
    global USER_INFLIGHT, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM, SMTP_SSL
    global _EMAIL_MODE, EMAIL_LOG_ONLY, EMAIL_LOGIN
    env = os.environ if environ is None else environ

    def flag(name: str, default: bool = False) -> bool:
        return env_flag(name, default, environ=env)

    BASE_URL = env.get("PLO5BP_BASE_URL", "http://127.0.0.1:8770").rstrip("/")
    DB_PATH = Path(env.get("PLO5BP_DB", str(REPO_ROOT / "data" / "public.db")))
    ADMIN_EMAILS = {
        e.strip().lower()
        for e in env.get("PLO5BP_ADMIN_EMAILS", "themilesgarcia@icloud.com").split(",")
        if e.strip()
    }
    #: Optional second factor for admin rights (SEC-019): when set, an admin must
    #: ALSO be signed in with one of these Google account ids (`sub`), so another
    #: identity asserting the same email can never inherit admin.
    ADMIN_SUBS = {
        s.strip() for s in env.get("PLO5BP_ADMIN_SUBS", "").split(",") if s.strip()
    }
    FREE_HANDS_PER_DAY = int(env.get("PLO5BP_FREE_HANDS", "5"))
    # (2026-09-22) The models are still in development, so for now the whole site
    # is FREE: every signed-in user is entitled — no daily hand quota, Study is
    # unlocked, checkout is closed. Sign-in stays (per-user sessions, abuse
    # control). PLO5BP_FREE_FOR_ALL=0 brings the paywall back unchanged; the
    # billing code and its tests are all still here.
    FREE_FOR_ALL = flag("PLO5BP_FREE_FOR_ALL", default=True)
    PRICE_CENTS = int(env.get("PLO5BP_PRICE_CENTS", "1000"))  # $10/mo

    # Dev login (review 2026-09-20 F4). The env flag alone is NOT enough: the
    # fake sign-in lets anyone become any email (the admin's included), so the
    # route only exists when the deployment's own BASE_URL is a loopback URL — a
    # prod/tunnel config (https://wrapgto.com) can never mount it even if the
    # flag leaks into its env file. Per-request checks are in `_dev_login_request_ok`.
    DEV_LOGIN_REQUESTED = flag("PLO5BP_DEV_LOGIN")
    BASE_IS_LOOPBACK = _is_loopback_host(urlsplit(BASE_URL).hostname)
    DEV_LOGIN = DEV_LOGIN_REQUESTED and BASE_IS_LOOPBACK
    # Starlette's TestClient reports client host "testclient" / Host "testserver".
    # Those are only accepted when a test fixture opts in explicitly.
    DEV_LOGIN_TESTCLIENT = flag("PLO5BP_DEV_LOGIN_TESTCLIENT")

    GOOGLE_CLIENT_ID = env.get("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = env.get("GOOGLE_CLIENT_SECRET", "")
    STRIPE_SECRET_KEY = env.get("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET = env.get("STRIPE_WEBHOOK_SECRET", "")
    STRIPE_PRICE_ID = env.get("STRIPE_PRICE_ID", "")
    # Stripe re-validation policy (review 2026-09-20 F2).
    #: Per-request network budget for a Stripe call (the library default is 80 s).
    STRIPE_TIMEOUT_S = float(env.get("PLO5BP_STRIPE_TIMEOUT", "8"))
    #: Fail-open cap: when Stripe cannot be reached, a cached "active" status is
    #: honoured only until ``current_period_end`` + this grace.
    STRIPE_GRACE = timedelta(days=float(env.get("PLO5BP_STRIPE_GRACE_DAYS", "3")))
    #: Minimum spacing of re-checks for one user once the period has ended (or
    #: after a failed check) — never one Stripe call per request.
    STRIPE_RETRY_S = float(env.get("PLO5BP_STRIPE_RETRY_S", "900"))

    MAX_USER_RUNTIMES = int(env.get("PLO5BP_MAX_RUNTIMES", "300"))
    #: A per-user runtime (study Session + TrainerSession) idle this long is
    #: dropped (PERF-024); coming back later starts a fresh hand, as after a restart.
    RUNTIME_IDLE_S = float(env.get("PLO5BP_RUNTIME_IDLE_S", str(6 * 3600)))
    # A signed-in user counts as "active" for this many seconds after their last
    # authenticated request (the admin top-bar deploy-safety counter).
    ACTIVE_WINDOW_S = int(env.get("PLO5BP_ACTIVE_WINDOW", "300"))

    # Name of the sign-in cookie. Browsers share cookies across PORTS of one host, so
    # two local servers on 127.0.0.1 would sign each other out; the local preview
    # launcher gives each port its own name. Production keeps the default.
    SESSION_COOKIE = (env.get("PLO5BP_SESSION_COOKIE") or "session").strip() or "session"

    # Request limits (SEC-015 / PERF-013 / PERF-014). All generous for a person;
    # they exist to stop a script or a stuck client from starving everyone else.
    MAX_BODY_BYTES = int(env.get("PLO5BP_MAX_BODY", str(64 * 1024)))
    #: Heavy (model / Monte-Carlo) requests: per-user token bucket.
    RATE_PER_S = float(env.get("PLO5BP_RATE_PER_S", "4"))
    RATE_BURST = float(env.get("PLO5BP_RATE_BURST", "120"))
    #: Site-wide cap on concurrently RUNNING heavy requests (CPU cores are shared
    #: by every user and the home-games grader). Others queue on the event loop.
    MODEL_SLOTS = model_slots(env)
    #: How long a heavy request may wait for its turn before a polite 503.
    MODEL_WAIT_S = float(env.get("PLO5BP_MODEL_WAIT_S", "30"))
    #: One user's heavy requests run one at a time; this many more may queue.
    USER_QUEUE = int(env.get("PLO5BP_USER_QUEUE", "4"))
    #: Concurrent light (non-streaming) requests per user.
    USER_INFLIGHT = int(env.get("PLO5BP_USER_INFLIGHT", "12"))

    # Email sign-in (ACC-008): a one-time link, disabled until a mail provider is
    # configured. `PLO5BP_EMAIL_LOGIN=log` (loopback deployments only) logs the
    # link instead of sending it, for local testing.
    SMTP_HOST = env.get("PLO5BP_SMTP_HOST", "").strip()
    SMTP_PORT = int(env.get("PLO5BP_SMTP_PORT", "587"))
    SMTP_USER = env.get("PLO5BP_SMTP_USER", "")
    SMTP_PASSWORD = env.get("PLO5BP_SMTP_PASSWORD", "")
    SMTP_FROM = env.get("PLO5BP_SMTP_FROM", "").strip()
    SMTP_SSL = flag("PLO5BP_SMTP_SSL")  # implicit TLS (port 465); default STARTTLS
    _EMAIL_MODE = env.get("PLO5BP_EMAIL_LOGIN", "").strip().lower()
    EMAIL_LOG_ONLY = _EMAIL_MODE == "log" and BASE_IS_LOOPBACK
    EMAIL_LOGIN = _EMAIL_MODE not in ("0", "false", "no", "off") and (
        bool(SMTP_HOST and SMTP_FROM) or EMAIL_LOG_ONLY
    )


_read_settings()

# Study-mode routes: subscription required (the full product). The trainer
# tree is the free-tier surface. `/format` switches the STUDY session's game
# (and 500'd on a free user's never-built env) — review 2026-09-20 F7. The
# `/study/*` tree (BE-009) is gated by PREFIX, so a new Study route under it
# can never be left open by forgetting this list; the root paths are the
# legacy spellings.
STUDY_PATHS = {
    "/state", "/cards", "/action", "/seats", "/config", "/undo", "/reset",
    "/format", "/spot", "/rewind",
}
STUDY_PREFIX = "/study/"
# No auth at all:
OPEN_PREFIXES = ("/static/", "/auth/", "/stripe/webhook", "/health")
OPEN_EXACT = {
    "/", "/me", "/favicon.ico", "/terms", "/privacy",
    "/apple-touch-icon.png", "/apple-touch-icon-precomposed.png",
    "/robots.txt", "/sitemap.xml",
}
#: Model / Monte-Carlo work: rate-limited per user and run behind the gates.
HEAVY_PREFIXES = ("/trainer/", STUDY_PREFIX)
#: Relative cost in the per-user bucket (Monte-Carlo EV and deals cost more).
HEAVY_COST = {
    "/trainer/act": 2.0,
    "/trainer/new_hand": 2.0,
    "/trainer/repeat": 2.0,
    "/trainer/whatif": 2.0,
}
# Free-tier metering (see AccessMiddleware): the explicit deal route, plus
# the trainer routes that deal IMPLICITLY when the session has no live hand.
NEW_HAND_PATH = "/trainer/new_hand"
IMPLICIT_DEAL_PATHS = frozenset({
    "/trainer/state", "/trainer/settings", "/trainer/act", "/trainer/stats/reset",
})
#: State-changing requests that legitimately arrive from another site.
CSRF_EXEMPT = frozenset({"/stripe/webhook"})
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

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
CREATE TABLE IF NOT EXISTS schema_migrations (
  component TEXT NOT NULL,
  version INTEGER NOT NULL,
  name TEXT NOT NULL,
  applied_at TEXT NOT NULL,
  breaking INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (component, version)
);
"""


@dataclass(frozen=True)
class Migration:
    """One numbered, idempotent schema step of a component (OPS-029 / OPS-043).

    ``statements`` run in order inside one transaction (``fn`` gets the raw
    connection for steps that need to look before they leap). ``breaking``
    marks a step older code cannot run against (a rename/drop, a changed
    meaning): a server that does not know such a step refuses to start on the
    database, loudly, instead of misreading it. Additive steps (new tables,
    new nullable/defaulted columns) are NOT breaking, so rolling the code back
    after one keeps working — that is why every step so far only adds."""

    version: int
    name: str
    statements: tuple[str, ...] = ()
    fn: Callable[[sqlite3.Connection], None] | None = None
    breaking: bool = False


class SchemaTooNew(RuntimeError):
    """The database has a breaking migration this code does not know."""


def _add_column(table: str, column: str, ddl: str) -> Callable[[sqlite3.Connection], None]:
    """Migration step: ``ALTER TABLE .. ADD COLUMN`` unless it already exists
    (databases migrated by the pre-versioning code already have some)."""

    def step(conn: sqlite3.Connection) -> None:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    return step


_READ_ONLY_RE = re.compile(r"^\s*(?:--[^\n]*\n\s*|/\*.*?\*/\s*)*(select|with)\b", re.I | re.S)
_WRITE_WORD_RE = re.compile(r"\b(insert|update|delete|replace|create|drop|alter)\b", re.I)


class Db:
    """sqlite wrapper: one WRITER connection behind a lock, plus a read-only
    connection per thread (PERF-012 / PERF-018).

    Writes (and every statement inside ``transaction()``) go through the one
    writer connection, serialized by its lock, exactly as before. A plain
    SELECT outside a transaction runs on the calling thread's own read-only
    connection instead: in WAL mode readers never wait for the writer nor for
    each other, and each SELECT sees the latest committed data — so a long
    stats query or an avatar write no longer stalls every other request.

    Every ``q()`` commits on its own unless it runs inside ``transaction()``,
    which makes a group of statements all-or-nothing (review 2026-09-20 G4:
    a home-game mutation must never be half-persisted); reads inside a
    transaction (or by a thread holding ``_lock``) see its own uncommitted
    writes.

    ``migrate(component, steps)`` applies numbered migrations (see
    :class:`Migration`) tracked in ``schema_migrations``."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        # Re-entrant: q() is called from inside transaction() on one thread.
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._tx_writes: list[str] = []
        self._local = threading.local()
        self._readers: list[sqlite3.Connection] = []
        self._readers_lock = threading.Lock()
        self._reader_failed = False
        self._closed = False
        #: Called with each committed write statement (the users cache listens).
        self.write_listeners: list[Callable[[str], None]] = []
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            # The usual WAL pairing: a crash of the APP never loses a commit;
            # only an OS crash / power cut can drop the very last ones.
            self._conn.execute("PRAGMA synchronous=NORMAL")
            # No existing table declares a foreign key, so this only affects
            # new tables that do.
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- connections --------------------------------------------------------------

    def _reader(self) -> sqlite3.Connection | None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if self._reader_failed or self._closed:
            return None
        try:
            uri = self.path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
        except sqlite3.Error as e:  # e.g. a filesystem without shared memory
            logger.warning(
                "read-only sqlite connections unavailable (%s): all reads use the writer", e
            )
            self._reader_failed = True
            return None
        self._local.conn = conn
        with self._readers_lock:
            self._readers.append(conn)
        return conn

    def _writer_held_here(self) -> bool:
        is_owned = getattr(self._lock, "_is_owned", None)
        if callable(is_owned):
            try:
                return bool(is_owned())
            except Exception:  # noqa: BLE001
                pass
        return self._tx_depth > 0

    @staticmethod
    def _is_read(sql: str) -> bool:
        return bool(_READ_ONLY_RE.match(sql)) and not _WRITE_WORD_RE.search(sql)

    def close(self) -> None:
        """Close every connection (a closed app — ``server.Site.close`` — or a test)."""
        self._closed = True
        with self._readers_lock:
            readers, self._readers = self._readers, []
        for c in readers:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass

    # -- statements ----------------------------------------------------------------

    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        if self._is_read(sql) and not self._writer_held_here():
            reader = self._reader()
            if reader is not None:
                try:
                    return reader.execute(sql, args).fetchall()
                except sqlite3.ProgrammingError:
                    # A reader closed under us (shutdown): fall back to the writer.
                    self._local.conn = None
        with self._lock:
            try:
                cur = self._conn.execute(sql, args)
                rows = cur.fetchall()
            except BaseException:
                if self._tx_depth == 0:
                    self._conn.rollback()
                raise
            write = not self._is_read(sql)
            if self._tx_depth == 0:
                self._conn.commit()
                if write:
                    self._notify((sql,))
            elif write:
                self._tx_writes.append(sql)
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
                    self._tx_writes.clear()
                    self._conn.rollback()
                raise
            else:
                self._tx_depth -= 1
                if self._tx_depth == 0:
                    self._conn.commit()
                    writes, self._tx_writes = self._tx_writes, []
                    self._notify(writes)

    def _notify(self, writes: Iterable[str]) -> None:
        for sql in writes:
            for fn in list(self.write_listeners):
                try:
                    fn(sql)
                except Exception:  # noqa: BLE001 — a listener must not fail a write
                    logger.exception("db write listener failed")

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

    def kv_delete(self, key: str) -> None:
        self.q("DELETE FROM kv WHERE key=?", (key,))

    # -- migrations -------------------------------------------------------------------

    def schema_versions(self) -> dict[str, int]:
        rows = self.q(
            "SELECT component, MAX(version) v FROM schema_migrations GROUP BY component"
        )
        return {r["component"]: int(r["v"]) for r in rows}

    def migrate(self, component: str, steps: list[Migration]) -> list[int]:
        """Apply ``component``'s missing steps in version order, each in its
        own transaction, logged. Returns the versions applied now.

        Refuses (``SchemaTooNew``) when the database already carries a
        BREAKING step of this component that the code does not know — e.g.
        after rolling the code back past a destructive migration. Several
        modules share the database, each with its own component name, so
        their numbering never collides."""
        versions = [s.version for s in steps]
        if len(set(versions)) != len(versions) or versions != sorted(versions):
            raise ValueError(f"{component}: migration versions must be unique and ascending")
        known = max(versions, default=0)
        with self._lock:
            applied = {
                int(r["version"]): bool(r["breaking"])
                for r in self._conn.execute(
                    "SELECT version, breaking FROM schema_migrations WHERE component=?",
                    (component,),
                ).fetchall()
            }
            unknown = {v: b for v, b in applied.items() if v > known}
            if any(unknown.values()):
                raise SchemaTooNew(
                    f"database schema for {component!r} has breaking migration(s) "
                    f"{sorted(v for v, b in unknown.items() if b)} this code does not "
                    f"know (it knows up to v{known}). Refusing to start: deploy the "
                    "newer code, or restore the pre-deploy database backup."
                )
            if unknown:
                logger.warning(
                    "database schema for %s is ahead of this code (additive steps %s) — "
                    "continuing (additive migrations are rollback-safe)",
                    component, sorted(unknown),
                )
            done: list[int] = []
            for step in steps:
                if step.version in applied:
                    continue
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    for stmt in step.statements:
                        self._conn.execute(stmt)
                    if step.fn is not None:
                        step.fn(self._conn)
                    self._conn.execute(
                        "INSERT INTO schema_migrations(component,version,name,applied_at,breaking)"
                        " VALUES(?,?,?,?,?)",
                        (component, step.version, step.name, _now(), 1 if step.breaking else 0),
                    )
                    self._conn.execute("COMMIT")
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    logger.exception(
                        "migration %s v%d (%s) failed", component, step.version, step.name
                    )
                    raise
                logger.info("migrated %s schema to v%d (%s)", component, step.version, step.name)
                done.append(step.version)
            return done


#: The public service's own schema steps. Append only; never renumber.
PUBLIC_MIGRATIONS: list[Migration] = [
    Migration(1, "users.homegame_access", fn=_add_column(
        "users", "homegame_access", "INTEGER NOT NULL DEFAULT 0")),
    Migration(2, "users.disabled", fn=_add_column(
        "users", "disabled", "INTEGER NOT NULL DEFAULT 0")),
    Migration(3, "users.session_version", fn=_add_column(
        "users", "session_version", "INTEGER NOT NULL DEFAULT 0")),
    Migration(4, "users.deleted_at", fn=_add_column("users", "deleted_at", "TEXT")),
    Migration(5, "admin_audit", statements=(
        "CREATE TABLE IF NOT EXISTS admin_audit ("
        " id INTEGER PRIMARY KEY, at TEXT NOT NULL, admin_user_id INTEGER,"
        " action TEXT NOT NULL, target_user_id INTEGER, detail TEXT NOT NULL DEFAULT '{}')",
        "CREATE INDEX IF NOT EXISTS admin_audit_at ON admin_audit(at)",
    )),
    Migration(6, "stripe_events", statements=(
        "CREATE TABLE IF NOT EXISTS stripe_events ("
        " id TEXT PRIMARY KEY, type TEXT NOT NULL, received_at TEXT NOT NULL)",
    )),
    Migration(7, "email_tokens", statements=(
        "CREATE TABLE IF NOT EXISTS email_tokens ("
        " token_hash TEXT PRIMARY KEY, email TEXT NOT NULL, next TEXT,"
        " created_at TEXT NOT NULL, expires_at TEXT NOT NULL, used_at TEXT)",
        "CREATE INDEX IF NOT EXISTS email_tokens_email ON email_tokens(email)",
    )),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class _DatabaseNotOpen:
    """``DB`` until an app installs this layer: importing the module opens no
    database (BE-007) — ``install`` opens the app's own (``_open_database``)."""

    path = None

    def close(self) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"public.DB.{name}: no database is open yet — an app opens its own when it "
            "installs the public layer (server.create_app)"
        )


#: The app's database (per app: ``_open_database``).
DB: Db = _DatabaseNotOpen()  # type: ignore[assignment]


def _open_database() -> Db:
    """Open (creating it if needed) and migrate the app's database, with the
    users cache listening to its writes."""
    global DB
    db = Db(DB_PATH)
    db.migrate("public", PUBLIC_MIGRATIONS)
    db.write_listeners.append(_on_db_write)
    DB = db
    return db


# --- Session keys (SEC-020) ----------------------------------------------------------


def _session_secret() -> str:
    """The legacy signing key kept in the database (created on first boot)."""
    s = DB.kv_get("session_secret")
    if not s:
        s = secrets.token_urlsafe(48)
        DB.kv_set("session_secret", s)
    return s


def _session_keys() -> list[str]:
    """Signing keys, CURRENT FIRST.

    ``PLO5BP_SESSION_SECRET`` (comma-separated: new key first, older keys after
    it for a rotation) keeps the key out of the database — a copy of
    ``public.db`` (a backup, say) can then no longer mint a login cookie. The
    old database key keeps verifying cookies for ONE cookie lifetime after the
    env key first appears (so nobody is signed out by the switch), and is then
    deleted. Without the env var the database key is used, as before."""
    env_keys = [
        k.strip() for k in os.environ.get("PLO5BP_SESSION_SECRET", "").split(",")
        if k.strip()
    ]
    if not env_keys:
        return [_session_secret()]
    since_raw = DB.kv_get("session_secret_env_since")
    if since_raw is None:
        since_raw = _now()
        DB.kv_set("session_secret_env_since", since_raw)
    legacy = DB.kv_get("session_secret")
    since = _parse_iso(since_raw) or datetime.now(timezone.utc)
    if legacy and datetime.now(timezone.utc) < since + timedelta(seconds=SESSION_MAX_AGE_S):
        return env_keys + [legacy]
    if legacy:
        DB.kv_delete("session_secret")
        logger.info(
            "retired the legacy database session key (env key in use since %s)", since_raw
        )
    return env_keys


class _RotatingSessionMiddleware(SessionMiddleware):
    """Starlette's cookie sessions, signing with the first key and accepting
    every listed key (itsdangerous signs with the LAST of its list)."""

    def __init__(self, app: Any, *, secret_keys: list[str], **kwargs: Any) -> None:
        super().__init__(app, secret_key=secret_keys[0], **kwargs)
        import itsdangerous

        self.signer = itsdangerous.TimestampSigner(list(reversed(secret_keys)))


# --- Users -------------------------------------------------------------------


class SignInConflict(Exception):
    """A Google identity asserted an email already bound to ANOTHER Google id."""


def _upsert_user(google_sub: str | None, email: str, name: str, picture: str) -> sqlite3.Row:
    """Find-or-create the account for a sign-in.

    (SEC-019) Google's ``sub`` is the stable identity: it is looked up FIRST
    (a changed Google email follows the same account). An email that already
    belongs to an account bound to a DIFFERENT sub is refused — a second
    identity asserting the same address must not land in someone else's
    account (admin included). Email-only sign-ins (dev / email link) prove
    control of the address and find the account by email.

    (BE-020) One transaction under the writer lock, and the insert itself is
    ``ON CONFLICT(email) DO UPDATE`` — two first sign-ins at once can no
    longer hit the UNIQUE constraint and 500 the callback."""
    email = email.strip().lower()
    now = _now()
    with DB.transaction():
        row = None
        if google_sub:
            row = DB.one("SELECT * FROM users WHERE google_sub=?", (google_sub,))
            if row is not None and row["email"] != email:
                taken = DB.one("SELECT id FROM users WHERE email=? AND id<>?", (email, row["id"]))
                if taken is None:
                    logger.info("user %s: Google email changed to %s", row["id"], email)
                    DB.q("UPDATE users SET email=? WHERE id=?", (email, row["id"]))
                else:
                    logger.warning(
                        "sign-in: Google id of user %s now asserts %s, which belongs to"
                        " user %s — keeping the accounts apart", row["id"], email, taken["id"],
                    )
        if row is None:
            row = DB.one("SELECT * FROM users WHERE email=?", (email,))
            if (row is not None and google_sub and row["google_sub"]
                    and row["google_sub"] != google_sub):
                logger.warning(
                    "sign-in refused: %s is bound to another Google identity (user %s)",
                    email, row["id"],
                )
                raise SignInConflict(email)
        if row is None:
            DB.q(
                "INSERT INTO users(google_sub,email,name,picture,created_at,last_login_at)"
                " VALUES(?,?,?,?,?,?) ON CONFLICT(email) DO UPDATE SET"
                " last_login_at=excluded.last_login_at",
                (google_sub, email, name, picture, now, now),
            )
            return DB.one("SELECT * FROM users WHERE email=?", (email,))
        DB.q(
            "UPDATE users SET google_sub=COALESCE(google_sub,?), name=?, picture=?,"
            " last_login_at=? WHERE id=?",
            (google_sub, name or row["name"], picture or row["picture"], now, row["id"]),
        )
        return DB.one("SELECT * FROM users WHERE id=?", (row["id"],))


def _user_by_id(uid: int) -> sqlite3.Row | None:
    return DB.one("SELECT * FROM users WHERE id=?", (uid,))


# Middleware user cache (PERF-018): the access check runs on EVERY request, on
# the event loop. Rows are cached briefly and the whole cache is invalidated by
# any committed write that touches `users` (Db write listener), so a grant,
# disable or sign-out takes effect on the very next request.
_USER_CACHE_TTL_S = 30.0
_USER_CACHE: dict[int, tuple[float, int, sqlite3.Row | None]]  # (per app: _reset_state)
_USER_CACHE_LOCK = threading.Lock()
_USERS_GEN: list[int]  # (per app: _reset_state)
_USERS_WRITE_RE = re.compile(r"\b(update|into|from)\s+users\b", re.I)


def _on_db_write(sql: str) -> None:
    if _USERS_WRITE_RE.search(sql):
        with _USER_CACHE_LOCK:
            _USERS_GEN[0] += 1
            _USER_CACHE.clear()


_MISS = object()


def _cached_user(uid: int) -> Any:
    with _USER_CACHE_LOCK:
        hit = _USER_CACHE.get(uid)
        if hit is None:
            return _MISS
        expires, gen, row = hit
        if gen != _USERS_GEN[0] or time.monotonic() > expires:
            _USER_CACHE.pop(uid, None)
            return _MISS
        return row


async def _lookup_user(uid: Any) -> sqlite3.Row | None:
    """The user row for a session uid: cache hit inline, miss off the loop."""
    try:
        uid = int(uid)
    except (TypeError, ValueError):
        return None
    row = _cached_user(uid)
    if row is not _MISS:
        return row
    with _USER_CACHE_LOCK:
        gen = _USERS_GEN[0]
    row = await run_in_threadpool(_user_by_id, uid)
    with _USER_CACHE_LOCK:
        if gen == _USERS_GEN[0]:
            if len(_USER_CACHE) > 5000:
                _USER_CACHE.clear()
            _USER_CACHE[uid] = (time.monotonic() + _USER_CACHE_TTL_S, gen, row)
    return row


def _user_live(user: sqlite3.Row | None) -> bool:
    return user is not None and not user["disabled"] and not user["deleted_at"]


def _is_admin(user: sqlite3.Row | None) -> bool:
    if not user or user["email"].lower() not in ADMIN_EMAILS:
        return False
    if ADMIN_SUBS:
        return (user["google_sub"] or "") in ADMIN_SUBS
    return True


def _homegame_access(user: sqlite3.Row | None) -> bool:
    """The home-games pages. Independent of subscription.

    Since clubs (2026-09-25) every signed-in user has them: what a user may SEE
    there — a club's tables, members and numbers — is the club's business
    (homegame.py). Signed out is denial (a sign-in page / the hidden 404).
    """
    return user is not None


# homegame.install: /admin's home-games switch now means "a member of the MAIN
# club" (the site's original circle). (admin_uid, uid, grant) -> member after;
# (uid) -> member?; () -> the set of members' user ids (the user list, PERF-010).
_GAMES_ACCESS_HOOK: Any = None
_GAMES_MEMBER_HOOK: Any = None
_GAMES_MEMBERS_HOOK: Any = None


def _games_members() -> Callable[[int], bool]:
    """``uid -> in the main club?`` for a whole user list: ONE lookup through
    the home games' batch hook (the per-user hook cost two queries per user)."""
    if _GAMES_MEMBERS_HOOK is not None:
        members = {int(u) for u in _GAMES_MEMBERS_HOOK()}
        return lambda uid: int(uid) in members
    if _GAMES_MEMBER_HOOK is not None:
        return lambda uid: bool(_GAMES_MEMBER_HOOK(int(uid)))
    return lambda uid: False


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
    serves ``/static/GAMES.JS`` as the same file. Any ``games.*`` file counts,
    so a new home-games client module needs no entry here (HGB-017)."""
    p = path.lower()
    return p in _GAMES_ASSETS or p.startswith("/static/games.")


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


_NO_STORE = {"Cache-Control": "no-store, must-revalidate"}


def _hidden_not_found(request: Request) -> HTMLResponse | JSONResponse:
    """404 with no feature name — the home-games surface must not advertise
    itself to anyone without access, including a 401/403 distinction. The
    browser page is the site's ordinary 404, so nothing tells them apart."""
    if mw.wants_html(request.headers, request.method):
        return HTMLResponse(mw.error_page(404), status_code=404, headers=_NO_STORE)
    return JSONResponse(mw.error_body(404, "Not Found"), status_code=404, headers=_NO_STORE)


_STRIPE_CLIENT_READY: bool  # (per app: _reset_state)


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


def _login_failure_reason(request: Any, exc: BaseException) -> str:
    """Why an OAuth callback failed, as the landing's `why=` code (site
    ACC-025): "cancelled" (the user pressed Cancel on Google's page),
    "expired" (a stale sign-in tab: the state no longer matches), else
    "error". Never echoes anything from the provider."""
    try:
        err = str(getattr(exc, "error", "") or request.query_params.get("error") or "")
    except Exception:  # noqa: BLE001
        err = ""
    if err == "access_denied":
        return "cancelled"
    if err == "mismatching_state" or "state" in type(exc).__name__.lower():
        return "expired"
    return "error"


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


def _stripe_live_key(key: str | None = None) -> bool:
    k = STRIPE_SECRET_KEY if key is None else key
    return k.startswith(("sk_live_", "rk_live_"))


def _check_stripe_key_is_safe() -> None:
    """(SEC-031) A LIVE Stripe key never runs next to the dev login or on a
    loopback deployment, where anyone at the machine can sign in as anyone —
    admin included — and could create real prices and checkouts."""
    if _stripe_live_key() and (DEV_LOGIN_REQUESTED or BASE_IS_LOOPBACK):
        raise RuntimeError(
            "Refusing to start: STRIPE_SECRET_KEY is a LIVE key, but "
            + ("PLO5BP_DEV_LOGIN is on" if DEV_LOGIN_REQUESTED else
               f"PLO5BP_BASE_URL ({BASE_URL}) is a loopback URL")
            + ". Use an sk_test_ key for local runs."
        )


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
_STRIPE_RETRY_AT: dict[int, float]  # (per app: _reset_state)
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
        and not FREE_FOR_ALL
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
#: Per-request [runtime | None, uid] holder (BE-003): the first access in a
#: request resolves the runtime, every later access reuses THAT object — no
#: registry lock per attribute, and an LRU eviction mid-request cannot swap
#: the Session under a handler. Worker threads share the holder (anyio copies
#: the context, and the copy references the same list).
_REQUEST_RUNTIME: ContextVar[list | None] = ContextVar("plo5bp_request_runtime", default=None)


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
    in-flight hand is lost — same as a browser refresh after eviction.

    (PERF-024) Runtimes idle longer than ``idle_s`` are swept too (at most
    once a minute, from ``get``), and a new runtime is BUILT outside the
    registry lock — building one reads the user's stats file."""

    def __init__(
        self,
        study_factory: Callable[[], Any],
        trainer_factory: Callable[[int], Any],
        *,
        capacity: int | None = None,
        idle_s: float | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self._study_factory = study_factory
        self._trainer_factory = trainer_factory
        self._map: OrderedDict[int, _Runtime] = OrderedDict()
        self._lock = threading.Lock()
        self.capacity = MAX_USER_RUNTIMES if capacity is None else int(capacity)
        self.idle_s = RUNTIME_IDLE_S if idle_s is None else float(idle_s)
        self._clock = clock
        self._next_sweep = 0.0

    def get(self, uid: int) -> _Runtime:
        now = self._clock()
        with self._lock:
            rt = self._map.get(uid)
            if rt is not None:
                rt.last_seen = now
                self._map.move_to_end(uid)
                self._maybe_sweep_locked(now, keep=uid)
                return rt
        built = _Runtime(self._study_factory(), self._trainer_factory(uid))
        built.last_seen = now
        with self._lock:
            rt = self._map.get(uid)
            if rt is None:  # (else another request built it first: use theirs)
                rt = self._map[uid] = built
            rt.last_seen = now
            self._map.move_to_end(uid)
            while len(self._map) > self.capacity:
                evicted_uid, _ = self._map.popitem(last=False)
                logger.info("evicted runtime for user %s (LRU cap)", evicted_uid)
            self._maybe_sweep_locked(now, keep=uid)
            return rt

    def peek(self, uid: int) -> _Runtime | None:
        """The user's runtime if one exists. Never creates or LRU-touches."""
        with self._lock:
            return self._map.get(uid)

    def drop(self, uid: int) -> None:
        """Forget a user's runtime (account disabled / deleted)."""
        with self._lock:
            self._map.pop(uid, None)

    def sweep(self, now: float | None = None) -> int:
        """Evict runtimes idle longer than ``idle_s``; returns how many."""
        with self._lock:
            return self._sweep_locked(self._clock() if now is None else now, keep=None)

    def _maybe_sweep_locked(self, now: float, keep: int | None) -> None:
        if now >= self._next_sweep:
            self._next_sweep = now + 60.0
            self._sweep_locked(now, keep)

    def _sweep_locked(self, now: float, keep: int | None) -> int:
        cutoff = now - self.idle_s
        stale = [u for u, rt in self._map.items() if rt.last_seen < cutoff and u != keep]
        for u in stale:
            del self._map[u]
        if stale:
            logger.info("swept %d idle runtime(s)", len(stale))
        return len(stale)

    def __len__(self) -> int:
        with self._lock:
            return len(self._map)


_REGISTRY: Registry | None  # (per app: built by install)


def _current_runtime() -> _Runtime | None:
    holder = _REQUEST_RUNTIME.get()
    if holder is not None:
        if holder[0] is None and _REGISTRY is not None:
            holder[0] = _REGISTRY.get(int(holder[1]))
        return holder[0]
    uid = _CURRENT_USER_ID.get()
    if uid is None or _REGISTRY is None:
        return None
    return _REGISTRY.get(uid)


# --- Active-user tracking --------------------------------------------------------

# uid -> last authenticated-request time. In-memory on purpose: it feeds the
# admin "safe to deploy?" counter, and a restart (the event it protects
# against) legitimately resets it.
_ACTIVITY: dict[int, float]  # (per app: _reset_state)
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


# --- Maintenance notice (FEAT-028) ------------------------------------------------

#: A notice lingers this long after its announced time, then clears itself.
MAINTENANCE_LINGER = timedelta(hours=2)


def maintenance_notice() -> dict[str, Any] | None:
    """The admin's "restart coming" heads-up, or None. Shared with every page
    that polls /me (Study, Trainer, home-game tables)."""
    raw = DB.kv_get("maintenance")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    at = _parse_iso(data.get("at"))
    if at is not None and datetime.now(timezone.utc) > at + MAINTENANCE_LINGER:
        return None
    return {"message": str(data.get("message") or ""), "at": data.get("at")}


# --- Admin audit (SEC-022) ----------------------------------------------------------


def _audit(action: str, target_user_id: int | None = None, **detail: Any) -> None:
    """Record an admin action: who, when, what, to whom."""
    try:
        DB.q(
            "INSERT INTO admin_audit(at,admin_user_id,action,target_user_id,detail)"
            " VALUES(?,?,?,?,?)",
            (_now(), _CURRENT_USER_ID.get(), action, target_user_id,
             json.dumps(detail, separators=(",", ":"), default=str)),
        )
    except Exception:  # noqa: BLE001 — never fail the action over its log line
        logger.exception("admin audit write failed (%s)", action)
    logger.info(
        "admin %s: %s user=%s %s", _CURRENT_USER_ID.get(), action, target_user_id, detail
    )


# --- Accounts: export / delete (ACC-009 / SEC-017) ------------------------------------

#: Other modules' parts of a user's data (home games registers its own):
#: name -> {"export": fn(uid) -> dict, "blockers": fn(uid) -> list[str],
#:          "anonymize": fn(uid) -> None}.
ACCOUNT_HOOKS: dict[str, dict[str, Callable[..., Any]]] = {}
_STATS_DIR: Path | None  # (per app: install)
_DELETED_NAME = "Deleted player"


def _trainer_stats_path(uid: int) -> Path | None:
    return None if _STATS_DIR is None else _STATS_DIR / f"u{int(uid)}.json"


def export_user(uid: int) -> dict[str, Any]:
    """Everything the site stores about one account, as plain JSON."""
    user = _user_by_id(uid)
    if user is None:
        raise HTTPException(status_code=404, detail="no such user")
    profile = {
        k: user[k] for k in (
            "id", "email", "name", "picture", "created_at", "last_login_at",
            "sub_status", "sub_source", "current_period_end",
        )
    }
    profile["google_account_linked"] = bool(user["google_sub"])
    out: dict[str, Any] = {
        "exported_at": _now(),
        "site": BASE_URL,
        "account": profile,
        "trainer_usage": [
            {"day": r["day"], "hands": r["hands"]}
            for r in DB.q("SELECT day, hands FROM usage WHERE user_id=? ORDER BY day", (uid,))
        ],
        "payments": [
            {"amount_cents": r["amount_cents"], "currency": r["currency"],
             "created_at": r["created_at"], "ref": r["stripe_ref"]}
            for r in DB.q("SELECT * FROM payments WHERE user_id=? ORDER BY created_at", (uid,))
        ],
    }
    path = _trainer_stats_path(uid)
    if path is not None and path.exists():
        try:
            out["trainer_stats"] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out["trainer_stats"] = None
    for name, hook in ACCOUNT_HOOKS.items():
        fn = hook.get("export")
        if fn is not None:
            out[name] = fn(uid)
    return out


def account_blockers(uid: int) -> list[str]:
    """Reasons an account cannot be deleted right now (empty = it can)."""
    user = _user_by_id(uid)
    if user is None:
        return ["no such account"]
    out: list[str] = []
    if (user["sub_status"] == "active" and user["sub_source"] == "stripe"
            and user["stripe_subscription_id"]):
        out.append("Cancel your subscription first (Manage billing), then delete the account.")
    for hook in ACCOUNT_HOOKS.values():
        fn = hook.get("blockers")
        if fn is not None:
            out.extend(fn(uid))
    return out


def delete_user(uid: int) -> None:
    """Delete an account: personal data goes, shared records stay consistent.

    The user row becomes an anonymous tombstone ("Deleted player", an
    unusable address, no Google link, signed out everywhere) instead of
    disappearing, because other players' home-game hand histories, head-to-
    heads and zero-sum ledgers reference its id — those keep adding up, they
    just no longer name the person. Usage counters, the trainer stats file,
    and (via the hooks) pictures, chat and host preferences are removed."""
    blockers = account_blockers(uid)
    if blockers:
        raise HTTPException(status_code=409, detail=" ".join(blockers))
    for hook in ACCOUNT_HOOKS.values():
        fn = hook.get("anonymize")
        if fn is not None:
            fn(uid)
    with DB.transaction():
        DB.q("DELETE FROM email_tokens WHERE email IN (SELECT email FROM users WHERE id=?)", (uid,))
        DB.q(
            "UPDATE users SET email=?, name=?, picture='', google_sub=NULL,"
            " stripe_customer_id=NULL, stripe_subscription_id=NULL,"
            " sub_status='none', sub_source='', current_period_end=NULL,"
            " homegame_access=0, disabled=1, deleted_at=?,"
            " session_version=session_version+1 WHERE id=?",
            (f"deleted-{int(uid)}@deleted.invalid", _DELETED_NAME, _now(), uid),
        )
        DB.q("DELETE FROM usage WHERE user_id=?", (uid,))
    path = _trainer_stats_path(uid)
    if path is not None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning("could not delete trainer stats of user %s: %s", uid, e)
    if _REGISTRY is not None:
        _REGISTRY.drop(uid)
    logger.info("account %s deleted (anonymized)", uid)


# --- Email sign-in (ACC-008) --------------------------------------------------------


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _send_login_email(email: str, link: str) -> None:
    """Deliver the sign-in link (blocking — callers run off the event loop)."""
    if EMAIL_LOG_ONLY:
        logger.warning("EMAIL LOGIN (log mode, loopback only) for %s: %s", email, link)
        return
    msg = EmailMessage()
    msg["Subject"] = "Your WrapGTO sign-in link"
    msg["From"] = SMTP_FROM
    msg["To"] = email
    msg.set_content(
        "Sign in to WrapGTO with this link (it works once, for 15 minutes):\n\n"
        f"{link}\n\n"
        "If you didn't ask for it, ignore this email — nobody can sign in without the link.\n"
    )
    msg.add_alternative(
        "<p>Sign in to WrapGTO:</p>"
        f'<p><a href="{_html.escape(link)}" style="display:inline-block;padding:10px 18px;'
        'background:#18d2c3;color:#04201d;border-radius:8px;text-decoration:none;'
        'font-weight:600">Sign in</a></p>'
        "<p>The link works once, for 15 minutes. If you didn't ask for it, ignore this email.</p>",
        subtype="html",
    )
    if SMTP_SSL:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            if SMTP_USER:
                s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            s.starttls()
            if SMTP_USER:
                s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)


def _page(title: str, body_html: str, status: int = 200) -> HTMLResponse:
    """A small branded server-rendered page (the sign-in flow screens)."""
    return HTMLResponse(mw.plain_page(title, body_html), status_code=status, headers=_NO_STORE)


# --- Access middleware ------------------------------------------------------------


def _wants_sub_route(path: str) -> bool:
    return path in STUDY_PATHS or path.startswith(STUDY_PREFIX)


def _open_route(path: str) -> bool:
    return path in OPEN_EXACT or path.startswith(OPEN_PREFIXES)


def _heavy_route(path: str) -> bool:
    return path in STUDY_PATHS or path.startswith(HEAVY_PREFIXES)


def _body_limit(path: str) -> int:
    return _BODY_LIMITS.get(path, MAX_BODY_BYTES)


def _same_site_request(headers: Headers) -> bool:
    """(SEC-006 / SEC-018) Did a state-changing request come from this site?

    Browsers say so outright in ``Sec-Fetch-Site`` (only "same-origin" — a
    sibling subdomain is "same-site" and is refused too — or "none", a user
    typing / a bookmark). Older browsers send ``Origin`` on every POST: its
    host must be this site's (the public URL or the Host the request came
    in on). A request with neither header is not from a browser, so it
    carries no ambient cookie to abuse."""
    sfs = headers.get("sec-fetch-site")
    if sfs is not None:
        return sfs.strip().lower() in ("same-origin", "none")
    origin = headers.get("origin")
    if origin is None:
        return True
    if origin.strip().lower() == "null":
        return False
    try:
        o_host = (urlsplit(origin).netloc or "").lower()
    except ValueError:
        return False
    allowed = {(urlsplit(BASE_URL).netloc or "").lower(), (headers.get("host") or "").lower()}
    allowed.discard("")
    return o_host in allowed


class _BodyTooLarge(Exception):
    pass


def _limited_receive(receive: Callable, limit: int) -> Callable:
    seen = [0]

    async def wrapped() -> dict:
        message = await receive()
        if message.get("type") == "http.request":
            seen[0] += len(message.get("body") or b"")
            if seen[0] > limit:
                raise _BodyTooLarge()
        return message

    return wrapped


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


# Limits shared by every request of the app (per app: _reset_state).
_HEAVY_RATE: RateLimiter
_USER_GATES: KeyedGates
MODEL_GATE: WorkGate
_INFLIGHT: KeyedCounter
#: Filled by `install`: does any route match this (http) scope?
_ROUTE_EXISTS: Callable[[dict], bool] | None


def _deny(
    scope: dict,
    status: int,
    detail: Any,
    code: str | None = None,
    *,
    headers: dict[str, str] | None = None,
    sign_in: bool = False,
    **extra: Any,
) -> Response:
    """A refusal from the access layer: branded page for a browser
    navigation, the one JSON error shape for everything else."""
    hdrs = {**_NO_STORE, **(headers or {})}
    if mw.wants_html(Headers(scope=scope), scope.get("method", "GET")):
        message = detail if isinstance(detail, str) and status not in (401, 404) else None
        return HTMLResponse(
            mw.error_page(status, message, sign_in=sign_in),
            status_code=status, headers=hdrs,
        )
    return JSONResponse(
        mw.error_body(status, detail, code, **extra), status_code=status, headers=hdrs
    )


class AccessMiddleware:
    """Auth + limits + entitlement + free-quota enforcement (pure ASGI).

    Pure ASGI (SEC-025): websocket connections pass through here too (the
    public app has none today — a test pins that — but a future one would
    still need a signed-in account and a same-site origin), and the SSE
    streams no longer go through BaseHTTPMiddleware's extra task."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "http":
            await self._http(scope, receive, send)
        elif scope["type"] == "websocket":
            await self._websocket(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    # -- identity -------------------------------------------------------------------

    async def _identify(self, scope: dict) -> tuple[sqlite3.Row | None, str | None]:
        """(live user | None, why-refused | None). A cookie from before a
        "sign out everywhere", or of a disabled/deleted account, is cleared
        here — the browser drops it on this very response."""
        sess = scope.get("session")
        if not isinstance(sess, dict):
            return None, None
        uid = sess.get("uid")
        if uid is None:
            return None, None
        user = await _lookup_user(uid)
        if user is None or user["deleted_at"]:
            sess.clear()
            return None, None
        if user["disabled"]:
            sess.clear()
            return None, "disabled"
        try:
            sv = int(sess.get("sv", 0))
        except (TypeError, ValueError):
            sv = -1
        if sv != int(user["session_version"] or 0):
            sess.clear()
            return None, None
        return user, None

    # -- websockets -------------------------------------------------------------------

    async def _websocket(self, scope: dict, receive: Callable, send: Callable) -> None:
        user, _why = await self._identify(scope)
        if user is None or not _same_site_request(Headers(scope=scope)):
            # Closing before accept = the server answers the handshake 403.
            await send({"type": "websocket.close", "code": 1008})
            return
        uid = int(user["id"])
        token = _CURRENT_USER_ID.set(uid)
        rt_token = _REQUEST_RUNTIME.set([None, uid])
        try:
            await self.app(scope, receive, send)
        finally:
            _REQUEST_RUNTIME.reset(rt_token)
            _CURRENT_USER_ID.reset(token)

    # -- http -----------------------------------------------------------------------------

    async def _http(self, scope: dict, receive: Callable, send: Callable) -> None:
        # Authorize on the NORMALIZED ASGI path (review 2026-09-20 F1/F7):
        # `scope["path"]` is what the router matches (request.url is rebuilt
        # from the Host header), and `_norm_path` closes the `//` and
        # trailing-slash spellings the layers below treat as equivalent.
        path = _norm_path(scope.get("path"))
        method = scope.get("method", "GET")
        headers = Headers(scope=scope)

        # (SEC-015) Bodies are capped before anything parses them: the
        # declared length at once, a chunked body as it streams in.
        limit = _body_limit(path)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                too_big = int(declared) > limit
            except ValueError:
                too_big = True
            if too_big:
                await _deny(scope, 413, "Request body too large.")(scope, receive, send)
                return
        receive = _limited_receive(receive, limit)

        # (SEC-006 / SEC-018) Cross-site state changes are refused outright,
        # before the cookie is even looked at.
        if method not in _SAFE_METHODS and path not in CSRF_EXEMPT:
            if not _same_site_request(headers):
                await _deny(scope, 403, "Cross-site request refused.", "cross_site")(
                    scope, receive, send)
                return

        user, why = await self._identify(scope)
        state = scope.setdefault("state", {})
        if user is not None and isinstance(state, dict):
            state["uid"] = int(user["id"])

        # Home games: signed out = the sign-in page for a browser, the hidden
        # 404 for everything else (clubs, 2026-09-25: every signed-in user has
        # them). Do this BEFORE the /static open-prefix short-circuit and
        # BEFORE the generic 401 so a signed-out probe of /games or
        # /static/games.js looks like a missing page.
        if _games_path(path) or _games_asset(path):
            if not _homegame_access(user):
                hook = _GAMES_INVITE_HOOK
                request = Request(scope, receive)
                invite = (
                    await run_in_threadpool(hook, request, path, user)
                    if hook is not None else None
                )
                resp = invite if invite is not None else _hidden_not_found(request)
                await resp(scope, receive, send)
                return

        if _open_route(path) and not _games_path(path):
            await self._call(scope, receive, send, user)
            return

        if user is None:
            if why == "disabled":
                await _deny(scope, 403, "This account has been disabled.", "account_disabled")(
                    scope, receive, send)
                return
            # A browser that opens a page that does not exist gets "not
            # found", not "please sign in" (ACC-014).
            if (mw.wants_html(headers, method) and _ROUTE_EXISTS is not None
                    and not _ROUTE_EXISTS(scope)):
                await _deny(scope, 404, None)(scope, receive, send)
                return
            await _deny(scope, 401, "auth required", "auth", sign_in=True, error="auth")(
                scope, receive, send)
            return

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
                await _deny(scope, 403, "admin only", "admin_only")(scope, receive, send)
                return

        if _wants_sub_route(path) and not entitled:
            await _deny(
                scope, 402, "Study mode requires a subscription.",
                "subscription_required", error="subscription_required",
            )(scope, receive, send)
            return

        if _heavy_route(path):
            await self._heavy(scope, receive, send, user, path, method, entitled)
            return

        if path.endswith("/stream"):  # long-lived: not counted as in-flight
            await self._call(scope, receive, send, user)
            return
        if not _INFLIGHT.enter(user_id):
            await _deny(scope, 429, "Too many requests at once — slow down a little.",
                        "busy", headers={"Retry-After": "1"})(scope, receive, send)
            return
        try:
            await self._call(scope, receive, send, user)
        finally:
            _INFLIGHT.leave(user_id)

    async def _heavy(
        self, scope: dict, receive: Callable, send: Callable,
        user: sqlite3.Row, path: str, method: str, entitled: bool,
    ) -> None:
        """Study / Trainer work (PERF-013 / PERF-014): a per-user token bucket
        refuses floods at once; the user's requests then run ONE AT A TIME
        (queued here on the event loop, never parking a worker thread on a
        session lock), and at most MODEL_SLOTS run site-wide."""
        user_id = int(user["id"])
        ok, retry = _HEAVY_RATE.allow(user_id, HEAVY_COST.get(path, 1.0))
        if not ok:
            await _deny(
                scope, 429, "Too many requests — give it a second.", "rate_limited",
                headers={"Retry-After": str(max(1, int(retry + 0.999)))},
            )(scope, receive, send)
            return
        async with _USER_GATES.hold(user_id, MODEL_WAIT_S) as mine:
            if not mine:
                await _deny(scope, 429, "Still working on your last request.", "busy",
                            headers={"Retry-After": "1"})(scope, receive, send)
                return
            async with MODEL_GATE.hold(MODEL_WAIT_S) as slot:
                if not slot:
                    await _deny(
                        scope, 503, "The site is busy right now — try again in a moment.",
                        "busy", headers={"Retry-After": "2"},
                    )(scope, receive, send)
                    return
                await self._metered(scope, receive, send, user, path, method, entitled)

    async def _metered(
        self, scope: dict, receive: Callable, send: Callable,
        user: sqlite3.Row, path: str, method: str, entitled: bool,
    ) -> None:
        """Free-tier metering around the call. The explicit deal counts for
        everyone (admin metrics); an implicit deal for non-entitled users only."""
        user_id = int(user["id"])
        consumed = False
        implicit_ts = None
        implicit_hand_no = 0
        used = 0
        if path == NEW_HAND_PATH and method == "POST":
            consumed = True
        elif not entitled and path in IMPLICIT_DEAL_PATHS:
            implicit_ts = _implicit_deal_session(user_id)
            if implicit_ts is not None:
                implicit_hand_no = int(implicit_ts.hand_no)
                consumed = True
        if consumed:
            used = await run_in_threadpool(_record_hand, user_id)
            if not entitled and used > FREE_HANDS_PER_DAY:
                await run_in_threadpool(_refund_hand, user_id)
                await _deny(
                    scope, 402,
                    f"Free limit reached: {FREE_HANDS_PER_DAY} trainer hands per day."
                    " Subscribe for unlimited.",
                    "free_limit", error="free_limit", used=used - 1,
                    limit=FREE_HANDS_PER_DAY, resets_at=_resets_at(),
                )(scope, receive, send)
                return
        if not consumed:
            await self._call(scope, receive, send, user)
            return

        status_box = [0]

        def no_deal(status: int) -> bool:
            # No hand was dealt: an error, a redirect (the router's
            # trailing-slash 307 — the follow-up request is metered), or an
            # implicit-deal candidate that ended up not dealing (e.g. 409).
            return status >= 300 or (
                implicit_ts is not None and int(implicit_ts.hand_no) == implicit_hand_no
            )

        async def metered_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                status_box[0] = int(message.get("status", 0))
                if not entitled and not no_deal(status_box[0]):
                    message = dict(message)
                    message["headers"] = list(message.get("headers") or []) + [
                        (b"x-free-hands-left",
                         str(max(0, FREE_HANDS_PER_DAY - used)).encode()),
                    ]
            await send(message)

        try:
            await self._call(scope, receive, metered_send, user)
        finally:
            if no_deal(status_box[0] or 500):
                await run_in_threadpool(_refund_hand, user_id)

    async def _call(self, scope: dict, receive: Callable, send: Callable,
                    user: sqlite3.Row | None) -> None:
        if user is None:
            await self._guarded(scope, receive, send)
            return
        uid = int(user["id"])
        token = _CURRENT_USER_ID.set(uid)
        rt_token = _REQUEST_RUNTIME.set([None, uid])
        try:
            await self._guarded(scope, receive, send)
        finally:
            _REQUEST_RUNTIME.reset(rt_token)
            _CURRENT_USER_ID.reset(token)

    async def _guarded(self, scope: dict, receive: Callable, send: Callable) -> None:
        started = [False]

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                started[0] = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except _BodyTooLarge:
            if started[0]:
                raise
            await _deny(scope, 413, "Request body too large.")(scope, receive, send)


# --- One app's state (BE-007) ----------------------------------------------------


def _reset_state() -> None:
    """Fresh per-app state: the user, Stripe and activity caches, the request
    budgets and work gates (built from the settings in force), and no per-user
    runtimes yet — ``install`` builds them, and opens the app's database. Kept
    across apps: the locks, ``ACCOUNT_HOOKS`` (code registration — the home games
    add theirs when imported) and the home-games hooks (``homegame.install`` sets
    them for every public app)."""
    global _USER_CACHE, _USERS_GEN, _STRIPE_CLIENT_READY, _STRIPE_RETRY_AT, _ACTIVITY
    global _REGISTRY, _STATS_DIR, _ROUTE_EXISTS
    global _EMAIL_RATE, _EMAIL_IP_RATE, _HEAVY_RATE, _USER_GATES, MODEL_GATE, _INFLIGHT
    with _USER_CACHE_LOCK:
        _USER_CACHE, _USERS_GEN = {}, [0]
    with _STRIPE_RETRY_LOCK:
        _STRIPE_RETRY_AT = {}
    with _ACTIVITY_LOCK:
        _ACTIVITY = {}
    _STRIPE_CLIENT_READY = False
    _REGISTRY = _STATS_DIR = _ROUTE_EXISTS = None
    #: Email sign-in links per address / per client: at most this many per window.
    _EMAIL_RATE = RateLimiter(rate=3 / 900.0, burst=3)
    _EMAIL_IP_RATE = RateLimiter(rate=10 / 3600.0, burst=10)
    _HEAVY_RATE = RateLimiter(rate=RATE_PER_S, burst=RATE_BURST)
    _USER_GATES = KeyedGates(1, max_waiting=USER_QUEUE)
    MODEL_GATE = WorkGate(MODEL_SLOTS, max_waiting=256)
    _INFLIGHT = KeyedCounter(USER_INFLIGHT)


_reset_state()


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
    system_info: Callable[[], dict[str, Any]] | None = None,
    model_admin: Callable[[str, str], dict[str, Any]] | None = None,
) -> None:
    """Wire auth/billing/admin into the public app. Called by
    ``server.create_app``.

    The app gets the service as the environment configures it NOW and state of
    its own (BE-007): the settings are read again, the caches and budgets start
    empty, and its database is opened (and migrated) here — importing this
    module opens none. The module serves one app at a time: installing into a
    new app retires the previous one's state (the app factory closes that app).

    ``system_info()`` (server-side facts: served models, build, health) feeds
    the admin System panel; ``model_admin(action, format)`` reloads /
    promotes / rolls back a format's checkpoint (OPS-027)."""
    global _REGISTRY, _STATS_DIR, _ROUTE_EXISTS

    _read_settings()
    _check_stripe_key_is_safe()
    _reset_state()
    _open_database()

    stats_dir = DB_PATH.parent / "trainer_stats"
    _STATS_DIR = stats_dir
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

    def _route_exists(scope: dict) -> bool:
        probe = {**scope, "type": "http"}
        for route in app.router.routes:
            try:
                match, _ = route.matches(probe)
            except Exception:  # noqa: BLE001
                continue
            if match != Match.NONE:
                return True
        return False

    _ROUTE_EXISTS = _route_exists

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

    def _sign_in(request: Request, user: sqlite3.Row) -> None:
        """Bind the session cookie to the account AND its session version."""
        request.session.clear()
        request.session["uid"] = int(user["id"])
        request.session["sv"] = int(user["session_version"] or 0)

    def _require_user(request: Request) -> sqlite3.Row:
        uid = request.session.get("uid")
        user = _user_by_id(int(uid)) if uid is not None else None
        if not _user_live(user):
            raise HTTPException(status_code=401, detail="auth required")
        return user

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
                    ),
                    "code": "unavailable",
                },
                status_code=503,
            )
        redirect_uri = f"{BASE_URL}/auth/callback"
        nxt = _safe_next(request.query_params.get("next"))
        if nxt:
            request.session["next"] = nxt  # back to the table link after sign-in
        else:
            request.session.pop("next", None)
        # (BE-021) Always offer the account chooser: someone with several
        # Google accounts can pick (or switch after signing out).
        return await oauth.google.authorize_redirect(
            request, redirect_uri, prompt="select_account"
        )

    @app.get("/auth/callback")
    async def auth_callback(request: Request):
        if oauth is None:
            raise HTTPException(status_code=503, detail="OAuth not configured")
        try:
            token = await oauth.google.authorize_access_token(request)
        except Exception as e:  # noqa: BLE001 — OAuth dance failures → login page
            logger.warning("oauth callback failed: %s", e)
            # The landing words each case (site ACC-025): Cancel on Google's
            # page, a stale tab (state no longer matches), anything else.
            return RedirectResponse(url=f"/?login=failed&why={_login_failure_reason(request, e)}")
        info = token.get("userinfo") or {}
        email = _verified_email(info)
        if not email:
            return RedirectResponse(url="/?login=failed&why=unverified")
        try:
            user = await run_in_threadpool(
                _upsert_user,
                info.get("sub"), email, info.get("name") or "", info.get("picture") or "",
            )
        except SignInConflict:
            return RedirectResponse(url="/?login=failed&why=conflict")
        if not _user_live(user):
            return RedirectResponse(url="/?login=failed&why=disabled")
        nxt = _safe_next(request.session.get("next"))
        _sign_in(request, user)
        return RedirectResponse(url=nxt or "/")

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
            if not _user_live(user):
                raise HTTPException(status_code=403, detail="account disabled")
            _sign_in(request, user)
            return RedirectResponse(url=_safe_next(next) or "/")

    # (review 2026-09-20 F7) Logout is state-changing, so POST is the real
    # verb. GET stays as app.js's fallback (it navigates there when the POST
    # fails), but only a same-site navigation signs out: another site can no
    # longer log a visitor out with a link or an <img> (BE-013).
    @app.api_route("/auth/logout", methods=["GET", "POST"])
    def auth_logout(request: Request):
        if request.method == "POST" or _same_site_request(request.headers):
            request.session.clear()
        # 303: a POST must be followed by a GET of "/".
        return RedirectResponse(url="/", status_code=303)

    # --- Email sign-in (ACC-008) -------------------------------------------------

    def _email_form(message: str = "", status: int = 200, nxt: str = "") -> HTMLResponse:
        note = f'<p class="note">{_html.escape(message)}</p>' if message else ""
        nxt_input = (
            f'<input type="hidden" name="next" value="{_html.escape(nxt)}">' if nxt else ""
        )
        return _page(
            "Sign in with email",
            note
            + "<p>We'll email you a one-time sign-in link.</p>"
            '<form method="post" action="/auth/email">'
            '<input name="email" type="email" required autocomplete="email"'
            ' aria-label="Email address" placeholder="you@example.com" class="field">'
            f"{nxt_input}"
            '<button class="btn primary wide" type="submit">Email me a link</button></form>'
            '<p class="small"><a href="/">Back</a></p>',
            status,
        )

    @app.get("/auth/email")
    def auth_email_form(next: str = ""):  # noqa: A002
        if not EMAIL_LOGIN:
            return _page(
                "Email sign-in isn't available",
                '<p>Use Google to sign in.</p><a class="btn primary" href="/">Back</a>',
                404,
            )
        return _email_form(nxt=_safe_next(next) or "")

    @app.post("/auth/email")
    async def auth_email_request(request: Request):
        if not EMAIL_LOGIN:
            raise HTTPException(status_code=404, detail="email sign-in is not configured")
        ctype = request.headers.get("content-type", "")
        browser = "application/json" not in ctype
        if browser:
            form = await request.form()
            body = {k: form.get(k) for k in ("email", "next")}
        else:
            body = await request.json()
            body = body if isinstance(body, dict) else {}
        email = str(body.get("email") or "").strip().lower()
        nxt = _safe_next(body.get("next")) or ""
        if not _EMAIL_RE.match(email):
            if browser:
                return _email_form("That doesn't look like an email address.", 400, nxt)
            raise HTTPException(status_code=400, detail="invalid email")
        client = request.client.host if request.client else "?"
        ok_addr, _ = _EMAIL_RATE.allow(email)
        ok_ip, _ = _EMAIL_IP_RATE.allow(client)
        if ok_addr and ok_ip:
            token = secrets.token_urlsafe(32)
            now = datetime.now(timezone.utc)
            await run_in_threadpool(
                DB.q,
                "INSERT INTO email_tokens(token_hash,email,next,created_at,expires_at)"
                " VALUES(?,?,?,?,?)",
                (_token_hash(token), email, nxt or None, _now(),
                 (now + EMAIL_TOKEN_TTL).isoformat(timespec="seconds")),
            )
            link = f"{BASE_URL}/auth/email/verify?token={quote(token)}"
            try:
                await run_in_threadpool(_send_login_email, email, link)
            except Exception as e:  # noqa: BLE001
                logger.warning("sign-in email to %s failed: %s", email, e)
                if browser:
                    return _email_form(
                        "We couldn't send the email just now — try again shortly.", 502, nxt
                    )
                raise HTTPException(status_code=502, detail="could not send the email")
        else:
            logger.info("email sign-in rate-limited (%s / %s)", email, client)
        # Same answer either way: no address enumeration, no rate-limit oracle.
        if browser:
            return _page(
                "Check your email",
                f"<p>If <b>{_html.escape(email)}</b> can receive mail, a sign-in link is on "
                "its way. It works once, for 15 minutes.</p>"
                '<a class="btn" href="/">Back</a>',
            )
        return {"sent": True}

    @app.get("/auth/email/verify")
    def auth_email_confirm(token: str = ""):
        """The emailed link lands on a confirmation button (POST), because
        mail scanners open links: a GET that signed in would burn the link
        before the person clicked it."""
        if not EMAIL_LOGIN:
            raise HTTPException(status_code=404, detail="email sign-in is not configured")
        return _page(
            "Sign in to WrapGTO",
            '<form method="post" action="/auth/email/verify">'
            f'<input type="hidden" name="token" value="{_html.escape(token)}">'
            '<button class="btn primary wide" type="submit">Continue</button></form>',
        )

    @app.post("/auth/email/verify")
    async def auth_email_verify(request: Request):
        if not EMAIL_LOGIN:
            raise HTTPException(status_code=404, detail="email sign-in is not configured")
        form = await request.form()
        token = str(form.get("token") or "")

        def consume() -> tuple[str, str | None] | None:
            with DB.transaction():
                row = DB.one(
                    "SELECT * FROM email_tokens WHERE token_hash=?", (_token_hash(token),)
                )
                if row is None or row["used_at"]:
                    return None
                exp = _parse_iso(row["expires_at"])
                if exp is None or datetime.now(timezone.utc) > exp:
                    return None
                DB.q("UPDATE email_tokens SET used_at=? WHERE token_hash=?",
                     (_now(), row["token_hash"]))
                return row["email"], row["next"]

        got = await run_in_threadpool(consume) if token else None
        if got is None:
            return _page(
                "This link has expired",
                "<p>Sign-in links work once, for 15 minutes.</p>"
                '<a class="btn primary" href="/auth/email">Send a new link</a>',
                400,
            )
        email, nxt = got
        user = await run_in_threadpool(_upsert_user, None, email, email.split("@")[0], "")
        if not _user_live(user):
            return _page("This account is disabled", '<a class="btn" href="/">Back</a>', 403)
        _sign_in(request, user)
        return RedirectResponse(url=_safe_next(nxt) or "/", status_code=303)

    @app.get("/me")
    def me(request: Request):
        uid = request.session.get("uid")
        user = _user_by_id(int(uid)) if uid is not None else None
        notice = maintenance_notice()
        signed_in = _user_live(user) and int(request.session.get("sv", 0) or 0) == int(
            user["session_version"] or 0
        )
        if not signed_in:
            out: dict[str, Any] = {
                "signed_in": False,
                "auth_configured": oauth is not None,
                # (review 2026-09-20 F4) Only a request that could actually
                # USE the dev login learns that it exists.
                "dev_login": _dev_login_request_ok(request),
                "email_login": EMAIL_LOGIN,
            }
            if notice:
                out["maintenance"] = notice
            return out
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
            # Self-service account endpoints (ACC-009).
            "account": {
                "export": "/account/export",
                "delete": "/account/delete",
                "signout_everywhere": "/account/signout_everywhere",
            },
        }
        if notice:
            payload["maintenance"] = notice
        # Only present when granted so a /me dump from a normal subscriber
        # does not advertise that a private games page exists. The href +
        # label ride along so the frontend can build the tab WITHOUT shipping
        # those literals to every visitor (review 2026-09-20 F1).
        if _homegame_access(user):
            payload["homegame"] = {"href": "/games", "label": "Home games"}
        return payload

    # --- Account self-service (ACC-009 / SEC-017 / SEC-016) --------------------------------

    def _download(data: dict[str, Any], uid: int) -> Response:
        body = json.dumps(data, indent=2, default=str)
        return Response(
            body, media_type="application/json",
            headers={
                **_NO_STORE,
                "Content-Disposition": f'attachment; filename="wrapgto-account-{int(uid)}.json"',
            },
        )

    @app.get("/account/export")
    def account_export(request: Request):
        user = _require_user(request)
        return _download(export_user(int(user["id"])), int(user["id"]))

    @app.post("/account/delete")
    def account_delete(request: Request, body: dict):
        """Delete the signed-in account. The body must repeat the account's
        email (``{"confirm": "<email>"}``) — a stray click can't do this."""
        user = _require_user(request)
        if str(body.get("confirm") or "").strip().lower() != user["email"].lower():
            raise HTTPException(status_code=400, detail="type your email address to confirm")
        delete_user(int(user["id"]))
        request.session.clear()
        return {"deleted": True}

    @app.post("/account/signout_everywhere")
    def account_signout_everywhere(request: Request):
        """End every other session of this account; this browser stays in."""
        user = _require_user(request)
        DB.q("UPDATE users SET session_version=session_version+1 WHERE id=?", (user["id"],))
        request.session["sv"] = int(_user_by_id(int(user["id"]))["session_version"])
        return {"ok": True}

    # --- Billing (Stripe) -----------------------------------------------------

    _price_lock = threading.Lock()

    def _price_id() -> str:
        """The subscription price. (BE-024) With a LIVE key it must be
        configured (STRIPE_PRICE_ID) — prices are never invented in live
        mode. Test mode creates one once, under a lock, and remembers it."""
        if STRIPE_PRICE_ID:
            return STRIPE_PRICE_ID
        if _stripe_live_key():
            raise HTTPException(
                status_code=503,
                detail="Billing is misconfigured (STRIPE_PRICE_ID is required with a live key).",
            )
        with _price_lock:
            cached = DB.kv_get("stripe_price_id")
            if cached:
                return cached
            s = _stripe()
            product = s.Product.create(name=STRIPE_PRODUCT_NAME)
            price = s.Price.create(
                product=product["id"],
                unit_amount=PRICE_CENTS,
                currency="usd",
                recurring={"interval": "month"},
            )
            DB.kv_set("stripe_price_id", price["id"])
            return price["id"]

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
        # (BE-023) A double click must not start a second subscription.
        if user["sub_status"] == "active" and user["sub_source"] in ("stripe", "comp"):
            raise HTTPException(status_code=409, detail="You already have an active subscription.")
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
        duplicate = await run_in_threadpool(_handle_stripe_event, event)
        return {"received": True, "duplicate": bool(duplicate)}

    def _apply_subscription(row: sqlite3.Row, sub: Any, sub_id: str) -> None:
        active = _sv(sub, "status") in _STRIPE_ACTIVE
        DB.q(
            "UPDATE users SET sub_status=?, sub_source='stripe', stripe_subscription_id=?,"
            " current_period_end=COALESCE(?, current_period_end), sub_checked_at=? WHERE id=?",
            ("active" if active else "none", sub_id, _sub_period_end(sub), _now(), row["id"]),
        )

    def _handle_stripe_event(event: Any) -> bool:
        """Apply one webhook event; True when it was a duplicate delivery.

        (BE-027) Stripe neither orders nor de-duplicates deliveries, so:
        processed event ids are remembered (a redelivery is acknowledged and
        skipped); a subscription update applies the subscription's CURRENT
        state, fetched from Stripe, never the possibly-stale status inside
        an out-of-order event; a deletion is final; a NEW subscription of a
        known customer is linked to the account."""
        event_id = _sv(event, "id")
        if event_id and DB.one("SELECT 1 FROM stripe_events WHERE id=?", (str(event_id),)):
            return True
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
        elif etype == "customer.subscription.deleted":
            sub_id = _obj_id(_sv(obj, "id"))
            row = DB.one("SELECT * FROM users WHERE stripe_subscription_id=?", (sub_id,))
            if row is not None:
                DB.q(
                    "UPDATE users SET sub_status='none', current_period_end=COALESCE(?,"
                    " current_period_end), sub_checked_at=? WHERE id=?",
                    (_sub_period_end(obj), _now(), row["id"]),
                )
        elif etype in ("customer.subscription.updated", "customer.subscription.created"):
            sub_id = _obj_id(_sv(obj, "id"))
            row = DB.one("SELECT * FROM users WHERE stripe_subscription_id=?", (sub_id,))
            if row is None:
                cust = _obj_id(_sv(obj, "customer"))
                row = (
                    DB.one("SELECT * FROM users WHERE stripe_customer_id=?", (cust,))
                    if cust else None
                )
            if row is not None and sub_id:
                try:
                    current = _stripe().Subscription.retrieve(sub_id)
                except Exception as e:  # noqa: BLE001
                    if not _stripe_missing(e):
                        raise  # 500 -> Stripe redelivers later
                    current = {"status": "canceled"}
                # A new subscription is linked only when it is live; an old
                # one ending never overwrites a newer live one.
                if (row["stripe_subscription_id"] in (None, "", sub_id)
                        or _sv(current, "status") in _STRIPE_ACTIVE):
                    _apply_subscription(row, current, sub_id)
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
        if event_id:
            DB.q(
                "INSERT OR IGNORE INTO stripe_events(id,type,received_at) VALUES(?,?,?)",
                (str(event_id), str(etype), _now()),
            )
        return False

    # --- Admin -----------------------------------------------------------------

    @app.get("/admin")
    def admin_page():
        page = static_dir / "admin.html"
        html = page.read_text(encoding="utf-8")
        # The self-hosted fonts once static/fonts/ has them (site FE-024).
        import sys

        site = sys.modules.get("plo5bp.ui.server")
        pages = getattr(site, "_PAGES", None)
        if pages is not None and hasattr(site, "_font_links"):
            html = site._font_links(html, site._font_versions(pages.versions))
        return HTMLResponse(html, headers=_NO_STORE)

    @app.get("/admin/admin.js")
    def admin_script():
        """The admin page's script, served only under the admin-gated prefix
        (the page's CSP allows scripts from this origin, never inline)."""
        page = static_dir / "admin.js"
        if not page.exists():
            raise HTTPException(status_code=404)
        return Response(
            page.read_text(encoding="utf-8"),
            media_type="text/javascript; charset=utf-8",
            headers=_NO_STORE,
        )

    @app.get("/admin/api/users")
    def admin_users():
        # A fixed number of queries however many users there are (PERF-010): the
        # usage columns from ONE aggregate, the home-games column from one lookup.
        rows = DB.q(
            """
            SELECT u.*, COALESCE(g.today, 0) AS hands_today, COALESCE(g.total, 0) AS hands_total
            FROM users u
            LEFT JOIN (
                SELECT user_id, SUM(hands) AS total,
                       SUM(CASE WHEN day=? THEN hands ELSE 0 END) AS today
                FROM usage GROUP BY user_id
            ) g ON g.user_id = u.id
            WHERE u.deleted_at IS NULL ORDER BY u.created_at DESC
            """,
            (_today(),),
        )
        active = set(_active_uids())
        in_games = _games_members()
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
                    "is_admin": _is_admin(r),
                    "hands_today": r["hands_today"],
                    "hands_total": r["hands_total"],
                    "period_end": r["current_period_end"],
                    "disabled": bool(r["disabled"]),
                    "active_now": int(r["id"]) in active,
                    "google_linked": bool(r["google_sub"]),
                    # (in the main club — see _GAMES_MEMBERS_HOOK)
                    "homegame_access": in_games(int(r["id"])),
                }
                for r in rows
            ]
        }

    @app.post("/admin/api/grant")
    def admin_grant(body: dict):
        uid = body_int(body, "user_id")
        action = body.get("action", "")
        user = _user_by_id(uid)
        if user is None or user["deleted_at"]:
            raise HTTPException(status_code=404, detail="no such user")
        if action == "grant":
            # (BE-023) A comp never overwrites a live Stripe subscription.
            if user["sub_status"] == "active" and user["sub_source"] == "stripe":
                raise HTTPException(
                    status_code=409,
                    detail="This user pays through Stripe — no comp needed.",
                )
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
        _audit(f"comp_{action}", uid)
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
        if user is None or user["deleted_at"]:
            raise HTTPException(status_code=404, detail="no such user")
        if action not in ("grant", "revoke"):
            raise HTTPException(status_code=400, detail="action must be grant|revoke")
        admin_uid = _CURRENT_USER_ID.get()
        member = False
        if _GAMES_ACCESS_HOOK is not None:
            member = bool(_GAMES_ACCESS_HOOK(
                int(admin_uid) if admin_uid is not None else uid, uid, action == "grant"
            ))
        DB.q("UPDATE users SET homegame_access=? WHERE id=?", (1 if action == "grant" else 0, uid))
        _audit(f"main_club_{action}", uid)
        return {"ok": True, "user_id": uid, "action": action, "homegame_access": member}

    @app.post("/admin/api/users/action")
    def admin_user_action(body: dict):
        """(SEC-016) disable | enable | signout ("sign out everywhere")."""
        uid = body_int(body, "user_id")
        action = str(body.get("action", ""))
        user = _user_by_id(uid)
        if user is None or user["deleted_at"]:
            raise HTTPException(status_code=404, detail="no such user")
        if action in ("disable", "signout") and _is_admin(user):
            raise HTTPException(
                status_code=409, detail="Admins can't be disabled or signed out here."
            )
        if action == "disable":
            DB.q(
                "UPDATE users SET disabled=1, session_version=session_version+1 WHERE id=?",
                (uid,),
            )
            if _REGISTRY is not None:
                _REGISTRY.drop(uid)
        elif action == "enable":
            DB.q("UPDATE users SET disabled=0 WHERE id=?", (uid,))
        elif action == "signout":
            DB.q("UPDATE users SET session_version=session_version+1 WHERE id=?", (uid,))
        else:
            raise HTTPException(
                status_code=400, detail="action must be disable|enable|signout"
            )
        _audit(f"user_{action}", uid)
        row = _user_by_id(uid)
        return {"ok": True, "user_id": uid, "action": action, "disabled": bool(row["disabled"])}

    @app.get("/admin/api/users/{uid}/export")
    def admin_user_export(uid: int):
        data = export_user(int(uid))
        _audit("user_export", int(uid))
        return _download(data, int(uid))

    @app.post("/admin/api/users/delete")
    def admin_user_delete(body: dict):
        uid = body_int(body, "user_id")
        user = _user_by_id(uid)
        if user is None or user["deleted_at"]:
            raise HTTPException(status_code=404, detail="no such user")
        if _is_admin(user):
            raise HTTPException(status_code=409, detail="Admin accounts can't be deleted here.")
        if str(body.get("confirm") or "").strip().lower() != user["email"].lower():
            raise HTTPException(status_code=400, detail="confirm with the user's email address")
        delete_user(uid)
        _audit("user_delete", uid, email=user["email"])
        return {"ok": True, "user_id": uid}

    @app.get("/admin/api/audit")
    def admin_audit(limit: int = 50):
        limit = max(1, min(int(limit), 500))
        rows = DB.q(
            "SELECT a.*, adm.email admin_email, tgt.email target_email FROM admin_audit a"
            " LEFT JOIN users adm ON adm.id=a.admin_user_id"
            " LEFT JOIN users tgt ON tgt.id=a.target_user_id"
            " ORDER BY a.id DESC LIMIT ?",
            (limit,),
        )
        out = []
        for r in rows:
            try:
                detail = json.loads(r["detail"] or "{}")
            except ValueError:
                detail = {}
            out.append({
                "at": r["at"], "action": r["action"],
                "admin": r["admin_email"], "target": r["target_email"],
                "target_user_id": r["target_user_id"], "detail": detail,
            })
        return {"entries": out}

    @app.get("/admin/api/metrics")
    def admin_metrics():
        users_n = DB.one("SELECT COUNT(*) c FROM users WHERE deleted_at IS NULL")["c"]
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
            # (BE-027) An ESTIMATE: active subscriptions × list price; promo
            # codes and discounts are not taken into account.
            "mrr_cents": stripe_active * PRICE_CENTS,
            "mrr_is_estimate": True,
            "revenue_cents_total": revenue,
            "price_cents": PRICE_CENTS,
            "free_hands_per_day": FREE_HANDS_PER_DAY,
            # (site ACC-031) the overview says when the paywall is off.
            "free_for_all": FREE_FOR_ALL,
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

    @app.get("/admin/api/system")
    def admin_system():
        """(FEAT-024) What the site is serving and whether it is healthy."""
        info: dict[str, Any] = dict(system_info()) if system_info is not None else {}
        me_row = _user_by_id(int(_CURRENT_USER_ID.get() or 0))
        try:
            db_bytes = DB.path.stat().st_size
        except OSError:
            db_bytes = None
        info.update({
            "runtimes": {"in_memory": len(_REGISTRY) if _REGISTRY else 0,
                         "capacity": MAX_USER_RUNTIMES, "idle_evict_s": RUNTIME_IDLE_S},
            "active_users": len(_active_uids()),
            "http": mw.METRICS.snapshot(),
            "work_gate": {"slots": MODEL_GATE.slots, "busy": MODEL_GATE.busy,
                          "waiting": MODEL_GATE.waiting},
            "limits": {"rate_per_s": RATE_PER_S, "burst": RATE_BURST,
                       "max_body_bytes": MAX_BODY_BYTES, "user_queue": USER_QUEUE},
            "db": {"path": DB.path.name, "bytes": db_bytes, "schema": DB.schema_versions()},
            "config": {
                "free_for_all": FREE_FOR_ALL,
                "google_sign_in": oauth is not None,
                "email_sign_in": EMAIL_LOGIN,
                "stripe": (
                    ("live" if _stripe_live_key() else "test") if STRIPE_SECRET_KEY else "off"
                ),
                "session_key": (
                    "env" if os.environ.get("PLO5BP_SESSION_SECRET") else "database"
                ),
                "admin_pinned_to_google_id": bool(ADMIN_SUBS),
            },
            "maintenance": maintenance_notice(),
            "you": {"email": me_row["email"] if me_row else None,
                    "google_id": me_row["google_sub"] if me_row else None},
        })
        return info

    @app.post("/admin/api/maintenance")
    def admin_maintenance(body: dict):
        """(FEAT-028) Announce (or clear) a restart: ``{"message": str,
        "in_minutes": int | None}`` or ``{"clear": true}``."""
        if body.get("clear"):
            DB.kv_delete("maintenance")
            _audit("maintenance_clear")
            return {"maintenance": None}
        message = str(body.get("message") or "").strip()[:280]
        if not message:
            raise HTTPException(status_code=400, detail="message required")
        at = None
        if body.get("in_minutes") is not None:
            minutes = body_int(body, "in_minutes", limit=7 * 24 * 60)
            at = (datetime.now(timezone.utc) + timedelta(minutes=max(0, minutes))).isoformat(
                timespec="seconds")
        DB.kv_set("maintenance", json.dumps({"message": message, "at": at, "set_at": _now()}))
        _audit("maintenance_set", message=message, at=at)
        return {"maintenance": maintenance_notice()}

    @app.post("/admin/api/models")
    def admin_models(body: dict):
        """(OPS-027) ``{"action": "reload"|"promote"|"rollback", "format": id}``:
        reload the format's checkpoint from disk, promote ``<file>.new`` over it
        (the old file is kept as ``<file>.prev``), or put ``.prev`` back — each
        verified with a test forward BEFORE the live model is swapped, and with
        no restart: nobody's Study spot or home-game hand is lost."""
        if model_admin is None:
            raise HTTPException(status_code=501, detail="model management unavailable")
        action = str(body.get("action") or "")
        fmt = str(body.get("format") or "")
        if action not in ("reload", "promote", "rollback"):
            raise HTTPException(status_code=400, detail="action must be reload|promote|rollback")
        try:
            result = model_admin(action, fmt)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 — surfaced to the admin, not a 500
            logger.exception("model %s of %s failed", action, fmt)
            _audit(f"model_{action}_failed", format=fmt, error=str(e))
            raise HTTPException(status_code=409, detail=f"{action} failed: {e}")
        _audit(f"model_{action}", format=fmt, checkpoint=result.get("checkpoint"),
               sha256=result.get("sha256"))
        return result

    # Middleware LAST (add_middleware prepends: Session must wrap Access).
    app.add_middleware(AccessMiddleware)
    app.add_middleware(
        _RotatingSessionMiddleware,
        secret_keys=_session_keys(),
        session_cookie=SESSION_COOKIE,
        max_age=SESSION_MAX_AGE_S,
        same_site="lax",
        https_only=BASE_URL.startswith("https"),
    )

    from plo5bp.ui import homegame as _homegame

    _homegame.install(app, static_dir=static_dir)

    logger.info(
        "PUBLIC service installed: base=%s db=%s admins=%s oauth=%s email=%s stripe=%s"
        " dev_login=%s session_key=%s",
        BASE_URL,
        DB_PATH,
        ",".join(sorted(ADMIN_EMAILS)),
        "on" if oauth else "OFF",
        "log" if EMAIL_LOG_ONLY else ("on" if EMAIL_LOGIN else "OFF"),
        ("live" if _stripe_live_key() else "test") if STRIPE_SECRET_KEY else "OFF",
        DEV_LOGIN,
        "env" if os.environ.get("PLO5BP_SESSION_SECRET") else "database",
    )
