"""Public-build service layer: auth, per-user state, free tier, billing, admin.

Installed by ``server.py`` ONLY when ``PLO5BP_PUBLIC`` is truthy (the same flag
that strips the live-capture routes). The local build never imports this.

What it adds around the existing app:

- **Google sign-in** (Authlib). No passwords stored — the only identity is
  Google's verified email. A loopback-only dev login (``PLO5BP_DEV_LOGIN=1``)
  exists so the flow can be exercised before OAuth credentials exist; it
  refuses non-127.0.0.1 clients.
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

import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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
PRICE_CENTS = int(os.environ.get("PLO5BP_PRICE_CENTS", "1000"))  # $10/mo
DEV_LOGIN = os.environ.get("PLO5BP_DEV_LOGIN", "").strip().lower() in ("1", "true", "yes")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")

MAX_USER_RUNTIMES = int(os.environ.get("PLO5BP_MAX_RUNTIMES", "300"))

# Study-mode routes: subscription required (the full product). The trainer
# tree is the free-tier surface.
STUDY_PATHS = {"/state", "/cards", "/action", "/seats", "/config", "/undo", "/reset"}
# No auth at all:
OPEN_PREFIXES = ("/static/", "/auth/", "/stripe/webhook", "/health")
OPEN_EXACT = {"/", "/me", "/favicon.ico", "/terms", "/privacy"}

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
  sub_checked_at TEXT
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
    """Small thread-safe sqlite wrapper (single connection + lock)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()

    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, args)
            rows = cur.fetchall()
            self._conn.commit()
            return rows

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


def _stripe():
    import stripe as _s

    _s.api_key = STRIPE_SECRET_KEY
    return _s


def _sv(obj: Any, key: str, default: Any = None) -> Any:
    """Safe field access for Stripe objects AND plain dicts (webhooks).

    stripe-python v15 StripeObjects support ``obj[key]`` / ``obj.key`` but no
    longer inherit dict — ``.get()`` raises AttributeError. This is the one
    accessor used for every Stripe payload field below."""
    try:
        return obj[key]
    except (KeyError, IndexError, TypeError):
        return default


def _refresh_stripe_status(user: sqlite3.Row) -> sqlite3.Row:
    """Lazily re-verify a stripe-sourced sub. Called when the cached status
    could be stale (period end passed, or >24h since last check). Keeps
    laptop hosting honest without requiring a public webhook endpoint."""
    if not (STRIPE_SECRET_KEY and user["stripe_subscription_id"]):
        return user
    stale = True
    if user["sub_checked_at"]:
        try:
            checked = datetime.fromisoformat(user["sub_checked_at"])
            stale = datetime.now(timezone.utc) - checked > timedelta(hours=24)
        except ValueError:
            pass
    past_end = False
    if user["current_period_end"]:
        try:
            past_end = datetime.now(timezone.utc) > datetime.fromisoformat(
                user["current_period_end"]
            )
        except ValueError:
            past_end = True
    if not (stale or past_end):
        return user
    try:
        sub = _stripe().Subscription.retrieve(user["stripe_subscription_id"])
        active = sub["status"] in ("active", "trialing", "past_due")
        period_end = None
        try:
            items = _sv(_sv(sub, "items", {}), "data", []) or []
            end_ts = (
                _sv(items[0], "current_period_end")
                if items
                else _sv(sub, "current_period_end")
            )
            if end_ts:
                period_end = datetime.fromtimestamp(int(end_ts), tz=timezone.utc).isoformat(
                    timespec="seconds"
                )
        except Exception:  # noqa: BLE001 — period end is cosmetic
            period_end = None
        DB.q(
            "UPDATE users SET sub_status=?, sub_source='stripe', current_period_end=?,"
            " sub_checked_at=? WHERE id=?",
            ("active" if active else "none", period_end, _now(), user["id"]),
        )
    except Exception as e:  # noqa: BLE001 — keep serving on Stripe hiccups
        logger.warning("stripe status refresh failed for user %s: %s", user["id"], e)
        DB.q("UPDATE users SET sub_checked_at=? WHERE id=?", (_now(), user["id"]))
    return _user_by_id(user["id"])


def _entitled(user: sqlite3.Row | None) -> bool:
    """Full access: admin, comp grant, or active stripe subscription."""
    if user is None:
        return False
    if _is_admin(user):
        return True
    if user["sub_status"] != "active":
        return False
    if user["sub_source"] == "comp":
        return True
    user = _refresh_stripe_status(user)
    return user["sub_status"] == "active"


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


_REGISTRY: Registry | None = None


def _current_runtime() -> _Runtime | None:
    uid = _CURRENT_USER_ID.get()
    if uid is None or _REGISTRY is None:
        return None
    return _REGISTRY.get(uid)


# --- Middleware ----------------------------------------------------------------


def _wants_sub_route(path: str) -> bool:
    return path in STUDY_PATHS


def _open_route(path: str) -> bool:
    return path in OPEN_EXACT or path.startswith(OPEN_PREFIXES)


class AccessMiddleware(BaseHTTPMiddleware):
    """Auth + entitlement + free-quota enforcement for the public build."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if _open_route(path):
            return await call_next(request)

        uid = request.session.get("uid")
        user = _user_by_id(int(uid)) if uid is not None else None
        if user is None:
            return JSONResponse(
                {"detail": "auth required", "error": "auth"}, status_code=401
            )

        entitled = _entitled(user)
        admin = _is_admin(user)

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

        consumed = False
        if path == "/trainer/new_hand" and request.method == "POST":
            used = _record_hand(user["id"])  # count everyone (admin metrics)
            consumed = True
            if not entitled and used > FREE_HANDS_PER_DAY:
                _refund_hand(user["id"])
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

        token = _CURRENT_USER_ID.set(int(user["id"]))
        try:
            response = await call_next(request)
        finally:
            _CURRENT_USER_ID.reset(token)

        if consumed and response.status_code >= 400:
            _refund_hand(user["id"])
        elif consumed and not entitled:
            left = max(0, FREE_HANDS_PER_DAY - _hands_today(user["id"]))
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
                        + (" — or use the dev login." if DEV_LOGIN else ".")
                    )
                },
                status_code=503,
            )
        redirect_uri = f"{BASE_URL}/auth/callback"
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
        email = (info.get("email") or "").strip().lower()
        if not email or not info.get("email_verified", True):
            return RedirectResponse(url="/?login=failed")
        user = _upsert_user(
            info.get("sub"), email, info.get("name") or "", info.get("picture") or ""
        )
        request.session["uid"] = int(user["id"])
        return RedirectResponse(url="/")

    if DEV_LOGIN:

        @app.get("/auth/dev")
        def auth_dev(request: Request, email: str, name: str = ""):
            """Loopback-only fake sign-in for local testing without OAuth.

            NEVER expose a tunnel with PLO5BP_DEV_LOGIN=1 — anyone could sign
            in as any email, including the admin's."""
            client = request.client.host if request.client else ""
            if client not in ("127.0.0.1", "::1", "localhost", "testclient"):
                raise HTTPException(status_code=403, detail="dev login is loopback-only")
            user = _upsert_user(None, email, name or email.split("@")[0], "")
            request.session["uid"] = int(user["id"])
            return RedirectResponse(url="/")

    @app.get("/auth/logout")
    def auth_logout(request: Request):
        request.session.clear()
        return RedirectResponse(url="/")

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
                "dev_login": DEV_LOGIN,
            }
        entitled = _entitled(user)
        user = _user_by_id(user["id"])  # re-read (refresh may have written)
        used = _hands_today(user["id"])
        return {
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
        }

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

    def _activate_from_checkout(sess: Any) -> None:
        uid = int(_sv(sess, "client_reference_id") or 0)
        user = _user_by_id(uid)
        if user is None:
            logger.warning(
                "checkout for unknown user ref %r", _sv(sess, "client_reference_id")
            )
            return
        sub_id = _sv(sess, "subscription")
        cust_id = _sv(sess, "customer")
        DB.q(
            "UPDATE users SET sub_status='active', sub_source='stripe',"
            " stripe_customer_id=?, stripe_subscription_id=?, sub_checked_at=?"
            " WHERE id=?",
            (
                cust_id if isinstance(cust_id, str) else _sv(cust_id or {}, "id"),
                sub_id if isinstance(sub_id, str) else _sv(sub_id or {}, "id"),
                _now(),
                uid,
            ),
        )
        ref = _sv(sess, "payment_intent") or _sv(sess, "invoice") or _sv(sess, "id")
        amount = int(_sv(sess, "amount_total") or 0)
        if ref and amount:
            try:
                DB.q(
                    "INSERT OR IGNORE INTO payments(user_id,stripe_ref,amount_cents,"
                    "currency,created_at) VALUES(?,?,?,?,?)",
                    (uid, str(ref), amount, _sv(sess, "currency") or "usd", _now()),
                )
            except Exception:  # noqa: BLE001
                logger.exception("payment record failed")

    @app.get("/billing/confirm")
    def billing_confirm(request: Request, session_id: str):
        """Success-redirect verification — the no-webhook activation path."""
        user = _require_user(request)
        if not STRIPE_SECRET_KEY:
            raise HTTPException(status_code=503, detail="billing not configured")
        sess = _stripe().checkout.Session.retrieve(session_id)
        if str(_sv(sess, "client_reference_id")) != str(user["id"]):
            raise HTTPException(status_code=403, detail="session belongs to another user")
        if _sv(sess, "payment_status") != "paid":
            return {"active": False, "status": _sv(sess, "payment_status")}
        _activate_from_checkout(sess)
        return {"active": True}

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
        etype = event["type"]
        obj = event["data"]["object"]
        if etype == "checkout.session.completed":
            _activate_from_checkout(obj)
        elif etype in ("customer.subscription.updated", "customer.subscription.deleted"):
            row = DB.one(
                "SELECT * FROM users WHERE stripe_subscription_id=?", (obj["id"],)
            )
            if row is not None:
                active = etype != "customer.subscription.deleted" and obj["status"] in (
                    "active",
                    "trialing",
                    "past_due",
                )
                DB.q(
                    "UPDATE users SET sub_status=?, sub_checked_at=? WHERE id=?",
                    ("active" if active else "none", _now(), row["id"]),
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
        return {"received": True}

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
                }
                for r in rows
            ]
        }

    @app.post("/admin/api/grant")
    def admin_grant(body: dict):
        uid = int(body.get("user_id", 0))
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

    # Middleware LAST (add_middleware prepends: Session must wrap Access).
    app.add_middleware(AccessMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=_session_secret(),
        max_age=30 * 24 * 3600,
        same_site="lax",
        https_only=BASE_URL.startswith("https"),
    )

    logger.info(
        "PUBLIC service installed: base=%s db=%s admins=%s oauth=%s stripe=%s dev_login=%s",
        BASE_URL,
        DB_PATH,
        ",".join(sorted(ADMIN_EMAILS)),
        "on" if oauth else "OFF",
        "on" if STRIPE_SECRET_KEY else "OFF",
        DEV_LOGIN,
    )
