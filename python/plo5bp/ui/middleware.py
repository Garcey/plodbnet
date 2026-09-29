"""Site-wide HTTP plumbing shared by both builds (pure ASGI, no BaseHTTPMiddleware).

- :class:`SiteMiddleware` — the outermost app layer:
  * security headers on EVERY response (SEC-001 / SEC-013 / FE-016): nosniff,
    ``Referrer-Policy: same-origin``, ``X-Frame-Options: DENY`` and, on HTML
    pages that do not set their own, the main-site Content-Security-Policy
    (``frame-ancestors 'none'``, scripts from this origin only). HSTS when the
    deployment is https. A header a route already set always wins (the home
    games pages carry their own CSP).
  * request timing + error tracking (OPS-024): slow (>= ``SLOW_MS``) and failed
    requests are logged with method, path, status, milliseconds and user;
    per-route timings and the most recent errors feed the admin System panel.
- :func:`error_body` / :func:`error_page` — the ONE error shape (BE-026):
  API clients get ``{"detail": ..., "code": <stable id>}`` (plus ``error_id``
  on 5xx); a browser navigation gets a small branded page with a way back
  (ACC-014 / ACC-028) instead of a line of JSON.
"""

from __future__ import annotations

import base64
import hashlib
import html
import logging
import re
import secrets
import threading
import time
from collections import deque
from typing import Any, Iterable

logger = logging.getLogger("plo5bp.ui.http")

#: Requests at least this slow are logged (and counted as slow).
SLOW_MS = 500.0

#: Status -> the stable machine-readable error code every error body carries.
ERROR_CODES: dict[int, str] = {
    400: "bad_request",
    401: "auth",
    402: "payment_required",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "too_large",
    415: "unsupported_media_type",
    422: "validation",
    429: "rate_limited",
    500: "internal",
    502: "upstream",
    503: "unavailable",
    504: "timeout",
}


def error_code(status: int) -> str:
    return ERROR_CODES.get(int(status), "error" if status >= 400 else "ok")


def new_error_id() -> str:
    """Short id a user can quote; the same id is in the server log line."""
    return secrets.token_hex(4)


def error_body(status: int, detail: Any, code: str | None = None, **extra: Any) -> dict[str, Any]:
    """``{"detail": detail, "code": code, **extra}`` — detail is kept as given
    (a string for most errors; the club gate's dict and FastAPI's 422 list
    pass through untouched so existing clients keep reading them)."""
    body: dict[str, Any] = {"detail": detail, "code": code or error_code(status)}
    body.update(extra)
    return body


def wants_html(headers: Any, method: str = "GET") -> bool:
    """A browser NAVIGATION (address bar, link, reload) — not fetch()/XHR,
    whose default Accept is ``*/*`` — asks for text/html first."""
    if method not in ("GET", "HEAD"):
        return False
    accept = ""
    try:
        accept = headers.get("accept", "") or ""
    except AttributeError:
        for k, v in headers or ():
            if k.lower() in (b"accept", "accept"):
                accept = v.decode("latin-1") if isinstance(v, bytes) else v
    return "text/html" in accept.lower()


_PAGE_TEXT: dict[int, tuple[str, str]] = {
    400: ("That didn't work", "The request couldn't be understood."),
    401: ("Please sign in", "Sign in to open this page."),
    403: ("Not available", "Your account can't open this page."),
    404: ("Page not found", "The link may be old, or the address mistyped."),
    405: ("Page not found", "The link may be old, or the address mistyped."),
    413: ("Too large", "That upload is bigger than the site accepts."),
    429: ("Slow down", "Too many requests in a short time — try again in a moment."),
    500: ("Something went wrong", "An unexpected error happened on our side."),
    502: ("Temporarily unavailable", "A service we rely on didn't answer — try again shortly."),
    503: ("Temporarily unavailable", "The site is busy or restarting — try again in a moment."),
}


_PAGE_CSS = """
:root{color-scheme:dark;--bg:#070b11;--panel:#0d131b;--border:rgba(255,255,255,.08);
--text:#dbe2ec;--bright:#f4f7fa;--muted:#8a97a6;--accent:#18d2c3}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px 16px;
background:radial-gradient(120% 80% at 50% 0%,#0e1722 0%,var(--bg) 60%);color:var(--text);
font:15px/1.5 "Geist","Segoe UI Variable Text","Segoe UI",Inter,system-ui,sans-serif}
main{width:100%;max-width:420px;background:var(--panel);border:1px solid var(--border);
border-radius:16px;padding:32px 28px;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,.45)}
.mark{width:44px;height:44px;margin:0 auto 18px;display:block}
.code{font:600 12px/1 "Geist Mono",ui-monospace,monospace;letter-spacing:.12em;color:var(--muted)}
h1{margin:10px 0 8px;font-size:22px;color:var(--bright)}
p{margin:0 0 22px;color:var(--muted)}
p.small{margin:16px 0 0;font-size:13px}
p.note{color:#f0a23c}
a{color:var(--accent)}
.btn{display:inline-block;padding:10px 18px;border-radius:10px;text-decoration:none;
font:inherit;font-weight:600;border:1px solid var(--border);color:var(--text);
background:transparent;cursor:pointer}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#04201d}
.btn.wide{width:100%}
.btn:focus-visible,.field:focus-visible{outline:2px solid var(--bright);outline-offset:2px}
.field{display:block;width:100%;padding:11px 12px;margin:0 0 12px;border-radius:10px;
border:1px solid rgba(255,255,255,.15);background:#0a1018;color:var(--bright);font:inherit}
.ref{margin:18px 0 0;font-size:12px}
code{font-family:"Geist Mono",ui-monospace,monospace;color:var(--text)}
"""


def _shell(title: str, inner: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>{html.escape(title)} — WrapGTO</title>
<link rel="icon" type="image/png" sizes="48x48" href="/static/brand/favicon-48.png">
<style>{_PAGE_CSS}</style></head>
<body><main>
<img class="mark" src="/static/brand/wrap-app-icon-dark.svg" alt="WrapGTO">
{inner}
</main></body></html>"""


def plain_page(title: str, body_html: str) -> str:
    """A branded, script-free page around trusted server-built ``body_html``
    (the sign-in flow screens). ``title`` is escaped; ``body_html`` is not."""
    return _shell(title, f"<h1>{html.escape(title)}</h1>\n{body_html}")


def error_page(
    status: int,
    message: str | None = None,
    *,
    error_id: str | None = None,
    sign_in: bool = False,
    title: str | None = None,
) -> str:
    """Branded, script-free error page (inline CSS only — allowed by the CSP)."""
    head, default_msg = _PAGE_TEXT.get(int(status), ("Something went wrong", ""))
    head = title or head
    msg = message if message else default_msg
    ref = (
        f'<p class="ref">Reference: <code>{html.escape(error_id)}</code></p>'
        if error_id else ""
    )
    primary = (
        '<a class="btn primary" href="/">Sign in</a>'
        if sign_in else '<a class="btn primary" href="/">Go to WrapGTO</a>'
    )
    return _shell(head, (
        f'<div class="code">{int(status)}</div>\n'
        f"<h1>{html.escape(head)}</h1>\n"
        f"<p>{html.escape(msg)}</p>\n{primary}\n{ref}"
    ))


# --- Content-Security-Policy ------------------------------------------------------

_INLINE_SCRIPT_RE = re.compile(
    r"<script(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script\s*>", re.I | re.S
)


def inline_script_hashes(html_text: str) -> list[str]:
    """CSP source expressions ('sha256-…') for every inline <script> in a
    SERVER-GENERATED page (never for pages carrying user content: a hash
    allow-lists exactly the scripts the server itself put there)."""
    out = []
    for body in _INLINE_SCRIPT_RE.findall(html_text):
        digest = hashlib.sha256(body.encode("utf-8")).digest()
        out.append("'sha256-" + base64.b64encode(digest).decode("ascii") + "'")
    return out


def site_csp(script_hashes: Iterable[str] = ()) -> str:
    """The main site's CSP (Study, Trainer, landing, legal and admin pages)."""
    script_src = " ".join(["'self'", *script_hashes])
    return "; ".join((
        "default-src 'self'",
        f"script-src {script_src}",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com data:",
        # https: — the account chip shows the Google profile picture.
        "img-src 'self' data: https:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ))


DEFAULT_CSP = site_csp()

#: Paths whose HTML must not get the default CSP (FastAPI's interactive docs
#: load a CDN bundle; they exist only in the local build).
_CSP_EXEMPT_PREFIXES = ("/docs", "/redoc")


# --- Metrics -----------------------------------------------------------------------


class _RouteStats:
    __slots__ = ("count", "errors", "slow", "total_ms", "max_ms", "recent")

    def __init__(self) -> None:
        self.count = 0
        self.errors = 0
        self.slow = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.recent: deque[float] = deque(maxlen=200)


class Metrics:
    """Per-route latency + the most recent errors, for the admin System panel."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._routes: dict[str, _RouteStats] = {}
        self.recent_errors: deque[dict[str, Any]] = deque(maxlen=50)
        self.started_at = time.time()
        self.requests = 0

    def observe(self, key: str, ms: float, status: int) -> None:
        with self._lock:
            self.requests += 1
            st = self._routes.get(key)
            if st is None:
                if len(self._routes) >= 500:  # bounded: odd paths collapse
                    key = "(other)"
                    st = self._routes.get(key)
                if st is None:
                    st = self._routes[key] = _RouteStats()
            st.count += 1
            st.total_ms += ms
            st.max_ms = max(st.max_ms, ms)
            st.recent.append(ms)
            if status >= 500:
                st.errors += 1
            if ms >= SLOW_MS:
                st.slow += 1

    def record_error(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self.recent_errors.appendleft(entry)

    def snapshot(self, top: int = 25) -> dict[str, Any]:
        with self._lock:
            rows = []
            for key, st in self._routes.items():
                recent = sorted(st.recent)
                p50 = recent[len(recent) // 2] if recent else 0.0
                p95 = recent[min(len(recent) - 1, int(len(recent) * 0.95))] if recent else 0.0
                rows.append({
                    "route": key, "count": st.count, "errors": st.errors, "slow": st.slow,
                    "avg_ms": round(st.total_ms / st.count, 1) if st.count else 0.0,
                    "p50_ms": round(p50, 1), "p95_ms": round(p95, 1),
                    "max_ms": round(st.max_ms, 1),
                })
            rows.sort(key=lambda r: r["count"] * r["avg_ms"], reverse=True)
            return {
                "requests": self.requests,
                "routes": rows[:top],
                "recent_errors": list(self.recent_errors)[:20],
            }


METRICS = Metrics()

#: Worker health checks shown by /health (``threads``) and the admin System
#: panel: name -> fn() -> {"ok": bool, ...}. Any module may register one
#: (the home-games clock and grader, OPS-011) — a False ``ok`` makes the site
#: "degraded" without failing it. Each app has its own (``server.Site``): this
#: name is the current app's registry (``use_health_checks``).
HEALTH_CHECKS: dict[str, Any] = {}


def register_health_check(name: str, fn: Any) -> None:
    HEALTH_CHECKS[str(name)] = fn


def use_health_checks(checks: dict[str, Any]) -> dict[str, Any]:
    """Make ``checks`` the registry ``register_health_check`` fills (the app
    factory switches it with the current app — BE-007); returns the old one."""
    global HEALTH_CHECKS
    old, HEALTH_CHECKS = HEALTH_CHECKS, checks
    return old


def _route_key(scope: dict[str, Any]) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None) or getattr(route, "path_format", None)
    if isinstance(path, str) and path:
        return f"{scope.get('method', '')} {path}"
    if scope.get("path", "").startswith("/static/"):
        return f"{scope.get('method', '')} /static/*"
    return f"{scope.get('method', '')} (unmatched)"


def request_user(scope: dict[str, Any]) -> Any:
    """The signed-in user id the access layer recorded for this request."""
    state = scope.get("state")
    if isinstance(state, dict):
        return state.get("uid")
    return getattr(state, "uid", None)


class SiteMiddleware:
    """Outermost ASGI layer: security headers + timing/error tracking."""

    def __init__(self, app: Any, *, hsts: bool = False, metrics: Metrics = METRICS):
        self.app = app
        self.hsts = bool(hsts)
        self.metrics = metrics

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        t0 = time.perf_counter()
        status_box = [0]
        path = scope.get("path", "")
        csp_exempt = path.startswith(_CSP_EXEMPT_PREFIXES)

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_box[0] = int(message.get("status", 0))
                headers = list(message.get("headers") or [])
                present = {k.lower() for k, _ in headers}
                add = []
                if b"x-content-type-options" not in present:
                    add.append((b"x-content-type-options", b"nosniff"))
                if b"referrer-policy" not in present:
                    add.append((b"referrer-policy", b"same-origin"))
                if b"x-frame-options" not in present and not csp_exempt:
                    add.append((b"x-frame-options", b"DENY"))
                if self.hsts and b"strict-transport-security" not in present:
                    add.append((b"strict-transport-security", b"max-age=31536000"))
                if b"content-security-policy" not in present and not csp_exempt:
                    ctype = b""
                    for k, v in headers:
                        if k.lower() == b"content-type":
                            ctype = v.lower()
                            break
                    if ctype.startswith(b"text/html"):
                        add.append((b"content-security-policy", DEFAULT_CSP.encode()))
                if add:
                    message = dict(message)
                    message["headers"] = headers + add
            await send(message)

        failed: BaseException | None = None
        try:
            await self.app(scope, receive, send_wrapper)
        except BaseException as e:  # noqa: BLE001 — observed, then re-raised
            failed = e
            raise
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            status = status_box[0] or (500 if failed is not None else 0)
            key = _route_key(scope)
            self.metrics.observe(key, ms, status)
            if status >= 500 or ms >= SLOW_MS:
                uid = request_user(scope)
                (logger.warning if status >= 500 else logger.info)(
                    "%s %s -> %s in %.0f ms (user %s)%s",
                    scope.get("method"), path, status or "?", ms,
                    uid if uid is not None else "-",
                    " [slow]" if ms >= SLOW_MS else "",
                )
