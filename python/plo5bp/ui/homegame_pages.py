"""The pages: the client's build (versioned file addresses), the page headers,
and the small script-free pages someone who is not signed in gets.

Split out of ``homegame`` (HGB-006) and imported by it: every name defined here is
re-exported as ``homegame.<name>``. The code reaches every other home-games name
through ``hg`` (the ``homegame`` module), looked up when it runs, so patching
``homegame.X`` in a test reaches this module too and ``homegame.use_context`` swaps
its state. Patch ``homegame.X``, never this module's copy.
"""

from __future__ import annotations

import hashlib
import html as html_mod
import importlib
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse, Response

from plo5bp.ui import public as pub

#: The home games' main module: every home-games name is looked up there when used.
hg = sys.modules.get("plo5bp.ui.homegame") or importlib.import_module("plo5bp.ui.homegame")
logger = logging.getLogger("plo5bp.ui.homegame")

__all__ = (
    "INVITE_HEADERS", "PAGE_CSP", "PAGE_HEADERS", "_ASSET_REF", "_BUILD_RECHECK_S",
    "_INVITE_CSS", "_SIGNIN_PATH", "_asset_hash", "_asset_hashes", "_build_cache",
    "_games_phrase", "_invite_response", "_page_html", "_public_name", "_signin_html",
    "client_build",
)


# --- the client's build (2026-09-28: FE-012 / OPS-039) ------------------------------
# The page links every client file by a versioned address (games.js?v=<its content
# hash>) that the browser keeps for a year: coming back to a table loads ~450 KB from
# the device instead of the network, and a deploy still lands at once (new content =
# new address; the page itself is never cached). `private`: a shared cache
# (Cloudflare) must never hand the gated files to anyone. The BUILD id (a hash over
# the page and every client file) rides in the lobby and table views, and the page
# carries the build it was served with (<meta name="hg-build">): a table still running
# an older page offers a refresh between hands (games.js / games.ui.js).
_BUILD_RECHECK_S = 5.0
_asset_hashes: dict[str, tuple[tuple[int, int], str]] = {}
_build_cache: dict[str, Any] = {"at": -1e9, "id": "", "dir": None}
_ASSET_REF = re.compile(r'(/games/static/)([\w.]+\.(?:js|css))"')


def _asset_hash(static_dir: Path, name: str) -> str:
    """12 hex digits of the file's SHA-256 ("" when it is missing); cached per mtime+size."""
    path = static_dir / name
    try:
        st = path.stat()
    except OSError:
        return ""
    key = (int(st.st_mtime_ns), int(st.st_size))
    hit = hg._asset_hashes.get(name)
    if hit is None or hit[0] != key:
        hit = (key, hashlib.sha256(path.read_bytes()).hexdigest()[:12])
        hg._asset_hashes[name] = hit
    return hit[1]


def client_build(static_dir: Path | None = None, *, fresh: bool = False) -> str:
    """The client build id ("" before `install`), re-read at most every few seconds."""
    d = static_dir or hg._build_cache["dir"]
    if d is None:
        return ""
    now = time.monotonic()
    if fresh or now - hg._build_cache["at"] >= hg._BUILD_RECHECK_S or hg._build_cache["dir"] != d:
        parts = [f"{n}:{hg._asset_hash(d, n)}" for n in sorted(hg.GAMES_ASSETS)]
        parts.append(f"games.html:{hg._asset_hash(d, 'games.html')}")
        hg._build_cache.update(at=now, dir=d, id=hashlib.sha256("|".join(parts).encode()).hexdigest()[:12])
    return hg._build_cache["id"]


def _page_html(static_dir: Path) -> str:
    """games.html with versioned client addresses and the build they belong to."""
    page = (static_dir / "games.html").read_text(encoding="utf-8")
    page = hg._ASSET_REF.sub(
        lambda m: f'{m.group(1)}{m.group(2)}?v={hg._asset_hash(static_dir, m.group(2))}"'
        if m.group(2) in hg.GAMES_ASSETS else m.group(0),
        page,
    )
    return page.replace(
        '<meta name="hg-build" content="" />',
        f'<meta name="hg-build" content="{hg.client_build(static_dir, fresh=True)}" />', 1,
    )


# Security headers for the home-games page (the lobby and every table). Scripts
# run ONLY from this site: no inline <script>, no inline event-handler attribute
# (onclick=...), no javascript: URL — a script injected through a name or a chat
# line cannot run. Styles/fonts come from this site and Google Fonts — and no inline
# style either (FE-011, 2026-09-28: the client puts every value its CSS needs through
# the CSSOM, games.js put / data-vars); images from this site or data: URIs (so
# injected CSS cannot send anything elsewhere either); no other site may frame the
# page. A new outside resource (a CDN, an image host) needs its origin added here;
# test_homegame_page_headers.py scans the client for inline handlers and styles the
# policy would silently block.
PAGE_CSP = "; ".join((
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
))
PAGE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-store, must-revalidate",
    "Content-Security-Policy": PAGE_CSP,
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    # allow-popups: "Open in Study" fills the new tab it opens before sending it on
    "Cross-Origin-Opener-Policy": "same-origin-allow-popups",
}


# --- pages for someone who is not signed in ----------------------------------------
#
# Home games need an account: the lobby, a table link or a club invite link opened
# signed out answers with a small page (no scripts) naming what it is and a
# "Sign in with Google" button that comes straight back to the same link.

_SIGNIN_PATH = re.compile(r"^/games(?:/t/([A-Za-z0-9_-]{1,40})|/join/([A-Za-z0-9_-]{6,40}))?$")
INVITE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-store, must-revalidate",
    "Content-Security-Policy": "; ".join((
        "default-src 'none'",
        "style-src 'unsafe-inline' https://fonts.googleapis.com",
        "font-src https://fonts.gstatic.com",
        "img-src 'self' data:",
        "form-action 'self'",
        "base-uri 'none'",
        "frame-ancestors 'none'",
    )),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}
_INVITE_CSS = """
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px 16px;
background:radial-gradient(1200px 600px at 50% -10%,#123127 0,#070b11 60%) #070b11;color:#e8edf3;
font:15px/1.5 Geist,system-ui,-apple-system,"Segoe UI",sans-serif}
.card{width:100%;max-width:420px;background:#0f1620;border:1px solid #223041;border-radius:18px;
padding:28px 24px;box-shadow:0 20px 60px rgba(0,0,0,.45);text-align:center}
.brand{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#8794a6;margin-bottom:14px}
h1{font-size:22px;line-height:1.25;margin:0 0 10px}p{margin:0 0 18px;color:#b7c2d0}
.btn{display:inline-block;width:100%;border:0;border-radius:12px;padding:13px 16px;font:inherit;
font-weight:700;cursor:pointer;text-decoration:none;background:linear-gradient(#f3d27a,#d9a93c);color:#1a1405}
.btn:focus-visible,.links a:focus-visible{outline:2px solid #f3d27a;outline-offset:3px}
.links{margin:18px 0 0;font-size:13px;color:#8794a6;display:flex;gap:6px 14px;justify-content:center;flex-wrap:wrap}
.links a{color:#b7c2d0;text-decoration:none}.links a:hover{color:#e8edf3;text-decoration:underline}
"""


def _signin_html(head: str, text: str, next_path: str) -> str:
    """``text`` is HTML (its names already escaped); ``head`` is plain text."""
    def esc(s: str) -> str:  # (text nodes: & < > only — "You're" stays readable)
        return html_mod.escape(str(s), quote=False)
    # (ACC-011: "<head> · WrapGTO" — the plain /games page used to read
    # "Home games · Home games"; ACC-012: links out, so a friend who follows an
    # invite can see what WrapGTO is before signing in)
    # Link previews (site ACC-007): an invite pasted into iMessage / WhatsApp /
    # Discord unfurls as "Join Miles's club" with the same sentence the page
    # shows, instead of a bare URL.
    desc = html_mod.escape(html_mod.unescape(re.sub(r"<[^>]+>", "", text)), quote=True)
    head_attr = html_mod.escape(str(head), quote=True)
    preview = (
        f'<meta name="description" content="{desc}">'
        '<meta property="og:type" content="website"><meta property="og:site_name" content="WrapGTO">'
        f'<meta property="og:title" content="{head_attr}"><meta property="og:description" content="{desc}">'
        '<meta property="og:image" content="https://wrapgto.com/static/brand/social-avatar-dark-1024.png">'
        '<meta name="twitter:card" content="summary">'
        f'<meta name="twitter:title" content="{head_attr}"><meta name="twitter:description" content="{desc}">'
    )
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{esc(head)} · WrapGTO</title>{preview}"
            '<link rel="icon" type="image/svg+xml" href="/static/brand/wrap-app-icon-dark.svg">'
            '<link rel="apple-touch-icon" sizes="180x180" href="/static/brand/apple-touch-icon.png">'
            '<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400..800&display=swap" rel="stylesheet">'
            f"<style>{hg._INVITE_CSS}</style></head><body><main class=\"card\">"
            f'<div class="brand">WrapGTO · Home games</div><h1>{esc(head)}</h1><p>{text}</p>'
            f'<a class="btn" href="/auth/login?next={html_mod.escape(next_path, quote=True)}">Sign in with Google</a>'
            '<nav class="links" aria-label="About WrapGTO"><a href="/">What is WrapGTO?</a>'
            '<a href="/terms">Terms</a><a href="/privacy">Privacy</a></nav></main></body></html>')


def _public_name(uid: int) -> str | None:
    """A player's name for a page anyone may open (signed out): the name they chose
    for the tables, or their account's own — never a club nickname, never anything
    from their email (SEC-005). None = "A friend"."""
    user = pub._user_by_id(int(uid))
    if user is None:
        return None
    return hg._chosen_name(int(uid)) or (None if hg._name_from_email(user) else hg._account_name(user))


def _games_phrase() -> str:
    """"PLO5, PLO6 and PLO67" — the games a table can deal here (from GAMES:
    HGB-013, the pages said "PLO5 and PLO6" after PLO67 arrived)."""
    labels = [g["label"] for g in hg.GAMES.values() if hg.PLO67_ON or not g["burns"]]
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def _invite_response(request: Request, path: str, user: Any) -> Response | None:
    """``pub._GAMES_INVITE_HOOK``: a /games page opened SIGNED OUT (or None for
    the hidden 404 — the API, the assets and unknown links)."""
    if user is not None or request.method.upper() not in ("GET", "HEAD"):
        return None
    if "text/html" not in request.headers.get("accept", "text/html"):
        return None
    m = hg._SIGNIN_PATH.match(path)
    if m is None:
        return None
    esc = html_mod.escape
    game_id, code = m.group(1), m.group(2)
    if game_id:
        table = pub.DB.one(
            "SELECT g.name, g.status, g.variant, g.host_user_id, c.name AS club_name FROM homegames g "
            "LEFT JOIN homegame_clubs c ON c.id=g.club_id WHERE g.id=?",
            (game_id,),
        )
        if table is None or table["status"] != "open":
            return None
        host_name = hg._public_name(int(table["host_user_id"]))
        club = f" in <b>{esc(str(table['club_name']))}</b>" if table["club_name"] else ""
        label = hg.GAMES[hg._norm_game(table["variant"])]["label"]
        html = hg._signin_html(
            f"You're invited to {table['name']}",
            f"<b>{esc(host_name or 'A friend')}</b> is hosting a {label} double-board bomb-pot "
            f"game{club}. Sign in with Google to join — you'll come straight back to the table.",
            path,
        )
    elif code:
        club = hg._club_by_code(code)
        if club is None:
            return None
        # (a signed-out page never shows an email, not even its first half — SEC-005)
        owner_name = hg._public_name(int(club["owner_user_id"]))
        html = hg._signin_html(
            f"Join {club['name']}",
            f"<b>{esc(owner_name or 'A friend')}</b> invited you to their home-games club: "
            f"{hg._games_phrase()} double-board bomb pots with friends, with the stats kept inside the club. "
            "Sign in with Google to join.",
            path,
        )
    else:
        html = hg._signin_html(
            "Home games",
            f"{hg._games_phrase()} double-board bomb pots with your friends, in private clubs. Sign in with Google "
            "to start a club or join one.",
            "/games",
        )
    return HTMLResponse(html, headers=dict(hg.INVITE_HEADERS))
