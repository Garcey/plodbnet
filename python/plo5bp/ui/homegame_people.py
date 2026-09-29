"""The people at the home games: their names, club nicknames and pictures.

A name follows one rule (FEAT-008 / FEAT-006 / SEC-005): in a club's context the
member's nickname there, else the name they chose for the tables, else their
account's name, else "Player <id>" — never an email (a masked one at most, for the
people deciding a join request). A picture is checked from its header alone (no
image library) and served with a sandboxing CSP.

Split out of ``homegame`` (HGB-006) and imported by it: every name defined here is
re-exported as ``homegame.<name>``. The code reaches every other home-games name
through ``hg`` (the ``homegame`` module), looked up when it runs, so patching
``homegame.X`` in a test reaches this module too and ``homegame.use_context`` swaps
its state. Patch ``homegame.X``, never this module's copy.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import logging
import re
import sys
from typing import Any

from fastapi import HTTPException

from plo5bp.ui import public as pub

#: The home games' main module: every home-games name is looked up there when used.
hg = sys.modules.get("plo5bp.ui.homegame") or importlib.import_module("plo5bp.ui.homegame")
logger = logging.getLogger("plo5bp.ui.homegame")

__all__ = (
    "AVATAR_HEADERS", "AVATAR_MAX_BYTES", "AVATAR_MAX_PX", "MAX_PLAYER_NAME", "_AVATAR_MIMES",
    "_PUBLIC_MAIL_DOMAINS", "_account_name", "_avatar_changed", "_avatar_url", "_chosen_name",
    "_clean_player_name", "_club_nicks", "_display_name", "_forget_names", "_image_info",
    "_mask_email", "_my_name_state", "_name_from_email", "_name_is_default",
    "_rename_at_tables", "_row_uid", "_set_avatar", "_set_nickname", "_set_table_name",
)


# --- names (FEAT-008 / FEAT-006 / SEC-005) -------------------------------------------------
#
# How the tables name a player, first that applies:
#   1. in a club's context (its tables, members, numbers): their NICKNAME in that club
#      (``homegame_club_members.nickname`` — set by them, or by the club's owner / admins);
#   2. the name they chose for the tables (``homegame_names``: "Name at the table");
#   3. their account's name (Google's, or what the emailed-link sign-in filled in);
#   4. "Player <id>" — never the email address (SEC-005: it used to fall back to the
#      address's local part, which is often the person's full name).
# A player with no chosen name whose account name is empty or just their email's local
# part is asked to choose one (``my_name_default``), and a page anyone may open
# (signed out) never shows such a name (``_public_name``).
# Stored hand records keep the name each seat had when the hand was played.

#: The longest name a player can choose (seat plates are narrow: FEAT-008).
MAX_PLAYER_NAME = 20


def _row_uid(row: Any) -> int | None:
    """The user id of a users row (``id``) or of a joined row (``user_id``)."""
    try:
        keys = row.keys()
    except AttributeError:
        keys = row
    for k in ("user_id", "id"):
        if k in keys and row[k] is not None:
            return int(row[k])
    return None


def _account_name(user: Any) -> str | None:
    """The account's own name (Google's, or the emailed-link sign-in's), or None."""
    return hg._clean_text(user["name"]) or None


def _name_from_email(user: Any) -> bool:
    """The account's name is only its email's local part (what the emailed-link
    sign-in fills in) — often the person's full name, never chosen as a name."""
    name = hg._clean_text(user["name"]).lower()
    return bool(name) and name == str(user["email"] or "").split("@")[0].strip().lower()


def _chosen_name(uid: int) -> str | None:
    """The name the player chose for the tables (cached; None = none)."""
    uid = int(uid)
    with hg.CTX.names_lock:
        if uid in hg.CTX.chosen_names:
            return hg.CTX.chosen_names[uid]
    row = pub.DB.one("SELECT name FROM homegame_names WHERE user_id=?", (uid,))
    name = str(row["name"]) if row is not None else None
    with hg.CTX.names_lock:
        if len(hg.CTX.chosen_names) > 20000:
            hg.CTX.chosen_names.clear()
        hg.CTX.chosen_names[uid] = name
    return name


def _club_nicks(club_id: Any) -> dict[int, str]:
    """{user id: nickname} of one club (cached per club)."""
    cid = str(club_id)
    with hg.CTX.names_lock:
        hit = hg.CTX.club_nicks.get(cid)
    if hit is not None:
        return hit
    nicks = {int(r["user_id"]): str(r["nickname"]) for r in pub.DB.q(
        "SELECT user_id, nickname FROM homegame_club_members WHERE club_id=? AND nickname IS NOT NULL "
        "AND nickname<>''", (cid,))}
    with hg.CTX.names_lock:
        if len(hg.CTX.club_nicks) > 2000:
            hg.CTX.club_nicks.clear()
        hg.CTX.club_nicks[cid] = nicks
    return nicks


def _display_name(user: Any, club_id: Any = None) -> str:
    """How the home games name ``user`` (a users row, or a joined row with name,
    email and user_id) — in ``club_id``'s context when given (see above)."""
    uid = hg._row_uid(user)
    if club_id and uid is not None:
        nick = hg._club_nicks(club_id).get(uid)
        if nick:
            return nick
    if uid is not None:
        chosen = hg._chosen_name(uid)
        if chosen:
            return chosen
    return hg._account_name(user) or (f"Player {uid}" if uid is not None else "Player")


def _name_is_default(user: Any) -> bool:
    """No name they chose — only "Player <id>" or their email's local part: the
    client asks them to choose one (FEAT-008 / SEC-005)."""
    uid = hg._row_uid(user)
    return (uid is None or hg._chosen_name(uid) is None) and (hg._account_name(user) is None or hg._name_from_email(user))


def _clean_player_name(v: Any, what: str = "name") -> str:
    """A player's name or nickname as typed: control and format characters out
    (SEC-007), 1–MAX_PLAYER_NAME characters, not an email address."""
    name = hg._clean_text(v)
    if not name:
        raise HTTPException(status_code=400, detail=f"type a {what}")
    if len(name) > hg.MAX_PLAYER_NAME:
        raise HTTPException(status_code=400, detail=f"a {what} is limited to {hg.MAX_PLAYER_NAME} characters")
    if "@" in name:
        raise HTTPException(status_code=400, detail=f"a {what} can't be an email address")
    return name


def _forget_names(uid: int | None = None, club_id: str | None = None) -> None:
    with hg.CTX.names_lock:
        if uid is not None:
            hg.CTX.chosen_names.pop(int(uid), None)
        if club_id is not None:
            hg.CTX.club_nicks.pop(str(club_id), None)


def _rename_at_tables(uid: int, club_id: str | None = None) -> None:
    """A name changed: the loaded tables that show it (``club_id``: only that club's)
    redraw with it now — seats, the viewer list, chat. One table lock at a time."""
    with hg.CTX.hub._lock:
        tables = [t for t in hg.CTX.hub._tables.values() if club_id is None or t.club_id == club_id]
    user = pub._user_by_id(int(uid))
    if user is None:
        return
    for t in tables:
        with t.lock:
            name = hg._display_name(user, t.club_id)
            changed = False
            for p in t.seats:
                if p is not None and p.user_id == int(uid) and p.name != name:
                    p.name = name
                    changed = True
            if int(uid) in t.names:
                t.names[int(uid)] = name
                changed = True
            if t.name_defaults.pop(int(uid), None) is not None:
                changed = True
            if t.chat_cache and any(m["user_id"] == int(uid) for m in t.chat_cache):
                t.chat_cache = None
                changed = True
            for r in t.requests:
                if r["user_id"] == int(uid):
                    r["name"] = name
                    changed = True
            if changed:
                t.rev += 1


def _set_table_name(uid: int, name: Any) -> dict[str, Any]:
    """FEAT-008: the name the tables show for this player (None / "" = back to the
    account's name). Returns the viewer's name state."""
    uid = int(uid)
    if name is None or not hg._clean_text(name):
        pub.DB.q("DELETE FROM homegame_names WHERE user_id=?", (uid,))
    else:
        pub.DB.q(
            "INSERT INTO homegame_names(user_id,name,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET name=excluded.name, updated_at=excluded.updated_at",
            (uid, hg._clean_player_name(name), pub._now()),
        )
    hg._forget_names(uid=uid)
    hg._rename_at_tables(uid)
    return hg._my_name_state(uid)


def _my_name_state(uid: int) -> dict[str, Any]:
    user = pub._user_by_id(int(uid))
    if user is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return {"name": hg._display_name(user), "chosen": hg._chosen_name(int(uid)),
            "account_name": hg._account_name(user), "default": hg._name_is_default(user)}


def _set_nickname(club_id: Any, by_uid: int, target: int, nickname: Any) -> dict[str, Any]:
    """FEAT-006: a member's name in ONE club — their own, or anyone's for the club's
    owner and admins (two Mikes, a joke name). None / "" = no nickname."""
    club, role = hg._require_club(club_id, by_uid)
    cid = str(club["id"])
    if int(target) != int(by_uid) and role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="only the club's owner or an admin can name someone else")
    if hg._club_role(cid, target) is None:
        raise HTTPException(status_code=404, detail="they are not in the club")
    nick = None if (nickname is None or not hg._clean_text(nickname)) else hg._clean_player_name(nickname, "nickname")
    pub.DB.q("UPDATE homegame_club_members SET nickname=? WHERE club_id=? AND user_id=?", (nick, cid, int(target)))
    hg._forget_names(club_id=cid)
    hg._rename_at_tables(int(target), cid)
    return hg._club_view(cid, by_uid)


# --- profile pictures (owner, 2026-09-26) ----------------------------------------
# The browser crops and re-encodes the photo to a small square (which also drops
# its EXIF metadata) and uploads it; the server has no image library, so it only
# accepts a PNG / JPEG / WebP whose header it can read, within size and pixel
# limits, and serves it back with nosniff + a sandboxing CSP. Stored in SQLite
# (small, and the nightly DB backup covers it). URLs carry a content version, so
# a new picture is a new URL and a browser may cache any one of them for good.
AVATAR_MAX_BYTES = 200_000
AVATAR_MAX_PX = 1024
AVATAR_HEADERS = {
    "Cache-Control": "private, max-age=31536000, immutable",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox",
    "Content-Disposition": "inline",
}
_AVATAR_MIMES = ("image/png", "image/jpeg", "image/webp")


def _image_info(b: bytes) -> tuple[str, int, int] | None:
    """(mime, width, height) of a PNG / JPEG / WebP read from its header, or None."""
    if b.startswith(b"\x89PNG\r\n\x1a\n") and len(b) >= 24 and b[12:16] == b"IHDR":
        return "image/png", int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")
    if b[:3] == b"\xff\xd8\xff":
        i = 2
        while i + 9 < len(b):
            if b[i] != 0xFF:
                return None
            marker = b[i + 1]
            if marker == 0xFF:  # fill byte
                i += 1
                continue
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                return "image/jpeg", int.from_bytes(b[i + 7:i + 9], "big"), int.from_bytes(b[i + 5:i + 7], "big")
            i += 2 + int.from_bytes(b[i + 2:i + 4], "big")
        return None
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP" and len(b) >= 30:
        chunk = b[12:16]
        if chunk == b"VP8 " and b[23:26] == b"\x9d\x01\x2a":
            return ("image/webp", int.from_bytes(b[26:28], "little") & 0x3FFF,
                    int.from_bytes(b[28:30], "little") & 0x3FFF)
        if chunk == b"VP8L" and b[20] == 0x2F:
            bits = int.from_bytes(b[21:25], "little")
            return "image/webp", (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8X":
            return ("image/webp", int.from_bytes(b[24:27], "little") + 1,
                    int.from_bytes(b[27:30], "little") + 1)
    return None


def _avatar_url(uid: Any) -> str | None:
    """The user's picture URL (with its content version), or None — cached."""
    if uid is None:
        return None
    uid = int(uid)
    with hg.CTX.avatar_lock:
        if uid in hg.CTX.avatar_urls:
            return hg.CTX.avatar_urls[uid]
    row = pub.DB.one("SELECT version FROM homegame_avatars WHERE user_id=?", (uid,))
    url = f"/games/api/avatars/{uid}?v={row['version']}" if row is not None else None
    with hg.CTX.avatar_lock:
        hg.CTX.avatar_urls[uid] = url
    return url


def _set_avatar(uid: int, data_url: Any, *, before_write: Any = None) -> str | None:
    """Store (or with None, remove) the user's picture; returns its new URL.
    ``before_write()`` runs once the picture has been read and accepted, right
    before it is stored (the route's upload budget: SEC-009 counts only real
    uploads, not rejected ones)."""
    uid = int(uid)
    if data_url is None:
        pub.DB.q("DELETE FROM homegame_avatars WHERE user_id=?", (uid,))
    else:
        m = re.fullmatch(r"data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=\s]+)", str(data_url))
        if m is None or len(m.group(2)) > hg.AVATAR_MAX_BYTES * 4 // 3 + 16:
            raise HTTPException(status_code=400, detail="send a PNG, JPEG or WebP picture under 200 KB")
        try:
            data = base64.b64decode("".join(m.group(2).split()), validate=True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="that picture could not be read") from e
        info = hg._image_info(data)
        if info is None or info[0] != m.group(1) or len(data) > hg.AVATAR_MAX_BYTES:
            raise HTTPException(status_code=400, detail="that picture could not be read")
        if not (16 <= info[1] <= hg.AVATAR_MAX_PX and 16 <= info[2] <= hg.AVATAR_MAX_PX):
            raise HTTPException(status_code=400, detail="pictures must be 16 to 1024 pixels on a side")
        if before_write is not None:
            before_write()
        version = hashlib.sha256(data).hexdigest()[:12]
        pub.DB.q(
            "INSERT INTO homegame_avatars(user_id,mime,data,version,updated_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET mime=excluded.mime, data=excluded.data, "
            "version=excluded.version, updated_at=excluded.updated_at",
            (uid, info[0], data, version, pub._now()),
        )
    with hg.CTX.avatar_lock:
        hg.CTX.avatar_urls.pop(uid, None)
    return hg._avatar_url(uid)


#: Mail providers anyone may have: shown in full in a masked address (a private
#: domain is shown as its first letter and ending only — SEC-005: it can name the
#: person or their company).
_PUBLIC_MAIL_DOMAINS = frozenset((
    "gmail.com", "googlemail.com", "icloud.com", "me.com", "mac.com", "outlook.com", "hotmail.com",
    "live.com", "msn.com", "yahoo.com", "ymail.com", "aol.com", "proton.me", "protonmail.com",
    "pm.me", "gmx.com", "gmx.net", "mail.com", "zoho.com", "fastmail.com", "hey.com", "yandex.com",
))


def _mask_email(email: Any) -> str:
    """Enough to tell two Alexes apart, not a mailing list: ``a•••@gmail.com``,
    ``a•••@s•••.com`` for a private domain."""
    s = str(email or "").strip()
    if "@" not in s:
        return ""
    user, dom = s.split("@", 1)
    dom = dom.lower()
    if dom not in hg._PUBLIC_MAIL_DOMAINS:
        tld = dom.rsplit(".", 1)[-1] if "." in dom else ""
        dom = f"{dom[:1]}•••" + (f".{tld}" if tld else "")
    return f"{user[:1]}•••@{dom}"


def _avatar_changed(uid: int) -> None:
    """The tables this player sits at redraw with the new picture now (one table
    lock at a time)."""
    with hg.CTX.hub._lock:
        tables = list(hg.CTX.hub._tables.values())
    for tb in tables:
        with tb.lock:
            if tb.player(uid) is not None:
                tb.rev += 1
