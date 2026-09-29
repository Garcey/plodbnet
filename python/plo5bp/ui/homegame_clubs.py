"""Clubs: the private circles the home games are played in.

A table belongs to exactly one club; only its members see it, sit, watch or open
its history, and every number is computed from one club's tables. Membership,
roles, join requests, invite links, the archive, the main club and its migration,
and the access check every table request goes through (``_table_access``).

Split out of ``homegame`` (HGB-006) and imported by it: every name defined here is
re-exported as ``homegame.<name>``. The code reaches every other home-games name
through ``hg`` (the ``homegame`` module), looked up when it runs, so patching
``homegame.X`` in a test reaches this module too and ``homegame.use_context`` swaps
its state. Patch ``homegame.X``, never this module's copy.
"""

from __future__ import annotations

import importlib
import logging
import re
import secrets
import sys
import time
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from fastapi import HTTPException

from plo5bp.ui import public as pub

if TYPE_CHECKING:  # (annotations only)
    from plo5bp.ui.homegame import LiveTable

#: The home games' main module: every home-games name is looked up there when used.
hg = sys.modules.get("plo5bp.ui.homegame") or importlib.import_module("plo5bp.ui.homegame")
logger = logging.getLogger("plo5bp.ui.homegame")

__all__ = (
    "CLUB_ROLES", "JOIN_REFRESH_S", "JOIN_RETRY_S", "MAX_CLUBS_OWNED", "MAX_CLUB_NAME",
    "_CLUBS_WRITE_RE", "_CODE_RE", "_ROLE_RANK", "_add_member", "_admin_games_access",
    "_archive_club", "_busy_in_club", "_club", "_club_by_code", "_club_for_new_table",
    "_club_member_ids", "_club_requests", "_club_role", "_club_settings", "_club_summary",
    "_club_view", "_create_club", "_decide_club_request", "_invite_info", "_join_by_invite",
    "_join_retry_in", "_leave_club", "_main_club", "_main_club_member", "_main_club_members",
    "_may_see_player",
    "_migrate_clubs", "_my_archived_clubs", "_my_club_ids", "_my_clubs", "_on_clubs_write",
    "_open_club_tables_in_memory", "_request_club", "_request_state", "_require_club",
    "_reset_invite", "_set_member", "_shares_club", "_table_access", "_table_club",
    "_table_join_requests",
)


# --- clubs (2026-09-25) -----------------------------------------------------------
#
# Home games are open to every signed-in user; a CLUB is the private circle: its
# tables, its members and its numbers. A table belongs to exactly one club. Only
# members see it in the lobby, sit, watch or open its history, and every ranking,
# accuracy score, profit table and head-to-head is computed from ONE club's tables:
# nothing ranks the whole user base. Anyone can start a club and host in it.
# People join with the club's invite link (the owner can make it "ask first"), or
# ask from a table link and wait for the owner or an admin to let them in.
# The site's original private circle is the MAIN club: the migration gives it every
# table and player that predate clubs, and the /admin home-games switch adds and
# removes people there.

CLUB_ROLES = ("owner", "admin", "member")
MAX_CLUBS_OWNED = 5
MAX_CLUB_NAME = 40
JOIN_RETRY_S = 60.0     # a declined request can be sent again after this
JOIN_REFRESH_S = 5.0    # how stale a club manager's view of the requests may be
_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{6,40}$")


def _club(club_id: Any) -> Any:
    if not club_id or not hg._CODE_RE.match(str(club_id)):
        return None
    return pub.DB.one("SELECT * FROM homegame_clubs WHERE id=?", (str(club_id),))


# PERF-004: a person's role in a club is asked for on every request (the access
# check), every view (the club badge, join requests) and every live-stream push.
# It is cached here — and the whole cache is dropped by ANY committed write to the
# clubs' tables (a DB write listener, like public.py's user cache), so a removal,
# a new admin or an archived club counts from the very next request.
_CLUBS_WRITE_RE = re.compile(r"\b(update|into|from)\s+homegame_club(s|_members)\b", re.I)


def _on_clubs_write(sql: str) -> None:
    if hg._CLUBS_WRITE_RE.search(sql):
        with hg.CTX.role_lock:
            hg.CTX.role_gen[0] += 1
            hg.CTX.role_cache.clear()


def _club_role(club_id: Any, uid: Any) -> str | None:
    if not club_id or uid is None:
        return None
    key = (str(club_id), int(uid))
    with hg.CTX.role_lock:
        if key in hg.CTX.role_cache:
            return hg.CTX.role_cache[key]
        gen = hg.CTX.role_gen[0]
    r = pub.DB.one(
        "SELECT role FROM homegame_club_members WHERE club_id=? AND user_id=?", key
    )
    role = str(r["role"]) if r is not None else None
    with hg.CTX.role_lock:
        if gen == hg.CTX.role_gen[0]:  # (a write in between: don't keep what may be old)
            if len(hg.CTX.role_cache) > 50000:
                hg.CTX.role_cache.clear()
            hg.CTX.role_cache[key] = role
    return role


def _require_club(club_id: Any, uid: int, *roles: str, archived: bool = False) -> tuple[Any, str]:
    """The club and the viewer's role in it. Not a member = 404 (a club's
    existence is nobody else's business); a member without one of ``roles`` = 403.
    An archived club (FEAT-007) is gone for everything but ``archived=True``
    (its owner restoring it)."""
    club = hg._club(club_id)
    role = hg._club_role(club["id"], uid) if club is not None else None
    if club is None or role is None or (club["archived_at"] and not archived):
        raise HTTPException(status_code=404, detail="Not Found")
    if roles and role not in roles:
        who = "the club's owner" if roles == ("owner",) else "the club's owner or an admin"
        raise HTTPException(status_code=403, detail=f"only {who} can do that")
    return club, role


def _club_member_ids(club_id: str) -> set[int]:
    return {int(r["user_id"]) for r in pub.DB.q(
        "SELECT user_id FROM homegame_club_members WHERE club_id=?", (club_id,))}


def _add_member(club_id: str, uid: int, role: str = "member") -> None:
    now = pub._now()
    pub.DB.q(
        "INSERT OR IGNORE INTO homegame_club_members(club_id,user_id,role,joined_at) VALUES(?,?,?,?)",
        (club_id, int(uid), role, now),
    )
    pub.DB.q(
        "UPDATE homegame_club_requests SET status='approved', decided_at=? "
        "WHERE club_id=? AND user_id=? AND status='pending'", (now, club_id, int(uid)),
    )


def _create_club(user: Any, name: Any, *, main: bool = False) -> str:
    name = hg._clean_text(name)
    if not name:
        raise HTTPException(status_code=400, detail="give the club a name")
    if len(name) > hg.MAX_CLUB_NAME:
        raise HTTPException(status_code=400, detail=f"a club name is limited to {hg.MAX_CLUB_NAME} characters")
    uid = int(user["id"])
    owned = pub.DB.one("SELECT COUNT(*) c FROM homegame_clubs WHERE owner_user_id=? AND archived_at IS NULL",
                       (uid,))["c"]  # (an archived club doesn't count: FEAT-007)
    if not main and int(owned) >= hg.MAX_CLUBS_OWNED:
        raise HTTPException(status_code=429, detail=f"you already run {hg.MAX_CLUBS_OWNED} clubs")
    cid = secrets.token_urlsafe(6)
    now = pub._now()
    with pub.DB.transaction():
        pub.DB.q(
            "INSERT INTO homegame_clubs(id,name,owner_user_id,invite_code,approve_joins,is_main,created_at) "
            "VALUES(?,?,?,?,0,?,?)",
            (cid, name, uid, secrets.token_urlsafe(9), 1 if main else 0, now),
        )
        pub.DB.q(
            "INSERT OR IGNORE INTO homegame_club_members(club_id,user_id,role,joined_at) VALUES(?,?,'owner',?)",
            (cid, uid, now),
        )
    return cid


def _main_club(create_for: int | None = None) -> str | None:
    """The site's original circle (see above). Made on first need: by the
    migration, or by the first /admin home-games grant (``create_for`` = the
    granting admin, who then owns it)."""
    r = pub.DB.one("SELECT id FROM homegame_clubs WHERE is_main=1 ORDER BY created_at LIMIT 1")
    if r is not None:
        return str(r["id"])
    owner = pub._user_by_id(int(create_for)) if create_for is not None else None
    if owner is None:
        return None
    first = (hg._display_name(owner) or "Home").split(" ")[0]
    return hg._create_club(owner, f"{first}'s club"[:hg.MAX_CLUB_NAME], main=True)


def _migrate_clubs() -> None:
    """Tables that predate clubs (``club_id`` NULL) join the MAIN club, and so
    does everyone who ever hosted or sat at one, plus everyone who had the old
    admin-granted home-games flag: the circle and its numbers carry on exactly as
    they were. Idempotent (nothing left to move = nothing happens)."""
    legacy = pub.DB.q("SELECT id, host_user_id FROM homegames WHERE club_id IS NULL ORDER BY created_at")
    if not legacy:
        return
    granted = [int(r["id"]) for r in pub.DB.q("SELECT id FROM users WHERE homegame_access=1 ORDER BY created_at")]
    admins = [int(r["id"]) for r in pub.DB.q("SELECT id, email FROM users ORDER BY created_at") if pub._is_admin(r)]
    owner = admins[0] if admins else int(legacy[0]["host_user_id"])
    cid = hg._main_club(create_for=owner)
    if cid is None:
        return
    members = set(granted) | {int(r["host_user_id"]) for r in legacy}
    for r in pub.DB.q("SELECT DISTINCT p.user_id FROM homegame_players p JOIN homegames g ON g.id=p.game_id "
                      "WHERE g.club_id IS NULL"):
        members.add(int(r["user_id"]))
    with pub.DB.transaction():
        for uid in sorted(members):
            if pub._user_by_id(uid) is not None:
                pub.DB.q(
                    "INSERT OR IGNORE INTO homegame_club_members(club_id,user_id,role,joined_at) "
                    "VALUES(?,?,'member',?)", (cid, uid, pub._now()),
                )
        pub.DB.q("UPDATE homegames SET club_id=? WHERE club_id IS NULL", (cid,))
    logger.info("clubs: %d table(s) and %d player(s) moved into the main club %s", len(legacy), len(members), cid)


def _club_for_new_table(uid: int, requested: Any) -> str:
    """Any member may host in a club. Without a choice (scripts, older clients):
    the main club when the host is in it, else their oldest club."""
    if requested:
        club, _ = hg._require_club(requested, uid)
        return str(club["id"])
    r = pub.DB.one(
        "SELECT m.club_id FROM homegame_club_members m JOIN homegame_clubs c ON c.id=m.club_id "
        "WHERE m.user_id=? AND c.archived_at IS NULL ORDER BY c.is_main DESC, m.joined_at LIMIT 1", (int(uid),),
    )
    if r is None:
        raise HTTPException(status_code=400, detail="start a club (or join one) before you host a table")
    return str(r["club_id"])


def _join_retry_in(decided_at: str | None) -> float:
    try:
        dt = datetime.fromisoformat(str(decided_at))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, hg.JOIN_RETRY_S - (datetime.now(timezone.utc) - dt).total_seconds())


def _request_state(club_id: str, uid: int) -> tuple[str | None, float]:
    r = pub.DB.one(
        "SELECT status, decided_at FROM homegame_club_requests WHERE club_id=? AND user_id=?", (club_id, int(uid))
    )
    if r is None:
        return None, 0.0
    status = str(r["status"])
    return status, (hg._join_retry_in(r["decided_at"]) if status == "declined" else 0.0)


def _request_club(club_id: str, uid: int) -> None:
    """Ask to join (from a table link, or an invite link of an ask-first club).
    Once pending, or declined less than a minute ago, asking again is a no-op."""
    if hg._club_role(club_id, uid) is not None:
        return
    status, retry = hg._request_state(club_id, uid)
    if status == "pending" or (status == "declined" and retry > 0):
        return
    pub.DB.q(
        "INSERT OR REPLACE INTO homegame_club_requests(club_id,user_id,status,created_at,decided_at) "
        "VALUES(?,?,'pending',?,NULL)", (club_id, int(uid), pub._now()),
    )
    user = pub._user_by_id(uid)
    for t in hg._open_club_tables_in_memory(club_id):
        with t.lock:
            t.join_checked_mono = 0.0
            hg._emit(t, "joinreq", f"{hg._display_name(user) if user else 'Someone'} asks to join the club")
            t.rev += 1


def _open_club_tables_in_memory(club_id: str) -> list[LiveTable]:
    with hg.CTX.hub._lock:
        return [t for t in hg.CTX.hub._tables.values() if t.club_id == club_id and t.status == "open"]


def _club_requests(club_id: str) -> list[dict[str, Any]]:
    """Pending requests, oldest first (people who got in meanwhile drop out)."""
    club = hg._club(club_id)
    rows = pub.DB.q(
        "SELECT r.user_id, r.created_at, u.name, u.email FROM homegame_club_requests r "
        "JOIN users u ON u.id=r.user_id WHERE r.club_id=? AND r.status='pending' ORDER BY r.created_at",
        (club_id,),
    )
    members = hg._club_member_ids(club_id)
    return [
        {"club_id": club_id, "club_name": str(club["name"]) if club else "", "user_id": int(r["user_id"]),
         "name": hg._display_name(r), "avatar": hg._avatar_url(r["user_id"]),
         "email": hg._mask_email(r["email"]), "since": str(r["created_at"])}
        for r in rows if int(r["user_id"]) not in members
    ]


def _decide_club_request(club_id: str, by_uid: int, target: int, allow: bool) -> dict[str, Any]:
    hg._require_club(club_id, by_uid, "owner", "admin")
    status, _ = hg._request_state(club_id, target)
    if status != "pending" or hg._club_role(club_id, target) is not None:
        raise HTTPException(status_code=409, detail="that request is gone")
    user = pub._user_by_id(target)
    if allow:
        hg._add_member(club_id, target)
    else:
        pub.DB.q("UPDATE homegame_club_requests SET status='declined', decided_at=? WHERE club_id=? AND user_id=?",
                 (pub._now(), club_id, int(target)))
    for t in hg._open_club_tables_in_memory(club_id):
        with t.lock:
            t.join_checked_mono = 0.0
            if allow:
                hg._emit(t, "join", f"{hg._display_name(user) if user else 'A new player'} joined the club")
            t.rev += 1
    return {"ok": True, "allowed": bool(allow)}


def _table_join_requests(t: LiveTable, viewer_id: int | None) -> list[dict[str, Any]]:
    """What the club's owner / an admin at this table is asked to decide."""
    if viewer_id is None or hg._club_role(t.club_id, viewer_id) not in ("owner", "admin"):
        return []
    now = time.monotonic()
    if now - t.join_checked_mono > hg.JOIN_REFRESH_S:
        t.join_checked_mono = now
        try:
            t.join_reqs = hg._club_requests(t.club_id) if t.club_id else []
        except Exception:  # noqa: BLE001 — cosmetic: never take the view down
            logger.exception("club requests (table %s)", t.game_id)
    return list(t.join_reqs)


def _table_club(t: LiveTable, viewer_id: int | None) -> dict[str, Any] | None:
    """The table's club as its view shows it (cached per table: names rarely change)."""
    now = time.monotonic()
    if t.club_info is None or now - t.club_checked_mono > 30.0:
        t.club_checked_mono = now
        club = hg._club(t.club_id)
        t.club_info = {"id": str(club["id"]), "name": str(club["name"])} if club is not None else None
    if t.club_info is None:
        return None
    return dict(t.club_info, role=hg._club_role(t.club_id, viewer_id))


def _table_access(t: LiveTable, uid: int) -> None:
    """Only the table's club may see it, sit, watch or act. Anyone else gets 403
    naming the club (and where their request stands), so a table link can offer
    "ask to join the club" instead of a dead end."""
    if t.club_id and hg._club_role(t.club_id, uid) is not None:
        return
    club = hg._club(t.club_id)
    if club is None:  # (no club at all cannot happen after the migration: its host only)
        if int(uid) == int(t.host_user_id):
            return
        raise HTTPException(status_code=404, detail="Not Found")
    status, retry = hg._request_state(str(club["id"]), uid)
    raise HTTPException(status_code=403, detail={
        "error": "club", "club": {"id": str(club["id"]), "name": str(club["name"])},
        "request": status, "retry_in": round(retry, 1),
        "message": f"This table belongs to the club “{club['name']}”.",
    })


def _busy_in_club(club_id: str, uid: int, who: str) -> str | None:
    """Why this person can't leave the club right now (``who`` = "you" or their
    name), or None: a seat at one of its open tables, or hosting one (a table
    whose host is outside its club would have nobody left to run it)."""
    r = pub.DB.one(
        "SELECT g.name FROM homegame_players p JOIN homegames g ON g.id=p.game_id "
        "WHERE g.club_id=? AND g.status='open' AND p.user_id=? AND p.seat IS NOT NULL LIMIT 1",
        (club_id, int(uid)),
    )
    if r is not None:
        return f"{who} {'have' if who == 'you' else 'has'} a seat at “{r['name']}” — leave the table first"
    r = pub.DB.one(
        "SELECT name FROM homegames WHERE club_id=? AND status='open' AND host_user_id=? LIMIT 1",
        (club_id, int(uid)),
    )
    if r is not None:
        return f"{who} {'host' if who == 'you' else 'hosts'} “{r['name']}” — hand the table over or close it first"
    return None


def _club_summary(r: Any, uid: int) -> dict[str, Any]:
    """One club in a person's list (``r`` from ``_my_clubs``' query, which counts
    its members, open tables and pending requests in the same statement)."""
    role = str(r["role"])
    return {
        "id": str(r["id"]), "name": str(r["name"]), "role": role, "is_main": bool(int(r["is_main"] or 0)),
        "members": int(r["members"] or 0), "open_tables": int(r["open_tables"] or 0),
        "requests": int(r["requests"] or 0) if role in ("owner", "admin") else 0,
    }


def _my_clubs(uid: int) -> list[dict[str, Any]]:
    """The person's clubs (not archived), main club first — ONE statement whatever
    their number (PERF-010: it was three queries more per club, every lobby poll)."""
    rows = pub.DB.q(
        "SELECT c.id, c.name, c.is_main, m.role, "
        "(SELECT COUNT(*) FROM homegame_club_members x WHERE x.club_id=c.id) AS members, "
        "(SELECT COUNT(*) FROM homegames g WHERE g.club_id=c.id AND g.status='open') AS open_tables, "
        "(SELECT COUNT(*) FROM homegame_club_requests q WHERE q.club_id=c.id AND q.status='pending' "
        " AND NOT EXISTS (SELECT 1 FROM homegame_club_members y WHERE y.club_id=q.club_id "
        " AND y.user_id=q.user_id)) AS requests "
        "FROM homegame_club_members m JOIN homegame_clubs c ON c.id=m.club_id "
        "WHERE m.user_id=? AND c.archived_at IS NULL ORDER BY c.is_main DESC, c.created_at", (int(uid),),
    )
    return [hg._club_summary(r, uid) for r in rows]


def _my_club_ids(uid: int) -> list[str]:
    """Just the ids of the person's clubs, in ``_my_clubs``' order (the stats scopes
    used to build the whole list to read them — PERF-010)."""
    return [str(r["id"]) for r in pub.DB.q(
        "SELECT c.id FROM homegame_club_members m JOIN homegame_clubs c ON c.id=m.club_id "
        "WHERE m.user_id=? AND c.archived_at IS NULL ORDER BY c.is_main DESC, c.created_at", (int(uid),))]


def _my_archived_clubs(uid: int) -> list[dict[str, Any]]:
    """The clubs this person archived (FEAT-007): only their owner sees them — to
    restore one."""
    return [{"id": str(r["id"]), "name": str(r["name"]), "archived_at": str(r["archived_at"])}
            for r in pub.DB.q(
                "SELECT id, name, archived_at FROM homegame_clubs WHERE owner_user_id=? "
                "AND archived_at IS NOT NULL ORDER BY archived_at DESC", (int(uid),))]


def _archive_club(club_id: Any, uid: int, on: bool) -> dict[str, Any]:
    """FEAT-007: the OWNER retires a club — or brings it back. Archived, it leaves
    every member's lobby and club list, its invite link stops working and nobody
    can host in it or ask to join; nothing is deleted (tables, hands, ledgers,
    stats, memberships), its old tables still open by link, and the owner can
    restore it from the club menu at any time. An archived club doesn't count
    toward the clubs a person may run. The main club (the site's own circle)
    can't be archived, nor a club with a table still open."""
    club, _ = hg._require_club(club_id, uid, "owner", archived=True)
    cid = str(club["id"])
    if on:
        if int(club["is_main"] or 0):
            raise HTTPException(status_code=400, detail="the main club can't be archived")
        busy = pub.DB.one("SELECT name FROM homegames WHERE club_id=? AND status='open' LIMIT 1", (cid,))
        if busy is not None:
            raise HTTPException(status_code=409, detail=f"close the table “{busy['name']}” first")
        if not club["archived_at"]:
            pub.DB.q("UPDATE homegame_clubs SET archived_at=? WHERE id=?", (pub._now(), cid))
    elif club["archived_at"]:
        owned = pub.DB.one("SELECT COUNT(*) c FROM homegame_clubs WHERE owner_user_id=? AND archived_at IS NULL",
                           (int(uid),))["c"]
        if int(owned) >= hg.MAX_CLUBS_OWNED:
            raise HTTPException(status_code=429, detail=f"you already run {hg.MAX_CLUBS_OWNED} clubs — archive one first")
        pub.DB.q("UPDATE homegame_clubs SET archived_at=NULL WHERE id=?", (cid,))
    return {"ok": True, "id": cid, "archived": bool(on), "clubs": hg._my_clubs(uid),
            "archived_clubs": hg._my_archived_clubs(uid)}


def _shares_club(a: int, b: int) -> bool:
    return pub.DB.one(
        "SELECT 1 FROM homegame_club_members x JOIN homegame_club_members y "
        "ON x.club_id=y.club_id WHERE x.user_id=? AND y.user_id=? LIMIT 1",
        (int(a), int(b)),
    ) is not None


def _may_see_player(viewer: int, player: int, club: str | None = None) -> bool:
    """May ``viewer`` look at ``player`` (their name, picture, numbers)? Only
    themselves, someone they share a club with — or, inside a club the viewer is
    in (``club``, already checked), someone who played there (a former member
    still has a card in the club's numbers). SEC-002: the player routes answered
    with the NAME of any user id, so counting 1, 2, 3 … listed the user base."""
    if int(viewer) == int(player) or hg._shares_club(viewer, player):
        return True
    return bool(club) and pub.DB.one(
        "SELECT 1 FROM homegame_hand_results r JOIN homegames g ON g.id=r.game_id "
        "WHERE r.user_id=? AND g.club_id=? LIMIT 1", (int(player), str(club)),
    ) is not None


_ROLE_RANK = {"owner": 0, "admin": 1, "member": 2}


def _club_view(club_id: Any, uid: int) -> dict[str, Any]:
    club, role = hg._require_club(club_id, uid)
    cid = str(club["id"])
    manage = role in ("owner", "admin")
    owner = pub._user_by_id(int(club["owner_user_id"]))
    rows = pub.DB.q(
        "SELECT m.user_id, m.role, m.joined_at, m.nickname, u.name, u.email FROM homegame_club_members m "
        "JOIN users u ON u.id=m.user_id WHERE m.club_id=?", (cid,),
    )
    members = sorted(
        ({"user_id": int(r["user_id"]), "name": hg._display_name(r, cid), "role": str(r["role"]),
          # FEAT-006: the club's nickname for them (None: none) and the name they go by
          # everywhere else, so everyone can tell who "Big Mike" is
          "nickname": r["nickname"] or None, "base_name": hg._display_name(r),
          "avatar": hg._avatar_url(r["user_id"]),
          "is_me": int(r["user_id"]) == int(uid), "joined_at": str(r["joined_at"])} for r in rows),
        key=lambda m: (hg._ROLE_RANK.get(m["role"], 9), m["name"].lower()),
    )
    return {
        "id": cid, "name": str(club["name"]), "role": role, "is_main": bool(int(club["is_main"] or 0)),
        "approve_joins": bool(int(club["approve_joins"] or 0)),
        "owner": {"user_id": int(club["owner_user_id"]), "name": hg._display_name(owner, cid) if owner else "?"},
        "members": members,
        "invite_code": str(club["invite_code"]) if manage else None,
        "requests": hg._club_requests(cid) if manage else [],
        "created_at": str(club["created_at"]),
    }


def _club_settings(club_id: Any, uid: int, body: dict) -> dict[str, Any]:
    club, _ = hg._require_club(club_id, uid, "owner")
    cid = str(club["id"])
    name = str(club["name"])
    if body.get("name") is not None:
        name = hg._clean_text(body.get("name"))
        if not name:
            raise HTTPException(status_code=400, detail="give the club a name")
        if len(name) > hg.MAX_CLUB_NAME:
            raise HTTPException(status_code=400, detail=f"a club name is limited to {hg.MAX_CLUB_NAME} characters")
    approve = hg._parse_bool(body, "approve_joins", bool(int(club["approve_joins"] or 0)))
    pub.DB.q("UPDATE homegame_clubs SET name=?, approve_joins=? WHERE id=?", (name, 1 if approve else 0, cid))
    return hg._club_view(cid, uid)


def _reset_invite(club_id: Any, uid: int) -> dict[str, Any]:
    club, _ = hg._require_club(club_id, uid, "owner", "admin")
    pub.DB.q("UPDATE homegame_clubs SET invite_code=? WHERE id=?", (secrets.token_urlsafe(9), str(club["id"])))
    return hg._club_view(str(club["id"]), uid)


def _set_member(club_id: Any, by_uid: int, target: int, *, role: Any = None, remove: bool = False) -> dict[str, Any]:
    club, my_role = hg._require_club(club_id, by_uid, "owner", "admin")
    cid = str(club["id"])
    their = hg._club_role(cid, target)
    if their is None:
        raise HTTPException(status_code=404, detail="they are not in the club")
    who = pub._user_by_id(target)
    name = hg._display_name(who, cid) if who is not None else "They"
    if remove:
        if int(target) == int(by_uid):
            raise HTTPException(status_code=400, detail="to go, use Leave club")
        if their == "owner":
            raise HTTPException(status_code=403, detail="the owner can't be removed")
        if my_role == "admin" and their != "member":
            raise HTTPException(status_code=403, detail="only the owner can remove an admin")
        busy = hg._busy_in_club(cid, target, name)
        if busy:
            raise HTTPException(status_code=409, detail=busy)
        pub.DB.q("DELETE FROM homegame_club_members WHERE club_id=? AND user_id=?", (cid, int(target)))
        return hg._club_view(cid, by_uid)
    if my_role != "owner":
        raise HTTPException(status_code=403, detail="only the club's owner can change roles")
    role = str(role or "")
    if role not in hg.CLUB_ROLES:
        raise HTTPException(status_code=400, detail="role must be owner, admin or member")
    if role == "owner":
        # handing the club over: the old owner stays on as an admin
        if int(target) == int(by_uid):
            return hg._club_view(cid, by_uid)
        with pub.DB.transaction():
            pub.DB.q("UPDATE homegame_clubs SET owner_user_id=? WHERE id=?", (int(target), cid))
            pub.DB.q("UPDATE homegame_club_members SET role='owner' WHERE club_id=? AND user_id=?", (cid, int(target)))
            pub.DB.q("UPDATE homegame_club_members SET role='admin' WHERE club_id=? AND user_id=?", (cid, int(by_uid)))
        return hg._club_view(cid, by_uid)
    if their == "owner":
        raise HTTPException(status_code=400, detail="the owner's role only changes by handing the club over")
    pub.DB.q("UPDATE homegame_club_members SET role=? WHERE club_id=? AND user_id=?", (role, cid, int(target)))
    return hg._club_view(cid, by_uid)


def _leave_club(club_id: Any, uid: int) -> None:
    club, role = hg._require_club(club_id, uid)
    cid = str(club["id"])
    if role == "owner":
        raise HTTPException(status_code=400, detail="hand the club to another member first (Club settings → Members)")
    busy = hg._busy_in_club(cid, uid, "you")
    if busy:
        raise HTTPException(status_code=409, detail=busy[0].upper() + busy[1:])
    pub.DB.q("DELETE FROM homegame_club_members WHERE club_id=? AND user_id=?", (cid, int(uid)))


def _club_by_code(code: Any) -> Any:
    """The club an invite code opens — never an archived one (FEAT-007)."""
    if not code or not hg._CODE_RE.match(str(code)):
        return None
    return pub.DB.one("SELECT * FROM homegame_clubs WHERE invite_code=? AND archived_at IS NULL", (str(code),))


def _invite_info(code: Any, uid: int | None) -> dict[str, Any]:
    club = hg._club_by_code(code)
    if club is None:
        raise HTTPException(status_code=404, detail="This invite link is no longer valid — ask for a new one.")
    cid = str(club["id"])
    owner = pub._user_by_id(int(club["owner_user_id"]))
    status, retry = hg._request_state(cid, uid) if uid is not None else (None, 0.0)
    return {
        "club": {"id": cid, "name": str(club["name"]), "owner_name": hg._display_name(owner) if owner else "?",
                 "members": len(hg._club_member_ids(cid))},
        "member": uid is not None and hg._club_role(cid, uid) is not None,
        "approve": bool(int(club["approve_joins"] or 0)),
        "request": status, "retry_in": round(retry, 1),
    }


def _join_by_invite(code: Any, uid: int) -> dict[str, Any]:
    club = hg._club_by_code(code)
    if club is None:
        raise HTTPException(status_code=404, detail="This invite link is no longer valid — ask for a new one.")
    cid = str(club["id"])
    if hg._club_role(cid, uid) is None:
        if bool(int(club["approve_joins"] or 0)):
            hg._request_club(cid, uid)
        else:
            hg._add_member(cid, uid)
    return hg._invite_info(code, uid)


def _main_club_member(uid: int) -> bool:
    """/admin's home-games column: is this person in the main club?"""
    cid = hg._main_club()
    return cid is not None and hg._club_role(cid, uid) is not None


def _main_club_members() -> set[int]:
    """/admin's home-games column for the whole user list at once (PERF-010: two
    queries in all, where ``_main_club_member`` per user cost two per user)."""
    cid = hg._main_club()
    return hg._club_member_ids(cid) if cid is not None else set()


def _admin_games_access(admin_uid: int, uid: int, grant: bool) -> bool:
    """/admin's home-games switch: add to / remove from the MAIN club (made on the
    first grant, owned by the granting admin). Returns membership afterwards."""
    cid = hg._main_club(create_for=admin_uid if grant else None)
    if cid is None:
        return False
    if grant:
        hg._add_member(cid, uid)
    elif hg._club_role(cid, uid) not in (None, "owner"):
        who = pub._user_by_id(uid)
        busy = hg._busy_in_club(cid, uid, hg._display_name(who) if who is not None else "They")
        if busy:
            raise HTTPException(status_code=409, detail=busy)
        pub.DB.q("DELETE FROM homegame_club_members WHERE club_id=? AND user_id=?", (cid, int(uid)))
    return hg._club_role(cid, uid) is not None
