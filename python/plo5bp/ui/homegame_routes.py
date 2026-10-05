"""The API: every /games/api route (HGB-002), its guard and request budgets (SEC-009).

Every route is a module-level function on ONE router (``install`` includes it).
The router's dependency runs first for every request: the signed-in viewer (else
the hidden 404) and their request budget. Every table route goes through
``_in_table``: the table's club only (``_table_access``), its lock, the
stale-request check where the client sends one, the action, then the viewer's view.

Split out of ``homegame`` (HGB-006) and imported by it: every name defined here is
re-exported as ``homegame.<name>``. The code reaches every other home-games name
through ``hg`` (the ``homegame`` module), looked up when it runs, so patching
``homegame.X`` in a test reaches this module too and ``homegame.use_context`` swaps
its state. Patch ``homegame.X``, never this module's copy.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import sys
import threading
import time
import weakref
from typing import Any, TYPE_CHECKING

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from plo5bp.ui import public as pub
from plo5bp.ui.ratelimit import RateLimiter

if TYPE_CHECKING:  # (annotations only)
    from plo5bp.ui.homegame import LiveTable

#: The home games' main module: every home-games name is looked up there when used.
hg = sys.modules.get("plo5bp.ui.homegame") or importlib.import_module("plo5bp.ui.homegame")
logger = logging.getLogger("plo5bp.ui.homegame")

__all__ = (
    "API_COST", "API_RATE", "AVATAR_RATE", "JOIN_RATE", "_RATE_SPEC", "_api_guard", "_in_table",
    "_limited", "_require_sync", "_scope_clubs", "_stream_sig", "_table_for", "_uid",
    "_user_or_404", "api_act", "api_auto_chips_self", "api_auto_stack", "api_auto_topup",
    "api_avatar", "api_bot", "api_bot_suggest", "api_chat", "api_clear_avatar", "api_close", "api_club", "api_club_archive",
    "api_club_create", "api_club_decide", "api_club_invite_reset", "api_club_leave",
    "api_club_member", "api_club_nickname", "api_club_request", "api_club_settings",
    "api_clubs", "api_community", "api_create", "api_deal", "api_exclude", "api_fair_commit",
    "api_fair_reveal", "api_fair_transcript", "api_get", "api_hand", "api_hand_choice", "api_hands",
    "api_hands_export", "api_health", "api_host_fold", "api_host_prefs", "api_invite",
    "api_invite_join", "api_kick", "api_leave", "api_list", "api_move", "api_my_hands",
    "api_my_name", "api_my_series", "api_my_stats", "api_player_hands", "api_player_series",
    "api_player_stats", "api_rabbit", "api_react", "api_rebuy", "api_receipt", "api_receipt_me",
    "api_remove_chips", "api_request", "api_run", "api_set_avatar", "api_set_my_name",
    "api_settings", "api_show", "api_sit", "api_sit_out", "api_sit_out_player", "api_stay",
    "api_stream", "api_street_pause", "api_transfer_host", "api_trust", "router",
)


def _require_sync(t: LiveTable, body: dict, *, with_seq: bool) -> None:
    """Reject a request composed against a state the table has left.

    (review 2026-09-20 G8) `/act` carried no hand or decision identity, so a
    click that arrived late landed on the NEXT decision — "Call $1" could
    call a different bet, and an armed pre-fold folded the next hand. The
    client echoes the ``hand_no`` (and for `/act` the ``action_seq``) of the
    state it was looking at; a mismatch is 409 and the client just re-renders.
    Optional so scripted callers keep working."""
    if not isinstance(body, dict):
        return
    if body.get("hand_no") is not None:
        if pub.body_int(body, "hand_no") != t.hand_no:
            raise HTTPException(status_code=409, detail="stale: the hand has moved on")
    if with_seq and body.get("action_seq") is not None:
        if pub.body_int(body, "action_seq") != t.action_seq:
            raise HTTPException(status_code=409, detail="stale: the action has moved on")


def _stream_sig(t: LiveTable) -> tuple:
    """Everything that can change what a viewer sees WITHOUT bumping ``rev``
    (the runout reveals itself by the clock; the event and reaction feeds; who is
    at the table — PERF-008: the heartbeat no longer resends the view, so a
    player arriving or leaving must change the signature). Cheap, lock-free reads."""
    now = time.monotonic()
    here = tuple(sorted(uid for uid, last in list(t.seen.items()) if now - last <= hg.PRESENCE_WINDOW_S))
    return (
        t.epoch, t.rev, t.phase, t.hand_no, t.action_seq, len(t.requests),
        hg._runout_shown_len(t) if t.runout_active else 0,
        hg._runout_settled_len(t) if t.runout_active else 0,  # (the equities follow it)
        hg._runout_award_index(t) if t.runout_active else 0,
        t.next_deal_mono is not None, t.bank_key, t.event_seq, t.reaction_seq, here,
    )


# --- the API (HGB-002 / SEC-009) ---------------------------------------------------------------
#
# Every /games/api route is a module-level function on ONE router (importable and
# testable on its own; ``install`` includes it). The router's dependency runs first
# for every request: the signed-in viewer (else the hidden 404) and their request
# budget (SEC-009). Every table route goes through ``_in_table``: the table's club
# only (``_table_access``), its lock, the stale-request check where the client
# sends one, the action, then the viewer's view — the access rule cannot be
# skipped by forgetting a line.

#: SEC-009: requests per user per second (sustained) and the burst on top, over the
#: whole /games/api (a browser polls a table ~2 a second at most; a runaway script
#: gets 429 + Retry-After). "0" = off. Off by default in the test session (scripted
#: clients hammer the API), where the tests that check it install their own.
_RATE_SPEC = os.environ.get(
    "PLO5BP_GAMES_RATE", "0" if "PYTEST_CURRENT_TEST" in os.environ else "20,200"
).split(",")
API_RATE: RateLimiter | None = (
    RateLimiter(rate=float(_RATE_SPEC[0]), burst=float(_RATE_SPEC[1] if len(_RATE_SPEC) > 1 else _RATE_SPEC[0]))
    if float(_RATE_SPEC[0] or 0) > 0 else None
)
#: What a request costs from that budget (reads that scan a lot cost more).
API_COST: dict[str, float] = {
    "/games/api/community": 5.0, "/games/api/my/stats": 5.0, "/games/api/my/hands": 3.0,
    "/games/api/players/{player_id}/stats": 5.0, "/games/api/players/{player_id}/hands": 3.0,
    "/games/api/tables/{game_id}/hands/export": 20.0, "/games/api/tables/{game_id}/hands": 2.0,
    "/games/api/my/series": 3.0, "/games/api/players/{player_id}/series": 3.0,
    "/games/api/tables/{game_id}/hands/{hand_no}/choice": 3.0,
    "/games/api/tables/{game_id}/bot/suggest": 3.0,
}
#: Join requests (each pops a toast at the club's open tables): 5, then one every 2 minutes.
JOIN_RATE = RateLimiter(rate=1 / 120.0, burst=5)
#: New profile pictures (up to 200 KB each into the database): 20, then one a minute.
AVATAR_RATE = RateLimiter(rate=1 / 60.0, burst=20)


def _limited(limiter: RateLimiter | None, key: Any, what: str, cost: float = 1.0) -> None:
    """429 with Retry-After when ``key`` is over ``limiter``'s budget."""
    if limiter is None:
        return
    ok, retry = limiter.allow(key, cost)
    if not ok:
        raise HTTPException(status_code=429, detail=what,
                            headers={"Retry-After": str(max(1, int(retry + 0.999)))})


def _uid() -> int:
    """The signed-in viewer (the access layer sets it); nobody = the hidden 404."""
    uid = pub._CURRENT_USER_ID.get()
    if uid is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return int(uid)


def _api_guard(request: Request) -> None:
    """The router's dependency: every /games/api request is signed in and within
    its user's request budget (SEC-009)."""
    uid = hg._uid()
    route = request.scope.get("route")
    hg._limited(hg.API_RATE, uid, "too many requests — give it a second",
             hg.API_COST.get(getattr(route, "path", ""), 1.0))


router = APIRouter(dependencies=[Depends(_api_guard)])


def _table_for(game_id: str) -> LiveTable:
    """Every table endpoint comes through here: members of the table's club only."""
    hg._ensure_workers()
    t = hg.CTX.hub.get(game_id)
    hg._table_access(t, hg._uid())
    return t


def _in_table(game_id: str, fn: Any, *, body: dict | None = None, sync: str | None = None,
              view: bool = True) -> Any:
    """The common path of a table route (HGB-002): the viewer; the table, for
    members of its club only; its lock; the stale-request check (``sync`` "hand"
    = the hand the client saw, "action" = the decision too); ``fn(t, uid)``; then
    the viewer's view (or ``fn``'s own answer with ``view=False``). ``fn`` reads
    the request's body itself: a stranger gets the club's 403 before anything
    about their body is looked at."""
    t = hg._table_for(game_id)
    uid = hg._uid()
    with t.lock:
        if sync is not None:
            hg._require_sync(t, body or {}, with_seq=(sync == "action"))
        out = fn(t, uid)
        return hg._view(t, uid) if view else out


def _user_or_404(uid: int) -> Any:
    user = pub._user_by_id(int(uid))
    if user is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return user


def _scope_clubs(uid: int, club: str | None, *, own: bool) -> list[str] | None:
    """Whose tables a stats query may read: one club the viewer is in; else, for
    the viewer's OWN numbers, everything they played (None); for somebody
    else's, only the clubs the viewer shares with them."""
    if club:
        row, _ = hg._require_club(club, uid)
        return [str(row["id"])]
    if own:
        return None
    return hg._my_club_ids(uid)


@router.get("/games/api/health")
def api_health():
    """Site admins: are the clock and the grader alive and keeping up (OPS-011)."""
    if not pub._is_admin(pub._user_by_id(hg._uid())):
        raise HTTPException(status_code=404, detail="Not Found")
    hg._ensure_workers()
    return hg.health()


@router.get("/games/api/tables")
def api_list(club: str | None = None):
    """The lobby: the viewer's clubs, one club's tables (plus the viewer's own
    seats elsewhere), their finished sessions there, and — for the club's
    owner and admins — who is asking to join. A fixed handful of queries
    whatever the number of clubs and tables (PERF-010)."""
    viewer = hg._uid()
    clubs = hg._my_clubs(viewer)
    by_id = {c["id"]: c for c in clubs}
    if club and club not in by_id:
        raise HTTPException(status_code=404, detail="Not Found")
    if not by_id:
        rows = []
    elif club:  # one club's tables, plus the viewer's own seats / hosting elsewhere
        rows = pub.DB.q(
            "SELECT * FROM homegames g WHERE g.status='open' AND (g.club_id=? OR (g.club_id IN (%s) AND ("
            "g.host_user_id=? OR EXISTS (SELECT 1 FROM homegame_players p WHERE p.game_id=g.id "
            "AND p.user_id=? AND p.seat IS NOT NULL)))) ORDER BY g.created_at DESC"
            % ",".join("?" * len(by_id)), tuple([club, *by_id, viewer, viewer]),
        )
    else:
        rows = pub.DB.q(
            "SELECT * FROM homegames WHERE status='open' AND club_id IN (%s) ORDER BY created_at DESC"
            % ",".join("?" * len(by_id)), tuple(by_id),
        )
    tables = []
    for row in hg._lobby_rows(rows, viewer):
        row["club_name"] = by_id[row["club_id"]]["name"]
        tables.append(row)
    manage = bool(club) and by_id[club]["role"] in ("owner", "admin")
    return {
        "clubs": clubs,
        "club": club,
        "tables": tables,
        "sessions": hg._my_sessions(viewer, club),
        "archived_clubs": hg._my_archived_clubs(viewer),  # (FEAT-007: the owner's, to restore)
        "join_requests": hg._club_requests(club) if manage else [],
        "my_avatar": hg._avatar_url(viewer),
        "my_name": hg._my_name_state(viewer),  # (FEAT-008)
        "client_build": hg.client_build(),  # (an older page offers a refresh — OPS-039)
        # the games this server deals — the client's create dialog is built from it (FE-004)
        "games": [dict(hg._game_info(code), available=bool(not hg.GAMES[code]["burns"] or hg.PLO67_ON)) for code in hg.GAMES],
    }


# --- clubs ----------------------------------------------------------------------------------------
@router.get("/games/api/clubs")
def api_clubs():
    return {"clubs": hg._my_clubs(hg._uid())}


@router.post("/games/api/clubs")
def api_club_create(body: dict = Body({})):
    uid = hg._uid()
    return hg._club_view(hg._create_club(hg._user_or_404(uid), (body or {}).get("name")), uid)


@router.get("/games/api/clubs/{club_id}")
def api_club(club_id: str):
    return hg._club_view(club_id, hg._uid())


@router.post("/games/api/clubs/{club_id}/settings")
def api_club_settings(club_id: str, body: dict = Body({})):
    return hg._club_settings(club_id, hg._uid(), body or {})


@router.post("/games/api/clubs/{club_id}/invite")
def api_club_invite_reset(club_id: str):
    """A new invite link (the old one stops working)."""
    return hg._reset_invite(club_id, hg._uid())


@router.post("/games/api/clubs/{club_id}/members")
def api_club_member(club_id: str, body: dict = Body({})):
    body = body or {}
    return hg._set_member(club_id, hg._uid(), pub.body_int(body, "user_id"), role=body.get("role"),
                       remove=hg._parse_bool(body, "remove", False))


@router.post("/games/api/clubs/{club_id}/archive")
def api_club_archive(club_id: str, body: dict = Body({})):
    """FEAT-007: the owner archives the club ({on: true}) or restores it ({on: false})."""
    return hg._archive_club(club_id, hg._uid(), hg._parse_bool(body or {}, "on", True))


@router.post("/games/api/clubs/{club_id}/leave")
def api_club_leave(club_id: str):
    hg._leave_club(club_id, hg._uid())
    return {"ok": True}


@router.post("/games/api/clubs/{club_id}/request")
def api_club_request(club_id: str):
    """Ask to join (from a table link: the club's id only travels in the answer
    a table link gives a non-member). The owner or an admin decides."""
    uid = hg._uid()
    club = hg._club(club_id)
    if club is None or club["archived_at"]:  # (an archived club takes nobody new — FEAT-007)
        raise HTTPException(status_code=404, detail="Not Found")
    cid = str(club["id"])
    if hg._club_role(cid, uid) is None and hg._request_state(cid, uid)[0] != "pending":
        # (each request pops a toast at the club's open tables — SEC-009)
        hg._limited(hg.JOIN_RATE, uid, "too many requests to join — try again in a little while")
    hg._request_club(cid, uid)
    status, retry = hg._request_state(cid, uid)
    return {"ok": True, "member": hg._club_role(cid, uid) is not None, "request": status,
            "retry_in": round(retry, 1)}


@router.post("/games/api/clubs/{club_id}/requests/decide")
def api_club_decide(club_id: str, body: dict = Body({})):
    return hg._decide_club_request(club_id, hg._uid(), pub.body_int(body, "user_id"),
                                hg._parse_bool(body or {}, "allow", False))  # HGB-020: "false" is no


@router.post("/games/api/clubs/{club_id}/nickname")
def api_club_nickname(club_id: str, body: dict = Body({})):
    """FEAT-006: {nickname, user_id?} — your name in this club (the owner and admins:
    anyone's). "" / null = none."""
    body = body or {}
    uid = hg._uid()
    target = pub.body_int(body, "user_id") if body.get("user_id") is not None else uid
    return hg._set_nickname(club_id, uid, target, body.get("nickname"))


@router.get("/games/api/invites/{code}")
def api_invite(code: str):
    return hg._invite_info(code, hg._uid())


@router.post("/games/api/invites/{code}/join")
def api_invite_join(code: str):
    uid = hg._uid()
    club = hg._club_by_code(code)
    if club is not None and bool(int(club["approve_joins"] or 0)) and hg._club_role(str(club["id"]), uid) is None:
        hg._limited(hg.JOIN_RATE, uid, "too many requests to join — try again in a little while")
    return hg._join_by_invite(code, uid)


# --- you ------------------------------------------------------------------------------------------
@router.get("/games/api/host_prefs")
def api_host_prefs():
    """The settings of the last table this user hosted (None = never)."""
    return {"prefs": hg._host_prefs(hg._uid())}


@router.get("/games/api/me/name")
def api_my_name():
    """FEAT-008: the name the tables show for you (and whether you chose it)."""
    return hg._my_name_state(hg._uid())


@router.post("/games/api/me/name")
def api_set_my_name(body: dict = Body({})):
    """{name}: your name at every table ("" / null = your account's name again)."""
    return hg._set_table_name(hg._uid(), (body or {}).get("name"))


@router.post("/games/api/me/avatar")
def api_set_avatar(body: dict = Body({})):
    """{data_url}: a PNG / JPEG / WebP picture (the page sends a 256 px square)."""
    uid = hg._uid()
    url = hg._set_avatar(uid, (body or {}).get("data_url") or "", before_write=lambda: hg._limited(
        hg.AVATAR_RATE, uid, "that's a lot of new pictures — try again in a minute"))
    hg._avatar_changed(uid)
    return {"avatar": url}


@router.delete("/games/api/me/avatar")
def api_clear_avatar():
    uid = hg._uid()
    url = hg._set_avatar(uid, None)
    hg._avatar_changed(uid)
    return {"avatar": url}


@router.get("/games/api/avatars/{user_id}")
def api_avatar(user_id: int):
    """Someone's picture — for them, or anyone sharing a club with them."""
    viewer = hg._uid()
    if not (int(user_id) == viewer or hg._shares_club(viewer, int(user_id))):
        raise HTTPException(status_code=404, detail="Not Found")
    row = pub.DB.one("SELECT mime, data FROM homegame_avatars WHERE user_id=?", (int(user_id),))
    if row is None or row["mime"] not in hg._AVATAR_MIMES:
        raise HTTPException(status_code=404, detail="Not Found")
    return Response(bytes(row["data"]), media_type=row["mime"], headers=hg.AVATAR_HEADERS)


# --- numbers --------------------------------------------------------------------------------------
@router.get("/games/api/my/hands")
def api_my_hands(sort: str = "time", dir: str = "desc", game: str | None = None,
                 offset: int = 0, limit: int = 40, club: str | None = None,
                 variant: str | None = None, since: str | None = None):
    """The signed-in player's hand database (one club's, or all of it; one
    game's — ``variant`` plo5 / plo6 / plo67 — or every game's). ``game`` = one table.
    ``since`` = one period (FEAT-004: ``_parse_since``)."""
    uid = hg._uid()
    return hg._my_hands(uid, sort, dir, game, offset, limit, clubs=hg._scope_clubs(uid, club, own=True),
                     variant=hg._game_filter(variant), since=hg._parse_since(since))


@router.get("/games/api/my/stats")
def api_my_stats(club: str | None = None, variant: str | None = None, since: str | None = None):
    uid = hg._uid()
    return hg._my_stats(uid, clubs=hg._scope_clubs(uid, club, own=True), variant=hg._game_filter(variant),
                     since=hg._parse_since(since))


@router.get("/games/api/community")
def api_community(club: str | None = None, variant: str | None = None, since: str | None = None):
    """One club's numbers for ONE game (``variant`` plo5 / plo6 / plo67; none = the
    club's most-played game): player cards, the pairwise money, all sessions.
    The rankings never mix clubs, nor games. (No club named: the main club,
    else the first.)"""
    uid = hg._uid()
    game = hg._game_filter(variant)
    if not club:
        mine = hg._my_club_ids(uid)
        if not mine:
            return {"club": None, "players": [], "pairs": [], "sessions": [], "can_manage": False,
                    "variant": game or hg.DEFAULT_GAME, "games": hg._games_summary({})}
        club = mine[0]
    row, role = hg._require_club(club, uid)
    return hg._community(uid, str(row["id"]), role == "owner", variant=game, since=hg._parse_since(since))


@router.get("/games/api/players/{player_id}/stats")
def api_player_stats(player_id: int, club: str | None = None, variant: str | None = None,
                     since: str | None = None):
    uid = hg._uid()
    clubs = hg._scope_clubs(uid, club, own=int(player_id) == uid)
    if pub._user_by_id(int(player_id)) is None or not hg._may_see_player(uid, int(player_id), club):
        raise HTTPException(status_code=404, detail="Not Found")  # SEC-002
    return hg._my_stats(int(player_id), clubs=clubs, variant=hg._game_filter(variant), since=hg._parse_since(since))


@router.get("/games/api/players/{player_id}/hands")
def api_player_hands(player_id: int, sort: str = "time", dir: str = "desc",
                     game: str | None = None, offset: int = 0, limit: int = 40, club: str | None = None,
                     variant: str | None = None, since: str | None = None):
    uid = hg._uid()
    clubs = hg._scope_clubs(uid, club, own=int(player_id) == uid)
    if not hg._may_see_player(uid, int(player_id), club):
        raise HTTPException(status_code=404, detail="Not Found")  # SEC-002
    return hg._my_hands(uid, sort, dir, game, offset, limit, player_id=int(player_id),
                     clubs=clubs, variant=hg._game_filter(variant), since=hg._parse_since(since))


# FEAT-012: the running net behind the profit graph, in the stats' own scope
@router.get("/games/api/my/series")
def api_my_series(game: str | None = None, club: str | None = None,
                  variant: str | None = None, since: str | None = None):
    uid = hg._uid()
    return hg._my_series(uid, clubs=hg._scope_clubs(uid, club, own=True), variant=hg._game_filter(variant),
                      game_id=game, since=hg._parse_since(since))


@router.get("/games/api/players/{player_id}/series")
def api_player_series(player_id: int, game: str | None = None, club: str | None = None,
                      variant: str | None = None, since: str | None = None):
    uid = hg._uid()
    clubs = hg._scope_clubs(uid, club, own=int(player_id) == uid)
    if pub._user_by_id(int(player_id)) is None or not hg._may_see_player(uid, int(player_id), club):
        raise HTTPException(status_code=404, detail="Not Found")  # (SEC-002, like their stats)
    return hg._my_series(int(player_id), clubs=clubs, variant=hg._game_filter(variant),
                      game_id=game, since=hg._parse_since(since))


# --- tables ---------------------------------------------------------------------------------------
@router.post("/games/api/tables")
def api_create(body: dict = Body({})):
    uid = hg._uid()
    user = hg._user_or_404(uid)
    t = hg._create_table(user, body or {})
    prefs = hg._host_prefs(uid) if (body or {}).get("remembered") else None
    with t.lock:
        if prefs:
            hg._apply_remembered_locked(t, uid, prefs)
        hg._remember_host_prefs(t)
        return hg._view(t, uid)


@router.get("/games/api/tables/{game_id}")
def api_get(game_id: str):
    return hg._in_table(game_id, lambda t, uid: None)


@router.post("/games/api/tables/{game_id}/sit")
def api_sit(game_id: str, body: dict = Body({})):
    def sit(t: LiveTable, uid: int) -> None:
        user = hg._user_or_404(uid)
        seat = pub.body_int(body, "seat", -1)
        buyin = hg._parse_cents(body, "buyin_cents", t.default_buyin_cents)
        if hg._needs_approval(t, uid):
            hg._request_locked(t, user, "sit", seat, buyin)
        else:
            hg._sit_locked(t, user, seat, buyin)
    return hg._in_table(game_id, sit)


@router.post("/games/api/tables/{game_id}/leave")
def api_leave(game_id: str, body: dict = Body({})):
    """Leave the seat. Holding cards: after this hand (the default) — or
    ``now`` = out of the hand at once (folded when facing a bet)."""
    return hg._in_table(game_id, lambda t, uid: hg._leave_locked(t, uid, now=hg._parse_bool(body or {}, "now", False)))


@router.post("/games/api/tables/{game_id}/sit_out")
def api_sit_out(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._sit_out_locked(
        t, uid, hg._parse_bool(body, "on", True), next_hand=hg._parse_bool(body, "next_hand", False)))


@router.post("/games/api/tables/{game_id}/settings")
def api_settings(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._settings_locked(t, uid, body or {}))


@router.post("/games/api/tables/{game_id}/transfer_host")
def api_transfer_host(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._transfer_host_locked(t, uid, pub.body_int(body, "user_id")))


@router.post("/games/api/tables/{game_id}/show")
def api_show(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, hg._show_locked, body=body, sync="hand")


@router.post("/games/api/tables/{game_id}/react")
def api_react(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._react_locked(t, uid, (body or {}).get("emote")))


@router.post("/games/api/tables/{game_id}/remove_chips")
def api_remove_chips(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._remove_chips_locked(
        t, uid, hg._parse_cents(body, "amount_cents"), queue_ok=hg._parse_bool(body, "queue", False)))


@router.post("/games/api/tables/{game_id}/stay")
def api_stay(game_id: str):
    return hg._in_table(game_id, hg._cancel_leave_locked)


@router.post("/games/api/tables/{game_id}/move")
def api_move(game_id: str, body: dict = Body({})):
    """Change seats between hands (FEAT-013): ``{"seat": n}``, an empty seat."""
    return hg._in_table(game_id, lambda t, uid: hg._move_locked(t, uid, pub.body_int(body, "seat", -1)))


@router.post("/games/api/tables/{game_id}/fair/commit")
def api_fair_commit(game_id: str, body: dict = Body({})):
    hg._in_table(game_id, lambda t, uid: hg._fair_commit_locked(
        t, uid, str(body.get("hand_id") or ""), str(body.get("commit") or "")), view=False)
    return {"ok": True}


@router.post("/games/api/tables/{game_id}/fair/reveal")
def api_fair_reveal(game_id: str, body: dict = Body({})):
    hg._in_table(game_id, lambda t, uid: hg._fair_reveal_locked(
        t, uid, str(body.get("hand_id") or ""), str(body.get("nonce") or "")), view=False)
    return {"ok": True}


@router.get("/games/api/tables/{game_id}/fair/{hand_no}")
def api_fair_transcript(game_id: str, hand_no: int):
    return hg._in_table(game_id, lambda t, uid: hg._fair_transcript(t, uid, int(hand_no)), view=False)


@router.post("/games/api/tables/{game_id}/exclude")
def api_exclude(game_id: str, body: dict = Body({})):
    def exclude(t: LiveTable, uid: int) -> dict[str, Any]:
        on = hg._parse_bool(body, "on", True)
        hg._exclude_locked(t, pub._user_by_id(uid), on)
        return {"ok": True, "id": t.game_id, "excluded": on}
    return hg._in_table(game_id, exclude, view=False)


@router.get("/games/api/tables/{game_id}/hands")
def api_hands(game_id: str, before: int | None = None, limit: int = 30):
    # SEC-008: any member of the table's club (checked by _in_table) may browse
    # its hands — the same hands their stats cards already open, one player at a
    # time. The CARDS still follow the table's reveal rule.
    return hg._in_table(game_id, lambda t, uid: hg._hands_list(t, uid, before, limit), view=False)


@router.get("/games/api/tables/{game_id}/hands/export")
def api_hands_export(game_id: str, format: str = "txt"):
    """The session's hands as a file (FEAT-003) — any member of the club, like
    the hand list; the cards follow the table's reveal rule. (Registered before
    /hands/{hand_no}, which would otherwise take "export" for a hand number; its
    own function holds the table's lock only to read what it needs.)"""
    return hg._hands_export(hg._table_for(game_id), hg._uid(), format)


@router.get("/games/api/tables/{game_id}/ledger/me")
def api_receipt_me(game_id: str):
    """Your money at this table, movement by movement (FEAT-002)."""
    return hg._in_table(game_id, lambda t, uid: hg._receipt(t, uid, uid), view=False)


@router.get("/games/api/tables/{game_id}/ledger/{user_id}")
def api_receipt(game_id: str, user_id: int):
    """The host: any player's receipt at the table (FEAT-002 B)."""
    return hg._in_table(game_id, lambda t, uid: hg._receipt(t, uid, int(user_id)), view=False)


@router.get("/games/api/tables/{game_id}/hands/{hand_no}")
def api_hand(game_id: str, hand_no: int):
    return hg._in_table(game_id, lambda t, uid: hg._hand_detail(t, uid, hand_no), view=False)


@router.get("/games/api/tables/{game_id}/hands/{hand_no}/choice")
def api_hand_choice(game_id: str, hand_no: int, i: int):
    """What the network plays at decision ``i`` of a stored hand (2026-10-03): the
    replayer shows it without a trip to Study. Only where the viewer may see the
    actor's cards (the replayer's own rule: your hands, hands shown down), and with
    Study's access (``public._entitled``)."""
    from plo5bp.ui import handreview_store as hrs

    rec = hg._in_table(game_id, lambda t, uid: hg._hand_detail(t, uid, hand_no), view=False)
    if not pub._entitled(pub._user_by_id(hg._uid())):
        raise HTTPException(status_code=402, detail={
            "error": "subscription_required", "message": "The network's choice needs a subscription, like Study."})
    return hrs.choice(rec, i)


@router.post("/games/api/tables/{game_id}/sit_out_player")
def api_sit_out_player(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._sit_out_locked(
        t, uid, hg._parse_bool(body, "on", True), target_uid=pub.body_int(body, "user_id")))


@router.post("/games/api/tables/{game_id}/kick")
def api_kick(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._kick_locked(t, uid, pub.body_int(body, "user_id")))


@router.post("/games/api/tables/{game_id}/street_pause")
def api_street_pause(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._set_street_pause_locked(t, uid, hg._parse_secs(body, "secs", 1.5)))


@router.post("/games/api/tables/{game_id}/rebuy")
def api_rebuy(game_id: str, body: dict = Body({})):
    def rebuy(t: LiveTable, uid: int) -> None:
        amount = hg._parse_cents(body, "amount_cents")
        queue_ok = hg._parse_bool(body, "queue", False)
        if hg._needs_approval(t, uid):
            hg._request_locked(t, hg._user_or_404(uid), "rebuy", None, amount)
        else:
            hg._topup_locked(t, uid, amount, queue_ok=queue_ok)
    return hg._in_table(game_id, rebuy)


@router.post("/games/api/tables/{game_id}/request")
def api_request(game_id: str, body: dict = Body({})):
    """Host: approve / decline a pending buy-in. Requester: cancel their own."""
    body = body or {}
    action = str(body.get("action") or "").strip().lower()

    def resolve(t: LiveTable, uid: int) -> None:
        if action == "cancel":
            hg._cancel_request_locked(t, uid)
        elif action not in ("approve", "deny"):
            raise HTTPException(status_code=400, detail="action must be approve, deny or cancel")
        else:
            hg._resolve_request_locked(
                t, uid, pub.body_int(body, "id"), action == "approve",
                hg._parse_bool(body, "trust", False),
                amount_cents=hg._parse_cents(body, "amount_cents") if body.get("amount_cents") is not None else None,
            )
    return hg._in_table(game_id, resolve)


@router.post("/games/api/tables/{game_id}/trust")
def api_trust(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._set_trust_locked(
        t, uid, pub.body_int(body, "user_id"), hg._parse_bool(body, "on", True)))


@router.post("/games/api/tables/{game_id}/auto_topup")
def api_auto_topup(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._auto_topup_host_locked(t, uid, body or {}))


@router.post("/games/api/tables/{game_id}/auto_chips_self")
def api_auto_chips_self(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._auto_chips_self_locked(t, uid, body or {}))


@router.post("/games/api/tables/{game_id}/auto_stack")
def api_auto_stack(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._auto_stack_host_locked(t, uid, body or {}))


@router.post("/games/api/tables/{game_id}/run")
def api_run(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._set_running_locked(t, uid, hg._parse_bool(body, "running", True)))


@router.post("/games/api/tables/{game_id}/deal")
def api_deal(game_id: str, body: dict = Body({})):
    # `hand_no` = the finished hand the client saw; 409 if another player (or the
    # server's auto-deal) already dealt the next one.
    return hg._in_table(game_id, hg._player_deal_locked, body=body, sync="hand")


@router.post("/games/api/tables/{game_id}/act")
def api_act(game_id: str, body: dict = Body({})):
    # (review 2026-09-20 G8) ONE lock acquisition: the raise-TO -> raise-BY
    # conversion used to read the actor's street commit under one `with
    # t.lock`, release, and act under a second — another action could land in
    # between and the conversion applied to the wrong node.
    def act(t: LiveTable, uid: int) -> None:
        gate = hg._gate_key(body.get("gate"))
        # HGB-019: "allin" / "all_in" is the maximum raise (it used to clamp an
        # absent amount UP to the minimum raise); the engine caps it at max_raise.
        all_in = str(body.get("gate")).strip().lower().replace("-", "_") in ("allin", "all_in")
        raise_to = (
            pub.body_int(body, "raise_to_chips")
            if gate == hg.GATE_RAISE and body.get("raise_to_chips") is not None
            else None
        )
        by = pub.body_int(body, "chips", 0)
        if all_in:
            by = hg.MAX_CENTS * hg.BB_CHIPS  # (more than any stack: clamped to max_raise)
        elif raise_to is not None:
            if t.env is None or t.env.current_actor() is None:
                raise HTTPException(status_code=400, detail="no hand in progress")
            raw = hg._obs_dict(t.env)
            # Optional raise-TO total in chips -> the engine's raise-BY.
            by = raise_to - int(raw["street_commit"][int(raw["actor"])])
        hg._act_locked(t, uid, gate, by)
    return hg._in_table(game_id, act, body=body, sync="action")


@router.post("/games/api/tables/{game_id}/bot")
def api_bot(game_id: str, body: dict = Body({})):
    """The site's owner: the network at their seat (homegame_bot) — ``mode`` "off",
    "assist" (it shows its move) or "auto" (it plays); ``mix`` = its full strategy."""
    body = body or {}
    mode, mix = body.get("mode"), hg._parse_bool(body, "mix", False)
    return hg._in_table(game_id, lambda t, uid: hg._set_bot_locked(t, uid, mode, mix))


@router.get("/games/api/tables/{game_id}/bot/suggest")
def api_bot_suggest(game_id: str):
    """The network's move for the owner's turn at a seat it assists (homegame_bot)."""
    t = hg._table_for(game_id)
    return hg._bot_suggestion(t, hg._uid())


@router.post("/games/api/tables/{game_id}/host_fold")
def api_host_fold(game_id: str):
    return hg._in_table(game_id, hg._host_fold_locked)


@router.post("/games/api/tables/{game_id}/rabbit")
def api_rabbit(game_id: str):
    return hg._in_table(game_id, hg._rabbit_locked)


@router.post("/games/api/tables/{game_id}/chat")
def api_chat(game_id: str, body: dict = Body({})):
    return hg._in_table(game_id, lambda t, uid: hg._chat_add(t, hg._user_or_404(uid), (body or {}).get("text", "")))


@router.post("/games/api/tables/{game_id}/close")
def api_close(game_id: str):
    return hg._in_table(game_id, hg._close_locked)


@router.get("/games/api/tables/{game_id}/stream")
async def api_stream(game_id: str, max_events: int = 0):
    """Live push (SSE): the viewer's state whenever it changes. Replaces the
    450 ms poll (which stays as the fallback). When nothing changes, a heartbeat
    every STREAM_HEARTBEAT_S is just an SSE comment (``: ping``) that keeps the
    connection open and marks the viewer present (PERF-008: it used to rebuild and
    resend the whole view — chat, events, ledger — to every viewer). At most
    MAX_STREAMS_PER_USER open at once per user (SEC-009). ``max_events`` (tests)
    counts pushes and pings."""
    cur = [hg._table_for(game_id)]  # the table object being streamed
    uid = hg._uid()
    if not hg.CTX.streams.enter(uid):
        raise HTTPException(status_code=429, detail="too many live connections — close a tab or two",
                            headers={"Retry-After": "5"})
    freed = threading.Lock()

    def release() -> None:  # (exactly once: the stream ended, or it never started)
        if freed.acquire(blocking=False):
            hg.CTX.streams.leave(uid)

    def snapshot() -> tuple[tuple, str | None]:
        # Every push resolves the table through the hub, exactly like a poll
        # (OPS-001): it counts as activity — a table watched only through its
        # stream used to be evicted after HUB_IDLE_EVICT_S while the streams
        # kept serving the dead copy, and the next request loaded a second one
        # that nobody saw — and a table the process reloaded anyway is followed
        # (the new copy's first push carries its new ``epoch``).
        try:
            t = hg.CTX.hub.get(game_id)
        except HTTPException:
            return (), None  # the table is gone
        cur[0] = t
        # (removed from the club mid-stream: the push ends, and the client's
        # next poll gets the "ask to join" answer)
        try:
            hg._table_access(t, uid)
        except HTTPException:
            return (), None
        with t.lock:
            view = hg._view(t, uid)
            return hg._stream_sig(t), json.dumps(view, separators=(",", ":"))

    def touch() -> str | None:
        # The heartbeat's work (PERF-008): the table is resolved through the hub
        # like a push (activity; a reloaded table is followed), the viewer must
        # still belong to its club, and the viewer counts as present. "reloaded" =
        # the hub holds a new copy: the next pass pushes its view (new epoch).
        try:
            t = hg.CTX.hub.get(game_id)
        except HTTPException:
            return None
        reloaded = t is not cur[0]
        cur[0] = t
        try:
            hg._table_access(t, uid)
        except HTTPException:
            return None
        with t.lock:
            t.seen[uid] = time.monotonic()
        return "reloaded" if reloaded else "ok"

    async def gen():
        try:
            sent = 0
            last_sig: tuple | None = None
            last_push = 0.0
            yield "retry: 1500\n\n"
            while not hg.CTX.streams_stop.is_set():
                now = time.monotonic()
                if hg._stream_sig(cur[0]) != last_sig:
                    last_sig, payload = await run_in_threadpool(snapshot)
                    if payload is None:
                        return
                    last_push = now
                    yield f"data: {payload}\n\n"
                    sent += 1
                elif now - last_push >= hg.STREAM_HEARTBEAT_S:
                    state = await run_in_threadpool(touch)
                    if state is None:
                        return
                    if state == "reloaded":
                        continue  # (the next pass pushes the new copy's view)
                    last_push = now
                    yield ": ping\n\n"
                    sent += 1
                else:
                    await asyncio.sleep(hg.STREAM_TICK_S)
                    continue
                if max_events and sent >= max_events:
                    return
                await asyncio.sleep(hg.STREAM_TICK_S)
        finally:
            release()

    stream = gen()
    weakref.finalize(stream, release)  # (a response dropped before its first chunk frees the slot too)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
