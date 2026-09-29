"""The record: hand histories, everyone's numbers, receipts and the lobby's rows.

Read-only. Every query follows the privacy rules of the live table (a viewer sees
the cards they could have seen, and the marks that follow them), counts one club's
tables at a time, leaves out excluded sessions and never serves a hand its table
has not published yet (``_unpublished_guard``).

Split out of ``homegame`` (HGB-006) and imported by it: every name defined here is
re-exported as ``homegame.<name>``. The code reaches every other home-games name
through ``hg`` (the ``homegame`` module), looked up when it runs, so patching
``homegame.X`` in a test reaches this module too and ``homegame.use_context`` swaps
its state. Patch ``homegame.X``, never this module's copy.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING

from fastapi import HTTPException
from fastapi.responses import Response

from plo5bp.ui import homegame_export, public as pub

if TYPE_CHECKING:  # (annotations only)
    from plo5bp.ui.homegame import LiveTable

#: The home games' main module: every home-games name is looked up there when used.
hg = sys.modules.get("plo5bp.ui.homegame") or importlib.import_module("plo5bp.ui.homegame")
logger = logging.getLogger("plo5bp.ui.homegame")

__all__ = (
    "SERIES_BUCKETS", "_MY_SORTS", "_PERIOD_DAYS", "_club_clause", "_community",
    "_games_played", "_games_summary", "_hand_detail", "_hand_for_viewer", "_hands_export",
    "_hands_list", "_hands_visible_upto", "_lobby_rows", "_my_hands", "_my_series",
    "_my_sessions", "_my_stats", "_parse_since", "_period_clause", "_receipt", "_seat_accuracy",
    "_session_results", "_table_h2h", "_unpublished_guard",
)


def _hand_for_viewer(
    rec: dict[str, Any], viewer_id: int, show_all_grades: bool = True
) -> dict[str, Any]:
    """A stored hand as THIS viewer may see it: own cards, plus hands tabled
    at showdown or shown voluntarily. Everything else is face-down — same
    rule as the live table (review 2026-09-20 G1/G2).

    Grades follow the cards (owner, 2026-09-26): you see the network's marks
    on your own actions and on the actions of a hand you can see (tabled at
    showdown or shown) — a mucked hand's marks would say what it was. The
    table's ``show_grades`` off = your own marks only. Every grade still counts
    in everyone's accuracy (`homegame_hand_results`); only the display hides."""
    out = dict(rec)
    seats = []
    for s in rec.get("seats") or []:
        s2 = dict(s)
        mine = int(s.get("user_id") or -1) == int(viewer_id)
        if not (mine or s.get("shown")):
            s2["hole"] = None
            if "hole_seq" in s2:  # (PLO67: the deal order would say the same cards)
                s2["hole_seq"] = None
        s2["is_me"] = mine
        s2["avatar"] = hg._avatar_url(s.get("user_id"))
        s2.pop("user_id", None)
        seats.append(s2)
    out["seats"] = seats
    my_seats = {int(s["seat"]) for s in seats if s.get("is_me")}
    seen = my_seats | {int(s["seat"]) for s in seats if s.get("shown")}
    grades = rec.get("grades")
    if grades is not None:
        keep = seen if show_all_grades else my_seats
        grades = [g for g in grades if int(g.get("seat", -1)) in keep]
    out["grades"] = grades
    out["grades_public"] = bool(show_all_grades)
    return out


def _hands_visible_upto(t: LiveTable) -> int:
    """Newest hand number whose result may be served: never the hand being
    played, never one whose runout is still revealing (G11). Safe to call
    without the table lock (``_unpublished_guard`` does)."""
    if t.phase == "in_hand" or hg._runout_blocking(t):
        return int(t.hand_no) - 1
    return int(t.hand_no)


def _hands_list(t: LiveTable, viewer_id: int, before: int | None, limit: int) -> dict[str, Any]:
    upto = hg._hands_visible_upto(t)
    if before is not None:
        upto = min(upto, int(before) - 1)
    limit = max(1, min(hg.HANDS_PAGE_MAX, int(limit)))
    rows = pub.DB.q(
        "SELECT hand_no, ended_at, pot_cents, summary FROM homegame_hands "
        "WHERE game_id=? AND hand_no<=? ORDER BY hand_no DESC LIMIT ?",
        (t.game_id, upto, limit),
    )
    hands = []
    for r in rows:
        try:
            rec = hg._hand_for_viewer(json.loads(r["summary"]), viewer_id, t.show_grades)
        except Exception:  # noqa: BLE001
            continue
        me = next((s for s in rec["seats"] if s.get("is_me")), None)
        hands.append({
            "hand_no": int(r["hand_no"]),
            "ended_at": r["ended_at"],
            "pot_cents": int(r["pot_cents"]),
            "showdown": bool(rec.get("showdown")),
            "board_a": rec.get("board_a") or [],
            "board_b": rec.get("board_b") or [],
            "winners": [
                {"name": s["name"], "delta_cents": s["delta_cents"]}
                for s in rec["seats"] if int(s.get("delta_cents") or 0) > 0
            ],
            "my_delta_cents": int(me["delta_cents"]) if me else None,
            "my_hole": me.get("hole") if me else None,
            "my_accuracy": hg._seat_accuracy(rec.get("grades"), me["seat"]) if me else None,
        })
    stats = pub.DB.q(
        "SELECT r.user_id, COUNT(*) hands, SUM(CASE WHEN r.delta_cents>0 THEN 1 ELSE 0 END) wins, "
        "SUM(r.showdown) showdowns, MAX(r.delta_cents) biggest, SUM(r.acc_sum) acc_sum, "
        "SUM(r.acc_n) acc_n, u.name, u.email "
        "FROM homegame_hand_results r JOIN users u ON u.id=r.user_id "
        "WHERE r.game_id=? AND r.hand_no<=? GROUP BY r.user_id ORDER BY hands DESC",
        (t.game_id, hg._hands_visible_upto(t)),
    )
    return {
        "hands": hands,
        # (from the rows read, not the ones shown: a record that fails to parse
        # must not end the list — PERF-011)
        "more": bool(rows) and int(rows[-1]["hand_no"]) > 1 and len(rows) == limit,
        "stats": [
            {
                "name": hg._display_name(r, t.club_id),
                "user_id": int(r["user_id"]),  # (the client matches on this, not the name — HGT-027)
                "is_me": int(r["user_id"]) == int(viewer_id),
                "hands": int(r["hands"] or 0),
                "wins": int(r["wins"] or 0),
                "showdowns": int(r["showdowns"] or 0),
                "biggest_win_cents": max(0, int(r["biggest"] or 0)),
                "accuracy": (
                    round(float(r["acc_sum"] or 0) / int(r["acc_n"]), 1)
                    if int(r["acc_n"] or 0) else None
                ),
                "graded": int(r["acc_n"] or 0),
            }
            for r in stats
        ],
        "h2h": hg._table_h2h(t),
    }


def _seat_accuracy(grades: list | None, seat: int) -> float | None:
    mine = [float(g["score"]) for g in (grades or []) if int(g.get("seat", -1)) == int(seat)]
    return round(sum(mine) / len(mine), 1) if mine else None


def _table_h2h(t: LiveTable) -> list[dict[str, Any]]:
    """Net money between every pair at this table: ``to`` is up ``cents`` on
    ``from``. Hands still revealing are excluded (G11)."""
    rows = pub.DB.q(
        "SELECT payer, payee, SUM(chips) chips FROM homegame_flows "
        "WHERE game_id=? AND hand_no<=? GROUP BY payer, payee",
        (t.game_id, hg._hands_visible_upto(t)),
    )
    gross = {(int(r["payer"]), int(r["payee"])): int(r["chips"] or 0) for r in rows}
    names: dict[int, str] = {}
    out = []
    for (a, b), v in gross.items():
        net = v - gross.get((b, a), 0)
        if net <= 0:
            continue
        for uid in (a, b):
            if uid not in names:
                u = pub._user_by_id(uid)
                names[uid] = hg._display_name(u, t.club_id) if u is not None else f"Player {uid}"
        out.append({"from": names[a], "to": names[b],
                    "cents": hg.chips_to_cents(net, t.bb_cents)})
    out.sort(key=lambda x: -x["cents"])
    return out


# --- the club: everyone's numbers, side by side ------------------------------------------
#
# Home games are a private circle (admin-granted), so stats are open inside it:
# every member sees every player's accuracy, profit and loss, the money between
# every pair, and can browse anyone's hands (cards still follow the reveal
# rule). Sessions an admin took out of the record (``homegames.excluded`` —
# test tables) count nowhere; that is a soft flag and can be undone.


def _unpublished_guard(game_col: str = "r.game_id", hand_col: str = "r.hand_no") -> tuple[str, list[Any]]:
    """SQL (`` AND NOT (game=? AND hand>?)`` per open loaded table) that leaves
    out every hand a table has not published yet — the hand being played and one
    whose runout is still revealing — so no aggregate (stats, head-to-head,
    history) shows a result a few seconds early (G11; SEC-003: ``_my_stats`` had
    no guard at all).

    Lock-free (PERF-009: it used to take every loaded table's lock in turn, so one
    busy table stalled every stats page). ``_hands_visible_upto`` is safe to read
    without the lock because of the order the table's state changes in: a deal
    flips ``phase`` before ``hand_no``, and a hand's end sets up its runout
    (``_capture_rabbit``) before ``phase`` leaves "in_hand" and before any result
    is written — so a torn read only ever hides a hand a moment longer. And a
    hand that ends while the stats are being read has a higher number than the
    threshold read before it, so it is left out too (the old list of revealing
    hands, read first, missed exactly that hand)."""
    with hg.CTX.hub._lock:
        tables = list(hg.CTX.hub._tables.values())
    pairs = [(tb.game_id, hg._hands_visible_upto(tb)) for tb in tables if tb.status == "open"]
    return (
        "".join(f" AND NOT ({game_col}=? AND {hand_col}>?)" for _ in pairs),
        [x for pair in pairs for x in pair],
    )


def _club_clause(clubs: list[str] | None, col: str = "g.club_id") -> tuple[str, list[Any]]:
    """`` AND g.club_id IN (...)`` for a stats scope (None = no restriction)."""
    if clubs is None:
        return "", []
    if not clubs:
        return " AND 0", []
    return f" AND {col} IN ({','.join('?' * len(clubs))})", list(clubs)


#: FEAT-004: the stats periods the API names (anything else: an ISO date/time).
_PERIOD_DAYS = {"7d": 7, "30d": 30, "90d": 90, "365d": 365}


def _parse_since(v: Any) -> str | None:
    """A stats period's start as the stored time format (UTC ISO, seconds), or None
    for all time: ``all`` / ``7d`` / ``30d`` / ``90d`` / ``365d`` / ``month`` (this
    calendar month, UTC) / ``year`` — or an ISO date/time, which is how a browser
    asks for "this month" in its own time zone. Anything else is a 400."""
    s = str(v or "").strip()
    if s.lower() in ("", "all"):
        return None
    now = datetime.now(timezone.utc)
    if s.lower() in hg._PERIOD_DAYS:
        dt = now - timedelta(days=hg._PERIOD_DAYS[s.lower()])
    elif s.lower() == "month":
        dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif s.lower() == "year":
        dt = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError as e:
            raise HTTPException(status_code=400, detail="since must be all, 7d, 30d, 90d, 365d, month, "
                                                        "year or a date") from e
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _period_clause(since: str | None) -> tuple[str, list[Any]]:
    """`` AND <the result's hand ended at or after since>`` over ``r`` (the same
    ``r.game_id`` / ``r.hand_no`` spelling as ``_unpublished_guard``, so the flows'
    ``.replace`` works for it too); no clause for all time. A primary-key lookup
    per result row."""
    if since is None:
        return "", []
    return (" AND (SELECT h2.ended_at FROM homegame_hands h2 WHERE h2.game_id=r.game_id "
            "AND h2.hand_no=r.hand_no)>=?", [since])


def _games_played(where: str, args: list[Any]) -> dict[str, int]:
    """Hands per game in a stats scope (``where`` over ``g`` = homegames, ``r`` =
    homegame_hand_results): which game switches a stats view offers."""
    out: dict[str, int] = {}
    for r in pub.DB.q(
        "SELECT g.variant v, COUNT(DISTINCT r.game_id || ':' || r.hand_no) n "
        f"FROM homegame_hand_results r JOIN homegames g ON g.id=r.game_id WHERE {where} GROUP BY g.variant",
        tuple(args),
    ):
        code = hg._norm_game(r["v"])
        out[code] = out.get(code, 0) + int(r["n"] or 0)
    return out


def _session_results(where: str, args: list[Any], since: str | None = None) -> list[dict[str, Any]]:
    """Each (player, table) in a stats scope (``where`` over ``r`` =
    homegame_hand_results and ``g`` = homegames): hands, wins, accuracy, the best
    hand, when they last played — and the EXACT net in cents (OPS-015):

    - a CLOSED session's net is its ledger (money out − money in): exactly what
      the players settle and what the lobby shows for it;
    - an open one's is the sum of its published hands in CHIPS, turned into cents
      once for the table (hands recorded before chips were stored add their
      per-hand cents).
    Per-hand cents used to be summed across hands, so a long session's stats
    drifted from its ledger by a cent every few hands.

    ``since`` (a stats period, FEAT-004 — ``where`` already filters its hands): a
    session that began before it is only partly in the period, so it is summed in
    chips too; the ledger cannot be split by time."""
    rows = pub.DB.q(
        "SELECT r.user_id, r.game_id, g.name, g.variant, g.status, g.sb_cents, g.bb_cents, g.ante_cents, "
        "COUNT(*) hands, SUM(CASE WHEN r.delta_cents>0 THEN 1 ELSE 0 END) wins, "
        "COALESCE(SUM(r.acc_sum),0) a, COALESCE(SUM(r.acc_n),0) n, MAX(r.delta_cents) best, "
        "COALESCE(SUM(r.delta_chips),0) chips, "
        "COALESCE(SUM(CASE WHEN r.delta_chips IS NULL THEN r.delta_cents END),0) legacy, "
        "MAX(h.ended_at) last, p.buyin_cents, p.leftover_cents, "
        "(SELECT MIN(hh.ended_at) FROM homegame_hands hh WHERE hh.game_id=r.game_id) first "
        "FROM homegame_hand_results r JOIN homegames g ON g.id=r.game_id "
        "LEFT JOIN homegame_hands h ON h.game_id=r.game_id AND h.hand_no=r.hand_no "
        "LEFT JOIN homegame_players p ON p.game_id=r.game_id AND p.user_id=r.user_id "
        f"WHERE {where} GROUP BY r.user_id, r.game_id",
        tuple(args),
    )
    out = []
    for r in rows:
        whole = since is None or (r["first"] is not None and str(r["first"]) >= since)
        if r["status"] == "closed" and r["buyin_cents"] is not None and whole:
            net = int(r["leftover_cents"]) - int(r["buyin_cents"])
        else:
            net = hg.chips_to_cents(int(r["chips"]), int(r["bb_cents"])) + int(r["legacy"])
        out.append({
            "user_id": int(r["user_id"]), "game_id": r["game_id"], "name": r["name"],
            "variant": hg._norm_game(r["variant"]), "open": r["status"] == "open",
            "sb_cents": int(r["sb_cents"]), "bb_cents": int(r["bb_cents"]), "ante_cents": int(r["ante_cents"]),
            "hands": int(r["hands"] or 0), "wins": int(r["wins"] or 0),
            "acc_sum": float(r["a"] or 0), "acc_n": int(r["n"] or 0),
            "best_cents": int(r["best"] or 0), "last": r["last"], "net_cents": net,
        })
    return out


def _games_summary(played: dict[str, int]) -> list[dict[str, Any]]:
    return [{"code": c, "label": g["label"], "hands": int(played.get(c, 0)), "graded": bool(g["graded"])}
            for c, g in hg.GAMES.items()]


def _community(viewer_id: int, club_id: str, can_manage: bool,
               variant: str | None = None, since: str | None = None) -> dict[str, Any]:
    """ONE club's numbers (rankings never mix clubs) for ONE game: PLO5, PLO6 and
    PLO67 are different games, so their profits, accuracy and head-to-head are kept
    apart (2026-09-26). ``variant`` None = the game the club has played most.
    ``can_manage`` = the club's owner: sees excluded sessions and may exclude /
    restore them. ``since`` = only hands from then on (FEAT-004: this month, the
    last 30 days …); the sessions list stays whole (it is for managing them)."""
    guard, gargs = hg._unpublished_guard()
    period, pargs = hg._period_clause(since)
    guard, gargs = guard + period, gargs + pargs
    played = hg._games_played(f"g.excluded=0 AND g.club_id=?{guard}", [club_id] + gargs)
    if variant is None:
        variant = max(hg.GAMES, key=lambda c: (played.get(c, 0), c == hg.DEFAULT_GAME))
    per: dict[int, dict[str, Any]] = {}
    for s in hg._session_results(f"g.excluded=0 AND g.club_id=? AND g.variant=?{guard}",
                              [club_id, variant] + gargs, since=since):
        agg = per.setdefault(s["user_id"], {"hands": 0, "wins": 0, "sessions": 0, "net": 0,
                                            "best": 0, "a": 0.0, "n": 0})
        agg["hands"] += s["hands"]
        agg["wins"] += s["wins"]
        agg["sessions"] += 1
        agg["net"] += s["net_cents"]  # (exact per session — OPS-015)
        agg["best"] = max(agg["best"], s["best_cents"])
        agg["a"] += s["acc_sum"]
        agg["n"] += s["acc_n"]
    users = {int(r["id"]): r for r in pub.DB.q(
        f"SELECT id, name, email FROM users WHERE id IN ({','.join('?' * len(per))})", tuple(per),
    )} if per else {}
    players = [
        {
            "user_id": uid, "name": hg._display_name(users[uid], club_id) if uid in users else f"Player {uid}",
            "avatar": hg._avatar_url(uid),
            "is_me": uid == int(viewer_id),
            "hands": a["hands"], "wins": a["wins"],
            "sessions": a["sessions"], "net_cents": a["net"],
            "best_cents": max(0, a["best"]),
            "accuracy": round(a["a"] / a["n"], 1) if a["n"] else None,
            "graded": a["n"],
        }
        for uid, a in per.items()
    ]
    players.sort(key=lambda x: -x["net_cents"])
    fguard = guard.replace("r.game_id", "f.game_id").replace("r.hand_no", "f.hand_no")
    gross: dict[tuple[int, int], float] = {}
    for r in pub.DB.q(
        "SELECT f.payer, f.payee, SUM(f.chips * g.bb_cents) v FROM homegame_flows f "
        f"JOIN homegames g ON g.id=f.game_id WHERE g.excluded=0 AND g.club_id=? AND g.variant=?{fguard} "
        "GROUP BY f.payer, f.payee", tuple([club_id, variant] + gargs),
    ):
        gross[(int(r["payer"]), int(r["payee"]))] = float(r["v"] or 0) / hg.BB_CHIPS
    pairs = []
    for (a, b), v in gross.items():
        net = v - gross.get((b, a), 0.0)
        if net > 0.5:  # b is up `cents` on a
            pairs.append({"from": a, "to": b, "cents": int(round(net))})
    sess = pub.DB.q(
        "SELECT g.id, g.name, g.variant, g.status, g.excluded, g.created_at, g.closed_at, g.hand_no, "
        "g.sb_cents, g.bb_cents, g.ante_cents, "
        "(SELECT COUNT(*) FROM homegame_players p WHERE p.game_id=g.id AND p.buyin_cents>0) players "
        "FROM homegames g WHERE g.club_id=? " + ("" if can_manage else "AND g.excluded=0 ") +
        "ORDER BY g.created_at DESC LIMIT 300", (club_id,),
    )
    return {
        "club": club_id, "players": players, "pairs": pairs, "since": since,
        # the game these numbers are for, and every game with its hand count (the switch)
        "variant": variant, "games": hg._games_summary(played),
        "can_manage": bool(can_manage),
        # (every session of the club, each tagged with its game — a list, not a total)
        "sessions": [
            {"id": r["id"], "name": r["name"], "variant": hg._norm_game(r["variant"]),
             "open": r["status"] == "open",
             "excluded": bool(int(r["excluded"] or 0)), "created_at": r["created_at"],
             "closed_at": r["closed_at"], "hands": int(r["hand_no"] or 0),
             "players": int(r["players"] or 0), "sb_cents": int(r["sb_cents"]),
             "bb_cents": int(r["bb_cents"]), "ante_cents": int(r["ante_cents"])}
            for r in sess
        ],
    }


# --- a player's lifetime database ---------------------------------------------------

_MY_SORTS = {
    "time": "h.ended_at",
    "pot": "h.pot_cents",
    "net": "r.delta_cents",
    "accuracy": "(CASE WHEN r.acc_n>0 THEN r.acc_sum/r.acc_n ELSE NULL END)",
}


def _my_hands(viewer_id: int, sort: str, direction: str, game_id: str | None,
              offset: int, limit: int, player_id: int | None = None,
              clubs: list[str] | None = None, variant: str | None = None,
              since: str | None = None) -> dict[str, Any]:
    """``player_id``'s hands (default: the viewer's own). The club is private and
    everyone may browse everyone's history — but the CARDS in it follow the live
    table's rule for the VIEWER: their own, plus hands that were tabled or shown.
    Browsing Riley's history never turns over a hand Riley mucked, and neither
    do its accuracy marks (they still count in Riley's totals), and a table whose
    host switched ``show_grades`` off shows nobody else's marks at all
    (SEC-004). ``variant`` = one game's hands only (None = every game).

    PERF-011: the page is sorted and cut on keys only (never the JSON records),
    with (game, hand) as the last tie-breakers so offset pages neither skip nor
    repeat a hand, and only the page's records are read."""
    player = int(player_id) if player_id is not None else int(viewer_id)
    col = hg._MY_SORTS.get(sort, hg._MY_SORTS["time"])
    if player != int(viewer_id) and sort == "accuracy":
        # someone else's per-hand marks show only where their hand was tabled or
        # shown, and only where the table shows marks at all — sorting on the
        # hidden ones would leak them (2026-09-26, SEC-004)
        col = f"(CASE WHEN (r.showdown=1 OR r.shown=1) AND g.show_grades=1 THEN {col} ELSE NULL END)"
    desc = str(direction).lower() != "asc"
    limit = max(1, min(hg.HANDS_PAGE_MAX, int(limit)))
    offset = max(0, int(offset))
    scope, sargs = hg._club_clause(clubs)
    where = "r.user_id=? AND g.excluded=0" + scope
    args: list[Any] = [player] + sargs
    if game_id:
        where += " AND r.game_id=?"
        args.append(str(game_id))
    if variant:
        where += " AND g.variant=?"
        args.append(str(variant))
    # A hand is never served while its table is still playing / revealing it.
    guard, gargs = hg._unpublished_guard()
    period, pargs = hg._period_clause(since)  # FEAT-004
    where += guard + period
    args += gargs + pargs
    total = pub.DB.one(
        "SELECT COUNT(*) c FROM homegame_hand_results r "
        f"JOIN homegames g ON g.id=r.game_id WHERE {where}", tuple(args)
    )["c"]
    rows = pub.DB.q(
        "SELECT r.game_id, r.hand_no, r.delta_cents, r.acc_sum, r.acc_n, h.ended_at, "
        "h.pot_cents, g.name AS table_name, g.variant, g.show_grades FROM homegame_hand_results r "
        "JOIN homegame_hands h ON h.game_id=r.game_id AND h.hand_no=r.hand_no "
        "JOIN homegames g ON g.id=r.game_id "
        f"WHERE {where} ORDER BY ({col} IS NULL), {col} {'DESC' if desc else 'ASC'}, "
        "h.ended_at DESC, r.game_id DESC, r.hand_no DESC LIMIT ? OFFSET ?",
        tuple(args + [limit, offset]),
    )
    summaries = {
        (x["game_id"], int(x["hand_no"])): x["summary"]
        for x in pub.DB.q(
            "SELECT game_id, hand_no, summary FROM homegame_hands WHERE "
            + " OR ".join("(game_id=? AND hand_no=?)" for _ in rows),
            tuple(v for r in rows for v in (r["game_id"], int(r["hand_no"]))),
        )
    } if rows else {}
    hands = []
    for r in rows:
        grades_public = bool(r["show_grades"])
        try:
            full = json.loads(summaries[(r["game_id"], int(r["hand_no"]))])
            seat_no = next(
                (int(x["seat"]) for x in full.get("seats") or []
                 if int(x.get("user_id") or -1) == player), None,
            )
            rec = hg._hand_for_viewer(full, viewer_id, show_all_grades=grades_public)
        except Exception:  # noqa: BLE001
            continue
        me = next((x for x in rec["seats"] if int(x["seat"]) == seat_no), None)
        seen = player == int(viewer_id) or bool(grades_public and me and me.get("shown"))
        hands.append({
            "game_id": r["game_id"], "table_name": r["table_name"],
            "variant": hg._norm_game(r["variant"]),
            "hand_no": int(r["hand_no"]), "ended_at": r["ended_at"],
            "pot_cents": int(r["pot_cents"]), "net_cents": int(r["delta_cents"]),
            "accuracy": (round(float(r["acc_sum"]) / int(r["acc_n"]), 1)
                         if seen and int(r["acc_n"] or 0) else None),
            "showdown": bool(rec.get("showdown")),
            "my_hole": me.get("hole") if me else None,
            "board_a": rec.get("board_a") or [], "board_b": rec.get("board_b") or [],
        })
    return {"hands": hands, "total": int(total), "offset": offset, "limit": limit}


def _my_stats(viewer_id: int, clubs: list[str] | None = None,
              variant: str | None = None, since: str | None = None) -> dict[str, Any]:
    """A player's numbers, within ``clubs`` (None = everything they played), for
    one game (``variant``) or all of them. ``games`` = hands per game in the same
    scope, whatever ``variant`` is (the switch between them)."""
    uid = int(viewer_id)
    who = pub._user_by_id(uid)
    scope, sargs = hg._club_clause(clubs)
    guard, gargs = hg._unpublished_guard()  # SEC-003: no result before the table has shown it
    period, pargs = hg._period_clause(since)  # FEAT-004
    scope += guard + period
    sargs = sargs + gargs + pargs
    played = hg._games_played(f"r.user_id=? AND g.excluded=0{scope}", [uid] + sargs)
    if variant:
        scope += " AND g.variant=?"
        sargs = sargs + [str(variant)]
    # per session, exact (OPS-015): a closed one's net is its ledger, an open one's
    # its hands summed in chips — the totals add the sessions up
    sessions = hg._session_results(f"r.user_id=? AND g.excluded=0{scope}", [uid] + sargs, since=since)
    sessions.sort(key=lambda s: (s["last"] or "", s["game_id"]), reverse=True)
    n = sum(s["acc_n"] for s in sessions)
    # head to head, in CENTS (chips are relative to each table's big blind)
    vs: dict[int, float] = {}
    fscope = scope.replace("r.game_id", "f.game_id").replace("r.hand_no", "f.hand_no")
    for r in pub.DB.q(
        "SELECT f.payer, f.payee, SUM(f.chips * g.bb_cents) v FROM homegame_flows f "
        f"JOIN homegames g ON g.id=f.game_id WHERE (f.payer=? OR f.payee=?) AND g.excluded=0{fscope} "
        "GROUP BY f.payer, f.payee", tuple([uid, uid] + sargs),
    ):
        other = int(r["payee"]) if int(r["payer"]) == uid else int(r["payer"])
        sign = -1 if int(r["payer"]) == uid else 1
        vs[other] = vs.get(other, 0.0) + sign * float(r["v"] or 0) / hg.BB_CHIPS
    versus = []
    one_club = clubs[0] if clubs is not None and len(clubs) == 1 else None  # (its nicknames)
    for other, cents in vs.items():
        u = pub._user_by_id(other)
        versus.append({"user_id": other,
                       "name": hg._display_name(u, one_club) if u is not None else f"Player {other}",
                       "net_cents": int(round(cents))})
    versus.sort(key=lambda x: -x["net_cents"])
    return {
        "user_id": uid, "name": hg._display_name(who, one_club) if who is not None else f"Player {uid}",
        "variant": variant, "games": hg._games_summary(played), "since": since,
        "hands": sum(s["hands"] for s in sessions), "wins": sum(s["wins"] for s in sessions),
        "net_cents": sum(s["net_cents"] for s in sessions),
        "accuracy": round(sum(s["acc_sum"] for s in sessions) / n, 1) if n else None, "graded": n,
        "sessions": [
            {"id": s["game_id"], "name": s["name"], "variant": s["variant"], "open": s["open"],
             "sb_cents": s["sb_cents"], "bb_cents": s["bb_cents"], "ante_cents": s["ante_cents"],
             "hands": s["hands"], "net_cents": s["net_cents"],
             "accuracy": round(s["acc_sum"] / s["acc_n"], 1) if s["acc_n"] else None,
             "last_played": s["last"]}
            for s in sessions[:200]
        ],
        "versus": versus,
    }


#: The profit graph's resolution: a long history is cut into this many runs of hands,
#: and each run keeps its lowest and highest point — the swings survive the thinning.
SERIES_BUCKETS = 240


def _my_series(player_id: int, clubs: list[str] | None = None, variant: str | None = None,
               game_id: str | None = None, since: str | None = None) -> dict[str, Any]:
    """A player's running net, hand by hand (FEAT-012: the profit graph in "My hands &
    stats"), in the same scope as ``_my_stats`` / ``_my_hands`` — clubs, one game, one
    table (``game_id``), one period — and like them never a hand its table is still
    revealing. ``points`` = [hand number, running net in cents] from [0, 0], thinned to
    at most 2 x SERIES_BUCKETS points on a long history; ``breaks`` = the hand numbers
    after which the next table's hands begin.

    Summed the way the stats sum (OPS-015), never by adding each hand's rounded cents:
    within a session the hands add up in CHIPS, turned into cents once (hands recorded
    before chips were stored add their cents), and a session's last hand lands on its
    exact net from ``_session_results`` — a closed session's ledger. So the last point
    IS the stats page's net for the same scope, to the cent."""
    scope, sargs = hg._club_clause(clubs)
    where = "r.user_id=? AND g.excluded=0" + scope
    args: list[Any] = [int(player_id)] + sargs
    if game_id:
        where += " AND r.game_id=?"
        args.append(str(game_id))
    if variant:
        where += " AND g.variant=?"
        args.append(str(variant))
    guard, gargs = hg._unpublished_guard()
    period, pargs = hg._period_clause(since)
    where += guard + period
    args += gargs + pargs
    rows = pub.DB.q(
        "SELECT r.delta_cents AS d, r.delta_chips AS dc, r.game_id AS gid, g.bb_cents AS bb "
        "FROM homegame_hand_results r "
        "JOIN homegame_hands h ON h.game_id=r.game_id AND h.hand_no=r.hand_no "
        "JOIN homegames g ON g.id=r.game_id "
        f"WHERE {where} ORDER BY h.ended_at, r.game_id, r.hand_no", tuple(args),
    )
    # each session's exact net, from the very rows the stats use (same scope, same period)
    exact = {s["game_id"]: s["net_cents"] for s in hg._session_results(where, list(args), since=since)} if rows else {}
    last_of = {r["gid"]: i for i, r in enumerate(rows)}
    chips: dict[str, int] = {}
    legacy: dict[str, int] = {}
    now: dict[str, int] = {}  # (each session's running net in cents, as last counted)
    run: list[int] = []
    breaks: list[int] = []
    cum, last = 0, None
    for i, r in enumerate(rows):
        gid = r["gid"]
        if last is not None and gid != last:
            breaks.append(len(run))
        last = gid
        if r["dc"] is None:
            legacy[gid] = legacy.get(gid, 0) + int(r["d"] or 0)
        else:
            chips[gid] = chips.get(gid, 0) + int(r["dc"])
        v = hg.chips_to_cents(chips.get(gid, 0), int(r["bb"] or 0)) + legacy.get(gid, 0)
        if i == last_of[gid] and gid in exact:
            v = exact[gid]
        cum += v - now.get(gid, 0)
        now[gid] = v
        run.append(cum)
    n = len(run)
    points = [[0, 0]]
    if n <= 2 * hg.SERIES_BUCKETS:
        points += [[k + 1, v] for k, v in enumerate(run)]
    else:
        size = n / hg.SERIES_BUCKETS
        for b in range(hg.SERIES_BUCKETS):
            lo, hi = int(b * size), max(int(b * size) + 1, int((b + 1) * size))
            seg = range(lo, min(hi, n))
            keep = {min(seg, key=run.__getitem__), max(seg, key=run.__getitem__)}
            points += [[i + 1, run[i]] for i in sorted(keep)]
        if points[-1][0] != n:
            points.append([n, run[-1]])
    return {
        "points": points, "hands": n, "net_cents": cum, "breaks": breaks[:200],
        "best_cents": max(run, default=0), "worst_cents": min(run, default=0),
    }


def _hands_export(t: LiveTable, viewer_id: int, fmt: str) -> Response:
    """The session's hand history as a download (FEAT-003): every hand the table
    has published, oldest first, each filtered for the viewer by ``_hand_for_viewer``
    — the table's reveal rule, so the file never shows a card the table did not.
    ``fmt`` "txt" (readable) or "json" (the records). The table's lock is held only
    to read what the export needs; the reading and formatting run outside it."""
    fmt = str(fmt or "txt").lower()
    if fmt not in ("txt", "json"):
        raise HTTPException(status_code=400, detail="format must be txt or json")
    with t.lock:
        upto = hg._hands_visible_upto(t)
        show_grades = bool(t.show_grades)
        g = hg.GAMES[hg._norm_game(t.variant)]
        meta = {
            "id": t.game_id, "name": t.name, "variant": hg._norm_game(t.variant),
            "game_label": g["label"], "game_name": g["name"],
            "bb_cents": int(t.bb_cents), "ante_cents": int(t.ante_cents),
            "status": t.status, "exported_at": pub._now(),
        }
        viewer = hg._user_name(t, viewer_id)
    hands = []
    for r in pub.DB.q(
        "SELECT summary FROM homegame_hands WHERE game_id=? AND hand_no<=? ORDER BY hand_no",
        (t.game_id, upto),
    ):
        try:
            rec = hg._hand_for_viewer(json.loads(r["summary"]), viewer_id, show_grades)
        except (ValueError, TypeError, KeyError):
            continue
        for s in rec.get("seats") or []:
            s.pop("avatar", None)  # (a page's picture address means nothing in a file)
        hands.append(rec)
    if fmt == "json":
        body, media = homegame_export.hand_history_json(meta, hands, viewer), "application/json"
    else:
        body, media = homegame_export.hand_history_text(meta, hands, viewer), "text/plain; charset=utf-8"
    name = homegame_export.filename(t.name, meta["exported_at"], fmt)
    return Response(body, media_type=media, headers={
        "Content-Disposition": f'attachment; filename="{name}"',
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    })


def _receipt(t: LiveTable, viewer_id: int, user_id: int) -> dict[str, Any]:
    """A player's money at this table, movement by movement (FEAT-002): every
    buy-in, top-up, automatic top-up, set-stack move, withdrawal and cash-out with
    its time and the hand it came after, then the totals — in, out, the stack's
    value now (the ledger's own figure) and the net. Your own; the host may open
    anyone's."""
    if int(user_id) != int(viewer_id) and int(viewer_id) != int(t.host_user_id):
        raise HTTPException(status_code=403, detail="only the host can see another player's receipt")
    rows = pub.DB.q(
        "SELECT kind, amount_cents, created_at, hand_no FROM homegame_ledger "
        "WHERE game_id=? AND user_id=? ORDER BY id", (t.game_id, int(user_id)),
    )
    ledger, seat_cents = hg._ledger_state(t, t.hand_start_stacks if hg._runout_blocking(t) else None)
    me = next((r for r in ledger if r["user_id"] == int(user_id)), None)
    if me is None and not rows:
        raise HTTPException(status_code=404, detail="no money moved for that player here")
    items = []
    for r in rows:
        kind, legacy = str(r["kind"]), r["hand_no"] is None
        items.append({
            "kind": kind,
            "label": (hg._LEGACY_LEDGER_LABELS if legacy else hg.LEDGER_LABELS).get(kind, kind),
            "direction": "in" if kind in hg.LEDGER_IN else "out",
            "cents": int(r["amount_cents"]),
            "at": r["created_at"],
            "hand_no": int(r["hand_no"]) if r["hand_no"] is not None else None,
        })
    money_in = sum(x["cents"] for x in items if x["direction"] == "in")
    money_out = sum(x["cents"] for x in items if x["direction"] == "out")
    stack = int(me["stack_cents"]) if me is not None and me["seated"] else 0
    return {
        "table": t.game_id, "table_name": t.name, "user_id": int(user_id),
        "name": me["name"] if me is not None else hg._user_name(t, user_id),
        "seated": bool(me and me["seated"]), "items": items,
        "in_cents": money_in, "out_cents": money_out, "stack_cents": stack,
        "net_cents": money_out + stack - money_in,
        "closed": t.status != "open",
    }


def _hand_detail(t: LiveTable, viewer_id: int, hand_no: int) -> dict[str, Any]:
    if int(hand_no) > hg._hands_visible_upto(t):
        raise HTTPException(status_code=404, detail="Not Found")
    row = pub.DB.one(
        "SELECT summary FROM homegame_hands WHERE game_id=? AND hand_no=?",
        (t.game_id, int(hand_no)),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Not Found")
    out = hg._hand_for_viewer(json.loads(row["summary"]), viewer_id, t.show_grades)
    out["game_id"] = t.game_id
    out["table_name"] = t.name
    return out


def _lobby_rows(rows: list[Any], viewer_id: int) -> list[dict[str, Any]]:
    """The lobby's rows for these open tables (``homegames`` rows), unlisted ones
    only for their host and the people who played there (link-only tables).
    THREE queries whatever the number of tables (PERF-010: three more per table
    every lobby poll, one more per unlisted one). No emails, no other people's
    user ids (review 2026-09-20 G12); names in each table's club context."""
    if not rows:
        return []
    ids = [r["id"] for r in rows]
    marks = ",".join("?" * len(ids))
    seated: dict[str, list[Any]] = {}
    for p in pub.DB.q(
        "SELECT p.game_id, p.user_id, p.seat, u.name, u.email FROM homegame_players p "
        f"JOIN users u ON u.id=p.user_id WHERE p.game_id IN ({marks}) AND p.seat IS NOT NULL ORDER BY p.seat",
        tuple(ids),
    ):
        seated.setdefault(p["game_id"], []).append(p)
    host_ids = sorted({int(r["host_user_id"]) for r in rows})
    hosts = {int(u["id"]): u for u in pub.DB.q(
        f"SELECT id, name, email FROM users WHERE id IN ({','.join('?' * len(host_ids))})", tuple(host_ids))}
    played = {x["game_id"] for x in pub.DB.q(
        f"SELECT game_id FROM homegame_players WHERE user_id=? AND game_id IN ({marks})",
        tuple([int(viewer_id)] + ids))}
    out = []
    for r in rows:
        is_host = int(r["host_user_id"]) == int(viewer_id)
        member = r["id"] in played
        if not bool(r["listed"]) and not (is_host or member):
            continue  # (link-only: it shows up for its host and the people who played there)
        players = seated.get(r["id"], [])
        host = hosts.get(int(r["host_user_id"]))
        out.append({
            "id": r["id"],
            "name": r["name"],
            "variant": hg._norm_game(r["variant"]),
            "num_seats": r["num_seats"],
            "seated": len(players),
            "sb_cents": r["sb_cents"],
            "bb_cents": r["bb_cents"],
            "ante_cents": r["ante_cents"],
            "default_buyin_cents": r["default_buyin_cents"],
            "hand_no": r["hand_no"],
            "running": bool(r["running"]),
            "host_name": hg._display_name(host, r["club_id"]) if host else "?",
            "created_at": r["created_at"],
            "players": [
                {"seat": int(p["seat"]), "name": hg._display_name(p, r["club_id"]),
                 "is_me": int(p["user_id"]) == int(viewer_id)}
                for p in players
            ],
            "is_host": is_host,
            "is_seated": any(int(p["user_id"]) == int(viewer_id) for p in players),
            "is_member": bool(member),
            "listed": bool(r["listed"]),
            "club_id": str(r["club_id"]),
        })
    return out


def _my_sessions(viewer_id: int, club_id: str | None = None, limit: int = 12) -> list[dict[str, Any]]:
    """The viewer's finished sessions (closed tables they played at), in one club —
    each with the viewer's part of settling it up (FEAT-001: "You pay Sam $42.10").
    Three queries for the whole list (PERF-010: it used to be one more per row)."""
    scope, sargs = hg._club_clause([club_id] if club_id else None)
    rows = pub.DB.q(
        "SELECT g.id, g.name, g.variant, g.sb_cents, g.bb_cents, g.ante_cents, g.hand_no, g.club_id, "
        "g.closed_at, p.buyin_cents, p.leftover_cents FROM homegame_players p "
        "JOIN homegames g ON g.id=p.game_id "
        f"WHERE p.user_id=? AND g.status='closed' AND p.buyin_cents>0 AND g.excluded=0{scope} "
        "ORDER BY g.closed_at DESC LIMIT ?",
        tuple([int(viewer_id)] + sargs + [int(limit)]),
    )
    ids = [r["id"] for r in rows]
    marks = ",".join("?" * len(ids))
    hands_of = {x["game_id"]: int(x["c"]) for x in pub.DB.q(
        f"SELECT game_id, COUNT(*) c FROM homegame_hand_results WHERE user_id=? AND game_id IN ({marks}) "
        "GROUP BY game_id", tuple([int(viewer_id)] + ids),
    )} if ids else {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    club_of = {r["id"]: r["club_id"] for r in rows}
    for x in (pub.DB.q(
        "SELECT p.game_id, p.user_id, p.buyin_cents, p.leftover_cents, u.name, u.email, u.id "
        f"FROM homegame_players p JOIN users u ON u.id=p.user_id WHERE p.game_id IN ({marks})",
        tuple(ids),
    ) if ids else []):
        ledgers.setdefault(x["game_id"], []).append({
            "user_id": int(x["user_id"]), "name": hg._display_name(x, club_of.get(x["game_id"])),
            "net_cents": int(x["leftover_cents"]) - int(x["buyin_cents"]),
        })
    out = []
    for r in rows:
        hands = hands_of.get(r["id"], 0)
        mine = [x for x in hg.settle_up(ledgers.get(r["id"], []), int(viewer_id)) if x["you"]]
        out.append({
            "id": r["id"],
            "name": r["name"],
            "variant": hg._norm_game(r["variant"]),
            "sb_cents": int(r["sb_cents"]),
            "bb_cents": int(r["bb_cents"]),
            "ante_cents": int(r["ante_cents"]),
            "closed_at": r["closed_at"],
            "hands": int(hands or 0),
            "buyin_cents": int(r["buyin_cents"]),
            "net_cents": int(r["leftover_cents"]) - int(r["buyin_cents"]),
            # the viewer's payments to settle the session: [{you: pay|get, name, cents}]
            "settle": [{"you": x["you"], "name": x["to_name"] if x["you"] == "pay" else x["from_name"],
                        "cents": x["cents"]} for x in mine],
        })
    return out
