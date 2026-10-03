"""Hand review — the store behind the paid page ``/games/review`` (2026-10-03).

A player drops the ``.zip`` ClubGG exports (no need to unpack it) and gets their
profit and loss beside their all-in EV, every hand in the home games' replayer
(Open in Study included) and every decision of theirs graded against the network
— with the worst-played hands one sort away. ``handreview`` reads the hands; this
module keeps them:

- ``review_hands`` — one row per (account, hand): the record, the numbers the
  page sorts and sums, and the grading job until it is graded. The primary key IS
  the duplicate check: a hand already stored is counted and skipped, whichever
  upload or file brings it again.
- ``review_uploads`` — each upload's progress and report.
- One IMPORT worker (uploads are read one at a time, site-wide, off the request)
  and one GRADER (the home games' ``grade_hand`` with the served PLO5 model; the
  player's own decisions only — other players' cards are never known).

The one PAID page of the site, even while ``public.FREE_FOR_ALL`` opens the rest:
it keeps your hand histories on the server. Every API route needs ``public._paid``
(402 otherwise); the page itself shows what the subscription buys. Hands are only
ever served to the account that uploaded them; "Delete my account" removes them
and "Download my data" includes their summary (``public.ACCOUNT_HOOKS``).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
import zlib
from typing import Any

from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from plo5bp.ui import handreview as hr
from plo5bp.ui import public as pub

logger = logging.getLogger(__name__)

#: Hands kept per account — the storage the subscription pays for.
MAX_HANDS_PER_USER = int(os.environ.get("PLO5BP_REVIEW_MAX_HANDS", "200000"))
#: Uploads waiting to be read, site-wide (each is held in memory until its turn).
UPLOAD_QUEUE_MAX = 6
#: Hands inserted per transaction (the writer lock is shared with the whole site).
INSERT_BATCH = 100
#: The profit graph's points at most (a long history is thinned, its last point kept).
SERIES_MAX_POINTS = 1500
GRADE_BATCH = 40
GRADE_IDLE_S = 20.0
GRADE_MAX_ATTEMPTS = 3

REVIEW_MIGRATIONS: list[Any] = [
    pub.Migration(1, "hand review base schema", statements=(
        "CREATE TABLE IF NOT EXISTS review_uploads ("
        " id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, filename TEXT NOT NULL DEFAULT '',"
        " bytes INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL,"
        " files INTEGER NOT NULL DEFAULT 0, read INTEGER NOT NULL DEFAULT 0,"
        " total INTEGER NOT NULL DEFAULT 0, added INTEGER NOT NULL DEFAULT 0,"
        " duplicates INTEGER NOT NULL DEFAULT 0, skipped INTEGER NOT NULL DEFAULT 0,"
        " skipped_detail TEXT, error TEXT, created_at TEXT NOT NULL, finished_at TEXT)",
        "CREATE INDEX IF NOT EXISTS review_uploads_user ON review_uploads(user_id, id)",
        "CREATE TABLE IF NOT EXISTS review_hands ("
        " user_id INTEGER NOT NULL, hand_key TEXT NOT NULL, site TEXT NOT NULL,"
        " played_ts INTEGER NOT NULL, played_at TEXT NOT NULL, bb_cents INTEGER NOT NULL,"
        " table_name TEXT NOT NULL DEFAULT '', net_cents INTEGER NOT NULL,"
        " ev_net_cents INTEGER NOT NULL, allin INTEGER NOT NULL DEFAULT 0,"
        " pot_cents INTEGER NOT NULL, showdown INTEGER NOT NULL DEFAULT 0,"
        " decisions INTEGER NOT NULL DEFAULT 0, acc_sum REAL, acc_n INTEGER, worst REAL,"
        " mistakes INTEGER, graded_at TEXT, grade_attempts INTEGER NOT NULL DEFAULT 0,"
        " record TEXT NOT NULL, job TEXT, raw BLOB, upload_id INTEGER, created_at TEXT NOT NULL,"
        " PRIMARY KEY (user_id, hand_key))",
        "CREATE INDEX IF NOT EXISTS review_hands_time ON review_hands(user_id, played_ts, hand_key)",
        "CREATE INDEX IF NOT EXISTS review_hands_jobs ON review_hands(created_at) WHERE job IS NOT NULL",
    )),
]


class Review:
    """The workers of ONE app (``install`` makes a fresh one current)."""

    def __init__(self) -> None:
        self.import_q: queue.Queue = queue.Queue(maxsize=UPLOAD_QUEUE_MAX)
        self.import_thread: threading.Thread | None = None
        self.grade_thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.grade_wake = threading.Event()
        self.lock = threading.Lock()
        self.busy_users: set[int] = set()  # an upload queued or being read


CTX = Review()


def _grading_on() -> bool:
    return os.environ.get("PLO5BP_REVIEW_GRADING", "1").strip().lower() not in ("0", "false", "no", "off")


# --- access ------------------------------------------------------------------------------------


def _uid() -> int:
    uid = pub._CURRENT_USER_ID.get()
    if uid is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return int(uid)


def _require_paid(uid: int) -> None:
    """402 unless the account pays (admin / comp / subscription) — Hand review is the
    site's one paid page, FREE_FOR_ALL or not (it stores your hands on the server)."""
    user = pub._user_by_id(uid)
    if not pub._paid(user):
        raise HTTPException(status_code=402, detail={
            "error": "subscription_required",
            "message": "Hand review keeps your hand histories on our server — it needs a subscription.",
            "price_cents": pub.PRICE_CENTS,
            "billing_configured": bool(pub.STRIPE_SECRET_KEY),
        })


def _guard(request: Request) -> None:
    from plo5bp.ui import homegame_routes as hgr

    hgr._api_guard(request)  # (signed in + the /games/api request budget)


router = APIRouter(dependencies=[Depends(_guard)])


# --- reading an upload (the import worker) ------------------------------------------------------


def queue_upload(uid: int, data: bytes, filename: str) -> dict[str, Any]:
    """Accept an upload: a row in ``review_uploads`` and a place in the import queue."""
    with CTX.lock:
        if uid in CTX.busy_users:
            raise HTTPException(status_code=409, detail="Your last upload is still being read — give it a moment.")
        if CTX.import_q.full():
            raise HTTPException(status_code=503, detail="Lots of uploads right now — try again in a minute.",
                                headers={"Retry-After": "30"})
        CTX.busy_users.add(uid)
    try:
        rows = pub.DB.q(
            "INSERT INTO review_uploads(user_id, filename, bytes, status, created_at) VALUES(?,?,?,?,?) RETURNING id",
            (uid, filename[:200], len(data), "queued", pub._now()),
        )
        upload_id = int(rows[0]["id"])
        CTX.import_q.put_nowait((upload_id, uid, data, filename))
    except BaseException:
        with CTX.lock:
            CTX.busy_users.discard(uid)
        raise
    _start_workers()
    return {"upload_id": upload_id, "status": "queued"}


def _set_upload(upload_id: int, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    pub.DB.q(f"UPDATE review_uploads SET {cols} WHERE id=?", (*fields.values(), upload_id))


def _import_loop(ctx: Review) -> None:
    while not ctx.stop.is_set():
        try:
            item = ctx.import_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            ctx.import_q.task_done()
            return
        upload_id, uid = item[0], item[1]
        try:
            run_import(*item)
        except Exception as e:  # noqa: BLE001 — one bad upload never stops the worker
            logger.exception("hand review import %s failed", upload_id)
            try:
                _set_upload(upload_id, status="failed", error=f"something went wrong reading it ({type(e).__name__})",
                            finished_at=pub._now())
            except Exception:  # noqa: BLE001
                pass
        finally:
            with ctx.lock:
                ctx.busy_users.discard(uid)
            ctx.import_q.task_done()


def _row(uid: int, upload_id: int, b: hr.BuiltHand, raw_text: str) -> tuple:
    rec = b.record
    return (
        uid, b.parsed.key, hr.SITE, int(b.parsed.ts), b.parsed.played_at, int(b.parsed.bb_cents),
        b.parsed.table_name[:80], int(b.net_cents), int(b.ev_net_cents), int(b.allin), int(b.pot_cents),
        int(bool(rec.get("showdown"))), int(b.decisions),
        json.dumps(rec, separators=(",", ":")),
        json.dumps(b.job, separators=(",", ":")) if b.gradable else None,
        zlib.compress(raw_text.encode("utf-8"), 6), upload_id, pub._now(),
    )


_INSERT = (
    "INSERT OR IGNORE INTO review_hands(user_id, hand_key, site, played_ts, played_at, bb_cents,"
    " table_name, net_cents, ev_net_cents, allin, pot_cents, showdown, decisions, record, job, raw,"
    " upload_id, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) RETURNING hand_key"
)


def run_import(upload_id: int, uid: int, data: bytes, filename: str) -> dict[str, Any]:
    """Read one upload into the account's hands (the worker; tests call it directly)."""
    _set_upload(upload_id, status="reading")
    try:
        files = hr.read_upload(data, filename)
        texts = [h for _name, t in files for h in hr.split_hands(t)]
        if not files:
            raise hr.UploadError("no_text_files", "the zip has no hand-history .txt files in it")
        if not texts:
            raise hr.UploadError("no_hands", "no hand histories were found in it")
        if len(texts) > hr.MAX_HANDS_PER_UPLOAD:
            raise hr.UploadError("too_many_hands", f"over {hr.MAX_HANDS_PER_UPLOAD:,} hands in one upload — split it")
    except hr.UploadError as e:
        _set_upload(upload_id, status="failed", error=e.detail or e.args[0], finished_at=pub._now())
        return {"status": "failed", "error": e.detail}
    del data
    _set_upload(upload_id, files=len(files), total=len(texts))
    have = {r["hand_key"] for r in pub.DB.q("SELECT hand_key FROM review_hands WHERE user_id=?", (uid,))}
    stored = len(have)
    n = {"added": 0, "dups": 0}
    skipped: dict[str, int] = {}
    batch: list[tuple] = []

    def flush() -> None:
        # (a hand another upload stored meanwhile is ignored by the key: a duplicate)
        with pub.DB.transaction():
            for row in batch:
                n["added" if pub.DB.q(_INSERT, row) else "dups"] += 1
        batch.clear()

    last_progress = time.monotonic()
    for k, text in enumerate(texts):
        try:
            ph = hr.parse_hand(text)
        except hr.HandError as e:
            skipped[e.reason] = skipped.get(e.reason, 0) + 1
            continue
        if ph.key in have:
            n["dups"] += 1
            continue
        if stored + n["added"] + len(batch) >= MAX_HANDS_PER_USER:
            skipped["storage_full"] = skipped.get("storage_full", 0) + 1
            continue
        try:
            built = hr.build_hand(ph)
        except hr.HandError as e:
            skipped[e.reason] = skipped.get(e.reason, 0) + 1
            continue
        have.add(ph.key)
        batch.append(_row(uid, upload_id, built, text))
        time.sleep(0.001)  # (the reading is pure Python: let the site's requests run between hands)
        if len(batch) >= INSERT_BATCH:
            flush()
        if time.monotonic() - last_progress > 1.0:
            last_progress = time.monotonic()
            _set_upload(upload_id, read=k + 1, added=n["added"] + len(batch), duplicates=n["dups"],
                        skipped=sum(skipped.values()))
    if batch:
        flush()
    added, dups = n["added"], n["dups"]
    _set_upload(
        upload_id, status="done", read=len(texts), added=added, duplicates=dups,
        skipped=sum(skipped.values()), skipped_detail=json.dumps(skipped) if skipped else None,
        finished_at=pub._now(),
    )
    CTX.grade_wake.set()
    return {"status": "done", "added": added, "duplicates": dups, "skipped": skipped}


# --- grading ---------------------------------------------------------------------------------


def _grade_loop(ctx: Review) -> None:
    from plo5bp.ui import homegame as hg

    while not ctx.stop.is_set():
        model = hg._grading_model()
        if model is None:
            ctx.grade_wake.wait(GRADE_IDLE_S)
            ctx.grade_wake.clear()
            continue
        rows = pub.DB.q(
            "SELECT user_id, hand_key, job FROM review_hands WHERE job IS NOT NULL"
            " ORDER BY created_at, played_ts LIMIT ?", (GRADE_BATCH,),
        )
        if not rows:
            ctx.grade_wake.wait(GRADE_IDLE_S)
            ctx.grade_wake.clear()
            continue
        for r in rows:
            if ctx.stop.is_set():
                return
            grade_one(int(r["user_id"]), str(r["hand_key"]), r["job"], model)


def grade_one(uid: int, key: str, job_json: str, model: Any) -> list[dict[str, Any]] | None:
    """Grade one stored hand's decisions (the player's own) and store the marks."""
    from plo5bp.ui import homegame as hg

    try:
        grades = hg.grade_hand(json.loads(job_json), model)
    except hg.NoGradingModel:
        return None
    except Exception:  # noqa: BLE001 — a replay the grader can't follow: tried again, then settled
        logger.exception("hand review grading failed (%s)", key)
        with pub.DB.transaction():
            pub.DB.q("UPDATE review_hands SET grade_attempts=grade_attempts+1 WHERE user_id=? AND hand_key=?",
                     (uid, key))
            row = pub.DB.one("SELECT grade_attempts FROM review_hands WHERE user_id=? AND hand_key=?", (uid, key))
        if row is None or int(row["grade_attempts"]) < GRADE_MAX_ATTEMPTS:
            return None
        grades = None
    store_grades(uid, key, grades)
    return grades


def store_grades(uid: int, key: str, grades: list[dict[str, Any]] | None) -> None:
    """The hand's marks land in its record and its numbers — and its job is done."""
    with pub.DB.transaction():
        row = pub.DB.one("SELECT record FROM review_hands WHERE user_id=? AND hand_key=?", (uid, key))
        if row is None:
            return
        rec = json.loads(row["record"])
        rec["grades"] = list(grades or [])
        if grades is None:
            rec["grades_note"] = "could not be graded"
        scores = [float(g["score"]) for g in rec["grades"]]
        mistakes = sum(1 for g in rec["grades"] if g.get("cat") in ("wrong", "blunder"))
        pub.DB.q(
            "UPDATE review_hands SET record=?, job=NULL, acc_sum=?, acc_n=?, worst=?, mistakes=?, graded_at=?"
            " WHERE user_id=? AND hand_key=?",
            (json.dumps(rec, separators=(",", ":")), float(sum(scores)) if scores else None,
             len(scores) or None, min(scores) if scores else None, mistakes, pub._now(), uid, key),
        )


# --- the numbers -----------------------------------------------------------------------------


def summary(uid: int) -> dict[str, Any]:
    r = pub.DB.one(
        "SELECT COUNT(*) n, COALESCE(SUM(net_cents),0) net, COALESCE(SUM(ev_net_cents),0) ev,"
        " COALESCE(SUM(allin),0) allins, COALESCE(SUM(showdown),0) showdowns,"
        " COALESCE(SUM(decisions),0) decisions, COALESCE(SUM(acc_sum),0) acc_sum,"
        " COALESCE(SUM(acc_n),0) acc_n, COALESCE(SUM(mistakes),0) mistakes,"
        " COALESCE(SUM(CASE WHEN job IS NOT NULL THEN 1 ELSE 0 END),0) pending,"
        " COALESCE(SUM(net_cents * 1.0 / bb_cents),0) net_bb, COALESCE(SUM(ev_net_cents * 1.0 / bb_cents),0) ev_bb,"
        " MIN(played_ts) first_ts, MAX(played_ts) last_ts"
        " FROM review_hands WHERE user_id=?", (uid,),
    )
    stakes = [dict(x) for x in pub.DB.q(
        "SELECT bb_cents, COUNT(*) hands FROM review_hands WHERE user_id=? GROUP BY bb_cents ORDER BY hands DESC",
        (uid,),
    )]
    uploads = [_upload_view(x) for x in pub.DB.q(
        "SELECT * FROM review_uploads WHERE user_id=? ORDER BY id DESC LIMIT 6", (uid,),
    )]
    n = int(r["n"])
    return {
        "hands": n, "net_cents": int(r["net"]), "ev_net_cents": int(r["ev"]),
        "net_bb": round(float(r["net_bb"]), 2), "ev_net_bb": round(float(r["ev_bb"]), 2),
        "allin_hands": int(r["allins"]), "showdowns": int(r["showdowns"]),
        "decisions": int(r["decisions"]), "graded": int(r["acc_n"]),
        "accuracy": round(float(r["acc_sum"]) / int(r["acc_n"]), 1) if int(r["acc_n"]) else None,
        "mistakes": int(r["mistakes"]), "grading_pending": int(r["pending"]),
        "first_ts": r["first_ts"], "last_ts": r["last_ts"], "stakes": stakes,
        "uploads": uploads, "max_hands": MAX_HANDS_PER_USER,
        "max_upload_mb": hr.MAX_UPLOAD_BYTES // (1024 * 1024),
    }


def _upload_view(r: Any) -> dict[str, Any]:
    return {
        "id": int(r["id"]), "filename": r["filename"], "status": r["status"], "files": int(r["files"]),
        "read": int(r["read"]), "total": int(r["total"]), "added": int(r["added"]),
        "duplicates": int(r["duplicates"]), "skipped": int(r["skipped"]),
        "skipped_detail": json.loads(r["skipped_detail"]) if r["skipped_detail"] else {},
        "error": r["error"], "created_at": r["created_at"], "finished_at": r["finished_at"],
    }


def series(uid: int) -> dict[str, Any]:
    """The running net and the running all-in EV result, hand by hand (oldest
    first), in dollars-and-cents and in big blinds — thinned to at most
    ``SERIES_MAX_POINTS`` points (the last one kept: it IS the totals)."""
    rows = pub.DB.q(
        "SELECT net_cents, ev_net_cents, bb_cents FROM review_hands WHERE user_id=? ORDER BY played_ts, hand_key",
        (uid,),
    )
    n = len(rows)
    step = max(1, -(-n // SERIES_MAX_POINTS))
    pts = [[0, 0, 0, 0.0, 0.0]]
    net = ev = 0
    net_bb = ev_bb = 0.0
    hi = lo = 0
    for k, r in enumerate(rows, start=1):
        net += int(r["net_cents"])
        ev += int(r["ev_net_cents"])
        bb = max(1, int(r["bb_cents"]))
        net_bb += int(r["net_cents"]) / bb
        ev_bb += int(r["ev_net_cents"]) / bb
        hi, lo = max(hi, net, ev), min(lo, net, ev)
        if k % step == 0 or k == n:
            pts.append([k, net, ev, round(net_bb, 2), round(ev_bb, 2)])
    return {"hands": n, "points": pts, "net_cents": net, "ev_net_cents": ev, "high_cents": hi, "low_cents": lo}


_SORTS = {
    "time": "played_ts {d}, hand_key {d}",
    "net": "net_cents {d}, played_ts DESC",
    "pot": "pot_cents {d}, played_ts DESC",
    "luck": "(net_cents - ev_net_cents) {d}, played_ts DESC",
    # worst-played first = the lowest decision score first (ungraded hands last)
    "worst": "(worst IS NULL) ASC, worst {d}, mistakes DESC, played_ts DESC",
    "accuracy": "(acc_n IS NULL) ASC, (acc_sum / acc_n) {d}, played_ts DESC",
}
_FILTERS = {
    "": "",
    "allin": " AND allin=1",
    "showdown": " AND showdown=1",
    "mistakes": " AND mistakes > 0",
    "won": " AND net_cents > 0",
    "lost": " AND net_cents < 0",
}


def hands(uid: int, sort: str, direction: str, offset: int, limit: int, filt: str) -> dict[str, Any]:
    if sort not in _SORTS:
        raise HTTPException(status_code=400, detail="unknown sort")
    if filt not in _FILTERS:
        raise HTTPException(status_code=400, detail="unknown filter")
    d = "ASC" if str(direction).lower() == "asc" else "DESC"
    if sort == "worst":
        # (descending = the worst-played first: the LOWEST decision score first)
        d = "DESC" if d == "ASC" else "ASC"
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    where = "user_id=?" + _FILTERS[filt]
    total = int(pub.DB.one(f"SELECT COUNT(*) n FROM review_hands WHERE {where}", (uid,))["n"])
    rows = pub.DB.q(
        f"SELECT hand_key, played_at, played_ts, table_name, net_cents, ev_net_cents, allin, pot_cents, showdown,"
        f" decisions, acc_sum, acc_n, worst, mistakes, job IS NOT NULL AS pending, record"
        f" FROM review_hands WHERE {where} ORDER BY {_SORTS[sort].format(d=d)} LIMIT ? OFFSET ?",
        (uid, limit, offset),
    )
    out = []
    for r in rows:
        rec = json.loads(r["record"])
        me = next((s for s in rec.get("seats", []) if s.get("is_me")), {})
        out.append({
            "key": r["hand_key"], "hand_id": rec.get("hand_id"), "played_at": r["played_at"],
            "table_name": r["table_name"], "net_cents": int(r["net_cents"]),
            "ev_net_cents": int(r["ev_net_cents"]), "allin": bool(r["allin"]),
            "pot_cents": int(r["pot_cents"]), "showdown": bool(r["showdown"]),
            "decisions": int(r["decisions"]), "my_hole": me.get("hole") or [],
            "board_a": rec.get("board_a") or [], "board_b": rec.get("board_b") or [],
            "accuracy": round(float(r["acc_sum"]) / int(r["acc_n"]), 1) if r["acc_n"] else None,
            "worst": round(float(r["worst"]), 1) if r["worst"] is not None else None,
            "mistakes": int(r["mistakes"] or 0), "grading": bool(r["pending"]),
            "players": len(rec.get("seats", [])),
        })
    return {"hands": out, "total": total, "offset": offset, "limit": limit}


_KEY_RE = re.compile(r"^[a-z]+:[A-Za-z0-9_\-]{1,64}$")


def hand(uid: int, key: str) -> dict[str, Any]:
    if not _KEY_RE.match(key or ""):
        raise HTTPException(status_code=404, detail="Not Found")
    row = pub.DB.one("SELECT record, job IS NOT NULL AS pending FROM review_hands WHERE user_id=? AND hand_key=?",
                     (uid, key))
    if row is None:
        raise HTTPException(status_code=404, detail="Not Found")
    rec = json.loads(row["record"])
    rec["grading"] = bool(row["pending"])
    return rec


def delete_all(uid: int) -> dict[str, Any]:
    with CTX.lock:
        if uid in CTX.busy_users:
            raise HTTPException(status_code=409, detail="An upload is still being read — wait for it to finish.")
    with pub.DB.transaction():
        n = int(pub.DB.one("SELECT COUNT(*) n FROM review_hands WHERE user_id=?", (uid,))["n"])
        pub.DB.q("DELETE FROM review_hands WHERE user_id=?", (uid,))
        pub.DB.q("DELETE FROM review_uploads WHERE user_id=?", (uid,))
    return {"deleted": n}


# --- the API -----------------------------------------------------------------------------------


@router.get("/games/api/review/summary")
def api_summary():
    uid = _uid()
    _require_paid(uid)
    return summary(uid)


@router.post("/games/api/review/upload")
async def api_upload(request: Request, name: str = ""):
    """The upload itself: the request body IS the file (the .zip ClubGG exports, or
    one of its .txt files). It is read in the background — poll ``uploads/{id}``."""
    uid = _uid()
    await run_in_threadpool(_require_paid, uid)
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="That file is empty.")
    if len(data) > hr.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Uploads are limited to {hr.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    name = re.sub(r"[^\w .()\-]", "_", str(name or "upload.zip"))[:120] or "upload.zip"
    return await run_in_threadpool(queue_upload, uid, data, name)


@router.get("/games/api/review/uploads/{upload_id}")
def api_upload_status(upload_id: int):
    uid = _uid()
    r = pub.DB.one("SELECT * FROM review_uploads WHERE id=? AND user_id=?", (int(upload_id), uid))
    if r is None:
        raise HTTPException(status_code=404, detail="Not Found")
    out = _upload_view(r)
    out["grading_pending"] = int(pub.DB.one(
        "SELECT COUNT(*) n FROM review_hands WHERE user_id=? AND job IS NOT NULL", (uid,))["n"])
    return out


@router.get("/games/api/review/series")
def api_series():
    uid = _uid()
    _require_paid(uid)
    return series(uid)


@router.get("/games/api/review/hands")
def api_hands(sort: str = "time", dir: str = "desc", offset: int = 0, limit: int = 40, filter: str = ""):
    uid = _uid()
    _require_paid(uid)
    return hands(uid, sort, dir, offset, limit, filter)


@router.get("/games/api/review/hands/{key}")
def api_hand(key: str):
    uid = _uid()
    _require_paid(uid)
    return hand(uid, key)


@router.post("/games/api/review/delete")
def api_delete(body: dict = Body(...)):
    """Delete every hand (and upload report) of the account — ``{"confirm": true}``.
    Allowed without a subscription: your data is yours to remove."""
    uid = _uid()
    if body.get("confirm") is not True:
        raise HTTPException(status_code=400, detail="confirm the deletion")
    return delete_all(uid)


# --- account export / deletion (public.ACCOUNT_HOOKS) --------------------------------------------


def _account_export(uid: int) -> dict[str, Any]:
    rows = pub.DB.q(
        "SELECT hand_key, played_at, table_name, net_cents, ev_net_cents, pot_cents, acc_sum, acc_n"
        " FROM review_hands WHERE user_id=? ORDER BY played_ts, hand_key", (int(uid),),
    )
    return {
        "hands": [{
            "hand": r["hand_key"], "played_at": r["played_at"], "table": r["table_name"],
            "net_cents": r["net_cents"], "allin_ev_net_cents": r["ev_net_cents"], "pot_cents": r["pot_cents"],
            "accuracy": round(r["acc_sum"] / r["acc_n"], 1) if r["acc_n"] else None,
        } for r in rows],
        "uploads": [_upload_view(r) for r in pub.DB.q(
            "SELECT * FROM review_uploads WHERE user_id=? ORDER BY id", (int(uid),))],
    }


def _account_anonymize(uid: int) -> None:
    """Deleting the account deletes its hand histories — they were only theirs."""
    with pub.DB.transaction():
        pub.DB.q("DELETE FROM review_hands WHERE user_id=?", (int(uid),))
        pub.DB.q("DELETE FROM review_uploads WHERE user_id=?", (int(uid),))


_hooks = getattr(pub, "ACCOUNT_HOOKS", None)
if isinstance(_hooks, dict):
    _hooks["hand_review"] = {"export": _account_export, "anonymize": _account_anonymize}


# --- install -----------------------------------------------------------------------------------


def _start_workers() -> None:
    ctx = CTX
    if ctx.import_thread is None or not ctx.import_thread.is_alive():
        ctx.import_thread = threading.Thread(target=_import_loop, args=(ctx,), name="review-import", daemon=True)
        ctx.import_thread.start()
    if _grading_on() and (ctx.grade_thread is None or not ctx.grade_thread.is_alive()):
        ctx.grade_thread = threading.Thread(target=_grade_loop, args=(ctx,), name="review-grader", daemon=True)
        ctx.grade_thread.start()


def stop_workers(timeout: float = 5.0) -> None:
    ctx = CTX
    ctx.stop.set()
    ctx.grade_wake.set()
    for t in (ctx.import_thread, ctx.grade_thread):
        if t is not None and t.is_alive():
            t.join(timeout)


def wait_idle(timeout: float = 30.0) -> bool:
    """Block until queued uploads are read (tests)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if CTX.import_q.unfinished_tasks == 0:
            return True
        time.sleep(0.02)
    return False


def install(app: FastAPI, *, static_dir: Any) -> None:
    """Mount Hand review on the public app (after the home games: it shares their
    page, client and API guard). A fresh context per app: the old one's workers stop."""
    global CTX
    from plo5bp.ui import homegame as hg

    old = CTX
    if old.import_thread is not None or old.grade_thread is not None:
        stop_workers()
    CTX = Review()
    pub.DB.migrate("handreview", REVIEW_MIGRATIONS)
    pub._BODY_LIMITS["/games/api/review/upload"] = hr.MAX_UPLOAD_BYTES

    @app.get("/games/review")
    def review_page():
        return HTMLResponse(hg._page_html(static_dir), headers=dict(hg.PAGE_HEADERS))

    app.include_router(router)
    app.router.on_shutdown.append(stop_workers)
    _start_workers()
    logger.info("hand review installed")
