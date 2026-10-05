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
- ``review_mistakes`` — every decision of the player's the network graded a wrong
  move or a blunder, with its learning state: the Trainer's mistakes drill deals them
  back (``drill_next`` / ``drill_result``, plugged in by ``trainer.set_drill_hooks``).
- One IMPORT worker (uploads are read one at a time, site-wide, off the request)
  and one GRADER (the home games' ``grade_hand`` with the served PLO5 model): every
  decision whose cards are known -- the player's own and, since 2026-10-05 (owner),
  the hands shown down. The player's numbers (accuracy, worst mark) and the mistakes
  drill count their own decisions only.
- Regrading (2026-10-05, owner, after a stronger network went live: "I would like my own
  uploaded hands to be regraded"): every grade records the network that made it
  (``graded_by`` = its checkpoint's sha256, 16 hex); ``regrade`` queues the player's hands
  an earlier network graded, and the grader rebuilds each one's job from the hand's text
  and grades it with the network served now — the numbers and the drill follow the new
  marks (a spot still a mistake keeps its learning state); a hand it can't redo keeps its
  marks.
- The network's choice at any decision of a stored hand (``choice``: the replayer
  asks; home-game hands too, through ``homegame_routes.api_hand_choice``).
- The Trainer's "My tables" (``my_tables``, plugged in by ``trainer.set_my_tables_hook``):
  the profile of the tables in the player's latest hands — players, ante, their own
  stack and their opponents' (``handreview.table_profile``) — cached until their hands
  change.

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
from calendar import timegm
from datetime import datetime
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
#: "My tables" (the Trainer) reads your latest hands — at most this many (the tables you
#: play NOW) — and needs at least PROFILE_MIN_HANDS (fewer: typical ClubGG tables).
PROFILE_HANDS = 5000
PROFILE_MIN_HANDS = 50
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

#: A hand whose OWN decisions still wait for their marks (SQL): a job that only adds the
#: marks of the hands shown down (``merge``, migration 3) leaves the player's as they are.
_OWN_PENDING = "(job IS NOT NULL AND instr(job, '\"merge\":true') = 0)"

#: The grades the mistakes drill deals back (a "wrong move" or a "blunder").
MISTAKE_CATS = ("wrong", "blunder")


def _mistake_rows(uid: int, key: str, ts: int, grades: list[dict[str, Any]] | None) -> list[tuple]:
    return [(int(uid), key, int(g["i"]), float(g["score"]), str(g["cat"]), int(ts))
            for g in grades or [] if g.get("cat") in MISTAKE_CATS]


_MISTAKE_UPSERT = (
    "INSERT INTO review_mistakes(user_id, hand_key, idx, score, cat, played_ts) VALUES(?,?,?,?,?,?)"
    " ON CONFLICT(user_id, hand_key, idx) DO UPDATE SET score=excluded.score, cat=excluded.cat"
)


def _own_grades(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """A record's marks of the player's own decisions (it also holds the hands shown down's)."""
    hero = next((int(s["seat"]) for s in rec.get("seats") or [] if s.get("is_me")), None)
    return [g for g in rec.get("grades") or [] if hero is None or int(g.get("seat", -1)) == hero]


def _backfill_mistakes(conn: Any) -> None:
    """Migration 2: the mistakes of the hands graded before the drill existed."""
    for uid, key, ts, record in conn.execute(
        "SELECT user_id, hand_key, played_ts, record FROM review_hands WHERE mistakes > 0"
    ).fetchall():
        for row in _mistake_rows(uid, key, ts, _own_grades(json.loads(record))):
            conn.execute(_MISTAKE_UPSERT, row)


REVIEW_MIGRATIONS.append(
    # The mistakes drill (2026-10-03): one row per mistake of yours (a decision graded
    # "wrong" or "blunder") with its learning state — `weight` goes down when you play
    # the spot right in the Trainer and back up when you miss it again.
    pub.Migration(2, "mistakes drill", statements=(
        "CREATE TABLE IF NOT EXISTS review_mistakes ("
        " user_id INTEGER NOT NULL, hand_key TEXT NOT NULL, idx INTEGER NOT NULL,"
        " score REAL NOT NULL, cat TEXT NOT NULL, played_ts INTEGER NOT NULL DEFAULT 0,"
        " weight REAL NOT NULL DEFAULT 1.0, fixed INTEGER NOT NULL DEFAULT 0,"
        " missed INTEGER NOT NULL DEFAULT 0, last_cat TEXT, last_at TEXT,"
        " PRIMARY KEY (user_id, hand_key, idx))",
    ), fn=_backfill_mistakes),
)


def _grade_shown_hands(conn: Any) -> None:
    """Migration 3 (2026-10-05, owner: a shown-down player's decisions graded too): every
    stored hand with a showdown gets its grading job rebuilt from its text. A hand still
    waiting takes the new job whole; a hand already graded gets one for the shown hands'
    decisions only (``merge``: the player's marks, numbers and drill stay as they are)."""
    rows = conn.execute(
        "SELECT user_id, hand_key, raw, job FROM review_hands WHERE showdown=1 AND raw IS NOT NULL"
    ).fetchall()
    for uid, key, raw, old_job in rows:
        try:
            p = hr.parse_hand(zlib.decompress(raw).decode("utf-8"))
            L = hr.ledger(p)
            job, _upto, _notes = hr.engine_replay(p, L)
        except Exception:  # noqa: BLE001 -- a hand this code can't rebuild keeps what it has
            logger.warning("hand review: no shown-hands job for %s", key, exc_info=True)
            continue
        if old_job is not None:
            conn.execute("UPDATE review_hands SET job=? WHERE user_id=? AND hand_key=?",
                         (json.dumps(job, separators=(",", ":")), uid, key))
            continue
        for a in job["actions"]:
            a[3] = bool(a[3]) and int(a[0]) != int(L.hero)
        if any(a[3] for a in job["actions"]):
            job["merge"] = True
            conn.execute("UPDATE review_hands SET job=?, grade_attempts=0 WHERE user_id=? AND hand_key=?",
                         (json.dumps(job, separators=(",", ":")), uid, key))


REVIEW_MIGRATIONS.append(
    pub.Migration(3, "grade the hands shown down", fn=_grade_shown_hands),
)
# (which network graded a hand: NULL = graded before this was kept -- an earlier network)
REVIEW_MIGRATIONS.append(
    pub.Migration(4, "review_hands.graded_by", fn=pub._add_column("review_hands", "graded_by", "TEXT")),
)


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
        self.drill: dict[int, dict[str, Any]] = {}  # user -> the mistakes drill's round
        self.profiles: dict[int, dict[str, Any]] = {}  # user -> "My tables" profile (cached)
        self.profile_gen: dict[int, int] = {}  # user -> bumped whenever their hands change


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
            _profile_stale(uid)  # (a failed upload may have stored some of its hands)
            ctx.import_q.task_done()


def _row(uid: int, upload_id: int, b: hr.BuiltHand, raw_text: str) -> tuple:
    rec = b.record
    return (
        uid, b.parsed.key, hr.SITE, int(b.parsed.ts), b.parsed.played_at, int(b.parsed.bb_cents),
        b.parsed.table_name[:80], int(b.net_cents), int(b.ev_net_cents), int(b.allin), int(b.pot_cents),
        int(bool(rec.get("showdown"))), int(b.decisions),
        json.dumps(rec, separators=(",", ":")),
        json.dumps(b.job, separators=(",", ":")) if any(a[3] for a in b.job["actions"]) else None,
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
    _profile_stale(uid)
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


#: The job of a hand graded again (``regrade``): the grader makes its real job from the
#: hand's text first, so the request queuing thousands of hands stays one UPDATE.
_REGRADE_JOB = '{"regrade":true}'


def model_id(model: Any) -> str | None:
    """Which network graded a hand: its checkpoint's sha256, 16 hex (as /health and
    docs/models.md print it; ``models.build_entry`` sets it) — None if it has none."""
    sha = getattr(model, "checkpoint_sha256", None)
    return str(sha)[:16] if sha else None


def grade_one(uid: int, key: str, job_json: str, model: Any) -> list[dict[str, Any]] | None:
    """Grade one stored hand's decisions (the player's own and the hands shown down) and
    store the marks. A regrade (``_REGRADE_JOB``) makes the job again from the hand's text
    first; a hand the network can't grade again keeps the marks it had."""
    from plo5bp.ui import homegame as hg

    job = json.loads(job_json)
    mid = model_id(model)
    again = bool(job.get("regrade"))
    if again:
        job = _rebuilt_job(uid, key)
        if job is None:
            _regrade_failed(uid, key, mid)
            return None
    try:
        grades = hg.grade_hand(job, model)
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
        if again:
            _regrade_failed(uid, key, mid)
            return None
        grades = None
    store_grades(uid, key, grades, merge=bool(job.get("merge")), model_id=mid)
    return grades


def _rebuilt_job(uid: int, key: str) -> dict[str, Any] | None:
    """A stored hand's grading job made again from its text — every decision whose cards
    are known — or None (no text kept, or this code can't follow the hand)."""
    row = pub.DB.one("SELECT raw FROM review_hands WHERE user_id=? AND hand_key=?", (uid, key))
    if row is None or row["raw"] is None:
        return None
    try:
        p = hr.parse_hand(zlib.decompress(row["raw"]).decode("utf-8"))
        job, _upto, _notes = hr.engine_replay(p, hr.ledger(p))
    except Exception:  # noqa: BLE001 — the hand keeps the marks it has
        logger.warning("hand review: %s can't be graded again", key, exc_info=True)
        return None
    return job


def _regrade_failed(uid: int, key: str, mid: str | None) -> None:
    """The network served now can't grade the hand again: its marks stay as they were (and
    it isn't offered for a regrade with this network again)."""
    logger.warning("hand review: %s keeps its earlier marks (the network served now can't grade it)", key)
    pub.DB.q("UPDATE review_hands SET job=NULL, graded_by=? WHERE user_id=? AND hand_key=?", (mid, uid, key))


def _hero_seat(rec: dict[str, Any]) -> int | None:
    return next((int(s["seat"]) for s in rec.get("seats") or [] if s.get("is_me")), None)


def store_grades(uid: int, key: str, grades: list[dict[str, Any]] | None, *, merge: bool = False,
                 model_id: str | None = None) -> None:
    """The hand's marks land in its record and its numbers — and its job is done. The
    numbers and the mistakes drill are the player's own decisions only (the record also
    holds the marks of the hands shown down). ``merge``: a job that graded only the hands
    shown (migration 3) -- the player's marks stay exactly as they were (and so does
    ``graded_by``: the network that made them). ``model_id``: the network grading now."""
    with pub.DB.transaction():
        row = pub.DB.one("SELECT record, played_ts FROM review_hands WHERE user_id=? AND hand_key=?", (uid, key))
        if row is None:
            return
        rec = json.loads(row["record"])
        if merge:
            if grades is not None:
                hero = _hero_seat(rec)
                shown = [g for g in grades if int(g.get("seat", -1)) != hero]
                rec["grades"] = sorted(_own_grades(rec) + shown, key=lambda g: int(g["i"]))
        else:
            rec["grades"] = list(grades or [])
            if grades is None:
                rec["grades_note"] = "could not be graded"
            else:
                rec.pop("grades_note", None)
        mine = _own_grades(rec)
        scores = [float(g["score"]) for g in mine]
        found = _mistake_rows(uid, key, int(row["played_ts"]), mine)
        pub.DB.q(
            "UPDATE review_hands SET record=?, job=NULL, acc_sum=?, acc_n=?, worst=?, mistakes=?, graded_at=?,"
            " graded_by=CASE WHEN ? THEN graded_by ELSE ? END WHERE user_id=? AND hand_key=?",
            (json.dumps(rec, separators=(",", ":")), float(sum(scores)) if scores else None,
             len(scores) or None, min(scores) if scores else None, len(found), pub._now(),
             int(merge), model_id, uid, key),
        )
        # (the drill's spots: a decision that is no longer a mistake leaves it; one that
        # still is keeps its learning state)
        keep = [r[2] for r in found]
        pub.DB.q(
            f"DELETE FROM review_mistakes WHERE user_id=? AND hand_key=? AND idx NOT IN ({','.join('?' * len(keep))})",
            (uid, key, *keep),
        )
        for r in found:
            pub.DB.q(_MISTAKE_UPSERT, r)


# --- the numbers -----------------------------------------------------------------------------

#: Hands in the order they were played: the time printed in the hand (to the second), and
#: within one second the hand number (ClubGG numbers rise with time; a key's digits
#: compare as a number when the shorter key comes first — "ring_99" before "ring_100").
CHRONO = "played_ts {d}, length(hand_key) {d}, hand_key {d}"


def _day_ts(day: str) -> int:
    """'YYYY-MM-DD' -> its first second, on the hands' own clock (read like played_ts)."""
    try:
        return timegm(datetime.strptime(day.strip(), "%Y-%m-%d").timetuple())
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates are YYYY-MM-DD.") from None


def _range_sql(start: str = "", end: str = "") -> tuple[str, tuple]:
    """The SQL condition and its parameters for the hands played from `start` through
    `end` (inclusive days, 'YYYY-MM-DD', on the clock the hands print — ClubGG's is the
    player's own); either may be empty (no bound)."""
    clause, params = "", []
    lo = _day_ts(start) if start else None
    hi = _day_ts(end) + 86_400 if end else None
    if lo is not None and hi is not None and lo >= hi:
        raise HTTPException(status_code=400, detail="The start date is after the end date.")
    if lo is not None:
        clause += " AND played_ts >= ?"
        params.append(lo)
    if hi is not None:
        clause += " AND played_ts < ?"
        params.append(hi)
    return clause, tuple(params)


def summary(uid: int, start: str = "", end: str = "") -> dict[str, Any]:
    rng, rp = _range_sql(start, end)
    r = pub.DB.one(
        "SELECT COUNT(*) n, COALESCE(SUM(net_cents),0) net, COALESCE(SUM(ev_net_cents),0) ev,"
        " COALESCE(SUM(allin),0) allins, COALESCE(SUM(showdown),0) showdowns,"
        " COALESCE(SUM(decisions),0) decisions, COALESCE(SUM(acc_sum),0) acc_sum,"
        " COALESCE(SUM(acc_n),0) acc_n, COALESCE(SUM(mistakes),0) mistakes,"
        f" COALESCE(SUM(CASE WHEN {_OWN_PENDING} THEN 1 ELSE 0 END),0) pending,"
        " COALESCE(SUM(net_cents * 1.0 / bb_cents),0) net_bb, COALESCE(SUM(ev_net_cents * 1.0 / bb_cents),0) ev_bb,"
        " MIN(played_ts) first_ts, MAX(played_ts) last_ts"
        f" FROM review_hands WHERE user_id=?{rng}", (uid, *rp),
    )
    every = pub.DB.one(
        "SELECT COUNT(*) n, MIN(played_ts) first_ts, MAX(played_ts) last_ts FROM review_hands WHERE user_id=?",
        (uid,),
    )
    stakes = [dict(x) for x in pub.DB.q(
        f"SELECT bb_cents, COUNT(*) hands FROM review_hands WHERE user_id=?{rng} GROUP BY bb_cents ORDER BY hands DESC",
        (uid, *rp),
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
        # the date range asked for, and every hand's (the page's bounds and its empty state)
        "range": {"start": start or None, "end": end or None},
        "all_hands": int(every["n"]), "all_first_ts": every["first_ts"], "all_last_ts": every["last_ts"],
        "uploads": uploads, "max_hands": MAX_HANDS_PER_USER,
        "max_upload_mb": hr.MAX_UPLOAD_BYTES // (1024 * 1024),
        "drill": drill_summary(uid),
        # every hand an earlier network graded, whatever the dates (the page offers a regrade)
        "regrade": {"older": _older_count(uid, _served_id())},
    }


def _served_id() -> str | None:
    """The id of the network grading now (None: no network, or one without a checkpoint)."""
    from plo5bp.ui import homegame as hg

    model = hg._grading_model()
    return None if model is None else model_id(model)


def _upload_view(r: Any) -> dict[str, Any]:
    return {
        "id": int(r["id"]), "filename": r["filename"], "status": r["status"], "files": int(r["files"]),
        "read": int(r["read"]), "total": int(r["total"]), "added": int(r["added"]),
        "duplicates": int(r["duplicates"]), "skipped": int(r["skipped"]),
        "skipped_detail": json.loads(r["skipped_detail"]) if r["skipped_detail"] else {},
        "error": r["error"], "created_at": r["created_at"], "finished_at": r["finished_at"],
    }


def series(uid: int, start: str = "", end: str = "") -> dict[str, Any]:
    """The running net and the running all-in EV result, hand by hand in the order the
    hands were PLAYED (`CHRONO` — whatever order they were uploaded in), from `start`
    through `end` when given (the running sums start at 0 there), in dollars-and-cents
    and in big blinds, with each point's time — thinned to at most ``SERIES_MAX_POINTS``
    points (the last one kept: it IS the totals). Point = [hand, net, ev, net bb, ev bb,
    played_ts]."""
    rng, rp = _range_sql(start, end)
    rows = pub.DB.q(
        f"SELECT net_cents, ev_net_cents, bb_cents, played_ts FROM review_hands WHERE user_id=?{rng}"
        f" ORDER BY {CHRONO.format(d='ASC')}",
        (uid, *rp),
    )
    n = len(rows)
    step = max(1, -(-n // SERIES_MAX_POINTS))
    pts = [[0, 0, 0, 0.0, 0.0, int(rows[0]["played_ts"]) if rows else None]]
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
            pts.append([k, net, ev, round(net_bb, 2), round(ev_bb, 2), int(r["played_ts"])])
    return {"hands": n, "points": pts, "net_cents": net, "ev_net_cents": ev, "high_cents": hi, "low_cents": lo}


_SORTS = {
    "time": CHRONO,
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


def hands(uid: int, sort: str, direction: str, offset: int, limit: int, filt: str,
          start: str = "", end: str = "") -> dict[str, Any]:
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
    rng, rp = _range_sql(start, end)
    where = "user_id=?" + _FILTERS[filt] + rng
    total = int(pub.DB.one(f"SELECT COUNT(*) n FROM review_hands WHERE {where}", (uid, *rp))["n"])
    rows = pub.DB.q(
        f"SELECT hand_key, played_at, played_ts, table_name, net_cents, ev_net_cents, allin, pot_cents, showdown,"
        f" decisions, acc_sum, acc_n, worst, mistakes, {_OWN_PENDING} AS pending, record"
        f" FROM review_hands WHERE {where} ORDER BY {_SORTS[sort].format(d=d)} LIMIT ? OFFSET ?",
        (uid, *rp, limit, offset),
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
    row = pub.DB.one(f"SELECT record, {_OWN_PENDING} AS pending FROM review_hands WHERE user_id=? AND hand_key=?",
                     (uid, key))
    if row is None:
        raise HTTPException(status_code=404, detail="Not Found")
    rec = json.loads(row["record"])
    rec["grading"] = bool(row["pending"])
    return rec


#: The player's hands an EARLIER network graded (SQL; takes the served network's id):
#: graded, their text kept, and not already waiting for their own marks (migration 3's
#: shown-hands job is replaced -- the regrade grades those decisions too). NULL
#: ``graded_by`` = graded before networks were recorded.
_OLDER = ("graded_at IS NOT NULL AND raw IS NOT NULL AND (graded_by IS NULL OR graded_by <> ?)"
          " AND (job IS NULL OR instr(job, '\"merge\":true') > 0)")


def _older_count(uid: int, mid: str | None) -> int:
    if mid is None:  # (a network with no checkpoint behind it: nothing to compare)
        return 0
    return int(pub.DB.one(f"SELECT COUNT(*) n FROM review_hands WHERE user_id=? AND {_OLDER}", (uid, mid))["n"])


def regrade(uid: int) -> dict[str, Any]:
    """Grade the player's hands again with the network served now — every one an earlier
    network graded (owner, 2026-10-05: "I would like my own uploaded hands to be
    regraded"). One UPDATE: each gets ``_REGRADE_JOB`` and the grader does the rest, the
    page counting them down like an upload's. 503 without a network."""
    from plo5bp.ui import homegame as hg

    model = hg._grading_model()
    if model is None:
        raise HTTPException(status_code=503, detail="The network isn't loaded on the server right now — try again in a minute.")
    mid = model_id(model)
    with pub.DB.transaction():
        n = _older_count(uid, mid)
        if n:
            pub.DB.q(f"UPDATE review_hands SET job=?, grade_attempts=0 WHERE user_id=? AND {_OLDER}",
                     (_REGRADE_JOB, uid, mid))
    CTX.grade_wake.set()
    return {"queued": n}


def delete_all(uid: int) -> dict[str, Any]:
    with CTX.lock:
        if uid in CTX.busy_users:
            raise HTTPException(status_code=409, detail="An upload is still being read — wait for it to finish.")
    with pub.DB.transaction():
        n = int(pub.DB.one("SELECT COUNT(*) n FROM review_hands WHERE user_id=?", (uid,))["n"])
        pub.DB.q("DELETE FROM review_hands WHERE user_id=?", (uid,))
        pub.DB.q("DELETE FROM review_uploads WHERE user_id=?", (uid,))
        pub.DB.q("DELETE FROM review_mistakes WHERE user_id=?", (uid,))
    with CTX.lock:
        CTX.drill.pop(uid, None)
    _profile_stale(uid)
    return {"deleted": n}


# --- "My tables": the profile of your tables (the Trainer deals from it) ---------------------


def _profile_stale(uid: int) -> None:
    with CTX.lock:
        CTX.profiles.pop(int(uid), None)
        CTX.profile_gen[int(uid)] = CTX.profile_gen.get(int(uid), 0) + 1


def my_tables(uid: int) -> dict[str, Any]:
    """The profile of the tables in the account's latest ``PROFILE_HANDS`` hands
    (``hr.table_profile``: players, ante, the account's own stack and its opponents'),
    with ``paid`` and ``min_hands``. Part of Hand review: an account that doesn't pay
    gets ``{"paid": False, "hands": 0}``. Cached until the account's hands change."""
    uid = int(uid)
    if not pub._paid(pub._user_by_id(uid)):
        return {"paid": False, "hands": 0, "min_hands": PROFILE_MIN_HANDS}
    with CTX.lock:
        prof = CTX.profiles.get(uid)
        gen = CTX.profile_gen.get(uid, 0)
    if prof is None:
        rows = pub.DB.q(
            "SELECT bb_cents, json_extract(record, '$.ante_cents') AS ante, json_extract(record, '$.seats') AS seats"
            " FROM review_hands WHERE user_id=? ORDER BY played_ts DESC, hand_key DESC LIMIT ?",
            (uid, PROFILE_HANDS),
        )
        prof = hr.table_profile([
            (int(r["bb_cents"]), int(r["ante"] or 0),
             [(int(s["start_cents"]), bool(s.get("is_me"))) for s in json.loads(r["seats"] or "[]")])
            for r in rows
        ])
        with CTX.lock:
            if CTX.profile_gen.get(uid, 0) == gen:  # (no upload / deletion landed meanwhile)
                CTX.profiles[uid] = prof
    return {"paid": True, "min_hands": PROFILE_MIN_HANDS, **prof}


def _my_tables_now() -> dict[str, Any] | None:
    """The signed-in player's ``my_tables`` (the Trainer's hook), None signed out."""
    uid = pub._CURRENT_USER_ID.get()
    return None if uid is None else my_tables(int(uid))


# --- the network's choice at a decision ------------------------------------------------------

_CHOICE_WHY = {
    "hidden_cards": "That player's cards weren't shown, so the network can't be asked about this decision.",
    "replay_refused": "The network's rules can't follow this hand to here (ClubGG allowed a raise after a "
                      "short all-in that they don't).",
    "replay_diverged": "The network's rules can't follow this hand to here.",
    "other_game": "The network plays PLO5 only.",
    "no_decision": "There's no decision there.",
    "no_board": "This hand's flops aren't in its history.",
}


def served_model() -> Any:
    """The served PLO5 network (the grader's) — 503 while there is none."""
    from plo5bp.ui import homegame as hg

    model = hg._grading_model()
    if model is None:
        raise HTTPException(status_code=503, detail="The network isn't loaded on the server right now — try again in a minute.")
    return model


def choice(rec: dict[str, Any], i: int) -> dict[str, Any]:
    """What the network plays at decision ``i`` of a hand record (Hand review's or a
    home game's): its fold / call / raise mix, its pick and its likeliest sizes — 400
    with the reason when the spot can't be rebuilt (cards not shown, another game, a
    line the network's rules don't follow)."""
    acts = rec.get("actions") or []
    if not 0 <= int(i) < len(acts):
        raise HTTPException(status_code=400, detail=_CHOICE_WHY["no_decision"])
    upto = rec.get("study_upto")
    if upto is not None and int(i) >= int(upto):
        raise HTTPException(status_code=400, detail=_CHOICE_WHY["replay_refused"])
    if acts[int(i)].get("auto"):
        raise HTTPException(status_code=400, detail="The clock acted there, not the player.")
    model = served_model()
    try:
        return hr.network_choice(rec, int(i), model)
    except hr.HandError as e:
        raise HTTPException(status_code=400, detail=_CHOICE_WHY.get(e.reason, "The network can't rebuild this spot.")) from e


# --- the mistakes drill ----------------------------------------------------------------------
# The Trainer deals your own mistakes back — every decision the network graded "wrong" or
# "blunder" (``review_mistakes``) — one spot at a time, in ROUNDS through them:
# - Worst first (the default): a spot's weight = how bad the mistake was x its learning
#   multiplier, and a round deals its spots in a weighted random order (Efraimidis-Spirakis
#   keys u^(1/w): the worst usually first, never one fixed order). Playing a spot right
#   (best / correct) halves its multiplier, down to 1/8, and a spot under 1 sits a round
#   out with probability 1 - multiplier: a fixed mistake comes back less often, but it
#   comes back. Missing it again (wrong / blunder) puts the multiplier back to at least 1,
#   x1.5 (up to 4), and deals it once more later in the same round. Close (an inaccuracy)
#   changes nothing.
# - Equal priority: every spot every round, in a uniformly shuffled order (the results
#   still count, for when you switch back).
# Either way the next spot comes from a DIFFERENT hand whenever there is one (a hand
# with two mistakes never deals them back to back — `_spread`).
# The round lives in memory (``CTX.drill``; a restart starts a new one); the learning
# state is in the database.

DRILL_FIXED = 0.5
DRILL_MISSED = 1.5
DRILL_MIN = 0.125
DRILL_MAX = 4.0
DRILL_FIXED_CATS = ("best", "correct")


def mistakes(uid: int) -> list[dict[str, Any]]:
    """Every mistake spot of the account with its learning state, worst first."""
    return [dict(r) for r in pub.DB.q(
        "SELECT hand_key AS key, idx AS i, score, cat, weight, fixed, missed FROM review_mistakes"
        " WHERE user_id=? ORDER BY score, played_ts DESC, hand_key, idx", (uid,))]


def severity(score: float) -> float:
    """How bad a mistake was: 1 for a blunder scored 0 down to 0.25 at the top of the
    "wrong move" band (scores below ``trainer.SCORING["inaccuracy_min"]``) — the worst
    one comes up about three times as often as the mildest."""
    from plo5bp.ui.trainer import SCORING

    top = float(SCORING["inaccuracy_min"])
    return 0.25 + 0.75 * (top - min(max(float(score), 0.0), top)) / top


def _spread(order: list[tuple[str, int]], last_key: str | None) -> list[tuple[str, int]]:
    """``order`` with no two spots of one hand in a row (nor first the hand dealt just
    before, ``last_key``) whenever that can be done, otherwise in its own order: each
    place takes the earliest spot of another hand than the one before — except, near a
    round's end, the hand that needs every other remaining place (more than half of
    what is left) goes first."""
    rest = list(order)
    counts: dict[str, int] = {}
    for k, _i in rest:
        counts[k] = counts.get(k, 0) + 1
    top = max(counts.values(), default=0)
    out: list[tuple[str, int]] = []
    prev = last_key
    while rest:
        n = len(rest)
        pick = None
        if 2 * top >= n:  # (only then can one hand need every other place)
            hand, c = max(counts.items(), key=lambda kv: kv[1])
            if c > n // 2 and hand != prev:
                pick = next(j for j, s in enumerate(rest) if s[0] == hand)
        if pick is None:
            pick = next((j for j, s in enumerate(rest) if s[0] != prev), 0)
        s = rest.pop(pick)
        counts[s[0]] -= 1
        out.append(s)
        prev = s[0]
    return out


def drill_order(spots: list[dict[str, Any]], prioritize: bool, rng: Any,
                last: tuple[str, int] | None = None) -> list[tuple[str, int]]:
    """One round of the drill: the spots it deals, in order (see the comment above)."""
    if not spots:
        return []
    if prioritize:
        chosen = [s for s in spots if float(s["weight"]) >= 1.0 or rng.random() < float(s["weight"])]
        if not chosen:  # (every spot fixed and none drawn: the likeliest one)
            chosen = [max(spots, key=lambda s: severity(s["score"]) * float(s["weight"]))]
        keyed = [(rng.random() ** (1.0 / (severity(s["score"]) * float(s["weight"]))), s) for s in chosen]
        keyed.sort(key=lambda kv: kv[0], reverse=True)
        chosen = [s for _k, s in keyed]
    else:
        chosen = list(spots)
        rng.shuffle(chosen)
    order = [(str(s["key"]), int(s["i"])) for s in chosen]
    return _spread(order, last[0] if last is not None else None)


def drill_next(uid: int, prioritize: bool, rng: Any = None) -> dict[str, Any] | None:
    """The next mistake spot to deal — None when the account has no mistakes (yet)."""
    import random

    _require_paid(uid)
    rng = rng or random.Random()
    for _attempt in range(4):
        with CTX.lock:
            st = CTX.drill.get(uid)
            if st is None or st["prioritize"] != bool(prioritize):  # (a new round in the new mode)
                st = CTX.drill[uid] = {
                    "queue": [], "round": st["round"] if st else 0, "size": 0, "total": 0,
                    "last": st["last"] if st else None, "prioritize": bool(prioritize), "again": set(),
                }
            fresh = not st["queue"]
        spots = mistakes(uid) if fresh else None
        with CTX.lock:
            if spots is not None and not st["queue"]:
                st["queue"] = drill_order(spots, bool(prioritize), rng, st["last"])
                st.update(round=st["round"] + 1, size=len(st["queue"]), total=len(spots), again=set())
            if not st["queue"]:
                return None
            key, i = st["queue"].pop(0)
            st["last"] = (key, i)
            meta = {"round": st["round"], "pos": st["size"] - len(st["queue"]), "size": st["size"],
                    "total": st["total"], "prioritize": bool(prioritize)}
        m = pub.DB.one("SELECT score, cat, weight, fixed, missed FROM review_mistakes"
                       " WHERE user_id=? AND hand_key=? AND idx=?", (uid, key, i))
        row = pub.DB.one("SELECT record FROM review_hands WHERE user_id=? AND hand_key=?", (uid, key))
        if m is None or row is None:
            continue  # (deleted meanwhile: the next one)
        rec = json.loads(row["record"])
        acts = rec.get("actions") or []
        if not 0 <= i < len(acts):
            continue
        return {
            "key": key, "i": i, "record": rec, **meta,
            "orig": {"cat": m["cat"], "score": round(float(m["score"]), 1), "label": acts[i].get("label") or ""},
            "learn": {"weight": round(float(m["weight"]), 3), "fixed": int(m["fixed"]), "missed": int(m["missed"])},
        }
    return None


def drill_result(uid: int, key: str, i: int, category: str) -> dict[str, Any] | None:
    """One attempt at a spot (not a practice repeat): right = its multiplier halves; a
    miss puts it back up and deals it once more later in this round."""
    import random

    if category in DRILL_FIXED_CATS:
        outcome = "fixed"
    elif category in MISTAKE_CATS:
        outcome = "missed"
    else:
        outcome = "close"
    with pub.DB.transaction():
        m = pub.DB.one("SELECT weight FROM review_mistakes WHERE user_id=? AND hand_key=? AND idx=?",
                       (uid, key, int(i)))
        if m is None:
            return None
        w = float(m["weight"])
        if outcome == "fixed":
            w = max(DRILL_MIN, w * DRILL_FIXED)
        elif outcome == "missed":
            w = min(DRILL_MAX, max(1.0, w) * DRILL_MISSED)
        pub.DB.q(
            "UPDATE review_mistakes SET weight=?, fixed=fixed+?, missed=missed+?, last_cat=?, last_at=?"
            " WHERE user_id=? AND hand_key=? AND idx=?",
            (w, int(outcome == "fixed"), int(outcome == "missed"), category, pub._now(), uid, key, int(i)),
        )
        m = pub.DB.one("SELECT fixed, missed FROM review_mistakes WHERE user_id=? AND hand_key=? AND idx=?",
                       (uid, key, int(i)))
    again = False
    if outcome == "missed":
        with CTX.lock:
            st = CTX.drill.get(uid)
            spot = (key, int(i))
            if st is not None and st["prioritize"] and st["queue"] and spot not in st["again"]:
                st["again"].add(spot)
                st["queue"].insert(random.randint(1, len(st["queue"])), spot)
                st["queue"] = _spread(st["queue"], key)  # (not straight after its own hand)
                st["size"] += 1
                again = True
    return {"outcome": outcome, "weight": round(w, 3), "fixed": int(m["fixed"]), "missed": int(m["missed"]),
            "again": again}


def drill_summary(uid: int) -> dict[str, Any]:
    r = pub.DB.one(
        "SELECT COUNT(*) n, COALESCE(SUM(weight < 1.0), 0) fixed, COALESCE(SUM(weight > 1.0), 0) tough,"
        " COALESCE(SUM(fixed + missed), 0) tries FROM review_mistakes WHERE user_id=?", (uid,),
    )
    return {"mistakes": int(r["n"]), "fixed": int(r["fixed"]), "struggling": int(r["tough"]),
            "attempts": int(r["tries"])}


# --- the API -----------------------------------------------------------------------------------


@router.get("/games/api/review/summary")
def api_summary(start: str = "", end: str = ""):
    """The numbers — of the hands played from `start` through `end` (YYYY-MM-DD,
    inclusive, both optional) — plus every hand's count and first / last time."""
    uid = _uid()
    _require_paid(uid)
    return summary(uid, start, end)


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


@router.post("/games/api/review/regrade")
def api_regrade():
    """Your hands an earlier network graded, graded again by the one served now
    (``{"queued": n}``; the summary's ``grading_pending`` counts them down)."""
    uid = _uid()
    _require_paid(uid)
    return regrade(uid)


@router.get("/games/api/review/uploads/{upload_id}")
def api_upload_status(upload_id: int):
    uid = _uid()
    r = pub.DB.one("SELECT * FROM review_uploads WHERE id=? AND user_id=?", (int(upload_id), uid))
    if r is None:
        raise HTTPException(status_code=404, detail="Not Found")
    out = _upload_view(r)
    out["grading_pending"] = int(pub.DB.one(
        f"SELECT COUNT(*) n FROM review_hands WHERE user_id=? AND {_OWN_PENDING}", (uid,))["n"])
    return out


@router.get("/games/api/review/series")
def api_series(start: str = "", end: str = ""):
    uid = _uid()
    _require_paid(uid)
    return series(uid, start, end)


@router.get("/games/api/review/hands")
def api_hands(sort: str = "time", dir: str = "desc", offset: int = 0, limit: int = 40, filter: str = "",
              start: str = "", end: str = ""):
    uid = _uid()
    _require_paid(uid)
    return hands(uid, sort, dir, offset, limit, filter, start, end)


@router.get("/games/api/review/hands/{key}")
def api_hand(key: str):
    uid = _uid()
    _require_paid(uid)
    return hand(uid, key)


@router.get("/games/api/review/hands/{key}/choice")
def api_hand_choice(key: str, i: int):
    """What the network plays at decision ``i`` of one of your hands (the replayer)."""
    uid = _uid()
    _require_paid(uid)
    return choice(hand(uid, key), i)


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
        pub.DB.q("DELETE FROM review_mistakes WHERE user_id=?", (int(uid),))
    with CTX.lock:
        CTX.drill.pop(int(uid), None)
    _profile_stale(uid)


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
    # (each answer is a forward of the network)
    hg.API_COST.setdefault("/games/api/review/hands/{key}/choice", 3.0)
    # The Trainer's mistakes drill reads its spots from here (trainer.set_drill_hooks).
    from plo5bp.ui import trainer as tr

    tr.set_drill_hooks(
        next_spot=lambda prioritize: drill_next(_uid(), prioritize),
        record=lambda key, i, category: drill_result(_uid(), key, i, category),
    )
    # ... and "My tables" deals like the tables in your hands (trainer.set_my_tables_hook).
    tr.set_my_tables_hook(_my_tables_now)

    @app.get("/games/review")
    def review_page():
        return HTMLResponse(hg._page_html(static_dir), headers=dict(hg.PAGE_HEADERS))

    app.include_router(router)
    app.router.on_shutdown.append(stop_workers)
    _start_workers()
    logger.info("hand review installed")
