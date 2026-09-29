"""Home games under concurrency (improvements backlog, 2026-09-28).

HGB-018: the lock order is written down in homegame.py ("locks (HGB-018)") and
the ``*_locked`` naming rule is ENFORCED in the test session — every such
function (and ``_view`` / ``_mutation`` / ``_persist_safe`` / ``_cash_out_seat``)
checks that the calling thread holds its table's lock, so a bare call anywhere
in the code a test reaches fails that test instead of racing the clock in
production.

TEST-006: many threads act, deal, sit, leave, top up, chat, watch and stream
on two tables at once for a few seconds, with the real clock thread running.
A deadlock fails the test (with every thread's stack), so does any 500, and
the money must add up exactly at the end: every table's ledger is zero-sum,
every hand's results sum to zero, the ledger rows agree with the players'
running totals.

Booted once for the module in PUBLIC mode against a temp DB
(`boot_public_server`, tests/python/conftest.py).
"""

from __future__ import annotations

import random
import sys
import threading
import time
import traceback

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"
NAMES = ["ann", "ben", "cal", "dee", "eli", "fay", "gus", "hal"]


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server()


@pytest.fixture(scope="module")
def hg(server):
    return sys.modules["plo5bp.ui.homegame"]


@pytest.fixture(scope="module")
def cast(server):
    def login(email):
        c = TestClient(server.app, raise_server_exceptions=False)
        assert c.get("/auth/dev", params={"email": email}).status_code == 200
        return c

    adm = login(ADMIN_EMAIL)
    players = [login(f"{n}@example.com") for n in NAMES]
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for n in NAMES:
        r = adm.post("/admin/api/games_access", json={"user_id": ids[f"{n}@example.com"], "action": "grant"})
        assert r.status_code == 200
    uid = {n: ids[f"{n}@example.com"] for n in NAMES}
    by_uid = {uid[n]: players[i] for i, n in enumerate(NAMES)}
    return {"p": players, "adm": adm, "uid": uid, "by_uid": by_uid, "app": server.app}


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _post(client, gid, what, body=None):
    return client.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _table(cast, players, **kw):
    """A table hosted by ``players[0]`` with the others seated in order."""
    p = cast["p"]
    body = {"name": "stress", "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 4000}
    body.update(kw)
    gid = _ok(p[players[0]].post("/games/api/tables", json=body))["id"]
    for seat, i in enumerate(players[1:], start=1):
        _ok(_post(p[i], gid, "sit", {"seat": seat, "buyin_cents": 4000}))
    return gid


# --- HGB-018: the `*_locked` rule is checked ------------------------------------------------


def test_lock_checks_are_on_and_cover_every_locked_function(hg):
    assert hg.LOCK_CHECKS, "the test session runs with the lock checks on"
    names = hg._locked_functions(vars(hg))
    assert len(names) > 40, names  # (the whole family, not a lucky few)
    unchecked = [n for n in names if not getattr(getattr(hg, n), "__hg_lock_checked__", False)]
    assert not unchecked, f"not lock-checked: {unchecked}"


def test_a_bare_call_of_a_locked_function_fails(cast, hg):
    gid = _table(cast, [0, 1])
    t = hg.HUB.get(gid)
    with pytest.raises(AssertionError, match="lock is not held"):
        hg._settle_locked(t)
    with pytest.raises(AssertionError, match="lock is not held"):
        hg._view(t, cast["uid"]["ann"])
    with t.lock:  # (held: fine — and re-entrant, as the code relies on)
        hg._settle_locked(t)
        with t.lock:
            assert hg._view(t, cast["uid"]["ann"])["id"] == gid

    # another thread holding the lock is not "held" for this one
    held, release = threading.Event(), threading.Event()

    def holder():
        with t.lock:
            held.set()
            release.wait(5)

    th = threading.Thread(target=holder, daemon=True)
    th.start()
    assert held.wait(5)
    try:
        with pytest.raises(AssertionError):
            hg._assert_lock_held(t)
    finally:
        release.set()
        th.join(5)
    with t.lock:
        hg._close_locked(t, cast["uid"]["ann"])


# --- TEST-006: many threads, two tables, a few seconds ----------------------------------------


def _stacks_of_every_thread() -> str:
    frames = sys._current_frames()
    out = []
    for th in threading.enumerate():
        fr = frames.get(th.ident)
        if fr is not None:
            out.append(f"--- {th.name}\n" + "".join(traceback.format_stack(fr)[-12:]))
    return "\n".join(out)


def test_many_threads_on_two_tables_never_deadlock_and_the_money_adds_up(cast, hg):
    p, uid = cast["p"], cast["uid"]
    # two 4-handed tables, the server dealing every second, a short shot clock
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    gids = [_table(cast, g, deal_delay_secs=1, decision_secs=5, allow_rathole=True) for g in groups]
    for g, gid in zip(groups, gids):
        _ok(_post(p[g[0]], gid, "run", {"running": True}))
    table_of = {i: gid for g, gid in zip(groups, gids) for i in g}
    host_of = {gid: g[0] for g, gid in zip(groups, gids)}

    stop = threading.Event()
    lock = threading.Lock()
    errors: list[str] = []
    counts = {"requests": 0, "acts": 0, "deals": 0, "sits": 0, "leaves": 0, "chats": 0, "topups": 0}

    def call(fn, *a, **kw):
        r = fn(*a, **kw)
        with lock:
            counts["requests"] += 1
            if r.status_code >= 500:
                errors.append(f"{r.status_code} {r.request.method} {r.request.url.path}: {r.text[:300]}")
        return r

    def player(i: int) -> None:
        c, gid, rng = p[i], table_of[i], random.Random(1000 + i)
        try:
            while not stop.is_set():
                r = call(c.get, f"/games/api/tables/{gid}")
                if r.status_code != 200:
                    continue
                s = r.json()
                me = s["my_seat"]
                roll = rng.random()
                if me is None:
                    # re-sit at a free seat (a buy-in; a race for a seat is a 409)
                    free = [x["seat"] for x in s["seats"] if x["empty"] and not x["reserved_by"]]
                    if free and roll < 0.5:
                        r = call(_post, c, gid, "sit", {"seat": rng.choice(free), "buyin_cents": 4000})
                        if r.status_code == 200:
                            with lock:
                                counts["sits"] += 1
                    continue
                if s["phase"] == "in_hand" and s["actor"] == me:
                    legal = [g for g in ("fold", "check_call", "raise") if s["legal"][g]]
                    gate = rng.choice(legal + ["check_call"] * 2 if "check_call" in legal else legal)
                    body = {"gate": gate, "hand_no": s["hand_no"], "action_seq": s["action_seq"]}
                    if gate == "raise":
                        rb = s["raise_bounds"]
                        body["chips"] = rng.randint(rb["min_chips"], max(rb["min_chips"], rb["max_chips"]))
                    if call(_post, c, gid, "act", body).status_code == 200:
                        with lock:
                            counts["acts"] += 1
                elif roll < 0.06:
                    if call(_post, c, gid, "chat", {"text": f"gl {i}"}).status_code == 200:
                        with lock:
                            counts["chats"] += 1
                elif roll < 0.10 and s["phase"] != "in_hand":
                    if call(_post, c, gid, "deal", {"hand_no": s["hand_no"]}).status_code == 200:
                        with lock:
                            counts["deals"] += 1
                elif roll < 0.13:
                    call(_post, c, gid, "sit_out", {"on": rng.random() < 0.5, "next_hand": rng.random() < 0.5})
                elif roll < 0.15:
                    call(_post, c, gid, "sit_out", {"on": False})
                elif roll < 0.17 and i != host_of[gid]:
                    if call(_post, c, gid, "leave", {"now": rng.random() < 0.3}).status_code == 200:
                        with lock:
                            counts["leaves"] += 1
                elif roll < 0.20:
                    if call(_post, c, gid, "rebuy", {"amount_cents": 500, "queue": True}).status_code == 200:
                        with lock:
                            counts["topups"] += 1
                elif roll < 0.21:
                    call(_post, c, gid, "remove_chips", {"amount_cents": 100, "queue": True})
                elif roll < 0.23:
                    call(_post, c, gid, "show", {"hand_no": s["hand_no"]})
                elif roll < 0.24:
                    call(_post, c, gid, "react", {"emote": "gg"})
                elif roll < 0.25:
                    call(c.get, f"/games/api/tables/{gid}/hands")
                elif roll < 0.26:
                    call(c.get, "/games/api/tables")
                elif roll < 0.27:
                    call(c.get, "/games/api/my/stats")
        except Exception:  # noqa: BLE001 — reported, never swallowed
            with lock:
                errors.append(f"player {i}: {traceback.format_exc()}")

    def streamer(i: int) -> None:
        c, gid = p[i], table_of[i]
        try:
            while not stop.is_set():
                with c.stream("GET", f"/games/api/tables/{gid}/stream", params={"max_events": 6}) as r:
                    if r.status_code >= 500:
                        with lock:
                            errors.append(f"stream {r.status_code}")
                    for _ in r.iter_text():
                        pass
        except Exception:  # noqa: BLE001
            with lock:
                errors.append(f"streamer {i}: {traceback.format_exc()}")

    threads = [threading.Thread(target=player, args=(i,), name=f"player-{i}", daemon=True) for i in range(8)]
    threads += [threading.Thread(target=streamer, args=(i,), name=f"stream-{i}", daemon=True) for i in (1, 5)]
    for th in threads:
        th.start()
    time.sleep(6.0)
    stop.set()
    deadline = time.monotonic() + 90.0
    for th in threads:
        th.join(max(0.1, deadline - time.monotonic()))
    alive = [th.name for th in threads if th.is_alive()]
    assert not alive, f"deadlocked: {alive}\n{_stacks_of_every_thread()}"
    assert not errors, "\n".join(errors[:10])
    print("stress:", counts)  # (shown with -s)
    assert counts["acts"] >= 10 and counts["requests"] >= 200, counts

    # --- wind down: pause, play the hand in progress out, close ------------------------------
    for gid in gids:
        host = p[host_of[gid]]
        _ok(_post(host, gid, "run", {"running": False}))
        t = hg.HUB.get(gid)
        for _ in range(200):
            with t.lock:
                if t.phase != "in_hand" or t.env is None or t.env.current_actor() is None:
                    break
                hg._host_fold_locked(t, t.host_user_id)
        with t.lock:  # skip the showdown's award animation
            if t.runout_active and t.runout_started_mono is not None:
                t.runout_started_mono -= 600.0
            hg._settle_locked(t)
            host_uid = t.host_user_id
        _ok(_post(cast["by_uid"][host_uid], gid, "close"))

    # --- the money ------------------------------------------------------------------------------
    q = hg.pub.DB.q
    for gid in gids:
        totals = q("SELECT COALESCE(SUM(buyin_cents),0) b, COALESCE(SUM(leftover_cents),0) l, "
                   "COALESCE(SUM(stack_chips),0) s FROM homegame_players WHERE game_id=?", (gid,))[0]
        assert totals["s"] == 0, "a closed table holds no chips"
        assert totals["b"] == totals["l"], f"{gid}: bought in {totals['b']}, cashed out {totals['l']}"
        # the ledger rows add up to each player's running totals (OPS-017)
        ins, outs = ",".join(f"'{k}'" for k in hg.LEDGER_IN), ",".join(f"'{k}'" for k in hg.LEDGER_OUT)
        for r in q("SELECT user_id, buyin_cents, leftover_cents FROM homegame_players WHERE game_id=?", (gid,)):
            led = q(f"SELECT COALESCE(SUM(CASE WHEN kind IN ({ins}) THEN amount_cents END),0) i, "
                    f"COALESCE(SUM(CASE WHEN kind IN ({outs}) THEN amount_cents END),0) o "
                    "FROM homegame_ledger WHERE game_id=? AND user_id=?", (gid, r["user_id"]))[0]
            assert (led["i"], led["o"]) == (r["buyin_cents"], r["leftover_cents"]), (gid, dict(r), dict(led))
        assert hg.reconcile_ledger(gid) == []
        # every hand's exact result is zero-sum
        bad = q("SELECT hand_no, SUM(delta_chips) s FROM homegame_hand_results WHERE game_id=? "
                "GROUP BY hand_no HAVING SUM(delta_chips) <> 0", (gid,))
        assert not bad, [dict(x) for x in bad]
        hands = q("SELECT COUNT(*) c FROM homegame_hands WHERE game_id=?", (gid,))[0]["c"]
        assert hands >= 1, "the tables played"
    health = hg.health()
    assert health["clock"]["alive"] and not health["tables"]["failing"] and not health["tables"]["unsaved"], health
    assert uid  # (the cast)
