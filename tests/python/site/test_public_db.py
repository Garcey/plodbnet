"""The public service's database layer and per-user registry, as units
(no app boot): per-thread read connections (PERF-012 / PERF-018), numbered
migrations with a refusal for unknown BREAKING steps (OPS-029 / OPS-043), and
the runtime registry's LRU / idle sweep (PERF-024 / TEST-013).

`plo5bp.ui.public` opens its own database at import; these tests build their
own `Db` objects on temp files, so they import it through the shared
`boot_public_server` env (a temp DB) to keep the real `data/` untouched."""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest


@pytest.fixture(scope="module")
def pub(boot_public_server):
    boot_public_server()
    import sys

    return sys.modules["plo5bp.ui.public"]


def _db(pub, tmp_path, name="t.db"):
    db = pub.Db(tmp_path / name)
    db.q("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    return db


def test_reads_on_other_threads_do_not_wait_for_the_writer(pub, tmp_path):
    """A SELECT on another thread runs on that thread's read-only connection:
    it completes while a transaction holds the writer lock, and it sees only
    COMMITTED data (read-committed, like before — never half a transaction)."""
    db = _db(pub, tmp_path)
    db.q("INSERT INTO t(v) VALUES('committed')")
    entered = threading.Event()
    release = threading.Event()

    def writer():
        with db.transaction():
            db.q("INSERT INTO t(v) VALUES('in-flight')")
            # A read INSIDE the transaction sees its own write.
            assert db.one("SELECT COUNT(*) c FROM t")["c"] == 2
            entered.set()
            release.wait(5)

    th = threading.Thread(target=writer)
    th.start()
    assert entered.wait(5)
    got: list = []
    reader = threading.Thread(target=lambda: got.append(db.one("SELECT COUNT(*) c FROM t")["c"]))
    t0 = time.monotonic()
    reader.start()
    reader.join(3)
    assert got == [1], "the reader saw uncommitted data or was blocked"
    assert time.monotonic() - t0 < 2.0
    release.set()
    th.join(5)
    assert db.one("SELECT COUNT(*) c FROM t")["c"] == 2
    db.close()


def test_reader_connections_cannot_write(pub, tmp_path):
    db = _db(pub, tmp_path)
    reader = db._reader()
    with pytest.raises(sqlite3.OperationalError):
        reader.execute("INSERT INTO t(v) VALUES('x')")
    # Writes are recognised and routed to the writer, even spelled oddly.
    db.q("  insert into t(v) values('y')")
    db.q("WITH x AS (SELECT 1) INSERT INTO t(v) SELECT 'z' FROM x")
    assert [r["v"] for r in db.q("SELECT v FROM t ORDER BY id")] == ["y", "z"]
    db.close()


def test_write_listeners_fire_after_commit_only(pub, tmp_path):
    db = _db(pub, tmp_path)
    seen: list[str] = []
    db.write_listeners.append(seen.append)
    db.q("SELECT 1")
    assert seen == []
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.q("INSERT INTO t(v) VALUES('rolled back')")
            raise RuntimeError
    assert seen == []
    with db.transaction():
        db.q("INSERT INTO t(v) VALUES('kept')")
        assert seen == []  # not before the commit
    assert len(seen) == 1 and "INSERT" in seen[0]
    db.close()


def test_migrations_apply_once_in_order_and_are_recorded(pub, tmp_path):
    db = _db(pub, tmp_path)
    M = pub.Migration
    steps = [
        M(1, "add a", fn=pub._add_column("t", "a", "INTEGER NOT NULL DEFAULT 0")),
        M(2, "table u", statements=("CREATE TABLE IF NOT EXISTS u (x TEXT)",)),
    ]
    assert db.migrate("comp", steps) == [1, 2]
    assert db.migrate("comp", steps) == []  # idempotent
    assert db.schema_versions()["comp"] == 2
    cols = {r[1] for r in db._conn.execute("PRAGMA table_info(t)")}
    assert "a" in cols
    # A column added by pre-versioning code: the step is a no-op, not an error.
    db.q("ALTER TABLE t ADD COLUMN b TEXT")
    assert db.migrate("comp2", [M(1, "add b", fn=pub._add_column("t", "b", "TEXT"))]) == [1]
    db.close()


def test_a_failed_migration_rolls_back_and_is_not_recorded(pub, tmp_path):
    db = _db(pub, tmp_path)

    def boom(conn):
        conn.execute("CREATE TABLE half (x TEXT)")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        db.migrate("c", [pub.Migration(1, "boom", fn=boom)])
    assert db.schema_versions().get("c") is None
    names = {r[0] for r in db._conn.execute("SELECT name FROM sqlite_master")}
    assert "half" not in names
    db.close()


def test_older_code_refuses_a_breaking_newer_schema(pub, tmp_path, caplog):
    """(OPS-043) Rollback safety: after a BREAKING step, code that does not
    know it refuses to start, loudly; unknown ADDITIVE steps only warn."""
    db = _db(pub, tmp_path)
    M = pub.Migration
    new_code = [M(1, "a", statements=("CREATE TABLE IF NOT EXISTS a (x)",)),
                M(2, "b", statements=("CREATE TABLE IF NOT EXISTS b (x)",))]
    db.migrate("c", new_code)
    db.migrate("c", new_code[:1])  # rolled-back code: additive step 2 unknown
    assert any("ahead of this code" in r.getMessage() for r in caplog.records)
    db.migrate("c", new_code + [M(3, "drop a", statements=("DROP TABLE a",), breaking=True)])
    with pytest.raises(pub.SchemaTooNew):
        db.migrate("c", new_code)
    db.close()


def test_the_real_database_carries_the_public_migrations(pub):
    versions = pub.DB.schema_versions()
    assert versions["public"] == max(m.version for m in pub.PUBLIC_MIGRATIONS)
    cols = {r[1] for r in pub.DB._conn.execute("PRAGMA table_info(users)")}
    assert {"disabled", "session_version", "deleted_at", "homegame_access"} <= cols


# --- Registry (PERF-024 / TEST-013) ------------------------------------------------------


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _registry(pub, **kw):
    made: list[int] = []

    def trainer(uid):
        made.append(uid)
        return f"ts{uid}"

    return pub.Registry(lambda: object(), trainer, **kw), made


def test_registry_lru_order_and_capacity(pub):
    reg, made = _registry(pub, capacity=2)
    a, b = reg.get(1), reg.get(2)
    assert reg.get(1) is a          # touch 1: 2 is now least recent
    reg.get(3)                      # evicts 2
    assert reg.peek(2) is None and reg.peek(1) is a and reg.peek(3) is not None
    assert made == [1, 2, 3]
    assert reg.get(2) is not b      # a fresh runtime after eviction


def test_registry_sweeps_idle_runtimes(pub):
    clock = _Clock()
    reg, _ = _registry(pub, idle_s=3600, clock=clock)
    reg.get(1)
    clock.t += 1800
    reg.get(2)
    clock.t += 1900                 # user 1 idle 3700 s, user 2 1900 s
    assert reg.sweep() == 1
    assert reg.peek(1) is None and reg.peek(2) is not None
    # The sweep also runs by itself (at most once a minute) from get().
    clock.t += 4000
    reg.get(3)
    assert reg.peek(2) is None and len(reg) == 1


def test_registry_builds_outside_its_lock(pub):
    """Building a runtime reads the user's stats file: other users' lookups
    must not wait for it."""
    started, release = threading.Event(), threading.Event()

    def slow_trainer(uid):
        if uid == 1:
            started.set()
            release.wait(5)
        return f"ts{uid}"

    reg = pub.Registry(lambda: object(), slow_trainer)
    th = threading.Thread(target=reg.get, args=(1,))
    th.start()
    assert started.wait(5)
    t0 = time.monotonic()
    reg.get(2)
    assert time.monotonic() - t0 < 1.0
    release.set()
    th.join(5)
    assert reg.peek(1) is not None


def test_a_request_keeps_its_runtime_through_an_eviction(pub, monkeypatch):
    """(BE-003) The runtime is resolved once per request; evicting it mid-
    request does not swap the object under the handler."""
    reg, _ = _registry(pub, capacity=1)
    monkeypatch.setattr(pub, "_REGISTRY", reg)
    token = pub._REQUEST_RUNTIME.set([None, 1])
    try:
        first = pub._current_runtime()
        reg.get(2)  # another user's request evicts user 1
        assert reg.peek(1) is None
        assert pub._current_runtime() is first
    finally:
        pub._REQUEST_RUNTIME.reset(token)
