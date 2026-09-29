"""(TOOL-022) The server's view cache under concurrent handlers.

The old cache wrote ``key`` and ``view`` as two steps into one shared dict from
threadpool handlers; an interleaving could serve one file's view under another's
key. ``ViewCache`` keeps (key -> view) under a lock and builds a key once.
"""

from __future__ import annotations

import threading
import time

from plo5bp.cfr_app.view_cache import ViewCache


def test_views_always_match_their_key_under_concurrent_builds():
    cache = ViewCache(max_entries=2)
    errors: list[str] = []

    def build_for(key):
        def build():
            time.sleep(0.001)
            return {"key": key}
        return build

    def worker(i: int):
        for n in range(200):
            key = ("file", (i + n) % 5)
            view = cache.get_or_build(key, build_for(key))
            if view["key"] != key:
                errors.append(f"{key} got {view['key']}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(cache) <= 2


def test_one_build_per_key_even_when_requests_overlap():
    cache = ViewCache()
    calls: list[int] = []
    release = threading.Event()

    def slow_build():
        calls.append(1)
        release.wait(5)
        return {"v": 1}

    out: list[dict] = []
    threads = [threading.Thread(target=lambda: out.append(cache.get_or_build("k", slow_build)))
               for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    release.set()
    for t in threads:
        t.join()
    assert calls == [1] and len(out) == 4 and all(v is out[0] for v in out)


def test_lru_eviction_and_failed_builds_cache_nothing():
    cache = ViewCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # touch a → b is the oldest
    cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3

    def boom():
        raise ValueError("bad file")

    try:
        cache.get_or_build("d", boom)
    except ValueError:
        pass
    assert cache.get("d") is None
    assert cache.get_or_build("d", lambda: 4) == 4
