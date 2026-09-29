"""Thread-safe cache of built strategy views for the CFR app server (TOOL-022).

The server used to keep ONE view in a shared dict and write its ``key`` and
``view`` as two separate steps from threadpool handlers (a live-view tick and a
file open run at the same time), so an interleaving could pair one file's key
with another's view — and the viewer then showed the wrong strategy.

:class:`ViewCache` stores ``key -> view`` entries under a lock (a small LRU: the
live job and the file the user flips to both stay warm), and
:meth:`ViewCache.get_or_build` builds each key at most once at a time: a second
request for a key that is being built waits for that build instead of parsing
the same 100+ MB report again.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable, Hashable

DEFAULT_ENTRIES = 2


class ViewCache:
    def __init__(self, max_entries: int = DEFAULT_ENTRIES) -> None:
        self._max = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._items: OrderedDict[Hashable, Any] = OrderedDict()
        self._building: dict[Hashable, threading.Lock] = {}

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            view = self._items.get(key)
            if view is not None:
                self._items.move_to_end(key)
            return view

    def put(self, key: Hashable, view: Any) -> None:
        if view is None:
            return
        with self._lock:
            self._items[key] = view
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def get_or_build(self, key: Hashable, build: Callable[[], Any]) -> Any:
        """The cached view for ``key``, building (once) and caching it if absent.
        ``build`` exceptions propagate and cache nothing."""
        view = self.get(key)
        if view is not None:
            return view
        with self._lock:
            gate = self._building.setdefault(key, threading.Lock())
        with gate:
            view = self.get(key)  # built by the request we waited for
            if view is None:
                view = build()
                self.put(key, view)
        with self._lock:
            if self._building.get(key) is gate and not gate.locked():
                del self._building[key]
        return view

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
