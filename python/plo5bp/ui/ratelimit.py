"""In-process rate limiting and work gating for the web app (PERF-013/014).

Three small, thread-safe primitives — no external services, O(1) per call:

- :class:`RateLimiter` — a token bucket per key (user id, IP, …):
  ``allow(key, cost) -> (ok, retry_after_s)``. Idle keys are swept.
- :class:`KeyedCounter` — at most ``limit`` concurrent holders per key
  (``enter`` / ``leave``), e.g. open live streams per user.
- :class:`WorkGate` — an asyncio-friendly FIFO semaphore with a wait
  timeout: at most ``slots`` holders at once, waiters queue on the event loop
  (never on a worker thread). One gate per user serialises that user's heavy
  requests; one global gate caps CPU-heavy model work site-wide. It works
  across event loops (Starlette's TestClient runs one loop per request): a
  waiter is woken on its own loop with ``call_soon_threadsafe``.

Reusable by any router (home games' SEC-009 included)."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Hashable


class RateLimited(Exception):
    """Raised by :meth:`RateLimiter.check`; ``retry_after`` in seconds."""

    def __init__(self, retry_after: float):
        super().__init__(f"rate limited, retry after {retry_after:.1f}s")
        self.retry_after = float(retry_after)


class RateLimiter:
    """Token bucket per key: ``burst`` tokens, refilled at ``rate`` per second.

    ``allow`` spends ``cost`` tokens when available. Buckets idle longer than
    ``idle_ttl`` seconds are dropped (a full bucket is the same as no bucket),
    so memory stays bounded by the keys active in that window."""

    def __init__(self, rate: float, burst: float, *, idle_ttl: float = 900.0,
                 clock: Any = time.monotonic):
        if rate <= 0 or burst <= 0:
            raise ValueError("rate and burst must be positive")
        self.rate = float(rate)
        self.burst = float(burst)
        self.idle_ttl = float(idle_ttl)
        self._clock = clock
        self._buckets: dict[Hashable, list[float]] = {}  # key -> [tokens, last]
        self._lock = threading.Lock()
        self._calls = 0

    def allow(self, key: Hashable, cost: float = 1.0) -> tuple[bool, float]:
        now = self._clock()
        with self._lock:
            self._calls += 1
            if self._calls % 1024 == 0:
                self._sweep(now)
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = [self.burst, now]
            tokens = min(self.burst, b[0] + (now - b[1]) * self.rate)
            b[1] = now
            if tokens >= cost:
                b[0] = tokens - cost
                return True, 0.0
            b[0] = tokens
            return False, (cost - tokens) / self.rate

    def check(self, key: Hashable, cost: float = 1.0) -> None:
        ok, retry = self.allow(key, cost)
        if not ok:
            raise RateLimited(retry)

    def reset(self, key: Hashable | None = None) -> None:
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)

    def _sweep(self, now: float) -> None:
        cutoff = now - self.idle_ttl
        for k in [k for k, b in self._buckets.items() if b[1] < cutoff]:
            del self._buckets[k]

    def __len__(self) -> int:
        return len(self._buckets)


class KeyedCounter:
    """At most ``limit`` concurrent holders per key."""

    def __init__(self, limit: int):
        self.limit = int(limit)
        self._counts: dict[Hashable, int] = {}
        self._lock = threading.Lock()

    def enter(self, key: Hashable) -> bool:
        with self._lock:
            n = self._counts.get(key, 0)
            if n >= self.limit:
                return False
            self._counts[key] = n + 1
            return True

    def leave(self, key: Hashable) -> None:
        with self._lock:
            n = self._counts.get(key, 0) - 1
            if n > 0:
                self._counts[key] = n
            else:
                self._counts.pop(key, None)

    def count(self, key: Hashable) -> int:
        with self._lock:
            return self._counts.get(key, 0)


class WorkGate:
    """FIFO semaphore for coroutines, with a queue limit and a wait timeout.

    ``slots`` holders at once; ``max_waiting`` more may queue (None = no
    limit). :meth:`acquire` returns False — without holding a slot — when the
    queue is full or ``timeout`` passes. A released slot is handed straight to
    the oldest waiter, so nobody can barge past the queue."""

    def __init__(self, slots: int, *, max_waiting: int | None = None):
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.slots = int(slots)
        self.max_waiting = max_waiting
        self._busy = 0
        self._waiters: deque[tuple[asyncio.AbstractEventLoop, asyncio.Future]] = deque()
        self._lock = threading.Lock()

    @property
    def busy(self) -> int:
        with self._lock:
            return self._busy

    @property
    def waiting(self) -> int:
        with self._lock:
            return sum(1 for _, f in self._waiters if not f.done())

    async def acquire(self, timeout: float | None = None) -> bool:
        with self._lock:
            if self._busy < self.slots and not self._waiters:
                self._busy += 1
                return True
            if self.max_waiting is not None:
                live = sum(1 for _, f in self._waiters if not f.done())
                if live >= self.max_waiting:
                    return False
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            entry = (loop, fut)
            self._waiters.append(entry)
        try:
            if timeout is None:
                await fut
            else:
                await asyncio.wait_for(fut, timeout)
            return True
        except asyncio.TimeoutError:
            self._abandon(entry, fut)
            return False
        except BaseException:
            self._abandon(entry, fut)
            raise

    def _abandon(
        self, entry: tuple[asyncio.AbstractEventLoop, asyncio.Future],
        fut: asyncio.Future,
    ) -> None:
        """A waiter gave up (timeout / cancellation). Still queued -> it just
        leaves the queue. Already handed a slot -> `_grant` sees the cancelled
        future and passes the slot on. Granted in the very instant the wait
        ended -> it owns the slot, so it gives it back. Nothing leaks."""
        with self._lock:
            try:
                self._waiters.remove(entry)
            except ValueError:
                pass
        if fut.done() and not fut.cancelled():
            self.release()
        elif not fut.done():
            fut.cancel()

    def release(self) -> None:
        with self._lock:
            while self._waiters:
                loop, fut = self._waiters.popleft()
                if fut.done():
                    continue
                # Hand the slot over directly (the busy count is unchanged).
                try:
                    loop.call_soon_threadsafe(self._grant, fut)
                    return
                except RuntimeError:  # that loop is closed: try the next
                    continue
            self._busy = max(0, self._busy - 1)

    def _grant(self, fut: asyncio.Future) -> None:
        if fut.done():  # timed out / cancelled meanwhile: pass the slot on
            self.release()
        else:
            fut.set_result(None)

    @asynccontextmanager
    async def hold(self, timeout: float | None = None) -> AsyncIterator[bool]:
        """``async with gate.hold(t) as ok:`` — ``ok`` False = no slot (the
        body runs anyway; check it). Released on exit when held."""
        ok = await self.acquire(timeout)
        try:
            yield ok
        finally:
            if ok:
                self.release()


class KeyedGates:
    """One :class:`WorkGate` per key (e.g. per user), created on demand and
    dropped again when idle (no holder, no waiter)."""

    def __init__(self, slots: int = 1, *, max_waiting: int | None = None):
        self.slots = int(slots)
        self.max_waiting = max_waiting
        self._gates: dict[Hashable, WorkGate] = {}
        self._refs: dict[Hashable, int] = {}
        self._lock = threading.Lock()

    @asynccontextmanager
    async def hold(self, key: Hashable, timeout: float | None = None) -> AsyncIterator[bool]:
        with self._lock:
            gate = self._gates.get(key)
            if gate is None:
                gate = self._gates[key] = WorkGate(self.slots, max_waiting=self.max_waiting)
            self._refs[key] = self._refs.get(key, 0) + 1
        try:
            async with gate.hold(timeout) as ok:
                yield ok
        finally:
            with self._lock:
                n = self._refs.get(key, 1) - 1
                if n <= 0:
                    self._refs.pop(key, None)
                    self._gates.pop(key, None)
                else:
                    self._refs[key] = n

    def __len__(self) -> int:
        with self._lock:
            return len(self._gates)
