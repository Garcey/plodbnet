"""plo5bp.ui.ratelimit: token buckets, keyed counters and the async work gate
(PERF-013 / PERF-014)."""

from __future__ import annotations

import asyncio
import threading

import pytest

from plo5bp.ui.ratelimit import KeyedCounter, KeyedGates, RateLimited, RateLimiter, WorkGate


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_bucket_allows_a_burst_then_throttles_to_the_rate():
    clock = FakeClock()
    rl = RateLimiter(rate=2.0, burst=5, clock=clock)
    assert all(rl.allow("u")[0] for _ in range(5))
    ok, retry = rl.allow("u")
    assert not ok and retry == pytest.approx(0.5)
    clock.t += 0.5  # one token back
    assert rl.allow("u") == (True, 0.0)
    assert rl.allow("u")[0] is False
    # other keys are independent
    assert rl.allow("v")[0] is True


def test_bucket_cost_and_check():
    clock = FakeClock()
    rl = RateLimiter(rate=1.0, burst=3, clock=clock)
    rl.check("u", cost=2)
    with pytest.raises(RateLimited) as e:
        rl.check("u", cost=2)
    assert e.value.retry_after == pytest.approx(1.0)
    clock.t += 10
    rl.check("u", cost=3)  # refills only up to the burst


def test_idle_buckets_are_swept():
    clock = FakeClock()
    rl = RateLimiter(rate=1.0, burst=1, idle_ttl=60, clock=clock)
    for i in range(100):
        rl.allow(i)
    clock.t += 120
    for _ in range(1024):  # the sweep runs every 1024 calls
        rl.allow("live")
    assert len(rl) == 1


def test_keyed_counter_caps_per_key():
    c = KeyedCounter(2)
    assert c.enter("a") and c.enter("a")
    assert not c.enter("a")
    assert c.enter("b")
    c.leave("a")
    assert c.count("a") == 1 and c.enter("a")


def _run(coro):
    return asyncio.run(coro)


def test_gate_slots_fifo_and_timeout():
    async def main():
        gate = WorkGate(1)
        order: list[int] = []
        assert await gate.acquire(0.1)

        async def waiter(i: int) -> None:
            assert await gate.acquire(2.0)
            order.append(i)
            await asyncio.sleep(0.01)
            gate.release()

        tasks = [asyncio.create_task(waiter(i)) for i in range(3)]
        await asyncio.sleep(0.02)
        assert gate.waiting == 3
        gate.release()
        await asyncio.gather(*tasks)
        assert order == [0, 1, 2]  # FIFO, nobody barged
        assert gate.busy == 0
        # a timed-out waiter holds nothing afterwards
        assert await gate.acquire(0.1)
        assert await gate.acquire(0.05) is False
        gate.release()
        assert gate.busy == 0 and gate.waiting == 0

    _run(main())


def test_gate_queue_limit_refuses_at_once():
    async def main():
        gate = WorkGate(1, max_waiting=1)
        assert await gate.acquire()
        t = asyncio.create_task(gate.acquire(1.0))
        await asyncio.sleep(0.01)
        assert await gate.acquire(1.0) is False  # queue full: refused immediately
        gate.release()
        assert await t is True
        gate.release()
        assert gate.busy == 0

    _run(main())


def test_gate_cancelled_waiter_does_not_leak_a_slot():
    async def main():
        gate = WorkGate(1)
        assert await gate.acquire()
        t = asyncio.create_task(gate.acquire(5.0))
        await asyncio.sleep(0.01)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        gate.release()
        assert gate.busy == 0
        assert await gate.acquire(0.1)
        gate.release()

    _run(main())


def test_gate_release_from_another_thread_and_loop():
    gate = WorkGate(1)
    got = threading.Event()

    async def hold_then_wait():
        assert await gate.acquire()
        # a second loop in another thread waits for the slot
        th = threading.Thread(target=lambda: asyncio.run(other()))
        th.start()
        await asyncio.sleep(0.05)
        gate.release()
        await asyncio.to_thread(th.join, 5)

    async def other():
        assert await gate.acquire(2.0)
        got.set()
        gate.release()

    _run(hold_then_wait())
    assert got.is_set() and gate.busy == 0


def test_keyed_gates_serialise_per_key_and_clean_up():
    async def main():
        gates = KeyedGates(1)
        active: dict[str, int] = {"a": 0, "max_a": 0}

        async def job(key: str) -> None:
            async with gates.hold(key, 2.0) as ok:
                assert ok
                if key == "a":
                    active["a"] += 1
                    active["max_a"] = max(active["max_a"], active["a"])
                    await asyncio.sleep(0.01)
                    active["a"] -= 1

        await asyncio.gather(*(job("a") for _ in range(4)), job("b"))
        assert active["max_a"] == 1
        assert len(gates) == 0

    _run(main())
