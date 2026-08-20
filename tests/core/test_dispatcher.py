"""The dispatch loop: who is served, when it wakes, and when it stops."""

import asyncio
import logging

import pytest

from spillway.core import dispatcher as dispatch
from spillway.core.clock import FakeClock
from spillway.core.cost import Cost
from spillway.core.dispatcher import Dispatcher
from spillway.core.errors import AdmissionDenied, AdmissionTimeout
from spillway.core.queue import Waiter, WaitQueue
from spillway.core.scope import Scope
from spillway.core.spillway import Spillway
from spillway.dimensions.concurrency import Concurrency
from spillway.dimensions.rate import Rate
from spillway.stores.memory import MemoryStore

ACME = Scope("tenant:acme")


@pytest.fixture
def clock():
    return FakeClock()


def build(clock, dimensions, store=None, **queue_arguments):
    store = store if store is not None else MemoryStore(clock=clock)
    limiter = Spillway(dimensions=dimensions, store=store, clock=clock, scope=ACME)
    queue = WaitQueue(**queue_arguments)
    return limiter, queue, Dispatcher(limiter=limiter, queue=queue, clock=clock)


def enqueue(limiter, queue, dispatcher, *, priority=0, deadline_ms=None, reserved=None):
    reserved = Cost() if reserved is None else reserved
    claims, names = limiter._claims_for(scope=ACME, priority=priority, reserved=reserved)
    waiter = Waiter(
        claims=claims,
        dimension_of_key=names,
        scope=ACME,
        priority=priority,
        reserved=reserved,
        deadline_ms=deadline_ms,
        queued_at_ms=limiter._clock.now_ms(),
        future=asyncio.get_running_loop().create_future(),
        refusal=AdmissionDenied("no room"),
    )
    queue.push(waiter)
    dispatcher.ensure_running()
    return waiter


async def spin(times=20):
    """Let the loop run without advancing anything."""
    for _ in range(times):
        await asyncio.sleep(0)


async def until_sleeping(clock, count=1):
    """Wait until the dispatcher has actually gone to sleep on the clock."""
    for _ in range(50):
        if clock.sleeping >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError("the dispatcher never went to sleep")


async def test_a_waiter_is_served_when_there_is_room(clock):
    limiter, queue, dispatcher = build(clock, [Rate("rpm", limit=60)])
    waiter = enqueue(limiter, queue, dispatcher)
    lease = await asyncio.wait_for(waiter.future, 1.0)
    assert lease.scope is ACME
    assert queue.depth == 0


async def test_a_waiter_is_served_when_a_rate_window_replenishes(clock):
    # No settlement is involved at all. The dispatcher sleeps for exactly as
    # long as the refusal said the charge would take to fit, and nothing else
    # happens in between.
    limiter, queue, dispatcher = build(clock, [Rate("rpm", limit=1)])
    await limiter.admit().acquire()
    waiter = enqueue(limiter, queue, dispatcher)
    await until_sleeping(clock)
    assert not waiter.future.done()
    clock.advance(60_000)
    lease = await asyncio.wait_for(waiter.future, 1.0)
    assert lease.state.value == "acquired"


async def test_a_waiter_is_served_when_capacity_is_released(clock):
    # A gauge frees when a request finishes rather than when time passes, so
    # there is no sleep to wake from and the event is the only way through.
    limiter, queue, dispatcher = build(clock, [Concurrency("generations", limit=1)])
    held = await limiter.admit().acquire()
    waiter = enqueue(limiter, queue, dispatcher)
    await spin()
    assert not waiter.future.done()
    assert clock.sleeping == 0
    held.settle(input=0, output=0)
    dispatcher.notify()
    await asyncio.wait_for(waiter.future, 1.0)


async def test_a_release_during_a_reservation_attempt_is_not_lost(clock):
    # The whole reason the event is cleared before the attempt rather than
    # after it. Cleared afterwards, this release is wiped and the waiter sleeps
    # through capacity that is sitting there free.
    class Gated:
        def __init__(self, inner, gate):
            self._inner, self._gate, self._opened = inner, gate, False

        async def reserve(self, claims, **arguments):
            if not self._opened:
                self._opened = True
                await self._gate.wait()
            return self._inner.reserve_sync(claims, **arguments)

        async def settle(self, lease_id, deltas):
            self._inner.settle_sync(lease_id, deltas)

        async def release(self, lease_id):
            self._inner.release_sync(lease_id)

        async def snapshot(self, keys):
            return self._inner.snapshot_sync(keys)

        def reserve_sync(self, claims, **arguments):
            return self._inner.reserve_sync(claims, **arguments)

        def settle_sync(self, lease_id, deltas):
            self._inner.settle_sync(lease_id, deltas)

        def release_sync(self, lease_id):
            self._inner.release_sync(lease_id)

        def snapshot_sync(self, keys):
            return self._inner.snapshot_sync(keys)

    inner = MemoryStore(clock=clock)
    gate = asyncio.Event()
    limiter, queue, dispatcher = build(
        clock, [Concurrency("generations", limit=1)], store=Gated(inner, gate)
    )
    slot = Concurrency("generations", limit=1).claim(Cost(), ACME)
    held = inner.reserve_sync([slot], ttl_ms=60_000.0, scope=ACME.key, priority=0)
    waiter = enqueue(limiter, queue, dispatcher)
    await spin()
    inner.release_sync(held.lease_id)
    dispatcher.notify()
    gate.set()
    await asyncio.wait_for(waiter.future, 1.0)


async def test_a_waiter_gives_up_at_its_deadline(clock):
    limiter, queue, dispatcher = build(clock, [Rate("rpm", limit=1)])
    await limiter.admit().acquire()
    waiter = enqueue(limiter, queue, dispatcher, deadline_ms=5_000.0)
    await until_sleeping(clock)
    clock.advance(5_000)
    with pytest.raises(AdmissionTimeout, match="gave up") as raised:
        await asyncio.wait_for(waiter.future, 1.0)
    assert raised.value.binding_dimension == "rpm"
    assert queue.depth == 0


async def test_the_dispatcher_wakes_for_a_deadline_before_a_rate_window(clock):
    # The window is a minute away and the deadline is five seconds away, so
    # sleeping on the window alone would overshoot by fifty five seconds.
    limiter, queue, dispatcher = build(clock, [Rate("rpm", limit=1)])
    await limiter.admit().acquire()
    waiter = enqueue(limiter, queue, dispatcher, deadline_ms=5_000.0)
    await until_sleeping(clock)
    clock.advance(4_999)
    await spin()
    assert not waiter.future.done()
    clock.advance(1)
    with pytest.raises(AdmissionTimeout):
        await asyncio.wait_for(waiter.future, 1.0)


async def test_a_waiter_behind_a_blocked_head_still_times_out(clock):
    # Deadlines are checked in every band on every pass. Checking only the
    # waiter being served is what makes this one wait for ever.
    limiter, queue, dispatcher = build(clock, [Concurrency("generations", limit=1)])
    await limiter.admit().acquire()
    head = enqueue(limiter, queue, dispatcher, priority=100)
    behind = enqueue(limiter, queue, dispatcher, priority=0, deadline_ms=5_000.0)
    await spin()
    clock.advance(5_000)
    dispatcher.notify()
    with pytest.raises(AdmissionTimeout):
        await asyncio.wait_for(behind.future, 1.0)
    assert not head.future.done()
    head.future.cancel()


async def test_a_higher_priority_waiter_is_served_first(clock):
    limiter, queue, dispatcher = build(clock, [Concurrency("generations", limit=1)])
    held = await limiter.admit().acquire()
    low = enqueue(limiter, queue, dispatcher, priority=0)
    high = enqueue(limiter, queue, dispatcher, priority=100)
    await spin()
    held.settle(input=0, output=0)
    dispatcher.notify()
    await asyncio.wait_for(high.future, 1.0)
    assert not low.future.done()
    low.future.cancel()


async def test_the_dispatcher_stops_once_the_queue_has_drained(clock):
    # A task nobody stops outlives the thing it was serving, and a limiter that
    # never has to wait should have no background task at all.
    limiter, queue, dispatcher = build(clock, [Rate("rpm", limit=60)])
    assert not dispatcher.running
    waiter = enqueue(limiter, queue, dispatcher)
    await asyncio.wait_for(waiter.future, 1.0)
    await spin()
    assert not dispatcher.running


async def test_a_waiter_cancelled_while_queued_is_dropped_and_the_next_is_served(clock):
    limiter, queue, dispatcher = build(clock, [Concurrency("generations", limit=1)])
    held = await limiter.admit().acquire()
    first = enqueue(limiter, queue, dispatcher, priority=100)
    second = enqueue(limiter, queue, dispatcher, priority=0)
    await spin()
    first.future.cancel()
    held.settle(input=0, output=0)
    dispatcher.notify()
    await asyncio.wait_for(second.future, 1.0)
    assert queue.depth == 0


async def test_a_lease_granted_to_a_cancelled_waiter_is_given_back(clock, caplog):
    # The only path in the stage that can leak held capacity: the reservation
    # succeeds for a caller who is no longer there to settle it.
    limiter, queue, dispatcher = build(clock, [Concurrency("generations", limit=1)])
    waiter = enqueue(limiter, queue, dispatcher)
    waiter.future.cancel()
    with caplog.at_level(logging.WARNING):
        await spin()
    assert limiter.snapshot().dimensions["generations"].used == 0.0


async def test_notifying_before_anything_waits_is_harmless(clock):
    _limiter, _queue, dispatcher = build(clock, [Rate("rpm", limit=60)])
    dispatcher.notify()
    assert not dispatcher.running


async def test_the_repr_says_whether_it_is_running(clock):
    _limiter, _queue, dispatcher = build(clock, [Rate("rpm", limit=60)])
    assert repr(dispatcher) == "Dispatcher(running=False, depth=0)"


async def test_the_reported_retry_shrinks_with_the_time_already_waited(clock):
    # The refusal was made when the window was a whole minute away. Reporting
    # that figure five seconds later would send the caller away for five
    # seconds longer than they need to be away.
    limiter, queue, dispatcher = build(clock, [Rate("rpm", limit=1)])
    await limiter.admit().acquire()
    waiter = enqueue(limiter, queue, dispatcher, deadline_ms=5_000.0)
    await until_sleeping(clock)
    clock.advance(5_000)
    with pytest.raises(AdmissionTimeout) as raised:
        await asyncio.wait_for(waiter.future, 1.0)
    assert raised.value.retry_after == pytest.approx(55.0)
    assert "55s" in str(raised.value)


async def test_a_gauge_timeout_says_that_waiting_longer_may_not_help(clock):
    limiter, queue, dispatcher = build(clock, [Concurrency("generations", limit=1)])
    await limiter.admit().acquire()
    waiter = enqueue(limiter, queue, dispatcher, deadline_ms=5_000.0)
    await spin()
    clock.advance(5_000)
    dispatcher.notify()
    with pytest.raises(AdmissionTimeout, match="rather than on a timer") as raised:
        await asyncio.wait_for(waiter.future, 1.0)
    assert raised.value.retry_after is None


async def test_the_loop_survives_an_unexpected_failure(clock, caplog, monkeypatch):
    # A dispatcher that died here would be a hang for every caller that ever
    # queues, and a hang is the failure nobody can diagnose from outside.
    monkeypatch.setattr(dispatch, "_warned_about_a_failure", False)

    class Broken(MemoryStore):
        def __init__(self, clock):
            super().__init__(clock=clock)
            self.failures = 0

        async def reserve(self, claims, **arguments):
            if self.failures == 0:
                self.failures += 1
                raise RuntimeError("the store fell over")
            return self.reserve_sync(claims, **arguments)

    limiter, queue, dispatcher = build(clock, [Rate("rpm", limit=60)], store=Broken(clock))
    with caplog.at_level(logging.ERROR):
        waiter = enqueue(limiter, queue, dispatcher)
        await until_sleeping(clock)
        clock.advance(dispatch.FAILURE_BACKOFF_MS)
        await asyncio.wait_for(waiter.future, 1.0)
    assert "unexpected failure" in caplog.text


async def test_the_failure_is_reported_once_rather_than_on_every_pass(clock, caplog, monkeypatch):
    monkeypatch.setattr(dispatch, "_warned_about_a_failure", False)

    class AlwaysBroken(MemoryStore):
        async def reserve(self, claims, **arguments):
            raise RuntimeError("the store fell over")

    limiter, queue, dispatcher = build(clock, [Rate("rpm", limit=60)], store=AlwaysBroken(clock))
    with caplog.at_level(logging.ERROR):
        waiter = enqueue(limiter, queue, dispatcher, deadline_ms=3_000.0)
        for _ in range(3):
            await until_sleeping(clock)
            clock.advance(dispatch.FAILURE_BACKOFF_MS)
            await spin()
        with pytest.raises(AdmissionTimeout):
            await asyncio.wait_for(waiter.future, 1.0)
    assert caplog.text.count("unexpected failure") == 1
