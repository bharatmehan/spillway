"""Waiting for capacity through the facade, on virtual time throughout."""

import asyncio

import pytest

from spillway.core.clock import FakeClock
from spillway.core.errors import AdmissionDenied, AdmissionTimeout, ConfigurationError, Shed
from spillway.core.scope import Priority
from spillway.core.spillway import Spillway
from spillway.dimensions.concurrency import Concurrency
from spillway.dimensions.rate import Rate
from spillway.stores.memory import MemoryStore


@pytest.fixture
def clock():
    return FakeClock()


def build(clock, dimensions, **arguments):
    return Spillway(
        dimensions=dimensions,
        store=MemoryStore(clock=clock),
        clock=clock,
        **arguments,
    )


async def spin(times=20):
    for _ in range(times):
        await asyncio.sleep(0)


async def until_sleeping(clock, count=1):
    for _ in range(50):
        if clock.sleeping >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError("nothing ever went to sleep")


async def test_a_waiter_is_admitted_as_soon_as_a_settlement_frees_capacity(clock):
    limiter = build(clock, [Concurrency("generations", limit=1)])
    held = await limiter.admit(timeout=30).acquire()
    waiting = asyncio.ensure_future(limiter.admit(timeout=30).acquire())
    await spin()
    assert not waiting.done()
    held.settle(input=10, output=10)
    lease = await asyncio.wait_for(waiting, 1.0)
    assert lease.state.value == "acquired"


async def test_a_waiter_is_admitted_when_a_rate_window_replenishes(clock):
    # No settlement is involved. Time alone is what frees a rate limit, and a
    # limiter that could only be woken by a settlement would hang here.
    limiter = build(clock, [Rate("rpm", limit=1)])
    await limiter.admit(timeout=120).acquire()
    waiting = asyncio.ensure_future(limiter.admit(timeout=120).acquire())
    await until_sleeping(clock)
    clock.advance(60_000)
    await asyncio.wait_for(waiting, 1.0)


async def test_a_waiter_is_admitted_when_an_abandoned_request_gives_its_room_back(clock):
    limiter = build(clock, [Concurrency("generations", limit=1)])
    held = await limiter.admit(timeout=30).acquire()
    waiting = asyncio.ensure_future(limiter.admit(timeout=30).acquire())
    await spin()
    held.abandon(reason="the call raised")
    await asyncio.wait_for(waiting, 1.0)


async def test_higher_priority_is_served_first_whatever_the_arrival_order(clock):
    limiter = build(clock, [Concurrency("generations", limit=1)])
    held = await limiter.admit(timeout=30).acquire()
    low = asyncio.ensure_future(limiter.admit(timeout=30, priority=0).acquire())
    await spin()
    high = asyncio.ensure_future(limiter.admit(timeout=30, priority=100).acquire())
    await spin()
    held.settle(input=0, output=0)
    lease = await asyncio.wait_for(high, 1.0)
    assert not low.done()
    lease.settle(input=0, output=0)
    await asyncio.wait_for(low, 1.0)


async def test_arrival_order_is_kept_within_a_band(clock):
    limiter = build(clock, [Concurrency("generations", limit=1)])
    held = await limiter.admit(timeout=30).acquire()
    served: list[str] = []

    async def wait(name):
        lease = await limiter.admit(timeout=30).acquire()
        served.append(name)
        lease.settle(input=0, output=0)

    tasks = []
    for name in ("first", "second", "third"):
        tasks.append(asyncio.ensure_future(wait(name)))
        await spin()
    held.settle(input=0, output=0)
    await asyncio.wait_for(asyncio.gather(*tasks), 1.0)
    assert served == ["first", "second", "third"]


async def test_a_timeout_gives_up_at_the_right_moment(clock):
    limiter = build(clock, [Rate("rpm", limit=1)])
    await limiter.admit(timeout=30).acquire()
    waiting = asyncio.ensure_future(limiter.admit(timeout=5).acquire())
    await until_sleeping(clock)
    clock.advance(4_999)
    await spin()
    assert not waiting.done()
    clock.advance(1)
    with pytest.raises(AdmissionTimeout) as raised:
        await asyncio.wait_for(waiting, 1.0)
    assert raised.value.binding_dimension == "rpm"
    assert raised.value.retry_after == pytest.approx(55.0)


async def test_a_deadline_behaves_exactly_like_the_equivalent_timeout(clock):
    limiter = build(clock, [Rate("rpm", limit=1)])
    await limiter.admit(timeout=30).acquire()
    waiting = asyncio.ensure_future(limiter.admit(deadline=5.0).acquire())
    await until_sleeping(clock)
    clock.advance(4_999)
    await spin()
    assert not waiting.done()
    clock.advance(1)
    with pytest.raises(AdmissionTimeout):
        await asyncio.wait_for(waiting, 1.0)


async def test_a_timeout_of_zero_reports_the_refusal_rather_than_a_timeout(clock):
    # The caller asked not to wait, so what reaches them is what actually
    # happened rather than the wait they never had.
    limiter = build(clock, [Rate("rpm", limit=1)])
    await limiter.admit(timeout=30).acquire()
    with pytest.raises(AdmissionDenied) as raised:
        await limiter.admit(timeout=0).acquire()
    assert not isinstance(raised.value, AdmissionTimeout)
    assert "No room on rpm" in str(raised.value)


async def test_a_request_that_can_never_fit_does_not_wait_for_it(clock):
    # No amount of waiting makes it fit, so it fails at once even with a
    # generous timeout, rather than blocking its band until the timeout runs.
    limiter = build(clock, [Rate("output_tpm", limit=4_000)])
    with pytest.raises(AdmissionDenied, match="can never be admitted"):
        await limiter.admit(max_tokens=5_200, timeout=30).acquire()


async def test_waiting_works_through_the_context_manager_too(clock):
    limiter = build(clock, [Concurrency("generations", limit=1)])
    held = await limiter.admit(timeout=30).acquire()

    async def wait():
        async with limiter.admit(timeout=30) as lease:
            lease.settle(input=1, output=1)
            return "served"

    waiting = asyncio.ensure_future(wait())
    await spin()
    held.settle(input=0, output=0)
    assert await asyncio.wait_for(waiting, 1.0) == "served"


async def test_the_common_case_never_touches_the_queue(clock):
    limiter = build(clock, [Rate("rpm", limit=60)])
    await limiter.admit(timeout=30).acquire()
    assert limiter._queue.depth == 0
    assert not limiter._dispatcher.running


async def test_a_caller_who_names_no_timeout_waits_for_the_limiter_default(clock):
    limiter = build(clock, [Rate("rpm", limit=1)])
    await limiter.admit().acquire()
    waiting = asyncio.ensure_future(limiter.admit().acquire())
    await until_sleeping(clock)
    assert not waiting.done()
    clock.advance(30_000)
    with pytest.raises(AdmissionTimeout):
        await asyncio.wait_for(waiting, 1.0)


async def test_the_limiter_default_can_be_shortened(clock):
    limiter = build(clock, [Rate("rpm", limit=1)], default_timeout=2.0)
    await limiter.admit().acquire()
    waiting = asyncio.ensure_future(limiter.admit().acquire())
    await until_sleeping(clock)
    clock.advance(2_000)
    with pytest.raises(AdmissionTimeout):
        await asyncio.wait_for(waiting, 1.0)


async def test_a_limiter_default_of_zero_refuses_rather_than_waits(clock):
    limiter = build(clock, [Rate("rpm", limit=1)], default_timeout=0)
    await limiter.admit().acquire()
    with pytest.raises(AdmissionDenied) as raised:
        await limiter.admit().acquire()
    assert not isinstance(raised.value, AdmissionTimeout)


async def test_a_limiter_default_of_none_waits_for_as_long_as_it_takes(clock):
    limiter = build(clock, [Rate("rpm", limit=1)], default_timeout=None)
    await limiter.admit().acquire()
    waiting = asyncio.ensure_future(limiter.admit().acquire())
    await until_sleeping(clock)
    clock.advance(300_000)
    await asyncio.wait_for(waiting, 1.0)


async def test_a_call_level_timeout_beats_the_limiter_default(clock):
    limiter = build(clock, [Rate("rpm", limit=1)], default_timeout=None)
    await limiter.admit().acquire()
    waiting = asyncio.ensure_future(limiter.admit(timeout=5).acquire())
    await until_sleeping(clock)
    clock.advance(5_000)
    with pytest.raises(AdmissionTimeout):
        await asyncio.wait_for(waiting, 1.0)


def test_a_negative_default_timeout_is_a_configuration_error(clock):
    with pytest.raises(ConfigurationError, match="cannot be negative"):
        build(clock, [], default_timeout=-1)


async def test_cancelling_a_queued_request_takes_it_out_of_the_queue(clock):
    limiter = build(clock, [Concurrency("generations", limit=1)])
    await limiter.admit().acquire()
    waiting = asyncio.ensure_future(limiter.admit().acquire())
    await spin()
    assert limiter._queue.depth == 1
    waiting.cancel()
    await spin()
    assert limiter._queue.depth == 0


async def test_cancelling_a_queued_request_still_propagates_the_cancellation(clock):
    limiter = build(clock, [Concurrency("generations", limit=1)])
    await limiter.admit().acquire()
    waiting = asyncio.ensure_future(limiter.admit().acquire())
    await spin()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting


async def test_a_cancelled_waiter_does_not_hold_its_band_against_the_others(clock):
    # It would otherwise sit at the head, selected but listened to by nobody,
    # while everything behind it waits for a lease it will never take.
    limiter = build(clock, [Concurrency("generations", limit=1)])
    held = await limiter.admit().acquire()
    leaving = asyncio.ensure_future(limiter.admit(priority=100).acquire())
    await spin()
    following = asyncio.ensure_future(limiter.admit(priority=0).acquire())
    await spin()
    leaving.cancel()
    await spin()
    held.settle(input=0, output=0)
    lease = await asyncio.wait_for(following, 1.0)
    assert lease.state.value == "acquired"


async def test_cancelling_a_queued_request_leaves_no_capacity_held(clock):
    limiter = build(clock, [Concurrency("generations", limit=1)])
    held = await limiter.admit().acquire()
    waiting = asyncio.ensure_future(limiter.admit().acquire())
    await spin()
    waiting.cancel()
    await spin()
    held.settle(input=0, output=0)
    await spin()
    assert limiter.snapshot().dimensions["generations"].used == 0.0


async def test_a_sheddable_arrival_bounces_when_its_own_band_is_full(clock):
    limiter = build(clock, [Concurrency("generations", limit=1)], queue_capacity=1)
    await limiter.admit().acquire()
    queued = asyncio.ensure_future(limiter.admit(priority=Priority.BATCH).acquire())
    await spin()
    with pytest.raises(Shed, match="could wait"):
        await limiter.admit(priority=Priority.BATCH).acquire()
    queued.cancel()


async def test_a_full_band_leaves_the_other_bands_alone(clock):
    # The failure mode this exists for: batch work filling the queue and
    # interactive requests finding no slot left.
    limiter = build(clock, [Concurrency("generations", limit=1)], queue_capacity=1)
    held = await limiter.admit().acquire()
    batch = asyncio.ensure_future(limiter.admit(priority=Priority.BATCH).acquire())
    await spin()
    interactive = asyncio.ensure_future(limiter.admit(priority=Priority.INTERACTIVE).acquire())
    await spin()
    held.settle(input=0, output=0)
    await asyncio.wait_for(interactive, 1.0)
    batch.cancel()


async def test_a_full_band_refuses_an_unsheddable_arrival(clock):
    limiter = build(clock, [Concurrency("generations", limit=1)], queue_capacity=1)
    await limiter.admit().acquire()
    queued = asyncio.ensure_future(limiter.admit().acquire())
    await spin()
    with pytest.raises(AdmissionDenied, match="queue is full"):
        await limiter.admit().acquire()
    queued.cancel()


async def test_shed_lowest_makes_room_for_a_higher_priority_arrival(clock):
    limiter = build(
        clock,
        [Concurrency("generations", limit=1)],
        queue_capacity=1,
        queue_full_policy="shed_lowest",
    )
    held = await limiter.admit().acquire()
    batch = asyncio.ensure_future(limiter.admit(priority=Priority.BATCH).acquire())
    await spin()
    normal = asyncio.ensure_future(limiter.admit(priority=Priority.NORMAL).acquire())
    await spin()
    displaced = asyncio.ensure_future(limiter.admit(priority=Priority.NORMAL).acquire())
    await spin()
    with pytest.raises(Shed, match="lowest priority"):
        await batch
    held.settle(input=0, output=0)
    lease = await asyncio.wait_for(normal, 1.0)
    lease.settle(input=0, output=0)
    await asyncio.wait_for(displaced, 1.0)


def test_an_unknown_queue_full_policy_is_a_configuration_error(clock):
    with pytest.raises(ConfigurationError, match="not a queue full policy"):
        build(clock, [], queue_full_policy="drop_oldest")


def test_a_queue_with_no_room_at_all_is_a_configuration_error(clock):
    with pytest.raises(ConfigurationError, match="at least one waiter"):
        build(clock, [], queue_capacity=0)


async def test_a_lease_reports_how_long_it_actually_waited(clock):
    # The answer to "why was this request three seconds slow" should be a value
    # already in hand rather than an investigation.
    limiter = build(clock, [Rate("rpm", limit=1)], default_timeout=120)
    await limiter.admit().acquire()
    waiting = asyncio.ensure_future(limiter.admit().acquire())
    await until_sleeping(clock)
    clock.advance(60_000)
    lease = await asyncio.wait_for(waiting, 1.0)
    assert lease.waited_ms == pytest.approx(60_000.0)
    assert lease.explain.waited_ms == pytest.approx(60_000.0)


async def test_a_request_that_did_not_wait_reports_no_wait(clock):
    limiter = build(clock, [Rate("rpm", limit=60)])
    lease = await limiter.admit().acquire()
    assert lease.waited_ms == 0.0
    assert lease.explain.queue_position is None


async def test_a_lease_reports_how_many_were_ahead_of_it(clock):
    limiter = build(clock, [Concurrency("generations", limit=1)])
    held = await limiter.admit().acquire()
    first = asyncio.ensure_future(limiter.admit().acquire())
    await spin()
    second = asyncio.ensure_future(limiter.admit().acquire())
    await spin()
    held.settle(input=0, output=0)
    lease = await asyncio.wait_for(first, 1.0)
    assert lease.explain.queue_position == 0
    lease.settle(input=0, output=0)
    behind = await asyncio.wait_for(second, 1.0)
    assert behind.explain.queue_position == 1


async def test_an_explanation_reads_as_a_sentence_about_the_wait(clock):
    limiter = build(clock, [Rate("rpm", limit=1)], default_timeout=120)
    await limiter.admit().acquire()
    waiting = asyncio.ensure_future(limiter.admit().acquire())
    await until_sleeping(clock)
    clock.advance(60_000)
    lease = await asyncio.wait_for(waiting, 1.0)
    assert "waited 60000ms" in str(lease.explain)
    assert "queued at 0" in str(lease.explain)
