"""How a lease ends: settled, defaulted, abandoned, or cancelled.

Four endings, each with a different right answer, and getting any of them wrong
either leaks capacity or charges for work that never happened.
"""

import asyncio
import logging

import pytest

from spillway.core import spillway as facade
from spillway.core.clock import FakeClock
from spillway.core.cost import Distribution, Estimate
from spillway.core.errors import AdmissionDenied, LeaseExpired
from spillway.core.lease import LeaseState
from spillway.core.spillway import Spillway
from spillway.dimensions.concurrency import Concurrency
from spillway.dimensions.rate import Rate
from spillway.stores.memory import MemoryStore


@pytest.fixture(autouse=True)
def unwarned(monkeypatch):
    monkeypatch.setattr(facade, "_warned_about_unsettled", False)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store(clock):
    return MemoryStore(clock=clock)


@pytest.fixture
def limiter(clock, store):
    return Spillway(
        dimensions=[Rate("output_tpm", limit=1_000), Concurrency("generations", limit=2)],
        store=store,
        clock=clock,
        scope="tenant:acme",
        # These are tests of the decision, not of the waiting. Anything refused
        # is refused here and now, rather than queueing for the default wait.
        default_timeout=0,
    )


def exactly(tokens):
    return Estimate(input=0, output=Distribution.point(tokens))


def used(store, key="tenant:acme:output_tpm"):
    return store.snapshot_sync([key])[key].used


async def test_the_context_hands_over_a_lease(limiter):
    async with limiter.admit(estimate=exactly(400)) as lease:
        assert lease.state is LeaseState.ACQUIRED


async def test_settling_inside_the_block_credits_the_difference_back(limiter, store):
    async with limiter.admit(estimate=exactly(800)) as lease:
        lease.settle(input=0, output=200)
    assert used(store) == pytest.approx(200.0)


async def test_leaving_without_settling_charges_the_whole_reservation(limiter, store):
    # Pessimistic and safe. The alternative, crediting back on the assumption
    # the request was cheap, would let the limit drift open silently.
    async with limiter.admit(estimate=exactly(800)):
        pass
    assert used(store) == pytest.approx(800.0)


async def test_leaving_without_settling_still_releases_the_gauge(limiter, store):
    async with limiter.admit(estimate=exactly(100)):
        pass
    assert used(store, "tenant:acme:generations") == 0.0


async def test_leaving_without_settling_says_so_once(limiter, caplog):
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            async with limiter.admit(estimate=exactly(1)):
                pass
    assert caplog.text.count("without reporting what it actually cost") == 1


async def test_the_warning_names_the_call_that_fixes_it(limiter, caplog):
    with caplog.at_level(logging.WARNING):
        async with limiter.admit(estimate=exactly(1)):
            pass
    assert "lease.settle(input=..., output=...)" in caplog.text


async def test_settling_explicitly_says_nothing(limiter, caplog):
    with caplog.at_level(logging.WARNING):
        async with limiter.admit(estimate=exactly(100)) as lease:
            lease.settle(input=0, output=50)
    assert caplog.text == ""


async def test_an_exception_returns_the_whole_reservation(limiter, store):
    # The request never ran, so nothing was consumed and charging for it would
    # make a failing dependency eat the caller's quota as well.
    with pytest.raises(RuntimeError):
        async with limiter.admit(estimate=exactly(800)):
            raise RuntimeError("the provider refused")
    assert used(store) == 0.0
    assert used(store, "tenant:acme:generations") == 0.0


async def test_an_exception_propagates_rather_than_being_swallowed(limiter):
    with pytest.raises(RuntimeError, match="the provider refused"):
        async with limiter.admit(estimate=exactly(1)):
            raise RuntimeError("the provider refused")


async def test_an_exception_after_settling_leaves_the_settlement_alone(limiter, store):
    with pytest.raises(RuntimeError):
        async with limiter.admit(estimate=exactly(800)) as lease:
            lease.settle(input=0, output=200)
            raise RuntimeError("failed after the call succeeded")
    assert used(store) == pytest.approx(200.0)


async def test_a_cancelled_task_returns_the_whole_reservation(limiter, store):
    started = asyncio.Event()

    async def held():
        async with limiter.admit(estimate=exactly(800)):
            started.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(held())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert used(store) == 0.0
    assert used(store, "tenant:acme:generations") == 0.0


async def test_a_cancelled_task_still_reports_as_cancelled(limiter):
    # Releasing the reservation must not swallow the cancellation, or a task
    # group would carry on believing the task finished normally.
    started = asyncio.Event()

    async def held():
        async with limiter.admit(estimate=exactly(1)):
            started.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(held())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_a_request_that_outran_its_expiry_does_not_lose_its_result(
    limiter, store, clock, caplog
):
    # The bookkeeping failed, not the caller's work. Throwing away a result
    # that arrived successfully over an expired reservation is the worse trade.
    with caplog.at_level(logging.WARNING):
        async with limiter.admit(estimate=exactly(100)):
            clock.advance(60_001)
            store.snapshot_sync([])
    assert "after its reservation had already expired" in caplog.text


async def test_an_explicit_settlement_after_expiry_still_raises(limiter, store, clock):
    # A direct question deserves a direct answer. Only the automatic
    # settlement stays quiet, because the caller did not ask it anything.
    lease = await limiter.admit(estimate=exactly(100)).acquire()
    clock.advance(60_001)
    store.snapshot_sync([])
    with pytest.raises(LeaseExpired):
        lease.settle(input=0, output=50)


async def test_a_refused_admission_raises_out_of_the_context(limiter):
    await limiter.admit(estimate=exactly(1_000)).acquire()
    with pytest.raises(AdmissionDenied):
        async with limiter.admit(estimate=exactly(1)):
            pass


async def test_a_refused_admission_leaves_nothing_to_clean_up(limiter, store):
    for _ in range(2):
        await limiter.admit(estimate=exactly(1)).acquire()
    held = used(store, "tenant:acme:generations")
    with pytest.raises(AdmissionDenied):
        async with limiter.admit(estimate=exactly(1)):
            pass
    assert used(store, "tenant:acme:generations") == held
