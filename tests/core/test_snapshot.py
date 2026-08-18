"""The snapshot: how full everything is, without reserving anything."""

import pytest

from spillway.core.clock import FakeClock
from spillway.core.cost import Distribution, Estimate
from spillway.core.spillway import Spillway
from spillway.dimensions.concurrency import Concurrency
from spillway.dimensions.rate import Rate
from spillway.stores.memory import MemoryStore


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def limiter(clock):
    return Spillway(
        dimensions=[Rate("output_tpm", limit=1_000), Concurrency("generations", limit=2)],
        store=MemoryStore(clock=clock),
        clock=clock,
        scope="tenant:acme",
    )


def exactly(tokens):
    return Estimate(input=0, output=Distribution.point(tokens))


def test_a_fresh_limiter_reports_every_dimension_as_empty(limiter):
    found = limiter.snapshot()
    assert set(found.dimensions) == {"output_tpm", "generations"}
    assert all(used.used == 0.0 for used in found.dimensions.values())


def test_a_fresh_limiter_still_reports_the_real_limits(limiter):
    # Reporting a limit of nothing before the first request would make a
    # dashboard built on this look broken until traffic arrived.
    found = limiter.snapshot()
    assert found.dimensions["output_tpm"].limit == 1_000.0
    assert found.dimensions["generations"].limit == 2.0


def test_a_fresh_limiter_reports_full_headroom(limiter):
    assert limiter.snapshot().dimensions["output_tpm"].headroom == 1.0


async def test_a_reservation_shows_up_in_the_snapshot(limiter):
    await limiter.admit(estimate=exactly(400)).acquire()
    found = limiter.snapshot()
    assert found.dimensions["output_tpm"].used == pytest.approx(400.0)
    assert found.dimensions["generations"].used == 1.0


async def test_settling_shows_the_credited_difference(limiter):
    # The five line check: reserve, do the work, settle, and see the surplus
    # back in the next snapshot.
    lease = await limiter.admit(estimate=exactly(800)).acquire()
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(800.0)
    lease.settle(input=0, output=200)
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(200.0)


async def test_a_snapshot_is_keyed_by_dimension_name_not_by_store_key(limiter):
    await limiter.admit(estimate=exactly(1)).acquire()
    assert "tenant:acme:output_tpm" not in limiter.snapshot().dimensions


async def test_a_snapshot_reports_the_scope_it_is_about(limiter):
    assert limiter.snapshot().scope == "tenant:acme"
    assert limiter.snapshot(scope="user:123").scope == "user:123"


async def test_a_snapshot_of_another_scope_reports_that_scope_only(limiter):
    await limiter.admit(scope="a", estimate=exactly(400)).acquire()
    assert limiter.snapshot(scope="b").dimensions["output_tpm"].used == 0.0
    assert limiter.snapshot(scope="a").dimensions["output_tpm"].used == pytest.approx(400.0)


async def test_a_snapshot_reserves_nothing(limiter):
    # Safe to call on a timer from a health check, which is only true if it
    # cannot itself affect what gets admitted.
    for _ in range(10):
        limiter.snapshot()
    await limiter.admit(estimate=exactly(1_000)).acquire()
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(1_000.0)


def test_a_limiter_with_no_limits_snapshots_to_nothing():
    assert Spillway().snapshot().dimensions == {}


async def test_rate_utilisation_falls_as_the_window_passes(limiter, clock):
    await limiter.admit(estimate=exactly(1_000)).acquire()
    clock.advance(30_000)
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(500.0)
