"""The lease: settlement, abandonment, and refusing to do either twice."""

import pytest

from spillway.core.clock import FakeClock
from spillway.core.cost import Cost
from spillway.core.errors import LeaseAlreadySettled, LeaseExpired
from spillway.core.lease import Lease, LeaseState
from spillway.core.scope import Scope
from spillway.dimensions.concurrency import Concurrency
from spillway.dimensions.rate import Rate
from spillway.observability.explain import AdmissionExplanation
from spillway.stores.memory import MemoryStore

ACME = Scope("tenant:acme")


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store(clock):
    return MemoryStore(clock=clock)


@pytest.fixture
def dimensions():
    return [Rate("output_tpm", limit=1_000), Concurrency("generations", limit=4)]


def take(store, clock, dimensions, reserved):
    claims = [d.claim(reserved, ACME) for d in dimensions]
    result = store.reserve_sync(claims, ttl_ms=60_000.0, scope=ACME.key, priority=0)
    assert result.granted
    return Lease(
        id=result.lease_id,
        scope=ACME,
        priority=0,
        reserved=reserved,
        acquired_at_ms=clock.now_ms(),
        dimensions=dimensions,
        store=store,
        explanation=AdmissionExplanation(admitted=True, scope=ACME.key, priority=0),
    )


def used(store, key="tenant:acme:output_tpm"):
    return store.snapshot_sync([key])[key].used


def test_a_new_lease_is_holding_its_reservation(store, clock, dimensions):
    lease = take(store, clock, dimensions, Cost(output_tokens=800))
    assert lease.state is LeaseState.ACQUIRED
    assert lease.reserved == Cost(output_tokens=800)
    assert lease.scope is ACME
    assert lease.acquired_at_ms == 0.0


def test_settling_credits_back_what_was_not_used(store, clock, dimensions):
    # The whole reason conservative reservation is affordable: the surplus is
    # free again within this request's own lifetime, not at the end of a window.
    lease = take(store, clock, dimensions, Cost(output_tokens=800))
    lease.settle(input=0, output=200)
    assert used(store) == pytest.approx(200.0)


def test_settling_records_an_overrun_rather_than_ignoring_it(store, clock, dimensions):
    lease = take(store, clock, dimensions, Cost(output_tokens=200))
    lease.settle(input=0, output=500)
    assert used(store) == pytest.approx(500.0)


def test_settling_releases_the_concurrency_slot(store, clock, dimensions):
    lease = take(store, clock, dimensions, Cost(output_tokens=100))
    lease.settle(input=0, output=100)
    assert used(store, "tenant:acme:generations") == 0.0


def test_settling_moves_the_lease_to_settled(store, clock, dimensions):
    lease = take(store, clock, dimensions, Cost(output_tokens=100))
    lease.settle(input=0, output=100)
    assert lease.state is LeaseState.SETTLED


def test_settling_twice_raises_rather_than_counting_twice(store, clock, dimensions):
    # A second settlement would credit back capacity that was already credited,
    # so every limit the request touched would drift open.
    lease = take(store, clock, dimensions, Cost(output_tokens=100))
    lease.settle(input=0, output=100)
    with pytest.raises(LeaseAlreadySettled, match="count the same request twice"):
        lease.settle(input=0, output=100)


def test_settling_after_abandoning_raises(store, clock, dimensions):
    lease = take(store, clock, dimensions, Cost(output_tokens=100))
    lease.abandon()
    with pytest.raises(LeaseAlreadySettled, match="abandoned"):
        lease.settle(input=0, output=100)


def test_extra_categories_are_carried_into_the_settlement(store, clock, dimensions):
    lease = take(store, clock, dimensions, Cost(output_tokens=100))
    lease.settle(input=0, output=100, cached_input=4_000)
    assert lease.state is LeaseState.SETTLED


def test_abandoning_returns_the_whole_reservation(store, clock, dimensions):
    # The request never ran, so nothing was consumed and nothing is reconciled.
    lease = take(store, clock, dimensions, Cost(output_tokens=800))
    lease.abandon()
    assert used(store) == 0.0
    assert used(store, "tenant:acme:generations") == 0.0


def test_abandoning_moves_the_lease_to_abandoned(store, clock, dimensions):
    lease = take(store, clock, dimensions, Cost(output_tokens=100))
    lease.abandon(reason="the call raised")
    assert lease.state is LeaseState.ABANDONED


def test_abandoning_twice_does_nothing_rather_than_raising(store, clock, dimensions):
    # Abandonment runs on the failure path, often from a finally block. Raising
    # a second error there buries the one that actually mattered.
    lease = take(store, clock, dimensions, Cost(output_tokens=100))
    lease.abandon()
    lease.abandon()
    assert used(store) == 0.0


def test_abandoning_after_settling_does_nothing(store, clock, dimensions):
    lease = take(store, clock, dimensions, Cost(output_tokens=800))
    lease.settle(input=0, output=200)
    lease.abandon()
    assert used(store) == pytest.approx(200.0)
    assert lease.state is LeaseState.SETTLED


def test_settling_a_lease_whose_capacity_was_reclaimed_says_so(store, clock, dimensions):
    lease = take(store, clock, dimensions, Cost(output_tokens=100))
    clock.advance(60_001)
    store.snapshot_sync([])
    with pytest.raises(LeaseExpired, match="raise the expiry"):
        lease.settle(input=0, output=100)
    assert lease.state is LeaseState.EXPIRED


def test_nothing_has_waited_yet_so_the_wait_is_zero(store, clock, dimensions):
    assert take(store, clock, dimensions, Cost(output_tokens=1)).waited_ms == 0.0


def test_the_explanation_is_reachable_from_the_lease(store, clock, dimensions):
    lease = take(store, clock, dimensions, Cost(output_tokens=1))
    assert lease.explain.admitted is True


def test_a_lease_prints_what_it_holds_and_whether_it_still_does(store, clock, dimensions):
    lease = take(store, clock, dimensions, Cost(output_tokens=1))
    assert "state='acquired'" in repr(lease)
    lease.abandon()
    assert "state='abandoned'" in repr(lease)
