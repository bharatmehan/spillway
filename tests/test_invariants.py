"""The six properties that must hold for every possible sequence of events.

Several of the mechanisms these cover look like unnecessary complication when
read in isolation: the dry run pass before applying a batch, the clamp on a
credit, the expiry heap. Each test names the invariant it defends and why, so
that someone tidying up later can see what breaks if it goes.
"""

import asyncio
import threading

import pytest
from hypothesis import given
from hypothesis import strategies as st

from spillway.core.clock import FakeClock
from spillway.core.cost import Distribution, Estimate
from spillway.core.spillway import Spillway
from spillway.dimensions.rate import Rate
from spillway.stores.base import Claim, ClaimKind, Delta
from spillway.stores.memory import MemoryStore

WINDOW_MS = 60_000.0
COSTS = st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False)


def rate(key, cost, limit=1_000.0):
    return Claim(key, ClaimKind.RATE, cost=cost, limit=limit, window_ms=WINDOW_MS)


def gauge(key, cost=1.0, limit=8.0):
    return Claim(key, ClaimKind.GAUGE, cost=cost, limit=limit)


def take(store, claims, ttl_ms=WINDOW_MS):
    return store.reserve_sync(claims, ttl_ms=ttl_ms, scope="acme", priority=0)


def state(store, keys):
    found = store.snapshot_sync(list(keys))
    return {key: found[key].used for key in keys}


# INV-1. Reserve then release returns every dimension to its exact prior state.
#
# Without it, a request that fails leaves a permanent dent in the limit. Nothing
# ever surfaces it, and the limiter quietly admits less than it was configured
# to over the life of the process.
@given(prior=st.lists(COSTS, max_size=6), cost=COSTS, slots=st.floats(0.0, 4.0))
def test_inv_1_reserve_then_release_leaves_no_trace(prior, cost, slots):
    store = MemoryStore(clock=FakeClock())
    keys = ["acme:tokens", "acme:slots"]
    for amount in prior:
        take(store, [rate("acme:tokens", amount)])

    before = state(store, keys)
    result = take(store, [rate("acme:tokens", cost), gauge("acme:slots", cost=slots)])
    if not result.granted:
        return
    store.release_sync(result.lease_id)

    after = state(store, keys)
    for key in keys:
        assert after[key] == pytest.approx(before[key], abs=1e-6)


# INV-2. Reserve then settle leaves utilisation equal to the actual cost.
#
# Both halves matter. Leaving it above the actual wastes capacity that was never
# used; leaving it below lets repeated underestimation break the limit while
# every individual request looks correct.
@given(reserved=st.integers(0, 900), actual=st.integers(0, 900))
def test_inv_2_settlement_lands_on_the_actual_cost(reserved, actual):
    clock = FakeClock()
    limiter = Spillway(
        dimensions=[Rate("output_tpm", limit=1_000)],
        store=MemoryStore(clock=clock),
        clock=clock,
        scope="acme",
    )
    context = limiter.admit(estimate=Estimate(input=0, output=Distribution.point(reserved)))
    lease = _run(context.acquire())
    lease.settle(input=0, output=actual)
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(actual, abs=1e-6)


# INV-3. A denied reservation mutates nothing at all.
#
# The one that catches the failure the whole store abstraction exists for. A
# request admitted against two limits and refused by the third leaves the first
# two wrongly consumed, and the overshoot only ever appears at the provider.
@given(
    costs=st.lists(st.floats(1.0, 100.0), min_size=3, max_size=6),
    blocked=st.integers(0, 5),
)
def test_inv_3_a_denied_reservation_consumes_nothing(costs, blocked):
    store = MemoryStore(clock=FakeClock())
    blocked = blocked % len(costs)
    keys = [f"acme:k{index}" for index in range(len(costs))]

    # Fill the gauge that will refuse, so the claim before it in the batch is
    # one that fits and the batch as a whole cannot.
    take(store, [gauge(keys[blocked], cost=1.0, limit=1.0)])
    batch = [
        gauge(key, cost=1.0, limit=1.0) if index == blocked else rate(key, cost)
        for index, (key, cost) in enumerate(zip(keys, costs, strict=True))
    ]

    before = state(store, keys)
    result = take(store, batch)
    assert not result.granted
    assert result.binding_key == keys[blocked]
    after = state(store, keys)
    for key in keys:
        assert after[key] == pytest.approx(before[key], abs=1e-9)


# INV-4. Concurrent reservations never exceed the limit in aggregate.
#
# The check then act race, which is endemic because the natural way to write a
# limiter has the gap built into its shape.
@pytest.mark.parametrize(("limit", "callers"), [(1, 32), (8, 64), (25, 100)])
def test_inv_4_concurrent_reservations_never_exceed_the_limit(limit, callers):
    store = MemoryStore(clock=FakeClock())
    start = threading.Barrier(callers)
    granted: list[str] = []
    lock = threading.Lock()

    def attempt():
        start.wait()
        result = take(store, [gauge("acme:slots", cost=1.0, limit=float(limit))])
        if result.granted:
            with lock:
                granted.append(result.lease_id)

    threads = [threading.Thread(target=attempt) for _ in range(callers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(granted) == limit
    assert state(store, ["acme:slots"])["acme:slots"] == float(limit)


# INV-5. Rate accounting never admits more than the limit allows over any window.
#
# Checked over every sliding window rather than over the whole run, because a
# limiter can respect its total and still let a burst through that a provider
# refuses. The bound includes one window of burst, which is what the burst
# allowance is for and is deliberately not zero.
@given(
    events=st.lists(
        st.tuples(st.integers(0, 20_000), st.floats(0.5, 40.0)),
        min_size=1,
        max_size=40,
    )
)
def test_inv_5_no_window_ever_exceeds_the_rate(events):
    limit = 100.0
    clock = FakeClock()
    store = MemoryStore(clock=clock)
    admitted: list[tuple[float, float]] = []

    for advance_ms, cost in events:
        clock.advance(advance_ms)
        if take(store, [rate("acme:tokens", cost, limit=limit)]).granted:
            admitted.append((clock.now_ms(), cost))

    for start in range(len(admitted)):
        total = 0.0
        for end in range(start, len(admitted)):
            total += admitted[end][1]
            width_ms = admitted[end][0] - admitted[start][0]
            allowed = limit * (width_ms / WINDOW_MS) + limit
            assert total <= allowed + 1e-6


# INV-5, second half. An idle key gets one window of burst and never more.
#
# Credit back rewinds the arrival time, so without a floor at the present moment
# a key that has been quiet accumulates credit and then admits an unbounded
# burst. Every individual admission looks correct while it happens, and the
# overshoot only ever appears as refusals from the provider.
@given(
    events=st.lists(
        st.tuples(st.integers(0, 5_000), st.floats(0.5, 40.0), st.floats(0.0, 1.0)),
        min_size=1,
        max_size=25,
    )
)
def test_inv_5_an_idle_key_never_banks_more_than_one_window_of_burst(events):
    limit = 100.0
    clock = FakeClock()
    store = MemoryStore(clock=clock)

    for advance_ms, cost, used_fraction in events:
        clock.advance(advance_ms)
        result = take(store, [rate("acme:tokens", cost, limit=limit)])
        if result.granted:
            unused = cost * (1.0 - used_fraction)
            store.settle_sync(
                result.lease_id,
                [Delta("acme:tokens", ClaimKind.RATE, amount=unused)],
            )

    clock.advance(WINDOW_MS * 10)
    burst = 0.0
    while take(store, [rate("acme:tokens", 1.0, limit=limit)]).granted:
        burst += 1.0
        if burst > limit + 10:
            break
    assert burst <= limit


# INV-6. Outstanding lease holdings sum to the current gauge utilisation.
#
# The accounting identity behind every gauge. If it drifts, either capacity is
# leaking, in which case the limiter slowly stops admitting anything, or it is
# being handed back twice, in which case the limit is not a limit.
@given(actions=st.lists(st.sampled_from(["take", "settle", "release", "wait"]), max_size=40))
def test_inv_6_held_gauge_equals_the_sum_of_outstanding_leases(actions):
    clock = FakeClock()
    store = MemoryStore(clock=clock)
    live: list[str] = []

    def held_by_leases() -> float:
        total = 0.0
        for lease in store._leases.values():
            for claim in lease.claims:
                if claim.kind is ClaimKind.GAUGE:
                    total += claim.cost
        return total

    for action in actions:
        if action == "take":
            result = take(store, [gauge("acme:slots", cost=1.0, limit=8.0)], ttl_ms=5_000.0)
            if result.granted:
                live.append(result.lease_id)
        elif action == "settle" and live:
            lease_id = live.pop()
            store.settle_sync(lease_id, [Delta("acme:slots", ClaimKind.GAUGE, amount=1.0)])
        elif action == "release" and live:
            store.release_sync(live.pop())
        elif action == "wait":
            clock.advance(2_500)
            store.snapshot_sync([])
            live = [lease_id for lease_id in live if lease_id in store._leases]

        assert store._gauge.get("acme:slots", 0.0) == pytest.approx(held_by_leases(), abs=1e-9)


def _run(coroutine):
    """Drive one coroutine to completion, for a property test that is not async."""
    return asyncio.run(coroutine)
