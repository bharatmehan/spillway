"""The properties that must hold for every possible sequence of events.

Several of the mechanisms these cover look like unnecessary complication when
read in isolation: the dry run pass before applying a batch, the clamp on a
credit, the expiry heap. Each test names the invariant it defends and why, so
that someone tidying up later can see what breaks if it goes.
"""

import asyncio
import contextlib
import math
import threading
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from spillway.core.clock import FakeClock
from spillway.core.cost import Cost, Distribution, Estimate
from spillway.core.errors import LeaseExpired
from spillway.core.spillway import Spillway
from spillway.dimensions.concurrency import Concurrency
from spillway.dimensions.rate import Rate
from spillway.estimators.base import Observation, RequestContext
from spillway.estimators.quantile import DEFAULT_QUANTILE_BOUNDS, QuantileEstimator
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


ARRIVALS = st.sampled_from([100, 0, -50])
ADVANCES = st.sampled_from([1, 500, 20_000])
SCENARIOS = st.lists(
    st.one_of(
        st.tuples(st.just("arrive"), ARRIVALS),
        st.tuples(st.just("settle"), st.just(0)),
        st.tuples(st.just("advance"), ADVANCES),
    ),
    min_size=1,
    max_size=25,
)


async def _quiet(limiter, waiting):
    """Yield until nothing has moved for a while.

    One quiet pass is not enough. A settlement travels through the loop in
    several hops, so a single yield with nothing to show for it means only that
    the next hop has not run yet.
    """
    previous = None
    still = 0
    for _ in range(400):
        current = (limiter._queue.depth, sum(1 for task in waiting if task.done()))
        still = still + 1 if current == previous else 0
        if still >= 10:
            return
        previous = current
        await asyncio.sleep(0)


def _finish(lease):
    """Settle a lease, tolerating one that outlived its expiry.

    Advancing a whole window at a time takes leases past the sixty second
    expiry, and the store has already taken that capacity back. A real call
    that ran that long lands in exactly the same place.
    """
    with contextlib.suppress(LeaseExpired):
        lease.settle(input=0, output=0)


async def _drive(operations, check=None):
    """Run a generated sequence of arrivals, settlements and clock advances."""
    clock = FakeClock()
    limiter = Spillway(
        dimensions=[Concurrency("slots", limit=2), Rate("rpm", limit=2)],
        store=MemoryStore(clock=clock),
        clock=clock,
        # Nothing may time out. A waiter here is only ever ended by being
        # admitted, so a wakeup that goes missing shows up as a stuck waiter
        # rather than being papered over by the deadline.
        default_timeout=None,
    )
    waiting: list[asyncio.Task[None]] = []
    held: list[object] = []

    async def arrive(priority):
        held.append(await limiter.admit(priority=priority).acquire())

    for name, argument in operations:
        if name == "arrive":
            waiting.append(asyncio.ensure_future(arrive(argument)))
        elif name == "settle" and held:
            _finish(held.pop(0))
        elif name == "advance":
            clock.advance(argument)
        await _quiet(limiter, waiting)
        if check is not None:
            check(limiter, waiting)

    # Nothing else arrives. Give everything back and let whole windows pass,
    # which is every source of capacity there is.
    for _ in range(len(waiting) + 3):
        while held:
            _finish(held.pop())
        await _quiet(limiter, waiting)
        clock.advance(60_000)
        await _quiet(limiter, waiting)
    return waiting


# INV-7. No wakeup is ever lost.
#
# Capacity appears in two completely different ways, a release and the passage
# of time, and the dispatcher has to hear about both. Missing either produces a
# hang that appears only under load and that nobody can reproduce afterwards.
# Once nothing more arrives and everything has been given back, every waiter
# that could be admitted has to have been admitted.
@given(operations=SCENARIOS)
def test_inv_7_no_wakeup_is_ever_lost(operations):
    waiting = asyncio.run(_drive(operations))
    for task in waiting:
        assert task.done(), "a waiter was never woken"
        task.result()


# INV-8. Queue accounting always adds up.
#
# Every request is queued or finished, never both and never neither, and the
# total depth is the sum of the bands. A waiter counted in two places is a
# waiter that gets served twice; one counted nowhere is one that is never
# served at all.
@given(operations=SCENARIOS)
def test_inv_8_every_waiter_is_queued_or_finished(operations):
    def check(limiter, waiting):
        queue = limiter._queue
        assert queue.depth == sum(queue.depths().values())
        assert queue.depth == sum(1 for task in waiting if not task.done())

    asyncio.run(_drive(operations, check))


SAMPLES = st.lists(st.integers(min_value=0, max_value=100_000), min_size=1, max_size=200)
QUANTILES = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _exact_quantile(samples, q):
    """The interpolated quantile with no floating point error at all.

    Computed with fractions so that the test is checking the implementation
    rather than reproducing its arithmetic, including the arithmetic that
    needed a tolerance in the first place.
    """
    ordered = sorted(samples)
    position = Fraction(q) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return Fraction(ordered[lower])
    below, above = ordered[lower], ordered[upper]
    return below + (above - below) * (position - lower)


# INV-9. A reservation at quantile q covers at least the q share of the history.
#
# This is the promise the whole estimator rests on, and it is a promise about
# an interpolation and a rounding, both of which are easy to get wrong in the
# direction that quietly under-reserves. Reserving at the ninth decile has to
# actually cover roughly nine in ten of the observations it was computed from,
# or every claim made about the overrun rate is false and the adaptive loop is
# correcting against a number that was never true.
@given(samples=SAMPLES, q=QUANTILES)
def test_inv_9_a_quantile_covers_its_share_of_the_samples(samples, q):
    reserved = Distribution.empirical(samples).quantile(q)
    covered = sum(1 for sample in samples if sample <= reserved)
    lower_index = math.floor(q * (len(samples) - 1))
    assert covered >= lower_index + 1


@given(samples=SAMPLES, q=QUANTILES)
def test_inv_9_a_quantile_is_never_below_the_point_it_was_asked_for(samples, q):
    # The half of INV-9 that the coverage count cannot see. Rounding an
    # interpolation down still covers the sample below it, so a limiter that
    # rounded the wrong way would satisfy the count above while reserving less
    # than the quantile every time one lands between two samples, which is most
    # of the time. The slack below is the documented snap: an interpolation
    # within a hair of a whole token is that token, not the one above it.
    reserved = Distribution.empirical(samples).quantile(q)
    exact = _exact_quantile(samples, q)
    assert reserved >= exact - Fraction(1, 1_000_000)
    assert reserved < exact + 1


@given(samples=SAMPLES, q=QUANTILES)
def test_inv_9_a_quantile_never_leaves_the_range_of_the_samples(samples, q):
    # Interpolation is between two observed values, so the answer cannot be
    # outside them. A reservation above the largest thing ever seen would be
    # waste invented from nothing.
    reserved = Distribution.empirical(samples).quantile(q)
    assert min(samples) <= reserved <= max(samples)


# INV-10. The estimator's bookkeeping stays well formed, whatever it is told.
#
# The ring is bounded, so the sample count can never exceed it however much
# traffic arrives, and can never fall below zero. The overrun ratio is a share
# and lives in [0, 1]. The error ratio is never negative, and is absent rather
# than infinite when a request generated nothing at all: reserved over zero has
# no value, and letting an infinity into the average would poison every number
# a user reads afterwards, including the one the adaptive loop would have used.
@given(
    lengths=st.lists(st.integers(min_value=0, max_value=100_000), max_size=300),
    reserved=st.integers(min_value=0, max_value=100_000),
    history=st.integers(min_value=1, max_value=50),
)
def test_inv_10_the_estimator_bookkeeping_stays_well_formed(lengths, reserved, history):
    estimator = QuantileEstimator(min_samples=1, history=history, adapt_quantile=True)
    context = RequestContext(model="claude")
    for length in lengths:
        estimator.record(
            Observation(
                context=context,
                reserved=Cost(output_tokens=reserved),
                actual=Cost(output_tokens=length),
                at_ms=0.0,
            )
        )
    stats = estimator.statistics(context)
    if not lengths:
        assert stats is None
        return
    assert 0 <= stats.samples <= history
    assert stats.samples == min(len(lengths), history)
    assert stats.observations == len(lengths)
    assert 0.0 <= stats.overrun_ratio <= 1.0
    assert stats.error_ratio is None or stats.error_ratio >= 0.0
    if all(length == 0 for length in lengths):
        assert stats.error_ratio is None
    low, high = DEFAULT_QUANTILE_BOUNDS
    assert low <= stats.quantile <= high
