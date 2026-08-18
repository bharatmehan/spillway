"""The in memory store: all or nothing reservation, settlement and release."""

import threading

import pytest

from spillway.core.clock import FakeClock
from spillway.core.errors import LeaseExpired
from spillway.stores.base import Claim, ClaimKind, Delta, Store, SyncStore
from spillway.stores.memory import MemoryStore

MINUTE_MS = 60_000.0


def rate(key="acme:input_tpm", cost=1.0, limit=10.0, window_ms=MINUTE_MS):
    return Claim(key, ClaimKind.RATE, cost=cost, limit=limit, window_ms=window_ms)


def gauge(key="acme:generations", cost=1.0, limit=2.0):
    return Claim(key, ClaimKind.GAUGE, cost=cost, limit=limit)


def reserve(store, claims, ttl_ms=MINUTE_MS):
    return store.reserve_sync(claims, ttl_ms=ttl_ms, scope="acme", priority=0)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store(clock):
    return MemoryStore(clock=clock)


def test_the_store_satisfies_both_protocols(store):
    synchronous: SyncStore = store
    asynchronous: Store = store
    assert synchronous is asynchronous


def test_a_reservation_within_every_limit_is_granted(store):
    result = reserve(store, [rate(), gauge()])
    assert result.granted
    assert result.lease_id is not None


def test_each_reservation_gets_its_own_lease(store):
    first = reserve(store, [gauge()])
    second = reserve(store, [gauge()])
    assert first.lease_id != second.lease_id


def test_a_full_gauge_refuses_and_names_itself(store):
    reserve(store, [gauge(limit=1.0)])
    result = reserve(store, [gauge(limit=1.0)])
    assert not result.granted
    assert result.binding_key == "acme:generations"


def test_a_full_gauge_reports_no_wait_because_time_will_not_free_it(store):
    # A gauge is freed by something settling, not by the clock advancing.
    # Reporting a wait here would send a caller to sleep for no reason.
    reserve(store, [gauge(limit=1.0)])
    assert reserve(store, [gauge(limit=1.0)]).retry_after_ms is None


def test_a_full_rate_key_reports_how_long_until_it_would_fit(store):
    reserve(store, [rate(cost=10.0, limit=10.0)])
    result = reserve(store, [rate(cost=1.0, limit=10.0)])
    assert not result.granted
    assert result.retry_after_ms == pytest.approx(MINUTE_MS / 10)


def test_waiting_out_the_reported_time_makes_room(store, clock):
    reserve(store, [rate(cost=10.0, limit=10.0)])
    refusal = reserve(store, [rate(cost=1.0, limit=10.0)])
    clock.advance(refusal.retry_after_ms)
    assert reserve(store, [rate(cost=1.0, limit=10.0)]).granted


def test_a_refused_batch_consumes_nothing_from_the_claims_that_fitted(store):
    # The failure the whole store abstraction exists to prevent. The rate claim
    # fits and the gauge does not, so the rate claim must be left untouched.
    reserve(store, [gauge(limit=1.0)])
    before = store.snapshot_sync(["acme:input_tpm"])["acme:input_tpm"]
    result = reserve(store, [rate(), gauge(limit=1.0)])
    assert not result.granted
    assert store.snapshot_sync(["acme:input_tpm"])["acme:input_tpm"].used == before.used


def test_a_refusal_is_reported_against_the_claim_that_bound_not_the_first_one(store):
    reserve(store, [gauge(limit=1.0)])
    assert reserve(store, [rate(), gauge(limit=1.0)]).binding_key == "acme:generations"


def test_two_claims_on_one_key_in_one_batch_accumulate(store):
    # Otherwise a request claiming twice against the same key would be measured
    # against the larger of the two rather than the sum.
    result = reserve(store, [gauge(cost=1.0, limit=2.0), gauge(cost=1.0, limit=2.0)])
    assert result.granted
    assert result.utilisation["acme:generations"].used == 2.0


def test_two_claims_on_one_key_can_refuse_each_other(store):
    result = reserve(store, [gauge(cost=1.0, limit=1.0), gauge(cost=1.0, limit=1.0)])
    assert not result.granted


def test_utilisation_is_reported_for_every_key_on_success(store):
    result = reserve(store, [rate(), gauge()])
    assert set(result.utilisation) == {"acme:input_tpm", "acme:generations"}


def test_utilisation_is_reported_for_every_key_on_refusal(store):
    # Explaining a refusal must not cost a second round trip, because a store
    # that is not in this process would charge for it.
    reserve(store, [gauge(limit=1.0)])
    result = reserve(store, [rate(), gauge(limit=1.0)])
    assert set(result.utilisation) == {"acme:input_tpm", "acme:generations"}


def test_utilisation_on_refusal_reflects_state_not_the_abandoned_attempt(store):
    reserve(store, [gauge(limit=1.0)])
    result = reserve(store, [rate(cost=5.0), gauge(limit=1.0)])
    assert result.utilisation["acme:input_tpm"].used == 0.0


def test_gauge_utilisation_counts_what_is_held(store):
    reserve(store, [gauge(cost=1.0, limit=2.0)])
    result = reserve(store, [gauge(cost=1.0, limit=2.0)])
    assert result.utilisation["acme:generations"].used == 2.0


def test_rate_utilisation_counts_units_consumed_in_the_window(store):
    result = reserve(store, [rate(cost=4.0, limit=10.0)])
    assert result.utilisation["acme:input_tpm"].used == pytest.approx(4.0)


def test_rate_utilisation_falls_as_the_window_passes(store, clock):
    reserve(store, [rate(cost=10.0, limit=10.0)])
    clock.advance(MINUTE_MS / 2)
    found = store.snapshot_sync(["acme:input_tpm"])["acme:input_tpm"]
    assert found.used == pytest.approx(5.0)


def test_releasing_gives_a_gauge_back(store):
    first = reserve(store, [gauge(limit=1.0)])
    store.release_sync(first.lease_id)
    assert reserve(store, [gauge(limit=1.0)]).granted


def test_releasing_returns_a_rate_key_to_exactly_where_it_was(store):
    # Reserve then release must leave no trace, or a caller whose request
    # failed would still be paying for it.
    before = store.snapshot_sync(["acme:input_tpm"])
    result = reserve(store, [rate(cost=5.0)])
    store.release_sync(result.lease_id)
    assert store.snapshot_sync(["acme:input_tpm"])["acme:input_tpm"].used == pytest.approx(
        before["acme:input_tpm"].used
    )


def test_releasing_an_unknown_lease_does_nothing(store):
    # Release runs on the failure path, often from a finally block. Raising a
    # second error while handling the first one buries the first one.
    store.release_sync("lease-does-not-exist")


def test_releasing_twice_does_not_give_the_capacity_back_twice(store):
    result = reserve(store, [gauge(cost=1.0, limit=2.0)])
    store.release_sync(result.lease_id)
    store.release_sync(result.lease_id)
    assert store.snapshot_sync(["acme:generations"])["acme:generations"].used == 0.0


def test_settling_below_the_reservation_credits_the_difference_back(store):
    result = reserve(store, [rate(cost=8.0, limit=10.0)])
    store.settle_sync(result.lease_id, [Delta("acme:input_tpm", ClaimKind.RATE, amount=6.0)])
    assert store.snapshot_sync(["acme:input_tpm"])["acme:input_tpm"].used == pytest.approx(2.0)


def test_settling_above_the_reservation_records_the_overrun_as_debt(store):
    result = reserve(store, [rate(cost=2.0, limit=10.0)])
    store.settle_sync(result.lease_id, [Delta("acme:input_tpm", ClaimKind.RATE, amount=-3.0)])
    assert store.snapshot_sync(["acme:input_tpm"])["acme:input_tpm"].used == pytest.approx(5.0)


def test_settling_releases_a_gauge(store):
    result = reserve(store, [gauge(limit=1.0)])
    store.settle_sync(result.lease_id, [Delta("acme:generations", ClaimKind.GAUGE, amount=1.0)])
    assert reserve(store, [gauge(limit=1.0)]).granted


def test_settling_ignores_a_delta_for_a_key_the_lease_never_claimed(store):
    result = reserve(store, [gauge(limit=1.0)])
    store.settle_sync(result.lease_id, [Delta("acme:elsewhere", ClaimKind.GAUGE, amount=1.0)])
    assert store.snapshot_sync(["acme:elsewhere"])["acme:elsewhere"].used == 0.0


def test_settling_twice_raises_rather_than_counting_twice(store):
    result = reserve(store, [gauge(limit=2.0)])
    store.settle_sync(result.lease_id, [])
    with pytest.raises(LeaseExpired, match="no longer outstanding"):
        store.settle_sync(result.lease_id, [])


def test_settling_an_unknown_lease_names_the_likely_fix(store):
    with pytest.raises(LeaseExpired, match="raise the expiry"):
        store.settle_sync("lease-does-not-exist", [])


def test_a_snapshot_of_an_unused_key_reports_it_as_empty(store):
    # A dimension should chart from the moment it is configured, not from the
    # moment it is first used.
    assert store.snapshot_sync(["never:touched"])["never:touched"].used == 0.0


def test_a_snapshot_reserves_nothing(store):
    store.snapshot_sync(["acme:generations"])
    assert reserve(store, [gauge(limit=1.0)]).granted


async def test_the_asynchronous_methods_do_the_same_work(store):
    result = await store.reserve([gauge(limit=1.0)], ttl_ms=MINUTE_MS, scope="acme", priority=0)
    assert result.granted
    assert (await store.snapshot(["acme:generations"]))["acme:generations"].used == 1.0
    await store.settle(result.lease_id, [Delta("acme:generations", ClaimKind.GAUGE, amount=1.0)])
    assert (await store.snapshot(["acme:generations"]))["acme:generations"].used == 0.0


async def test_the_asynchronous_release_gives_capacity_back(store):
    result = await store.reserve([gauge(limit=1.0)], ttl_ms=MINUTE_MS, scope="acme", priority=0)
    await store.release(result.lease_id)
    again = await store.reserve([gauge(limit=1.0)], ttl_ms=MINUTE_MS, scope="acme", priority=0)
    assert again.granted


def test_concurrent_reservations_from_many_threads_never_exceed_the_limit(store):
    # The lock is the only thing standing between a check and an act here, and
    # a gauge that overshoots is exactly the failure this library exists to stop.
    granted = []

    def take():
        result = reserve(store, [gauge(cost=1.0, limit=8.0)])
        if result.granted:
            granted.append(result.lease_id)

    threads = [threading.Thread(target=take) for _ in range(64)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(granted) == 8
    assert store.snapshot_sync(["acme:generations"])["acme:generations"].used == 8.0


def test_a_key_stays_reportable_after_every_lease_against_it_has_finished(store):
    # Utilisation has to outlive the leases that produced it, or a dimension
    # would appear to reset to nothing the moment the last request settled.
    result = reserve(store, [rate(cost=8.0, limit=10.0)])
    store.settle_sync(result.lease_id, [Delta("acme:input_tpm", ClaimKind.RATE, amount=6.0)])
    found = store.snapshot_sync(["acme:input_tpm"])["acme:input_tpm"]
    assert found.used == pytest.approx(2.0)
    assert found.limit == 10.0


def test_a_refused_key_is_still_reportable(store):
    reserve(store, [gauge(limit=1.0)])
    reserve(store, [gauge(limit=1.0)])
    assert store.snapshot_sync(["acme:generations"])["acme:generations"].limit == 1.0


def test_a_lease_that_outlives_its_expiry_gives_its_gauge_back(store, clock):
    # A process that dies mid request can never settle. Without reclamation
    # every gauge leaks monotonically until the limiter admits nothing at all,
    # and the symptom gives no hint of the cause.
    reserve(store, [gauge(limit=1.0)], ttl_ms=1_000.0)
    assert not reserve(store, [gauge(limit=1.0)]).granted
    clock.advance(1_001)
    assert reserve(store, [gauge(limit=1.0)]).granted


def test_expiry_does_not_credit_a_rate_charge_back(store, clock):
    # The call really went out and really consumed the quota. The only thing
    # missing is the report of how it ended, and inventing a refund would let
    # a crashing worker exceed the provider's limit indefinitely.
    interval_ms = MINUTE_MS / 10
    reserve(store, [rate(cost=5.0, limit=10.0)], ttl_ms=1_000.0)
    clock.advance(1_001)
    found = store.snapshot_sync(["acme:input_tpm"])["acme:input_tpm"]
    # Down only by what the window itself replenished in that time, which is
    # what a rate charge does anyway. No refund was granted on top of it.
    assert found.used == pytest.approx(5.0 - 1_001 / interval_ms)


def test_expiry_is_lazy_and_needs_something_to_ask(store, clock):
    # No background task, so there is nothing to leak if the process never
    # shuts down cleanly, and a store nobody uses does nothing at all.
    result = reserve(store, [gauge(limit=1.0)], ttl_ms=1_000.0)
    clock.advance(1_001)
    assert result.lease_id in store._leases
    store.snapshot_sync([])
    assert result.lease_id not in store._leases


def test_settling_a_lease_that_expired_raises(store, clock):
    result = reserve(store, [gauge(limit=1.0)], ttl_ms=1_000.0)
    clock.advance(1_001)
    store.snapshot_sync([])
    with pytest.raises(LeaseExpired, match="ran past its expiry"):
        store.settle_sync(result.lease_id, [])


def test_releasing_a_lease_that_expired_does_nothing(store, clock):
    result = reserve(store, [gauge(limit=1.0)], ttl_ms=1_000.0)
    clock.advance(1_001)
    store.snapshot_sync([])
    store.release_sync(result.lease_id)
    assert store.snapshot_sync(["acme:generations"])["acme:generations"].used == 0.0


def test_a_settled_lease_is_not_reclaimed_a_second_time_when_its_expiry_passes(store, clock):
    # The expiry entry outlives the settlement. Acting on it again would give
    # back capacity that was already given back, so the gauge would go negative
    # and admit more than its limit.
    first = reserve(store, [gauge(cost=1.0, limit=2.0)], ttl_ms=1_000.0)
    store.settle_sync(first.lease_id, [Delta("acme:generations", ClaimKind.GAUGE, amount=1.0)])
    reserve(store, [gauge(cost=1.0, limit=2.0)], ttl_ms=60_000.0)
    clock.advance(1_001)
    assert store.snapshot_sync(["acme:generations"])["acme:generations"].used == 1.0


def test_a_lease_expires_only_once_its_own_expiry_has_passed(store, clock):
    reserve(store, [gauge(limit=1.0)], ttl_ms=1_000.0)
    clock.advance(999)
    assert not reserve(store, [gauge(limit=1.0)]).granted


def test_expiries_are_reclaimed_in_order_whatever_order_they_were_taken_in(store, clock):
    reserve(store, [gauge(key="a", cost=1.0, limit=1.0)], ttl_ms=5_000.0)
    reserve(store, [gauge(key="b", cost=1.0, limit=1.0)], ttl_ms=1_000.0)
    clock.advance(1_001)
    assert reserve(store, [gauge(key="b", cost=1.0, limit=1.0)]).granted
    assert not reserve(store, [gauge(key="a", cost=1.0, limit=1.0)]).granted
