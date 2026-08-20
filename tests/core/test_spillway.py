"""The limiter facade: reserving, refusing, and explaining either."""

import pytest

from spillway.core.clock import FakeClock
from spillway.core.cost import Cost, Distribution, Estimate
from spillway.core.errors import AdmissionDenied, ConfigurationError
from spillway.core.lease import LeaseState
from spillway.core.scope import Priority, Scope
from spillway.core.spillway import RESERVATION_QUANTILE, Spillway
from spillway.dimensions.concurrency import Concurrency
from spillway.dimensions.rate import Rate
from spillway.stores.memory import MemoryStore


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
    )


def exactly(tokens):
    return Estimate(input=0, output=Distribution.point(tokens))


async def test_a_limiter_with_no_limits_admits_everything():
    # A legitimate first step: track and report, decide the numbers later.
    limiter = Spillway()
    lease = await limiter.admit().acquire()
    assert lease.state is LeaseState.ACQUIRED


async def test_admission_reserves_the_estimated_cost(limiter, store):
    await limiter.admit(estimate=exactly(400)).acquire()
    used = store.snapshot_sync(["tenant:acme:output_tpm"])["tenant:acme:output_tpm"]
    assert used.used == pytest.approx(400.0)


async def test_the_lease_carries_what_was_reserved(limiter):
    lease = await limiter.admit(estimate=exactly(400)).acquire()
    assert lease.reserved == Cost(input_tokens=0, output_tokens=400, requests=1)


async def test_the_reserved_output_is_the_quantile_of_the_prediction(limiter):
    estimate = Estimate(input=5, output=Distribution.point(300))
    lease = await limiter.admit(estimate=estimate).acquire()
    assert lease.reserved.output_tokens == Distribution.point(300).quantile(RESERVATION_QUANTILE)


async def test_an_estimate_is_derived_from_the_prompt_when_none_is_given(limiter):
    lease = await limiter.admit(prompt="a" * 36, max_tokens=250).acquire()
    assert lease.reserved.input_tokens == 10
    assert lease.reserved.output_tokens == 250


async def test_the_default_scope_is_used_when_none_is_named(limiter):
    lease = await limiter.admit(estimate=exactly(1)).acquire()
    assert lease.scope == Scope("tenant:acme")


async def test_a_named_scope_overrides_the_default(limiter, store):
    lease = await limiter.admit(scope="user:123", estimate=exactly(1)).acquire()
    assert lease.scope == Scope("user:123")


async def test_scopes_do_not_share_a_budget(limiter, store):
    # Two tenants drawing on one counter would be isolation in name only.
    await limiter.admit(scope="a", estimate=exactly(900)).acquire()
    lease = await limiter.admit(scope="b", estimate=exactly(900)).acquire()
    assert lease.state is LeaseState.ACQUIRED


async def test_a_full_dimension_refuses_the_next_request(limiter):
    await limiter.admit(estimate=exactly(1_000)).acquire()
    with pytest.raises(AdmissionDenied):
        await limiter.admit(estimate=exactly(1)).acquire()


async def test_a_refusal_names_the_dimension_not_the_store_key(limiter):
    # A caller configured a dimension called output_tpm. Telling them
    # "tenant:acme:output_tpm" ran out leaks an internal key layout at them.
    await limiter.admit(estimate=exactly(1_000)).acquire()
    with pytest.raises(AdmissionDenied) as caught:
        await limiter.admit(estimate=exactly(1)).acquire()
    assert caught.value.binding_dimension == "output_tpm"


async def test_a_rate_refusal_reports_the_wait_in_seconds(limiter):
    # Seconds because that is what a caller passes to sleep, and because the
    # provider's own retry-after header is in seconds.
    await limiter.admit(estimate=exactly(1_000)).acquire()
    with pytest.raises(AdmissionDenied) as caught:
        await limiter.admit(estimate=exactly(100)).acquire()
    assert caught.value.retry_after == pytest.approx(6.0)


async def test_a_gauge_refusal_reports_no_wait_and_says_why(limiter):
    for _ in range(2):
        await limiter.admit(estimate=exactly(1)).acquire()
    with pytest.raises(AdmissionDenied) as caught:
        await limiter.admit(estimate=exactly(1)).acquire()
    assert caught.value.binding_dimension == "generations"
    assert caught.value.retry_after is None
    assert "waiting on a timer will not help" in str(caught.value)


async def test_a_refusal_message_names_a_fix(limiter):
    await limiter.admit(estimate=exactly(1_000)).acquire()
    with pytest.raises(AdmissionDenied, match="raise the limit"):
        await limiter.admit(estimate=exactly(1)).acquire()


async def test_a_refusal_carries_the_whole_picture_not_just_the_binding_limit(limiter):
    await limiter.admit(estimate=exactly(1_000)).acquire()
    with pytest.raises(AdmissionDenied) as caught:
        await limiter.admit(estimate=exactly(1)).acquire()
    dimensions = caught.value.explanation.dimensions
    assert set(dimensions) == {"output_tpm", "generations"}
    assert dimensions["generations"].used == 1.0


async def test_a_grant_explains_itself_too(limiter):
    lease = await limiter.admit(estimate=exactly(400)).acquire()
    assert lease.explain.admitted is True
    assert set(lease.explain.dimensions) == {"output_tpm", "generations"}
    assert lease.explain.scope == "tenant:acme"


async def test_a_refused_request_consumes_nothing(limiter, store):
    # The claim on output_tpm fits and the one on generations does not, so
    # output_tpm must be left exactly where it was.
    for _ in range(2):
        await limiter.admit(estimate=exactly(1)).acquire()
    before = store.snapshot_sync(["tenant:acme:output_tpm"])["tenant:acme:output_tpm"].used
    with pytest.raises(AdmissionDenied):
        await limiter.admit(estimate=exactly(500)).acquire()
    after = store.snapshot_sync(["tenant:acme:output_tpm"])["tenant:acme:output_tpm"].used
    assert after == pytest.approx(before)


async def test_priority_is_recorded_on_the_lease(limiter):
    lease = await limiter.admit(estimate=exactly(1), priority=Priority.INTERACTIVE).acquire()
    assert lease.priority == 100


async def test_the_default_priority_is_normal(limiter):
    assert (await limiter.admit(estimate=exactly(1)).acquire()).priority == 0


async def test_the_waiting_arguments_are_accepted_and_do_nothing_yet(limiter):
    # In the signature now so that neither the calling code nor what an editor
    # shows changes on the day they start working.
    lease = await limiter.admit(
        estimate=exactly(1), timeout=5.0, deadline=None, weight=2.0
    ).acquire()
    assert lease.state is LeaseState.ACQUIRED


async def test_building_a_context_reserves_nothing(limiter, store):
    limiter.admit(estimate=exactly(900))
    assert store.snapshot_sync(["tenant:acme:output_tpm"])["tenant:acme:output_tpm"].used == 0.0


def test_a_plain_with_statement_refuses_and_names_the_alternative(limiter):
    # Quietly starting an event loop would deadlock inside a running one, and
    # outside one it would hide that the calling code is synchronous.
    with pytest.raises(RuntimeError, match="async with"), limiter.admit(estimate=exactly(1)):
        pass


def test_the_synchronous_refusal_explains_why_it_does_not_help(limiter):
    with pytest.raises(RuntimeError, match="deadlocks"):
        limiter.admit().__enter__()


def test_a_limiter_prints_what_it_enforces(limiter):
    assert repr(limiter) == ("Spillway(dimensions=[output_tpm, generations], scope='tenant:acme')")


def test_the_dimensions_are_readable(limiter):
    assert [dimension.name for dimension in limiter.dimensions] == ["output_tpm", "generations"]


def test_a_timeout_and_a_deadline_together_are_a_configuration_error():
    # They say the same thing two ways, and picking one silently would make a
    # caller believe a limit they never set.
    limiter = Spillway()
    with pytest.raises(ConfigurationError, match="not both"):
        limiter.admit(timeout=5.0, deadline=1_000.0)


def test_either_a_timeout_or_a_deadline_alone_is_accepted():
    limiter = Spillway()
    assert limiter.admit(timeout=5.0) is not None
    assert limiter.admit(deadline=1_000.0) is not None
