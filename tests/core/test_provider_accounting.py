"""Whatever a provider does to a reservation, the accounting still balances.

The adjustment is the first thing in this library that changes a reservation
after an estimator produced it, and it changes it upward. Every downstream
number is a difference against that reservation, so a mistake here does not
raise: it drifts. A limit quietly becomes smaller or larger than it was
configured to be, and stays that way until somebody notices rate limit
responses that nothing predicted.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from spillway.core.clock import FakeClock
from spillway.core.cost import Distribution
from spillway.core.spillway import Spillway
from spillway.dimensions.rate import Rate
from spillway.estimators.static import StaticEstimator

PROVIDERS = ["anthropic", "openai", "openai_compatible"]

COUNTS = st.integers(min_value=0, max_value=100_000)
MAXIMA = st.integers(min_value=1, max_value=100_000)


def _limiter(provider, predicted):
    """A limiter on a stopped clock, so nothing drains under the test."""
    return Spillway(
        provider=provider,
        clock=FakeClock(),
        estimator=StaticEstimator(output=Distribution.point(predicted)),
        dimensions=[
            Rate("input_tpm", limit=10_000_000),
            Rate("output_tpm", limit=10_000_000),
            Rate("rpm", limit=1_000_000),
        ],
    )


async def _round_trip(provider, predicted, maximum, actual_in, actual_out):
    """Admit one request, settle it, and report what the limits now hold."""
    limiter = _limiter(provider, predicted)
    lease = await limiter.admit(max_tokens=maximum).acquire()
    lease.settle_from({"usage": _usage(provider, actual_in, actual_out)})
    found = limiter.snapshot().dimensions
    return lease, found


def _usage(provider, given, produced):
    """A usage record in whichever shape this provider reports."""
    if provider == "anthropic":
        return {"input_tokens": given, "output_tokens": produced}
    return {"prompt_tokens": given, "completion_tokens": produced}


@pytest.mark.parametrize("provider", PROVIDERS)
@given(predicted=COUNTS, maximum=MAXIMA, actual_out=COUNTS)
async def test_the_limit_ends_holding_exactly_what_was_used(
    provider, predicted, maximum, actual_out
):
    # The invariant the adjustment must not break. Whatever was reserved and
    # whatever the provider did to it, once the real figure is reported the
    # limit holds the real figure and nothing else.
    _, found = await _round_trip(provider, predicted, maximum, 0, actual_out)
    assert found["output_tpm"].used == pytest.approx(actual_out)


@pytest.mark.parametrize("provider", PROVIDERS)
@given(predicted=COUNTS, maximum=MAXIMA, actual_out=COUNTS)
async def test_an_overrun_is_repaid_rather_than_discarded(provider, predicted, maximum, actual_out):
    # A request that produced more than was reserved has to leave the excess
    # charged. Clamping the difference at zero would let every overrun escape,
    # which is exactly how a limiter ends up admitting more than it allows.
    lease, found = await _round_trip(provider, predicted, maximum, 0, actual_out)
    if actual_out > lease.reserved.output_tokens:
        assert found["output_tpm"].used == pytest.approx(actual_out)


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_provider_never_reserves_less_than_was_predicted(provider):
    # An adjustment exists to match a provider that charges more than was
    # predicted. One that reserved less would hand back headroom the provider
    # does not agree exists.
    limiter = _limiter(provider, predicted=250)
    lease = await limiter.admit(max_tokens=4_096).acquire()
    assert lease.reserved.output_tokens >= 250
    lease.abandon()


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_an_abandoned_request_leaves_nothing_behind(provider):
    # Whatever the adjustment took, a request that never ran gives all of it
    # back. An adjusted reservation released by a path that only knew about
    # the unadjusted one would leak the difference on every failure.
    limiter = _limiter(provider, predicted=250)
    lease = await limiter.admit(max_tokens=4_096).acquire()
    lease.abandon(reason="the call raised")
    found = limiter.snapshot().dimensions
    assert found["output_tpm"].used == pytest.approx(0)
    assert found["input_tpm"].used == pytest.approx(0)
    assert found["rpm"].used == pytest.approx(0)
