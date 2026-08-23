"""Settling by handing over the provider's own response."""

import pytest

from spillway.core.clock import FakeClock
from spillway.core.errors import ConfigurationError, LeaseAlreadySettled
from spillway.core.spillway import Spillway
from spillway.dimensions.rate import Rate
from tests.providers._fixtures import load


def _limiter(**kwargs):
    """A limiter on a clock that does not move, so a window never drains."""
    return Spillway(clock=FakeClock(), **kwargs)


async def _settle_with(limiter, response, **admit):
    """Admit one request, settle it from `response`, and report the lease."""
    lease = await limiter.admit(**admit).acquire()
    lease.settle_from(response)
    return lease


async def test_it_reads_the_real_cost_off_a_captured_response():
    limiter = _limiter(provider="anthropic", dimensions=[Rate("output_tpm", limit=100_000)])
    captured = load("anthropic", "response-success")["body"]
    await _settle_with(limiter, captured, max_tokens=4_096)
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(503)


async def test_the_unused_reservation_goes_back():
    # The whole reason reserving conservatively is affordable. Four thousand
    # were held and five hundred were used, and the difference is available to
    # the next caller within this request's own lifetime.
    limiter = _limiter(provider="anthropic", dimensions=[Rate("output_tpm", limit=100_000)])
    captured = load("anthropic", "response-success")["body"]
    lease = await _settle_with(limiter, captured, max_tokens=4_096)
    assert lease.reserved.output_tokens == 4_096
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(503)


async def test_a_cached_read_is_not_charged_as_input():
    # The cached tokens sit in their own category, so the input limit sees only
    # what the provider actually counted against it.
    limiter = _limiter(provider="anthropic", dimensions=[Rate("input_tpm", limit=1_000_000)])
    captured = load("anthropic", "response-success-cached")["body"]
    await _settle_with(limiter, captured, prompt="x" * 400)
    assert limiter.snapshot().dimensions["input_tpm"].used == pytest.approx(50)


async def test_it_reads_the_other_provider_too():
    limiter = _limiter(provider="openai", dimensions=[Rate("tpm", limit=1_000_000)])
    captured = load("openai", "response-success-chat")["body"]
    await _settle_with(limiter, captured, max_tokens=100)
    assert limiter.snapshot().dimensions["tpm"].used == pytest.approx(29)


async def test_a_response_with_no_usage_settles_at_the_reserved_amount(caplog):
    # Safe and expensive, and it says so. The call already succeeded, and
    # throwing the caller's result away over the bookkeeping would be worse.
    limiter = _limiter(provider="anthropic", dimensions=[Rate("output_tpm", limit=100_000)])
    with caplog.at_level("WARNING"):
        lease = await _settle_with(limiter, {"id": "msg_1"}, max_tokens=250)
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(250)
    assert lease.state.value == "settled"


async def test_a_limiter_with_no_provider_says_what_to_do_instead():
    limiter = _limiter()
    lease = await limiter.admit(max_tokens=100).acquire()
    with pytest.raises(ConfigurationError, match="provider='anthropic'"):
        lease.settle_from({"usage": {"input_tokens": 1, "output_tokens": 1}})
    lease.abandon()


async def test_settling_from_a_response_twice_still_raises():
    # A second settlement counts the same request twice on every limit it
    # touched, whichever way it was spelled.
    limiter = _limiter(provider="anthropic")
    reply = {"usage": {"input_tokens": 10, "output_tokens": 20}}
    lease = await _settle_with(limiter, reply, max_tokens=100)
    with pytest.raises(LeaseAlreadySettled):
        lease.settle_from(reply)


async def test_it_accepts_a_usage_record_on_its_own():
    limiter = _limiter(provider="anthropic", dimensions=[Rate("output_tpm", limit=10_000)])
    await _settle_with(limiter, {"input_tokens": 5, "output_tokens": 7}, max_tokens=100)
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(7)


async def test_it_accepts_an_object_rather_than_a_mapping():
    # What a client library actually returns.
    class _Usage:
        input_tokens = 11
        output_tokens = 22
        cache_read_input_tokens = 500

    class _Message:
        usage = _Usage()

    limiter = _limiter(provider="anthropic", dimensions=[Rate("output_tpm", limit=10_000)])
    await _settle_with(limiter, _Message(), max_tokens=100)
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(22)
