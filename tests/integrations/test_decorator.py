"""Admitting around a function, for work no client covers."""

import pytest

from spillway.core.clock import FakeClock
from spillway.core.errors import ConfigurationError
from spillway.core.scope import Priority
from spillway.core.spillway import Spillway
from spillway.dimensions.rate import Rate
from spillway.integrations.context import scope_context
from spillway.integrations.decorator import admitted


def _limiter(**kwargs):
    return Spillway(
        clock=FakeClock(),
        dimensions=[Rate("output_tpm", limit=100_000), Rate("rpm", limit=1_000)],
        **kwargs,
    )


async def test_it_admits_and_settles_from_the_return_value():
    limiter = _limiter(provider="anthropic")

    @admitted(limiter, max_tokens=4_096)
    async def summarise(document):
        return {"usage": {"input_tokens": 120, "output_tokens": 48}}

    await summarise("a document")
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(48)


async def test_a_return_value_it_cannot_read_settles_at_the_reservation(caplog):
    # Safe and expensive, and it says which. The function already succeeded.
    limiter = _limiter(provider="anthropic")

    @admitted(limiter, max_tokens=500)
    async def summarise(document):
        return "just a string"

    with caplog.at_level("WARNING"):
        assert await summarise("a document") == "just a string"
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(500)


async def test_without_a_provider_it_settles_at_the_reservation():
    limiter = _limiter()

    @admitted(limiter, max_tokens=500)
    async def summarise(document):
        return {"usage": {"input_tokens": 1, "output_tokens": 2}}

    await summarise("a document")
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(500)


async def test_the_scope_can_be_computed_from_the_arguments():
    # The common case: the tenant is already one of the function's arguments,
    # and threading it into the decorator by hand would be a second place to
    # get it wrong.
    limiter = _limiter(provider="anthropic")

    @admitted(limiter, scope=lambda tenant, document: f"tenant:{tenant}", max_tokens=100)
    async def summarise(tenant, document):
        return {"usage": {"input_tokens": 1, "output_tokens": 30}}

    await summarise("acme", "a document")
    assert limiter.snapshot("tenant:acme").dimensions["output_tpm"].used == pytest.approx(30)
    assert limiter.snapshot("tenant:other").dimensions["output_tpm"].used == 0.0


async def test_the_surrounding_context_still_applies():
    limiter = _limiter(provider="anthropic")

    @admitted(limiter, max_tokens=100)
    async def summarise(document):
        return {"usage": {"input_tokens": 1, "output_tokens": 30}}

    with scope_context("tenant:acme"):
        await summarise("a document")
    assert limiter.snapshot("tenant:acme").dimensions["output_tpm"].used == pytest.approx(30)


async def test_a_failing_function_gives_the_reservation_back():
    limiter = _limiter(provider="anthropic")

    @admitted(limiter, max_tokens=4_096)
    async def summarise(document):
        raise RuntimeError("the model call failed")

    with pytest.raises(RuntimeError):
        await summarise("a document")
    assert limiter.snapshot().dimensions["output_tpm"].used == 0.0
    assert limiter.snapshot().dimensions["rpm"].used == 0.0


async def test_the_wrapped_function_keeps_its_identity():
    limiter = _limiter()

    @admitted(limiter)
    async def summarise(document):
        """Summarise a document."""

    assert summarise.__name__ == "summarise"
    assert summarise.__doc__ == "Summarise a document."


def test_a_synchronous_function_says_why_not_yet():
    limiter = _limiter()
    with pytest.raises(ConfigurationError, match="not an async function"):

        @admitted(limiter)
        def summarise(document):
            return document


async def test_the_priority_reaches_the_admission():
    limiter = _limiter(provider="anthropic")

    @admitted(limiter, priority=Priority.BATCH, max_tokens=100)
    async def summarise(document):
        return {"usage": {"input_tokens": 1, "output_tokens": 2}}

    await summarise("a document")
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(2)
