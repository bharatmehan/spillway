"""Two lines at construction, and every call site untouched."""

import anthropic
import httpx2
import openai
import pytest

from spillway.core.clock import FakeClock
from spillway.core.errors import ConfigurationError
from spillway.core.scope import Priority
from spillway.core.spillway import Spillway
from spillway.integrations import detect
from spillway.integrations.context import scope_context
from tests.servers.mock_provider import MockAnthropic, MockOpenAI

MESSAGE = [{"role": "user", "content": "Summarise this document."}]


@pytest.fixture(autouse=True)
def _forget_warnings():
    detect._warned_about_unofficial.clear()


def _anthropic(provider, **kwargs):
    # No retries, so one call is one request. The libraries retry a refusal
    # themselves, with backoff, which is their business and not this library's:
    # admission happens once per call whatever the transport does underneath.
    return anthropic.AsyncAnthropic(
        api_key="not-a-real-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=provider.transport()),
        **kwargs,
    )


def _openai(provider, **kwargs):
    return openai.AsyncOpenAI(
        api_key="not-a-real-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=provider.transport()),
        **kwargs,
    )


async def test_the_quickstart_admits_calls_and_settles_them():
    # The entire claim of this stage, in the shape a reader would write it.
    provider = MockAnthropic(input_tokens=2_095, output_tokens=503)
    client = Spillway.instrument(
        _anthropic(provider),
        rpm=1_000,
        input_tpm=2_000_000,
        output_tpm=400_000,
        clock=FakeClock(),
    )

    reply = await client.messages.create(
        model="claude-sonnet-5", max_tokens=4_096, messages=MESSAGE
    )

    assert reply.usage.output_tokens == 503
    found = Spillway.of(client).snapshot().dimensions
    assert found["output_tpm"].used == pytest.approx(503)
    assert found["rpm"].used == pytest.approx(1)


async def test_the_unused_reservation_is_credited_back():
    # Four thousand were held for a call that produced twenty five, and the
    # difference is free again within the request's own lifetime.
    provider = MockAnthropic(output_tokens=25)
    client = Spillway.instrument(_anthropic(provider), output_tpm=400_000, clock=FakeClock())
    await client.messages.create(model="claude-sonnet-5", max_tokens=4_096, messages=MESSAGE)
    used = Spillway.of(client).snapshot().dimensions["output_tpm"].used
    assert used == pytest.approx(25)


async def test_the_other_provider_works_the_same_way():
    provider = MockOpenAI(input_tokens=19, output_tokens=10)
    client = Spillway.instrument(_openai(provider), tpm=150_000, clock=FakeClock())
    await client.chat.completions.create(model="gpt-5", messages=MESSAGE, max_tokens=64)
    used = Spillway.of(client).snapshot().dimensions["tpm"].used
    assert used == pytest.approx(29)


async def test_a_second_endpoint_on_the_same_client_is_instrumented_too():
    provider = MockOpenAI(input_tokens=36, output_tokens=87)
    client = Spillway.instrument(_openai(provider), tpm=150_000, clock=FakeClock())
    await client.responses.create(model="gpt-5", input="hi", max_output_tokens=2_048)
    assert Spillway.of(client).snapshot().dimensions["tpm"].used == pytest.approx(123)


async def test_the_call_reaches_the_provider_unchanged():
    provider = MockAnthropic()
    client = Spillway.instrument(_anthropic(provider), rpm=1_000)
    await client.messages.create(model="claude-sonnet-5", max_tokens=64, messages=MESSAGE)
    assert provider.seen[0]["model"] == "claude-sonnet-5"
    assert provider.seen[0]["max_tokens"] == 64
    assert provider.seen[0]["messages"] == MESSAGE


async def test_the_reserved_keywords_never_reach_the_provider():
    # Both libraries reject an unknown keyword before sending, so one that
    # leaked would turn every instrumented call into a client error.
    provider = MockAnthropic()
    client = Spillway.instrument(_anthropic(provider), rpm=1_000)
    await client.messages.create(
        model="claude-sonnet-5",
        max_tokens=64,
        messages=MESSAGE,
        spillway_scope="tenant:acme",
        spillway_priority=Priority.BATCH,
        spillway_tags={"task": "summarise"},
    )
    sent = provider.seen[0]
    assert not any(name.startswith("spillway_") for name in sent)


async def test_a_keyword_scope_charges_the_right_budget():
    provider = MockAnthropic(output_tokens=25)
    client = Spillway.instrument(_anthropic(provider), output_tpm=400_000, clock=FakeClock())
    await client.messages.create(
        model="claude-sonnet-5", max_tokens=64, messages=MESSAGE, spillway_scope="tenant:acme"
    )
    limiter = Spillway.of(client)
    assert limiter.snapshot("tenant:acme").dimensions["output_tpm"].used == pytest.approx(25)
    assert limiter.snapshot("tenant:other").dimensions["output_tpm"].used == pytest.approx(0)


async def test_the_surrounding_context_charges_the_right_budget():
    # The realistic version. A middleware sets it once and no call site knows.
    provider = MockAnthropic(output_tokens=25)
    client = Spillway.instrument(_anthropic(provider), output_tpm=400_000, clock=FakeClock())
    with scope_context("tenant:acme"):
        await client.messages.create(model="claude-sonnet-5", max_tokens=64, messages=MESSAGE)
    limiter = Spillway.of(client)
    assert limiter.snapshot("tenant:acme").dimensions["output_tpm"].used == pytest.approx(25)


async def test_exactly_one_admission_happens_per_call():
    # Two would halve the effective limit, which is the one error a limiter
    # must not make.
    provider = MockAnthropic(output_tokens=100)
    client = Spillway.instrument(_anthropic(provider), output_tpm=400_000, clock=FakeClock())
    await client.messages.create(model="claude-sonnet-5", max_tokens=64, messages=MESSAGE)
    assert provider.calls == 1
    assert Spillway.of(client).snapshot().dimensions["output_tpm"].used == pytest.approx(100)


async def test_a_refusal_from_the_provider_releases_the_reservation():
    provider = MockAnthropic(capacity=1, output_tokens=25)
    client = Spillway.instrument(_anthropic(provider), output_tpm=400_000, clock=FakeClock())
    await client.messages.create(model="claude-sonnet-5", max_tokens=4_096, messages=MESSAGE)
    with pytest.raises(anthropic.RateLimitError):
        await client.messages.create(model="claude-sonnet-5", max_tokens=4_096, messages=MESSAGE)
    # The failed call held four thousand and gave all of it back, so only the
    # successful one is still counted.
    used = Spillway.of(client).snapshot().dimensions["output_tpm"].used
    assert used == pytest.approx(25)


async def test_the_instance_spelling_shares_one_limiter_across_two_clients():
    limiter = Spillway.for_provider("anthropic", output_tpm=400_000, clock=FakeClock())
    chat = limiter.instrument(_anthropic(MockAnthropic(output_tokens=25)))
    batch = limiter.instrument(_anthropic(MockAnthropic(output_tokens=25)))
    await chat.messages.create(model="claude-sonnet-5", max_tokens=64, messages=MESSAGE)
    await batch.messages.create(model="claude-sonnet-5", max_tokens=64, messages=MESSAGE)
    assert Spillway.of(chat) is Spillway.of(batch) is limiter
    assert limiter.snapshot().dimensions["output_tpm"].used == pytest.approx(50)


async def test_both_spellings_behave_the_same():
    built = Spillway.instrument(_anthropic(MockAnthropic()), rpm=1_000, clock=FakeClock())
    limiter = Spillway.for_provider("anthropic", rpm=1_000, clock=FakeClock())
    given = limiter.instrument(_anthropic(MockAnthropic()))
    for client in (built, given):
        await client.messages.create(model="claude-sonnet-5", max_tokens=64, messages=MESSAGE)
    assert Spillway.of(built).snapshot().dimensions["rpm"].used == pytest.approx(1)
    assert Spillway.of(given).snapshot().dimensions["rpm"].used == pytest.approx(1)


async def test_naming_no_limits_admits_everything_and_says_so_once(caplog):
    import spillway.core.spillway as facade

    facade._warned_about_no_limits = False
    provider = MockAnthropic()
    with caplog.at_level("WARNING"):
        first = Spillway.instrument(_anthropic(provider))
        Spillway.instrument(_anthropic(provider))
    await first.messages.create(model="claude-sonnet-5", max_tokens=64, messages=MESSAGE)
    assert Spillway.of(first).dimensions == ()
    warnings = [r for r in caplog.records if "no limits were named" in r.getMessage()]
    assert len(warnings) == 1
    assert "snapshot()" in warnings[0].getMessage()


async def test_usage_is_recorded_even_when_nothing_is_limited():
    # The point of the observe first step. Nothing is enforced and the real
    # cost is still known, which is what makes the next decision possible.
    provider = MockAnthropic(input_tokens=2_095, output_tokens=503)
    client = Spillway.instrument(_anthropic(provider))
    reply = await client.messages.create(
        model="claude-sonnet-5", max_tokens=4_096, messages=MESSAGE
    )
    assert reply.usage.output_tokens == 503


def test_instrumenting_an_instrumented_client_is_refused():
    client = Spillway.instrument(_anthropic(MockAnthropic()), rpm=1_000)
    with pytest.raises(ConfigurationError, match="already instrumented"):
        Spillway.instrument(client, rpm=1_000)


def test_a_synchronous_client_says_why_not_yet():
    synchronous = anthropic.Anthropic(api_key="k")
    with pytest.raises(ConfigurationError, match="synchronous"):
        Spillway.instrument(synchronous, rpm=1_000)


def test_an_unrecognised_client_names_both_fixes():
    class NotAClient:
        pass

    with pytest.raises(ConfigurationError, match="provider="):
        Spillway.instrument(NotAClient(), rpm=1_000)


def test_the_limiter_is_reachable_from_the_client():
    client = Spillway.instrument(_anthropic(MockAnthropic()), rpm=1_000)
    assert isinstance(Spillway.of(client), Spillway)


def test_a_bare_client_has_no_limiter_behind_it():
    with pytest.raises(ConfigurationError, match="not instrumented"):
        Spillway.of(_anthropic(MockAnthropic()))
