"""What instrumenting must not disturb, which is everything else."""

import anthropic
import httpx2
import openai
import pytest

from spillway.core.clock import FakeClock
from spillway.core.spillway import Spillway
from tests.servers.mock_provider import MockAnthropic

MESSAGE = [{"role": "user", "content": "hello"}]


def _bare(provider, **kwargs):
    return anthropic.AsyncAnthropic(
        api_key="not-a-real-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=provider.transport()),
        **kwargs,
    )


def test_it_is_still_the_class_that_was_passed_in():
    # Not a proxy standing in for one. Everything below follows from this, and
    # a wrapper that turned a typed client into an untyped one would be removed
    # by the first user who noticed.
    instrumented = Spillway.instrument(_bare(MockAnthropic()), rpm=1_000)
    assert isinstance(instrumented, anthropic.AsyncAnthropic)
    assert type(instrumented) is anthropic.AsyncAnthropic


def test_the_original_client_is_left_alone():
    provider = MockAnthropic()
    original = _bare(provider)
    Spillway.instrument(original, rpm=1_000)
    from spillway.integrations.instrument import HELD_BY

    assert getattr(original, HELD_BY, None) is None


async def test_calls_through_the_original_are_not_admitted():
    # Instrumented and bare are two independent objects. Somebody holding the
    # original should see exactly what they saw before.
    provider = MockAnthropic(output_tokens=25)
    original = _bare(provider)
    instrumented = Spillway.instrument(original, output_tpm=400_000, clock=FakeClock())

    await original.messages.create(model="claude-sonnet-5", max_tokens=64, messages=MESSAGE)
    assert provider.calls == 1
    assert Spillway.of(instrumented).snapshot().dimensions["output_tpm"].used == 0.0

    await instrumented.messages.create(model="claude-sonnet-5", max_tokens=64, messages=MESSAGE)
    assert Spillway.of(instrumented).snapshot().dimensions["output_tpm"].used == pytest.approx(25)


def test_configuration_survives_the_copy():
    provider = MockAnthropic()
    original = _bare(provider, base_url="https://api.anthropic.com", timeout=17.0)
    instrumented = Spillway.instrument(original, rpm=1_000)
    assert instrumented.api_key == original.api_key
    assert str(instrumented.base_url) == str(original.base_url)
    assert instrumented.timeout == original.timeout


def test_the_connection_pool_is_shared_rather_than_duplicated():
    # Copying reuses the existing HTTP client, so instrumenting costs no second
    # pool. If that ever stopped being true, every instrumented process would
    # quietly double its sockets.
    original = _bare(MockAnthropic())
    instrumented = Spillway.instrument(original, rpm=1_000)
    assert instrumented._client is original._client


def test_a_method_that_is_not_instrumented_is_untouched():
    original = _bare(MockAnthropic())
    instrumented = Spillway.instrument(original, rpm=1_000)
    # count_tokens is deliberately not in the endpoint list: it is a request,
    # but it is not a generation, and admitting it would charge output tokens
    # against a call that produces none.
    assert instrumented.messages.count_tokens.__func__ is original.messages.count_tokens.__func__


def test_every_other_attribute_still_resolves():
    original = _bare(MockAnthropic())
    instrumented = Spillway.instrument(original, rpm=1_000)
    for name in ("models", "beta", "messages", "with_options", "copy", "auth_headers"):
        assert hasattr(instrumented, name), name


def test_the_instrumented_method_still_looks_like_the_original():
    # Editors, documentation tools and anything reading a signature go through
    # these, so losing them would make the instrumented client feel wrong even
    # though it behaves correctly.
    original = _bare(MockAnthropic())
    instrumented = Spillway.instrument(original, rpm=1_000)
    assert instrumented.messages.create.__name__ == "create"
    assert instrumented.messages.create.__doc__ == original.messages.create.__doc__


def test_instrumenting_the_other_library_leaves_it_alone_too():
    from tests.servers.mock_provider import MockOpenAI

    provider = MockOpenAI()
    original = openai.AsyncOpenAI(
        api_key="k",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=provider.transport()),
    )
    instrumented = Spillway.instrument(original, tpm=150_000)
    assert type(instrumented) is openai.AsyncOpenAI
    assert instrumented._client is original._client
    assert original.chat.completions.create.__func__ is not None
