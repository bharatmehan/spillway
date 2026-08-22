"""One contract, run identically against every adapter.

This is what makes a contributed adapter trustworthy. A reviewer reads the
fixtures rather than reasoning about somebody else's provider semantics, and a
subtly broken adapter cannot get through.

Two of these assertions catch things nothing else would. The endpoint existence
check catches a client library renaming a method, which otherwise fails
silently: the call simply stops being admitted, goes out unmetered, and every
other test in this repository still passes. The delegation check catches an
endpoint list naming both a method and the method it calls, which would admit
one call twice and halve the limit.
"""

import inspect

import pytest

from spillway.core.cost import Cost
from spillway.estimators.base import RequestContext
from spillway.providers.anthropic import Anthropic
from spillway.providers.base import Outcome, ProviderAdapter
from spillway.providers.openai import OpenAI
from spillway.providers.openai_compatible import OpenAICompatible
from tests.providers._fixtures import every

ADAPTERS = [Anthropic(), OpenAI(), OpenAICompatible()]

# Which captured fixtures describe the schema each adapter speaks. The generic
# adapter speaks the same schema as the named one it shadows.
SCHEMA_OF = {"anthropic": "anthropic", "openai": "openai", "openai_compatible": "openai"}

CLIENTS = {"anthropic": "AsyncAnthropic", "openai": "AsyncOpenAI"}


def _client_for(adapter):
    """Build a real client of the kind this adapter recognises."""
    module = pytest.importorskip(adapter.client_module)
    return getattr(module, CLIENTS[adapter.client_module])(api_key="not-a-real-key")


def _resolve(client, path):
    """Walk a dotted endpoint path down a client, or raise trying."""
    found = client
    for part in path.split("."):
        found = getattr(found, part)
    return found


def _ids(adapters):
    return [a.name for a in adapters]


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_it_is_an_adapter(adapter):
    assert isinstance(adapter, ProviderAdapter)


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_it_names_itself(adapter):
    assert adapter.name
    assert adapter.client_module
    assert adapter.endpoints


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_official_hosts_are_hosts(adapter):
    # Compared against a client's base URL host, so a scheme or a path here
    # would never match anything and the limits would silently never apply.
    for host in adapter.official_hosts:
        assert "://" not in host, host
        assert "/" not in host, host


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_every_declared_endpoint_exists_on_the_client_library(adapter):
    # The one that catches SDK churn. A renamed method stops being
    # instrumented, the call goes out unadmitted, and nothing else notices.
    client = _client_for(adapter)
    for path in adapter.endpoints:
        assert callable(_resolve(client, path)), path


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_every_declared_endpoint_takes_keyword_arguments_only(adapter):
    # The request reader looks at keyword arguments alone. If a library ever
    # accepted a positional model or maximum, every reservation built from one
    # of those calls would silently be built from nothing.
    client = _client_for(adapter)
    for path in adapter.endpoints:
        # Unwrapped, because these methods carry a decorator that hides the
        # real signature behind a catch all one.
        signature = inspect.signature(inspect.unwrap(_resolve(client, path)))
        positional = [
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        assert positional == [], f"{path} takes {positional} positionally"


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_no_declared_endpoint_calls_another_declared_endpoint(adapter):
    # Listing both a method and the method it delegates to would admit one call
    # to the network twice, which halves the effective limit in the one place a
    # limiter must never be wrong.
    client = _client_for(adapter)
    leaves = {path.rsplit(".", 1)[1] for path in adapter.endpoints}
    for path in adapter.endpoints:
        source = inspect.getsource(inspect.unwrap(_resolve(client, path)))
        body = source.split("\n", 1)[1]
        for other in leaves - {path.rsplit(".", 1)[1]}:
            assert f"self.{other}(" not in body, f"{path} delegates to {other}"


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_adjustment_is_idempotent(adapter):
    context = RequestContext(model="m", max_tokens=4_096)
    once = adapter.adjust(Cost(input_tokens=100, output_tokens=250), context)
    assert adapter.adjust(once, context) == once


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_adjustment_never_reserves_less_than_it_was_given(adapter):
    # An adjustment exists to match a provider that charges more than was
    # predicted. One that charged less would hand back headroom the provider
    # does not agree exists, which is the failure this whole layer prevents.
    reserved = Cost(input_tokens=100, output_tokens=250)
    adjusted = adapter.adjust(reserved, RequestContext(max_tokens=4_096))
    assert adjusted.input_tokens >= reserved.input_tokens
    assert adjusted.output_tokens >= reserved.output_tokens


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_usage_reads_every_captured_response(adapter):
    for name, fixture in every(SCHEMA_OF[adapter.name], "response-"):
        cost = adapter.usage_from(fixture["body"])
        assert cost.input_tokens >= 0, name
        assert cost.output_tokens >= 0, name
        assert cost.requests == 1, name


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_reading_usage_twice_gives_the_same_answer(adapter):
    for name, fixture in every(SCHEMA_OF[adapter.name], "response-"):
        assert adapter.usage_from(fixture["body"]) == adapter.usage_from(fixture["body"]), name


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_extra_categories_are_never_folded_into_the_totals(adapter):
    # Cached reads sit outside the input total and reasoning tokens sit inside
    # the output one. Either being added again counts the same tokens twice on
    # every request that has them, which quietly shrinks the real limit.
    for name, fixture in every(SCHEMA_OF[adapter.name], "response-"):
        cost = adapter.usage_from(fixture["body"])
        assert cost.total_tokens == cost.input_tokens + cost.output_tokens, name


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_a_response_with_no_usage_raises_rather_than_reporting_zero(adapter):
    # Zero would settle the lease as though the call had been free, crediting
    # back the whole reservation and under counting the limit for ever.
    with pytest.raises(ValueError):
        adapter.usage_from({"id": "no-usage-here"})


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_no_failure_reads_as_success(adapter):
    assert adapter.classify(None, {"usage": {}}) is Outcome.OK
    for status in (400, 401, 403, 404, 413, 422, 429, 500, 502, 503, 504, 529):
        raised = _raised(status)
        assert adapter.classify(raised, None) is not Outcome.OK, status


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_a_client_error_is_never_congestion(adapter):
    # Reading a malformed request as overload makes the limiter collapse for a
    # reason no amount of backing off can fix.
    for status in (400, 401, 403, 404, 413, 422):
        assert not adapter.classify(_raised(status), None).is_congestion, status


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_a_server_failure_is_congestion(adapter):
    for status in (500, 502, 503):
        assert adapter.classify(_raised(status), None).is_congestion, status


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_retry_after_is_never_negative(adapter):
    for value in ("12", "0", "-5", "", "soon", "1e3"):
        found = adapter.retry_after(None, {"retry-after": value})
        assert found is None or found >= 0.0, value


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_silence_from_a_provider_is_not_an_empty_budget(adapter):
    # None and a state full of zeroes mean opposite things to a controller.
    assert adapter.parse_headers({}) is None


@pytest.mark.parametrize("adapter", ADAPTERS, ids=_ids(ADAPTERS))
def test_reported_state_is_internally_consistent(adapter):
    for name, fixture in every(SCHEMA_OF[adapter.name], "headers-"):
        state = adapter.parse_headers(fixture["headers"])
        if state is None:
            continue
        for limit_name, limit in state.limits.items():
            assert limit > 0, f"{name}: {limit_name}"
            left = state.remaining.get(limit_name)
            if left is not None:
                assert 0 <= left <= limit, f"{name}: {limit_name}"
        for wait in state.reset_at.values():
            assert wait >= 0, name


def _raised(status):
    """A client library error, as an adapter reads one: duck typed."""

    class _RaisedError(Exception):
        pass

    error = _RaisedError("provider said no")
    error.status_code = status
    error.body = None
    return error
