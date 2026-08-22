"""The generic adapter reads the schema and assumes nothing else."""

import pytest

from spillway.core.cost import Cost
from spillway.estimators.base import RequestContext
from spillway.providers.base import Outcome
from spillway.providers.openai_compatible import OpenAICompatible
from tests.providers._fixtures import every


@pytest.fixture
def adapter():
    return OpenAICompatible()


def test_no_host_is_official(adapter):
    # Which is what sends every client speaking this schema against an
    # unrecognised host here rather than to the named adapter.
    assert adapter.official_hosts == ()


@pytest.mark.parametrize(("name", "fixture"), every("openai", "request-"))
def test_it_reads_the_schema_it_speaks(adapter, name, fixture):
    context = adapter.request_from(fixture["endpoint"], fixture["kwargs"])
    assert context.model == fixture["expected_context"]["model"], name
    assert context.max_tokens == fixture["expected_context"]["max_tokens"], name


@pytest.mark.parametrize(("name", "fixture"), every("openai", "response-"))
def test_it_reads_usage_the_same_way(adapter, name, fixture):
    cost = adapter.usage_from(fixture["body"])
    assert cost.input_tokens == fixture["expected_cost"]["input_tokens"], name
    assert cost.output_tokens == fixture["expected_cost"]["output_tokens"], name


def test_the_maximum_is_not_charged(adapter):
    # A service that charges the maximum is a service this adapter is wrong
    # for, so assuming it would penalise every engine that does not.
    assert adapter.charges_max_tokens() is False
    reserved = Cost(input_tokens=100, output_tokens=250)
    assert adapter.adjust(reserved, RequestContext(max_tokens=4_096)) == reserved


def test_it_reports_no_provider_state_even_when_headers_look_familiar(adapter):
    # Some engines behind this adapter send these headers and some do not, and
    # those that do disagree about what they mean. Reading them would give a
    # controller a number it had guessed at and let it act confidently on it.
    familiar = {
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-limit-tokens": "150000",
        "x-ratelimit-remaining-tokens": "149984",
    }
    assert adapter.parse_headers(familiar) is None


def test_a_retry_hint_is_still_read(adapter):
    # The one header that means the same thing everywhere.
    assert adapter.retry_after(None, {"retry-after": "12"}) == 12.0
    assert adapter.retry_after(None, {}) is None


def test_failures_classify_the_same_way(adapter):
    class _RaisedError(Exception):
        def __init__(self, status):
            super().__init__("no")
            self.status_code = status
            self.body = None

    assert adapter.classify(None, {}) is Outcome.OK
    assert adapter.classify(_RaisedError(429), None) is Outcome.RATE_LIMITED
    assert adapter.classify(_RaisedError(503), None) is Outcome.OVERLOADED
    assert adapter.classify(_RaisedError(400), None) is Outcome.ERROR


def test_the_metrics_address_is_accepted_and_unused():
    # The seam a future engine side mode arrives through. Present so the shape
    # is not accidentally designed out, and reading nothing today.
    adapter = OpenAICompatible(metrics_url="http://localhost:8000/metrics")
    assert adapter.metrics_url == "http://localhost:8000/metrics"
