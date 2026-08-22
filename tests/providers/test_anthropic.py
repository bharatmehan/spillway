"""Every Anthropic fact, checked against the fixture it came from."""

from datetime import datetime, timezone

import pytest

from spillway.providers.anthropic import Anthropic
from spillway.providers.base import Outcome
from tests.providers._fixtures import every, load


class _RaisedError(Exception):
    """Stands in for a client library error, which is read duck typed."""

    def __init__(self, status, body=None, headers=None):
        super().__init__("provider said no")
        self.status_code = status
        self.body = body
        self.response = _Response(headers or {})


class _Response:
    def __init__(self, headers):
        self.headers = headers


@pytest.fixture
def adapter():
    # Fixed so that a reset timestamp converts to a wait somebody can assert.
    at = datetime(2026, 8, 22, 9, 30, 0, tzinfo=timezone.utc)
    return Anthropic(now=lambda: at)


@pytest.mark.parametrize(("name", "fixture"), every("anthropic", "request-"))
def test_request_reading_recovers_the_model_and_the_maximum(adapter, name, fixture):
    context = adapter.request_from(fixture["endpoint"], fixture["kwargs"])
    assert context.model == fixture["expected_context"]["model"], name
    assert context.max_tokens == fixture["expected_context"]["max_tokens"], name


def test_the_system_prompt_is_counted_as_input():
    # It is a separate parameter rather than a message, so reading the messages
    # alone would miss it, and a long system prompt is where that costs most.
    adapter = Anthropic()
    captured = load("anthropic", "request-messages-create-system")["kwargs"]
    with_system = adapter.request_from("messages.create", captured)
    bare = {"model": "m", "max_tokens": 1, "messages": []}
    without = adapter.request_from("messages.create", bare)
    assert with_system.prompt is not None
    assert without.prompt is None


@pytest.mark.parametrize(("name", "fixture"), every("anthropic", "response-"))
def test_usage_reading_matches_the_captured_response(adapter, name, fixture):
    cost = adapter.usage_from(fixture["body"])
    expected = fixture["expected_cost"]
    assert cost.input_tokens == expected["input_tokens"], name
    assert cost.output_tokens == expected["output_tokens"], name
    assert cost.extra["cached_input"] == expected["extra"]["cached_input"], name


def test_cache_writes_are_charged_and_cache_reads_are_not(adapter):
    # The single most valuable rule this adapter encodes. Folding the read back
    # into the input total would report a charge the provider never made, and
    # for a cache heavy workload that is most of the traffic.
    cost = adapter.usage_from(
        {
            "usage": {
                "input_tokens": 50,
                "output_tokens": 10,
                "cache_creation_input_tokens": 4_000,
                "cache_read_input_tokens": 200_000,
            }
        }
    )
    assert cost.input_tokens == 4_050
    assert cost.extra["cached_input"] == 200_000


def test_a_response_with_no_usage_says_what_to_do_instead(adapter):
    with pytest.raises(ValueError, match="settle the lease by hand"):
        adapter.usage_from({"id": "msg_1"})


@pytest.mark.parametrize(("name", "fixture"), every("anthropic", "error-"))
def test_classification_matches_the_captured_error(adapter, name, fixture):
    raised = _RaisedError(fixture["status"], fixture["body"], fixture["headers"])
    assert adapter.classify(raised, None).value == fixture["expected_outcome"], name


@pytest.mark.parametrize(("name", "fixture"), every("anthropic", "error-"))
def test_retry_after_matches_the_captured_error(adapter, name, fixture):
    raised = _RaisedError(fixture["status"], fixture["body"], fixture["headers"])
    found = adapter.retry_after(raised, fixture["headers"])
    assert found == fixture["expected_retry_after"], name


def test_the_spend_cap_is_not_congestion(adapter):
    # It arrives as the same status and the same error type as a rate limit and
    # is told apart only by a code. Read as congestion, a controller walks the
    # limit down against a condition that waiting never clears.
    cap = load("anthropic", "error-spend-cap")
    limit = load("anthropic", "error-rate-limit")
    assert adapter.classify(_RaisedError(429, cap["body"]), None) is Outcome.ERROR
    assert adapter.classify(_RaisedError(429, limit["body"]), None) is Outcome.RATE_LIMITED
    assert Outcome.RATE_LIMITED.is_congestion
    assert not Outcome.ERROR.is_congestion


def test_a_success_is_ok(adapter):
    assert adapter.classify(None, {"usage": {}}) is Outcome.OK


def test_header_parsing_matches_the_captured_headers(adapter):
    fixture = load("anthropic", "headers-ratelimit")
    state = adapter.parse_headers(fixture["headers"])
    assert state is not None
    for name, value in fixture["expected_state"]["limits"].items():
        assert state.limits[name] == value
    for name, value in fixture["expected_state"]["remaining"].items():
        assert state.remaining[name] == value


def test_a_reset_timestamp_becomes_a_wait(adapter):
    # The provider reports an absolute time and the controller needs a
    # duration. The clock is fixed at 09:30:00 for this adapter.
    state = adapter.parse_headers(load("anthropic", "headers-ratelimit")["headers"])
    assert state is not None
    assert state.reset_at["input_tpm"] == 60.0


def test_a_reset_already_past_is_zero_rather_than_negative(adapter):
    state = adapter.parse_headers(
        {
            "anthropic-ratelimit-requests-limit": "1000",
            "anthropic-ratelimit-requests-reset": "2026-08-22T09:00:00Z",
        }
    )
    assert state is not None
    assert state.reset_at["rpm"] == 0.0


def test_headers_from_a_provider_that_said_nothing_are_nothing(adapter):
    # None and an empty state mean different things. A controller stands down
    # on the first and would read the second as no budget left.
    assert adapter.parse_headers({}) is None
    assert adapter.parse_headers({"content-type": "application/json"}) is None


def test_the_maximum_is_not_charged_at_admission(adapter):
    from spillway.core.cost import Cost
    from spillway.estimators.base import RequestContext

    assert adapter.charges_max_tokens() is False
    reserved = Cost(input_tokens=100, output_tokens=250)
    assert adapter.adjust(reserved, RequestContext(max_tokens=4_096)) == reserved
