"""Every OpenAI fact, checked against the fixture it came from."""

import pytest

from spillway.core.cost import Cost
from spillway.estimators.base import RequestContext
from spillway.providers.base import Outcome
from spillway.providers.openai import OpenAI, parse_duration
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
    return OpenAI()


@pytest.mark.parametrize(("name", "fixture"), every("openai", "request-"))
def test_request_reading_recovers_the_model_and_the_maximum(adapter, name, fixture):
    context = adapter.request_from(fixture["endpoint"], fixture["kwargs"])
    assert context.model == fixture["expected_context"]["model"], name
    assert context.max_tokens == fixture["expected_context"]["max_tokens"], name


def test_two_endpoints_name_the_same_things_differently(adapter):
    # The reason reading a request is the adapter's job. One endpoint takes
    # messages and a completion maximum, the other an input and an output
    # maximum, and both are the same provider in the same client library.
    chat = adapter.request_from(
        "chat.completions.create",
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 7},
    )
    responses = adapter.request_from(
        "responses.create", {"model": "m", "input": "hi", "max_output_tokens": 7}
    )
    assert chat.max_tokens == responses.max_tokens == 7
    assert chat.prompt is not None
    assert responses.prompt is not None


@pytest.mark.parametrize(("name", "fixture"), every("openai", "response-"))
def test_usage_reading_matches_the_captured_response(adapter, name, fixture):
    cost = adapter.usage_from(fixture["body"])
    expected = fixture["expected_cost"]
    assert cost.input_tokens == expected["input_tokens"], name
    assert cost.output_tokens == expected["output_tokens"], name
    assert cost.extra["cached_input"] == expected["extra"]["cached_input"], name
    assert cost.extra["reasoning"] == expected["extra"]["reasoning"], name


def test_reasoning_tokens_are_recorded_and_not_added_again(adapter):
    # They are reported inside the output total rather than beside it. Adding
    # them would count the same tokens twice on every reasoning request.
    cost = adapter.usage_from(load("openai", "response-success-responses")["body"])
    assert cost.output_tokens == 87
    assert cost.extra["reasoning"] == 64


def test_a_response_with_no_usage_says_what_to_do_instead(adapter):
    with pytest.raises(ValueError, match="settle the lease by hand"):
        adapter.usage_from({"id": "chatcmpl-1"})


@pytest.mark.parametrize(("name", "fixture"), every("openai", "error-"))
def test_classification_matches_the_captured_error(adapter, name, fixture):
    raised = _RaisedError(fixture["status"], fixture["body"], fixture["headers"])
    assert adapter.classify(raised, None).value == fixture["expected_outcome"], name


@pytest.mark.parametrize(("name", "fixture"), every("openai", "error-"))
def test_retry_after_matches_the_captured_error(adapter, name, fixture):
    raised = _RaisedError(fixture["status"], fixture["body"], fixture["headers"])
    assert adapter.retry_after(raised, fixture["headers"]) == fixture["expected_retry_after"], name


def test_an_exhausted_quota_is_not_congestion(adapter):
    quota = load("openai", "error-quota")
    limit = load("openai", "error-rate-limit")
    assert adapter.classify(_RaisedError(429, quota["body"]), None) is Outcome.ERROR
    assert adapter.classify(_RaisedError(429, limit["body"]), None) is Outcome.RATE_LIMITED


def test_header_parsing_matches_the_captured_headers(adapter):
    fixture = load("openai", "headers-ratelimit")
    state = adapter.parse_headers(fixture["headers"])
    assert state is not None
    for name, value in fixture["expected_state"]["limits"].items():
        assert state.limits[name] == value
    for name, value in fixture["expected_state"]["remaining"].items():
        assert state.remaining[name] == value


def test_input_and_output_share_one_bucket(adapter):
    # The singular token limit header is the evidence. A caller against this
    # provider names one token limit, not two.
    state = adapter.parse_headers(load("openai", "headers-ratelimit")["headers"])
    assert state is not None
    assert set(state.limits) == {"rpm", "tpm"}


def test_headers_from_a_provider_that_said_nothing_are_nothing(adapter):
    assert adapter.parse_headers({}) is None


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("1s", 1.0),
        ("6m0s", 360.0),
        ("60ms", 0.06),
        ("1h2m3s", 3723.0),
        ("500ns", 5e-7),
        ("0s", 0.0),
    ],
)
def test_a_reset_duration_becomes_seconds(value, seconds):
    assert parse_duration(value) == pytest.approx(seconds)


@pytest.mark.parametrize("value", ["", "soon", "1 second", "6m0", "s", "1x", None])
def test_an_unreadable_reset_is_nothing_rather_than_zero(value):
    # Zero would read as "the limit has already reset", which is the one wrong
    # answer here: it invites an immediate retry into a limit that is still full.
    assert parse_duration(value) is None


def test_the_maximum_is_charged_at_admission(adapter):
    # The provider takes the maximum whether or not it is reached, so the
    # reservation has to take it too. Reserving the predicted quantile would
    # mean believing in headroom the provider does not agree exists.
    assert adapter.charges_max_tokens() is True
    reserved = Cost(input_tokens=100, output_tokens=250)
    adjusted = adapter.adjust(reserved, RequestContext(max_tokens=4_096))
    assert adjusted.output_tokens == 4_096
    assert adjusted.input_tokens == 100


def test_adjustment_is_idempotent(adapter):
    # A reservation is built once and may be read more than once.
    context = RequestContext(max_tokens=4_096)
    once = adapter.adjust(Cost(input_tokens=100, output_tokens=250), context)
    assert adapter.adjust(once, context) == once


def test_a_call_naming_no_maximum_is_left_alone(adapter):
    reserved = Cost(input_tokens=100, output_tokens=250)
    assert adapter.adjust(reserved, RequestContext()) == reserved
