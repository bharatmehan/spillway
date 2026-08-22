"""How OpenAI counts, and nothing about how much you are allowed.

Almost every rule here is the opposite of the other provider's, which is the
best argument that an adapter layer is worth having.

Input and output share one token bucket rather than sitting on two, which the
singular limit header confirms, so a caller names one token limit rather than
two.

Two endpoints in the same client library name the same three things
differently. One takes messages and a completion maximum, the other takes an
input and an output maximum, and their usage objects name the same two counts
prompt and completion in one case and input and output in the other. No caller
should be expected to know that, which is why reading the request and reading
the usage are both the adapter's job.

Reset values are durations rather than timestamps, the opposite convention to
the other provider.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from spillway.core.cost import Cost
from spillway.estimators.base import RequestContext
from spillway.providers._read import count, field, path, usage_of
from spillway.providers.base import Outcome, ProviderState

_NO_QUOTA = "insufficient_quota"

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|us|ns|[smh])")
_SECONDS_PER = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}


def read_request(endpoint: str, kwargs: Mapping[str, object]) -> RequestContext:
    """Read one intercepted call in either of this schema's two shapes.

    Shared with the generic compatible adapter, which speaks the same schema
    against a different service.

    Example:
        >>> read_request("responses.create", {"model": "m", "max_output_tokens": 8}).max_tokens
        8
        >>> read_request("chat.completions.create", {"max_completion_tokens": 4}).max_tokens
        4

        The older name is still accepted on the chat endpoint, and the newer
        one wins when a caller sends both.

        >>> read_request("chat.completions.create", {"max_tokens": 4}).max_tokens
        4
    """
    model = kwargs.get("model")
    maximum = kwargs.get("max_output_tokens")
    if maximum is None:
        maximum = kwargs.get("max_completion_tokens")
    if maximum is None:
        maximum = kwargs.get("max_tokens")
    prompt = kwargs.get("input")
    if prompt is None:
        prompt = kwargs.get("messages")
    del endpoint
    return RequestContext(
        prompt=prompt if isinstance(prompt, (str, list, tuple)) else None,
        max_tokens=count(maximum) if maximum is not None else None,
        model=str(model) if model is not None else None,
    )


def read_usage(response: object) -> Cost:
    """Read what a call really cost, in either of this schema's two shapes.

    Reasoning tokens are reported inside the output total rather than beside
    it, so they are recorded as a category and never added again. Adding them
    would count the same tokens twice on every reasoning request.

    Raises:
        ValueError: if there is no usage to read.

    Example:
        >>> cost = read_usage({"usage": {"prompt_tokens": 19, "completion_tokens": 10}})
        >>> cost.input_tokens, cost.output_tokens, cost.extra["reasoning"]
        (19, 10, 0)
    """
    usage = usage_of(response)
    if usage is None:
        message = (
            "No usage found on this response, so what the call really cost is unknown. "
            "Pass the response object the client returned, or settle the lease by hand "
            "with lease.settle(input=..., output=...)."
        )
        raise ValueError(message)
    given = field(usage, "prompt_tokens")
    produced = field(usage, "completion_tokens")
    cached = path(usage, "prompt_tokens_details", "cached_tokens")
    reasoning = path(usage, "completion_tokens_details", "reasoning_tokens")
    if given is None and produced is None:
        given = field(usage, "input_tokens")
        produced = field(usage, "output_tokens")
        cached = path(usage, "input_tokens_details", "cached_tokens")
        reasoning = path(usage, "output_tokens_details", "reasoning_tokens")
    return Cost(
        input_tokens=count(given),
        output_tokens=count(produced),
        requests=1,
        extra={"cached_input": count(cached), "reasoning": count(reasoning)},
    )


def classify_failure(exc: BaseException | None) -> Outcome:
    """Say what happened, in the terms a controller acts on.

    An exhausted quota arrives with the same status and the same exception
    class as an ordinary rate limit and is told apart only by its code. Waiting
    never clears it, so it is an error rather than congestion.
    """
    if exc is None:
        return Outcome.OK
    status = field(exc, "status_code")
    if status == 429:
        code = path(field(exc, "body"), "error", "code")
        return Outcome.ERROR if code == _NO_QUOTA else Outcome.RATE_LIMITED
    if isinstance(status, int) and status >= 500:
        return Outcome.OVERLOADED
    return Outcome.ERROR


def read_retry_after(exc: BaseException | None, headers: Mapping[str, str]) -> float | None:
    """How long the provider asked to be left alone for, in seconds."""
    seconds = _number(headers.get("retry-after"))
    if seconds is None and exc is not None:
        found = path(exc, "response", "headers")
        getter = field(found, "get") if found is not None else None
        if callable(getter):
            seconds = _number(getter("retry-after"))
    if seconds is None:
        return None
    return max(0.0, seconds)


class OpenAI:
    """OpenAI's accounting rules.

    Example:
        >>> adapter = OpenAI()
        >>> adapter.charges_max_tokens()
        True

        Which changes the reservation, because the provider charges the
        maximum whether or not it is reached.

        >>> from spillway.core.cost import Cost
        >>> from spillway.estimators.base import RequestContext
        >>> adapter.adjust(Cost(input_tokens=100, output_tokens=250),
        ...               RequestContext(max_tokens=4_096)).output_tokens
        4096
    """

    name = "openai"
    client_module = "openai"
    official_hosts = ("api.openai.com",)
    endpoints = (
        "chat.completions.create",
        "chat.completions.parse",
        "responses.create",
        "responses.parse",
    )

    def __repr__(self) -> str:
        """Name the provider."""
        return "OpenAI()"

    def request_from(self, endpoint: str, kwargs: Mapping[str, object]) -> RequestContext:
        """Read the model, the prompt and the requested maximum off one call."""
        return read_request(endpoint, kwargs)

    def adjust(self, cost: Cost, context: RequestContext) -> Cost:
        """Reserve the requested maximum, because the provider charges it.

        Matching the provider's own accounting matters more than being clever.
        Reserving the predicted quantile here would mean believing in headroom
        the provider does not agree exists, and the result is rate limit
        responses this library completely failed to predict, which is the worst
        outcome available to a limiter.

        The corollary is worth stating plainly rather than hiding: against this
        provider the output prediction does not help with the rate limit at
        all. What helps is the concurrency limit, and the observation that a
        requested maximum far above the real output length is throughput being
        thrown away.

        Idempotent, because a reservation is built once and may be read more
        than once.
        """
        if context.max_tokens is None:
            return cost
        return Cost(
            input_tokens=cost.input_tokens,
            output_tokens=context.max_tokens,
            requests=cost.requests,
            extra=cost.extra,
        )

    def usage_from(self, response: object) -> Cost:
        """Read what a call really cost, in either endpoint's shape."""
        return read_usage(response)

    def classify(self, exc: BaseException | None, response: object | None) -> Outcome:
        """Say what happened, in the terms a controller acts on."""
        del response
        return classify_failure(exc)

    def parse_headers(self, headers: Mapping[str, str]) -> ProviderState | None:
        """Read the provider's own account of each limit.

        Returns:
            The state, or None when none of the headers are present, so that
            "not told" stays distinguishable from "nothing left".
        """
        limits: dict[str, float] = {}
        remaining: dict[str, float] = {}
        resets: dict[str, float] = {}
        for name, family in (("rpm", "requests"), ("tpm", "tokens")):
            limit = _number(headers.get(f"x-ratelimit-limit-{family}"))
            if limit is not None:
                limits[name] = limit
            left = _number(headers.get(f"x-ratelimit-remaining-{family}"))
            if left is not None:
                remaining[name] = left
            reset = parse_duration(headers.get(f"x-ratelimit-reset-{family}"))
            if reset is not None:
                resets[name] = reset
        if not limits and not remaining:
            return None
        return ProviderState(limits=limits, remaining=remaining, reset_at=resets)

    def retry_after(self, exc: BaseException | None, headers: Mapping[str, str]) -> float | None:
        """How long the provider asked to be left alone for."""
        return read_retry_after(exc, headers)

    def charges_max_tokens(self) -> bool:
        """Yes, so the reservation matches what the provider itself takes.

        The provider's current documentation no longer states this outright, so
        the conservative branch is taken deliberately. Reserving less than the
        provider does produces rate limit responses nothing predicted, and
        reserving more only wastes headroom that settlement gives straight
        back.
        """
        return True


def parse_duration(value: str | None) -> float | None:
    """Read a reset value expressed as a duration, such as `1s` or `6m0s`.

    The opposite convention to the other provider, which sends timestamps.

    Example:
        >>> parse_duration("1s"), parse_duration("6m0s"), parse_duration("60ms")
        (1.0, 360.0, 0.06)
        >>> parse_duration("1h2m3s")
        3723.0
        >>> parse_duration("soon") is None
        True
    """
    if not value:
        return None
    text = value.strip()
    found = _DURATION.findall(text)
    if not found:
        return None
    # Rebuilt and compared so that trailing rubbish is refused rather than
    # silently contributing nothing, which would read as an immediate reset.
    if "".join(f"{amount}{unit}" for amount, unit in found) != text:
        return None
    return sum(float(amount) * _SECONDS_PER[unit] for amount, unit in found)


def _number(value: str | None) -> float | None:
    """Read a header value as a number, or nothing if it is not one."""
    if value is None:
        return None
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None
