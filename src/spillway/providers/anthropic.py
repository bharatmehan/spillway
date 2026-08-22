"""How Anthropic counts, and nothing about how much you are allowed.

Three facts drive everything here, all of them verified against the provider's
current documentation and each with a fixture behind it.

Requests, input tokens and output tokens sit on three separate buckets rather
than one combined one, so a caller names up to three limits rather than one.

Cache reads do not count toward the input limit on current models, and cache
writes do. The input token field counts only what follows the last cache
breakpoint, so what is charged is that field plus cache creation, and the
cached read is recorded as its own category rather than added in.

The requested maximum output length does not factor into output token
accounting. That is the fact worth the most: it means the predicted quantile is
fully usable here, and reserving the maximum would waste most of the headroom
for nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone

from spillway.core.cost import Cost
from spillway.estimators.base import RequestContext
from spillway.providers._read import count, field, path, usage_of
from spillway.providers.base import Outcome, ProviderState

_SPEND_CAP = "enforced_spend_limit_reached"

_LIMIT_HEADERS = {
    "rpm": "requests",
    "input_tpm": "input-tokens",
    "output_tpm": "output-tokens",
}
"""Which header family reports each limit this library names."""


def _now() -> datetime:
    """The current moment, for turning a reset timestamp into a wait."""
    return datetime.now(timezone.utc)


class Anthropic:
    """Anthropic's accounting rules.

    Args:
        now: Where the current moment comes from, for converting a reset
            timestamp into a number of seconds. Injectable because the provider
            reports an absolute time and the answer is otherwise untestable.

    Example:
        >>> adapter = Anthropic()
        >>> adapter.charges_max_tokens()
        False
        >>> context = adapter.request_from(
        ...     "messages.create",
        ...     {"model": "claude-sonnet-5", "max_tokens": 1024, "messages": []},
        ... )
        >>> context.model, context.max_tokens
        ('claude-sonnet-5', 1024)

        Cache reads are recorded apart from the input that was charged.

        >>> adapter.usage_from(
        ...     {"usage": {"input_tokens": 50, "output_tokens": 218,
        ...                "cache_read_input_tokens": 200_000}}
        ... )
        Cost(input_tokens=50, output_tokens=218, requests=1, extra={'cached_input': 200000})
    """

    name: str = "anthropic"
    client_module: str = "anthropic"
    official_hosts: tuple[str, ...] = ("api.anthropic.com",)
    endpoints: tuple[str, ...] = (
        "messages.create",
        "messages.parse",
        "beta.messages.create",
        "beta.messages.parse",
    )

    def __init__(self, *, now: Callable[[], datetime] = _now) -> None:
        """Hold where the current moment comes from."""
        self._now = now

    def __repr__(self) -> str:
        """Name the provider."""
        return "Anthropic()"

    def request_from(self, endpoint: str, kwargs: Mapping[str, object]) -> RequestContext:
        """Read the model, the prompt and the requested maximum off one call.

        The system prompt is a separate parameter rather than a message, so
        counting input from the messages alone would miss it entirely, and a
        long system prompt is exactly the case where that matters.

        Every endpoint here names its arguments identically, so which one was
        called does not change the answer. The other provider is not so
        obliging, which is why the endpoint is on the protocol at all.
        """
        del endpoint
        model = kwargs.get("model")
        max_tokens = kwargs.get("max_tokens")
        prompt: list[object] = []
        system = kwargs.get("system")
        if system is not None:
            prompt.append(system)
        messages = kwargs.get("messages")
        if isinstance(messages, (list, tuple)):
            prompt.extend(messages)
        return RequestContext(
            prompt=prompt or None,
            max_tokens=count(max_tokens) if max_tokens is not None else None,
            model=str(model) if model is not None else None,
        )

    def adjust(self, cost: Cost, context: RequestContext) -> Cost:
        """Return the reservation unchanged.

        This provider evaluates output tokens live against what is actually
        generated, so there is nothing to override. Present because the
        protocol has it and because the other provider does need it.
        """
        del context
        return cost

    def usage_from(self, response: object) -> Cost:
        """Read what a call really cost.

        The input charge is the input token field plus cache creation. Cache
        reads go into their own category rather than into the input total,
        because the provider does not count them against the input limit and
        folding them in would report a charge that was never made.

        Raises:
            ValueError: if there is no usage to read.
        """
        usage = usage_of(response)
        if usage is None:
            message = (
                "No usage found on this response, so what the call really cost is unknown. "
                "Pass the response object the client returned, or settle the lease by hand "
                "with lease.settle(input=..., output=...)."
            )
            raise ValueError(message)
        written = count(field(usage, "cache_creation_input_tokens"))
        read = count(field(usage, "cache_read_input_tokens"))
        return Cost(
            input_tokens=count(field(usage, "input_tokens")) + written,
            output_tokens=count(field(usage, "output_tokens")),
            requests=1,
            extra={"cached_input": read},
        )

    def classify(self, exc: BaseException | None, response: object | None) -> Outcome:
        """Say what happened, in the terms a controller acts on.

        The spend cap is the trap. It arrives as the same status and the same
        error type as an ordinary rate limit, and it is told apart only by an
        error code. Reading it as congestion would make a controller back the
        limit down against a condition that no amount of waiting clears.
        """
        del response
        if exc is None:
            return Outcome.OK
        status = field(exc, "status_code")
        if status == 429:
            code = path(field(exc, "body"), "error", "details", "error_code")
            return Outcome.ERROR if code == _SPEND_CAP else Outcome.RATE_LIMITED
        if status == 529:
            return Outcome.OVERLOADED
        if isinstance(status, int) and status >= 500:
            return Outcome.OVERLOADED
        if isinstance(status, int):
            return Outcome.ERROR
        return Outcome.ERROR

    def parse_headers(self, headers: Mapping[str, str]) -> ProviderState | None:
        """Read the provider's own account of each limit.

        Reset times arrive as absolute timestamps, so they are turned into
        seconds from now. A timestamp already in the past becomes zero rather
        than a negative wait.

        Returns:
            The state, or None when none of the headers are present, so that
            "not told" stays distinguishable from "nothing left".
        """
        limits: dict[str, float] = {}
        remaining: dict[str, float] = {}
        resets: dict[str, float] = {}
        for name, family in _LIMIT_HEADERS.items():
            limit = _number(headers.get(f"anthropic-ratelimit-{family}-limit"))
            if limit is not None:
                limits[name] = limit
            left = _number(headers.get(f"anthropic-ratelimit-{family}-remaining"))
            if left is not None:
                remaining[name] = left
            reset = self._seconds_until(headers.get(f"anthropic-ratelimit-{family}-reset"))
            if reset is not None:
                resets[name] = reset
        if not limits and not remaining:
            return None
        return ProviderState(limits=limits, remaining=remaining, reset_at=resets)

    def retry_after(self, exc: BaseException | None, headers: Mapping[str, str]) -> float | None:
        """How long the provider asked to be left alone for.

        Returns:
            Seconds, or None when the provider did not say. A spend cap sends
            no such header precisely because retrying will not help, so None is
            the honest answer there rather than zero.
        """
        seconds = _number(headers.get("retry-after"))
        if seconds is None and exc is not None:
            seconds = _number(_header_of(exc, "retry-after"))
        if seconds is None:
            return None
        return max(0.0, seconds)

    def charges_max_tokens(self) -> bool:
        """No. Output is metered against what is actually generated.

        Which is why the quantile machinery is worth having against this
        provider: reserving what nine requests in ten come in under is both
        safe and far cheaper than reserving the maximum.
        """
        return False

    def _seconds_until(self, value: str | None) -> float | None:
        """Turn an absolute reset timestamp into a wait in seconds."""
        if not value:
            return None
        text = value.strip()
        # The trailing designator is not accepted before 3.11, and this package
        # supports 3.10.
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            when = datetime.fromisoformat(text)
        except ValueError:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - self._now()).total_seconds())


def _number(value: str | None) -> float | None:
    """Read a header value as a number, or nothing if it is not one."""
    if value is None:
        return None
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None


def _header_of(exc: BaseException, name: str) -> str | None:
    """Pull one header off whatever response an exception is carrying."""
    headers = path(exc, "response", "headers")
    if headers is None:
        return None
    found = field(headers, "get")
    if callable(found):
        value = found(name)
        return value if isinstance(value, str) else None
    return None
