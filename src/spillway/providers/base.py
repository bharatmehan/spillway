"""What every provider adapter has to answer, and nothing more.

Providers do not count the same way. One meters requests, input and output on
three separate buckets and charges only what a call generated. Another puts
input and output in one bucket and charges the maximum the caller allowed,
reached or not. Reserving the wrong one produces rate limit responses the
library completely failed to predict.

An adapter is where one provider's rules live. Half reads a request, so a call's
cost is known before making it; half reads a response, so settlement can replace
the estimate with the truth. Neither half touches a network.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from spillway.core.cost import Cost
from spillway.estimators.base import RequestContext


class Outcome(Enum):
    """What became of one request, in the terms a controller cares about.

    What matters is the split between the two meaning "the provider is full"
    and the rest. Backing off is the wrong answer to a malformed request or an
    exhausted budget: neither clears, so the limit walks down and stays down.

    Example:
        >>> Outcome.RATE_LIMITED.value
        'rate_limited'
        >>> Outcome.RATE_LIMITED.is_congestion, Outcome.ERROR.is_congestion
        (True, False)
    """

    OK = "ok"
    """The provider answered."""

    RATE_LIMITED = "rate_limited"
    """A published limit was reached. Capacity returns on its own."""

    OVERLOADED = "overloaded"
    """The provider had no capacity, independently of any limit."""

    ERROR = "error"
    """Something else failed, and waiting will not change it."""

    ABANDONED = "abandoned"
    """The request never reached the provider."""

    @property
    def is_congestion(self) -> bool:
        """Whether backing off is a sensible response to this.

        Example:
            >>> [o.value for o in Outcome if o.is_congestion]
            ['rate_limited', 'overloaded']
        """
        return self in (Outcome.RATE_LIMITED, Outcome.OVERLOADED)


@dataclass(frozen=True)
class ProviderState:
    """What a provider said about its own limits, on one response.

    An adapter for a provider that reports nothing returns no state at all
    rather than an empty one, so "nothing left" and "not told" stay distinct.
    Read by the header driven controller, which arrives in a later release.

    Attributes:
        limits: The configured limit per dimension name.
        remaining: How much of each is left. Some providers round this.
        reset_at: When each refills, in seconds on the limiter's clock.
        observed_at: When this was read, on the same clock. State from four
            minutes ago describes a window that has since refilled.

    Example:
        >>> state = ProviderState(
        ...     limits={"input_tpm": 2_000_000.0},
        ...     remaining={"input_tpm": 1_907_000.0},
        ...     reset_at={"input_tpm": 41.0},
        ...     observed_at=12.5,
        ... )
        >>> state.headroom("input_tpm")
        0.9535
        >>> state.headroom("output_tpm") is None
        True
    """

    limits: Mapping[str, float] = field(default_factory=dict)
    remaining: Mapping[str, float] = field(default_factory=dict)
    reset_at: Mapping[str, float] = field(default_factory=dict)
    observed_at: float = 0.0

    def headroom(self, name: str) -> float | None:
        """What fraction of `name` is still free, or nothing if it was not reported.

        Returns:
            A fraction between zero and one, or None when the limit or the
            remaining figure is missing, or the limit is zero.
        """
        limit = self.limits.get(name)
        left = self.remaining.get(name)
        if limit is None or left is None or limit <= 0:
            return None
        return left / limit


@runtime_checkable
class ProviderAdapter(Protocol):
    """One provider's accounting rules, and how to recognise its client.

    Implement it structurally: nothing needs importing or subclassing, and the
    shared conformance suite runs against every adapter equally, including one
    written outside this repository.

    The four attributes are recognition. A method not listed in `endpoints` is
    forwarded untouched, which is the safe direction to fail.

    **An adapter has no opinion about what your limits are.** It encodes how a
    provider counts, never how much you are allowed. The caller names their
    limits and this describes how to charge against them.

    Attributes:
        name: What this provider is called, in errors and in messages.
        client_module: The top level module of a client speaking this protocol.
        official_hosts: Base URLs this provider's accounting applies to. A
            client pointed elsewhere speaks the same protocol against a
            different service.
        endpoints: Dotted method paths to instrument, such as
            `messages.create`. Only methods that reach the network: listing one
            that delegates to another admits the same call twice.

    Example:
        A minimal adapter for a service that meters requests and nothing else.

        >>> class Tiny:
        ...     name = "tiny"
        ...     client_module = "tinyai"
        ...     official_hosts = ("https://api.tiny.example",)
        ...     endpoints = ("chat.create",)
        ...
        ...     def request_from(self, endpoint, kwargs):
        ...         return RequestContext(model=str(kwargs.get("model", "")))
        ...
        ...     def adjust(self, cost, context):
        ...         return cost
        ...
        ...     def usage_from(self, response):
        ...         return Cost(input_tokens=0, output_tokens=0)
        ...
        ...     def classify(self, exc, response):
        ...         return Outcome.OK if exc is None else Outcome.ERROR
        ...
        ...     def parse_headers(self, headers):
        ...         return None
        ...
        ...     def retry_after(self, exc, headers):
        ...         return None
        ...
        ...     def charges_max_tokens(self):
        ...         return False
        >>> isinstance(Tiny(), ProviderAdapter)
        True
    """

    name: str
    client_module: str
    official_hosts: tuple[str, ...]
    endpoints: tuple[str, ...]

    def request_from(self, endpoint: str, kwargs: Mapping[str, object]) -> RequestContext:
        """Read one intercepted call into the context an estimator understands.

        The mirror of `usage_from`. Every provider names the prompt and the
        requested maximum differently, sometimes differently per endpoint,
        which is why recovering them is the adapter's job.

        Args:
            endpoint: Which of `endpoints` was called.
            kwargs: What the caller passed, with the reserved keywords already
                removed.
        """
        ...

    def adjust(self, cost: Cost, context: RequestContext) -> Cost:
        """Apply this provider's admission time accounting to a reservation.

        Where a provider charges the requested maximum whether or not it is
        reached, the predicted output is replaced by that maximum here.
        Reserving less than the provider does means believing in headroom it
        does not agree exists.

        Must be idempotent, and the conformance suite asserts it.
        """
        ...

    def usage_from(self, response: object) -> Cost:
        """Pull what a call really cost out of whatever the provider returned.

        Accepts a native response object, a plain mapping, or anything else
        carrying the fields. Typed as `object` because the set of things a
        provider might return is not closed; the adapter narrows it itself.

        Raises:
            ValueError: if no usage can be found. The caller settles at the
                reserved amount and says so, which is safe and wasteful.
        """
        ...

    def classify(self, exc: BaseException | None, response: object | None) -> Outcome:
        """Say what happened, in terms a controller can act on.

        The one thing never to get wrong: a client error is not congestion.
        Reading a malformed request as overload collapses the limiter for a
        reason no amount of backing off can fix.
        """
        ...

    def parse_headers(self, headers: Mapping[str, str]) -> ProviderState | None:
        """Read the provider's own account of its limits, if it gives one.

        Returns:
            The state, or None for a provider that reports nothing, so silence
            is never read as an empty budget.
        """
        ...

    def retry_after(self, exc: BaseException | None, headers: Mapping[str, str]) -> float | None:
        """How many seconds the provider asked to be left alone for.

        Returns:
            Seconds, never negative, or None when the provider did not say.
            None rather than zero, which would invite an immediate retry.
        """
        ...

    def charges_max_tokens(self) -> bool:
        """Whether the requested maximum is charged at admission.

        The single most consequential fact about a provider. False means the
        predicted output quantile is fully usable. True means output prediction
        does not help against this provider's rate limit at all, and the
        documentation says so rather than implying a uniform benefit.
        """
        ...
