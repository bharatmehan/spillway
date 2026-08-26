"""The handle a caller holds while its request is in flight.

A lease is capacity that has been taken and not yet given back. Its whole job is
to make sure it is given back exactly once, whether the request succeeded, threw,
or was cancelled halfway through.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from enum import Enum

from spillway.core.cost import Cost
from spillway.core.errors import ConfigurationError, LeaseAlreadySettled, LeaseExpired
from spillway.core.scope import Scope
from spillway.dimensions.base import Dimension
from spillway.observability.explain import AdmissionExplanation
from spillway.providers.base import ProviderAdapter
from spillway.stores.base import Delta, SyncStore

_log = logging.getLogger(__name__)

_warned_about_unreadable_usage = False


class LeaseState(Enum):
    """Where a lease is in its life.

    Example:
        >>> LeaseState.ACQUIRED.value
        'acquired'
    """

    ACQUIRED = "acquired"
    """Capacity is held and the request is in flight."""

    SETTLED = "settled"
    """The real cost was reported and the difference reconciled."""

    ABANDONED = "abandoned"
    """The request never ran, so the whole reservation went back."""

    EXPIRED = "expired"
    """The lease outlived its expiry and its capacity was reclaimed without it."""


class Lease:
    """Capacity held for one in flight request.

    Settle it with the real cost when the request finishes. The difference
    between what was reserved and what was really used goes back immediately, so
    the surplus reaches the next caller within this request's own lifetime
    rather than at the end of the window.

    A lease that is never settled has its capacity recovered only when it
    expires, and until then the limit is smaller than it was configured to be.

    Attributes:
        id: Identifies this reservation to the store.
        scope: Whose budget this was taken from.
        priority: How urgent the request said it was.
        reserved: What was taken at admission.
        acquired_at_ms: When it was taken.
        state: Where this lease is in its life.

    A lease also takes two callbacks. `on_release` is called once whichever way
    it finishes, because capacity coming back is the event a waiter is asleep
    on. `on_settle` is called with the real cost and only from a settlement,
    since an abandoned request produced nothing to learn from.

    Example:
        >>> from spillway.core.clock import FakeClock
        >>> from spillway.core.scope import Scope
        >>> from spillway.dimensions.rate import Rate
        >>> from spillway.stores.memory import MemoryStore
        >>> clock, scope = FakeClock(), Scope("tenant:acme")
        >>> store = MemoryStore(clock=clock)
        >>> tokens = Rate("output_tpm", limit=1_000)
        >>> reserved = Cost(output_tokens=800)
        >>> result = store.reserve_sync(
        ...     [tokens.claim(reserved, scope)], ttl_ms=60_000.0, scope="tenant:acme", priority=0
        ... )
        >>> lease = Lease(
        ...     id=result.lease_id,
        ...     scope=scope,
        ...     priority=0,
        ...     reserved=reserved,
        ...     acquired_at_ms=clock.now_ms(),
        ...     dimensions=[tokens],
        ...     store=store,
        ...     explanation=AdmissionExplanation(admitted=True, scope="tenant:acme", priority=0),
        ... )
        >>> lease.settle(input=0, output=200)
        >>> lease.state
        <LeaseState.SETTLED: 'settled'>

        The six hundred tokens that were reserved and not used are already back.

        >>> store.snapshot_sync(["tenant:acme:output_tpm"])["tenant:acme:output_tpm"].used
        200.0
    """

    def __init__(
        self,
        *,
        id: str,
        scope: Scope,
        priority: int,
        reserved: Cost,
        acquired_at_ms: float,
        dimensions: Sequence[Dimension],
        store: SyncStore,
        explanation: AdmissionExplanation,
        waited_ms: float = 0.0,
        provider: ProviderAdapter | None = None,
        on_release: Callable[[], None] | None = None,
        on_settle: Callable[[Cost], None] | None = None,
    ) -> None:
        """Hold the reservation described by `explanation`."""
        self.id = id
        self.scope = scope
        self.priority = priority
        self.reserved = reserved
        self.acquired_at_ms = acquired_at_ms
        self.state = LeaseState.ACQUIRED
        self._dimensions = tuple(dimensions)
        self._store = store
        self._explanation = explanation
        self._waited_ms = waited_ms
        self._provider = provider
        self._on_release = on_release
        self._on_settle = on_settle
        self._reason: str | None = None

    def __repr__(self) -> str:
        """Show what is held and whether it still is."""
        return (
            f"Lease(id={self.id!r}, scope={self.scope.key!r}, "
            f"state={self.state.value!r}, reserved={self.reserved!r})"
        )

    @property
    def waited_ms(self) -> float:
        """How long the caller waited for this lease before getting it."""
        return self._waited_ms

    @property
    def explain(self) -> AdmissionExplanation:
        """Why this request was admitted, and how full everything was."""
        return self._explanation

    def settle(self, *, input: int, output: int, **extra: int) -> None:
        """Report the real cost and give back whatever was not used.

        Args:
            input: Input tokens the request actually consumed.
            output: Output tokens it actually generated.
            **extra: Any provider specific categories that were reserved,
                such as cached input tokens.

        Raises:
            LeaseAlreadySettled: if this lease was already settled or
                abandoned. Counting one request twice would corrupt every limit
                it touched.
            LeaseExpired: if the lease outlived its expiry and its capacity was
                already reclaimed.
        """
        self._require_held()
        actual = Cost(
            input_tokens=input,
            output_tokens=output,
            requests=self.reserved.requests,
            extra=extra,
        )
        deltas: list[Delta] = []
        for dimension in self._dimensions:
            delta = dimension.settle(self.reserved, actual, self.scope)
            if delta is not None:
                deltas.append(delta)
        # Reported before the store is asked, not after. The gap between
        # self.reserved and actual is the estimate error, and it is just as
        # true when the reservation turns out to have already expired: the
        # bookkeeping failed, the request still generated what it generated.
        if self._on_settle is not None:
            self._on_settle(actual)
        try:
            self._store.settle_sync(self.id, deltas)
        except LeaseExpired:
            self.state = LeaseState.EXPIRED
            raise
        finally:
            self._released()
        self.state = LeaseState.SETTLED

    def settle_from(self, response: object) -> None:
        """Report the real cost by reading it off the provider's own response.

        The same as `settle`, without needing to know which fields this
        provider names its counts after. Accepts the object the client library
        returned, a plain mapping, or a usage record on its own.

        Args:
            response: Whatever the call came back with.

        Raises:
            ConfigurationError: if the limiter was built with no provider, so
                there is nothing that knows how to read this.
            LeaseAlreadySettled: as `settle`.
            LeaseExpired: as `settle`.

        A response this provider cannot find usage on does not raise: the call
        already succeeded, and losing the caller's result over the bookkeeping
        is the worse trade. It settles at the full reserved amount, which is
        safe and expensive, and says so once.

        Example:
            >>> from spillway.core.spillway import Spillway
            >>> import asyncio
            >>> limiter = Spillway(provider="anthropic")
            >>> async def one_call() -> int:
            ...     async with limiter.admit(max_tokens=1_000) as lease:
            ...         reply = {"usage": {"input_tokens": 12, "output_tokens": 34}}
            ...         lease.settle_from(reply)
            ...         return lease.reserved.output_tokens
            >>> asyncio.run(one_call())
            1000
        """
        if self._provider is None:
            message = (
                "settle_from() needs a provider to read the response with, and this "
                "limiter has none. Pass provider='anthropic' or provider='openai' when "
                "building it, or settle by hand with "
                "lease.settle(input=..., output=...)."
            )
            raise ConfigurationError(message)
        try:
            actual = self._provider.usage_from(response)
        except (ValueError, TypeError, AttributeError):
            _warn_once_about_unreadable_usage(self._provider.name)
            self.settle(
                input=self.reserved.input_tokens,
                output=self.reserved.output_tokens,
            )
            return
        self.settle(
            input=actual.input_tokens,
            output=actual.output_tokens,
            **dict(actual.extra),
        )

    def abandon(self, reason: str | None = None) -> None:
        """Give the whole reservation back, because the request never ran.

        Nothing was consumed, so there is nothing to reconcile. Abandoning an
        already finished lease does nothing: this runs on the failure path and
        raising there would bury the original error.

        Args:
            reason: Recorded for the caller's benefit. Nothing branches on it.
        """
        if self.state is not LeaseState.ACQUIRED:
            return
        self._reason = reason
        try:
            self._store.release_sync(self.id)
        finally:
            self._released()
        self.state = LeaseState.ABANDONED

    def _released(self) -> None:
        """Say that this lease has given up whatever it was holding.

        Called once on every path that ends a lease, including a settlement
        that found the reservation already expired: the capacity came back when
        it was reclaimed, so a waiter should still be told.
        """
        if self._on_release is not None:
            self._on_release()

    def _require_held(self) -> None:
        """Refuse to act on a lease that is no longer holding anything.

        Raises:
            LeaseAlreadySettled: if the lease has already finished.
        """
        if self.state is LeaseState.ACQUIRED:
            return
        message = (
            f"Lease {self.id!r} is {self.state.value} and cannot be settled again. "
            f"A second settlement would count the same request twice on every limit "
            f"it touched. Settle exactly once, or let the context manager do it."
        )
        raise LeaseAlreadySettled(message)


def _warn_once_about_unreadable_usage(provider: str) -> None:
    """Say, once, that a response could not be read and what it cost.

    Once per process rather than per request. The fix is one change at one call
    site.
    """
    global _warned_about_unreadable_usage
    if _warned_about_unreadable_usage:
        return
    _warned_about_unreadable_usage = True
    _log.warning(
        "The %s adapter could not find usage on a response, so the full reserved "
        "amount was charged instead of what the call really cost. That is safe but "
        "expensive: nothing corrects a reservation if the real figure is never read. "
        "Pass the object the client returned, or call lease.settle(input=..., "
        "output=...) with the figures yourself.",
        provider,
    )
