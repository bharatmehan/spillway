"""The limiter itself.

One object, held for the life of the process, asked before every call to a model
whether that call may go now.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType

from spillway.core.clock import Clock, MonotonicClock
from spillway.core.cost import Cost, Estimate, default_estimate
from spillway.core.errors import AdmissionDenied, LeaseExpired
from spillway.core.lease import Lease, LeaseState
from spillway.core.scope import Priority, Scope
from spillway.dimensions.base import Dimension, claim_key
from spillway.observability.explain import AdmissionExplanation
from spillway.stores.base import Claim, DuplexStore, Utilisation
from spillway.stores.memory import MemoryStore

_log = logging.getLogger(__name__)

_warned_about_unsettled = False

RESERVATION_QUANTILE = 0.9
"""Which point of the predicted output distribution to reserve.

Reserving the median means overrunning half the time, which defeats the limit.
Reserving the worst case is what a provider does and it collapses throughput.
The ninth decile overruns around one request in ten, and because the surplus is
credited back the moment the real figure is known, holding it costs almost
nothing.

Nothing observes this yet: both distributions available so far answer every
quantile with the same number. It becomes load bearing once output length is
predicted from history.
"""

# ponytail: a flat expiry, wrong for both a two second classification and a six
# minute reasoning call. It becomes a function of observed durations once those
# are being measured. Until then a call that runs longer than this loses its
# reservation and cannot settle, which the resulting error says plainly.
DEFAULT_LEASE_TTL_MS = 60_000.0
"""How long a reservation may go unsettled before its capacity is reclaimed."""


@dataclass(frozen=True)
class Snapshot:
    """How full everything is right now, for one scope.

    Cheap enough for a health check, and it reserves nothing, so calling it on
    a timer cannot affect what gets admitted.

    Example:
        >>> from spillway.dimensions.rate import Rate
        >>> limiter = Spillway(dimensions=[Rate("rpm", limit=1_000)], scope="tenant:acme")
        >>> found = limiter.snapshot()
        >>> found.scope
        'tenant:acme'
        >>> found.dimensions["rpm"].limit
        1000.0
    """

    scope: str
    dimensions: Mapping[str, Utilisation]


class Spillway:
    """Decides whether a request may proceed, and tracks what it consumed.

    Hold one for the life of the process and share it across tasks. Every
    argument has a default, and `Spillway()` with none is valid: it tracks and
    reports and never refuses anything, which is a reasonable first step for
    someone gathering evidence before choosing limits.

    Args:
        dimensions: The limits to enforce. Empty means enforce nothing.
        store: Where reservations are recorded. Defaults to an in memory store,
            which is correct within one process and not across several.
        clock: Where time comes from.
        scope: The scope used when a caller names none.

    Example:
        >>> import asyncio
        >>> from spillway.dimensions.concurrency import Concurrency
        >>> from spillway.dimensions.rate import Rate
        >>> limiter = Spillway(
        ...     dimensions=[
        ...         Rate("rpm", limit=1_000),
        ...         Concurrency("generations", limit=2),
        ...     ]
        ... )
        >>> async def one_call() -> str:
        ...     lease = await limiter.admit(scope="tenant:acme", max_tokens=500).acquire()
        ...     try:
        ...         return "the model's answer"
        ...     finally:
        ...         lease.settle(input=120, output=48)
        >>> asyncio.run(one_call())
        "the model's answer"
    """

    def __init__(
        self,
        *,
        dimensions: Sequence[Dimension] = (),
        store: DuplexStore | None = None,
        clock: Clock | None = None,
        scope: str | Scope | None = None,
    ) -> None:
        """Assemble a limiter. Every argument has a usable default."""
        self._clock: Clock = clock if clock is not None else MonotonicClock()
        self._dimensions = tuple(dimensions)
        self._store: DuplexStore = store if store is not None else MemoryStore(clock=self._clock)
        self._default_scope = Scope.of(scope)

    def __repr__(self) -> str:
        """Show what is being enforced."""
        names = ", ".join(dimension.name for dimension in self._dimensions)
        return f"Spillway(dimensions=[{names}], scope={self._default_scope.key!r})"

    @property
    def dimensions(self) -> tuple[Dimension, ...]:
        """The limits being enforced."""
        return self._dimensions

    def admit(
        self,
        *,
        scope: str | Scope | None = None,
        priority: int = Priority.NORMAL,
        estimate: Estimate | None = None,
        prompt: str | Sequence[object] | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        timeout: float | None = None,
        deadline: float | None = None,
        weight: float = 1.0,
    ) -> AdmitContext:
        """Ask whether a request may proceed, and reserve what it will cost.

        Args:
            scope: Whose budget to draw from. Defaults to the limiter's.
            priority: How urgent this is. Negative means sheddable.
            estimate: The cost, if the caller knows it. Otherwise it is derived
                from `prompt`, `max_tokens` and `model`.
            prompt: Used to count input tokens when no estimate is given.
            max_tokens: The requested output limit.
            model: Recorded on the estimate when known.
            timeout: Reserved for waiting, which does not exist yet.
            deadline: Reserved for waiting, which does not exist yet.
            weight: Reserved for fair sharing, which does not exist yet.

        Returns:
            A context that reserves capacity when entered or acquired.

        The last three arguments are accepted and unused. They are in the
        signature now so that the shape a caller writes against, and the shape
        an editor shows them, does not change when they start working.
        """
        return AdmitContext(
            limiter=self,
            scope=Scope.of(scope if scope is not None else self._default_scope),
            priority=int(priority),
            estimate=estimate,
            prompt=prompt,
            max_tokens=max_tokens,
            model=model,
            timeout=timeout,
            deadline=deadline,
            weight=weight,
        )

    def snapshot(self, scope: str | Scope | None = None) -> Snapshot:
        """Report how full every limit is, without reserving anything.

        Args:
            scope: Whose budget to report on. Defaults to the limiter's.

        Returns:
            A snapshot keyed by dimension name.

        Limits come from the dimensions rather than from the store, so a
        dimension that has never been claimed against reports as empty out of
        its real limit instead of as empty out of nothing.
        """
        resolved = Scope.of(scope if scope is not None else self._default_scope)
        name_of_key = {claim_key(resolved, d.name): d for d in self._dimensions}
        found = self._store.snapshot_sync(list(name_of_key))
        return Snapshot(
            scope=resolved.key,
            dimensions={
                dimension.name: Utilisation(
                    used=found[key].used,
                    limit=dimension.limit,
                )
                for key, dimension in name_of_key.items()
            },
        )

    async def _acquire(
        self,
        *,
        scope: Scope,
        priority: int,
        reserved: Cost,
    ) -> Lease:
        """Reserve `reserved` across every dimension, or refuse.

        Raises:
            AdmissionDenied: if any dimension has no room.
        """
        claims: list[Claim] = []
        dimension_of_key: dict[str, str] = {}
        for dimension in self._dimensions:
            claim = dimension.claim(reserved, scope)
            if claim is not None:
                claims.append(claim)
                dimension_of_key[claim.key] = dimension.name

        result = await self._store.reserve(
            claims,
            ttl_ms=DEFAULT_LEASE_TTL_MS,
            scope=scope.key,
            priority=priority,
        )
        binding = (
            dimension_of_key.get(result.binding_key, result.binding_key)
            if result.binding_key is not None
            else None
        )
        explanation = AdmissionExplanation(
            admitted=result.granted,
            scope=scope.key,
            priority=priority,
            binding_dimension=binding,
            dimensions={
                dimension_of_key[key]: used
                for key, used in result.utilisation.items()
                if key in dimension_of_key
            },
        )
        if not result.granted or result.lease_id is None:
            raise AdmissionDenied(
                _refusal_message(binding, result.retry_after_ms),
                retry_after=(
                    None if result.retry_after_ms is None else result.retry_after_ms / 1000.0
                ),
                binding_dimension=binding,
                explanation=explanation,
            )
        return Lease(
            id=result.lease_id,
            scope=scope,
            priority=priority,
            reserved=reserved,
            acquired_at_ms=self._clock.now_ms(),
            dimensions=self._dimensions,
            store=self._store,
            explanation=explanation,
        )


def _refusal_message(binding: str | None, retry_after_ms: float | None) -> str:
    """Say what ran out and what to do about it."""
    what = f"No room on {binding}." if binding is not None else "No room."
    if retry_after_ms is None:
        return (
            f"{what} Capacity frees when an in flight request finishes, so waiting on a "
            f"timer will not help. Catch AdmissionDenied and retry, or raise the limit."
        )
    return (
        f"{what} It would fit in {retry_after_ms / 1000.0:.3g}s. Catch AdmissionDenied "
        f"and retry after that, or raise the limit."
    )


class AdmitContext:
    """A reservation that has been asked for but not yet taken.

    Returned by `Spillway.admit`. Nothing is reserved until it is acquired, so
    building one costs nothing and holds nothing.

    Example:
        >>> import asyncio
        >>> limiter = Spillway()
        >>> async def one_call() -> str:
        ...     async with limiter.admit(scope="tenant:acme") as lease:
        ...         answer = "the model's answer"
        ...         lease.settle(input=100, output=20)
        ...         return answer
        >>> asyncio.run(one_call())
        "the model's answer"

        Or without a context manager, for callers who cannot use one.

        >>> lease = asyncio.run(limiter.admit(scope="tenant:acme").acquire())
        >>> lease.scope.key
        'tenant:acme'
        >>> lease.settle(input=100, output=20)
    """

    def __init__(
        self,
        *,
        limiter: Spillway,
        scope: Scope,
        priority: int,
        estimate: Estimate | None,
        prompt: str | Sequence[object] | None,
        max_tokens: int | None,
        model: str | None,
        timeout: float | None,
        deadline: float | None,
        weight: float,
    ) -> None:
        """Record what was asked for, without asking for it yet."""
        self._limiter = limiter
        self._scope = scope
        self._priority = priority
        self._estimate = estimate
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._model = model
        self._timeout = timeout
        self._deadline = deadline
        self._weight = weight
        self._lease: Lease | None = None

    async def acquire(self) -> Lease:
        """Reserve the capacity and return the lease holding it.

        For callers who cannot use a context manager. Settling, or abandoning,
        is then the caller's responsibility and belongs in a finally block.

        Raises:
            AdmissionDenied: if any dimension has no room.
        """
        return await self._limiter._acquire(
            scope=self._scope,
            priority=self._priority,
            reserved=self._reserved(),
        )

    def _reserved(self) -> Cost:
        """Work out what to reserve, from the estimate or from the prompt."""
        estimate = self._estimate
        if estimate is None:
            estimate = default_estimate(
                self._prompt,
                max_tokens=self._max_tokens,
                model=self._model,
            )
        return Cost(
            input_tokens=estimate.input,
            output_tokens=estimate.output.quantile(RESERVATION_QUANTILE),
            requests=1,
        )

    async def __aenter__(self) -> Lease:
        """Reserve the capacity and hand over the lease.

        Raises:
            AdmissionDenied: if any dimension has no room.
        """
        self._lease = await self.acquire()
        return self._lease

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Give the capacity back, whichever way the block ended.

        Four endings, and each one has a different right answer.

        The block raised, or the task was cancelled: the reservation goes back
        whole, because nothing was consumed. Cancellation reaches here as an
        exception like any other, and the release path awaits nothing, so it
        cannot itself be interrupted partway through.

        The block succeeded and settled: nothing left to do.

        The block succeeded and did not settle: settle at the full reserved
        amount, which is pessimistic and safe, and say so once. Nothing will
        ever calibrate for a caller who never reports real costs, so every
        request keeps paying the estimate's full price.
        """
        lease = self._lease
        if lease is None or lease.state is not LeaseState.ACQUIRED:
            return False
        if exc_type is not None:
            lease.abandon(reason=exc_type.__name__)
            return False
        _warn_once_about_unsettled()
        try:
            lease.settle(
                input=lease.reserved.input_tokens,
                output=lease.reserved.output_tokens,
            )
        except LeaseExpired:
            # The call outran its expiry and the capacity is already back. Say
            # so, but do not raise: the caller's work succeeded, and throwing
            # their result away over the bookkeeping would be the worse trade.
            _log.warning(
                "A request finished after its reservation had already expired and been "
                "reclaimed, so its real cost was never recorded. The call took longer "
                "than the limiter was told to expect."
            )
        return False

    def __enter__(self) -> Lease:
        """Refuse to run, and say what to use instead.

        Raises:
            RuntimeError: always, until there is a synchronous facade.
        """
        message = (
            "Spillway.admit() does not work with a plain `with` statement yet. Use "
            "`async with limiter.admit(...) as lease:` instead. This deliberately does "
            "not start an event loop on your behalf: inside a running loop that "
            "deadlocks, and outside one it hides that the calling code is synchronous, "
            "which is a decision worth making on purpose."
        )
        raise RuntimeError(message)

    def __exit__(self, *_: object) -> None:
        """Unreachable, because entering always raises."""


def _warn_once_about_unsettled() -> None:
    """Say, once, that a lease was settled at its reserved amount by default.

    Once per process rather than per request, because the fix is one change in
    the calling code and repeating it every time would only teach people to
    filter the message out.
    """
    global _warned_about_unsettled
    if _warned_about_unsettled:
        return
    _warned_about_unsettled = True
    _log.warning(
        "A request finished without reporting what it actually cost, so the full "
        "reserved amount was charged. That is safe but expensive: the reservation is "
        "an estimate, and nothing corrects it if the real figure is never reported. "
        "Call lease.settle(input=..., output=...) before the block ends."
    )
