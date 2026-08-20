"""The limiter itself.

One object, held for the life of the process, asked before every call to a model
whether that call may go now.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType

from spillway.core.clock import Clock, MonotonicClock
from spillway.core.cost import Cost, Estimate, default_estimate
from spillway.core.dispatcher import Dispatcher
from spillway.core.errors import AdmissionDenied, ConfigurationError, LeaseExpired
from spillway.core.lease import Lease, LeaseState
from spillway.core.queue import DEFAULT_QUEUE_CAPACITY, QueueFullPolicy, Waiter, WaitQueue
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

DEFAULT_TIMEOUT_S = 30.0
"""How long a caller who names no timeout waits before giving up.

Waiting for ever is almost never what anyone meant, and it is the failure that
looks like the library hanging rather than like a limit being reached. Thirty
seconds is long enough to ride out an ordinary burst and short enough that a
request stuck behind something pathological still returns an error somebody can
read. Pass `default_timeout=None` to wait for as long as it takes.
"""


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
        default_timeout: How many seconds to wait for capacity when a caller
            names neither a timeout nor a deadline. Zero refuses rather than
            waits. None waits for as long as it takes.
        queue_capacity: How many requests may wait in each priority band. Per
            band rather than shared, so a flood of batch work cannot consume
            the slots an interactive request needs.
        queue_full_policy: What a full band does with a new arrival. "reject"
            refuses it. "shed_lowest" drops the lowest priority waiter to make
            room for a higher priority one, and refuses when the arrival is
            itself the lowest.

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
        default_timeout: float | None = DEFAULT_TIMEOUT_S,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
        queue_full_policy: QueueFullPolicy = "reject",
    ) -> None:
        """Assemble a limiter. Every argument has a usable default.

        Raises:
            ConfigurationError: if `default_timeout` is negative, if
                `queue_capacity` is below one, or if `queue_full_policy` is not
                one the queue knows.
        """
        if default_timeout is not None and default_timeout < 0:
            message = (
                f"default_timeout is how many seconds to wait, so it cannot be negative, "
                f"got {default_timeout}. Use 0 to refuse rather than wait, or None to wait "
                f"for as long as it takes."
            )
            raise ConfigurationError(message)
        self._default_timeout = default_timeout
        self._clock: Clock = clock if clock is not None else MonotonicClock()
        self._dimensions = tuple(dimensions)
        self._store: DuplexStore = store if store is not None else MemoryStore(clock=self._clock)
        self._default_scope = Scope.of(scope)
        self._queue = WaitQueue(capacity=queue_capacity, policy=queue_full_policy)
        self._dispatcher = Dispatcher(limiter=self, queue=self._queue, clock=self._clock)

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
            timeout: How many seconds to wait for capacity. Defaults to the
                limiter's own. Zero refuses rather than waits. Mutually
                exclusive with `deadline`.
            deadline: A fixed point to give up at, in seconds on the same
                monotonic scale the standard library reports. Mutually
                exclusive with `timeout`.
            weight: Reserved for fair sharing, which does not exist yet.

        Returns:
            A context that reserves capacity when entered or acquired.

        Raises:
            ConfigurationError: if both `timeout` and `deadline` are given.

        The last argument is accepted and unused. It is in the signature now so
        that the shape a caller writes against, and the shape an editor shows
        them, does not change when it starts working.
        """
        if timeout is not None and deadline is not None:
            message = (
                f"admit() takes a timeout or a deadline, not both, and got "
                f"timeout={timeout} and deadline={deadline}. They say the same thing two "
                f"ways and there is no honest answer when they disagree. Keep the one you "
                f"mean: timeout for seconds from now, deadline for a fixed point to stop at."
            )
            raise ConfigurationError(message)
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

    def _claims_for(
        self,
        *,
        scope: Scope,
        priority: int,
        reserved: Cost,
    ) -> tuple[tuple[Claim, ...], Mapping[str, str]]:
        """Work out what one admission has to take, and from which key.

        Built once and reused for every attempt, because a request that is
        retried after waiting asks for exactly what it asked for the first time.

        Returns:
            The claims, and the dimension name each claim's key belongs to,
            which is what turns a store level key back into something worth
            showing a caller.

        Raises:
            AdmissionDenied: if the request is larger than a limit and so could
                never have room, however long it waited.
        """
        claims: list[Claim] = []
        dimension_of_key: dict[str, str] = {}
        for dimension in self._dimensions:
            claim = dimension.claim(reserved, scope)
            if claim is None:
                continue
            if claim.cost > claim.limit:
                raise AdmissionDenied(
                    _impossible_message(dimension.name, claim.cost, claim.limit),
                    binding_dimension=dimension.name,
                    explanation=AdmissionExplanation(
                        admitted=False,
                        scope=scope.key,
                        priority=priority,
                        binding_dimension=dimension.name,
                    ),
                )
            claims.append(claim)
            dimension_of_key[claim.key] = dimension.name
        return tuple(claims), dimension_of_key

    async def _attempt(
        self,
        *,
        claims: Sequence[Claim],
        dimension_of_key: Mapping[str, str],
        scope: Scope,
        priority: int,
        reserved: Cost,
        waited_ms: float = 0.0,
        queue_position: int | None = None,
    ) -> Lease:
        """Ask the store for the whole batch once, and report what happened.

        One attempt, no waiting. Both the direct path and the dispatcher go
        through here, so a request that waited is admitted by exactly the same
        code as one that did not.

        The wait and the queue position are passed in rather than worked out
        here, because this is the one place that does not know whether there
        was a queue at all.

        Raises:
            AdmissionDenied: if any dimension has no room. It carries the
                binding dimension and how long until it would fit, which is
                what a waiter is scheduled on.
        """
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
            waited_ms=waited_ms,
            binding_dimension=binding,
            queue_position=queue_position,
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
            waited_ms=waited_ms,
            on_release=self._dispatcher.notify,
        )

    def _deadline_ms(
        self,
        *,
        timeout: float | None,
        deadline: float | None,
        now_ms: float,
    ) -> float | None:
        """When to stop waiting, on the limiter's clock.

        A deadline that has already passed, which includes a timeout of zero,
        means the caller asked not to wait, and the refusal reaches them
        unchanged rather than dressed up as a timeout.

        Returns:
            The moment to give up, or None to wait for as long as it takes.
        """
        if deadline is not None:
            return deadline * 1000.0
        if timeout is not None:
            return now_ms + timeout * 1000.0
        if self._default_timeout is None:
            return None
        return now_ms + self._default_timeout * 1000.0

    async def _acquire(
        self,
        *,
        scope: Scope,
        priority: int,
        reserved: Cost,
        timeout: float | None = None,
        deadline: float | None = None,
    ) -> Lease:
        """Reserve `reserved` across every dimension, waiting if allowed to.

        The reservation is attempted directly first and the queue is only
        reached on a refusal, so the case where there is room, which is nearly
        every case, pays nothing at all for the machinery below.

        Raises:
            AdmissionDenied: if there is no room and none arrives in time, or
                if the request is larger than a limit and so could never have
                room.
            AdmissionTimeout: if the wait ran out first.
            Shed: if the work is sheddable and its band is full.
        """
        started_ms = self._clock.now_ms()
        claims, dimension_of_key = self._claims_for(
            scope=scope,
            priority=priority,
            reserved=reserved,
        )
        deadline_ms = self._deadline_ms(
            timeout=timeout,
            deadline=deadline,
            now_ms=started_ms,
        )
        try:
            return await self._attempt(
                claims=claims,
                dimension_of_key=dimension_of_key,
                scope=scope,
                priority=priority,
                reserved=reserved,
            )
        except AdmissionDenied as refusal:
            if deadline_ms is not None and deadline_ms <= self._clock.now_ms():
                raise
            return await self._wait_for_room(
                Waiter(
                    claims=claims,
                    dimension_of_key=dimension_of_key,
                    scope=scope,
                    priority=priority,
                    reserved=reserved,
                    deadline_ms=deadline_ms,
                    queued_at_ms=started_ms,
                    future=asyncio.get_running_loop().create_future(),
                    refusal=refusal,
                    refused_at_ms=self._clock.now_ms(),
                )
            )

    async def _wait_for_room(self, waiter: Waiter) -> Lease:
        """Queue and wait until the dispatcher has an answer.

        A cancelled caller takes itself out of the queue on the way past. The
        dispatcher would eventually notice and drop it, but not before it has
        been selected, and a waiter nobody is listening for still holds its
        band's head against everyone behind it.

        Raises:
            AdmissionDenied: if the queue itself has no room for this waiter.
            AdmissionTimeout: if the wait ran out.
            Shed: if the work is sheddable and its band is full.
        """
        self._queue.push(waiter)
        self._dispatcher.ensure_running()
        try:
            return await waiter.future
        finally:
            # Harmless when the dispatcher has already taken it out, and
            # neither side can tell which of them got here first.
            self._queue.remove(waiter)


def _impossible_message(name: str, cost: float, limit: float) -> str:
    """Say that no amount of waiting will help, and what would.

    Without this the request sits at the head of its priority band for ever,
    blocking everything behind it, and nothing about the symptom points at the
    one request that is too big to ever fit.
    """
    return (
        f"This request reserves {cost:,.0f} against a {name} limit of {limit:,.0f}. "
        f"It can never be admitted, however long it waits. Either raise the limit, "
        f"or lower max_tokens, or split the request."
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
            AdmissionDenied: if there is no room and none arrives in time.
            AdmissionTimeout: if the wait ran out.
            Shed: if the work is sheddable and its band is full.
        """
        return await self._limiter._acquire(
            scope=self._scope,
            priority=self._priority,
            reserved=self._reserved(),
            timeout=self._timeout,
            deadline=self._deadline,
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
