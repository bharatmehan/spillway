"""The limiter itself.

One object, held for the life of the process, asked before every call to a model
whether that call may go now.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol, TypeVar, overload

from spillway.core.clock import Clock, MonotonicClock
from spillway.core.cost import Cost, Estimate
from spillway.core.dispatcher import Dispatcher
from spillway.core.errors import AdmissionDenied, ConfigurationError, LeaseExpired
from spillway.core.lease import Lease, LeaseState
from spillway.core.queue import DEFAULT_QUEUE_CAPACITY, QueueFullPolicy, Waiter, WaitQueue
from spillway.core.scope import Priority, Scope
from spillway.dimensions.base import Dimension, claim_key
from spillway.dimensions.concurrency import Concurrency
from spillway.dimensions.rate import Rate
from spillway.estimators.base import Estimator, Observation, RequestContext
from spillway.estimators.max_tokens import MaxTokensEstimator
from spillway.estimators.quantile import QuantileEstimator
from spillway.observability.explain import AdmissionExplanation
from spillway.providers.base import ProviderAdapter
from spillway.stores.base import Claim, DuplexStore, Utilisation
from spillway.stores.memory import MemoryStore

_log = logging.getLogger(__name__)

_warned_about_unsettled = False
_warned_about_no_limits = False

# ponytail: a flat expiry, wrong for both a two second classification and a six
# minute reasoning call. It becomes a function of observed durations once those
# are being measured. Until then a call that runs longer than this loses its
# reservation and cannot settle, which the resulting error says plainly.
DEFAULT_LEASE_TTL_MS = 60_000.0
"""How long a reservation may go unsettled before its capacity is reclaimed."""

DEFAULT_TIMEOUT_S = 30.0
"""How long a caller who names no timeout waits before giving up.

Long enough to ride out an ordinary burst, short enough that a request stuck
behind something pathological still returns a readable error rather than looking
like a hang. Pass `default_timeout=None` to wait for as long as it takes.
"""


ClientT = TypeVar("ClientT")


class ClassInstrument(Protocol):
    """`Spillway.instrument`, which builds the limiter as well as using it."""

    def __call__(
        self,
        client: ClientT,
        *,
        provider: ProviderAdapter | str | None = None,
        scope: str | Scope | None = None,
        priority: int | Priority | None = None,
        rpm: float | None = None,
        rpd: float | None = None,
        tpm: float | None = None,
        input_tpm: float | None = None,
        output_tpm: float | None = None,
        concurrency: int | None = None,
        dimensions: Sequence[Dimension] = (),
        store: DuplexStore | None = None,
        clock: Clock | None = None,
        estimator: Estimator | None = None,
        default_timeout: float | None = DEFAULT_TIMEOUT_S,
    ) -> ClientT:
        """Instrument `client`, building a limiter for it."""
        ...


class BoundInstrument(Protocol):
    """`spillway.instrument`, which uses the limiter you already built."""

    def __call__(
        self,
        client: ClientT,
        *,
        provider: ProviderAdapter | str | None = None,
        scope: str | Scope | None = None,
        priority: int | Priority | None = None,
    ) -> ClientT:
        """Instrument `client` against this limiter."""
        ...


class _Instrument:
    """One verb that means something slightly different on the class.

    Reached through the class it builds the limiter; reached through an
    instance it uses that one, and the limiter building arguments are then not
    accepted because the limiter already exists.

    Two verbs for one concept would be worse than one verb doing two jobs, so
    the cost is paid here instead: a name that binds differently on a class and
    an instance is neither a method nor a classmethod, and keeping strict type
    checking honest about it takes a descriptor with an overloaded lookup.
    Everything either spelling then does is ordinary.
    """

    # ponytail: a hand written descriptor, because no decorator in the standard
    # library binds one name to two behaviours. Roughly fifteen lines and it
    # never needs to grow. If Python ever ships one, delete this.

    @overload
    def __get__(self, instance: None, owner: type[Spillway]) -> ClassInstrument: ...

    @overload
    def __get__(self, instance: Spillway, owner: type[Spillway]) -> BoundInstrument: ...

    def __get__(
        self,
        instance: Spillway | None,
        owner: type[Spillway],
    ) -> ClassInstrument | BoundInstrument:
        """Hand back whichever spelling was reached for."""
        if instance is None:
            return _instrument_and_build
        return functools.partial(_instrument_with, instance)


def _instrument_and_build(
    client: ClientT,
    *,
    provider: ProviderAdapter | str | None = None,
    scope: str | Scope | None = None,
    priority: int | Priority | None = None,
    rpm: float | None = None,
    rpd: float | None = None,
    tpm: float | None = None,
    input_tpm: float | None = None,
    output_tpm: float | None = None,
    concurrency: int | None = None,
    dimensions: Sequence[Dimension] = (),
    store: DuplexStore | None = None,
    clock: Clock | None = None,
    estimator: Estimator | None = None,
    default_timeout: float | None = DEFAULT_TIMEOUT_S,
) -> ClientT:
    """Build a limiter for this client's provider, and instrument it."""
    from spillway.integrations.detect import adapter_for
    from spillway.integrations.instrument import patch

    adapter = adapter_for(client) if provider is None else provider
    limiter = Spillway.for_provider(
        adapter,
        rpm=rpm,
        rpd=rpd,
        tpm=tpm,
        input_tpm=input_tpm,
        output_tpm=output_tpm,
        concurrency=concurrency,
        dimensions=dimensions,
        store=store,
        clock=clock,
        estimator=estimator,
        default_timeout=default_timeout,
    )
    if not limiter.dimensions:
        _warn_once_about_no_limits()
    return patch(client, limiter, provider=adapter, scope=scope, priority=priority)


def _instrument_with(
    limiter: Spillway,
    client: ClientT,
    *,
    provider: ProviderAdapter | str | None = None,
    scope: str | Scope | None = None,
    priority: int | Priority | None = None,
) -> ClientT:
    """Instrument this client against a limiter that already exists."""
    from spillway.integrations.instrument import patch

    return patch(
        client,
        limiter,
        provider=provider if provider is not None else limiter.provider,
        scope=scope,
        priority=priority,
    )


def _warn_once_about_no_limits() -> None:
    """Say, once, that nothing is being enforced and how to change that."""
    global _warned_about_no_limits
    if _warned_about_no_limits:
        return
    _warned_about_no_limits = True
    _log.warning(
        "This client is instrumented but no limits were named, so every call is "
        "admitted and nothing is enforced. That is a reasonable place to start: let it "
        "run, then read Spillway.of(client).snapshot() to see what your traffic "
        "actually does. To enforce a limit, name it, for example "
        "Spillway.instrument(client, rpm=1_000), using the figures from your "
        "provider's own limits page."
    )


def _adapter_for(provider: ProviderAdapter | str | None) -> ProviderAdapter | None:
    """Turn whatever was passed as a provider into an adapter, or nothing.

    Imported inside the function to keep the dependency running one way: the
    providers know about the core, the core knows only the protocol.
    """
    if provider is None or not isinstance(provider, str):
        return provider
    from spillway.providers import by_name

    return by_name(provider)


_SECONDS_PER_DAY = 86_400.0

NAMED_LIMITS = ("rpm", "rpd", "tpm", "input_tpm", "output_tpm", "concurrency")
"""The limits that can be named directly rather than built as dimensions.

The names providers publish under, so a figure copied off their page goes in
without translation. Anything else is a `Dimension` passed to `dimensions`.
"""


def dimensions_from(
    *,
    rpm: float | None = None,
    rpd: float | None = None,
    tpm: float | None = None,
    input_tpm: float | None = None,
    output_tpm: float | None = None,
    concurrency: int | None = None,
) -> tuple[Dimension, ...]:
    """Turn named limits into the dimensions that enforce them.

    Pass the limits your provider actually gives you and leave the rest alone.
    A provider metering input and output on one bucket gives you a `tpm`; one
    metering them apart gives you `input_tpm` and `output_tpm`.

    This library ships no limit figures of its own. Yours are on your provider's
    own limits page and are the only ones true for your account.

    Args:
        rpm: Requests per minute.
        rpd: Requests per day.
        tpm: Tokens per minute, counting input and output together.
        input_tpm: Input tokens per minute, when metered on their own.
        output_tpm: Output tokens per minute, when metered on their own.
        concurrency: How many requests may be in flight at once. Rarely
            published, and worth setting anyway.

    Example:
        >>> found = dimensions_from(rpm=1_000, input_tpm=2_000_000, output_tpm=400_000)
        >>> [d.name for d in found]
        ['rpm', 'input_tpm', 'output_tpm']
        >>> dimensions_from(tpm=150_000)[0].limit
        150000.0

        Naming nothing is valid, and means observe without limiting.

        >>> dimensions_from()
        ()
    """
    built: list[Dimension] = []
    for name, limit, window in (
        ("rpm", rpm, 60.0),
        ("rpd", rpd, _SECONDS_PER_DAY),
        ("tpm", tpm, 60.0),
        ("input_tpm", input_tpm, 60.0),
        ("output_tpm", output_tpm, 60.0),
    ):
        if limit is not None:
            built.append(Rate(name, limit=limit, window=window))
    if concurrency is not None:
        built.append(Concurrency("generations", limit=concurrency))
    return tuple(built)


def _combine(
    built: tuple[Dimension, ...],
    named: tuple[Dimension, ...],
) -> tuple[Dimension, ...]:
    """Join dimensions passed as objects with those named as limits.

    Raises:
        ConfigurationError: if the same limit arrives both ways.
    """
    clash = {d.name for d in built} & {d.name for d in named}
    if clash:
        name = sorted(clash)[0]
        message = (
            f"The {name!r} limit was given twice, once as a dimension and once as "
            f"{name}=. There is no honest answer when the two disagree. Keep whichever "
            f"you meant: the keyword for a plain limit, the dimension for one that needs "
            f"a window, a meter or anything else spelled out."
        )
        raise ConfigurationError(message)
    return built + named


@dataclass(frozen=True)
class Snapshot:
    """How full everything is right now, for one scope.

    Reserves nothing, so calling it on a timer cannot affect what is admitted.

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
    reports and never refuses anything.

    A request that finds no room waits for up to `default_timeout` seconds
    rather than failing. Waiters are served highest priority first and, within a
    priority, in arrival order. Pass `timeout=0` to be refused instead.

    Args:
        dimensions: The limits to enforce. Empty means enforce nothing.
        store: Where reservations are recorded. Defaults to an in memory store,
            which is correct within one process and not across several.
        clock: Where time comes from.
        scope: The scope used when a caller names none.
        estimator: How to predict what a request will cost. Defaults to
            reserving the output maximum the caller allowed, which is safe and
            expensive. See `for_provider` for the learning default.
        provider: Whose accounting rules to apply, as an adapter or the name of
            one that ships. Decides how a reservation is adjusted before it is
            taken and how a response is read at settlement. Without one,
            `lease.settle_from` has nothing to read a response with.
        default_timeout: How many seconds to wait for capacity when a caller
            names neither a timeout nor a deadline. Zero refuses rather than
            waits. None waits for as long as it takes.
        queue_capacity: How many requests may wait in each priority band.
        queue_full_policy: What a full band does with a new arrival. "reject"
            refuses it. "shed_lowest" drops the lowest priority waiter to make
            room, and refuses when the arrival is itself the lowest.
        rpm: Requests per minute, if your provider gives you one.
        rpd: Requests per day.
        tpm: Tokens per minute, counting input and output together.
        input_tpm: Input tokens per minute, when metered on their own.
        output_tpm: Output tokens per minute, when metered on their own.
        concurrency: How many requests may be in flight at once.

    The last six build the matching dimensions for you. This library ships no
    limit figures of its own: yours are the only ones true for your account.

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
        estimator: Estimator | None = None,
        provider: ProviderAdapter | str | None = None,
        default_timeout: float | None = DEFAULT_TIMEOUT_S,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
        queue_full_policy: QueueFullPolicy = "reject",
        rpm: float | None = None,
        rpd: float | None = None,
        tpm: float | None = None,
        input_tpm: float | None = None,
        output_tpm: float | None = None,
        concurrency: int | None = None,
    ) -> None:
        """Assemble a limiter. Every argument has a usable default.

        Raises:
            ConfigurationError: if `default_timeout` is negative, if
                `queue_capacity` is below one, if `queue_full_policy` is not
                one the queue knows, or if a limit is named twice, once
                directly and once as a dimension.
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
        named = dimensions_from(
            rpm=rpm,
            rpd=rpd,
            tpm=tpm,
            input_tpm=input_tpm,
            output_tpm=output_tpm,
            concurrency=concurrency,
        )
        self._dimensions = _combine(tuple(dimensions), named)
        self._store: DuplexStore = store if store is not None else MemoryStore(clock=self._clock)
        self._default_scope = Scope.of(scope)
        self._estimator: Estimator = estimator if estimator is not None else MaxTokensEstimator()
        self._provider = _adapter_for(provider)
        self._queue = WaitQueue(capacity=queue_capacity, policy=queue_full_policy)
        self._dispatcher = Dispatcher(limiter=self, queue=self._queue, clock=self._clock)

    @classmethod
    def for_provider(
        cls,
        provider: ProviderAdapter | str,
        *,
        rpm: float | None = None,
        rpd: float | None = None,
        tpm: float | None = None,
        input_tpm: float | None = None,
        output_tpm: float | None = None,
        concurrency: int | None = None,
        dimensions: Sequence[Dimension] = (),
        store: DuplexStore | None = None,
        clock: Clock | None = None,
        scope: str | Scope | None = None,
        estimator: Estimator | None = None,
        default_timeout: float | None = DEFAULT_TIMEOUT_S,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
        queue_full_policy: QueueFullPolicy = "reject",
    ) -> Spillway:
        """Build a limiter that knows how one provider counts.

        The same as `Spillway(provider=...)`, except that it defaults to the
        estimator that learns what each route really produces rather than the
        one that reserves whatever maximum the caller allowed.

        Safe from cold: below its sample threshold the learning estimator defers
        to the requested maximum, so a fresh process behaves like the
        conservative one and improves as it watches.

        Args:
            provider: An adapter, or the name of one that ships.
            rpm: Requests per minute, if your provider gives you one.
            rpd: Requests per day.
            tpm: Tokens per minute, counting input and output together.
            input_tpm: Input tokens per minute, when metered on their own.
            output_tpm: Output tokens per minute, when metered on their own.
            concurrency: How many requests may be in flight at once.
            dimensions: Any limit the named ones cannot express.
            store: Where reservations are recorded.
            clock: Where time comes from.
            scope: The scope used when a caller names none.
            estimator: Overrides the learning default.
            default_timeout: How long to wait for capacity.
            queue_capacity: How many requests may wait in each priority band.
            queue_full_policy: What a full band does with a new arrival.

        Raises:
            ConfigurationError: if the provider is not one that ships, naming
                the ones that do.

        Example:
            >>> limiter = Spillway.for_provider("anthropic", rpm=1_000)
            >>> limiter.provider.name
            'anthropic'
            >>> [dimension.name for dimension in limiter.dimensions]
            ['rpm']

            Naming no limits observes without limiting, which is the intended
            first step: watch real traffic, then set a limit you can defend.

            >>> Spillway.for_provider("openai").dimensions
            ()
        """
        return cls(
            dimensions=dimensions,
            store=store,
            clock=clock,
            scope=scope,
            estimator=estimator if estimator is not None else QuantileEstimator(),
            provider=provider,
            default_timeout=default_timeout,
            queue_capacity=queue_capacity,
            queue_full_policy=queue_full_policy,
            rpm=rpm,
            rpd=rpd,
            tpm=tpm,
            input_tpm=input_tpm,
            output_tpm=output_tpm,
            concurrency=concurrency,
        )

    def __repr__(self) -> str:
        """Show what is being enforced."""
        names = ", ".join(dimension.name for dimension in self._dimensions)
        provider = "" if self._provider is None else f", provider={self._provider.name!r}"
        return f"Spillway(dimensions=[{names}], scope={self._default_scope.key!r}{provider})"

    @property
    def dimensions(self) -> tuple[Dimension, ...]:
        """The limits being enforced."""
        return self._dimensions

    @property
    def provider(self) -> ProviderAdapter | None:
        """Whose accounting rules are being applied, if anyone's."""
        return self._provider

    instrument = _Instrument()
    """Return a copy of a client whose completion methods go through a limiter.

    Two lines where the client is built, and every call site is untouched.

        client = Spillway.instrument(AsyncAnthropic(), rpm=1_000)
        reply = await client.messages.create(...)

    Reached through the class it builds the limiter from the limits you name.
    Reached through an instance it uses the one you already built, which is
    what to do the moment one quota is shared by more than one client or more
    than one process.

        chat = spillway.instrument(AsyncAnthropic())
        batch = spillway.instrument(AsyncAnthropic())

    Naming no limits admits everything, records what the traffic really costs,
    and says so once. That is the intended first step: this library ships no
    limit figures, and the ones worth enforcing are the ones your own traffic
    turns out to need.
    """

    @staticmethod
    def of(client: object) -> Spillway:
        """Return the limiter behind an instrumented client.

        How a health check reaches `snapshot()` without the application
        threading a limiter around beside every client it holds.

        Raises:
            ConfigurationError: if this client is not instrumented.
        """
        from spillway.integrations.instrument import limiter_of

        return limiter_of(client)

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
        tags: Mapping[str, str] | None = None,
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
            tags: Whatever the estimator should route on, such as
                `{"task": "summarise"}`. Nothing here affects admission
                directly. Output length is far more predictable within one task
                than across all of them, so naming the task is worth doing.
            weight: Reserved for fair sharing, which does not exist yet.

        Returns:
            A context that reserves capacity when entered or acquired.

        Raises:
            ConfigurationError: if both `timeout` and `deadline` are given.

        `weight` is accepted and unused. It is in the signature now so the
        shape a caller writes against does not change when it starts working.
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
            tags=dict(tags) if tags else {},
            weight=weight,
        )

    def snapshot(self, scope: str | Scope | None = None) -> Snapshot:
        """Report how full every limit is, without reserving anything.

        Args:
            scope: Whose budget to report on. Defaults to the limiter's.

        Returns:
            A snapshot keyed by dimension name.

        Limits come from the dimensions rather than the store, so one never
        claimed against reports as empty out of its real limit, not out of zero.
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
        on_settle: Callable[[Cost], None] | None = None,
    ) -> Lease:
        """Ask the store for the whole batch once, and report what happened.

        One attempt, no waiting. Both the direct path and the dispatcher come
        through here, so a request that waited is admitted by the same code as
        one that did not, and neither knows whether there was a queue.

        Raises:
            AdmissionDenied: if any dimension has no room. Carries the binding
                dimension and how long until it would fit, which is what a
                waiter is then scheduled on.
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
            provider=self._provider,
            on_release=self._dispatcher.notify,
            on_settle=on_settle,
        )

    def _deadline_ms(
        self,
        *,
        timeout: float | None,
        deadline: float | None,
        now_ms: float,
    ) -> float | None:
        """When to stop waiting, on the limiter's clock.

        A deadline already passed, which includes a timeout of zero, means the
        caller asked not to wait, so the refusal reaches them unchanged rather
        than as a timeout.

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

    def _learn_from(self, context: RequestContext, reserved: Cost) -> Callable[[Cost], None]:
        """Build the callback that tells the estimator how wrong it was.

        Recorded whether or not the estimator produced this reservation. An
        explicit estimate still generated real output on that route, and
        skipping those would blind the estimator when a caller mixes the two.
        """

        def learn(actual: Cost) -> None:
            self._estimator.record(
                Observation(
                    context=context,
                    reserved=reserved,
                    actual=actual,
                    at_ms=self._clock.now_ms(),
                )
            )

        return learn

    async def _acquire(
        self,
        *,
        scope: Scope,
        priority: int,
        reserved: Cost,
        context: RequestContext,
        timeout: float | None = None,
        deadline: float | None = None,
    ) -> Lease:
        """Reserve `reserved` across every dimension, waiting if allowed to.

        Attempted directly first, reaching the queue only on a refusal, so the
        usual case where there is room pays nothing for the machinery below.

        Raises:
            AdmissionDenied: if there is no room and none arrives in time, or
                if the request is larger than a limit and so could never have
                room.
            AdmissionTimeout: if the wait ran out first.
            Shed: if the work is sheddable and its band is full.
        """
        started_ms = self._clock.now_ms()
        on_settle = self._learn_from(context, reserved)
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
                on_settle=on_settle,
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
                    on_settle=on_settle,
                )
            )

    async def _wait_for_room(self, waiter: Waiter) -> Lease:
        """Queue and wait until the dispatcher has an answer.

        A cancelled caller takes itself out of the queue on the way past. The
        dispatcher would notice eventually, but not before selecting it, and a
        waiter nobody is listening for holds its band's head against the rest.

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
        tags: Mapping[str, str],
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
        self._tags = tags
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
        context = self._context()
        return await self._limiter._acquire(
            scope=self._scope,
            priority=self._priority,
            reserved=self._reserved(context),
            context=context,
            timeout=self._timeout,
            deadline=self._deadline,
        )

    def _context(self) -> RequestContext:
        """What the estimator is told about this request."""
        return RequestContext(
            prompt=self._prompt,
            max_tokens=self._max_tokens,
            model=self._model,
            scope=self._scope,
            tags=self._tags,
        )

    def _reserved(self, context: RequestContext) -> Cost:
        """Work out what to reserve, from the caller's estimate or the limiter's.

        Once per acquisition, never once per dispatch attempt: a request that
        waited reserves what it asked for when it arrived, or its place in the
        queue was earned against a different request.

        The provider gets the last word, on an explicit estimate as readily as a
        predicted one. Reserving less than the provider itself charges means
        believing in headroom it does not agree exists, which produces rate
        limit responses nothing predicted.
        """
        estimate = self._estimate
        if estimate is None:
            estimate = self._limiter._estimator.estimate(context)
        predicted = Cost(
            input_tokens=estimate.input,
            output_tokens=estimate.output.quantile(estimate.quantile),
            requests=1,
        )
        provider = self._limiter._provider
        if provider is None:
            return predicted
        return provider.adjust(predicted, context)

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
    ) -> Literal[False]:
        """Give the capacity back, whichever way the block ended.

        Never suppresses. The annotation says `False` rather than `bool` so a
        type checker knows a `return` inside the block is reached.

        Three endings:

        Raised or cancelled, the reservation goes back whole. Cancellation
        arrives as an exception like any other and the release path awaits
        nothing, so it cannot be interrupted partway through.

        Succeeded and settled, nothing to do.

        Succeeded and did not settle, it settles at the full reserved amount and
        says so once. Nothing calibrates for a caller who never reports real
        costs, so every request keeps paying the estimate's full price.
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
