"""The vocabulary a store speaks.

A store answers one question: may this request take all of this capacity right
now, and if not, what stopped it and for how long.

The types carry numbers and strings and nothing that only makes sense in Python,
because they cross into an implementation that may be a dictionary in this
process or a script on another machine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class ClaimKind(Enum):
    """Which of the two kinds of limit a claim is against.

    Part of the type rather than something a store infers. A rate claim is never
    given back, it ages out. A gauge claim is held until the request that took
    it settles or expires. Treating one as the other leaks concurrency in one
    direction and double counts rate in the other.

    Example:
        >>> ClaimKind.RATE.value
        'rate'
    """

    RATE = "rate"
    """Consumption over a rolling window, replenished by the passage of time."""

    GAUGE = "gauge"
    """A value currently held, given back explicitly when the request finishes."""


@dataclass(frozen=True)
class Claim:
    """A request for some capacity on one key.

    Attributes:
        key: What the capacity is drawn from, scope included, such as
            `"tenant:acme:input_tpm"`. Opaque to the store.
        kind: Whether this is consumed over a window or held.
        cost: How much to take.
        limit: The most this key may hold or consume per window.
        window_ms: The window length for a rate claim. Absent for a gauge,
            which has no window.

    Example:
        >>> rate = Claim("acme:rpm", ClaimKind.RATE, cost=1.0, limit=1000.0, window_ms=60_000.0)
        >>> rate.key, rate.cost, rate.window_ms
        ('acme:rpm', 1.0, 60000.0)
        >>> Claim("acme:generations", ClaimKind.GAUGE, cost=1.0, limit=64.0).window_ms is None
        True
    """

    key: str
    kind: ClaimKind
    cost: float
    limit: float
    window_ms: float | None = None

    def __post_init__(self) -> None:
        """Reject a claim no store could act on.

        Raises:
            ValueError: if the cost is negative, if the limit is negative or
                is not positive on a rate claim, or if the window is present
                on a gauge, absent on a rate claim, or not positive.
        """
        if self.cost < 0:
            message = f"Claim cost cannot be negative, got {self.cost} for key {self.key!r}."
            raise ValueError(message)
        if self.limit < 0:
            message = f"Claim limit cannot be negative, got {self.limit} for key {self.key!r}."
            raise ValueError(message)
        if self.kind is ClaimKind.RATE:
            if self.window_ms is None:
                message = (
                    f"A rate claim needs a window, and key {self.key!r} has none. "
                    f"Pass window_ms, or use ClaimKind.GAUGE if this is a held value."
                )
                raise ValueError(message)
            if self.window_ms <= 0:
                message = (
                    f"A rate window must be positive, got {self.window_ms} for key {self.key!r}."
                )
                raise ValueError(message)
            if self.limit <= 0:
                message = (
                    f"A rate claim needs a positive limit, got {self.limit} for key "
                    f"{self.key!r}. A rate of zero per window has no rate at all, so "
                    f"there is no interval to charge against."
                )
                raise ValueError(message)
        elif self.window_ms is not None:
            message = (
                f"A gauge claim has no window, but key {self.key!r} was given "
                f"window_ms={self.window_ms}. Drop it, or use ClaimKind.RATE."
            )
            raise ValueError(message)


@dataclass(frozen=True)
class Delta:
    """A correction to apply to one key once the real cost is known.

    Positive means capacity was reserved and not used, so it goes back. Negative
    means more was used than reserved, so it is owed. Both happen constantly,
    because output length is predicted rather than known.

    Example:
        >>> Delta("acme:output_tpm", ClaimKind.RATE, amount=765.0).amount
        765.0
        >>> Delta("acme:output_tpm", ClaimKind.RATE, amount=-150.0).amount
        -150.0
    """

    key: str
    kind: ClaimKind
    amount: float


@dataclass(frozen=True)
class Utilisation:
    """How full one key is.

    Reported for every key a reservation touched, on refusal as well as on
    success, so that explaining a decision never costs a second round trip.

    Example:
        >>> used = Utilisation(used=412.0, limit=1000.0)
        >>> round(used.headroom, 3)
        0.588
        >>> Utilisation(used=64.0, limit=64.0).headroom
        0.0
    """

    used: float
    limit: float

    @property
    def headroom(self) -> float:
        """The fraction of this key still free, from 1.0 down to 0.0.

        A limit of zero reports no headroom rather than dividing by zero.
        """
        if self.limit <= 0:
            return 0.0
        free = (self.limit - self.used) / self.limit
        if free < 0.0:
            return 0.0
        return free


@dataclass(frozen=True)
class ReserveResult:
    """What a store says when asked for capacity.

    A refusal names the key that ran out and how long until it would fit, so it
    can be explained and a waiter can be scheduled on it.

    Attributes:
        granted: Whether all the claims were applied.
        lease_id: Identifies the reservation, for settling or releasing it
            later. Present exactly when granted.
        binding_key: The first key that refused. Present exactly when refused.
        retry_after_ms: How long until the binding key could grant the same
            claim, or None when waiting would not help.
        utilisation: Every key the reservation touched, in both outcomes.

    Example:
        >>> refusal = ReserveResult.refused("acme:rpm", retry_after_ms=500.0)
        >>> refusal.granted, refusal.binding_key, refusal.retry_after_ms
        (False, 'acme:rpm', 500.0)
        >>> ReserveResult.granted_as("lease-1").granted
        True
    """

    granted: bool
    lease_id: str | None = None
    binding_key: str | None = None
    retry_after_ms: float | None = None
    utilisation: Mapping[str, Utilisation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a result that says two contradictory things.

        A store is an extension point, so a third party implementation returning
        a shape the library would misread fails here rather than three frames
        later.

        Raises:
            ValueError: if the result is internally inconsistent.
        """
        if self.granted:
            if self.lease_id is None:
                message = "A granted reservation must carry a lease_id to settle it with."
                raise ValueError(message)
            if self.binding_key is not None:
                message = (
                    f"A granted reservation cannot have a binding key, got {self.binding_key!r}."
                )
                raise ValueError(message)
        else:
            if self.lease_id is not None:
                message = f"A refused reservation cannot have a lease_id, got {self.lease_id!r}."
                raise ValueError(message)
            if self.binding_key is None:
                message = (
                    "A refused reservation must name the key that refused. Without it "
                    "the refusal cannot be explained and a waiter cannot be scheduled."
                )
                raise ValueError(message)

    @classmethod
    def granted_as(
        cls,
        lease_id: str,
        utilisation: Mapping[str, Utilisation] | None = None,
    ) -> ReserveResult:
        """Build the result of a successful reservation."""
        return cls(granted=True, lease_id=lease_id, utilisation=utilisation or {})

    @classmethod
    def refused(
        cls,
        binding_key: str,
        retry_after_ms: float | None = None,
        utilisation: Mapping[str, Utilisation] | None = None,
    ) -> ReserveResult:
        """Build the result of a refused reservation."""
        return cls(
            granted=False,
            binding_key=binding_key,
            retry_after_ms=retry_after_ms,
            utilisation=utilisation or {},
        )


class Store(Protocol):
    """Where reservations are recorded, asked for capacity one batch at a time.

    A store is never asked about one key. A request admitted against two limits
    and refused by the third would leave the first two wrongly consumed, so the
    whole set goes in together and either all of it applies or none does.

    Implement this to coordinate through something this library does not ship.
    The hard part is not the interface, it is the atomicity: `reserve` must be
    indivisible with respect to every other caller.

    Example:
        A store that grants everything, which is what a limiter with no limits
        configured effectively has.

        >>> import asyncio
        >>> class Unlimited:
        ...     async def reserve(self, claims, *, ttl_ms, scope, priority):
        ...         return ReserveResult.granted_as("lease-1")
        ...
        ...     async def settle(self, lease_id, deltas):
        ...         return None
        ...
        ...     async def release(self, lease_id):
        ...         return None
        ...
        ...     async def snapshot(self, keys):
        ...         return {}
        >>> store: Store = Unlimited()
        >>> asyncio.run(store.reserve([], ttl_ms=60_000.0, scope="acme", priority=0)).granted
        True
    """

    async def reserve(
        self,
        claims: Sequence[Claim],
        *,
        ttl_ms: float,
        scope: str,
        priority: int,
    ) -> ReserveResult:
        """Apply every claim, or none of them.

        Args:
            claims: The complete set of claims for one admission.
            ttl_ms: How long the reservation may go unsettled before its
                capacity is reclaimed. A process that dies mid request must not
                hold a gauge for ever.
            scope: Which caller this is for. Used for lease bookkeeping; the
                claims already carry scoped keys.
            priority: How urgent the request is.

        Returns:
            A granted result carrying a lease identifier, or a refusal naming
            the key that bound and how long until it would not have. Either way,
            utilisation for every key touched, so explaining the decision costs
            no second round trip.
        """
        ...

    async def settle(self, lease_id: str, deltas: Sequence[Delta]) -> None:
        """Apply the corrections for a finished request and end its lease.

        Gauges held by the lease are given back. Rate keys are credited or put
        into debt according to the sign of each delta.
        """
        ...

    async def release(self, lease_id: str) -> None:
        """End a lease and return its whole reservation, correcting nothing.

        For a request that never ran: it raised, or it was cancelled. Nothing
        was consumed, so nothing is reconciled.
        """
        ...

    async def snapshot(self, keys: Sequence[str]) -> Mapping[str, Utilisation]:
        """Report how full each key is, without reserving anything.

        Cheap and safe to call from a health check.
        """
        ...


class SyncStore(Protocol):
    """The same four operations, for callers with no event loop.

    A store may implement this, the asynchronous protocol, or both. Kept
    separate because the implementations genuinely differ: one talking over a
    network has real waiting to do, one keeping a dictionary in this process
    does not. The synchronous facade itself arrives in a later release.

    The names carry a suffix so one class can implement both protocols without
    either shadowing the other.

    Example:
        >>> class Unlimited:
        ...     def reserve_sync(self, claims, *, ttl_ms, scope, priority):
        ...         return ReserveResult.granted_as("lease-1")
        ...
        ...     def settle_sync(self, lease_id, deltas):
        ...         return None
        ...
        ...     def release_sync(self, lease_id):
        ...         return None
        ...
        ...     def snapshot_sync(self, keys):
        ...         return {}
        >>> store: SyncStore = Unlimited()
        >>> store.reserve_sync([], ttl_ms=60_000.0, scope="acme", priority=0).granted
        True
    """

    def reserve_sync(
        self,
        claims: Sequence[Claim],
        *,
        ttl_ms: float,
        scope: str,
        priority: int,
    ) -> ReserveResult:
        """Apply every claim, or none of them. See `Store.reserve`."""
        ...

    def settle_sync(self, lease_id: str, deltas: Sequence[Delta]) -> None:
        """Apply corrections and end the lease. See `Store.settle`."""
        ...

    def release_sync(self, lease_id: str) -> None:
        """Return the whole reservation and end the lease. See `Store.release`."""
        ...

    def snapshot_sync(self, keys: Sequence[str]) -> Mapping[str, Utilisation]:
        """Report how full each key is. See `Store.snapshot`."""
        ...


class DuplexStore(Store, SyncStore, Protocol):
    """A store that serves both facades, implementing both protocols.

    Every store this library ships is one. The distinction exists because a
    third party store may implement only the half it can support.

    Example:
        >>> from spillway.stores.memory import MemoryStore
        >>> store: DuplexStore = MemoryStore()
        >>> store.reserve_sync([], ttl_ms=60_000.0, scope="acme", priority=0).granted
        True
    """
