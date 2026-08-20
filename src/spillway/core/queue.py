"""Who is waiting for capacity, and who goes next.

A queue exists because refusing is not the only honest answer. Capacity that
is full now is usually free shortly, and a caller who said they can wait would
rather wait than handle an error.

What the queue decides is order, and order is the whole of the policy at this
stage: the highest priority band that has anyone in it, and within a band the
one who arrived first.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from spillway.core.cost import Cost
from spillway.core.errors import AdmissionDenied, ConfigurationError
from spillway.core.lease import Lease
from spillway.core.scope import Scope
from spillway.stores.base import Claim

DEFAULT_QUEUE_CAPACITY = 10_000
"""How many waiters one priority band holds before it refuses new arrivals.

Large enough that reaching it means something is wrong rather than busy, and
small enough that the backlog cannot quietly become the reason the process runs
out of memory.
"""


def _full_message(priority: int, capacity: int) -> str:
    """Say that the backlog is the problem, not the limit."""
    return (
        f"The priority {priority} queue is full at {capacity:,} waiters, so this request "
        f"was refused rather than making the backlog longer. Capacity is per band, so "
        f"other priorities are unaffected. Raise queue_capacity if the backlog is expected, "
        f"raise the limits if it is not, or send this work at a negative priority so it is "
        f"shed instead of queued."
    )


@dataclass(eq=False)
class Waiter:
    """One request queued for capacity that was not there when it asked.

    Compared by identity rather than by value, because two callers asking for
    the same thing at the same instant are two waiters and removing one must
    not remove the other.

    Attributes:
        claims: What this request needs, built once and reused on every
            attempt.
        dimension_of_key: Which dimension each claim's key belongs to, for
            turning a refusal back into something worth showing a caller.
        scope: Whose budget this draws on.
        priority: Which band it waits in. Higher goes first.
        reserved: What the lease will hold once it is granted.
        deadline_ms: When to give up, on the limiter's clock. None waits for
            ever, which is only reachable when a caller asks for it explicitly.
        queued_at_ms: When the caller first asked, so the wait can be reported.
        future: Where the lease, or the exception, is delivered.
        refusal: The refusal that put this waiter here, replaced by each later
            attempt. A timeout is reported with it, so it names the dimension
            that bound rather than merely saying time ran out.
        sequence: Arrival order across the whole queue, assigned on push.
        position: How many were ahead of it in its own band on arrival.

    Example:
        >>> import asyncio
        >>> async def main() -> int:
        ...     waiter = Waiter(
        ...         claims=(),
        ...         dimension_of_key={},
        ...         scope=Scope("tenant:acme"),
        ...         priority=0,
        ...         reserved=Cost(),
        ...         deadline_ms=None,
        ...         queued_at_ms=0.0,
        ...         future=asyncio.get_running_loop().create_future(),
        ...         refusal=AdmissionDenied("no room on rpm"),
        ...     )
        ...     return waiter.position
        >>> asyncio.run(main())
        0
    """

    claims: tuple[Claim, ...]
    dimension_of_key: Mapping[str, str]
    scope: Scope
    priority: int
    reserved: Cost
    deadline_ms: float | None
    queued_at_ms: float
    future: asyncio.Future[Lease] = field(repr=False)
    refusal: AdmissionDenied
    sequence: int = 0
    position: int = 0


class WaitQueue:
    """The waiters, in the order they will be served.

    One queue per limiter. Strictly by priority band, and first in first out
    within a band, which is the whole of the ordering policy for now.

    Capacity is per band rather than shared. A flood of batch work must not
    consume the slots an interactive request needs, and that is a real failure
    mode rather than a hypothetical one. A queue with no bound at all is a
    memory leak with extra steps: it turns a rate limit problem into an out of
    memory problem, which is strictly worse.

    Args:
        capacity: The most waiters one band may hold.

    Example:
        >>> import asyncio
        >>> def waiting(priority: int) -> Waiter:
        ...     return Waiter(
        ...         claims=(),
        ...         dimension_of_key={},
        ...         scope=Scope("tenant:acme"),
        ...         priority=priority,
        ...         reserved=Cost(),
        ...         deadline_ms=None,
        ...         queued_at_ms=0.0,
        ...         future=asyncio.get_running_loop().create_future(),
        ...         refusal=AdmissionDenied("no room"),
        ...     )
        >>> async def main() -> list[int]:
        ...     queue = WaitQueue()
        ...     for priority in (0, 100, 0):
        ...         queue.push(waiting(priority))
        ...     served = []
        ...     while queue.depth:
        ...         waiter = queue.select()
        ...         assert waiter is not None
        ...         served.append(waiter.priority)
        ...         queue.remove(waiter)
        ...     return served
        >>> asyncio.run(main())
        [100, 0, 0]
    """

    def __init__(self, *, capacity: int = DEFAULT_QUEUE_CAPACITY) -> None:
        """Start with nobody waiting, and room for `capacity` in each band.

        Raises:
            ConfigurationError: if `capacity` is not at least one, which would
                mean nothing could ever queue.
        """
        if capacity < 1:
            message = (
                f"A queue band needs room for at least one waiter, got capacity={capacity}. "
                f"A capacity of zero refuses everything that cannot be admitted immediately, "
                f"which is what leaving out the timeout already does."
            )
            raise ConfigurationError(message)
        self._capacity = capacity
        # One band per distinct priority, created on first use and dropped once
        # it empties, so the bands present are exactly the bands with someone
        # in them.
        self._bands: dict[int, deque[Waiter]] = {}
        self._sequence = itertools.count()

    def __repr__(self) -> str:
        """Show the depth of each band, highest priority first."""
        ordered = sorted(self._bands.items(), reverse=True)
        bands = ", ".join(f"{priority}: {len(band)}" for priority, band in ordered)
        return f"WaitQueue({{{bands}}})"

    @property
    def depth(self) -> int:
        """How many waiters there are in total."""
        return sum(len(band) for band in self._bands.values())

    def depths(self) -> Mapping[int, int]:
        """How many waiters there are in each band that has any."""
        return {priority: len(band) for priority, band in self._bands.items()}

    def push(self, waiter: Waiter) -> None:
        """Add a waiter to the back of its own band.

        Records the arrival order and how many were already ahead of it, which
        is the number worth reporting later: at selection it is always zero,
        because selection takes the head.

        Raises:
            AdmissionDenied: if this waiter's band is full.
        """
        band = self._bands.get(waiter.priority)
        if band is not None and len(band) >= self._capacity:
            raise AdmissionDenied(
                _full_message(waiter.priority, self._capacity),
                binding_dimension=waiter.refusal.binding_dimension,
                explanation=waiter.refusal.explanation,
            )
        if band is None:
            band = self._bands.setdefault(waiter.priority, deque())
        waiter.sequence = next(self._sequence)
        waiter.position = len(band)
        band.append(waiter)

    def select(self) -> Waiter | None:
        """Return whoever should be served next, without removing them.

        The head of the highest band that has anyone in it. Stage ten replaces
        the body of this method and nothing else, so the choice of who goes
        next stays in one place.

        Returns:
            The next waiter, or None if nobody is waiting.
        """
        # ponytail: a scan over the distinct priorities present, which is a
        # handful for every caller who uses the named conventions or anything
        # like them. A heap keyed on the band, if someone ever arrives with
        # thousands of distinct priority values.
        if not self._bands:
            return None
        return self._bands[max(self._bands)][0]

    def remove(self, waiter: Waiter) -> None:
        """Take a waiter out, wherever it is.

        Does nothing if it is not there. Both sides remove: the dispatcher when
        it delivers a lease, and the waiting caller when it is cancelled, and
        neither can tell whether the other got there first.
        """
        band = self._bands.get(waiter.priority)
        if band is None:
            return
        try:
            band.remove(waiter)
        except ValueError:
            return
        if not band:
            del self._bands[waiter.priority]

    def expire(self, now_ms: float) -> list[Waiter]:
        """Remove and return every waiter whose deadline has passed.

        Every band, not just the one being served. A waiter behind a head that
        cannot be admitted is still owed its own deadline, and checking only
        the selected waiter is what makes it wait for ever instead.
        """
        due: list[Waiter] = []
        for priority in list(self._bands):
            band = self._bands[priority]
            keeping = deque(w for w in band if w.deadline_ms is None or w.deadline_ms > now_ms)
            if len(keeping) == len(band):
                continue
            due.extend(w for w in band if w.deadline_ms is not None and w.deadline_ms <= now_ms)
            if keeping:
                self._bands[priority] = keeping
            else:
                del self._bands[priority]
        return due

    def earliest_deadline_ms(self) -> float | None:
        """When the first waiter gives up, or None if none of them ever will."""
        deadlines = [w.deadline_ms for band in self._bands.values() for w in band]
        present = [deadline for deadline in deadlines if deadline is not None]
        if not present:
            return None
        return min(present)
