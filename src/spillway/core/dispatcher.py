"""The one task that decides who gets served next.

A single dispatcher per limiter owns the queue. It picks the best waiter, asks
the store for that waiter's capacity, and either hands over a lease or waits
until something could have changed.

The obvious alternative, every waiter retrying on its own timer, is easier to
write and wrong. Whichever waiter happens to wake first wins, so priority
becomes advisory rather than real, and a burst of waiters becomes a burst of
contending reservation attempts against the one thing they are all queued
behind.

Nothing here polls. Capacity becomes available in exactly two ways and the
dispatcher waits on both at once: a gauge is given back when a request settles,
which arrives as an event, and a rate window replenishes with the passage of
time, which is a sleep of exactly as long as the last refusal said it would
take. Missing either produces a hang that only shows up under load.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from spillway.core.clock import Clock
from spillway.core.errors import AdmissionDenied, AdmissionTimeout
from spillway.core.queue import Waiter, WaitQueue

if TYPE_CHECKING:  # pragma: no cover - imported for typing, and the limiter owns the dispatcher
    from spillway.core.spillway import Spillway

_log = logging.getLogger(__name__)

_warned_about_a_failure = False

# ponytail: a flat second between attempts once something has gone wrong, with
# no backoff and no giving up. Waiters reach their own deadlines and are told,
# which is the outcome that matters, and this only decides how much log a badly
# broken store produces on the way there. Something adaptive if a real store
# turns out to fail in bursts that a fixed interval handles badly.
FAILURE_BACKOFF_MS = 1_000.0
"""How long to wait after an unexpected failure before trying again."""


def _warn_once_about_a_failure() -> None:
    """Report the first unexpected failure in the dispatch loop, with its traceback.

    Once per process. The loop carries on afterwards, because a dispatcher that
    died would be a hang for every caller that ever queues, and a hang is the
    failure nobody can diagnose. Repeating the same traceback on every pass
    would only teach people to filter this message out.
    """
    global _warned_about_a_failure
    if _warned_about_a_failure:
        return
    _warned_about_a_failure = True
    _log.exception(
        "The dispatcher hit an unexpected failure while serving a waiter. It will keep "
        "going, so queued requests will reach their own timeouts rather than hanging, "
        "but nothing is being admitted while this keeps happening. This is a bug, in "
        "this library or in a store implementing its protocol."
    )


def _still_to_wait(waiter: Waiter, now_ms: float) -> float | None:
    """How long the binding limit would still need, as of now.

    A rate refusal says how long until the same charge would fit, and that
    answer shrinks one for one with the time that passes. Reporting the figure
    the refusal carried, seconds or minutes after it was made, would send a
    caller away for longer than they need to be away.
    """
    retry_after = waiter.refusal.retry_after
    if retry_after is None:
        return None
    elapsed = (now_ms - waiter.refused_at_ms) / 1000.0
    remaining = retry_after - elapsed
    if remaining < 0.0:
        return 0.0
    return remaining


def _timeout_message(waiter: Waiter, waited_ms: float, retry_after: float | None) -> str:
    """Say how long the wait was, what bound it, and what to do next."""
    binding = waiter.refusal.binding_dimension
    what = f" on {binding}" if binding is not None else ""
    if retry_after is None:
        return (
            f"Waited {waited_ms / 1000.0:.3g}s for capacity{what} and gave up. That capacity "
            f"frees when an in flight request finishes rather than on a timer, so a longer "
            f"wait may not help. Raise the limit, or send fewer requests at once."
        )
    return (
        f"Waited {waited_ms / 1000.0:.3g}s for capacity{what} and gave up. It would have fit "
        f"in another {retry_after:.3g}s. Wait longer, raise the limit, or send this request "
        f"at a higher priority."
    )


class Dispatcher:
    """Serves the waiting queue, one waiter at a time, until it is empty.

    Started by the first waiter to queue and finished when the last one leaves,
    so a limiter that never has to wait never has a background task at all.

    Args:
        limiter: Whose store is asked, and who builds the lease.
        queue: Who is waiting.
        clock: Where time comes from, including the waiting.

    Example:
        >>> import asyncio
        >>> from spillway.core.clock import FakeClock
        >>> from spillway.core.cost import Cost
        >>> from spillway.core.errors import AdmissionDenied
        >>> from spillway.core.queue import WaitQueue, Waiter
        >>> from spillway.core.scope import Scope
        >>> from spillway.core.spillway import Spillway
        >>> from spillway.dimensions.rate import Rate
        >>> from spillway.stores.base import Claim, ClaimKind
        >>> async def main() -> str:
        ...     clock = FakeClock()
        ...     limiter = Spillway(dimensions=[Rate("rpm", limit=60)], clock=clock)
        ...     queue = WaitQueue()
        ...     dispatcher = Dispatcher(limiter=limiter, queue=queue, clock=clock)
        ...     waiter = Waiter(
        ...         claims=(
        ...             Claim(
        ...                 "tenant:acme:rpm",
        ...                 ClaimKind.RATE,
        ...                 cost=1.0,
        ...                 limit=60.0,
        ...                 window_ms=60_000.0,
        ...             ),
        ...         ),
        ...         dimension_of_key={"tenant:acme:rpm": "rpm"},
        ...         scope=Scope("tenant:acme"),
        ...         priority=0,
        ...         reserved=Cost(),
        ...         deadline_ms=None,
        ...         queued_at_ms=0.0,
        ...         future=asyncio.get_running_loop().create_future(),
        ...         refusal=AdmissionDenied("no room on rpm"),
        ...     )
        ...     queue.push(waiter)
        ...     dispatcher.ensure_running()
        ...     lease = await waiter.future
        ...     return f"{lease.scope.key}, queue depth {queue.depth}"
        >>> asyncio.run(main())
        'tenant:acme, queue depth 0'
    """

    def __init__(self, *, limiter: Spillway, queue: WaitQueue, clock: Clock) -> None:
        """Serve `queue` on behalf of `limiter`, waiting on `clock`."""
        self._limiter = limiter
        self._queue = queue
        self._clock = clock
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._released = asyncio.Event()

    def __repr__(self) -> str:
        """Show whether it is running and how much work it has."""
        return f"Dispatcher(running={self.running}, depth={self._queue.depth})"

    @property
    def running(self) -> bool:
        """Whether a dispatch task exists right now."""
        return self._task is not None

    def ensure_running(self) -> None:
        """Start the dispatch task if it is not already going.

        Called after every push. Starting lazily is what keeps a limiter that
        never blocks free of any background task, which matters because a task
        nobody stops is a task that outlives the thing it was serving.
        """
        if self._task is None or self._task.done():
            self._loop = asyncio.get_running_loop()
            self._task = asyncio.ensure_future(self._run())

    def notify(self) -> None:
        """Say that capacity has gone back, so a waiter may now fit.

        Routed through the event loop rather than setting the event directly.
        A lease is settled synchronously and may be settled from a worker
        thread, and setting an event from the wrong thread schedules a callback
        the loop never learns about, which is a wakeup lost for good.
        """
        loop = self._loop
        if loop is None:
            return
        # A RuntimeError here means the loop closed between the check and the
        # call, in which case there is nobody left to wake.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._released.set)

    async def _run(self) -> None:
        """Serve waiters until there are none, then stop.

        The emptiness check, the clearing of the task and the return happen
        with no await between them, so a push cannot slip in and find itself
        with no dispatcher.

        Nothing short of cancellation stops it. A dispatcher that died on an
        unexpected error would be a hang for every caller that ever queues,
        and a hang is the failure nobody can diagnose from the outside.
        """
        try:
            while self._queue.depth:
                try:
                    await self._serve_next()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _warn_once_about_a_failure()
                    await self._clock.sleep(FAILURE_BACKOFF_MS)
        finally:
            self._task = None

    async def _serve_next(self) -> None:
        """One pass: give up on whoever is out of time, then try whoever is next."""
        self._give_up_on_the_overdue()
        waiter = self._queue.select()
        if waiter is None:
            return
        if waiter.future.done():
            # Cancelled while queued. The waiting side removes itself too, and
            # neither side can tell which of them got here first.
            self._queue.remove(waiter)
            return
        # Cleared before the attempt rather than after it, or a release that
        # lands while the attempt is in flight is lost and the next waiter
        # sleeps through capacity that is sitting there free.
        self._released.clear()
        try:
            lease = await self._limiter._attempt(
                claims=waiter.claims,
                dimension_of_key=waiter.dimension_of_key,
                scope=waiter.scope,
                priority=waiter.priority,
                reserved=waiter.reserved,
                waited_ms=self._clock.now_ms() - waiter.queued_at_ms,
                queue_position=waiter.position,
                on_settle=waiter.on_settle,
            )
        except AdmissionDenied as refusal:
            waiter.refusal = refusal
            waiter.refused_at_ms = self._clock.now_ms()
            await self._until_something_changes(refusal)
            return
        self._queue.remove(waiter)
        if waiter.future.done():
            # Cancelled while the reservation was in flight. The capacity is
            # held by a lease nobody will ever settle, so it goes back here.
            lease.abandon(reason="cancelled")
            return
        waiter.future.set_result(lease)

    def _give_up_on_the_overdue(self) -> None:
        """Fail every waiter whose deadline has passed, in every band.

        Every pass, not only for the waiter being served. A waiter sitting
        behind a head that cannot be admitted is owed its own deadline, and
        checking only the selected one is what makes it wait for ever.
        """
        now_ms = self._clock.now_ms()
        for waiter in self._queue.expire(now_ms):
            if waiter.future.done():
                continue
            retry_after = _still_to_wait(waiter, now_ms)
            waiter.future.set_exception(
                AdmissionTimeout(
                    _timeout_message(waiter, now_ms - waiter.queued_at_ms, retry_after),
                    retry_after=retry_after,
                    binding_dimension=waiter.refusal.binding_dimension,
                    explanation=waiter.refusal.explanation,
                )
            )

    async def _until_something_changes(self, refusal: AdmissionDenied) -> None:
        """Wait for the first of a release, a replenished window, or a deadline."""
        delay_ms = self._next_wake_ms(refusal)
        if delay_ms is not None and delay_ms <= 0.0:
            # Something is already due. Yield so this cannot become a spin, and
            # let the next pass collect it.
            await asyncio.sleep(0)
            return
        pending: set[asyncio.Task[None]] = {asyncio.ensure_future(self._released_wait())}
        if delay_ms is not None:
            pending.add(asyncio.ensure_future(self._clock.sleep(delay_ms)))
        try:
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in pending:
                task.cancel()

    async def _released_wait(self) -> None:
        """Wait for capacity to come back, as something with no result."""
        await self._released.wait()

    def _next_wake_ms(self, refusal: AdmissionDenied) -> float | None:
        """How long until a timer could change the answer, or None if none can.

        A rate window says exactly when the same charge would fit. A gauge says
        nothing, because it frees when a request finishes rather than when time
        passes, so a waiter blocked on one wakes on the release event or on its
        own deadline and on nothing else.
        """
        candidates: list[float] = []
        if refusal.retry_after is not None:
            candidates.append(refusal.retry_after * 1000.0)
        earliest_ms = self._queue.earliest_deadline_ms()
        if earliest_ms is not None:
            candidates.append(earliest_ms - self._clock.now_ms())
        if not candidates:
            return None
        return min(candidates)
