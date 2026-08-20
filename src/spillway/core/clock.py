"""Time access for the whole library.

Every time reference goes through the `Clock` protocol. Nothing else in the
library reads a system clock directly.

This is not gold plating. Rate windows, lease expiry, feedback controllers and
fairness counters are all time dependent, and testing any of them against wall
clock time produces flaky tests that eventually get deleted. A clock that can be
advanced by hand makes the whole decision path deterministic.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from typing import Protocol

_NS_PER_MS = 1_000_000.0


class Clock(Protocol):
    """A source of monotonic time, in milliseconds.

    Milliseconds because every limit in this library is expressed per second,
    per minute or per day, and a float millisecond is precise enough for all of
    them while staying readable in an explanation.

    Implement this to drive the library from your own time source.

    Example:
        >>> import asyncio
        >>> class CountingClock:
        ...     def __init__(self) -> None:
        ...         self.calls = 0
        ...
        ...     def now_ms(self) -> float:
        ...         self.calls += 1
        ...         return float(self.calls)
        ...
        ...     async def sleep(self, delay_ms: float) -> None:
        ...         self.calls += int(delay_ms)
        >>> clock: Clock = CountingClock()
        >>> clock.now_ms()
        1.0
        >>> clock.now_ms()
        2.0
        >>> asyncio.run(clock.sleep(10))
        >>> clock.now_ms()
        13.0
    """

    def now_ms(self) -> float:
        """Return the current time in milliseconds.

        Only differences between two readings are meaningful. The origin is
        arbitrary and may differ between implementations.
        """
        ...

    async def sleep(self, delay_ms: float) -> None:
        """Wait for `delay_ms` milliseconds of this clock's time.

        On the real clock this is an ordinary asynchronous sleep. On a clock
        driven by hand it returns only when that clock is advanced past the
        wake time, which is what lets a test run ten minutes of waiting in a
        millisecond of real time.

        A delay of zero or less returns without waiting, though it may still
        yield control.
        """
        ...


class MonotonicClock:
    """The real clock, reading a monotonic source that never goes backwards.

    This is the default. A monotonic source is used rather than wall clock time
    so that a system clock adjustment cannot make a lease look expired, or a
    rate window look replenished, when neither has happened.

    Example:
        >>> import asyncio
        >>> clock = MonotonicClock()
        >>> clock.now_ms() <= clock.now_ms()
        True
        >>> asyncio.run(clock.sleep(1))
    """

    __slots__ = ()

    def now_ms(self) -> float:
        """Return the current monotonic time in milliseconds."""
        return time.monotonic_ns() / _NS_PER_MS

    async def sleep(self, delay_ms: float) -> None:
        """Wait for `delay_ms` milliseconds, giving the event loop up in the meantime."""
        await asyncio.sleep(delay_ms / 1000.0)


class FakeClock:
    """A clock that only moves when you tell it to.

    Use this in tests and simulations. Because it never advances on its own, a
    test can assert the exact millisecond at which a rate window replenishes or
    a lease expires, rather than sleeping and hoping.

    Sleeping on it does not sleep at all. A sleeper is recorded at its wake
    time and released when the clock is advanced past it, so a whole afternoon
    of waiting costs a millisecond of real time and produces the same sequence
    of events every run.

    Example:
        >>> clock = FakeClock()
        >>> clock.now_ms()
        0.0
        >>> clock.advance(1_500)
        >>> clock.now_ms()
        1500.0
        >>> clock.set(0)
        >>> clock.now_ms()
        0.0

        A sleeper waits until the clock is moved past its wake time.

        >>> import asyncio
        >>> async def wait_then_report(clock: FakeClock) -> float:
        ...     await clock.sleep(500)
        ...     return clock.now_ms()
        >>> async def main() -> float:
        ...     clock = FakeClock()
        ...     waiting = asyncio.ensure_future(wait_then_report(clock))
        ...     await asyncio.sleep(0)
        ...     clock.advance(500)
        ...     return await waiting
        >>> asyncio.run(main())
        500.0
    """

    __slots__ = ("_now_ms", "_sequence", "_sleepers")

    def __init__(self, now_ms: float = 0.0) -> None:
        """Start the clock at `now_ms`, which defaults to zero."""
        self._now_ms = float(now_ms)
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        # Sequence numbers keep the heap ordering total, so two sleepers due at
        # the same instant wake in the order they went to sleep instead of the
        # comparison falling through to the futures, which do not compare.
        self._sequence = itertools.count()

    def now_ms(self) -> float:
        """Return the current time, which changes only through `advance` or `set`."""
        return self._now_ms

    async def sleep(self, delay_ms: float) -> None:
        """Wait until this clock has been advanced by `delay_ms` milliseconds.

        Nothing happens on its own. If the clock is never advanced past the
        wake time, this waits for ever, which is the point: a test that forgets
        to advance fails rather than passing on a real sleep nobody noticed.
        """
        if delay_ms <= 0:
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._sleepers, (self._now_ms + delay_ms, next(self._sequence), future))
        await future

    @property
    def sleeping(self) -> int:
        """How many sleepers are waiting for this clock to move."""
        return sum(1 for _wake_at_ms, _sequence, future in self._sleepers if not future.done())

    def advance(self, delta_ms: float) -> None:
        """Move the clock forward by `delta_ms` milliseconds, releasing anything due.

        Raises:
            ValueError: if `delta_ms` is negative. A monotonic clock cannot go
                backwards, and a fake one that could would hide bugs the real
                one would expose. Use `set` if you deliberately want to rewind.
        """
        if delta_ms < 0:
            message = (
                f"advance() moves a monotonic clock forward, so delta_ms must not be "
                f"negative, got {delta_ms}. Use set() if you deliberately want to rewind."
            )
            raise ValueError(message)
        self._now_ms += float(delta_ms)
        self._release()

    def set(self, now_ms: float) -> None:
        """Set the clock to an absolute value, including one in the past."""
        self._now_ms = float(now_ms)
        self._release()

    def _release(self) -> None:
        """Wake every sleeper whose time has come, earliest first.

        The clock is already at its new value when this runs, rather than being
        stepped to each wake time in turn. A future that is resolved does not
        resume its coroutine until the loop next runs, so every sleeper woken by
        one advance would read the same time either way. A large advance is
        therefore the same thing as the process being descheduled for that long,
        which is a thing that really happens and is worth letting tests see.
        """
        while self._sleepers and self._sleepers[0][0] <= self._now_ms:
            _wake_at_ms, _sequence, future = heapq.heappop(self._sleepers)
            # Already done means the waiting task was cancelled while asleep.
            if not future.done():
                future.set_result(None)
