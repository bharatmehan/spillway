"""Time access for the whole library.

Every time reference goes through the `Clock` protocol. Nothing else in the
library reads a system clock directly.

This is not gold plating. Rate windows, lease expiry, feedback controllers and
fairness counters are all time dependent, and testing any of them against wall
clock time produces flaky tests that eventually get deleted. A clock that can be
advanced by hand makes the whole decision path deterministic.
"""

from __future__ import annotations

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
        >>> class CountingClock:
        ...     def __init__(self) -> None:
        ...         self.calls = 0
        ...
        ...     def now_ms(self) -> float:
        ...         self.calls += 1
        ...         return float(self.calls)
        >>> clock: Clock = CountingClock()
        >>> clock.now_ms()
        1.0
        >>> clock.now_ms()
        2.0
    """

    def now_ms(self) -> float:
        """Return the current time in milliseconds.

        Only differences between two readings are meaningful. The origin is
        arbitrary and may differ between implementations.
        """
        ...


class MonotonicClock:
    """The real clock, reading a monotonic source that never goes backwards.

    This is the default. A monotonic source is used rather than wall clock time
    so that a system clock adjustment cannot make a lease look expired, or a
    rate window look replenished, when neither has happened.

    Example:
        >>> clock = MonotonicClock()
        >>> clock.now_ms() <= clock.now_ms()
        True
    """

    __slots__ = ()

    def now_ms(self) -> float:
        """Return the current monotonic time in milliseconds."""
        return time.monotonic_ns() / _NS_PER_MS


class FakeClock:
    """A clock that only moves when you tell it to.

    Use this in tests and simulations. Because it never advances on its own, a
    test can assert the exact millisecond at which a rate window replenishes or
    a lease expires, rather than sleeping and hoping.

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
    """

    __slots__ = ("_now_ms",)

    def __init__(self, now_ms: float = 0.0) -> None:
        """Start the clock at `now_ms`, which defaults to zero."""
        self._now_ms = float(now_ms)

    def now_ms(self) -> float:
        """Return the current time, which changes only through `advance` or `set`."""
        return self._now_ms

    def advance(self, delta_ms: float) -> None:
        """Move the clock forward by `delta_ms` milliseconds.

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

    def set(self, now_ms: float) -> None:
        """Set the clock to an absolute value, including one in the past."""
        self._now_ms = float(now_ms)
