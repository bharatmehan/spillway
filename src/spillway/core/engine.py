"""The decision arithmetic, and nothing else.

Every function here is pure: the current time arrives as a parameter, new state
comes back as a return value, and nothing reads a clock, touches a store or
mutates an argument. That is what lets the synchronous and asynchronous entry
points share one implementation, and what lets the coordinated store's Lua
version be tested against this one.

The Lua port sets the house style. No comprehensions, no exceptions, no
operation that exists in Python and not in a small scripting language, and every
value a float, because that is the only numeric type the other side will have.

Rate limits use the generic cell rate algorithm. The whole state for a key is
one float, `tat_ms`, the theoretical arrival time, which is the moment the key
will have paid for everything charged to it so far. Two derived values recur
below and the caller computes both: `emission_interval_ms` is the time one unit
of cost buys, `window_ms / limit`, and `burst_ms` is how far ahead of now the
arrival time may run.

Gauges are the other kind of limit: a value held until the request that took it
finishes. The two are not interchangeable, and conflating them leaks concurrency
or double counts rate.
"""

from __future__ import annotations


def gcra_reserve(
    tat_ms: float,
    now_ms: float,
    cost: float,
    emission_interval_ms: float,
    burst_ms: float,
) -> tuple[bool, float, float]:
    """Charge `cost` against a rate key, if it fits.

    Pulls the arrival time up to now before charging, so a key idle for an hour
    gets one window of allowance rather than an hour of it.

    Returns:
        Whether it was granted, the arrival time to store, and how long until a
        refused charge would fit. A refusal returns the arrival time exactly as
        it went in, so a caller that stores the result unconditionally still
        mutates nothing. That is what makes an all or nothing batch of claims
        safe to evaluate one at a time.

    Example:
        A limit of two per second, so one unit of cost buys 500ms and a key may
        run 1000ms ahead. Two charges fit at once and the third does not.

        >>> tat = 0.0
        >>> granted, tat, retry = gcra_reserve(tat, 0.0, 1.0, 500.0, 1000.0)
        >>> granted, tat, retry
        (True, 500.0, 0.0)
        >>> granted, tat, retry = gcra_reserve(tat, 0.0, 1.0, 500.0, 1000.0)
        >>> granted, tat, retry
        (True, 1000.0, 0.0)
        >>> granted, tat, retry = gcra_reserve(tat, 0.0, 1.0, 500.0, 1000.0)
        >>> granted, tat, retry
        (False, 1000.0, 500.0)

        Waiting out the retry makes room, and no more than that.

        >>> gcra_reserve(tat, 500.0, 1.0, 500.0, 1000.0)
        (True, 1500.0, 0.0)
    """
    start_ms = tat_ms
    if now_ms > start_ms:
        start_ms = now_ms
    new_tat_ms = start_ms + cost * emission_interval_ms
    if new_tat_ms - burst_ms <= now_ms:
        return True, new_tat_ms, 0.0
    return False, tat_ms, new_tat_ms - burst_ms - now_ms


def gcra_credit(
    tat_ms: float,
    now_ms: float,
    amount: float,
    emission_interval_ms: float,
) -> float:
    """Return unused capacity to a rate key, by rewinding its arrival time.

    This is what makes reserving conservatively affordable: capacity held on an
    estimate and not used comes back within the request's own lifetime rather
    than when the window rolls.

    The rewind stops at the present moment, which keeps the stored number
    bounded. A key that credited back more than it charged would otherwise drift
    into the past for ever and lose float precision as it went.

    `amount` is never negative. An overrun goes through `gcra_debt` instead.

    Returns:
        The arrival time to store.

    Example:
        A key charged two units at a 500ms interval, then given one back.

        >>> gcra_credit(1000.0, 0.0, 1.0, 500.0)
        500.0

        Giving back more than was ever charged rewinds only as far as now.

        >>> gcra_credit(1000.0, 0.0, 5.0, 500.0)
        0.0
    """
    new_tat_ms = tat_ms - amount * emission_interval_ms
    if new_tat_ms < now_ms:
        new_tat_ms = now_ms
    return new_tat_ms


def gcra_debt(
    tat_ms: float,
    now_ms: float,
    amount: float,
    emission_interval_ms: float,
    window_ms: float,
    burst_ms: float,
) -> float:
    """Charge a rate key for capacity it used beyond what it reserved.

    An overrun cannot be refused, because the tokens are already spent. It
    becomes debt instead: the arrival time moves forward and the excess is
    repaid out of the following window.

    Clamped at one extra window, so a single pathological request cannot push
    the key far enough ahead to silence it for hours.

    Returns:
        The arrival time to store.

    Example:
        A modest overrun is simply carried forward.

        >>> gcra_debt(1000.0, 0.0, 1.0, 500.0, 1000.0, 1000.0)
        1500.0

        A ruinous one is capped at one window beyond the burst allowance.

        >>> gcra_debt(1000.0, 0.0, 100.0, 500.0, 1000.0, 1000.0)
        2000.0
    """
    new_tat_ms = tat_ms + amount * emission_interval_ms
    ceiling_ms = now_ms + burst_ms + window_ms
    if new_tat_ms > ceiling_ms:
        new_tat_ms = ceiling_ms
    return new_tat_ms


def gauge_reserve(held: float, cost: float, limit: float) -> tuple[bool, float]:
    """Take `cost` from a gauge, if it fits.

    Returns:
        Whether it was granted, and the value to store. A refusal returns the
        held value unchanged, for the same reason a refused rate charge does.

    Example:
        >>> gauge_reserve(63.0, 1.0, 64.0)
        (True, 64.0)
        >>> gauge_reserve(64.0, 1.0, 64.0)
        (False, 64.0)
    """
    new_held = held + cost
    if new_held > limit:
        return False, held
    return True, new_held


def gauge_release(held: float, amount: float) -> float:
    """Give `amount` back to a gauge, clamped at zero.

    A gauge below zero would admit more than its limit, and floating point error
    accumulated over millions of settlements is a real way to get there without
    any single mistake.

    Returns:
        The value to store.

    Example:
        >>> gauge_release(64.0, 1.0)
        63.0
        >>> gauge_release(1.0, 5.0)
        0.0
    """
    new_held = held - amount
    if new_held < 0.0:
        new_held = 0.0
    return new_held
