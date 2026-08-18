"""The decision arithmetic, and nothing else.

Every function here is pure. None of them reads a clock, touches a store, owns
state, or mutates an argument. The current time arrives as a parameter and new
state comes back as a return value. Three things depend on that:

- A synchronous caller and an asynchronous caller can share one implementation,
  so the two entry points are thin drivers rather than two engines with two sets
  of bugs.
- The whole decision path is testable without an event loop.
- A coordinated store has to run this same arithmetic inside a server side
  script, in another language, and a differential test can only assert the two
  agree if this side is a plain function over plain numbers.

That last point sets the house style for this module: no comprehensions, no
exceptions, no clever expressions, no operations that exist in Python and not in
a small scripting language. Every value is a float, because that is the only
numeric type the other implementation will have.

## Rate accounting

Rate limits use the generic cell rate algorithm. The entire state for one key is
a single float, the theoretical arrival time, which is the moment the key will
have fully paid for everything charged to it so far. It is O(1) in memory no
matter how much traffic passes, and it is smoother than a fixed window, which
admits a full limit at the end of one window and again at the start of the next.

Two derived values, both computed by the caller:

    emission_interval_ms = window_ms / limit    time one unit of cost buys
    burst_ms             = window_ms            how far ahead a key may run

## Gauge accounting

Gauges are the other kind of limit: a value currently held, released explicitly
when the request that took it finishes. Concurrency is one. They are genuinely
different from rate keys, which are never released and simply age out, and
conflating the two produces either leaked concurrency or double counted rate.
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

    Args:
        tat_ms: The key's stored theoretical arrival time.
        now_ms: The current time.
        cost: How many units to charge.
        emission_interval_ms: Time one unit of cost buys.
        burst_ms: How far ahead of now the arrival time may run.

    Returns:
        Whether it was granted, the arrival time to store, and how long until a
        refused charge would fit. On a refusal the arrival time comes back
        exactly as it went in, so a caller that stores the result unconditionally
        still mutates nothing. That is what makes an all or nothing batch of
        claims safe to evaluate one at a time.

    The first two lines are what bound a burst. Pulling the arrival time up to
    now before charging means a key that has been idle for an hour gets one
    window of allowance rather than an hour of it, which is the difference
    between a limiter and a counter that occasionally lets everything through.

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
    """Return unused capacity to a rate key.

    Crediting back is a rewind of the arrival time. This is what makes reserving
    conservatively affordable: capacity held on an estimate and not used is
    returned within the request's own lifetime, so it is available to the next
    caller rather than wasted until the window rolls.

    The rewind stops at the present moment. That floor is not what bounds a
    burst, since `gcra_reserve` pulls the arrival time up to now before charging
    anything and would ignore a value in the past anyway. It is here to keep the
    stored number bounded: a long lived key that credited back more than it
    charged would otherwise drift further into the past for ever, losing float
    precision as it went and misleading anything that read it directly.

    Args:
        tat_ms: The key's stored theoretical arrival time.
        now_ms: The current time.
        amount: How many units to give back. Never negative; an overrun goes
            through `gcra_debt` instead.
        emission_interval_ms: Time one unit of cost buys.

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

    An overrun cannot be refused, because the tokens are already spent. It is
    recorded as debt instead: the arrival time moves further forward, so the
    excess is repaid out of the following window automatically.

    The clamp bounds the debt at one extra window. Without it a single
    pathological request could push the arrival time so far ahead that the key
    admits nothing for hours, which turns one bad estimate into an outage.

    Args:
        tat_ms: The key's stored theoretical arrival time.
        now_ms: The current time.
        amount: How many units were used beyond the reservation.
        emission_interval_ms: Time one unit of cost buys.
        window_ms: The limit's window.
        burst_ms: How far ahead of now the arrival time may run.

    Returns:
        The arrival time to store.

    Example:
        A modest overrun is simply carried forward.

        >>> gcra_debt(1000.0, 0.0, 1.0, 500.0, 1000.0, 1000.0)
        1500.0

        A ruinous one is capped at one window beyond the burst allowance, so
        the key is silent for a while rather than for ever.

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

    Args:
        held: How much of the gauge is currently held.
        cost: How much to take.
        limit: The most that may be held at once.

    Returns:
        Whether it was granted, and the value to store. As with a rate charge, a
        refusal returns the held value unchanged, so a caller evaluating a batch
        of claims one at a time still mutates nothing when one of them refuses.

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
    """Give `amount` back to a gauge.

    Clamped at zero. A gauge below zero would admit more than its limit, and
    accumulated floating point error over millions of settlements is a real way
    to get there without any single mistake.

    Args:
        held: How much of the gauge is currently held.
        amount: How much to give back.

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
