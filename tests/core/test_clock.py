"""The clock protocol and its two implementations."""

import time

import pytest

from spillway.core.clock import Clock, FakeClock, MonotonicClock


def test_monotonic_clock_never_goes_backwards():
    clock = MonotonicClock()
    readings = [clock.now_ms() for _ in range(100)]
    assert readings == sorted(readings)


def test_monotonic_clock_reports_milliseconds():
    # A nanosecond source divided by a million. If the divisor were wrong every
    # window and every expiry in the library would be off by a factor of 1000.
    clock = MonotonicClock()
    before = time.monotonic_ns() / 1_000_000
    reading = clock.now_ms()
    after = time.monotonic_ns() / 1_000_000
    assert before <= reading <= after


def test_fake_clock_starts_at_zero_and_stays_there():
    clock = FakeClock()
    assert clock.now_ms() == 0.0
    assert clock.now_ms() == 0.0


def test_fake_clock_starts_where_it_is_told():
    assert FakeClock(1234.5).now_ms() == 1234.5


def test_fake_clock_advances_by_exactly_the_delta():
    clock = FakeClock()
    clock.advance(1_500)
    assert clock.now_ms() == 1500.0
    clock.advance(0.5)
    assert clock.now_ms() == 1500.5


def test_fake_clock_refuses_to_advance_backwards():
    clock = FakeClock(100)
    with pytest.raises(ValueError, match="must not be negative"):
        clock.advance(-1)
    assert clock.now_ms() == 100.0


def test_fake_clock_can_be_set_backwards_deliberately():
    clock = FakeClock(100)
    clock.set(0)
    assert clock.now_ms() == 0.0


def test_both_implementations_satisfy_the_protocol():
    clocks: list[Clock] = [MonotonicClock(), FakeClock()]
    for clock in clocks:
        assert isinstance(clock.now_ms(), float)


async def test_the_real_clock_actually_waits():
    # The delay is in milliseconds and the standard library sleep takes
    # seconds. A missing division by a thousand makes every wait in the
    # dispatcher a thousand times too long, which looks exactly like a hang.
    clock = MonotonicClock()
    before = clock.now_ms()
    await clock.sleep(20)
    assert clock.now_ms() - before >= 15.0
