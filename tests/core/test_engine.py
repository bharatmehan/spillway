"""The decision arithmetic, including every boundary that later matters.

These are the functions a coordinated store has to reimplement in another
language, so their behaviour at the edges is a specification rather than an
accident.
"""

from spillway.core.engine import gcra_credit, gcra_debt, gcra_reserve

# A limit of two per second: one unit of cost buys 500ms, and a key may run one
# window ahead of now.
INTERVAL = 500.0
BURST = 1000.0
WINDOW = 1000.0


def test_a_charge_against_an_idle_key_is_granted():
    assert gcra_reserve(0.0, 0.0, 1.0, INTERVAL, BURST) == (True, 500.0, 0.0)


def test_exactly_the_limit_fits_at_once():
    # The boundary case. A key that has consumed its whole window in an instant
    # is at the limit, not over it, so the last unit must be granted.
    tat = 0.0
    for _ in range(2):
        granted, tat, _retry = gcra_reserve(tat, 0.0, 1.0, INTERVAL, BURST)
        assert granted
    assert tat == BURST


def test_one_unit_past_the_limit_is_refused():
    granted, _tat, retry_after = gcra_reserve(BURST, 0.0, 1.0, INTERVAL, BURST)
    assert not granted
    assert retry_after == INTERVAL


def test_a_refusal_returns_the_arrival_time_unchanged():
    # This is the lowest level expression of the all or nothing rule. A caller
    # evaluating a batch of claims stores results as it goes, so a refused claim
    # that reported a moved arrival time would consume capacity it did not get.
    granted, tat, _retry = gcra_reserve(BURST, 0.0, 5.0, INTERVAL, BURST)
    assert not granted
    assert tat == BURST


def test_waiting_out_the_retry_after_makes_exactly_enough_room():
    _granted, tat, retry_after = gcra_reserve(BURST, 0.0, 1.0, INTERVAL, BURST)
    granted, _tat, _retry = gcra_reserve(tat, retry_after, 1.0, INTERVAL, BURST)
    assert granted


def test_waiting_one_tick_less_than_the_retry_after_is_still_refused():
    _granted, tat, retry_after = gcra_reserve(BURST, 0.0, 1.0, INTERVAL, BURST)
    granted, _tat, _retry = gcra_reserve(tat, retry_after - 0.001, 1.0, INTERVAL, BURST)
    assert not granted


def test_a_zero_cost_charge_is_always_granted_and_changes_nothing():
    granted, tat, retry_after = gcra_reserve(0.0, 0.0, 0.0, INTERVAL, BURST)
    assert (granted, tat, retry_after) == (True, 0.0, 0.0)


def test_a_zero_cost_charge_is_granted_even_at_the_limit():
    granted, tat, _retry = gcra_reserve(BURST, 0.0, 0.0, INTERVAL, BURST)
    assert granted
    assert tat == BURST


def test_an_idle_key_does_not_accumulate_credit():
    # The arrival time is pulled forward to now before charging, so a key that
    # has been quiet for an hour gets one window of burst, not an hour of it.
    granted, tat, _retry = gcra_reserve(0.0, 3_600_000.0, 1.0, INTERVAL, BURST)
    assert granted
    assert tat == 3_600_000.0 + INTERVAL


def test_a_clock_that_has_not_advanced_still_makes_progress():
    # Every call at the same instant must move the arrival time, or a caller in
    # a tight loop would be admitted for ever.
    tat = 0.0
    seen = []
    for _ in range(2):
        _granted, tat, _retry = gcra_reserve(tat, 0.0, 1.0, INTERVAL, BURST)
        seen.append(tat)
    assert seen == [500.0, 1000.0]


def test_a_charge_larger_than_the_whole_window_is_refused():
    granted, _tat, retry_after = gcra_reserve(0.0, 0.0, 3.0, INTERVAL, BURST)
    assert not granted
    assert retry_after == 500.0


def test_a_fractional_cost_is_charged_proportionally():
    granted, tat, _retry = gcra_reserve(0.0, 0.0, 0.5, INTERVAL, BURST)
    assert granted
    assert tat == 250.0


def test_a_credit_rewinds_the_arrival_time_by_what_was_returned():
    assert gcra_credit(BURST, 0.0, 1.0, INTERVAL) == 500.0


def test_a_credit_never_rewinds_into_the_past():
    # Without this clamp an idle key would bank credit and then admit a burst
    # far larger than its limit, which is the failure the limiter exists to stop.
    assert gcra_credit(BURST, 0.0, 5.0, INTERVAL) == 0.0


def test_a_credit_larger_than_the_reservation_stops_at_now():
    assert gcra_credit(600.0, 400.0, 100.0, INTERVAL) == 400.0


def test_a_credit_of_nothing_changes_nothing():
    assert gcra_credit(BURST, 0.0, 0.0, INTERVAL) == BURST


def test_reserving_then_crediting_the_whole_amount_restores_the_key():
    # Reserve then release must leave the key exactly as it was found, which is
    # the property the whole settlement path depends on.
    granted, tat, _retry = gcra_reserve(300.0, 0.0, 1.0, INTERVAL, BURST)
    assert granted
    assert gcra_credit(tat, 0.0, 1.0, INTERVAL) == 300.0


def test_a_debt_pushes_the_arrival_time_further_forward():
    assert gcra_debt(BURST, 0.0, 1.0, INTERVAL, WINDOW, BURST) == 1500.0


def test_a_debt_is_clamped_at_one_extra_window():
    # One bad estimate must cost a scope a pause, not an outage.
    assert gcra_debt(BURST, 0.0, 100.0, INTERVAL, WINDOW, BURST) == BURST + WINDOW


def test_the_debt_clamp_is_measured_from_now_not_from_the_arrival_time():
    assert gcra_debt(BURST, 5_000.0, 100.0, INTERVAL, WINDOW, BURST) == 5_000.0 + BURST + WINDOW


def test_a_debt_of_nothing_changes_nothing():
    assert gcra_debt(BURST, 0.0, 0.0, INTERVAL, WINDOW, BURST) == BURST


def test_a_debt_exactly_at_the_clamp_is_not_reduced():
    assert gcra_debt(BURST, 0.0, 2.0, INTERVAL, WINDOW, BURST) == 2000.0


def test_a_key_in_debt_refuses_until_the_debt_is_paid():
    tat = gcra_debt(BURST, 0.0, 2.0, INTERVAL, WINDOW, BURST)
    granted, _tat, retry_after = gcra_reserve(tat, 0.0, 1.0, INTERVAL, BURST)
    assert not granted
    assert retry_after == 1500.0
