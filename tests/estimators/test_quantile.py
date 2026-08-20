"""The estimator that learns what a route actually produces."""

import pytest

from spillway.core.cost import Cost, Distribution
from spillway.core.errors import ConfigurationError
from spillway.estimators.base import Observation, RequestContext
from spillway.estimators.quantile import DEFAULT_HISTORY, QuantileEstimator


def observe(context, actual, reserved=4_096, at_ms=0.0):
    return Observation(
        context=context,
        reserved=Cost(output_tokens=reserved),
        actual=Cost(output_tokens=actual),
        at_ms=at_ms,
    )


def teach(estimator, context, lengths, reserved=4_096):
    for length in lengths:
        estimator.record(observe(context, length, reserved=reserved))


def reserved_by(estimator, context):
    estimate = estimator.estimate(context)
    return estimate.output.quantile(estimate.quantile)


def test_with_no_history_it_reserves_what_the_caller_allowed():
    estimator = QuantileEstimator()
    assert reserved_by(estimator, RequestContext(max_tokens=4_096)) == 4_096


def test_with_a_history_it_reserves_that_history_s_quantile():
    estimator = QuantileEstimator(min_samples=5)
    context = RequestContext(model="claude", max_tokens=4_096)
    teach(estimator, context, [120, 300, 340, 380, 4_100])
    assert reserved_by(estimator, context) == 2_612


def test_it_reserves_far_less_than_the_maximum_on_a_short_route():
    # The whole point. Five hundred requests of about three hundred tokens
    # against a four thousand token ceiling.
    estimator = QuantileEstimator()
    context = RequestContext(model="claude", max_tokens=4_096)
    teach(estimator, context, [300] * 500)
    assert reserved_by(estimator, context) == 300


def test_routes_are_kept_apart():
    estimator = QuantileEstimator(route_key=lambda ctx: ctx.tags.get("task"), min_samples=5)
    labels = RequestContext(max_tokens=4_096, tags={"task": "label"})
    reports = RequestContext(max_tokens=4_096, tags={"task": "report"})
    teach(estimator, labels, [12] * 50)
    teach(estimator, reports, [3_000] * 50)
    assert reserved_by(estimator, labels) == 12
    assert reserved_by(estimator, reports) == 3_000


def test_the_default_route_key_groups_by_model_alone():
    # Weak on purpose, and this test says so: two unrelated tasks on one model
    # land in one history and the prediction is the worse for it.
    estimator = QuantileEstimator()
    labels = RequestContext(model="claude", tags={"task": "label"})
    reports = RequestContext(model="claude", tags={"task": "report"})
    teach(estimator, labels, [12] * 50)
    teach(estimator, reports, [3_000] * 50)
    assert estimator.samples(labels) == estimator.samples(reports) == 100


def test_the_ring_evicts_the_oldest_first():
    estimator = QuantileEstimator(min_samples=1, history=3)
    context = RequestContext(model="claude")
    teach(estimator, context, [10, 20, 30, 40])
    # The 10 is gone, so the smallest thing the history knows about is now 20.
    assert estimator.estimate(context).output.samples == (20, 30, 40)
    assert estimator.samples(context) == 3


def test_the_ring_never_grows_past_its_bound():
    estimator = QuantileEstimator(min_samples=1, history=10)
    context = RequestContext(model="claude")
    teach(estimator, context, range(1_000))
    assert estimator.samples(context) == 10


def test_the_history_is_a_recency_window():
    # Not just a memory bound. Output lengths drift as prompts and models
    # change, and a history that never forgot would answer today's question
    # with last quarter's traffic.
    estimator = QuantileEstimator(min_samples=1, history=100)
    context = RequestContext(model="claude")
    teach(estimator, context, [100] * 100)
    assert reserved_by(estimator, context) == 100
    teach(estimator, context, [900] * 100)
    assert reserved_by(estimator, context) == 900


def test_a_quantile_matches_a_direct_computation_over_the_same_samples():
    estimator = QuantileEstimator(quantile=0.75, min_samples=10)
    context = RequestContext(model="claude")
    lengths = [17, 4, 902, 55, 3, 640, 88, 21, 5, 300]
    teach(estimator, context, lengths)
    assert reserved_by(estimator, context) == Distribution.empirical(lengths).quantile(0.75)


def test_it_counts_input_per_request():
    estimator = QuantileEstimator(min_samples=1)
    context = RequestContext(model="claude", prompt="hello there")
    teach(estimator, context, [300])
    assert estimator.estimate(context).input == 4


def test_it_carries_the_model_through():
    estimator = QuantileEstimator(min_samples=1)
    context = RequestContext(model="claude")
    teach(estimator, context, [300])
    assert estimator.estimate(context).model == "claude"


def test_a_route_nobody_has_asked_about_has_no_samples():
    assert QuantileEstimator().samples(RequestContext(model="claude")) == 0


@pytest.mark.parametrize("quantile", [-0.01, 1.01])
def test_a_quantile_outside_the_unit_interval_is_refused(quantile):
    with pytest.raises(ConfigurationError, match="between 0 and 1"):
        QuantileEstimator(quantile=quantile)


@pytest.mark.parametrize("history", [0, -1])
def test_a_history_of_nothing_is_refused(history):
    with pytest.raises(ConfigurationError, match="at least one"):
        QuantileEstimator(history=history)


def test_the_defaults_are_the_documented_ones():
    assert repr(QuantileEstimator()) == "QuantileEstimator(quantile=0.9, routes=0)"
    assert QuantileEstimator(history=DEFAULT_HISTORY) is not None


def test_below_the_threshold_the_fallback_answers():
    # A measurement that does not exist yet must not bind. Reading a ninth
    # decile off four samples would hold back traffic on almost nothing.
    estimator = QuantileEstimator(min_samples=30)
    context = RequestContext(model="claude", max_tokens=4_096)
    teach(estimator, context, [10] * 29)
    assert reserved_by(estimator, context) == 4_096


def test_at_the_threshold_the_history_takes_over():
    estimator = QuantileEstimator(min_samples=30)
    context = RequestContext(model="claude", max_tokens=4_096)
    teach(estimator, context, [10] * 30)
    assert reserved_by(estimator, context) == 10


def test_the_fallback_may_be_any_estimator():
    from spillway.estimators.static import StaticEstimator

    estimator = QuantileEstimator(
        min_samples=30, fallback=StaticEstimator(output=Distribution.point(500))
    )
    context = RequestContext(model="claude", max_tokens=4_096)
    assert reserved_by(estimator, context) == 500
    teach(estimator, context, [10] * 30)
    assert reserved_by(estimator, context) == 10


def test_a_threshold_of_zero_still_needs_one_observation():
    # There is nothing to read a quantile from until something has been seen,
    # so the fallback answers the very first request either way.
    estimator = QuantileEstimator(min_samples=0)
    context = RequestContext(model="claude", max_tokens=4_096)
    assert reserved_by(estimator, context) == 4_096
    teach(estimator, context, [10])
    assert reserved_by(estimator, context) == 10


def test_the_threshold_is_per_route():
    estimator = QuantileEstimator(route_key=lambda ctx: ctx.tags.get("task"), min_samples=5)
    busy = RequestContext(max_tokens=4_096, tags={"task": "busy"})
    quiet = RequestContext(max_tokens=4_096, tags={"task": "quiet"})
    teach(estimator, busy, [10] * 5)
    assert reserved_by(estimator, busy) == 10
    assert reserved_by(estimator, quiet) == 4_096


def test_a_negative_threshold_is_refused():
    with pytest.raises(ConfigurationError, match="cannot be negative"):
        QuantileEstimator(min_samples=-1)


def test_a_threshold_the_history_can_never_reach_is_refused():
    # The ring would forget its oldest observation before the count got there,
    # so the route's own history would never be used at all.
    with pytest.raises(ConfigurationError, match="never be used at all"):
        QuantileEstimator(min_samples=100, history=50)


def test_a_route_nobody_has_seen_has_no_statistics():
    assert QuantileEstimator().statistics(RequestContext(model="claude")) is None


def test_the_overrun_ratio_counts_settlements_that_used_more_than_was_reserved():
    estimator = QuantileEstimator()
    context = RequestContext(model="claude")
    teach(estimator, context, [50] * 9, reserved=100)
    teach(estimator, context, [900], reserved=100)
    assert estimator.statistics(context).overrun_ratio == pytest.approx(0.1)


def test_reserving_exactly_what_was_used_is_not_an_overrun():
    # An overrun is having used more than was held, not having got it exactly
    # right. Counting the boundary would inflate the number the adaptive loop
    # reads and push the quantile up for no reason.
    estimator = QuantileEstimator()
    context = RequestContext(model="claude")
    teach(estimator, context, [100] * 10, reserved=100)
    assert estimator.statistics(context).overrun_ratio == 0.0


def test_the_error_ratio_averages_reserved_over_actual():
    estimator = QuantileEstimator()
    context = RequestContext(model="claude")
    teach(estimator, context, [100, 200], reserved=400)
    assert estimator.statistics(context).error_ratio == pytest.approx(3.0)


def test_the_error_ratio_is_undefined_rather_than_infinite_when_nothing_was_generated():
    # Reserved over zero has no value. Calling it infinite would poison the
    # average with a number that means nothing.
    estimator = QuantileEstimator()
    context = RequestContext(model="claude")
    teach(estimator, context, [0, 0], reserved=400)
    assert estimator.statistics(context).error_ratio is None


def test_a_settlement_that_generated_nothing_still_counts_as_an_observation():
    estimator = QuantileEstimator()
    context = RequestContext(model="claude")
    teach(estimator, context, [0, 0], reserved=400)
    assert estimator.statistics(context).observations == 2


def test_the_error_ratio_ignores_only_the_empty_settlements():
    estimator = QuantileEstimator()
    context = RequestContext(model="claude")
    teach(estimator, context, [0, 100], reserved=400)
    assert estimator.statistics(context).error_ratio == pytest.approx(4.0)


def test_observations_outlive_the_ring():
    # The ring forgets, on purpose. The count of what has been seen does not,
    # because it is what tells someone whether a route is busy at all.
    estimator = QuantileEstimator(min_samples=1, history=5)
    context = RequestContext(model="claude")
    teach(estimator, context, [10] * 50)
    stats = estimator.statistics(context)
    assert (stats.samples, stats.observations) == (5, 50)


def test_the_statistics_report_the_route_s_own_quantile():
    estimator = QuantileEstimator(quantile=0.75, min_samples=1)
    context = RequestContext(model="claude")
    teach(estimator, context, [10])
    assert estimator.statistics(context).quantile == 0.75
