"""The escape hatch."""

import pytest

from spillway.core.cost import RESERVATION_QUANTILE, Cost, Distribution
from spillway.estimators.base import Observation, RequestContext
from spillway.estimators.callable import CallableEstimator


def test_it_asks_the_wrapped_function():
    estimator = CallableEstimator(lambda context: Distribution.point(40))
    assert estimator.estimate(RequestContext()).output.value == 40


def test_the_function_sees_the_whole_context():
    def guess(context):
        return Distribution.point(40 if context.tags.get("task") == "label" else 800)

    estimator = CallableEstimator(guess)
    assert estimator.estimate(RequestContext(tags={"task": "label"})).output.value == 40
    assert estimator.estimate(RequestContext(tags={"task": "write"})).output.value == 800


def test_it_reserves_at_the_ninth_decile_unless_told_otherwise():
    estimator = CallableEstimator(lambda context: Distribution.point(1))
    assert estimator.estimate(RequestContext()).quantile == RESERVATION_QUANTILE


def test_it_honours_a_quantile_of_its_own():
    # Someone returning a real distribution from a real predictor should be
    # able to say how conservatively to read it.
    estimator = CallableEstimator(
        lambda context: Distribution.empirical([10, 20, 30]), quantile=1.0
    )
    estimate = estimator.estimate(RequestContext())
    assert estimate.output.quantile(estimate.quantile) == 30


def test_it_still_counts_input():
    estimator = CallableEstimator(lambda context: Distribution.point(1))
    assert estimator.estimate(RequestContext(prompt="hello there")).input == 4


def test_a_function_that_raises_is_not_swallowed():
    # A prediction that fails is a bug in the caller's code, and hiding it
    # behind a silent fallback would leave them reserving a number they never
    # asked for and no way to find out.
    def broken(context):
        raise ZeroDivisionError("bad maths")

    with pytest.raises(ZeroDivisionError):
        CallableEstimator(broken).estimate(RequestContext())


def test_it_names_the_function_it_wraps():
    def my_predictor(context):
        return Distribution.point(1)

    assert repr(CallableEstimator(my_predictor)) == "CallableEstimator(my_predictor)"


def test_recording_changes_nothing():
    estimator = CallableEstimator(lambda context: Distribution.point(40))
    estimator.record(
        Observation(
            context=RequestContext(),
            reserved=Cost(output_tokens=40),
            actual=Cost(output_tokens=900),
            at_ms=0.0,
        )
    )
    assert estimator.estimate(RequestContext()).output.value == 40
