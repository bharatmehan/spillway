"""The fixed prediction, for routes that really are predictable."""

from spillway.core.cost import Cost, Distribution
from spillway.estimators.base import Observation, RequestContext
from spillway.estimators.static import StaticEstimator


def test_it_predicts_the_same_output_whatever_it_is_asked():
    estimator = StaticEstimator(output=Distribution.point(120))
    for prompt in ("a", "a" * 10_000, None):
        assert estimator.estimate(RequestContext(prompt=prompt)).output.value == 120


def test_it_ignores_the_requested_maximum():
    # The caller said this route produces 120 tokens. Reserving 4096 because
    # they also set a ceiling would throw that knowledge away.
    estimator = StaticEstimator(output=Distribution.point(120))
    assert estimator.estimate(RequestContext(max_tokens=4_096)).output.value == 120


def test_it_still_counts_input_per_request():
    # Input is countable, so fixing it too would be inventing uncertainty
    # where there is none.
    estimator = StaticEstimator(output=Distribution.point(120))
    assert estimator.estimate(RequestContext(prompt="hello there")).input == 4


def test_it_carries_the_model_through():
    estimator = StaticEstimator(output=Distribution.point(1))
    assert estimator.estimate(RequestContext(model="claude")).model == "claude"


def test_recording_changes_nothing():
    estimator = StaticEstimator(output=Distribution.point(120))
    estimator.record(
        Observation(
            context=RequestContext(),
            reserved=Cost(output_tokens=120),
            actual=Cost(output_tokens=8_000),
            at_ms=0.0,
        )
    )
    assert estimator.estimate(RequestContext()).output.value == 120
