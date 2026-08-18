"""The output length distribution and the estimate that carries it."""

import pytest

from spillway.core.cost import Distribution, Estimate


def test_a_point_answers_every_quantile_with_the_same_value():
    known = Distribution.point(300)
    assert [known.quantile(q) for q in (0.0, 0.5, 0.9, 1.0)] == [300, 300, 300, 300]


def test_a_bound_answers_every_quantile_with_the_bound():
    assert Distribution.bounded_by(4096).quantile(0.5) == 4096


def test_the_two_kinds_are_distinguishable():
    # They answer identically today and mean different things. Later stages
    # treat a bound as a fallback to be replaced once observations exist, and a
    # point as a belief to be honoured, so the kinds must not be conflated.
    assert Distribution.point(100) != Distribution.bounded_by(100)


def test_a_negative_prediction_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        Distribution.point(-1)


@pytest.mark.parametrize("q", [-0.01, 1.01, 2.0])
def test_a_quantile_outside_the_unit_interval_is_refused(q):
    with pytest.raises(ValueError, match="between 0 and 1"):
        Distribution.point(10).quantile(q)


def test_zero_is_a_legitimate_prediction():
    assert Distribution.point(0).quantile(0.9) == 0


def test_an_estimate_carries_an_exact_input_and_a_predicted_output():
    estimate = Estimate(input=12_400, output=Distribution.point(415), model="claude")
    assert estimate.input == 12_400
    assert estimate.output.quantile(0.9) == 415
    assert estimate.model == "claude"


def test_an_estimate_has_no_model_by_default():
    assert Estimate(input=1, output=Distribution.point(1)).model is None


def test_a_negative_input_count_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        Estimate(input=-1, output=Distribution.point(1))
