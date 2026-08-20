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


def test_an_empirical_distribution_sorts_whatever_order_it_is_given():
    scrambled = Distribution.empirical([340, 120, 4_100, 300, 380])
    ordered = Distribution.empirical([120, 300, 340, 380, 4_100])
    assert scrambled == ordered


def test_an_empirical_quantile_lands_on_a_sample_when_it_falls_on_one():
    # Five samples, so q=0.5 sits exactly on index 2 and no interpolation runs.
    assert Distribution.empirical([120, 300, 340, 380, 4_100]).quantile(0.5) == 340


def test_an_empirical_quantile_interpolates_between_the_two_it_falls_between():
    # Position 0.9 * 4 = 3.6, six tenths of the way from 380 to 4100.
    assert Distribution.empirical([120, 300, 340, 380, 4_100]).quantile(0.9) == 2_612


def test_an_empirical_quantile_rounds_up():
    # Position 0.5 * 1 = 0.5, halfway between 10 and 11, which is 10.5. Rounding
    # down would reserve less than the quantile asked for, every single time the
    # quantile falls between two samples.
    assert Distribution.empirical([10, 11]).quantile(0.5) == 11


def test_the_extremes_of_an_empirical_distribution_are_its_extremes():
    observed = Distribution.empirical([120, 300, 4_100])
    assert (observed.quantile(0.0), observed.quantile(1.0)) == (120, 4_100)


def test_one_sample_answers_every_quantile_with_itself():
    single = Distribution.empirical([415])
    assert [single.quantile(q) for q in (0.0, 0.5, 1.0)] == [415, 415, 415]


def test_an_empirical_distribution_over_nothing_is_refused():
    # Not a fallback. Whatever built it had no history and should have said so.
    with pytest.raises(ValueError, match="at least one observed sample"):
        Distribution.empirical([])


def test_a_negative_sample_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        Distribution.empirical([300, -1])


def test_an_empirical_repr_counts_its_samples_rather_than_listing_them():
    assert repr(Distribution.empirical(range(1_000))) == (
        "Distribution(kind='empirical', value=999, samples=1000)"
    )


def test_the_simple_reprs_are_unchanged_by_the_new_field():
    assert repr(Distribution.point(300)) == "Distribution(kind='point', value=300)"
    assert repr(Distribution.bounded_by(4_096)) == "Distribution(kind='bounded', value=4096)"


def test_an_exact_boundary_is_not_pushed_up_by_arithmetic_noise():
    # 0.9 * 4 is 3.6000000000000005 rather than 3.6, so the interpolation lands
    # a hair above 2612. Ceiling that raw would reserve 2613 for no reason but
    # float representation, on every quantile that falls on a whole token.
    assert Distribution.empirical([120, 300, 340, 380, 4_100]).quantile(0.9) == 2_612
