"""The provider gets the last word on what a request reserves."""

from spillway.core.cost import Distribution, Estimate
from spillway.core.spillway import Spillway
from spillway.estimators.static import StaticEstimator
from spillway.providers.anthropic import Anthropic
from spillway.providers.openai import OpenAI


def _reserved(limiter, **kwargs):
    """What one admission would take, without taking it."""
    context = limiter.admit(**kwargs)
    return context._reserved(context._context())


def test_without_a_provider_the_prediction_stands():
    limiter = Spillway(estimator=StaticEstimator(output=Distribution.point(250)))
    assert _reserved(limiter, max_tokens=4_096).output_tokens == 250


def test_a_provider_that_charges_the_maximum_overrides_the_prediction():
    # The whole point of the adjustment. The provider takes 4096 whatever this
    # library predicted, so reserving 250 would leave the limiter believing in
    # headroom the provider does not agree exists.
    estimator = StaticEstimator(output=Distribution.point(250))
    limiter = Spillway(provider=OpenAI(), estimator=estimator)
    assert _reserved(limiter, max_tokens=4_096).output_tokens == 4_096


def test_a_provider_that_meters_what_was_generated_leaves_it_alone():
    # And this is where the prediction earns its keep: reserving what nine
    # requests in ten come in under, rather than a maximum nobody reaches.
    estimator = StaticEstimator(output=Distribution.point(250))
    limiter = Spillway(provider=Anthropic(), estimator=estimator)
    assert _reserved(limiter, max_tokens=4_096).output_tokens == 250


def test_an_explicit_estimate_is_adjusted_too():
    # A provider charges the maximum whoever did the predicting. Exempting a
    # caller who passed their own estimate would exempt them from the
    # provider's accounting, which is not something this library can grant.
    limiter = Spillway(provider=OpenAI())
    given = Estimate(input=100, output=Distribution.point(250))
    assert _reserved(limiter, estimate=given, max_tokens=4_096).output_tokens == 4_096


def test_the_input_side_is_untouched_by_the_adjustment():
    # The adjustment is about output accounting. Input is counted rather than
    # predicted, so there is nothing for a provider to override.
    estimator = StaticEstimator(output=Distribution.point(250))
    plain = Spillway(estimator=estimator)
    charged = Spillway(provider=OpenAI(), estimator=estimator)
    prompt = "summarise this document " * 20
    assert (
        _reserved(charged, prompt=prompt, max_tokens=4_096).input_tokens
        == _reserved(plain, prompt=prompt, max_tokens=4_096).input_tokens
        > 0
    )


def test_a_request_naming_no_maximum_is_left_alone():
    estimator = StaticEstimator(output=Distribution.point(250))
    limiter = Spillway(provider=OpenAI(), estimator=estimator)
    assert _reserved(limiter).output_tokens == 250
