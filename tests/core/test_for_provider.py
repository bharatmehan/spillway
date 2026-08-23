"""One call that builds a limiter which knows how a provider counts."""

import pytest

from spillway.core.clock import FakeClock
from spillway.core.cost import Distribution
from spillway.core.errors import ConfigurationError
from spillway.core.spillway import Spillway
from spillway.estimators.max_tokens import MaxTokensEstimator
from spillway.estimators.quantile import QuantileEstimator
from spillway.estimators.static import StaticEstimator
from spillway.providers.openai import OpenAI


def test_it_carries_the_provider():
    assert Spillway.for_provider("anthropic").provider.name == "anthropic"
    assert Spillway.for_provider(OpenAI()).provider.name == "openai"


def test_it_defaults_to_the_estimator_that_learns():
    # The difference that earns the separate name. Spillway() itself cannot
    # default to this without changing what every limiter built before this
    # existed does.
    assert isinstance(Spillway.for_provider("anthropic")._estimator, QuantileEstimator)
    assert isinstance(Spillway()._estimator, MaxTokensEstimator)


def test_the_learning_default_is_safe_from_cold():
    # Below its sample threshold it defers to the requested maximum, so a
    # fresh process reserves exactly what the conservative estimator would.
    learning = Spillway.for_provider("anthropic", clock=FakeClock())
    conservative = Spillway(clock=FakeClock())
    for limiter in (learning, conservative):
        context = limiter.admit(max_tokens=4_096)
        assert context._reserved(context._context()).output_tokens == 4_096


def test_the_estimator_can_be_overridden():
    given = StaticEstimator(output=Distribution.point(7))
    assert Spillway.for_provider("openai", estimator=given)._estimator is given


def test_named_limits_become_dimensions():
    limiter = Spillway.for_provider("anthropic", rpm=1_000, input_tpm=2_000_000)
    assert [d.name for d in limiter.dimensions] == ["rpm", "input_tpm"]


def test_naming_no_limits_observes_without_limiting():
    # The intended first step, and the only one now that no figures ship.
    assert Spillway.for_provider("openai").dimensions == ()


def test_an_unknown_provider_names_the_ones_that_exist():
    with pytest.raises(ConfigurationError, match="anthropic, openai, openai_compatible"):
        Spillway.for_provider("gemini")


async def test_the_provider_reaches_both_ends_of_a_request():
    # The point of building it this way round: the adjustment applies at
    # admission and the response is readable at settlement, from one call.
    limiter = Spillway.for_provider("openai", clock=FakeClock(), tpm=1_000_000)
    lease = await limiter.admit(max_tokens=4_096).acquire()
    assert lease.reserved.output_tokens == 4_096
    lease.settle_from({"usage": {"prompt_tokens": 19, "completion_tokens": 10}})
    assert limiter.snapshot().dimensions["tpm"].used == pytest.approx(29)
