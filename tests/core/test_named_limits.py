"""Named limits become the dimensions that enforce them, and nothing else."""

import pytest

from spillway.core.cost import Cost
from spillway.core.errors import ConfigurationError
from spillway.core.scope import Scope
from spillway.core.spillway import NAMED_LIMITS, Spillway, dimensions_from
from spillway.dimensions.concurrency import Concurrency
from spillway.dimensions.rate import Rate


def _window_ms(dimension):
    """The window a rate enforces over, read the way the store reads it."""
    claim = dimension.claim(Cost(input_tokens=1, output_tokens=1), Scope("tenant:acme"))
    return claim.window_ms


def test_naming_nothing_limits_nothing():
    # The observe and do not limit path, which is the intended first step now
    # that no limit figures ship with the library.
    assert dimensions_from() == ()


def test_each_named_limit_becomes_one_dimension():
    built = dimensions_from(
        rpm=1_000,
        rpd=10_000,
        tpm=150_000,
        input_tpm=2_000_000,
        output_tpm=400_000,
        concurrency=64,
    )
    assert [d.name for d in built] == [
        "rpm",
        "rpd",
        "tpm",
        "input_tpm",
        "output_tpm",
        "generations",
    ]


def test_every_named_limit_is_reachable():
    # NAMED_LIMITS is documentation, and documentation that drifts from the
    # signature is worse than none. Each name here has to actually build
    # something, or the list is promising a limit nobody can set.
    for name in NAMED_LIMITS:
        built = dimensions_from(**{name: 10})
        assert len(built) == 1, name


def test_a_daily_limit_gets_a_daily_window():
    # The one place a wrong constant would be invisible: an rpd built on a
    # sixty second window enforces a limit fourteen hundred times too tight,
    # and every other assertion about it still passes.
    rpd = dimensions_from(rpd=10_000)[0]
    assert isinstance(rpd, Rate)
    assert _window_ms(rpd) == 86_400_000.0


def test_a_minute_limit_gets_a_minute_window():
    rpm = dimensions_from(rpm=60)[0]
    assert isinstance(rpm, Rate)
    assert _window_ms(rpm) == 60_000.0


def test_concurrency_is_a_gauge_not_a_rate():
    # A concurrency limit expressed as a rate would admit the whole limit every
    # window regardless of what was still in flight, which is not a concurrency
    # limit at all.
    built = dimensions_from(concurrency=8)[0]
    assert isinstance(built, Concurrency)
    assert built.limit == 8.0


def test_a_limiter_takes_named_limits_directly():
    limiter = Spillway(rpm=1_000, input_tpm=2_000_000, output_tpm=400_000)
    assert [d.name for d in limiter.dimensions] == ["rpm", "input_tpm", "output_tpm"]


def test_named_limits_and_dimensions_compose():
    # The keyword form is sugar, not a replacement. Anything needing a window,
    # a meter or a name this library does not know is still a dimension.
    limiter = Spillway(
        dimensions=[Rate("images_per_minute", limit=10, meter="requests")],
        rpm=1_000,
    )
    assert [d.name for d in limiter.dimensions] == ["images_per_minute", "rpm"]


def test_naming_the_same_limit_twice_is_refused():
    # Two figures for one limit have no honest resolution, and silently
    # preferring one would enforce a number the caller can see they did not ask
    # for.
    with pytest.raises(ConfigurationError, match="given twice"):
        Spillway(dimensions=[Rate("rpm", limit=50)], rpm=1_000)


def test_a_limiter_naming_nothing_still_limits_nothing():
    assert Spillway().dimensions == ()
