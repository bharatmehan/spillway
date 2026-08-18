"""The property testing profiles are registered and one of them is loaded."""

import os

from hypothesis import settings


def test_a_profile_is_loaded():
    assert settings.default is not None


def test_the_loaded_profile_matches_the_environment():
    expected = os.environ.get("HYPOTHESIS_PROFILE", "dev")
    assert settings.get_profile(expected) is not None


def test_the_continuous_integration_profile_is_reproducible():
    # A property test that fails on a shared runner with a seed nobody can
    # reproduce is a test that gets deleted rather than fixed.
    assert settings.get_profile("ci").derandomize is True


def test_neither_profile_fails_a_test_for_being_slow():
    assert settings.get_profile("ci").deadline is None
    assert settings.get_profile("dev").deadline is None
