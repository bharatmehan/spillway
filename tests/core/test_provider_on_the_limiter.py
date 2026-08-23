"""A limiter can be told whose accounting rules to apply."""

import pytest

from spillway.core.errors import ConfigurationError
from spillway.core.spillway import Spillway
from spillway.providers.anthropic import Anthropic


def test_a_limiter_has_no_provider_by_default():
    # Which is correct rather than incomplete. Without one a reservation is
    # taken exactly as estimated, which is what every stage before this did.
    assert Spillway().provider is None


def test_a_provider_can_be_named_by_string():
    assert Spillway(provider="anthropic").provider is not None
    assert Spillway(provider="anthropic").provider.name == "anthropic"


def test_a_provider_can_be_passed_as_an_adapter():
    adapter = Anthropic()
    assert Spillway(provider=adapter).provider is adapter


def test_an_adapter_written_elsewhere_is_accepted():
    # The extension point is a protocol, so an object with the members is an
    # adapter and nothing has to be imported or subclassed to write one.
    class Mine:
        name = "mine"
        client_module = "mine"
        official_hosts = ()
        endpoints = ()

    assert Spillway(provider=Mine()).provider.name == "mine"


def test_an_unknown_provider_names_the_ones_that_exist():
    with pytest.raises(ConfigurationError, match="anthropic, openai, openai_compatible"):
        Spillway(provider="claude")


def test_the_provider_shows_in_the_repr():
    # A limiter is held for the life of a process and read in a debugger far
    # from where it was built, so what it is enforcing should be visible.
    assert "provider='openai'" in repr(Spillway(provider="openai"))
    assert "provider" not in repr(Spillway())
