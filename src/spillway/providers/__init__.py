"""One provider's accounting rules, encoded so nobody has to read three pages.

An adapter says how a provider counts, never how much you are allowed. You name
your limits; these describe how to charge against them.

    from spillway import providers

    limiter = Spillway(provider=providers.anthropic(), rpm=1_000)

Reaching for one of these by hand is rarely necessary. An instrumented client
recognises its own provider.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from spillway.core.errors import ConfigurationError
from spillway.providers.anthropic import Anthropic
from spillway.providers.base import Outcome, ProviderAdapter, ProviderState
from spillway.providers.openai import OpenAI
from spillway.providers.openai_compatible import OpenAICompatible

__all__ = [
    "Anthropic",
    "OpenAI",
    "OpenAICompatible",
    "Outcome",
    "ProviderAdapter",
    "ProviderState",
    "anthropic",
    "by_name",
    "known",
    "openai",
    "openai_compatible",
]


def anthropic(*, now: Callable[[], datetime] | None = None) -> Anthropic:
    """The Anthropic adapter.

    Args:
        now: Where the current moment comes from, for turning a reset
            timestamp into a wait. Only worth passing from a test.

    Example:
        >>> anthropic().charges_max_tokens()
        False
    """
    return Anthropic() if now is None else Anthropic(now=now)


def openai() -> OpenAI:
    """The OpenAI adapter.

    Example:
        >>> openai().charges_max_tokens()
        True
    """
    return OpenAI()


def openai_compatible(*, metrics_url: str | None = None) -> OpenAICompatible:
    """The generic adapter, for anything speaking the OpenAI schema elsewhere.

    Args:
        metrics_url: Where a self hosted engine exposes its own serving
            metrics. Accepted and unused.

    Example:
        >>> openai_compatible().official_hosts
        ()
    """
    return OpenAICompatible(metrics_url=metrics_url)


def known() -> tuple[str, ...]:
    """Every provider name this library recognises.

    Example:
        >>> known()
        ('anthropic', 'openai', 'openai_compatible')
    """
    return ("anthropic", "openai", "openai_compatible")


def by_name(name: str) -> ProviderAdapter:
    """Build an adapter from its name.

    Args:
        name: One of `known()`.

    Raises:
        ConfigurationError: if the name is not one this library ships, naming
            the ones it does.

    Example:
        >>> by_name("openai").name
        'openai'
    """
    # The annotation is load bearing rather than decorative. Satisfying the
    # protocol at runtime only means the members exist, and an adapter whose
    # attributes are annotated too narrowly passes that and still fails a
    # strict type check. Naming the protocol here is what makes the build say
    # so, which is how a contributor finds out rather than a user.
    builders: dict[str, Callable[[], ProviderAdapter]] = {
        "anthropic": anthropic,
        "openai": openai,
        "openai_compatible": openai_compatible,
    }
    builder = builders.get(name.strip().lower())
    if builder is None:
        message = (
            f"No provider called {name!r} ships with this library. The ones that do are: "
            f"{', '.join(known())}. Anything speaking the OpenAI schema against another "
            f"service is 'openai_compatible'. For a provider that is not here, pass your "
            f"own adapter: it is a protocol, so nothing needs to be imported or subclassed."
        )
        raise ConfigurationError(message)
    return builder()
