"""Working out which provider a client speaks to, and how far that goes.

Two questions, and conflating them is the mistake this module exists to avoid.

**What protocol does this client speak?** The top level module of its class
says so, reliably.

**Do that provider's accounting rules apply to it?** Only if it is pointed at
that provider. A client speaking the OpenAI schema against something else is a
hosted service or a local engine, charging no requested maximum, reporting
different usage categories and sending different headers.

So the base URL decides. An official host gets the named adapter, anything else
gets the generic one, which assumes nothing, and it says so once.
"""

from __future__ import annotations

import inspect
import logging
from urllib.parse import urlsplit

from spillway.core.errors import ConfigurationError
from spillway.providers import known
from spillway.providers.base import ProviderAdapter

_log = logging.getLogger(__name__)

_warned_about_unofficial: set[str] = set()


def module_of(client: object) -> str:
    """The top level module of a client's class.

    Example:
        >>> module_of(object())
        'builtins'
    """
    return type(client).__module__.split(".")[0]


def host_of(client: object) -> str | None:
    """The host a client is pointed at, if it says.

    Example:
        >>> class Client:
        ...     base_url = "https://api.example.com/v1/"
        >>> host_of(Client())
        'api.example.com'
        >>> host_of(object()) is None
        True
    """
    raw = getattr(client, "base_url", None)
    if raw is None:
        return None
    split = urlsplit(str(raw))
    return split.hostname


def adapter_for(client: object) -> ProviderAdapter:
    """Work out whose accounting rules apply to this client.

    Args:
        client: An instance of a provider's client library.

    Returns:
        The named adapter when the client is pointed at one of that provider's
        own hosts, and the generic compatible adapter otherwise.

    Raises:
        ConfigurationError: if nothing recognises this client at all, naming
            both ways to proceed.

    Example:
        >>> class NotAClient:
        ...     pass
        >>> adapter_for(NotAClient())
        Traceback (most recent call last):
        ...
        spillway.core.errors.ConfigurationError: ...
    """
    from spillway.providers import by_name

    module = module_of(client)
    named = [
        by_name(name)
        for name in known()
        if by_name(name).client_module == module and by_name(name).official_hosts
    ]
    if not named:
        message = (
            f"Nothing recognises a client from {module!r}. Spillway knows the client "
            f"libraries for: {', '.join(known())}. Either pass provider= with an adapter "
            f"of your own, which is a protocol so nothing needs importing or subclassing, "
            f"or use spillway.admit() directly and settle the lease by hand."
        )
        raise ConfigurationError(message)
    host = host_of(client)
    for adapter in named:
        if host is not None and host in adapter.official_hosts:
            return adapter
    generic = by_name("openai_compatible")
    if generic.client_module == module:
        _warn_once_about_unofficial(host, named[0].name)
        return generic
    # A client library with no generic counterpart, pointed somewhere else.
    # Its own adapter is still the right reader for its own wire format, and
    # the limits were never ours to assume in the first place.
    return named[0]


def is_asynchronous(client: object, adapter: ProviderAdapter) -> bool:
    """Whether this client's calls have to be awaited.

    Decided by asking one of the methods rather than reading the class name.
    The methods carry a decorator that hides the real function, so it is
    unwrapped first.

    Returns:
        True when the client is asynchronous, and True when there is nothing
        to ask, since that is the case this library supports.
    """
    for path in adapter.endpoints:
        method = _resolve(client, path)
        if not callable(method):
            continue
        return inspect.iscoroutinefunction(inspect.unwrap(method))
    return True


def _resolve(client: object, path: str) -> object | None:
    """Walk a dotted method path down a client, or give up quietly.

    A path this client library does not have is not an error. An adapter names
    the endpoints a current version exposes, and an older version simply has
    fewer of them, which should cost the caller nothing.
    """
    found: object = client
    for part in path.split("."):
        found = getattr(found, part, None)
        if found is None:
            return None
    return found


def _warn_once_about_unofficial(host: str | None, instead_of: str) -> None:
    """Say, once per host, that no published accounting is being assumed."""
    where = host or "an unnamed host"
    if where in _warned_about_unofficial:
        return
    _warned_about_unofficial.add(where)
    _log.warning(
        "This client speaks the %s protocol but is pointed at %s, so Spillway is not "
        "assuming %s's accounting rules for it. Requests are still admitted, settled "
        "and observed. If it really is %s behind a proxy, pass provider='%s' to say so.",
        instead_of,
        where,
        instead_of,
        instead_of,
        instead_of,
    )
