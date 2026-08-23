"""Turning a provider's own client into one that admits before it calls.

The whole mechanism is: copy the client, and replace the completion methods on
the copy. Everything else falls out of that.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, TypeVar

from spillway.core.errors import ConfigurationError
from spillway.core.scope import Priority, Scope
from spillway.integrations.context import CallContext, current
from spillway.integrations.detect import adapter_for, is_asynchronous
from spillway.providers.base import ProviderAdapter

if TYPE_CHECKING:  # pragma: no cover - imported for typing only, to avoid a cycle
    from spillway.core.spillway import Spillway

_log = logging.getLogger(__name__)

ClientT = TypeVar("ClientT")

HELD_BY = "_spillway"
"""Where the limiter is kept on an instrumented client.

Reading it is how `Spillway.of` works and how instrumenting the same client
twice is refused.
"""

RESERVED = ("spillway_scope", "spillway_priority", "spillway_tags")
"""Keywords taken out of a call before it is forwarded.

Prefixed so they cannot collide with a provider parameter, now or after a
provider adds one. A provider that ever ships a parameter starting this way has
bigger problems than this library.
"""


def resolve(
    kwargs: Mapping[str, object],
    *,
    scope: Scope | None = None,
    priority: int | None = None,
) -> tuple[dict[str, object], CallContext]:
    """Separate what Spillway reads from what the provider gets.

    Four sources, in this order, each beating the one before it: the limiter's
    own default, the client's, the surrounding context variable, and a reserved
    keyword on the call itself. The keyword is last because somebody writing it
    at a call site is being specific on purpose.

    Args:
        kwargs: What the caller passed.
        scope: The instrumented client's default scope.
        priority: The instrumented client's default priority.

    Returns:
        The keywords to forward, with the reserved ones removed, and what they
        said about scope, priority and tags.

    Example:
        >>> forwarded, found = resolve({"model": "m", "spillway_scope": "tenant:acme"})
        >>> forwarded
        {'model': 'm'}
        >>> found.scope.key
        'tenant:acme'

        Nothing reserved means nothing is taken out.

        >>> resolve({"model": "m"})[0]
        {'model': 'm'}
    """
    forwarded = {name: value for name, value in kwargs.items() if name not in RESERVED}
    surrounding = current()

    chosen_scope = surrounding.scope if surrounding.scope is not None else scope
    given_scope = kwargs.get("spillway_scope")
    if given_scope is not None:
        chosen_scope = Scope.of(given_scope) if isinstance(given_scope, (str, Scope)) else None

    chosen_priority = surrounding.priority if surrounding.priority is not None else priority
    given_priority = kwargs.get("spillway_priority")
    if isinstance(given_priority, (int, Priority)):
        chosen_priority = int(given_priority)

    tags = dict(surrounding.tags)
    given_tags = kwargs.get("spillway_tags")
    if isinstance(given_tags, Mapping):
        tags.update({str(key): str(value) for key, value in given_tags.items()})

    return forwarded, CallContext(scope=chosen_scope, priority=chosen_priority, tags=tags)


def patch(
    client: ClientT,
    limiter: Spillway,
    *,
    provider: ProviderAdapter | str | None = None,
    scope: str | Scope | None = None,
    priority: int | Priority | None = None,
) -> ClientT:
    """Return a copy of `client` whose completion methods go through `limiter`.

    The original is untouched, so an instrumented client and a bare one are two
    independent objects drawing on the same connection pool.

    Args:
        client: An instance of a provider's client library.
        limiter: What admits the calls.
        provider: Overrides detection.
        scope: Default scope for calls through this client.
        priority: Default priority for calls through this client.

    Raises:
        ConfigurationError: if the client is already instrumented, if nothing
            recognises it, or if it is synchronous.

    The mechanism, which is smaller than it sounds. Both client libraries
    expose a public copy that returns the same type and reuses the existing
    connection pool, and attach their resources as ordinary attributes on
    ordinary objects. So: copy, then replace the bound methods on the copy's
    own resources. The returned object genuinely is an instance of the class
    handed in, which is why editor completion, `isinstance` and strict type
    checking all keep working without anything having to pretend.
    """
    if getattr(client, HELD_BY, None) is not None:
        message = (
            "This client is already instrumented. Instrumenting it again would stack "
            "two limiters over one quota, so every call would be admitted twice and the "
            "effective limit would be half what was asked for. Instrument the original "
            "client, or reach the limiter behind this one with Spillway.of(client)."
        )
        raise ConfigurationError(message)

    adapter = _adapter(client, provider)
    if not is_asynchronous(client, adapter):
        message = (
            f"{type(client).__name__} is a synchronous client, and Spillway cannot "
            f"instrument one yet: admission has to wait for capacity, and waiting from "
            f"synchronous code needs a synchronous driver that does not exist. Use the "
            f"asynchronous client for now, or call spillway.admit() around the call "
            f"yourself."
        )
        raise ConfigurationError(message)

    instrumented = _copy(client)
    default_scope = Scope.of(scope) if scope is not None else None
    default_priority = int(priority) if priority is not None else None
    patched = 0
    for path in adapter.endpoints:
        resource, name = _owner_of(instrumented, path)
        if resource is None:
            continue
        original = getattr(resource, name)
        setattr(
            resource,
            name,
            _admitted(original, limiter, adapter, path, default_scope, default_priority),
        )
        patched += 1
    if patched == 0:
        message = (
            f"None of the methods the {adapter.name!r} adapter instruments exist on this "
            f"client, so instrumenting it would do nothing at all and every call would "
            f"go out unadmitted. The adapter expects: {', '.join(adapter.endpoints)}. "
            f"Check the client library version, or pass provider= with an adapter that "
            f"matches it."
        )
        raise ConfigurationError(message)
    setattr(instrumented, HELD_BY, limiter)
    return instrumented


def limiter_of(client: object) -> Spillway:
    """Return the limiter behind an instrumented client.

    How a health check reaches `snapshot()` without the application threading
    a limiter around beside every client it holds.

    Raises:
        ConfigurationError: if this client is not instrumented.
    """
    found = getattr(client, HELD_BY, None)
    if found is None:
        message = (
            "This client is not instrumented, so there is no limiter behind it. Pass "
            "the client returned by Spillway.instrument(...), not the one handed to it: "
            "instrumenting works on a copy and deliberately leaves the original alone."
        )
        raise ConfigurationError(message)
    return found  # type: ignore[no-any-return]


def _adapter(client: object, provider: ProviderAdapter | str | None) -> ProviderAdapter:
    """Whose rules to apply: the caller's choice, or whatever recognises this."""
    if provider is None:
        return adapter_for(client)
    if isinstance(provider, str):
        from spillway.providers import by_name

        return by_name(provider)
    return provider


def _copy(client: ClientT) -> ClientT:
    """Duplicate a client so the caller's own is left alone.

    Both client libraries expose this and reuse the existing connection pool
    when no new one is passed, so a copy costs no second pool.

    Raises:
        ConfigurationError: if this client cannot be copied, naming the fix.
    """
    duplicate = getattr(client, "copy", None)
    if not callable(duplicate):
        message = (
            f"{type(client).__name__} has no copy() method, so Spillway cannot "
            f"instrument it without modifying the client you passed in, which it will "
            f"not do. Call spillway.admit() around the call yourself instead."
        )
        raise ConfigurationError(message)
    made = duplicate()
    if not isinstance(made, type(client)):
        message = (
            f"{type(client).__name__}.copy() returned a {type(made).__name__} rather "
            f"than another {type(client).__name__}, so the instrumented client would "
            f"not be the type you passed in. Call spillway.admit() around the call "
            f"yourself instead."
        )
        raise ConfigurationError(message)
    return made


def _owner_of(client: object, path: str) -> tuple[object | None, str]:
    """Find the object a dotted endpoint path hangs off, and the last name.

    Walking the path is what builds each resource on the copy, and building it
    is what makes replacing a method on it stick to this client and to no
    other.
    """
    parts = path.split(".")
    owner: object = client
    for part in parts[:-1]:
        owner = getattr(owner, part, None)
        if owner is None:
            return None, parts[-1]
    if not callable(getattr(owner, parts[-1], None)):
        return None, parts[-1]
    return owner, parts[-1]


def _admitted(
    original: Callable[..., Awaitable[object]],
    limiter: Spillway,
    adapter: ProviderAdapter,
    endpoint: str,
    scope: Scope | None,
    priority: int | None,
) -> Callable[..., Awaitable[object]]:
    """Wrap one bound method so it admits, forwards, and settles.

    Deliberately thin. It is a translation layer, and every decision that
    appears in it is a decision the limiter cannot see, the simulation harness
    cannot reach, and the next provider will need a second copy of.
    """

    @functools.wraps(original)
    async def admitted(**kwargs: object) -> object:
        forwarded, call = resolve(kwargs, scope=scope, priority=priority)
        context = adapter.request_from(endpoint, forwarded)
        async with limiter.admit(
            scope=call.scope,
            priority=call.priority if call.priority is not None else Priority.NORMAL,
            prompt=context.prompt,
            max_tokens=context.max_tokens,
            model=context.model,
            tags=call.tags,
        ) as lease:
            response = await original(**forwarded)
            lease.settle_from(response)
            return response

    return admitted
