"""Turning a provider's own client into one that admits before it calls.

The whole mechanism is: copy the client, and replace the completion methods on
the copy. Everything else falls out of that.
"""

from __future__ import annotations

from collections.abc import Mapping

from spillway.core.scope import Priority, Scope
from spillway.integrations.context import CallContext, current

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
