"""Saying whose request this is, without threading it through every frame.

An instrumented client has no `admit()` call to pass arguments to, so this is
how scope and priority reach a limiter at all. That makes it load bearing
rather than a convenience: it is the difference between multi tenant limiting
being realistic and being theoretical.

    @app.middleware("http")
    async def set_scope(request, call_next):
        with scope_context(f"tenant:{request.state.tenant}"):
            return await call_next(request)

Every model call anywhere beneath that middleware is now correctly scoped, and
no function in between has to know that this library exists.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace

from spillway.core.scope import Priority, Scope


@dataclass(frozen=True)
class CallContext:
    """What the surrounding code said about the calls made inside it.

    Attributes:
        scope: Whose budget calls here draw on.
        priority: How urgent they are.
        tags: What the estimator should route on.

    Example:
        >>> with scope_context("tenant:acme", priority=Priority.INTERACTIVE):
        ...     found = current()
        >>> found.scope.key, found.priority
        ('tenant:acme', 100)
    """

    scope: Scope | None = None
    priority: int | None = None
    tags: Mapping[str, str] = field(default_factory=dict)


_EMPTY = CallContext()

_current: ContextVar[CallContext] = ContextVar("spillway_call_context", default=_EMPTY)


def current() -> CallContext:
    """What the surrounding code said, or an empty context if it said nothing.

    Example:
        >>> current().scope is None
        True
    """
    return _current.get()


@contextmanager
def scope_context(
    scope: str | Scope | None = None,
    *,
    priority: int | Priority | None = None,
    tags: Mapping[str, str] | None = None,
) -> Iterator[CallContext]:
    """Apply a scope, a priority and tags to every call made inside this block.

    Works across `await`, across tasks started inside it, and across any number
    of frames, because it is a context variable rather than an argument.

    Nesting adds rather than replaces. An inner block that names only a
    priority keeps the scope the outer one set, which is what makes it
    reasonable to set a tenant once at the edge of a request and mark one
    particular call as batch work deep inside it.

    Args:
        scope: Whose budget calls here draw on.
        priority: How urgent they are. Negative means sheddable.
        tags: What the estimator should route on, such as
            `{"task": "summarise"}`.

    Example:
        >>> with scope_context("tenant:acme"):
        ...     with scope_context(priority=Priority.BATCH) as inner:
        ...         pass
        >>> inner.scope.key, inner.priority
        ('tenant:acme', -100)

        And it is restored on the way out, including when the block raises.

        >>> current().scope is None
        True
    """
    outer = _current.get()
    merged = replace(
        outer,
        scope=Scope.of(scope) if scope is not None else outer.scope,
        priority=int(priority) if priority is not None else outer.priority,
        tags={**outer.tags, **tags} if tags else outer.tags,
    )
    token = _current.set(merged)
    try:
        yield merged
    finally:
        _current.reset(token)
