"""Admitting around a function, when there is no client to instrument.

Reach for this third. An instrumented client covers every call through it with
nothing at the call sites, `admit()` covers a single call where you want the
lease in your hand, and this covers what neither does: a model reached through
something unrecognised, or work worth limiting whole rather than per request.

    @admitted(limiter, max_tokens=2_000)
    async def summarise(document: str) -> str:
        ...

If there is a client to instrument, instrument it instead. This has to be
written at every function it applies to.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, TypeVar

from spillway.core.errors import ConfigurationError
from spillway.core.scope import Priority, Scope
from spillway.integrations.context import current

if TYPE_CHECKING:  # pragma: no cover - imported for typing only, to avoid a cycle
    from spillway.core.spillway import Spillway

_log = logging.getLogger(__name__)

_warned_about_unreadable_return = False

ResultT = TypeVar("ResultT")

ScopeSource = str | Scope | Callable[..., str | Scope]


def admitted(
    limiter: Spillway,
    *,
    scope: ScopeSource | None = None,
    priority: int | Priority | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> Callable[[Callable[..., Awaitable[ResultT]]], Callable[..., Awaitable[ResultT]]]:
    """Admit before the wrapped function runs, and settle after it returns.

    Args:
        limiter: What admits the call.
        scope: Whose budget this draws on. A string, or a callable over the
            wrapped function's own arguments, for the common case where the
            tenant is one of them.
        priority: How urgent this is.
        max_tokens: What to reserve, when the function knows and the limiter
            cannot.
        model: Recorded on the estimate.
        tags: What the estimator should route on.

    Raises:
        ConfigurationError: if the decorated function is not asynchronous.

    Settlement reads the return value through the limiter's provider, so a
    function returning the provider's own response settles exactly. One
    returning something else settles at the reserved amount and says so once,
    which is safe and expensive.

    Example:
        >>> import asyncio
        >>> from spillway.core.spillway import Spillway
        >>> limiter = Spillway(provider="anthropic")
        >>> @admitted(limiter, max_tokens=500, tags={"task": "summarise"})
        ... async def summarise(document: str) -> dict:
        ...     return {"usage": {"input_tokens": 120, "output_tokens": 48}}
        >>> asyncio.run(summarise("a long document"))
        {'usage': {'input_tokens': 120, 'output_tokens': 48}}
    """

    def decorate(function: Callable[..., Awaitable[ResultT]]) -> Callable[..., Awaitable[ResultT]]:
        if not inspect.iscoroutinefunction(inspect.unwrap(function)):
            message = (
                f"{function.__name__}() is not an async function, and admission has to "
                f"wait for capacity, which needs a synchronous driver that does not "
                f"exist yet. Make it async, or call spillway.admit() inside it yourself."
            )
            raise ConfigurationError(message)

        @functools.wraps(function)
        async def wrapper(*args: object, **kwargs: object) -> ResultT:
            surrounding = current()
            chosen = _scope_for(scope, args, kwargs)
            if chosen is None:
                chosen = surrounding.scope
            async with limiter.admit(
                scope=chosen,
                priority=_priority_for(priority, surrounding.priority),
                max_tokens=max_tokens,
                model=model,
                tags={**surrounding.tags, **(tags or {})},
            ) as lease:
                result = await function(*args, **kwargs)
                _settle(lease, result, limiter)
                return result

        return wrapper

    return decorate


def _scope_for(
    source: ScopeSource | None,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> Scope | None:
    """Work out the scope, calling it with the wrapped arguments if it is a callable."""
    if source is None:
        return None
    if callable(source):
        return Scope.of(source(*args, **kwargs))
    return Scope.of(source)


def _priority_for(given: int | Priority | None, surrounding: int | None) -> int:
    """The decorator's priority, then the surrounding one, then the default."""
    if given is not None:
        return int(given)
    if surrounding is not None:
        return surrounding
    return int(Priority.NORMAL)


def _settle(lease: object, result: object, limiter: Spillway) -> None:
    """Settle from the return value, or at the reserved amount if it cannot be read."""
    settle_from = getattr(lease, "settle_from", None)
    if limiter.provider is not None and callable(settle_from):
        try:
            settle_from(result)
        except Exception:
            # Deliberately wide. The wrapped call has already succeeded, and no
            # failure to read its return value is worth losing the caller's
            # result over. Settling at the reservation is safe and expensive,
            # and the warning says which of those it is.
            _warn_once_about_unreadable_return()
            _settle_at_reserved(lease)
        return
    _settle_at_reserved(lease)


def _settle_at_reserved(lease: object) -> None:
    """Charge the whole reservation, which is safe and wasteful."""
    settle = getattr(lease, "settle", None)
    reserved = getattr(lease, "reserved", None)
    if callable(settle) and reserved is not None:
        settle(input=reserved.input_tokens, output=reserved.output_tokens)


def _warn_once_about_unreadable_return() -> None:
    """Say, once, that a return value told us nothing about what it cost."""
    global _warned_about_unreadable_return
    if _warned_about_unreadable_return:
        return
    _warned_about_unreadable_return = True
    _log.warning(
        "A decorated function returned something the provider could not read usage "
        "from, so the full reserved amount was charged rather than what the call "
        "really cost. That is safe but expensive. Return the provider's own response, "
        "or instrument the client instead, which reads it without being asked."
    )
