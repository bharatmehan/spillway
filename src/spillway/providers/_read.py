"""Reading fields off whatever a provider client hands back.

A response is a typed object from a client library, or a plain mapping from a
test or a raw HTTP call, or an object from a library version that has since
renamed something. An adapter should not care which, and it must never raise on
the shape rather than on the substance: a missing usage field means settle at
the reserved amount and say so, not crash a request that already succeeded.
"""

from __future__ import annotations

from collections.abc import Mapping


def field(source: object, name: str) -> object | None:
    """Read `name` off a mapping or an object, whichever this is.

    Example:
        >>> field({"input_tokens": 12}, "input_tokens")
        12
        >>> class Usage:
        ...     input_tokens = 12
        >>> field(Usage(), "input_tokens")
        12
        >>> field({"a": 1}, "missing") is None
        True
    """
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def path(source: object, *names: str) -> object | None:
    """Read a nested field, stopping at the first step that is not there.

    Example:
        >>> path({"usage": {"details": {"cached": 7}}}, "usage", "details", "cached")
        7
        >>> path({"usage": None}, "usage", "details") is None
        True
    """
    current = source
    for name in names:
        if current is None:
            return None
        current = field(current, name)
    return current


def count(value: object) -> int:
    """Read a token count, treating anything unusable as zero.

    Providers send `null` for a category that did not apply, and a count that
    is absent is a count of none. A negative one is not something to propagate
    into accounting.

    Example:
        >>> count(1024), count(None), count("48"), count(-5), count("many")
        (1024, 0, 48, 0, 0)
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(value.strip()))
        except ValueError:
            return 0
    return 0


def usage_of(response: object) -> object | None:
    """Find the usage record on a response, wherever this provider put it.

    Accepts the record itself, so a caller who already has one can pass it
    straight in.

    Example:
        >>> usage_of({"usage": {"input_tokens": 3}})
        {'input_tokens': 3}
        >>> usage_of({"input_tokens": 3})
        {'input_tokens': 3}
        >>> usage_of(None) is None
        True
    """
    if response is None:
        return None
    found = field(response, "usage")
    if found is not None:
        return found
    # Already a usage record, which is what a fixture and a raw settlement both
    # hand over. Recognised by carrying any of the names a provider counts in.
    for name in ("input_tokens", "prompt_tokens", "output_tokens", "completion_tokens"):
        if field(response, name) is not None:
            return response
    return None
