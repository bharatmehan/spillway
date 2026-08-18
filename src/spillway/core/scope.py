"""Who a request is accounted to, and how urgently it wants to run.

Scope and priority are the two things a caller classifies a request by. Every
limit is tracked per scope, and priority decides who goes first once there is
a queue to order.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


@dataclass(frozen=True)
class Scope:
    """The key that limits and fairness are tracked against.

    A tenant, a user, an API key, an agent task: whatever the unit of isolation
    is for the caller. Requests in different scopes consume separate budgets on
    every dimension.

    This is a value object over a string rather than a bare string because it
    gains a fair share weight and a parent scope later, and widening a string
    parameter into a type afterwards is a change every caller has to notice.

    Example:
        >>> Scope("tenant:acme").key
        'tenant:acme'
        >>> Scope.of("user:123") == Scope("user:123")
        True
        >>> Scope.of(None) is DEFAULT_SCOPE
        True
    """

    key: str

    def __post_init__(self) -> None:
        """Reject a blank key, which would silently merge every caller's budget."""
        if not self.key.strip():
            message = (
                "A scope key cannot be blank. Pass a key that identifies the caller, "
                'such as "tenant:acme", or pass nothing to use the default scope.'
            )
            raise ValueError(message)

    def __str__(self) -> str:
        """Return the key, so a scope reads plainly in a message or a store key."""
        return self.key

    @classmethod
    def of(cls, value: str | Scope | None) -> Scope:
        """Coerce whatever a caller passed into a scope.

        Accepting a plain string at the public boundary keeps the common case a
        single word, while everything inside the library works with the type.
        """
        if value is None:
            return DEFAULT_SCOPE
        if isinstance(value, Scope):
            return value
        return cls(value)


DEFAULT_SCOPE = Scope("global")
"""The scope used when a caller names none. One shared budget for everything."""


class Priority(IntEnum):
    """How urgently a request wants to be admitted. Higher goes first.

    These four values are a convention, not a closed set. Any integer is
    accepted, so a caller with finer bands can use their own numbers.

    A negative priority means the work is sheddable: under saturation it may be
    dropped rather than queued, on the grounds that whoever submitted it said
    it can wait, and something that can wait can also not happen.

    Example:
        >>> Priority.INTERACTIVE > Priority.NORMAL > Priority.BATCH
        True
        >>> int(Priority.BATCH) < 0
        True
    """

    INTERACTIVE = 100
    """A person is waiting for this response."""

    NORMAL = 0
    """The default. Nothing special is claimed."""

    BACKGROUND = -50
    """Useful work with no one waiting. Sheddable."""

    BATCH = -100
    """Bulk work, wanted eventually. The first thing dropped under pressure."""
