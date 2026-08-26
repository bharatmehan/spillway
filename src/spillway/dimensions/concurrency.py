"""A limit on how many requests may be in flight at once.

The simplest gauge, and often the one that actually binds: a tokens per minute
limit can be comfortably clear while every request queues behind another,
because what is scarce is the capacity generating the responses.
"""

from __future__ import annotations

from spillway.core.cost import Cost
from spillway.core.errors import ConfigurationError
from spillway.core.scope import Scope
from spillway.dimensions.base import claim_key
from spillway.stores.base import Claim, ClaimKind, Delta


class Concurrency:
    """A limit on simultaneous in flight requests.

    One unit is taken at admission and given back at settlement. Size does not
    matter: one request occupies one slot whether it generates ten tokens or ten
    thousand.

    Args:
        name: What this limit is called.
        limit: How many requests may be in flight at once.
        adaptive: Must be False for now. Present so that passing it explains
            itself rather than raising an unexpected keyword error.

    Example:
        >>> from spillway.core.cost import Cost
        >>> from spillway.core.scope import Scope
        >>> generations = Concurrency("generations", limit=64)
        >>> claim = generations.claim(Cost(input_tokens=8_600), Scope("tenant:acme"))
        >>> claim.key, claim.cost, claim.limit
        ('tenant:acme:generations', 1.0, 64.0)

        The slot comes back whole, however wrong the estimate was.

        >>> generations.settle(
        ...     Cost(output_tokens=1_180), Cost(output_tokens=9_999), Scope("tenant:acme")
        ... ).amount
        1.0

    Raises:
        ConfigurationError: if the limit is not positive, or if adaptive
            control was requested.
    """

    kind = ClaimKind.GAUGE

    def __init__(self, name: str, *, limit: float, adaptive: bool = False) -> None:
        """Configure the limit, refusing a combination that cannot work."""
        if adaptive:
            message = (
                f"Concurrency({name!r}) cannot discover its own limit in this version, so "
                f"adaptive=True would be ignored silently. Remove it and set an explicit "
                f"limit for now."
            )
            raise ConfigurationError(message)
        if limit <= 0:
            message = (
                f"Concurrency({name!r}) needs a positive limit, got {limit}. A limit of "
                f"zero admits nothing at all, which is unlikely to be what was meant."
            )
            raise ConfigurationError(message)
        self._name = name
        self._limit = float(limit)

    @property
    def name(self) -> str:
        """What this limit is called."""
        return self._name

    @property
    def limit(self) -> float:
        """How many requests may be in flight at once."""
        return self._limit

    def __repr__(self) -> str:
        """Show the configuration."""
        return f"Concurrency({self._name!r}, limit={self._limit})"

    def claim(self, cost: Cost, scope: Scope) -> Claim | None:  # noqa: ARG002
        """Claim one slot, whatever the request is expected to cost."""
        return Claim(
            key=claim_key(scope, self._name),
            kind=ClaimKind.GAUGE,
            cost=1.0,
            limit=self._limit,
        )

    def settle(
        self,
        reserved: Cost,  # noqa: ARG002
        actual: Cost,  # noqa: ARG002
        scope: Scope,
    ) -> Delta | None:
        """Give the slot back in full.

        Unlike a rate limit there is no difference to reconcile: the slot was
        occupied and now it is not, however wrong the estimate was.
        """
        return Delta(key=claim_key(scope, self._name), kind=ClaimKind.GAUGE, amount=1.0)
