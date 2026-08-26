"""What every limitable axis has in common.

A dimension turns a request's cost into a claim on one key, and turns the
difference between reserved and actual into a correction. It does not decide
whether the claim fits, talk to a store, or know about any other dimension.

Whether a claim fits has to be decided for every dimension at once, or a request
is admitted against two limits and refused by the third with the first two
already consumed. So dimensions describe what they want, and something else
applies the whole set or none of it.
"""

from __future__ import annotations

from typing import Protocol

from spillway.core.cost import Cost
from spillway.core.scope import Scope
from spillway.stores.base import Claim, ClaimKind, Delta


def claim_key(scope: Scope, name: str) -> str:
    """Build the store key for one dimension within one scope.

    Every dimension keys the same way, so a store can be handed keys it knows
    nothing about and two dimensions cannot collide.

    Example:
        >>> claim_key(Scope("tenant:acme"), "input_tpm")
        'tenant:acme:input_tpm'
    """
    return f"{scope.key}:{name}"


class Dimension(Protocol):
    """One resource axis that a request can run out of.

    Implement this to limit something this library does not ship. Both methods
    are pure, so a dimension can be asked what it wants without any of it taking
    effect.

    Example:
        A dimension that limits requests carrying an unusually long prompt.

        >>> from spillway.stores.base import Claim, ClaimKind, Delta
        >>> class LongPrompts:
        ...     name = "long_prompts"
        ...     kind = ClaimKind.GAUGE
        ...     limit = 4.0
        ...
        ...     def claim(self, cost: Cost, scope: Scope) -> Claim | None:
        ...         if cost.input_tokens < 100_000:
        ...             return None
        ...         return Claim(claim_key(scope, self.name), self.kind, cost=1.0, limit=self.limit)
        ...
        ...     def settle(self, reserved: Cost, actual: Cost, scope: Scope) -> Delta | None:
        ...         if reserved.input_tokens < 100_000:
        ...             return None
        ...         return Delta(claim_key(scope, self.name), self.kind, amount=1.0)
        >>> dimension: Dimension = LongPrompts()
        >>> dimension.claim(Cost(input_tokens=10), Scope("acme")) is None
        True
        >>> dimension.claim(Cost(input_tokens=200_000), Scope("acme")).cost
        1.0
    """

    @property
    def name(self) -> str:
        """What this axis is called, in configuration and in an explanation."""
        ...

    @property
    def kind(self) -> ClaimKind:
        """Whether this axis is consumed over a window or held and given back."""
        ...

    @property
    def limit(self) -> float:
        """The most this axis allows, in whatever units it counts.

        Read each time rather than cached, because an adaptive limit changes
        underneath.
        """
        ...

    def claim(self, cost: Cost, scope: Scope) -> Claim | None:
        """Say what this request would take on this axis.

        Returns None when the axis does not apply, which is how a dimension
        opts out. A claim of zero would still be evaluated and reported.
        """
        ...

    def settle(self, reserved: Cost, actual: Cost, scope: Scope) -> Delta | None:
        """Say what correction the real cost calls for.

        A positive amount is capacity to give back, a negative one is an
        overrun to be repaid. Returns None when this axis did not apply, which
        must match the decision `claim` made for the same request.
        """
        ...
