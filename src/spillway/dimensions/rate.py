"""Limits on consumption over a rolling window.

Requests per minute, input tokens per minute, tokens per day. These are the
limits providers publish, and they are the ones a caller is most likely to hit
first.
"""

from __future__ import annotations

from spillway.core.cost import Cost, Meter
from spillway.core.errors import ConfigurationError
from spillway.core.scope import Scope
from spillway.dimensions.base import claim_key
from spillway.stores.base import Claim, ClaimKind, Delta

_MS_PER_SECOND = 1000.0


class Rate:
    """A limit on how much of something may be consumed per window.

    Args:
        name: What this limit is called. Providers publish names like `rpm`
            and `input_tpm`, and using the same name makes an explanation
            readable against the provider's own documentation.
        limit: How many units per window.
        meter: Which part of a request's cost this limit counts.
        window: The window in seconds. Sixty for a per minute limit.
        adaptive: Must be False. Present so that passing it produces an
            explanation rather than an unexpected keyword error.

    Example:
        >>> from spillway.core.cost import Cost
        >>> from spillway.core.scope import Scope
        >>> tokens = Rate("input_tpm", limit=400_000, meter="input_tokens")
        >>> claim = tokens.claim(Cost(input_tokens=8_600), Scope("tenant:acme"))
        >>> claim.key, claim.cost, claim.window_ms
        ('tenant:acme:input_tpm', 8600.0, 60000.0)

        Settling with a smaller real cost gives the difference back.

        >>> delta = tokens.settle(
        ...     Cost(input_tokens=8_600), Cost(input_tokens=8_200), Scope("tenant:acme")
        ... )
        >>> delta.amount
        400.0

    Raises:
        ConfigurationError: if the limit or window is not positive, or if
            adaptive control was requested.
    """

    kind = ClaimKind.RATE

    def __init__(
        self,
        name: str,
        *,
        limit: float,
        meter: Meter,
        window: float = 60.0,
        adaptive: bool = False,
    ) -> None:
        """Configure the limit, refusing a combination that cannot work."""
        if adaptive:
            message = (
                f"Rate dimensions cannot be adaptive. A provider's published limit is a "
                f"fact, not a hypothesis to probe, and searching for it means deliberately "
                f"exceeding it. Remove adaptive=True from Rate({name!r}), or make a "
                f"Concurrency dimension adaptive instead."
            )
            raise ConfigurationError(message)
        if limit <= 0:
            message = (
                f"Rate({name!r}) needs a positive limit, got {limit}. A limit of zero "
                f"admits nothing at all, which is unlikely to be what was meant."
            )
            raise ConfigurationError(message)
        if window <= 0:
            message = f"Rate({name!r}) needs a positive window in seconds, got {window}."
            raise ConfigurationError(message)
        self._name = name
        self._limit = float(limit)
        self._meter: Meter = meter
        self._window_ms = float(window) * _MS_PER_SECOND

    @property
    def name(self) -> str:
        """What this limit is called."""
        return self._name

    @property
    def limit(self) -> float:
        """How many units per window."""
        return self._limit

    @property
    def meter(self) -> Meter:
        """Which part of a request's cost this limit counts."""
        return self._meter

    @property
    def window_ms(self) -> float:
        """The window, in milliseconds."""
        return self._window_ms

    def __repr__(self) -> str:
        """Show the configuration, so a limiter prints readably."""
        return (
            f"Rate({self._name!r}, limit={self._limit}, meter={self._meter!r}, "
            f"window={self._window_ms / _MS_PER_SECOND})"
        )

    def claim(self, cost: Cost, scope: Scope) -> Claim | None:
        """Claim whatever part of `cost` this limit meters."""
        return Claim(
            key=claim_key(scope, self._name),
            kind=ClaimKind.RATE,
            cost=float(cost.metered(self._meter)),
            limit=self._limit,
            window_ms=self._window_ms,
        )

    def settle(self, reserved: Cost, actual: Cost, scope: Scope) -> Delta | None:
        """Give back the difference, or record it as owed if it went the other way."""
        difference = reserved.metered(self._meter) - actual.metered(self._meter)
        return Delta(
            key=claim_key(scope, self._name),
            kind=ClaimKind.RATE,
            amount=float(difference),
        )
