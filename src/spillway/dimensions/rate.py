"""Limits on consumption over a rolling window.

Requests per minute, input tokens per minute, tokens per day: the limits
providers publish, and the ones a caller hits first.
"""

from __future__ import annotations

from spillway.core.cost import Cost, Meter
from spillway.core.errors import ConfigurationError
from spillway.core.scope import Scope
from spillway.dimensions.base import claim_key
from spillway.stores.base import Claim, ClaimKind, Delta

_MS_PER_SECOND = 1000.0

INFERRED_METERS: dict[str, Meter] = {
    "rpm": "requests",
    "rpd": "requests",
    "input_tpm": "input_tokens",
    "output_tpm": "output_tokens",
    "tpm": "total_tokens",
}
"""Limit names common enough across providers that the meter is unambiguous.

Any other name needs an explicit meter. Guessing beyond this table would meter
the wrong thing silently, and that surfaces much later as unexplained rate limit
responses from the provider.
"""


def _infer_meter(name: str) -> Meter:
    """Work out what a limit counts from its name, or explain why it cannot.

    Raises:
        ConfigurationError: if the name is not one this library recognises.
    """
    meter = INFERRED_METERS.get(name)
    if meter is None:
        known = ", ".join(sorted(INFERRED_METERS))
        message = (
            f"Rate({name!r}) cannot tell what to count from its name, so it needs an "
            f"explicit meter. Pass one of meter='requests', meter='input_tokens', "
            f"meter='output_tokens' or meter='total_tokens'. The meter is inferred only "
            f"for these names: {known}."
        )
        raise ConfigurationError(message)
    return meter


class Rate:
    """A limit on how much of something may be consumed per window.

    Args:
        name: What this limit is called. Use the provider's own name, such as
            `rpm` or `input_tpm`, so an explanation reads against their docs.
        limit: How many units per window.
        meter: Which part of a request's cost this limit counts. Inferred from
            the name for the widely used ones listed in `INFERRED_METERS`, and
            required for anything else.
        window: The window in seconds. Sixty for a per minute limit.
        adaptive: Must be False. Present so that passing it produces an
            explanation rather than an unexpected keyword error.

    Example:
        >>> from spillway.core.cost import Cost
        >>> from spillway.core.scope import Scope
        >>> tokens = Rate("input_tpm", limit=400_000)
        >>> claim = tokens.claim(Cost(input_tokens=8_600), Scope("tenant:acme"))
        >>> claim.key, claim.cost, claim.window_ms
        ('tenant:acme:input_tpm', 8600.0, 60000.0)

        Settling with a smaller real cost gives the difference back.

        >>> delta = tokens.settle(
        ...     Cost(input_tokens=8_600), Cost(input_tokens=8_200), Scope("tenant:acme")
        ... )
        >>> delta.amount
        400.0

        A name outside the table needs to say what it counts.

        >>> Rate("images_per_minute", limit=50, meter="requests").meter
        'requests'

    Raises:
        ConfigurationError: if the meter cannot be determined, if the limit or
            window is not positive, or if adaptive control was requested.
    """

    kind = ClaimKind.RATE

    def __init__(
        self,
        name: str,
        *,
        limit: float,
        window: float = 60.0,
        meter: Meter | None = None,
        adaptive: bool = False,
    ) -> None:
        """Configure the limit, refusing a combination that cannot work."""
        if adaptive:
            message = (
                f"Rate dimensions cannot be adaptive. A provider's published limit is a "
                f"fact, not a hypothesis to probe, and searching for it means deliberately "
                f"exceeding it. Remove adaptive=True from Rate({name!r}) and set the limit "
                f"to the figure the provider publishes."
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
        self._meter: Meter = meter if meter is not None else _infer_meter(name)
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
