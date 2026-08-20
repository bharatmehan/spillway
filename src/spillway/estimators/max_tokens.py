"""Reserve whatever output length the caller said they would allow.

The safe, uninformed baseline, and the default until an estimator has a history
to read. It is correct and it is expensive: if a caller allows 4,096 tokens and
the median answer is 300, it holds roughly thirteen times what is used for the
whole length of the call.

It is also the right answer, permanently, against a provider that charges the
requested maximum against its own limits at admission. Reserving less than the
provider does buys nothing and guarantees a rate limit response nobody predicted.
"""

from __future__ import annotations

from spillway.core.cost import Estimate, default_estimate
from spillway.estimators.base import Observation, RequestContext


class MaxTokensEstimator:
    """Reserves the requested output maximum, and learns nothing.

    Input is counted with the character heuristic, so it is approximate and
    documented as such, and the real figure replaces it at settlement.

    Example:
        >>> estimator = MaxTokensEstimator()
        >>> context = RequestContext(prompt="hello there", max_tokens=256)
        >>> estimate = estimator.estimate(context)
        >>> estimate.input, estimate.output.quantile(estimate.quantile)
        (4, 256)

        With no maximum named, a flat default stands in.

        >>> estimator.estimate(RequestContext()).output.value
        1024
    """

    def __repr__(self) -> str:
        """Show that it takes no configuration."""
        return "MaxTokensEstimator()"

    def estimate(self, context: RequestContext) -> Estimate:
        """Reserve the requested maximum, or the flat default when none was named."""
        return default_estimate(
            context.prompt,
            max_tokens=context.max_tokens,
            model=context.model,
        )

    def record(self, observation: Observation) -> None:
        """Ignore it. Nothing here learns, so there is nothing to record."""
