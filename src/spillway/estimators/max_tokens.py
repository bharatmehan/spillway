"""Reserve whatever output length the caller said they would allow.

The safe, uninformed baseline, and the default until an estimator has a history
to read. Correct and expensive: a caller allowing 4,096 tokens whose median
answer is 300 holds roughly thirteen times what it uses, for the whole call.

It is also the permanently right answer against a provider that charges the
requested maximum at admission, where reserving less buys nothing and guarantees
a rate limit response nobody predicted.
"""

from __future__ import annotations

from spillway.core.cost import Estimate, default_estimate
from spillway.estimators.base import Observation, RequestContext


class MaxTokensEstimator:
    """Reserves the requested output maximum, and learns nothing.

    Input is counted with the character heuristic, so it is approximate, and the
    real figure replaces it at settlement.

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
