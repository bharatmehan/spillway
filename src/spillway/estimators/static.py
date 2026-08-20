"""Reserve a fixed output length, because this route really is predictable.

Not every workload needs a distribution. A classifier that answers with one of
five labels, or an extractor filling a fixed schema, produces very nearly the
same number of tokens every time, and a history of a thousand identical samples
would tell nobody anything they did not already know.
"""

from __future__ import annotations

from spillway.core.cost import Distribution, Estimate, count_input
from spillway.estimators.base import Observation, RequestContext


class StaticEstimator:
    """Reserves the same output prediction for every request, and learns nothing.

    Input is still counted per request with the character heuristic. Only the
    output prediction is fixed, because that is the only part that cannot be
    counted.

    Args:
        output: What to predict, every time.

    Example:
        >>> estimator = StaticEstimator(output=Distribution.point(120))
        >>> estimate = estimator.estimate(RequestContext(prompt="label this"))
        >>> estimate.output.quantile(estimate.quantile)
        120
    """

    def __init__(self, *, output: Distribution) -> None:
        """Predict `output` for everything."""
        self._output = output

    def __repr__(self) -> str:
        """Show what it always predicts."""
        return f"StaticEstimator(output={self._output!r})"

    def estimate(self, context: RequestContext) -> Estimate:
        """Predict the configured output, and count the input as usual."""
        return Estimate(
            input=count_input(context.prompt),
            output=self._output,
            model=context.model,
        )

    def record(self, observation: Observation) -> None:
        """Ignore it. The prediction is fixed by the caller, on purpose."""
