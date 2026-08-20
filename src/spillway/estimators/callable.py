"""The escape hatch, for anyone who can predict output length better than this.

The literature has real output length predictors, some of them fine tuned models
in their own right. This library does not ship one and will not: that is a
research artefact with a heavy dependency tree, and the promise that a
quickstart runs with nothing installed matters more.

What it ships instead is the socket. Wrap any function from a request context to
a distribution and it becomes an estimator.
"""

from __future__ import annotations

from collections.abc import Callable

from spillway.core.cost import Distribution, Estimate, count_input
from spillway.estimators.base import Observation, RequestContext


class CallableEstimator:
    """Predicts output length by calling whatever you give it.

    The function is called once per admission, on the calling task, before
    anything is reserved. That means it has to be quick and it must not perform
    any input or output: a prediction that blocks would add its own latency to
    every request the limiter admits, including the ones with capacity waiting.

    Args:
        predict: Called with the request context, returns the predicted output
            length distribution.
        quantile: Which point of that distribution to reserve. Left alone it is
            the ninth decile, the same as everywhere else.

    Example:
        >>> def guess(context: RequestContext) -> Distribution:
        ...     return Distribution.point(40 if context.tags.get("task") == "label" else 800)
        >>> estimator = CallableEstimator(guess)
        >>> estimate = estimator.estimate(RequestContext(tags={"task": "label"}))
        >>> estimate.output.quantile(estimate.quantile)
        40
    """

    def __init__(
        self,
        predict: Callable[[RequestContext], Distribution],
        *,
        quantile: float | None = None,
    ) -> None:
        """Predict with `predict`, reserving at `quantile`."""
        self._predict = predict
        self._quantile = quantile

    def __repr__(self) -> str:
        """Name the function doing the predicting."""
        name = getattr(self._predict, "__name__", repr(self._predict))
        return f"CallableEstimator({name})"

    def estimate(self, context: RequestContext) -> Estimate:
        """Ask the wrapped function, and count the input as usual."""
        output = self._predict(context)
        if self._quantile is None:
            return Estimate(input=count_input(context.prompt), output=output, model=context.model)
        return Estimate(
            input=count_input(context.prompt),
            output=output,
            model=context.model,
            quantile=self._quantile,
        )

    def record(self, observation: Observation) -> None:
        """Ignore it. Whatever is learning lives inside the wrapped function."""
