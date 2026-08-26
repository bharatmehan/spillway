"""How a request's cost is predicted, and how the prediction is corrected.

Input tokens are countable before a call and output tokens are not. An estimator
predicts a distribution, the limiter reserves a quantile of it, settlement
reports the truth, and the difference goes back immediately.

Nothing here claims the prediction is accurate. The claim is that being wrong
costs a little wasted headroom for the length of one request, and never an
overrun that breaks a limit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from spillway.core.cost import Cost, Estimate
from spillway.core.scope import Scope


@dataclass(frozen=True)
class RequestContext:
    """Everything known about a request before it is made.

    What an estimator is given, and what a route key callable receives. Frozen,
    because it is handed to user supplied code and then kept.

    Attributes:
        prompt: A string, or a sequence of message mappings in the shape the
            provider SDKs use.
        max_tokens: The output limit the caller asked for, when they named one.
        model: Which model this is going to, when known.
        scope: Whose budget the request draws on.
        tags: Whatever the caller wants to route on. Output length is far more
            predictable within one task than across all of them, so a tag
            naming the task is where the leverage is.

    Example:
        >>> context = RequestContext(
        ...     prompt="Summarise this document.",
        ...     max_tokens=2_000,
        ...     model="claude-sonnet-4-5",
        ...     tags={"task": "summarise"},
        ... )
        >>> (context.model, context.tags["task"])
        ('claude-sonnet-4-5', 'summarise')
    """

    prompt: str | Sequence[object] | None = None
    max_tokens: int | None = None
    model: str | None = None
    scope: Scope = field(default_factory=lambda: Scope.of(None))
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """What one settled request taught, for whoever is learning.

    The gap between `reserved` and `actual` is the estimate error. An estimator
    that learns reads the actual output length out of it; one that does not
    ignores the whole record.

    Attributes:
        context: What was known before the call, unchanged since admission, so
            a route key computed now matches the one computed then.
        reserved: What admission took, at whatever quantile applied.
        actual: What the request really cost.
        at_ms: When it settled, on the limiter's clock.

    Example:
        >>> observation = Observation(
        ...     context=RequestContext(model="claude-sonnet-4-5"),
        ...     reserved=Cost(input_tokens=12_400, output_tokens=1_180),
        ...     actual=Cost(input_tokens=12_400, output_tokens=415),
        ...     at_ms=6_200.0,
        ... )
        >>> observation.reserved.output_tokens - observation.actual.output_tokens
        765
    """

    context: RequestContext
    reserved: Cost
    actual: Cost
    at_ms: float


@runtime_checkable
class Estimator(Protocol):
    """Predicts a request's cost, and is told afterwards what it really was.

    Two methods, and `record` does nothing on the estimators that do not learn.
    It sits here rather than on a separate learning interface so swapping a
    fixed estimator for a learning one is a constructor argument.

    Implement it structurally: an object with these two methods is an estimator.

    Example:
        >>> from spillway.core.cost import Distribution
        >>> class AlwaysTwenty:
        ...     def estimate(self, context: RequestContext) -> Estimate:
        ...         return Estimate(input=0, output=Distribution.point(20))
        ...
        ...     def record(self, observation: Observation) -> None:
        ...         pass
        >>> isinstance(AlwaysTwenty(), Estimator)
        True
    """

    def estimate(self, context: RequestContext) -> Estimate:
        """Predict what `context` will cost."""
        ...

    def record(self, observation: Observation) -> None:
        """Take note of what a settled request really cost.

        Called once per settlement, whether or not this estimator produced the
        reservation. An explicit estimate still generated real output on that
        route, and ignoring those blinds an estimator when a caller mixes both.
        """
        ...
