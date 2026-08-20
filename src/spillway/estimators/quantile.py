"""Predict output length from what a route has actually produced.

The intellectual core of the library, and it is not complicated. Keep the recent
output lengths for each route. Reserve the point that most of them came in
under. Settle the truth, hand the difference back at once, and correct from the
error.

Nothing here claims the prediction is accurate, and nothing here ever should.
Output length is not knowable in advance and this does not make it knowable. The
claim is narrower and it is the one that matters: when the prediction is wrong,
the cost is a little wasted headroom for the length of one request, never an
overrun that breaks a limit and never a deadlock.

The leverage is the route key rather than any cleverness in here. Output length
is close to unpredictable across every call to a model and quite predictable
within one task, so grouping by what a request is for is worth more than any
amount of arithmetic over a group that mixes classification with report writing.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable
from dataclasses import dataclass

from spillway.core.cost import RESERVATION_QUANTILE, Distribution, Estimate, count_input
from spillway.core.errors import ConfigurationError
from spillway.estimators.base import Estimator, Observation, RequestContext
from spillway.estimators.max_tokens import MaxTokensEstimator

DEFAULT_HISTORY = 1_000
"""How many recent output lengths to keep per route.

Enough that a ninth decile means something, and few enough that the memory is an
afterthought at any sane route count. It is a recency window as much as a
sample: output length distributions drift as prompts and models change, and a
history that never forgot would answer today's question with last quarter's
traffic.
"""

# ponytail: a bounded ring per route, with the quantile computed by sorting.
# Memory is order of a thousand integers per route, so a deployment with many
# thousands of distinct routes is the trigger to reconsider. A streaming
# quantile sketch is the upgrade, and it is worth it only at that cardinality:
# at these sample sizes a ring is more accurate, not less.
DEFAULT_MIN_SAMPLES = 30
"""How many observations a route needs before its own history is trusted.

Below this, something else answers. A measurement that does not exist yet must
not bind: reading a ninth decile off four samples would hold back traffic on the
strength of almost nothing, which is worse than the safe and expensive answer.
"""


def _by_model(context: RequestContext) -> Hashable:
    """Group by the model alone.

    Deliberately weak. It is the only grouping available without knowing
    anything about the caller's system, and it mixes every kind of request that
    happens to share a model, which is exactly the mixture that makes output
    length look unpredictable. Pass a `route_key` that names the task and the
    same machinery does considerably better.
    """
    return context.model


@dataclass
class _Route:
    """What is remembered about one route."""

    ring: deque[int]
    quantile: float


class QuantileEstimator:
    """Reserves the point of a route's own history that most requests come in under.

    Args:
        quantile: Which point of the history to reserve. See the table below.
        route_key: Groups requests whose output lengths belong together.
            Defaults to the model alone, which is weak on purpose.
        min_samples: How many observations a route needs before its own history
            is used. Below it, `fallback` answers.
        fallback: What answers below the threshold. Defaults to reserving the
            maximum the caller allowed, which is safe and expensive and exactly
            right when nothing is known yet. Any estimator will do, so
            `StaticEstimator` is the way to say "until you know better, reserve
            five hundred".
        history: How many recent output lengths to keep per route.

    Choosing the quantile, which is the question everyone asks first:

    | Quantile | Overruns | Wasted headroom | Use it when |
    |---|---|---|---|
    | 0.50 | About half | Minimal | Never. Constant overrun defeats the limit |
    | 0.90 | One in ten | Moderate, returned fast | The default |
    | 0.99 | One in a hundred | Large on a heavy tail | A rate limit response is expensive |
    | 1.00 | None | The same as the maximum | You have no history at all |

    Output length distributions are heavy tailed, so the ninety ninth percentile
    is often many multiples of the ninetieth and raising the quantile costs in a
    way that is not linear. The ninth decile, with the surplus credited back the
    moment the real figure is known, is the operating point worth defaulting to.

    Example:
        >>> estimator = QuantileEstimator(
        ...     route_key=lambda ctx: ctx.tags.get("task"),
        ...     min_samples=5,
        ... )
        >>> context = RequestContext(max_tokens=4_096, tags={"task": "summarise"})

        With no history, it reserves what the caller allowed.

        >>> estimator.estimate(context).output.value
        4096

        After watching that route produce short answers, it reserves far less.

        >>> from spillway.core.cost import Cost
        >>> for length in [300, 320, 340, 380, 4_100]:
        ...     estimator.record(
        ...         Observation(
        ...             context=context,
        ...             reserved=Cost(output_tokens=4_096),
        ...             actual=Cost(output_tokens=length),
        ...             at_ms=0.0,
        ...         )
        ...     )
        >>> estimate = estimator.estimate(context)
        >>> estimate.output.quantile(estimate.quantile)
        2612
    """

    def __init__(
        self,
        *,
        quantile: float = RESERVATION_QUANTILE,
        route_key: Callable[[RequestContext], Hashable] = _by_model,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        fallback: Estimator | None = None,
        history: int = DEFAULT_HISTORY,
    ) -> None:
        """Learn per route, reserving at `quantile`.

        Raises:
            ConfigurationError: if `quantile` is outside [0, 1], if
                `min_samples` is negative, if `history` is below one, or if
                `min_samples` is above `history`, which would mean the
                threshold could never be reached.
        """
        if not 0.0 <= quantile <= 1.0:
            message = (
                f"quantile is which point of a route's history to reserve, so it must be "
                f"between 0 and 1 inclusive, got {quantile}. Use 0.9 to reserve what nine "
                f"requests in ten come in under."
            )
            raise ConfigurationError(message)
        if history < 1:
            message = (
                f"history is how many recent output lengths to remember per route, so it "
                f"must be at least one, got {history}. Use the default of "
                f"{DEFAULT_HISTORY:,} unless there is a reason not to."
            )
            raise ConfigurationError(message)
        if min_samples < 0:
            message = (
                f"min_samples is how many observations a route needs before its own "
                f"history is used, so it cannot be negative, got {min_samples}. Use 0 to "
                f"trust a route from its very first observation."
            )
            raise ConfigurationError(message)
        if min_samples > history:
            message = (
                f"min_samples is {min_samples:,} and history is {history:,}, so a route "
                f"would forget its oldest observation before it ever reached the "
                f"threshold and its own history would never be used at all. Lower "
                f"min_samples, or raise history."
            )
            raise ConfigurationError(message)
        self._quantile = quantile
        self._route_key = route_key
        self._min_samples = min_samples
        self._history = history
        self._routes: dict[Hashable, _Route] = {}
        self._fallback: Estimator = fallback if fallback is not None else MaxTokensEstimator()

    def __repr__(self) -> str:
        """Show the quantile and how many routes have been seen."""
        return f"QuantileEstimator(quantile={self._quantile}, routes={len(self._routes)})"

    def samples(self, context: RequestContext) -> int:
        """How many observations the route `context` belongs to has collected."""
        route = self._routes.get(self._route_key(context))
        return 0 if route is None else len(route.ring)

    def estimate(self, context: RequestContext) -> Estimate:
        """Reserve this route's own quantile, or defer below the threshold.

        Deferring rather than guessing, because a measurement that does not
        exist yet must not bind. Reading a ninth decile off four samples would
        hold back traffic on the strength of almost nothing, and being
        confidently wrong is worse here than being safe and expensive.
        """
        route = self._routes.get(self._route_key(context))
        if route is None or len(route.ring) < max(self._min_samples, 1):
            return self._fallback.estimate(context)
        return Estimate(
            input=count_input(context.prompt),
            output=Distribution.empirical(route.ring),
            model=context.model,
            quantile=route.quantile,
        )

    def record(self, observation: Observation) -> None:
        """Remember what this request really produced."""
        route = self._route_for(observation.context)
        route.ring.append(observation.actual.output_tokens)

    def _route_for(self, context: RequestContext) -> _Route:
        """The state for `context`'s route, created on first sight."""
        key = self._route_key(context)
        route = self._routes.get(key)
        if route is None:
            route = _Route(ring=deque(maxlen=self._history), quantile=self._quantile)
            self._routes[key] = route
        return route
