"""Predict output length from what a route has actually produced.

Keep the recent output lengths for each route, reserve the point that most of
them came in under, settle the truth, hand the difference back at once, correct
from the error.

Nothing here claims the prediction is accurate. Output length is not knowable in
advance and this does not make it knowable. When the prediction is wrong, the
cost is a little wasted headroom for one request, never an overrun that breaks a
limit and never a deadlock.

The leverage is the route key rather than any arithmetic in here: output length
is far more predictable within one task than across every call to a model, so
grouping by what a request is for beats any amount of cleverness over a group
that mixes classification with report writing.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

from spillway.core.cost import (
    RESERVATION_QUANTILE,
    Distribution,
    Estimate,
    count_input,
)
from spillway.core.errors import ConfigurationError
from spillway.estimators.base import Estimator, Observation, RequestContext
from spillway.estimators.max_tokens import MaxTokensEstimator

# ponytail: a bounded ring per route, with the quantile computed by sorting.
# Memory is order of a thousand integers per route, so a deployment with many
# thousands of distinct routes is the trigger to reconsider. A streaming
# quantile sketch is the upgrade, and it is worth it only at that cardinality:
# at these sample sizes a ring is more accurate, not less.
DEFAULT_HISTORY = 1_000
"""How many recent output lengths to keep per route.

Enough that a ninth decile means something, few enough that the memory is an
afterthought. A recency window as much as a sample: output lengths drift as
prompts and models change, and a history that never forgot would answer today's
question with last quarter's traffic.
"""

DEFAULT_MIN_SAMPLES = 30
"""How many observations a route needs before its own history is trusted.

Below this, the fallback answers. Reading a ninth decile off four samples would
hold back traffic on the strength of almost nothing.
"""


DEFAULT_ADAPT_EVERY = 100
"""How many observations one route collects before its quantile may move.

A quantile that moves per request oscillates, which is worse than a fixed
conservative one. A hundred observations puts the overrun count in double
figures at the ninth decile, enough for the share to mean something.
"""

DEFAULT_ADAPT_STEP = 0.02
"""How far the quantile moves when it moves. Small, and in one direction only."""

DEFAULT_QUANTILE_BOUNDS = (0.5, 0.99)
"""How far the quantile may travel in either direction.

The floor is where overrunning half the time begins. The ceiling is short of one
because these distributions are heavy tailed, so the last percent costs more than
everything below it, and reserving the maximum outright says that more honestly.
"""

STATISTICS_SPAN = 100
"""Roughly how many recent observations the reported ratios describe.

Weighted over this span rather than averaged over a route's whole life. A route
that spent its first thirty requests reserving the maximum would otherwise carry
that period for thousands of requests afterwards, reporting a well calibrated
estimator as wasting several times the headroom it needs.

The ring is already a recency window for the same reason. These follow it.
"""

# ponytail: a fixed band, one and a half times the promised overrun rate to
# raise and half of it to lower, with a flat step. Narrower would move the
# quantile on sampling noise and wider would leave a badly calibrated route
# uncorrected for longer than it should be. Something proportional to the size
# of the disagreement if a real workload shows the flat step converging too
# slowly.
RAISE_ABOVE = 1.5
"""Raise the quantile when overruns exceed this multiple of what it promised."""

LOWER_BELOW = 0.5
"""Lower the quantile when overruns fall below this multiple of what it promised."""


def _by_model(context: RequestContext) -> Hashable:
    """Group by the model alone.

    Deliberately weak. It is the only grouping available without knowing
    anything about the caller's system, and it mixes every kind of request that
    happens to share a model, which is exactly the mixture that makes output
    length look unpredictable. Pass a `route_key` that names the task and the
    same machinery does considerably better.
    """
    return context.model


def _thinned(samples: Sequence[int], keep: int) -> list[int]:
    """Reduce `samples` to at most `keep`, spread evenly across the whole.

    An even stride rather than a head or a tail, because the input is one
    history followed by another and taking either end would discard one side
    entirely.
    """
    if len(samples) <= keep:
        return list(samples)
    stride = math.ceil(len(samples) / keep)
    return list(samples[::stride])[:keep]


@dataclass
class _Route:
    """What is remembered about one route."""

    ring: deque[int]
    quantile: float
    observations: int = 0
    overruns: int = 0
    overrun_ratio: float = 0.0
    error_ratio: float | None = None
    since_adapt: int = 0
    overruns_since_adapt: int = 0

    def weigh_in(self, attribute: str, sample: float) -> None:
        """Fold `sample` into an exponentially weighted average.

        The first sample stands on its own rather than being averaged against a
        zero that never happened. Starting from zero would have every route
        climb up from nothing over its first hundred requests, which reads
        exactly like a route that has only just started behaving.
        """
        alpha = 2.0 / (STATISTICS_SPAN + 1)
        current: float | None = getattr(self, attribute)
        if current is None:
            setattr(self, attribute, sample)
            return
        setattr(self, attribute, current + alpha * (sample - current))


@dataclass(frozen=True)
class RouteStatistics:
    """How well a route's predictions have been doing.

    Attributes:
        samples: How many output lengths the ring is holding right now.
        observations: How many settlements this route has ever reported.
        quantile: What this route is currently reserving at.
        overrun_ratio: The share of recent settlements that used more than was
            reserved. Compare it with one minus the quantile: reserving at the
            ninth decile promises roughly one in ten, and a number far above
            that means the history no longer describes the traffic.
        error_ratio: Reserved divided by actual over recent settlements, or
            None until one of them has generated anything. Around 1.1 for a
            ninth decile reservation is healthy. Around 5 means the reservation
            is nowhere near the traffic and most of the headroom is being
            wasted, which is worth acting on even though nothing is technically
            wrong.

    Both ratios describe recent traffic rather than everything a route has ever
    seen. A route carries its own opening requests in a lifetime average for
    thousands of settlements afterwards, and would report a perfectly calibrated
    estimator as wasteful long after it had stopped being either.

    Example:
        >>> RouteStatistics(
        ...     samples=100, observations=100, quantile=0.9,
        ...     overrun_ratio=0.09, error_ratio=1.14,
        ... ).overrun_ratio
        0.09
    """

    samples: int
    observations: int
    quantile: float
    overrun_ratio: float
    error_ratio: float | None


class QuantileEstimator:
    """Reserves the point of a route's own history that most requests come in under.

    Args:
        quantile: Which point of the history to reserve. See the table below.
        route_key: Groups requests whose output lengths belong together.
            Defaults to the model alone, which is weak on purpose.
        min_samples: How many observations a route needs before its own history
            is used. Below it, `fallback` answers.
        fallback: What answers below the threshold. Defaults to reserving the
            maximum the caller allowed. Any estimator will do, so
            `StaticEstimator` says "until you know better, reserve five
            hundred".
        history: How many recent output lengths to keep per route.
        adapt_quantile: Whether a route may move its own quantile when the
            overruns it sees stop matching the ones it promised. Off by
            default, and worth turning on once a workload has settled.
        adapt_every: How many observations a route collects before its quantile
            may move again.
        adapt_step: How far it moves when it moves.
        quantile_bounds: How far it may travel, as a low and a high.

    Choosing the quantile, which is the question everyone asks first:

    | Quantile | Overruns | Wasted headroom | Use it when |
    |---|---|---|---|
    | 0.50 | About half | Minimal | Never. Constant overrun defeats the limit |
    | 0.90 | One in ten | Moderate, returned fast | The default |
    | 0.99 | One in a hundred | Large on a heavy tail | A rate limit response is expensive |
    | 1.00 | None | The same as the maximum | You have no history at all |

    These distributions are heavy tailed, so the ninety ninth percentile is often
    many multiples of the ninetieth and raising the quantile does not cost
    linearly. The ninth decile, with the surplus credited back the moment the
    real figure is known, is the operating point worth defaulting to.

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
        adapt_quantile: bool = False,
        adapt_every: int = DEFAULT_ADAPT_EVERY,
        adapt_step: float = DEFAULT_ADAPT_STEP,
        quantile_bounds: tuple[float, float] = DEFAULT_QUANTILE_BOUNDS,
    ) -> None:
        """Learn per route, reserving at `quantile`.

        Raises:
            ConfigurationError: if `quantile` is outside [0, 1], if
                `min_samples` is negative, if `history` is below one, or if
                `min_samples` is above `history`, which would mean the
                threshold could never be reached. Also if `adapt_every` is
                below one, if `adapt_step` is not positive, or if
                `quantile_bounds` is not an ordered pair inside [0, 1].
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
        low, high = quantile_bounds
        if not 0.0 <= low <= high <= 1.0:
            message = (
                f"quantile_bounds is how far the quantile may travel, as a low and a high, "
                f"so it must be ordered and inside [0, 1], got {quantile_bounds}. The "
                f"default of {DEFAULT_QUANTILE_BOUNDS} keeps it between overrunning half "
                f"the time and reserving very nearly the worst case."
            )
            raise ConfigurationError(message)
        if adapt_every < 1:
            message = (
                f"adapt_every is how many observations a route collects before its "
                f"quantile may move again, so it must be at least one, got {adapt_every}. "
                f"A quantile that moves per request is an oscillation, so the default of "
                f"{DEFAULT_ADAPT_EVERY} is deliberately slow."
            )
            raise ConfigurationError(message)
        if adapt_step <= 0.0:
            message = (
                f"adapt_step is how far the quantile moves when it moves, so it must be "
                f"positive, got {adapt_step}. Pass adapt_quantile=False to hold the "
                f"quantile still instead."
            )
            raise ConfigurationError(message)
        self._quantile = quantile
        self._route_key = route_key
        self._min_samples = min_samples
        self._adapt_quantile = adapt_quantile
        self._adapt_every = adapt_every
        self._adapt_step = adapt_step
        self._quantile_bounds = quantile_bounds
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

    def statistics(self, context: RequestContext) -> RouteStatistics | None:
        """How well this route's predictions have been doing, or None if unseen.

        The two ratios answer different questions and neither replaces the
        other. The overrun ratio says whether the quantile is still telling the
        truth. The error ratio says whether it is worth what it costs.
        """
        route = self._routes.get(self._route_key(context))
        if route is None:
            return None
        return RouteStatistics(
            samples=len(route.ring),
            observations=route.observations,
            quantile=route.quantile,
            overrun_ratio=route.overrun_ratio,
            error_ratio=route.error_ratio,
        )

    def record(self, observation: Observation) -> None:
        """Remember what this request really produced, and how wrong the guess was."""
        route = self._route_for(observation.context)
        actual = observation.actual.output_tokens
        reserved = observation.reserved.output_tokens
        route.ring.append(actual)
        route.observations += 1
        overran = actual > reserved
        if overran:
            route.overruns += 1
            route.overruns_since_adapt += 1
        route.weigh_in("overrun_ratio", 1.0 if overran else 0.0)
        if actual > 0:
            # Skipped rather than recorded when nothing was generated. Reserved
            # over zero is undefined, and calling it infinite would poison every
            # later reading with a number that means nothing.
            route.weigh_in("error_ratio", reserved / actual)
        if self._adapt_quantile:
            self._maybe_adapt(route)

    def _maybe_adapt(self, route: _Route) -> None:
        """Move this route's quantile if the last window disagreed with it.

        The overrun rate drives both directions, and the estimate error ratio
        deliberately drives neither. The overrun rate measures the promise the
        quantile makes: reserving at the ninth decile is a claim that about one
        request in ten will run over, and it is either true or it is not. The
        error ratio measures how wide the distribution is, which on a heavy
        tailed route is large however well calibrated the quantile is. Lowering
        the quantile because a route is wide would walk it down through a range
        where nothing changes at all, then off the far side of a mode, and
        overrun half the traffic at once. That is the oscillation this whole
        mechanism is rate limited to avoid, so it must not be the thing that
        causes it.
        """
        route.since_adapt += 1
        if route.since_adapt < self._adapt_every:
            return
        low, high = self._quantile_bounds
        promised = 1.0 - route.quantile
        seen = route.overruns_since_adapt / route.since_adapt
        if seen > promised * RAISE_ABOVE:
            route.quantile = min(route.quantile + self._adapt_step, high)
        elif seen < promised * LOWER_BELOW:
            route.quantile = max(route.quantile - self._adapt_step, low)
        route.since_adapt = 0
        route.overruns_since_adapt = 0

    def serialise(self) -> dict[Hashable, tuple[int, ...]]:
        """Hand back every route's history, as plain containers.

        Nothing calls this yet. It exists now so that sharing histories between
        workers, and keeping them across a restart, is a wiring change later
        rather than a redesign of everything above.

        Plain Python containers rather than an encoded form: whatever ends up
        storing these will have its own opinion about encoding, and choosing
        one here would be choosing it blind.

        Example:
            >>> from spillway.core.cost import Cost
            >>> estimator = QuantileEstimator(min_samples=1)
            >>> context = RequestContext(model="claude")
            >>> estimator.record(
            ...     Observation(
            ...         context=context,
            ...         reserved=Cost(output_tokens=400),
            ...         actual=Cost(output_tokens=310),
            ...         at_ms=0.0,
            ...     )
            ... )
            >>> estimator.serialise()
            {'claude': (310,)}
        """
        return {key: tuple(route.ring) for key, route in self._routes.items()}

    def merge(self, histories: Mapping[Hashable, Sequence[int]]) -> None:
        """Fold another instance's histories into this one.

        Concatenates and then resamples, keeping an even stride across the
        combined history when it is longer than the ring. Appending the
        incoming samples straight onto the ring would evict the entire local
        history whenever both sides are full, which is precisely the case that
        matters, and would make a merge a replacement.

        The result is a mixture of both sides rather than either one, which is
        the point: two workers that have each seen half the traffic should end
        up agreeing about all of it.
        """
        for key, incoming in histories.items():
            route = self._routes.get(key)
            if route is None:
                route = _Route(ring=deque(maxlen=self._history), quantile=self._quantile)
                self._routes[key] = route
            combined = [*route.ring, *incoming]
            route.ring.clear()
            route.ring.extend(_thinned(combined, self._history))

    def _route_for(self, context: RequestContext) -> _Route:
        """The state for `context`'s route, created on first sight."""
        key = self._route_key(context)
        route = self._routes.get(key)
        if route is None:
            route = _Route(ring=deque(maxlen=self._history), quantile=self._quantile)
            self._routes[key] = route
        return route
