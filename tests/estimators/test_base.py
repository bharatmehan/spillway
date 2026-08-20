"""The estimator protocol and the two records it passes around."""

from spillway.core.cost import Cost, Distribution, Estimate
from spillway.core.scope import Scope
from spillway.estimators.base import Estimator, Observation, RequestContext


class _Fixed:
    def estimate(self, context: RequestContext) -> Estimate:
        return Estimate(input=0, output=Distribution.point(20))

    def record(self, observation: Observation) -> None:
        pass


def test_an_object_with_the_two_methods_is_an_estimator():
    # Structural, so a user implements one without importing anything from
    # here. A base class would make the extension point a dependency.
    assert isinstance(_Fixed(), Estimator)


def test_an_object_missing_record_is_not_an_estimator():
    class _EstimateOnly:
        def estimate(self, context):
            return Estimate(input=0, output=Distribution.point(1))

    assert not isinstance(_EstimateOnly(), Estimator)


def test_a_context_needs_nothing_at_all():
    assert RequestContext().model is None


def test_a_context_defaults_to_the_global_scope():
    assert RequestContext().scope == Scope.of(None)


def test_a_context_carries_whatever_the_caller_wants_to_route_on():
    context = RequestContext(model="claude", tags={"task": "summarise"})
    assert (context.model, context.tags["task"]) == ("claude", "summarise")


def test_two_contexts_describing_the_same_request_are_equal():
    # A route key is computed from one of these at admission and from another
    # at settlement, so they have to compare by value rather than by identity.
    assert RequestContext(model="claude", tags={"task": "a"}) == RequestContext(
        model="claude", tags={"task": "a"}
    )


def test_an_observation_carries_both_costs():
    observation = Observation(
        context=RequestContext(model="claude"),
        reserved=Cost(input_tokens=12_400, output_tokens=1_180),
        actual=Cost(input_tokens=12_400, output_tokens=415),
        at_ms=6_200.0,
    )
    assert (observation.reserved - observation.actual).output_tokens == 765
