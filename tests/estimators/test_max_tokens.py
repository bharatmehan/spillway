"""The safe, uninformed baseline."""

from spillway.core.cost import DEFAULT_MAX_OUTPUT_TOKENS, Cost
from spillway.estimators.base import Observation, RequestContext
from spillway.estimators.max_tokens import MaxTokensEstimator


def test_it_reserves_exactly_what_the_caller_allowed():
    estimate = MaxTokensEstimator().estimate(RequestContext(max_tokens=4_096))
    assert estimate.output.quantile(estimate.quantile) == 4_096


def test_it_falls_back_to_the_flat_default_when_no_maximum_was_named():
    estimate = MaxTokensEstimator().estimate(RequestContext())
    assert estimate.output.value == DEFAULT_MAX_OUTPUT_TOKENS


def test_it_counts_input_with_the_character_heuristic():
    assert MaxTokensEstimator().estimate(RequestContext(prompt="hello there")).input == 4


def test_it_carries_the_model_through():
    context = RequestContext(model="claude", max_tokens=10)
    assert MaxTokensEstimator().estimate(context).model == "claude"


def test_recording_changes_nothing():
    # It has no history, so the same context has to answer the same way for
    # ever. A baseline that drifted would be no baseline.
    estimator = MaxTokensEstimator()
    context = RequestContext(max_tokens=4_096)
    before = estimator.estimate(context)
    estimator.record(
        Observation(
            context=context,
            reserved=Cost(output_tokens=4_096),
            actual=Cost(output_tokens=12),
            at_ms=0.0,
        )
    )
    assert estimator.estimate(context) == before
