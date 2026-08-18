"""The cost value type and its arithmetic."""

import dataclasses

import pytest

from spillway.core.cost import Cost


def test_defaults_are_one_request_and_no_tokens():
    assert Cost() == Cost(input_tokens=0, output_tokens=0, requests=1, extra={})


def test_total_tokens_is_input_plus_output():
    assert Cost(input_tokens=100, output_tokens=25).total_tokens == 125


def test_total_tokens_excludes_extra_categories():
    # Extra categories sit on their own provider limits. Folding them into the
    # combined total would count them twice against a provider that meters both.
    cost = Cost(input_tokens=100, output_tokens=25, extra={"cached_input": 4_000})
    assert cost.total_tokens == 125


def test_subtraction_is_componentwise():
    reserved = Cost(input_tokens=12_400, output_tokens=1_180)
    actual = Cost(input_tokens=12_400, output_tokens=415)
    assert reserved - actual == Cost(input_tokens=0, output_tokens=765, requests=0)


def test_subtraction_produces_negative_components_on_overrun():
    # An overrun must survive settlement as a negative number. Clamping it at
    # zero would silently discard the debt and break the limit over time.
    delta = Cost(output_tokens=100) - Cost(output_tokens=250)
    assert delta.output_tokens == -150


def test_subtraction_unions_extra_keys():
    left = Cost(extra={"cached_input": 100, "reasoning": 50})
    right = Cost(extra={"cached_input": 30, "audio": 7})
    assert (left - right).extra == {"audio": -7, "cached_input": 70, "reasoning": 50}


def test_subtraction_of_a_non_cost_is_not_implemented():
    assert Cost().__sub__(object()) is NotImplemented


def test_cost_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Cost().input_tokens = 5
