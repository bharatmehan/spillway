"""The estimate produced when the caller supplies nothing but a prompt."""

import pytest

from spillway.core.cost import (
    CHARACTERS_PER_TOKEN,
    DEFAULT_MAX_OUTPUT_TOKENS,
    default_estimate,
)


def test_input_is_characters_over_the_divisor_rounded_up():
    prompt = "a" * 36
    assert default_estimate(prompt, max_tokens=10).input == int(36 / CHARACTERS_PER_TOKEN)


def test_a_partial_token_rounds_up_rather_than_down():
    # Rounding down would under reserve on every short prompt, and many short
    # prompts is the common shape of agent traffic.
    assert default_estimate("a", max_tokens=10).input == 1


def test_an_empty_prompt_costs_no_input():
    assert default_estimate("", max_tokens=10).input == 0


def test_a_missing_prompt_costs_no_input():
    assert default_estimate(max_tokens=10).input == 0


def test_a_missing_maximum_falls_back_to_the_documented_default():
    assert default_estimate("hello").output.value == DEFAULT_MAX_OUTPUT_TOKENS


def test_the_reserved_output_is_the_requested_maximum():
    assert default_estimate("hello", max_tokens=4096).output.quantile(0.9) == 4096


def test_a_message_sequence_is_measured_by_its_content():
    messages = [
        {"role": "user", "content": "a" * 36},
        {"role": "assistant", "content": "b" * 36},
    ]
    assert (
        default_estimate(messages, max_tokens=10).input
        == default_estimate("a" * 72, max_tokens=10).input
    )


def test_nested_content_blocks_are_measured():
    messages = [{"role": "user", "content": [{"type": "text", "text": "unused"}]}]
    # The block has no "content" key, so it contributes nothing rather than
    # raising. A prompt shape this library does not recognise must not break
    # admission, it must only make the estimate worse.
    assert default_estimate(messages, max_tokens=10).input == 0


def test_a_message_without_content_contributes_nothing():
    assert default_estimate([{"role": "user"}], max_tokens=10).input == 0


def test_the_model_is_carried_through():
    assert default_estimate("hi", max_tokens=10, model="claude").model == "claude"


def test_a_negative_maximum_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        default_estimate("hi", max_tokens=-1)


def test_a_zero_maximum_is_allowed():
    assert default_estimate("hi", max_tokens=0).output.value == 0
