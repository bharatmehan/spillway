"""The rate dimension: what it claims, and what it refuses to be configured as."""

import pytest

from spillway.core.cost import Cost
from spillway.core.errors import ConfigurationError
from spillway.core.scope import Scope
from spillway.dimensions.rate import Rate
from spillway.stores.base import ClaimKind

ACME = Scope("tenant:acme")


def test_a_rate_claim_costs_what_the_meter_counts():
    tokens = Rate("input_tpm", limit=400_000, meter="input_tokens")
    claim = tokens.claim(Cost(input_tokens=8_600, output_tokens=99), ACME)
    assert claim.cost == 8_600.0


def test_a_request_meter_ignores_tokens_entirely():
    requests = Rate("rpm", limit=1_000, meter="requests")
    assert requests.claim(Cost(input_tokens=50_000), ACME).cost == 1.0


def test_a_total_meter_counts_input_plus_output():
    total = Rate("tpm", limit=1_000, meter="total_tokens")
    assert total.claim(Cost(input_tokens=100, output_tokens=25), ACME).cost == 125.0


def test_a_claim_carries_the_scoped_key_the_limit_and_the_window():
    tokens = Rate("input_tpm", limit=400_000, meter="input_tokens", window=60)
    claim = tokens.claim(Cost(input_tokens=1), ACME)
    assert claim.key == "tenant:acme:input_tpm"
    assert claim.kind is ClaimKind.RATE
    assert claim.limit == 400_000.0
    assert claim.window_ms == 60_000.0


def test_the_window_is_given_in_seconds_and_held_in_milliseconds():
    # Seconds at the boundary because that is how providers publish limits.
    # Milliseconds inside because that is what the arithmetic works in.
    assert Rate("rpd", limit=1, meter="requests", window=86_400).window_ms == 86_400_000.0


def test_settling_below_the_reservation_gives_the_difference_back():
    tokens = Rate("output_tpm", limit=80_000, meter="output_tokens")
    delta = tokens.settle(Cost(output_tokens=1_180), Cost(output_tokens=415), ACME)
    assert delta.amount == 765.0
    assert delta.key == "tenant:acme:output_tpm"


def test_settling_above_the_reservation_records_a_negative_amount():
    # The overrun has to survive as a debt. Losing the sign here would credit
    # capacity back that was never free, and the limit would drift open.
    tokens = Rate("output_tpm", limit=80_000, meter="output_tokens")
    delta = tokens.settle(Cost(output_tokens=100), Cost(output_tokens=250), ACME)
    assert delta.amount == -150.0


def test_settling_exactly_the_reservation_corrects_nothing():
    requests = Rate("rpm", limit=1_000, meter="requests")
    assert requests.settle(Cost(), Cost(), ACME).amount == 0.0


def test_the_claim_and_the_settlement_use_the_same_key():
    tokens = Rate("input_tpm", limit=1, meter="input_tokens")
    claim = tokens.claim(Cost(input_tokens=1), ACME)
    delta = tokens.settle(Cost(input_tokens=1), Cost(input_tokens=1), ACME)
    assert claim.key == delta.key


def test_an_adaptive_rate_dimension_is_refused_with_the_reason():
    # A published provider limit is a fact. Searching for it means deliberately
    # exceeding it, which is the thing the limiter exists to prevent.
    with pytest.raises(ConfigurationError, match="not a hypothesis to probe"):
        Rate("input_tpm", limit=400_000, meter="input_tokens", adaptive=True)


@pytest.mark.parametrize("limit", [0, -1])
def test_a_non_positive_limit_is_refused(limit):
    with pytest.raises(ConfigurationError, match="positive limit"):
        Rate("rpm", limit=limit, meter="requests")


@pytest.mark.parametrize("window", [0, -1])
def test_a_non_positive_window_is_refused(window):
    with pytest.raises(ConfigurationError, match="positive window"):
        Rate("rpm", limit=1, meter="requests", window=window)


def test_a_rate_dimension_prints_its_configuration():
    assert repr(Rate("rpm", limit=1_000, meter="requests")) == (
        "Rate('rpm', limit=1000.0, meter='requests', window=60.0)"
    )
