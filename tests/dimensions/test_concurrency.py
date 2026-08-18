"""The concurrency dimension."""

import pytest

from spillway.core.cost import Cost
from spillway.core.errors import ConfigurationError
from spillway.core.scope import Scope
from spillway.dimensions.concurrency import Concurrency
from spillway.stores.base import ClaimKind

ACME = Scope("tenant:acme")


def test_a_claim_takes_exactly_one_slot():
    assert Concurrency("generations", limit=64).claim(Cost(), ACME).cost == 1.0


def test_the_slot_taken_does_not_depend_on_the_size_of_the_request():
    # One request occupies one slot whether it generates ten tokens or ten
    # thousand. Size is what the rate and occupancy limits are for.
    generations = Concurrency("generations", limit=64)
    small = generations.claim(Cost(input_tokens=1, output_tokens=1), ACME)
    large = generations.claim(Cost(input_tokens=500_000, output_tokens=100_000), ACME)
    assert small.cost == large.cost == 1.0


def test_a_claim_is_a_gauge_carrying_the_scoped_key_and_the_limit():
    claim = Concurrency("generations", limit=64).claim(Cost(), ACME)
    assert claim.kind is ClaimKind.GAUGE
    assert claim.key == "tenant:acme:generations"
    assert claim.limit == 64.0


def test_a_gauge_claim_carries_no_window():
    assert Concurrency("generations", limit=64).claim(Cost(), ACME).window_ms is None


def test_settling_gives_the_whole_slot_back():
    delta = Concurrency("generations", limit=64).settle(Cost(), Cost(), ACME)
    assert delta.amount == 1.0
    assert delta.kind is ClaimKind.GAUGE


def test_the_slot_comes_back_whole_however_wrong_the_estimate_was():
    # A rate limit reconciles a difference. A slot has no difference to
    # reconcile: it was occupied, and now it is not.
    generations = Concurrency("generations", limit=64)
    over = generations.settle(Cost(output_tokens=100), Cost(output_tokens=9_999), ACME)
    under = generations.settle(Cost(output_tokens=9_999), Cost(output_tokens=100), ACME)
    assert over.amount == under.amount == 1.0


def test_the_claim_and_the_settlement_use_the_same_key():
    generations = Concurrency("generations", limit=64)
    assert generations.claim(Cost(), ACME).key == generations.settle(Cost(), Cost(), ACME).key


@pytest.mark.parametrize("limit", [0, -1])
def test_a_non_positive_limit_is_refused(limit):
    with pytest.raises(ConfigurationError, match="positive limit"):
        Concurrency("generations", limit=limit)


def test_asking_for_an_adaptive_limit_says_so_rather_than_ignoring_it():
    # Accepting the argument and doing nothing with it would leave someone
    # believing their limit was being tuned when it was fixed all along.
    with pytest.raises(ConfigurationError, match="would be ignored silently"):
        Concurrency("generations", limit=64, adaptive=True)


def test_a_concurrency_dimension_prints_its_configuration():
    assert repr(Concurrency("generations", limit=64)) == "Concurrency('generations', limit=64.0)"
