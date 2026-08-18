"""The vocabulary a store speaks: claims, deltas, utilisation and results."""

import pytest

from spillway.stores.base import Claim, ClaimKind, Delta, ReserveResult, Utilisation


def rate_claim(**overrides):
    args = {
        "key": "acme:rpm",
        "kind": ClaimKind.RATE,
        "cost": 1.0,
        "limit": 1000.0,
        "window_ms": 60_000.0,
    }
    args.update(overrides)
    return Claim(**args)


def test_a_rate_claim_carries_its_window():
    assert rate_claim().window_ms == 60_000.0


def test_a_gauge_claim_has_no_window():
    claim = Claim("acme:generations", ClaimKind.GAUGE, cost=1.0, limit=64.0)
    assert claim.window_ms is None


def test_a_rate_claim_without_a_window_is_refused():
    # A rate claim with no window has no emission interval, so a store would
    # have to invent one. Failing at construction names the fix instead.
    with pytest.raises(ValueError, match="needs a window"):
        rate_claim(window_ms=None)


@pytest.mark.parametrize("window", [0.0, -1.0])
def test_a_rate_claim_with_a_non_positive_window_is_refused(window):
    with pytest.raises(ValueError, match="must be positive"):
        rate_claim(window_ms=window)


def test_a_gauge_claim_with_a_window_is_refused():
    with pytest.raises(ValueError, match="has no window"):
        Claim("acme:generations", ClaimKind.GAUGE, cost=1.0, limit=64.0, window_ms=1.0)


def test_a_negative_cost_is_refused():
    with pytest.raises(ValueError, match="cost cannot be negative"):
        rate_claim(cost=-1.0)


def test_a_negative_limit_is_refused():
    with pytest.raises(ValueError, match="limit cannot be negative"):
        rate_claim(limit=-1.0)


def test_a_delta_keeps_the_sign_of_a_credit():
    assert Delta("acme:output_tpm", ClaimKind.RATE, amount=765.0).amount == 765.0


def test_a_delta_keeps_the_sign_of_an_overrun():
    # An overrun that lost its sign would be credited back as though the
    # request had used less than it reserved, which breaks the limit silently.
    assert Delta("acme:output_tpm", ClaimKind.RATE, amount=-150.0).amount == -150.0


def test_headroom_is_the_free_fraction():
    assert Utilisation(used=412.0, limit=1000.0).headroom == pytest.approx(0.588)


def test_a_full_key_reports_no_headroom():
    assert Utilisation(used=64.0, limit=64.0).headroom == 0.0


def test_an_empty_key_reports_all_headroom():
    assert Utilisation(used=0.0, limit=64.0).headroom == 1.0


def test_an_overdrawn_key_reports_no_headroom_rather_than_a_negative_one():
    # A key can go past its limit through an overrun. Reporting negative
    # headroom would put a nonsense number in front of a user.
    assert Utilisation(used=80.0, limit=64.0).headroom == 0.0


def test_a_zero_limit_reports_no_headroom_rather_than_dividing_by_zero():
    assert Utilisation(used=0.0, limit=0.0).headroom == 0.0


def test_a_granted_result_carries_a_lease():
    result = ReserveResult.granted_as("lease-1")
    assert result.granted
    assert result.lease_id == "lease-1"
    assert result.binding_key is None


def test_a_refused_result_names_the_binding_key_and_the_wait():
    result = ReserveResult.refused("acme:rpm", retry_after_ms=500.0)
    assert not result.granted
    assert result.binding_key == "acme:rpm"
    assert result.retry_after_ms == 500.0
    assert result.lease_id is None


def test_utilisation_defaults_to_empty_rather_than_none():
    assert ReserveResult.granted_as("lease-1").utilisation == {}


def test_a_granted_result_without_a_lease_is_refused_at_construction():
    # A store is an extension point. Catching an inconsistent result here names
    # the mistake, instead of failing three frames away when settlement runs.
    with pytest.raises(ValueError, match="must carry a lease_id"):
        ReserveResult(granted=True)


def test_a_granted_result_with_a_binding_key_is_refused_at_construction():
    with pytest.raises(ValueError, match="cannot have a binding key"):
        ReserveResult(granted=True, lease_id="lease-1", binding_key="acme:rpm")


def test_a_refused_result_with_a_lease_is_refused_at_construction():
    with pytest.raises(ValueError, match="cannot have a lease_id"):
        ReserveResult(granted=False, lease_id="lease-1", binding_key="acme:rpm")


def test_a_refused_result_without_a_binding_key_is_refused_at_construction():
    with pytest.raises(ValueError, match="must name the key that refused"):
        ReserveResult(granted=False)


def test_a_refusal_may_have_no_retry_after_when_waiting_would_not_help():
    assert ReserveResult.refused("acme:budget").retry_after_ms is None
