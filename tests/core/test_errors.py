"""The exception hierarchy."""

import pytest

from spillway.core.errors import (
    AdmissionDenied,
    AdmissionTimeout,
    ConfigurationError,
    LeaseAlreadySettled,
    LeaseError,
    LeaseExpired,
    MissingExtra,
    ScopeExhausted,
    Shed,
    SpillwayError,
    StoreCorruption,
    StoreError,
    StoreUnavailable,
)

EVERYTHING = [
    AdmissionDenied,
    AdmissionTimeout,
    ConfigurationError,
    LeaseAlreadySettled,
    LeaseError,
    LeaseExpired,
    MissingExtra,
    ScopeExhausted,
    Shed,
    StoreCorruption,
    StoreError,
    StoreUnavailable,
]


@pytest.mark.parametrize("error_type", EVERYTHING)
def test_everything_descends_from_one_base(error_type):
    # A caller must be able to catch everything this library raises with one
    # clause, and catch nothing else with it.
    assert issubclass(error_type, SpillwayError)
    assert issubclass(error_type, Exception)


@pytest.mark.parametrize("error_type", [AdmissionTimeout, Shed, ScopeExhausted])
def test_every_refusal_is_catchable_as_a_denial(error_type):
    assert issubclass(error_type, AdmissionDenied)


@pytest.mark.parametrize("error_type", [StoreUnavailable, StoreCorruption])
def test_store_failures_share_a_base(error_type):
    assert issubclass(error_type, StoreError)


@pytest.mark.parametrize("error_type", [LeaseAlreadySettled, LeaseExpired])
def test_lease_misuse_shares_a_base(error_type):
    assert issubclass(error_type, LeaseError)


def test_a_denial_carries_the_binding_dimension_and_a_retry_after():
    error = AdmissionDenied("full", retry_after=1.5, binding_dimension="output_tpm")
    assert error.retry_after == 1.5
    assert error.binding_dimension == "output_tpm"
    assert str(error) == "full"


def test_a_denial_carries_nothing_by_default():
    error = AdmissionDenied("full")
    assert error.retry_after is None
    assert error.binding_dimension is None


def test_a_timeout_inherits_the_denial_payload():
    error = AdmissionTimeout("waited too long", retry_after=0.25, binding_dimension="rpm")
    assert (error.retry_after, error.binding_dimension) == (0.25, "rpm")


def test_the_missing_extra_message_names_the_exact_install_command():
    # Someone hitting this is trying to get something working. Making them go
    # and look up the extra's name is the failure this message exists to avoid.
    error = MissingExtra("RedisStore", extra="redis")
    assert str(error) == (
        "RedisStore requires the redis extra. Install it with: pip install 'spillway[redis]'"
    )
    assert error.component == "RedisStore"
    assert error.extra == "redis"
