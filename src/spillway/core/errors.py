"""Everything this library raises.

The whole hierarchy is defined here, including members nothing raises yet, so a
caller can write the `except` clause they need before the feature that produces
it exists. Each of those says which one it is waiting for.

Every message names the fix, not just the problem.
"""

from __future__ import annotations


class SpillwayError(Exception):
    """Base class for everything this library raises.

    Catch this to catch anything from the limiter and nothing from anywhere
    else.

    Example:
        >>> try:
        ...     raise ConfigurationError("limit must be positive")
        ... except SpillwayError as error:
        ...     print(error)
        limit must be positive
    """


class AdmissionDenied(SpillwayError):
    """The request cannot proceed.

    Carries which dimension ran out and how long until it would not have, so a
    caller can act rather than merely fail.

    Attributes:
        retry_after: Seconds until the binding dimension could admit this
            request, or None when waiting would not help.
        binding_dimension: The name of the dimension that ran out, or None if
            the refusal was not attributable to one.
        explanation: How full every limit was when the refusal happened. Typed
            loosely to keep this module free of imports from the rest of the
            library.

    Example:
        >>> error = AdmissionDenied(
        ...     "no room on output_tpm", retry_after=1.5, binding_dimension="output_tpm"
        ... )
        >>> error.retry_after, error.binding_dimension
        (1.5, 'output_tpm')
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        binding_dimension: str | None = None,
        explanation: object | None = None,
    ) -> None:
        """Record the refusal along with what would make it succeed."""
        super().__init__(message)
        self.retry_after = retry_after
        self.binding_dimension = binding_dimension
        self.explanation = explanation


class AdmissionTimeout(AdmissionDenied):
    """Waited past the given timeout or deadline without being admitted."""


class Shed(AdmissionDenied):
    """Dropped rather than queued, because the work was marked sheddable.

    Distinct from the other refusals so a caller can retry later rather than
    treat it as an error. Nothing is wrong; the system is busy.
    """


class ScopeExhausted(AdmissionDenied):
    """A scope spent its total budget. Waiting will not help, so retry_after is None."""


class ConfigurationError(SpillwayError):
    """The limiter was assembled in a way that cannot work.

    Raised at construction rather than at admission, so a mistake surfaces on
    the first run rather than under load.
    """


class StoreError(SpillwayError):
    """Base class for failures of the coordination layer."""


class StoreUnavailable(StoreError):
    """The store could not be reached. Raised once there is a coordinated store.

    Only when the limiter is configured to propagate store failures. By default
    it degrades and keeps serving.
    """


class StoreCorruption(StoreError):
    """The store holds something this library did not write.

    Raised once there is a coordinated store. Usually a namespace collision with
    another application, or a different key layout from another version.
    """


class LeaseError(SpillwayError):
    """Base class for misuse of a lease."""


class LeaseAlreadySettled(LeaseError):
    """This lease was settled or abandoned already.

    Raised rather than ignored: a second settlement would count the same request
    twice and quietly corrupt every limit it touched.
    """


class LeaseExpired(LeaseError):
    """The lease outlived its expiry and its capacity was reclaimed.

    The call ran longer than the limiter was told to expect, so its capacity is
    already back in circulation and cannot be settled.
    """


class MissingExtra(SpillwayError):
    """An optional dependency is needed and is not installed.

    The message carries the exact install command.

    Example:
        >>> print(MissingExtra("RedisStore", extra="redis"))
        RedisStore requires the redis extra. Install it with: pip install 'spillway[redis]'
    """

    def __init__(self, component: str, *, extra: str) -> None:
        """Build the message from the component that needs `extra`."""
        message = (
            f"{component} requires the {extra} extra. "
            f"Install it with: pip install 'spillway[{extra}]'"
        )
        super().__init__(message)
        self.component = component
        self.extra = extra
