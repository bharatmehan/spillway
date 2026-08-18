"""Everything this library raises.

The whole hierarchy is defined here, including members nothing raises yet, so
that a caller can write the `except` clause they need without waiting for the
feature that produces it. Adding an exception class later is a change every
user has to notice; adding one now costs a few lines.

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

    Carries enough for the caller to act rather than merely fail: which
    dimension ran out, and how long until it would not have.

    Attributes:
        retry_after: Seconds until the binding dimension could admit this
            request, or None when waiting would not help.
        binding_dimension: The name of the dimension that ran out, or None if
            the refusal was not attributable to one.
        explanation: How full every limit was when the refusal happened.
            Typed loosely here so that the exception hierarchy stays free of
            imports from the rest of the library.

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

    Distinct from the other refusals so a caller can retry later instead of
    treating it as an error. Nothing is wrong; the system is busy and this
    request said it could wait.
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
    """The store could not be reached.

    Only raised when the limiter is configured to propagate store failures. By
    default it degrades and keeps serving instead.
    """


class StoreCorruption(StoreError):
    """The store holds something this library did not write.

    Usually a namespace collision with another application, or a version of
    this library that wrote a different key layout.
    """


class LeaseError(SpillwayError):
    """Base class for misuse of a lease."""


class LeaseAlreadySettled(LeaseError):
    """This lease was settled or abandoned already.

    Raised rather than ignored, because a second settlement would count the
    same request twice and quietly corrupt every limit it touched.
    """


class LeaseExpired(LeaseError):
    """The lease outlived its expiry and its capacity was reclaimed.

    The call it covered ran longer than the limiter was told to expect. Its
    capacity is already back in circulation, so it cannot be settled.
    """


class MissingExtra(SpillwayError):
    """An optional dependency is needed and is not installed.

    The message carries the exact command that fixes it, because a caller who
    hits this is trying to get something working and should not have to go and
    find out how.

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
