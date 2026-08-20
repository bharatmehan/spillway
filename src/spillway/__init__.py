"""Congestion control for language model API traffic.

Admission control across rate and concurrency, with capacity reserved from an
estimate before a call and settled against the real cost after it, so the
difference goes back immediately rather than at the end of a window.

A call that finds no room waits for it, highest priority first, rather than
failing on a limit that will have cleared in a second.

    from spillway import Concurrency, Rate, Spillway

    limiter = Spillway(
        dimensions=[
            Rate("rpm", limit=1_000),
            Rate("input_tpm", limit=400_000),
            Rate("output_tpm", limit=80_000),
            Concurrency("generations", limit=64),
        ]
    )

    async def call(prompt: str) -> str:
        async with limiter.admit(prompt=prompt, max_tokens=1_024) as lease:
            response = await your_client.create(prompt=prompt)
            lease.settle(input=response.usage.input, output=response.usage.output)
            return response.text

This list is curated and small. Anything not named here is reached through an
explicit submodule import, so what is supported is what an editor offers.
"""

from importlib.metadata import PackageNotFoundError, version

from spillway.core.cost import Cost, Distribution, Estimate
from spillway.core.errors import (
    AdmissionDenied,
    AdmissionTimeout,
    ConfigurationError,
    ScopeExhausted,
    Shed,
    SpillwayError,
)
from spillway.core.lease import Lease, LeaseState
from spillway.core.scope import Priority, Scope
from spillway.core.spillway import Spillway
from spillway.dimensions.concurrency import Concurrency
from spillway.dimensions.rate import Rate

try:
    __version__ = version("spillway")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "AdmissionDenied",
    "AdmissionTimeout",
    "Concurrency",
    "ConfigurationError",
    "Cost",
    "Distribution",
    "Estimate",
    "Lease",
    "LeaseState",
    "Priority",
    "Rate",
    "Scope",
    "ScopeExhausted",
    "Shed",
    "Spillway",
    "SpillwayError",
    "__version__",
]
