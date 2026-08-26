"""Congestion control for language model API traffic.

Admission control across rate and concurrency. Capacity is reserved from an
estimate before a call and settled against the real cost after it, so the
difference goes back immediately rather than at the end of a window. A call that
finds no room waits for it, highest priority first.

Two lines where the client is built, and every call site untouched:

    from anthropic import AsyncAnthropic
    from spillway import Spillway

    client = Spillway.instrument(AsyncAnthropic(), rpm=1_000, input_tpm=2_000_000)

    reply = await client.messages.create(model=..., messages=..., max_tokens=1_024)

The limits are yours. This library ships none of its own. Name the ones your
provider gives you: `rpm`, `rpd`, `tpm`, `input_tpm`, `output_tpm`,
`concurrency`. Naming none admits everything and records what the traffic really
costs, which is the intended first step:

    client = Spillway.instrument(AsyncAnthropic())
    ...
    Spillway.of(client).snapshot()

Scope and priority arrive from the surrounding code rather than every call site:

    with scope_context(f"tenant:{tenant}", priority=Priority.INTERACTIVE):
        ...

And underneath it, the admission itself, for a provider with no adapter or work
that is not an SDK call:

    async with limiter.admit(prompt=prompt, max_tokens=1_024) as lease:
        response = await your_client.create(prompt=prompt)
        lease.settle(input=response.usage.input, output=response.usage.output)

The export list below is curated and small. Anything not named in it is reached
through an explicit submodule import.
"""

from importlib.metadata import PackageNotFoundError, version

# Imported rather than left to a submodule import, so that `spillway.providers`
# resolves for anyone who reached the package first, which is everyone.
from spillway import providers
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
from spillway.estimators.base import RequestContext
from spillway.estimators.callable import CallableEstimator
from spillway.estimators.max_tokens import MaxTokensEstimator
from spillway.estimators.quantile import QuantileEstimator
from spillway.estimators.static import StaticEstimator
from spillway.integrations.context import scope_context

try:
    __version__ = version("spillway")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "AdmissionDenied",
    "AdmissionTimeout",
    "CallableEstimator",
    "Concurrency",
    "ConfigurationError",
    "Cost",
    "Distribution",
    "Estimate",
    "Lease",
    "LeaseState",
    "MaxTokensEstimator",
    "Priority",
    "QuantileEstimator",
    "Rate",
    "RequestContext",
    "Scope",
    "ScopeExhausted",
    "Shed",
    "Spillway",
    "SpillwayError",
    "StaticEstimator",
    "__version__",
    "providers",
    "scope_context",
]
