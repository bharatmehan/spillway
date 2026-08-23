"""The public surface.

Adding a name to the export list is a deliberate act, not a side effect of
writing a class. This test is what makes it deliberate: a new export has to be
added here too, which is the moment to ask whether it should be supported for
ever.
"""

import importlib

import pytest

import spillway

EXPECTED = [
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
    # A submodule rather than a name, so that `from spillway import providers`
    # works for anyone who reached the package first, which is everyone.
    "providers",
    "scope_context",
]


def test_version_is_non_empty():
    assert spillway.__version__
    assert spillway.__version__ != "0.0.0.dev0", (
        "the version fell back, so the package is not installed in this environment"
    )


def test_the_export_list_is_exactly_what_was_decided():
    assert spillway.__all__ == EXPECTED


@pytest.mark.parametrize("name", EXPECTED)
def test_every_exported_name_actually_exists(name):
    assert getattr(spillway, name) is not None


def test_a_star_import_brings_in_the_export_list_and_nothing_else():
    namespace: dict[str, object] = {}
    exec("from spillway import *", namespace)  # noqa: S102
    assert sorted(name for name in namespace if not name.startswith("__")) == sorted(
        name for name in EXPECTED if not name.startswith("__")
    )


def test_the_quickstart_needs_no_optional_dependency():
    # The reason the character heuristic and the in memory store exist. If the
    # first thing someone runs needs an extra installed, most of them stop here.
    limiter = spillway.Spillway(dimensions=[spillway.Rate("rpm", limit=10)])
    assert limiter.snapshot().dimensions["rpm"].limit == 10.0


def test_internals_are_not_reachable_from_the_top_level():
    # Everything else needs an explicit submodule import, so what an editor
    # offers at the top level is what is actually supported.
    for name in ("engine", "MemoryStore", "Claim", "Delta", "Utilisation"):
        assert not hasattr(spillway, name)


def test_the_extension_protocols_are_not_exported():
    # Estimators are constructed, so they are exported. The protocol behind
    # them is implemented, and a protocol needs no import to implement, so
    # exporting it would only enlarge what is promised for ever.
    for name in ("Estimator", "Observation", "Store", "Dimension", "Clock"):
        assert not hasattr(spillway, name)


def test_the_submodules_are_importable_for_anyone_who_needs_them():
    for name in ("spillway.stores.memory", "spillway.dimensions.base", "spillway.core.clock"):
        assert importlib.import_module(name) is not None
