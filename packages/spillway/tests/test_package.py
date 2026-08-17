"""The package imports and reports a version.

This exists so the test job is not vacuously green, and so a packaging fault is
caught by the test suite rather than by someone installing the distribution.
"""

import spillway


def test_version_is_non_empty():
    assert spillway.__version__
    assert spillway.__version__ != "0.0.0.dev0", (
        "the version fell back, so the package is not installed in this environment"
    )


def test_public_surface_is_only_the_version():
    assert spillway.__all__ == ["__version__"]
