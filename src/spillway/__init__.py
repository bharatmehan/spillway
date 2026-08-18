"""Congestion control for language model API traffic."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spillway")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
