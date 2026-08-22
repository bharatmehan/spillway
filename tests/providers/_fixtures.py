"""Reading the captured provider fixtures."""

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "providers"


def load(provider, name):
    """Read one captured fixture."""
    return json.loads((FIXTURES / provider / f"{name}.json").read_text())


def every(provider, prefix):
    """Read every captured fixture of one kind, newest naming first."""
    found = sorted(FIXTURES.joinpath(provider).glob(f"{prefix}*.json"))
    return [(path.stem, json.loads(path.read_text())) for path in found]
