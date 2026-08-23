"""The instrumented client's type follows the client's, or this fails."""

import subprocess
import sys
from pathlib import Path

import pytest

SUBJECT = Path(__file__).with_name("instrumented_client.py")

# The defining module rather than the re-export, which is what mypy prints.
EXPECTED = [
    ("built", "anthropic._client.AsyncAnthropic"),
    ("given", "openai._client.AsyncOpenAI"),
    ("recovered", "spillway.core.spillway.Spillway"),
]


@pytest.fixture(scope="module")
def revealed():
    """What a strict type check makes of the instrumented clients."""
    finished = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "mypy", "--strict", "--no-incremental", str(SUBJECT)],
        capture_output=True,
        text=True,
        check=False,
    )
    return finished.stdout


@pytest.mark.slow
@pytest.mark.parametrize(("name", "expected"), EXPECTED, ids=[n for n, _ in EXPECTED])
def test_the_type_survives_instrumenting(revealed, name, expected):
    # A wrapper that turned a typed client into an untyped one would be removed
    # by the first user who noticed, and they would be right to remove it. An
    # assertion in a docstring is not a test, so this runs the type checker.
    assert f'Revealed type is "{expected}"' in revealed, revealed


@pytest.mark.slow
def test_the_subject_itself_type_checks(revealed):
    # Revealing a type is not enough on its own: a file full of errors can
    # still reveal the right things. This is what makes the reveal meaningful.
    assert "error:" not in revealed, revealed
