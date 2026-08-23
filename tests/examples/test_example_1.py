"""The numbered example runs, and does what it says it does."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.servers.mock_provider import MockAnthropic, serving

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "01-quickstart" / "main.py"


def _environment(base_url):
    """The example's environment: this one, pointed at the mock provider.

    Inherited rather than replaced. A bare environment looks hermetic and is
    not portable: on Windows the event loop cannot load its socket provider
    without the variables that say where the system lives, so the example
    fails before it reaches a line of its own code.

    The real credential is removed rather than left in place. The base URL
    already sends every request to a local socket, so it would not reach the
    provider either way, but a test has no business handling somebody's key at
    all.
    """
    environment = {name: value for name, value in os.environ.items() if name != "ANTHROPIC_API_KEY"}
    environment["SPILLWAY_EXAMPLE_BASE_URL"] = base_url
    return environment


def _reported(output, name):
    """The used figure the example printed for one limit."""
    for line in output.splitlines():
        if line.startswith(f"{name}: "):
            return line.split(": ", 1)[1].split(" of ", 1)[0].replace(",", "")
    raise AssertionError(f"{name} was never reported:\n{output}")


@pytest.mark.integration
def test_the_quickstart_runs_and_reports_what_was_used():
    # A broken example is worse than none, because it is the first code anyone
    # copies. This runs the real file as a real script.
    provider = MockAnthropic(input_tokens=12, output_tokens=25)
    with serving(provider) as base_url:
        finished = subprocess.run(  # noqa: S603
            [sys.executable, str(EXAMPLE)],
            capture_output=True,
            text=True,
            env=_environment(base_url),
            check=False,
            timeout=60,
        )

    assert finished.returncode == 0, finished.stderr
    assert provider.calls == 3
    assert finished.stdout.count("answered with 25 output tokens") == 3

    # Every limit the example named is reported back.
    for name, limit in (("rpm", "1,000"), ("input_tpm", "2,000,000"), ("output_tpm", "400,000")):
        assert f"{name}: " in finished.stdout
        assert f"of {limit}" in finished.stdout

    # The claim the example is making, and the reason it is worth shipping.
    # Three calls each reserved 4096 output tokens, so 12,288 were held, and
    # each gave back everything it did not use the moment the real figure was
    # known. Anything close to the reserved figure would mean the credit back
    # never happened. Not an exact number, because the example runs on a real
    # clock and a per minute window drains while it is still printing.
    held = int(_reported(finished.stdout, "output_tpm"))
    assert held < 100, f"reserved 12,288 and still holding {held}"
