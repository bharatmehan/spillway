"""Two lines where the client is built, and every call site untouched.

Run it against the real provider:

    export ANTHROPIC_API_KEY=...
    uv run python examples/01-quickstart/main.py

Or against nothing at all, which is how the test suite runs it:

    SPILLWAY_EXAMPLE_BASE_URL=http://localhost:8080 uv run python examples/01-quickstart/main.py
"""

from __future__ import annotations

import asyncio
import os

from anthropic import AsyncAnthropic

from spillway import Spillway

PROMPT = "Name one river, and nothing else."


async def main() -> None:
    """Make three calls through an instrumented client and report the cost."""
    # The two lines. Everything below them is ordinary client code, and it
    # would be identical if Spillway were not here at all.
    #
    # The limits are the ones this account actually has. Spillway ships none of
    # its own: a rate limit belongs to an account rather than to a provider,
    # and the real figures are on the provider's own limits page.
    client = Spillway.instrument(
        AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "not-a-real-key"),
            base_url=os.environ.get("SPILLWAY_EXAMPLE_BASE_URL"),
        ),
        rpm=1_000,
        input_tpm=2_000_000,
        output_tpm=400_000,
    )

    for _ in range(3):
        reply = await client.messages.create(
            model="claude-sonnet-5",
            max_tokens=4_096,
            messages=[{"role": "user", "content": PROMPT}],
        )
        print(f"answered with {reply.usage.output_tokens} output tokens")  # noqa: T201

    # Each of those reserved 4096 output tokens at admission and gave back
    # everything it did not use, the moment the real figure was known. What is
    # left holding capacity is what was actually generated.
    found = Spillway.of(client).snapshot()
    for name, utilisation in found.dimensions.items():
        print(f"{name}: {utilisation.used:,.0f} of {utilisation.limit:,.0f}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
