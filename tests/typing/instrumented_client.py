"""Not run. Type checked, which is the whole point of it.

D4.5 requires the instrumented client to be the same type as the one handed in.
A runtime assertion cannot see that, because at runtime it plainly is that type
and the question is whether a type checker agrees. So this file is fed to the
type checker by a test, and what it reveals is the assertion.
"""

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from spillway.core.spillway import Spillway

built = Spillway.instrument(AsyncAnthropic(), rpm=1_000)
reveal_type(built)  # noqa: F821

limiter = Spillway(rpm=1_000)
given = limiter.instrument(AsyncOpenAI())
reveal_type(given)  # noqa: F821

recovered = Spillway.of(built)
reveal_type(recovered)  # noqa: F821
