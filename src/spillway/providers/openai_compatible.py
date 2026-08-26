"""Anything speaking the OpenAI schema against something that is not OpenAI.

vLLM, SGLang, Together, Fireworks, Groq, DeepInfra, LM Studio, Ollama, and a
model somebody is serving on their own hardware this afternoon.

Detection by protocol alone would call all of these OpenAI, which is right about
the wire format and wrong about the accounting. A local engine charges no
requested maximum, reports different usage categories and sends different
headers, so a client speaking the right schema against another host lands here.

It assumes nothing, which is the point of it.
"""

from __future__ import annotations

from collections.abc import Mapping

from spillway.core.cost import Cost
from spillway.estimators.base import RequestContext
from spillway.providers.base import Outcome, ProviderState
from spillway.providers.openai import classify_failure, read_request, read_retry_after, read_usage


class OpenAICompatible:
    """The OpenAI wire format, with none of OpenAI's accounting assumed.

    Args:
        metrics_url: Where a self hosted engine exposes its own serving
            metrics, such as cache occupancy and queue depth. Accepted and
            unused: it is the seam a later release reads engine state through,
            and it is present now so that shape is not designed out.

    Example:
        >>> adapter = OpenAICompatible()
        >>> adapter.charges_max_tokens()
        False
        >>> adapter.official_hosts
        ()

        It reads the schema it speaks, and says nothing about limits.

        >>> adapter.usage_from({"usage": {"prompt_tokens": 8, "completion_tokens": 3}}).input_tokens
        8
        >>> adapter.parse_headers({"x-ratelimit-limit-tokens": "1000"}) is None
        True
    """

    name: str = "openai_compatible"
    client_module: str = "openai"
    official_hosts: tuple[str, ...] = ()
    endpoints: tuple[str, ...] = (
        "chat.completions.create",
        "chat.completions.parse",
        "responses.create",
        "responses.parse",
    )

    def __init__(self, *, metrics_url: str | None = None) -> None:
        """Hold the unused metrics address."""
        self._metrics_url = metrics_url

    def __repr__(self) -> str:
        """Name the provider."""
        return "OpenAICompatible()"

    @property
    def metrics_url(self) -> str | None:
        """Where this engine's own serving metrics are, if anywhere.

        Nothing reads it yet.
        """
        return self._metrics_url

    def request_from(self, endpoint: str, kwargs: Mapping[str, object]) -> RequestContext:
        """Read the model, the prompt and the requested maximum off one call."""
        return read_request(endpoint, kwargs)

    def adjust(self, cost: Cost, context: RequestContext) -> Cost:
        """Return the reservation unchanged.

        Nothing is known about how this service accounts, and inventing a rule
        for an unknown engine is exactly the confident wrongness that sending a
        client here rather than to the named adapter is meant to avoid.
        """
        del context
        return cost

    def usage_from(self, response: object) -> Cost:
        """Read what a call really cost, in whichever shape the engine sent.

        Most engines implementing this schema report usage the way the chat
        endpoint does. One that reports none at all raises, and the caller
        settles at the reserved amount and says so.
        """
        return read_usage(response)

    def classify(self, exc: BaseException | None, response: object | None) -> Outcome:
        """Say what happened, in the terms a controller acts on."""
        del response
        return classify_failure(exc)

    def parse_headers(self, headers: Mapping[str, str]) -> ProviderState | None:
        """Report nothing, because nothing here can be relied on.

        Some services behind this adapter send rate limit headers and some do
        not, and those that do disagree about what they mean. Reading them as
        though they were the named provider's would produce a controller acting
        confidently on a number it had guessed at. The adaptive controllers,
        which discover a limit from observed behaviour rather than from a
        claim, are the right tool here and they arrive in a later stage.
        """
        del headers
        return None

    def retry_after(self, exc: BaseException | None, headers: Mapping[str, str]) -> float | None:
        """Read a retry hint, which is the one header that means one thing."""
        return read_retry_after(exc, headers)

    def charges_max_tokens(self) -> bool:
        """No. A service that does is a service this adapter is wrong for."""
        return False
