"""Provider servers that speak the real wire protocols, with no network.

Both client libraries accept an HTTP client built on a transport of our own,
and copying a client carries that transport through, so a whole provider is a
function from a request to a response: no socket, no thread, no port, and
nothing to flake.

The same handlers are mounted on a real server for the example test, which
needs an address something can be pointed at. One handler, two consumers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import httpx2


@dataclass
class MockProvider:
    """A provider that answers, counts, and can be pushed into refusing.

    Attributes:
        input_tokens: What every response reports having been sent.
        output_tokens: What every response reports having generated.
        capacity: How many requests to answer before refusing. None never
            refuses.
        retry_after: What a refusal asks the caller to wait.

    Example:
        >>> provider = MockProvider()
        >>> provider.seen
        []
    """

    input_tokens: int = 100
    output_tokens: int = 25
    capacity: int | None = None
    retry_after: int = 12
    seen: list[dict[str, object]] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def calls(self) -> int:
        """How many requests have arrived."""
        return len(self.seen)

    def transport(self) -> httpx2.MockTransport:
        """A transport that answers as this provider would."""
        return httpx2.MockTransport(self._handle)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        """Answer one request, recording what was asked for."""
        body = json.loads(request.content) if request.content else {}
        self.seen.append(body)
        if self.capacity is not None and self.calls > self.capacity:
            return self._refuse()
        return self._answer(str(request.url))

    def _refuse(self) -> httpx2.Response:
        """The shape a rate limit actually arrives in."""
        raise NotImplementedError

    def _answer(self, url: str) -> httpx2.Response:
        """The shape a success actually arrives in."""
        raise NotImplementedError


class MockAnthropic(MockProvider):
    """Answers as the Anthropic messages API does."""

    def _answer(self, url: str) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={
                "anthropic-ratelimit-requests-limit": "1000",
                "anthropic-ratelimit-requests-remaining": "998",
                "anthropic-ratelimit-input-tokens-limit": "2000000",
                "anthropic-ratelimit-input-tokens-remaining": "1907000",
                **self.headers,
            },
            json={
                "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": "Here is the summary."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        )

    def _refuse(self) -> httpx2.Response:
        return httpx2.Response(
            429,
            headers={"retry-after": str(self.retry_after)},
            json={
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Number of request tokens has exceeded your rate limit.",
                },
                "request_id": "req_011CQjmRUJTVsBscDkgxLXFc",
            },
        )


class MockOpenAI(MockProvider):
    """Answers as the OpenAI chat completions and responses APIs do."""

    def _answer(self, url: str) -> httpx2.Response:
        if url.rstrip("/").endswith("/responses"):
            return self._responses()
        return self._chat()

    def _chat(self) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={
                "x-ratelimit-limit-requests": "60",
                "x-ratelimit-remaining-requests": "59",
                "x-ratelimit-limit-tokens": "150000",
                "x-ratelimit-remaining-tokens": "149984",
                "x-ratelimit-reset-tokens": "6m0s",
                **self.headers,
            },
            json={
                "id": "chatcmpl-B9MHDbslfkBeAs8l4bebGdFOJ6PeG",
                "object": "chat.completion",
                "created": 1771753000,
                "model": "gpt-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Here is the summary."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": self.input_tokens,
                    "completion_tokens": self.output_tokens,
                    "total_tokens": self.input_tokens + self.output_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    def _responses(self) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers=dict(self.headers),
            json={
                "id": "resp_67ccd2bed1ec8190b14f964abc054267",
                "object": "response",
                "created_at": 1771753000,
                "model": "gpt-5",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Here is the summary.",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": self.input_tokens,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": self.output_tokens,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": self.input_tokens + self.output_tokens,
                },
            },
        )

    def _refuse(self) -> httpx2.Response:
        return httpx2.Response(
            429,
            headers={"retry-after": str(self.retry_after)},
            json={
                "error": {
                    "message": "Rate limit reached on tokens per min (TPM).",
                    "type": "tokens",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
        )


@contextmanager
def serving(provider: MockProvider) -> Iterator[str]:
    """Run `provider` on a real socket, and yield the address to point at.

    The transport is enough for everything in this repository, because the
    client libraries let one be injected. An example cannot do that: it has to
    be a script somebody can run, so it takes an address instead. Same handlers,
    two ways in.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", 0))
            body = self.rfile.read(length)
            answer = provider._handle(
                httpx2.Request("POST", f"http://localhost{self.path}", content=body)
            )
            payload = answer.content
            self.send_response(answer.status_code)
            for name, value in answer.headers.items():
                if name.lower() not in ("content-length", "transfer-encoding"):
                    self.send_header(name, value)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_: object) -> None:
            """Stay quiet. A test that passes should print nothing."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
