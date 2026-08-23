# Spillway

Congestion control for language model API traffic.

A Python library that decides whether a call to a language model API should proceed now, wait, or
be refused. It works on the resources that actually run out under load: request rate, concurrency,
and the amount of unfinished generation already committed. It is a library you import, not a
gateway you deploy.

## Two lines

```python
from anthropic import AsyncAnthropic
from spillway import Spillway

client = Spillway.instrument(AsyncAnthropic(), rpm=1_000, input_tpm=2_000_000)

reply = await client.messages.create(model=..., messages=..., max_tokens=1_024)
```

Added once, where the client is built. Every call site stays exactly as it was.

The limits are yours. This library ships none of its own, because a rate limit belongs to an
account rather than to a provider, and the real figures are on your provider's own page. Name the
ones it gives you, or name none of them and Spillway will admit everything, record what your
traffic really costs, and tell you what it saw.

## Status

Under development. The version number is deliberately below 0.1 and anything may change, names
and behaviour included.

What works today: admission control across request rate and concurrency in a single process, with
capacity reserved before a call and settled against the real cost after it, and with waiting,
priority and timeouts rather than a bare refusal. Output length is predicted per route from what
that route has actually produced, so a request reserves what most of its kind come in under
instead of the maximum the caller allowed. Anthropic and OpenAI clients can be instrumented
directly, and anything speaking the OpenAI protocol against another service is recognised as
such rather than assumed to be OpenAI.

What does not exist yet: occupancy, streaming, one limit shared across processes, and the
synchronous API.

If you want to know when that changes, watch this repository or read the changelog.

## License

Apache 2.0. See LICENSE and NOTICE.
