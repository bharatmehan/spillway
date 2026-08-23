# Quickstart

The whole integration, which is two lines where the client is built:

```python
client = Spillway.instrument(AsyncAnthropic(), rpm=1_000, input_tpm=2_000_000)
```

Every call site below it is untouched. `client.messages.create(...)` is the
same call it was, and now it waits for capacity when there is none, reserves
what it is predicted to cost, and gives back whatever it did not use as soon as
the real figure is known.

## Running it

Against the real provider:

```
export ANTHROPIC_API_KEY=...
uv run python examples/01-quickstart/main.py
```

Against nothing, which is how the tests run it: point it at any server
speaking the same protocol.

```
SPILLWAY_EXAMPLE_BASE_URL=http://localhost:8080 uv run python examples/01-quickstart/main.py
```

## What to look at

Each call asks for up to 4096 output tokens and almost certainly produces far
fewer. Watch the final snapshot: what is still holding capacity is what was
actually generated, not what was reserved. That difference is why reserving
conservatively is affordable, and it is returned within the request's own
lifetime rather than at the end of a window.

## The limits are yours

Spillway ships no limit figures for any provider. A rate limit belongs to an
account rather than to a provider, and the true numbers are on your provider's
own limits page. Name the ones it gives you.

Naming none of them is also valid, and is the better first step if you do not
know your limits yet:

```python
client = Spillway.instrument(AsyncAnthropic())
```

That admits everything and records what the traffic really costs. Let it run,
read `Spillway.of(client).snapshot()`, and set a limit you can defend.
