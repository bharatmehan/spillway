# Spillway

Congestion control for language model API traffic.

A Python library that decides whether a call to a language model API should proceed now, wait, or
be refused. It works on the resources that actually run out under load: request rate, concurrency,
and the amount of unfinished generation already committed. It is a library you import, not a
gateway you deploy.

## Status

Under development. The version number is deliberately below 0.1 and anything may change, names
and behaviour included.

What works today: admission control across request rate and concurrency in a single process, with
capacity reserved before a call and settled against the real cost after it, and with waiting,
priority and timeouts rather than a bare refusal. Output length is predicted per route from what
that route has actually produced, so a request reserves what most of its kind come in under
instead of the maximum the caller allowed.

What does not exist yet: occupancy, provider adapters, one limit shared across processes, and the
synchronous API.

If you want to know when that changes, watch this repository or read the changelog.

## License

Apache 2.0. See LICENSE and NOTICE.
