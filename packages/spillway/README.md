# Spillway

Congestion control for language model API traffic.

A Python library that decides whether a call to a language model API should proceed now, wait, or
be refused. It works on the resources that actually run out under load: request rate, concurrency,
and the amount of unfinished generation already committed. It is a library you import, not a
gateway you deploy.

## Status

Under development, and not yet usable. The version number is deliberately below 0.1 and there is
no public API yet. This release installs and reports its version. That is all it does.

Source and changelog: https://github.com/bharatmehan/spillway

## License

Apache 2.0.
