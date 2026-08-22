# Provider fixtures

What each adapter's facts are checked against. One directory per provider, one file per
shape, and every file carries where it came from and when it was read.

```json
"source": {
  "url": "https://...",
  "read_on": "2026-08-22"
}
```

**Why these exist.** A provider accounting rule written into an adapter with nothing behind
it does not survive its first change. The provider alters a field name or a header, nothing
in the suite fails, and the library quietly starts computing the wrong reservation. Every
constant in every adapter traces to one of these files or to a dated entry in the limits
table.

**Adding to them.** Transcribe from the provider's current documentation and record the page
and the date, or capture a real response and scrub every secret from it: keys, organisation
identifiers, account identifiers and anything else that names a person or a payer. Never
invent a shape. A plausible fixture is worse than none, because it looks like evidence.

The conformance suite reads every file here, so a fixture that no adapter can parse fails
the build rather than sitting unused.
