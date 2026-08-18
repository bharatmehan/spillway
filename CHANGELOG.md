# Changelog

Notable changes to this project, newest first. This project uses semantic versioning. While the
version is below 0.1, any release may change anything.

## Unreleased

### Added

- A `Clock` protocol with a monotonic implementation and a fake one that advances by hand. Every
  time reference in the library goes through it, which is what makes rate windows, lease expiry and
  feedback control testable without sleeping.
- A `Cost` value type: input tokens, output tokens, requests, and provider specific extra
  categories. Subtraction keeps the sign, because settlement is a difference and a negative
  component is an overrun to repay rather than a number to discard.
- `Distribution` and `Estimate`. Output length is predicted rather than known, so it is carried as
  a distribution with a `quantile` method. Two constructors for now: an exact point, and a worst
  case bound.
- A default estimate function. Input tokens are counted with a documented character heuristic that
  is accurate to roughly ten to fifteen percent, so the quickstart needs no tokenizer installed, and
  the real figure replaces it at settlement.
- `Scope` and `Priority`. A scope is the key every limit is tracked against; a priority is an
  ordinary integer, with four named conventions, where negative means the work is sheddable.
- The complete exception hierarchy under a single `SpillwayError` base. A refusal carries which
  dimension bound and how long until it would not have, so a caller can act rather than merely
  fail. The missing dependency error names the exact install command.
- Rate reservation arithmetic, using the generic cell rate algorithm. The whole state for a rate
  key is one float, so memory per key is constant no matter how much traffic passes, and a refusal
  reports how long until the same charge would fit.
- Credit and debt arithmetic for rate keys. Unused capacity is returned within a request's own
  lifetime, which is what makes reserving conservatively affordable, and an overrun becomes debt
  repaid from the next window, bounded so one bad estimate cannot silence a scope for hours.
- Gauge arithmetic, for limits on a value currently held rather than consumed over a window.
  Concurrency is one. Releasing is clamped at zero, because a gauge below zero would admit more
  than its limit.
- The types a store speaks in: `Claim`, `Delta`, `Utilisation` and `ReserveResult`. A refusal names
  the key that bound and how long until it would not have, because a bare yes or no cannot be
  explained to a user and forces a caller to poll.
- A `Dimension` protocol. A dimension turns a cost into a claim and a settlement into a correction,
  and does nothing else: it never decides whether a claim fits, because that decision has to be
  made for every dimension at once or a request gets admitted against two limits and refused by the
  third with the first two already spent.
- A `Rate` dimension, for limits consumed over a rolling window. It declares which part of a
  request's cost it counts, so a tokens per minute limit and a requests per minute limit can sit
  side by side without either counting the other's units. Asking for an adaptive rate limit is
  refused with an explanation rather than accepted.
- The meter is inferred from the dimension name for `rpm`, `rpd`, `input_tpm`, `output_tpm` and
  `tpm`, so the common case is one line. Any other name must say what it counts, because guessing
  would meter the wrong thing silently.
- A `Concurrency` dimension, limiting how many requests are in flight at once. One request takes
  one slot whatever it costs, and gets it back whole at settlement however wrong the estimate was.
- `Store` and `SyncStore` protocols. A store is asked for a whole batch of claims at once and
  applies all of them or none, which is the only way a request cannot be admitted against two
  limits and refused by the third with the first two already spent. One class may implement both.
- `MemoryStore`, the default store. Zero configuration and zero dependencies, so the quickstart
  runs on a clean environment. It is not safe across processes, and its docstring says so first:
  four workers each running one enforce the full limit four times over.
- Leases that are never settled are reclaimed once they outlive their expiry, so a process that
  dies mid request cannot leak a gauge. Only gauges come back: a rate charge was really spent, and
  inventing a refund would let a crashing worker exceed a provider's limit indefinitely.
- A warning, once per process, when an in memory store is used and the process looks like one of
  several workers. The overshoot this causes appears at the provider, and nothing locally points at
  the cause, so the warning is the only thing that connects the two.

### Changed

- Async tests need no decorator. The library's entry point is asynchronous, so an async test is the
  ordinary case here rather than the exception.
- The test run now executes every example in a docstring. A public docstring example is the first
  thing a reader copies, so an example that has stopped working is a defect rather than a typo.

## 0.0.1 (2026-08-17)

### Added

- Project skeleton: packaging, linting, strict type checking, test runner, and continuous
  integration across Python 3.10 through 3.14.
- The package imports and reports its version. There is no public API yet.
