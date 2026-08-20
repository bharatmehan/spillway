# Changelog

Notable changes to this project, newest first. This project uses semantic versioning. While the
version is below 0.1, any release may change anything.

## Unreleased

### Added

- An asynchronous sleep on the `Clock` protocol. Waiting is the one thing a limiter does that
  cannot be pure arithmetic, and routing it through the clock is what keeps it testable without
  the suite sleeping for real.
- Sleeping on the fake clock. A sleeper is released when the clock is advanced past its wake
  time, so a test can run ten minutes of waiting in a millisecond and get the same sequence of
  events every run.
- `admit()` refuses a timeout and a deadline given together. They say the same thing two ways and
  there is no honest answer when they disagree.
- A guard against a request that is larger than a limit it draws on. No amount of waiting makes
  it fit, so it is refused at once with the two numbers and the three ways to fix it, rather than
  waiting for a capacity that can never arrive.

## 0.0.2 (2026-08-18)

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
- `AdmissionExplanation`, carried by every decision either way. It reports how full every limit was,
  not just the one that ran out, because seeing what was not full is what tells someone the limit
  they were about to raise is not the one actually binding. It prints readably and converts to
  plain data.
- `Lease` and `LeaseState`. Settling reports the real cost and returns the difference immediately,
  so reserving conservatively costs nothing in steady state. Settling twice raises rather than
  counting the same request twice; abandoning twice does nothing, because it runs on the failure
  path where a second error buries the first.
- `Spillway`, the limiter itself, with a non blocking `admit`. Every argument has a default and
  `Spillway()` with none is valid: it tracks and reports and refuses nothing, which is a reasonable
  first step for someone gathering evidence before choosing limits. A refusal names the dimension
  the caller configured rather than an internal store key, reports the wait in seconds, and carries
  how full every limit was. A plain `with` statement refuses and names the asynchronous form rather
  than starting an event loop on the caller's behalf.
- `admit()` works as an asynchronous context manager, and handles all four ways a block can end.
  A raised exception or a cancelled task returns the whole reservation, because nothing was
  consumed. Leaving without settling charges the full reserved amount and says so once. A request
  that outran its expiry keeps its result: the bookkeeping failed, not the caller's work.
- `Spillway.snapshot()`, reporting how full every limit is for one scope without reserving
  anything, so it is safe to call from a health check. Limits come from the dimensions rather than
  from the store, so a dimension reports its real limit before its first request rather than
  appearing to have none.
- Property based testing, with a fixed seed in continuous integration and a random one locally. A
  fixed seed makes a red build reproducible on the machine of whoever has to fix it; a random one
  keeps the suite exploring rather than re examining the same cases for ever.
- Property tests for the six invariants the design rests on: reserve then release leaves no trace,
  settlement lands on the actual cost, a denied reservation consumes nothing, concurrent callers
  never exceed a limit in aggregate, no sliding window ever exceeds the rate, and outstanding
  leases sum to the gauge that is held.
- A curated top level export list: `Spillway`, `Scope`, `Priority`, `Rate`, `Concurrency`, `Cost`,
  `Estimate`, `Distribution`, `Lease`, `LeaseState` and the exception hierarchy. Everything else
  needs an explicit submodule import, so what an editor offers at the top level is what is
  supported.

### Fixed

- An explanation no longer prints a count as `1000.0/1000`. A rate window replenishes continuously,
  so a key that was exactly full a moment ago reads back a hair under, and showing that beside its
  limit made a correct limiter look like a broken one.

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
