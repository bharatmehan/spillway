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

### Changed

- The test run now executes every example in a docstring. A public docstring example is the first
  thing a reader copies, so an example that has stopped working is a defect rather than a typo.

## 0.0.1 (2026-08-17)

### Added

- Project skeleton: packaging, linting, strict type checking, test runner, and continuous
  integration across Python 3.10 through 3.14.
- The package imports and reports its version. There is no public API yet.
