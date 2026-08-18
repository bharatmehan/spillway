# Changelog

Notable changes to this project, newest first. This project uses semantic versioning. While the
version is below 0.1, any release may change anything.

## Unreleased

### Added

- A `Clock` protocol with a monotonic implementation and a fake one that advances by hand. Every
  time reference in the library goes through it, which is what makes rate windows, lease expiry and
  feedback control testable without sleeping.

## 0.0.1 (2026-08-17)

### Added

- Project skeleton: packaging, linting, strict type checking, test runner, and continuous
  integration across Python 3.10 through 3.14.
- The package imports and reports its version. There is no public API yet.
