# Contributing

Thank you for looking. This project is early, and the most useful contributions right now are
questions, reproductions, and opinions about the direction rather than large patches.

## Before you start

The version is below 0.1. Anything may change, including names and behaviour. If you are planning
more than a small change, open an issue first so you do not spend an evening on something that is
about to move.

## Setting up

You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```
git clone https://github.com/bharatmehan/spillway.git
cd spillway
just setup
```

That creates the environment and installs the package in editable mode. `just` on its own lists
every available task.

## The gate

```
just check
```

That runs the linter, the formatter check, strict type checking, and the fast tests. If it passes
locally, continuous integration passes. A divergence between the two is a fault in the workflow
configuration, not something you should work around.

`just fix` applies everything the linter and formatter can fix on their own.

## What a change should include

- Tests. Any non-trivial branch, loop, parser, or accounting path needs a test that fails if the
  logic breaks.
- A changelog entry, in `CHANGELOG.md`, under an "Unreleased" heading, describing the effect on a
  user rather than the mechanics of the patch.
- Type annotations. `mypy --strict` passes, and no `Any` appears in a public signature.
- Docstrings on anything public, with a runnable example.

House style, so that you are not surprised by a review comment:

- Commit messages follow conventional commits, with the module as the scope, for example
  `feat(dimensions): add occupancy gauge`.
- One concern per commit.
- No en dashes and no em dashes anywhere. Use a comma, a colon, a full stop, or parentheses.

## Licensing

By contributing, you agree that your contribution is licensed under Apache 2.0, the same license
as the project. There is no contributor licence agreement to sign, and there is no plan to
introduce one.

## Response time

This is maintained by one person alongside other work. Expect a first response within a week. If
something has gone quiet for longer than that, a polite nudge on the issue is welcome and will not
annoy anyone.

## Conduct

See CODE_OF_CONDUCT.md.
