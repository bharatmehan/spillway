default:
    @just --list

# Create the environment and install the project.
setup:
    uv sync

# Style and correctness, without changing anything.
lint:
    uv run ruff check .
    uv run ruff format --check .

# Apply every fix the linter and formatter can make on their own.
fix:
    uv run ruff check --fix .
    uv run ruff format .

types:
    uv run mypy

test:
    uv run pytest

test-fast:
    uv run pytest -m "not redis and not slow and not integration and not bench"

# The gate. If this passes locally, continuous integration passes.
check: lint types test-fast

# Run one numbered example against a mock provider, as its test does.
example number:
    uv run pytest tests/examples/test_example_{{number}}.py -m integration -q

build:
    uv build
