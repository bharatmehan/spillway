"""Keep the type checking subject away from the runtime collector."""

# It calls reveal_type, which exists for a type checker and not for an
# interpreter, so importing it raises. It is an input to a test rather than a
# test, and the test that feeds it to mypy lives beside it.
collect_ignore = ["instrumented_client.py"]
