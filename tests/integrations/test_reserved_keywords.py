"""What Spillway reads off a call, and what the provider gets to see."""

from spillway.core.scope import Priority, Scope
from spillway.integrations.context import scope_context
from spillway.integrations.instrument import RESERVED, resolve


def test_an_ordinary_call_is_forwarded_untouched():
    call = {"model": "m", "messages": [], "max_tokens": 10}
    assert resolve(call)[0] == call


def test_every_reserved_keyword_is_taken_out():
    # They must never reach the wire. Both client libraries reject an unknown
    # keyword before sending, so one that leaked would turn every call into a
    # client error rather than into a quiet extra field.
    call = {"model": "m", **dict.fromkeys(RESERVED, None)}
    assert resolve(call)[0] == {"model": "m"}


def test_a_keyword_scope_is_read():
    _, found = resolve({"spillway_scope": "tenant:acme"})
    assert found.scope == Scope("tenant:acme")


def test_a_keyword_priority_and_tags_are_read():
    _, found = resolve({"spillway_priority": Priority.BATCH, "spillway_tags": {"task": "sum"}})
    assert found.priority == Priority.BATCH
    assert found.tags == {"task": "sum"}


def test_the_client_default_is_used_when_nothing_else_says():
    _, found = resolve({}, scope=Scope("tenant:default"), priority=Priority.BACKGROUND)
    assert found.scope.key == "tenant:default"
    assert found.priority == Priority.BACKGROUND


def test_the_surrounding_context_beats_the_client_default():
    with scope_context("tenant:request"):
        _, found = resolve({}, scope=Scope("tenant:default"))
    assert found.scope.key == "tenant:request"


def test_a_keyword_beats_the_surrounding_context():
    # Somebody writing it at a call site is being specific on purpose.
    with scope_context("tenant:request"):
        _, found = resolve({"spillway_scope": "tenant:explicit"}, scope=Scope("tenant:default"))
    assert found.scope.key == "tenant:explicit"


def test_the_whole_chain_in_one_call():
    with scope_context("tenant:request", priority=Priority.INTERACTIVE, tags={"task": "chat"}):
        _, found = resolve(
            {"spillway_priority": Priority.BATCH, "spillway_tags": {"step": "draft"}},
            scope=Scope("tenant:default"),
            priority=Priority.NORMAL,
        )
    assert found.scope.key == "tenant:request"
    assert found.priority == Priority.BATCH
    assert found.tags == {"task": "chat", "step": "draft"}


def test_keyword_tags_add_to_the_surrounding_ones():
    with scope_context(tags={"task": "chat"}):
        _, found = resolve({"spillway_tags": {"task": "summarise", "step": "draft"}})
    assert found.tags == {"task": "summarise", "step": "draft"}


def test_nothing_anywhere_resolves_to_nothing():
    _, found = resolve({})
    assert found.scope is None
    assert found.priority is None
    assert found.tags == {}


def test_the_reserved_names_are_prefixed():
    # So they cannot collide with a provider parameter now or after a provider
    # adds one.
    assert all(name.startswith("spillway_") for name in RESERVED)
