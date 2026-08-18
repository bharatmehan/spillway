"""Scope coercion and the priority convention."""

import pytest

from spillway.core.scope import DEFAULT_SCOPE, Priority, Scope


def test_a_scope_carries_its_key():
    assert Scope("tenant:acme").key == "tenant:acme"


def test_scopes_with_the_same_key_are_equal():
    assert Scope("tenant:acme") == Scope("tenant:acme")


def test_a_scope_prints_as_its_key():
    assert f"{Scope('tenant:acme')}" == "tenant:acme"


def test_a_scope_is_hashable_so_it_can_key_a_mapping():
    assert {Scope("a"): 1}[Scope("a")] == 1


@pytest.mark.parametrize("blank", ["", " ", "\t\n"])
def test_a_blank_key_is_refused(blank):
    # A blank key would merge every caller into one budget while looking like
    # it was isolating them, which is worse than failing loudly at construction.
    with pytest.raises(ValueError, match="cannot be blank"):
        Scope(blank)


def test_coercion_passes_a_scope_through_unchanged():
    scope = Scope("tenant:acme")
    assert Scope.of(scope) is scope


def test_coercion_wraps_a_string():
    assert Scope.of("user:123") == Scope("user:123")


def test_coercion_of_nothing_gives_the_default_scope():
    assert Scope.of(None) is DEFAULT_SCOPE


def test_priorities_order_from_interactive_down_to_batch():
    assert Priority.INTERACTIVE > Priority.NORMAL > Priority.BACKGROUND > Priority.BATCH


def test_sheddable_priorities_are_exactly_the_negative_ones():
    assert Priority.BACKGROUND < 0
    assert Priority.BATCH < 0
    assert Priority.NORMAL == 0
    assert Priority.INTERACTIVE > 0


def test_a_priority_is_an_ordinary_integer():
    # The enumeration is a convention, not a closed set. A caller with finer
    # bands must be able to pass their own number without subclassing anything.
    assert Priority.NORMAL == 0
    assert sorted([Priority.BATCH, 7, Priority.INTERACTIVE]) == [Priority.BATCH, 7, 100]
