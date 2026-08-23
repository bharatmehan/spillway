"""The context variable is how scope reaches a call nobody passed it to."""

import asyncio

import pytest

from spillway.core.scope import Priority, Scope
from spillway.integrations.context import current, scope_context


def test_nothing_is_set_by_default():
    assert current().scope is None
    assert current().priority is None
    assert current().tags == {}


def test_it_reaches_a_call_several_frames_below():
    # The whole point. No function in between knows this library exists.
    def deep():
        return current().scope.key

    def middle():
        return deep()

    with scope_context("tenant:acme"):
        assert middle() == "tenant:acme"


async def test_it_survives_an_await():
    async def after_a_suspension():
        await asyncio.sleep(0)
        return current().scope.key

    with scope_context("tenant:acme"):
        assert await after_a_suspension() == "tenant:acme"


async def test_a_task_started_inside_inherits_it():
    async def in_a_task():
        return current().scope.key

    with scope_context("tenant:acme"):
        assert await asyncio.create_task(in_a_task()) == "tenant:acme"


async def test_a_task_started_outside_does_not():
    # Context variables copy at task creation, so a task made before the block
    # is unaffected by it. Worth pinning: it is the behaviour that makes this
    # safe under concurrency, and it would be surprising if it changed.
    started = asyncio.Event()
    seen = []

    async def waiting():
        started.set()
        await asyncio.sleep(0.01)
        seen.append(current().scope)

    task = asyncio.create_task(waiting())
    await started.wait()
    with scope_context("tenant:acme"):
        await task
    assert seen == [None]


def test_nesting_adds_rather_than_replaces():
    # Set the tenant once at the edge of a request, mark one call as batch work
    # deep inside it, and keep both.
    outer = scope_context("tenant:acme", tags={"task": "chat"})
    inner = scope_context(priority=Priority.BATCH, tags={"step": "draft"})
    with outer, inner as found:
        assert found.scope.key == "tenant:acme"
        assert found.priority == Priority.BATCH
        assert found.tags == {"task": "chat", "step": "draft"}


def test_an_inner_block_can_override_the_outer_one():
    with scope_context("tenant:acme"):
        with scope_context("tenant:other") as inner:
            assert inner.scope.key == "tenant:other"
        assert current().scope.key == "tenant:acme"


def test_it_restores_on_the_way_out():
    with scope_context("tenant:acme"):
        pass
    assert current().scope is None


def test_it_restores_when_the_block_raises():
    # The failure path is where a leaked scope would do real damage: every
    # later call in the process would be charged to whichever tenant last
    # failed.
    with pytest.raises(RuntimeError), scope_context("tenant:acme"):
        raise RuntimeError("the call failed")
    assert current().scope is None


def test_it_accepts_a_scope_object_as_well_as_a_string():
    with scope_context(Scope("tenant:acme")) as found:
        assert found.scope == Scope("tenant:acme")
