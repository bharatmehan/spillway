"""The limiter asking an estimator what a request will cost."""

import asyncio

import pytest

from spillway.core.clock import FakeClock
from spillway.core.cost import Cost, Distribution, Estimate
from spillway.core.scope import Scope
from spillway.core.spillway import Spillway
from spillway.dimensions.rate import Rate
from spillway.estimators.base import Observation, RequestContext
from spillway.estimators.static import StaticEstimator
from spillway.stores.memory import MemoryStore

ACME = Scope("tenant:acme")


class Recorder:
    """A static estimator that remembers everything it was asked and told."""

    def __init__(self, output=200):
        self._output = Distribution.point(output)
        self.asked: list[RequestContext] = []
        self.told: list[Observation] = []

    def estimate(self, context):
        self.asked.append(context)
        return StaticEstimator(output=self._output).estimate(context)

    def record(self, observation):
        self.told.append(observation)


def build(clock, estimator=None, **kwargs):
    return Spillway(
        dimensions=[Rate("output_tpm", limit=10_000)],
        store=MemoryStore(clock=clock),
        clock=clock,
        scope=ACME,
        estimator=estimator,
        **kwargs,
    )


async def test_the_limiter_reserves_what_its_estimator_predicts():
    clock = FakeClock()
    limiter = build(clock, StaticEstimator(output=Distribution.point(200)))
    async with limiter.admit(max_tokens=4_096) as lease:
        assert lease.reserved.output_tokens == 200
        lease.settle(input=0, output=180)


async def test_the_default_estimator_reserves_the_requested_maximum():
    # Nobody's behaviour changes by adding the argument.
    clock = FakeClock()
    async with build(clock).admit(max_tokens=4_096) as lease:
        assert lease.reserved.output_tokens == 4_096
        lease.settle(input=0, output=180)


async def test_an_explicit_estimate_bypasses_the_estimator():
    clock = FakeClock()
    estimator = Recorder()
    limiter = build(clock, estimator)
    given = Estimate(input=10, output=Distribution.point(999))
    async with limiter.admit(estimate=given) as lease:
        assert lease.reserved.output_tokens == 999
        lease.settle(input=10, output=50)
    assert estimator.asked == []


async def test_the_estimator_sees_what_the_caller_named():
    clock = FakeClock()
    estimator = Recorder()
    limiter = build(clock, estimator)
    async with limiter.admit(prompt="hello there", max_tokens=512, model="claude") as lease:
        lease.settle(input=4, output=50)
    context = estimator.asked[0]
    assert (context.prompt, context.max_tokens, context.model) == ("hello there", 512, "claude")
    assert context.scope == ACME


async def test_a_request_is_estimated_once_however_long_it_waits():
    # The dispatcher retries the reservation, not the prediction. A prediction
    # that moved while a request queued would mean its place in the queue was
    # earned against a different request from the one finally admitted.
    clock = FakeClock()
    estimator = Recorder(output=6_000)
    limiter = build(clock, estimator, default_timeout=120)
    first = await limiter.admit().acquire()
    assert len(estimator.asked) == 1
    waiting = asyncio.ensure_future(limiter.admit().acquire())
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(estimator.asked) == 2
    first.settle(input=0, output=10)
    second = await asyncio.wait_for(waiting, timeout=1)
    second.settle(input=0, output=10)
    assert len(estimator.asked) == 2


@pytest.mark.parametrize("output", [0, 1, 4_096])
async def test_any_prediction_the_estimator_makes_is_what_gets_reserved(output):
    clock = FakeClock()
    limiter = build(clock, StaticEstimator(output=Distribution.point(output)))
    async with limiter.admit() as lease:
        assert lease.reserved == Cost(input_tokens=0, output_tokens=output, requests=1)
        lease.settle(input=0, output=output)


async def test_tags_reach_the_estimator():
    clock = FakeClock()
    estimator = Recorder()
    limiter = build(clock, estimator)
    async with limiter.admit(tags={"task": "summarise"}) as lease:
        lease.settle(input=0, output=50)
    assert estimator.asked[0].tags == {"task": "summarise"}


async def test_tags_default_to_none_at_all():
    clock = FakeClock()
    estimator = Recorder()
    limiter = build(clock, estimator)
    async with limiter.admit() as lease:
        lease.settle(input=0, output=50)
    assert estimator.asked[0].tags == {}


async def test_tags_are_copied_rather_than_held():
    # The context is handed to user supplied code and then kept as the key to a
    # route's history. A caller reusing one dictionary across requests would
    # otherwise rewrite the past every time they changed it.
    clock = FakeClock()
    estimator = Recorder()
    limiter = build(clock, estimator)
    mutable = {"task": "summarise"}
    async with limiter.admit(tags=mutable) as lease:
        lease.settle(input=0, output=50)
    mutable["task"] = "translate"
    assert estimator.asked[0].tags == {"task": "summarise"}


async def test_tags_do_not_change_what_is_admitted():
    # They are routing information for the estimator and nothing else. A tag
    # that quietly affected a limit would be a limit nobody configured.
    clock = FakeClock()
    limiter = build(clock)
    async with limiter.admit(max_tokens=100, tags={"task": "anything"}) as lease:
        assert lease.reserved.output_tokens == 100
        lease.settle(input=0, output=10)


async def test_a_settlement_reaches_the_estimator():
    clock = FakeClock()
    estimator = Recorder(output=200)
    limiter = build(clock, estimator)
    async with limiter.admit(max_tokens=4_096, tags={"task": "summarise"}) as lease:
        lease.settle(input=12, output=415)
    observation = estimator.told[0]
    assert observation.reserved.output_tokens == 200
    assert observation.actual.output_tokens == 415
    assert observation.context.tags == {"task": "summarise"}


async def test_a_settlement_records_when_it_happened():
    clock = FakeClock()
    estimator = Recorder()
    limiter = build(clock, estimator)
    async with limiter.admit() as lease:
        clock.advance(6_200)
        lease.settle(input=0, output=415)
    assert estimator.told[0].at_ms == 6_200


async def test_an_abandoned_request_teaches_nothing():
    # It produced no output at all. Recording a zero would drag every route's
    # history toward zero and make the next prediction worse for no reason.
    clock = FakeClock()
    estimator = Recorder()
    limiter = build(clock, estimator)
    with pytest.raises(ZeroDivisionError):
        async with limiter.admit():
            raise ZeroDivisionError("the call failed")
    assert estimator.told == []


async def test_a_settlement_that_outran_its_expiry_still_teaches():
    # The bookkeeping failed. The request still generated what it generated,
    # and that is the fact the estimator is learning.
    from spillway.core.errors import LeaseExpired
    from spillway.core.spillway import DEFAULT_LEASE_TTL_MS

    clock = FakeClock()
    estimator = Recorder()
    store = MemoryStore(clock=clock)
    limiter = Spillway(
        dimensions=[Rate("output_tpm", limit=10_000)],
        store=store,
        clock=clock,
        scope=ACME,
        estimator=estimator,
    )
    lease = await limiter.admit().acquire()
    clock.advance(DEFAULT_LEASE_TTL_MS + 1)
    store.snapshot_sync([])
    with pytest.raises(LeaseExpired):
        lease.settle(input=0, output=415)
    assert estimator.told[0].actual.output_tokens == 415


async def test_an_explicit_estimate_still_teaches():
    # The reservation was the caller's, the output length is the route's.
    clock = FakeClock()
    estimator = Recorder()
    limiter = build(clock, estimator)
    given = Estimate(input=10, output=Distribution.point(999))
    async with limiter.admit(estimate=given) as lease:
        lease.settle(input=10, output=50)
    assert estimator.told[0].actual.output_tokens == 50


async def test_a_request_that_waited_still_teaches():
    clock = FakeClock()
    estimator = Recorder(output=6_000)
    limiter = build(clock, estimator, default_timeout=120)
    first = await limiter.admit().acquire()
    waiting = asyncio.ensure_future(limiter.admit().acquire())
    for _ in range(20):
        await asyncio.sleep(0)
    first.settle(input=0, output=10)
    second = await asyncio.wait_for(waiting, timeout=1)
    second.settle(input=0, output=25)
    assert [told.actual.output_tokens for told in estimator.told] == [10, 25]
