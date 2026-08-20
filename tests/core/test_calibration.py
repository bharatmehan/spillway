"""The comparison that motivates the whole design.

A synthetic bimodal workload, run twice: once reserving the maximum a caller
allowed, and once reserving what the route has actually been producing. The
second fits far more concurrent requests under the same limit, and its overrun
rate settles near what its quantile promised.

Assertions are on the direction of the difference and on the overrun rate,
never on a specific throughput number. A threshold here would be a number
nobody could defend, and the first thing to fail on someone else's machine.
"""

import random

from spillway.core.clock import FakeClock
from spillway.core.errors import SpillwayError
from spillway.core.scope import Scope
from spillway.core.spillway import Spillway
from spillway.dimensions.rate import Rate
from spillway.estimators.base import RequestContext
from spillway.estimators.max_tokens import MaxTokensEstimator
from spillway.estimators.quantile import QuantileEstimator
from spillway.stores.memory import MemoryStore

ACME = Scope("tenant:acme")
MAX_TOKENS = 4_096
LEARNING_REQUESTS = 500
LIMIT = 200_000.0
WINDOW_S = 60.0
WINDOW_MS = WINDOW_S * 1_000.0
CONTEXT = RequestContext(max_tokens=MAX_TOKENS, tags={"task": "reply"})


def workload(count, seed=20260820):
    """Nine short answers in every ten, and one long one. Deliberately awkward.

    A single mode would let any estimator look good. The long tail is what
    makes reserving the maximum expensive and reserving a median dangerous, so
    it is the shape worth testing against.
    """
    generator = random.Random(seed)  # noqa: S311 - a synthetic workload, not a secret
    return [
        int(generator.gauss(3_000, 200))
        if generator.random() < 0.1
        else int(generator.gauss(300, 40))
        for _ in range(count)
    ]


def build(estimator, clock):
    return Spillway(
        dimensions=[Rate("output_tpm", limit=LIMIT, window=WINDOW_S)],
        store=MemoryStore(clock=clock),
        clock=clock,
        scope=ACME,
        estimator=estimator,
        default_timeout=0,
    )


def admit(limiter):
    return limiter.admit(max_tokens=MAX_TOKENS, tags={"task": "reply"})


async def learn(limiter, clock, lengths):
    """Admit and settle each request in turn, and report the overruns.

    Sequential, so each reservation is handed back before the next is asked
    for. This measures the prediction rather than the limit.
    """
    admitted = 0
    overruns = 0
    reserved_total = 0
    for length in lengths:
        try:
            lease = await admit(limiter).acquire()
        except SpillwayError:
            clock.advance(50)
            continue
        admitted += 1
        reserved_total += lease.reserved.output_tokens
        overruns += length > lease.reserved.output_tokens
        lease.settle(input=0, output=length)
        clock.advance(50)
    return admitted, reserved_total, overruns


async def how_many_fit_at_once(limiter):
    """Admit until the limit refuses, holding every lease.

    This is where the size of a reservation actually costs something. Settled
    one at a time, an oversized reservation is handed straight back and never
    binds; held concurrently, it is the whole difference between the two
    estimators.
    """
    held = []
    while True:
        try:
            held.append(await admit(limiter).acquire())
        except SpillwayError:
            return len(held)


def taught(clock):
    """A quantile estimator keyed on the task, ready to watch the workload."""
    return QuantileEstimator(
        route_key=lambda context: context.tags.get("task"),
        min_samples=30,
    )


async def test_learning_the_route_fits_far_more_at_once_than_reserving_the_maximum():
    # The whole argument for the library, in one assertion. Same workload, same
    # limit. One reserves 4,096 tokens a request because that is what the
    # caller allowed. The other reserves what the route has been producing.
    lengths = workload(LEARNING_REQUESTS)

    clock = FakeClock()
    blunt = build(MaxTokensEstimator(), clock)
    await learn(blunt, clock, lengths)
    clock.advance(WINDOW_MS * 2)
    blunt_concurrent = await how_many_fit_at_once(blunt)

    clock = FakeClock()
    estimator = taught(clock)
    learned = build(estimator, clock)
    await learn(learned, clock, lengths)
    clock.advance(WINDOW_MS * 2)
    learned_concurrent = await how_many_fit_at_once(learned)

    assert learned_concurrent > blunt_concurrent * 4


async def test_learning_the_route_reserves_far_less_per_request():
    # Two measures, because they say different things. The average over the
    # whole run includes the opening requests, which reserve the maximum
    # because the route has no history yet and a measurement that does not
    # exist must not bind. The converged figure is what the route costs once it
    # has been watched, and it is the one the argument rests on.
    lengths = workload(LEARNING_REQUESTS)
    clock = FakeClock()
    estimator = taught(clock)
    admitted, reserved_total, _ = await learn(build(estimator, clock), clock, lengths)
    settled = estimator.estimate(CONTEXT)
    assert reserved_total / admitted < MAX_TOKENS / 2
    assert settled.output.quantile(settled.quantile) < MAX_TOKENS / 8


async def test_the_overrun_rate_settles_near_what_the_quantile_promised():
    # Reserving at the ninth decile is a claim that about one request in ten
    # runs over. If the claim is not true the number is decoration.
    lengths = workload(LEARNING_REQUESTS)
    clock = FakeClock()
    limiter = build(taught(clock), clock)
    admitted, _, overruns = await learn(limiter, clock, lengths)
    assert 0.0 < overruns / admitted < 0.25


async def test_a_higher_quantile_reserves_more_and_overruns_less():
    lengths = workload(LEARNING_REQUESTS)
    results = {}
    for quantile in (0.6, 0.99):
        clock = FakeClock()
        estimator = QuantileEstimator(
            route_key=lambda context: context.tags.get("task"),
            min_samples=30,
            quantile=quantile,
        )
        admitted, reserved_total, overruns = await learn(build(estimator, clock), clock, lengths)
        results[quantile] = (reserved_total / admitted, overruns / admitted)
    assert results[0.99][0] > results[0.6][0]
    assert results[0.99][1] < results[0.6][1]


async def test_the_reserved_amount_stops_moving():
    # It converges rather than wandering. An estimator that never settled would
    # make every capacity decision downstream of it unrepeatable.
    clock = FakeClock()
    estimator = taught(clock)
    limiter = build(estimator, clock)
    await learn(limiter, clock, workload(LEARNING_REQUESTS))
    first = estimator.estimate(CONTEXT)
    await learn(limiter, clock, workload(200, seed=7))
    second = estimator.estimate(CONTEXT)
    settled = first.output.quantile(first.quantile)
    moved = second.output.quantile(second.quantile)
    assert abs(moved - settled) / settled < 0.2
