"""The waiting queue: who is in it, and who goes next."""

import asyncio

import pytest

from spillway.core.cost import Cost
from spillway.core.errors import AdmissionDenied, ConfigurationError, Shed
from spillway.core.queue import Waiter, WaitQueue
from spillway.core.scope import Scope


def waiting(priority=0, deadline_ms=None):
    return Waiter(
        claims=(),
        dimension_of_key={},
        scope=Scope("tenant:acme"),
        priority=priority,
        reserved=Cost(),
        deadline_ms=deadline_ms,
        queued_at_ms=0.0,
        future=asyncio.get_event_loop().create_future(),
        refusal=AdmissionDenied("no room"),
    )


@pytest.fixture
def queue():
    return WaitQueue()


async def test_an_empty_queue_has_nobody_to_select(queue):
    assert queue.depth == 0
    assert queue.select() is None
    assert queue.depths() == {}


async def test_the_highest_band_is_served_first(queue):
    low, high = waiting(priority=0), waiting(priority=100)
    queue.push(low)
    queue.push(high)
    assert queue.select() is high


async def test_arrival_order_is_kept_within_a_band(queue):
    first, second = waiting(), waiting()
    queue.push(first)
    queue.push(second)
    assert queue.select() is first
    queue.remove(first)
    assert queue.select() is second


async def test_a_negative_band_is_still_ordered_below_a_zero_one(queue):
    # Priority is any integer, not a closed set, so the bands are the values
    # themselves and a negative one has to sort where its number says.
    batch, normal = waiting(priority=-100), waiting(priority=0)
    queue.push(batch)
    queue.push(normal)
    assert queue.select() is normal


async def test_removing_a_waiter_that_is_not_there_does_nothing(queue):
    # Both sides remove, and neither can tell whether the other got there
    # first, so a second removal has to be harmless.
    waiter = waiting()
    queue.push(waiter)
    queue.remove(waiter)
    queue.remove(waiter)
    assert queue.depth == 0


async def test_two_identical_waiters_are_still_two_waiters(queue):
    # Compared by value they would be equal, and removing one would take the
    # other out of the queue with it.
    first, second = waiting(), waiting()
    queue.push(first)
    queue.push(second)
    queue.remove(first)
    assert queue.depth == 1
    assert queue.select() is second


async def test_depths_are_reported_per_band(queue):
    queue.push(waiting(priority=100))
    queue.push(waiting(priority=0))
    queue.push(waiting(priority=0))
    assert queue.depths() == {100: 1, 0: 2}
    assert queue.depth == 3


async def test_a_band_disappears_once_it_empties(queue):
    waiter = waiting(priority=100)
    queue.push(waiter)
    queue.remove(waiter)
    assert queue.depths() == {}


async def test_arrival_position_counts_who_was_already_ahead(queue):
    first, second, other_band = waiting(), waiting(), waiting(priority=100)
    queue.push(first)
    queue.push(second)
    queue.push(other_band)
    assert (first.position, second.position, other_band.position) == (0, 1, 0)


async def test_expiring_returns_only_the_waiters_whose_time_has_passed(queue):
    due, later, forever = (
        waiting(deadline_ms=100.0),
        waiting(deadline_ms=500.0),
        waiting(deadline_ms=None),
    )
    for waiter in (due, later, forever):
        queue.push(waiter)
    assert queue.expire(100.0) == [due]
    assert queue.depth == 2


async def test_expiring_reaches_every_band_not_just_the_one_being_served(queue):
    # A waiter behind a head that cannot be admitted is still owed its own
    # deadline. Checking only the selected waiter is what makes it wait for
    # ever instead.
    blocked_head = waiting(priority=100, deadline_ms=None)
    behind = waiting(priority=0, deadline_ms=100.0)
    queue.push(blocked_head)
    queue.push(behind)
    assert queue.expire(200.0) == [behind]
    assert queue.select() is blocked_head


async def test_the_earliest_deadline_is_the_one_to_wake_for(queue):
    assert queue.earliest_deadline_ms() is None
    queue.push(waiting(deadline_ms=None))
    assert queue.earliest_deadline_ms() is None
    queue.push(waiting(priority=100, deadline_ms=900.0))
    queue.push(waiting(deadline_ms=300.0))
    assert queue.earliest_deadline_ms() == 300.0


async def test_the_repr_shows_the_bands_highest_first(queue):
    queue.push(waiting(priority=0))
    queue.push(waiting(priority=100))
    assert repr(queue) == "WaitQueue({100: 1, 0: 1})"


async def test_a_full_band_refuses_a_new_arrival():
    queue = WaitQueue(capacity=2)
    queue.push(waiting())
    queue.push(waiting())
    with pytest.raises(AdmissionDenied, match="queue is full"):
        queue.push(waiting())
    assert queue.depth == 2


async def test_capacity_is_per_band_not_shared():
    # A flood of batch work must not consume the slots an interactive request
    # needs. This is the whole reason capacity is counted per band.
    queue = WaitQueue(capacity=1)
    queue.push(waiting(priority=-100))
    queue.push(waiting(priority=100))
    assert queue.depths() == {-100: 1, 100: 1}


async def test_a_refused_arrival_leaves_no_empty_band_behind():
    queue = WaitQueue(capacity=1)
    with pytest.raises(AdmissionDenied):
        queue.push(waiting(priority=0))
        queue.push(waiting(priority=0))
    queue.remove(queue.select())
    assert queue.depths() == {}


async def test_a_full_band_frees_up_once_a_waiter_leaves():
    queue = WaitQueue(capacity=1)
    first = waiting()
    queue.push(first)
    with pytest.raises(AdmissionDenied):
        queue.push(waiting())
    queue.remove(first)
    queue.push(waiting())
    assert queue.depth == 1


def test_a_queue_with_no_room_at_all_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="at least one waiter"):
        WaitQueue(capacity=0)


async def test_a_negative_priority_arrival_is_shed_when_its_band_is_full():
    # The whole shedding rule, and it needs no threshold to tune: under
    # pressure the low bands fill first and their arrivals bounce, while the
    # interactive band carries on because its capacity is its own.
    queue = WaitQueue(capacity=1)
    queue.push(waiting(priority=-100))
    with pytest.raises(Shed, match="could wait"):
        queue.push(waiting(priority=-100))


async def test_a_negative_priority_arrival_queues_normally_when_there_is_room():
    queue = WaitQueue(capacity=2)
    queue.push(waiting(priority=-100))
    queue.push(waiting(priority=-100))
    assert queue.depth == 2


async def test_a_shed_refusal_is_still_an_admission_denied():
    # So a caller who does not care about the distinction catches one thing.
    queue = WaitQueue(capacity=1)
    queue.push(waiting(priority=-50))
    with pytest.raises(AdmissionDenied):
        queue.push(waiting(priority=-50))


async def test_a_zero_priority_arrival_is_refused_rather_than_shed():
    queue = WaitQueue(capacity=1)
    queue.push(waiting(priority=0))
    with pytest.raises(AdmissionDenied) as raised:
        queue.push(waiting(priority=0))
    assert not isinstance(raised.value, Shed)


async def test_shed_lowest_drops_the_lowest_priority_waiter_for_a_higher_arrival():
    queue = WaitQueue(capacity=1, policy="shed_lowest")
    batch = waiting(priority=-100)
    queue.push(batch)
    queue.push(waiting(priority=0))
    urgent = waiting(priority=0)
    queue.push(urgent)
    assert batch.future.done()
    with pytest.raises(Shed, match="lowest priority"):
        await batch.future
    assert queue.select() is not batch


async def test_shed_lowest_refuses_an_arrival_that_is_itself_the_lowest():
    queue = WaitQueue(capacity=1, policy="shed_lowest")
    queue.push(waiting(priority=100))
    queue.push(waiting(priority=0))
    with pytest.raises(AdmissionDenied):
        queue.push(waiting(priority=0))


async def test_shed_lowest_never_grows_the_total_number_waiting():
    # A band may sit over capacity by one for each waiter it displaced, which
    # is what makes the total the bound that matters rather than the band.
    queue = WaitQueue(capacity=1, policy="shed_lowest")
    queue.push(waiting(priority=-100))
    queue.push(waiting(priority=0))
    before = queue.depth
    queue.push(waiting(priority=0))
    assert queue.depth == before


async def test_shed_lowest_takes_the_newest_of_the_lowest_band():
    # The one that has waited least, so waiting is never wasted.
    queue = WaitQueue(capacity=2, policy="shed_lowest")
    older, newer = waiting(priority=-100), waiting(priority=-100)
    queue.push(older)
    queue.push(newer)
    queue.push(waiting(priority=0))
    queue.push(waiting(priority=0))
    queue.push(waiting(priority=0))
    assert newer.future.done()
    assert not older.future.done()
    newer.future.exception()


async def test_reject_is_the_default_because_silent_shedding_surprises_people():
    queue = WaitQueue(capacity=1)
    batch = waiting(priority=-100)
    queue.push(batch)
    queue.push(waiting(priority=0))
    with pytest.raises(AdmissionDenied):
        queue.push(waiting(priority=0))
    assert not batch.future.done()


def test_an_unknown_queue_full_policy_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="not a queue full policy"):
        WaitQueue(policy="drop_oldest")
