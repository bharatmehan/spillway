"""The admission explanation."""

import dataclasses

import pytest

from spillway.observability.explain import AdmissionExplanation
from spillway.stores.base import Utilisation

DIMENSIONS = {
    "rpm": Utilisation(used=412.0, limit=1000.0),
    "generations": Utilisation(used=64.0, limit=64.0),
}


def refusal(**overrides):
    args = {
        "admitted": False,
        "scope": "tenant:acme",
        "priority": 0,
        "binding_dimension": "generations",
        "dimensions": DIMENSIONS,
    }
    args.update(overrides)
    return AdmissionExplanation(**args)


def test_a_refusal_names_the_limit_that_ran_out():
    assert "refused on generations" in str(refusal())


def test_a_grant_names_the_limit_that_came_closest():
    # On a grant the binding dimension is the one to raise if throughput needs
    # to go up, which is the question someone reading this is usually asking.
    text = str(refusal(admitted=True))
    assert "admitted, tightest limit generations" in text


def test_a_grant_with_nothing_close_says_only_that_it_was_admitted():
    text = str(AdmissionExplanation(admitted=True, scope="acme", priority=0))
    assert text.startswith("admitted, scope acme")


def test_every_dimension_is_reported_not_only_the_binding_one():
    # Seeing what was not full is what tells someone the limit they were about
    # to raise is not the one that is actually binding.
    text = str(refusal())
    assert "rpm" in text
    assert "generations" in text


def test_the_binding_dimension_is_marked_in_the_listing():
    lines = str(refusal()).splitlines()
    binding = [line for line in lines if "<- binding" in line]
    assert len(binding) == 1
    assert "generations" in binding[0]


def test_headroom_is_shown_as_a_percentage():
    assert "59% free" in str(refusal())
    assert "0% free" in str(refusal())


def test_whole_numbers_print_without_a_decimal_point():
    # These are counts of requests and tokens. A trailing .0 on every one of
    # them makes the output harder to scan for no gain.
    assert "412/1000" in str(refusal())


def test_a_fractional_count_keeps_one_decimal_place():
    text = str(refusal(dimensions={"token_seconds": Utilisation(used=11.25, limit=20.0)}))
    assert "11.2/20" in text or "11.3/20" in text


def test_the_wait_is_reported():
    assert "waited 340ms" in str(refusal(waited_ms=340.0))


def test_nothing_waits_yet_so_the_wait_is_zero_by_default():
    assert "waited 0ms" in str(refusal())


def test_a_queue_position_is_shown_when_there_is_one():
    assert "queued at 7" in str(refusal(queue_position=7))


def test_no_queue_position_is_shown_when_there_is_none():
    assert "queued at" not in str(refusal())


def test_the_dictionary_form_carries_the_whole_decision():
    found = refusal().to_dict()
    assert found["admitted"] is False
    assert found["scope"] == "tenant:acme"
    assert found["priority"] == 0
    assert found["binding_dimension"] == "generations"
    assert found["waited_ms"] == 0.0
    assert found["queue_position"] is None


def test_the_dictionary_form_flattens_utilisation_into_plain_numbers():
    # It has to survive a structured logger and a trip through a log pipeline,
    # which means nothing in it may be an object from this library.
    found = refusal().to_dict()
    assert found["dimensions"]["generations"] == {
        "used": 64.0,
        "limit": 64.0,
        "headroom": 0.0,
    }


def test_the_controller_state_is_empty_while_every_limit_is_the_configured_number():
    assert refusal().to_dict()["controller"] == {}


def test_the_controller_state_is_copied_rather_than_shared():
    state = {"generations": {"algorithm": "vegas"}}
    found = AdmissionExplanation(
        admitted=True, scope="acme", priority=0, controller=state
    ).to_dict()
    assert found["controller"] == state
    assert found["controller"]["generations"] is not state["generations"]


def test_an_explanation_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        refusal().admitted = True


def test_a_count_a_hair_under_its_limit_prints_as_the_limit():
    # A rate window replenishes continuously, so a key that was exactly full a
    # moment ago reads back as 999.99997. Printing that beside a limit of 1000
    # makes a correct limiter look like a broken one.
    text = str(refusal(dimensions={"rpm": Utilisation(used=999.99997, limit=1000.0)}))
    assert "1000/1000" in text


def test_a_genuinely_fractional_count_still_shows_its_fraction():
    text = str(refusal(dimensions={"token_seconds": Utilisation(used=11.25, limit=20.0)}))
    assert "11.2/20" in text
