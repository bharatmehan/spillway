"""The warning that fires when an in memory store is used across worker processes.

Silently enforcing the whole limit in each of several workers is the single most
likely way someone concludes this library does not work: the overshoot shows up
at the provider, and nothing locally points at the cause.
"""

import logging
import sys

import pytest

from spillway.stores import memory
from spillway.stores.base import Claim, ClaimKind
from spillway.stores.memory import MemoryStore, multi_worker_hint


@pytest.fixture(autouse=True)
def unwarned(monkeypatch):
    monkeypatch.setattr(memory, "_warned_about_workers", False)
    for name in ("SERVER_SOFTWARE", "WEB_CONCURRENCY"):
        monkeypatch.delenv(name, raising=False)


def slot():
    return Claim("acme:generations", ClaimKind.GAUGE, cost=1.0, limit=4.0)


def admit(store):
    return store.reserve_sync([slot()], ttl_ms=60_000.0, scope="acme", priority=0)


def test_an_ordinary_process_looks_like_nothing_in_particular():
    assert multi_worker_hint() is None


def test_gunicorn_is_detected_from_the_server_software_variable(monkeypatch):
    monkeypatch.setenv("SERVER_SOFTWARE", "gunicorn/21.2.0")
    assert multi_worker_hint() == "SERVER_SOFTWARE names gunicorn"


def test_uwsgi_is_detected_from_the_server_software_variable(monkeypatch):
    monkeypatch.setenv("SERVER_SOFTWARE", "uWSGI/2.0.23")
    assert multi_worker_hint() == "this process is running under uWSGI"


def test_uwsgi_is_detected_from_the_module_it_injects(monkeypatch):
    monkeypatch.setitem(sys.modules, "uwsgi", object())
    assert multi_worker_hint() == "this process is running under uWSGI"


def test_a_worker_count_above_one_is_detected(monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    assert multi_worker_hint() == "WEB_CONCURRENCY is set to 4"


def test_a_single_worker_is_not_a_multi_worker_deployment(monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "1")
    assert multi_worker_hint() is None


def test_a_nonsense_worker_count_is_ignored_rather_than_crashing(monkeypatch):
    # A warning path that raises would take down admission itself, which is a
    # far worse outcome than a missed warning.
    monkeypatch.setenv("WEB_CONCURRENCY", "auto")
    assert multi_worker_hint() is None


def test_the_warning_fires_on_admission_under_a_multi_worker_server(monkeypatch, caplog):
    monkeypatch.setenv("SERVER_SOFTWARE", "gunicorn/21.2.0")
    with caplog.at_level(logging.WARNING):
        admit(MemoryStore())
    assert "one of several" in caplog.text
    assert "one shared store" in caplog.text


def test_the_warning_names_what_gave_the_deployment_away(monkeypatch, caplog):
    monkeypatch.setenv("WEB_CONCURRENCY", "8")
    with caplog.at_level(logging.WARNING):
        admit(MemoryStore())
    assert "WEB_CONCURRENCY is set to 8" in caplog.text


def test_the_warning_fires_once_however_many_requests_pass(monkeypatch, caplog):
    # Once, because a warning on every admission would be noise a user filters
    # out, which is the same as not warning at all.
    monkeypatch.setenv("SERVER_SOFTWARE", "gunicorn/21.2.0")
    store = MemoryStore()
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            admit(store)
    assert caplog.text.count("one of several") == 1


def test_the_warning_fires_once_per_process_not_once_per_store(monkeypatch, caplog):
    monkeypatch.setenv("SERVER_SOFTWARE", "gunicorn/21.2.0")
    with caplog.at_level(logging.WARNING):
        admit(MemoryStore())
        admit(MemoryStore())
    assert caplog.text.count("one of several") == 1


def test_nothing_is_said_in_an_ordinary_process(caplog):
    with caplog.at_level(logging.WARNING):
        admit(MemoryStore())
    assert caplog.text == ""
