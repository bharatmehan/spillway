"""The default store: a dictionary, a lock, and nothing else.

Zero configuration and zero dependencies, which is what makes the quickstart
work on a clean environment.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import os
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from spillway.core.clock import Clock, MonotonicClock
from spillway.core.engine import (
    gauge_release,
    gauge_reserve,
    gcra_credit,
    gcra_debt,
    gcra_reserve,
)
from spillway.core.errors import LeaseExpired
from spillway.stores.base import (
    Claim,
    ClaimKind,
    Delta,
    ReserveResult,
    Utilisation,
)

_log = logging.getLogger(__name__)

_warned_about_workers = False


# ponytail: a handful of strong signals, and it will miss deployment shapes it
# has never heard of. Nothing better is obviously available from inside a worker
# process, and a missed warning is no worse than the silence there would
# otherwise be. Add a signal when a real deployment goes undetected, not on
# speculation.
def multi_worker_hint() -> str | None:
    """Return what suggests this process is one of several, or None if nothing does.

    Example:
        >>> import os
        >>> os.environ["WEB_CONCURRENCY"] = "4"
        >>> multi_worker_hint()
        'WEB_CONCURRENCY is set to 4'
        >>> del os.environ["WEB_CONCURRENCY"]
    """
    software = os.environ.get("SERVER_SOFTWARE", "").lower()
    if "gunicorn" in software:
        return "SERVER_SOFTWARE names gunicorn"
    if "uwsgi" in software or "uwsgi" in sys.modules:
        return "this process is running under uWSGI"
    concurrency = os.environ.get("WEB_CONCURRENCY", "")
    if concurrency.isdigit() and int(concurrency) > 1:
        return f"WEB_CONCURRENCY is set to {concurrency}"
    return None


def _warn_once_about_workers() -> None:
    """Say something the first time an in memory store is used across workers.

    Once per process, and loudly. The overshoot from each worker enforcing the
    full limit appears at the provider, and nothing locally points at the cause.
    """
    global _warned_about_workers
    if _warned_about_workers:
        return
    hint = multi_worker_hint()
    if hint is None:
        return
    _warned_about_workers = True
    _log.warning(
        "MemoryStore is in use and this process looks like one of several (%s). Each "
        "process enforces the whole limit on its own, so total consumption will "
        "overshoot by roughly the number of workers and the provider will start "
        "refusing requests. Point every worker at one shared store, or run a single "
        "process.",
        hint,
    )


@dataclass(frozen=True)
class _Lease:
    """What one outstanding reservation is holding."""

    claims: tuple[Claim, ...]
    scope: str
    priority: int
    expires_at_ms: float


def _emission_interval_ms(claim: Claim) -> float:
    """Time one unit of cost buys on this claim's key."""
    window_ms = claim.window_ms
    if window_ms is None:  # pragma: no cover - Claim refuses this at construction
        message = f"Rate claim {claim.key!r} has no window."
        raise ValueError(message)
    return window_ms / claim.limit


def _window_ms(claim: Claim) -> float:
    """This claim's window, which a rate claim always has."""
    window_ms = claim.window_ms
    if window_ms is None:  # pragma: no cover - Claim refuses this at construction
        message = f"Rate claim {claim.key!r} has no window."
        raise ValueError(message)
    return window_ms


class MemoryStore:
    """Not safe across processes. Every process using one enforces the full limit alone.

    That first line is the important one. Under a server running four workers,
    four of these each admit up to the whole limit, so consumption overshoots
    fourfold while every worker believes it is behaving. Share a coordinated
    store across processes, or run one process.

    Within one process it is correct and fast: arithmetic over a handful of
    dictionary entries, guarded by one lock.

    A reservation whose process died is reclaimed once it outlives its expiry.
    Without that, gauges leak until nothing is admitted at all.

    Args:
        clock: Where time comes from. Defaults to the real monotonic clock.

    Example:
        >>> from spillway.core.clock import FakeClock
        >>> from spillway.stores.base import Claim, ClaimKind
        >>> clock = FakeClock()
        >>> store = MemoryStore(clock=clock)
        >>> slots = [Claim("acme:generations", ClaimKind.GAUGE, cost=1.0, limit=1.0)]
        >>> first = store.reserve_sync(slots, ttl_ms=60_000.0, scope="acme", priority=0)
        >>> first.granted
        True
        >>> store.reserve_sync(slots, ttl_ms=60_000.0, scope="acme", priority=0).binding_key
        'acme:generations'
        >>> store.release_sync(first.lease_id)
        >>> store.reserve_sync(slots, ttl_ms=60_000.0, scope="acme", priority=0).granted
        True
    """

    def __init__(self, clock: Clock | None = None) -> None:
        """Start empty, with nothing reserved."""
        self._clock: Clock = clock if clock is not None else MonotonicClock()
        # ponytail: one lock over the whole reserve and settle path. The section
        # is arithmetic over a few keys, so contention only appears well above
        # the admission rate any process making model calls will reach. Split it
        # per key if a measurement ever says otherwise, not before: per key
        # locking brings a lock ordering problem and a real deadlock risk with it.
        self._lock = threading.RLock()
        self._rate: dict[str, float] = {}
        self._gauge: dict[str, float] = {}
        # What each key is configured as, remembered separately from the leases
        # holding it, so that a key can still be reported on once every lease
        # against it has finished.
        # ponytail: nothing is ever evicted, so a process that sees unboundedly
        # many distinct scopes grows without limit. Bounded by scope count in
        # practice, and a scope is a tenant or a user rather than a request.
        # Evict a key once its rate has drained and its gauge is empty if a
        # deployment ever churns scopes fast enough to matter.
        self._config: dict[str, Claim] = {}
        self._leases: dict[str, _Lease] = {}
        # Expiries in order, reaped lazily. An entry for a lease that has since
        # been settled is left in place and skipped when it surfaces, because
        # finding and removing it would cost more than ignoring it.
        self._expiry: list[tuple[float, str]] = []
        self._next_id = itertools.count(1)

    def reserve_sync(
        self,
        claims: Sequence[Claim],
        *,
        ttl_ms: float,
        scope: str,
        priority: int,
    ) -> ReserveResult:
        """Apply every claim, or none of them. See `Store.reserve`."""
        _warn_once_about_workers()
        with self._lock:
            now_ms = self._clock.now_ms()
            self._reap(now_ms)
            rate_after: dict[str, float] = {}
            gauge_after: dict[str, float] = {}
            for claim in claims:
                self._config[claim.key] = claim

            for claim in claims:
                if claim.kind is ClaimKind.RATE:
                    tat_ms = rate_after.get(claim.key, self._rate.get(claim.key, now_ms))
                    granted, tat_ms, retry_after_ms = gcra_reserve(
                        tat_ms,
                        now_ms,
                        claim.cost,
                        _emission_interval_ms(claim),
                        _window_ms(claim),
                    )
                    if not granted:
                        return ReserveResult.refused(
                            claim.key,
                            retry_after_ms=retry_after_ms,
                            utilisation=self._utilisation(claims, now_ms),
                        )
                    rate_after[claim.key] = tat_ms
                else:
                    held = gauge_after.get(claim.key, self._gauge.get(claim.key, 0.0))
                    granted, held = gauge_reserve(held, claim.cost, claim.limit)
                    if not granted:
                        # A gauge frees when something settles, not when time
                        # passes, so there is no honest wait to report.
                        return ReserveResult.refused(
                            claim.key,
                            retry_after_ms=None,
                            utilisation=self._utilisation(claims, now_ms),
                        )
                    gauge_after[claim.key] = held

            self._rate.update(rate_after)
            self._gauge.update(gauge_after)
            lease_id = f"lease-{next(self._next_id)}"
            self._leases[lease_id] = _Lease(
                claims=tuple(claims),
                scope=scope,
                priority=priority,
                expires_at_ms=now_ms + ttl_ms,
            )
            heapq.heappush(self._expiry, (now_ms + ttl_ms, lease_id))
            return ReserveResult.granted_as(
                lease_id,
                utilisation=self._utilisation(claims, now_ms),
            )

    def settle_sync(self, lease_id: str, deltas: Sequence[Delta]) -> None:
        """Apply corrections and end the lease. See `Store.settle`.

        Raises:
            LeaseExpired: if the lease is not outstanding, which means it was
                already settled or its capacity was reclaimed.
        """
        with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                message = (
                    f"Lease {lease_id!r} is no longer outstanding, so it cannot be "
                    f"settled. It was either settled already or it ran past its expiry "
                    f"and its capacity was given back. If calls legitimately run this "
                    f"long, raise the expiry."
                )
                raise LeaseExpired(message)
            now_ms = self._clock.now_ms()
            by_key = {claim.key: claim for claim in lease.claims}
            for delta in deltas:
                claim = by_key.get(delta.key)
                if claim is None:
                    continue
                self._apply(claim, delta.amount, now_ms)

    def release_sync(self, lease_id: str) -> None:
        """Return the whole reservation and end the lease. See `Store.release`.

        Releasing a lease that is not outstanding does nothing rather than
        raising: release runs on the failure path, often from a finally block,
        where a second error would bury the first.
        """
        with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                return
            now_ms = self._clock.now_ms()
            for claim in lease.claims:
                self._apply(claim, claim.cost, now_ms)

    def snapshot_sync(self, keys: Sequence[str]) -> Mapping[str, Utilisation]:
        """Report how full each key is. See `Store.snapshot`.

        A key this store has never seen reports as empty rather than absent, so
        a dimension can be charted from when it is configured, not first used.
        """
        with self._lock:
            now_ms = self._clock.now_ms()
            self._reap(now_ms)
            found: dict[str, Utilisation] = {}
            for key in keys:
                claim = self._config.get(key)
                if claim is None:
                    found[key] = Utilisation(used=0.0, limit=0.0)
                else:
                    found[key] = self._utilisation_of(claim, now_ms)
            return found

    async def reserve(
        self,
        claims: Sequence[Claim],
        *,
        ttl_ms: float,
        scope: str,
        priority: int,
    ) -> ReserveResult:
        """Apply every claim, or none of them. See `Store.reserve`."""
        return self.reserve_sync(claims, ttl_ms=ttl_ms, scope=scope, priority=priority)

    async def settle(self, lease_id: str, deltas: Sequence[Delta]) -> None:
        """Apply corrections and end the lease. See `Store.settle`."""
        self.settle_sync(lease_id, deltas)

    async def release(self, lease_id: str) -> None:
        """Return the whole reservation and end the lease. See `Store.release`."""
        self.release_sync(lease_id)

    async def snapshot(self, keys: Sequence[str]) -> Mapping[str, Utilisation]:
        """Report how full each key is. See `Store.snapshot`."""
        return self.snapshot_sync(keys)

    def _reap(self, now_ms: float) -> None:
        """Reclaim anything held by a lease that outlived its expiry.

        A request whose process died can never settle, so without this every
        gauge leaks until nothing is admitted.

        Only gauges come back. A rate charge was really spent: the call went
        out, and only the report of how it ended is missing.

        Lazy on purpose. Reaping on the way into a reservation means no
        background task to leak, and a store nobody uses needs no reaping.
        """
        while self._expiry and self._expiry[0][0] <= now_ms:
            _expires_at_ms, lease_id = heapq.heappop(self._expiry)
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                continue
            for claim in lease.claims:
                if claim.kind is ClaimKind.GAUGE:
                    self._apply(claim, claim.cost, now_ms)

    def _apply(self, claim: Claim, amount: float, now_ms: float) -> None:
        """Give `amount` back on this claim's key, or take more if it is negative."""
        if claim.kind is ClaimKind.RATE:
            tat_ms = self._rate.get(claim.key, now_ms)
            interval_ms = _emission_interval_ms(claim)
            if amount >= 0:
                self._rate[claim.key] = gcra_credit(tat_ms, now_ms, amount, interval_ms)
            else:
                window_ms = _window_ms(claim)
                self._rate[claim.key] = gcra_debt(
                    tat_ms, now_ms, -amount, interval_ms, window_ms, window_ms
                )
        else:
            held = self._gauge.get(claim.key, 0.0)
            self._gauge[claim.key] = gauge_release(held, amount)

    def _utilisation(self, claims: Sequence[Claim], now_ms: float) -> dict[str, Utilisation]:
        """Report every key in `claims`, reading state rather than the pending set."""
        return {claim.key: self._utilisation_of(claim, now_ms) for claim in claims}

    def _utilisation_of(self, claim: Claim, now_ms: float) -> Utilisation:
        """How full one key is, in the units its own limit is expressed in."""
        if claim.kind is ClaimKind.RATE:
            tat_ms = self._rate.get(claim.key, now_ms)
            ahead_ms = tat_ms - now_ms
            if ahead_ms < 0.0:
                ahead_ms = 0.0
            used = ahead_ms / _emission_interval_ms(claim)
            return Utilisation(used=used, limit=claim.limit)
        return Utilisation(used=self._gauge.get(claim.key, 0.0), limit=claim.limit)
