# Changelog

Notable changes to this project, newest first. This project uses semantic versioning. While the
version is below 0.1, any release may change anything.

## Unreleased

### Added

- An empirical distribution over the output lengths a route has actually produced. This is what
  makes reserving less than the worst case possible: with a history to read, the ninth decile of
  what really happened is a far smaller number than the maximum a caller was willing to allow.
- An estimator protocol, with the request context it reads and the observation it learns from.
  Two methods: predict what a request will cost, and be told afterwards what it really cost. The
  second does nothing on an estimator that does not learn, so swapping in one that does is a
  constructor argument rather than a rewrite. Implement it structurally, importing nothing.
- `MaxTokensEstimator`, which reserves the output maximum a caller allowed and learns nothing.
  The safe, uninformed baseline, and the limiter's default. It is also the permanently correct
  answer against a provider that charges the requested maximum at admission, because reserving
  less than the provider does buys nothing and guarantees a rate limit response nobody predicted.
- `count_input`, the character heuristic on its own. Every estimator counts input the same way,
  and two of them counting it differently would be a difference no user could see or explain.
- `StaticEstimator`, which predicts the same output length every time. Genuinely useful where
  output length is genuinely predictable, such as a classifier answering with one of five labels
  or an extractor filling a fixed schema. Input is still counted per request, because inventing
  uncertainty about a number that can be counted would help nobody.
- `CallableEstimator`, which wraps any function from a request context to a distribution. This
  library does not ship an output length predictor and will not, because that is a research
  artefact with a heavy dependency tree and the quickstart has to run with nothing installed. It
  ships the socket instead, and this is the socket.
- An `estimator` argument on the limiter. It defaults to reserving the maximum a caller allowed,
  which is exactly what happened before, so nothing changes for anyone until they pass one.
  Prediction happens once per request rather than once per attempt: a request that waits reserves
  what it asked for when it arrived, because a prediction that moved while it queued would mean
  its place in the queue was earned against a different request.
- A `tags` mapping on `admit()`, for whatever the estimator should route on. Nothing here
  affects admission directly. It exists because output length is close to unpredictable across
  every call to a model and quite predictable within one task, so `tags={"task": "summarise"}` is
  usually worth more than any amount of cleverness elsewhere. Copied on the way in, so a caller
  reusing one dictionary cannot rewrite the past.
- Settled costs are reported back to the estimator. Every settlement carries what was reserved,
  what was really used, and what was known about the request beforehand, which is the whole of
  what an estimator needs to correct itself. Reported before the store is asked, so a request that
  outran its reservation still teaches: the bookkeeping failed, the request still generated what
  it generated. An abandoned request teaches nothing, because it produced nothing.
- `QuantileEstimator`, which predicts output length from what a route has actually produced. Keep
  the recent output lengths per route, reserve the point most of them came in under, settle the
  truth and hand the difference back at once. It makes no claim to accuracy and none should ever
  be made for it: the claim is that being wrong costs a little wasted headroom for the length of
  one request rather than an overrun that breaks a limit. The leverage is the route key rather
  than anything clever inside, and the default key, the model alone, is weak on purpose.
- The history per route is a bounded ring of recent output lengths, a thousand by default. Bounded
  because memory has to be, and recent because these distributions drift: a history that never
  forgot would answer today's question with last quarter's traffic.
- A sample threshold on the quantile estimator, thirty observations by default, below which
  another estimator answers instead. A measurement that does not exist yet must not bind: reading
  a ninth decile off four samples would hold back traffic on the strength of almost nothing, and
  being confidently wrong is worse there than being safe and expensive. The fallback is any
  estimator, so "until you know better, reserve five hundred" is one argument.
- Per route statistics on the quantile estimator: how many observations it has, what it is
  reserving at, the share of settlements that overran, and reserved divided by actual averaged
  over the ones that generated anything. The two ratios answer different questions and neither
  replaces the other. The overrun ratio says whether the quantile is still telling the truth. The
  error ratio says whether it is worth what it costs, and an error ratio near five is a caller
  wasting most of their headroom even though nothing is technically wrong. The error ratio is
  reported as absent rather than infinite when a request generated no output at all.
- Serialising and merging a quantile estimator's histories. Nothing calls them yet. They exist so
  that sharing what has been learned between workers, and keeping it across a restart, is a wiring
  change later rather than a redesign. A merge concatenates and then thins evenly, so two workers
  that have each seen half the traffic end up agreeing about all of it rather than one replacing
  the other.
- An adaptive quantile, off by default. When a route's overruns stop matching the share its
  quantile promised, the quantile moves, by a small step, at most once per hundred observations,
  and only within bounds. A quantile that moved per request would be an oscillation, and an
  oscillating estimator is worse than a fixed conservative one.
- Only the overrun rate moves it, in either direction. The estimate error ratio is reported and
  deliberately does not steer: on a heavy tailed route it is large however well calibrated the
  quantile is, and lowering the quantile because a route is wide would walk it down through a
  range where nothing changes, then off the far side of a mode, and overrun most of the traffic
  at once.
- The four estimators and the request context join the top level export list, so
  `from spillway import QuantileEstimator` works. The estimator protocol and the observation do
  not: a protocol needs no import to implement, and exporting one only enlarges what is promised
  for ever.
- A property covering the reservation quantile itself. Over generated histories and generated
  quantiles, the reserved amount covers at least its share of the observations, never falls below
  the point it was asked for, and never leaves the range of what was actually seen. This is the
  promise everything else rests on, and it is a promise about an interpolation and a rounding,
  both easy to get wrong in the direction that quietly under-reserves.
- The quantile to reserve at now travels on the estimate rather than being fixed by the limiter.
  An estimator that calibrates itself has to be able to move that number, and a number the limiter
  ignored would be no calibration at all. It still defaults to the ninth decile, so nothing
  changes for anyone who does not set it.

## 0.0.3 (2026-08-20)

### Added

- An asynchronous sleep on the `Clock` protocol. Waiting is the one thing a limiter does that
  cannot be pure arithmetic, and routing it through the clock is what keeps it testable without
  the suite sleeping for real.
- Sleeping on the fake clock. A sleeper is released when the clock is advanced past its wake
  time, so a test can run ten minutes of waiting in a millisecond and get the same sequence of
  events every run.
- `admit()` refuses a timeout and a deadline given together. They say the same thing two ways and
  there is no honest answer when they disagree.
- A guard against a request that is larger than a limit it draws on. No amount of waiting makes
  it fit, so it is refused at once with the two numbers and the three ways to fix it, rather than
  waiting for a capacity that can never arrive.
- A waiting queue, ordered strictly by priority band and first in first out within a band. It
  reports the depth of each band, hands back whoever is next, and gives up on a waiter whose
  deadline has passed wherever in the queue that waiter is sitting.
- A bound on each priority band, ten thousand waiters by default. Per band rather than shared, so
  a flood of batch work cannot consume the slots an interactive request needs, and bounded at all
  because an unbounded queue turns a rate limit problem into an out of memory problem.
- Shedding, as one sentence: a negative priority arrival is dropped rather than queued when its
  own band is at capacity. It raises `Shed` rather than a plain refusal, so a caller can retry
  later instead of treating a busy system as an error.
- A `shed_lowest` queue policy, alongside the default `reject`. A full band displaces the newest
  waiter in the lowest band below it rather than refusing, and refuses when the arrival is itself
  the lowest priority waiting. No waiter is ever dropped without another taking its place, so the
  total number waiting never grows.
- A callback on the lease, called once whichever way it finishes. Capacity coming back is the
  event somebody else is waiting for, and finding out by asking again on a timer would make every
  queued request pay for the delay.
- The dispatch loop: one task per limiter, which picks the best waiter, asks for its capacity, and
  hands over a lease or waits. It waits on both of the ways capacity appears at once, a release
  and a replenished rate window, and on the earliest deadline, so nothing polls and nothing is
  missed. It starts with the first waiter and stops with the last, so a limiter that never blocks
  has no background task at all.
- `admit()` waits for capacity when given a timeout or a deadline, instead of refusing straight
  away. The reservation is attempted directly first and the queue is only reached on a refusal, so
  the case where there is room pays nothing for the machinery. A timeout of zero, or a deadline
  already passed, reports what actually happened rather than a wait that never took place.
- A limiter level `default_timeout`, thirty seconds, applied when a caller names neither a timeout
  nor a deadline. Waiting for ever is almost never what anyone meant, and it is the failure that
  looks like the library hanging rather than like a limit being reached. Set it to zero to refuse
  rather than wait, or to None to wait for as long as it takes.
- A cancelled request takes itself out of the queue on the way past, so a waiter nobody is
  listening for cannot hold its band's head against everything behind it.
- The dispatch loop survives an unexpected failure. It reports the first one with its traceback
  and carries on, so queued requests reach their own timeouts and are told what happened, rather
  than waiting on a dispatcher that is no longer there.
- `queue_capacity` and `queue_full_policy` on the limiter, so the bound on each band and what a
  full band does with a new arrival are both things a caller sets rather than things they discover.
- Real wait times and queue positions on the lease and its explanation. The question "why was this
  request three seconds slow" is now answered by a value already in hand, naming the dimension that
  bound, how full it was, and how many requests were ahead in the queue.
- A property covering the wakeups themselves. Over generated sequences of arrivals, settlements
  and clock advances, every waiter that could be admitted is admitted once nothing more arrives.
  This is the class of bug that otherwise appears in production as an intermittent hang nobody can
  reproduce.
- A property covering the queue's own bookkeeping: every request is queued or finished, never both
  and never neither, and the total depth is always the sum of the bands.

### Changed

- A request that finds no room now waits up to thirty seconds for it by default, where before it
  was refused immediately. Pass `default_timeout=0` for the previous behaviour.

## 0.0.2 (2026-08-18)

### Added

- A `Clock` protocol with a monotonic implementation and a fake one that advances by hand. Every
  time reference in the library goes through it, which is what makes rate windows, lease expiry and
  feedback control testable without sleeping.
- A `Cost` value type: input tokens, output tokens, requests, and provider specific extra
  categories. Subtraction keeps the sign, because settlement is a difference and a negative
  component is an overrun to repay rather than a number to discard.
- `Distribution` and `Estimate`. Output length is predicted rather than known, so it is carried as
  a distribution with a `quantile` method. Two constructors for now: an exact point, and a worst
  case bound.
- A default estimate function. Input tokens are counted with a documented character heuristic that
  is accurate to roughly ten to fifteen percent, so the quickstart needs no tokenizer installed, and
  the real figure replaces it at settlement.
- `Scope` and `Priority`. A scope is the key every limit is tracked against; a priority is an
  ordinary integer, with four named conventions, where negative means the work is sheddable.
- The complete exception hierarchy under a single `SpillwayError` base. A refusal carries which
  dimension bound and how long until it would not have, so a caller can act rather than merely
  fail. The missing dependency error names the exact install command.
- Rate reservation arithmetic, using the generic cell rate algorithm. The whole state for a rate
  key is one float, so memory per key is constant no matter how much traffic passes, and a refusal
  reports how long until the same charge would fit.
- Credit and debt arithmetic for rate keys. Unused capacity is returned within a request's own
  lifetime, which is what makes reserving conservatively affordable, and an overrun becomes debt
  repaid from the next window, bounded so one bad estimate cannot silence a scope for hours.
- Gauge arithmetic, for limits on a value currently held rather than consumed over a window.
  Concurrency is one. Releasing is clamped at zero, because a gauge below zero would admit more
  than its limit.
- The types a store speaks in: `Claim`, `Delta`, `Utilisation` and `ReserveResult`. A refusal names
  the key that bound and how long until it would not have, because a bare yes or no cannot be
  explained to a user and forces a caller to poll.
- A `Dimension` protocol. A dimension turns a cost into a claim and a settlement into a correction,
  and does nothing else: it never decides whether a claim fits, because that decision has to be
  made for every dimension at once or a request gets admitted against two limits and refused by the
  third with the first two already spent.
- A `Rate` dimension, for limits consumed over a rolling window. It declares which part of a
  request's cost it counts, so a tokens per minute limit and a requests per minute limit can sit
  side by side without either counting the other's units. Asking for an adaptive rate limit is
  refused with an explanation rather than accepted.
- The meter is inferred from the dimension name for `rpm`, `rpd`, `input_tpm`, `output_tpm` and
  `tpm`, so the common case is one line. Any other name must say what it counts, because guessing
  would meter the wrong thing silently.
- A `Concurrency` dimension, limiting how many requests are in flight at once. One request takes
  one slot whatever it costs, and gets it back whole at settlement however wrong the estimate was.
- `Store` and `SyncStore` protocols. A store is asked for a whole batch of claims at once and
  applies all of them or none, which is the only way a request cannot be admitted against two
  limits and refused by the third with the first two already spent. One class may implement both.
- `MemoryStore`, the default store. Zero configuration and zero dependencies, so the quickstart
  runs on a clean environment. It is not safe across processes, and its docstring says so first:
  four workers each running one enforce the full limit four times over.
- Leases that are never settled are reclaimed once they outlive their expiry, so a process that
  dies mid request cannot leak a gauge. Only gauges come back: a rate charge was really spent, and
  inventing a refund would let a crashing worker exceed a provider's limit indefinitely.
- A warning, once per process, when an in memory store is used and the process looks like one of
  several workers. The overshoot this causes appears at the provider, and nothing locally points at
  the cause, so the warning is the only thing that connects the two.
- `AdmissionExplanation`, carried by every decision either way. It reports how full every limit was,
  not just the one that ran out, because seeing what was not full is what tells someone the limit
  they were about to raise is not the one actually binding. It prints readably and converts to
  plain data.
- `Lease` and `LeaseState`. Settling reports the real cost and returns the difference immediately,
  so reserving conservatively costs nothing in steady state. Settling twice raises rather than
  counting the same request twice; abandoning twice does nothing, because it runs on the failure
  path where a second error buries the first.
- `Spillway`, the limiter itself, with a non blocking `admit`. Every argument has a default and
  `Spillway()` with none is valid: it tracks and reports and refuses nothing, which is a reasonable
  first step for someone gathering evidence before choosing limits. A refusal names the dimension
  the caller configured rather than an internal store key, reports the wait in seconds, and carries
  how full every limit was. A plain `with` statement refuses and names the asynchronous form rather
  than starting an event loop on the caller's behalf.
- `admit()` works as an asynchronous context manager, and handles all four ways a block can end.
  A raised exception or a cancelled task returns the whole reservation, because nothing was
  consumed. Leaving without settling charges the full reserved amount and says so once. A request
  that outran its expiry keeps its result: the bookkeeping failed, not the caller's work.
- `Spillway.snapshot()`, reporting how full every limit is for one scope without reserving
  anything, so it is safe to call from a health check. Limits come from the dimensions rather than
  from the store, so a dimension reports its real limit before its first request rather than
  appearing to have none.
- Property based testing, with a fixed seed in continuous integration and a random one locally. A
  fixed seed makes a red build reproducible on the machine of whoever has to fix it; a random one
  keeps the suite exploring rather than re examining the same cases for ever.
- Property tests for the six invariants the design rests on: reserve then release leaves no trace,
  settlement lands on the actual cost, a denied reservation consumes nothing, concurrent callers
  never exceed a limit in aggregate, no sliding window ever exceeds the rate, and outstanding
  leases sum to the gauge that is held.
- A curated top level export list: `Spillway`, `Scope`, `Priority`, `Rate`, `Concurrency`, `Cost`,
  `Estimate`, `Distribution`, `Lease`, `LeaseState` and the exception hierarchy. Everything else
  needs an explicit submodule import, so what an editor offers at the top level is what is
  supported.

### Fixed

- An explanation no longer prints a count as `1000.0/1000`. A rate window replenishes continuously,
  so a key that was exactly full a moment ago reads back a hair under, and showing that beside its
  limit made a correct limiter look like a broken one.

### Changed

- Async tests need no decorator. The library's entry point is asynchronous, so an async test is the
  ordinary case here rather than the exception.
- The test run now executes every example in a docstring. A public docstring example is the first
  thing a reader copies, so an example that has stopped working is a defect rather than a typo.

## 0.0.1 (2026-08-17)

### Added

- Project skeleton: packaging, linting, strict type checking, test runner, and continuous
  integration across Python 3.10 through 3.14.
- The package imports and reports its version. There is no public API yet.
