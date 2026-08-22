"""A starting point for a provider this library does not ship.

Copy this file, rename it, fill in the eight members, and add fixtures under
the provider fixtures directory. The shared conformance suite then runs against
it automatically and will tell you what is missing.

Nothing here needs to be imported or subclassed by a user of your adapter. This
is a protocol, so an object with these members is an adapter.

**What an adapter never does.** It makes no network call, and it has no opinion
about how much anybody is allowed. A rate limit belongs to an account, not to a
provider, and the figure lives in that account's own console. Encode how your
provider counts and let the caller name their own limits.
"""

from __future__ import annotations

from collections.abc import Mapping

from spillway.core.cost import Cost
from spillway.estimators.base import RequestContext
from spillway.providers._read import count, field, usage_of
from spillway.providers.base import Outcome, ProviderState


class Template:
    """Rename this to your provider.

    Example:
        >>> adapter = Template()
        >>> adapter.charges_max_tokens()
        False
    """

    name: str = "template"
    """What your provider is called, in errors and in messages."""

    client_module: str = "yourprovider"
    """The top level module of a client speaking your provider's protocol.

    This is what recognises a client: the module of its class, nothing more.
    Your adapter is never imported by this library and never imports the client
    library either.
    """

    official_hosts: tuple[str, ...] = ("api.yourprovider.example",)
    """Hosts your accounting actually applies to. Bare hosts, no scheme.

    A client speaking your protocol against some other host is somebody else's
    service, and applying your rules to it would be wrong on the back of a
    correct protocol detection.
    """

    endpoints: tuple[str, ...] = ("chat.create",)
    """Dotted method paths to instrument.

    **Only methods that reach the network.** If one of your endpoints is a
    convenience wrapper that calls another, list the one it calls and not both,
    or a single request will be admitted twice and the caller's limit will be
    half what they asked for. The conformance suite checks this.
    """

    def request_from(self, endpoint: str, kwargs: Mapping[str, object]) -> RequestContext:
        """Read one intercepted call into a request context.

        Whatever the caller passed, minus the reserved keywords, which are
        removed before you see them. Pull out the model, the thing that will be
        counted as input, and the requested maximum output length.

        If your endpoints name these differently from one another, branch on
        `endpoint`. That is why it is here.
        """
        maximum = kwargs.get("max_tokens")
        model = kwargs.get("model")
        del endpoint
        return RequestContext(
            prompt=None,
            max_tokens=count(maximum) if maximum is not None else None,
            model=str(model) if model is not None else None,
        )

    def adjust(self, cost: Cost, context: RequestContext) -> Cost:
        """Apply your provider's admission time accounting to a reservation.

        Only interesting if your provider charges the requested maximum output
        length whether or not it is reached. If it meters what was actually
        generated, return the cost untouched, as here.

        Must be idempotent.
        """
        del context
        return cost

    def usage_from(self, response: object) -> Cost:
        """Read what a call really cost.

        Put any category your provider meters separately into `extra` rather
        than into the totals, and be careful about which way round yours works:
        a category reported *inside* a total must not be added again, and one
        reported *outside* it must not be folded in.

        Raises:
            ValueError: when there is no usage. Say what to do instead. The
                caller will settle at the reserved amount, which is safe and
                wasteful, and the message is how they find out why.
        """
        usage = usage_of(response)
        if usage is None:
            message = (
                "No usage found on this response, so what the call really cost is unknown. "
                "Pass the response object the client returned, or settle the lease by hand "
                "with lease.settle(input=..., output=...)."
            )
            raise ValueError(message)
        return Cost(
            input_tokens=count(field(usage, "input_tokens")),
            output_tokens=count(field(usage, "output_tokens")),
            requests=1,
        )

    def classify(self, exc: BaseException | None, response: object | None) -> Outcome:
        """Say what happened, in the terms a controller acts on.

        Read the exception duck typed rather than importing your client
        library. Both of the shipped adapters read `status_code` and `body` off
        whatever was raised.

        **The one thing not to get wrong.** A client error is not congestion,
        and neither is an exhausted budget. Both of the shipped providers have
        a failure that arrives looking exactly like a rate limit and never
        clears however long you wait. Check for yours.
        """
        del response
        if exc is None:
            return Outcome.OK
        status = field(exc, "status_code")
        if status == 429:
            return Outcome.RATE_LIMITED
        if isinstance(status, int) and status >= 500:
            return Outcome.OVERLOADED
        return Outcome.ERROR

    def parse_headers(self, headers: Mapping[str, str]) -> ProviderState | None:
        """Read your provider's own account of its limits, if it gives one.

        Return None when your provider reports nothing, as here. None and an
        empty state mean opposite things: a controller stands down on the
        first and would read the second as no budget left.

        Reset values differ wildly. One shipped provider sends absolute
        timestamps and the other sends durations. Convert yours to seconds from
        now, and never return a negative wait.
        """
        del headers
        return None

    def retry_after(self, exc: BaseException | None, headers: Mapping[str, str]) -> float | None:
        """How many seconds your provider asked to be left alone for.

        Return None rather than zero when it did not say. Zero reads as "retry
        immediately" into a limit that is still full.
        """
        del exc, headers
        return None

    def charges_max_tokens(self) -> bool:
        """Whether the requested maximum is charged at admission.

        The most consequential thing you will write here. False means the
        caller's predicted output is used, which is much cheaper. True means
        the requested maximum is reserved, which is expensive and correct when
        the provider does the same.

        If you are unsure, answer True. Being wrong that way wastes headroom
        that settlement gives straight back. Being wrong the other way produces
        rate limit responses that nothing predicted.
        """
        return False
