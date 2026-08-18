"""Why a request was admitted, or was not.

A limiter that cannot say why it refused is indistinguishable from a broken one.
Every admission decision carries one of these, whichever way it went, so the
answer to "why is this slow" is a value already in hand rather than an
investigation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from spillway.stores.base import Utilisation


def _number(value: float) -> str:
    """Format a count without the noise of a trailing decimal on a whole number."""
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}"


@dataclass(frozen=True)
class AdmissionExplanation:
    """The complete reason for one admission decision.

    Attributes:
        admitted: Whether the request was let through.
        scope: Whose budget the decision was made against.
        priority: How urgent the request said it was.
        waited_ms: How long the caller waited before this decision.
        binding_dimension: Which limit ran out. Set on a refusal, and on a
            grant it names the limit that came closest, which is the one to
            raise if throughput needs to go up.
        dimensions: How full every limit was, by dimension name.
        controller: What each adaptive limit was doing. Empty while every limit
            is the configured number.
        queue_position: Where the request sat while waiting. Absent while
            nothing waits.

    Example:
        >>> from spillway.stores.base import Utilisation
        >>> explanation = AdmissionExplanation(
        ...     admitted=False,
        ...     scope="tenant:acme",
        ...     priority=0,
        ...     binding_dimension="generations",
        ...     dimensions={
        ...         "rpm": Utilisation(used=412.0, limit=1000.0),
        ...         "generations": Utilisation(used=64.0, limit=64.0),
        ...     },
        ... )
        >>> print(explanation)
        refused on generations, scope tenant:acme, priority 0, waited 0ms
          rpm          412/1000   59% free
          generations  64/64       0% free  <- binding
        >>> explanation.to_dict()["binding_dimension"]
        'generations'
    """

    admitted: bool
    scope: str
    priority: int
    waited_ms: float = 0.0
    binding_dimension: str | None = None
    dimensions: Mapping[str, Utilisation] = field(default_factory=dict)
    controller: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    queue_position: int | None = None

    def __str__(self) -> str:
        """Render the decision as something worth pasting into a bug report."""
        if self.admitted:
            headline = "admitted"
            if self.binding_dimension is not None:
                headline = f"admitted, tightest limit {self.binding_dimension}"
        else:
            headline = "refused"
            if self.binding_dimension is not None:
                headline = f"refused on {self.binding_dimension}"
        parts = [
            f"{headline}, scope {self.scope}, priority {self.priority}, "
            f"waited {_number(self.waited_ms)}ms"
        ]
        if self.queue_position is not None:
            parts[0] += f", queued at {self.queue_position}"
        if self.dimensions:
            name_width = max(len(name) for name in self.dimensions)
            usage_width = max(
                len(f"{_number(u.used)}/{_number(u.limit)}") for u in self.dimensions.values()
            )
            for name, used in self.dimensions.items():
                usage = f"{_number(used.used)}/{_number(used.limit)}"
                line = (
                    f"  {name:<{name_width}}  {usage:<{usage_width}}  "
                    f"{used.headroom * 100:3.0f}% free"
                )
                if name == self.binding_dimension:
                    line += "  <- binding"
                parts.append(line)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, object]:
        """Return the decision as plain data, ready to serialise or log.

        Nothing in the result is an object from this library, so it survives a
        trip through a log pipeline or a structured logger unchanged.

        Example:
            >>> AdmissionExplanation(admitted=True, scope="acme", priority=0).to_dict()["admitted"]
            True
        """
        return {
            "admitted": self.admitted,
            "scope": self.scope,
            "priority": self.priority,
            "waited_ms": self.waited_ms,
            "binding_dimension": self.binding_dimension,
            "dimensions": {
                name: {"used": used.used, "limit": used.limit, "headroom": used.headroom}
                for name, used in self.dimensions.items()
            },
            "controller": {name: dict(state) for name, state in self.controller.items()},
            "queue_position": self.queue_position,
        }
