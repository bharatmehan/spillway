"""The dimension protocol and the key every dimension builds."""

from spillway.core.cost import Cost
from spillway.core.scope import Scope
from spillway.dimensions.base import Dimension, claim_key
from spillway.stores.base import Claim, ClaimKind, Delta


class Everything:
    """A dimension that claims one unit from every request."""

    name = "everything"
    kind = ClaimKind.GAUGE

    def claim(self, cost: Cost, scope: Scope) -> Claim | None:
        return Claim(claim_key(scope, self.name), self.kind, cost=1.0, limit=4.0)

    def settle(self, reserved: Cost, actual: Cost, scope: Scope) -> Delta | None:
        return Delta(claim_key(scope, self.name), self.kind, amount=1.0)


def test_the_key_joins_the_scope_and_the_dimension_name():
    assert claim_key(Scope("tenant:acme"), "input_tpm") == "tenant:acme:input_tpm"


def test_different_scopes_get_different_keys():
    # Two tenants sharing a key would share a budget while appearing isolated.
    assert claim_key(Scope("a"), "rpm") != claim_key(Scope("b"), "rpm")


def test_different_dimensions_in_one_scope_get_different_keys():
    assert claim_key(Scope("a"), "rpm") != claim_key(Scope("a"), "input_tpm")


def test_a_plain_class_satisfies_the_protocol_without_inheriting_anything():
    # Extension points are protocols so that a user implements one without
    # importing a base class from this library.
    dimension: Dimension = Everything()
    assert dimension.name == "everything"
    assert dimension.kind is ClaimKind.GAUGE
    assert dimension.claim(Cost(), Scope("acme")).key == "acme:everything"
    assert dimension.settle(Cost(), Cost(), Scope("acme")).amount == 1.0
