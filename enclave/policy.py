"""The per-bank output policy (Stage 3).

The enclave assembles a graph no participant could assemble, so the release
rule is the whole safety argument: **a bank receives alerts about its own
accounts only.** Bank A must learn that one of its accounts sits in a ring
spanning three banks. It must not learn which accounts at Bank B are in it.

That makes the policy a redaction problem with an easy failure mode: the
obvious implementation filters the *member list* and then hands over the ring's
edges, and the edges carry the other banks' identifiers. So this module builds
each bank's bundle by construction from that bank's own members outward, and
never by deleting fields from a full view. :func:`foreign_identifiers_in`
exists so the property can be asserted directly in tests rather than argued.

Counterparties at other banks are replaced by **per-bundle pseudonyms**
(``other-bank:bank-b:1``). Within one bank's bundle a pseudonym is stable, so
the bank can still see structure -- "my account paid three distinct accounts at
Bank B" -- but the pseudonym is derived from the bundle's own ordering and not
from the enclave id, so it is unlinkable across bundles and reveals nothing
about the account it stands for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from enclave.graph import CombinedGraph
from shared.contracts import Ring


@dataclass(frozen=True)
class RedactedEdge:
    """One edge as a single bank is allowed to see it.

    Exactly one endpoint is a real enclave id belonging to the receiving bank;
    the other is a pseudonym unless both ends belong to the receiver.
    """

    from_ref: str
    to_ref: str
    from_bank: str
    to_bank: str
    amount: str
    currency: str
    timestamp: str
    channel: str


@dataclass
class BankAlertBundle:
    """What one bank receives about one ring.

    Attributes:
        bank_id: The receiving bank.
        ring_id: Content-addressed ring id, shared across banks so two banks
            can refer to the same ring in a later lawful exchange without
            either learning the other's members.
        own_members: This bank's alertable accounts in the ring, real enclave
            ids. These are the only real identifiers in the bundle.
        bank_span: How many banks the ring touches. An aggregate: it is the
            finding, and it is not reducible to a member list.
        total_member_count: Size of the ring across all banks. Aggregate.
        other_bank_member_counts: bank id -> how many members that bank has.
            Discloses a count, never an identifier. Included because "your
            account is in a ring with 14 accounts at one other bank" is a
            materially different lead from "...with one", and the count is what
            carries that.
        bridged_by_unflagged: How many unflagged accounts were needed to
            connect the ring. A ring held together only by bridges is a weaker
            finding, and hiding that would overstate the evidence.
        edges: Edges touching this bank's accounts, with foreign endpoints
            pseudonymised.
    """

    bank_id: str
    ring_id: str
    own_members: tuple[str, ...]
    bank_span: int
    total_member_count: int
    other_bank_member_counts: dict[str, int] = field(default_factory=dict)
    bridged_by_unflagged: int = 0
    edges: tuple[RedactedEdge, ...] = ()

    def to_json(self) -> str:
        """Serialise the bundle. Also what the leak test inspects."""
        return json.dumps(
            {
                "bank_id": self.bank_id,
                "ring_id": self.ring_id,
                "own_members": list(self.own_members),
                "bank_span": self.bank_span,
                "total_member_count": self.total_member_count,
                "other_bank_member_counts": self.other_bank_member_counts,
                "bridged_by_unflagged": self.bridged_by_unflagged,
                "edges": [vars(e) for e in self.edges],
            },
            sort_keys=True,
            indent=2,
        )


class _Pseudonymiser:
    """Assigns stable per-bundle pseudonyms to accounts at other banks."""

    def __init__(self, owner_of: dict[str, str]) -> None:
        self._owner_of = owner_of
        self._assigned: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def ref_for(self, enclave_id: str, receiving_bank: str) -> str:
        """Return the receiver-visible reference for ``enclave_id``."""
        owner = self._owner_of.get(enclave_id)
        if owner == receiving_bank:
            return enclave_id
        if enclave_id not in self._assigned:
            label = owner or "unknown-bank"
            self._counts[label] = self._counts.get(label, 0) + 1
            self._assigned[enclave_id] = f"other-bank:{label}:{self._counts[label]}"
        return self._assigned[enclave_id]


def bundles_for_ring(
    ring: Ring,
    graph: CombinedGraph,
    bridged_by_unflagged: int = 0,
) -> dict[str, BankAlertBundle]:
    """Build one alert bundle per bank that has a member in ``ring``.

    A bank with no member in the ring receives nothing at all -- not an empty
    bundle. Telling a bank "a ring exists that you are not part of" is a
    disclosure with no benefit to it.

    Args:
        ring: The ring to release, with members as enclave ids.
        graph: The combined graph, for ownership and edges.
        bridged_by_unflagged: Count of unflagged bridging accounts, from the
            :class:`~enclave.rings.RingCandidate` this ring came from.

    Returns:
        bank id -> that bank's bundle.
    """
    member_ids = set(ring.all_member_ids)
    alertable = set(ring.alertable_member_ids)

    members_by_bank: dict[str, list[str]] = {}
    for member_id in sorted(member_ids):
        owner = graph.owner_of.get(member_id)
        if owner is None:
            # An account nobody claimed. It stays in the ring's aggregate size
            # but is released to no one: there is no bank entitled to it.
            continue
        members_by_bank.setdefault(owner, []).append(member_id)

    bundles: dict[str, BankAlertBundle] = {}
    for bank_id, own in sorted(members_by_bank.items()):
        own_alertable = tuple(m for m in own if m in alertable)
        if not own_alertable:
            # This bank's only involvement is a bridging account, which is not
            # an allegation. Nothing to alert it about.
            continue
        pseudonymiser = _Pseudonymiser(graph.owner_of)
        redacted: list[RedactedEdge] = []
        for edge in graph.edges:
            touches_own = edge.from_id in own or edge.to_id in own
            in_ring = edge.from_id in member_ids and edge.to_id in member_ids
            if not (touches_own and in_ring):
                continue
            redacted.append(
                RedactedEdge(
                    from_ref=pseudonymiser.ref_for(edge.from_id, bank_id),
                    to_ref=pseudonymiser.ref_for(edge.to_id, bank_id),
                    from_bank=edge.from_bank,
                    to_bank=edge.to_bank,
                    amount=str(edge.amount),
                    currency=edge.currency,
                    timestamp=edge.timestamp.isoformat(),
                    channel=edge.channel,
                )
            )
        bundles[bank_id] = BankAlertBundle(
            bank_id=bank_id,
            ring_id=ring.ring_id,
            own_members=own_alertable,
            bank_span=len(ring.banks_involved),
            total_member_count=len(member_ids),
            other_bank_member_counts={
                other: len(ids) for other, ids in sorted(members_by_bank.items())
                if other != bank_id
            },
            bridged_by_unflagged=bridged_by_unflagged,
            edges=tuple(redacted),
        )
    return bundles


def foreign_identifiers_in(bundle: BankAlertBundle, graph: CombinedGraph) -> set[str]:
    """Return any enclave id in ``bundle`` that belongs to another bank.

    The Stage 3 claim is exactly "this set is empty for every bundle", so it is
    computed here and asserted in the tests rather than left as prose. The
    check runs over the **serialised** bundle, not over the fields the builder
    happens to know about: a field added later that carries a foreign id would
    otherwise pass a structural check while still leaking.
    """
    serialised = bundle.to_json()
    return {
        enclave_id
        for enclave_id, owner in graph.owner_of.items()
        if owner != bundle.bank_id and enclave_id in serialised
    }
