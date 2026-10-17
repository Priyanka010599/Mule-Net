"""Clustering cross-bank rings inside the enclave.

This is **not** a reimplementation of ``ml/rule.py``. That rule is a velocity
rule over an account's full intra-bank history, and the enclave does not have
that history -- by design, since only boundary-crossing edges are shared. What
the enclave can do, and what no bank can do alone, is find the *connected
structure across bank boundaries* among accounts a scorer has flagged.

So the division of labour is:

* each bank scores its own accounts on its own full data, using ``ml/`` as it
  does today,
* the enclave takes the flagged accounts and finds the rings that span banks.

The clustering is connected components over cross-bank edges, restricted to
flagged accounts, with unflagged accounts able to act as connectors only when
an explicit hop budget allows it. Components are deterministic and independent
of input order: ties break on the sorted enclave id, so the same graph always
yields the same ring ids and an operator cannot influence the result by
reordering the inbox.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from hashlib import sha256

from enclave.graph import CombinedGraph
from shared.contracts import MembershipBasis, Ring, RingMember, RingStatus

#: Minimum number of distinct banks a component must touch to be a *cross-bank*
#: ring. A component inside one bank is that bank's own business and is not
#: released by the enclave: it would tell the bank nothing it could not compute
#: itself, while still costing a disclosure.
MIN_BANK_SPAN = 2


@dataclass(frozen=True)
class RingCandidate:
    """A connected component of flagged accounts spanning at least two banks.

    Attributes:
        members: Enclave account ids, sorted.
        banks: Banks the component touches, sorted.
        connector_ids: Unflagged accounts that bridge two flagged ones inside
            the hop budget. They are **not** emitted as ring members -- see
            :func:`to_ring` for why the frozen ``shared/`` contract has no
            honest place for them -- but they are kept here because the
            clustering used them, and a component whose only link is a bridge
            is a weaker finding than one whose members touch directly. The
            per-bank bundle reports their count, never their ids.
    """

    members: tuple[str, ...]
    banks: tuple[str, ...]
    connector_ids: frozenset[str]

    @property
    def flagged_members(self) -> tuple[str, ...]:
        """Component members that a bank actually flagged, sorted."""
        return tuple(m for m in self.members if m not in self.connector_ids)

    @property
    def bank_span(self) -> int:
        """How many banks this component touches."""
        return len(self.banks)


def _adjacency(graph: CombinedGraph) -> dict[str, set[str]]:
    """Undirected adjacency over cross-bank edges only.

    Direction is dropped deliberately: a ring is a structure, and money moving
    A->B or B->A links the two accounts either way.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if not edge.crosses_banks:
            continue
        adjacency[edge.from_id].add(edge.to_id)
        adjacency[edge.to_id].add(edge.from_id)
    return adjacency


def find_candidates(
    graph: CombinedGraph,
    flagged: set[str],
    max_connector_hops: int = 1,
) -> list[RingCandidate]:
    """Find cross-bank components among ``flagged`` accounts.

    Args:
        graph: The combined cross-bank graph.
        flagged: Enclave ids each owning bank flagged as suspicious.
        max_connector_hops: How many consecutive unflagged accounts may bridge
            two flagged ones. ``0`` means flagged accounts must be directly
            adjacent. The default of ``1`` catches the common mule pattern of a
            pass-through account between two collectors, and is capped low on
            purpose: a large budget merges unrelated components into one giant
            "ring", which is the classic way this kind of clustering produces
            confident nonsense.

    Returns:
        Candidates spanning at least :data:`MIN_BANK_SPAN` banks, ordered by
        their first member's id so the output is deterministic.
    """
    if max_connector_hops < 0:
        raise ValueError("max_connector_hops must not be negative")
    adjacency = _adjacency(graph)
    unvisited = {a for a in sorted(flagged) if a in adjacency}
    candidates: list[RingCandidate] = []

    while unvisited:
        seed = min(unvisited)
        component: set[str] = set()
        connectors: set[str] = set()
        # (account, hops of unflagged chain used to reach it)
        queue: deque[tuple[str, int]] = deque([(seed, 0)])
        visited_at: dict[str, int] = {}
        while queue:
            account, hops = queue.popleft()
            if visited_at.get(account, 999) <= hops:
                continue
            visited_at[account] = hops
            is_flagged = account in flagged
            if is_flagged:
                component.add(account)
                hops = 0  # reaching a flagged account resets the bridge budget
            else:
                connectors.add(account)
            for neighbour in sorted(adjacency.get(account, ())):
                next_hops = hops if neighbour in flagged else hops + 1
                if next_hops > max_connector_hops:
                    continue
                queue.append((neighbour, next_hops))

        unvisited -= component
        # A connector only earns its place if it actually sits between two
        # component members; one dangling off the edge bridges nothing.
        kept_connectors = {
            c for c in connectors if len(adjacency.get(c, set()) & component) >= 2
        }
        members = tuple(sorted(component | kept_connectors))
        # Span is measured over *flagged* members only. A component whose
        # cross-bank reach comes solely from an unflagged bridge is not a
        # cross-bank ring: the accounts anyone actually detected all sit at one
        # bank, and that bank can see them without an enclave.
        banks = tuple(sorted({graph.owner_of[m] for m in component if m in graph.owner_of}))
        if len(banks) < MIN_BANK_SPAN:
            continue
        candidates.append(
            RingCandidate(
                members=members,
                banks=banks,
                connector_ids=frozenset(kept_connectors),
            )
        )

    candidates.sort(key=lambda c: c.members[0] if c.members else "")
    return candidates


def ring_id_for(candidate: RingCandidate) -> str:
    """Derive a stable ring id from the component's membership.

    Content-addressed rather than random so that two runs over the same graph
    produce the same ring id, which is what lets a bank correlate an alert it
    received yesterday with the same ring today. It is a hash of the members,
    so it reveals nothing about them to a bank that does not already hold them.

    Derived from the *flagged* members -- the accounts :func:`to_ring` actually
    emits -- so the id identifies the ring that was published. Hashing the
    bridges in too would change the id when a bridge appeared or vanished,
    breaking exactly the day-to-day correlation this exists for.
    """
    digest = sha256("|".join(candidate.flagged_members).encode("utf-8")).hexdigest()
    return f"ring-enclave-{digest[:16]}"


def to_ring(candidate: RingCandidate) -> Ring:
    """Turn a candidate into the project's :class:`shared.contracts.Ring`.

    Only flagged accounts become members, each with basis ``VELOCITY``: they
    were detected by their own bank's stage-1 rule on that bank's full data,
    which is what ``VELOCITY`` means in ``shared/contracts.py``.

    **Bridging accounts are deliberately omitted, and this is a contract gap,
    not an oversight.** :class:`~shared.contracts.MembershipBasis` offers only
    ``velocity`` and ``identity_link``, and ``identity_link`` is validated to
    require ``link_scope``, ``linked_from_hashed_account_id`` and
    ``link_evidence_type`` -- the vocabulary of *document and device reuse*. A
    structural bridge on the transfer graph is neither: filing one as an
    identity link would attach a ``link_evidence_type`` that never existed,
    putting a fabricated evidence claim on a row about an account nobody
    flagged. Rather than misuse the contract, the bridge is dropped from the
    ring and reported only as the count
    :attr:`~enclave.policy.BankAlertBundle.bridged_by_unflagged`.

    Representing bridges properly needs a new ``MembershipBasis`` member in
    ``shared/contracts.py`` (say ``cross_bank_structure``, alertable ``False``,
    carrying no link fields). ``shared/`` is frozen for this change and owned by
    the architect, so that request is recorded in ``docs/data_sharing.md``
    rather than made here.
    """
    members = [
        RingMember(
            hashed_account_id=account_id,
            membership_basis=MembershipBasis.VELOCITY,
            alertable=True,
        )
        for account_id in candidate.flagged_members
    ]
    return Ring(
        ring_id=ring_id_for(candidate),
        members=members,
        banks_involved=list(candidate.banks),
        status=RingStatus.CANDIDATE,
    )
