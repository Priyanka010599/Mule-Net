"""Implementations of the seven tool stubs in ``shared/tool_interfaces.py``.

Signatures, argument defaults and raised exception types match that
module exactly; it is the contract and this file is the implementation.
Every identifier crossing these boundaries is a hashed account id.

All reads go through :class:`agents.data.DataStore`, so the same tool
code serves real ``ml/out`` tables and the mock fallback without change.

Two deliberate design points, both of which the report agent depends on:

* :func:`get_account_profile` and :func:`get_explanation` surface
  velocity and structure factors as the *drivers* of a score.
* :func:`get_document_reuse` returns identity matches as *evidence*. Its
  results carry no score and imply no guilt: roughly 1,442 legitimate
  accounts in this dataset share a document within a bank against about
  292 mules, and 400 legitimate dual-bank customers reuse an identifier
  across two banks. Reuse corroborates; it never decides.
"""

from __future__ import annotations

import functools
import json
import threading
from collections import namedtuple
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, TypeVar

from pydantic import BaseModel

from agents import cases, hidden_labels, regulations, tool_cache
from agents.data import DataStore, default_store, to_transfer
from agents.sql_guard import execute_select, validate_sql
from shared import metric_catalog, ring_race
from shared.contracts import (
    Account,
    AccountEvidence,
    AccountExplanation,
    Device,
    EvidenceScope,
    Investigation,
    KycDocument,
    KycSubmission,
    MembershipBasis,
    RingExplanation,
    RingMember,
    RiskScore,
    Transfer,
)
from shared.tool_interfaces import (
    ALLOWED_SQL_TABLES,
    SQL_ROW_LIMIT,
    assert_score_fields_rounded,
    round_score_fields,
)

# Re-exported so ``tools.SCORE_DECIMALS`` and ``tools.RingSubgraph`` keep
# resolving for callers and tests that already reference them. The
# redundant ``X as X`` form is the explicit re-export convention, and it
# is what stops a linter deleting a name this module does not itself use.
from shared.tool_interfaces import MAX_SUBGRAPH_ACCOUNTS as MAX_SUBGRAPH_ACCOUNTS
from shared.tool_interfaces import MAX_SUBGRAPH_TRANSFERS as MAX_SUBGRAPH_TRANSFERS
from shared.tool_interfaces import SCORE_DECIMALS as SCORE_DECIMALS
from shared.tool_interfaces import RingSubgraph as RingSubgraph

#: Any contract model carrying registered score fields.
_ScoreModelT = TypeVar("_ScoreModelT", bound=BaseModel)

#: Attributes :func:`get_document_reuse` checks for reuse, in report order.
REUSE_ATTRIBUTES: tuple[str, ...] = (
    "hashed_id_number",
    "normalized_address",
    "photo_embedding_ref",
)

#: Ceiling on paths returned by :func:`trace_money_flow`. A hub account
#: can branch combinatorially; without a cap a three-hop trace can return
#: millions of paths and starve the agent's context window.
MAX_TRACE_PATHS = 50

#: How many paths the search enumerates before giving up on counting.
#:
#: The cap above decides what comes back; this decides what can honestly
#: be *said* about what did not. Stopping the search at
#: :data:`MAX_TRACE_PATHS` meant the tool could not distinguish "there
#: were exactly 50" from "there were fifty thousand", so it reported
#: neither and the model assumed the former.
TRACE_COUNT_CEILING = MAX_TRACE_PATHS * 10


@dataclass(frozen=True)
class MoneyTrace:
    """Traced payment chains, carrying whether the list was cut down.

    Mirrors :class:`~shared.tool_interfaces.RingSubgraph`: the same
    ``truncated`` / ``total_*`` / ``note`` shape, for the same reason. A
    model handed 50 paths with no signal will say "funds moved along 50
    chains", which is a claim about the data rather than about the tool's
    ceiling, and the verifier will reject it.

    Iterating yields the paths, and ``len()`` counts those returned, so
    ``for path in trace_money_flow(...)`` and ``len(...)`` behave as they
    did when this was a bare list.

    Attributes:
        paths: The highest-value chains, at most :data:`MAX_TRACE_PATHS`,
            ordered by descending total amount.
        truncated: Whether paths were dropped.
        total_paths: How many were found. A lower bound when
            :attr:`total_is_lower_bound` is set.
        total_is_lower_bound: True when the search stopped at
            :data:`TRACE_COUNT_CEILING` before exhausting the graph, so
            ``total_paths`` is "at least this many", not "this many".
        note: Plain-language warning, empty when nothing was cut.
    """

    paths: list[list[Transfer]]
    truncated: bool
    total_paths: int
    total_is_lower_bound: bool
    note: str = ""

    def __iter__(self) -> Iterator[list[Transfer]]:
        """Yield the returned paths, as the bare list used to."""
        return iter(self.paths)

    def __len__(self) -> int:
        """Count the paths returned, not the paths that exist."""
        return len(self.paths)





class _TransferGraph:
    """Adjacency index over the transfers table, built once per store.

    Rows are materialised on demand from column lists, not up front.
    Building one ``itertuples`` row for each of ~570k transfers took ~46s
    on the real table and sat in front of the first ring-transfer question;
    positions per account via ``groupby`` plus column lists is ~5s, and a
    neighbour lookup builds only the handful of rows it returns.
    """

    def __init__(self, store: DataStore) -> None:
        frame = store.transfers_frame()
        columns = list(frame.columns)
        self._row = namedtuple("TransferRow", columns)  # noqa: PYI024 - built from the schema
        self._columns = [frame[column].tolist() for column in columns]
        self._outgoing = self._positions(frame, "from_hashed_account_id")
        self._incoming = self._positions(frame, "to_hashed_account_id")

    @staticmethod
    def _positions(frame: Any, column: str) -> dict[str, list[int]]:
        return {
            account: positions.tolist()
            for account, positions in frame.groupby(column, sort=False).indices.items()
        }

    def neighbours(self, account: str, direction: str) -> list[Any]:
        """Transfer rows leaving (downstream) or entering (upstream) an account."""
        table = self._outgoing if direction == "downstream" else self._incoming
        return [
            self._row(*(values[position] for values in self._columns))
            for position in table.get(account, ())
        ]


@functools.lru_cache(maxsize=1)
def _graph_cached() -> _TransferGraph:
    return _TransferGraph(default_store())


_GRAPH_LOCK = threading.Lock()


def _graph() -> _TransferGraph:
    """The shared index. Built under a lock so the startup warm-up and a
    request that arrives mid-build wait for one build instead of paying for two."""
    with _GRAPH_LOCK:
        return _graph_cached()


_graph.cache_clear = _graph_cached.cache_clear  # type: ignore[attr-defined]


def _store() -> DataStore:
    return default_store()


#: Memoises a deterministic tool on ``(data version, tool, args)``; see
#: :mod:`agents.tool_cache`.
_cached = tool_cache.cached(_store)


def get_ring_subgraph(
    ring_id: str,
    max_hops: int = 1,
) -> RingSubgraph:
    """Fetch a detected ring together with its local transaction subgraph.

    Args:
        ring_id: Identifier of the ring to expand.
        max_hops: How many transfer hops out from the ring members to
            include. Defaults to 1. Two hops on a real ring returns tens
            of thousands of accounts, because every mule account is a
            hub and the second hop pulls in the counterparties of every
            counterparty; ask for it only when one hop was not enough.

    Returns:
        A :class:`RingSubgraph`. It unpacks as the
        ``(ring, accounts, transfers)`` triple the contract specifies,
        and additionally reports whether the lists were capped at
        :data:`MAX_SUBGRAPH_ACCOUNTS` and
        :data:`MAX_SUBGRAPH_TRANSFERS`, with the true totals.

        **Check ``truncated`` before counting anything.** Ring members
        are always returned in full, so member counts are always safe;
        counterparty and transfer counts are not, when the flag is set.

    Raises:
        KeyError: If no ring exists with ``ring_id``.
        ValueError: If ``max_hops`` is less than 1.
    """
    if max_hops < 1:
        raise ValueError(f"max_hops must be at least 1, got {max_hops}")

    store = _store()
    ring = store.ring(ring_id)  # raises KeyError for an unknown ring
    graph = _graph()

    # Everyone, including same-bank context members: the subgraph is the
    # picture an analyst looks at, and omitting a context member would
    # hide the very edge that explains why it is in the ring.
    in_subgraph = set(ring.all_member_ids)
    frontier = set(in_subgraph)
    edges: dict[str, Any] = {}

    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for account in frontier:
            for direction in ("downstream", "upstream"):
                for row in graph.neighbours(account, direction):
                    edges[row.transfer_id] = row
                    counterparty = (
                        row.to_hashed_account_id
                        if direction == "downstream"
                        else row.from_hashed_account_id
                    )
                    if counterparty not in in_subgraph:
                        next_frontier.add(counterparty)
        in_subgraph |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    all_accounts = sorted(
        store.accounts(sorted(in_subgraph)), key=lambda a: a.hashed_account_id
    )
    all_transfers = sorted(
        (to_transfer(row) for row in edges.values()), key=lambda t: t.timestamp
    )

    # Ring members are never dropped: they are the subject of every claim
    # a caller will make, and a capped member list would produce a wrong
    # member count rather than merely a smaller picture. The cap falls on
    # the counterparties, which are context.
    members = set(ring.all_member_ids)
    member_accounts = [a for a in all_accounts if a.hashed_account_id in members]
    other_accounts = [a for a in all_accounts if a.hashed_account_id not in members]
    room = max(MAX_SUBGRAPH_ACCOUNTS - len(member_accounts), 0)
    accounts = member_accounts + other_accounts[:room]

    # Largest transfers first: if only some can be shown, the ones that
    # move the most money are the ones worth seeing.
    transfers = sorted(all_transfers, key=lambda t: t.amount, reverse=True)[
        :MAX_SUBGRAPH_TRANSFERS
    ]
    transfers.sort(key=lambda t: t.timestamp)

    truncated = len(accounts) < len(all_accounts) or len(transfers) < len(all_transfers)
    note = ""
    if truncated:
        note = (
            f"TRUNCATED: showing {len(accounts)} of {len(all_accounts)} accounts "
            f"and {len(transfers)} of {len(all_transfers)} transfers (largest by "
            "amount). All ring members are included, so member counts are exact; "
            "counterparty and transfer counts are not. Do not state a total "
            "derived from these lists -- use total_accounts and total_transfers."
        )

    return RingSubgraph(
        ring=ring,
        accounts=accounts,
        transfers=transfers,
        truncated=truncated,
        total_accounts=len(all_accounts),
        total_transfers=len(all_transfers),
        note=note,
    )




def _checked_round(model: _ScoreModelT) -> _ScoreModelT:
    """Round every registered score field, then verify it took.

    The assert is the point. Rounding by hand is what a tool *remembers*
    to do; asserting afterwards is what makes it unable to forget. When
    a contract model grows a score field this file has never heard of,
    the shared rounder handles it and this check proves it did -- rather
    than a raw float reaching an agent and failing a comparison later,
    somewhere with no evidence of where it came from.
    """
    rounded = round_score_fields(model)
    assert_score_fields_rounded(rounded)
    return rounded


#: Tables whose columns come from the pydantic contract rather than from
#: a BigQuery schema. ``shared/ddl.sql`` defines these as operational
#: tables and ``shared/bq_schemas.py`` deliberately does not carry them,
#: so the model is the authority. Both sources live in ``shared/``, so
#: neither can drift from the contract the way a hardcoded list would.
_CONTRACT_TABLES: dict[str, Any] = {
    "accounts": Account,
    "devices": Device,
    "kyc_documents": KycDocument,
    "kyc_submissions": KycSubmission,
    "cases": Investigation,
    "ring_members": RingMember,
}


def _columns_from_model(model: Any) -> list[dict[str, str]]:
    """Describe a pydantic model's fields as columns."""
    columns = []
    for field_name, field in model.model_fields.items():
        annotation = field.annotation
        type_name = getattr(annotation, "__name__", str(annotation))
        columns.append(
            {
                "name": field_name,
                "type": type_name,
                "required": "yes" if field.is_required() else "no",
                "description": (field.description or "").split("\n")[0],
            }
        )
    return columns


#: Appended to an array column's description. ``run_sql`` executes on
#: SQLite, where a list cell is stored as JSON text -- so the only way
#: to filter one is a string match, and a model told merely "ARRAY" will
#: write ``WHERE 'BANK_C' IN banks_involved`` and get nothing back.
#: SQLite's JSON1 extension is built in and available through ``run_sql``,
#: so an array column can be counted and expanded, not merely matched.
#: Saying only "filter with LIKE" is what made a live model refuse an
#: answerable question -- it concluded array length was unavailable and
#: gave up rather than reaching for json_array_length.
_ARRAY_HINT = (
    " (array stored as JSON text; SQLite JSON1 works: count with "
    "json_array_length(col), expand with json_each(col), match with "
    "col LIKE '%value%' -- e.g. json_array_length(banks_involved) = 3)"
)


def _columns_from_bq(fields: list[Any]) -> list[dict[str, str]]:
    """Describe a BigQuery schema's fields as columns."""
    columns = []
    for field in fields:
        is_array = field.mode == "REPEATED" or field.field_type == "RECORD"
        description = (field.description or "").split("\n")[0][:200]
        columns.append(
            {
                "name": field.name,
                "type": f"ARRAY<{field.field_type}>" if is_array else field.field_type,
                "required": "yes" if field.mode == "REQUIRED" else "no",
                "description": description + (_ARRAY_HINT if is_array else ""),
            }
        )
    return columns


@_cached
def describe_tables(table_name: str | None = None) -> dict[str, Any]:
    """List the tables ``run_sql`` may query, with their columns and types.

    Call this **before** writing a query. The alternative is guessing a
    column name, and a guess costs a whole turn: a query naming a column
    that does not exist fails, and the failure is all the model gets
    back.

    Args:
        table_name: Describe only this table. When ``None``, every
            allowed table is described.

    Returns:
        A mapping with ``"tables"`` -- table name to its column list,
        each column having ``name``, ``type``, ``required`` and
        ``description`` -- and ``"row_limit"``, the maximum number of
        rows any query may return.

    Raises:
        ValueError: If ``table_name`` is not an allowed table. The
            message lists the allowed ones.
    """
    from shared.bq_schemas import TABLE_SCHEMAS  # noqa: PLC0415 - avoids a cycle

    if table_name is not None and table_name not in ALLOWED_SQL_TABLES:
        raise ValueError(
            f"{table_name!r} is not a queryable table; allowed tables are "
            f"{sorted(ALLOWED_SQL_TABLES)}"
        )

    wanted = {table_name} if table_name else set(ALLOWED_SQL_TABLES)
    described: dict[str, list[dict[str, str]]] = {}
    for name in sorted(wanted):
        if name in _CONTRACT_TABLES:
            columns = _columns_from_model(_CONTRACT_TABLES[name])
            if name == "ring_members":
                # RingMember is the child row; the parent key is not on
                # the model but is on the table, per shared/ddl.sql.
                columns.insert(
                    0,
                    {
                        "name": "ring_id",
                        "type": "str",
                        "required": "yes",
                        "description": "The ring this membership belongs to.",
                    },
                )
            described[name] = columns
        elif name in TABLE_SCHEMAS:
            described[name] = _columns_from_bq(TABLE_SCHEMAS[name])
        else:  # pragma: no cover - guards a future allowlist addition
            described[name] = [
                {
                    "name": "(unknown)",
                    "type": "(unknown)",
                    "required": "no",
                    "description": (
                        "no schema published for this table in shared/; query it "
                        "with SELECT * and a small LIMIT to see its columns"
                    ),
                }
            ]
    described = {
        name: [c for c in columns if not hidden_labels.is_denied_column(c["name"])]
        for name, columns in described.items()
    }
    return {"tables": described, "row_limit": SQL_ROW_LIMIT}


@_cached
def get_ring_summary(ring_id: str) -> dict[str, Any]:
    """Summarise a ring: counts, banks, behaviour and its explanation.

    **Call this first.** It answers most questions about a ring in a
    couple of kilobytes, where :func:`get_ring_subgraph` returns every
    account and transfer and ran to roughly 113,000 prompt tokens on a
    live investigation -- enough to make the investigator the slowest
    agent in the system. Reach for the subgraph only when a specific
    account or payment actually has to be named.

    Every member is counted; nothing here is truncated.

    Args:
        ring_id: Identifier of the ring to summarise.

    Returns:
        A mapping with:

        * ``member_count`` -- every member, context included.
        * ``alertable_member_count`` -- the count an **unqualified**
          statement about the ring must use.
        * ``context_member_count`` -- same-bank identity links, shown to
          an analyst but never counted as alerts.
        * ``velocity_member_count`` / ``identity_link_member_count`` --
          the split by how each member arrived.
        * ``banks_involved`` and ``bank_count``.
        * ``identity_links`` -- one entry per linked member with its
          scope, evidence type and verbatim ``reason``.
        * ``behaviour`` -- aggregate stats over alertable members.
        * ``explanation`` -- the stored ring narrative and top factors.

    Raises:
        KeyError: If no ring exists with ``ring_id``.
    """
    store = _store()
    ring = store.ring(ring_id)  # raises KeyError for an unknown ring

    links = [
        {
            "hashed_account_id": member.hashed_account_id,
            "scope": member.link_scope.value if member.link_scope else None,
            "alertable": member.alertable,
            "evidence_type": member.link_evidence_type,
            # Verbatim: the reason is written as a lead and reviewed as
            # one. Paraphrasing is how a lead becomes an accusation.
            "reason": member.reason,
        }
        for member in ring.members
        if member.membership_basis is MembershipBasis.IDENTITY_LINK
    ]

    scores = [
        score.score
        for member_id in ring.alertable_member_ids
        if (score := store.risk_score(member_id)) is not None
    ]
    behaviour: dict[str, Any] = {
        "scored_alertable_members": len(scores),
        "mean_score": round(sum(scores) / len(scores), SCORE_DECIMALS)
        if scores
        else None,
        "max_score": round(max(scores), SCORE_DECIMALS) if scores else None,
        "min_score": round(min(scores), SCORE_DECIMALS) if scores else None,
    }

    try:
        explanation = store.ring_explanation(ring_id)
        explained = {
            "narrative": explanation.narrative,
            "top_factors": list(explanation.top_factors),
        }
    except KeyError:
        explained = {"narrative": "", "top_factors": []}

    return {
        "ring_id": ring.ring_id,
        "status": ring.status.value,
        "detected_at": ring.detected_at,
        "member_count": len(ring.all_member_ids),
        "alertable_member_count": len(ring.alertable_member_ids),
        "context_member_count": len(ring.context_member_ids),
        "velocity_member_count": len(ring.velocity_member_ids),
        "identity_link_member_count": len(ring.identity_link_member_ids),
        "banks_involved": sorted(ring.banks_involved),
        "bank_count": len(ring.banks_involved),
        "identity_links": links,
        "behaviour": behaviour,
        "explanation": explained,
        "counting_note": (
            f"An unqualified count of this ring is {len(ring.alertable_member_ids)}, "
            f"the alertable members. The full membership is "
            f"{len(ring.all_member_ids)} and may only be stated alongside the "
            f"{len(ring.context_member_ids)} same-bank context member(s) it includes."
        ),
    }


def get_explanation(
    subject_id: str,
    subject_type: str = "ring",
) -> RingExplanation | AccountExplanation:
    """Fetch the stored model explanation for a ring or an account.

    Args:
        subject_id: A ring id when ``subject_type`` is ``"ring"``, or a
            hashed account id when it is ``"account"``.
        subject_type: Either ``"ring"`` or ``"account"``.

    Returns:
        A :class:`~shared.contracts.RingExplanation` for a ring, or a
        :class:`~shared.contracts.AccountExplanation` for an account.

    Raises:
        KeyError: If no explanation has been generated for ``subject_id``.
        ValueError: If ``subject_type`` is not ``"ring"`` or ``"account"``.
    """
    if subject_type not in {"ring", "account"}:
        raise ValueError(
            f"subject_type must be 'ring' or 'account', got {subject_type!r}"
        )
    store = _store()
    if subject_type == "ring":
        return store.ring_explanation(subject_id)
    return _checked_round(store.account_explanation(subject_id))


def trace_money_flow(
    hashed_account_id: str,
    direction: str = "downstream",
    max_hops: int = 3,
    since: datetime | None = None,
) -> MoneyTrace:
    """Trace chains of transfers into or out of an account.

    Follows funds hop by hop to expose pass-through behaviour, including
    across bank boundaries.

    Args:
        hashed_account_id: Hashed account id to start from.
        direction: ``"downstream"`` to follow outgoing transfers,
            ``"upstream"`` to follow incoming ones.
        max_hops: Maximum chain length to return.
        since: Only consider transfers at or after this timestamp. When
            ``None``, the full retained history is used.

    Returns:
        One list of transfers per traced path, ordered along the flow of
        funds and sorted by descending total amount. At most
        :data:`MAX_TRACE_PATHS` paths are returned.

    Raises:
        KeyError: If ``hashed_account_id`` is unknown.
        ValueError: If ``direction`` is invalid or ``max_hops`` is less than 1.
    """
    if direction not in {"downstream", "upstream"}:
        raise ValueError(
            f"direction must be 'downstream' or 'upstream', got {direction!r}"
        )
    if max_hops < 1:
        raise ValueError(f"max_hops must be at least 1, got {max_hops}")

    _store().account(hashed_account_id)  # raises KeyError for an unknown account
    graph = _graph()

    paths: list[list[Any]] = []
    # Depth-first, tracking visited accounts per path so a cycle cannot
    # loop forever. Partial paths are kept: a two-hop chain that dead-ends
    # is still evidence of pass-through.
    stack: list[tuple[str, list[Any], set[str]]] = [
        (hashed_account_id, [], {hashed_account_id})
    ]
    # Explore past the return cap so the count reported back is real
    # rather than "however many I stopped at". Bounded, because a hub
    # account branches combinatorially and counting every path is the
    # same explosion the cap exists to avoid.
    while stack and len(paths) < TRACE_COUNT_CEILING:
        account, path, seen = stack.pop()
        rows = [
            row
            for row in graph.neighbours(account, direction)
            if since is None or row.timestamp >= since
        ]
        if not rows or len(path) >= max_hops:
            if path:
                paths.append(path)
            continue
        for row in rows:
            counterparty = (
                row.to_hashed_account_id
                if direction == "downstream"
                else row.from_hashed_account_id
            )
            if counterparty in seen:
                if path:
                    paths.append(path + [row])
                continue
            stack.append((counterparty, path + [row], seen | {counterparty}))

    converted = [[to_transfer(row) for row in path] for path in paths]
    converted.sort(key=lambda path: sum(t.amount for t in path), reverse=True)

    kept = converted[:MAX_TRACE_PATHS]
    hit_ceiling = len(converted) >= TRACE_COUNT_CEILING
    truncated = len(kept) < len(converted)
    note = ""
    if truncated:
        qualifier = "at least " if hit_ceiling else ""
        note = (
            f"TRUNCATED: showing the {len(kept)} highest-value paths of "
            f"{qualifier}{len(converted)} found. Do not state a path count or "
            "a total traced amount from this list -- use total_paths, and note "
            "it is a lower bound when total_is_lower_bound is true."
        )
    return MoneyTrace(
        paths=kept,
        truncated=truncated,
        total_paths=len(converted),
        total_is_lower_bound=hit_ceiling,
        note=note,
    )


def get_account_profile(
    hashed_account_id: str,
) -> tuple[Account, RiskScore | None, list[str]]:
    """Fetch an account with its latest risk score and ring memberships.

    Args:
        hashed_account_id: Hashed account id to look up.

    Returns:
        A tuple of the account, its most recent risk score (``None`` if the
        account has never been scored), and the ids of rings it belongs to.

    Raises:
        KeyError: If ``hashed_account_id`` is unknown.
    """
    store = _store()
    account = store.account(hashed_account_id)  # raises KeyError
    return (
        account,
        # None-guarded: an unscored account has no score to round, and
        # the shared rounder takes a model, not an Optional.
        _checked_round(score) if (score := store.risk_score(hashed_account_id)) else None,
        store.rings_for_account(hashed_account_id),
    )


def get_document_reuse(
    hashed_account_id: str,
) -> list[AccountEvidence]:
    """Find KYC documents this account shares with other accounts.

    Reuse is detected on the hashed id number, the normalized address and
    the photo embedding reference produced by ``docai/``.

    **This is evidence, not a detection.** Document reuse plays no part
    in the firing decision, and it is a weak signal on its own: far more
    legitimate customers reuse a document within one bank than mules do,
    and dual-bank customers legitimately reuse an id across two banks.
    Callers must therefore read every row together with its
    :attr:`~shared.contracts.AccountEvidence.scope` and must not present
    a ``single_bank`` row as incriminating.

    Args:
        hashed_account_id: Hashed account id to check.

    Returns:
        One :class:`~shared.contracts.AccountEvidence` row per reuse
        observed, each naming the reused attribute in ``evidence_type``,
        whether it is ``single_bank`` or ``cross_bank`` in ``scope``, the
        counterparty accounts and the banks they sit at. Rows from this
        tool always carry ``source_table="kyc_documents"``. Empty when
        the account reuses nothing.

    Raises:
        KeyError: If ``hashed_account_id`` has no KYC submission on file.
    """
    store = _store()
    own = store.kyc_documents(hashed_account_id)
    if not own:
        raise KeyError(f"no KYC submission on file for {hashed_account_id!r}")

    own_bank = store.account(hashed_account_id).bank_id
    all_docs = store.table("kyc_documents")
    others = all_docs[all_docs["hashed_account_id"] != hashed_account_id]
    bank_of = store.bank_index()

    rows: list[AccountEvidence] = []
    for attribute in REUSE_ATTRIBUTES:
        values = {
            getattr(doc, attribute)
            for doc in own
            if getattr(doc, attribute) is not None
        }
        if not values:
            continue
        hit = others[others[attribute].isin(values)]
        if hit.empty:
            continue

        counterparties = sorted(set(hit["hashed_account_id"]))
        # One row per scope: the same attribute can be reused by a
        # sibling at the same bank (noise) and by an account at another
        # bank (citable), and those two facts must not be merged.
        by_scope: dict[EvidenceScope, list[str]] = {}
        for account in counterparties:
            scope = (
                EvidenceScope.SINGLE_BANK
                if bank_of.get(account) == own_bank
                else EvidenceScope.CROSS_BANK
            )
            by_scope.setdefault(scope, []).append(account)

        for scope, accounts_in_scope in by_scope.items():
            banks = sorted({bank_of[a] for a in accounts_in_scope if a in bank_of})
            rows.append(
                AccountEvidence(
                    hashed_account_id=hashed_account_id,
                    evidence_type=f"{scope.value}_reused_{attribute}",
                    scope=scope,
                    detail=_reuse_detail(scope, attribute, accounts_in_scope, banks),
                    counterparty_hashed_account_ids=sorted(accounts_in_scope),
                    counterparty_bank_ids=banks,
                    observed_at=_reuse_observed_at(store, hashed_account_id,
                                                   accounts_in_scope),
                    source_table="kyc_documents",
                    detector_version=None,
                )
            )
    return rows


def _reuse_detail(
    scope: EvidenceScope,
    attribute: str,
    counterparties: list[str],
    banks: list[str],
) -> str:
    """Compose the ready-to-cite sentence for one reuse row.

    The caveat is part of the sentence, not an afterthought, so a
    narrative cannot quote the fact while dropping its qualification.
    """
    count = len(counterparties)
    where = ", ".join(banks) if banks else "an unknown bank"
    if scope is EvidenceScope.CROSS_BANK:
        return (
            f"The {attribute} on this account's KYC file is also on file for "
            f"{count} account(s) at {where}. Cross-bank reuse is the citable "
            "form of this signal, but it is corroborating evidence only: "
            "dual-bank customers reuse identifiers legitimately, so this does "
            "not by itself establish suspicion."
        )
    return (
        f"The {attribute} on this account's KYC file is also on file for "
        f"{count} account(s) at the same bank ({where}). Same-bank reuse is a "
        "weak signal that arises routinely among households and joint "
        "businesses, and must not be presented as incriminating."
    )


def _reuse_observed_at(
    store: DataStore,
    hashed_account_id: str,
    counterparties: list[str],
) -> datetime:
    """When the reuse link first became citable.

    The later of the two KYC submission times: a bank cannot act on a
    pair before both sides of it exist. ``AccountEvidence.observed_at``
    is required precisely so this is never silently defaulted to now.
    """
    submitted = store.submission_times()
    own = submitted.get(hashed_account_id)
    others = [submitted[a] for a in counterparties if a in submitted]
    candidates = [t for t in ([own] + others) if t is not None]
    if not candidates:
        raise KeyError(
            f"no KYC submission timestamp for {hashed_account_id!r} or its counterparties"
        )
    return max(candidates)


def search_regulations(
    query: str,
    top_k: int = 5,
) -> list[dict[str, str]]:
    """Search the AML regulation corpus for passages to cite in a report.

    Backed by Vertex AI Search over the indexed regulation documents when
    ``docai/`` provides it, and by the built-in corpus in
    :mod:`agents.regulations` otherwise.

    Args:
        query: Natural-language description of the suspicious behaviour.
        top_k: Maximum number of passages to return.

    Returns:
        A list of results ordered by relevance, each with the keys
        ``"citation"`` (regulation and section reference), ``"excerpt"``
        (the matching text) and ``"source_uri"``.

    Raises:
        ValueError: If ``query`` is empty or ``top_k`` is less than 1.
        RuntimeError: If the search backend is unavailable.
    """
    return regulations.search(query, top_k)


@_cached
def run_sql(query: str) -> list[dict[str, Any]]:
    """Run an analyst-supplied read-only query against the MuleNet tables.

    The query is validated before execution and must satisfy all of:

    * It is a single read-only ``SELECT`` statement. DML, DDL, and any
      other statement type is rejected, as are multiple statements.
    * Every table it references is in
      :data:`~shared.tool_interfaces.ALLOWED_SQL_TABLES`.
    * A ``LIMIT`` of at most
      :data:`~shared.tool_interfaces.SQL_ROW_LIMIT` (100) is enforced; if
      the query omits a limit, one is appended.

    Validation happens on a parsed syntax tree, never on the query text;
    see :mod:`agents.sql_guard` for the rules.

    Args:
        query: The ``SELECT`` statement to run.

    Returns:
        Up to :data:`~shared.tool_interfaces.SQL_ROW_LIMIT` rows, each as
        a column-name to value mapping.

    Raises:
        ValueError: If the query is not a single ``SELECT``, references a
            table outside the allowlist, or requests more than
            :data:`~shared.tool_interfaces.SQL_ROW_LIMIT` rows.
        RuntimeError: If the query is valid but fails to execute.
    """
    return execute_select(_store(), validate_sql(query))


@_cached
def get_ring_transfer_summary(ring_id: str) -> dict[str, Any]:
    """Aggregate the money moving inside a ring. Counts and totals, never rows.

    **Use this instead of querying ``transfers`` for a ring total.** It
    exists because the obvious alternative is ruinous: asked for a ring's
    total intra-ring amount, the model wrote a three-way join across
    ``transfers`` (101 MB, ~570k rows) and ``ring_members`` twice, and the
    request hit the 90-second wall clock without returning. This answers
    the same question off the adjacency index already built for the other
    tools, in a few hundred bytes.

    "Intra-ring" means **both** endpoints are members of this ring. A
    transfer from a member to an outside account is not intra-ring, and
    is not counted in any total here; ``boundary_transfer_count`` reports
    how many such edges exist so their absence is visible rather than
    silent.

    Every member is considered and no list is capped, so unlike
    :func:`get_ring_subgraph` **every number here is exact** and there is
    no truncation flag to check.

    Args:
        ring_id: Identifier of the ring to aggregate.

    Returns:
        A mapping with:

        * ``transfer_count`` -- intra-ring transfers.
        * ``total_amount`` / ``mean_amount`` / ``min_amount`` /
          ``max_amount`` -- exact to the cent, as strings, because these
          are money and a float total is a rounding argument waiting to
          happen.
        * ``currency`` -- the currency of the amounts, or ``"MIXED"``.
        * ``cross_bank_transfer_count`` / ``cross_bank_amount`` -- the
          subset whose two endpoints sit at different banks. This is the
          cross-boundary movement the detector fires on.
        * ``distinct_member_pairs`` -- ordered (from, to) pairs used.
        * ``boundary_transfer_count`` -- edges with exactly one endpoint
          in the ring, excluded from the totals above.
        * ``first_transfer_at`` / ``last_transfer_at`` / ``span_hours`` --
          the time window the intra-ring activity occupies. Velocity is
          what the detector fires on, so the span is what makes a total
          mean something.
        * ``member_count`` -- members considered, for cross-checking
          against :func:`get_ring_summary`.

    Raises:
        KeyError: If no ring exists with ``ring_id``.
    """
    store = _store()
    ring = store.ring(ring_id)  # raises KeyError for an unknown ring
    graph = _graph()
    bank_of = store.bank_index()

    members = set(ring.all_member_ids)
    intra: dict[str, Any] = {}
    boundary = 0
    # Outgoing only: iterating both directions would visit every
    # intra-ring edge twice, and de-duplicating on transfer_id already
    # handles the member-to-member case. Counting boundary edges needs
    # both, so incoming is walked separately below.
    for member in members:
        for row in graph.neighbours(member, "downstream"):
            if row.to_hashed_account_id in members:
                intra[row.transfer_id] = row
            else:
                boundary += 1
        for row in graph.neighbours(member, "upstream"):
            if row.from_hashed_account_id not in members:
                boundary += 1

    rows = list(intra.values())
    amounts = [Decimal(str(row.amount)) for row in rows]
    currencies = {str(getattr(row, "currency", "")) for row in rows} - {""}
    cross = [
        (row, amount)
        for row, amount in zip(rows, amounts, strict=True)
        if bank_of.get(row.from_hashed_account_id)
        != bank_of.get(row.to_hashed_account_id)
    ]
    timestamps = sorted(row.timestamp for row in rows)

    def money(value: Decimal) -> str:
        return str(value.quantize(Decimal("0.01")))

    span_hours = None
    if len(timestamps) >= 2:
        span_hours = round(
            (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0, 2
        )

    return {
        "ring_id": ring_id,
        "member_count": len(members),
        "transfer_count": len(rows),
        "total_amount": money(sum(amounts, Decimal(0))) if amounts else "0.00",
        "mean_amount": money(sum(amounts, Decimal(0)) / len(amounts))
        if amounts
        else None,
        "min_amount": money(min(amounts)) if amounts else None,
        "max_amount": money(max(amounts)) if amounts else None,
        "currency": (
            currencies.pop() if len(currencies) == 1 else ("MIXED" if currencies else None)
        ),
        "cross_bank_transfer_count": len(cross),
        "cross_bank_amount": money(sum((a for _, a in cross), Decimal(0)))
        if cross
        else "0.00",
        "distinct_member_pairs": len(
            {(row.from_hashed_account_id, row.to_hashed_account_id) for row in rows}
        ),
        "boundary_transfer_count": boundary,
        "first_transfer_at": timestamps[0].isoformat() if timestamps else None,
        "last_transfer_at": timestamps[-1].isoformat() if timestamps else None,
        "span_hours": span_hours,
        "counting_note": (
            "Exact: every member and every intra-ring transfer is counted, "
            "nothing is truncated. Totals cover transfers with BOTH endpoints "
            "in this ring; boundary_transfer_count is the edges excluded."
        ),
    }


class RaceDataUnavailable(RuntimeError):
    """The published race artifact is missing, unreadable or too old to trust.

    A tool error the model reads and relays, never a crash: the rest of Ask
    keeps working without it.
    """


class MetricsUnavailable(RuntimeError):
    """The published metric catalog is missing, unreadable or too old to trust."""


_PUBLISHED_LOCK = threading.Lock()
_PUBLISHED_CACHE: dict[str, tuple[tuple[int, int], Any]] = {}


def _published_json(
    name: str,
    schema_version: int,
    unavailable: type[RuntimeError],
    written_by: str,
    store: DataStore | None = None,
) -> dict[str, Any]:
    """An artifact under ``agents/out/``, parsed once per file version.

    Keyed on the file's mtime and size so a re-published artifact is picked up
    without a restart and an unchanged one is not re-parsed per call. Every
    failure is ``unavailable`` with a message that starts "... unavailable", so
    the model reports it plainly instead of improvising a number.
    """
    path = cases.cases_dir((store or _store()).config) / name
    try:
        stat = path.stat()
    except OSError:
        raise unavailable(
            f"data unavailable: {name} has not been published to this service "
            f"(it is written by `{written_by}`)"
        ) from None
    version = (stat.st_mtime_ns, stat.st_size)
    with _PUBLISHED_LOCK:
        cached = _PUBLISHED_CACHE.get(str(path))
        if cached and cached[0] == version:
            return cached[1]
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise unavailable(f"data unavailable: {name} could not be read") from None
        if not isinstance(value, dict) or value.get("schema_version") != schema_version:
            raise unavailable(f"data unavailable: {name} has an unexpected schema")
        _PUBLISHED_CACHE[str(path)] = (version, value)
        return value


def _race_artifact(store: DataStore | None = None) -> dict[str, Any]:
    try:
        return _published_json(
            ring_race.ARTIFACT_NAME,
            ring_race.SCHEMA_VERSION,
            RaceDataUnavailable,
            "python -m ui.build_snapshot",
            store,
        )
    except RaceDataUnavailable as exc:
        message = str(exc).replace("data unavailable", "race data unavailable")
        raise RaceDataUnavailable(message) from None


def _metric_catalog() -> dict[str, Any]:
    try:
        return _published_json(
            metric_catalog.ARTIFACT_NAME,
            metric_catalog.SCHEMA_VERSION,
            MetricsUnavailable,
            "python -m ml.claims",
        )
    except MetricsUnavailable as exc:
        message = str(exc).replace("data unavailable", "metrics unavailable")
        raise MetricsUnavailable(message) from None


def get_ring_race(ring_id: str) -> dict[str, Any]:
    """When single-bank and cross-bank flagged a ring, against when its money left.

    **The tool for "race to the exit" and "when did each flag this ring"
    questions.** It reads the evaluation run's detection timeline, joined to the
    detector's own ring id, so ``ring_id`` must be a DETECTED ring id such as
    ``ring-mulenet-velocity-rule-v2-00002``.

    Everything is copied from, or is a difference of two numbers in, that
    published timeline. Nothing is estimated, and no simulator ground truth
    (true ring ids, mule flags, roles, cohorts) is returned.

    Args:
        ring_id: A detected ring id that exists in the ``rings`` table.

    Returns:
        A mapping with:

        * ``ring_id``, ``span`` and ``banks_involved``.
        * ``ambiguous`` -- true when the detected cluster mixes parts of several
          simulated rings. Then ``available`` is false, ``reason`` says so and no
          race is given: any single race would describe one of them, not the
          cluster. Say that to the asker.
        * ``available`` -- false also when no timeline exists for the cluster.
        * ``scopes`` -- ``single_bank`` and ``cross_bank``, each with
          ``first_flag_T`` (the earliest horizon in minutes at which that scope
          had flagged the ring, or null), ``flag_offset_minutes``,
          ``minutes_before_exit`` (positive = before the money reached the exit
          account, negative = after, null if it never flagged inside the grid),
          ``minutes_after_exit`` (the same gap as a positive number when it
          flagged after the exit, else null),
          ``caught_in_time`` and ``flagged_full_window``.
        * ``exit_reached`` and ``exit_offset_minutes`` -- the exit account, and
          the minutes after the ring's onset at which money reached it.
        * ``cross_bank_earlier`` and ``earlier_by_minutes`` -- null unless both
          scopes flagged inside the grid.
        * ``joining_helps`` -- cross-bank caught it before the exit and either
          single-bank did not or cross-bank was strictly earlier. A rule over
          this one ring, not a metric.
        * ``horizon_grid`` and ``label`` -- the horizons measured, and that this
          is one simulated evaluation run.

    Raises:
        KeyError: If no ring exists with ``ring_id``.
        RaceDataUnavailable: If the race artifact has not been published here, or
            has no entry for this ring.
    """
    return ring_race_for(_store(), ring_id)


def ring_race_for(store: DataStore, ring_id: str) -> dict[str, Any]:
    """:func:`get_ring_race` over a given store, for callers that hold one."""
    ring = store.ring(ring_id)  # raises KeyError for an unknown ring
    artifact = _race_artifact(store)
    entry = artifact.get("rings", {}).get(ring_id)
    if entry is None:
        raise RaceDataUnavailable(
            f"race data unavailable for {ring_id}: the published artifact has no entry for it"
        )
    result: dict[str, Any] = {
        "ring_id": ring.ring_id,
        "span": len(ring.banks_involved),
        "banks_involved": sorted(ring.banks_involved),
        "label": artifact.get("label", ""),
        "horizon_grid": list(artifact.get("horizon_grid", [])),
        "available": bool(entry["available"]),
        "ambiguous": entry["ambiguous"],
    }
    if not entry["available"]:
        result["reason"] = entry["reason"]
        return result
    def scope_fields(scope: dict[str, Any]) -> dict[str, Any]:
        fields = {k: v for k, v in scope.items() if k != "scope"}
        before = fields["minutes_before_exit"]
        # The same number as a plain magnitude, so "flagged 80 minutes after
        # the exit" is a figure the answer check can find in a tool result:
        # it compares numbers as written, and the signed one reads -80.
        fields["minutes_after_exit"] = -before if before is not None and before < 0 else None
        return fields

    result.update(
        scopes={s["scope"]: scope_fields(s) for s in entry["scopes"]},
        exit_reached=entry["exit_reached"],
        exit_offset_minutes=entry["exit_offset_minutes"],
        cross_bank_earlier=entry["cross_bank_earlier"],
        earlier_by_minutes=entry["earlier_by_minutes"],
        joining_helps=entry["joining_helps"],
    )
    return result


def describe_metrics() -> dict[str, Any]:
    """List the charts and claims :func:`get_metric` can return. Call this to discover names.

    Returns:
        A mapping with ``charts`` (each ``name`` is a Metrics-tab chart you may
        pass to ``get_metric``, with its ``title`` and ``claim_ids``), ``claims``
        (every quotable claim, by ``id``, with its statement), ``retired``
        (numbers the project withdrew, which must NOT be quoted) and ``race``
        (the race-to-the-exit chart is answered by ``get_ring_race``, not here).

    Raises:
        MetricsUnavailable: If the metric catalog has not been published here.
    """
    catalog = _metric_catalog()
    return {
        "charts": [
            {"name": name, "title": chart["title"], "claim_ids": chart["claim_ids"]}
            for name, chart in catalog["charts"].items()
        ],
        "claims": [
            {"id": claim["id"], "statement": claim["statement"]}
            for claim in catalog["claims"].values()
        ],
        "retired": catalog["retired"],
        "race": (
            f"`{metric_catalog.RACE_CHART}` is answered by get_ring_race(ring_id) for a "
            "detected ring, not by get_metric"
        ),
    }


def get_metric(name: str) -> dict[str, Any]:
    """The measured value(s), qualifier and source file for a chart or a claim.

    **The tool for "explain this chart" and "what is the number for X" questions
    about the Metrics tab.** ``name`` is a chart name or a claim id from
    :func:`describe_metrics`. Values are copied from the published metrics files
    exactly as ``docs/claims.md`` states them; nothing is recomputed. **State the
    ``qualifier`` with every number**: several are wrong if quoted bare (a
    coverage share is unconditional, a lead-time median is conditional).

    Args:
        name: A chart name (for example ``false_alerts_per_catch_t5``) or a claim
            id (for example ``alert_burden_t5_cross_bank``).

    Returns:
        A mapping with ``kind`` (``chart`` or ``claim``), ``title``, ``values``
        (each: ``id``, ``statement``, ``value``, ``qualifier``, ``source_file``,
        ``path``, and ``percent`` for shares, the value to one decimal as the tab
        prints it), and for some charts ``extra`` figures and a
        ``documentation_discrepancy`` that must be reported, not resolved.

    Raises:
        ValueError: If ``name`` is not on the menu (the message lists it), or is
            a retired claim.
        MetricsUnavailable: If the metric catalog has not been published here.
    """
    catalog = _metric_catalog()
    charts, claims = catalog["charts"], catalog["claims"]
    if name in charts:
        chart = charts[name]
        result: dict[str, Any] = {
            "name": name,
            "kind": "chart",
            "title": chart["title"],
            "values": [claims[claim_id] for claim_id in chart["claim_ids"]],
        }
        for key in ("extra", "documentation_discrepancy"):
            if key in chart:
                result[key] = chart[key]
        return result
    if name in claims:
        claim = claims[name]
        return {"name": name, "kind": "claim", "title": claim["statement"], "values": [claim]}
    retired = next((item for item in catalog["retired"] if item["id"] == name), None)
    if retired is not None:
        raise ValueError(f"{name!r} was withdrawn and must not be quoted: {retired['reason']}")
    if name == metric_catalog.RACE_CHART:
        raise ValueError(f"{name!r} is answered by get_ring_race(ring_id), not get_metric")
    menu = ", ".join([*charts, *claims])
    raise ValueError(f"unknown metric {name!r}; the menu is: {menu}")


#: Every tool, in the order ``shared/tool_interfaces.py`` declares them,
#: with ``get_ring_transfer_summary`` appended.
#:
#: The appended one has no entry in ``shared/tool_interfaces.py`` because
#: that file is frozen. It is a local aggregate over tables the contract
#: already exposes, not a new data source, so nothing in ``shared/`` needs
#: to change for it to be correct -- but it is listed in
#: ``docs/post_submission.md`` as a signature that belongs there.
ALL_TOOLS = (
    describe_tables,
    get_ring_summary,
    get_ring_transfer_summary,
    get_ring_race,
    describe_metrics,
    get_metric,
    get_ring_subgraph,
    get_explanation,
    trace_money_flow,
    get_account_profile,
    get_document_reuse,
    search_regulations,
    run_sql,
)
