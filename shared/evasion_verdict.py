"""Structured evasion verdicts from the red-team clause tables.

Pure Python, no third-party imports: the UI image ships ``shared/`` but not
``ml/``, and both it (for the bundled red-team strategies) and the agents
service (for live ``/simulate`` runs) need the same derivation, so it lives
here once instead of being retyped on each side.

A clause table is what ``ml.redteam.clause_report`` produces per collector::

    {collector_id: {clause: {"value": ..., "passes": bool, "requirement": str}}}

The rules below are the ones ``ml.redteam._reasoning`` already applies in
prose; this returns them as data so a UI can write its own copy.
"""

from __future__ import annotations

from typing import Any

#: The five clauses of the shipped velocity rule, in the order
#: ``ml.redteam.clause_report`` emits them.
CLAUSES: tuple[str, ...] = (
    "passthrough_speed_seconds",
    "passthrough_ratio",
    "in_degree",
    "out_degree",
    "distinct_counterparty_banks",
)

RING_CLUSTERING = "ring_clustering"


def failing_clauses(clause_values: dict[str, dict[str, Any]]) -> list[str]:
    """Clauses that fail on **every** collector.

    A clause that holds one collector back while another fires is not what
    stopped the pattern, so it is not listed -- same test as
    ``ml.redteam._reasoning``.
    """
    tables = list(clause_values.values())
    if not tables:
        return []
    return [
        name
        for name in CLAUSES
        if name in tables[0] and all(not table[name]["passes"] for table in tables)
    ]


def structured_verdict(
    *,
    caught_by: str,
    clause_values: dict[str, dict[str, Any]],
    aimed_clause: str,
    flagged_accounts: int,
    flagged_collectors: int,
    rings_formed: int,
    stage2_alertable: int,
) -> dict[str, Any]:
    """The machine-readable verdict.

    ``aimed_clause_passed`` is ``True`` when the clause the strategy aimed at
    still passed (the pattern did not evade it), ``False`` when it failed,
    and ``None`` when that is not defined: an unrecognised clause name, or
    ``ring_clustering`` on a pattern whose collectors were never flagged (there
    was nothing for clustering to join). ``is_designed_miss`` is only ever
    true for a MISSED verdict whose failing clauses include the aimed one.
    """
    verdict = "MISSED" if caught_by == "missed" else "CAUGHT"
    failing = failing_clauses(clause_values)
    if aimed_clause in CLAUSES:
        aimed_passed: bool | None = aimed_clause not in failing
    elif aimed_clause == RING_CLUSTERING:
        aimed_passed = rings_formed > 0 if flagged_collectors else None
    else:
        aimed_passed = None
    return {
        "verdict": verdict,
        "failing_clauses": failing if verdict == "MISSED" else [],
        "aimed_clause": aimed_clause,
        "aimed_clause_passed": aimed_passed,
        "is_designed_miss": verdict == "MISSED" and aimed_clause in failing,
        "flagged_accounts": flagged_accounts,
        "cash_out_accounts_flagged": flagged_collectors,
        "rings_formed": rings_formed,
        "stage2_alertable": stage2_alertable,
    }
