"""The simulator's hidden labels, and the rule that Ask never exposes them.

MuleNet is a demo over synthetic data, so the generator knows which accounts
are "really" mules. Those labels exist to *score* the detector. The UI has a
Reveal control that shows them when the user asks; **Ask is never allowed to
show them**, whether the user asks for them or a prompt injection does,
because an answer from the chat would bypass that control.

Three layers, deliberately redundant:

1. ``ground_truth`` is not in ``shared.tool_interfaces.ALLOWED_SQL_TABLES``,
   so ``run_sql`` has never been able to read it. That is the first wall.
2. :mod:`agents.sql_guard` also refuses these tables and columns **by name**,
   anywhere in the parsed query (a CTE, a subquery, an alias, a quoted
   identifier). If the allowlist were ever widened, or a published table
   ever gained a label column, a query naming one still fails validation.
3. Defence in depth on what comes *out*: the SQL scratch database is built
   without these columns, ``describe_tables`` never lists them, and
   :func:`scrub` removes them from any tool result before the model sees it.

What this cannot do is stop the model *inferring* a label from permitted
data -- ``scores`` and ``rings`` are the detector's own output and are fine
to discuss. It stops the labels themselves from being read.
"""

from __future__ import annotations

from typing import Any

#: Tables that hold labels the generator knows and the detector must not.
#: ``ring_id_map`` joins detected rings to the generator's true rings.
DENIED_TABLES: frozenset[str] = frozenset({"ground_truth", "ring_id_map"})

#: Column names that are labels wherever they appear. ``role`` is the
#: generator's mule/collector/clean role; ``true_ring_id`` and the others
#: come from ``ring_id_map`` and ``ground_truth``.
DENIED_COLUMNS: frozenset[str] = frozenset(
    {
        "is_mule",
        "is_hard_negative",
        "cohort",
        "pattern_type",
        "role",
        "ring_bank_count",
        "true_ring_id",
        "true_ring_size",
        "true_rings_touched",
        "overlap_members",
    }
)


def is_denied_column(name: object) -> bool:
    return str(name).strip().lower() in DENIED_COLUMNS


def scrub(value: Any) -> Any:
    """``value`` with every denied key removed from any dict inside it."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if not is_denied_column(k)}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub(item) for item in value)
    return value
