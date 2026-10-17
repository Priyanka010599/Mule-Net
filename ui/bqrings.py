"""The ring index, derived in BigQuery instead of read from the snapshot.

Active only when ``MULENET_READ_MODE=bigquery``. The snapshot stays the
default and the fallback: everything here can fail, and when it does
:mod:`ui.datasource` serves the file and says so on the page.

Why this exists
---------------
The BigQuery dataset held nine tables and nothing read them —
``ingest/`` wrote, ``infra/load_bigquery.py`` loaded, the UI served a
JSON file and the agents' ``run_sql`` was SQLite over parquet. This puts
``rings``, ``transfers``, ``account_evidence`` and ``cases`` on a read
path: one query aggregates 572,890 transfer rows server-side into the
per-ring counts the ring index shows.

What the SQL derives, and from which definition
-----------------------------------------------
Every field below is the SQL restatement of the expression
``ui/build_snapshot.py`` uses. There is one definition per field, not two:

``span``
    ``ARRAY_LENGTH(banks_involved)`` — ``build_snapshot`` uses
    ``len(ring.banks_involved)``.
``total_member_count`` / ``alertable_member_count`` / ``context_member_count``
    ``ARRAY_LENGTH(members)``, and counts over ``UNNEST(members)``
    filtered on the materialised ``alertable`` flag. ``alertable`` is
    read from the row and never re-derived, exactly as
    ``shared/bq_schemas.py`` requires and for the same reason
    ``build_snapshot`` copies it: re-deriving it means joining evidence
    and comparing banks, and getting that wrong publishes an alert
    against somebody's relative.
``intra_ring_transfer_count`` / ``total_intra_ring_amount`` / ``onset``
    a self-join of ``transfers`` against the ring's member list on
    *both* endpoints, which is ``build_snapshot``'s ``intra_ring``
    predicate (``from`` and ``to`` both members). Count, ``SUM(amount)``
    and ``MIN(timestamp)`` over that set.
``cross_bank_evidence_count``
    ``build_snapshot`` groups a ring's ``account_evidence`` rows by
    ``(kind, scope)`` and counts the groups whose scope is
    ``cross_bank``; ``kind`` collapses the reused-id-number and
    reused-photo types into one identity finding. The SQL counts the
    same distinct ``(kind)`` values within ``scope = 'cross_bank'``.
    Scope is read from the row, never inferred by comparing bank lists.
``has_case``
    ``LEFT JOIN cases USING (ring_id)``, non-null. A ring with no row is
    "no case yet", never an invented case.

What BigQuery cannot answer
---------------------------
Four ``RingSummary`` fields are **not derivable from this dataset** and
are taken from the snapshot even in ``bigquery`` mode. The payload names
them (:data:`SNAPSHOT_ONLY_FIELDS`) so nobody reads the whole row as
having come from BigQuery:

``cross_bank_transfer_count``
    needs the bank of each transfer endpoint. There is no ``accounts``
    table in the dataset — ``shared/bq_schemas.py`` defines nine tables
    and none of them maps a hashed account id to a bank — and the ids
    are HMAC-SHA256 digests, so the bank cannot be recovered from the
    id either. ``rings.banks_involved`` gives the ring's banks but not
    which bank either end of a given transfer sits in.
``true_positive_alertable_members``
    needs ``simulator/out/ground_truth.parquet``. Ground truth is
    deliberately not a BigQuery table; there is nothing to join to.
``true_ring_id`` / ``match_is_ambiguous``
    come from ``ml/out/ring_id_map.parquet``, which is not loaded to
    BigQuery.
``has_replay``
    comes from ``ml/out/ring_detection_timeline.parquet``, likewise not
    loaded.
``detector_version``
    is now derivable — ``build_snapshot._detector_version_of`` reads it
    off the ring's own ``ring_id``, and ``rings.ring_id`` is in the
    dataset — but the derivation is not reachable from the serving
    image. It lives in ``build_snapshot``, which imports pandas, and
    ``infra/Dockerfile.ui`` deliberately omits pandas (see
    :func:`_identity_types`); a per-row lazy import on the hot path, or
    a second copy of the rule here, are both worse than reading the
    field from the snapshot. Moving the helper to a pandas-free module
    would let this field join the derived list.

    (Until recently it was not derivable at all: ``build_snapshot`` took
    ``findings[0]["detector_version"]``, the *first* ``account_evidence``
    row for the ring in parquet row order, which a BigQuery table has no
    way to reproduce.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC
from decimal import Decimal, InvalidOperation
from typing import Any

from ui.config import UiConfig

#: :class:`ui.models.RingSummary` fields this query does derive. Reported
#: in the payload beside the ones it cannot; see the module docstring.
BIGQUERY_FIELDS: tuple[str, ...] = (
    "ring_id",
    "span",
    "banks_involved",
    "status",
    "detected_at",
    "total_member_count",
    "alertable_member_count",
    "context_member_count",
    "intra_ring_transfer_count",
    "total_intra_ring_amount",
    "onset",
    "cross_bank_evidence_count",
    "has_case",
)

#: Fields that stay on the snapshot even in ``bigquery`` mode: the
#: dataset holds nothing to derive them from, or -- for
#: ``detector_version`` -- the one definition of the derivation is not
#: importable here. Each reason is in the module docstring; the page
#: shows this list so a viewer cannot read a ring row as wholly
#: BigQuery-derived.
SNAPSHOT_ONLY_FIELDS: tuple[str, ...] = (
    "cross_bank_transfer_count",
    "true_positive_alertable_members",
    "detector_version",
    "true_ring_id",
    "match_is_ambiguous",
    "has_replay",
)

#: The statement, written out whole rather than assembled from fragments:
#: a query a reviewer cannot read in one piece is a query nobody checks.
QUERY_TEMPLATE = """\
-- MuleNet ring index, derived in BigQuery. One statement, four tables.
-- Restricted to the rings the snapshot also holds DETAIL for: the ring
-- detail endpoint still reads that detail from the snapshot, so serving
-- a ring whose detail pane cannot open would break the demo this is
-- meant to strengthen.
WITH ring AS (
  SELECT ring_id, banks_involved, detected_at, status, members
  FROM `{project}.{dataset}.rings`
  WHERE ring_id IN UNNEST(@ring_ids)
),
member_id AS (
  SELECT r.ring_id, m.hashed_account_id
  FROM ring AS r, UNNEST(r.members) AS m
),
-- build_snapshot's intra_ring predicate: BOTH endpoints members of the
-- SAME ring. `g.ring_id = f.ring_id` is what enforces "same ring";
-- without it a transfer between two rings' members counts in both.
intra AS (
  SELECT f.ring_id, t.amount, t.timestamp
  FROM `{project}.{dataset}.transfers` AS t
  JOIN member_id AS f ON f.hashed_account_id = t.from_hashed_account_id
  JOIN member_id AS g
    ON g.ring_id = f.ring_id
   AND g.hashed_account_id = t.to_hashed_account_id
),
intra_agg AS (
  SELECT ring_id,
         COUNT(*) AS intra_ring_transfer_count,
         -- NUMERIC out as a string: money crosses the wire as a fixed
         -- 2-decimal string and is never a float here or in the browser.
         CAST(SUM(amount) AS STRING) AS intra_ring_amount,
         MIN(timestamp) AS onset
  FROM intra
  GROUP BY ring_id
),
-- One finding per (kind, scope), not per row: reused id number and
-- reused photo are one fact, and counting them separately would show a
-- single identity match as two corroborations. Scope is read from the
-- column, never inferred by comparing bank lists.
evidence AS (
  SELECT m.ring_id,
         COUNT(DISTINCT IF(
           e.scope = 'cross_bank',
           IF(e.evidence_type IN UNNEST(@identity_types),
              'identity_reuse', 'shared_device'),
           NULL)) AS cross_bank_evidence_count
  FROM member_id AS m
  JOIN `{project}.{dataset}.account_evidence` AS e
    ON e.hashed_account_id = m.hashed_account_id
  GROUP BY m.ring_id
)
SELECT
  r.ring_id,
  ARRAY_LENGTH(r.banks_involved) AS span,
  r.banks_involved,
  r.status,
  r.detected_at,
  ARRAY_LENGTH(r.members) AS total_member_count,
  (SELECT COUNT(DISTINCT m.hashed_account_id)
     FROM UNNEST(r.members) AS m WHERE m.alertable) AS alertable_member_count,
  (SELECT COUNT(*)
     FROM UNNEST(r.members) AS m WHERE NOT m.alertable) AS context_member_count,
  IFNULL(i.intra_ring_transfer_count, 0) AS intra_ring_transfer_count,
  i.intra_ring_amount,
  i.onset,
  IFNULL(e.cross_bank_evidence_count, 0) AS cross_bank_evidence_count,
  c.ring_id IS NOT NULL AS has_case
FROM ring AS r
LEFT JOIN intra_agg AS i USING (ring_id)
LEFT JOIN evidence AS e USING (ring_id)
LEFT JOIN `{project}.{dataset}.cases` AS c ON c.ring_id = r.ring_id
"""

class BigQueryRingsError(RuntimeError):
    """The BigQuery ring list did not answer usably.

    Every path out of :func:`fetch_ring_rows` that is not a complete
    result raises this — auth, API error, timeout, zero rows. The caller
    serves the snapshot and labels the page; it never renders a partial
    BigQuery result as a whole one.
    """


@dataclass(frozen=True)
class RingQueryResult:
    """Rows plus the job facts the page is required to state."""

    #: ``ring_id`` -> the derived fields, keyed by ``RingSummary`` name.
    rows: dict[str, dict[str, Any]]
    bytes_processed: int
    bytes_billed: int
    cache_hit: bool
    latency_ms: float
    table_ref: str


def build_query(cfg: UiConfig) -> str:
    """The statement, with the dataset resolved. Exposed for tests."""
    return QUERY_TEMPLATE.format(project=cfg.bq_project, dataset=cfg.bq_dataset)


def _iso(value: Any) -> str:
    """UTC ISO-8601, matching ``build_snapshot._iso``'s output exactly."""
    return value.astimezone(UTC).isoformat()


def _money(raw: str | None) -> str:
    """``SUM(amount)`` as the fixed 2-decimal string the wire format uses."""
    if raw is None:
        return "0.00"
    try:
        return f"{Decimal(raw):.2f}"
    except InvalidOperation as exc:  # pragma: no cover - BigQuery NUMERIC
        raise BigQueryRingsError(f"unparseable SUM(amount) {raw!r}") from exc


def _row_to_fields(row: Any) -> dict[str, Any]:
    return {
        "ring_id": str(row["ring_id"]),
        "span": int(row["span"]),
        "banks_involved": list(row["banks_involved"]),
        "status": str(row["status"]),
        "detected_at": _iso(row["detected_at"]),
        "total_member_count": int(row["total_member_count"]),
        "alertable_member_count": int(row["alertable_member_count"]),
        "context_member_count": int(row["context_member_count"]),
        "intra_ring_transfer_count": int(row["intra_ring_transfer_count"]),
        "total_intra_ring_amount": _money(row["intra_ring_amount"]),
        "onset": _iso(row["onset"]) if row["onset"] is not None else None,
        "cross_bank_evidence_count": int(row["cross_bank_evidence_count"]),
        "has_case": bool(row["has_case"]),
    }


#: One client per project, reused. Building it resolves Application
#: Default Credentials, which measured ~1.5 s -- as much as the query
#: itself -- and paying that on every ring-index read would double a
#: latency regression that is already the cost of this feature. Built
#: lazily and inside :func:`fetch_ring_rows` so a machine with no
#: credentials fails where the caller already falls back, never at import.
_CLIENTS: dict[str, Any] = {}


def _identity_types() -> list[str]:
    """The evidence types that collapse into one ``identity_reuse`` finding.

    Imported from ``build_snapshot``, not restated: it owns that collapse
    rule and two copies of the frozenset would be two rules. Imported
    LAZILY because ``build_snapshot`` imports pandas, and
    ``infra/Dockerfile.ui`` deliberately does not install pandas or
    pyarrow in the serving image -- they are ~120 MB of build-time-only
    dependency. In ``snapshot`` mode this function is never called, so
    the default path still imports nothing extra; in ``bigquery`` mode on
    an image without them, the ImportError becomes an ordinary
    :class:`BigQueryRingsError` and the page falls back to the snapshot
    and says so, which is the right outcome for a missing dependency.
    """
    try:
        from ui.build_snapshot import _IDENTITY_TYPES
    except Exception as exc:
        raise BigQueryRingsError(
            f"cannot resolve the identity-reuse types: {type(exc).__name__}: {exc}"
        ) from exc
    return sorted(_IDENTITY_TYPES)


def fetch_ring_rows(
    cfg: UiConfig, ring_ids: list[str], client: Any | None = None
) -> RingQueryResult:
    """Run the ring-index query. Raises :class:`BigQueryRingsError` on anything else.

    ``client`` is injected by the tests; in production it comes from
    :data:`_CLIENTS`.
    """
    if not ring_ids:
        raise BigQueryRingsError("no ring ids to query")
    try:
        from google.cloud import bigquery
    except Exception as exc:  # pragma: no cover - declared dependency
        raise BigQueryRingsError(f"google-cloud-bigquery unavailable: {exc}") from exc

    table_ref = f"{cfg.bq_project}.{cfg.bq_dataset}"
    started = time.perf_counter()
    try:
        if client is None:
            if cfg.bq_project not in _CLIENTS:
                _CLIENTS[cfg.bq_project] = bigquery.Client(project=cfg.bq_project)
            client = _CLIENTS[cfg.bq_project]
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("ring_ids", "STRING", ring_ids),
                bigquery.ArrayQueryParameter(
                    "identity_types", "STRING", _identity_types()
                ),
            ]
        )
        job = client.query(build_query(cfg), job_config=job_config)
        rows = list(job.result(timeout=cfg.bq_timeout_s))
    except BigQueryRingsError:
        raise
    except Exception as exc:
        # Deliberately broad: auth, transport, API error, timeout and a
        # malformed reply all mean the same thing to the caller, which is
        # "serve the file and say so". Narrowing this would let one
        # google-api exception class escape and 500 the page.
        raise BigQueryRingsError(f"{type(exc).__name__}: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000.0

    if not rows:
        # Empty is a failure, not an empty ring list: the ring ids came
        # from a snapshot that has them, so zero rows means the dataset
        # is unloaded or the query is wrong.
        raise BigQueryRingsError(
            f"query returned no rows for {len(ring_ids)} ring ids"
        )
    return RingQueryResult(
        rows={str(r["ring_id"]): _row_to_fields(r) for r in rows},
        bytes_processed=int(job.total_bytes_processed or 0),
        bytes_billed=int(job.total_bytes_billed or 0),
        cache_hit=bool(job.cache_hit),
        latency_ms=round(latency_ms, 1),
        table_ref=table_ref,
    )
