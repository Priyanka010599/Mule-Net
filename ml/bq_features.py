"""The per-account graph features again, as BigQuery SQL over ``transfers``.

Why this exists
---------------
``ml/features.py`` is and stays **canonical**. Every published number in
``docs/claims.md``, ``ml/out/`` and the deck comes from the pandas path.
This module is a *second, independent implementation* of the same
definitions in SQL, run against the 572,890-row ``transfers`` table
already loaded in BigQuery, for one purpose: to show that the
definitions are portable and that the warehouse can compute them, and to
catch definitional drift by disagreeing loudly when it exists. It is not
a source of truth and nothing reads its output.

What it can and cannot reproduce
--------------------------------
BigQuery holds exactly one of the six simulator tables: ``transfers``
(see ``infra/load_bigquery.py``; ``SOURCES`` lists no ``accounts``,
``devices``, ``kyc_submissions`` or ``kyc_documents``, and
``shared/bq_schemas.TABLE_SCHEMAS`` defines no schema for them). So
twelve of the sixteen features per scope are reproduced here:

    in_degree, out_degree, unique_in_counterparties,
    unique_out_counterparties, unique_counterparties,
    distinct_counterparty_banks, amount_in_total, amount_out_total,
    passthrough_ratio, passthrough_speed_seconds, round_amount_ratio,
    account_age_days

and four are **not reproducible in SQL today**:

    device_count, shared_device_count, document_reuse_count,
    document_reuse_flag

because they need ``devices`` and ``kyc_documents``/``kyc_submissions``,
which are not in BigQuery and cannot be added without editing
``shared/bq_schemas.py``.

The account dimension — ``hashed_account_id``, ``bank_id``,
``opened_at`` and the per-account ``as_of`` cutoff — is supplied as
query **parameters** (three parallel arrays over 30,000 accounts), not
as a table. That keeps the promise of "no new BigQuery table" while
still letting the SQL do the per-account, point-in-time work. It is also
honest about what BigQuery contributes: the 101 MB scan and all the
aggregation, over a 3 MB dimension pushed down with the query.

Point-in-time correctness
-------------------------
The cutoff rules are transcribed from ``ml.features``, not re-derived:

* a transfer enters an account's row only when
  ``timestamp <= that account's as_of`` (``_scope_features``);
* ``account_age_days`` is measured against ``as_of``, not the end of the
  window (``_scope_features``);
* the burst / pass-through computation runs on the **already filtered**
  event set, exactly as ``_passthrough_speed`` is called after the
  filter in ``_scope_features``.

``no_post_cutoff_leak`` in ``ml/test_bq_features.py`` is the test that
matters: it re-runs the same SQL against a ``transfers`` source in which
every post-cutoff row is deleted, and again against one in which every
post-cutoff row is corrupted and duplicated, and requires the feature
rows to be byte-identical. A leak here would not show up as a mismatch
against pandas — it would show up as a *better* feature — so equality
against pandas is not sufficient evidence on its own.

Exactness
---------
Integer counts must match exactly; there is no tolerance for them.

Money is where the two implementations genuinely differ, and the
difference is in pandas' favour of speed rather than accuracy:
``ml.features._edge_events`` does ``transfers["amount"].astype("float64")``,
so the pandas totals are **float64 sums of Decimal cents**, while
``SUM(amount)`` over a BigQuery ``NUMERIC`` column is exact. This module
therefore emits both: ``amount_in_total`` (exact, NUMERIC) and
``amount_in_total_f64`` (``SUM(CAST(amount AS FLOAT64))``, the pandas
analogue), so the comparison can attribute any disagreement to the cast
rather than to a definitional difference.

``account_age_days`` reproduces pandas' arithmetic rather than an
idealised version of it: ``Timedelta.total_seconds()`` is ``int64``
nanoseconds divided by ``1e9``, and at five years the nanosecond count
(~1.6e17) exceeds 2**53, so the int-to-float conversion rounds. Dividing
integer *seconds* by 86400 instead would be more accurate and would
differ from pandas in the last few ulps. The SQL takes the nanosecond
route on purpose, to be bit-identical to the canonical path.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd
from google.cloud import bigquery

from ml.config import MlConfig
from ml.features import (
    FEATURE_NAMES,
    NO_PASSTHROUGH,
    SCOPES,
    build_features,
    full_window_cutoff,
    load_tables,
)

#: Decimal zero, reused so ``exact_decimal_totals`` allocates one.
_DECIMAL_ZERO = Decimal(0)

#: Features this module reproduces in SQL, in ``FEATURE_NAMES`` order.
SQL_FEATURE_NAMES: tuple[str, ...] = tuple(
    name
    for name in FEATURE_NAMES
    if name
    not in {
        "device_count",
        "shared_device_count",
        "document_reuse_count",
        "document_reuse_flag",
    }
)

#: Features that need a table BigQuery does not have. Kept explicit so a
#: future ``devices`` load makes this list shrink rather than silently
#: leaving a gap in the comparison.
UNREPRODUCIBLE: dict[str, str] = {
    "device_count": "needs simulator table `devices` (not in BigQuery)",
    "shared_device_count": "needs simulator table `devices` (not in BigQuery)",
    "document_reuse_count": "needs `kyc_documents` + `kyc_submissions` (not in BigQuery)",
    "document_reuse_flag": "needs `kyc_documents` + `kyc_submissions` (not in BigQuery)",
}

#: Diagnostic columns that exist only to attribute money disagreements.
#: ``*_f64`` mirrors pandas' float64 cast; the unsuffixed column is the
#: exact NUMERIC sum the BigQuery schema actually asks for.
DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "amount_in_total_f64",
    "amount_out_total_f64",
    "passthrough_ratio_f64",
)

#: Integer-valued outputs. No tolerance is ever acceptable on these.
INTEGER_FEATURES: frozenset[str] = frozenset(
    {
        "in_degree",
        "out_degree",
        "unique_in_counterparties",
        "unique_out_counterparties",
        "unique_counterparties",
        "distinct_counterparty_banks",
    }
)

#: Exact-decimal money outputs. No tolerance is acceptable against the
#: NUMERIC sum either; a disagreement means a float crept in.
MONEY_FEATURES: frozenset[str] = frozenset({"amount_in_total", "amount_out_total"})

#: Smallest difference in each feature that could change a downstream
#: decision, so a proposed tolerance can be checked against something
#: rather than eyeballed. Sources: ``ml.rule.VELOCITY_MAX_SECONDS``
#: (3600 s, and speeds are whole seconds, so 1.0 s is the resolution),
#: ``ml.rule.VELOCITY_RATIO_RANGE``; ratios move in steps of one cent
#: over the inflow, and the largest inflow in the dataset is ~1e7, so
#: 1e-9 is a floor on a meaningful ratio step.
DECISION_RESOLUTION: dict[str, float] = {
    "amount_in_total": 0.01,
    "amount_out_total": 0.01,
    "passthrough_ratio": 1e-9,
    "passthrough_speed_seconds": 1.0,
    "round_amount_ratio": 1e-9,
    "account_age_days": 1.0 / 86400.0,
}

_QUERY = """
-- MuleNet per-account graph features, scope = {scope}
-- Second implementation of ml/features.py. pandas stays canonical.
WITH
dim AS (
  SELECT
    account_id AS hashed_account_id,
    @bank_ids[OFFSET(i)]  AS bank_id,
    @opened_ats[OFFSET(i)] AS opened_at,
    @as_ofs[OFFSET(i)]     AS as_of
  FROM UNNEST(@account_ids) AS account_id WITH OFFSET i
),
-- transfers, each endpoint's bank attached. LEFT JOIN, not INNER: an
-- endpoint outside the account table must still produce an event for
-- the endpoint that is inside it, with a NULL counterparty bank --
-- which is what `.map(bank)` -> NaN does in pandas.
edges AS (
  SELECT
    t.transfer_id,
    t.from_hashed_account_id,
    t.to_hashed_account_id,
    t.amount,
    t.timestamp,
    f.bank_id AS from_bank,
    d.bank_id AS to_bank
  FROM ({source}) AS t
  LEFT JOIN dim AS f ON f.hashed_account_id = t.from_hashed_account_id
  LEFT JOIN dim AS d ON d.hashed_account_id = t.to_hashed_account_id
),
scoped AS (
  SELECT * FROM edges
  {scope_filter}
),
-- one row per (account, direction), mirroring _edge_events.
events AS (
  SELECT
    from_hashed_account_id AS account,
    to_hashed_account_id   AS counterparty,
    to_bank                AS counterparty_bank,
    amount, timestamp, transfer_id,
    FALSE AS is_inbound
  FROM scoped
  UNION ALL
  SELECT
    to_hashed_account_id,
    from_hashed_account_id,
    from_bank,
    amount, timestamp, transfer_id,
    TRUE
  FROM scoped
),
-- POINT IN TIME: an account never sees a transfer after its own cutoff.
-- INNER JOIN on dim, so an account with no cutoff has no events, as in
-- pandas where NaN <= NaN is False.
visible AS (
  SELECT e.*, UNIX_SECONDS(e.timestamp) AS ts
  FROM events AS e
  JOIN dim AS d ON d.hashed_account_id = e.account
  WHERE e.timestamp <= d.as_of
),
agg AS (
  SELECT
    account,
    COUNTIF(is_inbound)      AS in_degree,
    COUNTIF(NOT is_inbound)  AS out_degree,
    COUNT(DISTINCT IF(is_inbound, counterparty, NULL))     AS unique_in_counterparties,
    COUNT(DISTINCT IF(NOT is_inbound, counterparty, NULL)) AS unique_out_counterparties,
    COUNT(DISTINCT counterparty)      AS unique_counterparties,
    COUNT(DISTINCT counterparty_bank) AS distinct_counterparty_banks,
    SUM(IF(is_inbound, amount, NUMERIC '0'))     AS amount_in_total,
    SUM(IF(NOT is_inbound, amount, NUMERIC '0')) AS amount_out_total,
    SUM(IF(is_inbound, CAST(amount AS FLOAT64), 0.0))     AS amount_in_total_f64,
    SUM(IF(NOT is_inbound, CAST(amount AS FLOAT64), 0.0)) AS amount_out_total_f64,
    COUNTIF(MOD(amount, NUMERIC '100') = NUMERIC '0')     AS round_count
  FROM visible
  GROUP BY account
),
-- bursts: a maximal run of an account's own events (both directions)
-- where consecutive events are at most @gap_seconds apart. The extra
-- ORDER BY terms only make ties deterministic; they cannot move a burst
-- boundary, because tied rows have a zero gap either way.
flagged AS (
  SELECT
    account, is_inbound, ts,
    IF(
      LAG(ts) OVER w IS NULL OR ts - LAG(ts) OVER w > @gap_seconds,
      1, 0
    ) AS is_new_burst,
    transfer_id
  FROM visible
  WINDOW w AS (PARTITION BY account ORDER BY ts, is_inbound, transfer_id)
),
bursted AS (
  SELECT
    account, is_inbound, ts,
    SUM(is_new_burst) OVER (
      PARTITION BY account ORDER BY ts, is_inbound, transfer_id
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS burst
  FROM flagged
),
burst_first_in AS (
  SELECT account, burst, MIN(IF(is_inbound, ts, NULL)) AS first_in
  FROM bursted
  GROUP BY account, burst
),
-- per burst: the first debit at or after that burst's first credit.
hops AS (
  SELECT
    b.account,
    CAST(MIN(b.ts) - f.first_in AS FLOAT64) AS delta
  FROM bursted AS b
  JOIN burst_first_in AS f USING (account, burst)
  WHERE NOT b.is_inbound AND f.first_in IS NOT NULL AND b.ts >= f.first_in
  GROUP BY b.account, b.burst, f.first_in
),
-- pandas median: mean of the two middle values for even counts, which
-- is exactly PERCENTILE_CONT at 0.5.
speed AS (
  SELECT DISTINCT
    account,
    PERCENTILE_CONT(delta, 0.5) OVER (PARTITION BY account) AS passthrough_speed_seconds
  FROM hops
)
SELECT
  d.hashed_account_id,
  d.as_of,
  '{scope}' AS scope,
  IFNULL(a.in_degree, 0)                  AS in_degree,
  IFNULL(a.out_degree, 0)                 AS out_degree,
  IFNULL(a.unique_in_counterparties, 0)   AS unique_in_counterparties,
  IFNULL(a.unique_out_counterparties, 0)  AS unique_out_counterparties,
  IFNULL(a.unique_counterparties, 0)      AS unique_counterparties,
  IFNULL(a.distinct_counterparty_banks, 0) AS distinct_counterparty_banks,
  IFNULL(a.amount_in_total, NUMERIC '0')  AS amount_in_total,
  IFNULL(a.amount_out_total, NUMERIC '0') AS amount_out_total,
  IFNULL(a.amount_in_total_f64, 0.0)      AS amount_in_total_f64,
  IFNULL(a.amount_out_total_f64, 0.0)     AS amount_out_total_f64,
  -- zero inbound => ratio undefined => 0.0 (ml/features.py docstring)
  IF(
    IFNULL(a.amount_in_total_f64, 0.0) > 0.0,
    CAST(a.amount_out_total AS FLOAT64) / CAST(a.amount_in_total AS FLOAT64),
    0.0
  ) AS passthrough_ratio,
  IF(
    IFNULL(a.amount_in_total_f64, 0.0) > 0.0,
    a.amount_out_total_f64 / a.amount_in_total_f64,
    0.0
  ) AS passthrough_ratio_f64,
  IFNULL(s.passthrough_speed_seconds, {no_passthrough}) AS passthrough_speed_seconds,
  IF(
    IFNULL(a.in_degree, 0) + IFNULL(a.out_degree, 0) > 0,
    CAST(a.round_count AS FLOAT64)
      / CAST(a.in_degree + a.out_degree AS FLOAT64),
    0.0
  ) AS round_amount_ratio,
  -- pandas: Timedelta.total_seconds() is int64 ns / 1e9, then / 86400.
  -- Reproduced, rounding included, rather than improved on.
  (CAST(TIMESTAMP_DIFF(d.as_of, d.opened_at, MICROSECOND) * 1000 AS FLOAT64) / 1e9)
    / 86400.0 AS account_age_days
FROM dim AS d
LEFT JOIN agg   AS a ON a.account = d.hashed_account_id
LEFT JOIN speed AS s ON s.account = d.hashed_account_id
ORDER BY d.hashed_account_id
"""

#: ``{scope_filter}`` per scope. single_bank cannot observe a transfer
#: with an endpoint at another bank at all, so those rows do not exist
#: for it. NULL banks compare false, matching NaN == NaN in pandas.
_SCOPE_FILTERS: dict[str, str] = {
    "single_bank": "WHERE from_bank = to_bank",
    "cross_bank": "",
}

#: Default ``{source}``: the loaded table, unmodified.
TRANSFERS_SOURCE = "SELECT * FROM `{project}.{dataset}.transfers`"

#: ``{source}`` variants for the leakage check. Both reference
#: ``@as_ofs`` rather than taking a new parameter, so the probe instant
#: is by construction "the latest cutoff any account has" and there is no
#: literal to interpolate.
#:
#: ``DROPPED`` deletes every post-cutoff row. ``CORRUPTED`` keeps the row
#: count but rewrites every post-cutoff row — amount x7, direction
#: reversed — *and* adds an unmodified duplicate of it, so a query that
#: peeks past a cutoff changes its answer in the amounts, the degrees and
#: the counterparty sets at once. Either variant must leave every feature
#: row untouched.
_PROBE = "(SELECT MAX(x) FROM UNNEST(@as_ofs) AS x)"

DROPPED_POST_CUTOFF_SOURCE = (
    "SELECT * FROM `{project}.{dataset}.transfers` WHERE timestamp <= " + _PROBE
)

CORRUPTED_POST_CUTOFF_SOURCE = (
    """
SELECT * FROM `{project}.{dataset}.transfers` WHERE timestamp <= """
    + _PROBE
    + """
UNION ALL
SELECT * REPLACE(
  CONCAT(transfer_id, '-flipped') AS transfer_id,
  amount * NUMERIC '7' AS amount,
  to_hashed_account_id AS from_hashed_account_id,
  from_hashed_account_id AS to_hashed_account_id)
FROM `{project}.{dataset}.transfers` WHERE timestamp > """
    + _PROBE
    + """
UNION ALL
SELECT * REPLACE(CONCAT(transfer_id, '-dup') AS transfer_id)
FROM `{project}.{dataset}.transfers` WHERE timestamp > """
    + _PROBE
)


def exact_decimal_totals(
    transfers: pd.DataFrame,
    cutoff: pd.Series,
    bank: pd.Series | None = None,
    same_bank_only: bool = False,
) -> pd.DataFrame:
    """Inbound/outbound totals with ``Decimal`` arithmetic end to end.

    A deliberately slow third implementation whose only job is to settle
    which of the other two is right about money. It never touches
    ``float`` — not in the sum, not in the comparison — so its totals are
    the arithmetic truth for a NUMERIC(15,2) column, and whichever of
    pandas and BigQuery disagrees with it is the one that lost precision.
    """
    if same_bank_only:
        if bank is None:
            raise ValueError("same_bank_only needs the account -> bank map")
        keep = transfers["from_hashed_account_id"].map(bank) == transfers[
            "to_hashed_account_id"
        ].map(bank)
        transfers = transfers.loc[keep]

    inbound: dict[str, Any] = {}
    outbound: dict[str, Any] = {}
    # Instants compared as nanoseconds since the epoch: same ordering as
    # the tz-aware comparison in ml.features, and no NaT/tz edge cases.
    ns = "datetime64[ns, UTC]"
    cutoffs = cutoff.dropna().astype(ns).astype("int64").to_dict()
    zero = _DECIMAL_ZERO
    for src, dst, amount, when in zip(
        transfers["from_hashed_account_id"].to_numpy(),
        transfers["to_hashed_account_id"].to_numpy(),
        transfers["amount"].to_numpy(),
        transfers["timestamp"].astype(ns).astype("int64").to_numpy(),
        strict=True,
    ):
        src_cut = cutoffs.get(src)
        if src_cut is not None and when <= src_cut:
            outbound[src] = outbound.get(src, zero) + amount
        dst_cut = cutoffs.get(dst)
        if dst_cut is not None and when <= dst_cut:
            inbound[dst] = inbound.get(dst, zero) + amount

    index = pd.Index(cutoff.index, name="hashed_account_id")
    return pd.DataFrame(
        {
            "amount_in_total": [inbound.get(a, zero) for a in index],
            "amount_out_total": [outbound.get(a, zero) for a in index],
        },
        index=index,
    )


def build_query(scope: str, source: str, no_passthrough: float) -> str:
    """Render the feature SQL for one visibility scope."""
    if scope not in _SCOPE_FILTERS:
        raise KeyError(f"unknown scope {scope!r}; known: {sorted(_SCOPE_FILTERS)}")
    return _QUERY.format(
        scope=scope,
        scope_filter=_SCOPE_FILTERS[scope],
        source=source,
        no_passthrough=repr(float(no_passthrough)),
    )


@dataclass
class QueryRun:
    """One executed query, with the cost evidence the report needs."""

    scope: str
    frame: pd.DataFrame
    bytes_processed: int
    bytes_billed: int
    cache_hit: bool
    seconds: float
    slot_millis: int
    job_seconds: float
    job_id: str

    @property
    def usd(self) -> float:
        """On-demand cost at $6.25/TiB of *billed* bytes."""
        return self.bytes_billed / 2**40 * 6.25

    def summary(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "job_id": self.job_id,
            "rows": len(self.frame),
            "bytes_processed": self.bytes_processed,
            "bytes_billed": self.bytes_billed,
            "mib_processed": round(self.bytes_processed / 2**20, 3),
            "cache_hit": self.cache_hit,
            "seconds": round(self.seconds, 3),
            "job_seconds": round(self.job_seconds, 3),
            "slot_millis": self.slot_millis,
            "usd": round(self.usd, 8),
        }


def account_dimension(
    tables: dict[str, pd.DataFrame], cutoff: pd.Series
) -> tuple[list[str], list[str], list[pd.Timestamp], list[pd.Timestamp]]:
    """The four parallel arrays the query takes as parameters."""
    accounts = tables["accounts"]
    aligned = cutoff.reindex(accounts["hashed_account_id"])
    return (
        accounts["hashed_account_id"].tolist(),
        accounts["bank_id"].tolist(),
        accounts["opened_at"].tolist(),
        list(aligned.to_numpy()),
    )


def run_scope(
    client: bigquery.Client,
    cfg: MlConfig,
    scope: str,
    dimension: tuple[list[str], list[str], list[Any], list[Any]],
    source: str | None = None,
    use_cache: bool = False,
) -> QueryRun:
    """Execute the feature SQL for one scope and return it with its cost."""
    account_ids, bank_ids, opened_ats, as_ofs = dimension
    sql = build_query(
        scope=scope,
        source=source or TRANSFERS_SOURCE.format(project=cfg.bq_project, dataset=cfg.bq_dataset),
        no_passthrough=NO_PASSTHROUGH,
    )
    config = bigquery.QueryJobConfig(
        use_query_cache=use_cache,
        query_parameters=[
            bigquery.ArrayQueryParameter("account_ids", "STRING", account_ids),
            bigquery.ArrayQueryParameter("bank_ids", "STRING", bank_ids),
            bigquery.ArrayQueryParameter("opened_ats", "TIMESTAMP", opened_ats),
            bigquery.ArrayQueryParameter("as_ofs", "TIMESTAMP", as_ofs),
            bigquery.ScalarQueryParameter("gap_seconds", "INT64", cfg.burst_gap_seconds),
        ],
    )
    started = time.perf_counter()
    job = client.query(sql, job_config=config)
    # Rows are materialised by hand rather than via ``to_dataframe()``:
    # that path needs the ``db-dtypes`` package, and it would map NUMERIC
    # onto a pandas extension dtype. Plain ``Decimal`` objects are what
    # the exactness check wants, and 30,000 rows is nothing.
    result = job.result()
    columns = [f.name for f in result.schema]
    frame = pd.DataFrame([dict(row) for row in result], columns=columns)
    elapsed = time.perf_counter() - started
    # `seconds` is wall clock including the 3 MB dimension upload and the
    # row download; `job_seconds` is BigQuery's own start-to-end, which is
    # the figure to quote for the query itself.
    job.reload()
    job_seconds = (job.ended - job.started).total_seconds() if job.ended and job.started else 0.0
    return QueryRun(
        scope=scope,
        frame=frame,
        bytes_processed=job.total_bytes_processed or 0,
        bytes_billed=job.total_bytes_billed or 0,
        cache_hit=bool(job.cache_hit),
        seconds=elapsed,
        slot_millis=job.slot_millis or 0,
        job_seconds=job_seconds,
        job_id=job.job_id,
    )


def sql_features(
    client: bigquery.Client,
    cfg: MlConfig,
    tables: dict[str, pd.DataFrame],
    cutoff: pd.Series,
    source: str | None = None,
) -> tuple[pd.DataFrame, list[QueryRun]]:
    """Both scopes, joined into one ``sb_``/``cb_`` row per account.

    Shaped exactly like ``ml.features.build_features`` output, minus the
    four columns listed in :data:`UNREPRODUCIBLE`.
    """
    dimension = account_dimension(tables, cutoff)
    runs = [run_scope(client, cfg, scope, dimension, source=source) for scope in SCOPES]
    blocks = []
    for run, prefix in zip(runs, SCOPES.values(), strict=True):
        block = run.frame.set_index("hashed_account_id").drop(columns=["scope", "as_of"])
        blocks.append(block.add_prefix(prefix))
    joined = pd.concat(blocks, axis=1).reset_index()
    # Both scopes are driven by the same `dim`, so either carries as_of.
    as_of = runs[0].frame.set_index("hashed_account_id")["as_of"]
    joined.insert(1, "as_of", as_of.reindex(joined["hashed_account_id"]).to_numpy())
    return joined, runs


@dataclass
class FeatureDelta:
    """Per-feature comparison result for one scope-prefixed column."""

    column: str
    dtype: str
    max_abs_delta: float
    mismatched_rows: int
    decision_resolution: float | None
    worst_account: str | None = None
    pandas_value: float | None = None
    sql_value: float | None = None

    @property
    def exact(self) -> bool:
        return self.max_abs_delta == 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "dtype": self.dtype,
            "max_abs_delta": self.max_abs_delta,
            "mismatched_rows": self.mismatched_rows,
            "exact": self.exact,
            "decision_resolution": self.decision_resolution,
            "worst_account": self.worst_account,
            "pandas_value": self.pandas_value,
            "sql_value": self.sql_value,
        }


def _feature_dtype(name: str) -> str:
    if name in INTEGER_FEATURES:
        return "int64"
    if name in MONEY_FEATURES:
        return "NUMERIC/Decimal"
    return "float64"


def compare(pandas_features: pd.DataFrame, sql: pd.DataFrame) -> list[FeatureDelta]:
    """Per-column max |delta| between the canonical and SQL tables.

    Both frames are keyed on ``hashed_account_id``. Money columns are
    compared as ``Decimal`` promoted to ``float64`` only for the delta
    figure; ``mismatched_rows`` is counted on exact equality, so a
    non-zero count with a tiny delta is still a mismatch and is reported
    as one.
    """
    left = pandas_features.set_index("hashed_account_id").sort_index()
    right = sql.set_index("hashed_account_id").sort_index()
    if not left.index.equals(right.index):
        raise ValueError(
            f"account sets differ: pandas={len(left)} sql={len(right)} "
            f"common={len(left.index.intersection(right.index))}"
        )

    deltas: list[FeatureDelta] = []
    for prefix in SCOPES.values():
        for name in SQL_FEATURE_NAMES:
            column = f"{prefix}{name}"
            want = pd.to_numeric(left[column], errors="raise").astype("float64")
            got = pd.to_numeric(right[column], errors="raise").astype("float64")
            diff = (got - want).abs()
            worst = diff.idxmax() if len(diff) else None
            deltas.append(
                FeatureDelta(
                    column=column,
                    dtype=_feature_dtype(name),
                    max_abs_delta=float(diff.max()) if len(diff) else 0.0,
                    mismatched_rows=int((diff != 0.0).sum()),
                    decision_resolution=DECISION_RESOLUTION.get(name),
                    worst_account=str(worst) if worst is not None else None,
                    pandas_value=float(want.loc[worst]) if worst is not None else None,
                    sql_value=float(got.loc[worst]) if worst is not None else None,
                )
            )
    return deltas


def staggered_cutoff(
    tables: dict[str, pd.DataFrame], steps: int = 24, latest: float = 1.0
) -> pd.Series:
    """Per-account cutoffs spread across the transfer window.

    The full-window cutoff makes the point-in-time filter a no-op, so an
    equality check at full window proves nothing about the cutoff logic.
    This spreads accounts deterministically over ``steps`` cutoffs
    between the first and last transfer, which exercises it.

    ``latest`` scales the whole ladder: at ``0.5`` no account's cutoff is
    past the middle of the window, so roughly half the transfers are
    strictly post-cutoff for *every* account. That is what the leakage
    check needs, because with ``latest=1.0`` some account's cutoff is the
    last transfer and "drop everything after the cutoff" drops nothing.
    """
    timestamps = tables["transfers"]["timestamp"]
    start, end = timestamps.min(), timestamps.max()
    accounts = pd.Index(tables["accounts"]["hashed_account_id"])
    offsets = (pd.RangeIndex(len(accounts)) % steps + 1) / steps * latest
    return pd.Series(start + (end - start) * offsets.to_numpy(), index=accounts)


def _comparable(frame: pd.DataFrame) -> pd.DataFrame:
    """The part of a SQL feature frame a leakage check may not change."""
    return frame.drop(columns=["as_of"]).set_index("hashed_account_id").sort_index()


def leakage_check(
    client: bigquery.Client,
    cfg: MlConfig,
    tables: dict[str, pd.DataFrame],
    latest: float = 0.5,
) -> dict[str, Any]:
    """Prove the SQL features carry no post-cutoff information.

    Equality against pandas cannot prove this: a leak makes a feature
    *better*, not different, and both implementations could leak the same
    way. So this varies the data instead of the implementation.

    With every account's cutoff at or before the midpoint of the window,
    the same query is run three times:

    ``baseline``   the real ``transfers`` table;
    ``dropped``    every post-cutoff row deleted;
    ``corrupted``  every post-cutoff row rewritten (amount x7, direction
                   reversed) *and* duplicated.

    All three must produce identical feature rows. A ``control`` run then
    moves the cutoffs to the end of the window and must produce
    *different* rows — without that, the two equalities above would also
    hold for a query that never read ``transfers`` at all.
    """
    mid = staggered_cutoff(tables, latest=latest)
    dimension = account_dimension(tables, mid)
    fmt = {"project": cfg.bq_project, "dataset": cfg.bq_dataset}

    variants = {
        "baseline": TRANSFERS_SOURCE,
        "dropped": DROPPED_POST_CUTOFF_SOURCE,
        "corrupted": CORRUPTED_POST_CUTOFF_SOURCE,
    }
    frames: dict[str, pd.DataFrame] = {}
    runs: list[QueryRun] = []
    for name, template in variants.items():
        run = run_scope(client, cfg, "cross_bank", dimension, source=template.format(**fmt))
        runs.append(run)
        frames[name] = _comparable(run.frame)

    control_run = run_scope(
        client,
        cfg,
        "cross_bank",
        account_dimension(tables, full_window_cutoff(tables)),
        source=TRANSFERS_SOURCE.format(**fmt),
    )
    runs.append(control_run)

    post_cutoff_rows = int((tables["transfers"]["timestamp"] > mid.max()).sum())
    changed = int((_comparable(control_run.frame) != frames["baseline"]).any(axis=1).sum())
    return {
        "probe_cutoff": str(mid.max()),
        "post_cutoff_transfers": post_cutoff_rows,
        "dropped_matches_baseline": frames["dropped"].equals(frames["baseline"]),
        "corrupted_matches_baseline": frames["corrupted"].equals(frames["baseline"]),
        "control_accounts_changed": changed,
        "control_is_live": changed > 0,
        "verdict": (
            frames["dropped"].equals(frames["baseline"])
            and frames["corrupted"].equals(frames["baseline"])
            and changed > 0
            and post_cutoff_rows > 0
        ),
        "queries": [r.summary() for r in runs],
        "total_usd": round(sum(r.usd for r in runs), 8),
    }


@dataclass
class ParityReport:
    """Everything the write-up needs, in one serialisable object."""

    scenario: str
    accounts: int
    deltas: list[FeatureDelta]
    runs: list[QueryRun] = field(default_factory=list)
    money: dict[str, Any] = field(default_factory=dict)

    @property
    def all_exact(self) -> bool:
        return all(d.exact for d in self.deltas)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "accounts": self.accounts,
            "all_exact": self.all_exact,
            "reproduced_features": list(SQL_FEATURE_NAMES),
            "unreproducible_features": UNREPRODUCIBLE,
            "canonical": "ml/features.py (pandas). This SQL is a second implementation.",
            "queries": [r.summary() for r in self.runs],
            "total_usd": round(sum(r.usd for r in self.runs), 8),
            "features": [d.to_dict() for d in self.deltas],
            "money_attribution": self.money,
        }


def money_attribution(
    pandas_features: pd.DataFrame,
    sql: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    cutoff: pd.Series,
) -> dict[str, Any]:
    """Say which side of a money disagreement is the inexact one.

    NUMERIC and ``Decimal`` are both exact, so a non-zero delta on a
    money column means a float got involved somewhere. This settles where
    by adding a third, all-``Decimal`` aggregation
    (:func:`exact_decimal_totals`) and checking both implementations
    against it:

    * BigQuery's ``SUM(amount)`` over NUMERIC is compared as ``Decimal``
      to ``Decimal``, so ``sql_mismatched_rows`` is an exact count;
    * the pandas column is already ``float64`` by the time it is written
      (``ml.features._edge_events`` casts before summing), so it can only
      be compared after conversion, and its error is reported as a max
      absolute delta in dollars.

    The finding this is designed to make unambiguous, rather than to
    paper over: the canonical path is the inexact one, and it is inexact
    at the cast, not in the summation order. ``ml/features.py`` is left
    alone — pandas stays canonical, and changing it to match the SQL
    would move a published number to chase a second implementation.
    """
    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]
    aligned = cutoff.reindex(tables["accounts"]["hashed_account_id"])
    left = pandas_features.set_index("hashed_account_id").sort_index()
    right = sql.set_index("hashed_account_id").sort_index()

    out: dict[str, Any] = {}
    for scope, prefix in SCOPES.items():
        exact = exact_decimal_totals(
            tables["transfers"], aligned, bank=bank, same_bank_only=(scope == "single_bank")
        ).sort_index()
        for name in sorted(MONEY_FEATURES):
            column = f"{prefix}{name}"
            truth = exact[name]
            sql_col = right[column]
            sql_bad = int((sql_col.to_numpy() != truth.to_numpy()).sum())
            pandas_delta = (
                left[column].astype("float64") - truth.map(float).astype("float64")
            ).abs()
            out[column] = {
                "sql_numeric_vs_exact_decimal_mismatched_rows": sql_bad,
                "sql_is_exact": sql_bad == 0,
                "pandas_float64_vs_exact_decimal_mismatched_rows": int((pandas_delta != 0.0).sum()),
                "pandas_max_abs_error_usd": float(pandas_delta.max()),
                "cause": (
                    "ml.features._edge_events casts Decimal NUMERIC(15,2) to float64 "
                    "before summing; the error is in the per-value cast, not the "
                    "summation order"
                ),
            }
    return out


def run_parity(
    client: bigquery.Client,
    cfg: MlConfig,
    tables: dict[str, pd.DataFrame],
    scenario: str,
    cutoff: pd.Series,
) -> ParityReport:
    """Build both implementations under one cutoff and compare them."""
    canonical = build_features(cfg, tables=tables, cutoff=cutoff)
    sql, runs = sql_features(client, cfg, tables, cutoff)
    return ParityReport(
        scenario=scenario,
        accounts=len(sql),
        deltas=compare(canonical, sql),
        money=money_attribution(canonical, sql, tables, cutoff),
        runs=runs,
    )


def format_table(report: ParityReport) -> str:
    """The per-feature exactness table, as text."""
    header = f"{'column':34s}{'dtype':17s}{'max|delta|':>14s}{'rows':>7s}  verdict"
    lines = [header, "-" * len(header)]
    for d in report.deltas:
        attribution = report.money.get(d.column, {})
        if d.exact:
            verdict = "EXACT"
        elif attribution.get("sql_is_exact"):
            # The SQL side was checked against an all-Decimal reference and
            # matched every row, so this is not a tolerance: it is pandas'
            # float64 cast, measured.
            verdict = f"SQL EXACT; pandas float64 cast off by <= ${d.max_abs_delta:.2e}"
        elif attribution:
            verdict = "MISMATCH: SQL disagrees with exact Decimal"
        elif d.decision_resolution is None:
            verdict = "MISMATCH"
        elif d.max_abs_delta < d.decision_resolution:
            verdict = f"tol {d.max_abs_delta:.3e} < {d.decision_resolution:.1e} decision step"
        else:
            verdict = f"MISMATCH exceeds {d.decision_resolution:.1e} decision step"
        lines.append(
            f"{d.column:34s}{d.dtype:17s}{d.max_abs_delta:>14.6e}{d.mismatched_rows:>7d}  {verdict}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Compare the SQL and pandas features, sample first then all accounts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("full_window", "staggered", "both"),
        default="both",
        help="full_window reproduces features.parquet; staggered exercises the cutoff",
    )
    parser.add_argument(
        "--leakage",
        action="store_true",
        help="run the point-in-time leakage check instead of the parity comparison",
    )
    args = parser.parse_args(argv)

    cfg = MlConfig.from_env()
    tables = load_tables(cfg)
    client = bigquery.Client(project=cfg.bq_project)

    scenarios = {
        "full_window": lambda: full_window_cutoff(tables),
        "staggered": lambda: staggered_cutoff(tables),
    }
    chosen = list(scenarios) if args.scenario == "both" else [args.scenario]

    out_dir = cfg.experiments_dir / "bq_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.leakage:
        result = leakage_check(client, cfg, tables)
        path = out_dir / "leakage.json"
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        for key in (
            "probe_cutoff",
            "post_cutoff_transfers",
            "dropped_matches_baseline",
            "corrupted_matches_baseline",
            "control_accounts_changed",
            "verdict",
        ):
            print(f"  {key:28s} {result[key]}")
        print(f"\nwrote {path}")
        return

    reports = []
    for name in chosen:
        report = run_parity(client, cfg, tables, name, scenarios[name]())
        reports.append(report)
        print(f"\n=== {name}: {report.accounts} accounts ===")
        for run in report.runs:
            s = run.summary()
            print(
                f"  {s['scope']:12s} {s['mib_processed']:>9.2f} MiB  "
                f"job {s['job_seconds']:>6.2f}s  wall {s['seconds']:>6.2f}s  "
                f"${s['usd']:.8f}  cache={s['cache_hit']}"
            )
        print(format_table(report))
        print(f"  all_exact={report.all_exact}")

    path = out_dir / "parity.json"
    path.write_text(
        json.dumps([r.to_dict() for r in reports], indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
