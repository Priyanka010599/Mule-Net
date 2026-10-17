"""Tests for the BigQuery-backed ring index.

Three things are checked here, in order of how much they matter:

1. **The snapshot is still the default.** With ``MULENET_READ_MODE``
   unset the app behaves exactly as it did, and the payload says so.
2. **The fallback is real and visible.** Every way the query can fail --
   exception, zero rows, a ring missing from the result -- serves the
   snapshot AND labels the payload. A silent fallback would let a judge
   read a file as a BigQuery result.
3. **The two paths agree.** The live test compares every field the SQL
   derives against the value ``build_snapshot`` derived for the same
   ring. It needs credentials and costs a scan, so it is opt-in via
   ``MULENET_BQ_LIVE=1`` and skipped otherwise; the offline tests cover
   the plumbing around it.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import pytest

from ui.bqrings import (
    BIGQUERY_FIELDS,
    SNAPSHOT_ONLY_FIELDS,
    BigQueryRingsError,
    build_query,
    fetch_ring_rows,
)
from ui.config import READ_MODES, UiConfig
from ui.datasource import (
    BIGQUERY_FALLBACK_LABEL,
    BIGQUERY_LIVE_LABEL,
    METRICS_READ_MODE_NOTE,
    DataSource,
)

CFG = UiConfig.from_env()
BQ_CFG = dataclasses.replace(CFG, read_mode="bigquery")

_needs_snapshot = pytest.mark.skipif(
    not CFG.snapshot_path.exists(),
    reason="run `python -m ui.build_snapshot` first",
)

_needs_bigquery = pytest.mark.skipif(
    os.environ.get("MULENET_BQ_LIVE", "") not in {"1", "true", "yes"},
    reason="set MULENET_BQ_LIVE=1 to run the live BigQuery agreement test "
    "(needs credentials, scans ~85 MiB)",
)


# ── the default is unchanged ─────────────────────────────────────────


def test_read_mode_defaults_to_snapshot() -> None:
    # The env var is not set in this suite, and that must mean "file".
    assert "MULENET_READ_MODE" not in os.environ
    assert CFG.read_mode == "snapshot"
    assert CFG.read_mode in READ_MODES


def test_config_rejects_an_unknown_read_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # A typo must not quietly resolve to the default: an operator who set
    # `bq` would otherwise demo a file believing BigQuery answered.
    monkeypatch.setenv("MULENET_READ_MODE", "bq")
    with pytest.raises(ValueError, match="MULENET_READ_MODE"):
        UiConfig.from_env()


@_needs_snapshot
def test_snapshot_mode_discloses_nothing_and_claims_nothing() -> None:
    src = DataSource(CFG).overview().ring_list_source
    assert (src.mode, src.served_by) == ("snapshot", "snapshot")
    # No label: there is no degradation, so there is nothing to put on
    # screen. The page hides the line entirely in this mode.
    assert src.label is None
    assert src.bytes_processed is None
    assert src.fallback_reason is None


# ── the SQL says what the docstring says it says ─────────────────────


def test_query_reads_the_configured_dataset_and_the_four_tables() -> None:
    sql = build_query(BQ_CFG)
    ref = f"{BQ_CFG.bq_project}.{BQ_CFG.bq_dataset}"
    for table in ("rings", "transfers", "account_evidence", "cases"):
        assert f"`{ref}.{table}`" in sql
    # Nothing is read from a table the dataset does not have.
    assert "ground_truth" not in sql
    assert "accounts" not in sql


def test_alertable_is_read_from_the_column_and_never_re_derived() -> None:
    sql = build_query(BQ_CFG)
    assert "WHERE m.alertable" in sql
    assert "WHERE NOT m.alertable" in sql
    # Re-deriving alertable means joining evidence and comparing banks.
    # That derivation must not appear: getting it wrong publishes an
    # alert against somebody's relative.
    assert "link_scope" not in sql


def test_intra_ring_join_requires_both_ends_in_the_SAME_ring() -> None:
    # Without this predicate a transfer between two different rings'
    # members would be counted into both rings' hop counts.
    assert "g.ring_id = f.ring_id" in build_query(BQ_CFG)


def test_evidence_collapses_identity_types_into_one_finding() -> None:
    sql = build_query(BQ_CFG)
    # One finding per (kind, scope), matching build_snapshot's grouping;
    # counting reused-id and reused-photo separately would show one
    # identity match as two corroborations.
    assert "COUNT(DISTINCT IF(" in sql
    assert "'identity_reuse', 'shared_device'" in sql
    assert "e.scope = 'cross_bank'" in sql


def test_the_fields_bigquery_cannot_answer_are_named_and_disjoint() -> None:
    assert not set(BIGQUERY_FIELDS) & set(SNAPSHOT_ONLY_FIELDS)
    for field in SNAPSHOT_ONLY_FIELDS:
        assert field in {"cross_bank_transfer_count",
                         "true_positive_alertable_members",
                         "detector_version",
                         "true_ring_id",
                         "match_is_ambiguous",
                         "has_replay"}


# ── the query layer's own failure modes ──────────────────────────────


class _StubJob:
    def __init__(self, rows: list[dict[str, Any]] | Exception) -> None:
        self._rows = rows
        self.total_bytes_processed = 12345
        self.total_bytes_billed = 20480
        self.cache_hit = False

    def result(self, timeout: float | None = None) -> list[dict[str, Any]]:
        if isinstance(self._rows, Exception):
            raise self._rows
        return self._rows


class _StubClient:
    def __init__(self, rows: list[dict[str, Any]] | Exception) -> None:
        self._rows = rows

    def query(self, sql: str, job_config: Any = None) -> _StubJob:
        return _StubJob(self._rows)


def _bq_row(ring_id: str, **over: Any) -> dict[str, Any]:
    from datetime import UTC, datetime

    row = {
        "ring_id": ring_id,
        "span": 3,
        "banks_involved": ["BANK_A", "BANK_B", "BANK_C"],
        "status": "candidate",
        "detected_at": datetime(2025, 1, 30, 23, 59, 58, tzinfo=UTC),
        "total_member_count": 5,
        "alertable_member_count": 4,
        "context_member_count": 1,
        "intra_ring_transfer_count": 41,
        "intra_ring_amount": "7654.3",
        "onset": datetime(2025, 1, 10, 1, 2, 3, tzinfo=UTC),
        "cross_bank_evidence_count": 2,
        "has_case": True,
    }
    row.update(over)
    return row


def _derived(ring_id: str, **over: Any) -> dict[str, Any]:
    """A ``RingQueryResult.rows`` entry, i.e. a BigQuery row already
    converted the way :func:`fetch_ring_rows` converts it."""
    from ui.bqrings import _row_to_fields

    fields = _row_to_fields(_bq_row(ring_id))
    fields.update(over)
    return fields


def test_fetch_raises_rather_than_returning_a_partial_answer() -> None:
    client = _StubClient(RuntimeError("403 caller lacks bigquery.jobs.create"))
    with pytest.raises(BigQueryRingsError, match="RuntimeError"):
        fetch_ring_rows(BQ_CFG, ["ring-1"], client=client)


def test_zero_rows_is_a_failure_not_an_empty_ring_list() -> None:
    # The ring ids came from a snapshot that has them, so nothing coming
    # back means the dataset is unloaded or the query is wrong.
    with pytest.raises(BigQueryRingsError, match="no rows"):
        fetch_ring_rows(BQ_CFG, ["ring-1"], client=_StubClient([]))


def test_fetch_reports_the_job_facts_and_the_2dp_money_string() -> None:
    result = fetch_ring_rows(
        BQ_CFG, ["ring-1"], client=_StubClient([_bq_row("ring-1")])
    )
    assert result.bytes_processed == 12345
    assert result.bytes_billed == 20480
    assert result.cache_hit is False
    assert result.latency_ms >= 0
    assert result.table_ref == f"{BQ_CFG.bq_project}.{BQ_CFG.bq_dataset}"
    row = result.rows["ring-1"]
    # Money crosses the wire as a fixed 2-decimal string, like every
    # other amount in this API.
    assert row["total_intra_ring_amount"] == "7654.30"
    assert row["detected_at"] == "2025-01-30T23:59:58+00:00"


# ── the fallback, at the layer the page reads ────────────────────────


def _bq_source(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> Any:
    """Run one overview() in bigquery mode with ``fetch_ring_rows`` stubbed."""

    def fake(cfg: UiConfig, ring_ids: list[str], client: Any = None) -> Any:
        if isinstance(outcome, Exception):
            raise outcome
        return outcome(ring_ids)

    monkeypatch.setattr("ui.datasource.fetch_ring_rows", fake)
    return DataSource(BQ_CFG).overview()


@_needs_snapshot
def test_a_failed_query_serves_the_snapshot_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = DataSource(CFG).overview()
    view = _bq_source(monkeypatch, BigQueryRingsError("DeadlineExceeded: 20s"))
    src = view.ring_list_source

    assert (src.mode, src.served_by) == ("bigquery", "snapshot")
    assert src.label == BIGQUERY_FALLBACK_LABEL
    # The wording must not be reusable as the chat dock's degradation:
    # one means "a file answered instead of a query", the other means "a
    # keyword matcher answered instead of Gemini".
    assert "offline demo mode" not in src.label
    assert "offline demo mode" not in src.detail
    assert src.fallback_reason == "DeadlineExceeded: 20s"
    assert "DeadlineExceeded" in src.detail
    assert "NOTHING on this page came from BigQuery" in src.detail
    # No BigQuery field may be claimed as derived when none was.
    assert src.bigquery_fields == []
    assert src.bytes_processed is None
    # And the rings themselves are exactly the snapshot's.
    assert view.rings == baseline.rings


@_needs_snapshot
def test_a_ring_missing_from_the_result_is_also_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ui.bqrings import RingQueryResult

    def partial(ring_ids: list[str]) -> RingQueryResult:
        kept = ring_ids[:-1]
        return RingQueryResult(
            rows={r: _derived(r) for r in kept},
            bytes_processed=1,
            bytes_billed=1,
            cache_hit=False,
            latency_ms=1.0,
            table_ref="p.d",
        )

    src = _bq_source(monkeypatch, partial).ring_list_source
    # A row per ring or nothing: half a BigQuery answer rendered as a
    # whole one is the failure this guards against.
    assert src.served_by == "snapshot"
    assert "absent from" in (src.fallback_reason or "")


@_needs_snapshot
def test_a_successful_query_labels_the_page_and_states_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ui.bqrings import RingQueryResult

    def ok(ring_ids: list[str]) -> RingQueryResult:
        return RingQueryResult(
            rows={r: _derived(r) for r in ring_ids},
            bytes_processed=89_530_395,
            bytes_billed=90_177_536,
            cache_hit=False,
            latency_ms=3193.0,
            table_ref="mulenet.mulenet",
        )

    src = _bq_source(monkeypatch, ok).ring_list_source
    assert (src.mode, src.served_by) == ("bigquery", "bigquery")
    assert src.label == BIGQUERY_LIVE_LABEL
    assert src.bytes_processed == 89_530_395
    assert src.latency_ms == 3193.0
    assert "89,530,395 bytes scanned" in src.detail
    # The fields the dataset cannot answer are named on the page, not
    # only in a docstring, so a viewer cannot read the whole row as
    # BigQuery-derived.
    for field in SNAPSHOT_ONLY_FIELDS:
        assert field in src.detail
    assert src.snapshot_fields == list(SNAPSHOT_ONLY_FIELDS)


@_needs_snapshot
def test_bigquery_overrides_only_the_fields_it_derives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ui.bqrings import RingQueryResult

    baseline = {r.ring_id: r for r in DataSource(CFG).overview().rings}
    target = next(iter(baseline))

    def ok(ring_ids: list[str]) -> RingQueryResult:
        rows = {r: _derived(r) for r in ring_ids}
        # A value no snapshot row has, so an override is unmistakable.
        rows[target]["intra_ring_transfer_count"] = 999_999
        return RingQueryResult(
            rows=rows,
            bytes_processed=1,
            bytes_billed=1,
            cache_hit=True,
            latency_ms=1.0,
            table_ref="p.d",
        )

    served = {r.ring_id: r for r in _bq_source(monkeypatch, ok).rings}
    assert served[target].intra_ring_transfer_count == 999_999
    for field in SNAPSHOT_ONLY_FIELDS:
        assert getattr(served[target], field) == getattr(baseline[target], field)


# ── the metrics page stays honest about its file source ──────────────


@_needs_snapshot
def test_metrics_has_no_read_mode_note_in_snapshot_mode() -> None:
    assert DataSource(CFG).metrics().read_mode_note is None


@_needs_snapshot
def test_metrics_denies_a_bigquery_source_in_bigquery_mode() -> None:
    payload = DataSource(BQ_CFG).metrics()
    assert payload.read_mode_note == METRICS_READ_MODE_NOTE
    # It says the numbers came from a file and why they could not come
    # from BigQuery, and source_detail still names the file that was
    # actually opened rather than a configured one.
    assert "no metrics table" in payload.read_mode_note
    assert payload.source_detail.startswith(("read live from", "bundled copy in"))
    assert "bigquery" not in payload.source_detail.lower()
    if payload.coverage is not None:
        assert "bigquery" not in payload.coverage.source.lower()


# ── the two paths agree (live; opt-in) ───────────────────────────────


@_needs_snapshot
@_needs_bigquery
def test_bigquery_and_snapshot_ring_lists_agree() -> None:
    """Same rings, same derived values, field by field.

    A disagreement here is a finding about one of the two definitions,
    not a reason to trust the newer path.
    """
    snapshot_rings = {r.ring_id: r for r in DataSource(CFG).overview().rings}
    result = fetch_ring_rows(BQ_CFG, sorted(snapshot_rings))

    assert set(result.rows) == set(snapshot_rings)
    mismatches: list[str] = []
    for ring_id, derived in result.rows.items():
        expected = snapshot_rings[ring_id]
        for field in BIGQUERY_FIELDS:
            if field == "ring_id":
                continue
            want, got = getattr(expected, field), derived[field]
            if want != got:
                mismatches.append(f"{ring_id}.{field}: snapshot {want!r} != bq {got!r}")
    assert not mismatches, "\n".join(mismatches)
