"""Tests for the ingest sinks and their idempotency guarantees.

The DuckDB sink stands in for the stores that need a server: it uses the
same column list, the same primary key and the same
``DECIMAL(15, 2)``/``TIMESTAMPTZ`` types, so money and timezone round
trips are genuinely exercised. The Postgres and BigQuery statements are
checked structurally instead -- see ``test_postgres_insert_*`` -- because
no server or credentials exist in this environment.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from ingest.config import IngestConfig
from ingest.sinks import (
    POSTGRES_INSERT_SQL,
    TRANSFER_COLUMNS,
    BigQuerySink,
    DuckDBSink,
    NoopSink,
    PermanentWriteError,
    build_sinks,
    to_json_row,
    to_row_tuple,
)
from shared.bq_schemas import TRANSFERS_SCHEMA
from shared.contracts import Transfer

IST = timezone(timedelta(hours=5, minutes=30))


def make_transfer(**overrides: object) -> Transfer:
    """A valid transfer with every field pinned, so tests vary one thing."""
    fields: dict[str, object] = {
        "transfer_id": "t-1",
        "from_hashed_account_id": "acct-a",
        "to_hashed_account_id": "acct-b",
        "amount": Decimal("1234.56"),
        "currency": "USD",
        "timestamp": datetime(2026, 3, 1, 12, 30, tzinfo=UTC),
        "channel": "wire",
    }
    fields.update(overrides)
    return Transfer(**fields)


def make_config(tmp_path: Path, **overrides: object) -> IngestConfig:
    """Build a config explicitly, so ambient env vars cannot affect a test."""
    fields: dict[str, object] = {
        "gcp_project": None,
        "transfers_topic": "mulenet-transfers",
        "transfers_subscription": "mulenet-transfers-sub",
        "dead_letter_topic": "mulenet-transfers-dlq",
        "pubsub_emulator_host": None,
        "push_token": None,
        "max_message_bytes": 1_048_576,
        "analytics_sink": "duckdb",
        "bq_dataset": "mulenet",
        "bq_transfers_table": "transfers",
        "operational_sink": "duckdb",
        "pg_dsn": "postgresql://mulenet:local-dev-not-a-secret@127.0.0.1:5432/mulenet",
        "dlq_sink": "file",
        "duckdb_path": tmp_path / "warehouse.duckdb",
        "dlq_file": tmp_path / "deadletter.jsonl",
    }
    fields.update(overrides)
    return IngestConfig(**fields)  # type: ignore[arg-type]


@pytest.fixture
def sink(tmp_path: Path) -> Iterator[DuckDBSink]:
    """A fresh file-backed DuckDB sink, closed after the test."""
    instance = DuckDBSink(tmp_path / "warehouse.duckdb")
    yield instance
    instance.close()


def test_columns_match_the_shared_bigquery_schema() -> None:
    """Drift guard: our column order must cover exactly TRANSFERS_SCHEMA."""
    assert TRANSFER_COLUMNS == tuple(field.name for field in TRANSFERS_SCHEMA)


def test_postgres_insert_is_idempotent_on_transfer_id() -> None:
    """Duplicate deliveries must be dropped by the primary key, not error."""
    assert "ON CONFLICT (transfer_id) DO NOTHING" in POSTGRES_INSERT_SQL
    assert POSTGRES_INSERT_SQL.count("%s") == len(TRANSFER_COLUMNS)


def test_postgres_insert_quotes_the_timestamp_column() -> None:
    """``timestamp`` is a reserved word in Postgres and in shared/ddl.sql."""
    assert '"timestamp"' in POSTGRES_INSERT_SQL


def test_duplicate_transfer_id_is_written_once(sink: DuckDBSink) -> None:
    transfer = make_transfer()
    sink.write([transfer])
    sink.write([transfer])
    sink.write([transfer, transfer])
    assert sink.count() == 1


def test_duplicate_does_not_overwrite_the_first_write(sink: DuckDBSink) -> None:
    """A redelivery of the same id must not clobber what already landed."""
    sink.write([make_transfer(amount=Decimal("10.00"))])
    sink.write([make_transfer(amount=Decimal("99.99"))])
    stored = sink.fetch_one("t-1")
    assert stored is not None
    assert stored["amount"] == Decimal("10.00")


def test_decimal_amount_survives_the_round_trip(sink: DuckDBSink) -> None:
    """0.1 + 0.2 style float error must never reach a money column."""
    amount = Decimal("1234567.89")
    sink.write([make_transfer(amount=amount)])
    stored = sink.fetch_one("t-1")
    assert stored is not None
    assert stored["amount"] == amount
    assert str(stored["amount"]) == "1234567.89"


def test_amount_is_not_routed_through_a_float(sink: DuckDBSink) -> None:
    """The classic 0.1 + 0.2 drift must not appear in a money column."""
    # Decimal(0.1 + 0.2) is 0.3000000000000000444089209850062616169452667236328125
    drifted = Decimal(0.1 + 0.2)
    assert drifted != Decimal("0.30")
    sink.write([make_transfer(amount=Decimal("0.30"), transfer_id="t-cents")])
    stored = sink.fetch_one("t-cents")
    assert stored is not None
    assert stored["amount"] == Decimal("0.30")
    assert stored["amount"] != drifted


def test_tz_aware_timestamp_is_preserved(sink: DuckDBSink) -> None:
    """A non-UTC offset must land as the same instant, still tz-aware."""
    moment = datetime(2026, 3, 1, 18, 0, tzinfo=IST)
    sink.write([make_transfer(timestamp=moment)])
    stored = sink.fetch_one("t-1")
    assert stored is not None
    assert stored["timestamp"].tzinfo is not None
    assert stored["timestamp"] == moment
    assert stored["timestamp"].astimezone(UTC).hour == 12


def test_row_tuple_keeps_decimal_and_tzinfo() -> None:
    """The DB-API binding must hand the driver a Decimal, never a float."""
    row = to_row_tuple(make_transfer(amount=Decimal("5.05")))
    assert row[TRANSFER_COLUMNS.index("amount")] == Decimal("5.05")
    assert isinstance(row[TRANSFER_COLUMNS.index("amount")], Decimal)
    assert row[TRANSFER_COLUMNS.index("timestamp")].tzinfo is not None


def test_json_row_serializes_money_as_a_two_dp_string() -> None:
    """BigQuery NUMERIC takes a string; a float here would lose cents."""
    row = to_json_row(make_transfer(amount=Decimal("7.5")))
    assert row["amount"] == "7.50"
    assert row["timestamp"].startswith("2026-03-01T12:30")


def test_bigquery_sink_uses_transfer_id_as_the_insert_id() -> None:
    """insertId-based dedup is best effort, but it must at least be wired."""
    captured: dict[str, object] = {}

    class FakeClient:
        def insert_rows_json(
            self, table: str, rows: list[dict[str, object]], row_ids: list[str]
        ) -> list[dict[str, object]]:
            captured.update(table=table, rows=rows, row_ids=row_ids)
            return []

    BigQuerySink("proj.ds.transfers", client=FakeClient()).write([make_transfer()])
    assert captured["row_ids"] == ["t-1"]
    assert captured["table"] == "proj.ds.transfers"


def test_bigquery_row_errors_are_permanent() -> None:
    """Per-row rejections are schema problems; redelivery cannot fix them."""

    class FakeClient:
        def insert_rows_json(
            self, table: str, rows: list[dict[str, object]], row_ids: list[str]
        ) -> list[dict[str, object]]:
            return [{"index": 0, "errors": [{"reason": "invalid"}]}]

    sink = BigQuerySink("proj.ds.transfers", client=FakeClient())
    with pytest.raises(PermanentWriteError):
        sink.write([make_transfer()])


def test_empty_batch_touches_no_backend() -> None:
    """An empty write must not open a connection or call the API."""

    class ExplodingClient:
        def insert_rows_json(self, *args: object, **kwargs: object) -> list[object]:
            raise AssertionError("should not be called for an empty batch")

    BigQuerySink("proj.ds.transfers", client=ExplodingClient()).write([])


def test_build_sinks_shares_one_duckdb_connection(tmp_path: Path) -> None:
    """Both roles on one DuckDB file must not open two locking connections."""
    sinks = build_sinks(make_config(tmp_path))
    try:
        assert len(sinks) == 1
        assert isinstance(sinks[0], DuckDBSink)
    finally:
        for instance in sinks:
            instance.close()


def test_build_sinks_pairs_postgres_with_duckdb(tmp_path: Path) -> None:
    """The documented local layout: real Postgres plus the DuckDB fallback."""
    sinks = build_sinks(make_config(tmp_path, operational_sink="postgres"))
    try:
        assert [instance.name for instance in sinks] == ["postgres", "duckdb"]
    finally:
        for instance in sinks:
            instance.close()


def test_noop_sink_discards_without_raising(tmp_path: Path) -> None:
    sinks = build_sinks(
        make_config(tmp_path, operational_sink="noop", analytics_sink="noop")
    )
    assert all(isinstance(instance, NoopSink) for instance in sinks)
    sinks[0].write([make_transfer()])
