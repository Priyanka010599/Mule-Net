"""Row sinks for the ingest service: AlloyDB/Postgres, BigQuery, DuckDB.

Column names and types are taken from ``shared/ddl.sql`` (operational) and
``shared/bq_schemas.TRANSFERS_SCHEMA`` (analytical); nothing is redefined
here beyond the column *order* used to bind parameters.

Idempotency
-----------
Pub/Sub is at-least-once, so the same ``transfer_id`` will arrive twice.

* Postgres/AlloyDB: ``INSERT ... ON CONFLICT (transfer_id) DO NOTHING``
  against the primary key. Exactly-once in the operational store.
* DuckDB (local analytical fallback): ``INSERT OR IGNORE`` against the same
  primary key. Exactly-once locally.
* BigQuery: streaming inserts have no upsert. We pass ``row_ids`` equal to
  the ``transfer_id`` so BigQuery's own insertId de-duplication applies --
  which is **best effort only**, over a roughly one-minute window, and is
  explicitly not guaranteed. The production-grade alternative is to stream
  into a staging table and ``MERGE`` into ``transfers`` on ``transfer_id``
  on a schedule; that is a BigQuery-side job and is not implemented here.
  Treat the BigQuery table as at-least-once and de-duplicate at read time
  (``QUALIFY ROW_NUMBER() OVER (PARTITION BY transfer_id) = 1``).

Failure classification
----------------------
``TransientWriteError`` means "retry me": the caller returns a non-2xx so
Pub/Sub redelivers. ``PermanentWriteError`` means "this row will never
land": the caller dead-letters it and acks, so a poison row cannot wedge
the subscription.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from ingest.config import IngestConfig
from shared.contracts import Transfer

_LOG = logging.getLogger(__name__)

#: Column order for ``transfers``. Mirrors ``shared/ddl.sql``; ``timestamp``
#: is quoted in SQL because it is a reserved word.
TRANSFER_COLUMNS: tuple[str, ...] = (
    "transfer_id",
    "from_hashed_account_id",
    "to_hashed_account_id",
    "amount",
    "currency",
    "timestamp",
    "channel",
)

_QUOTED_COLUMNS = ", ".join(f'"{name}"' for name in TRANSFER_COLUMNS)
_PLACEHOLDERS = ", ".join(["%s"] * len(TRANSFER_COLUMNS))

#: Upsert-free deduplicating insert for AlloyDB/Postgres.
POSTGRES_INSERT_SQL = (
    f"INSERT INTO transfers ({_QUOTED_COLUMNS}) VALUES ({_PLACEHOLDERS}) "
    "ON CONFLICT (transfer_id) DO NOTHING"
)

#: Local analytical mirror of ``TRANSFERS_SCHEMA``. No foreign keys: this is
#: the warehouse copy, which must accept a transfer whose accounts have not
#: been loaded yet.
DUCKDB_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS transfers (
    transfer_id            VARCHAR PRIMARY KEY,
    from_hashed_account_id VARCHAR        NOT NULL,
    to_hashed_account_id   VARCHAR        NOT NULL,
    amount                 DECIMAL(15, 2) NOT NULL,
    currency               VARCHAR        NOT NULL,
    "timestamp"            TIMESTAMPTZ    NOT NULL,
    channel                VARCHAR        NOT NULL
)
"""

#: DuckDB's DB-API uses qmark parameters, unlike psycopg's pyformat.
_QMARKS = ", ".join(["?"] * len(TRANSFER_COLUMNS))

DUCKDB_INSERT_SQL = (
    f"INSERT OR IGNORE INTO transfers ({_QUOTED_COLUMNS}) VALUES ({_QMARKS})"
)


class WriteError(Exception):
    """Base class for sink write failures."""


class TransientWriteError(WriteError):
    """The write may succeed on redelivery (connection lost, quota, 5xx)."""


class PermanentWriteError(WriteError):
    """The write will never succeed (constraint violation, schema mismatch)."""


class TransferSink(Protocol):
    """Somewhere a batch of validated transfers can be written."""

    name: str

    def write(self, transfers: Sequence[Transfer]) -> None:
        """Write the batch, de-duplicating on ``transfer_id`` where possible."""

    def close(self) -> None:
        """Release any client or connection held by the sink."""


def to_row_tuple(transfer: Transfer) -> tuple[Any, ...]:
    """Bind parameters in ``TRANSFER_COLUMNS`` order.

    ``amount`` stays a :class:`~decimal.Decimal` and ``timestamp`` stays a
    timezone-aware :class:`~datetime.datetime`; both DB-API drivers adapt
    them natively, so money never passes through a binary float.
    """
    return (
        transfer.transfer_id,
        transfer.from_hashed_account_id,
        transfer.to_hashed_account_id,
        transfer.amount,
        transfer.currency,
        transfer.timestamp,
        transfer.channel,
    )


def to_json_row(transfer: Transfer) -> dict[str, Any]:
    """Serialize for BigQuery's JSON streaming API.

    ``model_dump(mode="json")`` applies the shared ``MoneyAmount``
    serializer, so ``amount`` is a 2-decimal-place string -- which is what
    a BigQuery ``NUMERIC`` column wants, and avoids float rounding.
    """
    return transfer.model_dump(mode="json")


class NoopSink:
    """Drops rows, loudly. Used when a backend is deliberately unavailable."""

    def __init__(self, name: str) -> None:
        self.name = name

    def write(self, transfers: Sequence[Transfer]) -> None:
        _LOG.warning(
            "sink=%s DISCARDING %d transfer(s): this sink is a no-op",
            self.name,
            len(transfers),
        )

    def close(self) -> None:
        return None


class DuckDBSink:
    """File-backed local sink. Stands in for BigQuery with no credentials.

    Used as the analytical sink by default so the whole pipeline runs
    offline, and usable as the operational sink in tests where no Postgres
    server exists. Not a Postgres substitute in production: it has no
    foreign keys and no concurrent writers.
    """

    name = "duckdb"

    def __init__(self, path: Path, *, name: str = "duckdb") -> None:
        import duckdb  # imported lazily so the module loads without duckdb

        self.name = name
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(path))
        self._conn.execute(DUCKDB_CREATE_SQL)

    def write(self, transfers: Sequence[Transfer]) -> None:
        if not transfers:
            return
        import duckdb

        rows = [to_row_tuple(t) for t in transfers]
        try:
            self._conn.executemany(DUCKDB_INSERT_SQL, rows)
        except duckdb.ConstraintException as exc:
            raise PermanentWriteError(f"duckdb constraint: {exc}") from exc
        except duckdb.ConversionException as exc:
            raise PermanentWriteError(f"duckdb conversion: {exc}") from exc
        except duckdb.IOException as exc:
            raise TransientWriteError(f"duckdb io: {exc}") from exc

    def count(self) -> int:
        """Rows currently in the local ``transfers`` table."""
        result = self._conn.execute("SELECT count(*) FROM transfers").fetchone()
        return int(result[0]) if result else 0

    def fetch_one(self, transfer_id: str) -> dict[str, Any] | None:
        """Read one stored transfer back, for round-trip assertions.

        ``timestamp`` is read as epoch microseconds rather than as a
        TIMESTAMPTZ value: DuckDB's Python conversion for that type needs
        ``pytz``, which is not a dependency of this project. The integer
        is the exact stored instant, rebuilt here as tz-aware UTC.
        """
        columns = ", ".join(
            'epoch_us("timestamp")' if name == "timestamp" else f'"{name}"'
            for name in TRANSFER_COLUMNS
        )
        row = self._conn.execute(
            f"SELECT {columns} FROM transfers WHERE transfer_id = ?", [transfer_id]
        ).fetchone()
        if row is None:
            return None
        record = dict(zip(TRANSFER_COLUMNS, row, strict=True))
        record["amount"] = Decimal(str(record["amount"]))
        record["timestamp"] = datetime.fromtimestamp(
            record["timestamp"] / 1_000_000, tz=UTC
        )
        return record

    def close(self) -> None:
        self._conn.close()


class PostgresSink:
    """AlloyDB / PostgreSQL operational sink.

    Opens one connection lazily and reconnects on a dropped connection. A
    batch is a single ``executemany`` in one transaction, so a batch either
    lands whole or not at all.
    """

    name = "postgres"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn: Any | None = None

    def _connect(self) -> Any:
        import psycopg  # imported lazily: not needed for the duckdb-only path

        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg.connect(self.dsn, autocommit=False)
            except psycopg.OperationalError as exc:
                raise TransientWriteError(f"postgres connect: {exc}") from exc
        return self._conn

    def write(self, transfers: Sequence[Transfer]) -> None:
        if not transfers:
            return
        import psycopg

        conn = self._connect()
        rows = [to_row_tuple(t) for t in transfers]
        try:
            with conn.cursor() as cur:
                cur.executemany(POSTGRES_INSERT_SQL, rows)
            conn.commit()
        except psycopg.IntegrityError as exc:
            # Unknown account (foreign key) or a failed CHECK: redelivering
            # the identical row cannot fix it.
            conn.rollback()
            raise PermanentWriteError(f"postgres integrity: {exc}") from exc
        except psycopg.DataError as exc:
            conn.rollback()
            raise PermanentWriteError(f"postgres data: {exc}") from exc
        except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
            self._conn = None
            raise TransientWriteError(f"postgres unavailable: {exc}") from exc

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None


class BigQuerySink:
    """BigQuery analytical sink using the streaming insert API.

    ``row_ids`` are the ``transfer_id`` values, which enables BigQuery's
    best-effort insertId de-duplication (see the module docstring for its
    limits).
    """

    name = "bigquery"

    def __init__(self, table_id: str, client: Any | None = None) -> None:
        self.table_id = table_id
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from google.cloud import bigquery  # imported lazily

            self._client = bigquery.Client()
        return self._client

    def write(self, transfers: Sequence[Transfer]) -> None:
        if not transfers:
            return
        from google.api_core import exceptions as gexc

        rows = [to_json_row(t) for t in transfers]
        row_ids = [t.transfer_id for t in transfers]
        try:
            errors = self._get_client().insert_rows_json(
                self.table_id, rows, row_ids=row_ids
            )
        except (gexc.ServiceUnavailable, gexc.TooManyRequests, gexc.DeadlineExceeded) as exc:
            raise TransientWriteError(f"bigquery unavailable: {exc}") from exc
        except gexc.GoogleAPICallError as exc:
            raise TransientWriteError(f"bigquery api: {exc}") from exc
        if errors:
            # Per-row errors from insert_rows_json are schema/value problems.
            raise PermanentWriteError(f"bigquery rejected rows: {errors}")

    def close(self) -> None:
        self._client = None


def build_sinks(cfg: IngestConfig) -> list[TransferSink]:
    """Instantiate the operational and analytical sinks named by the config.

    When both roles resolve to DuckDB they share one connection to one
    file: two connections to the same DuckDB database would contend for
    its file lock, and the rows would be identical anyway.
    """
    duckdb_sink: DuckDBSink | None = None
    if "duckdb" in (cfg.operational_sink, cfg.analytics_sink):
        duckdb_sink = DuckDBSink(cfg.duckdb_path)

    sinks: list[TransferSink] = []

    if cfg.operational_sink == "postgres":
        sinks.append(PostgresSink(cfg.pg_dsn))
    elif duckdb_sink is not None and cfg.operational_sink == "duckdb":
        sinks.append(duckdb_sink)
    else:
        sinks.append(NoopSink("operational"))

    if cfg.analytics_sink == "bigquery":
        sinks.append(BigQuerySink(cfg.bq_table_id))
    elif duckdb_sink is not None and cfg.analytics_sink == "duckdb":
        if duckdb_sink not in sinks:
            sinks.append(duckdb_sink)
    else:
        sinks.append(NoopSink("analytics"))

    return sinks
