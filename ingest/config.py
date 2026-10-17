"""Environment-driven configuration for the MuleNet ingest service.

Every runtime knob is read from an environment variable here (CLAUDE.md:
"Config via env vars in one config.py per module"). Nothing else in
``ingest/`` reads ``os.environ``.

Secrets (the AlloyDB DSN with its password, the push shared token) are
never defaulted to anything usable. In Cloud Run they are injected as
environment variables straight from Secret Manager::

    gcloud run deploy mulenet-ingest \\
        --set-secrets MULENET_INGEST_PG_DSN=mulenet-alloydb-dsn:latest \\
        --set-secrets MULENET_INGEST_PUSH_TOKEN=mulenet-push-token:latest

The local defaults point at ``127.0.0.1`` with an obviously fake password
so nothing here can be mistaken for a real credential.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from shared.pubsub_schema import (
    TRANSFERS_DEAD_LETTER_TOPIC,
    TRANSFERS_SUBSCRIPTION,
    TRANSFERS_TOPIC,
)

#: Local-only DSN. Loopback host, placeholder password: never a real credential.
DEFAULT_PG_DSN = "postgresql://mulenet:local-dev-not-a-secret@127.0.0.1:5432/mulenet"

#: Local sink files live next to the code and are gitignored.
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"

#: BigQuery dataset/table holding the analytical copy of ``transfers``.
DEFAULT_BQ_DATASET = "mulenet"
DEFAULT_BQ_TRANSFERS_TABLE = "transfers"

#: Pub/Sub's own hard limit for a single message payload.
DEFAULT_MAX_MESSAGE_BYTES = 1_048_576

#: Where the analytical copy goes. ``duckdb`` is the no-cloud fallback.
ANALYTICS_SINKS = frozenset({"bigquery", "duckdb", "noop"})

#: Where the operational copy goes. ``postgres`` is AlloyDB in cloud.
OPERATIONAL_SINKS = frozenset({"postgres", "duckdb", "noop"})

#: The deployed stack runs Pub/Sub and BigQuery only -- no Cloud SQL and no
#: AlloyDB is provisioned (see docs/deployment.md). So when a real GCP project
#: is configured and we are not pointed at the Pub/Sub emulator, the
#: operational copy defaults to ``noop`` rather than to a database that is not
#: there. The local emulator path still gets ``duckdb``, and an operator who
#: does provision Postgres turns it back on explicitly with
#: ``MULENET_INGEST_OPERATIONAL_SINK=postgres``.
CLOUD_OPERATIONAL_SINK = "noop"
LOCAL_OPERATIONAL_SINK = "duckdb"

#: Where rejected messages go. Empty env value means "pubsub if a project
#: is configured, file otherwise".
DLQ_SINKS = frozenset({"pubsub", "file", "noop"})


def _env_choice(name: str, default: str, allowed: frozenset[str]) -> str:
    value = os.environ.get(name) or default
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - operator error
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class IngestConfig:
    """Resolved ingest settings."""

    gcp_project: str | None
    transfers_topic: str
    transfers_subscription: str
    dead_letter_topic: str
    pubsub_emulator_host: str | None
    push_token: str | None
    max_message_bytes: int

    analytics_sink: str
    bq_dataset: str
    bq_transfers_table: str

    operational_sink: str
    pg_dsn: str

    dlq_sink: str
    duckdb_path: Path
    dlq_file: Path

    @classmethod
    def from_env(cls) -> IngestConfig:
        """Build a config from the process environment, applying defaults."""
        out_dir = _env_path("MULENET_INGEST_OUT_DIR", DEFAULT_OUT_DIR)
        project = os.environ.get("MULENET_GCP_PROJECT") or None
        dlq_default = "pubsub" if project else "file"
        emulator = os.environ.get("PUBSUB_EMULATOR_HOST") or None
        operational_default = (
            CLOUD_OPERATIONAL_SINK
            if project and not emulator
            else LOCAL_OPERATIONAL_SINK
        )
        return cls(
            gcp_project=project,
            transfers_topic=os.environ.get("MULENET_TRANSFERS_TOPIC", TRANSFERS_TOPIC),
            transfers_subscription=os.environ.get(
                "MULENET_TRANSFERS_SUBSCRIPTION", TRANSFERS_SUBSCRIPTION
            ),
            dead_letter_topic=os.environ.get(
                "MULENET_TRANSFERS_DLQ_TOPIC", TRANSFERS_DEAD_LETTER_TOPIC
            ),
            pubsub_emulator_host=os.environ.get("PUBSUB_EMULATOR_HOST") or None,
            push_token=os.environ.get("MULENET_INGEST_PUSH_TOKEN") or None,
            max_message_bytes=_env_int(
                "MULENET_INGEST_MAX_MESSAGE_BYTES", DEFAULT_MAX_MESSAGE_BYTES
            ),
            analytics_sink=_env_choice(
                "MULENET_INGEST_ANALYTICS_SINK", "duckdb", ANALYTICS_SINKS
            ),
            bq_dataset=os.environ.get("MULENET_BQ_DATASET", DEFAULT_BQ_DATASET),
            bq_transfers_table=os.environ.get(
                "MULENET_BQ_TRANSFERS_TABLE", DEFAULT_BQ_TRANSFERS_TABLE
            ),
            operational_sink=_env_choice(
                "MULENET_INGEST_OPERATIONAL_SINK",
                operational_default,
                OPERATIONAL_SINKS,
            ),
            pg_dsn=os.environ.get("MULENET_INGEST_PG_DSN", DEFAULT_PG_DSN),
            dlq_sink=_env_choice("MULENET_INGEST_DLQ_SINK", dlq_default, DLQ_SINKS),
            duckdb_path=_env_path(
                "MULENET_INGEST_DUCKDB_PATH", out_dir / "warehouse.duckdb"
            ),
            dlq_file=_env_path(
                "MULENET_INGEST_DLQ_FILE", out_dir / "deadletter.jsonl"
            ),
        )

    @property
    def dead_letter_topic_path(self) -> str:
        """Fully-qualified dead-letter topic path, for the Pub/Sub client."""
        if not self.gcp_project:
            raise ValueError(
                "MULENET_GCP_PROJECT is required for the pubsub dead-letter sink"
            )
        return f"projects/{self.gcp_project}/topics/{self.dead_letter_topic}"

    @property
    def bq_table_id(self) -> str:
        """``project.dataset.table`` for the analytical transfers table."""
        if not self.gcp_project:
            raise ValueError("MULENET_GCP_PROJECT is required for the bigquery sink")
        return f"{self.gcp_project}.{self.bq_dataset}.{self.bq_transfers_table}"
