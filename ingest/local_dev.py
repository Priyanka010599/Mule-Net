"""Local end-to-end helpers for the ingest service.

Not used in production: this wires up the Pub/Sub emulator and a local
Postgres so ``simulator/replay.py`` -> ingest -> stores runs with no cloud
credentials. See ``ingest/README.md`` for the full command sequence.

    python -m ingest.local_dev emulator-setup   # topics + push subscription + DLQ
    python -m ingest.local_dev bootstrap-db     # shared/ddl.sql + accounts
    python -m ingest.local_dev count            # rows landed
    python -m ingest.local_dev dlq              # parked messages
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ingest.config import IngestConfig
from ingest.dlq import FileDeadLetter
from ingest.sinks import DuckDBSink
from shared.pubsub_schema import MAX_DELIVERY_ATTEMPTS

_LOG = logging.getLogger(__name__)

#: DDL owned by the architect. Applied verbatim, never edited from here.
DDL_PATH = Path(__file__).resolve().parents[1] / "shared" / "ddl.sql"

#: Where the push subscription delivers when everything runs on this machine.
DEFAULT_PUSH_ENDPOINT = "http://localhost:8080/pubsub/push"

_ACCOUNT_INSERT_SQL = (
    "INSERT INTO accounts "
    "(hashed_account_id, bank_id, opened_at, account_type, status) "
    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (hashed_account_id) DO NOTHING"
)


def setup_emulator(cfg: IngestConfig, push_endpoint: str) -> None:
    """Create the transfers topic, the dead-letter topic and a push subscription.

    Idempotent: re-running against an emulator that already has them is a
    no-op. Requires ``PUBSUB_EMULATOR_HOST`` to be set.
    """
    from google.api_core import exceptions as gexc
    from google.cloud import pubsub_v1

    if not cfg.pubsub_emulator_host:
        raise SystemExit("PUBSUB_EMULATOR_HOST is not set; start the emulator first")
    if not cfg.gcp_project:
        raise SystemExit("MULENET_GCP_PROJECT is required (any string for the emulator)")

    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    topic = publisher.topic_path(cfg.gcp_project, cfg.transfers_topic)
    dlq_topic = publisher.topic_path(cfg.gcp_project, cfg.dead_letter_topic)
    for path in (topic, dlq_topic):
        try:
            publisher.create_topic(request={"name": path})
            print(f"created topic {path}")
        except gexc.AlreadyExists:
            print(f"topic exists {path}")

    sub = subscriber.subscription_path(cfg.gcp_project, cfg.transfers_subscription)
    request = {
        "name": sub,
        "topic": topic,
        "push_config": {"push_endpoint": push_endpoint},
        "dead_letter_policy": {
            "dead_letter_topic": dlq_topic,
            "max_delivery_attempts": MAX_DELIVERY_ATTEMPTS,
        },
    }
    try:
        subscriber.create_subscription(request=request)
        print(f"created push subscription {sub} -> {push_endpoint}")
    except gexc.AlreadyExists:
        print(f"subscription exists {sub}")
    except gexc.InvalidArgument as exc:
        # Older emulator builds reject dead_letter_policy. The service
        # dead-letters poison messages itself, so this only loses the
        # retry-exhaustion path.
        _LOG.warning("emulator rejected the dead-letter policy (%s); retrying without", exc)
        request.pop("dead_letter_policy")
        try:
            subscriber.create_subscription(request=request)
            print(f"created push subscription {sub} (no dead-letter policy)")
        except gexc.AlreadyExists:
            print(f"subscription exists {sub}")

    dlq_sub = subscriber.subscription_path(
        cfg.gcp_project, f"{cfg.dead_letter_topic}-sub"
    )
    try:
        subscriber.create_subscription(request={"name": dlq_sub, "topic": dlq_topic})
        print(f"created dead-letter pull subscription {dlq_sub}")
    except gexc.AlreadyExists:
        print(f"subscription exists {dlq_sub}")


def apply_ddl(dsn: str) -> None:
    """Run ``shared/ddl.sql`` against a Postgres/AlloyDB instance."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        conn.execute(DDL_PATH.read_text(encoding="utf-8"))
        conn.commit()
    print(f"applied {DDL_PATH}")


def load_accounts(dsn: str, accounts_parquet: Path) -> int:
    """Load ``accounts.parquet`` so transfers satisfy their foreign keys.

    ``transfers`` references ``accounts`` in ``shared/ddl.sql``, so the
    accounts a replay refers to must exist before ingest can write. In
    production the onboarding path does this; locally the simulator's
    parquet is the source.
    """
    import pandas as pd
    import psycopg

    frame = pd.read_parquet(
        accounts_parquet,
        columns=["hashed_account_id", "bank_id", "opened_at", "account_type", "status"],
    )
    rows = [
        (
            row["hashed_account_id"],
            row["bank_id"],
            row["opened_at"].to_pydatetime(),
            row["account_type"],
            str(row["status"]),
        )
        for _, row in frame.iterrows()
    ]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.executemany(_ACCOUNT_INSERT_SQL, rows)
        conn.commit()
    print(f"loaded {len(rows)} accounts from {accounts_parquet}")
    return len(rows)


def count_rows(cfg: IngestConfig) -> None:
    """Print how many transfers landed in each configured store."""
    if cfg.operational_sink == "postgres":
        import psycopg

        with psycopg.connect(cfg.pg_dsn) as conn:
            row = conn.execute("SELECT count(*) FROM transfers").fetchone()
        print(f"postgres transfers: {row[0] if row else 0}")
    if "duckdb" in (cfg.operational_sink, cfg.analytics_sink):
        sink = DuckDBSink(cfg.duckdb_path)
        try:
            print(f"duckdb transfers ({cfg.duckdb_path}): {sink.count()}")
        finally:
            sink.close()
    if cfg.analytics_sink == "bigquery":
        print(f"bigquery table: {cfg.bq_table_id} (query it with bq/BigQuery console)")


def show_dlq(cfg: IngestConfig) -> None:
    """Print parked messages from the local file dead-letter sink."""
    if cfg.dlq_sink != "file":
        print(
            f"dead-letter sink is {cfg.dlq_sink!r}; "
            f"pull {cfg.dead_letter_topic}-sub to inspect it"
        )
        return
    records = FileDeadLetter(cfg.dlq_file).records()
    print(f"{len(records)} dead-lettered message(s) in {cfg.dlq_file}")
    for record in records:
        print(f"  {record['dead_lettered_at']} {record['reason']}: {record['error'][:120]}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m ingest.local_dev``."""
    logging.basicConfig(level=logging.INFO)
    cfg = IngestConfig.from_env()
    parser = argparse.ArgumentParser(description="Local helpers for the ingest service.")
    sub = parser.add_subparsers(dest="command", required=True)

    emulator = sub.add_parser("emulator-setup", help="create topics and push subscription")
    emulator.add_argument("--push-endpoint", default=DEFAULT_PUSH_ENDPOINT)

    bootstrap = sub.add_parser("bootstrap-db", help="apply shared/ddl.sql and load accounts")
    bootstrap.add_argument(
        "--accounts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "simulator" / "out" / "accounts.parquet",
    )
    bootstrap.add_argument("--dsn", default=cfg.pg_dsn)

    sub.add_parser("count", help="row counts per store")
    sub.add_parser("dlq", help="show dead-lettered messages")

    args = parser.parse_args(argv)

    if args.command == "emulator-setup":
        setup_emulator(cfg, args.push_endpoint)
    elif args.command == "bootstrap-db":
        apply_ddl(args.dsn)
        if args.accounts.exists():
            load_accounts(args.dsn, args.accounts)
        else:
            print(f"no accounts parquet at {args.accounts}; run simulator/generator.py first")
    elif args.command == "count":
        count_rows(cfg)
    elif args.command == "dlq":
        show_dlq(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
