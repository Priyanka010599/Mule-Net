"""Replay generated transfers to Pub/Sub at a configurable speed multiplier.

``--speed 60`` means one simulated minute passes per real second. ``--dry-run``
prints the encoded envelopes instead of publishing, so the replay path is
testable with no GCP credentials.

Topic naming and the wire format both come from ``shared/pubsub_schema.py``:
messages are :class:`~shared.pubsub_schema.TransferMessage` envelopes encoded
with :func:`~shared.pubsub_schema.encode_transfer`, published to
``TRANSFERS_TOPIC``. ``MULENET_TRANSFERS_TOPIC`` can override the topic name
for a sandbox deployment.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd

from shared.contracts import Transfer
from shared.pubsub_schema import TRANSFERS_TOPIC, encode_transfer
from simulator.config import DEFAULT_TRANSFERS_TOPIC, SimulatorConfig

#: Never sleep longer than this between two messages, whatever the gap.
DEFAULT_MAX_SLEEP_S = 5.0

#: Used when the originating bank cannot be resolved from accounts.parquet.
UNKNOWN_BANK = "UNKNOWN"


def resolve_topic(cfg: SimulatorConfig) -> str:
    """The shared topic name, unless the env var explicitly overrides it."""
    if cfg.transfers_topic != DEFAULT_TRANSFERS_TOPIC:
        return cfg.transfers_topic
    return TRANSFERS_TOPIC


def load_transfers(path: Path) -> pd.DataFrame:
    """Read transfers.parquet ordered by simulated time."""
    frame = pd.read_parquet(path)
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_bank_index(path: Path) -> dict[str, str]:
    """Map hashed_account_id -> bank_id, for the envelope's ``bank_id`` field."""
    frame = pd.read_parquet(path, columns=["hashed_account_id", "bank_id"])
    return dict(zip(frame["hashed_account_id"], frame["bank_id"], strict=True))


def iter_transfers(frame: pd.DataFrame) -> Iterator[Transfer]:
    """Validate each row against the shared ``Transfer`` contract."""
    for row in frame.to_dict(orient="records"):
        yield Transfer(**row)


def replay(
    frame: pd.DataFrame,
    *,
    speed: float = 60.0,
    topic_path: str | None = None,
    dry_run: bool = True,
    limit: int | None = None,
    max_sleep_s: float = DEFAULT_MAX_SLEEP_S,
    bank_index: dict[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    emit: Callable[[str], None] = print,
) -> int:
    """Publish (or print) transfers in simulated-time order. Returns the count."""
    if speed <= 0:
        raise ValueError("--speed must be greater than zero")

    banks = bank_index or {}
    publisher = None
    if not dry_run:
        if not topic_path:
            raise ValueError("a fully-qualified topic path is required unless --dry-run")
        from google.cloud import pubsub_v1  # imported lazily: no GCP dep for dry runs

        publisher = pubsub_v1.PublisherClient()

    sent = 0
    previous: pd.Timestamp | None = None
    for transfer in iter_transfers(frame):
        if limit is not None and sent >= limit:
            break
        current = pd.Timestamp(transfer.timestamp)
        if previous is not None:
            wait = (current - previous).total_seconds() / speed
            if wait > 0:
                sleep(min(wait, max_sleep_s))
        previous = current

        bank_id = banks.get(transfer.from_hashed_account_id, UNKNOWN_BANK)
        data = encode_transfer(transfer, bank_id)
        if publisher is None:
            emit(data.decode("utf-8"))
        else:
            publisher.publish(topic_path, data, bank_id=bank_id, channel=transfer.channel)
        sent += 1
    return sent


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m simulator.replay``."""
    cfg = SimulatorConfig.from_env()
    parser = argparse.ArgumentParser(description="Replay transfers.parquet to Pub/Sub.")
    parser.add_argument("--input", type=Path, default=cfg.out_dir / "transfers.parquet")
    parser.add_argument("--accounts", type=Path, default=cfg.out_dir / "accounts.parquet")
    parser.add_argument(
        "--speed",
        type=float,
        default=60.0,
        help="simulated seconds per real second (60 = 1 sim minute per real second)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print instead of publish")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--topic", default=resolve_topic(cfg))
    parser.add_argument("--project", default=cfg.gcp_project)
    parser.add_argument("--max-sleep", type=float, default=DEFAULT_MAX_SLEEP_S)
    args = parser.parse_args(argv)

    topic_path = None
    if not args.dry_run:
        if not args.project:
            parser.error("MULENET_GCP_PROJECT or --project is required unless --dry-run")
        topic_path = f"projects/{args.project}/topics/{args.topic}"

    bank_index = load_bank_index(args.accounts) if args.accounts.exists() else {}
    sent = replay(
        load_transfers(args.input),
        speed=args.speed,
        topic_path=topic_path,
        dry_run=args.dry_run,
        limit=args.limit,
        max_sleep_s=args.max_sleep,
        bank_index=bank_index,
    )
    print(f"replayed {sent} transfers to {topic_path or '<dry-run>'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
