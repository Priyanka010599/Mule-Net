"""Dead-letter sinks for messages the ingest service will never accept.

Two things can dead-letter a message, and they are different:

1. Pub/Sub's *own* dead-letter policy, which forwards a message to
   ``TRANSFERS_DEAD_LETTER_TOPIC`` after ``MAX_DELIVERY_ATTEMPTS`` failed
   deliveries (both from ``shared/pubsub_schema.py``). That covers
   transient failures we asked to be retried.
2. This module, which publishes to the same topic *immediately* when the
   message can never succeed -- bad base64, a payload that fails the
   ``shared/contracts`` models, a row the database structurally rejects --
   and then acks the original. Without this, a poison message would burn
   five redeliveries before Pub/Sub retired it, and every one of those is
   a wasted Cloud Run invocation.

The original bytes are republished unchanged so the message can be
replayed after a fix; the reason travels in the message attributes.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ingest.config import IngestConfig

_LOG = logging.getLogger(__name__)

#: Pub/Sub caps an attribute value at 1024 bytes; exception text can exceed
#: that, so it is truncated rather than rejected at publish time.
MAX_ATTRIBUTE_CHARS = 900


def _attributes(
    reason: str, error: str, message_id: str, subscription: str
) -> dict[str, str]:
    return {
        "reason": reason[:MAX_ATTRIBUTE_CHARS],
        "error": error[:MAX_ATTRIBUTE_CHARS],
        "original_message_id": message_id[:MAX_ATTRIBUTE_CHARS],
        "subscription": subscription[:MAX_ATTRIBUTE_CHARS],
    }


class DeadLetterSink(Protocol):
    """Somewhere a rejected message can be parked for later inspection."""

    name: str

    def publish(
        self,
        data: bytes,
        *,
        reason: str,
        error: str,
        message_id: str = "",
        subscription: str = "",
    ) -> None:
        """Park one rejected message together with why it was rejected."""


class NoopDeadLetter:
    """Logs and discards. Only sensible when nothing is worth replaying."""

    name = "noop"

    def publish(
        self,
        data: bytes,
        *,
        reason: str,
        error: str,
        message_id: str = "",
        subscription: str = "",
    ) -> None:
        _LOG.error(
            "dead-letter DISCARDED reason=%s message_id=%s error=%s",
            reason,
            message_id,
            error,
        )


class FileDeadLetter:
    """Appends JSON Lines to a local file. The no-cloud dead-letter path.

    One object per line: the attributes plus the original ``data``
    re-encoded as base64, which is exactly what a Pub/Sub push body
    carries, so a parked message can be replayed byte-for-byte.
    """

    name = "file"

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        data: bytes,
        *,
        reason: str,
        error: str,
        message_id: str = "",
        subscription: str = "",
    ) -> None:
        record = {
            "dead_lettered_at": datetime.now(UTC).isoformat(),
            "data_b64": base64.b64encode(data).decode("ascii"),
            **_attributes(reason, error, message_id, subscription),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        _LOG.error(
            "dead-lettered to %s reason=%s message_id=%s", self.path, reason, message_id
        )

    def records(self) -> list[dict[str, Any]]:
        """Read back every parked message. Used by tests and local demos."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


class PubSubDeadLetter:
    """Publishes to the shared dead-letter topic.

    Works against the real service and against ``gcloud beta emulators
    pubsub`` unchanged: the client reads ``PUBSUB_EMULATOR_HOST`` itself.
    """

    name = "pubsub"

    def __init__(self, topic_path: str, publisher: Any | None = None) -> None:
        self.topic_path = topic_path
        self._publisher = publisher

    def _get_publisher(self) -> Any:
        if self._publisher is None:
            from google.cloud import pubsub_v1  # imported lazily

            self._publisher = pubsub_v1.PublisherClient()
        return self._publisher

    def publish(
        self,
        data: bytes,
        *,
        reason: str,
        error: str,
        message_id: str = "",
        subscription: str = "",
    ) -> None:
        future = self._get_publisher().publish(
            self.topic_path,
            data,
            **_attributes(reason, error, message_id, subscription),
        )
        # Block: the original is acked as soon as this returns, so the
        # dead letter must be durable first or the message is simply lost.
        future.result()
        _LOG.error(
            "dead-lettered to %s reason=%s message_id=%s",
            self.topic_path,
            reason,
            message_id,
        )


def build_dead_letter(cfg: IngestConfig) -> DeadLetterSink:
    """Instantiate the dead-letter sink named by the config."""
    if cfg.dlq_sink == "pubsub":
        return PubSubDeadLetter(cfg.dead_letter_topic_path)
    if cfg.dlq_sink == "file":
        return FileDeadLetter(cfg.dlq_file)
    return NoopDeadLetter()
