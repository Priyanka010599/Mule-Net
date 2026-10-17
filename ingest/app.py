"""Cloud Run ingest service: Pub/Sub push endpoint for transfer events.

Run locally::

    uvicorn ingest.app:app --port 8080

The endpoint receives Pub/Sub's standard push body, decodes the envelope
with ``shared/pubsub_schema`` helpers, validates against
``shared/contracts``, and writes the transfer to both the operational
store (AlloyDB/Postgres) and the analytical store (BigQuery), or their
local stand-ins.

HTTP status codes are the ack protocol, so they are chosen deliberately:

===== ============================================================
Code  Meaning to Pub/Sub
===== ============================================================
204   Ack. Handled -- written, or dead-lettered on purpose.
400   Ack (4xx is not retried). The body was not a Pub/Sub push
      request at all, so it did not come from Pub/Sub.
401   Ack. The shared push token did not match.
503   **Nack.** A backend was briefly unavailable; redeliver.
===== ============================================================

Nothing that can never succeed returns 5xx, so a poison message cannot
wedge the subscription: it is published to the dead-letter topic and
acked on its first delivery.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import logging
import math
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ingest.config import IngestConfig
from ingest.dlq import DeadLetterSink, build_dead_letter
from ingest.sinks import PermanentWriteError, TransferSink, TransientWriteError, build_sinks
from shared.contracts import Transfer
from shared.pubsub_schema import decode_message

_LOG = logging.getLogger(__name__)

#: Path Pub/Sub posts to. Set as the subscription's ``pushEndpoint``.
PUSH_PATH = "/pubsub/push"


def max_body_bytes(max_message_bytes: int) -> int:
    """Largest push body we will read for a given max message payload.

    ``message.data`` arrives base64-encoded, which inflates it by 4/3;
    the JSON envelope around it adds a few hundred bytes. The slack keeps
    a legitimately maximum-sized message from being rejected.
    """
    return math.ceil(max_message_bytes * 4 / 3) + 4096


class PushMessage(BaseModel):
    """The ``message`` object inside a Pub/Sub push request.

    This is Pub/Sub's transport format, not a MuleNet data contract -- the
    MuleNet payload is the base64 ``data``, which
    :func:`shared.pubsub_schema.decode_message` owns.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    data: str = ""
    attributes: dict[str, str] = Field(default_factory=dict)
    message_id: str = Field(default="", alias="messageId")
    publish_time: str | None = Field(default=None, alias="publishTime")


class PushRequest(BaseModel):
    """The body Pub/Sub POSTs to a push endpoint."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    message: PushMessage
    subscription: str = ""


def write_transfers(sinks: Sequence[TransferSink], transfers: Sequence[Transfer]) -> None:
    """Write a batch to every sink in turn.

    Push delivers one message per HTTP request, so in practice this batch
    holds one transfer; the sinks take a sequence so the same code path
    serves ``ingest.local_dev`` backfills and any future pull consumer
    without a second implementation.

    Sinks are not written atomically across backends. That is safe because
    every sink de-duplicates on ``transfer_id``: if the second sink fails
    transiently and Pub/Sub redelivers, re-writing the first is a no-op.
    """
    for sink in sinks:
        sink.write(transfers)


def build_app(
    cfg: IngestConfig,
    *,
    sinks: Sequence[TransferSink] | None = None,
    dead_letter: DeadLetterSink | None = None,
) -> FastAPI:
    """Create the FastAPI app.

    Args:
        cfg: Resolved settings.
        sinks: Pre-built sinks. Built from ``cfg`` on startup when omitted.
        dead_letter: Pre-built dead-letter sink. Built from ``cfg`` when omitted.

    Returns:
        The configured application.
    """
    state: dict[str, Any] = {"sinks": list(sinks) if sinks is not None else None,
                             "dead_letter": dead_letter}
    body_limit = max_body_bytes(cfg.max_message_bytes)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if state["sinks"] is None:
            state["sinks"] = build_sinks(cfg)
        if state["dead_letter"] is None:
            state["dead_letter"] = build_dead_letter(cfg)
        _LOG.info(
            "ingest ready sinks=%s dead_letter=%s",
            [s.name for s in state["sinks"]],
            state["dead_letter"].name,
        )
        try:
            yield
        finally:
            for sink in state["sinks"]:
                sink.close()

    app = FastAPI(title="MuleNet ingest", lifespan=lifespan)

    def _park(data: bytes, reason: str, error: str, envelope: PushRequest | None) -> Response:
        """Dead-letter a message and ack it, so it is never redelivered."""
        state["dead_letter"].publish(
            data,
            reason=reason,
            error=error,
            message_id=envelope.message.message_id if envelope else "",
            subscription=envelope.subscription if envelope else "",
        )
        return Response(status_code=204)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Cloud Run startup/liveness probe.

        **Not** ``/healthz``. Cloud Run's front end reserves paths ending
        in ``z`` and answers them with its own HTML 404, so ``/healthz``
        never reaches the container on a ``*.run.app`` host. A probe
        pointed at it reports permanently unhealthy however well the
        service is running, so the name is load-bearing.
        """
        return {"status": "ok"}

    @app.post(PUSH_PATH)
    async def push(request: Request) -> Response:
        """Consume one Pub/Sub push delivery."""
        if cfg.push_token is not None:
            supplied = request.query_params.get("token", "")
            if not hmac.compare_digest(supplied, cfg.push_token):
                _LOG.warning("rejected push with a bad token")
                return Response(status_code=401)

        raw = await request.body()
        if len(raw) > body_limit:
            # The body is deliberately not copied into the dead letter.
            return _park(
                b"",
                "payload_too_large",
                f"push body {len(raw)} bytes exceeds {body_limit}",
                None,
            )

        try:
            envelope = PushRequest.model_validate(json.loads(raw))
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
            # Pub/Sub never sends this shape; some other client did.
            _LOG.warning("not a Pub/Sub push request: %s", exc)
            return Response(status_code=400)

        try:
            data = base64.b64decode(envelope.message.data, validate=True)
        except (binascii.Error, ValueError) as exc:
            return _park(
                envelope.message.data.encode("utf-8", "replace"),
                "invalid_base64",
                str(exc),
                envelope,
            )

        try:
            message = decode_message(data)
        except ValidationError as exc:
            return _park(data, "schema_validation_failed", str(exc), envelope)
        except (UnicodeDecodeError, ValueError) as exc:
            return _park(data, "invalid_json", str(exc), envelope)

        try:
            write_transfers(state["sinks"], [message.payload])
        except TransientWriteError as exc:
            _LOG.warning("transient write failure, asking for redelivery: %s", exc)
            return Response(status_code=503)
        except PermanentWriteError as exc:
            return _park(data, "permanent_write_failure", str(exc), envelope)

        _LOG.info(
            "ingested transfer_id=%s bank_id=%s message_id=%s",
            message.payload.transfer_id,
            message.bank_id,
            envelope.message.message_id,
        )
        return Response(status_code=204)

    return app


app = build_app(IngestConfig.from_env())
