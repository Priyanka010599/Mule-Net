"""Pub/Sub message envelope for MuleNet transfer events.

The payload is the :class:`shared.contracts.Transfer` model itself --
fields are never redefined here (see CLAUDE.md). Publishers in
``simulator/`` and ``ingest/`` encode with :func:`encode_message`; the
Cloud Run consumer decodes with :func:`decode_message`.

Naming convention for Pub/Sub resources (used by ``infra/``)::

    topic                   mulenet-transfers
    subscription            mulenet-transfers-sub
    dead-letter topic       mulenet-transfers-dlq
    dead-letter subscription mulenet-transfers-dlq-sub

Messages that fail ``MAX_DELIVERY_ATTEMPTS`` deliveries are forwarded
to the dead-letter topic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from shared.contracts import Transfer

#: Topic carrying live transfer events from all three banks.
TRANSFERS_TOPIC = "mulenet-transfers"

#: Pull subscription consumed by the Cloud Run ingest service.
TRANSFERS_SUBSCRIPTION = "mulenet-transfers-sub"

#: Dead-letter topic for transfer messages that repeatedly fail.
TRANSFERS_DEAD_LETTER_TOPIC = "mulenet-transfers-dlq"

#: Subscription used to inspect dead-lettered transfer messages.
TRANSFERS_DEAD_LETTER_SUBSCRIPTION = "mulenet-transfers-dlq-sub"

#: Deliveries attempted before a message is dead-lettered.
MAX_DELIVERY_ATTEMPTS = 5


class TransferMessage(BaseModel):
    """Envelope wrapping one :class:`~shared.contracts.Transfer` event.

    Attributes:
        message_id: Unique id for this envelope, for idempotent consumption.
        published_at: When the publisher emitted the message
            (timezone-aware UTC, matching the TIMESTAMPTZ columns).
        bank_id: Bank that originated the transfer, for routing and
            for single-bank vs cross-bank feature scoping.
        payload: The transfer itself.
    """

    message_id: str = Field(default_factory=lambda: str(uuid4()))
    published_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    bank_id: str
    payload: Transfer


def encode_message(message: TransferMessage) -> bytes:
    """Serialize an envelope to UTF-8 JSON bytes for Pub/Sub publishing.

    Args:
        message: The envelope to publish.

    ``Transfer.amount`` is a :data:`~shared.contracts.MoneyAmount`, so it is
    written as a JSON string with exactly two decimal places and decoded
    back to a :class:`~decimal.Decimal` without going through a float.

    Returns:
        UTF-8 encoded JSON suitable for the Pub/Sub message ``data`` field.
    """
    return message.model_dump_json().encode("utf-8")


def decode_message(data: bytes) -> TransferMessage:
    """Parse Pub/Sub message bytes back into an envelope.

    Args:
        data: The raw ``data`` field of a received Pub/Sub message.

    Returns:
        The decoded envelope.

    Raises:
        pydantic.ValidationError: If the payload does not match the schema.
    """
    return TransferMessage.model_validate_json(data.decode("utf-8"))


def encode_transfer(transfer: Transfer, bank_id: str) -> bytes:
    """Wrap a transfer in a fresh envelope and serialize it.

    Args:
        transfer: The transfer to publish.
        bank_id: Bank originating the transfer.

    Returns:
        UTF-8 encoded JSON for the Pub/Sub message ``data`` field.
    """
    return encode_message(TransferMessage(bank_id=bank_id, payload=transfer))


def decode_transfer(data: bytes) -> Transfer:
    """Parse Pub/Sub message bytes and return only the transfer payload.

    Args:
        data: The raw ``data`` field of a received Pub/Sub message.

    Returns:
        The transfer carried by the envelope.

    Raises:
        pydantic.ValidationError: If the payload does not match the schema.
    """
    return decode_message(data).payload
