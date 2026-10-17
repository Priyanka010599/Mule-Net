"""Tests for the Pub/Sub push endpoint.

These drive the real FastAPI app over HTTP with real Pub/Sub push bodies,
so the envelope decode, the ack/nack status codes and the dead-letter
routing are all exercised end to end. The stores are the local DuckDB and
file sinks, so nothing here needs a server or credentials.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ingest.app import PUSH_PATH, build_app, max_body_bytes
from ingest.dlq import FileDeadLetter
from ingest.sinks import DuckDBSink, PermanentWriteError, TransientWriteError
from ingest.test_sinks import IST, make_config, make_transfer
from shared.contracts import Transfer
from shared.pubsub_schema import TransferMessage, encode_message


class Harness:
    """The app under test plus the sinks it writes to."""

    def __init__(self, client: TestClient, sink: DuckDBSink, dlq: FileDeadLetter) -> None:
        self.client = client
        self.sink = sink
        self.dlq = dlq

    def post(self, data: bytes, *, message_id: str = "m-1", **params: str) -> Any:
        """POST one Pub/Sub push delivery carrying ``data``."""
        body = {
            "subscription": "projects/local/subscriptions/mulenet-transfers-sub",
            "message": {
                "data": base64.b64encode(data).decode("ascii"),
                "attributes": {"bank_id": "BANK_A"},
                "messageId": message_id,
                "publishTime": "2026-03-01T12:30:00Z",
            },
        }
        return self.client.post(PUSH_PATH, json=body, params=params)


@contextmanager
def make_harness(
    tmp_path: Path, *, sinks: Sequence[Any] | None = None, **cfg_overrides: object
) -> Iterator[Harness]:
    """Build an app wired to local sinks and yield it inside its lifespan."""
    cfg = make_config(tmp_path, **cfg_overrides)
    store = DuckDBSink(cfg.duckdb_path)
    dlq = FileDeadLetter(cfg.dlq_file)
    app = build_app(cfg, sinks=list(sinks) if sinks is not None else [store], dead_letter=dlq)
    with TestClient(app) as client:
        yield Harness(client, store, dlq)


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    with make_harness(tmp_path) as instance:
        yield instance


def envelope_bytes(transfer: Transfer, bank_id: str = "BANK_A") -> bytes:
    """The exact wire bytes ``simulator/replay.py`` publishes."""
    return encode_message(TransferMessage(bank_id=bank_id, payload=transfer))


def test_health(harness: Harness) -> None:
    assert harness.client.get("/health").json() == {"status": "ok"}


def test_the_probe_is_not_on_a_path_cloud_run_reserves(harness: Harness) -> None:
    """``/healthz`` must not come back: the front end swallows it.

    Cloud Run answers paths ending in ``z`` with its own HTML 404, so a
    probe there reports permanently unhealthy. This fails if the route is
    ever renamed back.
    """
    assert harness.client.get("/healthz").status_code == 404


def test_valid_push_is_acked_and_lands(harness: Harness) -> None:
    """The happy path: decode the shared envelope, write, ack with 2xx."""
    response = harness.post(envelope_bytes(make_transfer()))
    assert response.status_code == 204
    assert harness.sink.count() == 1
    stored = harness.sink.fetch_one("t-1")
    assert stored is not None
    assert stored["from_hashed_account_id"] == "acct-a"
    assert stored["channel"] == "wire"


def test_duplicate_delivery_is_idempotent(harness: Harness) -> None:
    """At-least-once delivery must not produce two rows."""
    data = envelope_bytes(make_transfer())
    assert harness.post(data, message_id="m-1").status_code == 204
    assert harness.post(data, message_id="m-2").status_code == 204
    assert harness.sink.count() == 1
    assert harness.dlq.records() == []


def test_decimal_amount_survives_the_http_round_trip(harness: Harness) -> None:
    """JSON over the wire must not turn money into a float."""
    harness.post(envelope_bytes(make_transfer(amount=Decimal("9876543.21"))))
    stored = harness.sink.fetch_one("t-1")
    assert stored is not None
    assert stored["amount"] == Decimal("9876543.21")


def test_tz_aware_timestamp_survives_the_http_round_trip(harness: Harness) -> None:
    """A +05:30 timestamp must land as the same instant, still tz-aware."""
    moment = datetime(2026, 3, 1, 18, 0, tzinfo=IST)
    harness.post(envelope_bytes(make_transfer(timestamp=moment)))
    stored = harness.sink.fetch_one("t-1")
    assert stored is not None
    assert stored["timestamp"].tzinfo is not None
    assert stored["timestamp"] == moment


def test_validation_failure_is_dead_lettered_and_acked(harness: Harness) -> None:
    """A payload that fails shared/contracts must never wedge the sub."""
    bad = b'{"message_id":"x","published_at":"2026-03-01T12:30:00Z","bank_id":"BANK_A"}'
    response = harness.post(bad)
    assert response.status_code == 204  # acked: redelivery would fail identically
    assert harness.sink.count() == 0
    records = harness.dlq.records()
    assert len(records) == 1
    assert records[0]["reason"] == "schema_validation_failed"
    assert records[0]["original_message_id"] == "m-1"


def test_dead_lettered_message_keeps_the_original_bytes(harness: Harness) -> None:
    """Parked messages must be replayable byte-for-byte after a fix."""
    bad = b'{"bank_id":"BANK_A"}'
    harness.post(bad)
    record = harness.dlq.records()[0]
    assert base64.b64decode(record["data_b64"]) == bad


def test_non_json_payload_is_dead_lettered(harness: Harness) -> None:
    response = harness.post(b"this is not json at all")
    assert response.status_code == 204
    assert harness.dlq.records()[0]["reason"] == "schema_validation_failed"


def test_invalid_base64_is_dead_lettered(harness: Harness) -> None:
    """``message.data`` that is not base64 cannot have come from a good publisher."""
    body = {
        "subscription": "projects/local/subscriptions/mulenet-transfers-sub",
        "message": {"data": "!!!not base64!!!", "messageId": "m-9"},
    }
    response = harness.client.post(PUSH_PATH, json=body)
    assert response.status_code == 204
    assert harness.dlq.records()[0]["reason"] == "invalid_base64"


def test_malformed_push_body_is_rejected_without_dead_lettering(harness: Harness) -> None:
    """Not a Pub/Sub push request at all: a bad client, nothing to park."""
    assert harness.client.post(PUSH_PATH, json={"nope": True}).status_code == 400
    assert harness.client.post(PUSH_PATH, content=b"{{{").status_code == 400
    assert harness.dlq.records() == []


def test_oversized_payload_is_dead_lettered_without_copying_it(tmp_path: Path) -> None:
    """A huge body is acked and parked, never redelivered, never copied."""
    with make_harness(tmp_path, max_message_bytes=64) as harness:
        limit = max_body_bytes(64)
        response = harness.post(b"x" * (limit + 1))
        assert response.status_code == 204
        assert harness.sink.count() == 0
        record = harness.dlq.records()[0]
        assert record["reason"] == "payload_too_large"
        assert record["data_b64"] == ""


class FailingSink:
    """A sink that always raises the error it was constructed with."""

    def __init__(self, error: Exception, name: str = "failing") -> None:
        self.name = name
        self.error = error
        self.calls = 0

    def write(self, transfers: Sequence[Transfer]) -> None:
        self.calls += 1
        raise self.error

    def close(self) -> None:
        return None


def test_transient_failure_asks_for_redelivery(tmp_path: Path) -> None:
    """A briefly unavailable backend must nack, not dead-letter."""
    sink = FailingSink(TransientWriteError("postgres unavailable"))
    with make_harness(tmp_path, sinks=[sink]) as harness:
        response = harness.post(envelope_bytes(make_transfer()))
        assert response.status_code == 503
        assert harness.dlq.records() == []


def test_permanent_failure_is_dead_lettered_and_acked(tmp_path: Path) -> None:
    """A row the database structurally rejects must not be retried forever."""
    sink = FailingSink(PermanentWriteError("postgres integrity: unknown account"))
    with make_harness(tmp_path, sinks=[sink]) as harness:
        response = harness.post(envelope_bytes(make_transfer()))
        assert response.status_code == 204
        record = harness.dlq.records()[0]
        assert record["reason"] == "permanent_write_failure"
        assert "unknown account" in record["error"]


def test_every_sink_receives_the_batch(tmp_path: Path) -> None:
    """Operational and analytical stores are both written, not just one."""
    first = DuckDBSink(tmp_path / "a.duckdb", name="operational")
    second = DuckDBSink(tmp_path / "b.duckdb", name="analytics")
    with make_harness(tmp_path, sinks=[first, second]) as harness:
        assert harness.post(envelope_bytes(make_transfer())).status_code == 204
        assert first.count() == 1
        assert second.count() == 1


def test_push_token_is_enforced(tmp_path: Path) -> None:
    """With a token configured, an unauthenticated push is refused."""
    with make_harness(tmp_path, push_token="local-dev-token") as harness:
        assert harness.post(envelope_bytes(make_transfer())).status_code == 401
        assert harness.post(envelope_bytes(make_transfer()), token="wrong").status_code == 401
        ok = harness.post(envelope_bytes(make_transfer()), token="local-dev-token")
        assert ok.status_code == 204
        assert harness.sink.count() == 1
