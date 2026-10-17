"""Tests for the Pub/Sub replay path. Everything here runs without GCP."""

from __future__ import annotations

import pandas as pd
import pytest

from shared.pubsub_schema import TRANSFERS_TOPIC, decode_message
from simulator.config import SimulatorConfig
from simulator.replay import (
    UNKNOWN_BANK,
    load_bank_index,
    load_transfers,
    replay,
    resolve_topic,
)

ACCOUNT_A = "a" * 64
ACCOUNT_B = "b" * 64


@pytest.fixture
def frame() -> pd.DataFrame:
    start = pd.Timestamp("2025-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "transfer_id": [f"{i:064d}" for i in range(4)],
            "from_hashed_account_id": [ACCOUNT_A, ACCOUNT_B, ACCOUNT_A, ACCOUNT_B],
            "to_hashed_account_id": [ACCOUNT_B, ACCOUNT_A, ACCOUNT_B, ACCOUNT_A],
            "amount": [10.0, 20.5, 1000.0, 7.25],
            "currency": "USD",
            "timestamp": [start + pd.Timedelta(minutes=m) for m in (0, 1, 2, 62)],
            "channel": ["p2p", "ach", "instant", "card"],
        }
    )


def test_dry_run_emits_one_envelope_per_transfer(frame) -> None:
    emitted: list[str] = []
    sent = replay(frame, dry_run=True, sleep=lambda _: None, emit=emitted.append)
    assert sent == len(frame) == len(emitted)

    message = decode_message(emitted[0].encode("utf-8"))
    assert message.payload.from_hashed_account_id == ACCOUNT_A
    assert message.payload.amount == 10.0
    assert message.bank_id == UNKNOWN_BANK


def test_bank_index_populates_the_envelope(frame) -> None:
    emitted: list[str] = []
    replay(
        frame,
        dry_run=True,
        bank_index={ACCOUNT_A: "BANK_A", ACCOUNT_B: "BANK_B"},
        sleep=lambda _: None,
        emit=emitted.append,
    )
    banks = [decode_message(e.encode("utf-8")).bank_id for e in emitted]
    assert banks == ["BANK_A", "BANK_B", "BANK_A", "BANK_B"]


def test_speed_multiplier_scales_the_wait(frame) -> None:
    """--speed 60 means one simulated minute costs one real second."""
    slept: list[float] = []
    replay(frame, speed=60.0, dry_run=True, sleep=slept.append, emit=lambda _: None)
    # gaps of 1, 1 and 60 simulated minutes -> 1s, 1s and 60s capped at max_sleep
    assert slept == [1.0, 1.0, 5.0]

    slept.clear()
    replay(
        frame, speed=120.0, dry_run=True, max_sleep_s=60.0, sleep=slept.append, emit=lambda _: None
    )
    assert slept == [0.5, 0.5, 30.0]


def test_limit_stops_early(frame) -> None:
    emitted: list[str] = []
    assert replay(frame, dry_run=True, limit=2, sleep=lambda _: None, emit=emitted.append) == 2
    assert len(emitted) == 2


def test_zero_speed_is_rejected(frame) -> None:
    with pytest.raises(ValueError, match="speed"):
        replay(frame, speed=0.0, dry_run=True)


def test_publishing_without_a_topic_is_rejected(frame) -> None:
    with pytest.raises(ValueError, match="topic"):
        replay(frame, dry_run=False, topic_path=None)


def test_round_trip_through_parquet(frame, tmp_path) -> None:
    transfers = tmp_path / "transfers.parquet"
    frame.to_parquet(transfers, index=False)
    accounts = tmp_path / "accounts.parquet"
    pd.DataFrame(
        {
            "hashed_account_id": [ACCOUNT_A, ACCOUNT_B],
            "bank_id": ["BANK_A", "BANK_C"],
        }
    ).to_parquet(accounts, index=False)

    loaded = load_transfers(transfers)
    assert loaded["timestamp"].is_monotonic_increasing
    assert load_bank_index(accounts) == {ACCOUNT_A: "BANK_A", ACCOUNT_B: "BANK_C"}


def test_topic_defaults_to_the_shared_constant(monkeypatch) -> None:
    monkeypatch.delenv("MULENET_TRANSFERS_TOPIC", raising=False)
    assert resolve_topic(SimulatorConfig.from_env()) == TRANSFERS_TOPIC

    monkeypatch.setenv("MULENET_TRANSFERS_TOPIC", "sandbox-transfers")
    assert resolve_topic(SimulatorConfig.from_env()) == "sandbox-transfers"
