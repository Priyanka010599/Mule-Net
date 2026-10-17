"""Suggested questions are answered from the rings table with no model call."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agents import adk_agents, data, precomputed, service
from agents.config import AgentsConfig, ServiceLimits
from agents.limits import RequestGuard


def _ring(ring_id: str, alertable: list[bool], banks: list[str]) -> dict[str, Any]:
    return {
        "ring_id": ring_id,
        "members": [
            {"hashed_account_id": f"{ring_id}-{i}", "alertable": flag}
            for i, flag in enumerate(alertable)
        ],
        "banks_involved": banks,
    }


class FakeStore:
    def __init__(self, rows: list[dict[str, Any]], version: str = "v1") -> None:
        self.rows = rows
        self.version = version
        self.reads = 0

    def table(self, name: str) -> pd.DataFrame:
        assert name == "rings"
        self.reads += 1
        return pd.DataFrame(self.rows, columns=["ring_id", "members", "banks_involved"])

    def data_version(self, names: tuple[str, ...] = ()) -> str:
        return self.version


RINGS = [
    _ring("ring-b", [True, True, False], ["BANK_A", "BANK_B"]),
    _ring("ring-a", [True, False, True], ["BANK_A", "BANK_C", "BANK_B"]),  # ties with ring-b
    _ring("ring-c", [True], ["BANK_A"]),
]


@pytest.mark.parametrize(
    "text",
    [
        precomputed.Q_RING_COUNT,
        "  how MANY rings are in the dataset ",
        "About ring ring-c: How many rings are in the dataset?",
    ],
)
def test_matching_ignores_case_punctuation_whitespace_and_ring_prefix(text: str) -> None:
    assert precomputed.is_suggested(text)


def test_other_questions_are_not_matched() -> None:
    assert not precomputed.is_suggested("How many rings are there in BANK_A?")
    assert precomputed.lookup("something else", FakeStore(RINGS)) is None


def test_answers_are_computed_from_the_table_with_the_definition_of_largest() -> None:
    store = FakeStore(RINGS)
    count = precomputed.lookup(precomputed.Q_RING_COUNT, store)
    banks = precomputed.lookup(precomputed.Q_LARGEST_BANKS, store)
    alertable = precomputed.lookup(precomputed.Q_LARGEST_ALERTABLE, store)
    assert count is not None and "3 rings" in count.answer
    # Tie on 3 members: the smaller ring id wins.
    assert banks is not None
    assert "ring-a" in banks.answer and "3 member" in banks.answer
    assert "BANK_A, BANK_B and BANK_C" in banks.answer
    assert "most members" in banks.answer and "ring id" in banks.answer
    assert alertable is not None and "2 alertable" in alertable.answer


def test_cached_by_data_version() -> None:
    store = FakeStore(RINGS, version="cache-1")
    precomputed.lookup(precomputed.Q_RING_COUNT, store)
    precomputed.lookup(precomputed.Q_LARGEST_BANKS, store)
    assert store.reads == 1
    store.version = "cache-2"
    store.rows = RINGS[:1]
    again = precomputed.lookup(precomputed.Q_RING_COUNT, store)
    assert again is not None and "1 rings" in again.answer
    assert store.reads == 2


def test_empty_table_falls_through_to_the_model() -> None:
    assert precomputed.lookup(precomputed.Q_RING_COUNT, FakeStore([], "empty")) is None


def test_chat_serves_it_first_without_model_or_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore(RINGS, version="chat-1")
    monkeypatch.setattr(data.DataStore, "table", lambda _self, name: store.table(name))
    monkeypatch.setattr(data.DataStore, "data_version", lambda _self, names=(): store.version)

    def boom(*_a: Any, **_k: Any) -> str:
        raise AssertionError("the model must not be called")

    monkeypatch.setattr(adk_agents, "_run_agent", boom)
    guard = RequestGuard(ServiceLimits(session_limit=1, ip_limit=1, daily_cap=1))
    monkeypatch.setattr(service, "missing_tables", lambda _c: [])
    client = TestClient(
        service.create_app(AgentsConfig.from_env(), guard=guard, require_data=False)
    )
    for _ in range(3):  # a 1-per-window guard would refuse the second
        response = client.post(
            "/chat",
            json={"session_id": "s", "message": precomputed.Q_LARGEST_ALERTABLE},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["source"] == "precomputed"
        assert body["backend"] == "precomputed"
        assert "2 alertable" in body["answer"]
    assert guard.remaining_today() == 1


# ---------------------------------------------------- the race-to-the-exit question

from pathlib import Path  # noqa: E402

from agents import tools  # noqa: E402
from agents.test_chat_claims import Tiny, _config, tiny  # noqa: E402, F401
from agents.test_ring_race import HELPS, TOGETHER, _publish, out_dir  # noqa: E402, F401
from shared import ring_race  # noqa: E402


def _model_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fake(name: str, message: str, *args: Any, **kwargs: Any) -> str:
        calls.append(name)
        raise AssertionError("the model must not be called for a precomputed answer")

    monkeypatch.setattr(adk_agents, "_run_agent", fake)
    return calls


def test_the_race_question_is_answered_from_the_tool_data_with_no_model_call(
    monkeypatch: pytest.MonkeyPatch,
    tiny: Tiny,  # noqa: F811
    out_dir: Path,  # noqa: F811
) -> None:
    _publish(out_dir, tiny.ring_id, ring_race.entry(100.0, HELPS, False))
    calls = _model_calls(monkeypatch)
    monkeypatch.setattr(service, "missing_tables", lambda _c: [])
    limits = ServiceLimits(session_limit=1000, ip_limit=1000, window_s=60.0, daily_cap=10_000)
    api = TestClient(service.create_app(_config(mode="mock", use_vertex=False), limits))
    body = api.post(
        "/chat",
        json={
            "session_id": "s",
            "message": f"About ring {tiny.ring_id}: {precomputed.Q_RACE}",
            "ui_context": {"tab": "banks", "ring_id": tiny.ring_id, "chart": "race_to_exit"},
        },
    ).json()
    assert calls == []
    assert body["source"] == "precomputed" and body["backend"] == "precomputed"
    assert body["sources"] == ["get_ring_race"] and body["verified"] is False
    race = tools.get_ring_race(tiny.ring_id)
    answer = body["answer"]
    assert tiny.ring_id in answer
    assert f"{race['exit_offset_minutes']:g} minutes after the ring began" in answer
    assert "Single-bank first flagged it at T=180 minutes, 80 minutes after" in answer
    assert "Cross-bank first flagged it at T=5 minutes, 95 minutes before" in answer
    assert "Cross-bank flagged it 175 minutes earlier than single-bank" in answer


def test_both_scopes_flagging_together_is_said_plainly(
    tiny: Tiny,  # noqa: F811
    out_dir: Path,  # noqa: F811
) -> None:
    _publish(out_dir, tiny.ring_id, ring_race.entry(95.42, TOGETHER, False))
    canned = precomputed.race_answer(data.default_store(), tiny.ring_id)
    assert canned is not None
    assert (
        "Both flagged it at the same horizon, so joining did not make it earlier." in canned.answer
    )


@pytest.mark.parametrize(
    "entry", [ring_race.entry(95.0, TOGETHER, True), ring_race.entry(None, None, None)]
)
def test_an_ambiguous_or_untimed_cluster_is_left_to_the_model(
    tiny: Tiny,  # noqa: F811
    out_dir: Path,  # noqa: F811
    entry: dict,
) -> None:
    _publish(out_dir, tiny.ring_id, entry)
    store = data.default_store()
    assert precomputed.lookup(precomputed.Q_RACE, store, tiny.ring_id) is None


def test_no_ring_an_unknown_ring_or_no_artifact_leaves_the_race_question_to_the_model(
    tiny: Tiny,  # noqa: F811
    out_dir: Path,  # noqa: F811
) -> None:
    store = data.default_store()
    assert precomputed.lookup(precomputed.Q_RACE, store, None) is None
    assert precomputed.lookup(precomputed.Q_RACE, store, tiny.ring_id) is None  # nothing published
    _publish(out_dir, tiny.ring_id, ring_race.entry(100.0, HELPS, False))
    assert precomputed.lookup(precomputed.Q_RACE, store, "ring-no-such-ring") is None
    assert precomputed.lookup(precomputed.Q_RACE, store, tiny.ring_id) is not None


def test_the_race_question_is_not_a_plain_suggested_question() -> None:
    assert not precomputed.is_suggested(precomputed.Q_RACE)
    assert precomputed.normalise(precomputed.Q_RACE) in {
        precomputed.normalise(f"About ring ring-x: {precomputed.Q_RACE}")
    }
