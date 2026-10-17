"""``POST /chat`` reply metadata, request ids, the log line and the token cap."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agents import adk_agents, data, service
from agents.config import AgentsConfig, ServiceLimits
from agents.limits import RequestGuard

SECRET_QUESTION = "zq-secret-question-text how many rings"
ACCOUNT = "acct-hash-0123456789"


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rings = pd.DataFrame(
        {"ring_id": ["r1"], "members": [[{"alertable": True}]], "banks_involved": [["BANK_A"]]}
    )
    monkeypatch.setattr(data.DataStore, "table", lambda _self, _name: rings)
    monkeypatch.setattr(data.DataStore, "data_version", lambda _self, names=(): "c")
    monkeypatch.setattr(service, "missing_tables", lambda _c: [])
    state: dict[str, Any] = {"answer": "There are 12 rings.", "fail": None}

    def fake_run_agent(
        name: str,
        message: str,
        config: Any = None,
        sink: set[str] | None = None,
        citations: Any = None,
    ) -> str:
        if state["fail"] is not None:
            raise state["fail"]
        usage = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=5,
            total_token_count=125,
        )
        adk_agents._accumulate_usage(SimpleNamespace(usage_metadata=usage, author=name))
        if name == "router":
            return json.dumps({"destination": "analyst_qa", "subject_id": None, "reason": ""})
        if sink is not None:
            sink.update({"run_sql", "rings"})
            # The grounding check refuses a figure no tool returned, so the
            # fake tool "returns" the number the canned answer states.
            if hasattr(sink, "record_result"):
                sink.record_result("run_sql", [{"n": 12, "banks": 3}])
        return json.dumps({"answer": state["answer"], "basis": "", "unit": ""})

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    return state


def _client(**limits: Any) -> TestClient:
    base: dict[str, Any] = {
        "session_limit": 1000,
        "ip_limit": 1000,
        "window_s": 60.0,
        "daily_cap": 10_000,
    }
    guard = RequestGuard(ServiceLimits(**{**base, **limits}))
    app = service.create_app(AgentsConfig.from_env(), guard=guard, require_data=False)
    return TestClient(app)


def _post(client: TestClient, message: str = "how many rings", **extra: Any) -> Any:
    return client.post("/chat", json={"session_id": "s", "message": message}, **extra)


def test_reply_separates_prose_from_metadata(model: dict[str, Any]) -> None:
    response = _post(_client())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "There are 12 rings."
    assert body["sources"] == ["rings", "run_sql"]  # unchanged plain list
    assert body["source_details"] == [{"tool": "run_sql", "args_summary": "tables: rings"}]
    assert set(body["timings"]) == {"total_ms", "router_ms", "answer_ms"}
    assert [t["agent"] for t in body["agent_trace"]] == ["router", "analyst_qa"]
    assert body["agent_trace"][1]["tool"] == "run_sql"
    assert body["model"] and body["source"] == "model"
    assert body["request_id"] == response.headers["X-Request-Id"]


def test_internal_labels_never_reach_the_answer(model: dict[str, Any]) -> None:
    model["answer"] = "Router: Narrative: Ring r1 has 3 banks. (Grounded in get_ring_summary)"
    answer = _post(_client()).json()["answer"]
    assert answer == "Ring r1 has 3 banks."
    for label in ("router", "narrative", "grounded in"):
        assert label not in answer.lower()


def test_request_id_header_honoured_minted_and_on_errors(model: dict[str, Any]) -> None:
    client = _client()
    assert _post(client, headers={"X-Request-Id": "abc-12345678"}).headers["X-Request-Id"] == (
        "abc-12345678"
    )
    minted = _post(client, headers={"X-Request-Id": "bad id!"}).headers["X-Request-Id"]
    assert minted != "bad id!" and len(minted) >= 8
    too_long = client.post("/chat", json={"session_id": "s", "message": "x" * 5000})
    assert too_long.status_code == 422 and "X-Request-Id" in too_long.headers


def test_one_log_line_with_counts_but_no_question_or_ids(
    model: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    response = _post(_client(), f"{SECRET_QUESTION} {ACCOUNT}")
    assert response.status_code == 200
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("chat_request ")]
    assert len(lines) == 1
    rec = json.loads(lines[0].removeprefix("chat_request "))
    assert rec["request_id"] == response.headers["X-Request-Id"]
    assert rec["status"] == 200 and rec["tools"] == ["run_sql"]
    assert rec["tokens"]["total"] == 250 and rec["question_chars"] == len(
        f"{SECRET_QUESTION} {ACCOUNT}"
    )
    assert rec["latency_ms"] >= 0
    everything = " ".join(r.getMessage() for r in caplog.records)
    assert SECRET_QUESTION not in everything and ACCOUNT not in everything


def test_failures_are_logged_with_status_and_still_no_question(
    model: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    model["fail"] = OSError("socket closed")
    response = _post(_client(), SECRET_QUESTION)
    assert response.status_code == 503
    assert set(response.json()) == {"error_type", "message", "retry_after_s"}
    msgs = [r.getMessage() for r in caplog.records]
    line = next(m for m in msgs if m.startswith("chat_request "))
    rec = json.loads(line.removeprefix("chat_request "))
    assert rec["status"] == 503 and rec["error_type"] == "upstream_unavailable"
    assert SECRET_QUESTION not in " ".join(r.getMessage() for r in caplog.records)


def test_a_429_is_logged_and_has_the_envelope(
    model: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    client = _client(session_limit=1)
    assert _post(client).status_code == 200
    limited = _post(client)
    assert limited.status_code == 429
    assert limited.json()["error_type"] == "rate_limited"
    assert "Retry-After" in limited.headers
    assert any('"status": 429' in r.getMessage() for r in caplog.records)


def test_daily_token_cap_refuses_after_the_budget_is_spent(model: dict[str, Any]) -> None:
    client = _client(daily_token_cap=200)
    assert _post(client).status_code == 200  # spends 250 tokens
    refused = _post(client)
    assert refused.status_code == 429
    assert refused.json()["error_type"] == "daily_cap"


def test_message_length_default_is_modest() -> None:
    assert ServiceLimits().max_message_chars == 1000
