"""``/api/chat`` contract additions: precomputed answers, metadata, errors, context."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ui.chat import AgentsClient, AgentsRouter, AgentsUnavailable
from ui.models import ChatRequest

PRECOMPUTED = {
    "answer": "There are 45 rings in the dataset.",
    "backend": "precomputed",
    "verified": False,
    "sources": ["rings"],
    "latency_ms": 3,
    "source": "precomputed",
}


class StubClient:
    """Stands in for AgentsClient: records the call, returns or raises."""

    def __init__(self, result: dict[str, Any] | AgentsUnavailable) -> None:
        self.result = result
        self.calls: list[tuple[Any, ...]] = []

    def chat(
        self, session_id: str, message: str, ui_context: dict[str, str] | None = None
    ) -> dict[str, Any]:
        self.calls.append((session_id, message, ui_context))
        if isinstance(self.result, AgentsUnavailable):
            raise self.result
        return self.result


def test_a_precomputed_answer_is_accepted_and_labelled() -> None:
    AgentsClient._validated(dict(PRECOMPUTED))  # backend "precomputed" is allowed
    reply = AgentsRouter(StubClient(PRECOMPUTED)).reply(  # type: ignore[arg-type]
        ChatRequest(message="How many rings are in the dataset?"), None
    )
    assert reply.source == "precomputed"
    assert reply.backend == "precomputed"
    assert reply.offline_demo is False


def test_any_other_backend_is_still_refused() -> None:
    with pytest.raises(AgentsUnavailable):
        AgentsClient._validated({**PRECOMPUTED, "backend": "mock"})


# -- ui_context forwarding ------------------------------------------------


def test_ui_context_is_forwarded_with_the_open_ring_filled_in() -> None:
    stub = StubClient({**PRECOMPUTED, "source": "model", "backend": "gemini"})
    request = ChatRequest(message="What am I looking at?", ui_context={"tab": "joint"})  # type: ignore[arg-type]
    AgentsRouter(stub).reply(request, SimpleNamespace(ring_id="ring-1"))  # type: ignore[arg-type]
    assert stub.calls[0][2] == {"tab": "joint", "ring_id": "ring-1"}


def test_the_context_ring_wins_over_the_open_ring() -> None:
    stub = StubClient(PRECOMPUTED)
    request = ChatRequest(message="q", ui_context={"ring_id": "ring-ctx"})  # type: ignore[arg-type]
    AgentsRouter(stub).reply(request, SimpleNamespace(ring_id="ring-ctx"))  # type: ignore[arg-type]
    assert stub.calls[0][2] == {"ring_id": "ring-ctx"}


def test_a_plain_question_sends_no_ui_context() -> None:
    stub = StubClient(PRECOMPUTED)
    AgentsRouter(stub).reply(ChatRequest(message="hello"), None)  # type: ignore[arg-type]
    assert stub.calls[0][2] is None


# -- edge limits, request id, degraded error ------------------------------


def test_chat_is_rate_limited_per_client_at_the_edge_with_the_envelope() -> None:
    from fastapi.testclient import TestClient

    from ui.main import app

    with TestClient(app) as client:
        limiter = app.state.limiters["chat"]
        post = client.post
        statuses = [post("/api/chat", json={"message": "hi"}) for _ in range(limiter._limit + 1)]
    assert [r.status_code for r in statuses[:-1]] == [200] * limiter._limit
    last = statuses[-1]
    assert last.status_code == 429
    body = last.json()
    assert body["code"] == "rate_limited" and body["retry_after_seconds"] >= 1
    assert body["request_id"] == last.headers["X-Request-Id"]


def test_a_degraded_reply_carries_the_agent_error_additively() -> None:
    stub = StubClient(AgentsUnavailable("rate_limited", 12, "too fast"))
    reply = AgentsRouter(stub).reply(ChatRequest(message="hello"), None)  # type: ignore[arg-type]
    assert reply.offline_demo is True and reply.source == "mock"
    assert reply.error == {"code": "rate_limited", "message": "too fast", "retry_after_seconds": 12}
    assert reply.answer == reply.reply


def test_metadata_from_agents_is_passed_through() -> None:
    payload = {
        **PRECOMPUTED,
        "backend": "gemini",
        "source": "model",
        "source_details": [{"tool": "run_sql", "args_summary": "tables: rings"}],
        "timings": {"total_ms": 9, "router_ms": 4},
        "agent_trace": [{"agent": "router", "tool": "", "ms": 4}],
        "model": "m",
        "request_id": "rid-12345678",
    }
    reply = AgentsRouter(StubClient(payload)).reply(ChatRequest(message="q"), None)  # type: ignore[arg-type]
    assert (reply.answer, reply.model, reply.request_id) == (payload["answer"], "m", "rid-12345678")
    assert reply.timings["router_ms"] == 4 and reply.source_details[0]["tool"] == "run_sql"
    assert reply.sources == ["rings"]


# -- chart context ---------------------------------------------------------


def test_the_chart_is_forwarded_in_ui_context() -> None:
    stub = StubClient(PRECOMPUTED)
    request = ChatRequest(
        message="Explain this chart",
        ui_context={"tab": "metrics", "chart": "false_alerts_per_catch_t5"},  # type: ignore[arg-type]
    )
    AgentsRouter(stub).reply(request, None)  # type: ignore[arg-type]
    assert stub.calls[0][2] == {"tab": "metrics", "chart": "false_alerts_per_catch_t5"}


def test_a_bad_chart_is_a_422_with_a_field_level_error_and_nothing_is_forwarded() -> None:
    from fastapi.testclient import TestClient

    from ui.main import app

    with TestClient(app) as client:
        response = client.post(
            "/api/chat", json={"message": "q", "ui_context": {"chart": "not-a-chart"}}
        )
    assert response.status_code == 422
    body = response.json()
    assert any(
        e["field"].endswith("chart") and "chart must be one of" in e["message"]
        for e in body["errors"]
    )
