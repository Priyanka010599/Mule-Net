"""``POST /chat/jobs`` and ``GET /chat/jobs/{request_id}``.

Offline and free: the model is replaced by a stub whose latency and tool
use the test controls, over the same six-account fixture dataset
``test_chat_claims`` uses. The router, the tools, the verifier, the
grounding check and the HTTP layer are the real code.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents import adk_agents, chat_jobs, service, tools
from agents.config import ServiceLimits
from agents.test_chat_claims import Tiny, _config, tiny  # noqa: F401 - tiny is a fixture

QUESTION = {"session_id": "s1", "message": "How many accounts are in the dataset?"}
ANSWER = {
    "answer": "6",
    "basis": "SELECT COUNT(*) AS n FROM accounts",
    "unit": "accounts",
    "claims": [
        {"id": "c1", "kind": "table_row_count", "subject": "accounts", "value": 6, "text": ""}
    ],
}
ROUTE = json.dumps({"destination": "analyst_qa", "subject_id": None, "reason": "a count"})


class StubModel:
    """A model with gates: each phase blocks until the test releases it."""

    def __init__(self, delay_s: float = 0.0, answer: Any = ANSWER) -> None:
        self.delay_s = delay_s
        self.answer = answer
        self.router_gate = threading.Event()
        self.answer_gate = threading.Event()
        self.router_gate.set()
        self.answer_gate.set()
        self.prompts: list[tuple[str, str]] = []

    def __call__(
        self,
        name: str,
        message: str,
        config: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        self.prompts.append((name, message))
        time.sleep(self.delay_s)
        if name == "router":
            self.router_gate.wait(10)
            return ROUTE
        # Tools run on a bare thread, as ADK's synchronous Runner does.
        worker = threading.Thread(
            target=lambda: adk_agents._model_tool(tools.run_sql, sink)(
                "SELECT COUNT(*) AS n FROM accounts"
            )
        )
        worker.start()
        worker.join()
        self.answer_gate.wait(10)
        return json.dumps(self.answer)


def _app(monkeypatch: pytest.MonkeyPatch, model: StubModel, **limits: Any) -> TestClient:
    monkeypatch.setattr(adk_agents, "_run_agent", model)
    monkeypatch.setattr(service, "missing_tables", lambda _config: [])
    bounds = {"session_limit": 1000, "ip_limit": 1000, "window_s": 60.0, "daily_cap": 10_000}
    bounds.update(limits)
    return TestClient(service.create_app(_config(), ServiceLimits(**bounds)))


def _wait_for(client: TestClient, request_id: str, predicate: Any, timeout: float = 10) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/chat/jobs/{request_id}").json()
        if predicate(body):
            return body
        time.sleep(0.02)
    raise AssertionError(f"timed out; last body {body}")


def test_post_returns_at_once_while_the_model_is_still_working(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    model = StubModel()
    model.router_gate.clear()
    with _app(monkeypatch, model) as client:
        started = time.perf_counter()
        response = client.post("/chat/jobs", json=QUESTION)
        elapsed = time.perf_counter() - started
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "working"
        assert body["request_id"]
        assert elapsed < 2.0
        model.router_gate.set()
        _wait_for(client, body["request_id"], lambda b: b["status"] == "done")


def test_stages_advance_routing_querying_writing_then_done(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    model = StubModel()
    model.router_gate.clear()
    model.answer_gate.clear()
    with _app(monkeypatch, model) as client:
        request_id = client.post("/chat/jobs", json=QUESTION).json()["request_id"]
        routing = client.get(f"/chat/jobs/{request_id}").json()
        assert (routing["status"], routing["stage"]) == ("working", "routing")

        model.router_gate.set()
        # The tool ran (so provenance says "writing": the model is composing)
        # and the answer is held back by the gate.
        writing = _wait_for(client, request_id, lambda b: b["stage"] == "writing")
        assert writing["status"] == "working"
        assert "answer" not in writing

        model.answer_gate.set()
        done = _wait_for(client, request_id, lambda b: b["status"] == "done")
        assert done["error"] is None


def test_a_finished_job_carries_exactly_what_post_chat_returns(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    with _app(monkeypatch, StubModel()) as client:
        sync = client.post("/chat", json=QUESTION).json()
        request_id = client.post("/chat/jobs", json=QUESTION).json()["request_id"]
        done = _wait_for(client, request_id, lambda b: b["status"] == "done")
    for key in ("answer", "backend", "verified", "sources", "regulation_citations"):
        assert done[key] == sync[key], key
    assert done["verified"] is True
    assert done["answer"] == "6 accounts"
    assert isinstance(done["latency_ms"], int)


def test_post_chat_is_unchanged_by_the_job_routes(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    with _app(monkeypatch, StubModel()) as client:
        body = client.post("/chat", json=QUESTION).json()
    # The original six fields are all still there (new ones are additive).
    assert {
        "answer", "backend", "verified", "sources", "regulation_citations", "latency_ms"
    } <= set(body)


def test_a_failed_job_carries_the_pinned_error_envelope(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    # Two empty completions end the attempts: 409 no_tool_evidence.

    def empty(name: str, *args: Any, **kwargs: Any) -> str:
        return ROUTE if name == "router" else ""

    monkeypatch.setattr(adk_agents, "_run_agent", empty)
    monkeypatch.setattr(service, "missing_tables", lambda _config: [])
    limits = ServiceLimits(session_limit=1000, ip_limit=1000, window_s=60.0, daily_cap=10_000)
    with TestClient(service.create_app(_config(), limits)) as client:
        request_id = client.post("/chat/jobs", json=QUESTION).json()["request_id"]
        failed = _wait_for(client, request_id, lambda b: b["status"] == "failed")
    assert set(failed["error"]) == {"error_type", "message", "retry_after_s"}
    assert failed["error"]["error_type"] == "no_tool_evidence"
    assert "answer" not in failed


def test_a_rate_limited_question_is_refused_by_the_post_itself(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    with _app(monkeypatch, StubModel(), session_limit=1) as client:
        first = client.post("/chat/jobs", json=QUESTION)
        assert first.status_code == 200
        second = client.post("/chat/jobs", json=QUESTION)
    assert second.status_code == 429
    assert set(second.json()) == {"error_type", "message", "retry_after_s"}
    assert second.json()["error_type"] == "rate_limited"


def test_concurrent_jobs_are_capped(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    monkeypatch.setenv("MULENET_AGENTS_MAX_JOBS", "1")
    model = StubModel()
    model.router_gate.clear()
    with _app(monkeypatch, model) as client:
        first = client.post("/chat/jobs", json=QUESTION)
        assert first.status_code == 200
        second = client.post("/chat/jobs", json={**QUESTION, "session_id": "s2"})
        assert second.status_code == 429
        assert second.json()["error_type"] == "rate_limited"
        assert second.json()["retry_after_s"] == 5
        model.router_gate.set()
        _wait_for(client, first.json()["request_id"], lambda b: b["status"] == "done")
        # The slot is free again once the job finishes.
        third = client.post("/chat/jobs", json={**QUESTION, "session_id": "s3"})
        assert third.status_code == 200


def test_an_unknown_job_is_a_404_in_the_envelope(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    with _app(monkeypatch, StubModel()) as client:
        response = client.get("/chat/jobs/nope")
    assert response.status_code == 404
    assert set(response.json()) == {"error_type", "message", "retry_after_s"}


def test_finished_jobs_expire_after_the_ttl() -> None:
    now = [100.0]
    store = chat_jobs.JobStore(ttl_s=60, max_concurrent=2, clock=lambda: now[0])
    job = store.create()
    store.finish(job, result={"answer": "x"})
    now[0] += 59
    assert store.get(job.request_id) is not None
    now[0] += 2
    assert store.get(job.request_id) is None


def test_a_working_job_does_not_expire_and_the_cap_counts_only_working() -> None:
    now = [0.0]
    store = chat_jobs.JobStore(ttl_s=1, max_concurrent=1, clock=lambda: now[0])
    job = store.create()
    now[0] += 1000
    assert store.get(job.request_id) is not None
    with pytest.raises(chat_jobs.JobCapExceeded):
        store.create()
    store.finish(job, result={})
    assert store.create() is not None


def test_stage_is_a_no_op_outside_a_job() -> None:
    chat_jobs.stage("querying")  # must not raise
    assert type(chat_jobs.new_sink()).__name__ == "RunSink"
