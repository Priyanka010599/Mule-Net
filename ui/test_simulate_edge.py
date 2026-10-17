"""Item 3 at the public edge: a structured 429 with Retry-After, and the agents
service's field-level 422 passed through instead of dressed up as an outage."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from ui import test_chat_agents as fake
from ui.chat import AgentsRouter
from ui.errors import client_ip
from ui.main import app
from ui.ratelimit import SlidingWindow
from ui.test_chat_agents import fake_agents  # noqa: F401 - pytest fixture

VALIDATION_422 = (
    422,
    {
        "error_type": "validation_error",
        "message": "one or more fields are out of range",
        "retry_after_s": None,
        "errors": [
            {"field": "collectors", "message": "must be between 1 and 4", "low": 1, "high": 4}
        ],
    },
)


@pytest.fixture
def edge(monkeypatch: pytest.MonkeyPatch, fake_agents: Any) -> Any:  # noqa: F811
    monkeypatch.setenv("MULENET_UI_SIMULATE_RATE_PER_MIN", "2")
    monkeypatch.setitem(fake.ERRORS, "validation", VALIDATION_422)
    with TestClient(app) as client:
        app.state.chat = AgentsRouter(fake._client(fake_agents))
        fake_agents.simulate_mode = fake.SIMULATE_HAPPY
        yield client, fake_agents


def test_the_third_request_in_a_minute_is_a_structured_429(edge: Any) -> None:
    client, _ = edge
    assert client.post("/api/simulate", json={"collectors": 1}).status_code == 200
    assert client.post("/api/simulate", json={"collectors": 1}).status_code == 200
    limited = client.post("/api/simulate", json={"collectors": 1})
    assert limited.status_code == 429
    body = limited.json()
    assert body["code"] == "rate_limited"
    assert isinstance(body["retry_after_seconds"], int) and body["retry_after_seconds"] >= 1
    assert limited.headers["Retry-After"] == str(body["retry_after_seconds"])
    assert body["message"] and body["detail"] == body["message"]


def test_the_limit_is_per_client_and_keyed_on_the_unforgeable_address(edge: Any) -> None:
    client, _ = edge

    def post(forwarded: str) -> int:
        return client.post(
            "/api/simulate", json={}, headers={"X-Forwarded-For": forwarded}
        ).status_code

    assert [post("1.1.1.1, 9.9.9.9") for _ in range(2)] == [200, 200]
    # Same real (rightmost) address behind a different forged leftmost one: still limited.
    assert post("2.2.2.2, 9.9.9.9") == 429
    # A genuinely different client is not.
    assert post("3.3.3.3, 8.8.8.8") == 200


def test_a_validation_422_from_agents_reaches_the_browser_as_a_422(edge: Any) -> None:
    client, agents = edge
    agents.simulate_mode = "validation"
    response = client.post("/api/simulate", json={"collectors": 99})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["errors"][0]["field"] == "collectors"
    assert "offline_demo" not in body  # not an outage dressed up as a mock verdict


def test_redteam_payload_lists_each_strategy_for_the_counter() -> None:
    with TestClient(app) as client:
        body = client.get("/api/redteam").json()
    rows = body["strategies"]
    assert [r["name"] for r in rows] == [v["name"] for v in body["verdicts"]]
    assert len(rows) == body["strategies_run"]
    for row, verdict in zip(rows, body["verdicts"], strict=True):
        assert row["targets_clause"] == verdict["aimed_at_clause"]
        assert row["result"] == verdict["caught_by"]
        assert row["caught"] == verdict["caught"]
        # A miss on the clause it aimed at is the designed one.
        assert row["is_designed_miss"] == (
            not row["caught"] and row["targets_clause"] in row["failing_clauses"]
        )


def test_sliding_window_frees_up_after_the_window() -> None:
    now = [0.0]
    window = SlidingWindow(2, 10.0, clock=lambda: now[0])
    assert window.check("a") is None
    assert window.check("a") is None
    assert window.check("a") == 10
    now[0] = 4.0
    assert window.check("a") == 6
    assert window.check("b") is None  # another key has its own budget
    now[0] = 10.1
    assert window.check("a") is None


def test_client_ip_prefers_the_rightmost_forwarded_entry() -> None:
    class _Req:
        headers = {"x-forwarded-for": "6.6.6.6, 7.7.7.7"}  # noqa: RUF012
        client = None

    assert client_ip(_Req()) == "7.7.7.7"  # type: ignore[arg-type]
