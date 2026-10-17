"""Chat replies say which mode answered, why live was not used, and keep
internal labels out of the answer text.

``mode`` is the machine-readable twin of ``offline_demo``; ``failure_reason``
is the agents service's own ``error_type``; ``trace`` carries the provenance
and any agent label that was written into the text. The agents service is
replaced by a stub: nothing here calls Gemini.
"""

from __future__ import annotations

from typing import Any

from ui.chat import AgentsRouter, AgentsUnavailable, MockRouter, finalize_reply
from ui.models import ChatReply, ChatRequest

LIVE = {
    "answer": "There are 45 ring groups.",
    "backend": "gemini",
    "verified": False,
    "sources": ["get_ring_summary"],
    "latency_ms": 6123,
}


class _Client:
    def __init__(self, payload: dict[str, Any] | None = None, error: Exception | None = None):
        self._payload, self._error = payload, error

    def chat(self, session_id: str, message: str, *_context: Any) -> dict[str, Any]:
        if self._error:
            raise self._error
        assert self._payload is not None
        return self._payload


def _ask(client: _Client) -> ChatReply:
    router = AgentsRouter(client, fallback=MockRouter())  # type: ignore[arg-type]
    return finalize_reply(router.reply(ChatRequest(message="What am I looking at?"), None))


def test_a_live_answer_is_mode_live_with_no_failure() -> None:
    reply = _ask(_Client(LIVE))
    assert reply.mode == "live"
    assert reply.failure_reason is None
    assert reply.offline_demo is False
    assert reply.trace.agent == "router"
    assert reply.trace.sources == ["get_ring_summary"]
    assert reply.trace.latency_ms == 6123


def test_a_malformed_response_falls_back_labelled_and_says_why() -> None:
    reply = _ask(_Client(error=AgentsUnavailable("malformed_response", None, "retries used up")))
    assert reply.mode == "offline"
    assert reply.offline_demo is True
    assert reply.offline_notice
    assert reply.failure_reason == "malformed_response"
    assert reply.failure_detail == "retries used up"
    assert reply.backend == "mock"


def test_an_unreachable_service_is_offline_with_its_reason() -> None:
    reply = _ask(_Client(error=AgentsUnavailable("unreachable")))
    assert (reply.mode, reply.failure_reason, reply.failure_detail) == (
        "offline",
        "unreachable",
        None,
    )


def test_a_mock_only_configuration_is_offline_without_a_failure() -> None:
    reply = finalize_reply(MockRouter().reply(ChatRequest(message="hello"), None))
    assert reply.mode == "offline"
    assert reply.failure_reason is None
    assert reply.offline_demo is False


def test_internal_labels_move_from_the_text_to_the_trace() -> None:
    raw = ChatReply(
        reply="[router] narrative: The ring spans 2 banks. Grounded in rings · 6123 ms server-side",
        backend="gemini",
        agent="router",
        sources=["rings"],
        latency_ms=6123,
    )
    out = finalize_reply(raw)
    assert out.reply == "The ring spans 2 banks."
    assert out.trace.labels == ["router", "narrative"]


def test_an_answer_with_no_labels_is_left_untouched() -> None:
    raw = ChatReply(reply="The narrative says nothing here.", backend="gemini", agent="router")
    assert finalize_reply(raw).reply == "The narrative says nothing here."
    assert finalize_reply(raw).trace.labels == []


def test_each_agents_error_type_has_its_own_distinct_notice() -> None:
    """A deliberate decline must not read like a failure, nor like a parse error."""
    from ui.chat import offline_notice

    notices = {
        et: offline_notice(et)
        for et in ("no_tool_evidence", "unverified", "data_unavailable", "malformed_response")
    }
    assert "The live agent failed (" not in "".join(notices.values())  # none hit the fallback
    assert len(set(notices.values())) == 4
    assert "declined to publish an answer" in notices["no_tool_evidence"]
    assert "without running any tool" in notices["no_tool_evidence"]
    assert "declined to publish an answer" in notices["unverified"]
    assert "checked" in notices["unverified"]
    assert "not loaded" in notices["data_unavailable"]
    assert "could not parse" in notices["malformed_response"]
    # Declines are not described as failures or parsing problems.
    for et in ("no_tool_evidence", "unverified"):
        assert "parse" not in notices[et] and "internal error" not in notices[et]
