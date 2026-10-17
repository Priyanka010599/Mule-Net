"""The ``agents`` chat backend, against a local fake agents service.

Nothing here mocks ``urllib``. A fake HTTP server implementing
``docs/agents_api.md`` -- the happy path *and* every documented error
shape -- is started on a loopback port, and ``ui/chat.py`` talks to it
over a real socket. That is a truer test: it exercises the request body,
the ``Authorization`` header, status-code handling, JSON decoding and the
timeout, none of which a patched transport would check. It also means a
server-side contract mismatch shows up here rather than in a demo.

The fake deliberately does **not** implement Gemini. It returns the
documented shapes, which is all the client is allowed to depend on.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ui.chat import (
    OFFLINE_DEMO_LABEL,
    AgentsClient,
    AgentsRouter,
    AgentsUnavailable,
    MockRouter,
    build_backend,
    with_ring_context,
)
from ui.config import UiConfig
from ui.models import ChatRequest

#: A ``200`` exactly as docs/agents_api.md specifies it.
HAPPY: dict[str, Any] = {
    "answer": "ring_014 spans two banks and cashes out through one account.",
    "backend": "gemini",
    "verified": True,
    "sources": ["rings", "account_evidence"],
    "latency_ms": 1432,
}

#: A ``200`` from ``GET /screen/{id}``, matching agents/service.py's ScreenOut.
SCREEN_HAPPY: dict[str, Any] = {
    "hashed_account_id": "acct-x",
    "flagged": True,
    "ring_ids": ["ring-x-00148"],
    "reason": "account is a member of a detected mule ring",
    "risk_score": 0.87,
}

#: A ``200`` from ``GET /simulate/bounds``, matching agents.simulate.bounds_payload.
SIMULATE_BOUNDS_HAPPY: dict[str, Any] = {
    "fields": {
        "collectors": {"low": 1, "high": 4, "default": 1, "integer": True},
        "mules_per_collector": {"low": 1, "high": 40, "default": 8, "integer": True},
    },
    "evadable_clauses": ["in_degree", "out_degree"],
}

#: A ``200`` from ``POST /simulate``, matching agents/service.py's SimulateOut.
SIMULATE_HAPPY: dict[str, Any] = {
    "name": "judge-built pattern",
    "description": "2 collector(s), 8 mule(s) feeding each.",
    "targets": "judge-specified pattern",
    "aimed_at_clause": "in_degree",
    "why_it_should_evade": "Hand-picked live.",
    "pattern": {
        "collectors": 2,
        "mules_per_collector": 8,
        "onward_accounts": 4,
        "hop_delay_seconds": 600.0,
        "amount_per_inbound_hop": 5000.0,
        "retained_fraction": 0.05,
        "counterparty_banks": 2,
        "reuse_device": False,
        "reuse_kyc_id": False,
        "clamped": [],
    },
    "caught": True,
    "caught_by": "velocity_ring",
    "reasoning": "Caught by the stage-1 velocity rule, and a ring formed.",
    "flagged_accounts": 2,
    "flagged_collectors": 2,
    "rings_formed": 1,
    "stage2_alertable": 0,
    "stage2_context": 0,
    "clause_values": {
        "acct-1": {
            "in_degree": {"value": 8, "passes": True, "requirement": "x >= 16"},
        },
    },
}

#: A ``200`` from ``GET /subset/options``, matching agents.subset.options_payload.
SUBSET_OPTIONS_HAPPY: dict[str, Any] = {
    "horizons": [
        {"key": "30m", "label": "30 minutes after the first transfer", "minutes": 30},
        {"key": "24h", "label": "1 day after the first transfer", "minutes": 1440},
    ],
    "banks": ["all", "BANK_A", "BANK_B", "BANK_C"],
}

#: A ``200`` from ``POST /subset/load``, matching agents/service.py's SubsetLoadOut.
SUBSET_LOAD_HAPPY: dict[str, Any] = {
    "horizon": "24h",
    "horizon_label": "1 day after the first transfer",
    "bank": "all",
    "cutoff": "2025-01-02T00:00:13+00:00",
    "transfers_considered": 14481,
    "accounts_in_scope": 30000,
    "load_ms": 991,
}

#: A ``200`` from ``POST /subset/score``, matching agents/service.py's SubsetScoreOut.
SUBSET_SCORE_HAPPY: dict[str, Any] = {
    "horizon": "7d",
    "horizon_label": "7 days after the first transfer",
    "bank": "BANK_B",
    "cutoff": "2025-01-08T00:00:13+00:00",
    "transfers_considered": 118406,
    "accounts_in_scope": 10000,
    "cross_bank_flagged": 10,
    "single_bank_flagged": 0,
    "flagged_accounts": [
        {
            "hashed_account_id": "acct-collector",
            "bank_id": "BANK_B",
            "caught_cross_bank": True,
            "caught_single_bank": False,
            "in_degree": 33,
            "out_degree": 11,
            "distinct_counterparty_banks": 3,
            "passthrough_ratio": 1.10,
            "passthrough_speed_seconds": 1260.0,
        }
    ],
    "flagged_accounts_truncated": False,
    "load_ms": 2028,
    "score_ms": 2,
    "total_ms": 2030,
}

#: A ``200`` from ``POST /live-proof/start``, matching agents/service.py's
#: LiveProofStartOut (the not-capped case).
LIVE_PROOF_START_HAPPY: dict[str, Any] = {"capped": False, "run_id": "abc123"}

#: A ``200`` from ``POST /live-proof/start`` once the cap is hit.
LIVE_PROOF_START_CAPPED: dict[str, Any] = {
    "capped": True,
    "run_id": None,
    "runs_today": 10,
    "limit": 10,
}

#: A ``200`` from ``GET /live-proof/status/{run_id}``, matching
#: agents/service.py's LiveProofStatusOut.
LIVE_PROOF_STATUS_HAPPY: dict[str, Any] = {
    "run_id": "abc123",
    "stage": "vm_running",
    "extra": {"vm_name": "mulenet-live-proof-abc123"},
    "result": None,
    "error": None,
}

#: Every documented error row, keyed by the ``error_type`` the client
#: must come back with. ``retry_after_s`` is an integer where the caller
#: should wait and ``null`` otherwise, as the contract states.
ERRORS: dict[str, tuple[int, dict[str, Any]]] = {
    "rate_limited": (
        429,
        {"error_type": "rate_limited", "message": "too fast", "retry_after_s": 12},
    ),
    "daily_cap": (
        429,
        {"error_type": "daily_cap", "message": "cap reached", "retry_after_s": 3600},
    ),
    "timeout": (
        504,
        {"error_type": "timeout", "message": "gemini timed out", "retry_after_s": None},
    ),
    "upstream_unavailable": (
        503,
        {
            "error_type": "upstream_unavailable",
            "message": "vertex refused",
            "retry_after_s": None,
        },
    ),
    "internal": (
        500,
        {"error_type": "internal", "message": "something broke", "retry_after_s": None},
    ),
    "not_found": (
        404,
        {"error_type": "not_found", "message": "no such run", "retry_after_s": None},
    ),
}


class _FakeAgents(BaseHTTPRequestHandler):
    """``GET /health`` and ``POST /chat`` per docs/agents_api.md."""

    #: Set per test on the server object: ``None`` for the happy path, or
    #: a key of :data:`ERRORS`, or ``"garbage"`` / ``"not_gemini"``.
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return  # pytest does not need an access log

    def _send(self, status: int, body: Any) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/simulate/bounds":
            server: Any = self.server
            server.seen_authorization = self.headers.get("Authorization")
            mode = server.simulate_bounds_mode
            if isinstance(mode, str) and mode in ERRORS:
                status, payload = ERRORS[mode]
                self._send(status, payload)
            elif isinstance(mode, dict):
                self._send(200, mode)
            else:
                self._send(200, SIMULATE_BOUNDS_HAPPY)
        elif self.path == "/subset/options":
            server: Any = self.server
            server.seen_authorization = self.headers.get("Authorization")
            mode = server.subset_options_mode
            if isinstance(mode, str) and mode in ERRORS:
                status, payload = ERRORS[mode]
                self._send(status, payload)
            elif isinstance(mode, dict):
                self._send(200, mode)
            else:
                self._send(200, SUBSET_OPTIONS_HAPPY)
        elif self.path.startswith("/live-proof/status/"):
            server: Any = self.server
            server.seen_authorization = self.headers.get("Authorization")
            server.seen_live_proof_status_id = self.path.removeprefix("/live-proof/status/")
            mode = server.live_proof_status_mode
            if isinstance(mode, str) and mode in ERRORS:
                status, payload = ERRORS[mode]
                self._send(status, payload)
            elif isinstance(mode, dict):
                self._send(200, mode)
            else:
                self._send(200, LIVE_PROOF_STATUS_HAPPY)
        elif self.path.startswith("/screen/"):
            server: Any = self.server
            server.seen_authorization = self.headers.get("Authorization")
            server.seen_screen_id = self.path.removeprefix("/screen/")
            mode = server.screen_mode
            if isinstance(mode, str) and mode in ERRORS:
                status, payload = ERRORS[mode]
                self._send(status, payload)
            elif mode == "garbage":
                raw_bytes = b"<html>not json</html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(raw_bytes)))
                self.end_headers()
                self.wfile.write(raw_bytes)
            elif isinstance(mode, dict):
                self._send(200, mode)
            else:
                self._send(200, SCREEN_HAPPY)
        else:
            self._send(404, {"error_type": "no_case", "message": "", "retry_after_s": None})

    def do_POST(self) -> None:
        server: Any = self.server
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        server.seen_authorization = self.headers.get("Authorization")
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if self.path == "/simulate":
            server.seen_simulate_body = body
            mode = server.simulate_mode
            if isinstance(mode, str) and mode in ERRORS:
                status, payload = ERRORS[mode]
                self._send(status, payload)
            elif isinstance(mode, dict):
                self._send(200, mode)
            else:
                self._send(200, SIMULATE_HAPPY)
            return
        if self.path == "/subset/load":
            server.seen_subset_load_body = body
            mode = server.subset_load_mode
            if isinstance(mode, str) and mode in ERRORS:
                status, payload = ERRORS[mode]
                self._send(status, payload)
            elif isinstance(mode, dict):
                self._send(200, mode)
            else:
                self._send(200, SUBSET_LOAD_HAPPY)
            return
        if self.path in ("/live-proof/start", "/live-proof/tampered-start"):
            mode = server.live_proof_start_mode
            if isinstance(mode, str) and mode in ERRORS:
                status, payload = ERRORS[mode]
                self._send(status, payload)
            elif isinstance(mode, dict):
                self._send(200, mode)
            elif mode == "capped":
                self._send(200, LIVE_PROOF_START_CAPPED)
            else:
                self._send(200, LIVE_PROOF_START_HAPPY)
            return
        if self.path == "/subset/score":
            server.seen_subset_score_body = body
            mode = server.subset_score_mode
            if isinstance(mode, str) and mode in ERRORS:
                status, payload = ERRORS[mode]
                self._send(status, payload)
            elif isinstance(mode, dict):
                self._send(200, mode)
            else:
                self._send(200, SUBSET_SCORE_HAPPY)
            return
        if self.path != "/chat":
            self._send(404, {"error_type": "no_case", "message": "", "retry_after_s": None})
            return
        # 422 is FastAPI's default shape, which is not the error envelope.
        if not isinstance(body, dict) or not {"session_id", "message"} <= set(body):
            self._send(422, {"detail": [{"loc": ["body"], "msg": "field required"}]})
            return
        server.seen_body = body
        mode = server.mode
        if mode in ERRORS:
            status, payload = ERRORS[mode]
            self._send(status, payload)
        elif mode == "garbage":
            raw_bytes = b"<html>not json</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(raw_bytes)))
            self.end_headers()
            self.wfile.write(raw_bytes)
        elif mode == "not_gemini":
            # The one thing the contract forbids the service to do.
            self._send(200, {**HAPPY, "backend": "mock"})
        elif mode == "cloud_run_403":
            # Cloud Run's own rejection of a missing or invalid token,
            # produced before application code runs: not the JSON envelope.
            raw_bytes = b"<html><title>403 Forbidden</title></html>"
            self.send_response(403)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(raw_bytes)))
            self.end_headers()
            self.wfile.write(raw_bytes)
        else:
            self._send(200, HAPPY)


@pytest.fixture
def fake_agents() -> Iterator[Any]:
    """A fake agents service on a loopback port, torn down after the test."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeAgents)
    server.mode = None  # type: ignore[attr-defined]
    server.screen_mode = None  # type: ignore[attr-defined]
    server.simulate_mode = None  # type: ignore[attr-defined]
    server.simulate_bounds_mode = None  # type: ignore[attr-defined]
    server.subset_options_mode = None  # type: ignore[attr-defined]
    server.subset_load_mode = None  # type: ignore[attr-defined]
    server.subset_score_mode = None  # type: ignore[attr-defined]
    server.live_proof_start_mode = None  # type: ignore[attr-defined]
    server.live_proof_status_mode = None  # type: ignore[attr-defined]
    server.seen_live_proof_status_id = None  # type: ignore[attr-defined]
    server.seen_body = None  # type: ignore[attr-defined]
    server.seen_simulate_body = None  # type: ignore[attr-defined]
    server.seen_subset_load_body = None  # type: ignore[attr-defined]
    server.seen_subset_score_body = None  # type: ignore[attr-defined]
    server.seen_screen_id = None  # type: ignore[attr-defined]
    server.seen_authorization = None  # type: ignore[attr-defined]
    server.url = f"http://127.0.0.1:{server.server_address[1]}"  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _client(server: Any, **kwargs: Any) -> AgentsClient:
    return AgentsClient(
        base_url=server.url,
        timeout_s=5.0,
        token_provider=lambda audience: f"id-token-for-{audience}",
        **kwargs,
    )


# ── the contract, field by field ──────────────────────────────────


def test_the_request_body_is_exactly_session_id_and_message(fake_agents: Any) -> None:
    _client(fake_agents).chat("sess-1", "trace the money flow")
    assert fake_agents.seen_body == {
        "session_id": "sess-1",
        "message": "trace the money flow",
    }


def test_the_open_ring_is_named_in_the_question() -> None:
    """Without it, "trace the money flow" has no subject and falls to the mock."""
    assert (
        with_ring_context("trace the money flow", "ring-x-00148")
        == "About ring ring-x-00148: trace the money flow"
    )


def test_a_question_that_already_names_the_ring_is_left_alone() -> None:
    message = "how many banks does ring-x-00148 span?"
    assert with_ring_context(message, "ring-x-00148") == message


def test_no_open_ring_leaves_the_question_alone() -> None:
    assert with_ring_context("trace the money flow", None) == "trace the money flow"


def test_a_200_is_carried_through_without_renaming_anything(fake_agents: Any) -> None:
    reply = AgentsRouter(_client(fake_agents)).reply(
        ChatRequest(message="trace the money flow"), None
    )
    assert reply.reply == HAPPY["answer"]
    assert reply.backend == "gemini"
    assert reply.verified is True
    assert reply.sources == ["rings", "account_evidence"]
    assert reply.latency_ms == 1432
    assert reply.offline_demo is False
    assert reply.offline_notice == ""


def test_the_health_probe_costs_no_model_call(fake_agents: Any) -> None:
    """Sanity-check the fake against the contract's other endpoint."""
    with urllib.request.urlopen(f"{fake_agents.url}/health", timeout=5) as response:
        assert json.loads(response.read()) == {"status": "ok"}


# ── screen ─────────────────────────────────────────────────────────


def test_a_flagged_screen_result_is_carried_through(fake_agents: Any) -> None:
    fake_agents.screen_mode = SCREEN_HAPPY
    result = AgentsRouter(_client(fake_agents)).screen("acct-x")
    assert result.hashed_account_id == "acct-x"
    assert result.flagged is True
    assert result.backend == "agents"
    assert result.ring_ids == ["ring-x-00148"]
    assert result.risk_score == 0.87
    assert result.offline_demo is False


def test_an_unflagged_account_carries_no_ring_ids_or_reason(fake_agents: Any) -> None:
    fake_agents.screen_mode = {
        "hashed_account_id": "acct-clean",
        "flagged": False,
        "ring_ids": [],
        "reason": "",
        "risk_score": None,
    }
    result = AgentsRouter(_client(fake_agents)).screen("acct-clean")
    assert result.flagged is False
    assert result.ring_ids == []
    assert result.risk_score is None


def test_the_screen_request_hits_the_documented_path(fake_agents: Any) -> None:
    AgentsRouter(_client(fake_agents)).screen("acct-y")
    assert fake_agents.seen_screen_id == "acct-y"


def test_an_unreachable_agents_service_degrades_screen_visibly(
    fake_agents: Any,
) -> None:
    """Same visible-degradation rule as chat: never a silent false-clear."""
    fake_agents.screen_mode = "internal"
    result = AgentsRouter(_client(fake_agents)).screen("acct-z")
    assert result.flagged is False
    assert result.backend == "mock"
    assert result.offline_demo is True
    assert OFFLINE_DEMO_LABEL in result.offline_notice.lower()


def test_the_mock_never_invents_a_flagged_verdict() -> None:
    """MockRouter has no account table to check against, so it must not guess."""
    result = MockRouter().screen("acct-anything")
    assert result.flagged is False
    assert result.backend == "mock"


# ── live evasion simulator ────────────────────────────────────────


def test_simulate_bounds_are_carried_through(fake_agents: Any) -> None:
    fake_agents.simulate_bounds_mode = SIMULATE_BOUNDS_HAPPY
    bounds = AgentsRouter(_client(fake_agents)).simulate_bounds()
    assert bounds.backend == "agents"
    assert bounds.offline_demo is False
    assert set(bounds.fields) == {"collectors", "mules_per_collector"}
    assert bounds.fields["collectors"].high == 4
    assert bounds.evadable_clauses == ["in_degree", "out_degree"]


def test_a_live_verdict_is_carried_through_without_renaming_anything(
    fake_agents: Any,
) -> None:
    fake_agents.simulate_mode = SIMULATE_HAPPY
    verdict = AgentsRouter(_client(fake_agents)).simulate({"collectors": 2})
    assert verdict.caught is True
    assert verdict.caught_by == "velocity_ring"
    assert verdict.pattern.collectors == 2
    assert verdict.flagged_collectors == 2
    assert verdict.backend == "agents"
    assert verdict.offline_demo is False


def test_the_simulate_request_body_is_forwarded_verbatim(fake_agents: Any) -> None:
    AgentsRouter(_client(fake_agents)).simulate(
        {"collectors": 3, "evades_clause": "passthrough_ratio"}
    )
    assert fake_agents.seen_simulate_body == {
        "collectors": 3,
        "evades_clause": "passthrough_ratio",
    }


def test_an_unreachable_agents_service_degrades_simulate_visibly(
    fake_agents: Any,
) -> None:
    """No silent fallback: a judge must never mistake the mock for a real run."""
    fake_agents.simulate_mode = "internal"
    verdict = AgentsRouter(_client(fake_agents)).simulate({"collectors": 2})
    assert verdict.backend == "mock"
    assert verdict.offline_demo is True
    assert verdict.caught is False
    assert OFFLINE_DEMO_LABEL in verdict.offline_notice.lower()
    # This panel has no offline equivalent, unlike chat and screen -- the
    # notice must say so rather than claiming snapshot-backed numbers.
    assert "no offline equivalent" in verdict.offline_notice.lower()


def test_an_unreachable_agents_service_degrades_bounds_visibly(
    fake_agents: Any,
) -> None:
    fake_agents.simulate_bounds_mode = "internal"
    bounds = AgentsRouter(_client(fake_agents)).simulate_bounds()
    assert bounds.backend == "mock"
    assert bounds.offline_demo is True
    assert bounds.fields == {}


def test_the_mock_never_invents_a_live_verdict() -> None:
    """MockRouter has no detector to run this pattern through, so it must say so."""
    result = MockRouter().simulate({"collectors": 2})
    assert result.caught is False
    assert result.caught_by == "unavailable"
    assert result.backend == "mock"
    assert result.offline_demo is True


# ── auth ───────────────────────────────────────────────────────────


def test_the_call_carries_a_bearer_token_minted_for_the_agents_audience(
    fake_agents: Any,
) -> None:
    """The audience is the agents service, and the token is this server's."""
    _client(fake_agents).chat("sess-1", "hello")
    assert fake_agents.seen_authorization == f"Bearer id-token-for-{fake_agents.url}"


def test_a_separate_audience_overrides_the_base_url(fake_agents: Any) -> None:
    """A load balancer in front changes the audience, not the URL."""
    _client(fake_agents, audience="https://agents.example").chat("sess-1", "hello")
    assert fake_agents.seen_authorization == "Bearer id-token-for-https://agents.example"


def test_a_token_that_cannot_be_minted_degrades_rather_than_500s(
    fake_agents: Any,
) -> None:
    def broken(_audience: str) -> str:
        raise RuntimeError("no metadata server here")

    router = AgentsRouter(
        AgentsClient(base_url=fake_agents.url, timeout_s=5.0, token_provider=broken)
    )
    reply = router.reply(ChatRequest(message="hello"), None)
    assert reply.offline_demo is True
    assert "ID token" in reply.offline_notice


def test_a_cloud_run_403_is_read_even_though_it_is_not_the_json_envelope(
    fake_agents: Any,
) -> None:
    fake_agents.mode = "cloud_run_403"
    with pytest.raises(AgentsUnavailable) as raised:
        _client(fake_agents).chat("sess-1", "hello")
    assert raised.value.error_type == "forbidden"


# ── the documented error shapes ───────────────────────────────────


@pytest.mark.parametrize("error_type", sorted(ERRORS))
def test_every_documented_error_shape_is_recognised(
    fake_agents: Any, error_type: str
) -> None:
    fake_agents.mode = error_type
    with pytest.raises(AgentsUnavailable) as raised:
        _client(fake_agents).chat("sess-1", "hello")
    assert raised.value.error_type == error_type
    expected = ERRORS[error_type][1]["retry_after_s"]
    assert raised.value.retry_after_s == expected


def test_an_unreachable_service_is_not_an_exception_the_browser_sees() -> None:
    """Port 1 on loopback refuses instantly, so this needs no timeout wait."""
    client = AgentsClient(base_url="http://127.0.0.1:1", timeout_s=2.0)
    with pytest.raises(AgentsUnavailable) as raised:
        client.chat("sess-1", "hello")
    assert raised.value.error_type == "unreachable"


def test_a_non_json_200_is_refused_rather_than_shown(fake_agents: Any) -> None:
    fake_agents.mode = "garbage"
    with pytest.raises(AgentsUnavailable) as raised:
        _client(fake_agents).chat("sess-1", "hello")
    assert raised.value.error_type == "bad_response"


def test_an_answer_not_labelled_gemini_is_never_shown_as_gemini(
    fake_agents: Any,
) -> None:
    """docs/agents_api.md: ``backend`` is the literal "gemini".

    The service must never return that label for an answer it did not get
    from Gemini on Vertex, and this client must never render one that
    arrives under any other label as though it had.
    """
    fake_agents.mode = "not_gemini"
    with pytest.raises(AgentsUnavailable) as raised:
        _client(fake_agents).chat("sess-1", "hello")
    assert raised.value.error_type == "bad_response"


@pytest.mark.parametrize(
    "missing", ["answer", "backend", "verified", "sources", "latency_ms"]
)
def test_a_reply_missing_a_declared_field_is_not_the_documented_shape(
    missing: str,
) -> None:
    with pytest.raises(AgentsUnavailable):
        AgentsClient._validated({k: v for k, v in HAPPY.items() if k != missing})


# ── the visible fallback ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("error_type", "must_say"),
    [
        ("rate_limited", "going too fast"),
        ("daily_cap", "finished for today"),
        ("timeout", "past its timeout"),
        ("upstream_unavailable", "Vertex AI was unreachable"),
        ("internal", "internal error"),
    ],
)
def test_each_failure_case_gets_its_own_wording(
    fake_agents: Any, error_type: str, must_say: str
) -> None:
    """The two 429s must not collapse into one message.

    "You are going too fast" tells a judge to wait a moment. "This demo is
    finished for today" tells them to stop. Rendering both as "rate
    limited, try again" would have them retrying into a spend cap.
    """
    fake_agents.mode = error_type
    reply = AgentsRouter(_client(fake_agents)).reply(ChatRequest(message="hello"), None)
    assert reply.offline_demo is True
    assert reply.offline_notice.startswith("Offline demo mode.")
    assert must_say in reply.offline_notice
    retry = ERRORS[error_type][1]["retry_after_s"]
    if retry:
        assert f"Retry in {retry}s." in reply.offline_notice


def test_the_two_429s_do_not_share_wording(fake_agents: Any) -> None:
    notices = {}
    for error_type in ("rate_limited", "daily_cap"):
        fake_agents.mode = error_type
        notices[error_type] = (
            AgentsRouter(_client(fake_agents))
            .reply(ChatRequest(message="hello"), None)
            .offline_notice
        )
    assert notices["rate_limited"] != notices["daily_cap"]
    assert "finished for today" not in notices["rate_limited"]
    assert "going too fast" not in notices["daily_cap"]


def test_a_degraded_answer_never_claims_it_came_from_gemini(fake_agents: Any) -> None:
    """The point of the whole exercise: no silent fallback.

    A judge who cannot tell a Gemini answer from a keyword match has been
    told something false by omission, so the degraded reply keeps the mock
    label, says so in words, and those words reach the page.
    """
    fake_agents.mode = "upstream_unavailable"
    reply = AgentsRouter(_client(fake_agents)).reply(ChatRequest(message="hello"), None)
    assert reply.backend == "mock"
    assert reply.backend != "gemini"
    assert "not from" in reply.offline_notice
    assert "Gemini" in reply.offline_notice
    # The reply still answers, so the panel is not dead -- it is honest.
    assert reply.reply


def test_the_mock_on_its_own_never_claims_gemini() -> None:
    assert MockRouter().reply(ChatRequest(message="hello"), None).backend == "mock"


def test_the_offline_label_is_the_one_the_page_renders() -> None:
    """The on-screen label and the notice come from the same constant."""
    app_js = (Path(__file__).resolve().parent / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert f"'{OFFLINE_DEMO_LABEL}'" in app_js
    assert "chat-offline" in app_js
    assert "offline_notice" in app_js


def _app_js() -> str:
    return (Path(__file__).resolve().parent / "static" / "app.js").read_text(
        encoding="utf-8"
    )


def test_the_chat_surface_carries_no_verification_indicator() -> None:
    """No verified badge on chat, in either state.

    ``verified`` reports whether ``agents/verify.py`` re-derived the
    answer's claims. On chat it is false for almost every answer because
    most questions carry nothing re-derivable -- but a verification badge
    beside a live Gemini figure reads as "machine-checked" whichever way it
    points, and a chat answer was not. The badge belongs to
    investigations/cases, whose claims genuinely were re-derived, and the
    case panel keeps it (``verified_claim_count``).
    """
    app_js = _app_js()
    # ``extra`` is the chat reply object, so this is the chat surface only.
    assert "extra.verified" not in app_js
    assert "verified_claim_count" in app_js  # still shown on the case panel


def test_a_chat_answer_says_what_it_was_grounded_in() -> None:
    """Sources are rendered as grounding, and an empty list says so."""
    app_js = _app_js()
    assert "Grounded in " in app_js
    assert "extra.sources" in app_js
    # An empty ``sources`` is valid by contract and means nothing was
    # consulted: never a trailing "Grounded in " with nothing after it.
    assert "nothing was consulted" in app_js


# ── the URL never reaches the browser ─────────────────────────────


def test_the_agents_url_is_never_served_to_client_side_code(
    monkeypatch: pytest.MonkeyPatch, fake_agents: Any
) -> None:
    """The agents service has no public invoker access.

    A browser cannot call it, so telling the browser where it is would be
    pure leakage: it invites a direct call that can only 403, and it
    publishes internal topology. The URL lives in ``ui/config.py`` and
    stays there.
    """
    secret = "agents-internal-abcdef.a.run.app"
    monkeypatch.setenv("MULENET_UI_CHAT_BACKEND", "agents")
    monkeypatch.setenv("MULENET_UI_AGENTS_URL", f"https://{secret}")
    monkeypatch.setenv("MULENET_UI_AGENTS_AUTH", "0")
    cfg = UiConfig.from_env()
    assert cfg.agents_url == f"https://{secret}"

    from ui.main import app

    with TestClient(app) as client:
        assert secret not in client.get("/").text
        assert secret not in client.get("/static/app.js").text
        body = client.post("/api/chat", json={"message": "hello"})
        assert body.status_code == 200
        assert secret not in body.text
        # It degraded, because nothing is listening on that hostname.
        assert body.json()["offline_demo"] is True


def test_a_browser_supplied_token_is_never_forwarded(
    monkeypatch: pytest.MonkeyPatch, fake_agents: Any
) -> None:
    """``ui`` presents its own identity, never the caller's.

    Forwarding a browser-supplied bearer token would make the agents
    service's access control a function of whatever a public web page
    chose to send.
    """
    monkeypatch.setenv("MULENET_UI_CHAT_BACKEND", "agents")
    monkeypatch.setenv("MULENET_UI_AGENTS_URL", fake_agents.url)
    monkeypatch.setenv("MULENET_UI_AGENTS_AUTH", "0")
    from ui.main import app

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "hello"},
            headers={"Authorization": "Bearer browser-forged-token"},
        )
    assert response.status_code == 200
    assert response.json()["backend"] == "gemini"
    # No token at all here, because MULENET_UI_AGENTS_AUTH=0 disables
    # minting -- and crucially not the browser's either.
    assert fake_agents.seen_authorization is None


def test_the_session_id_is_not_taken_from_the_request(
    monkeypatch: pytest.MonkeyPatch, fake_agents: Any
) -> None:
    """``session_id`` is a rate-limit bucket key, so ui chooses it.

    docs/agents_api.md says it grants no authority and is not identity. A
    caller free to pick it could reset the per-session limit on every
    message, and that limit is the cost control.
    """
    monkeypatch.setenv("MULENET_UI_CHAT_BACKEND", "agents")
    monkeypatch.setenv("MULENET_UI_AGENTS_URL", fake_agents.url)
    monkeypatch.setenv("MULENET_UI_AGENTS_AUTH", "0")
    from ui.main import app

    with TestClient(app) as client:
        client.post("/api/chat", json={"message": "one", "session_id": "attacker"})
        first = fake_agents.seen_body["session_id"]
        client.post("/api/chat", json={"message": "two", "session_id": "attacker-2"})
        second = fake_agents.seen_body["session_id"]
    assert first == second
    assert "attacker" not in first
    assert first.startswith("mulenet-ui-")


# ── config ────────────────────────────────────────────────────────


def test_the_agents_backend_requires_a_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing URL is a misconfiguration, not something to degrade from.

    Starting up and answering every question from the mock under an
    ``agents`` label is exactly the silent fallback this work removes.
    """
    monkeypatch.setenv("MULENET_UI_CHAT_BACKEND", "agents")
    monkeypatch.delenv("MULENET_UI_AGENTS_URL", raising=False)
    with pytest.raises(ValueError, match="MULENET_UI_AGENTS_URL"):
        UiConfig.from_env()


# ── live data-subset panel ────────────────────────────────────────


def test_subset_options_are_carried_through(fake_agents: Any) -> None:
    fake_agents.subset_options_mode = SUBSET_OPTIONS_HAPPY
    options = AgentsRouter(_client(fake_agents)).subset_options()
    assert options.backend == "agents"
    assert options.offline_demo is False
    assert [h.key for h in options.horizons] == ["30m", "24h"]
    assert options.banks == ["all", "BANK_A", "BANK_B", "BANK_C"]


def test_a_live_load_result_is_carried_through_without_renaming_anything(
    fake_agents: Any,
) -> None:
    fake_agents.subset_load_mode = SUBSET_LOAD_HAPPY
    result = AgentsRouter(_client(fake_agents)).subset_load("24h", "all")
    assert result.transfers_considered == 14481
    assert result.accounts_in_scope == 30000
    assert result.backend == "agents"
    assert result.offline_demo is False


def test_a_live_score_result_is_carried_through_without_renaming_anything(
    fake_agents: Any,
) -> None:
    fake_agents.subset_score_mode = SUBSET_SCORE_HAPPY
    result = AgentsRouter(_client(fake_agents)).subset_score("7d", "BANK_B")
    assert result.cross_bank_flagged == 10
    assert result.single_bank_flagged == 0
    assert len(result.flagged_accounts) == 1
    assert result.flagged_accounts[0].hashed_account_id == "acct-collector"
    assert result.backend == "agents"
    assert result.offline_demo is False


def test_the_subset_request_body_carries_horizon_and_bank(fake_agents: Any) -> None:
    AgentsRouter(_client(fake_agents)).subset_load("6h", "BANK_A")
    assert fake_agents.seen_subset_load_body == {"horizon": "6h", "bank": "BANK_A"}
    AgentsRouter(_client(fake_agents)).subset_score("3d", "BANK_C")
    assert fake_agents.seen_subset_score_body == {"horizon": "3d", "bank": "BANK_C"}


def test_an_unreachable_agents_service_degrades_subset_load_visibly(
    fake_agents: Any,
) -> None:
    """No silent fallback: a judge must never mistake the mock for a real run."""
    fake_agents.subset_load_mode = "internal"
    result = AgentsRouter(_client(fake_agents)).subset_load("24h", "all")
    assert result.backend == "mock"
    assert result.offline_demo is True
    assert result.transfers_considered == 0
    assert OFFLINE_DEMO_LABEL in result.offline_notice.lower()
    assert "no offline equivalent" in result.offline_notice.lower()


def test_an_unreachable_agents_service_degrades_subset_score_visibly(
    fake_agents: Any,
) -> None:
    fake_agents.subset_score_mode = "internal"
    result = AgentsRouter(_client(fake_agents)).subset_score("24h", "all")
    assert result.backend == "mock"
    assert result.offline_demo is True
    assert result.cross_bank_flagged == 0
    assert result.flagged_accounts == []
    assert "no offline equivalent" in result.offline_notice.lower()


def test_an_unreachable_agents_service_degrades_subset_options_visibly(
    fake_agents: Any,
) -> None:
    fake_agents.subset_options_mode = "internal"
    options = AgentsRouter(_client(fake_agents)).subset_options()
    assert options.backend == "mock"
    assert options.offline_demo is True
    assert options.horizons == []


def test_the_mock_never_invents_a_live_subset_result() -> None:
    """MockRouter has no feature builder or detector to run a subset
    through, so it must say so rather than guess a horizon's stats."""
    load = MockRouter().subset_load("24h", "all")
    assert load.transfers_considered == 0
    assert load.backend == "mock"
    assert load.offline_demo is True

    score = MockRouter().subset_score("24h", "all")
    assert score.cross_bank_flagged == 0
    assert score.flagged_accounts == []
    assert score.backend == "mock"
    assert score.offline_demo is True


def test_build_backend_returns_the_mock_by_default() -> None:
    assert isinstance(build_backend(UiConfig.from_env()), MockRouter)


def test_build_backend_wires_the_real_client_for_agents(
    monkeypatch: pytest.MonkeyPatch, fake_agents: Any
) -> None:
    monkeypatch.setenv("MULENET_UI_CHAT_BACKEND", "agents")
    monkeypatch.setenv("MULENET_UI_AGENTS_URL", fake_agents.url)
    monkeypatch.setenv("MULENET_UI_AGENTS_AUTH", "0")
    backend = build_backend(UiConfig.from_env())
    assert isinstance(backend, AgentsRouter)
    assert backend.reply(ChatRequest(message="hello"), None).backend == "gemini"


# --------------------------------------------------------------------------
# live-proof
# --------------------------------------------------------------------------


def test_live_proof_start_is_carried_through_without_renaming_anything(
    fake_agents: Any,
) -> None:
    result = AgentsRouter(_client(fake_agents)).live_proof_start()
    assert result.capped is False
    assert result.run_id == "abc123"
    assert result.backend == "agents"
    assert result.offline_demo is False


def test_a_cap_hit_is_carried_through_as_a_200_not_an_error(fake_agents: Any) -> None:
    """The cap hit is the service's own non-degraded answer -- it must
    never be routed through the mock fallback or shown as offline."""
    fake_agents.live_proof_start_mode = "capped"
    result = AgentsRouter(_client(fake_agents)).live_proof_start()
    assert result.capped is True
    assert result.run_id is None
    assert result.runs_today == 10
    assert result.limit == 10
    assert result.backend == "agents"
    assert result.offline_demo is False


def test_an_unreachable_agents_service_degrades_live_proof_start_visibly(
    fake_agents: Any,
) -> None:
    fake_agents.live_proof_start_mode = "internal"
    result = AgentsRouter(_client(fake_agents)).live_proof_start()
    assert result.backend == "mock"
    assert result.offline_demo is True
    assert result.capped is False
    assert result.run_id is None
    assert "no offline equivalent" in result.offline_notice.lower()


def test_live_proof_status_is_carried_through_and_hits_the_right_path(
    fake_agents: Any,
) -> None:
    result = AgentsRouter(_client(fake_agents)).live_proof_status("abc123")
    assert fake_agents.seen_live_proof_status_id == "abc123"
    assert result.stage == "vm_running"
    assert result.extra == {"vm_name": "mulenet-live-proof-abc123"}
    assert result.backend == "agents"
    assert result.offline_demo is False


def test_an_unknown_run_id_degrades_status_rather_than_raising(fake_agents: Any) -> None:
    fake_agents.live_proof_status_mode = "not_found"
    result = AgentsRouter(_client(fake_agents)).live_proof_status("no-such-run")
    assert result.backend == "mock"
    assert result.offline_demo is True
    assert result.stage == "failed"


def test_the_mock_never_returns_a_run_id_to_poll_for() -> None:
    """MockRouter has no VM to create, so there must be nothing to poll."""
    result = MockRouter().live_proof_start()
    assert result.run_id is None
    assert result.backend == "mock"
    assert result.offline_demo is True


def test_tampered_start_is_carried_through_and_hits_its_own_path(fake_agents: Any) -> None:
    result = AgentsRouter(_client(fake_agents)).live_proof_tampered_start()
    assert result.capped is False
    assert result.run_id == "abc123"
    assert result.backend == "agents"
    assert result.offline_demo is False


def test_tampered_start_shares_the_cap_response_shape(fake_agents: Any) -> None:
    fake_agents.live_proof_start_mode = "capped"
    result = AgentsRouter(_client(fake_agents)).live_proof_tampered_start()
    assert result.capped is True
    assert result.runs_today == 10
    assert result.offline_demo is False


def test_an_unreachable_agents_service_degrades_tampered_start_visibly(fake_agents: Any) -> None:
    fake_agents.live_proof_start_mode = "internal"
    result = AgentsRouter(_client(fake_agents)).live_proof_tampered_start()
    assert result.backend == "mock"
    assert result.offline_demo is True
    assert result.run_id is None


def test_the_mock_never_returns_a_tampered_run_id_either() -> None:
    result = MockRouter().live_proof_tampered_start()
    assert result.run_id is None
    assert result.offline_demo is True
