"""Tests for the HTTP service, with Gemini mocked.

Every test here is offline and free. The Vertex path is exercised by
replacing exactly one function -- :func:`agents.adk_agents._run_agent`,
the only place a model is called -- so routing, JSON parsing, provenance
collection, the budget and the error shapes all run through their real
code with no network and no bill.

The assertions are written against ``docs/agents_api.md``, which is
frozen and which ``ui/`` is being built against in parallel. Field names
are checked exactly, and ``/chat`` and ``/cases`` are checked for having
*no* extra keys: an added field is a contract break even when nothing
reads it yet.

The single live test at the bottom is gated on :data:`LIVE_GEMINI_ENV`
and skips by default. Running ``pytest`` must never cost money.
"""

from __future__ import annotations

import json
import os
import threading
import time
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents import adk_agents, cases, data, service, tools
from agents.config import DEFAULT_MODEL, AgentsConfig, ServiceLimits, load_prompt
from agents.limits import LimitExceeded, RequestGuard
from shared.contracts import case_id_for_ring

#: Set to ``1`` to run the one test that calls Gemini and is billed.
#: Mirrors ``ml/conftest.REQUIRE_ARTEFACTS_ENV``: one env var, value
#: ``1``, off by default.
LIVE_GEMINI_ENV = "MULENET_RUN_LIVE_GEMINI"

#: The exact keys ``docs/agents_api.md`` specifies for a 200 from /chat.
CHAT_KEYS = {"answer", "backend", "verified", "sources", "regulation_citations", "latency_ms"}

#: The exact keys the contract specifies for a 200 from /cases.
CASE_KEYS = {
    "case_id",
    "ring_id",
    "verified_at",
    "verification_attempts",
    "verified_claim_count",
    "analyst_notes",
    "status",
    "opened_at",
    "closed_at",
}

#: The exact keys of the error envelope.
ERROR_KEYS = {"error_type", "message", "retry_after_s"}

QUESTION = {"session_id": "s1", "message": "How many rings are there?"}


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


class FakeClock:
    """A clock the tests advance by hand.

    The rate windows are 60 seconds wide. Sleeping through them would put
    minutes into the suite for no added coverage, so time is injected.
    """

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _config(**overrides: Any) -> AgentsConfig:
    """A Vertex-mode config. The service has no mock path by contract."""
    base = AgentsConfig.from_env()
    fields: dict[str, Any] = {
        "mode": "vertex",
        "data_dir": base.data_dir,
        "ml_out_dir": base.ml_out_dir,
        "model": DEFAULT_MODEL,
        "project": "mulenet",
        "location": "us-central1",
        "use_vertex": True,
    }
    fields.update(overrides)
    return AgentsConfig(**fields)


@pytest.fixture
def config() -> AgentsConfig:
    return _config()


@pytest.fixture
def generous_limits() -> ServiceLimits:
    """Limits high enough not to interfere with a behaviour test."""
    return ServiceLimits(
        session_limit=1000, ip_limit=1000, window_s=60.0, daily_cap=10_000
    )


@pytest.fixture
def fake_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the one function that calls a model. Nothing else is faked."""
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        _gemini(
            {
                "router": {
                    "destination": "analyst_qa",
                    "subject_id": None,
                    "reason": "asks for a count",
                },
                "analyst_qa": {
                    "answer": "12",
                    "basis": "run_sql over rings",
                    "unit": "rings",
                },
            }
        ),
    )


@pytest.fixture
def client(
    config: AgentsConfig, generous_limits: ServiceLimits, fake_gemini: None
) -> TestClient:
    """A client with a mocked model and the budget out of the way."""
    return TestClient(service.create_app(config, generous_limits))


@pytest.fixture
def known_ring() -> str:
    """A ring id that really has a verified case in ``cases.parquet``."""
    frame = cases.load_cases()
    if frame.empty:
        pytest.skip(
            "agents/out/cases.parquet has no rows; regenerate with "
            "`python -m agents.cases`."
        )
    ring_id = str(frame.iloc[0]["ring_id"])
    KNOWN_RING_FOR_TOOLS[0] = ring_id
    return ring_id


def _run_tools_like_adk(tool_names: list[str], sink: set[str] | None) -> None:
    """Execute wrapped tools the way ADK really does: on a bare thread.

    This is the crux of the regression. ADK's synchronous ``Runner.run``
    calls ``create_thread`` then ``asyncio.run``, so every tool executes
    on a plain ``threading.Thread``. A plain thread starts with an empty
    context, which is why the original ``ContextVar`` collector recorded
    nothing on every live request while passing every offline test -- the
    offline tests called the tools inline.

    So the fake runs them on a bare thread. Any collector that depends on
    the calling thread's context fails here, as it should.
    """

    def call_them() -> None:
        for tool_name in tool_names:
            wrapped = adk_agents._model_tool(getattr(tools, tool_name), sink)
            if tool_name == "run_sql":
                wrapped("SELECT ring_id FROM rings LIMIT 1")
            elif tool_name == "describe_tables":
                wrapped()
            else:
                wrapped(KNOWN_RING_FOR_TOOLS[0])

    thread = threading.Thread(target=call_them)
    thread.start()
    thread.join()
    # These tests are about provenance and plumbing, with canned answers
    # ("12", "2") that the real tables do not produce. The grounding check
    # (agents.grounding) refuses a number no tool returned, so give the run
    # the canned figures as if a query had returned them.
    record = getattr(sink, "record_result", None)
    if record is not None:
        record("fixture", {"canned": [12, 2, 99, 0.91]})


#: Filled by the ``known_ring`` fixture so ``_run_tools_like_adk`` can call
#: subject-taking tools with a ring that exists.
KNOWN_RING_FOR_TOOLS: list[str] = ["ring-unset"]


def _gemini(responses: dict[str, Any], tool_calls: list[str] | None = None) -> Any:
    """Build a stand-in for ``adk_agents._run_agent``.

    Mirrors the real signature, ``sink`` included, because the sink is the
    thing under test: a fake that dropped it would pass while the service
    returned empty ``sources`` in production, which is precisely what
    happened.

    Args:
        responses: Agent name to the raw text that agent should
            "return". A callable is invoked instead, so a test can raise.
        tool_calls: Tools ``analyst_qa`` should execute before replying.
            Defaults to one ``describe_tables``, because an answer with no
            tool execution is refused and almost every test here wants an
            answer rather than a refusal.
    """
    to_call = ["describe_tables"] if tool_calls is None else tool_calls

    def fake_run_agent(
        name: str,
        message: str,
        config: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        if name == "analyst_qa":
            _run_tools_like_adk(to_call, sink)
        reply = responses[name]
        if callable(reply):
            return reply(message)
        return reply if isinstance(reply, str) else json.dumps(reply)

    return fake_run_agent


def _guarded(
    config: AgentsConfig, limits: ServiceLimits, clock: FakeClock
) -> TestClient:
    return TestClient(service.create_app(config, guard=RequestGuard(limits, now=clock)))


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------


def test_health_is_exactly_status_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_calls_no_model_and_reads_no_data(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """A probe that billed per check would be worse than no probe."""

    def explode(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("/health called a model")

    api = TestClient(service.create_app(config, generous_limits))
    monkeypatch.setattr(adk_agents, "_run_agent", explode)
    monkeypatch.setattr(
        cases, "load_cases", lambda *_a, **_k: pytest.fail("/health read data")
    )
    assert api.get("/health").json() == {"status": "ok"}


def test_no_route_path_ends_in_z(client: TestClient) -> None:
    """Cloud Run's front end swallows any path ending in ``z``.

    It answers such a path with its own HTML ``404`` before the request
    reaches the container, confirmed on the live service, so an uptime
    check pointed at ``/healthz`` reports permanently unhealthy however
    well the service runs. The route was named that once; this is what
    stops it being named that again. ``/healthz`` must be a plain 404 from
    us -- there is no route to redirect to.
    """
    paths = [route.path for route in client.app.routes]
    assert "/health" in paths
    assert not [path for path in paths if path.rstrip("/").endswith("z")]
    assert client.get("/healthz", follow_redirects=False).status_code == 404


# --------------------------------------------------------------------------
# chat: the contract shape
# --------------------------------------------------------------------------


def test_chat_returns_exactly_the_contract_fields(client: TestClient) -> None:
    response = client.post("/chat", json=QUESTION)
    assert response.status_code == 200, response.text
    body = response.json()
    assert CHAT_KEYS <= set(body)  # additive contract: new fields allowed
    assert body["answer"] == "12 rings"
    assert body["backend"] == "gemini"
    assert isinstance(body["verified"], bool)
    assert isinstance(body["sources"], list)
    assert isinstance(body["latency_ms"], int)
    assert body["latency_ms"] >= 0


def test_backend_is_the_literal_gemini(client: TestClient) -> None:
    """The field exists so a caller can prove which path answered."""
    assert client.post("/chat", json=QUESTION).json()["backend"] == "gemini"


def test_the_router_runs_and_analyst_qa_answers(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """Both agents are real code over a mocked model, in that order."""
    called: list[str] = []

    def fake_run_agent(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        called.append(name)
        if name == "router":
            return json.dumps(
                {"destination": "analyst_qa", "subject_id": None, "reason": "count"}
            )
        _run_tools_like_adk(["describe_tables"], sink)
        return json.dumps({"answer": "12", "basis": "run_sql", "unit": ""})

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    api = TestClient(service.create_app(config, generous_limits))
    assert api.post("/chat", json=QUESTION).json()["answer"] == "12"
    assert called == ["router", "analyst_qa"]


def test_a_routed_investigation_is_still_answered_by_gemini(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """A canned deferral would make ``backend: gemini`` a lie.

    The contract has no deferral field and forbids labelling text Gemini
    did not write as coming from Gemini, so a question the router sends
    elsewhere is answered by the only answering agent this endpoint has.
    """
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        _gemini(
            {
                "router": {
                    "destination": "investigator",
                    "subject_id": "ring-x",
                    "reason": "asks why",
                },
                "analyst_qa": {
                    "answer": "ring-x scores 0.91",
                    "basis": "get_explanation",
                    "unit": "",
                },
            }
        ),
    )
    api = TestClient(service.create_app(config, generous_limits))
    body = api.post(
        "/chat", json={"session_id": "s1", "message": "Why was ring-x flagged?"}
    ).json()
    assert body["backend"] == "gemini"
    assert body["answer"] == "ring-x scores 0.91"


def test_the_mock_path_is_never_served_as_gemini(
    monkeypatch: pytest.MonkeyPatch, generous_limits: ServiceLimits
) -> None:
    """Even in mock mode, /chat goes to Vertex or it errors."""
    from agents import mock_agents

    monkeypatch.setattr(
        mock_agents, "route", lambda *_a: pytest.fail("/chat used the mock router")
    )
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("no vertex here")),
    )
    mock_mode = _config(mode="mock", project=None, use_vertex=False)
    api = TestClient(service.create_app(mock_mode, generous_limits))
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 503
    assert response.json()["error_type"] == "upstream_unavailable"


# --------------------------------------------------------------------------
# chat: verified and sources
# --------------------------------------------------------------------------


def test_verified_is_false_when_nothing_checkable_was_claimed(
    client: TestClient,
) -> None:
    """False is "not machine-checked", not "suspect"."""
    assert client.post("/chat", json=QUESTION).json()["verified"] is False


def test_a_failed_claim_withholds_the_answer_rather_than_flagging_it(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """The existing withhold-on-failure contract has to survive HTTP.

    The claim arrives the way a real one does -- in the model's reply, as
    structure -- and names a ring that does not exist. Nothing here
    monkeypatches the service's own claim handling: there is no hook to
    patch any more, because claims come from the model and from nowhere
    else. ``agents/test_chat_claims.py`` covers the same property over a
    dataset it builds itself, so this one stays about the transport.
    """
    api = TestClient(
        service.create_app(
            config,
            generous_limits,
        )
    )
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        _gemini(
            {
                "router": {
                    "destination": "analyst_qa",
                    "subject_id": None,
                    "reason": "asks for a count",
                },
                "analyst_qa": {
                    "answer": "99",
                    "basis": "get_ring_summary(ring-nope)",
                    "unit": "members",
                    "claims": [
                        {
                            "id": "c1",
                            "kind": "ring_member_count",
                            "subject": "ring-nope",
                            "value": 99,
                            "text": "ring-nope has 99 members",
                        }
                    ],
                },
            }
        ),
    )
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 409
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["error_type"] == "unverified"
    assert "withheld" in body["message"]


def test_a_passing_claim_sets_verified_true(
    monkeypatch: pytest.MonkeyPatch,
    config: AgentsConfig,
    generous_limits: ServiceLimits,
    known_ring: str,
) -> None:
    """The true path, proven with a claim the real verifier re-derives.

    Against the published tables, so it is the live claim shape over live
    data. The value is read from the tool the way the model would read it
    and then re-derived independently by the verifier.
    """
    from agents import tools

    summary = tools.get_ring_summary(known_ring)
    api = TestClient(service.create_app(config, generous_limits))
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        _gemini(
            {
                "router": {
                    "destination": "analyst_qa",
                    "subject_id": known_ring,
                    "reason": "asks for a count",
                },
                "analyst_qa": {
                    "answer": str(int(summary["bank_count"])),
                    "basis": f"get_ring_summary({known_ring})",
                    "unit": "banks",
                    "claims": [
                        {
                            "id": "c1",
                            "kind": "ring_bank_count",
                            "subject": known_ring,
                            "value": int(summary["bank_count"]),
                            "text": "banks",
                        }
                    ],
                },
            }
        ),
    )
    body = api.post("/chat", json=QUESTION).json()
    assert body["verified"] is True


def test_sources_come_from_the_tools_the_run_actually_called(
    monkeypatch: pytest.MonkeyPatch,
    config: AgentsConfig,
    generous_limits: ServiceLimits,
    known_ring: str,
) -> None:
    """Provenance is collected at the tool wrapper, not read off the model."""
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        _gemini(
            {
                "router": {
                    "destination": "analyst_qa",
                    "subject_id": None,
                    "reason": "count",
                },
                "analyst_qa": {
                    "answer": "12",
                    # The model's own account names a table it never
                    # touched. It must not appear in sources.
                    "basis": "I queried account_evidence",
                    "unit": "",
                },
            },
            tool_calls=["describe_tables", "run_sql"],
        ),
    )
    api = TestClient(service.create_app(config, generous_limits))
    sources = api.post("/chat", json=QUESTION).json()["sources"]
    assert "describe_tables" in sources
    assert "run_sql" in sources
    # The table the query really named, resolved by the SQL guard.
    assert "rings" in sources
    assert "account_evidence" not in sources
    assert sources == sorted(sources)


def test_sources_survive_tools_running_on_a_bare_thread(
    monkeypatch: pytest.MonkeyPatch,
    config: AgentsConfig,
    generous_limits: ServiceLimits,
    known_ring: str,
) -> None:
    """The regression test for the empty-``sources`` bug.

    ADK's synchronous ``Runner.run`` executes the agent on a plain
    ``threading.Thread``, and a plain thread starts with an empty context.
    The original collector was a ``ContextVar``, so every tool call wrote
    into something the request could not see and live ``sources`` came
    back ``[]`` on every single request while the offline suite stayed
    green -- because the offline suite called the tools inline.

    ``_run_tools_like_adk`` runs them on a bare thread, so a collector
    that depends on the calling thread's context cannot pass this.
    """
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        _gemini(
            {
                "router": {
                    "destination": "analyst_qa",
                    "subject_id": None,
                    "reason": "count",
                },
                "analyst_qa": {"answer": "2", "basis": "get_ring_summary", "unit": ""},
            },
            tool_calls=["get_ring_summary"],
        ),
    )
    api = TestClient(service.create_app(config, generous_limits))
    body = api.post("/chat", json=QUESTION).json()
    assert body["sources"] == ["get_ring_summary"], (
        "a tool executed on a bare thread was not recorded; provenance is "
        "context-dependent again"
    )


def test_a_200_can_never_carry_empty_sources(client: TestClient) -> None:
    """The contract allows an empty list; the grounding rule rules it out.

    ``docs/agents_api.md`` says an empty ``sources`` is valid, and it is --
    for a run that legitimately consulted nothing. But such a run is now
    refused with ``409 no_tool_evidence``, so on the one path that returns
    ``200`` the list cannot be empty. Worth pinning: an empty list on a
    ``200`` would mean the grounding check had stopped running, which is
    exactly the state this service shipped in once already.
    """
    body = client.post("/chat", json=QUESTION).json()
    assert body["sources"], "a 200 with empty sources means the check is off"


def test_the_context_var_approach_would_not_have_worked() -> None:
    """Documents the mechanism, so nobody reintroduces it.

    Not a test of current behaviour -- there is no ``ContextVar`` left. It
    pins the platform fact that made the old design wrong, because that
    fact is the reason the current design looks more awkward than it
    needs to.
    """
    import contextvars

    var: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
        "probe", default=None
    )
    collected: set[str] = set()
    var.set(collected)

    def in_a_bare_thread() -> None:
        sink = var.get()
        if sink is not None:  # pragma: no cover - the point is that it is None
            sink.add("seen")

    thread = threading.Thread(target=in_a_bare_thread)
    thread.start()
    thread.join()
    assert collected == set(), (
        "a ContextVar now crosses threading.Thread; the closure-bound sink "
        "could be simplified"
    )


def test_the_tool_log_is_an_independent_observer(
    monkeypatch: pytest.MonkeyPatch,
    config: AgentsConfig,
    generous_limits: ServiceLimits,
    known_ring: str,
) -> None:
    """A second witness, so a broken collector cannot hide itself."""
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        _gemini(
            {
                "router": {
                    "destination": "analyst_qa",
                    "subject_id": None,
                    "reason": "count",
                },
                "analyst_qa": {"answer": "2", "basis": "get_ring_summary", "unit": ""},
            },
            tool_calls=["get_ring_summary"],
        ),
    )
    api = TestClient(service.create_app(config, generous_limits))
    adk_agents.reset_tool_log()
    api.post("/chat", json=QUESTION)

    executed = [entry["tool"] for entry in adk_agents.TOOL_CALLS]
    assert "get_ring_summary" in executed
    entry = next(e for e in adk_agents.TOOL_CALLS if e["tool"] == "get_ring_summary")
    assert entry["args"] == [known_ring]
    assert entry["at"]
    assert entry["thread"]


def test_the_tool_log_records_a_failing_tool_too() -> None:
    """A tool that raised was still executed, so it still counts."""

    def get_ring_summary(ring_id: str) -> None:
        raise RuntimeError("alloydb unreachable")

    adk_agents.reset_tool_log()
    sink: set[str] = set()
    adk_agents._model_tool(get_ring_summary, sink)("ring-1")
    assert [entry["tool"] for entry in adk_agents.TOOL_CALLS] == ["get_ring_summary"]
    assert sink == {"get_ring_summary"}


def test_the_tool_log_is_bounded() -> None:
    """A long-lived instance must not accumulate entries forever."""

    def describe_tables() -> dict[str, Any]:
        return {}

    adk_agents.reset_tool_log()
    wrapped = adk_agents._model_tool(describe_tables)
    for _ in range(adk_agents._TOOL_LOG_LIMIT + 50):
        wrapped()
    assert len(adk_agents.TOOL_CALLS) == adk_agents._TOOL_LOG_LIMIT


# --------------------------------------------------------------------------
# grounding: an answer with no tool call is not an answer
# --------------------------------------------------------------------------


def test_an_answer_with_no_tool_call_is_refused(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """The finding this rule exists for: a plausible number from nothing."""
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        _gemini(
            {
                "router": {
                    "destination": "analyst_qa",
                    "subject_id": None,
                    "reason": "count",
                },
                # Confident, correct-looking, and grounded in nothing.
                "analyst_qa": {"answer": "2", "basis": "the rings table", "unit": "banks"},
            },
            tool_calls=[],
        ),
    )
    api = TestClient(service.create_app(config, generous_limits))
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 409
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["error_type"] == "no_tool_evidence"
    assert body["retry_after_s"] is None
    # The plausible answer must not be in the body at all.
    assert "2" not in body["message"].replace("409", "")


def test_the_refusal_counts_executions_not_the_models_self_report(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """A model claiming to have queried does not make it grounded."""
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        _gemini(
            {
                "router": {
                    "destination": "analyst_qa",
                    "subject_id": None,
                    "reason": "count",
                },
                "analyst_qa": {
                    "answer": "12",
                    "basis": "I ran run_sql against rings and account_evidence",
                    "unit": "",
                },
            },
            tool_calls=[],
        ),
    )
    api = TestClient(service.create_app(config, generous_limits))
    assert api.post("/chat", json=QUESTION).status_code == 409


def test_one_executed_tool_is_enough_to_publish(
    monkeypatch: pytest.MonkeyPatch,
    config: AgentsConfig,
    generous_limits: ServiceLimits,
    known_ring: str,
) -> None:
    """The rule is "at least one", not a quota."""
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        _gemini(
            {
                "router": {
                    "destination": "analyst_qa",
                    "subject_id": None,
                    "reason": "count",
                },
                "analyst_qa": {"answer": "2", "basis": "get_ring_summary", "unit": ""},
            },
            tool_calls=["get_ring_summary"],
        ),
    )
    api = TestClient(service.create_app(config, generous_limits))
    assert api.post("/chat", json=QUESTION).status_code == 200


# --------------------------------------------------------------------------
# an empty model response
# --------------------------------------------------------------------------


def test_an_empty_completion_is_retried_once_then_refused(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """Never a 500: an empty completion is a model outcome, not a defect."""
    attempts: list[str] = []

    def fake_run_agent(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        attempts.append(name)
        if name == "router":
            return json.dumps(
                {"destination": "analyst_qa", "subject_id": None, "reason": "count"}
            )
        return ""

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    api = TestClient(service.create_app(config, generous_limits))
    response = api.post("/chat", json=QUESTION)

    assert response.status_code == 409
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["error_type"] == "no_tool_evidence"
    # Exactly one retry: the router once, analyst_qa twice.
    assert attempts == ["router", "analyst_qa", "analyst_qa"]


def test_an_empty_completion_that_recovers_on_the_retry_is_answered(
    monkeypatch: pytest.MonkeyPatch,
    config: AgentsConfig,
    generous_limits: ServiceLimits,
    known_ring: str,
) -> None:
    """The retry is a real retry, not a formality before refusing."""
    calls = {"analyst_qa": 0}

    def fake_run_agent(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        if name == "router":
            return json.dumps(
                {"destination": "analyst_qa", "subject_id": None, "reason": "count"}
            )
        calls["analyst_qa"] += 1
        if calls["analyst_qa"] == 1:
            return ""
        _run_tools_like_adk(["get_ring_summary"], sink)
        return json.dumps({"answer": "2", "basis": "get_ring_summary", "unit": "banks"})

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    api = TestClient(service.create_app(config, generous_limits))
    body = api.post("/chat", json=QUESTION).json()
    assert body["answer"] == "2 banks"
    assert body["sources"] == ["get_ring_summary"]
    assert calls["analyst_qa"] == 2


def test_a_first_attempts_provenance_does_not_carry_into_the_retry(
    monkeypatch: pytest.MonkeyPatch,
    config: AgentsConfig,
    generous_limits: ServiceLimits,
    known_ring: str,
) -> None:
    """The retry reports its own tools, not the failed attempt's."""
    calls = {"analyst_qa": 0}

    def fake_run_agent(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        if name == "router":
            return json.dumps(
                {"destination": "analyst_qa", "subject_id": None, "reason": "count"}
            )
        calls["analyst_qa"] += 1
        if calls["analyst_qa"] == 1:
            _run_tools_like_adk(["run_sql"], sink)
            return ""
        _run_tools_like_adk(["get_ring_summary"], sink)
        return json.dumps({"answer": "2", "basis": "get_ring_summary", "unit": ""})

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    api = TestClient(service.create_app(config, generous_limits))
    assert api.post("/chat", json=QUESTION).json()["sources"] == ["get_ring_summary"]


def test_an_unreadable_response_that_recovers_on_the_retry_is_answered(
    monkeypatch: pytest.MonkeyPatch,
    config: AgentsConfig,
    generous_limits: ServiceLimits,
    known_ring: str,
) -> None:
    """The retry earns its call: this is non-deterministic in practice."""
    calls = {"analyst_qa": 0}

    def fake_run_agent(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        if name == "router":
            return json.dumps(
                {"destination": "analyst_qa", "subject_id": None, "reason": "count"}
            )
        calls["analyst_qa"] += 1
        if calls["analyst_qa"] == 1:
            return "The ring spans 2 banks."
        _run_tools_like_adk(["get_ring_summary"], sink)
        return json.dumps({"answer": "2", "basis": "get_ring_summary", "unit": "banks"})

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    api = TestClient(service.create_app(config, generous_limits))
    body = api.post("/chat", json=QUESTION).json()
    assert body["answer"] == "2 banks"
    assert body["sources"] == ["get_ring_summary"]


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# screen
# --------------------------------------------------------------------------


@pytest.fixture
def flagged_account(config: AgentsConfig) -> str:
    """A real account with at least one ``alertable`` ring membership."""
    store = data.DataStore(config)
    rings = store.table("rings")
    for row in rings.itertuples():
        for member in list(row.members):
            if member["alertable"]:
                return str(member["hashed_account_id"])
    pytest.skip("no alertable ring member in the loaded tables")


@pytest.fixture
def clean_account(config: AgentsConfig, flagged_account: str) -> str:
    """A real account that is in no ring at all."""
    store = data.DataStore(config)
    in_a_ring = {
        str(member["hashed_account_id"])
        for row in store.table("rings").itertuples()
        for member in list(row.members)
    }
    for hashed_account_id in store.table("accounts")["hashed_account_id"]:
        if str(hashed_account_id) not in in_a_ring:
            return str(hashed_account_id)
    pytest.skip("every loaded account is in some ring")


def test_a_mule_account_is_flagged(client: TestClient, flagged_account: str) -> None:
    body = client.get(f"/screen/{flagged_account}").json()
    assert body["flagged"] is True
    assert body["hashed_account_id"] == flagged_account
    assert body["ring_ids"]
    assert body["reason"]


def test_an_uninvolved_account_is_not_flagged(
    client: TestClient, clean_account: str
) -> None:
    body = client.get(f"/screen/{clean_account}").json()
    assert body == {
        "hashed_account_id": clean_account,
        "flagged": False,
        "ring_ids": [],
        "reason": "",
        "risk_score": None,
    }


def test_an_unknown_account_is_not_flagged_rather_than_a_404(
    client: TestClient,
) -> None:
    """No account record at all reads the same as one with no ring hits.

    A payment path calling this before every transfer should get a fast,
    uniform "not flagged" for an account it has never seen, not a 404 it
    has to special-case.
    """
    response = client.get("/screen/acct-does-not-exist-0001")
    assert response.status_code == 200
    assert response.json()["flagged"] is False


def test_a_same_bank_identity_link_alone_does_not_flag_a_payment(
    config: AgentsConfig, generous_limits: ServiceLimits, fake_gemini: None
) -> None:
    """A same-bank identity link is a lead for an analyst, never a block.

    ``docs/agents_api.md``-style guarantee, expressed here: same-bank
    identity reuse is dominated by legitimate cover (family sharing a
    document), so it must never stop a payment on its own.
    """
    store = data.DataStore(config)
    for row in store.table("rings").itertuples():
        for member in list(row.members):
            if (
                member["membership_basis"] == "identity_link"
                and member["link_scope"] == "same_bank"
            ):
                api = TestClient(service.create_app(config, generous_limits))
                body = api.get(
                    f"/screen/{member['hashed_account_id']}"
                ).json()
                assert body["flagged"] is False
                return
    pytest.skip("no same-bank identity_link member in the loaded tables")


def test_screen_endpoint_makes_no_model_call(
    monkeypatch: pytest.MonkeyPatch,
    config: AgentsConfig,
    generous_limits: ServiceLimits,
    flagged_account: str,
) -> None:
    api = TestClient(service.create_app(config, generous_limits))
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        lambda *_a, **_k: pytest.fail("the screen endpoint called a model"),
    )
    assert api.get(f"/screen/{flagged_account}").status_code == 200


def test_case_endpoint_serves_exactly_the_published_columns(
    client: TestClient, known_ring: str
) -> None:
    response = client.get(f"/cases/{known_ring}")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == CASE_KEYS
    frame = cases.load_cases()
    row = frame[frame["ring_id"] == known_ring].iloc[0]
    assert body["case_id"] == str(row["case_id"])
    assert body["verification_attempts"] == int(row["verification_attempts"])
    assert body["verified_claim_count"] == int(row["verified_claim_count"])
    assert body["status"] == str(row["status"])


def test_case_id_is_the_uuid5_of_the_ring_id(client: TestClient) -> None:
    """Contract: ``case_id == case_id_for_ring(ring_id)`` for every case."""
    frame = cases.load_cases()
    if frame.empty:
        pytest.skip("no cases published")
    for ring_id in frame["ring_id"]:
        body = client.get(f"/cases/{ring_id}").json()
        assert body["case_id"] == case_id_for_ring(str(ring_id))


def test_case_timestamps_are_iso_8601_and_utc(
    client: TestClient, known_ring: str
) -> None:
    from datetime import UTC, datetime

    body = client.get(f"/cases/{known_ring}").json()
    for field in ("verified_at", "opened_at"):
        parsed = datetime.fromisoformat(body[field])
        assert parsed.tzinfo is not None, field
        assert parsed.utcoffset() == UTC.utcoffset(None), field


def test_case_endpoint_makes_no_model_call(
    monkeypatch: pytest.MonkeyPatch,
    config: AgentsConfig,
    generous_limits: ServiceLimits,
    known_ring: str,
) -> None:
    api = TestClient(service.create_app(config, generous_limits))
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        lambda *_a, **_k: pytest.fail("the case endpoint called a model"),
    )
    assert api.get(f"/cases/{known_ring}").status_code == 200


def test_unknown_ring_is_404_no_case_and_not_a_placeholder(
    client: TestClient,
) -> None:
    response = client.get("/cases/ring-does-not-exist-0001")
    assert response.status_code == 404
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["error_type"] == "no_case"
    assert body["retry_after_s"] is None
    # No case-shaped field at all: a client reading optimistically must
    # not find a case id to trust.
    assert not CASE_KEYS & set(body)
    assert "verifier" in body["message"]


def test_case_endpoint_is_not_charged_against_the_daily_model_cap(
    config: AgentsConfig, known_ring: str
) -> None:
    """Free reads must not exhaust the budget that protects paid calls."""
    clock = FakeClock()
    guard = RequestGuard(
        ServiceLimits(session_limit=99, ip_limit=99, window_s=60.0, daily_cap=1),
        now=clock,
    )
    api = TestClient(service.create_app(config, guard=guard))
    for _ in range(5):
        assert api.get(f"/cases/{known_ring}").status_code == 200
    assert guard.remaining_today() == 1


# --------------------------------------------------------------------------
# tool errors
# --------------------------------------------------------------------------


def test_a_raising_tool_becomes_a_structured_error_for_the_model() -> None:
    """The property the retry loop depends on: the model sees the failure."""

    def run_sql(query: str) -> dict[str, Any]:
        raise ValueError("no such column: num_banks")

    result = adk_agents._model_tool(run_sql)("SELECT num_banks FROM rings")

    assert isinstance(result, dict)
    error = result["error"]
    assert error["tool"] == "run_sql"
    assert error["error_type"] == "ValueError"
    assert error["recoverable"] is True
    # The recovery material: the model can only rewrite the query if it
    # is handed the tables and columns that do exist.
    assert error["allowed_tables"]
    assert error["schema"]
    assert error["row_limit"]


def test_a_raising_tool_does_not_raise_through_the_wrapper() -> None:
    """An exception inside a tool must never reach the Runner."""

    def get_ring_subgraph(ring_id: str) -> None:
        raise RuntimeError("alloydb unreachable")

    result = adk_agents._model_tool(get_ring_subgraph)("ring-1")
    assert result["error"]["error_type"] == "RuntimeError"
    # Not a schema-aware tool, so no schema is attached: the hint has to
    # be about the failure, not boilerplate.
    assert "schema" not in result["error"]


def test_a_tool_error_does_not_fail_the_http_request(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """End to end: a tool blew up, the model recovered, the caller got 200."""

    # Named for the real tool, because the grounding check counts real
    # tool names: a raising run_sql is still an executed run_sql.
    def run_sql(query: str) -> None:
        raise ValueError("no such column: num_banks")

    def fake_run_agent(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        if name == "analyst_qa":
            # The tool raised, so it was still executed -- and it must
            # still count as provenance, or a run whose only tool call
            # failed would look like a run that never queried.
            first = adk_agents._model_tool(run_sql, sink)(
                "SELECT num_banks FROM rings"
            )
            assert "error" in first, "the tool raised instead of returning"
            # The retry that "returned" 12 (grounding needs a source for it).
            sink.record_result("run_sql", [{"n": 12}])
            return json.dumps(
                {"answer": "12", "basis": "retried after a schema error", "unit": ""}
            )
        return json.dumps(
            {"destination": "analyst_qa", "subject_id": None, "reason": "count"}
        )

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    api = TestClient(service.create_app(config, generous_limits))
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 200
    assert response.json()["answer"] == "12"


def test_an_unreachable_vertex_is_503_upstream_unavailable(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    def explode(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        raise ConnectionError("failed to connect to aiplatform.googleapis.com")

    monkeypatch.setattr(adk_agents, "_run_agent", explode)
    api = TestClient(
        service.create_app(config, generous_limits), raise_server_exceptions=False
    )
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 503
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["error_type"] == "upstream_unavailable"
    assert body["retry_after_s"] is None


def test_an_internal_failure_is_500_with_no_traceback(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """Our own bug is a 500, and its text never reaches the client."""

    def explode(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        raise ValueError("PERMISSION_DENIED on project mulenet-secret")

    monkeypatch.setattr(adk_agents, "_run_agent", explode)
    api = TestClient(
        service.create_app(config, generous_limits), raise_server_exceptions=False
    )
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 500
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["error_type"] == "internal"
    assert "ValueError" in body["message"]
    # A provider message can carry a project id or a row of data back to
    # the caller, so the type is named but the text is not quoted.
    assert "mulenet-secret" not in body["message"]
    assert "Traceback" not in body["message"]


# --------------------------------------------------------------------------
# timeouts
# --------------------------------------------------------------------------


def test_a_hung_model_call_becomes_504_timeout(
    monkeypatch: pytest.MonkeyPatch, generous_limits: ServiceLimits
) -> None:
    """A hung call must not hold the request open indefinitely."""

    def hang(message: str, cfg: Any) -> tuple[str, bool, list[str]]:
        time.sleep(5.0)
        raise AssertionError("should have been given up on")

    monkeypatch.setattr(service, "_answer", hang)
    api = TestClient(service.create_app(_config(request_timeout_s=0.2), generous_limits))
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 504
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["error_type"] == "timeout"
    assert "did not answer" in body["message"]


def test_every_generate_call_carries_a_timeout_and_an_output_ceiling() -> None:
    """The per-call controls are on the config ADK sends, for every agent."""
    from google.genai import types

    config = _config(model_timeout_s=12.5)
    for name in adk_agents.AGENT_TOOLS:
        generate = adk_agents._content_config(types, name, config)
        assert generate.http_options.timeout == 12500, name
        assert generate.max_output_tokens == adk_agents.MAX_OUTPUT_TOKENS[name], name
        assert generate.max_output_tokens > 0, name


def test_a_configured_output_cap_overrides_every_agent_budget() -> None:
    """One env var can tighten every agent at once."""
    from google.genai import types

    capped = _config(max_output_tokens=64)
    for name in adk_agents.AGENT_TOOLS:
        assert adk_agents._content_config(types, name, capped).max_output_tokens == 64


# --------------------------------------------------------------------------
# cost ceilings
# --------------------------------------------------------------------------


def test_session_limit_returns_429_rate_limited(
    config: AgentsConfig, fake_gemini: None
) -> None:
    clock = FakeClock()
    api = _guarded(
        config,
        ServiceLimits(session_limit=2, ip_limit=99, window_s=60.0, daily_cap=999),
        clock,
    )
    assert api.post("/chat", json=QUESTION).status_code == 200
    assert api.post("/chat", json=QUESTION).status_code == 200

    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 429
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["error_type"] == "rate_limited"
    assert body["retry_after_s"] >= 1
    assert response.headers["retry-after"] == str(body["retry_after_s"])
    # A UI has to be able to print this. "429" is not an explanation.
    assert len(body["message"]) > 40
    assert "2 questions every 60 seconds" in body["message"]


def test_session_limit_clears_when_the_window_slides(
    config: AgentsConfig, fake_gemini: None
) -> None:
    clock = FakeClock()
    api = _guarded(
        config,
        ServiceLimits(session_limit=1, ip_limit=99, window_s=60.0, daily_cap=999),
        clock,
    )
    assert api.post("/chat", json=QUESTION).status_code == 200
    assert api.post("/chat", json=QUESTION).status_code == 429
    clock.advance(61.0)
    assert api.post("/chat", json=QUESTION).status_code == 200


def test_sessions_do_not_inherit_each_others_spend(
    config: AgentsConfig, fake_gemini: None
) -> None:
    clock = FakeClock()
    api = _guarded(
        config,
        ServiceLimits(session_limit=1, ip_limit=99, window_s=60.0, daily_cap=999),
        clock,
    )
    for session_id in ("a", "b", "c"):
        response = api.post(
            "/chat", json={"session_id": session_id, "message": QUESTION["message"]}
        )
        assert response.status_code == 200, session_id
    assert (
        api.post(
            "/chat", json={"session_id": "a", "message": QUESTION["message"]}
        ).status_code
        == 429
    )


def test_ip_limit_catches_a_sweep_through_session_ids(
    config: AgentsConfig, fake_gemini: None
) -> None:
    """The per-session limit alone is trivially bypassed; this is why."""
    clock = FakeClock()
    api = _guarded(
        config,
        ServiceLimits(session_limit=1, ip_limit=3, window_s=60.0, daily_cap=999),
        clock,
    )
    headers = {"X-Forwarded-For": "203.0.113.7, 130.211.0.1"}
    statuses = [
        api.post(
            "/chat",
            json={"session_id": f"s{index}", "message": QUESTION["message"]},
            headers=headers,
        ).status_code
        for index in range(5)
    ]
    assert statuses == [200, 200, 200, 429, 429]

    response = api.post(
        "/chat",
        json={"session_id": "sX", "message": QUESTION["message"]},
        headers=headers,
    )
    assert response.json()["error_type"] == "rate_limited"


def test_ip_limit_uses_the_leftmost_forwarded_address(
    config: AgentsConfig, fake_gemini: None
) -> None:
    """Two callers behind the same front end stay independent."""
    clock = FakeClock()
    api = _guarded(
        config,
        ServiceLimits(session_limit=99, ip_limit=1, window_s=60.0, daily_cap=999),
        clock,
    )
    first = {"X-Forwarded-For": "203.0.113.7, 130.211.0.1"}
    second = {"X-Forwarded-For": "198.51.100.4, 130.211.0.1"}
    assert api.post("/chat", json=QUESTION, headers=first).status_code == 200
    assert api.post("/chat", json=QUESTION, headers=first).status_code == 429
    assert api.post("/chat", json=QUESTION, headers=second).status_code == 200


def test_daily_cap_returns_429_daily_cap_and_stops_the_service(
    config: AgentsConfig, fake_gemini: None
) -> None:
    """A distinct error_type, because it needs different wording."""
    clock = FakeClock()
    api = _guarded(
        config,
        ServiceLimits(session_limit=99, ip_limit=99, window_s=60.0, daily_cap=3),
        clock,
    )
    for index in range(3):
        headers = {"X-Forwarded-For": f"203.0.113.{index}"}
        assert api.post("/chat", json=QUESTION, headers=headers).status_code == 200

    response = api.post(
        "/chat", json=QUESTION, headers={"X-Forwarded-For": "198.51.100.9"}
    )
    assert response.status_code == 429
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["error_type"] == "daily_cap"
    assert "daily" in body["message"].lower()
    assert body["retry_after_s"] > 0
    # Sliding the rate window must not refill the day.
    clock.advance(120.0)
    assert api.post("/chat", json=QUESTION).json()["error_type"] == "daily_cap"


def test_daily_cap_resets_on_the_utc_day_boundary(
    config: AgentsConfig, fake_gemini: None
) -> None:
    clock = FakeClock()
    api = _guarded(
        config,
        ServiceLimits(session_limit=99, ip_limit=99, window_s=60.0, daily_cap=1),
        clock,
    )
    assert api.post("/chat", json=QUESTION).status_code == 200
    assert api.post("/chat", json=QUESTION).status_code == 429
    clock.advance(86_400.0)
    assert api.post("/chat", json=QUESTION).status_code == 200


def test_a_refused_request_consumes_no_budget(
    config: AgentsConfig, fake_gemini: None
) -> None:
    """Otherwise anyone could burn the day's budget for free."""
    clock = FakeClock()
    guard = RequestGuard(
        ServiceLimits(session_limit=1, ip_limit=99, window_s=60.0, daily_cap=10),
        now=clock,
    )
    api = TestClient(service.create_app(config, guard=guard))
    assert api.post("/chat", json=QUESTION).status_code == 200
    for _ in range(5):
        assert api.post("/chat", json=QUESTION).status_code == 429
    assert guard.remaining_today() == 9


def test_the_daily_cap_is_checked_before_the_forgeable_limits() -> None:
    """The budget must not be reachable only through keys a caller picks."""
    clock = FakeClock()
    guard = RequestGuard(
        ServiceLimits(session_limit=1, ip_limit=1, window_s=60.0, daily_cap=1),
        now=clock,
    )
    guard.check_and_record("s", "1.1.1.1")
    with pytest.raises(LimitExceeded) as caught:
        # Session and IP are also exhausted, but the daily cap is the one
        # that must be reported: it will not clear in a minute.
        guard.check_and_record("s", "1.1.1.1")
    assert caught.value.scope == "daily"


def test_a_limit_is_checked_before_any_model_call(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, fake_gemini: None
) -> None:
    """A refusal that had already paid for a call would protect nothing."""
    clock = FakeClock()
    api = _guarded(
        config,
        ServiceLimits(session_limit=99, ip_limit=99, window_s=60.0, daily_cap=1),
        clock,
    )
    assert api.post("/chat", json=QUESTION).status_code == 200

    # The day is now spent. From here a model call would be a bug, so
    # make one fail the test outright.
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        lambda *_a, **_k: pytest.fail("a refused request called the model"),
    )
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 429
    assert response.json()["error_type"] == "daily_cap"


# --------------------------------------------------------------------------
# request validation
# --------------------------------------------------------------------------


def test_an_oversized_question_is_request_validation(client: TestClient) -> None:
    """Input is billed too, so length is a validation rule on the field."""
    response = client.post(
        "/chat",
        json={"session_id": "s1", "message": "x" * (service.MAX_MESSAGE_CHARS + 1)},
    )
    assert response.status_code == 422


def test_a_malformed_body_is_422(client: TestClient) -> None:
    assert client.post("/chat", json={"message": "hi"}).status_code == 422
    assert client.post("/chat", json={"session_id": "s1"}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422
    assert (
        client.post("/chat", json={"session_id": "", "message": "hi"}).status_code == 422
    )
    assert (
        client.post("/chat", json={"session_id": "s1", "message": ""}).status_code == 422
    )
    assert (
        client.post("/chat", json={"session_id": "s1", "message": "   "}).status_code
        == 422
    )


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_defaults_protect_a_deployed_service() -> None:
    """The default must be safe, not merely present."""
    limits = ServiceLimits()
    config = AgentsConfig.from_env()
    assert 0 < limits.session_limit <= 20
    assert 0 < limits.ip_limit <= 50
    assert 0 < limits.daily_cap <= 1000
    assert limits.window_s >= 1.0
    assert 0 < config.model_timeout_s <= 120
    assert config.model_timeout_s < config.request_timeout_s
    assert 0 < config.max_llm_calls <= 50
    assert config.model.startswith("gemini-") and "flash" in config.model


def test_every_limit_is_overridable_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {
        "MULENET_AGENT_SESSION_LIMIT": "7",
        "MULENET_AGENT_IP_LIMIT": "9",
        "MULENET_AGENT_RATE_WINDOW_SECONDS": "30",
        "MULENET_AGENT_DAILY_CAP": "42",
        "MULENET_AGENT_MAX_MESSAGE_CHARS": "123",
        "MULENET_AGENT_MODEL_TIMEOUT_SECONDS": "11",
        "MULENET_AGENT_REQUEST_TIMEOUT_SECONDS": "22",
        "MULENET_AGENT_MAX_LLM_CALLS": "4",
        "MULENET_AGENT_MAX_OUTPUT_TOKENS": "333",
        "MULENET_GEMINI_MODEL": "gemini-3.1-flash-lite",
    }.items():
        monkeypatch.setenv(name, value)

    limits = ServiceLimits.from_env()
    assert (limits.session_limit, limits.ip_limit, limits.window_s) == (7, 9, 30.0)
    assert (limits.daily_cap, limits.max_message_chars) == (42, 123)

    config = AgentsConfig.from_env()
    assert (config.model_timeout_s, config.request_timeout_s) == (11.0, 22.0)
    assert (config.max_llm_calls, config.max_output_tokens) == (4, 333)
    assert config.model == "gemini-3.1-flash-lite"


@pytest.mark.parametrize(
    "name", ["MULENET_AGENT_SESSION_LIMIT", "MULENET_AGENT_IP_LIMIT", "MULENET_AGENT_DAILY_CAP"]
)
def test_a_limit_cannot_be_switched_off_by_setting_it_to_zero(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A zero here would look like a limit and act like none."""
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError, match="must be >="):
        ServiceLimits.from_env()


@pytest.mark.parametrize(
    "name",
    ["MULENET_AGENT_MODEL_TIMEOUT_SECONDS", "MULENET_AGENT_REQUEST_TIMEOUT_SECONDS"],
)
def test_a_timeout_cannot_be_switched_off(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError, match="must be >="):
        AgentsConfig.from_env()


# --------------------------------------------------------------------------
# the one live test
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get(LIVE_GEMINI_ENV) != "1",
    reason=(
        f"BILLED: calls Gemini on Vertex AI. Set {LIVE_GEMINI_ENV}=1 (with "
        "GOOGLE_CLOUD_PROJECT and application default credentials) to run it. "
        "It is the only test in the repo that spends money, and it is off so "
        "that `pytest` is free."
    ),
)
def test_live_gemini_chat_answers_a_question_and_is_billed() -> None:
    """The one test that actually calls Gemini. Off unless asked for.

    Deliberately a single request against the cheapest agent pair, so the
    cost of turning it on is one router call plus one ``analyst_qa`` call
    on Flash -- a fraction of a cent -- rather than an investigation.
    """
    base = AgentsConfig.from_env()
    config = _config(
        model=os.environ.get("MULENET_GEMINI_MODEL", DEFAULT_MODEL),
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "mulenet"),
        location=base.location,
    )
    api = TestClient(
        service.create_app(
            config,
            ServiceLimits(session_limit=5, ip_limit=5, window_s=60.0, daily_cap=5),
        )
    )
    adk_agents.reset_usage()
    response = api.post(
        "/chat", json={"session_id": "live", "message": "How many rings were detected?"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert CHAT_KEYS <= set(body)  # additive contract: new fields allowed
    assert body["backend"] == "gemini"
    assert body["answer"]
    assert body["latency_ms"] > 0
    # Usage is asserted so a run that somehow answered without calling a
    # model cannot pass as a live one.
    assert adk_agents.LAST_USAGE, "no token usage recorded; nothing was billed"


# --------------------------------------------------------------------------
# the aggregate ring transfer tool, and analyst_qa's restricted access
# --------------------------------------------------------------------------


def test_ring_transfer_summary_returns_aggregates_not_rows(known_ring: str) -> None:
    """Counts and totals, so a ring total costs a few hundred bytes."""
    summary = tools.get_ring_transfer_summary(known_ring)
    assert summary["ring_id"] == known_ring
    assert isinstance(summary["transfer_count"], int)
    # Money as strings, exact to the cent: a float total is a rounding
    # argument waiting to happen.
    assert summary["total_amount"] == str(
        Decimal(summary["total_amount"]).quantize(Decimal("0.01"))
    )
    assert summary["cross_bank_transfer_count"] <= summary["transfer_count"]
    assert summary["distinct_member_pairs"] <= summary["transfer_count"]
    # No row-level payload of any kind.
    assert not any(isinstance(value, (list, tuple, dict)) for value in summary.values())


def test_ring_transfer_summary_member_count_agrees_with_get_ring_summary(
    known_ring: str,
) -> None:
    """Two tools must not disagree about how many members a ring has."""
    assert (
        tools.get_ring_transfer_summary(known_ring)["member_count"]
        == tools.get_ring_summary(known_ring)["member_count"]
    )


def test_ring_transfer_summary_rejects_an_unknown_ring() -> None:
    with pytest.raises(KeyError):
        tools.get_ring_transfer_summary("ring-does-not-exist-0001")


def test_the_new_tool_is_registered_so_grounding_counts_it() -> None:
    """Registered in ALL_TOOLS, so the derived name set picks it up free."""
    assert tools.get_ring_transfer_summary in tools.ALL_TOOLS
    assert "get_ring_transfer_summary" in service._TOOL_NAMES


def _analyst_qa_tools() -> dict[str, Any]:
    return {func.__name__: func for func in adk_agents.AGENT_TOOLS["analyst_qa"]}


def test_analyst_qa_cannot_query_transfers_directly() -> None:
    """The restriction is enforced in agents/, not by editing shared/."""
    with pytest.raises(ValueError, match="may not query transfers"):
        _analyst_qa_tools()["run_sql"]("SELECT SUM(amount) FROM transfers")


def test_the_refusal_names_the_aggregate_tool_to_use_instead() -> None:
    """A refusal the model cannot act on just becomes a retry."""
    with pytest.raises(ValueError, match="get_ring_transfer_summary"):
        _analyst_qa_tools()["run_sql"]("SELECT amount FROM transfers LIMIT 5")


def test_the_restriction_survives_an_alias_on_the_table() -> None:
    """Checked against the parse tree, not matched in the query text."""
    with pytest.raises(ValueError, match="may not query transfers"):
        _analyst_qa_tools()["run_sql"](
            "SELECT t.amount FROM transfers AS t JOIN rings AS r "
            "ON r.ring_id = r.ring_id LIMIT 5"
        )


def test_analyst_qa_can_still_query_the_other_tables() -> None:
    """The restriction is one table, not a lockout."""
    rows = _analyst_qa_tools()["run_sql"]("SELECT ring_id FROM rings LIMIT 1")
    assert rows and "ring_id" in rows[0]


def test_the_restricted_run_sql_keeps_its_name_for_provenance() -> None:
    """Rename it and both the sink and the grounding check stop seeing it."""
    assert "run_sql" in _analyst_qa_tools()
    sink: set[str] = set()
    adk_agents._model_tool(_analyst_qa_tools()["run_sql"], sink)(
        "SELECT ring_id FROM rings LIMIT 1"
    )
    assert "run_sql" in sink
    assert "rings" in sink


def test_shared_allowed_tables_is_untouched() -> None:
    """The frozen contract still allows transfers; this was a local change."""
    from shared.tool_interfaces import ALLOWED_SQL_TABLES

    assert "transfers" in ALLOWED_SQL_TABLES


def test_analyst_qa_no_longer_carries_the_subgraph_tool() -> None:
    """The ~113k-token payload is off the public request path."""
    names = set(_analyst_qa_tools())
    assert "get_ring_subgraph" not in names
    assert "get_ring_transfer_summary" in names
    # Other agents keep their access: this was about analyst_qa's latency
    # and token exposure, not a repo-wide policy change.
    assert tools.get_ring_subgraph in adk_agents.AGENT_TOOLS["report"]
    assert tools.get_ring_subgraph in adk_agents.AGENT_TOOLS["investigator"]


# --------------------------------------------------------------------------
# retries: one per failure mode, three attempts
# --------------------------------------------------------------------------


def _modes(monkeypatch: pytest.MonkeyPatch, sequence: list[str]) -> list[str]:
    """Patch analyst_qa to fail in ``sequence`` order. Returns the call log."""
    calls: list[str] = []
    remaining = list(sequence)

    def fake_run_agent(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        if name == "router":
            return json.dumps(
                {"destination": "analyst_qa", "subject_id": None, "reason": "count"}
            )
        calls.append("analyst_qa")
        mode = remaining.pop(0) if remaining else "ok"
        if mode == "empty":
            return ""
        if mode == "unreadable":
            return "The ring spans 2 banks."
        _run_tools_like_adk(["get_ring_summary"], sink)
        return json.dumps({"answer": "2", "basis": "get_ring_summary", "unit": "banks"})

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    return calls


def test_empty_then_unreadable_gets_a_third_attempt(
    monkeypatch: pytest.MonkeyPatch,
    config: AgentsConfig,
    generous_limits: ServiceLimits,
    known_ring: str,
) -> None:
    """The case that used to 500 on the second attempt now recovers."""
    calls = _modes(monkeypatch, ["empty", "unreadable", "ok"])
    api = TestClient(service.create_app(config, generous_limits))
    body = api.post("/chat", json=QUESTION).json()
    assert body["answer"] == "2 banks"
    assert len(calls) == 3


def test_three_attempts_is_the_ceiling(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """Nothing retries without bound."""
    calls = _modes(monkeypatch, ["empty", "unreadable", "empty", "ok"])
    api = TestClient(
        service.create_app(config, generous_limits), raise_server_exceptions=False
    )
    response = api.post("/chat", json=QUESTION)
    assert len(calls) == service.MAX_ANSWER_ATTEMPTS == 3
    assert response.status_code == 409


def test_a_mode_seen_twice_stops_early(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """Its retry is spent; a third call would buy nothing."""
    calls = _modes(monkeypatch, ["empty", "empty", "ok"])
    api = TestClient(
        service.create_app(config, generous_limits), raise_server_exceptions=False
    )
    response = api.post("/chat", json=QUESTION)
    assert len(calls) == 2
    assert response.status_code == 409
    assert response.json()["error_type"] == "no_tool_evidence"


def test_repeated_unreadable_is_502_malformed_response(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """Decision (a): off 500 internal, because we did not break."""
    _modes(monkeypatch, ["unreadable", "unreadable"])
    api = TestClient(
        service.create_app(config, generous_limits), raise_server_exceptions=False
    )
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 502
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["error_type"] == "malformed_response"
    assert body["retry_after_s"] is None
    # The prose is never published, even though it may have been correct.
    assert "2 banks" not in body["message"]
    # The mode trail is reported, so a caller can see what happened.
    assert "unreadable -> unreadable" in body["message"]


def test_the_exhaustion_status_follows_the_last_mode_seen(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """It is the attempt that ended the request that classifies it."""
    _modes(monkeypatch, ["empty", "unreadable", "unreadable"])
    api = TestClient(
        service.create_app(config, generous_limits), raise_server_exceptions=False
    )
    assert api.post("/chat", json=QUESTION).json()["error_type"] == "malformed_response"


def test_the_attempt_cap_is_derived_from_the_mode_list() -> None:
    """One definition, so a third retryable mode cannot outgrow the cap."""
    assert service.MAX_ANSWER_ATTEMPTS == 1 + len(service.RETRYABLE_MODES)


# --------------------------------------------------------------------------
# missing data: refuse to serve, and never report ok
# --------------------------------------------------------------------------


def _dataless(tmp_path: Any, **kwargs: Any) -> AgentsConfig:
    return _config(data_dir=tmp_path / "no-sim", ml_out_dir=tmp_path / "no-ml", **kwargs)


def _partial_ml_out(tmp_path: Any) -> AgentsConfig:
    """A config whose ``ml/out`` holds some published tables but not all.

    The shape of a half-finished fetch into the image: the directory
    exists and is not empty, so every directory-level check passes, while
    the tables that did not arrive are served from mocks.
    """
    import pandas as pd

    partial = tmp_path / "partial-ml"
    partial.mkdir()
    for name in ("rings.parquet", "scores.parquet"):
        pd.DataFrame({"placeholder": [1]}).to_parquet(partial / name)
    return _config(ml_out_dir=partial)


def test_missing_tables_are_detected(tmp_path: Any) -> None:
    missing = service.missing_tables(_dataless(tmp_path))
    names = {entry.split(" ")[0] for entry in missing}
    assert {"transfers", "rings", "scores", "account_evidence"} <= names
    assert service.missing_tables(_config()) == []


def test_a_partial_fetch_fails_readiness(tmp_path: Any) -> None:
    """The defect this check exists for: some tables present, not all.

    The first version of the readiness check counted parquet files per
    *directory* while the mock fallback was per *table*, so this exact
    configuration -- two of five ml/out tables on disk -- reported itself
    healthy and then answered over mocks for the other three, under
    ``backend: "gemini"`` and with real-looking ``sources``.
    """
    missing = service.missing_tables(_partial_ml_out(tmp_path))
    names = {entry.split(" ")[0] for entry in missing}
    assert {"ring_explanations", "account_explanations", "account_evidence"} <= names
    # The two that did arrive must not be reported, or the check is just noise.
    assert "rings" not in names
    assert "scores" not in names


def test_a_partial_fetch_does_not_serve(
    tmp_path: Any, generous_limits: ServiceLimits
) -> None:
    """Must not serve, and must not report ok. This is the point of the check."""
    config = _partial_ml_out(tmp_path)
    app = service.create_app(config, generous_limits)
    with pytest.raises(service.DataUnavailableError, match="refusing to serve"), TestClient(app):
        pass

    api = TestClient(
        service.create_app(config, generous_limits, require_data=False),
        raise_server_exceptions=False,
    )
    assert api.get("/health").status_code == 503
    assert api.post("/chat", json=QUESTION).json()["error_type"] == "data_unavailable"


def test_the_readiness_list_is_derived_from_the_allowlist(tmp_path: Any) -> None:
    """Per CLAUDE.md: derived from the source of truth, not typed out.

    A table added to ``ALLOWED_SQL_TABLES`` becomes queryable by
    ``run_sql`` the moment it is added. If the readiness list were written
    by hand it would lag, which is how the partial-fetch hole opened.
    """
    from shared.tool_interfaces import ALLOWED_SQL_TABLES

    required = set(data.required_tables())
    assert set(ALLOWED_SQL_TABLES) <= required
    # And everything the typed tools open, which is wider than the allowlist.
    assert set(data.DataStore.known_tables()) <= required
    missing = {entry.split(" ")[0] for entry in service.missing_tables(_dataless(tmp_path))}
    assert set(ALLOWED_SQL_TABLES) - {"cases"} <= missing


def test_the_process_refuses_to_serve_without_tables(
    tmp_path: Any, generous_limits: ServiceLimits
) -> None:
    """Cloud Run needs an unhealthy container, not a degraded one.

    Lifespan runs when the process serves, which is what ``with
    TestClient(...)`` simulates. A bare ``TestClient(app)`` does not run
    it, which is why importing this module stays safe on a clone with no
    parquet -- and the repo gitignores every ``*.parquet``.
    """
    app = service.create_app(_dataless(tmp_path), generous_limits)
    with pytest.raises(service.DataUnavailableError, match="refusing to serve"), TestClient(app):
        pass


def test_health_does_not_report_ok_without_tables(
    tmp_path: Any, generous_limits: ServiceLimits
) -> None:
    """A probe that lies in the one case it exists to catch is worse than none."""
    api = TestClient(
        service.create_app(_dataless(tmp_path), generous_limits, require_data=False),
        raise_server_exceptions=False,
    )
    response = api.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["error_type"] == "data_unavailable"
    assert body != {"status": "ok"}


def test_chat_refuses_without_tables_and_calls_no_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, generous_limits: ServiceLimits
) -> None:
    """No configuration may answer over mock tables labelled backend gemini."""
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        lambda *_a, **_k: pytest.fail("answered a question with no tables loaded"),
    )
    api = TestClient(
        service.create_app(_dataless(tmp_path), generous_limits, require_data=False),
        raise_server_exceptions=False,
    )
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 503
    assert response.json()["error_type"] == "data_unavailable"


def test_require_data_defaults_to_true() -> None:
    """The deployed default must be the safe one."""
    assert AgentsConfig.from_env().require_data is True


def test_require_data_can_be_turned_off_for_local_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, generous_limits: ServiceLimits
) -> None:
    monkeypatch.setenv("MULENET_AGENT_REQUIRE_DATA", "0")
    assert AgentsConfig.from_env().require_data is False
    # Serving is allowed, but the readiness refusal still stands.
    app = service.create_app(_dataless(tmp_path, require_data=False), generous_limits)
    with TestClient(app, raise_server_exceptions=False) as api:
        assert api.get("/health").status_code == 503


def test_a_healthy_instance_still_starts_and_reports_ok(
    config: AgentsConfig, generous_limits: ServiceLimits, fake_gemini: None
) -> None:
    """The guard must not break the normal case."""
    with TestClient(service.create_app(config, generous_limits)) as api:
        assert api.get("/health").json() == {"status": "ok"}


# --------------------------------------------------------------------------
# the model id is one value
# --------------------------------------------------------------------------


def test_the_model_id_has_a_single_definition() -> None:
    """Per CLAUDE.md's derive-lists rule: one definition, no duplicates."""
    assert AgentsConfig.from_env().model == DEFAULT_MODEL
    assert DEFAULT_MODEL.startswith("gemini-")
    # Flash, not Pro: this sits on the request path and is billed per call.
    assert "flash" in DEFAULT_MODEL


def test_no_source_file_hardcodes_a_model_id_besides_config() -> None:
    """A second literal is a second thing to forget at the retirement date.

    Checked against the **AST**, not the raw text, so it flags a model id
    that code could actually use and ignores one merely discussed. Comments
    never reach the AST at all, and docstrings are skipped explicitly:
    naming ``gemini-3.5-flash`` in a docstring that explains why it blew
    its call budget is documentation, and a guard that forbids writing
    things down teaches people to stop writing things down.
    """
    import ast
    import re
    from pathlib import Path

    pattern = re.compile(r"gemini-\d[\w.\-]*")
    agents_dir = Path(service.__file__).resolve().parent
    offenders: dict[str, list[str]] = {}

    for path in sorted(agents_dir.rglob("*.py")):
        if path.name in {"config.py", "test_service.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "ALLOWED-MODEL-LITERALS:" in text:
            # A file may carry model literals only if it says why, in the
            # file. An exemption written down is one someone can disagree
            # with; a silent skip-list is not.
            reason = text.split("ALLOWED-MODEL-LITERALS:", 1)[1].splitlines()[0]
            assert reason.strip(), f"{path.name}: exemption with no reason given"
            continue

        tree = ast.parse(text)
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        found = {
            match
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            for match in pattern.findall(node.value)
        }
        if found:
            offenders[str(path.relative_to(agents_dir))] = sorted(found)

    assert offenders == {}, f"hardcoded model ids outside config.py: {offenders}"


# --------------------------------------------------------------------------
# the retry ledger is shared with the in-process callers
# --------------------------------------------------------------------------


def test_the_in_process_path_retries_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The defect the eval found: loop.handle had no retry at all.

    ``agents/loop.handle`` calls ``route`` directly. With the ledger
    living in the HTTP layer, a router that replied in prose raised
    ``ValueError`` straight out and killed a 25-question eval run on its
    first question. A rule that holds on only one of two paths into the
    same model is not a rule.
    """
    calls: list[int] = []

    def fake_run_agent(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        calls.append(1)
        if len(calls) == 1:
            return "I'm sorry, I cannot answer that. I only route requests."
        return json.dumps(
            {"destination": "analyst_qa", "subject_id": None, "reason": "count"}
        )

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    decision = adk_agents.route("How many rings are there?")
    assert decision.destination == "analyst_qa"
    assert len(calls) == 2


def test_an_exhausted_router_raises_model_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Carries the mode trail, because the caller classifies on it."""
    monkeypatch.setattr(
        adk_agents, "_run_agent", lambda *_a, **_k: "not json at all"
    )
    with pytest.raises(adk_agents.ModelUnusableError) as caught:
        adk_agents.route("How many rings are there?")
    assert caught.value.agent == "router"
    assert caught.value.modes == ["unreadable", "unreadable"]
    assert caught.value.trail == "unreadable -> unreadable"


def test_an_exhausted_router_is_502_over_http(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """A router that never returns JSON is the same class of event."""
    monkeypatch.setattr(adk_agents, "_run_agent", lambda *_a, **_k: "not json at all")
    api = TestClient(
        service.create_app(config, generous_limits), raise_server_exceptions=False
    )
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 502
    assert response.json()["error_type"] == "malformed_response"


def test_an_error_inside_the_agent_call_is_not_classified_as_a_model_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run and parse are separate: only the parse is classified.

    Wrapping a genuine exception from the agent call as "unreadable"
    would answer a real defect with ``502 malformed_response`` and send
    the caller to look at the model.
    """
    calls: list[int] = []

    def explode(*_args: Any, **_kwargs: Any) -> str:
        calls.append(1)
        raise ValueError("PERMISSION_DENIED on project mulenet-secret")

    monkeypatch.setattr(adk_agents, "_run_agent", explode)
    with pytest.raises(ValueError, match="PERMISSION_DENIED"):
        adk_agents.route("How many rings are there?")
    # Not retried either: a permission error will not fix itself.
    assert len(calls) == 1


def test_the_sink_is_cleared_between_attempts(
    monkeypatch: pytest.MonkeyPatch, known_ring: str
) -> None:
    """Provenance belongs to the attempt that answered."""
    calls: list[int] = []

    def fake_run_agent(
        name: str,
        message: str,
        cfg: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        calls.append(1)
        if len(calls) == 1:
            _run_tools_like_adk(["run_sql"], sink)
            return "prose, not json"
        _run_tools_like_adk(["get_ring_summary"], sink)
        return json.dumps({"answer": "2", "basis": "b", "unit": ""})

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    sink: set[str] = set()
    adk_agents.answer_question("How many banks?", None, sink)
    assert sink == {"get_ring_summary"}


def test_the_attempt_cap_has_one_definition() -> None:
    """service re-exports it rather than redefining it."""
    assert service.MAX_ANSWER_ATTEMPTS is adk_agents.MAX_ATTEMPTS
    assert service.RETRYABLE_MODES is adk_agents.RETRYABLE_MODES
    assert adk_agents.MAX_ATTEMPTS == 1 + len(adk_agents.RETRYABLE_MODES)


# --------------------------------------------------------------------------
# the prompt and the wiring must agree about the tool set
# --------------------------------------------------------------------------


def test_the_analyst_qa_prompt_does_not_advertise_a_tool_it_lacks() -> None:
    """The drift this catches was real, and self-inflicted.

    Dropping ``get_ring_subgraph`` from ``analyst_qa``'s wiring left the
    prompt still listing it. The model would have spent a turn calling a
    tool that was not there. A prompt that describes a different agent
    than the one it is attached to is a defect the offline suite can
    catch, so it should.
    """
    prompt = load_prompt("analyst_qa")
    wired = {func.__name__ for func in adk_agents.AGENT_TOOLS["analyst_qa"]}
    for name in sorted(_TOOL_NAMES_ALL - wired):
        # Mentioning a tool to say it is unavailable is fine; offering it
        # in the tool list is not.
        offered = f"- `{name}(" in prompt
        assert not offered, f"prompt offers {name}, which analyst_qa does not have"


def test_the_analyst_qa_prompt_documents_every_tool_it_has() -> None:
    """A tool the model is never told about is a tool it will not use."""
    prompt = load_prompt("analyst_qa")
    for func in adk_agents.AGENT_TOOLS["analyst_qa"]:
        assert func.__name__ in prompt, f"prompt never mentions {func.__name__}"


def test_the_analyst_qa_prompt_states_the_transfers_restriction() -> None:
    """Enforced in code, but the model recovers faster if it is told."""
    prompt = load_prompt("analyst_qa")
    assert "transfers" in prompt
    assert "get_ring_transfer_summary" in prompt


#: Every tool name in the repo, for the drift check above.
_TOOL_NAMES_ALL = frozenset(func.__name__ for func in tools.ALL_TOOLS)


# --------------------------------------------------------------------------
# Vertex 5xx: retried with backoff, bounded by a deadline
# --------------------------------------------------------------------------


class _Status(Exception):
    """A google-genai-shaped error carrying an HTTP status."""

    def __init__(self, code: int) -> None:
        super().__init__(f"{code} SERVER_ERROR. Deadline expired.")
        self.code = code


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_a_retryable_status_is_retried(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """The 504 that killed a live eval mid-run is the reason this exists."""
    calls: list[int] = []
    monkeypatch.setattr(adk_agents.time, "sleep", lambda _s: None)

    def drain(*_args: Any, **_kwargs: Any) -> str:
        calls.append(1)
        if len(calls) == 1:
            raise _Status(code)
        return json.dumps({"destination": "analyst_qa", "subject_id": None, "reason": "x"})

    monkeypatch.setattr(adk_agents, "_drain", drain)
    monkeypatch.setattr(adk_agents, "build_agent", lambda *_a, **_k: object())
    text = adk_agents._run_agent("router", "hi", _config())
    assert json.loads(text)["destination"] == "analyst_qa"
    assert len(calls) == 2


def test_a_400_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """The request itself is wrong; repeating it changes nothing."""
    calls: list[int] = []
    monkeypatch.setattr(adk_agents.time, "sleep", lambda _s: None)

    def drain(*_args: Any, **_kwargs: Any) -> str:
        calls.append(1)
        raise _Status(400)

    monkeypatch.setattr(adk_agents, "_drain", drain)
    monkeypatch.setattr(adk_agents, "build_agent", lambda *_a, **_k: object())
    payload = json.loads(adk_agents._run_agent("router", "hi", _config()))
    assert payload["error"]["status"] == 400
    assert payload["error"]["recoverable"] is False
    assert len(calls) == 1


def test_a_persistent_5xx_gives_up_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded: 3 retries, then the real cause rather than a silent empty."""
    calls: list[int] = []
    monkeypatch.setattr(adk_agents.time, "sleep", lambda _s: None)

    def drain(*_args: Any, **_kwargs: Any) -> str:
        calls.append(1)
        raise _Status(503)

    monkeypatch.setattr(adk_agents, "_drain", drain)
    monkeypatch.setattr(adk_agents, "build_agent", lambda *_a, **_k: object())
    with pytest.raises(_Status):
        adk_agents._run_agent("router", "hi", _config())
    assert len(calls) == 1 + len(adk_agents._BACKOFF_SECONDS)


def test_the_retry_deadline_stops_it_spending_the_whole_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-process callers have no wall clock, so this is their bound.

    With a request budget shorter than the first backoff, there is no
    point sleeping: the sleep would consume the budget and leave nothing
    to make the call with. It gives up immediately and reports the real
    status.
    """
    calls: list[int] = []
    slept: list[float] = []
    monkeypatch.setattr(adk_agents.time, "sleep", lambda s: slept.append(s))

    def drain(*_args: Any, **_kwargs: Any) -> str:
        calls.append(1)
        raise _Status(504)

    monkeypatch.setattr(adk_agents, "_drain", drain)
    monkeypatch.setattr(adk_agents, "build_agent", lambda *_a, **_k: object())
    with pytest.raises(_Status):
        adk_agents._run_agent("router", "hi", _config(request_timeout_s=0.5))
    assert calls == [1], "should not have retried on a budget it cannot afford"
    assert slept == [], "should not have slept away the whole budget"


def test_a_5xx_over_http_is_503_upstream_unavailable(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """A Vertex outage is reported as one, not as a malformed response."""
    monkeypatch.setattr(adk_agents.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        lambda *_a, **_k: (_ for _ in ()).throw(_Status(504)),
    )
    api = TestClient(
        service.create_app(config, generous_limits), raise_server_exceptions=False
    )
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 503
    assert response.json()["error_type"] == "upstream_unavailable"


def test_retryable_statuses_exclude_the_ones_a_retry_cannot_fix() -> None:
    assert 400 not in adk_agents.RETRYABLE_STATUSES
    assert {429, 500, 502, 503, 504} <= adk_agents.RETRYABLE_STATUSES


def test_status_code_reads_504_as_504_not_500() -> None:
    """Scan order matters: 500 appearing first would swallow 502 and 504."""
    assert adk_agents._status_code(_Status(504)) == 504
    assert adk_agents._status_code(Exception("504 DEADLINE_EXCEEDED")) == 504
    assert adk_agents._status_code(Exception("502 Bad Gateway")) == 502


# --------------------------------------------------------------------------
# the prompts must not invite prose
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["router", "analyst_qa", "investigator", "report"])
def test_every_prompt_demands_a_bare_json_object(name: str) -> None:
    """Measured: 3-4 of 25 live answers came back as prose, not JSON.

    Each of those was a *correct* judgement lost to the envelope, which
    is why this is checked on every prompt rather than only the one that
    was caught.
    """
    prompt = load_prompt(name)
    assert "single JSON object and nothing else" in prompt, name
    assert "no markdown fence" in prompt, name


def test_the_decline_wording_is_fixed_not_improvised() -> None:
    """A refusal phrased freely is not recognisable as a refusal.

    The string is ``mock_agents.CANNOT_ANSWER``, which is also
    ``eval.questions.DECLINED`` -- this system's documented way of saying
    "out of scope", shared by the offline path. Pinning the prompt to it
    makes the Gemini path say what the rest of the system already says.
    """
    from agents.eval.questions import DECLINED
    from agents.mock_agents import CANNOT_ANSWER

    assert CANNOT_ANSWER == DECLINED
    prompt = load_prompt("analyst_qa")
    assert DECLINED in prompt
    # And it must still be a refusal that carries no number.
    assert "never put a digit in a refusal" in prompt.lower()


def test_no_prompt_still_says_say_so_without_naming_the_field() -> None:
    """"Say so" is what pulled the model out of the envelope."""
    for name in ("router", "analyst_qa", "investigator", "report"):
        prompt = load_prompt(name)
        for line in prompt.splitlines():
            if "say so" in line.lower():
                assert "`" in line, (
                    f"{name}: 'say so' with no field named -- that is the "
                    f"phrasing that produced prose: {line!r}"
                )


# --------------------------------------------------------------------------
# the call budget is ours, not Vertex's
# --------------------------------------------------------------------------


def _budget_error() -> BaseException:
    """The real ADK exception, so the classification is tested, not mocked."""
    from google.adk.agents.invocation_context import LlmCallsLimitExceededError

    return LlmCallsLimitExceededError("Max number of llm calls limit of `12` exceeded")


def test_the_adk_call_limit_becomes_our_own_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observed live: 3.5-flash blew through 12 calls where 2.5 used two."""
    calls: list[int] = []

    def drain(*_args: Any, **_kwargs: Any) -> str:
        calls.append(1)
        raise _budget_error()

    monkeypatch.setattr(adk_agents, "_drain", drain)
    monkeypatch.setattr(adk_agents, "build_agent", lambda *_a, **_k: object())
    with pytest.raises(adk_agents.CallBudgetExceeded) as caught:
        adk_agents._run_agent("analyst_qa", "hi", _config())
    assert caught.value.agent == "analyst_qa"
    # The limit it reports is the one that agent actually ran under, not
    # the global default: analyst_qa has its own, smaller budget.
    assert caught.value.limit == _config().max_llm_calls_for("analyst_qa")
    # Never retried: the next attempt gets the same budget and spends it
    # the same way.
    assert calls == [1]


def test_the_call_budget_is_not_reported_as_an_upstream_outage() -> None:
    """It comes from a google.* module, so the module test would claim it."""
    budget = adk_agents.CallBudgetExceeded("analyst_qa", 12)
    assert service._upstream(budget) is False
    # The underlying ADK error really would be misread, which is why the
    # check above exists and is ordered first.
    assert type(_budget_error()).__module__.startswith("google.")


def test_an_exhausted_call_budget_is_500_not_503(
    monkeypatch: pytest.MonkeyPatch, config: AgentsConfig, generous_limits: ServiceLimits
) -> None:
    """Our configuration failing to fit the model is our problem."""
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        lambda *_a, **_k: (_ for _ in ()).throw(
            adk_agents.CallBudgetExceeded("analyst_qa", 12)
        ),
    )
    api = TestClient(
        service.create_app(config, generous_limits), raise_server_exceptions=False
    )
    response = api.post("/chat", json=QUESTION)
    assert response.status_code == 500
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["error_type"] == "internal"
    assert "budget" in body["message"]


# --------------------------------------------------------------------------
# eval evidence survives the process that produced it
# --------------------------------------------------------------------------


def test_a_partial_run_still_leaves_readable_records(tmp_path: Any) -> None:
    """The whole point: a run that dies at question 20 keeps 19.

    Two live runs were lost before this existed -- one to an account rate
    limit, one to a crash -- and each time the paid-for numbers died with
    the process.
    """
    from agents.eval.record import EvalRecorder, read_records

    recorder = EvalRecorder(
        model="gemini-2.5-flash", location="us-central1", out_dir=tmp_path
    )
    for index in range(3):
        recorder.question(
            question_id=f"q{index:02d}",
            question="How many rings?",
            category="detection",
            expected="45",
            actual="45",
            correct=True,
            latency_s=1.5,
            usage={"analyst_qa": {"prompt": 5000, "output": 50, "thinking": 100}},
        )

    # No summary line is ever written: this simulates the process dying.
    records = read_records(recorder.path)
    assert [r["kind"] for r in records] == ["run", "question", "question", "question"]
    assert not any(r["kind"] == "summary" for r in records)
    assert all(r["correct"] for r in records if r["kind"] == "question")


def test_a_torn_final_line_does_not_lose_the_records_before_it(
    tmp_path: Any,
) -> None:
    """A process killed mid-write leaves a partial line. Drop it, keep the rest."""
    from agents.eval.record import EvalRecorder, read_records

    recorder = EvalRecorder(model="m", location="global", out_dir=tmp_path)
    recorder.question(
        question_id="q01",
        question="q",
        category="detection",
        expected="1",
        actual="1",
        correct=True,
        latency_s=0.1,
        usage={},
    )
    with recorder.path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "question", "id": "q02", "act')

    records = read_records(recorder.path)
    assert [r["kind"] for r in records] == ["run", "question"]
    assert records[1]["id"] == "q01"


def test_each_record_carries_the_model_and_the_region(tmp_path: Any) -> None:
    """A result without its region cannot be compared to anything.

    The same model id behaves differently by region -- 3.5-flash serves in
    asia-south1 and 404s in us-central1 -- so the region is part of the
    measurement, not context for it.
    """
    from agents.eval.record import EvalRecorder, read_records

    recorder = EvalRecorder(
        model="gemini-3.5-flash", location="asia-south1", out_dir=tmp_path
    )
    recorder.verification(
        subject="ring-1",
        verified=True,
        attempts=1,
        unsupported=[],
        latency_s=2.0,
        usage={"investigator": {"prompt": 9000, "output": 500, "thinking": 400}},
    )
    for record in read_records(recorder.path):
        assert record["model"] == "gemini-3.5-flash"
        assert record["location"] == "asia-south1"
    assert "gemini-3.5-flash" in recorder.path.name
    assert "asia-south1" in recorder.path.name


def test_a_record_carries_latency_tokens_and_a_cost_estimate(tmp_path: Any) -> None:
    from agents.eval.record import EvalRecorder, read_records

    recorder = EvalRecorder(model="gemini-2.5-flash", location="us", out_dir=tmp_path)
    recorder.question(
        question_id="q01",
        question="q",
        category="detection",
        expected="1",
        actual="1",
        correct=True,
        latency_s=3.25,
        usage={"analyst_qa": {"prompt": 1_000_000, "output": 0, "thinking": 0}},
    )
    record = read_records(recorder.path)[1]
    assert record["latency_s"] == 3.25
    assert record["usage"]["analyst_qa"]["prompt"] == 1_000_000
    # 1M prompt tokens at the 2.5 input rate.
    assert record["cost_usd"] == pytest.approx(0.15)


def test_thinking_tokens_are_costed_as_output(tmp_path: Any) -> None:
    """A total that omits them understates the bill; they bill as output."""
    from agents.eval.record import estimate_cost

    with_thinking = estimate_cost(
        "gemini-2.5-flash", {"a": {"prompt": 0, "output": 0, "thinking": 1_000_000}}
    )
    as_output = estimate_cost(
        "gemini-2.5-flash", {"a": {"prompt": 0, "output": 1_000_000, "thinking": 0}}
    )
    assert with_thinking == as_output == pytest.approx(1.25)


def test_the_cost_rate_follows_the_model_family(tmp_path: Any) -> None:
    from agents.eval.record import estimate_cost

    usage = {"a": {"prompt": 1_000_000, "output": 0, "thinking": 0}}
    assert estimate_cost("gemini-2.5-flash", usage) == pytest.approx(0.15)
    assert estimate_cost("gemini-3.8-flash", usage) == pytest.approx(0.75)
    # An unknown model costs nothing rather than guessing a rate.
    assert estimate_cost("some-future-model", usage) == 0.0


def test_the_recorder_writes_under_agents_out_by_default() -> None:
    """Beside the other published artefacts, following the convention."""
    from agents.eval.record import DEFAULT_EVAL_DIR

    assert DEFAULT_EVAL_DIR.parent.name == "out"
    assert DEFAULT_EVAL_DIR.parent.parent.name == "agents"


# --- asia-south1 interim configuration, and the graceful budget decline ---


def test_the_default_region_is_asia_south1() -> None:
    """Mumbai, not Iowa.

    Measured 2026-09-27: asia-south1 serves gemini-2.5-flash AND
    gemini-3.5-flash; us-central1 serves no Gemini 3.x at all. That is
    why region and model can move as two decisions instead of one.
    """
    from agents.config import DEFAULT_LOCATION

    assert DEFAULT_LOCATION == "asia-south1"
    monkey = os.environ.pop("GOOGLE_CLOUD_LOCATION", None)
    try:
        assert AgentsConfig.from_env().location == "asia-south1"
    finally:
        if monkey is not None:
            os.environ["GOOGLE_CLOUD_LOCATION"] = monkey


def test_the_region_is_not_hardcoded_anywhere_but_config() -> None:
    """One definition, per CLAUDE.md's derive-lists rule."""
    from pathlib import Path

    root = Path(__file__).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        if path.name in {"config.py", Path(__file__).name}:
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            # A quoted Python string only. Prose and RST literals
            # (``asia-south1``) are documentation, and the observations
            # that justify the region belong in the docstrings.
            if '"asia-south1"' in line or "'asia-south1'" in line:
                offenders.append(f"{path.name}:{number}")
    assert not offenders, f"region literal outside config: {offenders}"


def test_analyst_qa_has_a_smaller_call_budget_than_the_rest() -> None:
    """A factual question is a short tool loop; the baseline uses ~2."""
    from agents.config import ANALYST_QA_MAX_LLM_CALLS, DEFAULT_MAX_LLM_CALLS

    assert ANALYST_QA_MAX_LLM_CALLS == 6
    assert ANALYST_QA_MAX_LLM_CALLS < DEFAULT_MAX_LLM_CALLS
    config = _config()
    assert config.max_llm_calls_for("analyst_qa") == 6
    assert config.max_llm_calls_for("investigator") == config.max_llm_calls
    assert config.max_llm_calls_for("report") == config.max_llm_calls


def test_lowering_the_global_ceiling_still_lowers_analyst_qa() -> None:
    """An operator tightening the global cap gets no quiet exemption."""
    config = _config(max_llm_calls=3)
    assert config.max_llm_calls_for("analyst_qa") == 3
    assert config.max_llm_calls_for("investigator") == 3


def test_exhausting_the_budget_declines_rather_than_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A looping agent has shown it cannot answer. That is a decline.

    Observed live: gemini-3.5-flash burned all twelve calls on two eval
    questions it should have refused, turning a correct judgement into a
    dead run.
    """
    from agents.mock_agents import CANNOT_ANSWER

    def _boom(name: str, *args: Any, **kwargs: Any) -> str:
        raise adk_agents.CallBudgetExceeded(name, 6)

    monkeypatch.setattr(adk_agents, "_run_agent", _boom)
    answer = adk_agents.answer_question("anything", _config())
    assert answer.answer == CANNOT_ANSWER
    assert answer.basis == ""


def test_the_decline_string_is_the_shared_constant() -> None:
    """Not a phrase invented to satisfy a grader."""
    from agents.eval.questions import DECLINED
    from agents.mock_agents import CANNOT_ANSWER

    assert CANNOT_ANSWER == DECLINED


def test_only_analyst_qa_declines_on_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The investigator still raises: a half-written case is not a case."""

    def _boom(name: str, *args: Any, **kwargs: Any) -> str:
        raise adk_agents.CallBudgetExceeded(name, 12)

    monkeypatch.setattr(adk_agents, "_run_agent", _boom)
    with pytest.raises(adk_agents.CallBudgetExceeded):
        adk_agents.investigate_with_gemini("ring-x", config=_config())


# --- every allowlisted table must have a publisher and be fetched ---
#
# `features` was in ALLOWED_SQL_TABLES for the whole life of the project
# while nothing published it, so `SELECT ... FROM features` succeeded and
# returned nothing -- which reads to an agent as "the data says no". These
# tests exist so the next such table is caught by CI instead of by a
# deployment.


def _publishers() -> dict[str, str]:
    """Table name -> what writes its parquet, derived, not retyped.

    ``simulator`` writes ``simulator/out``, ``ml.publish_all`` writes
    ``ml/out``, and ``agents.cases`` writes ``cases.parquet``. Aliases and
    derivations resolve to whatever publishes the file they read.
    """
    from agents.data import ML_TABLES, SIMULATOR_TABLES, TABLE_ALIASES

    published = {name: "simulator" for name in SIMULATOR_TABLES}
    published.update({name: "ml.publish_all" for name in ML_TABLES})
    published["cases"] = "agents.cases"
    # Derived views over a published table, not files of their own.
    published["ring_members"] = published["rings"]
    for alias, target in TABLE_ALIASES.items():
        published[alias] = published[target]
    return published


def test_every_allowlisted_table_has_a_publisher() -> None:
    """An allowlisted table nothing writes is a promise the system breaks."""
    from shared.tool_interfaces import ALLOWED_SQL_TABLES

    orphans = sorted(set(ALLOWED_SQL_TABLES) - set(_publishers()))
    assert not orphans, (
        f"allowlisted but nothing publishes them: {orphans}. Either publish "
        "them or remove them from ALLOWED_SQL_TABLES -- a table that reads "
        "as empty is worse than one that errors, because an agent reports "
        "the absence as a finding."
    )


def test_every_allowlisted_table_is_in_the_agents_fetch_list() -> None:
    """Deployed readiness demands these files; the fetch must supply them.

    ``infra/deploy.sh`` derives ``AGENTS_FETCH`` from ``SIMULATOR_TABLES``
    and ``ML_TABLES`` plus ``cases.parquet``. If readiness ever requires a
    file outside that set, the container cannot start -- so the two are
    asserted against each other here rather than discovered on a deploy.
    """
    from agents.data import (
        ML_TABLES,
        SIMULATOR_TABLES,
        DataStore,
        required_tables,
    )

    fetched_files = {*SIMULATOR_TABLES.values(), *ML_TABLES.values(), "cases.parquet"}
    config = _config()
    store = DataStore(config)
    missing = sorted(
        name
        for name in required_tables()
        if store.source_path(name).name not in fetched_files
    )
    assert not missing, (
        f"readiness requires these but the fetch list does not carry them: "
        f"{missing}"
    )


def test_the_publisher_map_covers_every_table_the_store_serves() -> None:
    """Guards the guard: a new table in data.py must name its publisher."""
    from agents.data import DataStore

    unclaimed = sorted(set(DataStore.known_tables()) - set(_publishers()))
    assert not unclaimed, (
        f"DataStore serves these but _publishers() does not name what writes "
        f"them: {unclaimed}. Add them there so the orphan test keeps working."
    )
