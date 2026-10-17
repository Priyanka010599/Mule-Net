"""Tests for the live evasion-pattern simulator (``POST /simulate``).

Two things are checked that ``agents/test_service.py`` does not need to:
that :mod:`agents.simulate`'s inlined copy of ``simulator.generator.BANKS``
and ``hmac_hex`` has not drifted from the real source (see
``agents/simulate.py``'s module docstring for why a copy exists at all),
and that :func:`agents.simulate.run_simulation` produces the same verdicts
as calling :func:`ml.redteam.evaluate` directly would -- i.e. that routing
through the lite generator stand-in changes nothing about the detection
result, only where the ids come from.

Every test here is offline and free: this path makes no model call at all.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents import service, simulate
from agents.config import AgentsConfig
from agents.limits import ServiceLimits
from agents.test_service import _config  # the vertex-mode config builder
from ml import redteam
from ml.config import MlConfig


def test_the_lite_generator_matches_the_real_one() -> None:
    """``simulator.generator`` is only imported here, in a test -- never on
    the live request path (see ``agents/simulate.py``'s module docstring).
    A drift between the two would be silent everywhere else."""
    from simulator import generator

    assert simulate.BANKS == generator.BANKS
    key = b"drift-check-key"
    for raw in ("a", "redteam:acct:1:00000", ""):
        assert simulate.hmac_hex(key, raw) == generator.hmac_hex(key, raw)


def test_generator_is_patched_to_the_lite_stand_in() -> None:
    """Importing agents.simulate must have redirected ml.redteam._generator,
    or the live path would still reach for simulator.generator (and Pillow)."""
    assert redteam._generator() is simulate._LITE_GENERATOR


def test_bounds_payload_is_read_off_ml_redteam_not_retyped() -> None:
    payload = simulate.bounds_payload()
    assert set(payload["fields"]) == set(redteam.PATTERN_BOUNDS)
    for name, (low, high, default) in redteam.PATTERN_BOUNDS.items():
        field = payload["fields"][name]
        assert field["low"] == low
        assert field["high"] == high
        assert field["default"] == default
        assert field["integer"] == (name in redteam.INTEGER_FIELDS)
    assert set(payload["evadable_clauses"]) == redteam.EVADABLE_CLAUSES


def test_run_simulation_matches_calling_evaluate_directly() -> None:
    """Routing through agents.simulate must not change the detection result."""
    raw = {"collectors": 2, "mules_per_collector": 20, "hop_delay_seconds": 5}
    result = simulate.run_simulation(raw, "passthrough_speed_seconds")

    pattern = redteam._coerce_pattern(raw)
    strategy = redteam.EvasionStrategy(
        name="x",
        description="x",
        targets="x",
        evades_clause="passthrough_speed_seconds",
        why_it_should_evade="x",
        pattern=pattern,
    )
    direct = redteam.evaluate(strategy, simulate._STRATEGY_INDEX, MlConfig.from_env())

    assert result["caught"] == direct.caught
    assert result["caught_by"] == direct.caught_by
    assert result["flagged_accounts"] == direct.flagged_accounts
    assert result["rings_formed"] == direct.rings_formed


def test_a_pattern_that_stays_under_every_floor_is_reported_missed() -> None:
    result = simulate.run_simulation(
        {"collectors": 1, "mules_per_collector": 15, "counterparty_banks": 1},
        "in_degree",
    )
    assert result["caught"] is False
    assert result["caught_by"] == "missed"
    assert "in_degree" in result["reasoning"]


def test_an_out_of_range_field_is_clamped_and_named() -> None:
    result = simulate.run_simulation({"collectors": 99}, "in_degree")
    assert result["pattern"]["collectors"] == redteam.PATTERN_BOUNDS["collectors"][1]
    assert any("collectors" in note for note in result["pattern"]["clamped"])


def test_a_nonsense_evades_clause_is_kept_verbatim() -> None:
    """Same treatment ml.redteam.parse_strategies gives a model's claim."""
    result = simulate.run_simulation({}, "not_a_real_clause")
    assert result["aimed_at_clause"] == "not_a_real_clause"


# --------------------------------------------------------------------------
# through the HTTP service
# --------------------------------------------------------------------------


@pytest.fixture
def dataless_config(tmp_path: Any) -> AgentsConfig:
    """No published tables needed: the simulator never reads them."""
    base = _config()
    return AgentsConfig(
        mode=base.mode,
        data_dir=tmp_path / "no-sim",
        ml_out_dir=tmp_path / "no-ml",
        model=base.model,
        project=base.project,
        location=base.location,
        use_vertex=base.use_vertex,
        require_data=False,
    )


#: Limits high enough not to interfere with a behaviour test -- mirrors
#: agents/test_service.py's fixture of the same name, kept local here
#: rather than imported to avoid a pytest-fixture/import name clash.
_GENEROUS_LIMITS = ServiceLimits(
    session_limit=1000, ip_limit=1000, window_s=60.0, daily_cap=10_000
)


@pytest.fixture
def sim_client(dataless_config: AgentsConfig) -> TestClient:
    return TestClient(
        service.create_app(dataless_config, _GENEROUS_LIMITS, require_data=False)
    )


def test_simulate_bounds_endpoint_needs_no_data(sim_client: TestClient) -> None:
    """Reachable even on an instance /health would refuse -- see
    agents/service.py's simulate_bounds() docstring."""
    response = sim_client.get("/simulate/bounds")
    assert response.status_code == 200
    body = response.json()
    assert set(body["fields"]) == set(redteam.PATTERN_BOUNDS)


def test_simulate_endpoint_computes_a_real_verdict(sim_client: TestClient) -> None:
    response = sim_client.post(
        "/simulate",
        json={
            "collectors": 2,
            "mules_per_collector": 20,
            "counterparty_banks": 2,
            "evades_clause": "passthrough_speed_seconds",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["caught"] is True
    assert body["caught_by"] == "velocity_ring"
    assert body["rings_formed"] == 1
    assert body["pattern"]["collectors"] == 2


def test_simulate_endpoint_makes_no_model_call(
    monkeypatch: pytest.MonkeyPatch, sim_client: TestClient
) -> None:
    from agents import adk_agents

    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        lambda *_a, **_k: pytest.fail("the simulate endpoint called a model"),
    )
    assert sim_client.post("/simulate", json={"collectors": 1}).status_code == 200


def test_simulate_endpoint_never_consults_the_request_guard(
    dataless_config: AgentsConfig,
) -> None:
    """No model call, no spend to bound -- a starved guard must not block it."""
    guard = ServiceLimits(session_limit=0, ip_limit=0, window_s=60.0, daily_cap=0)
    api = TestClient(
        service.create_app(dataless_config, guard, require_data=False)
    )
    assert api.post("/simulate", json={"collectors": 1}).status_code == 200
    assert api.get("/simulate/bounds").status_code == 200


def test_an_omitted_field_takes_its_default_and_is_not_reported_clamped(
    sim_client: TestClient,
) -> None:
    body = sim_client.post("/simulate", json={}).json()
    assert body["pattern"]["clamped"] == []
    assert body["pattern"]["collectors"] == redteam.PATTERN_BOUNDS["collectors"][2]
