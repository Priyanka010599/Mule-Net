"""Item 3: /simulate validates input, and returns the structured verdict and
clause table the UI renders its own copy from."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents import service, simulate
from agents.config import AgentsConfig
from agents.test_simulate import _GENEROUS_LIMITS, _config
from shared.evasion_verdict import CLAUSES


@pytest.fixture
def api(tmp_path: Any) -> TestClient:
    base = _config()
    cfg = AgentsConfig(
        mode=base.mode,
        data_dir=tmp_path / "no-sim",
        ml_out_dir=tmp_path / "no-ml",
        model=base.model,
        project=base.project,
        location=base.location,
        use_vertex=base.use_vertex,
        require_data=False,
    )
    return TestClient(service.create_app(cfg, _GENEROUS_LIMITS, require_data=False))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("collectors", 5),
        ("collectors", 0),
        ("collectors", 1.5),
        ("mules_per_collector", 41),
        ("onward_accounts", 21),
        ("hop_delay_seconds", 604801),
        ("amount_per_inbound_hop", 99),
        ("amount_per_inbound_hop", 1_000_001),
        ("retained_fraction", 1.0),
        ("retained_fraction", -0.1),
        ("counterparty_banks", 4),
    ],
)
def test_out_of_range_input_is_a_422_with_a_field_level_error(
    api: TestClient, field: str, value: Any
) -> None:
    response = api.post("/simulate", json={field: value})
    assert response.status_code == 422
    body = response.json()
    assert body["error_type"] == "validation_error"
    assert [e["field"] for e in body["errors"]] == [field]
    assert "message" in body["errors"][0]


def test_the_documented_extremes_are_accepted(api: TestClient) -> None:
    high = {
        "collectors": 4,
        "mules_per_collector": 40,
        "onward_accounts": 20,
        "hop_delay_seconds": 604800,
        "amount_per_inbound_hop": 1_000_000,
        "retained_fraction": 0.99,
        "counterparty_banks": 3,
    }
    low = {
        "collectors": 1,
        "mules_per_collector": 1,
        "onward_accounts": 1,
        "hop_delay_seconds": 0,
        "amount_per_inbound_hop": 100,
        "retained_fraction": 0,
        "counterparty_banks": 1,
    }
    assert api.post("/simulate", json=high).status_code == 200
    assert api.post("/simulate", json=low).status_code == 200


def test_every_bad_field_is_reported_not_just_the_first(api: TestClient) -> None:
    body = api.post("/simulate", json={"collectors": 9, "counterparty_banks": 7}).json()
    assert sorted(e["field"] for e in body["errors"]) == ["collectors", "counterparty_banks"]


def test_hold_600_measures_1020_because_speed_counts_from_the_first_credit() -> None:
    """The 600 -> 1020 report: 8 mules spaced 60s -> a 420s inbound leg."""
    result = simulate.run_simulation(
        {"collectors": 1, "mules_per_collector": 8, "hop_delay_seconds": 600},
        "passthrough_speed_seconds",
    )
    speed = next(c for c in result["clauses"] if c["clause"] == "passthrough_speed_seconds")
    assert speed["input_field"] == "hop_delay_seconds"
    assert speed["input_value"] == 600
    assert speed["measured"] == 1020
    assert speed["expected_measured"] == 1020
    assert speed["decomposition"] == {
        "hold_seconds": 600,
        "inbound_leg_seconds": 420,
        "inbound_spacing_seconds": 60,
    }
    assert speed["note"] == "speed_counts_from_first_inbound_credit"
    assert (speed["min"], speed["max"], speed["passes"]) == (0.0, 3600.0, True)


def test_a_missed_pattern_names_what_stopped_it_and_whether_that_was_the_aim() -> None:
    raw = {"collectors": 1, "mules_per_collector": 15, "counterparty_banks": 1}
    designed = simulate.run_simulation(raw, "in_degree")
    assert designed["verdict"] == "MISSED"
    assert {"in_degree", "distinct_counterparty_banks"} <= set(designed["failing_clauses"])
    assert designed["aimed_clause"] == "in_degree"
    assert designed["aimed_clause_passed"] is False
    assert designed["is_designed_miss"] is True

    accident = simulate.run_simulation(raw, "passthrough_speed_seconds")
    assert accident["verdict"] == "MISSED"
    assert accident["aimed_clause_passed"] is True  # it aimed at speed, and speed passed
    assert accident["is_designed_miss"] is False  # missed, but not the designed way


def test_a_caught_pattern_has_no_failing_clauses_and_counts_cash_outs() -> None:
    result = simulate.run_simulation(
        {"collectors": 2, "mules_per_collector": 20, "counterparty_banks": 2},
        "passthrough_speed_seconds",
    )
    assert result["verdict"] == "CAUGHT"
    assert result["failing_clauses"] == []
    assert result["is_designed_miss"] is False
    assert result["cash_out_accounts_flagged"] == result["flagged_collectors"] >= 1
    assert result["rings_formed"] == 1
    assert [c["clause"] for c in result["clauses"]] == list(CLAUSES)


def test_thresholds_in_the_clause_table_are_the_rule_constants() -> None:
    from ml import rule

    table = {c["clause"]: c for c in simulate.run_simulation({}, "in_degree")["clauses"]}
    assert table["in_degree"]["min"] == rule.VELOCITY_MIN_IN_DEGREE
    assert table["out_degree"]["min"] == rule.VELOCITY_MIN_OUT_DEGREE
    assert table["distinct_counterparty_banks"]["min"] == rule.MIN_BANKS
    assert table["passthrough_speed_seconds"]["max"] == rule.VELOCITY_MAX_SECONDS
    ratio = table["passthrough_ratio"]
    assert (ratio["min"], ratio["max"]) == rule.VELOCITY_RATIO_RANGE
