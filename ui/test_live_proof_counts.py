"""Account-count scope on the live proof, and input vs measured delay on the verdict."""

from __future__ import annotations

from typing import Any

import pytest

from ui.config import UiConfig
from ui.datasource import DataSource
from ui.models import SimulateVerdict


def _verdict(clauses: list[dict[str, Any]]) -> SimulateVerdict:
    return SimulateVerdict(
        name="n", description="d", targets="t", aimed_at_clause="c", why_it_should_evade="w",
        pattern={
            "collectors": 1, "mules_per_collector": 2, "onward_accounts": 1,
            "hop_delay_seconds": 600, "amount_per_inbound_hop": 100, "retained_fraction": 0.1,
            "counterparty_banks": 1, "reuse_device": False, "reuse_kyc_id": False,
        },
        caught=False, caught_by="", reasoning="r", flagged_accounts=0, flagged_collectors=0,
        rings_formed=0, stage2_alertable=0, stage2_context=0, clause_values={}, clauses=clauses,
    )  # fmt: skip


def _speed(given: float, measured: float | None, leg: float = 420.0) -> dict[str, Any]:
    return {
        "clause": "passthrough_speed_seconds",
        "input_value": given,
        "measured": measured,
        "decomposition": {"hold_seconds": given, "inbound_leg_seconds": leg},
    }


def test_measured_longer_than_input_is_explained_without_blaming_the_detector() -> None:
    v = _verdict([_speed(600, 1020)])
    assert (v.input_delay_seconds, v.measured_delay_seconds) == (600, 1020)
    assert v.delay_added_seconds == 420
    assert "420s were added" in v.delay_explanation
    assert "not detector latency" in v.delay_explanation


def test_equal_delays_carry_no_explanation() -> None:
    v = _verdict([_speed(600, 600)])
    assert v.measured_delay_seconds == 600
    assert v.delay_added_seconds is None and v.delay_explanation == ""


@pytest.mark.parametrize("clauses", [[], [_speed(600, None)]])
def test_no_speed_measurement_leaves_the_fields_null(clauses: list[dict[str, Any]]) -> None:
    v = _verdict(clauses)
    assert v.input_delay_seconds is None and v.measured_delay_seconds is None


def test_enclave_payload_exposes_generated_and_activity_counts() -> None:
    ds = DataSource(UiConfig.from_env())
    ds._snapshot = {"dataset": {"accounts_generated": 30000, "accounts_with_activity": 30000}}
    e = ds.enclave()
    assert (e.accounts_generated, e.accounts_with_activity) == (30000, 30000)
    # The enclave's own, smaller scope keeps its own honest name.
    assert e.accounts_in_enclave_scope == e.account_count
    assert e.edges_in_enclave_scope == e.edge_count
