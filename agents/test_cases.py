"""Tests for the ``cases`` producer.

The property under test is negative and it is the important one: a case
must be impossible to write for an investigation that did not verify.
Several tests therefore assert that construction *fails*, because the
contract refusing the shape is a stronger guarantee than the producer
remembering to check it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agents import tools
from agents.cases import (
    CASE_COLUMNS,
    build_case,
    empty_cases_frame,
    load_cases,
    record_verified_investigation,
    upsert_cases,
)
from agents.config import AgentsConfig
from agents.data import DataStore, default_store
from agents.loop import VerifiedInvestigation, investigate_verified
from agents.verify import Claim, VerificationResult
from shared.contracts import (
    MAX_VERIFICATION_ATTEMPTS,
    Investigation,
    InvestigationStatus,
    case_id_for_ring,
)

RING = "ring-test-cases-0001"


def _verified(attempts: int = 1, claims: int = 3) -> VerifiedInvestigation:
    return VerifiedInvestigation(
        narrative="Flagged on velocity and structure.",
        claims=[Claim(f"c{i}", "ring_bank_count", RING, 2, "") for i in range(claims)],
        evidence=[],
        raw={},
        verification=VerificationResult(passed=True),
        attempts=attempts,
        warnings=[],
        verified=True,
    )


def _withheld() -> VerifiedInvestigation:
    """What loop.py returns when verification fails twice.

    Note the empty claim list: the loop drops the claims along with the
    withheld narrative, and that is what makes the shape unwritable.
    """
    return VerifiedInvestigation(
        narrative="",
        claims=[],
        evidence=[],
        raw={"withheld_narrative": "..."},
        verification=VerificationResult(passed=False, unsupported=["c1: wrong"]),
        attempts=2,
        warnings=["UNVERIFIED"],
        verified=False,
    )


@pytest.fixture
def isolated(tmp_path, monkeypatch) -> AgentsConfig:
    """Point the producer at a temporary directory."""
    monkeypatch.setattr("agents.cases.DEFAULT_OUT_DIR", tmp_path)
    return AgentsConfig.from_env()


# ------------------------------------------------- a case means "verified"


def test_withheld_investigation_writes_nothing(isolated: AgentsConfig) -> None:
    assert record_verified_investigation(RING, _withheld(), isolated) is None
    assert load_cases(isolated).empty


def test_build_case_refuses_a_withheld_outcome() -> None:
    with pytest.raises(ValueError, match="did not verify"):
        build_case(RING, _withheld())


def test_the_contract_itself_refuses_a_withheld_shape() -> None:
    """Belt and braces: even bypassing the producer, it cannot be built.

    ``verified_claim_count >= 1`` is what does it, because a withheld
    outcome carries no claims.
    """
    with pytest.raises(ValidationError):
        Investigation(
            case_id=case_id_for_ring(RING),
            ring_id=RING,
            verified_at=datetime.now(UTC),
            verification_attempts=2,
            verified_claim_count=0,
        )


def test_contract_refuses_a_third_attempt() -> None:
    with pytest.raises(ValidationError):
        Investigation(
            case_id=case_id_for_ring(RING),
            ring_id=RING,
            verified_at=datetime.now(UTC),
            verification_attempts=MAX_VERIFICATION_ATTEMPTS + 1,
            verified_claim_count=1,
        )


def test_contract_refuses_a_non_derived_case_id() -> None:
    with pytest.raises(ValidationError):
        Investigation(
            case_id="not-a-derived-id",
            ring_id=RING,
            verified_at=datetime.now(UTC),
            verification_attempts=1,
            verified_claim_count=1,
        )


# ------------------------------------------------------------- one per ring


def test_case_id_is_derived_from_the_ring() -> None:
    case = build_case(RING, _verified())
    assert case.case_id == case_id_for_ring(RING)


def test_reverification_updates_rather_than_duplicating(
    isolated: AgentsConfig,
) -> None:
    """A second verified investigation is the same case, re-verified."""
    first = record_verified_investigation(RING, _verified(attempts=1, claims=3), isolated)
    assert first is not None
    opened = load_cases(isolated)["opened_at"].iloc[0]

    counts = upsert_cases([build_case(RING, _verified(attempts=2, claims=5))], isolated)
    assert counts == {
        "inserted": 0,
        "updated": 1,
        "skipped_closed": 0,
        "total": 1,
    }

    frame = load_cases(isolated)
    assert len(frame) == 1
    assert frame["verified_claim_count"].iloc[0] == 5
    assert frame["verification_attempts"].iloc[0] == 2
    # The case was not re-opened; it is the same case.
    assert frame["opened_at"].iloc[0] == opened


def test_a_closed_case_is_left_alone(isolated: AgentsConfig) -> None:
    """Closing is an analyst's decision; re-verification must not undo it."""
    closed = Investigation(
        case_id=case_id_for_ring(RING),
        ring_id=RING,
        verified_at=datetime.now(UTC),
        verification_attempts=1,
        verified_claim_count=2,
        status=InvestigationStatus.CLOSED,
        closed_at=datetime.now(UTC),
    )
    upsert_cases([closed], isolated)

    counts = upsert_cases([build_case(RING, _verified(claims=9))], isolated)
    assert counts["skipped_closed"] == 1
    frame = load_cases(isolated)
    assert frame["status"].iloc[0] == InvestigationStatus.CLOSED.value
    assert frame["verified_claim_count"].iloc[0] == 2


# ------------------------------------------------------------------ writing


def test_written_columns_are_in_contract_order(isolated: AgentsConfig) -> None:
    record_verified_investigation(RING, _verified(), isolated)
    assert tuple(load_cases(isolated).columns) == CASE_COLUMNS


def test_manifest_is_written_beside_the_parquet(isolated: AgentsConfig) -> None:
    import json

    from agents.cases import manifest_path

    record_verified_investigation(RING, _verified(), isolated)
    manifest = json.loads(manifest_path(isolated).read_text(encoding="utf-8"))
    assert manifest["tables"]["cases"]["rows"] == 1
    assert (
        manifest["tables"]["cases"]["validated_against"]
        == "shared.contracts.Investigation"
    )


def test_new_case_opens_open_and_unclosed(isolated: AgentsConfig) -> None:
    case = record_verified_investigation(RING, _verified(), isolated)
    assert case is not None
    assert case.status is InvestigationStatus.OPEN
    assert case.closed_at is None


def test_attempts_and_claim_count_come_from_the_outcome(
    isolated: AgentsConfig,
) -> None:
    case = record_verified_investigation(RING, _verified(attempts=2, claims=4), isolated)
    assert case is not None
    assert case.verification_attempts == 2
    assert case.verified_claim_count == 4


# ------------------------------------------------------------- readable back


def test_empty_frame_has_the_full_schema() -> None:
    """The old stub was missing the three verification columns."""
    assert tuple(empty_cases_frame().columns) == CASE_COLUMNS


def test_cases_is_queryable_through_run_sql() -> None:
    assert isinstance(tools.run_sql("SELECT * FROM cases LIMIT 1"), list)


def test_data_layer_reads_the_produced_table() -> None:
    """The published table is real data, not the empty stub."""
    store = default_store()
    frame = store.table("cases")
    assert tuple(frame.columns) == CASE_COLUMNS
    if len(frame):
        assert store.status("cases").is_mock is False


def test_produced_cases_match_the_rings_they_came_from() -> None:
    """Every published case id is the uuid5 of its own ring id."""
    frame = default_store().table("cases")
    if frame.empty:
        pytest.skip("no cases published yet")
    for row in frame.itertuples():
        assert row.case_id == case_id_for_ring(row.ring_id)
        assert row.verified_claim_count >= 1
        assert 1 <= row.verification_attempts <= MAX_VERIFICATION_ATTEMPTS


def test_every_published_case_round_trips_through_the_contract() -> None:
    """The parquet must not contain a row the contract would reject."""
    frame = default_store().table("cases")
    if frame.empty:
        pytest.skip("no cases published yet")
    for row in frame.to_dict(orient="records"):
        record = dict(row)
        if record.get("closed_at") is not None and record["closed_at"] != record[
            "closed_at"
        ]:
            record["closed_at"] = None
        Investigation.model_validate(record)


def test_live_loop_outcome_produces_a_writable_case() -> None:
    """End to end on a real ring, in mock mode."""
    store: DataStore = default_store()
    ring_id = str(store.table("rings")["ring_id"].iloc[0])
    outcome = investigate_verified(ring_id)
    if not outcome.verified:
        pytest.skip("ring did not verify in mock mode")
    case = build_case(ring_id, outcome)
    assert case.ring_id == ring_id
    assert case.verified_claim_count == len(outcome.claims)
