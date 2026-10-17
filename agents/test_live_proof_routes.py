"""Route-level tests for POST /live-proof/start and GET /live-proof/status/{run_id}.

Complements agents/test_live_proof_service.py's direct, route-free proof
that the cap blocks server-side. These tests go through the real FastAPI
app (agents.service.create_app) to check the HTTP contract: a cap hit is
a 200 with capped: true (never a dead button or a raw error), and an
unknown run_id is a 404 in the service's own error envelope.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from google.cloud import storage

from agents import live_proof_service as lps
from agents import service
from agents.limits import ServiceLimits
from agents.test_live_proof_service import _FakeCapBucket, _FakeCapStorageClient
from agents.test_service import _config  # the vertex-mode config builder


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same fake GCS-backed cap as agents/test_live_proof_service.py -- the
    # cap is durable now, not a module-level list, so there is nothing
    # left to reset by assignment; a fresh fake backend per test is what
    # gives each test a clean cap count instead.
    fake_bucket = _FakeCapBucket()
    monkeypatch.setattr(storage, "Client", lambda **kwargs: _FakeCapStorageClient(fake_bucket))
    monkeypatch.setenv("MULENET_LIVE_PROOF_CAP_BUCKET", "fake-cap-bucket")
    monkeypatch.setattr(lps, "_runs", {})


@pytest.fixture
def client() -> TestClient:
    limits = ServiceLimits(session_limit=1000, ip_limit=1000, window_s=60.0, daily_cap=10_000)
    return TestClient(service.create_app(_config(), limits))


def test_start_returns_capped_true_once_the_limit_is_hit(client: TestClient) -> None:
    for _ in range(lps.MAX_LIVE_RUNS_PER_DAY):
        lps.check_and_reserve_cap_slot()

    resp = client.post("/live-proof/start")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "capped": True,
        "run_id": None,
        "runs_today": lps.MAX_LIVE_RUNS_PER_DAY,
        "limit": lps.MAX_LIVE_RUNS_PER_DAY,
        "attached": False,
        "retry_after_seconds": body["retry_after_seconds"],
    }
    assert body["retry_after_seconds"] > 0


def test_start_returns_a_run_id_when_under_the_cap(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_execute(run: lps.LiveProofRun) -> None:
        run.stage = "failed"
        run.error = "no GCP credentials in this test"

    monkeypatch.setattr(lps, "_execute", _fake_execute)

    resp = client.post("/live-proof/start")
    assert resp.status_code == 200
    body = resp.json()
    assert body["capped"] is False
    assert isinstance(body["run_id"], str) and body["run_id"]


def test_tampered_start_returns_a_run_id_and_marks_the_run_tampered(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fake_execute(run: lps.LiveProofRun) -> None:
        run.stage = "failed"
        run.error = "no GCP credentials in this test"

    monkeypatch.setattr(lps, "_execute", _fake_execute)

    resp = client.post("/live-proof/tampered-start")
    assert resp.status_code == 200
    body = resp.json()
    assert body["capped"] is False
    assert isinstance(body["run_id"], str) and body["run_id"]
    assert lps._runs[body["run_id"]].tampered is True


def test_tampered_start_shares_the_cap_with_live_proof(client: TestClient) -> None:
    """Hitting the cap via normal /live-proof/start calls must also
    block /live-proof/tampered-start -- they are the same counter."""
    for _ in range(lps.MAX_LIVE_RUNS_PER_DAY):
        lps.check_and_reserve_cap_slot()

    resp = client.post("/live-proof/tampered-start")
    assert resp.status_code == 200
    body = resp.json()
    assert body["capped"] is True
    assert body["runs_today"] == lps.MAX_LIVE_RUNS_PER_DAY


def test_status_404s_for_an_unknown_run(client: TestClient) -> None:
    resp = client.get("/live-proof/status/no-such-run")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error_type"] == "not_found"


def test_status_reports_the_registered_runs_shape(client: TestClient) -> None:
    run = lps.LiveProofRun(run_id="r1", stage="vm_running", extra={"vm_name": "x"})
    lps._runs["r1"] = run

    resp = client.get("/live-proof/status/r1")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "run_id": "r1",
        "stage": "vm_running",
        "extra": {"vm_name": "x"},
        "result": None,
        "error": None,
        "log_trail": None,  # no Cloud Logging read until the run has finished
        "tampered": False,
        "started_at": body["started_at"],
        "finished_at": None,
        "duration_seconds": None,
        "plan": body["plan"],
        "steps": [],
        "steps_done": 0,
        "steps_total": len(body["plan"]),
        "summary": None,
    }
