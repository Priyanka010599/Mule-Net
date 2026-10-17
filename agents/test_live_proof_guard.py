"""Item 1: single-flight, launch limits, reaper, per-step status, run summary."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents import live_proof_guard as guard
from agents import live_proof_service as lps
from agents import service
from agents.limits import ServiceLimits
from agents.test_live_proof_service import (  # noqa: F401 - autouse fixture + fakes
    _FakeCapBucket,
    cap_bucket,
)
from agents.test_service import _config

CFG = guard.GuardConfig(cooldown_s=60, global_per_hour=3, ip_per_hour=2)


@pytest.fixture(autouse=True)
def _no_vm_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch GCP: no orphan sweep, and the run body is a stub."""
    monkeypatch.setattr(lps, "_reap_quietly", lambda *a, **k: None)

    async def stub(run: lps.LiveProofRun) -> None:
        run.finish(ok=True)

    monkeypatch.setattr(lps, "_execute", stub)


@pytest.fixture
def client() -> TestClient:
    limits = ServiceLimits(session_limit=1000, ip_limit=1000, window_s=60.0, daily_cap=10_000)
    return TestClient(service.create_app(_config(), limits))


# ---------------------------------------------------------------- single flight


def test_a_second_request_attaches_to_the_run_in_flight() -> None:
    first = asyncio.run(lps.start_live_proof("1.1.1.1"))
    second = asyncio.run(lps.start_live_proof("2.2.2.2"))
    assert first.attached is False
    assert second.attached is True and second.run_id == first.run_id
    assert lps.runs_today() == 1  # the attach launched nothing and spent nothing


def test_the_tampered_button_also_attaches_while_a_run_is_in_flight() -> None:
    first = asyncio.run(lps.start_live_proof("1.1.1.1"))
    tamper = asyncio.run(lps.start_tamper_test("1.1.1.1"))
    assert tamper.attached and tamper.run_id == first.run_id


def test_simultaneous_starts_launch_exactly_one_vm() -> None:
    outcomes: list[lps.StartOutcome] = []

    def go() -> None:
        outcomes.append(asyncio.run(lps.start_live_proof("9.9.9.9")))

    threads = [threading.Thread(target=go) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for o in outcomes if not o.attached) == 1
    assert len({o.run_id for o in outcomes}) == 1
    assert lps.runs_today() == 1


def test_releasing_the_lock_frees_it_and_only_the_owner_can_release(
    cap_bucket: _FakeCapBucket,  # noqa: F811
) -> None:
    first = asyncio.run(lps.start_live_proof("1.1.1.1"))
    guard.release_lock(cap_bucket, "someone-else")  # not the owner: a no-op, never raises
    assert guard.active_run_id(cap_bucket) == first.run_id
    guard.release_lock(cap_bucket, first.run_id)
    assert guard.active_run_id(cap_bucket) is None


def test_an_expired_lock_is_taken_over(cap_bucket: _FakeCapBucket) -> None:  # noqa: F811
    assert guard.acquire_lock(cap_bucket, "old", CFG, now=1000.0).acquired
    assert not guard.acquire_lock(cap_bucket, "new", CFG, now=1000.0 + CFG.lock_ttl_s - 1).acquired
    taken = guard.acquire_lock(cap_bucket, "new", CFG, now=1000.0 + CFG.lock_ttl_s + 1)
    assert taken.acquired
    assert guard.active_run_id(cap_bucket, now=1000.0 + CFG.lock_ttl_s + 2) == "new"


# ----------------------------------------------------------------------- limits


def test_cooldown_then_global_then_per_client_limits(
    cap_bucket: _FakeCapBucket,  # noqa: F811
) -> None:
    guard.record_launch(cap_bucket, "a", CFG, now=1000.0)
    with pytest.raises(guard.LaunchDenied) as cooldown:
        guard.precheck_limits(cap_bucket, "b", CFG, now=1030.0)
    assert cooldown.value.code == "cooldown" and cooldown.value.retry_after_s == 30

    guard.record_launch(cap_bucket, "a", CFG, now=1100.0)
    with pytest.raises(guard.LaunchDenied) as per_client:
        guard.record_launch(cap_bucket, "a", CFG, now=1200.0)
    assert per_client.value.code == "rate_limited_ip"
    assert per_client.value.retry_after_s == 3600 - 200  # until its oldest launch ages out

    guard.record_launch(cap_bucket, "b", CFG, now=1200.0)  # a different client still can
    with pytest.raises(guard.LaunchDenied) as everyone:
        guard.record_launch(cap_bucket, "c", CFG, now=1300.0)
    assert everyone.value.code == "rate_limited_global"
    guard.record_launch(cap_bucket, "c", CFG, now=1000.0 + 3601 + 60)  # an hour on, it empties


def test_clients_are_stored_hashed_never_as_addresses(
    cap_bucket: _FakeCapBucket,  # noqa: F811
) -> None:
    guard.record_launch(cap_bucket, "203.0.113.7", CFG, now=1000.0)
    stored = cap_bucket.objects[guard.STATE_PATH][0]
    assert "203.0.113.7" not in stored and guard.client_key("203.0.113.7") in stored


def test_a_denied_launch_releases_the_lock_and_spends_no_daily_slot(
    cap_bucket: _FakeCapBucket,  # noqa: F811
) -> None:
    guard.record_launch(cap_bucket, "1.1.1.1", guard.GuardConfig(), now=time.time())
    with pytest.raises(guard.LaunchDenied):
        asyncio.run(lps.start_live_proof("2.2.2.2"))  # inside the default 60s cooldown
    assert guard.active_run_id(cap_bucket) is None
    assert lps.runs_today() == 0


def test_the_http_start_is_a_structured_429_with_retry_after(
    client: TestClient,
    cap_bucket: _FakeCapBucket,  # noqa: F811
) -> None:
    first = client.post("/live-proof/start").json()
    guard.release_lock(cap_bucket, first["run_id"])  # the stub run ended; the cooldown remains
    denied = client.post("/live-proof/tampered-start")  # a tampered run counts the same
    assert denied.status_code == 429
    body = denied.json()
    assert body["error_type"] == "cooldown" and body["retry_after_s"] >= 1
    assert denied.headers["Retry-After"] == str(body["retry_after_s"])


def test_the_http_start_attaches_instead_of_launching(client: TestClient) -> None:
    first = client.post("/live-proof/start").json()
    again = client.post("/live-proof/start").json()
    assert first["attached"] is False
    assert again["attached"] is True and again["run_id"] == first["run_id"]


# --------------------------------------------------------------- steps and summary


def test_steps_carry_real_timestamps_and_status() -> None:
    run = lps.LiveProofRun(run_id="r1")
    run.record_stage("requesting_vm", {}, now=100.0)
    run.record_stage("vm_running", {"vm_name": "x"}, now=110.0)
    run.record_stage("wrong_key_attempt", {"bank_tried": "b"}, now=200.0)
    run.record_stage("wrong_key_attempt", {"failed_as_expected": True}, now=201.0)  # same step
    run.record_stage("correct_key_attempt", {}, now=205.0)
    run.finish(ok=True, now=210.0)
    pub = run.public()
    assert [s["key"] for s in pub["steps"]] == [
        "requesting_vm",
        "vm_running",
        "wrong_key_attempt",
        "correct_key_attempt",
    ]
    assert [s["status"] for s in pub["steps"]] == ["done"] * 4
    assert pub["steps"][0]["started_at"] != pub["steps"][0]["finished_at"]
    assert pub["steps"][2]["detail"] == {"failed_as_expected": True}
    assert (pub["stage"], pub["steps_total"]) == ("done", 8)


def test_a_failed_run_marks_the_step_that_was_running() -> None:
    run = lps.LiveProofRun(run_id="r1")
    run.record_stage("requesting_vm", {}, now=100.0)
    run.record_stage("awaiting_enclave", {}, now=105.0)
    run.error = "boom"
    run.finish(ok=False, now=120.0)
    assert [s["status"] for s in run.steps] == ["done", "failed"]
    assert run.stage == "failed"


def test_summary_reports_the_enclaves_own_numbers() -> None:
    run = lps.LiveProofRun(run_id="r1", started_at=1000.0)
    run.result = {
        "ok": True,
        "evidence": {
            "outcome": "completed",
            "key_release_ok": True,
            "image_digest": "sha256:abc",
            "edge_count": 349524,
            "account_count": 29973,
            "ring_count": 1,
        },
        "wrong_key_error": "InvalidTag",
    }
    run.finish(ok=True, now=1150.0)
    s = run.public()["summary"]
    assert s["got_key"] is True
    assert (s["joined_edges"], s["accounts"], s["cross_bank_rings_found"]) == (349524, 29973, 1)
    assert s["image_digest"] == "sha256:abc" and s["duration_seconds"] == 150.0
    assert s["scope_reason"] == "enclave_sees_cross_bank_edges_only"
    assert s["wrong_key_error"] == "InvalidTag"


def test_a_tampered_summary_carries_the_refusal_not_made_up_counts() -> None:
    run = lps.LiveProofRun(run_id="r1", tampered=True)
    run.result = {
        "ok": True,
        "tampered": True,
        "evidence": {"outcome": "key_release_refused"},
        "approved_digest": "sha256:good",
        "tampered_digest": "sha256:bad",
        "refusal_kind": "kms_decrypt_refused",
        "refusal_message": "403",
    }
    run.finish(ok=True)
    s = run.public()["summary"]
    assert (s["image_digest"], s["approved_digest"]) == ("sha256:bad", "sha256:good")
    assert s["refusal_kind"] == "kms_decrypt_refused" and s["joined_edges"] is None


def test_another_instance_can_serve_a_status_from_the_persisted_record(
    client: TestClient,
    cap_bucket: _FakeCapBucket,  # noqa: F811
) -> None:
    run = lps.LiveProofRun(run_id="abc123")
    run.record_stage("requesting_vm", {})
    guard.persist_run(cap_bucket, "abc123", run.public())
    body = client.get("/live-proof/status/abc123").json()
    assert body["run_id"] == "abc123" and body["steps"][0]["key"] == "requesting_vm"
    assert client.get("/live-proof/status/nope").status_code == 404


# ----------------------------------------------------------------- timeout, reaper


def test_a_run_past_the_overall_timeout_fails_and_deletes_its_vm(
    monkeypatch: pytest.MonkeyPatch,
    cap_bucket: _FakeCapBucket,  # noqa: F811
) -> None:
    deleted: list[str] = []
    monkeypatch.undo()  # drop the autouse stubs so the real _execute runs
    real_execute = lps._execute
    monkeypatch.setattr(lps, "_load_config_from_env", lambda: object())
    monkeypatch.setattr(lps, "_impersonated_credentials", lambda principal: object())
    monkeypatch.setattr(lps, "_delete_vm_quietly", deleted.append)
    monkeypatch.setattr(lps, "run_live_proof", lambda *a, **k: time.sleep(1.0))
    run = lps.LiveProofRun(
        run_id="r9", gcfg=guard.GuardConfig(overall_timeout_s=0.05), bucket=cap_bucket
    )
    guard.acquire_lock(cap_bucket, "r9", CFG)
    asyncio.run(real_execute(run))
    assert run.stage == "failed" and "timeout" in (run.error or "")
    assert deleted == ["r9"]
    assert guard.active_run_id(cap_bucket) is None  # the lock is released on every exit path


def _vm(name: str, age_s: float, run: str, now: float) -> Any:
    from datetime import UTC, datetime

    created = datetime.fromtimestamp(now - age_s, tz=UTC).isoformat()
    return SimpleNamespace(name=name, creation_timestamp=created, labels={"mulenet-run": run})


def test_the_reaper_deletes_only_old_vms_that_no_active_run_owns() -> None:
    now = 1_800_000_000.0
    vms = [
        _vm("old-orphan", 5000, "r1", now),
        _vm("young", 100, "r2", now),
        _vm("old-but-active", 5000, "live", now),
    ]
    asked: list[str] = []

    class _Instances:
        def list(self, request: Any) -> list[Any]:
            assert "live-proof" in request.filter
            return vms

        def delete(self, project: str, zone: str, instance: str) -> None:
            asked.append(instance)

    out = guard.reap_orphans(_Instances(), "p", "z", CFG, keep_run_id="live", now=now)
    assert out == asked == ["old-orphan"]


def test_a_vm_that_fails_to_delete_does_not_stop_the_sweep() -> None:
    now = 1_800_000_000.0
    vms = [_vm("stuck", 9000, "a", now), _vm("fine", 9000, "b", now)]

    class _Instances:
        def list(self, request: Any) -> list[Any]:
            return vms

        def delete(self, project: str, zone: str, instance: str) -> None:
            if instance == "stuck":
                raise RuntimeError("busy")

    assert guard.reap_orphans(_Instances(), "p", "z", CFG, now=now) == ["fine"]


def test_vms_are_labelled_for_the_reaper_and_a_platform_deadline_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from enclave.live_proof import LiveProofConfig, _build_instance

    cfg = LiveProofConfig(
        project="p",
        region="r",
        zone="z",
        image_digest="sha256:x",
        workload_sa_email="w@p",
        inbox_bucket="i",
        outbox_bucket="o",
        evidence_bucket="e",
        kms_edge_key="k",
        kms_hash_key="h",
        hash_key_uri="u",
        wip_audience="a",
        bank_ids=("bank-a",),
        outbox_public_keys={"bank-a": b"0" * 32},
        flagged_account_ids={},
        inbox_run_id="i",
        correct_bank_id="bank-a",
        correct_private_key=b"",
        wrong_bank_id="bank-b",
        wrong_private_key=b"",
    )
    monkeypatch.delenv("MULENET_LIVE_PROOF_MAX_RUN_SECONDS", raising=False)
    vm = _build_instance(cfg, "mulenet-live-proof-abc", "abc", "sha256:x")
    assert dict(vm.labels) == {"purpose": "live-proof", "mulenet-run": "abc"}
    assert vm.scheduling.max_run_duration.seconds == 0  # off unless asked for

    monkeypatch.setenv("MULENET_LIVE_PROOF_MAX_RUN_SECONDS", "900")
    vm = _build_instance(cfg, "mulenet-live-proof-abc", "abc", "sha256:x")
    assert vm.scheduling.max_run_duration.seconds == 900
    assert vm.scheduling.instance_termination_action == "DELETE"
