"""Tests for ``agents.live_proof_logs``: turning audit-log entries into the
judge-facing trail, and how that trail settles (pending -> complete/partial)."""

from __future__ import annotations

from typing import Any

from agents import live_proof_logs as lpl

VM = "mulenet-live-proof-abc123def456"
KEY = "projects/p/locations/asia-south1/keyRings/r/cryptoKeys/edge-key/cryptoKeyVersions/1"


def _compute(method: str, *, last: bool = True) -> dict[str, Any]:
    return {
        "timestamp": "2026-10-07T05:01:12.345Z",
        "insertId": f"id-{method}-{last}",
        "operation": {"last": last} if last else {"first": True},
        "protoPayload": {
            "methodName": f"v1.compute.instances.{method}",
            "resourceName": f"projects/p/zones/z/instances/{VM}",
        },
    }


def _kms(code: int = 0, subject: str = "") -> dict[str, Any]:
    return {
        "timestamp": "2026-10-07T05:02:30Z",
        "insertId": "kms-1",
        "protoPayload": {
            "methodName": "Decrypt",
            "status": {"code": code, "message": "denied by policy"} if code else {},
            "authenticationInfo": {"principalSubject": subject},
        },
    }


def test_filters_pin_the_vm_the_key_and_the_window() -> None:
    compute, kms = lpl._filters("p", VM, KEY, 1_791_350_000)
    assert f"instances/{VM}" in compute and "%2Factivity" in compute
    assert "cloudkms.googleapis.com" in kms and "%2Fdata_access" in kms
    assert 'cryptoKeys/edge-key"' in kms
    assert "2026-10-0" in compute  # window start is rendered as an ISO UTC timestamp


def test_compute_rows_keep_only_the_completed_entry_of_each_operation() -> None:
    started = lpl._entry_row(_compute("insert", last=False))
    done = lpl._entry_row(_compute("insert"))
    assert started is None
    assert done is not None
    assert done["event"] == "Confidential VM created"
    assert done["detail"] == VM
    deleted = lpl._entry_row(_compute("delete"))
    assert deleted is not None and deleted["event"] == "Confidential VM deleted"


def test_kms_allowed_vs_refused_and_caller_kind() -> None:
    allowed = lpl._entry_row(
        _kms(subject="principal://iam.googleapis.com/projects/1/workloadIdentityPools/w/subject/s")
    )
    refused = lpl._entry_row(_kms(code=7, subject="serviceAccount:x"))
    assert allowed is not None and refused is not None
    assert allowed["outcome"] == "ok" and "attested workload identity" in allowed["detail"]
    assert refused["outcome"] == "refused" and "denied by policy" in refused["detail"]


def test_rows_never_carry_principals_or_project_numbers() -> None:
    row = lpl._entry_row(_kms(subject="user:someone@example.com"))
    assert "someone@example.com" not in str(row)


def _rows(*kinds: str) -> list[dict[str, Any]]:
    events = {
        "create": "Confidential VM created",
        "delete": "Confidential VM deleted",
        "kms": "KMS key released",
    }
    return [{"event": events[k], "time": "t"} for k in kinds]


def test_trail_is_complete_only_with_vm_pair_and_a_kms_entry() -> None:
    rows = _rows("create", "delete", "kms")
    trail = lpl.build_trail(rows, finished_at=100.0, now=101.0, since_ts=0)
    assert trail["status"] == "complete"


def test_trail_is_pending_inside_the_grace_window_then_partial() -> None:
    rows = _rows("create", "delete")  # no KMS entry, e.g. Data Access logs off
    inside = lpl.build_trail(rows, finished_at=100.0, now=150.0, since_ts=0)
    late = 100.0 + lpl.PENDING_GRACE_S + 1
    after = lpl.build_trail(rows, finished_at=100.0, now=late, since_ts=0)
    assert inside["status"] == "pending"
    assert after["status"] == "partial" and "KMS decrypt" in after["note"]


def test_service_skips_logging_while_running_and_caches_a_settled_trail(monkeypatch: Any) -> None:
    import asyncio

    from agents import live_proof_service as lps

    calls: list[str] = []

    def fake_fetch(vm_name: str, *_a: Any) -> dict[str, Any]:
        calls.append(vm_name)
        return {"status": "complete", "entries": [], "note": "", "window_start": ""}

    monkeypatch.setattr(lps, "fetch_trail", fake_fetch)
    run = lps.LiveProofRun(run_id="abc123def456")

    assert asyncio.run(lps.log_trail_for(run)) is None  # still running: no log read
    assert calls == []

    run.finished_at = 1.0
    first = asyncio.run(lps.log_trail_for(run))
    second = asyncio.run(lps.log_trail_for(run))
    assert first is second and first is not None and first["status"] == "complete"
    assert calls == [VM]  # settled trail is read once, then served from the run


def test_a_logging_permission_error_is_reported_not_raised(monkeypatch: Any) -> None:
    class _Resp:
        status_code = 403

    class _Denied(Exception):
        response = _Resp()

    def boom(*_a: Any, **_k: Any) -> None:
        raise _Denied()

    import google.auth
    import google.auth.transport.requests as transport

    monkeypatch.setenv("MULENET_ENCLAVE_PROJECT", "p")
    monkeypatch.setattr(google.auth, "default", lambda **_k: (object(), "p"))
    monkeypatch.setattr(transport, "AuthorizedSession", lambda _c: object())
    monkeypatch.setattr(lpl, "_list_entries", boom)

    trail = lpl.fetch_trail(VM, 1_791_350_000, 1_791_350_200, 1_791_350_300)
    assert trail["status"] == "no_access"
    assert trail["entries"] == []
