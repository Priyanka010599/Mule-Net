"""Cloud Logging trail for one live-proof run, for the judge-facing card.

The enclave evidence object proves a VM ran; this proves the *cloud* saw
it: the Compute audit entries for the VM create and delete, and the Cloud
KMS decrypt entry that allowed or refused the key. Each entry is read back
from Cloud Logging for this run's VM name and time window, so the card shows
what the platform recorded, not what this service claims.

REST (``entries:list``) rather than ``google-cloud-logging``: the agents
image does not carry that package, and one POST does not justify adding it.

Two log families, with different availability:

* Compute ``insert``/``delete`` are **Admin Activity** audit logs -- always
  on, nothing to enable.
* KMS ``Decrypt`` is a **Data Access** audit log, off by default. Until
  ``DATA_READ`` is enabled for ``cloudkms.googleapis.com`` the KMS row never
  appears, and the trail ends ``partial`` rather than pretending otherwise.

Principals, emails and project numbers are deliberately not returned: this
payload goes on a public screen.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from typing import Any

_LOGS_URL = "https://logging.googleapis.com/v2/entries:list"
_REQUEST_TIMEOUT_S = 15

#: Audit entries lag the event by seconds to ~1 minute. Keep reporting
#: ``pending`` this long after a run ends before settling on ``partial``.
PENDING_GRACE_S = 120.0

#: Start the window a little before the run so clock skew between this
#: container and the log timestamps cannot clip the first entry.
_WINDOW_SKEW_S = 30


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _key_id(kms_key: str) -> str | None:
    match = re.search(r"cryptoKeys/([^/]+)", kms_key)
    return match.group(1) if match else None


def _filters(project: str, vm_name: str, kms_key: str, since_ts: float) -> tuple[str, str]:
    since = _iso(since_ts)
    compute = (
        f'logName="projects/{project}/logs/cloudaudit.googleapis.com%2Factivity" '
        'AND protoPayload.methodName=~"compute\\.instances\\.(insert|delete)$" '
        f'AND protoPayload.resourceName:"instances/{vm_name}" '
        f'AND timestamp>="{since}"'
    )
    kms = (
        f'logName="projects/{project}/logs/cloudaudit.googleapis.com%2Fdata_access" '
        'AND protoPayload.serviceName="cloudkms.googleapis.com" '
        'AND protoPayload.methodName=~"Decrypt$" '
        f'AND timestamp>="{since}"'
    )
    key_id = _key_id(kms_key)
    if key_id:
        kms += f' AND protoPayload.resourceName:"cryptoKeys/{key_id}"'
    return compute, kms


def _list_entries(session: Any, project: str, filter_: str) -> list[dict[str, Any]]:
    response = session.post(
        _LOGS_URL,
        json={
            "resourceNames": [f"projects/{project}"],
            "filter": filter_,
            "orderBy": "timestamp asc",
            "pageSize": 50,
        },
        timeout=_REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json().get("entries", [])


def _caller_kind(subject: str) -> str:
    # An attested Confidential Space workload reaches KMS as a federated
    # workload-identity principal, not as a service account -- the
    # distinction the attestation proof rests on.
    if "workloadIdentityPools" in subject:
        return "attested workload identity"
    return "service account"


def _entry_row(entry: dict[str, Any]) -> dict[str, Any] | None:
    payload = entry.get("protoPayload", {})
    method = payload.get("methodName", "")
    when = entry.get("timestamp", "")
    insert_id = entry.get("insertId", "")
    if method.endswith("compute.instances.insert") or method.endswith("compute.instances.delete"):
        # Each operation logs a start and an end entry; the end one
        # (operation.last) is the "it actually happened" record.
        if entry.get("operation", {}).get("last") is not True:
            return None
        created = method.endswith("insert")
        return {
            "time": when,
            "event": "Confidential VM created" if created else "Confidential VM deleted",
            "detail": payload.get("resourceName", "").split("/")[-1],
            "outcome": "ok",
            "source": "Compute Engine audit log (Admin Activity)",
            "insert_id": insert_id,
        }
    if method.endswith("Decrypt"):
        status = payload.get("status", {})
        subject = payload.get("authenticationInfo", {}).get("principalSubject", "")
        allowed = not status.get("code", 0)
        detail = f"caller: {_caller_kind(subject)}"
        if not allowed:
            detail += f" - {status.get('message', 'denied')[:120]}"
        return {
            "time": when,
            "event": "KMS key released" if allowed else "KMS refused the key",
            "detail": detail,
            "outcome": "ok" if allowed else "refused",
            "source": "Cloud KMS audit log (Data Access)",
            "insert_id": insert_id,
        }
    return None


def build_trail(
    rows: list[dict[str, Any]], *, finished_at: float | None, now: float, since_ts: float
) -> dict[str, Any]:
    """Classify what was found. ``complete`` = VM created, VM deleted and a
    KMS entry all present; otherwise ``pending`` until the grace window after
    the run's end passes, then ``partial`` with what exists."""
    events = {row["event"] for row in rows}
    has_vm = {"Confidential VM created", "Confidential VM deleted"} <= events
    has_kms = bool(events & {"KMS key released", "KMS refused the key"})
    if has_vm and has_kms:
        status, note = "complete", ""
    elif finished_at is not None and now - finished_at > PENDING_GRACE_S:
        status = "partial"
        missing = []
        if not has_vm:
            missing.append("VM create/delete")
        if not has_kms:
            missing.append("KMS decrypt (Data Access audit logs may not be enabled)")
        note = "Not found in Cloud Logging: " + ", ".join(missing) + "."
    else:
        status, note = "pending", "Audit log entries usually land within a minute of the event."
    return {"status": status, "entries": rows, "note": note, "window_start": _iso(since_ts)}


def fetch_trail(
    vm_name: str, run_started_at: float, finished_at: float | None, now: float
) -> dict[str, Any]:
    """Query Cloud Logging for this run. Never raises: a failure is returned
    as ``no_access`` / ``error`` so the status poll that carries it stays up."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    since_ts = run_started_at - _WINDOW_SKEW_S
    project = os.environ.get("MULENET_ENCLAVE_PROJECT", "")
    kms_key = os.environ.get("MULENET_ENCLAVE_KMS_KEY", "")
    if not project:
        return {
            "status": "error",
            "entries": [],
            "note": "log trail is not configured",
            "window_start": "",
        }
    try:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        session = AuthorizedSession(credentials)
        compute_filter, kms_filter = _filters(project, vm_name, kms_key, since_ts)
        entries = _list_entries(session, project, compute_filter) + _list_entries(
            session, project, kms_filter
        )
    except Exception as exc:  # noqa: BLE001 - a log read must never fail the run's status
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        denied = status_code == 403
        return {
            "status": "no_access" if denied else "error",
            "entries": [],
            "note": "The agents service has no permission to read Cloud Logging."
            if denied
            else "Cloud Logging could not be read.",
            "window_start": _iso(since_ts),
        }
    rows = [row for row in (_entry_row(e) for e in entries) if row is not None]
    rows.sort(key=lambda r: r["time"])
    return build_trail(rows, finished_at=finished_at, now=now, since_ts=since_ts)
