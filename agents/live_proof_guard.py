"""Cost and safety guards around the paid live-proof VM.

The daily cap (``agents/live_proof_service.py``) says how many VMs a day. This
module is everything else that stops one click from becoming a bill:

* **Single flight.** One run at a time across every instance. A lock object in
  GCS (``cap/active-run.json``) is created with ``if_generation_match`` so only
  one writer wins; a second request attaches to the winner's ``run_id`` instead
  of launching a VM. The lock has a TTL, so a crashed instance cannot wedge the
  feature, and is released in a ``finally``.
* **Short-term limits.** A cooldown between launches, a global launches-per-hour
  cap and a per-client launches-per-hour cap, kept in one conditional-write
  object (``cap/live-proof-guard.json``). Clients are stored as a truncated
  SHA-256, never as an address.
* **A reaper.** Deletes labelled live-proof VMs older than ``reap_after_s``
  that no active run owns: the backstop for a crash, a hung delete, or a client
  that closed the tab while the process was throttled.
* **Run records.** Each run's public status is written to ``cap/runs/<id>.json``
  so any instance can answer a status poll for a run another instance started.

Everything lives under ``cap/`` because that is the only prefix
``infra/live_proof_iam.sh`` grants ``mulenet-agents`` object access to.

GCS access goes through a small ``bucket`` object (``get_blob``/``blob``), the
same shape ``agents.live_proof_service`` already uses, so tests can fake it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

_log = logging.getLogger("agents.live_proof_guard")

LOCK_PATH = "cap/active-run.json"
STATE_PATH = "cap/live-proof-guard.json"
RUNS_PREFIX = "cap/runs/"

#: Bounded retries on write contention, as in the daily cap.
_MAX_ATTEMPTS = 10

#: Label every live-proof VM carries (``enclave/live_proof._build_instance``);
#: the reaper's filter. Nothing else in the project has it.
VM_LABEL_FILTER = 'labels.purpose = "live-proof"'


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class GuardConfig:
    cooldown_s: int = 60
    global_per_hour: int = 6
    ip_per_hour: int = 3
    #: Longest a run may take end to end (VM create, evidence wait, delete).
    #: The run is a background task, so this is NOT bounded by Cloud Run's
    #: request timeout; the status polls are short requests.
    overall_timeout_s: int = 720
    #: Must outlive ``overall_timeout_s`` with room for the delete, or a
    #: second launch could start while the first VM still exists.
    lock_ttl_s: int = 1500
    reap_after_s: int = 1200

    @classmethod
    def from_env(cls) -> GuardConfig:
        return cls(
            cooldown_s=_env_int("MULENET_LIVE_PROOF_COOLDOWN_S", 60),
            global_per_hour=_env_int("MULENET_LIVE_PROOF_GLOBAL_PER_HOUR", 6),
            ip_per_hour=_env_int("MULENET_LIVE_PROOF_IP_PER_HOUR", 3),
            overall_timeout_s=_env_int("MULENET_LIVE_PROOF_OVERALL_TIMEOUT_S", 720),
            lock_ttl_s=_env_int("MULENET_LIVE_PROOF_LOCK_TTL_S", 1500),
            reap_after_s=_env_int("MULENET_LIVE_PROOF_REAP_AFTER_S", 1200),
        )


class LaunchDenied(Exception):
    """A short-term limit stopped a launch. ``code`` is machine-readable."""

    def __init__(self, code: str, message: str, retry_after_s: float) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retry_after_s = max(1, int(retry_after_s))


def client_key(ip: str) -> str:
    """What is stored in place of a client address."""
    return hashlib.sha256(f"mulenet-live-proof:{ip}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------- GCS helpers


def _read(bucket: Any, path: str) -> tuple[dict[str, Any] | None, int]:
    blob = bucket.get_blob(path)
    if blob is None:
        return None, 0
    return json.loads(blob.download_as_text()), blob.generation


def _write(bucket: Any, path: str, data: dict[str, Any], generation: int) -> None:
    bucket.blob(path).upload_from_string(
        json.dumps(data), content_type="application/json", if_generation_match=generation
    )


# ----------------------------------------------------------------- single flight


@dataclass(frozen=True)
class LockResult:
    acquired: bool
    #: The holder's run id: ours when acquired, the in-flight run's otherwise.
    run_id: str
    expires_at: float


def acquire_lock(
    bucket: Any, run_id: str, cfg: GuardConfig, now: float | None = None
) -> LockResult:
    """Take the single-flight lock, or report who holds it.

    Free, or held but past its TTL, means we may write; the write is a
    compare-and-swap on the generation we just read, so of two simultaneous
    callers exactly one succeeds and the other re-reads and sees the winner.
    """
    from google.api_core.exceptions import PreconditionFailed

    now = time.time() if now is None else now
    for _ in range(_MAX_ATTEMPTS):
        held, generation = _read(bucket, LOCK_PATH)
        if held is not None and held["expires_at"] > now:
            return LockResult(False, held["run_id"], held["expires_at"])
        mine = {"run_id": run_id, "started_at": now, "expires_at": now + cfg.lock_ttl_s}
        try:
            _write(bucket, LOCK_PATH, mine, generation)
        except PreconditionFailed:
            continue
        return LockResult(True, run_id, mine["expires_at"])
    raise RuntimeError("could not take or read the live-proof lock; live proof is unavailable")


def active_run_id(bucket: Any, now: float | None = None) -> str | None:
    """The run id holding an unexpired lock, if any."""
    now = time.time() if now is None else now
    held, _ = _read(bucket, LOCK_PATH)
    return held["run_id"] if held is not None and held["expires_at"] > now else None


def release_lock(bucket: Any, run_id: str) -> None:
    """Release the lock iff this run still owns it. Never raises."""
    from google.api_core.exceptions import NotFound, PreconditionFailed

    try:
        held, generation = _read(bucket, LOCK_PATH)
        if held is not None and held["run_id"] == run_id:
            bucket.blob(LOCK_PATH).delete(if_generation_match=generation)
    except (PreconditionFailed, NotFound):
        pass  # already gone, or taken over after our TTL: nothing of ours to release
    except Exception:  # noqa: BLE001 - the TTL is the backstop; releasing must not mask a run's result
        _log.warning("could not release the live-proof lock for run %s", run_id, exc_info=True)


# ------------------------------------------------------------- short-term limits


def check_limits(
    state: dict[str, Any] | None, ip_key: str, cfg: GuardConfig, now: float
) -> LaunchDenied | None:
    """The first limit this launch would break, or ``None``. Pure."""
    launches = [x for x in (state or {}).get("launches", []) if x["t"] > now - 3600]
    if launches and cfg.cooldown_s > 0:
        wait = max(x["t"] for x in launches) + cfg.cooldown_s - now
        if wait > 0:
            return LaunchDenied(
                "cooldown",
                "A live proof just ran. Please wait a moment before starting another.",
                wait + 0.999,
            )
    if len(launches) >= cfg.global_per_hour:
        wait = min(x["t"] for x in launches) + 3600 - now
        return LaunchDenied(
            "rate_limited_global",
            "Live proof is at its hourly limit for everyone. Try again later.",
            wait + 0.999,
        )
    mine = [x for x in launches if x["ip"] == ip_key]
    if len(mine) >= cfg.ip_per_hour:
        wait = min(x["t"] for x in mine) + 3600 - now
        return LaunchDenied(
            "rate_limited_ip",
            "You have reached this address's hourly live-proof limit. Try again later.",
            wait + 0.999,
        )
    return None


def precheck_limits(bucket: Any, ip: str, cfg: GuardConfig, now: float | None = None) -> None:
    """Raise :class:`LaunchDenied` if a launch from ``ip`` would break a limit.
    Read-only: nothing is recorded until :func:`record_launch`."""
    now = time.time() if now is None else now
    state, _ = _read(bucket, STATE_PATH)
    denied = check_limits(state, client_key(ip), cfg, now)
    if denied is not None:
        raise denied


def record_launch(bucket: Any, ip: str, cfg: GuardConfig, now: float | None = None) -> None:
    """Record one launch, re-checking the limits in the same conditional write
    so two racing launches cannot both take the last slot."""
    from google.api_core.exceptions import PreconditionFailed

    now = time.time() if now is None else now
    key = client_key(ip)
    for _ in range(_MAX_ATTEMPTS):
        state, generation = _read(bucket, STATE_PATH)
        denied = check_limits(state, key, cfg, now)
        if denied is not None:
            raise denied
        launches = [x for x in (state or {}).get("launches", []) if x["t"] > now - 3600]
        launches.append({"t": now, "ip": key})
        try:
            _write(bucket, STATE_PATH, {"launches": launches}, generation)
            return
        except PreconditionFailed:
            continue
    raise RuntimeError("could not record the live-proof launch; live proof is unavailable")


# -------------------------------------------------------------------- run records


def persist_run(bucket: Any, run_id: str, public: dict[str, Any]) -> None:
    """Best-effort copy of a run's public status, so another instance can serve
    its status. Never raises: it is a convenience, the run does not depend on it."""
    try:
        bucket.blob(f"{RUNS_PREFIX}{run_id}.json").upload_from_string(
            json.dumps(public), content_type="application/json"
        )
    except Exception:  # noqa: BLE001
        _log.warning("could not persist live-proof run %s", run_id, exc_info=True)


def load_run(bucket: Any, run_id: str) -> dict[str, Any] | None:
    try:
        data, _ = _read(bucket, f"{RUNS_PREFIX}{run_id}.json")
        return data
    except Exception:  # noqa: BLE001
        _log.warning("could not read live-proof run %s", run_id, exc_info=True)
        return None


# --------------------------------------------------------------------- the reaper


def _age_s(creation_timestamp: str, now: float) -> float:
    return now - datetime.fromisoformat(creation_timestamp).timestamp()


def reap_orphans(
    instances: Any,
    project: str,
    zone: str,
    cfg: GuardConfig,
    keep_run_id: str | None = None,
    now: float | None = None,
) -> list[str]:
    """Delete labelled live-proof VMs older than ``cfg.reap_after_s``.

    ``instances`` is a ``compute_v1.InstancesClient`` (impersonating the
    live-proof service account). A VM belonging to ``keep_run_id`` (the
    in-flight run, matched by its ``mulenet-run`` label) is left alone however
    old, and a delete that fails is logged and skipped so one stuck VM cannot
    stop the sweep. Returns the names it asked to delete.
    """
    from google.cloud import compute_v1

    now = time.time() if now is None else now
    deleted: list[str] = []
    request = compute_v1.ListInstancesRequest(project=project, zone=zone, filter=VM_LABEL_FILTER)
    for vm in instances.list(request=request):
        if keep_run_id and dict(vm.labels).get("mulenet-run") == keep_run_id:
            continue
        try:
            if _age_s(vm.creation_timestamp, now) < cfg.reap_after_s:
                continue
            _log.warning("reaping orphaned live-proof VM %s", vm.name)
            instances.delete(project=project, zone=zone, instance=vm.name)
            deleted.append(vm.name)
        except Exception:  # noqa: BLE001
            _log.warning("could not reap live-proof VM %s", getattr(vm, "name", "?"), exc_info=True)
    return deleted
