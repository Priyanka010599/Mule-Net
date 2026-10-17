"""Server-side orchestration for the judge-facing live enclave proof.

Two responsibilities kept in one small module rather than split:

1. **The cap.** A rolling 24-hour count of live runs, checked and
   incremented *before* any VM creation is attempted -- never just a UI
   affordance. Stored in a small JSON object in GCS
   (``MULENET_LIVE_PROOF_CAP_BUCKET``), not in process memory: an
   in-memory counter resets on every redeploy AND on every cold start
   (``mulenet-agents`` runs at ``min-instances=0``) -- observed live,
   this exact reset happened mid-session before this was fixed -- so it
   never actually bounded "10 per rolling 24h", only "10 per instance
   lifetime". The read -> prune -> check -> write is a conditional write
   (``if_generation_match``), the same idiom ``enclave/inbox.py``'s
   ``write_outbox`` already uses for its own idempotent-create guard, so
   concurrent reservations from multiple instances can't both claim the
   same slot. ``--max-instances=1`` stays set as defence in depth, but
   this code no longer depends on it to be correct.
2. **The run registry.** Each live run gets a ``run_id`` and a
   background asyncio task; its current stage is tracked in memory so
   ``GET /live-proof/status/{run_id}`` can be polled by the UI without
   holding one HTTP request open for the full ~2-4 minute cycle. (This
   part IS still in-memory and still depends on one instance -- a run in
   flight when the instance recycles loses its status, though the VM
   lifecycle it started completes/cleans up regardless.)

Credentials: this process runs as ``mulenet-agents``, which holds only
``roles/iam.serviceAccountTokenCreator`` on ``mulenet-live-proof`` -- the
actual VM create/delete permission lives entirely on the impersonated
identity, minted fresh per run via ``google.auth.impersonated_credentials``,
the same pattern ``infra/enclave_send_edges.py`` already uses for bank
identities. ``mulenet-agents`` itself is never granted any compute
permission directly. The cap file is read/written under
``mulenet-agents``'s own identity instead (no impersonation): it is not
compute or decrypt material, so it does not belong behind the
narrowly-scoped live-proof SA.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agents import live_proof_guard as guard
from agents.live_proof_logs import fetch_trail
from enclave.live_proof import LiveProofConfig, run_live_proof

_log = logging.getLogger("agents.live_proof_service")

#: Hard cap: live runs per rolling 24h, enforced before VM creation.
#: ~$0.0767/hour for a c2d-standard-2 Confidential VM in asia-south1
#: (Google's published on-demand + Confidential VM surcharge rates,
#: verified this session, not guessed) and a real run is ~2-4 minutes, so
#: even this cap's full use costs under $0.10/day -- the cap exists to
#: bound VM churn and keep this a deliberate feature, not a scriptable
#: one, not because the dollar cost itself is a real risk.
MAX_LIVE_RUNS_PER_DAY = 10
_DAY_S = 86_400.0

#: One small object, not a bucket per entry: ten timestamps a day is
#: nothing worth spreading across objects, and one object is what makes
#: a single conditional write enough to serialise every reservation.
_CAP_OBJECT_PATH = "cap/live-proof-runs.json"
#: Bounded retries on write contention -- at this call volume (<=10/day)
#: a real collision is rare, and failing loudly after this many beats an
#: unbounded retry loop silently stalling a request.
_CAP_MAX_ATTEMPTS = 10

_runs: dict[str, "LiveProofRun"] = {}


@dataclass
class LiveProofRun:
    run_id: str
    stage: str = "queued"
    extra: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    #: True for a "try tampered code" run. The UI already knows which
    #: button it pressed, but carrying this on the run itself means
    #: GET /live-proof/status/{run_id} is self-describing too.
    tampered: bool = False
    #: Set when the run reaches done/failed; the log trail's "still
    #: waiting for audit entries" grace window counts from here.
    finished_at: float | None = None
    #: Last Cloud Logging read for this run, and when it was made, so a
    #: 3-second status poll does not turn into a 3-second logging query.
    log_trail: dict[str, Any] | None = None
    log_trail_at: float = 0.0
    #: One record per stage the run actually reached, with real timestamps
    #: (``record_stage``). Never filled in ahead of time: a step that has not
    #: happened is in :meth:`plan` only.
    steps: list[dict[str, Any]] = field(default_factory=list)
    #: The background task driving this run (see ``_start``), and the guard
    #: settings and bucket it was started with.
    task: Any = field(default=None, repr=False)
    gcfg: Any = field(default=None, repr=False)
    bucket: Any = field(default=None, repr=False)

    def plan(self) -> list[dict[str, str]]:
        """The steps this kind of run will go through, in order."""
        plan = TAMPER_PLAN if self.tampered else LIVE_PLAN
        return [{"key": k, "label": label} for k, label in plan]

    def record_stage(self, stage: str, extra: dict[str, Any], now: float | None = None) -> None:
        """Note that the run reached ``stage``: close the step that was
        running, open this one. A repeated stage (the wrong-key attempt is
        reported twice, start then result) updates the open step in place."""
        now = time.time() if now is None else now
        self.stage, self.extra = stage, extra
        if stage == "failed":  # run_live_proof's cleanup-failure report, not a step
            return
        if self.steps and self.steps[-1]["key"] == stage:
            self.steps[-1]["detail"] = extra
        else:
            self._close_open_step("done", now)
            self.steps.append(
                {
                    "key": stage,
                    "label": STEP_LABELS.get(stage, stage),
                    "status": "running",
                    "started_at": _iso(now),
                    "finished_at": None,
                    "detail": extra,
                }
            )

    def _close_open_step(self, status: str, now: float) -> None:
        if self.steps and self.steps[-1]["status"] == "running":
            self.steps[-1]["status"] = status
            self.steps[-1]["finished_at"] = _iso(now)

    def finish(self, *, ok: bool, now: float | None = None) -> None:
        """End the run. On success the last open step is done; on failure the
        step that was running is the one that failed."""
        now = time.time() if now is None else now
        self.finished_at = now
        self._close_open_step("done" if ok else "failed", now)
        self.stage = "done" if ok else "failed"

    def public(self) -> dict[str, Any]:
        """What the status endpoint (and the persisted copy) carries."""
        finished = self.finished_at
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "extra": self.extra,
            "result": self.result,
            "error": self.error,
            "tampered": self.tampered,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(finished) if finished is not None else None,
            "duration_seconds": (
                round(finished - self.started_at, 1) if finished is not None else None
            ),
            "plan": self.plan(),
            "steps": self.steps,
            "steps_done": sum(1 for s in self.steps if s["status"] == "done"),
            "steps_total": len(self.plan()),
            "summary": run_summary(self) if self.result is not None else None,
        }


#: Steps a run goes through, as (stage key, label). These are the stages
#: ``enclave.live_proof.run_live_proof`` really reports; nothing is invented
#: between them (see that module's note on what is observable).
LIVE_PLAN: tuple[tuple[str, str], ...] = (
    ("requesting_vm", "Requesting a Confidential VM"),
    ("vm_running", "VM running, attesting"),
    ("awaiting_enclave", "Enclave working (attestation, KMS key release, decrypt, re-encrypt)"),
    ("evidence_received", "Evidence object received"),
    ("wrong_key_attempt", "Decrypt attempt with the WRONG bank's key"),
    ("correct_key_attempt", "Decrypt attempt with the CORRECT bank's key"),
    ("cleaning_up", "Deleting the VM"),
    ("done", "Done"),
)
TAMPER_PLAN: tuple[tuple[str, str], ...] = (
    ("requesting_vm", "Requesting a Confidential VM running the tampered image"),
    ("vm_running", "VM running, attesting"),
    ("awaiting_enclave", "Enclave asking KMS for the key"),
    ("evidence_received", "Evidence object received"),
    ("kms_refused", "KMS refused the key"),
    ("cleaning_up", "Deleting the VM"),
    ("done", "Done"),
)
STEP_LABELS: dict[str, str] = {**dict(LIVE_PLAN), **dict(TAMPER_PLAN), "failed": "Failed"}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_summary(run: LiveProofRun) -> dict[str, Any]:
    """The headline facts of a finished run, from the evidence the enclave itself
    wrote (not from anything this process claims).

    ``joined_edges``/``accounts`` are what the enclave saw, which is the
    cross-bank edges only -- see ``scope_reason`` and
    ``ui.build_snapshot.dataset_scope_counts`` for why they are smaller than
    the dataset's totals. Fields the evidence does not have (a refused run
    never counted any edges) are ``None``, not zero.
    """
    result = run.result or {}
    evidence = result.get("evidence") or {}
    tampered = bool(result.get("tampered"))
    finished = run.finished_at
    return {
        "run_id": run.run_id,
        "started_at": _iso(run.started_at),
        "finished_at": _iso(finished) if finished is not None else None,
        "duration_seconds": round(finished - run.started_at, 1) if finished is not None else None,
        "tampered": tampered,
        "image_digest": result.get("tampered_digest") if tampered else evidence.get("image_digest"),
        "approved_digest": result.get("approved_digest") if tampered else evidence.get("image_digest"),
        "outcome": evidence.get("outcome"),
        "got_key": evidence.get("key_release_ok"),
        "joined_edges": evidence.get("edge_count"),
        "accounts": evidence.get("account_count"),
        "cross_bank_rings_found": evidence.get("ring_count"),
        "edges_in_enclave_scope": evidence.get("edge_count"),
        "accounts_in_enclave_scope": evidence.get("account_count"),
        "scope_reason": "enclave_sees_cross_bank_edges_only",
        "wrong_key_error": result.get("wrong_key_error"),
        "refusal_kind": result.get("refusal_kind"),
        "refusal_message": result.get("refusal_message"),
        "tampered_line_changed": result.get("tampered_line_changed"),
    }


#: Minimum gap between Cloud Logging reads for one run.
_LOG_TRAIL_REFRESH_S = 8.0

#: Statuses that will not change on another read.
_LOG_TRAIL_FINAL = {"complete", "partial", "no_access"}


class CapExceeded(Exception):
    """Raised when the rolling-24h cap is hit. Never a bare 500 -- the
    caller falls back to the static evidence card with a plain count."""

    def __init__(self, count: int, limit: int, retry_after_s: int | None = None) -> None:
        super().__init__(f"{count}/{limit} live runs already used today")
        self.count = count
        self.limit = limit
        #: Seconds until the oldest counted run leaves the rolling window.
        self.retry_after_s = retry_after_s


def _cap_bucket_name() -> str:
    value = os.environ.get("MULENET_LIVE_PROOF_CAP_BUCKET")
    if not value:
        raise RuntimeError("MULENET_LIVE_PROOF_CAP_BUCKET is not set; live proof is unavailable")
    return value


def _load_cap_state(bucket: Any) -> tuple[list[float], int]:
    """Returns ``(timestamps, generation)``. ``generation=0`` means the
    object does not exist yet -- the same sentinel GCS's own
    ``if_generation_match=0`` uses for "create only if absent", so the
    first-ever write and every later conditional overwrite share one
    code path below.

    Two calls (metadata, then content) rather than one: a blob's
    ``.generation``/``.size`` are not reliably populated as a side
    effect of ``download_as_text()`` alone on this client -- observed
    live elsewhere this session (``ciphertext_bytes_total`` came back
    ``None`` after exactly that assumption) -- so metadata is fetched
    explicitly via ``get_blob`` instead of assumed.
    """
    blob = bucket.get_blob(_CAP_OBJECT_PATH)
    if blob is None:
        return [], 0
    data = json.loads(blob.download_as_text())
    return [float(t) for t in data.get("timestamps", [])], blob.generation


def check_and_reserve_cap_slot() -> None:
    """Raise :class:`CapExceeded` if the cap is already hit; otherwise
    reserve a slot immediately, atomically with the check (no VM is
    created between the check and the reservation -- this function does
    both in the same call, which is what makes it a server-side
    enforcement rather than a race-prone UI-side one).

    Durable and shared: the read -> prune -> check -> write round trip
    is retried on ``PreconditionFailed`` (another instance, or another
    request on this one, won the same conditional write) rather than
    silently double-granting the last slot. Blocking network I/O --
    callers on the event loop must run this via a thread, same as
    ``start_live_proof`` does.
    """
    from google.api_core.exceptions import PreconditionFailed
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(_cap_bucket_name())
    now = time.time()
    cutoff = now - _DAY_S

    for _attempt in range(_CAP_MAX_ATTEMPTS):
        timestamps, generation = _load_cap_state(bucket)
        timestamps = [t for t in timestamps if t >= cutoff]
        if len(timestamps) >= MAX_LIVE_RUNS_PER_DAY:
            raise CapExceeded(
                len(timestamps),
                MAX_LIVE_RUNS_PER_DAY,
                retry_after_s=int(min(timestamps) + _DAY_S - now) + 1,
            )
        timestamps.append(now)
        try:
            bucket.blob(_CAP_OBJECT_PATH).upload_from_string(
                json.dumps({"timestamps": timestamps}),
                content_type="application/json",
                if_generation_match=generation,
            )
            return
        except PreconditionFailed:
            continue  # someone else's write won the race; re-read and retry
    raise RuntimeError(
        f"could not reserve a live-proof cap slot after {_CAP_MAX_ATTEMPTS} "
        "concurrent attempts; live proof is unavailable"
    )


def runs_today() -> int:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(_cap_bucket_name())
    timestamps, _generation = _load_cap_state(bucket)
    cutoff = time.time() - _DAY_S
    return len([t for t in timestamps if t >= cutoff])


def _impersonated_credentials(target_principal: str) -> Any:
    import google.auth
    from google.auth import impersonated_credentials

    source_credentials, _ = google.auth.default()
    return impersonated_credentials.Credentials(
        source_credentials=source_credentials,
        target_principal=target_principal,
        target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def _load_config_from_env() -> LiveProofConfig:
    import os

    def req(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"{name} is not set; live proof is unavailable")
        return value

    return LiveProofConfig(
        project=req("MULENET_ENCLAVE_PROJECT"),
        region=req("MULENET_ENCLAVE_REGION"),
        zone=req("MULENET_ENCLAVE_ZONE"),
        image_digest=req("MULENET_ENCLAVE_IMAGE_DIGEST"),
        workload_sa_email=req("MULENET_ENCLAVE_WORKLOAD_SA"),
        inbox_bucket=req("MULENET_ENCLAVE_INBOX_BUCKET"),
        outbox_bucket=req("MULENET_ENCLAVE_OUTBOX_BUCKET"),
        evidence_bucket=req("MULENET_ENCLAVE_EVIDENCE_BUCKET"),
        kms_edge_key=req("MULENET_ENCLAVE_KMS_KEY"),
        kms_hash_key=req("MULENET_ENCLAVE_HASH_KMS_KEY"),
        hash_key_uri=req("MULENET_ENCLAVE_HASH_KEY_URI"),
        wip_audience=req("MULENET_ENCLAVE_WIP_AUDIENCE"),
        bank_ids=(bank_ids := tuple(req("MULENET_ENCLAVE_BANK_IDS").split(","))),
        outbox_public_keys={
            bank_id: base64.b64decode(
                req(f"MULENET_ENCLAVE_OUTBOX_PUBKEY_{bank_id.upper().replace('-', '_')}")
            )
            for bank_id in bank_ids
        },
        # Optional per bank (a bank can legitimately have flagged nothing)
        # -- unlike the pubkeys above, this is not req()'d. See
        # enclave/live_proof.py:LiveProofConfig.flagged_account_ids for
        # why it exists at all: without it ring-finding has nothing to
        # cluster and the run completes with zero rings, not a bug.
        flagged_account_ids={
            bank_id: tuple(
                h
                for h in os.environ.get(
                    f"MULENET_ENCLAVE_FLAGGED_{bank_id.upper().replace('-', '_')}", ""
                ).split(",")
                if h
            )
            for bank_id in bank_ids
        },
        inbox_run_id=req("MULENET_ENCLAVE_LIVE_PROOF_INBOX_RUN_ID"),
        correct_bank_id=req("MULENET_LIVE_PROOF_CORRECT_BANK"),
        correct_private_key=base64.b64decode(req("MULENET_LIVE_PROOF_CORRECT_KEY_B64")),
        wrong_bank_id=req("MULENET_LIVE_PROOF_WRONG_BANK"),
        wrong_private_key=base64.b64decode(req("MULENET_LIVE_PROOF_WRONG_KEY_B64")),
    )


@dataclass(frozen=True)
class StartOutcome:
    """What a start request resolved to."""

    run_id: str
    #: True when another run was already in flight and this request simply
    #: joined it: no VM was launched and nothing was counted against any limit.
    attached: bool


async def start_live_proof(client_ip: str = "unknown") -> StartOutcome:
    """Start a run, or attach to the one already in flight.

    The order is the safety property: take the single-flight lock, check the
    short-term limits, reserve a slot in the daily cap, record the launch --
    all before any VM work. :class:`CapExceeded` and
    :class:`agents.live_proof_guard.LaunchDenied` come back as ordinary
    exceptions for the route to turn into a plain answer (never a raw 500).
    """
    return await _start(tampered=False, client_ip=client_ip)


async def start_tamper_test(client_ip: str = "unknown") -> StartOutcome:
    """Same as :func:`start_live_proof`, launching
    ``enclave.live_proof.TAMPERED_IMAGE_DIGEST`` instead of the approved
    image. It takes the SAME lock and counts against the SAME cooldown,
    hourly limits and daily cap: a tampered run spins up exactly the same
    kind of real Confidential VM, so it is bounded the same way.
    """
    return await _start(tampered=True, client_ip=client_ip)


def _cap_bucket() -> Any:
    from google.cloud import storage

    return storage.Client().bucket(_cap_bucket_name())


async def _start(*, tampered: bool, client_ip: str) -> StartOutcome:
    gcfg = guard.GuardConfig.from_env()
    bucket = await asyncio.to_thread(_cap_bucket)
    run_id = uuid.uuid4().hex[:12]

    lock = await asyncio.to_thread(guard.acquire_lock, bucket, run_id, gcfg)
    if not lock.acquired:
        return StartOutcome(lock.run_id, attached=True)
    try:
        await asyncio.to_thread(guard.precheck_limits, bucket, client_ip, gcfg)
        await asyncio.to_thread(check_and_reserve_cap_slot)
        # A launch that loses this last race has already spent a daily slot;
        # at <=10 launches a day that is a rounding error, and the alternative
        # (record first) would count launches that the daily cap then refused.
        await asyncio.to_thread(guard.record_launch, bucket, client_ip, gcfg)
        await asyncio.to_thread(_reap_quietly, run_id, gcfg)
    except BaseException:
        await asyncio.to_thread(guard.release_lock, bucket, run_id)
        raise
    run = LiveProofRun(run_id=run_id, tampered=tampered, gcfg=gcfg, bucket=bucket)
    _runs[run_id] = run
    # Held on the run: the loop keeps only a weak reference to a task, and a
    # run must not be garbage-collected mid-flight.
    run.task = asyncio.create_task(_execute(run))
    return StartOutcome(run_id, attached=False)


def get_run(run_id: str) -> LiveProofRun | None:
    return _runs.get(run_id)


def load_persisted(run_id: str) -> dict[str, Any] | None:
    """A run's public status as another instance last wrote it, or ``None``."""
    return guard.load_run(_cap_bucket(), run_id)


async def log_trail_for(run: LiveProofRun) -> dict[str, Any] | None:
    """What Cloud Logging recorded for this run, or ``None`` while the run
    is still going (its VM entries are not all written yet).

    Cached on the run: a settled trail is returned as-is, an unsettled one
    is re-read at most every ``_LOG_TRAIL_REFRESH_S``.
    """
    if run.finished_at is None:
        return None
    now = time.time()
    cached = run.log_trail
    if cached is not None and (
        cached["status"] in _LOG_TRAIL_FINAL or now - run.log_trail_at < _LOG_TRAIL_REFRESH_S
    ):
        return cached
    run.log_trail = await asyncio.to_thread(
        fetch_trail, f"mulenet-live-proof-{run.run_id}", run.started_at, run.finished_at, now
    )
    run.log_trail_at = now
    return run.log_trail


_LIVE_PROOF_SA = "mulenet-live-proof@mulenet.iam.gserviceaccount.com"


def _delete_vm_quietly(run_id: str) -> None:
    """Ask for this run's VM to be deleted, whatever state it is in.

    The orchestrating thread deletes its own VM in a ``finally``; this is the
    second, independent attempt for the case where that thread is the thing
    that hung. A 404 (already gone) is the success case.
    """
    try:
        from google.cloud import compute_v1

        cfg = _load_config_from_env()
        client = compute_v1.InstancesClient(credentials=_impersonated_credentials(_LIVE_PROOF_SA))
        client.delete(project=cfg.project, zone=cfg.zone, instance=f"mulenet-live-proof-{run_id}")
    except Exception:  # noqa: BLE001 - best effort; the reaper is the next backstop
        _log.warning("could not delete the VM of live-proof run %s", run_id, exc_info=True)


def reap(keep_run_id: str | None = None, gcfg: guard.GuardConfig | None = None) -> list[str]:
    """Delete orphaned live-proof VMs; returns the names asked to be deleted."""
    from google.cloud import compute_v1

    cfg = _load_config_from_env()
    client = compute_v1.InstancesClient(credentials=_impersonated_credentials(_LIVE_PROOF_SA))
    return guard.reap_orphans(
        client, cfg.project, cfg.zone, gcfg or guard.GuardConfig.from_env(), keep_run_id
    )


def _reap_quietly(keep_run_id: str | None, gcfg: guard.GuardConfig) -> None:
    try:
        reap(keep_run_id, gcfg)
    except Exception:  # noqa: BLE001 - a failed sweep must never block a launch
        _log.warning("orphan sweep failed", exc_info=True)


async def reaper_loop(interval_s: float = 300.0) -> None:
    """Sweep for orphaned VMs forever. Started with the service; on a
    CPU-throttled instance it only advances while a request is in flight, which
    is why the sweep also runs before every launch and the VM can carry its own
    platform-side deadline (``MULENET_LIVE_PROOF_MAX_RUN_SECONDS``)."""
    while True:
        await asyncio.sleep(interval_s)
        try:
            bucket = await asyncio.to_thread(_cap_bucket)
            keep = await asyncio.to_thread(guard.active_run_id, bucket)
            await asyncio.to_thread(_reap_quietly, keep, guard.GuardConfig.from_env())
        except Exception:  # noqa: BLE001
            _log.warning("reaper loop iteration failed", exc_info=True)


async def _execute(run: LiveProofRun) -> None:
    gcfg = run.gcfg or guard.GuardConfig.from_env()
    bucket = run.bucket if run.bucket is not None else await asyncio.to_thread(_cap_bucket)

    def on_stage(stage: str, extra: dict[str, Any]) -> None:
        run.record_stage(stage, extra)
        guard.persist_run(bucket, run.run_id, run.public())

    ok = False
    try:
        cfg = _load_config_from_env()
        credentials = _impersonated_credentials(_LIVE_PROOF_SA)
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_live_proof, cfg, run.run_id, credentials, on_stage, run.tampered
            ),
            timeout=gcfg.overall_timeout_s,
        )
        run.result = result
        ok = True
    except TimeoutError:
        _log.error(
            "live-proof run %s exceeded %ss; deleting its VM", run.run_id, gcfg.overall_timeout_s
        )
        run.error = f"timeout: the run did not finish within {gcfg.overall_timeout_s}s"
        await asyncio.to_thread(_delete_vm_quietly, run.run_id)
    except Exception as exc:  # noqa: BLE001 - the UI must see *something*, never hang forever
        # Logged with the full traceback: run.error is only str(exc), and
        # Cloud Run's own request logs carry no background-task output
        # otherwise -- a failure here would otherwise leave no trace
        # anywhere to debug from. Observed live: a masked exception from
        # enclave/live_proof.py's cleanup path once hid the real cause
        # (a missing IAM permission) behind an unrelated 404. For a
        # tampered run, this is ALSO the path that catches "the key was
        # actually released" -- see run_live_proof's tampered branch --
        # so this must never render as a quieter kind of success.
        _log.exception("live-proof run %s failed (tampered=%s)", run.run_id, run.tampered)
        run.error = str(exc)
    finally:
        run.finish(ok=ok)
        await asyncio.to_thread(guard.persist_run, bucket, run.run_id, run.public())
        await asyncio.to_thread(guard.release_lock, bucket, run.run_id)
