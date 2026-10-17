"""Tests for the live-proof cap and run registry (agents/live_proof_service.py).

The cap is the one piece the user explicitly asked to see proven: that it
blocks a run SERVER-SIDE once hit, not just as a UI affordance, AND that
it survives a restart and a race between two concurrent reservations --
not just that it works in one process in one call. That proof lives in
:func:`test_the_11th_call_is_blocked_before_any_vm_work`,
:func:`test_cap_persists_across_a_simulated_restart` and
:func:`test_concurrent_reservations_cannot_both_take_the_last_slot`.

The cap is now durable (a JSON object in GCS, written with
``if_generation_match`` the same way ``enclave/inbox.py``'s
``write_outbox`` already guards its own idempotent create), not a
module-level Python list -- a real counter in memory reset on every
redeploy and every cold start, observed live mid-session. ``_FakeCapBucket``
below models real GCS's conditional-write semantics closely enough to
exercise that retry loop for real, including genuine thread-level
contention for the concurrency test, not just two sequential calls.
``_runs`` (the run registry) is still a module-level global and still
reset per test.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from typing import Any

import pytest
from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage

from agents import live_proof_service as lps
from agents.config import DEFAULT_LOCATION


#: The daily-cap object. The bucket now holds several objects (the cap, the
#: single-flight lock, the launch-limit state, run records), keyed by path;
#: ``content``/``generation`` below stay views of this one so the cap tests
#: read exactly as they always did.
_CAP_PATH = "cap/live-proof-runs.json"


class _FakeCapBlob:
    def __init__(self, bucket: "_FakeCapBucket", name: str) -> None:
        self._bucket = bucket
        self._name = name

    @property
    def generation(self) -> int:
        return self._bucket.objects.get(self._name, ("", 0))[1]

    def download_as_text(self) -> str:
        return self._bucket.objects[self._name][0]

    def upload_from_string(
        self, data: str, content_type: str = "", if_generation_match: int | None = None
    ) -> None:
        # The lock is what makes this a faithful model of GCS's own
        # conditional write: the compare-and-swap is atomic server side,
        # and that atomicity -- not anything in this module's own code
        # -- is what the concurrency test below actually exercises.
        with self._bucket.lock:
            current = self._bucket.objects.get(self._name, ("", 0))[1]
            if if_generation_match is not None and if_generation_match != current:
                raise PreconditionFailed("generation mismatch")
            self._bucket.objects[self._name] = (data, current + 1)

    def delete(self, if_generation_match: int | None = None) -> None:
        with self._bucket.lock:
            current = self._bucket.objects.get(self._name, ("", 0))[1]
            if if_generation_match is not None and if_generation_match != current:
                raise PreconditionFailed("generation mismatch")
            self._bucket.objects.pop(self._name, None)


class _FakeCapBucket:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[str, int]] = {}
        self.lock = threading.Lock()

    @property
    def content(self) -> str | None:
        entry = self.objects.get(_CAP_PATH)
        return entry[0] if entry else None

    @property
    def generation(self) -> int:
        return self.objects.get(_CAP_PATH, ("", 0))[1]

    def get_blob(self, name: str) -> _FakeCapBlob | None:
        return _FakeCapBlob(self, name) if name in self.objects else None

    def blob(self, name: str) -> _FakeCapBlob:
        return _FakeCapBlob(self, name)


class _FakeCapStorageClient:
    def __init__(self, bucket: _FakeCapBucket) -> None:
        self._bucket = bucket

    def bucket(self, name: str) -> _FakeCapBucket:
        return self._bucket


@pytest.fixture(autouse=True)
def cap_bucket(monkeypatch: pytest.MonkeyPatch) -> _FakeCapBucket:
    fake_bucket = _FakeCapBucket()
    monkeypatch.setattr(storage, "Client", lambda **kwargs: _FakeCapStorageClient(fake_bucket))
    monkeypatch.setenv("MULENET_LIVE_PROOF_CAP_BUCKET", "fake-cap-bucket")
    monkeypatch.setattr(lps, "_runs", {})
    return fake_bucket


# ---------------------------------------------------------------------------
# the cap: server-side, checked and reserved before any VM work
# ---------------------------------------------------------------------------


def test_the_11th_call_is_blocked_before_any_vm_work() -> None:
    for _ in range(lps.MAX_LIVE_RUNS_PER_DAY):
        lps.check_and_reserve_cap_slot()  # must not raise

    with pytest.raises(lps.CapExceeded) as exc_info:
        lps.check_and_reserve_cap_slot()

    assert exc_info.value.count == lps.MAX_LIVE_RUNS_PER_DAY
    assert exc_info.value.limit == lps.MAX_LIVE_RUNS_PER_DAY
    # The slot was refused, not reserved anyway: a second call right after
    # must fail identically rather than having silently incremented.
    with pytest.raises(lps.CapExceeded):
        lps.check_and_reserve_cap_slot()


def test_runs_today_reflects_reserved_slots() -> None:
    assert lps.runs_today() == 0
    lps.check_and_reserve_cap_slot()
    lps.check_and_reserve_cap_slot()
    assert lps.runs_today() == 2


def test_cap_is_rolling_24h_not_a_calendar_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run older than 24h must be pruned so it stops counting against
    the cap -- this is what makes it a ROLLING window."""
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(lps.time, "time", lambda: clock["now"])

    for _ in range(lps.MAX_LIVE_RUNS_PER_DAY):
        lps.check_and_reserve_cap_slot()
    with pytest.raises(lps.CapExceeded):
        lps.check_and_reserve_cap_slot()

    clock["now"] += lps._DAY_S + 1.0
    # The old slots are now more than 24h old and must no longer count.
    lps.check_and_reserve_cap_slot()  # must not raise
    assert lps.runs_today() == 1


def test_second_reservation_overwrites_the_existing_cap_object(cap_bucket: _FakeCapBucket) -> None:
    """Regression for the production 403: the first reservation CREATES
    the cap object, and every later one OVERWRITES it with
    if_generation_match. Overwrite needs storage.objects.delete, which
    objectCreator lacks -- so only the create path was ever proven live,
    and the second click failed. This exercises the overwrite path
    explicitly: the object must already exist, the second write must land
    on the same object, and the generation must advance."""
    lps.check_and_reserve_cap_slot()
    first_generation = cap_bucket.generation
    assert cap_bucket.content is not None  # the object now exists

    lps.check_and_reserve_cap_slot()  # the overwrite path

    assert cap_bucket.generation == first_generation + 1
    assert lps.runs_today() == 2


def test_cap_persists_across_a_simulated_restart(cap_bucket: _FakeCapBucket) -> None:
    """The durability requirement: nothing in this module carries any
    python-level count between calls any more -- there is no module-level
    list left to reset -- so the only way the second batch's count can
    build on the first's is if it was read back from the (fake) durable
    GCS object each time, exactly as a real restart would have to."""
    for _ in range(4):
        lps.check_and_reserve_cap_slot()
    assert lps.runs_today() == 4

    # "Restart": a fresh storage.Client() per call is already what the
    # real code does on every invocation (no client is cached across
    # calls), so this is not a special code path -- it is the normal one,
    # demonstrated here by continuing against the same fake backend.
    for _ in range(6):
        lps.check_and_reserve_cap_slot()
    assert lps.runs_today() == 10
    with pytest.raises(lps.CapExceeded):
        lps.check_and_reserve_cap_slot()


def test_concurrent_reservations_cannot_both_take_the_last_slot(
    cap_bucket: _FakeCapBucket,
) -> None:
    """Two real threads race for the 10th slot. Exactly one must win --
    the retry-on-PreconditionFailed loop must never let both succeed,
    which would silently grant an 11th run past the cap."""
    for _ in range(9):
        lps.check_and_reserve_cap_slot()
    assert lps.runs_today() == 9

    results: list[str] = []
    results_lock = threading.Lock()

    def _try_reserve() -> None:
        try:
            lps.check_and_reserve_cap_slot()
            outcome = "ok"
        except lps.CapExceeded:
            outcome = "capped"
        with results_lock:
            results.append(outcome)

    t1 = threading.Thread(target=_try_reserve)
    t2 = threading.Thread(target=_try_reserve)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert sorted(results) == ["capped", "ok"]
    assert lps.runs_today() == 10


def test_start_live_proof_raises_cap_exceeded_synchronously_when_already_hit() -> None:
    """start_live_proof must refuse BEFORE creating any run_id or task --
    the route turns this into the static-evidence fallback, never a dead
    button or a raw error."""
    for _ in range(lps.MAX_LIVE_RUNS_PER_DAY):
        lps.check_and_reserve_cap_slot()

    with pytest.raises(lps.CapExceeded):
        asyncio.run(lps.start_live_proof())

    assert lps._runs == {}


# ---------------------------------------------------------------------------
# the run registry
# ---------------------------------------------------------------------------


def test_start_live_proof_returns_a_run_id_and_registers_it(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_execute(run: lps.LiveProofRun) -> None:
        run.stage = "done"
        run.result = {"ok": True}

    monkeypatch.setattr(lps, "_execute", _fake_execute)

    async def _drive() -> str:
        run_id = (await lps.start_live_proof()).run_id
        # Let the background task (asyncio.create_task in start_live_proof)
        # actually run before the loop is torn down.
        await asyncio.sleep(0)
        return run_id

    run_id = asyncio.run(_drive())
    assert run_id in lps._runs
    assert lps.get_run(run_id) is lps._runs[run_id]
    assert lps._runs[run_id].tampered is False


def test_start_tamper_test_marks_the_run_tampered_and_shares_the_cap(
    monkeypatch: pytest.MonkeyPatch, cap_bucket: _FakeCapBucket
) -> None:
    async def _fake_execute(run: lps.LiveProofRun) -> None:
        run.stage = "done"

    monkeypatch.setattr(lps, "_execute", _fake_execute)

    async def _drive() -> str:
        run_id = (await lps.start_tamper_test()).run_id
        await asyncio.sleep(0)
        return run_id

    run_id = asyncio.run(_drive())
    assert lps._runs[run_id].tampered is True
    # Same cap, same bucket, same counter -- a tampered run consumes a
    # slot exactly like a normal one.
    assert lps.runs_today() == 1


def test_get_run_returns_none_for_an_unknown_id() -> None:
    assert lps.get_run("no-such-run") is None


def test_execute_marks_the_run_failed_on_any_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> Any:
        raise RuntimeError("MULENET_ENCLAVE_PROJECT is not set; live proof is unavailable")

    monkeypatch.setattr(lps, "_load_config_from_env", _boom)

    run = lps.LiveProofRun(run_id="r1")
    asyncio.run(lps._execute(run))

    assert run.stage == "failed"
    assert run.error is not None and "MULENET_ENCLAVE_PROJECT" in run.error
    assert run.result is None


def test_execute_records_the_result_and_stage_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lps, "_load_config_from_env", lambda: object())
    monkeypatch.setattr(lps, "_impersonated_credentials", lambda principal: object())

    def _fake_run_live_proof(
        cfg: Any, run_id: str, credentials: Any, on_stage: Any, tampered: bool = False
    ) -> dict[str, Any]:
        on_stage("vm_running", {})
        return {"ok": True, "decrypted_preview": "hi"}

    monkeypatch.setattr(lps, "run_live_proof", _fake_run_live_proof)

    run = lps.LiveProofRun(run_id="r2")
    asyncio.run(lps._execute(run))

    assert run.stage == "done"
    assert run.result == {"ok": True, "decrypted_preview": "hi"}
    assert run.error is None


# ---------------------------------------------------------------------------
# config loading: fails closed with a clear error, never silently
# ---------------------------------------------------------------------------


def test_load_config_from_env_raises_cleanly_when_a_var_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MULENET_ENCLAVE_PROJECT", raising=False)
    with pytest.raises(RuntimeError, match="MULENET_ENCLAVE_PROJECT"):
        lps._load_config_from_env()


def test_load_config_from_env_decodes_base64_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_correct = b"\x00" * 32
    raw_wrong = b"\x01" * 32
    env = {
        "MULENET_ENCLAVE_PROJECT": "mulenet",
        "MULENET_ENCLAVE_REGION": DEFAULT_LOCATION,
        "MULENET_ENCLAVE_ZONE": "asia-south1-a",
        "MULENET_ENCLAVE_IMAGE_DIGEST": "sha256:deadbeef",
        "MULENET_ENCLAVE_WORKLOAD_SA": "wl@mulenet.iam.gserviceaccount.com",
        "MULENET_ENCLAVE_INBOX_BUCKET": "inbox",
        "MULENET_ENCLAVE_OUTBOX_BUCKET": "outbox",
        "MULENET_ENCLAVE_EVIDENCE_BUCKET": "evidence",
        "MULENET_ENCLAVE_KMS_KEY": "projects/mulenet/locations/asia-south1/keyRings/k/cryptoKeys/edge",
        "MULENET_ENCLAVE_HASH_KMS_KEY": "projects/mulenet/locations/asia-south1/keyRings/k/cryptoKeys/hash",
        "MULENET_ENCLAVE_HASH_KEY_URI": "gs://keys/hash-rekey.bin",
        "MULENET_ENCLAVE_WIP_AUDIENCE": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/pr",
        "MULENET_ENCLAVE_BANK_IDS": "bank-a,bank-b,bank-c",
        "MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_A": base64.b64encode(b"\x02" * 32).decode(),
        "MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_B": base64.b64encode(b"\x03" * 32).decode(),
        "MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_C": base64.b64encode(b"\x04" * 32).decode(),
        "MULENET_ENCLAVE_FLAGGED_BANK_A": "acct-a1,acct-a2",
        "MULENET_ENCLAVE_FLAGGED_BANK_B": "acct-b1",
        # BANK_C deliberately left unset -- optional, must default to ().
        "MULENET_ENCLAVE_LIVE_PROOF_INBOX_RUN_ID": "stage3v2",
        "MULENET_LIVE_PROOF_CORRECT_BANK": "bank-a",
        "MULENET_LIVE_PROOF_CORRECT_KEY_B64": base64.b64encode(raw_correct).decode(),
        "MULENET_LIVE_PROOF_WRONG_BANK": "bank-b",
        "MULENET_LIVE_PROOF_WRONG_KEY_B64": base64.b64encode(raw_wrong).decode(),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    cfg = lps._load_config_from_env()
    assert cfg.correct_private_key == raw_correct
    assert cfg.wrong_private_key == raw_wrong
    assert cfg.bank_ids == ("bank-a", "bank-b", "bank-c")
    assert cfg.flagged_account_ids == {
        "bank-a": ("acct-a1", "acct-a2"),
        "bank-b": ("acct-b1",),
        "bank-c": (),
    }
    assert cfg.outbox_public_keys == {
        "bank-a": b"\x02" * 32,
        "bank-b": b"\x03" * 32,
        "bank-c": b"\x04" * 32,
    }
