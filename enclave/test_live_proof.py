"""Tests for the judge-facing live-proof orchestrator (enclave/live_proof.py).

Everything here is offline: no real Compute Engine or GCS call is made.
``run_live_proof`` imports ``google.cloud.compute_v1``/``storage`` locally
on each call (so the ``agents`` container, which has no gcloud CLI, still
gets a clean module import), which means the already-installed real SDK
modules can be monkeypatched in place -- ``_build_instance`` and
``_metadata_items`` are exercised against the REAL ``compute_v1`` classes
(pure, local dataclass construction, no network), while
``InstancesClient``/``storage.Client`` are replaced with fakes for the
calls that would otherwise hit the network.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from google.cloud import compute_v1, storage

from enclave.live_proof import (
    STAGE_CLEANING_UP,
    STAGE_CORRECT_KEY_ATTEMPT,
    STAGE_EVIDENCE_RECEIVED,
    STAGE_FAILED,
    STAGE_KMS_REFUSED,
    STAGE_REQUESTING_VM,
    STAGE_VM_RUNNING,
    STAGE_WRONG_KEY_ATTEMPT,
    TAMPERED_IMAGE_DIGEST,
    LiveProofConfig,
    _build_instance,
    _metadata_items,
    run_live_proof,
)
from enclave.outbox_crypto import generate_keypair, seal_for_bank


def _cfg(**overrides: Any) -> tuple[LiveProofConfig, bytes]:
    correct_private, correct_public = generate_keypair()
    wrong_private, _wrong_public = generate_keypair()
    fields: dict[str, Any] = dict(
        project="mulenet",
        region="asia-south1",
        zone="asia-south1-a",
        image_digest="sha256:deadbeef",
        workload_sa_email="mulenet-enclave-workload@mulenet.iam.gserviceaccount.com",
        inbox_bucket="mulenet-enclave-inbox",
        outbox_bucket="mulenet-enclave-outbox",
        evidence_bucket="mulenet-enclave-evidence",
        kms_edge_key="projects/mulenet/locations/asia-south1/keyRings/mulenet-enclave/cryptoKeys/edge-wrap",
        kms_hash_key="projects/mulenet/locations/asia-south1/keyRings/mulenet-enclave/cryptoKeys/hash-rekey",
        hash_key_uri="gs://mulenet-enclave-keys/hash-rekey.bin",
        wip_audience="//iam.googleapis.com/projects/123/locations/global/workloadIdentityPools/mulenet-enclave-wip/providers/mulenet-attestation",
        bank_ids=("bank-a", "bank-b", "bank-c"),
        outbox_public_keys={
            "bank-a": correct_public,
            "bank-b": generate_keypair()[1],
            "bank-c": generate_keypair()[1],
        },
        flagged_account_ids={
            "bank-a": ("acct-flagged-a1",),
            "bank-b": ("acct-flagged-b1",),
            "bank-c": (),
        },
        inbox_run_id="stage3v2",
        correct_bank_id="bank-a",
        correct_private_key=correct_private,
        wrong_bank_id="bank-b",
        wrong_private_key=wrong_private,
        run_timeout_s=30,
    )
    fields.update(overrides)
    return LiveProofConfig(**fields), correct_public


# ---------------------------------------------------------------------------
# _metadata_items / _build_instance: pure, no network
# ---------------------------------------------------------------------------


def test_metadata_run_id_is_unique_per_run_not_the_static_inbox_run_id() -> None:
    """Regression: the workload (enclave/workload.py) uses ONE run id for
    its inbox prefix, outbox prefix AND evidence path. Reusing the static
    inbox_run_id made every run's deterministic ring id collide with the
    outbox bundle an earlier capture already wrote (a 412 on write) --
    observed live after the enclave had genuinely attested, decrypted and
    found the real ring. A fresh id per click avoids that collision."""
    cfg, _pub = _cfg()
    items = _metadata_items(cfg, "abc123", cfg.image_digest)
    by_key = {i.key: i.value for i in items}
    assert by_key["tee-env-MULENET_ENCLAVE_RUN_ID"] == "live-proof-abc123"
    assert by_key["tee-env-MULENET_ENCLAVE_ATTEMPT"] == "run"


def test_metadata_items_carry_the_resolved_image_and_bank_ids() -> None:
    cfg, _pub = _cfg()
    items = _metadata_items(cfg, "run1", cfg.image_digest)
    by_key = {i.key: i.value for i in items}
    assert by_key["tee-image-reference"] == (
        "asia-south1-docker.pkg.dev/mulenet/mulenet-enclave/scorer@sha256:deadbeef"
    )
    assert by_key["tee-env-MULENET_ENCLAVE_RUN_ID"] == "live-proof-run1"
    assert by_key["tee-env-MULENET_ENCLAVE_BANK_IDS"] == "bank-a,bank-b,bank-c"


def test_metadata_carries_every_banks_outbox_pubkey() -> None:
    """Regression: observed live -- missing even one of these makes
    enclave.config.EnclaveConfig.from_env() raise ValueError, the one
    path in enclave/workload.py's main() with no evidence-writing
    fallback at all. A VM hit this, ran to completion, and cleanly
    self-terminated while writing nothing, with no way to see why from
    outside the production image."""
    import base64

    cfg, correct_public = _cfg()
    items = _metadata_items(cfg, "run1", cfg.image_digest)
    by_key = {i.key: i.value for i in items}
    for bank_id, env_suffix in (("bank-a", "BANK_A"), ("bank-b", "BANK_B"), ("bank-c", "BANK_C")):
        key = f"tee-env-MULENET_ENCLAVE_OUTBOX_PUBKEY_{env_suffix}"
        assert key in by_key
        assert base64.b64decode(by_key[key]) == cfg.outbox_public_keys[bank_id]


def test_metadata_carries_each_banks_flagged_accounts() -> None:
    """Regression: observed live -- omitting this made ring-finding run
    over an empty flagged set. A run then completes cleanly, attests,
    and gets its KMS key, and still has nothing to decrypt: not a crash,
    just silently nothing to show."""
    cfg, _pub = _cfg()
    items = _metadata_items(cfg, "run1", cfg.image_digest)
    by_key = {i.key: i.value for i in items}
    assert by_key["tee-env-MULENET_ENCLAVE_FLAGGED_BANK_A"] == "acct-flagged-a1"
    assert by_key["tee-env-MULENET_ENCLAVE_FLAGGED_BANK_B"] == "acct-flagged-b1"
    assert by_key["tee-env-MULENET_ENCLAVE_FLAGGED_BANK_C"] == ""


def test_build_instance_is_confidential_sev_with_the_workload_sa() -> None:
    cfg, _pub = _cfg()
    instance = _build_instance(cfg, "mulenet-live-proof-abc123", "abc123", cfg.image_digest)
    assert instance.machine_type.endswith("/machineTypes/c2d-standard-2")
    assert instance.confidential_instance_config.enable_confidential_compute is True
    assert (
        instance.confidential_instance_config.confidential_instance_type
        == compute_v1.ConfidentialInstanceConfig.ConfidentialInstanceType.SEV.name
    )
    assert instance.service_accounts[0].email == cfg.workload_sa_email
    assert instance.labels == {"purpose": "live-proof", "mulenet-run": "abc123"}


# ---------------------------------------------------------------------------
# run_live_proof: fakes for InstancesClient and storage.Client
# ---------------------------------------------------------------------------


class _FakeOperation:
    def result(self, timeout: int) -> None:
        return None


class _FakeInstancesClient:
    def __init__(self, credentials: Any = None) -> None:
        self.credentials = credentials
        self.inserted: list[str] = []
        self.inserted_instances: list[Any] = []
        self.deleted: list[str] = []

    def insert(self, project: str, zone: str, instance_resource: Any) -> _FakeOperation:
        self.inserted.append(instance_resource.name)
        self.inserted_instances.append(instance_resource)
        return _FakeOperation()

    def delete(self, project: str, zone: str, instance: str) -> _FakeOperation:
        self.deleted.append(instance)
        return _FakeOperation()


class _FakeBlob:
    def __init__(self, content: str | None, size: int = 0) -> None:
        self._content = content
        self.size = size

    def exists(self) -> bool:
        return self._content is not None

    def download_as_text(self) -> str:
        assert self._content is not None
        return self._content


class _FakeBucket:
    def __init__(self, objects: dict[str, _FakeBlob]) -> None:
        self._objects = objects

    def blob(self, name: str) -> _FakeBlob:
        return self._objects.get(name, _FakeBlob(None))

    def copy_blob(self, source_blob: _FakeBlob, destination_bucket: _FakeBucket, new_name: str) -> None:
        if not source_blob.exists():
            raise FileNotFoundError(f"fake source blob missing: {new_name}")
        destination_bucket._objects[new_name] = source_blob


class _FakeStorageClient:
    def __init__(self, buckets: dict[str, dict[str, _FakeBlob]], project: str = "", credentials: Any = None) -> None:
        self._buckets = buckets

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self._buckets.get(name, {}))


def _inbox_seed(cfg: LiveProofConfig) -> dict[str, _FakeBlob]:
    """The sealed inbox ciphertext run_live_proof copies before every VM
    launch -- see _enclave_run_id's docstring. Content is irrelevant to
    these tests (it is opaque ciphertext either way); only its presence
    at the expected source path matters."""
    return {f"inbox/{bank_id}/{cfg.inbox_run_id}.json": _FakeBlob("ciphertext") for bank_id in cfg.bank_ids}


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    buckets: dict[str, dict[str, _FakeBlob]],
) -> _FakeInstancesClient:
    fake_instances = _FakeInstancesClient()
    monkeypatch.setattr(compute_v1, "InstancesClient", lambda credentials=None: fake_instances)
    monkeypatch.setattr(
        storage, "Client", lambda project="", credentials=None: _FakeStorageClient(buckets, project, credentials)
    )
    return fake_instances


def test_full_cycle_wrong_key_fails_then_correct_key_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, correct_public = _cfg()
    plaintext = b'{"ring_id": "r1", "alert": "demo"}'
    sealed = seal_for_bank(cfg.correct_bank_id, plaintext, correct_public)
    bundle_path = f"outbox/{cfg.correct_bank_id}/live-proof-run1/r1.json"
    evidence = {
        "outcome": "completed",
        "key_release_ok": True,
        "bundles_written": [bundle_path],
    }
    buckets = {
        cfg.inbox_bucket: _inbox_seed(cfg),
        cfg.evidence_bucket: {"runs/live-proof-run1/run.json": _FakeBlob(json.dumps(evidence))},
        cfg.outbox_bucket: {bundle_path: _FakeBlob(sealed.to_json(), size=999)},
    }
    fake_instances = _install_fakes(monkeypatch, buckets)

    stages: list[str] = []
    extras: list[dict[str, Any]] = []

    def on_stage(stage: str, extra: dict[str, Any]) -> None:
        stages.append(stage)
        extras.append(extra)

    result = run_live_proof(cfg, "run1", credentials=None, on_stage=on_stage)

    assert result["ok"] is True
    assert result["wrong_key_error"] == "InvalidTag"
    assert result["decrypted_preview"] == plaintext.decode()
    assert result["ciphertext_bytes_total"] == 999

    assert fake_instances.inserted == ["mulenet-live-proof-run1"]
    assert fake_instances.deleted == ["mulenet-live-proof-run1"]

    assert STAGE_REQUESTING_VM in stages
    assert STAGE_VM_RUNNING in stages
    assert STAGE_EVIDENCE_RECEIVED in stages
    assert STAGE_CLEANING_UP in stages

    wrong_extra = next(e for s, e in zip(stages, extras) if s == STAGE_WRONG_KEY_ATTEMPT and "failed_as_expected" in e)
    assert wrong_extra["failed_as_expected"] is True
    assert wrong_extra["error"] == "InvalidTag"

    correct_extra = next(e for s, e in zip(stages, extras) if s == STAGE_CORRECT_KEY_ATTEMPT and "succeeded" in e)
    assert correct_extra["succeeded"] is True


def _tee_image_reference(instance: Any) -> str | None:
    return next((i.value for i in instance.metadata.items if i.key == "tee-image-reference"), None)


def test_tampered_mode_launches_the_tampered_digest_not_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _pub = _cfg()
    evidence = {"outcome": "key_release_refused", "refusal_kind": "kms_decrypt_refused", "refusal_message": "403 denied"}
    buckets = {
        cfg.inbox_bucket: _inbox_seed(cfg),
        cfg.evidence_bucket: {"runs/live-proof-run6/run.json": _FakeBlob(json.dumps(evidence))},
    }
    fake_instances = _install_fakes(monkeypatch, buckets)

    run_live_proof(cfg, "run6", credentials=None, on_stage=lambda s, e: None, tampered=True)

    assert len(fake_instances.inserted_instances) == 1
    image_ref = _tee_image_reference(fake_instances.inserted_instances[0])
    assert image_ref is not None and TAMPERED_IMAGE_DIGEST in image_ref
    assert cfg.image_digest not in image_ref


def test_tampered_mode_succeeds_on_a_real_kms_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _pub = _cfg()
    evidence = {
        "outcome": "key_release_refused",
        "refusal_kind": "kms_decrypt_refused",
        "refusal_message": "403 Permission 'cloudkms.cryptoKeyVersions.useToDecrypt' denied",
    }
    buckets = {
        cfg.inbox_bucket: _inbox_seed(cfg),
        cfg.evidence_bucket: {"runs/live-proof-run7/run.json": _FakeBlob(json.dumps(evidence))},
    }
    fake_instances = _install_fakes(monkeypatch, buckets)

    stages: list[str] = []
    result = run_live_proof(
        cfg, "run7", credentials=None, on_stage=lambda s, e: stages.append(s), tampered=True
    )

    assert result["ok"] is True
    assert result["tampered"] is True
    assert result["approved_digest"] == cfg.image_digest
    assert result["tampered_digest"] == TAMPERED_IMAGE_DIGEST
    assert result["refusal_kind"] == "kms_decrypt_refused"
    assert "denied" in result["refusal_message"]
    assert STAGE_KMS_REFUSED in stages
    assert STAGE_CLEANING_UP in stages
    # Never reaches the decrypt-attempt stages: there is nothing to decrypt
    # when the key itself was refused.
    assert STAGE_WRONG_KEY_ATTEMPT not in stages
    assert STAGE_CORRECT_KEY_ATTEMPT not in stages
    assert fake_instances.deleted == ["mulenet-live-proof-run7"]


def test_tampered_mode_fails_loudly_if_the_key_was_actually_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one outcome that must NEVER be treated as success: the
    tampered digest somehow decrypting anyway."""
    cfg, _pub = _cfg()
    evidence = {"outcome": "completed", "key_release_ok": True, "bundles_written": []}
    buckets = {
        cfg.inbox_bucket: _inbox_seed(cfg),
        cfg.evidence_bucket: {"runs/live-proof-run8/run.json": _FakeBlob(json.dumps(evidence))},
    }
    _install_fakes(monkeypatch, buckets)

    with pytest.raises(RuntimeError, match="did not refuse as expected"):
        run_live_proof(cfg, "run8", credentials=None, on_stage=lambda s, e: None, tampered=True)


def test_tampered_mode_fails_loudly_on_the_wrong_refusal_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal is expected, but the WRONG kind (e.g. blocked at
    federation, not at KMS) would prove a different claim than the one
    this demo makes, so it must fail loudly too, not pass quietly."""
    cfg, _pub = _cfg()
    evidence = {"outcome": "key_release_refused", "refusal_kind": "sts_exchange_refused"}
    buckets = {
        cfg.inbox_bucket: _inbox_seed(cfg),
        cfg.evidence_bucket: {"runs/live-proof-run9/run.json": _FakeBlob(json.dumps(evidence))},
    }
    _install_fakes(monkeypatch, buckets)

    with pytest.raises(RuntimeError, match="did not refuse as expected"):
        run_live_proof(cfg, "run9", credentials=None, on_stage=lambda s, e: None, tampered=True)


#: Set to "1" to run the one test that calls real Cloud KMS. Mirrors
#: agents/test_service.py's LIVE_GEMINI_ENV pattern: off by default, so
#: pytest never needs real GCP credentials or costs anything by itself.
LIVE_KMS_POLICY_CHECK_ENV = "MULENET_RUN_LIVE_KMS_POLICY_CHECK"


@pytest.mark.skipif(
    os.environ.get(LIVE_KMS_POLICY_CHECK_ENV) != "1",
    reason=f"set {LIVE_KMS_POLICY_CHECK_ENV}=1 to check the real KMS policy (needs real GCP credentials)",
)
def test_tampered_digest_is_never_in_either_kms_keys_decrypt_policy() -> None:
    """The one condition that would make deliberately keeping
    TAMPERED_IMAGE_DIGEST around dangerous, per its own module
    docstring: if it ever appears in either key's decrypt policy, the
    demo's whole claim (attestation gates key release) is false, not
    just untested. Run for real against the live mulenet project
    before trusting the tamper-test feature with a judge watching."""
    from google.cloud import kms

    client = kms.KeyManagementServiceClient()
    member_suffix = f"attribute.image_digest/{TAMPERED_IMAGE_DIGEST}"
    for key_resource in (
        "projects/mulenet/locations/asia-south1/keyRings/mulenet-enclave/cryptoKeys/edge-wrap",
        "projects/mulenet/locations/asia-south1/keyRings/mulenet-enclave/cryptoKeys/hash-rekey",
    ):
        policy = client.get_iam_policy(request={"resource": key_resource})
        for binding in policy.bindings:
            if binding.role != "roles/cloudkms.cryptoKeyDecrypter":
                continue
            for member in binding.members:
                assert member_suffix not in member, (
                    f"{key_resource} grants decrypt to the TAMPERED digest "
                    f"({member!r}) -- revoke this immediately with "
                    f"infra/enclave_pin_digest.sh --revoke {TAMPERED_IMAGE_DIGEST}; "
                    "the tamper-test demo's whole claim depends on this never "
                    "being true"
                )


def test_inbox_is_copied_to_a_fresh_name_before_the_vm_is_created(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: observed live -- the enclave genuinely attested,
    decrypted and found the real ring, then crashed with a 412 writing
    its outbox bundle, because that deterministic path already existed
    from an earlier capture under the same (reused) run id."""
    cfg, correct_public = _cfg()
    plaintext = b"{}"
    sealed = seal_for_bank(cfg.correct_bank_id, plaintext, correct_public)
    bundle_path = f"outbox/{cfg.correct_bank_id}/live-proof-run9/r1.json"
    evidence = {"outcome": "completed", "key_release_ok": True, "bundles_written": [bundle_path]}
    buckets = {
        cfg.inbox_bucket: _inbox_seed(cfg),
        cfg.evidence_bucket: {"runs/live-proof-run9/run.json": _FakeBlob(json.dumps(evidence))},
        cfg.outbox_bucket: {bundle_path: _FakeBlob(sealed.to_json())},
    }
    _install_fakes(monkeypatch, buckets)

    run_live_proof(cfg, "run9", credentials=None, on_stage=lambda s, e: None)

    for bank_id in cfg.bank_ids:
        source_name = f"inbox/{bank_id}/{cfg.inbox_run_id}.json"
        copied_name = f"inbox/{bank_id}/live-proof-run9.json"
        assert copied_name in buckets[cfg.inbox_bucket], f"no copy made for {bank_id}"
        assert buckets[cfg.inbox_bucket][copied_name] is buckets[cfg.inbox_bucket][source_name]


def test_inbox_copy_failure_prevents_vm_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the source inbox ciphertext is missing, no VM should ever be
    requested -- there would be nothing for it to read."""
    cfg, _pub = _cfg()
    fake_instances = _install_fakes(monkeypatch, buckets={})  # no inbox seed at all

    with pytest.raises(FileNotFoundError):
        run_live_proof(cfg, "run10", credentials=None, on_stage=lambda s, e: None)

    assert fake_instances.inserted == []


def test_vm_is_deleted_even_when_evidence_never_arrives(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timeout must never leave a VM running -- the deletion happens in
    a `finally`, and this proves it fires on the failure path too."""
    cfg, _pub = _cfg(run_timeout_s=0)
    fake_instances = _install_fakes(monkeypatch, buckets={cfg.inbox_bucket: _inbox_seed(cfg)})

    stages: list[str] = []
    with pytest.raises(TimeoutError):
        run_live_proof(cfg, "run2", credentials=None, on_stage=lambda s, e: stages.append(s))

    assert fake_instances.deleted == ["mulenet-live-proof-run2"]
    assert STAGE_CLEANING_UP in stages


def test_insert_failure_is_not_masked_by_a_delete_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: observed live against real GCP -- insert() failed on a
    missing IAM permission, but the finally block still tried to delete
    a VM that was never created, and that 404 REPLACED the real error.
    No VM exists yet, so delete() must not even be attempted."""
    cfg, _pub = _cfg()
    fake_instances = _install_fakes(monkeypatch, buckets={cfg.inbox_bucket: _inbox_seed(cfg)})

    def _boom(project: str, zone: str, instance_resource: Any) -> _FakeOperation:
        raise RuntimeError("Required 'compute.instances.setMetadata' permission")

    monkeypatch.setattr(fake_instances, "insert", _boom)

    stages: list[str] = []
    with pytest.raises(RuntimeError, match="setMetadata"):
        run_live_proof(cfg, "run4", credentials=None, on_stage=lambda s, e: stages.append(s))

    assert fake_instances.deleted == []
    assert STAGE_CLEANING_UP in stages


def test_a_real_failure_wins_over_a_cleanup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: observed live -- the main cycle failed fast (a
    RuntimeError, far too quick to be a real attestation run), and then
    delete() ALSO timed out. The RuntimeError must propagate; the delete
    timeout must be recorded via on_stage but not replace it."""
    cfg, _pub = _cfg(run_timeout_s=0)  # no evidence ever found -> TimeoutError
    fake_instances = _install_fakes(monkeypatch, buckets={cfg.inbox_bucket: _inbox_seed(cfg)})

    def _boom_delete(project: str, zone: str, instance: str) -> _FakeOperation:
        raise RuntimeError("delete timed out")

    monkeypatch.setattr(fake_instances, "delete", _boom_delete)

    stages: list[str] = []
    with pytest.raises(TimeoutError):  # the ORIGINAL failure, not the delete timeout
        run_live_proof(cfg, "run5", credentials=None, on_stage=lambda s, e: stages.append(s))

    assert STAGE_FAILED in stages


def test_cleanup_failure_is_raised_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """"deletion must never be skipped silently" -- if delete() itself
    raises, that exception must propagate, not the original one."""
    cfg, correct_public = _cfg()
    plaintext = b"{}"
    sealed = seal_for_bank(cfg.correct_bank_id, plaintext, correct_public)
    bundle_path = f"outbox/{cfg.correct_bank_id}/live-proof-run3/r1.json"
    evidence = {"outcome": "completed", "key_release_ok": True, "bundles_written": [bundle_path]}
    buckets = {
        cfg.inbox_bucket: _inbox_seed(cfg),
        cfg.evidence_bucket: {"runs/live-proof-run3/run.json": _FakeBlob(json.dumps(evidence))},
        cfg.outbox_bucket: {bundle_path: _FakeBlob(sealed.to_json())},
    }
    fake_instances = _install_fakes(monkeypatch, buckets)

    def _boom(project: str, zone: str, instance: str) -> _FakeOperation:
        raise RuntimeError("delete failed")

    monkeypatch.setattr(fake_instances, "delete", _boom)

    stages: list[str] = []
    with pytest.raises(RuntimeError, match="delete failed"):
        run_live_proof(cfg, "run3", credentials=None, on_stage=lambda s, e: stages.append(s))

    assert STAGE_FAILED in stages
