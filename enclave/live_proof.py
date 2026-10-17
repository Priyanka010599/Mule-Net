"""Orchestrate one live, judge-facing encrypt -> attest -> decrypt proof.

Reuses the real inbox bundles already sealed by banks A/B/C earlier
(``infra/enclave_send_edges.py``) and the real, currently-pinned
Confidential Space image -- nothing here reseals input data, because the
point of this proof is that the ENCLAVE'S work (attestation, KMS key
release, decrypt, re-encrypt) is fresh and live, not that the input
changes. Each run gets its own ``run_id``, so each click is a genuinely
new VM, a genuinely new attestation exchange, and a genuinely new outbox
write.

Runs as ``mulenet-live-proof@...`` (impersonated by ``agents/``, which
holds only ``roles/iam.serviceAccountTokenCreator`` on it). That identity
has compute instance create/delete/get/list, disk create, and ``actAs``
on ``mulenet-enclave-workload`` -- and deliberately **no** KMS permission
at all: the gated KMS decrypt happens via the VM's own
attestation-federated identity at runtime, never via this orchestrating
identity, so it needs none.

This reimplements ``infra/enclave_run.sh``'s VM lifecycle via the Compute
Engine Python client rather than shelling to ``gcloud`` -- the ``agents``
container has no gcloud CLI, by design (infra/Dockerfile.agents).

Observable stages, and why there are not more of them
-------------------------------------------------------
Attestation, KMS key release, decrypt and re-encryption all happen
**inside** the attested VM, and the production Confidential Space image
does not redirect container stdout by design (confirmed live, this
session's earlier work) -- so an external orchestrator genuinely cannot
see finer-grained progress than "the VM is running" and "the evidence
object appeared". Reporting fake sub-steps in between would be a demo
embellishment, not a real progress indicator, so the stage list below
names exactly what can actually be observed, with the long, opaque wait
labelled as exactly that -- which is itself the point being demonstrated:
nobody outside the attested machine, including this orchestrator, can see
what happens in there.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: Stage names, in order. The UI renders these as a step list.
STAGE_REQUESTING_VM = "requesting_vm"
STAGE_VM_RUNNING = "vm_running"
STAGE_AWAITING_ENCLAVE = "awaiting_enclave"  # attestation + KMS + decrypt + re-encrypt, opaque by design
STAGE_EVIDENCE_RECEIVED = "evidence_received"
STAGE_WRONG_KEY_ATTEMPT = "wrong_key_attempt"
STAGE_CORRECT_KEY_ATTEMPT = "correct_key_attempt"
#: Tamper-test mode only: KMS refused the key, which IS the success case
#: for this mode -- see run_live_proof's tampered branch.
STAGE_KMS_REFUSED = "kms_refused"
STAGE_CLEANING_UP = "cleaning_up"
STAGE_DONE = "done"
STAGE_FAILED = "failed"

#: Google's documented Confidential Space metadata keys (same as
#: infra/enclave_run.sh, confirmed against Google's own published sample
#: scripts earlier this session).
_IMAGE_PROJECT = "confidential-space-images"
_IMAGE_FAMILY = "confidential-space"
_MACHINE_TYPE = "c2d-standard-2"

#: The tamper-test demo's known-refused image: the CURRENT approved source
#: with enclave/tamper.patch applied (one line, committed in git), built by
#: infra/enclave_build.sh tampered. Its digest is a DIFFERENT KMS IAM principal
#: than the approved one, with no grant on either key.
#:
#: Reused deliberately, by digest: never re-tagged. This digest is what the
#: demo hardcodes, so if KMS ever granted it decrypt the demo would be lying.
#: enclave/test_live_proof.py's live KMS-policy check fails the build if that
#: ever happens.
TAMPERED_IMAGE_DIGEST = "sha256:57ab9404a3ea7f29a6fd97d98880721e43fe14120c652f15bf787392c52153a5"

#: The literal line that was changed to produce TAMPERED_IMAGE_DIGEST,
#: for the UI to show next to the digest comparison.
TAMPERED_LINE_CHANGED = "manifest.flagged_count = len(flagged) + 1  # TAMPERED for the negative test"


@dataclass(frozen=True)
class LiveProofConfig:
    """Every resource name this orchestrator needs, resolved once."""

    project: str
    region: str
    zone: str
    image_digest: str  # sha256:...  -- the one digest KMS will release the key to
    workload_sa_email: str
    inbox_bucket: str
    outbox_bucket: str
    evidence_bucket: str
    kms_edge_key: str
    kms_hash_key: str
    hash_key_uri: str
    wip_audience: str
    bank_ids: tuple[str, ...]
    #: Each bank's X25519 public key (32 raw bytes), keyed by bank id.
    #: Required for every id in ``bank_ids`` -- enclave.config.EnclaveConfig
    #: .from_env() raises ValueError without one, and that is the ONE path
    #: in enclave/workload.py's main() with no evidence-writing fallback:
    #: observed live, a VM that boots, runs, and cleanly self-terminates
    #: (compute.instances.guestTerminate) while writing nothing at all,
    #: because this was missing from the metadata entirely.
    outbox_public_keys: dict[str, bytes]
    #: Each bank's flagged account ids (hashed, as the bank already uses
    #: them), keyed by bank id. In a full deployment this would travel
    #: inside the sealed inbox bundle; here it is env-var metadata
    #: instead (enclave/workload.py's _flagged_from_env, which owns the
    #: real explanation). Required: with none set, enclave.rings
    #: .find_candidates runs over an empty flagged set and -- correctly,
    #: not a bug -- finds zero rings. Observed live: a run with this
    #: omitted completes cleanly, attests, and gets its KMS key, and
    #: still has nothing to decrypt.
    flagged_account_ids: dict[str, tuple[str, ...]]
    #: run_id of the real, already-sealed inbox bundles this proof reuses.
    inbox_run_id: str
    correct_bank_id: str
    correct_private_key: bytes
    wrong_bank_id: str
    wrong_private_key: bytes
    run_timeout_s: int = 420


def _image_ref(cfg: LiveProofConfig, image_digest: str) -> str:
    return f"{cfg.region}-docker.pkg.dev/{cfg.project}/mulenet-enclave/scorer@{image_digest}"


def _enclave_run_id(run_id: str) -> str:
    """The run id handed to the workload itself -- unique per click.

    enclave/workload.py uses ONE run_id for the inbox prefix it reads,
    the outbox prefix it writes, AND the evidence object path -- all
    three, the same value. ``enclave/rings.py`` derives each ring's id
    deterministically from its flagged members (by design, so the same
    graph always yields the same ring id), so writing an outbox bundle
    under the SAME run_id as a previous run hits
    ``write_outbox``'s ``if_generation_match=0`` idempotent-create guard
    with a ``412`` -- observed live: the enclave attested, got its KMS
    key, decrypted, and found the real ring, then crashed trying to
    write a bundle that already existed from an earlier capture. Giving
    every live-proof click its OWN run id (not the static
    ``inbox_run_id`` the sealed bundles were captured under) sidesteps
    that collision entirely, without touching the pre-existing
    workload/outbox-naming code.
    """
    return f"live-proof-{run_id}"


def _metadata_items(cfg: LiveProofConfig, run_id: str, image_digest: str) -> list[Any]:
    """Same tee-env-* keys as infra/enclave_run.sh, as individual Items
    (the compute_v1 client's native form -- not the single ^~^-joined
    string gcloud's --metadata flag parses, which does not apply here).

    ``image_digest`` is which image this VM actually launches --
    ``cfg.image_digest`` (approved) for a normal run, or
    ``TAMPERED_IMAGE_DIGEST`` for a tamper test. Everything else about
    the metadata (buckets, keys, bank ids, flagged accounts) is
    identical either way: the workload code path is the same, it is
    only KMS's own IAM policy that treats the two images differently.
    """
    import base64

    from google.cloud import compute_v1

    pairs = {
        "tee-image-reference": _image_ref(cfg, image_digest),
        "tee-restart-policy": "Never",
        "MULENET_ENCLAVE_PROJECT": cfg.project,
        "MULENET_ENCLAVE_INBOX_BUCKET": cfg.inbox_bucket,
        "MULENET_ENCLAVE_OUTBOX_BUCKET": cfg.outbox_bucket,
        "MULENET_ENCLAVE_EVIDENCE_BUCKET": cfg.evidence_bucket,
        "MULENET_ENCLAVE_ATTEMPT": "run",
        "MULENET_ENCLAVE_KMS_KEY": cfg.kms_edge_key,
        "MULENET_ENCLAVE_HASH_KMS_KEY": cfg.kms_hash_key,
        "MULENET_ENCLAVE_HASH_KEY_URI": cfg.hash_key_uri,
        "MULENET_ENCLAVE_WIP_AUDIENCE": cfg.wip_audience,
        "MULENET_ENCLAVE_WORKLOAD_SA": cfg.workload_sa_email,
        "MULENET_ENCLAVE_BANK_IDS": ",".join(cfg.bank_ids),
        "MULENET_ENCLAVE_RUN_ID": _enclave_run_id(run_id),
    }
    for bank_id in cfg.bank_ids:
        env_suffix = bank_id.upper().replace("-", "_")
        pairs[f"MULENET_ENCLAVE_OUTBOX_PUBKEY_{env_suffix}"] = base64.b64encode(
            cfg.outbox_public_keys[bank_id]
        ).decode("ascii")
        pairs[f"MULENET_ENCLAVE_FLAGGED_{env_suffix}"] = ",".join(
            cfg.flagged_account_ids.get(bank_id, ())
        )
    items = []
    for key, value in pairs.items():
        full_key = key if key.startswith("tee-") else f"tee-env-{key}"
        items.append(compute_v1.Items(key=full_key, value=value))
    return items


def _scheduling() -> Any:
    """Maintenance policy, plus an optional platform-side kill switch.

    ``MULENET_LIVE_PROOF_MAX_RUN_SECONDS`` (unset by default) asks Compute
    Engine itself to DELETE the VM after that many seconds, independent of any
    process of ours being alive or scheduled -- the one cleanup that survives a
    throttled container. It is opt-in because it has not been exercised against
    a Confidential VM yet; try it with one live run before relying on it.
    """
    import os

    from google.cloud import compute_v1

    raw = os.environ.get("MULENET_LIVE_PROOF_MAX_RUN_SECONDS", "")
    if raw.isdigit() and int(raw) > 0:
        return compute_v1.Scheduling(
            on_host_maintenance="TERMINATE",
            max_run_duration=compute_v1.Duration(seconds=int(raw)),
            instance_termination_action="DELETE",
        )
    return compute_v1.Scheduling(on_host_maintenance="TERMINATE")


def _build_instance(cfg: LiveProofConfig, vm_name: str, run_id: str, image_digest: str) -> Any:
    from google.cloud import compute_v1

    return compute_v1.Instance(
        name=vm_name,
        machine_type=f"zones/{cfg.zone}/machineTypes/{_MACHINE_TYPE}",
        confidential_instance_config=compute_v1.ConfidentialInstanceConfig(
            enable_confidential_compute=True,
            # The enum member (ConfidentialInstanceType.SEV) fails proto-plus
            # construction on the installed SDK version ("bad argument type
            # for built-in operation") -- the string name works and matches
            # infra/enclave_config.sh's own ENCLAVE_CONFIDENTIAL_TYPE="SEV".
            confidential_instance_type="SEV",
        ),
        shielded_instance_config=compute_v1.ShieldedInstanceConfig(enable_secure_boot=True),
        scheduling=_scheduling(),
        disks=[
            compute_v1.AttachedDisk(
                boot=True,
                auto_delete=True,
                initialize_params=compute_v1.AttachedDiskInitializeParams(
                    source_image=f"projects/{_IMAGE_PROJECT}/global/images/family/{_IMAGE_FAMILY}",
                ),
            )
        ],
        network_interfaces=[compute_v1.NetworkInterface(name="default")],
        service_accounts=[
            compute_v1.ServiceAccount(
                email=cfg.workload_sa_email,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        ],
        metadata=compute_v1.Metadata(items=_metadata_items(cfg, run_id, image_digest)),
        # ``purpose`` is the reaper's filter; ``mulenet-run`` lets it (and a
        # human in the console) tie a VM to the run that owns it.
        labels={"purpose": "live-proof", "mulenet-run": run_id},
    )


def run_live_proof(
    cfg: LiveProofConfig,
    run_id: str,
    credentials: Any,
    on_stage: Callable[[str, dict[str, Any]], None],
    tampered: bool = False,
) -> dict[str, Any]:
    """Run one full cycle. Calls ``on_stage(stage_name, extra)`` as each
    observable stage completes. Returns the final result dict.

    ``credentials`` must already be the impersonated mulenet-live-proof
    credential -- this function does no credential resolution itself, so
    it stays testable with a fake.

    ``tampered=True`` launches ``TAMPERED_IMAGE_DIGEST`` instead of
    ``cfg.image_digest``, and inverts what counts as success: KMS
    refusing the key (``outcome == "key_release_refused"``,
    ``refusal_kind == "kms_decrypt_refused"``) is the demo working
    correctly. Anything else -- including the key being released, which
    would mean the tampered digest somehow has a KMS grant -- is raised
    loudly, never treated as a quieter variant of success.
    """
    from google.cloud import compute_v1, storage

    from enclave.outbox_crypto import SealedOutbox, unseal_with_private_key

    vm_name = f"mulenet-live-proof-{run_id}"
    enclave_run_id = _enclave_run_id(run_id)
    effective_digest = TAMPERED_IMAGE_DIGEST if tampered else cfg.image_digest
    instances = compute_v1.InstancesClient(credentials=credentials)
    storage_client = storage.Client(project=cfg.project, credentials=credentials)
    #: Set only once insert() actually succeeds. The finally block below
    #: must not attempt to delete a VM that was never created -- doing so
    #: anyway produces a 404 that REPLACES the real failure (e.g. a
    #: missing IAM permission on the insert call itself) with a
    #: meaningless "instance not found", which is exactly the kind of
    #: masked error that cost real debugging time the first time this
    #: ran for real against Google Cloud.
    vm_created = False

    # Copy the real, already-sealed inbox ciphertext to a fresh object
    # name under enclave_run_id -- no decryption needed, these are just
    # bytes -- so the workload's read_inbox() finds the same real banks'
    # contributions it always has, while writing its outbox and evidence
    # under a run id THIS click owns alone. See _enclave_run_id's
    # docstring for why reusing inbox_run_id here hit a real collision.
    inbox_source_bucket = storage_client.bucket(cfg.inbox_bucket)
    for bank_id in cfg.bank_ids:
        source_blob = inbox_source_bucket.blob(f"inbox/{bank_id}/{cfg.inbox_run_id}.json")
        inbox_source_bucket.copy_blob(
            source_blob, inbox_source_bucket, f"inbox/{bank_id}/{enclave_run_id}.json"
        )

    on_stage(STAGE_REQUESTING_VM, {})
    instance = _build_instance(cfg, vm_name, run_id, effective_digest)

    try:
        op = instances.insert(project=cfg.project, zone=cfg.zone, instance_resource=instance)
        op.result(timeout=120)
        vm_created = True
        on_stage(STAGE_VM_RUNNING, {"vm_name": vm_name})

        on_stage(STAGE_AWAITING_ENCLAVE, {})
        evidence_name = f"runs/{enclave_run_id}/run.json"
        deadline = time.monotonic() + cfg.run_timeout_s
        evidence: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            blob = storage_client.bucket(cfg.evidence_bucket).blob(evidence_name)
            if blob.exists():
                evidence = json.loads(blob.download_as_text())
                break
            time.sleep(5)
        if evidence is None:
            raise TimeoutError(
                f"no evidence object after {cfg.run_timeout_s}s; the real run may "
                "still be in progress -- see the static evidence card for a captured result"
            )
        on_stage(STAGE_EVIDENCE_RECEIVED, {"evidence": evidence})

        if tampered:
            outcome = evidence.get("outcome")
            refusal_kind = evidence.get("refusal_kind")
            if outcome != "key_release_refused" or refusal_kind != "kms_decrypt_refused":
                # Loud, not hidden: the key being released (or any other
                # unexpected outcome) for an unapproved digest is a real
                # finding, not a demo hiccup to paper over.
                raise RuntimeError(
                    "tamper test did not refuse as expected -- "
                    f"outcome={outcome!r} refusal_kind={refusal_kind!r} "
                    "(if outcome is 'completed', the key was RELEASED to an "
                    "unapproved image)"
                )
            on_stage(
                STAGE_KMS_REFUSED,
                {
                    "refusal_kind": refusal_kind,
                    "refusal_message": evidence.get("refusal_message"),
                },
            )
            return {
                "ok": True,
                "tampered": True,
                "evidence": evidence,
                "approved_digest": cfg.image_digest,
                "tampered_digest": effective_digest,
                "tampered_line_changed": TAMPERED_LINE_CHANGED,
                "refusal_kind": refusal_kind,
                "refusal_message": evidence.get("refusal_message"),
            }

        if evidence.get("outcome") != "completed" or not evidence.get("key_release_ok"):
            raise RuntimeError(f"enclave run did not complete cleanly: outcome={evidence.get('outcome')!r}")

        bundles = [b for b in evidence.get("bundles_written", []) if f"/{cfg.correct_bank_id}/" in b]
        if not bundles:
            raise RuntimeError(f"no outbox bundle written for {cfg.correct_bank_id!r} this run")
        outbox_blob = storage_client.bucket(cfg.outbox_bucket).blob(bundles[0])
        sealed = SealedOutbox.from_json(outbox_blob.download_as_text())

        on_stage(STAGE_WRONG_KEY_ATTEMPT, {"bank_tried": cfg.wrong_bank_id})
        wrong_key_error: str | None = None
        try:
            unseal_with_private_key(sealed, cfg.wrong_private_key)
        except Exception as exc:  # noqa: BLE001 - the failure itself is the evidence
            wrong_key_error = type(exc).__name__
        on_stage(
            STAGE_WRONG_KEY_ATTEMPT,
            {
                "bank_tried": cfg.wrong_bank_id,
                "failed_as_expected": wrong_key_error is not None,
                "error": wrong_key_error,
            },
        )

        on_stage(STAGE_CORRECT_KEY_ATTEMPT, {"bank_tried": cfg.correct_bank_id})
        plaintext = unseal_with_private_key(sealed, cfg.correct_private_key)
        on_stage(STAGE_CORRECT_KEY_ATTEMPT, {"bank_tried": cfg.correct_bank_id, "succeeded": True})

        return {
            "ok": True,
            "evidence": evidence,
            "object_path": bundles[0],
            "ciphertext_bytes_total": outbox_blob.size,
            "wrong_key_error": wrong_key_error,
            "decrypted_preview": plaintext.decode()[:2000],
        }
    finally:
        on_stage(STAGE_CLEANING_UP, {})
        # Observed live: the main body failed fast, then the delete
        # ITSELF also timed out (deletion can genuinely take >120s) --
        # and the bare `raise` below replaced the real, fast-failing
        # cause with an unrelated "delete timed out", a second layer of
        # the same masking bug the vm_created guard above fixes for the
        # "VM never existed" case. An exception is already propagating
        # here whenever the main body raised, so that one wins; the
        # cleanup failure is recorded via on_stage but only re-raised
        # when cleanup is the SOLE failure (the main body returned
        # cleanly).
        already_failing = sys.exc_info()[0] is not None
        if vm_created:
            try:
                # Observed live, twice: a Confidential VM's delete
                # operation routinely takes 130-140s to resolve, over the
                # 120s this previously waited -- flagging a clean,
                # successful teardown as a "cleanup_error" on every run.
                instances.delete(project=cfg.project, zone=cfg.zone, instance=vm_name).result(timeout=240)
            except Exception as exc:  # noqa: BLE001 - deletion must never be skipped silently
                on_stage(STAGE_FAILED, {"cleanup_error": str(exc), "vm_name": vm_name})
                if not already_failing:
                    raise
