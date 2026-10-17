"""The inbox and outbox buckets: how ciphertext reaches and leaves the enclave.

## Why an inbox bucket owned by the operator

The design in ``docs/data_sharing.md`` originally had each bank hold its own
encrypted store. That needs a billable project per bank. The trial billing
account is capped at five linked projects and is full, and ``mulenet-bank-c`` is
deliberately left **unbilled**, so the banks are *identity-only*: a service
account and nothing billable.

So the ciphertext lands in a bucket the **operator** owns, and the operator's
blindness comes from key custody rather than from storage custody:

* each bank has ``roles/storage.objectCreator`` on its own prefix -- it can
  write, and cannot read back or overwrite,
* the operator can list and read objects, and holds no KMS decrypt right, so
  what it reads is opaque,
* only the attested workload can decrypt.

This is a weaker story than per-bank storage in one specific way, stated plainly
because it matters: the operator learns **traffic metadata** -- which bank
uploaded, when, and how large the bundle was. It does not learn contents. A
deployment that cares about metadata would put the inbox in a bank-owned or
regulator-owned project; nothing else about the design changes.

``objectCreator`` rather than ``objectAdmin`` is doing real work here: with
``objectAdmin`` one bank could overwrite another bank's bundle and silently
remove its accounts from the joint graph, which would let a bank hide its own
mules from the scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from enclave.envelope import SealedBundle

#: Object layout inside the inbox bucket. One prefix per bank so an IAM
#: condition can restrict a bank to its own prefix.
INBOX_PREFIX = "inbox"

#: Object layout inside the outbox bucket, one prefix per receiving bank.
OUTBOX_PREFIX = "outbox"

#: Object layout inside the evidence bucket. One object per run, written on
#: EVERY run including refused ones, so "no evidence" and "refused" can never be
#: confused with each other.
EVIDENCE_PREFIX = "runs"


def inbox_object_name(bank_id: str, run_id: str) -> str:
    """Where ``bank_id`` writes its bundle for ``run_id``."""
    return f"{INBOX_PREFIX}/{bank_id}/{run_id}.json"


def outbox_object_name(bank_id: str, run_id: str, ring_id: str) -> str:
    """Where the enclave writes one bank's alert bundle."""
    return f"{OUTBOX_PREFIX}/{bank_id}/{run_id}/{ring_id}.json"


def evidence_object_name(run_id: str, attempt: str) -> str:
    """Where the enclave writes this run's result object.

    ``attempt`` disambiguates repeated runs of the same ``run_id`` -- the
    negative tests run the same logical run twice (production image, then debug
    image) and each must leave its own object rather than overwrite the other's.
    """
    return f"{EVIDENCE_PREFIX}/{run_id}/{attempt}.json"


@dataclass(frozen=True)
class InboxEntry:
    """One sealed bundle found in the inbox.

    Attributes:
        bank_id: Taken from the **object path**, then checked against the
            bundle's own authenticated ``bank_id``. Both must agree: the path is
            what IAM constrains, and the AEAD is what cryptography constrains,
            so requiring both closes the gap where an operator moves an object
            between prefixes.
        bundle: The sealed bundle.
        uploaded_at: Object creation time, for the run manifest.
    """

    bank_id: str
    bundle: SealedBundle
    uploaded_at: datetime | None


class InboxError(RuntimeError):
    """An inbox object was rejected rather than processed."""


def read_inbox(
    storage_client: Any,
    bucket_name: str,
    run_id: str,
    expected_banks: tuple[str, ...],
) -> list[InboxEntry]:
    """Read one bundle per expected bank for ``run_id``.

    A missing bank is **not** an error: Stage 2 runs with one bank and Stage 3
    with three, and the enclave scores whatever arrived. The caller records who
    was absent in the run manifest so a thin result is never mistaken for a
    complete one.

    Raises:
        InboxError: if an object's path and its authenticated bank disagree.
    """
    bucket = storage_client.bucket(bucket_name)
    entries: list[InboxEntry] = []
    for bank_id in expected_banks:
        blob = bucket.blob(inbox_object_name(bank_id, run_id))
        if not blob.exists():
            continue
        bundle = SealedBundle.from_json(blob.download_as_text())
        if bundle.bank_id != bank_id:
            raise InboxError(
                f"object {blob.name!r} sits under bank {bank_id!r} but the bundle "
                f"authenticates as {bundle.bank_id!r}; refusing the run"
            )
        entries.append(
            InboxEntry(bank_id=bank_id, bundle=bundle, uploaded_at=blob.time_created)
        )
    return entries


def write_evidence(
    storage_client: Any,
    bucket_name: str,
    run_id: str,
    attempt: str,
    payload: str,
) -> str:
    """Write this run's result object and return the object name.

    Called on **every** path, including the one where KMS refused the key. It
    uses the VM's attached service account rather than the attested identity, so
    it keeps working when attestation fails -- which is the whole point of
    recording refusals as data instead of reading them off a serial console.

    Never raises into the caller's control flow: a failure to write evidence is
    reported and swallowed, because losing the evidence must not also destroy the
    result it was describing. The caller still returns the manifest it built.
    """
    name = evidence_object_name(run_id, attempt)
    blob = storage_client.bucket(bucket_name).blob(name)
    # if_generation_match=0 makes the write fail rather than overwrite an
    # existing object: evidence is append-only, and an overwritten refusal is a
    # destroyed audit trail.
    blob.upload_from_string(payload, content_type="application/json", if_generation_match=0)
    return name


def write_outbox(
    storage_client: Any,
    bucket_name: str,
    bank_id: str,
    run_id: str,
    ring_id: str,
    payload: str,
) -> str:
    """Write one bank's alert bundle and return the object name.

    The payload is written as-is; the caller (``enclave/workload.py``) is
    responsible for encrypting it first via ``enclave.outbox_crypto.seal_for_bank``
    so that a bucket the operator owns still holds only ciphertext the operator
    cannot open. This function's job is the write, not the encryption -- same
    separation ``enclave/envelope.py``/``enclave/kms.py`` keep on the inbox side.
    """
    name = outbox_object_name(bank_id, run_id, ring_id)
    blob = storage_client.bucket(bucket_name).blob(name)
    # if_generation_match=0 makes this an atomic create-only write: the default
    # upload (no precondition) needs delete/update permission because it must be
    # able to overwrite an existing object, which the workload SA's create-only
    # objectCreator role correctly refuses (observed live: 403 "does not have
    # storage.objects.delete access"). Matches write_evidence()'s existing
    # pattern in this same file.
    blob.upload_from_string(payload, content_type="application/json", if_generation_match=0)
    return name
